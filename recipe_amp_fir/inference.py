import argparse
import toml
import os
import json
import torch
import librosa
import logging
import numpy as np
from tqdm import tqdm
from torch.utils import data
from soundfile import read, write
from scipy.signal import firwin, lfilter, unit_impulse, find_peaks
from scipy.fftpack import fft
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from .network import ConvTasNet
from .processor import AudiometricFIR
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..')))
from MSBG.ear import Ear
from MSBG.audiogram import Audiogram
from MBSTOI.mbstoi import mbstoi
from MBSTOI.dbstoi import dbstoi


def read_wavfile(path):
    wav, _ = read(path)
    return wav.transpose()


class InferenceDataset(data.Dataset):
    def __init__(self, scene_list, data_path, sr, downsample_factor, num_channels, norm):
        self.scene_list = scene_list
        self.data_path = data_path
        self.sr = sr
        self.downsample_factor = downsample_factor
        self.num_channels = num_channels
        self.norm = norm

        if self.num_channels == 2:
            self.mixed_suffix = '_mixed_CH1.wav'
            self.target_suffix = '_target_anechoic.wav'
        elif self.num_channels == 6:
            self.mixed_suffix = ['_mixed_CH1.wav', '_mixed_CH2.wav', '_mixed_CH3.wav']
            self.target_suffix = '_target_anechoic.wav'
        else:
            raise NotImplementedError

        self.lowpass_filter = firwin(1025, self.sr // (2 * self.downsample_factor), pass_zero='lowpass', fs=self.sr)

    def lowpass_filtering(self, x):
        return lfilter(self.lowpass_filter, 1, x)

    def __getitem__(self, item):
        if self.num_channels == 2:
            mixed = read_wavfile(os.path.join(self.data_path, self.scene_list[item] + self.mixed_suffix))
        elif self.num_channels == 6:
            mixed = []
            for suffix in self.mixed_suffix:
                mixed.append(read_wavfile(os.path.join(self.data_path, self.scene_list[item] + suffix)))
            mixed = np.concatenate(mixed, axis=0)
        else:
            raise NotImplementedError
        target = read_wavfile(os.path.join(self.data_path, self.scene_list[item] + self.target_suffix))

        if self.sr != 44100:
            mixed_resampled, target_resampled = [], []
            for i in range(mixed.shape[0]):
                mixed_resampled.append(librosa.resample(mixed[i], 44100, self.sr))
            for i in range(target.shape[0]):
                target_resampled.append(librosa.resample(target[i], 44100, self.sr))
            mixed = np.array(mixed_resampled)
            target = np.array(target_resampled)

        # if self.downsample_factor != 1:
        #     mixed_lowpass, target_lowpass = [], []
        #     for i in range(mixed.shape[0]):
        #         mixed_lowpass.append(self.lowpass_filtering(mixed[i]))
        #     for i in range(target.shape[0]):
        #         target_lowpass.append(self.lowpass_filtering(target[i]))
        #     mixed = np.array(mixed_lowpass)
        #     target = np.array(target_lowpass)

        x = mixed.copy()
        y = target.copy()

        if self.norm:
            x_max = np.max(np.abs(x))
            x = x / x_max
            y = y / x_max

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), self.scene_list[item]

    def __len__(self):
        return len(self.scene_list)


class Inferencer:
    def __init__(self, config):
        """ Ear """
        listener_idx = config['listener']['listener_idx']
        self.listener_idx = listener_idx
        self.sr = config['listener']['listener_sr']

        with open(config['listener']['listeners_path'], 'r') as f:
            listeners_file = json.load(f)
            audiogram_cfs = listeners_file[config['listener']['listener_idx']]['audiogram_cfs']
            audiogram_lvl_l = listeners_file[config['listener']['listener_idx']]['audiogram_levels_l']
            audiogram_lvl_r = listeners_file[config['listener']['listener_idx']]['audiogram_levels_r']
            f.close()

        src_pos = config['listener']['src_pos']
        left_audiogram = Audiogram(cfs=np.array(audiogram_cfs), levels=np.array(audiogram_lvl_l))
        right_audiogram = Audiogram(cfs=np.array(audiogram_cfs), levels=np.array(audiogram_lvl_r))

        audiograms = [left_audiogram, right_audiogram]
        self.ears = [Ear(audiogram=audiogram, src_pos=src_pos) for audiogram in audiograms]

        flat0dB_audiogram = Audiogram(cfs=np.array(audiogram_cfs), levels=np.zeros((np.shape(np.array(audiogram_cfs)))))
        self.flat0dB_ear = Ear(audiogram=flat0dB_audiogram, src_pos="ff")

        """ Dataloader """
        # Scenes for dev
        scene_list = []
        with open(config['listener']['scenes_listeners'], 'r') as f:
            scenes_listeners_file = json.load(f)
            for scene, listeners in scenes_listeners_file.items():
                if listener_idx in listeners:
                    scene_list.append(scene)
            f.close()
        self.scene_list = scene_list

        # Dataloader
        self.data_path = config['inference_dataset']['data_path']
        inference_dataset = InferenceDataset(scene_list=scene_list, **config['inference_dataset'])
        self.inference_dataloader = torch.utils.data.DataLoader(dataset=inference_dataset, **config['inference_dataloader'])

        """ Models """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'

        self.model_den_l_config = config['denoising_network_config']
        self.model_den_r_config = config['denoising_network_config']
        self.fir_l_config = config['processor_config']
        self.fir_r_config = config['processor_config']

        self.model_den_l_checkpoint_path = config['network_checkpoints']['denoising_left_checkpoint']
        self.model_den_r_checkpoint_path = config['network_checkpoints']['denoising_right_checkpoint']
        self.model_amp_l_checkpoint_path = config['network_checkpoints']['amplification_left_checkpoint']
        self.model_amp_r_checkpoint_path = config['network_checkpoints']['amplification_right_checkpoint']

        """ Inference setup """
        self.do_eval = config['setup']['do_eval']
        self.save_dir = config['setup']['save_dir']
        self.downsample_factor = config['inference_dataset']['downsample_factor']
        os.makedirs(self.save_dir, exist_ok=True)

        """ torchaudio resample"""
        import torchaudio
        self.downsample = torchaudio.transforms.Resample(orig_freq=self.sr, new_freq=self.sr // self.downsample_factor,
                                                           resampling_method='sinc_interpolation')
        self.upsample = torchaudio.transforms.Resample(orig_freq=self.sr // self.downsample_factor, new_freq=self.sr,
                                                           resampling_method='sinc_interpolation')

    def load_den_model(self, network_config, checkpoint_path):
        model = ConvTasNet(**network_config)
        model = torch.nn.parallel.DataParallel(model)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        return model

    def load_fir_model(self, fir_config, checkpoint_path):
        model = AudiometricFIR(nfir=fir_config['nfir'], sr=fir_config['sr'] // fir_config['downsample_factor'])
        model = torch.nn.parallel.DataParallel(model)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        return model

    def infer_left(self):
        model_den_l = self.load_den_model(self.model_den_l_config, self.model_den_l_checkpoint_path)
        model_amp_l = self.load_fir_model(self.fir_l_config, self.model_amp_l_checkpoint_path)

        for step, (noisy, clean, scene) in tqdm(enumerate(self.inference_dataloader), desc='inferencing'):
            noisy = noisy.to(self.device)
            downsampled_noisy = self.downsample(noisy)
            enhanced_l = model_amp_l(model_den_l(downsampled_noisy)).squeeze(1)
            upsampled_enhanced_l = self.upsample(enhanced_l)
            upsampled_enhanced_l = torch.clamp(upsampled_enhanced_l, -1, 1)
            out_l = upsampled_enhanced_l.detach().cpu().numpy()[0]

            write(os.path.join(self.save_dir, 'left_' + scene[0] + '_' + self.listener_idx + '_' + 'HA-output.wav'), out_l, self.sr)

    def infer_right(self):
        model_den_r = self.load_den_model(self.model_den_r_config, self.model_den_r_checkpoint_path)
        model_amp_r = self.load_fir_model(self.fir_r_config, self.model_amp_r_checkpoint_path)

        for step, (noisy, clean, scene) in tqdm(enumerate(self.inference_dataloader), desc='inferencing'):
            noisy = noisy.to(self.device)
            downsampled_noisy = self.downsample(noisy)
            enhanced_r = model_amp_r(model_den_r(downsampled_noisy)).squeeze(1)
            upsampled_enhanced_r = self.upsample(enhanced_r)
            upsampled_enhanced_r = torch.clamp(upsampled_enhanced_r, -1, 1)
            out_r = upsampled_enhanced_r.detach().cpu().numpy()[0]

            write(os.path.join(self.save_dir, 'right_' + scene[0] + '_' + self.listener_idx + '_' + 'HA-output.wav'),
                  out_r, self.sr)

    def infer_binuaral(self):
        torch.cuda.empty_cache()
        self.infer_left()
        torch.cuda.empty_cache()
        self.infer_right()

        for step, (noisy, clean, scene) in tqdm(enumerate(self.inference_dataloader), desc='inferencing'):
            wav_left = read(os.path.join(self.save_dir, 'left_' + scene[0] + '_' + self.listener_idx + '_' + 'HA-output.wav'))[0]
            wav_right = read(os.path.join(self.save_dir, 'right_' + scene[0] + '_' + self.listener_idx + '_' + 'HA-output.wav'))[0]
            out = np.stack([wav_left, wav_right], axis=0).transpose()
            write(os.path.join(self.save_dir, scene[0] + '_' + self.listener_idx + '_' + 'HA-output.wav'), out, self.sr)

    def pad(sefl, signal, length):
        """Zero pad signal to required length.
        Assumes required length is not less than input length.
        """
        assert length >= signal.shape[0]
        return np.pad(
            signal, [(0, length - signal.shape[0])] + [(0, 0)] * (len(signal.shape) - 1)
        )

    def listen(self, signal, ears):
        outputs = [
            ear.process(
                signal[:, i],
                add_calibration=False,
            )
            for i, ear in enumerate(ears)
        ]

        # Fix length difference if no smearing on one of two ears
        if len(outputs[0][0]) != len(outputs[1][0]):
            diff = len(outputs[0][0]) - len(outputs[1][0])
            if diff > 0:
                outputs[1][0] = np.flipud(self.pad(np.flipud(outputs[1][0]), len(outputs[0][0])))
            else:
                outputs[0][0] = np.flipud(self.pad(np.flipud(outputs[0][0]), len(outputs[1][0])))

        return np.squeeze(outputs).T

    def HL_simulate(self):
        for scene in tqdm(self.scene_list, desc='eval'):
            signal = read(os.path.join(self.save_dir, scene + '_' + self.listener_idx + '_' + 'HA-output.wav'))[0]
            output = self.listen(signal, self.ears)
            write(os.path.join(self.save_dir, scene + '_' + self.listener_idx + '_' + 'HL-output.wav'), output, self.sr)

            ddf_signal = np.zeros((np.shape(signal)))
            ddf_signal[:, 0] = unit_impulse(len(signal), int(self.sr / 2))
            ddf_signal[:, 1] = unit_impulse(len(signal), int(self.sr / 2))
            ddf_outputs = self.listen(ddf_signal, self.ears)
            write(os.path.join(self.save_dir, scene + '_' + self.listener_idx + '_' + 'HLddf-output.wav'), ddf_outputs, self.sr)

    def find_delay_impulse(self, ddf, initial_value=22050):
        """Find binaural delay in signal ddf given initial location of unit impulse, initial_value."""
        pk0 = find_peaks(ddf[:, 0])
        pk1 = find_peaks(ddf[:, 1])
        delay = np.zeros((2, 1))
        if len(pk0[0]) > 0:
            # m = np.max(ddf[pk0[0], 0])
            pkmax0 = np.argmax(ddf[:, 0])
            delay[0] = int(pkmax0 - initial_value)
        else:
            logging.error("Error in selecting peaks.")
        if len(pk1[0]) > 0:
            pkmax1 = np.argmax(ddf[:, 1])
            delay[1] = int(pkmax1 - initial_value)
        else:
            logging.error("Error in selecting peaks.")
        return delay

    def cal_SI(self):
        from pystoi import stoi
        all_sii = []
        for scene in tqdm(self.scene_list, desc='eval'):
            proc = read(os.path.join(self.save_dir, scene + '_' + self.listener_idx + '_' + 'HL-output.wav'))[0]
            clean = read(os.path.join(self.data_path, scene + '_target_anechoic.wav'))[0]
            ddf = read(os.path.join(self.save_dir, scene + '_' + self.listener_idx + '_' + 'HLddf-output.wav'))[0]

            delay = self.find_delay_impulse(ddf, initial_value=int(self.sr / 2))
            if delay[0] != delay[1]:
                logging.info(f"Difference in delay of {delay[0] - delay[1]}.")
            maxdelay = int(np.max(delay))

            # Allow for value lower than 1000 samples in case of unimpaired hearing
            if maxdelay > 2000:
                logging.error(f"Error in delay calculation for signal time-alignment.")

            cleanpad = np.zeros((len(clean) + maxdelay, 2))
            procpad = np.zeros((len(clean) + maxdelay, 2))

            if len(procpad) < len(proc):
                raise ValueError(f"Padded processed signal is too short.")

            cleanpad[int(delay[0]): int(len(clean) + int(delay[0])), 0] = clean[:, 0]
            cleanpad[int(delay[1]): int(len(clean) + int(delay[1])), 1] = clean[:, 1]
            procpad[: len(proc)] = proc
            # sii = mbstoi(
            #     cleanpad[:, 0],
            #     cleanpad[:, 1],
            #     procpad[:, 0],
            #     procpad[:, 1],
            #     gridcoarseness=1,
            # )
            sii = dbstoi(
                cleanpad[:, 0],
                cleanpad[:, 1],
                procpad[:, 0],
                procpad[:, 1],
                gridcoarseness=1,
            )
            # sii = stoi(cleanpad[:, 1], procpad[:, 1], self.sr, extended=False)
            print(sii)
            logging.info(f"{sii:3.4f}")
            all_sii.append(sii)
        print(np.array(all_sii).mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference config')
    parser.add_argument('-C', '--config', default='inference.toml')
    args = parser.parse_args()
    config = toml.load(args.config)

    inferencer = Inferencer(config)
    inferencer.infer_binuaral()

