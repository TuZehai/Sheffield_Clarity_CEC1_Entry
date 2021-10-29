import os
import json
import torch
import librosa

import numpy as np
from torch.utils import data
from soundfile import read, write
from scipy.signal import firwin, lfilter


def read_wavfile(path):
    wav, _ = read(path)
    return wav.transpose()


class TrainDataset(data.Dataset):
    def __init__(self, data_path, scene_path, sr, downsample_factor, wav_sample_len, wav_silence_len, num_channels, norm):
        self.data_path = data_path
        self.sr = sr
        self.downsample_factor = downsample_factor
        self.wav_sample_len = wav_sample_len
        self.wav_silence_len = wav_silence_len
        self.num_channels = num_channels
        self.norm = norm

        self.scene_list = []
        with open(scene_path, 'r') as f:
            scene_json = json.load(f)
            for i in range(len(scene_json)):
                self.scene_list.append(scene_json[i]['scene'])
            f.close()

        if self.num_channels == 2:
            self.mixed_suffix = '_mixed_CH1.wav'
            self.target_suffix = '_target_anechoic.wav'
        elif self.num_channels == 6:
            self.mixed_suffix = ['_mixed_CH1.wav', '_mixed_CH2.wav', '_mixed_CH3.wav']
            self.target_suffix = '_target_anechoic.wav'
        else:
            raise NotImplementedError

        self.lowpass_filter = firwin(1025, self.sr // (2 * self.downsample_factor), pass_zero='lowpass', fs=self.sr)

    def wav_sample(self, x, y):
        """
        A 2 second silence is in the beginning of clarity data
        Get rid of the silence segment in the beginning & sample a constant wav length for training.
        """
        silence_len = int(self.wav_silence_len * self.sr)
        x = x[:, silence_len:]
        y = y[:, silence_len:]

        wav_len = x.shape[1]
        sample_len = self.wav_sample_len * self.sr
        if wav_len > sample_len:
            start = np.random.randint(wav_len - sample_len)
            end = start + sample_len
            x = x[:, start:end]
            y = y[:, start:end]
            return x, y
        elif wav_len < sample_len:
            x = np.append(x, np.zeros([x.shape[1], sample_len - wav_len], dtype=np.float32))
            y = np.append(y, np.zeros([x.shape[1], sample_len - wav_len], dtype=np.float32))
            return x, y
        else:
            return x, y

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

        x, y = self.wav_sample(mixed, target)

        if self.norm:
            x_max = np.max(np.abs(x))
            x = x / x_max
            y = y / x_max

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.scene_list)


class ValidationDataset(data.Dataset):
    def __init__(self, data_path, scene_path, sr, downsample_factor, wav_sample_len, wav_silence_len, num_channels, norm):
        self.data_path = data_path
        self.sr = sr
        self.downsample_factor = downsample_factor
        self.wav_sample_len = wav_sample_len
        self.wav_silence_len = wav_silence_len
        self.num_channels = num_channels
        self.norm = norm

        self.scene_list = []
        with open(scene_path, 'r') as f:
            scene_json = json.load(f)
            for i in range(len(scene_json)):
                self.scene_list.append(scene_json[i]['scene'])
            f.close()

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

        x = mixed.copy()
        y = target.copy()

        if self.norm:
            x_max = np.max(np.abs(x))
            x = x / x_max
            y = y / x_max

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.scene_list)


