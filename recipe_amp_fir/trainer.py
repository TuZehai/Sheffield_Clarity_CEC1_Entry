import os
import sys
import torch
import json
import toml
from datetime import datetime
from tqdm import tqdm
from glob import glob
import soundfile as sf
import numpy as np
from tensorboardX import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from hearinglossmodel import MSBGHearingModel, torchloudnorm

from MSBG.ear import Ear
from MSBG.audiogram import Audiogram
from pystoi import stoi


class Trainer:
    def __init__(self, config, model_denoising, model_amp, optimizer, loss_func, train_dataloader, validation_dataloader, device):
        self.model_denoising = model_denoising
        self.model_amp = model_amp
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.device = device
        self.optimize_denoising_network = config['optimizer']['optimize_denoising_network']

        # training config
        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']
        self.save_checkpoint_interval = self.trainer_config['save_checkpoint_interval']
        self.clip_grad_norm_value = self.trainer_config['clip_grad_norm_value']
        self.exp_path = self.trainer_config['exp_path'] + '_' + str(datetime.now()).split('.')[0]
        self.resume = self.trainer_config['resume']

        self.log_path = os.path.join(self.exp_path, 'logs')
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.sample_path = os.path.join(self.exp_path, 'val_samples')
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)

        # save the config
        with open(os.path.join(self.exp_path, 'config.toml'), 'w') as f:
            toml.dump(config, f)
            f.close()

        # training visualisation
        self.writer = SummaryWriter(self.log_path)
        self.global_step = 0
        self.start_epoch = 1
        self.best_score = 0

        # TODO: resume training
        if self.resume:
            self._resume_checkpoint()

        # hearing loss model
        with open(config['listener']['listeners_path'], 'r') as f:
            listeners_file = json.load(f)
            audiogram_cfs = listeners_file[config['listener']['listener_idx']]['audiogram_cfs']
            audiogram_lvl_l = listeners_file[config['listener']['listener_idx']]['audiogram_levels_l']
            audiogram_lvl_r = listeners_file[config['listener']['listener_idx']]['audiogram_levels_r']
            f.close()
        if config['listener']['listener_ear'] == 'l':
            self.ear_idx = 0
            audiogram = audiogram_lvl_l
        elif config['listener']['listener_ear'] == 'r':
            self.ear_idx = 1
            audiogram = audiogram_lvl_r
        else:
            raise ValueError("No THIRD EAR MY FRIEND")
        self.sr = config['listener']['listener_sr']
        self.hearinglossmodel = MSBGHearingModel(audiogram=audiogram, audiometric=audiogram_cfs, sr=self.sr, spl_cali=True)
        self.normalhearingmodel = MSBGHearingModel(audiogram=np.zeros_like(audiogram), audiometric=audiogram_cfs, sr=self.sr, spl_cali=True)

        # msbg model
        src_pos = config['listener']['src_pos']
        MSBGaudiogram = Audiogram(cfs=np.array(audiogram_cfs), levels=np.array(audiogram))
        self.msbg_ear = Ear(src_pos, MSBGaudiogram)
        normalMSBGaudiogram = Audiogram(cfs=np.array(audiogram_cfs), levels=np.zeros_like(audiogram))
        self.normal_msbg_ear = Ear(src_pos, normalMSBGaudiogram)

        # downsample for convtasnet training
        self.downsample_factor = config['listener']['downsample_factor']
        """ torchaudio resample"""
        import torchaudio
        self.downsample = torchaudio.transforms.Resample(orig_freq=self.sr, new_freq=self.sr // self.downsample_factor,
                                                           resampling_method='sinc_interpolation')
        self.upsample = torchaudio.transforms.Resample(orig_freq=self.sr // self.downsample_factor, new_freq=self.sr,
                                                           resampling_method='sinc_interpolation')

        # loudness norm
        self.loudnorm = config['listener']['loudnorm']
        self.ln = torchloudnorm()

    def _set_train_mode(self, model):
        model.train()

    def _set_eval_mode(self, model):
        model.eval()

    def _save_checkpoint(self, model, model_name, epoch, score):
        state_dict = {
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'model': model.state_dict()
        }
        torch.save(state_dict, os.path.join(self.checkpoint_path, model_name + f'_{str(epoch).zfill(4)}.tar'))

        if score >= self.best_score:
            torch.save(state_dict, os.path.join(self.checkpoint_path, model_name + '_best.tar'))
            self.best_score = score.copy()

    def _resume_checkpoint(self):
        latest_checkpoints = sorted(glob(os.path.join(self.checkpoint_path, 'model_*.tar')))[-1]

        map_location = self.device
        checkpoint = torch.load(latest_checkpoints, map_location=map_location)
        self.start_epoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.model.load_state_dict(checkpoint['model'])

    def _train_epoch(self):
        for noisy, clean in tqdm(self.train_dataloader, desc='training'):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            clean = clean[:, self.ear_idx, :]

            downsampled_noisy = self.downsample(noisy)
            enhanced = self.model_amp(self.model_denoising(downsampled_noisy)).squeeze(1)
            upsampled_enhanced = self.upsample(enhanced)

            downsampled_clean = self.downsample(clean)
            upsampled_clean = self.upsample(downsampled_clean)

            if self.loudnorm:
                # normalize the loudness
                upsampled_enhanced = self.ln(upsampled_enhanced)
                upsampled_clean = self.ln(upsampled_clean)

            # hard clipping enhanced signal
            upsampled_enhanced = torch.clamp(upsampled_enhanced, -1, 1)

            # normal hearing simualation
            sim_clean = self.normalhearingmodel(upsampled_clean)
            # hearing loss simulation
            sim_enhanced = self.hearinglossmodel(upsampled_enhanced)

            loss = self.loss_func(sim_enhanced, sim_clean)
            self.optimizer.zero_grad()
            loss.backward()
            if self.optimize_denoising_network:
                torch.nn.utils.clip_grad_norm_(list(self.model_amp.parameters()) +
                                               list(self.model_denoising.parameters()), self.clip_grad_norm_value)
            else:
                torch.nn.utils.clip_grad_norm_(self.model_amp.parameters(), self.clip_grad_norm_value)
            self.optimizer.step()

            self.writer.add_scalars('loss', {'loss': loss}, self.global_step)
            self.global_step += 1

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        total_loss = 0
        stoi_scores = []
        for step, (noisy, clean) in tqdm(enumerate(self.validation_dataloader), desc='validating'):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            clean = clean[:, self.ear_idx, :]

            downsampled_noisy = self.downsample(noisy)
            enhanced = self.model_amp(self.model_denoising(downsampled_noisy)).squeeze(1)
            upsampled_enhanced = self.upsample(enhanced)

            downsampled_clean = self.downsample(clean)
            upsampled_clean = self.upsample(downsampled_clean)

            raw_clean = upsampled_clean.detach().squeeze(0).cpu().numpy()
            raw_enhanced = upsampled_enhanced.detach().squeeze(0).cpu().numpy()

            if self.loudnorm:
                # normalize the loudness
                upsampled_enhanced = self.ln(upsampled_enhanced)
                upsampled_clean = self.ln(upsampled_clean)

            # hard clipping enhanced signal
            upsampled_enhanced = torch.clamp(upsampled_enhanced, -1, 1)

            # normal hearing simualation
            sim_clean = self.normalhearingmodel(upsampled_clean)
            # hearing loss simulation
            sim_enhanced = self.hearinglossmodel(upsampled_enhanced)

            loss = self.loss_func(sim_enhanced, sim_clean)

            sim_clean = sim_clean.detach().squeeze(0).cpu().numpy()
            sim_enhanced = sim_enhanced.detach().squeeze(0).cpu().numpy()

            stoi_scores.append(stoi(sim_clean, sim_enhanced, self.sr))
            total_loss += loss

            if step < 3:
                sf.write(os.path.join(self.sample_path, 'enhanced_epoch{}_sample{}.wav'.format(epoch, step)),
                         sim_enhanced, self.sr)
                sf.write(os.path.join(self.sample_path, 'raw_epoch{}_sample{}.wav'.format(epoch, step)),
                         raw_enhanced, self.sr)
                msbg_enhanced = self.msbg_ear.process(np.clip(raw_enhanced, -1, 1))[0]
                sf.write(os.path.join(self.sample_path, 'msbg_enhanced_epoch{}_sample{}.wav'.format(epoch, step)),
                         msbg_enhanced, self.sr)

                if epoch == self.save_checkpoint_interval:
                    sf.write(os.path.join(self.sample_path, 'clean_sample{}.wav'.format(step)),
                             sim_clean, self.sr)
                    msbg_clean = self.normal_msbg_ear.process(raw_clean)[0]
                    sf.write(os.path.join(self.sample_path, 'msbg_clean_sample{}.wav'.format(step)),
                             msbg_clean, self.sr)

        ave_stoi = np.array(stoi_scores).mean()
        self.writer.add_scalars('val_loss', {'val_loss': total_loss / len(self.validation_dataloader),
                                             'stoi_ave': ave_stoi}, epoch)
        return ave_stoi

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._set_train_mode(self.model_amp)
            if self.optimize_denoising_network:
                self._set_train_mode(self.model_denoising)
            else:
                self._set_eval_mode(self.model_denoising)
            self._train_epoch()

            if epoch % self.save_checkpoint_interval == 0:
                if self.optimize_denoising_network:
                    self._set_eval_mode(self.model_denoising)
                self._set_eval_mode(self.model_amp)

                score = self._validation_epoch(epoch)

                self._save_checkpoint(self.model_amp, 'model_amp', epoch, score)
                if self.optimize_denoising_network:
                    self._save_checkpoint(self.model_denoising, 'model_denoising', epoch, score)
