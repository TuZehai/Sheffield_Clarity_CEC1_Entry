import os
import torch
import toml
from datetime import datetime
from tqdm import tqdm
from glob import glob
import soundfile as sf
import numpy as np
from tensorboardX import SummaryWriter
from pystoi import stoi


class Trainer:
    def __init__(self, config, model, optimizer, loss_func, train_dataloader, validation_dataloader, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.device = device

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
        if config['listener']['listener_ear'] == 'l':
            self.ear_idx = 0
        elif config['listener']['listener_ear'] == 'r':
            self.ear_idx = 1
        else:
            raise ValueError("No THIRD EAR MY FRIEND")
        self.sr = config['listener']['listener_sr']

        # downsample for convtasnet training
        self.downsample_factor = config['listener']['downsample_factor']
        if self.downsample_factor != 1:
            self.sr = int(self.sr / self.downsample_factor)
            """ torchaudio resample"""
            import torchaudio
            self.downsample = torchaudio.transforms.Resample(orig_freq=self.sr,
                                                             new_freq=self.sr // self.downsample_factor,
                                                             resampling_method='sinc_interpolation')
            self.upsample = torchaudio.transforms.Resample(orig_freq=self.sr // self.downsample_factor,
                                                           new_freq=self.sr,
                                                           resampling_method='sinc_interpolation')

    def _set_train_mode(self):
        self.model.train()

    def _set_eval_mode(self):
        self.model.eval()

    def _save_checkpoint(self, epoch, score):
        state_dict = {
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.state_dict()
        }
        torch.save(state_dict, os.path.join(self.checkpoint_path, f'model_{str(epoch).zfill(4)}.tar'))

        if score > self.best_score:
            torch.save(state_dict, os.path.join(self.checkpoint_path, 'model_best.tar'))
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

            if self.downsample_factor != 1:
                noisy = self.downsample(noisy)
                clean = self.downsample(clean)
                # noisy = torch.nn.functional.interpolate(noisy, scale_factor=1 / self.downsample_factor,
                #                                                     mode='linear')
                # clean = torch.nn.functional.interpolate(clean.unsqueeze(1), scale_factor=1 / self.downsample_factor,
                #                                                     mode='linear').squeeze(1)

            enhanced = self.model(noisy).squeeze(1)
            loss = self.loss_func(enhanced, clean)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
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

            if self.downsample_factor != 1:
                noisy = self.downsample(noisy)
                clean = self.downsample(clean)
            enhanced = self.model(noisy).squeeze(1)

            loss = self.loss_func(enhanced, clean)

            enhanced = enhanced.detach().squeeze(0).cpu().numpy()
            noisy = noisy.detach().squeeze(0).cpu().numpy()
            clean = clean.detach().squeeze(0).cpu().numpy()

            print(clean.shape, enhanced.shape, self.sr)
            stoi_scores.append(stoi(clean, enhanced, self.sr))
            total_loss += loss

            if step < 3:
                sf.write(os.path.join(self.sample_path, 'enhanced_epoch{}_sample{}.wav'.format(epoch, step)), enhanced, self.sr)
                if epoch == self.save_checkpoint_interval:
                    sf.write(os.path.join(self.sample_path, 'noisy_sample{}.wav'.format(step)), noisy[0], self.sr)
                    sf.write(os.path.join(self.sample_path, 'clean_sample{}.wav'.format(step)), clean, self.sr)

        ave_stoi = np.array(stoi_scores).mean()
        self.writer.add_scalars('val_loss', {'val_loss': total_loss / len(self.validation_dataloader),
                                             'stoi_ave': ave_stoi}, epoch)
        return ave_stoi

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._set_train_mode()
            self._train_epoch()
            if epoch % self.save_checkpoint_interval == 0:
                self._set_eval_mode()
                score = self._validation_epoch(epoch)
                self._save_checkpoint(epoch, score)


