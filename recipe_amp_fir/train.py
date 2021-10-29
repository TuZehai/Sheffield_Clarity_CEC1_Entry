import torch
import toml
import argparse

from .trainer import Trainer
from .network import ConvTasNet
from .processor import AudiometricFIR

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from datasets import TrainDataset, ValidationDataset
from losses import SISNRLoss, SNRLoss, STOILoss, STOILevelLoss


def run(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_denoising = ConvTasNet(**config['denoising_network_config'])
    model_denoising = torch.nn.parallel.DataParallel(model_denoising)
    checkpoint = torch.load(config['denoising_network_checkpoint']['checkpoint'], map_location=device)
    model_denoising.load_state_dict(checkpoint['model'])

    model_amp = AudiometricFIR(nfir=config['processor_config']['nfir'],
                               sr=config['processor_config']['sr'] // config['processor_config']['downsample_factor'])
    model_amp = torch.nn.parallel.DataParallel(model_amp.to(device))

    if config['optimizer']['optimize_denoising_network']:
        optimizer = torch.optim.Adam(params=list(model_amp.parameters()) + list(model_denoising.parameters()),
                                     lr=config['optimizer']['lr'])
    else:
        optimizer = torch.optim.Adam(params=model_amp.parameters(), lr=config['optimizer']['lr'])

    train_dataset = TrainDataset(**config['train_dataset'])
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, **config['train_dataloader'])

    validation_dataset = ValidationDataset(**config['validation_dataset'])
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, **config['validation_dataloader'])

    if config['loss']['loss_func'] == 'si-snr':
        loss = SISNRLoss()
    elif config['loss']['loss_func'] == 'mse':
        loss = torch.nn.MSELoss()
    elif config['loss']['loss_func'] == 'snr':
        loss = SNRLoss()
    elif config['loss']['loss_func'] == 'stoi':
        loss = STOILoss(sr=config['listener']['listener_sr'])
        loss = torch.nn.parallel.DataParallel(loss.to(device))
    elif config['loss']['loss_func'] == 'stoilevel':
        loss = STOILevelLoss(sr=config['listener']['listener_sr'], alpha=config['loss']['alpha'])
        loss = torch.nn.parallel.DataParallel(loss.to(device))
    else:
        raise NotImplementedError

    trainer = Trainer(config=config, model_denoising=model_denoising, model_amp=model_amp, optimizer=optimizer,
                      loss_func=loss, train_dataloader=train_dataloader, validation_dataloader=validation_dataloader,
                      device=device)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='amplification fir on Clarity')
    parser.add_argument('-C', '--config', default='amp_fir.toml')
    args = parser.parse_args()

    config = toml.load(args.config)
    run(config)
