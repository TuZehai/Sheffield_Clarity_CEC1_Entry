import toml
import torch
import argparse

from .trainer import Trainer
from .network import ConvTasNet

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from datasets import TrainDataset, ValidationDataset
from losses import SISNRLoss, SNRLoss, STOILoss


def run(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ConvTasNet(**config['network_config'])
    model = torch.nn.parallel.DataParallel(model.to(device))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['optimizer']['lr'])

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
    else:
        raise NotImplementedError

    trainer = Trainer(config=config, model=model, optimizer=optimizer, loss_func=loss,
                      train_dataloader=train_dataloader,
                      validation_dataloader=validation_dataloader, device=device)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='denoising convtasnet on Clarity')
    parser.add_argument('-C', '--config', default='den_convtasnet.toml')
    args = parser.parse_args()

    config = toml.load(args.config)
    run(config)

