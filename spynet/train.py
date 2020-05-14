import click
from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

import spynet
import spynet.transforms as OFT
from spynet import config
from spynet import dataset
from spynet.model import SpyNetUnit, SpyNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(dl: DataLoader,
                    optimizer: torch.optim.AdamW,
                    criterion_fn: nn.Module,
                    Gk: torch.nn.Module, prev_pyramid: torch.nn.Module = None, 
                    print_freq: int = 100,
                    epoch: int = 0):

    Gk.train()
    running_loss = 0.

    if prev_pyramid is not None:
        prev_pyramid.eval()

    for i, (x, y) in enumerate(dl):
        x = x[0].to(device), x[1].to(device)
        y = y.to(device)

        if prev_pyramid is not None:
            with torch.no_grad():
                Vk_1 = prev_pyramid(x)
        else:
            Vk_1 = None

        predictions = Gk(x, Vk_1)

        loss = criterion_fn(y, predictions)
        loss.backward()

        running_loss += loss.item()

        if (i + 1) % print_freq == 0:
            loss_mean = running_loss / i
            print(f'Epoch {epoch}] [{i}/{len(dl)}] loss {loss_mean:.4f}')


def load_pyramid(k: int, checkpoint_path: Union[str, Path]) -> torch.nn.Module:
    checkpoint_path = Path(checkpoint_path)
    units = []

    for i in range(k):
        unit = SpyNetUnit(6 if i == 0 else 8) for i in range(k)
        unit.device()
        chkp = torch.load(str(checkpoint_path / f'{i}'.pt), map_location=device)
        unit.load_state_dict(chkp)
        units.append(unit)
    
    return SpyNet(*units)


def train(**kwargs):

    def collate_fn(batch):
        im, flow = zip(*batch)
        return torch.stack(im), torch.stack(flow)

    train_tfms = T.Compose([
        OFT.Resize(config.GConf(kwargs['G']).image_size),
        OFT.RandomRotate(17),
        OFT.ToTensor(),
        OFT.Normalize(mean=[.485, .456, .406], 
                      std= [.229, .224, .225])
    ]) 

    valid_tfms = T.Compose([
        OFT.Resize(config.GConf(kwargs['G']).image_size),
        OFT.ToTensor(),
        OFT.Normalize(mean=[.485, .456, .406], 
                      std=[.229, .224, .225])
    ])
    
    train_ds = dataset.FlyingChairDataset(kwargs['root'], 
                                          transform=train_tfms)
    valid_ds = None
    
    train_dl = DataLoader(train_ds,
                          batch_size=kwargs['batch_size'],
                          num_workers=kwargs['dl_num_workers'],
                          shuffle=True,
                          collate_fn=collate_fn)

    # valid_dl = DataLoader(valid_ds,
    #                       batch_size=kwargs['batch_size'],
    #                       num_workers=kwargs['dl_num_workers'],
    #                       shuffle=False,
    #                       collate_fn=collate_fn)
    
    Gk = SpyNetUnit(6 if kwargs['G'] == 0 else 8)
    Gk.to(device)

    pyramid = load_pyramid(kwargs['G'], kwargs['checkpoint_dir'])

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=4e-5)
    loss_fn = spynet.nn.EPELoss()


@click.command()

@click.option('--root', 
              type=click.Path(file_okay=False, exists=True))

@click.option('--G', type=int, default=0)
@click.option('--checkpoint-dir', 
              type=click.Path(file_okay=False), default='models')
@click.option('--original-checkpoint', 
              type=click.Path(dir_okay=False, exists=True), 
              default='models/network-chairs-final.pytorch')
@click.option('--epochs', type=int, default=8)
@click.option('--batch-size', type=int, default=16)
@click.option('--dl-num-workers', type=int, default=4)

def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    main()
