import click
from pathlib import Path
from typing import Tuple, Union, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import spynet
import spynet.transforms as OFT


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AVAILABLE_PRETRAINED = ['sentinel', 'kitti', 'flying-chair', 'none']


def train_one_epoch(dl: DataLoader,
                    optimizer: torch.optim.AdamW,
                    criterion_fn: torch.nn.Module,
                    Gk: torch.nn.Module, 
                    prev_pyramid: torch.nn.Module = None, 
                    print_freq: int = 100,
                    header: str = ''):
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
                Vk_1 = F.interpolate(
                    Vk_1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            Vk_1 = None

        predictions = Gk(x, Vk_1, upsample_optical_flow=False)

        if Vk_1 is not None:
            y = y - Vk_1

        loss = criterion_fn(y, predictions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % print_freq == 0:
            loss_mean = running_loss / i
            print(f'{header} [{i}/{len(dl)}] loss {loss_mean:.4f}')

    loss_mean = running_loss / len(dl)
    print(f'{header} loss {loss_mean:.4f}')


def load_data(root: str, k: int) -> Tuple[Subset, Subset]:
    train_tfms = OFT.Compose([
        OFT.Resize(*spynet.config.GConf(k).image_size),
        OFT.RandomRotate(17),
        OFT.ToTensor(),
        OFT.Normalize(mean=[.485, .406, .456], 
                      std= [.229, .225, .224])
    ]) 

    valid_tfms = OFT.Compose([
        OFT.Resize(*spynet.config.GConf(k).image_size),
        OFT.ToTensor(),
        OFT.Normalize(mean=[.485, .406, .456], 
                      std= [.229, .225, .224])
    ])
    
    train_ds = spynet.dataset.FlyingChairDataset(root,  transform=train_tfms)
    valid_ds = spynet.dataset.FlyingChairDataset(root, transform=valid_tfms)
    train_len = int(len(train_ds) * .9)
    rand_idx = torch.randperm(len(train_ds)).tolist()

    train_ds = Subset(train_ds, rand_idx[:train_len])
    valid_ds = Subset(valid_ds, rand_idx[train_len:])

    return train_ds, valid_ds


def collate_fn(batch):
    frames, flow = zip(*batch)
    frame1, frame2 = zip(*frames)
    return (torch.stack(frame1), torch.stack(frame2)), torch.stack(flow)


def build_dl(train_ds: Subset, 
             valid_ds: Subset,
             batch_size: int,
             num_workers: int) -> Tuple[DataLoader, DataLoader]:

    train_dl = DataLoader(train_ds,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=True,
                          collate_fn=collate_fn)

    valid_dl = DataLoader(valid_ds,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=False,
                          collate_fn=collate_fn)

    return train_dl, valid_dl


def build_spynets(k: int, name: str, 
                  previous: Sequence[torch.nn.Module]) \
                      -> Tuple[spynet.SpyNetUnit, spynet.SpyNet]:

    if name != 'none':
        pretrained = spynet.SpyNet.from_pretrained(name, map_location=device)
        current_train = pretrained.units[k]
    else:
        current_train = spynet.SpyNetUnit()
        
    current_train.to(device)
    current_train.train()
    
    if k == 0:
        Gk = None
    else:
        Gk = spynet.SpyNet(previous)
        Gk.to(device)
        Gk.eval()

    return current_train, Gk


def train_one_level(k: int, 
                    previous: Sequence[spynet.SpyNetUnit],
                    **kwargs) -> spynet.SpyNetUnit:

    print(f'Training level {k}...')

    train_ds, valid_ds = load_data(kwargs['root'], k)
    train_dl, valid_dl = build_dl(train_ds, valid_ds, 
                                  kwargs['batch_size'],
                                  kwargs['dl_num_workers'])

    current_level, trained_pyramid = build_spynets(
        k, kwargs['finetune_name'], previous)
    optimizer = torch.optim.AdamW(current_level.parameters(),
                                  lr=1e-5,
                                  weight_decay=4e-5)
    loss_fn = spynet.nn.EPELoss()

    for epoch in range(kwargs['epochs']):
        train_one_epoch(train_dl, 
                        optimizer,
                        loss_fn,
                        current_level,
                        trained_pyramid,
                        print_freq=999999,
                        header=f'Epoch [{epoch}] [Level {k}]')

    torch.save(current_level.state_dict(), 
               str(Path(kwargs['checkpoint_dir']) / f'{k}.pt'))
    
    return current_level


def train(**kwargs):
    torch.manual_seed(0)
    previous = []
    for k in range(kwargs.pop('levels')):
        previous.append(train_one_level(k, previous, **kwargs))

    final = spynet.SpyNet(previous)
    torch.save(final.state_dict(), 
               str(Path(kwargs['checkpoint_dir']) / f'final.pt'))


@click.command()

@click.option('--root', 
              type=click.Path(file_okay=False, exists=True))

@click.option('--checkpoint-dir', 
              type=click.Path(file_okay=False), default='models/spynet.pt')
@click.option('--finetune-name', 
              type=click.Choice(AVAILABLE_PRETRAINED), 
              default='none')

@click.option('--epochs', type=int, default=8)
@click.option('--batch-size', type=int, default=16)
@click.option('--dl-num-workers', type=int, default=4)

@click.option('--levels', type=int, default=5)

def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    main()
