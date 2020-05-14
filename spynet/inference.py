import click
import random
from pathlib import Path
from typing import Union

import torch
import matplotlib.pyplot as plt

import spynet
import spynet.transforms as OFT


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_pyramid(k: int, checkpoint_path: Union[str, Path]) -> torch.nn.Module:
    checkpoint_path = Path(checkpoint_path)
    units = []

    for i in range(k):
        unit = spynet.SpyNetUnit()
        unit.eval()
        unit.to(device)

        chkp = torch.load(str(checkpoint_path / f'{i}.pt'), map_location=device)
        unit.load_state_dict(chkp)
        units.append(unit)
    
    return spynet.SpyNet(*units)


@click.command()
@click.option('--root', #required=True,
               type=click.Path(file_okay=False, exists=True))
@click.option('--checkpoint-dir', required=True,
               type=click.Path(file_okay=False, exists=True))
@click.option('--k', default=6, type=int)
def inference(root: str, checkpoint_dir: str, k: int) -> None:

    im_size = spynet.config.GConf(k - 1).image_size

    tfms = OFT.Compose([
        OFT.Resize(*im_size),
        OFT.ToTensor(),
        OFT.Normalize(mean=[.485, .406, .456], 
                      std= [.229, .225, .224])
    ])

    model = load_pyramid(k, checkpoint_dir)
    model.to(device)
    model.eval()
    
    ds = spynet.dataset.FlyingChairDataset(root, tfms)
    frames, of = ds[random.randint(0, len(ds) - 1)]
    frames = [o.unsqueeze(0).to(device) for o in frames]
    
    with torch.no_grad():
        Vk = model(frames)

    pred_of_im = spynet.flow.flow_to_image(
        Vk.cpu().squeeze().permute(1, 2, 0).numpy())

    true_of = spynet.flow.flow_to_image(
        of.cpu().squeeze().permute(1, 2, 0).numpy())

    plt.figure()

    plt.subplot(121)
    plt.title('Predictions')
    plt.imshow(pred_of_im)
    plt.axis('off')

    plt.subplot(122)
    plt.title('Ground Truth')
    plt.imshow(true_of)
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    inference()