import click
import random
from pathlib import Path
from typing import Union

import torch
import matplotlib.pyplot as plt

import spynet
import spynet.transforms as OFT


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



@click.command()
@click.option('--root', #required=True,
               type=click.Path(file_okay=False, exists=True))
@click.option('--checkpoint-name', required=True, type=str, default='sentinel')
def inference(root: str, checkpoint_name:str) -> None:

    tfms = OFT.Compose([
        OFT.ToTensor(),
        OFT.Normalize(mean=[.485, .406, .456], 
                      std= [.229, .225, .224])
    ])

    model = spynet.SpyNet.from_pretrained(checkpoint_name, map_location=device)
    model.to(device)
    model.eval()
    
    ds = spynet.dataset.FlyingChairDataset(root)
    o_frames, o_of = ds[random.randint(0, len(ds) - 1)]
    frames, of = tfms(o_frames, o_of)
    
    frames = [o.unsqueeze(0).to(device) for o in frames]
    
    with torch.no_grad():
        Vk = model(frames)[0]

    pred_of_im = spynet.flow.flow_to_image(Vk)
    true_of = spynet.flow.flow_to_image(of)

    plt.figure(figsize=(10, 4))

    plt.subplot(131)
    plt.title('Predictions')
    plt.imshow(pred_of_im)
    plt.axis('off')

    plt.subplot(132)
    plt.title('Ground Truth Frame 1')
    plt.imshow(o_frames[0])
    plt.axis('off')

    plt.subplot(133)
    plt.title('Ground Truth Frame 2')
    plt.imshow(o_frames[1])
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    inference()