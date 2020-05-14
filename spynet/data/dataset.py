from typing import Union

from pathlib import Path

import torch
import numpy as np

from PIL import Image


def read_flow(filename: str) -> np.ndarray:

    f = open(filename, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2)
    flow = flow.reshape((height, width, 2))

    return flow.astype(np.float32)


class FlyingChairDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 root: Union[Path, str],
                 transform = None) -> None:

        self.root = Path(root)
        self.ids = set([o.stem.split('_')[0] for o in self.root.iterdir()])
        self.ids = list(self.ids)
        self.transform = transform

    def __getitem__(self, idx: int):
        id_ = self.ids[idx]
        frame1_path = self.root / (id_ + '_img1.ppm')
        frame2_path = self.root / (id_ + '_img2.ppm')
        optical_flow_path = self.root / (id_ + '_flow.flo')

        frame1 = Image.open(str(frame1_path))
        frame2 = Image.open(str(frame2_path))
        optical_flow = read_flow(str(optical_flow_path))

        if self.transform is not None:
            (frame1, frame2), optical_flow = \
                self.transform((frame1, frame2), optical_flow)
            
        return (frame1, frame2), optical_flow

    def __len__(self) -> int:
        return len(self.ids)
