import random
from typing import Union, Tuple

from PIL import Image

import torch
import torchvision.transforms.functional as F

import numpy as np
from skimage import transform


ImageOrTensor = Union['Image', torch.Tensor]
Transformed = Tuple[Tuple[ImageOrTensor, ImageOrTensor], 
                    Union[np.ndarray, torch.Tensor]]


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        o = args
        for t in self.transforms:
            o = t(*o)
        return o


class Resize(object):

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
    
    def __call__(self, 
                 frames: Tuple['Image', 'Image'], 
                 optical_flow: np.ndarray) -> Transformed:
        
        frame1 = F.resize(frames[0], (self.height, self.width))
        frame2 = F.resize(frames[1], (self.height, self.width))
        optical_flow = transform.resize(optical_flow,
                                        (self.height, self.width))

        return (frame1, frame2), optical_flow


class RandomRotate(object):

    def __init__(self, minmax: Union[Tuple[int, int], int]) -> None:
        self.minmax = minmax
        if isinstance(minmax, int):
            self.minmax = (-minmax, minmax)
    
    def __call__(self, 
                 frames: Tuple['Image', 'Image'], 
                 optical_flow: np.ndarray) -> Transformed:
        angle = random.randint(*self.minmax)
        frame1 = F.rotate(frames[0], angle)
        frame2 = F.rotate(frames[1], angle)
        optical_flow = transform.rotate(optical_flow, angle)

        return (frame1, frame2), optical_flow


class Normalize(object):

    def __init__(self, 
                 mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]) -> None:
        self.mean = mean
        self.std = std
    
    def __call__(self, 
                 frames: Tuple[torch.Tensor, torch.Tensor], 
                 optical_flow: torch.Tensor) -> Transformed:

        frame1 = F.normalize(frames[0], self.mean, self.std)
        frame2 = F.normalize(frames[1], self.mean, self.std)

        return (frame1, frame2), optical_flow


class ToTensor(object):

    def __call__(self, 
                 frames: Tuple['Image', 'Image'], 
                 optical_flow: np.ndarray) -> Transformed:
        
        return ((F.to_tensor(frames[0]), F.to_tensor(frames[1])), 
                 torch.from_numpy(optical_flow).permute(2, 0, 1).float())
