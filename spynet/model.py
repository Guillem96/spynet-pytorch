from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import spynet


class SpyNetUnit(nn.Module):

    def __init__(self, input_channels: int = 8):
        super(SpyNetUnit, self).__init__()

        self.module = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, padding=3, stride=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),

            nn.Conv2d(32, 64, kernel_size=7, padding=3, stride=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),

            nn.Conv2d(64, 32, kernel_size=7, padding=3, stride=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),

            nn.Conv2d(32, 16, kernel_size=7, padding=3, stride=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),

            nn.Conv2d(16, 2, kernel_size=7, padding=3, stride=1))

    def forward(self, 
                frames: Tuple[torch.Tensor, torch.Tensor], 
                optical_flow: torch.Tensor = None,
                upsample_optical_flow: bool = True) -> torch.Tensor:
        f_frame, s_frame = frames

        if optical_flow is None:
            # If optical flow is None (k = 0) then create empty one having the
            # same size as the input frames, therefore there is no need to 
            # upsample it later
            upsample_optical_flow = False
            b, c, h, w = f_frame.size()
            optical_flow = torch.zeros(b, 2, h, w, device=s_frame.device)

        if upsample_optical_flow:
            optical_flow = F.interpolate(
                optical_flow, scale_factor=2, align_corners=True, 
                mode='bilinear')

        s_frame = spynet.nn.warp(s_frame, optical_flow, s_frame.device)
        s_frame = torch.cat([s_frame, optical_flow], dim=1)
        
        inp = torch.cat([f_frame, s_frame], dim=1)
        return self.module(inp)


class SpyNet(nn.Module):

    def __init__(self, *units: SpyNetUnit):
        super(SpyNet, self).__init__()
        self.units = nn.ModuleList(units)
    
    def forward(self, 
                frames: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        frames: Tuple[torch.Tensor, torch.Tensor]
            Highest resolution frames. Each tuple element has shape
            [BATCH, 3, HEIGHT, WIDTH]
        """
        Vk_1 = None

        for k, G in enumerate(self.units):
            im_size = spynet.config.GConf(k).image_size
            x1 = F.interpolate(frames[0], im_size, mode='bilinear',
                               align_corners=True)
            x2 = F.interpolate(frames[1], im_size, mode='bilinear',
                               align_corners=True)

            if Vk_1 is not None: # Upsample the previous optical flow
                Vk_1 = F.interpolate(
                    Vk_1, scale_factor=2, align_corners=True, 
                    mode='bilinear')

            Vk = G((x1, x2), Vk_1, upsample_optical_flow=False)
            Vk_1 = Vk + Vk_1 if Vk_1 is not None else Vk
        
        return Vk_1
