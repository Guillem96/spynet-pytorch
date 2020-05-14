import torch
import torch.nn.functional as F


def warp(image: torch.Tensor, 
         optical_flow: torch.Tensor,
         device: torch.device = torch.device('cpu')) -> torch.Tensor:

    b, c, im_h, im_w = image.size() 
    
    hor = torch.linspace(-1.0, 1.0, im_w).view(1, 1, 1, im_w)
    hor = hor.expand(b, -1, im_h, -1)

    vert = torch.linspace(-1.0, 1.0, im_h).view(1, 1, im_h, 1)
    vert = vert.expand(b, -1, -1, im_w)

    grid = torch.cat([hor, vert], 1).to(device)
    optical_flow = torch.cat([
        optical_flow[:, 0:1, :, :] / ((im_w - 1.0) / 2.0), 
        optical_flow[:, 1:2, :, :] / ((im_h - 1.0) / 2.0)], dim=1)

    # Channels last (which corresponds to optical flow vectors coordinates)
    grid = (grid + optical_flow).permute(0, 2, 3, 1)
    return F.grid_sample(image, grid=grid, padding_mode='border', 
                         align_corners=True)


class EPELoss(torch.nn.Module):

    def __init__(self):
        super(EPELoss, self).__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dist = (target - pred).pow(2).sum().sqrt()
        return dist.mean()
    