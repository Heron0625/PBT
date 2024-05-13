import torch
import torch.nn as nn
import torch.nn.functional as F
import math



def get_gaussian_kernel(kernel_size=45, sigma=10, channels=1):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid] ,dim=-1).float()

    mean = (kernel_size - 1)/2
    variance = sigma **2

    gaussian_kernel = (1./(2.*math.pi*variance))*torch.exp(-torch.sum((xy_grid-mean)**2., dim=-1) / (2*variance))
    # gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size//2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter



class SoftLoULoss1(nn.Module):
    def __init__(self, a= 0.):
        super(SoftLoULoss1, self).__init__()
        self.a = a
        if a < 0 or a > 1:
            raise ('loss error due to a:{}'.format(a))
        self.iou = None
        self.loss1 = 0.
        self.loss2 = 0.
    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        smooth = 0.00

        target = target.float()
        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)

        loss = 1 - torch.mean(loss)

        return loss


