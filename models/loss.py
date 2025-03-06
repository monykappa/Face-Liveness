import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
class PixelWiseLoss(nn.Module):
    def __init__(self, lambda_pixel=0.5):
        super(PixelWiseLoss, self).__init__()
        self.lambda_pixel = lambda_pixel
        self.criterion = nn.BCELoss()

    def forward(self, net_mask, net_label, target_mask, target_label):
        loss_pixel = self.criterion(net_mask, target_mask)
        loss_bce = self.criterion(net_label, target_label)
        return self.lambda_pixel * loss_pixel + (1 - self.lambda_pixel) * loss_bce