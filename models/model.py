import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

class PixelWise(nn.Module):
    def __init__(self, pretrained=True):
        super(PixelWise, self).__init__()
        dense = models.densenet161(pretrained=pretrained)
        features = list(dense.features.children())
        self.enc = nn.Sequential(*features[:8])
        self.dec = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(14 * 14, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        enc = self.enc(x)
        dec = self.dec(enc)
        out_map = torch.sigmoid(dec)
        out = self.dropout(out_map.view(-1, 14 * 14))
        out = self.linear(out)
        out = torch.sigmoid(out)
        return out_map, torch.flatten(out)