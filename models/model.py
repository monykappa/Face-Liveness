import torch    
from torch import  nn 
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

    def forward(self, x):
        enc = self.enc(x)
        dec = self.dec(enc)
        out_map = F.sigmoid(dec)
        out = self.linear(out_map.view(-1, 14 * 14))
        out = F.sigmoid(out)
        out = torch.flatten(out)
        return out_map, out
    
# class PixelWiseDeep(nn.Module):
#     def __init__(self, pretrained=True):
#         super(PixelWiseDeep, self).__init__()
#         dense = models.densenet161(pretrained=pretrained)
#         features = list(dense.features.children())
#         self.enc = nn.Sequential(*features[:10])
#         self.add_conv = nn.Sequential(
#             nn.Conv2d(1056, 512, kernel_size=3, stride=1, padding=1)
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, 384, kernel_size=3, stride=1, padding=1)
#             nn.BatchNorm2d(384),
#             nn.ReLU(),
#         )
#         self.dec = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
#         self.linear = nn.Linear(14 * 14, 1)

#     def forward(self, x):
#         enc = self.enc(x)
        
#         dec = self.dec(enc)
#         out_map = F.sigmoid(dec)
#         out = self.linear(out_map.view(-1, 14 * 14))
#         out = F.sigmoid(out)
#         out = torch.flatten(out)
#         return out_map, out