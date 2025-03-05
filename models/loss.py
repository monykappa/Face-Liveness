import torch 
from torch import nn  

class PixelWiseLoss(nn.Module):
    def __init__(self, lambda_pixel=0.5):
        super(PixelWiseLoss,self).__init__() 
        self.lambda_pixel = lambda_pixel
        self.criterion = nn.BCELoss()

    def forward(self, net_mask, net_label, target_mask, target_label):
        loss_pixel_map = self.criterion(net_mask, target_mask)
        loss_bce = self.criterion(net_label, target_label)
        loss = self.lambda_pixel* loss_pixel_map + (1 - self.lambda_pixel) * loss_bce
        return loss
    
        
        
        
