import torch
import torch.nn as nn
import torchvision
import numpy as np

class SuccessiveConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.successive_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.successive_conv(x)

class Decoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels,kernel_size=2,stride=2)
        self.conv = SuccessiveConv(in_channels, out_channels)
        self.se = SELayer(out_channels)
    def forward(self,x1,x2):
        up_x = self.up_sample(x1)
        x = torch.cat((up_x,x2),dim=1)
        return self.se(self.conv(x))

class Decoder2_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels,kernel_size=2,stride=2)
        self.conv = SuccessiveConv(in_channels//2*3, out_channels)
        self.se = SELayer(out_channels)
    def forward(self,x1,x2,x3):
        up_x = self.up_sample(x1)
        x = torch.cat((up_x,x2,x3),dim=1)
        return self.se(self.conv(x))

class Encoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels,first=False):
        super().__init__()
        if first:
            self.contracting_path = nn.Sequential(
                SuccessiveConv(in_channels, out_channels),
                SELayer(out_channels)
            )
        else:
            self.contracting_path = nn.Sequential(
                nn.MaxPool2d(2),
                SuccessiveConv(in_channels, out_channels),
                SELayer(out_channels)
            )
    
    def forward(self, x):
        return self.contracting_path(x)


class ASPP(nn.Module):
    #https://www.cnblogs.com/haiboxiaobai/p/13029920.html
    def __init__(self, in_channel=512, depth=1024):
        super(ASPP,self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear',align_corners=True)
 
    def forward(self, x):
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = self.upsample(image_features)
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net
    
class SELayer(nn.Module):
    #https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)