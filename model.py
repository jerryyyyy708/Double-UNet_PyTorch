import torch
import torch.nn as nn
import torchvision
from modules import SuccessiveConv,Decoder_Block,Decoder2_Block,Encoder_Block,ASPP,SELayer

class Double_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.VGG=torchvision.models.vgg19(pretrained = True)
        #VGG
        self.VGG_block1 = nn.Sequential(*self.VGG.features[:4])#64
        self.VGG_block2 = nn.Sequential(*self.VGG.features[4:9])#128
        self.VGG_block3 = nn.Sequential(*self.VGG.features[9:18])#256
        self.VGG_block4 = nn.Sequential(*self.VGG.features[18:27])#512
        self.VGG_block5 = nn.Sequential(*self.VGG.features[27:-1])
        #ASPP
        self.ASPP = ASPP()
        #Decoder1
        self.dec1_block1 = Decoder_Block(1024,512)
        self.dec1_block2 = Decoder_Block(512,256)
        self.dec1_block3 = Decoder_Block(256,128)
        self.dec1_block4 = Decoder_Block(128,64)
        self.dec1_conv = nn.Conv2d(64,1,1)
        #encoder 2
        self.enc2_block1 = Encoder_Block(3,64,first=True)
        self.enc2_block2 = Encoder_Block(64,128)
        self.enc2_block3 = Encoder_Block(128,256)
        self.enc2_block4 = Encoder_Block(256, 512)
        self.enc2_block5 = Encoder_Block(512, 512)
        #decoder2
        self.dec2_block1 = Decoder2_Block(1024,512)
        self.dec2_block2 = Decoder2_Block(512,256)
        self.dec2_block3 = Decoder2_Block(256,128)
        self.dec2_block4 = Decoder2_Block(128,64)
        self.dec2_conv = nn.Conv2d(64,1,1)