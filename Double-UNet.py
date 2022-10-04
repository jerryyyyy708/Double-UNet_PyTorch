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
        
    def forward(self,x):
        out1 = self.VGG_block1(x)#3,64
        out2 = self.VGG_block2(out1)#64,128
        out3 = self.VGG_block3(out2)#128,256
        out4 = self.VGG_block4(out3)#256,512
        output = self.VGG_block5(out4)
        aspp_out = self.ASPP(output)
        d1_1 = self.dec1_block1(aspp_out,out4)
        d1_2 = self.dec1_block2(d1_1,out3)
        d1_3 = self.dec1_block3(d1_2,out2)
        d1_4 = self.dec1_block4(d1_3,out1)
        d1_output = self.dec1_conv(d1_4)
        x2 = torch.matmul(x,d1_output)
        out5 = self.enc2_block1(x2)
        out6 = self.enc2_block2(out5)
        out7 = self.enc2_block3(out6)
        out8 = self.enc2_block4(out7)
        output2 = self.enc2_block5(out8)
        aspp_out2 = self.ASPP(output2)
        d2_1 = self.dec2_block1(aspp_out2,out8,out4)
        d2_2 = self.dec2_block2(d2_1,out7,out3)
        d2_3 = self.dec2_block3(d2_2,out6,out2)
        d2_4 = self.dec2_block4(d2_3,out5,out1)
        d2_output = self.dec2_conv(d2_4)
        #final_output = torch.cat((d1_output,d2_output))
        return d2_output
