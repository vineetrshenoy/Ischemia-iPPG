# CONFIDENTIAL (C) Mitsubishi Electric Research Labs (MERL) 2020
# Armand Comas
# August 2020
# Modified by Suhas Lohit for BP stratification
# May 2022

#(c) MERL 2024
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
from torchvision import models
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class ResNetUNet(nn.Module):
    def __init__(self, window_shrink, in_channels, in_size=[58, 314]):
        super().__init__()
        print('shallow_nw')
        ch1 = 64
        ch2 = 128
        ch3 = 256
        ch4 = 512 #+ 512
        ch1_linear = in_channels
        #TODO: Check Architecture of GRU
        # self.config = Config_MMSE()  
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.interpolate_mode = 'linear'
        # self.interpolate_mode = 'nearest'
        # self.noGRU = self.config.noGRU 
        self.noGRU = False 
        self.window_shrink = window_shrink 
        self.linear = nn.Linear(in_channels, ch1_linear).to(self.device) #48*4 -> 48 for RGB+R/G -> one channel 48

        self.dropout = nn.Dropout(p=0.3)
        self.btnck_gru_2 = nn.GRU(ch3, ch1, num_layers=2, batch_first=True).to(self.device)
        self.btnck_gru_1 = nn.GRU(ch2, ch1, num_layers=2, batch_first=True).to(self.device)
        self.btnck_gru_0 = nn.GRU(ch1, ch1, num_layers=2, batch_first=True).to(self.device)


        self.layer0 = self.convbnrelu(self.device, ch1_linear, ch1,9,  3,  4) #kernal, stride, padding
        self.layer0_1 = self.convbnrelu(self.device, ch1, ch1,  5,  1,  2)
        self.layer0_1x1 = self.convrelu(self.device, ch1, ch1, 1, 1, 0)

        self.layer1 = self.convbnrelu(self.device, ch1, ch2,  7,  1,  3)
        self.layer1_1 = self.convbnrelu(self.device, ch2, ch2,  5,  1,  2)
        self.layer1_1x1 = self.convrelu(self.device, ch2, ch2, 1, 1, 0)

        self.layer2 = self.convbnrelu(self.device, ch2, ch3,  7,  2,  3)
        self.layer2_1 = self.convbn(self.device, ch3, ch3,  3,  1,  1)
        self.layer2_1x1 = self.convrelu(self.device, ch3, ch3, 1, 1, 0)

        self.layer3_7 = self.convbnrelu(self.device, ch3, ch4 // 2,  7,  1,  1)
        self.layer3_9 = self.convbnrelu(self.device, ch3, ch4 // 2,  9,  1,  2)

        self.layer3 = self.convbnrelu(self.device, ch3, ch4,  7,  1,  1)
        self.layer3_1 = self.convbn(self.device, ch4, ch4,  3,  1,  1)
        self.layer3_1x1 = self.convrelu(self.device, ch4, ch4, 1, 1, 0)

        if self.noGRU:
            self.conv_up2 = self.convrelu(self.device, ch3 + ch4, ch4,  3,  1,  1) # If no gru.
            self.conv_up1 = self.convrelu(self.device, ch2 + ch4, ch3,  3,  1,  1)
            self.conv_up0 = self.convrelu(self.device, ch1 + ch3, ch1,  3,  1,  1)
        else:
            self.conv_up2 = self.convrelu(self.device, ch3 + ch4 + ch1, ch4,  3,  1,  1)
            self.conv_up1 = self.convrelu(self.device, ch2 + ch4 + ch1, ch3,  3,  1,  1)
            self.conv_up0 = self.convrelu(self.device, ch1 + ch3 + ch1, ch1,  3,  1,  1)

        self.conv_original_size0 = self.convrelu(self.device, ch1_linear, ch1,  3,  1,  1)
        self.conv_original_size1 = self.convrelu(self.device, ch1, ch1,  3,  1,  1)
        self.conv_original_size2 = self.convrelu(self.device, ch1 + ch1, ch1,  3,  1,  1)
        #
        self.conv_last = nn.Conv1d(ch1, 1, 1).to(self.device)

               

        self.celu = nn.CELU()

    def forward(self, input):
        
        bs, regions, siglen = input.shape
        input = torch.permute(input, [0, 2, 1])
        ## Adding linear projection 
        input = self.linear(input)
        
        input = torch.permute(input, [0, 2, 1])

        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        x0 = self.layer0(input)
        # x0 = self.layer0_1(x0)
        x0 = self.dropout(x0)

        x1 = self.layer1(x0)
        # x1 = self.layer1_1(x1)
        x1 = self.dropout(x1)

        x2 = self.layer2(x1)
        # x2 = self.layer2_1(x2)
        x2 = self.dropout(x2)

        x3 = self.layer3(x2)
        x3 = self.layer3_1(x3)
        x3 = self.dropout(x3)

        x3 = self.layer3_1x1(x3)
        x = F.interpolate(x3, size=x2.shape[-1:], mode=self.interpolate_mode)
        t_enc_2 = self.btnck_gru_2(x2.permute(0,2,1))[0].permute(0, 2, 1)
        x2 = self.layer2_1x1(x2)
        if self.noGRU:
            x = torch.cat([x, x2], dim=1)
        else:
            x = torch.cat([x, x2, t_enc_2], dim=1)
        x = self.conv_up2(x)

        x = F.interpolate(x, size=x1.shape[-1:], mode=self.interpolate_mode)
        t_enc_1 = self.btnck_gru_1(x1.permute(0,2,1))[0].permute(0, 2, 1)
        # print(t_enc_1.shape)
        x1 = self.layer1_1x1(x1)
        if self.noGRU:
            x = torch.cat([x, x1], dim=1)
        else:
            x = torch.cat([x, x1, t_enc_1], dim=1)
        x = self.conv_up1(x)

        x = F.interpolate(x, size=x0.shape[-1:], mode=self.interpolate_mode)
        t_enc_0 = self.btnck_gru_0(x0.permute(0,2,1))[0].permute(0, 2, 1)
        x0 = self.layer0_1x1(x0)
        if self.noGRU:
            x = torch.cat([x, x0], dim=1)
        else:
            x = torch.cat([x, x0, t_enc_0], dim=1)
        x = self.conv_up0(x)

        x = F.interpolate(x, size=x_original.shape[-1:], mode=self.interpolate_mode)
        # Split channels and subtract
        x = torch.cat([x, x_original], dim=1)
        x = self.dropout(self.conv_original_size2(x))

        rppg_waveform = self.conv_last(x)[..., self.window_shrink:].squeeze()
        rppg_waveform = rppg_waveform.unsqueeze(1)
              

        return rppg_waveform

    def convbnrelu(self, device, in_channels, out_channels, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=padding),
            nn.GroupNorm(out_channels//4
                         , out_channels),
            nn.CELU()
        ).to(self.device)

    def convrelu(self, device, in_channels, out_channels, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=padding),
            nn.CELU()
        ).to(self.device)

    def convbn(self, device, in_channels, out_channels, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=padding),
            nn.GroupNorm(out_channels//4
                         , out_channels)
        ).to(self.device)

    def avg_pool_block(self, device, pool_kernel, pool_stride):
        return nn.AvgPool2d(pool_kernel, stride=pool_stride).to(self.device)

    def last_conv_block(self, device, conv_in, conv_out, conv_kernel, conv_stride, conv_padding):
        return nn.Conv1d(conv_in, conv_out, kernel_size=conv_kernel, stride=conv_stride, padding=conv_padding).to(self.device)


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    x = torch.randn(1, 48, 250).to(device)
    model = ResNetUNet(0, 48)
    out = model(x)
    temp = 5
