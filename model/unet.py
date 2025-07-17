import torch
import torch.nn as nn
import torch.nn.functional as F
from .tfc_tdf import TFCTDFBlock
from .dual_path import DualPathModule

class DTTNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_sources, ftt_bins):
        super(DTTNet,self).__init__()
        self.encoder = nn.ModuleList([
            TFCTDFBlock(in_channels, 32),
            TFCTDFBlock(32,64),
            TFCTDFBlock(64,128),
            TFCTDFBlock(128,256)
        ])

        self.downsample = nn.MaxPool2(kernel_size=2)
        self.dual_path = DualPathModule(channels=256)

        self.decoder = nn.ModuleList([
            TFCTDFBlock(256 + 128, 128), # account for skip connection 
            TFCTDFBlock(128 + 64, 64),
            TFCTDFBlock(64+32, 32),
            TFCTDFBlock(32+ in_channels, 16)
        ])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(32,num_sources*in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        skips = []

        # Encoder
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
            x = self.downsample(x)

        x = self.dual_path(x)

        #Decoder
        for i, layer in enumerate(self.decoder):
            x = self.upsample(x)
            x = torch.cat([x, skips[-(i+1)]], dim=1)
            x = layer(x)
