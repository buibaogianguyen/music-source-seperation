import torch
import torch.nn as nn
import torch.nn.functional as F
from .tfc_tdf import TFCTDFBlock
from .dual_path import DualPathModule

class DTTNet(nn.Module):
    def __init__(self, in_channels, num_sources, fft_bins):
        super(DTTNet,self).__init__()
        self.encoder = nn.ModuleList([
            TFCTDFBlock(in_channels, 16),
            TFCTDFBlock(16,32),
            TFCTDFBlock(32,64),
            TFCTDFBlock(64,128)
        ])

        self.downsample = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        self.dual_path = DualPathModule(channels=128)

        self.decoder = nn.ModuleList([
            TFCTDFBlock(128+128, 64), # account for skip connection 
            TFCTDFBlock(64+64, 32),
            TFCTDFBlock(32+32, 16),
            TFCTDFBlock(16+16, 16)
        ])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(16,num_sources*in_channels, kernel_size=1)
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
            skip = skips[-(i+1)]
            if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
                x = x[:, :, :skip.size(2), :skip.size(3)]
                if x.size(2) < skip.size(2) or x.size(3) < skip.size(3):
                    x = F.pad(x, (0, skip.size(3) - x.size(3), 0, skip.size(2) - x.size(2)))

            x = torch.cat([x, skip], dim=1)
            x = layer(x)

        masks = self.out_conv(x)
        masks = self.sigmoid(masks)

        masks = masks.view(masks.size(0), -1, 2, masks.size(2), masks.size(3)) # (b,c,s,f,t)
        return masks
