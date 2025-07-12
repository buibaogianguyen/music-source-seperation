import torch
import torch.nn as nn
import torch.nn.functional as F

class TFCTDFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, min_bn_units):
        super(TFCTDFBlock, self).__init__
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, 
            padding=((kernel_size[0]-1)//2, (kernel_size[1]-1)//2)
            )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU
        self.tdf = nn.Linear(out_channels, min_bn_units)
        self.tdf_bn = nn.BatchNorm1d(min_bn_units)

        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            nn.Identity()

    def forward(self, x):
        # x shape: (batch, channels, freq, time)
        residual = self.residual(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        b, c, f, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b*f,t,c)
        x = self.tdf(x)
        x = self.tdf(x.permute(0,2,1)).permute(0,2,1)
        x = self.relu(x)
        x = x.reshape(b,f,t,-1).permute(0,3,1,2)
        return x + residual



