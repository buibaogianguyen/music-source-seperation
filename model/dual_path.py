import torch
import torch.nn as nn

class DualPathModule(nn.Module):
    def __init__(self, channels, hidden_dim=64):
        super(DualPathModule, self).__init__()

        # time
        self.intra_conv = nn.Conv2d(channels, hidden_dim, kernel_size=(1,3), padding=(0,1))
        self.intra_bn = nn.BatchNorm2d(hidden_dim)

        # freq
        self.inter_conv = nn.Conv2d(channels, hidden_dim, kernel_size=(3,1), padding=(1,0))
        self.inter_bn = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU()

    def forward(self,x):
        intra = self.intra_conv(x)
        intra = self.intra_bn(intra)
        intra = self.relu(intra)

        inter = self.inter_conv(intra)
        inter = self.inter_bn(inter)
        inter = self.relu(inter)

        return x + inter

    