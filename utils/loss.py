import torch
import torch.nn as nn

class TimeFreqDomainLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(MultiDomainLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.alpha = alpha