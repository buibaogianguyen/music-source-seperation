import torch
import torch.nn as nn

class TimeFreqDomainLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(TimeFreqDomainLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.alpha = alpha

    def forward(self, pred_wave, pred_spec, target_wave, target_spec):
        time_loss = self.l1(pred_wave, target_wave)
        freq_loss = self.l1(pred_spec.abs(), target_spec.abs())
        return self.alpha*time_loss+(1-self.alpha)*freq_loss