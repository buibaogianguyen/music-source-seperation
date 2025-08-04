import torch
import torch.nn as nn

class TimeFreqDomainLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(TimeFreqDomainLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.alpha = alpha

    def si_snr(self, x, s, eps=1e-8):
        s_target = torch.sum(x * s, dim=-1, keepdim=True) * s / (torch.sum(s ** 2, dim=-1, keepdim=True) + eps)
        e_noise = x - s_target
        return -20 * torch.log10(torch.norm(s_target, dim=-1) / (torch.norm(e_noise, dim=-1) + eps))

    def forward(self, pred_wave, pred_spec, target_wave, target_spec):
        time_loss = self.l1(pred_wave, target_wave)
        freq_loss = self.l1(pred_spec.abs(), target_spec.abs())
        time_freq_loss = self.alpha*time_loss+(1-self.alpha)*freq_loss
        loss = time_freq_loss + 0.5 * self.si_snr(pred_wave, target_wave)
        return loss.mean()