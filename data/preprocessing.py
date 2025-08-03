import torch
import torchaudio
import numpy as np

class Preprocessor:
    def __init__(self, fft_bins, hop_len, sample_rate):
        self.fft_bins = fft_bins
        self.hop_len = hop_len
        self.sample_rate = sample_rate
        self.windows = torch.hann_window(fft_bins)

    def waveform_to_spectrogram(self, waveform):
        original_shape = waveform.shape
        batch, channels, time = original_shape
        waveform = waveform.reshape(batch*channels, time)
        window = self.windows.to(waveform.device)
        spec = torch.stft(
            input=waveform,
            n_fft=self.fft_bins,
            hop_length=self.hop_len,
            win_length=self.fft_bins,
            return_complex=True,
            window=window,
        )
        spec = spec.reshape(batch, channels, spec.shape[-2], spec.shape[-1])
        return spec
    
    def spectrogram_to_waveform(self, spec, length=None):
        b, c, f, t = spec.shape
        spec = spec.reshape(b*c,f,t)
        window = self.windows.to(spec.device)
        waveform = torch.istft(
            input=spec,
            n_fft=self.fft_bins,
            hop_length=self.hop_len,
            win_length=self.fft_bins,
            window=window,
            length=length,
            return_complex=False
        )

        waveform = waveform.reshape(b, c, -1)
        return waveform
    
    def normalize_spectrogram(self, spec):
        mag = spec.abs()
        mag = (mag-mag.mean()) / (mag.std() + 1e-8)
        return mag
    
    def get_phase(self,spec):
        return torch.angle(spec)
