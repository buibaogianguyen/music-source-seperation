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
        spec = torch.stft(
            input=waveform,
            n_ftt=self.ftt_bins,
            hop_length=self.hop_len,
            win_length=self.windows,
            return_complex=True
        )

        return spec
    
    def spectrogram_to_waveform(self, spec, length):
        waveform = torch.istft(
            input=spec,
            n_ftt=self.fft_bins,
            hop_length=self.hop_len,
            win_length=self.windows,
            length=length
        )

        return waveform
