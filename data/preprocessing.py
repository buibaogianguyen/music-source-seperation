import torch
import torchaudio
import numpy as np

class AudioPreprocessor:
    def __init__(self, fft_bins, hop_len, sample_rate):
        self.fft_bins = fft_bins
        self.hop_len = hop_len
        self.sample_rate = sample_rate
        self.windows = torch.hann_window(fft_bins)

    def to_spectrogram(self, waveform):
        