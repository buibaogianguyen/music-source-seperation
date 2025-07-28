import torchaudio
import torch
import os

def load_audio(file_path, sample_rate=44100):
    waveform, samp_rate = torchaudio.load(file_path)
    if samp_rate != sample_rate:
        waveform = torchaudio.transforms.Resample(samp_rate, sample_rate)(waveform)

    return waveform

def save_audio(file_path, waveform, sample_rate=44100):
    torchaudio.save(file_path, waveform, sample_rate)

def load_musdb(track, segment_len, sample_rate=44100):
    try:
        mix_path = track.get('mixture')
        vocals_path = track.get('vocals')
        bg_path = track.get('accompaniment')

        if not all([mix_path, vocals_path, bg_path]):
            raise ValueError("Missing audio file paths in dataset")

        mix, sr = torchaudio.load(mix_path)
        vocals, sr_v = torchaudio.load(vocals_path)
        bg, sr_b = torchaudio.load(bg_path)

        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            mix = resampler(mix)
            vocals = resampler(vocals)
            bg = resampler(bg)

        return mix, vocals, bg
    except Exception as e:
        print(f"Error loading track: {e}")
        # Dummy tensors to avoid crash
        dummy = torch.zeros(2, segment_len)
        return dummy, dummy, dummy