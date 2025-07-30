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
    mix = load_audio(track['mixture'], sample_rate)
    vocals = load_audio(track['vocals'], sample_rate)
    bg = load_audio(track['accompaniment'], sample_rate)

    if mix.ndim == 1:
        mix = mix.unsqueeze(0)
    if vocals.ndim == 1:
        vocals = vocals.unsqueeze(0)
    if bg.ndim == 1:
        bg = bg.unsqueeze(0)

    if mix.size(-1) < segment_len:
        raise ValueError(f"Track too short, {mix.size(-1)} samples")
    start = torch.randint(0, mix.size(-1) - segment_len, (1,)).item()
    mix = mix[:, start:start+segment_len]
    vocals = vocals[:, start:start+segment_len]
    bg = bg[:, start:start+segment_len]

    return mix, vocals, bg