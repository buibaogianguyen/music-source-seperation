import torchaudio
import torch
import os

def load_musdb(track, segment_len, sample_rate=44100):
    # print("Loading:", track['mixture'])
    # print(torchaudio.get_audio_backend())
    mix, mix_sr = torchaudio.load(track['mixture'])
    vocals, vocals_sr = torchaudio.load(track['vocals'])
    drums, drums_sr = torchaudio.load(track['drums'])
    bass, bass_sr = torchaudio.load(track['bass'])
    other, other_sr = torchaudio.load(track['other'])

    if mix_sr != sample_rate:
        mix = torchaudio.transforms.Resample(mix_sr, sample_rate)(mix)
    if vocals_sr != sample_rate:
        vocals = torchaudio.transforms.Resample(vocals_sr, sample_rate)(vocals)
    if bass_sr != sample_rate:
        bass = torchaudio.transforms.Resample(bass_sr, sample_rate)(bass)
    if drums_sr != sample_rate:
        drums = torchaudio.transforms.Resample(drums_sr, sample_rate)(drums)
    if other_sr != sample_rate:
        other = torchaudio.transforms.Resample(other_sr, sample_rate)(other)

    bg = drums + bass + other

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