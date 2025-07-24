import torchaudio
import os

def load_audio(file_path, sample_rate=44100):
    waveform, samp_rate = torchaudio.load(file_path)
    if samp_rate != sample_rate:
        waveform = torchaudio.transforms.Resample(samp_rate, sample_rate)(waveform)

    return waveform

def save_audio(file_path, waveform, sample_rate=44100):
    torchaudio.save(file_path, waveform, sample_rate)

def load_musdb(track_path, sample_rate=44100):
    mixed = load_audio(os.path.join(track_path, 'mix.wav'), sample_rate)
    vocals = load_audio(os.path.join(track_path, 'vocals.wav'), sample_rate)
    instr = load_audio(os.path.join(track_path, 'accompaniment.wav'), sample_rate)

    return mixed, vocals, instr
