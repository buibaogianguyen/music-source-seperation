import torch
import torch.optim as optim
from data.preprocessing import Preprocessor
from model.dttnet import DTTNet
from utils.audio_utils import load_musdb
from utils.loss import TimeFreqDomainLoss
import os
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torchaudio
from collections import defaultdict

class MUSDBDataset(Dataset):
    def __init__(self, split='train', sample_rate=44100, segment_len = 44100*4):
        self.dataset = load_dataset('danjacobellis/musdb18HQ', split=split)
        self.tracks = self.dataset
        
        self.sample_rate = sample_rate
        self.segment_len = segment_len

        self.songs = defaultdict(dict)
        for entry in self.dataset:
            song_name = entry['path'].split('/')[-2]
            self.songs[song_name][entry['instrument']] = entry['audio']

        self.song_names = list(self.songs.keys())

    def __len__(self):
        return len(self.tracks) * 10
    
    def __getitem__(self, idx):
        song_idx = idx // 10
        song_name = self.song_names[song_idx]
        stems = self.songs[song_name]

        stem_tensors = []
        for stem_audio in stems.values():
            if isinstance(stem_audio, dict) and 'array' in stem_audio:
                stem_tensors.append(torch.tensor(stem_audio['array']).float())
            else:
                stem_tensors.append(torch.tensor(stem_audio).float())
        mixture = sum(stem_tensors)

        vocals = torch.tensor(stems['vocals']['array']).float() if isinstance(stems['vocals'], dict) else torch.tensor(stems['vocals']).float()
        accompaniment = mixture - vocals

        if mixture.ndim == 1:
            mixture = mixture.unsqueeze(0)
        if vocals.ndim == 1:
            vocals = vocals.unsqueeze(0)
        if accompaniment.ndim == 1:
            accompaniment = accompaniment.unsqueeze(0)

        if mixture.size(-1) < self.segment_len:
            print(f"Track {song_name} too short: {mixture.size(-1)} samples")
            return None

        start = torch.randint(0, mixture.size(-1) - self.segment_len, (1,)).item()
        mixture = mixture[:, start:start+self.segment_len]
        vocals = vocals[:, start:start+self.segment_len]
        accompaniment = accompaniment[:, start:start+self.segment_len]

        return mixture, vocals, accompaniment
    
    

def train(model, optim, criterion, epochs, device, dataloader, preprocessor):
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()

        for mix, vocals, bg in dataloader:
            mix, vocals, bg = mix.to(device), vocals.to(device), bg.to(device)
            mix_spec = preprocessor.waveform_to_spectrogram(mix)
            vocals_spec = preprocessor.waveform_to_spectrogram(vocals)
            bg_spec = preprocessor.waveform_to_spectrogram(bg)

            mix_mag = preprocessor.normalize_spectrogram(mix_spec)
            masks = model(mix_mag)

            pred_vocals_spec = masks[:, :, 0, :, :]
            pred_bg_spec = masks[:, :, 1, :, :]
            pred_vocals = preprocessor.spectrogram_to_waveform(pred_vocals_spec, mix.size(-1))
            pred_bg = preprocessor.spectrogram_to_waveform(pred_bg_spec, mix.size(-1))
            
            loss = criterion(pred_vocals, pred_vocals_spec, vocals, vocals_spec) + criterion(pred_bg, pred_bg_spec, bg, bg_spec)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            avg_loss = total_loss/len(dataloader)

            print(f'Epoch {epoch+1}, Loss: {avg_loss}')
            scheduler.step(avg_loss)
            torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pth')


if __name__ == '__main__':
    lr = 0.001
    epochs = 100

    model = DTTNet(in_channels=2, num_sources=2, fft_bins=2048)
    optim = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=7)
    criterion = TimeFreqDomainLoss(alpha=0.5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MUSDBDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    preprocessor = Preprocessor(fft_bins=2048, hop_len=512, sample_rate=44100)

    train(model, optim=optim, criterion=criterion, epochs=epochs, device=device, dataloader=dataloader, preprocessor=preprocessor)

