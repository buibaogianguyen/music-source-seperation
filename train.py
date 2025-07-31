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
import numpy as np
import kagglehub

class MUSDBDataset(Dataset):
    def __init__(self, root, split='train', sample_rate=44100, segment_len = 44100*4):
        self.sample_rate = sample_rate
        self.segment_len = segment_len
        self.tracks = []
        self.stems = ['mixture', 'vocals', 'drums', 'bass', 'other']

        split_dir = os.path.join(root, split)
        song_names = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]

        for song in song_names:
            song_path = os.path.join(split_dir, song)
            track = {}
            for stem in self.stems:
                stem_path = os.path.join(song_path, f"{stem}.wav")
                if os.path.exists(stem_path):
                    track[stem] = stem_path
            if len(track) == len(self.stems):
                self.tracks.append(track)

    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, idx):
        track = self.tracks[idx]
        
        mixture, vocals, bg = load_musdb(track, self.segment_len, self.sample_rate)

        return mixture, vocals, bg

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
    root = kagglehub.dataset_download("quanglvitlm/musdb18-hq")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DTTNet(in_channels=2, num_sources=2, fft_bins=2048)
    model = model.to(device)
    optim = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=7)
    criterion = TimeFreqDomainLoss(alpha=0.5)
    dataset = MUSDBDataset(root=root)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    preprocessor = Preprocessor(fft_bins=2048, hop_len=512, sample_rate=44100)

    train(model, optim=optim, criterion=criterion, epochs=epochs, device=device, dataloader=dataloader, preprocessor=preprocessor)

