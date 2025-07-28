import torch
import torch.optim as optim
from data.preprocessing import Preprocessor
from model.dttnet import DTTNet
from utils.audio_utils import load_musdb
from utils.loss import TimeFreqDomainLoss
import os
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

class MUSDBDataset(Dataset):
    def __init__(self, split='train', sample_rate=44100, segment_len = 44100*4):
        self.dataset = load_dataset('danjacobellis/musdb18HQ', split=split)
        
        self.sample_rate = sample_rate
        self.segment_len = segment_len

    def __len__(self):
        return len(self.tracks) * 10
    
    def __getitem__(self, idx):
        track_idx = idx // 10
        track = self.dataset[track_idx]

        mix, vocals, bg = load_musdb(track, self.sample_rate)

        start = torch.randint(0, mix.size(-1) - self.segment_length, (1,)).item()
        mix = mix[:, start:start+self.segment_length]
        vocals = vocals[:, start:start+self.segment_length]
        bg = bg[:, start:start+self.segment_length]
        
        return mix, vocals, bg

def train(model, optim, criterion, epochs, device, dataloader, preprocessor):
    for epoch in range(epochs):
        model.train()
        for mix, vocals, bg in dataloader:
            mix, vocals, bg = mix.to(device), vocals.to(device), bg.to(device)
            mix_spec = preprocessor.waveform_to_spectrogram(mix)
            vocals_spec = preprocessor.waveform_to_spectrogram(vocals)
            bg_spec = preprocessor.waveform_to_spectrogram(bg)


if __name__ == '__main__':
    lr = 0.001
    epochs = 100

    model = DTTNet(in_channels=2, num_sources=2, fft_bins=2048)
    optim = optim.Adam(model.parameters(), lr=lr)
    criterion = TimeFreqDomainLoss(alpha=0.5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MUSDBDataset(root_dir='')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    preprocessor = Preprocessor(fft_bins=2048, hop_len=512, sample_rate=44100)

    train(model, optim=optim, criterion=criterion, epochs=epochs, device=device, dataloader=dataloader, preprocessor=preprocessor)

