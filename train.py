import torch
import torch.optim as optim
from data.preprocessing import Preprocessor
from model.dttnet import DTTNet
from utils.audio_utils import load_musdb
from utils.loss import TimeFreqDomainLoss
import os
from torch.utils.data import DataLoader, Dataset

class MUSDBDataset(Dataset):
    def __init__(self, root_dir, sample_rate=44100, segment_len = 44100*4):
        self.root_dir = root_dir
        for d in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, d)):
                self.tracks = [os.path.join(root_dir, d)] 
        
        self.sample_rate = sample_rate
        self.segment_len = segment_len

    def __len__(self):
        return len(self.tracks) * 10
    
    def __getitem__(self, idx):
        track_path = self.tracks[idx // 10]
        mix, vocals, bg = load_musdb(track_path, self.sample_rate)
        start = torch.randint(0, mix.size(-1) - self.segment_length, (1,)).item()
        mix = mix[:, start:start+self.segment_length]
        vocals = vocals[:, start:start+self.segment_length]
        bg = bg[:, start:start+self.segment_length]
        return mix, vocals, bg

def train(model, optim, criterion, epochs, device, dataloader):
    for epoch in range(epochs):
        model.train()
        for mix, vocals, bg in dataloader:
            mix, vocals, bg = mix.to(device), vocals.to(device), bg.to(device)


if __name__ == '__main__':
    lr = 0.001
    epochs = 100

    model = DTTNet(in_channels=2, num_sources=2, fft_bins=2048)
    optim = optim.Adam(model.parameters(), lr=lr)
    criterion = TimeFreqDomainLoss(alpha=0.5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MUSDBDataset(root_dir='')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    train(model, optim=optim, criterion=criterion, epochs=epochs, device=device, dataloader=dataloader)

