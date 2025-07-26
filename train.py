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
        mix, vocals, background = load_musdb(track_path, self.sample_rate)
        start = torch.randint(0, mix.size(-1) - self.segment_length, (1,)).item()
        mix = mix[:, start:start+self.segment_length]
        vocals = vocals[:, start:start+self.segment_length]
        background = background[:, start:start+self.segment_length]
        return mix, vocals, background

def train(model, optim, criterion, epochs, device):
    for epoch in range(epochs):
        model.train()


