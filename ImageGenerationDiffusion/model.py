import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define beta schedule and precompute alphas
def get_noise_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1. - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas.to(device), alphas.to(device), alpha_bars.to(device)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device) * (-torch.log(torch.tensor(10000.0)) / half_dim))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class DenoiseModel(nn.Module):
    def __init__(self, time_dim=256):
        super().__init__()
        self.time_mlp1 = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, 128),
            nn.ReLU()
        )   

        self.time_mlp2 = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, 64),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 3, 3, padding=1)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, t):
        # Embed time and reshape to add to conv features
        t_emb1 = self.time_mlp1(t)[:, :, None, None]  # for 128-channel layers
        t_emb2 = self.time_mlp2(t)[:, :, None, None]  # for 64-channel layers
        
        
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h) + t_emb1)
        h = self.act(self.conv3(h))
        h = self.act(self.conv4(h) + t_emb2)
        h = self.conv5(h)  # output predicted noise

        return h
