import torch

import torch.nn as nn
from torchvision.models import resnet50


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(2048, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)

    def forward_one(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)
    
    def forward(self, img1, img2):
        emb1 = self.forward_one(img1)
        emb2 = self.forward_one(img2)
        distance = torch.norm(emb1 - emb2, p=2, dim=1)  # Euclidean distance

        return distance

