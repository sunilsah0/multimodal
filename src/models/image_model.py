import torch
import torch.nn as nn
from torchvision import models

class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        layers = list(base.children())[:-1]   # remove classifier
        self.backbone = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
        )

    def forward(self, x):
        feats = self.backbone(x)       # (batch, 2048, 1, 1)
        feats = feats.view(feats.size(0), -1)
        return self.fc(feats)
