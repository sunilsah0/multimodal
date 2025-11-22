import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # 128 from each modality → 384 total
        self.fc = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, sensor_emb, image_emb, text_emb):
        fused = torch.cat([sensor_emb, image_emb, text_emb], dim=1)
        return self.fc(fused)
