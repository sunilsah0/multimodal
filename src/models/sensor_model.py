import torch
import torch.nn as nn

class SensorModel(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        hn = torch.cat((hn[-1], hn[-2]), dim=1)   # concat both directions
        return self.fc(hn)
