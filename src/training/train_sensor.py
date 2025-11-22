import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os

from src.models.sensor_model import SensorModel


class SensorDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.randint(0, 5, (1,)).item()  # placeholder
        return seq, label


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for seq, label in tqdm(loader, desc="Training Sensor Model"):
        seq, label = seq.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for seq, label in loader:
            seq, label = seq.to(device), label.to(device)

            output = model(seq)
            loss = criterion(output, label)

            total_loss += loss.item()

    return total_loss / len(loader)


if __name__ == "__main__":
    DATA_PATH = "data/processed/sensor_data.npy"
    
    dataset = SensorDataset(DATA_PATH)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SensorModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1} — Loss: {train_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/sensor_model.pth")
    print("[✔] Sensor model saved")
