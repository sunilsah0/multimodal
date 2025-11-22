import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm import tqdm

from src.models.image_model import ImageModel


class ImageDataset(Dataset):
    def __init__(self, img_dir):
        self.files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = np.load(self.files[idx])
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        label = torch.randint(0, 5, (1,)).item()  # placeholder
        return img, label


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for img, label in tqdm(loader, desc="Training Image Model"):
        img, label = img.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


if __name__ == "__main__":
    IMG_DIR = "data/processed/images/"

    dataset = ImageDataset(IMG_DIR)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImageModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(8):
        loss = train_epoch(model, loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1} — Loss: {loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/image_model.pth")
    print("[✔] Image model saved")
