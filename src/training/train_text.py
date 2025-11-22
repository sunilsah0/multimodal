import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm import tqdm

from src.models.text_model import TextModel


class TextDataset(Dataset):
    def __init__(self, txt_dir):
        self.files = [os.path.join(txt_dir, f) for f in os.listdir(txt_dir)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        enc = np.load(self.files[idx], allow_pickle=True).item()

        input_ids = torch.tensor(enc["input_ids"], dtype=torch.int64)
        mask = torch.tensor(enc["attention_mask"], dtype=torch.int64)

        label = torch.randint(0, 5, (1,)).item()  # placeholder
        return input_ids, mask, label


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0

    for ids, mask, label in tqdm(loader, desc="Training Text Model"):
        ids, mask, label = ids.to(device), mask.to(device), label.to(device)

        optimizer.zero_grad()
        out = model(ids, mask)
        loss = criterion(out, label)

        loss.backward()
        optimizer.step()
        total += loss.item()

    return total / len(loader)


if __name__ == "__main__":
    TXT_DIR = "data/processed/text/"
    dataset = TextDataset(TXT_DIR)
    loader = DataLoader(dataset, batch_size=12, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TextModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(5):
        loss = train_epoch(model, loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1} — Loss: {loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/text_model.pth")
    print("[✔] Text model saved")
