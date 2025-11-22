import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

from src.models.sensor_model import SensorModel
from src.models.image_model import ImageModel
from src.models.text_model import TextModel
from src.models.fusion_model import FusionModel


class FusionDataset(Dataset):
    def __init__(self, sensor_path, img_dir, txt_dir):
        self.sensor = np.load(sensor_path, allow_pickle=True)
        self.imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        self.txts = [os.path.join(txt_dir, f) for f in os.listdir(txt_dir)]

    def __len__(self):
        return len(self.sensor)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sensor[idx], dtype=torch.float32)

        img = np.load(self.imgs[idx % len(self.imgs)])
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        txt_enc = np.load(self.txts[idx % len(self.txts)], allow_pickle=True).item()
        ids = torch.tensor(txt_enc["input_ids"], dtype=torch.int64)
        mask = torch.tensor(txt_enc["attention_mask"], dtype=torch.int64)

        label = torch.randint(0, 5, (1,)).item()
        return seq, img, ids, mask, label


def train_epoch(model, loaders, optimizers, criterion, device):
    sensor_m, img_m, txt_m, fusion_m = model
    opt_s, opt_i, opt_t, opt_f = optimizers
    loader = loaders

    fusion_m.train()

    for m in model:
        m.train()

    total_loss = 0

    for seq, img, ids, mask, label in tqdm(loader, desc="Training Fusion Model"):
        seq, img = seq.to(device), img.to(device)
        ids, mask, label = ids.to(device), mask.to(device), label.to(device)

        # Forward through each model
        s = sensor_m(seq)
        i = img_m(img)
        t = txt_m(ids, mask)

        out = fusion_m(s, i, t)
        loss = criterion(out, label)

        # Backprop fusion only (freeze unimodal heads if desired)
        opt_f.zero_grad()
        loss.backward()
        opt_f.step()

        total_loss += loss.item()

    return total_loss / len(loader)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained models
    sensor = SensorModel().to(device)
    image = ImageModel().to(device)
    text = TextModel().to(device)
    fusion = FusionModel().to(device)

    sensor.load_state_dict(torch.load("checkpoints/sensor_model.pth"))
    image.load_state_dict(torch.load("checkpoints/image_model.pth"))
    text.load_state_dict(torch.load("checkpoints/text_model.pth"))

    dataset = FusionDataset(
        "data/processed/sensor_data.npy",
        "data/processed/images/",
        "data/processed/text/"
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    optimizers = [
        torch.optim.Adam(sensor.parameters(), lr=1e-6),
        torch.optim.Adam(image.parameters(), lr=1e-6),
        torch.optim.Adam(text.parameters(), lr=1e-6),
        torch.optim.Adam(fusion.parameters(), lr=1e-4),
    ]

    for epoch in range(5):
        loss = train_epoch(
            (sensor, image, text, fusion),
            loader,
            optimizers,
            criterion,
            device
        )
        print(f"Fusion Epoch {epoch+1} — Loss: {loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(fusion.state_dict(), "checkpoints/fusion_model.pth")
    print("[✔] Fusion model saved")
