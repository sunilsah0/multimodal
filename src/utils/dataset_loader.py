import torch
from torch.utils.data import Dataset
import numpy as np
import os


class MultimodalDataset(Dataset):
    """
    Loads synchronized sensor, image, and text items.
    Assumes equal dataset lengths or cycles through shorter ones.
    """
    def __init__(self, sensor_file, image_dir, text_dir, num_classes=5):
        self.sensor = np.load(sensor_file, allow_pickle=True)
        self.images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.texts = sorted([os.path.join(text_dir, f) for f in os.listdir(text_dir)])
        self.num_classes = num_classes

    def __len__(self):
        return len(self.sensor)

    def __getitem__(self, idx):
        # Sensor
        seq = torch.tensor(self.sensor[idx], dtype=torch.float32)

        # Image
        img_np = np.load(self.images[idx % len(self.images)])
        img = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1)

        # Text
        txt_enc = np.load(self.texts[idx % len(self.texts)], allow_pickle=True).item()
        ids = torch.tensor(txt_enc["input_ids"], dtype=torch.int64)
        mask = torch.tensor(txt_enc["attention_mask"], dtype=torch.int64)

        # Dummy labels (can be replaced with real labels)
        label = torch.randint(0, self.num_classes, (1,)).item()

        return seq, img, ids, mask, label
