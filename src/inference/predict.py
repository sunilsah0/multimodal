import torch
import torch.nn.functional as F
import numpy as np
import cv2
from transformers import AutoTokenizer
import argparse
import json
import os

from src.models.sensor_model import SensorModel
from src.models.image_model import ImageModel
from src.models.text_model import TextModel
from src.models.fusion_model import FusionModel


# ---------------------------------------------------------
# Utility Preprocessing Functions
# ---------------------------------------------------------

def load_sensor_file(path):
    """Load a CSV sensor time-series file."""
    import pandas as pd
    df = pd.read_csv(path)
    df = df.interpolate().fillna(method="bfill").fillna(method="ffill")
    arr = torch.tensor(df.values, dtype=torch.float32)
    return arr.unsqueeze(0)  # (1, seq, features)


def load_image_file(path):
    """Load and preprocess an image."""
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    return img.unsqueeze(0)  # (1, 3, 224, 224)


def load_text_file(path, tokenizer):
    """Tokenize a text file."""
    with open(path, "r") as f:
        txt = f.read()

    enc = tokenizer(
        txt,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    return enc["input_ids"], enc["attention_mask"]


# ---------------------------------------------------------
# Inference Function
# ---------------------------------------------------------

def run_inference(sensor_path, image_path, text_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Load pretrained models
    sensor_m = SensorModel().to(device)
    image_m = ImageModel().to(device)
    text_m  = TextModel().to(device)
    fusion_m = FusionModel().to(device)

    sensor_m.load_state_dict(torch.load("checkpoints/sensor_model.pth", map_location=device))
    image_m.load_state_dict(torch.load("checkpoints/image_model.pth", map_location=device))
    text_m.load_state_dict(torch.load("checkpoints/text_model.pth", map_location=device))
    fusion_m.load_state_dict(torch.load("checkpoints/fusion_model.pth", map_location=device))

    sensor_m.eval()
    image_m.eval()
    text_m.eval()
    fusion_m.eval()

    # Preprocess all inputs
    sensor_seq = load_sensor_file(sensor_path).to(device)
    img = load_image_file(image_path).to(device)
    ids, mask = load_text_file(text_path, tokenizer)
    ids, mask = ids.to(device), mask.to(device)

    # Forward pass through unimodal heads
    with torch.no_grad():
        s = sensor_m(sensor_seq)
        i = image_m(img)
        t = text_m(ids, mask)

        # Fusion prediction
        logits = fusion_m(s, i, t)
        probs = F.softmax(logits, dim=1)

    pred_idx = torch.argmax(probs).item()
    confidence = probs[0][pred_idx].item()

    # Example fault categories (customize as needed)
    fault_labels = [
        "Engine Misfire",
        "Battery/Alternator Issue",
        "Transmission Fault",
        "Cooling System Failure",
        "Sensor Malfunction"
    ]

    result = {
        "fault_prediction": fault_labels[pred_idx],
        "confidence": round(confidence, 4)
    }

    return result


# ---------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Fault Diagnosis Inference")
    parser.add_argument("--sensor", required=True, help="Path to sensor CSV file")
    parser.add_argument("--image", required=True, help="Path to image file (jpg/png)")
    parser.add_argument("--text", required=True, help="Path to text service log")

    args = parser.parse_args()

    output = run_inference(args.sensor, args.image, args.text)
    print(json.dumps(output, indent=4))
