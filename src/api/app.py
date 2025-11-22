import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.models.sensor_model import SensorModel
from src.models.image_model import ImageModel
from src.models.text_model import TextModel
from src.models.fusion_model import FusionModel

app = FastAPI(title="Multimodal Fault Diagnosis API")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------
# Load tokenizer and models on startup
# ----------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

sensor_m = SensorModel().to(device)
image_m = ImageModel().to(device)
text_m = TextModel().to(device)
fusion_m = FusionModel().to(device)

sensor_m.load_state_dict(torch.load("checkpoints/sensor_model.pth", map_location=device))
image_m.load_state_dict(torch.load("checkpoints/image_model.pth", map_location=device))
text_m.load_state_dict(torch.load("checkpoints/text_model.pth", map_location=device))
fusion_m.load_state_dict(torch.load("checkpoints/fusion_model.pth", map_location=device))

sensor_m.eval()
image_m.eval()
text_m.eval()
fusion_m.eval()


# ----------------------------------------------------------
# Helper Preprocessing Functions
# ----------------------------------------------------------

def preprocess_sensor(file):
    df = pd.read_csv(file)
    df = df.interpolate().fillna(method="bfill").fillna(method="ffill")
    arr = torch.tensor(df.values, dtype=torch.float32)
    return arr.unsqueeze(0).to(device)


def preprocess_image(bytes_data):
    nparr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    return img.unsqueeze(0).to(device)


def preprocess_text(text):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ----------------------------------------------------------
# Routes
# ----------------------------------------------------------

@app.get("/")
def root():
    return {"status": "API is running"}


@app.post("/predict")
async def predict(
    sensor: UploadFile = File(...),
    image: UploadFile = File(...),
    text: UploadFile = File(...)
):
    # Sensor CSV
    sensor_tensor = preprocess_sensor(sensor.file)

    # Image
    image_bytes = await image.read()
    img_tensor = preprocess_image(image_bytes)

    # Text
    text_content = (await text.read()).decode("utf-8")
    ids, mask = preprocess_text(text_content)

    with torch.no_grad():
        s = sensor_m(sensor_tensor)
        i = image_m(img_tensor)
        t = text_m(ids, mask)

        logits = fusion_m(s, i, t)
        probs = F.softmax(logits, dim=1)

    pred_idx = torch.argmax(probs).item()
    confidence = float(probs[0][pred_idx])

    fault_labels = [
        "Engine Misfire",
        "Battery/Alternator Issue",
        "Transmission Fault",
        "Cooling System Failure",
        "Sensor Malfunction"
    ]

    return {
        "fault_prediction": fault_labels[pred_idx],
        "confidence": round(confidence, 4)
    }
