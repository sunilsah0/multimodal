import os
from transformers import AutoTokenizer
import numpy as np

RAW_DIR = "data/text/"
OUT_DIR = "data/processed/text/"
MODEL = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

def encode_text(text):
    return tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding="max_length",
        max_length=128
    )

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".txt")]

    for file in files:
        with open(os.path.join(RAW_DIR, file), "r") as f:
            txt = f.read()

        enc = encode_text(txt)
        np.save(os.path.join(OUT_DIR, file.replace(".txt", ".npy")), enc)

    print("[✔] Finished encoding text files")
