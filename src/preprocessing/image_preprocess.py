import os
import cv2
import numpy as np

RAW_DIR = "data/images/"
OUT_DIR = "data/processed/images/"

IMAGE_SIZE = 224

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(RAW_DIR) if f.endswith((".jpg", ".png"))]

    for f in files:
        img = preprocess_image(os.path.join(RAW_DIR, f))
        np.save(os.path.join(OUT_DIR, f"{f}.npy"), img)

    print("[✔] Processed all images")
