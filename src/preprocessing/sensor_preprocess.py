import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

RAW_DIR = "data/sensors/"
OUT_PATH = "data/processed/sensor_data.npy"

def load_sensor_files():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    data = []

    for file in files:
        df = pd.read_csv(os.path.join(RAW_DIR, file))
        df = df.interpolate().fillna(method="bfill").fillna(method="ffill")
        data.append(df.values)

    return data

def normalize_sequences(sequences):
    scaler = StandardScaler()
    flat = np.concatenate(sequences, axis=0)
    scaler.fit(flat)

    normalized = [scaler.transform(seq) for seq in sequences]
    return normalized

if __name__ == "__main__":
    sequences = load_sensor_files()
    sequences = normalize_sequences(sequences)
    np.save(OUT_PATH, sequences)
    print(f"[✔] Saved processed sensor data → {OUT_PATH}")
