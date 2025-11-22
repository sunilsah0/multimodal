### Multimodal Vehicle Fault Diagnosis System

A deep learning pipeline that combines sensor data, vehicle images, and text-based service records to detect faults with high accuracy using a multimodal fusion architecture.

             ┌───────────────────────┐
             │   Sensor Data (CSV)   │
             └───────────┬───────────┘
                         ▼
                 [LSTM Encoder]
                         │
                         ▼
                 Sensor Embeddings
                         │
──────────────────────────────────────────────
                         │
             ┌───────────────────────┐
             │ Vehicle Images (JPG)   │
             └───────────┬───────────┘
                         ▼
              [ResNet / EfficientNet]
                         │
                         ▼
                Image Embeddings
                         │
──────────────────────────────────────────────
                         │
             ┌───────────────────────┐
             │  Service Logs (TXT)   │
             └───────────┬───────────┘
                         ▼
                 [BERT Encoder]
                         │
                         ▼
                  Text Embeddings
                         │
──────────────────────────────────────────────
                         ▼
                 [Fusion Network]
                         ▼
              Final Fault Prediction
