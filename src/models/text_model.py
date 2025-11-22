import torch
import torch.nn as nn
from transformers import AutoModel

class TextModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)

        self.fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = output.last_hidden_state[:, 0]  # CLS token
        return self.fc(cls)
