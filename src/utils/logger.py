import os
import datetime
import json


class Logger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.log_path = os.path.join(log_dir, f"log_{timestamp}.json")

        self.data = {
            "timestamp": timestamp,
            "entries": []
        }

    def log(self, epoch, train_loss, val_loss=None, extra=None):
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        }
        if extra:
            entry["extra"] = extra

        self.data["entries"].append(entry)
        self._save()

    def _save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.data, f, indent=4)

    def __repr__(self):
        return f"Logger(path={self.log_path})"
