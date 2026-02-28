# apps/fashion-mnist/utils.py
import os
import json
import pandas as pd
from config import METRICS_DIR
import logging
import json
import torch.nn as nn
import torchvision.models as models

_CC_FIELDS_PATH = os.path.join(os.path.dirname(__file__), 'codecarbon_fields.json')
with open(_CC_FIELDS_PATH) as f:
    _CC_FIELDS = json.load(f)


def collect_codecarbon_metrics(cc_path: str) -> dict:
    if not os.path.exists(cc_path):
        return {}
    last = pd.read_csv(cc_path).iloc[-1]
    return {alias: last.get(col) for col, alias in _CC_FIELDS.items()}


def save_metrics(metrics: dict, name: str, run_config: dict = None):
    logger = get_logger()
    os.makedirs(METRICS_DIR, exist_ok=True)

    # CSV — metrics only (flat)
    csv_path = os.path.join(METRICS_DIR, f"{name}.csv")
    df_new = pd.DataFrame([metrics])
    if os.path.exists(csv_path):
        df_new = pd.concat([pd.read_csv(csv_path), df_new], ignore_index=True)
    df_new.to_csv(csv_path, index=False)

    # JSON — metrics + config snapshot together
    json_path = os.path.join(METRICS_DIR, f"{name}_latest.json")
    snapshot = {
        "run_config": run_config or {},
        "metrics":    metrics
    }
    with open(json_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    logger.info(f"[Metrics] CSV  → {csv_path}")
    logger.info(f"[Metrics] JSON → {json_path}")


def get_logger(name: str = "fashion-mnist") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        return self.fc1(x)

def get_model(name):
    if name == "SimpleCNN":
        return SimpleCNN()
    elif name == "resnet18":
        m = models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(512, 10)
        return m
    elif name == "vgg16":
        m = models.vgg16(weights=None)
        m.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        m.classifier[-1] = nn.Linear(4096, 10)
        return m
    elif name == "densenet121":
        m = models.densenet121(weights=None)
        m.features.conv0 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.classifier = nn.Linear(1024, 10)
        return m
    else:
        raise ValueError(f"Unknown model: {name}")