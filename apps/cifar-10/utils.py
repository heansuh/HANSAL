# apps/cifar10/utils.py
import csv
import os
import json
import pandas as pd
from config import METRICS_DIR
import logging
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

    csv_path = os.path.join(METRICS_DIR, f"{name}.csv")
    df_new = pd.DataFrame([metrics])
    if os.path.exists(csv_path):
        df_new = pd.concat([pd.read_csv(csv_path), df_new], ignore_index=True)
    df_new.to_csv(csv_path, index=False)

    json_path = os.path.join(METRICS_DIR, f"{name}_latest.json")
    snapshot = {
        "run_config": run_config or {},
        "metrics":    metrics
    }
    with open(json_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    logger.info(f"[Metrics] CSV  → {csv_path}")
    logger.info(f"[Metrics] JSON → {json_path}")


def get_logger(name: str = "cifar10") -> logging.Logger:  # ← updated default name
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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 3-channel RGB
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 8 * 8, 10)                    # 32x32 -> 16x16 -> 8x8

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # -> (32, 16, 16)
        x = self.pool(self.relu(self.conv2(x)))  # -> (64, 8, 8)
        x = x.view(-1, 64 * 8 * 8)
        return self.fc1(x)


def get_model(name):
    if name == "SimpleCNN":
        return SimpleCNN()
    elif name == "resnet18":
        m = models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 3-channel, no aggressive downsampling
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(512, 10)
        return m
    elif name == "vgg16":
        m = models.vgg16(weights=None)
        m.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 3-channel (actually default, but explicit)
        m.classifier[-1] = nn.Linear(4096, 10)
        return m
    elif name == "densenet121":
        m = models.densenet121(weights=None)
        m.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 3-channel
        m.classifier = nn.Linear(1024, 10)
        return m
    else:
        raise ValueError(f"Unknown model: {name}")


def extract_perun_metrics_to_csv(json_path: str, output_csv: str) -> None:
    logger = get_logger()
    METRICS = ["energy", "gpu_energy", "gpu_power", "runtime", "money"]

    if not os.path.isfile(json_path):
        logger.error(f"Perun JSON not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    app_name = data.get("id", "unknown")

    for run_entry in data.get("nodes", []):
        for run_id, run_node in run_entry.items():
            run_metrics = run_node.get("metrics", {})
            row = {
                "app_name": app_name,
                "run_id":   run_id,
            }
            for metric in METRICS:
                if metric in run_metrics:
                    if metric == "gpu_power":
                        row[metric] = run_metrics[metric].get("mean")
                    else:
                        row[metric] = run_metrics[metric].get("sum")
                else:
                    row[metric] = None
            rows.append(row)

    fieldnames = ["app_name", "run_id", "energy", "gpu_energy", "gpu_power", "runtime", "money"]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"[Perun] Saved {len(rows)} run(s) to {output_csv}")
