# apps/cifar10/src/train.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import pandas as pd
from datetime import datetime
from dateutil import parser as date_parser
from dateutil import tz
from codecarbon import EmissionsTracker
from zeus.monitor import ZeusMonitor
import perun
from sklearn.metrics import classification_report

from utils import save_metrics, collect_codecarbon_metrics, get_logger, get_model
from config import PROCESSED_PATH, MODEL_SAVE_PATH, METRICS_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL

logger = get_logger()

CC_PROJECT_NAME = "cifar10-train"
CC_OUTPUT_FILE  = "emissions_train.csv"


def get_simulation_time(date_str):
    if not date_str:
        return datetime.now().replace(tzinfo=tz.UTC)
    dt = date_parser.parse(date_str)
    return dt.replace(tzinfo=tz.UTC) if dt.tzinfo is None else dt


def get_latest_valid_dataset(path, cutoff_date, prefix):
    files = glob.glob(os.path.join(path, f"{prefix}_*.pt"))
    if not files:
        return None
    valid = []
    for f in files:
        try:
            time_str = os.path.basename(f).replace(f"{prefix}_", "").replace(".pt", "")
            file_dt  = datetime.strptime(time_str, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=tz.UTC)
            if file_dt <= cutoff_date:
                valid.append((file_dt, f))
        except ValueError:
            continue
    return sorted(valid, reverse=True)[0][1] if valid else None


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            preds = model(inputs.to(device)).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    return all_preds, all_labels


@perun.perun(data_out="train_perun_results", format="json")
def main(from_date=None, model_name=MODEL, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    cutoff_date = get_simulation_time(from_date)
    logger.info(f"Cutoff Date:  {cutoff_date}")
    logger.info(f"Model:        {model_name}")
    logger.info(f"Epochs:       {epochs}")
    logger.info(f"Batch size:   {batch_size}")
    logger.info(f"LR:           {lr}")

    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    device  = torch.device("cuda" if use_gpu else "cpu")
    zeus    = ZeusMonitor(gpu_indices=[0]) if use_gpu else None
    tracker = EmissionsTracker(project_name=CC_PROJECT_NAME,
                               output_dir=METRICS_DIR,
                               output_file=CC_OUTPUT_FILE,
                               log_level="error")
    logger.info(f"Training on: {device}")

    # ── Load Data ─────────────────────────────────────────────────
    train_file = get_latest_valid_dataset(PROCESSED_PATH, cutoff_date, "training")
    test_file  = get_latest_valid_dataset(PROCESSED_PATH, cutoff_date, "test")
    if not train_file:
        logger.error("No training dataset found — run ETL first.")
        return

    train_inputs, train_labels = torch.load(train_file)
    trainloader = data_utils.DataLoader(
        data_utils.TensorDataset(train_inputs, train_labels),
        batch_size=batch_size, shuffle=True
    )
    testloader = None
    if test_file:
        test_inputs, test_labels = torch.load(test_file)
        testloader = data_utils.DataLoader(
            data_utils.TensorDataset(test_inputs, test_labels),
            batch_size=batch_size, shuffle=False
        )
    logger.info(f"Train: {os.path.basename(train_file)}")
    logger.info(f"Test:  {os.path.basename(test_file) if test_file else 'not found'}")

    # ── Model ─────────────────────────────────────────────────────
    model     = get_model(model_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    metrics = {
        "timestamp":  datetime.now().isoformat(),
        "dataset":    "CIFAR10",
        "stage":      "train",
        "model":      model_name,
        "device":     str(device),
        "epochs":     epochs,
        "batch_size": batch_size,
        "lr":         lr,
    }

    # ── Model Complexity ──────────────────────────────────────────
    metrics["total_params"]     = sum(p.numel() for p in model.parameters())
    metrics["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics["model_size_mb"]    = round(sum(p.nbytes for p in model.parameters()) / 1e6, 4)
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(torch.randn(1, 3, 32, 32).to(device),), verbose=False)  # 3-channel, 32x32
        metrics["flops"] = int(flops)
    except ImportError:
        pass

    # ── Start Monitors ────────────────────────────────────────────
    tracker.start()
    if zeus: zeus.begin_window("training")
    if use_gpu: torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    epoch_losses = []

    try:
        logger.info("--- Starting Training ---")

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            if zeus: zeus.begin_window(f"epoch_{epoch}")

            for batch_inputs, batch_labels in trainloader:
                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_inputs), batch_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(trainloader)
            epoch_losses.append(avg_loss)

            if zeus:
                e_mes = zeus.end_window(f"epoch_{epoch}")
                logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | GPU Energy: {e_mes.total_energy:.2f}J")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        if use_gpu: torch.cuda.synchronize()
        total_time = time.perf_counter() - t0

        # ── Inference Latency ─────────────────────────────────────
        if zeus: zeus.begin_window("inference")
        latencies = []
        with torch.no_grad():
            for i, (batch_inputs, _) in enumerate(trainloader):
                if i >= 50: break
                batch_inputs = batch_inputs.to(device)
                if use_gpu: torch.cuda.synchronize()
                t = time.perf_counter()
                model(batch_inputs)
                if use_gpu: torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t) * 1000)
        if zeus:
            inf_mes = zeus.end_window("inference")
            metrics["zeus_inference_energy_J"] = round(inf_mes.total_energy, 4)

        # ── Accuracy ─────────────────────────────────────────────
        train_preds, train_true = evaluate(model, trainloader, device)
        train_report = classification_report(train_true, train_preds, output_dict=True, zero_division=0)

        if testloader:
            val_preds, val_true = evaluate(model, testloader, device)
            val_report = classification_report(val_true, val_preds, output_dict=True, zero_division=0)

        # ── Save Model ────────────────────────────────────────────
        model_filename = f"{model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, model_filename))
        logger.info(f"Model saved: {model_filename}")

        # ── Metrics ───────────────────────────────────────────────
        metrics.update({
            "total_train_time_s":        round(total_time, 4),
            "throughput_sps":            round((len(train_inputs) * epochs) / total_time, 2),
            "final_loss":                round(epoch_losses[-1], 6),
            "best_loss":                 round(min(epoch_losses), 6),
            "best_epoch":                int(epoch_losses.index(min(epoch_losses))) + 1,
            "loss_improvement":          round(epoch_losses[0] - epoch_losses[-1], 6),
            "epoch_losses":              str(epoch_losses),
            "inference_latency_mean_ms": round(sum(latencies) / len(latencies), 4),
            "inference_latency_std_ms":  round(pd.Series(latencies).std(), 4),
            "train_accuracy":            round(train_report["accuracy"], 4),
            "train_macro_f1":            round(train_report["macro avg"]["f1-score"], 4),
            "train_weighted_f1":         round(train_report["weighted avg"]["f1-score"], 4),
        })
        if testloader:
            metrics.update({
                "val_accuracy":     round(val_report["accuracy"], 4),
                "val_macro_f1":     round(val_report["macro avg"]["f1-score"], 4),
                "val_weighted_f1":  round(val_report["weighted avg"]["f1-score"], 4),
                "val_macro_prec":   round(val_report["macro avg"]["precision"], 4),
                "val_macro_recall": round(val_report["macro avg"]["recall"], 4),
                "val_error_rate":   round(1 - val_report["accuracy"], 4),
            })
        if use_gpu:
            metrics["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 3)

    finally:
        emissions = tracker.stop()
        metrics["cc_co2_kg"] = round(emissions, 8)

        if zeus:
            train_mes = zeus.end_window("training")
            metrics.update({"zeus_gpu_energy_J": round(train_mes.total_energy, 4),
                            "zeus_wall_time_s":  round(train_mes.time, 4)})

        metrics.update(collect_codecarbon_metrics(os.path.join(METRICS_DIR, CC_OUTPUT_FILE)))

        run_config = {
            "model":         model_name,
            "epochs":        epochs,
            "batch_size":    batch_size,
            "learning_rate": lr,
            "device":        str(device),
            "from_date":     from_date,
        }
        save_metrics(metrics, "train_benchmark", run_config=run_config)


if __name__ == '__main__':
    main()
