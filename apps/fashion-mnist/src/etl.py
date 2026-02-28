# apps/fashion-mnist/src/etl.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torchvision
import torchvision.transforms as transforms
import time
import argparse
from datetime import datetime
from dateutil import parser as date_parser
from dateutil import tz
from codecarbon import EmissionsTracker
from zeus.monitor import ZeusMonitor

from config import RAW_PATH, PROCESSED_PATH, METRICS_DIR
from utils import save_metrics, collect_codecarbon_metrics, get_logger

CC_PROJECT_NAME = "fashion-mnist-etl"
CC_OUTPUT_FILE  = "emissions_etl.csv"

logger = get_logger()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_date", type=str, default=None)
    return parser.parse_args()


def get_simulation_time(date_str):
    if not date_str:
        return datetime.now()
    dt = date_parser.parse(date_str)
    return dt.replace(tzinfo=tz.UTC) if dt.tzinfo is None else dt


def load_and_save(dataset, prefix, timestamp_str):
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), num_workers=2)
    t = time.perf_counter()
    images, labels = next(iter(loader))
    load_t = time.perf_counter() - t

    fname = f"{prefix}_{timestamp_str}.pt"
    s = time.perf_counter()
    torch.save((images, labels), os.path.join(PROCESSED_PATH, fname))
    save_t = time.perf_counter() - s

    logger.info(f"Saved: {fname}")
    return images, labels, load_t, save_t


def main():
    args          = parse_args()
    current_time  = get_simulation_time(args.from_date)
    timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.info(f"Simulation Time: {current_time}")

    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    zeus    = ZeusMonitor(gpu_indices=[0]) if use_gpu else None
    tracker = EmissionsTracker(project_name=CC_PROJECT_NAME,
                               output_dir=METRICS_DIR,
                               output_file=CC_OUTPUT_FILE,
                               log_level="error")

    metrics = {"timestamp": current_time.isoformat(), "dataset": "FashionMNIST", "stage": "etl"}

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    tracker.start()
    if zeus: zeus.begin_window("etl")
    if use_gpu: torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    try:
        logger.info("--- Starting ETL Pipeline ---")

        trainset = torchvision.datasets.FashionMNIST(RAW_PATH, train=True,  download=True, transform=transform)
        testset  = torchvision.datasets.FashionMNIST(RAW_PATH, train=False, download=True, transform=transform)

        train_images, _, load_t, save_t = load_and_save(trainset, "training", timestamp_str)
        _,            _, _,      _      = load_and_save(testset,  "test",     timestamp_str)

        total_time = time.perf_counter() - t0

        metrics.update({
            "total_etl_time_s": round(total_time, 4),
            "data_load_time_s": round(load_t, 4),
            "save_time_s":      round(save_t, 4),
            "n_train_samples":  len(trainset),
            "n_test_samples":   len(testset),
            "throughput_sps":   round(len(trainset) / total_time, 2),
            "image_shape":      str(tuple(train_images.shape[1:])),
            "dataset_size_mb":  round(train_images.element_size() * train_images.nelement() / 1e6, 3),
        })
        if use_gpu:
            metrics["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 3)

    finally:
        emissions = tracker.stop()
        metrics["cc_co2_kg"] = round(emissions, 8)

        if zeus:
            mes = zeus.end_window("etl")
            metrics.update({"zeus_gpu_energy_J": round(mes.total_energy, 4),
                            "zeus_wall_time_s":  round(mes.time, 4)})

        metrics.update(collect_codecarbon_metrics(os.path.join(METRICS_DIR, CC_OUTPUT_FILE)))

        run_config = {"from_date": args.from_date, "image_size": 28}
        save_metrics(metrics, "etl_benchmark", run_config=run_config)


if __name__ == '__main__':
    main()
