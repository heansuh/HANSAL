# apps/fashion-mnist/src/etl.py
import torch
import torchvision
import torchvision.transforms as transforms
import os
import argparse  # <--- NEW: For command line args
from datetime import datetime
from dateutil import parser as date_parser  # <--- NEW: Smart parsing
from dateutil import tz
from codecarbon import EmissionsTracker

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, '../data/raw')
PROCESSED_PATH = os.path.join(BASE_DIR, '../data/processed')
METRICS_DIR = os.path.join(BASE_DIR, '../metrics')

def parse_args():
    parser = argparse.ArgumentParser(description="ETL Pipeline with Time Travel")
    parser.add_argument(
        "--from_date", 
        type=str, 
        default=None, 
        help="Simulate a specific date/time (ISO format). Defaults to NOW if not set."
    )
    return parser.parse_args()

def get_simulation_time(date_str):
    """
    Parses string to datetime. 
    If time is missing, defaults to 00:00:00.
    If timezone is missing, defaults to UTC (+00:00).
    """
    if not date_str:
        return datetime.now()
    
    # Parse the string (handles "2026-01-19" and "2026-01-19T16:00:00+01:00")
    dt = date_parser.parse(date_str)
    
    # If no timezone info, assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz.UTC)
        
    return dt

def main():
    args = parse_args()
    
    # Determine the "Virtual" Current Time
    current_time = get_simulation_time(args.from_date)
    print(f"Simulation Time: {current_time}")

    # 1. Setup Energy Tracker
    os.makedirs(METRICS_DIR, exist_ok=True)
    tracker = EmissionsTracker(
        project_name="fashion-mnist-etl",
        output_dir=METRICS_DIR,
        output_file="emissions_etl.csv"
    )
    
    tracker.start()
    try:
        print("--- [Step 1] Starting ETL Pipeline ---")
        os.makedirs(PROCESSED_PATH, exist_ok=True)
        
        # 2. Extract & Transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = torchvision.datasets.FashionMNIST(
            root=RAW_PATH, train=True, download=True, transform=transform
        )
        
        # 3. Save with CUSTOM Timestamp
        data_loader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset))
        all_images, all_labels = next(iter(data_loader))
        
        # Format: YYYY-MM-DD_HH-MM-SS (Safe for filenames)
        timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"training_{timestamp_str}.pt"
        save_path = os.path.join(PROCESSED_PATH, filename)
        
        print(f"Saving versioned dataset: {filename}")
        torch.save((all_images, all_labels), save_path)
        
    finally:
        tracker.stop()

if __name__ == '__main__':
    main()