# apps/fashion-mnist/src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import time
import os
import glob
import argparse
from datetime import datetime
from dateutil import parser as date_parser
from dateutil import tz
from codecarbon import EmissionsTracker

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: In Docker, we map /app/data, so we point to that relative path
PROCESSED_PATH = os.path.join(BASE_DIR, '../data/processed')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, '../model.pth')
METRICS_DIR = os.path.join(BASE_DIR, '../metrics')
EPOCHS = 20

# --- MODEL DEFINITION ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = self.fc1(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_date", type=str, default=None, help="Train on data available AS OF this date.")
    return parser.parse_args()

def get_simulation_time(date_str):
    if not date_str:
        return datetime.now().replace(tzinfo=tz.UTC)
    dt = date_parser.parse(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz.UTC)
    return dt

def get_latest_valid_dataset(path, cutoff_date):
    search_pattern = os.path.join(path, "training_*.pt")
    files = glob.glob(search_pattern)
    if not files:
        return None
    valid_files = []
    for f in files:
        try:
            basename = os.path.basename(f)
            time_str = basename.replace("training_", "").replace(".pt", "")
            file_dt = datetime.strptime(time_str, "%Y-%m-%d_%H-%M-%S")
            file_dt = file_dt.replace(tzinfo=tz.UTC)
            if file_dt <= cutoff_date:
                valid_files.append((file_dt, f))
        except ValueError:
            continue
    if not valid_files:
        return None
    valid_files.sort(key=lambda x: x[0], reverse=True)
    return valid_files[0][1]

def main():
    args = parse_args()
    cutoff_date = get_simulation_time(args.from_date)
    
    # SETUP TRACKER
    os.makedirs(METRICS_DIR, exist_ok=True)
    tracker = EmissionsTracker(
        project_name="fashion-mnist-train",
        output_dir=METRICS_DIR,
        output_file="emissions_train.csv"
    )

    tracker.start()
    try:
        print("--- [Step 2] Starting Training Pipeline (Monitored) ---")
        
        data_file = get_latest_valid_dataset(PROCESSED_PATH, cutoff_date)
        if not data_file:
            print(f"❌ No dataset found! Run etl.py first.")
            return

        print(f"📂 Loading data: {os.path.basename(data_file)}")
        inputs, labels = torch.load(data_file)
        
        trainset = data_utils.TensorDataset(inputs, labels)
        trainloader = data_utils.DataLoader(trainset, batch_size=64, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {device}")
        
        model = SimpleCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(trainloader):.4f}")

        print(f"✅ Training Finished in {time.time() - start_time:.2f}s")
        
    finally:
        tracker.stop()

if __name__ == '__main__':
    main()