# apps/fashion-mnist/config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
RAW_PATH        = os.path.join(BASE_DIR, 'data/raw')
PROCESSED_PATH  = os.path.join(BASE_DIR, 'data/processed')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models')
METRICS_DIR     = os.path.join(BASE_DIR, 'metrics')

# Training
MODEL          = "SimpleCNN"   # SimpleCNN | resnet18 | densenet121
EPOCHS         = 50
BATCH_SIZE     = 256
LEARNING_RATE  = 0.001

# CodeCarbon
CC_PROJECT_NAME = "fashion-mnist"
CC_OUTPUT_FILE  = "emissions.csv"
