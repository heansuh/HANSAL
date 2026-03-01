# apps/cifar10/src/main.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import click
from config import EPOCHS, BATCH_SIZE, LEARNING_RATE, METRICS_DIR
from utils import extract_perun_metrics_to_csv

VALID_MODELS = ["SimpleCNN", "resnet18", "densenet121"]


@click.group()
def cli():
    """HANSAL ML Pipeline - CIFAR-10"""
    pass


@cli.command()
@click.option("--from_date", default=None, help="Simulate a specific date (ISO format)")
@click.option("--augment", is_flag=True, default=False, help="Enable data augmentation")
def etl(from_date, augment):
    """Run ETL pipeline"""
    from etl import main as run_etl
    run_etl(from_date=from_date, augment=augment)
    extract_perun_metrics_to_csv("etl_perun_results/main.json", os.path.join(METRICS_DIR, "perun_etl_metrics.csv"))


@cli.command()
@click.option("--from_date", default=None, help="Simulate a specific date (ISO format)")
@click.option("--model", default="SimpleCNN", show_default=True,
              type=click.Choice(VALID_MODELS), help="Model architecture")
@click.option("--epochs", default=EPOCHS, show_default=True, help="Number of epochs")
@click.option("--batch_size", default=BATCH_SIZE, show_default=True, help="Batch size")
@click.option("--lr", default=LEARNING_RATE, show_default=True, help="Learning rate")
def train(from_date, model, epochs, batch_size, lr):
    """Run training pipeline"""
    from train import main as run_train
    run_train(from_date=from_date, model_name=model, epochs=epochs, batch_size=batch_size, lr=lr)
    extract_perun_metrics_to_csv("train_perun_results/main.json", os.path.join(METRICS_DIR, "perun_train_metrics.csv"))


if __name__ == "__main__":
    cli()
