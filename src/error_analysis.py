"""
Error Analysis for Reconstruction Models
Per-sensor metrics and masking length analysis.
"""

import numpy as np
import torch
import json
import csv
from pathlib import Path
from typing import List, Dict
import sys

sys.path.append("src")
from data_loader import PHM2018Loader
from experiment_generator import MaskedSequenceGenerator
from baseline_model import InterpolationBaseline
from lstm_model import LSTMReconstructor, LSTMModelTrainer, ReconstructionDataset
from torch.utils.data import DataLoader


def calculate_per_sensor_metrics(sequences: List[Dict], predictions: np.ndarray, sensor_names: List[str]) -> Dict:
    """Calculate per-sensor MSE and MAE."""
    ground_truths = np.array([seq["masked_segment"] for seq in sequences])
    
    # Per-sensor metrics
    sensor_mse = np.mean((predictions - ground_truths) ** 2, axis=(0, 1))
    sensor_mae = np.mean(np.abs(predictions - ground_truths), axis=(0, 1))
    
    per_sensor = {}
    for i, name in enumerate(sensor_names):
        per_sensor[name] = {
            "mse": float(sensor_mse[i]),
            "mae": float(sensor_mae[i])
        }
    
    return per_sensor


def analyze_baseline_per_sensor():
    """Analyze baseline performance per sensor."""
    print("Analyzing baseline per-sensor metrics...")
    
    # Load data
    loader = PHM2018Loader()
    preprocessed = loader.load_preprocessed(Path("data/preprocessed/phm2018_subset.pkl"))
    sensor_names = preprocessed["sensor_columns"]
    
    # Load sequences
    generator = MaskedSequenceGenerator()
    test_sequences = generator.load_sequences(Path("data/sequences/test_sequences.pkl"))
    
    # Get baseline predictions
    baseline = InterpolationBaseline()
    predictions = []
    for seq in test_sequences:
        gap_length = seq["masked_segment"].shape[0]
        pred = baseline.predict(seq["context_before"], seq["context_after"], gap_length)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Calculate per-sensor metrics
    per_sensor = calculate_per_sensor_metrics(test_sequences, predictions, sensor_names)
    
    # Save
    results = {
        "model": "baseline",
        "per_sensor_metrics": per_sensor
    }
    
    with open("results/baseline_per_sensor.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save CSV
    with open("results/per_sensor_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sensor", "model", "mse", "mae"])
        for name, metrics in per_sensor.items():
            writer.writerow([name, "baseline", metrics["mse"], metrics["mae"]])
    
    print(f"Saved per-sensor metrics for baseline")
    return per_sensor


def analyze_lstm_per_sensor():
    """Analyze LSTM performance per sensor."""
    print("Analyzing LSTM per-sensor metrics...")
    
    # Load data
    loader = PHM2018Loader()
    preprocessed = loader.load_preprocessed(Path("data/preprocessed/phm2018_subset.pkl"))
    sensor_names = preprocessed["sensor_columns"]
    
    # Load sequences
    generator = MaskedSequenceGenerator()
    test_sequences = generator.load_sequences(Path("data/sequences/test_sequences.pkl"))
    
    # Load config
    with open("data/sequences/config.json", "r") as f:
        config = json.load(f)
    
    # Create dataset
    test_dataset = ReconstructionDataset(test_sequences)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Load model
    sample = test_sequences[0]
    input_dim = sample["context_before"].shape[1]
    output_dim = input_dim
    output_length = config["gap_length"]
    
    model = LSTMReconstructor(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        output_dim=output_dim,
        output_length=output_length,
        dropout=0.3
    )
    
    trainer = LSTMModelTrainer(model)
    trainer.load_checkpoint("models/best_model.pt")
    
    # Get predictions
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            context = batch["context"].to(trainer.device)
            pred = model(context)
            predictions.append(pred.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    
    # Calculate per-sensor metrics
    per_sensor = calculate_per_sensor_metrics(test_sequences, predictions, sensor_names)
    
    # Save
    results = {
        "model": "lstm",
        "per_sensor_metrics": per_sensor
    }
    
    with open("results/lstm_per_sensor.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Append to CSV
    with open("results/per_sensor_metrics.csv", "a", newline="") as f:
        writer = csv.writer(f)
        for name, metrics in per_sensor.items():
            writer.writerow([name, "lstm", metrics["mse"], metrics["mae"]])
    
    print(f"Saved per-sensor metrics for LSTM")
    return per_sensor


if __name__ == "__main__":
    import torch
    
    baseline_metrics = analyze_baseline_per_sensor()
    lstm_metrics = analyze_lstm_per_sensor()
    
    print("\nPer-Sensor Comparison:")
    print(f"{'Sensor':<30} {'Baseline MSE':<15} {'LSTM MSE':<15} {'Ratio':<10}")
    print("-" * 70)
    
    for sensor in baseline_metrics.keys():
        b_mse = baseline_metrics[sensor]["mse"]
        l_mse = lstm_metrics[sensor]["mse"]
        ratio = l_mse / b_mse if b_mse > 0 else float('inf')
        print(f"{sensor:<30} {b_mse:<15.6f} {l_mse:<15.6f} {ratio:<10.2f}x")
