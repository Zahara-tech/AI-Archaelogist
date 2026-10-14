"""
Generate reconstruction visualization plots for error analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append("src")
from data_loader import PHM2018Loader
from experiment_generator import MaskedSequenceGenerator
from baseline_model import InterpolationBaseline
from lstm_model import LSTMReconstructor, LSTMModelTrainer, ReconstructionDataset
from torch.utils.data import DataLoader
import torch
import json


def plot_reconstruction(sample_idx=0, sensor_idx=0):
    """Plot reconstruction for a single sample and sensor."""
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
    
    # Get sample
    seq = test_sequences[sample_idx]
    sensor_name = sensor_names[sensor_idx]
    
    # Get baseline prediction
    baseline = InterpolationBaseline()
    gap_length = seq["masked_segment"].shape[0]
    baseline_pred = baseline.predict(seq["context_before"], seq["context_after"], gap_length)
    
    # Get LSTM prediction
    test_dataset = ReconstructionDataset([seq])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
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
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            context = batch["context"].to(trainer.device)
            lstm_pred = model(context)
            lstm_pred = lstm_pred.cpu().numpy()[0]
    
    # Construct full signal
    context_before = seq["context_before"][:, sensor_idx]
    context_after = seq["context_after"][:, sensor_idx]
    ground_truth = seq["masked_segment"][:, sensor_idx]
    baseline_pred_sensor = baseline_pred[:, sensor_idx]
    lstm_pred_sensor = lstm_pred[:, sensor_idx]
    
    # Create time indices
    t_before = np.arange(len(context_before))
    t_gap = np.arange(len(context_before), len(context_before) + len(ground_truth))
    t_after = np.arange(len(context_before) + len(ground_truth), 
                        len(context_before) + len(ground_truth) + len(context_after))
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Full signal with reconstructions
    axes[0].plot(t_before, context_before, 'b-', label='Context Before', linewidth=2)
    axes[0].plot(t_gap, ground_truth, 'g-', label='Ground Truth', linewidth=2)
    axes[0].plot(t_gap, baseline_pred_sensor, 'r--', label='Baseline', linewidth=2)
    axes[0].plot(t_gap, lstm_pred_sensor, 'm:', label='LSTM', linewidth=2)
    axes[0].plot(t_after, context_after, 'b-', linewidth=2)
    axes[0].axvspan(t_gap[0], t_gap[-1], alpha=0.2, color='gray', label='Masked Region')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Normalized Value')
    axes[0].set_title(f'{sensor_name} - Full Reconstruction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Zoom on masked region
    axes[1].plot(t_gap, ground_truth, 'g-', label='Ground Truth', linewidth=2)
    axes[1].plot(t_gap, baseline_pred_sensor, 'r--', label='Baseline', linewidth=2)
    axes[1].plot(t_gap, lstm_pred_sensor, 'm:', label='LSTM', linewidth=2)
    axes[1].set_xlabel('Time Step (Masked Region)')
    axes[1].set_ylabel('Normalized Value')
    axes[1].set_title(f'{sensor_name} - Masked Region Zoom')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Error comparison
    baseline_error = np.abs(baseline_pred_sensor - ground_truth)
    lstm_error = np.abs(lstm_pred_sensor - ground_truth)
    
    axes[2].plot(t_gap, baseline_error, 'r-', label='Baseline Error', linewidth=2)
    axes[2].plot(t_gap, lstm_error, 'm-', label='LSTM Error', linewidth=2)
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Absolute Error')
    axes[2].set_title(f'{sensor_name} - Reconstruction Error')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / f"reconstruction_{sensor_name}_sample{sample_idx}.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    plt.close()


def generate_multiple_plots():
    """Generate plots for multiple sensors and samples."""
    # Select diverse sensors
    sensor_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    # Generate for first 3 samples
    for sample_idx in [0, 1, 2]:
        for sensor_idx in sensor_indices:
            try:
                plot_reconstruction(sample_idx, sensor_idx)
            except Exception as e:
                print(f"Error plotting sample {sample_idx}, sensor {sensor_idx}: {e}")


if __name__ == "__main__":
    generate_multiple_plots()
    print("\nGenerated reconstruction plots in results/plots/")
