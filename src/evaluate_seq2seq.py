"""
Evaluate Seq2Seq model and compare with baseline and old BiLSTM.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
import csv

sys.path.append("src")
from data_loader import PHM2018Loader
from experiment_generator import MaskedSequenceGenerator
from seq2seq_model import Seq2SeqReconstructor, Seq2SeqTrainer, Seq2SeqDataset
from bilstm_model import BiLSTMReconstructor, BiLSTMModelTrainer, BiLSTMDataset
from baseline_model import InterpolationBaseline
from torch.utils.data import DataLoader


def evaluate_seq2seq():
    """Evaluate seq2seq model on test set."""
    print("="*70)
    print("SEQ2SEQ EVALUATION")
    print("="*70)
    
    # Load data
    loader = PHM2018Loader()
    preprocessed = loader.load_preprocessed(Path("data/preprocessed/phm2018_subset.pkl"))
    sensor_names = preprocessed["sensor_columns"]
    
    generator = MaskedSequenceGenerator()
    test_sequences = generator.load_sequences(Path("data/sequences/test_sequences.pkl"))
    
    with open("data/sequences/config.json", "r") as f:
        config = json.load(f)
    
    # Load seq2seq model
    sample = test_sequences[0]
    input_dim = sample["context_before"].shape[1]
    output_dim = input_dim
    output_length = config["gap_length"]
    
    seq2seq_model = Seq2SeqReconstructor(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        output_dim=output_dim,
        output_length=output_length,
        dropout=0.3
    )
    
    seq2seq_trainer = Seq2SeqTrainer(seq2seq_model)
    seq2seq_trainer.load_checkpoint("models/seq2seq_best_model.pt")
    
    # Get predictions
    test_dataset = Seq2SeqDataset(test_sequences)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    seq2seq_model.eval()
    seq2seq_predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for batch in test_loader:
            context_before = batch["context_before"].to(seq2seq_trainer.device)
            context_after = batch["context_after"].to(seq2seq_trainer.device)
            target = batch["target"].to(seq2seq_trainer.device)
            
            pred = seq2seq_model(context_before, context_after, teacher_forcing_ratio=0.0)
            seq2seq_predictions.append(pred.cpu().numpy())
            ground_truths.append(target.cpu().numpy())
    
    seq2seq_predictions = np.concatenate(seq2seq_predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)
    
    # Calculate metrics
    mse = np.mean((seq2seq_predictions - ground_truths) ** 2)
    mae = np.mean(np.abs(seq2seq_predictions - ground_truths))
    
    # Per-sensor metrics
    sensor_mse = np.mean((seq2seq_predictions - ground_truths) ** 2, axis=(0, 1))
    sensor_mae = np.mean(np.abs(seq2seq_predictions - ground_truths), axis=(0, 1))
    
    # Per-timestep metrics
    timestep_mse = np.mean((seq2seq_predictions - ground_truths) ** 2, axis=(0, 2))
    timestep_mae = np.mean(np.abs(seq2seq_predictions - ground_truths), axis=(0, 2))
    
    # Prediction variance
    prediction_variance = np.var(seq2seq_predictions, axis=1)
    mean_variance = np.mean(prediction_variance)
    
    print(f"\nSeq2Seq Results:")
    print(f"  Overall MSE: {mse:.6f}")
    print(f"  Overall MAE: {mae:.6f}")
    print(f"  Prediction Variance Mean: {mean_variance:.6f}")
    
    print(f"\nPer-Timestep MSE:")
    for t, (t_mse, t_mae) in enumerate(zip(timestep_mse, timestep_mae)):
        print(f"  Timestep {t}: MSE={t_mse:.6f}, MAE={t_mae:.6f}")
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "sensor_mse": sensor_mse.tolist(),
        "sensor_mae": sensor_mae.tolist(),
        "timestep_mse": timestep_mse.tolist(),
        "timestep_mae": timestep_mae.tolist(),
        "prediction_variance_mean": float(mean_variance)
    }, seq2seq_predictions, ground_truths


def load_old_bilstm_predictions(test_sequences):
    """Load old BiLSTM predictions."""
    print("\n" + "="*70)
    print("LOADING OLD BILSTM PREDICTIONS")
    print("="*70)
    
    with open("data/sequences/config.json", "r") as f:
        config = json.load(f)
    
    sample = test_sequences[0]
    input_dim = sample["context_before"].shape[1]
    output_dim = input_dim
    output_length = config["gap_length"]
    
    model = BiLSTMReconstructor(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        output_dim=output_dim,
        output_length=output_length,
        dropout=0.3
    )
    
    trainer = BiLSTMModelTrainer(model)
    trainer.load_checkpoint("models/bilstm_best_model.pt")
    
    test_dataset = BiLSTMDataset(test_sequences)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            context_before = batch["context_before"].to(trainer.device)
            context_after = batch["context_after"].to(trainer.device)
            pred = model(context_before, context_after)
            predictions.append(pred.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    
    print(f"Loaded BiLSTM predictions: {predictions.shape}")
    
    return predictions


def load_baseline_predictions(test_sequences):
    """Load baseline predictions."""
    print("\n" + "="*70)
    print("LOADING BASELINE PREDICTIONS")
    print("="*70)
    
    baseline = InterpolationBaseline()
    predictions = []
    
    for seq in test_sequences:
        gap_length = seq["masked_segment"].shape[0]
        pred = baseline.predict(seq["context_before"], seq["context_after"], gap_length)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    print(f"Loaded baseline predictions: {predictions.shape}")
    
    return predictions


def compare_all_models(seq2seq_metrics, seq2seq_predictions, ground_truths, 
                      bilstm_predictions, baseline_predictions, sensor_names):
    """Compare all three models."""
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    # Calculate metrics for all models
    seq2seq_mse = seq2seq_metrics["mse"]
    seq2seq_mae = seq2seq_metrics["mae"]
    
    bilstm_mse = np.mean((bilstm_predictions - ground_truths) ** 2)
    bilstm_mae = np.mean(np.abs(bilstm_predictions - ground_truths))
    
    baseline_mse = np.mean((baseline_predictions - ground_truths) ** 2)
    baseline_mae = np.mean(np.abs(baseline_predictions - ground_truths))
    
    print(f"\nOverall Metrics:")
    print(f"{'Model':<15} {'MSE':<15} {'MAE':<15} {'vs Baseline'}")
    print("-" * 60)
    print(f"{'Baseline':<15} {baseline_mse:<15.6f} {baseline_mae:<15.6f} 1.00x")
    print(f"{'Old BiLSTM':<15} {bilstm_mse:<15.6f} {bilstm_mae:<15.6f} {bilstm_mse/baseline_mse:.2f}x")
    print(f"{'Seq2Seq':<15} {seq2seq_mse:<15.6f} {seq2seq_mae:<15.6f} {seq2seq_mse/baseline_mse:.2f}x")
    
    # Per-sensor comparison
    print(f"\nPer-Sensor Comparison:")
    print(f"{'Sensor':<30} {'Baseline MSE':<15} {'BiLSTM MSE':<15} {'Seq2Seq MSE':<15}")
    print("-" * 75)
    
    per_sensor_comparison = {}
    for i, sensor_name in enumerate(sensor_names):
        baseline_sensor_mse = np.mean((baseline_predictions[:, :, i] - ground_truths[:, :, i]) ** 2)
        bilstm_sensor_mse = np.mean((bilstm_predictions[:, :, i] - ground_truths[:, :, i]) ** 2)
        seq2seq_sensor_mse = seq2seq_metrics["sensor_mse"][i]
        
        per_sensor_comparison[sensor_name] = {
            "baseline_mse": float(baseline_sensor_mse),
            "bilstm_mse": float(bilstm_sensor_mse),
            "seq2seq_mse": float(seq2seq_sensor_mse)
        }
        
        print(f"{sensor_name:<30} {baseline_sensor_mse:<15.6f} {bilstm_sensor_mse:<15.6f} {seq2seq_sensor_mse:<15.6f}")
    
    # Determine best model
    models = {
        "baseline": baseline_mse,
        "bilstm": bilstm_mse,
        "seq2seq": seq2seq_mse
    }
    best_model = min(models, key=models.get)
    
    print(f"\nBest Model: {best_model.upper()} (MSE: {models[best_model]:.6f})")
    
    comparison = {
        "overall": {
            "baseline": {"mse": float(baseline_mse), "mae": float(baseline_mae)},
            "bilstm": {"mse": float(bilstm_mse), "mae": float(bilstm_mae)},
            "seq2seq": {"mse": float(seq2seq_mse), "mae": float(seq2seq_mae)}
        },
        "per_sensor": per_sensor_comparison,
        "best_model": best_model,
        "seq2seq_prediction_variance": seq2seq_metrics["prediction_variance_mean"],
        "seq2seq_timestep_mse": seq2seq_metrics["timestep_mse"],
        "seq2seq_timestep_mae": seq2seq_metrics["timestep_mae"]
    }
    
    return comparison


def generate_comparison_plots(seq2seq_predictions, bilstm_predictions, 
                             baseline_predictions, ground_truths, 
                             test_sequences, sensor_names):
    """Generate comparison plots for 10+ examples."""
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)
    
    plots_dir = Path("results/test_plots/seq2seq")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    baseline = InterpolationBaseline()
    
    test_indices = [0, 100, 500, 1000, 2000, 3000, 5000, 7000, 10000, 15000]
    
    for idx in test_indices:
        if idx >= len(test_sequences):
            continue
        
        seq = test_sequences[idx]
        gt = ground_truths[idx]
        seq2seq_pred = seq2seq_predictions[idx]
        bilstm_pred = bilstm_predictions[idx]
        
        gap_length = seq["masked_segment"].shape[0]
        baseline_pred = baseline.predict(seq["context_before"], seq["context_after"], gap_length)
        
        # Plot first 3 sensors
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        for sensor_idx in range(3):
            sensor_name = sensor_names[sensor_idx]
            
            ctx_before = seq["context_before"][:, sensor_idx]
            ctx_after = seq["context_after"][:, sensor_idx]
            gt_sensor = gt[:, sensor_idx]
            seq2seq_sensor = seq2seq_pred[:, sensor_idx]
            bilstm_sensor = bilstm_pred[:, sensor_idx]
            baseline_sensor = baseline_pred[:, sensor_idx]
            
            t_before = np.arange(len(ctx_before))
            t_gap = np.arange(len(ctx_before), len(ctx_before) + len(gt_sensor))
            t_after = np.arange(len(ctx_before) + len(gt_sensor), 
                                len(ctx_before) + len(gt_sensor) + len(ctx_after))
            
            axes[sensor_idx].plot(t_before, ctx_before, 'b-', label='Context Before', linewidth=2, alpha=0.7)
            axes[sensor_idx].plot(t_gap, gt_sensor, 'g-', label='Original', linewidth=3)
            axes[sensor_idx].plot(t_gap, baseline_sensor, 'r--', label='Baseline', linewidth=2)
            axes[sensor_idx].plot(t_gap, bilstm_sensor, 'm:', label='Old BiLSTM', linewidth=2)
            axes[sensor_idx].plot(t_gap, seq2seq_sensor, 'c-.', label='Seq2Seq', linewidth=2)
            axes[sensor_idx].plot(t_after, ctx_after, 'b-', linewidth=2, alpha=0.7)
            axes[sensor_idx].axvspan(t_gap[0], t_gap[-1], alpha=0.15, color='gray')
            axes[sensor_idx].set_xlabel('Time Step')
            axes[sensor_idx].set_ylabel('Normalized Value')
            axes[sensor_idx].set_title(f'Example {idx} - {sensor_name}')
            axes[sensor_idx].legend()
            axes[sensor_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = plots_dir / f"comparison_example_{idx}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    print(f"Generated {len(test_indices)} comparison plots")


def save_all_artifacts(seq2seq_metrics, comparison, sensor_names):
    """Save all evaluation artifacts."""
    print("\n" + "="*70)
    print("SAVING ARTIFACTS")
    print("="*70)
    
    # Save seq2seq metrics
    with open("results/seq2seq_metrics.json", "w") as f:
        json.dump(seq2seq_metrics, f, indent=2)
    print("Saved: results/seq2seq_metrics.json")
    
    # Save per-sensor metrics
    with open("results/seq2seq_per_sensor_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sensor", "mse", "mae"])
        for i, sensor_name in enumerate(sensor_names):
            writer.writerow([
                sensor_name,
                seq2seq_metrics["sensor_mse"][i],
                seq2seq_metrics["sensor_mae"][i]
            ])
    print("Saved: results/seq2seq_per_sensor_metrics.csv")
    
    # Save comparison
    with open("results/seq2seq_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("Saved: results/seq2seq_comparison.json")


if __name__ == "__main__":
    # Load data
    loader = PHM2018Loader()
    preprocessed = loader.load_preprocessed(Path("data/preprocessed/phm2018_subset.pkl"))
    sensor_names = preprocessed["sensor_columns"]
    
    generator = MaskedSequenceGenerator()
    test_sequences = generator.load_sequences(Path("data/sequences/test_sequences.pkl"))
    
    # Evaluate seq2seq
    seq2seq_metrics, seq2seq_predictions, ground_truths = evaluate_seq2seq()
    
    # Load other models
    bilstm_predictions = load_old_bilstm_predictions(test_sequences)
    baseline_predictions = load_baseline_predictions(test_sequences)
    
    # Compare
    comparison = compare_all_models(seq2seq_metrics, seq2seq_predictions, ground_truths,
                                   bilstm_predictions, baseline_predictions, sensor_names)
    
    # Generate plots
    generate_comparison_plots(seq2seq_predictions, bilstm_predictions,
                            baseline_predictions, ground_truths,
                            test_sequences, sensor_names)
    
    # Save artifacts
    save_all_artifacts(seq2seq_metrics, comparison, sensor_names)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
