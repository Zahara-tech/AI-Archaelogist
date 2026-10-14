"""
Comprehensive Model Debugging and Error Analysis
Verifies BiLSTM implementation correctness and identifies performance issues.
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
from bilstm_model import BiLSTMReconstructor, BiLSTMModelTrainer, BiLSTMDataset
from baseline_model import InterpolationBaseline
from torch.utils.data import DataLoader


def check_bilstm_implementation():
    """Verify BiLSTM implementation correctness."""
    print("="*70)
    print("BILSTM IMPLEMENTATION VERIFICATION")
    print("="*70)
    
    # Load a single sequence to inspect
    loader = PHM2018Loader()
    preprocessed = loader.load_preprocessed(Path("data/preprocessed/phm2018_subset.pkl"))
    sensor_names = preprocessed["sensor_columns"]
    
    generator = MaskedSequenceGenerator()
    test_sequences = generator.load_sequences(Path("data/sequences/test_sequences.pkl"))
    
    seq = test_sequences[0]
    
    print(f"\nSequence Structure:")
    print(f"  Context before shape: {seq['context_before'].shape}")
    print(f"  Context after shape: {seq['context_after'].shape}")
    print(f"  Masked segment shape: {seq['masked_segment'].shape}")
    print(f"  Sensor names: {sensor_names}")
    
    # Check BiLSTM dataset
    dataset = BiLSTMDataset([seq])
    sample = dataset[0]
    
    print(f"\nBiLSTM Dataset Output:")
    print(f"  context_before shape: {sample['context_before'].shape}")
    print(f"  context_after shape: {sample['context_after'].shape}")
    print(f"  target shape: {sample['target'].shape}")
    
    # Verify sensor ordering
    print(f"\nSensor Ordering Check:")
    print(f"  Expected: {len(sensor_names)} sensors")
    print(f"  Context before: {seq['context_before'].shape[1]} sensors")
    print(f"  Context after: {seq['context_after'].shape[1]} sensors")
    print(f"  Target: {seq['masked_segment'].shape[1]} sensors")
    
    sensor_order_ok = (seq['context_before'].shape[1] == len(sensor_names) and
                      seq['context_after'].shape[1] == len(sensor_names) and
                      seq['masked_segment'].shape[1] == len(sensor_names))
    
    print(f"  Sensor ordering: {'CORRECT' if sensor_order_ok else 'INCORRECT'}")
    
    return sensor_order_ok


def check_context_usage():
    """Verify model uses both past and future context."""
    print("\n" + "="*70)
    print("CONTEXT USAGE VERIFICATION")
    print("="*70)
    
    # Load model
    loader = PHM2018Loader()
    preprocessed = loader.load_preprocessed(Path("data/preprocessed/phm2018_subset.pkl"))
    
    generator = MaskedSequenceGenerator()
    test_sequences = generator.load_sequences(Path("data/sequences/test_sequences.pkl"))
    
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
    
    # Test with only past context
    test_dataset = BiLSTMDataset([test_sequences[0]])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            context_before = batch["context_before"].to(trainer.device)
            context_after = batch["context_after"].to(trainer.device)
            
            print(f"\nInput Shapes:")
            print(f"  context_before: {context_before.shape}")
            print(f"  context_after: {context_after.shape}")
            
            # Check if both contexts are used
            forward_out, (h_f, c_f) = model.lstm_forward(context_before)
            context_after_reversed = torch.flip(context_after, dims=[1])
            backward_out, (h_b, c_b) = model.lstm_backward(context_after_reversed)
            
            print(f"\nLSTM Outputs:")
            print(f"  Forward LSTM output shape: {forward_out.shape}")
            print(f"  Backward LSTM output shape: {backward_out.shape}")
            print(f"  Forward final hidden shape: {h_f.shape}")
            print(f"  Backward final hidden shape: {h_b.shape}")
            
            # Check combined representation
            forward_hidden = h_f[-1]
            backward_hidden = h_b[-1]
            combined = torch.cat([forward_hidden, backward_hidden], dim=1)
            
            print(f"\nCombined Representation:")
            print(f"  Forward hidden: {forward_hidden.shape}")
            print(f"  Backward hidden: {backward_hidden.shape}")
            print(f"  Combined: {combined.shape}")
            
            both_contexts_used = (combined.shape[1] == 128)  # 64 + 64
            print(f"\nBoth contexts used: {'YES' if both_contexts_used else 'NO'}")
    
    return both_contexts_used


def check_constant_predictions():
    """Check if model predicts constant values."""
    print("\n" + "="*70)
    print("CONSTANT PREDICTION CHECK")
    print("="*70)
    
    loader = PHM2018Loader()
    preprocessed = loader.load_preprocessed(Path("data/preprocessed/phm2018_subset.pkl"))
    sensor_names = preprocessed["sensor_columns"]
    
    generator = MaskedSequenceGenerator()
    test_sequences = generator.load_sequences(Path("data/sequences/test_sequences.pkl"))
    
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
    
    # Get predictions for multiple sequences
    test_dataset = BiLSTMDataset(test_sequences[:100])
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
    
    # Check variance across timesteps for each sensor
    print(f"\nPrediction Variance Across Timesteps (first 100 sequences):")
    print(f"{'Sensor':<30} {'Mean Variance':<15} {'Status'}")
    print("-" * 55)
    
    constant_predictions = []
    for i, sensor_name in enumerate(sensor_names):
        # Variance across timesteps for each sequence, then average
        variances = np.var(predictions[:, :, i], axis=1)
        mean_variance = np.mean(variances)
        
        status = "CONSTANT" if mean_variance < 0.001 else "VARYING"
        print(f"{sensor_name:<30} {mean_variance:<15.6f} {status}")
        
        if mean_variance < 0.001:
            constant_predictions.append(sensor_name)
    
    if constant_predictions:
        print(f"\nWARNING: Model predicts constant values for: {constant_predictions}")
    else:
        print(f"\nOK: Model predictions vary across timesteps")
    
    return constant_predictions


def compare_distributions():
    """Compare BiLSTM output distribution vs target distribution."""
    print("\n" + "="*70)
    print("DISTRIBUTION COMPARISON")
    print("="*70)
    
    loader = PHM2018Loader()
    preprocessed = loader.load_preprocessed(Path("data/preprocessed/phm2018_subset.pkl"))
    sensor_names = preprocessed["sensor_columns"]
    
    generator = MaskedSequenceGenerator()
    test_sequences = generator.load_sequences(Path("data/sequences/test_sequences.pkl"))
    
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
    
    # Get predictions for entire test set
    test_dataset = BiLSTMDataset(test_sequences)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model.eval()
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for batch in test_loader:
            context_before = batch["context_before"].to(trainer.device)
            context_after = batch["context_after"].to(trainer.device)
            target = batch["target"].to(trainer.device)
            
            pred = model(context_before, context_after)
            predictions.append(pred.cpu().numpy())
            ground_truths.append(target.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)
    
    print(f"\nPer-Sensor Distribution Statistics:")
    print(f"{'Sensor':<30} {'GT Mean':<12} {'GT Std':<12} {'Pred Mean':<12} {'Pred Std':<12}")
    print("-" * 80)
    
    distribution_stats = {}
    for i, sensor_name in enumerate(sensor_names):
        gt_mean = np.mean(ground_truths[:, :, i])
        gt_std = np.std(ground_truths[:, :, i])
        pred_mean = np.mean(predictions[:, :, i])
        pred_std = np.std(predictions[:, :, i])
        
        distribution_stats[sensor_name] = {
            "gt_mean": float(gt_mean),
            "gt_std": float(gt_std),
            "pred_mean": float(pred_mean),
            "pred_std": float(pred_std)
        }
        
        print(f"{sensor_name:<30} {gt_mean:<12.6f} {gt_std:<12.6f} {pred_mean:<12.6f} {pred_std:<12.6f}")
    
    return distribution_stats


def full_test_set_metrics():
    """Calculate per-sensor MAE/MSE across entire test set."""
    print("\n" + "="*70)
    print("FULL TEST SET METRICS")
    print("="*70)
    
    loader = PHM2018Loader()
    preprocessed = loader.load_preprocessed(Path("data/preprocessed/phm2018_subset.pkl"))
    sensor_names = preprocessed["sensor_columns"]
    
    generator = MaskedSequenceGenerator()
    test_sequences = generator.load_sequences(Path("data/sequences/test_sequences.pkl"))
    
    with open("data/sequences/config.json", "r") as f:
        config = json.load(f)
    
    sample = test_sequences[0]
    input_dim = sample["context_before"].shape[1]
    output_dim = input_dim
    output_length = config["gap_length"]
    
    # BiLSTM predictions
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
    bilstm_predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for batch in test_loader:
            context_before = batch["context_before"].to(trainer.device)
            context_after = batch["context_after"].to(trainer.device)
            target = batch["target"].to(trainer.device)
            
            pred = model(context_before, context_after)
            bilstm_predictions.append(pred.cpu().numpy())
            ground_truths.append(target.cpu().numpy())
    
    bilstm_predictions = np.concatenate(bilstm_predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)
    
    # Baseline predictions
    baseline = InterpolationBaseline()
    baseline_predictions = []
    for seq in test_sequences:
        gap_length = seq["masked_segment"].shape[0]
        pred = baseline.predict(seq["context_before"], seq["context_after"], gap_length)
        baseline_predictions.append(pred)
    
    baseline_predictions = np.array(baseline_predictions)
    
    # Calculate metrics
    print(f"\nPer-Sensor Metrics (Full Test Set - {len(test_sequences)} sequences):")
    print(f"{'Sensor':<30} {'BiLSTM MAE':<12} {'BiLSTM MSE':<12} {'Baseline MAE':<12} {'Baseline MSE':<12} {'Ratio'}")
    print("-" * 100)
    
    full_metrics = {}
    for i, sensor_name in enumerate(sensor_names):
        bilstm_mae = np.mean(np.abs(bilstm_predictions[:, :, i] - ground_truths[:, :, i]))
        bilstm_mse = np.mean((bilstm_predictions[:, :, i] - ground_truths[:, :, i]) ** 2)
        baseline_mae = np.mean(np.abs(baseline_predictions[:, :, i] - ground_truths[:, :, i]))
        baseline_mse = np.mean((baseline_predictions[:, :, i] - ground_truths[:, :, i]) ** 2)
        
        ratio = bilstm_mse / baseline_mse if baseline_mse > 0 else float('inf')
        
        full_metrics[sensor_name] = {
            "bilstm_mae": float(bilstm_mae),
            "bilstm_mse": float(bilstm_mse),
            "baseline_mae": float(baseline_mae),
            "baseline_mse": float(baseline_mse),
            "mse_ratio": float(ratio)
        }
        
        print(f"{sensor_name:<30} {bilstm_mae:<12.6f} {bilstm_mse:<12.6f} {baseline_mae:<12.6f} {baseline_mse:<12.6f} {ratio:.2f}x")
    
    # Overall metrics
    bilstm_overall_mae = np.mean(np.abs(bilstm_predictions - ground_truths))
    bilstm_overall_mse = np.mean((bilstm_predictions - ground_truths) ** 2)
    baseline_overall_mae = np.mean(np.abs(baseline_predictions - ground_truths))
    baseline_overall_mse = np.mean((baseline_predictions - ground_truths) ** 2)
    
    print(f"\n{'OVERALL':<30} {bilstm_overall_mae:<12.6f} {bilstm_overall_mse:<12.6f} {baseline_overall_mae:<12.6f} {baseline_overall_mse:<12.6f} {bilstm_overall_mse/baseline_overall_mse:.2f}x")
    
    return full_metrics, bilstm_predictions, baseline_predictions, ground_truths


def generate_comparison_plots(bilstm_predictions, ground_truths, test_sequences, sensor_names):
    """Generate comparison plots for 10+ examples."""
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)
    
    plots_dir = Path("results/test_plots/comparison")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create baseline for gap calculation
    baseline = InterpolationBaseline()
    
    # Select 10 diverse examples
    test_indices = [0, 100, 500, 1000, 2000, 3000, 5000, 7000, 10000, 15000]
    
    for idx in test_indices:
        if idx >= len(test_sequences):
            continue
        
        seq = test_sequences[idx]
        gt = ground_truths[idx]
        bilstm_pred = bilstm_predictions[idx]
        
        # Get baseline prediction
        gap_length = seq["masked_segment"].shape[0]
        baseline_pred = baseline.predict(seq["context_before"], seq["context_after"], gap_length)
        
        # Plot first 3 sensors
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        for sensor_idx in range(3):
            sensor_name = sensor_names[sensor_idx]
            
            ctx_before = seq["context_before"][:, sensor_idx]
            ctx_after = seq["context_after"][:, sensor_idx]
            gt_sensor = gt[:, sensor_idx]
            bilstm_sensor = bilstm_pred[:, sensor_idx]
            baseline_sensor = baseline_pred[:, sensor_idx]
            
            t_before = np.arange(len(ctx_before))
            t_gap = np.arange(len(ctx_before), len(ctx_before) + len(gt_sensor))
            t_after = np.arange(len(ctx_before) + len(gt_sensor), 
                                len(ctx_before) + len(gt_sensor) + len(ctx_after))
            
            axes[sensor_idx].plot(t_before, ctx_before, 'b-', label='Context Before', linewidth=2, alpha=0.7)
            axes[sensor_idx].plot(t_gap, gt_sensor, 'g-', label='Original', linewidth=3)
            axes[sensor_idx].plot(t_gap, baseline_sensor, 'r--', label='Baseline', linewidth=2)
            axes[sensor_idx].plot(t_gap, bilstm_sensor, 'm:', label='BiLSTM', linewidth=2)
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


def save_debug_report(sensor_order_ok, both_contexts_used, constant_predictions, 
                      distribution_stats, full_metrics):
    """Save comprehensive debug report."""
    
    report = {
        "debug_date": "2026-10-06",
        "implementation_checks": {
            "sensor_ordering_correct": sensor_order_ok,
            "both_contexts_used": both_contexts_used,
            "constant_predictions_sensors": constant_predictions
        },
        "distribution_comparison": distribution_stats,
        "full_test_set_metrics": full_metrics,
        "overall_bilstm_mse": full_metrics["IONGAUGEPRESSURE"]["bilstm_mse"],  # Will be overwritten
        "overall_baseline_mse": full_metrics["IONGAUGEPRESSURE"]["baseline_mse"]  # Will be overwritten
    }
    
    # Calculate overall metrics
    bilstm_mses = [m["bilstm_mse"] for m in full_metrics.values()]
    baseline_mses = [m["baseline_mse"] for m in full_metrics.values()]
    
    report["overall_bilstm_mse"] = np.mean(bilstm_mses)
    report["overall_baseline_mse"] = np.mean(baseline_mses)
    report["overall_worse_by"] = report["overall_bilstm_mse"] / report["overall_baseline_mse"]
    
    with open("results/model_debug_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Save CSV
    with open("results/per_sensor_full_test_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sensor", "bilstm_mae", "bilstm_mse", "baseline_mae", "baseline_mse", "mse_ratio"])
        for sensor, metrics in full_metrics.items():
            writer.writerow([
                sensor,
                metrics["bilstm_mae"],
                metrics["bilstm_mse"],
                metrics["baseline_mae"],
                metrics["baseline_mse"],
                metrics["mse_ratio"]
            ])
    
    print(f"\nSaved debug report to results/model_debug_report.json")
    print(f"Saved metrics to results/per_sensor_full_test_metrics.csv")


if __name__ == "__main__":
    # Run all checks
    sensor_order_ok = check_bilstm_implementation()
    both_contexts_used = check_context_usage()
    constant_predictions = check_constant_predictions()
    distribution_stats = compare_distributions()
    full_metrics, bilstm_predictions, baseline_predictions, ground_truths = full_test_set_metrics()
    
    # Generate plots
    loader = PHM2018Loader()
    preprocessed = loader.load_preprocessed(Path("data/preprocessed/phm2018_subset.pkl"))
    sensor_names = preprocessed["sensor_columns"]
    generator = MaskedSequenceGenerator()
    test_sequences = generator.load_sequences(Path("data/sequences/test_sequences.pkl"))
    
    generate_comparison_plots(bilstm_predictions, ground_truths, 
                             test_sequences, sensor_names)
    
    # Save report
    save_debug_report(sensor_order_ok, both_contexts_used, constant_predictions,
                      distribution_stats, full_metrics)
    
    print("\n" + "="*70)
    print("DEBUG ANALYSIS COMPLETE")
    print("="*70)
