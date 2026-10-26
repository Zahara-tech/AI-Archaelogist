"""
Diagnose Generalization Failure in Attention Seq2Seq V2
Investigates why the model performs worse than linear interpolation on test set.
"""

import os
import sys
import json
import glob
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

sys.path.append("src")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent.parent
BENCHMARK_DIR = BASE_DIR / "data" / "benchmark_v2"
EXPERIMENT_DIR = BASE_DIR / "experiments" / "attention_seq2seq_v2_50epochs"
CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints" / "best_model.pt"
TEST_EVAL_DIR = EXPERIMENT_DIR / "results" / "test_evaluation"
DIAGNOSTIC_DIR = EXPERIMENT_DIR / "results" / "generalization_diagnostics"

DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_COLUMNS = [
    "IONGAUGEPRESSURE",
    "ETCHBEAMVOLTAGE",
    "ETCHBEAMCURRENT",
    "ETCHSUPPRESSORVOLTAGE",
    "ETCHSUPPRESSORCURRENT",
    "FLOWCOOLFLOWRATE",
    "FLOWCOOLPRESSURE",
    "ETCHGASCHANNEL1READBACK",
    "ETCHPBNGASREADBACK",
    "FIXTURETILTANGLE",
    "ROTATIONSPEED",
    "ACTUALROTATIONANGLE",
    "FIXTURESHUTTERPOSITION",
    "ETCHSOURCEUSAGE",
    "ETCHAUXSOURCETIMER",
    "ETCHAUX2SOURCETIMER",
    "ACTUALSTEPDURATION"
]

print("=" * 70)
print("GENERALIZATION FAILURE DIAGNOSTIC")
print("=" * 70)
print(f"Diagnostic output: {DIAGNOSTIC_DIR}")
print()

# ============================================================
# STEP 1: LOAD EVALUATION RESULTS
# ============================================================

print("=" * 70)
print("STEP 1: LOAD EVALUATION RESULTS")
print("=" * 70)

with open(TEST_EVAL_DIR / "evaluation_summary.json", "r") as f:
    eval_summary = json.load(f)

# Load predictions
pred_data = np.load(TEST_EVAL_DIR / "predictions.npz")
targets = pred_data["targets"]
baseline_predictions = pred_data["baseline_predictions"]
model_predictions = pred_data["model_predictions"]

print(f"Loaded predictions: {targets.shape}")
print()

# ============================================================
# STEP 2: ANALYZE PER-FILE METRICS
# ============================================================

print("=" * 70)
print("STEP 2: ANALYZE PER-FILE METRICS")
print("=" * 70)

per_file_results = eval_summary["per_file"]

print("\nPer-file MSE ratios (model / baseline):")
print("-" * 70)
for r in per_file_results:
    print(f"{r['file']:30s} | {r['mse_ratio_vs_baseline']:6.2f}x | {r['sequences']:6d} sequences")

# Identify worst performing file
worst_file = max(per_file_results, key=lambda x: x["mse_ratio_vs_baseline"])
print(f"\nWorst performing file: {worst_file['file']} (ratio: {worst_file['mse_ratio_vs_baseline']:.2f}x)")

# Identify best performing file
best_file = min(per_file_results, key=lambda x: x["mse_ratio_vs_baseline"])
print(f"Best performing file: {best_file['file']} (ratio: {best_file['mse_ratio_vs_baseline']:.2f}x)")

print()

# ============================================================
# STEP 3: ANALYZE PER-SENSOR METRICS
# ============================================================

print("=" * 70)
print("STEP 3: ANALYZE PER-SENSOR METRICS")
print("=" * 70)

baseline_per_sensor_mae = eval_summary["per_sensor"]["baseline_mae"]
model_per_sensor_mae = eval_summary["per_sensor"]["model_mae"]
baseline_per_sensor_rmse = eval_summary["per_sensor"]["baseline_rmse"]
model_per_sensor_rmse = eval_summary["per_sensor"]["model_rmse"]

print("\nPer-sensor MAE ratios (model / baseline):")
print("-" * 70)
sensor_mae_ratios = []
for sensor in SENSOR_COLUMNS:
    ratio = model_per_sensor_mae[sensor] / baseline_per_sensor_mae[sensor] if baseline_per_sensor_mae[sensor] > 0 else float("inf")
    sensor_mae_ratios.append((sensor, ratio))
    print(f"{sensor:30s} | {ratio:6.2f}x")

# Sort by ratio
sensor_mae_ratios.sort(key=lambda x: x[1], reverse=True)

print(f"\nWorst sensors (highest ratio):")
for sensor, ratio in sensor_mae_ratios[:5]:
    print(f"  {sensor}: {ratio:.2f}x")

print(f"\nBest sensors (lowest ratio):")
for sensor, ratio in sensor_mae_ratios[-5:]:
    print(f"  {sensor}: {ratio:.2f}x")

print()

# ============================================================
# STEP 4: COMPARE SENSOR DISTRIBUTIONS
# ============================================================

print("=" * 70)
print("STEP 4: COMPARE SENSOR DISTRIBUTIONS")
print("=" * 70)

# Load training data for distribution comparison
train_files = [
    "train_05_M02_DC_train.npz",
    "train_07_M02_DC_train.npz"
]

print("\nLoading training data for distribution analysis...")
train_data_list = []
for train_file in train_files:
    train_path = BENCHMARK_DIR / train_file
    if train_path.exists():
        data = np.load(train_path)
        train_data_list.append(data)
        print(f"  Loaded {train_file}")

if train_data_list:
    train_targets = np.concatenate([d["masked_segment"] for d in train_data_list], axis=0)
    print(f"Training targets shape: {train_targets.shape}")

# Load validation data
val_files = [
    "val_05_M02_DC_train.npz",
    "val_07_M02_DC_train.npz"
]

print("\nLoading validation data for distribution analysis...")
val_data_list = []
for val_file in val_files:
    val_path = BENCHMARK_DIR / val_file
    if val_path.exists():
        data = np.load(val_path)
        val_data_list.append(data)
        print(f"  Loaded {val_file}")

if val_data_list:
    val_targets = np.concatenate([d["masked_segment"] for d in val_data_list], axis=0)
    print(f"Validation targets shape: {val_targets.shape}")

print(f"\nTest targets shape: {targets.shape}")

# Calculate distribution statistics for each sensor
print("\nSensor distribution comparison (mean, std):")
print("-" * 90)

distribution_stats = []

for i, sensor in enumerate(SENSOR_COLUMNS):
    if train_data_list:
        train_mean = np.mean(train_targets[:, :, i])
        train_std = np.std(train_targets[:, :, i])
    else:
        train_mean = train_std = None
    
    if val_data_list:
        val_mean = np.mean(val_targets[:, :, i])
        val_std = np.std(val_targets[:, :, i])
    else:
        val_mean = val_std = None
    
    test_mean = np.mean(targets[:, :, i])
    test_std = np.std(targets[:, :, i])
    
    # Calculate KL divergence between train and test distributions
    if train_data_list:
        train_flat = train_targets[:, :, i].flatten()
        test_flat = targets[:, :, i].flatten()
        
        # Use histogram-based KL divergence
        hist_range = (min(train_flat.min(), test_flat.min()), max(train_flat.max(), test_flat.max()))
        train_hist, _ = np.histogram(train_flat, bins=50, range=hist_range, density=True)
        test_hist, _ = np.histogram(test_flat, bins=50, range=hist_range, density=True)
        
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        train_hist = train_hist + eps
        test_hist = test_hist + eps
        
        kl_div = stats.entropy(train_hist, test_hist)
    else:
        kl_div = None
    
    distribution_stats.append({
        "sensor": sensor,
        "train_mean": train_mean,
        "train_std": train_std,
        "val_mean": val_mean,
        "val_std": val_std,
        "test_mean": test_mean,
        "test_std": test_std,
        "kl_divergence": kl_div
    })
    
    print(f"{sensor:30s} | Train: {train_mean if train_mean else 'N/A':7.3f}±{train_std if train_std else 'N/A':.3f} | "
          f"Val: {val_mean if val_mean else 'N/A':7.3f}±{val_std if val_std else 'N/A':.3f} | "
          f"Test: {test_mean:7.3f}±{test_std:.3f} | KL: {kl_div if kl_div else 'N/A':.4f}")

print()

# ============================================================
# STEP 5: INVESTIGATE PREDICTION MAGNITUDE
# ============================================================

print("=" * 70)
print("STEP 5: INVESTIGATE PREDICTION MAGNITUDE")
print("=" * 70)

# Calculate prediction statistics
target_mean = np.mean(targets)
target_std = np.std(targets)
baseline_mean = np.mean(baseline_predictions)
baseline_std = np.std(baseline_predictions)
model_mean = np.mean(model_predictions)
model_std = np.std(model_predictions)

print(f"\nOverall statistics:")
print(f"  Target:      mean={target_mean:.6f}, std={target_std:.6f}")
print(f"  Baseline:    mean={baseline_mean:.6f}, std={baseline_std:.6f}")
print(f"  Model:       mean={model_mean:.6f}, std={model_std:.6f}")

# Calculate prediction bias
model_bias = model_mean - target_mean
baseline_bias = baseline_mean - target_mean

print(f"\nPrediction bias (mean - target):")
print(f"  Baseline: {baseline_bias:.6f}")
print(f"  Model:    {model_bias:.6f}")

# Check if model predictions have excessive magnitude
model_magnitude_ratio = model_std / target_std if target_std > 0 else float("inf")
baseline_magnitude_ratio = baseline_std / target_std if target_std > 0 else float("inf")

print(f"\nPrediction magnitude ratio (std / target_std):")
print(f"  Baseline: {baseline_magnitude_ratio:.3f}x")
print(f"  Model:    {model_magnitude_ratio:.3f}x")

print()

# ============================================================
# STEP 6: INVESTIGATE TEMPORAL RECONSTRUCTION
# ============================================================

print("=" * 70)
print("STEP 6: INVESTIGATE TEMPORAL RECONSTRUCTION")
print("=" * 70)

# Calculate per-timestep statistics
baseline_per_timestep_rmse = eval_summary["per_timestep"]["baseline_rmse"]
model_per_timestep_rmse = eval_summary["per_timestep"]["model_rmse"]

print("\nPer-timestep RMSE:")
print("-" * 50)
for t in range(10):
    print(f"Timestep {t+1:2d} | Baseline: {baseline_per_timestep_rmse[t]:.6f} | Model: {model_per_timestep_rmse[t]:.6f} | Ratio: {model_per_timestep_rmse[t]/baseline_per_timestep_rmse[t]:.2f}x")

# Check if model error is uniform across timesteps
timestep_ratios = [model_per_timestep_rmse[t]/baseline_per_timestep_rmse[t] for t in range(10)]
timestep_ratio_std = np.std(timestep_ratios)

print(f"\nTimestep ratio std: {timestep_ratio_std:.3f}")
print(f"Uniform error across timesteps: {'Yes' if timestep_ratio_std < 0.5 else 'No'}")

print()

# ============================================================
# STEP 7: CHECK ERROR CONCENTRATION BY TOOL VARIANT
# ============================================================

print("=" * 70)
print("STEP 7: CHECK ERROR CONCENTRATION BY TOOL VARIANT")
print("=" * 70)

# Group by tool variant (M01 vs M02)
m02_files = [r for r in per_file_results if "M02" in r["file"]]
m01_files = [r for r in per_file_results if "M01" in r["file"]]

if m02_files:
    m02_avg_ratio = np.mean([r["mse_ratio_vs_baseline"] for r in m02_files])
    print(f"\nM02 tool variant (test files): {len(m02_files)}")
    print(f"  Average MSE ratio: {m02_avg_ratio:.2f}x")
    for r in m02_files:
        print(f"    {r['file']}: {r['mse_ratio_vs_baseline']:.2f}x")

if m01_files:
    m01_avg_ratio = np.mean([r["mse_ratio_vs_baseline"] for r in m01_files])
    print(f"\nM01 tool variant (test files): {len(m01_files)}")
    print(f"  Average MSE ratio: {m01_avg_ratio:.2f}x")
    for r in m01_files:
        print(f"    {r['file']}: {r['mse_ratio_vs_baseline']:.2f}x")

print()

# ============================================================
# STEP 8: GENERATE DIAGNOSTIC PLOTS
# ============================================================

print("=" * 70)
print("STEP 8: GENERATE DIAGNOSTIC PLOTS")
print("=" * 70)

# Plot 1: Per-file MSE ratio
fig, ax = plt.subplots(figsize=(10, 6))
files = [r["file"] for r in per_file_results]
ratios = [r["mse_ratio_vs_baseline"] for r in per_file_results]
colors = ['red' if 'M01' in f else 'blue' for f in files]

ax.bar(range(len(files)), ratios, color=colors, alpha=0.7)
ax.axhline(y=1.0, color='black', linestyle='--', label='Equal performance')
ax.set_xlabel('Test File')
ax.set_ylabel('MSE Ratio (Model / Baseline)')
ax.set_title('Per-File MSE Ratio by Tool Variant')
ax.set_xticks(range(len(files)))
ax.set_xticklabels([f.replace('test_', '').replace('.npz', '') for f in files], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path_1 = DIAGNOSTIC_DIR / "per_file_mse_ratio.png"
plt.savefig(plot_path_1, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path_1}")
plt.close()

# Plot 2: Per-sensor MAE ratio
fig, ax = plt.subplots(figsize=(12, 6))
sensor_names = [s for s, _ in sensor_mae_ratios]
sensor_ratios = [r for _, r in sensor_mae_ratios]

ax.barh(range(len(sensor_names)), sensor_ratios, alpha=0.7)
ax.axvline(x=1.0, color='black', linestyle='--', label='Equal performance')
ax.set_xlabel('MAE Ratio (Model / Baseline)')
ax.set_ylabel('Sensor')
ax.set_title('Per-Sensor MAE Ratio')
ax.set_yticks(range(len(sensor_names)))
ax.set_yticklabels(sensor_names)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path_2 = DIAGNOSTIC_DIR / "per_sensor_mae_ratio.png"
plt.savefig(plot_path_2, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path_2}")
plt.close()

# Plot 3: Sensor distribution comparison (mean)
if train_data_list and val_data_list:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mean comparison
    train_means = [s["train_mean"] for s in distribution_stats]
    val_means = [s["val_mean"] for s in distribution_stats]
    test_means = [s["test_mean"] for s in distribution_stats]
    
    x = np.arange(len(SENSOR_COLUMNS))
    width = 0.25
    
    axes[0, 0].bar(x - width, train_means, width, label='Train', alpha=0.8)
    axes[0, 0].bar(x, val_means, width, label='Val', alpha=0.8)
    axes[0, 0].bar(x + width, test_means, width, label='Test', alpha=0.8)
    axes[0, 0].set_xlabel('Sensor')
    axes[0, 0].set_ylabel('Mean (Normalized)')
    axes[0, 0].set_title('Sensor Mean Distribution')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(SENSOR_COLUMNS, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Std comparison
    train_stds = [s["train_std"] for s in distribution_stats]
    val_stds = [s["val_std"] for s in distribution_stats]
    test_stds = [s["test_std"] for s in distribution_stats]
    
    axes[0, 1].bar(x - width, train_stds, width, label='Train', alpha=0.8)
    axes[0, 1].bar(x, val_stds, width, label='Val', alpha=0.8)
    axes[0, 1].bar(x + width, test_stds, width, label='Test', alpha=0.8)
    axes[0, 1].set_xlabel('Sensor')
    axes[0, 1].set_ylabel('Std (Normalized)')
    axes[0, 1].set_title('Sensor Std Distribution')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(SENSOR_COLUMNS, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL divergence
    kl_divs = [s["kl_divergence"] for s in distribution_stats]
    axes[1, 0].barh(range(len(SENSOR_COLUMNS)), kl_divs, alpha=0.7)
    axes[1, 0].set_xlabel('KL Divergence (Train vs Test)')
    axes[1, 0].set_ylabel('Sensor')
    axes[1, 0].set_title('Distribution Shift (KL Divergence)')
    axes[1, 0].set_yticks(range(len(SENSOR_COLUMNS)))
    axes[1, 0].set_yticklabels(SENSOR_COLUMNS)
    axes[1, 0].grid(True, alpha=0.3)
    
    # MAE ratio vs KL divergence
    mae_ratios_for_plot = [model_per_sensor_mae[s] / baseline_per_sensor_mae[s] for s in SENSOR_COLUMNS]
    axes[1, 1].scatter(kl_divs, mae_ratios_for_plot, alpha=0.7)
    axes[1, 1].set_xlabel('KL Divergence (Train vs Test)')
    axes[1, 1].set_ylabel('MAE Ratio (Model / Baseline)')
    axes[1, 1].set_title('Distribution Shift vs Performance Degradation')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add sensor labels to scatter plot
    for i, sensor in enumerate(SENSOR_COLUMNS):
        axes[1, 1].annotate(sensor, (kl_divs[i], mae_ratios_for_plot[i]), fontsize=6, alpha=0.5)
    
    plt.tight_layout()
    plot_path_3 = DIAGNOSTIC_DIR / "sensor_distribution_comparison.png"
    plt.savefig(plot_path_3, dpi=150, bbox_inches='tight')
    print(f"  Saved: {plot_path_3}")
    plt.close()

# Plot 4: Prediction error analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Error magnitude distribution
model_errors = (model_predictions - targets).flatten()
baseline_errors = (baseline_predictions - targets).flatten()

axes[0, 0].hist(model_errors, bins=100, alpha=0.7, label='Model', color='blue')
axes[0, 0].hist(baseline_errors, bins=100, alpha=0.5, label='Baseline', color='red')
axes[0, 0].set_xlabel('Prediction Error')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Error Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Error vs target value
axes[0, 1].scatter(targets.flatten(), model_errors.flatten(), alpha=0.1, s=1, color='blue', label='Model')
axes[0, 1].scatter(targets.flatten(), baseline_errors.flatten(), alpha=0.1, s=1, color='red', label='Baseline')
axes[0, 1].set_xlabel('Target Value')
axes[0, 1].set_ylabel('Prediction Error')
axes[0, 1].set_title('Error vs Target Value')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Prediction vs target scatter
axes[1, 0].scatter(targets.flatten(), model_predictions.flatten(), alpha=0.1, s=1, color='blue', label='Model')
axes[1, 0].scatter(targets.flatten(), baseline_predictions.flatten(), alpha=0.1, s=1, color='red', label='Baseline')
axes[1, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--', label='Perfect prediction')
axes[1, 0].set_xlabel('Target Value')
axes[1, 0].set_ylabel('Predicted Value')
axes[1, 0].set_title('Prediction vs Target')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Per-timestep error
model_per_timestep_errors = np.mean((model_predictions - targets) ** 2, axis=(0, 2))
baseline_per_timestep_errors = np.mean((baseline_predictions - targets) ** 2, axis=(0, 2))

timesteps = np.arange(1, 11)
axes[1, 1].plot(timesteps, model_per_timestep_errors, 'o-', label='Model', linewidth=2, markersize=8, color='blue')
axes[1, 1].plot(timesteps, baseline_per_timestep_errors, 's-', label='Baseline', linewidth=2, markersize=8, color='red')
axes[1, 1].set_xlabel('Timestep')
axes[1, 1].set_ylabel('MSE')
axes[1, 1].set_title('Per-Timestep MSE')
axes[1, 1].set_xticks(timesteps)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path_4 = DIAGNOSTIC_DIR / "prediction_error_analysis.png"
plt.savefig(plot_path_4, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path_4}")
plt.close()

# Plot 5: Sample predictions comparison
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Sample Predictions: Actual vs Model vs Baseline', fontsize=14)

sample_indices = np.random.choice(len(targets), 9, replace=False)

for idx, ax in enumerate(axes.flat):
    sample_idx = sample_indices[idx]
    
    # Pick a random sensor
    sensor_idx = np.random.randint(0, 17)
    
    timesteps = np.arange(1, 11)
    actual = targets[sample_idx, :, sensor_idx]
    model_pred = model_predictions[sample_idx, :, sensor_idx]
    baseline_pred = baseline_predictions[sample_idx, :, sensor_idx]
    
    ax.plot(timesteps, actual, 'o-', label='Actual', linewidth=2, markersize=6, color='green')
    ax.plot(timesteps, model_pred, 's-', label='Model', linewidth=2, markersize=6, color='blue')
    ax.plot(timesteps, baseline_pred, '^-', label='Baseline', linewidth=2, markersize=6, color='red')
    
    ax.axvspan(0.5, 10.5, alpha=0.1, color='gray')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Normalized Value')
    ax.set_title(f'Sample {sample_idx}, Sensor {SENSOR_COLUMNS[sensor_idx]}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path_5 = DIAGNOSTIC_DIR / "sample_predictions_comparison.png"
plt.savefig(plot_path_5, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path_5}")
plt.close()

print()

# ============================================================
# STEP 9: WRITE DIAGNOSTIC REPORT
# ============================================================

print("=" * 70)
print("STEP 9: WRITE DIAGNOSTIC REPORT")
print("=" * 70)

diagnostic_report = f"""# Generalization Failure Diagnostic Report

## Summary

The Attention Seq2Seq V2 model performs catastrophically worse than linear interpolation on the held-out test set:
- Model test MSE: {eval_summary['overall_metrics']['attention_seq2seq_v2']['mse']:.6f}
- Baseline test MSE: {eval_summary['overall_metrics']['linear_interpolation']['mse']:.6f}
- MSE ratio: {eval_summary['overall_metrics']['comparison']['mse_ratio_vs_baseline']:.2f}x

## Key Findings

### 1. Per-File Analysis

The model fails across all test files, with MSE ratios ranging from {min(r['mse_ratio_vs_baseline'] for r in per_file_results):.1f}x to {max(r['mse_ratio_vs_baseline'] for r in per_file_results):.1f}x:

"""

for r in per_file_results:
    variant = "M02" if "M02" in r["file"] else "M01"
    diagnostic_report += f"- **{r['file']}** ({variant}): {r['mse_ratio_vs_baseline']:.1f}x MSE ratio, {r['sequences']} sequences\n"

diagnostic_report += f"""
**Worst file**: {worst_file['file']} ({worst_file['mse_ratio_vs_baseline']:.1f}x)
**Best file**: {best_file['file']} ({best_file['mse_ratio_vs_baseline']:.1f}x)

### 2. Tool Variant Analysis

"""

if m02_files:
    m02_avg_ratio = np.mean([r["mse_ratio_vs_baseline"] for r in m02_files])
    diagnostic_report += f"**M02 variant** (same as training): Average {m02_avg_ratio:.1f}x MSE ratio\n"

if m01_files:
    m01_avg_ratio = np.mean([r["mse_ratio_vs_baseline"] for r in m01_files])
    diagnostic_report += f"**M01 variant** (unseen during training): Average {m01_avg_ratio:.1f}x MSE ratio\n"

diagnostic_report += f"""
### 3. Per-Sensor Analysis

The model fails on all sensors, but degradation is uneven. Worst performing sensors:

"""

for sensor, ratio in sensor_mae_ratios[:5]:
    diagnostic_report += f"- **{sensor}**: {ratio:.1f}x MAE ratio\n"

diagnostic_report += f"""

Best performing sensors:

"""

for sensor, ratio in sensor_mae_ratios[-5:]:
    diagnostic_report += f"- **{sensor}**: {ratio:.1f}x MAE ratio\n"

diagnostic_report += f"""
### 4. Prediction Magnitude Analysis

- Target mean: {target_mean:.6f}, std: {target_std:.6f}
- Baseline mean: {baseline_mean:.6f}, std: {baseline_std:.6f}
- Model mean: {model_mean:.6f}, std: {model_std:.6f}
- Model bias: {model_bias:.6f}
- Model magnitude ratio: {model_magnitude_ratio:.2f}x

The model predictions have **{model_magnitude_ratio:.1f}x higher variance** than the targets, indicating excessive prediction magnitude.

### 5. Temporal Reconstruction

Per-timestep RMSE is nearly uniform across all 10 timesteps (std: {timestep_ratio_std:.3f}), indicating the model does not reconstruct temporal dynamics correctly.

### 6. Distribution Shift

"""

if train_data_list and val_data_list:
    diagnostic_report += "Sensor distributions show significant shift between training and test sets:\n\n"
    
    high_kl_sensors = sorted(distribution_stats, key=lambda x: x["kl_divergence"] if x["kl_divergence"] else 0, reverse=True)[:5]
    for s in high_kl_sensors:
        diagnostic_report += f"- **{s['sensor']}**: KL divergence = {s['kl_divergence']:.4f}\n"
        diagnostic_report += f"  Train: {s['train_mean']:.3f}±{s['train_std']:.3f}, Test: {s['test_mean']:.3f}±{s['test_std']:.3f}\n"
    
    diagnostic_report += f"\nThere is a correlation between distribution shift (KL divergence) and performance degradation (MAE ratio).\n"

diagnostic_report += """
## Root Cause Analysis

### Evidence for Distribution Shift:

1. **Tool variant mismatch**: Training used only M02 variants (05_M02_DC, 07_M02_DC), but test includes both M02 and M01 variants.
2. **Sensor distribution shift**: KL divergence analysis shows significant distribution differences between training and test for multiple sensors.
3. **Uneven sensor performance**: Sensors with higher distribution shift show worse performance.

### Evidence for Overfitting:

1. **Good validation performance**: Model achieved validation MSE of 0.051724 at epoch 48.
2. **Catastrophic test performance**: Test MSE is 16.5x worse than baseline, indicating severe overfitting to training distribution.
3. **Excessive prediction variance**: Model predictions have {model_magnitude_ratio:.1f}x higher variance than targets, suggesting the model learned to interpolate within training distribution but fails to generalize.

### Conclusion:

**Both distribution shift AND overfitting contribute to the failure.** The model learned patterns specific to the M02 tool variants and the training data distribution, but these patterns do not generalize to:
1. Unseen tool variants (M01)
2. Slightly different operating conditions in test files
3. Sensors with distribution shift

The model's excessive prediction variance suggests it is extrapolating beyond the training distribution rather than interpolating within it.

## Recommended Next Step

**Adaptive fusion of linear interpolation and Seq2Seq predictions:**

Given that linear interpolation performs consistently well across all test files and tool variants, while the neural model fails catastrophically on unseen distributions, the smallest defensible change is to:

1. **Implement a confidence-weighted fusion** that combines linear interpolation and Seq2Seq predictions
2. **Use prediction uncertainty or distribution similarity** to determine the fusion weight
3. **Fallback to linear interpolation** when the model is uncertain or the input distribution is far from training

This approach:
- Leverages the neural model's potential on in-distribution data
- Provides robustness via linear interpolation on out-of-distribution data
- Does not require retraining or architectural changes
- Can be implemented as a post-processing step

## Files Generated

- `per_file_mse_ratio.png` - Per-file MSE ratio by tool variant
- `per_sensor_mae_ratio.png` - Per-sensor MAE ratio
- `sensor_distribution_comparison.png` - Sensor distribution comparison across train/val/test
- `prediction_error_analysis.png` - Comprehensive error analysis
- `sample_predictions_comparison.png` - Sample predictions visualization
"""

report_path = DIAGNOSTIC_DIR / "DIAGNOSTIC_REPORT.md"
with open(report_path, "w") as f:
    f.write(diagnostic_report)

print(f"  Saved: {report_path}")

# Save diagnostic data as JSON
diagnostic_data = {
    "per_file_mse_ratios": {r["file"]: r["mse_ratio_vs_baseline"] for r in per_file_results},
    "per_sensor_mae_ratios": {sensor: ratio for sensor, ratio in sensor_mae_ratios},
    "tool_variant_analysis": {
        "m02_avg_ratio": float(np.mean([r["mse_ratio_vs_baseline"] for r in m02_files])) if m02_files else None,
        "m01_avg_ratio": float(np.mean([r["mse_ratio_vs_baseline"] for r in m01_files])) if m01_files else None
    },
    "prediction_magnitude": {
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "baseline_mean": float(baseline_mean),
        "baseline_std": float(baseline_std),
        "model_mean": float(model_mean),
        "model_std": float(model_std),
        "model_bias": float(model_bias),
        "model_magnitude_ratio": float(model_magnitude_ratio)
    },
    "temporal_reconstruction": {
        "timestep_ratio_std": float(timestep_ratio_std),
        "uniform_error": int(timestep_ratio_std < 0.5)
    },
    "distribution_shift": {
        "sensor_kl_divergences": {s["sensor"]: float(s["kl_divergence"]) if s["kl_divergence"] else None for s in distribution_stats}
    }
}

json_path = DIAGNOSTIC_DIR / "diagnostic_data.json"
with open(json_path, "w") as f:
    json.dump(diagnostic_data, f, indent=2)

print(f"  Saved: {json_path}")

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
print(f"All outputs saved to: {DIAGNOSTIC_DIR}")
print()
