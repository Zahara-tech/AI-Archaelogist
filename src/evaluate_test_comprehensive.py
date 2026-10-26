"""
Comprehensive Test Evaluation for Attention Seq2Seq V2 - 50 Epoch Experiment
Evaluates the best checkpoint on held-out test set with full diagnostics.
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
import csv

sys.path.append("src")
from attention_seq2seq_model import AttentionSeq2SeqReconstructor

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent.parent
BENCHMARK_DIR = BASE_DIR / "data" / "benchmark_v2"
EXPERIMENT_DIR = BASE_DIR / "experiments" / "attention_seq2seq_v2_50epochs"
CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints" / "best_model.pt"
TEST_EVAL_DIR = EXPERIMENT_DIR / "results" / "test_evaluation"

TEST_EVAL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

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
print("COMPREHENSIVE TEST EVALUATION - 50 EPOCH EXPERIMENT")
print("=" * 70)
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Test evaluation directory: {TEST_EVAL_DIR}")
print(f"Device: {DEVICE}")
print()

# ============================================================
# STEP 1: INSPECT EVALUATION PIPELINE
# ============================================================

print("=" * 70)
print("STEP 1: INSPECT EVALUATION PIPELINE")
print("=" * 70)

# Find test files
test_files = sorted(glob.glob(str(BENCHMARK_DIR / "test_*.npz")))
print(f"\nTest files found: {len(test_files)}")
for f in test_files:
    print(f"  - {os.path.basename(f)}")

# Load manifest
with open(BENCHMARK_DIR / "manifest.json", "r") as f:
    manifest = json.load(f)
    sensor_columns_from_manifest = manifest["sensor_columns"]

print(f"\nSensor columns from manifest: {len(sensor_columns_from_manifest)}")
print(f"Sensor columns match: {sensor_columns_from_manifest == SENSOR_COLUMNS}")

# Check scaler exists
scaler_path = BENCHMARK_DIR / "scaler.pkl"
print(f"\nScaler exists: {scaler_path.exists()}")
if scaler_path.exists():
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"Scaler type: {type(scaler).__name__}")
    print(f"Scaler n_features: {scaler.n_features_in_}")
    print(f"Scaler n_features matches sensors: {scaler.n_features_in_ == len(SENSOR_COLUMNS)}")

print()

# ============================================================
# STEP 2: LOAD MODEL
# ============================================================

print("=" * 70)
print("STEP 2: LOAD MODEL")
print("=" * 70)

model = AttentionSeq2SeqReconstructor(
    input_dim=17,
    hidden_dim=64,
    num_layers=2,
    output_dim=17,
    output_length=10,
    dropout=0.3,
    attention_dim=128,
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    best_epoch = checkpoint.get("best_epoch", "unknown")
    best_val_loss = checkpoint.get("best_val_loss", "unknown")
else:
    model.load_state_dict(checkpoint)
    best_epoch = "unknown"
    best_val_loss = "unknown"

model.to(DEVICE)
model.eval()

print(f"Model loaded successfully")
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss}")
print()

# ============================================================
# STEP 3: EVALUATE ON TEST SET
# ============================================================

print("=" * 70)
print("STEP 3: EVALUATE ON TEST SET")
print("=" * 70)

def linear_interpolation(before, after):
    """Reconstruct the 10-point gap using boundary values."""
    left = before[:, -1:, :]
    right = after[:, 0:1, :]
    steps = np.arange(1, 11, dtype=np.float32).reshape(1, 10, 1)
    alpha = steps / 11.0
    return left + alpha * (right - left)

def evaluate_model(before, after, target, batch_size=64):
    """Run model inference and calculate metrics."""
    predictions = []
    
    with torch.no_grad():
        for start in range(0, len(before), batch_size):
            end = min(start + batch_size, len(before))
            
            x_before = torch.tensor(before[start:end], dtype=torch.float32, device=DEVICE)
            x_after = torch.tensor(after[start:end], dtype=torch.float32, device=DEVICE)
            
            pred = model(x_before, x_after, teacher_forcing_ratio=0.0)
            predictions.append(pred.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    
    mse = float(np.mean((predictions - target) ** 2))
    mae = float(np.mean(np.abs(predictions - target)))
    rmse = float(np.sqrt(mse))
    
    return predictions, mse, mae, rmse

def calculate_per_sensor_metrics(predictions, targets, sensor_names):
    """Calculate MSE and MAE per sensor."""
    per_sensor_mse = {}
    per_sensor_mae = {}
    per_sensor_rmse = {}
    
    for i, sensor_name in enumerate(sensor_names):
        pred_sensor = predictions[:, :, i]
        target_sensor = targets[:, :, i]
        
        mse = float(np.mean((pred_sensor - target_sensor) ** 2))
        mae = float(np.mean(np.abs(pred_sensor - target_sensor)))
        rmse = float(np.sqrt(mse))
        
        per_sensor_mse[sensor_name] = mse
        per_sensor_mae[sensor_name] = mae
        per_sensor_rmse[sensor_name] = rmse
    
    return per_sensor_mse, per_sensor_mae, per_sensor_rmse

def calculate_per_timestep_metrics(predictions, targets):
    """Calculate MSE and MAE per timestep."""
    per_timestep_mse = []
    per_timestep_mae = []
    per_timestep_rmse = []
    
    for t in range(predictions.shape[1]):
        pred_t = predictions[:, t, :]
        target_t = targets[:, t, :]
        
        mse = float(np.mean((pred_t - target_t) ** 2))
        mae = float(np.mean(np.abs(pred_t - target_t)))
        rmse = float(np.sqrt(mse))
        
        per_timestep_mse.append(mse)
        per_timestep_mae.append(mae)
        per_timestep_rmse.append(rmse)
    
    return per_timestep_mse, per_timestep_mae, per_timestep_rmse

# Evaluate each test file
all_targets = []
all_baseline_predictions = []
all_model_predictions = []

per_file_results = []

for test_path in test_files:
    filename = os.path.basename(test_path)
    print(f"\nEvaluating: {filename}")
    
    data = np.load(test_path)
    before = data["context_before"].astype(np.float32)
    after = data["context_after"].astype(np.float32)
    target = data["masked_segment"].astype(np.float32)
    
    print(f"  Sequences: {len(target)}")
    print(f"  Shapes: before={before.shape}, after={after.shape}, target={target.shape}")
    
    # Linear interpolation
    baseline_pred = linear_interpolation(before, after)
    baseline_mse = float(np.mean((baseline_pred - target) ** 2))
    baseline_mae = float(np.mean(np.abs(baseline_pred - target)))
    baseline_rmse = float(np.sqrt(baseline_mse))
    
    # Model prediction
    model_pred, model_mse, model_mae, model_rmse = evaluate_model(before, after, target)
    
    # Per-sensor metrics
    baseline_per_sensor_mse, baseline_per_sensor_mae, baseline_per_sensor_rmse = calculate_per_sensor_metrics(
        baseline_pred, target, SENSOR_COLUMNS
    )
    model_per_sensor_mse, model_per_sensor_mae, model_per_sensor_rmse = calculate_per_sensor_metrics(
        model_pred, target, SENSOR_COLUMNS
    )
    
    # Per-timestep metrics
    baseline_per_timestep_mse, baseline_per_timestep_mae, baseline_per_timestep_rmse = calculate_per_timestep_metrics(
        baseline_pred, target
    )
    model_per_timestep_mse, model_per_timestep_mae, model_per_timestep_rmse = calculate_per_timestep_metrics(
        model_pred, target
    )
    
    # Ratios
    mse_ratio = model_mse / baseline_mse if baseline_mse > 0 else float("inf")
    mae_ratio = model_mae / baseline_mae if baseline_mae > 0 else float("inf")
    
    result = {
        "file": filename,
        "sequences": int(len(target)),
        "baseline_mse": baseline_mse,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "model_mse": model_mse,
        "model_mae": model_mae,
        "model_rmse": model_rmse,
        "mse_ratio_vs_baseline": mse_ratio,
        "mae_ratio_vs_baseline": mae_ratio,
        "baseline_per_sensor_mse": baseline_per_sensor_mse,
        "baseline_per_sensor_mae": baseline_per_sensor_mae,
        "baseline_per_sensor_rmse": baseline_per_sensor_rmse,
        "model_per_sensor_mse": model_per_sensor_mse,
        "model_per_sensor_mae": model_per_sensor_mae,
        "model_per_sensor_rmse": model_per_sensor_rmse,
        "baseline_per_timestep_mse": baseline_per_timestep_mse,
        "baseline_per_timestep_mae": baseline_per_timestep_mae,
        "baseline_per_timestep_rmse": baseline_per_timestep_rmse,
        "model_per_timestep_mse": model_per_timestep_mse,
        "model_per_timestep_mae": model_per_timestep_mae,
        "model_per_timestep_rmse": model_per_timestep_rmse
    }
    
    per_file_results.append(result)
    
    all_targets.append(target)
    all_baseline_predictions.append(baseline_pred)
    all_model_predictions.append(model_pred)
    
    print(f"  Linear Interpolation - MSE: {baseline_mse:.6f}, MAE: {baseline_mae:.6f}, RMSE: {baseline_rmse:.6f}")
    print(f"  Attention Seq2Seq V2  - MSE: {model_mse:.6f}, MAE: {model_mae:.6f}, RMSE: {model_rmse:.6f}")
    print(f"  MSE ratio: {mse_ratio:.3f}x, MAE ratio: {mae_ratio:.3f}x")

# Concatenate all results
all_targets = np.concatenate(all_targets, axis=0)
all_baseline_predictions = np.concatenate(all_baseline_predictions, axis=0)
all_model_predictions = np.concatenate(all_model_predictions, axis=0)

print(f"\nTotal test sequences: {len(all_targets)}")

# ============================================================
# STEP 4: CALCULATE OVERALL METRICS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: CALCULATE OVERALL METRICS")
print("=" * 70)

overall_baseline_mse = float(np.mean((all_baseline_predictions - all_targets) ** 2))
overall_baseline_mae = float(np.mean(np.abs(all_baseline_predictions - all_targets)))
overall_baseline_rmse = float(np.sqrt(overall_baseline_mse))

overall_model_mse = float(np.mean((all_model_predictions - all_targets) ** 2))
overall_model_mae = float(np.mean(np.abs(all_model_predictions - all_targets)))
overall_model_rmse = float(np.sqrt(overall_model_mse))

overall_mse_ratio = overall_model_mse / overall_baseline_mse
overall_mae_ratio = overall_model_mae / overall_baseline_mae

print(f"\nLinear Interpolation:")
print(f"  MSE: {overall_baseline_mse:.6f}")
print(f"  MAE: {overall_baseline_mae:.6f}")
print(f"  RMSE: {overall_baseline_rmse:.6f}")

print(f"\nAttention Seq2Seq V2:")
print(f"  MSE: {overall_model_mse:.6f}")
print(f"  MAE: {overall_model_mae:.6f}")
print(f"  RMSE: {overall_model_rmse:.6f}")

print(f"\nComparison:")
print(f"  MSE ratio: {overall_mse_ratio:.3f}x")
print(f"  MAE ratio: {overall_mae_ratio:.3f}x")

if overall_model_mse < overall_baseline_mse:
    print(f"\nRESULT: Attention Seq2Seq V2 BEATS linear interpolation by {(1 - overall_mse_ratio) * 100:.1f}% in MSE")
else:
    print(f"\nRESULT: Linear interpolation BEATS Attention Seq2Seq V2 by {(overall_mse_ratio - 1) * 100:.1f}% in MSE")

# Overall per-sensor metrics
overall_baseline_per_sensor_mse, overall_baseline_per_sensor_mae, overall_baseline_per_sensor_rmse = calculate_per_sensor_metrics(
    all_baseline_predictions, all_targets, SENSOR_COLUMNS
)
overall_model_per_sensor_mse, overall_model_per_sensor_mae, overall_model_per_sensor_rmse = calculate_per_sensor_metrics(
    all_model_predictions, all_targets, SENSOR_COLUMNS
)

# Overall per-timestep metrics
overall_baseline_per_timestep_mse, overall_baseline_per_timestep_mae, overall_baseline_per_timestep_rmse = calculate_per_timestep_metrics(
    all_baseline_predictions, all_targets
)
overall_model_per_timestep_mse, overall_model_per_timestep_mae, overall_model_per_timestep_rmse = calculate_per_timestep_metrics(
    all_model_predictions, all_targets
)

print()

# ============================================================
# STEP 5: VERIFY RESULTS
# ============================================================

print("=" * 70)
print("STEP 5: VERIFY RESULTS")
print("=" * 70)

# Check for NaN or infinite values
has_nan_model = np.isnan(all_model_predictions).any()
has_inf_model = np.isinf(all_model_predictions).any()
has_nan_baseline = np.isnan(all_baseline_predictions).any()
has_inf_baseline = np.isinf(all_baseline_predictions).any()

print(f"\nNaN in model predictions: {has_nan_model}")
print(f"Infinite in model predictions: {has_inf_model}")
print(f"NaN in baseline predictions: {has_nan_baseline}")
print(f"Infinite in baseline predictions: {has_inf_baseline}")

# Verify shapes
print(f"\nShape verification:")
print(f"  Targets shape: {all_targets.shape}")
print(f"  Model predictions shape: {all_model_predictions.shape}")
print(f"  Baseline predictions shape: {all_baseline_predictions.shape}")
print(f"  Shapes match: {all_targets.shape == all_model_predictions.shape == all_baseline_predictions.shape}")

# Verify expected shape
expected_shape = (len(all_targets), 10, 17)
print(f"  Expected shape: {expected_shape}")
print(f"  Shape correct: {all_model_predictions.shape == expected_shape}")

print()

# ============================================================
# STEP 6: GENERATE DIAGNOSTIC PLOTS
# ============================================================

print("=" * 70)
print("STEP 6: GENERATE DIAGNOSTIC PLOTS")
print("=" * 70)

# Plot 1: Actual vs Predicted for representative sensors
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
fig.suptitle('Actual vs Predicted vs Linear Interpolation (Sample 0)', fontsize=14)

sample_idx = 0
for i in range(min(16, len(SENSOR_COLUMNS))):
    ax = axes[i // 4, i % 4]
    timesteps = np.arange(1, 11)
    actual = all_targets[sample_idx, :, i]
    predicted = all_model_predictions[sample_idx, :, i]
    baseline = all_baseline_predictions[sample_idx, :, i]
    
    ax.plot(timesteps, actual, 'o-', label='Actual', linewidth=2, markersize=6, color='green')
    ax.plot(timesteps, predicted, 's-', label='Attention Seq2Seq', linewidth=2, markersize=6, color='blue')
    ax.plot(timesteps, baseline, '^-', label='Linear Interp', linewidth=2, markersize=6, color='red')
    
    ax.axvspan(0.5, 10.5, alpha=0.1, color='gray', label='Hidden Interval')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Normalized Value')
    ax.set_title(SENSOR_COLUMNS[i])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path_1 = TEST_EVAL_DIR / "actual_vs_predicted_sensors.png"
plt.savefig(plot_path_1, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path_1}")
plt.close()

# Plot 2: Per-sensor MAE comparison
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(SENSOR_COLUMNS))
width = 0.35

baseline_mae_values = [overall_baseline_per_sensor_mae[s] for s in SENSOR_COLUMNS]
model_mae_values = [overall_model_per_sensor_mae[s] for s in SENSOR_COLUMNS]

ax.bar(x - width/2, baseline_mae_values, width, label='Linear Interpolation', alpha=0.8, color='red')
ax.bar(x + width/2, model_mae_values, width, label='Attention Seq2Seq V2', alpha=0.8, color='blue')

ax.set_xlabel('Sensor')
ax.set_ylabel('MAE (Normalized)')
ax.set_title('Per-Sensor MAE: Linear Interpolation vs Attention Seq2Seq V2')
ax.set_xticks(x)
ax.set_xticklabels(SENSOR_COLUMNS, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path_2 = TEST_EVAL_DIR / "per_sensor_mae_comparison.png"
plt.savefig(plot_path_2, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path_2}")
plt.close()

# Plot 3: Per-sensor RMSE comparison
fig, ax = plt.subplots(figsize=(12, 6))
baseline_rmse_values = [overall_baseline_per_sensor_rmse[s] for s in SENSOR_COLUMNS]
model_rmse_values = [overall_model_per_sensor_rmse[s] for s in SENSOR_COLUMNS]

ax.bar(x - width/2, baseline_rmse_values, width, label='Linear Interpolation', alpha=0.8, color='red')
ax.bar(x + width/2, model_rmse_values, width, label='Attention Seq2Seq V2', alpha=0.8, color='blue')

ax.set_xlabel('Sensor')
ax.set_ylabel('RMSE (Normalized)')
ax.set_title('Per-Sensor RMSE: Linear Interpolation vs Attention Seq2Seq V2')
ax.set_xticks(x)
ax.set_xticklabels(SENSOR_COLUMNS, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path_3 = TEST_EVAL_DIR / "per_sensor_rmse_comparison.png"
plt.savefig(plot_path_3, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path_3}")
plt.close()

# Plot 4: Error distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

model_errors = (all_model_predictions - all_targets).flatten()
baseline_errors = (all_baseline_predictions - all_targets).flatten()

axes[0].hist(model_errors, bins=100, alpha=0.7, color='blue', label='Attention Seq2Seq V2')
axes[0].hist(baseline_errors, bins=100, alpha=0.5, color='red', label='Linear Interpolation')
axes[0].set_xlabel('Prediction Error (Normalized)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Error Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

bp = axes[1].boxplot([model_errors, baseline_errors])
axes[1].set_xticklabels(['Attention Seq2Seq V2', 'Linear Interpolation'])
axes[1].set_ylabel('Prediction Error (Normalized)')
axes[1].set_title('Error Distribution (Box Plot)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path_4 = TEST_EVAL_DIR / "error_distribution.png"
plt.savefig(plot_path_4, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path_4}")
plt.close()

# Plot 5: Per-timestep RMSE
fig, ax = plt.subplots(figsize=(10, 6))
timesteps = np.arange(1, 11)
ax.plot(timesteps, overall_baseline_per_timestep_rmse, 'o-', label='Linear Interpolation', linewidth=2, markersize=8, color='red')
ax.plot(timesteps, overall_model_per_timestep_rmse, 's-', label='Attention Seq2Seq V2', linewidth=2, markersize=8, color='blue')

ax.set_xlabel('Timestep')
ax.set_ylabel('RMSE (Normalized)')
ax.set_title('Per-Timestep RMSE: Linear Interpolation vs Attention Seq2Seq V2')
ax.set_xticks(timesteps)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path_5 = TEST_EVAL_DIR / "per_timestep_rmse.png"
plt.savefig(plot_path_5, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path_5}")
plt.close()

print()

# ============================================================
# STEP 7: SAVE RESULTS
# ============================================================

print("=" * 70)
print("STEP 7: SAVE RESULTS")
print("=" * 70)

# Save evaluation summary JSON
evaluation_summary = {
    "experiment": "attention_seq2seq_v2_50epochs",
    "checkpoint_path": str(CHECKPOINT_PATH),
    "best_epoch": best_epoch,
    "best_validation_loss": best_val_loss,
    "evaluation_timestamp": "2024-10-23",
    "normalization": "StandardScaler fitted on training data only, applied to all data",
    "metrics_scale": "Normalized (StandardScaler transformed)",
    "test_files": [os.path.basename(f) for f in test_files],
    "total_test_sequences": int(len(all_targets)),
    "protocol": {
        "context_before": 50,
        "gap_length": 10,
        "context_after": 50,
        "sensor_count": 17,
        "run_aware": True
    },
    "overall_metrics": {
        "linear_interpolation": {
            "mse": overall_baseline_mse,
            "mae": overall_baseline_mae,
            "rmse": overall_baseline_rmse
        },
        "attention_seq2seq_v2": {
            "mse": overall_model_mse,
            "mae": overall_model_mae,
            "rmse": overall_model_rmse
        },
        "comparison": {
            "mse_ratio_vs_baseline": overall_mse_ratio,
            "mae_ratio_vs_baseline": overall_mae_ratio,
            "better_method": "attention_seq2seq_v2" if overall_model_mse < overall_baseline_mse else "linear_interpolation"
        }
    },
    "verification": {
        "nan_in_model_predictions": int(has_nan_model),
        "inf_in_model_predictions": int(has_inf_model),
        "nan_in_baseline_predictions": int(has_nan_baseline),
        "inf_in_baseline_predictions": int(has_inf_baseline),
        "output_shape_correct": int(all_model_predictions.shape == expected_shape),
        "sensor_order_matches": int(sensor_columns_from_manifest == SENSOR_COLUMNS),
        "scaler_exists": int(scaler_path.exists())
    },
    "per_file": per_file_results,
    "per_sensor": {
        "baseline_mae": overall_baseline_per_sensor_mae,
        "baseline_rmse": overall_baseline_per_sensor_rmse,
        "model_mae": overall_model_per_sensor_mae,
        "model_rmse": overall_model_per_sensor_rmse
    },
    "per_timestep": {
        "baseline_rmse": overall_baseline_per_timestep_rmse,
        "model_rmse": overall_model_per_timestep_rmse
    }
}

summary_path = TEST_EVAL_DIR / "evaluation_summary.json"
with open(summary_path, "w") as f:
    json.dump(evaluation_summary, f, indent=2)
print(f"  Saved: {summary_path}")

# Save overall metrics CSV
overall_csv_path = TEST_EVAL_DIR / "overall_metrics.csv"
with open(overall_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Method", "MSE", "MAE", "RMSE"])
    writer.writerow(["Linear Interpolation", overall_baseline_mse, overall_baseline_mae, overall_baseline_rmse])
    writer.writerow(["Attention Seq2Seq V2", overall_model_mse, overall_model_mae, overall_model_rmse])
print(f"  Saved: {overall_csv_path}")

# Save per-file metrics CSV
per_file_csv_path = TEST_EVAL_DIR / "per_file_metrics.csv"
with open(per_file_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["File", "Sequences", "Baseline MSE", "Baseline MAE", "Baseline RMSE", 
                     "Model MSE", "Model MAE", "Model RMSE", "MSE Ratio", "MAE Ratio"])
    for r in per_file_results:
        writer.writerow([r["file"], r["sequences"], r["baseline_mse"], r["baseline_mae"], r["baseline_rmse"],
                        r["model_mse"], r["model_mae"], r["model_rmse"], r["mse_ratio_vs_baseline"], r["mae_ratio_vs_baseline"]])
print(f"  Saved: {per_file_csv_path}")

# Save per-sensor metrics CSV
per_sensor_csv_path = TEST_EVAL_DIR / "per_sensor_metrics.csv"
with open(per_sensor_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Sensor", "Baseline MAE", "Baseline RMSE", "Model MAE", "Model RMSE", "MAE Ratio", "RMSE Ratio"])
    for sensor in SENSOR_COLUMNS:
        b_mae = overall_baseline_per_sensor_mae[sensor]
        b_rmse = overall_baseline_per_sensor_rmse[sensor]
        m_mae = overall_model_per_sensor_mae[sensor]
        m_rmse = overall_model_per_sensor_rmse[sensor]
        mae_ratio = m_mae / b_mae if b_mae > 0 else float("inf")
        rmse_ratio = m_rmse / b_rmse if b_rmse > 0 else float("inf")
        writer.writerow([sensor, b_mae, b_rmse, m_mae, m_rmse, mae_ratio, rmse_ratio])
print(f"  Saved: {per_sensor_csv_path}")

# Save predictions NPZ
np.savez(
    TEST_EVAL_DIR / "predictions.npz",
    targets=all_targets,
    baseline_predictions=all_baseline_predictions,
    model_predictions=all_model_predictions
)
print(f"  Saved: {TEST_EVAL_DIR / 'predictions.npz'}")

print()

# ============================================================
# STEP 8: GENERATE MARKDOWN REPORT
# ============================================================

print("=" * 70)
print("STEP 8: GENERATE MARKDOWN REPORT")
print("=" * 70)

markdown_report = f"""# Test Evaluation Report: Attention Seq2Seq V2 (50 Epochs)

## Experiment Summary

- **Experiment Name**: attention_seq2seq_v2_50epochs
- **Checkpoint Path**: `{CHECKPOINT_PATH}`
- **Best Epoch**: {best_epoch}
- **Best Validation Loss**: {best_val_loss}
- **Evaluation Date**: 2024-10-23

## Test Configuration

- **Context Before**: 50 timesteps
- **Gap Length**: 10 timesteps
- **Context After**: 50 timesteps
- **Sensor Count**: 17
- **Run-Aware Splitting**: Yes
- **Normalization**: StandardScaler fitted on training data only, applied to all data
- **Metrics Scale**: Normalized (StandardScaler transformed)

## Test Files

{len(test_files)} held-out test files:
{chr(10).join(f"- {os.path.basename(f)}" for f in test_files)}

**Total Test Sequences**: {len(all_targets)}

## Overall Results

### Linear Interpolation
- **MSE**: {overall_baseline_mse:.6f}
- **MAE**: {overall_baseline_mae:.6f}
- **RMSE**: {overall_baseline_rmse:.6f}

### Attention Seq2Seq V2
- **MSE**: {overall_model_mse:.6f}
- **MAE**: {overall_model_mae:.6f}
- **RMSE**: {overall_model_rmse:.6f}

### Comparison
- **MSE Ratio**: {overall_mse_ratio:.3f}x
- **MAE Ratio**: {overall_mae_ratio:.3f}x

"""

if overall_model_mse < overall_baseline_mse:
    markdown_report += f"### Result\n\n**Attention Seq2Seq V2 outperforms linear interpolation** by {(1 - overall_mse_ratio) * 100:.1f}% in MSE.\n\n"
else:
    markdown_report += f"### Result\n\n**Linear interpolation outperforms Attention Seq2Seq V2** by {(overall_mse_ratio - 1) * 100:.1f}% in MSE.\n\n"

markdown_report += f"""## Verification

- **NaN in model predictions**: {has_nan_model}
- **Infinite in model predictions**: {has_inf_model}
- **NaN in baseline predictions**: {has_nan_baseline}
- **Infinite in baseline predictions**: {has_inf_baseline}
- **Output shape correct**: {all_model_predictions.shape == expected_shape}
- **Sensor order matches**: {sensor_columns_from_manifest == SENSOR_COLUMNS}
- **Scaler exists**: {scaler_path.exists()}

## Per-File Results

| File | Sequences | Baseline MSE | Baseline MAE | Baseline RMSE | Model MSE | Model MAE | Model RMSE | MSE Ratio | MAE Ratio |
|------|-----------|--------------|--------------|---------------|-----------|-----------|------------|-----------|-----------|
"""

for r in per_file_results:
    markdown_report += f"| {r['file']} | {r['sequences']} | {r['baseline_mse']:.6f} | {r['baseline_mae']:.6f} | {r['baseline_rmse']:.6f} | {r['model_mse']:.6f} | {r['model_mae']:.6f} | {r['model_rmse']:.6f} | {r['mse_ratio_vs_baseline']:.3f}x | {r['mae_ratio_vs_baseline']:.3f}x |\n"

markdown_report += """
## Per-Sensor Results

| Sensor | Baseline MAE | Baseline RMSE | Model MAE | Model RMSE | MAE Ratio | RMSE Ratio |
|--------|--------------|---------------|-----------|------------|-----------|------------|
"""

for sensor in SENSOR_COLUMNS:
    b_mae = overall_baseline_per_sensor_mae[sensor]
    b_rmse = overall_baseline_per_sensor_rmse[sensor]
    m_mae = overall_model_per_sensor_mae[sensor]
    m_rmse = overall_model_per_sensor_rmse[sensor]
    mae_ratio = m_mae / b_mae if b_mae > 0 else float("inf")
    rmse_ratio = m_rmse / b_rmse if b_rmse > 0 else float("inf")
    markdown_report += f"| {sensor} | {b_mae:.6f} | {b_rmse:.6f} | {m_mae:.6f} | {m_rmse:.6f} | {mae_ratio:.3f}x | {rmse_ratio:.3f}x |\n"

markdown_report += """
## Limitations

- Metrics are reported in normalized space (StandardScaler transformed).
- The test set was not used for model selection or hyperparameter tuning.
- Results are specific to the PHM 2018 Ion Mill Etch Tool dataset.
- The model was trained on two tool variants (05_M02_DC and 07_M02_DC) and tested on five tool variants.

## Files Generated

- `evaluation_summary.json` - Complete evaluation results in JSON format
- `overall_metrics.csv` - Overall metrics comparison
- `per_file_metrics.csv` - Per-file metrics breakdown
- `per_sensor_metrics.csv` - Per-sensor metrics breakdown
- `predictions.npz` - Raw predictions and targets
- `actual_vs_predicted_sensors.png` - Visualization of predictions for representative sensors
- `per_sensor_mae_comparison.png` - Per-sensor MAE comparison bar chart
- `per_sensor_rmse_comparison.png` - Per-sensor RMSE comparison bar chart
- `error_distribution.png` - Error distribution histograms and box plots
- `per_timestep_rmse.png` - Per-timestep RMSE comparison
"""

markdown_path = TEST_EVAL_DIR / "EVALUATION_REPORT.md"
with open(markdown_path, "w") as f:
    f.write(markdown_report)
print(f"  Saved: {markdown_path}")

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
print(f"All results saved to: {TEST_EVAL_DIR}")
print()
