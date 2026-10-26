"""
Diagnose Low Prediction Variance in Attention Seq2Seq V2
Investigates why the trained model produces nearly constant predictions.
"""

import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append("src")
from attention_seq2seq_model import AttentionSeq2SeqReconstructor

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent.parent
BENCHMARK_DIR = BASE_DIR / "data" / "benchmark_v2"
EXPERIMENT_DIR = BASE_DIR / "experiments" / "attention_seq2seq_v2_50epochs"
CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints" / "best_model.pt"
DIAGNOSTIC_DIR = EXPERIMENT_DIR / "results" / "diagnostics"

DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

VAL_FILES = [
    "val_05_M02_DC_train.npz",
    "val_07_M02_DC_train.npz"
]

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
print("DIAGNOSTIC: LOW PREDICTION VARIANCE INVESTIGATION")
print("=" * 70)
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Diagnostic output: {DIAGNOSTIC_DIR}")
print()

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")
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
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()
print("Model loaded.")
print()

# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("Loading validation data...")
val_data_list = []
for val_file in VAL_FILES:
    val_path = BENCHMARK_DIR / val_file
    data = np.load(val_path)
    val_data_list.append(data)
    print(f"  Loaded {val_file}: {len(data['context_before'])} sequences")

# Concatenate all validation data
context_before = np.concatenate([d["context_before"] for d in val_data_list], axis=0)
context_after = np.concatenate([d["context_after"] for d in val_data_list], axis=0)
targets = np.concatenate([d["masked_segment"] for d in val_data_list], axis=0)

print(f"\nTotal validation sequences: {len(context_before)}")
print(f"Shapes: before={context_before.shape}, after={context_after.shape}, target={targets.shape}")
print()

# ============================================================
# RUN PREDICTIONS
# ============================================================

print("Running predictions on validation samples...")
batch_size = 64
all_predictions = []

with torch.no_grad():
    for start in range(0, len(context_before), batch_size):
        end = min(start + batch_size, len(context_before))
        
        x_before = torch.tensor(
            context_before[start:end],
            dtype=torch.float32,
            device=DEVICE
        )
        x_after = torch.tensor(
            context_after[start:end],
            dtype=torch.float32,
            device=DEVICE
        )
        
        pred = model(x_before, x_after, teacher_forcing_ratio=0.0)
        all_predictions.append(pred.cpu().numpy())

predictions = np.concatenate(all_predictions, axis=0)
print(f"Predictions shape: {predictions.shape}")
print()

# ============================================================
# CALCULATE STATISTICS
# ============================================================

print("Calculating statistics...")
print()

# Statistics for predictions
pred_mean = np.mean(predictions)
pred_std = np.std(predictions)
pred_std_across_timesteps = np.std(predictions, axis=1).mean()  # Average std across timesteps
pred_std_across_sensors = np.std(predictions, axis=2).mean()  # Average std across sensors
pred_std_across_samples = np.std(predictions, axis=0).mean()  # Average std across samples

print("=" * 70)
print("PREDICTION STATISTICS")
print("=" * 70)
print(f"Mean: {pred_mean:.6f}")
print(f"Overall std: {pred_std:.6f}")
print(f"Average std across timesteps (per sample): {pred_std_across_timesteps:.6f}")
print(f"Average std across sensors (per sample): {pred_std_across_sensors:.6f}")
print(f"Average std across samples (per timestep/sensor): {pred_std_across_samples:.6f}")
print()

# Statistics for targets
target_mean = np.mean(targets)
target_std = np.std(targets)
target_std_across_timesteps = np.std(targets, axis=1).mean()
target_std_across_sensors = np.std(targets, axis=2).mean()
target_std_across_samples = np.std(targets, axis=0).mean()

print("=" * 70)
print("TARGET STATISTICS")
print("=" * 70)
print(f"Mean: {target_mean:.6f}")
print(f"Overall std: {target_std:.6f}")
print(f"Average std across timesteps (per sample): {target_std_across_timesteps:.6f}")
print(f"Average std across sensors (per sample): {target_std_across_sensors:.6f}")
print(f"Average std across samples (per timestep/sensor): {target_std_across_samples:.6f}")
print()

# Per-sensor statistics
print("=" * 70)
print("PER-SENSOR STATISTICS")
print("=" * 70)
print(f"{'Sensor':<30} | {'Pred Std':<10} | {'Target Std':<10} | {'Ratio':<10}")
print("-" * 70)

per_sensor_pred_std = []
per_sensor_target_std = []
per_sensor_ratios = []

for i, sensor in enumerate(SENSOR_COLUMNS):
    pred_sensor = predictions[:, :, i]
    target_sensor = targets[:, :, i]
    
    pred_std_sensor = np.std(pred_sensor)
    target_std_sensor = np.std(target_sensor)
    ratio = pred_std_sensor / target_std_sensor if target_std_sensor > 0 else float('inf')
    
    per_sensor_pred_std.append(pred_std_sensor)
    per_sensor_target_std.append(target_std_sensor)
    per_sensor_ratios.append(ratio)
    
    print(f"{sensor:<30} | {pred_std_sensor:.6f} | {target_std_sensor:.6f} | {ratio:.2f}x")

print()

# Per-timestep statistics
print("=" * 70)
print("PER-TIMESTEP STATISTICS")
print("=" * 70)
print(f"{'Timestep':<10} | {'Pred Std':<10} | {'Target Std':<10} | {'Ratio':<10}")
print("-" * 70)

per_timestep_pred_std = []
per_timestep_target_std = []
per_timestep_ratios = []

for t in range(10):
    pred_t = predictions[:, t, :]
    target_t = targets[:, t, :]
    
    pred_std_t = np.std(pred_t)
    target_std_t = np.std(target_t)
    ratio = pred_std_t / target_std_t if target_std_t > 0 else float('inf')
    
    per_timestep_pred_std.append(pred_std_t)
    per_timestep_target_std.append(target_std_t)
    per_timestep_ratios.append(ratio)
    
    print(f"{t+1:<10} | {pred_std_t:.6f} | {target_std_t:.6f} | {ratio:.2f}x")

print()

# ============================================================
# LINEAR INTERPOLATION BASELINE
# ============================================================

print("Calculating linear interpolation baseline...")

def linear_interpolation(before, after):
    """Reconstruct the 10-point gap using boundary values."""
    left = before[:, -1:, :]
    right = after[:, 0:1, :]
    steps = np.arange(1, 11, dtype=np.float32).reshape(1, 10, 1)
    alpha = steps / 11.0
    return left + alpha * (right - left)

baseline_pred = linear_interpolation(context_before, context_after)

baseline_std = np.std(baseline_pred)
baseline_std_across_timesteps = np.std(baseline_pred, axis=1).mean()
baseline_std_across_sensors = np.std(baseline_pred, axis=2).mean()

print("=" * 70)
print("LINEAR INTERPOLATION STATISTICS")
print("=" * 70)
print(f"Overall std: {baseline_std:.6f}")
print(f"Average std across timesteps: {baseline_std_across_timesteps:.6f}")
print(f"Average std across sensors: {baseline_std_across_sensors:.6f}")
print()

# ============================================================
# PLOT ACTUAL VS PREDICTED
# ============================================================

print("Generating plots...")

# Select a few samples for visualization
num_samples_to_plot = 5
sample_indices = np.random.choice(len(predictions), num_samples_to_plot, replace=False)

# Plot 1: Actual vs Predicted for first few sensors
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
fig.suptitle('Actual vs Predicted Values Across Sensors (Sample 0)', fontsize=14)

for i in range(min(16, len(SENSOR_COLUMNS))):
    ax = axes[i // 4, i % 4]
    timesteps = np.arange(1, 11)
    actual = targets[sample_indices[0], :, i]
    predicted = predictions[sample_indices[0], :, i]
    baseline = baseline_pred[sample_indices[0], :, i]
    
    ax.plot(timesteps, actual, 'o-', label='Actual', linewidth=2, markersize=6)
    ax.plot(timesteps, predicted, 's-', label='Predicted', linewidth=2, markersize=6)
    ax.plot(timesteps, baseline, '^-', label='Linear', linewidth=2, markersize=6)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Normalized Value')
    ax.set_title(SENSOR_COLUMNS[i])
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path_1 = DIAGNOSTIC_DIR / "actual_vs_predicted_sensors.png"
plt.savefig(plot_path_1, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path_1}")
plt.close()

# Plot 2: Per-sensor standard deviation comparison
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(SENSOR_COLUMNS))
width = 0.35

ax.bar(x - width/2, per_sensor_pred_std, width, label='Predictions', alpha=0.8)
ax.bar(x + width/2, per_sensor_target_std, width, label='Targets', alpha=0.8)

ax.set_xlabel('Sensor')
ax.set_ylabel('Standard Deviation')
ax.set_title('Per-Sensor Standard Deviation: Predictions vs Targets')
ax.set_xticks(x)
ax.set_xticklabels(SENSOR_COLUMNS, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path_2 = DIAGNOSTIC_DIR / "per_sensor_std_comparison.png"
plt.savefig(plot_path_2, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path_2}")
plt.close()

# Plot 3: Per-timestep standard deviation comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(1, 11)
width = 0.35

ax.bar(x - width/2, per_timestep_pred_std, width, label='Predictions', alpha=0.8)
ax.bar(x + width/2, per_timestep_target_std, width, label='Targets', alpha=0.8)

ax.set_xlabel('Timestep')
ax.set_ylabel('Standard Deviation')
ax.set_title('Per-Timestep Standard Deviation: Predictions vs Targets')
ax.set_xticks(x)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path_3 = DIAGNOSTIC_DIR / "per_timestep_std_comparison.png"
plt.savefig(plot_path_3, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path_3}")
plt.close()

# Plot 4: Prediction variance across samples for each sensor
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
fig.suptitle('Prediction Variance Across Samples (Per Sensor)', fontsize=14)

for i in range(min(16, len(SENSOR_COLUMNS))):
    ax = axes[i // 4, i % 4]
    
    # Calculate variance across samples for each timestep
    pred_var_per_timestep = np.var(predictions[:, :, i], axis=0)
    target_var_per_timestep = np.var(targets[:, :, i], axis=0)
    
    timesteps = np.arange(1, 11)
    ax.plot(timesteps, pred_var_per_timestep, 'o-', label='Predictions', linewidth=2, markersize=6)
    ax.plot(timesteps, target_var_per_timestep, 's-', label='Targets', linewidth=2, markersize=6)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Variance')
    ax.set_title(SENSOR_COLUMNS[i])
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path_4 = DIAGNOSTIC_DIR / "variance_across_samples_per_sensor.png"
plt.savefig(plot_path_4, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path_4}")
plt.close()

# Plot 5: Sample predictions across all sensors (heatmap)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sample_idx = sample_indices[0]
im0 = axes[0].imshow(predictions[sample_idx].T, aspect='auto', cmap='viridis')
axes[0].set_title('Predictions')
axes[0].set_xlabel('Timestep')
axes[0].set_ylabel('Sensor')
axes[0].set_yticks(np.arange(len(SENSOR_COLUMNS)))
axes[0].set_yticklabels(SENSOR_COLUMNS, fontsize=8)
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(targets[sample_idx].T, aspect='auto', cmap='viridis')
axes[1].set_title('Targets')
axes[1].set_xlabel('Timestep')
axes[1].set_ylabel('Sensor')
axes[1].set_yticks(np.arange(len(SENSOR_COLUMNS)))
axes[1].set_yticklabels(SENSOR_COLUMNS, fontsize=8)
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(baseline_pred[sample_idx].T, aspect='auto', cmap='viridis')
axes[2].set_title('Linear Interpolation')
axes[2].set_xlabel('Timestep')
axes[2].set_ylabel('Sensor')
axes[2].set_yticks(np.arange(len(SENSOR_COLUMNS)))
axes[2].set_yticklabels(SENSOR_COLUMNS, fontsize=8)
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plot_path_5 = DIAGNOSTIC_DIR / "sample_predictions_heatmap.png"
plt.savefig(plot_path_5, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path_5}")
plt.close()

print()

# ============================================================
# INSPECT MODEL ARCHITECTURE
# ============================================================

print("=" * 70)
print("MODEL ARCHITECTURE INSPECTION")
print("=" * 70)

# Check decoder output layer
print("\nDecoder output projection layer:")
print(f"  Type: {type(model.output_projection)}")
print(f"  Structure: {model.output_projection}")

# Check the final linear layer in the output projection
final_layer = model.output_projection[-1]
print(f"\nFinal output layer type: {type(final_layer)}")
print(f"  Weight shape: {final_layer.weight.shape}")
print(f"  Bias shape: {final_layer.bias.shape if final_layer.bias is not None else 'None'}")

# Check weight statistics
decoder_weight = final_layer.weight.data
decoder_bias = final_layer.bias.data

print(f"\nDecoder weight statistics:")
print(f"  Mean: {decoder_weight.mean():.6f}")
print(f"  Std: {decoder_weight.std():.6f}")
print(f"  Min: {decoder_weight.min():.6f}")
print(f"  Max: {decoder_weight.max():.6f}")

if decoder_bias is not None:
    print(f"\nDecoder bias statistics:")
    print(f"  Mean: {decoder_bias.mean():.6f}")
    print(f"  Std: {decoder_bias.std():.6f}")
    print(f"  Min: {decoder_bias.min():.6f}")
    print(f"  Max: {decoder_bias.max():.6f}")

# Check if weights are near zero
near_zero_weight = (torch.abs(decoder_weight) < 1e-6).sum().item()
near_zero_bias = (torch.abs(decoder_bias) < 1e-6).sum().item() if decoder_bias is not None else 0

print(f"\nNear-zero weights (< 1e-6): {near_zero_weight} / {decoder_weight.numel()}")
print(f"Near-zero biases (< 1e-6): {near_zero_bias} / {decoder_bias.numel() if decoder_bias is not None else 0}")

print()

# ============================================================
# SAVE DIAGNOSTIC REPORT
# ============================================================

print("Saving diagnostic report...")

report = {
    "investigation": "Low Prediction Variance in Attention Seq2Seq V2",
    "checkpoint_path": str(CHECKPOINT_PATH),
    "validation_sequences": int(len(context_before)),
    "prediction_statistics": {
        "mean": float(pred_mean),
        "overall_std": float(pred_std),
        "avg_std_across_timesteps": float(pred_std_across_timesteps),
        "avg_std_across_sensors": float(pred_std_across_sensors),
        "avg_std_across_samples": float(pred_std_across_samples)
    },
    "target_statistics": {
        "mean": float(target_mean),
        "overall_std": float(target_std),
        "avg_std_across_timesteps": float(target_std_across_timesteps),
        "avg_std_across_sensors": float(target_std_across_sensors),
        "avg_std_across_samples": float(target_std_across_samples)
    },
    "baseline_statistics": {
        "overall_std": float(baseline_std),
        "avg_std_across_timesteps": float(baseline_std_across_timesteps),
        "avg_std_across_sensors": float(baseline_std_across_sensors)
    },
    "per_sensor": {
        sensor: {
            "pred_std": float(per_sensor_pred_std[i]),
            "target_std": float(per_sensor_target_std[i]),
            "ratio": float(per_sensor_ratios[i])
        }
        for i, sensor in enumerate(SENSOR_COLUMNS)
    },
    "per_timestep": {
        f"timestep_{t+1}": {
            "pred_std": float(per_timestep_pred_std[t]),
            "target_std": float(per_timestep_target_std[t]),
            "ratio": float(per_timestep_ratios[t])
        }
        for t in range(10)
    },
    "model_inspection": {
        "decoder_weight_mean": float(decoder_weight.mean()),
        "decoder_weight_std": float(decoder_weight.std()),
        "decoder_weight_min": float(decoder_weight.min()),
        "decoder_weight_max": float(decoder_weight.max()),
        "near_zero_weights": near_zero_weight,
        "near_zero_biases": near_zero_bias
    },
    "findings": {
        "low_variance_confirmed": int(pred_std_across_timesteps < 0.001),
        "variance_ratio": float(pred_std / target_std),
        "baseline_variance_ratio": float(baseline_std / target_std)
    }
}

report_path = DIAGNOSTIC_DIR / "diagnostic_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"  Saved: {report_path}")
print()

print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
print(f"All outputs saved to: {DIAGNOSTIC_DIR}")
print()
print("SUMMARY:")
print(f"  Prediction std across timesteps: {pred_std_across_timesteps:.6f}")
print(f"  Target std across timesteps: {target_std_across_timesteps:.6f}")
print(f"  Ratio: {pred_std_across_timesteps / target_std_across_timesteps:.2%}")
print()
print(f"  Baseline std across timesteps: {baseline_std_across_timesteps:.6f}")
print(f"  Baseline ratio vs target: {baseline_std_across_timesteps / target_std_across_timesteps:.2%}")
