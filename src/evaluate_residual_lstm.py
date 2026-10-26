"""
Evaluate Residual LSTM Model on Held-Out Test Set
Compares residual correction against linear interpolation and Seq2Seq.
"""

import os
import sys
import json
import glob
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import time

sys.path.append("src")
from residual_lstm_model import ResidualLSTM
from attention_seq2seq_model import AttentionSeq2SeqReconstructor

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent.parent
BENCHMARK_DIR = BASE_DIR / "data" / "benchmark_v2"
EXPERIMENT_DIR = BASE_DIR / "experiments" / "attention_seq2seq_v2_50epochs"
RESIDUAL_DIR = EXPERIMENT_DIR / "residual_experiment"

CHECKPOINT_PATH = RESIDUAL_DIR / "checkpoints" / "best_model.pt"
SEQ2SEQ_CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints" / "best_model.pt"
RESULTS_DIR = RESIDUAL_DIR / "results"

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
print("RESIDUAL LSTM EVALUATION")
print("=" * 70)
print(f"Device: {DEVICE}")
print()

# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("LOADING MODELS")
print("=" * 70)

# Load residual model
residual_model = ResidualLSTM(
    input_dim=17,
    hidden_dim=64,
    num_layers=2,
    output_dim=17,
    output_length=10,
    dropout=0.2
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
residual_model.load_state_dict(checkpoint["model_state_dict"])
residual_model.to(DEVICE)
residual_model.eval()

print(f"Residual model loaded (epoch {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.6f})")

# Load Seq2Seq model for comparison
seq2seq_model = AttentionSeq2SeqReconstructor(
    input_dim=17,
    hidden_dim=64,
    num_layers=2,
    output_dim=17,
    output_length=10,
    dropout=0.3,
    attention_dim=128,
)

seq2seq_checkpoint = torch.load(SEQ2SEQ_CHECKPOINT_PATH, map_location=DEVICE)
if isinstance(seq2seq_checkpoint, dict) and "model_state_dict" in seq2seq_checkpoint:
    seq2seq_model.load_state_dict(seq2seq_checkpoint["model_state_dict"])
else:
    seq2seq_model.load_state_dict(seq2seq_checkpoint)

seq2seq_model.to(DEVICE)
seq2seq_model.eval()

print("Seq2Seq model loaded")

# Load scaler
with open(BENCHMARK_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("Scaler loaded")
print()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def linear_interpolation(before, after):
    """Reconstruct the 10-point gap using boundary values."""
    left = before[:, -1:, :]
    right = after[:, 0:1, :]
    steps = np.arange(1, 11, dtype=np.float32).reshape(1, 10, 1) / 11.0
    return left + steps * (right - left)

def run_residual_model(before, after, batch_size=128):
    """Run residual model inference."""
    predictions = []
    
    # Compute interpolation for each batch
    with torch.no_grad():
        for start in range(0, len(before), batch_size):
            end = min(start + batch_size, len(before))
            
            before_batch = torch.tensor(before[start:end], dtype=torch.float32, device=DEVICE)
            after_batch = torch.tensor(after[start:end], dtype=torch.float32, device=DEVICE)
            
            # Compute interpolation
            left = before_batch[:, -1:, :]
            right = after_batch[:, 0:1, :]
            steps = torch.arange(1, 11, dtype=torch.float32, device=DEVICE).reshape(1, 10, 1) / 11.0
            interp_batch = left + steps * (right - left)
            
            # Predict residual
            pred_residual = residual_model(before_batch, after_batch, interp_batch)
            
            # Final prediction
            pred_final = interp_batch + pred_residual
            predictions.append(pred_final.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)

def run_seq2seq_model(before, after, batch_size=64):
    """Run Seq2Seq model inference."""
    predictions = []
    
    with torch.no_grad():
        for start in range(0, len(before), batch_size):
            end = min(start + batch_size, len(before))
            
            x_before = torch.tensor(before[start:end], dtype=torch.float32, device=DEVICE)
            x_after = torch.tensor(after[start:end], dtype=torch.float32, device=DEVICE)
            
            pred = seq2seq_model(x_before, x_after, teacher_forcing_ratio=0.0)
            predictions.append(pred.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)

def calculate_metrics(predictions, targets):
    """Calculate MSE, MAE, RMSE."""
    mse = float(np.mean((predictions - targets) ** 2))
    mae = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(mse))
    return mse, mae, rmse

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

def inverse_transform(predictions, scaler):
    """Inverse transform predictions from normalized to original units."""
    n_samples, n_timesteps, n_features = predictions.shape
    predictions_reshaped = predictions.reshape(-1, n_features)
    predictions_original = scaler.inverse_transform(predictions_reshaped)
    return predictions_original.reshape(n_samples, n_timesteps, n_features)

# ============================================================
# EVALUATE ON TEST SET
# ============================================================

print("=" * 70)
print("EVALUATING ON TEST SET")
print("=" * 70)

test_files = sorted(glob.glob(str(BENCHMARK_DIR / "test_*.npz")))

all_targets = []
all_interp = []
all_residual = []
all_seq2seq = []

per_file_results = []

for test_path in test_files:
    filename = os.path.basename(test_path)
    print(f"\nEvaluating: {filename}")
    
    data = np.load(test_path)
    before = data["context_before"].astype(np.float32)
    after = data["context_after"].astype(np.float32)
    target = data["masked_segment"].astype(np.float32)
    
    print(f"  Sequences: {len(target)}")
    
    # Linear interpolation
    start_time = time.time()
    interp_pred = linear_interpolation(before, after)
    interp_time = time.time() - start_time
    interp_mse, interp_mae, interp_rmse = calculate_metrics(interp_pred, target)
    print(f"  Linear Interpolation: MSE={interp_mse:.6f}, MAE={interp_mae:.6f}, RMSE={interp_rmse:.6f}, Time={interp_time:.3f}s")
    
    # Residual model
    start_time = time.time()
    residual_pred = run_residual_model(before, after)
    residual_time = time.time() - start_time
    residual_mse, residual_mae, residual_rmse = calculate_metrics(residual_pred, target)
    print(f"  Residual LSTM: MSE={residual_mse:.6f}, MAE={residual_mae:.6f}, RMSE={residual_rmse:.6f}, Time={residual_time:.3f}s")
    
    # Seq2Seq model
    start_time = time.time()
    seq2seq_pred = run_seq2seq_model(before, after)
    seq2seq_time = time.time() - start_time
    seq2seq_mse, seq2seq_mae, seq2seq_rmse = calculate_metrics(seq2seq_pred, target)
    print(f"  Seq2Seq: MSE={seq2seq_mse:.6f}, MAE={seq2seq_mae:.6f}, RMSE={seq2seq_rmse:.6f}, Time={seq2seq_time:.3f}s")
    
    # Per-sensor metrics for best method
    best_method = min([
        ("Linear Interpolation", interp_mse),
        ("Residual LSTM", residual_mse),
        ("Seq2Seq", seq2seq_mse)
    ], key=lambda x: x[1])[0]
    
    if best_method == "Linear Interpolation":
        best_pred = interp_pred
    elif best_method == "Residual LSTM":
        best_pred = residual_pred
    else:
        best_pred = seq2seq_pred
    
    per_sensor_mse, per_sensor_mae, per_sensor_rmse = calculate_per_sensor_metrics(
        best_pred, target, SENSOR_COLUMNS
    )
    
    per_file_results.append({
        "file": filename,
        "sequences": int(len(target)),
        "linear_interp": {"mse": interp_mse, "mae": interp_mae, "rmse": interp_rmse, "time": interp_time},
        "residual_lstm": {"mse": residual_mse, "mae": residual_mae, "rmse": residual_rmse, "time": residual_time},
        "seq2seq": {"mse": seq2seq_mse, "mae": seq2seq_mae, "rmse": seq2seq_rmse, "time": seq2seq_time},
        "best_method": best_method,
        "per_sensor_mse": per_sensor_mse,
        "per_sensor_mae": per_sensor_mae,
        "per_sensor_rmse": per_sensor_rmse
    })
    
    # Store for overall evaluation
    all_targets.append(target)
    all_interp.append(interp_pred)
    all_residual.append(residual_pred)
    all_seq2seq.append(seq2seq_pred)

# Concatenate all test data
all_targets = np.concatenate(all_targets, axis=0)
all_interp = np.concatenate(all_interp, axis=0)
all_residual = np.concatenate(all_residual, axis=0)
all_seq2seq = np.concatenate(all_seq2seq, axis=0)

print(f"\nTotal test sequences: {len(all_targets)}")

# ============================================================
# OVERALL METRICS
# ============================================================

print("\n" + "=" * 70)
print("OVERALL TEST METRICS (NORMALIZED)")
print("=" * 70)

interp_mse, interp_mae, interp_rmse = calculate_metrics(all_interp, all_targets)
residual_mse, residual_mae, residual_rmse = calculate_metrics(all_residual, all_targets)
seq2seq_mse, seq2seq_mae, seq2seq_rmse = calculate_metrics(all_seq2seq, all_targets)

print(f"\nLinear Interpolation: MSE={interp_mse:.6f}, MAE={interp_mae:.6f}, RMSE={interp_rmse:.6f}")
print(f"Residual LSTM: MSE={residual_mse:.6f}, MAE={residual_mae:.6f}, RMSE={residual_rmse:.6f}")
print(f"Seq2Seq: MSE={seq2seq_mse:.6f}, MAE={seq2seq_mae:.6f}, RMSE={seq2seq_rmse:.6f}")

# Improvement over interpolation
residual_improvement = (1 - residual_mse / interp_mse) * 100
seq2seq_improvement = (1 - seq2seq_mse / interp_mse) * 100

print(f"\nResidual LSTM vs Linear Interpolation: {residual_improvement:+.2f}% MSE")
print(f"Seq2Seq vs Linear Interpolation: {seq2seq_improvement:+.2f}% MSE")

# ============================================================
# PER-SENSOR AND PER-TIMESTEP METRICS
# ============================================================

print("\n" + "=" * 70)
print("DETAILED METRICS")
print("=" * 70)

# Determine best method overall
best_method = min([
    ("Linear Interpolation", interp_mse),
    ("Residual LSTM", residual_mse),
    ("Seq2Seq", seq2seq_mse)
], key=lambda x: x[1])[0]

print(f"\nBest method overall: {best_method}")

if best_method == "Linear Interpolation":
    best_pred = all_interp
elif best_method == "Residual LSTM":
    best_pred = all_residual
else:
    best_pred = all_seq2seq

per_sensor_mse, per_sensor_mae, per_sensor_rmse = calculate_per_sensor_metrics(
    best_pred, all_targets, SENSOR_COLUMNS
)

per_timestep_mse, per_timestep_mae, per_timestep_rmse = calculate_per_timestep_metrics(
    best_pred, all_targets
)

print("\nPer-sensor MSE:")
for sensor in SENSOR_COLUMNS:
    print(f"  {sensor}: {per_sensor_mse[sensor]:.6f}")

print("\nPer-timestep MSE:")
for t in range(10):
    print(f"  Timestep {t+1}: {per_timestep_mse[t]:.6f}")

# ============================================================
# ORIGINAL UNITS EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL UNITS EVALUATION")
print("=" * 70)

targets_original = inverse_transform(all_targets, scaler)
interp_original = inverse_transform(all_interp, scaler)
residual_original = inverse_transform(all_residual, scaler)
seq2seq_original = inverse_transform(all_seq2seq, scaler)

interp_mse_orig, interp_mae_orig, interp_rmse_orig = calculate_metrics(interp_original, targets_original)
residual_mse_orig, residual_mae_orig, residual_rmse_orig = calculate_metrics(residual_original, targets_original)
seq2seq_mse_orig, seq2seq_mae_orig, seq2seq_rmse_orig = calculate_metrics(seq2seq_original, targets_original)

print(f"\nLinear Interpolation (original): MSE={interp_mse_orig:.6f}, MAE={interp_mae_orig:.6f}, RMSE={interp_rmse_orig:.6f}")
print(f"Residual LSTM (original): MSE={residual_mse_orig:.6f}, MAE={residual_mae_orig:.6f}, RMSE={residual_rmse_orig:.6f}")
print(f"Seq2Seq (original): MSE={seq2seq_mse_orig:.6f}, MAE={seq2seq_mae_orig:.6f}, RMSE={seq2seq_rmse_orig:.6f}")

# ============================================================
# SAVE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results = {
    "experiment": "residual_lstm_evaluation",
    "residual_checkpoint": str(CHECKPOINT_PATH),
    "seq2seq_checkpoint": str(SEQ2SEQ_CHECKPOINT_PATH),
    "overall_metrics_normalized": {
        "linear_interp": {"mse": interp_mse, "mae": interp_mae, "rmse": interp_rmse},
        "residual_lstm": {"mse": residual_mse, "mae": residual_mae, "rmse": residual_rmse},
        "seq2seq": {"mse": seq2seq_mse, "mae": seq2seq_mae, "rmse": seq2seq_rmse}
    },
    "overall_metrics_original": {
        "linear_interp": {"mse": interp_mse_orig, "mae": interp_mae_orig, "rmse": interp_rmse_orig},
        "residual_lstm": {"mse": residual_mse_orig, "mae": residual_mae_orig, "rmse": residual_rmse_orig},
        "seq2seq": {"mse": seq2seq_mse_orig, "mae": seq2seq_mae_orig, "rmse": seq2seq_rmse_orig}
    },
    "improvement_vs_interpolation": {
        "residual_lstm": {"mse_pct": residual_improvement},
        "seq2seq": {"mse_pct": seq2seq_improvement}
    },
    "best_method": best_method,
    "per_file_results": per_file_results,
    "per_sensor_metrics": {
        "mse": per_sensor_mse,
        "mae": per_sensor_mae,
        "rmse": per_sensor_rmse
    },
    "per_timestep_metrics": {
        "mse": per_timestep_mse,
        "mae": per_timestep_mae,
        "rmse": per_timestep_rmse
    }
}

results_path = RESULTS_DIR / "evaluation_results.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"  Saved: {results_path}")

# Save CSV summary
csv_path = RESULTS_DIR / "metrics_summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Method", "MSE (Norm)", "MAE (Norm)", "RMSE (Norm)", "MSE (Orig)", "MAE (Orig)", "RMSE (Orig)"])
    writer.writerow(["Linear Interpolation", interp_mse, interp_mae, interp_rmse, interp_mse_orig, interp_mae_orig, interp_rmse_orig])
    writer.writerow(["Residual LSTM", residual_mse, residual_mae, residual_rmse, residual_mse_orig, residual_mae_orig, residual_rmse_orig])
    writer.writerow(["Seq2Seq", seq2seq_mse, seq2seq_mae, seq2seq_rmse, seq2seq_mse_orig, seq2seq_mae_orig, seq2seq_rmse_orig])
print(f"  Saved: {csv_path}")

print()

# ============================================================
# GENERATE PLOTS
# ============================================================

print("=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

# Plot 1: Overall MSE comparison
fig, ax = plt.subplots(figsize=(10, 6))
methods = ["Linear Interpolation", "Residual LSTM", "Seq2Seq"]
mses = [interp_mse, residual_mse, seq2seq_mse]
colors = ['green' if m == best_method else 'blue' for m in methods]

ax.bar(range(len(methods)), mses, color=colors, alpha=0.7)
ax.set_xlabel('Method')
ax.set_ylabel('MSE (Normalized)')
ax.set_title('Overall MSE Comparison')
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path_1 = RESULTS_DIR / "overall_mse_comparison.png"
plt.savefig(plot_path_1, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path_1}")
plt.close()

# Plot 2: Per-sensor MSE
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(SENSOR_COLUMNS))
width = 0.25

ax.bar(x - width, [interp_mse] * len(SENSOR_COLUMNS), width, label='Linear Interpolation', alpha=0.7, color='blue')
ax.bar(x, [residual_mse] * len(SENSOR_COLUMNS), width, label='Residual LSTM', alpha=0.7, color='green')
ax.bar(x + width, [seq2seq_mse] * len(SENSOR_COLUMNS), width, label='Seq2Seq', alpha=0.7, color='red')
ax.set_xlabel('Sensor')
ax.set_ylabel('MSE (Normalized)')
ax.set_title('Per-Sensor MSE Comparison')
ax.set_xticks(x)
ax.set_xticklabels(SENSOR_COLUMNS, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path_2 = RESULTS_DIR / "per_sensor_mse_comparison.png"
plt.savefig(plot_path_2, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path_2}")
plt.close()

# Plot 3: Per-timestep MSE
fig, ax = plt.subplots(figsize=(10, 6))
timesteps = np.arange(1, 11)
interp_timestep_mse, _, _ = calculate_per_timestep_metrics(all_interp, all_targets)
residual_timestep_mse, _, _ = calculate_per_timestep_metrics(all_residual, all_targets)
seq2seq_timestep_mse, _, _ = calculate_per_timestep_metrics(all_seq2seq, all_targets)

ax.plot(timesteps, interp_timestep_mse, 'o-', label='Linear Interpolation', linewidth=2, markersize=6, color='blue')
ax.plot(timesteps, residual_timestep_mse, 's-', label='Residual LSTM', linewidth=2, markersize=6, color='green')
ax.plot(timesteps, seq2seq_timestep_mse, '^-', label='Seq2Seq', linewidth=2, markersize=6, color='red')
ax.set_xlabel('Timestep')
ax.set_ylabel('MSE (Normalized)')
ax.set_title('Per-Timestep MSE Comparison')
ax.set_xticks(timesteps)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path_3 = RESULTS_DIR / "per_timestep_mse_comparison.png"
plt.savefig(plot_path_3, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path_3}")
plt.close()

print()

print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
print(f"Results saved to: {RESULTS_DIR}")
print()
