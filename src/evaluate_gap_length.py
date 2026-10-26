"""
Evaluate All Methods for Different Gap Lengths
Compares linear interpolation, residual LSTM, and Seq2Seq across gap lengths 10, 20, 50.
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
GAP_BENCHMARK_BASE = BASE_DIR / "data" / "gap_length_benchmark"
EXPERIMENT_DIR = BASE_DIR / "experiments" / "gap_length_experiment"
SEQ2SEQ_CHECKPOINT_PATH = BASE_DIR / "experiments" / "attention_seq2seq_v2_50epochs" / "checkpoints" / "best_model.pt"

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

GAP_LENGTHS = [10, 20, 50]

print("=" * 70)
print("GAP LENGTH EVALUATION")
print("=" * 70)
print(f"Gap lengths: {GAP_LENGTHS}")
print(f"Device: {DEVICE}")
print()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def linear_interpolation(before, after, gap_length):
    """Reconstruct the gap using boundary values."""
    left = before[:, -1:, :]
    right = after[:, 0:1, :]
    steps = np.arange(1, gap_length + 1, dtype=np.float32).reshape(1, gap_length, 1) / (gap_length + 1)
    return left + steps * (right - left)

def run_residual_model(model, before, after, gap_length, batch_size=128):
    """Run residual model inference."""
    predictions = []
    
    with torch.no_grad():
        for start in range(0, len(before), batch_size):
            end = min(start + batch_size, len(before))
            
            before_batch = torch.tensor(before[start:end], dtype=torch.float32, device=DEVICE)
            after_batch = torch.tensor(after[start:end], dtype=torch.float32, device=DEVICE)
            
            # Compute interpolation
            left = before_batch[:, -1:, :]
            right = after_batch[:, 0:1, :]
            steps = torch.arange(1, gap_length + 1, dtype=torch.float32, device=DEVICE).reshape(1, gap_length, 1) / (gap_length + 1)
            interp_batch = left + steps * (right - left)
            
            # Predict residual
            pred_residual = model(before_batch, after_batch, interp_batch)
            
            # Final prediction
            pred_final = interp_batch + pred_residual
            predictions.append(pred_final.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)

def run_seq2seq_model(model, before, after, gap_length, batch_size=64):
    """Run Seq2Seq model inference (only for gap length 10)."""
    if gap_length != 10:
        return None
    
    predictions = []
    
    with torch.no_grad():
        for start in range(0, len(before), batch_size):
            end = min(start + batch_size, len(before))
            
            x_before = torch.tensor(before[start:end], dtype=torch.float32, device=DEVICE)
            x_after = torch.tensor(after[start:end], dtype=torch.float32, device=DEVICE)
            
            pred = model(x_before, x_after, teacher_forcing_ratio=0.0)
            predictions.append(pred.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)

def calculate_metrics(predictions, targets):
    """Calculate MSE, MAE, RMSE."""
    mse = float(np.mean((predictions - targets) ** 2))
    mae = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(mse))
    return mse, mae, rmse

def calculate_per_position_metrics(predictions, targets):
    """Calculate MSE and MAE per gap position."""
    n_positions = predictions.shape[1]
    per_position_mse = []
    per_position_mae = []
    per_position_rmse = []
    
    for t in range(n_positions):
        pred_t = predictions[:, t, :]
        target_t = targets[:, t, :]
        
        mse = float(np.mean((pred_t - target_t) ** 2))
        mae = float(np.mean(np.abs(pred_t - target_t)))
        rmse = float(np.sqrt(mse))
        
        per_position_mse.append(mse)
        per_position_mae.append(mae)
        per_position_rmse.append(rmse)
    
    return per_position_mse, per_position_mae, per_position_rmse

def inverse_transform(predictions, scaler):
    """Inverse transform predictions from normalized to original units."""
    n_samples, n_timesteps, n_features = predictions.shape
    predictions_reshaped = predictions.reshape(-1, n_features)
    predictions_original = scaler.inverse_transform(predictions_reshaped)
    return predictions_original.reshape(n_samples, n_timesteps, n_features)

# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("LOADING MODELS")
print("=" * 70)

# Load Seq2Seq model (only for gap length 10)
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
print("Seq2Seq model loaded (gap length 10 only)")

# Load residual models for each gap length
residual_models = {}
for gap_length in GAP_LENGTHS:
    checkpoint_path = EXPERIMENT_DIR / f"gap_{gap_length}" / "checkpoints" / "best_model.pt"
    model = ResidualLSTM(
        input_dim=17,
        hidden_dim=64,
        num_layers=2,
        output_dim=17,
        output_length=gap_length,
        dropout=0.2
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    
    residual_models[gap_length] = model
    print(f"Residual LSTM loaded for gap length {gap_length}")

print()

# ============================================================
# EVALUATE FOR EACH GAP LENGTH
# ============================================================

all_results = {}

for gap_length in GAP_LENGTHS:
    print("=" * 70)
    print(f"EVALUATING GAP LENGTH: {gap_length}")
    print("=" * 70)
    
    gap_dir = GAP_BENCHMARK_BASE / f"gap_{gap_length}"
    
    # Load scaler
    with open(gap_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Load test files
    test_files = sorted(glob.glob(str(gap_dir / "test_*.npz")))
    
    all_targets = []
    all_interp = []
    all_residual = []
    all_seq2seq = [] if gap_length == 10 else None
    
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
        interp_pred = linear_interpolation(before, after, gap_length)
        interp_time = time.time() - start_time
        interp_mse, interp_mae, interp_rmse = calculate_metrics(interp_pred, target)
        print(f"  Linear Interpolation: MSE={interp_mse:.6f}, MAE={interp_mae:.6f}, RMSE={interp_rmse:.6f}, Time={interp_time:.3f}s")
        
        # Residual model
        start_time = time.time()
        residual_pred = run_residual_model(residual_models[gap_length], before, after, gap_length)
        residual_time = time.time() - start_time
        residual_mse, residual_mae, residual_rmse = calculate_metrics(residual_pred, target)
        print(f"  Residual LSTM: MSE={residual_mse:.6f}, MAE={residual_mae:.6f}, RMSE={residual_rmse:.6f}, Time={residual_time:.3f}s")
        
        # Seq2Seq model (only for gap length 10)
        seq2seq_mse, seq2seq_mae, seq2seq_rmse = None, None, None
        seq2seq_time = None
        if gap_length == 10:
            start_time = time.time()
            seq2seq_pred = run_seq2seq_model(seq2seq_model, before, after, gap_length)
            seq2seq_time = time.time() - start_time
            seq2seq_mse, seq2seq_mae, seq2seq_rmse = calculate_metrics(seq2seq_pred, target)
            print(f"  Seq2Seq: MSE={seq2seq_mse:.6f}, MAE={seq2seq_mae:.6f}, RMSE={seq2seq_rmse:.6f}, Time={seq2seq_time:.3f}s")
            all_seq2seq.append(seq2seq_pred)
        
        # Per-position metrics for best method
        methods = {
            "Linear Interpolation": (interp_mse, interp_pred),
            "Residual LSTM": (residual_mse, residual_pred)
        }
        if gap_length == 10:
            methods["Seq2Seq"] = (seq2seq_mse, seq2seq_pred)
        
        best_method = min(methods.keys(), key=lambda k: methods[k][0])
        best_pred = methods[best_method][1]
        
        per_position_mse, per_position_mae, per_position_rmse = calculate_per_position_metrics(
            best_pred, target
        )
        
        per_file_results.append({
            "file": filename,
            "sequences": int(len(target)),
            "linear_interp": {"mse": interp_mse, "mae": interp_mae, "rmse": interp_rmse, "time": interp_time},
            "residual_lstm": {"mse": residual_mse, "mae": residual_mae, "rmse": residual_rmse, "time": residual_time},
            "seq2seq": {"mse": seq2seq_mse, "mae": seq2seq_mae, "rmse": seq2seq_rmse, "time": seq2seq_time} if gap_length == 10 else None,
            "best_method": best_method,
            "per_position_mse": per_position_mse,
            "per_position_mae": per_position_mae,
            "per_position_rmse": per_position_rmse
        })
        
        # Store for overall evaluation
        all_targets.append(target)
        all_interp.append(interp_pred)
        all_residual.append(residual_pred)
    
    # Concatenate all test data
    all_targets = np.concatenate(all_targets, axis=0)
    all_interp = np.concatenate(all_interp, axis=0)
    all_residual = np.concatenate(all_residual, axis=0)
    if gap_length == 10:
        all_seq2seq = np.concatenate(all_seq2seq, axis=0)
    
    print(f"\nTotal test sequences: {len(all_targets)}")
    
    # Overall metrics
    interp_mse, interp_mae, interp_rmse = calculate_metrics(all_interp, all_targets)
    residual_mse, residual_mae, residual_rmse = calculate_metrics(all_residual, all_targets)
    
    print(f"\nOverall Test Metrics (Normalized):")
    print(f"  Linear Interpolation: MSE={interp_mse:.6f}, MAE={interp_mae:.6f}, RMSE={interp_rmse:.6f}")
    print(f"  Residual LSTM: MSE={residual_mse:.6f}, MAE={residual_mae:.6f}, RMSE={residual_rmse:.6f}")
    
    if gap_length == 10:
        seq2seq_mse, seq2seq_mae, seq2seq_rmse = calculate_metrics(all_seq2seq, all_targets)
        print(f"  Seq2Seq: MSE={seq2seq_mse:.6f}, MAE={seq2seq_mae:.6f}, RMSE={seq2seq_rmse:.6f}")
    
    # Original units
    targets_original = inverse_transform(all_targets, scaler)
    interp_original = inverse_transform(all_interp, scaler)
    residual_original = inverse_transform(all_residual, scaler)
    
    interp_mse_orig, interp_mae_orig, interp_rmse_orig = calculate_metrics(interp_original, targets_original)
    residual_mse_orig, residual_mae_orig, residual_rmse_orig = calculate_metrics(residual_original, targets_original)
    
    print(f"\nOverall Test Metrics (Original Units):")
    print(f"  Linear Interpolation: MSE={interp_mse_orig:.6f}, MAE={interp_mae_orig:.6f}, RMSE={interp_rmse_orig:.6f}")
    print(f"  Residual LSTM: MSE={residual_mse_orig:.6f}, MAE={residual_mae_orig:.6f}, RMSE={residual_rmse_orig:.6f}")
    
    if gap_length == 10:
        seq2seq_original = inverse_transform(all_seq2seq, scaler)
        seq2seq_mse_orig, seq2seq_mae_orig, seq2seq_rmse_orig = calculate_metrics(seq2seq_original, targets_original)
        print(f"  Seq2Seq: MSE={seq2seq_mse_orig:.6f}, MAE={seq2seq_mae_orig:.6f}, RMSE={seq2seq_rmse_orig:.6f}")
    
    # Store results
    all_results[gap_length] = {
        "overall_normalized": {
            "linear_interp": {"mse": interp_mse, "mae": interp_mae, "rmse": interp_rmse},
            "residual_lstm": {"mse": residual_mse, "mae": residual_mae, "rmse": residual_rmse},
            "seq2seq": {"mse": seq2seq_mse, "mae": seq2seq_mae, "rmse": seq2seq_rmse} if gap_length == 10 else None
        },
        "overall_original": {
            "linear_interp": {"mse": interp_mse_orig, "mae": interp_mae_orig, "rmse": interp_rmse_orig},
            "residual_lstm": {"mse": residual_mse_orig, "mae": residual_mae_orig, "rmse": residual_rmse_orig},
            "seq2seq": {"mse": seq2seq_mse_orig, "mae": seq2seq_mae_orig, "rmse": seq2seq_rmse_orig} if gap_length == 10 else None
        },
        "per_file_results": per_file_results,
        "total_sequences": int(len(all_targets))
    }
    
    print()

# ============================================================
# SAVE RESULTS
# ============================================================

print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results_path = EXPERIMENT_DIR / "evaluation_results.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)
print(f"  Saved: {results_path}")

print()

# ============================================================
# GENERATE COMPARISON PLOTS
# ============================================================

print("=" * 70)
print("GENERATING COMPARISON PLOTS")
print("=" * 70)

# Plot 1: Overall MSE comparison across gap lengths
fig, ax = plt.subplots(figsize=(10, 6))
gap_labels = [str(g) for g in GAP_LENGTHS]
interp_mses = [all_results[g]["overall_normalized"]["linear_interp"]["mse"] for g in GAP_LENGTHS]
residual_mses = [all_results[g]["overall_normalized"]["residual_lstm"]["mse"] for g in GAP_LENGTHS]

x = np.arange(len(GAP_LENGTHS))
width = 0.35

ax.bar(x - width/2, interp_mses, width, label='Linear Interpolation', alpha=0.7, color='blue')
ax.bar(x + width/2, residual_mses, width, label='Residual LSTM', alpha=0.7, color='green')
ax.set_xlabel('Gap Length (timesteps)')
ax.set_ylabel('MSE (Normalized)')
ax.set_title('Overall MSE Comparison Across Gap Lengths')
ax.set_xticks(x)
ax.set_xticklabels(gap_labels)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = EXPERIMENT_DIR / "mse_comparison_by_gap_length.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path}")
plt.close()

# Plot 2: Improvement over interpolation
fig, ax = plt.subplots(figsize=(10, 6))
improvements = [(residual_mses[i] - interp_mses[i]) / interp_mses[i] * 100 for i in range(len(GAP_LENGTHS))]
colors = ['green' if imp < 0 else 'red' for imp in improvements]

ax.bar(gap_labels, improvements, color=colors, alpha=0.7)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Gap Length (timesteps)')
ax.set_ylabel('MSE Change vs Linear Interpolation (%)')
ax.set_title('Residual LSTM Performance vs Linear Interpolation')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = EXPERIMENT_DIR / "improvement_vs_interpolation.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path}")
plt.close()

# Plot 3: Per-position error for each gap length
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for idx, gap_length in enumerate(GAP_LENGTHS):
    ax = axes[idx]
    
    # Get per-position metrics from first test file as representative
    per_file = all_results[gap_length]["per_file_results"][0]
    positions = list(range(1, gap_length + 1))
    
    interp_pos_mse = []
    residual_pos_mse = []
    
    # Re-calculate per-position metrics for overall data
    gap_dir = GAP_BENCHMARK_BASE / f"gap_{gap_length}"
    test_files = sorted(glob.glob(str(gap_dir / "test_*.npz")))
    
    all_targets = []
    all_interp = []
    all_residual = []
    
    for test_path in test_files:
        data = np.load(test_path)
        before = data["context_before"].astype(np.float32)
        after = data["context_after"].astype(np.float32)
        target = data["masked_segment"].astype(np.float32)
        
        interp_pred = linear_interpolation(before, after, gap_length)
        residual_pred = run_residual_model(residual_models[gap_length], before, after, gap_length)
        
        all_targets.append(target)
        all_interp.append(interp_pred)
        all_residual.append(residual_pred)
    
    all_targets = np.concatenate(all_targets, axis=0)
    all_interp = np.concatenate(all_interp, axis=0)
    all_residual = np.concatenate(all_residual, axis=0)
    
    interp_pos_mse, _, _ = calculate_per_position_metrics(all_interp, all_targets)
    residual_pos_mse, _, _ = calculate_per_position_metrics(all_residual, all_targets)
    
    ax.plot(positions, interp_pos_mse, 'o-', label='Linear Interpolation', linewidth=2, markersize=6, color='blue')
    ax.plot(positions, residual_pos_mse, 's-', label='Residual LSTM', linewidth=2, markersize=6, color='green')
    ax.set_xlabel('Gap Position')
    ax.set_ylabel('MSE (Normalized)')
    ax.set_title(f'Gap Length: {gap_length}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = EXPERIMENT_DIR / "per_position_error_by_gap_length.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path}")
plt.close()

print()

print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
print(f"Results saved to: {EXPERIMENT_DIR}")
print()
