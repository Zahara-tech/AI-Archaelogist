"""
Diagnostic script to trace the reconstruction pipeline and identify scaling/inference issues.
"""

import sys
import numpy as np
import torch
import pickle
import json
from pathlib import Path

sys.path.append("src")
from attention_seq2seq_model import AttentionSeq2SeqReconstructor

# Paths
BENCHMARK_V2_PATH = Path("data/benchmark_v2")
MODEL_PATH = Path("models/attention_seq2seq_v2_best_model.pt")
SCALER_PATH = BENCHMARK_V2_PATH / "scaler.pkl"
MANIFEST_PATH = BENCHMARK_V2_PATH / "manifest.json"

DEVICE = torch.device("cpu")

print("="*70)
print("RECONSTRUCTION PIPELINE DIAGNOSTICS")
print("="*70)

# Load manifest
with open(MANIFEST_PATH, "r") as f:
    manifest = json.load(f)
sensor_columns = manifest["sensor_columns"]
print(f"\nSensor columns ({len(sensor_columns)}):")
for i, col in enumerate(sensor_columns):
    print(f"  {i}: {col}")

# Load scaler
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
print(f"\nScaler type: {type(scaler)}")
print(f"Scaler mean (first 5): {scaler.mean_[:5]}")
print(f"Scaler scale (first 5): {scaler.scale_[:5]}")

# Load model
print(f"\nLoading model from {MODEL_PATH}...")
model = AttentionSeq2SeqReconstructor(
    input_dim=17,
    hidden_dim=64,
    num_layers=2,
    output_dim=17,
    output_length=10,
    dropout=0.3,
    attention_dim=128
)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()
print("Model loaded successfully")

# Load a test sequence
test_file = BENCHMARK_V2_PATH / "test_01_M02_DC_test.npz"
print(f"\nLoading test file: {test_file.name}")
data = np.load(test_file)

# Use window index 5 (same as user's example)
window_index = 5
context_before = data["context_before"][window_index]  # (50, 17)
context_after = data["context_after"][window_index]    # (50, 17)
target = data["masked_segment"][window_index]         # (10, 17)

print(f"\nSequence shapes:")
print(f"  context_before: {context_before.shape}")
print(f"  context_after: {context_after.shape}")
print(f"  target (ground truth): {target.shape}")

print(f"\nData statistics (normalized, as stored in NPZ):")
print(f"  context_before min/max: {context_before.min():.6f} / {context_before.max():.6f}")
print(f"  context_after min/max: {context_after.min():.6f} / {context_after.max():.6f}")
print(f"  target min/max: {target.min():.6f} / {target.max():.6f}")

# Check IONGAUGEPRESSURE specifically (index 0)
sensor_idx = 0
sensor_name = sensor_columns[sensor_idx]
print(f"\n{sensor_name} (index {sensor_idx}) statistics:")
print(f"  context_before: min={context_before[:, sensor_idx].min():.8f}, max={context_before[:, sensor_idx].max():.8f}, mean={context_before[:, sensor_idx].mean():.8f}")
print(f"  context_after: min={context_after[:, sensor_idx].min():.8f}, max={context_after[:, sensor_idx].max():.8f}, mean={context_after[:, sensor_idx].mean():.8f}")
print(f"  target (actual): min={target[:, sensor_idx].min():.8f}, max={target[:, sensor_idx].max():.8f}, mean={target[:, sensor_idx].mean():.8f}")
print(f"  target values: {target[:, sensor_idx]}")

# Run model inference (as evaluation script does)
print(f"\nRunning model inference (evaluation mode)...")
context_before_tensor = torch.FloatTensor(context_before).unsqueeze(0).to(DEVICE)
context_after_tensor = torch.FloatTensor(context_after).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    predictions_norm = model(context_before_tensor, context_after_tensor, target=None, teacher_forcing_ratio=0.0)

predictions_norm = predictions_norm.squeeze(0).cpu().numpy()
print(f"Predictions shape: {predictions_norm.shape}")
print(f"Predictions min/max: {predictions_norm.min():.6f} / {predictions_norm.max():.6f}")

print(f"\n{sensor_name} predictions (normalized):")
print(f"  min={predictions_norm[:, sensor_idx].min():.8f}, max={predictions_norm[:, sensor_idx].max():.8f}, mean={predictions_norm[:, sensor_idx].mean():.8f}")
print(f"  values: {predictions_norm[:, sensor_idx]}")

# Linear interpolation baseline (as evaluation script does)
def linear_interpolation(before, after):
    # Add batch dimension for consistency
    if before.ndim == 2:
        before = before[np.newaxis, :, :]
    if after.ndim == 2:
        after = after[np.newaxis, :, :]
    left = before[:, -1:, :]
    right = after[:, 0:1, :]
    steps = np.arange(1, 11, dtype=np.float32).reshape(1, 10, 1)
    alpha = steps / 11.0
    result = left + alpha * (right - left)
    # Remove batch dimension if input was 2D
    if result.shape[0] == 1:
        result = result.squeeze(0)
    return result

baseline_pred = linear_interpolation(context_before, context_after)
print(f"\nBaseline predictions shape: {baseline_pred.shape}")
print(f"\n{sensor_name} baseline (normalized):")
print(f"  min={baseline_pred[:, sensor_idx].min():.8f}, max={baseline_pred[:, sensor_idx].max():.8f}, mean={baseline_pred[:, sensor_idx].mean():.8f}")
print(f"  values: {baseline_pred[:, sensor_idx]}")

# Calculate metrics on normalized data
neural_mse = float(np.mean((predictions_norm - target) ** 2))
neural_mae = float(np.mean(np.abs(predictions_norm - target)))
baseline_mse = float(np.mean((baseline_pred - target) ** 2))
baseline_mae = float(np.mean(np.abs(baseline_pred - target)))

print(f"\nMetrics on normalized data:")
print(f"  Neural MSE: {neural_mse:.8f}")
print(f"  Neural MAE: {neural_mae:.8f}")
print(f"  Baseline MSE: {baseline_mse:.8f}")
print(f"  Baseline MAE: {baseline_mae:.8f}")

# Per-sensor metrics for IONGAUGEPRESSURE
neural_sensor_mse = float(np.mean((predictions_norm[:, sensor_idx] - target[:, sensor_idx]) ** 2))
neural_sensor_mae = float(np.mean(np.abs(predictions_norm[:, sensor_idx] - target[:, sensor_idx])))
baseline_sensor_mse = float(np.mean((baseline_pred[:, sensor_idx] - target[:, sensor_idx]) ** 2))
baseline_sensor_mae = float(np.mean(np.abs(baseline_pred[:, sensor_idx] - target[:, sensor_idx])))

print(f"\n{sensor_name} metrics:")
print(f"  Neural MSE: {neural_sensor_mse:.8f}")
print(f"  Neural MAE: {neural_sensor_mae:.8f}")
print(f"  Baseline MSE: {baseline_sensor_mse:.8f}")
print(f"  Baseline MAE: {baseline_sensor_mae:.8f}")

# Check if data is actually normalized
print(f"\nChecking if data is normalized:")
print(f"  Context before mean (should be ~0 if normalized): {context_before.mean():.6f}")
print(f"  Context before std (should be ~1 if normalized): {context_before.std():.6f}")
print(f"  Target mean (should be ~0 if normalized): {target.mean():.6f}")
print(f"  Target std (should be ~1 if normalized): {target.std():.6f}")

# Test inverse scaling (should not be applied for benchmark data)
print(f"\nTesting inverse scaling (SHOULD NOT be applied to benchmark data):")
try:
    predictions_original = scaler.inverse_transform(predictions_norm)
    print(f"  Inverse scaled predictions shape: {predictions_original.shape}")
    print(f"  {sensor_name} inverse scaled: {predictions_original[:, sensor_idx]}")
    print(f"  WARNING: Inverse scaling was applied - this may be incorrect for benchmark data!")
except Exception as e:
    print(f"  Inverse scaling error: {e}")

print("\n" + "="*70)
print("DIAGNOSTICS COMPLETE")
print("="*70)
