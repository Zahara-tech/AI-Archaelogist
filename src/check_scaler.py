"""
Check scaler parameters and verify normalization
"""

import pickle
import numpy as np
from pathlib import Path

SCALER_PATH = Path("data/benchmark_v2/scaler.pkl")
BENCHMARK_V2_PATH = Path("data/benchmark_v2")

print("="*70)
print("SCALER VERIFICATION")
print("="*70)

# Load scaler
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

print(f"\nScaler type: {type(scaler)}")
print(f"Scaler mean (all 17 sensors):")
for i, (m, s) in enumerate(zip(scaler.mean_, scaler.scale_)):
    print(f"  {i}: mean={m:.6f}, scale={s:.6f}")

print(f"\nScaler fitted on: {scaler.n_features_in_} features")
print(f"Scaler n_samples_seen_: {scaler.n_samples_seen_}")

# Load a sample from training data to check
train_file = BENCHMARK_V2_PATH / "train_05_M02_DC_train.npz"
if train_file.exists():
    data = np.load(train_file)
    sample_context = data["context_before"][0]
    sample_target = data["masked_segment"][0]
    
    print(f"\nSample from {train_file.name}:")
    print(f"  context_before shape: {sample_context.shape}")
    print(f"  context_before mean: {sample_context.mean():.6f}")
    print(f"  context_before std: {sample_context.std():.6f}")
    print(f"  target mean: {sample_target.mean():.6f}")
    print(f"  target std: {sample_target.std():.6f}")
    
    # Try to inverse transform
    try:
        original_context = scaler.inverse_transform(sample_context)
        print(f"\nInverse transformed context:")
        print(f"  mean: {original_context.mean():.6f}")
        print(f"  std: {original_context.std():.6f}")
        print(f"  min: {original_context.min():.6f}")
        print(f"  max: {original_context.max():.6f}")
    except Exception as e:
        print(f"  Inverse transform error: {e}")

print("\n" + "="*70)
