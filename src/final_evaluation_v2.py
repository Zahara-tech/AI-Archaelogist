import os
import json
import glob
import pickle
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

from attention_seq2seq_model import AttentionSeq2SeqReconstructor


# ============================================================
# AI ARCHAEOLOGIST - FINAL V2 EVALUATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(BASE_DIR, "data", "benchmark_v2")
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "attention_seq2seq_v2_best_model.pt"
)
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cpu")

print("=" * 70)
print("AI ARCHAEOLOGIST - FINAL V2 EVALUATION")
print("=" * 70)
print("Device:", DEVICE)
print("Benchmark:", BENCHMARK_DIR)
print("Model:", MODEL_PATH)
print()


# ------------------------------------------------------------
# Find test files
# ------------------------------------------------------------

test_files = sorted(
    glob.glob(os.path.join(BENCHMARK_DIR, "test_*.npz"))
)

if not test_files:
    raise FileNotFoundError(
        "No test_*.npz files found in data/benchmark_v2"
    )

print("Test files found:", len(test_files))
for f in test_files:
    print("  -", os.path.basename(f))
print()


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("Loading Attention Seq2Seq v2...")

model = AttentionSeq2SeqReconstructor(
    input_dim=17,
    hidden_dim=64,
    num_layers=2,
    output_dim=17,
    output_length=10,
    dropout=0.3,
    attention_dim=128,
)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print()


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def linear_interpolation(before, after):
    """
    Reconstruct the 10-point gap using the boundary values
    immediately before and after the gap.
    """
    left = before[:, -1:, :]
    right = after[:, 0:1, :]

    steps = np.arange(1, 11, dtype=np.float32).reshape(1, 10, 1)
    alpha = steps / 11.0

    return left + alpha * (right - left)


def evaluate_model(before, after, target, batch_size=64):
    predictions = []

    with torch.no_grad():
        for start in range(0, len(before), batch_size):
            end = min(start + batch_size, len(before))

            x_before = torch.tensor(
                before[start:end],
                dtype=torch.float32,
                device=DEVICE
            )

            x_after = torch.tensor(
                after[start:end],
                dtype=torch.float32,
                device=DEVICE
            )

            pred = model(
                x_before,
                x_after,
                teacher_forcing_ratio=0.0
            )

            predictions.append(
                pred.cpu().numpy()
            )

    predictions = np.concatenate(predictions, axis=0)

    mse = float(np.mean((predictions - target) ** 2))
    mae = float(np.mean(np.abs(predictions - target)))

    return predictions, mse, mae


# ------------------------------------------------------------
# Evaluate every test file
# ------------------------------------------------------------

all_targets = []
all_baseline_predictions = []
all_model_predictions = []

results = []

for test_path in test_files:

    filename = os.path.basename(test_path)

    print("-" * 70)
    print("Evaluating:", filename)

    data = np.load(test_path)

    before = data["context_before"].astype(np.float32)
    after = data["context_after"].astype(np.float32)
    target = data["masked_segment"].astype(np.float32)

    print("Sequences:", len(target))
    print("Before:", before.shape)
    print("After :", after.shape)
    print("Target:", target.shape)

    # --------------------------------------------------------
    # Linear interpolation baseline
    # --------------------------------------------------------

    baseline_pred = linear_interpolation(before, after)

    baseline_mse = float(
        np.mean((baseline_pred - target) ** 2)
    )

    baseline_mae = float(
        np.mean(np.abs(baseline_pred - target))
    )

    # --------------------------------------------------------
    # Attention Seq2Seq v2
    # --------------------------------------------------------

    model_pred, model_mse, model_mae = evaluate_model(
        before,
        after,
        target
    )

    # --------------------------------------------------------
    # Ratio
    # --------------------------------------------------------

    mse_ratio = (
        model_mse / baseline_mse
        if baseline_mse > 0 else float("inf")
    )

    mae_ratio = (
        model_mae / baseline_mae
        if baseline_mae > 0 else float("inf")
    )

    result = {
        "file": filename,
        "sequences": int(len(target)),
        "baseline_mse": baseline_mse,
        "baseline_mae": baseline_mae,
        "attention_v2_mse": model_mse,
        "attention_v2_mae": model_mae,
        "mse_ratio_vs_baseline": mse_ratio,
        "mae_ratio_vs_baseline": mae_ratio
    }

    results.append(result)

    all_targets.append(target)
    all_baseline_predictions.append(baseline_pred)
    all_model_predictions.append(model_pred)

    print()
    print("Linear interpolation:")
    print("  MSE:", baseline_mse)
    print("  MAE:", baseline_mae)

    print("Attention Seq2Seq v2:")
    print("  MSE:", model_mse)
    print("  MAE:", model_mae)

    print("Attention / baseline:")
    print("  MSE ratio:", mse_ratio)
    print("  MAE ratio:", mae_ratio)
    print()


# ------------------------------------------------------------
# Overall evaluation
# ------------------------------------------------------------

print("=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

all_targets = np.concatenate(all_targets, axis=0)
all_baseline_predictions = np.concatenate(
    all_baseline_predictions, axis=0
)
all_model_predictions = np.concatenate(
    all_model_predictions, axis=0
)

overall_baseline_mse = float(
    np.mean((all_baseline_predictions - all_targets) ** 2)
)

overall_baseline_mae = float(
    np.mean(np.abs(all_baseline_predictions - all_targets))
)

overall_model_mse = float(
    np.mean((all_model_predictions - all_targets) ** 2)
)

overall_model_mae = float(
    np.mean(np.abs(all_model_predictions - all_targets))
)

overall_mse_ratio = (
    overall_model_mse / overall_baseline_mse
)

overall_mae_ratio = (
    overall_model_mae / overall_baseline_mae
)

print()
print("TOTAL TEST SEQUENCES:", len(all_targets))
print()

print("LINEAR INTERPOLATION")
print("  MSE:", overall_baseline_mse)
print("  MAE:", overall_baseline_mae)

print()

print("ATTENTION SEQ2SEQ V2")
print("  MSE:", overall_model_mse)
print("  MAE:", overall_model_mae)

print()

print("COMPARISON")
print("  MSE ratio:", overall_mse_ratio)
print("  MAE ratio:", overall_mae_ratio)

if overall_model_mse < overall_baseline_mse:
    print()
    print("RESULT: Attention Seq2Seq v2 BEATS linear interpolation.")
else:
    print()
    print("RESULT: Linear interpolation BEATS Attention Seq2Seq v2.")

print()


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

output = {
    "protocol": {
        "context_before": 50,
        "gap_length": 10,
        "context_after": 50,
        "sensor_count": 17,
        "run_aware": True,
        "test_files": len(test_files)
    },
    "per_file": results,
    "overall": {
        "test_sequences": int(len(all_targets)),
        "baseline_mse": overall_baseline_mse,
        "baseline_mae": overall_baseline_mae,
        "attention_v2_mse": overall_model_mse,
        "attention_v2_mae": overall_model_mae,
        "mse_ratio_vs_baseline": overall_mse_ratio,
        "mae_ratio_vs_baseline": overall_mae_ratio
    }
}

json_path = os.path.join(
    RESULTS_DIR,
    "final_evaluation_v2.json"
)

with open(json_path, "w") as f:
    json.dump(output, f, indent=2)

np.savez(
    os.path.join(RESULTS_DIR, "final_predictions_v2.npz"),
    targets=all_targets,
    baseline_predictions=all_baseline_predictions,
    attention_v2_predictions=all_model_predictions
)

print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
print("Saved:", json_path)
print(
    "Saved:",
    os.path.join(RESULTS_DIR, "final_predictions_v2.npz")
)
