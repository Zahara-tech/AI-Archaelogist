import os
import json
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from attention_seq2seq_model import AttentionSeq2SeqReconstructor


# ============================================================
# Configuration
# ============================================================

SEQUENCE_FILE = "data/sequences/test_sequences.pkl"
MODEL_FILE = "models/attention_seq2seq_best_model.pt"

BASELINE_METRICS_FILE = "results/baseline_metrics.json"

OUTPUT_METRICS = "results/attention_seq2seq_metrics.json"
OUTPUT_SENSOR = "results/attention_seq2seq_per_sensor_metrics.csv"
OUTPUT_COMPARISON = "results/attention_seq2seq_comparison.json"

BATCH_SIZE = 64

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
    "ACTUALSTEPDURATION",
]


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("ATTENTION SEQ2SEQ TEST EVALUATION")
print("=" * 70)
print(f"Device: {device}")


# ============================================================
# Dataset
# ============================================================

class ReconstructionDataset(Dataset):

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        item = self.sequences[idx]

        context_before = np.asarray(
            item["context_before"],
            dtype=np.float32,
        )

        context_after = np.asarray(
            item["context_after"],
            dtype=np.float32,
        )

        target = np.asarray(
            item["masked_segment"],
            dtype=np.float32,
        )

        return (
            torch.from_numpy(context_before),
            torch.from_numpy(context_after),
            torch.from_numpy(target),
        )


# ============================================================
# Load test data
# ============================================================

print(f"\nLoading test sequences: {SEQUENCE_FILE}")

with open(SEQUENCE_FILE, "rb") as f:
    test_sequences = pickle.load(f)

print(
    f"Test sequences: {len(test_sequences):,}"
)

first = test_sequences[0]

print(
    "context_before:",
    np.asarray(first["context_before"]).shape,
)

print(
    "context_after:",
    np.asarray(first["context_after"]).shape,
)

print(
    "masked_segment:",
    np.asarray(first["masked_segment"]).shape,
)


test_dataset = ReconstructionDataset(
    test_sequences
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# Load checkpoint
# ============================================================

print(f"\nLoading model: {MODEL_FILE}")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
)

model = AttentionSeq2SeqReconstructor(
    input_dim=checkpoint["input_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    num_layers=checkpoint["num_layers"],
    output_dim=checkpoint["output_dim"],
    output_length=checkpoint["output_length"],
    dropout=checkpoint["dropout"],
    attention_dim=checkpoint["attention_dim"],
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    f"Checkpoint epoch: {checkpoint['epoch']}"
)

print(
    f"Checkpoint validation MSE: "
    f"{checkpoint['best_val_loss']:.6f}"
)


# ============================================================
# Test inference
# ============================================================

all_predictions = []
all_targets = []

print("\nRunning test inference...")

with torch.no_grad():

    for batch_idx, batch in enumerate(test_loader):

        context_before = batch[0].to(device)
        context_after = batch[1].to(device)
        target = batch[2].to(device)

        prediction = model(
            context_before,
            context_after,
            target=None,
            teacher_forcing_ratio=0.0,
        )

        all_predictions.append(
            prediction.cpu().numpy()
        )

        all_targets.append(
            target.cpu().numpy()
        )

        if (batch_idx + 1) % 50 == 0:

            print(
                f"Processed "
                f"{batch_idx + 1}/"
                f"{len(test_loader)} batches"
            )


predictions = np.concatenate(
    all_predictions,
    axis=0,
)

targets = np.concatenate(
    all_targets,
    axis=0,
)

print(
    "\nPrediction shape:",
    predictions.shape,
)

print(
    "Target shape:",
    targets.shape,
)


# ============================================================
# Overall metrics
# ============================================================

errors = predictions - targets

mse = float(
    np.mean(errors ** 2)
)

mae = float(
    np.mean(np.abs(errors))
)

prediction_variance = float(
    np.mean(np.var(predictions, axis=1))
)

target_variance = float(
    np.mean(np.var(targets, axis=1))
)


print("\nOverall test metrics:")
print(
    f"MSE: {mse:.6f}"
)

print(
    f"MAE: {mae:.6f}"
)

print(
    f"Prediction variance: "
    f"{prediction_variance:.6f}"
)

print(
    f"Target variance: "
    f"{target_variance:.6f}"
)


# ============================================================
# Per-sensor metrics
# ============================================================

sensor_results = []

for sensor_idx, sensor_name in enumerate(SENSOR_COLUMNS):

    sensor_error = (
        predictions[:, :, sensor_idx]
        - targets[:, :, sensor_idx]
    )

    sensor_mse = float(
        np.mean(sensor_error ** 2)
    )

    sensor_mae = float(
        np.mean(np.abs(sensor_error))
    )

    sensor_pred_var = float(
        np.var(predictions[:, :, sensor_idx])
    )

    sensor_target_var = float(
        np.var(targets[:, :, sensor_idx])
    )

    sensor_results.append(
        {
            "sensor": sensor_name,
            "mse": sensor_mse,
            "mae": sensor_mae,
            "prediction_variance": sensor_pred_var,
            "target_variance": sensor_target_var,
        }
    )

sensor_df = pd.DataFrame(
    sensor_results
)

print("\nPer-sensor test metrics:")
print(
    sensor_df.to_string(index=False)
)


# ============================================================
# Per-timestep metrics
# ============================================================

timestep_mse = []

timestep_mae = []

for timestep in range(
    predictions.shape[1]
):

    timestep_error = (
        predictions[:, timestep, :]
        - targets[:, timestep, :]
    )

    timestep_mse.append(
        float(np.mean(timestep_error ** 2))
    )

    timestep_mae.append(
        float(np.mean(np.abs(timestep_error)))
    )


print("\nPer-timestep MSE:")

for timestep, value in enumerate(
    timestep_mse
):

    print(
        f"t{timestep}: {value:.6f}"
    )


# ============================================================
# Load baseline result
# ============================================================

baseline_mse = None
baseline_mae = None

if os.path.exists(BASELINE_METRICS_FILE):

    with open(
        BASELINE_METRICS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        baseline_data = json.load(f)

    baseline_mse = baseline_data.get(
        "mse",
        baseline_data.get("overall_mse"),
    )

    baseline_mae = baseline_data.get(
        "mae",
        baseline_data.get("overall_mae"),
    )

    print("\nBaseline metrics found:")

    print(
        f"Baseline MSE: {baseline_mse}"
    )

    print(
        f"Baseline MAE: {baseline_mae}"
    )


# ============================================================
# Comparison
# ============================================================

comparison = {
    "linear_interpolation": {
        "mse": baseline_mse,
        "mae": baseline_mae,
    },
    "attention_seq2seq": {
        "mse": mse,
        "mae": mae,
    },
}

if baseline_mse is not None:

    comparison["attention_seq2seq"]["mse_ratio_vs_baseline"] = (
        mse / baseline_mse
    )

    comparison["attention_seq2seq"]["mse_improvement_percent_vs_baseline"] = (
        (baseline_mse - mse)
        / baseline_mse
        * 100.0
    )

if baseline_mae is not None:

    comparison["attention_seq2seq"]["mae_ratio_vs_baseline"] = (
        mae / baseline_mae
    )

    comparison["attention_seq2seq"]["mae_improvement_percent_vs_baseline"] = (
        (baseline_mae - mae)
        / baseline_mae
        * 100.0
    )


# ============================================================
# Save results
# ============================================================

os.makedirs(
    "results",
    exist_ok=True,
)

metrics = {
    "model": "attention_seq2seq",
    "test_sequences": len(test_sequences),
    "mse": mse,
    "mae": mae,
    "prediction_variance": prediction_variance,
    "target_variance": target_variance,
    "per_timestep_mse": timestep_mse,
    "per_timestep_mae": timestep_mae,
}

with open(
    OUTPUT_METRICS,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
    )


sensor_df.to_csv(
    OUTPUT_SENSOR,
    index=False,
)


with open(
    OUTPUT_COMPARISON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        comparison,
        f,
        indent=2,
    )


# ============================================================
# Final report
# ============================================================

print()
print("=" * 70)
print("TEST EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Attention Seq2Seq MSE: {mse:.6f}"
)

print(
    f"Attention Seq2Seq MAE: {mae:.6f}"
)

if baseline_mse is not None:

    print(
        f"Baseline MSE: {baseline_mse:.6f}"
    )

    print(
        f"MSE ratio vs baseline: "
        f"{mse / baseline_mse:.2f}x"
    )

if baseline_mae is not None:

    print(
        f"Baseline MAE: {baseline_mae:.6f}"
    )

    print(
        f"MAE ratio vs baseline: "
        f"{mae / baseline_mae:.2f}x"
    )

print()
print(
    f"Saved: {OUTPUT_METRICS}"
)

print(
    f"Saved: {OUTPUT_SENSOR}"
)

print(
    f"Saved: {OUTPUT_COMPARISON}"
)

print("=" * 70)
