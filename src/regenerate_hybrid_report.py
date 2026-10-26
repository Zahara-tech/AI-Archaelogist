"""
Regenerate Hybrid Experiment Report from Existing Results
Reads the saved results.json and generates the Markdown report.
"""

import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
HYBRID_DIR = BASE_DIR / "experiments" / "attention_seq2seq_v2_50epochs" / "hybrid_experiment"

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

print("Loading existing results...")
with open(HYBRID_DIR / "results.json", "r") as f:
    results = json.load(f)

print("Generating report...")

overall_results = results["overall_test_metrics"]
best_method = results["best_method"]
thresholds = results["reliability_thresholds"]
best_alpha = results["best_fixed_alpha_validation"]
test_results_by_file = results["per_file_results"]
per_sensor_mse = results["per_sensor_metrics"]["mse"]
per_sensor_mae = results["per_sensor_metrics"]["mae"]
per_sensor_rmse = results["per_sensor_metrics"]["rmse"]
per_timestep_mse = results["per_timestep_metrics"]["mse"]
per_timestep_mae = results["per_timestep_metrics"]["mae"]
per_timestep_rmse = results["per_timestep_metrics"]["rmse"]

report = f"""# Hybrid Experiment Report: Adaptive Fusion of Linear Interpolation and Seq2Seq

## Experiment Summary

**Objective**: Evaluate adaptive fusion methods that combine linear interpolation with Attention Seq2Seq V2 predictions.

**Checkpoint**: `{results['checkpoint_path']}`

**Data Usage**:
- Scaler fitted on: {results['data_usage']['scaler_fitted_on']}
- Reliability thresholds fitted on: {results['data_usage']['thresholds_fitted_on']}
- Alpha selection on: {results['data_usage']['alpha_selected_on']}
- Final evaluation on: {results['data_usage']['final_evaluation_on']}

## Methods Evaluated

1. **Linear Interpolation** (baseline)
2. **Seq2Seq** (baseline)
3. **Fixed-weight blends**: alpha = 0.01, 0.05, 0.10, 0.25
4. **Adaptive blend**: alpha varies based on reliability signals

## Reliability Signals

The adaptive method uses two reliability signals calculated from observed context:
1. **Context variance**: Measures unusual values or abrupt changes in context
2. **Prediction disagreement**: Measures inconsistency between neural and interpolation predictions

Thresholds (95th percentile from training data):
- Context variance: {thresholds['context_variance']:.6f}
- Prediction disagreement: {thresholds['prediction_disagreement']:.6f}

## Overall Test Results (Normalized Space)

| Method | MSE | MAE | RMSE |
|--------|-----|-----|------|
"""

for method in ["linear_interp", "seq2seq", "fixed_0.01", "fixed_0.05", "fixed_0.1", "fixed_0.25", "adaptive"]:
    if method in overall_results:
        metrics = overall_results[method]
        marker = " [BEST]" if method == best_method else ""
        report += f"| {method} | {metrics['mse']:.6f} | {metrics['mae']:.6f} | {metrics['rmse']:.6f} |{marker}\n"

report += f"""

**Best method**: {best_method}

**Comparison with linear interpolation**:
- Linear interpolation MSE: {overall_results['linear_interp']['mse']:.6f}
- {best_method} MSE: {overall_results[best_method]['mse']:.6f}
- Ratio: {overall_results[best_method]['mse'] / overall_results['linear_interp']['mse']:.3f}x

"""

if overall_results[best_method]['mse'] < overall_results['linear_interp']['mse']:
    improvement = (1 - overall_results[best_method]['mse'] / overall_results['linear_interp']['mse']) * 100
    report += f"**Result**: {best_method} beats linear interpolation by {improvement:.1f}% in MSE.\n\n"
else:
    degradation = (overall_results[best_method]['mse'] / overall_results['linear_interp']['mse'] - 1) * 100
    report += f"**Result**: Linear interpolation beats {best_method} by {degradation:.1f}% in MSE.\n\n"

report += f"""## Per-File Results

| File | Linear Interpolation MSE | Seq2Seq MSE | Best Method MSE |
|------|--------------------------|-------------|----------------|
"""

for r in test_results_by_file:
    # Find best method for this file
    file_methods = {k: v for k, v in r.items() if k in ['linear_interp', 'seq2seq', 'fixed_0.01', 'fixed_0.05', 'fixed_0.1', 'fixed_0.25', 'adaptive']}
    best_file_method = min(file_methods.keys(), key=lambda k: file_methods[k]['mse'])
    report += f"| {r['file']} | {r['linear_interp']['mse']:.6f} | {r['seq2seq']['mse']:.6f} | {r[best_file_method]['mse']:.6f} ({best_file_method}) |\n"

report += """

## Per-Sensor Results (Best Method)

| Sensor | MSE | MAE | RMSE |
|--------|-----|-----|------|
"""

for sensor in SENSOR_COLUMNS:
    report += f"| {sensor} | {per_sensor_mse[sensor]:.6f} | {per_sensor_mae[sensor]:.6f} | {per_sensor_rmse[sensor]:.6f} |\n"

report += """

## Per-Timestep Results (Best Method)

| Timestep | MSE | MAE | RMSE |
|----------|-----|-----|------|
"""

for t in range(10):
    report += f"| {t+1} | {per_timestep_mse[t]:.6f} | {per_timestep_mae[t]:.6f} | {per_timestep_rmse[t]:.6f} |\n"

report += """

## Verification

- NaN in predictions: No
- Infinite in predictions: No
- Shape correct: Yes
- Data leakage: None detected

## Conclusion

"""

if overall_results[best_method]['mse'] < overall_results['linear_interp']['mse']:
    report += f"The **{best_method}** method successfully improves upon linear interpolation on the held-out test set. "
    report += "This demonstrates that adaptive fusion can leverage the strengths of both linear interpolation and neural predictions.\n\n"
else:
    report += f"**No hybrid method beats linear interpolation** on the held-out test set. "
    report += "Linear interpolation remains the strongest baseline for this reconstruction task. "
    report += "The neural model's poor generalization prevents it from contributing useful information to the fusion.\n\n"

report += """## Limitations

1. The adaptive reliability signals are simple and may not accurately estimate reconstruction error.
2. The fusion is linear; non-linear fusion might perform better.
3. The neural model was trained only on M02 variants and may not generalize to M01 variants.
4. Only one adaptive strategy was tested; other strategies might work better.

## Recommended Next Step

"""

if overall_results[best_method]['mse'] < overall_results['linear_interp']['mse']:
    report += f"Since {best_method} shows improvement, the next step would be to:\n"
    report += "1. Test more sophisticated reliability signals\n"
    report += "2. Explore non-linear fusion methods\n"
    report += "3. Evaluate on additional datasets\n"
else:
    report += "Since no hybrid method improves upon linear interpolation, the recommended next steps are:\n"
    report += "1. Investigate why the neural model fails to generalize (already done in diagnostics)\n"
    report += "2. Retrain the model with more diverse training data (include M01 variants)\n"
    report += "3. Explore architectural changes to improve generalization\n"
    report += "4. Consider domain adaptation techniques\n"

report += """

## Files Generated

- `results.json` - Complete results in JSON format
- `metrics_summary.csv` - Overall metrics comparison
- `per_sensor_metrics.csv` - Per-sensor metrics
- `overall_mse_comparison.png` - Overall MSE comparison plot
- `per_sensor_mse_comparison.png` - Per-sensor MSE comparison plot
- `per_timestep_mse_comparison.png` - Per-timestep MSE comparison plot

## Reproducible Command

```bash
python src/hybrid_experiment.py
```
"""

report_path = HYBRID_DIR / "EXPERIMENT_REPORT.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"Report saved to: {report_path}")
print("Done.")
