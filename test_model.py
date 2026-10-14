"""
Test Model Script
Loads trained model and runs inference on held-out test sequences.
Prints detailed predictions and generates reconstruction plots.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json

sys.path.append("src")
from data_loader import PHM2018Loader
from experiment_generator import MaskedSequenceGenerator
from bilstm_model import BiLSTMReconstructor, BiLSTMModelTrainer, BiLSTMDataset
from torch.utils.data import DataLoader


def test_model_inference():
    """Load trained model and test on held-out sequences."""
    
    print("="*70)
    print("MODEL INFERENCE TEST ON HELD-OUT SEQUENCES")
    print("="*70)
    
    # Load data
    loader = PHM2018Loader()
    preprocessed = loader.load_preprocessed(Path("data/preprocessed/phm2018_subset.pkl"))
    sensor_names = preprocessed["sensor_columns"]
    
    # Load test sequences
    generator = MaskedSequenceGenerator()
    test_sequences = generator.load_sequences(Path("data/sequences/test_sequences.pkl"))
    
    # Load config
    with open("data/sequences/config.json", "r") as f:
        config = json.load(f)
    
    print(f"\nTest Configuration:")
    print(f"  Total test sequences: {len(test_sequences)}")
    print(f"  Context length: {config['context_length']}")
    print(f"  Gap length: {config['gap_length']}")
    print(f"  Number of sensors: {len(sensor_names)}")
    
    # Load model
    sample = test_sequences[0]
    input_dim = sample["context_before"].shape[1]
    output_dim = input_dim
    output_length = config["gap_length"]
    
    model = BiLSTMReconstructor(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        output_dim=output_dim,
        output_length=output_length,
        dropout=0.3
    )
    
    trainer = BiLSTMModelTrainer(model)
    trainer.load_checkpoint("models/bilstm_best_model.pt")
    
    print(f"\nModel loaded: models/bilstm_best_model.pt")
    
    # Test on several examples
    n_examples = 5
    test_indices = [0, 100, 500, 1000, 5000]
    
    print(f"\n{'='*70}")
    print(f"TESTING ON {n_examples} HELD-OUT EXAMPLES")
    print(f"{'='*70}")
    
    plots_dir = Path("results/test_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    for idx in test_indices:
        if idx >= len(test_sequences):
            print(f"\nSkipping index {idx} (out of range)")
            continue
        
        seq = test_sequences[idx]
        
        print(f"\n{'='*70}")
        print(f"EXAMPLE {idx}")
        print(f"{'='*70}")
        
        # Get model prediction
        test_dataset = BiLSTMDataset([seq])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                context_before = batch["context_before"].to(trainer.device)
                context_after = batch["context_after"].to(trainer.device)
                prediction = model(context_before, context_after)
                prediction = prediction.cpu().numpy()[0]
        
        # Calculate metrics for each sensor
        ground_truth = seq["masked_segment"]
        
        print(f"\nMasked Region Indices: {seq['indices']}")
        print(f"Masked Region Length: {ground_truth.shape[0]} timesteps")
        print(f"\nPer-Sensor Reconstruction Results:")
        print(f"{'Sensor':<30} {'MAE':<12} {'MSE':<12}")
        print("-" * 55)
        
        for sensor_idx, sensor_name in enumerate(sensor_names):
            gt_sensor = ground_truth[:, sensor_idx]
            pred_sensor = prediction[:, sensor_idx]
            
            mae = np.mean(np.abs(pred_sensor - gt_sensor))
            mse = np.mean((pred_sensor - gt_sensor) ** 2)
            
            print(f"{sensor_name:<30} {mae:<12.6f} {mse:<12.6f}")
        
        # Overall metrics
        overall_mae = np.mean(np.abs(prediction - ground_truth))
        overall_mse = np.mean((prediction - ground_truth) ** 2)
        
        print(f"\n{'OVERALL':<30} {overall_mae:<12.6f} {overall_mse:<12.6f}")
        
        # Print actual values for first sensor
        sensor_name = sensor_names[0]
        print(f"\n{sensor_name} - Actual Values:")
        print(f"{'Timestep':<10} {'Original':<15} {'Reconstructed':<15} {'Error':<15}")
        print("-" * 55)
        
        for t in range(min(10, ground_truth.shape[0])):
            orig = ground_truth[t, 0]
            recon = prediction[t, 0]
            error = abs(recon - orig)
            print(f"{t:<10} {orig:<15.6f} {recon:<15.6f} {error:<15.6f}")
        
        # Generate plot
        plot_reconstruction(seq, prediction, sensor_names, idx, plots_dir)
        print(f"\nPlot saved to: {plots_dir}/test_example_{idx}.png")


def plot_reconstruction(seq, prediction, sensor_names, example_idx, save_dir):
    """Generate reconstruction plot for an example."""
    
    # Plot first 3 sensors
    sensors_to_plot = [0, 1, 2]
    
    fig, axes = plt.subplots(len(sensors_to_plot), 1, figsize=(14, 4 * len(sensors_to_plot)))
    
    if len(sensors_to_plot) == 1:
        axes = [axes]
    
    context_before = seq["context_before"]
    context_after = seq["context_after"]
    ground_truth = seq["masked_segment"]
    
    for i, sensor_idx in enumerate(sensors_to_plot):
        sensor_name = sensor_names[sensor_idx]
        
        # Construct full signal
        ctx_before = context_before[:, sensor_idx]
        ctx_after = context_after[:, sensor_idx]
        gt = ground_truth[:, sensor_idx]
        pred = prediction[:, sensor_idx]
        
        # Time indices
        t_before = np.arange(len(ctx_before))
        t_gap = np.arange(len(ctx_before), len(ctx_before) + len(gt))
        t_after = np.arange(len(ctx_before) + len(gt), 
                            len(ctx_before) + len(gt) + len(ctx_after))
        
        # Plot
        axes[i].plot(t_before, ctx_before, 'b-', label='Context Before', linewidth=2, alpha=0.7)
        axes[i].plot(t_gap, gt, 'g-', label='Original (Hidden)', linewidth=3)
        axes[i].plot(t_gap, pred, 'r--', label='Model Reconstruction', linewidth=2)
        axes[i].plot(t_after, ctx_after, 'b-', linewidth=2, alpha=0.7)
        axes[i].axvspan(t_gap[0], t_gap[-1], alpha=0.15, color='gray')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Normalized Value')
        axes[i].set_title(f'Example {example_idx} - {sensor_name}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / f"test_example_{example_idx}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    test_model_inference()
    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")
