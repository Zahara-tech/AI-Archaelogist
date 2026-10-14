"""
Model Loading and Inference Service for Attention Seq2Seq V2
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add src to path for model import
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from attention_seq2seq_model import AttentionSeq2SeqReconstructor


class ModelService:
    """Service for loading and running the Attention Seq2Seq V2 model."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.sensor_columns = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        
        # Paths
        PROJECT_ROOT = Path(__file__).resolve().parents[3]
        self.model_path = PROJECT_ROOT / "models" / "attention_seq2seq_v2_best_model.pt"
        self.scaler_path = PROJECT_ROOT / "data" / "benchmark_v2" / "scaler.pkl"
        self.manifest_path = PROJECT_ROOT / "data" / "benchmark_v2" / "manifest.json"
        self.training_history_path = PROJECT_ROOT / "results" / "attention_seq2seq_v2_training.json"
    
    def load_model(self) -> Dict:
        """Load the trained model and scaler."""
        try:
            # Load scaler
            if self.scaler_path.exists():
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
            else:
                raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")
            
            # Load manifest for sensor ordering
            if self.manifest_path.exists():
                import json
                with open(self.manifest_path, "r") as f:
                    manifest = json.load(f)
                    self.sensor_columns = manifest["sensor_columns"]
            else:
                raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")
            
            # Create model
            self.model = AttentionSeq2SeqReconstructor(
                input_dim=17,
                hidden_dim=64,
                num_layers=2,
                output_dim=17,
                output_length=10,
                dropout=0.3,
                attention_dim=128
            )
            
            # Load checkpoint
            if self.model_path.exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")
            
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            
            # Count parameters
            param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                "status": "loaded",
                "device": str(self.device),
                "parameter_count": param_count,
                "sensor_columns": self.sensor_columns,
                "model_path": str(self.model_path),
                "scaler_path": str(self.scaler_path)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        if not self.model_loaded:
            load_result = self.load_model()
            if load_result["status"] != "loaded":
                return load_result
        
        # Load training history if available
        training_history = None
        if self.training_history_path.exists():
            import json
            with open(self.training_history_path, "r") as f:
                training_history = json.load(f)
        else:
            training_history = {"error": "Training history file not found", "path": str(self.training_history_path)}
        
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": "Attention Seq2Seq V2",
            "architecture": "Bidirectional LSTM encoder with additive attention and LSTM decoder",
            "parameter_count": param_count,
            "input_dim": 17,
            "output_dim": 17,
            "context_length": 50,
            "gap_length": 10,
            "device": str(self.device),
            "model_loaded": True,
            "sensor_columns": self.sensor_columns,
            "training_history": training_history
        }
    
    def verify_model(self) -> Dict:
        """Verify model with synthetic input."""
        if not self.model_loaded:
            load_result = self.load_model()
            if load_result["status"] != "loaded":
                return load_result
        
        try:
            batch_size = 4
            context_length = 50
            gap_length = 10
            sensor_count = 17
            
            # Create synthetic input
            context_before = torch.randn(batch_size, context_length, sensor_count).to(self.device)
            context_after = torch.randn(batch_size, context_length, sensor_count).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(
                    context_before,
                    context_after,
                    target=None,
                    teacher_forcing_ratio=0.0
                )
            
            # Verify shape
            expected_shape = (batch_size, gap_length, sensor_count)
            shape_correct = predictions.shape == expected_shape
            
            # Verify finiteness
            finite = torch.all(torch.isfinite(predictions)).item()
            
            # Check variance
            variance = np.var(predictions.cpu().numpy(), axis=1)
            mean_variance = np.mean(variance)
            
            return {
                "status": "verified",
                "shape_correct": shape_correct,
                "expected_shape": expected_shape,
                "actual_shape": tuple(predictions.shape),
                "finite": finite,
                "mean_variance": float(mean_variance),
                "varying_predictions": mean_variance > 0.001
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def reconstruct(
        self,
        context_before: np.ndarray,
        context_after: np.ndarray,
        already_normalized: bool = False
    ) -> Dict:
        """
        Run reconstruction inference.
        
        Args:
            context_before: (context_length, num_sensors) array
            context_after: (context_length, num_sensors) array
            already_normalized: If True, skip scaler transformation (for benchmark data)
            
        Returns:
            Dictionary with reconstructed segment and metadata
        """
        if not self.model_loaded:
            load_result = self.load_model()
            if load_result["status"] != "loaded":
                return load_result
        
        try:
            # Validate input shape
            expected_context_shape = (50, 17)
            if context_before.shape != expected_context_shape:
                raise ValueError(f"context_before must have shape {expected_context_shape}, got {context_before.shape}")
            if context_after.shape != expected_context_shape:
                raise ValueError(f"context_after must have shape {expected_context_shape}, got {context_after.shape}")
            
            # Normalize using scaler (skip if already normalized)
            if already_normalized:
                context_before_norm = context_before
                context_after_norm = context_after
            else:
                # Scaler expects 2D input: (n_samples, n_features)
                context_before_norm = self.scaler.transform(context_before)
                context_after_norm = self.scaler.transform(context_after)
            
            # Convert to tensors and add batch dimension
            context_before_tensor = torch.FloatTensor(context_before_norm).unsqueeze(0).to(self.device)
            context_after_tensor = torch.FloatTensor(context_after_norm).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(
                    context_before_tensor,
                    context_after_tensor,
                    target=None,
                    teacher_forcing_ratio=0.0
                )
            
            # Remove batch dimension and convert to numpy
            predictions_norm = predictions.squeeze(0).cpu().numpy()
            
            # Inverse transform (skip if already normalized)
            if already_normalized:
                predictions_original = predictions_norm
            else:
                predictions_original = self.scaler.inverse_transform(predictions_norm)
            
            return {
                "status": "success",
                "reconstructed_segment": predictions_original,
                "reconstructed_segment_normalized": predictions_norm,
                "shape": predictions_original.shape
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# Global model service instance
model_service = ModelService()
