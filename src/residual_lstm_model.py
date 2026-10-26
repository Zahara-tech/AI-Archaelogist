"""
Residual Correction LSTM Model
Simple LSTM-based model to predict residuals from linear interpolation.
"""

import torch
import torch.nn as nn


class ResidualLSTM(nn.Module):
    """
    Simple LSTM model for residual prediction.
    
    Input: Concatenated context (50 before + 50 after + 10 interpolation)
    Output: Residual (10 timesteps x 17 sensors)
    """
    
    def __init__(
        self,
        input_dim=17,
        hidden_dim=64,
        num_layers=2,
        output_dim=17,
        output_length=10,
        dropout=0.2
    ):
        super(ResidualLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.output_length = output_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim * output_length)
        )
        
    def forward(self, context_before, context_after, interpolation):
        """
        Forward pass.
        
        Args:
            context_before: (batch, 50, 17)
            context_after: (batch, 50, 17)
            interpolation: (batch, 10, 17)
        
        Returns:
            residual: (batch, 10, 17)
        """
        batch_size = context_before.size(0)
        
        # Concatenate context and interpolation
        # Shape: (batch, 110, 17)
        full_context = torch.cat([context_before, interpolation, context_after], dim=1)
        
        # Project to hidden dimension
        # Shape: (batch, 110, hidden_dim)
        projected = self.input_projection(full_context)
        
        # LSTM encoding
        # Shape: (batch, 110, hidden_dim)
        lstm_out, _ = self.lstm(projected)
        
        # Use the last timestep
        # Shape: (batch, hidden_dim)
        last_hidden = lstm_out[:, -1, :]
        
        # Output projection
        # Shape: (batch, 10 * 17)
        output_flat = self.output_projection(last_hidden)
        
        # Reshape to (batch, 10, 17)
        residual = output_flat.view(batch_size, self.output_length, self.output_dim)
        
        return residual
