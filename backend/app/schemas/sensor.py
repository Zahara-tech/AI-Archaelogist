"""
Schemas for sensor reconstruction
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np


class ReconstructionRequest(BaseModel):
    """Request for sensor reconstruction."""
    
    context_before: List[List[float]] = Field(..., description="50 timesteps before gap, 17 sensors")
    context_after: List[List[float]] = Field(..., description="50 timesteps after gap, 17 sensors")
    
    class Config:
        json_schema_extra = {
            "example": {
                "context_before": [[0.1] * 17] * 50,
                "context_after": [[0.2] * 17] * 50
            }
        }


class ReconstructionResponse(BaseModel):
    """Response from sensor reconstruction."""
    
    status: str
    reconstructed_segment: Optional[List[List[float]]] = None
    shape: Optional[tuple] = None
    error: Optional[str] = None


class SensorColumn(BaseModel):
    """Sensor column information."""
    name: str
    index: int


class DatasetValidationRequest(BaseModel):
    """Request for dataset validation."""
    
    filename: str
    rows: int
    columns: int
    column_names: List[str]
    detected_types: Dict[str, str]
    missing_values: Dict[str, int]
    statistics: Optional[Dict[str, Dict[str, float]]] = None


class DatasetValidationResponse(BaseModel):
    """Response from dataset validation."""
    
    valid: bool
    errors: List[str]
    warnings: List[str]
    sensor_columns: List[SensorColumn]
    metadata_columns: List[str]
    can_reconstruct: bool


class ExperimentComparisonRequest(BaseModel):
    """Request for experiment comparison."""
    
    context_before: List[List[float]]
    context_after: List[List[float]]
    target: Optional[List[List[float]]] = None  # For evaluation only


class MetricsResponse(BaseModel):
    """Metrics response."""
    
    mse: float
    mae: float
    rmse: float
    per_sensor_mse: Dict[str, float]
    per_sensor_mae: Dict[str, float]
    per_timestep_mse: List[float]
    per_timestep_mae: List[float]


class ComparisonResponse(BaseModel):
    """Response from experiment comparison."""
    
    neural_metrics: Optional[MetricsResponse] = None
    baseline_metrics: Optional[MetricsResponse] = None
    better_method: str
    error: Optional[str] = None
