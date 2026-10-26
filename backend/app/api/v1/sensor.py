"""
Sensor Reconstruction and Dataset Endpoints
"""

import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Dict, Optional
import io
import sys
from pathlib import Path
from pydantic import BaseModel
import json

from app.services.model_service import model_service
from app.schemas.sensor import (
    ReconstructionRequest,
    ReconstructionResponse,
    DatasetValidationResponse,
    SensorColumn,
    ExperimentComparisonRequest,
    ComparisonResponse,
    MetricsResponse
)

router = APIRouter(prefix="/sensor", tags=["Sensor"])

# Known sensor columns from training
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

METADATA_COLUMNS = ["time", "Tool", "stage", "Lot", "runnum", "recipe", "recipe_step"]

# Benchmark path
BENCHMARK_V2_PATH = Path(__file__).resolve().parents[4] / "data" / "benchmark_v2"
GAP_BENCHMARK_BASE = Path(__file__).resolve().parents[4] / "data" / "gap_length_benchmark"
RESULTS_PATH = Path(__file__).resolve().parents[4] / "results"


# Additional schemas for benchmark endpoints
class AvailableWindowsResponse(BaseModel):
    windows: List[Dict]


class BenchmarkWindowRequest(BaseModel):
    filename: str
    window_index: int
    method: str = "attention_seq2seq"  # "attention_seq2seq", "residual_lstm", "linear_interpolation", "hybrid"
    gap_length: int = 10


class BenchmarkReconstructionResponse(BaseModel):
    filename: str
    window_index: int
    method: str
    gap_length: int
    predictions: List[List[float]]
    ground_truth: List[List[float]]
    metrics: MetricsResponse
    sensor_columns: List[str]


class DatasetInfoResponse(BaseModel):
    dataset_name: str
    files: List[Dict[str, str | int | float]]
    sensor_columns: List[str]
    context_length: int
    gap_length: int
    total_sequences: int


class SensorDataRequest(BaseModel):
    filename: str
    window_index: int
    sensors: Optional[List[str]] = None


class SensorDataResponse(BaseModel):
    filename: str
    window_index: int
    context_before: List[List[float]]
    context_after: List[List[float]]
    masked_segment: List[List[float]]
    sensor_columns: List[str]
    full_sequence: Optional[List[List[float]]] = None


class MaskedReconstructionRequest(BaseModel):
    filename: str
    window_index: int
    gap_start_index: int  # Index within the 110-step sequence (50 before + 10 gap + 50 after)
    gap_length: int = 10


class EvaluationMetricsResponse(BaseModel):
    model_name: str
    test_mse: float
    test_mae: float
    baseline_mse: float
    baseline_mae: float
    test_sequences: int
    test_files: int
    per_sensor_mse: Optional[Dict[str, float]] = None
    per_sensor_mae: Optional[Dict[str, float]] = None


@router.post("/reconstruct", response_model=ReconstructionResponse)
def reconstruct_sensor_data(req: ReconstructionRequest):
    """Run sensor reconstruction inference."""
    try:
        context_before = np.array(req.context_before)
        context_after = np.array(req.context_after)
        
        result = model_service.reconstruct(context_before, context_after)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        
        return ReconstructionResponse(
            status="success",
            reconstructed_segment=result["reconstructed_segment"].tolist(),
            shape=result["shape"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and validate a CSV dataset."""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Basic validation
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Detect column types
        column_types = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            column_types[col] = dtype
        
        # Count missing values
        missing_counts = df.isnull().sum().to_dict()
        
        # Identify sensor and metadata columns
        sensor_cols = [col for col in df.columns if col in SENSOR_COLUMNS]
        metadata_cols = [col for col in df.columns if col in METADATA_COLUMNS]
        
        # Basic statistics for numeric columns
        statistics = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            statistics[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }
        
        # Check for invalid values
        errors = []
        warnings = []
        
        # Check for infinite values
        for col in df.select_dtypes(include=[np.number]).columns:
            if np.isinf(df[col]).any():
                errors.append(f"Column '{col}' contains infinite values")
        
        # Check if sensor columns are present
        if len(sensor_cols) == 0:
            warnings.append("No known sensor columns detected")
        
        can_reconstruct = len(sensor_cols) == 17
        
        return {
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "column_types": column_types,
            "missing_values": missing_counts,
            "sensor_columns": sensor_cols,
            "metadata_columns": metadata_cols,
            "statistics": statistics,
            "errors": errors,
            "warnings": warnings,
            "can_reconstruct": can_reconstruct
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/validate")
def validate_dataset(data: dict):
    """Validate dataset structure for reconstruction."""
    try:
        errors = []
        warnings = []
        
        column_names = data.get("column_names", [])
        
        # Check for required sensor columns
        missing_sensors = [col for col in SENSOR_COLUMNS if col not in column_names]
        if missing_sensors:
            errors.append(f"Missing sensor columns: {missing_sensors}")
        
        # Check for metadata columns
        missing_metadata = [col for col in METADATA_COLUMNS if col not in column_names]
        if missing_metadata:
            warnings.append(f"Missing metadata columns: {missing_metadata}")
        
        sensor_cols = [col for col in column_names if col in SENSOR_COLUMNS]
        
        return DatasetValidationResponse(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sensor_columns=[SensorColumn(name=col, index=SENSOR_COLUMNS.index(col)) for col in sensor_cols],
            metadata_columns=[col for col in column_names if col in METADATA_COLUMNS],
            can_reconstruct=len(sensor_cols) == 17
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def linear_interpolation(context_before: np.ndarray, context_after: np.ndarray) -> np.ndarray:
    """Linear interpolation baseline reconstruction."""
    gap_length = 10
    
    # Handle 2D (50, 17) or 3D (N, 50, 17) input
    if context_before.ndim == 3:
        # Batch mode
        batch_size = context_before.shape[0]
        interpolated = np.zeros((batch_size, gap_length, context_before.shape[2]))
        for i in range(gap_length):
            alpha = (i + 1) / (gap_length + 1)
            interpolated[:, i, :] = (1 - alpha) * context_before[:, -1, :] + alpha * context_after[:, 0, :]
        return interpolated
    else:
        # Single window mode (50, 17)
        interpolated = np.zeros((gap_length, context_before.shape[1]))
        for i in range(gap_length):
            alpha = (i + 1) / (gap_length + 1)
            interpolated[i] = (1 - alpha) * context_before[-1] + alpha * context_after[0]
        return interpolated


def calculate_metrics(predictions: np.ndarray, target: np.ndarray) -> Dict:
    """Calculate reconstruction metrics."""
    mse = np.mean((predictions - target) ** 2)
    mae = np.mean(np.abs(predictions - target))
    rmse = np.sqrt(mse)
    
    # Per-sensor metrics
    per_sensor_mse = np.mean((predictions - target) ** 2, axis=0).tolist()
    per_sensor_mae = np.mean(np.abs(predictions - target), axis=0).tolist()
    
    # Per-timestep metrics
    per_timestep_mse = np.mean((predictions - target) ** 2, axis=1).tolist()
    per_timestep_mae = np.mean(np.abs(predictions - target), axis=1).tolist()
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "per_sensor_mse": dict(zip(SENSOR_COLUMNS, per_sensor_mse)),
        "per_sensor_mae": dict(zip(SENSOR_COLUMNS, per_sensor_mae)),
        "per_timestep_mse": per_timestep_mse,
        "per_timestep_mae": per_timestep_mae
    }


@router.post("/experiments/compare", response_model=ComparisonResponse)
def compare_methods(req: ExperimentComparisonRequest):
    """Compare neural model vs linear interpolation baseline."""
    try:
        context_before = np.array(req.context_before)
        context_after = np.array(req.context_after)
        target = np.array(req.target) if req.target else None
        
        # Neural model reconstruction
        neural_result = model_service.reconstruct(context_before, context_after)
        if neural_result["status"] != "success":
            raise HTTPException(status_code=500, detail=neural_result["error"])
        
        neural_predictions = neural_result["reconstructed_segment"]
        
        # Linear interpolation
        baseline_predictions = linear_interpolation(context_before, context_after)
        
        # Calculate metrics if target is available
        if target is not None:
            neural_metrics = calculate_metrics(neural_predictions, target)
            baseline_metrics = calculate_metrics(baseline_predictions, target)
            
            # Determine which is better
            better_method = "neural" if neural_metrics["mse"] < baseline_metrics["mse"] else "baseline"
            
            return ComparisonResponse(
                neural_metrics=MetricsResponse(**neural_metrics),
                baseline_metrics=MetricsResponse(**baseline_metrics),
                better_method=better_method
            )
        else:
            # No target available, return predictions only
            return ComparisonResponse(
                better_method="unknown",
                neural_metrics=None,
                baseline_metrics=None
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def sensor_health():
    """Check sensor reconstruction service health."""
    return {
        "model_loaded": model_service.model_loaded,
        "sensor_columns": SENSOR_COLUMNS,
        "metadata_columns": METADATA_COLUMNS
    }


# Benchmark window reconstruction endpoints

@router.get("/benchmark/windows")
def list_benchmark_windows():
    """List available benchmark test files and window counts."""
    try:
        test_files = sorted(BENCHMARK_V2_PATH.glob("test_*.npz"))
        
        if not test_files:
            raise HTTPException(status_code=404, detail="No benchmark test files found")
        
        windows_info = []
        for file_path in test_files:
            data = np.load(file_path)
            window_count = len(data["context_before"])
            windows_info.append({
                "filename": file_path.name,
                "window_count": int(window_count),
                "context_shape": data["context_before"].shape,
                "target_shape": data["masked_segment"].shape
            })
        
        return AvailableWindowsResponse(windows=windows_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmark/reconstruct-window")
def reconstruct_benchmark_window(request: BenchmarkWindowRequest):
    """Reconstruct a specific benchmark window with selected method."""
    try:
        # Validate method
        valid_methods = ["attention_seq2seq", "residual_lstm", "linear_interpolation", "hybrid"]
        if request.method not in valid_methods:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid method '{request.method}'. Valid methods: {valid_methods}"
            )
        
        # Validate gap length for method
        if request.method == "attention_seq2seq" and request.gap_length != 10:
            raise HTTPException(
                status_code=400,
                detail="Attention Seq2Seq only supports gap_length=10"
            )
        
        if request.method == "hybrid" and request.gap_length != 10:
            raise HTTPException(
                status_code=400,
                detail="Hybrid reconstruction only supports gap_length=10"
            )
        
        if request.method == "residual_lstm" and request.gap_length not in [10, 20, 50]:
            raise HTTPException(
                status_code=400,
                detail=f"Residual LSTM only supports gap_length in [10, 20, 50], got {request.gap_length}"
            )
        
        # Load the NPZ file
        file_path = BENCHMARK_V2_PATH / request.filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File {request.filename} not found")
        
        data = np.load(file_path)
        
        # Validate window index
        window_count = len(data["context_before"])
        if request.window_index < 0 or request.window_index >= window_count:
            raise HTTPException(
                status_code=422,
                detail=f"Window index {request.window_index} out of range [0, {window_count})"
            )
        
        # Extract window data (already normalized in benchmark)
        context_before = data["context_before"][request.window_index]  # (50, 17)
        context_after = data["context_after"][request.window_index]    # (50, 17)
        target = data["masked_segment"][request.window_index]         # (10, 17)
        
        # For gap length != 10, we need to handle differently
        if request.gap_length != 10:
            # For now, only support gap_length=10 with benchmark data
            # The benchmark was created with 10-timestep gaps
            raise HTTPException(
                status_code=400,
                detail=f"Benchmark data only supports gap_length=10. For other gap lengths, use gap_length_benchmark data."
            )
        
        # Run reconstruction with selected method
        result = model_service.reconstruct(
            context_before,
            context_after,
            already_normalized=True,
            method=request.method,
            gap_length=request.gap_length
        )
        
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result["error"])
        
        predictions = result["reconstructed_segment"]
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, target)
        
        return BenchmarkReconstructionResponse(
            filename=request.filename,
            window_index=request.window_index,
            method=request.method,
            gap_length=request.gap_length,
            predictions=predictions.tolist(),
            ground_truth=target.tolist(),
            metrics=MetricsResponse(**metrics),
            sensor_columns=SENSOR_COLUMNS
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class GapBenchmarkWindowRequest(BaseModel):
    gap_length: int
    filename: str
    window_index: int
    method: str = "linear_interpolation"  # "linear_interpolation", "residual_lstm"


class GapBenchmarkReconstructionResponse(BaseModel):
    gap_length: int
    filename: str
    window_index: int
    method: str
    predictions: List[List[float]]
    ground_truth: List[List[float]]
    metrics: MetricsResponse
    sensor_columns: List[str]


@router.get("/gap-benchmark/info")
def get_gap_benchmark_info():
    """Get information about available gap length benchmarks."""
    try:
        gap_lengths = []
        for gap_dir in GAP_BENCHMARK_BASE.glob("gap_*"):
            if gap_dir.is_dir():
                gap_length = int(gap_dir.name.split("_")[1])
                manifest_path = gap_dir / "manifest.json"
                
                if manifest_path.exists():
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    
                    # Count test sequences
                    test_files = sorted(gap_dir.glob("test_*.npz"))
                    total_sequences = 0
                    for test_file in test_files:
                        data = np.load(test_file)
                        total_sequences += len(data["context_before"])
                    
                    gap_lengths.append({
                        "gap_length": gap_length,
                        "context_length": manifest["config"]["context_length"],
                        "total_sequences": total_sequences,
                        "test_files": len(test_files)
                    })
        
        return {
            "available_gap_lengths": sorted(gap_lengths, key=lambda x: x["gap_length"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gap-benchmark/{gap_length}/windows")
def list_gap_benchmark_windows(gap_length: int):
    """List available test files for a specific gap length."""
    try:
        gap_dir = GAP_BENCHMARK_BASE / f"gap_{gap_length}"
        if not gap_dir.exists():
            raise HTTPException(status_code=404, detail=f"Gap length {gap_length} not found")
        
        test_files = sorted(gap_dir.glob("test_*.npz"))
        
        if not test_files:
            raise HTTPException(status_code=404, detail="No test files found")
        
        windows_info = []
        for file_path in test_files:
            data = np.load(file_path)
            window_count = len(data["context_before"])
            windows_info.append({
                "filename": file_path.name,
                "window_count": int(window_count),
                "context_shape": data["context_before"].shape,
                "target_shape": data["masked_segment"].shape
            })
        
        return AvailableWindowsResponse(windows=windows_info)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gap-benchmark/reconstruct-window")
def reconstruct_gap_benchmark_window(request: GapBenchmarkWindowRequest):
    """Reconstruct a window from gap length benchmark."""
    try:
        # Validate gap length
        gap_dir = GAP_BENCHMARK_BASE / f"gap_{request.gap_length}"
        if not gap_dir.exists():
            raise HTTPException(status_code=404, detail=f"Gap length {request.gap_length} not found")
        
        # Validate method
        valid_methods = ["linear_interpolation", "attention_seq2seq", "residual_lstm", "hybrid"]
        if request.method not in valid_methods:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid method '{request.method}'. Valid methods: {valid_methods}"
            )
        
        # Validate gap length for method
        if request.method == "attention_seq2seq" and request.gap_length != 10:
            raise HTTPException(
                status_code=400,
                detail="Attention Seq2Seq only supports gap_length=10"
            )
        
        if request.method == "hybrid" and request.gap_length != 10:
            raise HTTPException(
                status_code=400,
                detail="Hybrid reconstruction only supports gap_length=10"
            )
        
        # Load the NPZ file
        file_path = gap_dir / request.filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File {request.filename} not found")
        
        data = np.load(file_path)
        
        # Validate window index
        window_count = len(data["context_before"])
        if request.window_index < 0 or request.window_index >= window_count:
            raise HTTPException(
                status_code=422,
                detail=f"Window index {request.window_index} out of range [0, {window_count})"
            )
        
        # Extract window data (already normalized)
        context_before = data["context_before"][request.window_index]
        context_after = data["context_after"][request.window_index]
        target = data["masked_segment"][request.window_index]
        
        # Run reconstruction
        result = model_service.reconstruct(
            context_before,
            context_after,
            already_normalized=True,
            method=request.method,
            gap_length=request.gap_length
        )
        
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result["error"])
        
        predictions = result["reconstructed_segment"]
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, target)
        
        return GapBenchmarkReconstructionResponse(
            gap_length=request.gap_length,
            filename=request.filename,
            window_index=request.window_index,
            method=request.method,
            predictions=predictions.tolist(),
            ground_truth=target.tolist(),
            metrics=MetricsResponse(**metrics),
            sensor_columns=SENSOR_COLUMNS
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gap-benchmark/{gap_length}/evaluation-summary")
def get_gap_evaluation_summary(gap_length: int):
    """Get evaluation summary for a specific gap length from saved results."""
    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[4]
        results_path = PROJECT_ROOT / "experiments" / "gap_length_experiment" / "evaluation_results.json"
        
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Evaluation results not found")
        
        with open(results_path, 'r') as f:
            all_results = json.load(f)
        
        gap_key = str(gap_length)
        if gap_key not in all_results:
            raise HTTPException(status_code=404, detail=f"No results for gap length {gap_length}")
        
        results = all_results[gap_key]
        
        return {
            "gap_length": gap_length,
            "overall_normalized": results["overall_normalized"],
            "overall_original": results["overall_original"],
            "total_sequences": results["total_sequences"],
            "per_file_results": results["per_file_results"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dataset/info", response_model=DatasetInfoResponse)
def get_dataset_info():
    """Get dataset information including files, sensor columns, and statistics."""
    try:
        # Load manifest
        manifest_path = BENCHMARK_V2_PATH / "manifest.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail="Manifest file not found")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Count total sequences from all test files
        test_files = sorted(BENCHMARK_V2_PATH.glob("test_*.npz"))
        total_sequences = 0
        files_info = []
        
        for file_path in test_files:
            data = np.load(file_path)
            window_count = len(data["context_before"])
            total_sequences += window_count
            files_info.append({
                "filename": file_path.name,
                "window_count": int(window_count),
                "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2)
            })
        
        return DatasetInfoResponse(
            dataset_name="PHM 2018 Ion Mill Etch Tool",
            files=files_info,
            sensor_columns=manifest["sensor_columns"],
            context_length=manifest["config"]["context_length"],
            gap_length=manifest["config"]["gap_length"],
            total_sequences=total_sequences
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dataset/sensor-data", response_model=SensorDataResponse)
def get_sensor_data(request: SensorDataRequest):
    """Get sensor data for a specific window for plotting."""
    try:
        file_path = BENCHMARK_V2_PATH / request.filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File {request.filename} not found")
        
        data = np.load(file_path)
        
        # Validate window index
        window_count = len(data["context_before"])
        if request.window_index < 0 or request.window_index >= window_count:
            raise HTTPException(
                status_code=422,
                detail=f"Window index {request.window_index} out of range [0, {window_count})"
            )
        
        # Extract data
        context_before = data["context_before"][request.window_index]  # (50, 17)
        context_after = data["context_after"][request.window_index]    # (50, 17)
        masked_segment = data["masked_segment"][request.window_index]  # (10, 17)
        
        # Filter sensors if specified
        sensor_indices = None
        if request.sensors:
            # Map sensor names to indices
            sensor_indices = [SENSOR_COLUMNS.index(s) for s in request.sensors if s in SENSOR_COLUMNS]
            if not sensor_indices:
                raise HTTPException(status_code=400, detail="No valid sensor names provided")
            
            context_before = context_before[:, sensor_indices]
            context_after = context_after[:, sensor_indices]
            masked_segment = masked_segment[:, sensor_indices]
            sensor_columns = request.sensors
        else:
            sensor_columns = SENSOR_COLUMNS
        
        # Construct full sequence (50 before + 10 gap + 50 after = 110)
        full_sequence = np.concatenate([context_before, masked_segment, context_after], axis=0)
        
        return SensorDataResponse(
            filename=request.filename,
            window_index=request.window_index,
            context_before=context_before.tolist(),
            context_after=context_after.tolist(),
            masked_segment=masked_segment.tolist(),
            sensor_columns=sensor_columns,
            full_sequence=full_sequence.tolist()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dataset/masked-reconstruct")
def masked_reconstruction(request: MaskedReconstructionRequest):
    """Reconstruct a masked segment at a custom gap location."""
    try:
        file_path = BENCHMARK_V2_PATH / request.filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File {request.filename} not found")
        
        data = np.load(file_path)
        
        # Validate window index
        window_count = len(data["context_before"])
        if request.window_index < 0 or request.window_index >= window_count:
            raise HTTPException(
                status_code=422,
                detail=f"Window index {request.window_index} out of range [0, {window_count})"
            )
        
        # Extract full sequence (50 before + 10 gap + 50 after = 110)
        context_before = data["context_before"][request.window_index]  # (50, 17)
        context_after = data["context_after"][request.window_index]    # (50, 17)
        masked_segment = data["masked_segment"][request.window_index]  # (10, 17)
        
        full_sequence = np.concatenate([context_before, masked_segment, context_after], axis=0)
        
        # Validate gap location
        sequence_length = len(full_sequence)
        if request.gap_start_index < 0 or request.gap_start_index >= sequence_length:
            raise HTTPException(
                status_code=422,
                detail=f"Gap start index {request.gap_start_index} out of range [0, {sequence_length})"
            )
        
        if request.gap_start_index + request.gap_length > sequence_length:
            raise HTTPException(
                status_code=422,
                detail=f"Gap extends beyond sequence end"
            )
        
        # Extract context before and after the custom gap
        custom_context_before = full_sequence[:request.gap_start_index]
        custom_context_after = full_sequence[request.gap_start_index + request.gap_length:]
        custom_ground_truth = full_sequence[request.gap_start_index:request.gap_start_index + request.gap_length]
        
        # Ensure we have enough context (need at least 50 before and 50 after for the model)
        if len(custom_context_before) < 50:
            raise HTTPException(
                status_code=422,
                detail=f"Not enough context before gap: need 50, have {len(custom_context_before)}"
            )
        
        if len(custom_context_after) < 50:
            raise HTTPException(
                status_code=422,
                detail=f"Not enough context after gap: need 50, have {len(custom_context_after)}"
            )
        
        # Use the last 50 before and first 50 after for model input
        model_context_before = custom_context_before[-50:]
        model_context_after = custom_context_after[:50]
        
        # Neural model reconstruction
        neural_result = model_service.reconstruct(model_context_before, model_context_after, already_normalized=True)
        if neural_result["status"] != "success":
            raise HTTPException(status_code=500, detail=neural_result["error"])
        
        neural_predictions = neural_result["reconstructed_segment"]
        
        # Linear interpolation baseline
        baseline_predictions = linear_interpolation(model_context_before, model_context_after)
        
        # Calculate metrics
        neural_metrics = calculate_metrics(neural_predictions, custom_ground_truth)
        baseline_metrics = calculate_metrics(baseline_predictions, custom_ground_truth)
        
        better_method = "neural" if neural_metrics["mse"] < baseline_metrics["mse"] else "baseline"
        
        return {
            "filename": request.filename,
            "window_index": request.window_index,
            "gap_start_index": request.gap_start_index,
            "gap_length": request.gap_length,
            "context_before": model_context_before.tolist(),
            "context_after": model_context_after.tolist(),
            "ground_truth": custom_ground_truth.tolist(),
            "neural_predictions": neural_predictions.tolist(),
            "baseline_predictions": baseline_predictions.tolist(),
            "neural_metrics": MetricsResponse(**neural_metrics),
            "baseline_metrics": MetricsResponse(**baseline_metrics),
            "better_method": better_method,
            "sensor_columns": SENSOR_COLUMNS
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/evaluation-metrics", response_model=EvaluationMetricsResponse)
def get_evaluation_metrics():
    """Get verified evaluation metrics from saved artifacts."""
    try:
        # Load evaluation results from final_evaluation_v2.json
        evaluation_path = RESULTS_PATH / "final_evaluation_v2.json"
        if not evaluation_path.exists():
            raise HTTPException(status_code=404, detail="Evaluation results not found. Run final_evaluation_v2.py first.")
        
        with open(evaluation_path, 'r') as f:
            evaluation_data = json.load(f)
        
        overall = evaluation_data.get("overall", {})
        protocol = evaluation_data.get("protocol", {})
        
        # Load training history for model info
        training_history_path = RESULTS_PATH / "attention_seq2seq_v2_training.json"
        training_data = None
        if training_history_path.exists():
            with open(training_history_path, 'r') as f:
                training_data = json.load(f)
        
        return EvaluationMetricsResponse(
            model_name="Attention Seq2Seq V2",
            test_mse=overall.get("attention_v2_mse", None),
            test_mae=overall.get("attention_v2_mae", None),
            baseline_mse=overall.get("baseline_mse", None),
            baseline_mae=overall.get("baseline_mae", None),
            test_sequences=overall.get("test_sequences", None),
            test_files=protocol.get("test_files", None),
            per_sensor_mse=None,  # Not available in saved artifacts
            per_sensor_mae=None,  # Not available in saved artifacts
            training_history=training_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/evaluation-details")
def get_evaluation_details():
    """Get detailed evaluation metrics including per-file results."""
    try:
        evaluation_path = RESULTS_PATH / "final_evaluation_v2.json"
        if not evaluation_path.exists():
            raise HTTPException(status_code=404, detail="Evaluation results not found. Run final_evaluation_v2.py first.")
        
        with open(evaluation_path, 'r') as f:
            evaluation_data = json.load(f)
        
        return evaluation_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/residual-data")
def get_residual_data(limit: int = 10000):
    """Get residual data for residual analysis plot."""
    try:
        predictions_path = RESULTS_PATH / "final_predictions_v2.npz"
        if not predictions_path.exists():
            raise HTTPException(status_code=404, detail="Prediction data not found. Run final_evaluation_v2.py first.")
        
        data = np.load(predictions_path)
        targets = data["targets"]
        baseline_predictions = data["baseline_predictions"]
        attention_v2_predictions = data["attention_v2_predictions"]
        
        # Sample data to avoid sending too much data
        total_samples = len(targets)
        sample_indices = np.random.choice(total_samples, min(limit, total_samples), replace=False)
        
        sample_targets = targets[sample_indices]
        sample_baseline = baseline_predictions[sample_indices]
        sample_attention = attention_v2_predictions[sample_indices]
        
        # Calculate residuals (actual - prediction)
        baseline_residuals = sample_targets - sample_baseline
        attention_residuals = sample_targets - sample_attention
        
        return {
            "sample_count": len(sample_indices),
            "total_samples": total_samples,
            "targets": sample_targets.tolist(),
            "baseline_predictions": sample_baseline.tolist(),
            "attention_predictions": sample_attention.tolist(),
            "baseline_residuals": baseline_residuals.tolist(),
            "attention_residuals": attention_residuals.tolist(),
            "sensor_columns": SENSOR_COLUMNS
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/epoch-comparison")
def get_epoch_comparison():
    """Get training results for different epoch counts."""
    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[4]
        history_path = PROJECT_ROOT / "experiments" / "attention_seq2seq_v2_50epochs" / "results" / "training_history.json"
        
        if not history_path.exists():
            raise HTTPException(status_code=404, detail="Training history not found")
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        training_results = history.get("training_results", {})
        train_losses = history.get("train_losses", [])
        val_losses = history.get("val_losses", [])
        
        # Extract data at different epoch points
        epoch_data = []
        for target_epoch in [20, 30, 40, 50]:
            if target_epoch <= len(train_losses):
                epoch_data.append({
                    "requested_epochs": target_epoch,
                    "completed_epochs": target_epoch,
                    "train_loss": train_losses[target_epoch - 1],
                    "val_loss": val_losses[target_epoch - 1] if target_epoch <= len(val_losses) else None,
                    "status": "evaluated"
                })
            else:
                epoch_data.append({
                    "requested_epochs": target_epoch,
                    "completed_epochs": None,
                    "train_loss": None,
                    "val_loss": None,
                    "status": "not_evaluated"
                })
        
        # Add best epoch info
        best_epoch = training_results.get("best_epoch")
        best_val_loss = training_results.get("best_val_loss")
        
        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "total_epochs": training_results.get("total_epochs"),
            "total_time": training_results.get("total_time"),
            "epoch_data": epoch_data,
            "train_losses": train_losses,
            "val_losses": val_losses
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/training-history")
def get_training_history():
    """Get training history including loss curves and hyperparameters."""
    try:
        training_history_path = RESULTS_PATH / "attention_seq2seq_v2_training.json"
        if not training_history_path.exists():
            raise HTTPException(status_code=404, detail="Training history not found")
        
        with open(training_history_path, 'r') as f:
            training_data = json.load(f)
        
        return training_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
