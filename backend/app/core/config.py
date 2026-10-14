"""
AI Archaeologist Configuration and Settings.
Supports PostgreSQL and SQLite database backends, customizable anomaly thresholds,
and modular service endpoints.
"""

from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
import os

# Get project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Archaeologist"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database: Uses SQLite by default for zero-setup execution, PostgreSQL when DATABASE_URL is configured
    DATABASE_URL: str = Field(
        default=f"sqlite:///{PROJECT_ROOT / 'ai_archaeologist.db'}",
        description="Database connection string (sqlite:///... or postgresql+psycopg2://user:pass@localhost:5432/ai_archaeologist)"
    )
    
    # Storage paths
    DATA_DIR: str = "data"
    UPLOAD_DIR: str = "data/raw/uploads"
    SYNTHETIC_DIR: str = "data/synthetic"
    
    # Anomaly Detection Baseline Thresholds (Configurable)
    TEMP_WARNING_C: float = 65.0
    TEMP_CRITICAL_C: float = 75.0
    VIBRATION_WARNING_MMS: float = 3.5
    VIBRATION_CRITICAL_MMS: float = 6.0
    COOLANT_PRESSURE_MIN_BAR: float = 2.5
    SPINDLE_STALL_RPM_DROP: int = 1500
    
    # Temporal Engine Settings
    TEMPORAL_GAP_THRESHOLD_SECONDS: float = 30.0  # Gaps > 30s in active manufacturing trigger gap investigation
    CONFLICT_TOLERANCE_SECONDS: float = 5.0      # Discrepancy > 5s between different sources triggers conflict flag
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173", "*"]
    
    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.DATA_DIR, exist_ok=True)
