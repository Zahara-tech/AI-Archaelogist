"""
Evidence Inspection and Sensor Telemetry Streaming Endpoints.
"""

import os
import pandas as pd
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.database.models import EvidenceModel, IncidentModel
from app.schemas.evidence import EvidenceResponse
from app.services.report_service import ReportService

router = APIRouter(prefix="/incidents/{incident_id}", tags=["Evidence & Telemetry"])

@router.get("/evidence", response_model=List[EvidenceResponse])
def get_incident_evidence(incident_id: str, db: Session = Depends(get_db)):
    evidence_items = db.query(EvidenceModel).filter(EvidenceModel.incident_id == incident_id).all()
    return evidence_items

@router.get("/sensor-data")
def get_sensor_telemetry(incident_id: str, db: Session = Depends(get_db)):
    """Returns parsed 1Hz sensor telemetry for frontend charting (temperature, vibration, coolant pressure, rpm)."""
    sensor_ev = db.query(EvidenceModel).filter(
        EvidenceModel.incident_id == incident_id,
        EvidenceModel.file_type.ilike("%sensor%")
    ).first()

    if not sensor_ev or not os.path.exists(sensor_ev.file_path):
        return {"records": [], "metrics": []}

    df = pd.read_csv(sensor_ev.file_path)
    records = df.to_dict(orient="records")
    return {
        "incident_id": incident_id,
        "total_points": len(records),
        "columns": list(df.columns),
        "records": records
    }

@router.get("/summary")
def get_investigation_summary(incident_id: str, db: Session = Depends(get_db)):
    service = ReportService(db)
    try:
        return service.generate_investigation_summary(incident_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
