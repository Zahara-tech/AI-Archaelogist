"""
Incidents REST API Endpoints.
CRUD, Evidence Upload, Synthetic Package Loader, and Processing triggers.
"""

import os
import shutil
import glob
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.database.models import IncidentModel, EvidenceModel
from app.schemas.incident import IncidentCreate, IncidentResponse, IncidentSummaryResponse
from app.schemas.evidence import EvidenceResponse
from app.services.incident_service import IncidentService
from app.core.config import settings

router = APIRouter(prefix="/incidents", tags=["Incidents"])

@router.post("", response_model=IncidentResponse)
def create_incident(data: IncidentCreate, db: Session = Depends(get_db)):
    service = IncidentService(db)
    incident = service.create_incident(data)
    return incident

@router.get("", response_model=List[IncidentSummaryResponse])
def list_incidents(db: Session = Depends(get_db)):
    incidents = db.query(IncidentModel).order_by(IncidentModel.created_at.desc()).all()
    results = []
    for inc in incidents:
        results.append(IncidentSummaryResponse(
            id=inc.id,
            title=inc.title,
            machine_id=inc.machine_id,
            component_id=inc.component_id,
            status=inc.status,
            severity=inc.severity,
            created_at=inc.created_at,
            event_count=len(inc.events),
            evidence_count=len(inc.evidence_items),
            gap_count=len(inc.reconstructions),
            conflict_count=len(inc.conflicts)
        ))
    return results

@router.get("/{incident_id}", response_model=IncidentResponse)
def get_incident(incident_id: str, db: Session = Depends(get_db)):
    service = IncidentService(db)
    incident = service.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
    return incident

@router.post("/{incident_id}/upload", response_model=EvidenceResponse)
async def upload_evidence(
    incident_id: str,
    file: UploadFile = File(...),
    file_type: str = Form("sensor_csv"),
    summary: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    service = IncidentService(db)
    incident = service.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    incident_dir = os.path.join(settings.UPLOAD_DIR, incident_id)
    os.makedirs(incident_dir, exist_ok=True)
    
    file_path = os.path.join(incident_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    evidence = service.add_evidence(
        incident_id=incident_id,
        file_name=file.filename,
        file_type=file_type,
        file_path=file_path,
        summary=summary
    )
    return evidence

@router.post("/{incident_id}/load-synthetic")
def load_synthetic_package(incident_id: str, scenario: str = "cnc_milling_incident_001", db: Session = Depends(get_db)):
    """Loads all synthetic evidence files into the incident for instant testing and investigation."""
    service = IncidentService(db)
    incident = service.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    scenario_dir = os.path.join(settings.SYNTHETIC_DIR, scenario)
    if not os.path.exists(scenario_dir):
        # Generate on-demand if not present
        from scripts.generate_synthetic_data import generate_cnc_incident_dataset
        generate_cnc_incident_dataset(scenario_dir)

    files_mapping = [
        ("sensors.csv", "sensor_csv", "1Hz telemetry: Temperature, Vibration, Coolant Pressure, RPM, Power, State"),
        ("machine_logs.csv", "machine_log", "PLC and CNC controller event codes and alarm logs"),
        ("operator_report.txt", "operator_report", "Witness statement and physical actions by Operator OP-302"),
        ("maintenance_report.txt", "maintenance_report", "Diagnostic log and failure findings by Technician MNT-109"),
        ("production_records.csv", "production_record", "MES batch traveler and scrap disposition record"),
        ("cctv_metadata.json", "cctv_video", "Optical surveillance detections from CAM-STATION-04-OVERHEAD")
    ]

    added = []
    for fname, ftype, summary in files_mapping:
        fpath = os.path.join(scenario_dir, fname)
        if os.path.exists(fpath):
            ev = service.add_evidence(
                incident_id=incident_id,
                file_name=fname,
                file_type=ftype,
                file_path=fpath,
                summary=summary
            )
            added.append(ev.file_name)

    return {"incident_id": incident_id, "loaded_files": added, "status": "evidence_loaded"}

@router.post("/{incident_id}/process")
def process_incident(incident_id: str, db: Session = Depends(get_db)):
    service = IncidentService(db)
    try:
        result = service.process_incident_evidence(incident_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
