"""
Missing-Event Inference and Reconstruction Endpoints.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.schemas.reconstruction import ReconstructionRequest, ReconstructionResultResponse, ReconstructedEventResponse
from app.services.reconstruction_service import ReconstructionService

router = APIRouter(prefix="/incidents/{incident_id}", tags=["Reconstruction"])

@router.post("/reconstruct", response_model=ReconstructionResultResponse)
def run_reconstruction(
    incident_id: str,
    req: ReconstructionRequest = ReconstructionRequest(),
    db: Session = Depends(get_db)
):
    service = ReconstructionService(db)
    try:
        result = service.reconstruct_incident_gaps(incident_id, gap_id=req.gap_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {str(e)}")

@router.get("/reconstructions", response_model=List[ReconstructedEventResponse])
def get_reconstructions(incident_id: str, db: Session = Depends(get_db)):
    service = ReconstructionService(db)
    return service.get_reconstructions(incident_id)
