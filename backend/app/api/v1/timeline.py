"""
Timeline, Events, and Conflict Endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.database.models import EventModel, ConflictModel
from app.schemas.event import EventResponse, EvidenceReference
from app.schemas.timeline import UnifiedTimelineResponse, ConflictItem
from app.services.incident_service import IncidentService

router = APIRouter(prefix="/incidents/{incident_id}", tags=["Timeline & Events"])

@router.get("/timeline", response_model=UnifiedTimelineResponse)
def get_incident_timeline(incident_id: str, db: Session = Depends(get_db)):
    service = IncidentService(db)
    try:
        return service.get_timeline(incident_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/events", response_model=List[EventResponse])
def get_incident_events(
    incident_id: str,
    source: Optional[str] = Query(None, description="Filter by source modality"),
    observed_or_inferred: Optional[str] = Query(None, description="Filter 'observed' or 'inferred'"),
    certainty_type: Optional[str] = Query(None, description="Filter by certainty type"),
    db: Session = Depends(get_db)
):
    query = db.query(EventModel).filter(EventModel.incident_id == incident_id)
    if source:
        query = query.filter(EventModel.source == source)
    if observed_or_inferred:
        query = query.filter(EventModel.observed_or_inferred == observed_or_inferred)
    if certainty_type:
        query = query.filter(EventModel.certainty_type == certainty_type)

    db_events = query.order_by(EventModel.timestamp_start.asc()).all()
    return [
        EventResponse(
            id=e.id,
            incident_id=e.incident_id,
            timestamp_start=e.timestamp_start,
            timestamp_end=e.timestamp_end,
            event_type=e.event_type,
            description=e.description,
            source=e.source,
            confidence=e.confidence,
            certainty_type=e.certainty_type,
            observed_or_inferred=e.observed_or_inferred,
            machine_id=e.machine_id,
            component_id=e.component_id,
            location=e.location,
            evidence_refs=[EvidenceReference(**ref) for ref in e.evidence_refs],
            extra_metadata=e.extra_metadata or {}
        )
        for e in db_events
    ]

@router.get("/conflicts", response_model=List[ConflictItem])
def get_incident_conflicts(incident_id: str, db: Session = Depends(get_db)):
    conflicts = db.query(ConflictModel).filter(ConflictModel.incident_id == incident_id).all()
    return conflicts
