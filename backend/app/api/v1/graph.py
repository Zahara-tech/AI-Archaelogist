"""
NetworkX Evidence Graph Visualization Endpoint.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.database.models import EventModel, ConflictModel, ReconstructedEventModel
from app.schemas.event import EventResponse, EvidenceReference
from app.schemas.timeline import ConflictItem
from app.schemas.reconstruction import ReconstructedEventResponse
from app.schemas.graph import EvidenceGraphResponse
from app.reasoning.evidence_graph import EvidenceGraphBuilder

router = APIRouter(prefix="/incidents/{incident_id}", tags=["Evidence Graph"])

@router.get("/graph", response_model=EvidenceGraphResponse)
def get_incident_graph(incident_id: str, db: Session = Depends(get_db)):
    db_events = db.query(EventModel).filter(EventModel.incident_id == incident_id).all()
    if not db_events:
        raise HTTPException(status_code=404, detail="No processed events found for this incident. Run /process first.")

    event_responses = [
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

    conflicts = [
        ConflictItem(
            id=c.id,
            incident_id=c.incident_id,
            conflict_type=c.conflict_type,
            description=c.description,
            source_a=c.source_a,
            source_b=c.source_b,
            evidence_a=c.evidence_a or {},
            evidence_b=c.evidence_b or {},
            time_discrepancy_seconds=c.time_discrepancy_seconds,
            status=c.status,
            created_at=c.created_at
        )
        for c in db.query(ConflictModel).filter(ConflictModel.incident_id == incident_id).all()
    ]

    reconstructions = [
        ReconstructedEventResponse(
            id=r.id,
            incident_id=r.incident_id,
            gap_start=r.gap_start,
            gap_end=r.gap_end,
            candidate_event_type=r.candidate_event_type,
            description=r.description,
            confidence=r.confidence,
            supporting_evidence=r.supporting_evidence or [],
            conflicting_evidence=r.conflicting_evidence or [],
            reasoning=r.reasoning,
            status=r.status,
            created_at=r.created_at
        )
        for r in db.query(ReconstructedEventModel).filter(ReconstructedEventModel.incident_id == incident_id).all()
    ]

    builder = EvidenceGraphBuilder()
    return builder.build_graph(incident_id, event_responses, conflicts, reconstructions)
