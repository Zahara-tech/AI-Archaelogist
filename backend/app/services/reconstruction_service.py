"""
Missing-Event Reconstruction Service.
Executes gap analysis and candidate hypothesis generation across multi-modal evidence.
"""

from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime
import uuid

from app.database.models import IncidentModel, EventModel, ReconstructedEventModel
from app.schemas.event import EventResponse, EvidenceReference
from app.schemas.reconstruction import ReconstructedEventResponse, ReconstructionResultResponse
from app.reasoning.missing_event_detector import MissingEventDetector
from app.reasoning.candidate_reconstructor import CandidateReconstructor

class ReconstructionService:
    def __init__(self, db: Session):
        self.db = db
        self.missing_detector = MissingEventDetector()
        self.reconstructor = CandidateReconstructor()

    def reconstruct_incident_gaps(self, incident_id: str, gap_id: Optional[str] = None) -> ReconstructionResultResponse:
        incident = self.db.query(IncidentModel).filter(IncidentModel.id == incident_id).first()
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")

        # Load all events for this incident
        db_events = self.db.query(EventModel).filter(EventModel.incident_id == incident_id).all()
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

        gaps = self.missing_detector.detect_gaps(event_responses, machine_id=incident.machine_id)
        
        if gap_id:
            gaps = [g for g in gaps if g.gap_id == gap_id]

        all_candidates: List[ReconstructedEventResponse] = []

        # Clear existing reconstructions for these gaps to prevent duplicates
        self.db.query(ReconstructedEventModel).filter(ReconstructedEventModel.incident_id == incident_id).delete()
        self.db.commit()

        for gap in gaps:
            candidates = self.reconstructor.reconstruct_gap(incident_id, gap, event_responses)
            for cand in candidates:
                # Save to database
                db_rec = ReconstructedEventModel(
                    id=cand.id,
                    incident_id=incident_id,
                    gap_start=cand.gap_start,
                    gap_end=cand.gap_end,
                    candidate_event_type=cand.candidate_event_type,
                    description=cand.description,
                    confidence=cand.confidence,
                    supporting_evidence=cand.supporting_evidence,
                    conflicting_evidence=cand.conflicting_evidence,
                    reasoning=cand.reasoning,
                    status=cand.status,
                    created_at=cand.created_at
                )
                self.db.add(db_rec)
                all_candidates.append(cand)

        self.db.commit()

        summary = (
            f"Analyzed {len(gaps)} temporal gap(s). Generated {len(all_candidates)} plausible "
            "missing-event reconstruction candidates supported by video observations, sensor patterns, "
            "and operator statements."
        )

        return ReconstructionResultResponse(
            incident_id=incident_id,
            gaps_analyzed=len(gaps),
            candidates=all_candidates,
            summary=summary
        )

    def get_reconstructions(self, incident_id: str) -> List[ReconstructedEventResponse]:
        db_recs = self.db.query(ReconstructedEventModel).filter(ReconstructedEventModel.incident_id == incident_id).all()
        return [
            ReconstructedEventResponse(
                id=r.id,
                incident_id=r.incident_id,
                gap_start=r.gap_start,
                gap_end=r.gap_end,
                candidate_event_type=r.candidate_event_type,
                description=r.description,
                confidence=r.confidence,
                supporting_evidence=r.supporting_evidence,
                conflicting_evidence=r.conflicting_evidence,
                reasoning=r.reasoning,
                status=r.status,
                created_at=r.created_at
            )
            for r in db_recs
        ]
