"""
Incident Investigation Service.
Orchestrates multi-modal evidence ingestion, processor execution, event persistence,
gap analysis, and conflict detection.
"""

import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.database.models import IncidentModel, EvidenceModel, EventModel, ConflictModel, ReconstructedEventModel
from app.schemas.incident import IncidentCreate, IncidentResponse
from app.schemas.evidence import EvidenceCreate
from app.schemas.event import EventBase, EventResponse, EvidenceReference
from app.schemas.timeline import UnifiedTimelineResponse
from app.processors.sensor_processor import SensorProcessor
from app.processors.log_processor import LogProcessor
from app.processors.text_processor import TextReportProcessor
from app.processors.video_processor import VideoProcessor
from app.processors.production_processor import ProductionProcessor
from app.reasoning.temporal_engine import TemporalEventEngine

class IncidentService:
    def __init__(self, db: Session):
        self.db = db
        self.temporal_engine = TemporalEventEngine()
        self.sensor_proc = SensorProcessor()
        self.log_proc = LogProcessor()
        self.op_report_proc = TextReportProcessor("operator_report")
        self.mnt_report_proc = TextReportProcessor("maintenance_report")
        self.video_proc = VideoProcessor()
        self.prod_proc = ProductionProcessor()

    def create_incident(self, data: IncidentCreate) -> IncidentModel:
        incident_id = data.id or f"INC-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        incident = IncidentModel(
            id=incident_id,
            title=data.title,
            machine_id=data.machine_id,
            component_id=data.component_id,
            severity=data.severity,
            description=data.description,
            extra_metadata=data.extra_metadata or {}
        )
        self.db.add(incident)
        self.db.commit()
        self.db.refresh(incident)
        return incident

    def get_incident(self, incident_id: str) -> Optional[IncidentModel]:
        return self.db.query(IncidentModel).filter(IncidentModel.id == incident_id).first()

    def list_incidents(self) -> List[IncidentModel]:
        return self.db.query(IncidentModel).order_by(IncidentModel.created_at.desc()).all()

    def add_evidence(self, incident_id: str, file_name: str, file_type: str, file_path: str, summary: Optional[str] = None) -> EvidenceModel:
        evidence = EvidenceModel(
            id=f"EVID-{uuid.uuid4().hex[:8].upper()}",
            incident_id=incident_id,
            file_name=file_name,
            file_type=file_type,
            file_path=file_path,
            summary=summary,
            uploaded_at=datetime.utcnow(),
            processed=0
        )
        self.db.add(evidence)
        self.db.commit()
        self.db.refresh(evidence)
        return evidence

    def process_incident_evidence(self, incident_id: str) -> Dict[str, Any]:
        incident = self.get_incident(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")

        incident.status = "processing"
        self.db.commit()

        # Clear existing extracted events & conflicts to allow re-processing
        self.db.query(EventModel).filter(EventModel.incident_id == incident_id, EventModel.observed_or_inferred == "observed").delete()
        self.db.query(ConflictModel).filter(ConflictModel.incident_id == incident_id).delete()
        self.db.commit()

        evidence_items = self.db.query(EvidenceModel).filter(EvidenceModel.incident_id == incident_id).all()
        extracted_events: List[EventBase] = []

        for item in evidence_items:
            path = item.file_path
            f_type = item.file_type.lower()
            f_name = item.file_name.lower()

            try:
                if "sensor" in f_type or f_name.endswith(".csv") and "sensor" in f_name:
                    evts = self.sensor_proc.process(path, incident_id)
                elif "log" in f_type or f_name.endswith(".csv") and "log" in f_name:
                    evts = self.log_proc.process(path, incident_id)
                elif "operator" in f_type or "operator" in f_name:
                    evts = self.op_report_proc.process(path, incident_id)
                elif "maintenance" in f_type or "maintenance" in f_name:
                    evts = self.mnt_report_proc.process(path, incident_id)
                elif "cctv" in f_type or "video" in f_type or f_name.endswith((".json", ".mp4", ".avi")):
                    evts = self.video_proc.process(path, incident_id)
                elif "production" in f_type or "record" in f_name:
                    evts = self.prod_proc.process(path, incident_id)
                else:
                    # Fallback based on extension
                    if f_name.endswith(".txt"):
                        evts = self.op_report_proc.process(path, incident_id)
                    else:
                        evts = []

                extracted_events.extend(evts)
                item.processed = 1
                item.summary = f"Extracted {len(evts)} events"
            except Exception as e:
                item.summary = f"Processing error: {str(e)}"

        # Save all extracted events to database
        db_events: List[EventModel] = []
        for i, ev in enumerate(extracted_events):
            event_id = f"EVT_{incident_id[:8]}_{ev.source[:4].upper()}_{i+1:03d}"
            db_event = EventModel(
                id=event_id,
                incident_id=incident_id,
                timestamp_start=ev.timestamp_start,
                timestamp_end=ev.timestamp_end,
                event_type=ev.event_type,
                description=ev.description,
                source=ev.source,
                confidence=ev.confidence,
                certainty_type=ev.certainty_type,
                observed_or_inferred=ev.observed_or_inferred,
                machine_id=ev.machine_id or incident.machine_id,
                component_id=ev.component_id or incident.component_id,
                location=ev.location,
                evidence_refs=[ref.model_dump() for ref in ev.evidence_refs],
                extra_metadata=ev.extra_metadata or {}
            )
            self.db.add(db_event)
            db_events.append(db_event)

        self.db.commit()

        # Convert to Pydantic responses for temporal analysis
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

        # Run Temporal Engine
        timeline = self.temporal_engine.build_timeline(incident_id, event_responses, machine_id=incident.machine_id)

        # Save detected conflicts to database
        for c in timeline.conflicts:
            db_conflict = ConflictModel(
                id=c.id,
                incident_id=incident_id,
                conflict_type=c.conflict_type,
                description=c.description,
                source_a=c.source_a,
                source_b=c.source_b,
                evidence_a=c.evidence_a,
                evidence_b=c.evidence_b,
                time_discrepancy_seconds=c.time_discrepancy_seconds,
                status=c.status,
                created_at=c.created_at
            )
            self.db.add(db_conflict)

        # Update Incident status and time window
        incident.status = "analyzed"
        if timeline.time_range_start:
            incident.start_time = timeline.time_range_start
        if timeline.time_range_end:
            incident.end_time = timeline.time_range_end

        self.db.commit()

        return {
            "incident_id": incident_id,
            "status": incident.status,
            "total_events": len(db_events),
            "gaps_detected": len(timeline.temporal_gaps),
            "conflicts_detected": len(timeline.conflicts),
            "start_time": timeline.time_range_start.isoformat() if timeline.time_range_start else None,
            "end_time": timeline.time_range_end.isoformat() if timeline.time_range_end else None
        }

    def get_timeline(self, incident_id: str) -> UnifiedTimelineResponse:
        incident = self.get_incident(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")

        # Load observed events + any accepted/candidate reconstructed events
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

        return self.temporal_engine.build_timeline(incident_id, event_responses, machine_id=incident.machine_id)
