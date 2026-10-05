"""
Multi-Source Evidence Conflict and Contradiction Detector.
Identifies temporal discrepancies, state contradictions, and entity mismatches across modalities.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from app.schemas.event import EventResponse
from app.schemas.timeline import ConflictItem
from app.core.config import settings

class ConflictDetector:
    def __init__(self, conflict_tolerance_seconds: float = settings.CONFLICT_TOLERANCE_SECONDS):
        self.conflict_tolerance_seconds = conflict_tolerance_seconds

    def detect_conflicts(self, incident_id: str, events: List[EventResponse]) -> List[ConflictItem]:
        conflicts: List[ConflictItem] = []
        
        # 1. Look for door opening / E-stop timing discrepancies between Machine Log and CCTV/Operator
        log_events = [e for e in events if e.source == "machine_log"]
        cctv_events = [e for e in events if e.source == "cctv_video"]
        operator_events = [e for e in events if e.source == "operator_report"]

        # Check: HMI fault ack / door open vs CCTV visual door open
        for log_e in log_events:
            if "DOOR" in log_e.description.upper() or "HMI" in log_e.description.upper() or "INTERLOCK" in log_e.description.upper():
                for cctv_e in cctv_events:
                    if "DOOR" in cctv_e.description.upper() or "SAFETY" in cctv_e.description.upper():
                        diff = abs((log_e.timestamp_start - cctv_e.timestamp_start).total_seconds())
                        if diff >= self.conflict_tolerance_seconds:
                            conflicts.append(ConflictItem(
                                id=f"CONF_{log_e.id}_{cctv_e.id}",
                                incident_id=incident_id,
                                conflict_type="timestamp_mismatch",
                                description=(
                                    f"Temporal discrepancy of {diff:.1f}s detected regarding safety enclosure access / operator intervention: "
                                    f"PLC machine log reports '{log_e.description}' at {log_e.timestamp_start.strftime('%H:%M:%S')}, "
                                    f"whereas CCTV visual surveillance recorded '{cctv_e.description}' at {cctv_e.timestamp_start.strftime('%H:%M:%S')}."
                                ),
                                source_a=log_e.source,
                                source_b=cctv_e.source,
                                evidence_a={
                                    "event_id": log_e.id,
                                    "description": log_e.description,
                                    "timestamp": log_e.timestamp_start.isoformat(),
                                    "confidence": log_e.confidence,
                                    "evidence_refs": [ref.model_dump() for ref in log_e.evidence_refs]
                                },
                                evidence_b={
                                    "event_id": cctv_e.id,
                                    "description": cctv_e.description,
                                    "timestamp": cctv_e.timestamp_start.isoformat(),
                                    "confidence": cctv_e.confidence,
                                    "evidence_refs": [ref.model_dump() for ref in cctv_e.evidence_refs]
                                },
                                time_discrepancy_seconds=diff,
                                status="open",
                                created_at=datetime.utcnow()
                            ))

        # Check: Log fault acknowledgment vs Operator stated rush time
        for log_e in log_events:
            if "HMI_ACK" in log_e.description.upper():
                for op_e in operator_events:
                    if "EMERGENCY" in op_e.description.upper() or "FLUSH" in op_e.description.upper():
                        diff = abs((log_e.timestamp_start - op_e.timestamp_start).total_seconds())
                        if diff >= 20.0: # Significant lag between physical action and controller acknowledgment
                            conflicts.append(ConflictItem(
                                id=f"CONF_ACK_{log_e.id}_{op_e.id}",
                                incident_id=incident_id,
                                conflict_type="delayed_acknowledgment_discrepancy",
                                description=(
                                    f"Discrepancy of {diff:.1f}s between human operator action and controller registration: "
                                    f"Operator reports executing emergency response at ~{op_e.timestamp_start.strftime('%H:%M:%S')}, "
                                    f"while PLC HMI log shows fault acknowledgment logged at {log_e.timestamp_start.strftime('%H:%M:%S')}."
                                ),
                                source_a=op_e.source,
                                source_b=log_e.source,
                                evidence_a={
                                    "event_id": op_e.id,
                                    "description": op_e.description,
                                    "timestamp": op_e.timestamp_start.isoformat(),
                                    "certainty": op_e.certainty_type
                                },
                                evidence_b={
                                    "event_id": log_e.id,
                                    "description": log_e.description,
                                    "timestamp": log_e.timestamp_start.isoformat(),
                                    "certainty": log_e.certainty_type
                                },
                                time_discrepancy_seconds=diff,
                                status="open",
                                created_at=datetime.utcnow()
                            ))

        # Deduplicate conflicts by id
        unique_conflicts = {c.id: c for c in conflicts}
        return list(unique_conflicts.values())
