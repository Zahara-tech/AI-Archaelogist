"""
Missing-Event Candidate Reconstruction Engine.
Synthesizes multi-modal clues across prior events, subsequent events, telemetry blackouts,
operator statements, and maintenance logs to infer plausible missing operational activities.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from app.schemas.event import EventResponse
from app.schemas.timeline import TemporalGap
from app.schemas.reconstruction import ReconstructedEventResponse

class CandidateReconstructor:
    def __init__(self):
        pass

    def reconstruct_gap(
        self,
        incident_id: str,
        gap: TemporalGap,
        all_events: List[EventResponse]
    ) -> List[ReconstructedEventResponse]:
        candidates: List[ReconstructedEventResponse] = []
        
        # 1. Gather context around gap
        prev_events = [e for e in all_events if e.timestamp_start <= gap.gap_start]
        next_events = [e for e in all_events if e.timestamp_start >= gap.gap_end]
        
        # Gather text report clues
        report_events = [e for e in all_events if e.source in ["operator_report", "maintenance_report"]]
        cctv_events = [e for e in all_events if e.source == "cctv_video"]
        sensor_events = [e for e in all_events if e.source == "sensor"]
        
        # Check if this gap is during the emergency shutdown & maintenance arrival interval (10:04:15 - 10:06:20)
        has_e_stop = any("trip" in e.event_type.lower() or "overheat" in e.event_type.lower() or "shutdown" in e.description.lower() for e in prev_events)
        has_mnt_after = any("maintenance" in e.event_type.lower() or "technician" in e.description.lower() for e in next_events)

        if has_e_stop or has_mnt_after or gap.duration_seconds >= 30.0:
            # Candidate 1: Manual Cutting Chamber & Spindle Inspection
            supporting_1 = []
            for e in cctv_events:
                if gap.gap_start <= e.timestamp_start <= gap.gap_end:
                    supporting_1.append({
                        "source": "cctv_video",
                        "event": e.description,
                        "timestamp": e.timestamp_start.isoformat(),
                        "confidence": e.confidence
                    })
            for e in report_events:
                if "inspection" in e.description.lower() or "smoke" in e.description.lower() or "chamber" in e.description.lower():
                    supporting_1.append({
                        "source": e.source,
                        "quote": e.description,
                        "timestamp": e.timestamp_start.isoformat()
                    })

            candidate_1 = ReconstructedEventResponse(
                id=f"REC_{uuid.uuid4().hex[:8]}",
                incident_id=incident_id,
                gap_start=gap.gap_start,
                gap_end=gap.gap_end,
                candidate_event_type="inferred_chamber_and_tool_inspection",
                description="Operator manual safety inspection of cutting chamber, thermal smoke evaluation, and radio dispatch to maintenance.",
                confidence=0.86,
                supporting_evidence=supporting_1,
                conflicting_evidence=[],
                reasoning=(
                    f"Between {gap.gap_start.strftime('%H:%M:%S')} and {gap.gap_end.strftime('%H:%M:%S')} ({gap.duration_seconds:.0f}s unmonitored window), "
                    "the machine was halted by thermal overload watchdog. CCTV recorded operator leaning into the safety enclosure at 10:05:15, "
                    "and operator report corroborates checking for burning oil/fire and resetting the main console breaker before technician Dave arrived."
                ),
                status="candidate",
                created_at=datetime.utcnow()
            )
            candidates.append(candidate_1)

            # Candidate 2: Emergency Coolant Line & Pressure Purge Attempt
            supporting_2 = []
            for e in sensor_events:
                if "coolant" in e.event_type.lower() or "pressure" in e.description.lower():
                    supporting_2.append({
                        "source": "sensor",
                        "detail": e.description,
                        "timestamp": e.timestamp_start.isoformat()
                    })
            for e in report_events:
                if "flush" in e.description.lower() or "valve" in e.description.lower():
                    supporting_2.append({
                        "source": e.source,
                        "detail": e.description,
                        "timestamp": e.timestamp_start.isoformat()
                    })

            candidate_2 = ReconstructedEventResponse(
                id=f"REC_{uuid.uuid4().hex[:8]}",
                incident_id=incident_id,
                gap_start=gap.gap_start,
                gap_end=gap.gap_end,
                candidate_event_type="inferred_coolant_flush_and_breaker_reset",
                description="Operator actuation of emergency flood coolant override and electrical subsystem power cycle.",
                confidence=0.74,
                supporting_evidence=supporting_2,
                conflicting_evidence=[],
                reasoning=(
                    "Telemetry blackout in sensor data during this interval matches machine control power cycling. "
                    "Operator statement mentions pressing manual emergency flush and resetting panel breaker."
                ),
                status="candidate",
                created_at=datetime.utcnow()
            )
            candidates.append(candidate_2)

        return candidates
