"""
Temporal Event Engine.
Coordinates multi-modal event normalization, chronological sorting, gap detection,
conflict identification, and unified timeline generation.
"""

from typing import List, Optional
from datetime import datetime
from app.schemas.event import EventResponse
from app.schemas.timeline import UnifiedTimelineResponse, TemporalGap, ConflictItem
from app.reasoning.missing_event_detector import MissingEventDetector
from app.reasoning.conflict_detector import ConflictDetector

class TemporalEventEngine:
    def __init__(self):
        self.missing_detector = MissingEventDetector()
        self.conflict_detector = ConflictDetector()

    def build_timeline(
        self,
        incident_id: str,
        events: List[EventResponse],
        machine_id: Optional[str] = None
    ) -> UnifiedTimelineResponse:
        if not events:
            return UnifiedTimelineResponse(
                incident_id=incident_id,
                total_events=0,
                observed_events_count=0,
                inferred_events_count=0,
                events=[],
                temporal_gaps=[],
                conflicts=[]
            )

        # 1. Normalize and Sort chronologically
        sorted_events = sorted(events, key=lambda e: e.timestamp_start)

        # 2. Count observed vs inferred
        observed_count = sum(1 for e in sorted_events if e.observed_or_inferred == "observed")
        inferred_count = sum(1 for e in sorted_events if e.observed_or_inferred == "inferred")

        # 3. Detect Temporal Gaps
        temporal_gaps = self.missing_detector.detect_gaps(sorted_events, machine_id=machine_id)

        # 4. Detect Multi-Source Conflicts
        conflicts = self.conflict_detector.detect_conflicts(incident_id, sorted_events)

        # 5. Determine overall time range
        start_time = sorted_events[0].timestamp_start if sorted_events else None
        end_time = sorted_events[-1].timestamp_start if sorted_events else None

        return UnifiedTimelineResponse(
            incident_id=incident_id,
            total_events=len(sorted_events),
            observed_events_count=observed_count,
            inferred_events_count=inferred_count,
            time_range_start=start_time,
            time_range_end=end_time,
            events=sorted_events,
            temporal_gaps=temporal_gaps,
            conflicts=conflicts
        )
