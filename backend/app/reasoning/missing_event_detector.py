"""
Missing-Event and Temporal Gap Detector.
Identifies unexplained intervals, telemetry blackouts, or discontinuous operational transitions
where critical activity occurred without direct automated logging.
"""

from typing import List, Optional
from datetime import datetime, timedelta
from app.schemas.event import EventResponse
from app.schemas.timeline import TemporalGap
from app.core.config import settings

class MissingEventDetector:
    def __init__(self, gap_threshold_seconds: float = settings.TEMPORAL_GAP_THRESHOLD_SECONDS):
        self.gap_threshold_seconds = gap_threshold_seconds

    def detect_gaps(self, events: List[EventResponse], machine_id: Optional[str] = None) -> List[TemporalGap]:
        if not events or len(events) < 2:
            return []

        # Ensure events are sorted by timestamp_start
        sorted_events = sorted(events, key=lambda e: e.timestamp_start)
        gaps: List[TemporalGap] = []
        
        # Filter for high-impact events or general sequence
        for i in range(len(sorted_events) - 1):
            curr_evt = sorted_events[i]
            next_evt = sorted_events[i + 1]

            # Use end timestamp if available, else start timestamp
            t_curr = curr_evt.timestamp_end if curr_evt.timestamp_end else curr_evt.timestamp_start
            t_next = next_evt.timestamp_start

            duration = (t_next - t_curr).total_seconds()

            if duration >= self.gap_threshold_seconds:
                gap_id = f"GAP_{curr_evt.id}_{next_evt.id}"
                gap_desc = (
                    f"Temporal gap of {duration:.1f}s identified between "
                    f"'{curr_evt.description[:60]}...' ({curr_evt.timestamp_start.strftime('%H:%M:%S')}) and "
                    f"'{next_evt.description[:60]}...' ({next_evt.timestamp_start.strftime('%H:%M:%S')}). "
                    f"Machine state or human intervention during this window requires reconstruction."
                )

                gaps.append(TemporalGap(
                    gap_id=gap_id,
                    gap_start=t_curr,
                    gap_end=t_next,
                    duration_seconds=duration,
                    previous_event_id=curr_evt.id,
                    previous_event_desc=curr_evt.description,
                    next_event_id=next_evt.id,
                    next_event_desc=next_evt.description,
                    machine_id=machine_id or curr_evt.machine_id,
                    description=gap_desc,
                    reconstruction_candidates_count=0
                ))

        return gaps
