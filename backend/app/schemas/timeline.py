"""
Pydantic Schemas for Unified Timeline, Temporal Gaps, and Discrepancies.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from app.schemas.event import EventResponse

class TemporalGap(BaseModel):
    gap_id: str
    gap_start: datetime
    gap_end: datetime
    duration_seconds: float
    previous_event_id: Optional[str] = None
    previous_event_desc: Optional[str] = None
    next_event_id: Optional[str] = None
    next_event_desc: Optional[str] = None
    machine_id: Optional[str] = None
    description: str
    reconstruction_candidates_count: int = 0

class ConflictItem(BaseModel):
    id: str
    incident_id: str
    conflict_type: str
    description: str
    source_a: str
    source_b: str
    evidence_a: Dict[str, Any] = Field(default_factory=dict)
    evidence_b: Dict[str, Any] = Field(default_factory=dict)
    time_discrepancy_seconds: float = 0.0
    status: str = "open"
    created_at: datetime

    class Config:
        from_attributes = True

class UnifiedTimelineResponse(BaseModel):
    incident_id: str
    total_events: int
    observed_events_count: int
    inferred_events_count: int
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    events: List[EventResponse] = Field(default_factory=list)
    temporal_gaps: List[TemporalGap] = Field(default_factory=list)
    conflicts: List[ConflictItem] = Field(default_factory=list)
