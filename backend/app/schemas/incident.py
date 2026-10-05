"""
Pydantic Schemas for Industrial Incidents.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from app.schemas.evidence import EvidenceResponse
from app.schemas.event import EventResponse

class IncidentBase(BaseModel):
    title: str
    machine_id: Optional[str] = None
    component_id: Optional[str] = None
    severity: str = "HIGH"
    description: Optional[str] = None
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)

class IncidentCreate(IncidentBase):
    id: Optional[str] = None

class IncidentResponse(IncidentBase):
    id: str
    status: str
    created_at: datetime
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    evidence_items: List[EvidenceResponse] = Field(default_factory=list)
    events: List[EventResponse] = Field(default_factory=list)

    class Config:
        from_attributes = True

class IncidentSummaryResponse(BaseModel):
    id: str
    title: str
    machine_id: Optional[str]
    component_id: Optional[str]
    status: str
    severity: str
    created_at: datetime
    event_count: int
    evidence_count: int
    gap_count: int
    conflict_count: int

    class Config:
        from_attributes = True
