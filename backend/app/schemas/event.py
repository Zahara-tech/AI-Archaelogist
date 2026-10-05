"""
Pydantic Schemas for Standardized Industrial Events.
Adheres strictly to the Common Event Schema specification.
"""

from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field

CertaintyType = Literal["observed", "reported", "inferred", "conflicted"]
ObservedOrInferred = Literal["observed", "inferred"]
SourceType = Literal[
    "sensor", 
    "machine_log", 
    "operator_report", 
    "maintenance_report", 
    "cctv_video", 
    "production_record", 
    "inferred_reconstruction"
]

class EvidenceReference(BaseModel):
    source_type: str = Field(..., description="Type of source e.g. sensor_csv, machine_log")
    file_name: Optional[str] = None
    line_number: Optional[int] = None
    row_index: Optional[int] = None
    timestamp_raw: Optional[str] = None
    metric_name: Optional[str] = None
    metric_value: Optional[Any] = None
    bounding_box: Optional[List[float]] = None
    quote: Optional[str] = None
    details: Optional[Dict[str, Any]] = Field(default_factory=dict)

class EventBase(BaseModel):
    timestamp_start: datetime
    timestamp_end: Optional[datetime] = None
    event_type: str
    description: str
    source: SourceType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    certainty_type: CertaintyType = "observed"
    observed_or_inferred: ObservedOrInferred = "observed"
    machine_id: Optional[str] = None
    component_id: Optional[str] = None
    location: Optional[str] = None
    evidence_refs: List[EvidenceReference] = Field(default_factory=list)
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)

class EventCreate(EventBase):
    id: Optional[str] = None
    incident_id: str

class EventResponse(EventBase):
    id: str
    incident_id: str

    class Config:
        from_attributes = True
