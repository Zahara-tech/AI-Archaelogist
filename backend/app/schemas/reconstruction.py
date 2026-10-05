"""
Pydantic Schemas for Missing-Event Reconstruction and Hypotheses.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class ReconstructedEventCreate(BaseModel):
    incident_id: str
    gap_start: datetime
    gap_end: datetime
    candidate_event_type: str
    description: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    conflicting_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning: str
    status: str = "candidate"

class ReconstructedEventResponse(ReconstructedEventCreate):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

class ReconstructionRequest(BaseModel):
    gap_id: Optional[str] = None
    use_llm: bool = True

class ReconstructionResultResponse(BaseModel):
    incident_id: str
    gaps_analyzed: int
    candidates: List[ReconstructedEventResponse]
    summary: str
