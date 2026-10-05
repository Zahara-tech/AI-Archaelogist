"""
Pydantic Schemas for Evidence Items and Uploads.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class EvidenceBase(BaseModel):
    file_name: str
    file_type: str
    file_path: str
    summary: Optional[str] = None
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)

class EvidenceCreate(EvidenceBase):
    incident_id: str

class EvidenceResponse(EvidenceBase):
    id: str
    incident_id: str
    uploaded_at: datetime
    processed: int

    class Config:
        from_attributes = True
