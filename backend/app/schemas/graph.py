"""
Pydantic Schemas for NetworkX Evidence Graph Export.
Nodes, Edges, and Causal Links for UI Visualization.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class GraphNode(BaseModel):
    id: str
    label: str
    type: str # event_observed, event_inferred, evidence_sensor, evidence_log, evidence_report, evidence_video, conflict
    source: Optional[str] = None
    timestamp: Optional[str] = None
    confidence: Optional[float] = None
    certainty: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)

class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str # occurred_before, supported_by, conflicts_with, inferred_from, caused_by
    label: Optional[str] = None
    weight: float = 1.0

class EvidenceGraphResponse(BaseModel):
    incident_id: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    stats: Dict[str, Any] = Field(default_factory=dict)
