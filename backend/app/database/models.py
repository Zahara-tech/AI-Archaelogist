"""
SQLAlchemy ORM Models for AI Archaeologist.
Traceability from Raw Evidence -> Extracted Events -> Reconstructions & Conflicts.
"""

from sqlalchemy import Column, String, Float, DateTime, Text, ForeignKey, JSON, Integer
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database.connection import Base

class IncidentModel(Base):
    __tablename__ = "incidents"
    
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    machine_id = Column(String, index=True, nullable=True)
    component_id = Column(String, index=True, nullable=True)
    status = Column(String, default="created", index=True) # created, processing, analyzed, reconstructed, closed
    severity = Column(String, default="HIGH") # CRITICAL, HIGH, MEDIUM, LOW
    created_at = Column(DateTime, default=datetime.utcnow)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    description = Column(Text, nullable=True)
    extra_metadata = Column(JSON, default=dict)
    
    # Relationships
    evidence_items = relationship("EvidenceModel", back_populates="incident", cascade="all, delete-orphan")
    events = relationship("EventModel", back_populates="incident", cascade="all, delete-orphan")
    reconstructions = relationship("ReconstructedEventModel", back_populates="incident", cascade="all, delete-orphan")
    conflicts = relationship("ConflictModel", back_populates="incident", cascade="all, delete-orphan")

class EvidenceModel(Base):
    __tablename__ = "evidence"
    
    id = Column(String, primary_key=True, index=True)
    incident_id = Column(String, ForeignKey("incidents.id"), index=True, nullable=False)
    file_name = Column(String, nullable=False)
    file_type = Column(String, nullable=False) # sensor_csv, machine_log, operator_report, maintenance_report, production_record, cctv_video
    file_path = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Integer, default=0) # 0 = false, 1 = true
    summary = Column(Text, nullable=True)
    extra_metadata = Column(JSON, default=dict)
    
    incident = relationship("IncidentModel", back_populates="evidence_items")

class EventModel(Base):
    __tablename__ = "events"
    
    id = Column(String, primary_key=True, index=True)
    incident_id = Column(String, ForeignKey("incidents.id"), index=True, nullable=False)
    timestamp_start = Column(DateTime, index=True, nullable=False)
    timestamp_end = Column(DateTime, nullable=True)
    event_type = Column(String, index=True, nullable=False)
    description = Column(Text, nullable=False)
    source = Column(String, index=True, nullable=False) # sensor, machine_log, operator_report, maintenance_report, cctv_video, production_record, inferred_reconstruction
    confidence = Column(Float, default=1.0)
    certainty_type = Column(String, default="observed") # observed, reported, inferred, conflicted
    observed_or_inferred = Column(String, default="observed") # "observed" vs "inferred"
    machine_id = Column(String, nullable=True)
    component_id = Column(String, nullable=True)
    location = Column(String, nullable=True)
    evidence_refs = Column(JSON, default=list) # Traceability back to specific row, line, frame
    extra_metadata = Column(JSON, default=dict)
    
    incident = relationship("IncidentModel", back_populates="events")

class ReconstructedEventModel(Base):
    __tablename__ = "reconstructed_events"
    
    id = Column(String, primary_key=True, index=True)
    incident_id = Column(String, ForeignKey("incidents.id"), index=True, nullable=False)
    gap_start = Column(DateTime, nullable=False)
    gap_end = Column(DateTime, nullable=False)
    candidate_event_type = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    confidence = Column(Float, default=0.75)
    supporting_evidence = Column(JSON, default=list)
    conflicting_evidence = Column(JSON, default=list)
    reasoning = Column(Text, nullable=False)
    status = Column(String, default="candidate") # candidate, accepted, rejected
    created_at = Column(DateTime, default=datetime.utcnow)
    
    incident = relationship("IncidentModel", back_populates="reconstructions")

class ConflictModel(Base):
    __tablename__ = "conflicts"
    
    id = Column(String, primary_key=True, index=True)
    incident_id = Column(String, ForeignKey("incidents.id"), index=True, nullable=False)
    conflict_type = Column(String, nullable=False) # timestamp_mismatch, state_contradiction, entity_discrepancy
    description = Column(Text, nullable=False)
    source_a = Column(String, nullable=False)
    source_b = Column(String, nullable=False)
    evidence_a = Column(JSON, default=dict)
    evidence_b = Column(JSON, default=dict)
    time_discrepancy_seconds = Column(Float, default=0.0)
    status = Column(String, default="open") # open, resolved
    created_at = Column(DateTime, default=datetime.utcnow)
    
    incident = relationship("IncidentModel", back_populates="conflicts")
