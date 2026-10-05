"""
Unit and Integration Tests for Temporal Engine, Gaps, Conflicts, and Graph.
"""

import os
import pytest
from app.database.connection import SessionLocal, init_db
from app.services.incident_service import IncidentService
from app.services.reconstruction_service import ReconstructionService
from app.schemas.incident import IncidentCreate
from app.reasoning.evidence_graph import EvidenceGraphBuilder

def test_end_to_end_pipeline():
    init_db()
    db = SessionLocal()
    try:
        service = IncidentService(db)
        
        # 1. Create Incident
        inc = service.create_incident(IncidentCreate(
            id="INC-E2E-TEST-001",
            title="E2E Thermal Seizure Test Incident",
            machine_id="CNC-MILL-04",
            component_id="AERO-BLADE-8842",
            severity="CRITICAL",
            description="Test incident for end-to-end verification"
        ))
        assert inc.id == "INC-E2E-TEST-001"

        # 2. Add Evidence
        syn_dir = "data/synthetic/cnc_milling_incident_001"
        service.add_evidence(inc.id, "sensors.csv", "sensor_csv", os.path.join(syn_dir, "sensors.csv"))
        service.add_evidence(inc.id, "machine_logs.csv", "machine_log", os.path.join(syn_dir, "machine_logs.csv"))
        service.add_evidence(inc.id, "operator_report.txt", "operator_report", os.path.join(syn_dir, "operator_report.txt"))
        service.add_evidence(inc.id, "maintenance_report.txt", "maintenance_report", os.path.join(syn_dir, "maintenance_report.txt"))
        service.add_evidence(inc.id, "cctv_metadata.json", "cctv_video", os.path.join(syn_dir, "cctv_metadata.json"))
        service.add_evidence(inc.id, "production_records.csv", "production_record", os.path.join(syn_dir, "production_records.csv"))

        # 3. Process Evidence
        result = service.process_incident_evidence(inc.id)
        assert result["total_events"] > 20
        assert result["gaps_detected"] > 0
        assert result["conflicts_detected"] > 0

        # 4. Get Unified Timeline
        timeline = service.get_timeline(inc.id)
        assert timeline.total_events > 20
        assert len(timeline.temporal_gaps) > 0
        assert len(timeline.conflicts) > 0

        # Verify conflict details (HMI / Door Open mismatch)
        conflict_types = [c.conflict_type for c in timeline.conflicts]
        assert "timestamp_mismatch" in conflict_types

        # 5. Reconstruct Missing Events
        rec_service = ReconstructionService(db)
        rec_result = rec_service.reconstruct_incident_gaps(inc.id)
        assert rec_result.gaps_analyzed > 0
        assert len(rec_result.candidates) > 0

        # Verify candidate properties
        for cand in rec_result.candidates:
            assert cand.confidence > 0.0
            assert cand.confidence <= 1.0
            assert len(cand.supporting_evidence) > 0
            assert cand.reasoning is not None

        # 6. Build Evidence Graph
        graph_builder = EvidenceGraphBuilder()
        graph = graph_builder.build_graph(inc.id, timeline.events, timeline.conflicts, rec_result.candidates)
        assert graph.stats["total_nodes"] > 25
        assert graph.stats["total_edges"] > 20

    finally:
        db.close()
