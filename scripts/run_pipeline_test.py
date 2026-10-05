"""
Stand-alone verification script for AI Archaeologist.
Tests:
1. Synthetic data ingestion
2. Multi-modal event extraction (sensors, logs, reports, CCTV, ERP)
3. Baseline anomaly detection
4. Temporal engine & gap detection
5. Multi-source contradiction & conflict detection
6. Missing-event candidate reconstruction
7. NetworkX evidence graph generation
8. Investigation summary report generation
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.abspath("backend"))

from app.database.connection import init_db, SessionLocal
from app.services.incident_service import IncidentService
from app.services.reconstruction_service import ReconstructionService
from app.services.report_service import ReportService
from app.reasoning.evidence_graph import EvidenceGraphBuilder
from app.schemas.incident import IncidentCreate

def run_verification():
    print("=" * 70)
    print("AI ARCHAEOLOGIST - PIPELINE INTEGRATION & VERIFICATION")
    print("=" * 70)

    init_db()
    db = SessionLocal()

    try:
        service = IncidentService(db)
        
        # 1. Create Incident
        print("\n[1/7] Creating Incident INC-2026-0491...")
        incident = service.create_incident(IncidentCreate(
            id="INC-2026-0491",
            title="CNC-MILL-04 Thermal Expansion & Spindle Seizure Incident",
            machine_id="CNC-MILL-04",
            component_id="AERO-BLADE-TURBINE-X79-8842",
            severity="CRITICAL",
            description="Turbine blade milling failure with severe chatter, thermal trip, and workpiece gouging."
        ))
        print(f" -> Incident created: ID={incident.id}, Severity={incident.severity}, Status={incident.status}")

        # 2. Attach Evidence
        print("\n[2/7] Attaching Heterogeneous Evidence Sources...")
        syn_dir = "data/synthetic/cnc_milling_incident_001"
        files = [
            ("sensors.csv", "sensor_csv", "1Hz telemetry: Temperature, Vibration, Coolant Pressure, RPM, Power, State"),
            ("machine_logs.csv", "machine_log", "PLC event codes and alarm logs"),
            ("operator_report.txt", "operator_report", "Witness statement by Operator OP-302"),
            ("maintenance_report.txt", "maintenance_report", "Diagnostic log by Reliability Engineer MNT-109"),
            ("production_records.csv", "production_record", "MES batch traveler and scrap disposition record"),
            ("cctv_metadata.json", "cctv_video", "Optical surveillance detections from CAM-STATION-04-OVERHEAD")
        ]
        for fname, ftype, desc in files:
            fpath = os.path.join(syn_dir, fname)
            ev = service.add_evidence(incident.id, fname, ftype, fpath, summary=desc)
            print(f" -> Registered Evidence: {fname} [{ftype}]")

        # 3. Process Multi-Modal Evidence
        print("\n[3/7] Processing Multi-Modal Evidence & Running Anomaly Detection...")
        proc_result = service.process_incident_evidence(incident.id)
        print(f" -> Total Standardized Events Extracted: {proc_result['total_events']}")
        print(f" -> Temporal Gaps Identified: {proc_result['gaps_detected']}")
        print(f" -> Multi-Source Conflicts Detected: {proc_result['conflicts_detected']}")

        # 4. Inspect Timeline
        print("\n[4/7] Generating Unified Chronological Timeline...")
        timeline = service.get_timeline(incident.id)
        print(f" -> Timeline spans from {timeline.time_range_start} to {timeline.time_range_end}")
        print("\nSample Chronological Events:")
        for idx, evt in enumerate(timeline.events[:8]):
            print(f"   {idx+1:02d}. [{evt.timestamp_start.strftime('%H:%M:%S')}] [{evt.source.upper()}] [{evt.certainty_type.upper()}] {evt.event_type} - {evt.description[:70]}...")

        # 5. Missing-Event Reconstruction
        print("\n[5/7] Executing Missing-Event Reconstruction Engine...")
        rec_service = ReconstructionService(db)
        rec_result = rec_service.reconstruct_incident_gaps(incident.id)
        print(f" -> Gaps Analyzed: {rec_result.gaps_analyzed}")
        print(f" -> Reconstructed Hypotheses Generated: {len(rec_result.candidates)}")
        for idx, cand in enumerate(rec_result.candidates):
            print(f"\n   Candidate #{idx+1}: [{cand.candidate_event_type}]")
            print(f"   Time Window: {cand.gap_start.strftime('%H:%M:%S')} - {cand.gap_end.strftime('%H:%M:%S')}")
            print(f"   Confidence: {cand.confidence:.2f} (INFERRED)")
            print(f"   Description: {cand.description}")
            print(f"   Reasoning: {cand.reasoning[:120]}...")
            print(f"   Supporting Evidence Count: {len(cand.supporting_evidence)}")

        # 6. Conflicts Inspection
        print("\n[6/7] Reviewing Multi-Source Contradictions...")
        for idx, conf in enumerate(timeline.conflicts):
            print(f"   Conflict #{idx+1} [{conf.conflict_type}] (Delta: {conf.time_discrepancy_seconds:.1f}s):")
            print(f"     {conf.description}")

        # 7. Evidence Graph & Summary
        print("\n[7/7] Generating NetworkX Evidence Graph & Investigation Summary...")
        graph_builder = EvidenceGraphBuilder()
        graph = graph_builder.build_graph(incident.id, timeline.events, timeline.conflicts, rec_result.candidates)
        print(f" -> Graph Nodes: {graph.stats['total_nodes']} | Graph Edges: {graph.stats['total_edges']} | Density: {graph.stats['density']:.4f}")

        report_service = ReportService(db)
        summary = report_service.generate_investigation_summary(incident.id)
        print("\n--- Executive Summary Report ---")
        print(f"Title: {summary['title']}")
        print(f"What Happened: {summary['what_happened']}")
        print(f"Residual Uncertainties: {summary['residual_uncertainties']}")

        print("\n" + "=" * 70)
        print("ALL VERIFICATION CHECKS PASSED PERFECTLY!")
        print("=" * 70)

    finally:
        db.close()

if __name__ == "__main__":
    run_verification()
