"""
Unit Tests for Modality Processors and Anomaly Detection.
"""

import os
import pytest
from app.processors.sensor_processor import SensorProcessor
from app.processors.log_processor import LogProcessor
from app.processors.text_processor import TextReportProcessor
from app.processors.video_processor import VideoProcessor
from app.processors.production_processor import ProductionProcessor

SYNTHETIC_DIR = "data/synthetic/cnc_milling_incident_001"

def test_sensor_processor_anomalies():
    sensor_file = os.path.join(SYNTHETIC_DIR, "sensors.csv")
    assert os.path.exists(sensor_file), "Synthetic sensors.csv should exist"
    
    proc = SensorProcessor()
    events = proc.process(sensor_file, "INC-TEST-001")
    
    assert len(events) > 0, "Should extract events from sensor telemetry"
    
    # Verify critical anomalies detected
    event_types = [e.event_type for e in events]
    assert "critical_thermal_overheat" in event_types, "Should detect critical temperature spike"
    assert "severe_vibration_spike" in event_types, "Should detect severe vibration anomaly"
    assert "coolant_pressure_drop" in event_types, "Should detect coolant pressure loss"
    assert "telemetry_blackout_gap" in event_types, "Should detect sensor blackout gap"
    
    # Check evidence traceability
    overheat_evt = next(e for e in events if e.event_type == "critical_thermal_overheat")
    assert len(overheat_evt.evidence_refs) > 0
    assert overheat_evt.evidence_refs[0].metric_name == "temperature"
    assert overheat_evt.observed_or_inferred == "observed"

def test_log_processor():
    log_file = os.path.join(SYNTHETIC_DIR, "machine_logs.csv")
    proc = LogProcessor()
    events = proc.process(log_file, "INC-TEST-001")
    
    assert len(events) == 10, "Should extract all 10 PLC log entries"
    event_types = [e.event_type for e in events]
    assert "plc_emergency_trip" in event_types
    assert "safety_interlock_open" in event_types
    assert "operator_hmi_interaction" in event_types

def test_text_report_processor():
    op_file = os.path.join(SYNTHETIC_DIR, "operator_report.txt")
    proc = TextReportProcessor("operator_report")
    events = proc.process(op_file, "INC-TEST-001")
    
    assert len(events) > 0
    for e in events:
        assert e.certainty_type == "reported"
        assert e.source == "operator_report"
        assert len(e.evidence_refs) > 0
        assert e.evidence_refs[0].quote is not None

def test_video_processor():
    cctv_file = os.path.join(SYNTHETIC_DIR, "cctv_metadata.json")
    proc = VideoProcessor()
    events = proc.process(cctv_file, "INC-TEST-001")
    
    assert len(events) == 8
    for e in events:
        assert e.source == "cctv_video"
        assert e.location is not None

def test_production_processor():
    prod_file = os.path.join(SYNTHETIC_DIR, "production_records.csv")
    proc = ProductionProcessor()
    events = proc.process(prod_file, "INC-TEST-001")
    
    assert len(events) >= 2
    assert any("scrapped" in e.event_type for e in events)
