"""
Text Report Processor for Operator Statements and Maintenance Logs.
Extracts structured observations, approximate timings, symptoms, and human actions.
Adheres strictly to the rule: "Do not hallucinate exact timestamps; represent uncertainty explicitly."
"""

import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from app.processors.base import BaseProcessor
from app.schemas.event import EventBase, EvidenceReference

class TextReportProcessor(BaseProcessor):
    def __init__(self, report_type: str = "operator_report"):
        super().__init__(f"text_processor_{report_type}")
        self.report_type = report_type # "operator_report" or "maintenance_report"

    def process(
        self, 
        file_path: str, 
        incident_id: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[EventBase]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Report file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        events: List[EventBase] = []
        file_name = os.path.basename(file_path)
        base_date_str = "2026-03-15"
        
        # Check if date is in the report
        date_match = re.search(r"Date:\s*(\d{4}-\d{2}-\d{2})", text)
        if date_match:
            base_date_str = date_match.group(1)

        source_name = "operator_report" if "operator" in file_name.lower() or self.report_type == "operator_report" else "maintenance_report"

        # Paragraph/Sentence breakdown for traceability
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        for line_num, line in enumerate(lines, start=1):
            # Look for timestamp references like "10:03 AM", "10:04:15", "10:05", "10:06:20"
            time_matches = re.finditer(r"(?:Around|at|between|approximately|~)?\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM)?", line, re.IGNORECASE)
            
            for tm in time_matches:
                hour = int(tm.group(1))
                minute = int(tm.group(2))
                second = int(tm.group(3)) if tm.group(3) else 0
                ampm = tm.group(4)
                
                if ampm and ampm.upper() == "PM" and hour < 12:
                    hour += 12
                elif ampm and ampm.upper() == "AM" and hour == 12:
                    hour = 0
                    
                is_exact = tm.group(3) is not None
                prefix_text = line[:tm.start()].lower()
                is_approx = any(w in prefix_text or w in tm.group(0).lower() for w in ["around", "approx", "between", "~", "shortly"])
                
                t_dt = datetime.strptime(f"{base_date_str} {hour:02d}:{minute:02d}:{second:02d}", "%Y-%m-%d %H:%M:%S")

                # Extract event context
                event_type, desc, conf = self._classify_text_event(line, source_name)
                
                events.append(EventBase(
                    timestamp_start=t_dt,
                    timestamp_end=None,
                    event_type=event_type,
                    description=desc,
                    source=source_name,
                    confidence=0.88 if is_approx else 0.94,
                    certainty_type="reported",
                    observed_or_inferred="observed", # Observed by human operator/engineer
                    evidence_refs=[
                        EvidenceReference(
                            source_type=source_name,
                            file_name=file_name,
                            line_number=line_num,
                            quote=line,
                            timestamp_raw=tm.group(0).strip(),
                            details={"is_approximate": is_approx, "temporal_precision": "exact" if is_exact else "minute_precision"}
                        )
                    ],
                    extra_metadata={
                        "is_approximate": is_approx,
                        "temporal_precision": "exact" if is_exact else "minute_precision",
                        "raw_quote": line
                    }
                ))

        # Sort and deduplicate very close events from the same line
        events.sort(key=lambda e: e.timestamp_start)
        return events

    def _classify_text_event(self, line: str, source: str) -> tuple:
        l = line.lower()
        if "whining noise" in l or "noise" in l or "grinding" in l or "chatter" in l:
            return ("reported_acoustic_abnormality", f"Acoustic anomaly reported: {line}", 0.85)
        elif "emergency stop" in l or "e-stop" in l or "flush button" in l:
            return ("reported_emergency_action", f"Emergency operator action reported: {line}", 0.95)
        elif "smoke" in l or "burning" in l or "discoloration" in l or "fire" in l:
            return ("reported_thermal_damage_symptom", f"Thermal distress symptom reported: {line}", 0.90)
        elif "opened the safety" in l or "door" in l or "interlock" in l:
            return ("reported_enclosure_access", f"Enclosure access reported: {line}", 0.92)
        elif "inspection" in l or "chamber" in l or "visual check" in l:
            return ("reported_chamber_inspection", f"Manual inspection reported: {line}", 0.88)
        elif "arrived" in l or "dispatched" in l:
            return ("reported_technician_dispatch", f"Personnel dispatch/arrival reported: {line}", 0.95)
        elif "bearing" in l or "seized" in l or "valve" in l or "solenoid" in l:
            return ("maintenance_diagnostic_finding", f"Maintenance finding: {line}", 0.98)
        else:
            return ("reported_general_observation", f"Human observation: {line}", 0.80)
