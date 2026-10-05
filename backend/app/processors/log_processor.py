"""
Machine / PLC Log Processor.
Normalizes PLC event codes, errors, warnings, interlocks, and state notifications
into standardized industrial event objects.
"""

import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.processors.base import BaseProcessor
from app.schemas.event import EventBase, EvidenceReference

class LogProcessor(BaseProcessor):
    def __init__(self):
        super().__init__("log_processor")

    def process(
        self, 
        file_path: str, 
        incident_id: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[EventBase]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Machine log file not found: {file_path}")

        df = pd.read_csv(file_path)
        events: List[EventBase] = []
        file_name = os.path.basename(file_path)

        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp_dt").reset_index(drop=True)

        for idx, row in df.iterrows():
            event_code = str(row.get("event_code", "LOG_EVENT"))
            description = str(row.get("description", ""))
            machine_id = str(row.get("machine_id", "UNKNOWN_MACHINE"))
            t_event = row["timestamp_dt"].to_pydatetime()

            # Categorize event type
            if event_code.startswith("ERR_"):
                event_type = "plc_emergency_trip"
                confidence = 1.0
            elif event_code.startswith("ALM_"):
                event_type = "plc_safety_alarm"
                confidence = 0.98
            elif event_code.startswith("WRN_"):
                event_type = "plc_warning"
                confidence = 0.95
            elif "INTERLOCK" in event_code or "DOOR" in event_code:
                event_type = "safety_interlock_open"
                confidence = 1.0
            elif "HMI_" in event_code:
                event_type = "operator_hmi_interaction"
                confidence = 0.98
            elif "MNT_" in event_code:
                event_type = "maintenance_diagnostic_mode"
                confidence = 1.0
            elif "SPINDLE" in event_code:
                event_type = "spindle_state_change"
                confidence = 0.95
            else:
                event_type = f"plc_{event_code.lower()}"
                confidence = 0.90

            events.append(EventBase(
                timestamp_start=t_event,
                event_type=event_type,
                description=f"[{event_code}] {description}",
                source="machine_log",
                confidence=confidence,
                certainty_type="observed",
                observed_or_inferred="observed",
                machine_id=machine_id,
                evidence_refs=[
                    EvidenceReference(
                        source_type="machine_log",
                        file_name=file_name,
                        line_number=int(idx) + 2, # Account for header line
                        timestamp_raw=str(row["timestamp"]),
                        details={"event_code": event_code, "raw_description": description}
                    )
                ],
                extra_metadata={"event_code": event_code, "log_line": int(idx) + 2}
            ))

        return events
