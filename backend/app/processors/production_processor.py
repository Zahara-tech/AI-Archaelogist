"""
Production & ERP Record Processor.
Processes traveler batches, quality rejection logs, and workpiece routing records.
"""

import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.processors.base import BaseProcessor
from app.schemas.event import EventBase, EvidenceReference

class ProductionProcessor(BaseProcessor):
    def __init__(self):
        super().__init__("production_processor")

    def process(
        self, 
        file_path: str, 
        incident_id: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[EventBase]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Production record file not found: {file_path}")

        df = pd.read_csv(file_path)
        events: List[EventBase] = []
        file_name = os.path.basename(file_path)

        for idx, row in df.iterrows():
            comp_id = str(row.get("component_id", "UNKNOWN_PART"))
            station = str(row.get("station", "STATION_04"))
            result = str(row.get("result", "COMPLETED"))
            defect_code = str(row.get("defect_code", "NONE"))
            
            if "start_time" in row and pd.notna(row["start_time"]):
                t_start = pd.to_datetime(row["start_time"]).to_pydatetime()
                events.append(EventBase(
                    timestamp_start=t_start,
                    event_type="production_batch_start",
                    description=f"Workpiece {comp_id} batch processing started at {station}.",
                    source="production_record",
                    confidence=1.0,
                    certainty_type="observed",
                    observed_or_inferred="observed",
                    component_id=comp_id,
                    evidence_refs=[
                        EvidenceReference(
                            source_type="production_record",
                            file_name=file_name,
                            row_index=int(idx),
                            details={"component_id": comp_id, "station": station}
                        )
                    ]
                ))

            if "end_time" in row and pd.notna(row["end_time"]):
                t_end = pd.to_datetime(row["end_time"]).to_pydatetime()
                events.append(EventBase(
                    timestamp_start=t_end,
                    event_type="production_batch_scrapped" if "SCRAP" in result else "production_batch_finish",
                    description=f"Workpiece {comp_id} processing ended: Result = {result}, Defect Code = {defect_code}.",
                    source="production_record",
                    confidence=1.0,
                    certainty_type="observed",
                    observed_or_inferred="observed",
                    component_id=comp_id,
                    evidence_refs=[
                        EvidenceReference(
                            source_type="production_record",
                            file_name=file_name,
                            row_index=int(idx),
                            details={"result": result, "defect_code": defect_code}
                        )
                    ],
                    extra_metadata={"result": result, "defect_code": defect_code}
                ))

        return events
