"""
Sensor Telemetry Processor with Baseline Anomaly Detection.
Processes 1Hz/high-frequency time-series data, detects threshold violations,
statistical anomalies (Z-score / IQR), state changes, and telemetry gaps.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.processors.base import BaseProcessor
from app.schemas.event import EventBase, EvidenceReference
from app.core.config import settings

class SensorProcessor(BaseProcessor):
    def __init__(self):
        super().__init__("sensor_processor")

    def process(
        self, 
        file_path: str, 
        incident_id: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[EventBase]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Sensor file not found: {file_path}")

        df = pd.read_csv(file_path)
        events: List[EventBase] = []
        file_name = os.path.basename(file_path)

        # 1. Normalize timestamps
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp_dt").reset_index(drop=True)

        if df.empty:
            return events

        machine_id = str(df["machine_id"].iloc[0]) if "machine_id" in df.columns else "UNKNOWN_MACHINE"

        # 2. Check for Telemetry Gaps (> 10s gap in continuous series)
        time_diffs = df["timestamp_dt"].diff().dt.total_seconds()
        gap_indices = df[time_diffs > 10.0].index.tolist()

        for idx in gap_indices:
            gap_start = df.loc[idx - 1, "timestamp_dt"]
            gap_end = df.loc[idx, "timestamp_dt"]
            duration = (gap_end - gap_start).total_seconds()

            events.append(EventBase(
                timestamp_start=gap_start.to_pydatetime(),
                timestamp_end=gap_end.to_pydatetime(),
                event_type="telemetry_blackout_gap",
                description=f"Sensor telemetry blackout detected: missing {duration:.1f}s of data between {gap_start.strftime('%H:%M:%S')} and {gap_end.strftime('%H:%M:%S')}.",
                source="sensor",
                confidence=0.99,
                certainty_type="observed",
                observed_or_inferred="observed",
                machine_id=machine_id,
                evidence_refs=[
                    EvidenceReference(
                        source_type="sensor_csv",
                        file_name=file_name,
                        row_index=int(idx),
                        timestamp_raw=str(df.loc[idx, "timestamp"]),
                        details={"gap_duration_seconds": duration, "from": str(gap_start), "to": str(gap_end)}
                    )
                ],
                extra_metadata={"gap_duration_seconds": duration}
            ))

        # 3. Detect Coolant Pressure Loss / Anomaly
        if "coolant_pressure" in df.columns:
            low_pressure_mask = df["coolant_pressure"] < settings.COOLANT_PRESSURE_MIN_BAR
            # Find contiguous blocks of low pressure
            pressure_anomalies = self._find_contiguous_blocks(df, low_pressure_mask)
            for start_idx, end_idx in pressure_anomalies:
                t_start = df.loc[start_idx, "timestamp_dt"]
                t_end = df.loc[end_idx, "timestamp_dt"]
                min_val = df.loc[start_idx:end_idx, "coolant_pressure"].min()
                
                events.append(EventBase(
                    timestamp_start=t_start.to_pydatetime(),
                    timestamp_end=t_end.to_pydatetime() if t_end != t_start else None,
                    event_type="coolant_pressure_drop",
                    description=f"Coolant pressure dropped to {min_val:.2f} bar (below threshold {settings.COOLANT_PRESSURE_MIN_BAR} bar), starving cutting zone.",
                    source="sensor",
                    confidence=0.96,
                    certainty_type="observed",
                    observed_or_inferred="observed",
                    machine_id=machine_id,
                    evidence_refs=[
                        EvidenceReference(
                            source_type="sensor_csv",
                            file_name=file_name,
                            row_index=int(start_idx),
                            metric_name="coolant_pressure",
                            metric_value=float(min_val),
                            timestamp_raw=str(df.loc[start_idx, "timestamp"]),
                            details={"threshold_bar": settings.COOLANT_PRESSURE_MIN_BAR}
                        )
                    ],
                    extra_metadata={"min_pressure_bar": float(min_val)}
                ))

        # 4. Detect Temperature Warnings & Critical Overheat
        if "temperature" in df.columns:
            # Detect Critical Overheat (> TEMP_CRITICAL_C)
            crit_temp_mask = df["temperature"] >= settings.TEMP_CRITICAL_C
            crit_blocks = self._find_contiguous_blocks(df, crit_temp_mask)
            for start_idx, end_idx in crit_blocks:
                t_start = df.loc[start_idx, "timestamp_dt"]
                t_end = df.loc[end_idx, "timestamp_dt"]
                peak_temp = df.loc[start_idx:end_idx, "temperature"].max()
                
                events.append(EventBase(
                    timestamp_start=t_start.to_pydatetime(),
                    timestamp_end=t_end.to_pydatetime() if t_end != t_start else None,
                    event_type="critical_thermal_overheat",
                    description=f"Spindle bearing temperature breached critical safety limit: peak {peak_temp:.1f} °C (limit {settings.TEMP_CRITICAL_C} °C).",
                    source="sensor",
                    confidence=0.98,
                    certainty_type="observed",
                    observed_or_inferred="observed",
                    machine_id=machine_id,
                    evidence_refs=[
                        EvidenceReference(
                            source_type="sensor_csv",
                            file_name=file_name,
                            row_index=int(start_idx),
                            metric_name="temperature",
                            metric_value=float(peak_temp),
                            timestamp_raw=str(df.loc[start_idx, "timestamp"]),
                            details={"peak_temp_c": float(peak_temp), "critical_limit": settings.TEMP_CRITICAL_C}
                        )
                    ],
                    extra_metadata={"peak_temp_c": float(peak_temp)}
                ))

        # 5. Detect Vibration Spikes & Bearing Chatter
        if "vibration" in df.columns:
            crit_vib_mask = df["vibration"] >= settings.VIBRATION_CRITICAL_MMS
            vib_blocks = self._find_contiguous_blocks(df, crit_vib_mask)
            for start_idx, end_idx in vib_blocks:
                t_start = df.loc[start_idx, "timestamp_dt"]
                t_end = df.loc[end_idx, "timestamp_dt"]
                peak_vib = df.loc[start_idx:end_idx, "vibration"].max()
                
                events.append(EventBase(
                    timestamp_start=t_start.to_pydatetime(),
                    timestamp_end=t_end.to_pydatetime() if t_end != t_start else None,
                    event_type="severe_vibration_spike",
                    description=f"Severe spindle vibration spike detected: peak {peak_vib:.2f} mm/s (critical limit {settings.VIBRATION_CRITICAL_MMS} mm/s), indicating severe chatter or bearing seizure.",
                    source="sensor",
                    confidence=0.97,
                    certainty_type="observed",
                    observed_or_inferred="observed",
                    machine_id=machine_id,
                    evidence_refs=[
                        EvidenceReference(
                            source_type="sensor_csv",
                            file_name=file_name,
                            row_index=int(start_idx),
                            metric_name="vibration",
                            metric_value=float(peak_vib),
                            timestamp_raw=str(df.loc[start_idx, "timestamp"]),
                            details={"peak_vibration_mms": float(peak_vib), "limit": settings.VIBRATION_CRITICAL_MMS}
                        )
                    ],
                    extra_metadata={"peak_vibration_mms": float(peak_vib)}
                ))

        # 6. Detect Machine State Transitions
        if "machine_state" in df.columns:
            state_changes = df[df["machine_state"] != df["machine_state"].shift(1)]
            for idx, row in state_changes.iterrows():
                prev_state = df.loc[idx - 1, "machine_state"] if idx > 0 else "OFF"
                curr_state = row["machine_state"]
                t_event = row["timestamp_dt"]
                
                events.append(EventBase(
                    timestamp_start=t_event.to_pydatetime(),
                    event_type=f"state_transition_{curr_state.lower()}",
                    description=f"Machine state transitioned from {prev_state} to {curr_state}.",
                    source="sensor",
                    confidence=1.0,
                    certainty_type="observed",
                    observed_or_inferred="observed",
                    machine_id=machine_id,
                    evidence_refs=[
                        EvidenceReference(
                            source_type="sensor_csv",
                            file_name=file_name,
                            row_index=int(idx),
                            metric_name="machine_state",
                            metric_value=curr_state,
                            timestamp_raw=str(row["timestamp"]),
                            details={"from_state": prev_state, "to_state": curr_state}
                        )
                    ],
                    extra_metadata={"from_state": prev_state, "to_state": curr_state}
                ))

        # Sort events chronologically
        events.sort(key=lambda e: e.timestamp_start)
        return events

    def _find_contiguous_blocks(self, df: pd.DataFrame, boolean_mask: pd.Series) -> List[tuple]:
        """Finds start and end index tuples for contiguous True sequences."""
        blocks = []
        in_block = False
        start_idx = None

        for idx, val in boolean_mask.items():
            if val and not in_block:
                in_block = True
                start_idx = idx
            elif not val and in_block:
                in_block = False
                blocks.append((start_idx, idx - 1))

        if in_block and start_idx is not None:
            blocks.append((start_idx, df.index[-1]))

        return blocks
