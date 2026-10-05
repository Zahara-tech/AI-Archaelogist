"""
Synthetic Industrial Incident Data Generator for AI Archaeologist.
Generates realistic manufacturing datasets with sensor telemetry, PLC machine logs,
operator/maintenance notes, production ERP records, and CCTV visual evidence.
"""

import os
import json
import csv
from datetime import datetime, timedelta
import numpy as np

def generate_cnc_incident_dataset(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    base_time = datetime(2026, 3, 15, 10, 0, 0)
    machine_id = "CNC-MILL-04"
    component_id = "AERO-BLADE-TURBINE-X79-8842"
    
    # -------------------------------------------------------------
    # 1. SENSORS.CSV (1 Hz telemetry over 10 minutes: 10:00:00 - 10:10:00)
    # -------------------------------------------------------------
    sensors_file = os.path.join(output_dir, "sensors.csv")
    with open(sensors_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "machine_id", "temperature", "vibration", 
            "coolant_pressure", "spindle_rpm", "power_draw_kw", "machine_state"
        ])
        
        # 0 - 90s: Warmup / Normal idle-to-run
        # 91 - 195s: Normal Machining (Temp ~42C, Vib ~1.2mm/s, Coolant ~5.2bar, RPM ~12000)
        # 195 - 222s: Coolant pressure drops to 1.1 bar
        # 222 - 250s: Temperature rises 48C -> 84.5C
        # 250 - 255s: Vibration spikes 1.8 -> 9.8 mm/s
        # 255s (10:04:15): E-Stop / Thermal Trip
        # 256 - 310s: Machine stops, RPM drops to 0, Temp slowly dissipates
        # 311 - 380s (10:05:11 - 10:06:20): Intentional Sensor Telemetry Blackout / Gap during emergency power reset
        # 381 - 600s: Maintenance diagnostic mode (RPM 0, Temp 35C, Vib 0.1)
        
        np.random.seed(42)
        total_seconds = 600
        
        for sec in range(total_seconds):
            # Simulate telemetry gap during power reset between sec 310 and 375
            if 310 <= sec <= 375:
                continue # Missing readings gap in telemetry!
                
            current_time = base_time + timedelta(seconds=sec)
            iso_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            
            if sec < 90:
                state = "RUNNING"
                temp = 38.0 + (sec / 90.0) * 4.0 + np.random.normal(0, 0.2)
                vib = 0.8 + np.random.normal(0, 0.05)
                pressure = 5.1 + np.random.normal(0, 0.1)
                rpm = int(8000 + (sec / 90.0) * 4000)
                power = 12.5 + np.random.normal(0, 0.3)
            elif sec < 195:
                state = "RUNNING"
                temp = 42.0 + np.random.normal(0, 0.4)
                vib = 1.2 + np.random.normal(0, 0.08)
                pressure = 5.2 + np.random.normal(0, 0.1)
                rpm = 12000 + int(np.random.normal(0, 30))
                power = 24.2 + np.random.normal(0, 0.5)
            elif sec < 222:
                # Coolant failure begins
                state = "RUNNING"
                decay = (sec - 195) / 27.0
                pressure = max(1.1, 5.2 - decay * 4.1 + np.random.normal(0, 0.05))
                temp = 42.0 + decay * 6.0 + np.random.normal(0, 0.3)
                vib = 1.3 + decay * 0.5 + np.random.normal(0, 0.08)
                rpm = 12000 + int(np.random.normal(0, 40))
                power = 24.8 + np.random.normal(0, 0.6)
            elif sec < 250:
                # Severe thermal spike
                state = "RUNNING"
                thermal_progress = (sec - 222) / 28.0
                temp = 48.0 + (thermal_progress ** 1.6) * 36.5 + np.random.normal(0, 0.4)
                vib = 1.8 + (thermal_progress ** 1.8) * 4.6 + np.random.normal(0, 0.1)
                pressure = 1.1 + np.random.normal(0, 0.05)
                rpm = 12000 - int(thermal_progress * 800)
                power = 26.0 + thermal_progress * 8.0 + np.random.normal(0, 0.8)
            elif sec < 255:
                # Critical chatter & seizure phase
                state = "CRITICAL_OVERHEAT"
                temp = 84.5 + (sec - 250) * 1.5 + np.random.normal(0, 0.3)
                vib = 6.4 + (sec - 250) * 0.9 + np.random.normal(0, 0.2)
                pressure = 1.0 + np.random.normal(0, 0.05)
                rpm = 11000 - (sec - 250) * 1200
                power = 34.0 + np.random.normal(0, 1.0)
            elif sec < 310:
                # Emergency stopped
                state = "STOPPED_FAULT"
                t_after = sec - 255
                temp = max(55.0, 89.0 - t_after * 0.6)
                vib = max(0.1, 9.8 * np.exp(-t_after / 5.0))
                pressure = 0.2
                rpm = 0
                power = 1.8
            else:
                # Maintenance diagnosis mode
                state = "MAINTENANCE_HOLD"
                temp = 36.5 + np.random.normal(0, 0.2)
                vib = 0.15 + np.random.normal(0, 0.02)
                pressure = 0.0
                rpm = 0
                power = 2.1
                
            writer.writerow([
                iso_time,
                machine_id,
                round(float(temp), 2),
                round(float(vib), 3),
                round(float(pressure), 2),
                int(rpm),
                round(float(power), 2),
                state
            ])

    # -------------------------------------------------------------
    # 2. MACHINE_LOGS.CSV (PLC and Controller Events)
    # -------------------------------------------------------------
    logs_file = os.path.join(output_dir, "machine_logs.csv")
    logs = [
        ("2026-03-15 10:00:00", machine_id, "SYS_001_INIT", "Part clamp confirmed. CNC cycle initiated in AUTO mode."),
        ("2026-03-15 10:01:30", machine_id, "SPINDLE_SPINUP", "Spindle reached target speed 12000 RPM. Tool T04 engaged."),
        ("2026-03-15 10:03:15", machine_id, "WRN_COOLANT_FLOW", "Warning: Coolant flow sensor FL-102 below nominal lower limit (1.1 bar)."),
        ("2026-03-15 10:03:42", machine_id, "WRN_TEMP_HIGH", "Spindle bearing thermocouple TC-01 exceeded caution threshold 70.0 C."),
        ("2026-03-15 10:04:10", machine_id, "ALM_VIB_EXCESSIVE", "Spindle vibration accelerometer ACC-02 exceeded safety limit (6.4 mm/s)."),
        ("2026-03-15 10:04:15", machine_id, "ERR_E07_THERMAL_SEIZE", "Emergency shutdown executed by PLC watchdog: Spindle bearing thermal expansion trip."),
        ("2026-03-15 10:04:51", machine_id, "INTERLOCK_DOOR_OPEN", "Front safety enclosure interlock opened manually."),
        ("2026-03-15 10:05:02", machine_id, "HMI_ACK_FAULT", "Operator HMI acknowledged error code ERR_E07."),
        ("2026-03-15 10:06:20", machine_id, "MNT_DIAG_START", "Maintenance diagnostic key activated. PLC transitioned to MAINTENANCE_HOLD."),
        ("2026-03-15 10:08:10", machine_id, "SYS_PART_EJECT", "Part clamp released. Workpiece marked SCRAP by QA supervisor.")
    ]
    with open(logs_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "machine_id", "event_code", "description"])
        for row in logs:
            writer.writerow(row)

    # -------------------------------------------------------------
    # 3. OPERATOR_REPORT.TXT
    # -------------------------------------------------------------
    operator_file = os.path.join(output_dir, "operator_report.txt")
    operator_text = """INCIDENT REPORT - PRODUCTION SHIFT 1
Date: 2026-03-15
Author: John Miller (Operator Badge #OP-302)
Cell: Station 04 - CNC Precision Milling
Workpiece ID: AERO-BLADE-TURBINE-X79-8842

Statement:
I was operating Station 4 during the morning run. We were profiling the aero turbine blade (batch TB-8842). 
Around 10:03 AM, I noticed an unusual high-pitched whining noise from the spindle enclosure. Shortly after, the yellow beacon began blinking and coolant vapor started appearing through the viewing glass.

At approximately 10:04:15, the machine triggered an emergency stop and made a loud grinding chatter. I immediately rushed to the station, pressed the physical emergency flush button, and opened the safety enclosure door around 10:04:50 to check for fire or tool breakage. I observed smoke and severe burn discoloration on the toolholder.

Between 10:05 and 10:06, I performed a quick visual inspection of the internal cutting chamber, verified no active flame, and reset the main panel breaker while calling maintenance on the radio. The workpiece blade surface had severe thermal gouges. Technician Dave arrived around 10:06.
"""
    with open(operator_file, "w", encoding="utf-8") as f:
        f.write(operator_text)

    # -------------------------------------------------------------
    # 4. MAINTENANCE_REPORT.TXT
    # -------------------------------------------------------------
    maintenance_file = os.path.join(output_dir, "maintenance_report.txt")
    maintenance_text = """MAINTENANCE INCIDENT INVESTIGATION LOG
Incident Ref: MNT-INC-2026-088
Date: 2026-03-15
Technician: Dave Vance (Sr. Reliability Engineer, Badge #MNT-109)
Machine: CNC-MILL-04 (DMG Mori 5-Axis)

Initial Findings:
Dispatched to Station 4 at 10:05:30 following radio call from operator John Miller. Arrived at station at 10:06:15.
Connected diagnostic laptop to PLC interface at 10:06:20.

Observations:
1. Spindle ceramic bearing pack seized due to severe thermal expansion. Peak bearing temperature estimated >90°C.
2. Coolant solenoid valve SOL-22 was stuck in 20% partially-closed position due to chip contamination in return filter, starving the flood coolant nozzles during high-speed trochoidal milling.
3. Carbide end mill (T04, 12mm 5-flute) experienced micro-welding onto Inconel workpiece, causing catastrophic chatter at 10:04:10 before the thermal watchdog triggered at 10:04:15.
4. Physical inspection of enclosure confirms operator opened door at ~10:04:51 and inspected spindle chamber before my arrival.
5. Workpiece TB-8842 sustained deep gouge defects (>1.8mm depth) beyond rework limits. Workpiece condemned to scrap.
"""
    with open(maintenance_file, "w", encoding="utf-8") as f:
        f.write(maintenance_text)

    # -------------------------------------------------------------
    # 5. PRODUCTION_RECORDS.CSV (ERP / MES records)
    # -------------------------------------------------------------
    prod_file = os.path.join(output_dir, "production_records.csv")
    with open(prod_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["component_id", "station", "operator_id", "start_time", "end_time", "result", "defect_code"])
        writer.writerow([
            component_id,
            "Station 04 - CNC Milling",
            "OP-302",
            "2026-03-15 10:00:00",
            "2026-03-15 10:08:10",
            "SCRAP_REJECTED",
            "DEF_THERMAL_GOUGE_SEIZURE"
        ])

    # -------------------------------------------------------------
    # 6. CCTV_METADATA.JSON (Visual detection events from Station 4 camera)
    # -------------------------------------------------------------
    cctv_file = os.path.join(output_dir, "cctv_metadata.json")
    cctv_data = {
        "camera_id": "CAM-STATION-04-OVERHEAD",
        "fps": 30,
        "resolution": "1920x1080",
        "start_time": "2026-03-15 10:00:00",
        "duration_seconds": 600,
        "detections": [
            {
                "timestamp": "2026-03-15 10:00:05",
                "event_type": "operator_present",
                "object_class": "person",
                "region": "loading_bay",
                "confidence": 0.96,
                "description": "Operator OP-302 loading workpiece fixture."
            },
            {
                "timestamp": "2026-03-15 10:00:45",
                "event_type": "operator_exit",
                "object_class": "person",
                "region": "safety_perimeter",
                "confidence": 0.94,
                "description": "Operator exits safety enclosure and closes door."
            },
            {
                "timestamp": "2026-03-15 10:01:30",
                "event_type": "spindle_rotation_visible",
                "object_class": "machine_spindle",
                "region": "work_chamber",
                "confidence": 0.92,
                "description": "Spindle rotation detected, machining in progress."
            },
            {
                "timestamp": "2026-03-15 10:04:16",
                "event_type": "beacon_alarm_flashing",
                "object_class": "stack_light",
                "region": "machine_top",
                "confidence": 0.98,
                "description": "Stack light switched from green to flashing red/yellow alarm."
            },
            {
                "timestamp": "2026-03-15 10:04:35",
                "event_type": "operator_approaching",
                "object_class": "person",
                "region": "station_approach",
                "confidence": 0.95,
                "description": "Operator entering workstation safety perimeter."
            },
            {
                "timestamp": "2026-03-15 10:04:51",
                "event_type": "safety_door_opened",
                "object_class": "enclosure_door",
                "region": "machine_front",
                "confidence": 0.97,
                "description": "Visual confirmation: Enclosure door opened by operator."
            },
            {
                "timestamp": "2026-03-15 10:05:15",
                "event_type": "operator_inspection_action",
                "object_class": "person",
                "region": "work_chamber",
                "confidence": 0.88,
                "description": "Operator leaning into work chamber, visual check of toolholder."
            },
            {
                "timestamp": "2026-03-15 10:06:15",
                "event_type": "maintenance_technician_arrived",
                "object_class": "person",
                "region": "control_console",
                "confidence": 0.93,
                "description": "Second person (wearing high-vis maintenance vest) arrives at console."
            }
        ]
    }
    with open(cctv_file, "w", encoding="utf-8") as f:
        json.dump(cctv_data, f, indent=2)

    print(f"Synthetic CNC incident dataset generated successfully in {output_dir}")

if __name__ == "__main__":
    generate_cnc_incident_dataset("data/synthetic/cnc_milling_incident_001")
