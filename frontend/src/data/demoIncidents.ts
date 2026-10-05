import type { IncidentSummary, IncidentDetail, TemporalGap, ConflictItem, ReconstructedEvent } from '../types';

export const demoIncidents: IncidentSummary[] = [
  {
    id: 'INC-2026-0142',
    title: 'Ion Mill Process Anomaly',
    machine_id: 'IM-04',
    component_id: 'SPINDLE-01',
    status: 'Under Investigation',
    severity: 'high',
    created_at: '2026-01-15T10:07:43Z',
    event_count: 9,
    evidence_count: 24,
    gap_count: 1,
    conflict_count: 1
  },
  {
    id: 'INC-2026-0138',
    title: 'Unexpected Process Deviation',
    machine_id: 'IM-02',
    component_id: 'COOLANT-SYS',
    status: 'Resolved',
    severity: 'medium',
    created_at: '2026-01-12T14:32:18Z',
    event_count: 7,
    evidence_count: 31,
    gap_count: 0,
    conflict_count: 0
  },
  {
    id: 'INC-2026-0135',
    title: 'Temperature Spike Event',
    machine_id: 'IM-01',
    component_id: 'BEARING-03',
    status: 'Under Investigation',
    severity: 'high',
    created_at: '2026-01-10T09:15:22Z',
    event_count: 5,
    evidence_count: 18,
    gap_count: 2,
    conflict_count: 2
  },
  {
    id: 'INC-2026-0130',
    title: 'Vibration Pattern Change',
    machine_id: 'IM-03',
    component_id: 'MOTOR-02',
    status: 'Resolved',
    severity: 'low',
    created_at: '2026-01-08T16:45:11Z',
    event_count: 4,
    evidence_count: 12,
    gap_count: 0,
    conflict_count: 0
  }
];

export const demoIncidentDetail: IncidentDetail = {
  id: 'INC-2026-0142',
  title: 'Ion Mill Process Anomaly',
  machine_id: 'IM-04',
  component_id: 'SPINDLE-01',
  severity: 'high',
  status: 'Under Investigation',
  created_at: '2026-01-15T10:07:43Z',
  start_time: '2026-01-15T10:02:11Z',
  end_time: '2026-01-15T10:07:43Z',
  description: 'Abnormal process behavior detected during ion milling operation. Initial sensor readings indicate temperature and vibration anomalies followed by machine state change.',
  evidence_items: [
    {
      id: 'EVID-001',
      incident_id: 'INC-2026-0142',
      file_name: 'sensor_data_im04.csv',
      file_type: 'sensor',
      file_path: '/data/sensor_data_im04.csv',
      uploaded_at: '2026-01-15T10:08:00Z',
      processed: 1,
      summary: '124,832 sensor readings from IM-04'
    },
    {
      id: 'EVID-002',
      incident_id: 'INC-2026-0142',
      file_name: 'machine_log_im04.log',
      file_type: 'machine_log',
      file_path: '/data/machine_log_im04.log',
      uploaded_at: '2026-01-15T10:08:15Z',
      processed: 1,
      summary: '342 machine log events'
    },
    {
      id: 'EVID-003',
      incident_id: 'INC-2026-0142',
      file_name: 'process_records.xlsx',
      file_type: 'production_record',
      file_path: '/data/process_records.xlsx',
      uploaded_at: '2026-01-15T10:08:30Z',
      processed: 1,
      summary: '18 process records'
    },
    {
      id: 'EVID-004',
      incident_id: 'INC-2026-0142',
      file_name: 'incident_report_0142.pdf',
      file_type: 'operator_report',
      file_path: '/data/incident_report_0142.pdf',
      uploaded_at: '2026-01-15T10:09:00Z',
      processed: 1,
      summary: 'Operator incident report'
    }
  ],
  events: [
    {
      id: 'EVT-001',
      incident_id: 'INC-2026-0142',
      timestamp_start: '2026-01-15T10:02:11Z',
      event_type: 'process_start',
      description: 'PROCESS START',
      source: 'machine_log',
      confidence: 0.98,
      certainty_type: 'observed',
      observed_or_inferred: 'observed',
      machine_id: 'IM-04',
      component_id: 'SPINDLE-01',
      evidence_refs: [
        {
          source_type: 'machine_log',
          file_name: 'machine_log_im04.log',
          line_number: 1,
          timestamp_raw: '10:02:11',
          quote: 'Process started - IM-04'
        }
      ]
    },
    {
      id: 'EVT-002',
      incident_id: 'INC-2026-0142',
      timestamp_start: '2026-01-15T10:02:48Z',
      event_type: 'normal_operation',
      description: 'Operating normally',
      source: 'sensor',
      confidence: 0.95,
      certainty_type: 'observed',
      observed_or_inferred: 'observed',
      machine_id: 'IM-04',
      evidence_refs: [
        {
          source_type: 'sensor',
          file_name: 'sensor_data_im04.csv',
          row_index: 37,
          metric_name: 'temperature',
          metric_value: 45.2
        }
      ]
    },
    {
      id: 'EVT-003',
      incident_id: 'INC-2026-0142',
      timestamp_start: '2026-01-15T10:03:27Z',
      event_type: 'sensor_deviation',
      description: 'Sensor deviation detected',
      source: 'sensor',
      confidence: 0.89,
      certainty_type: 'observed',
      observed_or_inferred: 'observed',
      machine_id: 'IM-04',
      evidence_refs: [
        {
          source_type: 'sensor',
          file_name: 'sensor_data_im04.csv',
          row_index: 76,
          metric_name: 'vibration',
          metric_value: 4.2
        }
      ]
    },
    {
      id: 'EVT-004',
      incident_id: 'INC-2026-0142',
      timestamp_start: '2026-01-15T10:03:51Z',
      event_type: 'temperature_anomaly',
      description: 'Temperature anomaly',
      source: 'sensor',
      confidence: 0.92,
      certainty_type: 'observed',
      observed_or_inferred: 'observed',
      machine_id: 'IM-04',
      evidence_refs: [
        {
          source_type: 'sensor',
          file_name: 'sensor_data_im04.csv',
          row_index: 100,
          metric_name: 'temperature',
          metric_value: 68.5
        }
      ]
    },
    {
      id: 'EVT-005',
      incident_id: 'INC-2026-0142',
      timestamp_start: '2026-01-15T10:04:13Z',
      event_type: 'vibration_change',
      description: 'Vibration pattern changed',
      source: 'sensor',
      confidence: 0.87,
      certainty_type: 'observed',
      observed_or_inferred: 'observed',
      machine_id: 'IM-04',
      evidence_refs: [
        {
          source_type: 'sensor',
          file_name: 'sensor_data_im04.csv',
          row_index: 122,
          metric_name: 'vibration',
          metric_value: 6.8
        }
      ]
    },
    {
      id: 'EVT-006',
      incident_id: 'INC-2026-0142',
      timestamp_start: '2026-01-15T10:05:02Z',
      event_type: 'machine_state_change',
      description: 'Machine state changed',
      source: 'machine_log',
      confidence: 0.94,
      certainty_type: 'observed',
      observed_or_inferred: 'observed',
      machine_id: 'IM-04',
      evidence_refs: [
        {
          source_type: 'machine_log',
          file_name: 'machine_log_im04.log',
          line_number: 287,
          timestamp_raw: '10:05:02',
          quote: 'State transition: NORMAL -> WARNING'
        }
      ]
    },
    {
      id: 'EVT-007',
      incident_id: 'INC-2026-0142',
      timestamp_start: '2026-01-15T10:05:18Z',
      event_type: 'fault_condition',
      description: 'Fault condition detected',
      source: 'machine_log',
      confidence: 0.91,
      certainty_type: 'observed',
      observed_or_inferred: 'observed',
      machine_id: 'IM-04',
      evidence_refs: [
        {
          source_type: 'machine_log',
          file_name: 'machine_log_im04.log',
          line_number: 295,
          timestamp_raw: '10:05:18',
          quote: 'FAULT: Spindle bearing temperature exceeded threshold'
        }
      ]
    },
    {
      id: 'EVT-008',
      incident_id: 'INC-2026-0142',
      timestamp_start: '2026-01-15T10:06:03Z',
      event_type: 'recovery_sequence',
      description: 'Recovery sequence initiated',
      source: 'machine_log',
      confidence: 0.96,
      certainty_type: 'observed',
      observed_or_inferred: 'observed',
      machine_id: 'IM-04',
      evidence_refs: [
        {
          source_type: 'machine_log',
          file_name: 'machine_log_im04.log',
          line_number: 312,
          timestamp_raw: '10:06:03',
          quote: 'Recovery sequence started'
        }
      ]
    },
    {
      id: 'EVT-009',
      incident_id: 'INC-2026-0142',
      timestamp_start: '2026-01-15T10:07:43Z',
      event_type: 'process_complete',
      description: 'Process completed with warnings',
      source: 'machine_log',
      confidence: 0.97,
      certainty_type: 'observed',
      observed_or_inferred: 'observed',
      machine_id: 'IM-04',
      evidence_refs: [
        {
          source_type: 'machine_log',
          file_name: 'machine_log_im04.log',
          line_number: 342,
          timestamp_raw: '10:07:43',
          quote: 'Process complete - warnings logged'
        }
      ]
    }
  ]
};

export const demoTemporalGaps: TemporalGap[] = [
  {
    gap_id: 'GAP-001',
    gap_start: '2026-01-15T10:04:20Z',
    gap_end: '2026-01-15T10:05:02Z',
    duration_seconds: 42,
    previous_event_id: 'EVT-005',
    previous_event_desc: 'Vibration pattern changed',
    next_event_id: 'EVT-006',
    next_event_desc: 'Machine state changed',
    machine_id: 'IM-04',
    description: 'Evidence gap between vibration change and machine state transition',
    reconstruction_candidates_count: 1
  }
];

export const demoConflicts: ConflictItem[] = [
  {
    id: 'CONF-001',
    incident_id: 'INC-2026-0142',
    conflict_type: 'timestamp_discrepancy',
    description: 'Timestamp mismatch between sensor and machine log',
    source_a: 'sensor',
    source_b: 'machine_log',
    evidence_a: { timestamp: '10:04:18', event: 'vibration spike' },
    evidence_b: { timestamp: '10:04:23', event: 'state change logged' },
    time_discrepancy_seconds: 5,
    status: 'Requires review',
    created_at: '2026-01-15T10:10:00Z'
  }
];

export const demoReconstructedEvents: ReconstructedEvent[] = [
  {
    id: 'RECON-001',
    incident_id: 'INC-2026-0142',
    gap_start: '2026-01-15T10:04:20Z',
    gap_end: '2026-01-15T10:05:02Z',
    candidate_event_type: 'abnormal_process_transition',
    description: 'Abnormal process transition preceding machine-state change',
    confidence: 0.78,
    supporting_evidence: [
      { type: 'sensor_pattern', description: 'Sensor pattern changed before gap' },
      { type: 'temporal', description: 'Machine state changed immediately after gap' },
      { type: 'historical', description: 'Temporal pattern matches previous fault sequences' }
    ],
    conflicting_evidence: [],
    reasoning: 'The vibration pattern change at 10:04:13 was followed by a machine state change at 10:05:02. The 42-second gap suggests an intermediate abnormal transition event that was not directly logged.',
    status: 'inferred',
    created_at: '2026-01-15T10:12:00Z'
  }
];
