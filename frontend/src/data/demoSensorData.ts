import type { SensorDataResponse } from '../types';

export const demoSensorData: SensorDataResponse = {
  incident_id: 'INC-2026-0142',
  total_points: 120,
  columns: ['timestamp', 'machine_id', 'temperature', 'vibration', 'coolant_pressure', 'spindle_rpm', 'power_draw_kw', 'machine_state'],
  records: Array.from({ length: 120 }, (_, i) => ({
    timestamp: `2026-01-15T10:${String(Math.floor(i / 60)).padStart(2, '0')}:${String(i % 60).padStart(2, '0')}:00Z`,
    machine_id: 'IM-04',
    temperature: 45 + Math.sin(i / 20) * 5 + (i > 60 ? Math.random() * 15 : 0),
    vibration: 2 + Math.sin(i / 15) * 1 + (i > 60 ? Math.random() * 3 : 0),
    coolant_pressure: 3.5 + Math.sin(i / 25) * 0.5,
    spindle_rpm: 8000 + Math.sin(i / 30) * 200,
    power_draw_kw: 12 + Math.sin(i / 20) * 2,
    machine_state: i > 65 ? 'WARNING' : 'NORMAL'
  }))
};
