import React from 'react';
import { 
  BookOpen, 
  Cpu, 
  Database, 
  Layers, 
  ShieldCheck, 
  AlertTriangle, 
  Activity, 
  GitBranch, 
  Calculator,
  Compass,
  Zap,
  ArrowRight
} from 'lucide-react';
import { Link } from 'react-router-dom';

export const AboutProject: React.FC = () => {
  const sensorList = [
    { name: 'IONGAUGEPRESSURE', category: 'Vacuum / Chamber', desc: 'Chamber vacuum pressure monitoring process atmospheric stability.' },
    { name: 'ETCHBEAMVOLTAGE', category: 'Ion Beam Energy', desc: 'Acceleration voltage applied to generate directed ion etching stream.' },
    { name: 'ETCHBEAMCURRENT', category: 'Ion Beam Energy', desc: 'Total electrical beam current impacting target semiconductor wafer.' },
    { name: 'ETCHSUPPRESSORVOLTAGE', category: 'Grid Optics', desc: 'Negative potential on suppressor grid preventing electron backstreaming.' },
    { name: 'ETCHSUPPRESSORCURRENT', category: 'Grid Optics', desc: 'Suppressor grid leakage current monitoring ion optics wear.' },
    { name: 'FLOWCOOLFLOWRATE', category: 'Thermal Management', desc: 'Backside helium coolant fluid/gas flow rate maintaining platen temperature.' },
    { name: 'FLOWCOOLPRESSURE', category: 'Thermal Management', desc: 'Wafer clamp backside gas pressure preventing thermal overheating.' },
    { name: 'ETCHGASCHANNEL1READBACK', category: 'Process Gas Delivery', desc: 'Argon primary ionization process gas flow mass controller readback.' },
    { name: 'ETCHPBNGASREADBACK', category: 'Process Gas Delivery', desc: 'Secondary purge and balance gas channel readback telemetry.' },
    { name: 'FIXTURETILTANGLE', category: 'Stage Kinematics', desc: 'Physical mechanical tilt angle of the wafer chuck relative to ion beam axis.' },
    { name: 'ROTATIONSPEED', category: 'Stage Kinematics', desc: 'Platen rotational velocity in revolutions per minute ensuring uniform etch.' },
    { name: 'ACTUALROTATIONANGLE', category: 'Stage Kinematics', desc: 'Instantaneous encoder rotational angle tracking wafer orientation.' },
    { name: 'FIXTURESHUTTERPOSITION', category: 'Mechanical Actuators', desc: 'Aperture shutter state isolating wafer during beam stabilization.' },
    { name: 'ETCHSOURCEUSAGE', category: 'Consumables & Wear', desc: 'Accumulated operational runtime hours on the plasma ion generation source.' },
    { name: 'ETCHAUXSOURCETIMER', category: 'Consumables & Wear', desc: 'Auxiliary filament operating lifetime tracking maintenance degradation.' },
    { name: 'ETCHAUX2SOURCETIMER', category: 'Consumables & Wear', desc: 'Secondary discharge chamber filament timer tracking wear status.' },
    { name: 'ACTUALSTEPDURATION', category: 'Recipe Timing', desc: 'Elapsed execution time within the active semiconductor etching recipe step.' }
  ];

  return (
    <div className="main-content">
      {/* Page Header */}
      <div style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem' }}>
          <span className="badge badge-cyan">Documentation & Architecture</span>
          <span className="badge badge-emerald">PHM 2018 Benchmark</span>
        </div>
        <h1 style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>
          About the AI Archaeologist Project
        </h1>
        <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', margin: '0.35rem 0 0 0' }}>
          Technical background, problem formulation, telemetry channel catalog, reconstruction algorithms, and operational scope.
        </p>
      </div>

      {/* The Industrial Problem Section */}
      <div className="glass-panel" style={{ marginBottom: '1.75rem' }}>
        <div className="glass-panel-header">
          <div className="panel-title">
            <Compass size={18} color="var(--accent-cyan)" />
            The Industrial Missing-Telemetry Problem
          </div>
          <span className="badge badge-slate">Problem Formulation</span>
        </div>

        <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: 1.7 }}>
          <p style={{ marginBottom: '1rem' }}>
            Modern industrial semiconductor manufacturing depends on high-precision ion beam etching tools. 
            These multi-million dollar machines produce continuous multichannel telemetry streams sampled at 1 Hz. 
            In operational factories, telemetry sequences frequently suffer from lost packets, transmission buffer dropouts, 
            brief sensor recalibration blinds, or logging system failures.
          </p>
          <p style={{ marginBottom: '1rem' }}>
            When sensor telemetry is missing, engineers cannot retrospectively audit process drift, verify recipe compliance, 
            or maintain reliable equipment health records. The <strong>AI Archaeologist</strong> project investigates 
            computational methods to "excavate" and reconstruct missing telemetry windows from surrounding observed history.
          </p>
          <div style={{
            padding: '0.85rem 1.15rem',
            borderRadius: '6px',
            background: 'rgba(6, 182, 212, 0.08)',
            border: '1px solid rgba(6, 182, 212, 0.25)',
            color: 'var(--text-primary)',
            fontSize: '0.85rem'
          }}>
            <strong style={{ color: 'var(--accent-cyan)' }}>Task Scope:</strong> This is an <em>offline reconstruction</em> problem (bidirectional smoothing), not online causal forecasting. The system utilizes both telemetry recorded before the gap (left context) and telemetry recorded after the gap (right context) to reconstruct the missing intermediate trajectory.
          </div>
        </div>
      </div>

      {/* The 17 Sensor Features Catalog */}
      <div className="glass-panel" style={{ marginBottom: '1.75rem' }}>
        <div className="glass-panel-header">
          <div className="panel-title">
            <Database size={18} color="var(--accent-cyan)" />
            PHM 2018 Dataset: 17 Industrial Sensor Channels
          </div>
          <span className="badge badge-emerald">Ion Mill Etch Tool</span>
        </div>

        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
          The benchmark dataset originates from the IEEE PHM 2018 Prognostics and Health Management competition. 
          Each time step comprises 17 synchronized physical parameters recorded across etching runs:
        </p>

        <div className="table-container">
          <table className="industrial-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Sensor Channel ID</th>
                <th>Subsystem Category</th>
                <th>Physical Industrial Function</th>
              </tr>
            </thead>
            <tbody>
              {sensorList.map((s, idx) => (
                <tr key={s.name}>
                  <td className="mono-cell" style={{ color: 'var(--text-muted)' }}>
                    {String(idx + 1).padStart(2, '0')}
                  </td>
                  <td style={{ fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                    {s.name}
                  </td>
                  <td>
                    <span className="badge badge-slate">{s.category}</span>
                  </td>
                  <td style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                    {s.desc}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* The Four Reconstruction Approaches */}
      <div className="glass-panel" style={{ marginBottom: '1.75rem' }}>
        <div className="glass-panel-header">
          <div className="panel-title">
            <Layers size={18} color="var(--accent-cyan)" />
            The Four Reconstruction Approaches
          </div>
          <span className="badge badge-purple">Algorithmic Formulation</span>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
          gap: '1.25rem'
        }}>
          {/* Approach 1 */}
          <div style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '8px',
            padding: '1.25rem'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.65rem' }}>
              <span style={{ fontSize: '0.95rem', fontWeight: 800, color: 'var(--accent-blue)' }}>
                1. Linear Interpolation
              </span>
              <span className="badge badge-blue">Baseline</span>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '0.75rem' }}>
              A deterministic mathematical baseline that estimates missing values by linearly extrapolating between the last observed point before the gap and the first observed point after the gap.
            </p>
            <div className="mono-cell" style={{
              background: 'var(--bg-card)',
              padding: '0.65rem',
              borderRadius: '6px',
              border: '1px solid var(--border-subtle)',
              fontSize: '0.75rem',
              color: 'var(--text-primary)'
            }}>
              ŷ[t] = y_before[-1] + (t / (M + 1)) × (y_after[0] - y_before[-1])
            </div>
          </div>

          {/* Approach 2 */}
          <div style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '8px',
            padding: '1.25rem'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.65rem' }}>
              <span style={{ fontSize: '0.95rem', fontWeight: 800, color: 'var(--accent-purple)' }}>
                2. Attention Seq2Seq
              </span>
              <span className="badge badge-purple">Neural</span>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '0.75rem' }}>
              A deep sequence-to-sequence neural network comprising a bidirectional LSTM encoder, Bahdanau additive attention, and an autoregressive LSTM decoder (~331K parameters).
            </p>
            <div className="mono-cell" style={{
              background: 'var(--bg-card)',
              padding: '0.65rem',
              borderRadius: '6px',
              border: '1px solid var(--border-subtle)',
              fontSize: '0.75rem',
              color: 'var(--text-primary)'
            }}>
              Decoder: h_t = LSTM(h_t-1, [c_t; ŷ_t-1]), where c_t = Attention(Encoder)
            </div>
          </div>

          {/* Approach 3 */}
          <div style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '8px',
            padding: '1.25rem'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.65rem' }}>
              <span style={{ fontSize: '0.95rem', fontWeight: 800, color: 'var(--accent-emerald)' }}>
                3. Hybrid Reconstruction
              </span>
              <span className="badge badge-emerald">Adaptive Blend</span>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '0.75rem' }}>
              Combines the mathematical baseline with Attention Seq2Seq using a convex combination weight ($\alpha = 0.01$) tuned strictly on validation machines without test ground-truth leakage.
            </p>
            <div className="mono-cell" style={{
              background: 'var(--bg-card)',
              padding: '0.65rem',
              borderRadius: '6px',
              border: '1px solid var(--border-subtle)',
              fontSize: '0.75rem',
              color: 'var(--text-primary)'
            }}>
              ŷ_hybrid = α × ŷ_seq2seq + (1 - α) × ŷ_linear (α = 0.01)
            </div>
          </div>

          {/* Approach 4 */}
          <div style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '8px',
            padding: '1.25rem'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.65rem' }}>
              <span style={{ fontSize: '0.95rem', fontWeight: 800, color: 'var(--accent-amber)' }}>
                4. Residual LSTM
              </span>
              <span className="badge badge-amber">Preferred AI</span>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '0.75rem' }}>
              Anchors directly to the linear interpolation baseline and trains an LSTM solely to predict the residual error ($\Delta$). This enforces physical continuity while recovering non-linear curvature.
            </p>
            <div className="mono-cell" style={{
              background: 'var(--bg-card)',
              padding: '0.65rem',
              borderRadius: '6px',
              border: '1px solid var(--border-subtle)',
              fontSize: '0.75rem',
              color: 'var(--text-primary)'
            }}>
              ŷ_final = ŷ_linear + LSTM_residual(context_before, context_after, ŷ_linear)
            </div>
          </div>
        </div>
      </div>

      {/* Evaluation Protocol & Metrics Definition */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))',
        gap: '1.5rem',
        marginBottom: '1.75rem'
      }}>
        {/* Metrics Definitions */}
        <div className="glass-panel">
          <div className="glass-panel-header">
            <div className="panel-title">
              <Calculator size={16} color="var(--accent-cyan)" />
              Evaluation Metric Definitions
            </div>
            <span className="badge badge-cyan">Standardized</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', fontSize: '0.85rem' }}>
            <div>
              <strong style={{ color: 'var(--text-primary)' }}>Mean Squared Error (MSE):</strong>
              <div className="mono-cell" style={{ color: 'var(--accent-cyan)', fontSize: '0.78rem', margin: '0.2rem 0' }}>
                MSE = (1 / (N × 17)) Σ (y_true - y_pred)²
              </div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', margin: 0 }}>
                Penalizes large outlier errors quadratically. The primary objective function during model training.
              </p>
            </div>

            <div>
              <strong style={{ color: 'var(--text-primary)' }}>Mean Absolute Error (MAE):</strong>
              <div className="mono-cell" style={{ color: 'var(--accent-emerald)', fontSize: '0.78rem', margin: '0.2rem 0' }}>
                MAE = (1 / (N × 17)) Σ |y_true - y_pred|
              </div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', margin: 0 }}>
                Reflects the average magnitude of reconstruction error across all sensor timesteps in original or z-score space.
              </p>
            </div>

            <div>
              <strong style={{ color: 'var(--text-primary)' }}>Root Mean Squared Error (RMSE):</strong>
              <div className="mono-cell" style={{ color: 'var(--accent-amber)', fontSize: '0.78rem', margin: '0.2rem 0' }}>
                RMSE = √(MSE)
              </div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', margin: 0 }}>
                Provides an error spread metric in the same dimensional units as the normalized features.
              </p>
            </div>
          </div>
        </div>

        {/* Data Separation Protocol */}
        <div className="glass-panel">
          <div className="glass-panel-header">
            <div className="panel-title">
              <ShieldCheck size={16} color="var(--accent-emerald)" />
              Data Hygiene & Zero-Leakage Protocol
            </div>
            <span className="badge badge-emerald">Audited</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.85rem' }}>
            <p style={{ color: 'var(--text-secondary)', lineHeight: 1.6, margin: 0 }}>
              To ensure rigorous scientific validity, the evaluation adheres to strict industrial machine isolation:
            </p>

            <ul style={{ listStyleType: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
              <li style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                <span style={{ color: 'var(--accent-emerald)', fontWeight: 700 }}>•</span>
                <span><strong>Training Machines:</strong> 05_M02_DC and 07_M02_DC (exclusively used to fit normalizer and train neural weights).</span>
              </li>
              <li style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                <span style={{ color: 'var(--accent-emerald)', fontWeight: 700 }}>•</span>
                <span><strong>Validation Splits:</strong> Held-out runs from training tools used solely for early stopping and alpha tuning.</span>
              </li>
              <li style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                <span style={{ color: 'var(--accent-emerald)', fontWeight: 700 }}>•</span>
                <span><strong>Held-Out Test Machines:</strong> 01_M02_DC, 02_M02_DC, 03_M01_DC, 04_M01_DC, 06_M01_DC (strictly never exposed during training).</span>
              </li>
            </ul>

            <div style={{
              padding: '0.65rem 0.85rem',
              borderRadius: '6px',
              background: 'rgba(255, 255, 255, 0.02)',
              border: '1px solid var(--border-subtle)',
              fontSize: '0.78rem',
              color: 'var(--text-secondary)'
            }}>
              <strong>No Feature Leakage:</strong> StandardScalers are fitted strictly on training data; test features are transformed using frozen training means and standard deviations.
            </div>
          </div>
        </div>
      </div>

      {/* Operational Scope & Limitations */}
      <div className="glass-panel" style={{ marginBottom: '2rem' }}>
        <div className="glass-panel-header">
          <div className="panel-title">
            <AlertTriangle size={18} color="var(--accent-amber)" />
            Operational Boundaries & Limitations
          </div>
          <span className="badge badge-amber">Production Disclaimer</span>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '1rem',
          fontSize: '0.85rem',
          color: 'var(--text-secondary)'
        }}>
          <div>
            <h4 style={{ color: 'var(--text-primary)', fontSize: '0.9rem', marginBottom: '0.35rem' }}>
              Offline vs Real-Time Forecasting
            </h4>
            <p style={{ margin: 0, lineHeight: 1.5 }}>
              This platform performs retrospective telemetry infilling. It requires observed context before and after the dropout. It is not an autoregressive forecaster predicting future unobserved machine states.
            </p>
          </div>

          <div>
            <h4 style={{ color: 'var(--text-primary)', fontSize: '0.9rem', marginBottom: '0.35rem' }}>
              No Unsubstantiated Claims
            </h4>
            <p style={{ margin: 0, lineHeight: 1.5 }}>
              We do not claim automated fault diagnosis, wafer defect prevention, or predictive maintenance. The project evaluates purely telemetry signal reconstruction accuracy on ion etching telemetry.
            </p>
          </div>

          <div>
            <h4 style={{ color: 'var(--text-primary)', fontSize: '0.9rem', marginBottom: '0.35rem' }}>
              Computational Cost Trade-off
            </h4>
            <p style={{ margin: 0, lineHeight: 1.5 }}>
              While Residual LSTM achieves a 1.44% to 2.68% error reduction on gaps of 20 to 50 steps, it introduces ~3.6 ms of GPU/CPU latency per window versus near-zero latency for linear interpolation.
            </p>
          </div>
        </div>
      </div>

      {/* Navigation CTA */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem' }}>
        <Link to="/reconstruction-studio" className="btn btn-primary" style={{ padding: '0.75rem 1.75rem' }}>
          <Cpu size={18} />
          Launch Reconstruction Studio
        </Link>
        <Link to="/model-performance" className="btn btn-secondary" style={{ padding: '0.75rem 1.75rem' }}>
          <Activity size={18} />
          Inspect Benchmark Tables
        </Link>
      </div>
    </div>
  );
};
