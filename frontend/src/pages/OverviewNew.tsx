import React from 'react';
import { Activity, Brain, TrendingUp, AlertCircle, CheckCircle, BookOpen, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';

export const OverviewNew: React.FC = () => {
  return (
    <div className="main-content">
      {/* Header */}
      <div style={{ marginBottom: '2.5rem' }}>
        <h1 style={{
          fontSize: '2.25rem',
          fontWeight: 800,
          color: 'var(--text-primary)',
          marginBottom: '0.5rem',
          letterSpacing: '-0.5px'
        }}>
          AI Archaeologist
        </h1>
        <p style={{
          fontSize: '1.1rem',
          color: 'var(--text-secondary)',
          fontWeight: 400
        }}>
          AI-assisted reconstruction of missing industrial sensor telemetry
        </p>
      </div>

      {/* Problem Statement */}
      <div className="glass-panel" style={{ marginBottom: '1.5rem' }}>
        <div className="glass-panel-header">
          <h2 style={{
            fontSize: '1.1rem',
            fontWeight: 700,
            color: 'var(--text-primary)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <AlertCircle size={20} color="#38bdf8" />
            The Problem
          </h2>
        </div>
        <div style={{ color: 'var(--text-secondary)', lineHeight: '1.7' }}>
          <p style={{ marginBottom: '1rem' }}>
            Industrial equipment generates continuous sensor data streams. Due to sensor malfunctions, 
            communication errors, or maintenance activities, some observations may be missing. 
            This project investigates methods to reconstruct those missing values.
          </p>
          <p>
            <strong style={{ color: 'var(--text-primary)' }}>Dataset:</strong> PHM 2018 Ion Mill Etch Tool telemetry with 
            <span style={{ color: '#38bdf8', fontWeight: 600 }}> 17 sensor features</span> including pressure, voltage, 
            current, flow rates, and rotation parameters from semiconductor manufacturing equipment.
          </p>
        </div>
      </div>

      {/* Workflow Visualization */}
      <div className="glass-panel" style={{ marginBottom: '1.5rem' }}>
        <div className="glass-panel-header">
          <h2 style={{
            fontSize: '1.1rem',
            fontWeight: 700,
            color: 'var(--text-primary)'
          }}>
            Reconstruction Workflow
          </h2>
        </div>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          flexWrap: 'wrap',
          padding: '1rem 0'
        }}>
          {[
            { label: 'Observed History', color: '#10b981' },
            { label: 'Missing Gap', color: '#f59e0b' },
            { label: 'Reconstruction Method', color: '#a855f7' },
            { label: 'Reconstructed Signal', color: '#06b6d4' },
            { label: 'Error Evaluation', color: '#3b82f6' }
          ].map((step, i) => (
            <React.Fragment key={step.label}>
              <div style={{
                padding: '0.5rem 1rem',
                background: `rgba(${parseInt(step.color.slice(1,3), 16)}, ${parseInt(step.color.slice(3,5), 16)}, ${parseInt(step.color.slice(5,7), 16)}, 0.12)`,
                border: `1px solid ${step.color}40`,
                borderRadius: '8px',
                color: step.color,
                fontSize: '0.85rem',
                fontWeight: 600,
                textAlign: 'center'
              }}>
                {step.label}
              </div>
              {i < 4 && <ArrowRight size={16} color="var(--border-light)" />}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Four Approaches */}
      <div className="glass-panel" style={{ marginBottom: '1.5rem' }}>
        <div className="glass-panel-header">
          <h2 style={{
            fontSize: '1.1rem',
            fontWeight: 700,
            color: 'var(--text-primary)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <Brain size={20} color="#a855f7" />
            Four Reconstruction Approaches
          </h2>
        </div>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '1rem'
        }}>
          <div style={{
            padding: '1rem',
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '8px'
          }}>
            <h3 style={{
              fontSize: '0.95rem',
              fontWeight: 700,
              color: 'var(--text-primary)',
              marginBottom: '0.5rem'
            }}>
              1. Linear Interpolation
            </h3>
            <p style={{
              fontSize: '0.85rem',
              color: 'var(--text-secondary)',
              lineHeight: '1.5'
            }}>
              Mathematical baseline that estimates missing values using linear interpolation between observed boundary values.
              No training required, works for any gap length.
            </p>
          </div>
          <div style={{
            padding: '1rem',
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '8px'
          }}>
            <h3 style={{
              fontSize: '0.95rem',
              fontWeight: 700,
              color: 'var(--text-primary)',
              marginBottom: '0.5rem'
            }}>
              2. Attention Seq2Seq
            </h3>
            <p style={{
              fontSize: '0.85rem',
              color: 'var(--text-secondary)',
              lineHeight: '1.5'
            }}>
              Deep learning model with bidirectional LSTM encoder, additive attention, and LSTM decoder.
              Trained for 50 epochs, only supports gap length 10.
            </p>
          </div>
          <div style={{
            padding: '1rem',
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '8px'
          }}>
            <h3 style={{
              fontSize: '0.95rem',
              fontWeight: 700,
              color: 'var(--text-primary)',
              marginBottom: '0.5rem'
            }}>
              3. Hybrid Reconstruction
            </h3>
            <p style={{
              fontSize: '0.85rem',
              color: 'var(--text-secondary)',
              lineHeight: '1.5'
            }}>
              Combines linear interpolation and Attention Seq2Seq using a fixed-weight blend 
              (10% neural, 90% interpolation). Only supports gap length 10.
            </p>
          </div>
          <div style={{
            padding: '1rem',
            background: 'rgba(249, 115, 22, 0.08)',
            border: '1px solid rgba(249, 115, 22, 0.3)',
            borderRadius: '8px'
          }}>
            <h3 style={{
              fontSize: '0.95rem',
              fontWeight: 700,
              color: 'var(--text-primary)',
              marginBottom: '0.5rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}>
              4. Residual LSTM
              <span style={{
                fontSize: '0.7rem',
                background: 'rgba(249, 115, 22, 0.2)',
                color: '#f97316',
                padding: '0.15rem 0.5rem',
                borderRadius: '4px',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                Preferred AI
              </span>
            </h3>
            <p style={{
              fontSize: '0.85rem',
              color: 'var(--text-secondary)',
              lineHeight: '1.5'
            }}>
              Learns corrections to linear interpolation using a trained LSTM-based model.
              Supports gaps 10, 20, and 50.
            </p>
          </div>
        </div>
      </div>

      {/* Key Finding */}
      <div className="glass-panel" style={{
        marginBottom: '1.5rem',
        background: 'rgba(56, 189, 248, 0.08)',
        border: '1px solid ' + 'rgba(56, 189, 248, 0.3)'
      }}>
        <div className="glass-panel-header" style={{
          borderBottomColor: 'rgba(56, 189, 248, 0.3)'
        }}>
          <h2 style={{
            fontSize: '1.1rem',
            fontWeight: 700,
            color: 'var(--text-primary)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <TrendingUp size={20} color="#38bdf8" />
            What did our experiments reveal?
          </h2>
        </div>
        <div style={{ color: 'var(--text-secondary)', lineHeight: '1.7' }}>
          <p style={{
            fontSize: '1rem',
            color: '#38bdf8',
            fontWeight: 500,
            marginBottom: '0.5rem'
          }}>
            More complex AI models do not automatically outperform mathematical methods. 
            In our reported gap-length experiments, Residual LSTM achieved modestly lower MSE for gaps of 20 and 50 observations, 
            while linear interpolation performed better for a gap of 10. This motivates further development of Residual LSTM for longer gaps.
          </p>
        </div>
      </div>

      {/* Gap-Length Results */}
      <div className="glass-panel" style={{ marginBottom: '1.5rem' }}>
        <div className="glass-panel-header">
          <h2 style={{
            fontSize: '1.1rem',
            fontWeight: 700,
            color: 'var(--text-primary)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <Activity size={20} color="#38bdf8" />
            Gap-Length Experiment Results
          </h2>
        </div>
        <div style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
          <p>Normalized test MSE from actual evaluation results (verified from evaluation_results.json):</p>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{
            width: '100%',
            fontSize: '0.85rem',
            borderCollapse: 'collapse'
          }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                <th style={{
                  padding: '0.75rem',
                  textAlign: 'left',
                  color: 'var(--text-muted)',
                  fontWeight: 600,
                  fontSize: '0.75rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em'
                }}>
                  Gap Length
                </th>
                <th style={{
                  padding: '0.75rem',
                  textAlign: 'right',
                  color: 'var(--text-muted)',
                  fontWeight: 600,
                  fontSize: '0.75rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em'
                }}>
                  Linear Interpolation MSE
                </th>
                <th style={{
                  padding: '0.75rem',
                  textAlign: 'right',
                  color: 'var(--text-muted)',
                  fontWeight: 600,
                  fontSize: '0.75rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em'
                }}>
                  Residual LSTM MSE
                </th>
                <th style={{
                  padding: '0.75rem',
                  textAlign: 'center',
                  color: 'var(--text-muted)',
                  fontWeight: 600,
                  fontSize: '0.75rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em'
                }}>
                  Better
                </th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                <td style={{ padding: '0.75rem', color: 'var(--text-primary)', fontWeight: 500 }}>10</td>
                <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>0.007700</td>
                <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>0.007900</td>
                <td style={{ padding: '0.75rem', textAlign: 'center' }}>
                  <span style={{ color: '#10b981', fontWeight: 600 }}>Linear</span>
                </td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                <td style={{ padding: '0.75rem', color: 'var(--text-primary)', fontWeight: 500 }}>20</td>
                <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>0.007654</td>
                <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>0.007544</td>
                <td style={{ padding: '0.75rem', textAlign: 'center' }}>
                  <span style={{ color: '#f97316', fontWeight: 600 }}>Residual LSTM</span>
                </td>
              </tr>
              <tr>
                <td style={{ padding: '0.75rem', color: 'var(--text-primary)', fontWeight: 500 }}>50</td>
                <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>0.024311</td>
                <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>0.023659</td>
                <td style={{ padding: '0.75rem', textAlign: 'center' }}>
                  <span style={{ color: '#f97316', fontWeight: 600 }}>Residual LSTM</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div style={{
          marginTop: '1rem',
          padding: '0.75rem',
          background: 'rgba(245, 158, 11, 0.08)',
          border: '1px solid rgba(245, 158, 11, 0.3)',
          borderRadius: '6px'
        }}>
          <p style={{
            fontSize: '0.8rem',
            color: '#fbbf24',
            margin: 0
          }}>
            <strong>Note:</strong> These results are verified from evaluation_results.json. 
            Residual LSTM shows modest improvements for gaps 20 and 50 (~1.4% and ~2.7% MSE reduction respectively), 
            but performed worse for gap 10 (~2.6% higher MSE).
          </p>
        </div>
      </div>

      {/* Why Residual LSTM */}
      <div className="glass-panel" style={{
        marginBottom: '1.5rem',
        background: 'rgba(249, 115, 22, 0.08)',
        border: '1px solid ' + 'rgba(249, 115, 22, 0.3)'
      }}>
        <div className="glass-panel-header" style={{
          borderBottomColor: 'rgba(249, 115, 22, 0.3)'
        }}>
          <h2 style={{
            fontSize: '1.1rem',
            fontWeight: 700,
            color: 'var(--text-primary)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <CheckCircle size={20} color="#f97316" />
            Why Residual LSTM?
          </h2>
        </div>
        <div style={{ color: 'var(--text-secondary)', lineHeight: '1.7' }}>
          <p style={{ marginBottom: '0.75rem' }}>
            <strong style={{ color: 'var(--text-primary)' }}>Architecture:</strong> Residual LSTM starts with a linear interpolation estimate and learns a correction to improve the reconstructed readings.
          </p>
          <code style={{
            display: 'block',
            background: 'var(--bg-primary)',
            padding: '0.75rem',
            borderRadius: '6px',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.8rem',
            color: '#38bdf8',
            marginBottom: '0.75rem'
          }}>
            Final prediction = interpolation estimate + learned correction
          </code>
          <ul style={{ paddingLeft: '1.25rem', margin: 0 }}>
            <li style={{ marginBottom: '0.5rem' }}>Uses interpolation as a strong starting point</li>
            <li style={{ marginBottom: '0.5rem' }}>Learns corrections from training examples</li>
            <li style={{ marginBottom: '0.5rem' }}>In the reported experiment, it slightly reduced MSE for gaps of 20 and 50 readings</li>
            <li style={{ marginBottom: '0.5rem' }}>Did not outperform interpolation for the 10-reading gap</li>
            <li>Additional computational cost must be considered</li>
          </ul>
          <p style={{
            marginTop: '1rem',
            fontSize: '0.95rem',
            color: '#f97316',
            fontWeight: 600
          }}>
            Preferred AI candidate: Residual LSTM. Practical baseline and fallback: Linear Interpolation.
          </p>
        </div>
      </div>

      {/* Project Conclusion */}
      <div className="glass-panel" style={{
        marginBottom: '2rem',
        background: 'var(--bg-secondary)',
        border: '2px solid var(--border-color)'
      }}>
        <div className="glass-panel-header">
          <h2 style={{
            fontSize: '1.1rem',
            fontWeight: 700,
            color: 'var(--text-primary)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <BookOpen size={20} color="#94a3b8" />
            Project Conclusion
          </h2>
        </div>
        <div style={{ color: 'var(--text-secondary)', lineHeight: '1.7' }}>
          <p style={{ fontSize: '1rem' }}>
            Our experiments show that AI-based reconstruction does not automatically outperform mathematical interpolation. 
            Linear interpolation performed better for the short gap in our reported evaluation, while Residual LSTM achieved 
            modestly lower MSE for the longer gaps tested. We therefore select Residual LSTM as our preferred AI candidate 
            for further development, while retaining linear interpolation as a fast baseline and fallback. 
            Further evaluation is required before making an industrial deployment claim.
          </p>
        </div>
      </div>

      {/* CTA to Reconstruction Studio */}
      <Link 
        to="/reconstruction-studio"
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.75rem',
          padding: '1rem 2rem',
          background: 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)',
          color: '#ffffff',
          borderRadius: '10px',
          textDecoration: 'none',
          fontWeight: 700,
          fontSize: '1rem',
          boxShadow: '0 4px 20px rgba(6, 182, 212, 0.3)',
          transition: 'all 0.2s ease'
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.transform = 'translateY(-2px)';
          e.currentTarget.style.boxShadow = '0 6px 25px rgba(6, 182, 212, 0.4)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = 'translateY(0)';
          e.currentTarget.style.boxShadow = '0 4px 20px rgba(6, 182, 212, 0.3)';
        }}
      >
        <Brain size={20} />
        Open Reconstruction Studio
        <ArrowRight size={20} />
      </Link>
    </div>
  );
};
