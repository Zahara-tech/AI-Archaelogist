import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { OverviewNew } from './pages/OverviewNew';
import { SensorReconstructionDashboard } from './pages/SensorReconstructionDashboard';
import { ModelPerformanceNew } from './pages/ModelPerformanceNew';
import { TrainingExperiments } from './pages/TrainingExperiments';
import { Methodology } from './pages/Methodology';

function App() {
  return (
    <Router>
      <div style={{ display: 'flex', minHeight: '100vh' }}>
        <Sidebar />
        <div style={{ flex: 1, marginLeft: '260px' }}>
          <TopBar />
          <main style={{ minHeight: 'calc(100vh - 64px)' }}>
            <Routes>
              <Route path="/" element={<Navigate to="/overview" replace />} />
              <Route path="/overview" element={<OverviewNew />} />
              <Route path="/reconstruction-studio" element={<SensorReconstructionDashboard />} />
              <Route path="/model-performance" element={<ModelPerformanceNew />} />
              <Route path="/training-experiments" element={<TrainingExperiments />} />
              <Route path="/about" element={<Methodology />} />
              <Route path="*" element={<Navigate to="/overview" replace />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;
