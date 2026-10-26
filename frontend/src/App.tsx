import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { Overview } from './pages/Overview';
import { ReconstructionStudio } from './pages/ReconstructionStudio';
import { ModelPerformance } from './pages/ModelPerformance';
import { TrainingExperiments } from './pages/TrainingExperiments';
import { AboutProject } from './pages/AboutProject';

function App() {
  return (
    <Router>
      <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--bg-primary)' }}>
        <Sidebar />
        <div style={{ flex: 1, marginLeft: '260px', display: 'flex', flexDirection: 'column' }}>
          <TopBar />
          <main style={{ marginTop: '68px', minHeight: 'calc(100vh - 68px)', flex: 1 }}>
            <Routes>
              <Route path="/" element={<Navigate to="/overview" replace />} />
              <Route path="/overview" element={<Overview />} />
              <Route path="/reconstruction-studio" element={<ReconstructionStudio />} />
              <Route path="/model-performance" element={<ModelPerformance />} />
              <Route path="/training-experiments" element={<TrainingExperiments />} />
              <Route path="/about" element={<AboutProject />} />
              <Route path="*" element={<Navigate to="/overview" replace />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;
