import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { DatasetExplorer } from './pages/DatasetExplorer';
import { ReconstructionLab } from './pages/ReconstructionLab';
import { ModelPerformance } from './pages/ModelPerformance';
import { Experiments } from './pages/Experiments';

function App() {
  return (
    <Router>
      <div style={{ display: 'flex', minHeight: '100vh' }}>
        <Sidebar />
        <div style={{ flex: 1, marginLeft: '260px' }}>
          <TopBar />
          <main style={{ minHeight: 'calc(100vh - 64px)' }}>
            <Routes>
              <Route path="/" element={<Navigate to="/dataset-explorer" replace />} />
              <Route path="/dataset-explorer" element={<DatasetExplorer />} />
              <Route path="/reconstruction-lab" element={<ReconstructionLab />} />
              <Route path="/model-performance" element={<ModelPerformance />} />
              <Route path="/experiments" element={<Experiments />} />
              <Route path="*" element={<Navigate to="/dataset-explorer" replace />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;
