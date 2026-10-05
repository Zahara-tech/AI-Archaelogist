import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { Overview } from './pages/Overview';
import { Incidents } from './pages/Incidents';
import { IncidentDetail } from './pages/IncidentDetail';
import { CreateIncident } from './pages/CreateIncident';
import { Evidence } from './pages/Evidence';
import { Timeline } from './pages/Timeline';
import { Reconstruction } from './pages/Reconstruction';
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
              <Route path="/" element={<Overview />} />
              <Route path="/incidents" element={<Incidents />} />
              <Route path="/incidents/new" element={<CreateIncident />} />
              <Route path="/incidents/:id" element={<IncidentDetail />} />
              <Route path="/evidence" element={<Evidence />} />
              <Route path="/timeline" element={<Timeline />} />
              <Route path="/reconstruction" element={<Reconstruction />} />
              <Route path="/experiments" element={<Experiments />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;
