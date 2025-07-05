import React, { useState } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000/infielder';

const initialForm = {
  height: '',
  weight: '',
  hand_speed_max: '',
  bat_speed_max: '',
  rot_acc_max: '',
  sixty_time: '',
  thirty_time: '',
  ten_yard_time: '',
  run_speed_max: '',
  exit_velo_max: '',
  exit_velo_avg: '',
  distance_max: '',
  sweet_spot_p: '',
  inf_velo: '',
  throwing_hand: '',
  hitting_handedness: '',
  player_region: '',
  primary_position: '',
  recruiting_class: '',
};

const positions = [
  'SS', '2B', '3B', '1B', 'C', 'OF', 'P', 'UTL'
];

const handOptions = [
  { value: '', label: 'Select...' },
  { value: 'R', label: 'Right' },
  { value: 'L', label: 'Left' },
  { value: 'S', label: 'Switch' },
];

const regionOptions = [
  '', 'Northeast', 'South', 'Midwest', 'West'
];

const NAV_OPTIONS = [
  { key: 'predictions', label: 'Recruitment Predictions' },
  { key: 'email', label: 'Email Templates' },
  { key: 'fit', label: 'Best Fit' },
];

function App() {
  const [form, setForm] = useState(initialForm);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState('predictions');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    // Only send filled fields, but exclude recruiting_class
    const payload: Record<string, any> = {};
    Object.entries(form).forEach(([k, v]) => {
      if (v !== '' && k !== 'recruiting_class') payload[k] = isNaN(Number(v)) ? v : Number(v);
    });
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Prediction failed');
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleExample = () => {
    setForm({
      height: '72', weight: '180', hand_speed_max: '22.5', bat_speed_max: '75', rot_acc_max: '18',
      sixty_time: '6.8', thirty_time: '3.2', ten_yard_time: '1.7', run_speed_max: '22', exit_velo_max: '88', exit_velo_avg: '78',
      distance_max: '320', sweet_spot_p: '0.75', inf_velo: '78', throwing_hand: 'R', hitting_handedness: 'R', player_region: 'West', primary_position: 'SS',
      recruiting_class: '',
    });
  };

  const handleClear = () => {
    setForm(initialForm);
    setResult(null);
    setError(null);
  };

  return (
    <div className="scoutlab-bg">
      <header className="scoutlab-header">
        <div className="scoutlab-logo-placeholder"></div>
        <div className="scoutlab-title">ScoutLab</div>
        <nav className="scoutlab-nav">
          {NAV_OPTIONS.map(opt => (
            <button
              key={opt.key}
              className={page === opt.key ? 'nav-active' : ''}
              onClick={() => setPage(opt.key)}
            >
              {opt.label}
            </button>
          ))}
        </nav>
      </header>
      <main className="scoutlab-main">
        {page === 'predictions' && (
          <div className="recruit-app-bg">
            <div className="recruit-app-container">
              <header className="recruit-header">
                <div className="recruit-logo">AI Baseball Recruit</div>
                <div className="recruit-subtitle">Infielder College Projection</div>
              </header>
              <form className="recruit-form" onSubmit={handleSubmit}>
                <div className="recruit-form-row">
                  <div className="recruit-form-group">
                    <label>Height (inches)</label>
                    <input name="height" type="number" step="0.1" value={form.height} onChange={handleChange} />
                  </div>
                  <div className="recruit-form-group">
                    <label>Weight (lbs)</label>
                    <input name="weight" type="number" step="0.1" value={form.weight} onChange={handleChange} />
                  </div>
                  <div className="recruit-form-group">
                    <label>Primary Position</label>
                    <select name="primary_position" value={form.primary_position} onChange={handleChange}>
                      <option value="">Select...</option>
                      {positions.map(pos => <option key={pos} value={pos}>{pos}</option>)}
                    </select>
                  </div>
                  <div className="recruit-form-group">
                    <label>Recruiting Class</label>
                    <input name="recruiting_class" type="text" value={form.recruiting_class} onChange={handleChange} />
                  </div>
                </div>
                <div className="recruit-form-row">
                  <div className="recruit-form-group">
                    <label>Hand Speed Max (mph)</label>
                    <input name="hand_speed_max" type="number" step="0.1" value={form.hand_speed_max} onChange={handleChange} />
                  </div>
                  <div className="recruit-form-group">
                    <label>Bat Speed Max (mph)</label>
                    <input name="bat_speed_max" type="number" step="0.1" value={form.bat_speed_max} onChange={handleChange} />
                  </div>
                  <div className="recruit-form-group">
                    <label>Rotational Accel Max</label>
                    <input name="rot_acc_max" type="number" step="0.1" value={form.rot_acc_max} onChange={handleChange} />
                  </div>
                  <div className="recruit-form-group">
                    <label>Infield Velo (mph)</label>
                    <input name="inf_velo" type="number" step="0.1" value={form.inf_velo} onChange={handleChange} />
                  </div>
                </div>
                <div className="recruit-form-row">
                  <div className="recruit-form-group">
                    <label>Sixty Time (s)</label>
                    <input name="sixty_time" type="number" step="0.01" value={form.sixty_time} onChange={handleChange} />
                  </div>
                  <div className="recruit-form-group">
                    <label>Thirty Time (s)</label>
                    <input name="thirty_time" type="number" step="0.01" value={form.thirty_time} onChange={handleChange} />
                  </div>
                  <div className="recruit-form-group">
                    <label>Ten Yard Time (s)</label>
                    <input name="ten_yard_time" type="number" step="0.01" value={form.ten_yard_time} onChange={handleChange} />
                  </div>
                  <div className="recruit-form-group">
                    <label>Run Speed Max (mph)</label>
                    <input name="run_speed_max" type="number" step="0.1" value={form.run_speed_max} onChange={handleChange} />
                  </div>
                </div>
                <div className="recruit-form-row">
                  <div className="recruit-form-group">
                    <label>Exit Velo Max (mph)</label>
                    <input name="exit_velo_max" type="number" step="0.1" value={form.exit_velo_max} onChange={handleChange} />
                  </div>
                  <div className="recruit-form-group">
                    <label>Exit Velo Avg (mph)</label>
                    <input name="exit_velo_avg" type="number" step="0.1" value={form.exit_velo_avg} onChange={handleChange} />
                  </div>
                  <div className="recruit-form-group">
                    <label>Distance Max (ft)</label>
                    <input name="distance_max" type="number" step="0.1" value={form.distance_max} onChange={handleChange} />
                  </div>
                  <div className="recruit-form-group">
                    <label>Sweet Spot %</label>
                    <input name="sweet_spot_p" type="number" step="0.01" min="0" max="1" value={form.sweet_spot_p} onChange={handleChange} />
                  </div>
                </div>
                <div className="recruit-form-row">
                  <div className="recruit-form-group">
                    <label>Throwing Hand</label>
                    <select name="throwing_hand" value={form.throwing_hand} onChange={handleChange}>
                      {handOptions.slice(0,3).map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                    </select>
                  </div>
                  <div className="recruit-form-group">
                    <label>Hitting Handedness</label>
                    <select name="hitting_handedness" value={form.hitting_handedness} onChange={handleChange}>
                      {handOptions.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                    </select>
                  </div>
                  <div className="recruit-form-group">
                    <label>Player Region</label>
                    <select name="player_region" value={form.player_region} onChange={handleChange}>
                      {regionOptions.map(opt => (
                        <option key={opt} value={opt}>{opt === '' ? 'Select...' : opt}</option>
                      ))}
                    </select>
                  </div>
                </div>
                <div className="recruit-form-actions">
                  <button type="button" onClick={handleExample}>Load Example</button>
                  <button type="button" onClick={handleClear}>Clear</button>
                  <button type="submit" disabled={loading}>{loading ? 'Predicting...' : 'Predict'}</button>
                </div>
              </form>
              <div className="recruit-result-panel">
                {error && <div className="recruit-error">{error}</div>}
                {result && (
                  <div className="recruit-result-card">
                    <h2>Prediction Result</h2>
                    <div className="recruit-prediction-main">{result.prediction}</div>
                    <div className="recruit-confidence">Confidence: {(result.confidence * 100).toFixed(1)}%</div>
                    <div className="recruit-stage">Stage: {result.stage}</div>
                    <div className="recruit-probabilities">
                      {Object.entries(result.probabilities).map(([cat, prob]) => {
                        const probNum = Number(prob);
                        return (
                          <div key={cat} className="recruit-prob-row">
                            <span className="recruit-prob-label">{cat}</span>
                            <div className="recruit-prob-bar-bg">
                              <div className="recruit-prob-bar" style={{width: `${(probNum*100).toFixed(1)}%`}}></div>
                            </div>
                            <span className="recruit-prob-value">{(probNum*100).toFixed(1)}%</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
              <footer className="recruit-footer">
                <span>Powered by AI &bull; Built for Baseball Recruiting</span>
              </footer>
            </div>
          </div>
        )}
        {page === 'email' && (
          <div className="scoutlab-placeholder">Email Templates coming soon.</div>
        )}
        {page === 'fit' && (
          <div className="scoutlab-placeholder">Best Fit feature coming soon.</div>
        )}
      </main>
    </div>
  );
}

export default App;
