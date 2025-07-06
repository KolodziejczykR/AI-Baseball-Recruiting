import React, { useState } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000';

const positions = [
  { value: 'SS', label: 'Shortstop (SS)' },
  { value: '2B', label: 'Second Base (2B)' },
  { value: '3B', label: 'Third Base (3B)' },
  { value: '1B', label: 'First Base (1B)' },
];

const regions = [
  { value: 'Northeast', label: 'Northeast' },
  { value: 'Southeast', label: 'Southeast' },
  { value: 'Midwest', label: 'Midwest' },
  { value: 'Southwest', label: 'Southwest' },
  { value: 'West', label: 'West' },
  { value: 'Any', label: 'Any Region' },
];

const handOptions = [
  { value: '', label: 'Select...' },
  { value: 'R', label: 'Right' },
  { value: 'L', label: 'Left' },
  { value: 'S', label: 'Switch' },
];

const graduationYears = [
  '2025', '2026', '2027', '2028', '2029'
];

const exampleInput = {
  height: '72',
  weight: '180',
  hand_speed_max: '22.5',
  bat_speed_max: '75',
  rot_acc_max: '18',
  sixty_time: '6.8',
  thirty_time: '3.2',
  ten_yard_time: '1.7',
  run_speed_max: '22',
  exit_velo_max: '88',
  exit_velo_avg: '78',
  distance_max: '320',
  sweet_spot_p: '0.75',
  inf_velo: '78',
  throwing_hand: 'R',
  hitting_handedness: 'R',
  player_region: 'West',
  primary_position: 'SS',
  graduationYear: '2025',
};

interface PlayerState {
  height: string;
  weight: string;
  hand_speed_max: string;
  bat_speed_max: string;
  rot_acc_max: string;
  sixty_time: string;
  thirty_time: string;
  ten_yard_time: string;
  run_speed_max: string;
  exit_velo_max: string;
  exit_velo_avg: string;
  distance_max: string;
  sweet_spot_p: string;
  inf_velo: string;
  throwing_hand: string;
  hitting_handedness: string;
  player_region: string;
  primary_position: string;
  graduationYear: string; // not sent to backend
}

interface PredictionResult {
  prediction: string;
  probabilities: Record<string, number>;
  confidence: number;
  stage: string;
}

const initialPlayer: PlayerState = {
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
  graduationYear: '',
};

const steps = [
  { key: 'input', label: 'Input' },
  { key: 'ml-prediction', label: 'ML Model Prediction' },
  { key: 'final-recommendations', label: 'Final Recommendations' },
];

function App() {
  const [player, setPlayer] = useState<PlayerState>(initialPlayer);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState('input');

  const handleChange = (field: keyof PlayerState, value: string) => {
    setPlayer((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    setStep('ml-prediction');

    // Only infielders supported for now
    if (!['SS', '2B', '3B', '1B'].includes(player.primary_position)) {
      setLoading(false);
      setStep('input');
      setError('Only infielders are supported for now.');
      return;
    }

    // Map frontend fields to backend keys (exclude graduationYear)
    const payload: Record<string, any> = {};
    Object.entries(player).forEach(([k, v]) => {
      if (k !== 'graduationYear' && v !== '') payload[k] = isNaN(Number(v)) ? v : Number(v);
    });
    try {
      const res = await fetch(`${API_BASE}/infielder/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Prediction failed');
      setResult(data);
      setStep('final-recommendations');
    } catch (err) {
      let msg = 'Unknown error';
      if (err instanceof Error) msg = err.message;
      setError(msg);
      setStep('input');
    } finally {
      setLoading(false);
    }
  };

  const handleShowExample = () => {
    setPlayer(exampleInput);
  };

  const resetForm = () => {
    setPlayer(initialPlayer);
    setResult(null);
    setError(null);
    setStep('input');
  };

  const renderStepper = () => (
    <div className="flex justify-between items-center mb-8">
      {steps.map((s, idx) => (
        <div key={s.key} className="flex-1 flex flex-col items-center">
          <div className={`rounded-full w-8 h-8 flex items-center justify-center font-bold text-white ${step === s.key || steps.findIndex(st => st.key === step) > idx ? 'bg-blue-600' : 'bg-gray-300'}`}>{idx + 1}</div>
          <div className={`mt-2 text-xs font-semibold ${step === s.key ? 'text-blue-700' : 'text-gray-500'}`}>{s.label}</div>
        </div>
      ))}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-2xl font-bold text-blue-600">ScoutLab</span>
          </div>
          <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-semibold">Beta Version</span>
        </div>
      </header>
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-3xl mx-auto">
          <div className="text-center mb-10">
            <h2 className="text-4xl font-bold text-gray-900 mb-2">Find Your Perfect College Baseball Match</h2>
            <p className="text-lg text-gray-600 mb-4">AI-powered recruitment analysis to connect you with the right college baseball programs</p>
          </div>
          {renderStepper()}
          <form onSubmit={handleSubmit} className="bg-white rounded-xl shadow-lg p-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block font-medium mb-1">Primary Position</label>
                <select className="w-full border rounded px-3 py-2" value={player.primary_position} onChange={e => handleChange('primary_position', e.target.value)} required>
                  <option value="">Select position</option>
                  {positions.map(pos => <option key={pos.value} value={pos.value}>{pos.label}</option>)}
                </select>
              </div>
              <div>
                <label className="block font-medium mb-1">Graduation/Class Year</label>
                <select className="w-full border rounded px-3 py-2" value={player.graduationYear} onChange={e => handleChange('graduationYear', e.target.value)}>
                  <option value="">Select year</option>
                  {graduationYears.map(y => <option key={y} value={y}>{y}</option>)}
                </select>
              </div>
              <div>
                <label className="block font-medium mb-1">Height (inches)</label>
                <input className="w-full border rounded px-3 py-2" value={player.height} onChange={e => handleChange('height', e.target.value)} placeholder="e.g., 72" type="number" />
              </div>
              <div>
                <label className="block font-medium mb-1">Weight (lbs)</label>
                <input className="w-full border rounded px-3 py-2" value={player.weight} onChange={e => handleChange('weight', e.target.value)} placeholder="e.g., 180" type="number" />
              </div>
              <div>
                <label className="block font-medium mb-1">Throwing Hand</label>
                <select className="w-full border rounded px-3 py-2" value={player.throwing_hand} onChange={e => handleChange('throwing_hand', e.target.value)}>
                  {handOptions.slice(0,3).map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                </select>
              </div>
              <div>
                <label className="block font-medium mb-1">Hitting Handedness</label>
                <select className="w-full border rounded px-3 py-2" value={player.hitting_handedness} onChange={e => handleChange('hitting_handedness', e.target.value)}>
                  {handOptions.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                </select>
              </div>
              <div>
                <label className="block font-medium mb-1">Player Region</label>
                <select className="w-full border rounded px-3 py-2" value={player.player_region} onChange={e => handleChange('player_region', e.target.value)}>
                  <option value="">Select region</option>
                  {regions.map(r => <option key={r.value} value={r.value}>{r.label}</option>)}
                </select>
              </div>
              <div>
                <label className="block font-medium mb-1">Hand Speed Max (mph)</label>
                <input className="w-full border rounded px-3 py-2" value={player.hand_speed_max} onChange={e => handleChange('hand_speed_max', e.target.value)} placeholder="e.g., 22.5" type="number" />
              </div>
              <div>
                <label className="block font-medium mb-1">Bat Speed Max (mph)</label>
                <input className="w-full border rounded px-3 py-2" value={player.bat_speed_max} onChange={e => handleChange('bat_speed_max', e.target.value)} placeholder="e.g., 75" type="number" />
              </div>
              <div>
                <label className="block font-medium mb-1">Rotational Acceleration Max</label>
                <input className="w-full border rounded px-3 py-2" value={player.rot_acc_max} onChange={e => handleChange('rot_acc_max', e.target.value)} placeholder="e.g., 18" type="number" />
              </div>
              <div>
                <label className="block font-medium mb-1">60-Yard Dash (s)</label>
                <input className="w-full border rounded px-3 py-2" value={player.sixty_time} onChange={e => handleChange('sixty_time', e.target.value)} placeholder="e.g., 6.8" type="number" />
              </div>
              <div>
                <label className="block font-medium mb-1">30-Yard Dash (s)</label>
                <input className="w-full border rounded px-3 py-2" value={player.thirty_time} onChange={e => handleChange('thirty_time', e.target.value)} placeholder="e.g., 3.2" type="number" />
              </div>
              <div>
                <label className="block font-medium mb-1">10-Yard Dash (s)</label>
                <input className="w-full border rounded px-3 py-2" value={player.ten_yard_time} onChange={e => handleChange('ten_yard_time', e.target.value)} placeholder="e.g., 1.7" type="number" />
              </div>
              <div>
                <label className="block font-medium mb-1">Run Speed Max (mph)</label>
                <input className="w-full border rounded px-3 py-2" value={player.run_speed_max} onChange={e => handleChange('run_speed_max', e.target.value)} placeholder="e.g., 22" type="number" />
              </div>
              <div>
                <label className="block font-medium mb-1">Exit Velo Max (mph)</label>
                <input className="w-full border rounded px-3 py-2" value={player.exit_velo_max} onChange={e => handleChange('exit_velo_max', e.target.value)} placeholder="e.g., 88" type="number" />
              </div>
              <div>
                <label className="block font-medium mb-1">Exit Velo Avg (mph)</label>
                <input className="w-full border rounded px-3 py-2" value={player.exit_velo_avg} onChange={e => handleChange('exit_velo_avg', e.target.value)} placeholder="e.g., 78" type="number" />
              </div>
              <div>
                <label className="block font-medium mb-1">Distance Max (ft)</label>
                <input className="w-full border rounded px-3 py-2" value={player.distance_max} onChange={e => handleChange('distance_max', e.target.value)} placeholder="e.g., 320" type="number" />
              </div>
              <div>
                <label className="block font-medium mb-1">Sweet Spot % (0-1)</label>
                <input className="w-full border rounded px-3 py-2" value={player.sweet_spot_p} onChange={e => handleChange('sweet_spot_p', e.target.value)} placeholder="e.g., 0.75" type="number" step="0.01" min="0" max="1" />
              </div>
              <div>
                <label className="block font-medium mb-1">Infield Velo (mph)</label>
                <input className="w-full border rounded px-3 py-2" value={player.inf_velo} onChange={e => handleChange('inf_velo', e.target.value)} placeholder="e.g., 78" type="number" />
              </div>
            </div>
            {error && <div className="text-red-600 mb-4 font-semibold text-center">{error}</div>}
            <div className="flex flex-col md:flex-row justify-center items-center mt-6 gap-4">
              <button
                type="submit"
                className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg font-bold text-lg disabled:opacity-50"
                disabled={loading || !player.primary_position}
              >
                {loading ? 'Finding Matches...' : 'Find My College Matches'}
              </button>
              <button
                type="button"
                className="px-6 py-3 rounded-lg border font-semibold"
                onClick={handleShowExample}
              >
                Show Example
              </button>
              {step !== 'input' && (
                <button type="button" className="px-6 py-3 rounded-lg border font-semibold" onClick={resetForm}>Start Over</button>
              )}
            </div>
          </form>

          {result && typeof result === 'object' && (
            <div className="mt-10 bg-white rounded-xl shadow-lg p-8">
              <h3 className="text-2xl font-bold text-center mb-4">Your Prediction</h3>
              <div className="flex flex-col items-center space-y-2">
                <span className="text-lg font-semibold">Primary Prediction:</span>
                <span className="text-2xl font-bold text-blue-700">{result.prediction}</span>
                <div className="w-full mt-4">
                  {result.prediction === 'Non D1' ? (
                    <>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Non D1</span>
                        <span>{Math.round((result.probabilities?.['Non D1'] || 0) * 100)}%</span>
                      </div>
                      <div className="w-full h-2 bg-yellow-100 rounded mb-2">
                        <div className="h-2 bg-yellow-600 rounded" style={{ width: `${(result.probabilities?.['Non D1'] || 0) * 100}%` }}></div>
                      </div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>D1</span>
                        <span>{Math.round((result.probabilities?.['D1'] || 0) * 100)}%</span>
                      </div>
                      <div className="w-full h-2 bg-blue-100 rounded mb-2">
                        <div className="h-2 bg-blue-600 rounded" style={{ width: `${(result.probabilities?.['D1'] || 0) * 100}%` }}></div>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Power 4 D1</span>
                        <span>{Math.round((result.probabilities?.['Power 4 D1'] || 0) * 100)}%</span>
                      </div>
                      <div className="w-full h-2 bg-blue-100 rounded mb-2">
                        <div className="h-2 bg-blue-600 rounded" style={{ width: `${(result.probabilities?.['Power 4 D1'] || 0) * 100}%` }}></div>
                      </div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Non P4 D1</span>
                        <span>{Math.round((result.probabilities?.['Non P4 D1'] || 0) * 100)}%</span>
                      </div>
                      <div className="w-full h-2 bg-green-100 rounded mb-2">
                        <div className="h-2 bg-green-600 rounded" style={{ width: `${(result.probabilities?.['Non P4 D1'] || 0) * 100}%` }}></div>
                      </div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Non D1</span>
                        <span>{Math.round((result.probabilities?.['Non D1'] || 0) * 100)}%</span>
                      </div>
                      <div className="w-full h-2 bg-yellow-100 rounded mb-2">
                        <div className="h-2 bg-yellow-600 rounded" style={{ width: `${(result.probabilities?.['Non D1'] || 0) * 100}%` }}></div>
                      </div>
                    </>
                  )}
                </div>
                <div className="mt-4 text-gray-600 text-center">Confidence: <span className="font-bold">{Math.round((result.confidence || 0) * 100)}%</span></div>
                <div className="mt-2 text-gray-500 text-center text-sm">Stage: {result.stage}</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
