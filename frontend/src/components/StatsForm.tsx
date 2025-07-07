import React, { useState } from 'react';
import FadeTransition from './FadeTransition';

const API_BASE = 'http://localhost:8000';

const infielderFields = [
  { key: 'height', label: 'Height (inches)', type: 'number', placeholder: 'e.g., 72' },
  { key: 'weight', label: 'Weight (lbs)', type: 'number', placeholder: 'e.g., 180' },
  { key: 'hand_speed_max', label: 'Hand Speed Max (mph)', type: 'number', placeholder: 'e.g., 22.5' },
  { key: 'bat_speed_max', label: 'Bat Speed Max (mph)', type: 'number', placeholder: 'e.g., 75' },
  { key: 'rot_acc_max', label: 'Rotational Acceleration Max', type: 'number', placeholder: 'e.g., 18' },
  { key: 'sixty_time', label: '60-Yard Dash (s)', type: 'number', placeholder: 'e.g., 6.8' },
  { key: 'thirty_time', label: '30-Yard Dash (s)', type: 'number', placeholder: 'e.g., 3.2' },
  { key: 'ten_yard_time', label: '10-Yard Dash (s)', type: 'number', placeholder: 'e.g., 1.7' },
  { key: 'run_speed_max', label: 'Run Speed Max (mph)', type: 'number', placeholder: 'e.g., 22' },
  { key: 'exit_velo_max', label: 'Exit Velo Max (mph)', type: 'number', placeholder: 'e.g., 88' },
  { key: 'exit_velo_avg', label: 'Exit Velo Avg (mph)', type: 'number', placeholder: 'e.g., 78' },
  { key: 'distance_max', label: 'Distance Max (ft)', type: 'number', placeholder: 'e.g., 320' },
  { key: 'sweet_spot_p', label: 'Sweet Spot % (0-1)', type: 'number', placeholder: 'e.g., 0.75', step: '0.01', min: 0, max: 1 },
  { key: 'inf_velo', label: 'Infield Velo (mph)', type: 'number', placeholder: 'e.g., 78' },
  { key: 'throwing_hand', label: 'Throwing Hand', type: 'select', options: ['R', 'L'] },
  { key: 'hitting_handedness', label: 'Hitting Handedness', type: 'select', options: ['R', 'L', 'S'] },
  { key: 'player_region', label: 'Player Region', type: 'select', options: ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West', 'Any'] },
  { key: 'primary_position', label: 'Primary Position', type: 'hidden', value: 'infield' },
];

const outfielderFields = [
  { key: 'height', label: 'Height (inches)', type: 'number', placeholder: 'e.g., 72' },
  { key: 'weight', label: 'Weight (lbs)', type: 'number', placeholder: 'e.g., 180' },
  { key: 'hand_speed_max', label: 'Hand Speed Max (mph)', type: 'number', placeholder: 'e.g., 22.5' },
  { key: 'bat_speed_max', label: 'Bat Speed Max (mph)', type: 'number', placeholder: 'e.g., 75' },
  { key: 'rot_acc_max', label: 'Rotational Acceleration Max', type: 'number', placeholder: 'e.g., 18' },
  { key: 'sixty_time', label: '60-Yard Dash (s)', type: 'number', placeholder: 'e.g., 6.8' },
  { key: 'thirty_time', label: '30-Yard Dash (s)', type: 'number', placeholder: 'e.g., 3.2' },
  { key: 'ten_yard_time', label: '10-Yard Dash (s)', type: 'number', placeholder: 'e.g., 1.7' },
  { key: 'run_speed_max', label: 'Run Speed Max (mph)', type: 'number', placeholder: 'e.g., 22' },
  { key: 'exit_velo_max', label: 'Exit Velo Max (mph)', type: 'number', placeholder: 'e.g., 88' },
  { key: 'exit_velo_avg', label: 'Exit Velo Avg (mph)', type: 'number', placeholder: 'e.g., 78' },
  { key: 'distance_max', label: 'Distance Max (ft)', type: 'number', placeholder: 'e.g., 320' },
  { key: 'sweet_spot_p', label: 'Sweet Spot % (0-1)', type: 'number', placeholder: 'e.g., 0.75', step: '0.01', min: 0, max: 1 },
  { key: 'of_velo', label: 'Outfield Velo (mph)', type: 'number', placeholder: 'e.g., 78' },
  { key: 'throwing_hand', label: 'Throwing Hand', type: 'select', options: ['R', 'L'] },
  { key: 'hitting_handedness', label: 'Hitting Handedness', type: 'select', options: ['R', 'L', 'S'] },
  { key: 'player_region', label: 'Player Region', type: 'select', options: ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West', 'Any'] },
  { key: 'primary_position', label: 'Primary Position', type: 'hidden', value: 'OF' },
];

const infielderExample = {
  height: '72', weight: '180', hand_speed_max: '22.5', bat_speed_max: '75', rot_acc_max: '18', sixty_time: '6.8', thirty_time: '3.2', ten_yard_time: '1.7', run_speed_max: '22', exit_velo_max: '88', exit_velo_avg: '78', distance_max: '320', sweet_spot_p: '0.75', inf_velo: '78', throwing_hand: 'R', hitting_handedness: 'R', player_region: 'West', primary_position: 'infield',
};
const outfielderExample = {
  height: '72', weight: '180', hand_speed_max: '22.5', bat_speed_max: '75', rot_acc_max: '18', sixty_time: '6.8', thirty_time: '3.2', ten_yard_time: '1.7', run_speed_max: '22', exit_velo_max: '88', exit_velo_avg: '78', distance_max: '320', sweet_spot_p: '0.75', of_velo: '78', throwing_hand: 'R', hitting_handedness: 'R', player_region: 'West', primary_position: 'OF',
};

const StatsForm: React.FC<{ role: 'hitter' | 'pitcher'; position: string; onBack: () => void }> = ({ role, position, onBack }) => {
  const isInfielder = ['SS', '2B', '3B', '1B'].includes(position);
  const isOutfielder = position === 'OF';
  const [form, setForm] = useState<Record<string, string>>(isInfielder ? infielderExample : isOutfielder ? outfielderExample : {});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const fields = isInfielder ? infielderFields : isOutfielder ? outfielderFields : [];
  const example = isInfielder ? infielderExample : isOutfielder ? outfielderExample : {};
  const endpoint = isInfielder ? '/infielder/predict' : isOutfielder ? '/outfielder/predict' : '';

  const handleChange = (key: string, value: string) => {
    setForm((prev: any) => ({ ...prev, [key]: value }));
  };

  const handleExample = () => {
    setForm(example);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload: Record<string, any> = {};
      Object.entries(form).forEach(([k, v]) => {
        if (v !== '') payload[k] = isNaN(Number(v)) ? v : Number(v);
      });
      // Add the specific position to the payload
      payload.primary_position = position;
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Prediction failed');
      setResult(data);
    } catch (err) {
      let msg = 'Unknown error';
      if (err instanceof Error) msg = err.message;
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  if (!isInfielder && !isOutfielder) {
    return (
      <FadeTransition>
        <div className="flex flex-col items-center justify-center min-h-[60vh]">
          <div className="bg-white rounded-xl shadow-lg p-10 max-w-md w-full text-center">
            <h2 className="text-2xl font-bold mb-6">Coming soon</h2>
            <div className="mb-6 text-gray-500">Support for this position is coming soon.</div>
            <button className="text-blue-600 underline mt-2" onClick={onBack}>Back</button>
          </div>
        </div>
      </FadeTransition>
    );
  }

  return (
    <FadeTransition>
      <div className="flex flex-col items-center justify-center min-h-[60vh]">
        <div className="bg-white rounded-xl shadow-lg p-10 max-w-xl w-full text-center">
          <h2 className="text-2xl font-bold mb-6">Enter your stats ({isInfielder ? 'Infielder' : 'Outfielder'}: {position})</h2>
          <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            {fields.map(field => field.type === 'hidden' ? null : (
              <div key={field.key} className="text-left">
                <label className="block font-medium mb-1">{field.label}</label>
                {field.type === 'select' ? (
                  <select
                    className="w-full border rounded px-3 py-2"
                    value={form[field.key] || ''}
                    onChange={e => handleChange(field.key, e.target.value)}
                  >
                    <option value="">Select...</option>
                    {field.options && field.options.map((opt: string) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    className="w-full border rounded px-3 py-2"
                    type={field.type}
                    value={form[field.key] || ''}
                    onChange={e => handleChange(field.key, e.target.value)}
                    placeholder={field.placeholder}
                    {...(field.step ? { step: field.step } : {})}
                    {...(field.min !== undefined ? { min: field.min } : {})}
                    {...(field.max !== undefined ? { max: field.max } : {})}
                  />
                )}
              </div>
            ))}
            <div className="col-span-1 md:col-span-2 flex flex-col md:flex-row gap-4 mt-4 justify-center items-center">
              <button
                type="submit"
                className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg font-bold text-lg disabled:opacity-50"
                disabled={loading}
              >
                {loading ? 'Predicting...' : 'Predict'}
              </button>
              <button
                type="button"
                className="px-6 py-3 rounded-lg border font-semibold"
                onClick={handleExample}
              >
                Load Example
              </button>
              <button className="px-6 py-3 rounded-lg border font-semibold" type="button" onClick={onBack}>Back</button>
            </div>
          </form>
          {error && <div className="text-red-600 mb-4 font-semibold text-center">{error}</div>}
          {result && (
            <div className="mt-8 bg-gray-50 rounded-xl shadow p-6">
              <h3 className="text-xl font-bold mb-2">Prediction Result</h3>
              <div className="mb-2 text-lg font-semibold">{result.prediction}</div>
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
          )}
        </div>
      </div>
    </FadeTransition>
  );
};

export default StatsForm; 