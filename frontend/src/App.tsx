import React, { useState } from 'react';
import Background from './components/Background';
import RoleSelect from './components/RoleSelect';
import PositionSelect from './components/PositionSelect';
import FadeTransition from './components/FadeTransition';
import StatsForm from './components/StatsForm';
import './App.css';

const API_BASE = 'http://localhost:8000';

const positions = [
  { value: 'SS', label: 'Shortstop (SS)' },
  { value: '2B', label: 'Second Base (2B)' },
  { value: '3B', label: 'Third Base (3B)' },
  { value: '1B', label: 'First Base (1B)' },
  { value: 'OF', label: 'Outfield (OF)' },
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

const App: React.FC = () => {
  const [page, setPage] = useState<'role' | 'position' | 'stats'>('role');
  const [role, setRole] = useState<'hitter' | 'pitcher' | null>(null);
  const [position, setPosition] = useState<string>('');

  const handleRoleSelect = (selectedRole: 'hitter' | 'pitcher') => {
    setRole(selectedRole);
    setPage('position');
  };

  const handlePositionSelect = (selectedPosition: string) => {
    setPosition(selectedPosition);
    setPage('stats');
  };

  const handleBackToRole = () => {
    setRole(null);
    setPosition('');
    setPage('role');
  };

  const handleBackToPosition = () => {
    setPosition('');
    setPage('position');
  };

  return (
    <Background>
      {page === 'role' && (
        <FadeTransition>
          <RoleSelect onSelect={handleRoleSelect} />
        </FadeTransition>
      )}
      {page === 'position' && role && (
        <FadeTransition>
          <PositionSelect role={role} onSelect={handlePositionSelect} onBack={handleBackToRole} />
        </FadeTransition>
      )}
      {page === 'stats' && role && position && (
        <StatsForm role={role} position={position} onBack={handleBackToPosition} />
      )}
    </Background>
  );
};

export default App;
