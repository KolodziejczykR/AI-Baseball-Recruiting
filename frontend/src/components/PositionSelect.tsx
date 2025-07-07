import React from 'react';

const hitterPositions = [
  { value: 'SS', label: 'Shortstop (SS)' },
  { value: '2B', label: 'Second Base (2B)' },
  { value: '3B', label: 'Third Base (3B)' },
  { value: '1B', label: 'First Base (1B)' },
  { value: 'OF', label: 'Outfield (OF)' },
  { value: 'C', label: 'Catcher (C)' },
];
const pitcherPositions = [
  { value: 'P', label: 'Pitcher (P)' },
];

const PositionSelect: React.FC<{
  role: 'hitter' | 'pitcher',
  onSelect: (position: string) => void,
  onBack: () => void
}> = ({ role, onSelect, onBack }) => {
  const positions = role === 'hitter' ? hitterPositions : pitcherPositions;
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] fade-in">
      <div className="bg-white rounded-xl shadow-lg p-10 max-w-md w-full text-center">
        <h2 className="text-2xl font-bold mb-6">Select your primary position</h2>
        <div className="grid grid-cols-1 gap-4 mb-6">
          {positions.map(pos => (
            <button
              key={pos.value}
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg font-bold text-lg transition"
              onClick={() => onSelect(pos.value)}
            >
              {pos.label}
            </button>
          ))}
        </div>
        <button className="text-blue-600 underline mt-2" onClick={onBack}>Back</button>
      </div>
    </div>
  );
};

export default PositionSelect; 