import React from 'react';

const RoleSelect: React.FC<{ onSelect: (role: 'hitter' | 'pitcher') => void }> = ({ onSelect }) => (
  <div className="flex flex-col items-center justify-center min-h-[60vh]">
    <div className="bg-white rounded-xl shadow-lg p-10 max-w-md w-full text-center">
      <h2 className="text-2xl font-bold mb-6">Are you a hitter or a pitcher?</h2>
      <div className="flex gap-6 justify-center">
        <button
          className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg font-bold text-lg transition"
          onClick={() => onSelect('hitter')}
        >
          Hitter
        </button>
        <button
          className="bg-green-600 hover:bg-green-700 text-white px-8 py-3 rounded-lg font-bold text-lg transition"
          onClick={() => onSelect('pitcher')}
        >
          Pitcher
        </button>
      </div>
    </div>
  </div>
);

export default RoleSelect; 