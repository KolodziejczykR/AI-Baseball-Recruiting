import React from 'react';

const Background: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
    <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-2xl font-bold text-blue-600">ScoutLab</span>
        </div>
        <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-semibold">Beta Version</span>
      </div>
    </header>
    {children}
  </div>
);

export default Background; 