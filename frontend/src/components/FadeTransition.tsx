import React from 'react';

const FadeTransition: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="fade-in">
    {children}
  </div>
);

export default FadeTransition;
// Add this to index.css:
// .fade-in { animation: fadeIn 0.5s; }
// @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } } 