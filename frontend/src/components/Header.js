import React from 'react';

const Header = ({ darkMode, onToggleDarkMode }) => {
  return (
    <header className="header" role="banner">
      <div className="header-content">
        <div className="header-text">
          <h1 className="title">
            <span className="emoji">🌱</span>
            Detector de Enfermedades en Plantas
            <span className="emoji">🔬</span>
          </h1>
          <p className="subtitle">
            Sistema de diagnóstico agrícola con CNN | Proyecto académico para fitopatología
          </p>
        </div>
        <button 
          className="dark-mode-toggle"
          onClick={onToggleDarkMode}
          onKeyPress={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              onToggleDarkMode();
            }
          }}
          aria-label={darkMode ? 'Cambiar a modo claro' : 'Cambiar a modo oscuro'}
          aria-pressed={darkMode}
          title={darkMode ? 'Cambiar a modo claro' : 'Cambiar a modo oscuro'}
        >
          <span aria-hidden="true">{darkMode ? '☀️' : '🌙'}</span>
          <span className="sr-only">{darkMode ? 'Modo claro' : 'Modo oscuro'}</span>
        </button>
      </div>
    </header>
  );
};

export default Header;
