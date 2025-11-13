import React from 'react';

const Header = ({ darkMode, onToggleDarkMode }) => {
  return (
    <header className="header" role="banner">
      <div className="header-content">
        {/* Logo/Icon */}
        <div className="header-logo">
          <span className="header-icon" aria-hidden="true">🌱</span>
        </div>

        {/* Título y Subtítulo */}
        <div className="header-text">
          <h1 className="title">
            Detector de Enfermedades en Plantas
            <span className="emoji">🔬</span>
          </h1>
          <p className="subtitle">
            Sistema de diagnóstico agrícola con CNN
          </p>
        </div>

        {/* Stats Inline */}
        <div className="header-stats">
          <div className="stat-item">
            <span className="stat-icon" aria-hidden="true">🎯</span>
            <span className="stat-value">95%</span>
            <span>precisión</span>
          </div>
          <div className="stat-divider" aria-hidden="true"></div>
          <div className="stat-item">
            <span className="stat-icon" aria-hidden="true">🦠</span>
            <span className="stat-value">15</span>
            <span>enfermedades</span>
          </div>
        </div>

        {/* Actions */}
        <div className="header-actions">
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
      </div>
    </header>
  );
};

export default Header;
