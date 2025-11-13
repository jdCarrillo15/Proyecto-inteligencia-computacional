import React from 'react';

const TipsCard = () => {
  return (
    <aside className="info-card tips-card" aria-label="Guía de mejores prácticas">
      <h3><span aria-hidden="true">💡</span> Guía para Mejores Resultados</h3>
      <ul className="tips-list">
        <li className="tip-item">
          <span className="tip-icon" aria-hidden="true">📸</span>
          <span className="tip-text">Sube fotos claras de hojas afectadas</span>
        </li>
        <li className="tip-item">
          <span className="tip-icon" aria-hidden="true">👁️</span>
          <span className="tip-text">Asegúrate de que los síntomas sean visibles</span>
        </li>
        <li className="tip-item">
          <span className="tip-icon" aria-hidden="true">☀️</span>
          <span className="tip-text">Mejor con luz natural (evita flash)</span>
        </li>
        <li className="tip-item">
          <span className="tip-icon" aria-hidden="true">🎯</span>
          <span className="tip-text">Evita fondos complejos o distracciones</span>
        </li>
        <li className="tip-item">
          <span className="tip-icon" aria-hidden="true">🔍</span>
          <span className="tip-text">Enfoca la hoja completa en el encuadre</span>
        </li>
      </ul>
    </aside>
  );
};

export default TipsCard;
