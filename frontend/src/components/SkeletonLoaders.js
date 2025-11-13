import React from 'react';

/**
 * SkeletonCircle - Skeleton loader circular para emojis/avatares
 * @param {Object} props
 * @param {string} props.size - Tamaño del círculo (default: '80px')
 * @param {string} props.className - Clases CSS adicionales
 */
const SkeletonCircle = ({ size = '80px', className = '' }) => {
  return (
    <div 
      className={`skeleton skeleton-circle ${className}`}
      style={{ width: size, height: size }}
      role="status"
      aria-label="Cargando..."
    />
  );
};

/**
 * SkeletonLine - Skeleton loader en forma de línea para textos
 * @param {Object} props
 * @param {string} props.width - Ancho de la línea (default: '100%')
 * @param {string} props.height - Alto de la línea (default: '16px')
 * @param {string} props.className - Clases CSS adicionales
 */
const SkeletonLine = ({ width = '100%', height = '16px', className = '' }) => {
  return (
    <div 
      className={`skeleton skeleton-line ${className}`}
      style={{ width, height }}
      role="status"
      aria-label="Cargando..."
    />
  );
};

/**
 * SkeletonCard - Skeleton loader para tarjetas completas
 * @param {Object} props
 * @param {number} props.lines - Número de líneas de texto (default: 3)
 * @param {boolean} props.showCircle - Mostrar círculo superior (default: false)
 * @param {string} props.className - Clases CSS adicionales
 */
const SkeletonCard = ({ lines = 3, showCircle = false, className = '' }) => {
  return (
    <div className={`skeleton-card ${className}`} role="status" aria-label="Cargando contenido...">
      {showCircle && (
        <div className="skeleton-card-header">
          <SkeletonCircle size="60px" />
        </div>
      )}
      
      <div className="skeleton-card-body">
        {/* Línea de título más gruesa */}
        <SkeletonLine width="70%" height="24px" className="skeleton-title" />
        
        {/* Líneas de contenido con anchos variados */}
        {Array.from({ length: lines }).map((_, index) => {
          // Variación de anchos: 90%, 95%, 80%, 85%, 75%...
          const widths = ['90%', '95%', '80%', '85%', '75%'];
          const width = widths[index % widths.length];
          
          return (
            <SkeletonLine 
              key={index}
              width={width}
              height="16px"
              className="skeleton-text"
            />
          );
        })}
      </div>
    </div>
  );
};

/**
 * PredictionSkeleton - Skeleton especializado para resultados de predicción
 * Replica la estructura visual del componente PredictionResults
 */
const PredictionSkeleton = () => {
  return (
    <div className="card results-card skeleton-results" role="status" aria-busy="true" aria-label="Analizando imagen...">
      {/* Banner de estado (simulado) */}
      <div className="skeleton-status-banner">
        <SkeletonCircle size="32px" />
        <SkeletonLine width="150px" height="20px" />
      </div>

      {/* Resultado principal */}
      <div className="skeleton-main-result">
        <SkeletonCircle size="100px" className="skeleton-emoji" />
        
        <div className="skeleton-result-text">
          <SkeletonLine width="250px" height="28px" className="skeleton-disease-name" />
          <SkeletonLine width="180px" height="18px" className="skeleton-plant-type" />
        </div>
      </div>

      {/* Gráfica de confianza */}
      <div className="skeleton-chart-section">
        <SkeletonLine width="140px" height="20px" className="skeleton-section-title" />
        <div className="skeleton-radial-chart">
          <SkeletonCircle size="180px" />
        </div>
      </div>

      {/* Información de enfermedad */}
      <div className="skeleton-info-section">
        <SkeletonLine width="120px" height="20px" className="skeleton-section-title" />
        <SkeletonLine width="100%" height="16px" />
        <SkeletonLine width="95%" height="16px" />
        <SkeletonLine width="90%" height="16px" />
      </div>

      {/* Top 5 predicciones */}
      <div className="skeleton-chart-section">
        <SkeletonLine width="180px" height="20px" className="skeleton-section-title" />
        <div className="skeleton-bar-chart">
          {[100, 80, 60, 45, 30].map((width, index) => (
            <div key={index} className="skeleton-bar-item">
              <SkeletonCircle size="32px" />
              <div className="skeleton-bar" style={{ width: `${width}%` }}>
                <SkeletonLine width="100%" height="28px" />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Timeline de severidad */}
      <div className="skeleton-timeline-section">
        <SkeletonLine width="160px" height="20px" className="skeleton-section-title" />
        <div className="skeleton-timeline">
          {[1, 2, 3, 4].map((index) => (
            <div key={index} className="skeleton-timeline-stage">
              <SkeletonCircle size="48px" />
              <SkeletonLine width="80px" height="14px" />
            </div>
          ))}
        </div>
      </div>

      {/* Heatmap */}
      <div className="skeleton-heatmap-section">
        <SkeletonLine width="140px" height="20px" className="skeleton-section-title" />
        <div className="skeleton-leaf-container">
          <div className="skeleton-leaf">
            <SkeletonCircle size="220px" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionSkeleton;
export { SkeletonCircle, SkeletonLine, SkeletonCard, PredictionSkeleton };
