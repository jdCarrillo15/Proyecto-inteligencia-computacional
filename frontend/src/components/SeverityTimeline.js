import React from 'react';

/**
 * Componente Timeline de Severidad
 * Visualiza la progresión de enfermedad por etapas con estado activo
 * 
 * @param {string} severityLevel - Nivel de severidad ('Severidad Baja', 'Severidad Media', 'Severidad Alta', 'Saludable')
 * @param {number} confidence - Porcentaje de confianza (0-100)
 * @returns {JSX.Element} Timeline visual de severidad
 */
const SeverityTimeline = ({ severityLevel = 'Severidad Media', confidence = 0 }) => {
  // Convertir confidence a número (por si viene como string desde la API)
  const confidenceValue = typeof confidence === 'number' ? confidence : parseFloat(confidence) || 0;
  
  // Definir etapas del timeline
  const stages = [
    {
      id: 'early',
      label: 'Temprana',
      description: 'Síntomas iniciales',
      icon: '🌱',
      color: '#10b981', // green-500
      bgColor: '#d1fae5', // green-100
      range: '0-25%',
      minConfidence: 0,
      maxConfidence: 25
    },
    {
      id: 'medium',
      label: 'Media',
      description: 'Síntomas visibles',
      icon: '⚠️',
      color: '#f59e0b', // yellow-500
      bgColor: '#fef3c7', // yellow-100
      range: '26-50%',
      minConfidence: 26,
      maxConfidence: 50
    },
    {
      id: 'advanced',
      label: 'Avanzada',
      description: 'Requiere atención',
      icon: '🔶',
      color: '#f97316', // orange-500
      bgColor: '#ffedd5', // orange-100
      range: '51-75%',
      minConfidence: 51,
      maxConfidence: 75
    },
    {
      id: 'critical',
      label: 'Crítica',
      description: 'Acción urgente',
      icon: '🚨',
      color: '#dc2626', // red-600
      bgColor: '#fee2e2', // red-100
      range: '76-100%',
      minConfidence: 76,
      maxConfidence: 100
    }
  ];

  // Determinar etapa activa basada en severityLevel
  const getActiveStageIndex = () => {
    const level = severityLevel.toLowerCase();
    
    // Si es saludable, no hay etapa activa (retorna -1)
    if (level.includes('saludable') || level === 'saludable') {
      return -1;
    }
    
    // Mapeo de severidad a índice de etapa
    if (level.includes('baja')) return 0; // Temprana
    if (level.includes('media')) return 1; // Media
    if (level.includes('alta')) return 2; // Avanzada
    
    // Por defecto, usar confianza para determinar
    if (confidenceValue <= 25) return 0;
    if (confidenceValue <= 50) return 1;
    if (confidenceValue <= 75) return 2;
    return 3;
  };

  const activeIndex = getActiveStageIndex();
  const isHealthy = activeIndex === -1;

  // Calcular progreso de la línea conectora
  const getProgressPercentage = () => {
    if (isHealthy) return 0;
    if (activeIndex === -1) return 0;
    return (activeIndex / (stages.length - 1)) * 100;
  };

  const progressPercentage = getProgressPercentage();

  return (
    <div className="severity-timeline">
      <div className="timeline-header">
        <h4 className="timeline-title">
          📈 Progresión de Severidad
        </h4>
        {isHealthy ? (
          <div className="timeline-status healthy">
            <span className="status-icon">✅</span>
            <span className="status-text">Planta Saludable - Sin Progresión</span>
          </div>
        ) : (
          <div className="timeline-status">
            <span className="status-text">
              Nivel Actual: <strong>{severityLevel}</strong>
            </span>
          </div>
        )}
      </div>

      <div className="timeline-container">
        {/* Línea conectora de fondo */}
        <div className="timeline-track">
          <div 
            className="timeline-progress"
            style={{ 
              width: `${progressPercentage}%`,
              backgroundColor: activeIndex >= 0 ? stages[activeIndex].color : '#e5e7eb'
            }}
          />
        </div>

        {/* Etapas del timeline */}
        <div className="timeline-stages">
          {stages.map((stage, index) => {
            const isActive = index === activeIndex;
            const isPassed = index < activeIndex;
            const isFuture = index > activeIndex;

            return (
              <div 
                key={stage.id}
                className={`timeline-stage ${isActive ? 'active' : ''} ${isPassed ? 'passed' : ''} ${isFuture ? 'future' : ''} ${isHealthy ? 'disabled' : ''}`}
              >
                {/* Marcador circular */}
                <div 
                  className="stage-marker"
                  style={{
                    backgroundColor: (isActive || isPassed) && !isHealthy ? stage.color : '#e5e7eb',
                    borderColor: isActive && !isHealthy ? stage.color : '#d1d5db',
                    boxShadow: isActive && !isHealthy ? `0 0 0 4px ${stage.bgColor}` : 'none'
                  }}
                >
                  <span className="stage-icon">{stage.icon}</span>
                </div>

                {/* Información de la etapa */}
                <div className="stage-info">
                  <div 
                    className="stage-label"
                    style={{ color: (isActive || isPassed) && !isHealthy ? stage.color : '#6b7280' }}
                  >
                    {stage.label}
                  </div>
                  <div className="stage-description">
                    {stage.description}
                  </div>
                  <div className="stage-range">
                    {stage.range}
                  </div>
                </div>

                {/* Badge de estado activo */}
                {isActive && !isHealthy && (
                  <div 
                    className="stage-badge"
                    style={{ backgroundColor: stage.color }}
                  >
                    ACTUAL
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Nota explicativa */}
      <div className="timeline-note">
        <span className="note-icon">💡</span>
        <span className="note-text">
          {isHealthy 
            ? 'Esta planta está saludable y no presenta síntomas de enfermedad.'
            : 'Este timeline muestra la progresión típica de la enfermedad detectada. Actúa rápido para prevenir avance.'}
        </span>
      </div>
    </div>
  );
};

export default SeverityTimeline;
