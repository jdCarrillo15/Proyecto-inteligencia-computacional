import React from 'react';

/**
 * ProgressStepper - Indicador de progreso horizontal con etapas del análisis
 * 
 * Etapas del proceso:
 * 1. Cargando imagen ✓
 * 2. Preprocesando... (spinner)
 * 3. Analizando con CNN... (progress bar)
 * 4. Generando reporte... (casi listo)
 * 
 * @param {Object} props
 * @param {number} props.currentStep - Etapa actual (1-4)
 * @param {number} props.progress - Progreso de la etapa actual (0-100)
 */
const ProgressStepper = ({ currentStep = 1, progress = 0 }) => {
  const steps = [
    {
      id: 1,
      label: 'Cargando imagen',
      icon: '📤',
      description: 'Subiendo archivo'
    },
    {
      id: 2,
      label: 'Preprocesando',
      icon: '⚙️',
      description: 'Ajustando dimensiones'
    },
    {
      id: 3,
      label: 'Analizando con CNN',
      icon: '🧠',
      description: 'Red neuronal trabajando'
    },
    {
      id: 4,
      label: 'Generando reporte',
      icon: '📊',
      description: 'Casi listo'
    }
  ];

  /**
   * Determina el estado de cada paso
   * @param {number} stepId - ID del paso
   * @returns {string} - 'completed', 'in-progress', 'pending'
   */
  const getStepStatus = (stepId) => {
    if (stepId < currentStep) return 'completed';
    if (stepId === currentStep) return 'in-progress';
    return 'pending';
  };

  /**
   * Renderiza el ícono del paso según su estado
   */
  const renderStepIcon = (step) => {
    const status = getStepStatus(step.id);

    if (status === 'completed') {
      return (
        <div className="step-icon completed">
          <span className="checkmark">✓</span>
        </div>
      );
    }

    if (status === 'in-progress') {
      return (
        <div className="step-icon in-progress">
          <span className="spinner"></span>
        </div>
      );
    }

    return (
      <div className="step-icon pending">
        <span className="circle"></span>
      </div>
    );
  };

  return (
    <div className="progress-stepper" role="progressbar" aria-valuenow={currentStep} aria-valuemin="1" aria-valuemax="4">
      <div className="stepper-container">
        {steps.map((step, index) => {
          const status = getStepStatus(step.id);
          const isLast = index === steps.length - 1;

          return (
            <div key={step.id} className="stepper-item-wrapper">
              <div className={`stepper-item ${status}`}>
                {/* Icono del paso */}
                <div className="step-icon-wrapper">
                  {renderStepIcon(step)}
                  <div className="step-emoji">{step.icon}</div>
                </div>

                {/* Contenido del paso */}
                <div className="step-content">
                  <div className="step-label">{step.label}</div>
                  <div className="step-description">{step.description}</div>
                  
                  {/* Progress bar solo para el paso en progreso */}
                  {status === 'in-progress' && step.id === 3 && (
                    <div className="step-progress-bar">
                      <div 
                        className="step-progress-fill"
                        style={{ width: `${progress}%` }}
                        role="progressbar"
                        aria-valuenow={progress}
                        aria-valuemin="0"
                        aria-valuemax="100"
                      >
                        <span className="progress-percentage">{progress}%</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Línea conectora entre pasos */}
              {!isLast && (
                <div className={`stepper-connector ${status === 'completed' ? 'completed' : ''}`}>
                  <div className="connector-line"></div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Barra de progreso global */}
      <div className="global-progress">
        <div className="global-progress-track">
          <div 
            className="global-progress-fill"
            style={{ 
              width: `${((currentStep - 1) / (steps.length - 1)) * 100 + (progress / steps.length)}%` 
            }}
          ></div>
        </div>
        <div className="progress-label">
          Paso {currentStep} de {steps.length} • {Math.round(((currentStep - 1) / steps.length) * 100 + (progress / steps.length))}% completado
        </div>
      </div>
    </div>
  );
};

/**
 * SimpleProgressBar - Barra de progreso simple y reutilizable
 * @param {Object} props
 * @param {number} props.progress - Progreso (0-100)
 * @param {string} props.color - Color de la barra (default: primary)
 * @param {string} props.label - Etiqueta opcional
 * @param {boolean} props.showPercentage - Mostrar porcentaje (default: true)
 */
export const SimpleProgressBar = ({ 
  progress = 0, 
  color = 'primary', 
  label = '', 
  showPercentage = true 
}) => {
  return (
    <div className="simple-progress-bar">
      {label && <div className="progress-bar-label">{label}</div>}
      <div className="progress-bar-track">
        <div 
          className={`progress-bar-fill progress-${color}`}
          style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
          role="progressbar"
          aria-valuenow={progress}
          aria-valuemin="0"
          aria-valuemax="100"
        >
          {showPercentage && progress > 5 && (
            <span className="progress-bar-percentage">{Math.round(progress)}%</span>
          )}
        </div>
      </div>
    </div>
  );
};

/**
 * CircularProgress - Spinner circular para estados de carga
 * @param {Object} props
 * @param {string} props.size - Tamaño ('sm', 'md', 'lg')
 * @param {string} props.color - Color del spinner
 */
export const CircularProgress = ({ size = 'md', color = 'primary' }) => {
  const sizeMap = {
    sm: '16px',
    md: '24px',
    lg: '40px'
  };

  return (
    <div 
      className={`circular-progress circular-progress-${size}`}
      style={{ 
        width: sizeMap[size], 
        height: sizeMap[size],
        borderTopColor: `var(--color-${color})` 
      }}
      role="status"
      aria-label="Cargando..."
    ></div>
  );
};

export default ProgressStepper;
