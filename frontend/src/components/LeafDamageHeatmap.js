import React from 'react';

/**
 * Componente Heatmap de Área Afectada en Hoja
 * Simulación visual del daño en la hoja usando SVG con gradientes dinámicos
 * 
 * @param {string} diseaseName - Nombre de la enfermedad detectada
 * @param {number} confidence - Porcentaje de confianza (0-100)
 * @returns {JSX.Element} Heatmap visual de daño en hoja
 */
const LeafDamageHeatmap = ({ diseaseName = '', confidence = 0 }) => {
  // Convertir confidence a número (por si viene como string desde la API)
  const confidenceValue = typeof confidence === 'number' ? confidence : parseFloat(confidence) || 0;
  
  // Calcular porcentaje de área afectada según enfermedad
  const calculateAffectedArea = () => {
    const disease = diseaseName.toLowerCase();
    const baseConfidence = confidenceValue / 100;
    
    // Late blight: alta propagación (60-80%)
    if (disease.includes('late_blight') || disease.includes('late blight')) {
      return Math.min(60 + (baseConfidence * 20), 80);
    }
    
    // Black rot: alta propagación (55-75%)
    if (disease.includes('black_rot') || disease.includes('black rot')) {
      return Math.min(55 + (baseConfidence * 20), 75);
    }
    
    // Early blight: propagación media (30-50%)
    if (disease.includes('early_blight') || disease.includes('early blight')) {
      return Math.min(30 + (baseConfidence * 20), 50);
    }
    
    // Northern Leaf Blight: propagación media (35-55%)
    if (disease.includes('northern_leaf_blight') || disease.includes('northern leaf blight')) {
      return Math.min(35 + (baseConfidence * 20), 55);
    }
    
    // Bacterial spot: propagación media (25-45%)
    if (disease.includes('bacterial_spot') || disease.includes('bacterial spot')) {
      return Math.min(25 + (baseConfidence * 20), 45);
    }
    
    // Scab: propagación baja (20-40%)
    if (disease.includes('scab')) {
      return Math.min(20 + (baseConfidence * 20), 40);
    }
    
    // Cedar apple rust: propagación baja (15-35%)
    if (disease.includes('cedar_apple_rust') || disease.includes('cedar apple rust')) {
      return Math.min(15 + (baseConfidence * 20), 35);
    }
    
    // Common rust: propagación baja-media (20-40%)
    if (disease.includes('common_rust') || disease.includes('common rust')) {
      return Math.min(20 + (baseConfidence * 20), 40);
    }
    
    // Leaf Mold: propagación baja (15-35%)
    if (disease.includes('leaf_mold') || disease.includes('leaf mold')) {
      return Math.min(15 + (baseConfidence * 20), 35);
    }
    
    // Por defecto: propagación media (25-45%)
    return Math.min(25 + (baseConfidence * 20), 45);
  };

  const affectedPercentage = calculateAffectedArea();
  
  // Obtener color de severidad
  const getSeverityColor = () => {
    if (affectedPercentage >= 60) return { primary: '#dc2626', secondary: '#991b1b', label: 'Crítico' };
    if (affectedPercentage >= 40) return { primary: '#f97316', secondary: '#c2410c', label: 'Alto' };
    if (affectedPercentage >= 20) return { primary: '#f59e0b', secondary: '#d97706', label: 'Moderado' };
    return { primary: '#10b981', secondary: '#059669', label: 'Leve' };
  };

  const severityColor = getSeverityColor();
  const healthyColor = '#10b981'; // Verde sano
  
  return (
    <div className="leaf-damage-heatmap">
      <div className="heatmap-header">
        <h4 className="heatmap-title">
          🍃 Análisis de Área Afectada
        </h4>
        <div className="damage-stats">
          <div className="stat-item">
            <span className="stat-label">Área Dañada</span>
            <span className="stat-value" style={{ color: severityColor.primary }}>
              {affectedPercentage.toFixed(1)}%
            </span>
          </div>
          <div className="stat-divider"></div>
          <div className="stat-item">
            <span className="stat-label">Severidad</span>
            <span className="stat-badge" style={{ 
              backgroundColor: `${severityColor.primary}20`,
              color: severityColor.primary,
              borderColor: severityColor.primary
            }}>
              {severityColor.label}
            </span>
          </div>
        </div>
      </div>

      <div className="heatmap-visualization">
        <svg
          viewBox="0 0 300 400"
          className="leaf-svg"
          xmlns="http://www.w3.org/2000/svg"
          role="img"
          aria-label={`Hoja con ${affectedPercentage.toFixed(1)}% de área afectada`}
        >
          {/* Definir gradientes */}
          <defs>
            {/* Gradiente de daño (verde a rojo) */}
            <linearGradient id="damageGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop 
                offset="0%" 
                style={{ stopColor: severityColor.primary, stopOpacity: 0.9 }} 
              />
              <stop 
                offset={`${100 - affectedPercentage}%`} 
                style={{ stopColor: severityColor.secondary, stopOpacity: 0.7 }} 
              />
              <stop 
                offset="100%" 
                style={{ stopColor: healthyColor, stopOpacity: 0.95 }} 
              />
            </linearGradient>

            {/* Sombra interior para profundidad */}
            <filter id="innerShadow">
              <feGaussianBlur in="SourceGraphic" stdDeviation="2"/>
              <feOffset dx="1" dy="1" result="offsetblur"/>
              <feFlood floodColor="#000000" floodOpacity="0.2"/>
              <feComposite in2="offsetblur" operator="in"/>
              <feMerge>
                <feMergeNode/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>

            {/* Gradiente para venas */}
            <linearGradient id="veinGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" style={{ stopColor: '#065f46', stopOpacity: 0.3 }} />
              <stop offset="100%" style={{ stopColor: '#064e3b', stopOpacity: 0.5 }} />
            </linearGradient>
          </defs>

          {/* Forma de la hoja */}
          <path
            d="M150 20 Q180 50 190 100 Q200 150 195 200 Q190 250 180 290 Q170 330 150 370 Q130 330 120 290 Q110 250 105 200 Q100 150 110 100 Q120 50 150 20 Z"
            fill="url(#damageGradient)"
            stroke="#047857"
            strokeWidth="2"
            filter="url(#innerShadow)"
            className="leaf-shape"
          />

          {/* Vena central */}
          <path
            d="M150 20 Q150 100 150 200 Q150 300 150 370"
            stroke="url(#veinGradient)"
            strokeWidth="3"
            fill="none"
            strokeLinecap="round"
            className="leaf-vein main-vein"
          />

          {/* Venas secundarias - lado izquierdo */}
          <path
            d="M150 80 Q130 90 120 110"
            stroke="url(#veinGradient)"
            strokeWidth="1.5"
            fill="none"
            opacity="0.6"
            className="leaf-vein"
          />
          <path
            d="M150 140 Q130 150 115 170"
            stroke="url(#veinGradient)"
            strokeWidth="1.5"
            fill="none"
            opacity="0.6"
            className="leaf-vein"
          />
          <path
            d="M150 200 Q130 215 120 240"
            stroke="url(#veinGradient)"
            strokeWidth="1.5"
            fill="none"
            opacity="0.6"
            className="leaf-vein"
          />
          <path
            d="M150 260 Q135 280 125 310"
            stroke="url(#veinGradient)"
            strokeWidth="1.5"
            fill="none"
            opacity="0.6"
            className="leaf-vein"
          />

          {/* Venas secundarias - lado derecho */}
          <path
            d="M150 80 Q170 90 180 110"
            stroke="url(#veinGradient)"
            strokeWidth="1.5"
            fill="none"
            opacity="0.6"
            className="leaf-vein"
          />
          <path
            d="M150 140 Q170 150 185 170"
            stroke="url(#veinGradient)"
            strokeWidth="1.5"
            fill="none"
            opacity="0.6"
            className="leaf-vein"
          />
          <path
            d="M150 200 Q170 215 180 240"
            stroke="url(#veinGradient)"
            strokeWidth="1.5"
            fill="none"
            opacity="0.6"
            className="leaf-vein"
          />
          <path
            d="M150 260 Q165 280 175 310"
            stroke="url(#veinGradient)"
            strokeWidth="1.5"
            fill="none"
            opacity="0.6"
            className="leaf-vein"
          />

          {/* Manchas de enfermedad (simuladas) */}
          {affectedPercentage > 20 && (
            <>
              <ellipse cx="140" cy="60" rx="12" ry="8" 
                fill={severityColor.primary} 
                opacity="0.4"
                className="damage-spot"
              />
              <ellipse cx="170" cy="90" rx="15" ry="10" 
                fill={severityColor.primary} 
                opacity="0.35"
                className="damage-spot"
              />
              <ellipse cx="125" cy="120" rx="10" ry="12" 
                fill={severityColor.primary} 
                opacity="0.4"
                className="damage-spot"
              />
            </>
          )}
          
          {affectedPercentage > 40 && (
            <>
              <ellipse cx="165" cy="150" rx="18" ry="14" 
                fill={severityColor.primary} 
                opacity="0.45"
                className="damage-spot"
              />
              <ellipse cx="135" cy="180" rx="14" ry="11" 
                fill={severityColor.primary} 
                opacity="0.4"
                className="damage-spot"
              />
            </>
          )}
          
          {affectedPercentage > 60 && (
            <>
              <ellipse cx="155" cy="220" rx="20" ry="16" 
                fill={severityColor.secondary} 
                opacity="0.5"
                className="damage-spot"
              />
              <ellipse cx="130" cy="250" rx="16" ry="13" 
                fill={severityColor.secondary} 
                opacity="0.45"
                className="damage-spot"
              />
              <ellipse cx="170" cy="280" rx="14" ry="12" 
                fill={severityColor.secondary} 
                opacity="0.4"
                className="damage-spot"
              />
            </>
          )}
        </svg>

        {/* Leyenda de colores */}
        <div className="heatmap-legend">
          <div className="legend-item">
            <div className="legend-color healthy" style={{ backgroundColor: healthyColor }}></div>
            <span className="legend-label">Tejido Sano</span>
          </div>
          <div className="legend-item">
            <div className="legend-color damaged" style={{ backgroundColor: severityColor.primary }}></div>
            <span className="legend-label">Tejido Afectado</span>
          </div>
        </div>
      </div>

      {/* Información adicional */}
      <div className="heatmap-info">
        <div className="info-card">
          <span className="info-icon">📊</span>
          <div className="info-content">
            <div className="info-label">Patrón de Propagación</div>
            <div className="info-value">
              {affectedPercentage < 20 ? 'Localizado - Fácil de controlar' :
               affectedPercentage < 40 ? 'Moderado - Requiere atención' :
               affectedPercentage < 60 ? 'Extendido - Acción inmediata' :
               'Severo - Tratamiento urgente'}
            </div>
          </div>
        </div>
      </div>

      {/* Nota explicativa */}
      <div className="heatmap-note">
        <span className="note-icon">💡</span>
        <span className="note-text">
          Esta visualización simula el área aproximada afectada por la enfermedad. 
          El gradiente de color representa la progresión del daño desde el punto de infección.
        </span>
      </div>
    </div>
  );
};

export default LeafDamageHeatmap;
