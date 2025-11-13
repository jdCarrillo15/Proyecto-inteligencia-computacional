import React from 'react';
import { RadialBarChart, RadialBar, PolarAngleAxis } from 'recharts';

/**
 * Componente de gráfica radial para mostrar el nivel de confianza
 * Utiliza Recharts para una visualización moderna e impactante
 * 
 * @param {number} confidence - Porcentaje de confianza (0-100)
 * @param {number} size - Tamaño del gráfico en píxeles (default: 160)
 * @returns {JSX.Element} Gráfica radial de confianza
 */
const ConfidenceRadialChart = ({ confidence = 0, size = 160 }) => {
  // Determinar el color basado en el nivel de confianza
  const getConfidenceColor = (value) => {
    if (value >= 80) {
      return {
        fill: 'var(--success)',
        label: 'Alta',
        gradient: ['var(--success)', 'var(--success-dark)']
      };
    } else if (value >= 60) {
      return {
        fill: 'var(--warning)',
        label: 'Media',
        gradient: ['var(--warning)', 'var(--warning-dark)']
      };
    } else {
      return {
        fill: 'var(--danger)',
        label: 'Baja',
        gradient: ['var(--danger)', 'var(--danger-dark)']
      };
    }
  };

  const colorConfig = getConfidenceColor(confidence);
  
  // Datos para el gráfico radial
  const data = [
    {
      name: 'Confianza',
      value: confidence,
      fill: colorConfig.fill
    }
  ];

  // Dimensiones del gráfico
  const outerRadius = size / 2 - 10;
  const innerRadius = outerRadius - 20; // Grosor de 20px

  return (
    <div className="confidence-radial-chart">
      <div className="radial-chart-container">
        <RadialBarChart
          width={size}
          height={size}
          cx={size / 2}
          cy={size / 2}
          innerRadius={innerRadius}
          outerRadius={outerRadius}
          barSize={20}
          data={data}
          startAngle={90}
          endAngle={-270}
        >
          <PolarAngleAxis
            type="number"
            domain={[0, 100]}
            angleAxisId={0}
            tick={false}
          />
          <RadialBar
            background
            dataKey="value"
            cornerRadius={10}
            fill={colorConfig.fill}
            isAnimationActive={true}
            animationDuration={800}
            animationBegin={0}
            animationEasing="ease-out"
          />
        </RadialBarChart>
        
        {/* Texto central con porcentaje y label */}
        <div className="radial-chart-center">
          <div 
            className="confidence-percentage"
            style={{ color: colorConfig.fill }}
          >
            {confidence.toFixed(1)}%
          </div>
          <div className="confidence-label">
            Confianza
          </div>
          <div 
            className="confidence-level"
            style={{ color: colorConfig.fill }}
          >
            {colorConfig.label}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConfidenceRadialChart;
