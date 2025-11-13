import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer } from 'recharts';
import { getDiseaseEmoji } from '../utils/diseaseHelpers';

/**
 * Componente de gráfica de barras horizontales para el Top 5 de predicciones
 * Muestra comparativa visual de todas las predicciones del modelo
 * 
 * @param {Array} predictions - Array de predicciones con class, probability, percentage
 * @param {number} maxPredictions - Número máximo de predicciones a mostrar (default: 5)
 * @returns {JSX.Element} Bar chart horizontal de predicciones
 */
const TopPredictionsChart = ({ predictions = [], maxPredictions = 5 }) => {
  // Limitar a top N predicciones
  const topPredictions = predictions.slice(0, maxPredictions);

  // Formatear datos para Recharts
  const chartData = topPredictions.map((pred, index) => ({
    name: pred.class,
    displayName: truncateText(formatClassName(pred.class), 20),
    emoji: getDiseaseEmoji(pred.class),
    value: pred.probability * 100,
    percentage: pred.percentage,
    isPrimary: index === 0,
    fullName: formatClassName(pred.class)
  }));

  // Truncar texto largo para labels
  function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength - 3) + '...';
  }

  // Formatear nombre de clase (capitalizar y reemplazar guiones bajos)
  function formatClassName(className) {
    return className
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }

  // Tooltip personalizado con información completa
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="top-predictions-tooltip">
          <div className="tooltip-header">
            <span className="tooltip-emoji">{data.emoji}</span>
            <span className="tooltip-title">{data.fullName}</span>
          </div>
          <div className="tooltip-value">
            Probabilidad: <strong>{data.value.toFixed(2)}%</strong>
          </div>
          {data.isPrimary && (
            <div className="tooltip-badge">
              🏆 Predicción Principal
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  // Label personalizado con emoji y nombre truncado
  const CustomYAxisTick = ({ x, y, payload }) => {
    const data = chartData.find(item => item.name === payload.value);
    if (!data) return null;

    return (
      <g transform={`translate(${x},${y})`}>
        <text
          x={-10}
          y={0}
          dy={4}
          textAnchor="end"
          fill="var(--text-secondary)"
          fontSize="13px"
          fontWeight={data.isPrimary ? '600' : '400'}
          fontFamily="var(--font-body)"
        >
          <tspan fontFamily="Segoe UI Emoji, Apple Color Emoji, Noto Color Emoji">{data.emoji} </tspan>
          <tspan>{data.displayName}</tspan>
        </text>
      </g>
    );
  };

  // Altura dinámica basada en cantidad de predicciones
  const chartHeight = Math.max(250, topPredictions.length * 60);

  return (
    <div className="top-predictions-chart">
      <h4 className="predictions-chart-title">
        📊 Top {topPredictions.length} Predicciones
      </h4>
      
      <ResponsiveContainer width="100%" height={chartHeight}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 10, right: 30, left: 150, bottom: 10 }}
        >
          <XAxis 
            type="number" 
            domain={[0, 100]}
            tick={{ 
              fill: 'var(--text-secondary)', 
              fontSize: 12,
              fontFamily: 'var(--font-body)'
            }}
            axisLine={{ stroke: 'var(--border-light)' }}
            tickLine={{ stroke: 'var(--border-light)' }}
            label={{ 
              value: 'Probabilidad (%)', 
              position: 'insideBottom',
              offset: -5,
              style: {
                fill: 'var(--text-secondary)',
                fontSize: 12,
                fontWeight: 600,
                fontFamily: 'var(--font-body)'
              }
            }}
          />
          <YAxis 
            type="category" 
            dataKey="name"
            tick={<CustomYAxisTick />}
            axisLine={{ stroke: 'var(--border-light)' }}
            tickLine={false}
            width={140}
          />
          <Tooltip 
            content={<CustomTooltip />}
            cursor={{ fill: 'var(--bg-tertiary)', opacity: 0.3 }}
          />
          <Bar 
            dataKey="value" 
            radius={[0, 8, 8, 0]}
            animationDuration={800}
            animationEasing="ease-out"
          >
            {chartData.map((entry, index) => (
              <Cell 
                key={`cell-${index}`}
                fill={entry.isPrimary ? 'var(--primary)' : '#e5e7eb'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="predictions-chart-note">
        💡 La barra destacada representa la predicción principal del modelo
      </div>
    </div>
  );
};

export default TopPredictionsChart;
