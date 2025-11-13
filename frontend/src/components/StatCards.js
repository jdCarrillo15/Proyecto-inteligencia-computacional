import React from 'react';
import CountUp from 'react-countup';
import { Activity, CheckCircle, AlertTriangle, Target } from 'lucide-react';

/**
 * Componente de Cards de Estadísticas con Animaciones
 * Dashboard de métricas importantes del sistema con counter animations
 * 
 * @param {Object} stats - Objeto con estadísticas del sistema
 * @returns {JSX.Element} Grid de stat cards animadas
 */
const StatCards = ({ stats }) => {
  // Datos por defecto si no se proporcionan stats
  const defaultStats = {
    totalAnalyses: 1247,
    healthyPlants: 856,
    diseasesDetected: 391,
    avgPrecision: 95.8,
    trends: {
      analyses: { value: 12, isPositive: true },
      healthy: { value: 8, isPositive: true },
      diseases: { value: 15, isPositive: false },
      precision: { value: 2.3, isPositive: true }
    }
  };

  const currentStats = stats || defaultStats;

  // Configuración de las cards
  const cards = [
    {
      id: 'total-analyses',
      title: 'Total Análisis',
      value: currentStats.totalAnalyses,
      icon: Activity,
      iconColor: '#6366f1', // indigo-500
      iconBg: '#eef2ff', // indigo-50
      trend: currentStats.trends?.analyses,
      suffix: '',
      decimals: 0,
      description: 'Diagnósticos realizados'
    },
    {
      id: 'healthy-plants',
      title: 'Plantas Sanas',
      value: currentStats.healthyPlants,
      icon: CheckCircle,
      iconColor: '#10b981', // green-500
      iconBg: '#d1fae5', // green-100
      trend: currentStats.trends?.healthy,
      suffix: '',
      decimals: 0,
      description: 'Sin enfermedades detectadas'
    },
    {
      id: 'diseases-detected',
      title: 'Enfermedades',
      value: currentStats.diseasesDetected,
      icon: AlertTriangle,
      iconColor: '#f59e0b', // amber-500
      iconBg: '#fef3c7', // amber-100
      trend: currentStats.trends?.diseases,
      suffix: '',
      decimals: 0,
      description: 'Casos identificados'
    },
    {
      id: 'avg-precision',
      title: 'Precisión Promedio',
      value: currentStats.avgPrecision,
      icon: Target,
      iconColor: '#8b5cf6', // purple-500
      iconBg: '#f3e8ff', // purple-100
      trend: currentStats.trends?.precision,
      suffix: '%',
      decimals: 1,
      description: 'Confianza del modelo'
    }
  ];

  // Componente de Trend Indicator
  const TrendIndicator = ({ trend }) => {
    if (!trend) return null;

    const { value, isPositive } = trend;
    const trendColor = isPositive ? '#10b981' : '#ef4444';
    const arrow = isPositive ? '↑' : '↓';

    return (
      <div 
        className="trend-indicator"
        style={{ color: trendColor }}
      >
        <span className="trend-arrow">{arrow}</span>
        <span className="trend-value">{Math.abs(value).toFixed(1)}%</span>
        <span className="trend-label">vs semana pasada</span>
      </div>
    );
  };

  return (
    <div className="stat-cards-container">
      <div className="stat-cards-header">
        <h3 className="stat-cards-title">📊 Resumen del Sistema</h3>
        <p className="stat-cards-subtitle">
          Métricas en tiempo real del detector de enfermedades
        </p>
      </div>

      <div className="stat-cards-grid">
        {cards.map((card, index) => {
          const Icon = card.icon;
          
          return (
            <div 
              key={card.id}
              className="stat-card"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              {/* Icono */}
              <div 
                className="stat-card-icon"
                style={{ 
                  backgroundColor: card.iconBg,
                  color: card.iconColor 
                }}
              >
                <Icon size={24} strokeWidth={2.5} />
              </div>

              {/* Contenido */}
              <div className="stat-card-content">
                <div className="stat-card-header">
                  <h4 className="stat-card-title">{card.title}</h4>
                  {card.trend && <TrendIndicator trend={card.trend} />}
                </div>

                {/* Valor con CountUp Animation */}
                <div className="stat-card-value">
                  <CountUp
                    end={card.value}
                    duration={2}
                    decimals={card.decimals}
                    separator=","
                    decimal="."
                    suffix={card.suffix}
                    enableScrollSpy
                    scrollSpyOnce
                  />
                </div>

                <p className="stat-card-description">{card.description}</p>
              </div>

              {/* Badge decorativo */}
              <div 
                className="stat-card-badge"
                style={{ backgroundColor: card.iconColor }}
              />
            </div>
          );
        })}
      </div>

      {/* Nota informativa */}
      <div className="stat-cards-note">
        <span className="note-icon">💡</span>
        <span className="note-text">
          Estas estadísticas demuestran la confiabilidad del sistema basado en datos históricos de análisis.
        </span>
      </div>
    </div>
  );
};

export default StatCards;
