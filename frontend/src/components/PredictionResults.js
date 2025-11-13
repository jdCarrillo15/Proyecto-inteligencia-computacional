import React, { useState } from 'react';
import { 
  getDiseaseEmoji, 
  isHealthy, 
  getHealthStatus, 
  getSeverityLevel,
  getConfidenceColor,
  getDiseaseInfo,
  getPlantType,
  getHealthyClassName,
  getResourceLinks,
  getResourceIcon
} from '../utils/diseaseHelpers';
import ConfidenceRadialChart from './ConfidenceRadialChart';

const PredictionResults = ({ prediction }) => {
  const [showComparison, setShowComparison] = useState(false);

  if (!prediction || !prediction.success) {
    return (
      <div className="card placeholder-card" role="status" aria-label="Esperando imagen para diagnóstico">
        <div className="placeholder-content">
          <div className="placeholder-icon" aria-hidden="true">🎯</div>
          <h3>Esperando imagen...</h3>
          <p>Sube una foto de una hoja de planta para comenzar el diagnóstico</p>
          <div className="supported-plants-title">
            <h4><span aria-hidden="true">🌱</span> Cultivos Soportados</h4>
          </div>
          <div className="supported-fruits">
            <div className="fruit-chip">
              <span aria-hidden="true">🍎</span> Manzana <span className="chip-count">(4 clases)</span>
            </div>
            <div className="fruit-chip">
              <span aria-hidden="true">🌽</span> Maíz <span className="chip-count">(3 clases)</span>
            </div>
            <div className="fruit-chip">
              <span aria-hidden="true">🥔</span> Papa <span className="chip-count">(3 clases)</span>
            </div>
            <div className="fruit-chip">
              <span aria-hidden="true">🍅</span> Tomate <span className="chip-count">(5 clases)</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const diseaseData = getDiseaseInfo(prediction.predicted_class);

  return (
    <article className="card results-card" role="region" aria-live="polite">
      <h2 className="card-title">✨ Resultado del Diagnóstico</h2>
      
      {/* Estado de Salud */}
      <div 
        className={`health-status-banner ${isHealthy(prediction.predicted_class) ? 'healthy-animation' : 'disease-animation'}`}
        style={{ 
          backgroundColor: getHealthStatus(prediction.predicted_class).bgColor,
          borderLeft: `6px solid ${getHealthStatus(prediction.predicted_class).color}`
        }}
      >
        <span className={`health-icon ${isHealthy(prediction.predicted_class) ? 'checkmark-animation' : 'alert-animation'}`}>
          {getHealthStatus(prediction.predicted_class).icon}
        </span>
        <span 
          className="health-text"
          style={{ color: getHealthStatus(prediction.predicted_class).color }}
        >
          {getHealthStatus(prediction.predicted_class).status}
        </span>
      </div>

      {/* Resultado Principal */}
      <div className="prediction-result">
        <div className="fruit-result">
          <span className="fruit-emoji-large">
            {getDiseaseEmoji(prediction.predicted_class)}
          </span>
          <h3 className="fruit-name">
            {prediction.predicted_class.charAt(0).toUpperCase() + 
             prediction.predicted_class.slice(1).replace(/_/g, ' ')}
          </h3>
          
          {!isHealthy(prediction.predicted_class) && (
            <div 
              className="severity-badge"
              style={{ 
                backgroundColor: getSeverityLevel(prediction.predicted_class, prediction.confidence).color + '20',
                color: getSeverityLevel(prediction.predicted_class, prediction.confidence).color,
                border: `2px solid ${getSeverityLevel(prediction.predicted_class, prediction.confidence).color}`
              }}
            >
              <span className="severity-icon">
                {getSeverityLevel(prediction.predicted_class, prediction.confidence).urgency === 'high' ? '🔴' : 
                 getSeverityLevel(prediction.predicted_class, prediction.confidence).urgency === 'medium' ? '🟡' : '🟠'}
              </span>
              {getSeverityLevel(prediction.predicted_class, prediction.confidence).level}
            </div>
          )}
        </div>

        {/* Gráfica Radial de Confianza */}
        <ConfidenceRadialChart 
          confidence={prediction.confidence_percentage}
          size={180}
        />
      </div>

      {/* Todas las Predicciones */}
      <div className="all-predictions">
        <h4 className="predictions-title">📊 Todas las Predicciones</h4>
        {prediction.all_predictions.map((pred, index) => (
          <div key={index} className="prediction-item">
            <div className="prediction-label">
              <span className="prediction-emoji">{getDiseaseEmoji(pred.class)}</span>
              <span className="prediction-class">
                {pred.class.charAt(0).toUpperCase() + pred.class.slice(1)}
              </span>
            </div>
            <div className="prediction-bar-container">
              <div 
                className="prediction-bar"
                style={{ 
                  width: `${pred.probability * 100}%`,
                  backgroundColor: index === 0 ? getConfidenceColor(pred.probability) : '#e5e7eb'
                }}
              />
            </div>
            <div className="prediction-percentage">{pred.percentage}%</div>
          </div>
        ))}
      </div>

      {/* Información de Enfermedad */}
      {!isHealthy(prediction.predicted_class) && diseaseData && (
        <div className="disease-info-card">
          <h4 className="disease-info-title">📋 Información de la Enfermedad</h4>
          
          <div className="disease-info-section">
            <div className="info-label">🔬 Nombre Científico</div>
            <div className="info-value scientific-name">{diseaseData.scientificName}</div>
          </div>

          <div className="disease-info-section">
            <div className="info-label">📝 Descripción</div>
            <div className="info-value">{diseaseData.description}</div>
          </div>

          <div className="disease-info-section">
            <div className="info-label">🔍 Síntomas Principales</div>
            <ul className="symptoms-list">
              {diseaseData.symptoms.map((symptom, idx) => (
                <li key={idx}>{symptom}</li>
              ))}
            </ul>
          </div>

          <div className="disease-info-section">
            <div className="info-label">⚠️ Nivel de Severidad</div>
            <div 
              className="info-value severity-level"
              style={{ 
                color: getSeverityLevel(prediction.predicted_class, prediction.confidence).color,
                fontWeight: '700'
              }}
            >
              {getSeverityLevel(prediction.predicted_class, prediction.confidence).level}
            </div>
          </div>

          <div className="disease-info-section treatment-section">
            <div className="info-label">💊 Tratamiento Recomendado</div>
            <div className="info-value">{diseaseData.treatment}</div>
          </div>

          <div className="disease-info-section">
            <div className="info-label">🛡️ Prevención</div>
            <div className="info-value">{diseaseData.prevention}</div>
          </div>

          <div className="disease-info-footer">
            <p>⚠️ <strong>Nota:</strong> Esta información es orientativa. Consulte con un ingeniero agrónomo para diagnóstico y tratamiento profesional.</p>
          </div>
        </div>
      )}

      {/* Comparación y Recursos */}
      {!isHealthy(prediction.predicted_class) && (
        <section className="comparison-section">
          <button 
            className="comparison-toggle-btn"
            onClick={() => setShowComparison(!showComparison)}
            aria-expanded={showComparison}
          >
            <span>{showComparison ? '▼' : '▶'}</span> Ver comparación visual y recursos
          </button>

          {showComparison && (
            <div className="comparison-content">
              {/* Comparación Visual */}
              <div className="comparison-card">
                <h4 className="comparison-title">🔄 Comparación: Sana vs Enferma</h4>
                
                <div className="feature-status-banner development">
                  <span className="status-icon">🚧</span>
                  <div className="status-content">
                    <strong>Funcionalidad en desarrollo</strong>
                    <p>La galería visual comparativa está en implementación. Mientras tanto, puedes consultar ejemplos visuales en los recursos externos.</p>
                  </div>
                </div>

                <div className="comparison-grid">
                  <div className="comparison-item healthy">
                    <div className="comparison-label healthy-label">✅ Planta Saludable</div>
                    <div className="comparison-placeholder">
                      <span className="plant-emoji-large">
                        {getDiseaseEmoji(getHealthyClassName(getPlantType(prediction.predicted_class)))}
                      </span>
                      <p className="comparison-description">
                        {getPlantType(prediction.predicted_class)?.replace('_', ' ')} sin síntomas de enfermedad
                      </p>
                      <div className="placeholder-info">
                        <p><strong>Características saludables:</strong></p>
                        <ul>
                          <li>Hojas verdes uniformes</li>
                          <li>Sin manchas o decoloraciones</li>
                          <li>Crecimiento vigoroso</li>
                          <li>Sin signos de marchitamiento</li>
                        </ul>
                      </div>
                    </div>
                  </div>

                  <div className="comparison-divider">vs</div>

                  <div className="comparison-item diseased">
                    <div className="comparison-label diseased-label">⚠️ Planta Enferma</div>
                    <div className="comparison-placeholder">
                      <span className="plant-emoji-large">
                        {getDiseaseEmoji(prediction.predicted_class)}
                      </span>
                      <p className="comparison-description">
                        {prediction.predicted_class.replace(/_/g, ' ').split('___')[1]}
                      </p>
                      {diseaseData && (
                        <div className="placeholder-info">
                          <p><strong>Síntomas principales:</strong></p>
                          <ul>
                            {diseaseData.symptoms.slice(0, 4).map((symptom, idx) => (
                              <li key={idx}>{symptom}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                <div className="comparison-alternatives">
                  <h5>🔍 Mientras tanto, puedes ver ejemplos visuales en:</h5>
                  <div className="alternative-links">
                    <a 
                      href="https://plantvillage.psu.edu/" 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="alternative-link"
                    >
                      <span>📘</span> PlantVillage - Atlas de Enfermedades
                    </a>
                    <a 
                      href="https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset" 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="alternative-link"
                    >
                      <span>📊</span> Dataset Kaggle - Imágenes de Entrenamiento
                    </a>
                  </div>
                </div>

                <div className="comparison-note">
                  💡 <strong>Tip:</strong> Compare los síntomas visibles en su cultivo con ejemplos documentados.
                </div>
              </div>

              {/* Dataset Info */}
              <div className="dataset-info-card">
                <h4 className="dataset-title">📊 Información del Dataset</h4>
                <div className="dataset-stats">
                  <div className="stat-item">
                    <span className="stat-icon">🖼️</span>
                    <div className="stat-content">
                      <strong>15,000+</strong>
                      <p>Imágenes totales</p>
                    </div>
                  </div>
                  <div className="stat-item">
                    <span className="stat-icon">🌿</span>
                    <div className="stat-content">
                      <strong>15</strong>
                      <p>Clases de enfermedades</p>
                    </div>
                  </div>
                  <div className="stat-item">
                    <span className="stat-icon">🔬</span>
                    <div className="stat-content">
                      <strong>4</strong>
                      <p>Tipos de cultivos</p>
                    </div>
                  </div>
                  <div className="stat-item">
                    <span className="stat-icon">✅</span>
                    <div className="stat-content">
                      <strong>95%+</strong>
                      <p>Precisión del modelo</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Recursos Externos */}
              <div className="resources-card">
                <h4 className="resources-title">🔗 Recursos Adicionales</h4>
                <p className="resources-subtitle">Fuentes confiables para profundizar en el diagnóstico y manejo</p>
                <div className="resources-list">
                  {getResourceLinks(prediction.predicted_class).map((link, idx) => (
                    <a 
                      key={idx}
                      href={link.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className={`resource-link resource-type-${link.type}`}
                    >
                      <span className="resource-icon">{getResourceIcon(link.type)}</span>
                      <span className="resource-title">{link.title}</span>
                      <span className="resource-arrow">→</span>
                    </a>
                  ))}
                </div>
                <div className="resources-footer">
                  <p className="resources-note">
                    💡 <strong>Tip:</strong> Estos enlaces te llevan a fuentes académicas y oficiales.
                  </p>
                </div>
              </div>
            </div>
          )}
        </section>
      )}
    </article>
  );
};

export default PredictionResults;
