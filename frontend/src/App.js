import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const API_URL = 'http://localhost:5000';

  const handleFileSelect = (file) => {
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
      setError(null);
      setPrediction(null);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    handleFileSelect(file);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!selectedFile) {
      setError('Por favor selecciona una imagen');
      return;
    }

    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setPrediction(response.data);
      } else {
        setError(response.data.error || 'Error al procesar la imagen');
      }
    } catch (err) {
      setError('Error de conexión con el servidor. Asegúrate de que el backend esté ejecutándose.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setPrediction(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getDiseaseEmoji = (diseaseName) => {
    const emojis = {
      // Manzana (Apple)
      'apple___apple_scab': '🍎🟤',
      'apple___black_rot': '🍎⚫',
      'apple___cedar_apple_rust': '🍎🦠',
      'apple___healthy': '🍎🌿',
      
      // Maíz (Corn/Maize)
      'corn_(maize)___common_rust_': '🌽🟤',
      'corn_(maize)___healthy': '🌽🌿',
      'corn_(maize)___northern_leaf_blight': '🌽🍄',
      
      // Papa (Potato)
      'potato___early_blight': '🥔🟤',
      'potato___healthy': '🥔🌿',
      'potato___late_blight': '🥔🍄',
      
      // Tomate (Tomato)
      'tomato___bacterial_spot': '🍅🦠',
      'tomato___early_blight': '🍅🟤',
      'tomato___healthy': '🍅🌿',
      'tomato___late_blight': '🍅🍄',
      'tomato___leaf_mold': '🍅🟢',
    };
    return emojis[diseaseName.toLowerCase()] || '🌱❓';
  };

  const isHealthy = (diseaseName) => {
    return diseaseName.toLowerCase().includes('healthy');
  };

  const getHealthStatus = (diseaseName) => {
    return isHealthy(diseaseName) ? {
      status: 'Planta Sana',
      icon: '✅',
      color: '#10b981',
      bgColor: '#d1fae5'
    } : {
      status: 'Planta Enferma',
      icon: '⚠️',
      color: '#dc2626',
      bgColor: '#fee2e2'
    };
  };

  const getSeverityLevel = (diseaseName, confidence) => {
    if (isHealthy(diseaseName)) {
      return { level: 'Saludable', color: '#10b981', urgency: 'low' };
    }
    
    // Clasificar severidad basada en tipo de enfermedad y confianza
    const disease = diseaseName.toLowerCase();
    
    // Enfermedades más severas (hongos tardíos, pudrición)
    if (disease.includes('late_blight') || disease.includes('black_rot')) {
      return { level: 'Severidad Alta', color: '#dc2626', urgency: 'high' };
    }
    
    // Enfermedades moderadas (hongos tempranos, bacterias)
    if (disease.includes('early_blight') || disease.includes('bacterial') || 
        disease.includes('northern_leaf_blight')) {
      return { level: 'Severidad Media', color: '#f59e0b', urgency: 'medium' };
    }
    
    // Enfermedades leves (moho, manchas, roya)
    if (disease.includes('leaf_mold') || disease.includes('scab') || 
        disease.includes('rust') || disease.includes('common_rust')) {
      return { level: 'Severidad Baja', color: '#f97316', urgency: 'low-medium' };
    }
    
    return { level: 'Severidad Media', color: '#f59e0b', urgency: 'medium' };
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#10b981';
    if (confidence >= 0.6) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1 className="title">
            <span className="emoji">🌱</span>
            Detector de Enfermedades en Plantas
            <span className="emoji">🔬</span>
          </h1>
          <p className="subtitle">
            Sistema de diagnóstico agrícola con CNN | Proyecto académico para fitopatología
          </p>
        </header>

        <div className="main-content">
          <div className="upload-section">
            <div className="card">
              <h2 className="card-title">📤 Subir Imagen</h2>
              
              <form onSubmit={handleSubmit}>
                <div
                  className={`upload-area ${dragActive ? 'drag-active' : ''}`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current.click()}
                >
                  {preview ? (
                    <div className="preview-container">
                      <img src={preview} alt="Preview" className="preview-image" />
                    </div>
                  ) : (
                    <>
                      <div className="upload-icon">📸</div>
                      <div className="upload-text">
                        Arrastra una imagen aquí
                      </div>
                      <div className="upload-subtext">
                        o haz clic para seleccionar
                      </div>
                      <div className="upload-formats">
                        JPG, JPEG, PNG (máx. 16MB)
                      </div>
                    </>
                  )}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/jpeg,image/jpg,image/png"
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                  />
                </div>

                {error && (
                  <div className="alert alert-error">
                    ⚠️ {error}
                  </div>
                )}

                <div className="button-group">
                  {selectedFile && (
                    <>
                      <button
                        type="submit"
                        className="btn btn-primary"
                        disabled={loading}
                      >
                        {loading ? '🔄 Analizando...' : '🔍 Detectar Enfermedad'}
                      </button>
                      <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={handleReset}
                        disabled={loading}
                      >
                        🔄 Nueva Imagen
                      </button>
                    </>
                  )}
                </div>
              </form>
            </div>

            {/* Información */}
            <div className="info-card">
              <h3>ℹ️ Información del Sistema</h3>
              <ul className="info-list">
                <li>Detecta 15 enfermedades en 4 cultivos</li>
                <li>Cultivos: Tomate, Papa, Maíz y Manzana</li>
                <li>Dataset de Kaggle con 15,000+ imágenes</li>
                <li>Modelo CNN con Transfer Learning (TensorFlow)</li>
                <li>Precisión del modelo: ~95%</li>
                <li>Tiempo de predicción: &lt;1 segundo</li>
              </ul>
            </div>
          </div>

          <div className="results-section">
            {prediction && prediction.success ? (
              <div className="card results-card">
                <h2 className="card-title">✨ Resultado del Diagnóstico</h2>
                
                {/* Estado de Salud Prominente */}
                <div 
                  className="health-status-banner"
                  style={{ 
                    backgroundColor: getHealthStatus(prediction.predicted_class).bgColor,
                    borderLeft: `6px solid ${getHealthStatus(prediction.predicted_class).color}`
                  }}
                >
                  <span className="health-icon">
                    {getHealthStatus(prediction.predicted_class).icon}
                  </span>
                  <span 
                    className="health-text"
                    style={{ color: getHealthStatus(prediction.predicted_class).color }}
                  >
                    {getHealthStatus(prediction.predicted_class).status}
                  </span>
                </div>

                <div className="prediction-result">
                  <div className="fruit-result">
                    <span className="fruit-emoji-large">
                      {getDiseaseEmoji(prediction.predicted_class)}
                    </span>
                    <h3 className="fruit-name">
                      {prediction.predicted_class.charAt(0).toUpperCase() + 
                       prediction.predicted_class.slice(1).replace(/_/g, ' ')}
                    </h3>
                    
                    {/* Indicador de Severidad */}
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

                  <div className="confidence-container">
                    <div className="confidence-label">Confianza del Modelo</div>
                    <div 
                      className="confidence-value"
                      style={{ color: getConfidenceColor(prediction.confidence) }}
                    >
                      {prediction.confidence_percentage}%
                    </div>
                    <div className="confidence-bar">
                      <div 
                        className="confidence-fill"
                        style={{ 
                          width: `${prediction.confidence * 100}%`,
                          backgroundColor: getConfidenceColor(prediction.confidence)
                        }}
                      />
                    </div>
                  </div>
                </div>

                <div className="all-predictions">
                  <h4 className="predictions-title">📊 Todas las Predicciones</h4>
                  {prediction.all_predictions.map((pred, index) => (
                    <div key={index} className="prediction-item">
                      <div className="prediction-label">
                        <span className="prediction-emoji">
                          {getDiseaseEmoji(pred.class)}
                        </span>
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
                      <div className="prediction-percentage">
                        {pred.percentage}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="card placeholder-card">
                <div className="placeholder-content">
                  <div className="placeholder-icon">🎯</div>
                  <h3>Esperando imagen...</h3>
                  <p>Sube una foto de una hoja de planta para comenzar el diagnóstico</p>
                  <div className="supported-plants-title">
                    <h4>🌱 Cultivos Soportados</h4>
                  </div>
                  <div className="supported-fruits">
                    <div className="fruit-chip" title="Mancha negra, Sarna, Roya del cedro, Saludable">
                      🍎 Manzana <span className="chip-count">(4 clases)</span>
                    </div>
                    <div className="fruit-chip" title="Roya común, Tizón del norte, Saludable">
                      🌽 Maíz <span className="chip-count">(3 clases)</span>
                    </div>
                    <div className="fruit-chip" title="Tizón temprano, Tizón tardío, Saludable">
                      🥔 Papa <span className="chip-count">(3 clases)</span>
                    </div>
                    <div className="fruit-chip" title="Mancha bacteriana, Tizón temprano, Tizón tardío, Moho de hoja, Saludable">
                      🍅 Tomate <span className="chip-count">(5 clases)</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <footer className="footer">
          <p>🎓 Proyecto Inteligencia Computacional - UPTC</p>
          <p>Sistema de diagnóstico agrícola y fitopatología | Desarrollado con React + TensorFlow</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
