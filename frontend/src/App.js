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

  const getFruitEmoji = (fruitName) => {
    const emojis = {
      'manzana': '🍎',
      'banano': '🍌',
      'mango': '🥭',
      'naranja': '🍊',
      'pera': '🍐',
    };
    return emojis[fruitName.toLowerCase()] || '🍇';
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
            <span className="emoji">🍎</span>
            Clasificador de Frutas
            <span className="emoji">🍌</span>
          </h1>
          <p className="subtitle">
            Inteligencia Artificial con CNN | Sube una imagen para identificar la fruta
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
                        {loading ? '🔄 Analizando...' : '🔍 Clasificar Fruta'}
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
              <h3>ℹ️ Información</h3>
              <ul className="info-list">
                <li>Modelo CNN entrenado con TensorFlow</li>
                <li>Clasifica 5 tipos de frutas</li>
                <li>Precisión del modelo: ~95%</li>
                <li>Tiempo de predicción: &lt;1 segundo</li>
              </ul>
            </div>
          </div>

          <div className="results-section">
            {prediction && prediction.success ? (
              <div className="card results-card">
                <h2 className="card-title">✨ Resultado</h2>
                
                <div className="prediction-result">
                  <div className="fruit-result">
                    <span className="fruit-emoji-large">
                      {getFruitEmoji(prediction.predicted_class)}
                    </span>
                    <h3 className="fruit-name">
                      {prediction.predicted_class.charAt(0).toUpperCase() + 
                       prediction.predicted_class.slice(1)}
                    </h3>
                  </div>

                  <div className="confidence-container">
                    <div className="confidence-label">Confianza</div>
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
                          {getFruitEmoji(pred.class)}
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
                  <p>Sube una foto de una fruta para comenzar la clasificación</p>
                  <div className="supported-fruits">
                    <div className="fruit-chip">🍎 Manzana</div>
                    <div className="fruit-chip">🍌 Banano</div>
                    <div className="fruit-chip">🥭 Mango</div>
                    <div className="fruit-chip">🍊 Naranja</div>
                    <div className="fruit-chip">🍐 Pera</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <footer className="footer">
          <p>🎓 Proyecto Inteligencia Computacional - UPTC</p>
          <p>Desarrollado con React + TensorFlow</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
