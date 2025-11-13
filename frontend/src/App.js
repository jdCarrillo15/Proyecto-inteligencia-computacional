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
  const [showComparison, setShowComparison] = useState(false);
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

  const getDiseaseInfo = (diseaseName) => {
    const diseaseData = {
      'apple___apple_scab': {
        scientificName: 'Venturia inaequalis',
        description: 'Manchas verde-oliva a marrón en hojas y frutos. Causa defoliación prematura y afecta la calidad de la fruta.',
        symptoms: ['Manchas circulares oscuras', 'Deformación de hojas', 'Lesiones en frutos'],
        treatment: 'Aplicar fungicidas preventivos (captan, mancozeb). Eliminar hojas caídas. Podar para mejorar circulación de aire.',
        prevention: 'Variedades resistentes, manejo sanitario, espaciamiento adecuado'
      },
      'apple___black_rot': {
        scientificName: 'Botryosphaeria obtusa',
        description: 'Pudrición negra que causa manchas foliares, cancros en ramas y pudrición de frutos. Altamente destructiva.',
        symptoms: ['Manchas púrpuras con bordes definidos', 'Frutos momificados', 'Cancros en ramas'],
        treatment: 'Fungicidas sistémicos (myclobutanil, difenoconazole). Podar y destruir tejido infectado. Aplicar en floración.',
        prevention: 'Higiene del huerto, poda sanitaria, eliminar frutos momificados'
      },
      'apple___cedar_apple_rust': {
        scientificName: 'Gymnosporangium juniperi-virginianae',
        description: 'Roya que requiere dos hospederos (manzano y enebro). Causa manchas amarillas-naranjas en hojas.',
        symptoms: ['Manchas amarillas brillantes', 'Pústulas naranjas', 'Defoliación temprana'],
        treatment: 'Fungicidas protectores (mancozeb, ziram). Aplicar desde botón rosa hasta 4 semanas después. Eliminar enebros cercanos.',
        prevention: 'Plantar variedades resistentes, alejar de enebros'
      },
      'corn_(maize)___common_rust_': {
        scientificName: 'Puccinia sorghi',
        description: 'Roya común que forma pústulas café-rojizas en hojas. Reduce fotosíntesis y rendimiento del cultivo.',
        symptoms: ['Pústulas ovales café-rojizas', 'Dispersión en ambas caras de hojas', 'Amarillamiento prematuro'],
        treatment: 'Fungicidas foliares (triazoles, estrobilurinas). Aplicar al detectar primeros síntomas. Rotación de cultivos.',
        prevention: 'Híbridos resistentes, siembra temprana, nutrición balanceada'
      },
      'corn_(maize)___northern_leaf_blight': {
        scientificName: 'Setosphaeria turcica',
        description: 'Tizón foliar que causa lesiones elípticas grises-verdosas. Puede reducir rendimiento hasta 50% en condiciones favorables.',
        symptoms: ['Lesiones alargadas elípticas', 'Color gris-verde a marrón', 'Coalescencia de lesiones'],
        treatment: 'Fungicidas (azoxistrobina, propiconazol). Aplicar preventivamente en zonas endémicas. Manejo de residuos.',
        prevention: 'Variedades resistentes, rotación de cultivos, enterrar residuos'
      },
      'potato___early_blight': {
        scientificName: 'Alternaria solani',
        description: 'Tizón temprano que causa manchas concéntricas en hojas. Común en condiciones cálidas y húmedas.',
        symptoms: ['Manchas circulares con anillos concéntricos', 'Amarillamiento alrededor de manchas', 'Afecta hojas inferiores primero'],
        treatment: 'Fungicidas (clorotalonil, mancozeb, azoxistrobina). Aplicar cada 7-10 días. Fertilización balanceada.',
        prevention: 'Rotación de cultivos, semilla certificada, riego por goteo'
      },
      'potato___late_blight': {
        scientificName: 'Phytophthora infestans',
        description: 'Tizón tardío devastador. Causó la hambruna irlandesa. Puede destruir cultivos en días bajo condiciones favorables.',
        symptoms: ['Lesiones húmedas gris-verdosas', 'Marchitez rápida', 'Pudrición de tubérculos'],
        treatment: 'Fungicidas sistémicos (metalaxil, mandipropamid). Aplicación preventiva obligatoria. Destruir plantas infectadas.',
        prevention: 'Monitoreo constante, variedades resistentes, evitar riego por aspersión nocturno'
      },
      'tomato___bacterial_spot': {
        scientificName: 'Xanthomonas spp.',
        description: 'Mancha bacteriana que afecta hojas, tallos y frutos. Se propaga por agua y herramientas contaminadas.',
        symptoms: ['Manchas pequeñas oscuras con halo amarillo', 'Lesiones en frutos', 'Defoliación severa'],
        treatment: 'Aplicar cobre fijo o bactericidas. Eliminar plantas severamente afectadas. Desinfectar herramientas.',
        prevention: 'Semilla tratada, rotación 3 años, evitar trabajo con plantas mojadas'
      },
      'tomato___early_blight': {
        scientificName: 'Alternaria solani',
        description: 'Tizón temprano con manchas concéntricas características. Afecta hojas maduras primero.',
        symptoms: ['Manchas con anillos concéntricos ("ojo de buey")', 'Hojas inferiores afectadas primero', 'Caída prematura de hojas'],
        treatment: 'Fungicidas (mancozeb, clorotalonil, azoxistrobina). Aplicar preventivamente. Remover hojas basales.',
        prevention: 'Mulching, riego por goteo, espaciamiento adecuado, nutrición balanceada'
      },
      'tomato___late_blight': {
        scientificName: 'Phytophthora infestans',
        description: 'Tizón tardío altamente destructivo. Puede aniquilar plantaciones enteras en 7-10 días.',
        symptoms: ['Lesiones grandes irregulares gris-verdosas', 'Moho blanco en envés', 'Pudrición de frutos'],
        treatment: 'Fungicidas sistémicos urgentes (cymoxanil, metalaxil). Destruir plantas infectadas. Aplicación preventiva crítica.',
        prevention: 'Monitoreo diario, variedades resistentes, plásticos protectores, ventilación'
      },
      'tomato___leaf_mold': {
        scientificName: 'Passalora fulva',
        description: 'Moho de la hoja común en invernaderos. Prospera en alta humedad (>85%) y poca ventilación.',
        symptoms: ['Manchas amarillas en haz', 'Moho verde-oliva en envés', 'Enrollamiento de hojas'],
        treatment: 'Fungicidas (clorotalonil, mancozeb). Mejorar ventilación. Reducir humedad. Eliminar hojas afectadas.',
        prevention: 'Variedades resistentes, ventilación adecuada, control de humedad, espaciamiento'
      }
    };

    return diseaseData[diseaseName.toLowerCase()] || null;
  };

  const getPlantType = (diseaseName) => {
    const disease = diseaseName.toLowerCase();
    if (disease.includes('apple')) return 'Apple';
    if (disease.includes('corn') || disease.includes('maize')) return 'Corn_(maize)';
    if (disease.includes('potato')) return 'Potato';
    if (disease.includes('tomato')) return 'Tomato';
    return null;
  };

  const getHealthyClassName = (plantType) => {
    const healthyMap = {
      'Apple': 'Apple___healthy',
      'Corn_(maize)': 'Corn_(maize)___healthy',
      'Potato': 'Potato___healthy',
      'Tomato': 'Tomato___healthy'
    };
    return healthyMap[plantType];
  };

  const getResourceLinks = (diseaseName) => {
    const disease = diseaseName.toLowerCase();
    const links = [];
    
    // Wikipedia links (educativos)
    if (disease.includes('apple_scab')) {
      links.push({ title: 'Wikipedia - Apple Scab', url: 'https://en.wikipedia.org/wiki/Apple_scab' });
    } else if (disease.includes('black_rot')) {
      links.push({ title: 'Wikipedia - Black Rot', url: 'https://en.wikipedia.org/wiki/Black_rot_(grape)' });
    } else if (disease.includes('late_blight')) {
      links.push({ title: 'Wikipedia - Late Blight', url: 'https://en.wikipedia.org/wiki/Phytophthora_infestans' });
    } else if (disease.includes('early_blight')) {
      links.push({ title: 'Wikipedia - Early Blight', url: 'https://en.wikipedia.org/wiki/Alternaria_solani' });
    }
    
    // Plant Village (recurso general)
    links.push({ title: 'PlantVillage - Base de conocimiento', url: 'https://plantvillage.psu.edu/' });
    
    // Kaggle dataset
    links.push({ title: 'Dataset Kaggle - Plant Disease', url: 'https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset' });
    
    return links;
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

            {/* Guía de Usuario */}
            <div className="info-card tips-card">
              <h3>💡 Guía para Mejores Resultados</h3>
              <ul className="tips-list">
                <li className="tip-item">
                  <span className="tip-icon">📸</span>
                  <span className="tip-text">Sube fotos claras de hojas afectadas</span>
                </li>
                <li className="tip-item">
                  <span className="tip-icon">👁️</span>
                  <span className="tip-text">Asegúrate de que los síntomas sean visibles</span>
                </li>
                <li className="tip-item">
                  <span className="tip-icon">☀️</span>
                  <span className="tip-text">Mejor con luz natural (evita flash)</span>
                </li>
                <li className="tip-item">
                  <span className="tip-icon">🎯</span>
                  <span className="tip-text">Evita fondos complejos o distracciones</span>
                </li>
                <li className="tip-item">
                  <span className="tip-icon">🔍</span>
                  <span className="tip-text">Enfoca la hoja completa en el encuadre</span>
                </li>
              </ul>
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

                {/* Tarjeta de Información de Enfermedad */}
                {!isHealthy(prediction.predicted_class) && getDiseaseInfo(prediction.predicted_class) && (
                  <div className="disease-info-card">
                    <h4 className="disease-info-title">📋 Información de la Enfermedad</h4>
                    
                    <div className="disease-info-section">
                      <div className="info-label">🔬 Nombre Científico</div>
                      <div className="info-value scientific-name">
                        {getDiseaseInfo(prediction.predicted_class).scientificName}
                      </div>
                    </div>

                    <div className="disease-info-section">
                      <div className="info-label">📝 Descripción</div>
                      <div className="info-value">
                        {getDiseaseInfo(prediction.predicted_class).description}
                      </div>
                    </div>

                    <div className="disease-info-section">
                      <div className="info-label">🔍 Síntomas Principales</div>
                      <ul className="symptoms-list">
                        {getDiseaseInfo(prediction.predicted_class).symptoms.map((symptom, idx) => (
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
                      <div className="info-value">
                        {getDiseaseInfo(prediction.predicted_class).treatment}
                      </div>
                    </div>

                    <div className="disease-info-section">
                      <div className="info-label">🛡️ Prevención</div>
                      <div className="info-value">
                        {getDiseaseInfo(prediction.predicted_class).prevention}
                      </div>
                    </div>

                    <div className="disease-info-footer">
                      <p>⚠️ <strong>Nota:</strong> Esta información es orientativa. Consulte con un ingeniero agrónomo para diagnóstico y tratamiento profesional.</p>
                    </div>
                  </div>
                )}

                {/* Comparación Visual y Recursos */}
                {!isHealthy(prediction.predicted_class) && (
                  <div className="comparison-section">
                    <button 
                      className="comparison-toggle-btn"
                      onClick={() => setShowComparison(!showComparison)}
                    >
                      {showComparison ? '▼' : '▶'} Ver comparación visual y recursos
                    </button>

                    {showComparison && (
                      <div className="comparison-content">
                        {/* Comparación Sana vs Enferma */}
                        <div className="comparison-card">
                          <h4 className="comparison-title">🔄 Comparación: Sana vs Enferma</h4>
                          <div className="comparison-grid">
                            <div className="comparison-item healthy">
                              <div className="comparison-label healthy-label">
                                ✅ Planta Saludable
                              </div>
                              <div className="comparison-placeholder">
                                <span className="plant-emoji-large">
                                  {getDiseaseEmoji(getHealthyClassName(getPlantType(prediction.predicted_class)))}
                                </span>
                                <p className="comparison-description">
                                  {getPlantType(prediction.predicted_class)?.replace('_', ' ')} sin síntomas de enfermedad
                                </p>
                              </div>
                            </div>

                            <div className="comparison-divider">vs</div>

                            <div className="comparison-item diseased">
                              <div className="comparison-label diseased-label">
                                ⚠️ Planta Enferma
                              </div>
                              <div className="comparison-placeholder">
                                <span className="plant-emoji-large">
                                  {getDiseaseEmoji(prediction.predicted_class)}
                                </span>
                                <p className="comparison-description">
                                  {prediction.predicted_class.replace(/_/g, ' ').split('___')[1]}
                                </p>
                              </div>
                            </div>
                          </div>
                          <div className="comparison-note">
                            💡 <strong>Tip:</strong> Compare los síntomas visibles en su cultivo con ejemplos documentados para confirmar el diagnóstico.
                          </div>
                        </div>

                        {/* Galería de Ejemplos */}
                        <div className="gallery-card">
                          <h4 className="gallery-title">📸 Galería de Ejemplos</h4>
                          <div className="gallery-grid">
                            <div className="gallery-item">
                              <div className="gallery-placeholder">
                                <span className="gallery-icon">🌿</span>
                                <p>Estadio inicial</p>
                              </div>
                            </div>
                            <div className="gallery-item">
                              <div className="gallery-placeholder">
                                <span className="gallery-icon">⚠️</span>
                                <p>Estadio medio</p>
                              </div>
                            </div>
                            <div className="gallery-item">
                              <div className="gallery-placeholder">
                                <span className="gallery-icon">🔴</span>
                                <p>Estadio avanzado</p>
                              </div>
                            </div>
                          </div>
                          <p className="gallery-note">
                            📚 Las imágenes de ejemplo están disponibles en el dataset de entrenamiento (15,000+ imágenes)
                          </p>
                        </div>

                        {/* Recursos Externos */}
                        <div className="resources-card">
                          <h4 className="resources-title">🔗 Recursos Adicionales</h4>
                          <div className="resources-list">
                            {getResourceLinks(prediction.predicted_class).map((link, idx) => (
                              <a 
                                key={idx}
                                href={link.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="resource-link"
                              >
                                <span className="resource-icon">🔗</span>
                                <span className="resource-title">{link.title}</span>
                                <span className="resource-arrow">→</span>
                              </a>
                            ))}
                          </div>
                          <div className="learn-more">
                            <button className="learn-more-btn">
                              📖 Ver más sobre esta enfermedad
                            </button>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
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
