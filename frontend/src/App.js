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
  const [darkMode, setDarkMode] = useState(false);
  const [imageZoomed, setImageZoomed] = useState(false);
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
    
    // Mapeo robusto de recursos específicos por enfermedad
    const resourcesMap = {
      // Manzana (Apple)
      'apple___apple_scab': [
        { title: 'Wikipedia - Sarna del Manzano', url: 'https://en.wikipedia.org/wiki/Apple_scab', type: 'encyclopedia' },
        { title: 'PlantVillage - Apple Scab', url: 'https://plantvillage.psu.edu/topics/apple/infos', type: 'guide' },
        { title: 'Extension - Manejo de Sarna', url: 'https://extension.umn.edu/plant-diseases/apple-scab', type: 'extension' },
        { title: 'EPA - Fungicidas Aprobados', url: 'https://www.epa.gov/pesticides', type: 'official' }
      ],
      'apple___black_rot': [
        { title: 'Wikipedia - Pudrición Negra', url: 'https://en.wikipedia.org/wiki/Black_rot_(apple)', type: 'encyclopedia' },
        { title: 'PlantVillage - Black Rot', url: 'https://plantvillage.psu.edu/topics/apple/infos', type: 'guide' },
        { title: 'Cornell - Black Rot Management', url: 'https://www.cornell.edu/', type: 'extension' }
      ],
      'apple___cedar_apple_rust': [
        { title: 'Wikipedia - Roya del Cedro', url: 'https://en.wikipedia.org/wiki/Gymnosporangium_juniperi-virginianae', type: 'encyclopedia' },
        { title: 'PlantVillage - Cedar Apple Rust', url: 'https://plantvillage.psu.edu/topics/apple/infos', type: 'guide' },
        { title: 'Extension - Control de Roya', url: 'https://extension.umn.edu/plant-diseases/cedar-apple-rust', type: 'extension' }
      ],
      'apple___healthy': [
        { title: 'Guía de Cultivo de Manzanas', url: 'https://extension.umn.edu/fruit/apples', type: 'guide' },
        { title: 'Manejo Integrado de Plagas', url: 'https://www.epa.gov/safepestcontrol/integrated-pest-management-ipm-principles', type: 'official' }
      ],
      
      // Maíz (Corn/Maize)
      'corn_(maize)___common_rust_': [
        { title: 'Wikipedia - Roya Común del Maíz', url: 'https://en.wikipedia.org/wiki/Puccinia_sorghi', type: 'encyclopedia' },
        { title: 'PlantVillage - Common Rust', url: 'https://plantvillage.psu.edu/topics/corn-maize/infos', type: 'guide' },
        { title: 'Extension - Corn Rust Management', url: 'https://extension.umn.edu/corn-pest-management/rust-corn', type: 'extension' },
        { title: 'CIMMYT - Corn Diseases', url: 'https://www.cimmyt.org/', type: 'research' }
      ],
      'corn_(maize)___healthy': [
        { title: 'Guía de Cultivo de Maíz', url: 'https://extension.umn.edu/crop-production/corn', type: 'guide' },
        { title: 'USDA - Corn Production', url: 'https://www.usda.gov/', type: 'official' }
      ],
      'corn_(maize)___northern_leaf_blight': [
        { title: 'Wikipedia - Tizón Foliar del Norte', url: 'https://en.wikipedia.org/wiki/Northern_corn_leaf_blight', type: 'encyclopedia' },
        { title: 'PlantVillage - Northern Leaf Blight', url: 'https://plantvillage.psu.edu/topics/corn-maize/infos', type: 'guide' },
        { title: 'Extension - Blight Control', url: 'https://extension.umn.edu/corn-pest-management/northern-corn-leaf-blight', type: 'extension' },
        { title: 'IPM - Manejo Integrado', url: 'https://www.epa.gov/safepestcontrol/integrated-pest-management-ipm-principles', type: 'official' }
      ],
      
      // Papa (Potato)
      'potato___early_blight': [
        { title: 'Wikipedia - Alternaria (Tizón Temprano)', url: 'https://en.wikipedia.org/wiki/Alternaria_solani', type: 'encyclopedia' },
        { title: 'PlantVillage - Early Blight', url: 'https://plantvillage.psu.edu/topics/potato/infos', type: 'guide' },
        { title: 'Extension - Early Blight Management', url: 'https://extension.umn.edu/diseases/early-blight-potato-and-tomato', type: 'extension' },
        { title: 'CIP - International Potato Center', url: 'https://cipotato.org/', type: 'research' }
      ],
      'potato___healthy': [
        { title: 'Guía de Cultivo de Papa', url: 'https://extension.umn.edu/vegetables/growing-potatoes', type: 'guide' },
        { title: 'CIP - Potato Resources', url: 'https://cipotato.org/', type: 'research' }
      ],
      'potato___late_blight': [
        { title: 'Wikipedia - Phytophthora infestans', url: 'https://en.wikipedia.org/wiki/Phytophthora_infestans', type: 'encyclopedia' },
        { title: 'PlantVillage - Late Blight', url: 'https://plantvillage.psu.edu/topics/potato/infos', type: 'guide' },
        { title: 'Extension - Late Blight Management', url: 'https://extension.umn.edu/diseases/late-blight', type: 'extension' },
        { title: 'CIP - Late Blight Resources', url: 'https://cipotato.org/crops/potato/potato-diseases/late-blight/', type: 'research' },
        { title: 'USAblight - Alerta Temprana', url: 'https://usablight.org/', type: 'tool' }
      ],
      
      // Tomate (Tomato)
      'tomato___bacterial_spot': [
        { title: 'Wikipedia - Mancha Bacteriana', url: 'https://en.wikipedia.org/wiki/Bacterial_leaf_spot', type: 'encyclopedia' },
        { title: 'PlantVillage - Bacterial Spot', url: 'https://plantvillage.psu.edu/topics/tomato/infos', type: 'guide' },
        { title: 'Extension - Bacterial Disease Control', url: 'https://extension.umn.edu/diseases/bacterial-diseases-tomato', type: 'extension' }
      ],
      'tomato___early_blight': [
        { title: 'Wikipedia - Alternaria (Tizón Temprano)', url: 'https://en.wikipedia.org/wiki/Alternaria_solani', type: 'encyclopedia' },
        { title: 'PlantVillage - Early Blight', url: 'https://plantvillage.psu.edu/topics/tomato/infos', type: 'guide' },
        { title: 'Extension - Early Blight in Tomatoes', url: 'https://extension.umn.edu/diseases/early-blight-potato-and-tomato', type: 'extension' }
      ],
      'tomato___healthy': [
        { title: 'Guía de Cultivo de Tomates', url: 'https://extension.umn.edu/vegetables/growing-tomatoes', type: 'guide' },
        { title: 'USDA - Tomato Production', url: 'https://www.usda.gov/', type: 'official' }
      ],
      'tomato___late_blight': [
        { title: 'Wikipedia - Phytophthora infestans', url: 'https://en.wikipedia.org/wiki/Phytophthora_infestans', type: 'encyclopedia' },
        { title: 'PlantVillage - Late Blight', url: 'https://plantvillage.psu.edu/topics/tomato/infos', type: 'guide' },
        { title: 'Extension - Late Blight in Tomatoes', url: 'https://extension.umn.edu/diseases/late-blight', type: 'extension' },
        { title: 'USAblight - Monitoring System', url: 'https://usablight.org/', type: 'tool' }
      ],
      'tomato___leaf_mold': [
        { title: 'Wikipedia - Moho de la Hoja', url: 'https://en.wikipedia.org/wiki/Cladosporium_fulvum', type: 'encyclopedia' },
        { title: 'PlantVillage - Leaf Mold', url: 'https://plantvillage.psu.edu/topics/tomato/infos', type: 'guide' },
        { title: 'Extension - Leaf Mold Management', url: 'https://extension.umn.edu/diseases/leaf-mold-tomato', type: 'extension' }
      ]
    };
    
    // Recursos generales que siempre se incluyen
    const generalResources = [
      { title: 'PlantVillage - Base de Conocimiento', url: 'https://plantvillage.psu.edu/', type: 'general' },
      { title: 'Dataset Kaggle - Plant Disease', url: 'https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset', type: 'data' }
    ];
    
    // Obtener recursos específicos de la enfermedad
    const specificResources = resourcesMap[disease] || [];
    
    // Combinar recursos específicos con generales
    const allResources = [...specificResources, ...generalResources];
    
    // Validar y retornar solo URLs válidas
    return allResources.filter(link => {
      try {
        new URL(link.url);
        return true;
      } catch (e) {
        console.warn(`URL inválida detectada: ${link.url}`);
        return false;
      }
    });
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#10b981';
    if (confidence >= 0.6) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <div className={`App ${darkMode ? 'dark-mode' : ''}`}>
      <div className="container">
        <header className="header" role="banner">
          <div className="header-content">
            <div className="header-text">
              <h1 className="title">
                <span className="emoji">🌱</span>
                Detector de Enfermedades en Plantas
                <span className="emoji">🔬</span>
              </h1>
              <p className="subtitle">
                Sistema de diagnóstico agrícola con CNN | Proyecto académico para fitopatología
              </p>
            </div>
            <button 
              className="dark-mode-toggle"
              onClick={() => setDarkMode(!darkMode)}
              onKeyPress={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setDarkMode(!darkMode);
                }
              }}
              aria-label={darkMode ? 'Cambiar a modo claro' : 'Cambiar a modo oscuro'}
              aria-pressed={darkMode}
              title={darkMode ? 'Cambiar a modo claro' : 'Cambiar a modo oscuro'}
            >
              <span aria-hidden="true">{darkMode ? '☀️' : '🌙'}</span>
              <span className="sr-only">{darkMode ? 'Modo claro' : 'Modo oscuro'}</span>
            </button>
          </div>
        </header>

        <main className="main-content" role="main">
          <section className="upload-section" aria-label="Sección de carga de imagen">
            <div className="card">
              <h2 className="card-title" id="upload-section-title">📤 Subir Imagen</h2>
              
              <form onSubmit={handleSubmit} aria-labelledby="upload-section-title">
                <div
                  className={`upload-area ${dragActive ? 'drag-active' : ''}`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current.click()}
                  role="button"
                  tabIndex={0}
                  aria-label="Área de carga de imagen. Haz clic para seleccionar o arrastra una imagen aquí"
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      fileInputRef.current.click();
                    }
                  }}
                >
                  {preview ? (
                    <div className="preview-container">
                      <img 
                        src={preview} 
                        alt="Vista previa de la imagen de hoja de planta cargada" 
                        className={`preview-image ${imageZoomed ? 'zoomed' : ''}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          setImageZoomed(!imageZoomed);
                        }}
                        onKeyPress={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            e.stopPropagation();
                            setImageZoomed(!imageZoomed);
                          }
                        }}
                        tabIndex={0}
                        role="button"
                        aria-label={imageZoomed ? 'Imagen ampliada. Presiona para alejar' : 'Imagen de preview. Presiona para ampliar'}
                      />
                      <div className="zoom-hint">
                        {imageZoomed ? '👆 Toca para alejar' : '👆 Toca para ampliar'}
                      </div>
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
                    aria-label="Seleccionar imagen de hoja de planta"
                    id="file-input"
                  />
                </div>

                {error && (
                  <div className="alert alert-error" role="alert" aria-live="assertive">
                    <span aria-hidden="true">⚠️</span> {error}
                  </div>
                )}

                <div className="button-group">
                  {selectedFile && (
                    <>
                      <button
                        type="submit"
                        className="btn btn-primary"
                        disabled={loading}
                        aria-label={loading ? 'Analizando imagen' : 'Detectar enfermedad en la planta'}
                        aria-busy={loading}
                      >
                        {loading ? (
                          <span className="loading-content">
                            <span className="scanning-icon" aria-hidden="true">🔍</span>
                            <span className="leaf-icon" aria-hidden="true">🍃</span>
                            Analizando...
                          </span>
                        ) : <><span aria-hidden="true">🔍</span> Detectar Enfermedad</>}
                      </button>
                      <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={handleReset}
                        disabled={loading}
                        aria-label="Limpiar y subir nueva imagen"
                      >
                        <span aria-hidden="true">🔄</span> Nueva Imagen
                      </button>
                    </>
                  )}
                </div>
              </form>
            </div>

            {/* Guía de Usuario */}
            <aside className="info-card tips-card" aria-label="Guía de mejores prácticas">
              <h3><span aria-hidden="true">💡</span> Guía para Mejores Resultados</h3>
              <ul className="tips-list">
                <li className="tip-item">
                  <span className="tip-icon" aria-hidden="true">📸</span>
                  <span className="tip-text">Sube fotos claras de hojas afectadas</span>
                </li>
                <li className="tip-item">
                  <span className="tip-icon" aria-hidden="true">👁️</span>
                  <span className="tip-text">Asegúrate de que los síntomas sean visibles</span>
                </li>
                <li className="tip-item">
                  <span className="tip-icon" aria-hidden="true">☀️</span>
                  <span className="tip-text">Mejor con luz natural (evita flash)</span>
                </li>
                <li className="tip-item">
                  <span className="tip-icon" aria-hidden="true">🎯</span>
                  <span className="tip-text">Evita fondos complejos o distracciones</span>
                </li>
                <li className="tip-item">
                  <span className="tip-icon" aria-hidden="true">🔍</span>
                  <span className="tip-text">Enfoca la hoja completa en el encuadre</span>
                </li>
              </ul>
            </aside>

            {/* Información */}
            <aside className="info-card" aria-label="Información del sistema">
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
          </section>

          <section className="results-section" aria-label="Sección de resultados del diagnóstico">
            {prediction && prediction.success ? (
              <article className="card results-card" role="region" aria-live="polite">
                <h2 className="card-title">✨ Resultado del Diagnóstico</h2>
                
                {/* Estado de Salud Prominente */}
                <div 
                  className={`health-status-banner ${isHealthy(prediction.predicted_class) ? 'healthy-animation' : 'disease-animation'}`}
                  style={{ 
                    backgroundColor: getHealthStatus(prediction.predicted_class).bgColor,
                    borderLeft: `6px solid ${getHealthStatus(prediction.predicted_class).color}`
                  }}
                  role="status"
                  aria-label={`Estado de salud: ${getHealthStatus(prediction.predicted_class).status}`}
                >
                  <span className={`health-icon ${isHealthy(prediction.predicted_class) ? 'checkmark-animation' : 'alert-animation'}`} aria-hidden="true">
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

                <div className="all-predictions" role="list" aria-label="Lista completa de predicciones ordenadas por confianza">
                  <h4 className="predictions-title"><span aria-hidden="true">📊</span> Todas las Predicciones</h4>
                  {prediction.all_predictions.map((pred, index) => (
                    <div 
                      key={index} 
                      className="prediction-item"
                      role="listitem"
                      aria-label={`${pred.class.charAt(0).toUpperCase() + pred.class.slice(1)}: ${pred.percentage}% de confianza`}
                    >
                      <div className="prediction-label">
                        <span className="prediction-emoji" aria-hidden="true">
                          {getDiseaseEmoji(pred.class)}
                        </span>
                        <span className="prediction-class">
                          {pred.class.charAt(0).toUpperCase() + pred.class.slice(1)}
                        </span>
                      </div>
                      <div className="prediction-bar-container" role="progressbar" aria-valuenow={pred.probability * 100} aria-valuemin="0" aria-valuemax="100">
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
                    <h4 className="disease-info-title"><span aria-hidden="true">📋</span> Información de la Enfermedad</h4>
                    
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
                  <section className="comparison-section" aria-label="Comparación visual y recursos adicionales">
                    <button 
                      className="comparison-toggle-btn"
                      onClick={() => setShowComparison(!showComparison)}
                      onKeyPress={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          setShowComparison(!showComparison);
                        }
                      }}
                      aria-expanded={showComparison}
                      aria-controls="comparison-content"
                      aria-label={showComparison ? 'Ocultar comparación visual y recursos' : 'Ver comparación visual y recursos'}
                    >
                      <span aria-hidden="true">{showComparison ? '▼' : '▶'}</span> Ver comparación visual y recursos
                    </button>

                    {showComparison && (
                      <div className="comparison-content" id="comparison-content">
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
                          <h4 className="gallery-title"><span aria-hidden="true">📸</span> Galería de Ejemplos</h4>
                          <div className="gallery-grid">
                            <div className="gallery-item" role="img" aria-label="Ejemplo de enfermedad en estadio inicial">
                              <div className="gallery-placeholder">
                                <span className="gallery-icon" aria-hidden="true">🌿</span>
                                <p>Estadio inicial</p>
                              </div>
                            </div>
                            <div className="gallery-item" role="img" aria-label="Ejemplo de enfermedad en estadio medio">
                              <div className="gallery-placeholder">
                                <span className="gallery-icon" aria-hidden="true">⚠️</span>
                                <p>Estadio medio</p>
                              </div>
                            </div>
                            <div className="gallery-item" role="img" aria-label="Ejemplo de enfermedad en estadio avanzado">
                              <div className="gallery-placeholder">
                                <span className="gallery-icon" aria-hidden="true">🔴</span>
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
                          <p className="resources-subtitle">Fuentes confiables para profundizar en el diagnóstico y manejo</p>
                          <div className="resources-list">
                            {getResourceLinks(prediction.predicted_class).map((link, idx) => {
                              // Seleccionar icono según tipo de recurso
                              const getResourceIcon = (type) => {
                                const iconMap = {
                                  'encyclopedia': '📖',
                                  'guide': '📘',
                                  'extension': '🌾',
                                  'official': '🏛️',
                                  'research': '🔬',
                                  'tool': '🛠️',
                                  'general': '🌐',
                                  'data': '📊'
                                };
                                return iconMap[type] || '🔗';
                              };

                              return (
                                <a 
                                  key={idx}
                                  href={link.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className={`resource-link resource-type-${link.type}`}
                                  aria-label={`${link.title} (abre en nueva pestaña)`}
                                >
                                  <span className="resource-icon" aria-hidden="true">
                                    {getResourceIcon(link.type)}
                                  </span>
                                  <span className="resource-title">{link.title}</span>
                                  <span className="resource-arrow" aria-hidden="true">→</span>
                                </a>
                              );
                            })}
                          </div>
                          {getResourceLinks(prediction.predicted_class).length === 0 && (
                            <div className="no-resources">
                              <p>ℹ️ No hay recursos específicos disponibles para esta clasificación.</p>
                            </div>
                          )}
                          <div className="resources-footer">
                            <p className="resources-note">
                              💡 <strong>Tip:</strong> Estos enlaces te llevan a fuentes académicas y oficiales para información detallada sobre diagnóstico, tratamiento y prevención.
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </section>
                )}
              </article>
            ) : (
              <div className="card placeholder-card" role="status" aria-label="Esperando imagen para diagnóstico">
                <div className="placeholder-content">
                  <div className="placeholder-icon" aria-hidden="true">🎯</div>
                  <h3>Esperando imagen...</h3>
                  <p>Sube una foto de una hoja de planta para comenzar el diagnóstico</p>
                  <div className="supported-plants-title">
                    <h4><span aria-hidden="true">🌱</span> Cultivos Soportados</h4>
                  </div>
                  <div className="supported-fruits">
                    <div className="fruit-chip" title="Mancha negra, Sarna, Roya del cedro, Saludable" aria-label="Manzana: 4 clases de enfermedades soportadas">
                      <span aria-hidden="true">🍎</span> Manzana <span className="chip-count">(4 clases)</span>
                    </div>
                    <div className="fruit-chip" title="Roya común, Tizón del norte, Saludable" aria-label="Maíz: 3 clases de enfermedades soportadas">
                      <span aria-hidden="true">🌽</span> Maíz <span className="chip-count">(3 clases)</span>
                    </div>
                    <div className="fruit-chip" title="Tizón temprano, Tizón tardío, Saludable" aria-label="Papa: 3 clases de enfermedades soportadas">
                      <span aria-hidden="true">🥔</span> Papa <span className="chip-count">(3 clases)</span>
                    </div>
                    <div className="fruit-chip" title="Mancha bacteriana, Tizón temprano, Tizón tardío, Moho de hoja, Saludable" aria-label="Tomate: 5 clases de enfermedades soportadas">
                      <span aria-hidden="true">🍅</span> Tomate <span className="chip-count">(5 clases)</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </section>
        </main>

        <footer className="footer" role="contentinfo">
          <p><span aria-hidden="true">🎓</span> Proyecto Inteligencia Computacional - UPTC</p>
          <p>Sistema de diagnóstico agrícola y fitopatología | Desarrollado con React + TensorFlow</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
