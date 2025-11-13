import React, { useState, useRef, useEffect } from 'react';
import './App.css';

// Componentes
import Header from './components/Header';
import Footer from './components/Footer';
import ImageUpload from './components/ImageUpload';
import TipsCard from './components/TipsCard';
import SystemInfoCard from './components/SystemInfoCard';
import PredictionResults from './components/PredictionResults';
import PredictionSkeleton from './components/SkeletonLoaders';

// Utilidades
import { predictDisease } from './utils/api';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [imageZoomed, setImageZoomed] = useState(false);
  const [showSkeleton, setShowSkeleton] = useState(false);
  const fileInputRef = useRef(null);
  const skeletonTimerRef = useRef(null);

  // Efecto para controlar el timing del skeleton (300ms)
  useEffect(() => {
    if (loading) {
      // Iniciar timer de 300ms antes de mostrar skeleton
      skeletonTimerRef.current = setTimeout(() => {
        setShowSkeleton(true);
      }, 300);
    } else {
      // Limpiar timer y ocultar skeleton cuando termine la carga
      if (skeletonTimerRef.current) {
        clearTimeout(skeletonTimerRef.current);
        skeletonTimerRef.current = null;
      }
      setShowSkeleton(false);
    }

    // Cleanup al desmontar
    return () => {
      if (skeletonTimerRef.current) {
        clearTimeout(skeletonTimerRef.current);
      }
    };
  }, [loading]);

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

    try {
      const data = await predictDisease(selectedFile);

      if (data.success) {
        setPrediction(data);
      } else {
        setError(data.error || 'Error al procesar la imagen');
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

  return (
    <div className={`App ${darkMode ? 'dark-mode' : ''}`}>
      <div className="container">
        <Header darkMode={darkMode} onToggleDarkMode={() => setDarkMode(!darkMode)} />
        
        <StatCards />

        <main className="main-content" role="main">
          <section className="upload-section" aria-label="Sección de carga de imagen">
            <div className="card">
              <h2 className="card-title" id="upload-section-title">📤 Subir Imagen</h2>
              
              <form onSubmit={handleSubmit} aria-labelledby="upload-section-title">
                <ImageUpload
                  preview={preview}
                  dragActive={dragActive}
                  imageZoomed={imageZoomed}
                  fileInputRef={fileInputRef}
                  onDrag={handleDrag}
                  onDrop={handleDrop}
                  onFileChange={handleFileChange}
                  onToggleZoom={() => setImageZoomed(!imageZoomed)}
                  onClickUpload={() => fileInputRef.current.click()}
                />

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
                        aria-busy={loading}
                      >
                        {loading ? (
                          <span className="loading-content">
                            <span className="scanning-icon">🔍</span>
                            <span className="leaf-icon">🍃</span>
                            Analizando...
                          </span>
                        ) : <><span>🔍</span> Detectar Enfermedad</>}
                      </button>
                      <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={handleReset}
                        disabled={loading}
                      >
                        <span>🔄</span> Nueva Imagen
                      </button>
                    </>
                  )}
                </div>
              </form>
            </div>

            <TipsCard />
            <SystemInfoCard />
          </section>

          <section className="results-section" aria-label="Sección de resultados del diagnóstico">
            {showSkeleton ? (
              <PredictionSkeleton />
            ) : (
              <PredictionResults prediction={prediction} />
            )}
          </section>
        </main>

        <Footer />
      </div>
    </div>
  );
}

export default App;
