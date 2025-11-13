import React from 'react';

const ImageUpload = ({ 
  preview, 
  dragActive, 
  imageZoomed,
  fileInputRef,
  onDrag,
  onDrop,
  onFileChange,
  onToggleZoom,
  onClickUpload 
}) => {
  return (
    <div
      className={`upload-area ${dragActive ? 'drag-active' : ''}`}
      onDragEnter={onDrag}
      onDragLeave={onDrag}
      onDragOver={onDrag}
      onDrop={onDrop}
      onClick={onClickUpload}
      role="button"
      tabIndex={0}
      aria-label="Área de carga de imagen. Haz clic para seleccionar o arrastra una imagen aquí"
      onKeyPress={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClickUpload();
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
              onToggleZoom();
            }}
            onKeyPress={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                e.stopPropagation();
                onToggleZoom();
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
        onChange={onFileChange}
        style={{ display: 'none' }}
        aria-label="Seleccionar imagen de hoja de planta"
        id="file-input"
      />
    </div>
  );
};

export default ImageUpload;
