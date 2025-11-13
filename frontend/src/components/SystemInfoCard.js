import React from 'react';

const SystemInfoCard = () => {
  return (
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
    </aside>
  );
};

export default SystemInfoCard;
