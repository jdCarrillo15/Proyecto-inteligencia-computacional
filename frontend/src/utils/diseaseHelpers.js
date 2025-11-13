import { diseaseEmojis, diseaseInfo, diseaseResources, generalResources } from '../data/diseaseData';

// Obtener emoji de enfermedad
export const getDiseaseEmoji = (diseaseName) => {
  return diseaseEmojis[diseaseName?.toLowerCase()] || '🌱❓';
};

// Verificar si la planta está sana
export const isHealthy = (diseaseName) => {
  return diseaseName?.toLowerCase().includes('healthy');
};

// Obtener estado de salud
export const getHealthStatus = (diseaseName) => {
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

// Obtener nivel de severidad
export const getSeverityLevel = (diseaseName, confidence) => {
  if (isHealthy(diseaseName)) {
    return { level: 'Saludable', color: '#10b981', urgency: 'low' };
  }
  
  const disease = diseaseName?.toLowerCase() || '';
  
  if (disease.includes('late_blight') || disease.includes('black_rot')) {
    return { level: 'Severidad Alta', color: '#dc2626', urgency: 'high' };
  }
  
  if (disease.includes('early_blight') || disease.includes('bacterial') || 
      disease.includes('northern_leaf_blight')) {
    return { level: 'Severidad Media', color: '#f59e0b', urgency: 'medium' };
  }
  
  if (disease.includes('leaf_mold') || disease.includes('scab') || 
      disease.includes('rust') || disease.includes('common_rust')) {
    return { level: 'Severidad Baja', color: '#f97316', urgency: 'low-medium' };
  }
  
  return { level: 'Severidad Media', color: '#f59e0b', urgency: 'medium' };
};

// Obtener información de enfermedad
export const getDiseaseInfo = (diseaseName) => {
  return diseaseInfo[diseaseName?.toLowerCase()] || null;
};

// Obtener tipo de planta
export const getPlantType = (diseaseName) => {
  const disease = diseaseName?.toLowerCase() || '';
  if (disease.includes('apple')) return 'Apple';
  if (disease.includes('corn') || disease.includes('maize')) return 'Corn_(maize)';
  if (disease.includes('potato')) return 'Potato';
  if (disease.includes('tomato')) return 'Tomato';
  return null;
};

// Obtener nombre de clase saludable
export const getHealthyClassName = (plantType) => {
  const healthyMap = {
    'Apple': 'Apple___healthy',
    'Corn_(maize)': 'Corn_(maize)___healthy',
    'Potato': 'Potato___healthy',
    'Tomato': 'Tomato___healthy'
  };
  return healthyMap[plantType];
};

// Obtener recursos externos
export const getResourceLinks = (diseaseName) => {
  const disease = diseaseName?.toLowerCase() || '';
  const specificResources = diseaseResources[disease] || [];
  const allResources = [...specificResources, ...generalResources];
  
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

// Obtener icono de recurso
export const getResourceIcon = (type) => {
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

// Obtener color de confianza
export const getConfidenceColor = (confidence) => {
  if (confidence >= 0.8) return '#10b981';
  if (confidence >= 0.6) return '#f59e0b';
  return '#ef4444';
};
