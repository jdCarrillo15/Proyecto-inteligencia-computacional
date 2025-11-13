# Estructura del Frontend

## 📁 Organización de Carpetas

```
src/
├── components/          # Componentes React reutilizables
│   ├── Header.js       # Encabezado con título y toggle modo oscuro
│   ├── Footer.js       # Pie de página
│   ├── ImageUpload.js  # Componente de carga de imágenes
│   ├── TipsCard.js     # Tarjeta de consejos
│   ├── SystemInfoCard.js  # Tarjeta de información del sistema
│   └── PredictionResults.js  # Resultados de predicción completos
│
├── data/               # Datos y configuraciones
│   ├── config.js       # Configuración de la app (API_URL, etc.)
│   └── diseaseData.js  # Datos de enfermedades, emojis y recursos
│
├── utils/              # Funciones de utilidad
│   ├── api.js          # Llamadas a la API del backend
│   └── diseaseHelpers.js  # Helpers para manejo de enfermedades
│
├── styles/             # Estilos modulares por componente
│   ├── base.css        # Estilos globales, resets, animaciones
│   ├── Header.css      # Estilos del header
│   ├── Footer.css      # Estilos del footer
│   ├── ImageUpload.css # Estilos del componente de upload
│   ├── InfoCards.css   # Estilos de TipsCard y SystemInfoCard
│   └── PredictionResults.css  # Estilos de resultados
│
├── App.js              # Componente principal de la aplicación
├── App.css     # Imports centralizados de estilos modulares
├── index.js            # Punto de entrada de React
└── index.css           # Estilos base
```

## 🧩 Componentes

### `Header.js`
- Encabezado de la aplicación
- Toggle de modo oscuro
- Título y subtítulo

### `Footer.js`
- Pie de página con información del proyecto

### `ImageUpload.js`
- Área de drag & drop para imágenes
- Preview de imagen con zoom
- Input de archivo

### `TipsCard.js`
- Tarjeta con consejos para mejores resultados

### `SystemInfoCard.js`
- Información sobre el sistema de detección

### `PredictionResults.js`
- Muestra resultados de predicción
- Información de enfermedades
- Comparación visual
- Recursos externos

## 📊 Datos

### `config.js`
Configuración de la aplicación:
- `API_URL`: URL del backend
- `MAX_FILE_SIZE`: Tamaño máximo de archivo
- `ACCEPTED_FILE_TYPES`: Tipos de archivo aceptados

### `diseaseData.js`
Datos de enfermedades:
- `diseaseEmojis`: Emojis por enfermedad
- `diseaseInfo`: Información detallada (científica, síntomas, tratamiento)
- `diseaseResources`: Enlaces a recursos externos
- `generalResources`: Recursos generales

## 🛠️ Utilidades

### `api.js`
Funciones para comunicación con backend:
- `predictDisease(file)`: Enviar imagen para predicción

### `diseaseHelpers.js`
Helpers para manejo de enfermedades:
- `getDiseaseEmoji(name)`: Obtener emoji
- `isHealthy(name)`: Verificar si es saludable
- `getHealthStatus(name)`: Estado de salud
- `getSeverityLevel(name, confidence)`: Nivel de severidad
- `getDiseaseInfo(name)`: Información de enfermedad
- `getPlantType(name)`: Tipo de planta
- `getResourceLinks(name)`: Recursos externos
- `getConfidenceColor(confidence)`: Color según confianza

## 🎨 Ventajas de la Modularización

1. **Código más limpio**: Cada componente tiene una responsabilidad única
2. **Fácil mantenimiento**: Cambios aislados en módulos específicos
3. **Reutilización**: Componentes y utilidades reutilizables
4. **Testing**: Más fácil probar componentes individuales
5. **Escalabilidad**: Agregar features sin afectar código existente

## 🔄 Migración

El archivo `App.js` anterior (~1000 líneas) fue modularizado en:
- 6 componentes React
- 2 archivos de datos
- 2 archivos de utilidades
- 1 archivo principal simplificado (~180 líneas)
