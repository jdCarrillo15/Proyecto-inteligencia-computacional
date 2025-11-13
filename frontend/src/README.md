# Arquitectura del Frontend

## Estructura de directorios

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

## Catálogo de componentes

### `Header.js`
- Componente de cabecera de la aplicación
- Control de alternancia de tema (claro/oscuro)
- Presentación de título y descripción

### `Footer.js`
- Componente de pie de página con metadata del proyecto

### `ImageUpload.js`
- Área interactiva con funcionalidad drag & drop
- Sistema de previsualización con zoom
- Selector de archivos

### `TipsCard.js`
- Tarjeta informativa con recomendaciones de uso

### `SystemInfoCard.js`
- Panel informativo sobre capacidades del sistema

### `PredictionResults.js`
- Visualización de resultados de clasificación
- Fichas técnicas de patologías
- Módulo comparativo visual
- Enlaces a recursos bibliográficos

## Módulos de datos

### `config.js`
Parámetros de configuración:
- `API_URL`: Dirección del servidor backend
- `MAX_FILE_SIZE`: Límite de tamaño de archivo
- `ACCEPTED_FILE_TYPES`: Formatos de imagen soportados

### `diseaseData.js`
Base de datos de patologías:
- `diseaseEmojis`: Iconografía asociada a enfermedades
- `diseaseInfo`: Fichas técnicas (nomenclatura, sintomatología, tratamiento)
- `diseaseResources`: Referencias bibliográficas externas
- `generalResources`: Recursos complementarios

## Módulo de utilidades

### `api.js`
Funciones de comunicación con backend:
- `predictDisease(file)`: Envío de imagen para clasificación

### `diseaseHelpers.js`
Funciones auxiliares para gestión de patologías:
- `getDiseaseEmoji(name)`: Obtención de iconografía
- `isHealthy(name)`: Validación de estado saludable
- `getHealthStatus(name)`: Determinación de estado fitosanitario
- `getSeverityLevel(name, confidence)`: Cálculo de nivel de severidad
- `getDiseaseInfo(name)`: Recuperación de ficha técnica
- `getPlantType(name)`: Identificación de especie vegetal
- `getResourceLinks(name)`: Obtención de referencias bibliográficas
- `getConfidenceColor(confidence)`: Asignación de codificación cromática

## Ventajas de la arquitectura modular

1. **Separación de responsabilidades**: Cada componente posee una función específica bien definida
2. **Mantenibilidad mejorada**: Modificaciones localizadas sin impacto en otros módulos
3. **Reutilización de código**: Componentes y utilidades aplicables en múltiples contextos
4. **Testing**: Más fácil probar componentes individuales
5. **Escalabilidad**: Agregar features sin afectar código existente

## 🔄 Migración

El archivo `App.js` anterior (~1000 líneas) fue modularizado en:
- 6 componentes React
- 2 archivos de datos
- 2 archivos de utilidades
- 1 archivo principal simplificado (~180 líneas)
