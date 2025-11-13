# Frontend - Detector de Enfermedades en Plantas

Interfaz web en React para el sistema de diagnóstico agrícola.

## Qué hace

Esta es la parte visual del proyecto. Aquí los usuarios pueden subir fotos de hojas, ver los resultados del análisis y consultar información sobre las enfermedades detectadas.

## Características principales

- Arrastrar y soltar imágenes
- Vista previa con zoom (útil en móviles)
- Modo oscuro
- Indicadores visuales de salud de la planta
- Información detallada de enfermedades (síntomas, causas, tratamientos)
- Comparación entre hojas sanas y enfermas
- Tips para tomar mejores fotos
- Diseño responsive
- Accesible con teclado y lectores de pantalla

## Requisitos

- Node.js 14 o más reciente
- El backend corriendo en puerto 5000

## Instalación

```bash
npm install
```

## Uso

Primero asegúrate de que el backend esté corriendo, luego:

```bash
npm start
```

Se abre automáticamente en http://localhost:3000

## Comandos

- `npm start` - Modo desarrollo
- `npm run build` - Versión de producción
- `npm test` - Correr tests
- `npm run eject` - Sacar configuración (no hay vuelta atrás)

## Conexión con el Backend

La app se conecta a `http://localhost:5000` por defecto. Si el backend está en otro puerto, cambia `API_URL` en `src/App.js`.

## Stack

- React 19
- Axios (para llamadas HTTP)
- CSS puro (sin frameworks)
- Create React App

## Funcionalidades

**Carga de imágenes:**
- Arrastrar y soltar
- Click para explorar archivos
- Vista previa
- Solo acepta JPG, JPEG y PNG

**Resultados:**
- Clase predicha con emoji
- Porcentaje de confianza con código de color
- Lista completa de todas las predicciones
- Nivel de gravedad de la enfermedad

**Información adicional:**
- Nombre científico de la enfermedad
- Síntomas principales
- Causas comunes
- Tratamiento recomendado
- Enlaces a recursos externos

## Estructura

```
frontend/
├── public/          # Archivos estáticos
├── src/
│   ├── App.js      # Todo el código React
│   ├── App.css     # Todos los estilos
│   └── index.js    # Entry point
└── package.json
```

## Configuración

Para cambiar el puerto del backend, edita `API_URL` en `App.js`:

```javascript
const API_URL = 'http://localhost:PUERTO';
```

## Accesibilidad

Implementado siguiendo WCAG 2.1 AA:
- Navegación por teclado completa
- Etiquetas ARIA
- Contraste de colores apropiado
- Compatible con lectores de pantalla

## Proyecto Académico

Parte del curso de Inteligencia Computacional - UPTC
