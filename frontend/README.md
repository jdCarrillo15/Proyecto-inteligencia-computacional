# 🍎 Frontend - Clasificador de Frutas

Frontend moderno desarrollado con React para el clasificador de frutas con Inteligencia Artificial.

## 🚀 Características

- ✨ Interfaz moderna y atractiva con gradientes y animaciones
- 📤 Carga de imágenes mediante drag & drop o selector
- 🔍 Predicción en tiempo real con el modelo CNN
- 📊 Visualización de confianza y todas las predicciones
- 📱 Diseño responsive para móviles y tablets
- 🎨 Emojis de frutas para mejor UX

## 📋 Prerequisitos

- Node.js (versión 14 o superior)
- npm o yarn
- Backend corriendo en `http://localhost:5000`

## 🔧 Instalación

1. Instala las dependencias:
```bash
npm install
```

## 🎯 Uso

1. Asegúrate de que el backend esté corriendo en el puerto 5000

2. Inicia el servidor de desarrollo:
```bash
npm start
```

3. Abre tu navegador en [http://localhost:3000](http://localhost:3000)

## 📦 Scripts Disponibles

- `npm start` - Ejecuta la aplicación en modo desarrollo
- `npm run build` - Crea una versión optimizada para producción
- `npm test` - Ejecuta las pruebas
- `npm run eject` - Expulsa la configuración (irreversible)

## 🌐 Integración con Backend

El frontend se conecta al backend mediante:
- URL base: `http://localhost:5000`
- Endpoint de predicción: `POST /predict`
- CORS habilitado en el backend

## 🎨 Tecnologías Utilizadas

- **React 19** - Framework de JavaScript
- **Axios** - Cliente HTTP
- **CSS3** - Estilos con gradientes y animaciones
- **Create React App** - Configuración inicial

## 📱 Funcionalidades

### Subida de Imágenes
- Arrastra y suelta imágenes
- Click para seleccionar archivo
- Previsualización antes de clasificar
- Validación de formato (JPG, JPEG, PNG)

### Resultados
- Fruta identificada con emoji
- Porcentaje de confianza con color dinámico
- Barra de progreso visual
- Lista completa de predicciones con probabilidades

## 🔧 Configuración

Si el backend corre en un puerto diferente, modifica `API_URL` en `src/App.js`:

```javascript
const API_URL = 'http://localhost:PUERTO';
```

## 🏗️ Estructura del Proyecto

```
frontend/
├── public/
│   ├── index.html
│   └── manifest.json
├── src/
│   ├── App.js          # Componente principal
│   ├── App.css         # Estilos
│   ├── index.js        # Punto de entrada
│   └── index.css       # Estilos globales
└── package.json
```

## 🎓 Proyecto Académico

Desarrollado para el curso de Inteligencia Computacional - UPTC

## 📄 Licencia

Este proyecto es parte de un trabajo académico.
