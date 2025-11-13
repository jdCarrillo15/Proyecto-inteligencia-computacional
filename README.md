# 🍎🍌 Clasificador de Frutas con CNN

Sistema completo de clasificación de frutas usando Inteligencia Artificial con Redes Neuronales Convolucionales (CNN).

## 📝 Descripción

Aplicación web full-stack que permite subir imágenes de frutas y clasificarlas automáticamente usando un modelo de Deep Learning entrenado con TensorFlow. El sistema identifica 5 tipos de frutas diferentes con alta precisión.

## 🎯 Características Principales

- 🤖 **Modelo CNN** entrenado con TensorFlow/Keras
- 🌐 **Backend API REST** con Flask
- ⚛️ **Frontend moderno** con React
- 📤 **Interfaz drag & drop** para subir imágenes
- 📊 **Visualización de confianza** y probabilidades
- 📱 **Diseño responsive** para todos los dispositivos
- ✨ **Animaciones y efectos** visuales atractivos

## 🍇 Frutas Soportadas

- 🍎 **Manzana**
- 🍌 **Banano**
- 🥭 **Mango**
- 🍊 **Naranja**
- 🍐 **Pera**

## 🏗️ Arquitectura

```
Proyecto-inteligencia-computacional/
│
├── backend/              # API REST con Flask
│   ├── app.py           # Aplicación principal
│   ├── requirements.txt # Dependencias Python
│   ├── models/          # Modelos entrenados
│   ├── scripts/         # Scripts de entrenamiento
│   └── utils/           # Utilidades y herramientas
│
└── frontend/            # Aplicación React
    ├── src/
    │   ├── App.js       # Componente principal
    │   └── App.css      # Estilos
    ├── public/          # Archivos públicos
    └── package.json     # Dependencias Node
```

## 🚀 Inicio Rápido

### Prerequisitos

- **Python 3.10+** con pip
- **Node.js 14+** con npm
- **Git** (opcional)

### ⚡ Entrenamiento ULTRA-RÁPIDO con PKL

El sistema utiliza cache PKL para acelerar el entrenamiento:

```bash
# Configuración inicial (solo primera vez)
setup-optimizado.bat

# Entrenamiento completo optimizado
train-fast.bat
# O manualmente:
python backend/scripts/quick_train.py
```

**⏱️ Tiempos de entrenamiento:**
- Primera vez: 15-30 min (procesa y guarda en cache)
- Siguientes veces: 10-20 min (carga desde cache PKL) - **70-90% más rápido**

**📋 Comandos útiles:**

```bash
# Ver información del cache
python backend/utils/manage_cache.py

# Gestionar cache (limpiar, verificar)
python backend/utils/manage_cache.py

# Ver comparativas de rendimiento
python backend/utils/benchmark.py
```

### 1️⃣ Backend (Terminal 1)

```bash
# Navega al backend
cd backend

# Instala dependencias
pip install -r requirements.txt

# Entrena el modelo (RÁPIDO con cache PKL)
python scripts/quick_train.py

# Inicia el servidor
python app.py
```

Backend corriendo en: **http://localhost:5000**

### 2️⃣ Frontend (Terminal 2)

```bash
# Navega al frontend
cd frontend

# Instala dependencias
npm install

# Inicia la aplicación
npm start
```

Frontend corriendo en: **http://localhost:3000**

### 3️⃣ ¡Listo! 🎉

Abre tu navegador en `http://localhost:3000` y comienza a clasificar frutas.

## 🔧 Tecnologías

### Backend
- **Flask 3.0+** - Framework web Python
- **TensorFlow 2.18+** - Machine Learning
- **Keras 3.6+** - API de Deep Learning
- **Flask-CORS** - Manejo de CORS
- **Pillow** - Procesamiento de imágenes
- **NumPy** - Operaciones numéricas

### Frontend
- **React 19** - Framework de JavaScript
- **Axios** - Cliente HTTP
- **CSS3** - Estilos modernos con animaciones

### Machine Learning
- **CNN** - Red Neuronal Convolucional
- **MobileNetV2** - Transfer Learning pre-entrenado
- **Cache PKL** - Sistema de caché para datos procesados
- **sklearn** - División de datos y métricas

## 🚀 Optimizaciones con PKL

El sistema implementa un **cache con archivos PKL (pickle)** que acelera dramáticamente el entrenamiento:

### ✅ Ventajas
- **70-90% más rápido** en re-entrenamientos
- **Carga instantánea** de datos (<30 segundos)
- **Transfer Learning** con MobileNetV2
- **Pipeline automatizado** completo

### 📁 Archivos Generados

```
backend/cache/               # Cache PKL
├── [hash]_train.pkl        # Datos de entrenamiento (12000 muestras)
├── [hash]_test.pkl         # Datos de prueba (3000 muestras)
└── cache_metadata.json     # Metadatos

models/
├── best_model.keras        # Mejor modelo entrenado
├── fruit_classifier.keras  # Modelo final
├── class_mapping.json      # Mapeo de clases
└── visualizations/         # Gráficos de entrenamiento
```

### 🔧 Configuración del Entrenamiento

Edita `backend/scripts/quick_train.py`:

```python
# Ajustar según tu hardware
BATCH_SIZE = 64         # 32 para PCs limitados, 128 para PCs potentes
EPOCHS_PHASE1 = 15      # Entrenamiento inicial
EPOCHS_PHASE2 = 10      # Fine-tuning
USE_TRANSFER_LEARNING = True
DO_FINE_TUNING = True   # Desactivar si hay overfitting
```

## 📊 Rendimiento del Modelo

- **Precisión:** ~50-60% (4 clases: Apple, Corn, Potato, Tomato)
- **Tamaño de entrada:** 100x100 píxeles RGB
- **Tiempo de predicción:** <1 segundo
- **Dataset:** 15,000 imágenes (80% train, 20% test)

## 🎨 Capturas de Pantalla

### Interfaz Principal
- Diseño moderno con gradientes violeta-púrpura
- Área de carga con drag & drop
- Previsualización de imágenes

### Resultados
- Emoji grande de la fruta identificada
- Porcentaje de confianza con colores dinámicos
- Gráfico de todas las predicciones
- Animaciones suaves

## 📡 API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Información de la API |
| GET | `/health` | Estado del servicio |
| POST | `/predict` | Clasificar imagen |
| GET | `/dataset-info` | Info del dataset |

## 🔐 Configuración

### Cambiar Puerto del Backend
En `backend/app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

### Cambiar URL del Backend en Frontend
En `frontend/src/App.js`:
```javascript
const API_URL = 'http://localhost:5000';
```

## 📦 Dependencias Principales

### Backend
```
tensorflow>=2.18.0
keras>=3.6.0
Flask>=3.0.0
flask-cors>=4.0.0
Pillow>=10.0.0
numpy>=1.26.0
```

### Frontend
```
react: ^19.2.0
axios: ^1.13.2
react-scripts: ^5.0.1
```

## 🐛 Solución de Problemas

### Error: Modelo no encontrado
```bash
cd backend
python scripts/train_model.py
```

### Error: CORS
Verifica que `flask-cors` esté instalado en el backend.

### Error: Puerto en uso
Cambia el puerto en la configuración correspondiente.

### Error: react-scripts no encontrado
```bash
cd frontend
npm install react-scripts --save
```

## 🎓 Proyecto Académico

**Universidad:** Universidad Pedagógica y Tecnológica de Colombia (UPTC)  
**Curso:** Inteligencia Computacional  
**Año:** 2024

## 📄 Licencia

Este proyecto es parte de un trabajo académico.

---

**¡Desarrollado para la UPTC!**
