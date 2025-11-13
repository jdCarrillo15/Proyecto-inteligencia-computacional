# Detector de Enfermedades en Plantas

Sistema de diagnóstico agrícola usando redes neuronales convolucionales para identificar enfermedades en cultivos de manzana, maíz, papa y tomate.

## Descripción

Este proyecto es una aplicación web que ayuda a detectar enfermedades en plantas mediante el análisis de imágenes de hojas. Usa un modelo CNN entrenado con TensorFlow para clasificar 15 tipos diferentes de enfermedades en 4 cultivos comunes.

El sistema está pensado como herramienta educativa y de apoyo inicial para agricultores, aunque siempre se recomienda consultar con un agrónomo profesional para tratamientos definitivos.

## Características

- Detección de 15 enfermedades diferentes en 4 cultivos
- Interfaz web sencilla con arrastrar y soltar
- Información detallada sobre cada enfermedad (síntomas, causas, tratamientos)
- Modo oscuro
- Comparación visual entre hojas sanas y enfermas
- Diseño responsive que funciona en móviles
- Accesible (cumple WCAG 2.1 AA)

## Cultivos y Enfermedades Soportadas

**Manzana (4 clases)**
- Sarna del manzano
- Pudrición negra
- Roya del cedro
- Hojas sanas

**Maíz (3 clases)**
- Roya común
- Tizón del norte
- Hojas sanas

**Papa (3 clases)**
- Tizón temprano
- Tizón tardío
- Hojas sanas

**Tomate (5 clases)**
- Mancha bacteriana
- Tizón temprano
- Tizón tardío
- Moho de la hoja
- Hojas sanas

## Estructura del Proyecto

```
Proyecto-inteligencia-computacional/
├── backend/              # API REST en Flask
│   ├── app.py           # Servidor principal
│   ├── requirements.txt # Librerías Python
│   ├── models/          # Modelos entrenados (.keras)
│   ├── scripts/         # Entrenamiento y predicción
│   └── utils/           # Diagnóstico y pruebas
│
├── frontend/            # Interfaz en React
│   ├── src/
│   │   ├── App.js      # Lógica principal
│   │   └── App.css     # Estilos
│   └── package.json    # Dependencias Node
│
└── dataset/            # Imágenes de entrenamiento
    └── raw/
        └── New Plant Diseases Dataset(Augmented)/
```

## Cómo Empezar

Necesitas Python 3.10+ y Node.js 14+ instalados.

### Backend

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
=======
Abre una terminal:

```bash
cd backend
pip install -r requirements.txt

# Entrena el modelo (RÁPIDO con cache PKL)
python scripts/quick_train.py

# Inicia el servidor
=======
python app.py
```

El servidor arranca en http://localhost:5000

### Frontend

Abre otra terminal:

```bash
cd frontend
npm install
npm start
```

La interfaz se abre automáticamente en http://localhost:3000

**Nota:** Si es la primera vez, puede que tengas que entrenar el modelo primero con `python scripts/train_model.py` desde la carpeta backend. Esto puede tardar un rato dependiendo de tu máquina.

## Tecnologías Usadas

**Backend:**
- Flask 3.0 (servidor web)
- TensorFlow 2.18 y Keras 3.6 (modelo de IA)
- Pillow (procesamiento de imágenes)
- Flask-CORS (para conectar con el frontend)

**Frontend:**
- React 19 (interfaz de usuario)
- Axios (llamadas HTTP)
- CSS3 (estilos y animaciones)

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

### Variables de Entorno

**Backend** (`backend/.env`):
```env
FLASK_ENV=development
DEBUG=True
ALLOWED_ORIGINS=http://localhost:3000
```

**Frontend** (`frontend/.env`):
```env
REACT_APP_API_URL=http://localhost:5000
REACT_APP_ENV=development
```

> **Nota:** Los archivos `.env` no se incluyen en Git. Copia `.env.example` a `.env` y ajusta los valores.

### Cambiar Puerto del Backend
En `backend/app.py` o en `backend/.env`:
```python
PORT=5000
```

**Dataset:**
- New Plant Diseases Dataset (Kaggle)
- Más de 15,000 imágenes aumentadas
- 15 clases distribuidas en 4 cultivos


## Sobre el Modelo

El modelo es una CNN entrenada con transfer learning usando arquitecturas preentrenadas. Procesa imágenes de 100x100 píxeles en RGB y da resultados en menos de un segundo.

La precisión varía según la calidad de la foto y las condiciones de iluminación, pero generalmente está por encima del 90% en imágenes claras de hojas individuales.

## Características de la Interfaz

- Paleta de colores verdes (tema agrícola)
- Modo claro y oscuro
- Arrastrar y soltar imágenes
- Vista previa con zoom en móviles
- Indicadores de salud (sana vs enferma) con colores
- Niveles de gravedad para enfermedades
- Información científica de cada enfermedad
- Comparación visual entre hojas sanas y enfermas
- Guía con tips para tomar buenas fotos
- Enlaces a recursos externos (artículos, estudios)

## API

El backend expone estos endpoints:

- `GET /` - Info de la API
- `GET /health` - Verificar que el modelo está cargado
- `POST /predict` - Enviar imagen y recibir predicción
- `GET /dataset-info` - Estadísticas del dataset

Para cambiar puertos o URLs, edita `app.py` en backend y `App.js` en frontend.

## Problemas Comunes

**El modelo no se encuentra:**
```bash
cd backend
python scripts/train_model.py
```
Esto va a tomar un rato la primera vez.

**Error de CORS:**
Asegúrate de tener `flask-cors` instalado. Si no: `pip install flask-cors`

**Puerto ocupado:**
Cambia el puerto en `app.py` (backend) o en `package.json` (frontend).

**Dependencias faltantes:**
Borra las carpetas `node_modules` y `venv`, luego reinstala todo desde cero.

## Dataset

Usamos el "New Plant Diseases Dataset" de Kaggle con imágenes aumentadas. Incluye miles de fotos de hojas con diferentes enfermedades y condiciones de iluminación.

Si quieres usar tu propio dataset, necesitas reorganizar las imágenes en carpetas por clase dentro de `dataset/raw/` y ajustar el script de entrenamiento.

## Accesibilidad

La interfaz cumple con WCAG 2.1 nivel AA:
- Navegación completa por teclado
- Compatible con lectores de pantalla
- Contraste de colores adecuado
- Etiquetas ARIA en todos los elementos

## Contexto Académico

Proyecto desarrollado para la clase de Inteligencia Computacional en la Universidad Pedagógica y Tecnológica de Colombia (UPTC).

El objetivo es aplicar conceptos de CNN y transfer learning en un problema real del sector agrícola, combinando machine learning con desarrollo web full-stack.
