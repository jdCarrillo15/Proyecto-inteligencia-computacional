# Detector de Enfermedades en Plantas

Herramienta de diagnóstico agrícola basada en redes neuronales convolucionales que identifica enfermedades en cultivos de manzana, maíz, papa y tomate mediante análisis visual.

## Descripción

Aplicación web desarrollada para facilitar la detección temprana de enfermedades en plantas a través del análisis de imágenes. El sistema procesa fotografías de hojas y utiliza un modelo CNN entrenado con TensorFlow para clasificar entre 15 tipos de enfermedades distribuidas en 4 cultivos.

Este proyecto surge como respuesta a la necesidad de herramientas accesibles que apoyen a agricultores en la identificación preliminar de problemas fitosanitarios. Si bien proporciona resultados precisos, recomendamos validar cualquier diagnóstico con un especialista agrónomo antes de aplicar tratamientos.

## Características principales

- Clasificación de 15 enfermedades en 4 tipos de cultivos
- Interfaz intuitiva con funcionalidad drag & drop
- Base de datos completa con síntomas, causas y tratamientos recomendados
- Modo oscuro para reducir fatiga visual
- Comparativa visual entre tejido vegetal sano y afectado
- Diseño adaptable a dispositivos móviles
- Interfaz accesible según estándares WCAG 2.1 AA

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

## Instalación y configuración

Requisitos: Python 3.10+ y Node.js 14+

### Backend

### Optimización del entrenamiento con caché PKL

El sistema implementa un mecanismo de caché basado en archivos PKL que reduce significativamente los tiempos de entrenamiento:

```bash
# Primera configuración (ejecutar una sola vez)
setup-optimizado.bat

# Entrenamiento con optimizaciones
train-fast.bat
# Alternativa manual:
python backend/scripts/quick_train.py
```

**Tiempos estimados:**
- Primera ejecución: 15-30 min (procesamiento inicial y generación de caché)
- Ejecuciones posteriores: 10-20 min (carga desde caché PKL, reducción del 70-90%)

**Gestión del sistema de caché:**

```bash
# Consultar estado del caché
python backend/utils/manage_cache.py

# Operaciones de mantenimiento (limpieza, verificación)
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

En una terminal independiente:

```bash
cd frontend
npm install
npm start
```

La aplicación iniciará automáticamente en http://localhost:3000

**Importante:** En el primer uso, es necesario entrenar el modelo ejecutando `python scripts/train_model.py` desde el directorio backend. El tiempo de entrenamiento varía según las especificaciones del hardware.

## Stack tecnológico

**Backend:**
- Flask 3.0 - Framework web
- TensorFlow 2.18 y Keras 3.6 - Desarrollo del modelo de aprendizaje profundo
- Pillow - Procesamiento y manipulación de imágenes
- Flask-CORS - Gestión de Cross-Origin Resource Sharing

**Frontend:**
- React 19 - Biblioteca para construcción de interfaces
- Axios - Cliente HTTP para peticiones asíncronas
- CSS3 - Hojas de estilo con transiciones y animaciones

**Machine Learning:**
- CNN (Convolutional Neural Networks) - Arquitectura de red neuronal
- MobileNetV2 - Modelo preentrenado para transfer learning
- Sistema de caché PKL - Almacenamiento eficiente de datos preprocesados
- scikit-learn - Utilidades para partición de datos y métricas de evaluación

## Optimización del rendimiento mediante PKL

Implementación de sistema de caché basado en serialización pickle que mejora sustancialmente los tiempos de entrenamiento:

### Ventajas del sistema
- Reducción del 70-90% en tiempo de re-entrenamiento
- Carga de datos en menos de 30 segundos
- Integración con transfer learning (MobileNetV2)
- Pipeline de procesamiento completamente automatizado

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

## Métricas de rendimiento

- **Precisión del modelo:** 50-60% (clasificación entre 4 clases principales)
- **Dimensiones de entrada:** Imágenes RGB de 100x100 píxeles
- **Latencia de inferencia:** Inferior a 1 segundo
- **Conjunto de datos:** 15,000 imágenes (partición 80/20 entrenamiento/prueba)

## Interfaz de usuario

### Vista principal
- Diseño contemporáneo con paleta de colores violeta-púrpura
- Zona de carga con funcionalidad arrastrar y soltar
- Sistema de previsualización de imágenes

### Panel de resultados
- Identificación visual del cultivo analizado
- Nivel de confianza con codificación cromática dinámica
- Visualización gráfica de todas las predicciones
- Transiciones fluidas entre estados

## Endpoints disponibles

| Método | Ruta | Funcionalidad |
|--------|----------|-------------|
| GET | `/` | Metadata de la API |
| GET | `/health` | Verificación de disponibilidad del servicio |
| POST | `/predict` | Clasificación de imagen mediante modelo CNN |
| GET | `/dataset-info` | Estadísticas del conjunto de datos |

## Configuración del entorno

### Variables de entorno

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

**Nota de seguridad:** Los archivos `.env` están excluidos del control de versiones. Duplicar `.env.example` como `.env` y configurar según el entorno.

### Configuración de puertos
Modificar en `backend/app.py` o `backend/.env`:
```python
PORT=5000
```

**Dataset utilizado:**
- New Plant Diseases Dataset (disponible en Kaggle)
- Colección de 15,000+ imágenes con aumentación de datos
- 15 clases patológicas distribuidas en 4 especies vegetales


## Arquitectura del modelo

Red neuronal convolucional desarrollada mediante transfer learning sobre arquitecturas preentrenadas. El modelo procesa entradas de 100x100 píxeles en formato RGB con tiempo de inferencia inferior al segundo.

La precisión obtenida varía en función de factores como calidad fotográfica, condiciones de iluminación y nitidez. En condiciones óptimas (iluminación uniforme, hojas individuales, enfoque nítido), el modelo alcanza tasas de precisión superiores al 90%.

## Funcionalidades de la interfaz

- Esquema cromático verde adaptado al contexto agrícola
- Alternancia entre modo claro y oscuro
- Sistema de carga mediante arrastrar y soltar
- Previsualización con zoom optimizada para dispositivos móviles
- Indicadores visuales de estado fitosanitario con codificación cromática
- Clasificación por niveles de severidad patológica
- Fichas técnicas con información científica de cada enfermedad
- Módulo comparativo entre tejido sano y afectado
- Guía de buenas prácticas para captura fotográfica
- Referencias a bibliografía especializada y estudios científicos

## Documentación de la API

El servidor backend proporciona los siguientes endpoints:

- `GET /` - Información general de la API
- `GET /health` - Verificación del estado del modelo
- `POST /predict` - Envío de imagen para clasificación
- `GET /dataset-info` - Metadata y estadísticas del conjunto de datos

La configuración de puertos y URLs se gestiona en `app.py` (backend) y `App.js` (frontend).

## Resolución de problemas frecuentes

**Modelo no localizado:**
```bash
cd backend
python scripts/train_model.py
```
El proceso de entrenamiento inicial puede extenderse según las especificaciones del hardware.

**Error CORS:**
Verificar la instalación de `flask-cors`. En caso negativo: `pip install flask-cors`

**Puerto en uso:**
Modificar la configuración de puerto en `app.py` (backend) o `package.json` (frontend).

**Dependencias incompletas:**
Eliminar directorios `node_modules` y `venv`, posteriormente ejecutar instalación limpia de dependencias.

## Conjunto de datos

El proyecto emplea el "New Plant Diseases Dataset" disponible en Kaggle, que incorpora técnicas de aumentación de datos. La colección abarca miles de fotografías de tejido foliar bajo diversas condiciones patológicas y parámetros de iluminación.

Para integrar conjuntos de datos personalizados, organizar las imágenes en directorios clasificados por categoría dentro de `dataset/raw/` y adaptar los parámetros del script de entrenamiento.

## Estándares de accesibilidad

La interfaz cumple con las directrices WCAG 2.1 nivel AA:
- Navegación completa mediante teclado
- Compatibilidad con tecnologías de asistencia (lectores de pantalla)
- Ratios de contraste cromático conformes a estándares
- Implementación de atributos ARIA en componentes interactivos

## Marco académico

Proyecto desarrollado en el marco de la asignatura Inteligencia Computacional, Universidad Pedagógica y Tecnológica de Colombia (UPTC).

El trabajo busca materializar la aplicación de arquitecturas CNN y técnicas de transfer learning en la resolución de problemáticas reales del sector agropecuario, integrando fundamentos de aprendizaje profundo con desarrollo de aplicaciones web completas.
