# 🌿 Sistema de Detección de Enfermedades en Plantas

> Sistema completo de diagnóstico agrícola basado en Deep Learning con CNN (MobileNetV2) para identificación de enfermedades en cultivos de manzana, maíz, papa y tomate.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org)

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Instalación](#-instalación)
- [Uso Rápido](#-uso-rápido)
- [Pipeline de Entrenamiento](#-pipeline-de-entrenamiento-fase-2)
- [Métricas y Requisitos](#-métricas-y-requisitos)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Cultivos Soportados](#-cultivos-y-enfermedades)
- [Documentación](#-documentación)

---

## 🎯 Descripción

Sistema de clasificación de enfermedades en plantas mediante análisis de imágenes. Utiliza **transfer learning con MobileNetV2** para detectar **15 clases** distribuidas en 4 cultivos principales. El sistema prioriza **detectar todas las enfermedades** (alta recall) aunque genere algunas falsas alarmas, ya que es más seguro tratar preventivamente que perder un cultivo.

### ✨ Ventajas Clave

- ⚡ **Inferencia rápida**: < 500ms por imagen
- 🎯 **Alta precisión**: Macro F1-Score ≥ 75%
- 🔍 **Prioridad en recall**: ≥ 80% para enfermedades críticas
- 📊 **Métricas detalladas**: Evaluación completa con reportes Excel
- 🔄 **Pipeline automatizado**: Desde preparación hasta validación
- 💾 **Cache optimizado**: Split 70/15/15 con PKL para entrenamiento rápido

---

## 🚀 Características

### Modelo y Predicción
- ✅ Transfer learning con **MobileNetV2** (ImageNet)
- ✅ Optimización con **Adam** (lr=1e-4)
- ✅ **Data augmentation** avanzado
- ✅ **Class weights** para balanceo
- ✅ **Early stopping** y **ReduceLROnPlateau**
- ✅ Checkpoints duales (best + last)

### Evaluación y Validación
- ✅ **25+ métricas** (accuracy, precision, recall, F1 per-class)
- ✅ **Validación automática** contra requisitos obligatorios
- ✅ **Análisis de fallos** con recomendaciones específicas
- ✅ **Reportes Excel** con 4 hojas (métricas, confusion matrix, metadata)
- ✅ **Visualizaciones** (confusion matrix, métricas por clase)

### Producción
- ✅ **Inference script** optimizado
- ✅ **Latencia verificada** < 500ms
- ✅ **Memory footprint** < 500MB
- ✅ **API REST** con Flask
- ✅ **Frontend React** intuitivo
- ✅ **Tests de readiness** automatizados

---

## 📦 Instalación

### Requisitos
- **Python**: 3.10 o superior
- **Node.js**: 14 o superior
- **RAM**: Mínimo 8GB (16GB recomendado)
- **Disco**: ~5GB para dataset + modelos

### Instalación Rápida

```bash
# 1. Clonar repositorio
git clone https://github.com/jdCarrillo15/Proyecto-inteligencia-computacional.git
cd Proyecto-inteligencia-computacional

# 2. Configurar backend
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 3. Configurar frontend
cd ../frontend
npm install
```

---

## ⚡ Uso Rápido

### ⚠️ IMPORTANTE: Si tienes problemas de RAM

Si el sistema colapsa o la RAM sube a 90%+, lee **[SOLUCION_RAM.md](./SOLUCION_RAM.md)** primero.

**Solución rápida:**
```bat
# Si tienes 12-16 GB RAM
.\prepare_safe.bat

# Si tienes 8 GB RAM o menos
python prepare_ultralight.py
```

### Scripts Batch (Windows) - **RECOMENDADO**

```bat
# 1. Entrenar modelo (Fase 2)
train.bat

# 2. Evaluar modelo (Pasos 3, 4, 5)
evaluate.bat

# 3. Test de readiness (Paso 6)
test_ready.bat

# 4. Iniciar backend
start-backend.bat

# 5. Iniciar frontend
start-frontend.bat
```

### Comandos Manual

```bash
# Entrenamiento
python backend/scripts/train.py

# Evaluación completa (incluye validación + análisis)
python backend/scripts/evaluate_model.py

# Test de readiness
python backend/scripts/test_ready.py

# Inferencia
python backend/scripts/inference.py --image ruta/imagen.jpg
python backend/scripts/inference.py --batch ruta/carpeta/
python backend/scripts/inference.py --info

# Servidor backend
cd backend
python app.py

# Servidor frontend
cd frontend
npm start
```

---

## 🔄 Pipeline de Entrenamiento (Fase 2)

### Pipeline Completo: 6 Pasos Automatizados

```
┌───────────────────────────────────────────────────────────────┐
│  FASE 2: PIPELINE DE ENTRENAMIENTO Y EVALUACIÓN              │
└───────────────────────────────────────────────────────────────┘

📊 PASO 1: Preparación de Dataset
   • Split 70/15/15 (train/val/test)
   • Normalización ImageNet
   • Data augmentation
   • Cache PKL optimizado
   ↓
🏋️ PASO 2: Entrenamiento
   • Adam (lr=1e-4)
   • Batch size: 64
   • Max epochs: 100
   • Early stopping (patience=15)
   • Checkpoints: best_model + last_model
   ↓
📈 PASO 3: Evaluación
   • Métricas completas (25+)
   • Confusion matrix
   • Reporte Excel (4 hojas)
   • Visualizaciones
   ↓
✅ PASO 4: Validación contra Requisitos
   • Macro F1 ≥ 75% (OBLIGATORIO)
   • Accuracy ≥ 75% (OBLIGATORIO)
   • Recall críticos ≥ 80% (OBLIGATORIO)
   • Estado: APROBADO / CONDICIONAL / RECHAZADO
   ↓
🔍 PASO 5: Análisis de Problemas (si falla)
   • Clases problemáticas
   • Patrones de confusión
   • Recomendaciones específicas
   • Acciones prioritarias
   ↓
🚀 PASO 6: Testing Final
   • Guardar/cargar modelo
   • Latencia < 500ms
   • Memoria < 500MB
   • Inference script listo
```

### Ejecutar Pipeline Completo

```bash
# Opción 1: Scripts batch (Recomendado en Windows)
train.bat        # Paso 2
evaluate.bat     # Pasos 3, 4, 5 (automático)
test_ready.bat   # Paso 6

# Opción 2: Python directo
python backend/scripts/train.py           # Paso 2
python backend/scripts/evaluate_model.py  # Pasos 3, 4, 5
python backend/scripts/test_ready.py      # Paso 6
```

---

## 📊 Métricas y Requisitos

### Umbrales OBLIGATORIOS

| Métrica | Umbral | Descripción |
|---------|--------|-------------|
| **Macro F1-Score** | ≥ 75% | Promedio balanceado de F1 por clase |
| **Overall Accuracy** | ≥ 75% | Accuracy general en test set |
| **Recall Críticos** | ≥ 80% | Recall mínimo para enfermedades críticas |

### Clases Críticas (Alta Prioridad)

1. **Potato___Late_blight**: Tizón tardío - pérdida total de cultivo
2. **Tomato___Late_blight**: Tizón tardío - altamente contagioso
3. **Corn_(maize)___Northern_Leaf_Blight**: Propagación rápida

### Estados de Validación

- **✅ APROBADO**: Cumple todos los requisitos obligatorios + objetivos
- **⚠️ CONDICIONAL**: Cumple obligatorios pero no objetivos → Ajustar hiperparámetros
- **❌ RECHAZADO**: No cumple obligatorios → Investigar y reentrenar

### Reportes Generados

| Archivo | Descripción |
|---------|-------------|
| `metrics/evaluation_results.json` | Métricas completas en JSON |
| `metrics/evaluation_results.xlsx` | Reporte Excel (4 hojas) |
| `metrics/validation_report.json` | Estado de validación + acciones |
| `metrics/failure_analysis.json` | Análisis de problemas (si falla) |
| `metrics/readiness_report.json` | Tests de producción |
| `metrics/training_history.json` | Historial de entrenamiento |

---

## 📁 Estructura del Proyecto

```
Proyecto-inteligencia-computacional/
│
├── 📜 README.md                    # Este archivo
├── 📜 MODEL_REQUIREMENTS.md        # Requisitos detallados del modelo
├── 📜 GUIA_SCRIPTS.md             # Guía de scripts disponibles
│
├── 🔧 Scripts Batch
│   ├── train.bat                  # Entrenamiento (Paso 2)
│   ├── evaluate.bat               # Evaluación completa (Pasos 3-5)
│   ├── test_ready.bat             # Testing readiness (Paso 6)
│   ├── start-backend.bat          # Iniciar servidor Flask
│   ├── start-frontend.bat         # Iniciar app React
│   └── clean_cache.bat            # Limpiar cache PKL
│
├── 📦 backend/
│   ├── app.py                     # API REST Flask
│   ├── config.py                  # Configuración centralizada
│   ├── requirements.txt           # Dependencias Python
│   │
│   ├── 📂 scripts/
│   │   ├── train.py               # Paso 2: Entrenamiento
│   │   ├── evaluate_model.py      # Paso 3: Evaluación
│   │   ├── validate_requirements.py  # Paso 4: Validación
│   │   ├── analyze_failures.py    # Paso 5: Análisis
│   │   ├── test_ready.py          # Paso 6: Testing
│   │   ├── inference.py           # Inferencia optimizada
│   │   ├── prepare_dataset.py     # Paso 1: Preparación
│   │   └── detailed_metrics.py    # Sistema de métricas
│   │
│   ├── 📂 utils/
│   │   ├── data_cache.py          # Cache PKL
│   │   └── manage_cache.py        # Gestión de cache
│   │
│   ├── 📂 models/                 # Modelos entrenados
│   │   ├── best_model.keras       # Mejor modelo
│   │   ├── last_model.keras       # Último checkpoint
│   │   └── visualizations/        # Gráficos generados
│   │
│   └── 📂 cache/                  # Cache PKL (70/15/15)
│       ├── train_data.pkl
│       ├── val_data.pkl
│       └── test_data.pkl
│
├── 📂 frontend/
│   ├── package.json               # Dependencias Node.js
│   ├── 📂 src/
│   │   ├── App.js                 # Componente principal
│   │   ├── 📂 components/         # Componentes React
│   │   ├── 📂 data/              # Base de datos enfermedades
│   │   ├── 📂 styles/            # Estilos modulares
│   │   └── 📂 utils/             # Utilidades frontend
│   │
│   └── 📂 public/
│
├── 📂 dataset/
│   └── 📂 raw/
│       └── New Plant Diseases Dataset(Augmented)/
│           ├── 📂 train/         # 70% - 28,428 imágenes
│           └── 📂 test/          # Split: 15% val + 15% test
│
└── 📂 metrics/                    # Reportes y métricas
    ├── evaluation_results.json
    ├── evaluation_results.xlsx
    ├── validation_report.json
    ├── failure_analysis.json
    ├── readiness_report.json
    └── training_history.json
```

---

## 🌾 Cultivos y Enfermedades

### 15 Clases en 4 Cultivos

#### 🍎 Manzana (4 clases)
1. `Apple___Apple_scab` - Sarna del manzano
2. `Apple___Black_rot` - Pudrición negra
3. `Apple___Cedar_apple_rust` - Roya del cedro del manzano
4. `Apple___healthy` - Hojas sanas

#### 🌽 Maíz (3 clases)
5. `Corn_(maize)___Common_rust_` - Roya común
6. `Corn_(maize)___Northern_Leaf_Blight` - ⚠️ **Tizón del norte** (CRÍTICO)
7. `Corn_(maize)___healthy` - Hojas sanas

#### 🥔 Papa (3 clases)
8. `Potato___Early_blight` - Tizón temprano
9. `Potato___Late_blight` - ⚠️ **Tizón tardío** (CRÍTICO)
10. `Potato___healthy` - Hojas sanas

#### 🍅 Tomate (5 clases)
11. `Tomato___Bacterial_spot` - Mancha bacteriana
12. `Tomato___Early_blight` - Tizón temprano
13. `Tomato___Late_blight` - ⚠️ **Tizón tardío** (CRÍTICO)
14. `Tomato___Leaf_Mold` - Moho de la hoja
15. `Tomato___healthy` - Hojas sanas

> **Nota**: Las 3 enfermedades críticas (⚠️) requieren Recall ≥ 80% obligatorio.

---

## 🔧 Configuración Avanzada

### Ajustar Hiperparámetros

Editar `backend/config.py`:

```python
# Entrenamiento
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0001

# Callbacks
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# Umbrales de validación
PERFORMANCE_THRESHOLDS = {
    'min_macro_f1': 0.75,
    'min_overall_accuracy': 0.75,
    'min_critical_recall': 0.80,
}
```

### Data Augmentation

```python
# config.py
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}
```

---

## 📚 Documentación

### Documentos Principales

- **[MODEL_REQUIREMENTS.md](./MODEL_REQUIREMENTS.md)**: Requisitos detallados del modelo, umbrales y métricas
- **[GUIA_SCRIPTS.md](./GUIA_SCRIPTS.md)**: Guía completa de todos los scripts disponibles
- **[backend/README.md](./backend/README.md)**: Documentación del backend
- **[frontend/README.md](./frontend/README.md)**: Documentación del frontend

### Scripts de Validación

```bash
python backend/scripts/validate_paso2.py  # Validar entrenamiento
python backend/scripts/validate_paso3.py  # Validar evaluación
python backend/scripts/validate_paso4.py  # Validar validación
python backend/scripts/validate_paso5.py  # Validar análisis
python backend/scripts/validate_paso6.py  # Validar readiness
```

---

## 🛠️ Stack Tecnológico

### Backend
- **Flask 3.0** - Framework web
- **TensorFlow 2.18** - Deep learning
- **Keras 3.6** - API de alto nivel
- **Pillow** - Procesamiento de imágenes
- **psutil** - Monitoreo de recursos

### Frontend
- **React 19** - Biblioteca UI
- **Axios** - Cliente HTTP
- **CSS3** - Estilos modernos

### Machine Learning
- **MobileNetV2** - Arquitectura base (ImageNet)
- **Transfer Learning** - Fine-tuning progresivo
- **Cache PKL** - Optimización de datos
- **scikit-learn** - Métricas y utilidades

---

## 📈 Estadísticas

- **Líneas de código**: ~15,000+
- **Scripts Python**: 20+
- **Componentes React**: 15+
- **Dataset**: 28,428 imágenes
- **Clases**: 15
- **Parámetros del modelo**: ~3.5M
- **Validaciones**: 118 checks automatizados

---

<div align="center">

**[⬆ Volver arriba](#-sistema-de-detección-de-enfermedades-en-plantas)**

</div>
