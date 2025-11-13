# Backend - Clasificador de Enfermedades de Plantas

Sistema de clasificación de enfermedades en plantas usando Deep Learning con Transfer Learning (MobileNetV2).

## 📁 Estructura

```
backend/
├── app.py                      # API REST Flask
├── config.py                   # Configuración centralizada
├── requirements.txt            # Dependencias
├── scripts/
│   ├── train.py               # Entrenamiento del modelo
│   ├── prepare_dataset.py     # Preparación de datos
│   └── predict.py             # Predicciones desde terminal
├── utils/
│   ├── data_cache.py          # Sistema de cache
│   └── manage_cache.py        # Gestión del cache
└── cache/                     # Cache (generado automáticamente)
```

## 🚀 Uso

### 1. Instalar dependencias
```bash
pip install -r backend/requirements.txt
```

### 2. Entrenar el modelo
```bash
python backend/scripts/train.py
```

El script hace automáticamente:
- ✅ Detecta si necesita preparar datos
- ✅ Usa cache si existe
- ✅ Entrena con Transfer Learning
- ✅ Evalúa y guarda el modelo
- ✅ Genera visualizaciones

**Tiempo estimado:**
- Primera vez: 15-30 min (prepara datos + entrena)
- Con cache: 10-20 min (solo entrena)
- Re-entrenamiento: 10-15 min (cache + train)

### 3. Probar predicciones
```bash
python backend/scripts/predict.py dataset/raw/test/AppleScab1.JPG
python backend/scripts/predict.py dataset/raw/test/TomatoHealthy1.JPG --all
```

### 4. Iniciar API
```bash
python backend/app.py
```
API disponible en: http://localhost:5000

## 📊 Scripts Disponibles

### `train.py` ⭐
Script principal de entrenamiento:
```bash
python backend/scripts/train.py
```

**Características:**
- Detecta automáticamente si hay cache
- Prepara datos si es necesario
- Entrena y evalúa el modelo
- Guarda todo automáticamente

### `prepare_dataset.py`
Preparación manual de datos (opcional):
```bash
python backend/scripts/prepare_dataset.py
```
Nota: `train.py` ya prepara datos automáticamente si es necesario.

### `predict.py`
Predicciones desde terminal:
```bash
python backend/scripts/predict.py <imagen> [--all] [--model <ruta>]
```

## 🎯 15 Enfermedades Clasificadas

1. Apple___Apple_scab
2. Apple___Black_rot
3. Apple___Cedar_apple_rust
4. Apple___healthy
5. Corn_(maize)___Common_rust_
6. Corn_(maize)___healthy
7. Corn_(maize)___Northern_Leaf_Blight
8. Potato___Early_blight
9. Potato___healthy
10. Potato___Late_blight
11. Tomato___Bacterial_spot
12. Tomato___Early_blight
13. Tomato___healthy
14. Tomato___Late_blight
15. Tomato___Leaf_Mold

## 🧠 Arquitectura del Modelo

- **Base:** MobileNetV2 pre-entrenado (ImageNet)
- **Data Augmentation:** RandomFlip, RandomRotation, RandomZoom, RandomContrast
- **Regularización:** Dropout 0.3, Batch size 32
- **Optimizador:** Adam (lr=0.001)

## 📈 Resultados Esperados

- **Precisión objetivo:** 60-80%
- **Tiempo de entrenamiento:** 15-30 min (primera vez)
- **15 clases:** Apple, Corn, Potato, Tomato (sanas y enfermas)

## 📚 API REST

### POST /predict
Clasificar imagen:
```bash
curl -X POST -F "file=@imagen.jpg" http://localhost:5000/predict
```

### GET /health
Estado del servicio

### GET /
Info de la API

## 🔧 Solución de Problemas

**"Cache no encontrado"**
```bash
python backend/scripts/train.py  # Regenera automáticamente
```

**"Modelo no encontrado"**
```bash
python backend/scripts/train.py
```

**"Baja precisión"**
- Asegúrate de que fine-tuning esté desactivado
- Verifica que data augmentation esté activo
- Limpia cache y re-entrena

## 📝 Notas

- **train.py:** Script principal, hace todo automáticamente
- **Cache:** Acelera entrenamientos reutilizando datos procesados
- **Transfer Learning:** Usa MobileNetV2 pre-entrenado
- **Data Augmentation:** Previene overfitting

---

**Stack:** TensorFlow 2.18, Keras 3.6, Flask 3.0, OpenCV 4.8
