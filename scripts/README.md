# 📜 Scripts Principales

Esta carpeta contiene los scripts principales del proyecto para el procesamiento de datos, entrenamiento y predicción.

## Archivos

### 🧹 data_preparation.py
**Propósito:** Limpieza y preparación del dataset

**Funcionalidades:**
- Verificación de imágenes corruptas
- Validación de dimensiones
- Redimensionamiento a 100x100 píxeles
- Normalización de valores (0-1)
- División train/test (80/20)
- Generación de visualizaciones

**Uso:**
```bash
python scripts/data_preparation.py
```

**Salida:**
- `dataset/processed/train/` - Datos de entrenamiento
- `dataset/processed/test/` - Datos de prueba
- `dataset/processed/visualizations/` - Gráficos

---

### 🧠 train_model.py
**Propósito:** Entrenamiento del modelo CNN

**Funcionalidades:**
- Construcción de arquitectura CNN
- Data augmentation
- Entrenamiento con callbacks
- Evaluación y métricas
- Exportación del modelo

**Uso:**
```bash
python scripts/train_model.py
```

**Salida:**
- `models/fruit_classifier.h5` - Modelo entrenado
- `models/best_model.h5` - Mejor modelo
- `models/class_mapping.json` - Mapeo de clases
- `models/visualizations/` - Métricas y gráficos

---

### 🔍 predict.py
**Propósito:** Predicción desde línea de comandos

**Funcionalidades:**
- Carga del modelo entrenado
- Preprocesamiento de imagen
- Predicción con probabilidades
- Visualización de resultados

**Uso:**
```bash
# Predicción simple
python scripts/predict.py imagen.jpg

# Mostrar todas las probabilidades
python scripts/predict.py imagen.jpg --all

# Usar modelo específico
python scripts/predict.py imagen.jpg --model models/best_model.h5 --all
```

---

## Flujo de Trabajo

```
1. data_preparation.py
   ↓
   Dataset limpio y organizado
   ↓
2. train_model.py
   ↓
   Modelo entrenado (.h5)
   ↓
3. predict.py
   ↓
   Predicciones
```

## Notas

- Ejecuta los scripts en orden: preparación → entrenamiento → predicción
- Todos los scripts incluyen logging detallado
- Los parámetros se pueden configurar en `config.py`
