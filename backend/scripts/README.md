# Scripts del Sistema

Directorio que contiene los scripts fundamentales para procesamiento de datos, entrenamiento del modelo y ejecución de inferencias.

## Archivos

### data_preparation.py
**Objetivo:** Preprocesamiento y acondicionamiento del conjunto de datos

**Operaciones realizadas:**
- Verificación de integridad de archivos de imagen
- Validación dimensional
- Redimensionamiento uniforme a 100x100 píxeles
- Normalización de valores de píxeles al rango [0,1]
- Partición entrenamiento/prueba (ratio 80/20)
- Generación de gráficos exploratorios

**Ejecución:**
```bash
python scripts/data_preparation.py
```

**Estructura de salida:**
- `dataset/processed/train/` - Conjunto de entrenamiento
- `dataset/processed/test/` - Conjunto de prueba
- `dataset/processed/visualizations/` - Visualizaciones

---

### train_model.py
**Objetivo:** Entrenamiento y optimización de la red neuronal convolucional

**Procesos implementados:**
- Construcción de arquitectura CNN
- Aplicación de técnicas de aumentación de datos
- Entrenamiento con callbacks de control
- Cálculo de métricas de evaluación
- Persistencia del modelo entrenado

**Ejecución:**
```bash
python scripts/train_model.py
```

**Artefactos generados:**
- `models/fruit_classifier.h5` - Modelo final
- `models/best_model.h5` - Modelo con mejor rendimiento
- `models/class_mapping.json` - Diccionario de clases
- `models/visualizations/` - Métricas y curvas de aprendizaje

---

### predict.py
**Objetivo:** Inferencia mediante interfaz de línea de comandos

**Capacidades:**
- Carga dinámica del modelo entrenado
- Preprocesamiento automático de imagen de entrada
- Generación de predicciones con distribuciones de probabilidad
- Presentación formateada de resultados

**Sintaxis de uso:**
```bash
# Inferencia básica
python scripts/predict.py imagen.jpg

# Visualizar distribución completa de probabilidades
python scripts/predict.py imagen.jpg --all

# Especificar ruta de modelo personalizado
python scripts/predict.py imagen.jpg --model models/best_model.h5 --all
```

---

## Flujo de ejecución recomendado

```
1. data_preparation.py
   ↓
   Conjunto de datos preprocesado y particionado
   ↓
2. train_model.py
   ↓
   Modelo entrenado y serializado (.h5)
   ↓
3. predict.py
   ↓
   Inferencias sobre imágenes
```

## Consideraciones técnicas

- Secuencia de ejecución: preparación → entrenamiento → inferencia
- Todos los scripts incorporan logging exhaustivo
- Parámetros configurables centralizados en `config.py`
