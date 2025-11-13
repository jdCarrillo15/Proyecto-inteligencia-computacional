# Backend - Sistema de Clasificación Fitopatológica

Servicio de clasificación de enfermedades en plantas basado en aprendizaje profundo mediante técnicas de transfer learning sobre arquitectura MobileNetV2.

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

## Guía de uso

### 1. Instalación de dependencias
```bash
pip install -r backend/requirements.txt
```

### 2. Entrenamiento del modelo
```bash
python backend/scripts/train.py
```

Procesos automatizados:
- Detección de necesidad de preparación de datos
- Utilización de caché si está disponible
- Entrenamiento mediante transfer learning
- Evaluación y persistencia del modelo
- Generación de visualizaciones de rendimiento

**Tiempos de ejecución estimados:**
- Ejecución inicial: 15-30 min (preparación + entrenamiento)
- Con caché disponible: 10-20 min (solo entrenamiento)
- Re-entrenamiento: 10-15 min (caché + entrenamiento)

### 3. Evaluación mediante predicciones
```bash
python backend/scripts/predict.py dataset/raw/test/AppleScab1.JPG
python backend/scripts/predict.py dataset/raw/test/TomatoHealthy1.JPG --all
```

### 4. Inicialización del servidor API
```bash
python backend/app.py
```
Servicio disponible en: http://localhost:5000

## Scripts del sistema

### `train.py` (Principal)
Script principal para el proceso de entrenamiento:
```bash
python backend/scripts/train.py
```

**Funcionalidades integradas:**
- Detección automática de caché disponible
- Preparación de datos según necesidad
- Entrenamiento y evaluación del modelo
- Persistencia automática de resultados

### `prepare_dataset.py`
Preparación manual del conjunto de datos (uso opcional):
```bash
python backend/scripts/prepare_dataset.py
```
Observación: El script `train.py` gestiona automáticamente la preparación de datos.

### `predict.py`
Inferencia desde línea de comandos:
```bash
python backend/scripts/predict.py <imagen> [--all] [--model <ruta>]
```

## Categorías de clasificación (15 clases)

1. Manzana - Sarna del manzano
2. Manzana - Pudrición negra
3. Manzana - Roya del cedro
4. Manzana - Tejido sano
5. Maíz - Roya común
6. Maíz - Tejido sano
7. Maíz - Tizón del norte
8. Papa - Tizón temprano
9. Papa - Tejido sano
10. Papa - Tizón tardío
11. Tomate - Mancha bacteriana
12. Tomate - Tizón temprano
13. Tomate - Tejido sano
14. Tomate - Tizón tardío
15. Tomato___Leaf_Mold

## Arquitectura del modelo

- **Modelo base:** MobileNetV2 preentrenado en ImageNet
- **Aumentación de datos:** RandomFlip, RandomRotation, RandomZoom, RandomContrast
- **Técnicas de regularización:** Dropout 0.3, Batch size 32
- **Optimizador:** Adam con tasa de aprendizaje 0.001

## Resultados esperados

- **Precisión objetivo:** 60-80%
- **Tiempo de entrenamiento inicial:** 15-30 minutos
- **Clasificación:** 15 categorías patológicas en 4 especies vegetales

## Endpoints de la API

### POST /predict
Clasificación de imagen:
```bash
curl -X POST -F "file=@imagen.jpg" http://localhost:5000/predict
```

### GET /health
Verificación de disponibilidad del servicio

### GET /
Metadata de la API

## Resolución de problemas

**"Caché no localizado"**
```bash
python backend/scripts/train.py  # Regeneración automática
```

**"Modelo no localizado"**
```bash
python backend/scripts/train.py
```

**"Precisión por debajo de lo esperado"**
- Verificar que el fine-tuning esté desactivado
- Confirmar activación de aumentación de datos
- Eliminar caché y ejecutar re-entrenamiento

## Notas técnicas

- **train.py:** Script principal con ejecución automatizada completa
- **Sistema de caché:** Optimización de entrenamientos mediante reutilización de datos procesados
- **Transfer Learning:** Implementación basada en MobileNetV2 preentrenado
- **Aumentación de datos:** Mitigación de sobreajuste

---

**Stack tecnológico:** TensorFlow 2.18, Keras 3.6, Flask 3.0, OpenCV 4.8
