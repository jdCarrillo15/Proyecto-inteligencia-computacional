# Backend - API de Diagnóstico

Servidor Flask que expone el modelo CNN para detectar enfermedades en plantas.

## Qué hace

Recibe imágenes de hojas, las procesa y devuelve la predicción del modelo junto con las probabilidades de cada clase. También provee endpoints para verificar el estado del servicio.

## Características

- API REST sin HTML (solo JSON)
- CORS configurado para React
- Carga el modelo TensorFlow al iniciar
- Procesa imágenes con Pillow
- Devuelve todas las predicciones ordenadas por confianza
- Manejo de errores robusto

## Requisitos

- Python 3.10 o superior
- pip
- Modelo entrenado (`models/fruit_classifier.keras`)

## Instalación

Crear entorno virtual (recomendado):

```bash
python -m venv venv
```

Activarlo:

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Uso

Si no tienes el modelo entrenado:

```bash
python scripts/train_model.py
```

Iniciar servidor:

```bash
python app.py
```

Corre en http://localhost:5000

## Endpoints

**GET /**

Info básica de la API (nombre, versión, endpoints disponibles).

**GET /health**

Verifica que el modelo esté cargado:

```json
{
  "status": "ok",
  "model_loaded": true,
  "classes": ["apple___apple_scab", "corn_(maize)___healthy", ...]
}
```

**POST /predict**

Enviar imagen para clasificar.

Request:
- Content-Type: `multipart/form-data`
- Field: `file` (JPG, JPEG o PNG)

Response:
```json
{
  "success": true,
  "predicted_class": "tomato___late_blight",
  "confidence": 0.9532,
  "confidence_percentage": "95.32",
  "all_predictions": [
    {"class": "tomato___late_blight", "probability": 0.9532, "percentage": "95.32"},
    {"class": "tomato___early_blight", "probability": 0.0312, "percentage": "3.12"},
    ...
  ],
  "image_data": "base64..."
}
```

**GET /dataset-info**

Devuelve info sobre visualizaciones del dataset (si existen).

## Stack

- Flask 3.0
- Flask-CORS (para conectar con React)
- TensorFlow 2.18 y Keras 3.6
- Pillow (procesamiento de imágenes)
- NumPy

## Estructura

```
backend/
├── app.py                    # Servidor Flask
├── config.py                 # Configuración
├── requirements.txt          # Dependencias
├── models/                   # Modelos entrenados
│   ├── fruit_classifier.keras
│   └── class_mapping.json
├── scripts/                  # Entrenamiento
│   ├── train_model.py
│   └── predict.py
└── utils/                    # Herramientas
    ├── diagnose_model.py
    └── quick_test.py
```

## Seguridad

- Límite de 16MB por archivo
- Solo acepta JPG, JPEG y PNG
- Valida dimensiones mínimas
- Manejo de excepciones en todo el flujo

## Problemas Comunes

**Modelo no encontrado:**
```bash
python scripts/train_model.py
```

**CORS no funciona:**
```bash
pip install flask-cors
```

**Puerto ocupado:**

Cambia el puerto en `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=OTRO_PUERTO)
```

## Sobre el Modelo

El modelo es una CNN entrenada con transfer learning. Procesa imágenes de 100x100 píxeles en RGB y clasifica entre 15 tipos de enfermedades en 4 cultivos diferentes.

La precisión depende de la calidad de la imagen y las condiciones de captura, pero generalmente supera el 90% en fotos claras.

## Proyecto Académico

Desarrollado para Inteligencia Computacional - UPTC
