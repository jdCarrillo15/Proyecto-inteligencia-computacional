# 🔧 Backend - API de Clasificación de Frutas

API REST desarrollada con Flask para servir el modelo de clasificación de frutas con CNN.

## 🚀 Características

- 🔌 API REST pura sin vistas HTML
- 🌐 CORS habilitado para frontend React
- 🤖 Modelo CNN con TensorFlow/Keras
- 📊 Predicciones con confianza y probabilidades
- 🖼️ Procesamiento de imágenes con PIL
- ✅ Endpoints de salud y diagnóstico

## 📋 Prerequisitos

- Python 3.10 o superior
- pip (gestor de paquetes de Python)
- Modelo entrenado en `models/fruit_classifier.keras`

## 🔧 Instalación

1. Crea un entorno virtual (recomendado):
```bash
python -m venv venv
```

2. Activa el entorno virtual:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## 🎯 Uso

1. Asegúrate de tener el modelo entrenado:
```bash
python scripts/train_model.py
```

2. Inicia el servidor:
```bash
python app.py
```

3. El servidor estará disponible en `http://localhost:5000`

## 📡 Endpoints

### GET `/`
Información general de la API
```json
{
  "name": "Fruit Classifier API",
  "version": "1.0.0",
  "description": "API para clasificación de frutas usando CNN",
  "endpoints": {...}
}
```

### GET `/health`
Estado del servicio y modelo
```json
{
  "status": "ok",
  "model_loaded": true,
  "classes": ["manzana", "banano", "mango", "naranja", "pera"]
}
```

### POST `/predict`
Clasificar una imagen de fruta

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (imagen JPG, JPEG, PNG)

**Response:**
```json
{
  "success": true,
  "predicted_class": "manzana",
  "confidence": 0.9876,
  "confidence_percentage": "98.76",
  "all_predictions": [
    {
      "class": "manzana",
      "probability": 0.9876,
      "percentage": "98.76"
    },
    ...
  ],
  "image_data": "base64_encoded_image"
}
```

### GET `/dataset-info`
Información sobre visualizaciones del dataset

## 🛠️ Tecnologías

- **Flask 3.0+** - Framework web
- **Flask-CORS** - Manejo de CORS
- **TensorFlow 2.18+** - Machine Learning
- **Keras 3.6+** - API de alto nivel para redes neuronales
- **Pillow 10.0+** - Procesamiento de imágenes
- **NumPy** - Operaciones numéricas

## 📁 Estructura

```
backend/
├── app.py              # Aplicación Flask principal
├── config.py           # Configuraciones
├── requirements.txt    # Dependencias
├── models/            # Modelos entrenados
│   ├── fruit_classifier.keras
│   └── class_mapping.json
├── scripts/           # Scripts de entrenamiento
│   ├── train_model.py
│   └── predict.py
└── utils/             # Utilidades
    ├── diagnose_model.py
    └── quick_test.py
```

## 🔐 Seguridad

- Límite de tamaño de archivo: 16MB
- Validación de formato de imagen
- Validación de dimensiones mínimas
- Manejo de errores robusto

## 🐛 Solución de Problemas

### Modelo no encontrado
```bash
# Entrena el modelo primero
python scripts/train_model.py
```

### Error de CORS
Verifica que `flask-cors` esté instalado:
```bash
pip install flask-cors
```

### Puerto en uso
Modifica el puerto en `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=OTRO_PUERTO)
```

## 📊 Modelo

- **Arquitectura:** CNN con MobileNetV2
- **Entrada:** Imágenes 100x100 RGB
- **Salida:** 5 clases de frutas
- **Precisión:** ~95%

## 🎓 Proyecto Académico

Desarrollado para el curso de Inteligencia Computacional - UPTC

## 📄 Licencia

Este proyecto es parte de un trabajo académico.
