"""
API Flask para clasificación de frutas usando el modelo CNN entrenado.
Proporciona endpoints REST para predicciones de imágenes.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path
import io
import base64

app = Flask(__name__)

# Directorio base del proyecto
BASE_DIR = Path(__file__).parent.parent

# Configuración de CORS segura
# Para desarrollo: permite localhost:3000
# Para producción: configura ALLOWED_ORIGINS en variable de entorno
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(',')

CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False,
        "max_age": 3600  # Cache preflight requests por 1 hora
    }
})

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Variables globales para el modelo
model = None
class_names = None
img_size = (100, 100)


def load_model_and_classes():
    """Carga el modelo y el mapeo de clases."""
    global model, class_names
    
    # Intentar cargar formato .keras primero, luego .h5
    model_path_keras = BASE_DIR / 'models' / 'fruit_classifier.keras'
    model_path_h5 = BASE_DIR / 'models' / 'fruit_classifier.h5'
    mapping_path = BASE_DIR / 'models' / 'class_mapping.json'
    
    model_path = None
    if model_path_keras.exists():
        model_path = model_path_keras
    elif model_path_h5.exists():
        model_path = model_path_h5
    
    if not model_path:
        print("⚠️  Advertencia: No se encontró el modelo entrenado.")
        print("Por favor, ejecuta 'train_model.py' primero.")
        return False
    
    # Cargar modelo con custom objects para MobileNetV2
    print(f"📦 Cargando modelo desde: {model_path}")
    try:
        # Intentar cargar normalmente
        model = keras.models.load_model(model_path, compile=False)
        
        # Recompilar el modelo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Modelo cargado exitosamente")
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        print("\n💡 Solución: Re-entrena el modelo con:")
        print("   python3 scripts/train_model.py")
        print("   (Guardará en formato .keras compatible)")
        return False
    
    # Cargar mapeo de clases
    if mapping_path.exists():
        with open(mapping_path, 'r') as f:
            class_mapping = json.load(f)
            class_names = class_mapping['class_names']
    else:
        # Default: 15 clases específicas del dataset de Kaggle
        class_names = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___healthy',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Potato___Early_blight',
            'Potato___healthy',
            'Potato___Late_blight',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___healthy',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold'
        ]
    
    print(f"✅ Clases cargadas ({len(class_names)}): {class_names}")
    return True


def preprocess_image(image_file):
    """
    Preprocesa una imagen para predicción.
    
    Args:
        image_file: Archivo de imagen
        
    Returns:
        tuple: (imagen_procesada, imagen_original, es_válida, mensaje_error)
    """
    try:
        # Leer imagen
        img = Image.open(image_file)
        
        # Guardar copia original para mostrar
        img_original = img.copy()
        
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Verificar dimensiones mínimas
        if img.size[0] < 50 or img.size[1] < 50:
            return None, None, False, "La imagen es demasiado pequeña (mínimo 50x50 píxeles)"
        
        # Redimensionar
        img_resized = img.resize(img_size)
        
        # Convertir a array y normalizar
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        
        # Añadir dimensión de batch
        img_batch = np.expand_dims(img_array, axis=0)
        
        return img_batch, img_original, True, None
        
    except Exception as e:
        return None, None, False, f"Error al procesar la imagen: {str(e)}"


def format_class_name_for_frontend(class_name):
    """
    Convierte el nombre de clase del modelo al formato que espera el frontend.
    Ejemplo: 'Apple___Apple_scab' -> 'apple___apple_scab'
    
    Args:
        class_name: Nombre de clase del modelo
        
    Returns:
        str: Nombre de clase en formato lowercase
    """
    return class_name.lower()


def predict_fruit(image_file):
    """
    Realiza la predicción de la fruta.
    
    Args:
        image_file: Archivo de imagen
        
    Returns:
        dict: Resultado de la predicción
    """
    if model is None:
        return {
            'success': False,
            'error': 'Modelo no cargado. Por favor, entrena el modelo primero.'
        }
    
    # Preprocesar imagen
    img_processed, img_original, is_valid, error_msg = preprocess_image(image_file)
    
    if not is_valid:
        return {
            'success': False,
            'error': error_msg
        }
    
    # Realizar predicción
    predictions = model.predict(img_processed, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    # Obtener nombre de clase en formato que espera el frontend
    predicted_class_name = format_class_name_for_frontend(class_names[predicted_class_idx])
    
    # Obtener todas las probabilidades
    all_predictions = []
    for idx, prob in enumerate(predictions[0]):
        all_predictions.append({
            'class': format_class_name_for_frontend(class_names[idx]),
            'probability': float(prob),
            'percentage': f"{float(prob) * 100:.2f}"
        })
    
    # Ordenar por probabilidad
    all_predictions.sort(key=lambda x: x['probability'], reverse=True)
    
    # Convertir imagen original a base64 para mostrar
    buffered = io.BytesIO()
    img_original.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        'success': True,
        'predicted_class': predicted_class_name,
        'confidence': confidence,
        'confidence_percentage': f"{confidence * 100:.2f}",
        'all_predictions': all_predictions,
        'image_data': img_str
    }


@app.route('/')
def index():
    """Endpoint raíz - información de la API."""
    return jsonify({
        'name': 'Fruit Classifier API',
        'version': '1.0.0',
        'description': 'API para clasificación de frutas usando CNN',
        'endpoints': {
            '/health': 'Estado del servicio',
            '/predict': 'Realizar predicción (POST)',
            '/dataset-info': 'Información del dataset'
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para realizar predicciones."""
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No se proporcionó ningún archivo'
        })
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No se seleccionó ningún archivo'
        })
    
    # Verificar extensión
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    file_ext = Path(file.filename).suffix
    
    if file_ext not in allowed_extensions:
        return jsonify({
            'success': False,
            'error': f'Formato de archivo no válido. Use: {", ".join(allowed_extensions)}'
        })
    
    # Realizar predicción
    result = predict_fruit(file)
    
    return jsonify(result)


@app.route('/health')
def health():
    """Endpoint de salud para verificar que la app está funcionando."""
    formatted_classes = [format_class_name_for_frontend(cls) for cls in class_names] if class_names else []
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'classes': formatted_classes,
        'num_classes': len(formatted_classes)
    })


@app.route('/dataset-info')
def dataset_info():
    """Endpoint para obtener información del dataset."""
    viz_path = BASE_DIR / 'dataset' / 'processed' / 'visualizations'
    
    info = {
        'visualizations_available': viz_path.exists()
    }
    
    if viz_path.exists():
        info['visualizations'] = [
            str(f.name) for f in viz_path.glob('*.png')
        ]
    
    return jsonify(info)


if __name__ == '__main__':
    print("\n🍎 API REST - CLASIFICADOR DE ENFERMEDADES EN PLANTAS 🌿")
    print("=" * 60)
    
    # Cargar modelo
    if load_model_and_classes():
        print("\n🚀 Iniciando servidor Flask API...")
        print("📡 API disponible en: http://localhost:5000")
        print("🌐 Frontend React: http://localhost:3000")
        print(f"🔒 CORS habilitado para: {', '.join(ALLOWED_ORIGINS)}")
        print("=" * 60)
        
        # Información de configuración
        print("\n⚙️  CONFIGURACIÓN:")
        print(f"  - Debug mode: {app.debug}")
        print(f"  - Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f}MB")
        print(f"  - Allowed origins: {len(ALLOWED_ORIGINS)} origin(s)")
        
        print("\n💡 PARA PRODUCCIÓN:")
        print("  1. Establece la variable de entorno ALLOWED_ORIGINS:")
        print("     export ALLOWED_ORIGINS='https://tudominio.com,https://www.tudominio.com'")
        print("  2. Desactiva debug mode")
        print("  3. Usa un servidor WSGI (gunicorn, waitress)")
        print("  4. Configura HTTPS/SSL")
        print("=" * 60)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ No se pudo iniciar la aplicación.")
        print("Asegúrate de haber entrenado el modelo primero ejecutando 'train_model.py'")


"""
NOTAS DE SEGURIDAD Y DESPLIEGUE:

1. CORS (Cross-Origin Resource Sharing):
   - Por defecto permite: http://localhost:3000 (desarrollo)
   - Para producción, configura ALLOWED_ORIGINS con tus dominios:
     Windows: set ALLOWED_ORIGINS=https://tudominio.com,https://www.tudominio.com
     Linux/Mac: export ALLOWED_ORIGINS=https://tudominio.com,https://www.tudominio.com

2. Variables de entorno recomendadas para producción:
   - ALLOWED_ORIGINS: Dominios permitidos (separados por coma)
   - FLASK_ENV: production
   - SECRET_KEY: Clave secreta única (si usas sesiones)
   - MAX_CONTENT_LENGTH: Tamaño máximo de archivo

3. Servidor de producción:
   - NO usar Flask development server (app.run)
   - Usar Gunicorn, uWSGI o Waitress
   - Ejemplo con Gunicorn:
     gunicorn -w 4 -b 0.0.0.0:5000 app:app

4. Seguridad adicional:
   - Implementar rate limiting (Flask-Limiter)
   - Validación de archivos robusta
   - HTTPS obligatorio en producción
   - Logging y monitoreo de errores
   - Configurar firewall y security groups

5. Optimizaciones:
   - Cache de modelo en memoria (ya implementado)
   - Compresión de respuestas (Flask-Compress)
   - CDN para archivos estáticos
   - Load balancing para alta carga
"""
