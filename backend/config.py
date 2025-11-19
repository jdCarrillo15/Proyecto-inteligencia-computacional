"""
Archivo de configuración centralizado para el proyecto.
Modifica estos valores según tus necesidades.
"""

from pathlib import Path

# ============================================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================================

# Directorio base del proyecto
BASE_DIR = Path(__file__).parent

# Rutas del dataset
DATASET_RAW_DIR = BASE_DIR / 'dataset' / 'raw'
DATASET_PROCESSED_DIR = BASE_DIR / 'dataset' / 'processed'
DATASET_TRAIN_DIR = DATASET_PROCESSED_DIR / 'train'
DATASET_TEST_DIR = DATASET_PROCESSED_DIR / 'test'
DATASET_VIZ_DIR = DATASET_PROCESSED_DIR / 'visualizations'

# Rutas de modelos
MODELS_DIR = BASE_DIR / 'models'
MODEL_PATH = MODELS_DIR / 'fruit_classifier.keras'
BEST_MODEL_PATH = MODELS_DIR / 'best_model.keras'
CLASS_MAPPING_PATH = MODELS_DIR / 'class_mapping.json'
MODEL_VIZ_DIR = MODELS_DIR / 'visualizations'

# Rutas de la aplicación web
STATIC_DIR = BASE_DIR / 'static'
UPLOAD_DIR = STATIC_DIR / 'uploads'
TEMPLATES_DIR = BASE_DIR / 'templates'

# ============================================================================
# CONFIGURACIÓN DE DATOS
# ============================================================================

# Clases de enfermedades de plantas (15 clases específicas del dataset de Kaggle)
CLASSES = [
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
NUM_CLASSES = len(CLASSES)  # 15 clases

# Tamaño de las imágenes (224x224 = estándar ImageNet para transfer learning)
# IMPORTANTE: Al cambiar esto, debes BORRAR backend/cache/*.pkl
IMG_WIDTH = 224   # Aumentado de 100 para mejor detección de síntomas
IMG_HEIGHT = 224  # Preserva texturas finas y manchas pequeñas
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
IMG_CHANNELS = 3  # RGB

# División de datos
TRAIN_SPLIT = 0.8  # 80% entrenamiento, 20% prueba
TEST_SPLIT = 1.0 - TRAIN_SPLIT

# Formatos de imagen aceptados
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

# ============================================================================
# CONFIGURACIÓN DEL MODELO
# ============================================================================

# Arquitectura
CONV_FILTERS = [32, 64, 128, 256]  # Filtros en capas convolucionales
DENSE_UNITS = [512, 256]  # Unidades en capas densas
DROPOUT_RATE = 0.5
CONV_DROPOUT_RATE = 0.25

# Hiperparámetros de entrenamiento
BATCH_SIZE = 16  # Reducido de 32 debido a resolución 224x224 (5x más píxeles)
EPOCHS = 50
LEARNING_RATE = 0.001  # LR inicial para transfer learning

# Optimizador
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'categorical_crossentropy'
METRICS = ['accuracy']

# Callbacks
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-7

# ============================================================================
# CONFIGURACIÓN DE DATA AUGMENTATION
# ============================================================================

# Parámetros de augmentation para entrenamiento
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# ============================================================================
# CONFIGURACIÓN DE MÉTRICAS Y UMBRALES DEL MODELO
# ============================================================================

# Umbrales de rendimiento (basados en MODEL_REQUIREMENTS.md)
PERFORMANCE_THRESHOLDS = {
    # Mínimos aceptables (rechazo del modelo si no se cumplen)
    'min_recall_per_class': 0.60,  # 60% - umbral de rechazo
    'min_precision_per_class': 0.60,  # 60% - umbral de rechazo
    'min_f1_per_class': 0.60,  # 60% - umbral de rechazo
    'min_macro_f1': 0.70,  # 70% - umbral crítico
    'min_overall_accuracy': 0.75,  # 75% - umbral crítico
    
    # Objetivos (cumplimiento esperado)
    'target_recall_per_class': 0.70,  # 70% - objetivo por clase
    'target_precision_per_class': 0.65,  # 65% - objetivo por clase
    'target_f1_per_class': 0.67,  # 67% - objetivo por clase
    'target_macro_f1': 0.75,  # 75% - objetivo principal
    'target_weighted_f1': 0.78,  # 78% - objetivo secundario
    'target_overall_accuracy': 0.80,  # 80% - objetivo de accuracy
    
    # Ideales (excelencia)
    'ideal_recall_per_class': 0.85,  # 85% - rendimiento ideal
    'ideal_precision_per_class': 0.80,  # 80% - rendimiento ideal
    'ideal_f1_per_class': 0.82,  # 82% - rendimiento ideal
    'ideal_macro_f1': 0.85,  # 85% - rendimiento ideal
    'ideal_overall_accuracy': 0.90,  # 90% - rendimiento ideal
}

# Clases críticas que requieren mayor recall (enfermedades de alto impacto)
CRITICAL_DISEASE_CLASSES = [
    'Potato___Late_blight',  # Tizón tardío - pérdida total de cultivo
    'Tomato___Late_blight',  # Tizón tardío - altamente contagioso
    'Corn_(maize)___Northern_Leaf_Blight',  # Tizón del norte - propagación rápida
]

# Umbral especial para clases críticas
CRITICAL_DISEASE_MIN_RECALL = 0.75  # 75% mínimo
CRITICAL_DISEASE_TARGET_RECALL = 0.80  # 80% objetivo

# Prioridad de métricas (para optimización)
METRIC_PRIORITY = 'macro_f1'  # Opciones: 'macro_f1', 'recall', 'accuracy'

# Tolerancia de falsos positivos vs falsos negativos
# Valores > 1.0 priorizan recall (minimizar falsos negativos)
# Valores < 1.0 priorizan precision (minimizar falsos positivos)
FN_FP_TOLERANCE_RATIO = 1.5  # Preferimos 1.5x falsos positivos sobre falsos negativos

# ============================================================================
# CONFIGURACIÓN DE BALANCEO DE CLASES
# ============================================================================

# Activar balanceo de clases en prepare_dataset.py
APPLY_CLASS_BALANCING = True

# Target de muestras por clase después de oversampling
TARGET_SAMPLES_PER_CLASS = 2500

# Objetivo de ratio de balance (max/min samples)
TARGET_BALANCE_RATIO = 2.0  # Máximo 2:1 de diferencia entre clases

# Usar Focal Loss en lugar de Categorical Crossentropy
USE_FOCAL_LOSS = False  # Cambiar a True para usar Focal Loss
FOCAL_LOSS_GAMMA = 2.0  # Factor de enfoque para Focal Loss

# Usar class weights en entrenamiento (recomendado siempre)
USE_CLASS_WEIGHTS = True

# ============================================================================
# CONFIGURACIÓN DE LA APLICACIÓN WEB
# ============================================================================

# Flask
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# Límite de tamaño de archivo (en bytes)
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB

# ============================================================================
# CONFIGURACIÓN DE VISUALIZACIONES
# ============================================================================

# Colores para gráficos
COLORS = ['#FF6B6B', '#FFD93D', '#6BCB77', '#FF8C42', '#4D96FF']

# DPI para guardar imágenes
FIGURE_DPI = 300

# Tamaño de figuras
FIGURE_SIZE_SMALL = (10, 6)
FIGURE_SIZE_MEDIUM = (12, 8)
FIGURE_SIZE_LARGE = (15, 10)

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

# Nivel de logging
LOG_LEVEL = 'INFO'

# Formato de logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# EMOJIS PARA FRUTAS
# ============================================================================

FRUIT_EMOJIS = {
    'manzana': '🍎',
    'banano': '🍌',
    'mango': '🥭',
    'naranja': '🍊',
    'pera': '🍐'
}

# ============================================================================
# MENSAJES
# ============================================================================

MESSAGES = {
    'no_dataset': """
❌ Error: No se encontró el dataset en '{path}'

📋 Estructura esperada:
dataset/raw/
  ├── manzana/
  ├── banano/
  ├── mango/
  ├── naranja/
  └── pera/

Por favor, crea esta estructura y coloca las imágenes correspondientes.
""",
    
    'no_model': """
❌ Error: No se encontró el modelo entrenado.

Por favor, ejecuta primero:
  python train_model.py
""",
    
    'training_complete': """
✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE

📁 Archivos generados:
  - {model_path} (modelo principal)
  - {best_model_path} (mejor modelo)
  - {class_mapping_path} (mapeo de clases)
  - {viz_dir} (visualizaciones)
""",
    
    'cleaning_complete': """
✅ Proceso de limpieza completado exitosamente!

📁 Dataset limpio guardado en: {output_path}
📊 Visualizaciones guardadas en: {viz_path}
"""
}

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def create_directories():
    """Crea todos los directorios necesarios."""
    directories = [
        DATASET_RAW_DIR,
        DATASET_PROCESSED_DIR,
        DATASET_TRAIN_DIR,
        DATASET_TEST_DIR,
        DATASET_VIZ_DIR,
        MODELS_DIR,
        MODEL_VIZ_DIR,
        UPLOAD_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_config_summary():
    """Retorna un resumen de la configuración."""
    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║           CONFIGURACIÓN DEL PROYECTO                         ║
╚══════════════════════════════════════════════════════════════╝

📊 DATOS:
  - Clases: {', '.join(CLASSES)}
  - Tamaño de imagen: {IMG_WIDTH}x{IMG_HEIGHT}
  - División: {TRAIN_SPLIT*100:.0f}% train, {TEST_SPLIT*100:.0f}% test

🧠 MODELO:
  - Filtros Conv: {CONV_FILTERS}
  - Unidades Dense: {DENSE_UNITS}
  - Batch size: {BATCH_SIZE}
  - Épocas: {EPOCHS}
  - Learning rate: {LEARNING_RATE}

🌐 APLICACIÓN:
  - Host: {FLASK_HOST}
  - Puerto: {FLASK_PORT}
  - Max file size: {MAX_FILE_SIZE / (1024*1024):.0f} MB

📁 RUTAS:
  - Dataset: {DATASET_RAW_DIR}
  - Modelos: {MODELS_DIR}
  - Uploads: {UPLOAD_DIR}
"""
    return summary


if __name__ == "__main__":
    # Mostrar configuración
    print(get_config_summary())
    
    # Crear directorios
    print("\n📁 Creando directorios necesarios...")
    create_directories()
    print("✅ Directorios creados exitosamente")
