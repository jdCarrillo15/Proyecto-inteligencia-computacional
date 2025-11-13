"""
Script de utilidad para realizar predicciones desde línea de comandos.
Útil para pruebas rápidas sin necesidad de la aplicación web.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import json


def load_model_and_classes(model_path='models/fruit_classifier.h5'):
    """Carga el modelo y el mapeo de clases."""
    if not Path(model_path).exists():
        print(f"❌ Error: No se encontró el modelo en '{model_path}'")
        print("Por favor, ejecuta 'train_model.py' primero.")
        return None, None
    
    # Cargar modelo
    model = keras.models.load_model(model_path)
    
    # Cargar mapeo de clases
    mapping_path = Path(model_path).parent / 'class_mapping.json'
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
    
    return model, class_names


def preprocess_image(image_path, img_size=(100, 100)):
    """Preprocesa una imagen para predicción."""
    try:
        # Cargar imagen
        img = Image.open(image_path)
        
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionar
        img_resized = img.resize(img_size)
        
        # Convertir a array y normalizar
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        
        # Añadir dimensión de batch
        img_batch = np.expand_dims(img_array, axis=0)
        
        return img_batch, True, None
        
    except Exception as e:
        return None, False, str(e)


def predict_image(model, class_names, image_path, show_all=False):
    """Realiza la predicción de una imagen."""
    # Preprocesar
    img_processed, success, error = preprocess_image(image_path)
    
    if not success:
        print(f"❌ Error al procesar la imagen: {error}")
        return
    
    # Predecir
    predictions = model.predict(img_processed, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Emojis
    fruit_emojis = {
        'manzana': '🍎',
        'banano': '🍌',
        'mango': '🥭',
        'naranja': '🍊',
        'pera': '🍐'
    }
    
    # Mostrar resultado principal
    predicted_class = class_names[predicted_class_idx]
    emoji = fruit_emojis.get(predicted_class, '🍓')
    
    print("\n" + "=" * 60)
    print("RESULTADO DE LA PREDICCIÓN")
    print("=" * 60)
    print(f"\n{emoji}  Fruta detectada: {predicted_class.upper()}")
    print(f"📊 Confianza: {confidence * 100:.2f}%")
    
    # Mostrar todas las predicciones si se solicita
    if show_all:
        print("\n" + "-" * 60)
        print("TODAS LAS PREDICCIONES:")
        print("-" * 60)
        
        # Crear lista de predicciones ordenadas
        all_preds = []
        for idx, prob in enumerate(predictions[0]):
            all_preds.append((class_names[idx], prob))
        
        all_preds.sort(key=lambda x: x[1], reverse=True)
        
        for fruit, prob in all_preds:
            emoji = fruit_emojis.get(fruit, '🍓')
            bar_length = int(prob * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"{emoji} {fruit:10s} {bar} {prob * 100:6.2f}%")
    
    print("=" * 60 + "\n")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Clasificador de Frutas - Predicción desde línea de comandos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python predict.py imagen.jpg
  python predict.py imagen.jpg --all
  python predict.py imagen.jpg --model models/best_model.h5 --all
        """
    )
    
    parser.add_argument(
        'image',
        type=str,
        help='Ruta a la imagen de fruta a clasificar'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/fruit_classifier.h5',
        help='Ruta al modelo entrenado (default: models/fruit_classifier.h5)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Mostrar todas las predicciones con sus probabilidades'
    )
    
    args = parser.parse_args()
    
    # Verificar que existe la imagen
    if not Path(args.image).exists():
        print(f"❌ Error: No se encontró la imagen '{args.image}'")
        sys.exit(1)
    
    print("\n🍎 CLASIFICADOR DE FRUTAS - CNN 🍌")
    print("=" * 60)
    print(f"📁 Imagen: {args.image}")
    print(f"🧠 Modelo: {args.model}")
    print("=" * 60)
    
    # Cargar modelo
    print("\n⏳ Cargando modelo...")
    model, class_names = load_model_and_classes(args.model)
    
    if model is None:
        sys.exit(1)
    
    print("✅ Modelo cargado exitosamente")
    
    # Realizar predicción
    print("🔍 Analizando imagen...")
    predict_image(model, class_names, args.image, show_all=args.all)


if __name__ == "__main__":
    main()
