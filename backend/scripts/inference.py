#!/usr/bin/env python3
"""
Script de Inferencia para Producción
====================================
Realiza predicciones optimizadas con el modelo entrenado.

Características:
✅ Carga rápida del modelo
✅ Preprocessing optimizado
✅ Predicciones en batch
✅ Manejo de errores robusto
✅ CLI y API programática

Uso:
    # Predicción única
    python backend/scripts/inference.py --image path/to/image.jpg
    
    # Predicción en batch
    python backend/scripts/inference.py --batch path/to/images/
    
    # Como módulo
    from backend.scripts.inference import PlantDiseasePredictor
    predictor = PlantDiseasePredictor()
    result = predictor.predict("image.jpg")
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
from PIL import Image
import argparse

# Añadir backend al path
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import tensorflow as tf
from tensorflow import keras

from config import IMG_SIZE, CLASSES


class PlantDiseasePredictor:
    """Predictor optimizado para detección de enfermedades en plantas."""
    
    def __init__(self, model_path: str = None):
        """
        Inicializa el predictor.
        
        Args:
            model_path: Ruta al modelo entrenado (default: models/best_model.keras)
        """
        if model_path is None:
            model_path = backend_dir.parent / 'models' / 'best_model.keras'
        
        self.model_path = Path(model_path)
        self.model = None
        self.class_names = CLASSES
        self.img_size = IMG_SIZE
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo desde disco."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
        
        print(f"📂 Cargando modelo desde {self.model_path}...")
        start_time = time.time()
        
        self.model = keras.models.load_model(self.model_path)
        
        load_time = time.time() - start_time
        print(f"✅ Modelo cargado en {load_time:.3f}s")
        print(f"   • Parámetros: {self.model.count_params():,}")
        print(f"   • Clases: {len(self.class_names)}")
    
    def preprocess_image(self, image_path: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocesa una imagen para predicción.
        
        Args:
            image_path: Ruta a la imagen, array numpy o PIL Image
        
        Returns:
            Array numpy con forma (1, 224, 224, 3) normalizado
        """
        # Cargar imagen si es ruta
        if isinstance(image_path, (str, Path)):
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
            img = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, Image.Image):
            img = image_path.convert('RGB')
        elif isinstance(image_path, np.ndarray):
            img = Image.fromarray(image_path.astype('uint8'), 'RGB')
        else:
            raise ValueError(f"Tipo de entrada no soportado: {type(image_path)}")
        
        # Redimensionar
        img = img.resize(self.img_size)
        
        # Convertir a array y normalizar
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Añadir dimensión de batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image: Union[str, Path, np.ndarray, Image.Image],
                top_k: int = 5) -> Dict[str, Any]:
        """
        Realiza predicción para una imagen.
        
        Args:
            image: Imagen a predecir
            top_k: Número de predicciones top a retornar
        
        Returns:
            Diccionario con resultados de predicción:
            {
                'predicted_class': str,
                'confidence': float,
                'top_predictions': List[Dict],
                'inference_time_ms': float,
                'all_probabilities': Dict[str, float]
            }
        """
        # Preprocesar
        start_time = time.time()
        img_array = self.preprocess_image(image)
        preprocess_time = (time.time() - start_time) * 1000
        
        # Predecir
        start_time = time.time()
        predictions = self.model.predict(img_array, verbose=0)[0]
        inference_time = (time.time() - start_time) * 1000
        
        # Obtener top-k predicciones
        top_indices = np.argsort(predictions)[::-1][:top_k]
        top_predictions = [
            {
                'class': self.class_names[idx],
                'confidence': float(predictions[idx]),
                'display_name': self._format_class_name(self.class_names[idx])
            }
            for idx in top_indices
        ]
        
        # Predicción principal
        predicted_idx = top_indices[0]
        predicted_class = self.class_names[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        # Todas las probabilidades
        all_probs = {
            self.class_names[i]: float(predictions[i])
            for i in range(len(self.class_names))
        }
        
        return {
            'predicted_class': predicted_class,
            'display_name': self._format_class_name(predicted_class),
            'confidence': confidence,
            'top_predictions': top_predictions,
            'preprocessing_time_ms': preprocess_time,
            'inference_time_ms': inference_time,
            'total_time_ms': preprocess_time + inference_time,
            'all_probabilities': all_probs
        }
    
    def predict_batch(self, images: List[Union[str, Path]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Realiza predicciones en batch para múltiples imágenes.
        
        Args:
            images: Lista de rutas a imágenes
            batch_size: Tamaño del batch para procesamiento
        
        Returns:
            Lista de diccionarios con resultados
        """
        results = []
        
        print(f"🔄 Procesando {len(images)} imágenes en batches de {batch_size}...")
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_arrays = []
            
            # Preprocesar batch
            for img_path in batch_images:
                try:
                    img_array = self.preprocess_image(img_path)
                    batch_arrays.append(img_array[0])  # Remover dimensión de batch
                except Exception as e:
                    print(f"⚠️  Error procesando {img_path}: {e}")
                    results.append({
                        'image': str(img_path),
                        'error': str(e)
                    })
                    continue
            
            if not batch_arrays:
                continue
            
            # Predecir batch
            batch_arrays = np.array(batch_arrays)
            start_time = time.time()
            predictions = self.model.predict(batch_arrays, verbose=0)
            batch_time = (time.time() - start_time) * 1000
            
            # Procesar resultados
            for j, (img_path, pred) in enumerate(zip(batch_images, predictions)):
                top_idx = np.argmax(pred)
                results.append({
                    'image': str(img_path),
                    'predicted_class': self.class_names[top_idx],
                    'display_name': self._format_class_name(self.class_names[top_idx]),
                    'confidence': float(pred[top_idx]),
                    'inference_time_ms': batch_time / len(batch_arrays)
                })
            
            print(f"   ✅ Procesadas {min(i + batch_size, len(images))}/{len(images)} imágenes")
        
        return results
    
    def _format_class_name(self, class_name: str) -> str:
        """
        Formatea el nombre de clase para display.
        
        Args:
            class_name: Nombre de clase raw (ej: 'Tomato___Late_blight')
        
        Returns:
            Nombre formateado (ej: 'Tomato - Late blight')
        """
        parts = class_name.split('___')
        if len(parts) == 2:
            crop, condition = parts
            crop = crop.replace('_', ' ')
            condition = condition.replace('_', ' ')
            return f"{crop} - {condition}"
        return class_name.replace('_', ' ')
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna información del modelo.
        
        Returns:
            Diccionario con info del modelo
        """
        return {
            'model_path': str(self.model_path),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'num_parameters': int(self.model.count_params()),
            'img_size': self.img_size
        }


def main():
    """Función principal para CLI."""
    parser = argparse.ArgumentParser(
        description='Inferencia de enfermedades en plantas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Predicción única
  python backend/scripts/inference.py --image dataset/test/Tomato___Late_blight/image1.jpg
  
  # Predicción con top-5
  python backend/scripts/inference.py --image image.jpg --top-k 5
  
  # Batch de imágenes
  python backend/scripts/inference.py --batch dataset/test/Tomato___Late_blight/
  
  # Información del modelo
  python backend/scripts/inference.py --info
        """
    )
    
    parser.add_argument('--model', type=str, default=None,
                       help='Ruta al modelo (default: models/best_model.keras)')
    parser.add_argument('--image', type=str,
                       help='Ruta a una imagen para predicción')
    parser.add_argument('--batch', type=str,
                       help='Directorio con imágenes para batch prediction')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Número de top predicciones a mostrar (default: 5)')
    parser.add_argument('--info', action='store_true',
                       help='Mostrar información del modelo')
    parser.add_argument('--json', action='store_true',
                       help='Salida en formato JSON')
    
    args = parser.parse_args()
    
    # Inicializar predictor
    try:
        predictor = PlantDiseasePredictor(args.model)
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return 1
    
    # Mostrar info del modelo
    if args.info:
        info = predictor.get_model_info()
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print("\n📊 INFORMACIÓN DEL MODELO")
            print("=" * 50)
            print(f"Ruta: {info['model_path']}")
            print(f"Clases: {info['num_classes']}")
            print(f"Parámetros: {info['num_parameters']:,}")
            print(f"Input shape: {info['input_shape']}")
            print(f"Output shape: {info['output_shape']}")
        return 0
    
    # Predicción única
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Imagen no encontrada: {image_path}")
            return 1
        
        print(f"\n🔍 Prediciendo: {image_path.name}")
        print("=" * 50)
        
        result = predictor.predict(image_path, top_k=args.top_k)
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n✅ PREDICCIÓN: {result['display_name']}")
            print(f"📊 Confianza: {result['confidence']*100:.2f}%")
            print(f"⏱️  Tiempo total: {result['total_time_ms']:.2f}ms")
            print(f"   • Preprocessing: {result['preprocessing_time_ms']:.2f}ms")
            print(f"   • Inferencia: {result['inference_time_ms']:.2f}ms")
            
            print(f"\n📋 Top {args.top_k} Predicciones:")
            for i, pred in enumerate(result['top_predictions'], 1):
                print(f"   {i}. {pred['display_name']}: {pred['confidence']*100:.2f}%")
        
        return 0
    
    # Predicción en batch
    if args.batch:
        batch_dir = Path(args.batch)
        if not batch_dir.exists():
            print(f"❌ Directorio no encontrado: {batch_dir}")
            return 1
        
        # Buscar imágenes
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        images = [f for f in batch_dir.iterdir() 
                 if f.is_file() and f.suffix in image_extensions]
        
        if not images:
            print(f"❌ No se encontraron imágenes en {batch_dir}")
            return 1
        
        print(f"\n🔍 Procesando {len(images)} imágenes de {batch_dir}")
        print("=" * 50)
        
        results = predictor.predict_batch(images)
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\n✅ Completado! {len(results)} predicciones")
            
            # Resumen
            correct_count = sum(1 for r in results if 'error' not in r)
            error_count = len(results) - correct_count
            
            if error_count > 0:
                print(f"⚠️  {error_count} errores durante procesamiento")
            
            # Mostrar algunas predicciones
            print(f"\n📋 Primeras 10 predicciones:")
            for i, result in enumerate(results[:10], 1):
                if 'error' in result:
                    print(f"   {i}. {Path(result['image']).name}: ERROR - {result['error']}")
                else:
                    img_name = Path(result['image']).name
                    print(f"   {i}. {img_name}: {result['display_name']} ({result['confidence']*100:.1f}%)")
        
        return 0
    
    # Sin argumentos
    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
