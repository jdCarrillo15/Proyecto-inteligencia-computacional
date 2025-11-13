"""
Script OPTIMIZADO de preparación de datos con soporte de CACHE PKL.
Procesa el dataset una sola vez y lo guarda para entrenamientos futuros.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import sys

# Agregar el directorio backend al path para imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from utils.data_cache import DataCache, create_data_arrays_from_directory


class FastDatasetProcessor:
    """Procesador optimizado de dataset con soporte de cache PKL."""
    
    def __init__(self, raw_dataset_path, processed_path, img_size=(100, 100)):
        """
        Inicializa el procesador.
        
        Args:
            raw_dataset_path: Ruta al dataset original
            processed_path: Ruta donde se guardará el dataset procesado
            img_size: Tamaño de las imágenes
        """
        self.raw_dataset_path = Path(raw_dataset_path)
        self.processed_path = Path(processed_path)
        self.img_size = img_size
        self.cache = DataCache()
        
        # Detectar automáticamente las clases del dataset
        self.classes = self._detect_classes()
        
        print(f"📁 Dataset: {self.raw_dataset_path}")
        print(f"📊 Clases detectadas: {self.classes}")
    
    def _detect_classes(self):
        """Detecta automáticamente las clases del dataset."""
        # Buscar en la estructura del dataset
        train_path = self.raw_dataset_path / "New Plant Diseases Dataset(Augmented)" / "train"
        
        if train_path.exists():
            # Filtrar solo las clases de frutas que nos interesan
            all_classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
            
            # Mapeo de clases del dataset a nuestras categorías
            class_mapping = {
                'Apple': 'Apple',
                'Corn_(maize)': 'Corn',
                'Potato': 'Potato',
                'Tomato': 'Tomato'
            }
            
            # Filtrar clases relevantes
            filtered_classes = []
            for cls in all_classes:
                for key in class_mapping:
                    if cls.startswith(key):
                        if class_mapping[key] not in filtered_classes:
                            filtered_classes.append(class_mapping[key])
            
            return sorted(filtered_classes)
        
        # Si no existe, buscar directamente en raw
        if self.raw_dataset_path.exists():
            classes = sorted([d.name for d in self.raw_dataset_path.iterdir() 
                            if d.is_dir() and not d.name.startswith('.')])
            if classes:
                return classes
        
        # Default
        return ['Apple', 'Corn', 'Potato', 'Tomato']
    
    def prepare_optimized(self, use_cache=True, force_reprocess=False):
        """
        Prepara el dataset de forma optimizada usando cache.
        
        Args:
            use_cache: Si True, usa cache PKL si está disponible
            force_reprocess: Si True, fuerza el reprocesamiento aunque exista cache
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test, class_names)
        """
        config = {
            'img_size': self.img_size,
            'classes': self.classes,
            'balance': False
        }
        
        # Intentar cargar desde cache
        if use_cache and not force_reprocess:
            print("\n🔍 Verificando cache existente...")
            
            train_cache = self.cache.load(str(self.raw_dataset_path), config, 'train')
            test_cache = self.cache.load(str(self.raw_dataset_path), config, 'test')
            
            if train_cache and test_cache:
                X_train, y_train, class_names = train_cache
                X_test, y_test, _ = test_cache
                
                # Validar que el cache tenga datos de test válidos
                if len(X_test) > 0:
                    print("\n✅ ¡CACHE ENCONTRADO! Cargando datos procesados...")
                    print(f"\n📊 Datos cargados desde cache:")
                    print(f"  - Entrenamiento: {X_train.shape[0]} muestras")
                    print(f"  - Prueba: {X_test.shape[0]} muestras")
                    print(f"  - Clases: {class_names}")
                    
                    return X_train, y_train, X_test, y_test, class_names
                else:
                    print("⚠️  Cache encontrado pero test está vacío. Reprocesando con división train/test...")
                    # Limpiar cache inválido
                    cache_key = self.cache._get_cache_key(str(self.raw_dataset_path), config)
                    self.cache.clear(cache_key)
            else:
                print("⚠️  Cache no encontrado. Procesando dataset...")
        
        # Procesar dataset desde cero
        print("\n" + "=" * 60)
        print("🔄 PROCESAMIENTO RÁPIDO DE DATASET")
        print("=" * 60)
        
        # Buscar directorios de train y test
        train_dir = self._find_train_dir()
        test_dir = self._find_test_dir()
        
        if not train_dir:
            print("❌ Error: No se encontró directorio de entrenamiento")
            print("\nBusca el dataset en alguna de estas ubicaciones:")
            print("  - dataset/raw/New Plant Diseases Dataset(Augmented)/train")
            print("  - dataset/raw/train")
            print("  - dataset/processed/train")
            return None
        
        print(f"\n📂 Train: {train_dir}")
        if test_dir:
            print(f"📂 Test: {test_dir}")
        else:
            print(f"📂 Test: No encontrado (se dividirá train en 80/20)")
        
        # Cargar y procesar imágenes
        print("\n⏳ Procesando imágenes...")
        X_train, y_train, class_names = self._load_images_fast(train_dir)
        
        if len(X_train) == 0:
            print("❌ Error: No se cargaron imágenes del directorio de entrenamiento")
            return None
        
        # Cargar test o dividir train
        if test_dir:
            X_test, y_test, _ = self._load_images_fast(test_dir)
            # Si test está vacío, dividir train
            if len(X_test) == 0:
                print("\n⚠️  Test vacío, dividiendo datos de train (80/20)...")
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
                )
        else:
            # No hay directorio test, dividir train
            print("\n⚠️  Directorio test no encontrado, dividiendo datos de train (80/20)...")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
            )
        
        # Guardar en cache para futuros entrenamientos
        if use_cache:
            print("\n💾 Guardando en cache para futuros entrenamientos...")
            self.cache.save(X_train, y_train, class_names, 
                          str(self.raw_dataset_path), config, 'train')
            self.cache.save(X_test, y_test, class_names, 
                          str(self.raw_dataset_path), config, 'test')
        
        print(f"\n✅ Procesamiento completado:")
        print(f"  - Entrenamiento: {X_train.shape[0]} muestras")
        print(f"  - Prueba: {X_test.shape[0]} muestras")
        print(f"  - Clases: {class_names}")
        
        return X_train, y_train, X_test, y_test, class_names
    
    def _find_train_dir(self):
        """Encuentra el directorio de entrenamiento."""
        possible_paths = [
            self.raw_dataset_path / "New Plant Diseases Dataset(Augmented)" / "train",
            self.raw_dataset_path / "train",
            self.processed_path / "train"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def _find_test_dir(self):
        """Encuentra el directorio de prueba."""
        possible_paths = [
            self.raw_dataset_path / "New Plant Diseases Dataset(Augmented)" / "test",
            self.raw_dataset_path / "test",
            self.processed_path / "test"
        ]
        
        for path in possible_paths:
            if path.exists():
                # Verificar que tenga subdirectorios
                subdirs = [d for d in path.iterdir() if d.is_dir()]
                if subdirs:
                    return path
        
        return None
    
    def _load_images_fast(self, directory):
        """
        Carga imágenes de forma rápida y eficiente.
        
        Args:
            directory: Directorio con subdirectorios por clase
            
        Returns:
            tuple: (X, y, class_names)
        """
        directory = Path(directory)
        
        # Obtener subdirectorios (clases)
        all_class_dirs = sorted([d for d in directory.iterdir() if d.is_dir()])
        
        # Agrupar por categoría principal
        class_groups = {}
        for class_dir in all_class_dirs:
            name = class_dir.name
            
            # Extraer categoría principal
            for target_class in self.classes:
                if target_class.lower() in name.lower():
                    if target_class not in class_groups:
                        class_groups[target_class] = []
                    class_groups[target_class].append(class_dir)
                    break
        
        X_list = []
        y_list = []
        class_names = sorted(class_groups.keys())
        
        # Si no hay clases, retornar vacío
        if not class_names:
            print(f"\n⚠️  No se encontraron clases en {directory.name}")
            return np.array([]), np.array([]), []
        
        print(f"\n📸 Cargando imágenes desde {directory.name}...")
        
        for class_idx, class_name in enumerate(class_names):
            class_dirs = class_groups.get(class_name, [])
            if not class_dirs:
                continue
            total_images = 0
            
            for class_dir in class_dirs:
                # Obtener todas las imágenes
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_files.extend(list(class_dir.glob(ext)))
                
                # Limitar cantidad por clase para entrenamiento rápido
                max_per_class = 1000  # Ajusta según necesidad
                image_files = image_files[:max_per_class]
                
                for img_path in image_files:
                    try:
                        # Cargar y procesar imagen
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        
                        # Convertir BGR a RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Redimensionar
                        img = cv2.resize(img, self.img_size)
                        
                        # Normalizar
                        img = img.astype(np.float32) / 255.0
                        
                        X_list.append(img)
                        y_list.append(class_idx)
                        total_images += 1
                    
                    except Exception as e:
                        continue
            
            print(f"  ✅ {class_name}: {total_images} imágenes")
        
        # Convertir a arrays numpy
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        
        # Convertir labels a one-hot encoding
        y_onehot = np.zeros((len(y), len(class_names)), dtype=np.float32)
        y_onehot[np.arange(len(y)), y] = 1
        
        return X, y_onehot, class_names
    
    def clear_cache(self):
        """Limpia el cache."""
        print("\n🗑️  Limpiando cache...")
        self.cache.clear()
    
    def show_cache_info(self):
        """Muestra información del cache."""
        self.cache.print_info()


def main():
    """Función principal para preparar el dataset."""
    print("\n🚀 PREPARACIÓN RÁPIDA DE DATASET CON CACHE PKL")
    print("=" * 60)
    
    # Configuración
    RAW_DATASET = "dataset/raw"
    PROCESSED_DATASET = "dataset/processed"
    IMG_SIZE = (100, 100)
    
    # Crear procesador
    processor = FastDatasetProcessor(RAW_DATASET, PROCESSED_DATASET, IMG_SIZE)
    
    # Opciones
    print("\n⚙️  Opciones:")
    print("  1. Usar cache (rápido)")
    print("  2. Forzar reprocesamiento")
    print("  3. Ver información del cache")
    print("  4. Limpiar cache")
    
    # Para automatizar, usar opción 1
    option = input("\nSelecciona una opción [1]: ").strip() or "1"
    
    if option == "3":
        processor.show_cache_info()
        return
    
    if option == "4":
        processor.clear_cache()
        return
    
    force_reprocess = (option == "2")
    
    # Preparar datos
    result = processor.prepare_optimized(
        use_cache=True,
        force_reprocess=force_reprocess
    )
    
    if result:
        X_train, y_train, X_test, y_test, class_names = result
        
        print("\n" + "=" * 60)
        print("✅ DATOS LISTOS PARA ENTRENAMIENTO")
        print("=" * 60)
        print(f"\n📊 Resumen:")
        print(f"  - X_train: {X_train.shape}")
        print(f"  - y_train: {y_train.shape}")
        print(f"  - X_test: {X_test.shape}")
        print(f"  - y_test: {y_test.shape}")
        print(f"  - Clases: {class_names}")
        
        print("\n💡 Siguiente paso:")
        print("   Ejecuta 'python scripts/train_model_fast.py' para entrenar")


if __name__ == "__main__":
    main()
