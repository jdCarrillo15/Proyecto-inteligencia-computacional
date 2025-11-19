"""
Script de preparación de datos con sistema de cache.
Procesa el dataset y lo guarda para entrenamientos futuros.
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
from utils.aggressive_augmenter import AggressiveAugmenter
from utils.model_metrics import calculate_class_weights


class DatasetProcessor:
    """Procesador de dataset con sistema de cache PKL y balanceo de clases."""
    
    def __init__(self, raw_dataset_path, processed_path, img_size=(100, 100), 
                 apply_balancing=True, target_samples=2500):
        """
        Inicializa el procesador.
        
        Args:
            raw_dataset_path: Ruta al dataset original
            processed_path: Ruta donde se guardará el dataset procesado
            img_size: Tamaño de las imágenes
            apply_balancing: Si True, aplica oversampling para balancear clases
            target_samples: Número objetivo de muestras por clase después de balanceo
        """
        self.raw_dataset_path = Path(raw_dataset_path)
        self.processed_path = Path(processed_path)
        self.img_size = img_size
        self.cache = DataCache()
        self.apply_balancing = apply_balancing
        self.target_samples = target_samples
        
        # Inicializar augmentador agresivo si se requiere balanceo
        if self.apply_balancing:
            self.augmenter = AggressiveAugmenter(
                img_size=img_size,
                target_samples_per_class=target_samples
            )
        
        # Detectar automáticamente las clases del dataset
        self.classes = self._detect_classes()
        
        print(f"📁 Dataset: {self.raw_dataset_path}")
        print(f"📊 Clases detectadas: {len(self.classes)}")
        print(f"⚖️  Balanceo de clases: {'Activado' if apply_balancing else 'Desactivado'}")
    
    def _detect_classes(self):
        """Detecta automáticamente las clases del dataset."""
        # Buscar en la estructura del dataset
        train_path = self.raw_dataset_path / "New Plant Diseases Dataset(Augmented)" / "train"
        
        if train_path.exists():
            # Obtener todas las clases de enfermedades específicas
            all_classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
            
            # Filtrar solo las clases específicas que queremos (15 enfermedades)
            target_classes = [
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
            
            # Filtrar solo las clases que existen en el dataset
            filtered_classes = [cls for cls in target_classes if cls in all_classes]
            
            if filtered_classes:
                return filtered_classes
        
        # Si no existe, buscar directamente en raw
        if self.raw_dataset_path.exists():
            classes = sorted([d.name for d in self.raw_dataset_path.iterdir() 
                            if d.is_dir() and not d.name.startswith('.')])
            if classes:
                return classes
        
        # Default - 15 clases específicas
        return [
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
    
    def prepare_optimized(self, use_cache=True, force_reprocess=False):
        """
        Prepara el dataset de forma optimizada usando cache y balanceo.
        
        Args:
            use_cache: Si True, usa cache PKL si está disponible
            force_reprocess: Si True, fuerza el reprocesamiento aunque exista cache
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test, class_names, class_weights)
        """
        config = {
            'img_size': self.img_size,
            'classes': self.classes,
            'balance': self.apply_balancing,
            'target_samples': self.target_samples if self.apply_balancing else None
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
                    
                    # Calcular class weights
                    class_weights = calculate_class_weights(y_train, len(class_names))
                    
                    return X_train, y_train, X_test, y_test, class_names, class_weights
                else:
                    print("⚠️  Cache encontrado pero test está vacío. Reprocesando con división train/test...")
                    # Limpiar cache inválido
                    cache_key = self.cache._get_cache_key(str(self.raw_dataset_path), config)
                    self.cache.clear(cache_key)
            else:
                print("⚠️  Cache no encontrado. Procesando dataset...")
        
        # Procesar dataset desde cero
        print("\n" + "=" * 60)
        print("🔄 PROCESAMIENTO DE DATASET")
        print("=" * 60)
        print(f"\n📊 CONFIGURACIÓN:")
        print(f"  - Resolución: {self.img_size[0]}x{self.img_size[1]} ({self.img_size[0]*self.img_size[1]:,} píxeles)")
        print(f"  - Clases objetivo: {len(self.classes)}")
        
        # Estimación de tamaño de cache
        pixels = self.img_size[0] * self.img_size[1]
        estimated_mb = (pixels * 3 * 15000) / (1024 * 1024)  # Estimación para ~15k imágenes
        print(f"\n💾 Tamaño estimado de cache: ~{estimated_mb:.0f} MB")
        if pixels > 20000:  # Mayor a ~140x140
            print(f"  ⏱️  Alta resolución - La generación puede tardar 15-25 min")
        
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
        X_train, y_train, class_names = self._load_images(train_dir)
        
        if len(X_train) == 0:
            print("❌ Error: No se cargaron imágenes del directorio de entrenamiento")
            return None
        
        # Aplicar balanceo ANTES de dividir train/test si está activado
        if self.apply_balancing:
            # Contar muestras por clase ANTES del balanceo
            y_train_indices = np.argmax(y_train, axis=1)
            class_counts = {class_names[i]: np.sum(y_train_indices == i) for i in range(len(class_names))}
            
            print(f"\n⚖️  Aplicando balanceo de clases...")
            X_train, y_train = self.augmenter.balance_dataset(X_train, y_train, class_counts)
        
        # Cargar test o dividir train
        if test_dir:
            X_test, y_test, _ = self._load_images(test_dir)
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
            
            # Calcular tamaños
            train_size_mb = (X_train.nbytes + y_train.nbytes) / (1024 * 1024)
            test_size_mb = (X_test.nbytes + y_test.nbytes) / (1024 * 1024)
            
            print(f"  - Train: {X_train.shape[0]} muestras (~{train_size_mb:.1f} MB)")
            self.cache.save(X_train, y_train, class_names, 
                          str(self.raw_dataset_path), config, 'train')
            
            print(f"  - Test: {X_test.shape[0]} muestras (~{test_size_mb:.1f} MB)")
            self.cache.save(X_test, y_test, class_names, 
                          str(self.raw_dataset_path), config, 'test')
            
            print(f"  ✅ Cache guardado: {train_size_mb + test_size_mb:.1f} MB total")
        
        # Calcular class weights para entrenamiento
        print(f"\n⚖️  Calculando pesos de clase...")
        class_weights = calculate_class_weights(y_train, len(class_names))
        
        print(f"\n📊 Pesos de clase calculados:")
        for idx, weight in class_weights.items():
            print(f"  {class_names[idx]:<45} Weight: {weight:.3f}")
        
        print(f"\n✅ Procesamiento completado:")
        print(f"  - Entrenamiento: {X_train.shape[0]} muestras")
        print(f"  - Prueba: {X_test.shape[0]} muestras")
        print(f"  - Clases: {class_names}")
        
        return X_train, y_train, X_test, y_test, class_names, class_weights
    
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
    
    def _load_images(self, directory):
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
        
        # Usar las clases específicas tal como están en el dataset
        class_groups = {}
        for class_dir in all_class_dirs:
            dir_name = class_dir.name
            
            # Buscar coincidencia exacta con las clases objetivo
            if dir_name in self.classes:
                class_groups[dir_name] = [class_dir]
        
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
                
                # Limitar cantidad por clase si es necesario
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
    print("\n🚀 PREPARACIÓN DE DATASET")
    print("=" * 60)
    
    # Configuración
    RAW_DATASET = "dataset/raw"
    PROCESSED_DATASET = "dataset/processed"
    IMG_SIZE = (224, 224)  # Usar 224x224 como está configurado
    APPLY_BALANCING = True  # Activar balanceo de clases
    TARGET_SAMPLES = 2500  # Objetivo de muestras por clase
    
    # Crear procesador
    processor = DatasetProcessor(
        RAW_DATASET, 
        PROCESSED_DATASET, 
        IMG_SIZE,
        apply_balancing=APPLY_BALANCING,
        target_samples=TARGET_SAMPLES
    )
    
    # Opciones
    print("\n⚙️  Opciones:")
    print("  1. Usar cache")
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
        X_train, y_train, X_test, y_test, class_names, class_weights = result
        
        print("\n" + "=" * 80)
        print("✅ DATOS LISTOS PARA ENTRENAMIENTO")
        print("=" * 80)
        print(f"\n📊 Resumen:")
        print(f"  - X_train: {X_train.shape}")
        print(f"  - y_train: {y_train.shape}")
        print(f"  - X_test: {X_test.shape}")
        print(f"  - y_test: {y_test.shape}")
        print(f"  - Clases: {len(class_names)}")
        print(f"  - Class weights: {len(class_weights)} clases")
        
        # Verificar balance final
        y_train_indices = np.argmax(y_train, axis=1)
        final_counts = np.bincount(y_train_indices, minlength=len(class_names))
        final_ratio = np.max(final_counts) / np.min(final_counts) if np.min(final_counts) > 0 else 0
        
        print(f"\n⚖️  Balance final del dataset:")
        print(f"  - Ratio: {final_ratio:.2f}:1")
        if final_ratio <= 2.0:
            print(f"  - ✅ Objetivo alcanzado (≤ 2:1)")
        else:
            print(f"  - ⚠️  Ratio mayor a 2:1")
        
        print("\n💡 Siguiente paso:")
        print("   Ejecuta 'python backend/scripts/train.py' para entrenar")


if __name__ == "__main__":
    main()
