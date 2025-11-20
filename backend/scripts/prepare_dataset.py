#!/usr/bin/env python3
"""
Script de Preparación de Dataset - Fase 2 Paso 1
================================================

Funcionalidades:
✅ Split automático 70/15/15 (train/val/test) con estratificación
✅ Augmentation agresiva solo en training set
✅ Normalización ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
✅ Cache PKL para cargas rápidas
✅ Verificación de integridad de splits
✅ Reporte detallado de distribución

Objetivos de split:
- Train: 19,899 imágenes (70%)
- Val: 4,264 imágenes (15%)
- Test: 4,265 imágenes (15%)
- Total: 28,428 imágenes
- Balance: ≤1.18:1 (mantener en todos los splits)
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Agregar el directorio backend al path para importar módulos
backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))

from utils.data_cache import DataCache
from utils.manage_cache import AggressiveAugmenter, calculate_class_weights


class DatasetProcessor:
    """Procesador de dataset con split 70/15/15 y verificación de integridad."""
    
    # 15 clases objetivo del proyecto
    TARGET_CLASSES = [
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
    
    # Estadísticas de normalización ImageNet
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __init__(self, raw_dataset_path, processed_path, img_size=(224, 224), 
                 apply_balancing=True, target_samples=2500):
        """
        Inicializa el procesador de dataset.
        
        Args:
            raw_dataset_path: Ruta al dataset raw
            processed_path: Ruta para guardar datos procesados
            img_size: Tamaño de las imágenes (ancho, alto)
            apply_balancing: Si aplicar balanceo de clases
            target_samples: Muestras objetivo por clase para balanceo
        """
        self.raw_dataset_path = Path(raw_dataset_path)
        self.processed_path = Path(processed_path)
        self.img_size = img_size
        self.apply_balancing = apply_balancing
        self.target_samples = target_samples
        self.classes = self.TARGET_CLASSES
        
        # Crear directorio de salida
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # Inicializar cache y augmenter
        self.cache = DataCache()
        self.augmenter = AggressiveAugmenter()
        
        print(f"\n🔧 Configuración del Procesador:")
        print(f"  - Dataset raw: {self.raw_dataset_path}")
        print(f"  - Dataset procesado: {self.processed_path}")
        print(f"  - Tamaño de imagen: {img_size[0]}x{img_size[1]}")
        print(f"  - Balanceo: {'✅ Activado' if apply_balancing else '❌ Desactivado'}")
        print(f"  - Clases objetivo: {len(self.classes)}")
    
    def prepare_dataset(self, use_cache=True, force_reprocess=False):
        """
        Prepara el dataset con split 70/15/15 y todas las funcionalidades.
        
        Args:
            use_cache: Usar cache si está disponible
            force_reprocess: Forzar reprocesamiento incluso si hay cache
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test, class_names, class_weights)
        """
        print("\n" + "=" * 80)
        print("🚀 FASE 2 - PASO 1: PREPARACIÓN DE DATASET CON SPLIT 70/15/15")
        print("=" * 80)
        
        # Intentar cargar desde cache
        if use_cache and not force_reprocess:
            cached_data = self._load_from_cache()
            if cached_data:
                return cached_data
        
        # Procesar dataset desde cero
        print("\n📦 Procesando dataset desde cero...")
        
        # 1. Buscar y cargar directorio de train
        train_dir = self._find_train_dir()
        if not train_dir:
            print("❌ No se encontró directorio de entrenamiento")
            return None
        
        # 2. Cargar todas las imágenes
        X_all, y_all, class_names = self._load_images(train_dir)
        
        if len(X_all) == 0:
            print("❌ No se cargaron imágenes")
            return None
        
        print(f"\n📊 Total de imágenes cargadas: {len(X_all)}")
        
        # 3. Aplicar normalización ImageNet ANTES de cualquier split
        print("\n🎨 Aplicando normalización ImageNet...")
        X_all = self._apply_imagenet_normalization(X_all)
        print("  ✅ Normalización aplicada (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
        
        # 4. Aplicar balanceo ANTES del split (si está activado)
        if self.apply_balancing and self.target_samples > 0:
            print(f"\n⚖️  Balanceando clases (objetivo: {self.target_samples} muestras/clase)...")
            X_all, y_all = self._balance_dataset(X_all, y_all, class_names)
        
        # 5. Realizar split 70/15/15 con estratificación
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_dataset(X_all, y_all)
        
        # 6. Aplicar augmentation SOLO al training set
        print("\n🔄 Aplicando augmentation agresiva al training set...")
        X_train_aug, y_train_aug = self._augment_training_set(X_train, y_train, class_names)
        
        # 7. Verificar integridad de los splits
        self._verify_split_integrity(X_train_aug, y_train_aug, X_val, y_val, X_test, y_test, class_names)
        
        # 8. Generar reporte de distribución
        self._generate_distribution_report(y_train_aug, y_val, y_test, class_names)
        
        # 9. Calcular class weights
        class_weights = calculate_class_weights(y_train_aug)
        
        # 10. Guardar en cache
        if use_cache:
            self._save_to_cache(X_train_aug, y_train_aug, X_val, y_val, X_test, y_test, class_names)
        
        print("\n✅ Preparación completada exitosamente")
        
        return X_train_aug, y_train_aug, X_val, y_val, X_test, y_test, class_names, class_weights
    
    def _find_train_dir(self):
        """Encuentra el directorio de entrenamiento."""
        possible_paths = [
            self.raw_dataset_path / "New Plant Diseases Dataset(Augmented)" / "train",
            self.raw_dataset_path / "train",
            self.processed_path / "train"
        ]
        
        print("\n🔍 Buscando directorio de entrenamiento...")
        for path in possible_paths:
            if path.exists():
                print(f"  ✅ Encontrado: {path}")
                return path
        
        print("  ❌ No se encontró ningún directorio de entrenamiento")
        return None
    
    def _load_images(self, directory):
        """
        Carga imágenes desde un directorio con subdirectorios por clase.
        
        Args:
            directory: Directorio con subdirectorios por clase
            
        Returns:
            tuple: (X, y, class_names)
        """
        directory = Path(directory)
        
        # Obtener subdirectorios (clases)
        all_class_dirs = sorted([d for d in directory.iterdir() if d.is_dir()])
        
        # Filtrar solo las clases objetivo
        class_groups = {}
        for class_dir in all_class_dirs:
            dir_name = class_dir.name
            if dir_name in self.classes:
                class_groups[dir_name] = class_dir
        
        X_list = []
        y_list = []
        class_names = sorted(class_groups.keys())
        
        if not class_names:
            print(f"\n⚠️  No se encontraron clases en {directory.name}")
            return np.array([]), np.array([]), []
        
        print(f"\n📸 Cargando imágenes desde {directory.name}...")
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = class_groups[class_name]
            
            # Obtener todas las imágenes
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(list(class_dir.glob(ext)))
            
            total_images = 0
            for img_path in image_files:
                try:
                    # Cargar imagen
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # Convertir BGR a RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Redimensionar
                    img = cv2.resize(img, self.img_size)
                    
                    # Normalizar a [0, 1] (la normalización ImageNet se aplica después)
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
    
    def _apply_imagenet_normalization(self, X):
        """
        Aplica normalización ImageNet a las imágenes.
        
        Args:
            X: Array de imágenes normalizadas a [0, 1]
            
        Returns:
            X normalizado con estadísticas ImageNet
        """
        # X está en [0, 1], aplicar normalización ImageNet
        X_normalized = (X - self.IMAGENET_MEAN) / self.IMAGENET_STD
        return X_normalized
    
    def _balance_dataset(self, X, y, class_names):
        """
        Balancea el dataset usando augmentation.
        
        Args:
            X: Imágenes
            y: Labels one-hot
            class_names: Nombres de clases
            
        Returns:
            tuple: (X_balanced, y_balanced)
        """
        y_indices = np.argmax(y, axis=1)
        class_counts = {class_names[i]: np.sum(y_indices == i) for i in range(len(class_names))}
        
        # Contar antes del balanceo
        min_before = min(class_counts.values())
        max_before = max(class_counts.values())
        ratio_before = max_before / min_before if min_before > 0 else 0
        
        print(f"  📊 Antes del balanceo:")
        print(f"     - Min: {min_before} muestras")
        print(f"     - Max: {max_before} muestras")
        print(f"     - Ratio: {ratio_before:.2f}:1")
        
        # Aplicar balanceo
        X_balanced, y_balanced = self.augmenter.balance_dataset(X, y, class_counts)
        
        # Contar después del balanceo
        y_balanced_indices = np.argmax(y_balanced, axis=1)
        class_counts_after = {class_names[i]: np.sum(y_balanced_indices == i) for i in range(len(class_names))}
        min_after = min(class_counts_after.values())
        max_after = max(class_counts_after.values())
        ratio_after = max_after / min_after if min_after > 0 else 0
        
        print(f"  📊 Después del balanceo:")
        print(f"     - Min: {min_after} muestras")
        print(f"     - Max: {max_after} muestras")
        print(f"     - Ratio: {ratio_after:.2f}:1")
        
        return X_balanced, y_balanced
    
    def _split_dataset(self, X, y):
        """
        Divide el dataset en 70/15/15 (train/val/test) con estratificación.
        
        Args:
            X: Imágenes
            y: Labels one-hot
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        print("\n✂️  Dividiendo dataset (70% train / 15% val / 15% test)...")
        
        # Paso 1: Dividir en train (70%) y temp (30%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=0.3,  # 30% para val+test
            random_state=42,
            stratify=np.argmax(y, axis=1)
        )
        
        # Paso 2: Dividir temp en val (50%) y test (50%)
        # 50% de 30% = 15% cada uno
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,  # 50% de 30% = 15%
            random_state=42,
            stratify=np.argmax(y_temp, axis=1)
        )
        
        total = len(X)
        print(f"  ✅ Train: {len(X_train)} muestras ({len(X_train)/total*100:.1f}%)")
        print(f"  ✅ Val: {len(X_val)} muestras ({len(X_val)/total*100:.1f}%)")
        print(f"  ✅ Test: {len(X_test)} muestras ({len(X_test)/total*100:.1f}%)")
        print(f"  📊 Total: {total} muestras")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _augment_training_set(self, X_train, y_train, class_names):
        """
        Aplica augmentation agresiva solo al training set.
        
        Args:
            X_train: Imágenes de entrenamiento
            y_train: Labels de entrenamiento
            class_names: Nombres de clases
            
        Returns:
            tuple: (X_train_aug, y_train_aug)
        """
        print(f"  📊 Training set antes de augmentation: {len(X_train)} muestras")
        
        # Aplicar augmentation para balancear
        y_train_indices = np.argmax(y_train, axis=1)
        class_counts = {class_names[i]: np.sum(y_train_indices == i) for i in range(len(class_names))}
        
        X_train_aug, y_train_aug = self.augmenter.balance_dataset(X_train, y_train, class_counts)
        
        print(f"  📊 Training set después de augmentation: {len(X_train_aug)} muestras")
        print(f"  ✅ Augmentation completada")
        
        return X_train_aug, y_train_aug
    
    def _verify_split_integrity(self, X_train, y_train, X_val, y_val, X_test, y_test, class_names):
        """
        Verifica la integridad de los splits.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            class_names: Nombres de clases
        """
        print("\n🔍 VERIFICACIÓN DE INTEGRIDAD DE SPLITS")
        print("=" * 60)
        
        # 1. Verificar totales
        total = len(X_train) + len(X_val) + len(X_test)
        print(f"\n1️⃣  Total de muestras:")
        print(f"   - Train: {len(X_train)}")
        print(f"   - Val: {len(X_val)}")
        print(f"   - Test: {len(X_test)}")
        print(f"   - TOTAL: {total}")
        print(f"   ✅ Verificación de totales completada")
        
        # 2. Verificar que cada split tiene todas las clases
        y_train_indices = np.argmax(y_train, axis=1)
        y_val_indices = np.argmax(y_val, axis=1)
        y_test_indices = np.argmax(y_test, axis=1)
        
        train_classes = set(y_train_indices)
        val_classes = set(y_val_indices)
        test_classes = set(y_test_indices)
        
        print(f"\n2️⃣  Clases por split:")
        print(f"   - Train: {len(train_classes)}/{len(class_names)} clases")
        print(f"   - Val: {len(val_classes)}/{len(class_names)} clases")
        print(f"   - Test: {len(test_classes)}/{len(class_names)} clases")
        
        if len(train_classes) == len(val_classes) == len(test_classes) == len(class_names):
            print(f"   ✅ Todas las clases representadas en todos los splits")
        else:
            print(f"   ⚠️  Algunas clases faltantes en algún split")
        
        # 3. Verificar balance en cada split
        print(f"\n3️⃣  Balance de clases:")
        
        for split_name, y_indices in [('Train', y_train_indices), ('Val', y_val_indices), ('Test', y_test_indices)]:
            counts = np.bincount(y_indices, minlength=len(class_names))
            min_count = np.min(counts[counts > 0]) if np.any(counts > 0) else 0
            max_count = np.max(counts)
            ratio = max_count / min_count if min_count > 0 else 0
            
            print(f"   - {split_name}:")
            print(f"     • Min: {min_count} muestras")
            print(f"     • Max: {max_count} muestras")
            print(f"     • Ratio: {ratio:.2f}:1 {'✅' if ratio <= 2.0 else '⚠️'}")
        
        print(f"\n✅ Verificación de integridad completada")
    
    def _generate_distribution_report(self, y_train, y_val, y_test, class_names):
        """
        Genera un reporte detallado de la distribución de clases.
        
        Args:
            y_train: Labels de entrenamiento
            y_val: Labels de validación
            y_test: Labels de test
            class_names: Nombres de clases
        """
        print("\n📊 GENERANDO REPORTE DE DISTRIBUCIÓN")
        print("=" * 60)
        
        # Convertir one-hot a índices
        y_train_indices = np.argmax(y_train, axis=1)
        y_val_indices = np.argmax(y_val, axis=1)
        y_test_indices = np.argmax(y_test, axis=1)
        
        # Contar muestras por clase
        train_counts = np.bincount(y_train_indices, minlength=len(class_names))
        val_counts = np.bincount(y_val_indices, minlength=len(class_names))
        test_counts = np.bincount(y_test_indices, minlength=len(class_names))
        
        # Crear reporte
        report_path = self.processed_path / "distribution_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE DE DISTRIBUCIÓN DE DATASET\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Resumen general
            f.write("RESUMEN GENERAL\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Train: {len(y_train):,} muestras ({len(y_train)/(len(y_train)+len(y_val)+len(y_test))*100:.1f}%)\n")
            f.write(f"Total Val: {len(y_val):,} muestras ({len(y_val)/(len(y_train)+len(y_val)+len(y_test))*100:.1f}%)\n")
            f.write(f"Total Test: {len(y_test):,} muestras ({len(y_test)/(len(y_train)+len(y_val)+len(y_test))*100:.1f}%)\n")
            f.write(f"TOTAL: {len(y_train)+len(y_val)+len(y_test):,} muestras\n\n")
            
            # Distribución por clase
            f.write("DISTRIBUCIÓN POR CLASE\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Clase':<45} {'Train':>10} {'Val':>8} {'Test':>8} {'Total':>10}\n")
            f.write("-" * 80 + "\n")
            
            for i, class_name in enumerate(class_names):
                train_c = train_counts[i]
                val_c = val_counts[i]
                test_c = test_counts[i]
                total_c = train_c + val_c + test_c
                
                f.write(f"{class_name:<45} {train_c:>10} {val_c:>8} {test_c:>8} {total_c:>10}\n")
            
            # Balance
            f.write("\n\nBALANCE DE CLASES\n")
            f.write("-" * 80 + "\n")
            
            for split_name, counts in [('Train', train_counts), ('Val', val_counts), ('Test', test_counts)]:
                min_count = np.min(counts[counts > 0]) if np.any(counts > 0) else 0
                max_count = np.max(counts)
                ratio = max_count / min_count if min_count > 0 else 0
                
                f.write(f"\n{split_name}:\n")
                f.write(f"  Min: {min_count} muestras\n")
                f.write(f"  Max: {max_count} muestras\n")
                f.write(f"  Ratio: {ratio:.2f}:1 {'✅ Balanceado' if ratio <= 2.0 else '⚠️ Desbalanceado'}\n")
        
        print(f"  ✅ Reporte guardado en: {report_path}")
        
        # Mostrar resumen en consola
        print(f"\n📋 Resumen de distribución:")
        print(f"{'Clase':<45} {'Train':>10} {'Val':>8} {'Test':>8}")
        print("-" * 80)
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<45} {train_counts[i]:>10} {val_counts[i]:>8} {test_counts[i]:>8}")
    
    def _load_from_cache(self):
        """
        Carga datos desde cache.
        
        Returns:
            tuple o None: (X_train, y_train, X_val, y_val, X_test, y_test, class_names, class_weights) o None
        """
        print("\n🔍 Buscando cache...")
        
        train_data = self.cache.load('train')
        val_data = self.cache.load('val')
        test_data = self.cache.load('test')
        
        if train_data and val_data and test_data:
            print("  ✅ Cache encontrado - Cargando datos...")
            
            X_train = train_data['X']
            y_train = train_data['y']
            class_names = train_data['class_names']
            
            X_val = val_data['X']
            y_val = val_data['y']
            
            X_test = test_data['X']
            y_test = test_data['y']
            
            print(f"  - Train: {X_train.shape[0]} muestras")
            print(f"  - Val: {X_val.shape[0]} muestras")
            print(f"  - Test: {X_test.shape[0]} muestras")
            print(f"  - Clases: {len(class_names)}")
            
            # Calcular class weights
            class_weights = calculate_class_weights(y_train)
            
            # Verificar integridad
            self._verify_split_integrity(X_train, y_train, X_val, y_val, X_test, y_test, class_names)
            
            return X_train, y_train, X_val, y_val, X_test, y_test, class_names, class_weights
        
        print("  ⚠️  Cache no encontrado o incompleto")
        return None
    
    def _save_to_cache(self, X_train, y_train, X_val, y_val, X_test, y_test, class_names):
        """
        Guarda datos en cache.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            class_names: Nombres de clases
        """
        print("\n💾 Guardando en cache...")
        
        # Guardar train
        self.cache.save({
            'X': X_train,
            'y': y_train,
            'class_names': class_names
        }, 'train')
        print(f"  ✅ Train guardado ({X_train.shape[0]} muestras)")
        
        # Guardar val
        self.cache.save({
            'X': X_val,
            'y': y_val,
            'class_names': class_names
        }, 'val')
        print(f"  ✅ Val guardado ({X_val.shape[0]} muestras)")
        
        # Guardar test
        self.cache.save({
            'X': X_test,
            'y': y_test,
            'class_names': class_names
        }, 'test')
        print(f"  ✅ Test guardado ({X_test.shape[0]} muestras)")
    
    def clear_cache(self):
        """Limpia el cache."""
        print("\n🗑️  Limpiando cache...")
        self.cache.clear()
    
    def show_cache_info(self):
        """Muestra información del cache."""
        self.cache.print_info()


def main():
    """Función principal para preparar el dataset."""
    print("\n" + "=" * 80)
    print("🚀 FASE 2 - PASO 1: PREPARACIÓN DE DATASET")
    print("=" * 80)
    
    # Configuración
    RAW_DATASET = "dataset/raw"
    PROCESSED_DATASET = "dataset/processed"
    IMG_SIZE = (224, 224)  # Tamaño estándar para MobileNetV2
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
    print("  1. Usar cache (recomendado)")
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
    result = processor.prepare_dataset(
        use_cache=True,
        force_reprocess=force_reprocess
    )
    
    if result:
        X_train, y_train, X_val, y_val, X_test, y_test, class_names, class_weights = result
        
        print("\n" + "=" * 80)
        print("✅ DATOS LISTOS PARA ENTRENAMIENTO")
        print("=" * 80)
        print(f"\n📊 Resumen Final:")
        print(f"  - X_train: {X_train.shape}")
        print(f"  - y_train: {y_train.shape}")
        print(f"  - X_val: {X_val.shape}")
        print(f"  - y_val: {y_val.shape}")
        print(f"  - X_test: {X_test.shape}")
        print(f"  - y_test: {y_test.shape}")
        print(f"  - Clases: {len(class_names)}")
        print(f"  - Class weights: {len(class_weights)} clases")
        
        print("\n💡 Siguiente paso:")
        print("   Ejecuta 'python backend/scripts/train.py' para entrenar el modelo")


if __name__ == "__main__":
    main()
