"""
Sistema de oversampling con data augmentation agresiva para balancear clases minoritarias.
Implementa augmentation intensiva para alcanzar ratio de balance ≤ 2:1
"""

import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys

# Agregar el directorio backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from config import IMG_SIZE, CLASSES, NUM_CLASSES


class AggressiveAugmenter:
    """
    Data augmentation agresiva para clases minoritarias.
    Aplica transformaciones intensivas para generar variaciones realistas.
    """
    
    def __init__(self, img_size=IMG_SIZE, target_samples_per_class=2500):
        """
        Inicializa el augmentador agresivo.
        
        Args:
            img_size: Tamaño objetivo de las imágenes (width, height)
            target_samples_per_class: Número objetivo de muestras por clase después de oversampling
        """
        self.img_size = img_size
        self.target_samples_per_class = target_samples_per_class
        
        # Configurar augmentation agresiva
        self.augmenter = ImageDataGenerator(
            rotation_range=45,           # Rotación ±45 grados (más agresivo que 20)
            width_shift_range=0.3,       # Desplazamiento horizontal (más que 0.2)
            height_shift_range=0.3,      # Desplazamiento vertical (más que 0.2)
            shear_range=0.3,             # Shear transformation (más que 0.2)
            zoom_range=0.3,              # Zoom 0.7-1.3x (más amplio que 0.2)
            horizontal_flip=True,        # Flip horizontal
            vertical_flip=True,          # Flip vertical (nuevo, útil para hojas)
            brightness_range=[0.7, 1.3], # Brillo ±30%
            fill_mode='reflect',         # Modo de relleno (mejor que 'nearest')
            channel_shift_range=20.0,    # Cambio de color por canal
        )
    
    def add_gaussian_noise(self, image, mean=0, std=0.05):
        """
        Agrega ruido gaussiano a la imagen.
        
        Args:
            image: Imagen de entrada (0-1 range)
            mean: Media del ruido
            std: Desviación estándar del ruido
            
        Returns:
            Imagen con ruido
        """
        noise = np.random.normal(mean, std, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0.0, 1.0)
    
    def random_crop_and_resize(self, image, crop_factor=0.8):
        """
        Crop aleatorio seguido de resize.
        
        Args:
            image: Imagen de entrada
            crop_factor: Factor de crop (0.8 = mantener 80% de la imagen)
            
        Returns:
            Imagen cropped y resized
        """
        h, w = image.shape[:2]
        
        # Calcular tamaño del crop
        crop_h = int(h * crop_factor)
        crop_w = int(w * crop_factor)
        
        # Posición aleatoria del crop
        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)
        
        # Crop
        cropped = image[top:top+crop_h, left:left+crop_w]
        
        # Resize al tamaño original
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def augment_image(self, image, num_augmentations=5, add_noise=True, add_crop=True):
        """
        Genera múltiples versiones aumentadas de una imagen.
        
        Args:
            image: Imagen original (numpy array, 0-1 range)
            num_augmentations: Número de versiones aumentadas a generar
            add_noise: Si True, agrega ruido gaussiano
            add_crop: Si True, aplica random crop
            
        Returns:
            Lista de imágenes aumentadas
        """
        augmented_images = []
        
        # Expandir dimensiones para ImageDataGenerator
        img_expanded = np.expand_dims(image, 0)
        
        # Generar augmentaciones
        aug_iter = self.augmenter.flow(img_expanded, batch_size=1)
        
        for i in range(num_augmentations):
            # Obtener imagen aumentada
            aug_img = next(aug_iter)[0]
            
            # Aplicar transformaciones adicionales aleatoriamente
            if add_noise and np.random.random() > 0.5:
                aug_img = self.add_gaussian_noise(aug_img, std=np.random.uniform(0.01, 0.05))
            
            if add_crop and np.random.random() > 0.5:
                crop_factor = np.random.uniform(0.75, 0.95)
                aug_img = self.random_crop_and_resize(aug_img, crop_factor)
            
            augmented_images.append(aug_img)
        
        return augmented_images
    
    def balance_dataset(self, X, y, class_counts):
        """
        Balancea el dataset mediante oversampling con augmentation agresiva.
        
        Args:
            X: Array de imágenes (N, H, W, C)
            y: Array de labels (N, num_classes) one-hot encoded
            class_counts: Diccionario con conteo por clase
            
        Returns:
            X_balanced, y_balanced: Dataset balanceado
        """
        print("\n" + "=" * 80)
        print("AGGRESSIVE OVERSAMPLING AND AUGMENTATION")
        print("=" * 80)
        
        # Convertir y a índices si es one-hot
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_indices = np.argmax(y, axis=1)
        else:
            y_indices = y
        
        # Calcular estadísticas
        max_samples = max(class_counts.values())
        min_samples = min(class_counts.values())
        avg_samples = np.mean(list(class_counts.values()))
        
        print(f"\nOriginal dataset statistics:")
        print(f"  Max samples per class: {max_samples}")
        print(f"  Min samples per class: {min_samples}")
        print(f"  Average samples per class: {avg_samples:.0f}")
        print(f"  Original balance ratio: {max_samples / min_samples:.2f}:1")
        
        # Determinar target basado en percentil 75
        target = max(self.target_samples_per_class, int(np.percentile(list(class_counts.values()), 75)))
        
        print(f"\nTarget samples per class: {target}")
        
        X_balanced = []
        y_balanced = []
        
        for class_idx in range(NUM_CLASSES):
            # Obtener muestras de esta clase
            class_mask = (y_indices == class_idx)
            X_class = X[class_mask]
            y_class = y[class_mask]
            
            current_count = len(X_class)
            class_name = CLASSES[class_idx]
            
            # Agregar todas las muestras originales
            X_balanced.append(X_class)
            y_balanced.append(y_class)
            
            # Calcular cuántas muestras adicionales necesitamos
            needed = target - current_count
            
            if needed > 0:
                # Calcular factor de augmentation
                aug_factor = int(np.ceil(needed / current_count))
                
                print(f"\n  {class_name}:")
                print(f"    Original: {current_count} samples")
                print(f"    Target: {target} samples")
                print(f"    Needed: {needed} samples")
                print(f"    Augmentation factor: {aug_factor}x")
                
                # Generar muestras aumentadas
                augmented_samples = []
                augmented_labels = []
                
                samples_generated = 0
                
                for original_img, original_label in zip(X_class, y_class):
                    # Determinar cuántas augmentaciones hacer para esta imagen
                    num_augs = min(aug_factor, (needed - samples_generated) // current_count + 1)
                    
                    if samples_generated >= needed:
                        break
                    
                    # Generar augmentaciones
                    aug_images = self.augment_image(
                        original_img, 
                        num_augmentations=num_augs,
                        add_noise=True,
                        add_crop=True
                    )
                    
                    for aug_img in aug_images:
                        augmented_samples.append(aug_img)
                        augmented_labels.append(original_label)
                        samples_generated += 1
                        
                        if samples_generated >= needed:
                            break
                
                # Agregar muestras aumentadas
                if augmented_samples:
                    X_balanced.append(np.array(augmented_samples))
                    y_balanced.append(np.array(augmented_labels))
                    
                    print(f"    Generated: {samples_generated} augmented samples")
                    print(f"    Final: {current_count + samples_generated} samples")
            else:
                print(f"\n  {class_name}: {current_count} samples (no augmentation needed)")
        
        # Concatenar todos los arrays
        X_balanced = np.concatenate(X_balanced, axis=0)
        y_balanced = np.concatenate(y_balanced, axis=0)
        
        # Shuffle
        indices = np.arange(len(X_balanced))
        np.random.shuffle(indices)
        X_balanced = X_balanced[indices]
        y_balanced = y_balanced[indices]
        
        # Calcular estadísticas finales
        y_balanced_indices = np.argmax(y_balanced, axis=1) if len(y_balanced.shape) > 1 else y_balanced
        final_counts = np.bincount(y_balanced_indices, minlength=NUM_CLASSES)
        
        final_max = np.max(final_counts)
        final_min = np.min(final_counts)
        final_ratio = final_max / final_min if final_min > 0 else 0
        
        print("\n" + "=" * 80)
        print("BALANCING RESULTS")
        print("=" * 80)
        print(f"\nFinal dataset statistics:")
        print(f"  Total samples: {len(X_balanced):,} (original: {len(X):,})")
        print(f"  Samples per class: {final_min} - {final_max}")
        print(f"  Average per class: {np.mean(final_counts):.0f}")
        print(f"  Final balance ratio: {final_ratio:.2f}:1")
        
        if final_ratio <= 2.0:
            print(f"\n✅ Balance ratio ≤ 2:1 achieved!")
        else:
            print(f"\n⚠️  Balance ratio > 2:1 (target not met)")
        
        print("\nPer-class distribution:")
        for idx, count in enumerate(final_counts):
            print(f"  {CLASSES[idx]:<45} {count:>5} samples")
        
        return X_balanced, y_balanced


def create_balanced_augmentation_pipeline(
    img_size=IMG_SIZE,
    rotation_range=45,
    width_shift=0.3,
    height_shift=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=(0.7, 1.3)
):
    """
    Crea un pipeline de augmentation configurado para balancear clases.
    
    Returns:
        ImageDataGenerator configurado
    """
    return ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift,
        height_shift_range=height_shift,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        brightness_range=brightness_range,
        shear_range=0.3,
        channel_shift_range=20.0,
        fill_mode='reflect',
        preprocessing_function=None  # Se puede agregar custom preprocessing aquí
    )


if __name__ == "__main__":
    print("Aggressive Augmenter - Utility Module")
    print("Este módulo debe ser importado en prepare_dataset.py")
    print("\nUsage:")
    print("  from utils.aggressive_augmenter import AggressiveAugmenter")
    print("  augmenter = AggressiveAugmenter(target_samples_per_class=2500)")
    print("  X_balanced, y_balanced = augmenter.balance_dataset(X, y, class_counts)")
