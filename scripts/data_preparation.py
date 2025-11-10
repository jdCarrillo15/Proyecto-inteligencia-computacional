"""
Script de limpieza y preparación de datos para el dataset de frutas.
Incluye verificación, limpieza, redimensionamiento y visualizaciones.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
import shutil
from collections import Counter
import random

class DatasetCleaner:
    def __init__(self, dataset_path, output_path, img_size=(100, 100)):
        """
        Inicializa el limpiador de dataset.
        
        Args:
            dataset_path: Ruta al dataset original
            output_path: Ruta donde se guardará el dataset limpio
            img_size: Tamaño objetivo para las imágenes (ancho, alto)
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.img_size = img_size
        self.classes = ['manzana', 'banano', 'mango', 'naranja', 'pera']
        self.stats = {
            'total_images': 0,
            'corrupted': 0,
            'invalid_dimensions': 0,
            'processed': 0,
            'class_distribution': {}
        }
        
    def create_output_structure(self):
        """Crea la estructura de carpetas para el dataset limpio."""
        for split in ['train', 'test']:
            for class_name in self.classes:
                output_dir = self.output_path / split / class_name
                output_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear carpeta para visualizaciones
        viz_dir = self.output_path / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
    def is_valid_image(self, img_path):
        """
        Verifica si una imagen es válida.
        
        Args:
            img_path: Ruta a la imagen
            
        Returns:
            tuple: (es_válida, imagen_array o None)
        """
        try:
            # Intentar abrir con PIL
            img = Image.open(img_path)
            img.verify()  # Verificar integridad
            
            # Reabrir para procesamiento (verify() cierra el archivo)
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Verificar que tenga dimensiones válidas
            if img_array.size == 0:
                return False, None
            
            # Convertir a RGB si es necesario
            if len(img_array.shape) == 2:  # Escala de grises
                img = img.convert('RGB')
                img_array = np.array(img)
            elif img_array.shape[2] == 4:  # RGBA
                img = img.convert('RGB')
                img_array = np.array(img)
            
            return True, img_array
            
        except Exception as e:
            print(f"Error al procesar {img_path}: {str(e)}")
            return False, None
    
    def resize_and_normalize(self, img_array):
        """
        Redimensiona y normaliza una imagen.
        
        Args:
            img_array: Array numpy de la imagen
            
        Returns:
            Array numpy redimensionado y normalizado
        """
        # Redimensionar
        img_resized = cv2.resize(img_array, self.img_size)
        
        # Normalizar valores entre 0 y 1
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized
    
    def augment_image(self, img_array):
        """
        Aplica data augmentation a una imagen.
        
        Args:
            img_array: Array numpy de la imagen (0-255)
            
        Returns:
            Array numpy de la imagen aumentada
        """
        # Convertir a PIL Image
        img = Image.fromarray(img_array.astype(np.uint8))
        
        # Aplicar transformaciones aleatorias
        transformations = []
        
        # 1. Rotación aleatoria (-30 a 30 grados)
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            img = img.rotate(angle, fillcolor=(255, 255, 255))
        
        # 2. Flip horizontal
        if random.random() > 0.5:
            img = ImageOps.mirror(img)
        
        # 3. Flip vertical
        if random.random() > 0.7:
            img = ImageOps.flip(img)
        
        # 4. Ajuste de brillo (0.7 a 1.3)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(img)
            factor = random.uniform(0.7, 1.3)
            img = enhancer.enhance(factor)
        
        # 5. Ajuste de contraste (0.8 a 1.2)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(img)
            factor = random.uniform(0.8, 1.2)
            img = enhancer.enhance(factor)
        
        # 6. Ajuste de saturación (0.8 a 1.2)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Color(img)
            factor = random.uniform(0.8, 1.2)
            img = enhancer.enhance(factor)
        
        # 7. Zoom aleatorio (crop y resize)
        if random.random() > 0.6:
            width, height = img.size
            crop_factor = random.uniform(0.8, 0.95)
            new_width = int(width * crop_factor)
            new_height = int(height * crop_factor)
            
            left = random.randint(0, width - new_width)
            top = random.randint(0, height - new_height)
            
            img = img.crop((left, top, left + new_width, top + new_height))
            img = img.resize((width, height))
        
        return np.array(img)
    
    def balance_dataset(self, class_images_dict, target_count=None):
        """
        Balancea el dataset usando data augmentation.
        
        Args:
            class_images_dict: Diccionario {clase: [(path, array), ...]}
            target_count: Número objetivo de imágenes por clase (None = usar el máximo)
            
        Returns:
            Diccionario balanceado con imágenes aumentadas
        """
        print("\n" + "=" * 60)
        print("🔄 BALANCEANDO DATASET CON DATA AUGMENTATION")
        print("=" * 60)
        
        # Determinar el target count
        if target_count is None:
            target_count = max(len(images) for images in class_images_dict.values())
        
        # Limitar el target para evitar datasets demasiado grandes
        max_target = 15000
        if target_count > max_target:
            target_count = max_target
            print(f"⚠️  Limitando target a {max_target} imágenes por clase")
        
        balanced_dict = {}
        
        for class_name, images in class_images_dict.items():
            current_count = len(images)
            needed = target_count - current_count
            
            emoji = {'manzana': '🍎', 'banano': '🍌', 'naranja': '🍊', 
                    'mango': '🥭', 'pera': '🍐'}.get(class_name, '🍏')
            
            print(f"\n{emoji} {class_name.capitalize()}:")
            print(f"   Original: {current_count:,} imágenes")
            
            if needed <= 0:
                print(f"   ✅ Ya tiene suficientes imágenes")
                balanced_dict[class_name] = images
                continue
            
            print(f"   🔄 Generando {needed:,} imágenes aumentadas...")
            
            # Copiar las imágenes originales
            augmented_images = images.copy()
            
            # Generar imágenes aumentadas
            augmentation_rounds = (needed // current_count) + 1
            
            for round_num in range(augmentation_rounds):
                if len(augmented_images) >= target_count:
                    break
                
                for img_path, img_array in images:
                    if len(augmented_images) >= target_count:
                        break
                    
                    # Aplicar augmentation
                    aug_img = self.augment_image(img_array)
                    augmented_images.append((img_path, aug_img))
            
            # Truncar al target exacto
            balanced_dict[class_name] = augmented_images[:target_count]
            
            print(f"   ✅ Final: {len(balanced_dict[class_name]):,} imágenes")
        
        print("\n" + "=" * 60)
        print("✅ BALANCEO COMPLETADO")
        print("=" * 60)
        
        return balanced_dict
    
    def clean_dataset(self, train_split=0.8, balance=True, target_per_class=None):
        """
        Limpia y balancea el dataset completo.
        
        Args:
            train_split: Proporción de datos para entrenamiento (0.8 = 80%)
            balance: Si True, balancea el dataset usando data augmentation
            target_per_class: Número objetivo de imágenes por clase (None = usar el máximo)
        """
        print("=" * 60)
        print("INICIANDO LIMPIEZA Y PREPARACIÓN DEL DATASET")
        print("=" * 60)
        
        self.create_output_structure()
        
        # Diccionario para almacenar imágenes válidas por clase
        class_valid_images = {}
        
        # Paso 1: Validar y cargar todas las imágenes
        for class_name in self.classes:
            class_path = self.dataset_path / class_name
            
            if not class_path.exists():
                print(f"\n⚠️  Advertencia: No se encontró la carpeta '{class_name}'")
                continue
            
            print(f"\n📁 Procesando clase: {class_name}")
            
            # Obtener todas las imágenes
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(list(class_path.rglob(ext)))
            
            valid_images = []
            
            for img_path in image_files:
                self.stats['total_images'] += 1
                
                is_valid, img_array = self.is_valid_image(img_path)
                
                if not is_valid:
                    self.stats['corrupted'] += 1
                    continue
                
                # Verificar dimensiones mínimas
                if img_array.shape[0] < 50 or img_array.shape[1] < 50:
                    self.stats['invalid_dimensions'] += 1
                    continue
                
                valid_images.append((img_path, img_array))
            
            class_valid_images[class_name] = valid_images
            print(f"  ✅ Validadas: {len(valid_images)} imágenes")
        
        # Paso 2: Balancear dataset si está habilitado
        if balance:
            class_valid_images = self.balance_dataset(class_valid_images, target_per_class)
        
        # Paso 3: Dividir en train/test y guardar
        for class_name, valid_images in class_valid_images.items():
            print(f"\n💾 Guardando {class_name}...")
            
            # Dividir en train/test
            np.random.shuffle(valid_images)
            split_idx = int(len(valid_images) * train_split)
            train_images = valid_images[:split_idx]
            test_images = valid_images[split_idx:]
            
            # Procesar y guardar imágenes de entrenamiento
            for idx, (img_path, img_array) in enumerate(train_images):
                img_processed = self.resize_and_normalize(img_array)
                output_path = self.output_path / 'train' / class_name / f"{class_name}_{idx:04d}.png"
                
                # Guardar (convertir de vuelta a 0-255 para guardar)
                img_to_save = (img_processed * 255).astype(np.uint8)
                Image.fromarray(img_to_save).save(output_path)
                self.stats['processed'] += 1
            
            # Procesar y guardar imágenes de prueba
            for idx, (img_path, img_array) in enumerate(test_images):
                img_processed = self.resize_and_normalize(img_array)
                output_path = self.output_path / 'test' / class_name / f"{class_name}_{idx:04d}.png"
                
                img_to_save = (img_processed * 255).astype(np.uint8)
                Image.fromarray(img_to_save).save(output_path)
                self.stats['processed'] += 1
            
            # Actualizar estadísticas de distribución
            self.stats['class_distribution'][class_name] = {
                'train': len(train_images),
                'test': len(test_images),
                'total': len(valid_images)
            }
            
            print(f"  ✅ Guardadas: {len(valid_images)} imágenes")
            print(f"     - Entrenamiento: {len(train_images)}")
            print(f"     - Prueba: {len(test_images)}")
        
        self.print_summary()
        self.create_visualizations()
    
    def print_summary(self):
        """Imprime un resumen de la limpieza."""
        print("\n" + "=" * 60)
        print("RESUMEN DE LIMPIEZA")
        print("=" * 60)
        print(f"Total de imágenes encontradas: {self.stats['total_images']}")
        print(f"Imágenes corruptas eliminadas: {self.stats['corrupted']}")
        print(f"Imágenes con dimensiones inválidas: {self.stats['invalid_dimensions']}")
        print(f"Imágenes procesadas exitosamente: {self.stats['processed']}")
        print("\nDistribución por clase:")
        for class_name, counts in self.stats['class_distribution'].items():
            print(f"  {class_name}: {counts['total']} total (Train: {counts['train']}, Test: {counts['test']})")
    
    def create_visualizations(self):
        """Crea visualizaciones del dataset limpio."""
        print("\n📊 Generando visualizaciones...")
        
        viz_path = self.output_path / 'visualizations'
        
        # 1. Distribución de clases
        self._plot_class_distribution(viz_path)
        
        # 2. Ejemplos de imágenes limpias
        self._plot_sample_images(viz_path)
        
        # 3. Gráfico de barras comparativo train/test
        self._plot_train_test_split(viz_path)
        
        print(f"✅ Visualizaciones guardadas en: {viz_path}")
    
    def _plot_class_distribution(self, viz_path):
        """Gráfico de distribución de clases."""
        plt.figure(figsize=(10, 6))
        
        classes = list(self.stats['class_distribution'].keys())
        totals = [self.stats['class_distribution'][c]['total'] for c in classes]
        
        colors = ['#FF6B6B', '#FFD93D', '#6BCB77', '#FF8C42', '#4D96FF']
        plt.bar(classes, totals, color=colors, edgecolor='black', linewidth=1.5)
        
        plt.title('Distribución de Clases en el Dataset', fontsize=16, fontweight='bold')
        plt.xlabel('Clase de Fruta', fontsize=12)
        plt.ylabel('Número de Imágenes', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Añadir valores sobre las barras
        for i, v in enumerate(totals):
            plt.text(i, v + max(totals)*0.02, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(viz_path / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sample_images(self, viz_path):
        """Muestra ejemplos de imágenes limpias de cada clase."""
        fig, axes = plt.subplots(len(self.classes), 5, figsize=(15, 3*len(self.classes)))
        fig.suptitle('Ejemplos de Imágenes Limpias por Clase', fontsize=16, fontweight='bold')
        
        for i, class_name in enumerate(self.classes):
            class_path = self.output_path / 'train' / class_name
            image_files = list(class_path.glob('*.png'))[:5]
            
            for j, img_path in enumerate(image_files):
                img = Image.open(img_path)
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_title(class_name.capitalize(), fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(viz_path / 'sample_images.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_train_test_split(self, viz_path):
        """Gráfico comparativo de división train/test."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        classes = list(self.stats['class_distribution'].keys())
        train_counts = [self.stats['class_distribution'][c]['train'] for c in classes]
        test_counts = [self.stats['class_distribution'][c]['test'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_counts, width, label='Entrenamiento', 
                       color='#4ECDC4', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, test_counts, width, label='Prueba', 
                       color='#FF6B6B', edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Clase de Fruta', fontsize=12)
        ax.set_ylabel('Número de Imágenes', fontsize=12)
        ax.set_title('División Train/Test por Clase', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Añadir valores sobre las barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(viz_path / 'train_test_split.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Función principal para ejecutar la limpieza."""
    # Configuración de rutas
    DATASET_PATH = 'dataset/raw'  # Carpeta con el dataset original
    OUTPUT_PATH = 'dataset/processed'  # Carpeta para el dataset limpio
    
    print("\n🍎 SISTEMA DE LIMPIEZA DE DATASET DE FRUTAS 🍌")
    print("=" * 60)
    
    # Verificar que existe el dataset
    if not Path(DATASET_PATH).exists():
        print(f"\n❌ Error: No se encontró el dataset en '{DATASET_PATH}'")
        print("\n📋 Estructura esperada:")
        print("dataset/raw/")
        print("  ├── manzana/")
        print("  ├── banano/")
        print("  ├── mango/")
        print("  ├── naranja/")
        print("  └── pera/")
        print("\nPor favor, crea esta estructura y coloca las imágenes correspondientes.")
        return
    
    # Crear instancia del limpiador
    cleaner = DatasetCleaner(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PATH,
        img_size=(100, 100)
    )
    
    # Ejecutar limpieza con balanceo
    print("\n⚙️  Configuración:")
    print("  - Balanceo: ACTIVADO (data augmentation)")
    print("  - Target: Igualar todas las clases al máximo")
    print("  - Train/Test split: 80/20")
    print()
    
    cleaner.clean_dataset(
        train_split=0.8,
        balance=True,  # Activar balanceo
        target_per_class=None  # None = usar el máximo (manzanas)
    )
    
    print("\n✅ Proceso completado exitosamente!")
    print(f"📁 Dataset limpio guardado en: {OUTPUT_PATH}")
    print("\n💡 El dataset ahora está balanceado. Todas las clases tienen")
    print("   la misma cantidad de imágenes gracias a data augmentation.")


if __name__ == "__main__":
    main()
