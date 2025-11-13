"""
Script de entrenamiento de la Red Neuronal Convolucional (CNN) para clasificación de frutas.
Utiliza TensorFlow y Keras para construir, entrenar y exportar el modelo.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Configurar semilla para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)


class FruitClassifierCNN:
    def __init__(self, img_size=(100, 100), num_classes=15, use_transfer_learning=True):
        """
        Inicializa el clasificador CNN.
        
        Args:
            img_size: Tamaño de las imágenes de entrada (ancho, alto)
            num_classes: Número de clases a clasificar (15 enfermedades específicas por defecto)
            use_transfer_learning: Si True, usa MobileNetV2 pre-entrenado
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.use_transfer_learning = use_transfer_learning
        self.model = None
        self.base_model = None
        self.history = None
        # 15 clases específicas del dataset de Kaggle
        self.class_names = [
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
        
    def build_model(self):
        """Construye la arquitectura de la CNN con o sin transfer learning."""
        
        if self.use_transfer_learning:
            print("\n🚀 Usando Transfer Learning con MobileNetV2")
            print("   (Modelo pre-entrenado en ImageNet)")
            
            # Cargar MobileNetV2 pre-entrenado
            self.base_model = keras.applications.MobileNetV2(
                input_shape=(self.img_size[0], self.img_size[1], 3),
                include_top=False,
                weights='imagenet'
            )
            
            # Congelar el modelo base inicialmente
            self.base_model.trainable = False
            
            # Construir el modelo completo
            inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 3))
            
            # Preprocesamiento específico de MobileNetV2
            x = keras.applications.mobilenet_v2.preprocess_input(inputs)
            
            # Base model
            x = self.base_model(x, training=False)
            
            # Capas de clasificación personalizadas
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            x = layers.Dense(256, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            
            x = layers.Dense(128, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            
            model = keras.Model(inputs, outputs)
            
        else:
            print("\n🔨 Construyendo CNN desde cero")
            model = models.Sequential([
                # Primera capa convolucional
                layers.Conv2D(32, (3, 3), activation='relu', 
                             input_shape=(self.img_size[0], self.img_size[1], 3),
                             padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Segunda capa convolucional
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Tercera capa convolucional
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Cuarta capa convolucional
                layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Aplanar para capas densas
                layers.Flatten(),
                
                # Capas densas
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                
                # Capa de salida
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        # Compilar el modelo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print("\n" + "=" * 60)
        print("ARQUITECTURA DEL MODELO")
        print("=" * 60)
        print(f"Parámetros totales: {model.count_params():,}")
        if self.use_transfer_learning:
            trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
            print(f"Parámetros entrenables: {trainable:,}")
            print(f"Capas congeladas: {len(self.base_model.layers)}")
        print("=" * 60)
        
        return model
    
    def prepare_data_generators(self, train_dir, test_dir, batch_size=32):
        """
        Prepara los generadores de datos con data augmentation.
        
        Args:
            train_dir: Directorio con datos de entrenamiento
            test_dir: Directorio con datos de prueba
            batch_size: Tamaño del batch
            
        Returns:
            tuple: (train_generator, test_generator)
        """
        # Data augmentation para entrenamiento
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Solo rescaling para prueba
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Crear generadores
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Guardar mapeo de clases
        self.class_indices = train_generator.class_indices
        
        print("\n📊 Generadores de datos creados:")
        print(f"  - Entrenamiento: {train_generator.samples} imágenes")
        print(f"  - Prueba: {test_generator.samples} imágenes")
        print(f"  - Clases: {list(self.class_indices.keys())}")
        
        return train_generator, test_generator
    
    def train(self, train_generator, test_generator, epochs=50, fine_tune=True):
        """
        Entrena el modelo en dos fases si usa transfer learning.
        
        Fase 1: Entrena solo las capas superiores (base congelada)
        Fase 2: Fine-tuning (descongela algunas capas del base model)
        
        Args:
            train_generator: Generador de datos de entrenamiento
            test_generator: Generador de datos de prueba
            epochs: Número de épocas total
            fine_tune: Si True, hace fine-tuning después del entrenamiento inicial
            
        Returns:
            History object con el historial de entrenamiento
        """
        # Crear directorio para modelos
        Path('models').mkdir(exist_ok=True)
        
        if self.use_transfer_learning:
            # FASE 1: Entrenar solo capas superiores
            print("\n" + "=" * 60)
            print("🎯 FASE 1: ENTRENAMIENTO DE CAPAS SUPERIORES")
            print("=" * 60)
            print("Base model: CONGELADO")
            print(f"Épocas: {epochs // 2}")
            print()
            
            callbacks_phase1 = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    'models/phase1_model.keras',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Entrenar fase 1
            history1 = self.model.fit(
                train_generator,
                epochs=epochs // 2,
                validation_data=test_generator,
                callbacks=callbacks_phase1,
                verbose=1
            )
            
            if fine_tune:
                # FASE 2: Fine-tuning
                print("\n" + "=" * 60)
                print("🔥 FASE 2: FINE-TUNING")
                print("=" * 60)
                
                # Descongelar las últimas capas del base model
                self.base_model.trainable = True
                
                # Congelar todas las capas excepto las últimas
                fine_tune_at = len(self.base_model.layers) - 30  # Descongelar últimas 30 capas
                
                for layer in self.base_model.layers[:fine_tune_at]:
                    layer.trainable = False
                
                trainable_layers = sum([1 for layer in self.base_model.layers if layer.trainable])
                print(f"Capas descongeladas: {trainable_layers}/{len(self.base_model.layers)}")
                
                # Recompilar con learning rate más bajo
                self.model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # LR 10x menor
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                print(f"Learning rate: 0.0001 (reducido)")
                print(f"Épocas: {epochs // 2}")
                print()
                
                callbacks_phase2 = [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=7,
                        restore_best_weights=True,
                        verbose=1
                    ),
                    ModelCheckpoint(
                        'models/best_model.keras',
                        monitor='val_accuracy',
                        save_best_only=True,
                        verbose=1
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=3,
                        min_lr=1e-8,
                        verbose=1
                    )
                ]
                
                # Entrenar fase 2
                history2 = self.model.fit(
                    train_generator,
                    epochs=epochs // 2,
                    validation_data=test_generator,
                    callbacks=callbacks_phase2,
                    verbose=1
                )
                
                # Combinar historiales
                self.history = history1
                for key in history1.history.keys():
                    self.history.history[key].extend(history2.history[key])
            else:
                self.history = history1
                
        else:
            # Entrenamiento normal sin transfer learning
            print("\n" + "=" * 60)
            print("INICIANDO ENTRENAMIENTO")
            print("=" * 60)
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    'models/best_model.keras',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            self.history = self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=test_generator,
                callbacks=callbacks,
                verbose=1
            )
        
        return self.history
    
    def evaluate(self, test_generator):
        """
        Evalúa el modelo en el conjunto de prueba.
        
        Args:
            test_generator: Generador de datos de prueba
        """
        print("\n" + "=" * 60)
        print("EVALUACIÓN DEL MODELO")
        print("=" * 60)
        
        # Evaluar
        test_loss, test_accuracy = self.model.evaluate(test_generator)
        
        print(f"\n📊 Resultados en conjunto de prueba:")
        print(f"  - Pérdida: {test_loss:.4f}")
        print(f"  - Precisión: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Predicciones para matriz de confusión
        test_generator.reset()
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Reporte de clasificación
        print("\n📋 Reporte de clasificación:")
        print(classification_report(
            true_classes, 
            predicted_classes, 
            target_names=self.class_names
        ))
        
        # Matriz de confusión
        cm = confusion_matrix(true_classes, predicted_classes)
        self._plot_confusion_matrix(cm)
        
        return test_loss, test_accuracy
    
    def _plot_confusion_matrix(self, cm):
        """Visualiza la matriz de confusión."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Número de predicciones'})
        plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
        plt.ylabel('Clase Real', fontsize=12)
        plt.xlabel('Clase Predicha', fontsize=12)
        plt.tight_layout()
        
        # Guardar
        viz_path = Path('models/visualizations')
        viz_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Matriz de confusión guardada en: {viz_path / 'confusion_matrix.png'}")
    
    def plot_training_history(self):
        """Visualiza el historial de entrenamiento."""
        if self.history is None:
            print("⚠️  No hay historial de entrenamiento disponible.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Precisión
        axes[0].plot(self.history.history['accuracy'], label='Entrenamiento', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Validación', linewidth=2)
        axes[0].set_title('Precisión del Modelo', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Época', fontsize=12)
        axes[0].set_ylabel('Precisión', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Pérdida
        axes[1].plot(self.history.history['loss'], label='Entrenamiento', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Validación', linewidth=2)
        axes[1].set_title('Pérdida del Modelo', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Época', fontsize=12)
        axes[1].set_ylabel('Pérdida', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar
        viz_path = Path('models/visualizations')
        viz_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_path / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Historial de entrenamiento guardado en: {viz_path / 'training_history.png'}")
    
    def save_model(self, filepath='models/fruit_classifier.keras'):
        """
        Guarda el modelo entrenado.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        self.model.save(filepath)
        
        # Guardar también el mapeo de clases
        class_mapping = {
            'class_indices': self.class_indices,
            'class_names': self.class_names
        }
        
        mapping_path = Path(filepath).parent / 'class_mapping.json'
        with open(mapping_path, 'w') as f:
            json.dump(class_mapping, f, indent=4)
        
        print(f"\n✅ Modelo guardado en: {filepath}")
        print(f"✅ Mapeo de clases guardado en: {mapping_path}")
    
    def load_model(self, filepath='models/fruit_classifier.keras'):
        """
        Carga un modelo previamente entrenado.
        
        Args:
            filepath: Ruta del modelo a cargar
        """
        self.model = keras.models.load_model(filepath)
        
        # Cargar mapeo de clases
        mapping_path = Path(filepath).parent / 'class_mapping.json'
        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                class_mapping = json.load(f)
                self.class_indices = class_mapping['class_indices']
                self.class_names = class_mapping['class_names']
        
        print(f"✅ Modelo cargado desde: {filepath}")


def main():
    """Función principal para entrenar el modelo."""
    print("\n🍎 SISTEMA DE ENTRENAMIENTO CNN - CLASIFICADOR DE FRUTAS 🍌")
    print("=" * 60)
    
    # Configuración
    TRAIN_DIR = 'dataset/processed/train'
    TEST_DIR = 'dataset/processed/test'
    IMG_SIZE = (100, 100)
    BATCH_SIZE = 75  # Aumentado para procesar más imágenes por vez
    EPOCHS = 30  # Reducido - transfer learning converge rápido
    USE_TRANSFER_LEARNING = True  # Cambiar a False para entrenar desde cero
    
    # Verificar que existen los datos procesados
    if not Path(TRAIN_DIR).exists() or not Path(TEST_DIR).exists():
        print("\n❌ Error: No se encontraron los datos procesados.")
        print("Por favor, ejecuta primero 'data_preparation.py' para limpiar el dataset.")
        return
    
    print("\n⚙️  CONFIGURACIÓN:")
    print(f"  - Transfer Learning: {'✅ ACTIVADO (MobileNetV2)' if USE_TRANSFER_LEARNING else '❌ Desactivado'}")
    print(f"  - Tamaño de imagen: {IMG_SIZE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Épocas totales: {EPOCHS}")
    if USE_TRANSFER_LEARNING:
        print(f"  - Fase 1 (capas superiores): {EPOCHS // 2} épocas")
        print(f"  - Fase 2 (fine-tuning): {EPOCHS // 2} épocas")
    print()
    
    # Crear instancia del clasificador
    classifier = FruitClassifierCNN(
        img_size=IMG_SIZE, 
        num_classes=15,
        use_transfer_learning=USE_TRANSFER_LEARNING
    )
    
    # Construir modelo
    classifier.build_model()
    
    # Preparar datos
    train_gen, test_gen = classifier.prepare_data_generators(
        TRAIN_DIR, 
        TEST_DIR, 
        batch_size=BATCH_SIZE
    )
    
    # Entrenar (con entrenamiento por fases si usa transfer learning)
    classifier.train(
        train_gen, 
        test_gen, 
        epochs=EPOCHS,
        fine_tune=USE_TRANSFER_LEARNING  # Fine-tuning solo si usa transfer learning
    )
    
    # Evaluar
    classifier.evaluate(test_gen)
    
    # Visualizar historial
    classifier.plot_training_history()
    
    # Guardar modelo
    classifier.save_model('models/fruit_classifier.keras')
    
    print("\n" + "=" * 60)
    print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print("\n📁 Archivos generados:")
    print("  - models/fruit_classifier.keras (modelo principal)")
    print("  - models/best_model.keras (mejor modelo durante entrenamiento)")
    if USE_TRANSFER_LEARNING:
        print("  - models/phase1_model.keras (modelo fase 1)")
    print("  - models/class_mapping.json (mapeo de clases)")
    print("  - models/visualizations/ (gráficos y métricas)")
    print("\n💡 VENTAJAS DEL TRANSFER LEARNING:")
    print("  ✅ Entrenamiento 3-5x más rápido")
    print("  ✅ Mejor precisión con menos datos")
    print("  ✅ Menos propenso a overfitting")
    print("  ✅ Aprovecha conocimiento de ImageNet (1.4M imágenes)")


if __name__ == "__main__":
    main()
