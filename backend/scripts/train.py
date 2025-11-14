"""
Script de entrenamiento del modelo de clasificación de enfermedades.

Este script:
1. Prepara los datos automáticamente (con cache PKL)
2. Entrena el modelo con Transfer Learning
3. Evalúa y guarda resultados

Uso:
    python backend/scripts/train.py
    
El sistema detecta automáticamente si necesita preparar datos o puede
usar el cache existente.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Agregar el directorio backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from utils.data_cache import DataCache

# Configurar para mejor rendimiento
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir logs de TensorFlow
tf.random.set_seed(42)
np.random.seed(42)

# Optimizaciones de TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("✅ GPU detectada y configurada")


class PlantDiseaseClassifier:
    """Clasificador de enfermedades de plantas con Transfer Learning."""
    
    def __init__(self, img_size=(100, 100), num_classes=15, use_transfer_learning=True):
        """
        Inicializa el clasificador.
        
        Args:
            img_size: Tamaño de las imágenes
            num_classes: Número de clases (15 enfermedades específicas por defecto)
            use_transfer_learning: Usar MobileNetV2 pre-entrenado
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.use_transfer_learning = use_transfer_learning
        self.model = None
        self.history = None
        self.cache = DataCache()
    
    def build_model(self):
        """Construye el modelo optimizado."""
        
        if self.use_transfer_learning:
            print("\n🚀 Construyendo modelo con MobileNetV2 (Transfer Learning)")
            
            # Data augmentation
            data_augmentation = keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.15),
                layers.RandomZoom(0.15),
                layers.RandomContrast(0.1),
            ], name="data_augmentation")
            
            # Modelo base pre-entrenado
            base_model = keras.applications.MobileNetV2(
                input_shape=(self.img_size[0], self.img_size[1], 3),
                include_top=False,
                weights='imagenet'
            )
            
            # Congelar base model
            base_model.trainable = False
            
            # Construir modelo con augmentation
            inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 3))
            x = data_augmentation(inputs)
            x = keras.applications.mobilenet_v2.preprocess_input(x)
            x = base_model(x, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            
            model = keras.Model(inputs, outputs)
            self.base_model = base_model
            self.data_augmentation = data_augmentation
            
        else:
            print("\n🔨 Construyendo CNN desde cero")
            
            model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', 
                            input_shape=(self.img_size[0], self.img_size[1], 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                layers.Flatten(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        # Compilar con learning rate ajustado
        initial_lr = 0.001 if self.use_transfer_learning else 0.0005
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print(f"\n✅ Modelo construido: {model.count_params():,} parámetros")
        return model
    
    def train_with_arrays(self, X_train, y_train, X_test, y_test, 
                         epochs=20, batch_size=64):
        """
        Entrena el modelo con arrays numpy (datos desde cache).
        
        Args:
            X_train: Array de imágenes de entrenamiento
            y_train: Labels de entrenamiento (one-hot)
            X_test: Array de imágenes de prueba
            y_test: Labels de prueba (one-hot)
            epochs: Número de épocas
            batch_size: Tamaño del batch
            
        Returns:
            History object
        """
        print("\n" + "=" * 60)
        print("🎯 INICIANDO ENTRENAMIENTO")
        print("=" * 60)
        print(f"  - Muestras train: {len(X_train):,}")
        print(f"  - Muestras test: {len(X_test):,}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Épocas: {epochs}")
        print()
        
        # Crear directorio para modelos
        Path('models').mkdir(exist_ok=True)
        
        # Callbacks optimizados
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
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
                patience=4,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Entrenar
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        print(f"\n⏱️  Tiempo de entrenamiento: {training_time/60:.2f} minutos")
        
        return self.history
    
    def fine_tune(self, X_train, y_train, X_test, y_test, 
                  epochs_phase2=15, batch_size=64):
        """
        Fine-tuning del modelo con descongelamiento gradual (solo si usa transfer learning).
        
        Estrategia de 2 Fases:
        - Fase 2a: Descongelar últimas 50 capas con LR bajo
        - Fase 2b: Descongelar últimas 100 capas con LR muy bajo
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_test, y_test: Datos de prueba
            epochs_phase2: Épocas totales de fine-tuning (se divide en 2 subfases)
            batch_size: Tamaño del batch
        """
        if not self.use_transfer_learning:
            print("⚠️  Fine-tuning solo disponible con transfer learning")
            return
        
        total_layers = len(self.base_model.layers)
        print(f"\n📊 Base model tiene {total_layers} capas totales")
        
        # ==================================================================
        # FASE 2a: Descongelamiento de features de alto nivel (últimas 50 capas)
        # ==================================================================
        print("\n" + "=" * 60)
        print("🔥 FASE 2a: FINE-TUNING - Features de Alto Nivel")
        print("=" * 60)
        
        # Descongelar base model
        self.base_model.trainable = True
        
        # Congelar todas excepto las últimas 50 capas
        fine_tune_at = max(0, total_layers - 50)
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        trainable_layers = sum([1 for layer in self.base_model.layers if layer.trainable])
        print(f"  - Capas congeladas: {fine_tune_at}")
        print(f"  - Capas descongeladas: {trainable_layers}")
        print(f"  - Learning Rate: 0.0001")
        
        # Recompilar con LR bajo
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks para Fase 2a
        callbacks_2a = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=6,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/finetuned_phase2a.keras',
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
        
        # Entrenar Fase 2a
        epochs_2a = max(epochs_phase2 // 2, 7)  # Mínimo 7 epochs
        start_time_2a = time.time()
        
        history_2a = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs_2a,
            validation_data=(X_test, y_test),
            callbacks=callbacks_2a,
            verbose=1
        )
        
        time_2a = time.time() - start_time_2a
        print(f"\n⏱️  Tiempo Fase 2a: {time_2a/60:.2f} minutos")
        
        # ==================================================================
        # FASE 2b: Descongelamiento de más capas (últimas 100 capas)
        # ==================================================================
        print("\n" + "=" * 60)
        print("🔥 FASE 2b: FINE-TUNING - Features Intermedias")
        print("=" * 60)
        
        # Descongelar más capas (últimas 100)
        fine_tune_at_2b = max(0, total_layers - 100)
        for layer in self.base_model.layers[:fine_tune_at_2b]:
            layer.trainable = False
        for layer in self.base_model.layers[fine_tune_at_2b:]:
            layer.trainable = True
        
        trainable_layers_2b = sum([1 for layer in self.base_model.layers if layer.trainable])
        print(f"  - Capas congeladas: {fine_tune_at_2b}")
        print(f"  - Capas descongeladas: {trainable_layers_2b}")
        print(f"  - Learning Rate: 0.00005 (muy bajo para estabilidad)")
        
        # Recompilar con LR muy bajo
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks para Fase 2b
        callbacks_2b = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
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
        
        # Entrenar Fase 2b
        epochs_2b = epochs_phase2 - epochs_2a
        start_time_2b = time.time()
        
        history_2b = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs_2b,
            validation_data=(X_test, y_test),
            callbacks=callbacks_2b,
            verbose=1
        )
        
        time_2b = time.time() - start_time_2b
        print(f"\n⏱️  Tiempo Fase 2b: {time_2b/60:.2f} minutos")
        
        # Combinar historiales
        for key in self.history.history.keys():
            if key in history_2a.history:
                self.history.history[key].extend(history_2a.history[key])
            if key in history_2b.history:
                self.history.history[key].extend(history_2b.history[key])
        
        total_ft_time = time_2a + time_2b
        print(f"\n⏱️  Tiempo total de fine-tuning: {total_ft_time/60:.2f} minutos")
        print(f"\n✅ Fine-tuning completado con {trainable_layers_2b} capas entrenables")
    
    def evaluate(self, X_test, y_test, class_names):
        """
        Evalúa el modelo.
        
        Args:
            X_test: Datos de prueba
            y_test: Labels de prueba (one-hot)
            class_names: Nombres de las clases
        """
        print("\n" + "=" * 60)
        print("📊 EVALUACIÓN DEL MODELO")
        print("=" * 60)
        
        # Evaluar
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\n✅ Resultados:")
        print(f"  - Pérdida: {test_loss:.4f}")
        print(f"  - Precisión: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Predicciones
        predictions = self.model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        # Reporte de clasificación
        print("\n📋 Reporte de clasificación:")
        print(classification_report(true_classes, predicted_classes, 
                                   target_names=class_names, digits=4))
        
        # Matriz de confusión
        cm = confusion_matrix(true_classes, predicted_classes)
        self._plot_confusion_matrix(cm, class_names)
        
        return test_loss, test_accuracy
    
    def _plot_confusion_matrix(self, cm, class_names):
        """Visualiza matriz de confusión."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
        plt.ylabel('Clase Real', fontsize=12)
        plt.xlabel('Clase Predicha', fontsize=12)
        plt.tight_layout()
        
        viz_path = Path('models/visualizations')
        viz_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Matriz guardada en: {viz_path / 'confusion_matrix.png'}")
    
    def plot_training_history(self):
        """Visualiza historial de entrenamiento."""
        if self.history is None:
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
        
        viz_path = Path('models/visualizations')
        viz_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_path / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Historial guardado en: {viz_path / 'training_history.png'}")
    
    def save_model(self, filepath='models/fruit_classifier.keras', class_names=None):
        """Guarda el modelo y metadatos."""
        self.model.save(filepath)
        
        if class_names:
            class_mapping = {
                'class_names': class_names,
                'num_classes': len(class_names),
                'img_size': self.img_size
            }
            
            mapping_path = Path(filepath).parent / 'class_mapping.json'
            with open(mapping_path, 'w') as f:
                json.dump(class_mapping, f, indent=4)
            
            print(f"✅ Modelo guardado: {filepath}")
            print(f"✅ Mapeo guardado: {mapping_path}")


def main():
    """
    Función principal de entrenamiento.
    
    IMPORTANTE: Si actualizas las clases del modelo, limpia el cache:
        python backend/utils/manage_cache.py
        Opción [2] - Limpiar cache
    """
    print("\n" + "=" * 70)
    print("🚀 ENTRENAMIENTO DE CLASIFICADOR DE ENFERMEDADES")
    print("=" * 70)
    
    # ================================================================
    # CONFIGURACIÓN
    # ================================================================
    RAW_DATASET = "dataset/raw"
    PROCESSED_DATASET = "dataset/processed"
    IMG_SIZE = (100, 100)
    
    # Parámetros de entrenamiento optimizados
    EPOCHS_PHASE1 = 15      # Entrenamiento inicial (capas Dense)
    EPOCHS_PHASE2 = 15      # Fine-tuning gradual (2 subfases)
    BATCH_SIZE = 32         # Batch size para regularización
    USE_TRANSFER_LEARNING = True
    DO_FINE_TUNING = True   # Activado con estrategia gradual mejorada
    
    print("\n⚙️  CONFIGURACIÓN:")
    print(f"  - Transfer Learning: {'✅ MobileNetV2' if USE_TRANSFER_LEARNING else '❌'}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Épocas Fase 1 (clasificador): {EPOCHS_PHASE1}")
    if DO_FINE_TUNING and USE_TRANSFER_LEARNING:
        print(f"  - Épocas Fase 2 (fine-tuning gradual): {EPOCHS_PHASE2}")
        print(f"    • Subfase 2a: ~{EPOCHS_PHASE2//2} epochs (últimas 50 capas)")
        print(f"    • Subfase 2b: ~{EPOCHS_PHASE2 - EPOCHS_PHASE2//2} epochs (últimas 100 capas)")
    
    # Cargar datos desde cache
    cache = DataCache()
    
    # Configuración con las 15 clases específicas del dataset
    config = {
        'img_size': IMG_SIZE,
        'classes': [
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
        ],
        'balance': False
    }
    
    print("\n📂 Cargando datos desde cache...")
    train_data = cache.load(RAW_DATASET, config, 'train')
    test_data = cache.load(RAW_DATASET, config, 'test')
    
    if not train_data or not test_data:
        print("\n⚠️  Cache no encontrado. Preparando datos automáticamente...")
        print("=" * 70)
        
        # Importar y ejecutar preparación de datos
        from prepare_dataset import DatasetProcessor
        
        processor = DatasetProcessor(RAW_DATASET, PROCESSED_DATASET, IMG_SIZE)
        result = processor.prepare_optimized(use_cache=True, force_reprocess=False)
        
        if not result:
            print("\n❌ Error preparando datos. Verifica que el dataset exista en:")
            print(f"   {RAW_DATASET}/New Plant Diseases Dataset(Augmented)/train/")
            return
        
        # Cargar datos recién preparados
        train_data = cache.load(RAW_DATASET, config, 'train')
        test_data = cache.load(RAW_DATASET, config, 'test')
        
        if not train_data or not test_data:
            print("\n❌ Error cargando datos preparados")
            return
    
    X_train, y_train, class_names = train_data
    X_test, y_test, _ = test_data
    
    num_classes = len(class_names)
    
    print(f"\n✅ Datos cargados:")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_test: {X_test.shape}")
    print(f"  - Clases: {class_names}")
    
    # Crear y construir modelo
    classifier = PlantDiseaseClassifier(
        img_size=IMG_SIZE,
        num_classes=num_classes,
        use_transfer_learning=USE_TRANSFER_LEARNING
    )
    
    classifier.build_model()
    
    # FASE 1: Entrenamiento inicial
    print("\n" + "=" * 60)
    print("FASE 1: ENTRENAMIENTO INICIAL")
    print("=" * 60)
    
    total_start = time.time()
    
    classifier.train_with_arrays(
        X_train, y_train,
        X_test, y_test,
        epochs=EPOCHS_PHASE1,
        batch_size=BATCH_SIZE
    )
    
    # FASE 2: Fine-tuning (opcional)
    if DO_FINE_TUNING and USE_TRANSFER_LEARNING:
        classifier.fine_tune(
            X_train, y_train,
            X_test, y_test,
            epochs=EPOCHS_PHASE2,
            batch_size=BATCH_SIZE
        )
    
    total_time = time.time() - total_start
    
    # Evaluación final
    classifier.evaluate(X_test, y_test, class_names)
    
    # Visualizaciones
    classifier.plot_training_history()
    
    # Guardar modelo
    classifier.save_model('models/fruit_classifier.keras', class_names)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"\n⏱️  Tiempo total: {total_time/60:.2f} minutos")
    print(f"\n📁 Archivos generados:")
    print("  - models/best_model.keras")
    print("  - models/fruit_classifier.keras")
    print("  - models/class_mapping.json")
    print("  - models/visualizations/")
    
    print("\n💡 CARACTERÍSTICAS:")
    print("  ✅ Preparación automática: Detecta y prepara datos si es necesario")
    print("  ✅ Cache PKL: Datos se guardan para reuso")
    print("  ✅ Transfer Learning: Usa MobileNetV2 pre-entrenado")
    print("  ✅ Data Augmentation: Previene overfitting")
    print("  ✅ Fine-tuning Gradual: Descongelamiento progresivo en 2 fases")
    print("  ✅ Optimizado: Hiperparámetros balanceados")
    
    print("\n🎯 PRÓXIMOS PASOS:")
    print("  1. Probar predicciones: python backend/scripts/predict.py <imagen>")
    print("  2. Iniciar API: python backend/app.py")
    print("  3. Re-entrenar: python backend/scripts/train.py")


if __name__ == "__main__":
    main()
