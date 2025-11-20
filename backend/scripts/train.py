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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support, accuracy_score,
    top_k_accuracy_score
)
from datetime import datetime
import psutil

# Agregar el directorio backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from utils.data_cache import DataCache
from scripts.detailed_metrics import DetailedMetrics
from config import IMG_SIZE, CLASSES, NUM_CLASSES, BATCH_SIZE

# Configurar para mejor rendimiento
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir logs de TensorFlow
tf.random.set_seed(42)
np.random.seed(42)

# Optimizaciones de TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("✅ GPU detectada y configurada")


class FineTuningMonitor(Callback):
    """
    Callback personalizado para monitorear señales de éxito y problemas durante fine-tuning.
    
    Señales de éxito:
    - ✅ Val accuracy sube gradualmente
    - ✅ Val loss baja sin oscilar mucho
    - ✅ Gap train-val no es muy grande (<10%)
    
    Señales de problemas:
    - ❌ Val loss explota → LR demasiado alto
    - ❌ Overfitting severo (train 95%, val 50%) → Más regularización
    - ❌ No mejora nada → Posible problema en datos
    """
    
    def __init__(self, phase_name="Fine-tuning"):
        super().__init__()
        self.phase_name = phase_name
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.val_loss_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        train_loss = logs.get('loss', 0)
        train_acc = logs.get('accuracy', 0)
        
        # Calcular gap train-val
        acc_gap = abs(train_acc - val_acc)
        
        # Guardar historial
        self.val_loss_history.append(val_loss)
        
        # Detectar volatilidad en val_loss
        if len(self.val_loss_history) >= 3:
            recent_losses = self.val_loss_history[-3:]
            volatility = max(recent_losses) - min(recent_losses)
        else:
            volatility = 0
        
        print(f"\n📊 [{self.phase_name}] Epoch {epoch + 1} - Monitoreo:")
        
        # SEÑALES DE ÉXITO
        success_signals = []
        
        if val_acc > self.best_val_acc:
            improvement = (val_acc - self.best_val_acc) * 100
            success_signals.append(f"✅ Val accuracy mejora: +{improvement:.2f}%")
            self.best_val_acc = val_acc
            self.epochs_no_improve = 0
        
        if val_loss < self.best_val_loss:
            success_signals.append(f"✅ Val loss baja: {val_loss:.4f}")
            self.best_val_loss = val_loss
        
        if acc_gap < 0.10:
            success_signals.append(f"✅ Gap train-val saludable: {acc_gap*100:.1f}%")
        
        if volatility < 0.2:
            success_signals.append("✅ Val loss estable (baja oscilación)")
        
        # SEÑALES DE PROBLEMAS
        problem_signals = []
        
        # Val loss explota
        if len(self.val_loss_history) >= 2:
            if val_loss > self.val_loss_history[-2] * 1.5:
                problem_signals.append("❌ ALERTA: Val loss explota - LR puede ser muy alto")
        
        # Overfitting severo
        if train_acc > 0.95 and val_acc < 0.70:
            problem_signals.append(f"❌ OVERFITTING SEVERO: train={train_acc:.1%}, val={val_acc:.1%}")
        elif acc_gap > 0.15:
            problem_signals.append(f"⚠️  Gap train-val alto: {acc_gap*100:.1f}% (>15%)")
        
        # Estancamiento
        if val_acc <= self.best_val_acc:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= 5:
                problem_signals.append(f"⚠️  Sin mejora por {self.epochs_no_improve} epochs")
        
        # Volatilidad alta
        if volatility > 0.3:
            problem_signals.append(f"⚠️  Val loss oscila mucho: volatilidad={volatility:.3f}")
        
        # Imprimir señales
        if success_signals:
            print("  " + "\n  ".join(success_signals))
        
        if problem_signals:
            print("  " + "\n  ".join(problem_signals))
        
        if not success_signals and not problem_signals:
            print("  🔵 Entrenamiento en progreso normal")
        
        # Métricas actuales
        print(f"  📋 Métricas: train_acc={train_acc:.1%}, val_acc={val_acc:.1%}, gap={acc_gap*100:.1f}%")


class PlantDiseaseClassifier:
    """Clasificador de enfermedades de plantas con Transfer Learning."""
    
    def __init__(self, img_size=IMG_SIZE, num_classes=NUM_CLASSES, use_transfer_learning=True):
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
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
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
        
        # Compilar con configuración: Adam lr=1e-4
        initial_lr = 1e-4 if self.use_transfer_learning else 5e-5
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print(f"\n✅ Modelo construido: {model.count_params():,} parámetros")
        print(f"   - Optimizer: Adam (lr={initial_lr})")
        print(f"   - Loss: CrossEntropyLoss")
        return model
    
    def train_with_arrays(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                         epochs=20, batch_size=64, class_weights=None):
        """
        Entrena el modelo con arrays numpy (datos desde cache).
        
        Args:
            X_train: Array de imágenes de entrenamiento
            y_train: Labels de entrenamiento (one-hot)
            X_val: Array de imágenes de validación
            y_val: Labels de validación (one-hot)
            X_test: Array de imágenes de prueba
            y_test: Labels de prueba (one-hot)
            epochs: Número de épocas
            batch_size: Tamaño del batch
            class_weights: Dict con pesos por clase (del config.py)
            
        Returns:
            History object
        """
        print("\n" + "=" * 60)
        print("🎯 INICIANDO ENTRENAMIENTO (SPLIT 70/15/15)")
        print("=" * 60)
        print(f"  - Muestras train: {len(X_train):,}")
        print(f"  - Muestras val: {len(X_val):,}")
        print(f"  - Muestras test: {len(X_test):,}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Épocas máximas: {epochs}")
        if class_weights:
            print(f"  - Class weights: ✅ Aplicados (balance de clases)")
        print()
        
        # Crear directorios
        Path('models').mkdir(exist_ok=True)
        Path('metrics').mkdir(exist_ok=True)
        
        # Callbacks optimizados
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,          # paciencia 15-20 epochs
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            ModelCheckpoint(
                'models/last_model.keras',
                monitor='val_loss',
                save_best_only=False,
                save_weights_only=False,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,   
                patience=5,   
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
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        print(f"\n⏱️  Tiempo de entrenamiento: {training_time/60:.2f} minutos")
        
        # Guardar historial de entrenamiento como JSON
        self._save_training_history(self.history, 'metrics/training_history.json')
        
        return self.history
    
    def fine_tune(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                  epochs_phase2=15, batch_size=64):
        """
        Fine-tuning del modelo con descongelamiento gradual (solo si usa transfer learning).
        
        Estrategia basada en análisis de features de MobileNetV2:
        - Capas 0-50:   Features básicas (bordes, texturas) → MANTENER CONGELADAS
        - Capas 51-100: Features intermedias (patrones)
        - Capas 101-154: Features complejas (objetos)
        
        Descongela capas 101-154 (features complejas)
        Descongela capas 51-154 (añade features intermedias)
        
        Nota: BatchNormalization layers se manejan cuidadosamente con batch_size pequeño.
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_test, y_test: Datos de prueba
            epochs_phase2: Épocas totales de fine-tuning
            batch_size: Tamaño del batch
        """
        if not self.use_transfer_learning:
            print("⚠️  Fine-tuning solo disponible con transfer learning")
            return
        
        total_layers = len(self.base_model.layers)
        print(f"\n📊 Base model tiene {total_layers} capas totales")
        
        # ==================================================================
        # Descongelamiento de features complejas (capas 101-154)
        # ==================================================================
        print("\n" + "=" * 60)
        print("🔥 FINE-TUNING - Features Complejas (Capas 101-154)")
        print("=" * 60)
        print("  🌿 Objetivo: Adaptar detección de objetos completos a morfología de hojas")
        
        # Descongelar base model
        self.base_model.trainable = True
        
        # Estrategia: Descongelar solo capas 101-154 (features complejas)
        # Mantener congeladas 0-100 (features básicas e intermedias)
        fine_tune_at = min(101, total_layers - 1)
        
        for i, layer in enumerate(self.base_model.layers):
            if i < fine_tune_at:
                layer.trainable = False
            else:
                # Proteger BatchNormalization con batch_size pequeño
                if 'BatchNormalization' in layer.__class__.__name__ and batch_size < 16:
                    layer.trainable = False
                else:
                    layer.trainable = True
        
        trainable_layers = sum([1 for layer in self.base_model.layers if layer.trainable])
        frozen_bn = sum([1 for layer in self.base_model.layers[fine_tune_at:] 
                        if 'BatchNormalization' in layer.__class__.__name__ and not layer.trainable])
        
        print(f"  - Rango de capas: {fine_tune_at}-{total_layers} (features complejas)")
        print(f"  - Capas congeladas: 0-{fine_tune_at-1} (features básicas/intermedias)")
        print(f"  - Capas descongeladas: {trainable_layers}")
        if frozen_bn > 0:
            print(f"  - BatchNorm protegidas: {frozen_bn} (batch_size={batch_size} < 16)")
        print(f"  - Learning Rate: 0.00005 (20x más bajo que proteger ImageNet)")
        print(f"  - LR Decay: factor=0.2, patience=5, min_lr=0.000001")
        
        # Recompilar con LR más conservador
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks para Fase 2a (fine-tuning conservador)
        callbacks_2a = [
            FineTuningMonitor(phase_name="Fase 2a"),
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
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
                factor=0.2,          # Decay suave para evitar olvido catastrófico
                patience=5,          # Más paciente en fine-tuning
                min_lr=0.000001,     # Mínimo ultra-bajo para Fase 2a
                verbose=1
            )
        ]
        
        # Entrenar Fase 2a
        epochs_2a = max(epochs_phase2 // 2, 5)  # Mínimo 5 epochs (reducido)
        start_time_2a = time.time()
        
        history_2a = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs_2a,
            validation_data=(X_val, y_val),
            callbacks=callbacks_2a,
            verbose=1
        )
        
        time_2a = time.time() - start_time_2a
        print(f"\n⏱️  Tiempo Fase 2a: {time_2a/60:.2f} minutos")
        
        # ==================================================================
        # FASE 2b: Descongelamiento de features intermedias (capas 51-154)
        # ==================================================================
        print("\n" + "=" * 60)
        print("🔥 FASE 2b: FINE-TUNING - Features Intermedias (Capas 51-154)")
        print("=" * 60)
        print("  🍃 Objetivo: Adaptar detección de patrones/formas a síntomas de enfermedades")
        
        # Estrategia: Descongelar capas 51-154 (features intermedias + complejas)
        # Mantener congeladas 0-50 (features básicas: bordes, texturas)
        fine_tune_at_2b = min(51, total_layers - 1)
        
        for i, layer in enumerate(self.base_model.layers):
            if i < fine_tune_at_2b:
                layer.trainable = False
            else:
                # Proteger BatchNormalization con batch_size pequeño
                if 'BatchNormalization' in layer.__class__.__name__ and batch_size < 16:
                    layer.trainable = False
                else:
                    layer.trainable = True
        
        trainable_layers_2b = sum([1 for layer in self.base_model.layers if layer.trainable])
        frozen_bn_2b = sum([1 for layer in self.base_model.layers[fine_tune_at_2b:] 
                           if 'BatchNormalization' in layer.__class__.__name__ and not layer.trainable])
        
        print(f"  - Rango de capas: {fine_tune_at_2b}-{total_layers} (features intermedias/complejas)")
        print(f"  - Capas congeladas: 0-{fine_tune_at_2b-1} (features básicas preservadas)")
        print(f"  - Capas descongeladas: {trainable_layers_2b}")
        if frozen_bn_2b > 0:
            print(f"  - BatchNorm protegidas: {frozen_bn_2b} (batch_size={batch_size} < 16)")
        print(f"  - Learning Rate: 0.00001 (ultra-bajo para evitar catastrophic forgetting)")
        print(f"  - LR Decay: factor=0.2, patience=5, min_lr=0.000001")
        
        # Recompilar con LR ultra-conservador
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks para Fase 2b (fine-tuning ultra-conservador)
        callbacks_2b = [
            FineTuningMonitor(phase_name="Fase 2b"),
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
                factor=0.2,          # Decay suave para preservar features ImageNet
                patience=5,          # Más paciente con más capas descongeladas
                min_lr=0.000001,     # Mínimo ultra-bajo para Fase 2b
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
            validation_data=(X_val, y_val),
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
    
    def calculate_per_class_metrics(self, y_true, y_pred, class_names):
        """
        Calcula métricas detalladas por clase.
        
        Returns:
            dict: Métricas por clase y agregadas
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Métricas agregadas
        macro_avg = {
            'precision': precision.mean(),
            'recall': recall.mean(),
            'f1': f1.mean()
        }
        
        weighted_avg = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        return {
            'per_class': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            },
            'macro_avg': macro_avg,
            'weighted_avg': {
                'precision': weighted_avg[0],
                'recall': weighted_avg[1],
                'f1': weighted_avg[2]
            }
        }
    
    def calculate_per_crop_metrics(self, y_true, y_pred, class_names):
        """
        Calcula accuracy por cultivo (Apple, Corn, Potato, Tomato).
        
        Returns:
            dict: Accuracy por cultivo
        """
        crops = {
            'Apple': ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy'],
            'Corn': ['Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight'],
            'Potato': ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight'],
            'Tomato': ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold']
        }
        
        per_crop = {}
        for crop, crop_classes in crops.items():
            # Filtrar índices de este cultivo
            crop_indices = [i for i, cls in enumerate(class_names) if cls in crop_classes]
            
            if not crop_indices:
                continue
            
            # Filtrar predicciones de este cultivo
            mask = np.isin(y_true, crop_indices)
            if mask.sum() == 0:
                continue
            
            crop_y_true = y_true[mask]
            crop_y_pred = y_pred[mask]
            
            # Calcular accuracy
            crop_accuracy = accuracy_score(crop_y_true, crop_y_pred)
            per_crop[crop] = {
                'accuracy': crop_accuracy,
                'samples': mask.sum()
            }
        
        return per_crop
    
    def calculate_healthy_vs_diseased(self, y_true, y_pred, class_names):
        """
        Calcula métricas binarias: Healthy vs Diseased.
        
        Returns:
            dict: Métricas binarias y matriz de confusión 2x2
        """
        # Identificar clases healthy
        healthy_classes = [i for i, cls in enumerate(class_names) if 'healthy' in cls.lower()]
        
        # Convertir a binario: 0=healthy, 1=diseased
        y_true_binary = np.array([0 if y in healthy_classes else 1 for y in y_true])
        y_pred_binary = np.array([0 if y in healthy_classes else 1 for y in y_pred])
        
        # Matriz de confusión 2x2
        cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
        
        # Calcular métricas
        tn, fp, fn, tp = cm_binary.ravel()
        
        return {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'confusion_matrix': cm_binary,
            'true_negatives': tn,   # Healthy → Healthy (correcto)
            'false_positives': fp,  # Healthy → Diseased (falso positivo)
            'false_negatives': fn,  # Diseased → Healthy (falso negativo - CRÍTICO)
            'true_positives': tp    # Diseased → Diseased (correcto)
        }
    
    def analyze_top_confusions(self, cm, class_names, top_n=10):
        """
        Identifica las top N confusiones más frecuentes.
        
        Returns:
            list: Lista de tuplas (clase_real, clase_pred, frecuencia)
        """
        confusions = []
        
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    confusions.append((class_names[i], class_names[j], cm[i, j]))
        
        # Ordenar por frecuencia
        confusions.sort(key=lambda x: x[2], reverse=True)
        
        return confusions[:top_n]
    
    def evaluate(self, X_test, y_test, class_names):
        """
        Evalúa el modelo con métricas detalladas y visualizaciones avanzadas.
        
        Args:
            X_test: Datos de prueba
            y_test: Labels de prueba (one-hot)
            class_names: Nombres de las clases
        """
        print("\n" + "=" * 80)
        print("📊 EVALUACIÓN DETALLADA DEL MODELO")
        print("=" * 80)
        
        # Evaluar
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\n🎯 Resultados Globales:")
        print(f"  - Test Loss: {test_loss:.4f}")
        print(f"  - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Predicciones
        print("\n⏳ Calculando predicciones...")
        predictions = self.model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        # Inicializar sistema de métricas detalladas
        metrics_system = DetailedMetrics()
        
        # Calcular todas las métricas
        print("📈 Calculando métricas detalladas...")
        
        # 1. Métricas por clase
        class_metrics = metrics_system.calculate_per_class_metrics(
            true_classes, predicted_classes, class_names
        )
        
        # 2. Métricas por cultivo
        crop_metrics = metrics_system.calculate_per_crop_metrics(
            true_classes, predicted_classes, class_names
        )
        
        # 3. Healthy vs Diseased
        binary_metrics = metrics_system.calculate_healthy_vs_diseased(
            true_classes, predicted_classes, class_names
        )
        
        # 4. Top-K Accuracy
        top3_acc = top_k_accuracy_score(true_classes, predictions, k=3)
        top5_acc = top_k_accuracy_score(true_classes, predictions, k=5)
        
        # 5. Análisis de confusiones
        cm = confusion_matrix(true_classes, predicted_classes)
        top_confusions = metrics_system.analyze_top_confusions(cm, class_names, top_n=10)
        
        # Imprimir métricas en consola con formato estructurado
        metrics_system.print_detailed_metrics(
            class_metrics, crop_metrics, binary_metrics,
            top3_acc, top5_acc, top_confusions, class_names, 
            test_loss, test_accuracy
        )
        
        # Generar visualizaciones
        print("\n📊 Generando visualizaciones...")
        
        metrics_system.plot_confusion_matrix_detailed(cm, class_names)
        print("  ✅ confusion_matrix_detailed.png")
        
        metrics_system.plot_per_class_metrics(class_metrics, class_names)
        print("  ✅ per_class_metrics.png")
        
        metrics_system.plot_per_crop_performance(crop_metrics, test_accuracy)
        print("  ✅ per_crop_performance.png")
        
        metrics_system.plot_healthy_vs_diseased(binary_metrics)
        print("  ✅ healthy_vs_diseased.png")
        
        # Generar reporte detallado
        print("\n📝 Generando reporte detallado...")
        
        training_config = {
            'Resolución': f"{self.img_size[0]}x{self.img_size[1]}",
            'Transfer Learning': 'MobileNetV2' if self.use_transfer_learning else 'CNN desde cero',
            'Número de clases': self.num_classes
        }
        
        metrics_system.generate_detailed_report(
            test_loss, test_accuracy, class_metrics, crop_metrics,
            binary_metrics, top3_acc, top5_acc, top_confusions,
            class_names, training_config
        )
        
        print("\n" + "=" * 80)
        print("✅ EVALUACIÓN COMPLETADA")
        print("=" * 80)
        print("\n📁 Archivos generados:")
        print("  - models/visualizations/confusion_matrix_detailed.png")
        print("  - models/visualizations/per_class_metrics.png")
        print("  - models/visualizations/per_crop_performance.png")
        print("  - models/visualizations/healthy_vs_diseased.png")
        print("  - models/visualizations/training_report.txt")
        
        return test_loss, test_accuracy
    
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
    
    def _save_training_history(self, history, filepath):
        """Guarda el historial de entrenamiento como JSON."""
        history_dict = {
            'loss': [float(x) for x in history.history.get('loss', [])],
            'accuracy': [float(x) for x in history.history.get('accuracy', [])],
            'val_loss': [float(x) for x in history.history.get('val_loss', [])],
            'val_accuracy': [float(x) for x in history.history.get('val_accuracy', [])],
            'epochs': len(history.history.get('loss', [])),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=4)
        
        print(f"✅ Training history guardado: {filepath}")
    
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
    # IMG_SIZE importado desde config.py para consistencia (224x224)
    
    # Parámetros de entrenamiento
    EPOCHS_PHASE1 = 50      # Reducido para pruebas (era 100)
    EPOCHS_PHASE2 = 5       # Reducido (era 10)
    # Usar el batch size definido en backend/config.py por defecto (más seguro)
    BATCH_SIZE_OVERRIDE = BATCH_SIZE
    USE_TRANSFER_LEARNING = True
    DO_FINE_TUNING = False  # ❌ DESACTIVADO para reducir uso de memoria
    
    # Usar batch size (toma el mínimo entre override y config, seguridad)
    BATCH_SIZE_TRAIN = min(BATCH_SIZE_OVERRIDE, BATCH_SIZE)
    
    print("\n⚙️  CONFIGURACIÓN:")
    print(f"  - Transfer Learning: {'✅ MobileNetV2' if USE_TRANSFER_LEARNING else '❌'}")
    print(f"  - Resolución de Imagen: {IMG_SIZE[0]}x{IMG_SIZE[1]} ({IMG_SIZE[0]*IMG_SIZE[1]:,} píxeles)")
    print(f"  - Batch Size: {BATCH_SIZE_TRAIN} (recomendado: ajustar según RAM/GPU)")
    print(f"  - Epochs máximo Fase 1: {EPOCHS_PHASE1} (con early stopping 15-20)")
    print(f"  - Optimizer: Adam (lr=1e-4)")
    print(f"  - Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    if DO_FINE_TUNING and USE_TRANSFER_LEARNING:
        print(f"  - Épocas Fase 2 (fine-tuning gradual): {EPOCHS_PHASE2}")
        print(f"    • Subfase 2a: ~{EPOCHS_PHASE2//2} epochs (capas 101-154: features complejas)")
        print(f"    • Subfase 2b: ~{EPOCHS_PHASE2 - EPOCHS_PHASE2//2} epochs (capas 51-154: +features intermedias)")
        print(f"    • Capas 0-50 permanecen congeladas (features básicas de ImageNet)")
        print(f"  - Monitoreo: Sistema automático de detección de éxito/problemas activo")
    
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
        'balance': False  # DEBE coincidir con el cache creado
    }
    
    print("\n📂 Cargando datos desde cache (split 70/15/15)...")
    
    def _normalize_cache_result(res):
        """Normaliza resultado de cache a formato dict."""
        if res is None:
            return None
        if isinstance(res, tuple) and len(res) >= 3:
            data, labels, class_names = res[0], res[1], res[2]
            return {'X': data, 'y': labels, 'class_names': class_names}
        if isinstance(res, dict):
            if 'X' in res and 'y' in res and 'class_names' in res:
                return res
        return None
    
    # Intentar cargar desde cache con múltiples rutas posibles
    cache_dir = Path('backend/cache')
    pkl_files = list(cache_dir.glob('*_train.pkl'))
    
    train_data = None
    val_data = None
    test_data = None
    
    if pkl_files:
        # Usar el cache más reciente
        latest_cache = max(pkl_files, key=lambda p: p.stat().st_mtime)
        cache_key = latest_cache.stem.replace('_train', '')
        
        print(f"  ✅ Cache encontrado: {cache_key}")
        
        # Cargar usando la clave directa
        train_path = cache_dir / f"{cache_key}_train.pkl"
        val_path = cache_dir / f"{cache_key}_val.pkl"
        test_path = cache_dir / f"{cache_key}_test.pkl"
        
        if train_path.exists() and val_path.exists() and test_path.exists():
            import pickle
            
            with open(train_path, 'rb') as f:
                train_cache = pickle.load(f)
                train_data = {
                    'X': train_cache['data'],
                    'y': train_cache['labels'],
                    'class_names': train_cache['class_names']
                }
            
            with open(val_path, 'rb') as f:
                val_cache = pickle.load(f)
                val_data = {
                    'X': val_cache['data'],
                    'y': val_cache['labels'],
                    'class_names': val_cache['class_names']
                }
            
            with open(test_path, 'rb') as f:
                test_cache = pickle.load(f)
                test_data = {
                    'X': test_cache['data'],
                    'y': test_cache['labels'],
                    'class_names': test_cache['class_names']
                }
            
            print(f"  ✅ Train: {train_data['X'].shape[0]} muestras")
            print(f"  ✅ Val: {val_data['X'].shape[0]} muestras")
            print(f"  ✅ Test: {test_data['X'].shape[0]} muestras")
    else:
        # Intentar con el método original (por compatibilidad)
        train_res = cache.load(RAW_DATASET, config, 'train')
        val_res = cache.load(RAW_DATASET, config, 'val')
        test_res = cache.load(RAW_DATASET, config, 'test')

        train_data = _normalize_cache_result(train_res)
        val_data = _normalize_cache_result(val_res)
        test_data = _normalize_cache_result(test_res)
    
    if not train_data or not val_data or not test_data:
        print("\n❌ Error: Cache no encontrado.")
        print("\n🔧 SOLUCIÓN:")
        print("   Ejecuta primero uno de estos comandos para generar el cache:")
        print("   - python prepare_ultralight.py  (recomendado)")
        print("   - python prepare_minimal.py     (si ultralight falló)")
        print("   - .\\prepare_safe.bat            (alternativa)")
        print("\n   Luego vuelve a ejecutar: python backend/scripts/train.py")
        return
    
    X_train = train_data['X']
    y_train = train_data['y']
    class_names = train_data['class_names']

    X_val = val_data['X']
    y_val = val_data['y']

    X_test = test_data['X']
    y_test = test_data['y']

    # ---------------------- Seguridad: comprobación de memoria -----------------
    try:
        mem = psutil.virtual_memory()
        total_bytes = int(X_train.nbytes + X_val.nbytes + X_test.nbytes)
        total_mb = total_bytes / (1024 * 1024)
        avail_mb = mem.available / (1024 * 1024)
        used_percent = mem.percent
        
        print(f"\n🧮 VERIFICACIÓN DE MEMORIA:")
        print(f"   - Arrays dataset: {total_mb:.1f} MB")
        print(f"   - Memoria disponible: {avail_mb:.1f} MB")
        print(f"   - Memoria usada sistema: {used_percent:.1f}%")
        
        # LÍMITE ESTRICTO: arrays no pueden ocupar más del 50% de memoria disponible
        if total_bytes > mem.available * 0.5:
            print(f"\n❌ ERROR CRÍTICO: Arrays demasiado grandes ({total_mb:.1f} MB)")
            print(f"   Esto consumiría más del 50% de tu RAM disponible ({avail_mb:.1f} MB).")
            print(f"\n🔧 SOLUCIONES INMEDIATAS:")
            print(f"   1. Borrar cache: rm backend/cache/*.pkl")
            print(f"   2. Ya configuré APPLY_BALANCING=False en config.py")
            print(f"   3. Ejecuta: python backend/scripts/prepare_dataset.py")
            print(f"      (Selecciona opción 4 para limpiar, luego opción 2 para reprocesar)")
            print(f"\n   Esto generará un dataset MÁS PEQUEÑO sin oversampling.")
            print(f"\nAbortando para proteger tu sistema.")
            return
        
        # Advertencia si usa más del 30%
        if total_bytes > mem.available * 0.3:
            print(f"   ⚠️  ADVERTENCIA: Arrays usan {total_bytes*100/mem.available:.1f}% de RAM disponible")
            print(f"   El entrenamiento puede ser lento o inestable.")
        else:
            print(f"   ✅ Uso de memoria aceptable ({total_bytes*100/mem.available:.1f}% de RAM disponible)")
            
    except Exception as e:
        print(f"⚠️  No se pudo comprobar memoria: {e}")
    
    num_classes = len(class_names)
    
    print(f"\n✅ Datos cargados (split 70/15/15):")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_val: {X_val.shape}")
    print(f"  - X_test: {X_test.shape}")
    print(f"  - Clases: {class_names}")
    
    # Validación de shapes
    expected_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    actual_shape = X_train.shape[1:]
    if actual_shape != expected_shape:
        print(f"\n⚠️  ALERTA: Shape mismatch detectado!")
        print(f"  - Esperado: {expected_shape}")
        print(f"  - Actual: {actual_shape}")
        print(f"  - Acción requerida: BORRAR backend/cache/*.pkl y re-ejecutar")
        return
    else:
        print(f"  ✅ Validación de shape exitosa: {actual_shape}")
    
    # Calcular class_weights desde y_train
    from utils.model_metrics import calculate_class_weights
    class_weights = calculate_class_weights(y_train, num_classes)
    
    # Crear y construir modelo
    classifier = PlantDiseaseClassifier(
        img_size=IMG_SIZE,
        num_classes=num_classes,
        use_transfer_learning=USE_TRANSFER_LEARNING
    )
    
    classifier.build_model()
    
    # Entrenamiento inicial
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO INICIAL")
    print("=" * 60)
    
    total_start = time.time()
    
    classifier.train_with_arrays(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        epochs=EPOCHS_PHASE1,
        batch_size=BATCH_SIZE_TRAIN,
        class_weights=class_weights
    )
    
    # Fine-tuning
    if DO_FINE_TUNING and USE_TRANSFER_LEARNING:
        classifier.fine_tune(
            X_train, y_train,
            X_val, y_val,
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
    print("  - models/best_model.keras (mejor checkpoint)")
    print("  - models/last_model.keras (último checkpoint)")
    print("  - models/fruit_classifier.keras (modelo final)")
    print("  - models/class_mapping.json")
    print("  - metrics/training_history.json (pérdida y accuracy por epoch)")
    print("  - models/visualizations/")
    
    print("\n💡 CARACTERÍSTICAS:")
    print("  ✅ Split 70/15/15: Train/Val/Test separados")
    print("  ✅ Class Weights: Balance de clases aplicado")
    print("  ✅ Transfer Learning: MobileNetV2 pre-entrenado (ImageNet)")
    print("  ✅ Optimizer: Adam (lr=1e-4)")
    print("  ✅ Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    print("  ✅ Early Stopping: Patience 15 epochs")
    print("  ✅ Checkpoints: Best + Last model guardados")
    print("  ✅ Training History: JSON con métricas por epoch")
    print("  ✅ Batch Size: 64 (optimizado)")
    print("  ✅ Max Epochs: 100 (con early stopping)")
    
    print("\n🎯 PRÓXIMOS PASOS:")
    print("  1. Probar predicciones: python backend/scripts/predict.py <imagen>")
    print("  2. Iniciar API: python backend/app.py")
    print("  3. Re-entrenar: python backend/scripts/train.py")
    print("\n⚠️  IMPORTANTE - Si cambias IMG_SIZE:")
    print("  1. BORRAR: backend/cache/*.pkl")
    print("  2. BORRAR: models/*.keras (modelos incompatibles)")
    print("  3. Re-ejecutar entrenamiento completo")


if __name__ == "__main__":
    main()
