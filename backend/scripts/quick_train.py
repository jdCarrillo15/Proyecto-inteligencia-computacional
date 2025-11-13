"""
🚀 ENTRENAMIENTO COMPLETO TODO-EN-UNO CON PKL CACHE
Este script ejecuta todo el pipeline de forma automática y optimizada:
1. Prepara los datos (con cache PKL)
2. Entrena el modelo (ultra-rápido)
3. Evalúa y guarda resultados
"""

import sys
from pathlib import Path
import time

# Agregar backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from scripts.data_preparation_fast import FastDatasetProcessor
from scripts.train_model_fast import FastFruitClassifier
import tensorflow as tf
import numpy as np

# Configurar seeds
tf.random.set_seed(42)
np.random.seed(42)


def main():
    print("\n" + "=" * 70)
    print("🚀 PIPELINE COMPLETO DE ENTRENAMIENTO CON PKL CACHE")
    print("=" * 70)
    
    # =================================================================
    # CONFIGURACIÓN
    # =================================================================
    RAW_DATASET = "dataset/raw"
    PROCESSED_DATASET = "dataset/processed"
    IMG_SIZE = (100, 100)
    
    # Parámetros de entrenamiento
    EPOCHS_PHASE1 = 20      # Entrenamiento inicial (aumentado)
    EPOCHS_PHASE2 = 8       # Fine-tuning (reducido)
    BATCH_SIZE = 32         # Batch reducido para mejor regularización
    USE_TRANSFER_LEARNING = True
    DO_FINE_TUNING = False   # Desactivado por defecto (causa overfitting)
    FORCE_REPROCESS = False  # True para ignorar cache y reprocesar
    
    print("\n⚙️  CONFIGURACIÓN DEL PIPELINE:")
    print("=" * 70)
    print(f"📁 Dataset: {RAW_DATASET}")
    print(f"📐 Tamaño imagen: {IMG_SIZE}")
    print(f"🔄 Transfer Learning: {'✅ MobileNetV2' if USE_TRANSFER_LEARNING else '❌ Desactivado'}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print(f"📊 Épocas Fase 1: {EPOCHS_PHASE1}")
    if DO_FINE_TUNING and USE_TRANSFER_LEARNING:
        print(f"🔥 Épocas Fine-tuning: {EPOCHS_PHASE2}")
    print(f"💾 Forzar reprocesamiento: {'✅ Sí' if FORCE_REPROCESS else '❌ No (usar cache)'}")
    print("=" * 70)
    
    # =================================================================
    # PASO 1: PREPARACIÓN DE DATOS
    # =================================================================
    print("\n" + "=" * 70)
    print("📊 PASO 1/3: PREPARACIÓN DE DATOS")
    print("=" * 70)
    
    processor = FastDatasetProcessor(RAW_DATASET, PROCESSED_DATASET, IMG_SIZE)
    
    start_prep = time.time()
    result = processor.prepare_optimized(
        use_cache=True,
        force_reprocess=FORCE_REPROCESS
    )
    prep_time = time.time() - start_prep
    
    if not result:
        print("\n❌ Error en la preparación de datos")
        return
    
    X_train, y_train, X_test, y_test, class_names = result
    num_classes = len(class_names)
    
    # Validar que hay datos de entrenamiento
    if len(X_train) == 0:
        print("\n❌ Error: No hay datos de entrenamiento")
        print("\nVerifica que el dataset esté en la ubicación correcta:")
        print("  dataset/raw/New Plant Diseases Dataset(Augmented)/train/")
        return
    
    # Si no hay test, advertir pero no detener
    if len(X_test) == 0:
        print("\n⚠️  ADVERTENCIA: No hay datos de prueba")
        print("El sistema dividirá los datos de entrenamiento automáticamente...")
        print("\nRegenerando cache con división train/test...")
        
        # Forzar reprocesamiento
        processor = FastDatasetProcessor(RAW_DATASET, PROCESSED_DATASET, IMG_SIZE)
        start_prep = time.time()
        result = processor.prepare_optimized(use_cache=True, force_reprocess=True)
        prep_time = time.time() - start_prep
        
        if not result:
            print("\n❌ Error en la preparación de datos")
            return
        
        X_train, y_train, X_test, y_test, class_names = result
    
    print(f"\n✅ Datos preparados en {prep_time:.2f} segundos")
    print(f"  - Entrenamiento: {X_train.shape}")
    print(f"  - Prueba: {X_test.shape}")
    print(f"  - Clases ({num_classes}): {class_names}")
    
    # =================================================================
    # PASO 2: CONSTRUCCIÓN Y ENTRENAMIENTO DEL MODELO
    # =================================================================
    print("\n" + "=" * 70)
    print("🤖 PASO 2/3: ENTRENAMIENTO DEL MODELO")
    print("=" * 70)
    
    # Crear clasificador
    classifier = FastFruitClassifier(
        img_size=IMG_SIZE,
        num_classes=num_classes,
        use_transfer_learning=USE_TRANSFER_LEARNING
    )
    
    # Construir modelo
    classifier.build_model()
    
    # Entrenar
    start_train = time.time()
    
    # Fase 1: Entrenamiento inicial
    print("\n🎯 Fase 1: Entrenamiento inicial")
    classifier.train_with_arrays(
        X_train, y_train,
        X_test, y_test,
        epochs=EPOCHS_PHASE1,
        batch_size=BATCH_SIZE
    )
    
    # Fase 2: Fine-tuning (opcional)
    if DO_FINE_TUNING and USE_TRANSFER_LEARNING:
        print("\n🔥 Fase 2: Fine-tuning")
        classifier.fine_tune(
            X_train, y_train,
            X_test, y_test,
            epochs=EPOCHS_PHASE2,
            batch_size=BATCH_SIZE
        )
    
    train_time = time.time() - start_train
    
    print(f"\n✅ Entrenamiento completado en {train_time/60:.2f} minutos")
    
    # =================================================================
    # PASO 3: EVALUACIÓN Y GUARDADO
    # =================================================================
    print("\n" + "=" * 70)
    print("📈 PASO 3/3: EVALUACIÓN Y GUARDADO")
    print("=" * 70)
    
    # Evaluar
    test_loss, test_accuracy = classifier.evaluate(X_test, y_test, class_names)
    
    # Visualizaciones
    classifier.plot_training_history()
    
    # Guardar modelo
    classifier.save_model('models/fruit_classifier.keras', class_names)
    
    # =================================================================
    # RESUMEN FINAL
    # =================================================================
    total_time = prep_time + train_time
    
    print("\n" + "=" * 70)
    print("✅ ¡PIPELINE COMPLETADO EXITOSAMENTE!")
    print("=" * 70)
    
    print(f"\n⏱️  TIEMPOS:")
    print(f"  - Preparación: {prep_time:.2f} segundos")
    print(f"  - Entrenamiento: {train_time/60:.2f} minutos")
    print(f"  - Total: {total_time/60:.2f} minutos")
    
    print(f"\n📊 RESULTADOS:")
    print(f"  - Precisión en test: {test_accuracy*100:.2f}%")
    print(f"  - Pérdida en test: {test_loss:.4f}")
    
    print(f"\n📁 ARCHIVOS GENERADOS:")
    print("  ✅ models/best_model.keras (mejor modelo)")
    print("  ✅ models/fruit_classifier.keras (modelo final)")
    print("  ✅ models/class_mapping.json (mapeo de clases)")
    print("  ✅ models/visualizations/ (gráficos)")
    print("  ✅ backend/cache/*.pkl (datos procesados)")
    
    print(f"\n💡 VENTAJAS DEL SISTEMA OPTIMIZADO:")
    print("  ✅ Cache PKL: Próximos entrenamientos serán instantáneos")
    print("  ✅ Transfer Learning: 3-5x más rápido que entrenar desde cero")
    print("  ✅ Batch optimizado: Máximo rendimiento de GPU/CPU")
    print("  ✅ Early stopping: Evita overfitting automáticamente")
    
    print(f"\n🎯 PRÓXIMOS PASOS:")
    print("  1. Probar el modelo: python backend/scripts/predict.py")
    print("  2. Iniciar la app: python backend/app.py")
    print("  3. Re-entrenar (instantáneo): python backend/scripts/quick_train.py")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Entrenamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
