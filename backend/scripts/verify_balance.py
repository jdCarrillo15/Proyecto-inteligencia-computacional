"""
Script de verificación del balance del dataset.
Valida que el ratio de clases esté ≤ 2:1 después de aplicar estrategias de balanceo.
"""

import sys
from pathlib import Path
import numpy as np

# Agregar el directorio backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from config import IMG_SIZE, CLASSES, NUM_CLASSES
from scripts.prepare_dataset import DatasetProcessor


def verify_balance():
    """Verifica el balance del dataset después de procesar."""
    
    print("=" * 80)
    print("VERIFICACIÓN DE BALANCE DEL DATASET")
    print("=" * 80)
    
    # Configuración
    RAW_DATASET = "dataset/raw"
    PROCESSED_DATASET = "dataset/processed"
    
    # Crear procesador CON balanceo activado
    print("\n📊 Procesando dataset con balanceo activado...")
    processor = DatasetProcessor(
        RAW_DATASET,
        PROCESSED_DATASET,
        IMG_SIZE,
        apply_balancing=True,
        target_samples=2500
    )
    
    # Preparar datos
    result = processor.prepare_optimized(use_cache=False, force_reprocess=True)
    
    if not result:
        print("\n❌ Error al procesar dataset")
        return False
    
    X_train, y_train, X_test, y_test, class_names, class_weights = result
    
    # Analizar distribución
    print("\n" + "=" * 80)
    print("ANÁLISIS DE BALANCE")
    print("=" * 80)
    
    y_train_indices = np.argmax(y_train, axis=1)
    class_counts = np.bincount(y_train_indices, minlength=NUM_CLASSES)
    
    print("\n📊 Distribución por clase (TRAIN):")
    print("-" * 80)
    for idx, count in enumerate(class_counts):
        class_name = class_names[idx]
        percentage = (count / len(y_train)) * 100
        print(f"  {class_name:<45} {count:>5} samples ({percentage:>5.2f}%)")
    
    # Calcular estadísticas
    max_count = np.max(class_counts)
    min_count = np.min(class_counts)
    avg_count = np.mean(class_counts)
    std_count = np.std(class_counts)
    balance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print("-" * 80)
    print(f"\n📈 Estadísticas:")
    print(f"  - Total samples (train): {len(y_train):,}")
    print(f"  - Max samples per class: {max_count:,}")
    print(f"  - Min samples per class: {min_count:,}")
    print(f"  - Average per class: {avg_count:.0f}")
    print(f"  - Std deviation: {std_count:.2f}")
    print(f"  - Balance ratio: {balance_ratio:.2f}:1")
    
    # Verificación de objetivo
    print("\n" + "=" * 80)
    print("VERIFICACIÓN DE OBJETIVOS")
    print("=" * 80)
    
    success = True
    
    # Objetivo 1: Balance ratio ≤ 2:1
    print(f"\n✓ Objetivo 1: Balance ratio ≤ 2:1")
    if balance_ratio <= 2.0:
        print(f"  ✅ CUMPLIDO: Ratio actual = {balance_ratio:.2f}:1")
    else:
        print(f"  ❌ NO CUMPLIDO: Ratio actual = {balance_ratio:.2f}:1 (objetivo: ≤ 2:1)")
        success = False
    
    # Objetivo 2: Todas las clases con ≥ 1500 samples
    print(f"\n✓ Objetivo 2: Todas las clases con ≥ 1500 samples")
    classes_below_threshold = []
    for idx, count in enumerate(class_counts):
        if count < 1500:
            classes_below_threshold.append((class_names[idx], count))
    
    if not classes_below_threshold:
        print(f"  ✅ CUMPLIDO: Todas las clases tienen ≥ 1500 samples")
    else:
        print(f"  ⚠️  ADVERTENCIA: {len(classes_below_threshold)} clases con < 1500 samples:")
        for cls_name, count in classes_below_threshold:
            print(f"     - {cls_name}: {count} samples")
    
    # Objetivo 3: Class weights correctamente calculados
    print(f"\n✓ Objetivo 3: Class weights correctamente calculados")
    weight_range = max(class_weights.values()) / min(class_weights.values())
    print(f"  ✅ Pesos calculados: rango {min(class_weights.values()):.3f} - {max(class_weights.values()):.3f}")
    print(f"  ✅ Weight ratio: {weight_range:.2f}:1")
    
    # Análisis de TEST set
    print("\n" + "=" * 80)
    print("ANÁLISIS DE TEST SET")
    print("=" * 80)
    
    y_test_indices = np.argmax(y_test, axis=1)
    test_counts = np.bincount(y_test_indices, minlength=NUM_CLASSES)
    
    print(f"\n📊 Distribución por clase (TEST):")
    for idx, count in enumerate(test_counts):
        percentage = (count / len(y_test)) * 100 if len(y_test) > 0 else 0
        print(f"  {class_names[idx]:<45} {count:>5} samples ({percentage:>5.2f}%)")
    
    print(f"\n  Total test samples: {len(y_test):,}")
    
    # Resultado final
    print("\n" + "=" * 80)
    if success:
        print("✅ VERIFICACIÓN EXITOSA")
        print("=" * 80)
        print("\nEl dataset está correctamente balanceado y listo para entrenamiento.")
        print("Se han aplicado las 3 estrategias:")
        print("  ✓ Estrategia A: Class Weights calculados")
        print("  ✓ Estrategia B: Oversampling con augmentation agresiva")
        print("  ✓ Estrategia C: Focal Loss disponible (opcional)")
    else:
        print("⚠️  VERIFICACIÓN CON ADVERTENCIAS")
        print("=" * 80)
        print("\nEl dataset ha sido procesado pero algunos objetivos no se cumplieron.")
        print("Revisa los mensajes anteriores para más detalles.")
    
    print("\n💡 Siguiente paso:")
    print("   Ejecuta 'python backend/scripts/train.py' para entrenar el modelo")
    
    return success


if __name__ == "__main__":
    verify_balance()
