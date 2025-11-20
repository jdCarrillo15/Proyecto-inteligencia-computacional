#!/usr/bin/env python3
"""
Script MINIMAL - Para sistemas con RAM MUY limitada (4-6 GB)
=============================================================

Configuración MÍNIMA ABSOLUTA:
- Máximo 100 imágenes por clase
- Sin balanceo
- Total: ~1,500 imágenes
- RAM estimada: ~1 GB

SOLO para validar que el sistema funciona.
"""

import os
import sys
from pathlib import Path

# Agregar backend al path
backend_dir = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_dir))

from scripts.prepare_dataset import DatasetProcessor

def main():
    print("\n" + "=" * 80)
    print("🔴 MODO MINIMAL - Para Sistemas con RAM MUY Limitada (< 6 GB)")
    print("=" * 80)
    
    # Configuración MINIMAL
    RAW_DATASET = "dataset/raw"
    PROCESSED_DATASET = "dataset/processed"
    IMG_SIZE = (224, 224)
    APPLY_BALANCING = False  # CRÍTICO: Desactivado
    TARGET_SAMPLES = 0  # 0 = No augmentation (evita OOM)
    MAX_SAMPLES_PER_CLASS = 100  # MINIMAL: solo 100 por clase
    
    print("\n⚙️  Configuración MINIMAL:")
    print(f"   - Límite por clase: {MAX_SAMPLES_PER_CLASS} imágenes")
    print(f"   - Total máximo: ~{MAX_SAMPLES_PER_CLASS * 15} imágenes")
    print(f"   - RAM estimada: ~1 GB")
    print(f"   - Balanceo: DESACTIVADO")
    print("\n🔴 Esta es la configuración MÍNIMA para probar funcionamiento.")
    print("    Solo úsala si ULTRALIGHT también falló.\n")
    
    input("Presiona Enter para continuar o Ctrl+C para cancelar...")
    
    # Crear procesador
    processor = DatasetProcessor(
        RAW_DATASET,
        PROCESSED_DATASET,
        IMG_SIZE,
        apply_balancing=APPLY_BALANCING,
        target_samples=TARGET_SAMPLES,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS
    )
    
    # Preparar dataset
    result = processor.prepare_dataset(
        use_cache=True,
        force_reprocess=True
    )
    
    if result:
        X_train, y_train, X_val, y_val, X_test, y_test, class_names, class_weights = result
        
        print("\n" + "=" * 80)
        print("✅ PREPARACIÓN MINIMAL COMPLETADA")
        print("=" * 80)
        print(f"\n📊 Resumen:")
        print(f"  - Train: {X_train.shape[0]} muestras")
        print(f"  - Val: {X_val.shape[0]} muestras")
        print(f"  - Test: {X_test.shape[0]} muestras")
        print(f"  - Total: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]} muestras")
        print(f"\n⚠️  ADVERTENCIA: Con tan pocos datos, el modelo NO será preciso.")
        print("   Esto es SOLO para validar que el entrenamiento funciona.")
        print("\n💡 Si funcionó, aumenta gradualmente MAX_SAMPLES_PER_CLASS:")
        print("   - Siguiente prueba: 150")
        print("   - Luego: 200")
        print("   - Objetivo ideal: 500-800")
    else:
        print("\n❌ Error en la preparación")

if __name__ == "__main__":
    main()
