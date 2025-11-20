#!/usr/bin/env python3
"""
Script ULTRA-LIGHT de preparación - Para sistemas con RAM limitada (< 8 GB)
=============================================================================

Configuración extremadamente conservadora:
- Máximo 150 imágenes por clase
- Sin balanceo
- Total: ~2,250 imágenes
- RAM estimada: ~1.5 GB
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
    print("🛡️  MODO ULTRA-LIGHT - Para Sistemas con RAM Limitada")
    print("=" * 80)
    
    # Configuración ULTRA conservadora
    RAW_DATASET = "dataset/raw"
    PROCESSED_DATASET = "dataset/processed"
    IMG_SIZE = (224, 224)
    APPLY_BALANCING = False  # CRÍTICO: Desactivado
    TARGET_SAMPLES = 0  # 0 = No augmentation (evita OOM)
    MAX_SAMPLES_PER_CLASS = 150  # ULTRA-LIGHT: solo 150 por clase
    
    print("\n⚙️  Configuración ULTRA-LIGHT:")
    print(f"   - Límite por clase: {MAX_SAMPLES_PER_CLASS} imágenes")
    print(f"   - Total máximo: ~{MAX_SAMPLES_PER_CLASS * 15} imágenes")
    print(f"   - RAM estimada: ~1.5 GB")
    print(f"   - Balanceo: DESACTIVADO")
    print("\n⚠️  Esta configuración usa MÍNIMO de datos para validar que funcione.")
    print("    Una vez confirmado, puedes aumentar gradualmente MAX_SAMPLES_PER_CLASS.\n")
    
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
        force_reprocess=True  # Forzar reprocesamiento
    )
    
    if result:
        X_train, y_train, X_val, y_val, X_test, y_test, class_names, class_weights = result
        
        print("\n" + "=" * 80)
        print("✅ PREPARACIÓN ULTRA-LIGHT COMPLETADA")
        print("=" * 80)
        print(f"\n📊 Resumen:")
        print(f"  - Train: {X_train.shape[0]} muestras")
        print(f"  - Val: {X_val.shape[0]} muestras")
        print(f"  - Test: {X_test.shape[0]} muestras")
        print(f"  - Total: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]} muestras")
        print(f"\n💡 Siguiente paso:")
        print("   python backend/scripts/train.py")
    else:
        print("\n❌ Error en la preparación")

if __name__ == "__main__":
    main()
