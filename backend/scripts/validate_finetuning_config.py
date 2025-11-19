"""
Script de validación de las optimizaciones del fine-tuning (Paso 2.4).
Verifica que todos los parámetros estén configurados correctamente.
"""

import sys
from pathlib import Path

# Agregar el directorio backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


def validate_fine_tuning_config():
    """Valida la configuración de fine-tuning optimizada."""
    
    print("=" * 80)
    print("VALIDACIÓN DE OPTIMIZACIONES DE FINE-TUNING (PASO 2.4)")
    print("=" * 80)
    
    # Importar train.py para acceder a las configuraciones
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_module", 
        backend_dir / "scripts" / "train.py")
    train_module = importlib.util.module_from_spec(spec)
    
    print("\n⏳ Leyendo configuraciones de train.py...")
    
    # Leer archivo para verificar parámetros
    train_file = backend_dir / "scripts" / "train.py"
    with open(train_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # ========== CHECKLIST ==========
    print("\n" + "▼" * 80)
    print("CHECKLIST DE OPTIMIZACIONES")
    print("▼" * 80)
    
    checks = []
    
    # 1. Verificar EPOCHS_PHASE2 = 10
    if "EPOCHS_PHASE2 = 10" in content:
        checks.append(("✅", "EPOCHS_PHASE2 reducido de 20 a 10"))
    else:
        checks.append(("❌", "EPOCHS_PHASE2 NO está en 10"))
    
    # 2. Verificar Dropout = 0.5
    if "Dropout(0.5)" in content:
        dropout_count = content.count("Dropout(0.5)")
        checks.append(("✅", f"Dropout aumentado a 0.5 ({dropout_count} ocurrencias encontradas)"))
    else:
        checks.append(("❌", "Dropout NO está en 0.5"))
    
    # 3. Verificar Learning Rate Fase 2a = 0.00005
    if "learning_rate=0.00005" in content:
        checks.append(("✅", "Learning Rate Fase 2a = 0.00005 (más conservador)"))
    else:
        checks.append(("❌", "Learning Rate Fase 2a NO está en 0.00005"))
    
    # 4. Verificar Learning Rate Fase 2b = 0.00001
    if "learning_rate=0.00001" in content:
        checks.append(("✅", "Learning Rate Fase 2b = 0.00001 (ultra-conservador)"))
    else:
        checks.append(("❌", "Learning Rate Fase 2b NO está en 0.00001"))
    
    # 5. Verificar min_lr = 0.000001
    min_lr_count = content.count("min_lr=0.000001")
    if min_lr_count >= 2:
        checks.append(("✅", f"min_lr = 0.000001 en ReduceLROnPlateau ({min_lr_count} ocurrencias)"))
    else:
        checks.append(("❌", "min_lr NO está configurado correctamente"))
    
    # 6. Verificar Early Stopping patience
    if "patience=5," in content or "patience=7," in content:
        patience_5 = content.count("patience=5,")
        patience_7 = content.count("patience=7,")
        checks.append(("✅", f"Early Stopping patience configurado (5: {patience_5} veces, 7: {patience_7} veces)"))
    else:
        checks.append(("❌", "Early Stopping patience NO está configurado correctamente"))
    
    # 7. Verificar epochs_2a mínimo reducido a 5
    if "max(epochs_phase2 // 2, 5)" in content:
        checks.append(("✅", "epochs_2a mínimo reducido a 5 (antes era 7)"))
    else:
        checks.append(("⚠️", "epochs_2a mínimo NO está en 5 (podría estar en valor anterior)"))
    
    # Imprimir resultados
    for status, message in checks:
        print(f"  {status} {message}")
    
    # ========== RESUMEN DE CONFIGURACIÓN ==========
    print("\n" + "▼" * 80)
    print("RESUMEN DE CONFIGURACIÓN OPTIMIZADA")
    print("▼" * 80)
    
    print("\n📋 FASE 1: Entrenamiento Inicial")
    print("  • EPOCHS_PHASE1:       15")
    print("  • Learning Rate:       0.001")
    print("  • Dropout:             0.5 (aumentado desde 0.3)")
    print("  • Early Stop Patience: 7")
    print("  • Monitor:             val_accuracy")
    
    print("\n📋 FASE 2a: Fine-tuning Features Complejas (Capas 101-154)")
    print("  • Epochs:              ~5 (mínimo)")
    print("  • Learning Rate:       0.00005 (reducido desde 0.0001)")
    print("  • Early Stop Patience: 5 (reducido desde 6)")
    print("  • ReduceLR min_lr:     0.000001 (reducido desde 0.00001)")
    print("  • Monitor:             val_accuracy")
    
    print("\n📋 FASE 2b: Fine-tuning Features Intermedias (Capas 51-154)")
    print("  • Epochs:              ~5 (resto de EPOCHS_PHASE2)")
    print("  • Learning Rate:       0.00001 (reducido desde 0.00005)")
    print("  • Early Stop Patience: 5")
    print("  • ReduceLR min_lr:     0.000001 (reducido desde 0.00001)")
    print("  • Monitor:             val_accuracy")
    
    print("\n📋 TOTAL EPOCHS:")
    print("  • Fase 1:              15 epochs")
    print("  • Fase 2 (2a + 2b):    10 epochs (reducido desde 20)")
    print("  • TOTAL MÁXIMO:        25 epochs (reducido desde 35)")
    
    # ========== JUSTIFICACIÓN ==========
    print("\n" + "▼" * 80)
    print("JUSTIFICACIÓN DE CAMBIOS")
    print("▼" * 80)
    
    print("\n✅ EPOCHS_PHASE2: 20 → 10")
    print("   Razón: Con desbalanceo corregido, converge más rápido")
    print("   Beneficio: Evita overfitting en clases pequeñas")
    
    print("\n✅ Learning Rates más conservadores")
    print("   Fase 2a: 0.0001 → 0.00005 (50% más bajo)")
    print("   Fase 2b: 0.00005 → 0.00001 (50% más bajo)")
    print("   Razón: Proteger features pre-entrenadas de ImageNet")
    print("   Beneficio: Reduce riesgo de catastrophic forgetting")
    
    print("\n✅ Early Stopping más agresivo")
    print("   Patience: 6-7 → 5-7 (más agresivo en fine-tuning)")
    print("   Razón: Parar antes si no mejora")
    print("   Beneficio: Evita sobreentrenamiento y ahorra tiempo")
    
    print("\n✅ Dropout aumentado")
    print("   Dropout rate: 0.3 → 0.5 (66% más alto)")
    print("   Razón: Prevenir memorización del desbalanceo")
    print("   Beneficio: Mejor generalización en clases minoritarias")
    
    # ========== IMPACTO ESPERADO ==========
    print("\n" + "▼" * 80)
    print("IMPACTO ESPERADO")
    print("▼" * 80)
    
    print("\n🎯 Resultado:")
    print("  ✅ Entrenamiento más estable")
    print("  ✅ Menos overfitting en clases minoritarias")
    print("  ✅ Mejor preservación de features ImageNet")
    print("  ✅ Convergencia más rápida (~30% reducción en epochs totales)")
    print("  ✅ Menor tiempo de entrenamiento (~20-25 min ahorrados)")
    
    # Verificar si todos los checks pasaron
    print("\n" + "=" * 80)
    failed_checks = [check for check in checks if check[0] == "❌"]
    warning_checks = [check for check in checks if check[0] == "⚠️"]
    
    if not failed_checks and not warning_checks:
        print("✅ VALIDACIÓN EXITOSA - TODAS LAS OPTIMIZACIONES APLICADAS")
    elif warning_checks and not failed_checks:
        print("⚠️  VALIDACIÓN CON ADVERTENCIAS - REVISAR CONFIGURACIONES")
    else:
        print("❌ VALIDACIÓN FALLIDA - CORREGIR CONFIGURACIONES")
    print("=" * 80)
    
    return len(failed_checks) == 0


def main():
    """Ejecuta la validación."""
    success = validate_fine_tuning_config()
    
    if success:
        print("\n💡 Próximos pasos:")
        print("  1. Preparar datos: python backend/scripts/prepare_dataset.py")
        print("  2. Entrenar modelo: python backend/scripts/train.py")
        print("  3. Observar convergencia más rápida y estable")
    else:
        print("\n⚠️  Revisar y corregir las configuraciones marcadas con ❌")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
