#!/usr/bin/env python3
"""
Script para diagnosticar problemas del modelo y el dataset
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset():
    """Analiza el balance del dataset"""
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          🔍 DIAGNÓSTICO DEL DATASET Y MODELO                ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "dataset" / "raw"
    
    if not raw_dir.exists():
        print("❌ No se encontró el dataset en dataset/raw/")
        return
    
    # Contar imágenes por clase
    print("📊 BALANCE DEL DATASET")
    print("─" * 60)
    
    fruit_counts = {}
    fruit_emojis = {
        'manzana': '🍎',
        'banano': '🍌',
        'naranja': '🍊',
        'mango': '🥭',
        'pera': '🍐'
    }
    
    total = 0
    for fruit_dir in sorted(raw_dir.iterdir()):
        if fruit_dir.is_dir():
            count = sum(1 for f in fruit_dir.rglob('*') 
                       if f.suffix.lower() in ['.jpg', '.jpeg', '.png'])
            fruit_counts[fruit_dir.name] = count
            total += count
    
    # Mostrar estadísticas
    max_count = max(fruit_counts.values())
    
    for fruit, count in sorted(fruit_counts.items(), key=lambda x: x[1], reverse=True):
        emoji = fruit_emojis.get(fruit, '🍏')
        percentage = (count / total) * 100
        bar = '█' * int(percentage / 2)
        ratio = count / max_count
        
        print(f"{emoji} {fruit.capitalize():10} | {count:6,} ({percentage:5.1f}%) {bar}")
        
        if ratio < 0.1:
            print(f"   ⚠️  CRÍTICO: Solo {ratio*100:.1f}% del máximo")
        elif ratio < 0.3:
            print(f"   ⚠️  BAJO: Solo {ratio*100:.1f}% del máximo")
    
    print("─" * 60)
    print(f"Total: {total:,} imágenes")
    print()
    
    # Análisis de desbalance
    print("🎯 ANÁLISIS DE DESBALANCE")
    print("─" * 60)
    
    min_count = min(fruit_counts.values())
    max_count = max(fruit_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"Ratio de desbalance: {imbalance_ratio:.1f}x")
    
    if imbalance_ratio > 20:
        print("❌ CRÍTICO: Desbalance extremo (>20x)")
        print("   El modelo estará muy sesgado hacia la clase mayoritaria")
    elif imbalance_ratio > 10:
        print("⚠️  ALTO: Desbalance significativo (>10x)")
        print("   Se recomienda usar técnicas de balanceo")
    elif imbalance_ratio > 5:
        print("⚠️  MODERADO: Desbalance notable (>5x)")
        print("   Considerar usar class weights")
    else:
        print("✅ ACEPTABLE: Dataset relativamente balanceado")
    
    print()
    
    # Calcular class weights recomendados
    print("⚖️  CLASS WEIGHTS RECOMENDADOS")
    print("─" * 60)
    print("Para compensar el desbalance, usa estos pesos en el entrenamiento:")
    print()
    
    class_weights = {}
    for fruit, count in fruit_counts.items():
        weight = total / (len(fruit_counts) * count)
        class_weights[fruit] = weight
        emoji = fruit_emojis.get(fruit, '🍏')
        print(f"{emoji} {fruit.capitalize():10} : {weight:.2f}")
    
    print()
    
    # Verificar modelo entrenado
    print("🧠 ANÁLISIS DEL MODELO")
    print("─" * 60)
    
    model_path = project_root / "models" / "fruit_classifier.h5"
    class_mapping_path = project_root / "models" / "class_mapping.json"
    
    if not model_path.exists():
        print("❌ No se encontró modelo entrenado")
        print("   Ejecuta: python scripts/train_model.py")
    else:
        print(f"✅ Modelo encontrado: {model_path.name}")
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   Tamaño: {size_mb:.1f} MB")
        
        if class_mapping_path.exists():
            with open(class_mapping_path, 'r') as f:
                class_mapping = json.load(f)
            print(f"   Clases: {', '.join(class_mapping.values())}")
    
    print()
    
    # Recomendaciones
    print("💡 RECOMENDACIONES PARA MEJORAR LA PRECISIÓN")
    print("─" * 60)
    
    recommendations = []
    
    if imbalance_ratio > 10:
        recommendations.append(
            "1. 🔄 BALANCEAR DATASET:\n"
            "   - Aumentar imágenes de clases minoritarias\n"
            "   - Usar data augmentation agresivo en clases pequeñas\n"
            "   - Considerar undersampling de manzanas"
        )
    
    recommendations.append(
        "2. ⚖️  USAR CLASS WEIGHTS:\n"
        "   - Modifica train_model.py para incluir class_weight\n"
        "   - Esto penaliza más los errores en clases minoritarias"
    )
    
    recommendations.append(
        "3. 🎨 DATA AUGMENTATION:\n"
        "   - Rotaciones, flips, zoom, brillo\n"
        "   - Especialmente importante para naranjas y mangos"
    )
    
    recommendations.append(
        "4. 🎯 AJUSTAR ARQUITECTURA:\n"
        "   - Más capas convolucionales para capturar detalles\n"
        "   - Dropout más alto para evitar overfitting\n"
        "   - Transfer learning (ResNet, MobileNet)"
    )
    
    recommendations.append(
        "5. 📊 VALIDACIÓN:\n"
        "   - Usar stratified split para mantener proporciones\n"
        "   - Validar con imágenes de diferentes fuentes\n"
        "   - Revisar matriz de confusión"
    )
    
    for rec in recommendations:
        print(rec)
        print()
    
    print("═" * 60)
    print()
    
    # Guardar class weights en archivo
    weights_file = project_root / "models" / "class_weights.json"
    weights_file.parent.mkdir(exist_ok=True)
    
    # Ordenar por índice de clase
    ordered_weights = {}
    class_order = ['manzana', 'banano', 'mango', 'naranja', 'pera']
    for idx, fruit in enumerate(class_order):
        if fruit in class_weights:
            ordered_weights[idx] = round(class_weights[fruit], 3)
    
    with open(weights_file, 'w') as f:
        json.dump(ordered_weights, f, indent=2)
    
    print(f"💾 Class weights guardados en: {weights_file}")
    print()
    
    return fruit_counts, class_weights

if __name__ == "__main__":
    analyze_dataset()
