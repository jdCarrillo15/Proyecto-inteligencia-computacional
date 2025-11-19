"""
Script de prueba para validar el módulo de métricas.
Simula métricas de un modelo para verificar los umbrales.
"""

import numpy as np
import sys
from pathlib import Path

# Agregar el directorio backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from utils.model_metrics import ModelMetricsValidator, calculate_class_weights
from config import CLASSES, NUM_CLASSES

def simulate_predictions(num_samples=1000, quality='good'):
    """
    Simula predicciones de un modelo.
    
    Args:
        num_samples: Número de muestras por clase
        quality: 'good', 'medium', 'poor'
    """
    y_true = []
    y_pred = []
    
    for class_idx in range(NUM_CLASSES):
        # Generar etiquetas verdaderas
        true_labels = np.full(num_samples, class_idx)
        y_true.extend(true_labels)
        
        # Generar predicciones según la calidad
        if quality == 'good':
            # 85% accuracy
            correct = int(num_samples * 0.85)
            pred_labels = np.concatenate([
                np.full(correct, class_idx),
                np.random.randint(0, NUM_CLASSES, num_samples - correct)
            ])
        elif quality == 'medium':
            # 75% accuracy
            correct = int(num_samples * 0.75)
            pred_labels = np.concatenate([
                np.full(correct, class_idx),
                np.random.randint(0, NUM_CLASSES, num_samples - correct)
            ])
        else:  # poor
            # 60% accuracy
            correct = int(num_samples * 0.60)
            pred_labels = np.concatenate([
                np.full(correct, class_idx),
                np.random.randint(0, NUM_CLASSES, num_samples - correct)
            ])
        
        np.random.shuffle(pred_labels)
        y_pred.extend(pred_labels)
    
    return np.array(y_true), np.array(y_pred)


def test_validator():
    """Prueba el validador con diferentes escenarios."""
    
    print("=" * 80)
    print("TESTING MODEL METRICS VALIDATOR")
    print("=" * 80)
    
    scenarios = [
        ('Good Model (85% accuracy)', 'good'),
        ('Medium Model (75% accuracy)', 'medium'),
        ('Poor Model (60% accuracy)', 'poor')
    ]
    
    for scenario_name, quality in scenarios:
        print(f"\n\n{'=' * 80}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'=' * 80}\n")
        
        # Simular predicciones
        y_true, y_pred = simulate_predictions(num_samples=100, quality=quality)
        
        # Crear validador
        validator = ModelMetricsValidator(y_true, y_pred, CLASSES)
        
        # Generar reporte
        print(validator.generate_report(detailed=False))
        
        # Mostrar peores clases
        print("\n🔍 TOP 5 WORST PERFORMING CLASSES:")
        worst_classes = validator.get_worst_classes(n=5, metric='f1')
        for class_name, f1_score in worst_classes:
            print(f"   {class_name}: F1 = {f1_score:.2%}")
        
        # Estado de clases críticas
        print("\n⚠️  CRITICAL CLASSES STATUS:")
        critical_status = validator.get_critical_classes_status()
        for class_name, status_info in critical_status.items():
            print(f"   {status_info['status']} {class_name}: Recall = {status_info['recall']:.2%}")


def test_class_weights():
    """Prueba el cálculo de pesos de clase."""
    
    print("\n\n" + "=" * 80)
    print("TESTING CLASS WEIGHTS CALCULATION")
    print("=" * 80 + "\n")
    
    # Usar conteos reales del dataset (del audit report)
    y_train = []
    class_sizes = [
        2016,  # Apple___Apple_scab
        1987,  # Apple___Black_rot
        1760,  # Apple___Cedar_apple_rust
        2008,  # Apple___healthy
        1907,  # Corn_(maize)___Common_rust_
        1859,  # Corn_(maize)___healthy
        1908,  # Corn_(maize)___Northern_Leaf_Blight
        1939,  # Potato___Early_blight
        1824,  # Potato___healthy
        1939,  # Potato___Late_blight
        1702,  # Tomato___Bacterial_spot
        1920,  # Tomato___Early_blight
        1926,  # Tomato___healthy
        1851,  # Tomato___Late_blight
        1882   # Tomato___Leaf_Mold
    ]
    
    # Verificar que tenemos 15 clases
    assert len(class_sizes) == NUM_CLASSES, f"Error: esperadas {NUM_CLASSES} clases, pero se proporcionaron {len(class_sizes)}"
    
    for class_idx, size in enumerate(class_sizes):
        y_train.extend([class_idx] * size)
    
    y_train = np.array(y_train)
    
    # Calcular pesos
    class_weights = calculate_class_weights(y_train, NUM_CLASSES)
    
    print("Class Weights (inverse frequency, normalized):")
    print("(Based on real dataset distribution from audit)\n")
    for class_idx, weight in class_weights.items():
        class_name = CLASSES[class_idx]
        samples = class_sizes[class_idx]
        print(f"  {class_name:<45} Samples: {samples:>5}  Weight: {weight:.3f}")
    
    total_samples = sum(class_sizes)
    print(f"\n✅ Total samples: {total_samples:,}")
    print(f"✅ Average samples per class: {total_samples / NUM_CLASSES:.0f}")
    print(f"✅ Average weight: {np.mean(list(class_weights.values())):.3f} (should be ~1.0)")
    print(f"✅ Weight range: {min(class_weights.values()):.3f} - {max(class_weights.values()):.3f}")
    print(f"✅ Balance ratio: {max(class_sizes) / min(class_sizes):.2f}:1")


if __name__ == "__main__":
    test_validator()
    test_class_weights()
    
    print("\n\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 80)
