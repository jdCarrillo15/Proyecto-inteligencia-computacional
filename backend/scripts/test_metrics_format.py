"""
Script de prueba para validar el formato de métricas detalladas.
Simula datos de evaluación para verificar que el output sea correcto.
"""

import numpy as np
import sys
from pathlib import Path

# Agregar el directorio backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from scripts.detailed_metrics import DetailedMetrics


def generate_mock_data():
    """Genera datos simulados para prueba."""
    
    # 15 clases
    class_names = [
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
    
    # Simular predicciones y verdades
    np.random.seed(42)
    n_samples = 1800
    
    y_true = np.random.randint(0, 15, n_samples)
    
    # Simular predicciones con ~75% accuracy
    y_pred = y_true.copy()
    noise_mask = np.random.random(n_samples) > 0.75
    y_pred[noise_mask] = np.random.randint(0, 15, noise_mask.sum())
    
    # Simular probabilidades (para top-k)
    predictions_proba = np.random.dirichlet(np.ones(15) * 5, size=n_samples)
    
    # Asegurar que la clase verdadera tenga alta probabilidad
    for i in range(n_samples):
        if np.random.random() > 0.25:  # 75% de las veces
            predictions_proba[i, y_true[i]] = np.max(predictions_proba[i]) + 0.3
            predictions_proba[i] = predictions_proba[i] / predictions_proba[i].sum()
    
    return y_true, y_pred, predictions_proba, class_names


def main():
    """Ejecuta prueba de formato de métricas."""
    
    print("=" * 80)
    print("PRUEBA DE FORMATO DE MÉTRICAS DETALLADAS")
    print("=" * 80)
    
    # Generar datos mock
    y_true, y_pred, predictions_proba, class_names = generate_mock_data()
    
    # Inicializar sistema de métricas
    metrics_system = DetailedMetrics()
    
    # Calcular todas las métricas
    print("\n⏳ Calculando métricas...")
    
    class_metrics = metrics_system.calculate_per_class_metrics(
        y_true, y_pred, class_names
    )
    
    crop_metrics = metrics_system.calculate_per_crop_metrics(
        y_true, y_pred, class_names
    )
    
    binary_metrics = metrics_system.calculate_healthy_vs_diseased(
        y_true, y_pred, class_names
    )
    
    # Top-K Accuracy simulado
    from sklearn.metrics import top_k_accuracy_score, confusion_matrix
    top3_acc = top_k_accuracy_score(y_true, predictions_proba, k=3)
    top5_acc = top_k_accuracy_score(y_true, predictions_proba, k=5)
    
    # Top confusiones
    cm = confusion_matrix(y_true, y_pred)
    top_confusions = metrics_system.analyze_top_confusions(cm, class_names, top_n=10)
    
    # Test loss y accuracy simulados
    test_loss = 0.8234
    test_accuracy = (y_true == y_pred).mean()
    
    # Imprimir métricas
    print("✅ Métricas calculadas\n")
    
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
    
    # Generar reporte
    print("\n📝 Generando reporte...")
    
    training_config = {
        'Resolución': '224x224',
        'Transfer Learning': 'MobileNetV2',
        'Número de clases': 15,
        'Batch size': 16,
        'Épocas entrenadas': 25
    }
    
    metrics_system.generate_detailed_report(
        test_loss, test_accuracy, class_metrics, crop_metrics,
        binary_metrics, top3_acc, top5_acc, top_confusions,
        class_names, training_config
    )
    
    print("\n" + "=" * 80)
    print("✅ PRUEBA COMPLETADA")
    print("=" * 80)
    print("\n📁 Archivos generados en: models/visualizations/")
    print("  - confusion_matrix_detailed.png")
    print("  - per_class_metrics.png")
    print("  - per_crop_performance.png")
    print("  - healthy_vs_diseased.png")
    print("  - training_report.txt")


if __name__ == '__main__':
    main()
