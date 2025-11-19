"""
Utilidades para cálculo y validación de métricas del modelo.
Implementa los umbrales definidos en MODEL_REQUIREMENTS.md
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import sys
from pathlib import Path

# Agregar el directorio backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from config import (
    PERFORMANCE_THRESHOLDS,
    CRITICAL_DISEASE_CLASSES,
    CRITICAL_DISEASE_MIN_RECALL,
    CRITICAL_DISEASE_TARGET_RECALL,
    CLASSES
)


class ModelMetricsValidator:
    """Validador de métricas del modelo según requisitos del proyecto."""
    
    def __init__(self, y_true, y_pred, class_names=None):
        """
        Inicializa el validador.
        
        Args:
            y_true: Etiquetas verdaderas (one-hot o índices)
            y_pred: Predicciones del modelo (one-hot, probabilidades o índices)
            class_names: Lista de nombres de clases (opcional)
        """
        self.y_true = self._convert_to_indices(y_true)
        self.y_pred = self._convert_to_indices(y_pred)
        self.class_names = class_names or CLASSES
        
        # Calcular métricas
        self.metrics = self._calculate_metrics()
        
    def _convert_to_indices(self, y):
        """Convierte one-hot o probabilidades a índices de clase."""
        if len(y.shape) > 1 and y.shape[1] > 1:
            return np.argmax(y, axis=1)
        return y
    
    def _calculate_metrics(self):
        """Calcula todas las métricas requeridas."""
        metrics = {}
        
        # Métricas globales
        metrics['overall_accuracy'] = accuracy_score(self.y_true, self.y_pred)
        
        # Métricas por clase (macro average)
        metrics['macro_precision'] = precision_score(
            self.y_true, self.y_pred, average='macro', zero_division=0
        )
        metrics['macro_recall'] = recall_score(
            self.y_true, self.y_pred, average='macro', zero_division=0
        )
        metrics['macro_f1'] = f1_score(
            self.y_true, self.y_pred, average='macro', zero_division=0
        )
        
        # Métricas ponderadas
        metrics['weighted_precision'] = precision_score(
            self.y_true, self.y_pred, average='weighted', zero_division=0
        )
        metrics['weighted_recall'] = recall_score(
            self.y_true, self.y_pred, average='weighted', zero_division=0
        )
        metrics['weighted_f1'] = f1_score(
            self.y_true, self.y_pred, average='weighted', zero_division=0
        )
        
        # Métricas por clase individual
        precision_per_class = precision_score(
            self.y_true, self.y_pred, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            self.y_true, self.y_pred, average=None, zero_division=0
        )
        f1_per_class = f1_score(
            self.y_true, self.y_pred, average=None, zero_division=0
        )
        
        metrics['per_class'] = {}
        for idx, class_name in enumerate(self.class_names):
            metrics['per_class'][class_name] = {
                'precision': float(precision_per_class[idx]),
                'recall': float(recall_per_class[idx]),
                'f1': float(f1_per_class[idx])
            }
        
        # Matriz de confusión
        metrics['confusion_matrix'] = confusion_matrix(self.y_true, self.y_pred)
        
        return metrics
    
    def validate_requirements(self):
        """
        Valida si el modelo cumple con los requisitos mínimos.
        
        Returns:
            dict: Diccionario con resultados de validación
        """
        validation_results = {
            'passed': True,
            'status': 'APPROVED',
            'failures': [],
            'warnings': [],
            'summary': {}
        }
        
        thresholds = PERFORMANCE_THRESHOLDS
        
        # 1. Validar Macro F1-Score
        if self.metrics['macro_f1'] < thresholds['min_macro_f1']:
            validation_results['passed'] = False
            validation_results['failures'].append(
                f"Macro F1-Score ({self.metrics['macro_f1']:.2%}) < "
                f"mínimo requerido ({thresholds['min_macro_f1']:.2%})"
            )
        elif self.metrics['macro_f1'] < thresholds['target_macro_f1']:
            validation_results['warnings'].append(
                f"Macro F1-Score ({self.metrics['macro_f1']:.2%}) < "
                f"objetivo ({thresholds['target_macro_f1']:.2%})"
            )
        
        # 2. Validar Overall Accuracy
        if self.metrics['overall_accuracy'] < thresholds['min_overall_accuracy']:
            validation_results['passed'] = False
            validation_results['failures'].append(
                f"Overall Accuracy ({self.metrics['overall_accuracy']:.2%}) < "
                f"mínimo requerido ({thresholds['min_overall_accuracy']:.2%})"
            )
        elif self.metrics['overall_accuracy'] < thresholds['target_overall_accuracy']:
            validation_results['warnings'].append(
                f"Overall Accuracy ({self.metrics['overall_accuracy']:.2%}) < "
                f"objetivo ({thresholds['target_overall_accuracy']:.2%})"
            )
        
        # 3. Validar métricas por clase
        classes_below_threshold = []
        critical_classes_below_threshold = []
        
        for class_name, class_metrics in self.metrics['per_class'].items():
            recall = class_metrics['recall']
            f1 = class_metrics['f1']
            
            # Verificar recall mínimo
            if recall < thresholds['min_recall_per_class']:
                classes_below_threshold.append(class_name)
                validation_results['passed'] = False
                validation_results['failures'].append(
                    f"Clase '{class_name}': Recall ({recall:.2%}) < "
                    f"mínimo ({thresholds['min_recall_per_class']:.2%})"
                )
            
            # Verificar F1 mínimo
            if f1 < thresholds['min_f1_per_class']:
                if class_name not in classes_below_threshold:
                    classes_below_threshold.append(class_name)
                validation_results['passed'] = False
                validation_results['failures'].append(
                    f"Clase '{class_name}': F1-Score ({f1:.2%}) < "
                    f"mínimo ({thresholds['min_f1_per_class']:.2%})"
                )
            
            # Verificar clases críticas
            if class_name in CRITICAL_DISEASE_CLASSES:
                if recall < CRITICAL_DISEASE_MIN_RECALL:
                    critical_classes_below_threshold.append(class_name)
                    validation_results['passed'] = False
                    validation_results['failures'].append(
                        f"Clase CRÍTICA '{class_name}': Recall ({recall:.2%}) < "
                        f"mínimo crítico ({CRITICAL_DISEASE_MIN_RECALL:.2%})"
                    )
                elif recall < CRITICAL_DISEASE_TARGET_RECALL:
                    validation_results['warnings'].append(
                        f"Clase CRÍTICA '{class_name}': Recall ({recall:.2%}) < "
                        f"objetivo ({CRITICAL_DISEASE_TARGET_RECALL:.2%})"
                    )
        
        # 4. Verificar número de clases problemáticas
        if len(classes_below_threshold) > 3:
            validation_results['passed'] = False
            validation_results['failures'].append(
                f"Demasiadas clases bajo umbral: {len(classes_below_threshold)} > 3 permitidas"
            )
        
        # Determinar status final
        if not validation_results['passed']:
            validation_results['status'] = 'REJECTED'
        elif validation_results['warnings']:
            validation_results['status'] = 'CONDITIONAL'
        else:
            validation_results['status'] = 'APPROVED'
        
        # Resumen
        validation_results['summary'] = {
            'macro_f1': self.metrics['macro_f1'],
            'overall_accuracy': self.metrics['overall_accuracy'],
            'classes_below_threshold': len(classes_below_threshold),
            'critical_classes_below_threshold': len(critical_classes_below_threshold),
            'total_failures': len(validation_results['failures']),
            'total_warnings': len(validation_results['warnings'])
        }
        
        return validation_results
    
    def generate_report(self, detailed=True):
        """
        Genera un reporte detallado de las métricas.
        
        Args:
            detailed: Si True, incluye métricas por clase
            
        Returns:
            str: Reporte formateado
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MODEL PERFORMANCE REPORT")
        report_lines.append("=" * 80)
        
        # Métricas globales
        report_lines.append("\n📊 OVERALL METRICS:")
        report_lines.append(f"   Overall Accuracy:    {self.metrics['overall_accuracy']:.2%}")
        report_lines.append(f"   Macro F1-Score:      {self.metrics['macro_f1']:.2%}")
        report_lines.append(f"   Weighted F1-Score:   {self.metrics['weighted_f1']:.2%}")
        report_lines.append(f"   Macro Recall:        {self.metrics['macro_recall']:.2%}")
        report_lines.append(f"   Macro Precision:     {self.metrics['macro_precision']:.2%}")
        
        # Validación
        validation = self.validate_requirements()
        report_lines.append(f"\n🎯 VALIDATION STATUS: {validation['status']}")
        
        if validation['failures']:
            report_lines.append("\n❌ FAILURES:")
            for failure in validation['failures']:
                report_lines.append(f"   - {failure}")
        
        if validation['warnings']:
            report_lines.append("\n⚠️  WARNINGS:")
            for warning in validation['warnings']:
                report_lines.append(f"   - {warning}")
        
        if validation['status'] == 'APPROVED':
            report_lines.append("\n✅ Model meets all minimum requirements!")
        
        # Métricas por clase
        if detailed:
            report_lines.append("\n📋 PER-CLASS METRICS:")
            report_lines.append("-" * 80)
            report_lines.append(f"{'Class':<45} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
            report_lines.append("-" * 80)
            
            for class_name, class_metrics in sorted(
                self.metrics['per_class'].items(),
                key=lambda x: x[1]['f1']
            ):
                is_critical = class_name in CRITICAL_DISEASE_CLASSES
                marker = "⚠️ " if is_critical else "   "
                
                report_lines.append(
                    f"{marker}{class_name:<42} "
                    f"{class_metrics['precision']:>9.2%} "
                    f"{class_metrics['recall']:>9.2%} "
                    f"{class_metrics['f1']:>9.2%}"
                )
            
            report_lines.append("-" * 80)
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def get_worst_classes(self, n=5, metric='f1'):
        """
        Obtiene las n clases con peor rendimiento.
        
        Args:
            n: Número de clases a retornar
            metric: Métrica a usar ('precision', 'recall', 'f1')
            
        Returns:
            list: Lista de tuplas (class_name, metric_value)
        """
        class_scores = [
            (class_name, class_metrics[metric])
            for class_name, class_metrics in self.metrics['per_class'].items()
        ]
        
        return sorted(class_scores, key=lambda x: x[1])[:n]
    
    def get_critical_classes_status(self):
        """
        Obtiene el estado de las clases críticas.
        
        Returns:
            dict: Estado de cada clase crítica
        """
        status = {}
        
        for class_name in CRITICAL_DISEASE_CLASSES:
            if class_name in self.metrics['per_class']:
                class_metrics = self.metrics['per_class'][class_name]
                recall = class_metrics['recall']
                
                if recall >= CRITICAL_DISEASE_TARGET_RECALL:
                    status_text = "✅ EXCELLENT"
                elif recall >= CRITICAL_DISEASE_MIN_RECALL:
                    status_text = "⚠️  ACCEPTABLE"
                else:
                    status_text = "❌ CRITICAL"
                
                status[class_name] = {
                    'recall': recall,
                    'status': status_text,
                    'metrics': class_metrics
                }
        
        return status


def calculate_class_weights(y_train, num_classes):
    """
    Calcula pesos de clase para manejar desbalanceo.
    
    Args:
        y_train: Etiquetas de entrenamiento (one-hot o índices)
        num_classes: Número total de clases
        
    Returns:
        dict: Diccionario de pesos por clase
    """
    # Convertir a índices si es one-hot
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train = np.argmax(y_train, axis=1)
    
    # Contar muestras por clase
    class_counts = np.bincount(y_train, minlength=num_classes)
    
    # Calcular peso inverso de frecuencia
    total_samples = len(y_train)
    class_weights = total_samples / (num_classes * class_counts)
    
    # Normalizar para que el peso promedio sea 1.0
    class_weights = class_weights / class_weights.mean()
    
    # Convertir a diccionario
    return {i: weight for i, weight in enumerate(class_weights)}


if __name__ == "__main__":
    # Ejemplo de uso
    print("Model Metrics Validator - Utility Module")
    print("Este módulo debe ser importado en el script de entrenamiento.")
    print("\nUso:")
    print("  from utils.model_metrics import ModelMetricsValidator")
    print("  validator = ModelMetricsValidator(y_true, y_pred, class_names)")
    print("  print(validator.generate_report())")
