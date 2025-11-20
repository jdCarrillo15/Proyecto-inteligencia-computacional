#!/usr/bin/env python3
"""
Análisis de Problemas del Modelo - Paso 5
==========================================
Analiza fallos del modelo cuando no pasa validación.

Ejecutado automáticamente cuando el modelo es RECHAZADO o CONDICIONAL.

Funcionalidades:
✅ Identificar clases con bajo rendimiento
✅ Analizar confusion matrix (confusiones más frecuentes)
✅ Detectar patrones de error
✅ Generar reporte con recomendaciones específicas:
   - Aumentar data augmentation
   - Ajustar class weights
   - Cambiar arquitectura
   - Aumentar epochs o learning rate

Uso:
    python backend/scripts/analyze_failures.py
    python backend/scripts/analyze_failures.py --results metrics/evaluation_results.json
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Añadir backend al path
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from config import (
    PERFORMANCE_THRESHOLDS,
    CRITICAL_DISEASE_CLASSES,
    CRITICAL_DISEASE_MIN_RECALL,
    CLASSES
)

# Colores para terminal
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'


class FailureAnalyzer:
    """Analiza problemas del modelo y genera recomendaciones."""
    
    def __init__(self, results_path: str = None, validation_path: str = None):
        """
        Inicializa el analizador.
        
        Args:
            results_path: Ruta al JSON con resultados de evaluación
            validation_path: Ruta al JSON con reporte de validación
        """
        if results_path is None:
            results_path = backend_dir.parent / 'metrics' / 'evaluation_results.json'
        if validation_path is None:
            validation_path = backend_dir.parent / 'metrics' / 'validation_report.json'
        
        self.results_path = Path(results_path)
        self.validation_path = Path(validation_path)
        self.results = None
        self.validation = None
        self.analysis = {
            'timestamp': datetime.now().isoformat(),
            'problematic_classes': [],
            'confusion_patterns': [],
            'error_analysis': {},
            'recommendations': [],
            'priority_actions': []
        }
    
    def load_data(self) -> bool:
        """Carga datos de evaluación y validación."""
        # Cargar resultados de evaluación
        if not self.results_path.exists():
            print(f"{RED}❌ No se encontró {self.results_path}{RESET}")
            print(f"{YELLOW}⚠️  Ejecuta primero: python backend/scripts/evaluate_model.py{RESET}")
            return False
        
        try:
            with open(self.results_path, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
        except Exception as e:
            print(f"{RED}❌ Error al cargar resultados: {e}{RESET}")
            return False
        
        # Cargar reporte de validación (opcional)
        if self.validation_path.exists():
            try:
                with open(self.validation_path, 'r', encoding='utf-8') as f:
                    self.validation = json.load(f)
            except Exception as e:
                print(f"{YELLOW}⚠️  No se pudo cargar reporte de validación: {e}{RESET}")
        
        return True
    
    def identify_problematic_classes(self) -> List[Dict[str, Any]]:
        """
        Identifica clases con bajo rendimiento.
        
        Returns:
            Lista de clases problemáticas con sus métricas
        """
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}🔍 IDENTIFICACIÓN DE CLASES PROBLEMÁTICAS{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")
        
        per_class = self.results.get('per_class_metrics', {})
        problematic = []
        
        min_recall = PERFORMANCE_THRESHOLDS['min_recall_per_class']
        min_precision = PERFORMANCE_THRESHOLDS['min_precision_per_class']
        min_f1 = PERFORMANCE_THRESHOLDS['min_f1_per_class']
        
        target_recall = PERFORMANCE_THRESHOLDS['target_recall_per_class']
        target_precision = PERFORMANCE_THRESHOLDS['target_precision_per_class']
        target_f1 = PERFORMANCE_THRESHOLDS['target_f1_per_class']
        
        for class_name, metrics in per_class.items():
            recall = metrics.get('recall', 0.0)
            precision = metrics.get('precision', 0.0)
            f1 = metrics.get('f1_score', 0.0)
            support = metrics.get('support', 0)
            
            issues = []
            severity = 'OK'
            
            # Detectar problemas
            if recall < min_recall:
                issues.append(f"Recall crítico: {recall:.4f} < {min_recall:.2f}")
                severity = 'CRITICAL'
            elif recall < target_recall:
                issues.append(f"Recall bajo: {recall:.4f} < {target_recall:.2f}")
                if severity == 'OK':
                    severity = 'WARNING'
            
            if precision < min_precision:
                issues.append(f"Precision crítica: {precision:.4f} < {min_precision:.2f}")
                severity = 'CRITICAL'
            elif precision < target_precision:
                issues.append(f"Precision baja: {precision:.4f} < {target_precision:.2f}")
                if severity == 'OK':
                    severity = 'WARNING'
            
            if f1 < min_f1:
                issues.append(f"F1 crítico: {f1:.4f} < {min_f1:.2f}")
                severity = 'CRITICAL'
            elif f1 < target_f1:
                issues.append(f"F1 bajo: {f1:.4f} < {target_f1:.2f}")
                if severity == 'OK':
                    severity = 'WARNING'
            
            # Clases críticas requieren mayor recall
            if class_name in CRITICAL_DISEASE_CLASSES:
                if recall < CRITICAL_DISEASE_MIN_RECALL:
                    issues.append(f"Recall CRÍTICO (enfermedad crítica): {recall:.4f} < {CRITICAL_DISEASE_MIN_RECALL:.2f}")
                    severity = 'CRITICAL'
            
            if issues:
                problematic.append({
                    'class_name': class_name,
                    'severity': severity,
                    'recall': recall,
                    'precision': precision,
                    'f1_score': f1,
                    'support': support,
                    'issues': issues,
                    'is_critical_disease': class_name in CRITICAL_DISEASE_CLASSES
                })
        
        # Ordenar por severidad y luego por F1
        severity_order = {'CRITICAL': 0, 'WARNING': 1, 'OK': 2}
        problematic.sort(key=lambda x: (severity_order[x['severity']], x['f1_score']))
        
        # Mostrar resultados
        if not problematic:
            print(f"{GREEN}✅ No se detectaron clases problemáticas{RESET}")
            print(f"{GREEN}   Todas las clases cumplen con los umbrales mínimos{RESET}\n")
        else:
            print(f"{RED}⚠️  Se detectaron {len(problematic)} clases problemáticas:{RESET}\n")
            
            for item in problematic:
                if item['severity'] == 'CRITICAL':
                    icon = f"{RED}🔴{RESET}"
                    severity_label = f"{RED}CRÍTICO{RESET}"
                else:
                    icon = f"{YELLOW}⚠️{RESET}"
                    severity_label = f"{YELLOW}ADVERTENCIA{RESET}"
                
                class_display = item['class_name'].replace('_', ' ')
                if item['is_critical_disease']:
                    class_display = f"{class_display} {MAGENTA}[ENFERMEDAD CRÍTICA]{RESET}"
                
                print(f"{icon} {BOLD}{class_display}{RESET} - {severity_label}")
                print(f"   Recall: {item['recall']:.4f} | Precision: {item['precision']:.4f} | F1: {item['f1_score']:.4f} | Support: {item['support']}")
                
                for issue in item['issues']:
                    print(f"   • {issue}")
                print()
        
        self.analysis['problematic_classes'] = problematic
        return problematic
    
    def analyze_confusion_matrix(self) -> List[Dict[str, Any]]:
        """
        Analiza confusion matrix para detectar confusiones frecuentes.
        
        Returns:
            Lista de patrones de confusión detectados
        """
        print(f"{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}🔀 ANÁLISIS DE CONFUSION MATRIX{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")
        
        cm = self.results.get('confusion_matrix', [])
        class_names = self.results.get('class_names', CLASSES)
        
        if not cm:
            print(f"{RED}❌ No se encontró confusion matrix en los resultados{RESET}\n")
            return []
        
        cm = np.array(cm)
        n_classes = len(class_names)
        
        # Encontrar confusiones significativas (excluyendo diagonal)
        confusions = []
        
        for i in range(n_classes):
            total_true = cm[i, :].sum()
            if total_true == 0:
                continue
            
            for j in range(n_classes):
                if i == j:  # Saltar diagonal (predicciones correctas)
                    continue
                
                count = cm[i, j]
                if count == 0:
                    continue
                
                # Porcentaje de veces que la clase i se confundió con j
                confusion_rate = count / total_true
                
                # Solo considerar confusiones > 5%
                if confusion_rate > 0.05:
                    confusions.append({
                        'true_class': class_names[i],
                        'predicted_class': class_names[j],
                        'count': int(count),
                        'total_true': int(total_true),
                        'confusion_rate': confusion_rate
                    })
        
        # Ordenar por tasa de confusión
        confusions.sort(key=lambda x: x['confusion_rate'], reverse=True)
        
        # Mostrar top confusiones
        if not confusions:
            print(f"{GREEN}✅ No se detectaron confusiones significativas (>5%){RESET}\n")
        else:
            print(f"{YELLOW}⚠️  Top confusiones detectadas:{RESET}\n")
            
            for i, conf in enumerate(confusions[:10], 1):  # Top 10
                true_cls = conf['true_class'].replace('_', ' ')
                pred_cls = conf['predicted_class'].replace('_', ' ')
                rate = conf['confusion_rate'] * 100
                count = conf['count']
                total = conf['total_true']
                
                print(f"{i:2d}. {BOLD}{true_cls}{RESET}")
                print(f"    → confundido con {CYAN}{pred_cls}{RESET}")
                print(f"    📊 {count}/{total} veces ({rate:.1f}%)\n")
        
        self.analysis['confusion_patterns'] = confusions
        return confusions
    
    def detect_error_patterns(self) -> Dict[str, Any]:
        """
        Detecta patrones de error en el modelo.
        
        Returns:
            Diccionario con análisis de patrones
        """
        print(f"{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}🔎 DETECCIÓN DE PATRONES DE ERROR{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")
        
        patterns = {
            'low_recall_classes': [],
            'low_precision_classes': [],
            'inter_crop_confusion': defaultdict(list),
            'healthy_vs_diseased_errors': [],
            'similar_symptom_confusion': []
        }
        
        per_class = self.results.get('per_class_metrics', {})
        
        # Clases con bajo recall (falsos negativos altos)
        low_recall = [(name, m['recall']) for name, m in per_class.items() 
                      if m['recall'] < PERFORMANCE_THRESHOLDS['target_recall_per_class']]
        low_recall.sort(key=lambda x: x[1])
        
        if low_recall:
            print(f"{YELLOW}📉 Clases con bajo Recall (alto False Negative):{RESET}")
            for name, recall in low_recall[:5]:
                print(f"   • {name.replace('_', ' ')}: {recall:.4f}")
            patterns['low_recall_classes'] = low_recall
            print()
        
        # Clases con baja precision (falsos positivos altos)
        low_precision = [(name, m['precision']) for name, m in per_class.items() 
                        if m['precision'] < PERFORMANCE_THRESHOLDS['target_precision_per_class']]
        low_precision.sort(key=lambda x: x[1])
        
        if low_precision:
            print(f"{YELLOW}📈 Clases con baja Precision (alto False Positive):{RESET}")
            for name, precision in low_precision[:5]:
                print(f"   • {name.replace('_', ' ')}: {precision:.4f}")
            patterns['low_precision_classes'] = low_precision
            print()
        
        # Detectar confusión entre cultivos (Apple, Corn, Potato, Tomato)
        confusions = self.analysis.get('confusion_patterns', [])
        for conf in confusions:
            true_crop = conf['true_class'].split('___')[0]
            pred_crop = conf['predicted_class'].split('___')[0]
            
            if true_crop != pred_crop:
                patterns['inter_crop_confusion'][true_crop].append({
                    'confused_with': pred_crop,
                    'rate': conf['confusion_rate']
                })
        
        if patterns['inter_crop_confusion']:
            print(f"{RED}🌾 Confusión entre diferentes cultivos:{RESET}")
            for crop, confusions_list in patterns['inter_crop_confusion'].items():
                print(f"   • {crop}:")
                for c in confusions_list[:3]:
                    print(f"     → {c['confused_with']}: {c['rate']*100:.1f}%")
            print()
        
        # Detectar confusión healthy vs diseased
        for conf in confusions:
            true_healthy = 'healthy' in conf['true_class'].lower()
            pred_healthy = 'healthy' in conf['predicted_class'].lower()
            
            if true_healthy != pred_healthy:
                patterns['healthy_vs_diseased_errors'].append(conf)
        
        if patterns['healthy_vs_diseased_errors']:
            print(f"{MAGENTA}🔬 Confusión Healthy vs Diseased:{RESET}")
            for conf in patterns['healthy_vs_diseased_errors'][:5]:
                true_cls = conf['true_class'].replace('_', ' ')
                pred_cls = conf['predicted_class'].replace('_', ' ')
                print(f"   • {true_cls} → {pred_cls}: {conf['confusion_rate']*100:.1f}%")
            print()
        
        self.analysis['error_analysis'] = patterns
        return patterns
    
    def generate_recommendations(self, problematic_classes: List[Dict], 
                                 confusions: List[Dict],
                                 error_patterns: Dict) -> List[str]:
        """
        Genera recomendaciones específicas basadas en el análisis.
        
        Args:
            problematic_classes: Clases con bajo rendimiento
            confusions: Patrones de confusión
            error_patterns: Patrones de error detectados
        
        Returns:
            Lista de recomendaciones priorizadas
        """
        print(f"{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}💡 RECOMENDACIONES ESPECÍFICAS{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")
        
        recommendations = []
        priority_actions = []
        
        # Analizar problemas globales
        global_metrics = self.results.get('global_metrics', {})
        macro_f1 = global_metrics.get('macro_f1', 0.0)
        accuracy = global_metrics.get('accuracy', 0.0)
        
        # 1. Problemas de Data Augmentation
        if len(problematic_classes) > len(CLASSES) * 0.3:  # >30% clases problemáticas
            rec = {
                'category': 'Data Augmentation',
                'priority': 'HIGH',
                'issue': f'{len(problematic_classes)} clases ({len(problematic_classes)/len(CLASSES)*100:.0f}%) con bajo rendimiento',
                'actions': [
                    'Aumentar intensidad de augmentation en config.py:',
                    '  - rotation_range: 20 → 30',
                    '  - zoom_range: 0.2 → 0.3',
                    '  - Agregar brightness_range: [0.8, 1.2]',
                    '  - Agregar vertical_flip: True',
                    'Aplicar augmentation más agresivo en clases problemáticas',
                    'Considerar técnicas avanzadas: mixup, cutout, random erasing'
                ]
            }
            recommendations.append(rec)
            priority_actions.append("🎨 Aumentar Data Augmentation")
        
        # 2. Problemas de Class Weights
        if problematic_classes:
            low_support_classes = [c for c in problematic_classes if c['support'] < 100]
            if low_support_classes:
                rec = {
                    'category': 'Class Weights',
                    'priority': 'HIGH',
                    'issue': f'{len(low_support_classes)} clases con pocas muestras (<100)',
                    'actions': [
                        'Ajustar class weights en train.py para clases con bajo soporte:',
                        '  - Aumentar peso para: ' + ', '.join([c['class_name'] for c in low_support_classes[:3]]),
                        'Considerar oversampling para clases minoritarias',
                        'Verificar TARGET_SAMPLES_PER_CLASS en config.py',
                        'Aumentar TARGET_BALANCE_RATIO si hay desbalanceo severo'
                    ]
                }
                recommendations.append(rec)
                priority_actions.append("⚖️  Ajustar Class Weights")
        
        # 3. Problemas de Learning Rate
        if macro_f1 < PERFORMANCE_THRESHOLDS['min_macro_f1']:
            rec = {
                'category': 'Learning Rate',
                'priority': 'HIGH',
                'issue': f'Macro F1 bajo: {macro_f1:.4f} < {PERFORMANCE_THRESHOLDS["min_macro_f1"]:.2f}',
                'actions': [
                    'Ajustar learning rate en train.py:',
                    '  - Si converge rápido pero bajo rendimiento: aumentar a 2e-4 o 5e-4',
                    '  - Si oscila mucho: reducir a 5e-5',
                    'Probar learning rate scheduler diferente:',
                    '  - CosineAnnealingLR para convergencia suave',
                    '  - OneCycleLR para training más rápido',
                    'Ajustar ReduceLROnPlateau: patience=5 → 7, factor=0.5 → 0.3'
                ]
            }
            recommendations.append(rec)
            priority_actions.append("📊 Ajustar Learning Rate")
        
        # 4. Problemas de Epochs
        if accuracy < PERFORMANCE_THRESHOLDS['target_overall_accuracy']:
            rec = {
                'category': 'Training Duration',
                'priority': 'MEDIUM',
                'issue': f'Accuracy no alcanza objetivo: {accuracy:.4f} < {PERFORMANCE_THRESHOLDS["target_overall_accuracy"]:.2f}',
                'actions': [
                    'Aumentar número de epochs en train.py:',
                    '  - EPOCHS_PHASE1: 100 → 150',
                    'Ajustar early stopping patience:',
                    '  - patience: 15 → 20',
                    'Considerar entrenamiento en múltiples fases:',
                    '  - Fase 1: Congelar base, entrenar top layers',
                    '  - Fase 2: Fine-tuning gradual de capas base'
                ]
            }
            recommendations.append(rec)
            priority_actions.append("⏱️  Aumentar Epochs")
        
        # 5. Problemas de confusión entre clases similares
        if len(confusions) > 10:
            rec = {
                'category': 'Architecture',
                'priority': 'MEDIUM',
                'issue': f'{len(confusions)} confusiones significativas detectadas',
                'actions': [
                    'Considerar arquitectura más compleja:',
                    '  - EfficientNetB1 o B2 (más capacidad que MobileNetV2)',
                    '  - ResNet50 o ResNet101 para features más discriminativas',
                    'Aumentar tamaño de imagen:',
                    '  - IMG_SIZE: 224 → 299 (cuidado con memoria)',
                    'Agregar capas de attention para enfoque en síntomas:',
                    '  - Attention mechanism después de base model',
                    '  - Spatial attention para regiones importantes'
                ]
            }
            recommendations.append(rec)
            priority_actions.append("🏗️  Cambiar Arquitectura")
        
        # 6. Problemas específicos de clases críticas
        critical_issues = [c for c in problematic_classes if c['is_critical_disease']]
        if critical_issues:
            rec = {
                'category': 'Critical Diseases',
                'priority': 'CRITICAL',
                'issue': f'{len(critical_issues)} enfermedades críticas con bajo rendimiento',
                'actions': [
                    'PRIORIDAD MÁXIMA - Enfermedades críticas:',
                    '  - ' + ', '.join([c['class_name'] for c in critical_issues]),
                    'Acciones específicas:',
                    '  - Aumentar weight de estas clases en class_weights',
                    '  - Aplicar augmentation más intenso',
                    '  - Recolectar más datos si es posible',
                    '  - Usar focal loss (USE_FOCAL_LOSS=True en config.py)',
                    '  - Aumentar FOCAL_LOSS_GAMMA para enfoque en casos difíciles'
                ]
            }
            recommendations.insert(0, rec)  # Primera prioridad
            priority_actions.insert(0, "🔴 CRÍTICO: Mejorar enfermedades críticas")
        
        # 7. Confusión inter-cultivo
        if error_patterns['inter_crop_confusion']:
            rec = {
                'category': 'Data Quality',
                'priority': 'HIGH',
                'issue': 'Confusión entre diferentes tipos de cultivos',
                'actions': [
                    'Revisar calidad de imágenes:',
                    '  - Verificar que imágenes estén correctamente etiquetadas',
                    '  - Eliminar imágenes ambiguas o de baja calidad',
                    'Mejorar preprocessing:',
                    '  - Aplicar crop automático para enfocarse en hojas',
                    '  - Normalización por cultivo si hay diferencias de iluminación',
                    'Augmentation específico por cultivo:',
                    '  - Ajustar parámetros según características de cada planta'
                ]
            }
            recommendations.append(rec)
            priority_actions.append("🔍 Revisar calidad de datos")
        
        # 8. Confusión healthy vs diseased
        if error_patterns['healthy_vs_diseased_errors']:
            rec = {
                'category': 'Feature Learning',
                'priority': 'HIGH',
                'issue': 'Confusión entre plantas sanas y enfermas',
                'actions': [
                    'Mejorar capacidad de detectar síntomas sutiles:',
                    '  - Aumentar tamaño de imagen para capturar detalles',
                    '  - Usar arquitectura con mejor resolución espacial',
                    'Ajustar class weights:',
                    '  - Aumentar peso de clases "healthy" si tienen bajo recall',
                    '  - O aumentar peso de clases enfermas si tienen bajo recall',
                    'Preprocessing específico:',
                    '  - Realce de contraste para destacar síntomas',
                    '  - Color augmentation cuidadoso (no cambiar colores de síntomas)'
                ]
            }
            recommendations.append(rec)
            priority_actions.append("🌿 Mejorar detección healthy vs diseased")
        
        # Mostrar recomendaciones
        for i, rec in enumerate(recommendations, 1):
            if rec['priority'] == 'CRITICAL':
                priority_icon = f"{RED}🔴 CRÍTICO{RESET}"
            elif rec['priority'] == 'HIGH':
                priority_icon = f"{YELLOW}⚠️  ALTO{RESET}"
            else:
                priority_icon = f"{BLUE}ℹ️  MEDIO{RESET}"
            
            print(f"{BOLD}{i}. {rec['category']}{RESET} - {priority_icon}")
            print(f"   {CYAN}Problema:{RESET} {rec['issue']}")
            print(f"   {GREEN}Acciones:{RESET}")
            for action in rec['actions']:
                print(f"   {action}")
            print()
        
        self.analysis['recommendations'] = recommendations
        self.analysis['priority_actions'] = priority_actions
        
        return recommendations
    
    def generate_summary(self):
        """Genera resumen del análisis."""
        print(f"{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}📋 RESUMEN DEL ANÁLISIS{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")
        
        # Estadísticas
        n_problematic = len(self.analysis['problematic_classes'])
        n_critical = len([c for c in self.analysis['problematic_classes'] 
                         if c['severity'] == 'CRITICAL'])
        n_confusions = len(self.analysis['confusion_patterns'])
        n_recommendations = len(self.analysis['recommendations'])
        
        print(f"{BOLD}Clases Problemáticas:{RESET} {n_problematic}")
        if n_critical > 0:
            print(f"  {RED}• Críticas: {n_critical}{RESET}")
        if n_problematic - n_critical > 0:
            print(f"  {YELLOW}• Advertencias: {n_problematic - n_critical}{RESET}")
        
        print(f"\n{BOLD}Patrones de Confusión:{RESET} {n_confusions}")
        print(f"{BOLD}Recomendaciones:{RESET} {n_recommendations}")
        
        # Acciones prioritarias
        if self.analysis['priority_actions']:
            print(f"\n{BOLD}🎯 ACCIONES PRIORITARIAS:{RESET}")
            for i, action in enumerate(self.analysis['priority_actions'], 1):
                print(f"{i}. {action}")
    
    def save_report(self, output_path: str = None):
        """
        Guarda el reporte de análisis en JSON.
        
        Args:
            output_path: Ruta donde guardar el reporte
        """
        if output_path is None:
            output_path = backend_dir.parent / 'metrics' / 'failure_analysis.json'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\n{GREEN}✅ Reporte de análisis guardado en: {output_path}{RESET}")
    
    def run_analysis(self) -> Dict[str, Any]:
        """
        Ejecuta el análisis completo.
        
        Returns:
            Diccionario con análisis completo
        """
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}🔍 ANÁLISIS DE PROBLEMAS DEL MODELO - PASO 5{RESET}")
        print(f"{BOLD}{'='*70}{RESET}")
        
        # Cargar datos
        if not self.load_data():
            return None
        
        # 1. Identificar clases problemáticas
        problematic_classes = self.identify_problematic_classes()
        
        # 2. Analizar confusion matrix
        confusions = self.analyze_confusion_matrix()
        
        # 3. Detectar patrones de error
        error_patterns = self.detect_error_patterns()
        
        # 4. Generar recomendaciones
        recommendations = self.generate_recommendations(
            problematic_classes, confusions, error_patterns
        )
        
        # 5. Generar resumen
        self.generate_summary()
        
        # 6. Guardar reporte
        self.save_report()
        
        return self.analysis


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Análisis de problemas del modelo (Paso 5)'
    )
    parser.add_argument(
        '--results',
        type=str,
        default=None,
        help='Ruta al archivo JSON con resultados de evaluación'
    )
    parser.add_argument(
        '--validation',
        type=str,
        default=None,
        help='Ruta al archivo JSON con reporte de validación'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Ruta donde guardar el reporte de análisis'
    )
    
    args = parser.parse_args()
    
    # Ejecutar análisis
    analyzer = FailureAnalyzer(args.results, args.validation)
    result = analyzer.run_analysis()
    
    if result is not None:
        if args.output:
            analyzer.save_report(args.output)
    
    sys.exit(0 if result is not None else 1)


if __name__ == '__main__':
    main()
