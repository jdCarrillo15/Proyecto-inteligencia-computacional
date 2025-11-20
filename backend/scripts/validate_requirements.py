#!/usr/bin/env python3
"""
Validación Automática de Requisitos - Paso 4
==============================================
Compara métricas reales contra requisitos definidos en config.py
Genera reporte de validación y define acciones siguientes.

Este script se ejecuta AUTOMÁTICAMENTE después de evaluate_model.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Añadir backend al path para imports
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from config import (
    PERFORMANCE_THRESHOLDS,
    CRITICAL_DISEASE_CLASSES,
    CRITICAL_DISEASE_MIN_RECALL
)

# Colores para terminal
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'


class RequirementsValidator:
    """Valida métricas del modelo contra requisitos obligatorios."""
    
    def __init__(self, results_path: str = None):
        """
        Inicializa el validador.
        
        Args:
            results_path: Ruta al archivo JSON con resultados de evaluación
        """
        if results_path is None:
            results_path = backend_dir.parent / 'metrics' / 'evaluation_results.json'
        
        self.results_path = Path(results_path)
        self.results = None
        self.validation_report = {
            'timestamp': datetime.now().isoformat(),
            'obligatory_checks': {},
            'optional_checks': {},
            'critical_classes': {},
            'overall_status': None,
            'recommended_actions': [],
            'warnings': [],
            'passed': False
        }
    
    def load_results(self) -> bool:
        """Carga los resultados de evaluación."""
        if not self.results_path.exists():
            print(f"{RED}❌ No se encontró {self.results_path}{RESET}")
            print(f"{YELLOW}⚠️  Ejecuta primero: python backend/scripts/evaluate_model.py{RESET}")
            return False
        
        try:
            with open(self.results_path, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
            return True
        except Exception as e:
            print(f"{RED}❌ Error al cargar resultados: {e}{RESET}")
            return False
    
    def validate_obligatory_requirements(self) -> bool:
        """
        Valida requisitos OBLIGATORIOS (Paso 4).
        
        Returns:
            True si todos los requisitos obligatorios se cumplen
        """
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}🔍 VALIDACIÓN DE REQUISITOS OBLIGATORIOS{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")
        
        global_metrics = self.results.get('global_metrics', {})
        
        # 1. Macro F1-Score ≥ 75%
        macro_f1 = global_metrics.get('macro_f1', 0.0)
        min_macro_f1 = PERFORMANCE_THRESHOLDS['min_macro_f1']
        macro_f1_passed = macro_f1 >= min_macro_f1
        
        self.validation_report['obligatory_checks']['macro_f1'] = {
            'value': macro_f1,
            'threshold': min_macro_f1,
            'passed': macro_f1_passed
        }
        
        status_icon = f"{GREEN}✅{RESET}" if macro_f1_passed else f"{RED}❌{RESET}"
        print(f"{status_icon} Macro F1-Score: {macro_f1:.4f} (≥ {min_macro_f1:.2f} requerido)")
        
        # 2. Accuracy ≥ 75%
        accuracy = global_metrics.get('accuracy', 0.0)
        min_accuracy = PERFORMANCE_THRESHOLDS['min_overall_accuracy']
        accuracy_passed = accuracy >= min_accuracy
        
        self.validation_report['obligatory_checks']['accuracy'] = {
            'value': accuracy,
            'threshold': min_accuracy,
            'passed': accuracy_passed
        }
        
        status_icon = f"{GREEN}✅{RESET}" if accuracy_passed else f"{RED}❌{RESET}"
        print(f"{status_icon} Accuracy:       {accuracy:.4f} (≥ {min_accuracy:.2f} requerido)")
        
        # 3. Recall clases críticas ≥ 80%
        print(f"\n{BOLD}📌 Clases Críticas (Recall ≥ {CRITICAL_DISEASE_MIN_RECALL:.0%}):{RESET}")
        
        per_class = self.results.get('per_class_metrics', {})
        critical_passed = True
        critical_results = {}
        
        for critical_class in CRITICAL_DISEASE_CLASSES:
            class_metrics = per_class.get(critical_class, {})
            recall = class_metrics.get('recall', 0.0)
            passed = recall >= CRITICAL_DISEASE_MIN_RECALL
            
            critical_results[critical_class] = {
                'recall': recall,
                'threshold': CRITICAL_DISEASE_MIN_RECALL,
                'passed': passed
            }
            
            if not passed:
                critical_passed = False
            
            status_icon = f"{GREEN}✅{RESET}" if passed else f"{RED}❌{RESET}"
            class_display = critical_class.replace('_', ' ')
            print(f"  {status_icon} {class_display}: {recall:.4f}")
        
        self.validation_report['critical_classes'] = critical_results
        self.validation_report['obligatory_checks']['critical_recall'] = {
            'passed': critical_passed,
            'threshold': CRITICAL_DISEASE_MIN_RECALL
        }
        
        # Resultado general
        all_passed = macro_f1_passed and accuracy_passed and critical_passed
        
        print(f"\n{BOLD}{'='*70}{RESET}")
        if all_passed:
            print(f"{GREEN}{BOLD}✅ TODOS LOS REQUISITOS OBLIGATORIOS CUMPLIDOS{RESET}")
        else:
            print(f"{RED}{BOLD}❌ REQUISITOS OBLIGATORIOS NO CUMPLIDOS{RESET}")
        print(f"{BOLD}{'='*70}{RESET}")
        
        return all_passed
    
    def validate_optional_requirements(self) -> Dict[str, bool]:
        """
        Valida requisitos opcionales (objetivos e ideales).
        
        Returns:
            Diccionario con resultados de validación opcional
        """
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}📊 VALIDACIÓN DE OBJETIVOS (OPCIONALES){RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")
        
        global_metrics = self.results.get('global_metrics', {})
        
        # Objetivos
        macro_f1 = global_metrics.get('macro_f1', 0.0)
        weighted_f1 = global_metrics.get('weighted_f1', 0.0)
        accuracy = global_metrics.get('accuracy', 0.0)
        
        target_macro_f1 = PERFORMANCE_THRESHOLDS['target_macro_f1']
        target_weighted_f1 = PERFORMANCE_THRESHOLDS['target_weighted_f1']
        target_accuracy = PERFORMANCE_THRESHOLDS['target_overall_accuracy']
        
        ideal_macro_f1 = PERFORMANCE_THRESHOLDS['ideal_macro_f1']
        ideal_accuracy = PERFORMANCE_THRESHOLDS['ideal_overall_accuracy']
        
        # Macro F1
        macro_f1_target = macro_f1 >= target_macro_f1
        macro_f1_ideal = macro_f1 >= ideal_macro_f1
        
        if macro_f1_ideal:
            icon = f"{GREEN}🌟{RESET}"
            level = "IDEAL"
        elif macro_f1_target:
            icon = f"{BLUE}🎯{RESET}"
            level = "OBJETIVO"
        else:
            icon = f"{YELLOW}⚠️{RESET}"
            level = "MÍNIMO"
        
        print(f"{icon} Macro F1-Score: {macro_f1:.4f} - {level}")
        print(f"   Objetivo: {target_macro_f1:.2f} | Ideal: {ideal_macro_f1:.2f}")
        
        self.validation_report['optional_checks']['macro_f1'] = {
            'value': macro_f1,
            'target': target_macro_f1,
            'ideal': ideal_macro_f1,
            'target_met': macro_f1_target,
            'ideal_met': macro_f1_ideal
        }
        
        # Accuracy
        accuracy_target = accuracy >= target_accuracy
        accuracy_ideal = accuracy >= ideal_accuracy
        
        if accuracy_ideal:
            icon = f"{GREEN}🌟{RESET}"
            level = "IDEAL"
        elif accuracy_target:
            icon = f"{BLUE}🎯{RESET}"
            level = "OBJETIVO"
        else:
            icon = f"{YELLOW}⚠️{RESET}"
            level = "MÍNIMO"
        
        print(f"{icon} Accuracy:       {accuracy:.4f} - {level}")
        print(f"   Objetivo: {target_accuracy:.2f} | Ideal: {ideal_accuracy:.2f}")
        
        self.validation_report['optional_checks']['accuracy'] = {
            'value': accuracy,
            'target': target_accuracy,
            'ideal': ideal_accuracy,
            'target_met': accuracy_target,
            'ideal_met': accuracy_ideal
        }
        
        # Weighted F1
        weighted_f1_target = weighted_f1 >= target_weighted_f1
        
        icon = f"{BLUE}🎯{RESET}" if weighted_f1_target else f"{YELLOW}⚠️{RESET}"
        print(f"{icon} Weighted F1:    {weighted_f1:.4f}")
        print(f"   Objetivo: {target_weighted_f1:.2f}")
        
        self.validation_report['optional_checks']['weighted_f1'] = {
            'value': weighted_f1,
            'target': target_weighted_f1,
            'target_met': weighted_f1_target
        }
        
        return {
            'targets_met': macro_f1_target and accuracy_target and weighted_f1_target,
            'ideal_met': macro_f1_ideal and accuracy_ideal
        }
    
    def determine_actions(self, obligatory_passed: bool, 
                         optional_results: Dict[str, bool]) -> str:
        """
        Determina el estado del modelo y acciones recomendadas.
        
        Args:
            obligatory_passed: Si los requisitos obligatorios se cumplieron
            optional_results: Resultados de validación opcional
        
        Returns:
            Estado del modelo: 'APPROVED', 'CONDITIONAL', 'REJECTED'
        """
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}🎬 DECISIÓN Y ACCIONES RECOMENDADAS{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")
        
        if not obligatory_passed:
            status = 'REJECTED'
            print(f"{RED}{BOLD}❌ MODELO RECHAZADO{RESET}")
            print(f"\n{BOLD}Razones:{RESET}")
            
            checks = self.validation_report['obligatory_checks']
            if not checks['macro_f1']['passed']:
                print(f"  • Macro F1-Score insuficiente: {checks['macro_f1']['value']:.4f} < {checks['macro_f1']['threshold']:.2f}")
            
            if not checks['accuracy']['passed']:
                print(f"  • Accuracy insuficiente: {checks['accuracy']['value']:.4f} < {checks['accuracy']['threshold']:.2f}")
            
            if not checks['critical_recall']['passed']:
                print(f"  • Recall de clases críticas insuficiente (< {checks['critical_recall']['threshold']:.0%})")
                for cls, data in self.validation_report['critical_classes'].items():
                    if not data['passed']:
                        print(f"    - {cls}: {data['recall']:.4f}")
            
            print(f"\n{BOLD}📋 Acciones requeridas:{RESET}")
            actions = [
                "1. 🔍 Investigar causas del bajo rendimiento:",
                "   - Analizar confusion matrix para detectar confusiones frecuentes",
                "   - Revisar distribución de clases en dataset",
                "   - Verificar calidad de imágenes (ruido, iluminación)",
                "2. 🔧 Ajustar hiperparámetros:",
                "   - Incrementar learning rate (ej: 2e-4 o 5e-4)",
                "   - Modificar batch size (32 o 128)",
                "   - Aumentar épocas de entrenamiento",
                "3. 🎯 Mejorar estrategias:",
                "   - Aplicar augmentation más agresivo",
                "   - Ajustar class weights para clases problemáticas",
                "   - Considerar focal loss para desbalanceo",
                "4. ♻️  Reentrenar modelo con ajustes",
                "5. 📊 Re-evaluar y validar nuevamente"
            ]
            
        elif optional_results['ideal_met']:
            status = 'APPROVED'
            print(f"{GREEN}{BOLD}✅ MODELO APROBADO - RENDIMIENTO IDEAL{RESET}")
            print(f"\n{GREEN}🌟 El modelo ha alcanzado objetivos ideales de rendimiento{RESET}")
            
            print(f"\n{BOLD}📋 Acciones recomendadas:{RESET}")
            actions = [
                "1. 📝 Documentar resultados y configuración:",
                "   - Guardar métricas finales en documentación",
                "   - Registrar hiperparámetros utilizados",
                "   - Documentar arquitectura y decisiones",
                "2. 💾 Preservar modelo:",
                "   - Hacer backup de best_model.keras",
                "   - Versionar en Git (Git LFS si es necesario)",
                "   - Guardar training_history.json",
                "3. 🚀 Preparar para producción:",
                "   - Probar predicciones en casos reales",
                "   - Validar tiempos de inferencia",
                "   - Preparar API/interfaz de usuario",
                "4. 📊 Monitoreo continuo:",
                "   - Establecer métricas de seguimiento",
                "   - Definir umbral de re-entrenamiento"
            ]
            
        elif optional_results['targets_met']:
            status = 'APPROVED'
            print(f"{GREEN}{BOLD}✅ MODELO APROBADO - OBJETIVOS CUMPLIDOS{RESET}")
            print(f"\n{BLUE}🎯 El modelo cumple objetivos establecidos{RESET}")
            
            print(f"\n{BOLD}📋 Acciones recomendadas:{RESET}")
            actions = [
                "1. 📝 Documentar resultados actuales",
                "2. 💾 Guardar modelo y configuración",
                "3. 🔬 (Opcional) Optimización adicional:",
                "   - Fine-tuning para alcanzar rendimiento ideal",
                "   - Ajustes menores en hiperparámetros",
                "   - Pruebas con arquitecturas alternativas",
                "4. 🚀 Preparar para producción"
            ]
            
        else:
            status = 'CONDITIONAL'
            print(f"{YELLOW}{BOLD}⚠️  MODELO CONDICIONAL{RESET}")
            print(f"\n{YELLOW}Cumple requisitos obligatorios pero no alcanza objetivos ideales{RESET}")
            
            print(f"\n{BOLD}📋 Acciones recomendadas:{RESET}")
            actions = [
                "1. 🔧 Ajustar hiperparámetros (refinamiento):",
                "   - Modificar learning rate (ej: 5e-5 o 2e-4)",
                "   - Ajustar patience de early stopping",
                "   - Probar diferentes batch sizes",
                "2. 📊 Analizar métricas detalladas:",
                "   - Identificar clases con bajo rendimiento",
                "   - Revisar confusion matrix",
                "   - Analizar errores frecuentes",
                "3. 🎯 Optimización dirigida:",
                "   - Ajustar class weights",
                "   - Mejorar augmentation",
                "   - Aumentar datos para clases débiles",
                "4. ♻️  Re-entrenar con ajustes menores",
                "5. 📝 Documentar si se acepta rendimiento actual"
            ]
        
        for action in actions:
            print(f"  {action}")
        
        self.validation_report['overall_status'] = status
        self.validation_report['recommended_actions'] = actions
        
        return status
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Genera resumen de validación.
        
        Returns:
            Diccionario con resumen completo
        """
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}📋 RESUMEN DE VALIDACIÓN{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")
        
        # Status badge
        status = self.validation_report['overall_status']
        if status == 'APPROVED':
            badge = f"{GREEN}✅ APROBADO{RESET}"
        elif status == 'CONDITIONAL':
            badge = f"{YELLOW}⚠️  CONDICIONAL{RESET}"
        else:
            badge = f"{RED}❌ RECHAZADO{RESET}"
        
        print(f"{BOLD}Estado del Modelo:{RESET} {badge}")
        print(f"{BOLD}Timestamp:{RESET} {self.validation_report['timestamp']}")
        
        # Requisitos obligatorios
        print(f"\n{BOLD}Requisitos Obligatorios:{RESET}")
        oblig = self.validation_report['obligatory_checks']
        
        print(f"  • Macro F1-Score: {oblig['macro_f1']['value']:.4f} / {oblig['macro_f1']['threshold']:.2f} "
              f"{'✅' if oblig['macro_f1']['passed'] else '❌'}")
        
        print(f"  • Accuracy:       {oblig['accuracy']['value']:.4f} / {oblig['accuracy']['threshold']:.2f} "
              f"{'✅' if oblig['accuracy']['passed'] else '❌'}")
        
        print(f"  • Recall Críticas: {'✅ PASS' if oblig['critical_recall']['passed'] else '❌ FAIL'}")
        
        # Objetivos opcionales
        print(f"\n{BOLD}Objetivos Opcionales:{RESET}")
        opt = self.validation_report['optional_checks']
        
        if 'macro_f1' in opt:
            mf1 = opt['macro_f1']
            icon = "🌟" if mf1['ideal_met'] else "🎯" if mf1['target_met'] else "⚠️"
            print(f"  {icon} Macro F1-Score: {mf1['value']:.4f} "
                  f"(Target: {mf1['target']:.2f}, Ideal: {mf1['ideal']:.2f})")
        
        if 'accuracy' in opt:
            acc = opt['accuracy']
            icon = "🌟" if acc['ideal_met'] else "🎯" if acc['target_met'] else "⚠️"
            print(f"  {icon} Accuracy:       {acc['value']:.4f} "
                  f"(Target: {acc['target']:.2f}, Ideal: {acc['ideal']:.2f})")
        
        self.validation_report['passed'] = (status == 'APPROVED')
        
        return self.validation_report
    
    def save_report(self, output_path: str = None):
        """
        Guarda el reporte de validación en JSON.
        
        Args:
            output_path: Ruta donde guardar el reporte
        """
        if output_path is None:
            output_path = backend_dir.parent / 'metrics' / 'validation_report.json'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.validation_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n{GREEN}✅ Reporte guardado en: {output_path}{RESET}")
    
    def run_validation(self) -> bool:
        """
        Ejecuta la validación completa.
        
        Returns:
            True si el modelo es aprobado
        """
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}🔍 VALIDACIÓN AUTOMÁTICA DE REQUISITOS - PASO 4{RESET}")
        print(f"{BOLD}{'='*70}{RESET}")
        
        # Cargar resultados
        if not self.load_results():
            return False
        
        # Validar requisitos obligatorios
        obligatory_passed = self.validate_obligatory_requirements()
        
        # Validar objetivos opcionales
        optional_results = self.validate_optional_requirements()
        
        # Determinar acciones
        status = self.determine_actions(obligatory_passed, optional_results)
        
        # Generar resumen
        self.generate_summary()
        
        # Guardar reporte
        self.save_report()
        
        return status == 'APPROVED'


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validación automática de requisitos del modelo'
    )
    parser.add_argument(
        '--results',
        type=str,
        default=None,
        help='Ruta al archivo JSON con resultados de evaluación'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Ruta donde guardar el reporte de validación'
    )
    
    args = parser.parse_args()
    
    # Ejecutar validación
    validator = RequirementsValidator(args.results)
    approved = validator.run_validation()
    
    # Exit code
    sys.exit(0 if approved else 1)


if __name__ == '__main__':
    main()
