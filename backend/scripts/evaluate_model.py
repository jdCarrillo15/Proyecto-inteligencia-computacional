#!/usr/bin/env python3
"""
Script de Evaluación del Modelo - Paso 3
========================================

Evalúa el modelo entrenado en el test set y genera reportes completos.

Funcionalidades:
✅ Cargar best_model.keras
✅ Hacer predicciones en test set
✅ Calcular todas las métricas:
   - Accuracy general
   - Macro F1-Score
   - Weighted F1-Score
   - Recall por clase (CRÍTICO)
   - Precision por clase
✅ Generar confusion matrix
✅ Generar reporte Excel: evaluation_results.xlsx
✅ Integración con sistema de métricas detalladas existente

Uso:
    python backend/scripts/evaluate_model.py
    python backend/scripts/evaluate_model.py --model models/best_model.keras
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import argparse

# Agregar el directorio backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    top_k_accuracy_score
)

from utils.data_cache import DataCache
from scripts.detailed_metrics import DetailedMetrics
from config import IMG_SIZE, CLASSES, NUM_CLASSES

# Configurar TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(42)
np.random.seed(42)


class ModelEvaluator:
    """Evaluador completo del modelo con generación de reportes."""
    
    def __init__(self, model_path='models/best_model.keras'):
        """
        Inicializa el evaluador.
        
        Args:
            model_path: Ruta al modelo a evaluar
        """
        self.model_path = Path(model_path)
        self.model = None
        self.metrics_system = DetailedMetrics()
        self.results = {}
        
        print(f"\n🔧 Evaluador inicializado:")
        print(f"  - Modelo: {self.model_path}")
        
    def load_model(self):
        """Carga el modelo desde disco."""
        print(f"\n📂 Cargando modelo desde {self.model_path}...")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"❌ Modelo no encontrado: {self.model_path}")
        
        self.model = keras.models.load_model(self.model_path)
        print(f"✅ Modelo cargado exitosamente")
        print(f"  - Parámetros: {self.model.count_params():,}")
        
    def load_test_data(self):
        """Carga el test set desde cache."""
        print("\n📦 Cargando test set desde cache...")
        
        cache = DataCache()
        test_data = cache.load('test')
        
        if not test_data:
            raise ValueError("❌ Test data no encontrado en cache. Ejecuta prepare_dataset.py primero.")
        
        X_test = test_data['X']
        y_test = test_data['y']
        class_names = test_data['class_names']
        
        print(f"✅ Test set cargado:")
        print(f"  - Muestras: {len(X_test):,}")
        print(f"  - Shape: {X_test.shape}")
        print(f"  - Clases: {len(class_names)}")
        
        return X_test, y_test, class_names
    
    def evaluate(self, X_test, y_test, class_names):
        """
        Evalúa el modelo en el test set (Paso 3).
        
        Args:
            X_test: Datos de prueba
            y_test: Labels de prueba (one-hot)
            class_names: Nombres de las clases
            
        Returns:
            dict: Resultados completos de evaluación
        """
        print("\n" + "=" * 80)
        print("📊 PASO 3: EVALUACIÓN EN TEST SET")
        print("=" * 80)
        
        # 1. Predicciones
        print("\n⏳ Generando predicciones...")
        predictions = self.model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        print(f"✅ Predicciones generadas: {len(predictions):,}")
        
        # 2. Métricas básicas
        print("\n📈 Calculando métricas básicas...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # 3. Precision, Recall, F1-Score por clase
        precision, recall, f1, support = precision_recall_fscore_support(
            true_classes, predicted_classes, average=None, zero_division=0
        )
        
        # 4. Métricas agregadas
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_classes, predicted_classes, average='macro', zero_division=0
        )
        
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            true_classes, predicted_classes, average='weighted', zero_division=0
        )
        
        # 5. Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # 6. Top-K Accuracy
        top3_acc = top_k_accuracy_score(true_classes, predictions, k=3)
        top5_acc = top_k_accuracy_score(true_classes, predictions, k=5)
        
        # Almacenar resultados
        self.results = {
            'global_metrics': {
                'test_loss': float(test_loss),
                'test_accuracy': float(test_accuracy),
                'macro_precision': float(macro_precision),
                'macro_recall': float(macro_recall),
                'macro_f1': float(macro_f1),
                'weighted_precision': float(weighted_precision),
                'weighted_recall': float(weighted_recall),
                'weighted_f1': float(weighted_f1),
                'top3_accuracy': float(top3_acc),
                'top5_accuracy': float(top5_acc)
            },
            'per_class_metrics': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1': f1.tolist(),
                'support': support.tolist()
            },
            'confusion_matrix': cm.tolist(),
            'class_names': class_names,
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path)
        }
        
        # Imprimir resultados en consola
        self._print_results(class_names, precision, recall, f1, support)
        
        return self.results
    
    def _print_results(self, class_names, precision, recall, f1, support):
        """Imprime resultados en consola con formato estructurado."""
        print("\n" + "=" * 80)
        print("✅ RESULTADOS DE EVALUACIÓN (PASO 3)")
        print("=" * 80)
        
        # 1. Métricas Globales
        print("\n1️⃣  MÉTRICAS GLOBALES")
        print("-" * 80)
        gm = self.results['global_metrics']
        print(f"  • Test Loss:          {gm['test_loss']:.4f}")
        print(f"  • Test Accuracy:      {gm['test_accuracy']:.4f} ({gm['test_accuracy']*100:.2f}%)")
        print(f"  • Macro F1-Score:     {gm['macro_f1']:.4f} ⭐ (CRÍTICO)")
        print(f"  • Weighted F1-Score:  {gm['weighted_f1']:.4f}")
        print(f"  • Macro Precision:    {gm['macro_precision']:.4f}")
        print(f"  • Macro Recall:       {gm['macro_recall']:.4f}")
        print(f"  • Top-3 Accuracy:     {gm['top3_accuracy']:.4f} ({gm['top3_accuracy']*100:.2f}%)")
        print(f"  • Top-5 Accuracy:     {gm['top5_accuracy']:.4f} ({gm['top5_accuracy']*100:.2f}%)")
        
        # 2. Métricas por Clase (ordenadas por F1-Score)
        print("\n2️⃣  MÉTRICAS POR CLASE (Ordenadas por F1-Score)")
        print("-" * 80)
        print(f"{'Clase':<45} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>8}")
        print("-" * 80)
        
        # Ordenar por F1-Score descendente
        sorted_indices = np.argsort(f1)[::-1]
        
        for idx in sorted_indices:
            class_name = class_names[idx]
            p = precision[idx]
            r = recall[idx]
            f = f1[idx]
            s = support[idx]
            
            # Indicador visual de performance
            if f >= 0.90:
                indicator = "🟢"
            elif f >= 0.75:
                indicator = "🟡"
            else:
                indicator = "🔴"
            
            print(f"{class_name:<45} {p:>10.4f} {r:>10.4f} {f:>10.4f} {s:>8} {indicator}")
        
        # 3. Recall por Clase (CRÍTICO según Paso 3)
        print("\n3️⃣  RECALL POR CLASE (CRÍTICO) - Top 5 Mejores y Peores")
        print("-" * 80)
        
        # Top 5 mejores
        print("🏆 Top 5 Mejores Recall:")
        best_recall_indices = np.argsort(recall)[::-1][:5]
        for rank, idx in enumerate(best_recall_indices, 1):
            print(f"  {rank}. {class_names[idx]:<45} {recall[idx]:.4f} ({recall[idx]*100:.2f}%)")
        
        # Top 5 peores
        print("\n⚠️  Top 5 Peores Recall:")
        worst_recall_indices = np.argsort(recall)[:5]
        for rank, idx in enumerate(worst_recall_indices, 1):
            print(f"  {rank}. {class_names[idx]:<45} {recall[idx]:.4f} ({recall[idx]*100:.2f}%)")
        
        # 4. Estadísticas de Confusion Matrix
        cm = np.array(self.results['confusion_matrix'])
        print("\n4️⃣  CONFUSION MATRIX - Estadísticas")
        print("-" * 80)
        
        # Diagonal (predicciones correctas)
        diagonal_sum = np.trace(cm)
        total_sum = np.sum(cm)
        
        print(f"  • Total de muestras:        {total_sum:,}")
        print(f"  • Predicciones correctas:   {diagonal_sum:,} ({diagonal_sum/total_sum*100:.2f}%)")
        print(f"  • Predicciones incorrectas: {total_sum - diagonal_sum:,} ({(total_sum-diagonal_sum)/total_sum*100:.2f}%)")
        
        # Top confusiones
        print("\n  📊 Top 5 Confusiones:")
        off_diagonal = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i][j] > 0:
                    off_diagonal.append((cm[i][j], i, j))
        
        off_diagonal.sort(reverse=True)
        for rank, (count, true_idx, pred_idx) in enumerate(off_diagonal[:5], 1):
            true_class = class_names[true_idx]
            pred_class = class_names[pred_idx]
            print(f"    {rank}. {true_class} → {pred_class}: {count} veces")
    
    def generate_detailed_visualizations(self, X_test, y_test, class_names):
        """Genera visualizaciones detalladas usando DetailedMetrics."""
        print("\n📊 Generando visualizaciones detalladas...")
        
        predictions = self.model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        # Calcular métricas adicionales
        class_metrics = self.metrics_system.calculate_per_class_metrics(
            true_classes, predicted_classes, class_names
        )
        
        crop_metrics = self.metrics_system.calculate_per_crop_metrics(
            true_classes, predicted_classes, class_names
        )
        
        binary_metrics = self.metrics_system.calculate_healthy_vs_diseased(
            true_classes, predicted_classes, class_names
        )
        
        cm = confusion_matrix(true_classes, predicted_classes)
        top_confusions = self.metrics_system.analyze_top_confusions(cm, class_names, top_n=10)
        
        # Generar visualizaciones
        self.metrics_system.plot_confusion_matrix(cm, class_names)
        self.metrics_system.plot_per_class_metrics(class_metrics, class_names)
        self.metrics_system.plot_per_crop_metrics(crop_metrics)
        self.metrics_system.plot_healthy_vs_diseased(binary_metrics)
        
        print("✅ Visualizaciones generadas en models/visualizations/")
    
    def save_results_json(self, filepath='metrics/evaluation_results.json'):
        """Guarda resultados como JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"✅ Resultados guardados: {filepath}")
    
    def generate_excel_report(self, filepath='metrics/evaluation_results.xlsx'):
        """
        Genera reporte Excel completo (Paso 3).
        
        Args:
            filepath: Ruta del archivo Excel de salida
        """
        print("\n📊 Generando reporte Excel...")
        
        try:
            import pandas as pd
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment
        except ImportError:
            print("⚠️  pandas y openpyxl requeridos para Excel. Instalando...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "openpyxl"])
            import pandas as pd
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Crear Excel writer
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            
            # 1. Hoja de Métricas Globales
            gm = self.results['global_metrics']
            df_global = pd.DataFrame([
                ['Test Loss', gm['test_loss']],
                ['Test Accuracy', gm['test_accuracy']],
                ['Macro F1-Score', gm['macro_f1']],
                ['Weighted F1-Score', gm['weighted_f1']],
                ['Macro Precision', gm['macro_precision']],
                ['Macro Recall', gm['macro_recall']],
                ['Weighted Precision', gm['weighted_precision']],
                ['Weighted Recall', gm['weighted_recall']],
                ['Top-3 Accuracy', gm['top3_accuracy']],
                ['Top-5 Accuracy', gm['top5_accuracy']]
            ], columns=['Métrica', 'Valor'])
            
            df_global.to_excel(writer, sheet_name='Métricas Globales', index=False)
            
            # 2. Hoja de Métricas por Clase
            pcm = self.results['per_class_metrics']
            df_class = pd.DataFrame({
                'Clase': self.results['class_names'],
                'Precision': pcm['precision'],
                'Recall': pcm['recall'],
                'F1-Score': pcm['f1'],
                'Support': pcm['support']
            })
            
            # Ordenar por F1-Score descendente
            df_class = df_class.sort_values('F1-Score', ascending=False)
            df_class.to_excel(writer, sheet_name='Métricas por Clase', index=False)
            
            # 3. Hoja de Confusion Matrix
            cm = np.array(self.results['confusion_matrix'])
            df_cm = pd.DataFrame(
                cm,
                index=self.results['class_names'],
                columns=self.results['class_names']
            )
            df_cm.to_excel(writer, sheet_name='Confusion Matrix')
            
            # 4. Hoja de Metadata
            df_meta = pd.DataFrame([
                ['Fecha de Evaluación', self.results['timestamp']],
                ['Modelo', self.results['model_path']],
                ['Total de Clases', len(self.results['class_names'])],
                ['Total de Muestras', sum(pcm['support'])]
            ], columns=['Campo', 'Valor'])
            
            df_meta.to_excel(writer, sheet_name='Metadata', index=False)
        
        print(f"✅ Reporte Excel generado: {filepath}")
        print(f"  - 4 hojas: Métricas Globales, Por Clase, Confusion Matrix, Metadata")


def main():
    """Función principal de evaluación."""
    parser = argparse.ArgumentParser(description='Evaluar modelo en test set (Paso 3)')
    parser.add_argument('--model', type=str, default='models/best_model.keras',
                       help='Ruta al modelo a evaluar (default: models/best_model.keras)')
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🚀 EVALUACIÓN DEL MODELO - PASO 3")
    print("=" * 80)
    
    try:
        # 1. Inicializar evaluador
        evaluator = ModelEvaluator(model_path=args.model)
        
        # 2. Cargar modelo
        evaluator.load_model()
        
        # 3. Cargar test data
        X_test, y_test, class_names = evaluator.load_test_data()
        
        # 4. Evaluar (Paso 3)
        results = evaluator.evaluate(X_test, y_test, class_names)
        
        # 5. Guardar resultados JSON
        evaluator.save_results_json()
        
        # 6. Generar reporte Excel (Paso 3)
        evaluator.generate_excel_report()
        
        # 7. Generar visualizaciones detalladas
        evaluator.generate_detailed_visualizations(X_test, y_test, class_names)
        
        # Resumen final
        print("\n" + "=" * 80)
        print("✅ EVALUACIÓN COMPLETADA")
        print("=" * 80)
        print(f"\n📁 Archivos generados:")
        print("  - metrics/evaluation_results.json (métricas en JSON)")
        print("  - metrics/evaluation_results.xlsx (reporte Excel completo)")
        print("  - models/visualizations/confusion_matrix.png")
        print("  - models/visualizations/class_metrics.png")
        print("  - models/visualizations/crop_performance.png")
        print("  - models/visualizations/healthy_vs_diseased.png")
        
        gm = results['global_metrics']
        print(f"\n🎯 Resumen de Performance:")
        print(f"  - Accuracy: {gm['test_accuracy']*100:.2f}%")
        print(f"  - Macro F1: {gm['macro_f1']:.4f}")
        print(f"  - Weighted F1: {gm['weighted_f1']:.4f}")
        
        print("\n💡 Próximos pasos:")
        print("  1. Revisar metrics/evaluation_results.xlsx")
        print("  2. Analizar visualizaciones en models/visualizations/")
        print("  3. Identificar clases con bajo recall y mejorar dataset")
        
    except Exception as e:
        print(f"\n❌ Error durante evaluación: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
