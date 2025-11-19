"""
Sistema de métricas detalladas para evaluación del modelo.
Incluye análisis por clase, por cultivo, healthy vs diseased, y visualizaciones avanzadas.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score,
    top_k_accuracy_score, confusion_matrix
)


class DetailedMetrics:
    """Calcula y visualiza métricas detalladas del modelo."""
    
    def __init__(self):
        self.viz_dir = Path('models/visualizations')
        self.viz_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_per_class_metrics(self, y_true, y_pred, class_names):
        """
        Calcula métricas detalladas por clase.
        
        Returns:
            dict: Métricas por clase y agregadas
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Métricas agregadas
        macro_avg = {
            'precision': precision.mean(),
            'recall': recall.mean(),
            'f1': f1.mean()
        }
        
        weighted_avg = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        return {
            'per_class': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            },
            'macro_avg': macro_avg,
            'weighted_avg': {
                'precision': weighted_avg[0],
                'recall': weighted_avg[1],
                'f1': weighted_avg[2]
            }
        }
    
    def calculate_per_crop_metrics(self, y_true, y_pred, class_names):
        """
        Calcula accuracy por cultivo (Apple, Corn, Potato, Tomato).
        
        Returns:
            dict: Accuracy por cultivo
        """
        crops = {
            'Apple': ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy'],
            'Corn': ['Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight'],
            'Potato': ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight'],
            'Tomato': ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold']
        }
        
        per_crop = {}
        for crop, crop_classes in crops.items():
            # Filtrar índices de este cultivo
            crop_indices = [i for i, cls in enumerate(class_names) if cls in crop_classes]
            
            if not crop_indices:
                continue
            
            # Filtrar predicciones de este cultivo
            mask = np.isin(y_true, crop_indices)
            if mask.sum() == 0:
                continue
            
            crop_y_true = y_true[mask]
            crop_y_pred = y_pred[mask]
            
            # Calcular accuracy
            crop_accuracy = accuracy_score(crop_y_true, crop_y_pred)
            per_crop[crop] = {
                'accuracy': crop_accuracy,
                'samples': int(mask.sum())
            }
        
        return per_crop
    
    def calculate_healthy_vs_diseased(self, y_true, y_pred, class_names):
        """
        Calcula métricas binarias: Healthy vs Diseased.
        
        Returns:
            dict: Métricas binarias y matriz de confusión 2x2
        """
        # Identificar clases healthy
        healthy_classes = [i for i, cls in enumerate(class_names) if 'healthy' in cls.lower()]
        
        # Convertir a binario: 0=healthy, 1=diseased
        y_true_binary = np.array([0 if y in healthy_classes else 1 for y in y_true])
        y_pred_binary = np.array([0 if y in healthy_classes else 1 for y in y_pred])
        
        # Matriz de confusión 2x2
        cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
        
        # Calcular métricas
        tn, fp, fn, tp = cm_binary.ravel()
        
        return {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'confusion_matrix': cm_binary,
            'true_negatives': int(tn),   # Healthy → Healthy (correcto)
            'false_positives': int(fp),  # Healthy → Diseased (falso positivo)
            'false_negatives': int(fn),  # Diseased → Healthy (falso negativo - CRÍTICO)
            'true_positives': int(tp)    # Diseased → Diseased (correcto)
        }
    
    def analyze_top_confusions(self, cm, class_names, top_n=10):
        """
        Identifica las top N confusiones más frecuentes.
        
        Returns:
            list: Lista de tuplas (clase_real, clase_pred, frecuencia)
        """
        confusions = []
        
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    confusions.append((class_names[i], class_names[j], int(cm[i, j])))
        
        # Ordenar por frecuencia
        confusions.sort(key=lambda x: x[2], reverse=True)
        
        return confusions[:top_n]
    
    def print_detailed_metrics(self, metrics, per_crop, binary_metrics, 
                              top3_acc, top5_acc, top_confusions, class_names):
        """Imprime métricas detalladas en consola."""
        
        # 1. Métricas por clase
        print("\n" + "-" * 80)
        print("📊 MÉTRICAS POR CLASE")
        print("-" * 80)
        
        print(f"\n{'Clase':<35} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-" * 80)
        
        for i, cls in enumerate(class_names):
            print(f"{cls:<35} "
                  f"{metrics['per_class']['precision'][i]:>10.4f} "
                  f"{metrics['per_class']['recall'][i]:>10.4f} "
                  f"{metrics['per_class']['f1'][i]:>10.4f} "
                  f"{int(metrics['per_class']['support'][i]):>10}")
        
        print("-" * 80)
        print(f"{'Macro Average':<35} "
              f"{metrics['macro_avg']['precision']:>10.4f} "
              f"{metrics['macro_avg']['recall']:>10.4f} "
              f"{metrics['macro_avg']['f1']:>10.4f}")
        print(f"{'Weighted Average':<35} "
              f"{metrics['weighted_avg']['precision']:>10.4f} "
              f"{metrics['weighted_avg']['recall']:>10.4f} "
              f"{metrics['weighted_avg']['f1']:>10.4f}")
        
        # 2. Métricas por cultivo
        print("\n" + "-" * 80)
        print("🌱 MÉTRICAS POR CULTIVO")
        print("-" * 80)
        
        for crop, crop_metrics in per_crop.items():
            emoji = "🍎" if crop == "Apple" else "🌽" if crop == "Corn" else "🥔" if crop == "Potato" else "🍅"
            print(f"{emoji} {crop:<12} Accuracy: {crop_metrics['accuracy']:.4f} ({crop_metrics['samples']} muestras)")
        
        # 3. Healthy vs Diseased
        print("\n" + "-" * 80)
        print("🌿 HEALTHY VS DISEASED (Análisis Binario)")
        print("-" * 80)
        
        print(f"\nBinary Accuracy: {binary_metrics['accuracy']:.4f}")
        print(f"\nMatriz de Confusión 2x2:")
        print(f"                   Pred: Healthy  Pred: Diseased")
        print(f"True: Healthy      {binary_metrics['true_negatives']:>10}  {binary_metrics['false_positives']:>15}")
        print(f"True: Diseased     {binary_metrics['false_negatives']:>10}  {binary_metrics['true_positives']:>15}")
        print(f"\n⚠️  False Negatives (enfermo → sano): {binary_metrics['false_negatives']} - CRÍTICO")
        print(f"⚠️  False Positives (sano → enfermo): {binary_metrics['false_positives']}")
        
        # 4. Top-K Accuracy
        print("\n" + "-" * 80)
        print("🎯 TOP-K ACCURACY")
        print("-" * 80)
        
        print(f"Top-3 Accuracy: {top3_acc:.4f} (clase correcta en top 3 predicciones)")
        print(f"Top-5 Accuracy: {top5_acc:.4f} (clase correcta en top 5 predicciones)")
        
        # 5. Análisis de confusiones
        print("\n" + "-" * 80)
        print("🔍 TOP 10 CONFUSIONES MÁS FRECUENTES")
        print("-" * 80)
        
        for i, (true_cls, pred_cls, count) in enumerate(top_confusions, 1):
            print(f"{i:2}. {true_cls:<35} → {pred_cls:<35} ({count:>3} veces)")
    
    def plot_confusion_matrix_detailed(self, cm, class_names):
        """Visualización 1: Confusion Matrix Detallada."""
        plt.figure(figsize=(16, 14))
        
        # Normalizar por fila (mostrar porcentajes)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='RdYlGn_r', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Proporción'},
                    vmin=0, vmax=1)
        plt.title('Matriz de Confusión Detallada\n(Color: proporción, Números: cantidad absoluta)', 
                 fontsize=14, pad=20)
        plt.ylabel('Clase Real', fontsize=12)
        plt.xlabel('Clase Predicha', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_per_class_metrics(self, metrics, class_names):
        """Visualización 2: Métricas por Clase (Barplot)."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Ordenar por F1-Score
        sorted_indices = np.argsort(metrics['per_class']['f1'])
        sorted_classes = [class_names[i] for i in sorted_indices]
        sorted_f1 = metrics['per_class']['f1'][sorted_indices]
        sorted_precision = metrics['per_class']['precision'][sorted_indices]
        sorted_recall = metrics['per_class']['recall'][sorted_indices]
        
        y_pos = np.arange(len(sorted_classes))
        
        ax.barh(y_pos, sorted_f1, alpha=0.8, label='F1-Score', color='steelblue')
        ax.barh(y_pos, sorted_precision, alpha=0.6, label='Precision', color='orange')
        ax.barh(y_pos, sorted_recall, alpha=0.6, label='Recall', color='green')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_classes, fontsize=9)
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title('Métricas por Clase (Ordenado por F1-Score)', fontsize=14, pad=20)
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        ax.axvline(x=0.6, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_per_crop_performance(self, per_crop, test_accuracy):
        """Visualización 3: Per-Crop Performance."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        crops_list = list(per_crop.keys())
        accuracies = [per_crop[crop]['accuracy'] for crop in crops_list]
        colors = ['#2ecc71' if acc > 0.8 else '#f39c12' if acc > 0.6 else '#e74c3c' for acc in accuracies]
        
        bars = ax.bar(crops_list, accuracies, color=colors, alpha=0.8, edgecolor='black')
        
        # Añadir línea de accuracy promedio
        ax.axhline(y=test_accuracy, color='blue', linestyle='--', label=f'Accuracy Global: {test_accuracy:.4f}')
        
        # Añadir valores en las barras
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Rendimiento por Cultivo', fontsize=14, pad=20)
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'per_crop_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_healthy_vs_diseased(self, binary_metrics):
        """Visualización 4: Healthy vs Diseased."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm_bin = binary_metrics['confusion_matrix']
        sns.heatmap(cm_bin, annot=True, fmt='d', cmap='RdYlGn_r',
                   xticklabels=['Healthy', 'Diseased'],
                   yticklabels=['Healthy', 'Diseased'],
                   cbar_kws={'label': 'Cantidad'},
                   ax=ax)
        
        ax.set_title(f'Healthy vs Diseased\nBinary Accuracy: {binary_metrics["accuracy"]:.4f}', 
                    fontsize=14, pad=20)
        ax.set_ylabel('Clase Real', fontsize=12)
        ax.set_xlabel('Clase Predicha', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'healthy_vs_diseased.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_detailed_report(self, test_loss, test_accuracy, metrics, per_crop, 
                                binary_metrics, top3_acc, top5_acc, top_confusions, 
                                class_names, training_config=None):
        """Genera reporte detallado en archivo de texto."""
        
        report_path = self.viz_dir / 'training_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE DE ENTRENAMIENTO - CLASIFICADOR DE ENFERMEDADES DE PLANTAS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuración
            if training_config:
                f.write("-" * 80 + "\n")
                f.write("CONFIGURACIÓN DEL ENTRENAMIENTO\n")
                f.write("-" * 80 + "\n")
                for key, value in training_config.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Métricas globales
            f.write("-" * 80 + "\n")
            f.write("MÉTRICAS GLOBALES\n")
            f.write("-" * 80 + "\n")
            f.write(f"Test Loss:          {test_loss:.4f}\n")
            f.write(f"Test Accuracy:      {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")
            f.write(f"Macro Avg F1:       {metrics['macro_avg']['f1']:.4f}\n")
            f.write(f"Weighted Avg F1:    {metrics['weighted_avg']['f1']:.4f}\n")
            f.write(f"Top-3 Accuracy:     {top3_acc:.4f}\n")
            f.write(f"Top-5 Accuracy:     {top5_acc:.4f}\n\n")
            
            # Métricas por clase
            f.write("-" * 80 + "\n")
            f.write("MÉTRICAS POR CLASE\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Clase':<35} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n")
            f.write("-" * 80 + "\n")
            
            for i, cls in enumerate(class_names):
                f.write(f"{cls:<35} "
                       f"{metrics['per_class']['precision'][i]:>10.4f} "
                       f"{metrics['per_class']['recall'][i]:>10.4f} "
                       f"{metrics['per_class']['f1'][i]:>10.4f} "
                       f"{int(metrics['per_class']['support'][i]):>10}\n")
            
            f.write("-" * 80 + "\n")
            f.write(f"{'Macro Average':<35} "
                   f"{metrics['macro_avg']['precision']:>10.4f} "
                   f"{metrics['macro_avg']['recall']:>10.4f} "
                   f"{metrics['macro_avg']['f1']:>10.4f}\n")
            f.write(f"{'Weighted Average':<35} "
                   f"{metrics['weighted_avg']['precision']:>10.4f} "
                   f"{metrics['weighted_avg']['recall']:>10.4f} "
                   f"{metrics['weighted_avg']['f1']:>10.4f}\n\n")
            
            # Métricas por cultivo
            f.write("-" * 80 + "\n")
            f.write("ANÁLISIS POR CULTIVO\n")
            f.write("-" * 80 + "\n")
            for crop, crop_metrics in per_crop.items():
                f.write(f"{crop:<15} Accuracy: {crop_metrics['accuracy']:.4f} "
                       f"({crop_metrics['samples']} muestras)\n")
            f.write("\n")
            
            # Healthy vs Diseased
            f.write("-" * 80 + "\n")
            f.write("HEALTHY VS DISEASED (Análisis Binario)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Binary Accuracy: {binary_metrics['accuracy']:.4f}\n\n")
            f.write(f"Matriz de Confusión 2x2:\n")
            f.write(f"                   Pred: Healthy  Pred: Diseased\n")
            f.write(f"True: Healthy      {binary_metrics['true_negatives']:>10}  {binary_metrics['false_positives']:>15}\n")
            f.write(f"True: Diseased     {binary_metrics['false_negatives']:>10}  {binary_metrics['true_positives']:>15}\n\n")
            f.write(f"False Negatives (enfermo → sano): {binary_metrics['false_negatives']} - CRÍTICO\n")
            f.write(f"False Positives (sano → enfermo): {binary_metrics['false_positives']}\n\n")
            
            # Top confusiones
            f.write("-" * 80 + "\n")
            f.write("TOP 10 CONFUSIONES MÁS FRECUENTES\n")
            f.write("-" * 80 + "\n")
            for i, (true_cls, pred_cls, count) in enumerate(top_confusions, 1):
                f.write(f"{i:2}. {true_cls:<35} → {pred_cls:<35} ({count:>3} veces)\n")
            f.write("\n")
            
            # Recomendaciones
            f.write("-" * 80 + "\n")
            f.write("RECOMENDACIONES\n")
            f.write("-" * 80 + "\n")
            
            # Detectar sesgo
            if metrics['macro_avg']['f1'] < metrics['weighted_avg']['f1'] - 0.1:
                f.write("⚠️  Detectado sesgo hacia clases mayoritarias\n")
                f.write("💡 Sugerencia: Aplicar class weights o balancear dataset\n\n")
            
            # Detectar overfitting en healthy vs diseased
            if binary_metrics['false_negatives'] > binary_metrics['false_positives'] * 2:
                f.write("⚠️  Alto número de falsos negativos (enfermo → sano)\n")
                f.write("💡 Sugerencia: Aumentar data augmentation para clases enfermas\n\n")
            
            # Detectar bajo rendimiento por cultivo
            for crop, crop_metrics in per_crop.items():
                if crop_metrics['accuracy'] < test_accuracy - 0.15:
                    f.write(f"⚠️  Bajo rendimiento en cultivo {crop}: {crop_metrics['accuracy']:.4f}\n")
                    f.write(f"💡 Sugerencia: Revisar calidad de datos para {crop}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("FIN DEL REPORTE\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n✅ Reporte guardado en: {report_path}")
