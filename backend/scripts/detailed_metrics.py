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
                              top3_acc, top5_acc, top_confusions, class_names, test_loss, test_accuracy):
        """Imprime métricas detalladas en consola con formato estructurado."""
        
        print("\n" + "=" * 80)
        print("📊 REPORTE DE MÉTRICAS - DESPUÉS DEL ENTRENAMIENTO")
        print("=" * 80)
        
        # ========== 1. MÉTRICAS GLOBALES ==========
        print("\n" + "▼" * 80)
        print("MÉTRICAS GLOBALES")
        print("▼" * 80)
        print(f"  • Test Loss:           {test_loss:.4f}")
        print(f"  • Test Accuracy:       {test_accuracy:.2%}")
        print(f"  • Macro F1-Score:      {metrics['macro_avg']['f1']:.2%}  ← MÉTRICA PRINCIPAL")
        print(f"  • Weighted F1-Score:   {metrics['weighted_avg']['f1']:.2%}")
        print(f"  • Top-3 Accuracy:      {top3_acc:.2%}")
        
        # ========== 2. POR CLASE (15 clases) ==========
        print("\n" + "▼" * 80)
        print("POR CLASE (15 clases) - Ordenadas por F1-Score")
        print("▼" * 80)
        
        # Ordenar por F1-Score descendente
        sorted_indices = np.argsort(metrics['per_class']['f1'])[::-1]
        
        print(f"\n{'#':<3} {'Clase':<40} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>9}")
        print("-" * 85)
        
        for rank, i in enumerate(sorted_indices, 1):
            cls = class_names[i]
            prec = metrics['per_class']['precision'][i]
            rec = metrics['per_class']['recall'][i]
            f1 = metrics['per_class']['f1'][i]
            sup = int(metrics['per_class']['support'][i])
            
            # Indicador visual de rendimiento
            indicator = "✓" if f1 >= 0.75 else "!" if f1 >= 0.60 else "⚠"
            
            print(f"{rank:<3} {indicator} {cls:<38} {prec:>9.2%} {rec:>9.2%} {f1:>9.2%} {sup:>9}")
        
        print("-" * 85)
        print(f"{'':>4} {'Macro Average':<38} "
              f"{metrics['macro_avg']['precision']:>9.2%} "
              f"{metrics['macro_avg']['recall']:>9.2%} "
              f"{metrics['macro_avg']['f1']:>9.2%}")
        print(f"{'':>4} {'Weighted Average':<38} "
              f"{metrics['weighted_avg']['precision']:>9.2%} "
              f"{metrics['weighted_avg']['recall']:>9.2%} "
              f"{metrics['weighted_avg']['f1']:>9.2%}")
        
        # ========== 3. POR CULTIVO (4 cultivos) ==========
        print("\n" + "▼" * 80)
        print("POR CULTIVO (4 cultivos)")
        print("▼" * 80)
        
        crop_emojis = {"Apple": "🍎", "Corn": "🌽", "Potato": "🥔", "Tomato": "🍅"}
        
        for crop, crop_metrics in sorted(per_crop.items()):
            emoji = crop_emojis.get(crop, "🌱")
            acc = crop_metrics['accuracy']
            samples = crop_metrics['samples']
            
            # Identificar problemas
            problem_marker = " ← PROBLEMA" if acc < 0.70 else ""
            
            print(f"  {emoji} {crop:<10} {acc:.2%} accuracy ({samples:,} muestras){problem_marker}")
        
        # ========== 4. ANÁLISIS BINARIO (Healthy vs Diseased) ==========
        print("\n" + "▼" * 80)
        print("ANÁLISIS BINARIO (Healthy vs Diseased)")
        print("▼" * 80)
        
        print(f"  • Accuracy:          {binary_metrics['accuracy']:.2%}")
        print(f"  • True Negatives:    {binary_metrics['true_negatives']:,} (sano → sano)")
        print(f"  • False Positives:   {binary_metrics['false_positives']:,} (sano → enfermo)")
        print(f"  • False Negatives:   {binary_metrics['false_negatives']:,} (enfermo → sano)  ← CRÍTICO")
        print(f"  • True Positives:    {binary_metrics['true_positives']:,} (enfermo → enfermo)")
        
        # Indicadores adicionales
        total_diseased = binary_metrics['false_negatives'] + binary_metrics['true_positives']
        fn_rate = binary_metrics['false_negatives'] / total_diseased if total_diseased > 0 else 0
        
        if fn_rate > 0.10:
            print(f"\n  ⚠️  ALERTA: Tasa de falsos negativos = {fn_rate:.2%} (>{10:.0%})")
            print(f"      → Riesgo: Enfermedades sin detectar pueden propagarse")
        
        # ========== 5. TOP 10 CONFUSIONES ==========
        print("\n" + "▼" * 80)
        print("TOP 10 CONFUSIONES")
        print("▼" * 80)
        
        for i, (true_cls, pred_cls, count) in enumerate(top_confusions, 1):
            print(f"  {i:2}. {true_cls:<38} → {pred_cls:<38} ({count:>3} veces)")
        
        print("\n" + "=" * 80)
    
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
        """Visualización 2: Métricas por Clase (Barplot detallado)."""
        fig, ax = plt.subplots(figsize=(14, 11))
        
        # Ordenar por F1-Score
        sorted_indices = np.argsort(metrics['per_class']['f1'])
        sorted_classes = [class_names[i] for i in sorted_indices]
        sorted_f1 = metrics['per_class']['f1'][sorted_indices]
        sorted_precision = metrics['per_class']['precision'][sorted_indices]
        sorted_recall = metrics['per_class']['recall'][sorted_indices]
        
        y_pos = np.arange(len(sorted_classes))
        bar_height = 0.25
        
        # Crear barras horizontales
        bars_f1 = ax.barh(y_pos + bar_height, sorted_f1, bar_height, 
                          label='F1-Score', color='steelblue', alpha=0.9)
        bars_prec = ax.barh(y_pos, sorted_precision, bar_height, 
                            label='Precision', color='orange', alpha=0.8)
        bars_rec = ax.barh(y_pos - bar_height, sorted_recall, bar_height, 
                           label='Recall', color='green', alpha=0.8)
        
        # Añadir valores en las barras (solo F1-Score)
        for i, (bar, value) in enumerate(zip(bars_f1, sorted_f1)):
            if value > 0.05:  # Solo si hay espacio
                ax.text(value - 0.03, bar.get_y() + bar.get_height()/2, 
                       f'{value:.2%}', 
                       ha='right', va='center', fontsize=8, 
                       fontweight='bold', color='white')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_classes, fontsize=9)
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title('Métricas por Clase (Ordenado por F1-Score)\n' +
                    f'Macro Avg F1: {metrics["macro_avg"]["f1"]:.2%}', 
                    fontsize=14, pad=20, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        
        # Líneas de referencia con etiquetas
        ax.axvline(x=0.6, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.text(0.6, len(sorted_classes) - 0.5, ' 60% (mínimo)', 
               fontsize=9, color='red', va='top')
        
        ax.axvline(x=0.75, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.text(0.75, len(sorted_classes) - 0.5, ' 75% (target)', 
               fontsize=9, color='orange', va='top')
        
        ax.set_xlim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_per_crop_performance(self, per_crop, test_accuracy):
        """Visualización 3: Per-Crop Performance con contexto detallado."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        crops_list = sorted(per_crop.keys())
        accuracies = [per_crop[crop]['accuracy'] for crop in crops_list]
        samples = [per_crop[crop]['samples'] for crop in crops_list]
        
        # Colores según rendimiento
        colors = ['#27ae60' if acc > 0.8 else '#f39c12' if acc > 0.7 else '#e74c3c' for acc in accuracies]
        
        bars = ax.bar(crops_list, accuracies, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
        
        # Añadir líneas de referencia
        ax.axhline(y=test_accuracy, color='#3498db', linestyle='--', linewidth=2, 
                  label=f'Accuracy Global: {test_accuracy:.2%}', alpha=0.7)
        ax.axhline(y=0.70, color='red', linestyle=':', linewidth=1.5, 
                  label='Umbral Mínimo: 70%', alpha=0.5)
        ax.axhline(y=0.80, color='green', linestyle=':', linewidth=1.5, 
                  label='Objetivo Ideal: 80%', alpha=0.5)
        
        # Añadir valores en las barras con samples
        for bar, acc, samp, crop in zip(bars, accuracies, samples, crops_list):
            height = bar.get_height()
            
            # Porcentaje
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{acc:.1%}',
                   ha='center', va='bottom', fontsize=13, fontweight='bold')
            
            # Número de muestras
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'{samp:,}\nmuestras',
                   ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        
        # Emojis en etiquetas del eje X
        emoji_map = {"Apple": "🍎", "Corn": "🌽", "Potato": "🥔", "Tomato": "🍅"}
        x_labels = [f"{emoji_map.get(crop, '🌱')}\n{crop}" for crop in crops_list]
        ax.set_xticks(range(len(crops_list)))
        ax.set_xticklabels(x_labels, fontsize=11)
        
        ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
        ax.set_title('Rendimiento por Cultivo (4 cultivos)\nAnálisis de distribución y desempeño', 
                    fontsize=15, pad=20, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'per_crop_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_healthy_vs_diseased(self, binary_metrics):
        """Visualización 4: Healthy vs Diseased con análisis detallado."""
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
        
        # ===== SUBPLOT 1: Matriz de Confusión =====
        ax1 = fig.add_subplot(gs[0])
        
        cm_bin = binary_metrics['confusion_matrix']
        
        # Normalizar para mostrar porcentajes
        cm_normalized = cm_bin.astype('float') / cm_bin.sum(axis=1)[:, np.newaxis]
        
        # Heatmap con anotaciones duales (porcentaje + cantidad)
        sns.heatmap(cm_normalized, annot=False, cmap='RdYlGn_r',
                   xticklabels=['Healthy', 'Diseased'],
                   yticklabels=['Healthy', 'Diseased'],
                   cbar_kws={'label': 'Proporción'},
                   vmin=0, vmax=1, ax=ax1, square=True)
        
        # Añadir anotaciones personalizadas
        for i in range(2):
            for j in range(2):
                count = cm_bin[i, j]
                pct = cm_normalized[i, j] * 100
                text_color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
                ax1.text(j + 0.5, i + 0.5, f'{pct:.1f}%\n({count:,})',
                        ha='center', va='center', fontsize=12, 
                        fontweight='bold', color=text_color)
        
        ax1.set_title(f'Matriz de Confusión Binaria\nBinary Accuracy: {binary_metrics["accuracy"]:.2%}', 
                     fontsize=13, pad=15, fontweight='bold')
        ax1.set_ylabel('Clase Real', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Clase Predicha', fontsize=11, fontweight='bold')
        
        # ===== SUBPLOT 2: Desglose de Métricas =====
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        
        tn = binary_metrics['true_negatives']
        fp = binary_metrics['false_positives']
        fn = binary_metrics['false_negatives']
        tp = binary_metrics['true_positives']
        
        total = tn + fp + fn + tp
        
        # Calcular métricas adicionales
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall para diseased
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall para healthy
        precision_diseased = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Texto del desglose
        metrics_text = f"""
╔══════════════════════════════════════╗
║   DESGLOSE DE PREDICCIONES           ║
╚══════════════════════════════════════╝

✅ CORRECTAS ({tn + tp:,} / {total:,})
   • True Negatives:  {tn:>6,}  (sano → sano)
   • True Positives:  {tp:>6,}  (enfermo → enfermo)

❌ ERRORES ({fp + fn:,} / {total:,})
   • False Positives: {fp:>6,}  (sano → enfermo)
   • False Negatives: {fn:>6,}  (enfermo → sano) ⚠️

─────────────────────────────────────

📊 MÉTRICAS CLAVE

Sensitivity (Recall Diseased):  {sensitivity:.2%}
   → De los enfermos, detecta {sensitivity:.1%}

Specificity (Recall Healthy):   {specificity:.2%}
   → De los sanos, identifica {specificity:.1%}

Precision (Diseased):           {precision_diseased:.2%}
   → Al predecir "enfermo", acierta {precision_diseased:.1%}

─────────────────────────────────────

⚠️  ALERTA CRÍTICA:
    False Negatives = {fn:,}
    → Plantas enfermas clasificadas como sanas
    → Riesgo de propagación no detectada
"""
        
        ax2.text(0.05, 0.95, metrics_text, 
                transform=ax2.transAxes,
                fontsize=9.5, 
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
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
            
            # ========== MÉTRICAS GLOBALES ==========
            f.write("▼" * 80 + "\n")
            f.write("MÉTRICAS GLOBALES\n")
            f.write("▼" * 80 + "\n")
            f.write(f"  • Test Loss:           {test_loss:.4f}\n")
            f.write(f"  • Test Accuracy:       {test_accuracy:.2%}\n")
            f.write(f"  • Macro F1-Score:      {metrics['macro_avg']['f1']:.2%}  ← MÉTRICA PRINCIPAL\n")
            f.write(f"  • Weighted F1-Score:   {metrics['weighted_avg']['f1']:.2%}\n")
            f.write(f"  • Top-3 Accuracy:      {top3_acc:.2%}\n\n")
            
            # ========== POR CLASE (15 clases) ==========
            f.write("▼" * 80 + "\n")
            f.write("POR CLASE (15 clases) - Ordenadas por F1-Score\n")
            f.write("▼" * 80 + "\n\n")
            
            # Ordenar por F1-Score descendente
            sorted_indices = np.argsort(metrics['per_class']['f1'])[::-1]
            
            f.write(f"{'#':<3} {'Clase':<40} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>9}\n")
            f.write("-" * 85 + "\n")
            
            for rank, i in enumerate(sorted_indices, 1):
                cls = class_names[i]
                prec = metrics['per_class']['precision'][i]
                rec = metrics['per_class']['recall'][i]
                f1 = metrics['per_class']['f1'][i]
                sup = int(metrics['per_class']['support'][i])
                
                indicator = "✓" if f1 >= 0.75 else "!" if f1 >= 0.60 else "⚠"
                
                f.write(f"{rank:<3} {indicator} {cls:<38} {prec:>9.2%} {rec:>9.2%} {f1:>9.2%} {sup:>9}\n")
            
            f.write("-" * 85 + "\n")
            f.write(f"{'':>4} {'Macro Average':<38} "
                   f"{metrics['macro_avg']['precision']:>9.2%} "
                   f"{metrics['macro_avg']['recall']:>9.2%} "
                   f"{metrics['macro_avg']['f1']:>9.2%}\n")
            f.write(f"{'':>4} {'Weighted Average':<38} "
                   f"{metrics['weighted_avg']['precision']:>9.2%} "
                   f"{metrics['weighted_avg']['recall']:>9.2%} "
                   f"{metrics['weighted_avg']['f1']:>9.2%}\n\n")
            
            # ========== POR CULTIVO (4 cultivos) ==========
            f.write("▼" * 80 + "\n")
            f.write("POR CULTIVO (4 cultivos)\n")
            f.write("▼" * 80 + "\n")
            
            crop_emojis = {"Apple": "🍎", "Corn": "🌽", "Potato": "🥔", "Tomato": "🍅"}
            
            for crop, crop_metrics in sorted(per_crop.items()):
                emoji = crop_emojis.get(crop, "🌱")
                acc = crop_metrics['accuracy']
                samples = crop_metrics['samples']
                problem_marker = " ← PROBLEMA" if acc < 0.70 else ""
                f.write(f"  {emoji} {crop:<10} {acc:.2%} accuracy ({samples:,} muestras){problem_marker}\n")
            f.write("\n")
            
            # ========== ANÁLISIS BINARIO (Healthy vs Diseased) ==========
            f.write("▼" * 80 + "\n")
            f.write("ANÁLISIS BINARIO (Healthy vs Diseased)\n")
            f.write("▼" * 80 + "\n")
            f.write(f"  • Accuracy:          {binary_metrics['accuracy']:.2%}\n")
            f.write(f"  • True Negatives:    {binary_metrics['true_negatives']:,} (sano → sano)\n")
            f.write(f"  • False Positives:   {binary_metrics['false_positives']:,} (sano → enfermo)\n")
            f.write(f"  • False Negatives:   {binary_metrics['false_negatives']:,} (enfermo → sano)  ← CRÍTICO\n")
            f.write(f"  • True Positives:    {binary_metrics['true_positives']:,} (enfermo → enfermo)\n\n")
            
            # Métricas adicionales
            total_diseased = binary_metrics['false_negatives'] + binary_metrics['true_positives']
            fn_rate = binary_metrics['false_negatives'] / total_diseased if total_diseased > 0 else 0
            
            tp = binary_metrics['true_positives']
            fn = binary_metrics['false_negatives']
            fp = binary_metrics['false_positives']
            tn = binary_metrics['true_negatives']
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            f.write(f"  📊 Sensitivity (Recall Diseased): {sensitivity:.2%}\n")
            f.write(f"  📊 Specificity (Recall Healthy):  {specificity:.2%}\n\n")
            
            if fn_rate > 0.10:
                f.write(f"  ⚠️  ALERTA: Tasa de falsos negativos = {fn_rate:.2%} (>10%)\n")
                f.write(f"      → Riesgo: Enfermedades sin detectar pueden propagarse\n\n")
            
            # ========== TOP 10 CONFUSIONES ==========
            f.write("▼" * 80 + "\n")
            f.write("TOP 10 CONFUSIONES\n")
            f.write("▼" * 80 + "\n")
            for i, (true_cls, pred_cls, count) in enumerate(top_confusions, 1):
                f.write(f"  {i:2}. {true_cls:<38} → {pred_cls:<38} ({count:>3} veces)\n")
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
