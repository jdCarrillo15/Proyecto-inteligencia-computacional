"""
Script de auditoría del dataset para diagnóstico completo.
Genera un análisis detallado del estado actual del proyecto.
"""

import os
import sys
from pathlib import Path
import pickle
import json
import pandas as pd

# Agregar el directorio backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from config import IMG_SIZE, CLASSES


def count_images_by_class(train_path):
    """Cuenta imágenes por cada clase en el dataset."""
    class_counts = {}
    total_images = 0
    
    for class_name in sorted([d.name for d in train_path.iterdir() if d.is_dir()]):
        class_path = train_path / class_name
        image_files = list(class_path.glob("*.*"))
        count = len(image_files)
        class_counts[class_name] = count
        total_images += count
    
    return class_counts, total_images


def analyze_class_distribution(class_counts, total_images):
    """Analiza el desbalanceo de clases."""
    if not class_counts:
        return None
    
    max_class = max(class_counts.items(), key=lambda x: x[1])
    min_class = min(class_counts.items(), key=lambda x: x[1])
    
    imbalance_ratio = max_class[1] / min_class[1] if min_class[1] > 0 else 0
    
    return {
        'max_class': max_class,
        'min_class': min_class,
        'imbalance_ratio': imbalance_ratio,
        'avg_images': total_images / len(class_counts) if class_counts else 0
    }


def check_cache_status(cache_dir):
    """Verifica el estado del cache PKL."""
    cache_files = list(cache_dir.glob("*.pkl"))
    
    if not cache_files:
        return {"status": "No cache", "files": [], "details": {}}
    
    cache_info = {"status": "Cache exists", "files": [], "details": {}}
    
    for cache_file in cache_files:
        cache_info["files"].append(cache_file.name)
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    cache_info["details"][cache_file.name] = {
                        "keys": list(data.keys()),
                        "img_size": data.get("img_size", "Unknown"),
                        "num_classes": data.get("num_classes", "Unknown"),
                        "size_mb": cache_file.stat().st_size / (1024 * 1024)
                    }
        except Exception as e:
            cache_info["details"][cache_file.name] = {"error": str(e)}
    
    return cache_info


def check_model_metrics(models_dir):
    """Verifica métricas del último entrenamiento."""
    model_files = list(models_dir.glob("*.keras"))
    metrics_info = {"models": [], "latest_metrics": None}
    
    for model_file in model_files:
        metrics_info["models"].append({
            "name": model_file.name,
            "size_mb": model_file.stat().st_size / (1024 * 1024),
            "modified": model_file.stat().st_mtime
        })
    
    # Buscar archivo de métricas si existe
    metrics_file = models_dir / "training_history.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                metrics_info["latest_metrics"] = json.load(f)
        except:
            pass
    
    return metrics_info


def create_excel_report(class_counts, total_images, analysis, output_path):
    """Crea reporte Excel con distribución de clases."""
    data = []
    
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_images * 100) if total_images > 0 else 0
        severity = "CRÍTICO" if count < 200 else ("BAJO" if count < 1000 else "MAYORÍA")
        
        data.append({
            "Clase": class_name,
            "Número de Imágenes": count,
            "Porcentaje (%)": round(percentage, 2),
            "Estado": severity
        })
    
    df = pd.DataFrame(data)
    
    # Agregar fila de totales
    df.loc[len(df)] = ["TOTAL", total_images, 100.00, ""]
    
    # Agregar análisis de desbalanceo
    if analysis:
        df.loc[len(df)] = ["", "", "", ""]
        df.loc[len(df)] = ["Clase con más imágenes", analysis['max_class'][0], analysis['max_class'][1], ""]
        df.loc[len(df)] = ["Clase con menos imágenes", analysis['min_class'][0], analysis['min_class'][1], ""]
        df.loc[len(df)] = ["Ratio de desbalanceo", "", round(analysis['imbalance_ratio'], 2), ""]
        df.loc[len(df)] = ["Promedio por clase", "", round(analysis['avg_images'], 2), ""]
    
    df.to_excel(output_path, index=False, sheet_name="Distribución de Clases")
    print(f"\n✅ Reporte Excel guardado en: {output_path}")


def main():
    """Función principal de auditoría."""
    print("=" * 80)
    print("AUDITORÍA DEL ESTADO ACTUAL DEL PROYECTO")
    print("=" * 80)
    
    # Configuración de rutas
    base_dir = backend_dir.parent
    train_path = base_dir / "dataset" / "raw" / "New Plant Diseases Dataset(Augmented)" / "train"
    cache_dir = backend_dir / "cache"
    models_dir = base_dir / "models"
    
    # 1. IMG_SIZE actual
    print(f"\n📐 IMG_SIZE actual en config.py: {IMG_SIZE}")
    
    # 2. Contar imágenes por clase
    print("\n📊 Contando imágenes por clase...")
    if not train_path.exists():
        print(f"❌ ERROR: No se encontró el directorio de entrenamiento: {train_path}")
        return
    
    class_counts, total_images = count_images_by_class(train_path)
    
    print(f"\n📈 DISTRIBUCIÓN DE CLASES:")
    print("-" * 80)
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"{class_name:45s}: {count:5d} imágenes ({percentage:5.2f}%)")
    print("-" * 80)
    print(f"{'TOTAL':45s}: {total_images:5d} imágenes")
    
    # 3. Análisis de desbalanceo
    analysis = analyze_class_distribution(class_counts, total_images)
    if analysis:
        print(f"\n⚠️  ANÁLISIS DE DESBALANCEO:")
        print(f"   Clase con MÁS imágenes: {analysis['max_class'][0]} ({analysis['max_class'][1]} imgs)")
        print(f"   Clase con MENOS imágenes: {analysis['min_class'][0]} ({analysis['min_class'][1]} imgs)")
        print(f"   Ratio de desbalanceo: {analysis['imbalance_ratio']:.2f}:1")
        print(f"   Promedio por clase: {analysis['avg_images']:.0f} imágenes")
    
    # 4. Verificar cache PKL
    print(f"\n💾 ESTADO DEL CACHE:")
    cache_info = check_cache_status(cache_dir)
    print(f"   Estado: {cache_info['status']}")
    if cache_info['files']:
        for filename, details in cache_info['details'].items():
            print(f"   - {filename}:")
            if 'error' in details:
                print(f"     ❌ Error: {details['error']}")
            else:
                print(f"     IMG_SIZE: {details.get('img_size', 'Unknown')}")
                print(f"     Clases: {details.get('num_classes', 'Unknown')}")
                print(f"     Tamaño: {details.get('size_mb', 0):.2f} MB")
    
    # 5. Verificar modelos y métricas
    print(f"\n🤖 MODELOS Y MÉTRICAS:")
    if models_dir.exists():
        metrics_info = check_model_metrics(models_dir)
        if metrics_info['models']:
            for model in metrics_info['models']:
                print(f"   - {model['name']}: {model['size_mb']:.2f} MB")
        else:
            print("   ⚠️  No se encontraron modelos entrenados")
        
        if metrics_info['latest_metrics']:
            print(f"\n   Última métrica de entrenamiento:")
            latest = metrics_info['latest_metrics']
            if isinstance(latest, dict):
                for key, value in list(latest.items())[:5]:  # Mostrar primeras 5 métricas
                    print(f"     {key}: {value}")
    else:
        print("   ⚠️  Directorio de modelos no existe")
    
    # 6. Crear reporte Excel
    print(f"\n📊 Generando reporte Excel...")
    output_excel = base_dir / "dataset_audit_report.xlsx"
    create_excel_report(class_counts, total_images, analysis, output_excel)
    
    print("\n" + "=" * 80)
    print("✅ AUDITORÍA COMPLETADA")
    print("=" * 80)
    print(f"\nResumen:")
    print(f"  - Total de imágenes: {total_images:,}")
    print(f"  - Total de clases: {len(class_counts)}")
    print(f"  - IMG_SIZE configurado: {IMG_SIZE}")
    print(f"  - Cache: {cache_info['status']}")
    print(f"  - Desbalanceo: {analysis['imbalance_ratio']:.2f}:1" if analysis else "")
    print(f"\n📄 Reporte completo guardado en: {output_excel}")


if __name__ == "__main__":
    main()
