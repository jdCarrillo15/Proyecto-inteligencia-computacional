"""
Script de limpieza de cache y modelos.
Elimina todos los archivos .pkl, .json del cache y .keras de models/

ADVERTENCIA: Este script eliminará TODOS los datos procesados y modelos entrenados.
Úsalo cuando cambies IMG_SIZE o necesites regenerar desde cero.
"""

import os
import shutil
from pathlib import Path
import sys

# Agregar el directorio backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from config import IMG_SIZE


def get_file_size_mb(file_path):
    """Obtiene el tamaño de un archivo en MB."""
    return file_path.stat().st_size / (1024 * 1024)


def clean_cache(cache_dir):
    """Limpia todos los archivos del directorio cache."""
    cache_dir = Path(cache_dir)
    
    if not cache_dir.exists():
        print(f"  ⚠️  Directorio {cache_dir} no existe")
        return 0, 0
    
    files_deleted = 0
    total_size_mb = 0
    
    # Buscar archivos .pkl y .json
    for pattern in ['*.pkl', '*.json']:
        for file in cache_dir.glob(pattern):
            if file.is_file():
                size_mb = get_file_size_mb(file)
                total_size_mb += size_mb
                print(f"    🗑️  Eliminando: {file.name} ({size_mb:.2f} MB)")
                file.unlink()
                files_deleted += 1
    
    return files_deleted, total_size_mb


def clean_models(models_dir):
    """Limpia todos los modelos del directorio models."""
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        print(f"  ⚠️  Directorio {models_dir} no existe")
        return 0, 0
    
    files_deleted = 0
    total_size_mb = 0
    
    # Buscar archivos .keras, .h5, .json
    for pattern in ['*.keras', '*.h5', '*.json']:
        for file in models_dir.glob(pattern):
            if file.is_file():
                size_mb = get_file_size_mb(file)
                total_size_mb += size_mb
                print(f"    🗑️  Eliminando: {file.name} ({size_mb:.2f} MB)")
                file.unlink()
                files_deleted += 1
    
    # Limpiar directorio de visualizaciones si existe
    viz_dir = models_dir / 'visualizations'
    if viz_dir.exists() and viz_dir.is_dir():
        print(f"    🗑️  Eliminando visualizaciones...")
        shutil.rmtree(viz_dir)
        print(f"    ✅ Directorio visualizations eliminado")
    
    return files_deleted, total_size_mb


def main():
    """Función principal de limpieza."""
    print("=" * 80)
    print("LIMPIEZA DE CACHE Y MODELOS")
    print("=" * 80)
    
    print(f"\n📐 Configuración actual:")
    print(f"  - IMG_SIZE: {IMG_SIZE}")
    print(f"  - Resolución: {IMG_SIZE[0]}x{IMG_SIZE[1]} = {IMG_SIZE[0]*IMG_SIZE[1]:,} píxeles")
    
    # Mostrar advertencia
    print("\n⚠️  ADVERTENCIA:")
    print("  Este script eliminará:")
    print("  - Todos los archivos .pkl y .json de backend/cache/")
    print("  - Todos los archivos .keras, .h5, .json de models/")
    print("  - El directorio models/visualizations/")
    print("\n  Esta acción NO se puede deshacer.")
    
    # Confirmar
    response = input("\n¿Deseas continuar? (escribe 'SI' para confirmar): ").strip()
    
    if response != "SI":
        print("\n❌ Operación cancelada")
        return
    
    # Obtener directorios
    base_dir = backend_dir.parent
    cache_dir = backend_dir / 'cache'
    models_dir = base_dir / 'models'
    
    print("\n" + "=" * 80)
    print("INICIANDO LIMPIEZA")
    print("=" * 80)
    
    # Limpiar cache
    print("\n📁 Limpiando directorio cache...")
    cache_files, cache_size = clean_cache(cache_dir)
    
    if cache_files > 0:
        print(f"  ✅ {cache_files} archivos eliminados ({cache_size:.2f} MB)")
    else:
        print(f"  ℹ️  No hay archivos para eliminar")
    
    # Limpiar modelos
    print("\n📁 Limpiando directorio models...")
    models_files, models_size = clean_models(models_dir)
    
    if models_files > 0:
        print(f"  ✅ {models_files} archivos eliminados ({models_size:.2f} MB)")
    else:
        print(f"  ℹ️  No hay archivos para eliminar")
    
    # Resumen
    total_files = cache_files + models_files
    total_size = cache_size + models_size
    
    print("\n" + "=" * 80)
    print("LIMPIEZA COMPLETADA")
    print("=" * 80)
    print(f"\n📊 Resumen:")
    print(f"  - Total de archivos eliminados: {total_files}")
    print(f"  - Espacio liberado: {total_size:.2f} MB")
    
    if total_files > 0:
        print(f"\n✅ Cache y modelos eliminados exitosamente")
        print(f"\n💡 Siguientes pasos:")
        print(f"  1. Ejecuta: python backend/scripts/prepare_dataset.py")
        print(f"  2. Espera 15-25 minutos para generar cache con resolución {IMG_SIZE}")
        print(f"  3. Ejecuta: python backend/scripts/train.py")
        print(f"  4. Espera 1.5-2 horas para entrenar modelo completo")
    else:
        print(f"\nℹ️  No había archivos para limpiar")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
