"""
Script de validación de configuración.
Verifica que IMG_SIZE sea consistente en todos los archivos críticos
y detecta posibles conflictos ANTES de entrenar.
"""

import sys
import ast
from pathlib import Path
import re

# Agregar el directorio backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from config import IMG_SIZE, IMG_WIDTH, IMG_HEIGHT


def extract_img_size_from_file(file_path):
    """
    Extrae definiciones de IMG_SIZE o img_size de un archivo Python.
    
    Returns:
        List of tuples: [(line_number, value_string, is_hardcoded)]
    """
    findings = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Buscar definiciones directas de img_size
            # Ej: img_size = (100, 100), IMG_SIZE = (224, 224)
            match = re.search(r'(?:IMG_SIZE|img_size)\s*=\s*\((\d+)\s*,\s*(\d+)\)', line)
            if match:
                width, height = match.groups()
                findings.append((i, f"({width}, {height})", True))
            
            # Buscar default parameters
            # Ej: def __init__(self, img_size=(100, 100))
            match = re.search(r'img_size\s*=\s*\((\d+)\s*,\s*(\d+)\)', line)
            if match and 'def ' in line:
                width, height = match.groups()
                findings.append((i, f"({width}, {height})", True))
        
    except Exception as e:
        print(f"  ⚠️  Error al leer {file_path.name}: {e}")
    
    return findings


def check_file_imports_config(file_path):
    """Verifica si un archivo importa IMG_SIZE desde config.py"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Buscar imports
        if 'from config import' in content and 'IMG_SIZE' in content:
            return True
        if 'import config' in content:
            return True
        
        return False
    except:
        return False


def validate_configuration():
    """Valida la configuración de IMG_SIZE en todo el proyecto."""
    
    print("=" * 80)
    print("VALIDACIÓN DE CONFIGURACIÓN IMG_SIZE")
    print("=" * 80)
    
    print(f"\n📐 Configuración de referencia (config.py):")
    print(f"  - IMG_WIDTH: {IMG_WIDTH}")
    print(f"  - IMG_HEIGHT: {IMG_HEIGHT}")
    print(f"  - IMG_SIZE: {IMG_SIZE}")
    print(f"  - Píxeles totales: {IMG_WIDTH * IMG_HEIGHT:,}")
    
    # Archivos críticos a verificar
    base_dir = backend_dir.parent
    critical_files = [
        backend_dir / 'app.py',
        backend_dir / 'scripts' / 'train.py',
        backend_dir / 'scripts' / 'prepare_dataset.py',
        backend_dir / 'scripts' / 'predict.py',
        backend_dir / 'utils' / 'data_cache.py',
        backend_dir / 'utils' / 'aggressive_augmenter.py',
    ]
    
    print("\n" + "=" * 80)
    print("VERIFICANDO ARCHIVOS CRÍTICOS")
    print("=" * 80)
    
    issues_found = []
    warnings_found = []
    
    for file_path in critical_files:
        if not file_path.exists():
            print(f"\n⚠️  {file_path.name}: NO EXISTE")
            warnings_found.append(f"{file_path.name}: Archivo no encontrado")
            continue
        
        print(f"\n📄 {file_path.name}")
        
        # Verificar si importa desde config
        imports_config = check_file_imports_config(file_path)
        
        if imports_config:
            print(f"  ✅ Importa IMG_SIZE desde config.py")
        else:
            print(f"  ⚠️  No importa IMG_SIZE desde config.py")
            warnings_found.append(f"{file_path.name}: No importa desde config.py")
        
        # Buscar definiciones hardcoded
        findings = extract_img_size_from_file(file_path)
        
        if findings:
            for line_num, value, is_hardcoded in findings:
                if is_hardcoded:
                    if value != str(IMG_SIZE):
                        print(f"  ❌ Línea {line_num}: img_size = {value} (CONFLICTO con config.py)")
                        issues_found.append(f"{file_path.name}:{line_num} - Valor hardcoded {value} != {IMG_SIZE}")
                    else:
                        print(f"  ℹ️  Línea {line_num}: img_size = {value} (coincide con config.py)")
        else:
            if not imports_config:
                print(f"  ⚠️  No se encontraron definiciones de img_size")
    
    # Verificar cache existente
    print("\n" + "=" * 80)
    print("VERIFICANDO CACHE EXISTENTE")
    print("=" * 80)
    
    cache_dir = backend_dir / 'cache'
    cache_files = list(cache_dir.glob('*.pkl')) if cache_dir.exists() else []
    
    if cache_files:
        print(f"\n⚠️  Se encontraron {len(cache_files)} archivos de cache:")
        for cache_file in cache_files:
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            print(f"  - {cache_file.name} ({size_mb:.2f} MB)")
        
        print(f"\n⚠️  ADVERTENCIA: Si el cache usa una resolución diferente a {IMG_SIZE},")
        print(f"  debes eliminarlo antes de entrenar:")
        print(f"  python backend/scripts/clean_cache_and_models.py")
        
        warnings_found.append(f"Cache existente - verificar resolución")
    else:
        print(f"\n✅ No hay cache existente (se generará con resolución {IMG_SIZE})")
    
    # Verificar modelos existentes
    print("\n" + "=" * 80)
    print("VERIFICANDO MODELOS EXISTENTES")
    print("=" * 80)
    
    models_dir = base_dir / 'models'
    model_files = []
    if models_dir.exists():
        model_files = list(models_dir.glob('*.keras')) + list(models_dir.glob('*.h5'))
    
    if model_files:
        print(f"\n⚠️  Se encontraron {len(model_files)} modelos entrenados:")
        for model_file in model_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  - {model_file.name} ({size_mb:.2f} MB)")
        
        print(f"\n⚠️  ADVERTENCIA: Si los modelos fueron entrenados con resolución diferente")
        print(f"  a {IMG_SIZE}, debes eliminarlos:")
        print(f"  python backend/scripts/clean_cache_and_models.py")
        
        warnings_found.append(f"Modelos existentes - verificar resolución")
    else:
        print(f"\n✅ No hay modelos entrenados (se entrenará con resolución {IMG_SIZE})")
    
    # Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN DE VALIDACIÓN")
    print("=" * 80)
    
    if issues_found:
        print(f"\n❌ {len(issues_found)} PROBLEMAS CRÍTICOS ENCONTRADOS:")
        for issue in issues_found:
            print(f"  - {issue}")
        print(f"\n⚠️  ACCIÓN REQUERIDA: Corrige estos problemas antes de entrenar")
        return False
    
    if warnings_found:
        print(f"\n⚠️  {len(warnings_found)} ADVERTENCIAS:")
        for warning in warnings_found:
            print(f"  - {warning}")
        print(f"\n💡 Revisa estas advertencias pero puedes continuar")
    else:
        print(f"\n✅ No se encontraron problemas")
    
    print(f"\n📊 Configuración validada:")
    print(f"  - IMG_SIZE consistente: {IMG_SIZE}")
    print(f"  - Resolución: {IMG_SIZE[0]}x{IMG_SIZE[1]} = {IMG_SIZE[0]*IMG_SIZE[1]:,} píxeles")
    print(f"  - Archivos verificados: {len(critical_files)}")
    
    if not issues_found:
        print(f"\n✅ VALIDACIÓN EXITOSA")
        print(f"\n💡 Siguientes pasos:")
        if cache_files or model_files:
            print(f"  1. (Opcional) Limpia cache/modelos antiguos:")
            print(f"     python backend/scripts/clean_cache_and_models.py")
            print(f"  2. Prepara datos: python backend/scripts/prepare_dataset.py")
            print(f"  3. Entrena modelo: python backend/scripts/train.py")
        else:
            print(f"  1. Prepara datos: python backend/scripts/prepare_dataset.py")
            print(f"  2. Entrena modelo: python backend/scripts/train.py")
        
        return True
    
    return False


if __name__ == "__main__":
    success = validate_configuration()
    sys.exit(0 if success else 1)
