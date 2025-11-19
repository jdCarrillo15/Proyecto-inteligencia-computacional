"""
Utilidad para gestionar el cache PKL del sistema.
Permite limpiar, ver información y optimizar el cache.
"""

import sys
import json
from pathlib import Path

# Agregar backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from utils.data_cache import DataCache
import config


def main():
    print("\n" + "=" * 60)
    print("🔧 GESTOR DE CACHE PKL")
    print("=" * 60)
    
    cache = DataCache()
    
    while True:
        print("\n📋 OPCIONES:")
        print("  1. Ver información del cache")
        print("  2. Limpiar todo el cache")
        print("  3. Verificar integridad del cache")
        print("  4. Salir")
        
        choice = input("\nSelecciona una opción [1-4]: ").strip()
        
        if choice == "1":
            # Ver información
            cache.print_info()
            
        elif choice == "2":
            # Limpiar cache
            print("\n⚠️  ADVERTENCIA: Esta acción borrará:")
            print("  - Todos los archivos *.pkl (datos cacheados)")
            print("  - Todos los archivos *.json (metadatos)")
            print("  - El cache se regenerará automáticamente al entrenar")
            print("\n🕒 Tiempo de regeneración: 15-25 min (con 224x224)")
            
            confirm = input("\n¿Continuar con la limpieza? (s/n): ").strip().lower()
            if confirm == 's':
                cache.clear()
                print("\n✅ Cache limpiado exitosamente")
                print("\n🎯 Próximos pasos:")
                print("  1. Verifica IMG_SIZE en config.py")
                print("  2. Ejecuta: python backend/scripts/train.py")
                print("  3. El sistema regenerará el cache automáticamente")
            else:
                print("❌ Operación cancelada")
        
        elif choice == "3":
            # Verificar integridad y compatibilidad
            print("\n🔍 Verificando integridad y compatibilidad del cache...")
            info = cache.get_info()
            
            if info['total_files'] == 0:
                print("⚠️  No hay archivos en el cache")
                print("\n💡 Siguiente paso: Ejecuta train.py para generar cache")
            else:
                print(f"\n✅ Archivos encontrados: {info['total_files']} archivos, {info['total_size_mb']:.2f} MB")
                
                # Verificar metadatos
                if cache.metadata:
                    print(f"✅ Metadatos cargados: {len(cache.metadata)} datasets")
                    
                    # Verificar compatibilidad con configuración actual
                    current_img_size = config.IMG_SIZE
                    print(f"\n📊 Configuración actual: IMG_SIZE = {current_img_size}")
                    
                    compatible = True
                    for dataset_key, metadata in cache.metadata.items():
                        cached_img_size = tuple(metadata.get('img_size', [0, 0]))
                        print(f"\n💾 Cache '{dataset_key}':")
                        print(f"  - IMG_SIZE cacheado: {cached_img_size}")
                        print(f"  - Clases: {len(metadata.get('classes', []))}")
                        print(f"  - Muestras train: {metadata.get('num_train', 'N/A')}")
                        print(f"  - Muestras test: {metadata.get('num_test', 'N/A')}")
                        
                        if cached_img_size != current_img_size:
                            print(f"  ❌ INCOMPATIBLE: Cache usa {cached_img_size}, config usa {current_img_size}")
                            compatible = False
                        else:
                            print(f"  ✅ COMPATIBLE")
                    
                    if not compatible:
                        print("\n⚠️  ACCIÓN REQUERIDA:")
                        print("  1. Ejecuta opción 2 para limpiar cache")
                        print("  2. Re-ejecuta train.py para regenerar cache")
                    else:
                        print("\n✅ Cache totalmente compatible con configuración actual")
                else:
                    print("⚠️  No hay metadatos - Cache posiblemente corrupto")
                    print("\n💡 Recomendación: Limpia el cache (opción 2)")
        
        elif choice == "4":
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("❌ Opción inválida")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
