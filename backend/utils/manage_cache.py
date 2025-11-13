"""
Utilidad para gestionar el cache PKL del sistema.
Permite limpiar, ver información y optimizar el cache.
"""

import sys
from pathlib import Path

# Agregar backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from utils.data_cache import DataCache


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
            confirm = input("\n⚠️  ¿Seguro que quieres limpiar el cache? (s/n): ").strip().lower()
            if confirm == 's':
                cache.clear()
                print("✅ Cache limpiado exitosamente")
            else:
                print("❌ Operación cancelada")
        
        elif choice == "3":
            # Verificar integridad
            print("\n🔍 Verificando integridad del cache...")
            info = cache.get_info()
            
            if info['total_files'] == 0:
                print("⚠️  No hay archivos en el cache")
            else:
                print(f"✅ Cache OK: {info['total_files']} archivos, {info['total_size_mb']:.2f} MB")
                
                # Verificar metadatos
                if cache.metadata:
                    print(f"✅ Metadatos OK: {len(cache.metadata)} datasets")
                else:
                    print("⚠️  No hay metadatos")
        
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
