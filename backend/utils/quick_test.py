#!/usr/bin/env python
"""
Script de prueba rápida para verificar que todo funciona correctamente.
Ejecuta una serie de tests básicos sin necesidad de un dataset completo.
"""

import sys
from pathlib import Path


def print_header(title):
    """Imprime un encabezado formateado."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_imports():
    """Prueba que todas las librerías se puedan importar."""
    print_header("🔍 TEST 1: Verificando Imports")
    
    libraries = [
        ('tensorflow', 'TensorFlow'),
        ('keras', 'Keras'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('sklearn', 'scikit-learn'),
        ('flask', 'Flask'),
    ]
    
    failed = []
    for module, name in libraries:
        try:
            __import__(module)
            print(f"  ✅ {name:20s} OK")
        except ImportError as e:
            print(f"  ❌ {name:20s} FALLO")
            failed.append((name, str(e)))
    
    if failed:
        print("\n⚠️  Algunas librerías no se pudieron importar:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        return False
    
    print("\n✅ Todos los imports exitosos")
    return True


def test_tensorflow():
    """Prueba funcionalidad básica de TensorFlow."""
    print_header("🧠 TEST 2: Verificando TensorFlow")
    
    try:
        import tensorflow as tf
        import numpy as np
        
        print(f"  TensorFlow versión: {tf.__version__}")
        
        # Verificar GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  🚀 GPU disponible: {len(gpus)} dispositivo(s)")
        else:
            print(f"  💻 Usando CPU")
        
        # Crear un modelo simple
        print("\n  Creando modelo de prueba...")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # Hacer una predicción de prueba
        test_input = np.random.random((1, 5))
        prediction = model.predict(test_input, verbose=0)
        
        print(f"  ✅ Modelo creado y predicción realizada")
        print(f"  📊 Shape de salida: {prediction.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def test_image_processing():
    """Prueba procesamiento de imágenes."""
    print_header("🖼️  TEST 3: Verificando Procesamiento de Imágenes")
    
    try:
        import numpy as np
        from PIL import Image
        import cv2
        
        # Crear imagen de prueba
        print("  Creando imagen de prueba...")
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Redimensionar
        img_resized = img.resize((50, 50))
        print(f"  ✅ Redimensionamiento: {img.size} → {img_resized.size}")
        
        # Convertir con OpenCV
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        print(f"  ✅ Conversión OpenCV: {img_cv.shape}")
        
        # Normalizar
        img_normalized = img_array.astype(np.float32) / 255.0
        print(f"  ✅ Normalización: rango [{img_normalized.min():.2f}, {img_normalized.max():.2f}]")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def test_flask():
    """Prueba que Flask funcione."""
    print_header("🌐 TEST 4: Verificando Flask")
    
    try:
        from flask import Flask, jsonify
        
        # Crear app de prueba
        app = Flask(__name__)
        
        @app.route('/test')
        def test_route():
            return jsonify({'status': 'ok'})
        
        print("  ✅ Flask app creada correctamente")
        print("  ✅ Ruta de prueba configurada")
        
        # Verificar que se puede crear un contexto
        with app.app_context():
            print("  ✅ Contexto de aplicación funcional")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def test_file_structure():
    """Verifica la estructura de archivos."""
    print_header("📁 TEST 5: Verificando Estructura de Archivos")
    
    required_files = [
        'data_preparation.py',
        'train_model.py',
        'app.py',
        'predict.py',
        'config.py',
        'requirements.txt',
        'README.md',
        'templates/index.html'
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"  ✅ {file_path:30s} ({size:,} bytes)")
        else:
            print(f"  ❌ {file_path:30s} NO ENCONTRADO")
            all_exist = False
    
    return all_exist


def test_config():
    """Prueba el archivo de configuración."""
    print_header("⚙️  TEST 6: Verificando Configuración")
    
    try:
        import backend.config as config
        
        print(f"  ✅ Clases: {config.CLASSES}")
        print(f"  ✅ Tamaño de imagen: {config.IMG_SIZE}")
        print(f"  ✅ Batch size: {config.BATCH_SIZE}")
        print(f"  ✅ Épocas: {config.EPOCHS}")
        print(f"  ✅ Learning rate: {config.LEARNING_RATE}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def test_model_creation():
    """Prueba crear el modelo CNN."""
    print_header("🏗️  TEST 7: Verificando Creación de Modelo CNN")
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, models
        
        print("  Creando arquitectura CNN...")
        
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(5, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"  ✅ Modelo creado con {model.count_params():,} parámetros")
        print(f"  ✅ Input shape: (100, 100, 3)")
        print(f"  ✅ Output shape: (5,)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "🧪 SUITE DE PRUEBAS RÁPIDAS 🧪" + " " * 22 + "║")
    print("╚" + "═" * 68 + "╝")
    
    tests = [
        ("Imports", test_imports),
        ("TensorFlow", test_tensorflow),
        ("Procesamiento de Imágenes", test_image_processing),
        ("Flask", test_flask),
        ("Estructura de Archivos", test_file_structure),
        ("Configuración", test_config),
        ("Creación de Modelo", test_model_creation),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ Error inesperado en {name}: {str(e)}")
            results[name] = False
    
    # Resumen
    print_header("📊 RESUMEN DE RESULTADOS")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"  {name:30s} {status}")
    
    print("\n" + "=" * 70)
    print(f"  Total: {passed}/{total} tests pasados ({passed/total*100:.1f}%)")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 ¡TODOS LOS TESTS PASARON!")
        print("\n✅ El sistema está listo para usar:")
        print("   1. Configura tu dataset: python download_sample_dataset.py")
        print("   2. Limpia los datos: python data_preparation.py")
        print("   3. Entrena el modelo: python train_model.py")
        print("   4. Inicia la app: python app.py")
    else:
        print("\n⚠️  ALGUNOS TESTS FALLARON")
        print("\nAcciones recomendadas:")
        print("   1. Verifica la instalación: python verify_installation.py")
        print("   2. Reinstala dependencias: pip install -r requirements.txt")
        print("   3. Revisa los errores arriba")
    
    print("\n")
    
    return passed == total


def main():
    """Función principal."""
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrumpidos por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error fatal: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
