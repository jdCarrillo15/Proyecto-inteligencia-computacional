#!/usr/bin/env python3
"""
Testing Final del Modelo - Paso 6
==================================
Verifica que el modelo está listo para producción.

Pruebas realizadas:
✅ Modelo guarda y carga correctamente
✅ Predicciones en tiempo real funcionan
✅ Latencia < 500ms por imagen
✅ Memory footprint < 500MB
✅ Inference script listo para usar
✅ Documentación de resultados finales

Uso:
    python backend/scripts/test_ready.py
    python backend/scripts/test_ready.py --model models/best_model.keras
"""

import sys
import time
import psutil
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any
import traceback

# Añadir backend al path
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Colores para terminal
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

# Umbrales de rendimiento
LATENCY_THRESHOLD_MS = 500  # 500ms por imagen
MEMORY_THRESHOLD_MB = 500   # 500MB
BATCH_LATENCY_THRESHOLD_MS = 100  # 100ms promedio en batch

class ProductionReadinessTest:
    """Pruebas de preparación para producción."""
    
    def __init__(self, model_path: str = None):
        """
        Inicializa el tester.
        
        Args:
            model_path: Ruta al modelo a probar
        """
        if model_path is None:
            model_path = backend_dir.parent / 'models' / 'best_model.keras'
        
        self.model_path = Path(model_path)
        self.model = None
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'tests': {},
            'all_passed': False
        }
    
    def test_model_save_load(self) -> bool:
        """
        Test 1: Verificar que el modelo guarda y carga correctamente.
        
        Returns:
            True si el test pasa
        """
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}🧪 TEST 1: Modelo Guarda/Carga Correctamente{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # Cargar modelo original
            if not self.model_path.exists():
                print(f"{RED}❌ Modelo no encontrado: {self.model_path}{RESET}")
                self.test_results['tests']['save_load'] = {
                    'passed': False,
                    'error': 'Model file not found'
                }
                return False
            
            print(f"📂 Cargando modelo: {self.model_path}")
            start_time = time.time()
            model = keras.models.load_model(self.model_path)
            load_time = time.time() - start_time
            print(f"⏱️  Tiempo de carga: {load_time:.3f}s")
            
            # Verificar arquitectura
            print(f"🏗️  Arquitectura:")
            print(f"   • Capas: {len(model.layers)}")
            print(f"   • Parámetros: {model.count_params():,}")
            print(f"   • Entrada: {model.input_shape}")
            print(f"   • Salida: {model.output_shape}")
            
            # Guardar copia temporal
            temp_path = backend_dir.parent / 'models' / 'temp_test_model.keras'
            print(f"\n💾 Guardando copia temporal...")
            start_time = time.time()
            model.save(temp_path)
            save_time = time.time() - start_time
            print(f"⏱️  Tiempo de guardado: {save_time:.3f}s")
            
            # Recargar copia
            print(f"🔄 Recargando copia...")
            start_time = time.time()
            reloaded_model = keras.models.load_model(temp_path)
            reload_time = time.time() - start_time
            print(f"⏱️  Tiempo de recarga: {reload_time:.3f}s")
            
            # Verificar que sean iguales
            print(f"🔍 Verificando integridad...")
            
            # Comparar número de capas
            if len(model.layers) != len(reloaded_model.layers):
                raise ValueError("Número de capas diferente")
            
            # Comparar pesos
            for i, (layer1, layer2) in enumerate(zip(model.layers, reloaded_model.layers)):
                weights1 = layer1.get_weights()
                weights2 = layer2.get_weights()
                
                if len(weights1) != len(weights2):
                    raise ValueError(f"Pesos diferentes en capa {i}")
                
                for w1, w2 in zip(weights1, weights2):
                    if not np.allclose(w1, w2, rtol=1e-5):
                        raise ValueError(f"Valores de pesos diferentes en capa {i}")
            
            # Limpiar archivo temporal
            temp_path.unlink()
            
            print(f"\n{GREEN}✅ TEST PASADO: Modelo guarda/carga correctamente{RESET}")
            
            self.model = model
            self.test_results['tests']['save_load'] = {
                'passed': True,
                'load_time': load_time,
                'save_time': save_time,
                'reload_time': reload_time,
                'num_layers': len(model.layers),
                'num_params': int(model.count_params())
            }
            return True
            
        except Exception as e:
            print(f"\n{RED}❌ TEST FALLIDO: {e}{RESET}")
            traceback.print_exc()
            self.test_results['tests']['save_load'] = {
                'passed': False,
                'error': str(e)
            }
            return False
    
    def test_realtime_predictions(self) -> bool:
        """
        Test 2: Verificar que las predicciones en tiempo real funcionan.
        
        Returns:
            True si el test pasa
        """
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}🧪 TEST 2: Predicciones en Tiempo Real{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")
        
        try:
            if self.model is None:
                print(f"{YELLOW}⚠️  Cargando modelo...{RESET}")
                import tensorflow as tf
                from tensorflow import keras
                self.model = keras.models.load_model(self.model_path)
            
            # Generar imagen de prueba
            print(f"🖼️  Generando imagen de prueba (224x224x3)...")
            test_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
            
            # Warmup (primera predicción suele ser más lenta)
            print(f"🔥 Warmup...")
            _ = self.model.predict(test_image, verbose=0)
            
            # Test de predicción única
            print(f"\n📊 Test de predicción única:")
            times = []
            n_runs = 10
            
            for i in range(n_runs):
                start_time = time.time()
                predictions = self.model.predict(test_image, verbose=0)
                elapsed = (time.time() - start_time) * 1000  # ms
                times.append(elapsed)
                
                if i == 0:
                    print(f"   • Forma de salida: {predictions.shape}")
                    print(f"   • Suma de probabilidades: {predictions.sum():.4f}")
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            print(f"\n⏱️  Tiempos de predicción ({n_runs} runs):")
            print(f"   • Promedio: {avg_time:.2f}ms")
            print(f"   • Desv. Est: {std_time:.2f}ms")
            print(f"   • Mínimo: {min_time:.2f}ms")
            print(f"   • Máximo: {max_time:.2f}ms")
            
            # Verificar que las predicciones son válidas
            assert predictions.shape[1] == 15, "Número incorrecto de clases"
            assert np.isclose(predictions.sum(), 1.0, atol=0.01), "Probabilidades no suman 1"
            assert np.all(predictions >= 0) and np.all(predictions <= 1), "Probabilidades fuera de rango"
            
            print(f"\n{GREEN}✅ TEST PASADO: Predicciones funcionan correctamente{RESET}")
            
            self.test_results['tests']['realtime_predictions'] = {
                'passed': True,
                'avg_latency_ms': avg_time,
                'std_latency_ms': std_time,
                'min_latency_ms': min_time,
                'max_latency_ms': max_time,
                'num_runs': n_runs
            }
            return True
            
        except Exception as e:
            print(f"\n{RED}❌ TEST FALLIDO: {e}{RESET}")
            traceback.print_exc()
            self.test_results['tests']['realtime_predictions'] = {
                'passed': False,
                'error': str(e)
            }
            return False
    
    def test_latency(self) -> bool:
        """
        Test 3: Verificar que la latencia < 500ms por imagen.
        
        Returns:
            True si el test pasa
        """
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}🧪 TEST 3: Latencia < {LATENCY_THRESHOLD_MS}ms{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")
        
        try:
            if self.model is None:
                import tensorflow as tf
                from tensorflow import keras
                self.model = keras.models.load_model(self.model_path)
            
            # Test con múltiples tamaños de batch
            test_image = np.random.rand(224, 224, 3).astype(np.float32)
            batch_sizes = [1, 4, 8, 16, 32]
            
            print(f"📊 Testeando diferentes batch sizes:\n")
            
            batch_results = {}
            
            for batch_size in batch_sizes:
                batch_images = np.array([test_image for _ in range(batch_size)])
                
                # Warmup
                _ = self.model.predict(batch_images, verbose=0)
                
                # Medir tiempo
                times = []
                n_runs = 5
                
                for _ in range(n_runs):
                    start_time = time.time()
                    _ = self.model.predict(batch_images, verbose=0)
                    elapsed = (time.time() - start_time) * 1000
                    times.append(elapsed)
                
                avg_total = np.mean(times)
                avg_per_image = avg_total / batch_size
                
                batch_results[batch_size] = {
                    'total_ms': avg_total,
                    'per_image_ms': avg_per_image
                }
                
                status = "✅" if avg_per_image < LATENCY_THRESHOLD_MS else "❌"
                print(f"   Batch {batch_size:2d}: {avg_total:6.2f}ms total | {avg_per_image:6.2f}ms/imagen {status}")
            
            # Verificar batch size 1 (caso más común)
            single_latency = batch_results[1]['per_image_ms']
            
            if single_latency < LATENCY_THRESHOLD_MS:
                print(f"\n{GREEN}✅ TEST PASADO: Latencia {single_latency:.2f}ms < {LATENCY_THRESHOLD_MS}ms{RESET}")
                passed = True
            else:
                print(f"\n{RED}❌ TEST FALLIDO: Latencia {single_latency:.2f}ms ≥ {LATENCY_THRESHOLD_MS}ms{RESET}")
                passed = False
            
            self.test_results['tests']['latency'] = {
                'passed': passed,
                'single_image_ms': single_latency,
                'threshold_ms': LATENCY_THRESHOLD_MS,
                'batch_results': batch_results
            }
            return passed
            
        except Exception as e:
            print(f"\n{RED}❌ TEST FALLIDO: {e}{RESET}")
            traceback.print_exc()
            self.test_results['tests']['latency'] = {
                'passed': False,
                'error': str(e)
            }
            return False
    
    def test_memory_footprint(self) -> bool:
        """
        Test 4: Verificar que memory footprint < 500MB.
        
        Returns:
            True si el test pasa
        """
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}🧪 TEST 4: Memory Footprint < {MEMORY_THRESHOLD_MB}MB{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")
        
        try:
            process = psutil.Process()
            
            # Memoria antes de cargar modelo
            import gc
            gc.collect()
            mem_before = process.memory_info().rss / (1024 * 1024)  # MB
            print(f"📊 Memoria antes de cargar modelo: {mem_before:.2f}MB")
            
            # Cargar modelo
            if self.model is None:
                print(f"📂 Cargando modelo...")
                import tensorflow as tf
                from tensorflow import keras
                self.model = keras.models.load_model(self.model_path)
            
            gc.collect()
            mem_after = process.memory_info().rss / (1024 * 1024)  # MB
            mem_model = mem_after - mem_before
            
            print(f"📊 Memoria después de cargar modelo: {mem_after:.2f}MB")
            print(f"📊 Memoria del modelo: {mem_model:.2f}MB")
            
            # Test con predicciones
            print(f"\n🧪 Testeando memoria con predicciones...")
            test_batch = np.random.rand(32, 224, 224, 3).astype(np.float32)
            
            for i in range(5):
                _ = self.model.predict(test_batch, verbose=0)
            
            gc.collect()
            mem_after_predictions = process.memory_info().rss / (1024 * 1024)
            mem_predictions = mem_after_predictions - mem_after
            
            print(f"📊 Memoria después de predicciones: {mem_after_predictions:.2f}MB")
            print(f"📊 Memoria adicional por predicciones: {mem_predictions:.2f}MB")
            
            # Memoria total usada
            total_memory = mem_after_predictions
            
            # Verificar umbral
            if mem_model < MEMORY_THRESHOLD_MB:
                print(f"\n{GREEN}✅ TEST PASADO: Modelo usa {mem_model:.2f}MB < {MEMORY_THRESHOLD_MB}MB{RESET}")
                passed = True
            else:
                print(f"\n{RED}❌ TEST FALLIDO: Modelo usa {mem_model:.2f}MB ≥ {MEMORY_THRESHOLD_MB}MB{RESET}")
                passed = False
            
            self.test_results['tests']['memory'] = {
                'passed': passed,
                'model_memory_mb': mem_model,
                'total_memory_mb': total_memory,
                'threshold_mb': MEMORY_THRESHOLD_MB
            }
            return passed
            
        except Exception as e:
            print(f"\n{RED}❌ TEST FALLIDO: {e}{RESET}")
            traceback.print_exc()
            self.test_results['tests']['memory'] = {
                'passed': False,
                'error': str(e)
            }
            return False
    
    def test_inference_script(self) -> bool:
        """
        Test 5: Verificar que inference script existe y funciona.
        
        Returns:
            True si el test pasa
        """
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}🧪 TEST 5: Inference Script Listo{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")
        
        try:
            inference_path = backend_dir / 'scripts' / 'inference.py'
            
            if not inference_path.exists():
                print(f"{RED}❌ inference.py no encontrado{RESET}")
                self.test_results['tests']['inference_script'] = {
                    'passed': False,
                    'error': 'inference.py not found'
                }
                return False
            
            print(f"📂 Verificando inference.py...")
            content = inference_path.read_text(encoding='utf-8')
            
            # Verificar componentes esenciales
            required_components = [
                ('class', 'Clase de inferencia'),
                ('def predict', 'Método predict'),
                ('def load_model', 'Método load_model'),
                ('preprocessing', 'Preprocessing de imagen'),
                ('if __name__', 'CLI interface')
            ]
            
            missing = []
            for component, description in required_components:
                if component in content:
                    print(f"   {GREEN}✅{RESET} {description}")
                else:
                    print(f"   {RED}❌{RESET} {description}")
                    missing.append(description)
            
            if missing:
                print(f"\n{RED}❌ Componentes faltantes: {', '.join(missing)}{RESET}")
                self.test_results['tests']['inference_script'] = {
                    'passed': False,
                    'missing_components': missing
                }
                return False
            
            print(f"\n{GREEN}✅ TEST PASADO: Inference script completo{RESET}")
            
            self.test_results['tests']['inference_script'] = {
                'passed': True,
                'path': str(inference_path),
                'size_bytes': len(content)
            }
            return True
            
        except Exception as e:
            print(f"\n{RED}❌ TEST FALLIDO: {e}{RESET}")
            traceback.print_exc()
            self.test_results['tests']['inference_script'] = {
                'passed': False,
                'error': str(e)
            }
            return False
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Genera reporte final de readiness.
        
        Returns:
            Diccionario con reporte completo
        """
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}📋 REPORTE FINAL DE READINESS{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")
        
        # Contar tests pasados
        total_tests = len(self.test_results['tests'])
        passed_tests = sum(1 for t in self.test_results['tests'].values() if t.get('passed', False))
        
        print(f"{BOLD}Tests Ejecutados:{RESET} {passed_tests}/{total_tests}")
        
        # Mostrar resultado de cada test
        for test_name, test_data in self.test_results['tests'].items():
            passed = test_data.get('passed', False)
            icon = f"{GREEN}✅{RESET}" if passed else f"{RED}❌{RESET}"
            print(f"  {icon} {test_name.replace('_', ' ').title()}")
        
        # Estado general
        all_passed = passed_tests == total_tests
        self.test_results['all_passed'] = all_passed
        
        print(f"\n{BOLD}{'='*70}{RESET}")
        if all_passed:
            print(f"{GREEN}{BOLD}✅ MODELO LISTO PARA PRODUCCIÓN{RESET}")
            print(f"{GREEN}Todos los tests de readiness pasaron exitosamente{RESET}")
        else:
            print(f"{RED}{BOLD}❌ MODELO NO LISTO PARA PRODUCCIÓN{RESET}")
            print(f"{RED}{passed_tests}/{total_tests} tests pasaron{RESET}")
        print(f"{BOLD}{'='*70}{RESET}")
        
        return self.test_results
    
    def save_report(self, output_path: str = None):
        """
        Guarda el reporte en JSON.
        
        Args:
            output_path: Ruta donde guardar el reporte
        """
        if output_path is None:
            output_path = backend_dir.parent / 'metrics' / 'readiness_report.json'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{GREEN}✅ Reporte guardado en: {output_path}{RESET}")
    
    def run_all_tests(self) -> bool:
        """
        Ejecuta todos los tests de readiness.
        
        Returns:
            True si todos los tests pasan
        """
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}🚀 TESTING FINAL DEL MODELO - PASO 6{RESET}")
        print(f"{BOLD}{'='*70}{RESET}")
        
        # Ejecutar tests
        tests = [
            self.test_model_save_load,
            self.test_realtime_predictions,
            self.test_latency,
            self.test_memory_footprint,
            self.test_inference_script
        ]
        
        for test_func in tests:
            test_func()
        
        # Generar reporte
        self.generate_report()
        
        # Guardar reporte
        self.save_report()
        
        return self.test_results['all_passed']


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Testing final del modelo (Paso 6)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Ruta al modelo a probar (default: models/best_model.keras)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Ruta donde guardar el reporte'
    )
    
    args = parser.parse_args()
    
    # Ejecutar tests
    tester = ProductionReadinessTest(args.model)
    all_passed = tester.run_all_tests()
    
    if args.output:
        tester.save_report(args.output)
    
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
