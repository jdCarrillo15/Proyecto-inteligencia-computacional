"""
Sistema de cache con PKL para acelerar el procesamiento de datos.
Guarda y carga datos procesados para evitar recalcular en cada entrenamiento.
"""

import pickle
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
import json


class DataCache:
    """Gestiona el cache de datos procesados con archivos PKL."""
    
    def __init__(self, cache_dir='backend/cache'):
        """
        Inicializa el sistema de cache.
        
        Args:
            cache_dir: Directorio donde se guardarán los archivos PKL
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """Carga metadatos del cache."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Guarda metadatos del cache."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
    
    def _get_cache_key(self, dataset_path, config):
        """
        Genera una clave única para el cache basada en la configuración.
        
        Args:
            dataset_path: Ruta al dataset
            config: Diccionario con configuración (img_size, classes, etc.)
            
        Returns:
            String con la clave de cache
        """
        # Crear string único con la configuración
        config_str = f"{dataset_path}_{config['img_size']}_{config['classes']}_{config.get('balance', False)}"
        
        # Hash MD5 para nombre corto
        cache_key = hashlib.md5(config_str.encode()).hexdigest()
        return cache_key
    
    def get_cache_path(self, cache_key, split='train'):
        """Obtiene la ruta del archivo de cache."""
        return self.cache_dir / f"{cache_key}_{split}.pkl"
    
    def exists(self, dataset_path, config, split='train'):
        """
        Verifica si existe un cache válido.
        
        Args:
            dataset_path: Ruta al dataset
            config: Configuración del procesamiento
            split: 'train' o 'test'
            
        Returns:
            bool: True si existe cache válido
        """
        cache_key = self._get_cache_key(dataset_path, config)
        cache_path = self.get_cache_path(cache_key, split)
        
        if not cache_path.exists():
            return False
        
        # Verificar que el cache no sea demasiado viejo (opcional)
        # Por ahora solo verificamos existencia
        return True
    
    def save(self, data, labels, class_names, dataset_path, config, split='train'):
        """
        Guarda datos procesados en cache.
        
        Args:
            data: Array numpy con las imágenes procesadas
            labels: Array numpy con las etiquetas
            class_names: Lista con nombres de clases
            dataset_path: Ruta al dataset original
            config: Configuración del procesamiento
            split: 'train' o 'test'
        """
        cache_key = self._get_cache_key(dataset_path, config)
        cache_path = self.get_cache_path(cache_key, split)
        
        print(f"💾 Guardando cache en {cache_path.name}...")
        
        # Empaquetar datos
        cache_data = {
            'data': data,
            'labels': labels,
            'class_names': class_names,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'data_shape': data.shape,
            'labels_shape': labels.shape
        }
        
        # Guardar con pickle (protocolo 4 para mejor rendimiento)
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=4)
        
        # Actualizar metadatos
        self.metadata[cache_key] = {
            'dataset_path': str(dataset_path),
            'config': config,
            'created': datetime.now().isoformat(),
            'splits': self.metadata.get(cache_key, {}).get('splits', {})
        }
        self.metadata[cache_key]['splits'][split] = {
            'path': str(cache_path),
            'data_shape': list(data.shape),
            'labels_shape': list(labels.shape),
            'num_samples': len(data)
        }
        self._save_metadata()
        
        file_size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"✅ Cache guardado: {file_size_mb:.2f} MB, {len(data)} muestras")
    
    def load(self, dataset_path, config, split='train'):
        """
        Carga datos del cache.
        
        Args:
            dataset_path: Ruta al dataset original
            config: Configuración del procesamiento
            split: 'train' o 'test'
            
        Returns:
            tuple: (data, labels, class_names) o None si no existe
        """
        cache_key = self._get_cache_key(dataset_path, config)
        cache_path = self.get_cache_path(cache_key, split)
        
        if not cache_path.exists():
            return None
        
        print(f"📂 Cargando cache desde {cache_path.name}...")
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            data = cache_data['data']
            labels = cache_data['labels']
            class_names = cache_data['class_names']
            
            print(f"✅ Cache cargado: {len(data)} muestras, shape={data.shape}")
            
            return data, labels, class_names
        
        except Exception as e:
            print(f"⚠️ Error al cargar cache: {e}")
            return None
    
    def clear(self, cache_key=None):
        """
        Limpia archivos de cache.
        
        Args:
            cache_key: Clave específica a limpiar, o None para limpiar todo
        """
        if cache_key:
            # Limpiar cache específico
            for split in ['train', 'test']:
                cache_path = self.get_cache_path(cache_key, split)
                if cache_path.exists():
                    cache_path.unlink()
                    print(f"🗑️  Cache eliminado: {cache_path.name}")
            
            if cache_key in self.metadata:
                del self.metadata[cache_key]
                self._save_metadata()
        else:
            # Limpiar todo el cache
            for pkl_file in self.cache_dir.glob('*.pkl'):
                pkl_file.unlink()
                print(f"🗑️  Cache eliminado: {pkl_file.name}")
            
            self.metadata = {}
            self._save_metadata()
        
        print("✅ Cache limpiado")
    
    def get_info(self):
        """Obtiene información sobre el cache."""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob('*.pkl'))
        total_files = len(list(self.cache_dir.glob('*.pkl')))
        
        info = {
            'total_files': total_files,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
            'cached_datasets': len(self.metadata)
        }
        
        return info
    
    def print_info(self):
        """Imprime información del cache."""
        info = self.get_info()
        
        print("\n" + "=" * 60)
        print("INFORMACIÓN DEL CACHE")
        print("=" * 60)
        print(f"📁 Directorio: {info['cache_dir']}")
        print(f"📊 Archivos PKL: {info['total_files']}")
        print(f"💾 Tamaño total: {info['total_size_mb']:.2f} MB")
        print(f"🗂️  Datasets cacheados: {info['cached_datasets']}")
        
        if self.metadata:
            print("\n📋 Datasets en cache:")
            for cache_key, meta in self.metadata.items():
                print(f"\n  🔑 {cache_key[:8]}...")
                print(f"     Dataset: {Path(meta['dataset_path']).name}")
                print(f"     Creado: {meta['created'][:19]}")
                for split, split_info in meta.get('splits', {}).items():
                    print(f"     - {split}: {split_info['num_samples']} muestras")
        
        print("=" * 60)


def create_data_arrays_from_directory(directory, img_size=(100, 100), classes=None):
    """
    Crea arrays numpy directamente desde un directorio de imágenes.
    Función auxiliar para trabajar con el cache.
    
    Args:
        directory: Directorio con subdirectorios por clase
        img_size: Tamaño de las imágenes
        classes: Lista de nombres de clases esperadas
        
    Returns:
        tuple: (data_array, labels_array, class_names)
    """
    from PIL import Image
    import numpy as np
    
    directory = Path(directory)
    
    if classes is None:
        classes = sorted([d.name for d in directory.iterdir() if d.is_dir()])
    
    data_list = []
    labels_list = []
    
    print(f"\n📂 Cargando imágenes desde {directory.name}...")
    
    for class_idx, class_name in enumerate(classes):
        class_path = directory / class_name
        
        if not class_path.exists():
            print(f"⚠️  Clase '{class_name}' no encontrada en {directory}")
            continue
        
        # Obtener todas las imágenes
        image_files = list(class_path.glob('*.png')) + \
                      list(class_path.glob('*.jpg')) + \
                      list(class_path.glob('*.jpeg'))
        
        print(f"  📸 {class_name}: {len(image_files)} imágenes")
        
        for img_path in image_files:
            try:
                # Cargar y procesar imagen
                img = Image.open(img_path).convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img, dtype=np.float32) / 255.0  # Normalizar
                
                data_list.append(img_array)
                labels_list.append(class_idx)
            
            except Exception as e:
                print(f"⚠️  Error al cargar {img_path.name}: {e}")
                continue
    
    # Convertir a arrays numpy
    data_array = np.array(data_list, dtype=np.float32)
    labels_array = np.array(labels_list, dtype=np.int32)
    
    print(f"\n✅ Arrays creados: data={data_array.shape}, labels={labels_array.shape}")
    
    return data_array, labels_array, classes


if __name__ == "__main__":
    # Test del sistema de cache
    cache = DataCache()
    cache.print_info()
