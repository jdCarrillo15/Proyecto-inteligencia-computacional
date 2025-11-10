# 🍎 Clasificador de Frutas con CNN 🍌

Sistema completo de clasificación de frutas utilizando Redes Neuronales Convolucionales (CNN) desarrollado con TensorFlow y Keras. Incluye limpieza de datos, entrenamiento del modelo y aplicación web interactiva.

## 📋 Descripción

Este proyecto implementa un clasificador de imágenes de frutas capaz de identificar 5 tipos diferentes:
- 🍎 Manzana
- 🍌 Banano
- 🥭 Mango
- 🍊 Naranja
- 🍐 Pera

El sistema incluye:
1. **Limpieza y preparación de datos** con visualizaciones
2. **Modelo CNN** entrenado con TensorFlow/Keras
3. **Aplicación web** con Flask para predicciones en tiempo real

## 🚀 Características

### Preparación de Datos
- ✅ Verificación de archivos corruptos o vacíos
- ✅ Eliminación de imágenes con dimensiones inconsistentes
- ✅ Redimensionamiento uniforme a 100x100 píxeles
- ✅ Normalización de valores de píxel (0-1)
- ✅ División automática: 80% entrenamiento, 20% prueba
- ✅ Visualizaciones de distribución de clases y ejemplos

### Modelo CNN
- ✅ Arquitectura con capas Conv2D, MaxPooling2D y Dense
- ✅ Activaciones ReLU y Softmax
- ✅ Optimizador Adam
- ✅ Función de pérdida: Categorical Crossentropy
- ✅ Data Augmentation para mejorar generalización
- ✅ Early Stopping y Model Checkpointing
- ✅ Exportación en formato .h5

### Aplicación Web
- ✅ Interfaz moderna y responsiva
- ✅ Carga de imágenes por drag & drop o selección
- ✅ Validación de formato de imagen
- ✅ Predicción en tiempo real
- ✅ Visualización de confianza del modelo
- ✅ Compatible con Chrome, Firefox y Edge
- ✅ Funciona en Windows y Linux

## 📁 Estructura del Proyecto

```
Proyecto-inteligencia-computacional/
├── dataset/
│   ├── raw/                    # Dataset original
│   │   ├── manzana/
│   │   ├── banano/
│   │   ├── mango/
│   │   ├── naranja/
│   │   └── pera/
│   └── processed/              # Dataset limpio
│       ├── train/
│       ├── test/
│       └── visualizations/
├── models/                     # Modelos entrenados
│   ├── fruit_classifier.h5
│   ├── best_model.h5
│   ├── class_mapping.json
│   └── visualizations/
├── scripts/                    # Scripts principales
│   ├── data_preparation.py    # Limpieza de datos
│   ├── train_model.py         # Entrenamiento
│   └── predict.py             # Predicción CLI
├── utils/                      # Utilidades
│   ├── verify_installation.py # Verificación
│   ├── download_sample_dataset.py
│   └── quick_test.py          # Tests rápidos
├── docs/                       # Documentación
│   ├── GUIA_RAPIDA.md
│   ├── INICIO.txt
│   └── RESUMEN_PROYECTO.md
├── templates/                  # Templates HTML
│   └── index.html
├── static/                     # Archivos estáticos
│   └── uploads/
├── app.py                      # Aplicación Flask
├── config.py                   # Configuración
├── Makefile                    # Comandos make
├── requirements.txt            # Dependencias
└── README.md                   # Este archivo
```

## 🔧 Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone <url-del-repositorio>
cd Proyecto-inteligencia-computacional
```

2. **Crear entorno virtual (recomendado)**
```bash
python -m venv venv

# En Linux/Mac:
source venv/bin/activate

# En Windows:
venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

## 📊 Preparación del Dataset

### 1. Organizar el Dataset

Crea la siguiente estructura y coloca tus imágenes:

```
dataset/raw/
├── manzana/
│   ├── imagen1.jpg
│   ├── imagen2.png
│   └── ...
├── banano/
│   └── ...
├── mango/
│   └── ...
├── naranja/
│   └── ...
└── pera/
    └── ...
```

**Formatos aceptados:** `.jpg`, `.jpeg`, `.png`

### 2. Ejecutar Limpieza de Datos

```bash
python scripts/data_preparation.py
```

Este script:
- Verifica la integridad de todas las imágenes
- Elimina archivos corruptos
- Redimensiona a 100x100 píxeles
- Normaliza valores de píxel
- Divide en train/test (80/20)
- Genera visualizaciones en `dataset/processed/visualizations/`

**Visualizaciones generadas:**
- `class_distribution.png` - Distribución de clases
- `sample_images.png` - Ejemplos de imágenes limpias
- `train_test_split.png` - División train/test

## 🧠 Entrenamiento del Modelo

```bash
python scripts/train_model.py
```

### Configuración del Entrenamiento

El script utiliza los siguientes parámetros por defecto:
- **Tamaño de imagen:** 100x100 píxeles
- **Batch size:** 32
- **Épocas:** 50 (con early stopping)
- **Optimizador:** Adam (lr=0.001)
- **Data Augmentation:** Rotación, zoom, flip horizontal

### Arquitectura del Modelo

```
Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
Conv2D(256) → BatchNorm → MaxPool → Dropout(0.25)
Flatten
Dense(512) → BatchNorm → Dropout(0.5)
Dense(256) → BatchNorm → Dropout(0.5)
Dense(5, softmax)
```

### Callbacks Implementados

- **Early Stopping:** Detiene el entrenamiento si no hay mejora en 10 épocas
- **Model Checkpoint:** Guarda el mejor modelo según val_accuracy
- **Reduce LR on Plateau:** Reduce learning rate si no hay mejora

### Archivos Generados

- `models/fruit_classifier.h5` - Modelo final
- `models/best_model.h5` - Mejor modelo durante entrenamiento
- `models/class_mapping.json` - Mapeo de clases
- `models/visualizations/confusion_matrix.png` - Matriz de confusión
- `models/visualizations/training_history.png` - Historial de entrenamiento

## 🌐 Aplicación Web

### Iniciar la Aplicación

```bash
python app.py
```

La aplicación estará disponible en: **http://localhost:5000**

### Características de la Interfaz

1. **Carga de Imágenes**
   - Arrastra y suelta imágenes
   - O haz clic para seleccionar
   - Validación automática de formato

2. **Predicción**
   - Procesamiento automático de la imagen
   - Muestra la clase predicha con emoji
   - Porcentaje de confianza
   - Ranking de todas las predicciones

3. **Validación**
   - Verifica formato de imagen
   - Valida dimensiones mínimas
   - Mensajes de error informativos

### Endpoints de la API

- `GET /` - Página principal
- `POST /predict` - Realizar predicción
- `GET /health` - Estado de la aplicación
- `GET /dataset-info` - Información del dataset

## 🎯 Uso del Sistema

### Flujo Completo

1. **Preparar datos:**
```bash
python scripts/data_preparation.py
```

2. **Entrenar modelo:**
```bash
python scripts/train_model.py
```

3. **Iniciar aplicación:**
```bash
python app.py
```

4. **Usar la aplicación:**
   - Abre http://localhost:5000 en tu navegador
   - Sube una imagen de fruta
   - Haz clic en "Clasificar Fruta"
   - Visualiza los resultados

## 📈 Métricas y Evaluación

El sistema genera automáticamente:

1. **Durante la limpieza:**
   - Estadísticas de imágenes procesadas
   - Distribución de clases
   - Ejemplos visuales

2. **Durante el entrenamiento:**
   - Accuracy y Loss por época
   - Métricas de validación
   - Matriz de confusión
   - Reporte de clasificación (Precision, Recall, F1-Score)

## 🔍 Solución de Problemas

### Error: "No se encontró el dataset"
- Verifica que la carpeta `dataset/raw/` existe
- Asegúrate de tener las 5 subcarpetas de frutas
- Verifica que hay imágenes en cada carpeta

### Error: "Modelo no cargado"
- Ejecuta primero `train_model.py`
- Verifica que existe `models/fruit_classifier.h5`

### Error de memoria durante el entrenamiento
- Reduce el `batch_size` en `train_model.py`
- Cierra otras aplicaciones que consuman RAM

### La aplicación web no inicia
- Verifica que el puerto 5000 no esté en uso
- Instala todas las dependencias: `pip install -r requirements.txt`

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **TensorFlow 2.15** - Framework de deep learning
- **Keras** - API de alto nivel para redes neuronales
- **OpenCV** - Procesamiento de imágenes
- **Flask** - Framework web
- **Matplotlib/Seaborn** - Visualizaciones
- **NumPy** - Operaciones numéricas
- **Pillow** - Manipulación de imágenes

## 📝 Notas Importantes

1. **Dataset:** Se recomienda tener al menos 100 imágenes por clase para buenos resultados
2. **Calidad:** Las imágenes deben ser claras y mostrar principalmente la fruta
3. **Formato:** Acepta JPG y PNG
4. **Tamaño:** Las imágenes se redimensionan automáticamente a 100x100
5. **Navegadores:** Compatible con Chrome, Firefox, Edge (versiones recientes)

## 🎓 Mejoras Futuras

- [ ] Agregar más clases de frutas
- [ ] Implementar transfer learning (VGG16, ResNet)
- [ ] Añadir validación cruzada
- [ ] Desplegar en la nube (Heroku, AWS, GCP)
- [ ] Crear API REST completa
- [ ] Agregar autenticación de usuarios
- [ ] Implementar historial de predicciones

## 👥 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📧 Contacto

Para preguntas o sugerencias, por favor abre un issue en el repositorio.

---

**Desarrollado con ❤️ usando TensorFlow y Flask**
