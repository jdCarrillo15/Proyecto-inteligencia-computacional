# 🚀 Guía Rápida de Inicio

Esta guía te ayudará a poner en marcha el clasificador de frutas en pocos minutos.

## ⚡ Inicio Rápido (3 pasos)

### 1️⃣ Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2️⃣ Preparar el Dataset

Crea la estructura de carpetas y coloca tus imágenes:

```
dataset/raw/
├── manzana/     # Coloca aquí imágenes de manzanas
├── banano/      # Coloca aquí imágenes de bananos
├── mango/       # Coloca aquí imágenes de mangos
├── naranja/     # Coloca aquí imágenes de naranjas
└── pera/        # Coloca aquí imágenes de peras
```

Luego ejecuta:

```bash
python data_preparation.py
```

### 3️⃣ Entrenar y Usar

```bash
# Entrenar el modelo (puede tomar varios minutos)
python train_model.py

# Iniciar la aplicación web
python app.py
```

Abre tu navegador en: **http://localhost:5000**

## 📝 Comandos Útiles

### Predicción desde Terminal

```bash
# Predicción simple
python predict.py mi_imagen.jpg

# Ver todas las probabilidades
python predict.py mi_imagen.jpg --all

# Usar un modelo específico
python predict.py mi_imagen.jpg --model models/best_model.h5 --all
```

### Verificar Estado

```bash
# Ver estructura del proyecto
tree -L 3

# Verificar instalación de dependencias
pip list | grep -E "tensorflow|keras|flask"
```

## 🎯 Checklist de Verificación

Antes de entrenar, asegúrate de:

- [ ] Python 3.8+ instalado
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] Dataset organizado en `dataset/raw/`
- [ ] Al menos 50-100 imágenes por clase
- [ ] Imágenes en formato JPG o PNG

## 🔧 Configuración Personalizada

### Cambiar Tamaño de Imagen

Edita en `data_preparation.py` y `train_model.py`:

```python
img_size = (150, 150)  # Cambiar de 100x100 a 150x150
```

### Ajustar División Train/Test

En `data_preparation.py`:

```python
cleaner.clean_dataset(train_split=0.7)  # 70% train, 30% test
```

### Modificar Épocas de Entrenamiento

En `train_model.py`:

```python
EPOCHS = 100  # Cambiar de 50 a 100 épocas
```

### Cambiar Batch Size

En `train_model.py`:

```python
BATCH_SIZE = 16  # Reducir si hay problemas de memoria
```

## 📊 Interpretación de Resultados

### Durante el Entrenamiento

```
Epoch 10/50
45/45 [==============================] - 12s 267ms/step
loss: 0.3456 - accuracy: 0.8923 - val_loss: 0.4123 - val_accuracy: 0.8567
```

- **loss**: Pérdida en entrenamiento (menor es mejor)
- **accuracy**: Precisión en entrenamiento (mayor es mejor)
- **val_loss**: Pérdida en validación
- **val_accuracy**: Precisión en validación (métrica clave)

### Matriz de Confusión

La matriz muestra:
- **Diagonal**: Predicciones correctas
- **Fuera de diagonal**: Confusiones entre clases

### Métricas de Clasificación

- **Precision**: De las predicciones positivas, cuántas fueron correctas
- **Recall**: De los casos positivos reales, cuántos se detectaron
- **F1-Score**: Media armónica de precision y recall

## 🐛 Solución Rápida de Problemas

### "No module named 'tensorflow'"
```bash
pip install tensorflow==2.15.0
```

### "No se encontró el dataset"
```bash
# Crear estructura de carpetas
mkdir -p dataset/raw/{manzana,banano,mango,naranja,pera}
```

### "Out of Memory" durante entrenamiento
```python
# En train_model.py, reducir batch_size
BATCH_SIZE = 16  # o incluso 8
```

### Puerto 5000 ocupado
```python
# En app.py, cambiar el puerto
app.run(debug=True, host='0.0.0.0', port=8080)
```

### Modelo no carga en la app
```bash
# Verificar que existe el modelo
ls -lh models/fruit_classifier.h5

# Si no existe, entrenar primero
python train_model.py
```

## 📈 Mejorando el Rendimiento

### 1. Más Datos
- Objetivo: 200-500 imágenes por clase
- Usar data augmentation (ya incluido)

### 2. Ajustar Hiperparámetros
- Learning rate: Probar 0.0001 o 0.01
- Batch size: Probar 16, 32, 64
- Épocas: Aumentar a 100

### 3. Transfer Learning
- Usar modelos pre-entrenados (VGG16, ResNet50)
- Requiere modificar `train_model.py`

## 🎓 Próximos Pasos

1. **Experimentar con el modelo**
   - Probar diferentes arquitecturas
   - Ajustar hiperparámetros
   - Agregar más capas

2. **Mejorar la aplicación**
   - Agregar más funcionalidades
   - Mejorar el diseño
   - Implementar historial de predicciones

3. **Desplegar en producción**
   - Usar Docker
   - Desplegar en Heroku/AWS
   - Crear API REST

## 📚 Recursos Adicionales

- [Documentación de TensorFlow](https://www.tensorflow.org/)
- [Guía de Keras](https://keras.io/)
- [Tutorial de Flask](https://flask.palletsprojects.com/)
- [Dataset de Frutas en Kaggle](https://www.kaggle.com/datasets)

## 💡 Consejos Pro

1. **Usa GPU si está disponible** - El entrenamiento será mucho más rápido
2. **Guarda checkpoints** - Ya implementado con ModelCheckpoint
3. **Monitorea el overfitting** - Compara train vs validation accuracy
4. **Experimenta con data augmentation** - Ajusta los parámetros en `train_model.py`
5. **Documenta tus experimentos** - Anota qué cambios mejoran el modelo

---

**¿Necesitas ayuda?** Abre un issue en el repositorio o consulta el README.md completo.
