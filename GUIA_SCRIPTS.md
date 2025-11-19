# 🚀 Guía Rápida de Uso - Scripts

## 📋 Orden de Ejecución

### 🆕 PRIMERA VEZ (Configuración completa):

```batch
# Paso 1: Configurar proyecto (solo una vez)
setup.bat

# Paso 2: Limpiar cache antiguo (obligatorio por cambio de resolución)
clean_cache.bat

# Paso 3: Entrenar modelo con optimizaciones v2.0
train.bat

# Paso 4: Iniciar aplicación
start-backend.bat   # Terminal 1
start-frontend.bat  # Terminal 2
```

**Tiempo estimado primera vez**: 2-3 horas
- setup.bat: 5-10 minutos
- clean_cache.bat: instantáneo
- train.bat: 1.5-2 horas
- start backend/frontend: instantáneo

---

### ✅ USO NORMAL (Ya configurado):

```batch
# Solo necesitas estos 2 scripts:
start-backend.bat   # Terminal 1 - API Flask en http://localhost:5000
start-frontend.bat  # Terminal 2 - React en http://localhost:3000
```

**Abrir navegador**: `http://localhost:3000`

---

### 🔄 RE-ENTRENAR (Mejorar modelo):

```batch
# Opción A: Re-entrenar sin limpiar cache (más rápido)
train.bat

# Opción B: Re-entrenar desde cero (limpiar todo)
clean_cache.bat
train.bat
```

---

### 🧹 LIMPIAR CACHE (Solo cuando sea necesario):

```batch
clean_cache.bat
```

**Cuándo usar**:
- ✅ Primera vez (cambio 100×100 → 224×224)
- ✅ Cambias `IMG_SIZE` en `backend/config.py`
- ✅ Error: `Shape mismatch` o incompatibilidad
- ✅ Quieres re-entrenar desde cero
- ❌ NO usar en uso normal

---

## 📊 Descripción de Scripts

### 1. `setup.bat` - Configuración Inicial ⚙️
**Uso**: Solo la primera vez

**Qué hace**:
1. Verifica Python ≥ 3.10
2. Crea entorno virtual (venv)
3. Instala dependencias Python (requirements.txt)
4. Verifica Node.js
5. Instala dependencias React (npm install)
6. Crea directorios necesarios

**Cuándo ejecutar**:
- Primera vez que usas el proyecto
- Cambias de computadora
- Borraste el entorno virtual

---

### 2. `clean_cache.bat` - Limpiar Cache 🧹
**Uso**: Solo cuando cambies resolución o haya errores

**Qué hace**:
1. Verifica procesos Python activos (advertencia)
2. Elimina `backend/cache/*.pkl` (datos 100×100)
3. Elimina `backend/cache/*.json` (metadatos)
4. Elimina `models/*.keras` (modelos antiguos)

**Archivos eliminados**:
```
backend/cache/
├── X_train.pkl          ❌ (100×100 incompatible)
├── y_train.pkl          ❌
├── X_test.pkl           ❌
├── y_test.pkl           ❌
├── class_names.pkl      ❌
└── cache_metadata.json  ❌

models/
└── plant_disease_model.keras  ❌ (entrenado con 100×100)
```

**Tiempo**: Instantáneo

**Cuándo ejecutar**:
- ✅ Primera vez (migración 100×100 → 224×224)
- ✅ Cambias `IMG_SIZE` en config.py
- ✅ Error: `ValueError: Input shape mismatch`
- ✅ Quieres entrenar desde cero

---

### 3. `train.bat` - Entrenamiento Optimizado 🎯
**Uso**: Entrenar el modelo

**Qué hace**:
1. Verifica Python
2. Instala/actualiza dependencias
3. Ejecuta `backend/scripts/train.py`:
   - Detecta cache PKL existente
   - Si no hay cache: Prepara datos (15-25 min)
   - Si hay cache: Carga instantánea
   - **Phase 1** (15 epochs): Clasificador
   - **Phase 2a** (10 epochs): Fine-tuning capas 101-154
   - **Phase 2b** (10 epochs): Fine-tuning capas 51-154
   - Evalúa con métricas detalladas
   - Genera 4 visualizaciones + reporte

**Optimizaciones activas**:
- ✅ Fine-tuning progresivo (3 fases)
- ✅ Learning rates conservadoras (0.001 → 0.0001 → 0.00005)
- ✅ Resolución 224×224 (5x más detalle)
- ✅ Métricas detalladas (20+ métricas)
- ✅ Cache PKL automático

**Archivos generados**:
```
models/
├── plant_disease_model.keras               # Modelo entrenado
├── training_history.json                    # Historial
└── visualizations/
    ├── confusion_matrix_detailed.png        # Matriz 16×14
    ├── per_class_metrics.png                # Precision/Recall/F1
    ├── per_crop_performance.png             # Por cultivo
    ├── healthy_vs_diseased.png              # Matriz 2×2
    ├── training_history.png                 # Loss/Accuracy
    └── training_report.txt                  # Reporte detallado
```

**Tiempo estimado**:
- Primera vez (sin cache): 1.5-2 horas
- Con cache existente: 1-1.5 horas

**Cuándo ejecutar**:
- Primera vez después de `clean_cache.bat`
- Quieres mejorar el modelo
- Agregaste más datos al dataset

---

### 4. `start-backend.bat` - Iniciar API 🔧
**Uso**: Iniciar servidor Flask

**Qué hace**:
1. Activa entorno virtual (venv)
2. Ejecuta `backend/app.py`
3. Carga modelo `plant_disease_model.keras`
4. Inicia API en `http://localhost:5000`

**Endpoints disponibles**:
```
GET  /                    # Health check
POST /predict             # Predicción de enfermedad
```

**Requisitos previos**:
- ✅ Modelo entrenado (`models/plant_disease_model.keras`)
- ✅ Dependencias instaladas

**Mantener abierto**: Sí (Terminal 1)

---

### 5. `start-frontend.bat` - Iniciar Interfaz 🎨
**Uso**: Iniciar aplicación React

**Qué hace**:
1. Ejecuta `npm start` en carpeta `frontend/`
2. Compila React app
3. Abre navegador en `http://localhost:3000`

**Requisitos previos**:
- ✅ Backend corriendo (`start-backend.bat`)
- ✅ Node.js instalado
- ✅ Dependencias instaladas (`npm install`)

**Mantener abierto**: Sí (Terminal 2)

---

## 🔍 Troubleshooting

### Error: "Python no está instalado"
**Solución**:
```batch
# Verificar Python
python --version

# Si falla, instalar desde:
# https://www.python.org/downloads/
```

### Error: "Shape mismatch" o "Expected (100, 100, 3)"
**Causa**: Cache antiguo con resolución 100×100

**Solución**:
```batch
clean_cache.bat
train.bat
```

### Error: "No module named 'tensorflow'"
**Solución**:
```batch
# Re-instalar dependencias
pip install -r backend\requirements.txt
```

### Error: "Port 5000 already in use"
**Causa**: Backend ya está corriendo

**Solución**:
```batch
# Opción A: Cerrar proceso Python
taskkill /F /IM python.exe

# Opción B: Cambiar puerto en backend/app.py
# app.run(port=5001)
```

### Error: "Port 3000 already in use"
**Causa**: Frontend ya está corriendo

**Solución**:
```batch
# Cerrar proceso Node.js
taskkill /F /IM node.exe
```

### El entrenamiento es muy lento
**Causas posibles**:
- No tienes GPU (10x más lento en CPU)
- Batch size muy grande para tu RAM/VRAM

**Solución**:
```python
# En backend/scripts/train.py
BATCH_SIZE = 8  # Reducir de 16 a 8 si hay problemas de memoria
```

### Cache no se regenera automáticamente
**Verificación**:
```batch
# Ver estado del cache
python backend\utils\manage_cache.py

# Opción 1: Ver estado del cache
# Opción 3: Verificar compatibilidad
```

---

## 📊 Métricas Generadas

Después de ejecutar `train.bat`, revisa:

### 1. Console Output
```
📊 EVALUACIÓN DETALLADA DEL MODELO
Test Accuracy: 95.67%
Top-3 Accuracy: 98.45%
Top-5 Accuracy: 99.12%

Métricas por clase (15):
Métricas por cultivo (4):
Healthy vs Diseased:
Top 10 confusiones:
```

### 2. Visualizaciones PNG (models/visualizations/)
- `confusion_matrix_detailed.png` - Ver qué clases se confunden
- `per_class_metrics.png` - Ver rendimiento por enfermedad
- `per_crop_performance.png` - Ver rendimiento por cultivo
- `healthy_vs_diseased.png` - Ver detección binaria

### 3. Reporte TXT (models/visualizations/training_report.txt)
- Configuración completa
- Métricas detalladas
- Recomendaciones automáticas

---

## 🎯 Workflow Completo

### Primera vez:
```batch
1. setup.bat              # 5-10 min
2. clean_cache.bat        # instantáneo
3. train.bat              # 1.5-2 horas
4. start-backend.bat      # mantener abierto
5. start-frontend.bat     # mantener abierto
6. Abrir: http://localhost:3000
```

### Uso diario:
```batch
1. start-backend.bat      # Terminal 1
2. start-frontend.bat     # Terminal 2
3. Usar aplicación
```

### Re-entrenar:
```batch
1. Cerrar backend/frontend (Ctrl+C)
2. train.bat              # 1-1.5 horas (con cache)
3. start-backend.bat
4. start-frontend.bat
```