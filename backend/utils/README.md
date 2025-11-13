# 🛠️ Utilidades

Esta carpeta contiene scripts de utilidad para configuración, verificación y testing del proyecto.

## Archivos

### ✅ verify_installation.py
**Propósito:** Verificar que todo esté correctamente instalado

**Funcionalidades:**
- Verifica versión de Python
- Comprueba dependencias instaladas
- Valida estructura de directorios
- Verifica archivos del proyecto
- Comprueba dataset y modelo
- Prueba imports de TensorFlow

**Uso:**
```bash
python utils/verify_installation.py
```

**Cuándo usar:**
- Después de clonar el repositorio
- Después de instalar dependencias
- Para diagnosticar problemas
- Antes de comenzar a trabajar

---

### 📥 download_sample_dataset.py
**Propósito:** Ayudar a configurar el dataset

**Funcionalidades:**
- Crear estructura de carpetas vacía
- Guía para descargar desde Kaggle
- Crear imágenes de prueba (testing)
- Instrucciones paso a paso

**Uso:**
```bash
python utils/download_sample_dataset.py
```

**Opciones:**
1. Crear estructura vacía
2. Guía de Kaggle
3. Generar imágenes de prueba
4. Salir

---

### 🧪 quick_test.py
**Propósito:** Suite de tests rápidos

**Funcionalidades:**
- Test de imports
- Test de TensorFlow
- Test de procesamiento de imágenes
- Test de Flask
- Test de estructura de archivos
- Test de configuración
- Test de creación de modelo

**Uso:**
```bash
python utils/quick_test.py
```

**Salida:**
- Reporte detallado de cada test
- Resumen de tests pasados/fallados
- Recomendaciones de acción

---

## Cuándo Usar Cada Utilidad

### verify_installation.py
✅ **Usar cuando:**
- Acabas de clonar el proyecto
- Instalaste nuevas dependencias
- Algo no funciona correctamente
- Quieres un diagnóstico completo

### download_sample_dataset.py
✅ **Usar cuando:**
- No tienes un dataset propio
- Necesitas crear la estructura de carpetas
- Quieres probar el sistema rápidamente
- Necesitas guía para descargar datos

### quick_test.py
✅ **Usar cuando:**
- Quieres verificar que todo funciona
- Hiciste cambios en el código
- Antes de entrenar el modelo
- Para debugging rápido

---

## Flujo Recomendado

```
1. verify_installation.py
   ↓
   ¿Todo OK?
   ↓
2. download_sample_dataset.py
   ↓
   Dataset configurado
   ↓
3. quick_test.py
   ↓
   Tests pasados → ¡Listo para usar!
```

## Notas

- Estos scripts NO modifican el modelo ni los datos
- Son seguros de ejecutar en cualquier momento
- Proporcionan información útil para debugging
- No requieren GPU ni recursos intensivos
