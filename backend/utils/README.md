# Utilidades del Sistema

Directorio de scripts auxiliares para configuración, validación y pruebas del sistema.

## Archivos

### verify_installation.py
**Objetivo:** Validación integral de la configuración del entorno

**Comprobaciones realizadas:**
- Validación de versión de Python
- Verificación de dependencias instaladas
- Inspección de estructura de directorios
- Validación de archivos críticos del proyecto
- Confirmación de disponibilidad de dataset y modelo
- Pruebas de importación de TensorFlow

**Ejecución:**
```bash
python utils/verify_installation.py
```

**Casos de uso recomendados:**
- Posterior a la clonación del repositorio
- Después de actualizar dependencias
- Para diagnóstico de problemas de configuración
- Antes de iniciar desarrollo

---

### download_sample_dataset.py
**Objetivo:** Asistencia en la configuración del conjunto de datos

**Funcionalidades disponibles:**
- Generación de estructura de directorios
- Instructivo para descarga desde Kaggle
- Creación de imágenes sintéticas para pruebas
- Guía paso a paso

**Ejecución:**
```bash
python utils/download_sample_dataset.py
```

**Menú de opciones:**
1. Generar estructura de directorios
2. Mostrar guía de descarga Kaggle
3. Crear dataset de prueba sintético
4. Salir del programa

---

### quick_test.py
**Objetivo:** Batería de pruebas rápidas del sistema

**Pruebas ejecutadas:**
- Validación de importaciones
- Verificación de TensorFlow
- Prueba de procesamiento de imágenes
- Validación de Flask
- Inspección de estructura de archivos
- Revisión de configuración
- Prueba de instanciación del modelo

**Ejecución:**
```bash
python utils/quick_test.py
```

**Información generada:**
- Informe detallado por prueba
- Resumen de pruebas exitosas y fallidas
- Sugerencias de corrección

---

## Guía de uso de utilidades

### verify_installation.py
**Escenarios de aplicación:**
- Posterior a clonación del repositorio
- Después de actualizar dependencias
- Cuando se presentan problemas de funcionamiento
- Para obtener un diagnóstico completo del sistema

### download_sample_dataset.py
**Escenarios de aplicación:**
- Ausencia de conjunto de datos propio
- Necesidad de crear estructura de directorios
- Pruebas rápidas del sistema
- Requerimiento de orientación para descarga de datos

### quick_test.py
**Escenarios de aplicación:**
- Verificación de funcionamiento general
- Posterior a modificaciones en el código
- Previo al proceso de entrenamiento
- Diagnóstico rápido de problemas

---

## Secuencia de ejecución recomendada

```
1. verify_installation.py
   ↓
   Validación exitosa
   ↓
2. download_sample_dataset.py
   ↓
   Conjunto de datos configurado
   ↓
3. quick_test.py
   ↓
   Pruebas exitosas → Sistema operativo
```

## Consideraciones importantes

- Estos scripts no modifican modelo ni datos de entrenamiento
- Pueden ejecutarse de forma segura en cualquier momento
- Generan información valiosa para diagnóstico
- No requieren GPU ni recursos computacionales intensivos
