@echo off
REM ================================================================
REM ENTRENAMIENTO OPTIMIZADO CON PKL CACHE
REM Este script ejecuta el entrenamiento completo automaticamente
REM ================================================================

echo.
echo ================================================================
echo   ENTRENAMIENTO CON OPTIMIZACIONES v2.0
echo ================================================================
echo.
echo Optimizaciones activas:
echo   [x] Fine-tuning progresivo (3 fases)
echo   [x] Learning rates conservadoras (prevenir forgetting)
echo   [x] Resolucion 224x224 (5x mas detalle)
echo   [x] Metricas detalladas (20+ metricas)
echo   [x] Cache PKL automatico
echo.
echo ================================================================
echo.

cd /d "%~dp0"

REM Verificar que Python esta instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado o no esta en el PATH
    pause
    exit /b 1
)

echo [1/3] Verificando dependencias...
pip install -q -r backend\requirements.txt
if errorlevel 1 (
    echo ERROR: No se pudieron instalar las dependencias
    pause
    exit /b 1
)

echo.
echo [2/3] Iniciando entrenamiento optimizado...
echo.

REM Ejecutar el script de entrenamiento unificado
python backend\scripts\train.py

if errorlevel 1 (
    echo.
    echo ERROR: El entrenamiento fallo
    pause
    exit /b 1
)

echo.
echo ================================================================
echo [3/3] ENTRENAMIENTO COMPLETADO EXITOSAMENTE!
echo ================================================================
echo.
echo Archivos generados:
echo   - models/plant_disease_model.keras (modelo entrenado)
echo   - models/visualizations/confusion_matrix_detailed.png
echo   - models/visualizations/per_class_metrics.png
echo   - models/visualizations/per_crop_performance.png
echo   - models/visualizations/healthy_vs_diseased.png
echo   - models/visualizations/training_report.txt
echo.
echo Para usar el modelo:
echo   1. start-backend.bat  (iniciar API)
echo   2. start-frontend.bat (iniciar interfaz)
echo.
echo Presiona cualquier tecla para salir...
pause >nul
