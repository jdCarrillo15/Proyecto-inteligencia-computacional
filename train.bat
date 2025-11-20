@echo off
REM ================================================================
REM FASE 2 - ENTRENAMIENTO OPTIMIZADO
REM Pipeline completo: Preparacion -> Entrenamiento -> Evaluacion
REM ================================================================

echo.
echo ================================================================
echo   PIPELINE DE ENTRENAMIENTO - FASE 2
echo ================================================================
echo.
echo Configuracion optimizada (Paso 2):
echo   [x] Adam optimizer (lr=1e-4)
echo   [x] Batch size: 64
echo   [x] Max epochs: 100
echo   [x] Early stopping (patience=15)
echo   [x] ReduceLROnPlateau (factor=0.5, patience=5)
echo   [x] Dual checkpoints (best + last)
echo   [x] Class weights balancing
echo   [x] Cache PKL (70/15/15 split)
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
echo [2/3] Iniciando entrenamiento (Fase 2 - Paso 2)...
echo.
echo Esto puede tomar varias horas dependiendo de tu hardware.
echo El progreso se guardara en:
echo   - models/best_model.keras (mejor modelo)
echo   - models/last_model.keras (ultimo checkpoint)
echo   - metrics/training_history.json (historial completo)
echo.

REM Ejecutar el script de entrenamiento
python backend\scripts\train.py

if errorlevel 1 (
    echo.
    echo ERROR: El entrenamiento fallo. Revisa los logs.
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
