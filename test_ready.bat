@echo off
REM ================================================================
REM TEST DE READINESS - PASO 6
REM Verifica que el modelo esta listo para produccion
REM ================================================================

echo.
echo ================================================================
echo   TEST DE READINESS PARA PRODUCCION - PASO 6
echo ================================================================
echo.
echo Tests a ejecutar:
echo   [1] Modelo guarda/carga correctamente
echo   [2] Predicciones en tiempo real funcionan
echo   [3] Latencia ^< 500ms por imagen
echo   [4] Memory footprint ^< 500MB
echo   [5] Inference script listo para usar
echo.
echo ================================================================
echo.

cd /d "%~dp0"

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado o no esta en el PATH
    pause
    exit /b 1
)

echo [1/2] Verificando dependencias...
pip install -q psutil tensorflow

echo.
echo [2/2] Ejecutando tests de readiness...
echo.

REM Ejecutar tests
python backend\scripts\test_ready.py

if errorlevel 1 (
    echo.
    echo ================================================================
    echo ADVERTENCIA: Algunos tests fallaron
    echo ================================================================
    echo.
    echo Revisa el reporte: metrics/readiness_report.json
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo ================================================================
    echo EXITO: Modelo listo para produccion
    echo ================================================================
    echo.
    echo El modelo ha pasado todos los tests de readiness.
    echo.
    echo Archivos disponibles:
    echo   - models/best_model.keras (modelo optimizado)
    echo   - backend/scripts/inference.py (script de inferencia)
    echo   - metrics/readiness_report.json (reporte detallado)
    echo.
    echo Uso de inferencia:
    echo   python backend/scripts/inference.py --image ruta/imagen.jpg
    echo   python backend/scripts/inference.py --batch ruta/carpeta/
    echo   python backend/scripts/inference.py --info
    echo.
    pause
    exit /b 0
)
