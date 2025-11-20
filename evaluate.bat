@echo off
REM ================================================================
REM EVALUACION COMPLETA DEL MODELO - PASOS 3, 4 y 5
REM Pipeline: Evaluacion -> Validacion -> Analisis (si necesario)
REM ================================================================

echo.
echo ================================================================
echo   PIPELINE DE EVALUACION COMPLETO
echo ================================================================
echo.
echo Este script ejecuta:
echo   [1] Paso 3: Evaluacion del modelo (metricas + Excel)
echo   [2] Paso 4: Validacion contra requisitos
echo   [3] Paso 5: Analisis de problemas (si falla validacion)
echo.
echo Outputs generados:
echo   - metrics/evaluation_results.json
echo   - metrics/evaluation_results.xlsx
echo   - metrics/validation_report.json
echo   - metrics/failure_analysis.json (si falla)
echo   - models/visualizations/*.png
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
pip install -q pandas openpyxl scikit-learn matplotlib seaborn

echo.
echo [2/2] Ejecutando evaluacion completa...
echo.

REM Ejecutar evaluacion (incluye validacion y analisis automatico)
python backend\scripts\evaluate_model.py

if errorlevel 1 (
    echo.
    echo ================================================================
    echo ADVERTENCIA: El modelo no cumplio los requisitos obligatorios
    echo ================================================================
    echo.
    echo Revisa los reportes generados:
    echo   1. metrics/validation_report.json - Estado de validacion
    echo   2. metrics/failure_analysis.json - Analisis de problemas
    echo   3. metrics/evaluation_results.xlsx - Metricas detalladas
    echo.
    echo Sigue las recomendaciones prioritarias del analisis.
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo ================================================================
    echo EXITO: Modelo APROBADO y listo para produccion
    echo ================================================================
    echo.
    echo Proximos pasos:
    echo   1. Revisar metrics/evaluation_results.xlsx
    echo   2. Documentar configuracion final
    echo   3. Preparar deployment (ejecutar: test_ready.bat)
    echo.
    pause
    exit /b 0
)
