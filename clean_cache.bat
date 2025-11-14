@echo off
REM ================================================================
REM Script para limpiar cache PKL y modelos antiguos
REM Usar SOLO cuando cambies IMG_SIZE o haya problemas de cache
REM ================================================================

echo.
echo ================================================================
echo     LIMPIEZA DE CACHE Y MODELOS INCOMPATIBLES
echo ================================================================
echo.
echo Este script eliminara:
echo   - backend/cache/*.pkl (datos cacheados con resolucion antigua)
echo   - backend/cache/*.json (metadatos)
echo   - models/*.keras (modelos entrenados con resolucion antigua)
echo.
echo SOLO ejecuta esto si:
echo   1. Cambiaste IMG_SIZE en config.py
echo   2. Hay errores de shape mismatch
echo   3. Quieres re-entrenar desde cero
echo.

pause

echo.
echo [1/3] Verificando procesos Python...
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo.
    echo ADVERTENCIA: Python esta ejecutandose
    echo Cierra todos los procesos Python antes de continuar
    echo.
    pause
)

echo.
echo [2/3] Limpiando cache PKL...
cd backend\cache
if exist *.pkl (
    del /F /Q *.pkl
    echo OK - Archivos PKL eliminados
) else (
    echo INFO - No hay archivos PKL
)

if exist *.json (
    del /F /Q *.json
    echo OK - Archivos JSON eliminados
) else (
    echo INFO - No hay archivos JSON
)
cd ..\..

echo.
echo [3/3] Limpiando modelos antiguos...
if exist models\*.keras (
    del /F /Q models\*.keras
    echo OK - Modelos Keras eliminados
) else (
    echo INFO - No hay modelos Keras antiguos
)

if exist models\*.json (
    del /F /Q models\*.json
    echo OK - Archivos JSON de modelos eliminados
) else (
    echo INFO - No hay archivos JSON de modelos
)

echo.
echo ================================================================
echo     LIMPIEZA COMPLETADA
echo ================================================================
echo.
echo Proximos pasos:
echo   1. Verifica que IMG_SIZE = (224, 224) en backend/config.py
echo   2. Ejecuta: python backend/scripts/train.py
echo   3. El sistema regenerara automaticamente el cache
echo.
echo Tiempo estimado primera vez (con 224x224):
echo   - Generacion de cache: 15-25 min
echo   - Entrenamiento completo: 1.5-2 horas
echo.

pause
