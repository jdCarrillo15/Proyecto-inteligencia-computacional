@echo off
REM ================================================================
REM ENTRENAMIENTO OPTIMIZADO CON PKL CACHE
REM Este script ejecuta el entrenamiento completo automaticamente
REM ================================================================

echo.
echo ============================================================
echo ENTRENAMIENTO RAPIDO CON CACHE PKL
echo ============================================================
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

REM Ejecutar el script de entrenamiento rapido
python backend\scripts\quick_train.py

if errorlevel 1 (
    echo.
    echo ERROR: El entrenamiento fallo
    pause
    exit /b 1
)

echo.
echo [3/3] Entrenamiento completado exitosamente!
echo.
echo Presiona cualquier tecla para salir...
pause >nul
