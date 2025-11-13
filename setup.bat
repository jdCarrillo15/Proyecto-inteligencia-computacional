@echo off
REM ================================================================
REM SCRIPT DE PRIMER USO - CONFIGURACION INICIAL
REM ================================================================

echo.
echo ============================================================
echo CONFIGURACION INICIAL DEL PROYECTO
echo ============================================================
echo.
echo Este script te ayudara a configurar el proyecto por primera vez
echo.

cd /d "%~dp0"

REM Verificar Python
echo [1/4] Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado
    echo.
    echo Por favor instala Python 3.10 o superior desde:
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version
echo ✓ Python encontrado
echo.

REM Crear entorno virtual (opcional pero recomendado)
echo [2/4] ¿Deseas crear un entorno virtual? (recomendado)
echo Esto aislara las dependencias del proyecto
set /p CREATE_VENV="Crear entorno virtual? (s/n): "

if /i "%CREATE_VENV%"=="s" (
    echo.
    echo Creando entorno virtual...
    python -m venv venv
    
    echo Activando entorno virtual...
    call venv\Scripts\activate.bat
    echo ✓ Entorno virtual creado y activado
) else (
    echo Continuando sin entorno virtual...
)

echo.

REM Instalar dependencias
echo [3/4] Instalando dependencias del backend...
echo Esto puede tomar varios minutos...
echo.

pip install -r backend\requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: No se pudieron instalar las dependencias
    pause
    exit /b 1
)

echo.
echo ✓ Dependencias instaladas correctamente
echo.

REM Verificar dataset
echo [4/4] Verificando dataset...

if exist "dataset\raw\New Plant Diseases Dataset(Augmented)\train" (
    echo ✓ Dataset encontrado en dataset/raw/
) else (
    echo.
    echo ⚠ ADVERTENCIA: Dataset no encontrado
    echo.
    echo El dataset debe estar en:
    echo   dataset/raw/New Plant Diseases Dataset(Augmented)/
    echo.
    echo Por favor descarga el dataset y colócalo en la ubicación correcta
    echo antes de entrenar el modelo.
)

echo.
echo ============================================================
echo CONFIGURACION COMPLETADA
echo ============================================================
echo.
echo Siguiente paso: Entrenar el modelo
echo.
echo Opciones disponibles:
echo   1. Entrenamiento con batch:
echo      ^> train-fast.bat
echo.
echo   2. Manual con Python:
echo      ^> python backend\scripts\train.py
echo.
echo Documentación:
echo   - ENTRENAMIENTO_RAPIDO.md  ^(guía de uso^)
echo   - OPTIMIZACION.md          ^(detalles técnicos^)
echo   - INSTRUCCIONES_PKL.txt    ^(referencia rápida^)
echo.
echo ============================================================
echo.
pause
