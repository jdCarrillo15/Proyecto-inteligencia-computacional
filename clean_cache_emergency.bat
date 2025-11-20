@echo off
echo.
echo ============================================
echo   LIMPIEZA DE EMERGENCIA - Cache Dataset
echo ============================================
echo.
echo Este script borrara el cache PKL que esta
echo causando problemas de memoria.
echo.
pause

echo.
echo Borrando cache...
if exist "backend\cache\*.pkl" (
    del /Q "backend\cache\*.pkl"
    echo ✅ Cache PKL eliminado
) else (
    echo ⚠️  No se encontro cache PKL
)

if exist "backend\cache\cache_metadata.json" (
    del /Q "backend\cache\cache_metadata.json"
    echo ✅ Metadata eliminado
)

echo.
echo ============================================
echo   LIMPIEZA COMPLETADA
echo ============================================
echo.
echo Ahora ejecuta estos comandos en orden:
echo.
echo 1. python backend/scripts/prepare_dataset.py
echo    (Selecciona opcion 2: Forzar reprocesamiento)
echo.
echo 2. python backend/scripts/train.py
echo.
pause
