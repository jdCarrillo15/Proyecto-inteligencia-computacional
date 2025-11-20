@echo off
echo.
echo ============================================
echo   PREPARACION SEGURA DE DATASET
echo ============================================
echo.
echo Este script usa limites conservadores para
echo evitar que el sistema se quede sin memoria.
echo.
echo Configuracion:
echo - Max 200 imagenes por clase
echo - Sin balanceo/oversampling  
echo - Total: ~3,000 imagenes max
echo - RAM estimada: ~2 GB
echo.
echo Asegurate de tener al menos 4 GB RAM libres.
echo Cierra navegadores y otras aplicaciones.
echo.
pause

echo.
echo Limpiando cache viejo...
if exist "backend\cache\*.pkl" (
    del /Q "backend\cache\*.pkl"
    echo Cache eliminado
)

echo.
echo Iniciando preparacion de dataset...
echo.
python backend/scripts/prepare_dataset.py

echo.
echo ============================================
echo   PREPARACION COMPLETADA
echo ============================================
echo.
echo Si termino correctamente, ahora ejecuta:
echo   python backend/scripts/train.py
echo.
pause
