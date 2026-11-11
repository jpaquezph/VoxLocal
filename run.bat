@echo off
REM Inicia VoxLocal en http://127.0.0.1:7860 (offline)
cd /d "%~dp0"
set PY=.venv\python.exe
if not exist "%PY%" set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
  echo No se encontro el entorno. Ejecute primero: powershell -ExecutionPolicy Bypass -File setup.ps1
  pause
  exit /b 1
)
"%PY%" -m voxlocal %*
if errorlevel 1 (
  echo.
  echo VoxLocal se cerro con un error. Revise el archivo outputs\voxlocal.log
  pause
)
