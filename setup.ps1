# Instalación de VoxLocal (Windows). Ejecutar una sola vez con conexión a internet:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
# Después la aplicación funciona sin conexión.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\python.exe"

if (-not (Test-Path $py)) {
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        Write-Host "Creando entorno conda Python 3.11 en .venv ..."
        conda create -y -p .venv python=3.11
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        Write-Host "Creando venv con Python 3.11 ..."
        py -3.11 -m venv .venv
        $py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    } else {
        throw "Se necesita conda o Python 3.11 (py launcher). Coqui TTS no soporta Python 3.13+."
    }
}
if (-not (Test-Path $py)) { $py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe" }

& $py -m pip install --upgrade pip
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "GPU NVIDIA detectada: instalando PyTorch con CUDA 12.6"
    & $py -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
} else {
    Write-Host "Sin GPU NVIDIA: instalando PyTorch CPU (se usará cuantización int8)"
    & $py -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
}
& $py -m pip install -r requirements.txt
& $py download_models.py
Write-Host "`nListo. Inicie la app con run.bat"
