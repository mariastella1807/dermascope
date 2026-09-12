# Crea el entorno virtual del proyecto.
#
# PyTorch aun no publica ruedas estables para Python 3.14, que es el interprete por
# defecto de esta maquina. El entorno se fija en 3.12 (o 3.11 si 3.12 no esta instalado).
# Si ninguno esta disponible, instalar desde https://www.python.org/downloads/

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$interpreter = $null
foreach ($version in @("3.12", "3.11")) {
    & py "-$version" --version 2>$null
    if ($?) { $interpreter = $version; break }
}

if (-not $interpreter) {
    Write-Host "No se encontro Python 3.11 ni 3.12." -ForegroundColor Red
    Write-Host "Instalalos con: winget install Python.Python.3.12"
    exit 1
}

Write-Host "Usando Python $interpreter" -ForegroundColor Green

if (-not (Test-Path ".venv")) {
    & py "-$interpreter" -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Entorno listo. Activalo con:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Verificacion:" -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -c "import torch, torchvision, streamlit; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
