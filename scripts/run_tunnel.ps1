# Expone el aplicativo Streamlit por Cloudflare Tunnel (§4.6).
#
# Levanta Streamlit en segundo plano y abre un tunel rapido de Cloudflare, que entrega
# una URL publica https://<algo>.trycloudflare.com sin necesidad de cuenta ni dominio.
#
# Para la sustentacion: correr esto, esperar la URL en la salida de cloudflared y
# compartirla. El tunel rapido no requiere login, pero la URL cambia en cada ejecucion,
# asi que hay que generarla el mismo dia y no antes.
#
# Requisito: cloudflared en el PATH.
#   winget install --id Cloudflare.cloudflared

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$port = 8501

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Host "cloudflared no esta en el PATH." -ForegroundColor Red
    Write-Host "Instalalo con: winget install --id Cloudflare.cloudflared"
    exit 1
}

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "No existe .venv. Corre primero .\scripts\setup_env.ps1" -ForegroundColor Red
    exit 1
}

Write-Host "Levantando Streamlit en el puerto $port ..." -ForegroundColor Cyan
# headless=true evita que Streamlit intente abrir el navegador y pida el correo de
# telemetria la primera vez, que en una demo en vivo bloquea el arranque.
$streamlit = Start-Process -FilePath $python `
    -ArgumentList "-m", "streamlit", "run", "app/streamlit_app.py", `
                  "--server.port", $port, "--server.headless", "true" `
    -PassThru

try {
    Start-Sleep -Seconds 6
    if ($streamlit.HasExited) {
        Write-Host "Streamlit termino inesperadamente. Corre el comando a mano para ver el error." -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host "Abriendo el tunel. La URL publica aparece abajo como trycloudflare.com" -ForegroundColor Green
    Write-Host "Ctrl+C cierra el tunel y detiene Streamlit." -ForegroundColor Yellow
    Write-Host ""
    & cloudflared tunnel --url "http://localhost:$port"
}
finally {
    if (-not $streamlit.HasExited) {
        Write-Host "`nDeteniendo Streamlit ..." -ForegroundColor Cyan
        Stop-Process -Id $streamlit.Id -Force -ErrorAction SilentlyContinue
    }
}
