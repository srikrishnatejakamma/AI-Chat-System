param([string]$ChatModel = '', [switch]$BuildFrontend)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$pythonPath = Join-Path (Get-Location) '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    $launcherPath = if ($launcher) { $launcher.Source } else { 'C:/Windows/py.exe' }
    if (-not (Test-Path -LiteralPath $launcherPath)) { throw 'Install Python 3.12 first.' }
    & $launcherPath -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Unable to create the Python environment.' }
    & $pythonPath -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Unable to install dependencies. Check your internet connection.' }
}
if ($ChatModel) { $env:LLM_MODEL = $ChatModel }
$env:PYTHONDONTWRITEBYTECODE = '1'
& $pythonPath -c 'import fastapi, uvicorn, multipart'
if ($LASTEXITCODE -ne 0) {
    & $pythonPath -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Unable to install API dependencies.' }
}
if ($BuildFrontend -or -not (Test-Path -LiteralPath 'frontend/dist/index.html')) {
    $npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
    $npmPath = if ($npmCommand) { $npmCommand.Source } else { 'C:/Program Files/nodejs/npm.cmd' }
    if (-not (Test-Path -LiteralPath $npmPath)) { throw 'Install Node.js 22, then rerun this script to build the React interface.' }
    $env:PATH = (Split-Path $npmPath -Parent) + ';' + $env:PATH
    Push-Location frontend
    try {
        if (-not (Test-Path -LiteralPath node_modules)) {
            & $npmPath ci --no-fund --no-audit
            if ($LASTEXITCODE -ne 0) { throw 'Unable to install React dependencies.' }
        }
        & $npmPath run build
        if ($LASTEXITCODE -ne 0) { throw 'Unable to build the React interface.' }
    } finally { Pop-Location }
}
& $pythonPath -m uvicorn api:app --host 127.0.0.1 --port 8501
