param([string]$ChatModel = '')
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
& $pythonPath -m streamlit run app.py
