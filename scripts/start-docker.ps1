param(
    [string]$ChatModel = "phi3"
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)

$docker = Get-Command docker -ErrorAction SilentlyContinue
if (-not $docker) {
    $candidates = @(
        'C:/Program Files/Rancher Desktop/resources/resources/win32/bin/docker.exe',
        'C:/Program Files/Docker/Docker/resources/bin/docker.exe'
    )
    $dockerPath = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $dockerPath) { throw 'Install and start Docker Desktop or Rancher Desktop first.' }
    $env:PATH = (Split-Path $dockerPath -Parent) + ';' + $env:PATH
} else {
    $dockerPath = $docker.Source
}

function Invoke-Docker {
    & $dockerPath @args
    if ($LASTEXITCODE -ne 0) { throw "Docker command failed. Check that Docker is running, then retry." }
}

# Explicitly use a smaller model by default for an 8 GB Docker VM.
$env:LLM_MODEL = $ChatModel
Invoke-Docker compose up --build -d
Invoke-Docker compose exec -T ollama ollama pull nomic-embed-text
Invoke-Docker compose exec -T ollama ollama pull $ChatModel
Write-Host 'Ready: http://localhost:8501' -ForegroundColor Green
Write-Host 'Refresh the connection in the sidebar if models still appear unavailable.'
