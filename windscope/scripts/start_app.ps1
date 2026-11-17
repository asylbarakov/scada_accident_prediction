$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    & docker image inspect wind-scada-baseline:1.0 *> $null
    if ($LASTEXITCODE -ne 0) {
        & "$PSScriptRoot/prepare_baseline.ps1"
    } else {
        & "$PSScriptRoot/prepare_baseline.ps1" -SkipBuild
    }
    & docker compose up -d --build
    if ($LASTEXITCODE -ne 0) {throw 'Application startup failed'}
    Write-Output 'WindScope: http://localhost:3100 — API docs: http://localhost:8100/docs'
} finally {Pop-Location}
