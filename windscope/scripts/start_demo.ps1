param([switch]$SkipPrepare)
$ErrorActionPreference='Stop'
$demoWorkspace=Split-Path -Parent $PSScriptRoot
Push-Location $demoWorkspace
try {
    if (-not $SkipPrepare) {
        & python scripts/prepare_demo.py
        if ($LASTEXITCODE -ne 0) { throw 'Demo source preparation failed' }
        & "$PSScriptRoot/prepare_baseline.ps1" -SkipBuild
        if ($LASTEXITCODE -ne 0) { throw 'Reference baseline verification failed' }
    }
    if (-not (Test-Path '.wheelhouse/xgboost_cpu-3.1.2-py3-none-manylinux_2_28_x86_64.whl')) {
        & python scripts/fetch_wheels.py
        if ($LASTEXITCODE -ne 0) { throw 'Pinned wheel download failed' }
    }
    & docker image inspect wind-scada-baseline:1.0 *> $null
    if ($LASTEXITCODE -ne 0) {
        & docker build -f Dockerfile.baseline -t wind-scada-baseline:1.0 .
        if ($LASTEXITCODE -ne 0) { throw 'Baseline image build failed' }
    }
    & docker image inspect windscope-h7:1.0 *> $null
    if ($LASTEXITCODE -ne 0) {
        & docker build -f forecast/Dockerfile -t windscope-h7:1.0 .
        if ($LASTEXITCODE -ne 0) { throw 'Feature runtime build failed' }
    }
    & docker image inspect windscope-xgb-h24:1.0 *> $null
    if ($LASTEXITCODE -ne 0) {
        & docker build --network none -f experiments/xgb_h24/Dockerfile -t windscope-xgb-h24:1.0 .
        if ($LASTEXITCODE -ne 0) { throw 'Personal runtime build failed' }
    }
    if (-not (Test-Path 'artifacts/personal_runtime_registry/registry.json')) { throw 'Reviewed native model registry is missing' }
    & docker compose up -d --build
    if ($LASTEXITCODE -ne 0) { throw 'Demo startup failed' }
    Write-Output 'WindScope demo: http://localhost:3100 | Advanced monitoring: http://localhost:3100/monitoring'
} finally { Pop-Location }
