param([switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$referencePath = Join-Path $projectRoot 'reference/SWTFPS_Wind-Turbine-Failure-Prediction'
$commit = 'b0922f9ab22a65b582d8c0d0e810d8e959500e76'
if (-not (Test-Path -LiteralPath $referencePath)) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $referencePath) -Force | Out-Null
    & git clone --depth 1 --branch v1.0.0 'https://github.com/SaraMahyan/SWTFPS_Wind-Turbine-Failure-Prediction.git' $referencePath
    if ($LASTEXITCODE -ne 0) { throw 'Reference clone failed' }
}
$actualCommit = & git -C $referencePath rev-parse HEAD
if ($actualCommit -ne $commit) { throw 'Reference commit differs from the reviewed release' }
$changed = & git -C $referencePath status --porcelain
if ($changed) { throw 'Reference checkout is modified; preserve it and inspect the changes' }
$modelPath = Join-Path $referencePath "Codes/Models' Packeges/prototype_model_best.pkl"
if ((Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne '98584786a6a4421f74e599743378c67aadaeaa67ffbfdd582a14f5d235617fd8') {
    throw 'Model artifact hash mismatch'
}
$manifest = Get-Content -LiteralPath (Join-Path $projectRoot 'data/scenarios_manifest.json') -Raw | ConvertFrom-Json
$rawPath = Join-Path $projectRoot 'data/raw/wind_farm_a_v2'
New-Item -ItemType Directory -Path $rawPath -Force | Out-Null
foreach ($file in $manifest.files) {
    $destination = Join-Path $rawPath $file.local_path
    if (-not (Test-Path -LiteralPath $destination)) {
        $encodedPath = [Uri]::EscapeDataString($file.source_path)
        $uri = 'https://www.kaggle.com/api/v1/datasets/download/azizkasimov/wind-turbine-scada-data-for-early-fault-detection/' + $encodedPath + '?datasetVersionNumber=2'
        Invoke-WebRequest -UseBasicParsing -TimeoutSec 180 -Uri $uri -OutFile $destination
    }
    if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) {
        throw "Source hash mismatch: $($file.local_path). Existing data was not overwritten."
    }
    Write-Output "Verified primary Kaggle v2 file: $($file.local_path)"
}
if (-not $SkipBuild) {
    Push-Location $projectRoot
    try {
        & python scripts/fetch_wheels.py
        if ($LASTEXITCODE -ne 0) { throw 'Pinned wheel download failed' }
        & docker build -f Dockerfile.baseline -t wind-scada-baseline:1.0 .
        if ($LASTEXITCODE -ne 0) { throw 'Baseline environment build failed' }
    } finally { Pop-Location }
}
Write-Output 'Reference and scenario provenance verified. No model has been loaded by setup.'
