param([int]$Activate=3,[int]$Resolve=3,[int]$Cooldown=60,[string]$OutputDirectory='artifacts/v2')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$outputPath=[IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))
if (-not $outputPath.StartsWith($projectRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) {throw 'Output must stay inside the workspace'}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$raw="type=bind,source=$(Join-Path $projectRoot 'data/raw/wind_farm_a_v2'),target=/input,readonly"
$predictions="type=bind,source=$(Join-Path $projectRoot 'artifacts/ml_baseline'),target=/predictions,readonly"
$code="type=bind,source=$(Join-Path $projectRoot 'backend'),target=/work/backend,readonly"
$output="type=bind,source=$outputPath,target=/output"
& docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp --mount $raw --mount $predictions --mount $code --mount $output windscope-backend:1.0 python -m app.evaluate_alerts --activate $Activate --resolve $Resolve --cooldown $Cooldown
if ($LASTEXITCODE -ne 0) {throw 'Alert policy evaluation failed'}
