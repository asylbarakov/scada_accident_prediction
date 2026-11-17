$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$auditOutput=Join-Path $projectRoot 'artifacts/model_audit'
New-Item -ItemType Directory -Path $auditOutput -Force | Out-Null
Push-Location $projectRoot
try {
    $researchMount="type=bind,source=$(Join-Path $projectRoot 'project'),target=/research,readonly"
    $auditMount="type=bind,source=$(Join-Path $projectRoot 'audit'),target=/audit,readonly"
    $outputMount="type=bind,source=$auditOutput,target=/audit-output"
    & docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 --memory 1g --cpus 2 --tmpfs /tmp --mount $researchMount --mount $auditMount --mount $outputMount windscope-h7:1.0 python /audit/h7_predictive_audit.py
    if ($LASTEXITCODE -ne 0) {throw 'Predictive audit failed'}
    & docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 --memory 1g --cpus 2 --tmpfs /tmp --mount $researchMount --mount $auditMount --mount $outputMount windscope-h7:1.0 python /audit/check_readiness.py
    if ($LASTEXITCODE -ne 0) {throw 'Readiness audit failed'}
    & docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp -e MPLCONFIGDIR=/tmp/mpl --entrypoint python --mount $auditMount --mount $outputMount wind-scada-baseline:1.0 /audit/plot_h7_audit.py
    if ($LASTEXITCODE -ne 0) {throw 'Audit figure generation failed'}
} finally {Pop-Location}
