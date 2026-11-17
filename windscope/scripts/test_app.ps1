$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$evidence=Join-Path $projectRoot 'artifacts/full_stack'
New-Item -ItemType Directory -Path $evidence -Force | Out-Null
Push-Location $projectRoot
try {
    & docker compose exec -T api python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) {throw 'Backend tests failed'}
    $frontendMount="type=bind,source=$(Join-Path $projectRoot 'frontend'),target=/app,readonly"
    & docker run --rm --network none --read-only --cap-drop ALL --mount $frontendMount -w /app node:22-alpine node --test tests/sensor-selection.test.mjs tests/alert-view.test.mjs tests/forecast-view.test.mjs tests/bootstrap-recovery.test.mjs tests/monitor-status.test.mjs
    if ($LASTEXITCODE -ne 0) {throw 'Frontend behavior tests failed'}
    $checkMount="type=bind,source=$(Join-Path $projectRoot 'backend/tests'),target=/checks,readonly"
    $evidenceMount="type=bind,source=$evidence,target=/evidence"
    $baselineMount="type=bind,source=$(Join-Path $projectRoot 'artifacts/ml_baseline'),target=/baseline_predictions,readonly"
    & docker run --rm --network windscope_edge --read-only --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp --mount $checkMount --mount $evidenceMount --mount $baselineMount windscope-backend:1.0 python /checks/e2e_live.py
    if ($LASTEXITCODE -ne 0) {throw 'End-to-end tests failed'}
    & docker run --rm --network windscope_edge --read-only --cap-drop ALL --tmpfs /tmp --mount $checkMount --mount $evidenceMount windscope-backend:1.0 python /checks/restart_check.py prepare
    if ($LASTEXITCODE -ne 0) {throw 'Restart preparation failed'}
    & docker compose restart worker api
    if ($LASTEXITCODE -ne 0) {throw 'Restart command failed'}
    & docker run --rm --network windscope_edge --read-only --cap-drop ALL --tmpfs /tmp --mount $checkMount --mount $evidenceMount windscope-backend:1.0 python /checks/restart_check.py verify
    if ($LASTEXITCODE -ne 0) {throw 'Restart recovery check failed'}
} finally {Pop-Location}
