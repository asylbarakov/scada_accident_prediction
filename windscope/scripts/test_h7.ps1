$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$evidence=Join-Path $projectRoot 'artifacts/h7'
New-Item -ItemType Directory -Path $evidence -Force | Out-Null
Push-Location $projectRoot
try {
    $researchMount="type=bind,source=$(Join-Path $projectRoot 'project'),target=/research,readonly"
    $forecastMount="type=bind,source=$(Join-Path $projectRoot 'forecast'),target=/work/forecast,readonly"
    & docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 --memory 768m --cpus 2 --tmpfs /tmp --mount $researchMount --mount $forecastMount windscope-h7:1.0 python -m unittest discover -s forecast/tests -v
    if ($LASTEXITCODE -ne 0) {throw 'H7 artifact or causal inference checks failed'}
    & docker compose exec -T api python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) {throw 'Backend tests failed'}
    $frontendMount="type=bind,source=$(Join-Path $projectRoot 'frontend'),target=/app,readonly"
    & docker run --rm --network none --read-only --cap-drop ALL --mount $frontendMount -w /app node:22-alpine node --test tests/sensor-selection.test.mjs tests/alert-view.test.mjs tests/forecast-view.test.mjs tests/bootstrap-recovery.test.mjs tests/monitor-status.test.mjs
    if ($LASTEXITCODE -ne 0) {throw 'Frontend behavior checks failed'}
    $checkMount="$(Join-Path $projectRoot 'backend/tests'):/checks:ro"
    $evidenceMount="${evidence}:/verification"
    & docker compose run --rm --no-deps -v $checkMount -v $evidenceMount worker python /checks/e2e_h7.py
    if ($LASTEXITCODE -ne 0) {throw 'Live H7 replay checks failed'}
} finally {Pop-Location}
