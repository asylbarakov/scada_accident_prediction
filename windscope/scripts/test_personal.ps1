$ErrorActionPreference = 'Stop'
$demoWorkspace = Split-Path -Parent $PSScriptRoot
& docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --memory 1g --tmpfs /tmp `
    --mount "type=bind,source=$demoWorkspace/personal_forecast,target=/work/personal_forecast,readonly" `
    --mount "type=bind,source=$demoWorkspace/experiments/personal_h24/personal.py,target=/work/personal.py,readonly" `
    --mount "type=bind,source=$demoWorkspace/artifacts/personal_runtime_registry,target=/registry,readonly" `
    --mount "type=bind,source=$demoWorkspace/project/data,target=/research/data,readonly" `
    --mount "type=bind,source=$demoWorkspace/artifacts/personal_h24_v1,target=/research-output,readonly" `
    --workdir /work --entrypoint python windscope-xgb-h24:1.0 -m unittest discover -s personal_forecast/tests -v
if ($LASTEXITCODE -ne 0) { throw 'Personal inference verification failed' }
