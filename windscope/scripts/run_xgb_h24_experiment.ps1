param([string]$OutputDirectory = 'artifacts/xgb_h24_reproduction_v1')
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$outputPath = [IO.Path]::GetFullPath((Join-Path $workspace $OutputDirectory))
if (-not $outputPath.StartsWith($workspace + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Output must stay inside workspace' }
if (Test-Path -LiteralPath (Join-Path $outputPath 'data.parquet')) { throw 'Choose a fresh output directory' }
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$originalPath = Join-Path $workspace "reference/SWTFPS_Wind-Turbine-Failure-Prediction/Codes/Models' Packeges/prototype_model_best.pkl"
$common = @('run','--rm','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','2g','--cpus','2','--pids-limit','128','--tmpfs','/tmp',
    '--mount',"type=bind,source=$workspace/project,target=/research,readonly",
    '--mount',"type=bind,source=$workspace/artifacts/h24_reproduction_v1,target=/previous,readonly",
    '--mount',"type=bind,source=$workspace/experiments/h24,target=/previous-code,readonly",
    '--mount',"type=bind,source=$workspace/experiments/nbm_h24,target=/evaluation,readonly",
    '--mount',"type=bind,source=$workspace/experiments/xgb_h24,target=/experiment,readonly",
    '--mount',"type=bind,source=$originalPath,target=/original/model.pkl,readonly",
    '--mount',"type=bind,source=$outputPath,target=/output",'--workdir','/experiment','windscope-xgb-h24:1.0')
& docker @common python -m unittest -v
if ($LASTEXITCODE -ne 0) { throw 'XGBoost tests failed' }
foreach ($stage in @('prepare','train')) {
    & docker @common python -u run.py $stage
    if ($LASTEXITCODE -ne 0) { throw "XGBoost stage failed: $stage" }
}
foreach ($check in @('verify.py','analyze.py')) {
    & docker @common python $check
    if ($LASTEXITCODE -ne 0) { throw "XGBoost verification failed: $check" }
}
$plot = @('run','--rm','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','1g','--cpus','2','--pids-limit','128','--tmpfs','/tmp','-e','MPLCONFIGDIR=/tmp/mpl',
    '--mount',"type=bind,source=$workspace/experiments/xgb_h24,target=/experiment,readonly",
    '--mount',"type=bind,source=$outputPath,target=/output",'--entrypoint','python','wind-scada-baseline:1.0','/experiment/plot.py')
& docker @plot
if ($LASTEXITCODE -ne 0) { throw 'XGBoost plotting failed' }
Write-Output "XGBoost artifacts: $outputPath"
