param([string]$OutputDirectory = 'artifacts/personal_h24_reproduction_v1')
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$outputPath = [IO.Path]::GetFullPath((Join-Path $workspace $OutputDirectory))
if (-not $outputPath.StartsWith($workspace + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Output must stay inside workspace' }
if (Test-Path -LiteralPath (Join-Path $outputPath 'data_gate.json')) { throw 'Choose a fresh output directory' }
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$common = @('run','--rm','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','2g','--cpus','2','--pids-limit','128','--tmpfs','/tmp',
    '--mount',"type=bind,source=$workspace/project,target=/research,readonly",
    '--mount',"type=bind,source=$workspace/artifacts/h24_reproduction_v1,target=/previous,readonly",
    '--mount',"type=bind,source=$workspace/experiments/h24,target=/feature-code,readonly",
    '--mount',"type=bind,source=$workspace/experiments/nbm_h24,target=/evaluation,readonly",
    '--mount',"type=bind,source=$workspace/experiments/personal_h24,target=/experiment,readonly",
    '--mount',"type=bind,source=$outputPath,target=/output",'--workdir','/experiment','windscope-xgb-h24:1.0')
& docker @common python -m unittest -v
if ($LASTEXITCODE -ne 0) { throw 'Personal model tests failed' }
foreach ($stage in @('prepare','train')) {
    & docker @common python -u run.py $stage
    if ($LASTEXITCODE -ne 0) { throw "Personal-model stage failed: $stage" }
}
& docker @common python verify.py
if ($LASTEXITCODE -ne 0) { throw 'Personal-model verification failed' }
& docker @common python diagnose.py
if ($LASTEXITCODE -ne 0) { throw 'Personal-model diagnostics failed' }
$plot = @('run','--rm','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','1g','--cpus','2','--pids-limit','128','--tmpfs','/tmp','-e','MPLCONFIGDIR=/tmp/mpl',
    '--mount',"type=bind,source=$workspace/experiments/personal_h24,target=/experiment,readonly",
    '--mount',"type=bind,source=$outputPath,target=/output",'--entrypoint','python','wind-scada-baseline:1.0','/experiment/plot.py')
& docker @plot
if ($LASTEXITCODE -ne 0) { throw 'Personal-model plotting failed' }
Write-Output "Personal-model evidence: $outputPath"
