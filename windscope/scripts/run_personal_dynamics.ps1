param([string]$OutputDirectory='artifacts/personal_dynamics_v2')
$ErrorActionPreference='Stop'
$workspace=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$outputPath=[IO.Path]::GetFullPath((Join-Path $workspace $OutputDirectory))
if (-not $outputPath.StartsWith($workspace+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Output must stay inside workspace' }
if (Test-Path -LiteralPath (Join-Path $outputPath 'data_gate.json')) { throw 'Choose a fresh output directory' }
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$common=@('run','--rm','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','2g','--cpus','2','--pids-limit','128','--tmpfs','/tmp',
    '--mount',"type=bind,source=$workspace/experiments/personal_dynamics,target=/experiment,readonly",
    '--mount',"type=bind,source=$workspace/experiments/personal_h24,target=/personal-code,readonly",
    '--mount',"type=bind,source=$workspace/experiments/h24,target=/feature-code,readonly",
    '--mount',"type=bind,source=$workspace/experiments/nbm_h24,target=/evaluation,readonly",
    '--mount',"type=bind,source=$workspace/artifacts/personal_h24_v1,target=/baseline,readonly",
    '--mount',"type=bind,source=$workspace/artifacts/h24_reproduction_v1,target=/previous,readonly",
    '--mount',"type=bind,source=$workspace/project,target=/research,readonly",
    '--mount',"type=bind,source=$outputPath,target=/output",'--workdir','/experiment','--entrypoint','python','windscope-xgb-h24:1.0')
& docker @common -m unittest -v
if ($LASTEXITCODE -ne 0) { throw 'Causal feature tests failed' }
& docker @common audit.py
if ($LASTEXITCODE -ne 0) { throw 'Baseline error audit failed' }
& docker @common mapping_audit.py
if ($LASTEXITCODE -ne 0) { throw 'Sensor mapping audit failed' }
foreach ($stage in @('prepare','train','verify')) {
    & docker @common -u run.py $stage
    if ($LASTEXITCODE -ne 0) { throw "Residual dynamics stage failed: $stage" }
}
& docker @common summarize.py
if ($LASTEXITCODE -ne 0) { throw 'Common exposure comparison failed' }
$plot=@('run','--rm','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','512m','--tmpfs','/tmp','-e','MPLCONFIGDIR=/tmp/mpl',
    '--mount',"type=bind,source=$workspace/experiments/personal_dynamics,target=/experiment,readonly",
    '--mount',"type=bind,source=$outputPath,target=/output",'--entrypoint','python','wind-scada-baseline:1.0','/experiment/plot.py')
& docker @plot
if ($LASTEXITCODE -ne 0) { throw 'Comparison figure failed' }
Write-Output "Residual dynamics evidence: $outputPath"
