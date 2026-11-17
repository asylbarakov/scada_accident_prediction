param([string]$OutputDirectory='artifacts/personal_relations_v4')
$ErrorActionPreference='Stop'
$workspace=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$outputPath=[IO.Path]::GetFullPath((Join-Path $workspace $OutputDirectory))
if (-not $outputPath.StartsWith($workspace+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) {throw 'Output must stay inside workspace'}
if (Test-Path -LiteralPath (Join-Path $outputPath 'metrics.json')) {throw 'Choose a fresh output directory'}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$common=@('run','--rm','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','2g','--cpus','2','--pids-limit','128','--tmpfs','/tmp',
    '--mount',"type=bind,source=$workspace/experiments/personal_relations,target=/experiment,readonly",
    '--mount',"type=bind,source=$workspace/experiments/personal_h24,target=/personal-code,readonly",
    '--mount',"type=bind,source=$workspace/experiments/personal_extended,target=/extended-code,readonly",
    '--mount',"type=bind,source=$workspace/experiments/h24,target=/feature-code,readonly",
    '--mount',"type=bind,source=$workspace/experiments/nbm_h24,target=/evaluation,readonly",
    '--mount',"type=bind,source=$workspace/artifacts/personal_h24_v1,target=/baseline,readonly",
    '--mount',"type=bind,source=$workspace/project,target=/research,readonly",
    '--mount',"type=bind,source=$outputPath,target=/output",'--workdir','/experiment','--entrypoint','python','windscope-xgb-h24:1.0')
& docker @common -m unittest -v
if ($LASTEXITCODE -ne 0) {throw 'Physical relation tests failed'}
& docker @common -u run.py
if ($LASTEXITCODE -ne 0) {throw 'Physical relation experiment failed'}
& docker @common -u run.py verify
if ($LASTEXITCODE -ne 0) {throw 'Native relation replay verification failed'}
& docker @common diagnose.py
if ($LASTEXITCODE -ne 0) {throw 'Relation attribution failed'}
$plot=@('run','--rm','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','512m','--tmpfs','/tmp','-e','MPLCONFIGDIR=/tmp/mpl',
    '--mount',"type=bind,source=$workspace/experiments/personal_relations,target=/experiment,readonly",
    '--mount',"type=bind,source=$outputPath,target=/output",'--entrypoint','python','wind-scada-baseline:1.0','/experiment/plot.py')
& docker @plot
if ($LASTEXITCODE -ne 0) {throw 'Relation comparison figure failed'}
