param([string]$OutputDirectory='artifacts/personal_extended_v3')
$ErrorActionPreference='Stop'
$workspace=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$outputPath=[IO.Path]::GetFullPath((Join-Path $workspace $OutputDirectory))
if (-not $outputPath.StartsWith($workspace+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) {throw 'Output must stay inside workspace'}
if (Test-Path -LiteralPath (Join-Path $outputPath 'metrics.json')) {throw 'Choose a fresh output directory'}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$common=@('run','--rm','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','2g','--cpus','2','--pids-limit','128','--tmpfs','/tmp',
    '--mount',"type=bind,source=$workspace/experiments/personal_extended,target=/experiment,readonly",
    '--mount',"type=bind,source=$workspace/experiments/personal_h24,target=/personal-code,readonly",
    '--mount',"type=bind,source=$workspace/experiments/nbm_h24,target=/evaluation,readonly",
    '--mount',"type=bind,source=$workspace/artifacts/personal_h24_v1,target=/baseline,readonly",
    '--mount',"type=bind,source=$workspace/project,target=/research,readonly",
    '--mount',"type=bind,source=$outputPath,target=/output",'--workdir','/experiment','--entrypoint','python','windscope-xgb-h24:1.0')
& docker @common -m unittest -v
if ($LASTEXITCODE -ne 0) {throw 'Extended sensor tests failed'}
& docker @common -u extended.py
if ($LASTEXITCODE -ne 0) {throw 'Extended sensor experiment failed'}
