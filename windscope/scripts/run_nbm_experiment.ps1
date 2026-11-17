param([string]$OutputDirectory = 'artifacts/nbm_h24_reproduction_v1')
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$outputPath = [IO.Path]::GetFullPath((Join-Path $workspace $OutputDirectory))
if (-not $outputPath.StartsWith($workspace + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Output must stay inside workspace' }
if ((Test-Path -LiteralPath (Join-Path $outputPath 'metrics.json')) -or (Get-ChildItem -LiteralPath $outputPath -Filter 'fold_*' -ErrorAction SilentlyContinue)) { throw 'Choose fresh output directory' }
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$common = @('run','--rm','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','2g','--cpus','2','--pids-limit','128','--tmpfs','/tmp',
    '--mount',"type=bind,source=$workspace/project,target=/research,readonly",
    '--mount',"type=bind,source=$workspace/artifacts/h24_reproduction_v1,target=/previous,readonly",
    '--mount',"type=bind,source=$workspace/experiments/h24,target=/previous-code,readonly",
    '--mount',"type=bind,source=$workspace/experiments/nbm_h24,target=/experiment,readonly",
    '--mount',"type=bind,source=$outputPath,target=/output",'--workdir','/experiment','windscope-h7:1.0')
& docker @common python -m unittest -v
if ($LASTEXITCODE -ne 0) { throw 'NBM tests failed' }
foreach ($script in @('diagnose_h24.py','run.py','analyze_residuals.py','verify.py')) {
    & docker @common python -u $script
    if ($LASTEXITCODE -ne 0) { throw "NBM stage failed: $script" }
}
$plot = @('run','--rm','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','1g','--cpus','2','--pids-limit','128','--tmpfs','/tmp','-e','MPLCONFIGDIR=/tmp/mpl',
    '--mount',"type=bind,source=$workspace/experiments/nbm_h24,target=/experiment,readonly",
    '--mount',"type=bind,source=$outputPath,target=/output",'--entrypoint','python','wind-scada-baseline:1.0','/experiment/plot.py')
& docker @plot
if ($LASTEXITCODE -ne 0) { throw 'NBM plotting failed' }
Write-Output "NBM evidence: $outputPath"
