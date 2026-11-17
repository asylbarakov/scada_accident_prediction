param([string]$OutputDirectory = 'artifacts/h24_loto_v1')
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$outputPath = [IO.Path]::GetFullPath((Join-Path $workspace $OutputDirectory))
if (-not $outputPath.StartsWith($workspace + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Output must stay inside this workspace' }
if (Test-Path -LiteralPath (Join-Path $outputPath 'data.parquet')) { throw 'Choose a fresh output directory; results are never overwritten' }
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$common = @('run','--rm','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','2g','--cpus','2','--pids-limit','128','--tmpfs','/tmp',
    '--mount',"type=bind,source=$workspace/project,target=/research,readonly",
    '--mount',"type=bind,source=$workspace/experiments/h24,target=/experiment,readonly",
    '--mount',"type=bind,source=$outputPath,target=/output",'--workdir','/experiment','windscope-h7:1.0')
& docker @common python -m unittest -v
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
foreach ($stage in @('prepare','train')) {
    & docker @common python -u run.py $stage
    if ($LASTEXITCODE -ne 0) { throw "H24 stage failed: $stage" }
}
foreach ($check in @('diagnostics.py','verify.py')) {
    & docker @common python $check
    if ($LASTEXITCODE -ne 0) { throw "H24 verification failed: $check" }
}
$plot = @('run','--rm','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','1g','--cpus','2','--pids-limit','128','--tmpfs','/tmp','-e','MPLCONFIGDIR=/tmp/mpl',
    '--mount',"type=bind,source=$workspace/experiments/h24,target=/experiment,readonly",
    '--mount',"type=bind,source=$outputPath,target=/output",'--entrypoint','python','wind-scada-baseline:1.0','/experiment/plot.py')
& docker @plot
if ($LASTEXITCODE -ne 0) { throw 'Plotting failed' }
Write-Output "H24 artifacts: $outputPath"
