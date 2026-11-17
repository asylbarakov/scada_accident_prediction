param(
    [ValidateSet('Test','Infer','Evaluate','Freeze')][string]$Stage = 'Infer',
    [ValidateSet('all','train','prediction')][string]$Partition = 'all',
    [int]$Limit = 0,
    [string]$OutputDirectory = 'artifacts/ml_baseline'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$outputPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))
if (-not $outputPath.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'OutputDirectory must remain inside this project'
}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$rawPath = Join-Path $projectRoot 'data/raw/wind_farm_a_v2'
$modelPath = Join-Path $projectRoot "reference/SWTFPS_Wind-Turbine-Failure-Prediction/Codes/Models' Packeges/prototype_model_best.pkl"
$containerArgs = @('create','--network','none','--read-only','--cap-drop','ALL',
    '--security-opt','no-new-privileges','--user','10001:10001',
    '--cpus','2','--memory','2g','--pids-limit','128',
    '--tmpfs','/tmp:rw,noexec,nosuid,size=268435456',
    '--mount',"type=bind,source=$(Join-Path $projectRoot 'baseline'),target=/work/baseline,readonly",
    '--mount',"type=bind,source=$outputPath,target=/output")
switch ($Stage) {
    'Test' {
        $containerArgs += @('--mount',"type=bind,source=$(Join-Path $projectRoot 'tests'),target=/work/tests,readonly")
        $containerArgs += @('--mount',"type=bind,source=$modelPath,target=/model/baseline.pkl,readonly")
        $containerArgs += @('--env','BASELINE_TEST_MODEL=/model/baseline.pkl')
        $command = @('-m','unittest','discover','-s','tests','-v')
    }
    'Infer' {
        foreach ($id in @('0','25')) {
            $containerArgs += @('--mount',"type=bind,source=$(Join-Path $rawPath ($id+'.csv')),target=/input/$id.csv,readonly")
        }
        $containerArgs += @('--mount',"type=bind,source=$modelPath,target=/model/baseline.pkl,readonly")
        $command = @('-m','baseline.run','--model','/model/baseline.pkl','--input','/input/0.csv',
            '--input','/input/25.csv','--output','/output','--partition',$Partition,'--limit',"$Limit")
    }
    'Evaluate' {
        $containerArgs += @('--mount',"type=bind,source=$(Join-Path $rawPath 'event_info.csv'),target=/annotations/event_info.csv,readonly")
        $command = @('-m','baseline.analyze','--predictions','/output','--annotations','/annotations/event_info.csv')
    }
    'Freeze' {
        $command = @('-m','pip','--disable-pip-version-check','--no-cache-dir','freeze','--all')
    }
}
$containerId = & docker @containerArgs 'wind-scada-baseline:1.0' @command
if ($LASTEXITCODE -ne 0) { throw 'Container creation failed' }
$containerId = "$containerId".Trim()
try {
    $inspection = & docker inspect $containerId | ConvertFrom-Json
    $state = $inspection[0]
    $proof = [ordered]@{
        stage = $Stage
        image_id = $state.Image
        network_mode = $state.HostConfig.NetworkMode
        read_only_root = $state.HostConfig.ReadonlyRootfs
        user = $state.Config.User
        cap_drop = $state.HostConfig.CapDrop
        security_options = $state.HostConfig.SecurityOpt
        memory_bytes = $state.HostConfig.Memory
        nano_cpus = $state.HostConfig.NanoCpus
        mounts = @($state.Mounts | Select-Object Source,Destination,RW)
    }
    $proof | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $outputPath ("containment_$Stage.json")) -Encoding UTF8
    if ($Stage -eq 'Freeze') {
        & docker start -a $containerId | Set-Content -LiteralPath (Join-Path $outputPath 'environment.freeze.txt') -Encoding UTF8
    } else {
        & docker start -a $containerId
    }
    if ($LASTEXITCODE -ne 0) { throw "Container stage $Stage failed" }
    $exitCode = & docker inspect --format '{{.State.ExitCode}}' $containerId
    if ($exitCode -ne '0') { throw "Container exited with code $exitCode" }
} finally {
    & docker rm $containerId | Out-Null
}
