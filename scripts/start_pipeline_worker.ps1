[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$dataDirectory = Join-Path $projectRoot ".data"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python virtual environment was not found: $python"
}
New-Item -ItemType Directory -Path $dataDirectory -Force | Out-Null
Set-Location -LiteralPath $projectRoot

$configPath = Join-Path $projectRoot "config\cameras.toml"
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    $configPath = Join-Path $projectRoot "config\cameras.example.toml"
}
$nasRoot = $env:CCTV_NAS_ROOT
if ([string]::IsNullOrWhiteSpace($nasRoot)) {
    $nasRoot = (& $python -c "import sys, tomllib; c = tomllib.loads(open(sys.argv[1], 'rb').read().decode('utf-8')); print(c.get('project', {}).get('source_root_windows') or '')" $configPath).Trim()
}
if ([string]::IsNullOrWhiteSpace($nasRoot)) {
    $nasRoot = "\\192.168.1.100\cctv"
}
$cameraIds = @(& $python -c "import sys, tomllib; c = tomllib.loads(open(sys.argv[1], 'rb').read().decode('utf-8')); print('\n'.join(cam['id'] for cam in c.get('cameras', []) if cam.get('id')))" $configPath | Where-Object { $_ -and $_.Trim() })

$arguments = @(
    "-m", "cctv_ai.cli", "pipeline-worker",
    "--db", ".data\cctv_index.sqlite3",
    "--config", $configPath,
    "--root", $nasRoot,
    "--since", "2026-07-22T00:00:00+08:00"
)
foreach ($camId in $cameraIds) {
    $arguments += @("--camera", $camId.Trim())
}
$arguments += @(
    "--probe-limit-per-camera", "128",
    "--probe-workers", "16",
    "--audio-limit-per-camera", "128",
    "--audio-workers", "16",
    "--watermark-limit-per-camera", "8",
    "--watermark-workers", "2",
    "--watermark-event-label", "baby_cry",
    "--watermark-hwaccel", "cuda",
    # Long bedside recordings can exceed 60 seconds even while decoding
    # normally from the NAS. Keep the detector busy instead of discarding
    # completed inference work and repeating it on the next retry.
    "--audio-timeout-seconds", "180",
    "--stale-job-seconds", "900",
    "--error-retry-delay-seconds", "21600",
    "--interval-seconds", "1",
    "--scan-every-cycles", "120",
    "--mode", "all-camera-incremental"
)

$stdout = Join-Path $dataDirectory "pipeline-worker.out.log"
$stderr = Join-Path $dataDirectory "pipeline-worker.err.log"
$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru
try {
    $process.PriorityClass = "High"
    $priorityDeadline = (Get-Date).AddSeconds(10)
    do {
        $child = Get-CimInstance Win32_Process |
            Where-Object {
                $_.ParentProcessId -eq $process.Id -and
                $_.Name -eq "python.exe"
            } |
            Select-Object -First 1
        if ($null -ne $child) {
            (Get-Process -Id $child.ProcessId).PriorityClass = "High"
            break
        }
        Start-Sleep -Milliseconds 100
    } while ((Get-Date) -lt $priorityDeadline)
} catch {
    # Concurrency remains effective if Windows denies priority adjustment.
}
$process.WaitForExit()
exit $process.ExitCode
