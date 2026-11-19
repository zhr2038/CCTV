[CmdletBinding()]
param(
    [switch]$SkipScan,
    [ValidateRange(1, 1440)]
    [int]$ScanFreshMinutes = 240
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$dataDirectory = Join-Path $projectRoot ".data"
$chainLog = Join-Path $dataDirectory "visual-chain.log"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python virtual environment was not found: $python"
}
New-Item -ItemType Directory -Path $dataDirectory -Force | Out-Null
Set-Location -LiteralPath $projectRoot

function Write-ChainLog {
    param([Parameter(Mandatory = $true)][string]$Message)
    $timestamp = [DateTimeOffset]::Now.ToString("o")
    Add-Content -LiteralPath $chainLog -Value "$timestamp $Message"
}

function Invoke-LoggedPython {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$StandardOutput,
        [Parameter(Mandatory = $true)][string]$StandardError,
        [Parameter(Mandatory = $true)][string]$Label
    )
    # Windows PowerShell promotes any native stderr line to an ErrorRecord
    # when the script-wide preference is Stop. ML libraries use stderr for
    # harmless progress logs, so capture it without aborting and trust the
    # native process exit code.
    $previousErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $python @Arguments 1> $StandardOutput 2> $StandardError
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorPreference
    }
    if ($null -eq $exitCode) {
        throw "$Label ended without an exit code"
    }
    Write-ChainLog "$Label exited $exitCode"
    return [int]$exitCode
}

$env:PYTHONPATH = Join-Path $projectRoot "src"
$env:HF_HOME = Join-Path $dataDirectory "models\huggingface"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:CUDA_MODULE_LOADING = "LAZY"
try {
    (Get-Process -Id $PID).PriorityClass = "High"
} catch {
    # Child Python processes still run normally if priority cannot be raised.
}

$analysisOffset = [DateTimeOffset]::Now.Offset
$analysisToday = (Get-Date).Date
$analysisSince = [DateTimeOffset]::new(
    $analysisToday.AddDays(-29),
    $analysisOffset
)
$analysisUntil = [DateTimeOffset]::new(
    $analysisToday.AddDays(1),
    $analysisOffset
)

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

$scanLog = Join-Path $dataDirectory "visual-scan.out.log"
$scanErrorLog = Join-Path $dataDirectory "visual-scan.err.log"
$scanArguments = @(
    "-m", "cctv_ai.cli", "scan",
    "--db", ".data\cctv_index.sqlite3",
    "--config", $configPath,
    "--root", $nasRoot,
    "--since", $analysisSince.ToString("o")
)
$recentScan = (
    (Test-Path -LiteralPath $scanLog -PathType Leaf) -and
    ((Get-Item -LiteralPath $scanLog).LastWriteTime -ge
        (Get-Date).AddMinutes(-$ScanFreshMinutes))
)
if ($SkipScan) {
    Write-ChainLog "scan skipped by operator"
} elseif ($recentScan) {
    Write-ChainLog (
        "scan reused; completed within the last " +
        "$ScanFreshMinutes minutes"
    )
} else {
    $scanOutput = & $python @scanArguments 2>&1
    if ($LASTEXITCODE -eq 0) {
        $scanOutput | Out-File -FilePath $scanLog -Append -Encoding utf8
        Write-ChainLog "scan completed"
    } else {
        $scanOutput |
            Out-File -FilePath $scanErrorLog -Append -Encoding utf8
        Write-ChainLog "scan unavailable; continuing from local index"
        # Keep processing the local index when the NAS is unavailable.
    }
}

$arguments = @(
    "-m", "cctv_ai.cli", "visual-worker",
    "--db", ".data\cctv_index.sqlite3",
    "--config", $configPath,
    "--root", $nasRoot,
    "--since", $analysisSince.ToString("o"),
    "--until", $analysisUntil.ToString("o"),
    "--output-root", ".data\visual\keyframes",
    "--detector-model", ".data\models\yolo26s.pt",
    "--embedding-model", "google/siglip2-base-patch16-224",
    "--model-cache", ".data\models\huggingface",
    "--probe-limit", "8192",
    "--probe-workers", "32",
    "--extract-limit", "8192",
    "--extract-workers", "24",
    "--watermark-limit", "512",
    "--watermark-workers", "4",
    "--watermark-interval-seconds", "120",
    "--inference-file-limit", "8192",
    "--inference-frame-limit", "24000",
    "--detector-batch-size", "128",
    "--embedding-batch-size", "96",
    "--bedside-interval-seconds", "30",
    "--kitchen-interval-seconds", "20",
    "--error-retry-delay-seconds", "3600",
    "--interval-seconds", "1",
    # A recoverable watermark error must not hold the Qwen stages for an hour.
    # Interrupted jobs are re-queued at the start of the next visual cycle.
    "--max-cycles", "12"
)

$stdout = Join-Path $dataDirectory "visual-worker.out.log"
$stderr = Join-Path $dataDirectory "visual-worker.err.log"
$workerExitCode = Invoke-LoggedPython `
    -Arguments $arguments `
    -StandardOutput $stdout `
    -StandardError $stderr `
    -Label "visual worker"
if ($workerExitCode -ne 0) {
    exit $workerExitCode
}

# Give every power-on window a meaningful amount of search-index progress
# before slower NAS-native verification. This stage is resumable and becomes
# an immediate no-op once the Qwen3 index catches up.
$priorityEmbeddingArguments = @(
    "-m", "cctv_ai.cli", "visual-embed-qwen3",
    "--db", ".data\cctv_index.sqlite3",
    "--models-config", "config\models.toml",
    "--image-root", ".data\visual\keyframes",
    "--limit", "16384",
    # Commit 2048 rows at a time while the model config uses 128-frame CUDA
    # microbatches. This gives the encoder enough work to prefetch the next
    # CPU-prepared microbatch during the current GPU forward pass.
    "--batch-size", "2048"
)
$priorityEmbeddingStdout = Join-Path `
    $dataDirectory "qwen3-priority-embedding.out.log"
$priorityEmbeddingStderr = Join-Path `
    $dataDirectory "qwen3-priority-embedding.err.log"
$priorityEmbeddingExitCode = Invoke-LoggedPython `
    -Arguments $priorityEmbeddingArguments `
    -StandardOutput $priorityEmbeddingStdout `
    -StandardError $priorityEmbeddingStderr `
    -Label "Qwen3 priority embedding backfill"
if ($priorityEmbeddingExitCode -ne 0) {
    exit $priorityEmbeddingExitCode
}

# The fast detector/embedding pass deliberately produces broad candidates.
# Refine only those candidates with the larger local VLM, then atomically
# rebuild the diary from the conservative, cross-checked observations.
$verifierModel = Join-Path $dataDirectory "models\qwen3-vl-8b-instruct"
if (-not (Test-Path -LiteralPath (
    Join-Path $verifierModel "model.safetensors.index.json"
) -PathType Leaf)) {
    throw "Local visual verifier model was not found: $verifierModel"
}

$verifyArguments = @(
    "-m", "cctv_ai.cli", "visual-verify",
    "--db", ".data\cctv_index.sqlite3",
    "--config", $configPath,
    "--since", $analysisSince.ToString("o"),
    "--until", $analysisUntil.ToString("o"),
    "--image-root", ".data\visual\keyframes",
    "--media-root", $nasRoot,
    "--models-config", "config\models.toml",
    "--task", "feeding",
    "--task", "toy",
    "--task", "sleep_presence",
    "--task", "posture",
    "--batch-size", "8",
    # Event-level native verification replaces tens of thousands of noisy
    # single-frame toy checks. The upper bound covers a complete month while
    # database upserts keep restarts resumable.
    "--max-frames", "0",
    "--temporal-max-frames", "4096"
)
$verifyStdout = Join-Path $dataDirectory "visual-verify.out.log"
$verifyStderr = Join-Path $dataDirectory "visual-verify.err.log"
$verifyExitCode = Invoke-LoggedPython `
    -Arguments $verifyArguments `
    -StandardOutput $verifyStdout `
    -StandardError $verifyStderr `
    -Label "visual verifier"
if ($verifyExitCode -ne 0) {
    exit $verifyExitCode
}

$rebuildArguments = @(
    "-m", "cctv_ai.cli", "visual-rebuild",
    "--db", ".data\cctv_index.sqlite3",
    "--config", $configPath,
    "--since", $analysisSince.ToString("o"),
    "--until", $analysisUntil.ToString("o")
)
$rebuildStdout = Join-Path $dataDirectory "visual-rebuild.out.log"
$rebuildStderr = Join-Path $dataDirectory "visual-rebuild.err.log"
$rebuildExitCode = Invoke-LoggedPython `
    -Arguments $rebuildArguments `
    -StandardOutput $rebuildStdout `
    -StandardError $rebuildStderr `
    -Label "visual rebuild"
if ($rebuildExitCode -ne 0) {
    exit $rebuildExitCode
}

# Escalate only the earliest high-value posture candidates. The milestone
# command records the 8B/32B review in existing activity-event JSON and skips
# candidates already decided by the same 32B checkpoint.
$milestoneArguments = @(
    "-m", "cctv_ai.cli", "visual-milestones",
    "--db", ".data\cctv_index.sqlite3",
    "--models-config", "config\models.toml",
    "--image-root", ".data\visual\keyframes",
    "--since", $analysisSince.ToString("o"),
    "--until", $analysisUntil.ToString("o"),
    "--limit", "12"
)
$milestoneStdout = Join-Path $dataDirectory "visual-milestones.out.log"
$milestoneStderr = Join-Path $dataDirectory "visual-milestones.err.log"
$milestoneExitCode = Invoke-LoggedPython `
    -Arguments $milestoneArguments `
    -StandardOutput $milestoneStdout `
    -StandardError $milestoneStderr `
    -Label "visual milestones"
if ($milestoneExitCode -ne 0) {
    exit $milestoneExitCode
}

# Build the new Qwen3 retrieval index continuously for as long as the computer
# remains on. Each model batch is committed independently, so shutdown is safe
# and the next logon resumes at the first missing keyframe. Search keeps using
# the complete legacy SigLIP vectors until Qwen3 coverage reaches 95%.
$embeddingArguments = @(
    "-m", "cctv_ai.cli", "visual-embed-qwen3",
    "--db", ".data\cctv_index.sqlite3",
    "--models-config", "config\models.toml",
    "--image-root", ".data\visual\keyframes",
    "--batch-size", "2048"
)
$embeddingStdout = Join-Path $dataDirectory "qwen3-embedding.out.log"
$embeddingStderr = Join-Path $dataDirectory "qwen3-embedding.err.log"
$embeddingExitCode = Invoke-LoggedPython `
    -Arguments $embeddingArguments `
    -StandardOutput $embeddingStdout `
    -StandardError $embeddingStderr `
    -Label "Qwen3 embedding backfill"
exit $embeddingExitCode
