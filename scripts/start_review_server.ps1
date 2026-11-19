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
$lanSubnet = if ($env:CCTV_LAN_SUBNET) { $env:CCTV_LAN_SUBNET } else { "192.168.1.0/24" }

$arguments = @(
    "-m", "cctv_ai.cli", "serve",
    "--db", ".data\cctv_index.sqlite3",
    "--config", $configPath,
    "--root", $nasRoot,
    "--host", "0.0.0.0",
    "--port", "8765",
    "--allowed-network", "127.0.0.0/8",
    "--allowed-network", "::1/128",
    "--allowed-network", $lanSubnet
)

$stdout = Join-Path $dataDirectory "review-server.out.log"
$stderr = Join-Path $dataDirectory "review-server.err.log"
$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru `
    -Wait
exit $process.ExitCode
