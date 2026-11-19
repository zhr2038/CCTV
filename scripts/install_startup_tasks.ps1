[CmdletBinding()]
param(
    [switch]$StartNow
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$powershell = Join-Path $env:SystemRoot (
    "System32\WindowsPowerShell\v1.0\powershell.exe"
)
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal `
    -UserId $currentUser `
    -LogonType Interactive `
    -RunLevel Limited
$logonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $currentUser
$visualRecurringTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(5) `
    -RepetitionInterval (New-TimeSpan -Hours 3) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5)

$tasks = @(
    @{
        Name = "CCTV AI Visual Worker"
        Script = Join-Path $projectRoot "scripts\start_visual_worker.ps1"
        Description = "Local RTX visual indexing and baby diary backfill."
        Triggers = @($logonTrigger, $visualRecurringTrigger)
    },
    @{
        Name = "CCTV AI Pipeline Worker"
        Script = Join-Path $projectRoot "scripts\start_pipeline_worker.ps1"
        Description = "Read-only local AI processing for the baby-room CCTV archive."
        Triggers = @($logonTrigger)
    },
    @{
        Name = "CCTV AI Review Server"
        Script = Join-Path $projectRoot "scripts\start_review_server.ps1"
        Description = "LAN-only CCTV diary and event review web server."
        Triggers = @($logonTrigger)
    }
)

foreach ($task in $tasks) {
    $argument = (
        "-NoProfile -NonInteractive -WindowStyle Hidden " +
        "-ExecutionPolicy Bypass -File `"$($task.Script)`""
    )
    $action = New-ScheduledTaskAction `
        -Execute $powershell `
        -Argument $argument `
        -WorkingDirectory $projectRoot
    Register-ScheduledTask `
        -TaskName $task.Name `
        -Action $action `
        -Trigger $task.Triggers `
        -Principal $principal `
        -Settings $settings `
        -Description $task.Description `
        -Force | Out-Null
    if ($StartNow) {
        Start-ScheduledTask -TaskName $task.Name
    }
}

Get-ScheduledTask -TaskName ($tasks.Name) |
    Select-Object TaskName, State
