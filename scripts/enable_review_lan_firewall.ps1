[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ruleName = "CCTV AI Review Server (Home LAN)"
$homeSubnet = "192.168.1.0/24"

$existing = Get-NetFirewallRule `
    -DisplayName $ruleName `
    -ErrorAction SilentlyContinue

if ($null -eq $existing) {
    New-NetFirewallRule `
        -DisplayName $ruleName `
        -Description "Allow CCTV review UI only from the trusted home LAN." `
        -Direction Inbound `
        -Action Allow `
        -Enabled True `
        -Profile Any `
        -Protocol TCP `
        -LocalPort 8765 `
        -RemoteAddress $homeSubnet | Out-Null
} else {
    Set-NetFirewallRule `
        -DisplayName $ruleName `
        -Direction Inbound `
        -Action Allow `
        -Enabled True `
        -Profile Any
    Get-NetFirewallPortFilter `
        -AssociatedNetFirewallRule $existing |
        Set-NetFirewallPortFilter `
            -Protocol TCP `
            -LocalPort 8765
    Get-NetFirewallAddressFilter `
        -AssociatedNetFirewallRule $existing |
        Set-NetFirewallAddressFilter `
            -RemoteAddress $homeSubnet
}

Write-Output "Firewall rule enabled for $homeSubnet on TCP 8765."
