<#
.SYNOPSIS
  Copy the raw KIBA table from the original thesis workspace into this repo's data/raw folder
  (convenience for the author's machine; everyone else should use scripts/download_kiba.py).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\link_local_data.ps1
  powershell -ExecutionPolicy Bypass -File scripts\link_local_data.ps1 -Source "E:\backup\kiba.tab"
#>
param(
    # TDC download kept in the original Phase 1 workspace
    [string]$Source = "D:\CLG\AU\projects\Phase1\scripts\data\kiba.tab",
    [switch]$Symlink
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$dest = Join-Path $root "data\raw\kiba.tab"

if (-not (Test-Path $Source)) {
    Write-Error "Source file not found: $Source. Run 'python scripts/download_kiba.py' instead."
}
New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null

if ($Symlink) {
    # requires Developer Mode or an elevated shell on Windows
    New-Item -ItemType SymbolicLink -Path $dest -Target $Source -Force | Out-Null
    Write-Host "Symlinked $dest -> $Source"
} else {
    Copy-Item -Path $Source -Destination $dest -Force
    Write-Host "Copied $Source -> $dest ($([math]::Round((Get-Item $dest).Length / 1MB, 1)) MB)"
}
Write-Host "Next: python scripts/preprocess.py"
