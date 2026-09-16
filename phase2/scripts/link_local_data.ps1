<#
.SYNOPSIS
    Point the pipeline at an existing local MIMIC-III / TWOSIDES download without copying anything.

.DESCRIPTION
    Creates directory junctions (no admin rights needed) under data/raw/ that point to your
    local copies, so that the default relative paths used in the README work:

        data/raw/mimic-iii        -> <MimicDir>        (credentialed MIMIC-III v1.4 CSVs)
        data/raw/mimic-iii-demo   -> <DemoDir>         (public demo, optional)
        data/raw/twosides.csv     -> <TwosidesCsv>     (hard link / copy is NOT made; a small .path file is written)

    data/raw/ is git-ignored, so nothing licensed can leak into the repository.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/link_local_data.ps1 `
        -MimicDir "D:\CLG\AU\projects\phase2\mimic-iii-clinical-database-1.4" `
        -DemoDir  "D:\CLG\AU\projects\phase2\mimic-iii-clinical-database-demo-1.4" `
        -TwosidesCsv "D:\CLG\AU\projects\phase2\twosides.csv"
#>
param(
    [Parameter(Mandatory = $true)][string]$MimicDir,
    [string]$DemoDir = "",
    [string]$TwosidesCsv = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$raw = Join-Path $root "data/raw"
New-Item -ItemType Directory -Force -Path $raw | Out-Null

function New-Junction([string]$link, [string]$target) {
    if (-not (Test-Path $target)) { throw "target does not exist: $target" }
    if (Test-Path $link) { (Get-Item $link).Delete() }
    New-Item -ItemType Junction -Path $link -Target (Resolve-Path $target) | Out-Null
    Write-Host "[link] $link -> $target"
}

New-Junction (Join-Path $raw "mimic-iii") $MimicDir
if ($DemoDir) { New-Junction (Join-Path $raw "mimic-iii-demo") $DemoDir }
if ($TwosidesCsv) {
    if (-not (Test-Path $TwosidesCsv)) { throw "TWOSIDES file not found: $TwosidesCsv" }
    Set-Content -Path (Join-Path $raw "twosides.path") -Value (Resolve-Path $TwosidesCsv).Path
    Write-Host "[link] data/raw/twosides.path -> $TwosidesCsv (pass --twosides `$(Get-Content data/raw/twosides.path) to build_ddi_graph.py)"
}
Write-Host "[link] done. Next: python scripts/preprocess_mimic.py --mimic_dir data/raw/mimic-iii --out data/processed/mimic3"
