<#
.SYNOPSIS
    End-to-end smoke run of the Phase-2 pipeline on the public MIMIC-III demo (100 patients).

.DESCRIPTION
    1. downloads the MIMIC-III Clinical Database Demo v1.4 (ODbL, ~ 40 MB unzipped) if needed,
    2. preprocesses it into admission records,
    3. builds the heterogeneous graph (uses the shipped TWOSIDES pair table + PubChem mapping),
    4. trains the full model for a few epochs,
    5. runs a 2-variant x 1-seed mini ablation and makes the figures,
    6. runs the unit tests.

    Total runtime: ~3-5 minutes on CPU.  Numbers obtained on the demo are NOT meaningful -
    the demo is far too small - the point is to check that every stage runs.

.PARAMETER Python
    Interpreter to use.  Defaults to the phase-2 venv next to this repo, then to `python`.
.PARAMETER DemoDir
    Where the demo CSVs live / should be downloaded to (default: data/raw/mimic-iii-demo).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/demo_mode.ps1
#>
param(
    [string]$Python = "",
    [string]$DemoDir = "data/raw/mimic-iii-demo",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $Python) {
    $candidates = @("$root/../.venv_phase2/Scripts/python.exe", "$root/.venv/Scripts/python.exe", "python")
    foreach ($c in $candidates) { if ($c -eq "python" -or (Test-Path $c)) { $Python = $c; break } }
}
Write-Host "[demo] python = $Python"

# 1. demo data ------------------------------------------------------------------------------
if (-not (Test-Path "$DemoDir/PRESCRIPTIONS.csv")) {
    New-Item -ItemType Directory -Force -Path $DemoDir | Out-Null
    $zip = "$DemoDir/mimic-iii-clinical-database-demo-1.4.zip"
    Write-Host "[demo] downloading MIMIC-III demo v1.4 from PhysioNet ..."
    Invoke-WebRequest -Uri "https://physionet.org/static/published-projects/mimiciii-demo/mimic-iii-clinical-database-demo-1.4.zip" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $DemoDir -Force
    $inner = Get-ChildItem $DemoDir -Directory | Where-Object { $_.Name -like "mimic-iii-clinical-database-demo*" } | Select-Object -First 1
    if ($inner) { Get-ChildItem $inner.FullName | Move-Item -Destination $DemoDir -Force; Remove-Item $inner.FullName -Recurse -Force }
    Remove-Item $zip -Force
}
Write-Host "[demo] MIMIC-III demo at $DemoDir"

# 2-3. preprocess + graph -----------------------------------------------------------------------
& $Python -W ignore scripts/preprocess_mimic.py --mimic_dir $DemoDir --out data/processed/demo `
    --min_drug_admissions 3 --min_diag_count 2 --min_proc_count 2
if ($LASTEXITCODE -ne 0) { throw "preprocess failed" }
& $Python -W ignore scripts/build_hetero_graph.py --processed_dir data/processed/demo `
    --min_cooccur 2 --top_k_per_node 20 --drug_cooccur_min 2
if ($LASTEXITCODE -ne 0) { throw "graph build failed" }

# 4. one training run -----------------------------------------------------------------------------
& $Python -W ignore scripts/train.py --config configs/demo.yaml --results_dir results_demo
if ($LASTEXITCODE -ne 0) { throw "training failed" }

# 5. mini ablation + figures ------------------------------------------------------------------------
& $Python -W ignore scripts/run_ablation.py --processed_dir data/processed/demo --results_dir results_demo `
    --variants full no_ddi no_mol baseline_lr --seeds 0 --set train.epochs=8 model.hidden_dim=32 train.batch_size=32 `
    "train.select_metric=jaccard+prauc" --title "Demo ablation (MIMIC-III demo, NOT meaningful)"
if ($LASTEXITCODE -ne 0) { throw "ablation failed" }
& $Python -W ignore scripts/make_figures.py --results_dir results_demo
if ($LASTEXITCODE -ne 0) { throw "figures failed" }

# 6. tests ---------------------------------------------------------------------------------------
if (-not $SkipTests) {
    & $Python -m pytest -q tests
    if ($LASTEXITCODE -ne 0) { throw "tests failed" }
}
Write-Host "[demo] OK - see results_demo/ (git-ignored)"
