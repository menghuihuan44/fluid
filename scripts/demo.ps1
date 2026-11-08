# Demo: end-to-end self-test without a display or a game.
# Captures the synthetic mock source, runs interpolation + upscaling, and
# writes a JSON report with per-stage timings.
# Usage:  pwsh -File scripts/demo.ps1
param(
    [string]$Size = "1280x720",
    [int]$Frames = 300,
    [string]$OutDir = "out"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    cargo build --release -p fluid-cli
    $exe = Join-Path $root "target/release/fluid.exe"
    if (-not (Test-Path $exe)) { $exe = Join-Path $root "target/release/fluid" }

    Write-Host "== devices ==" -ForegroundColor Cyan
    & $exe devices

    Write-Host "== demo: mock capture, 2x interpolation, 1.5x upscale ==" -ForegroundColor Cyan
    & $exe run --capture mock --no-present --frames $Frames --multiplier 2 `
        --scale 1.5 --filter adaptive --report (Join-Path $OutDir "demo-report.json")

    Write-Host "report: $(Join-Path $OutDir 'demo-report.json')" -ForegroundColor Green
}
finally {
    Pop-Location
}
