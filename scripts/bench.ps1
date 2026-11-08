# Fluid benchmark sweep: runs the synthetic pipeline at several sizes and
# writes one JSON report per size under out/bench/.
# Usage:  pwsh -File scripts/bench.ps1 [-Frames 240]
param(
    [int]$Frames = 240,
    [string]$OutDir = "out/bench"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    cargo build --release -p fluid-cli
    $exe = Join-Path $root "target/release/fluid.exe"
    if (-not (Test-Path $exe)) { $exe = Join-Path $root "target/release/fluid" }

    $sizes = @("1280x720", "1920x1080", "2560x1440")
    foreach ($size in $sizes) {
        $report = Join-Path $OutDir ("bench-" + $size + ".json")
        Write-Host "== bench $size ==" -ForegroundColor Cyan
        & $exe bench --size $size --frames $Frames --warmup 10 --json --report $report
        if ($LASTEXITCODE -ne 0) { throw "bench failed for $size" }
    }
    Write-Host "reports written to $OutDir" -ForegroundColor Green
}
finally {
    Pop-Location
}
