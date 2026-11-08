#Requires -Version 5.1
<#
.SYNOPSIS
  Clean Fluid build caches and scratch artifacts.

.DESCRIPTION
  Deletes everything that can be regenerated: cargo target directories, isolated
  CARGO_HOME copies, and the scratch files under out/ that are not release
  deliverables. Keeps out/fluid-<version>-win64.zip, the QA reports and the small
  measurement JSON files used by the docs.

.PARAMETER All
  Also delete the kept measurement JSON files (keeps only the release zip and QA reports).

.PARAMETER DryRun
  Print what would be removed without deleting anything.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\clean.ps1
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\clean.ps1 -DryRun
#>
param(
    [switch]$All,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

function Get-SizeMb([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    $item = Get-Item -LiteralPath $Path -Force
    if (-not $item.PSIsContainer) { return [math]::Round($item.Length / 1MB, 1) }
    $sum = (Get-ChildItem -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue |
        Measure-Object -Property Length -Sum).Sum
    if ($null -eq $sum) { return 0 }
    return [math]::Round($sum / 1MB, 1)
}

$removedMb = 0.0
$removedCount = 0

function Remove-Path([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $mb = Get-SizeMb $Path
    if ($DryRun) {
        Write-Host ('  would remove {0,-42} {1,10:N1} MB' -f $Label, $mb)
    } else {
        Remove-Item -LiteralPath $Path -Recurse -Force
        Write-Host ('  removed      {0,-42} {1,10:N1} MB' -f $Label, $mb)
    }
    $script:removedMb += $mb
    $script:removedCount++
}

Write-Host 'Fluid cleanup'
Write-Host ''
Write-Host 'build caches:'

# cargo target directories (workspace default plus the per-agent ones)
Get-ChildItem -LiteralPath $root -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like 'target*' } |
    ForEach-Object { Remove-Path $_.FullName $_.Name }

# isolated CARGO_HOME copies used to dodge the shared package-cache lock
Get-ChildItem -LiteralPath $root -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like '.cargo-*' } |
    ForEach-Object { Remove-Path $_.FullName $_.Name }

Write-Host ''
Write-Host 'out/ scratch files:'

$keep = @('fluid-0.1.0-win64.zip', 'QA-REPORT.md', 'QA-REPORT2.md', 'QA-REPORT3.md')
if (-not $All) {
    $keep += @(
        'gpu-bench.json', 'upscale-quality.json', 'interp-quality.json',
        'bench-1080p.json', 'fluid.toml', 'run-report.json',
        'capture-report.json', 'video-report.json', 'window-run.json',
        'final-gui-smoke.json'
    )
}

$outDir = Join-Path $root 'out'
if (Test-Path -LiteralPath $outDir) {
    Get-ChildItem -LiteralPath $outDir -Force |
        Where-Object { $keep -notcontains $_.Name } |
        ForEach-Object { Remove-Path $_.FullName (Join-Path 'out' $_.Name) }
}

Write-Host ''
if ($DryRun) {
    Write-Host ('dry run: {0} item(s), {1:N1} MB would be freed' -f $removedCount, $removedMb)
} else {
    Write-Host ('done: {0} item(s), {1:N1} MB freed' -f $removedCount, $removedMb)
}