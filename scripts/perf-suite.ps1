# Fluid performance suite: run a fixed matrix of benchmarks on this machine,
# keep the JSON reports and emit a markdown table for the docs.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\perf-suite.ps1
#   powershell ... -Fluid target-package\release\fluid.exe -Frames 120
#   powershell ... -SkipCpuCompare    # only the default (auto) backend rows
#
# Outputs: out\perf-suite\*.json and out\perf-suite.md
param(
    [string]$Fluid = '',
    [string]$OutDir = 'out',
    [int]$Frames = 90,
    [switch]$SkipCpuCompare
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if ($OutDir -notmatch '^[A-Za-z]:') { $OutDir = Join-Path $root $OutDir }
$suiteDir = Join-Path $OutDir 'perf-suite'
New-Item -ItemType Directory -Force -Path $suiteDir | Out-Null

function Resolve-Fluid {
    if ($Fluid) {
        if (Test-Path $Fluid) { return (Resolve-Path $Fluid).Path }
        throw "fluid executable not found: $Fluid"
    }
    $candidates = @(
        (Join-Path $root 'target-package\release\fluid.exe'),
        (Join-Path $root 'target-lead3\release\fluid.exe'),
        (Join-Path $root 'target-lead2\release\fluid.exe'),
        (Join-Path $root 'target\release\fluid.exe')
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    $cmd = Get-Command fluid.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw 'fluid.exe not found; build it (cargo build --release -p fluid-cli) or pass -Fluid <path>'
}

$exe = Resolve-Fluid
Write-Host ('fluid: ' + $exe)

$profiles = @(
    @{ name = '720p upscale 2x bilinear'; args = @('--size','1280x720','--scale','2','--filter','bilinear','--sharpness','0.3') },
    @{ name = '720p 2x interp + 1.5x';    args = @('--size','1280x720','--multiplier','2','--scale','1.5','--filter','bilinear') },
    @{ name = '1080p 2x interp native';   args = @('--size','1920x1080','--multiplier','2','--scale','1.0') },
    @{ name = '1080p 2x interp 1.5x';     args = @('--size','1920x1080','--multiplier','2','--scale','1.5','--filter','bilinear') },
    @{ name = '1080p to 4K nearest';      args = @('--size','1920x1080','--scale','2','--filter','nearest') },
    @{ name = '1080p to 4K bilinear';     args = @('--size','1920x1080','--scale','2','--filter','bilinear') },
    @{ name = '1080p to 4K adaptive';     args = @('--size','1920x1080','--scale','2','--filter','adaptive') },
    @{ name = '1080p pass-through';       args = @('--size','1920x1080','--multiplier','1','--scale','1.0','--sharpness','0') }
)

$rows = New-Object System.Collections.ArrayList
function Invoke-Row([string]$label, [string[]]$extra) {
    $safe = ($label -replace '[^A-Za-z0-9]+','-').Trim('-')
    $file = Join-Path $suiteDir ($safe + '.json')
    $argv = @('bench') + $extra + @('--frames', $Frames, '--report', $file)
    Write-Host ('  -> ' + $label)
    # fluid logs to stderr; PowerShell 5.1 turns native stderr into a terminating error under Stop,
    # so run it through Start-Process with redirected streams instead of a direct call.
    $stdout = Join-Path $suiteDir '_stdout.tmp'
    $stderr = Join-Path $suiteDir '_stderr.tmp'
    $proc = Start-Process -FilePath $exe -ArgumentList $argv -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if ($proc.ExitCode -ne 0) { Write-Warning ('fluid exited with ' + $proc.ExitCode + ' for ' + $label) }
    if (-not (Test-Path $file)) { Write-Warning ('no report for ' + $label); return }
    $j = Get-Content -Raw -Path $file | ConvertFrom-Json
    [void]$rows.Add([pscustomobject]@{
        Name = $label
        Args = ($extra -join ' ')
        OutFps = [double]$j.throughput_out_fps
        ProcessMs = [double]$j.process_mean_ms
        FlowMs = [double]$j.flow_mean_ms
        InterpMs = [double]$j.interp_mean_ms
        UpscaleMs = [double]$j.upscale_mean_ms
        Backend = [string]$j.backend
        InterpBackend = [string]$j.interp_backend
        WallS = [double]$j.wall_s
    })
}

foreach ($p in $profiles) { Invoke-Row $p.name $p.args }
if (-not $SkipCpuCompare) {
    Write-Host 'CPU comparison rows:'
    foreach ($p in $profiles) {
        $cpuArgs = $p.args + @('--backend','cpu','--interp-backend','cpu')
        Invoke-Row ($p.name + ' (cpu only)') $cpuArgs
    }
}

$stamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
$lines = New-Object System.Collections.ArrayList
[void]$lines.Add('# Fluid performance suite')
[void]$lines.Add('')
[void]$lines.Add('- generated: ' + $stamp)
[void]$lines.Add('- fluid: ' + $exe)
[void]$lines.Add('- frames per run: ' + $Frames)
[void]$lines.Add('- logical processors: ' + [System.Environment]::ProcessorCount)
[void]$lines.Add('')
[void]$lines.Add('| scenario | out fps | process ms | flow ms | interp ms | upscale ms | upscale backend | interp backend |')
[void]$lines.Add('|---|---:|---:|---:|---:|---:|---|---|')
foreach ($r in $rows) {
    [void]$lines.Add(('| {0} | {1:N1} | {2:N2} | {3:N2} | {4:N2} | {5:N2} | {6} | {7} |' -f $r.Name, $r.OutFps, $r.ProcessMs, $r.FlowMs, $r.InterpMs, $r.UpscaleMs, $r.Backend, $r.InterpBackend))
}
[void]$lines.Add('')
[void]$lines.Add('Notes: engine-level numbers (synthetic source, null sink), no display cadence included.')
[void]$lines.Add('Run on an idle machine; a compiling or shared machine reports lower fps.')
$md = Join-Path $OutDir 'perf-suite.md'
Set-Content -Path $md -Value ($lines -join [Environment]::NewLine) -Encoding UTF8
Write-Host ('wrote ' + $md)
$rows | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
