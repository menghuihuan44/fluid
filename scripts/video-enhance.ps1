#Requires -Version 5.1
<#
.SYNOPSIS
    Enhance a video with Fluid: ffmpeg decode -> Fluid interpolate/upscale -> ffmpeg encode.

.DESCRIPTION
    Fluid reads and writes y4m directly, and it also has built-in MJPEG AVI
    readers/writers, so .y4m and .avi files need no ffmpeg at all: for those the
    script calls Fluid on the files themselves. Everything else is handled by
    wiring two ffmpeg processes around "fluid video -i - -o -":

        ffmpeg -i IN ... -f yuv4mpegpipe - | fluid video -i - -o - ... | ffmpeg ... OUT

    Audio is never processed by Fluid: the output ffmpeg copies the audio track
    from the original file with a second "-i IN" input (and re-encodes it to
    Opus for .webm). The built-in AVI path has no audio track.

    If ffmpeg is not installed the script explains how to install it, and still
    supports the built-in .y4m / .avi paths (Fluid alone, no transcoding).

.PARAMETER In
    Input video: a .y4m file or a MJPEG .avi (both read by Fluid directly, no
    ffmpeg), or anything ffmpeg can decode (mp4, mkv, mov, webm, ...).

.PARAMETER Out
    Output path. The extension picks the container defaults: .mp4/.m4v/.mov
    (H.264 or H.265, faststart), .mkv (either), .webm (VP9 + Opus), or the
    built-in .y4m / .avi (no ffmpeg, AVI is MJPEG and has no audio track).

.PARAMETER Multiplier
    Frames generated per input frame; 2 doubles the framerate. Default 2.

.PARAMETER Scale
    Upscale factor relative to the source (1.0 = native, 1.5, 2.0; 0 disables
    upscaling). Default 1.0.

.PARAMETER Filter
    Upscale filter: off, nearest, bilinear, bicubic, lanczos, adaptive, temporal.
    Omit to use the Fluid config default.

.PARAMETER Sharpness
    Sharpening strength 0.0 .. 1.0. Omit to use the Fluid config default.

.PARAMETER Codec
    h264 (default), h265, or vp9. .webm output forces vp9.

.PARAMETER Crf
    Constant-quality value 0..51; 0 means the codec default (h264 18, h265 22,
    vp9 32).

.PARAMETER Preset
    x264/x265 speed preset. Default medium.

.PARAMETER Bitrate
    Target video bitrate in kbps; 0 (default) keeps constant-quality CRF mode.

.PARAMETER JpegQuality
    MJPEG quality 1..100 for a .avi output (default 88). Only valid together
    with a .avi output; ignored for every other container.

.PARAMETER Fluid
    Path to fluid.exe. Defaults to FLUID_EXE, target/release or target/debug in
    this repository, then PATH.

.PARAMETER Ffmpeg
    Path to ffmpeg.exe. Defaults to PATH, FFMPEG, then common install folders.

.EXAMPLE
    .\scripts\video-enhance.ps1 -In clip.mp4 -Out clip_2x.mp4
    .\scripts\video-enhance.ps1 -In in.mp4 -Out out.mp4 -Multiplier 2 -Scale 2 -Filter lanczos -Sharpness 0.3 -Crf 18
    .\scripts\video-enhance.ps1 -In clip.y4m -Out clip_out.y4m -Multiplier 2
    .\scripts\video-enhance.ps1 -In clip.avi -Out clip_2x.avi -Multiplier 2 -Scale 1.5 -JpegQuality 92
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$In,

    [Parameter(Mandatory = $true, Position = 1)]
    [string]$Out,

    [ValidateRange(1, 8)]
    [int]$Multiplier = 2,

    [ValidateRange(0.0, 8.0)]
    [double]$Scale = 1.0,

    [ValidateSet('off', 'nearest', 'bilinear', 'bicubic', 'lanczos', 'adaptive', 'temporal')]
    [string]$Filter = '',

    [double]$Sharpness = -1.0,

    [ValidateSet('h264', 'h265', 'vp9')]
    [string]$Codec = 'h264',

    [ValidateRange(0, 51)]
    [int]$Crf = 0,

    [ValidateSet('ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow')]
    [string]$Preset = 'medium',

    [ValidateRange(0, 1000000)]
    [int]$Bitrate = 0,

    [ValidateRange(0, 100)]
    [int]$JpegQuality = 0,

    [string]$Fluid = '',

    [string]$Ffmpeg = ''
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

function Format-Num {
    param([double]$Value)
    return $Value.ToString('0.####', [Globalization.CultureInfo]::InvariantCulture)
}

function Quote-Argument {
    param([string]$Value)
    # Windows file names cannot contain a double quote, so wrapping in quotes is
    # enough to survive cmd.exe word splitting.
    return '"' + $Value + '"'
}

function Find-Executable {
    param(
        [string]$Explicit,
        [string[]]$Names,
        [string[]]$Candidates
    )
    if ($Explicit) {
        if (Test-Path -LiteralPath $Explicit -PathType Leaf) {
            return (Resolve-Path -LiteralPath $Explicit).Path
        }
        $cmd = Get-Command $Explicit -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
        return $null
    }
    foreach ($name in $Names) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}

function Get-FfmpegCandidates {
    $list = New-Object System.Collections.ArrayList
    if ($env:FFMPEG) { [void]$list.Add($env:FFMPEG) }
    $winget = $null
    if ($env:LOCALAPPDATA) {
        $winget = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'
    }
    if ($winget -and (Test-Path -LiteralPath $winget -PathType Container)) {
        $packages = Get-ChildItem -LiteralPath $winget -Directory -Filter 'Gyan.FFmpeg*' -ErrorAction SilentlyContinue
        foreach ($package in $packages) {
            $exe = Get-ChildItem -LiteralPath $package.FullName -Recurse -Filter 'ffmpeg.exe' -File -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($exe) { [void]$list.Add($exe.FullName) }
        }
    }
    if ($env:ProgramFiles) {
        [void]$list.Add((Join-Path $env:ProgramFiles 'ffmpeg\bin\ffmpeg.exe'))
    }
    if ($env:ProgramData) {
        [void]$list.Add((Join-Path $env:ProgramData 'chocolatey\bin\ffmpeg.exe'))
    }
    if ($env:USERPROFILE) {
        [void]$list.Add((Join-Path $env:USERPROFILE 'scoop\shims\ffmpeg.exe'))
    }
    [void]$list.Add('C:\ffmpeg\bin\ffmpeg.exe')
    return $list
}

function Get-FluidCandidates {
    $list = New-Object System.Collections.ArrayList
    if ($env:FLUID_EXE) { [void]$list.Add($env:FLUID_EXE) }
    [void]$list.Add((Join-Path $root 'target\release\fluid.exe'))
    [void]$list.Add((Join-Path $root 'target\debug\fluid.exe'))
    return $list
}

function Show-FfmpegHelp {
    Write-Host ''
    Write-Host 'ffmpeg was not found on PATH.' -ForegroundColor Yellow
    Write-Host 'Fluid ships no video codecs, so mp4/mkv/webm need ffmpeg to decode and encode.'
    Write-Host ''
    Write-Host 'Install it with one of:'
    Write-Host '  winget install Gyan.FFmpeg'
    Write-Host '  scoop install ffmpeg'
    Write-Host '  choco install ffmpeg'
    Write-Host 'or download a build from https://www.gyan.dev/ffmpeg/builds/ and add its bin folder to PATH.'
    Write-Host 'You can also set the FFMPEG environment variable, or pass -Ffmpeg <path>.'
    Write-Host ''
}

function Invoke-Fluid {
    param([string[]]$Arguments)
    & $script:fluidExe @Arguments
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "fluid exited with code $code"
    }
}

# ---------------------------------------------------------------------------
# Resolve paths and tools.
# ---------------------------------------------------------------------------
if (-not (Test-Path -LiteralPath $In -PathType Leaf)) {
    throw "input file not found: $In"
}
$inPath = (Resolve-Path -LiteralPath $In).Path
if ([IO.Path]::IsPathRooted($Out)) {
    $outPath = $Out
} else {
    $outPath = Join-Path (Get-Location).ProviderPath $Out
}
$outPath = [IO.Path]::GetFullPath($outPath)

$inExt = [IO.Path]::GetExtension($inPath).ToLowerInvariant()
$outExt = [IO.Path]::GetExtension($outPath).ToLowerInvariant()

# Extensions Fluid can read/write on its own, without any ffmpeg involvement.
$nativeExts = @('.y4m', '.avi')
$inNative = $nativeExts -contains $inExt
$outNative = $nativeExts -contains $outExt

if ($PSBoundParameters.ContainsKey('Sharpness') -and ($Sharpness -lt 0.0 -or $Sharpness -gt 1.0)) {
    throw '-Sharpness must be between 0.0 and 1.0'
}

$script:fluidExe = Find-Executable -Explicit $Fluid -Names @('fluid.exe', 'fluid') -Candidates (Get-FluidCandidates)
if (-not $script:fluidExe) {
    throw "fluid executable not found; run 'cargo build --release -p fluid-cli' or pass -Fluid <path>"
}
$ffmpegExe = Find-Executable -Explicit $Ffmpeg -Names @('ffmpeg.exe', 'ffmpeg') -Candidates (Get-FfmpegCandidates)

$commonFluid = @('--multiplier', "$Multiplier", '--scale', (Format-Num $Scale))
if ($Filter) { $commonFluid += @('--filter', $Filter) }
if ($PSBoundParameters.ContainsKey('Sharpness')) { $commonFluid += @('--sharpness', (Format-Num $Sharpness)) }

# --jpeg-quality only exists for the built-in MJPEG AVI writer.
$jpegArgs = @()
if ($outExt -eq '.avi') {
    if ($JpegQuality -gt 0) { $jpegArgs = @('--jpeg-quality', "$JpegQuality") }
} elseif ($JpegQuality -gt 0) {
    throw '-JpegQuality can only be used with a .avi output'
}

if ($inNative -and $outNative) {
    # Both sides are built-in: let Fluid read and write the files directly.
    Write-Host 'Fluid video enhance (built-in path: no ffmpeg)' -ForegroundColor Cyan
    Write-Host "  input : $inPath"
    Write-Host "  output: $outPath"
    if ($outExt -eq '.avi') {
        Write-Host ("  codec : MJPEG AVI, quality {0}, no audio track" -f $(if ($JpegQuality -gt 0) { $JpegQuality } else { 88 })) -ForegroundColor DarkGray
    }
    $outDir = [IO.Path]::GetDirectoryName($outPath)
    if ($outDir -and -not (Test-Path -LiteralPath $outDir -PathType Container)) {
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    }
    $fileArgs = @('video', '-i', $inPath, '-o', $outPath) + $commonFluid + $jpegArgs
    Invoke-Fluid -Arguments $fileArgs
    Write-Host "wrote $outPath" -ForegroundColor Green
    return
}

if (-not $ffmpegExe) {
    Show-FfmpegHelp
    $needed = @()
    if (-not $inNative) { $needed += "decode '$inExt' input" }
    if (-not $outNative) { $needed += "encode '$outExt' output" }
    Write-Host ("error: ffmpeg is required to " + ($needed -join ' and ') + '.') -ForegroundColor Red
    Write-Host '       install it (see above), or use .y4m / .avi for both input and output (Fluid reads/writes those itself).' -ForegroundColor Red
    exit 1
}

if ($outExt -eq '') {
    throw "output path needs a container extension (for example .mp4, .mkv, .webm or .y4m): $outPath"
}
if ($outExt -eq '.webm' -and $Codec -ne 'vp9') {
    Write-Host 'note: .webm output requires VP9; switching the codec to vp9' -ForegroundColor Yellow
    $Codec = 'vp9'
}

$outDir = [IO.Path]::GetDirectoryName($outPath)
if (-not $outDir) { $outDir = (Get-Location).ProviderPath }
if (-not (Test-Path -LiteralPath $outDir -PathType Container)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

$crfValue = $Crf
if ($crfValue -eq 0) {
    switch ($Codec) {
        'h264' { $crfValue = 18 }
        'h265' { $crfValue = 22 }
        'vp9'  { $crfValue = 32 }
    }
}

# Native outputs are written to a temporary sibling first, so a failed run never
# leaves a half-written file at the requested path.
$tmpOut = Join-Path $outDir ([IO.Path]::GetFileNameWithoutExtension($outPath) + '.fluid-tmp' + $outExt)

# ---------------------------------------------------------------------------
# Decode stage (only when the input is not a built-in .y4m/.avi file).
# ---------------------------------------------------------------------------
$decodeArgs = @(
    '-hide_banner', '-nostdin', '-loglevel', 'error',
    '-i', $inPath,
    '-map', '0:v:0',
    '-f', 'yuv4mpegpipe',
    '-pix_fmt', 'yuv420p',
    '-'
)

# ---------------------------------------------------------------------------
# Fluid stage: read the input file for built-in inputs, y4m stdin otherwise;
# write the output file for built-in outputs, y4m stdout otherwise.
# ---------------------------------------------------------------------------
$fluidInArgs = if ($inNative) { @($inPath) } else { @('-') }
$fluidOutArgs = if ($outNative) { @($tmpOut) } else { @('-') }
$fluidPipeArgs = @('video', '-i') + $fluidInArgs + @('-o') + $fluidOutArgs + $commonFluid + $jpegArgs

# ---------------------------------------------------------------------------
# Encode stage (only when the output is not a built-in .y4m/.avi container).
# ---------------------------------------------------------------------------
$isMp4Family = $outExt -in @('.mp4', '.m4v', '.mov')
$isWebm = $outExt -eq '.webm'

$encodeArgs = @()
if (-not $outNative) {
    $encodeArgs = @(
        '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'yuv4mpegpipe', '-i', '-',
        '-i', $inPath,
        '-map', '0:v:0', '-map', '1:a?'
    )
    switch ($Codec) {
        'h264' {
            $encodeArgs += @('-c:v', 'libx264', '-preset', $Preset)
            if ($Bitrate -gt 0) {
                $encodeArgs += @('-b:v', ($Bitrate.ToString() + 'k'),
                    '-maxrate', ((2 * $Bitrate).ToString() + 'k'),
                    '-bufsize', ((4 * $Bitrate).ToString() + 'k'))
            } else {
                $encodeArgs += @('-crf', "$crfValue")
            }
        }
        'h265' {
            $encodeArgs += @('-c:v', 'libx265', '-preset', $Preset)
            if ($isMp4Family) { $encodeArgs += @('-tag:v', 'hvc1') }
            if ($Bitrate -gt 0) {
                $encodeArgs += @('-b:v', ($Bitrate.ToString() + 'k'),
                    '-maxrate', ((2 * $Bitrate).ToString() + 'k'),
                    '-bufsize', ((4 * $Bitrate).ToString() + 'k'))
            } else {
                $encodeArgs += @('-crf', "$crfValue")
            }
        }
        'vp9' {
            $encodeArgs += @('-c:v', 'libvpx-vp9', '-deadline', 'good', '-cpu-used', '2', '-row-mt', '1')
            if ($Bitrate -gt 0) {
                $encodeArgs += @('-b:v', ($Bitrate.ToString() + 'k'))
            } else {
                $encodeArgs += @('-crf', "$crfValue", '-b:v', '0')
            }
        }
    }
    $encodeArgs += @('-pix_fmt', 'yuv420p')
    if ($isWebm) {
        $encodeArgs += @('-c:a', 'libopus', '-b:a', '160k')
    } else {
        $encodeArgs += @('-c:a', 'copy')
    }
    $encodeArgs += @('-map_metadata', '1', '-map_chapters', '1')
    if ($isMp4Family) { $encodeArgs += @('-movflags', '+faststart') }
    $encodeArgs += $tmpOut
}

# ---------------------------------------------------------------------------
# Show the plan, then run everything through one cmd.exe batch file so the
# pipe between the three processes stays inside a single console session.
# ---------------------------------------------------------------------------
Write-Host 'Fluid video enhance' -ForegroundColor Cyan
Write-Host "  input    : $inPath"
Write-Host "  output   : $outPath"
Write-Host "  fluid    : $script:fluidExe"
if ($ffmpegExe) { Write-Host "  ffmpeg   : $ffmpegExe" }
Write-Host "  frames   : x$Multiplier ($(if ($Filter) { $Filter } else { 'config default filter' }))"
Write-Host "  scale    : x$(Format-Num $Scale)"
if (-not $outNative) {
    Write-Host "  encode   : $Codec crf $crfValue preset $Preset"
} elseif ($outExt -eq '.avi') {
    Write-Host ("  codec    : MJPEG AVI, quality {0}, no audio track" -f $(if ($JpegQuality -gt 0) { $JpegQuality } else { 88 }))
}

function Build-CommandLine {
    param([string[]]$Arguments)
    return (($Arguments | ForEach-Object { Quote-Argument $_ }) -join ' ')
}

# One to three pipe segments: ffmpeg decode | fluid | ffmpeg encode, dropping
# the ffmpeg stages whose side is an extension Fluid handles itself.
$segments = @()
if (-not $inNative) {
    $segments += (Quote-Argument $ffmpegExe) + ' ' + (Build-CommandLine $decodeArgs)
}
$segments += (Quote-Argument $script:fluidExe) + ' ' + (Build-CommandLine $fluidPipeArgs)
if (-not $outNative) {
    $segments += (Quote-Argument $ffmpegExe) + ' ' + (Build-CommandLine $encodeArgs)
}
$cmdLine = $segments -join ' | '

$batchPath = Join-Path ([IO.Path]::GetTempPath()) ('fluid-video-' + [Guid]::NewGuid().ToString('N') + '.cmd')
$batchLines = @(
    '@echo off',
    'setlocal',
    $cmdLine,
    'exit /b %errorlevel%'
)

try {
    Set-Content -LiteralPath $batchPath -Value $batchLines -Encoding ASCII
    Write-Host 'running the pipeline ...' -ForegroundColor Cyan
    $process = Start-Process -FilePath 'cmd.exe' -ArgumentList @('/d', '/s', '/c', $batchPath) -NoNewWindow -PassThru -Wait
    $exitCode = $process.ExitCode
    if ($exitCode -ne 0) {
        throw "the video pipeline failed (exit code $exitCode); see the messages above"
    }
    if (-not (Test-Path -LiteralPath $tmpOut -PathType Leaf)) {
        throw 'the pipeline finished but produced no output file; is the input an empty or audio-only file?'
    }
    $produced = Get-Item -LiteralPath $tmpOut
    if ($produced.Length -eq 0) {
        throw 'the pipeline finished but the output file is empty'
    }
    Move-Item -LiteralPath $tmpOut -Destination $outPath -Force
    Write-Host ("wrote {0} ({1:N0} bytes)" -f $outPath, $produced.Length) -ForegroundColor Green
}
finally {
    if (Test-Path -LiteralPath $batchPath -PathType Leaf) {
        Remove-Item -LiteralPath $batchPath -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $tmpOut -PathType Leaf) {
        Remove-Item -LiteralPath $tmpOut -Force -ErrorAction SilentlyContinue
    }
}
