# Capture a Fluid window to a PNG (used for docs and smoke verification).
#
#   .\scripts\screenshot.ps1 -Out out\gui.png
#   .\scripts\screenshot.ps1 -Title "Fluid" -Out out\gui.png -Delay 2
#
# Needs a real desktop session (it is a normal window screenshot, no capture driver).
# The script marks itself DPI aware first: otherwise GetWindowRect returns virtualized
# coordinates on a 150%/200% display and the capture comes out half size (QA2 D8).

param(
    [string]$Title = "",
    [string]$Out = "out\gui.png",
    [double]$Delay = 1.0,
    [string]$ProcessName = "fluid-gui"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public struct FluidRect { public int Left, Top, Right, Bottom; }
public static class FluidWin {
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out FluidRect rect);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int cmd);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern IntPtr FindWindowW(string cls, string title);
}
"@

# Physical pixels, not virtualized ones: this has to run before any window query.
[void][FluidWin]::SetProcessDPIAware()

Start-Sleep -Seconds $Delay
$hwnd = [IntPtr]::Zero
$label = ""

if ($Title -ne "") {
    $hwnd = [FluidWin]::FindWindowW($null, $Title)
    if ($hwnd -ne [IntPtr]::Zero) { $label = $Title }
}

if ($hwnd -eq [IntPtr]::Zero -and $ProcessName -ne "") {
    $proc = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowHandle -ne 0 } |
        Select-Object -First 1
    if ($proc) {
        $hwnd = $proc.MainWindowHandle
        $label = $proc.MainWindowTitle
    }
}

if ($hwnd -eq [IntPtr]::Zero) {
    throw "no window found (title '$Title', process '$ProcessName'); is the app running?"
}

[void][FluidWin]::ShowWindow($hwnd, 9)   # SW_RESTORE
[void][FluidWin]::SetForegroundWindow($hwnd)
Start-Sleep -Milliseconds 400

$rect = New-Object FluidRect
if (-not [FluidWin]::GetWindowRect($hwnd, [ref]$rect)) {
    throw "GetWindowRect failed for window $hwnd"
}
$width = $rect.Right - $rect.Left
$height = $rect.Bottom - $rect.Top
if ($width -le 0 -or $height -le 0) {
    throw ("window has an empty size: {0}x{1}" -f $width, $height)
}

$dir = Split-Path -Parent $Out
if (-not $dir) { $dir = "." }
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

$file = Join-Path (Resolve-Path -LiteralPath $dir).Path (Split-Path -Leaf $Out)
$bmp = New-Object System.Drawing.Bitmap $width, $height
$gfx = [System.Drawing.Graphics]::FromImage($bmp)
$gfx.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bmp.Size)
$gfx.Dispose()
$bmp.Save($file, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()

Write-Output ("captured '{0}' ({1}x{2}) -> {3}" -f $label, $width, $height, $file)
