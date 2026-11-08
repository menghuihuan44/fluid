# Fluid 打包脚本（Windows x64 便携版）
#
# 用法：
#   .\scripts\package.ps1                 # 构建 release 并打包到 out\fluid-<version>-win64.zip
#   .\scripts\package.ps1 -SkipBuild      # 复用已有 target\release 二进制
#
# 产物是纯便携目录：解压即用，不需要安装、不需要管理员权限、不写注册表。
[CmdletBinding()]
param(
    [string]$Configuration = 'release',
    [string]$OutDir = 'out',
    [string]$BinDir = '',
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$version = '0.1.0'
$manifest = Join-Path $root 'Cargo.toml'
if (Test-Path $manifest) {
    $match = Select-String -Path $manifest -Pattern '^version = ["'']([0-9.]+)["'']' | Select-Object -First 1
    if ($match -and $match.Matches.Count -gt 0) { $version = $match.Matches[0].Groups[1].Value }
}

if (-not $SkipBuild) {
    Write-Host '==> cargo build --release -p fluid-cli -p fluid-gui'
    cargo build --release -p fluid-cli -p fluid-gui
    if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' }
}

$binDir = if ($BinDir) {
    if ([System.IO.Path]::IsPathRooted($BinDir)) { $BinDir } else { [System.IO.Path]::GetFullPath((Join-Path $root $BinDir)) }
} else {
    Join-Path $root (Join-Path 'target' $Configuration)
}
$exeFluid = Join-Path $binDir 'fluid.exe'
$exeGui = Join-Path $binDir 'fluid-gui.exe'
foreach ($exe in @($exeFluid, $exeGui)) {
    if (-not (Test-Path $exe)) { throw "missing binary: $exe (run cargo build --release first)" }
}

$stage = Join-Path $root (Join-Path $OutDir "fluid-$version-win64")
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Path $stage -Force | Out-Null
Copy-Item $exeFluid $stage
Copy-Item $exeGui $stage
# Android debug APK（可选）：同一套 Rust 代码交叉编译出来的手机版。
$apk = Join-Path $root (Join-Path $OutDir 'fluid-android-debug.apk')
if (Test-Path $apk) { Copy-Item $apk (Join-Path $stage 'fluid-android-debug.apk') }
Copy-Item (Join-Path $root 'README.md') $stage
if (Test-Path (Join-Path $root 'LICENSE')) { Copy-Item (Join-Path $root 'LICENSE') $stage }
# 文档与图片：README 里用的是 docs/... 相对路径，包内保持同样的目录结构。
$docsDir = Join-Path $root 'docs'
if (Test-Path $docsDir) {
    New-Item -ItemType Directory -Path (Join-Path $stage 'docs') -Force | Out-Null
    Copy-Item (Join-Path $docsDir '*.md') (Join-Path $stage 'docs') -ErrorAction SilentlyContinue
    $imgDir = Join-Path $docsDir 'images'
    if (Test-Path $imgDir) {
        New-Item -ItemType Directory -Path (Join-Path $stage 'docs\images') -Force | Out-Null
        Copy-Item (Join-Path $imgDir '*.png') (Join-Path $stage 'docs\images') -ErrorAction SilentlyContinue
    }
}
$scriptsDir = Join-Path $root 'scripts'
if (Test-Path $scriptsDir) {
    New-Item -ItemType Directory -Path (Join-Path $stage 'scripts') -Force | Out-Null
    Copy-Item (Join-Path $scriptsDir 'video-enhance.ps1') (Join-Path $stage 'scripts') -ErrorAction SilentlyContinue
    Copy-Item (Join-Path $scriptsDir 'bench.ps1') (Join-Path $stage 'scripts') -ErrorAction SilentlyContinue
}

$startText = @(
    'Fluid 便携版（Windows x64）',
    '==========================',
    '',
    '解压后直接双击 fluid-gui.exe 即可（中文图形界面，新手推荐）。命令行用法见 README.md。',
    '',
    '第一次使用建议：图形界面 -> 实时增强 -> 画面来源选“内置测试画面” -> 点“开始增强”，',
    '确认能看到插帧效果后，再切到“窗口捕获”选游戏窗口（比全屏省很多），或者“屏幕捕获”捕获整个桌面。',
    '放大算法建议 adaptive（默认），机器弱就选 bilinear；计算后端保持“自动”即可，有显卡会自动用 GPU。',
    '',
    '注意：',
    '- 本程序不注入游戏进程、不改游戏文件、不读游戏内存，只是一个屏幕捕获 + 处理 + 呈现的外部工具。',
    '- 但它仍然是一个“额外的屏幕捕获程序”，请自行遵守游戏的服务条款。',
    '- 需要 Windows 10/11 x64；若提示缺少 VCRUNTIME140.dll，安装微软 VC++ 运行库（x64）即可。',
    '',
    'Android：压缩包里附带 fluid-android-debug.apk（arm64-v8a，debug 签名，minSdk 26），',
    'adb install fluid-android-debug.apk 即可安装；手机端用法见 docs/ANDROID.md（开发机无真机，未实机验证）。'
)
Set-Content -Path (Join-Path $stage '启动说明.txt') -Value $startText -Encoding UTF8

$zip = Join-Path $root (Join-Path $OutDir "fluid-$version-win64.zip")
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal
$sizeMb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host "packed: $zip ($sizeMb MB)"
Write-Host "stage:  $stage"
