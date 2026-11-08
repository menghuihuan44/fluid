# Build Fluid for Android: native library (cargo-ndk) and optionally the APK (Gradle).
#
# Usage (from the repository root):
#   pwsh -File scripts/build-android.ps1                    # native libs + debug APK
#   pwsh -File scripts/build-android.ps1 -Mode lib          # native libs only
#   pwsh -File scripts/build-android.ps1 -Profile release -Mode apk
#   pwsh -File scripts/build-android.ps1 -Targets arm64-v8a,armeabi-v7a
#
# Requirements when it actually builds:
#   * Rust target(s) for the Android ABI(s) (installed automatically via rustup)
#   * cargo-ndk                    (cargo install cargo-ndk)
#   * Android NDK                  (ANDROID_NDK_HOME, ANDROID_NDK_ROOT, or SDK/ndk/<ver>)
#   * Android SDK platform 34      (ANDROID_HOME or ANDROID_SDK_ROOT)      [APK only]
#   * JDK 17 or newer              (JAVA_HOME or java on PATH)             [APK only]
#   * Gradle: android/gradlew.bat or a "gradle" on PATH                    [APK only]
#
# Verified toolchain (this exact set produced a signed debug APK, see
# docs/ANDROID.md "Build an APK today"):
#   JDK 17 Temurin    C:\Program Files\Eclipse Adoptium\jdk-17.0.21.9-hotspot
#   Android SDK       C:\Android  (cmdline-tools;latest, platform-tools,
#                     platforms;android-34, build-tools;34.0.0,
#                     ndk;26.1.10909125)
#   cargo-ndk         4.1.2   (cargo install cargo-ndk --locked)
#   Gradle            8.7     (C:\Tools\gradle-8.7)
#
# Notes:
#   * The NDK points at API 26 (--platform 26 / -P 26): NDK r26 no longer ships
#     libnativewindow.so for API 21, which is cargo-ndk's default, so linking the
#     winit/ndk-sys stack fails there. minSdk is 26, so this costs nothing.
#   * The Gradle wrapper jar (gradle/wrapper/gradle-wrapper.jar) is a binary and
#     is not committed; run "gradle wrapper --gradle-version 8.7" inside android/
#     (or open the project in Android Studio once) to generate it.
param(
    [ValidateSet("apk", "lib")]
    [string]$Mode = "apk",
    [ValidateSet("debug", "release")]
    [string]$Profile = "debug",
    [string[]]$Targets = @("arm64-v8a"),
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$androidDir = Join-Path $root "android"
if (-not (Test-Path (Join-Path $androidDir "app\build.gradle.kts"))) {
    throw "android/app/build.gradle.kts not found under $androidDir; run this script from the repository"
}
if (-not $OutDir) { $OutDir = Join-Path $androidDir "app\src\main\jniLibs" }
elseif (-not [System.IO.Path]::IsPathRooted($OutDir)) { $OutDir = Join-Path $root $OutDir }

$rustTargets = @{
    "arm64-v8a"   = "aarch64-linux-android"
    "armeabi-v7a" = "armv7-linux-androideabi"
    "x86_64"      = "x86_64-linux-android"
}

function Fail([string]$message, [string[]]$hints) {
    Write-Host $message -ForegroundColor Red
    foreach ($hint in $hints) { Write-Host "  $hint" -ForegroundColor Yellow }
    exit 1
}

function Test-Command([string]$name) {
    return $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

# --- prerequisites: NDK + cargo-ndk ---------------------------------------
if (-not (Test-Command "cargo-ndk")) {
    Fail "cargo-ndk is not installed, so the Rust library cannot be cross-compiled." @(
        "cargo install cargo-ndk",
        "Then re-run this script."
    )
}

$ndk = $env:ANDROID_NDK_HOME
if (-not $ndk) { $ndk = $env:ANDROID_NDK_ROOT }
if (-not $ndk) {
    $sdk = $env:ANDROID_HOME
    if (-not $sdk) { $sdk = $env:ANDROID_SDK_ROOT }
    if ($sdk) {
        $ndkRoot = Join-Path $sdk "ndk"
        if (Test-Path $ndkRoot) {
            $latest = Get-ChildItem $ndkRoot -Directory |
                Sort-Object { [version](($_.Name -split "-")[0]) } -ErrorAction SilentlyContinue |
                Select-Object -Last 1
            if ($latest) { $ndk = $latest.FullName }
        }
    }
}
if (-not $ndk) {
    Fail "Android NDK not found." @(
        "Install the NDK from Android Studio (SDK Manager -> NDK (Side by side)),",
        "then set ANDROID_NDK_HOME to e.g. C:\Users\you\AppData\Local\Android\Sdk\ndk\26.1.10909125",
        "and re-run this script."
    )
}
$env:ANDROID_NDK_HOME = $ndk
Write-Host "NDK: $ndk"

# --- Rust targets ---------------------------------------------------------
$installed = @(& rustup target list --installed)
foreach ($abi in $Targets) {
    if (-not $rustTargets.ContainsKey($abi)) {
        Fail "unknown ABI '$abi' (expected arm64-v8a, armeabi-v7a or x86_64)" @()
    }
    $triple = $rustTargets[$abi]
    if ($installed -notcontains $triple) {
        Write-Host "installing Rust target $triple"
        & rustup target add $triple
        if ($LASTEXITCODE -ne 0) { Fail "rustup target add $triple failed" @() }
    }
}

# --- native library -------------------------------------------------------
Push-Location $root
try {
    $buildArgs = @()
    foreach ($abi in $Targets) { $buildArgs += @("-t", $abi) }
    # minSdk 26: NDK r26 has no libnativewindow.so for API 21 (cargo-ndk default).
    $buildArgs += @("-P", "26")
    # --lib: only build the cdylib (libfluid_gui.so); the desktop bin target is not
    # meant for Android and would just waste time here.
    $buildArgs += @("-o", $OutDir, "build", "--lib", "-p", "fluid-gui")
    if ($Profile -eq "release") { $buildArgs += "--release" }

    Write-Host "cargo ndk $($buildArgs -join ' ')"
    & cargo ndk @buildArgs
    if ($LASTEXITCODE -ne 0) { Fail "cargo ndk build failed" @() }

    foreach ($abi in $Targets) {
        $so = Join-Path $OutDir "$abi\libfluid_gui.so"
        if (-not (Test-Path $so)) { Fail "expected library $so was not produced" @() }
        Write-Host "built $so ($([math]::Round((Get-Item $so).Length / 1MB, 1)) MB)"
    }
}
finally {
    Pop-Location
}

if ($Mode -eq "lib") {
    Write-Host "native libraries written to $OutDir" -ForegroundColor Green
    exit 0
}

# --- APK (Gradle) ---------------------------------------------------------
if (-not $env:JAVA_HOME -and -not (Test-Command "java")) {
    Fail "No JDK found; Gradle needs JDK 17 or newer." @(
        "Install Android Studio (bundles a JDK) or a standalone JDK 17+,",
        "then set JAVA_HOME to its root and re-run this script."
    )
}
$javaMajor = 0
try {
    $versionText = (& java -version 2>&1 | Select-Object -First 1) -join " "
    if ($versionText -match '"([0-9]+)(\.([0-9]+))?') {
        $javaMajor = [int]$Matches[1]
        if ($javaMajor -eq 1 -and $Matches[3]) { $javaMajor = [int]$Matches[3] }
    }
} catch {
    $javaMajor = 0
}
if ($javaMajor -ne 0 -and $javaMajor -lt 17) {
    Fail "JDK $javaMajor found but Android Gradle Plugin 8.5 requires JDK 17+." @(
        "Point JAVA_HOME at a JDK 17 or newer (Android Studio -> bundled JDK)."
    )
}

$sdkHome = $env:ANDROID_HOME
if (-not $sdkHome) { $sdkHome = $env:ANDROID_SDK_ROOT }
if (-not $sdkHome) {
    Fail "Android SDK not found (ANDROID_HOME / ANDROID_SDK_ROOT are empty)." @(
        "Install Android Studio and its SDK, then set ANDROID_HOME to e.g.",
        "C:\Users\you\AppData\Local\Android\Sdk",
        "or open android/ in Android Studio and build there (Build > Build Bundle(s) / APK(s))."
    )
}
if (-not (Test-Path (Join-Path $sdkHome "platforms\android-34"))) {
    Write-Host "warning: $sdkHome\platforms\android-34 not found; install Android SDK Platform 34." -ForegroundColor Yellow
}

$gradlew = Join-Path $androidDir "gradlew.bat"
$gradle = $null
if (Test-Path $gradlew) {
    $gradle = $gradlew
} elseif (Test-Command "gradle") {
    $gradle = "gradle"
} else {
    Fail "No Gradle launcher found: android/gradlew.bat is missing and 'gradle' is not on PATH." @(
        "Easiest: open the android/ folder in Android Studio once; it generates the wrapper",
        "(android/gradle/wrapper/gradle-wrapper.jar) and can build the APK from the IDE.",
        "Command line alternative: install Gradle 8.7, run 'gradle wrapper --gradle-version 8.7'",
        "inside android/, then re-run this script."
    )
}

$task = if ($Profile -eq "release") { ":app:assembleRelease" } else { ":app:assembleDebug" }
Write-Host "Gradle $task (native libraries already built; -Pfluid.skipCargo=true)"
Push-Location $androidDir
try {
    if ($gradle -eq "gradle") {
        & gradle $task "-Pfluid.skipCargo=true"
    } else {
        & $gradle $task "-Pfluid.skipCargo=true"
    }
    if ($LASTEXITCODE -ne 0) { Fail "Gradle $task failed" @() }
}
finally {
    Pop-Location
}

$apkRoot = Join-Path $androidDir "app\build\outputs\apk"
$apks = @()
if (Test-Path $apkRoot) {
    $apks = @(Get-ChildItem $apkRoot -Recurse -Filter "*.apk" | Sort-Object LastWriteTime -Descending)
}
if ($apks.Count -eq 0) {
    Fail "Gradle finished but no APK was found under $apkRoot" @()
}
foreach ($apk in $apks) {
    $hash = (Get-FileHash $apk.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host "APK: $($apk.FullName) ($([math]::Round($apk.Length / 1MB, 1)) MB)" -ForegroundColor Green
    Write-Host "     sha256: $hash"
}
Write-Host ""
Write-Host "Install on a connected device:"
Write-Host ('    adb install -r "' + $apks[0].FullName + '"')
