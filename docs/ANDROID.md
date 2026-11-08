# Fluid on Android

Fluid on Android is split into two halves:

* a small **Kotlin app shell** in `android/` that owns the Activity, asks for the
  screen-capture permission, creates the `MediaProjection` virtual display and
  provides the `SurfaceView` used for output;
* the **Rust platform layer** (`fluid-capture`, `fluid-present`, plus the JNI glue in
  `crates/fluid-gui/src/android_jni.rs`) that turns an `ImageReader` into
  `Frame`s and copies `Frame`s into an `ANativeWindow`.

The engine, optical-flow, interpolation, upscaling and video crates are pure Rust
and are unchanged by the platform: the pipeline still sees
`FrameSource` -> `Frame` -> `FrameSink`.

> **Status: APK builds, is signed, and now drives the full engine; not yet run on
> a device.**
> On 2026-11-08 the full Android toolchain was installed outside the repository
> (JDK 17, SDK platform 34 + build-tools 34.0.0, NDK 26.1.10909125, cargo-ndk
> 4.1.2, Gradle 8.7) and a real debug APK was produced and inspected:
> `out/fluid-android-debug.apk` - 11,952,887 bytes, sha256
> `f4c95f214a2e3afaac9139256188d0efec497806e5d4b1dc58373dd2ce58a4f2`, signed with
> the Android debug key (v2 scheme), containing
> `lib/arm64-v8a/libfluid_gui.so` (11,129,144 bytes) with all nine JNI entry
> points below. The Kotlin shell no longer just installs handles: pressing
> 开始增强 starts the real capture -> engine -> present pipeline through
> `nativeStartEnhance`, and the status line is fed by `nativeLastReport`.
> What is still **not** verified: installing this APK on a phone, the system
> consent dialog on a real device, the JNI boundary on a real ART VM, the
> MediaProjection / ANativeWindow data paths, and on-device performance. There is
> no connected device here, so treat runtime behaviour as reviewed-by-inspection
> and compile-verified.

## Why Android is different

Desktop Fluid can open a capture API for the whole desktop. Android does not
allow that: screen capture requires a `MediaProjection` token that the *user*
grants through a system dialog, it can only be used while a foreground service
with `foregroundServiceType="mediaProjection"` is running, and the status bar
shows its own recording indicator. Presentation is an `ANativeWindow` (a Surface
from a `SurfaceView`), not a borderless top-level window.

Consequently the Android build cannot be a pure CLI: it needs the app shell,
and every capture session is explicitly authorised by the user.

## Architecture

```text
  MainActivity            ScreenCaptureService                     Rust
  ------------            ---------------------                     ----
  start button  --intent--> startForeground(mediaProjection)
  permission UI            getMediaProjection(resultCode, data)
  (system dialog)          ImageReader.newInstance(RGBA_8888, 3)
                           createVirtualDisplay(AUTO_MIRROR)
                           FluidNative.nativeInstallReader(reader) --> fluid_capture::android::install
                                                                       (global ref of ImageReader)
  SurfaceView.surfaceCreated
    FluidNative.nativeInstallSurface(surface) ---------------------> fluid_present::android::install_from_surface
                                                                       (ANativeWindow from Surface)

  pipeline: FrameSource (android-mediaprojection) -> engine -> FrameSink (android-anativewindow)

  start button --(both handles installed)--> FluidNative.nativeStartEnhance
                                               -> one dedicated thread:
                                                    fluid_capture::open
                                                    fluid_present::open
                                                    Pipeline::run
                                                    close both
  status poll (500 ms) --FluidNative.nativeLastReport--> {running,state,elapsed_s,error,report}
  stop button ---------FluidNative.nativeStopEnhance--> cancel flag (pipeline exits with a report)
```

Both directions are "install a handle, then open normally":

* `fluid_capture::open(&CaptureConfig)` returns the MediaProjection source once an
  `ImageReader` has been installed; `FrameSource::backend_name()` is
  `"android-mediaprojection"`.
* `fluid_present::open(&PresentConfig, dims)` returns the ANativeWindow sink once a
  `Surface` has been installed; `FrameSink::backend_name()` is
  `"android-anativewindow"`.

## Repository layout

```text
android/
  settings.gradle.kts          Gradle project (rootProject "fluid-android")
  build.gradle.kts             AGP 8.5.2 + Kotlin 1.9.24 (plugins only)
  gradle.properties
  gradle/wrapper/gradle-wrapper.properties   Gradle 8.7 (wrapper jar NOT committed)
  app/build.gradle.kts         compileSdk 34, minSdk 26, targetSdk 34, arm64-v8a,
                               jniLibs source dir, opens Cargo build via
                               :app:cargoNdkBuild (skippable with -Pfluid.skipCargo=true)
  app/src/main/AndroidManifest.xml
  app/src/main/java/com/fluid/app/MainActivity.kt
  app/src/main/java/com/fluid/app/ScreenCaptureService.kt
  app/src/main/java/com/fluid/app/FluidNative.kt
  app/src/main/res/layout/activity_main.xml
  app/src/main/res/values/strings.xml
crates/fluid-gui/src/android_jni.rs          JNI_OnLoad + the two native methods
crates/fluid-capture/src/android.rs          MediaProjection FrameSource
crates/fluid-present/src/android.rs          ANativeWindow FrameSink
scripts/build-android.ps1 | scripts/build-android.sh
```

## Native interface

### Exported JNI symbols (`crates/fluid-gui/src/android_jni.rs`)

| Symbol | Kotlin declaration | Notes |
| --- | --- | --- |
| `JNI_OnLoad` | (implicit) | Stores the `JavaVM*`, returns JNI 1.6 |
| `Java_com_fluid_app_FluidNative_nativeInstallReader` | `external fun nativeInstallReader(reader: ImageReader): Int` | Converts the local ref to a global ref and calls `fluid_capture::android::install` |
| `Java_com_fluid_app_FluidNative_nativeInstallSurface` | `external fun nativeInstallSurface(surface: Surface): Int` | Calls `fluid_present::android::install_from_surface` |
| `Java_com_fluid_app_FluidNative_nativeStartEnhance` | `external fun nativeStartEnhance(multiplier: Int, scale: Float, filter: String, quality: String): Int` | Validates the four UI selections, spawns the pipeline thread, returns immediately |
| `Java_com_fluid_app_FluidNative_nativeStopEnhance` | `external fun nativeStopEnhance(): Int` | Sets the cooperative cancel flag; the thread ends and publishes a report |
| `Java_com_fluid_app_FluidNative_nativeIsRunning` | `external fun nativeIsRunning(): Int` | 1 while the pipeline thread is alive, 0 otherwise |
| `Java_com_fluid_app_FluidNative_nativeLastReport` | `external fun nativeLastReport(): String?` | JSON status document, see below |
| `Java_com_fluid_app_FluidNative_nativeUninstallReader` | `external fun nativeUninstallReader(): Int` | Drops the ImageReader global ref after a session ended |
| `Java_com_fluid_app_FluidNative_nativeUninstallSurface` | `external fun nativeUninstallSurface(): Int` | Releases the ANativeWindow (after the pipeline stopped) |

Every method except `nativeLastReport` returns `0` on success and a negative code
otherwise; the shell maps them to user-facing text (`FluidNative.describe`).
Nothing throws across JNI.

| Code | Constant | Meaning |
| --- | --- | --- |
| 0 | `FluidNative.OK` | installed |
| -1 | `ERROR_NO_JAVA_VM` | `JNI_OnLoad` never ran (library loaded by an unusual path) |
| -2 | `ERROR_BAD_ARGUMENT` | null `ImageReader`/`Surface`/`JNIEnv` |
| -3 | `ERROR_JNI` | creating the global reference failed (e.g. JVM out of local refs) |
| -4 | `ERROR_INSTALL` | the Rust backend rejected the handle (see docs on the `install` functions) |
| -5 | `ERROR_BUSY` | `nativeStartEnhance` while a pipeline thread is still alive |
| -6 | `ERROR_PIPELINE` | the pipeline thread could not be spawned, or the config was rejected |
| -100 | Kotlin-only `ERROR_NATIVE_LIBRARY` | `UnsatisfiedLinkError`: the `.so` was not packaged/loaded |

### Status document (`nativeLastReport`)

```json
{"running": true, "state": "running", "elapsed_s": 12.345,
 "error": null,
 "report": {"source": "android-mediaprojection", "sink": "android-anativewindow",
            "out": [1920, 1080], "output_fps": 58.2, "interp_backend": "cpu",
            "backend": "cpu", "...": "PipelineReport fields"}}
```

* `state` is `idle` (never started or still starting), `running`, `finished`
  (`report` is the last `PipelineReport` JSON) or `error` (`error` is a message).
* While `running` the report is `null`; `elapsed_s` counts the current run.
* The string is built by hand (no serde on the phone) and can be `null` only if
  the JNI String allocation fails.

### Pipeline control semantics

* One run at a time. `nativeStartEnhance` returns `ERROR_BUSY` instead of
  queueing a second run; the Kotlin shell must stop first.
* The thread opens capture, opens the presenter, runs the pipeline and closes
  both on the same thread - required because both Android backends are
  thread-affine. `nativeStopEnhance` only flips an `AtomicBool`; the loop checks
  it once per frame, returns the partial report and the thread exits.
* `nativeStartEnhance` never blocks on capture: a missing ImageReader/ANativeWindow
  surfaces later through `nativeLastReport` as `state:"error"`.
* `elapsed_s` and the report are kept after the thread ends, so the UI can keep
  showing the last result; `nativeStartEnhance` resets them on the next run.

### Rust-side entry points

```rust
// fluid-capture
pub type ImageReaderRef = jni::refs::Global<jni::objects::JObject<'static>>;
pub fn install(jvm: *mut jni::sys::JavaVM, image_reader: ImageReaderRef) -> Result<()>;
pub fn is_installed() -> bool;
pub fn uninstall();

// fluid-present
pub fn install(window: *mut ndk_sys::ANativeWindow) -> Result<()>;
pub unsafe fn install_from_surface(env: *mut jni::sys::JNIEnv, surface: jni::sys::jobject) -> Result<()>;
pub fn is_installed() -> bool;
pub fn uninstall();
pub fn request_close();
pub fn clear_close_request();
```

`install*` takes ownership of a reference (global ref / acquired ANativeWindow)
and replaces any previous installation.

## Permission flow (implemented in the shell)

1. The user taps **开始屏幕捕获** in `MainActivity`.
2. `MediaProjectionManager.createScreenCaptureIntent()` is launched with
   `startActivityForResult`; Android shows the system consent dialog.
3. On `RESULT_OK`, `MainActivity` starts `ScreenCaptureService` as a foreground
   service, passing `resultCode` and the projection `Intent`.
4. The service calls `startForeground(...)` **before** touching the token
   (mandatory on Android 10+, strictly enforced on 14), then
   `getMediaProjection(resultCode, data)`.
5. It creates a screen-sized `ImageReader` (`PixelFormat.RGBA_8888`, 3 buffers), a
   `VirtualDisplay` with `VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR`, registers a
   `MediaProjection.Callback`, and calls
   `FluidNative.nativeInstallReader(reader)`.
6. When the `SurfaceView`'s surface is created, `MainActivity` calls
   `FluidNative.nativeInstallSurface(holder.surface)`.
7. With the reader and the surface both installed, `MainActivity` calls
   `FluidNative.nativeStartEnhance(multiplier, scale, filter, quality)` and starts
   polling `nativeLastReport` every 500 ms.

The mapping from the three RadioGroups is fixed in `MainActivity.beginEnhance`:

| UI | JNI argument | Engine setting |
| --- | --- | --- |
| 插帧倍率 关 / 2x / 3x | `multiplier` 1 / 2 / 3 | `interpolator.mode` Off / Fixed, `multiplier` |
| 分辨率提升 关 / 1.5x / 2x | `scale` 1.0 / 1.5 / 2.0 | `upscaler.scale` factor |
| 画质 流畅 / 均衡 / 高画质 | `quality` `ultra-performance` / `balanced` / `quality` | `FluidConfig::apply_preset` |
| (derived) | `filter` `off` / `bilinear` / `adaptive` / `bicubic` | `upscaler.filter` |

`nativeStartEnhance` also pins `latency.max_queue = 1` and
`present.vsync = false` (a SurfaceView has no DWM-style vsync flush to honour).

Stopping: the 停止增强 button, the notification action, a system-side revocation
(`MediaProjection.Callback.onStop`) or the activity leaving the foreground all
end up in `releaseSession()`: set the cancel flag, release the virtual display,
close the `ImageReader`, then after a 300 ms grace period
`nativeUninstallReader()` runs and the service stops itself. `surfaceDestroyed`
stops the pipeline first and only then calls `nativeUninstallSurface()`, because
the presenter reads the ANativeWindow on every frame. The token is single-use - a
new session needs a new consent dialog.

## Build

### Prerequisites

| Tool | Verified version | Needed for |
| --- | --- | --- |
| Rust Android target | `aarch64-linux-android` | native library (the script installs it) |
| `cargo-ndk` | 4.1.2 (`cargo install cargo-ndk --locked`) | native library |
| Android NDK | 26.1.10909125 (side by side) | native library; set `ANDROID_NDK_HOME` |
| Android SDK | platform 34 (ext 7, rev 3), build-tools 34.0.0, platform-tools | APK |
| JDK | 17 (Temurin 17.0.21 used) | APK (AGP 8.5 requirement) |
| Gradle | 8.7 (wrapper, generated once) | APK |

The layout used for the verified build (all outside the repository):

```text
C:/Android/                       ANDROID_HOME
  cmdline-tools/latest/bin/sdkmanager.bat
  platforms/android-34
  build-tools/34.0.0              aapt2.exe, apksigner.bat, zipalign.exe
  ndk/26.1.10909125               ANDROID_NDK_HOME
  platform-tools                  adb.exe
C:/Tools/gradle-8.7               gradle wrapper --gradle-version 8.7
C:/Program Files/Eclipse Adoptium/jdk-17.0.21.9-hotspot   JAVA_HOME
```

`android/local.properties` must contain one line: `sdk.dir=C:/Android` (Gradle
otherwise only looks at `ANDROID_HOME`).

### Native library only

```powershell
# from the repository root; builds android/app/src/main/jniLibs/arm64-v8a/libfluid_gui.so
pwsh -File scripts/build-android.ps1 -Mode lib

# equivalent raw command (-P 26 is required, see the note below)
cargo ndk -t arm64-v8a -P 26 -o android/app/src/main/jniLibs build --release --lib -p fluid-gui
```

```bash
./scripts/build-android.sh lib
```

Use `--release`: debug flow estimation is roughly an order of magnitude slower and
will not reach anything near real time.

### APK

```powershell
pwsh -File scripts/build-android.ps1                 # native libs + assembleDebug
pwsh -File scripts/build-android.ps1 -Profile release
```

The script builds the native library into `app/src/main/jniLibs` first and passes
`-Pfluid.skipCargo=true` to Gradle, so Gradle itself does not need `cargo-ndk`.

Without the script, Gradle can do everything (it runs `cargo ndk` when
`ANDROID_NDK_HOME` is set and `-Pfluid.skipCargo` is absent):

```bash
cd android
./gradlew :app:assembleDebug          # app/build/outputs/apk/debug/app-debug.apk
```

**API level note:** `-P 26` (`--platform 26`) is mandatory with NDK r26.
cargo-ndk defaults to API 21, and NDK r26 no longer ships
`libnativewindow.so` for that level, so linking the winit/ndk-sys stack fails with
`ld.lld: error: unable to find library -lnativewindow`. `minSdk` is 26, so
targeting API 26 costs nothing. The Gradle `cargoNdkBuild` task already passes
`-P 26`, and both build scripts do the same.

**Wrapper note:** `gradle/wrapper/gradle-wrapper.jar` is a binary and is not
committed. Generate it once with `gradle wrapper --gradle-version 8.7` inside
`android/`, or just open `android/` in Android Studio (which can build the APK
without a command-line Gradle installation).

### Verified build (commands that produced the APK)

These are the exact steps that worked on a clean Windows 11 machine, with the
toolchain paths from the table above (`ANDROID_HOME=C:\Android`,
`ANDROID_NDK_HOME=C:\Android\ndk\26.1.10909125`,
`JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.21.9-hotspot`,
`C:\Users\<you>\.cargo\bin` on `PATH` for `cargo-ndk`):

```powershell
# 1. cross-compile the Rust cdylib (3m07s cold for arm64-v8a, ~20 s incremental)
cargo ndk -t arm64-v8a -P 26 -o android\app\src\main\jniLibs build -p fluid-gui --release
#    -> android\app\src\main\jniLibs\arm64-v8a\libfluid_gui.so (~11.1 MB; the exact
#       size moves with the source, it is a release cdylib)

# 2. assemble the debug APK (Gradle 8.7 wrapper, ~40 s; native lib already built)
cd android
gradlew.bat --console=plain clean :app:assembleDebug -Pfluid.skipCargo=true
#    -> app\build\outputs\apk\debug\app-debug.apk (~12 MB)
#    `clean` is recommended: an incremental APK can keep stale zero padding from
#    the previous native library and grow past 20 MB with identical content.

# 3. inspect the APK
C:\Android\build-tools\34.0.0\aapt2.exe dump badging app\build\outputs\apk\debug\app-debug.apk
C:\Android\build-tools\34.0.0\apksigner.bat verify --print-certs --verbose app\build\outputs\apk\debug\app-debug.apk
```

Or just run the helper script, which does all three steps and prints the sha256:

```powershell
pwsh -File scripts/build-android.ps1            # lib + debug APK
pwsh -File scripts/build-android.ps1 -Mode lib  # only the .so
```

What the produced APK actually contains (checked with `aapt2`, `apksigner` and a
zip listing):

| Item | Value |
| --- | --- |
| File | `out/fluid-android-debug.apk` (also `android/app/build/outputs/apk/debug/app-debug.apk`) |
| Size / sha256 | 11,952,887 bytes / `f4c95f214a2e3afaac9139256188d0efec497806e5d4b1dc58373dd2ce58a4f2` |
| Package | `com.fluid.app`, versionCode 1, versionName 0.1.0 |
| SDK levels | minSdk 26, targetSdk 34, compileSdk 34 |
| Permissions | `FOREGROUND_SERVICE`, `FOREGROUND_SERVICE_MEDIA_PROJECTION`, `POST_NOTIFICATIONS` |
| Components | activity `com.fluid.app.MainActivity` (launcher), service `com.fluid.app.ScreenCaptureService` |
| Native code | `arm64-v8a` only; `lib/arm64-v8a/libfluid_gui.so` 11,129,144 bytes, stored uncompressed (`jniLibs.useLegacyPackaging=false`) |
| JNI symbols | `JNI_OnLoad` + `nativeInstallReader`, `nativeInstallSurface`, `nativeStartEnhance`, `nativeStopEnhance`, `nativeIsRunning`, `nativeLastReport`, `nativeUninstallReader`, `nativeUninstallSurface` (checked with `llvm-nm -D` on the packaged `.so`) |
| Signing | debug certificate `CN=Android Debug` (SHA-256 `1577636e...3bee90`), APK Signature Scheme v2 only |

### Install and run

```bash
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.fluid.app/.MainActivity
adb logcat -s FluidMain FluidCapture   # shell-side logs
```

On Android 13+ the app also asks for `POST_NOTIFICATIONS` so the mandatory
foreground-service notification is visible.

## Runtime behaviour to know

* Capture always produces `Rgba8` frames; the width/height are the display size at
  the moment the `ImageReader` was created. Rotation or a display-size change
  needs a reopened capture session (the shell holds the activity in
  `configChanges=orientation|screenSize|...`, so rotating does not recreate it -
  see the gaps below).
* `next_frame` never returns `Ok(None)` while the source is open: if no new image
  is available within `capture.timeout_ms` the previous frame is repeated with a
  fresh `pts_ns`/`index` (same behaviour as the DXGI backend).
* `capture.region` is interpreted relative to the captured display and is clipped
  to it, like the desktop backends.
* Presentation uses `ANativeWindow_setBuffersGeometry(RGBA_8888)` +
  `ANativeWindow_lock` / copy / `ANativeWindow_unlockAndPost`; frames may have any
  pixel format the pipeline emits (`Rgba8`, `Bgra8`, `Rgb8`, `Luma8` are
  converted).
* `PresentConfig.vsync`, `fullscreen`, `borderless`, `title` and `monitor` have no
  effect on Android (the SurfaceView decides geometry); `low_latency` is accepted
  and ignored.
* Neither backend is a fan of being used from two threads at once: `open`, the
  frame loop and `close` all happen on one dedicated pipeline thread started by
  `nativeStartEnhance`.
* The pipeline runs until the user stops it; there is no frame limit. Stopping is
  cooperative: the current frame finishes, `Pipeline::run` returns a report with
  the frames processed so far, and the thread closes the source and the sink.
* The Activity only polls once per 500 ms and never joins the thread, so a slow
  phone frame cannot block the UI. Buttons are enabled/disabled from the JSON
  status, not from optimistic local state.
* If the screen is locked or the projection is revoked the pipeline keeps
  replaying the last frame (same contract as the desktop backends); the Activity
  stops on lifecycle events.

## Recommended Android defaults

Mobile SoCs are power- and thermal-limited, so the goal is usually "720p at 2x
interpolation" rather than "1080p 2x + 1.5x upscale":

```toml
[upscaler]
filter = "bilinear"
sharpness = 0.2

[upscaler.scale]
mode = "factor"
factor = 1.0

[upscaler.temporal]
enabled = false

[interpolator]
mode = "fixed"
multiplier = 2
quality = "ultra-performance"

[interpolator.flow]
pyramid_levels = 2
block_size = 16
search_radius = 4
refine_iters = 1

[latency]
max_queue = 1
pacing = "unlimited"
```

Cap the output with `--fps-cap` at the panel refresh rate and leave temporal
upscaling off (it keeps history and costs memory bandwidth).

## Measuring on a device

```bash
# bench binary for the phone (separate from the APK)
cargo ndk -t arm64-v8a build --release -p fluid-cli
adb push target/aarch64-linux-android/release/fluid /data/local/tmp/fluid
adb shell chmod +x /data/local/tmp/fluid
adb shell /data/local/tmp/fluid bench --size 1280x720 --frames 120 --json
```

Do this on the target device: latency, thermals and the panel refresh rate all
differ from any desktop or emulator measurement.

## Known gaps (honest list)

The APK builds, is signed, and calls the real pipeline, but no Android device was
available, so the following are explicitly **unverified**:

* Installing and launching the APK on a phone (`adb install` has not been run).
* The JNI boundary on a real ART VM (symbol resolution, `JNI_OnLoad`, global-ref
  lifetime, the return codes, the hand-built status JSON).
* The Kotlin start/stop chain on a real device: consent dialog, foreground-service
  timing, the 500 ms status poll, button state transitions.
* `ImageReader` pacing, the `AHardwareBuffer` lock path, and the
  `ANativeWindow_lockAndPost` path on real GPUs.
* Performance of the pipeline on a phone (see the measurement section), including
  thermal throttling during long runs.
* The Gradle `cargoNdkBuild` task that drives `cargo ndk` from inside Gradle
  (the verified build always passed `-Pfluid.skipCargo=true` and built the
  library separately; the two halves are known to work, their wiring is not
  exercised yet).
* Release builds (`assembleRelease`): the verified APK is a debug-signed
  `assembleDebug` build.
* The UI layout on real phone sizes/cutouts: it is a plain `LinearLayout` with
  48 dp controls and API 30+ window-insets padding, but no device screenshot exists.

Known design gaps:

* Only `arm64-v8a` is configured (`abiFilters`); `armeabi-v7a`/`x86_64` need both
  the Gradle ABI list and the script targets to change.
* Rotation: the virtual display and `ImageReader` are created at the display size
  at start time, and the activity handles rotation without being recreated, so a
  rotated session keeps its original dimensions. A production build should
  reinstall the reader/display on configuration change.
* Audio is not captured; only the video frames are.
* `POST_NOTIFICATIONS` denial is tolerated (notification silently absent) rather
  than reported.
* The Gradle wrapper jar is not committed (see the build section).

## Status and roadmap

- [x] All non-platform crates compile for `aarch64-linux-android` (pure Rust).
- [x] `MediaProjectionSource`: `ImageReader` -> RGBA `Frame` (JNI install).
- [x] `ANativeWindowSink`: RGBA `Frame` -> ANativeWindow lock/unlockAndPost.
- [x] Kotlin app shell: permission flow, foreground service, start/stop, SurfaceView.
- [x] JNI bridge: `JNI_OnLoad` + install-reader/install-surface with status codes.
- [x] Gradle project that packages the native library (APK build path).
- [x] Real debug APK built and statically verified (aapt2 / apksigner / zip list),
      `libfluid_gui.so` packaged for arm64-v8a.
- [x] `fluidRun` equivalent: `nativeStartEnhance` / `nativeStopEnhance` /
      `nativeIsRunning` / `nativeLastReport` plus uninstall wrappers, driving the
      real engine on one dedicated thread.
- [ ] Device verification: APK install, permission grant, visible interpolated output.
- [ ] Rotation/session reinstall, 32-bit/x86_64 ABIs.
- [ ] Vulkan compute path for mid-range/high-end phones (shared with desktop).