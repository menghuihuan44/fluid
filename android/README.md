# Fluid Android app shell

This Gradle project builds the Android app around the Rust platform layer:

* `MainActivity` asks for the MediaProjection permission, owns the `SurfaceView`
  used for output, and starts/stops the engine through `FluidNative`
  (`nativeStartEnhance` / `nativeStopEnhance`) while polling
  `nativeLastReport` for the status line;
* `ScreenCaptureService` keeps the projection alive as a foreground service and
  creates the screen-sized `ImageReader`;
* the native library (`libfluid_gui.so`) is built from `crates/fluid-gui` with
  `cargo-ndk` and provides the JNI methods declared in `FluidNative.kt`.

## Build

From the repository root:

```powershell
pwsh -File scripts/build-android.ps1              # native libs + debug APK
pwsh -File scripts/build-android.ps1 -Mode lib    # native library only
```

or let Gradle run Cargo itself (requires `cargo-ndk` + `ANDROID_NDK_HOME`):

```bash
./gradlew :app:assembleDebug
```

`gradle/wrapper/gradle-wrapper.jar` is a binary and is not committed. Generate it
with `gradle wrapper --gradle-version 8.7` inside this directory, or open the
folder in Android Studio and build from there.

Notes for a working build (verified 2026-11-08, producing a debug APK):

* use NDK 26.1.10909125 with `-P 26` (or `--platform 26`): NDK r26 has no
  `libnativewindow.so` for API 21, cargo-ndk's default;
* `android/local.properties` needs `sdk.dir=C:/Android` (or your SDK path);
* `gradlew.bat --console=plain :app:assembleDebug -Pfluid.skipCargo=true` after
  building the `.so` with `cargo ndk ... build -p fluid-gui --release`;
* the resulting `app/build/outputs/apk/debug/app-debug.apk` is 11,952,887 bytes and
  contains `lib/arm64-v8a/libfluid_gui.so` (11,129,144 bytes) with all nine JNI
  entry points. The app now starts the real capture -> engine -> present pipeline.
  No device was available, so runtime behaviour is still untested - see
  `docs/ANDROID.md`.

## Install

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.fluid.app/.MainActivity
```

The complete architecture, permission flow, native API and the list of
unverified points live in [`docs/ANDROID.md`](../docs/ANDROID.md).