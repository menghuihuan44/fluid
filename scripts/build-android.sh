#!/usr/bin/env bash
# Build Fluid for Android: native library (cargo-ndk) and optionally the APK (Gradle).
#
# Usage (from the repository root):
#   ./scripts/build-android.sh                 # native libs + debug APK
#   ./scripts/build-android.sh lib             # native libs only
#   ./scripts/build-android.sh apk release     # release APK
#   ./scripts/build-android.sh lib debug arm64-v8a,armeabi-v7a
#
# Requirements when it actually builds: rustup Android targets, cargo-ndk,
# Android NDK (ANDROID_NDK_HOME / ANDROID_NDK_ROOT / SDK ndk/<ver>), and for the
# APK also Android SDK platform 34, JDK 17+ and Gradle (android/gradlew or a
# "gradle" on PATH). The Gradle wrapper jar is not committed; run
# "gradle wrapper --gradle-version 8.7" in android/ or open the project in
# Android Studio once.
#
# Verified toolchain (this set produced a signed debug APK; see docs/ANDROID.md):
#   JDK 17 Temurin, Android SDK with cmdline-tools/platform-tools/android-34/
#   build-tools 34.0.0/NDK 26.1.10909125, cargo-ndk 4.1.2, Gradle 8.7.
# The NDK is pointed at API 26 (-P 26) because NDK r26 has no libnativewindow.so
# for API 21 (cargo-ndk's default); minSdk is 26 so this costs nothing.
set -euo pipefail

MODE="${1:-apk}"
PROFILE="${2:-debug}"
ABIS="${3:-arm64-v8a}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANDROID_DIR="${ROOT}/android"
JNI_LIBS="${JNI_LIBS:-${ANDROID_DIR}/app/src/main/jniLibs}"

case "${MODE}" in
    apk|lib) ;;
    *) echo "unknown mode ${MODE} (expected apk or lib)" >&2; exit 1 ;;
esac
case "${PROFILE}" in
    debug|release) ;;
    *) echo "unknown profile ${PROFILE} (expected debug or release)" >&2; exit 1 ;;
esac

fail() {
    echo "error: $1" >&2
    shift || true
    for hint in "$@"; do echo "  $hint" >&2; done
    exit 1
}

[ -f "${ANDROID_DIR}/app/build.gradle.kts" ] || fail "android/app/build.gradle.kts not found; run from the repository"

command -v cargo-ndk >/dev/null 2>&1 || fail "cargo-ndk is not installed." "cargo install cargo-ndk"

NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}"
if [ -z "${NDK}" ]; then
    SDK="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
    if [ -n "${SDK}" ] && [ -d "${SDK}/ndk" ]; then
        NDK="$(ls -1d "${SDK}"/ndk/*/ 2>/dev/null | sort -V | tail -n 1)"
        NDK="${NDK%/}"
    fi
fi
[ -n "${NDK}" ] || fail "Android NDK not found." \
    "Install the NDK from Android Studio (SDK Manager), then set ANDROID_NDK_HOME."
export ANDROID_NDK_HOME="${NDK}"
echo "NDK: ${NDK}"

IFS=, read -r -a ABI_LIST <<< "${ABIS}"
ARGS=()
for abi in "${ABI_LIST[@]}"; do
    case "${abi}" in
        arm64-v8a)   triple=aarch64-linux-android ;;
        armeabi-v7a) triple=armv7-linux-androideabi ;;
        x86_64)      triple=x86_64-linux-android ;;
        *) fail "unknown ABI ${abi} (expected arm64-v8a, armeabi-v7a or x86_64)" ;;
    esac
    if ! rustup target list --installed | grep -qx "${triple}"; then
        echo "installing Rust target ${triple}"
        rustup target add "${triple}"
    fi
    ARGS+=(-t "${abi}")
done

cd "${ROOT}"
# --lib: only build the cdylib (libfluid_gui.so); the desktop bin target is not
# meant for Android and would just waste time here.
# -P 26: minSdk 26; NDK r26 has no libnativewindow.so for API 21 (cargo-ndk default).
BUILD_ARGS=("${ARGS[@]}" -P 26 -o "${JNI_LIBS}" build --lib -p fluid-gui)
if [ "${PROFILE}" = "release" ]; then BUILD_ARGS+=(--release); fi
echo "cargo ndk ${BUILD_ARGS[*]}"
cargo ndk "${BUILD_ARGS[@]}"
for abi in "${ABI_LIST[@]}"; do
    so="${JNI_LIBS}/${abi}/libfluid_gui.so"
    [ -f "${so}" ] || fail "expected library ${so} was not produced"
    echo "built ${so}"
done

if [ "${MODE}" = "lib" ]; then
    echo "native libraries written to ${JNI_LIBS}"
    exit 0
fi

command -v java >/dev/null 2>&1 || fail "No JDK found; Gradle needs JDK 17+." "Set JAVA_HOME to a JDK 17+."
JAVA_MAJOR="$(java -version 2>&1 | head -n 1 | sed -E 's/.*"([0-9]+)(\.([0-9]+))?.*/\1/')"
if [ -n "${JAVA_MAJOR}" ] && [ "${JAVA_MAJOR}" != "1" ] && [ "${JAVA_MAJOR}" -lt 17 ]; then
    fail "JDK ${JAVA_MAJOR} found but Android Gradle Plugin 8.5 requires JDK 17+." \
        "Set JAVA_HOME to a JDK 17+."
fi

SDK="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
[ -n "${SDK}" ] || fail "Android SDK not found (ANDROID_HOME / ANDROID_SDK_ROOT are empty)." \
    "Install Android Studio, or open android/ in Android Studio and build the APK there."
[ -d "${SDK}/platforms/android-34" ] || \
    echo "warning: ${SDK}/platforms/android-34 not found; install Android SDK Platform 34" >&2

if [ -x "${ANDROID_DIR}/gradlew" ]; then
    GRADLE="${ANDROID_DIR}/gradlew"
elif command -v gradle >/dev/null 2>&1; then
    GRADLE="gradle"
else
    fail "No Gradle launcher found: android/gradlew is missing and gradle is not on PATH." \
        "Easiest: open android/ in Android Studio once (it generates the wrapper and builds the APK)." \
        "Alternative: install Gradle 8.7, run gradle wrapper --gradle-version 8.7 in android/."
fi

TASK=":app:assembleDebug"
if [ "${PROFILE}" = "release" ]; then TASK=":app:assembleRelease"; fi
echo "Gradle ${TASK} (native libraries already built; -Pfluid.skipCargo=true)"
(
    cd "${ANDROID_DIR}"
    if [ "${GRADLE}" = "gradle" ]; then
        gradle "${TASK}" -Pfluid.skipCargo=true
    else
        "${GRADLE}" "${TASK}" -Pfluid.skipCargo=true
    fi
)

APK="$(find "${ANDROID_DIR}/app/build/outputs/apk" -name "*.apk" -print 2>/dev/null | head -n 1 || true)"
[ -n "${APK}" ] || fail "Gradle finished but no APK was found under android/app/build/outputs/apk"
echo "APK: ${APK}"
if command -v sha256sum >/dev/null 2>&1; then echo "     sha256: $(sha256sum "${APK}" | cut -d' ' -f1)"; fi
echo ""
echo "Install on a connected device:"
echo "    adb install -r ${APK}"
