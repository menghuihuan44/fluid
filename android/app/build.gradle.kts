// Fluid Android application module.
//
// The Kotlin code here is only platform glue around the Rust pipeline:
// MediaProjection -> ImageReader -> fluid_capture::android::install, and
// SurfaceView -> fluid_present::android::install_from_surface. Frame
// generation, interpolation and upscaling all stay in Rust.
import java.io.ByteArrayOutputStream
import org.gradle.api.GradleException

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.fluid.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.fluid.app"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"

        // Phones only for now: 64-bit ARM covers every device from 2016 on and
        // keeps the APK at a single native library.
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // The glue uses no reflection or serialization; the default ProGuard
            // file is enough if shrinking is enabled later.
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    sourceSets["main"].jniLibs.srcDirs("src/main/jniLibs")

    packaging {
        jniLibs {
            // Uncompressed native libraries load without an inflate step, which
            // keeps the first frame after installing the APK fast.
            useLegacyPackaging = false
        }
    }
}

dependencies {
    // Intentionally empty: the shell uses only android.* framework classes.
    // AndroidX would require android.useAndroidX=true and is not needed.
}

// ---------------------------------------------------------------------------
// cargo-ndk integration
//
// The Rust side is cross-compiled with cargo-ndk into app/src/main/jniLibs,
// which Gradle then packages into the APK. This requires:
//   * cargo + cargo-ndk  (cargo install cargo-ndk)
//   * ANDROID_NDK_HOME or ANDROID_NDK_ROOT pointing at an installed NDK
//   * rustup target add aarch64-linux-android
//
// The task is wired into jniLibs merging only, so IDE sync, ./gradlew tasks and
// unit-test runs keep working on machines without the NDK.
val repoRoot: File = rootProject.projectDir.parentFile
val jniLibsDir: File = file("src/main/jniLibs")
val skipCargo: Boolean = providers.gradleProperty("fluid.skipCargo").orNull == "true"

val cargoNdkBuild = tasks.register<Exec>("cargoNdkBuild") {
    group = "build"
    description = "Cross-compile the Fluid Rust GUI for arm64-v8a with cargo-ndk."
    workingDir = repoRoot

    doFirst {
        val ndk = System.getenv("ANDROID_NDK_HOME") ?: System.getenv("ANDROID_NDK_ROOT")
        if (ndk.isNullOrBlank()) {
            throw GradleException(
                "ANDROID_NDK_HOME / ANDROID_NDK_ROOT is not set. Install 'NDK (Side by side)' " +
                    "in Android Studio's SDK Manager and point the variable at <sdk>/ndk/<version>. " +
                    "Pass -Pfluid.skipCargo=true to build a Kotlin-only APK instead.",
            )
        }
        val version = try {
            val output = ByteArrayOutputStream()
            exec {
                commandLine("cargo", "ndk", "--version")
                standardOutput = output
            }
            output.toString().trim()
        } catch (error: Exception) {
            throw GradleException(
                "cargo-ndk is not installed (cargo ndk --version failed: " +
                    error.message + "). Install it with: cargo install cargo-ndk",
            )
        }
        logger.lifecycle("cargo-ndk: " + version)
    }

    commandLine(
        "cargo",
        "ndk",
        "-t",
        "arm64-v8a",
        // minSdk 26: NDK r26 has no libnativewindow.so for API 21 (cargo-ndk default).
        "-P",
        "26",
        "-o",
        jniLibsDir.absolutePath,
        "build",
        "--release",
        // Only the cdylib: jniLibs needs libfluid_gui.so, not the desktop bin.
        "--lib",
        "-p",
        "fluid-gui",
    )

    doLast {
        val library = File(jniLibsDir, "arm64-v8a/libfluid_gui.so")
        if (!library.isFile) {
            throw GradleException(
                "cargo-ndk finished but " + library.absolutePath + " is missing. " +
                    "fluid-gui must expose a cdylib library target for the APK to load it.",
            )
        }
    }
}

if (skipCargo) {
    logger.lifecycle("fluid.skipCargo=true: packaging the Kotlin shell only, no native library.")
} else {
    tasks.matching { it.name.startsWith("merge") && it.name.endsWith("JniLibFolders") }
        .configureEach { dependsOn(cargoNdkBuild) }
}
