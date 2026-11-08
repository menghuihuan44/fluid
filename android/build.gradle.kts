// Root Gradle build for the Fluid Android shell.
//
// Versions are pinned deliberately: AGP 8.5.x requires JDK 17, Kotlin 1.9.24 is
// the last release line that works with AGP 8.5 without extra configuration.
plugins {
    id("com.android.application") version "8.5.2" apply false
    id("org.jetbrains.kotlin.android") version "1.9.24" apply false
}
