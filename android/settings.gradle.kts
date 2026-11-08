// Fluid Android app shell (Gradle Kotlin DSL).
//
// Open this directory in Android Studio, or build from the command line with
// scripts/build-android.ps1 / scripts/build-android.sh from the repository root.
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "fluid-android"
include(":app")
