package com.fluid.app

import android.media.ImageReader
import android.view.Surface

/**
 * JNI bridge to the Rust platform layer.
 *
 * The native functions are implemented in crates/fluid-gui/src/android_jni.rs
 * (compiled only for target_os = "android"). The install calls hand the Android
 * platform objects to Rust; the enhance calls then run the real Fluid pipeline
 * on the phone:
 *
 *  * [nativeInstallReader] stores a global reference to the [ImageReader] created
 *    by [ScreenCaptureService] and makes it the MediaProjection capture source
 *    (fluid_capture::android::install).
 *  * [nativeInstallSurface] converts the [Surface] of the preview SurfaceView into
 *    an ANativeWindow and makes it the presentation target
 *    (fluid_present::android::install_from_surface).
 *  * [nativeStartEnhance] spawns ONE background thread that opens the capture
 *    source and the present sink, runs fluid-engine and closes both again. The
 *    call returns immediately; [nativeLastReport] shows progress and the result.
 *  * [nativeStopEnhance] sets the cooperative cancel flag; the thread finishes the
 *    frame in flight, writes its report and exits.
 *  * [nativeUninstallSurface] / [nativeUninstallReader] release the platform
 *    handles after the pipeline stopped.
 *
 * Return values are small negative error codes (see the constants) instead of
 * exceptions: calling these from a SurfaceHolder callback must never crash the
 * app, and the JavaVM / JNI environment can legitimately be unavailable when the
 * native library was not loaded (for example a Kotlin-only debug APK built with
 * -Pfluid.skipCargo=true).
 */
object FluidNative {

    /** Loaded from jniLibs/arm64-v8a/libfluid_gui.so. */
    private const val LIBRARY = "fluid_gui"

    const val OK = 0
    const val ERROR_NO_JAVA_VM = -1
    const val ERROR_BAD_ARGUMENT = -2
    const val ERROR_JNI = -3
    const val ERROR_INSTALL = -4

    /** A pipeline thread is still alive; stop it before starting again. */
    const val ERROR_BUSY = -5

    /** The pipeline thread could not be spawned or the config was rejected. */
    const val ERROR_PIPELINE = -6

    init {
        System.loadLibrary(LIBRARY)
    }

    /**
     * Install the MediaProjection capture source.
     *
     * Must be called from a JVM-attached thread (any JNI callback or the Android
     * main thread is attached by definition).
     */
    external fun nativeInstallReader(reader: ImageReader): Int

    /**
     * Install the preview SurfaceView as the presentation window.
     *
     * Call from SurfaceHolder.Callback.surfaceCreated; the surface may be
     * recreated after rotation or resume, in which case call it again with the new
     * Surface. Rust acquires its own ANativeWindow reference.
     */
    external fun nativeInstallSurface(surface: Surface): Int

    /**
     * Start the live enhancement pipeline.
     *
     * @param multiplier total frames per source frame (1 = no interpolation).
     * @param scale output scale factor (1.0 = keep the captured resolution).
     * @param filter upscaling filter name, or "off".
     * @param quality quality preset name (ultra-performance / balanced / quality).
     * @return [OK] when the background thread was started, otherwise an ERROR_* code.
     */
    external fun nativeStartEnhance(multiplier: Int, scale: Float, filter: String, quality: String): Int

    /** Ask the running pipeline to stop. Returns immediately; poll [nativeIsRunning]. */
    external fun nativeStopEnhance(): Int

    /** 1 while the pipeline thread is alive, 0 otherwise. */
    external fun nativeIsRunning(): Int

    /**
     * Status snapshot as a JSON document:
     * {"running":bool,"state":"idle|running|finished|error","elapsed_s":double,
     *  "error":string|null,"report":{...}|null}
     *
     * The report object is the fluid-engine PipelineReport JSON once a run has
     * finished (source, sink, src, out, multiplier, filter, backend,
     * interp_backend, preset, fps, latency, ...).
     */
    external fun nativeLastReport(): String?

    /** Release the ANativeWindow installed by [nativeInstallSurface]. */
    external fun nativeUninstallSurface(): Int

    /** Release the MediaProjection ImageReader installed by [nativeInstallReader]. */
    external fun nativeUninstallReader(): Int

    /** Human readable form of a native return code, for logs and the status line. */
    fun describe(code: Int): String = when (code) {
        OK -> "ok"
        ERROR_NO_JAVA_VM -> "no JavaVM (JNI_OnLoad was not called)"
        ERROR_BAD_ARGUMENT -> "invalid argument (scale/filter/quality)"
        ERROR_JNI -> "JNI call failed"
        ERROR_INSTALL -> "native install failed (see logcat)"
        ERROR_BUSY -> "已有增强任务在运行"
        ERROR_PIPELINE -> "无法启动处理管线"
        else -> "unknown error " + code
    }
}
