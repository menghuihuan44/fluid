package com.fluid.app

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.WindowInsets
import android.widget.Button
import android.widget.RadioGroup
import android.widget.TextView
import org.json.JSONObject

/**
 * Fluid Android shell.
 *
 * The Activity owns the UI and the MediaProjection consent dialog; all heavy
 * work happens in Rust. Pressing 开始增强:
 *
 * 1. If no capture session exists yet, ask for MediaProjection consent.
 * 2. Start [ScreenCaptureService] with the granted token; the foreground service
 *    creates the virtual display + ImageReader and calls
 *    [FluidNative.nativeInstallReader]. It reports the result through
 *    [ScreenCaptureService.Listener.onReaderInstalled].
 * 3. surfaceCreated installs the preview Surface as the presentation window
 *    ([FluidNative.nativeInstallSurface]).
 * 4. Once both are ready, [FluidNative.nativeStartEnhance] starts the pipeline on
 *    its own background thread. This Activity only polls
 *    [FluidNative.nativeLastReport] every 500 ms for the status line.
 *
 * 停止增强 sets the cooperative cancel flag, waits (asynchronously, never
 * blocking the main thread) until the pipeline thread ended, then stops the
 * capture service. Surface teardown stops the pipeline before uninstalling the
 * ANativeWindow, because the presenter reads the window pointer on every frame.
 */
@Suppress("DEPRECATION") // startActivityForResult keeps the shell free of AndroidX.
class MainActivity : Activity(), SurfaceHolder.Callback, ScreenCaptureService.Listener {

    private lateinit var surfaceView: SurfaceView
    private lateinit var statusView: TextView
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var interpGroup: RadioGroup
    private lateinit var scaleGroup: RadioGroup
    private lateinit var qualityGroup: RadioGroup

    private val mainHandler = Handler(Looper.getMainLooper())

    /** The SurfaceView currently has a valid Surface installed into Rust. */
    private var surfaceReady = false

    /** The MediaProjection ImageReader is installed into Rust. */
    private var readerInstalled = false

    /** A system consent dialog is on screen, so onStop must not tear anything down. */
    private var awaitingPermission = false

    /** The user pressed 开始增强 and we are waiting for surface/reader readiness. */
    private var pendingStart = false

    private var lastStatus = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        surfaceView = findViewById(R.id.surface_view)
        statusView = findViewById(R.id.status_text)
        startButton = findViewById(R.id.start_button)
        stopButton = findViewById(R.id.stop_button)
        interpGroup = findViewById(R.id.interp_group)
        scaleGroup = findViewById(R.id.scale_group)
        qualityGroup = findViewById(R.id.quality_group)

        surfaceView.holder.addCallback(this)
        startButton.setOnClickListener { onStartClicked() }
        stopButton.setOnClickListener { onStopClicked() }

        ScreenCaptureService.setListener(this)
        applySafeArea()
        requestNotificationPermissionIfNeeded()
        setStatus(getString(R.string.status_idle))
        mainHandler.post(statusPoll)
    }

    // ------------------------------------------------------------------
    // Start / stop
    // ------------------------------------------------------------------

    private fun onStartClicked() {
        if (nativeRunning()) {
            setStatus("已经在运行，请先点“停止增强”。")
            return
        }
        if (awaitingPermission) {
            setStatus("正在等待录屏权限…")
            return
        }
        pendingStart = true
        if (!surfaceReady) {
            setStatus("正在等待预览画面就绪…")
            return
        }
        if (!readerInstalled) {
            setStatus("正在请求录屏权限…")
            requestScreenCapture()
            return
        }
        beginEnhance()
    }

    private fun onStopClicked() {
        setStatus("正在停止…")
        stopPipelineThen { finishStop() }
    }

    /** Ask the pipeline thread to stop, then run [action] once it exited. */
    private fun stopPipelineThen(action: () -> Unit) {
        if (!nativeRunning()) {
            action()
            return
        }
        try {
            FluidNative.nativeStopEnhance()
        } catch (error: LinkageError) {
            Log.w(TAG, "nativeStopEnhance failed", error)
        }
        val deadline = System.currentTimeMillis() + STOP_TIMEOUT_MS
        val poll = object : Runnable {
            override fun run() {
                if (!nativeRunning() || System.currentTimeMillis() > deadline) {
                    action()
                    return
                }
                mainHandler.postDelayed(this, STOP_POLL_MS)
            }
        }
        mainHandler.postDelayed(poll, STOP_POLL_MS)
    }

    private fun finishStop() {
        stopService(Intent(this, ScreenCaptureService::class.java))
        readerInstalled = false
        pendingStart = false
        startButton.isEnabled = true
        stopButton.isEnabled = false
        setStatus("已停止。")
    }

    private fun beginEnhance() {
        val multiplier = when (interpGroup.checkedRadioButtonId) {
            R.id.interp_2x -> 2
            R.id.interp_3x -> 3
            else -> 1
        }
        val scale = when (scaleGroup.checkedRadioButtonId) {
            R.id.scale_15 -> 1.5f
            R.id.scale_2 -> 2.0f
            else -> 1.0f
        }
        val quality = when (qualityGroup.checkedRadioButtonId) {
            R.id.quality_performance -> "ultra-performance"
            R.id.quality_high -> "quality"
            else -> "balanced"
        }
        val filter = if (scale <= 1.0f) {
            "off"
        } else {
            when (qualityGroup.checkedRadioButtonId) {
                R.id.quality_performance -> "bilinear"
                R.id.quality_high -> "bicubic"
                else -> "adaptive"
            }
        }

        val code = try {
            FluidNative.nativeStartEnhance(multiplier, scale, filter, quality)
        } catch (error: LinkageError) {
            Log.e(TAG, "nativeStartEnhance failed", error)
            pendingStart = false
            setStatus("原生库未加载，无法开始增强。")
            return
        }
        if (code == FluidNative.OK) {
            startButton.isEnabled = false
            stopButton.isEnabled = true
            setStatus("正在启动增强…")
        } else {
            pendingStart = false
            setStatus("无法开始增强：" + FluidNative.describe(code))
        }
    }

    private fun maybeBeginEnhance() {
        if (!pendingStart) {
            return
        }
        if (!surfaceReady || !readerInstalled) {
            return
        }
        pendingStart = false
        beginEnhance()
    }

    // ------------------------------------------------------------------
    // MediaProjection permission
    // ------------------------------------------------------------------

    private fun requestScreenCapture() {
        val manager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        awaitingPermission = true
        try {
            startActivityForResult(manager.createScreenCaptureIntent(), REQUEST_CAPTURE)
        } catch (error: RuntimeException) {
            // Some OEM builds throw when the projection service is unavailable.
            awaitingPermission = false
            pendingStart = false
            Log.e(TAG, "cannot create screen capture intent", error)
            setStatus("无法请求录屏权限：" + error.message)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_CAPTURE) {
            return
        }
        awaitingPermission = false
        if (resultCode != RESULT_OK || data == null) {
            pendingStart = false
            setStatus("录屏权限被拒绝，无法捕获屏幕。")
            return
        }
        // The foreground service must exist before the token is handed to
        // getMediaProjection() on Android 14.
        startForegroundService(ScreenCaptureService.startIntent(this, resultCode, data))
        setStatus("权限已授予，正在准备屏幕捕获…")
    }

    // ------------------------------------------------------------------
    // ScreenCaptureService.Listener
    // ------------------------------------------------------------------

    override fun onReaderInstalled(code: Int) {
        readerInstalled = code == FluidNative.OK
        if (!readerInstalled) {
            pendingStart = false
            setStatus("屏幕捕获初始化失败：" + FluidNative.describe(code))
            return
        }
        maybeBeginEnhance()
    }

    override fun onCaptureStopped() {
        readerInstalled = false
        pendingStart = false
        if (nativeRunning()) {
            try {
                FluidNative.nativeStopEnhance()
            } catch (error: LinkageError) {
                Log.w(TAG, "nativeStopEnhance failed", error)
            }
        }
        setStatus("屏幕捕获已停止。")
    }

    // ------------------------------------------------------------------
    // SurfaceView -> Rust presenter
    // ------------------------------------------------------------------

    override fun surfaceCreated(holder: SurfaceHolder) {
        val code = try {
            FluidNative.nativeInstallSurface(holder.surface)
        } catch (error: LinkageError) {
            // Kotlin-only debug APK: make the failure visible instead of crashing.
            // LinkageError covers UnsatisfiedLinkError and the sticky
            // ExceptionInInitializerError/NoClassDefFoundError raised on later calls
            // after System.loadLibrary() failed once.
            Log.e(TAG, "native library not loaded", error)
            setStatus("原生库未加载（请先用 cargo-ndk 构建 libfluid_gui.so）。")
            return
        }
        surfaceReady = code == FluidNative.OK
        if (!surfaceReady) {
            setStatus("预览窗口安装失败：" + FluidNative.describe(code))
            return
        }
        maybeBeginEnhance()
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        // No native call: the presenter keeps its own dimensions and rescales the
        // frame to the window buffer geometry when it presents.
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        // The presenter reads the installed ANativeWindow on every frame, so the
        // pipeline has to end before the window is released.
        surfaceReady = false
        pendingStart = false
        stopPipelineThen {
            try {
                FluidNative.nativeUninstallSurface()
            } catch (error: LinkageError) {
                Log.w(TAG, "nativeUninstallSurface failed", error)
            }
        }
    }

    // ------------------------------------------------------------------
    // Status line
    // ------------------------------------------------------------------

    /** Polls the native JSON status every 500 ms; never blocks the UI thread. */
    private val statusPoll = object : Runnable {
        override fun run() {
            refreshFromNative()
            mainHandler.postDelayed(this, POLL_INTERVAL_MS)
        }
    }

    private fun refreshFromNative() {
        val document = try {
            FluidNative.nativeLastReport()
        } catch (error: LinkageError) {
            return
        } ?: return
        val json = try {
            JSONObject(document)
        } catch (error: Exception) {
            return
        }
        val running = json.optBoolean("running", false)
        startButton.isEnabled = !running
        stopButton.isEnabled = running
        when (json.optString("state", "idle")) {
            "running" -> {
                val seconds = json.optDouble("elapsed_s", 0.0).toInt()
                setStatus("运行中 · 已运行 " + seconds + " 秒")
            }
            "finished" -> setStatus(describeReport(json.optJSONObject("report")))
            "error" -> setStatus("增强出错：" + json.optString("error", "未知错误"))
            else -> Unit
        }
    }

    private fun describeReport(report: JSONObject?): String {
        if (report == null) {
            return "已停止"
        }
        val fps = report.optDouble("output_fps", 0.0)
        val out = report.optJSONArray("out")
        val size = if (out != null && out.length() >= 2) {
            out.optInt(0).toString() + "x" + out.optInt(1)
        } else {
            "?"
        }
        val interp = report.optString("interp_backend", "none")
        val backend = report.optString("backend", "none")
        return "已停止 · 输出 " + String.format(java.util.Locale.US, "%.1f", fps) + " fps · " + size +
            " · 插帧 " + interp + " · 放大 " + backend
    }

    private fun nativeRunning(): Boolean {
        return try {
            FluidNative.nativeIsRunning() == 1
        } catch (error: LinkageError) {
            false
        }
    }

    private fun setStatus(text: String) {
        if (text == lastStatus) {
            return
        }
        lastStatus = text
        statusView.text = text
        Log.i(TAG, text)
    }

    // ------------------------------------------------------------------
    // Window insets / permissions / lifecycle
    // ------------------------------------------------------------------

    /** Keep the buttons above the gesture bar and the status text below the notch. */
    private fun applySafeArea() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
            return
        }
        window.setDecorFitsSystemWindows(false)
        val root = findViewById<View>(R.id.root)
        val basePadding = (12 * resources.displayMetrics.density).toInt()
        root.setOnApplyWindowInsetsListener { view, insets ->
            val bars = insets.getInsets(
                WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout(),
            )
            view.setPadding(
                bars.left + basePadding,
                bars.top + basePadding,
                bars.right + basePadding,
                bars.bottom + basePadding,
            )
            insets
        }
    }

    private fun requestNotificationPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
            return
        }
        if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            return
        }
        requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), REQUEST_NOTIFICATIONS)
    }

    override fun onResume() {
        super.onResume()
        // The service may already be capturing (for example after an Activity
        // recreation): the ImageReader is installed into Rust, so only the surface
        // is missing and surfaceCreated will continue the start flow.
        if (!readerInstalled && ScreenCaptureService.isCapturing()) {
            readerInstalled = true
            maybeBeginEnhance()
        }
    }

    override fun onStop() {
        super.onStop()
        // Leaving the app stops the enhancement (the foreground service keeps the
        // projection only while the user is actually watching). The consent dialog
        // also stops the Activity, so that case is skipped.
        if (!isChangingConfigurations && !awaitingPermission && nativeRunning()) {
            stopPipelineThen { finishStop() }
        }
    }

    override fun onDestroy() {
        ScreenCaptureService.setListener(null)
        mainHandler.removeCallbacksAndMessages(null)
        if (nativeRunning()) {
            try {
                FluidNative.nativeStopEnhance()
            } catch (error: LinkageError) {
                Log.w(TAG, "nativeStopEnhance failed", error)
            }
        }
        if (!isChangingConfigurations) {
            stopService(Intent(this, ScreenCaptureService::class.java))
        }
        super.onDestroy()
    }

    companion object {
        private const val TAG = "FluidMain"
        private const val REQUEST_CAPTURE = 1001
        private const val REQUEST_NOTIFICATIONS = 1002
        private const val POLL_INTERVAL_MS = 500L
        private const val STOP_POLL_MS = 100L
        private const val STOP_TIMEOUT_MS = 2000L
    }
}
