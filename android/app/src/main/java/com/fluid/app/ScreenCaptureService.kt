package com.fluid.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log

/**
 * Foreground service that owns the MediaProjection session.
 *
 * Why a service: Android allows a capture token to be used only while a
 * foreground service with the mediaProjection type is running (Android 10+ and
 * strictly on Android 14). Keeping the projection here also means rotating the
 * Activity does not necessarily tear the capture down.
 *
 * Flow:
 *
 *  MainActivity --consent--> resultCode + Intent --> this service
 *        |                                              |
 *        |                                   startForeground (mediaProjection)
 *        |                                   getMediaProjection
 *        |                                   ImageReader.newInstance(RGBA_8888)
 *        |                                   createVirtualDisplay(AUTO_MIRROR)
 *        v                                              |
 *  FluidNative.nativeInstallSurface              FluidNative.nativeInstallReader
 *        |                                              |
 *  ANativeWindow presenter  <-- nativeStartEnhance ->  MediaProjection source
 *
 * The service publishes two callbacks to the Activity through [Listener] so the
 * UI can chain "permission granted -> capture ready -> start enhancing" without
 * polling or blocking the main thread.
 */
class ScreenCaptureService : Service() {

    private var projection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null
    private var running = false
    private val mainHandler = Handler(Looper.getMainLooper())

    private val projectionCallback = object : MediaProjection.Callback() {
        override fun onStop() {
            // The user revoked the capture from the system status bar / dialog.
            Log.i(TAG, "MediaProjection stopped by the system or the user")
            releaseSession()
            stopSelf()
        }
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent != null && ACTION_STOP == intent.action) {
            releaseSession()
            stopSelf()
            return START_NOT_STICKY
        }
        if (running) {
            Log.i(TAG, "capture already running, ignoring duplicate start")
            return START_NOT_STICKY
        }

        // Must happen within a few seconds of startForegroundService().
        startForegroundCompat(getString(R.string.capture_text))

        val started = startCapture(intent)
        if (!started) {
            releaseSession()
            stopSelf()
        }
        // A MediaProjection token cannot be reused after the process dies, so
        // restarting with START_STICKY would only produce a broken session.
        return START_NOT_STICKY
    }

    private fun startCapture(intent: Intent?): Boolean {
        val code = intent?.getIntExtra(EXTRA_RESULT_CODE, 0) ?: 0
        val data = intent?.let { readProjectionData(it) }
        if (code == 0 || data == null) {
            Log.e(TAG, "missing MediaProjection result; was startIntent() used?")
            notifyReaderInstalled(FluidNative.ERROR_BAD_ARGUMENT)
            return false
        }

        val manager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        val newProjection = try {
            manager.getMediaProjection(code, data)
        } catch (error: SecurityException) {
            Log.e(TAG, "getMediaProjection rejected", error)
            notifyReaderInstalled(FluidNative.ERROR_INSTALL)
            return false
        } ?: run {
            Log.e(TAG, "getMediaProjection returned null")
            notifyReaderInstalled(FluidNative.ERROR_INSTALL)
            return false
        }

        val metrics = resources.displayMetrics
        val width = metrics.widthPixels
        val height = metrics.heightPixels
        if (width <= 0 || height <= 0) {
            Log.e(TAG, "invalid display size " + width + "x" + height)
            newProjection.stop()
            notifyReaderInstalled(FluidNative.ERROR_INSTALL)
            return false
        }

        val reader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, MAX_IMAGES)
        val display = try {
            newProjection.createVirtualDisplay(
                VIRTUAL_DISPLAY_NAME,
                width,
                height,
                metrics.densityDpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                reader.surface,
                null,
                null,
            )
        } catch (error: RuntimeException) {
            Log.e(TAG, "createVirtualDisplay failed", error)
            reader.close()
            newProjection.stop()
            notifyReaderInstalled(FluidNative.ERROR_INSTALL)
            return false
        }

        newProjection.registerCallback(projectionCallback, mainHandler)

        // Hand the reader to Rust. The native side takes its own global reference,
        // so Kotlin keeps owning (and closing) the ImageReader here.
        val installCode = try {
            FluidNative.nativeInstallReader(reader)
        } catch (error: LinkageError) {
            // Covers UnsatisfiedLinkError and the sticky initialisation failures after it.
            Log.e(TAG, "native library not loaded", error)
            ERROR_NATIVE_LIBRARY
        }
        if (installCode != FluidNative.OK) {
            Log.e(TAG, "nativeInstallReader failed: " + FluidNative.describe(installCode))
            display.release()
            reader.close()
            newProjection.unregisterCallback(projectionCallback)
            newProjection.stop()
            notifyReaderInstalled(installCode)
            return false
        }

        projection = newProjection
        virtualDisplay = display
        imageReader = reader
        running = true
        active = true
        Log.i(TAG, "MediaProjection capture started: " + width + "x" + height + " @" + metrics.densityDpi + "dpi")
        updateNotification("正在录制屏幕 " + width + "x" + height)
        notifyReaderInstalled(FluidNative.OK)
        return true
    }

    private fun releaseSession() {
        val wasRunning = running
        running = false
        active = false
        if (wasRunning) {
            // Stop the pipeline before the source disappears: it checks the cancel
            // flag once per frame and then reports normally instead of hammering a
            // closed ImageReader.
            try {
                FluidNative.nativeStopEnhance()
            } catch (error: LinkageError) {
                Log.w(TAG, "nativeStopEnhance failed", error)
            }
        }
        virtualDisplay?.release()
        virtualDisplay = null
        imageReader?.close()
        imageReader = null
        projection?.unregisterCallback(projectionCallback)
        projection?.stop()
        projection = null
        if (wasRunning) {
            // Drop Rust's global reference a moment later so the pipeline thread has
            // time to notice the cancel flag before the ImageReader goes away.
            mainHandler.postDelayed({
                try {
                    FluidNative.nativeUninstallReader()
                } catch (error: LinkageError) {
                    Log.w(TAG, "nativeUninstallReader failed", error)
                }
            }, RELEASE_DELAY_MS)
            notifyCaptureStopped()
        }
    }

    override fun onDestroy() {
        releaseSession()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    // ------------------------------------------------------------------
    // Listener plumbing (same process, so a plain static is enough)
    // ------------------------------------------------------------------

    private fun notifyReaderInstalled(code: Int) {
        mainHandler.post { listener?.onReaderInstalled(code) }
    }

    private fun notifyCaptureStopped() {
        mainHandler.post { listener?.onCaptureStopped() }
    }

    // ------------------------------------------------------------------
    // Notification
    // ------------------------------------------------------------------

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.capture_title),
            NotificationManager.IMPORTANCE_LOW,
        )
        channel.description = getString(R.string.capture_text)
        channel.setShowBadge(false)
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.createNotificationChannel(channel)
    }

    private fun startForegroundCompat(text: String) {
        val notification = buildNotification(text)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION,
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun updateNotification(text: String) {
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.notify(NOTIFICATION_ID, buildNotification(text))
    }

    @Suppress("DEPRECATION") // Notification.Builder.addAction(int, ...) works on all supported API levels.
    private fun buildNotification(text: String): Notification {
        val openApp = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val stop = PendingIntent.getService(
            this,
            1,
            Intent(this, ScreenCaptureService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.capture_title))
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setOngoing(true)
            .setContentIntent(openApp)
            .addAction(android.R.drawable.ic_menu_close_clear_cancel, "停止", stop)
            .build()
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private fun readProjectionData(intent: Intent): Intent? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            intent.getParcelableExtra(EXTRA_RESULT_DATA, Intent::class.java)
        } else {
            @Suppress("DEPRECATION")
            intent.getParcelableExtra(EXTRA_RESULT_DATA)
        }
    }

    /**
     * Callbacks consumed by MainActivity. Delivered on the main thread.
     */
    interface Listener {
        /** Result of FluidNative.nativeInstallReader; [FluidNative.OK] means capture is ready. */
        fun onReaderInstalled(code: Int)

        /** The projection ended (notification stop, permission revoked, service stop). */
        fun onCaptureStopped()
    }

    companion object {
        private const val TAG = "FluidCapture"
        private const val CHANNEL_ID = "fluid-screen-capture"
        private const val NOTIFICATION_ID = 41
        private const val VIRTUAL_DISPLAY_NAME = "fluid-screen"
        private const val MAX_IMAGES = 3
        private const val RELEASE_DELAY_MS = 300L
        private const val EXTRA_RESULT_CODE = "com.fluid.app.EXTRA_RESULT_CODE"
        private const val EXTRA_RESULT_DATA = "com.fluid.app.EXTRA_RESULT_DATA"
        private const val ACTION_STOP = "com.fluid.app.action.STOP_CAPTURE"
        /** Pseudo return code when the native library could not be loaded. */
        private const val ERROR_NATIVE_LIBRARY = -100

        /** Current Activity listener; the Activity clears it in onDestroy. */
        @Volatile
        private var listener: Listener? = null

        /**
         * True while a projection session is live. Used by MainActivity after an
         * Activity recreation to learn that the ImageReader is already installed
         * into Rust (the one-shot listener callback has been lost by then).
         */
        @Volatile
        private var active = false

        /** Whether a capture session with an installed ImageReader is live. */
        fun isCapturing(): Boolean = active

        /** Register the Activity that wants capture lifecycle callbacks. */
        fun setListener(next: Listener?) {
            listener = next
        }

        /** Intent that starts the capture service with a granted projection token. */
        fun startIntent(context: Context, resultCode: Int, data: Intent): Intent {
            return Intent(context, ScreenCaptureService::class.java)
                .putExtra(EXTRA_RESULT_CODE, resultCode)
                .putExtra(EXTRA_RESULT_DATA, data)
        }

        /** Intent that stops a running capture session. */
        fun stopIntent(context: Context): Intent {
            return Intent(context, ScreenCaptureService::class.java).setAction(ACTION_STOP)
        }
    }
}
