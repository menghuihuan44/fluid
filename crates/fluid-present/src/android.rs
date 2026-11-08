//! Android presentation through ANativeWindow (SurfaceView / SurfaceTexture).
//!
//! Android has no free-standing window that Rust can create: the app owns the
//! Surface (usually a SurfaceView or a TextureView) and the native side is
//! handed an ANativeWindow pointer once per surface lifetime. The window is not
//! owned by this crate; it is registered here, used for every present, and its
//! reference is released by uninstall().
//!
//! # Kotlin glue
//!
//! The SurfaceView must notify Rust when its surface appears and disappears;
//! the native side converts the Surface into an ANativeWindow and registers it.
//!
//! Kotlin side:
//!
//! - class FluidView(context: Context) : SurfaceView(context), SurfaceHolder.Callback
//! - override fun surfaceCreated(holder: SurfaceHolder) { nativeInstall(holder.surface) }
//! - override fun surfaceDestroyed(holder: SurfaceHolder) { nativeUninstall() }
//! - external fun nativeInstall(surface: Surface)
//! - external fun nativeUninstall()
//! - external fun nativeRequestClose()
//! - main thread / UI callback: nativeRequestClose() so the render loop stops
//!   (FluidSink::should_close() turns true and the app can tear down the engine).
//! - keep the display awake yourself; this backend never changes window flags.
//!
//! Rust JNI entry point (in the app crate, not in this library):
//!
//! - #[unsafe(no_mangle)] pub extern "system" fn Java_com_example_app_FluidView_nativeInstall(
//!       env: jni::sys::JNIEnv, _class: jni::sys::jclass, surface: jni::sys::jobject) {
//!       let mut env = unsafe { jni::Env::from_raw(env) }.expect("valid JNIEnv");
//!       fluid_present::android::install_from_surface(env.get_raw(), surface).expect("window");
//!   }
//!
//! install_from_surface() wraps ANativeWindow_fromSurface and install().
//!
//! # Present path
//!
//! Every present:
//!
//! 1. checks the close flag (FluidSink::should_close / request_close),
//! 2. configures the buffer geometry to the frame size once per size change,
//! 3. ANativeWindow_lock() and converts the frame row by row into the locked
//!    buffer, honouring buffer.stride (the surface row pitch is often larger
//!    than the frame width) with blit::write_frame_rows,
//! 4. ANativeWindow_unlockAndPost() hands the buffer to SurfaceFlinger.
//!
//! The surface buffer format is RGBA_8888; the alpha byte is always written as
//! 255 (opaque). Source frames may be Rgba8, Bgra8, Rgb8 or Luma8 and are
//! converted by blit::write_frame_rows.
//!
//! # Threading and lifecycle
//!
//! install(), present() and uninstall() must all happen on the same thread that
//! receives the JNI/UI callbacks (normally the Android main thread or the
//! dedicated render thread that owns the Surface). ANativeWindow is not
//! thread-safe. The sink keeps no pointer: it looks the window up in the
//! registry on every present, so a surface destroyed between frames returns a
//! cancelled/unsupported error instead of touching freed memory.
//!
//! PresentConfig::vsync, fullscreen, borderless, title and monitor are ignored
//! on Android (SurfaceFlinger owns those decisions); low_latency is accepted as
//! a hint and has no effect on the ANativeWindow path.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use fluid_core::{Dims, FluidError, Frame, FrameSink, PresentConfig, Result};

use crate::blit;

/// Set when the app wants the render loop to stop (for example when the user
/// presses back or the activity goes away).
static CLOSE_REQUESTED: AtomicBool = AtomicBool::new(false);

/// The registered window, stored as an integer because raw pointers are not
/// Send and the registry is a global.
static WINDOW: OnceLock<Mutex<Option<usize>>> = OnceLock::new();

fn window_slot() -> &'static Mutex<Option<usize>> {
    WINDOW.get_or_init(|| Mutex::new(None))
}

fn registry_error() -> FluidError {
    FluidError::internal("android present: ANativeWindow registry is poisoned")
}

/// Register the window that frames are presented into.
///
/// Takes ownership of one reference to the window: pass a pointer returned by
/// ANativeWindow_fromSurface() (as install_from_surface() does) and call
/// uninstall() when the surface goes away. Safe to call again with a new
/// Surface; the previous reference is released.
// The pointer is only handed to the NDK (acquire/release), never dereferenced by
// Rust code. The caller still owns a valid window reference, as documented.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn install(window: *mut ndk_sys::ANativeWindow) -> Result<()> {
    if window.is_null() {
        return Err(FluidError::invalid(
            "android present: ANativeWindow pointer must not be null",
        ));
    }
    // SAFETY: the caller transfers a window reference it owns; acquire keeps it
    // alive while it is registered, release happens in uninstall()/replace.
    unsafe { ndk_sys::ANativeWindow_acquire(window) };
    let mut slot = window_slot().lock().map_err(|_| registry_error())?;
    if let Some(previous) = slot.replace(window as usize) {
        // SAFETY: previous was acquired by install() and is owned by this crate.
        unsafe { ndk_sys::ANativeWindow_release(previous as *mut ndk_sys::ANativeWindow) };
    }
    Ok(())
}

/// Convert a Java Surface into an ANativeWindow and install it.
///
/// This is the convenience entry point for a JNI callback such as
/// SurfaceHolder.Callback.surfaceCreated(holder) -> nativeInstall(holder.surface):
/// pass the JNIEnv and the java.lang.Object surface jobject straight through.
///
/// # Safety
///
/// env must be the valid JNIEnv of the current thread, and surface a valid
/// java Surface jobject local reference (or null).
pub unsafe fn install_from_surface(
    env: *mut jni::sys::JNIEnv,
    surface: jni::sys::jobject,
) -> Result<()> {
    if env.is_null() {
        return Err(FluidError::invalid(
            "android present: JNIEnv pointer must not be null",
        ));
    }
    if surface.is_null() {
        return Err(FluidError::invalid(
            "android present: Surface jobject must not be null",
        ));
    }
    // SAFETY: the caller guarantees a valid JNIEnv and Surface; the returned
    // window reference is owned by this call and handed to install().
    let window = unsafe { ndk_sys::ANativeWindow_fromSurface(env as *mut _, surface as *mut _) };
    if window.is_null() {
        return Err(FluidError::platform(
            "android present: ANativeWindow_fromSurface returned null (is the Surface alive?)",
        ));
    }
    install(window)
}

/// Release the registered window (call from surfaceDestroyed).
pub fn uninstall() -> Result<()> {
    let mut slot = window_slot().lock().map_err(|_| registry_error())?;
    if let Some(previous) = slot.take() {
        // SAFETY: acquired by install(), owned by this crate.
        unsafe { ndk_sys::ANativeWindow_release(previous as *mut ndk_sys::ANativeWindow) };
    }
    CLOSE_REQUESTED.store(false, Ordering::Relaxed);
    Ok(())
}

/// Whether a window is currently registered.
pub fn is_installed() -> bool {
    window_slot()
        .lock()
        .map(|slot| slot.is_some())
        .unwrap_or(false)
}

/// Ask the render loop to stop: FluidSink::should_close() returns true and
/// further presents fail with FluidError::cancelled. Call from the UI thread.
pub fn request_close() {
    CLOSE_REQUESTED.store(true, Ordering::Relaxed);
}

/// Clear a close request, e.g. when the activity resumes and a new surface is
/// installed.
pub fn clear_close_request() {
    CLOSE_REQUESTED.store(false, Ordering::Relaxed);
}

fn close_requested() -> bool {
    CLOSE_REQUESTED.load(Ordering::Relaxed)
}

/// Current window pointer, or an error explaining how to install one.
fn current_window() -> Result<*mut ndk_sys::ANativeWindow> {
    let slot = window_slot().lock().map_err(|_| registry_error())?;
    match *slot {
        Some(pointer) => Ok(pointer as *mut ndk_sys::ANativeWindow),
        None => Err(FluidError::unsupported(
            "android present: no ANativeWindow installed; call \
             fluid_present::android::install(ANativeWindow) (or install_from_surface) \
             from the SurfaceHolder callback first",
        )),
    }
}

/// Open a presenter that blits frames into the registered ANativeWindow.
pub(crate) fn open(_config: &PresentConfig, dims: Dims) -> Result<Box<dyn FrameSink>> {
    if dims.is_empty() {
        return Err(FluidError::invalid(
            "present: frame dimensions must be non-zero",
        ));
    }
    // Fail early with the actionable "no window installed" message rather than
    // at the first present.
    let _ = current_window()?;
    Ok(Box::new(AndroidSink {
        dims,
        geometry: None,
        closed: false,
    }))
}

/// FrameSink that copies frames into the registered ANativeWindow buffer.
struct AndroidSink {
    dims: Dims,
    /// Last geometry configured through ANativeWindow_setBuffersGeometry.
    geometry: Option<Dims>,
    closed: bool,
}

impl AndroidSink {
    fn copy_geometry(&self, window: *mut ndk_sys::ANativeWindow) -> Result<()> {
        let format = ndk_sys::ANativeWindow_LegacyFormat::WINDOW_FORMAT_RGBA_8888.0 as i32;
        // SAFETY: window was validated by current_window() and is registered for
        // the lifetime of this call; the app only calls this between
        // surfaceCreated and surfaceDestroyed.
        let status = unsafe {
            ndk_sys::ANativeWindow_setBuffersGeometry(
                window,
                self.dims.width as i32,
                self.dims.height as i32,
                format,
            )
        };
        if status != 0 {
            return Err(FluidError::platform(format!(
                "android present: ANativeWindow_setBuffersGeometry({}x{}) failed with {status}",
                self.dims.width, self.dims.height
            )));
        }
        Ok(())
    }

    /// Lock, blit and post one frame. The lock is always released.
    fn blit_and_post(&mut self, window: *mut ndk_sys::ANativeWindow, frame: &Frame) -> Result<()> {
        // SAFETY: ANativeWindow_Buffer is plain old data; zeroed is a valid
        // initial value that ANativeWindow_lock overwrites.
        let mut buffer: ndk_sys::ANativeWindow_Buffer = unsafe { std::mem::zeroed() };
        // SAFETY: window is valid and unlocked; null dirty bounds means "whole
        // buffer" and is explicitly allowed by the NDK.
        let status =
            unsafe { ndk_sys::ANativeWindow_lock(window, &mut buffer, std::ptr::null_mut()) };
        if status != 0 {
            return Err(FluidError::platform(format!(
                "android present: ANativeWindow_lock failed with {status}"
            )));
        }
        let copied = copy_locked(frame, &buffer);
        // SAFETY: balances the successful ANativeWindow_lock above, even when
        // the copy failed; the buffer must be posted to become visible.
        let posted = unsafe { ndk_sys::ANativeWindow_unlockAndPost(window) };
        copied?;
        if posted != 0 {
            return Err(FluidError::platform(format!(
                "android present: ANativeWindow_unlockAndPost failed with {posted}"
            )));
        }
        Ok(())
    }
}

/// Convert the frame into a locked ANativeWindow buffer.
fn copy_locked(frame: &Frame, buffer: &ndk_sys::ANativeWindow_Buffer) -> Result<()> {
    if buffer.bits.is_null() {
        return Err(FluidError::platform(
            "android present: ANativeWindow buffer has no pixels",
        ));
    }
    if buffer.stride <= 0 || buffer.width <= 0 || buffer.height <= 0 {
        return Err(FluidError::platform(format!(
            "android present: ANativeWindow buffer has invalid geometry {}x{} stride {}",
            buffer.width, buffer.height, buffer.stride
        )));
    }
    let dims = frame.dims;
    if buffer.width < dims.width as i32 || buffer.height < dims.height as i32 {
        return Err(FluidError::platform(format!(
            "android present: ANativeWindow buffer is {}x{} but the frame is {}x{}",
            buffer.width, buffer.height, dims.width, dims.height
        )));
    }
    let stride = buffer.stride as usize;
    let len = stride
        .checked_mul(buffer.height as usize)
        .and_then(|rows| rows.checked_mul(blit::WINDOW_PIXEL_BYTES))
        .ok_or_else(|| {
            FluidError::platform("android present: ANativeWindow buffer is too large")
        })?;
    // SAFETY: a successfully locked ANativeWindow_Buffer exposes writable pixel
    // memory for stride * height pixels of 4 bytes until unlockAndPost; the
    // slice never outlives that lock.
    let dst = unsafe { std::slice::from_raw_parts_mut(buffer.bits as *mut u8, len) };
    blit::write_frame_rows(frame, dst, stride)
}

impl FrameSink for AndroidSink {
    fn dims(&self) -> Dims {
        self.dims
    }

    fn backend_name(&self) -> &'static str {
        "android-anativewindow"
    }

    fn present(&mut self, frame: &Frame) -> Result<()> {
        if self.closed || close_requested() {
            return Err(FluidError::cancelled());
        }
        let window = current_window()?;
        if frame.dims != self.dims {
            self.dims = frame.dims;
            self.geometry = None;
        }
        if self.geometry != Some(self.dims) {
            self.copy_geometry(window)?;
            self.geometry = Some(self.dims);
        }
        self.blit_and_post(window, frame)
    }

    fn should_close(&self) -> bool {
        self.closed || close_requested()
    }

    fn close(&mut self) {
        self.closed = true;
    }
}
