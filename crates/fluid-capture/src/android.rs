//! Android capture backend: MediaProjection -> ImageReader -> tight RGBA8 frames.
//!
//! ## Why the app has to do part of the work
//!
//! Android exposes no API that lets an arbitrary native process read screen
//! pixels. The foreground app must own a MediaProjection session and mirror the
//! display into an ImageReader; this module then pulls android.media.Image
//! objects out of that reader through JNI and converts them to
//! fluid_core::Frame values. The projection token, the permission dialog and the
//! foreground service are Kotlin responsibilities; this file owns the native
//! readback, pacing, region cropping and the Android lifecycle errors.
//!
//! ## Kotlin glue (the expected sequence)
//!
//! ```text
//! // 1. Ask for consent once per app session.
//! val manager = getSystemService(MediaProjectionManager::class.java)
//! startActivityForResult(manager.createScreenCaptureIntent(), REQ_CAPTURE)
//!
//! // 2. In onActivityResult(resultCode, data) create the projection. On Android
//! //    14+ this must happen inside a foreground service of type mediaProjection.
//! val projection = manager.getMediaProjection(resultCode, data)
//!
//! // 3. Create the ImageReader this backend reads from. RGBA_8888 is the only
//! //    format the native code accepts; 2-3 buffers keep latency low.
//! val metrics = resources.displayMetrics
//! val reader = ImageReader.newInstance(
//!         metrics.widthPixels, metrics.heightPixels, PixelFormat.RGBA_8888, 3)
//!
//! // 4. Mirror the default display into the reader surface.
//! val display = getSystemService(DisplayManager::class.java)
//!         .getDisplay(Display.DEFAULT_DISPLAY)
//! projection.createVirtualDisplay(
//!         'fluid', metrics.widthPixels, metrics.heightPixels, metrics.densityDpi,
//!         DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
//!         reader.surface, null, Handler(Looper.getMainLooper()))
//!
//! // 5. Hand the JavaVM and a global reference to the reader to Rust. Do this
//! //    from a native method so the JNIEnv is a real per-process VM.
//! external fun nativeInstall(reader: ImageReader)
//! ```
//!
//! The native entry point is tiny: take the JNIEnv it is given, create a global
//! reference for the reader and call install:
//!
//! ```rust,ignore
//! #[unsafe(no_mangle)]
//! #[unsafe(no_mangle)]
//! pub extern "system" fn Java_com_example_app_NativeBridge_nativeInstall(
//!     raw_env: jni::sys::JNIEnv,
//!     _class: jni::sys::jclass,
//!     reader: jni::sys::jobject,
//! ) {
//!     // SAFETY: a JNI entry point is handed a valid JNIEnv for this thread.
//!     let env = unsafe { jni::Env::from_raw(raw_env) };
//!     // SAFETY: reader is the non-null ImageReader argument of this native method.
//!     let local = unsafe { jni::objects::JObject::from_raw(&env, reader) };
//!     let global = env.new_global_ref(local).expect("global ref");
//!     let vm = env.get_java_vm().expect("JavaVM");
//!     fluid_capture::android::install(vm.get_raw(), global).expect("install reader");
//! }
//! ```
//!
//! ## What a frame looks like
//!
//! * Rgba8, tightly packed, alpha forced to 255.
//! * Dimensions are the configured region size, or the full display size.
//! * pts_ns is the wall clock at copy time; index counts delivered frames.
//! * When no new image is ready before CaptureConfig::timeout_ms, the previous
//!   frame is delivered again with a fresh pts_ns/index (exactly like the DXGI
//!   backend), so next_frame never reports the end of a live stream.
//!
//! ## Readback paths
//!
//! The preferred path is Image.getHardwareBuffer() (API 28+) handed to
//! AHardwareBuffer_lock, which avoids one copy. If the method is missing, the
//! lock fails or the buffer is not RGBA8, the backend falls back to the
//! officially supported Image.Plane direct ByteBuffer, honouring
//! getRowStride() and getPixelStride(). Both paths feed the same tested
//! stride-aware copy routine in rowcopy.
//!
//! ## Threading and lifecycle
//!
//! install may be called from any JVM-attached thread; the JavaVM and the
//! reader global reference are stored once and cloned into each capture
//! source. Frames are pulled from whatever thread calls next_frame, which the
//! engine owns for the whole session. The reader must stay alive while a source
//! is open: uninstall only drops our reference. Rotation or a display size
//! change cancels the MediaProjection session; create a new reader, call
//! install again and reopen the capture source.

use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use fluid_core::{
    CaptureConfig, Dims, FluidError, Frame, FrameSource, PixelFormat, Rect, Result, now_ns,
    sleep_until,
};
use jni::objects::{JByteBuffer, JObject, JObjectArray};
use jni::refs::{Global, Reference};
use jni::{Env, JValueOwned, JavaVM, jni_sig, jni_str};

use crate::MonitorInfo;
use crate::rowcopy;

/// android.graphics.PixelFormat.RGBA_8888, the only accepted reader format.
const PIXEL_FORMAT_RGBA_8888: i32 = 1;

/// Refresh rate reported for the mirrored display (MediaProjection does not
/// report one; the engine only uses this as a hint when max_fps is unset).
const NOMINAL_REFRESH_HZ: f32 = 60.0;

/// Owned global reference to the Kotlin android.media.ImageReader.
///
/// jni::objects::GlobalRef is still available in jni 0.22 but is a deprecated
/// alias with a generic parameter; this alias keeps the android API readable.
pub type ImageReaderRef = Global<JObject<'static>>;

/// One installed MediaProjection session.
struct Projection {
    jvm: JavaVM,
    reader: ImageReaderRef,
    dims: Dims,
}

static INSTALLED: OnceLock<Mutex<Option<Arc<Projection>>>> = OnceLock::new();

fn registry() -> &'static Mutex<Option<Arc<Projection>>> {
    INSTALLED.get_or_init(|| Mutex::new(None))
}

/// Install the Kotlin-owned MediaProjection reader and make it the capture
/// source for this process.
///
/// * jvm must be the process JavaVM pointer (from JNI_OnLoad, a native method
///   Env, or JNIEnv::GetJavaVM). It is not owned by us and must stay valid for
///   the lifetime of the process, which is the normal JNI contract.
/// * image_reader is an owned global reference; the backend keeps it alive, so
///   Kotlin may delete its own reference after the call.
///
/// Calling this again replaces the previous session (for example after a
/// rotation recreated the projection). Querying the reader also validates the
/// format and records the display size, so a wrongly sized reader fails here
/// instead of producing garbage frames later.
///
/// Must be called from a thread that is attached to the JVM (a JNI entry point
/// is, by definition).
// The pointer is never dereferenced directly: it is wrapped by JavaVM::from_raw,
// whose validity contract is documented on the function. Keeping the entry point
// safe is convenient for JNI glue code, which has to call it without unsafe.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn install(jvm: *mut jni::sys::JavaVM, image_reader: ImageReaderRef) -> Result<()> {
    if jvm.is_null() {
        return Err(FluidError::invalid(
            "android capture: JavaVM pointer must not be null",
        ));
    }
    if image_reader.as_obj().is_null() {
        return Err(FluidError::invalid(
            "android capture: ImageReader global reference must not be null",
        ));
    }
    // SAFETY: the caller passes the process JavaVM pointer, which outlives the
    // backend by the JNI contract; from_raw only wraps it, it does not take
    // ownership.
    let jvm = unsafe { JavaVM::from_raw(jvm) };
    let dims = jvm.attach_current_thread(|env| -> std::result::Result<Dims, GrabError> {
        let width = env
            .call_method(
                image_reader.as_obj(),
                jni_str!("getWidth"),
                jni_sig!("()I"),
                &[],
            )?
            .i()?;
        let height = env
            .call_method(
                image_reader.as_obj(),
                jni_str!("getHeight"),
                jni_sig!("()I"),
                &[],
            )?
            .i()?;
        let format = env
            .call_method(
                image_reader.as_obj(),
                jni_str!("getImageFormat"),
                jni_sig!("()I"),
                &[],
            )?
            .i()?;
        if width <= 0 || height <= 0 {
            return Err(GrabError::Fluid(FluidError::invalid(format!(
                "android capture: ImageReader reports an invalid size {width}x{height}"
            ))));
        }
        if format != PIXEL_FORMAT_RGBA_8888 {
            return Err(GrabError::Fluid(FluidError::invalid(format!(
                "android capture: ImageReader must use PixelFormat.RGBA_8888 ({PIXEL_FORMAT_RGBA_8888}), got format {format}"
            ))));
        }
        Ok(Dims::new(width as u32, height as u32))
    })?;
    let projection = Arc::new(Projection {
        jvm,
        reader: image_reader,
        dims,
    });
    let mut slot = registry()
        .lock()
        .map_err(|_| FluidError::internal("android capture: installation lock is poisoned"))?;
    *slot = Some(projection);
    Ok(())
}

/// True once a MediaProjection session has been installed.
pub fn is_installed() -> bool {
    registry()
        .lock()
        .map(|slot| slot.is_some())
        .unwrap_or(false)
}

/// Drop our reference to the installed reader (capture sources fail until
/// install is called again). The Kotlin ImageReader itself is not closed.
///
/// Call this from a JVM-attached thread so dropping the global reference does
/// not have to attach: the JNI layer deletes global references on the calling
/// thread.
pub fn uninstall() -> Result<()> {
    let mut slot = registry()
        .lock()
        .map_err(|_| FluidError::internal("android capture: installation lock is poisoned"))?;
    *slot = None;
    Ok(())
}

fn active_projection() -> Result<Arc<Projection>> {
    let slot = registry()
        .lock()
        .map_err(|_| FluidError::internal("android capture: installation lock is poisoned"))?;
    slot.clone().ok_or_else(|| {
        FluidError::unsupported(
            "android capture: no MediaProjection session installed; call fluid_capture::android::install(JavaVM, ImageReader global ref) from JNI first",
        )
    })
}

/// Report the mirrored display as a single synthetic monitor.
///
/// MediaProjection captures a virtual display, not a physical monitor, so there
/// is exactly one entry once install has run and none before that.
pub(crate) fn enumerate_monitors() -> Result<Vec<MonitorInfo>> {
    let Some(projection) = installed_projection() else {
        return Ok(Vec::new());
    };
    Ok(vec![MonitorInfo {
        index: 0,
        name: "android-display".to_string(),
        dims: projection.dims,
        refresh_hz: NOMINAL_REFRESH_HZ,
        primary: true,
        left: 0,
        top: 0,
    }])
}

fn installed_projection() -> Option<Arc<Projection>> {
    registry().lock().ok().and_then(|slot| slot.clone())
}

pub(crate) fn open(config: &CaptureConfig) -> Result<Box<dyn FrameSource>> {
    let projection = active_projection()?;
    let screen = projection.dims;
    let crop = match config.region {
        Some(region) => {
            let clipped = Rect::from(region).intersect(screen);
            if clipped.is_empty() {
                return Err(FluidError::invalid(format!(
                    "capture.region {}x{} at ({}, {}) does not intersect the {}x{} display",
                    region.w, region.h, region.x, region.y, screen.width, screen.height
                )));
            }
            clipped
        }
        None => Rect::new(0, 0, screen.width, screen.height),
    };
    Ok(Box::new(AndroidSource {
        projection,
        dims: crop.dims(),
        crop,
        fps_cap: config.max_fps,
        timeout_ms: config.timeout_ms.max(1),
        index: 0,
        delivered: 0,
        start: Instant::now(),
        cached: None,
        closed: false,
        hardware_ok: true,
    }))
}

struct AndroidSource {
    projection: Arc<Projection>,
    /// Dimensions of the frames handed to the engine (the crop size).
    dims: Dims,
    /// Region of the image to read, in display coordinates.
    crop: Rect,
    fps_cap: f32,
    timeout_ms: u32,
    index: u64,
    delivered: u64,
    start: Instant,
    cached: Option<Frame>,
    closed: bool,
    /// False once the hardware-buffer fast path proved unusable, so later
    /// frames go straight to the plane path.
    hardware_ok: bool,
}

impl AndroidSource {
    /// Throttle to max_fps against the first frame, so jitter does not
    /// accumulate into drift (same policy as the mock and DXGI sources).
    fn pace(&self) {
        if self.fps_cap > 0.0 && self.fps_cap.is_finite() && self.delivered > 0 {
            let period = Duration::from_secs_f64(1.0 / f64::from(self.fps_cap));
            sleep_until(self.start + period.mul_f64(self.delivered as f64));
        }
    }

    /// Stamp, count and cache a frame; returns it to the caller.
    fn deliver(&mut self, mut frame: Frame) -> Frame {
        frame.pts_ns = now_ns();
        frame.index = self.index;
        self.index += 1;
        self.delivered += 1;
        self.cached = Some(frame.clone());
        frame
    }

    /// Pull one image out of the reader, if the producer has one ready.
    fn grab(&mut self) -> Result<Option<(Frame, bool)>> {
        let projection = Arc::clone(&self.projection);
        let image_dims = projection.dims;
        let crop = self.crop;
        let try_hardware = self.hardware_ok;
        let result = projection.jvm.attach_current_thread(
            |env: &mut Env| -> std::result::Result<Option<(Frame, bool)>, GrabError> {
                let acquired = env.call_method(
                    projection.reader.as_obj(),
                    jni_str!("acquireLatestImage"),
                    jni_sig!("()Landroid/media/Image;"),
                    &[],
                )?;
                let image = match acquired {
                    JValueOwned::Object(image) => image,
                    _ => return Ok(None),
                };
                if image.is_null() {
                    return Ok(None);
                }
                let frame = read_image(env, &image, image_dims, crop, try_hardware);
                // The Image owns a native graphic buffer that is only recycled
                // once closed; always close it, even when readback failed.
                let _ = env.call_method(&image, jni_str!("close"), jni_sig!("()V"), &[]);
                frame.map(Some)
            },
        );
        Ok(result?)
    }
}

impl FrameSource for AndroidSource {
    fn dims(&self) -> Dims {
        self.dims
    }

    fn fps(&self) -> f32 {
        if self.fps_cap > 0.0 {
            self.fps_cap
        } else {
            NOMINAL_REFRESH_HZ
        }
    }

    fn backend_name(&self) -> &'static str {
        "android-mediaprojection"
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        if self.closed {
            return Ok(None);
        }
        self.pace();
        let deadline = Instant::now() + Duration::from_millis(u64::from(self.timeout_ms));
        loop {
            match self.grab()? {
                Some((frame, hardware_ok)) => {
                    self.hardware_ok = hardware_ok;
                    return Ok(Some(self.deliver(frame)));
                }
                None => {
                    if Instant::now() >= deadline {
                        break;
                    }
                    // acquireLatestImage is level-triggered but the producer
                    // runs on another thread; a short poll keeps latency low.
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
        // No frame before the timeout: repeat the last one (or black before the
        // very first frame) so the engine keeps a steady frame cadence.
        let repeated = self
            .cached
            .clone()
            .unwrap_or_else(|| Frame::black(self.dims, PixelFormat::Rgba8));
        Ok(Some(self.deliver(repeated)))
    }

    fn close(&mut self) {
        self.closed = true;
        self.cached = None;
    }
}

/// Error type local to this module so JNI and fluid errors can share the
/// attach_current_thread callback (whose error type only has to implement
/// From<jni::errors::Error>).
enum GrabError {
    Jni(jni::errors::Error),
    Fluid(FluidError),
}

impl From<jni::errors::Error> for GrabError {
    fn from(error: jni::errors::Error) -> Self {
        GrabError::Jni(error)
    }
}

impl From<FluidError> for GrabError {
    fn from(error: FluidError) -> Self {
        GrabError::Fluid(error)
    }
}

impl From<GrabError> for FluidError {
    fn from(error: GrabError) -> Self {
        match error {
            GrabError::Jni(error) => FluidError::platform(format!(
                "android capture: JNI call into android.media.ImageReader failed: {error}"
            )),
            GrabError::Fluid(error) => error,
        }
    }
}

/// Read the current image, preferring the zero-copy hardware buffer path.
///
/// The returned bool says whether the hardware path may still be attempted on
/// later frames.
fn read_image(
    env: &mut Env,
    image: &JObject,
    image_dims: Dims,
    crop: Rect,
    try_hardware: bool,
) -> std::result::Result<(Frame, bool), GrabError> {
    if try_hardware {
        match read_hardware_buffer(env, image, image_dims, crop) {
            Ok(Some(frame)) => return Ok((frame, true)),
            // No hardware buffer was attached to this image; keep trying later.
            Ok(None) => return Ok((read_planes(env, image, image_dims, crop)?, true)),
            // API < 28, a closed buffer or a rejected lock: fall back for good.
            Err(_unavailable) => return Ok((read_planes(env, image, image_dims, crop)?, false)),
        }
    }
    Ok((read_planes(env, image, image_dims, crop)?, false))
}

/// Zero-copy readback through Image.getHardwareBuffer() + AHardwareBuffer.
///
/// Returns Ok(None) when this image has no usable hardware buffer and
/// Err when the fast path is unavailable on this device, in which case the
/// caller disables it for the rest of the session.
fn read_hardware_buffer(
    env: &mut Env,
    image: &JObject,
    image_dims: Dims,
    crop: Rect,
) -> std::result::Result<Option<Frame>, GrabError> {
    let value = env.call_method(
        image,
        jni_str!("getHardwareBuffer"),
        jni_sig!("()Landroid/hardware/HardwareBuffer;"),
        &[],
    )?;
    let buffer = match value {
        JValueOwned::Object(buffer) => buffer,
        _ => return Ok(None),
    };
    if buffer.is_null() {
        return Ok(None);
    }
    // SAFETY: env is a valid JNIEnv for this thread and buffer is a live local
    // reference to an android.hardware.HardwareBuffer. The returned
    // AHardwareBuffer is owned by the Java object (do not release it) and stays
    // valid while buffer is alive, which is until this function returns.
    let raw = unsafe {
        ndk_sys::AHardwareBuffer_fromHardwareBuffer(
            env.get_raw() as *mut _,
            buffer.as_raw() as *mut _,
        )
    };
    if raw.is_null() {
        return Ok(None);
    }
    let mut desc = ndk_sys::AHardwareBuffer_Desc {
        width: 0,
        height: 0,
        layers: 0,
        format: 0,
        usage: 0,
        stride: 0,
        rfu0: 0,
        rfu1: 0,
    };
    // SAFETY: raw is a valid borrowed AHardwareBuffer and desc is writable.
    unsafe { ndk_sys::AHardwareBuffer_describe(raw, &mut desc) };
    let rgba = ndk_sys::AHardwareBuffer_Format::AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM.0;
    let xrgb = ndk_sys::AHardwareBuffer_Format::AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM.0;
    if desc.format != rgba && desc.format != xrgb {
        return Ok(None);
    }
    if desc.width != image_dims.width || desc.height != image_dims.height {
        return Ok(None);
    }
    let stride_px = desc.stride as usize;
    if stride_px == 0 {
        return Ok(None);
    }
    // AHardwareBuffer_Desc::stride is expressed in pixels, not bytes.
    let Some(row_stride) = stride_px.checked_mul(rowcopy::OUT_PIXEL_BYTES) else {
        return Ok(None);
    };
    let Some(total) = row_stride.checked_mul(desc.height as usize) else {
        return Ok(None);
    };
    let usage = hardware_buffer_read_usage();
    let mut pixels: *mut c_void = std::ptr::null_mut();
    // SAFETY: raw is a live borrowed AHardwareBuffer; fence -1 means no explicit
    // sync fence and a null rect means the whole buffer.
    let lock =
        unsafe { ndk_sys::AHardwareBuffer_lock(raw, usage, -1, std::ptr::null(), &mut pixels) };
    if lock != 0 || pixels.is_null() {
        return Ok(None);
    }
    // SAFETY: a successful lock grants CPU access to total bytes until the
    // matching unlock below. The slice does not outlive the lock.
    let source = unsafe { std::slice::from_raw_parts(pixels as *const u8, total) };
    let dst_len = (crop.w as usize)
        .saturating_mul(crop.h as usize)
        .saturating_mul(rowcopy::OUT_PIXEL_BYTES);
    let mut data = vec![0u8; dst_len];
    let copied = rowcopy::copy_window_to_rgba8(
        source,
        row_stride,
        rowcopy::OUT_PIXEL_BYTES,
        image_dims,
        crop,
        &mut data,
    );
    // SAFETY: balances the successful lock above; the buffer itself stays owned
    // by the Java HardwareBuffer object.
    let _ = unsafe { ndk_sys::AHardwareBuffer_unlock(raw, std::ptr::null_mut()) };
    copied.map_err(GrabError::Fluid)?;
    Ok(Some(Frame {
        dims: crop.dims(),
        fmt: PixelFormat::Rgba8,
        data,
        pts_ns: 0,
        index: 0,
    }))
}

/// Portable fallback: read the direct ByteBuffer of Image.Plane 0.
///
/// This is the API every Android version since 21 supports; the strides are
/// producer-chosen (row padding and, in theory, interleaved pixels), so all the
/// arithmetic goes through the shared row-copy helper.
fn read_planes(
    env: &mut Env,
    image: &JObject,
    image_dims: Dims,
    crop: Rect,
) -> std::result::Result<Frame, GrabError> {
    let value = env.call_method(
        image,
        jni_str!("getPlanes"),
        jni_sig!("()[Landroid/media/Image$Plane;"),
        &[],
    )?;
    let planes_object = match value {
        JValueOwned::Object(planes) => planes,
        _ => {
            return Err(GrabError::Fluid(FluidError::platform(
                "android capture: Image.getPlanes returned a non-array value",
            )));
        }
    };
    if planes_object.is_null() {
        return Err(GrabError::Fluid(FluidError::platform(
            "android capture: Image.getPlanes returned null",
        )));
    }
    // SAFETY: the value came straight from Image.getPlanes, so it is a local
    // reference to an Image.Plane[]; into_raw transfers ownership to the
    // wrapper so no reference is deleted twice.
    let planes: JObjectArray<'_, JObject> = unsafe {
        JObjectArray::<JObject>::from_raw(env, planes_object.into_raw() as jni::sys::jobjectArray)
    };
    let count = planes.len(env)?;
    if count == 0 {
        return Err(GrabError::Fluid(FluidError::platform(
            "android capture: Image reports no planes",
        )));
    }
    let plane = planes.get_element(env, 0)?;
    if plane.is_null() {
        return Err(GrabError::Fluid(FluidError::platform(
            "android capture: Image.Plane is null",
        )));
    }
    let row_stride = env
        .call_method(&plane, jni_str!("getRowStride"), jni_sig!("()I"), &[])?
        .i()?;
    let pixel_stride = env
        .call_method(&plane, jni_str!("getPixelStride"), jni_sig!("()I"), &[])?
        .i()?;
    let value = env.call_method(
        &plane,
        jni_str!("getBuffer"),
        jni_sig!("()Ljava/nio/ByteBuffer;"),
        &[],
    )?;
    let buffer_object = match value {
        JValueOwned::Object(buffer) => buffer,
        _ => {
            return Err(GrabError::Fluid(FluidError::platform(
                "android capture: Image.Plane.getBuffer returned a non-buffer value",
            )));
        }
    };
    let buffer = env.cast_local::<JByteBuffer>(buffer_object)?;
    let capacity = env.get_direct_buffer_capacity(&buffer)?;
    let address = env.get_direct_buffer_address(&buffer)?;
    if address.is_null() || capacity == 0 {
        return Err(GrabError::Fluid(FluidError::platform(
            "android capture: Image.Plane buffer is not a mapped direct buffer",
        )));
    }
    // SAFETY: a direct ByteBuffer reports a stable address valid while the
    // Image (and therefore its planes) is open; we only read it and the caller
    // closes the Image after this function returns.
    let source = unsafe { std::slice::from_raw_parts(address as *const u8, capacity) };
    let dst_len = (crop.w as usize)
        .saturating_mul(crop.h as usize)
        .saturating_mul(rowcopy::OUT_PIXEL_BYTES);
    let mut data = vec![0u8; dst_len];
    rowcopy::copy_window_to_rgba8(
        source,
        row_stride.max(0) as usize,
        pixel_stride.max(0) as usize,
        image_dims,
        crop,
        &mut data,
    )?;
    Ok(Frame {
        dims: crop.dims(),
        fmt: PixelFormat::Rgba8,
        data,
        pts_ns: 0,
        index: 0,
    })
}

/// AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN as the u64 the NDK expects.
///
/// C c_ulong is 64-bit on arm64 but 32-bit on armv7, so the cast is needed for
/// some Android ABIs even though it looks redundant on this one.
#[allow(clippy::unnecessary_cast)]
fn hardware_buffer_read_usage() -> u64 {
    ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN.0 as u64
}
