//! JNI glue between the Kotlin app shell (android/) and the Rust platform layer.
//!
//! The Android build never receives capture or presentation handles on its own:
//! Android only lets an app capture the screen after the *user* grants a
//! MediaProjection token, and presentation needs a Surface owned by the view
//! hierarchy. Both are Java objects, so the Kotlin shell passes them down here,
//! and this module then runs the real Fluid pipeline on their behalf.
//!
//! Entry points:
//!
//! * JNI_OnLoad stores the process JavaVM* so later native methods can attach to
//!   the JVM. Kotlin loads libfluid_gui.so, so this runs before any call.
//! * Java_com_fluid_app_FluidNative_nativeInstallReader converts the local
//!   ImageReader reference into a global reference and hands it to
//!   fluid_capture::android::install.
//! * Java_com_fluid_app_FluidNative_nativeInstallSurface forwards the JNIEnv and
//!   Surface to fluid_present::android::install_from_surface.
//! * Java_com_fluid_app_FluidNative_nativeStartEnhance / nativeStopEnhance /
//!   nativeIsRunning / nativeLastReport drive the pipeline from the phone UI.
//! * Java_com_fluid_app_FluidNative_nativeUninstallSurface / nativeUninstallReader
//!   release the platform handles again from SurfaceHolder / Service callbacks.
//!
//! Threading: capture and presentation are thread-affine on Android (the
//! MediaProjection ImageReader and the ANativeWindow must be opened, used and
//! closed on one thread). nativeStartEnhance therefore spawns exactly one
//! dedicated thread that opens the source, opens the sink, runs the pipeline and
//! closes both again. The UI thread only ever flips the cancellation flag and
//! reads a JSON status snapshot, so it never blocks on capture or on a phone
//! frame taking 50 ms or more.
//!
//! Return codes are mirrored by FluidNative on the Kotlin side; a negative value
//! means "the shell should show an error" and never throws across JNI.
//!
//! This module is compiled only for target_os = "android"; the desktop GUI keeps
//! its existing entry points untouched.

use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use jni::objects::{JObject, JString};
use jni::sys::{JNIEnv as RawJniEnv, JavaVM as RawJavaVm, jfloat, jint, jobject, jstring};

use fluid_core::{FluidConfig, InterpMode, ScaleMode, UpscaleFilter};
use fluid_engine::{Pipeline, PipelineConfig, PipelineReport};

/// JNI version reported by JNI_OnLoad. 1.6 is the baseline Android supports.
const JNI_VERSION_1_6: jint = 0x0001_0006;
/// Negative value returned to the JVM when JNI_OnLoad cannot initialise.
const JNI_ERR: jint = -1;

/// Error codes returned by the native methods; keep in sync with
/// android/app/src/main/java/com/fluid/app/FluidNative.kt.
const ERROR_NO_JAVA_VM: jint = -1;
const ERROR_BAD_ARGUMENT: jint = -2;
const ERROR_JNI: jint = -3;
const ERROR_INSTALL: jint = -4;
/// nativeStartEnhance was called while a pipeline thread is still alive.
const ERROR_BUSY: jint = -5;
/// The pipeline thread could not be spawned, or the config was rejected.
const ERROR_PIPELINE: jint = -6;

/// Process-wide JavaVM pointer captured by JNI_OnLoad.
///
/// The JVM owns the pointer and guarantees it outlives every native method, so a
/// plain atomic pointer is enough; it is never dereferenced here, only passed to
/// fluid_capture::android::install, which wraps it with JavaVM::from_raw.
static JAVA_VM: AtomicPtr<RawJavaVm> = AtomicPtr::new(ptr::null_mut());

/// State of the single pipeline thread.
#[derive(Default)]
struct EnhanceState {
    /// Handle of the last spawned thread; is_finished() tells whether it is done.
    join: Option<JoinHandle<()>>,
    /// Cancellation flag handed to the pipeline; set by nativeStopEnhance.
    cancel: Option<Arc<AtomicBool>>,
    /// When the current (or last) run started, for the elapsed counter.
    started: Option<Instant>,
    /// JSON of the last finished report (PipelineReport::to_json).
    last_report: Option<String>,
    /// Human readable error of the last run, if it failed.
    last_error: Option<String>,
    /// Wall clock duration of the last finished run, in seconds.
    last_elapsed_s: f64,
}

static ENHANCE: OnceLock<Mutex<EnhanceState>> = OnceLock::new();

fn enhance_state() -> &'static Mutex<EnhanceState> {
    ENHANCE.get_or_init(|| Mutex::new(EnhanceState::default()))
}

fn lock_state() -> MutexGuard<'static, EnhanceState> {
    match enhance_state().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// True while the pipeline thread is still running.
fn thread_alive(state: &EnhanceState) -> bool {
    state
        .join
        .as_ref()
        .map(|join| !join.is_finished())
        .unwrap_or(false)
}

/// JNI_OnLoad is called by the JVM right after System.loadLibrary.
#[unsafe(no_mangle)]
pub extern "system" fn JNI_OnLoad(vm: *mut RawJavaVm, _reserved: *mut c_void) -> jint {
    if vm.is_null() {
        return JNI_ERR;
    }
    JAVA_VM.store(vm, Ordering::Release);
    JNI_VERSION_1_6
}

fn installed_java_vm() -> std::result::Result<*mut RawJavaVm, jint> {
    let vm = JAVA_VM.load(Ordering::Acquire);
    if vm.is_null() {
        Err(ERROR_NO_JAVA_VM)
    } else {
        Ok(vm)
    }
}

/// Read a Java String argument into a Rust String.
fn java_string_arg(env: *mut RawJniEnv, value: jstring) -> std::result::Result<String, jint> {
    if env.is_null() || value.is_null() {
        return Err(ERROR_BAD_ARGUMENT);
    }

    // SAFETY: the JVM passes the JNIEnv* of the calling (attached) thread and a
    // local reference valid for the duration of this native method.
    // EnvUnowned::from_raw only wraps the pointer for the lifetime of this call.
    let mut unowned = unsafe { jni::EnvUnowned::from_raw(env) };
    let outcome = unowned.with_env_no_catch(|env| -> jni::errors::Result<String> {
        // SAFETY: value is a valid local reference owned by this JNI frame; the
        // wrapper does not take ownership of it.
        let string = unsafe { JString::from_raw(env, value) };
        string.try_to_string(env)
    });
    match outcome.into_outcome() {
        jni::Outcome::Ok(text) => Ok(text),
        jni::Outcome::Err(_) | jni::Outcome::Panic(_) => Err(ERROR_JNI),
    }
}

/// Create a Java String for a JNI return value; null on failure.
fn java_string_result(env: *mut RawJniEnv, text: &str) -> jstring {
    if env.is_null() {
        return ptr::null_mut();
    }

    // SAFETY: the JVM passes the JNIEnv* of the calling (attached) thread; the
    // returned local reference is owned by this JNI frame.
    let mut unowned = unsafe { jni::EnvUnowned::from_raw(env) };
    let outcome = unowned.with_env_no_catch(|env| -> jni::errors::Result<jstring> {
        let string = env.new_string(text)?;
        Ok(string.into_raw())
    });
    match outcome.into_outcome() {
        jni::Outcome::Ok(raw) => raw,
        jni::Outcome::Err(_) | jni::Outcome::Panic(_) => ptr::null_mut(),
    }
}

/// Escape a string for a small hand written JSON document.
fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other if (other as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", other as u32));
            }
            other => out.push(other),
        }
    }
    out
}

/// Build a FluidConfig from the four phone UI selections.
fn enhance_config(
    multiplier: jint,
    scale: jfloat,
    filter: &str,
    quality: &str,
) -> std::result::Result<FluidConfig, jint> {
    let mut config = FluidConfig::default();
    config.profile = "android".to_string();

    let multiplier = multiplier.clamp(1, 8) as u8;
    config.interpolator.mode = if multiplier >= 2 {
        InterpMode::Fixed
    } else {
        InterpMode::Off
    };
    config.interpolator.multiplier = multiplier;

    let preset = quality
        .parse::<fluid_core::QualityPreset>()
        .map_err(|_| ERROR_BAD_ARGUMENT)?;
    config.apply_preset(preset);

    config.upscaler.filter = UpscaleFilter::parse(filter).map_err(|_| ERROR_BAD_ARGUMENT)?;
    config.upscaler.scale.mode = ScaleMode::Factor;
    config.upscaler.scale.factor = if scale.is_finite() {
        scale.clamp(0.25, 8.0)
    } else {
        1.0
    };

    // The phone pipeline presents into a SurfaceView: queueing more than one
    // frame only adds latency there, and Android has no DWM-style vsync flush to
    // honour, so vsync stays off.
    config.latency.max_queue = 1;
    config.present.vsync = false;
    config.validate().map_err(|_| ERROR_BAD_ARGUMENT)?;
    Ok(config)
}

/// Run the pipeline once on the dedicated Android thread.
fn run_enhance(config: FluidConfig, cancel: Arc<AtomicBool>) {
    let started = Instant::now();
    let outcome = run_pipeline(&config, cancel);
    let elapsed = started.elapsed().as_secs_f64();

    let mut state = lock_state();
    state.last_elapsed_s = elapsed;
    state.cancel = None;
    match outcome {
        Ok(report) => {
            state.last_report = Some(report.to_json());
            state.last_error = None;
        }
        Err(message) => {
            state.last_report = None;
            state.last_error = Some(message);
        }
    }
}

fn run_pipeline(
    config: &FluidConfig,
    cancel: Arc<AtomicBool>,
) -> std::result::Result<PipelineReport, String> {
    let mut source = fluid_capture::open(&config.capture)
        .map_err(|error| format!("capture open failed: {error}"))?;
    let outcome = drive(config, &mut *source, cancel);
    // Closing the MediaProjection source is quick and must not be skipped on the
    // error paths either, so it runs before the result is returned.
    source.close();
    outcome
}

fn drive(
    config: &FluidConfig,
    source: &mut dyn fluid_core::FrameSource,
    cancel: Arc<AtomicBool>,
) -> std::result::Result<PipelineReport, String> {
    let src_dims = source.dims();
    if src_dims.is_empty() {
        return Err("capture reported empty frame dimensions".to_string());
    }

    let run_config = PipelineConfig {
        pace: true,
        stats_interval: None,
        verbose: false,
        close_on_finish: true,
        cancel: Some(cancel),
        ..Default::default()
    };
    let mut pipeline = Pipeline::with_config(config, src_dims, run_config)
        .map_err(|error| format!("pipeline setup failed: {error}"))?;
    let out_dims = pipeline.out_dims();
    let mut sink = fluid_present::open(&config.present, out_dims)
        .map_err(|error| format!("present open failed: {error}"))?;

    pipeline
        .run(source, &mut *sink)
        .map_err(|error| format!("pipeline failed: {error}"))
}

/// JSON status snapshot read by the Kotlin status poller.
fn status_json(state: &EnhanceState) -> String {
    let running = thread_alive(state);
    let name = if running {
        "running"
    } else if state.last_error.is_some() {
        "error"
    } else if state.last_report.is_some() {
        "finished"
    } else {
        "idle"
    };
    let elapsed = if running {
        state
            .started
            .map(|started| started.elapsed().as_secs_f64())
            .unwrap_or(0.0)
    } else {
        state.last_elapsed_s
    };

    let mut out = String::with_capacity(640);
    out.push_str(&format!(
        "{{\"running\":{},\"state\":\"{}\",\"elapsed_s\":{:.3},",
        running, name, elapsed
    ));
    match state.last_error.as_deref() {
        Some(error) => {
            out.push_str("\"error\":\"");
            out.push_str(&json_escape(error));
            out.push_str("\",");
        }
        None => out.push_str("\"error\":null,"),
    }
    match state.last_report.as_deref() {
        Some(report) => {
            out.push_str("\"report\":");
            out.push_str(report);
        }
        None => out.push_str("\"report\":null"),
    }
    out.push('}');
    out
}

/// FluidNative.nativeInstallReader(reader: ImageReader): Int
///
/// Creates the global reference the capture backend keeps for the session. The
/// global reference is owned by fluid_capture; Kotlin may drop its own reference
/// after this call, and the ImageReader stays usable because the global
/// reference keeps it alive.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_fluid_app_FluidNative_nativeInstallReader(
    env: *mut RawJniEnv,
    _this: jobject,
    reader: jobject,
) -> jint {
    if env.is_null() || reader.is_null() {
        return ERROR_BAD_ARGUMENT;
    }
    let vm = match installed_java_vm() {
        Ok(vm) => vm,
        Err(code) => return code,
    };

    // SAFETY: the JVM passes the JNIEnv* of the calling (attached) thread and a
    // local reference valid for the duration of this native method.
    let mut unowned = unsafe { jni::EnvUnowned::from_raw(env) };
    let outcome = unowned.with_env_no_catch(
        |env| -> jni::errors::Result<fluid_capture::android::ImageReaderRef> {
            // SAFETY: reader is a valid local reference owned by this JNI frame;
            // the wrapper does not take ownership of it.
            let reader_object = unsafe { JObject::from_raw(env, reader) };
            env.new_global_ref(reader_object)
        },
    );
    let global_reader = match outcome.into_outcome() {
        jni::Outcome::Ok(global) => global,
        jni::Outcome::Err(_) | jni::Outcome::Panic(_) => return ERROR_JNI,
    };

    match fluid_capture::android::install(vm, global_reader) {
        Ok(()) => 0,
        Err(_) => ERROR_INSTALL,
    }
}

/// FluidNative.nativeInstallSurface(surface: Surface): Int
///
/// The presenter converts the Java Surface into an ANativeWindow itself
/// (ANativeWindow_fromSurface), acquiring its own reference, so nothing beyond
/// this call is needed on the Kotlin side.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_fluid_app_FluidNative_nativeInstallSurface(
    env: *mut RawJniEnv,
    _this: jobject,
    surface: jobject,
) -> jint {
    if env.is_null() || surface.is_null() {
        return ERROR_BAD_ARGUMENT;
    }

    // SAFETY: the JVM guarantees a valid JNIEnv* and a local reference to the
    // Surface for the duration of this call; install_from_surface immediately
    // converts the Surface into an owned ANativeWindow reference.
    match unsafe { fluid_present::android::install_from_surface(env, surface) } {
        Ok(()) => 0,
        Err(_) => ERROR_INSTALL,
    }
}

/// FluidNative.nativeUninstallReader(): Int
///
/// Drops the MediaProjection ImageReader global reference. The Kotlin service
/// calls this after the pipeline has stopped and before it closes the reader.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_fluid_app_FluidNative_nativeUninstallReader(
    _env: *mut RawJniEnv,
    _this: jobject,
) -> jint {
    match fluid_capture::android::uninstall() {
        Ok(()) => 0,
        Err(_) => ERROR_INSTALL,
    }
}

/// FluidNative.nativeUninstallSurface(): Int
///
/// Releases the ANativeWindow acquired by nativeInstallSurface. Must be called
/// after the pipeline stopped (the presenter reads the window pointer on every
/// frame); the Kotlin shell stops the run first and only then uninstalls.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_fluid_app_FluidNative_nativeUninstallSurface(
    _env: *mut RawJniEnv,
    _this: jobject,
) -> jint {
    match fluid_present::android::uninstall() {
        Ok(()) => 0,
        Err(_) => ERROR_INSTALL,
    }
}

/// FluidNative.nativeStartEnhance(multiplier: Int, scale: Float, filter: String,
/// quality: String): Int
///
/// Starts one pipeline run on a dedicated background thread. Returns 0 when the
/// thread was spawned; ERROR_BUSY when a run is still in flight; ERROR_BAD_ARGUMENT
/// for an unparsable filter/quality or invalid scale; ERROR_PIPELINE when the
/// thread or the capture/present backends could not start.
///
/// The call returns immediately, before capture is opened, so a slow or failing
/// backend surfaces through nativeLastReport instead of blocking the UI.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_fluid_app_FluidNative_nativeStartEnhance(
    env: *mut RawJniEnv,
    _this: jobject,
    multiplier: jint,
    scale: jfloat,
    filter: jstring,
    quality: jstring,
) -> jint {
    let filter = match java_string_arg(env, filter) {
        Ok(text) => text,
        Err(code) => return code,
    };
    let quality = match java_string_arg(env, quality) {
        Ok(text) => text,
        Err(code) => return code,
    };
    let config = match enhance_config(multiplier, scale, &filter, &quality) {
        Ok(config) => config,
        Err(code) => return code,
    };

    let mut state = lock_state();
    // Reap a finished handle; refuse to start a second run while one is alive.
    if let Some(join) = state.join.take() {
        if !join.is_finished() {
            state.join = Some(join);
            return ERROR_BUSY;
        }
    }

    let cancel = Arc::new(AtomicBool::new(false));
    state.cancel = Some(Arc::clone(&cancel));
    state.started = Some(Instant::now());
    state.last_report = None;
    state.last_error = None;
    state.last_elapsed_s = 0.0;

    match thread::Builder::new()
        .name("fluid-android-pipeline".to_string())
        .spawn(move || run_enhance(config, cancel))
    {
        Ok(join) => {
            state.join = Some(join);
            0
        }
        Err(_) => {
            state.cancel = None;
            state.started = None;
            ERROR_PIPELINE
        }
    }
}

/// FluidNative.nativeStopEnhance(): Int
///
/// Sets the cooperative cancellation flag the pipeline checks once per frame.
/// Returns immediately: the pipeline returns its report and the thread ends on
/// its own; Kotlin polls nativeIsRunning/nativeLastReport for the result.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_fluid_app_FluidNative_nativeStopEnhance(
    _env: *mut RawJniEnv,
    _this: jobject,
) -> jint {
    let state = lock_state();
    if let Some(flag) = state.cancel.as_ref() {
        flag.store(true, Ordering::Relaxed);
    }
    0
}

/// FluidNative.nativeIsRunning(): Int
///
/// 1 while the pipeline thread is alive, 0 otherwise.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_fluid_app_FluidNative_nativeIsRunning(
    _env: *mut RawJniEnv,
    _this: jobject,
) -> jint {
    let state = lock_state();
    if thread_alive(&state) { 1 } else { 0 }
}

/// FluidNative.nativeLastReport(): String
///
/// JSON status document: {running, state, elapsed_s, error, report}. state is
/// one of idle / running / finished / error; report is the PipelineReport JSON
/// (source, sink, src, out, multiplier, filter, backend, interp_backend, preset,
/// fps, latency, ...) once a run finished, null otherwise.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_fluid_app_FluidNative_nativeLastReport(
    env: *mut RawJniEnv,
    _this: jobject,
) -> jstring {
    let document = {
        let state = lock_state();
        status_json(&state)
    };
    java_string_result(env, &document)
}
