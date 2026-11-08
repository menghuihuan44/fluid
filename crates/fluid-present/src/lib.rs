//! Cross-platform frame presentation.
//!
//! Every sink implements `fluid_core::FrameSink` so the engine stays platform
//! agnostic. Frames are copied into a backend-specific scratch buffer and
//! pushed synchronously, so the sink never retains caller memory.
//!
//! Backends:
//! - Windows: DXGI flip-model swap chain (backend name dxgi-swapchain) for the
//!   lowest-latency path, and a borderless always-on-top Win32 window blitted
//!   with GDI (backend name win32-window) as the compatible path. See
//!   swapchain.rs and win.rs; both share the window plumbing in window.rs.
//! - Any:     `NullSink` for headless benchmarking and file output.
//! - Android: ANativeWindow surface (cfg(target_os = "android")); the app
//!   registers its SurfaceView window with android::install first, see the
//!   android module documentation for the Kotlin glue.
//!
//! Backend selection follows `PresentConfig::backend`: `window` forces GDI,
//! `dxgi` forces the swap chain (a real error is returned if the driver cannot
//! provide one), and `auto` tries the swap chain first and falls back to GDI
//! with a warning, so a session without D3D11 still shows frames.

use fluid_core::{Dims, FluidError, Frame, FrameSink, PresentConfig, Result};

// Only the Android backend consumes this module on device; the helper is still
// compiled on every platform so its unit tests run in CI.
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
mod blit;

/// Android ANativeWindow backend (only compiled on Android).
///
/// The app must install the SurfaceView's ANativeWindow before calling
/// [`open`]; the module docs show the Kotlin glue and the JNI entry point.
#[cfg(target_os = "android")]
pub mod android;
#[cfg(windows)]
mod swapchain;
#[cfg(windows)]
mod win;
#[cfg(windows)]
mod window;

/// Open the configured present backend at the given output resolution.
///
/// The returned sink must be used from one thread at a time. On Windows the
/// window is created on the calling thread and all later calls (present,
/// should_close, close, drop) must happen on that same thread.
///
/// Accepted hints: `vsync` waits for the display refresh (Present sync
/// interval 1 on the swap chain path, a bounded frame limiter on the GDI path);
/// `fullscreen` stretches the window over the whole monitor; `borderless`
/// drops the title bar; `monitor` selects the target display (clamped to the
/// last display when out of range); `title` names the window;
/// `close_on_escape` makes VK_ESCAPE request a close; `low_latency` is
/// accepted as a hint (the flip-model path already presents with the minimum
/// queue the compositor allows, the GDI path has no queue to bypass).
pub fn open(config: &PresentConfig, dims: Dims) -> Result<Box<dyn FrameSink>> {
    if dims.is_empty() {
        return Err(FluidError::invalid(
            "present: frame dimensions must be non-zero",
        ));
    }
    open_native(config, dims)
}

#[cfg(windows)]
fn open_native(config: &PresentConfig, dims: Dims) -> Result<Box<dyn FrameSink>> {
    use fluid_core::PresentBackendKind;
    match config.backend {
        PresentBackendKind::Window => win::open(config, dims),
        PresentBackendKind::Dxgi => swapchain::open(config, dims),
        PresentBackendKind::Auto => match swapchain::open(config, dims) {
            Ok(sink) => Ok(sink),
            Err(err) => {
                fluid_core::fluid_warn!(
                    "dxgi swap chain presentation unavailable ({err}); falling back to the Win32 GDI window"
                );
                win::open(config, dims)
            }
        },
    }
}

#[cfg(target_os = "android")]
fn open_native(config: &PresentConfig, dims: Dims) -> Result<Box<dyn FrameSink>> {
    use fluid_core::PresentBackendKind;
    match config.backend {
        PresentBackendKind::Dxgi => Err(FluidError::unsupported(
            "dxgi presentation is only available on Windows (Android uses ANativeWindow)",
        )),
        // Android has exactly one on-screen backend, so window and auto both
        // mean the ANativeWindow surface.
        PresentBackendKind::Window | PresentBackendKind::Auto => android::open(config, dims),
    }
}

#[cfg(not(any(windows, target_os = "android")))]
fn open_native(config: &PresentConfig, _dims: Dims) -> Result<Box<dyn FrameSink>> {
    use fluid_core::PresentBackendKind;
    if config.backend == PresentBackendKind::Dxgi {
        return Err(FluidError::unsupported(
            "dxgi presentation is only available on Windows",
        ));
    }
    Err(fluid_core::FluidError::unsupported(
        "no presentation backend is compiled for this platform (Windows uses a D3D11 flip swap chain or a Win32 window, Android needs ANativeWindow)",
    ))
}

/// Headless sink: consumes frames, keeps statistics, never displays anything.
pub struct NullSink {
    dims: Dims,
    frames: u64,
    bytes: u64,
    closed: bool,
}

impl NullSink {
    pub fn new(dims: Dims) -> Self {
        Self {
            dims,
            frames: 0,
            bytes: 0,
            closed: false,
        }
    }

    pub fn frames(&self) -> u64 {
        self.frames
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl FrameSink for NullSink {
    fn dims(&self) -> Dims {
        self.dims
    }

    fn backend_name(&self) -> &'static str {
        "null"
    }

    fn present(&mut self, frame: &Frame) -> Result<()> {
        if self.closed {
            return Err(FluidError::cancelled());
        }
        self.frames += 1;
        self.bytes += frame.data.len() as u64;
        Ok(())
    }

    fn should_close(&self) -> bool {
        self.closed
    }

    fn close(&mut self) {
        self.closed = true;
    }
}

/// Convenience constructor for the headless sink.
pub fn null_sink(dims: Dims) -> NullSink {
    NullSink::new(dims)
}

/// Platform capability probe used by `fluid devices` and `fluid doctor`.
///
/// The boolean says whether this build can actually present that way right
/// now; "window" reports whether any on-screen backend exists at all, and on
/// Windows "dxgi-swapchain" asks the driver for the D3D11 device and DXGI
/// factory a flip-model chain needs (false on machines without a usable
/// hardware device, for example a bare remote session).
pub fn backend_report() -> Vec<(String, bool)> {
    let mut v = vec![("null".to_string(), true)];
    v.push((
        "window".to_string(),
        cfg!(any(windows, target_os = "android")),
    ));
    #[cfg(windows)]
    {
        v.push(("win32-window".to_string(), true));
        v.push(("dxgi-swapchain".to_string(), swapchain::available()));
    }
    #[cfg(target_os = "android")]
    {
        v.push(("android-anativewindow".to_string(), true));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluid_core::PixelFormat;

    #[test]
    fn null_sink_counts() {
        let mut s = null_sink(Dims::new(8, 8));
        let f = Frame::black(Dims::new(8, 8), PixelFormat::Rgba8);
        s.present(&f).unwrap();
        s.present(&f).unwrap();
        assert_eq!(s.frames(), 2);
        assert!(!s.should_close());
        s.close();
        assert!(s.should_close());
        assert!(s.present(&f).is_err());
    }

    #[test]
    fn null_sink_tracks_bytes() {
        let mut s = null_sink(Dims::new(4, 4));
        let rgba = Frame::black(Dims::new(4, 4), PixelFormat::Rgba8);
        let rgb = Frame::black(Dims::new(4, 4), PixelFormat::Rgb8);
        s.present(&rgba).unwrap();
        s.present(&rgb).unwrap();
        assert_eq!(s.bytes(), (4 * 4 * 4 + 4 * 4 * 3) as u64);
        assert_eq!(s.dims(), Dims::new(4, 4));
        assert_eq!(s.backend_name(), "null");
    }

    #[test]
    fn backend_report_lists_backends() {
        let report = backend_report();
        assert!(report.iter().any(|(name, ok)| name == "null" && *ok));
        assert!(report.iter().any(|(name, _)| name == "window"));
        assert!(report.iter().all(|(name, _)| !name.is_empty()));
        #[cfg(windows)]
        {
            assert!(
                report
                    .iter()
                    .any(|(name, ok)| name == "win32-window" && *ok)
            );
            // The value depends on the machine (a headless session has no
            // D3D11 device), but the backend must always be enumerated now.
            assert!(report.iter().any(|(name, _)| name == "dxgi-swapchain"));
        }
        #[cfg(target_os = "android")]
        assert!(
            report
                .iter()
                .any(|(name, ok)| name == "android-anativewindow" && *ok)
        );
    }

    #[test]
    fn open_rejects_empty_dims() {
        let config = PresentConfig::default();
        assert!(open(&config, Dims::new(0, 10)).is_err());
    }

    #[test]
    fn open_smoke_tolerates_headless_machines() {
        let config = PresentConfig {
            title: "Fluid test".to_string(),
            ..PresentConfig::default()
        };
        let dims = Dims::new(320, 180);
        match open(&config, dims) {
            Ok(mut sink) => {
                assert_eq!(sink.dims(), dims);
                assert!(!sink.backend_name().is_empty());
                assert!(!sink.should_close());
                // Do not present here: a test must not flash a real window
                // with arbitrary content on the developer's desktop.
                sink.close();
            }
            Err(err) => {
                // Headless sessions and unsupported platforms may refuse.
                assert!(!err.to_string().is_empty());
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn explicit_backends_open_or_fail_cleanly() {
        use fluid_core::PresentBackendKind;
        let dims = Dims::new(320, 180);
        for backend in [PresentBackendKind::Window, PresentBackendKind::Dxgi] {
            let config = PresentConfig {
                backend,
                title: "Fluid backend test".to_string(),
                ..PresentConfig::default()
            };
            match open(&config, dims) {
                Ok(mut sink) => {
                    let expected = if backend == PresentBackendKind::Dxgi {
                        "dxgi-swapchain"
                    } else {
                        "win32-window"
                    };
                    assert_eq!(sink.backend_name(), expected);
                    assert_eq!(sink.dims(), dims);
                    sink.close();
                }
                Err(err) => {
                    // An explicit backend may still be unavailable (no D3D11
                    // adapter, remote session); it must fail clearly, not panic.
                    assert!(!err.to_string().is_empty());
                }
            }
        }
    }

    #[cfg(not(any(windows, target_os = "android")))]
    #[test]
    fn dxgi_backend_is_unsupported_here() {
        use fluid_core::PresentBackendKind;
        let config = PresentConfig {
            backend: PresentBackendKind::Dxgi,
            ..PresentConfig::default()
        };
        let err = open(&config, Dims::new(320, 180)).unwrap_err();
        assert!(err.to_string().contains("dxgi"));
    }
}
