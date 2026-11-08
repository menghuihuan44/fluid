//! Cross-platform screen capture.
//!
//! Every backend returns tight `Rgba8` frames and implements
//! `fluid_core::FrameSource` so the engine stays platform agnostic.
//!
//! Backends:
//! - Windows: DXGI Desktop Duplication (Windows 8+), lowest latency, no hooks.
//! - Windows: Window capture of one application window via Windows.Graphics.Capture
//!   (fallback: DXGI duplication cropped to the window), see `open_window`.
//! - Any:     `MockSource` synthetic pattern for tests, benchmarks and CI.
//! - Android: MediaProjection/ImageReader through JNI (cfg(target_os =
//!   "android"); the Kotlin side installs the reader first, see the `android`
//!   module documentation).
//!
//! The mock pattern is fully deterministic (`mock_frame(dims, phase, index)`
//! is a pure function), so tests, benchmarks and `fluid-video` can use it as
//! golden data. Live backends emit `Rgba8`, set `pts_ns` from the monotonic
//! wall clock at the moment the frame is copied and increment `index` once per
//! delivered frame; `next_frame` never returns `Ok(None)` while a live source
//! is open.

mod pattern;
// Desktop pixel-format conversion (HDR/10-bit -> 8-bit RGBA): pure functions,
// host-tested everywhere; only the Windows DXGI backends call them.
#[cfg_attr(not(windows), allow(dead_code))]
mod format;
// Only the Android backend consumes this module on device; the helper is still
// compiled on every platform so its unit tests run in CI.
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
mod rowcopy;

/// Android MediaProjection backend (only compiled on Android).
///
/// The host app must create the MediaProjection + ImageReader and hand the
/// reader to `android::install` before calling [`open`]; the module docs show
/// the Kotlin glue and the JNI entry point.
#[cfg(target_os = "android")]
pub mod android;
#[cfg(windows)]
mod cursor;
#[cfg(windows)]
mod win;
// Window capture types and pure selection/geometry helpers: compiled on every
// platform so their unit tests run in CI; only Windows has a backend.
#[cfg(windows)]
mod win_window;
#[cfg_attr(not(windows), allow(dead_code))]
mod window;

pub use window::{WindowInfo, WindowSelector};

use fluid_core::{
    CaptureBackendKind, CaptureConfig, Dims, Frame, FrameSource, Rect, Result, now_ns, sleep_until,
};
use std::time::{Duration, Instant};

/// A monitor as reported by the OS.
#[derive(Clone, Debug, PartialEq)]
pub struct MonitorInfo {
    pub index: usize,
    pub name: String,
    pub dims: Dims,
    pub refresh_hz: f32,
    pub primary: bool,
    pub left: i32,
    pub top: i32,
}

impl std::fmt::Display for MonitorInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {} {}x{} @{:.1}Hz{} at ({}, {})",
            self.index,
            self.name,
            self.dims.width,
            self.dims.height,
            self.refresh_hz,
            if self.primary { " (primary)" } else { "" },
            self.left,
            self.top
        )
    }
}

/// Enumerate the platform monitors, ordered by desktop position.
///
/// The index reported here is the value `CaptureConfig::monitor` and
/// `PresentConfig::monitor` expect, and the same order is used by the
/// presentation backend. Platforms without a backend return an empty list.
pub fn enumerate_monitors() -> Result<Vec<MonitorInfo>> {
    enumerate_native()
}

#[cfg(windows)]
fn enumerate_native() -> Result<Vec<MonitorInfo>> {
    win::enumerate_monitors()
}

#[cfg(target_os = "android")]
fn enumerate_native() -> Result<Vec<MonitorInfo>> {
    android::enumerate_monitors()
}

#[cfg(not(any(windows, target_os = "android")))]
fn enumerate_native() -> Result<Vec<MonitorInfo>> {
    Ok(Vec::new())
}

/// True when a backend can be opened on this platform/build.
pub fn backend_available(kind: CaptureBackendKind) -> bool {
    match kind {
        CaptureBackendKind::Mock => true,
        CaptureBackendKind::Auto => cfg!(windows) || cfg!(target_os = "android"),
        CaptureBackendKind::Dxgi => cfg!(windows),
    }
}

/// Open the best available capture backend for the configuration.
///
/// When `config.window` selects an application window, this is equivalent to
/// [`open_window`] with the matching selector (the `Mock` backend still wins so
/// headless demos and tests stay repeatable).
///
/// `Mock` ignores the monitor and captures a region sized synthetic pattern
/// (region size when configured, otherwise 1920x1080) at `max_fps` (or 60 Hz
/// when uncapped), which makes headless demos and benchmarks repeatable.
pub fn open(config: &CaptureConfig) -> Result<Box<dyn FrameSource>> {
    config.validate()?;
    if let Some(window) = &config.window
        && window.is_set()
        && config.backend != CaptureBackendKind::Mock
    {
        let selector = WindowSelector {
            id: (window.id != 0).then_some(window.id),
            title_contains: (!window.title.trim().is_empty()).then(|| window.title.clone()),
            process: (!window.process.trim().is_empty()).then(|| window.process.clone()),
        };
        return open_window(&selector, config);
    }
    match config.backend {
        CaptureBackendKind::Mock => {
            let dims = config
                .region
                .map(|region| Rect::from(region).dims())
                .filter(|dims| !dims.is_empty())
                .unwrap_or_else(|| Dims::new(1920, 1080));
            let fps = if config.max_fps > 0.0 {
                config.max_fps
            } else {
                60.0
            };
            Ok(Box::new(MockSource::new(dims, fps)))
        }
        CaptureBackendKind::Auto | CaptureBackendKind::Dxgi => open_native(config),
    }
}

#[cfg(windows)]
fn open_native(config: &CaptureConfig) -> Result<Box<dyn FrameSource>> {
    win::open(config)
}

#[cfg(target_os = "android")]
fn open_native(config: &CaptureConfig) -> Result<Box<dyn FrameSource>> {
    android::open(config)
}

#[cfg(not(any(windows, target_os = "android")))]
fn open_native(_config: &CaptureConfig) -> Result<Box<dyn FrameSource>> {
    Err(fluid_core::FluidError::unsupported(
        "no screen capture backend is compiled for this platform (Windows uses DXGI desktop duplication, Android needs MediaProjection)",
    ))
}

/// Enumerate capturable top-level application windows.
///
/// The foreground window comes first, the rest follow in z-order; invisible
/// windows, cloaked windows (other virtual desktops), tool windows and windows
/// without a title are skipped. Only Windows implements window capture today,
/// so other platforms return an `Unsupported` error instead of an empty list.
pub fn enumerate_windows() -> Result<Vec<WindowInfo>> {
    enumerate_windows_native()
}

#[cfg(windows)]
fn enumerate_windows_native() -> Result<Vec<WindowInfo>> {
    win_window::enumerate_windows()
}

#[cfg(not(windows))]
fn enumerate_windows_native() -> Result<Vec<WindowInfo>> {
    Err(fluid_core::FluidError::unsupported(
        "window capture is only implemented on Windows; Android would need MediaProjection with a window token, and other platforms have no backend",
    ))
}

/// Open a live capture source for one application window.
///
/// Windows uses Windows.Graphics.Capture (a window is captured even when it is
/// covered by another window) and falls back to DXGI desktop duplication cropped
/// to the window when WGC is unavailable; other platforms return
/// `Unsupported`. The output is always `Rgba8`, the size is fixed when the
/// source is opened (window content or the configured region inside it; reopen
/// to adopt a new window size) and `next_frame` never returns `Ok(None)`:
/// minimized or closed windows replay the last frame.
pub fn open_window(
    selector: &WindowSelector,
    config: &CaptureConfig,
) -> Result<Box<dyn FrameSource>> {
    config.validate()?;
    open_window_native(selector, config)
}

#[cfg(windows)]
fn open_window_native(
    selector: &WindowSelector,
    config: &CaptureConfig,
) -> Result<Box<dyn FrameSource>> {
    win_window::open(selector, config)
}

#[cfg(not(windows))]
fn open_window_native(
    _selector: &WindowSelector,
    _config: &CaptureConfig,
) -> Result<Box<dyn FrameSource>> {
    Err(fluid_core::FluidError::unsupported(
        "window capture is only implemented on Windows; Android would need MediaProjection with a window token, and other platforms have no backend",
    ))
}

/// Deterministic animated test pattern; used by mocks and by fluid-video.
///
/// The image is a scrolling gradient with a moving box on a Lissajous path and
/// a 32 pixel frame counter encoded in the bottom row (see `pattern`), so a
/// consumer can verify both motion and frame identity from pixels alone.
pub fn mock_frame(dims: Dims, phase: f64, index: u64) -> Frame {
    pattern::render(dims, phase, index)
}

/// Synthetic capture source with optional frame limit.
pub struct MockSource {
    dims: Dims,
    fps: f32,
    index: u64,
    limit: Option<u64>,
    start: Instant,
}

impl MockSource {
    pub fn new(dims: Dims, fps: f32) -> Self {
        Self {
            dims,
            fps,
            index: 0,
            limit: None,
            start: Instant::now(),
        }
    }

    pub fn with_limit(mut self, frames: u64) -> Self {
        self.limit = Some(frames);
        self
    }
}

impl FrameSource for MockSource {
    fn dims(&self) -> Dims {
        self.dims
    }

    fn fps(&self) -> f32 {
        self.fps
    }

    fn backend_name(&self) -> &'static str {
        "mock"
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        if let Some(limit) = self.limit
            && self.index >= limit
        {
            return Ok(None);
        }
        // Pace against the source start time (not per call) so small scheduling
        // hiccups do not accumulate into drift. The first frame is immediate.
        if self.fps > 0.0 && self.fps.is_finite() && self.index > 0 {
            let period = Duration::from_secs_f64(1.0 / self.fps as f64);
            sleep_until(self.start + period.mul_f64(self.index as f64));
        }
        // One full motion cycle every 120 frames keeps the pattern lively
        // without depending on wall-clock time (determinism).
        let phase = (self.index as f64 / 120.0).rem_euclid(1.0);
        let mut frame = pattern::render(self.dims, phase, self.index);
        frame.pts_ns = now_ns();
        frame.index = self.index;
        self.index += 1;
        Ok(Some(frame))
    }
}

/// Convenience constructor for the mock source.
pub fn mock_source(dims: Dims, fps: f32) -> MockSource {
    MockSource::new(dims, fps)
}

/// Apply a capture region to a full monitor frame.
pub fn crop_to_region(frame: &Frame, region: Option<Rect>) -> Result<Frame> {
    match region {
        Some(r) if !r.is_empty() => frame.crop(r),
        _ => Ok(frame.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use fluid_core::PixelFormat;

    #[test]
    fn mock_source_respects_limit() {
        let mut s = mock_source(Dims::new(16, 16), 0.0).with_limit(3);
        let mut n = 0;
        while s.next_frame().unwrap().is_some() {
            n += 1;
        }
        assert_eq!(n, 3);
        // The stream stays ended after the limit is reached.
        assert!(s.next_frame().unwrap().is_none());
    }

    #[test]
    fn mock_frames_differ() {
        let a = mock_frame(Dims::new(16, 16), 0.0, 0);
        let b = mock_frame(Dims::new(16, 16), 1.0, 1);
        assert_ne!(a.checksum(), b.checksum());
        assert_eq!(a.dims, Dims::new(16, 16));
        assert_eq!(a.fmt, PixelFormat::Rgba8);
    }

    #[test]
    fn open_window_validates_config_before_dispatch() {
        let selector = WindowSelector {
            id: Some(1),
            ..WindowSelector::default()
        };
        let config = CaptureConfig {
            region: Some(fluid_core::RectConfig {
                x: 0,
                y: 0,
                w: 0,
                h: 0,
            }),
            ..CaptureConfig::default()
        };
        let error = match open_window(&selector, &config) {
            Ok(source) => panic!("expected a config error, opened {}", source.backend_name()),
            Err(error) => error,
        };
        assert!(error.to_string().contains("region"));
    }

    #[test]
    fn enumerate_windows_is_sane_or_unsupported() {
        match enumerate_windows() {
            Ok(windows) => assert!(windows.len() <= 512),
            Err(fluid_core::FluidError::Unsupported(_)) => {}
            Err(fluid_core::FluidError::Platform(_)) => {}
            Err(error) => panic!("unexpected enumerate_windows error: {error}"),
        }
    }

    #[test]
    fn region_crop_works() {
        let f = mock_frame(Dims::new(32, 32), 0.0, 0);
        let c = crop_to_region(&f, Some(Rect::new(4, 4, 16, 8))).unwrap();
        assert_eq!(c.dims, Dims::new(16, 8));
    }

    #[test]
    fn mock_frame_is_deterministic() {
        let dims = Dims::new(48, 32);
        let a = mock_frame(dims, 0.25, 7);
        let b = mock_frame(dims, 0.25, 7);
        assert_eq!(a.checksum(), b.checksum());
        assert_eq!(a.data, b.data);
        // Whole turns of the phase are equivalent.
        assert_eq!(
            mock_frame(dims, 0.0, 3).checksum(),
            mock_frame(dims, 1.0, 3).checksum()
        );
        // A different frame index changes the encoded counter.
        assert_ne!(
            mock_frame(dims, 0.25, 7).checksum(),
            mock_frame(dims, 0.25, 8).checksum()
        );
        // Shape invariants.
        assert_eq!(a.data.len(), dims.area() as usize * 4);
        assert!(a.data.as_chunks::<4>().0.iter().all(|px| px[3] == 255));
    }

    #[test]
    fn mock_frame_encodes_index_in_bottom_row() {
        let dims = Dims::new(64, 8);
        let frame = mock_frame(dims, 0.0, 0b101);
        let y = dims.height - 1;
        assert_eq!(&frame.pixel(0, y)[..3], &[255, 255, 255], "bit 0 set");
        assert_eq!(&frame.pixel(1, y)[..3], &[0, 0, 0], "bit 1 clear");
        assert_eq!(&frame.pixel(2, y)[..3], &[255, 255, 255], "bit 2 set");
        assert_eq!(frame.pixel(3, y)[3], 255, "counter pixels are opaque");
    }

    #[test]
    fn mock_frame_handles_empty_dims() {
        let frame = mock_frame(Dims::new(0, 0), 0.5, 0);
        assert!(frame.data.is_empty());
        assert_eq!(frame.fmt, PixelFormat::Rgba8);
    }

    #[test]
    fn mock_source_paces_to_fps() {
        let mut source = mock_source(Dims::new(16, 16), 100.0).with_limit(3);
        let start = Instant::now();
        while source.next_frame().unwrap().is_some() {}
        let elapsed = start.elapsed();
        // Frames 1 and 2 are each delayed by one 10 ms period.
        assert!(
            elapsed >= Duration::from_millis(15),
            "too fast: {elapsed:?}"
        );
        assert!(elapsed < Duration::from_secs(2), "too slow: {elapsed:?}");
    }

    #[test]
    fn mock_source_without_fps_is_unthrottled() {
        let mut source = mock_source(Dims::new(8, 8), 0.0).with_limit(8);
        let start = Instant::now();
        while source.next_frame().unwrap().is_some() {}
        assert!(start.elapsed() < Duration::from_millis(500));
    }

    #[test]
    fn mock_sources_are_reproducible() {
        let mut a = mock_source(Dims::new(24, 24), 0.0).with_limit(5);
        let mut b = mock_source(Dims::new(24, 24), 0.0).with_limit(5);
        for expected_index in 0..5 {
            let fa = a.next_frame().unwrap().unwrap();
            let fb = b.next_frame().unwrap().unwrap();
            assert_eq!(fa.checksum(), fb.checksum());
            assert_eq!(fa.index, expected_index);
            assert_eq!(fb.index, expected_index);
            assert_eq!(fa.dims, Dims::new(24, 24));
        }
        assert!(a.next_frame().unwrap().is_none());
    }

    #[test]
    fn mock_source_reports_shape() {
        let source = mock_source(Dims::new(40, 30), 30.0);
        assert_eq!(source.dims(), Dims::new(40, 30));
        assert_eq!(source.fps(), 30.0);
        assert_eq!(source.backend_name(), "mock");
    }

    #[test]
    fn monitor_info_formats() {
        let monitor = MonitorInfo {
            index: 1,
            name: "DISPLAY2".to_string(),
            dims: Dims::new(2560, 1440),
            refresh_hz: 143.9,
            primary: true,
            left: 1920,
            top: 0,
        };
        assert_eq!(
            monitor.to_string(),
            "[1] DISPLAY2 2560x1440 @143.9Hz (primary) at (1920, 0)"
        );
        let secondary = MonitorInfo {
            index: 2,
            name: String::new(),
            dims: Dims::new(800, 600),
            refresh_hz: 60.0,
            primary: false,
            left: -800,
            top: -10,
        };
        assert_eq!(secondary.to_string(), "[2]  800x600 @60.0Hz at (-800, -10)");
    }

    #[test]
    fn backend_availability_matches_platform() {
        assert!(backend_available(CaptureBackendKind::Mock));
        assert_eq!(backend_available(CaptureBackendKind::Dxgi), cfg!(windows));
        assert_eq!(
            backend_available(CaptureBackendKind::Auto),
            cfg!(windows) || cfg!(target_os = "android")
        );
    }

    #[test]
    fn open_mock_honours_region_size() {
        let config = CaptureConfig {
            backend: CaptureBackendKind::Mock,
            region: Some(fluid_core::RectConfig {
                x: 10,
                y: 20,
                w: 320,
                h: 180,
            }),
            max_fps: 0.0,
            ..CaptureConfig::default()
        };
        let mut source = open(&config).unwrap();
        assert_eq!(source.dims(), Dims::new(320, 180));
        assert_eq!(source.backend_name(), "mock");
        let frame = source.next_frame().unwrap().unwrap();
        assert_eq!(frame.dims, Dims::new(320, 180));
    }

    #[test]
    fn open_mock_defaults_to_1080p() {
        let config = CaptureConfig {
            backend: CaptureBackendKind::Mock,
            ..CaptureConfig::default()
        };
        let source = open(&config).unwrap();
        assert_eq!(source.dims(), Dims::new(1920, 1080));
    }

    #[test]
    fn open_rejects_invalid_config() {
        let config = CaptureConfig {
            backend: CaptureBackendKind::Mock,
            timeout_ms: 1000,
            region: Some(fluid_core::RectConfig {
                x: 0,
                y: 0,
                w: 0,
                h: 10,
            }),
            ..CaptureConfig::default()
        };
        assert!(open(&config).is_err());
    }

    #[test]
    fn open_native_tolerates_headless_machines() {
        let config = CaptureConfig::default();
        match open(&config) {
            Ok(mut source) => {
                let dims = source.dims();
                assert!(dims.width > 0 && dims.height > 0, "dims {dims}");
                assert!(!source.backend_name().is_empty());
                // A single pull must not end a live stream.
                match source.next_frame() {
                    Ok(Some(frame)) => assert_eq!(frame.dims, dims),
                    Ok(None) => panic!("live capture source returned Ok(None)"),
                    Err(_) => {}
                }
                source.close();
            }
            Err(err) => {
                // Headless CI, remote sessions and unsupported platforms may
                // legitimately fail to open a desktop duplication client.
                let text = err.to_string();
                assert!(!text.is_empty());
            }
        }
    }

    #[test]
    fn enumerate_monitors_tolerates_headless_machines() {
        match enumerate_monitors() {
            Ok(monitors) => {
                for (position, monitor) in monitors.iter().enumerate() {
                    assert_eq!(monitor.index, position);
                    assert!(monitor.dims.width > 0 && monitor.dims.height > 0);
                    assert!(monitor.refresh_hz > 0.0);
                    assert!(monitor.to_string().contains('x'));
                }
            }
            Err(err) => assert!(!err.to_string().is_empty()),
        }
    }

    #[test]
    fn crop_to_region_variants() {
        let full = mock_frame(Dims::new(64, 48), 0.0, 0);
        // No region returns the frame unchanged.
        let same = crop_to_region(&full, None).unwrap();
        assert_eq!(same.checksum(), full.checksum());
        // An empty region is treated as "no region".
        let empty = crop_to_region(&full, Some(Rect::new(0, 0, 0, 0))).unwrap();
        assert_eq!(empty.checksum(), full.checksum());
        // An oversized region is clipped to the frame.
        let clipped = crop_to_region(&full, Some(Rect::new(32, 24, 100, 100))).unwrap();
        assert_eq!(clipped.dims, Dims::new(32, 24));
        // A region outside the frame is an error, matching Frame::crop.
        assert!(crop_to_region(&full, Some(Rect::new(1000, 1000, 8, 8))).is_err());
    }
}
