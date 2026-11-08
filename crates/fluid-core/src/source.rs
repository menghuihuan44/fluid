//! Platform abstraction traits shared by every frame producer and consumer.
//!
//! `FrameSource` and `FrameSink` are the frozen contract between the capture,
//! video, present and engine crates. Anything that can deliver frames (DXGI
//! desktop duplication, a y4m file, a synthetic pattern) implements
//! `FrameSource`; anything that can display or store frames (a window, a y4m
//! writer, a PNG directory, the null sink) implements `FrameSink`.

use crate::error::Result;
use crate::frame::{Dims, Frame};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Current wall clock time in nanoseconds since the Unix epoch.
pub fn now_ns() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_nanos() as i64,
        Err(_) => 0,
    }
}

/// Milliseconds helper for pacing.
pub fn now_ms() -> f64 {
    now_ns() as f64 / 1_000_000.0
}

/// Block until `deadline`, sleeping in short slices so shutdown stays responsive.
pub fn sleep_until(deadline: Instant) {
    loop {
        let now = Instant::now();
        if now >= deadline {
            return;
        }
        let left = deadline - now;
        if left > Duration::from_millis(1) {
            std::thread::sleep(left - Duration::from_millis(1));
        } else {
            std::thread::sleep(Duration::from_micros(200));
        }
    }
}

/// Source of frames: capture backends, video decoders, generators.
pub trait FrameSource: Send {
    /// Size of frames produced by this source.
    fn dims(&self) -> Dims;
    /// Nominal source frame rate in Hz (0 when unknown).
    fn fps(&self) -> f32 {
        0.0
    }
    /// Human readable backend name for logs and telemetry.
    fn backend_name(&self) -> &'static str;
    /// Blocking pull. `Ok(None)` means the stream ended (files, finite mocks).
    fn next_frame(&mut self) -> Result<Option<Frame>>;
    /// Release platform resources.
    fn close(&mut self) {}
}

/// Consumer of frames: present backends, file writers, analysis sinks.
pub trait FrameSink: Send {
    /// Size of frames this sink accepts (output resolution).
    fn dims(&self) -> Dims;
    /// Human readable backend name for logs and telemetry.
    fn backend_name(&self) -> &'static str;
    /// Display or store one frame. Implementations must not retain the frame.
    fn present(&mut self, frame: &Frame) -> Result<()>;
    /// True once the user asked the sink to close (window close / Escape).
    fn should_close(&self) -> bool {
        false
    }
    /// Release platform resources.
    fn close(&mut self) {}
}
