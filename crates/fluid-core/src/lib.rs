//! Fluid core: shared types, configuration, logging and metrics.
//!
//! Fluid is a low-latency, cross-platform frame generation (interpolation) and
//! upscaling suite written in Rust. It works entirely as an external
//! capture -> process -> present pipeline: it never touches game memory, never
//! injects into a process and never modifies the captured source.

#![forbid(unsafe_code)]

pub mod config;
pub mod error;
pub mod frame;
pub mod log;
pub mod math;
pub mod source;
pub mod stats;

pub use config::*;
pub use error::{FluidError, Result};
pub use frame::{Dims, Frame, LumaPlane, PixelFormat, Rect};
pub use log::{Level, Stopwatch};
pub use source::{FrameSink, FrameSource, now_ms, now_ns, sleep_until};
pub use stats::{FpsMeter, FrameStats, LatencyStats};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Human readable build/version banner.
pub fn version_string() -> String {
    format!(
        "fluid {} ({} {}, {})",
        VERSION,
        std::env::consts::OS,
        std::env::consts::ARCH,
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_reported() {
        let v = super::version_string();
        assert!(v.starts_with("fluid "));
    }
}
