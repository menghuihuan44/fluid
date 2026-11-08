//! Process-wide DPI awareness (Windows).
//!
//! Windows virtualises screen coordinates for processes that never declare DPI
//! awareness: a 2880x1800 panel at 200% scaling reports 1440x900, window sizes
//! are in scaled units and captured pixels disagree with reported monitors.
//! Declaring per-monitor-v2 awareness up front keeps every number in this
//! program in physical pixels, so devices/run/bench/video all agree on the same
//! resolution and a --region the user copies actually captures that many pixels.
//!
//! The present backend also declares awareness when it creates a window, but
//! enumerating devices or benchmarking never opens one, and by then the process
//! mode is already fixed for good - hence this early call.

/// Declare per-monitor DPI awareness for the current process.
///
/// Best effort: on non-Windows builds and on Windows versions that reject the
/// call this is a no-op, and the program keeps the previous behaviour.
pub fn enable() {
    #[cfg(windows)]
    {
        use windows::Win32::UI::HiDpi::{
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext,
        };
        // SAFETY: the call only changes process-wide DPI policy and is
        // documented as safe to invoke at any point before the first window;
        // the result is deliberately ignored because older Windows releases
        // return an error for the v2 context and we then stay DPI unaware.
        unsafe {
            let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        }
    }
}
