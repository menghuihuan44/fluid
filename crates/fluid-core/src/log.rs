//! Tiny, dependency-free logger so Fluid stays lightweight.
//!
//! Levels are filtered by a global atomic, so disabled statements cost one
//! relaxed load. Output is written to stderr with a UTC timestamp.

use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Log severity.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum Level {
    Off = 0,
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
    Trace = 5,
}

impl Level {
    pub fn parse(s: &str) -> Level {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" => Level::Off,
            "error" => Level::Error,
            "warn" | "warning" => Level::Warn,
            "info" => Level::Info,
            "debug" => Level::Debug,
            "trace" => Level::Trace,
            _ => Level::Info,
        }
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            Level::Off => "off",
            Level::Error => "error",
            Level::Warn => "warn",
            Level::Info => "info",
            Level::Debug => "debug",
            Level::Trace => "trace",
        }
    }
}

static LEVEL: AtomicU8 = AtomicU8::new(Level::Info as u8);

/// Set the global log level.
pub fn set_level(level: Level) {
    LEVEL.store(level as u8, Ordering::Relaxed);
}

/// Current global log level.
pub fn level() -> Level {
    match LEVEL.load(Ordering::Relaxed) {
        0 => Level::Off,
        1 => Level::Error,
        2 => Level::Warn,
        3 => Level::Info,
        4 => Level::Debug,
        _ => Level::Trace,
    }
}

/// True when a message at parm level would be emitted.
#[inline]
pub fn enabled(level: Level) -> bool {
    (level as u8) <= LEVEL.load(Ordering::Relaxed)
}

/// Emit one log line. Prefer the fluid_* macros.
pub fn log_event(level: Level, args: std::fmt::Arguments<'_>) {
    if !enabled(level) {
        return;
    }
    eprintln!("[{} {}] {}", timestamp_utc(), level.as_str(), args);
}

fn timestamp_utc() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let ms = now.subsec_millis();
    let tod = secs % 86_400;
    let (h, m, s) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

/// Monotonic stopwatch, nanosecond resolution.
#[derive(Clone, Copy, Debug)]
pub struct Stopwatch {
    start: Instant,
}

impl Stopwatch {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed_ns() as f64 / 1_000_000.0
    }

    pub fn restart(&mut self) -> u64 {
        let now = Instant::now();
        let elapsed = now.duration_since(self.start).as_nanos() as u64;
        self.start = now;
        elapsed
    }
}

impl Default for Stopwatch {
    fn default() -> Self {
        Self::start()
    }
}

/// Log at error level.
#[macro_export]
macro_rules! fluid_error {
    ($($arg:tt)*) => {
        $crate::log::log_event($crate::log::Level::Error, format_args!($($arg)*))
    };
}

/// Log at warn level.
#[macro_export]
macro_rules! fluid_warn {
    ($($arg:tt)*) => {
        $crate::log::log_event($crate::log::Level::Warn, format_args!($($arg)*))
    };
}

/// Log at info level.
#[macro_export]
macro_rules! fluid_info {
    ($($arg:tt)*) => {
        $crate::log::log_event($crate::log::Level::Info, format_args!($($arg)*))
    };
}

/// Log at debug level.
#[macro_export]
macro_rules! fluid_debug {
    ($($arg:tt)*) => {
        $crate::log::log_event($crate::log::Level::Debug, format_args!($($arg)*))
    };
}

/// Log at trace level.
#[macro_export]
macro_rules! fluid_trace {
    ($($arg:tt)*) => {
        $crate::log::log_event($crate::log::Level::Trace, format_args!($($arg)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_roundtrip() {
        assert_eq!(Level::parse("DEBUG"), Level::Debug);
        assert_eq!(Level::parse("nonsense"), Level::Info);
        let prev = level();
        set_level(Level::Trace);
        assert!(enabled(Level::Trace));
        set_level(Level::Error);
        assert!(!enabled(Level::Info));
        set_level(prev);
    }

    #[test]
    fn stopwatch_advances() {
        let sw = Stopwatch::start();
        std::thread::sleep(std::time::Duration::from_millis(2));
        assert!(sw.elapsed_ns() > 1_000_000);
    }
}
