pub mod config;
pub mod stats;
pub mod time;

pub use config::{
    AppConfig, CaptureMode, FgConfig, HudAnchor, HudConfig, OutputMode, PresentMode, ScaleConfig,
    ScaleMode, WindowRef,
};
pub use stats::{RunState, SharedState, Stats};
