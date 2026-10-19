use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Remembered target window (title + class, used to re-locate it next launch).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowRef {
    pub title: String,
    pub class: String,
}

/// Presentation mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PresentMode {
    /// SyncInterval=0 with manual pacing; lowest latency.
    #[default]
    LowLatency,
    /// SyncInterval=1; most stable.
    Vsync,
}

/// Spatial scaling algorithm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleMode {
    /// Nearest-neighbour reconstruction (useful for crisp pixel art).
    Passthrough,
    /// Bilinear reconstruction; lower cost on weak integrated GPUs.
    Bilinear,
    /// Adaptive Catmull-Rom reconstruction + contrast-adaptive sharpening.
    #[default]
    Adaptive,
}

/// Output canvas.
/// What area fluid should capture.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CaptureMode {
    /// Capture a whole monitor with Desktop Duplication.
    Monitor,
    /// Track the most recent non-fluid foreground window with WGC. This is the
    /// recommended mode: no window list is needed and the overlay cannot feed
    /// captured frames back into the capture source.
    #[default]
    ForegroundWindow,
    /// Use a specific window from the enhanced window list.
    Window,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputMode {
    /// Overlay matches the target window size.
    #[default]
    MatchWindow,
    /// Stretch the target window to fill its monitor (external upscaling).
    Monitor,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TargetConfig {
    pub window: Option<WindowRef>,
    /// Selected monitor device name (for example "\\\\.\\DISPLAY1").
    pub monitor: Option<String>,
    pub mode: CaptureMode,
    /// Start the pipeline automatically on launch when the target exists.
    pub auto_start: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DisplayConfig {
    pub present_mode: PresentMode,
    pub show_hud: bool,
    /// Keep running in the tray when the settings window is closed.
    pub close_to_tray: bool,
    /// Retained for configuration compatibility. Capture now always minimizes
    /// the settings window, because its topmost overlay would otherwise hide the
    /// settings UI.
    pub hide_on_start: bool,
    /// Enable Ctrl+Alt+Shift+F9 (start/stop) and Ctrl+Alt+Shift+F10 (HUD toggle).
    pub global_hotkeys: bool,
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            present_mode: PresentMode::LowLatency,
            show_hud: true,
            close_to_tray: false,
            hide_on_start: true,
            global_hotkeys: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ScaleConfig {
    pub mode: ScaleMode,
    pub output: OutputMode,
    /// 0..=100 contrast-adaptive sharpening strength.
    pub sharpness: u8,
    /// Preserve the target aspect ratio and letterbox when filling a monitor.
    pub preserve_aspect: bool,
}

impl Default for ScaleConfig {
    fn default() -> Self {
        Self {
            mode: ScaleMode::Adaptive,
            output: OutputMode::MatchWindow,
            sharpness: 30,
            preserve_aspect: true,
        }
    }
}

/// Frame generation configuration. Disabled by default; the motion field is
/// only computed when the user explicitly enables interpolation.
/// Anchor position of the real-time monitor HUD.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum HudAnchor {
    #[default]
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    /// Normalized position inside the overlay; 0..100 on both axes.
    Custom,
}

/// Real-time monitor HUD layout and visible sections.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct HudConfig {
    pub anchor: HudAnchor,
    /// Pixel margin from the selected edge.
    pub margin: u8,
    /// 0..=100 normalized X position for Custom anchor.
    pub custom_x: u8,
    /// 0..=100 normalized Y position for Custom anchor.
    pub custom_y: u8,
    /// Bitmap font scale, 1..=3.
    pub font_scale: u8,
    /// Legacy per-pixel background alpha, retained for config compatibility.
    pub background_alpha: u8,
    /// Opaque HUD panel color (sRGB).
    pub background_color: [u8; 3],
    /// Whole HUD window opacity, 10..=100 percent.
    pub opacity: u8,
    pub show_source: bool,
    pub show_performance: bool,
    pub show_quality: bool,
    pub show_framegen: bool,
    pub show_system: bool,
}

impl Default for HudConfig {
    fn default() -> Self {
        Self {
            anchor: HudAnchor::TopLeft,
            margin: 12,
            custom_x: 50,
            custom_y: 50,
            font_scale: 2,
            background_alpha: 160,
            background_color: [10, 14, 22],
            opacity: 90,
            show_source: true,
            show_performance: true,
            show_quality: true,
            show_framegen: true,
            show_system: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct FgConfig {
    pub enabled: bool,
    pub motion_compensation: bool,
    /// 0..=100 interpolation confidence gate. Higher values keep the
    /// interpolated result across more of the confidence range; low-confidence,
    /// occluded and scene-cut pixels always fall back to the real current frame.
    pub strength: u8,
    /// 0..=100 scene-cut/occlusion protection. Higher values fall back more
    /// readily to the real current frame.
    pub scene_protection: u8,
    /// Requested output/source rate multiple. It is capped by display refresh
    /// and by available source frames; the renderer never invents extra source
    /// content.
    pub max_multiplier: u8,
}

impl Default for FgConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            motion_compensation: true,
            strength: 85,
            scene_protection: 60,
            max_multiplier: 2,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub target: TargetConfig,
    pub display: DisplayConfig,
    pub scale: ScaleConfig,
    pub fg: FgConfig,
    pub hud: HudConfig,
}

impl AppConfig {
    pub fn config_path() -> Option<PathBuf> {
        directories::ProjectDirs::from("com", "fluid-project", "fluid")
            .map(|d| d.config_dir().join("config.toml"))
    }

    pub fn load() -> (AppConfig, Option<PathBuf>) {
        let Some(path) = Self::config_path() else {
            return (AppConfig::default(), None);
        };
        match std::fs::read_to_string(&path) {
            Ok(text) => match toml::from_str(&text) {
                Ok(cfg) => (cfg, Some(path)),
                Err(e) => {
                    tracing::warn!("config parse failed, using defaults: {e}");
                    (AppConfig::default(), Some(path))
                }
            },
            Err(_) => (AppConfig::default(), Some(path)),
        }
    }

    pub fn save(&self) -> Result<(), String> {
        let Some(path) = Self::config_path() else {
            return Err("cannot locate config directory".into());
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let text = toml::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&path, text).map_err(|e| e.to_string())?;
        tracing::info!("config saved: {}", path.display());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_config_is_migrated_with_defaults() {
        let text = r#"
[target]
[display]
present_mode = "LowLatency"
show_hud = true
[scale]
mode = "Passthrough"
[fg]
enabled = false
"#;
        let cfg: AppConfig = toml::from_str(text).expect("old config must parse");
        assert_eq!(cfg.scale.mode, ScaleMode::Passthrough);
        assert_eq!(cfg.scale.output, OutputMode::MatchWindow);
        assert_eq!(cfg.scale.sharpness, 30);
        assert!(cfg.scale.preserve_aspect);
        assert!(cfg.fg.motion_compensation);
        assert_eq!(cfg.fg.strength, 85);
        assert_eq!(cfg.fg.scene_protection, 60);
        assert_eq!(cfg.fg.max_multiplier, 2);
    }

    #[test]
    fn auto_start_is_preserved() {
        let text = r#"
[target]
auto_start = true
[target.window]
title = "ZCode"
class = "Chrome_WidgetWin_1"
"#;
        let cfg: AppConfig = toml::from_str(text).expect("parse auto_start");
        assert!(cfg.target.auto_start);
        let encoded = toml::to_string_pretty(&cfg).expect("encode auto_start");
        println!("{encoded}");
        let decoded: AppConfig = toml::from_str(&encoded).expect("decode auto_start");
        assert!(decoded.target.auto_start);
    }

    #[test]
    fn config_roundtrip() {
        let cfg = AppConfig {
            scale: ScaleConfig {
                mode: ScaleMode::Adaptive,
                output: OutputMode::Monitor,
                sharpness: 65,
                preserve_aspect: true,
            },
            fg: FgConfig {
                enabled: true,
                motion_compensation: false,
                strength: 70,
                scene_protection: 40,
                max_multiplier: 3,
            },
            ..AppConfig::default()
        };
        let text = toml::to_string_pretty(&cfg).expect("serialize");
        let decoded: AppConfig = toml::from_str(&text).expect("deserialize");
        assert_eq!(cfg, decoded);
    }
}
