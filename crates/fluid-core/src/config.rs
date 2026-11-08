//! User facing configuration: TOML serialisable and fully defaulted.
//!
//! Every struct uses serde defaults, so a configuration file may contain only
//! the keys a user wants to override and stays forward compatible when new
//! options are added.

use std::fmt;
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{FluidError, Result};
use crate::frame::{Dims, Rect};

/// One of the five tuning presets. Presets only touch quality knobs, never
/// capture/present settings, so switching them at runtime is always safe.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QualityPreset {
    UltraPerformance,
    Performance,
    #[default]
    Balanced,
    Quality,
    UltraQuality,
}

impl QualityPreset {
    pub const ALL: [QualityPreset; 5] = [
        QualityPreset::UltraPerformance,
        QualityPreset::Performance,
        QualityPreset::Balanced,
        QualityPreset::Quality,
        QualityPreset::UltraQuality,
    ];

    pub const fn as_str(&self) -> &'static str {
        match self {
            QualityPreset::UltraPerformance => "ultra-performance",
            QualityPreset::Performance => "performance",
            QualityPreset::Balanced => "balanced",
            QualityPreset::Quality => "quality",
            QualityPreset::UltraQuality => "ultra-quality",
        }
    }

    /// Motion estimation effort: (pyramid levels, block size, search radius,
    /// refinement iterations).
    pub const fn flow_tuning(&self) -> (u32, u32, u32, u32) {
        match self {
            QualityPreset::UltraPerformance => (2, 16, 4, 1),
            QualityPreset::Performance => (2, 8, 6, 1),
            QualityPreset::Balanced => (3, 8, 8, 2),
            QualityPreset::Quality => (4, 8, 10, 3),
            QualityPreset::UltraQuality => (4, 8, 12, 4),
        }
    }

    /// Blend weight of a synthesised frame.
    pub const fn interp_blend(&self) -> f32 {
        match self {
            QualityPreset::UltraPerformance => 0.85,
            QualityPreset::Performance => 0.92,
            QualityPreset::Balanced => 1.0,
            QualityPreset::Quality => 1.0,
            QualityPreset::UltraQuality => 1.0,
        }
    }
}

impl fmt::Display for QualityPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for QualityPreset {
    type Err = FluidError;

    fn from_str(s: &str) -> Result<Self> {
        let norm = s.trim().to_ascii_lowercase().replace('_', "-");
        let mut names = String::new();
        for p in QualityPreset::ALL {
            if !names.is_empty() {
                names.push_str(", ");
            }
            names.push_str(p.as_str());
        }
        for p in QualityPreset::ALL {
            if p.as_str() == norm {
                return Ok(p);
            }
        }
        Err(FluidError::config(format!(
            "unknown quality preset {s:?} (expected one of: {names})"
        )))
    }
}
/// Spatial upscaling filter selected by the user.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UpscaleFilter {
    /// Do not upscale (spatial pass disabled).
    #[default]
    Off,
    Nearest,
    Bilinear,
    Bicubic,
    /// Windowed Lanczos-3, the highest quality classic filter.
    Lanczos,
    /// Adaptive edge-directed filter (Fluid EASU-style), best sharpness/cost.
    Adaptive,
    /// Adaptive filter plus temporal accumulation from previous frames.
    Temporal,
}

impl UpscaleFilter {
    pub fn parse(s: &str) -> Result<Self> {
        let v = match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" => UpscaleFilter::Off,
            "nearest" => UpscaleFilter::Nearest,
            "bilinear" | "linear" => UpscaleFilter::Bilinear,
            "bicubic" | "cubic" => UpscaleFilter::Bicubic,
            "lanczos" | "lanczos3" => UpscaleFilter::Lanczos,
            "adaptive" | "easu" => UpscaleFilter::Adaptive,
            "temporal" | "taa" => UpscaleFilter::Temporal,
            other => {
                return Err(FluidError::config(format!(
                    "unknown upscale filter {other:?}"
                )));
            }
        };
        Ok(v)
    }

    pub const fn needs_temporal_history(&self) -> bool {
        matches!(self, UpscaleFilter::Temporal)
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            UpscaleFilter::Off => "off",
            UpscaleFilter::Nearest => "nearest",
            UpscaleFilter::Bilinear => "bilinear",
            UpscaleFilter::Bicubic => "bicubic",
            UpscaleFilter::Lanczos => "lanczos",
            UpscaleFilter::Adaptive => "adaptive",
            UpscaleFilter::Temporal => "temporal",
        }
    }
}

impl fmt::Display for UpscaleFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for UpscaleFilter {
    type Err = FluidError;

    fn from_str(s: &str) -> Result<Self> {
        UpscaleFilter::parse(s)
    }
}

/// How the output resolution is derived from the source resolution.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScaleMode {
    /// Keep the source resolution.
    Native,
    /// Multiply by scale.factor.
    #[default]
    Factor,
    /// Use an explicit target resolution.
    Target,
}

/// Resolution selection.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ScaleConfig {
    pub mode: ScaleMode,
    pub factor: f32,
    pub target_w: u32,
    pub target_h: u32,
}

impl Default for ScaleConfig {
    fn default() -> Self {
        Self {
            mode: ScaleMode::Factor,
            factor: 1.0,
            target_w: 0,
            target_h: 0,
        }
    }
}

impl ScaleConfig {
    pub fn resolve(&self, src: Dims) -> Dims {
        match self.mode {
            ScaleMode::Native => src,
            ScaleMode::Factor => src.scaled(self.factor),
            ScaleMode::Target => {
                if self.target_w == 0 || self.target_h == 0 {
                    src
                } else {
                    Dims::new(self.target_w, self.target_h)
                }
            }
        }
    }

    /// Output dims may never be zero.
    pub fn validate(&self) -> Result<()> {
        match self.mode {
            ScaleMode::Factor => {
                if !(0.25..=8.0).contains(&self.factor) || !self.factor.is_finite() {
                    return Err(FluidError::config(format!(
                        "upscaler.scale.factor must be within 0.25..=8.0, got {}",
                        self.factor
                    )));
                }
            }
            ScaleMode::Target => {
                if self.target_w == 0 || self.target_h == 0 {
                    return Err(FluidError::config(
                        "upscaler.scale.mode = target requires target_w and target_h",
                    ));
                }
            }
            ScaleMode::Native => {}
        }
        Ok(())
    }
}

/// Temporal accumulation settings (used by the temporal upscaler).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TemporalConfig {
    pub enabled: bool,
    pub blend: f32,
    pub history_frames: u8,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            blend: 0.12,
            history_frames: 2,
        }
    }
}

/// Which implementation runs the upscaling stage.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UpscaleBackendKind {
    /// Use the GPU backend when a device is available, otherwise the CPU path.
    #[default]
    Auto,
    /// Portable CPU path (SIMD + worker threads), works everywhere.
    Cpu,
    /// GPU compute path (wgpu: Vulkan on Windows/Android, GLES fallback).
    Gpu,
}

impl UpscaleBackendKind {
    pub const ALL: [UpscaleBackendKind; 3] = [Self::Auto, Self::Cpu, Self::Gpu];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }

    /// Parse a backend name, accepting the usual aliases.
    pub fn parse(text: &str) -> Result<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "auto" | "default" => Ok(Self::Auto),
            "cpu" | "simd" => Ok(Self::Cpu),
            "gpu" | "d3d11" | "compute" => Ok(Self::Gpu),
            other => Err(FluidError::config(format!(
                "unknown upscaler backend {other:?} (expected auto, cpu or gpu)"
            ))),
        }
    }
}

impl fmt::Display for UpscaleBackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for UpscaleBackendKind {
    type Err = FluidError;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

/// Upscaling / sharpening configuration.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct UpscalerConfig {
    pub filter: UpscaleFilter,
    pub scale: ScaleConfig,
    /// Contrast adaptive sharpening amount, 0.0 disables sharpening.
    pub sharpness: f32,
    pub temporal: TemporalConfig,
    /// Which implementation runs the upscaling stage.
    pub backend: UpscaleBackendKind,
}

impl UpscalerConfig {
    pub fn validate(&self) -> Result<()> {
        self.scale.validate()?;
        if !(0.0..=1.0).contains(&self.sharpness) {
            return Err(FluidError::config(format!(
                "upscaler.sharpness must be within 0.0..=1.0, got {}",
                self.sharpness
            )));
        }
        if self.temporal.enabled && !(0.0..=1.0).contains(&self.temporal.blend) {
            return Err(FluidError::config(
                "upscaler.temporal.blend must be within 0.0..=1.0",
            ));
        }
        Ok(())
    }
}
/// Which implementation runs the frame generation (optical flow + synthesis) stage.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InterpBackendKind {
    /// Use the GPU backend when a device is available, otherwise the CPU path.
    #[default]
    Auto,
    /// Portable CPU path (SIMD + worker threads), works everywhere.
    Cpu,
    /// GPU compute path (wgpu); fails if unavailable.
    Gpu,
}

impl InterpBackendKind {
    pub const ALL: [InterpBackendKind; 3] = [Self::Auto, Self::Cpu, Self::Gpu];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }

    /// Parse a backend name, accepting the usual aliases.
    pub fn parse(text: &str) -> Result<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "auto" | "default" => Ok(Self::Auto),
            "cpu" | "simd" => Ok(Self::Cpu),
            "gpu" | "wgpu" | "compute" => Ok(Self::Gpu),
            other => Err(FluidError::config(format!(
                "unknown interpolation backend {other:?} (expected auto, cpu or gpu)"
            ))),
        }
    }
}

impl fmt::Display for InterpBackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for InterpBackendKind {
    type Err = FluidError;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

/// Interpolation mode.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InterpMode {
    /// Disabled: only upscaling runs.
    #[default]
    Off,
    /// Generate multiplier - 1 frames per real frame at even spacing.
    Fixed,
    /// Generate frames only when the interval exceeds the refresh budget.
    Adaptive,
}

/// Optical flow estimation tuning.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct FlowConfig {
    pub pyramid_levels: u32,
    pub block_size: u32,
    pub search_radius: u32,
    pub refine_iters: u32,
    /// Vectors below this confidence are treated as unreliable.
    pub confidence_min: f32,
}

impl Default for FlowConfig {
    fn default() -> Self {
        let (py, bs, sr, it) = QualityPreset::Balanced.flow_tuning();
        Self {
            pyramid_levels: py,
            block_size: bs,
            search_radius: sr,
            refine_iters: it,
            confidence_min: 0.2,
        }
    }
}

impl FlowConfig {
    pub fn validate(&self) -> Result<()> {
        if !(1..=5).contains(&self.pyramid_levels) {
            return Err(FluidError::config(
                "interpolator.flow.pyramid_levels must be 1..=5",
            ));
        }
        if !(4..=32).contains(&self.block_size) || !self.block_size.is_power_of_two() {
            return Err(FluidError::config(
                "interpolator.flow.block_size must be a power of two within 4..=32",
            ));
        }
        if !(1..=32).contains(&self.search_radius) {
            return Err(FluidError::config(
                "interpolator.flow.search_radius must be 1..=32",
            ));
        }
        if self.refine_iters > 8 {
            return Err(FluidError::config(
                "interpolator.flow.refine_iters must be <= 8",
            ));
        }
        if !(0.0..=1.0).contains(&self.confidence_min) {
            return Err(FluidError::config(
                "interpolator.flow.confidence_min must be within 0.0..=1.0",
            ));
        }
        Ok(())
    }
}

/// Frame generation configuration.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct InterpolatorConfig {
    pub mode: InterpMode,
    /// Total frames per source frame in fixed mode (2 = double, 3 = triple).
    pub multiplier: u8,
    pub quality: QualityPreset,
    pub occlusion_fill: bool,
    pub blend: f32,
    /// Mean absolute luma delta above which a frame pair counts as a scene cut.
    pub scene_cut_threshold: f32,
    pub flow: FlowConfig,
    /// Which implementation runs the interpolation stage.
    pub backend: InterpBackendKind,
}

impl Default for InterpolatorConfig {
    fn default() -> Self {
        Self {
            mode: InterpMode::Off,
            multiplier: 2,
            quality: QualityPreset::Balanced,
            occlusion_fill: true,
            blend: QualityPreset::Balanced.interp_blend(),
            scene_cut_threshold: 0.18,
            flow: FlowConfig::default(),
            backend: InterpBackendKind::Auto,
        }
    }
}

impl InterpolatorConfig {
    pub fn validate(&self) -> Result<()> {
        if !(1..=8).contains(&self.multiplier) {
            return Err(FluidError::config("interpolator.multiplier must be 1..=8"));
        }
        if !(0.0..=1.0).contains(&self.blend) {
            return Err(FluidError::config(
                "interpolator.blend must be within 0.0..=1.0",
            ));
        }
        if !(0.0..=1.0).contains(&self.scene_cut_threshold) {
            return Err(FluidError::config(
                "interpolator.scene_cut_threshold must be within 0.0..=1.0",
            ));
        }
        self.flow.validate()
    }
}
/// Capture backend selection.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CaptureBackendKind {
    /// Pick the best backend for the platform.
    #[default]
    Auto,
    /// Windows Desktop Duplication (DXGI), lowest latency on Windows 8+.
    Dxgi,
    /// Synthetic test pattern, useful for benchmarking and CI.
    Mock,
}

/// Optional capture region (a sub-rectangle of the monitor).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RectConfig {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl From<RectConfig> for Rect {
    fn from(r: RectConfig) -> Self {
        Rect::new(r.x, r.y, r.w, r.h)
    }
}

/// Capture one application window instead of a whole monitor.
///
/// Every field is optional; `id` (a native window handle, HWND on
/// Windows) wins when non-zero, and the remaining fields must all match.
/// Left empty this means "capture the configured monitor".
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct WindowTarget {
    /// Native window id, 0 = not set.
    pub id: u64,
    /// Case-insensitive substring of the window title; empty = not set.
    pub title: String,
    /// Process executable name, with or without the `.exe` suffix.
    pub process: String,
}

impl WindowTarget {
    /// True when at least one selector field is filled in.
    pub fn is_set(&self) -> bool {
        self.id != 0 || !self.title.trim().is_empty() || !self.process.trim().is_empty()
    }

    pub fn validate(&self) -> Result<()> {
        if self.id != 0 && (!self.title.trim().is_empty() || !self.process.trim().is_empty()) {
            return Err(FluidError::config(
                "capture.window: give either an id or a title/process match, not both",
            ));
        }
        Ok(())
    }
}

/// Desktop / screen capture configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CaptureConfig {
    pub backend: CaptureBackendKind,
    /// Monitor index as reported by fluid doctor.
    pub monitor: usize,
    /// Capture only part of the monitor.
    pub region: Option<RectConfig>,
    /// Windows only: composite the mouse cursor into the frame.
    pub include_cursor: bool,
    /// Per-frame grab timeout in milliseconds.
    pub timeout_ms: u32,
    /// Optional capture rate cap in Hz; 0 means as fast as the source updates.
    pub max_fps: f32,
    /// Capture a single application window instead of the monitor.
    pub window: Option<WindowTarget>,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            backend: CaptureBackendKind::Auto,
            monitor: 0,
            region: None,
            include_cursor: true,
            timeout_ms: 8,
            max_fps: 0.0,
            window: None,
        }
    }
}

impl CaptureConfig {
    pub fn validate(&self) -> Result<()> {
        if let Some(r) = self.region
            && (r.w == 0 || r.h == 0)
        {
            return Err(FluidError::config("capture.region must have non-zero size"));
        }
        if self.timeout_ms > 1000 {
            return Err(FluidError::config("capture.timeout_ms must be <= 1000"));
        }
        if self.max_fps < 0.0 {
            return Err(FluidError::config("capture.max_fps must be >= 0"));
        }
        if let Some(window) = &self.window {
            window.validate()?;
        }
        Ok(())
    }
}

/// Presentation backend selection.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PresentBackendKind {
    /// Pick the best backend for the platform.
    #[default]
    Auto,
    /// GDI blit into a normal Win32 window (`win32-window`); works everywhere,
    /// including remote desktop and software-only machines.
    Window,
    /// DXGI flip-model swap chain (`dxgi-swapchain`): lower latency, proper
    /// vsync and tearing support, and it can go exclusive fullscreen.
    Dxgi,
}

/// Presentation configuration for the realtime overlay window.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PresentConfig {
    /// Which presentation path to use.
    pub backend: PresentBackendKind,
    pub vsync: bool,
    pub fullscreen: bool,
    pub borderless: bool,
    pub title: String,
    /// Present the newest frame immediately instead of queuing.
    pub low_latency: bool,
    /// Close the overlay when Escape is pressed.
    pub close_on_escape: bool,
    /// Monitor the overlay opens on.
    pub monitor: usize,
}

impl Default for PresentConfig {
    fn default() -> Self {
        Self {
            backend: PresentBackendKind::Auto,
            vsync: true,
            fullscreen: false,
            borderless: true,
            title: "Fluid".to_string(),
            low_latency: true,
            close_on_escape: true,
            monitor: 0,
        }
    }
}

/// Frame pacing policy.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PacingMode {
    /// Follow the present backend refresh rate.
    #[default]
    Auto,
    Vsync,
    /// No sleep at all, frame rate limited only by processing cost.
    Unlimited,
}
/// Latency budget configuration.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LatencyConfig {
    /// Maximum number of in-flight frames before dropping.
    pub max_queue: usize,
    pub pacing: PacingMode,
    /// Soft cap in Hz, 0 disables the cap.
    pub target_fps: f32,
    /// Extra delay before presenting, in microseconds.
    pub present_delay_us: u32,
}

impl Default for LatencyConfig {
    fn default() -> Self {
        Self {
            max_queue: 2,
            pacing: PacingMode::Auto,
            target_fps: 0.0,
            present_delay_us: 0,
        }
    }
}

/// Video pipeline configuration.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct VideoConfig {
    pub fps_num: u32,
    pub fps_den: u32,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            fps_num: 60,
            fps_den: 1,
        }
    }
}

impl VideoConfig {
    pub fn fps(&self) -> f32 {
        if self.fps_den == 0 {
            0.0
        } else {
            self.fps_num as f32 / self.fps_den as f32
        }
    }
}

/// Logging configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    /// off | error | warn | info | debug | trace
    pub level: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
        }
    }
}

/// Root configuration document.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct FluidConfig {
    /// Free form profile name, informational only.
    pub profile: String,
    pub upscaler: UpscalerConfig,
    pub interpolator: InterpolatorConfig,
    pub capture: CaptureConfig,
    pub present: PresentConfig,
    pub latency: LatencyConfig,
    pub video: VideoConfig,
    pub logging: LoggingConfig,
}

impl Default for FluidConfig {
    fn default() -> Self {
        Self {
            profile: "default".to_string(),
            upscaler: UpscalerConfig {
                filter: UpscaleFilter::Adaptive,
                backend: UpscaleBackendKind::Auto,
                scale: ScaleConfig::default(),
                sharpness: 0.3,
                temporal: TemporalConfig::default(),
            },
            interpolator: InterpolatorConfig::default(),
            capture: CaptureConfig::default(),
            present: PresentConfig::default(),
            latency: LatencyConfig::default(),
            video: VideoConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl FluidConfig {
    /// A configuration tuned for the given preset.
    pub fn with_preset(preset: QualityPreset) -> Self {
        let mut cfg = FluidConfig::default();
        cfg.apply_preset(preset);
        cfg
    }

    /// Apply a quality preset to quality knobs only.
    pub fn apply_preset(&mut self, preset: QualityPreset) {
        self.profile = preset.as_str().to_string();
        self.interpolator.quality = preset;
        let (py, bs, sr, it) = preset.flow_tuning();
        self.interpolator.flow.pyramid_levels = py;
        self.interpolator.flow.block_size = bs;
        self.interpolator.flow.search_radius = sr;
        self.interpolator.flow.refine_iters = it;
        self.interpolator.blend = preset.interp_blend();
        self.upscaler.sharpness = match preset {
            QualityPreset::UltraPerformance => 0.2,
            QualityPreset::Performance => 0.25,
            QualityPreset::Balanced => 0.3,
            QualityPreset::Quality => 0.4,
            QualityPreset::UltraQuality => 0.5,
        };
    }

    pub fn from_toml_str(text: &str) -> Result<Self> {
        let cfg: FluidConfig =
            toml::from_str(text).map_err(|e| FluidError::config(format!("parse error: {e}")))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn to_toml_string(&self) -> Result<String> {
        let text = toml::to_string_pretty(self)
            .map_err(|e| FluidError::config(format!("serialisation error: {e}")))?;
        Ok(tidy_floats(&text))
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| FluidError::config(format!("cannot read {}: {e}", path.display())))?;
        Self::from_toml_str(&text)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, self.to_toml_string()?)
            .map_err(|e| FluidError::config(format!("cannot write {}: {e}", path.display())))?;
        Ok(())
    }

    /// Load a config file if it exists, otherwise return defaults.
    pub fn load_or_default(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.exists() {
            Self::load(path)
        } else {
            Ok(FluidConfig::default())
        }
    }

    /// Apply the configured log level to the global logger.
    pub fn install_logging(&self) {
        crate::log::set_level(crate::log::Level::parse(&self.logging.level));
    }

    pub fn validate(&self) -> Result<()> {
        self.upscaler.validate()?;
        self.interpolator.validate()?;
        self.capture.validate()?;
        if self.latency.max_queue == 0 || self.latency.max_queue > 16 {
            return Err(FluidError::config(
                "latency.max_queue must be within 1..=16",
            ));
        }
        if self.latency.target_fps < 0.0 {
            return Err(FluidError::config("latency.target_fps must be >= 0"));
        }
        if self.video.fps_num == 0 || self.video.fps_den == 0 {
            return Err(FluidError::config(
                "video.fps_num and video.fps_den must be non-zero",
            ));
        }
        Ok(())
    }

    /// Commented template written by fluid config init.
    pub fn template() -> String {
        let mut out = String::new();
        out.push_str("# Fluid configuration\n");
        out.push_str("# Delete any line to inherit its default value.\n");
        out.push_str(
            "# Presets: ultra-performance | performance | balanced | quality | ultra-quality\n",
        );
        out.push_str(
            "# Filters: off | nearest | bilinear | bicubic | lanczos | adaptive | temporal\n\n",
        );
        let body = FluidConfig::default().to_toml_string().unwrap_or_default();
        out.push_str(&body);
        out
    }
}
impl InterpMode {
    pub const ALL: [InterpMode; 3] = [InterpMode::Off, InterpMode::Fixed, InterpMode::Adaptive];

    pub fn as_str(&self) -> &'static str {
        match self {
            InterpMode::Off => "off",
            InterpMode::Fixed => "fixed",
            InterpMode::Adaptive => "adaptive",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "0" => Ok(InterpMode::Off),
            "fixed" | "on" | "1" => Ok(InterpMode::Fixed),
            "adaptive" | "auto" => Ok(InterpMode::Adaptive),
            other => Err(FluidError::config(format!(
                "unknown interpolation mode '{other}' (off | fixed | adaptive)"
            ))),
        }
    }
}

impl fmt::Display for InterpMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for InterpMode {
    type Err = FluidError;
    fn from_str(s: &str) -> Result<Self> {
        InterpMode::parse(s)
    }
}

impl PresentBackendKind {
    pub const ALL: [PresentBackendKind; 3] = [
        PresentBackendKind::Auto,
        PresentBackendKind::Window,
        PresentBackendKind::Dxgi,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            PresentBackendKind::Auto => "auto",
            PresentBackendKind::Window => "window",
            PresentBackendKind::Dxgi => "dxgi",
        }
    }

    pub fn parse(text: &str) -> Result<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "auto" | "default" => Ok(PresentBackendKind::Auto),
            "window" | "gdi" | "win32" => Ok(PresentBackendKind::Window),
            "dxgi" | "swapchain" | "flip" => Ok(PresentBackendKind::Dxgi),
            other => Err(FluidError::config(format!(
                "unknown present backend {other:?} (expected auto, window or dxgi)"
            ))),
        }
    }
}

impl fmt::Display for PresentBackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PresentBackendKind {
    type Err = FluidError;

    fn from_str(s: &str) -> Result<Self> {
        PresentBackendKind::parse(s)
    }
}

impl CaptureBackendKind {
    pub const ALL: [CaptureBackendKind; 3] = [
        CaptureBackendKind::Auto,
        CaptureBackendKind::Dxgi,
        CaptureBackendKind::Mock,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            CaptureBackendKind::Auto => "auto",
            CaptureBackendKind::Dxgi => "dxgi",
            CaptureBackendKind::Mock => "mock",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "default" => Ok(CaptureBackendKind::Auto),
            "dxgi" | "desktop" | "dd" => Ok(CaptureBackendKind::Dxgi),
            "mock" | "test" | "synthetic" => Ok(CaptureBackendKind::Mock),
            other => Err(FluidError::config(format!(
                "unknown capture backend '{other}' (auto | dxgi | mock)"
            ))),
        }
    }
}

impl fmt::Display for CaptureBackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for CaptureBackendKind {
    type Err = FluidError;
    fn from_str(s: &str) -> Result<Self> {
        CaptureBackendKind::parse(s)
    }
}

impl PacingMode {
    pub const ALL: [PacingMode; 3] = [PacingMode::Auto, PacingMode::Vsync, PacingMode::Unlimited];

    pub fn as_str(&self) -> &'static str {
        match self {
            PacingMode::Auto => "auto",
            PacingMode::Vsync => "vsync",
            PacingMode::Unlimited => "unlimited",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "default" => Ok(PacingMode::Auto),
            "vsync" | "sync" => Ok(PacingMode::Vsync),
            "unlimited" | "none" | "free" => Ok(PacingMode::Unlimited),
            other => Err(FluidError::config(format!(
                "unknown pacing mode '{other}' (auto | vsync | unlimited)"
            ))),
        }
    }
}

impl fmt::Display for PacingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PacingMode {
    type Err = FluidError;
    fn from_str(s: &str) -> Result<Self> {
        PacingMode::parse(s)
    }
}

/// Round noisy f32-to-f64 float literals (0.30000001192092896) to a readable
/// short form (0.3) so generated config files stay human editable.
fn tidy_floats(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i] as char;
        let starts_number = c.is_ascii_digit()
            || (c == '-' && i + 1 < bytes.len() && (bytes[i + 1] as char).is_ascii_digit());
        if starts_number && (i == 0 || !is_ident_byte(bytes[i - 1])) {
            let start = i;
            let mut j = i;
            if bytes[j] as char == '-' {
                j += 1;
            }
            while j < bytes.len() && (bytes[j] as char).is_ascii_digit() {
                j += 1;
            }
            let mut is_float = false;
            if j < bytes.len() && bytes[j] as char == '.' {
                is_float = true;
                j += 1;
                while j < bytes.len() && (bytes[j] as char).is_ascii_digit() {
                    j += 1;
                }
            }
            if is_float {
                if let Ok(value) = text[start..j].parse::<f64>() {
                    let mut short = format!("{value:.6}");
                    while short.ends_with('0') {
                        short.pop();
                    }
                    if short.ends_with('.') {
                        short.push('0');
                    }
                    out.push_str(&short);
                } else {
                    out.push_str(&text[start..j]);
                }
            } else {
                out.push_str(&text[start..j]);
            }
            i = j;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'.' || b == b'-'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_floats_are_readable() {
        let text = FluidConfig::template();
        assert!(text.contains("sharpness = 0.3"), "{text}");
        assert!(!text.contains("0.30000001192092896"), "{text}");
        let cfg = FluidConfig::from_toml_str(&text).unwrap();
        assert!((cfg.upscaler.sharpness - 0.3).abs() < 1e-6);
    }

    #[test]
    fn tidy_floats_keeps_integers() {
        let s = tidy_floats("a = 60\nb = 0.18000000715255737\nc = -1.5\nd = 1e-3");
        assert!(s.contains("a = 60"));
        assert!(s.contains("b = 0.18"));
        assert!(s.contains("c = -1.5"));
        assert!(s.contains("d = 1e-3"));
    }

    #[test]
    fn interp_backend_parses_and_roundtrips() {
        assert_eq!(InterpBackendKind::default(), InterpBackendKind::Auto);
        assert_eq!(
            "gpu".parse::<InterpBackendKind>().unwrap(),
            InterpBackendKind::Gpu
        );
        assert_eq!(
            "SIMD".parse::<InterpBackendKind>().unwrap(),
            InterpBackendKind::Cpu
        );
        assert!(matches!(
            "vulkan".parse::<InterpBackendKind>(),
            Err(FluidError::Config(_))
        ));
        let cfg = FluidConfig::default();
        let text = cfg.to_toml_string().unwrap();
        assert!(text.contains("backend = \"auto\""), "{text}");
        let back = FluidConfig::from_toml_str(&text).unwrap();
        assert_eq!(back.interpolator.backend, InterpBackendKind::Auto);
    }

    #[test]
    fn defaults_are_valid() {
        FluidConfig::default().validate().unwrap();
    }

    #[test]
    fn all_presets_are_valid() {
        for p in QualityPreset::ALL {
            let cfg = FluidConfig::with_preset(p);
            cfg.validate()
                .unwrap_or_else(|e| panic!("preset {p:?} invalid: {e}"));
        }
    }

    #[test]
    fn toggle_roundtrip() {
        let cfg = FluidConfig::default();
        let text = cfg.to_toml_string().unwrap();
        let back = FluidConfig::from_toml_str(&text).unwrap();
        assert_eq!(back.interpolator.multiplier, cfg.interpolator.multiplier);
        assert_eq!(back.upscaler.filter, cfg.upscaler.filter);
        assert_eq!(back.capture.include_cursor, cfg.capture.include_cursor);
    }

    #[test]
    fn present_backend_parses_and_roundtrips() {
        assert_eq!(PresentBackendKind::default(), PresentBackendKind::Auto);
        assert_eq!(
            "dxgi".parse::<PresentBackendKind>().unwrap(),
            PresentBackendKind::Dxgi
        );
        assert_eq!(
            "GDI".parse::<PresentBackendKind>().unwrap(),
            PresentBackendKind::Window
        );
        assert!("vulkan".parse::<PresentBackendKind>().is_err());
        assert_eq!(PresentBackendKind::Dxgi.as_str(), "dxgi");
        let cfg = FluidConfig::default();
        assert_eq!(cfg.present.backend, PresentBackendKind::Auto);
        let text = cfg.to_toml_string().unwrap();
        assert!(text.contains("backend = \"auto\""), "{text}");
        let back = FluidConfig::from_toml_str(&text).unwrap();
        assert_eq!(back.present.backend, PresentBackendKind::Auto);
    }
    #[test]
    fn partial_toml_uses_defaults() {
        let cfg = FluidConfig::from_toml_str("profile = \"abc\"\n").unwrap();
        assert_eq!(cfg.profile, "abc");
        assert_eq!(cfg.interpolator.multiplier, 2);
    }

    #[test]
    fn rejects_bad_values() {
        let bad = "[\"interpolator\"]\nmultiplier = 0\n";
        assert!(FluidConfig::from_toml_str(bad).is_err());
        let bad = "[\"upscaler\"]\nsharpness = 5.0\n";
        assert!(FluidConfig::from_toml_str(bad).is_err());
    }

    #[test]
    fn scale_resolution() {
        let src = Dims::new(1920, 1080);
        let mut s = ScaleConfig::default();
        assert_eq!(s.resolve(src), src);
        s.mode = ScaleMode::Factor;
        s.factor = 2.0;
        assert_eq!(s.resolve(src), Dims::new(3840, 2160));
        s.mode = ScaleMode::Target;
        s.target_w = 2560;
        s.target_h = 1440;
        assert_eq!(s.resolve(src), Dims::new(2560, 1440));
    }

    #[test]
    fn preset_parsing() {
        assert_eq!(
            "balanced".parse::<QualityPreset>().unwrap(),
            QualityPreset::Balanced
        );
        assert_eq!(
            "ultra-quality".parse::<QualityPreset>().unwrap(),
            QualityPreset::UltraQuality
        );
        assert!("nope".parse::<QualityPreset>().is_err());
        assert_eq!(
            "lanczos".parse::<UpscaleFilter>().unwrap(),
            UpscaleFilter::Lanczos
        );
        assert_eq!(UpscaleFilter::Off.to_string(), "off");
    }

    #[test]
    fn template_is_parseable() {
        FluidConfig::from_toml_str(&FluidConfig::template()).unwrap();
    }

    #[test]
    fn file_roundtrip() {
        let dir = std::env::temp_dir().join("fluid_cfg_test");
        let path = dir.join("fluid.toml");
        let cfg = FluidConfig::with_preset(QualityPreset::Quality);
        cfg.save(&path).unwrap();
        let back = FluidConfig::load(&path).unwrap();
        assert_eq!(back.interpolator.quality, QualityPreset::Quality);
        let missing = dir.join("does_not_exist.toml");
        let d = FluidConfig::load_or_default(&missing).unwrap();
        assert_eq!(d.interpolator.multiplier, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn window_target_validates_and_marks_set() {
        let empty = WindowTarget::default();
        assert!(!empty.is_set());
        empty.validate().unwrap();
        let by_title = WindowTarget {
            title: "Notepad".to_string(),
            ..WindowTarget::default()
        };
        assert!(by_title.is_set());
        by_title.validate().unwrap();
        let by_process = WindowTarget {
            process: "notepad.exe".to_string(),
            ..WindowTarget::default()
        };
        assert!(by_process.is_set());
        // An id and a text match together are ambiguous by definition.
        let both = WindowTarget {
            id: 7,
            title: "Notepad".to_string(),
            ..WindowTarget::default()
        };
        assert!(both.validate().is_err());
        let cfg = CaptureConfig {
            window: Some(both),
            ..CaptureConfig::default()
        };
        assert!(cfg.validate().is_err());
    }
}
