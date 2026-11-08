//! Beginner-friendly UI choices mapped onto the `fluid-core` config types.
//!
//! The UI deliberately offers a small, well explained set of choices.  This
//! module maps those choices to and from `FluidConfig` so the rest of the GUI
//! never has to know about the raw enums.

use fluid_core::{
    Dims, FluidConfig, InterpBackendKind, InterpMode, QualityPreset, ScaleConfig, ScaleMode,
    UpscaleBackendKind, UpscaleFilter,
};

/// The five quality presets, in the order shown in the UI.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PresetChoice {
    UltraPerformance,
    Performance,
    Balanced,
    Quality,
    UltraQuality,
}

impl PresetChoice {
    pub const ALL: [Self; 5] = [
        Self::UltraPerformance,
        Self::Performance,
        Self::Balanced,
        Self::Quality,
        Self::UltraQuality,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::UltraPerformance => "极速",
            Self::Performance => "流畅",
            Self::Balanced => "均衡",
            Self::Quality => "高画质",
            Self::UltraQuality => "极致",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Self::UltraPerformance => "最低开销，适合核显和老电脑",
            Self::Performance => "开销较小，画质够用",
            Self::Balanced => "速度与画质兼顾，推荐大多数用户",
            Self::Quality => "更多计算、更清晰，需要较强的显卡/CPU",
            Self::UltraQuality => "最高画质，配置较低的电脑可能跑不满帧",
        }
    }

    pub fn to_preset(self) -> QualityPreset {
        match self {
            Self::UltraPerformance => QualityPreset::UltraPerformance,
            Self::Performance => QualityPreset::Performance,
            Self::Balanced => QualityPreset::Balanced,
            Self::Quality => QualityPreset::Quality,
            Self::UltraQuality => QualityPreset::UltraQuality,
        }
    }

    pub fn from_preset(preset: QualityPreset) -> Self {
        match preset {
            QualityPreset::UltraPerformance => Self::UltraPerformance,
            QualityPreset::Performance => Self::Performance,
            QualityPreset::Balanced => Self::Balanced,
            QualityPreset::Quality => Self::Quality,
            QualityPreset::UltraQuality => Self::UltraQuality,
        }
    }
}

/// Frame generation multiplier as exposed to the user (off / 2x / 3x / 4x).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InterpChoice {
    Off,
    X2,
    X3,
    X4,
}

impl InterpChoice {
    pub const ALL: [Self; 4] = [Self::Off, Self::X2, Self::X3, Self::X4];

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "关",
            Self::X2 => "2x",
            Self::X3 => "3x",
            Self::X4 => "4x",
        }
    }

    pub fn multiplier(self) -> u8 {
        match self {
            Self::Off => 1,
            Self::X2 => 2,
            Self::X3 => 3,
            Self::X4 => 4,
        }
    }
}

/// Resolution scaling as exposed to the user (off / 1.5x / 2x / 4K).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScaleChoice {
    Off,
    X1_5,
    X2,
    X4K,
}

impl ScaleChoice {
    pub const ALL: [Self; 4] = [Self::Off, Self::X1_5, Self::X2, Self::X4K];

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "关",
            Self::X1_5 => "1.5x",
            Self::X2 => "2x",
            Self::X4K => "4K",
        }
    }

    pub fn scale_config(self) -> ScaleConfig {
        match self {
            Self::Off => ScaleConfig {
                mode: ScaleMode::Native,
                ..ScaleConfig::default()
            },
            Self::X1_5 => ScaleConfig {
                mode: ScaleMode::Factor,
                factor: 1.5,
                ..ScaleConfig::default()
            },
            Self::X2 => ScaleConfig {
                mode: ScaleMode::Factor,
                factor: 2.0,
                ..ScaleConfig::default()
            },
            Self::X4K => ScaleConfig {
                mode: ScaleMode::Target,
                target_w: 3840,
                target_h: 2160,
                ..ScaleConfig::default()
            },
        }
    }
}

/// The upscaling filters offered in the UI.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FilterChoice {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos,
    Adaptive,
    Temporal,
}

impl FilterChoice {
    pub const ALL: [Self; 6] = [
        Self::Nearest,
        Self::Bilinear,
        Self::Bicubic,
        Self::Lanczos,
        Self::Adaptive,
        Self::Temporal,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Bilinear => "bilinear",
            Self::Bicubic => "bicubic",
            Self::Lanczos => "lanczos",
            Self::Adaptive => "adaptive",
            Self::Temporal => "temporal",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Self::Nearest => "最快，边缘有锯齿",
            Self::Bilinear => "速度快，画面略软",
            Self::Bicubic => "速度与清晰度平衡",
            Self::Lanczos => "最锐利，开销最大",
            Self::Adaptive => "按画面内容自动选择，推荐",
            Self::Temporal => "利用前后帧信息，动态画面更稳",
        }
    }

    pub fn to_filter(self) -> UpscaleFilter {
        match self {
            Self::Nearest => UpscaleFilter::Nearest,
            Self::Bilinear => UpscaleFilter::Bilinear,
            Self::Bicubic => UpscaleFilter::Bicubic,
            Self::Lanczos => UpscaleFilter::Lanczos,
            Self::Adaptive => UpscaleFilter::Adaptive,
            Self::Temporal => UpscaleFilter::Temporal,
        }
    }

    pub fn from_filter(filter: UpscaleFilter) -> Self {
        match filter {
            UpscaleFilter::Nearest => Self::Nearest,
            UpscaleFilter::Bilinear => Self::Bilinear,
            UpscaleFilter::Bicubic => Self::Bicubic,
            UpscaleFilter::Lanczos => Self::Lanczos,
            UpscaleFilter::Temporal => Self::Temporal,
            // "off" is not a choice in the UI; adaptive is the safe default.
            UpscaleFilter::Adaptive | UpscaleFilter::Off => Self::Adaptive,
        }
    }
}

/// Which implementation runs the upscaling stage (自动 / CPU / GPU).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BackendChoice {
    Auto,
    Cpu,
    Gpu,
}

impl BackendChoice {
    pub const ALL: [Self; 3] = [Self::Auto, Self::Cpu, Self::Gpu];

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "自动（推荐）",
            Self::Cpu => "CPU",
            Self::Gpu => "GPU",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Self::Auto => "自动会优先用 GPU，不可用时回退 CPU；GPU 需要支持 Direct3D 11 的显卡。",
            Self::Cpu => "只用处理器计算，任何电脑都能用，速度取决于 CPU。",
            Self::Gpu => "用显卡计算，通常比 CPU 快；需要支持 Direct3D 11 的显卡。",
        }
    }

    pub fn to_backend(self) -> UpscaleBackendKind {
        match self {
            Self::Auto => UpscaleBackendKind::Auto,
            Self::Cpu => UpscaleBackendKind::Cpu,
            Self::Gpu => UpscaleBackendKind::Gpu,
        }
    }

    pub fn from_backend(kind: UpscaleBackendKind) -> Self {
        match kind {
            UpscaleBackendKind::Auto => Self::Auto,
            UpscaleBackendKind::Cpu => Self::Cpu,
            UpscaleBackendKind::Gpu => Self::Gpu,
        }
    }

    /// Parse a backend name, falling back to 自动 for anything unknown.
    ///
    /// Hand-edited config files or an older/newer CLI may contain a backend
    /// name this GUI build does not know; the UI must still open.
    pub fn from_name(name: &str) -> Self {
        UpscaleBackendKind::parse(name)
            .map(Self::from_backend)
            .unwrap_or(Self::Auto)
    }
}

/// Human readable name of the backend a finished run actually used.
///
/// The engine reports the backend that ran ("cpu", "d3d11-compute") or
/// "none" when upscaling was disabled, so the UI can tell the user what
/// really happened instead of what was requested.
pub fn backend_display_name(report_backend: &str) -> String {
    match report_backend.trim() {
        "" | "none" => "未启用放大".to_owned(),
        "cpu" => "CPU（处理器）".to_owned(),
        "d3d11-compute" | "d3d11" | "gpu" => "GPU（Direct3D 11 计算）".to_owned(),
        "wgpu" => "GPU（图形处理器）".to_owned(),
        other => other.to_owned(),
    }
}

/// Which implementation runs the frame generation (插帧) stage.
///
/// The engine tries the GPU first for Auto and silently falls back to the CPU
/// path, so the UI must always show what actually ran instead of what was
/// requested.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InterpBackendChoice {
    Auto,
    Cpu,
    Gpu,
}

impl InterpBackendChoice {
    pub const ALL: [Self; 3] = [Self::Auto, Self::Cpu, Self::Gpu];

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "自动（推荐）",
            Self::Cpu => "CPU",
            Self::Gpu => "GPU",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Self::Auto => {
                "自动会优先用 GPU 做插帧，GPU 不可用或出错时自动回退 CPU；报告里显示实际用的后端。"
            }
            Self::Cpu => "只用处理器做插帧，任何电脑都能用，速度取决于 CPU。",
            Self::Gpu => "用显卡做插帧；显卡不可用时会自动回退 CPU，状态栏会显示实际后端。",
        }
    }

    pub fn to_backend(self) -> InterpBackendKind {
        match self {
            Self::Auto => InterpBackendKind::Auto,
            Self::Cpu => InterpBackendKind::Cpu,
            Self::Gpu => InterpBackendKind::Gpu,
        }
    }

    pub fn from_backend(kind: InterpBackendKind) -> Self {
        match kind {
            InterpBackendKind::Auto => Self::Auto,
            InterpBackendKind::Cpu => Self::Cpu,
            InterpBackendKind::Gpu => Self::Gpu,
        }
    }

    /// Parse a backend name, falling back to 自动 for anything unknown.
    pub fn from_name(name: &str) -> Self {
        InterpBackendKind::parse(name)
            .map(Self::from_backend)
            .unwrap_or(Self::Auto)
    }
}

/// Human readable name of the frame generation backend a finished run used.
///
/// The engine reports "cpu", "wgpu-flow" or "none" (interpolation disabled).
pub fn interp_backend_display_name(report_backend: &str) -> String {
    match report_backend.trim() {
        "" | "none" => "未启用插帧".to_owned(),
        "cpu" => "CPU（处理器）".to_owned(),
        "wgpu-flow" | "wgpu" | "gpu" => "GPU（计算）".to_owned(),
        // Fused interpolation + upscaling in one GPU submission (discrete adapters).
        "wgpu-fused" => "GPU（融合计算）".to_owned(),
        other => other.to_owned(),
    }
}

/// All settings the user can tweak in the quick controls.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct UiQuality {
    pub preset: PresetChoice,
    pub interp: InterpChoice,
    pub scale: ScaleChoice,
    pub filter: FilterChoice,
    /// Which implementation runs the upscaling stage.
    pub backend: BackendChoice,
    /// Which implementation runs the frame generation (插帧) stage.
    pub interp_backend: InterpBackendChoice,
    pub sharpness: f32,
    pub vsync: bool,
    pub borderless: bool,
}

impl Default for UiQuality {
    fn default() -> Self {
        let config = FluidConfig::default();
        Self::from_config(&config)
    }
}

impl UiQuality {
    /// Reflect an existing config (loaded from disk) in the UI choices.
    pub fn from_config(config: &FluidConfig) -> Self {
        let preset = QualityPreset::ALL
            .iter()
            .copied()
            .find(|preset| preset.as_str() == config.profile)
            .unwrap_or(QualityPreset::Balanced);

        let interp = match (config.interpolator.mode, config.interpolator.multiplier) {
            (InterpMode::Off, _) => InterpChoice::Off,
            (_, 3) => InterpChoice::X3,
            (_, 4) => InterpChoice::X4,
            (_, 2) => InterpChoice::X2,
            (_, _) => InterpChoice::X2,
        };

        let scale = match config.upscaler.scale.mode {
            ScaleMode::Native => ScaleChoice::Off,
            ScaleMode::Target => ScaleChoice::X4K,
            ScaleMode::Factor if (config.upscaler.scale.factor - 2.0).abs() < 0.01 => {
                ScaleChoice::X2
            }
            ScaleMode::Factor => ScaleChoice::X1_5,
        };

        Self {
            preset: PresetChoice::from_preset(preset),
            interp,
            scale,
            filter: FilterChoice::from_filter(config.upscaler.filter),
            backend: BackendChoice::from_backend(config.upscaler.backend),
            interp_backend: InterpBackendChoice::from_backend(config.interpolator.backend),
            sharpness: config.upscaler.sharpness,
            vsync: config.present.vsync,
            borderless: config.present.borderless,
        }
    }

    /// Write the UI choices into a `FluidConfig`.
    pub fn apply(&self, config: &mut FluidConfig) {
        config.apply_preset(self.preset.to_preset());
        config.interpolator.mode = if self.interp == InterpChoice::Off {
            InterpMode::Off
        } else {
            InterpMode::Fixed
        };
        config.interpolator.multiplier = self.interp.multiplier();
        config.upscaler.scale = self.scale.scale_config();
        config.upscaler.filter = if self.scale == ScaleChoice::Off {
            UpscaleFilter::Off
        } else {
            self.filter.to_filter()
        };
        config.upscaler.backend = self.backend.to_backend();
        config.interpolator.backend = self.interp_backend.to_backend();
        config.upscaler.sharpness = self.sharpness.clamp(0.0, 1.0);
        config.present.vsync = self.vsync;
        config.present.borderless = self.borderless;
        config.present.fullscreen = self.borderless;
    }
}

impl UiQuality {
    /// Pick a preset and move the sharpness slider to that preset's default.
    pub fn select_preset(&mut self, preset: PresetChoice) {
        self.preset = preset;
        let mut probe = FluidConfig::default();
        probe.apply_preset(preset.to_preset());
        self.sharpness = probe.upscaler.sharpness;
    }
}

/// The resolution of the built-in test pattern.
pub const SYNTHETIC_DIMS: Dims = Dims {
    width: 1280,
    height: 720,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn quality() -> UiQuality {
        UiQuality {
            preset: PresetChoice::Balanced,
            interp: InterpChoice::Off,
            scale: ScaleChoice::Off,
            filter: FilterChoice::Adaptive,
            backend: BackendChoice::Auto,
            interp_backend: InterpBackendChoice::Auto,
            sharpness: 0.3,
            vsync: true,
            borderless: false,
        }
    }

    #[test]
    fn preset_maps_to_sharpness_and_profile() {
        let mut config = FluidConfig::default();
        let mut ui = quality();
        ui.select_preset(PresetChoice::Quality);
        ui.apply(&mut config);
        assert_eq!(config.profile, QualityPreset::Quality.as_str());
        assert!((ui.sharpness - 0.4).abs() < 1e-4);
        assert!((config.upscaler.sharpness - 0.4).abs() < 1e-4);
    }

    #[test]
    fn select_preset_moves_the_sharpness_slider() {
        let mut ui = quality();
        ui.select_preset(PresetChoice::UltraPerformance);
        assert!((ui.sharpness - 0.2).abs() < 1e-4);
        ui.select_preset(PresetChoice::UltraQuality);
        assert!((ui.sharpness - 0.5).abs() < 1e-4);
    }

    #[test]
    fn interpolation_choice_maps_to_mode_and_multiplier() {
        let mut config = FluidConfig::default();
        let mut ui = quality();
        ui.interp = InterpChoice::X3;
        ui.apply(&mut config);
        assert_eq!(config.interpolator.mode, InterpMode::Fixed);
        assert_eq!(config.interpolator.multiplier, 3);
    }

    #[test]
    fn interpolation_off_disables_the_interpolator() {
        let mut config = FluidConfig::default();
        let mut ui = quality();
        ui.interp = InterpChoice::Off;
        ui.apply(&mut config);
        assert_eq!(config.interpolator.mode, InterpMode::Off);
    }

    #[test]
    fn scale_choice_resolves_expected_output_size() {
        let source = Dims::new(1280, 720);
        assert_eq!(
            ScaleChoice::Off.scale_config().resolve(source),
            Dims::new(1280, 720)
        );
        assert_eq!(
            ScaleChoice::X1_5.scale_config().resolve(source),
            Dims::new(1920, 1080)
        );
        assert_eq!(
            ScaleChoice::X2.scale_config().resolve(source),
            Dims::new(2560, 1440)
        );
        assert_eq!(
            ScaleChoice::X4K.scale_config().resolve(source),
            Dims::new(3840, 2160)
        );
    }

    #[test]
    fn scale_off_turns_upscaling_off() {
        let mut config = FluidConfig::default();
        let mut ui = quality();
        ui.scale = ScaleChoice::Off;
        ui.filter = FilterChoice::Lanczos;
        ui.apply(&mut config);
        assert_eq!(config.upscaler.filter, UpscaleFilter::Off);
    }

    #[test]
    fn scale_on_maps_the_selected_filter() {
        let mut config = FluidConfig::default();
        let mut ui = quality();
        ui.scale = ScaleChoice::X2;
        ui.filter = FilterChoice::Lanczos;
        ui.apply(&mut config);
        assert_eq!(config.upscaler.filter, UpscaleFilter::Lanczos);
        assert_eq!(config.upscaler.scale.mode, ScaleMode::Factor);
        assert!((config.upscaler.scale.factor - 2.0).abs() < 1e-4);
    }

    #[test]
    fn vsync_and_borderless_are_persisted() {
        let mut config = FluidConfig::default();
        let mut ui = quality();
        ui.vsync = false;
        ui.borderless = true;
        ui.apply(&mut config);
        assert!(!config.present.vsync);
        assert!(config.present.borderless);
        assert!(config.present.fullscreen);
    }

    #[test]
    fn from_config_roundtrips_the_main_choices() {
        let mut config = FluidConfig::default();
        let mut ui = quality();
        ui.preset = PresetChoice::UltraQuality;
        ui.interp = InterpChoice::X2;
        ui.scale = ScaleChoice::X1_5;
        ui.filter = FilterChoice::Bicubic;
        ui.sharpness = 0.55;
        ui.vsync = false;
        ui.borderless = false;
        ui.apply(&mut config);

        let back = UiQuality::from_config(&config);
        assert_eq!(back.preset, PresetChoice::UltraQuality);
        assert_eq!(back.interp, InterpChoice::X2);
        assert_eq!(back.scale, ScaleChoice::X1_5);
        assert_eq!(back.filter, FilterChoice::Bicubic);
        assert!((back.sharpness - 0.55).abs() < 1e-4);
        assert!(!back.vsync);
    }

    #[test]
    fn backend_choice_roundtrips_through_config() {
        for choice in BackendChoice::ALL {
            let mut config = FluidConfig::default();
            let mut ui = quality();
            ui.backend = choice;
            ui.apply(&mut config);
            assert_eq!(config.upscaler.backend, choice.to_backend());
            assert_eq!(UiQuality::from_config(&config).backend, choice);
        }
    }

    #[test]
    fn backend_mapping_writes_cpu_and_gpu() {
        let mut config = FluidConfig::default();
        let mut ui = quality();
        ui.backend = BackendChoice::Cpu;
        ui.apply(&mut config);
        assert_eq!(config.upscaler.backend, UpscaleBackendKind::Cpu);
        ui.backend = BackendChoice::Gpu;
        ui.apply(&mut config);
        assert_eq!(config.upscaler.backend, UpscaleBackendKind::Gpu);
    }

    #[test]
    fn default_quality_requests_the_auto_backend() {
        assert_eq!(UiQuality::default().backend, BackendChoice::Auto);
    }

    #[test]
    fn unknown_backend_name_falls_back_to_auto() {
        assert_eq!(BackendChoice::from_name("auto"), BackendChoice::Auto);
        assert_eq!(BackendChoice::from_name("CPU"), BackendChoice::Cpu);
        assert_eq!(BackendChoice::from_name(" d3d11 "), BackendChoice::Gpu);
        // Older/newer config files may use a name this build does not know yet.
        assert_eq!(BackendChoice::from_name("vulkan"), BackendChoice::Auto);
        assert_eq!(BackendChoice::from_name(""), BackendChoice::Auto);
    }

    #[test]
    fn backend_display_names_are_beginner_friendly() {
        assert_eq!(backend_display_name("cpu"), "CPU（处理器）");
        assert_eq!(
            backend_display_name("d3d11-compute"),
            "GPU（Direct3D 11 计算）"
        );
        assert_eq!(backend_display_name("wgpu"), "GPU（图形处理器）");
        assert_eq!(backend_display_name("none"), "未启用放大");
        assert_eq!(backend_display_name(""), "未启用放大");
        assert_eq!(backend_display_name("metal"), "metal");
    }

    #[test]
    fn interp_backend_choice_roundtrips_through_config() {
        for choice in InterpBackendChoice::ALL {
            let mut config = FluidConfig::default();
            let mut ui = quality();
            ui.interp_backend = choice;
            ui.apply(&mut config);
            assert_eq!(config.interpolator.backend, choice.to_backend());
            assert_eq!(UiQuality::from_config(&config).interp_backend, choice);
        }
    }

    #[test]
    fn interp_backend_mapping_writes_cpu_and_gpu() {
        let mut config = FluidConfig::default();
        let mut ui = quality();
        ui.interp_backend = InterpBackendChoice::Cpu;
        ui.apply(&mut config);
        assert_eq!(config.interpolator.backend, InterpBackendKind::Cpu);
        ui.interp_backend = InterpBackendChoice::Gpu;
        ui.apply(&mut config);
        assert_eq!(config.interpolator.backend, InterpBackendKind::Gpu);
    }

    #[test]
    fn default_quality_requests_the_auto_interp_backend() {
        assert_eq!(
            UiQuality::default().interp_backend,
            InterpBackendChoice::Auto
        );
    }

    #[test]
    fn unknown_interp_backend_name_falls_back_to_auto() {
        assert_eq!(
            InterpBackendChoice::from_name("auto"),
            InterpBackendChoice::Auto
        );
        assert_eq!(
            InterpBackendChoice::from_name("CPU"),
            InterpBackendChoice::Cpu
        );
        assert_eq!(
            InterpBackendChoice::from_name(" wgpu "),
            InterpBackendChoice::Gpu
        );
        assert_eq!(
            InterpBackendChoice::from_name("vulkan"),
            InterpBackendChoice::Auto
        );
        assert_eq!(
            InterpBackendChoice::from_name(""),
            InterpBackendChoice::Auto
        );
    }

    #[test]
    fn interp_backend_display_names_are_beginner_friendly() {
        assert_eq!(interp_backend_display_name("cpu"), "CPU（处理器）");
        assert_eq!(interp_backend_display_name("wgpu-flow"), "GPU（计算）");
        assert_eq!(interp_backend_display_name("wgpu-fused"), "GPU（融合计算）");
        assert_eq!(interp_backend_display_name("none"), "未启用插帧");
        assert_eq!(interp_backend_display_name(""), "未启用插帧");
        assert_eq!(interp_backend_display_name("metal"), "metal");
    }
}
