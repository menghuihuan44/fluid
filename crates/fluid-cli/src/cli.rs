//! Command line surface: argument structs plus the mapping from flags onto a
//! validated FluidConfig.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use fluid_core::{
    Dims, FluidConfig, FluidError, InterpMode, QualityPreset, RectConfig, ScaleMode, UpscaleFilter,
};

/// Convenience alias for clap value parsers.
type ParseResult<T> = std::result::Result<T, String>;

/// Real-time frame generation and upscaling for any game, video or screen.
#[derive(Debug, Parser)]
#[command(
    name = "fluid",
    version,
    about = "Real-time frame generation and upscaling for any game, video or screen",
    long_about = None,
    propagate_version = true
)]
pub struct Cli {
    /// Log level: off | error | warn | info | debug | trace.
    #[arg(long, global = true, value_name = "LEVEL", default_value = "info")]
    pub log_level: String,

    /// Suppress informational output.
    #[arg(long, global = true)]
    pub quiet: bool,

    /// Emit machine readable output where the command supports it.
    #[arg(long, global = true)]
    pub json: bool,

    /// TOML configuration file to load (defaults are used when omitted).
    #[arg(long, global = true, value_name = "FILE")]
    pub config: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Option<Command>,
}

/// Top level subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Capture the screen in real time, synthesise frames and show the result.
    Run(RunArgs),
    /// Process a y4m video or a PNG sequence offline.
    Video(VideoArgs),
    /// Benchmark frame generation + upscaling on a synthetic source.
    Bench(BenchArgs),
    /// List displays, capture backends, present backends and windows.
    Devices(DevicesArgs),
    /// Print, validate or write the default configuration.
    Config(ConfigArgs),
    /// Inspect this machine, measure it and recommend settings.
    Doctor(DoctorArgs),
    /// Open the Fluid desktop app (graphical interface).
    Gui(GuiArgs),
    /// Show version and build information.
    Info,
}

/// Options for the device / backend listing.
#[derive(Debug, Args, Default)]
pub struct DevicesArgs {
    /// Also list capturable application windows (pick one with --window).
    #[arg(long)]
    pub windows: bool,
}

/// Options for the graphical desktop app.
#[derive(Debug, Args)]
pub struct GuiArgs {
    /// Close automatically after N seconds (used by automated checks).
    #[arg(long, value_name = "SECS")]
    pub smoke: Option<f64>,

    /// Extra arguments forwarded to the GUI process.
    #[arg(last = true, value_name = "ARGS")]
    pub extra: Vec<String>,
}

/// Options for the machine inspection / recommendation command.
#[derive(Debug, Args)]
pub struct DoctorArgs {
    /// Frames measured per probe profile (more frames = more stable, slower).
    #[arg(long, value_name = "N", default_value_t = 20)]
    pub frames: u64,

    /// Output framerate you want to reach; drives the recommendation.
    #[arg(long, value_name = "HZ", default_value_t = 60.0)]
    pub target_fps: f64,
}

/// Options for the real-time screen pipeline.
#[derive(Debug, Args)]
pub struct RunArgs {
    /// Capture backend: auto | dxgi | mock.
    #[arg(long, value_name = "BACKEND")]
    pub capture: Option<String>,

    /// Monitor index to capture (see the devices subcommand).
    #[arg(long, value_name = "N")]
    pub monitor: Option<usize>,

    /// Capture only this region of the monitor, as WxH+X+Y.
    #[arg(long, value_name = "REGION", value_parser = parse_region)]
    pub region: Option<RectConfig>,

    /// Capture one application window whose title contains this text.
    #[arg(long, value_name = "TITLE", conflicts_with_all = ["window_id", "window_process"])]
    pub window: Option<String>,

    /// Capture the window owned by this process name (with or without .exe).
    #[arg(long, value_name = "NAME", conflicts_with = "window_id")]
    pub window_process: Option<String>,

    /// Capture the window with this native id (see `fluid devices --windows`).
    #[arg(long, value_name = "ID")]
    pub window_id: Option<u64>,

    /// Do not open a window: capture and process, then discard frames.
    #[arg(long)]
    pub no_present: bool,

    /// Stop after N captured frames (useful for testing).
    #[arg(long, value_name = "N")]
    pub frames: Option<u64>,

    /// Do not composite the mouse cursor into captured frames.
    #[arg(long)]
    pub no_cursor: bool,

    /// Soft cap on the output framerate in Hz (0 disables the cap).
    #[arg(long, value_name = "HZ")]
    pub present_fps: Option<f32>,

    /// Write a JSON report to this file when the run finishes.
    #[arg(long, value_name = "FILE")]
    pub report: Option<PathBuf>,

    #[command(flatten)]
    pub over: Overrides,
}

/// Options for offline video processing.
#[derive(Debug, Args)]
pub struct VideoArgs {
    /// Input y4m file, a directory containing PNG frames, or "-" for stdin.
    #[arg(long, short, value_name = "PATH")]
    pub input: PathBuf,

    /// Output y4m file, .avi file (MJPEG, no ffmpeg needed), a directory that
    /// receives PNG frames, or "-" for stdout.
    #[arg(long, short, value_name = "PATH")]
    pub output: PathBuf,

    /// JPEG quality for .avi output (1-100, default 88).
    #[arg(long, value_name = "Q", value_parser = clap::value_parser!(u8).range(1..=100))]
    pub jpeg_quality: Option<u8>,

    /// Stop after N input frames.
    #[arg(long, value_name = "N")]
    pub frames: Option<u64>,

    /// Write a JSON report to this file when the run finishes.
    #[arg(long, value_name = "FILE")]
    pub report: Option<PathBuf>,

    #[command(flatten)]
    pub over: Overrides,
}

/// Options for the synthetic benchmark.
#[derive(Debug, Args)]
pub struct BenchArgs {
    /// Source frame size as WxH.
    #[arg(long, value_name = "WxH", default_value = "1920x1080", value_parser = parse_dims)]
    pub size: Dims,

    /// Number of frames to process.
    #[arg(long, value_name = "N", default_value_t = 120)]
    pub frames: u64,

    /// Frames to process before measurements start.
    #[arg(long, value_name = "N", default_value_t = 10)]
    pub warmup: u64,

    /// Write a JSON report to this file when the benchmark finishes.
    #[arg(long, value_name = "FILE")]
    pub report: Option<PathBuf>,

    #[command(flatten)]
    pub over: Overrides,
}

/// Options for the config subcommand.
#[derive(Debug, Args)]
pub struct ConfigArgs {
    /// Validate this config file and print the normalised version.
    #[arg(long, value_name = "FILE")]
    pub check: Option<PathBuf>,

    /// Write the annotated template to this file.
    #[arg(long, value_name = "FILE")]
    pub write: Option<PathBuf>,

    /// Print the effective config after applying the override flags.
    #[arg(long)]
    pub effective: bool,

    #[command(flatten)]
    pub over: Overrides,
}

/// Quality / latency overrides shared by run, video and bench.
///
/// Everything here maps directly onto FluidConfig so the same flag works for
/// live capture and offline video. Later flags win over earlier ones.
#[derive(Debug, Clone, Default, Args)]
pub struct Overrides {
    /// Quality preset: ultra-performance | performance | balanced | quality | ultra-quality.
    #[arg(long, value_name = "PRESET")]
    pub preset: Option<String>,

    /// Frame generation mode: off | fixed | adaptive.
    #[arg(long, value_name = "MODE")]
    pub interp: Option<String>,

    /// Generate this many frames per real frame (1 disables generation).
    #[arg(long, value_name = "N")]
    pub multiplier: Option<u8>,

    /// Upscale factor relative to the source, e.g. 1.5 or 2.0 (0 disables).
    #[arg(long, value_name = "FACTOR")]
    pub scale: Option<f32>,

    /// Upscale to an exact output size as WxH.
    #[arg(long, value_name = "WxH", value_parser = parse_dims)]
    pub target: Option<Dims>,

    /// Upscale filter: off | nearest | bilinear | bicubic | lanczos | adaptive | temporal.
    #[arg(long, value_name = "FILTER")]
    pub filter: Option<String>,

    /// Upscaling backend: auto | cpu | gpu (gpu uses wgpu compute: Vulkan on Windows and Android).
    #[arg(long, value_name = "BACKEND")]
    pub backend: Option<String>,

    /// Frame generation backend: auto | cpu | gpu (gpu runs optical flow and synthesis on the GPU).
    #[arg(long, value_name = "INTERP_BACKEND")]
    pub interp_backend: Option<String>,

    /// Sharpening strength, 0.0 .. 1.0.
    #[arg(long, value_name = "0..1")]
    pub sharpness: Option<f32>,

    /// Enable the temporal (history aware) upscale path.
    #[arg(long)]
    pub temporal: bool,

    /// Disable the temporal upscale path.
    #[arg(long)]
    pub no_temporal: bool,

    /// Present backend: auto | window | dxgi (dxgi is a flip-model swap chain: lower latency, vsync and tearing support).
    #[arg(long, value_name = "PRESENT_BACKEND")]
    pub present_backend: Option<String>,

    /// Present with vsync (lowest tearing, slightly higher latency).
    #[arg(long)]
    pub vsync: bool,

    /// Present without vsync (lowest latency).
    #[arg(long)]
    pub no_vsync: bool,

    /// Open the preview window fullscreen.
    #[arg(long)]
    pub fullscreen: bool,

    /// Open a borderless preview window.
    #[arg(long)]
    pub borderless: bool,

    /// Maximum in-flight frames (1 = lowest latency).
    #[arg(long, value_name = "N")]
    pub max_queue: Option<usize>,

    /// Soft cap on the output framerate in Hz.
    #[arg(long, value_name = "HZ")]
    pub fps_cap: Option<f32>,
}

impl Overrides {
    /// True when no override flag was supplied.
    pub fn is_empty(&self) -> bool {
        self.preset.is_none()
            && self.interp.is_none()
            && self.multiplier.is_none()
            && self.scale.is_none()
            && self.target.is_none()
            && self.filter.is_none()
            && self.backend.is_none()
            && self.interp_backend.is_none()
            && self.present_backend.is_none()
            && self.sharpness.is_none()
            && !self.temporal
            && !self.no_temporal
            && !self.vsync
            && !self.no_vsync
            && !self.fullscreen
            && !self.borderless
            && self.max_queue.is_none()
            && self.fps_cap.is_none()
    }

    /// Apply these overrides to a configuration, then validate it.
    pub fn apply(&self, cfg: &mut FluidConfig) -> ParseResult<()> {
        if let Some(preset) = self.preset.as_deref() {
            let preset: QualityPreset = preset.parse().map_err(|e: FluidError| e.to_string())?;
            cfg.apply_preset(preset);
        }
        if let Some(mode) = self.interp.as_deref() {
            cfg.interpolator.mode = mode.parse().map_err(|e: FluidError| e.to_string())?;
        }
        if let Some(mult) = self.multiplier {
            cfg.interpolator.multiplier = mult;
            if mult > 1 && cfg.interpolator.mode == InterpMode::Off {
                cfg.interpolator.mode = InterpMode::Fixed;
            }
            if mult <= 1 {
                cfg.interpolator.mode = InterpMode::Off;
            }
        }
        if let Some(filter) = self.filter.as_deref() {
            cfg.upscaler.filter = filter.parse().map_err(|e: FluidError| e.to_string())?;
        }
        if let Some(scale) = self.scale {
            if scale <= 0.0 {
                cfg.upscaler.scale.mode = ScaleMode::Native;
                cfg.upscaler.filter = UpscaleFilter::Off;
            } else {
                cfg.upscaler.scale.mode = ScaleMode::Factor;
                cfg.upscaler.scale.factor = scale;
            }
        }
        if let Some(target) = self.target {
            cfg.upscaler.scale.mode = ScaleMode::Target;
            cfg.upscaler.scale.target_w = target.width;
            cfg.upscaler.scale.target_h = target.height;
            if cfg.upscaler.filter == UpscaleFilter::Off {
                cfg.upscaler.filter = UpscaleFilter::Adaptive;
            }
        }
        if let Some(backend) = self.backend.as_deref() {
            cfg.upscaler.backend = backend.parse().map_err(|e: FluidError| e.to_string())?;
        }
        if let Some(backend) = self.interp_backend.as_deref() {
            cfg.interpolator.backend = backend.parse().map_err(|e: FluidError| e.to_string())?;
        }
        if let Some(backend) = self.present_backend.as_deref() {
            cfg.present.backend = backend.parse().map_err(|e: FluidError| e.to_string())?;
        }
        if let Some(sharp) = self.sharpness {
            cfg.upscaler.sharpness = sharp;
        }
        if self.temporal {
            cfg.upscaler.temporal.enabled = true;
        }
        if self.no_temporal {
            cfg.upscaler.temporal.enabled = false;
        }
        if self.vsync {
            cfg.present.vsync = true;
        }
        if self.no_vsync {
            cfg.present.vsync = false;
        }
        if self.fullscreen {
            cfg.present.fullscreen = true;
        }
        if self.borderless {
            cfg.present.borderless = true;
        }
        if let Some(max) = self.max_queue {
            cfg.latency.max_queue = max;
        }
        if let Some(fps) = self.fps_cap {
            cfg.latency.target_fps = fps;
        }
        cfg.validate().map_err(|e| e.to_string())
    }
}

/// Load the base config from disk (or defaults) then apply CLI overrides.
pub fn prepare_config(cli: &Cli, over: &Overrides) -> ParseResult<FluidConfig> {
    let mut cfg = match cli.config.as_ref() {
        Some(path) => {
            FluidConfig::load(path).map_err(|e| format!("cannot load {}: {e}", path.display()))?
        }
        None => FluidConfig::default(),
    };
    let level = if cli.quiet {
        "warn"
    } else {
        cli.log_level.trim()
    };
    cfg.logging.level = level.to_string();
    over.apply(&mut cfg)?;
    Ok(cfg)
}

/// Parse WxH into dimensions.
pub fn parse_dims(s: &str) -> ParseResult<Dims> {
    let text = s.trim().to_ascii_lowercase().replace('*', "x");
    let (w, h) = text
        .split_once('x')
        .ok_or_else(|| format!("expected WxH (for example 1920x1080), got '{s}'"))?;
    let w: u32 = w
        .trim()
        .parse()
        .map_err(|_| format!("bad width in '{s}'"))?;
    let h: u32 = h
        .trim()
        .parse()
        .map_err(|_| format!("bad height in '{s}'"))?;
    if w == 0 || h == 0 {
        return Err(format!("dimensions must be non-zero, got '{s}'"));
    }
    Ok(Dims::new(w, h))
}

/// Parse a WxH+X+Y capture region.
pub fn parse_region(s: &str) -> ParseResult<RectConfig> {
    let cleaned = s.trim().replace(',', "+").replace('*', "x");
    let (size, rest) = cleaned
        .split_once('x')
        .ok_or_else(|| format!("expected WxH+X+Y (for example 1280x720+100+50), got '{s}'"))?;
    let nums: Vec<u32> = std::iter::once(size)
        .chain(rest.split('+').filter(|p| !p.trim().is_empty()))
        .map(|p| {
            p.trim()
                .parse::<u32>()
                .map_err(|_| format!("bad number '{p}' in '{s}'"))
        })
        .collect::<ParseResult<Vec<u32>>>()?;
    if nums.len() != 4 {
        return Err(format!(
            "expected WxH+X+Y (for example 1280x720+100+50), got '{s}'"
        ));
    }
    if nums[0] == 0 || nums[1] == 0 {
        return Err(format!(
            "region width and height must be non-zero, got '{s}'"
        ));
    }
    let (w, h, x, y) = (nums[0], nums[1], nums[2], nums[3]);
    Ok(RectConfig { x, y, w, h })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_dims() {
        assert_eq!(parse_dims("1920x1080").unwrap(), Dims::new(1920, 1080));
        assert_eq!(parse_dims(" 640 X 480 ").unwrap(), Dims::new(640, 480));
        assert!(parse_dims("1920").is_err());
        assert!(parse_dims("0x0").is_err());
    }

    #[test]
    fn parses_region() {
        let r = parse_region("1280x720+100+50").unwrap();
        assert_eq!((r.w, r.h, r.x, r.y), (1280, 720, 100, 50));
        assert!(parse_region("1280x720").is_err());
    }

    #[test]
    fn overrides_apply() {
        let mut cfg = FluidConfig::default();
        let over = Overrides {
            preset: Some("performance".to_string()),
            multiplier: Some(3),
            scale: Some(2.0),
            filter: Some("lanczos".to_string()),
            ..Default::default()
        };
        over.apply(&mut cfg).unwrap();
        assert_eq!(cfg.profile, "performance");
        assert_eq!(cfg.interpolator.multiplier, 3);
        assert_eq!(cfg.upscaler.scale.factor, 2.0);
        assert_eq!(cfg.upscaler.filter, UpscaleFilter::Lanczos);
    }

    #[test]
    fn interp_backend_override_applies() {
        let mut cfg = FluidConfig::default();
        let over = Overrides {
            interp_backend: Some("gpu".to_string()),
            ..Default::default()
        };
        over.apply(&mut cfg).unwrap();
        assert_eq!(cfg.interpolator.backend, fluid_core::InterpBackendKind::Gpu);
        let bad = Overrides {
            interp_backend: Some("d3d11".to_string()),
            ..Default::default()
        };
        assert!(bad.apply(&mut FluidConfig::default()).is_err());
    }

    #[test]
    fn backend_override_applies() {
        let mut cfg = FluidConfig::default();
        let over = Overrides {
            backend: Some("gpu".to_string()),
            ..Default::default()
        };
        over.apply(&mut cfg).unwrap();
        assert_eq!(cfg.upscaler.backend, fluid_core::UpscaleBackendKind::Gpu);
        let bad = Overrides {
            backend: Some("vulkan".to_string()),
            ..Default::default()
        };
        assert!(bad.apply(&mut FluidConfig::default()).is_err());
    }

    #[test]
    fn present_backend_override_applies() {
        let mut cfg = FluidConfig::default();
        let over = Overrides {
            present_backend: Some("dxgi".to_string()),
            ..Default::default()
        };
        over.apply(&mut cfg).unwrap();
        assert_eq!(cfg.present.backend, fluid_core::PresentBackendKind::Dxgi);
        let bad = Overrides {
            present_backend: Some("directx".to_string()),
            ..Default::default()
        };
        assert!(bad.apply(&mut FluidConfig::default()).is_err());
    }
    #[test]
    fn cli_accepts_bench_defaults() {
        let cli = Cli::parse_from(["fluid", "bench", "--frames", "3"]);
        match cli.command {
            Some(Command::Bench(args)) => {
                assert_eq!(args.size, Dims::new(1920, 1080));
                assert_eq!(args.frames, 3);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn multiplier_one_disables_generation() {
        let mut cfg = FluidConfig::default();
        cfg.interpolator.mode = InterpMode::Fixed;
        let over = Overrides {
            multiplier: Some(1),
            ..Default::default()
        };
        over.apply(&mut cfg).unwrap();
        assert_eq!(cfg.interpolator.mode, InterpMode::Off);
    }
}
