//! GPU compute upscaling for Fluid, built on [wgpu].
//!
//! This backend mirrors the CPU pipeline in 'fluid-upscale' exactly - the same
//! pixel-centre sampling, the same windowed kernels, the same adaptive detail
//! restore, the same integer sharpening and the same fixed-point temporal blend -
//! but evaluates the resampling on the GPU with wgpu compute shaders. wgpu picks
//! the native backend at runtime (Vulkan, D3D12 or GLES on Windows, Vulkan or
//! GLES on Android), so the exact same code runs on a desktop and on a phone.
//!
//! # Quality
//!
//! Every spatial filter is bit-exact against the CPU reference for nearest
//! neighbour and for identity resizes, and within one least-significant bit per
//! channel for the windowed filters (bilinear, bicubic, Lanczos-3 and adaptive),
//! where the CPU's SIMD path fuses a multiply-add that the GPU evaluates as a
//! separate multiply and add.
//!
//! The anti-ringing clamp of the vertical pass and the detail-limited sharpen
//! reproduce the current CPU reference as well, so on the reference machine the
//! GPU output is byte-identical to the CPU output for every filter, not merely
//! within the one-LSB contract. Sharpening and the temporal blend use the same
//! integer arithmetic on both sides and are exact by construction.
//!
//! The temporal blend is the CPU's motion-adaptive variant, not a global EMA: a
//! small kernel measures the mean absolute luma difference of the current resize
//! against the history for every 16 x 16 destination tile, runs the same f32
//! smoothstep ramp, and rounds the result to the same twelve-bit weight. The
//! blend kernel then applies that weight per tile, so a fast-moving tile follows
//! the current frame (no ghost) while a static tile keeps exactly the configured
//! blend (dither is still smoothed). The weight buffer is a few kilobytes even at
//! 4K and is created once, so this stays allocation-free in steady state and
//! byte-identical to 'fluid-upscale'.
//!
//! # Usage
//!
//! ```no_run
//! use std::sync::Arc;
//! use fluid_core::{Dims, Frame, PixelFormat, UpscaleFilter};
//! use fluid_gpu::{GpuContext, GpuUpscaler};
//!
//! # fn main() -> fluid_core::Result<()> {
//! let ctx = Arc::new(GpuContext::new()?);
//! println!("{} on {}", ctx.info().name, ctx.info().backend);
//!
//! let src = Dims::new(1280, 720);
//! let dst = Dims::new(2560, 1440);
//! let mut upscaler = GpuUpscaler::new(
//!     Arc::clone(&ctx),
//!     UpscaleFilter::Lanczos,
//!     src,
//!     dst,
//!     0.2,
//!     None,
//! )?;
//!
//! let frame = Frame::zeroed(src, PixelFormat::Rgba8);
//! let mut out = Frame::zeroed(dst, PixelFormat::Rgba8);
//! upscaler.upscale(&frame, &mut out)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Performance
//!
//! All GPU resources - input, intermediate, ping-pong output, history, staging
//! and tap tables - are created once in [GpuUpscaler::new]. A frame only writes
//! the source bytes into the existing upload buffer, records the compute passes
//! and reads the result back, so steady state performs no allocation beyond the
//! command encoder every graphics API requires. Every dispatch is followed by a
//! synchronous readback, which makes [UpscaleBackend::last_cost_ns] an honest
//! wall-clock measurement and avoids unbounded queue growth.
//!
//! # Unsupported systems
//!
//! [probe] returns [FluidError::Unsupported] when no adapter is present, and so
//! do [GpuContext::new] and [GpuUpscaler::new]. Nothing panics when the machine
//! has no usable GPU: callers can fall back to the CPU upscaler. A failed
//! command is reported as [FluidError::Internal] from [GpuUpscaler::upscale]
//! instead of aborting the process, so a driver bug degrades to the CPU backend.
//!
//! # Backend selection
//!
//! [GpuContext] enumerates every wgpu adapter and ranks D3D12 first, then Vulkan
//! and finally OpenGL ES, so a Windows desktop runs on the native D3D12 driver
//! and an Android phone on Vulkan or GLES without a code change. DX12 and the
//! older `fluid-capture` path need different versions of the `windows` crate;
//! the workspace keeps both buildable with a `gpu-allocator` patch in the root
//! `Cargo.toml` (`[patch.crates-io] gpu-allocator = { path = "vendor/gpu-allocator" }`).
//! If the preferred adapter cannot be created the next one is tried, and if none
//! works the constructors return [FluidError::Unsupported].
//!
//! # GPU frame interpolation
//!
//! [GpuInterpolator] keeps the whole frame-interpolation pipeline on the GPU. It
//! derives luma planes and a scene-change metric from the two input frames,
//! estimates dense forward and backward optical flow with [GpuFlow], and
//! synthesises each intermediate frame with motion-compensated sampling in a
//! compute shader. [GpuFlow] is the estimator on its own: it returns the same
//! [fluid_flow::FlowEstimate] (dense gather fields plus per-pixel confidence)
//! as the CPU estimator.
//!
//! Precision strategy: every kernel works in `f32` on unpacked bytes, so the
//! synthesis arithmetic is the same as 'fluid-interp's and only the fused vs
//! unfused multiply-add order can differ. All intermediate buffers, pipelines and
//! bind groups are created once in [GpuInterpolator::new]; generating frames only
//! uploads the two frames, submits one command buffer for the flow field and one
//! per intermediate frame, and allocates nothing but the returned [Frame]s.
//!
//! Limitations:
//!
//! * Scene changes are detected from the mean absolute luma difference only, and
//!   reported only when it exceeds twice the configured threshold. This is
//!   conservative: every reported cut is also a cut the CPU detector reports, so
//!   a scene change never produces a smeared blend.
//! * The CPU estimator halves the matching resolution on large frames for speed;
//!   [GpuFlow] follows the same 12k-block policy (and the same
//!   `FLUID_FULL_RES_FLOW` escape hatch), so both backends produce the same field
//!   for a given frame size.
//! * [GpuInterpolator::generate] takes one flow estimate for the pair, exactly
//!   like 'fluid_interp::Interpolator::generate'; it is not a long-running
//!   temporal interpolator with its own history.
//!
//! Reproduce the comparison benchmark with:
//!
//! ```text
//! cargo run --release -p fluid-gpu --example bench_interp -- --json out/gpu-interp-bench.json
//! ```
//!
//! # Fused interpolation + upscaling
//!
//! [GpuPipeline] fuses the two stages above into one GPU pipeline: it estimates
//! the flow of a frame pair, synthesises each intermediate frame at the source
//! resolution, and then runs the [GpuUpscaler] spatial filter, adaptive
//! refinement, sharpening and temporal blend directly on the synthesised pixels.
//! Only the finished destination-resolution frames come back to the CPU, so the
//! intermediate source-resolution readback of the two-step chain disappears.
//!
//! ```no_run
//! # fn main() -> fluid_core::Result<()> {
//! use std::sync::Arc;
//! use fluid_core::{Dims, Frame, InterpolatorConfig, PixelFormat, UpscaleFilter};
//! use fluid_gpu::{GpuContext, GpuPipeline};
//!
//! let ctx = Arc::new(GpuContext::new()?);
//! let mut pipeline = GpuPipeline::new(
//!     Arc::clone(&ctx),
//!     Dims::new(1280, 720),
//!     Dims::new(2560, 1440),
//!     UpscaleFilter::Adaptive,
//!     0.2,
//!     None,
//!     &InterpolatorConfig::default(),
//! )?;
//! let src = Dims::new(1280, 720);
//! let prev = Frame::zeroed(src, PixelFormat::Rgba8);
//! let next = Frame::zeroed(src, PixelFormat::Rgba8);
//! let frames = pipeline.generate(&prev, &next)?;
//! assert_eq!(frames.len(), 1);
//! # Ok(())
//! # }
//! ```
//!
//! Precision strategy: the synthesis kernels use the same `f32` arithmetic as
//! 'fluid-interp', and the upscaling stage is the identical shader set
//! [GpuUpscaler] uses, including the anti-ringing clamp and the detail-limited
//! sharpen. On the reference machine the fused output is byte-identical to the
//! two-step GPU chain; [GpuPipeline::backend_name] reports `"wgpu-fused"`.
//!
//! The saving is bounded by transfer cost. On the integrated GPU tested here the
//! source-resolution writeback is only a small part of each frame, while the
//! destination-sized upscale passes, the sharpen and the destination readback
//! dominate, so the fused pipeline measures 0.96x the wall time of the separate
//! chain overall (0.82x-1.00x per case, with the differences within the noise of
//! a shared machine: the fused path is usually a few percent slower, not a
//! multiple). A discrete GPU with its own memory would gain more, because the
//! writeback it removes is more expensive there. Both chains are byte-identical.
//!
//! Reproduce with:
//!
//! ```text
//! cargo run --release -p fluid-gpu --example bench_fused -- --json out/gpu-fused-bench.json
//! ```

#![forbid(unsafe_code)]

mod convert;
mod device;
mod flow;
mod fused;
mod interp;
mod pipeline;
mod synth;
mod taps;

pub use crate::flow::GpuFlow;
pub use crate::fused::GpuPipeline;
pub use crate::interp::GpuInterpolator;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use fluid_core::{Dims, FluidError, Frame, PixelFormat, Result, TemporalConfig, UpscaleFilter};
use fluid_upscale::UpscaleBackend;

use crate::pipeline::{Config, Spatial, TemporalMode};
use crate::taps::Kernel;

/// Name this backend reports through [UpscaleBackend::backend_name].
///
/// The concrete graphics API is available in [GpuInfo::backend], since wgpu
/// selects Vulkan, D3D12 or GLES at runtime.
pub const BACKEND_NAME: &str = "wgpu";

/// Description of the compute device this backend selected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuInfo {
    /// Adapter name, for example 'Intel(R) Iris(R) Xe Graphics'.
    pub name: String,
    /// Graphics API wgpu selected: 'vulkan', 'dx12', 'gl', 'metal' or 'webgpu'.
    pub backend: &'static str,
    /// PCI vendor id, or 0 when the adapter does not report one.
    pub vendor_id: u32,
    /// Human-readable summary of the compute limits that matter to us.
    pub max_feature_level: String,
    /// Adapter class wgpu reports: 'discrete', 'integrated', 'virtual', 'cpu'
    /// or 'other'. The engine uses it to decide whether moving frame
    /// interpolation to the GPU pays off at the current frame size.
    pub device_type: &'static str,
}

impl std::fmt::Display for GpuInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({}, {})", self.name, self.device_type, self.backend)
    }
}

/// Probe for a usable compute adapter without creating a device.
///
/// This is the cheap availability check: when it returns
/// [FluidError::Unsupported] there is no adapter at all, so callers should use
/// the CPU upscaler instead of trying to create a [GpuContext].
pub fn probe() -> Result<GpuInfo> {
    device::probe_info()
}

/// A compute device, shared by every [GpuUpscaler] built from it.
///
/// Constructing the device is the expensive step (driver initialisation), so an
/// application creates one context and clones the [Arc] into each upscaler.
/// Passes from different upscalers are serialised through the context, which
/// keeps the command stream simple and makes frame timing honest.
pub struct GpuContext {
    info: GpuInfo,
    shared: Mutex<Shared>,
}

/// The wgpu objects every upscaler needs; guarded so passes serialise.
pub(crate) struct Shared {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    /// Last uncaptured wgpu validation error, stored by the device handler.
    pub(crate) errors: device::ErrorSlot,
}

impl Shared {
    /// Take the last recorded wgpu error, if a command failed validation.
    pub(crate) fn take_error(&self) -> Option<String> {
        self.errors
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }
}

impl GpuContext {
    /// Create the compute device, or report [FluidError::Unsupported] when the
    /// machine has no adapter that can run compute shaders.
    pub fn new() -> Result<Self> {
        let (device, queue, info, errors) = device::create_device()?;
        Ok(Self {
            info,
            shared: Mutex::new(Shared {
                device,
                queue,
                errors,
            }),
        })
    }

    /// Description of the device that was created.
    pub fn info(&self) -> &GpuInfo {
        &self.info
    }

    /// The concrete graphics API in use ('vulkan', 'dx12', ...).
    pub fn backend(&self) -> &'static str {
        self.info.backend
    }

    /// Adapter class: 'discrete', 'integrated', 'virtual', 'cpu' or 'other'.
    pub fn device_type(&self) -> &'static str {
        self.info.device_type
    }

    pub(crate) fn lock(&self) -> Result<std::sync::MutexGuard<'_, Shared>> {
        self.shared
            .lock()
            .map_err(|_| FluidError::internal("gpu: the device lock was poisoned"))
    }
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

/// GPU implementation of the spatial upscaling stage.
///
/// One upscaler owns every buffer it needs for a fixed source and destination
/// size, so create it once per resolution and call [GpuUpscaler::upscale] for
/// every frame. The filter, sharpness and temporal configuration are frozen at
/// construction, exactly like the CPU 'Upscaler'.
pub struct GpuUpscaler {
    ctx: Arc<GpuContext>,
    filter: UpscaleFilter,
    src: Dims,
    dst: Dims,
    sharpness: f32,
    temporal: Option<TemporalConfig>,
    /// None only for the Off identity case, which is a CPU pass-through.
    pipeline: Option<pipeline::Pipeline>,
    /// Canonical RGBA8 source texels, reused every frame.
    packed: Vec<u32>,
    /// Format of the previous source frame, so a format change restarts the
    /// temporal history exactly like the CPU backend's history format check.
    last_fmt: Option<PixelFormat>,
    last_cost_ns: u64,
}

impl GpuUpscaler {
    /// Build the upscaler and upload its immutable state to the GPU.
    ///
    /// 'sharpness' is ignored for [UpscaleFilter::Off], matching the CPU
    /// pipeline where Off means "do nothing". 'temporal' is only consulted for
    /// [UpscaleFilter::Temporal]; None uses [TemporalConfig::default].
    ///
    /// Returns [FluidError::Unsupported] when the adapter cannot hold the
    /// buffers, and [FluidError::InvalidArgument] for empty dimensions.
    pub fn new(
        ctx: Arc<GpuContext>,
        filter: UpscaleFilter,
        src: Dims,
        dst: Dims,
        sharpness: f32,
        temporal: Option<TemporalConfig>,
    ) -> Result<Self> {
        if src.is_empty() {
            return Err(FluidError::invalid(format!(
                "gpu upscale: source size {src} is empty"
            )));
        }
        if dst.is_empty() {
            return Err(FluidError::invalid(format!(
                "upscale: destination size {dst} is empty"
            )));
        }

        let (spatial, temporal_mode) = spatial_config(filter, src, dst, temporal);
        let sharpen_q8 = sharpen_strength_q8(filter, sharpness);

        // An identity resize with no sharpening and no temporal blend is an
        // exact byte copy on the CPU, so sending it through the GPU would only
        // add upload plus readback latency; copy it directly, exactly like
        // fluid-upscale's pass-through. Off always takes this path (it never
        // sharpens) and every other filter does when src and dst are equal.
        let pass_through = src == dst && sharpen_q8 == 0 && temporal_mode.is_none();
        let temporal_config = temporal.unwrap_or_default();
        let config = Config {
            src,
            dst,
            spatial,
            sharpen_q8,
            temporal: temporal_mode,
            blend_s: temporal_blend_weight(&temporal_config),
        };

        let pipeline = if pass_through {
            None
        } else {
            let shared = ctx.lock()?;
            Some(pipeline::Pipeline::new(
                &shared.device,
                &shared.queue,
                config,
            )?)
        };

        let packed = if pass_through {
            Vec::new()
        } else {
            vec![0u32; src.area() as usize]
        };

        Ok(Self {
            ctx,
            filter,
            src,
            dst,
            sharpness,
            temporal,
            pipeline,
            packed,
            last_fmt: None,
            last_cost_ns: 0,
        })
    }

    /// The filter this upscaler was built for.
    pub fn filter(&self) -> UpscaleFilter {
        self.filter
    }

    /// The sharpening amount this upscaler was built with.
    pub fn sharpness(&self) -> f32 {
        self.sharpness
    }

    /// Drop the temporal history, for a scene cut or a stream discontinuity.
    pub fn reset(&mut self) {
        if let Some(pipeline) = self.pipeline.as_mut() {
            pipeline.reset_history();
        }
    }

    /// Convert one frame into 'out', reusing every GPU resource.
    ///
    /// 'src' must have the source dimensions this upscaler was built with (any
    /// [PixelFormat]); 'out' must have the destination dimensions and 'src.fmt'.
    /// Frame metadata (pts and index) is copied across, the same contract the
    /// CPU backend follows.
    pub fn upscale(&mut self, src: &Frame, out: &mut Frame) -> Result<()> {
        let started = Instant::now();
        if src.dims != self.src {
            return Err(FluidError::invalid(format!(
                "gpu upscale: source frame is {}, expected {}",
                src.dims, self.src
            )));
        }
        check_out(self.dst, src.fmt, out)?;

        // fluid-upscale's history is only valid for the same frame format; a
        // format change there discards history, so do the same here.
        if self.last_fmt != Some(src.fmt) {
            if let Some(pipeline) = self.pipeline.as_mut() {
                pipeline.reset_history();
            }
            self.last_fmt = Some(src.fmt);
        }

        if self.pipeline.is_some() {
            self.fill_packed(src)?;
            let shared = self.ctx.lock()?;
            let pipeline = self.pipeline.as_mut().expect("checked above");
            pipeline.execute(&shared.device, &shared.queue, &self.packed, out)?;
            // wgpu reports dispatch-time validation errors through the device
            // handler instead of panicking; surface them as a normal error so the
            // engine can fall back to the CPU upscaler.
            if let Some(error) = shared.take_error() {
                return Err(FluidError::internal(format!(
                    "gpu upscale: wgpu error: {error}"
                )));
            }
        } else {
            // Identity resize with nothing else to do: byte copy, like the
            // CPU fast path.
            out.data.copy_from_slice(&src.data);
        }

        out.pts_ns = src.pts_ns;
        out.index = src.index;
        self.last_cost_ns = started.elapsed().as_nanos() as u64;
        Ok(())
    }

    /// Wall-clock nanoseconds the last [GpuUpscaler::upscale] call took.
    pub fn last_cost_ns(&self) -> u64 {
        self.last_cost_ns
    }

    /// Expand one source frame into the persistent canonical RGBA8 buffer.
    ///
    /// This is the only per-frame CPU work besides the readback conversion; the
    /// buffer was allocated in [GpuUpscaler::new] and is overwritten in place.
    fn fill_packed(&mut self, src: &Frame) -> Result<()> {
        let width = self.src.width as usize;
        let needed = self.src.area() as usize;
        if self.packed.len() != needed {
            return Err(FluidError::internal(
                "gpu upscale: the canonical source buffer has the wrong size",
            ));
        }
        let fmt = src.fmt;
        for y in 0..self.src.height as usize {
            let source = src.row(y as u32);
            let target = &mut self.packed[y * width..(y + 1) * width];
            convert::expand_row(source, fmt, bytemuck::cast_slice_mut(target));
        }
        Ok(())
    }
}

impl std::fmt::Debug for GpuUpscaler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuUpscaler")
            .field("filter", &self.filter)
            .field("src", &self.src)
            .field("dst", &self.dst)
            .field("sharpness", &self.sharpness)
            .field("temporal", &self.temporal)
            .field("device", &self.ctx.info)
            .finish_non_exhaustive()
    }
}

impl UpscaleBackend for GpuUpscaler {
    fn backend_name(&self) -> &'static str {
        BACKEND_NAME
    }

    fn src_dims(&self) -> Dims {
        self.src
    }

    fn dst_dims(&self) -> Dims {
        self.dst
    }

    fn upscale(&mut self, src: &Frame, out: &mut Frame) -> Result<()> {
        GpuUpscaler::upscale(self, src, out)
    }

    fn reset(&mut self) {
        GpuUpscaler::reset(self);
    }

    fn last_cost_ns(&self) -> u64 {
        self.last_cost_ns
    }
}

/// Map the public filter to the GPU pass configuration.
///
/// This mirrors 'fluid-upscale' mapping in resample.rs: Off falls back to
/// bilinear (its identity case never reaches the GPU), Temporal and Adaptive
/// share the cubic base with the detail restore, and an identity resize is an
/// exact copy for every filter.
pub(crate) fn spatial_config(
    filter: UpscaleFilter,
    src: Dims,
    dst: Dims,
    temporal: Option<TemporalConfig>,
) -> (Spatial, Option<TemporalMode>) {
    let (kernel, refine) = match filter {
        UpscaleFilter::Off => (Kernel::Bilinear, false),
        UpscaleFilter::Nearest => (Kernel::Nearest, false),
        UpscaleFilter::Bilinear => (Kernel::Bilinear, false),
        UpscaleFilter::Bicubic => (Kernel::Bicubic, false),
        UpscaleFilter::Lanczos => (Kernel::Lanczos3, false),
        UpscaleFilter::Adaptive | UpscaleFilter::Temporal => (Kernel::Bicubic, true),
    };

    let mode = if filter == UpscaleFilter::Temporal {
        let config = temporal.unwrap_or_default();
        if config.enabled {
            Some(temporal_mode(&config))
        } else {
            None
        }
    } else {
        None
    };

    let spatial = if src == dst {
        // Identity resize: the CPU returns an exact copy for every filter.
        Spatial::Identity
    } else if kernel == Kernel::Nearest {
        Spatial::Nearest
    } else {
        Spatial::Separable { kernel, refine }
    };
    (spatial, mode)
}

/// Which temporal stage the blend factor implies.
fn temporal_mode(config: &TemporalConfig) -> TemporalMode {
    let alpha = temporal_alpha(config);
    if alpha >= 1.0 {
        TemporalMode::Identity
    } else if temporal_blend_weight(config) <= 0 {
        TemporalMode::Freeze
    } else {
        TemporalMode::Normal
    }
}

/// Per-frame temporal blend factor, spread over the configured horizon.
fn temporal_alpha(config: &TemporalConfig) -> f32 {
    let frames = config.history_frames.clamp(1, 8) as f32;
    if config.blend.is_finite() {
        (config.blend * 2.0 / frames).clamp(0.0, 1.0)
    } else {
        0.12
    }
}

/// Twelve-bit fixed point blend weight, the same scale the CPU uses.
pub(crate) fn temporal_blend_weight(config: &TemporalConfig) -> i32 {
    let alpha = temporal_alpha(config);
    if alpha >= 1.0 {
        // The CPU short-circuits a full-current blend before computing s.
        4096
    } else {
        (alpha * 4096.0 + 0.5) as i32
    }
}

/// Quantised sharpening strength, zero when sharpening must not run.
///
/// Matches 'fluid-upscale': Off never sharpens, and the amount is clamped to
/// 0..=1 and quantised to 8 bits of a 256-scale weight.
pub(crate) fn sharpen_strength_q8(filter: UpscaleFilter, sharpness: f32) -> i32 {
    if filter == UpscaleFilter::Off || sharpness.is_nan() || sharpness <= 0.0 {
        return 0;
    }
    (sharpness.clamp(0.0, 1.0) * 256.0 + 0.5) as i32
}

/// Validate the caller-owned output frame, with the CPU backend's messages.
fn check_out(dst: Dims, fmt: PixelFormat, out: &Frame) -> Result<()> {
    if out.dims != dst || out.fmt != fmt {
        return Err(FluidError::invalid(format!(
            "upscale: output frame is {} {}, expected {} {}",
            out.dims, out.fmt, dst, fmt
        )));
    }
    if out.data.len() != fmt.buffer_len(dst) {
        return Err(FluidError::invalid(
            "upscale: output frame buffer length does not match its dimensions",
        ));
    }
    Ok(())
}
