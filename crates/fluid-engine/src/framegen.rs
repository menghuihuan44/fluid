//! Pluggable frame generation backends.
//!
//! The CPU backend runs the first-party optical flow estimator plus the motion
//! compensated synthesiser. The GPU backend (wgpu compute) is used when a device
//! is available and the configuration asks for it; both are selected here so the
//! rest of the processor does not care which one runs.

use std::sync::Arc;
use std::time::Instant;

use fluid_core::{
    Dims, FluidConfig, FluidError, Frame, InterpBackendKind, InterpolatorConfig, LumaPlane, Result,
    UpscaleFilter,
};
use fluid_flow::{FlowParams, estimate};
use fluid_gpu::{GpuContext, GpuInterpolator, GpuPipeline};
use fluid_interp::{InterpMetrics, Interpolator};

/// Frames produced for one pair of source frames, plus telemetry.
pub struct Generated {
    /// Synthesised frames, in presentation order (alpha = 1/m .. (m-1)/m).
    pub frames: Vec<Frame>,
    /// Aggregated synthesis quality metrics.
    pub metrics: InterpMetrics,
    /// Time spent estimating optical flow, in nanoseconds.
    pub flow_ns: u64,
    /// Time spent synthesising intermediate frames, in nanoseconds.
    pub synth_ns: u64,
}

/// A frame generation implementation.
pub trait FrameGenBackend: Send {
    /// Short stable name reported in the pipeline report ("cpu", "wgpu-flow").
    fn backend_name(&self) -> &'static str;

    /// Frames produced per source frame (1 = disabled).
    fn multiplier(&self) -> u8;

    /// Generate the intermediate frames between `prev` and `next`.
    ///
    /// The luma planes are provided by the caller because scene-cut detection
    /// needs them anyway; backends that work on colour frames can ignore them.
    fn generate(
        &mut self,
        prev: &Frame,
        next: &Frame,
        luma_prev: &LumaPlane,
        luma_next: &LumaPlane,
    ) -> Result<Generated>;

    /// Drop temporal state (scene cut, resolution change, seek).
    fn reset(&mut self);

    /// Soft time budget for one `generate` call, in nanoseconds.
    ///
    /// Only used by `interpolator.mode = "adaptive"`: the processor tells the
    /// backend how much time is left before the next source frame is due, and
    /// a backend that can stop early generates fewer intermediates instead of
    /// falling behind. `None` restores the full configured multiplier.
    fn set_budget_ns(&mut self, _budget_ns: Option<u64>) {}

    /// Cost of the last `generate` call in nanoseconds, 0 when not measured.
    fn last_cost_ns(&self) -> u64 {
        0
    }
}

/// CPU backend: pyramid block matching + LK refinement, then synthesis.
pub struct CpuFrameGen {
    interp: Interpolator,
    flow: FlowParams,
    multiplier: u8,
    budget_ns: Option<u64>,
}

impl CpuFrameGen {
    pub fn new(cfg: &InterpolatorConfig) -> Self {
        Self {
            interp: Interpolator::new(cfg),
            flow: FlowParams::from(&cfg.flow),
            multiplier: cfg.multiplier.max(1),
            budget_ns: None,
        }
    }
}

impl FrameGenBackend for CpuFrameGen {
    fn backend_name(&self) -> &'static str {
        "cpu"
    }

    fn multiplier(&self) -> u8 {
        self.multiplier
    }

    fn generate(
        &mut self,
        prev: &Frame,
        next: &Frame,
        luma_prev: &LumaPlane,
        luma_next: &LumaPlane,
    ) -> Result<Generated> {
        let ft = Instant::now();
        let flow = estimate(luma_prev, luma_next, &self.flow)?;
        let flow_ns = ft.elapsed().as_nanos() as u64;
        let m = u32::from(self.multiplier.max(1));
        let mut frames = Vec::with_capacity(m.saturating_sub(1) as usize);
        let mut metrics = InterpMetrics::default();
        let t = Instant::now();
        for k in 1..m {
            // Adaptive mode: stop generating as soon as the remaining time
            // budget for this source frame is used up. The output then simply
            // falls back to the real frame only, so the run keeps up instead
            // of accumulating latency.
            if let Some(budget) = self.budget_ns
                && t.elapsed().as_nanos() as u64 >= budget
            {
                break;
            }
            let alpha = k as f32 / m as f32;
            let (mid, m2) = self.interp.interpolate(prev, next, alpha, &flow)?;
            metrics.frames += 1;
            metrics.blended_px += m2.blended_px;
            metrics.occluded_px += m2.occluded_px;
            metrics.mean_confidence += m2.mean_confidence;
            metrics.mean_flow_px += m2.mean_flow_px;
            frames.push(mid);
        }
        let synth_ns = t.elapsed().as_nanos() as u64;
        if metrics.frames > 0 {
            let n = metrics.frames as f32;
            metrics.mean_confidence /= n;
            metrics.mean_flow_px /= n;
        }
        Ok(Generated {
            frames,
            metrics,
            flow_ns,
            synth_ns,
        })
    }

    fn reset(&mut self) {}

    fn set_budget_ns(&mut self, budget_ns: Option<u64>) {
        self.budget_ns = budget_ns;
    }
}

/// GPU backend: optical flow and synthesis both run as wgpu compute passes.
///
/// One [`GpuInterpolator`] owns every buffer it needs, so per-pair work is only
/// the two frame uploads and the readback of the generated frames.
pub struct GpuFrameGen {
    interp: GpuInterpolator,
    multiplier: u8,
    /// Adaptive budget for the next generate call, if any.
    budget_ns: Option<u64>,
    /// Duration of the last GPU generate, used to decide the next budget.
    last_generate_ns: u64,
}

impl GpuFrameGen {
    /// Create the GPU backend for one frame size.
    ///
    /// Errors (no adapter, unsupported limits) are for the caller to turn into a
    /// fallback; the engine treats a GPU failure as "slower", never "broken".
    pub fn new(ctx: Arc<GpuContext>, dims: Dims, cfg: &InterpolatorConfig) -> Result<Self> {
        let multiplier = cfg.multiplier.clamp(1, 8);
        let interp = GpuInterpolator::new(ctx, dims, u32::from(multiplier), cfg)?;
        if interp.multiplier() != u32::from(multiplier) {
            return Err(FluidError::internal(format!(
                "gpu interpolation multiplier mismatch: asked {multiplier}, got {}",
                interp.multiplier()
            )));
        }
        Ok(Self {
            interp,
            multiplier,
            budget_ns: None,
            last_generate_ns: 0,
        })
    }
}

impl FrameGenBackend for GpuFrameGen {
    fn backend_name(&self) -> &'static str {
        self.interp.backend_name()
    }

    fn multiplier(&self) -> u8 {
        self.multiplier
    }

    fn generate(
        &mut self,
        prev: &Frame,
        next: &Frame,
        _luma_prev: &LumaPlane,
        _luma_next: &LumaPlane,
    ) -> Result<Generated> {
        // Adaptive mode: skip synthesis entirely when the previous pass already
        // overran the frame budget. The GPU produces every intermediate in one
        // dispatch, so the only cheap escape is all-or-nothing; refusing to
        // generate keeps the run at the source cadence instead of falling
        // behind and accumulating latency.
        if let Some(budget) = self.budget_ns
            && self.last_generate_ns > 0
            && self.last_generate_ns > budget
        {
            return Ok(Generated {
                frames: Vec::new(),
                metrics: InterpMetrics::default(),
                flow_ns: 0,
                synth_ns: 0,
            });
        }
        // The GPU computes luma, flow and synthesis itself, so the planes the
        // caller passes in (used by the CPU optical flow) are not needed here.
        let started = Instant::now();
        let frames = self.interp.generate(prev, next)?;
        let elapsed = started.elapsed().as_nanos() as u64;
        self.last_generate_ns = elapsed;
        let metrics = InterpMetrics {
            frames: frames.len() as u32,
            ..InterpMetrics::default()
        };
        Ok(Generated {
            frames,
            metrics,
            // The GPU fuses flow and synthesis into one set of dispatches, so the
            // split is not observable from outside; report it as synthesis time
            // and keep the flow timer at zero instead of inventing a ratio.
            flow_ns: 0,
            synth_ns: elapsed,
        })
    }

    fn reset(&mut self) {
        self.interp.reset();
    }

    fn last_cost_ns(&self) -> u64 {
        self.interp.last_cost_ns()
    }

    fn set_budget_ns(&mut self, budget_ns: Option<u64>) {
        self.budget_ns = budget_ns;
    }
}

/// Largest frame (in pixels) where the integrated-GPU interpolation path is
/// allowed in auto mode.
///
/// Measured on an Intel Iris Xe (i7-1260P, Vulkan), 2x interpolation, engine
/// level, idle machine: at 1280x720 the GPU path is 41-44 ms per pair against
/// 45-52 ms on the CPU (~1.15x), while at 1920x1080 it is 92-101 ms against
/// 84-92 ms (~1.1x slower). The upload plus readback of two frames costs more
/// than the compute saves on a large frame but not on a small one. A discrete
/// GPU has far more bandwidth headroom, so it always gets the GPU path.
pub const INTEGRATED_GPU_INTERP_MAX_PIXELS: u64 = 1_300_000;

/// Should auto mode run frame interpolation on the GPU at this frame size?
fn auto_prefers_gpu(device_type: &str, dims: Dims) -> bool {
    match device_type {
        "discrete" | "virtual" => true,
        "cpu" => false,
        _ => dims.area() <= INTEGRATED_GPU_INTERP_MAX_PIXELS,
    }
}

/// Build the backend the configuration asks for.
///
/// Auto and Gpu try the GPU backend first and fall back to the CPU one when no
/// adapter is usable; Cpu never touches the GPU. A broken driver must only make
/// a run slower, never fail it, and the backend that actually ran is reported.
pub fn make_framegen(
    cfg: &InterpolatorConfig,
    dims: Dims,
    gpu: Option<Arc<GpuContext>>,
) -> Box<dyn FrameGenBackend> {
    match cfg.backend {
        InterpBackendKind::Cpu => Box::new(CpuFrameGen::new(cfg)),
        InterpBackendKind::Gpu => {
            if let Some(ctx) = gpu {
                match GpuFrameGen::new(ctx, dims, cfg) {
                    Ok(backend) => {
                        fluid_core::fluid_info!(
                            "interpolation backend: gpu ({}, {}x{} -> {} frames per pair)",
                            backend.backend_name(),
                            dims.width,
                            dims.height,
                            backend.multiplier()
                        );
                        return Box::new(backend);
                    }
                    Err(err) => {
                        fluid_core::fluid_warn!(
                            "gpu interpolation unavailable ({}x{}): {err}; falling back to the cpu backend",
                            dims.width,
                            dims.height
                        );
                    }
                }
            } else {
                fluid_core::fluid_warn!(
                    "no usable gpu adapter for interpolation; falling back to the cpu backend"
                );
            }
            Box::new(CpuFrameGen::new(cfg))
        }
        InterpBackendKind::Auto => {
            if let Some(ctx) = gpu {
                let device_type = ctx.device_type();
                if auto_prefers_gpu(device_type, dims) {
                    match GpuFrameGen::new(ctx, dims, cfg) {
                        Ok(backend) => {
                            fluid_core::fluid_info!(
                                "interpolation backend: gpu ({}, {} adapter, {}x{} -> {} frames per pair)",
                                backend.backend_name(),
                                device_type,
                                dims.width,
                                dims.height,
                                backend.multiplier()
                            );
                            return Box::new(backend);
                        }
                        Err(err) => {
                            fluid_core::fluid_debug!(
                                "gpu interpolation unavailable ({}x{}): {err}",
                                dims.width,
                                dims.height
                            );
                        }
                    }
                } else {
                    fluid_core::fluid_debug!(
                        "interpolation backend: cpu (auto: {} adapter at {}x{} keeps the cpu path)",
                        device_type,
                        dims.width,
                        dims.height
                    );
                }
            }
            Box::new(CpuFrameGen::new(cfg))
        }
    }
}

/// Fused GPU backend: interpolation and upscaling in one GPU submission.
///
/// The two-step GPU chain interpolates at the source resolution and then upscales
/// every intermediate frame, which means each intermediate leaves the GPU and is
/// uploaded again. The fused pipeline keeps the intermediate in a GPU buffer and
/// upscales it there, so a pair costs one upload of the two source frames and one
/// readback of the destination-sized intermediates.
///
/// At the atom level the kernels cost the same (a crate-level A/B measured 0.97x on
/// an integrated adapter, i.e. no shader-level win, because the destination readback
/// still crosses the shared memory bus). At the *engine* level however the fused path
/// measured 1.04-1.09x (paired median 1.059x over 18 pairs on an Intel Iris Xe) since
/// it saves one GPU round trip and sync per source frame: the two-step chain upscales
/// the interpolated frame as a separate submission. The margin is small and
/// hardware-dependent, so `FLUID_INTERP_FUSED=1/0` forces either side for A/B runs.
/// [`fused_prefers_gpu`] enables it for every GPU adapter class.
pub struct GpuFusedFrameGen {
    pipeline: GpuPipeline,
    multiplier: u8,
    /// Adaptive budget for the next generate call, if any.
    budget_ns: Option<u64>,
    /// Duration of the last fused generate, used to decide the next budget.
    last_generate_ns: u64,
}

impl GpuFusedFrameGen {
    /// Build the fused backend for one source size and output size.
    ///
    /// The upscaling stage uses the same filter, sharpness and temporal settings as
    /// the standalone GPU upscaler, so the pixels match the two-step chain exactly.
    pub fn new(ctx: Arc<GpuContext>, cfg: &FluidConfig, src: Dims, dst: Dims) -> Result<Self> {
        let multiplier = cfg.interpolator.multiplier.clamp(1, 8);
        let pipeline = GpuPipeline::new(
            ctx,
            src,
            dst,
            cfg.upscaler.filter,
            cfg.upscaler.sharpness,
            Some(cfg.upscaler.temporal),
            &cfg.interpolator,
        )?;
        Ok(Self {
            pipeline,
            multiplier,
            budget_ns: None,
            last_generate_ns: 0,
        })
    }
}

impl FrameGenBackend for GpuFusedFrameGen {
    fn backend_name(&self) -> &'static str {
        self.pipeline.backend_name()
    }

    fn multiplier(&self) -> u8 {
        self.multiplier
    }

    fn generate(
        &mut self,
        prev: &Frame,
        next: &Frame,
        _luma_prev: &LumaPlane,
        _luma_next: &LumaPlane,
    ) -> Result<Generated> {
        // Same all-or-nothing adaptive escape as the two-step GPU backend: the
        // fused pipeline produces every intermediate in one submission, so the
        // only cheap option is to skip the whole pair and stay at the source
        // cadence instead of falling behind.
        if let Some(budget) = self.budget_ns
            && self.last_generate_ns > 0
            && self.last_generate_ns > budget
        {
            return Ok(Generated {
                frames: Vec::new(),
                metrics: InterpMetrics::default(),
                flow_ns: 0,
                synth_ns: 0,
            });
        }
        let started = Instant::now();
        // Frames come back already at the destination size, so the processor's
        // upscale stage is skipped for them (processor.rs checks the frame size).
        let frames = self.pipeline.generate(prev, next)?;
        let elapsed = started.elapsed().as_nanos() as u64;
        self.last_generate_ns = elapsed;
        let metrics = InterpMetrics {
            frames: frames.len() as u32,
            ..InterpMetrics::default()
        };
        Ok(Generated {
            frames,
            metrics,
            // Flow, synthesis and upscaling are fused into one GPU submission, so
            // the split is not observable; report the whole cost as synthesis time.
            flow_ns: 0,
            synth_ns: elapsed,
        })
    }

    fn reset(&mut self) {
        self.pipeline.reset();
    }

    fn last_cost_ns(&self) -> u64 {
        self.pipeline.last_cost_ns()
    }

    fn set_budget_ns(&mut self, budget_ns: Option<u64>) {
        self.budget_ns = budget_ns;
    }
}

/// Should auto mode fuse interpolation and upscaling into one GPU pipeline?
///
/// Only when the output is actually resized (a 1:1 output has no upscale stage to
/// fuse). All GPU adapter classes qualify: on this machine's integrated adapter the
/// engine-level paired A/B was 1.059x (see out/fused-engine-bench.md), and discrete
/// adapters have more bandwidth headroom still. `FLUID_INTERP_FUSED=1` forces the
/// fused path on and `=0` forbids it, which is how the fused path is exercised on
/// machines without a discrete GPU.
fn fused_prefers_gpu(device_type: &str, src: Dims, dst: Dims) -> bool {
    dst != src && matches!(device_type, "discrete" | "virtual" | "integrated")
}

/// `FLUID_INTERP_FUSED` override: Some(true) = force, Some(false) = forbid.
fn fused_override() -> Option<bool> {
    let raw = std::env::var("FLUID_INTERP_FUSED").ok()?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "on" | "true" | "yes" | "force" => Some(true),
        "0" | "off" | "false" | "no" => Some(false),
        _ => None,
    }
}

/// Build the frame generation backend, including the fused GPU option.
///
/// This is the engine-facing entry point: it decides between the fused pipeline
/// (interpolation + the configured upscaling in one GPU submission) and the plain
/// [`make_framegen`] backends. A failed fused construction is only a warning; the
/// two-step or CPU path still runs.
pub fn make_framegen_for(
    cfg: &FluidConfig,
    src: Dims,
    dst: Dims,
    gpu: Option<Arc<GpuContext>>,
) -> Box<dyn FrameGenBackend> {
    let backend_wants_gpu = !matches!(cfg.interpolator.backend, InterpBackendKind::Cpu);
    // The fused pipeline always ends with an upscale; with the upscale filter off
    // the processor does not resize at all, so keep the two-step/CPU backends.
    if backend_wants_gpu
        && cfg.upscaler.filter != UpscaleFilter::Off
        && let Some(ctx) = gpu.clone()
    {
        let device_type = ctx.device_type();
        let allowed = fused_override().unwrap_or_else(|| fused_prefers_gpu(device_type, src, dst));
        if allowed {
            match GpuFusedFrameGen::new(ctx, cfg, src, dst) {
                Ok(backend) => {
                    fluid_core::fluid_info!(
                        "interpolation backend: gpu fused ({}, {} adapter, {}x{} -> {}x{} -> {} frames per pair)",
                        backend.backend_name(),
                        device_type,
                        src.width,
                        src.height,
                        dst.width,
                        dst.height,
                        backend.multiplier()
                    );
                    return Box::new(backend);
                }
                Err(err) => {
                    fluid_core::fluid_warn!(
                        "gpu fused interpolation unavailable ({}x{} -> {}x{}): {err}; falling back to the two-step path",
                        src.width,
                        src.height,
                        dst.width,
                        dst.height
                    );
                }
            }
        }
    }
    make_framegen(&cfg.interpolator, src, gpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluid_core::Dims;
    use fluid_interp::Interpolator;

    #[test]
    fn cpu_backend_matches_direct_interpolator() {
        let d = Dims::new(96, 64);
        let prev = fluid_video::synthetic_pattern(d, 0.0, 0);
        let next = fluid_video::synthetic_pattern(d, 1.0, 1);
        let cfg = InterpolatorConfig {
            multiplier: 3,
            ..InterpolatorConfig::default()
        };
        let luma_prev = prev.to_luma();
        let luma_next = next.to_luma();
        let mut backend = CpuFrameGen::new(&cfg);
        assert_eq!(backend.backend_name(), "cpu");
        assert_eq!(backend.multiplier(), 3);
        let out = backend
            .generate(&prev, &next, &luma_prev, &luma_next)
            .unwrap();
        assert_eq!(out.frames.len(), 2);
        assert_eq!(out.metrics.frames, 2);
        assert!(out.synth_ns > 0 && out.flow_ns > 0);

        // Same flow, same alpha => byte-identical frames to calling the
        // interpolator directly, which is what the engine did before the
        // backend split.
        let flow = fluid_flow::estimate(
            &luma_prev,
            &luma_next,
            &fluid_flow::FlowParams::from(&cfg.flow),
        )
        .unwrap();
        let direct = Interpolator::new(&cfg);
        for (k, frame) in out.frames.iter().enumerate() {
            let alpha = (k + 1) as f32 / 3.0;
            let (want, _) = direct.interpolate(&prev, &next, alpha, &flow).unwrap();
            assert_eq!(frame.data, want.data, "frame {k} differs");
        }
    }

    #[test]
    fn adaptive_budget_caps_generated_frames() {
        let d = Dims::new(96, 64);
        let prev = fluid_video::synthetic_pattern(d, 0.0, 0);
        let next = fluid_video::synthetic_pattern(d, 1.0, 1);
        let cfg = InterpolatorConfig {
            multiplier: 4,
            ..InterpolatorConfig::default()
        };
        let luma_prev = prev.to_luma();
        let luma_next = next.to_luma();
        let mut backend = CpuFrameGen::new(&cfg);
        assert_eq!(backend.multiplier(), 4);

        // A zero budget means "no time for synthesis": the backend must
        // still estimate flow (the caller needs the metrics) but generate
        // no intermediate frames.
        backend.set_budget_ns(Some(0));
        let out = backend
            .generate(&prev, &next, &luma_prev, &luma_next)
            .unwrap();
        assert!(out.frames.is_empty(), "{} frames", out.frames.len());
        assert_eq!(out.metrics.frames, 0);

        // None restores the configured multiplier.
        backend.set_budget_ns(None);
        let out = backend
            .generate(&prev, &next, &luma_prev, &luma_next)
            .unwrap();
        assert_eq!(out.frames.len(), 3);
    }
    #[test]
    fn auto_backend_without_a_gpu_is_the_cpu_backend() {
        let d = Dims::new(64, 48);
        let cfg = InterpolatorConfig {
            multiplier: 2,
            ..InterpolatorConfig::default()
        };
        let mut backend = make_framegen(&cfg, d, None);
        assert_eq!(backend.backend_name(), "cpu");
        assert_eq!(backend.multiplier(), 2);
        let prev = fluid_video::synthetic_pattern(d, 0.0, 0);
        let next = fluid_video::synthetic_pattern(d, 1.0, 1);
        let out = backend
            .generate(&prev, &next, &prev.to_luma(), &next.to_luma())
            .unwrap();
        assert_eq!(out.frames.len(), 1);
    }

    #[test]
    fn explicit_cpu_backend_never_asks_the_gpu() {
        let d = Dims::new(64, 48);
        let cfg = InterpolatorConfig {
            multiplier: 2,
            backend: InterpBackendKind::Cpu,
            ..InterpolatorConfig::default()
        };
        let backend = make_framegen(&cfg, d, None);
        assert_eq!(backend.backend_name(), "cpu");
    }

    #[test]
    fn auto_policy_keeps_large_frames_on_the_cpu_for_integrated_gpus() {
        assert!(auto_prefers_gpu("discrete", Dims::new(1920, 1080)));
        assert!(auto_prefers_gpu("virtual", Dims::new(1920, 1080)));
        assert!(!auto_prefers_gpu("cpu", Dims::new(640, 360)));
        assert!(auto_prefers_gpu("integrated", Dims::new(1280, 720)));
        assert!(!auto_prefers_gpu("integrated", Dims::new(1920, 1080)));
        assert!(!auto_prefers_gpu("integrated", Dims::new(2560, 1440)));
        assert!(auto_prefers_gpu("other", Dims::new(800, 600)));
    }

    #[test]
    fn fused_policy_needs_a_resize_and_a_discrete_adapter() {
        let hd = Dims::new(1280, 720);
        let qhd = Dims::new(2560, 1440);
        // A discrete adapter with an actual resize gets the fused pipeline.
        assert!(fused_prefers_gpu("discrete", hd, qhd));
        assert!(fused_prefers_gpu("virtual", hd, qhd));
        // Integrated adapters measured 0.82-1.00x on the fused path, so they keep
        // the two-step chain; a 1:1 output has no upscale stage to fuse at all.
        // Integrated adapters get it too: engine-level paired A/B measured 1.059x.
        assert!(fused_prefers_gpu("integrated", hd, qhd));
        assert!(!fused_prefers_gpu("discrete", hd, hd));
        assert!(!fused_prefers_gpu("cpu", hd, qhd));
        assert!(!fused_prefers_gpu("other", hd, qhd));
    }

    #[test]
    fn fused_entry_point_without_a_gpu_is_the_cpu_backend() {
        let src = Dims::new(64, 48);
        let dst = Dims::new(128, 96);
        let cfg = FluidConfig::default();
        let backend = make_framegen_for(&cfg, src, dst, None);
        assert_eq!(backend.backend_name(), "cpu");
        assert_eq!(backend.multiplier(), 2);
    }
}
