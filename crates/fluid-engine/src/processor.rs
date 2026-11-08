//! Frame processing stage: optional frame generation (interpolation) plus
//! optional upscaling.
//!
//! Ordering contract: for every real source frame the processor returns the
//! frames to present, in presentation order. The last element is always the
//! upscaled real frame; preceding elements are synthesised frames between the
//! previous real frame and this one (alpha = 1/multiplier .. (m-1)/multiplier).
//!
//! Tradeoff: generating m-1 intermediates before the real frame costs m-1
//! synthesis passes of extra latency. Set interpolator.multiplier = 1 (or
//! mode = off) for the lowest-latency path, or keep it small (2-3) for
//! smoother motion. Scene cuts never generate intermediates.

use std::sync::{Arc, OnceLock};
use std::time::Instant;

use fluid_core::{
    Dims, FluidConfig, FluidError, Frame, InterpMode, LumaPlane, Result, UpscaleBackendKind,
    UpscaleFilter,
};
use fluid_flow::SceneCutDetector;
use fluid_gpu::{GpuContext, GpuUpscaler};
use fluid_interp::InterpMetrics;
use fluid_upscale::{UpscaleBackend, Upscaler};

use crate::framegen::{FrameGenBackend, make_framegen_for};

/// The process-wide compute device, created on first use.
///
/// Building a wgpu device initialises the graphics driver, which costs tens of
/// milliseconds and process resources, so every FrameProcessor shares one
/// context instead of creating its own. A missing adapter is cached too: a
/// machine without a usable GPU is not probed again on every pipeline build.
fn gpu_context() -> Option<Arc<GpuContext>> {
    static CONTEXT: OnceLock<Option<Arc<GpuContext>>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| match GpuContext::new() {
            Ok(ctx) => {
                fluid_core::fluid_info!(
                    "gpu compute device ready: {} (wgpu, {})",
                    ctx.info().name,
                    ctx.info().backend
                );
                Some(Arc::new(ctx))
            }
            Err(e) => {
                fluid_core::fluid_debug!("gpu compute device unavailable ({e})");
                None
            }
        })
        .clone()
}

/// Logical cores available to the CPU pipeline, for progress and warning logs.
fn cpu_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Build the upscaling backend the configuration asks for.
///
/// Auto and Gpu try the wgpu compute backend first and fall back to the CPU
/// implementation when the machine has no usable adapter or the filter/size
/// combination cannot be uploaded; Cpu never touches the GPU. Falling back is
/// deliberate: a broken driver must never turn into a failed run, it only makes
/// the run slower, and the backend that actually ran is reported in the
/// pipeline report.
fn make_upscaler(cfg: &FluidConfig, src_dims: Dims, out_dims: Dims) -> Box<dyn UpscaleBackend> {
    let requested = cfg.upscaler.backend;
    // A 1:1 resize is pure memory traffic: on the GPU it costs a full upload
    // plus readback (measured ~21 ms at 1080p on Intel Iris Xe) while the CPU
    // path only has to run the sharpen pass (~2-8 ms) and produces identical
    // pixels. Do not pay the transfer just to be "on the GPU".
    if src_dims == out_dims {
        return Box::new(Upscaler::new(&cfg.upscaler, out_dims));
    }
    let use_gpu = matches!(
        requested,
        UpscaleBackendKind::Auto | UpscaleBackendKind::Gpu
    );
    if use_gpu {
        if let Some(ctx) = gpu_context() {
            match GpuUpscaler::new(
                Arc::clone(&ctx),
                cfg.upscaler.filter,
                src_dims,
                out_dims,
                cfg.upscaler.sharpness,
                Some(cfg.upscaler.temporal),
            ) {
                Ok(up) => return Box::new(up),
                Err(e) => {
                    fluid_core::fluid_warn!(
                        "gpu upscaling unavailable ({}x{} -> {}x{}): {e}; falling back to the cpu backend",
                        src_dims.width,
                        src_dims.height,
                        out_dims.width,
                        out_dims.height
                    );
                }
            }
        } else if requested == UpscaleBackendKind::Gpu {
            fluid_core::fluid_warn!(
                "no usable gpu adapter; falling back to the cpu backend ({} threads)",
                cpu_threads()
            );
        }
    }
    Box::new(Upscaler::new(&cfg.upscaler, out_dims))
}

/// Per-pass timings in nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StageTimings {
    pub flow_ns: u64,
    pub interp_ns: u64,
    pub upscale_ns: u64,
    pub total_ns: u64,
}

/// Result of processing one source frame.
#[derive(Clone, Debug)]
pub struct ProcessOutcome {
    pub frames: Vec<Frame>,
    pub interpolated: u32,
    pub scene_cut: bool,
    /// True when the source frame was pixel-identical to the previous one, so
    /// the generated frames were reused instead of being re-synthesised.
    pub static_frame: bool,
    pub metrics: InterpMetrics,
    pub timings: StageTimings,
}

impl ProcessOutcome {
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

/// Stateful frame processor owning the interpolator, upscaler and last frame.
pub struct FrameProcessor {
    cfg: FluidConfig,
    src_dims: Dims,
    out_dims: Dims,
    framegen: Option<Box<dyn FrameGenBackend>>,
    upscaler: Option<Box<dyn UpscaleBackend>>,
    detector: SceneCutDetector,
    prev: Option<Frame>,
    prev_luma: Option<LumaPlane>,
    /// Last presented real frame, reused for pixel-identical source frames.
    last_out: Option<Frame>,
    scene_cuts: u64,
    /// `interpolator.mode == "adaptive"`: cap synthesis by the source cadence.
    adaptive: bool,
    /// Source cadence used to size the adaptive budget.
    src_fps: f32,
    /// EMA of the per-source-frame cost of everything except synthesis.
    base_cost_ema_ns: f64,
}

impl FrameProcessor {
    /// Build a processor for a source resolution, resolving the output size.
    pub fn new(cfg: &FluidConfig, src_dims: Dims) -> Result<Self> {
        if src_dims.is_empty() {
            return Err(FluidError::invalid("source dimensions must be non-zero"));
        }
        cfg.validate()?;
        let mut cfg = cfg.clone();
        let out_dims = cfg.upscaler.scale.resolve(src_dims);
        if cfg.upscaler.filter == UpscaleFilter::Off && out_dims != src_dims {
            // A scale request without a filter is still a resize request.
            cfg.upscaler.filter = UpscaleFilter::Bilinear;
        }
        let framegen: Option<Box<dyn FrameGenBackend>> =
            if cfg.interpolator.mode != InterpMode::Off && cfg.interpolator.multiplier > 1 {
                Some(make_framegen_for(&cfg, src_dims, out_dims, gpu_context()))
            } else {
                None
            };
        let adaptive = cfg.interpolator.mode == InterpMode::Adaptive;
        let upscaler: Option<Box<dyn UpscaleBackend>> = if cfg.upscaler.filter != UpscaleFilter::Off
        {
            Some(make_upscaler(&cfg, src_dims, out_dims))
        } else {
            None
        };
        Ok(Self {
            detector: SceneCutDetector::new(cfg.interpolator.scene_cut_threshold),
            cfg,
            src_dims,
            out_dims,
            framegen,
            upscaler,
            prev: None,
            prev_luma: None,
            last_out: None,
            scene_cuts: 0,
            adaptive,
            src_fps: 60.0,
            base_cost_ema_ns: 0.0,
        })
    }

    pub fn src_dims(&self) -> Dims {
        self.src_dims
    }

    pub fn out_dims(&self) -> Dims {
        self.out_dims
    }

    pub fn multiplier(&self) -> u32 {
        self.framegen
            .as_ref()
            .map(|f| u32::from(f.multiplier()))
            .unwrap_or(1)
    }

    pub fn interpolation_enabled(&self) -> bool {
        self.framegen.is_some()
    }

    /// Name of the active frame generation backend ("cpu", "wgpu-flow"), if any.
    pub fn interp_backend(&self) -> Option<&'static str> {
        self.framegen.as_ref().map(|f| f.backend_name())
    }

    pub fn upscaling_enabled(&self) -> bool {
        self.upscaler.is_some()
    }

    /// Name of the active upscaling backend ("cpu", "d3d11-compute"), if any.
    pub fn upscale_backend(&self) -> Option<&'static str> {
        self.upscaler.as_ref().map(|u| u.backend_name())
    }

    pub fn scene_cuts(&self) -> u64 {
        self.scene_cuts
    }

    pub fn config(&self) -> &FluidConfig {
        &self.cfg
    }

    /// Tell the processor the source cadence in Hz.
    ///
    /// Only adaptive interpolation uses it: the per-source-frame budget is
    /// one source interval minus the measured cost of flow and upscaling, so
    /// synthesis stops when it would make the run fall behind the source.
    pub fn set_source_fps(&mut self, fps: f32) {
        if fps.is_finite() && fps > 1.0 {
            self.src_fps = fps;
        }
    }

    /// True when interpolation is in adaptive (time-budgeted) mode.
    pub fn adaptive_interp(&self) -> bool {
        self.adaptive
    }

    /// Time left for synthesis on the next source frame, if adaptive.
    fn interp_budget_ns(&self) -> Option<u64> {
        if !self.adaptive {
            return None;
        }
        let interval_ns = 1.0e9 / f64::from(self.src_fps.max(1.0));
        let base_ns = self.base_cost_ema_ns.clamp(0.0, interval_ns);
        // Keep 15% of the interval as headroom so a jittery frame does not
        // push the whole pipeline behind the compositor.
        Some((((interval_ns - base_ns) * 0.85).max(0.0)) as u64)
    }
    /// Forget temporal state (resolution change, scene cut, seek).
    pub fn reset(&mut self) {
        self.prev = None;
        self.prev_luma = None;
        self.last_out = None;
        if let Some(u) = self.upscaler.as_mut() {
            u.reset();
        }
        if let Some(f) = self.framegen.as_mut() {
            f.reset();
        }
    }

    /// Process one source frame, returning the frames to present in order.
    pub fn process(&mut self, frame: Frame) -> Result<ProcessOutcome> {
        let t0 = Instant::now();
        if frame.dims != self.src_dims {
            return Err(FluidError::invalid(format!(
                "frame is {:?} but processor was built for {:?}",
                frame.dims, self.src_dims
            )));
        }
        let mut frames: Vec<Frame> = Vec::with_capacity(8);
        let mut metrics = InterpMetrics::default();
        let mut scene_cut = false;
        let mut flow_ns = 0u64;
        let mut interp_ns = 0u64;
        let mut upscale_ns = 0u64;
        let mut interpolated = 0u32;
        let mut pending: Vec<Frame> = Vec::new();
        let first_frame = self.prev.is_none();
        let mut static_frame = false;
        let interp_budget = self.interp_budget_ns();

        if self.framegen.is_some() {
            // Pixel-identical source frames have zero motion: every
            // synthesised frame is the same image, so reuse the last
            // presented real frame instead of running flow, synthesis and
            // upscaling again. A static desktop or a paused game hits this
            // path on almost every frame (DXGI repeats the previous image),
            // and it is the difference between ~1 ms and ~50 ms per frame at
            // 1080p. The comparison is a byte comparison, so the fast path
            // can never change the output pixels.
            let identical = match self.prev.as_ref() {
                Some(prev) => {
                    prev.dims == frame.dims && prev.fmt == frame.fmt && prev.data == frame.data
                }
                None => false,
            };
            if identical && self.last_out.is_some() {
                static_frame = true;
                let m = self.framegen.as_ref().map(|f| f.multiplier()).unwrap_or(1);
                metrics.frames = u32::from(m.saturating_sub(1));
                if let Some(last) = self.last_out.clone() {
                    for _ in 1..m {
                        frames.push(last.clone());
                        interpolated += 1;
                    }
                }
            } else if let Some(prev) = self.prev.as_ref() {
                let luma_prev = match self.prev_luma.take() {
                    Some(l) => l,
                    None => prev.to_luma(),
                };
                let luma_cur = frame.to_luma();
                scene_cut = self.detector.is_cut(&luma_prev, &luma_cur);
                if scene_cut {
                    self.scene_cuts += 1;
                    if let Some(u) = self.upscaler.as_mut() {
                        u.reset();
                    }
                    if let Some(f) = self.framegen.as_mut() {
                        f.reset();
                    }
                } else if let Some(backend) = self.framegen.as_mut() {
                    backend.set_budget_ns(interp_budget);
                    let generated = backend.generate(prev, &frame, &luma_prev, &luma_cur)?;
                    flow_ns = generated.flow_ns;
                    interp_ns = generated.synth_ns;
                    metrics = generated.metrics;
                    pending.extend(generated.frames);
                }
                self.prev_luma = Some(luma_cur);
            } else {
                self.prev_luma = Some(frame.to_luma());
            }
        }

        for mid in pending {
            // A fused GPU backend may hand back frames that are already at
            // the output resolution; in that case there is nothing to upscale.
            if mid.dims == self.out_dims {
                frames.push(mid);
            } else {
                frames.push(self.upscale_frame(mid, &mut upscale_ns)?);
            }
            interpolated += 1;
        }

        metrics.scene_cut = scene_cut;

        let real = if static_frame {
            self.last_out.clone().ok_or_else(|| {
                FluidError::internal("static frame fast path without a cached frame")
            })?
        } else {
            let real = self.upscale_frame(frame.clone(), &mut upscale_ns)?;
            if self.framegen.is_some() {
                self.last_out = Some(real.clone());
            }
            real
        };
        frames.push(real.clone());
        // The very first source frame has no previous frame to interpolate
        // against. Duplicate it so the output keeps a constant `multiplier`
        // cadence from the first frame on (important for offline video where
        // the output framerate is written into the container).
        if first_frame && let Some(fg) = self.framegen.as_ref() {
            for _ in 1..fg.multiplier() {
                frames.push(real.clone());
            }
        }
        self.prev = Some(frame);

        let total_ns = t0.elapsed().as_nanos() as u64;
        if self.adaptive {
            // Track everything except synthesis: flow plus the upscale of the
            // real frame is what the next budget has to pay for anyway.
            let base_ns = total_ns.saturating_sub(interp_ns) as f64;
            self.base_cost_ema_ns = if self.base_cost_ema_ns <= 0.0 {
                base_ns
            } else {
                self.base_cost_ema_ns * 0.8 + base_ns * 0.2
            };
        }

        Ok(ProcessOutcome {
            frames,
            interpolated,
            scene_cut,
            static_frame,
            metrics,
            timings: StageTimings {
                flow_ns,
                interp_ns,
                upscale_ns,
                total_ns,
            },
        })
    }

    fn upscale_frame(&mut self, frame: Frame, upscale_ns: &mut u64) -> Result<Frame> {
        match self.upscaler.as_mut() {
            Some(up) => {
                let t = Instant::now();
                let mut out = Frame::zeroed(up.dst_dims(), frame.fmt);
                up.upscale(&frame, &mut out)?;
                *upscale_ns += t.elapsed().as_nanos() as u64;
                Ok(out)
            }
            None => Ok(frame),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluid_core::{Frame, PixelFormat, QualityPreset};

    fn frame(dims: Dims, value: u8) -> Frame {
        Frame::new(
            dims,
            PixelFormat::Rgba8,
            vec![value; dims.area() as usize * 4],
        )
        .expect("valid frame")
    }

    #[test]
    fn rejects_empty_dims() {
        let cfg = FluidConfig::default();
        assert!(FrameProcessor::new(&cfg, Dims::new(0, 0)).is_err());
    }

    #[test]
    fn interpolation_off_passes_one_frame() {
        let d = Dims::new(8, 8);
        let mut cfg = FluidConfig::default();
        cfg.interpolator.mode = InterpMode::Off;
        cfg.upscaler.filter = UpscaleFilter::Off;
        let mut p = FrameProcessor::new(&cfg, d).unwrap();
        assert_eq!(p.out_dims(), d);
        let out = p.process(frame(d, 10)).unwrap();
        assert_eq!(out.frames.len(), 1);
        assert_eq!(out.frames[0].dims, d);
        let out2 = p.process(frame(d, 20)).unwrap();
        assert_eq!(out2.frames.len(), 1);
    }

    #[test]
    fn scale_resolves_output_dims() {
        let d = Dims::new(16, 16);
        let mut cfg = FluidConfig::default();
        cfg.upscaler.filter = UpscaleFilter::Bilinear;
        cfg.upscaler.scale.factor = 2.0;
        let p = FrameProcessor::new(&cfg, d).unwrap();
        assert_eq!(p.out_dims(), Dims::new(32, 32));
    }

    #[test]
    fn explicit_cpu_backend_is_used_for_resizing() {
        let d = Dims::new(8, 8);
        let mut cfg = FluidConfig::default();
        cfg.upscaler.backend = UpscaleBackendKind::Cpu;
        cfg.upscaler.filter = UpscaleFilter::Bilinear;
        cfg.upscaler.scale.factor = 2.0;
        let p = FrameProcessor::new(&cfg, d).unwrap();
        assert_eq!(p.out_dims(), Dims::new(16, 16));
        assert_eq!(p.upscale_backend(), Some("cpu"));
    }

    #[test]
    fn identity_resize_prefers_cpu_even_when_gpu_is_requested() {
        let mut cfg = FluidConfig::default();
        cfg.upscaler.backend = UpscaleBackendKind::Gpu;
        cfg.upscaler.filter = UpscaleFilter::Adaptive;
        cfg.upscaler.scale.mode = fluid_core::ScaleMode::Native;
        cfg.upscaler.sharpness = 0.3;
        let dims = Dims::new(64, 48);
        let processor = FrameProcessor::new(&cfg, dims).unwrap();
        assert_eq!(processor.upscale_backend(), Some("cpu"));
        assert_eq!(processor.out_dims(), dims);
    }

    #[test]
    fn off_filter_has_no_backend() {
        let d = Dims::new(8, 8);
        let mut cfg = FluidConfig::default();
        cfg.upscaler.backend = UpscaleBackendKind::Gpu;
        cfg.upscaler.filter = UpscaleFilter::Off;
        cfg.upscaler.scale.factor = 1.0;
        let p = FrameProcessor::new(&cfg, d).unwrap();
        assert_eq!(p.upscale_backend(), None);
    }

    #[test]
    fn static_source_frames_reuse_the_previous_output() {
        let d = Dims::new(16, 12);
        let mut cfg = FluidConfig::default();
        cfg.interpolator.mode = InterpMode::Fixed;
        cfg.interpolator.multiplier = 2;
        cfg.upscaler.filter = UpscaleFilter::Off;
        cfg.upscaler.scale.factor = 1.0;
        let mut p = FrameProcessor::new(&cfg, d).unwrap();
        let first = p.process(frame(d, 40)).unwrap();
        assert!(!first.static_frame);
        assert_eq!(
            first.frames.len(),
            2,
            "first frame duplicates to keep cadence"
        );
        let second = p.process(frame(d, 40)).unwrap();
        assert!(
            second.static_frame,
            "identical frame must take the fast path"
        );
        assert_eq!(second.frames.len(), 2);
        assert_eq!(second.interpolated, 1);
        assert_eq!(second.timings.flow_ns, 0);
        assert_eq!(second.timings.interp_ns, 0);
        for f in &second.frames {
            assert_eq!(
                f.data, first.frames[0].data,
                "reused output must be byte-identical"
            );
        }
        // A different frame must leave the fast path again.
        let third = p.process(frame(d, 41)).unwrap();
        assert!(!third.static_frame);
    }

    #[test]
    fn generated_frames_already_at_output_dims_are_not_upscaled_again() {
        struct FixedBackend {
            multiplier: u8,
            dst: Dims,
        }
        impl crate::framegen::FrameGenBackend for FixedBackend {
            fn backend_name(&self) -> &'static str {
                "test-dst"
            }
            fn multiplier(&self) -> u8 {
                self.multiplier
            }
            fn generate(
                &mut self,
                _prev: &Frame,
                next: &Frame,
                _luma_prev: &LumaPlane,
                _luma_cur: &LumaPlane,
            ) -> Result<crate::framegen::Generated> {
                let frames = (0..u32::from(self.multiplier) - 1)
                    .map(|_| Frame::zeroed(self.dst, next.fmt))
                    .collect();
                Ok(crate::framegen::Generated {
                    frames,
                    metrics: InterpMetrics::default(),
                    flow_ns: 1,
                    synth_ns: 2,
                })
            }
            fn reset(&mut self) {}
        }
        let src = Dims::new(8, 8);
        let out = Dims::new(16, 16);
        let mut cfg = FluidConfig::default();
        cfg.upscaler.filter = UpscaleFilter::Bilinear;
        cfg.upscaler.scale.factor = 2.0;
        let mut p = FrameProcessor {
            cfg: cfg.clone(),
            src_dims: src,
            out_dims: out,
            framegen: Some(Box::new(FixedBackend {
                multiplier: 3,
                dst: out,
            })),
            upscaler: Some(Box::new(Upscaler::new(&cfg.upscaler, out))),
            detector: SceneCutDetector::new(cfg.interpolator.scene_cut_threshold),
            prev: None,
            prev_luma: None,
            last_out: None,
            scene_cuts: 0,
            adaptive: false,
            src_fps: 60.0,
            base_cost_ema_ns: 0.0,
        };
        let first = p.process(frame(src, 1)).unwrap();
        assert_eq!(first.frames.len(), 3);
        let second = p.process(frame(src, 2)).unwrap();
        // 2 generated frames (already 16x16) + the upscaled real frame.
        assert_eq!(second.frames.len(), 3);
        assert_eq!(second.interpolated, 2);
        for f in &second.frames {
            assert_eq!(f.dims, out, "all presented frames must be output sized");
        }
        assert_eq!(second.timings.flow_ns, 1);
        assert_eq!(second.timings.interp_ns, 2);
    }
    #[test]
    fn preset_multiplier_is_respected() {
        let d = Dims::new(16, 16);
        let mut cfg = FluidConfig::with_preset(QualityPreset::Performance);
        cfg.interpolator.mode = InterpMode::Fixed;
        cfg.interpolator.multiplier = 4;
        cfg.upscaler.filter = UpscaleFilter::Off;
        let mut p = FrameProcessor::new(&cfg, d).unwrap();
        assert_eq!(p.multiplier(), 4);
        let _ = p.process(frame(d, 1));
        let out = p.process(frame(d, 2)).unwrap();
        assert_eq!(out.frames.len(), 4);
        assert_eq!(out.interpolated, 3);
    }
}
