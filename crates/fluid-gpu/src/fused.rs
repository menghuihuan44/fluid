//! Fused frame interpolation and upscaling in a single GPU pass chain.
//!
//! The two-step path is: [crate::GpuInterpolator] synthesises each intermediate
//! frame at the source resolution and reads it back, then [crate::GpuUpscaler]
//! uploads that frame again to upscale it. Both round trips are pure overhead:
//! the intermediate frame never has to leave the GPU. [GpuPipeline] keeps it
//! there - the synthesis kernels write the frame into a buffer that the
//! upscaling, sharpening and temporal-blending kernels read directly - and only
//! the finished destination-resolution frames are read back.
//!
//! Resource layout is fixed at construction (flow pyramid, synthesis buffers,
//! upscaling buffer pair, history, staging) so 'generate' performs two uploads
//! and one readback per intermediate frame, and allocates no GPU resource.

use std::sync::Arc;
use std::time::Instant;

use fluid_core::{
    Dims, FluidError, Frame, InterpolatorConfig, Result, TemporalConfig, UpscaleFilter,
};

use crate::GpuContext;
use crate::device;
use crate::flow::FlowRig;
use crate::pipeline::{self, Config};
use crate::synth::{Synth, lerp_i64};

/// Fused GPU interpolation + upscaling pipeline.
///
/// One pipeline owns the flow estimator, the synthesis stage and the upscaling
/// stage for a fixed source size, destination size and configuration, so create
/// it once per resolution and call [GpuPipeline::generate] for every frame pair.
pub struct GpuPipeline {
    ctx: Arc<GpuContext>,
    src: Dims,
    dst: Dims,
    multiplier: u32,
    rig: FlowRig,
    synth: Synth,
    upscale: pipeline::Pipeline,
    last_cost_ns: u64,
}

impl std::fmt::Debug for GpuPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuPipeline")
            .field("src", &self.src)
            .field("dst", &self.dst)
            .field("multiplier", &self.multiplier)
            .field("last_cost_ns", &self.last_cost_ns)
            .finish_non_exhaustive()
    }
}

impl GpuPipeline {
    /// Build the fused pipeline for one source size, destination size and
    /// upscaling configuration.
    ///
    /// 'filter', 'sharpness' and 'temporal' mean exactly what they mean for
    /// [crate::GpuUpscaler]: the intermediate frames are synthesised at the
    /// source resolution and then upscaled with the same spatial filter,
    /// sharpening amount and temporal blending the two-step chain would use.
    ///
    /// Returns [FluidError::Unsupported] when the adapter cannot hold the
    /// buffers; callers should fall back to the two-step or CPU path.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: Arc<GpuContext>,
        src: Dims,
        dst: Dims,
        filter: UpscaleFilter,
        sharpness: f32,
        temporal: Option<TemporalConfig>,
        cfg: &InterpolatorConfig,
    ) -> Result<Self> {
        if src.is_empty() {
            return Err(FluidError::invalid(format!(
                "gpu fused: source size {src} is empty"
            )));
        }
        if dst.is_empty() {
            return Err(FluidError::invalid(format!(
                "upscale: destination size {dst} is empty"
            )));
        }
        let multiplier = u32::from(cfg.multiplier.clamp(1, 8));
        let (spatial, temporal_mode) = crate::spatial_config(filter, src, dst, temporal);
        let sharpen_q8 = crate::sharpen_strength_q8(filter, sharpness);
        let temporal_config = temporal.unwrap_or_default();
        let config = Config {
            src,
            dst,
            spatial,
            sharpen_q8,
            temporal: temporal_mode,
            blend_s: crate::temporal_blend_weight(&temporal_config),
        };

        let (device, queue) = {
            let shared = ctx.lock()?;
            (shared.device.clone(), shared.queue.clone())
        };
        let rig = FlowRig::new(&device, &queue, src, &cfg.flow)?;
        let synth = Synth::new(&device, &queue, src, cfg, &rig)?;
        // The upscaling stage reads the synthesis output buffer directly, so no
        // intermediate frame is ever copied through the CPU.
        let upscale = pipeline::Pipeline::with_source(&device, &queue, config, synth.out_px())?;

        Ok(Self {
            ctx,
            src,
            dst,
            multiplier,
            rig,
            synth,
            upscale,
            last_cost_ns: 0,
        })
    }

    /// Source frame size this pipeline was built for.
    pub fn src_dims(&self) -> Dims {
        self.src
    }

    /// Destination frame size this pipeline was built for.
    pub fn dst_dims(&self) -> Dims {
        self.dst
    }

    /// Frame size accepted by [GpuPipeline::generate].
    pub fn dims(&self) -> Dims {
        self.src
    }

    /// Number of intermediate frames produced per pair.
    pub fn multiplier(&self) -> u32 {
        self.multiplier
    }

    /// Stable backend identifier reported to callers.
    pub fn backend_name(&self) -> &'static str {
        "wgpu-fused"
    }

    /// Wall time of the last [GpuPipeline::generate] call in nanoseconds,
    /// including the source upload, the GPU passes and the destination readback.
    pub fn last_cost_ns(&self) -> u64 {
        self.last_cost_ns
    }

    /// Drop the temporal history, for a scene cut or a stream discontinuity.
    ///
    /// The interpolated frames of the next pair start a fresh history, exactly
    /// like [crate::GpuUpscaler::reset].
    pub fn reset(&mut self) {
        self.upscale.reset_history();
        self.last_cost_ns = 0;
    }

    /// Interpolate 'prev' and 'next' and upscale every intermediate frame.
    ///
    /// Returns the intermediate frames already at the destination size (one per
    /// multiplier step, so multiplier - 1 frames), or an empty vector for a scene
    /// change. [FluidError::Internal] means a GPU command failed; callers should
    /// fall back to the CPU path.
    pub fn generate(&mut self, prev: &Frame, next: &Frame) -> Result<Vec<Frame>> {
        let started = Instant::now();
        if prev.dims != self.src || next.dims != self.src {
            return Err(FluidError::invalid(format!(
                "gpu fused: frame size {} / {} does not match {}",
                prev.dims, next.dims, self.src
            )));
        }
        if prev.fmt != next.fmt {
            return Err(FluidError::invalid(format!(
                "gpu fused: frame format {} / {} differ",
                prev.fmt, next.fmt
            )));
        }
        if self.multiplier <= 1 {
            return Ok(Vec::new());
        }
        let fmt = prev.fmt;

        // Clone the device handles so packing and readback never hold the context
        // lock across a mutable borrow of this pipeline.
        let (device, queue, errors) = {
            let shared = self.ctx.lock()?;
            (
                shared.device.clone(),
                shared.queue.clone(),
                shared.errors.clone(),
            )
        };
        self.synth.pack(prev, next, &queue);

        // Luma planes, the whole flow schedule and the scene-change metric.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fluid-gpu fused flow"),
        });
        self.synth.encode_luma(&mut encoder);
        self.rig.encode(&mut encoder);
        self.synth.encode_cut(&mut encoder);
        queue.submit(Some(encoder.finish()));
        if let Some(error) = device::take_error(&errors) {
            return Err(FluidError::internal(format!(
                "gpu fused: wgpu error: {error}"
            )));
        }
        if self.synth.scene_cut(&device)? {
            self.last_cost_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            return Ok(Vec::new());
        }

        let mut frames = Vec::with_capacity(self.multiplier.saturating_sub(1) as usize);
        for step in 1..self.multiplier {
            let alpha = step as f32 / self.multiplier as f32;
            self.synth.write_alpha(&queue, alpha);

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fluid-gpu fused frame"),
            });
            self.synth.encode_synth(&mut encoder);
            self.upscale.encode_frame(&mut encoder);
            queue.submit(Some(encoder.finish()));
            if let Some(error) = device::take_error(&errors) {
                return Err(FluidError::internal(format!(
                    "gpu fused: wgpu error: {error}"
                )));
            }

            let mut frame = Frame::zeroed(self.dst, fmt);
            self.upscale.finish_frame(&device, &mut frame)?;
            frame.index = prev.index;
            frame.pts_ns = lerp_i64(prev.pts_ns, next.pts_ns, alpha);
            frames.push(frame);
        }

        self.last_cost_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        Ok(frames)
    }
}
