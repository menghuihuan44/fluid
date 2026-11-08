//! GPU frame interpolation: GPU optical flow followed by GPU motion-compensated
//! synthesis.
//!
//! The synthesis arithmetic lives in [crate::synth], which the fused
//! interpolation + upscaling pipeline [crate::GpuPipeline] reuses, so the two
//! paths cannot drift apart. This module adds the flow estimator and the
//! frame-level plumbing: upload of the pair, the scene-change decision, one
//! command buffer per intermediate frame and the synchronous readback.
//!
//! Scene-change detection is deliberately conservative on the GPU. The CPU
//! detector combines a luma histogram distance with the mean absolute luma
//! difference; this backend only computes the mean absolute difference, and
//! only reports a cut when twice the configured threshold is exceeded. That
//! keeps every reported cut a cut the CPU detector would also report, so a
//! scene change can never silently produce a smeared blend frame.

use std::sync::Arc;
use std::time::Instant;

use fluid_core::{Dims, FluidError, Frame, InterpolatorConfig, Result};

use crate::GpuContext;
use crate::device;
use crate::flow::FlowRig;
use crate::synth::{Synth, lerp_i64};

/// GPU frame interpolator.
///
/// One interpolator owns the flow estimator, the synthesis pipeline and every
/// buffer for a fixed frame size, so a pipeline creates it once and calls
/// [GpuInterpolator::generate] for each frame pair. No large buffer is allocated
/// per frame; only the returned [Frame]s are.
pub struct GpuInterpolator {
    ctx: Arc<GpuContext>,
    dims: Dims,
    multiplier: u32,
    rig: FlowRig,
    synth: Synth,
    last_cost_ns: u64,
}

impl std::fmt::Debug for GpuInterpolator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuInterpolator")
            .field("dims", &self.dims)
            .field("multiplier", &self.multiplier)
            .field("last_cost_ns", &self.last_cost_ns)
            .finish_non_exhaustive()
    }
}

impl GpuInterpolator {
    /// Create the interpolator for one frame size and configuration.
    ///
    /// Fails with [FluidError::Unsupported] when the device lacks the buffers
    /// this size needs; callers should fall back to the CPU interpolator.
    pub fn new(
        ctx: Arc<GpuContext>,
        dims: Dims,
        multiplier: u32,
        cfg: &InterpolatorConfig,
    ) -> Result<Self> {
        if dims.is_empty() {
            return Err(FluidError::invalid(
                "gpu interpolation: frame dimensions are empty",
            ));
        }
        let multiplier = multiplier.clamp(1, 8);
        let (device, queue) = {
            let shared = ctx.lock()?;
            (shared.device.clone(), shared.queue.clone())
        };
        let rig = FlowRig::new(&device, &queue, dims, &cfg.flow)?;
        let synth = Synth::new(&device, &queue, dims, cfg, &rig)?;

        Ok(Self {
            ctx,
            dims,
            multiplier,
            rig,
            synth,
            last_cost_ns: 0,
        })
    }

    /// Frame size this interpolator was built for.
    pub fn dims(&self) -> Dims {
        self.dims
    }

    /// Number of intermediate frames produced per pair.
    pub fn multiplier(&self) -> u32 {
        self.multiplier
    }

    /// Stable backend identifier reported to callers.
    pub fn backend_name(&self) -> &'static str {
        "wgpu-flow"
    }

    /// Wall time of the last [GpuInterpolator::generate] call in nanoseconds.
    pub fn last_cost_ns(&self) -> u64 {
        self.last_cost_ns
    }

    /// Drop cached state; buffers stay allocated and the interpolator stays
    /// usable.
    pub fn reset(&mut self) {
        self.last_cost_ns = 0;
    }

    /// Generate the intermediate frames between 'prev' and 'next'.
    ///
    /// Returns one frame per multiplier step (multiplier - 1 frames), each with
    /// the point in time between the two inputs, or an empty vector for a scene
    /// change. [FluidError::Internal] means a GPU command failed; callers should
    /// treat that like any other backend failure and fall back to the CPU path.
    pub fn generate(&mut self, prev: &Frame, next: &Frame) -> Result<Vec<Frame>> {
        let started = Instant::now();
        if prev.dims != self.dims || next.dims != self.dims {
            return Err(FluidError::invalid(format!(
                "gpu interpolation: frame size {} / {} does not match {}",
                prev.dims, next.dims, self.dims
            )));
        }
        if prev.fmt != next.fmt {
            return Err(FluidError::invalid(format!(
                "gpu interpolation: frame format {} / {} differ",
                prev.fmt, next.fmt
            )));
        }
        if self.multiplier <= 1 {
            return Ok(Vec::new());
        }
        let fmt = prev.fmt;

        // Clone the device handles so packing and readback never hold the context
        // lock across a mutable borrow of this interpolator.
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
            label: Some("fluid-gpu interp flow"),
        });
        self.synth.encode_luma(&mut encoder);
        self.rig.encode(&mut encoder);
        self.synth.encode_cut(&mut encoder);
        queue.submit(Some(encoder.finish()));
        if let Some(error) = device::take_error(&errors) {
            return Err(FluidError::internal(format!(
                "gpu interpolation: wgpu error: {error}"
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
                label: Some("fluid-gpu interp frame"),
            });
            self.synth.encode_synth(&mut encoder);
            self.synth.encode_copy_out(&mut encoder);
            queue.submit(Some(encoder.finish()));
            if let Some(error) = device::take_error(&errors) {
                return Err(FluidError::internal(format!(
                    "gpu interpolation: wgpu error: {error}"
                )));
            }

            let mut frame = self.synth.read_out(&device, fmt)?;
            frame.index = prev.index;
            frame.pts_ns = lerp_i64(prev.pts_ns, next.pts_ns, alpha);
            frames.push(frame);
        }

        self.last_cost_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        Ok(frames)
    }
}
