//! GPU motion-compensated synthesis stage, shared by both interpolation front
//! ends.
//!
//! [crate::GpuInterpolator] runs this stage and returns its frames at the
//! source resolution; [crate::GpuPipeline] instead feeds the very same
//! source-resolution buffer straight into the upscaling compute kernels. There is
//! one implementation, so the fused path cannot drift away from the two-step
//! path.
//!
//! The synthesis reproduces 'fluid-interp's 'synth' exactly: the dense forward
//! gather flow compensates both input frames, the two bilinear samples are
//! averaged with the interpolation alpha, and the result is optionally mixed
//! with a plain cross-fade by the configured blend weight. Pixels whose flow
//! confidence falls below 'fluid-flow's default threshold get either the
//! cross-fade (when occlusion filling is off) or the compensated value, exactly
//! like the CPU implementation.
//!
//! Scene-change detection is deliberately conservative. The CPU detector
//! combines a luma histogram distance with the mean absolute luma difference;
//! this backend only computes the mean absolute difference, and only reports a
//! cut when twice the configured threshold is exceeded. That keeps every
//! reported cut a cut the CPU detector would also report, so a scene change can
//! never silently produce a smeared blend frame.

use fluid_core::{Dims, FluidError, Frame, InterpolatorConfig, PixelFormat, Result};
use wgpu::{Buffer, BufferUsages, CommandEncoder, ComputePipeline, Device, Queue};

use crate::convert;
use crate::flow::{FlowRig, bind_group, groups, params_buffer, words};
use crate::pipeline;

const SHADER_LUMA: &str = concat!(
    include_str!("flow_common.wgsl"),
    include_str!("shader_interp_luma.wgsl")
);
const SHADER_CUT: &str = concat!(
    include_str!("flow_common.wgsl"),
    include_str!("shader_interp_cut.wgsl")
);
const SHADER_INTERP: &str = concat!(
    include_str!("flow_common.wgsl"),
    include_str!("shader_interp.wgsl")
);

/// Confidence threshold of 'fluid-flow's default parameters, which
/// 'fluid-interp' hardcodes into its synthesis.
pub(crate) const MIN_CONFIDENCE: f32 = 0.2;

/// Linear interpolation of the two frame timestamps at 't', rounded to the
/// nearest nanosecond like the CPU interpolator.
pub(crate) fn lerp_i64(a: i64, b: i64, t: f32) -> i64 {
    let a = a as f64;
    let b = b as f64;
    (a + (b - a) * f64::from(t)).round() as i64
}

/// Persistent synthesis state for one frame size.
///
/// The flow estimator is owned by the caller: this stage only records the luma
/// pre-pass that feeds [FlowRig::encode] and the synthesis dispatch that consumes
/// the resulting flow fields. Every buffer is allocated once, so generating a
/// frame only writes two parameter blocks and records passes.
pub(crate) struct Synth {
    dims: Dims,
    frame_bytes: u64,
    blend: f32,
    occlusion_fill: bool,
    scene_cut_threshold: f32,

    pipeline_luma: ComputePipeline,
    pipeline_cut: ComputePipeline,
    pipeline_interp: ComputePipeline,
    bind_luma: wgpu::BindGroup,
    bind_cut: wgpu::BindGroup,
    bind_interp: wgpu::BindGroup,

    prev_px: Buffer,
    next_px: Buffer,
    out_px: Buffer,
    /// Owned here (not just by the bind groups) so the intermediate motion sums
    /// and their parameter blocks stay alive for the whole life of the stage.
    #[allow(dead_code)]
    partials: Buffer,
    cut_sum: Buffer,
    #[allow(dead_code)]
    luma_params: Buffer,
    #[allow(dead_code)]
    cut_params: Buffer,
    interp_params: Buffer,
    staging: Buffer,
    cut_staging: Buffer,
    /// Canonical RGBA scratch pixels, grown once and reused every frame.
    pixels: Vec<u32>,
}

impl Synth {
    /// Build the stage for one frame size and interpolation configuration.
    pub(crate) fn new(
        device: &Device,
        queue: &Queue,
        dims: Dims,
        cfg: &InterpolatorConfig,
        rig: &FlowRig,
    ) -> Result<Self> {
        let area = dims.area();
        let frame_bytes = area * 4;
        let grid = groups(dims.width, dims.height);

        let pixels = vec![0u32; area as usize];
        let prev_px = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fluid-gpu interp prev"),
            size: frame_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let next_px = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fluid-gpu interp next"),
            size: frame_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_px = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fluid-gpu interp out"),
            size: frame_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let partial_count = u64::from(grid.0) * u64::from(grid.1) * 4;
        let partials = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fluid-gpu interp cut partials"),
            size: partial_count.max(4),
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let cut_sum = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fluid-gpu interp cut sum"),
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fluid-gpu interp readback"),
            size: frame_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cut_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fluid-gpu interp cut readback"),
            size: 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline_luma = pipeline::make_pipeline(device, "fluid-gpu interp luma", SHADER_LUMA)?;
        let pipeline_cut = pipeline::make_pipeline(device, "fluid-gpu interp cut", SHADER_CUT)?;
        let pipeline_interp =
            pipeline::make_pipeline(device, "fluid-gpu interp synth", SHADER_INTERP)?;

        let luma_params = params_buffer(
            device,
            queue,
            "fluid-gpu interp luma params",
            words(
                [dims.width, dims.height, 0, 0],
                [0; 4],
                [grid.0, grid.1, 0, 0],
                [0.0; 4],
                [0.0; 4],
            ),
        );
        let cut_params = params_buffer(
            device,
            queue,
            "fluid-gpu interp cut params",
            words(
                [0; 4],
                [0; 4],
                [grid.0 * grid.1, 0, 0, 0],
                [0.0; 4],
                [0.0; 4],
            ),
        );
        let interp_params = params_buffer(
            device,
            queue,
            "fluid-gpu interp params",
            words(
                [dims.width, dims.height, 0, 0],
                [0; 4],
                [u32::from(cfg.occlusion_fill), 0, 0, 0],
                [0.5, cfg.blend, MIN_CONFIDENCE, 0.5],
                [0.0; 4],
            ),
        );

        let (luma_prev, luma_next) = rig.level0_luma();
        let bind_luma = bind_group(
            device,
            &pipeline_luma,
            "fluid-gpu interp luma bindings",
            &[
                (0, &prev_px),
                (1, &next_px),
                (2, luma_prev),
                (3, luma_next),
                (4, &partials),
                (5, &luma_params),
            ],
        );
        let bind_cut = bind_group(
            device,
            &pipeline_cut,
            "fluid-gpu interp cut bindings",
            &[(0, &partials), (1, &cut_sum), (2, &cut_params)],
        );
        let bind_interp = bind_group(
            device,
            &pipeline_interp,
            "fluid-gpu interp synth bindings",
            &[
                (0, &prev_px),
                (1, &next_px),
                (2, &rig.dense_forward),
                (3, &rig.confidence),
                (4, &out_px),
                (5, &interp_params),
            ],
        );

        Ok(Self {
            dims,
            frame_bytes,
            blend: cfg.blend,
            occlusion_fill: cfg.occlusion_fill,
            scene_cut_threshold: cfg.scene_cut_threshold,
            pipeline_luma,
            pipeline_cut,
            pipeline_interp,
            bind_luma,
            bind_cut,
            bind_interp,
            prev_px,
            next_px,
            out_px,
            partials,
            cut_sum,
            luma_params,
            cut_params,
            interp_params,
            staging,
            cut_staging,
            pixels,
        })
    }

    /// The source-resolution buffer the synthesis kernel writes.
    pub(crate) fn out_px(&self) -> &Buffer {
        &self.out_px
    }

    /// Convert two frames into canonical RGBA pixels and upload them.
    ///
    /// Taking the scratch pixels and the target buffer as separate arguments
    /// keeps the two field borrows disjoint.
    fn pack_into(pixels: &mut [u32], dims: Dims, frame: &Frame, queue: &Queue, target: &Buffer) {
        let stride = dims.width as usize * 4;
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(pixels);
        for y in 0..dims.height as usize {
            let src = frame.row(y as u32);
            convert::expand_row(src, frame.fmt, &mut bytes[y * stride..(y + 1) * stride]);
        }
        queue.write_buffer(target, 0, bytemuck::cast_slice(pixels));
    }

    /// Upload the frame pair. Both frames must have this stage's dimensions.
    pub(crate) fn pack(&mut self, prev: &Frame, next: &Frame, queue: &Queue) {
        let dims = self.dims;
        Self::pack_into(&mut self.pixels, dims, prev, queue, &self.prev_px);
        Self::pack_into(&mut self.pixels, dims, next, queue, &self.next_px);
    }

    /// Record the luma pre-pass that feeds the flow pyramid.
    pub(crate) fn encode_luma(&self, encoder: &mut CommandEncoder) {
        let grid = groups(self.dims.width, self.dims.height);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fluid-gpu interp luma"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline_luma);
        pass.set_bind_group(0, &self.bind_luma, &[]);
        pass.dispatch_workgroups(grid.0, grid.1, 1);
    }

    /// Record the scene-change reduction and copy its sum to the readback slot.
    pub(crate) fn encode_cut(&self, encoder: &mut CommandEncoder) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fluid-gpu interp cut"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_cut);
            pass.set_bind_group(0, &self.bind_cut, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.cut_sum, 0, &self.cut_staging, 0, 4);
    }

    /// Read the scene-change sum back and decide whether this pair is a cut.
    ///
    /// Call only after the command buffer holding [Synth::encode_cut] has been
    /// submitted.
    pub(crate) fn scene_cut(&mut self, device: &Device) -> Result<bool> {
        let cut_slice = self.cut_staging.slice(..);
        cut_slice.map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|error| {
                FluidError::internal(format!("gpu interpolation: device poll failed: {error}"))
            })?;
        let cut_scaled = {
            let view = cut_slice.get_mapped_range().map_err(|error| {
                FluidError::internal(format!("gpu interpolation: cut readback failed: {error}"))
            })?;
            let bytes: &[u8] = &view;
            if bytes.len() < 4 {
                return Err(FluidError::internal(
                    "gpu interpolation: cut readback is short",
                ));
            }
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        };
        self.cut_staging.unmap();

        let area = self.dims.area();
        let mean_diff = f64::from(cut_scaled) / (64.0 * area as f64);
        Ok(mean_diff > 2.0 * f64::from(self.scene_cut_threshold))
    }

    /// Set the interpolation alpha (0..=1) for the next synthesis dispatch.
    pub(crate) fn write_alpha(&self, queue: &Queue, alpha: f32) {
        let params = words(
            [self.dims.width, self.dims.height, 0, 0],
            [0; 4],
            [u32::from(self.occlusion_fill), 0, 0, 0],
            [alpha, self.blend, MIN_CONFIDENCE, 1.0 - alpha],
            [0.0; 4],
        );
        queue.write_buffer(&self.interp_params, 0, bytemuck::cast_slice(&params));
    }

    /// Record the motion-compensated synthesis into [Synth::out_px].
    pub(crate) fn encode_synth(&self, encoder: &mut CommandEncoder) {
        let grid = groups(self.dims.width, self.dims.height);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fluid-gpu interp synth"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline_interp);
        pass.set_bind_group(0, &self.bind_interp, &[]);
        pass.dispatch_workgroups(grid.0, grid.1, 1);
    }

    /// Record the copy from the synthesis buffer to the source-resolution
    /// readback slot.
    pub(crate) fn encode_copy_out(&self, encoder: &mut CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.out_px, 0, &self.staging, 0, self.frame_bytes);
    }

    /// Wait for the submitted commands and convert the staged source frame.
    pub(crate) fn read_out(&mut self, device: &Device, fmt: PixelFormat) -> Result<Frame> {
        let slice = self.staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|error| {
                FluidError::internal(format!("gpu interpolation: device poll failed: {error}"))
            })?;

        let mut frame = Frame::zeroed(self.dims, fmt);
        {
            let view = slice.get_mapped_range().map_err(|error| {
                FluidError::internal(format!("gpu interpolation: readback failed: {error}"))
            })?;
            let bytes: &[u8] = &view;
            let stride = self.dims.width as usize * 4;
            for y in 0..self.dims.height as usize {
                let row = &bytes[y * stride..(y + 1) * stride];
                convert::shrink_row(row, fmt, frame.row_mut(y as u32));
            }
        }
        self.staging.unmap();
        Ok(frame)
    }
}
