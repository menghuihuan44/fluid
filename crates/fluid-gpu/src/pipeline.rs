//! The wgpu compute pipeline: persistent buffers, bind groups and the pass
//! schedule that mirrors \`fluid-upscale\`'s CPU order of operations.
//!
//! One [\`Pipeline\`] is built per [\`crate::GpuUpscaler\`] and owns every GPU
//! resource it will ever need. \`upscale\` only writes the source bytes into the
//! existing upload buffer, records the compute passes and reads the result back,
//! so steady state performs no resource allocation.
//!
//! Canonical pixel format: a packed \`u32\` per pixel in RGBA8 byte order
//! (\`r | g << 8 | b << 16 | a << 24\`). The kernels run the CPU's integer and
//! fixed-point arithmetic on those bytes, which is what keeps the GPU output
//! byte-identical for nearest/identity and within one LSB for the filters whose
//! CPU reference accumulates with a different multiply/add order.
//!
//! Pass order (identical to the CPU upscaler):
//! 1. spatial: identity copy, nearest copy, or horizontal + vertical passes
//!    followed by the adaptive detail restore,
//! 2. temporal blend, which keeps the pre-sharpen frame as history,
//! 3. sharpening, which reads the pre-sharpen frame and writes a different one.

use fluid_core::{Dims, FluidError, Frame, PixelFormat, Result};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, Buffer, BufferDescriptor, BufferUsages,
    CommandEncoder, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device,
    Queue, ShaderModule, ShaderModuleDescriptor, ShaderSource,
};

use crate::convert;
use crate::device::block_on;
use crate::taps::{self, Kernel};

/// WGSL sources: every module shares the parameter block and byte helpers.
const SHADER_COPY: &str = concat!(
    include_str!("shader_common.wgsl"),
    include_str!("shader_copy.wgsl")
);
const SHADER_NEAREST: &str = concat!(
    include_str!("shader_common.wgsl"),
    include_str!("shader_nearest.wgsl")
);
const SHADER_H: &str = concat!(
    include_str!("shader_common.wgsl"),
    include_str!("shader_h.wgsl")
);
const SHADER_V: &str = concat!(
    include_str!("shader_common.wgsl"),
    include_str!("shader_v.wgsl")
);
const SHADER_REFINE: &str = concat!(
    include_str!("shader_common.wgsl"),
    include_str!("shader_refine.wgsl")
);
const SHADER_SHARPEN: &str = concat!(
    include_str!("shader_common.wgsl"),
    include_str!("shader_sharpen.wgsl")
);
const SHADER_BLEND: &str = concat!(
    include_str!("shader_common.wgsl"),
    include_str!("shader_blend.wgsl")
);
const SHADER_TEMPORAL_WEIGHTS: &str = concat!(
    include_str!("shader_common.wgsl"),
    include_str!("shader_temporal_weights.wgsl")
);

/// Bytes per canonical texel.
const TEXEL: u64 = 4;

/// Destination tile size of the motion grid used by the temporal blend, in
/// pixels. Matches MOTION_TILE in fluid-upscale's temporal stage.
const MOTION_TILE: u32 = 16;

/// Which spatial resampler the kernel configuration asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Spatial {
    /// Source and destination dimensions match: an exact copy, never a resample.
    Identity,
    /// Nearest neighbour: two index tables, one pass.
    Nearest,
    /// Separable windowed kernel, optionally with the adaptive detail restore.
    Separable { kernel: Kernel, refine: bool },
}

/// How the temporal stage behaves for the current blend factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TemporalMode {
    /// Blend factor maps to a full-current frame: history becomes the spatial frame.
    Identity,
    /// Blend factor saturates to zero: the frame freezes to the history frame.
    Freeze,
    /// Regular exponential blend written in place over the history buffer.
    Normal,
}

/// Static description of one upscaler instance.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Config {
    pub src: Dims,
    pub dst: Dims,
    pub spatial: Spatial,
    pub sharpen_q8: i32,
    pub temporal: Option<TemporalMode>,
    pub blend_s: i32,
}

/// A destination-sized buffer in the ping-pong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Buf {
    Out0,
    Out1,
    History,
}

/// Persistent GPU state for one configured upscaler.
pub(crate) struct Pipeline {
    src: Dims,
    dst: Dims,
    src_bytes: u64,
    dst_bytes: u64,
    spatial: Spatial,
    sharpen_q8: i32,
    temporal: Option<TemporalMode>,

    params: Buffer,
    /// Source texels. Owned for the regular upscaler, which uploads each frame;
    /// in the fused path this is a clone of the synthesis output buffer, so the
    /// bind groups and the writer always refer to the same allocation.
    upload: Buffer,
    /// True when 'upload' belongs to the caller and must not be written here.
    external_source: bool,
    out0: Buffer,
    out1: Buffer,
    scratch: Option<Buffer>,
    history: Option<Buffer>,
    /// Per-tile current-frame weights for the motion-adaptive temporal blend,
    /// one u32 per 16 x 16 pixel tile of the destination frame.
    motion: Option<Buffer>,
    /// Motion grid dimensions, in tiles.
    motion_cols: u32,
    motion_rows: u32,
    staging: Buffer,

    pipelines: Pipelines,
    bg_copy: Option<BindGroup>,
    bg_nearest: Option<BindGroup>,
    bg_h: Option<BindGroup>,
    bg_v: Option<BindGroup>,
    bg_refine: Option<BindGroup>,
    bg_sharpen_history: Option<BindGroup>,
    bg_sharpen_out0: Option<BindGroup>,
    bg_sharpen_out1: Option<BindGroup>,
    bg_blend_out0: Option<BindGroup>,
    bg_blend_out1: Option<BindGroup>,
    bg_weights_out0: Option<BindGroup>,
    bg_weights_out1: Option<BindGroup>,

    history_valid: bool,
}

struct Pipelines {
    copy: ComputePipeline,
    nearest: ComputePipeline,
    h: ComputePipeline,
    v: ComputePipeline,
    refine: ComputePipeline,
    sharpen: ComputePipeline,
    blend: ComputePipeline,
    temporal_weights: ComputePipeline,
}

impl Pipeline {
    /// Allocate every resource for this configuration.
    pub(crate) fn new(device: &Device, queue: &Queue, config: Config) -> Result<Self> {
        Self::build(device, queue, config, None)
    }

    /// Build a pipeline that reads its source texels from an existing GPU
    /// buffer instead of the per-frame upload buffer.
    ///
    /// The fused interpolation + upscaling path uses this: the synthesis kernel
    /// writes the intermediate frame at the source resolution straight into
    /// 'source', so the upscaling passes run on it without a CPU round trip.
    pub(crate) fn with_source(
        device: &Device,
        queue: &Queue,
        config: Config,
        source: &Buffer,
    ) -> Result<Self> {
        Self::build(device, queue, config, Some(source))
    }

    /// Shared constructor: 'source' is None for an owned upload buffer.
    fn build(
        device: &Device,
        queue: &Queue,
        config: Config,
        source: Option<&Buffer>,
    ) -> Result<Self> {
        if config.src.is_empty() || config.dst.is_empty() {
            return Err(FluidError::invalid(
                "gpu upscale: source and destination must be non-empty",
            ));
        }
        let src_bytes = config.src.area() * TEXEL;
        let dst_bytes = config.dst.area() * TEXEL;
        if src_bytes > u64::from(u32::MAX) || dst_bytes > u64::from(u32::MAX) {
            return Err(FluidError::invalid(
                "gpu upscale: frame is larger than the 4 GiB buffer limit",
            ));
        }

        // Every kernel is built up front, inside one validation error scope, so a
        // shader typo fails GpuUpscaler::new with a clear FluidError instead of
        // panicking inside wgpu's default uncaptured-error handler. That is what
        // lets fluid-engine fall back to the CPU upscaler.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipelines = Pipelines {
            copy: make_pipeline(device, "fluid-gpu copy", SHADER_COPY)?,
            nearest: make_pipeline(device, "fluid-gpu nearest", SHADER_NEAREST)?,
            h: make_pipeline(device, "fluid-gpu horizontal", SHADER_H)?,
            v: make_pipeline(device, "fluid-gpu vertical", SHADER_V)?,
            refine: make_pipeline(device, "fluid-gpu refine", SHADER_REFINE)?,
            sharpen: make_pipeline(device, "fluid-gpu sharpen", SHADER_SHARPEN)?,
            blend: make_pipeline(device, "fluid-gpu blend", SHADER_BLEND)?,
            temporal_weights: make_pipeline(
                device,
                "fluid-gpu temporal weights",
                SHADER_TEMPORAL_WEIGHTS,
            )?,
        };

        let upload = match source {
            Some(source) => source.clone(),
            None => device.create_buffer(&BufferDescriptor {
                label: Some("fluid-gpu upload"),
                size: src_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };
        let external_source = source.is_some();
        let out0 = device.create_buffer(&BufferDescriptor {
            label: Some("fluid-gpu out0"),
            size: dst_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out1 = device.create_buffer(&BufferDescriptor {
            label: Some("fluid-gpu out1"),
            size: dst_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some("fluid-gpu staging"),
            size: dst_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scratch = match config.spatial {
            Spatial::Separable { .. } => Some(device.create_buffer(&BufferDescriptor {
                label: Some("fluid-gpu scratch"),
                size: u64::from(config.dst.width) * u64::from(config.src.height) * TEXEL,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            })),
            _ => None,
        };
        let history = config.temporal.map(|_| {
            device.create_buffer(&BufferDescriptor {
                label: Some("fluid-gpu history"),
                size: dst_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        // The motion-adaptive blend stores one twelve-bit weight per 16 x 16
        // tile, so the extra buffer is a few kilobytes even at 4K.
        let motion_cols = config.dst.width.div_ceil(MOTION_TILE).max(1);
        let motion_rows = config.dst.height.div_ceil(MOTION_TILE).max(1);
        let motion = config.temporal.map(|_| {
            device.create_buffer(&BufferDescriptor {
                label: Some("fluid-gpu temporal weights"),
                size: u64::from(motion_cols) * u64::from(motion_rows) * 4,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        });

        // Tap tables: at most six taps per output coordinate, uploaded once.
        let mut nx = 1u32;
        let mut ny = 1u32;
        let mut tap_buffers = TapBuffers::default();
        match config.spatial {
            Spatial::Identity => {}
            Spatial::Nearest => {
                let h = taps::build(Kernel::Nearest, config.src.width, config.dst.width);
                let v = taps::build(Kernel::Nearest, config.src.height, config.dst.height);
                nx = h.n;
                ny = v.n;
                tap_buffers.h_idx = Some(upload_u32(device, queue, "fluid-gpu h index", &h.idx));
                tap_buffers.v_idx = Some(upload_u32(device, queue, "fluid-gpu v index", &v.idx));
            }
            Spatial::Separable { kernel, .. } => {
                let h = taps::build(kernel, config.src.width, config.dst.width);
                let v = taps::build(kernel, config.src.height, config.dst.height);
                nx = h.n;
                ny = v.n;
                tap_buffers.h_idx = Some(upload_u32(device, queue, "fluid-gpu h index", &h.idx));
                tap_buffers.h_w = Some(upload_f32(device, queue, "fluid-gpu h weights", &h.w));
                tap_buffers.v_idx = Some(upload_u32(device, queue, "fluid-gpu v index", &v.idx));
                tap_buffers.v_w = Some(upload_f32(device, queue, "fluid-gpu v weights", &v.w));
            }
        }

        let params = device.create_buffer(&BufferDescriptor {
            label: Some("fluid-gpu params"),
            size: 48,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Bicubic and Lanczos3 (and the adaptive upscaler that uses the bicubic
        // base) enable the anti-ringing clamp in the vertical pass. It is part
        // of the shared parameter block because every kernel reads the same
        // buffer and only the vertical pass looks at this bit.
        let antiring = matches!(
            config.spatial,
            Spatial::Separable {
                kernel: Kernel::Bicubic | Kernel::Lanczos3,
                ..
            }
        );
        // Slots 9 and 10 carry the motion grid dimensions, which the temporal
        // weight and blend kernels need to index the per-tile weight buffer.
        let params_data: [u32; 12] = [
            config.src.width,
            config.src.height,
            config.dst.width,
            config.dst.height,
            nx,
            ny,
            config.sharpen_q8 as u32,
            config.blend_s as u32,
            u32::from(antiring),
            motion_cols,
            motion_rows,
            0,
        ];
        queue.write_buffer(&params, 0, bytemuck::cast_slice(&params_data));

        let mut pipeline = Self {
            src: config.src,
            dst: config.dst,
            src_bytes,
            dst_bytes,
            spatial: config.spatial,
            sharpen_q8: config.sharpen_q8,
            temporal: config.temporal,
            params,
            upload,
            external_source,
            out0,
            out1,
            scratch,
            history,
            motion,
            motion_cols,
            motion_rows,
            staging,
            pipelines,
            bg_copy: None,
            bg_nearest: None,
            bg_h: None,
            bg_v: None,
            bg_refine: None,
            bg_sharpen_history: None,
            bg_sharpen_out0: None,
            bg_sharpen_out1: None,
            bg_blend_out0: None,
            bg_blend_out1: None,
            bg_weights_out0: None,
            bg_weights_out1: None,
            history_valid: false,
        };
        pipeline.build_bind_groups(device, &tap_buffers)?;
        if let Some(error) = block_on(scope.pop()) {
            return Err(FluidError::internal(format!(
                "gpu upscale: shader, pipeline or bind group validation failed: {error}"
            )));
        }
        Ok(pipeline)
    }

    /// Create exactly the bind groups the configured pass schedule uses.
    fn build_bind_groups(&mut self, device: &Device, taps: &TapBuffers) -> Result<()> {
        match self.spatial {
            Spatial::Identity => {
                self.bg_copy = Some(make_bind_group(
                    device,
                    &self.pipelines.copy,
                    "fluid-gpu copy",
                    &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.upload.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: self.out0.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: self.params.as_entire_binding(),
                        },
                    ],
                ));
            }
            Spatial::Nearest => {
                let (Some(h_idx), Some(v_idx)) = (&taps.h_idx, &taps.v_idx) else {
                    return Err(FluidError::internal(
                        "gpu upscale: nearest taps were not built",
                    ));
                };
                self.bg_nearest = Some(make_bind_group(
                    device,
                    &self.pipelines.nearest,
                    "fluid-gpu nearest",
                    &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.upload.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: h_idx.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: v_idx.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: self.out0.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: self.params.as_entire_binding(),
                        },
                    ],
                ));
            }
            Spatial::Separable { refine, .. } => {
                let (Some(h_idx), Some(h_w), Some(v_idx), Some(v_w)) =
                    (&taps.h_idx, &taps.h_w, &taps.v_idx, &taps.v_w)
                else {
                    return Err(FluidError::internal(
                        "gpu upscale: separable taps were not built",
                    ));
                };
                let scratch = self
                    .scratch
                    .as_ref()
                    .ok_or_else(|| FluidError::internal("gpu upscale: missing scratch buffer"))?;
                self.bg_h = Some(make_bind_group(
                    device,
                    &self.pipelines.h,
                    "fluid-gpu horizontal",
                    &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.upload.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: h_idx.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: h_w.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: scratch.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: self.params.as_entire_binding(),
                        },
                    ],
                ));
                self.bg_v = Some(make_bind_group(
                    device,
                    &self.pipelines.v,
                    "fluid-gpu vertical",
                    &[
                        BindGroupEntry {
                            binding: 0,
                            resource: scratch.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: v_idx.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: v_w.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: self.out0.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: self.params.as_entire_binding(),
                        },
                        // The anti-ringing clamp compares against the source
                        // luma of the horizontal taps, so the vertical pass sees
                        // the same source buffer and horizontal index table the
                        // horizontal pass used.
                        BindGroupEntry {
                            binding: 5,
                            resource: self.upload.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 6,
                            resource: h_idx.as_entire_binding(),
                        },
                    ],
                ));
                if refine {
                    self.bg_refine = Some(make_bind_group(
                        device,
                        &self.pipelines.refine,
                        "fluid-gpu refine",
                        &[
                            BindGroupEntry {
                                binding: 0,
                                resource: self.out0.as_entire_binding(),
                            },
                            BindGroupEntry {
                                binding: 1,
                                resource: self.out1.as_entire_binding(),
                            },
                            BindGroupEntry {
                                binding: 2,
                                resource: self.params.as_entire_binding(),
                            },
                        ],
                    ));
                }
            }
        }

        if let Some(history) = &self.history {
            self.bg_sharpen_history = Some(make_bind_group(
                device,
                &self.pipelines.sharpen,
                "fluid-gpu sharpen history",
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: history.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: self.out0.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: self.params.as_entire_binding(),
                    },
                ],
            ));
            let motion = self
                .motion
                .as_ref()
                .ok_or_else(|| FluidError::internal("gpu upscale: missing temporal weights"))?;
            // The motion estimator reads the spatial frame and the history and
            // writes the per-tile weights.
            self.bg_weights_out0 = Some(make_bind_group(
                device,
                &self.pipelines.temporal_weights,
                "fluid-gpu weights out0",
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: self.out0.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: history.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: motion.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: self.params.as_entire_binding(),
                    },
                ],
            ));
            self.bg_weights_out1 = Some(make_bind_group(
                device,
                &self.pipelines.temporal_weights,
                "fluid-gpu weights out1",
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: self.out1.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: history.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: motion.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: self.params.as_entire_binding(),
                    },
                ],
            ));
            self.bg_blend_out0 = Some(make_bind_group(
                device,
                &self.pipelines.blend,
                "fluid-gpu blend out0",
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: self.out0.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: history.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: motion.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: self.params.as_entire_binding(),
                    },
                ],
            ));
            self.bg_blend_out1 = Some(make_bind_group(
                device,
                &self.pipelines.blend,
                "fluid-gpu blend out1",
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: self.out1.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: history.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: motion.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: self.params.as_entire_binding(),
                    },
                ],
            ));
        }
        self.bg_sharpen_out0 = Some(make_bind_group(
            device,
            &self.pipelines.sharpen,
            "fluid-gpu sharpen out0",
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.out0.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.out1.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.params.as_entire_binding(),
                },
            ],
        ));
        self.bg_sharpen_out1 = Some(make_bind_group(
            device,
            &self.pipelines.sharpen,
            "fluid-gpu sharpen out1",
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.out1.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.out0.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.params.as_entire_binding(),
                },
            ],
        ));
        Ok(())
    }

    /// Drop the temporal history (scene cut / resolution change).
    pub(crate) fn reset_history(&mut self) {
        self.history_valid = false;
    }

    /// Run one frame: upload, dispatch, read back into \`out\`.
    ///
    /// \`out\` must already have the destination dimensions and \`out.fmt\`; the
    /// caller validates that. \`packed\` holds \`src.area()\` canonical texels.
    pub(crate) fn execute(
        &mut self,
        device: &Device,
        queue: &Queue,
        packed: &[u32],
        out: &mut Frame,
    ) -> Result<()> {
        if self.external_source {
            return Err(FluidError::internal(
                "gpu upscale: this pipeline reads its source from a GPU buffer",
            ));
        }
        if packed.len() as u64 * TEXEL != self.src_bytes {
            return Err(FluidError::internal(
                "gpu upscale: upload buffer does not match the source size",
            ));
        }
        queue.write_buffer(&self.upload, 0, bytemuck::cast_slice(packed));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fluid-gpu frame"),
        });
        self.encode_frame(&mut encoder);
        queue.submit(Some(encoder.finish()));

        self.read_back(device, out)
    }

    /// Record the upscaling passes and stage the finished frame, without
    /// submitting.
    ///
    /// The fused pipeline records this after its synthesis dispatch in the very
    /// same command buffer, so one submit produces one finished destination
    /// frame.
    pub(crate) fn encode_frame(&mut self, encoder: &mut CommandEncoder) {
        let final_buf = self.encode(encoder);
        encoder.copy_buffer_to_buffer(self.buffer(final_buf), 0, &self.staging, 0, self.dst_bytes);
    }

    /// Wait for the submitted passes and copy the staged destination frame into
    /// 'out'.
    pub(crate) fn finish_frame(&self, device: &Device, out: &mut Frame) -> Result<()> {
        self.read_back(device, out)
    }

    /// Record the compute passes and return the buffer holding the final frame.
    fn encode(&mut self, encoder: &mut CommandEncoder) -> Buf {
        let mut spatial = Buf::Out0;
        match self.spatial {
            Spatial::Identity => {
                let bind = self.bg_copy.as_ref().expect("copy bind group");
                dispatch_2d(
                    encoder,
                    &self.pipelines.copy,
                    bind,
                    self.dst.width,
                    self.dst.height,
                );
            }
            Spatial::Nearest => {
                let bind = self.bg_nearest.as_ref().expect("nearest bind group");
                dispatch_2d(
                    encoder,
                    &self.pipelines.nearest,
                    bind,
                    self.dst.width,
                    self.dst.height,
                );
            }
            Spatial::Separable { refine, .. } => {
                let h = self.bg_h.as_ref().expect("horizontal bind group");
                dispatch_2d(
                    encoder,
                    &self.pipelines.h,
                    h,
                    self.dst.width,
                    self.src.height,
                );
                let v = self.bg_v.as_ref().expect("vertical bind group");
                dispatch_2d(
                    encoder,
                    &self.pipelines.v,
                    v,
                    self.dst.width,
                    self.dst.height,
                );
                if refine {
                    let bind = self.bg_refine.as_ref().expect("refine bind group");
                    dispatch_2d(
                        encoder,
                        &self.pipelines.refine,
                        bind,
                        self.dst.width,
                        self.dst.height,
                    );
                    spatial = Buf::Out1;
                }
            }
        }

        let mut pre = spatial;
        if let Some(mode) = self.temporal {
            let history = self
                .history
                .as_ref()
                .expect("temporal mode without a history buffer");
            if !self.history_valid {
                // First frame after construction/reset: the spatial frame seeds
                // the history and is output unchanged, exactly like the CPU.
                encoder.copy_buffer_to_buffer(self.buffer(spatial), 0, history, 0, self.dst_bytes);
                self.history_valid = true;
            } else {
                match mode {
                    TemporalMode::Identity => {
                        encoder.copy_buffer_to_buffer(
                            self.buffer(spatial),
                            0,
                            history,
                            0,
                            self.dst_bytes,
                        );
                    }
                    TemporalMode::Freeze => pre = Buf::History,
                    TemporalMode::Normal => {
                        // The CPU's motion-adaptive blend first measures how far
                        // each 16 x 16 tile moved, then blends each tile with its
                        // own weight. Both passes are required: the weights make
                        // a fast-moving tile follow the current frame instead of
                        // trailing a ghost of the old one.
                        let (weights_bind, blend_bind) = match spatial {
                            Buf::Out0 => {
                                (self.bg_weights_out0.as_ref(), self.bg_blend_out0.as_ref())
                            }
                            _ => (self.bg_weights_out1.as_ref(), self.bg_blend_out1.as_ref()),
                        };
                        let weights_bind = weights_bind.expect("temporal weights bind group");
                        dispatch_2d(
                            encoder,
                            &self.pipelines.temporal_weights,
                            weights_bind,
                            self.motion_cols,
                            self.motion_rows,
                        );
                        let bind = blend_bind.expect("blend bind group");
                        dispatch_2d(
                            encoder,
                            &self.pipelines.blend,
                            bind,
                            self.dst.width,
                            self.dst.height,
                        );
                        pre = Buf::History;
                    }
                }
            }
        }

        if self.sharpen_q8 > 0 {
            let (bind, target) = match pre {
                Buf::History => (
                    self.bg_sharpen_history
                        .as_ref()
                        .expect("sharpen bind group"),
                    Buf::Out0,
                ),
                Buf::Out0 => (
                    self.bg_sharpen_out0.as_ref().expect("sharpen bind group"),
                    Buf::Out1,
                ),
                Buf::Out1 => (
                    self.bg_sharpen_out1.as_ref().expect("sharpen bind group"),
                    Buf::Out0,
                ),
            };
            dispatch_2d(
                encoder,
                &self.pipelines.sharpen,
                bind,
                self.dst.width,
                self.dst.height,
            );
            return target;
        }
        pre
    }

    fn buffer(&self, which: Buf) -> &Buffer {
        match which {
            Buf::Out0 => &self.out0,
            Buf::Out1 => &self.out1,
            Buf::History => self.history.as_ref().expect("history buffer"),
        }
    }

    /// Wait for the frame, map the staging buffer and convert it into \`out\`.
    fn read_back(&self, device: &Device, out: &mut Frame) -> Result<()> {
        let slice = self.staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        if let Err(error) = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        }) {
            return Err(FluidError::internal(format!(
                "gpu upscale: waiting for the readback failed: {error}"
            )));
        }
        let fmt = out_fmt(out);
        // The staging buffer always holds canonical RGBA8 texels, so its row
        // pitch is four bytes per pixel even when 'out' is packed (Rgb8/Luma8).
        let canonical_stride = self.dst.width as usize * TEXEL as usize;
        {
            let view = slice.get_mapped_range().map_err(|error| {
                FluidError::internal(format!("gpu upscale: mapping the readback failed: {error}"))
            })?;
            for y in 0..self.dst.height as usize {
                let row = &view[y * canonical_stride..(y + 1) * canonical_stride];
                convert::shrink_row(row, fmt, out.row_mut(y as u32));
            }
        }
        self.staging.unmap();
        Ok(())
    }
}

fn out_fmt(out: &Frame) -> PixelFormat {
    out.fmt
}

/// Read-only tap table buffers, uploaded once at construction.
#[derive(Default)]
struct TapBuffers {
    h_idx: Option<Buffer>,
    h_w: Option<Buffer>,
    v_idx: Option<Buffer>,
    v_w: Option<Buffer>,
}

fn upload_u32(device: &Device, queue: &Queue, label: &str, data: &[u32]) -> Buffer {
    let bytes = bytemuck::cast_slice(data);
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

fn upload_f32(device: &Device, queue: &Queue, label: &str, data: &[f32]) -> Buffer {
    let bytes = bytemuck::cast_slice(data);
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

fn make_module(device: &Device, label: &str, source: &'static str) -> ShaderModule {
    device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(source.into()),
    })
}

pub(crate) fn make_pipeline(
    device: &Device,
    label: &str,
    source: &'static str,
) -> Result<ComputePipeline> {
    let module = make_module(device, label, source);
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    Ok(pipeline)
}

fn make_bind_group(
    device: &Device,
    pipeline: &ComputePipeline,
    label: &str,
    entries: &[BindGroupEntry<'_>],
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some(label),
        layout: &pipeline.get_bind_group_layout(0),
        entries,
    })
}

fn dispatch_2d(
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    bind: &BindGroup,
    width: u32,
    height: u32,
) {
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind, &[]);
    pass.dispatch_workgroups(width.div_ceil(16).max(1), height.div_ceil(16).max(1), 1);
}
