//! GPU optical-flow estimator.
//!
//! 'fluid-flow' is the reference implementation and this module reproduces its
//! algorithm in compute shaders: a 2x box pyramid, a decimated SAD coarse search
//! over the full radius, a 3x3 patch refinement followed by a separable
//! parabolic fit, a damped Gauss-Newton sub-pixel pass, a dense field expansion
//! and a forward-backward consistency confidence map. Keeping the arithmetic
//! identical is what makes the synthesised intermediate frames match the CPU
//! interpolation output closely enough to be drop-in compatible.
//!
//! The reduced-resolution policy of 'fluid-flow' is reproduced as well: frames
//! whose block grid exceeds the 12k block budget are matched on a half-resolution
//! pyramid level (the dense field is still expanded at full resolution), and the
//! `FLUID_FULL_RES_FLOW` escape hatch forces full resolution on both sides. All
//! pyramids, gradient planes and grids are allocated once when the estimator is
//! created; estimating a frame pair only uploads the two luma planes and submits
//! one command buffer.

use std::sync::Arc;
use std::time::Instant;

use fluid_core::{Dims, FlowConfig, FluidError, LumaPlane, Result};
use fluid_flow::{FlowEstimate, FlowField};
use wgpu::BufferUsages;

use crate::GpuContext;
use crate::device;

/// Compute tile edge; every kernel is dispatched as a 2D grid of these tiles.
const TILE: u32 = 16;

const SHADER_DOWN: &str = concat!(
    include_str!("flow_common.wgsl"),
    include_str!("shader_flow_down.wgsl")
);
const SHADER_GRAD: &str = concat!(
    include_str!("flow_common.wgsl"),
    include_str!("shader_flow_grad.wgsl")
);
const SHADER_COARSE: &str = concat!(
    include_str!("flow_common.wgsl"),
    include_str!("shader_flow_coarse.wgsl")
);
const SHADER_REFINE: &str = concat!(
    include_str!("flow_common.wgsl"),
    include_str!("shader_flow_refine.wgsl")
);
const SHADER_LK: &str = concat!(
    include_str!("flow_common.wgsl"),
    include_str!("shader_flow_lk.wgsl")
);
const SHADER_FILL: &str = concat!(
    include_str!("flow_common.wgsl"),
    include_str!("shader_flow_fill.wgsl")
);
const SHADER_CONF_GRID: &str = concat!(
    include_str!("flow_common.wgsl"),
    include_str!("shader_flow_conf_grid.wgsl")
);
const SHADER_CONF_UPSAMPLE: &str = concat!(
    include_str!("flow_common.wgsl"),
    include_str!("shader_flow_conf_upsample.wgsl")
);

/// Geometry of one pyramid level.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LevelDims {
    pub(crate) w: u32,
    pub(crate) h: u32,
    pub(crate) cols: u32,
    pub(crate) rows: u32,
}

/// Which compute pipeline a recorded pass runs.
#[derive(Clone, Copy)]
enum Kind {
    Down,
    Grad,
    Coarse,
    Refine,
    Lk,
    Fill,
    ConfGrid,
    ConfUpsample,
}

/// One recorded dispatch: pipeline, bind group and 2D workgroup grid.
struct Pass {
    kind: Kind,
    bind: wgpu::BindGroup,
    groups: (u32, u32),
}

struct Pipelines {
    down: wgpu::ComputePipeline,
    grad: wgpu::ComputePipeline,
    coarse: wgpu::ComputePipeline,
    refine: wgpu::ComputePipeline,
    lk: wgpu::ComputePipeline,
    fill: wgpu::ComputePipeline,
    conf_grid: wgpu::ComputePipeline,
    conf_upsample: wgpu::ComputePipeline,
}

impl Pipelines {
    fn get(&self, kind: Kind) -> &wgpu::ComputePipeline {
        match kind {
            Kind::Down => &self.down,
            Kind::Grad => &self.grad,
            Kind::Coarse => &self.coarse,
            Kind::Refine => &self.refine,
            Kind::Lk => &self.lk,
            Kind::Fill => &self.fill,
            Kind::ConfGrid => &self.conf_grid,
            Kind::ConfUpsample => &self.conf_upsample,
        }
    }
}

/// Luma planes, gradient planes, block grids and dense fields for one frame
/// size, with the complete pass schedule recorded once at construction.
pub(crate) struct FlowRig {
    pub(crate) prev_luma: Vec<wgpu::Buffer>,
    pub(crate) next_luma: Vec<wgpu::Buffer>,
    pub(crate) dense_forward: wgpu::Buffer,
    pub(crate) dense_backward: wgpu::Buffer,
    pub(crate) confidence: wgpu::Buffer,
    /// Half-resolution confidence grid, upsampled into 'confidence'.
    #[allow(dead_code)]
    conf_grid: wgpu::Buffer,
    pipelines: Pipelines,
    passes: Vec<Pass>,
    /// Uniform parameter buffers, kept alive for the lifetime of the schedules.
    #[allow(dead_code)]
    uniforms: Vec<wgpu::Buffer>,
}

pub(crate) fn groups(w: u32, h: u32) -> (u32, u32) {
    (w.div_ceil(TILE).max(1), h.div_ceil(TILE).max(1))
}

/// Full-resolution block count above which matching runs on a half-resolution
/// grid, identical to 'fluid-flow' LARGE_FRAME_BLOCKS.
const LARGE_FRAME_BLOCKS: u32 = 12_000;

/// Truthy environment switch: 1/true/yes (case-insensitive).
fn switch_truthy(v: &str) -> bool {
    let v = v.trim();
    v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
}

/// Escape hatch shared with 'fluid-flow': `FLUID_FULL_RES_FLOW=1` forces
/// full-resolution matching and confidence on both the CPU and the GPU side.
fn force_full_res() -> bool {
    std::env::var("FLUID_FULL_RES_FLOW")
        .ok()
        .map(|v| switch_truthy(&v))
        .unwrap_or(false)
}

/// True when the block grid is large enough for the reduced-resolution fast path.
fn reduced_resolution(dims: Dims, bs: u32) -> bool {
    if force_full_res() {
        return false;
    }
    let bs = bs.max(1);
    (dims.width / bs) * (dims.height / bs) > LARGE_FRAME_BLOCKS
}

/// Pyramid level at which block matching runs, and therefore the resolution of
/// the confidence grid: level 1 means half resolution for large frames.
fn target_level(half: bool, levels: usize) -> usize {
    if half {
        (levels.saturating_sub(1)).min(1)
    } else {
        0
    }
}

fn create_storage(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    extra: BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4),
        usage: BufferUsages::STORAGE | extra,
        mapped_at_creation: false,
    })
}

/// Pack the uniform parameter block: six 16-byte rows, floats as bit patterns.
pub(crate) fn words(
    dims: [u32; 4],
    cfg: [u32; 4],
    extra: [u32; 4],
    flt: [f32; 4],
    flt2: [f32; 4],
) -> [u32; 24] {
    let mut out = [0u32; 24];
    out[0..4].copy_from_slice(&dims);
    out[4..8].copy_from_slice(&cfg);
    out[8..12].copy_from_slice(&extra);
    out[12..16].fill(0);
    for (i, v) in flt.iter().enumerate() {
        out[16 + i] = v.to_bits();
    }
    for (i, v) in flt2.iter().enumerate() {
        out[20 + i] = v.to_bits();
    }
    out
}

pub(crate) fn params_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    words: [u32; 24],
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: 96,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&words));
    buffer
}

pub(crate) fn bind_group(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    label: &str,
    entries: &[(u32, &wgpu::Buffer)],
) -> wgpu::BindGroup {
    let bindings: Vec<wgpu::BindGroupEntry<'_>> = entries
        .iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &bindings,
    })
}

impl FlowRig {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dims: Dims,
        cfg: &FlowConfig,
    ) -> Result<Self> {
        if dims.is_empty() {
            return Err(FluidError::invalid("gpu flow: frame dimensions are empty"));
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let built = Self::build(device, queue, dims, cfg);
        let validation = device::block_on(scope.pop());
        match (built, validation) {
            (Err(error), _) => Err(error),
            (Ok(_), Some(error)) => Err(FluidError::internal(format!(
                "gpu flow: shader, pipeline or buffer validation failed: {error}"
            ))),
            (Ok(rig), None) => Ok(rig),
        }
    }

    fn build(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dims: Dims,
        cfg: &FlowConfig,
    ) -> Result<Self> {
        let bs = cfg.block_size.max(1);
        let radius = cfg.search_radius.max(1);
        let want = cfg.pyramid_levels.clamp(1, 5);
        let refine_iters = cfg.refine_iters.clamp(1, 4);

        let mut count = 1u32;
        {
            let (mut w, mut h) = (dims.width, dims.height);
            while count < want && w > bs && h > bs {
                w = (w / 2).max(1);
                h = (h / 2).max(1);
                count += 1;
            }
        }
        let mut levels: Vec<LevelDims> = Vec::with_capacity(count as usize);
        {
            let (mut w, mut h) = (dims.width, dims.height);
            for _ in 0..count {
                levels.push(LevelDims {
                    w,
                    h,
                    cols: (w / bs).max(1),
                    rows: (h / bs).max(1),
                });
                w = (w / 2).max(1);
                h = (h / 2).max(1);
            }
        }

        // Reduced-resolution matching policy of 'fluid-flow': large frames are
        // matched on the half-resolution pyramid level, the dense field and the
        // confidence map are still emitted at full resolution.
        let target = target_level(reduced_resolution(dims, bs), levels.len()).min(levels.len() - 1);
        let level_scale = (1u32 << target) as f32;
        let tl = levels[target];
        let inv = 1.0 / (level_scale * bs as f32);
        let gather = -level_scale;

        let area = dims.area();
        let limits = device.limits();
        let max_storage = limits.max_storage_buffer_binding_size;
        let max_buffer = limits.max_buffer_size;
        for (what, size) in [("dense field", area * 8), ("confidence", area * 4)] {
            if size > max_buffer || size > max_storage {
                return Err(FluidError::unsupported(format!(
                    "gpu flow: {what} buffer of {size} bytes exceeds the device limits"
                )));
            }
        }

        let pipelines = Pipelines {
            down: crate::pipeline::make_pipeline(device, "fluid-gpu flow down", SHADER_DOWN)?,
            grad: crate::pipeline::make_pipeline(device, "fluid-gpu flow grad", SHADER_GRAD)?,
            coarse: crate::pipeline::make_pipeline(device, "fluid-gpu flow coarse", SHADER_COARSE)?,
            refine: crate::pipeline::make_pipeline(device, "fluid-gpu flow refine", SHADER_REFINE)?,
            lk: crate::pipeline::make_pipeline(device, "fluid-gpu flow lk", SHADER_LK)?,
            fill: crate::pipeline::make_pipeline(device, "fluid-gpu flow fill", SHADER_FILL)?,
            conf_grid: crate::pipeline::make_pipeline(
                device,
                "fluid-gpu flow conf grid",
                SHADER_CONF_GRID,
            )?,
            conf_upsample: crate::pipeline::make_pipeline(
                device,
                "fluid-gpu flow conf upsample",
                SHADER_CONF_UPSAMPLE,
            )?,
        };

        let mut prev_luma: Vec<wgpu::Buffer> = Vec::with_capacity(levels.len());
        let mut next_luma: Vec<wgpu::Buffer> = Vec::with_capacity(levels.len());
        for level in &levels {
            let size = u64::from(level.w) * u64::from(level.h) * 4;
            prev_luma.push(create_storage(
                device,
                "fluid-gpu flow prev luma",
                size,
                BufferUsages::COPY_DST,
            ));
            next_luma.push(create_storage(
                device,
                "fluid-gpu flow next luma",
                size,
                BufferUsages::COPY_DST,
            ));
        }

        let l0 = levels[0];
        let grad_next_x = create_storage(
            device,
            "fluid-gpu flow grad next x",
            area * 4,
            BufferUsages::empty(),
        );
        let grad_next_y = create_storage(
            device,
            "fluid-gpu flow grad next y",
            area * 4,
            BufferUsages::empty(),
        );
        let grad_prev_x = create_storage(
            device,
            "fluid-gpu flow grad prev x",
            area * 4,
            BufferUsages::empty(),
        );
        let grad_prev_y = create_storage(
            device,
            "fluid-gpu flow grad prev y",
            area * 4,
            BufferUsages::empty(),
        );

        let mut grids: Vec<Vec<[wgpu::Buffer; 2]>> = Vec::with_capacity(2);
        for _ in 0..2 {
            let mut per_level: Vec<[wgpu::Buffer; 2]> = Vec::with_capacity(levels.len());
            for level in &levels {
                let size = u64::from(level.cols) * u64::from(level.rows) * 8;
                per_level.push([
                    create_storage(device, "fluid-gpu flow grid a", size, BufferUsages::empty()),
                    create_storage(device, "fluid-gpu flow grid b", size, BufferUsages::empty()),
                ]);
            }
            grids.push(per_level);
        }
        let lk_grids = [
            create_storage(
                device,
                "fluid-gpu flow lk forward",
                u64::from(l0.cols) * u64::from(l0.rows) * 8,
                BufferUsages::empty(),
            ),
            create_storage(
                device,
                "fluid-gpu flow lk backward",
                u64::from(l0.cols) * u64::from(l0.rows) * 8,
                BufferUsages::empty(),
            ),
        ];

        let dense_forward = create_storage(
            device,
            "fluid-gpu flow forward",
            area * 8,
            BufferUsages::COPY_SRC,
        );
        let dense_backward = create_storage(
            device,
            "fluid-gpu flow backward",
            area * 8,
            BufferUsages::COPY_SRC,
        );
        let confidence = create_storage(
            device,
            "fluid-gpu flow confidence",
            area * 4,
            BufferUsages::COPY_SRC,
        );
        let gw = dims.width.div_ceil(2);
        let gh = dims.height.div_ceil(2);
        let conf_grid = create_storage(
            device,
            "fluid-gpu flow confidence grid",
            u64::from(gw) * u64::from(gh) * 4,
            BufferUsages::empty(),
        );

        let mut passes: Vec<Pass> = Vec::with_capacity(2 * levels.len() + 3 * 2 + 6);
        let mut uniforms: Vec<wgpu::Buffer> = Vec::with_capacity(passes.capacity());

        // Pyramid: level l is a 2x box downscale of level l-1, for both frames.
        for l in 1..levels.len() {
            let src_level = levels[l - 1];
            let dst_level = levels[l];
            for (label, src, dst) in [
                ("fluid-gpu flow down prev", &prev_luma[l - 1], &prev_luma[l]),
                ("fluid-gpu flow down next", &next_luma[l - 1], &next_luma[l]),
            ] {
                let params = params_buffer(
                    device,
                    queue,
                    label,
                    words(
                        [dst_level.w, dst_level.h, 0, 0],
                        [0; 4],
                        [src_level.w, src_level.h, 0, 0],
                        [0.0; 4],
                        [0.0; 4],
                    ),
                );
                uniforms.push(params);
                let bind = bind_group(
                    device,
                    &pipelines.down,
                    label,
                    &[(0, src), (1, dst), (2, &uniforms[uniforms.len() - 1])],
                );
                passes.push(Pass {
                    kind: Kind::Down,
                    bind,
                    groups: groups(dst_level.w, dst_level.h),
                });
            }
        }

        // Central-difference gradient planes of the destination level, per direction.
        for (label, luma, out_x, out_y) in [
            (
                "fluid-gpu flow grad next",
                &next_luma[target],
                &grad_next_x,
                &grad_next_y,
            ),
            (
                "fluid-gpu flow grad prev",
                &prev_luma[target],
                &grad_prev_x,
                &grad_prev_y,
            ),
        ] {
            let params = params_buffer(
                device,
                queue,
                label,
                words([tl.w, tl.h, 0, 0], [0; 4], [0; 4], [0.0; 4], [0.0; 4]),
            );
            uniforms.push(params);
            let bind = bind_group(
                device,
                &pipelines.grad,
                label,
                &[
                    (0, luma),
                    (1, out_x),
                    (2, out_y),
                    (3, &uniforms[uniforms.len() - 1]),
                ],
            );
            passes.push(Pass {
                kind: Kind::Grad,
                bind,
                groups: groups(tl.w, tl.h),
            });
        }

        // Matching: coarse search, level-by-level refinement, sub-pixel pass, fill.
        let last = levels.len() - 1;
        for dir in 0..2usize {
            let (src_side, dst_side) = if dir == 0 {
                (&prev_luma, &next_luma)
            } else {
                (&next_luma, &prev_luma)
            };
            let (grad_x, grad_y) = if dir == 0 {
                (&grad_next_x, &grad_next_y)
            } else {
                (&grad_prev_x, &grad_prev_y)
            };
            let dir_grids = &grids[dir];

            let coarsest = levels[last];
            let label = if dir == 0 {
                "fluid-gpu flow coarse forward"
            } else {
                "fluid-gpu flow coarse backward"
            };
            let params = params_buffer(
                device,
                queue,
                label,
                words(
                    [coarsest.w, coarsest.h, coarsest.cols, coarsest.rows],
                    [bs, radius, 0, 0],
                    [0; 4],
                    [0.0; 4],
                    [0.0; 4],
                ),
            );
            uniforms.push(params);
            let bind = bind_group(
                device,
                &pipelines.coarse,
                label,
                &[
                    (0, &src_side[last]),
                    (1, &dst_side[last]),
                    (2, &dir_grids[last][0]),
                    (3, &uniforms[uniforms.len() - 1]),
                ],
            );
            passes.push(Pass {
                kind: Kind::Coarse,
                bind,
                groups: groups(coarsest.cols, coarsest.rows),
            });

            let mut current = dir_grids[last][0].clone();
            for new_level in (target..last).rev() {
                let old = levels[new_level + 1];
                let new = levels[new_level];
                let pass_count = if new_level == target { refine_iters } else { 1 };
                let patch_step = if new_level == target { 1 } else { 2 };
                let scale_x = new.w as f32 / old.w as f32;
                let scale_y = new.h as f32 / old.h as f32;
                for p in 0..pass_count {
                    let slot = (p as usize) % 2;
                    let label = if dir == 0 {
                        "fluid-gpu flow refine forward"
                    } else {
                        "fluid-gpu flow refine backward"
                    };
                    let params = params_buffer(
                        device,
                        queue,
                        label,
                        words(
                            [new.w, new.h, new.cols, new.rows],
                            [bs, patch_step, u32::from(p > 0), 0],
                            [old.cols, old.rows, 0, 0],
                            [scale_x, scale_y, 0.0, 0.0],
                            [0.0; 4],
                        ),
                    );
                    uniforms.push(params);
                    let bind = bind_group(
                        device,
                        &pipelines.refine,
                        label,
                        &[
                            (0, &src_side[new_level]),
                            (1, &dst_side[new_level]),
                            (2, &current),
                            (3, &dir_grids[new_level][slot]),
                            (4, &uniforms[uniforms.len() - 1]),
                        ],
                    );
                    passes.push(Pass {
                        kind: Kind::Refine,
                        bind,
                        groups: groups(new.cols, new.rows),
                    });
                    current = dir_grids[new_level][slot].clone();
                }
            }

            let label = if dir == 0 {
                "fluid-gpu flow lk forward"
            } else {
                "fluid-gpu flow lk backward"
            };
            let params = params_buffer(
                device,
                queue,
                label,
                words(
                    [tl.w, tl.h, tl.cols, tl.rows],
                    [bs, radius, 0, 0],
                    [0; 4],
                    [0.0; 4],
                    [0.0; 4],
                ),
            );
            uniforms.push(params);
            let bind = bind_group(
                device,
                &pipelines.lk,
                label,
                &[
                    (0, &src_side[target]),
                    (1, &dst_side[target]),
                    (2, grad_x),
                    (3, grad_y),
                    (4, &current),
                    (5, &lk_grids[dir]),
                    (6, &uniforms[uniforms.len() - 1]),
                ],
            );
            passes.push(Pass {
                kind: Kind::Lk,
                bind,
                groups: groups(tl.cols, tl.rows),
            });

            let label = if dir == 0 {
                "fluid-gpu flow fill forward"
            } else {
                "fluid-gpu flow fill backward"
            };
            let dense = if dir == 0 {
                &dense_forward
            } else {
                &dense_backward
            };
            let params = params_buffer(
                device,
                queue,
                label,
                words(
                    [l0.w, l0.h, tl.cols, tl.rows],
                    [0; 4],
                    [0; 4],
                    [0.0; 4],
                    [inv, gather, 0.0, 0.0],
                ),
            );
            uniforms.push(params);
            let bind = bind_group(
                device,
                &pipelines.fill,
                label,
                &[
                    (0, &lk_grids[dir]),
                    (1, dense),
                    (2, &uniforms[uniforms.len() - 1]),
                ],
            );
            passes.push(Pass {
                kind: Kind::Fill,
                bind,
                groups: groups(l0.w, l0.h),
            });
        }

        // Forward-backward consistency, evaluated on a half-resolution grid and
        // bilinearly upsampled, exactly like 'fluid-flow' fill_and_confidence.
        let label = "fluid-gpu flow confidence grid";
        let params = params_buffer(
            device,
            queue,
            label,
            words(
                [l0.w, l0.h, tl.cols, tl.rows],
                [gw, gh, 0, 0],
                [0; 4],
                [0.0; 4],
                [inv, gather, 0.0, 0.0],
            ),
        );
        uniforms.push(params);
        let bind = bind_group(
            device,
            &pipelines.conf_grid,
            label,
            &[
                (0, &lk_grids[0]),
                (1, &lk_grids[1]),
                (2, &prev_luma[0]),
                (3, &conf_grid),
                (4, &uniforms[uniforms.len() - 1]),
            ],
        );
        passes.push(Pass {
            kind: Kind::ConfGrid,
            bind,
            groups: groups(gw, gh),
        });

        let label = "fluid-gpu flow confidence upsample";
        let params = params_buffer(
            device,
            queue,
            label,
            words([l0.w, l0.h, gw, gh], [0; 4], [0; 4], [0.0; 4], [0.0; 4]),
        );
        uniforms.push(params);
        let bind = bind_group(
            device,
            &pipelines.conf_upsample,
            label,
            &[
                (0, &conf_grid),
                (1, &confidence),
                (2, &uniforms[uniforms.len() - 1]),
            ],
        );
        passes.push(Pass {
            kind: Kind::ConfUpsample,
            bind,
            groups: groups(l0.w, l0.h),
        });

        Ok(Self {
            prev_luma,
            next_luma,
            dense_forward,
            dense_backward,
            confidence,
            conf_grid,
            pipelines,
            passes,
            uniforms,
        })
    }

    /// Record the whole flow schedule (pyramid, gradients, matching, fill and
    /// confidence) into one compute pass.
    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fluid-gpu flow"),
            timestamp_writes: None,
        });
        for step in &self.passes {
            pass.set_pipeline(self.pipelines.get(step.kind));
            pass.set_bind_group(0, &step.bind, &[]);
            pass.dispatch_workgroups(step.groups.0, step.groups.1, 1);
        }
    }

    pub(crate) fn level0_luma(&self) -> (&wgpu::Buffer, &wgpu::Buffer) {
        (&self.prev_luma[0], &self.next_luma[0])
    }
}

/// GPU dense optical-flow estimator.
///
/// One estimator owns every buffer for a fixed frame size and flow
/// configuration, so a pipeline creates it once and calls
/// [GpuFlow::estimate] per frame pair. [GpuFlow::estimate] returns exactly the
/// same [FlowEstimate] shape as 'fluid_flow::estimate' (dense gather fields plus
/// per-pixel confidence) and allocates only that result.
pub struct GpuFlow {
    ctx: Arc<GpuContext>,
    dims: Dims,
    rig: FlowRig,
    staging: wgpu::Buffer,
    last_cost_ns: u64,
}

impl std::fmt::Debug for GpuFlow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuFlow")
            .field("dims", &self.dims)
            .field("last_cost_ns", &self.last_cost_ns)
            .finish_non_exhaustive()
    }
}

impl GpuFlow {
    /// Create the estimator for one frame size; fails with
    /// [FluidError::Unsupported] when the device cannot hold the required
    /// buffers.
    pub fn new(ctx: Arc<GpuContext>, dims: Dims, cfg: &FlowConfig) -> Result<Self> {
        if dims.is_empty() {
            return Err(FluidError::invalid("gpu flow: frame dimensions are empty"));
        }
        let (device, queue) = {
            let shared = ctx.lock()?;
            (shared.device.clone(), shared.queue.clone())
        };
        let rig = FlowRig::new(&device, &queue, dims, cfg)?;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fluid-gpu flow readback"),
            size: dims.area() * 20,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            ctx,
            dims,
            rig,
            staging,
            last_cost_ns: 0,
        })
    }

    /// Frame size this estimator was built for.
    pub fn dims(&self) -> Dims {
        self.dims
    }

    /// Stable backend identifier reported to callers.
    pub fn backend_name(&self) -> &'static str {
        "wgpu-flow"
    }

    /// Wall time of the last [GpuFlow::estimate] call in nanoseconds.
    pub fn last_cost_ns(&self) -> u64 {
        self.last_cost_ns
    }

    /// Drop cached state; the estimator keeps its buffers and stays usable.
    pub fn reset(&mut self) {
        self.last_cost_ns = 0;
    }

    /// Estimate the dense forward and backward flow between two luma planes.
    pub fn estimate(&mut self, prev: &LumaPlane, next: &LumaPlane) -> Result<FlowEstimate> {
        let started = Instant::now();
        if prev.dims != self.dims || next.dims != self.dims {
            return Err(FluidError::invalid(format!(
                "gpu flow: frame size {} / {} does not match {}",
                prev.dims, next.dims, self.dims
            )));
        }
        let area = self.dims.area();
        if prev.data.len() as u64 != area || next.data.len() as u64 != area {
            return Err(FluidError::invalid(
                "gpu flow: luma plane buffer length does not match its dimensions",
            ));
        }

        let shared = self.ctx.lock()?;
        shared
            .queue
            .write_buffer(&self.rig.prev_luma[0], 0, bytemuck::cast_slice(&prev.data));
        shared
            .queue
            .write_buffer(&self.rig.next_luma[0], 0, bytemuck::cast_slice(&next.data));

        let mut encoder = shared
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fluid-gpu flow"),
            });
        self.rig.encode(&mut encoder);
        encoder.copy_buffer_to_buffer(&self.rig.dense_forward, 0, &self.staging, 0, area * 8);
        encoder.copy_buffer_to_buffer(
            &self.rig.dense_backward,
            0,
            &self.staging,
            area * 8,
            area * 8,
        );
        encoder.copy_buffer_to_buffer(&self.rig.confidence, 0, &self.staging, area * 16, area * 4);
        shared.queue.submit(Some(encoder.finish()));
        if let Some(error) = shared.take_error() {
            return Err(FluidError::internal(format!(
                "gpu flow: wgpu error: {error}"
            )));
        }

        let slice = self.staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        shared
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|error| {
                FluidError::internal(format!("gpu flow: device poll failed: {error}"))
            })?;

        let mut forward = FlowField::new(self.dims);
        let mut backward = FlowField::new(self.dims);
        let confidence;
        {
            let view = slice.get_mapped_range().map_err(|error| {
                FluidError::internal(format!("gpu flow: readback failed: {error}"))
            })?;
            let bytes: &[u8] = &view;
            let count = area as usize;
            let fwd =
                bytemuck::try_cast_slice::<u8, f32>(&bytes[..count * 8]).map_err(|error| {
                    FluidError::internal(format!("gpu flow: forward field readback: {error}"))
                })?;
            let bwd = bytemuck::try_cast_slice::<u8, f32>(&bytes[count * 8..count * 16]).map_err(
                |error| FluidError::internal(format!("gpu flow: backward field readback: {error}")),
            )?;
            let conf = bytemuck::try_cast_slice::<u8, f32>(&bytes[count * 16..count * 20])
                .map_err(|error| {
                    FluidError::internal(format!("gpu flow: confidence readback: {error}"))
                })?;
            for (i, vector) in forward.vectors_mut().iter_mut().enumerate() {
                vector.x = fwd[2 * i];
                vector.y = fwd[2 * i + 1];
            }
            for (i, vector) in backward.vectors_mut().iter_mut().enumerate() {
                vector.x = bwd[2 * i];
                vector.y = bwd[2 * i + 1];
            }
            confidence = conf.to_vec();
        }
        self.staging.unmap();
        self.last_cost_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        Ok(FlowEstimate::new(forward, backward, confidence))
    }
}
