//! Parity test: the fused interpolation + upscaling pipeline must produce the
//! same frames as the two-step chain (GPU interpolation followed by GPU
//! upscaling), within one least-significant bit.
//!
//! The two paths share their shaders and their CPU tap tables, so the only
//! expected difference is floating-point rounding: the two-step chain writes the
//! intermediate frame into a buffer and reloads it, while the fused path keeps
//! the exact same bytes on the GPU.  Every compared frame must therefore match
//! to within one LSB, and the test prints the observed maximum and mean
//! difference per case.
//!
//! The tests skip gracefully (printing why) when no Vulkan/GLES adapter exists.

use fluid_core::{Dims, Frame, InterpolatorConfig, PixelFormat, TemporalConfig, UpscaleFilter};
use fluid_gpu::{GpuContext, GpuInterpolator, GpuPipeline, GpuUpscaler};
use std::sync::Arc;

/// Source size for the parity runs: a quarter of 720p, so the test exercises
/// every pass (flow, synthesis, upscaling, sharpening, temporal blending) while
/// staying fast.
const SRC: Dims = Dims::new(640, 360);
const DST: Dims = Dims::new(1280, 720);

/// Analytic scene: a continuous function of continuous pixel coordinates.
type Scene = fn(f32, f32, Dims) -> f32;

/// Ten soft blobs in normalised coordinates.
const BLOBS: [(f32, f32, f32, f32); 10] = [
    (0.16, 0.20, 0.075, 0.55),
    (0.61, 0.14, 0.06, 0.35),
    (0.85, 0.37, 0.08, 0.6),
    (0.34, 0.54, 0.07, 0.45),
    (0.72, 0.67, 0.09, 0.3),
    (0.11, 0.76, 0.065, 0.5),
    (0.49, 0.87, 0.075, 0.4),
    (0.92, 0.84, 0.05, 0.55),
    (0.27, 0.36, 0.04, 0.25),
    (0.57, 0.44, 0.035, 0.2),
];

fn blobs(u: f32, v: f32, dims: Dims) -> f32 {
    let uf = u / dims.width as f32;
    let vf = v / dims.height as f32;
    let mut value = 0.12f32;
    for &(bx, by, sigma, amp) in BLOBS.iter() {
        let dx = uf - bx;
        let dy = vf - by;
        let r2 = (dx * dx + dy * dy) / (sigma * sigma);
        if r2 < 9.0 {
            value += amp * (-r2).exp();
        }
    }
    value.clamp(0.0, 1.0)
}

fn render_scene(scene: Scene, dims: Dims, shift_x: f32, shift_y: f32) -> Frame {
    let mut frame = Frame::zeroed(dims, PixelFormat::Rgba8);
    for y in 0..dims.height {
        for x in 0..dims.width {
            let u = x as f32 + 0.5 - shift_x;
            let v = y as f32 + 0.5 - shift_y;
            let byte = (scene(u, v, dims).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            let i = (y as usize * dims.width as usize + x as usize) * 4;
            frame.data[i] = byte;
            frame.data[i + 1] = byte;
            frame.data[i + 2] = byte;
            frame.data[i + 3] = 255;
        }
    }
    frame
}

fn frame_diff(a: &Frame, b: &Frame) -> (i32, f64) {
    assert_eq!(a.dims, b.dims, "frame sizes differ");
    assert_eq!(a.data.len(), b.data.len(), "frame buffers differ");
    let mut max = 0i32;
    let mut sum = 0u64;
    for (x, y) in a.data.iter().zip(b.data.iter()) {
        let d = (i32::from(*x) - i32::from(*y)).abs();
        max = max.max(d);
        sum += d as u64;
    }
    (max, sum as f64 / a.data.len() as f64)
}

fn gpu_context() -> Option<Arc<GpuContext>> {
    match GpuContext::new() {
        Ok(ctx) => {
            let info = ctx.info();
            eprintln!("fluid-gpu fused parity on {} ({})", info.name, info.backend);
            Some(Arc::new(ctx))
        }
        Err(error) => {
            eprintln!("skipping gpu fused parity: no usable adapter: {error}");
            None
        }
    }
}

/// Run the two-step chain and the fused pipeline over the same two frame pairs.
///
/// Returns the two-step frames and the fused frames, flattened pair by pair.
#[allow(clippy::type_complexity)]
fn run_both(
    ctx: &Arc<GpuContext>,
    cfg: &InterpolatorConfig,
    filter: UpscaleFilter,
    sharpness: f32,
    temporal: Option<TemporalConfig>,
    pairs: &[(Frame, Frame)],
) -> (Vec<Frame>, Vec<Frame>) {
    // Two-step chain: GPU interpolation, readback, GPU upscale.
    let mut interp = GpuInterpolator::new(ctx.clone(), SRC, u32::from(cfg.multiplier), cfg)
        .expect("gpu interpolator should build when an adapter is present");
    let mut upscaler = GpuUpscaler::new(ctx.clone(), filter, SRC, DST, sharpness, temporal)
        .expect("gpu upscaler should build when an adapter is present");
    let mut chain = Vec::new();
    for (prev, next) in pairs {
        let intermediates = interp
            .generate(prev, next)
            .expect("gpu interpolator should succeed");
        for intermediate in intermediates {
            let mut out = Frame::zeroed(DST, intermediate.fmt);
            upscaler
                .upscale(&intermediate, &mut out)
                .expect("gpu upscaler should succeed");
            chain.push(out);
        }
    }

    // Fused chain: flow, synthesis and upscaling all on the GPU; only the
    // finished destination frames are read back.
    let mut fused = GpuPipeline::new(ctx.clone(), SRC, DST, filter, sharpness, temporal, cfg)
        .expect("gpu fused pipeline should build when an adapter is present");
    let mut fused_frames = Vec::new();
    for (prev, next) in pairs {
        fused_frames.extend(
            fused
                .generate(prev, next)
                .expect("gpu fused pipeline should succeed"),
        );
    }

    (chain, fused_frames)
}

#[test]
fn fused_pipeline_matches_two_step_chain() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    let cfg = InterpolatorConfig {
        multiplier: 2,
        scene_cut_threshold: 0.99,
        ..Default::default()
    };
    // Two pairs, so the temporal history is exercised by the second pair.
    let pairs = [
        (
            render_scene(blobs, SRC, 0.0, 0.0),
            render_scene(blobs, SRC, 2.0, 1.0),
        ),
        (
            render_scene(blobs, SRC, 2.0, 1.0),
            render_scene(blobs, SRC, 4.0, 2.0),
        ),
    ];
    let filters = [
        UpscaleFilter::Bilinear,
        UpscaleFilter::Adaptive,
        UpscaleFilter::Lanczos,
    ];
    let sharpness_values = [0.0f32, 0.3];
    let temporal_values = [None, Some(TemporalConfig::default())];

    let mut worst_max = 0i32;
    let mut worst_mean = 0.0f64;
    for filter in filters {
        for sharpness in sharpness_values {
            for temporal in temporal_values {
                let (chain, fused) = run_both(&ctx, &cfg, filter, sharpness, temporal, &pairs);
                assert_eq!(
                    chain.len(),
                    fused.len(),
                    "{filter:?} sharpness {sharpness} temporal {}: frame counts differ",
                    temporal.is_some()
                );
                assert!(!fused.is_empty(), "{filter:?}: no frames generated");
                let mut case_max = 0i32;
                let mut case_mean = 0.0f64;
                for (two_step, fused_frame) in chain.iter().zip(fused.iter()) {
                    assert_eq!(two_step.dims, DST);
                    assert_eq!(fused_frame.dims, DST);
                    assert_eq!(two_step.fmt, fused_frame.fmt);
                    let (max, mean) = frame_diff(two_step, fused_frame);
                    case_max = case_max.max(max);
                    case_mean = case_mean.max(mean);
                }
                eprintln!(
                    "{filter:<9?} sharpness {sharpness:.1} temporal {:<5} max {case_max} LSB  mean {case_mean:.4} LSB  frames {}",
                    temporal.is_some(),
                    fused.len()
                );
                worst_max = worst_max.max(case_max);
                worst_mean = worst_mean.max(case_mean);
                assert!(
                    case_max <= 1,
                    "{filter:?} sharpness {sharpness} temporal {}: fused output differs from the two-step chain by {case_max} LSB",
                    temporal.is_some()
                );
            }
        }
    }
    eprintln!("worst fused-vs-chain diff: {worst_max} LSB (mean {worst_mean:.4} LSB)");
}

#[test]
fn fused_pipeline_is_byte_identical_for_identity_off() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    let cfg = InterpolatorConfig {
        multiplier: 2,
        scene_cut_threshold: 0.99,
        ..Default::default()
    };
    let prev = render_scene(blobs, SRC, 0.0, 0.0);
    let next = render_scene(blobs, SRC, 2.0, 1.0);
    // Off at 1:1 is a pass-through on the CPU and an identity copy on the GPU,
    // so the interpolated frame must survive the fused path byte for byte.
    let chain = GpuInterpolator::new(ctx.clone(), SRC, 2, &cfg)
        .expect("gpu interpolator should build")
        .generate(&prev, &next)
        .expect("gpu interpolator should succeed");
    let fused = GpuPipeline::new(ctx.clone(), SRC, SRC, UpscaleFilter::Off, 0.0, None, &cfg)
        .expect("gpu fused pipeline should build")
        .generate(&prev, &next)
        .expect("gpu fused pipeline should succeed");
    assert_eq!(chain.len(), fused.len());
    assert!(!fused.is_empty());
    for (expected, actual) in chain.iter().zip(fused.iter()) {
        let (max, mean) = frame_diff(expected, actual);
        assert_eq!(max, 0, "identity Off fused frame differs (mean {mean:.4})");
    }
}

#[test]
fn fused_pipeline_repeats_and_resets_deterministically() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    let cfg = InterpolatorConfig {
        multiplier: 3,
        ..Default::default()
    };
    let mut fused = GpuPipeline::new(
        ctx,
        SRC,
        DST,
        UpscaleFilter::Adaptive,
        0.3,
        Some(TemporalConfig::default()),
        &cfg,
    )
    .expect("gpu fused pipeline should build when an adapter is present");
    assert_eq!(fused.backend_name(), "wgpu-fused");
    assert_eq!(fused.src_dims(), SRC);
    assert_eq!(fused.dst_dims(), DST);
    assert_eq!(fused.dims(), SRC);
    assert_eq!(fused.multiplier(), 3);

    let mut prev = render_scene(blobs, SRC, 0.0, 0.0);
    let mut next = render_scene(blobs, SRC, 2.0, 1.0);
    prev.pts_ns = 0;
    next.pts_ns = 1_000_000_000;
    let first = fused.generate(&prev, &next).expect("first generate");
    assert_eq!(first.len(), 2, "multiplier 3 should insert two frames");
    assert_eq!(first[0].dims, DST);
    assert_eq!(first[0].index, prev.index);
    assert!(first[0].pts_ns < first[1].pts_ns);
    assert!(fused.last_cost_ns() > 0);

    // A repeated call continues the temporal history, exactly like the two-step
    // chain would; the same inputs therefore stay byte-identical.
    let second = fused.generate(&prev, &next).expect("second generate");
    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!(a.data, b.data, "repeated generate must be identical");
    }

    fused.reset();
    let after_reset = fused.generate(&prev, &next).expect("generate after reset");
    for (a, b) in first.iter().zip(after_reset.iter()) {
        assert_eq!(a.data, b.data, "reset must not change the result");
    }

    // Invalid frame sizes are rejected, not silently cropped.
    let wrong = render_scene(blobs, Dims::new(320, 180), 0.0, 0.0);
    assert!(fused.generate(&wrong, &next).is_err());
}
