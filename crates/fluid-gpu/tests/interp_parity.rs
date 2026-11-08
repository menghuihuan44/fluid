//! Quality parity tests: GPU motion-compensated interpolation versus the CPU
//! reference in fluid-interp.
//!
//! Each scene is rendered analytically at two sub-pixel positions, so the true
//! intermediate frame (half a motion step) is known exactly.  The GPU
//! interpolator follows the same reduced-resolution matching policy as the CPU
//! estimator (half-resolution matching for large frames, `FLUID_FULL_RES_FLOW`
//! forces full resolution on both sides), so its quality must stay within 1.5 dB
//! of the CPU result PSNR against the analytic truth, and the two frames must
//! agree closely enough to rule out holes and tearing.
//!
//! The tests skip gracefully (printing why) when no Vulkan/GLES adapter exists.

use fluid_core::{Dims, Frame, InterpolatorConfig, PixelFormat};
use fluid_gpu::{GpuContext, GpuInterpolator};
use std::sync::Arc;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const MULTIPLIER: u32 = 2;

/// Analytic scene: a continuous function of continuous pixel coordinates.
type Scene = fn(f32, f32, Dims) -> f32;

/// Ten soft blobs in normalised coordinates: the texture the CPU quality
/// harness uses for its translation guard (58 dB at 720p).
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

/// Thin vertical lines: triangle wave with an 8 px period (1 px features).
fn thin_lines(u: f32, _v: f32, _dims: Dims) -> f32 {
    let t = (u / 8.0).rem_euclid(1.0);
    0.15 + 0.70 * (1.0 - (2.0 * t - 1.0).abs())
}

/// High contrast edges: square wave with a 1 px anti-aliased transition.
fn edges(u: f32, v: f32, _dims: Dims) -> f32 {
    let period = 64.0;
    let t = u.rem_euclid(period);
    let high = t / period < 0.5;
    let edge = (t - 32.0).abs().min(t).min((t - period).abs());
    let aa = (edge - 0.5).clamp(-1.0, 1.0) * 0.5 + 0.5;
    let base = if high { 0.92 } else { 0.06 };
    let other = if high { 0.06 } else { 0.92 };
    other + (base - other) * aa + 0.02 * (v * 0.01 * std::f32::consts::TAU).sin()
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

fn frame_mae(a: &Frame, b: &Frame) -> f64 {
    let sum: f64 = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| f64::from((i32::from(*x) - i32::from(*y)).abs()))
        .sum();
    sum / a.data.len() as f64
}

fn frame_max_abs(a: &Frame, b: &Frame) -> i32 {
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| (i32::from(*x) - i32::from(*y)).abs())
        .max()
        .unwrap_or(0)
}

fn frame_psnr(a: &Frame, b: &Frame) -> f64 {
    assert_eq!(a.dims, b.dims);
    assert_eq!(a.data.len(), b.data.len());
    let mut sum = 0.0f64;
    for (x, y) in a.data.iter().zip(b.data.iter()) {
        let d = f64::from(i32::from(*x) - i32::from(*y));
        sum += d * d;
    }
    let mse = sum / a.data.len() as f64;
    if mse <= 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0).log10() - 10.0 * mse.log10()
    }
}

fn gpu_context() -> Option<Arc<GpuContext>> {
    match GpuContext::new() {
        Ok(ctx) => {
            let info = ctx.info();
            eprintln!(
                "fluid-gpu interpolation parity on {} ({})",
                info.name, info.backend
            );
            Some(Arc::new(ctx))
        }
        Err(error) => {
            eprintln!("skipping gpu interpolation parity: no usable adapter: {error}");
            None
        }
    }
}

struct SceneCase {
    name: &'static str,
    scene: Scene,
    shift_x: f32,
    shift_y: f32,
}

#[test]
fn gpu_interpolation_matches_cpu_on_analytic_scenes() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    // The scene-cut detector is a separate feature: this test compares
    // interpolation quality, so both backends get a threshold so high that the
    // analytic pairs (which are real motion, not cuts) are always interpolated.
    let cfg = InterpolatorConfig {
        multiplier: MULTIPLIER as u8,
        scene_cut_threshold: 0.99,
        ..Default::default()
    };
    let dims = Dims::new(WIDTH, HEIGHT);
    // An adapter exists, so a pipeline failure is a bug: fail loudly instead of
    // silently skipping the quality gate.
    let mut gpu = GpuInterpolator::new(ctx, dims, MULTIPLIER, &cfg)
        .expect("gpu interpolator should build when an adapter is present");
    let mut cpu = fluid_interp::Interpolator::new(&cfg);
    let cases = [
        SceneCase {
            name: "subpixel translation",
            scene: blobs,
            shift_x: 2.5,
            shift_y: 0.0,
        },
        SceneCase {
            name: "diagonal motion",
            scene: blobs,
            shift_x: 2.0,
            shift_y: 2.0,
        },
        SceneCase {
            name: "thin lines",
            scene: thin_lines,
            shift_x: 1.5,
            shift_y: 0.0,
        },
        SceneCase {
            name: "high contrast edges",
            scene: edges,
            shift_x: 2.5,
            shift_y: 1.5,
        },
    ];
    let mut failures = Vec::new();
    for case in cases {
        let prev = render_scene(case.scene, dims, 0.0, 0.0);
        let next = render_scene(case.scene, dims, case.shift_x, case.shift_y);
        let truth = render_scene(case.scene, dims, case.shift_x * 0.5, case.shift_y * 0.5);
        let cpu_frames = cpu
            .generate(&prev, &next, &cfg.flow)
            .expect("cpu interpolator should succeed");
        let gpu_frames = gpu
            .generate(&prev, &next)
            .expect("gpu interpolator should succeed");
        assert_eq!(
            cpu_frames.len(),
            1,
            "{}: cpu should generate one frame",
            case.name
        );
        assert_eq!(
            gpu_frames.len(),
            1,
            "{}: gpu should generate one frame",
            case.name
        );
        let cpu_truth = frame_psnr(&cpu_frames[0], &truth);
        let gpu_truth = frame_psnr(&gpu_frames[0], &truth);
        let gpu_cpu = frame_psnr(&gpu_frames[0], &cpu_frames[0]);
        eprintln!(
            "{:<22} cpu-vs-truth {:>6.2} dB  gpu-vs-truth {:>6.2} dB  gpu-vs-cpu {:>6.2} dB  \
             mae {:.3}/{:.3} LSB  max {}/{}/{}",
            case.name,
            cpu_truth,
            gpu_truth,
            gpu_cpu,
            frame_mae(&cpu_frames[0], &truth),
            frame_mae(&gpu_frames[0], &truth),
            frame_max_abs(&cpu_frames[0], &truth),
            frame_max_abs(&gpu_frames[0], &truth),
            frame_max_abs(&gpu_frames[0], &cpu_frames[0]),
        );
        if gpu_truth < cpu_truth - 1.5 {
            failures.push(format!(
                "{}: gpu {gpu_truth:.2} dB is more than 1.5 dB below cpu {cpu_truth:.2} dB",
                case.name
            ));
        }
        if gpu_cpu < 20.0 {
            failures.push(format!(
                "{}: gpu and cpu intermediate frames differ too much ({gpu_cpu:.2} dB)",
                case.name
            ));
        }
    }
    assert!(failures.is_empty(), "quality gate failed: {failures:#?}");
    assert_eq!(gpu.backend_name(), "wgpu-flow");
    assert_eq!(gpu.dims(), dims);
    assert_eq!(gpu.multiplier(), MULTIPLIER);
    assert!(gpu.last_cost_ns() > 0);
}

#[test]
fn gpu_interpolation_repeats_and_resets_deterministically() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    let cfg = InterpolatorConfig {
        multiplier: 3,
        ..Default::default()
    };
    let dims = Dims::new(320, 180);
    let mut gpu = GpuInterpolator::new(ctx, dims, 3, &cfg)
        .expect("gpu interpolator should build when an adapter is present");
    let mut prev = render_scene(blobs, dims, 0.0, 0.0);
    let mut next = render_scene(blobs, dims, 2.0, 1.0);
    prev.pts_ns = 0;
    next.pts_ns = 1_000_000_000;
    let first = gpu.generate(&prev, &next).expect("first generate");
    assert_eq!(first.len(), 2, "multiplier 3 should insert two frames");
    assert!(first[0].pts_ns < first[1].pts_ns);
    assert_eq!(first[0].index, prev.index);
    let second = gpu.generate(&prev, &next).expect("second generate");
    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!(a.data, b.data, "repeated generate must be identical");
    }
    gpu.reset();
    let after_reset = gpu.generate(&prev, &next).expect("generate after reset");
    for (a, b) in first.iter().zip(after_reset.iter()) {
        assert_eq!(a.data, b.data, "reset must not change the result");
    }
}
