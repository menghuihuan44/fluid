//! CPU vs GPU frame-interpolation + upscaling benchmark for the fused wgpu
//! pipeline.
//!
//! Run with:
//!
//!     cargo run --release -p fluid-gpu --example bench_fused -- --json out/gpu-fused-bench.json
//!
//! Three totals are measured for the same analytic frame pair at multiplier 2,
//! for 720p -> 1440p and 1080p -> 4K:
//!
//! * 'cpu_chain_ms': the portable CPU path - fluid-flow, fluid-interp and
//!   fluid-upscale for the inserted frame (including the upscale of it).
//! * 'gpu_separate_ms': today's two-step GPU path - GpuInterpolator::generate
//!   reads the intermediate frame back to the CPU, then GpuUpscaler::upscale
//!   uploads it again.
//! * 'gpu_fused_ms': GpuPipeline::generate - flow, synthesis and
//!   upscaling/sharpening stay on the GPU and only the finished destination
//!   frame is read back.
//!
//! Every number is the median of the measured runs after the warmup runs, and
//! each GPU number includes the source upload and the destination readback, so
//! the comparison is honest end-to-end work per inserted frame. The fused and
//! two-step frames are compared on the last run (max LSB) to show the shortcut
//! does not change the picture.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use fluid_core::{
    Dims, FlowConfig, Frame, InterpolatorConfig, LumaPlane, PixelFormat, UpscaleFilter,
    UpscalerConfig,
};
use fluid_gpu::{GpuContext, GpuInterpolator, GpuPipeline, GpuUpscaler, probe};
use fluid_upscale::{UpscaleBackend, Upscaler};
use serde_json::json;

/// Source -> destination pairs the suite reports.
const SIZES: [(u32, u32, u32, u32); 2] = [(1280, 720, 2560, 1440), (1920, 1080, 3840, 2160)];

/// Filters and sharpening amounts the suite reports.
const FILTERS: [UpscaleFilter; 3] = [
    UpscaleFilter::Bilinear,
    UpscaleFilter::Adaptive,
    UpscaleFilter::Lanczos,
];
const SHARPNESS: [f32; 2] = [0.0, 0.3];

/// Inserted frames per input pair (multiplier 2 -> one inserted frame).
const MULTIPLIER: u32 = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = Options::parse()?;
    println!(
        "fluid-gpu fused interpolation + upscaling benchmark (backend: {})",
        fluid_gpu::BACKEND_NAME
    );
    let info = match probe() {
        Ok(info) => info,
        Err(error) => {
            println!("no usable GPU adapter: {error}");
            return write_unavailable(&options, &error.to_string());
        }
    };
    println!(
        "GPU: {} ({}) vendor {:#06x}",
        info.name, info.backend, info.vendor_id
    );
    let context = match GpuContext::new() {
        Ok(context) => Arc::new(context),
        Err(error) => {
            println!("GPU device unavailable: {error}");
            return write_unavailable(&options, &error.to_string());
        }
    };
    println!(
        "warmup {} frame(s), then {} measured frame(s) per case (median reported)",
        options.warmup, options.frames
    );

    let cfg = InterpolatorConfig {
        multiplier: MULTIPLIER as u8,
        ..Default::default()
    };
    let flow_params = fluid_flow::FlowParams::from(&FlowConfig::default());

    let mut rows = Vec::new();
    let mut cpu_total_ms = 0.0f64;
    let mut separate_total_ms = 0.0f64;
    let mut fused_total_ms = 0.0f64;

    for (src_w, src_h, dst_w, dst_h) in SIZES {
        let src = Dims::new(src_w, src_h);
        let dst = Dims::new(dst_w, dst_h);
        let size = format!("{src_w}x{src_h}->{dst_w}x{dst_h}");
        let prev = moving_frame(src, 0.0, 0.0);
        let next = moving_frame(src, 2.5, 1.5);
        let prev_luma = LumaPlane::from_frame(&prev);
        let next_luma = LumaPlane::from_frame(&next);
        println!(
            "\n{size} ({:.2} MP source) multiplier {MULTIPLIER}",
            src.area() as f64 / 1_000_000.0
        );

        for filter in FILTERS {
            for sharpness in SHARPNESS {
                // --- CPU chain: flow, synthesise, upscale. -------------------
                let upscale_cfg = UpscalerConfig {
                    filter,
                    sharpness,
                    ..Default::default()
                };
                let mut cpu_upscaler = Upscaler::new(&upscale_cfg, dst);
                let cpu_interp = fluid_interp::Interpolator::new(&cfg);
                let mut cpu_out = Frame::zeroed(dst, PixelFormat::Rgba8);
                let mut cpu_times = Vec::with_capacity(options.frames);
                for _ in 0..options.frames {
                    let started = Instant::now();
                    let flow = fluid_flow::estimate(&prev_luma, &next_luma, &flow_params)?;
                    let (intermediate, _metrics) =
                        cpu_interp.interpolate(&prev, &next, 0.5, &flow)?;
                    UpscaleBackend::upscale(&mut cpu_upscaler, &intermediate, &mut cpu_out)?;
                    cpu_times.push(started.elapsed());
                }

                // --- Two-step GPU chain: interpolate, read back, upscale. ----
                let mut gpu_interp =
                    GpuInterpolator::new(Arc::clone(&context), src, MULTIPLIER, &cfg)?;
                let mut gpu_upscaler =
                    GpuUpscaler::new(Arc::clone(&context), filter, src, dst, sharpness, None)?;
                let mut separate_out = Frame::zeroed(dst, PixelFormat::Rgba8);
                for _ in 0..options.warmup {
                    let frames = gpu_interp.generate(&prev, &next)?;
                    for frame in &frames {
                        gpu_upscaler.upscale(frame, &mut separate_out)?;
                    }
                }

                // --- Fused GPU chain: everything stays on the GPU. ----------
                let mut fused = GpuPipeline::new(
                    Arc::clone(&context),
                    src,
                    dst,
                    filter,
                    sharpness,
                    None,
                    &cfg,
                )?;
                let mut fused_frames = Vec::new();
                for _ in 0..options.warmup {
                    fused_frames = fused.generate(&prev, &next)?;
                }

                // Interleave the two GPU paths, alternating which one runs first,
                // so drift from a loaded desktop hits both measurements equally.
                let mut separate_times = Vec::with_capacity(options.frames);
                let mut fused_times = Vec::with_capacity(options.frames);
                for index in 0..options.frames {
                    if index % 2 == 0 {
                        separate_times.push(time_separate(
                            &mut gpu_interp,
                            &mut gpu_upscaler,
                            &prev,
                            &next,
                            &mut separate_out,
                        )?);
                        fused_times.push(time_fused(&mut fused, &prev, &next, &mut fused_frames)?);
                    } else {
                        fused_times.push(time_fused(&mut fused, &prev, &next, &mut fused_frames)?);
                        separate_times.push(time_separate(
                            &mut gpu_interp,
                            &mut gpu_upscaler,
                            &prev,
                            &next,
                            &mut separate_out,
                        )?);
                    }
                }

                let cpu_ms = ns_to_ms(median(&mut cpu_times));
                let separate_ms = ns_to_ms(median(&mut separate_times));
                let fused_ms = ns_to_ms(median(&mut fused_times));
                let speedup = separate_ms / fused_ms.max(0.000_001);
                let max_lsb = fused_frames
                    .iter()
                    .map(|frame| max_abs_diff(frame, &separate_out))
                    .max()
                    .unwrap_or(0);
                let name = format!("{filter:?}").to_lowercase();
                println!(
                    "  {name:<9} sharp {sharpness:.1}  cpu {cpu_ms:>8.2} ms  separate {separate_ms:>8.2} ms  fused {fused_ms:>8.2} ms  {:>5.2}x vs separate  max {max_lsb} LSB",
                    speedup
                );
                cpu_total_ms += cpu_ms;
                separate_total_ms += separate_ms;
                fused_total_ms += fused_ms;
                rows.push(json!({
                    "size": size,
                    "multiplier": MULTIPLIER,
                    "filter": name,
                    "sharpness": sharpness,
                    "cpu_chain_ms": round3(cpu_ms),
                    "gpu_separate_ms": round3(separate_ms),
                    "gpu_fused_ms": round3(fused_ms),
                    "speedup_vs_gpu_separate": round3(speedup),
                    "max_lsb": max_lsb,
                }));
            }
        }
    }

    let fused_total_fps = 1000.0 * rows.len() as f64 / fused_total_ms.max(0.001);
    println!(
        "\ntotal over {} cases: cpu {cpu_total_ms:.2} ms, gpu separate {separate_total_ms:.2} ms, gpu fused {fused_total_ms:.2} ms",
        rows.len()
    );
    println!(
        "speedup of fused vs two-step {:.2}x, fused {:.2} fps across cases",
        separate_total_ms / fused_total_ms.max(0.001),
        fused_total_fps
    );

    if let Some(path) = &options.json {
        let report = json!({
            "backend_name": fluid_gpu::BACKEND_NAME,
            "gpu_available": true,
            "adapter": {
                "name": info.name,
                "backend": info.backend,
                "vendor_id": info.vendor_id,
                "max_feature_level": info.max_feature_level,
            },
            "multiplier": MULTIPLIER,
            "warmup_frames": options.warmup,
            "measured_frames": options.frames,
            "cpu_total_ms": round3(cpu_total_ms),
            "gpu_separate_total_ms": round3(separate_total_ms),
            "gpu_fused_total_ms": round3(fused_total_ms),
            "overall_speedup_vs_gpu_separate": round3(
                separate_total_ms / fused_total_ms.max(0.001)
            ),
            "cases": rows,
        });
        write_json(path, &report)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

fn write_unavailable(options: &Options, error: &str) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(path) = &options.json {
        let report = json!({
            "backend_name": fluid_gpu::BACKEND_NAME,
            "gpu_available": false,
            "error": error,
            "cases": [],
        });
        write_json(path, &report)?;
        println!("wrote {} (GPU unavailable)", path.display());
    }
    Ok(())
}

/// Analytic moving texture, sampled at pixel centres shifted by the given
/// sub-pixel amount; the pair differs by (2.5, 1.5) px of smooth motion.
fn moving_frame(dims: Dims, shift_x: f32, shift_y: f32) -> Frame {
    let mut frame = Frame::zeroed(dims, PixelFormat::Rgba8);
    for y in 0..dims.height {
        let row = frame.row_mut(y);
        for x in 0..dims.width {
            let u = x as f32 + 0.5 - shift_x;
            let v = y as f32 + 0.5 - shift_y;
            let a = (u * 0.045 + v * 0.025) * std::f32::consts::TAU;
            let b = (v * 0.07 - u * 0.012) * std::f32::consts::TAU;
            let value = (0.5 + 0.20 * a.sin() + 0.14 * (b + 1.3).sin()).clamp(0.0, 1.0);
            let byte = (value * 255.0 + 0.5) as u8;
            let i = x as usize * 4;
            row[i] = byte;
            row[i + 1] = byte;
            row[i + 2] = byte;
            row[i + 3] = 255;
        }
    }
    frame
}

fn max_abs_diff(a: &Frame, b: &Frame) -> i32 {
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| (i32::from(*x) - i32::from(*y)).abs())
        .max()
        .unwrap_or(0)
}

/// One two-step chain iteration: interpolate, read the intermediate frame
/// back, upload it again and upscale. Leaves the last frame in 'out'.
fn time_separate(
    interp: &mut GpuInterpolator,
    upscaler: &mut GpuUpscaler,
    prev: &Frame,
    next: &Frame,
    out: &mut Frame,
) -> Result<Duration, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let frames = interp.generate(prev, next)?;
    for frame in &frames {
        upscaler.upscale(frame, out)?;
    }
    Ok(started.elapsed())
}

/// One fused iteration: flow, synthesis and upscaling never leave the GPU.
fn time_fused(
    fused: &mut GpuPipeline,
    prev: &Frame,
    next: &Frame,
    frames_out: &mut Vec<Frame>,
) -> Result<Duration, Box<dyn std::error::Error>> {
    let started = Instant::now();
    *frames_out = fused.generate(prev, next)?;
    Ok(started.elapsed())
}

fn median(times: &mut [Duration]) -> u64 {
    if times.is_empty() {
        return 0;
    }
    times.sort_unstable();
    let mid = times.len() / 2;
    if times.len() % 2 == 1 {
        times[mid].as_nanos() as u64
    } else {
        (times[mid - 1].as_nanos() + times[mid].as_nanos()) as u64 / 2
    }
}

fn ns_to_ms(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn write_json(path: &PathBuf, value: &serde_json::Value) -> std::io::Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(value)? + "\n")
}

/// Command line: --json <path>, --warmup <frames>, --frames <frames>.
struct Options {
    json: Option<PathBuf>,
    warmup: usize,
    frames: usize,
}

impl Options {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut options = Options {
            json: None,
            warmup: 3,
            frames: 9,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--json" => {
                    let value = args.next().ok_or("--json needs a path")?;
                    options.json = Some(PathBuf::from(value));
                }
                "--warmup" => {
                    let value = args.next().ok_or("--warmup needs a count")?;
                    options.warmup = value.parse()?;
                }
                "--frames" => {
                    let value = args.next().ok_or("--frames needs a count")?;
                    options.frames = value.parse()?;
                }
                "--help" | "-h" => {
                    println!(
                        "usage: bench_fused [--json <path>] [--warmup <frames>] [--frames <frames>]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other:?}").into()),
            }
        }
        if options.frames == 0 {
            return Err("--frames must be at least 1".into());
        }
        Ok(options)
    }
}
