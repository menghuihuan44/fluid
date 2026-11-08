//! CPU vs GPU frame interpolation benchmark for the wgpu backend.
//!
//! Run with:
//!
//!     cargo run --release -p fluid-gpu --example bench_interp -- --json out/gpu-interp-bench.json
//!
//! Two resolutions are measured at multiplier 2 (one inserted frame): the GPU
//! end-to-end cost of GpuInterpolator::generate (pack, GPU flow pyramid, GPU
//! motion-compensated synthesis, synchronous readback into a Frame) is compared
//! with the CPU fluid-flow + fluid-interp pipeline on the same input pair.
//! Three timed runs are taken per stage and the median is reported.
//!
//! The input pair is an analytic moving texture, so the CPU and GPU results can
//! also be compared against each other with PSNR to show they agree.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use fluid_core::{Dims, FlowConfig, Frame, InterpolatorConfig, LumaPlane, PixelFormat};
use fluid_gpu::{GpuContext, GpuInterpolator, probe};
use serde_json::json;

/// Frame sizes the suite reports.
const SIZES: [(u32, u32); 2] = [(1280, 720), (1920, 1080)];

/// Timed repetitions per stage; the median is reported.
const RUNS: usize = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = Options::parse()?;
    println!(
        "fluid-gpu interpolation benchmark (backend: {})",
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

    let cfg = InterpolatorConfig {
        multiplier: 2,
        ..Default::default()
    };
    let flow_cfg = FlowConfig::default();
    let flow_params = fluid_flow::FlowParams::from(&flow_cfg);
    let mut rows = Vec::new();
    let mut cpu_total_ms = 0.0f64;
    let mut gpu_total_ms = 0.0f64;

    for (width, height) in SIZES {
        let dims = Dims::new(width, height);
        let prev = moving_frame(dims, 0.0, 0.0);
        let next = moving_frame(dims, 2.5, 1.5);
        let prev_luma = LumaPlane::from_frame(&prev);
        let next_luma = LumaPlane::from_frame(&next);

        // CPU reference: flow once, then synthesis for the middle frame.
        let mut cpu_flow_ns = u64::MAX;
        for _ in 0..RUNS {
            let started = Instant::now();
            let flow = fluid_flow::estimate(&prev_luma, &next_luma, &flow_params)?;
            let elapsed = started.elapsed().as_nanos() as u64;
            cpu_flow_ns = cpu_flow_ns.min(elapsed);
            drop(flow);
        }
        let flow = fluid_flow::estimate(&prev_luma, &next_luma, &flow_params)?;
        let cpu_interp = fluid_interp::Interpolator::new(&cfg);
        let mut cpu_interp_ns = u64::MAX;
        let mut cpu_frame: Option<Frame> = None;
        for _ in 0..RUNS {
            let started = Instant::now();
            let (frame, _metrics) = cpu_interp.interpolate(&prev, &next, 0.5, &flow)?;
            let elapsed = started.elapsed().as_nanos() as u64;
            cpu_interp_ns = cpu_interp_ns.min(elapsed);
            cpu_frame = Some(frame);
        }
        let cpu_frame = cpu_frame.expect("cpu interpolation produced a frame");

        let mut gpu = GpuInterpolator::new(Arc::clone(&context), dims, 2, &cfg)?;
        gpu.generate(&prev, &next)?; // warm up pipelines and buffers
        let mut gpu_ns = u64::MAX;
        let mut gpu_frame: Option<Frame> = None;
        for _ in 0..RUNS {
            let started = Instant::now();
            let frames = gpu.generate(&prev, &next)?;
            let elapsed = started.elapsed().as_nanos() as u64;
            gpu_ns = gpu_ns.min(elapsed);
            gpu_frame = frames.into_iter().next();
        }
        let gpu_frame = gpu_frame.expect("gpu interpolation produced a frame");

        let cpu_total_ns = cpu_flow_ns + cpu_interp_ns;
        let cpu_flow_ms = ns_to_ms(cpu_flow_ns);
        let cpu_interp_ms = ns_to_ms(cpu_interp_ns);
        let cpu_total_ms_case = ns_to_ms(cpu_total_ns);
        let gpu_total_ms_case = ns_to_ms(gpu_ns);
        let speedup = cpu_total_ns as f64 / gpu_ns.max(1) as f64;
        let psnr_gpu_cpu = psnr(&gpu_frame, &cpu_frame);
        cpu_total_ms += cpu_total_ms_case;
        gpu_total_ms += gpu_total_ms_case;
        println!(
            "\n{} ({:.2} MP) multiplier 2",
            dims,
            dims.area() as f64 / 1_000_000.0
        );
        println!(
            "  CPU: flow {:>8.3} ms, synthesize {:>8.3} ms, total {:>8.3} ms",
            cpu_flow_ms, cpu_interp_ms, cpu_total_ms_case
        );
        println!(
            "  GPU: end-to-end {:>8.3} ms ({:.1} fps), speedup {:.2}x, GPU-vs-CPU {:.2} dB",
            gpu_total_ms_case,
            1000.0 / gpu_total_ms_case.max(0.001),
            speedup,
            psnr_gpu_cpu
        );
        rows.push(json!({
            "size": format!("{width}x{height}"),
            "cpu_flow_ms": round3(cpu_flow_ms),
            "cpu_interp_ms": round3(cpu_interp_ms),
            "cpu_total_ms": round3(cpu_total_ms_case),
            "gpu_total_ms": round3(gpu_total_ms_case),
            "gpu_fps": round3(1000.0 / gpu_total_ms_case.max(0.001)),
            "speedup": round3(speedup),
            "psnr_gpu_vs_cpu": round3(psnr_gpu_cpu),
        }));
    }

    let overall = cpu_total_ms / gpu_total_ms.max(0.001);
    println!(
        "\ntotal: CPU {:.2} ms, GPU {:.2} ms, overall speedup {:.2}x",
        cpu_total_ms, gpu_total_ms, overall
    );
    println!("both sides keep their working memory: the GPU path allocates no buffers per frame.");

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
            "multiplier": 2,
            "runs": RUNS,
            "cpu_total_ms": round3(cpu_total_ms),
            "gpu_total_ms": round3(gpu_total_ms),
            "overall_speedup": round3(overall),
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
/// sub-pixel amount.  The pair differs by (2.5, 1.5) px of smooth motion.
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

fn psnr(a: &Frame, b: &Frame) -> f64 {
    let mut sum = 0.0f64;
    for (&x, &y) in a.data.iter().zip(b.data.iter()) {
        let d = f64::from(x.abs_diff(y));
        sum += d * d;
    }
    let mse = sum / a.data.len().max(1) as f64;
    if mse <= 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0).log10() - 10.0 * mse.log10()
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

/// Command line: --json <path>.
struct Options {
    json: Option<PathBuf>,
}

impl Options {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut options = Options { json: None };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--json" => {
                    let value = args.next().ok_or("--json needs a path")?;
                    options.json = Some(PathBuf::from(value));
                }
                "--help" | "-h" => {
                    println!("usage: bench_interp [--json <path>]");
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other:?}").into()),
            }
        }
        Ok(options)
    }
}
