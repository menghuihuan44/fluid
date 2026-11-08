//! CPU vs GPU upscaling benchmark for the wgpu backend.
//!
//! Run with:
//!
//!     cargo run --release -p fluid-gpu --example bench_gpu -- --json out/gpu-bench.json
//!
//! Two resolutions are measured, 720p -> 1440p and 1080p -> 4K, for every
//! spatial filter plus the temporal (TAA-style) filter, with and without
//! sharpening. The GPU number is honest end-to-end per frame: linearization of
//! the source into canonical RGBA8, upload, compute passes and the synchronous
//! readback that converts back into the caller's frame. The CPU number is the
//! portable fluid-upscale pipeline driven through the same UpscaleBackend
//! trait with a reused output frame, so neither side pays a per-frame output
//! allocation.
//!
//! Output is a table on stdout and, with --json <path>, the same data as JSON
//! (adapter identity, per-case timings, speedups and parity diffs).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use fluid_core::{Dims, Frame, PixelFormat, TemporalConfig, UpscaleFilter, UpscalerConfig};
use fluid_gpu::{GpuContext, GpuUpscaler, probe};
use fluid_upscale::{UpscaleBackend, Upscaler};
use serde_json::json;

/// Source -> destination pairs the suite reports.
const SIZES: [(u32, u32, u32, u32); 2] = [(1280, 720, 2560, 1440), (1920, 1080, 3840, 2160)];

/// One row: filter, sharpening amount.
const CASES: [(UpscaleFilter, f32); 8] = [
    (UpscaleFilter::Nearest, 0.0),
    (UpscaleFilter::Bilinear, 0.0),
    (UpscaleFilter::Bicubic, 0.0),
    (UpscaleFilter::Lanczos, 0.0),
    (UpscaleFilter::Adaptive, 0.0),
    (UpscaleFilter::Temporal, 0.0),
    (UpscaleFilter::Bicubic, 0.4),
    (UpscaleFilter::Lanczos, 0.4),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = Options::parse()?;

    println!(
        "fluid-gpu benchmark (requested backend: {})",
        fluid_gpu::BACKEND_NAME
    );
    let context = match probe() {
        Ok(info) => {
            println!(
                "GPU: {} ({}) vendor {:#06x}",
                info.name, info.backend, info.vendor_id
            );
            println!("limits: {}", info.max_feature_level);
            match GpuContext::new() {
                Ok(context) => Some(Arc::new(context)),
                Err(error) => {
                    println!("GPU unavailable, CPU-only benchmark: {error}");
                    return write_unavailable(&options, &error.to_string());
                }
            }
        }
        Err(error) => {
            println!("no usable GPU adapter, CPU-only benchmark: {error}");
            return write_unavailable(&options, &error.to_string());
        }
    };
    let Some(context) = context else {
        return Ok(());
    };

    println!(
        "warmup {} frame(s), then {} measured frame(s) per case (median reported)",
        options.warmup, options.frames
    );

    let mut rows = Vec::new();
    let mut cpu_total_ns = 0u64;
    let mut gpu_total_ns = 0u64;
    for (src_w, src_h, dst_w, dst_h) in SIZES {
        let src_dims = Dims::new(src_w, src_h);
        let dst_dims = Dims::new(dst_w, dst_h);
        let sequence = sequence(src_dims, options.warmup + options.frames);
        println!(
            "\n{src_dims} -> {dst_dims} ({:.2} MP -> {:.2} MP)",
            megapixels(src_dims),
            megapixels(dst_dims)
        );
        println!(
            "{:<10} {:>8} {:>10} {:>10} {:>8} {:>9} {:>7}",
            "filter", "sharpen", "CPU ms", "GPU ms", "speedup", "max diff", "parity"
        );
        for (filter, sharpness) in CASES {
            let temporal = TemporalConfig::default();
            let config = UpscalerConfig {
                filter,
                sharpness,
                temporal,
                ..UpscalerConfig::default()
            };

            let mut cpu = Upscaler::new(&config, dst_dims);
            let mut cpu_out = Frame::zeroed(dst_dims, PixelFormat::Rgba8);
            let cpu_ns = measure(&mut cpu, &sequence, &mut cpu_out, options.warmup);

            let temporal_option = (filter == UpscaleFilter::Temporal).then_some(temporal);
            let mut gpu = GpuUpscaler::new(
                Arc::clone(&context),
                filter,
                src_dims,
                dst_dims,
                sharpness,
                temporal_option,
            )?;
            let mut gpu_out = Frame::zeroed(dst_dims, PixelFormat::Rgba8);
            let gpu_ns = measure(&mut gpu, &sequence, &mut gpu_out, options.warmup);

            let (max_diff, mean_diff) = diff(&cpu_out, &gpu_out);
            let tolerance = if filter == UpscaleFilter::Nearest {
                0
            } else if sharpness > 0.0 {
                2
            } else {
                1
            };
            let cpu_ms = ns_to_ms(cpu_ns);
            let gpu_ms = ns_to_ms(gpu_ns);
            let speedup = cpu_ns as f64 / gpu_ns.max(1) as f64;
            let parity = max_diff <= tolerance;
            println!(
                "{:<10} {:>8.1} {:>10.3} {:>10.3} {:>7.2}x {:>9} {:>7}",
                filter_name(filter),
                sharpness,
                cpu_ms,
                gpu_ms,
                speedup,
                max_diff,
                if parity { "ok" } else { "FAIL" }
            );
            cpu_total_ns += cpu_ns;
            gpu_total_ns += gpu_ns;
            rows.push(json!({
                "src": format!("{src_w}x{src_h}"),
                "dst": format!("{dst_w}x{dst_h}"),
                "filter": filter_name(filter),
                "sharpness": sharpness,
                "cpu_ms": round3(cpu_ms),
                "gpu_ms": round3(gpu_ms),
                "speedup": round3(speedup),
                "max_abs_diff": max_diff,
                "mean_abs_diff": round4(mean_diff),
                "tolerance_lsb": tolerance,
                "parity": parity,
            }));
        }
    }

    let overall = cpu_total_ns as f64 / gpu_total_ns.max(1) as f64;
    println!(
        "\ntotal: CPU {:.2} ms, GPU {:.2} ms, overall speedup {:.2}x",
        ns_to_ms(cpu_total_ns),
        ns_to_ms(gpu_total_ns),
        overall
    );
    println!(
        "steady state allocates no GPU resources per frame: source, ping-pong, scratch, history,\n\
         staging, tap tables and the parameter block are all created once in GpuUpscaler::new."
    );

    if let Some(path) = &options.json {
        let report = json!({
            "backend_name": fluid_gpu::BACKEND_NAME,
            "gpu_available": true,
            "adapter": {
                "name": context.info().name,
                "backend": context.info().backend,
                "vendor_id": context.info().vendor_id,
                "max_feature_level": context.info().max_feature_level,
            },
            "warmup_frames": options.warmup,
            "measured_frames": options.frames,
            "cpu_total_ms": round3(ns_to_ms(cpu_total_ns)),
            "gpu_total_ms": round3(ns_to_ms(gpu_total_ns)),
            "overall_speedup": round3(overall),
            "cases": rows,
        });
        write_json(path, &report)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

/// A CPU-only report, written when the machine has no usable adapter.
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

/// Measure one backend over a frame sequence and leave the last frame in out.
fn measure(
    backend: &mut dyn UpscaleBackend,
    sequence: &[Frame],
    out: &mut Frame,
    warmup: usize,
) -> u64 {
    let mut sample = |src: &Frame| {
        let started = Instant::now();
        backend.upscale(src, out).expect("benchmark upscale failed");
        started.elapsed().as_nanos() as u64
    };
    for src in &sequence[..warmup.min(sequence.len())] {
        sample(src);
    }
    let mut timings = Vec::with_capacity(sequence.len());
    for src in sequence.iter().skip(warmup.min(sequence.len())) {
        timings.push(sample(src));
    }
    timings.sort_unstable();
    timings[timings.len() / 2]
}

/// Deterministic frame sequence; each frame has the same dimensions.
fn sequence(dims: Dims, frames: usize) -> Vec<Frame> {
    (0..frames.max(1) as u32)
        .map(|seed| pattern(dims, seed))
        .collect()
}

fn pattern(dims: Dims, seed: u32) -> Frame {
    let mut frame = Frame::zeroed(dims, PixelFormat::Rgba8);
    for y in 0..dims.height {
        let row = frame.row_mut(y);
        for x in 0..dims.width {
            let i = x as usize * 4;
            row[i] = (x
                .wrapping_mul(17)
                .wrapping_add(y * 31)
                .wrapping_add(seed * 7)
                % 256) as u8;
            row[i + 1] = (x
                .wrapping_mul(3)
                .wrapping_add(y * 11)
                .wrapping_add(seed * 29)
                % 256) as u8;
            row[i + 2] = (x
                .wrapping_mul(5)
                .wrapping_add(y * 7)
                .wrapping_add(seed * 13)
                % 256) as u8;
            row[i + 3] = (x.wrapping_add(y).wrapping_add(seed) % 256) as u8;
        }
    }
    frame
}

fn diff(a: &Frame, b: &Frame) -> (u8, f64) {
    let mut max = 0u8;
    let mut sum = 0u64;
    for (&x, &y) in a.data.iter().zip(&b.data) {
        let d = x.abs_diff(y);
        max = max.max(d);
        sum += d as u64;
    }
    (max, sum as f64 / a.data.len() as f64)
}

fn filter_name(filter: UpscaleFilter) -> &'static str {
    match filter {
        UpscaleFilter::Off => "off",
        UpscaleFilter::Nearest => "nearest",
        UpscaleFilter::Bilinear => "bilinear",
        UpscaleFilter::Bicubic => "bicubic",
        UpscaleFilter::Lanczos => "lanczos",
        UpscaleFilter::Adaptive => "adaptive",
        UpscaleFilter::Temporal => "temporal",
    }
}

fn megapixels(dims: Dims) -> f64 {
    dims.area() as f64 / 1_000_000.0
}

fn ns_to_ms(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
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

/// Command line: --json <path>, --frames <n>, --warmup <n>.
struct Options {
    json: Option<PathBuf>,
    frames: usize,
    warmup: usize,
}

impl Options {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut options = Options {
            json: None,
            frames: 5,
            warmup: 2,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--json" => {
                    let value = args.next().ok_or("--json needs a path")?;
                    options.json = Some(PathBuf::from(value));
                }
                "--frames" => {
                    let value = args.next().ok_or("--frames needs a count")?;
                    options.frames = value.parse()?;
                }
                "--warmup" => {
                    let value = args.next().ok_or("--warmup needs a count")?;
                    options.warmup = value.parse()?;
                }
                "--help" | "-h" => {
                    println!("usage: bench_gpu [--json <path>] [--frames <n>] [--warmup <n>]");
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other:?}").into()),
            }
        }
        if options.frames == 0 {
            options.frames = 1;
        }
        Ok(options)
    }
}
