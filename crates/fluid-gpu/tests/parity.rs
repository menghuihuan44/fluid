//! CPU-parity tests for the wgpu upscaling backend.
//!
//! The GPU backend is a port of the portable CPU pipeline in 'fluid-upscale',
//! so the contract these tests enforce is simple and strict:
//!
//! - nearest-neighbour and identity resizes are byte-identical to the CPU;
//! - the windowed filters (bilinear, bicubic, Lanczos-3, adaptive) stay within
//!   one least-significant bit per channel (the CPU's SIMD path fuses a
//!   multiply-add, the GPU evaluates a multiply and an add);
//! - sharpening on an exact base is byte-identical, and only widens to two LSBs
//!   when it amplifies a one-LSB base difference;
//! - the temporal blend, which is pure integer arithmetic, stays within one LSB
//!   of the CPU sequence.
//!
//! Machines without a usable adapter skip gracefully: 'probe' returns
//! [fluid_core::FluidError::Unsupported] and every test returns early, so the
//! suite stays green on a headless CI runner while still failing loudly when a
//! probe succeeds but device creation breaks.

use std::sync::{Arc, OnceLock};

use fluid_core::{
    Dims, FluidError, Frame, PixelFormat, TemporalConfig, UpscaleFilter, UpscalerConfig,
};
use fluid_gpu::{BACKEND_NAME, GpuContext, GpuUpscaler, probe};
use fluid_upscale::{UpscaleBackend, Upscaler};

/// Every spatial filter the backend implements.
const FILTERS: [UpscaleFilter; 5] = [
    UpscaleFilter::Nearest,
    UpscaleFilter::Bilinear,
    UpscaleFilter::Bicubic,
    UpscaleFilter::Lanczos,
    UpscaleFilter::Adaptive,
];

/// Every frame format the backend accepts.
const FORMATS: [PixelFormat; 4] = [
    PixelFormat::Rgba8,
    PixelFormat::Bgra8,
    PixelFormat::Rgb8,
    PixelFormat::Luma8,
];

/// An integer 2x resize and a non-integer one with odd sizes.
const RESIZES: [(Dims, Dims); 2] = [
    (Dims::new(96, 54), Dims::new(192, 108)),
    (Dims::new(37, 21), Dims::new(101, 57)),
];

/// One shared device for the whole test binary; None when there is no GPU.
fn gpu() -> Option<Arc<GpuContext>> {
    static CONTEXT: OnceLock<Option<Arc<GpuContext>>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| match probe() {
            Ok(info) => {
                let context =
                    GpuContext::new().expect("probe succeeded but device creation failed");
                eprintln!(
                    "fluid-gpu parity tests on {} ({})",
                    context.info().name,
                    context.info().backend
                );
                let _ = info;
                Some(Arc::new(context))
            }
            Err(FluidError::Unsupported(reason)) => {
                eprintln!("skipping GPU parity tests: no adapter ({reason})");
                None
            }
            Err(other) => panic!("probe failed with an unexpected error: {other}"),
        })
        .clone()
}

/// Deterministic, high-entropy test pattern in any supported format.
fn pattern(dims: Dims, fmt: PixelFormat, seed: u32) -> Frame {
    let mut frame = Frame::zeroed(dims, fmt);
    let bpp = fmt.bytes_per_pixel();
    for y in 0..dims.height {
        let row = frame.row_mut(y);
        for (x, px) in row.chunks_exact_mut(bpp).enumerate() {
            let x = x as u32;
            let r = (x
                .wrapping_mul(17)
                .wrapping_add(y * 31)
                .wrapping_add(seed * 7)
                % 256) as u8;
            let g = (x
                .wrapping_mul(3)
                .wrapping_add(y * 11)
                .wrapping_add(seed * 29)
                % 256) as u8;
            let b = (x
                .wrapping_mul(5)
                .wrapping_add(y * 7)
                .wrapping_add(seed * 13)
                % 256) as u8;
            let a = (x.wrapping_add(y).wrapping_add(seed) % 256) as u8;
            match fmt {
                PixelFormat::Rgba8 => px.copy_from_slice(&[r, g, b, a]),
                PixelFormat::Bgra8 => px.copy_from_slice(&[b, g, r, a]),
                PixelFormat::Rgb8 => px.copy_from_slice(&[r, g, b]),
                PixelFormat::Luma8 => px[0] = r,
            }
        }
    }
    frame
}

/// Largest and mean absolute byte difference between two frames.
fn diff(a: &Frame, b: &Frame) -> (u8, f64) {
    assert_eq!(a.dims, b.dims, "frame dimensions differ");
    assert_eq!(a.fmt, b.fmt, "frame formats differ");
    assert_eq!(a.data.len(), b.data.len(), "frame buffers differ");
    let mut max = 0u8;
    let mut sum = 0u64;
    for (&x, &y) in a.data.iter().zip(&b.data) {
        let d = x.abs_diff(y);
        max = max.max(d);
        sum += d as u64;
    }
    (max, sum as f64 / a.data.len() as f64)
}

/// A CPU reference upscaler and the GPU backend driven with the same config.
struct Pair {
    cpu: Upscaler,
    gpu: GpuUpscaler,
    out_cpu: Frame,
    out_gpu: Frame,
}

impl Pair {
    fn new(
        context: &Arc<GpuContext>,
        filter: UpscaleFilter,
        sharpness: f32,
        src: Dims,
        dst: Dims,
        temporal: TemporalConfig,
    ) -> Self {
        let config = UpscalerConfig {
            filter,
            sharpness,
            temporal,
            ..UpscalerConfig::default()
        };
        let temporal_option = (filter == UpscaleFilter::Temporal).then_some(temporal);
        Self {
            cpu: Upscaler::new(&config, dst),
            gpu: GpuUpscaler::new(
                Arc::clone(context),
                filter,
                src,
                dst,
                sharpness,
                temporal_option,
            )
            .expect("GPU upscaler construction"),
            out_cpu: Frame::zeroed(dst, PixelFormat::Rgba8),
            out_gpu: Frame::zeroed(dst, PixelFormat::Rgba8),
        }
    }

    /// Run one frame through both backends and return (max, mean) LSB diff.
    fn run(&mut self, src: &Frame) -> (u8, f64) {
        self.out_cpu = Frame::zeroed(self.cpu.dst_dims(), src.fmt);
        self.out_gpu = Frame::zeroed(self.gpu.dst_dims(), src.fmt);
        UpscaleBackend::upscale(&mut self.cpu, src, &mut self.out_cpu).expect("CPU upscale");
        self.gpu
            .upscale(src, &mut self.out_gpu)
            .expect("GPU upscale");
        diff(&self.out_cpu, &self.out_gpu)
    }
}

#[test]
fn info_reports_a_real_backend() {
    let Some(context) = gpu() else {
        return;
    };
    let info = context.info();
    assert_eq!(BACKEND_NAME, "wgpu");
    assert!(!info.name.is_empty(), "adapter name must not be empty");
    assert!(
        matches!(info.backend, "vulkan" | "gl" | "dx12" | "metal" | "webgpu"),
        "unexpected backend name {:?}",
        info.backend
    );
    assert_eq!(context.backend(), info.backend);
    let probed = probe().expect("a context exists, so probe must succeed");
    assert_eq!(probed.backend, info.backend);
    assert_eq!(probed.vendor_id, info.vendor_id);
    assert!(!info.max_feature_level.is_empty());
}

#[test]
fn nearest_is_byte_identical() {
    let Some(context) = gpu() else {
        return;
    };
    for fmt in FORMATS {
        for (src_dims, dst_dims) in RESIZES {
            let src = pattern(src_dims, fmt, 1);
            let mut pair = Pair::new(
                &context,
                UpscaleFilter::Nearest,
                0.0,
                src_dims,
                dst_dims,
                TemporalConfig::default(),
            );
            let (max, mean) = pair.run(&src);
            assert_eq!(
                max, 0,
                "nearest {fmt} {src_dims}->{dst_dims} differs (mean {mean:.3} LSB)"
            );
        }
    }
}

#[test]
fn identity_resizes_are_byte_identical() {
    let Some(context) = gpu() else {
        return;
    };
    let dims = Dims::new(64, 36);
    for filter in FILTERS {
        let src = pattern(dims, PixelFormat::Rgba8, 3);
        let mut pair = Pair::new(&context, filter, 0.0, dims, dims, TemporalConfig::default());
        let (max, mean) = pair.run(&src);
        assert_eq!(
            max, 0,
            "{filter:?} identity resize differs (mean {mean:.3} LSB)"
        );
    }
}

#[test]
fn windowed_filters_are_within_one_lsb() {
    let Some(context) = gpu() else {
        return;
    };
    let mut worst = 0u8;
    for filter in [
        UpscaleFilter::Bilinear,
        UpscaleFilter::Bicubic,
        UpscaleFilter::Lanczos,
        UpscaleFilter::Adaptive,
    ] {
        for (src_dims, dst_dims) in RESIZES {
            let src = pattern(src_dims, PixelFormat::Rgba8, 7);
            let mut pair = Pair::new(
                &context,
                filter,
                0.0,
                src_dims,
                dst_dims,
                TemporalConfig::default(),
            );
            let (max, mean) = pair.run(&src);
            println!("{filter:?} {src_dims}->{dst_dims}: max {max} mean {mean:.4} LSB");
            worst = worst.max(max);
            assert!(
                max <= 1,
                "{filter:?} {src_dims}->{dst_dims}: max {max} LSB (mean {mean:.4})"
            );
        }
    }
    println!("worst windowed spatial diff: {worst} LSB");
}

#[test]
fn bgra_rgb_luma_match_within_one_lsb() {
    let Some(context) = gpu() else {
        return;
    };
    for fmt in [PixelFormat::Bgra8, PixelFormat::Rgb8, PixelFormat::Luma8] {
        for filter in [UpscaleFilter::Bilinear, UpscaleFilter::Bicubic] {
            let (src_dims, dst_dims) = RESIZES[1];
            let src = pattern(src_dims, fmt, 11);
            let mut pair = Pair::new(
                &context,
                filter,
                0.0,
                src_dims,
                dst_dims,
                TemporalConfig::default(),
            );
            let (max, mean) = pair.run(&src);
            println!("{filter:?} {fmt}: max {max} mean {mean:.4} LSB");
            assert!(max <= 1, "{filter:?} {fmt}: max {max} LSB (mean {mean:.4})");
        }
    }
}

#[test]
fn identity_with_sharpening_is_byte_identical() {
    let Some(context) = gpu() else {
        return;
    };
    // Identity resize makes the base an exact copy, so the integer sharpen
    // kernel must reproduce the CPU byte for byte even with a strong amount.
    let dims = Dims::new(61, 43);
    for filter in [UpscaleFilter::Bilinear, UpscaleFilter::Lanczos] {
        let src = pattern(dims, PixelFormat::Rgba8, 5);
        let mut pair = Pair::new(&context, filter, 0.6, dims, dims, TemporalConfig::default());
        let (max, mean) = pair.run(&src);
        assert_eq!(
            max, 0,
            "{filter:?} identity sharpen differs (mean {mean:.3} LSB)"
        );
    }
}

#[test]
fn sharpening_is_within_two_lsbs() {
    let Some(context) = gpu() else {
        return;
    };
    // Sharpen amplifies a one-LSB base difference, so the sharpened frame may
    // differ by two LSBs even though the sharpening arithmetic itself is exact.
    for filter in [
        UpscaleFilter::Bilinear,
        UpscaleFilter::Bicubic,
        UpscaleFilter::Lanczos,
        UpscaleFilter::Adaptive,
    ] {
        for sharpness in [0.2, 0.8] {
            let (src_dims, dst_dims) = RESIZES[0];
            let src = pattern(src_dims, PixelFormat::Rgba8, 13);
            let mut pair = Pair::new(
                &context,
                filter,
                sharpness,
                src_dims,
                dst_dims,
                TemporalConfig::default(),
            );
            let (max, mean) = pair.run(&src);
            println!("{filter:?} sharpness {sharpness}: max {max} mean {mean:.4} LSB");
            assert!(
                max <= 2,
                "{filter:?} sharpness {sharpness}: max {max} LSB (mean {mean:.4})"
            );
        }
    }
}

#[test]
fn repeated_calls_are_deterministic() {
    let Some(context) = gpu() else {
        return;
    };
    let (src_dims, dst_dims) = RESIZES[0];
    let src = pattern(src_dims, PixelFormat::Rgba8, 17);
    let mut pair = Pair::new(
        &context,
        UpscaleFilter::Lanczos,
        0.25,
        src_dims,
        dst_dims,
        TemporalConfig::default(),
    );
    let first = {
        pair.run(&src);
        pair.out_gpu.data.clone()
    };
    for _ in 0..3 {
        let (max, _) = pair.run(&src);
        assert_eq!(max, 0, "the CPU reference is deterministic");
        assert_eq!(pair.out_gpu.data, first, "GPU output changed between calls");
    }
}

#[test]
fn temporal_sequence_matches_cpu() {
    let Some(context) = gpu() else {
        return;
    };
    let (src_dims, dst_dims) = RESIZES[0];
    let temporal = TemporalConfig {
        enabled: true,
        blend: 0.35,
        history_frames: 3,
    };
    let mut pair = Pair::new(
        &context,
        UpscaleFilter::Temporal,
        0.0,
        src_dims,
        dst_dims,
        temporal,
    );
    for seed in 0..6 {
        let src = pattern(src_dims, PixelFormat::Rgba8, seed);
        let (max, mean) = pair.run(&src);
        assert!(
            max <= 1,
            "temporal frame {seed}: max {max} LSB (mean {mean:.4})"
        );
    }
}

/// A low-contrast background with a small static dither plus a bright bar that
/// moves a few pixels per frame: some tiles stay static (the blend keeps its
/// configured weight), others move far enough that the motion-adaptive blend
/// must follow the current frame, and the partially covered tiles land in the
/// middle of the smoothstep ramp.
fn motion_frame(dims: Dims, step: u32) -> Frame {
    let mut frame = Frame::zeroed(dims, PixelFormat::Rgba8);
    let bar_x = (step * 5) % dims.width.max(1);
    let bar_y = dims.height / 3;
    for y in 0..dims.height {
        let row = frame.row_mut(y);
        for (x, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let x = x as u32;
            let background = (x + y) % 8 + 96;
            let in_bar = x >= bar_x && x < bar_x + 7 && y >= bar_y && y < bar_y + dims.height / 3;
            let base = if in_bar { 232 } else { background };
            // A one-LSB dither that never moves, so static tiles stay below the
            // motion low threshold.
            let dither = ((x.wrapping_mul(7).wrapping_add(y * 13)) % 3) as i32 - 1;
            let v = (base as i32 + dither).clamp(0, 255) as u8;
            px.copy_from_slice(&[v, v / 2 + 40, 255 - v, 255]);
        }
    }
    frame
}

#[test]
fn temporal_motion_adaptive_matches_cpu() {
    let Some(context) = gpu() else {
        return;
    };
    let src_dims = Dims::new(128, 72);
    let dst_dims = Dims::new(256, 144);
    let temporal = TemporalConfig {
        enabled: true,
        blend: 0.35,
        history_frames: 3,
    };
    let mut pair = Pair::new(
        &context,
        UpscaleFilter::Temporal,
        0.0,
        src_dims,
        dst_dims,
        temporal,
    );
    let mut worst = 0u8;
    for step in 0..6 {
        let src = motion_frame(src_dims, step);
        let (max, mean) = pair.run(&src);
        println!("motion-adaptive temporal frame {step}: max {max} mean {mean:.4} LSB");
        assert!(
            max <= 1,
            "motion-adaptive temporal frame {step}: max {max} LSB (mean {mean:.4})"
        );
        worst = worst.max(max);
    }
    println!("worst motion-adaptive temporal diff: {worst} LSB");
}

#[test]
fn temporal_reset_restarts_the_sequence() {
    let Some(context) = gpu() else {
        return;
    };
    let (src_dims, dst_dims) = RESIZES[0];
    let temporal = TemporalConfig::default();
    let mut pair = Pair::new(
        &context,
        UpscaleFilter::Temporal,
        0.0,
        src_dims,
        dst_dims,
        temporal,
    );
    let first = pattern(src_dims, PixelFormat::Rgba8, 21);
    let second = pattern(src_dims, PixelFormat::Rgba8, 22);
    pair.run(&first);
    pair.run(&second);
    // Reset both backends, then a fresh first frame must match again.
    UpscaleBackend::reset(&mut pair.cpu);
    pair.gpu.reset();
    let (max, mean) = pair.run(&second);
    // After a reset both sides present the plain spatial frame, so this is the
    // same one-LSB budget as the spatial filter.
    assert!(
        max <= 1,
        "after reset the first frame differs by {max} LSB (mean {mean:.3})"
    );
}

#[test]
fn off_identity_is_a_passthrough() {
    let Some(context) = gpu() else {
        return;
    };
    let dims = Dims::new(48, 27);
    let src = pattern(dims, PixelFormat::Rgba8, 23);
    let mut gpu_upscaler = GpuUpscaler::new(
        Arc::clone(&context),
        UpscaleFilter::Off,
        dims,
        dims,
        0.9,
        None,
    )
    .expect("Off identity upscaler");
    let mut out = Frame::zeroed(dims, PixelFormat::Rgba8);
    gpu_upscaler.upscale(&src, &mut out).expect("upscale");
    assert_eq!(out.data, src.data, "Off identity must copy the frame");
}

#[test]
fn trait_and_inherent_paths_agree() {
    let Some(context) = gpu() else {
        return;
    };
    let (src_dims, dst_dims) = RESIZES[1];
    let src = pattern(src_dims, PixelFormat::Bgra8, 27);
    let mut pair = Pair::new(
        &context,
        UpscaleFilter::Adaptive,
        0.3,
        src_dims,
        dst_dims,
        TemporalConfig::default(),
    );
    assert_eq!(pair.gpu.backend_name(), BACKEND_NAME);
    assert_eq!(pair.gpu.src_dims(), src_dims);
    assert_eq!(pair.gpu.dst_dims(), dst_dims);
    // Pair::run drives the CPU through the trait and the GPU inherently; drive
    // them again swapped to prove both entry points agree.
    let (max, _) = pair.run(&src);
    assert!(max <= 2);
    let mut out_a = Frame::zeroed(dst_dims, src.fmt);
    let mut out_b = Frame::zeroed(dst_dims, src.fmt);
    pair.gpu.upscale(&src, &mut out_a).expect("inherent");
    UpscaleBackend::upscale(&mut pair.gpu, &src, &mut out_b).expect("trait");
    assert_eq!(out_a.data, out_b.data);
    assert!(pair.gpu.last_cost_ns() > 0);
}

#[test]
fn invalid_arguments_are_rejected() {
    let Some(context) = gpu() else {
        return;
    };
    let ok_dst = Dims::new(32, 18);
    let empty = Dims::new(0, 0);
    let err = GpuUpscaler::new(
        Arc::clone(&context),
        UpscaleFilter::Bilinear,
        empty,
        ok_dst,
        0.0,
        None,
    )
    .expect_err("empty source must be rejected");
    assert!(matches!(err, FluidError::InvalidArgument(_)));
    let err = GpuUpscaler::new(
        Arc::clone(&context),
        UpscaleFilter::Bilinear,
        Dims::new(16, 9),
        empty,
        0.0,
        None,
    )
    .expect_err("empty destination must be rejected");
    assert!(matches!(err, FluidError::InvalidArgument(_)));

    let src_dims = Dims::new(16, 9);
    let mut gpu_upscaler = GpuUpscaler::new(
        Arc::clone(&context),
        UpscaleFilter::Bilinear,
        src_dims,
        ok_dst,
        0.0,
        None,
    )
    .expect("valid upscaler");
    let wrong_source = pattern(Dims::new(17, 9), PixelFormat::Rgba8, 29);
    let mut out = Frame::zeroed(ok_dst, PixelFormat::Rgba8);
    let err = gpu_upscaler
        .upscale(&wrong_source, &mut out)
        .expect_err("wrong source dimensions must be rejected");
    assert!(matches!(err, FluidError::InvalidArgument(_)));

    let src = pattern(src_dims, PixelFormat::Rgba8, 30);
    let mut wrong_out = Frame::zeroed(Dims::new(32, 19), PixelFormat::Rgba8);
    let err = gpu_upscaler
        .upscale(&src, &mut wrong_out)
        .expect_err("wrong output dimensions must be rejected");
    assert!(matches!(err, FluidError::InvalidArgument(_)));

    let mut wrong_fmt = Frame::zeroed(ok_dst, PixelFormat::Bgra8);
    let err = gpu_upscaler
        .upscale(&src, &mut wrong_fmt)
        .expect_err("wrong output format must be rejected");
    assert!(matches!(err, FluidError::InvalidArgument(_)));
}

#[test]
fn probe_is_consistent_with_an_existing_context() {
    let Some(context) = gpu() else {
        return;
    };
    let info = probe().expect("probe must succeed when a context exists");
    assert_eq!(info.name, context.info().name);
    assert_eq!(info.backend, context.info().backend);
}
/// Every filter must build its compute pipeline on the real adapter.
///
/// This is the regression test for shader typos: wgpu parses and validates the
/// WGSL at create_shader_module/create_compute_pipeline time, and the pipeline
/// builder surfaces that as an Err (never a panic), so a reserved keyword or a
/// malformed expression fails here instead of inside a frame.
#[test]
fn every_filter_builds_its_pipeline() {
    let Some(context) = gpu() else {
        eprintln!("skipping GPU pipeline validation: no adapter");
        return;
    };
    let (_, dst_dims) = RESIZES[0];
    let src_dims = RESIZES[0].0;
    for filter in FILTERS
        .iter()
        .copied()
        .chain([UpscaleFilter::Off, UpscaleFilter::Temporal])
    {
        let temporal = (filter == UpscaleFilter::Temporal).then(TemporalConfig::default);
        let mut upscaler = GpuUpscaler::new(
            Arc::clone(&context),
            filter,
            src_dims,
            dst_dims,
            0.3,
            temporal,
        )
        .unwrap_or_else(|error| panic!("{filter:?} pipeline failed to build: {error}"));

        // Run two frames so the temporal blend and the sharpen-from-history bind
        // groups are exercised too, not just the spatial schedule.
        let mut out = Frame::zeroed(dst_dims, PixelFormat::Rgba8);
        for seed in 0..2 {
            let src = pattern(src_dims, PixelFormat::Rgba8, seed);
            upscaler
                .upscale(&src, &mut out)
                .unwrap_or_else(|error| panic!("{filter:?} frame {seed} failed: {error}"));
        }
        assert_eq!(out.dims, dst_dims);
    }
}

/// Identity resizes must also build for every filter (copy/nearest paths).
#[test]
fn every_filter_builds_on_identity_resizes() {
    let Some(context) = gpu() else {
        return;
    };
    let dims = Dims::new(61, 43);
    for filter in FILTERS
        .iter()
        .copied()
        .chain([UpscaleFilter::Off, UpscaleFilter::Temporal])
    {
        let temporal = (filter == UpscaleFilter::Temporal).then(TemporalConfig::default);
        let mut upscaler =
            GpuUpscaler::new(Arc::clone(&context), filter, dims, dims, 0.0, temporal)
                .unwrap_or_else(|error| panic!("{filter:?} identity pipeline failed: {error}"));
        let src = pattern(dims, PixelFormat::Rgba8, 5);
        let mut out = Frame::zeroed(dims, PixelFormat::Rgba8);
        upscaler
            .upscale(&src, &mut out)
            .unwrap_or_else(|error| panic!("{filter:?} identity frame failed: {error}"));
        assert_eq!(
            out.data, src.data,
            "{filter:?} identity resize must be exact"
        );
    }
}
