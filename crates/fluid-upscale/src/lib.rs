#![forbid(unsafe_code)]
//! Spatial and temporal upscaling.
//!
//! # Algorithm
//!
//! Every spatial filter is separable: a horizontal pass resamples source rows
//! into a u8 scratch buffer, then a vertical pass resamples those scratch rows
//! into the destination frame. For each output coordinate both axes precompute
//! a short tap table (source indices, edge clamped, plus f32 weights normalised
//! to sum to one), so the inner loops are fixed-length dot products with no
//! per-pixel coordinate arithmetic and flat sources reconstruct exactly.
//!
//! * Nearest: 1 tap, an index copy on both axes.
//! * Bilinear: 2 taps, triangle kernel.
//! * Bicubic: 4 taps, Catmull-Rom (a = -0.5).
//! * Lanczos: 6 taps, Lanczos-3 windowed sinc.
//! * Adaptive (FSR1/EASU style): Catmull-Rom base plus a gradient-directed
//!   4-tap detail restore fused into the vertical pass, using the three luma
//!   rows already in flight. Along strong edges it only sharpens across the
//!   edge, which keeps diagonals and checkerboards crisp at a fraction of a
//!   full EASU kernel's cost.
//! * Temporal: Adaptive plus a motion-adaptive exponential moving average
//!   against a history frame to damp shimmer. The blend weight comes from the
//!   temporal config and the history length scales the horizon. Where the
//!   current frame and the history agree the configured EMA weight applies
//!   exactly; where they differ the weight ramps toward the current frame so
//!   fast content leaves no trail. History is stored pre-sharpening so
//!   sharpening is never applied twice.
//!
//! The motion test is a 16x16 tile grid evaluated before the blend: the luma of
//! four samples per tile in the resized current frame is compared with the same
//! samples of the history frame, the absolute differences are averaged, and the
//! mean maps through a smoothstep from 8 to 40 luma levels to a per-tile alpha:
//! alpha = base + (1 - base) * smoothstep(8, 40, diff). A static scene stays at
//! base bit for bit (dither-level differences average well under 8, so the
//! flicker metric is unchanged), while real motion inside a tile pushes the
//! blend toward the current frame and removes the EMA ghost. Note the direction:
//! the configured blend is the weight of the current frame, not of the history,
//! so "blend less on motion" means a LARGER weight here. Weights are twelve-bit
//! fixed point, computed once per tile in row bands and reused for every pixel,
//! so parallel and serial runs produce identical bytes for any worker count.
//!
//! Sharpening is a contrast-adaptive unsharp mask (RCAS-ish): a 3x3 cross
//! high pass, scaled by the amount and steered along the dominant local
//! gradient, is clamped so the result stays inside the full 3x3 min/max
//! envelope of the source. Soft edges steepen while hard steps and every
//! existing local extreme are left alone, so no halo or new extremum appears.
//! It runs on luma and applies the same delta to r/g/b, never to alpha.
//!
//! Bicubic, Lanczos and Adaptive are anti-ringed in the FSR RCAS style: every
//! output sample is clamped to the min/max luma of the source samples inside
//! its own tap footprint. The horizontal pass builds a per-column bounds table
//! from a sliding-window min/max over the source luma row, and the vertical
//! pass combines the bounds of its tap rows and clamps the finished row, so
//! the stage rides inside the existing two-pass structure with no extra
//! full-frame pass. This removes the ringing halos around hard edges and thin
//! lines, at the cost of the analytic zone-plate band (see Quality).
//!
//! All pixels formats Rgba8/Bgra8/Rgb8/Luma8 are supported and preserved; alpha
//! is copied through untouched. Sampling is edge clamped on all four sides.
//! The row buffers are pre-allocated; the row loops allocate nothing and the
//! crate contains no unsafe code.
//!
//! # SIMD
//!
//! The hot kernels are dispatched at run time with `pulp`: `simd::dispatch`
//! detects the local CPU once per pass (AVX2 on x86-64, NEON on aarch64) and
//! runs the whole worker band inside a function compiled for that ISA, so LLVM
//! vectorises the vertical resample loop, the sharpen kernel and the block
//! copies with 256-bit instructions. Calling `pulp::Arch::dispatch` directly is
//! not enough: it routes through an extra generic layer that drops the
//! `target_feature` attributes, so the helper matches the arch token and calls
//! the token's own `vectorize` method instead. The vertical pass keeps a
//! 16-byte register accumulator per output chunk and streams one source row per
//! tap in ascending weight order, so the dispatched code performs the same IEEE
//! operations in the same order as the scalar loop. The horizontal pass uses
//! the `wide` crate's `f32x4` on the four RGBA channels of one pixel, and the
//! anti-ringing bounds tables use 16-lane min/max. The temporal blend stays
//! scalar: the u8 -> u32 -> u16 -> u8 widening and saturating chain cost more
//! than the integer blend itself (in-process measurement: 17.1 ms vector vs
//! 7.2 ms scalar over a 4K frame).
//!
//! # Backends
//!
//! [`UpscaleBackend`] is the object-safe, Send interface the engine drives: the
//! caller owns the destination frame and a backend fills it in place. A GPU
//! backend (compute shaders) plugs in without engine changes, while the CPU
//! [`Upscaler`] implements the same trait with byte-identical output for every
//! filter, including temporal history. The inherent methods stay available.
//!
//! # Parallelism
//!
//! Every full-frame pass splits its rows into contiguous chunks and runs them on
//! `std::thread::scope` workers: the horizontal resample pass splits source
//! rows, the vertical pass splits destination rows, and the sharpen pass, the
//! temporal blend and the history snapshot do the same. Chunks touch disjoint
//! output, and the adaptive filter recomputes the luma rows at a chunk boundary
//! from the scratch buffer instead of reading a neighbour's output. The sharpen
//! pass copies the unsharpened luma of the rows just outside each band before
//! spawning, so band-local state is identical to the serial ring buffer. Every
//! output byte therefore runs exactly the same arithmetic as before and the
//! parallel output is bit-identical to the serial path for any worker count.
//! Frames below 64K pixels, and frames with fewer than 24 rows per available
//! worker, stay serial so thread startup never dominates. The worker count is
//! `available_parallelism`, optionally overridden by `FLUID_THREADS`.
//!
//! # Performance
//!
//! Release, 1920x1080 -> 3840x2160 RGBA8, i7-1260P (16 logical cores), best of
//! three runs of the in-crate probes; the machine is shared, so end-to-end
//! numbers wobble by +/-30% between runs and the per-kernel ratios below come
//! from same-process A/B measurements.
//!
//! `Upscaler::upscale` at sharpness 0 / 0.3: Nearest 4.8/15.9 ms, Bilinear
//! 11.1/23.4 ms, Bicubic 15.0/27.6 ms, Lanczos 19.9/32.5 ms, Adaptive
//! 24.3/38.2 ms, Temporal 25.3/39.9 ms (temporal disabled/enabled). Forcing one
//! worker gives 16.9/64.98/83.27/104.5/137.6 ms for Nearest/Bilinear/Bicubic/
//! Lanczos/Adaptive, i.e. 4.3-4.9x scaling at 16 workers. 1280x720 -> 2560x1440
//! Bilinear end-to-end is 5.7/11.4 ms.
//!
//! The sharpen stage is 9.5-11.8 ms over 8.3 Mpx with 16 workers (57-63 ms serial),
//! and the fused blend + history snapshot stage 5.9 ms (11.7-12.8 ms serial). Reusing
//! the upscaler's scratch buffer instead of reallocating it per frame saves about
//! 4 ms at 4K.
//!
//! Same-process SIMD vs scalar A/B: the vertical pass (bicubic, 4 taps, 4K wide,
//! 1080 source rows) is 37.6 ms as a row-accumulator f32x4 loop versus 66.8 ms
//! for the per-lane tap loop it replaced, and 44.5 ms for the scalar row
//! accumulator; the horizontal pass is 27.8 vs 29.3 ms scalar (a few percent);
//! the sharpen kernel is 57.2 vs 77.0 ms scalar over 4K rows.
//!
//! Runtime dispatch (task-54) is worth about 5% on the vertical pass of 4K
//! output (13.2 versus 13.9 ms for 540 rows, same-process interleaved A/B),
//! roughly 1.0-1.3x on the sharpen kernel, and nothing measurable elsewhere.
//! At 4K the resize passes are DRAM-bandwidth bound: a bilinear
//! 1920x1080 -> 3840x2160 frame streams about 100 MB of source and destination
//! rows, and the parallel pass already sustains 5-6 GB/s, so widening the
//! arithmetic cannot deliver 1.25x end to end. The disassembly of the
//! dispatched bodies does contain 256-bit `vaddps`/`vmulps`, so the limit is
//! memory traffic rather than code generation.
//!
//! Anti-ringing costs about 8-15% on the Adaptive 2x path (serial, identical
//! machine state: 73.0 ms without the clamp versus 79.2 ms with it, best of
//! interleaved runs; medians 86 versus 99 ms). The source luma plane, the
//! horizontal bounds table and the vertical clamp are fused into the existing
//! passes, and the per-column lookup uses a sliding-window u8x16 min/max table
//! instead of re-reducing each tap window.
//!
//! Motion-adaptive temporal blending adds one tile pass over 16x16 blocks plus a
//! per-tile weight lookup in the blend. Measured over 8.3 Mpx with 16 workers:
//! the weights pass is 0.63 ms and the blended snapshot 1.21 ms against 1.40 ms
//! for plain EMA (1280x720: 0.59 + 0.88 versus 0.95 ms), i.e. well under 2% of a
//! Temporal frame.

//! # Quality
//!
//! The `quality` module scores every filter against analytic ground truth: a
//! six-band chart (hard edges, thin lines, checkerboard, zone plate, glyph
//! blocks, gradients) rendered at the native output size, downscaled with an
//! exact box average, upscaled again, then compared with PSNR (mean and worst
//! tile), SSIM (11x11 Gaussian), a gradient-energy ratio and a temporal flicker
//! metric. It is analysis-only and never runs on the pipeline path.
//!
//! Regenerate the report with:
//!
//! ```text
//! cargo run --release -p fluid-upscale --example quality -- --json out/upscale-quality.json
//! ```
//!
//! Measured on the 54-scenario matrix (three reference sizes x six filters x
//! sharpness 0.0/0.3/0.6; quality numbers are deterministic, the millisecond
//! columns swing with load):
//!
//! - An exact 2x factor is the only case where Nearest wins mean PSNR, because
//!   duplicating the block average is the inverse of the box downscaler:
//!   1280x720 from 640x360 gives 15.89/15.84/15.80 dB at sharpness 0.0/0.3/0.6
//!   against bilinear 14.57/14.74/14.80, bicubic 15.37/15.53/15.54, Lanczos
//!   15.48/15.68/15.74, Adaptive 15.50/15.53/15.49 and Temporal
//!   23.87/23.78/23.74. At 1920x1080 from 960x540 the ordering is the same
//!   (nearest 16.18-16.12, Lanczos 15.74-15.97, Adaptive 15.69-15.72, bicubic
//!   15.61-15.74, bilinear 14.84-15.01).
//! - The reconstructive kernels trade a couple of tenths of a dB for smoother
//!   output: at 1920x1080 they keep more gradient energy than bilinear (Lanczos
//!   0.320, Adaptive 0.302, bicubic 0.297 versus 0.224) and win the zone-plate
//!   band (36.5 dB bicubic, 37.9 Adaptive, 36.8 Lanczos versus 26.4 nearest and
//!   31.6 bilinear), while nearest's block edges inflate its own gradient ratio
//!   to 0.436.
//! - At 3x (640x360 -> 1920x1080) Adaptive is the best spatial filter: 13.80 dB
//!   at sharpness 0.0 and 13.85 at 0.3/0.6 versus nearest 13.74-13.77, Lanczos
//!   13.76-13.80, bicubic 13.76-13.83, bilinear 13.68-13.72 dB.
//! - Anti-ringing (Bicubic, Lanczos, Adaptive) buys edge quality: against the
//!   pre-clamp build the hard-edge band gains 0.21/0.55/0.47 dB for bicubic,
//!   0.48/1.23/1.42 for Lanczos and 0.49/0.49/0.69 for Adaptive at 1920x1080
//!   sharpness 0.0/0.3/0.6 (at 1280x720 the gains are up to 0.63/1.53/1.81 dB).
//!   Thin lines gain 0.0-0.22 dB, mean PSNR for the clamped filters never drops
//!   (worst case 0.00 dB), SSIM rises by 0.0001-0.0139 and flicker falls by
//!   0.10-0.29. The cost is the zone-plate band, 0.04-0.41 dB down at 2x and
//!   0.23-1.24 dB down at 3x: that analytic texture sits above the input
//!   Nyquist, where ringing happens to land closer to the reference, so the
//!   clamp trades it for clean edges. Nearest and Bilinear never get the clamp,
//!   so their unsharpened output is unchanged; with sharpening enabled they move
//!   by at most 0.03 dB because the sharpen envelope widened to the full 3x3.
//! - Sharpening 0.3/0.6 raises the hard-edge band through that same envelope
//!   clamp (1920x1080 bicubic 32.3 -> 34.0/34.5 dB, Lanczos 32.5 -> 34.3/35.3,
//!   Adaptive 33.5 -> 34.3/34.0), lifts overall PSNR by 0.01-0.10 dB and SSIM by
//!   0.001-0.017. Nearest barely moves (within 0.05 dB) because its output is
//!   already blocky.
//! - Temporal stability: the static-chart flicker metric (up to three LSB of
//!   input dither) is 0.05-0.12, against 1.46-2.36 for the spatial filters,
//!   roughly 20x less visible noise, and motion-adaptive blending leaves it
//!   bit-identical: dither differences stay below the 8-level motion threshold.
//! - Temporal motion: the harness renders a disc sliding 2/6/12/24 reference
//!   pixels per frame over six frames and scores every output against its own
//!   reference. Against plain EMA the worst-frame PSNR rises 0.58/1.82/3.17/4.77
//!   dB at 1920x1080 and 0.81/2.46/4.04/5.29 dB at 1280x720, the trail error
//!   behind the disc drops from 97-113 to 0-9.6 u8 levels, and the speed-0
//!   control is unchanged. On the moving-shape temporal chart mean PSNR rises
//!   from 20.4-21.1 to 22.9-24.2 dB because EMA lagged the shape and the
//!   motion-adaptive blend follows it.
//! - The worst 64x64 tile is 7.4-7.5 dB for every spatial filter by design: the
//!   one-pixel checkerboard and the one-pixel lines are at the Nyquist limit of
//!   the low-resolution input, so the box downscale erases them before any filter
//!   runs. The per-band breakdown in the JSON separates that from real losses.

mod coeff;
mod pixel;
pub mod quality;
mod resample;
mod sharpen;
mod simd;
use std::time::Instant;

use fluid_core::{
    Dims, FluidError, Frame, PixelFormat, Result, TemporalConfig, UpscaleFilter, UpscalerConfig,
};

/// Backend-agnostic upscaling interface.
///
/// The engine can hold a `Box<dyn UpscaleBackend>` and drive the CPU upscaler or
/// a GPU backend without knowing which one it got. The trait is object safe and
/// Send (a pipeline may own the backend on its own thread), and every type in
/// its signatures is either a fluid-core type or a plain value, so another crate
/// can implement it without private types leaking.
///
/// Contract:
/// - upscale writes exactly dst_dims() pixels, in src.fmt, into out. The frame
///   must already have those dimensions and format (the engine knows both from
///   dst_dims() and src.fmt); otherwise FluidError::invalid is returned. On
///   success out carries src.pts_ns and src.index.
/// - Stateful backends keep temporal history across calls; reset drops that
///   history (scene cut) without changing the geometry configuration.
/// - The output must be deterministic for identical inputs and history.
///
/// The CPU Upscaler implements this with backend_name() == "cpu" and
/// byte-identical output to its inherent upscale method.
pub trait UpscaleBackend: Send {
    /// Short stable name for telemetry and logs (for example "cpu", "d3d11").
    fn backend_name(&self) -> &'static str;

    /// Source dimensions the backend was last driven with.
    ///
    /// The CPU backend learns these from the frames it receives, so before the
    /// first upscale call this is Dims::default() (0x0). A backend with a fixed
    /// input size (typical for GPU pipelines) can report it at construction.
    fn src_dims(&self) -> Dims;

    /// Output dimensions every upscale call produces.
    fn dst_dims(&self) -> Dims;

    /// Upscale src into out; see the trait contract for the buffer rules.
    fn upscale(&mut self, src: &Frame, out: &mut Frame) -> Result<()>;

    /// Drop temporal history without touching the configuration.
    fn reset(&mut self);

    /// Wall-clock duration of the last upscale call, in nanoseconds.
    ///
    /// Backends that do not measure it keep the default of 0.
    fn last_cost_ns(&self) -> u64 {
        0
    }
}

/// Stateful upscaler. Tracks history for the temporal filter.
#[derive(Clone, Debug)]
pub struct Upscaler {
    cfg: UpscalerConfig,
    dst: Dims,
    history: Option<Frame>,
    /// Per-stream resample scratch (horizontal rows, luma plan, bounds) reused
    /// across frames so the stateful path does
    /// not allocate tens of megabytes per frame.
    scratch: resample::ResampleScratch,
    /// Reusable fixed-tile motion estimate for the motion-adaptive blend.
    motion: MotionGrid,
    /// Source dimensions of the last successful upscale (backend trait report).
    last_src: Dims,
    last_cost_ns: u64,
}

impl Upscaler {
    pub fn new(cfg: &UpscalerConfig, dst: Dims) -> Self {
        Self {
            cfg: *cfg,
            dst,
            history: None,
            scratch: resample::ResampleScratch::new(),
            motion: MotionGrid::default(),
            last_src: Dims::default(),
            last_cost_ns: 0,
        }
    }

    pub fn dst_dims(&self) -> Dims {
        self.dst
    }

    pub fn filter(&self) -> UpscaleFilter {
        self.cfg.filter
    }

    /// Drop temporal history (scene cut, resolution change).
    pub fn reset(&mut self) {
        self.history = None;
    }

    pub fn last_cost_ns(&self) -> u64 {
        self.last_cost_ns
    }

    /// Convert one frame to the configured output resolution.
    ///
    /// Order of operations: spatial (or temporal) resize, temporal history
    /// blend, then sharpening. The Off filter with a matching destination size
    /// is an exact pass-through (this also skips sharpening: Off means "do
    /// nothing"). Off with a different destination falls back to bilinear.
    pub fn upscale(&mut self, src: &Frame) -> Result<Frame> {
        resample::check_dst(self.dst)?;
        let mut out = Frame::zeroed(self.dst, src.fmt);
        self.upscale_into(src, &mut out)?;
        Ok(out)
    }

    /// Shared implementation of the inherent and trait entry points: filters,
    /// blends and sharpens src into out, updating the history and the cost
    /// measurement.
    ///
    /// The caller owns the output buffer, so a trait-driven engine reuses one
    /// frame per stream instead of allocating one per frame. out must have
    /// dst_dims() and src.fmt; check_out rejects anything else.
    fn upscale_into(&mut self, src: &Frame, out: &mut Frame) -> Result<()> {
        let started = Instant::now();
        resample::check_dst(self.dst)?;
        check_out(self.dst, src.fmt, out)?;
        let filter = self.cfg.filter;
        let threads = resample::threads_for(
            self.dst.area().max(src.dims.area()),
            self.dst.height.max(src.dims.height) as usize,
        );
        if filter == UpscaleFilter::Off && src.dims == self.dst {
            // Exact pass-through: Off means do nothing, not even sharpening.
            out.data.copy_from_slice(&src.data);
        } else {
            let effective = if filter == UpscaleFilter::Off {
                UpscaleFilter::Bilinear
            } else {
                filter
            };
            resample::resample_into(src, self.dst, effective, threads, &mut self.scratch, out)?;
        }

        let temporal = self.cfg.temporal;
        if filter == UpscaleFilter::Temporal && temporal.enabled {
            let blend = temporal_blend_alpha(&temporal);
            let previous = self.history.take();
            let had_history = matches!(&previous, Some(h)
                if h.dims == out.dims && h.fmt == out.fmt);
            let mut history = match previous {
                Some(hist) if had_history => hist,
                _ => Frame::zeroed(out.dims, out.fmt),
            };
            if had_history {
                temporal_blend_motion(&mut history, out, blend, &mut self.motion);
            } else {
                copy_frame_into(out, &mut history, threads);
            }
            self.history = Some(history);
        } else {
            self.history = None;
        }

        out.pts_ns = src.pts_ns;
        out.index = src.index;
        if self.cfg.sharpness > 0.0 && filter != UpscaleFilter::Off {
            sharpen::sharpen_frame(out, self.cfg.sharpness);
        }
        self.last_src = src.dims;
        self.last_cost_ns = started.elapsed().as_nanos() as u64;
        Ok(())
    }
}

impl UpscaleBackend for Upscaler {
    fn backend_name(&self) -> &'static str {
        "cpu"
    }

    fn src_dims(&self) -> Dims {
        self.last_src
    }

    fn dst_dims(&self) -> Dims {
        self.dst
    }

    fn upscale(&mut self, src: &Frame, out: &mut Frame) -> Result<()> {
        self.upscale_into(src, out)
    }

    fn reset(&mut self) {
        Upscaler::reset(self);
    }

    fn last_cost_ns(&self) -> u64 {
        self.last_cost_ns
    }
}

/// Validate a caller-owned output frame against the destination contract.
fn check_out(dst: Dims, fmt: PixelFormat, out: &Frame) -> Result<()> {
    if out.dims != dst || out.fmt != fmt {
        return Err(FluidError::invalid(format!(
            "upscale: output frame is {} {}, expected {} {}",
            out.dims, out.fmt, dst, fmt
        )));
    }
    if out.data.len() != fmt.buffer_len(dst) {
        return Err(FluidError::invalid(
            "upscale: output frame buffer length does not match its dimensions",
        ));
    }
    Ok(())
}

/// Per-frame temporal blend factor: the configured blend is spread over the
/// history horizon so that history_frames = 1 gives the configured blend and
/// longer horizons respond more slowly. Always within 0..=1.
fn temporal_blend_alpha(t: &TemporalConfig) -> f32 {
    let frames = t.history_frames.clamp(1, 8) as f32;
    if t.blend.is_finite() {
        (t.blend * 2.0 / frames).clamp(0.0, 1.0)
    } else {
        0.12
    }
}

/// Stateless resize with the given filter.
///
/// Temporal behaves like Adaptive here (there is no history to blend with), and
/// Off is an exact pass-through when the sizes match, otherwise bilinear.
pub fn resize(src: &Frame, dst: Dims, filter: UpscaleFilter) -> Result<Frame> {
    resample::check_dst(dst)?;
    match filter {
        UpscaleFilter::Off if src.dims != dst => {
            resample::resample(src, dst, UpscaleFilter::Bilinear)
        }
        other => resample::resample(src, dst, other),
    }
}

/// Resize followed by sharpening.
pub fn resize_sharp(
    src: &Frame,
    dst: Dims,
    filter: UpscaleFilter,
    sharpness: f32,
) -> Result<Frame> {
    let mut out = resize(src, dst, filter)?;
    sharpen_in_place(&mut out, sharpness);
    Ok(out)
}

/// Contrast-adaptive unsharp mask, amount in 0..=1 (0 and non-finite are no-ops).
pub fn sharpen_in_place(frame: &mut Frame, amount: f32) {
    sharpen::sharpen_frame(frame, amount);
}

/// Blend a history frame with the current frame (0 = history, 1 = current).
///
/// Mismatched shapes or formats return the current frame unchanged. The result
/// carries the current frame's pts/index.
pub fn temporal_blend(history: &Frame, current: &Frame, blend: f32) -> Frame {
    if history.dims != current.dims || history.fmt != current.fmt {
        return current.clone();
    }
    let b = if blend.is_finite() {
        blend.clamp(0.0, 1.0)
    } else {
        1.0
    };
    if b >= 1.0 {
        return current.clone();
    }
    let mut out = current.clone();
    temporal_blend_in_place(history, &mut out, b);
    out
}

/// Fixed point 12-bit exponential moving average, written in place over
/// current (0 = history, 1 = current, non-finite = current).
///
/// The caller guarantees identical dims and formats. Row bands are processed by
/// the same worker count as the spatial passes; every byte is a pure function of
/// the two inputs, so the parallel result is bit-identical to the serial one.
fn temporal_blend_in_place(history: &Frame, current: &mut Frame, blend: f32) {
    let threads = resample::threads_for(current.dims.area(), current.dims.height as usize);
    temporal_blend_threads(history, current, blend, threads);
}

/// Temporal blend with an explicit worker count (1 is the serial path).
fn temporal_blend_threads(history: &Frame, current: &mut Frame, blend: f32, threads: usize) {
    debug_assert!(history.dims == current.dims && history.fmt == current.fmt);
    let b = if blend.is_finite() {
        blend.clamp(0.0, 1.0)
    } else {
        1.0
    };
    if b >= 1.0 {
        return;
    }
    let s = (b * 4096.0 + 0.5) as i32;
    if s <= 0 {
        current.data.copy_from_slice(&history.data);
        return;
    }
    let stride = current.stride();
    let rows = current.dims.height as usize;
    let chunks = resample::row_chunks(rows, threads.max(1));
    let h: &[u8] = &history.data;
    if chunks.len() <= 1 {
        for y in 0..rows {
            blend_row(
                current.row_mut(y as u32),
                &h[y * stride..(y + 1) * stride],
                s,
            );
        }
        return;
    }
    let data: &mut [u8] = &mut current.data[..rows * stride];
    std::thread::scope(|scope| {
        let mut rest: &mut [u8] = data;
        for &(y0, y1) in &chunks {
            let len = (y1 - y0) * stride;
            let (head, tail) = rest.split_at_mut(len);
            rest = tail;
            scope.spawn(move || {
                for (i, y) in (y0..y1).enumerate() {
                    blend_row(
                        &mut head[i * stride..(i + 1) * stride],
                        &h[y * stride..(y + 1) * stride],
                        s,
                    );
                }
            });
        }
    });
}

/// Blend one byte: history + ((current - history) * s + 2048) >> 12.
///
/// `s` is the twelve-bit fixed point weight, so the arithmetic shift rounds the
/// signed delta down exactly like the reference formula. The result always lies
/// between the two inputs, so it needs no clamp.
#[inline]
fn blend_byte(current: u8, history: u8, s: i32) -> u8 {
    let d = current as i32 - history as i32;
    (history as i32 + ((d * s + 2048) >> 12)) as u8
}

/// Blend history into current in place.
#[inline]
pub(crate) fn blend_row(current: &mut [u8], history: &[u8], s: i32) {
    let n = current.len().min(history.len());
    for (c, h) in current[..n].iter_mut().zip(&history[..n]) {
        *c = blend_byte(*c, *h, s);
    }
}

/// Blend history into current and write the blended result straight back into
/// the history buffer in the same pass: the caller needs both the presented
/// frame and the next history snapshot, and they are the same bytes before
/// sharpening, so this saves a whole full-frame copy per temporal frame.
///
/// This convenience wrapper resolves the default worker count; the shipped
/// motion-adaptive path calls the explicit-thread form directly.
#[cfg(test)]
fn temporal_blend_snapshot(history: &mut Frame, current: &mut Frame, blend: f32) {
    debug_assert!(history.dims == current.dims && history.fmt == current.fmt);
    let b = if blend.is_finite() {
        blend.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let s = (b * 4096.0 + 0.5) as i32;
    let threads = resample::threads_for(current.dims.area(), current.dims.height as usize);
    temporal_blend_snapshot_threads(history, current, s, threads);
}

/// Fused blend/snapshot with an explicit worker count (1 is the serial path).
fn temporal_blend_snapshot_threads(
    history: &mut Frame,
    current: &mut Frame,
    s: i32,
    threads: usize,
) {
    let stride = current.stride();
    let rows = current.dims.height as usize;
    let chunks = resample::row_chunks(rows, threads.max(1));
    let hdata: &mut [u8] = &mut history.data[..rows * stride];
    let cdata: &mut [u8] = &mut current.data[..rows * stride];
    if chunks.len() <= 1 {
        for y in 0..rows {
            blend_snapshot_row(
                &mut hdata[y * stride..(y + 1) * stride],
                &mut cdata[y * stride..(y + 1) * stride],
                s,
            );
        }
        return;
    }
    std::thread::scope(|scope| {
        let mut hrest: &mut [u8] = hdata;
        let mut crest: &mut [u8] = cdata;
        for &(y0, y1) in &chunks {
            let len = (y1 - y0) * stride;
            let (hhead, htail) = hrest.split_at_mut(len);
            hrest = htail;
            let (chead, ctail) = crest.split_at_mut(len);
            crest = ctail;
            scope.spawn(move || blend_snapshot_row(hhead, chead, s));
        }
    });
}

/// Blend history into current and write the blended bytes back into both
/// buffers in one pass: the presented frame and the next history snapshot are
/// the same bytes before sharpening, so this saves a full-frame copy.
///
/// Each byte is read and written independently, so overwriting the history byte
/// it was read from is safe. Row bands are split like the other passes and the
/// result is bit-identical to the serial one.
#[inline]
fn blend_snapshot_row(history: &mut [u8], current: &mut [u8], s: i32) {
    let n = history.len().min(current.len());
    for (h, c) in history[..n].iter_mut().zip(&mut current[..n]) {
        let v = blend_byte(*c, *h, s);
        *h = v;
        *c = v;
    }
}

// Motion-adaptive temporal blending.
//
// A plain exponential moving average leaves a trail behind fast motion: the
// history keeps the previous position of an object with most of its weight for
// several frames. The motion-adaptive variant estimates per-tile motion between
// the current resize and the history and moves all the way to the current frame
// where that difference is large, so fast content never ghosts, while static
// regions keep exactly the configured blend (the dither the temporal filter
// exists to damp is far below the low threshold).
//
// The estimate is a small fixed-tile grid owned by the upscaler, not a
// full-frame plane. Tiles are independent, so the weights and the blended bytes
// are identical for every worker count.

/// Destination tile size of the motion grid, in pixels.
///
/// 16 pixel tiles keep the grid tiny (about 32 KB at 4K) while still localising
/// a moving detail to a few tiles.
const MOTION_TILE: u32 = 16;

/// Number of luma samples per tile axis (a 16 x 16 tile becomes 4 x 4 samples).
const MOTION_SAMPLES: u32 = 4;

/// Tile luma difference below which the temporal blend keeps its configured
/// weight: one to three LSB of dither and sensor noise must not defeat the
/// smoothing the filter exists for.
const MOTION_LO: f32 = 8.0;

/// Tile luma difference at or above which the blend follows the current frame
/// completely, which removes the EMA trail behind fast motion.
const MOTION_HI: f32 = 40.0;

/// Reusable fixed-tile motion estimate for the motion-adaptive temporal blend.
///
/// The destination frame is split into MOTION_TILE x MOTION_TILE pixel tiles.
/// Each tile stores the twelve-bit weight of the current frame in the blend:
/// static tiles keep the configured temporal weight, while tiles whose mean luma
/// difference between the current resize and the history is large move all the
/// way to the current frame. Every tile depends only on its own pixels, so the
/// weights are identical for any worker count.
#[derive(Clone, Debug, Default)]
struct MotionGrid {
    /// Tiles per row.
    cols: u32,
    /// Tile rows.
    rows: u32,
    /// Twelve-bit current-frame weight per tile, row major.
    weights: Vec<u16>,
}

impl MotionGrid {
    /// Resize the grid for a destination frame and report the tile dimensions.
    fn prepare(&mut self, dst: Dims) -> (u32, u32) {
        let cols = dst.width.div_ceil(MOTION_TILE).max(1);
        let rows = dst.height.div_ceil(MOTION_TILE).max(1);
        if self.cols != cols || self.rows != rows {
            self.cols = cols;
            self.rows = rows;
            self.weights.resize((cols as usize) * (rows as usize), 0);
        }
        (cols, rows)
    }
}

/// Smooth 0..=1 ramp from MOTION_LO to MOTION_HI.
#[inline]
fn motion_factor(diff: f32) -> f32 {
    let t = ((diff - MOTION_LO) / (MOTION_HI - MOTION_LO)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Convert a 0..=1 blend weight to the twelve-bit fixed point used by the byte
/// blend, with the same rounding as the constant-weight path.
#[inline]
fn blend_ticks(blend: f32) -> i32 {
    let b = if blend.is_finite() {
        blend.clamp(0.0, 1.0)
    } else {
        1.0
    };
    (b * 4096.0 + 0.5) as i32
}

/// Fill the motion grid from the current frame and the previous history frame.
///
/// Each tile averages the absolute BT.709 luma difference over a 4 x 4 sample
/// grid and turns it into a current-frame weight: the configured weight on a
/// static tile, 4096 when the tile changed enough that history would ghost.
#[allow(clippy::too_many_arguments)]
fn motion_weights(
    current: &Frame,
    history: &Frame,
    base_s: i32,
    threads: usize,
    grid: &mut MotionGrid,
) {
    debug_assert!(current.dims == history.dims && current.fmt == history.fmt);
    let (cols, rows) = grid.prepare(current.dims);
    let layout = pixel::Layout::of(current.fmt);
    let stride = current.stride();
    let width = current.dims.width;
    let height = current.dims.height;
    let cdata: &[u8] = &current.data;
    let hdata: &[u8] = &history.data;
    let chunks = resample::row_chunks(rows as usize, threads);
    if chunks.len() <= 1 {
        for ty in 0..rows {
            let row = &mut grid.weights[(ty as usize) * (cols as usize)..][..cols as usize];
            motion_weights_row(
                row, ty, cdata, hdata, stride, width, height, &layout, base_s,
            );
        }
        return;
    }
    let mut rest: &mut [u16] = &mut grid.weights;
    std::thread::scope(|scope| {
        for &(y0, y1) in &chunks {
            let len = (y1 - y0) * cols as usize;
            let (head, tail) = rest.split_at_mut(len);
            rest = tail;
            scope.spawn(move || {
                for (i, ty) in (y0..y1).enumerate() {
                    let row = &mut head[i * cols as usize..(i + 1) * cols as usize];
                    motion_weights_row(
                        row, ty as u32, cdata, hdata, stride, width, height, &layout, base_s,
                    );
                }
            });
        }
    });
}

/// One tile row of the motion grid.
#[allow(clippy::too_many_arguments)]
fn motion_weights_row(
    out: &mut [u16],
    ty: u32,
    current: &[u8],
    history: &[u8],
    stride: usize,
    width: u32,
    height: u32,
    layout: &pixel::Layout,
    base_s: i32,
) {
    let bpp = layout.bpp;
    let step = MOTION_TILE / MOTION_SAMPLES;
    let tile_y = ty * MOTION_TILE;
    let room = (4096 - base_s) as f32;
    let scale = (MOTION_SAMPLES * MOTION_SAMPLES) as f32;
    for (tx, slot) in out.iter_mut().enumerate() {
        let tile_x = tx as u32 * MOTION_TILE;
        let mut sum = 0u32;
        for sy in 0..MOTION_SAMPLES {
            let y = (tile_y + sy * step + step / 2).min(height - 1);
            let row = y as usize * stride;
            for sx in 0..MOTION_SAMPLES {
                let x = (tile_x + sx * step + step / 2).min(width - 1);
                let i = row + x as usize * bpp;
                let c = pixel::luma8(&current[i..i + bpp], layout) as i32;
                let h = pixel::luma8(&history[i..i + bpp], layout) as i32;
                sum += (c - h).unsigned_abs();
            }
        }
        let mean = sum as f32 / scale;
        let ticks = base_s as f32 + room * motion_factor(mean);
        *slot = (ticks + 0.5).clamp(0.0, 4096.0) as u16;
    }
}

/// Motion-adaptive temporal blend: blend with the history, but move to the
/// current frame wherever the two differ enough that history would ghost.
///
/// Static tiles keep the configured blend exactly, so the flicker metric is
/// unchanged. The history snapshot is fused into the same pass as before.
fn temporal_blend_motion(
    history: &mut Frame,
    current: &mut Frame,
    blend: f32,
    grid: &mut MotionGrid,
) {
    debug_assert!(history.dims == current.dims && history.fmt == current.fmt);
    let threads = resample::threads_for(current.dims.area(), current.dims.height as usize);
    let base_s = blend_ticks(blend);
    if base_s <= 0 {
        temporal_blend_snapshot_threads(history, current, 0, threads);
        return;
    }
    motion_weights(current, history, base_s, threads, grid);
    temporal_blend_motion_threads(history, current, grid, threads);
}

/// Fused motion-adaptive blend/snapshot with an explicit worker count.
fn temporal_blend_motion_threads(
    history: &mut Frame,
    current: &mut Frame,
    grid: &MotionGrid,
    threads: usize,
) {
    let stride = current.stride();
    let rows = current.dims.height as usize;
    let width = current.dims.width;
    let bpp = current.fmt.bytes_per_pixel();
    let cols = grid.cols as usize;
    let weights = &grid.weights;
    let hdata: &mut [u8] = &mut history.data[..rows * stride];
    let cdata: &mut [u8] = &mut current.data[..rows * stride];
    let chunks = resample::row_chunks(rows, threads.max(1));
    if chunks.len() <= 1 {
        for y in 0..rows {
            blend_motion_row(
                &mut hdata[y * stride..][..stride],
                &mut cdata[y * stride..][..stride],
                y,
                weights,
                cols,
                width,
                bpp,
            );
        }
        return;
    }
    std::thread::scope(|scope| {
        let mut hrest: &mut [u8] = hdata;
        let mut crest: &mut [u8] = cdata;
        for &(y0, y1) in &chunks {
            let len = (y1 - y0) * stride;
            let (hhead, htail) = hrest.split_at_mut(len);
            hrest = htail;
            let (chead, ctail) = crest.split_at_mut(len);
            crest = ctail;
            scope.spawn(move || {
                for (i, y) in (y0..y1).enumerate() {
                    blend_motion_row(
                        &mut hhead[i * stride..][..stride],
                        &mut chead[i * stride..][..stride],
                        y,
                        weights,
                        cols,
                        width,
                        bpp,
                    );
                }
            });
        }
    });
}

/// Blend one destination row with the per-tile weights of the motion grid.
///
/// Tiles are whole pixels, so a run of consecutive bytes shares one weight and
/// the inner loop stays a flat integer blend.
#[inline]
fn blend_motion_row(
    history: &mut [u8],
    current: &mut [u8],
    y: usize,
    weights: &[u16],
    cols: usize,
    width: u32,
    bpp: usize,
) {
    let base = (y / MOTION_TILE as usize) * cols;
    for tx in 0..cols {
        let s = weights[base + tx] as i32;
        let x0 = tx * MOTION_TILE as usize;
        let x1 = (x0 + MOTION_TILE as usize).min(width as usize);
        if x0 >= x1 {
            continue;
        }
        let b0 = x0 * bpp;
        let b1 = x1 * bpp;
        for (h, c) in history[b0..b1].iter_mut().zip(&mut current[b0..b1]) {
            let v = blend_byte(*c, *h, s);
            *h = v;
            *c = v;
        }
    }
}

/// Snapshot the temporal history into an existing frame, with the row copy split
/// over workers so snapshotting does not serialise the frame time. The
/// destination keeps its allocation, so the temporal path does not allocate a
/// new full-frame buffer every frame.
fn copy_frame_into(src: &Frame, dst: &mut Frame, threads: usize) {
    debug_assert!(src.dims == dst.dims && src.fmt == dst.fmt);
    dst.pts_ns = src.pts_ns;
    dst.index = src.index;
    let stride = src.stride();
    let rows = src.dims.height as usize;
    let chunks = resample::row_chunks(rows, threads);
    if chunks.len() <= 1 {
        dst.data.copy_from_slice(&src.data);
        return;
    }
    let s: &[u8] = &src.data;
    let data: &mut [u8] = &mut dst.data[..rows * stride];
    std::thread::scope(|scope| {
        let mut rest: &mut [u8] = data;
        for &(y0, y1) in &chunks {
            let len = (y1 - y0) * stride;
            let (head, tail) = rest.split_at_mut(len);
            rest = tail;
            scope.spawn(move || {
                head.copy_from_slice(&s[y0 * stride..y1 * stride]);
            });
        }
    });
}

/// False if the filter cannot be used for the requested resolution: Off only
/// works as a pass-through, every other filter needs a non-empty destination.
pub fn filter_supported(filter: UpscaleFilter, src: Dims, dst: Dims) -> bool {
    match filter {
        UpscaleFilter::Off => src == dst,
        _ => !dst.is_empty(),
    }
}

/// Rough relative cost, useful for telemetry and auto-tuning.
pub fn filter_cost_hint(filter: UpscaleFilter) -> u32 {
    match filter {
        UpscaleFilter::Off => 0,
        UpscaleFilter::Nearest => 1,
        UpscaleFilter::Bilinear => 2,
        UpscaleFilter::Bicubic => 4,
        UpscaleFilter::Lanczos => 6,
        UpscaleFilter::Adaptive => 5,
        UpscaleFilter::Temporal => 7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluid_core::PixelFormat;
    use std::time::Instant;

    const ALL_FILTERS: [UpscaleFilter; 6] = [
        UpscaleFilter::Bilinear,
        UpscaleFilter::Nearest,
        UpscaleFilter::Bicubic,
        UpscaleFilter::Lanczos,
        UpscaleFilter::Adaptive,
        UpscaleFilter::Temporal,
    ];
    const ALL_FORMATS: [PixelFormat; 4] = [
        PixelFormat::Rgba8,
        PixelFormat::Bgra8,
        PixelFormat::Rgb8,
        PixelFormat::Luma8,
    ];

    fn solid(d: Dims, fmt: PixelFormat, r: u8, g: u8, b: u8) -> Frame {
        let mut data = Vec::with_capacity(fmt.buffer_len(d));
        for _ in 0..d.area() {
            match fmt {
                PixelFormat::Rgba8 => data.extend_from_slice(&[r, g, b, 255]),
                PixelFormat::Bgra8 => data.extend_from_slice(&[b, g, r, 255]),
                PixelFormat::Rgb8 => data.extend_from_slice(&[r, g, b]),
                PixelFormat::Luma8 => data.push(r),
            }
        }
        Frame::new(d, fmt, data).unwrap()
    }

    /// Horizontal linear gradient in every colour channel, alpha 255.
    fn gradient(d: Dims, fmt: PixelFormat) -> Frame {
        let mut data = Vec::with_capacity(fmt.buffer_len(d));
        let denom = d.width.saturating_sub(1).max(1);
        for _ in 0..d.height {
            for x in 0..d.width {
                let r = ((x * 255) / denom) as u8;
                let g = ((x * 200) / denom) as u8;
                let b = ((x * 120) / denom) as u8;
                match fmt {
                    PixelFormat::Rgba8 => data.extend_from_slice(&[r, g, b, 255]),
                    PixelFormat::Bgra8 => data.extend_from_slice(&[b, g, r, 255]),
                    PixelFormat::Rgb8 => data.extend_from_slice(&[r, g, b]),
                    PixelFormat::Luma8 => data.push(r),
                }
            }
        }
        Frame::new(d, fmt, data).unwrap()
    }

    /// Checkerboard made of blocks of the given edge length.
    fn checker_blocks(w: u32, h: u32, block: u32) -> Frame {
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let v = if ((x / block) + (y / block)).is_multiple_of(2) {
                    0u8
                } else {
                    255
                };
                data.extend_from_slice(&[v, v, v, 255]);
            }
        }
        Frame::new(Dims::new(w, h), PixelFormat::Rgba8, data).unwrap()
    }

    /// Deterministic pseudo-random frame; every byte varies so band-boundary
    /// bugs change the output.
    fn noise_frame(d: Dims, fmt: PixelFormat, seed: u32) -> Frame {
        let mut data = vec![0u8; fmt.buffer_len(d)];
        let mut s = seed | 1;
        for (i, b) in data.iter_mut().enumerate() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *b = ((s >> 9) as u8).wrapping_add(i as u8).wrapping_mul(29);
        }
        Frame::new(d, fmt, data).unwrap()
    }

    fn luma_of(f: &Frame, x: u32, y: u32) -> u8 {
        let px = f.pixel(x, y);
        match f.fmt {
            PixelFormat::Luma8 => px[0],
            _ => {
                let l = pixel::Layout::of(f.fmt);
                pixel::luma8(px, &l)
            }
        }
    }

    #[test]
    fn flat_colour_is_preserved_by_every_filter_and_format() {
        let targets = [
            Dims::new(10, 8),
            Dims::new(3, 2),
            Dims::new(13, 7),
            Dims::new(1, 1),
        ];
        for fmt in ALL_FORMATS {
            let src = solid(Dims::new(5, 4), fmt, 90, 150, 40);
            for filter in ALL_FILTERS {
                for dst in targets {
                    let out = resize(&src, dst, filter).unwrap();
                    assert_eq!(out.dims, dst, "{fmt:?} {filter:?}");
                    assert_eq!(out.fmt, fmt, "{fmt:?} {filter:?}");
                    assert_eq!(out.data.len(), fmt.buffer_len(dst));
                    for y in 0..dst.height {
                        for x in 0..dst.width {
                            let px = out.pixel(x, y);
                            match fmt {
                                PixelFormat::Rgba8 => {
                                    assert!(px[0].abs_diff(90) <= 1, "{fmt:?} {filter:?} r");
                                    assert!(px[1].abs_diff(150) <= 1, "{fmt:?} {filter:?} g");
                                    assert!(px[2].abs_diff(40) <= 1, "{fmt:?} {filter:?} b");
                                    assert_eq!(px[3], 255, "alpha must be preserved");
                                }
                                PixelFormat::Bgra8 => {
                                    assert!(px[2].abs_diff(90) <= 1);
                                    assert!(px[1].abs_diff(150) <= 1);
                                    assert!(px[0].abs_diff(40) <= 1);
                                    assert_eq!(px[3], 255);
                                }
                                PixelFormat::Rgb8 => {
                                    assert!(px[0].abs_diff(90) <= 1);
                                    assert!(px[1].abs_diff(150) <= 1);
                                    assert!(px[2].abs_diff(40) <= 1);
                                }
                                PixelFormat::Luma8 => assert!(px[0].abs_diff(90) <= 1),
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn matching_size_is_byte_exact_for_every_filter() {
        let src = gradient(Dims::new(7, 5), PixelFormat::Rgba8);
        for filter in ALL_FILTERS {
            let out = resize(&src, src.dims, filter).unwrap();
            assert_eq!(out.data, src.data, "{filter:?}");
        }
        let off = resize(&src, src.dims, UpscaleFilter::Off).unwrap();
        assert_eq!(off.data, src.data);
    }

    #[test]
    fn checkerboard_stays_sharp_with_lanczos_and_adaptive() {
        // 4px blocks upscaled 2x: the edges must stay near full contrast (a
        // 1px checker is at Nyquist and no 4-tap kernel can legally restore it).
        let src = checker_blocks(16, 16, 4);
        let dst = Dims::new(32, 32);
        for filter in [UpscaleFilter::Lanczos, UpscaleFilter::Adaptive] {
            let out = resize(&src, dst, filter).unwrap();
            let mut mn = 255u8;
            let mut mx = 0u8;
            for y in 2..30 {
                for x in 2..30 {
                    let v = luma_of(&out, x, y);
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
            }
            // Ringing normalisation costs a little range; anything close to
            // the 0..255 extremes still means the pattern survived intact.
            assert!(
                mn <= 45,
                "{filter:?}: dark squares lost contrast (min {mn})"
            );
            assert!(
                mx >= 210,
                "{filter:?}: bright squares lost contrast (max {mx})"
            );
        }
    }

    #[test]
    fn gradient_direction_is_preserved() {
        let src = gradient(Dims::new(16, 4), PixelFormat::Rgba8);
        let dst = Dims::new(40, 12);
        for filter in ALL_FILTERS {
            let out = resize(&src, dst, filter).unwrap();
            let y = dst.height / 2;
            let mut prev = luma_of(&out, 1, y);
            for x in 2..dst.width - 1 {
                let v = luma_of(&out, x, y);
                assert!(
                    v as i32 >= prev as i32 - 1,
                    "{filter:?}: gradient reversed at x={x}"
                );
                prev = v;
            }
            assert!(
                luma_of(&out, dst.width - 2, y) > luma_of(&out, 1, y),
                "{filter:?}: brightness must increase left to right"
            );
        }
    }

    #[test]
    fn bilinear_up_then_down_approximates_the_source() {
        let src = gradient(Dims::new(32, 16), PixelFormat::Rgba8);
        let up = resize(&src, Dims::new(64, 32), UpscaleFilter::Bilinear).unwrap();
        let down = resize(&up, Dims::new(32, 16), UpscaleFilter::Bilinear).unwrap();
        for y in 1..15 {
            for x in 1..31 {
                let a = down.pixel(x, y);
                let b = src.pixel(x, y);
                for c in 0..3 {
                    assert!(
                        a[c].abs_diff(b[c]) <= 3,
                        "round trip differs at ({x},{y}) ch{c}: {} vs {}",
                        a[c],
                        b[c]
                    );
                }
                assert_eq!(a[3], 255);
            }
        }
    }

    #[test]
    fn odd_sizes_and_tiny_frames_are_handled() {
        let one = solid(Dims::new(1, 1), PixelFormat::Rgba8, 10, 20, 30);
        for filter in ALL_FILTERS {
            let out = resize(&one, Dims::new(4, 4), filter).unwrap();
            assert_eq!(out.dims, Dims::new(4, 4));
            let px = out.pixel(3, 3);
            assert!(px[0].abs_diff(10) <= 1 && px[1].abs_diff(20) <= 1 && px[2].abs_diff(30) <= 1);
        }
        let thin = gradient(Dims::new(1, 9), PixelFormat::Luma8);
        for filter in ALL_FILTERS {
            let out = resize(&thin, Dims::new(5, 3), filter).unwrap();
            assert_eq!(out.dims, Dims::new(5, 3));
        }
    }

    #[test]
    fn empty_destination_is_an_error() {
        let src = solid(Dims::new(4, 4), PixelFormat::Rgba8, 1, 2, 3);
        for filter in ALL_FILTERS {
            assert!(resize(&src, Dims::new(0, 0), filter).is_err());
            assert!(resize(&src, Dims::new(8, 0), filter).is_err());
        }
    }

    #[test]
    fn temporal_blend_endpoints_and_midpoint() {
        let a = solid(Dims::new(4, 4), PixelFormat::Rgba8, 0, 0, 0);
        let b = solid(Dims::new(4, 4), PixelFormat::Rgba8, 100, 40, 200);
        assert_eq!(temporal_blend(&a, &b, 0.0).data, a.data);
        assert_eq!(temporal_blend(&a, &b, 1.0).data, b.data);
        let mid = temporal_blend(&a, &b, 0.5);
        assert!(mid.pixel(0, 0)[0].abs_diff(50) <= 1);
        assert!(mid.pixel(0, 0)[1].abs_diff(20) <= 1);
        assert!(mid.pixel(0, 0)[2].abs_diff(100) <= 1);
        // Out of range / non-finite blends are clamped.
        assert_eq!(temporal_blend(&a, &b, -2.0).data, a.data);
        assert_eq!(temporal_blend(&a, &b, 7.0).data, b.data);
        assert_eq!(temporal_blend(&a, &b, f32::NAN).data, b.data);
    }

    #[test]
    fn temporal_blend_mismatched_shapes_returns_current() {
        let a = solid(Dims::new(4, 4), PixelFormat::Rgba8, 0, 0, 0);
        let b = solid(Dims::new(2, 2), PixelFormat::Rgba8, 50, 50, 50);
        assert_eq!(temporal_blend(&a, &b, 0.5).data, b.data);
        let c = solid(Dims::new(4, 4), PixelFormat::Luma8, 9, 9, 9);
        assert_eq!(temporal_blend(&a, &c, 0.5).data, c.data);
    }

    #[test]
    fn temporal_blend_carries_current_metadata() {
        let a = solid(Dims::new(2, 2), PixelFormat::Rgba8, 0, 0, 0);
        let b = solid(Dims::new(2, 2), PixelFormat::Rgba8, 100, 100, 100)
            .with_pts(5)
            .with_index(3);
        let out = temporal_blend(&a, &b, 0.25);
        assert_eq!(out.pts_ns, 5);
        assert_eq!(out.index, 3);
    }

    #[test]
    fn temporal_blend_parallel_matches_serial_across_thread_counts() {
        // 61 rows over 4 workers gives uneven bands.
        for fmt in ALL_FORMATS {
            let history = noise_frame(Dims::new(53, 61), fmt, 11);
            let current = noise_frame(Dims::new(53, 61), fmt, 29);
            let mut serial = current.clone();
            temporal_blend_threads(&history, &mut serial, 0.3, 1);
            for threads in [2usize, 3, 4, 16] {
                let mut parallel = current.clone();
                temporal_blend_threads(&history, &mut parallel, 0.3, threads);
                assert_eq!(
                    serial.data, parallel.data,
                    "{fmt:?} threads={threads} parallel != serial"
                );
            }
            // The endpoints are exact for every thread count.
            let mut snap = current.clone();
            temporal_blend_threads(&history, &mut snap, 0.0, 4);
            assert_eq!(snap.data, history.data, "{fmt:?} blend 0");
            let mut snap = current.clone();
            temporal_blend_threads(&history, &mut snap, 1.0, 4);
            assert_eq!(snap.data, current.data, "{fmt:?} blend 1");
        }
    }

    #[test]
    fn fused_blend_snapshot_matches_reference_and_serial() {
        let s = |b: f32| (b.clamp(0.0, 1.0) * 4096.0 + 0.5) as i32;
        for fmt in ALL_FORMATS {
            let old_history = noise_frame(Dims::new(53, 61), fmt, 5);
            let current = noise_frame(Dims::new(53, 61), fmt, 13);
            let expected = temporal_blend(&old_history, &current, 0.37);
            let mut serial_h = old_history.clone();
            let mut serial_c = current.clone();
            temporal_blend_snapshot_threads(&mut serial_h, &mut serial_c, s(0.37), 1);
            assert_eq!(serial_c.data, expected.data, "{fmt:?} current");
            assert_eq!(serial_h.data, expected.data, "{fmt:?} history");
            for threads in [2usize, 3, 4, 16] {
                let mut h = old_history.clone();
                let mut c = current.clone();
                temporal_blend_snapshot_threads(&mut h, &mut c, s(0.37), threads);
                assert_eq!(c.data, expected.data, "{fmt:?} threads={threads} current");
                assert_eq!(h.data, expected.data, "{fmt:?} threads={threads} history");
            }
            // Endpoints: 1.0 makes both buffers the current frame, 0.0 both the
            // history frame.
            let mut h = old_history.clone();
            let mut c = current.clone();
            temporal_blend_snapshot(&mut h, &mut c, 1.0);
            assert_eq!(c.data, current.data, "{fmt:?} blend 1 current");
            assert_eq!(h.data, current.data, "{fmt:?} blend 1 history");
            let mut h = old_history.clone();
            let mut c = current.clone();
            temporal_blend_snapshot(&mut h, &mut c, 0.0);
            assert_eq!(c.data, old_history.data, "{fmt:?} blend 0 current");
            assert_eq!(h.data, old_history.data, "{fmt:?} blend 0 history");
        }
    }

    #[test]
    fn parallel_frame_copy_matches_serial() {
        for fmt in ALL_FORMATS {
            let src = noise_frame(Dims::new(37, 91), fmt, 7)
                .with_pts(42)
                .with_index(3);
            let mut serial = Frame::zeroed(src.dims, fmt);
            copy_frame_into(&src, &mut serial, 1);
            assert_eq!(serial.data, src.data);
            assert_eq!(serial.pts_ns, 42);
            assert_eq!(serial.index, 3);
            for threads in [2usize, 4, 16] {
                let mut parallel = Frame::zeroed(src.dims, fmt);
                copy_frame_into(&src, &mut parallel, threads);
                assert_eq!(parallel.data, src.data, "{fmt:?} threads={threads}");
                assert_eq!(parallel.dims, src.dims);
                assert_eq!(parallel.fmt, fmt);
            }
        }
    }

    #[test]
    fn sharpen_zero_is_a_byte_exact_noop() {
        let mut f = gradient(Dims::new(13, 9), PixelFormat::Rgba8);
        let before = f.data.clone();
        sharpen_in_place(&mut f, 0.0);
        assert_eq!(f.data, before);
        sharpen_in_place(&mut f, f32::NAN);
        assert_eq!(f.data, before);
        sharpen_in_place(&mut f, -1.0);
        assert_eq!(f.data, before);
    }

    #[test]
    fn sharpen_never_changes_shape_or_format() {
        for fmt in ALL_FORMATS {
            let mut f = gradient(Dims::new(9, 7), fmt);
            sharpen_in_place(&mut f, 0.8);
            assert_eq!(f.dims, Dims::new(9, 7));
            assert_eq!(f.fmt, fmt);
            assert_eq!(f.data.len(), fmt.buffer_len(Dims::new(9, 7)));
        }
    }

    #[test]
    fn sharpen_raises_local_contrast_within_the_input_range() {
        // Smooth edge (blurred step): sharpening should steepen it while never
        // leaving the original value range, so no halos appear.
        let profile: [u8; 8] = [100, 100, 110, 130, 150, 160, 160, 160];
        let mut data = Vec::new();
        for _ in 0..8u32 {
            for v in profile {
                data.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let mut f = Frame::new(Dims::new(8, 8), PixelFormat::Rgba8, data).unwrap();
        let before = f.data.clone();
        sharpen_in_place(&mut f, 1.0);
        let mn = *before.iter().min().unwrap();
        let mx = *before.iter().max().unwrap();
        assert!(f.data.iter().all(|v| *v >= mn && *v <= mx), "stay in range");
        let var = |d: &[u8]| -> f64 {
            let m = d.iter().map(|v| *v as f64).sum::<f64>() / d.len() as f64;
            d.iter().map(|v| (*v as f64 - m).powi(2)).sum::<f64>() / d.len() as f64
        };
        assert!(var(&f.data) > var(&before), "soft edge should get sharper");
        // alpha untouched
        for i in 0..f.data.len() / 4 {
            assert_eq!(f.data[i * 4 + 3], 255);
        }
    }

    #[test]
    fn sharpen_is_stronger_at_higher_amount() {
        let profile: [u8; 8] = [100, 100, 110, 130, 150, 160, 160, 160];
        let mut data = Vec::new();
        for _ in 0..4u32 {
            data.extend_from_slice(&profile);
        }
        let base = Frame::new(Dims::new(8, 4), PixelFormat::Luma8, data).unwrap();
        let mut weak = base.clone();
        let mut strong = base.clone();
        sharpen_in_place(&mut weak, 0.25);
        sharpen_in_place(&mut strong, 1.0);
        let energy = |f: &Frame| -> u64 {
            f.data
                .windows(2)
                .map(|w| (w[1] as i64 - w[0] as i64).unsigned_abs())
                .sum()
        };
        assert!(energy(&strong) >= energy(&weak));
        assert!(energy(&weak) >= energy(&base));
    }

    #[test]
    fn filter_support_and_cost_hints_are_sane() {
        let a = Dims::new(2, 2);
        let b = Dims::new(4, 4);
        assert!(filter_supported(UpscaleFilter::Off, a, a));
        assert!(!filter_supported(UpscaleFilter::Off, a, b));
        assert!(filter_supported(UpscaleFilter::Bilinear, a, b));
        assert!(!filter_supported(
            UpscaleFilter::Bilinear,
            a,
            Dims::new(0, 4)
        ));
        assert!(filter_supported(UpscaleFilter::Lanczos, a, Dims::new(1, 1)));
        assert_eq!(filter_cost_hint(UpscaleFilter::Off), 0);
        assert_eq!(filter_cost_hint(UpscaleFilter::Nearest), 1);
        assert_eq!(filter_cost_hint(UpscaleFilter::Bilinear), 2);
        assert_eq!(filter_cost_hint(UpscaleFilter::Bicubic), 4);
        assert_eq!(filter_cost_hint(UpscaleFilter::Lanczos), 6);
        assert_eq!(filter_cost_hint(UpscaleFilter::Adaptive), 5);
        assert_eq!(filter_cost_hint(UpscaleFilter::Temporal), 7);
    }

    fn cfg(filter: UpscaleFilter, sharpness: f32) -> UpscalerConfig {
        UpscalerConfig {
            filter,
            sharpness,
            ..UpscalerConfig::default()
        }
    }

    #[test]
    fn upscaler_dispatches_and_records_cost() {
        let src = gradient(Dims::new(64, 48), PixelFormat::Rgba8);
        let dst = Dims::new(128, 96);
        for filter in ALL_FILTERS {
            let mut up = Upscaler::new(&cfg(filter, 0.0), dst);
            assert_eq!(up.dst_dims(), dst);
            assert_eq!(up.filter(), filter);
            assert_eq!(up.last_cost_ns(), 0);
            let out = up.upscale(&src).unwrap();
            assert_eq!(out.dims, dst);
            assert_eq!(out.pts_ns, src.pts_ns);
            assert_eq!(out.index, src.index);
            assert!(up.last_cost_ns() > 0, "{filter:?}: cost should be recorded");
            let reference = resize(&src, dst, filter).unwrap();
            assert_eq!(out.data, reference.data, "{filter:?}: dispatch mismatch");
        }
    }

    #[test]
    fn upscaler_off_is_a_pass_through() {
        let src = gradient(Dims::new(8, 8), PixelFormat::Bgra8);
        let mut up = Upscaler::new(&cfg(UpscaleFilter::Off, 0.9), src.dims);
        let out = up.upscale(&src).unwrap();
        assert_eq!(
            out.data, src.data,
            "Off must not touch the frame, not even sharpen"
        );
    }

    #[test]
    fn upscaler_off_falls_back_to_bilinear() {
        let src = gradient(Dims::new(8, 8), PixelFormat::Rgba8);
        let dst = Dims::new(20, 12);
        let mut up = Upscaler::new(&cfg(UpscaleFilter::Off, 0.0), dst);
        let out = up.upscale(&src).unwrap();
        let reference = resize(&src, dst, UpscaleFilter::Bilinear).unwrap();
        assert_eq!(out.data, reference.data);
    }

    #[test]
    fn upscaler_applies_configured_sharpness() {
        let profile: [u8; 8] = [100, 100, 110, 130, 150, 160, 160, 160];
        let mut data = Vec::new();
        for _ in 0..8u32 {
            for v in profile {
                data.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let src = Frame::new(Dims::new(8, 8), PixelFormat::Rgba8, data).unwrap();
        let dst = Dims::new(16, 16);
        let plain = Upscaler::new(&cfg(UpscaleFilter::Adaptive, 0.0), dst)
            .upscale(&src)
            .unwrap();
        let sharp = Upscaler::new(&cfg(UpscaleFilter::Adaptive, 0.6), dst)
            .upscale(&src)
            .unwrap();
        let reference = resize(&src, dst, UpscaleFilter::Adaptive).unwrap();
        assert_eq!(
            plain.data, reference.data,
            "sharpness 0 must equal plain resize"
        );
        assert_ne!(sharp.data, plain.data, "sharpness must change the output");
        assert_eq!(sharp.dims, dst);
        assert_eq!(sharp.fmt, src.fmt);
    }

    #[test]
    fn upscaler_temporal_snaps_to_current_on_full_field_change() {
        let mut c = cfg(UpscaleFilter::Temporal, 0.0);
        c.temporal = TemporalConfig {
            enabled: true,
            blend: 0.25,
            history_frames: 2,
        };
        let mut up = Upscaler::new(&c, Dims::new(8, 8));
        let first = solid(Dims::new(4, 4), PixelFormat::Rgba8, 90, 90, 90);
        let second = solid(Dims::new(4, 4), PixelFormat::Rgba8, 190, 190, 190);
        let o1 = up.upscale(&first).unwrap();
        assert!(
            o1.pixel(0, 0)[0].abs_diff(90) <= 1,
            "first frame has no history"
        );
        // The whole frame changed by far more than the motion high threshold, so
        // every motion tile follows the current frame instead of keeping a ghost.
        let o2 = up.upscale(&second).unwrap();
        assert!(
            o2.pixel(0, 0)[0].abs_diff(190) <= 1,
            "motion-adaptive blend must snap to the current frame, got {}",
            o2.pixel(0, 0)[0]
        );
        up.reset();
        let o3 = up.upscale(&second).unwrap();
        assert!(
            o3.pixel(0, 0)[0].abs_diff(190) <= 1,
            "reset must drop history"
        );
    }

    #[test]
    fn upscaler_temporal_keeps_ema_below_the_motion_threshold() {
        let mut c = cfg(UpscaleFilter::Temporal, 0.0);
        c.temporal = TemporalConfig {
            enabled: true,
            blend: 0.25,
            history_frames: 2,
        };
        // alpha = 0.25 * 2 / 2 = 0.25, in twelve-bit fixed point 1024. A 6 LSB
        // change is below MOTION_LO, so the tile keeps that weight and the blend
        // is the plain EMA: 90 + round(0.25 * 6) = 91.
        let mut up = Upscaler::new(&c, Dims::new(8, 8));
        let first = solid(Dims::new(4, 4), PixelFormat::Rgba8, 90, 90, 90);
        let second = solid(Dims::new(4, 4), PixelFormat::Rgba8, 96, 96, 96);
        let _ = up.upscale(&first).unwrap();
        let o2 = up.upscale(&second).unwrap();
        assert!(
            o2.pixel(0, 0)[0].abs_diff(91) <= 1,
            "static content must keep the configured EMA, got {}",
            o2.pixel(0, 0)[0]
        );
    }

    #[test]
    fn motion_blend_parallel_matches_serial_across_thread_counts() {
        for fmt in ALL_FORMATS {
            let history = noise_frame(Dims::new(97, 61), fmt, 3);
            let current = noise_frame(Dims::new(97, 61), fmt, 7);
            // The grid is data-only, so workers cannot change it.
            let mut grid_serial = MotionGrid::default();
            motion_weights(&current, &history, 492, 1, &mut grid_serial);
            let mut grid_parallel = MotionGrid::default();
            motion_weights(&current, &history, 492, 8, &mut grid_parallel);
            assert_eq!(
                grid_serial.weights, grid_parallel.weights,
                "{fmt:?} motion grid depends on the worker count"
            );
            let mut serial_h = history.clone();
            let mut serial_c = current.clone();
            temporal_blend_motion_threads(&mut serial_h, &mut serial_c, &grid_serial, 1);
            for threads in [2usize, 3, 4, 16] {
                let mut h = history.clone();
                let mut c = current.clone();
                temporal_blend_motion_threads(&mut h, &mut c, &grid_serial, threads);
                assert_eq!(h.data, serial_h.data, "{fmt:?} threads={threads} history");
                assert_eq!(c.data, serial_c.data, "{fmt:?} threads={threads} current");
            }
        }
    }

    #[test]
    fn motion_grid_is_static_on_an_unchanged_frame() {
        // Identical frames: no motion, so every tile keeps the configured weight.
        let frame = Frame::zeroed(Dims::new(64, 48), PixelFormat::Rgba8);
        let mut grid = MotionGrid::default();
        motion_weights(&frame, &frame, 492, 1, &mut grid);
        assert!(grid.weights.iter().all(|w| *w == 492));
        // A white block in tile (0,0) and nothing else: that tile must reach the
        // full current weight while a far tile keeps the static weight.
        let mut other = frame.clone();
        for y in 0..16 {
            for x in 0..16 {
                let i = (y * 64 + x) * 4;
                other.data[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        motion_weights(&other, &frame, 492, 1, &mut grid);
        let cols = grid.cols as usize;
        assert_eq!(grid.weights[0], 4096);
        assert_eq!(grid.weights[2 * cols + 3], 492);
    }

    #[test]
    fn upscaler_temporal_disabled_matches_adaptive() {
        let src = gradient(Dims::new(8, 8), PixelFormat::Rgba8);
        let dst = Dims::new(16, 16);
        let mut c = cfg(UpscaleFilter::Temporal, 0.0);
        c.temporal = TemporalConfig {
            enabled: false,
            blend: 0.5,
            history_frames: 3,
        };
        let mut up = Upscaler::new(&c, dst);
        let out = up.upscale(&src).unwrap();
        let reference = resize(&src, dst, UpscaleFilter::Adaptive).unwrap();
        assert_eq!(out.data, reference.data);
    }

    #[test]
    fn upscaler_rejects_empty_destination() {
        let src = solid(Dims::new(4, 4), PixelFormat::Rgba8, 1, 1, 1);
        let mut up = Upscaler::new(&cfg(UpscaleFilter::Bilinear, 0.0), Dims::new(0, 4));
        assert!(up.upscale(&src).is_err());
    }

    #[test]
    fn resize_sharp_composes_both_stages() {
        let src = gradient(Dims::new(8, 8), PixelFormat::Rgba8);
        let dst = Dims::new(20, 20);
        let plain = resize(&src, dst, UpscaleFilter::Bicubic).unwrap();
        let sharp = resize_sharp(&src, dst, UpscaleFilter::Bicubic, 0.7).unwrap();
        assert_eq!(sharp.dims, dst);
        assert_ne!(sharp.data, plain.data);
        let zero = resize_sharp(&src, dst, UpscaleFilter::Bicubic, 0.0).unwrap();
        assert_eq!(zero.data, plain.data, "sharpness 0 must be a no-op");
        let nan = resize_sharp(&src, dst, UpscaleFilter::Bicubic, f32::NAN).unwrap();
        assert_eq!(nan.data, plain.data);
    }

    /// Manual performance probe: cargo test --release -p fluid-upscale -- --ignored --nocapture
    ///
    /// Reports the serial path (one worker) next to the default parallel path
    /// so the speedup is visible per filter.
    #[test]
    #[ignore]
    fn perf_1080p_to_4k() {
        let src = gradient(Dims::new(1920, 1080), PixelFormat::Rgba8);
        let dst = Dims::new(3840, 2160);
        let threads = crate::resample::default_threads(src.dims, dst);
        println!("threads: {threads} for {} -> {}", src.dims, dst);
        let mut best = [f64::MAX; 5];
        let mut best_serial = [f64::MAX; 5];
        for (fi, filter) in [
            UpscaleFilter::Nearest,
            UpscaleFilter::Bilinear,
            UpscaleFilter::Bicubic,
            UpscaleFilter::Lanczos,
            UpscaleFilter::Adaptive,
        ]
        .into_iter()
        .enumerate()
        {
            for _ in 0..3 {
                let warm = resize(&src, dst, filter).unwrap();
                std::hint::black_box(&warm);
                let t = Instant::now();
                let out = resize(&src, dst, filter).unwrap();
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                std::hint::black_box(&out);
                best[fi] = best[fi].min(ms);

                let t = Instant::now();
                let serial = crate::resample::resample_threads(&src, dst, filter, 1).unwrap();
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                best_serial[fi] = best_serial[fi].min(ms);
                // The full-size output must stay bit-identical to serial.
                assert_eq!(out.data, serial.data, "{filter:?} parallel != serial");
                std::hint::black_box(&serial);
            }
            println!(
                "{filter:?}: {:.2} ms parallel ({} threads) vs {:.2} ms serial, {:.2}x",
                best[fi],
                threads,
                best_serial[fi],
                best_serial[fi] / best[fi]
            );
        }
    }

    /// End-to-end probe for the stateful upscaler.
    ///
    /// Run with: cargo test --release -p fluid-upscale -- --ignored --nocapture perf_end_to_end
    #[test]
    #[ignore]
    fn perf_end_to_end() {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let env = std::env::var("FLUID_THREADS").unwrap_or_else(|_| "<unset>".to_string());
        println!("available_parallelism={threads} FLUID_THREADS={env}");
        let ms = |t: Instant| t.elapsed().as_secs_f64() * 1000.0;
        let src = gradient(Dims::new(1920, 1080), PixelFormat::Rgba8);
        let dst = Dims::new(3840, 2160);
        let mut sharp4k = gradient(dst, PixelFormat::Rgba8);
        sharp4k.pts_ns = 0;

        // Individual stages on a 4K frame.
        let mut best = f64::MAX;
        for _ in 0..3 {
            let mut f = sharp4k.clone();
            let t = Instant::now();
            sharpen_in_place(&mut f, 0.3);
            best = best.min(ms(t));
            std::hint::black_box(&f);
        }
        println!("stage sharpen 4K amount 0.3: {best:.2} ms");
        let hist = gradient(dst, PixelFormat::Bgra8);
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            let f = temporal_blend(&hist, &sharp4k, 0.25);
            best = best.min(ms(t));
            std::hint::black_box(&f);
        }
        println!("stage temporal_blend 4K: {best:.2} ms");

        // Worker scaling of the two passes that were serial before.
        for threads in [1usize, 2, 4, 8, 16] {
            let mut f = sharp4k.clone();
            let mut sbest = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                crate::sharpen::sharpen_frame_threads(&mut f, 0.3, threads);
                sbest = sbest.min(ms(t));
            }
            std::hint::black_box(&f);
            let mut g = sharp4k.clone();
            let mut bbest = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                temporal_blend_threads(&hist, &mut g, 0.25, threads);
                bbest = bbest.min(ms(t));
            }
            std::hint::black_box(&g);
            println!("scaling threads={threads}: sharpen {sbest:.2} ms | blend {bbest:.2} ms");
        }

        println!("== 1080p -> 4K end-to-end ==");
        for filter in ALL_FILTERS {
            let mut best_resize = f64::MAX;
            let mut best_plain = f64::MAX;
            let mut best_sharp = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                let out = resize(&src, dst, filter).unwrap();
                best_resize = best_resize.min(ms(t));
                std::hint::black_box(&out);

                let mut c = cfg(filter, 0.0);
                c.temporal.enabled = false;
                let mut u = Upscaler::new(&c, dst);
                let _ = u.upscale(&src).unwrap();
                let t = Instant::now();
                let out = u.upscale(&src).unwrap();
                best_plain = best_plain.min(ms(t));
                std::hint::black_box(&out);

                let c = cfg(filter, 0.3);
                let mut u = Upscaler::new(&c, dst);
                let _ = u.upscale(&src).unwrap();
                let t = Instant::now();
                let out = u.upscale(&src).unwrap();
                best_sharp = best_sharp.min(ms(t));
                let cost = u.last_cost_ns() as f64 / 1.0e6;
                std::hint::black_box(&out);
                print!(" {filter:?} last={cost:.2} ms;");
            }
            println!(
                "\n{filter:?}: resize {best_resize:.2} ms | upscaler sharp=0 {best_plain:.2} ms | sharp=0.3 {best_sharp:.2} ms"
            );
        }

        println!("== 720p -> 1440p bilinear end-to-end ==");
        let small = gradient(Dims::new(1280, 720), PixelFormat::Rgba8);
        let small_dst = Dims::new(2560, 1440);
        let mut c = cfg(UpscaleFilter::Bilinear, 0.3);
        let mut u = Upscaler::new(&c, small_dst);
        let _ = u.upscale(&small).unwrap();
        let t = Instant::now();
        let out = u.upscale(&small).unwrap();
        println!("upscaler sharp=0.3: {:.2} ms", ms(t));
        std::hint::black_box(&out);
        c.sharpness = 0.0;
        let mut u = Upscaler::new(&c, small_dst);
        let _ = u.upscale(&small).unwrap();
        let t = Instant::now();
        let out = u.upscale(&small).unwrap();
        println!("upscaler sharp=0.0: {:.2} ms", ms(t));
        std::hint::black_box(&out);
    }

    /// Every filter, including Off and stateful Temporal history, must produce
    /// identical bytes through the trait object and the inherent method.
    #[test]
    fn trait_object_matches_inherent_byte_for_byte() {
        let src_dims = Dims::new(48, 32);
        let dst = Dims::new(83, 57);
        let frames = [
            gradient(src_dims, PixelFormat::Rgba8)
                .with_pts(1_000)
                .with_index(7),
            noise_frame(src_dims, PixelFormat::Rgba8, 5)
                .with_pts(2_000)
                .with_index(8),
            checker_blocks(48, 32, 3).with_pts(3_000).with_index(9),
        ];
        for filter in ALL_FILTERS.into_iter().chain([UpscaleFilter::Off]) {
            let mut c = cfg(filter, 0.3);
            c.temporal = TemporalConfig {
                enabled: true,
                blend: 0.2,
                history_frames: 3,
            };
            let mut inherent = Upscaler::new(&c, dst);
            let mut backend: Box<dyn UpscaleBackend> = Box::new(Upscaler::new(&c, dst));
            assert_eq!(backend.dst_dims(), dst);
            assert_eq!(backend.src_dims(), Dims::default());
            for (i, frame) in frames.iter().enumerate() {
                let expected = inherent.upscale(frame).unwrap();
                let mut out = Frame::zeroed(dst, frame.fmt);
                backend.upscale(frame, &mut out).unwrap();
                assert_eq!(
                    out.data, expected.data,
                    "{filter:?} frame {i}: trait bytes differ from inherent"
                );
                assert_eq!(out.dims, expected.dims);
                assert_eq!(out.fmt, expected.fmt);
                assert_eq!(out.pts_ns, frame.pts_ns, "{filter:?} frame {i}: pts");
                assert_eq!(out.index, frame.index, "{filter:?} frame {i}: index");
                assert_eq!(backend.src_dims(), frame.dims, "{filter:?} frame {i}: src");
                assert!(backend.last_cost_ns() > 0, "{filter:?} frame {i}: cost");
            }
        }
    }

    #[test]
    fn box_dyn_backend_works_end_to_end() {
        fn assert_send<T: Send>() {}
        assert_send::<Box<dyn UpscaleBackend>>();

        let src = gradient(Dims::new(32, 24), PixelFormat::Bgra8)
            .with_pts(77)
            .with_index(4);
        let dst = Dims::new(64, 48);
        let c = cfg(UpscaleFilter::Adaptive, 0.4);
        let mut backend: Box<dyn UpscaleBackend> = Box::new(Upscaler::new(&c, dst));
        assert_eq!(backend.backend_name(), "cpu");
        assert_eq!(backend.dst_dims(), dst);
        assert_eq!(backend.src_dims(), Dims::default(), "no frame seen yet");
        let mut out = Frame::zeroed(dst, src.fmt);
        backend.upscale(&src, &mut out).unwrap();
        let expected = Upscaler::new(&c, dst).upscale(&src).unwrap();
        assert_eq!(out.data, expected.data, "boxed backend differs");
        assert_eq!(out.pts_ns, 77);
        assert_eq!(out.index, 4);
        assert_eq!(backend.src_dims(), src.dims);
        assert!(backend.last_cost_ns() > 0);
    }

    #[test]
    fn trait_off_is_a_pass_through_without_sharpening() {
        let src = gradient(Dims::new(8, 8), PixelFormat::Luma8);
        let c = cfg(UpscaleFilter::Off, 0.9);
        let mut backend: Box<dyn UpscaleBackend> = Box::new(Upscaler::new(&c, src.dims));
        let mut out = Frame::zeroed(src.dims, src.fmt);
        backend.upscale(&src, &mut out).unwrap();
        assert_eq!(out.data, src.data, "Off must be an exact pass-through");
    }

    #[test]
    fn trait_reset_matches_inherent_reset() {
        let mut c = cfg(UpscaleFilter::Temporal, 0.2);
        c.temporal = TemporalConfig {
            enabled: true,
            blend: 0.3,
            history_frames: 2,
        };
        let dst = Dims::new(24, 18);
        let frames = [
            solid(Dims::new(12, 9), PixelFormat::Rgba8, 40, 60, 80),
            solid(Dims::new(12, 9), PixelFormat::Rgba8, 200, 10, 120),
            noise_frame(Dims::new(12, 9), PixelFormat::Rgba8, 3),
        ];
        let mut inherent = Upscaler::new(&c, dst);
        let mut backend: Box<dyn UpscaleBackend> = Box::new(Upscaler::new(&c, dst));
        for (i, frame) in frames.iter().enumerate() {
            let expected = inherent.upscale(frame).unwrap();
            let mut out = Frame::zeroed(dst, frame.fmt);
            backend.upscale(frame, &mut out).unwrap();
            assert_eq!(out.data, expected.data, "frame {i} before reset");
            if i == 1 {
                inherent.reset();
                backend.reset();
            }
            if i == 2 {
                let fresh = Upscaler::new(&c, dst).upscale(frame).unwrap();
                assert_eq!(out.data, fresh.data, "reset must drop history");
            }
        }
    }

    #[test]
    fn trait_rejects_mismatched_output_frames() {
        let src = gradient(Dims::new(16, 16), PixelFormat::Rgba8);
        let dst = Dims::new(32, 32);
        let mut backend: Box<dyn UpscaleBackend> =
            Box::new(Upscaler::new(&cfg(UpscaleFilter::Bilinear, 0.0), dst));
        let mut wrong_dims = Frame::zeroed(Dims::new(16, 16), src.fmt);
        assert!(backend.upscale(&src, &mut wrong_dims).is_err());
        let mut wrong_fmt = Frame::zeroed(dst, PixelFormat::Luma8);
        assert!(backend.upscale(&src, &mut wrong_fmt).is_err());
        let mut good = Frame::zeroed(dst, src.fmt);
        backend.upscale(&src, &mut good).unwrap();
        assert_eq!(good.data.len(), src.fmt.buffer_len(dst));
    }

    /// A minimal out-of-crate backend: only the required methods, which
    /// exercises the provided last_cost_ns default and object safety.
    struct DummyBackend {
        seen: Dims,
    }

    impl UpscaleBackend for DummyBackend {
        fn backend_name(&self) -> &'static str {
            "dummy"
        }
        fn src_dims(&self) -> Dims {
            self.seen
        }
        fn dst_dims(&self) -> Dims {
            Dims::new(4, 4)
        }
        fn upscale(&mut self, src: &Frame, out: &mut Frame) -> Result<()> {
            self.seen = src.dims;
            out.data.fill(7);
            out.pts_ns = src.pts_ns;
            out.index = src.index;
            Ok(())
        }
        fn reset(&mut self) {
            self.seen = Dims::default();
        }
    }

    #[test]
    fn minimal_backend_uses_provided_defaults() {
        let src = solid(Dims::new(2, 2), PixelFormat::Rgba8, 1, 2, 3)
            .with_pts(5)
            .with_index(6);
        let mut backend = DummyBackend {
            seen: Dims::default(),
        };
        assert_eq!(backend.last_cost_ns(), 0, "provided default");
        let mut out = Frame::zeroed(Dims::new(4, 4), src.fmt);
        backend.upscale(&src, &mut out).unwrap();
        assert!(out.data.iter().all(|b| *b == 7));
        assert_eq!(out.pts_ns, 5);
        assert_eq!(out.index, 6);
        assert_eq!(backend.src_dims(), Dims::new(2, 2));
        UpscaleBackend::reset(&mut backend);
        assert_eq!(backend.src_dims(), Dims::default());
        let boxed: Box<dyn UpscaleBackend> = Box::new(backend);
        assert_eq!(boxed.backend_name(), "dummy");
    }
}
