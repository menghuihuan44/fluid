//! Separable spatial resampling: the horizontal pass writes a u8 scratch
//! buffer, the vertical pass writes the destination frame, and the adaptive
//! filter fuses a gradient-directed detail-restore pass into the vertical one.
//!
//! Both passes are monomorphised over the channel count (`const BPP: usize`),
//! so the compiler can unroll the channel loop and keep the accumulators in
//! registers. The inner loop over taps reads precomputed indices and weights,
//! so there is no coordinate arithmetic per pixel and no allocation inside the
//! row loops: one scratch buffer and one f32x4 accumulator per worker are
//! allocated per resize call and reused for every row of that worker.
//!
//! # Parallelism
//!
//! Rows are independent in both passes, so each pass splits them into
//! contiguous chunks and hands one chunk to each `std::thread::scope` worker: the
//! horizontal pass splits source rows (each becomes one scratch row) and the
//! vertical pass splits destination rows. The threads borrow the buffers
//! directly and are joined before the call returns, so there is no `unsafe`,
//! no atomics, no global state and no allocation shared between workers.
//!
//! Parallelism never changes the arithmetic: every output byte runs the same
//! tap loop as the serial path, so both paths are bit-identical. The adaptive
//! vertical pass needs the *unrefined* luma of the rows above and below the
//! pixels it refines. A worker therefore recomputes its two boundary rows from
//! the scratch buffer instead of reading (and racing with) a neighbour output
//! row. Frames below `PAR_MIN_PIXELS` and very short frames stay serial so thread
//! startup cannot dominate.
//!
//! The adaptive (FSR1/EASU style) filter uses Catmull-Rom for the base upscale
//! and then restores detail with a 4-tap cross kernel whose response is steered
//! by the local luma gradient: along strong edges it sharpens in the
//! perpendicular direction only, so diagonal edges and checkerboards keep their
//! shape. This is much cheaper than a full EASU kernel and needs no extra pass
//! over the image, because the three luma rows needed by the kernel are already
//! in flight.
//!
//! # Anti-ringing
//!
//! Bicubic, Lanczos and Adaptive have negative lobes, so a hard edge can ring:
//! the reconstruction overshoots the local source range. The horizontal pass
//! therefore also records, per destination column and source row, the min and
//! max luma of the taps it used, and the vertical pass clamps the finished luma
//! sample to the min/max over the taps it combined. The clamp is a no-op on
//! smooth content (the bounds contain the sample) and removes the overshoot on
//! edges - the FSR/RCAS anti-ringing idea spread over the existing two passes
//! with no extra full-frame pass. It only moves luma (all colour channels by the
//! same delta, alpha untouched), so hue is preserved.

use std::thread;

use fluid_core::{Dims, FluidError, Frame, Result, UpscaleFilter};

use wide::f32x4;

use crate::coeff::{Kernel, Taps, build};
use crate::pixel::{Layout, luma_row, luma_row_f32, store_u8};
use crate::sharpen::{REFINE_STRENGTH_Q8, apply_delta_row};
use crate::simd::{dispatch, load4, load16, store4};

/// Below this destination pixel count the serial path runs: spawning worker
/// threads costs more than the resize itself on small frames.
const PAR_MIN_PIXELS: usize = 64 * 1024;

/// Never hand a worker fewer than this many rows. Keeps chunk bookkeeping and
/// the duplicate boundary rows negligible.
const MIN_ROWS_PER_THREAD: usize = 24;

/// Reject empty destination sizes early so callers get a clear error.
pub(crate) fn check_dst(dst: Dims) -> Result<()> {
    if dst.is_empty() {
        Err(FluidError::invalid(format!(
            "upscale: destination size {dst} is empty"
        )))
    } else {
        Ok(())
    }
}

/// Map the configuration filter onto a reconstruction kernel plus the adaptive
/// flag. Off is resolved by the caller (pass-through or bilinear fallback).
pub(crate) fn kernel_of(filter: UpscaleFilter) -> (Kernel, bool) {
    match filter {
        UpscaleFilter::Nearest => (Kernel::Nearest, false),
        UpscaleFilter::Bilinear => (Kernel::Bilinear, false),
        UpscaleFilter::Bicubic => (Kernel::Bicubic, false),
        UpscaleFilter::Lanczos => (Kernel::Lanczos3, false),
        UpscaleFilter::Adaptive | UpscaleFilter::Temporal => (Kernel::Bicubic, true),
        UpscaleFilter::Off => (Kernel::Bilinear, false),
    }
}

/// Worker count for one resize.
///
/// Capped by `available_parallelism`, by the row count (a chunk must hold at
/// least one row, and short frames get fewer workers) and by `FLUID_THREADS`
/// when that environment variable parses to a positive integer. Frames below
/// `PAR_MIN_PIXELS` return 1, which selects the serial path.
pub(crate) fn default_threads(src: Dims, dst: Dims) -> usize {
    threads_for(
        dst.area().max(src.area()),
        src.height.max(dst.height) as usize,
    )
}

/// Worker count for a pass that touches `work_pixels` pixels spread over `rows`
/// rows. Shared by resampling, sharpening and the temporal blend so every stage
/// uses the same worker count for a given frame.
pub(crate) fn threads_for(work_pixels: u64, rows: usize) -> usize {
    if work_pixels < PAR_MIN_PIXELS as u64 {
        return 1;
    }
    let avail = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    if avail <= 1 {
        return 1;
    }
    let requested = std::env::var("FLUID_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(avail);
    requested.min(avail).min(rows / MIN_ROWS_PER_THREAD).max(1)
}

/// Everything a pass needs, borrowed for the duration of the resize.
struct Plan<'a> {
    taps_x: &'a Taps,
    taps_y: &'a Taps,
    dw: usize,
    sw: usize,
    sh: usize,
    threads: usize,
    adaptive: bool,
    layout: &'a Layout,
}

/// Reusable buffers for one separable resample.
///
/// A stateful caller (a long-lived upscaler) owns one and passes it back on every
/// frame, so the horizontal-pass rows, the source luma plane and the anti-ringing
/// bounds are allocated and page-faulted once per stream instead of once per
/// frame.
#[derive(Clone, Debug, Default)]
pub(crate) struct ResampleScratch {
    hpass: Vec<u8>,
    luma: Vec<u8>,
    bounds_min: Vec<u8>,
    bounds_max: Vec<u8>,
}

impl ResampleScratch {
    /// Empty scratch; the buffers grow on the first frame and are reused after.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// Grow a reusable buffer to at least len bytes without touching its contents.
#[inline]
fn grow(buffer: &mut Vec<u8>, len: usize) {
    if buffer.len() < len {
        buffer.resize(len, 0);
    }
}

/// Source luma plane plus the per-column bounds the horizontal pass fills in.
struct RingPlanes<'a> {
    luma: &'a [u8],
    min: &'a mut [u8],
    max: &'a mut [u8],
}

/// Filled anti-ringing bounds: one byte per destination column, one row per
/// source row.
#[derive(Clone, Copy)]
struct RingBounds<'a> {
    min: &'a [u8],
    max: &'a [u8],
    dw: usize,
}

/// Stateless filtered resize. Format, pts and index are preserved.
pub(crate) fn resample(src: &Frame, dst: Dims, filter: UpscaleFilter) -> Result<Frame> {
    resample_threads(src, dst, filter, default_threads(src.dims, dst))
}

/// Resample with an explicit worker count (1 is the serial path).
///
/// Exposed inside the crate so tests and the ignored perf probe can compare the
/// parallel and serial results byte for byte.
pub(crate) fn resample_threads(
    src: &Frame,
    dst: Dims,
    filter: UpscaleFilter,
    threads: usize,
) -> Result<Frame> {
    let mut scratch = ResampleScratch::new();
    resample_into_scratch(src, dst, filter, threads, &mut scratch)
}

/// Resample into a caller-owned frame using the caller's scratch buffer.
///
/// Validates the destination size and writes exactly dst pixels in src.fmt into
/// out. out must already have that shape and format; the crate-level entry
/// points check it in release builds too, so this only debug-asserts it.
/// Metadata is deliberately left alone: the caller decides where pts and index
/// come from (the upscaler copies them from src, a backend may remap them).
pub(crate) fn resample_into(
    src: &Frame,
    dst: Dims,
    filter: UpscaleFilter,
    threads: usize,
    scratch: &mut ResampleScratch,
    out: &mut Frame,
) -> Result<()> {
    check_dst(dst)?;
    debug_assert_eq!(out.dims, dst);
    debug_assert_eq!(out.fmt, src.fmt);
    debug_assert_eq!(out.data.len(), src.fmt.buffer_len(dst));
    let (kernel, adaptive) = kernel_of(filter);
    if src.dims == dst {
        // Nothing to resample: an exact copy.
        out.data.copy_from_slice(&src.data);
        return Ok(());
    }
    let bpp = src.fmt.bytes_per_pixel();
    let layout = Layout::of(src.fmt);
    let sw = src.dims.width as usize;
    let sh = src.dims.height as usize;
    let dw = dst.width as usize;

    let taps_x = build(kernel, src.dims.width, dst.width);
    let taps_y = build(kernel, src.dims.height, dst.height);

    let threads = threads.max(1).min(dst.height as usize).max(1);
    let plan = Plan {
        taps_x: &taps_x,
        taps_y: &taps_y,
        dw,
        sw,
        sh,
        threads,
        adaptive,
        layout: &layout,
    };

    if kernel == Kernel::Nearest && !adaptive {
        nearest_scaled(src, bpp, &plan, out);
        return Ok(());
    }

    // The buffers live in the caller's scratch so a streaming caller pays the
    // allocation and the page faults once instead of once per frame.
    let ResampleScratch {
        hpass,
        luma,
        bounds_min,
        bounds_max,
    } = &mut *scratch;
    let scratch_len = dw * sh * bpp;
    grow(hpass, scratch_len);
    // Negative-lobe kernels ring on hard edges; the anti-ringing clamp needs the
    // source luma plane and one min/max pair per (source row, destination column).
    let antiring = matches!(kernel, Kernel::Bicubic | Kernel::Lanczos3);
    if antiring {
        grow(luma, sw * sh);
        grow(bounds_min, dw * sh);
        grow(bounds_max, dw * sh);
        source_luma(src, &layout, threads, &mut luma[..sw * sh]);
    }
    let hpass = &mut hpass[..scratch_len];
    // The ring is only built for the negative-lobe kernels; every other kernel
    // leaves the bounds buffers untouched, so no extra memory is touched.
    let ring = if antiring {
        Some(RingPlanes {
            luma: &luma[..sw * sh],
            min: &mut bounds_min[..dw * sh],
            max: &mut bounds_max[..dw * sh],
        })
    } else {
        None
    };
    match bpp {
        1 => resample_passes::<1>(src, hpass, out, &plan, ring),
        2 => resample_passes::<2>(src, hpass, out, &plan, ring),
        3 => resample_passes::<3>(src, hpass, out, &plan, ring),
        4 => resample_passes::<4>(src, hpass, out, &plan, ring),
        _ => {
            return Err(FluidError::unsupported(
                "upscale: unsupported bytes per pixel",
            ));
        }
    }
    Ok(())
}

/// Resample reusing the caller's horizontal-pass scratch buffer.
///
/// Convenience wrapper that allocates the output frame for stateless callers
/// (the free resize functions); it preserves pts and index from src.
pub(crate) fn resample_into_scratch(
    src: &Frame,
    dst: Dims,
    filter: UpscaleFilter,
    threads: usize,
    scratch: &mut ResampleScratch,
) -> Result<Frame> {
    check_dst(dst)?;
    let mut out = Frame::zeroed(dst, src.fmt);
    resample_into(src, dst, filter, threads, scratch, &mut out)?;
    out.pts_ns = src.pts_ns;
    out.index = src.index;
    Ok(out)
}

/// Split `0..rows` into at most `threads` balanced, contiguous, non-empty
/// chunks.
pub(crate) fn row_chunks(rows: usize, threads: usize) -> Vec<(usize, usize)> {
    if rows == 0 {
        return Vec::new();
    }
    let n = threads.clamp(1, rows);
    let per = rows.div_ceil(n);
    (0..rows)
        .step_by(per)
        .map(|s| (s, (s + per).min(rows)))
        .collect()
}

fn resample_passes<const BPP: usize>(
    src: &Frame,
    scratch: &mut [u8],
    out: &mut Frame,
    plan: &Plan<'_>,
    ring: Option<RingPlanes<'_>>,
) {
    hpass::<BPP>(src, scratch, plan);
    match ring {
        Some(planes) => {
            let RingPlanes { luma, min, max } = planes;
            hbounds(luma, min, max, plan);
            let bounds = RingBounds {
                min: &*min,
                max: &*max,
                dw: plan.dw,
            };
            vpass::<BPP>(scratch, out, plan, Some(bounds));
        }
        None => vpass::<BPP>(scratch, out, plan, None),
    }
}

/// Nearest-neighbour resize: an integer index copy on both axes into out.
fn nearest_scaled(src: &Frame, bpp: usize, plan: &Plan<'_>, out: &mut Frame) {
    let dw = plan.dw;
    let dh = out.dims.height as usize;
    let stride = dw * bpp;
    let src_stride = src.dims.width as usize * bpp;
    let chunks = row_chunks(dh, plan.threads);
    if chunks.len() <= 1 {
        for y in 0..dh {
            let sy = plan.taps_y.idx[y] as usize;
            nearest_row(
                &src.data[sy * src_stride..(sy + 1) * src_stride],
                out.row_mut(y as u32),
                bpp,
                plan.taps_x,
                dw,
            );
        }
        return;
    }
    let data = &mut out.data[..dh * stride];
    thread::scope(|scope| {
        let mut rest: &mut [u8] = data;
        for &(y0, y1) in &chunks {
            let len = (y1 - y0) * stride;
            let (head, tail) = rest.split_at_mut(len);
            rest = tail;
            let src_data: &[u8] = &src.data;
            scope.spawn(move || {
                for (i, y) in (y0..y1).enumerate() {
                    let sy = plan.taps_y.idx[y] as usize;
                    nearest_row(
                        &src_data[sy * src_stride..(sy + 1) * src_stride],
                        &mut head[i * stride..(i + 1) * stride],
                        bpp,
                        plan.taps_x,
                        dw,
                    );
                }
            });
        }
    });
}

#[inline]
fn nearest_row(srow: &[u8], drow: &mut [u8], bpp: usize, taps_x: &Taps, dw: usize) {
    for x in 0..dw {
        let sx = taps_x.idx[x] as usize * bpp;
        let base = x * bpp;
        drow[base..base + bpp].copy_from_slice(&srow[sx..sx + bpp]);
    }
}

/// Fill out with the BT.709 luma of every source pixel, one byte per pixel.
///
/// The anti-ringing clamp compares reconstructed luma against the min/max of the
/// luma of the taps it used, so the plane is captured once before the passes.
fn source_luma(src: &Frame, layout: &Layout, threads: usize, out: &mut [u8]) {
    let w = src.dims.width as usize;
    let h = src.dims.height as usize;
    let chunks = row_chunks(h, threads);
    if chunks.len() <= 1 {
        for y in 0..h {
            luma_row(src.row(y as u32), &mut out[y * w..(y + 1) * w], layout);
        }
        return;
    }
    thread::scope(|scope| {
        let mut rest: &mut [u8] = out;
        for &(y0, y1) in &chunks {
            let len = (y1 - y0) * w;
            let (head, tail) = rest.split_at_mut(len);
            rest = tail;
            scope.spawn(move || {
                for (i, y) in (y0..y1).enumerate() {
                    luma_row(src.row(y as u32), &mut head[i * w..(i + 1) * w], layout);
                }
            });
        }
    });
}

/// Fill the per-destination-column min/max luma of the horizontal tap window for
/// every source row.
fn hbounds(luma: &[u8], min: &mut [u8], max: &mut [u8], plan: &Plan<'_>) {
    let rows = plan.sh;
    let sw = plan.sw;
    let dw = plan.dw;
    let chunks = row_chunks(rows, plan.threads);
    if chunks.len() <= 1 {
        for y in 0..rows {
            hbounds_row(
                &luma[y * sw..(y + 1) * sw],
                &mut min[y * dw..(y + 1) * dw],
                &mut max[y * dw..(y + 1) * dw],
                plan.taps_x,
            );
        }
        return;
    }
    thread::scope(|scope| {
        let mut rest_min: &mut [u8] = min;
        let mut rest_max: &mut [u8] = max;
        for &(y0, y1) in &chunks {
            let len = (y1 - y0) * dw;
            let (hmin, tail_min) = rest_min.split_at_mut(len);
            rest_min = tail_min;
            let (hmax, tail_max) = rest_max.split_at_mut(len);
            rest_max = tail_max;
            scope.spawn(move || {
                for (i, y) in (y0..y1).enumerate() {
                    hbounds_row(
                        &luma[y * sw..(y + 1) * sw],
                        &mut hmin[i * dw..(i + 1) * dw],
                        &mut hmax[i * dw..(i + 1) * dw],
                        plan.taps_x,
                    );
                }
            });
        }
    });
}

/// Min/max luma over the horizontal taps of every destination column.
///
/// Tap indices are contiguous and edge clamped, so every column is either a full
/// sliding window of n luma samples or, near the edges, one truncated by the
/// source bounds. The windows are built once per source row with vector min/max
/// and then looked up per column, which is far cheaper than combining n samples
/// per column.
#[inline]
fn hbounds_row(luma: &[u8], min: &mut [u8], max: &mut [u8], taps: &Taps) {
    let n = taps.n;
    let dw = min.len();
    let sw = luma.len();
    if n < 2 || sw <= n || dw == 0 {
        for d in 0..dw {
            let wbase = d * n;
            let mut lo = u8::MAX;
            let mut hi = 0u8;
            for j in 0..n {
                let v = luma[taps.idx[wbase + j] as usize];
                lo = lo.min(v);
                hi = hi.max(v);
            }
            min[d] = lo;
            max[d] = hi;
        }
        return;
    }
    let mut wmin = vec![0u8; sw];
    let mut wmax = vec![0u8; sw];
    let mut k = 0;
    while k + 16 + n - 1 <= sw {
        let mut lo = load16(luma, k);
        let mut hi = lo;
        for j in 1..n {
            let v = load16(luma, k + j);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        wmin[k..k + 16].copy_from_slice(&lo.to_array());
        wmax[k..k + 16].copy_from_slice(&hi.to_array());
        k += 16;
    }
    while k < sw {
        let end = (k + n).min(sw);
        let mut lo = u8::MAX;
        let mut hi = 0u8;
        for &v in &luma[k..end] {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        wmin[k] = lo;
        wmax[k] = hi;
        k += 1;
    }
    for d in 0..dw {
        let wbase = d * n;
        let s = taps.idx[wbase] as usize;
        if taps.idx[wbase + 1] == 0 {
            // The window starts before the left edge, so the taps repeat the
            // first sample; combine the actual clamped taps.
            let mut lo = u8::MAX;
            let mut hi = 0u8;
            for j in 0..n {
                let v = luma[taps.idx[wbase + j] as usize];
                lo = lo.min(v);
                hi = hi.max(v);
            }
            min[d] = lo;
            max[d] = hi;
        } else {
            min[d] = wmin[s];
            max[d] = wmax[s];
        }
    }
}

/// Combine the per-source-row bounds into the bounds of the vertical tap window
/// of destination row y, one byte per destination column.
#[inline]
fn row_bounds(
    taps: &Taps,
    y: usize,
    bounds: RingBounds<'_>,
    out_min: &mut [u8],
    out_max: &mut [u8],
) {
    let n = taps.n;
    let wbase = y * n;
    let dw = bounds.dw;
    let mut d = 0;
    while d + 16 <= dw {
        let mut lo = load16(bounds.min, taps.idx[wbase] as usize * dw + d);
        let mut hi = load16(bounds.max, taps.idx[wbase] as usize * dw + d);
        for j in 1..n {
            let off = taps.idx[wbase + j] as usize * dw + d;
            lo = lo.min(load16(bounds.min, off));
            hi = hi.max(load16(bounds.max, off));
        }
        out_min[d..d + 16].copy_from_slice(&lo.to_array());
        out_max[d..d + 16].copy_from_slice(&hi.to_array());
        d += 16;
    }
    while d < dw {
        let mut lo = u8::MAX;
        let mut hi = 0u8;
        for j in 0..n {
            let off = taps.idx[wbase + j] as usize * dw + d;
            lo = lo.min(bounds.min[off]);
            hi = hi.max(bounds.max[off]);
        }
        out_min[d] = lo;
        out_max[d] = hi;
        d += 1;
    }
}

/// Clamp one finished row to the anti-ringing luma bounds.
///
/// The reconstruction kernels have negative lobes, so a hard edge can overshoot
/// the local source range and ring. The clamp moves only luma (all colour
/// channels by the same delta) and never touches alpha; on smooth content the
/// bounds already contain the sample and nothing changes.
#[inline]
fn clamp_row<const BPP: usize>(row: &mut [u8], min: &[u8], max: &[u8], layout: &Layout) {
    let w = min.len();
    if BPP <= 2 {
        for (x, v) in row[..w].iter_mut().enumerate() {
            let lo = min[x] as i32;
            let hi = max[x] as i32;
            let value = *v as i32;
            if value < lo || value > hi {
                *v = value.clamp(lo, hi) as u8;
            }
        }
        return;
    }
    let (ir, ig, ib) = (layout.r, layout.g, layout.b);
    for x in 0..w {
        let base = x * BPP;
        let lo = min[x] as i32;
        let hi = max[x] as i32;
        let luma = (54 * row[base + ir] as i32
            + 183 * row[base + ig] as i32
            + 19 * row[base + ib] as i32
            + 128)
            >> 8;
        if luma < lo || luma > hi {
            let delta = luma.clamp(lo, hi) - luma;
            for off in [ir, ig, ib] {
                let v = row[base + off] as i32 + delta;
                row[base + off] = v.clamp(0, 255) as u8;
            }
        }
    }
}

#[inline]
fn clamp_row_at<const BPP: usize>(
    row: &mut [u8],
    y: usize,
    taps: &Taps,
    ring: Option<RingBounds<'_>>,
    out_min: &mut [u8],
    out_max: &mut [u8],
    layout: &Layout,
) {
    if let Some(bounds) = ring {
        row_bounds(taps, y, bounds, out_min, out_max);
        clamp_row::<BPP>(row, out_min, out_max, layout);
    }
}

/// Horizontal pass driver: every source row becomes a scratch row of width dw.
fn hpass<const BPP: usize>(src: &Frame, scratch: &mut [u8], plan: &Plan<'_>) {
    let rows = src.dims.height as usize;
    let src_stride = src.dims.width as usize * BPP;
    let dst_stride = plan.dw * BPP;
    let data: &[u8] = &src.data[..rows * src_stride];
    let chunks = row_chunks(rows, plan.threads);
    if chunks.len() <= 1 {
        for y in 0..rows {
            let srow = &data[y * src_stride..(y + 1) * src_stride];
            let drow = &mut scratch[y * dst_stride..(y + 1) * dst_stride];
            hpass_row::<BPP>(srow, drow, plan.taps_x);
        }
        return;
    }
    thread::scope(|scope| {
        let mut rest: &mut [u8] = &mut scratch[..rows * dst_stride];
        for &(y0, y1) in &chunks {
            let len = (y1 - y0) * dst_stride;
            let (head, tail) = rest.split_at_mut(len);
            rest = tail;
            scope.spawn(move || {
                for (i, y) in (y0..y1).enumerate() {
                    hpass_row::<BPP>(
                        &data[y * src_stride..(y + 1) * src_stride],
                        &mut head[i * dst_stride..(i + 1) * dst_stride],
                        plan.taps_x,
                    );
                }
            });
        }
    });
}

#[inline]
pub(crate) fn hpass_row<const BPP: usize>(src: &[u8], dst: &mut [u8], taps: &Taps) {
    if BPP == 4 {
        hpass_row_rgba(src, dst, taps);
        return;
    }
    let n = taps.n;
    let dw = taps.idx.len() / n;
    for d in 0..dw {
        let mut acc = [0.0f32; BPP];
        let wbase = d * n;
        for j in 0..n {
            let off = taps.idx[wbase + j] as usize * BPP;
            let w = taps.w[wbase + j];
            let px = &src[off..off + BPP];
            for c in 0..BPP {
                acc[c] += px[c] as f32 * w;
            }
        }
        let base = d * BPP;
        for c in 0..BPP {
            dst[base + c] = store_u8(acc[c]);
        }
    }
}

/// Horizontal pass for four-byte pixels.
///
/// The four lanes are the channels of one destination pixel, so the tap weight
/// is broadcast and the source pixels are read as whole 4-byte vectors: four
/// multiply-adds per tap instead of four separate scalar ones. The tap order is
/// unchanged, so the bytes match the scalar path exactly.
#[inline]
fn hpass_row_rgba(src: &[u8], dst: &mut [u8], taps: &Taps) {
    let n = taps.n;
    let dw = taps.idx.len() / n;
    for d in 0..dw {
        let wbase = d * n;
        let mut acc = load4(src, taps.idx[wbase] as usize * 4) * f32x4::splat(taps.w[wbase]);
        for j in 1..n {
            acc = load4(src, taps.idx[wbase + j] as usize * 4)
                .mul_add(f32x4::splat(taps.w[wbase + j]), acc);
        }
        store4(acc, dst, d * 4);
    }
}

/// Vertical pass driver: combines scratch rows into destination rows, and in
/// adaptive mode refines each written row once its neighbourhood exists.
fn vpass<const BPP: usize>(
    scratch: &[u8],
    out: &mut Frame,
    plan: &Plan<'_>,
    ring: Option<RingBounds<'_>>,
) {
    let dh = out.dims.height as usize;
    let stride = plan.dw * BPP;
    let chunks = row_chunks(dh, plan.threads);
    if plan.threads <= 1 || chunks.len() <= 1 {
        vpass_all::<BPP>(scratch, out, plan.taps_y, plan.adaptive, plan.layout, ring);
        return;
    }
    let data = &mut out.data[..dh * stride];
    if plan.adaptive {
        vpass_adaptive::<BPP>(scratch, data, plan, &chunks, ring);
    } else {
        vpass_plain::<BPP>(scratch, data, plan, &chunks, ring);
    }
}

/// One contiguous row chunk of the non-adaptive vertical pass.
fn vpass_plain<const BPP: usize>(
    scratch: &[u8],
    data: &mut [u8],
    plan: &Plan<'_>,
    chunks: &[(usize, usize)],
    ring: Option<RingBounds<'_>>,
) {
    let stride = plan.dw * BPP;
    let dw = plan.dw;
    thread::scope(|scope| {
        let mut rest: &mut [u8] = data;
        for &(y0, y1) in chunks {
            let len = (y1 - y0) * stride;
            let (head, tail) = rest.split_at_mut(len);
            rest = tail;
            scope.spawn(move || {
                let mut bmin = vec![0u8; if ring.is_some() { dw } else { 0 }];
                let mut bmax = vec![0u8; if ring.is_some() { dw } else { 0 }];
                dispatch(move || {
                    for (i, y) in (y0..y1).enumerate() {
                        let drow = &mut head[i * stride..(i + 1) * stride];
                        vpass_row::<BPP>(scratch, stride, drow, plan.taps_y, y);
                        clamp_row_at::<BPP>(
                            drow,
                            y,
                            plan.taps_y,
                            ring,
                            &mut bmin,
                            &mut bmax,
                            plan.layout,
                        );
                    }
                });
            });
        }
    });
}

/// One contiguous row chunk of the adaptive vertical pass.
///
/// Each worker streams its chunk exactly like the serial loop: write the base
/// row, capture its luma, then refine the previous row (which needs the luma
/// rows above and below it). Only three luma rows rotate through the worker, so
/// the detail-restore pass adds O(width) scratch per worker rather than a full
/// chunk-sized buffer.
///
/// The luma rows just outside the chunk are recomputed from the scratch buffer,
/// so a worker never reads an output row another worker owns. Luma is always
/// captured from *base* rows before those rows are refined, which is also what
/// the serial loop does, so the parallel and serial outputs are bit-identical.
fn vpass_adaptive<const BPP: usize>(
    scratch: &[u8],
    data: &mut [u8],
    plan: &Plan<'_>,
    chunks: &[(usize, usize)],
    ring: Option<RingBounds<'_>>,
) {
    let stride = plan.dw * BPP;
    let dw = plan.dw;
    let dh = data.len() / stride;
    thread::scope(|scope| {
        let mut rest: &mut [u8] = data;
        for &(y0, y1) in chunks {
            let len = (y1 - y0) * stride;
            let (head, tail) = rest.split_at_mut(len);
            rest = tail;
            scope.spawn(move || {
                dispatch(move || {
                    // ring_luma[0] = luma[y-2], [1] = luma[y-1], [2] = luma[y].
                    // The rows stay in f32 so the refine kernel converts each luma
                    // sample once, not once per tap.
                    let mut ring_luma: [Vec<f32>; 3] =
                        [vec![0.0f32; dw], vec![0.0f32; dw], vec![0.0f32; dw]];
                    let mut deltas = vec![0i32; dw];
                    let mut edge = vec![0u8; stride];
                    let mut bmin = vec![0u8; if ring.is_some() { dw } else { 0 }];
                    let mut bmax = vec![0u8; if ring.is_some() { dw } else { 0 }];
                    if y0 > 0 {
                        vpass_row::<BPP>(scratch, stride, &mut edge, plan.taps_y, y0 - 1);
                        clamp_row_at::<BPP>(
                            &mut edge,
                            y0 - 1,
                            plan.taps_y,
                            ring,
                            &mut bmin,
                            &mut bmax,
                            plan.layout,
                        );
                        luma_row_f32(&edge, &mut ring_luma[1], plan.layout);
                    }
                    for (i, y) in (y0..y1).enumerate() {
                        let drow = &mut head[i * stride..(i + 1) * stride];
                        vpass_row::<BPP>(scratch, stride, drow, plan.taps_y, y);
                        clamp_row_at::<BPP>(
                            drow,
                            y,
                            plan.taps_y,
                            ring,
                            &mut bmin,
                            &mut bmax,
                            plan.layout,
                        );
                        luma_row_f32(drow, &mut ring_luma[2], plan.layout);
                        if y > y0 {
                            // Row y-1 has all three luma rows; top is clamped to
                            // mid for the second row of the image.
                            let top = if y >= 2 { &ring_luma[0] } else { &ring_luma[1] };
                            let prev = &mut head[(i - 1) * stride..i * stride];
                            apply_delta_row(
                                prev,
                                top,
                                &ring_luma[1],
                                &ring_luma[2],
                                &mut deltas,
                                plan.layout,
                                REFINE_STRENGTH_Q8,
                            );
                        }
                        ring_luma.rotate_left(1);
                    }
                    if y1 == dh {
                        // Last row of the frame: bot is clamped to mid.
                        let last = &mut head[(y1 - y0 - 1) * stride..(y1 - y0) * stride];
                        let top = if dh >= 2 {
                            &ring_luma[0]
                        } else {
                            &ring_luma[1]
                        };
                        apply_delta_row(
                            last,
                            top,
                            &ring_luma[1],
                            &ring_luma[1],
                            &mut deltas,
                            plan.layout,
                            REFINE_STRENGTH_Q8,
                        );
                    } else {
                        // Chip in the luma of the first row the next worker owns,
                        // then refine this chunk's last row.
                        vpass_row::<BPP>(scratch, stride, &mut edge, plan.taps_y, y1);
                        clamp_row_at::<BPP>(
                            &mut edge,
                            y1,
                            plan.taps_y,
                            ring,
                            &mut bmin,
                            &mut bmax,
                            plan.layout,
                        );
                        luma_row_f32(&edge, &mut ring_luma[2], plan.layout);
                        let last = &mut head[(y1 - y0 - 1) * stride..(y1 - y0) * stride];
                        let top = if y1 >= 2 {
                            &ring_luma[0]
                        } else {
                            &ring_luma[1]
                        };
                        apply_delta_row(
                            last,
                            top,
                            &ring_luma[1],
                            &ring_luma[2],
                            &mut deltas,
                            plan.layout,
                            REFINE_STRENGTH_Q8,
                        );
                    }
                })
            });
        }
    });
}

/// Streaming serial vertical pass, used for small frames and single workers.
fn vpass_all<const BPP: usize>(
    scratch: &[u8],
    out: &mut Frame,
    taps: &Taps,
    adaptive: bool,
    layout: &Layout,
    bounds: Option<RingBounds<'_>>,
) {
    let dw = out.dims.width as usize;
    let dh = out.dims.height as usize;
    let stride = dw * BPP;
    let mut ring: [Vec<f32>; 3] = [vec![0.0f32; dw], vec![0.0f32; dw], vec![0.0f32; dw]];
    let mut deltas = vec![0i32; dw];
    let mut bmin = vec![0u8; if bounds.is_some() { dw } else { 0 }];
    let mut bmax = vec![0u8; if bounds.is_some() { dw } else { 0 }];
    dispatch(move || {
        for y in 0..dh {
            {
                let drow = out.row_mut(y as u32);
                vpass_row::<BPP>(scratch, stride, drow, taps, y);
                clamp_row_at::<BPP>(drow, y, taps, bounds, &mut bmin, &mut bmax, layout);
            }
            if !adaptive {
                continue;
            }
            let slot = y % 3;
            luma_row_f32(out.row(y as u32), &mut ring[slot], layout);
            if y >= 1 {
                let mid = (y - 1) % 3;
                let top = y.saturating_sub(2) % 3;
                let (t, m, b) = (&ring[top], &ring[mid], &ring[slot]);
                let drow = out.row_mut((y - 1) as u32);
                apply_delta_row(drow, t, m, b, &mut deltas, layout, REFINE_STRENGTH_Q8);
            }
        }
        if adaptive && dh >= 1 {
            let last = dh - 1;
            let mid = last % 3;
            let top = if last >= 1 { (last - 1) % 3 } else { mid };
            let (t, m) = (&ring[top], &ring[mid]);
            let drow = out.row_mut(last as u32);
            apply_delta_row(drow, t, m, m, &mut deltas, layout, REFINE_STRENGTH_Q8);
        }
    });
}

/// One output row of the vertical pass.
///
/// The row is processed in 16-byte chunks: for each chunk the taps are
/// accumulated in ascending tap order into 16 f32 registers, and every
/// destination byte is rounded once at the end. Keeping the accumulator in
/// registers instead of a full-width f32 row removes two passes over a
/// row-sized scratch buffer per output row, which matters because the vertical
/// pass is memory-bound at 4K. The tap count is a const generic so the tap loop
/// is fully unrolled and auto-vectorises inside the runtime-dispatch closure.
///
/// Summing in ascending tap order with a separate multiply and add gives the
/// same IEEE result as the previous f32x4 mul_add chain, so the output stays
/// bit-identical to the scalar reference.
#[inline(always)]
pub(crate) fn vpass_row<const BPP: usize>(
    scratch: &[u8],
    stride: usize,
    dst: &mut [u8],
    taps: &Taps,
    y: usize,
) {
    match taps.n {
        1 => vpass_row_n::<BPP, 1>(scratch, stride, dst, taps, y),
        2 => vpass_row_n::<BPP, 2>(scratch, stride, dst, taps, y),
        4 => vpass_row_n::<BPP, 4>(scratch, stride, dst, taps, y),
        6 => vpass_row_n::<BPP, 6>(scratch, stride, dst, taps, y),
        n => vpass_row_dyn::<BPP>(scratch, stride, dst, taps, y, n),
    }
}

#[inline(always)]
fn vpass_row_n<const BPP: usize, const N: usize>(
    scratch: &[u8],
    stride: usize,
    dst: &mut [u8],
    taps: &Taps,
    y: usize,
) {
    let wbase = y * N;
    let mut idx = [0usize; N];
    let mut ws = [0.0f32; N];
    for (j, (i, w)) in idx.iter_mut().zip(ws.iter_mut()).enumerate() {
        *i = taps.idx[wbase + j] as usize * stride;
        *w = taps.w[wbase + j];
    }
    let mut off = 0;
    while off + 16 <= stride {
        let mut acc = [0.0f32; 16];
        for j in 0..N {
            let row = &scratch[idx[j] + off..idx[j] + off + 16];
            let w = ws[j];
            for (a, s) in acc.iter_mut().zip(row) {
                *a += *s as f32 * w;
            }
        }
        for (d, a) in dst[off..off + 16].iter_mut().zip(acc.iter()) {
            *d = store_u8(*a);
        }
        off += 16;
    }
    while off < stride {
        let mut a = 0.0f32;
        for j in 0..N {
            a += scratch[idx[j] + off] as f32 * ws[j];
        }
        dst[off] = store_u8(a);
        off += 1;
    }
}

fn vpass_row_dyn<const BPP: usize>(
    scratch: &[u8],
    stride: usize,
    dst: &mut [u8],
    taps: &Taps,
    y: usize,
    n: usize,
) {
    let wbase = y * n;
    let mut off = 0;
    while off + 16 <= stride {
        let mut acc = [0.0f32; 16];
        for j in 0..n {
            let base = taps.idx[wbase + j] as usize * stride;
            let row = &scratch[base + off..base + off + 16];
            let w = taps.w[wbase + j];
            for (a, s) in acc.iter_mut().zip(row) {
                *a += *s as f32 * w;
            }
        }
        for (d, a) in dst[off..off + 16].iter_mut().zip(acc.iter()) {
            *d = store_u8(*a);
        }
        off += 16;
    }
    while off < stride {
        let mut a = 0.0f32;
        for j in 0..n {
            let base = taps.idx[wbase + j] as usize * stride;
            a += scratch[base + off] as f32 * taps.w[wbase + j];
        }
        dst[off] = store_u8(a);
        off += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluid_core::PixelFormat;

    const FORMATS: [PixelFormat; 4] = [
        PixelFormat::Rgba8,
        PixelFormat::Bgra8,
        PixelFormat::Rgb8,
        PixelFormat::Luma8,
    ];
    const FILTERS: [UpscaleFilter; 6] = [
        UpscaleFilter::Off,
        UpscaleFilter::Nearest,
        UpscaleFilter::Bilinear,
        UpscaleFilter::Bicubic,
        UpscaleFilter::Lanczos,
        UpscaleFilter::Adaptive,
    ];

    /// Deterministic pseudo-random image: every byte differs, so an off-by-one
    /// in the parallel split shows up immediately.
    fn image(dims: Dims, fmt: PixelFormat, seed: u32) -> Frame {
        let bpp = fmt.bytes_per_pixel();
        let mut data = vec![0u8; dims.area() as usize * bpp];
        let mut s = seed | 1;
        for (i, b) in data.iter_mut().enumerate() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *b = ((s >> 8) as u8).wrapping_add(i as u8).wrapping_mul(31);
        }
        let mut f = Frame::new(dims, fmt, data).unwrap();
        f.pts_ns = 7_000_000_123;
        f.index = 42;
        f
    }

    fn assert_same(a: &Frame, b: &Frame, what: &str) {
        assert_eq!(a.dims, b.dims, "{what}: dims");
        assert_eq!(a.fmt, b.fmt, "{what}: fmt");
        assert_eq!(a.pts_ns, b.pts_ns, "{what}: pts");
        assert_eq!(a.index, b.index, "{what}: index");
        assert_eq!(a.data.len(), b.data.len(), "{what}: len");
        if let Some(i) = a.data.iter().zip(b.data.iter()).position(|(x, y)| x != y) {
            panic!(
                "{what}: first differing byte at {i} (x={}, y={}): {} vs {}",
                i % (a.dims.width as usize * a.fmt.bytes_per_pixel()),
                i / (a.dims.width as usize * a.fmt.bytes_per_pixel()),
                a.data[i],
                b.data[i]
            );
        }
    }

    #[test]
    fn row_chunks_cover_every_row_once() {
        for rows in [0usize, 1, 2, 5, 23, 97, 1000] {
            for threads in 1..=8usize {
                let chunks = row_chunks(rows, threads);
                if rows == 0 {
                    assert!(chunks.is_empty());
                    continue;
                }
                assert_eq!(
                    chunks.first().unwrap().0,
                    0,
                    "rows={rows} threads={threads}"
                );
                assert_eq!(
                    chunks.last().unwrap().1,
                    rows,
                    "rows={rows} threads={threads}"
                );
                assert!(chunks.len() <= threads.min(rows));
                for pair in chunks.windows(2) {
                    assert_eq!(pair[0].1, pair[1].0, "chunks must be contiguous");
                }
                let total: usize = chunks.iter().map(|&(a, b)| b - a).sum();
                assert_eq!(total, rows);
                for &(a, b) in &chunks {
                    assert!(a < b, "empty chunk");
                }
            }
        }
    }

    /// The sliding-window bounds are an optimisation of the per-column tap
    /// reduction; both must agree bite for bite.
    #[test]
    fn hbounds_match_naive_tap_reduction() {
        for kernel in [Kernel::Bicubic, Kernel::Lanczos3] {
            for (sw, dw) in [(1u32, 4u32), (3, 9), (8, 16), (13, 7), (64, 200), (97, 31)] {
                let luma: Vec<u8> = (0..sw)
                    .map(|i| ((i.wrapping_mul(37).wrapping_add(11)) % 251) as u8)
                    .collect();
                let taps = build(kernel, sw, dw);
                let mut min = vec![0u8; dw as usize];
                let mut max = vec![0u8; dw as usize];
                hbounds_row(&luma, &mut min, &mut max, &taps);
                for d in 0..dw as usize {
                    let mut lo = u8::MAX;
                    let mut hi = 0u8;
                    for j in 0..taps.n {
                        let v = luma[taps.idx[d * taps.n + j] as usize];
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                    assert_eq!(min[d], lo, "{kernel:?} {sw}->{dw} column {d}");
                    assert_eq!(max[d], hi, "{kernel:?} {sw}->{dw} column {d}");
                }
            }
        }
    }

    #[test]
    fn parallel_matches_serial_all_filters_and_formats() {
        let src_dims = Dims::new(96, 64);
        let dst = Dims::new(192, 144);
        for (fi, filter) in FILTERS.iter().enumerate() {
            for (pi, fmt) in FORMATS.iter().enumerate() {
                let src = image(src_dims, *fmt, (fi * 31 + pi * 7 + 1) as u32);
                let serial = resample_threads(&src, dst, *filter, 1).unwrap();
                let parallel = resample_threads(&src, dst, *filter, 3).unwrap();
                assert_same(&serial, &parallel, &format!("{filter:?}/{fmt}"));
            }
        }
    }

    #[test]
    fn parallel_matches_serial_odd_sizes() {
        let src = image(Dims::new(37, 29), PixelFormat::Rgba8, 9);
        let dst = Dims::new(83, 67);
        for filter in FILTERS {
            let serial = resample_threads(&src, dst, filter, 1).unwrap();
            let parallel = resample_threads(&src, dst, filter, 4).unwrap();
            assert_same(&serial, &parallel, &format!("{filter:?} odd"));
        }
    }

    #[test]
    fn adaptive_chunk_edges_match_serial() {
        // 203 destination rows over 3 workers gives uneven chunks, and the
        // detail-restore kernel reads one row above and below each chunk.
        let src = image(Dims::new(50, 101), PixelFormat::Rgba8, 3);
        let dst = Dims::new(100, 203);
        let serial = resample_threads(&src, dst, UpscaleFilter::Adaptive, 1).unwrap();
        for threads in [2usize, 3, 5, 7] {
            let parallel = resample_threads(&src, dst, UpscaleFilter::Adaptive, threads).unwrap();
            assert_same(&serial, &parallel, &format!("adaptive threads={threads}"));
        }
    }

    #[test]
    fn adaptive_single_row_matches_serial() {
        let src = image(Dims::new(40, 6), PixelFormat::Rgba8, 11);
        let dst = Dims::new(80, 1);
        let serial = resample_threads(&src, dst, UpscaleFilter::Adaptive, 1).unwrap();
        let parallel = resample_threads(&src, dst, UpscaleFilter::Adaptive, 4).unwrap();
        assert_same(&serial, &parallel, "adaptive one row");
    }

    #[test]
    fn identity_is_exact_copy() {
        let src = image(Dims::new(64, 48), PixelFormat::Bgra8, 5);
        let out = resample_threads(&src, src.dims, UpscaleFilter::Lanczos, 4).unwrap();
        assert_same(&src, &out, "identity");
        assert_eq!(out.data, src.data);
    }

    #[test]
    fn nearest_parallel_matches_serial_and_replicates() {
        let src = image(Dims::new(2, 2), PixelFormat::Rgba8, 13);
        let dst = Dims::new(4, 4);
        let serial = resample_threads(&src, dst, UpscaleFilter::Nearest, 1).unwrap();
        let parallel = resample_threads(&src, dst, UpscaleFilter::Nearest, 2).unwrap();
        assert_same(&serial, &parallel, "nearest");
        for y in 0..4u32 {
            for x in 0..4u32 {
                assert_eq!(
                    parallel.pixel(x, y),
                    src.pixel(x / 2, y / 2),
                    "nearest must replicate 2x2 blocks at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn small_frames_use_the_serial_default() {
        // 200x160 is below PAR_MIN_PIXELS, so no workers are spawned.
        assert_eq!(default_threads(Dims::new(100, 80), Dims::new(200, 160)), 1);
        assert_eq!(default_threads(Dims::new(16, 16), Dims::new(32, 32)), 1);
    }

    #[test]
    fn default_threads_is_bounded() {
        let avail = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let threads = default_threads(Dims::new(1920, 1080), Dims::new(3840, 2160));
        assert!(threads >= 1 && threads <= avail);
        assert!(threads <= 2160 / MIN_ROWS_PER_THREAD);
        // A very short but large frame still gets at least one worker.
        assert_eq!(default_threads(Dims::new(4000, 4), Dims::new(8000, 8)), 1);
    }

    #[test]
    fn empty_destination_is_rejected() {
        let src = image(Dims::new(8, 8), PixelFormat::Rgba8, 1);
        assert!(resample_threads(&src, Dims::new(0, 4), UpscaleFilter::Bilinear, 2).is_err());
        assert!(resample_threads(&src, Dims::new(4, 0), UpscaleFilter::Bilinear, 2).is_err());
    }

    /// Scalar reference for the separable RGBA path: the same tap tables and the
    /// same f32 tap order (including the fused multiply-add the vector code
    /// uses), so every destination byte must match exactly.
    fn reference_rgba(src: &Frame, dst: Dims, kernel: Kernel) -> Frame {
        let sw = src.dims.width as usize;
        let sh = src.dims.height as usize;
        let dw = dst.width as usize;
        let dh = dst.height as usize;
        let layout = Layout::of(PixelFormat::Rgba8);
        let tx = build(kernel, src.dims.width, dst.width);
        let ty = build(kernel, src.dims.height, dst.height);
        let antiring = matches!(kernel, Kernel::Bicubic | Kernel::Lanczos3);
        let mut luma = vec![0u8; sw * sh];
        for y in 0..sh {
            luma_row(src.row(y as u32), &mut luma[y * sw..(y + 1) * sw], &layout);
        }
        let mut hmin = vec![0u8; dw * sh];
        let mut hmax = vec![0u8; dw * sh];
        if antiring {
            for y in 0..sh {
                hbounds_row(
                    &luma[y * sw..(y + 1) * sw],
                    &mut hmin[y * dw..(y + 1) * dw],
                    &mut hmax[y * dw..(y + 1) * dw],
                    &tx,
                );
            }
        }
        let mut scratch = vec![0u8; dw * sh * 4];
        for y in 0..sh {
            let row = src.row(y as u32);
            for x in 0..dw {
                for c in 0..4 {
                    let mut a = 0.0f32;
                    for t in 0..tx.n {
                        let v = row[tx.idx[x * tx.n + t] as usize * 4 + c] as f32;
                        let w = tx.w[x * tx.n + t];
                        a = if t == 0 { v * w } else { v.mul_add(w, a) };
                    }
                    scratch[(y * dw + x) * 4 + c] = store_u8(a);
                }
            }
        }
        let bounds = RingBounds {
            min: &hmin,
            max: &hmax,
            dw,
        };
        let mut rb_min = vec![0u8; dw];
        let mut rb_max = vec![0u8; dw];
        let mut out = Frame::zeroed(dst, PixelFormat::Rgba8);
        for y in 0..dh {
            if antiring {
                row_bounds(&ty, y, bounds, &mut rb_min, &mut rb_max);
            }
            let drow = out.row_mut(y as u32);
            for x in 0..dw {
                for c in 0..4 {
                    let mut a = 0.0f32;
                    for t in 0..ty.n {
                        let sy = ty.idx[y * ty.n + t] as usize;
                        let v = scratch[(sy * dw + x) * 4 + c] as f32;
                        let w = ty.w[y * ty.n + t];
                        a = if t == 0 { v * w } else { v.mul_add(w, a) };
                    }
                    drow[x * 4 + c] = store_u8(a);
                }
            }
            if antiring {
                clamp_row::<4>(drow, &rb_min, &rb_max, &layout);
            }
        }
        out
    }

    fn kernel_filter(kernel: Kernel) -> UpscaleFilter {
        match kernel {
            Kernel::Nearest => UpscaleFilter::Nearest,
            Kernel::Bilinear => UpscaleFilter::Bilinear,
            Kernel::Bicubic => UpscaleFilter::Bicubic,
            Kernel::Lanczos3 => UpscaleFilter::Lanczos,
        }
    }

    #[test]
    fn vectorised_rgba_pass_matches_scalar_reference() {
        let src = image(Dims::new(5, 4), PixelFormat::Rgba8, 0x51ab_c0de);
        for (dst, kernel) in [
            (Dims::new(13, 9), Kernel::Bicubic),
            (Dims::new(11, 7), Kernel::Lanczos3),
            (Dims::new(20, 2), Kernel::Bilinear),
        ] {
            let expect = reference_rgba(&src, dst, kernel);
            let actual = resample(&src, dst, kernel_filter(kernel)).unwrap();
            assert_eq!(actual.data, expect.data, "{dst}: {kernel:?}");
        }
    }
}
