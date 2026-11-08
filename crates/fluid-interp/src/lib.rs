#![forbid(unsafe_code)]
//! Frame synthesis: build in-between frames from two real frames.
//!
//! Motion-compensated interpolation: every output pixel is gathered from
//! `prev` and `next` along the estimated flow and blended by the temporal
//! position. Pixels whose flow confidence is below the threshold are
//! disocclusions or cut regions: with `occlusion_fill` they are filled with the
//! same motion-compensated sample (the field around them still follows the
//! uncovered surface) and otherwise they fall back to a plain temporal blend.
//! The measured effect of that fill on synthetic ground truth is +6 to +14 dB
//! PSNR versus copying raw pixels from the nearer frame, see
//! `examples/quality.rs`.
//!
//! Synthesis runs in fixed 16-row bands on `std::thread::scope` workers (up to
//! `available_parallelism()`, capped at 16), so the generated pixels and the
//! aggregate metrics are identical regardless of the thread count;
//! `generate` reuses internal luma scratch buffers between calls.
//!
//! Four-channel formats run through a hand-written `wide::f32x4` kernel: one
//! pixel's four channels occupy the four lanes, so the bilinear gathers, the
//! temporal blend and the crossfade mix are vectorised while the per-lane
//! arithmetic order (and therefore every output byte) matches the scalar path.
//! Each bilinear tap is fetched with a single packed 32-bit load that is then
//! unpacked into the four lanes.
//!
//! A final edge-preserving unsharp pass restores the contrast that the bilinear
//! gather costs thin high-contrast features such as subtitles, crosshairs and UI
//! rules (the worst interpolation scene by far). It only rewrites pixels whose
//! two-axis gradient exceeds `SHARP_THRESHOLD` and whose flow is non-zero (a
//! static pixel is copied exactly, so static content stays byte-identical), the
//! correction is clamped to the local neighbourhood range so a step edge cannot
//! gain a halo, and the gain fades to zero towards alpha 0 and 1, where the
//! output must reproduce a real frame exactly. Set `FLUID_NO_SHARP=1` to disable
//! it, for A/B measurements or feeds that should not be sharpened.
//!
//! When the flow estimate reports a significant global rotation, synthesis
//! switches to the rotation-aware path: the rotation part of the motion is
//! applied analytically with the temporal fraction and only the residual flow
//! is interpolated linearly, so both gather points follow the same arc (see
//! `synth_band_rot`).

use fluid_core::{Dims, FlowConfig, FluidError, Frame, InterpolatorConfig, LumaPlane, Result};
use fluid_flow::{FlowEstimate, FlowParams, SceneCutDetector, Vec2};
use wide::f32x4;

/// Runtime parameters of the synthesiser.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterpParams {
    /// Weight of the synthesised frame versus the neighbours (0..=1).
    pub blend: f32,
    /// Fill occluded pixels with a motion-compensated sample instead of a
    /// plain temporal blend.
    pub occlusion_fill: bool,
    /// Scenes cut when the frame difference score exceeds this.
    pub scene_cut_threshold: f32,
    /// Use crossfade instead of motion compensation when flow is unusable.
    pub fallback_crossfade: bool,
}

impl Default for InterpParams {
    fn default() -> Self {
        Self {
            blend: 1.0,
            occlusion_fill: true,
            scene_cut_threshold: 0.18,
            fallback_crossfade: true,
        }
    }
}

impl From<&InterpolatorConfig> for InterpParams {
    fn from(c: &InterpolatorConfig) -> Self {
        Self {
            blend: c.blend,
            occlusion_fill: c.occlusion_fill,
            scene_cut_threshold: c.scene_cut_threshold,
            fallback_crossfade: true,
        }
    }
}

/// Per-pass telemetry.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct InterpMetrics {
    pub frames: u32,
    pub blended_px: u64,
    pub occluded_px: u64,
    pub mean_confidence: f32,
    pub mean_flow_px: f32,
    pub scene_cut: bool,
}

impl std::fmt::Display for InterpMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "frames={} occluded={} conf={:.3} flow={:.2}px",
            self.frames, self.occluded_px, self.mean_confidence, self.mean_flow_px
        )
    }
}

/// Bilinear sample of one pixel from a tightly packed frame, edge clamped.
#[inline]
fn sample_px<const BPP: usize>(data: &[u8], w: usize, h: usize, x: f32, y: f32) -> [f32; BPP] {
    let max_x = w as f32 - 1.0;
    let max_y = h as f32 - 1.0;
    let x = if x.is_finite() {
        x.clamp(0.0, max_x)
    } else {
        0.0
    };
    let y = if y.is_finite() {
        y.clamp(0.0, max_y)
    } else {
        0.0
    };
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let i00 = (y0 * w + x0) * BPP;
    let i10 = (y0 * w + x1) * BPP;
    let i01 = (y1 * w + x0) * BPP;
    let i11 = (y1 * w + x1) * BPP;
    let mut out = [0.0f32; BPP];
    for (c, slot) in out.iter_mut().enumerate() {
        let a = f32::from(data[i00 + c]);
        let b = f32::from(data[i10 + c]);
        let d = f32::from(data[i01 + c]);
        let e = f32::from(data[i11 + c]);
        let top = a + (b - a) * fx;
        let bot = d + (e - d) * fx;
        *slot = top + (bot - top) * fy;
    }
    out
}

/// Four tightly packed bytes as an `f32x4` of channel values.
#[inline]
fn load_bytes4(src: &[u8]) -> f32x4 {
    f32x4::from([
        f32::from(src[0]),
        f32::from(src[1]),
        f32::from(src[2]),
        f32::from(src[3]),
    ])
}

/// One packed four-channel pixel, unpacked into an `f32x4` of channel values.
///
/// A single bounds-checked 32-bit load plus three shifts replaces four
/// separately bounds-checked byte loads. The synthesiser is load-port bound, so
/// this is most of the reason the four-channel kernel is fast.
#[inline]
fn load_px4(data: &[u8], i: usize) -> f32x4 {
    let p = match data.get(i..i + 4) {
        Some(bytes) => u32::from_le_bytes(bytes.try_into().expect("four bytes")),
        None => 0,
    };
    f32x4::from([
        (p & 0xff) as f32,
        ((p >> 8) & 0xff) as f32,
        ((p >> 16) & 0xff) as f32,
        (p >> 24) as f32,
    ])
}

/// Round four channel values and store them as four bytes.
#[inline]
fn write_u8x4(dst: &mut [u8], v: f32x4) {
    let lanes = v.to_array();
    dst[0] = round_u8(lanes[0]);
    dst[1] = round_u8(lanes[1]);
    dst[2] = round_u8(lanes[2]);
    dst[3] = round_u8(lanes[3]);
}

/// Bilinear sample of one four-channel pixel, edge clamped, all four channels
/// at once. Per-lane arithmetic matches `sample_px` exactly.
#[inline]
fn sample_px4(data: &[u8], w: usize, h: usize, x: f32, y: f32) -> f32x4 {
    const BPP: usize = 4;
    let max_x = w as f32 - 1.0;
    let max_y = h as f32 - 1.0;
    let x = if x.is_finite() {
        x.clamp(0.0, max_x)
    } else {
        0.0
    };
    let y = if y.is_finite() {
        y.clamp(0.0, max_y)
    } else {
        0.0
    };
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let fx = f32x4::splat(x - x0 as f32);
    let fy = f32x4::splat(y - y0 as f32);
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let i00 = (y0 * w + x0) * BPP;
    let i10 = (y0 * w + x1) * BPP;
    let i01 = (y1 * w + x0) * BPP;
    let i11 = (y1 * w + x1) * BPP;
    let a = load_px4(data, i00);
    let b = load_px4(data, i10);
    let d = load_px4(data, i01);
    let e = load_px4(data, i11);
    let top = a + (b - a) * fx;
    let bot = d + (e - d) * fx;
    top + (bot - top) * fy
}

/// Linear blend of two packed four-channel pixels.
#[inline]
fn lerp_bytes4(a: &[u8], b: &[u8], t: f32x4) -> f32x4 {
    let ca = load_bytes4(a);
    let cb = load_bytes4(b);
    ca + (cb - ca) * t
}
/// Round a channel value back to 8 bits with saturation.
#[inline]
fn round_u8(v: f32) -> u8 {
    if v.is_nan() {
        return 0;
    }
    (v.clamp(0.0, 255.0) + 0.5) as u8
}

/// Interpolate a 64-bit timestamp by the temporal position.
#[inline]
fn lerp_i64(a: i64, b: i64, t: f32) -> i64 {
    (a as f64 + (b as f64 - a as f64) * f64::from(t)).round() as i64
}

/// Local-gradient threshold of the edge-preserving unsharp pass, in 8-bit
/// levels summed over the two axes (51/255 is 0.2 in normalised value units).
const SHARP_THRESHOLD: f32 = 51.0;

/// Gain of the edge-preserving unsharp pass at alpha 0.5.
const SHARP_GAIN: f32 = 1.0;

/// Edge-preserving 2D unsharp mask over an already synthesised frame.
///
/// The bilinear gather samples both endpoints roughly one pixel off a thin
/// feature, which halves its amplitude and spreads it over two pixels: that
/// costs 8-10 dB of PSNR on a one-pixel line (see `examples/quality.rs`). This
/// pass inverts that blur where the local gradient proves a real edge exists
/// and copies every other pixel through unchanged, so it never touches smooth
/// texture, grain or flat regions. Channels are independent; a four-channel
/// frame runs one pixel per `wide::f32x4` with its lanes as the channels.
fn sharpen_frame(src: &Frame, forward: &[Vec2], alpha: f32, threads: usize) -> Frame {
    let mut dst = Frame::zeroed(src.dims, src.fmt);
    dst.index = src.index;
    dst.pts_ns = src.pts_ns;
    // The blur is worst at the temporal midpoint; keep the endpoints exact.
    let gain = SHARP_GAIN * 4.0 * alpha * (1.0 - alpha);
    // A pixel whose flow is (nearly) zero is sampled exactly at its own
    // position, so the gather introduced no blur there and there is nothing to
    // restore. Skipping those pixels keeps static content bit-identical.
    let any_motion = forward.iter().any(|v| v.x.abs() + v.y.abs() > 1.0e-3);
    if gain <= 0.0 || !any_motion || sharpening_disabled() {
        dst.data.copy_from_slice(&src.data);
        return dst;
    }
    let w = src.dims.width as usize;
    let h = src.dims.height as usize;
    let bpp = src.fmt.bytes_per_pixel();
    let row_len = w * bpp;
    let src_data = &src.data;
    match bpp {
        1 => sharpen_rows::<1>(
            src_data,
            &mut dst.data,
            forward,
            w,
            h,
            row_len,
            gain,
            threads,
        ),
        3 => sharpen_rows::<3>(
            src_data,
            &mut dst.data,
            forward,
            w,
            h,
            row_len,
            gain,
            threads,
        ),
        _ => sharpen_rows::<4>(
            src_data,
            &mut dst.data,
            forward,
            w,
            h,
            row_len,
            gain,
            threads,
        ),
    }
    dst
}

/// True when the two signed gradients around a pixel point opposite ways and
/// the opposite one is at least half as strong, which is what a feature only a
/// few pixels wide looks like. The magnitude test rejects noise on the
/// perpendicular axis of a long straight edge.
#[inline]
fn flip(a: f32, b: f32) -> bool {
    a * b < 0.0 && b.abs() * 4.0 >= a.abs() && b.abs() * 4.0 >= SHARP_THRESHOLD
}

/// True when the gradient reverses within +-3 px along either axis, i.e. the
/// pixel sits on the flank of a feature a few pixels wide (a line) rather than
/// on a single step edge. Only lines are worth sharpening: restoring the
/// amplitude of a thin feature gains several dB, while ringing a step edge
/// loses more than it gains.
#[inline]
#[allow(clippy::too_many_arguments)]
fn thin_feature4(
    src: &[u8],
    row_len: usize,
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    o_l: usize,
    o_r: usize,
    o_u: usize,
    o_d: usize,
) -> bool {
    let o_xm3 = y * row_len + x.saturating_sub(3) * 4;
    let o_xp3 = y * row_len + (x + 3).min(w - 1) * 4;
    let o_ym3 = y.saturating_sub(3) * row_len + x * 4;
    let o_yp3 = (y + 3).min(h - 1) * row_len + x * 4;
    let l = load_bytes4(&src[o_l..o_l + 4]);
    let r = load_bytes4(&src[o_r..o_r + 4]);
    let u = load_bytes4(&src[o_u..o_u + 4]);
    let d = load_bytes4(&src[o_d..o_d + 4]);
    let xm3 = load_bytes4(&src[o_xm3..o_xm3 + 4]);
    let xp3 = load_bytes4(&src[o_xp3..o_xp3 + 4]);
    let ym3 = load_bytes4(&src[o_ym3..o_ym3 + 4]);
    let yp3 = load_bytes4(&src[o_yp3..o_yp3 + 4]);
    let zero = f32x4::splat(0.0);
    let one = f32x4::splat(1.0);
    let gx = r - l;
    let gy = d - u;
    let four = f32x4::splat(4.0);
    let floor = f32x4::splat(SHARP_THRESHOLD);
    let mut flips = zero;
    for (a, b) in [(gx, l - xm3), (gx, xp3 - r), (gy, u - ym3), (gy, yp3 - d)] {
        let opposite = (a * b).simd_lt(zero).select(one, zero);
        let mb = b.abs() * four;
        let strong = (mb - a.abs()).simd_ge(zero).select(one, zero);
        let loud = (mb - floor).simd_ge(zero).select(one, zero);
        flips += opposite * strong * loud;
    }
    flips.simd_gt(zero).any()
}

/// Row-band worker of `sharpen_frame`; bands of `SYNTH_BAND_ROWS` rows depend
/// only on the frame height, so the result is independent of the worker count.
#[allow(clippy::too_many_arguments)]
fn sharpen_rows<const BPP: usize>(
    src: &[u8],
    dst: &mut [u8],
    forward: &[Vec2],
    w: usize,
    h: usize,
    row_len: usize,
    gain: f32,
    threads: usize,
) {
    let threshold = f32x4::splat(SHARP_THRESHOLD);
    let half = f32x4::splat(0.5);
    let two = f32x4::splat(2.0);
    let gain4 = f32x4::splat(gain);
    parallel_row_bands(dst, row_len, SYNTH_BAND_ROWS, threads, |band, first_row| {
        let rows = band.len() / row_len;
        for r in 0..rows {
            let y = first_row + r;
            let up = y.saturating_sub(1) * row_len;
            let dn = (y + 1).min(h - 1) * row_len;
            let bo = r * row_len;
            for x in 0..w {
                let xl = x.saturating_sub(1);
                let xr = (x + 1).min(w - 1);
                let o = y * row_len + x * BPP;
                let o_l = y * row_len + xl * BPP;
                let o_r = y * row_len + xr * BPP;
                let o_u = up + x * BPP;
                let o_d = dn + x * BPP;
                let d = bo + x * BPP;
                // Amount of interpolation blur at this pixel: bilinear sampling
                // with a whole-pixel displacement is an exact copy, so the
                // correction fades in with the displacement magnitude.
                let mv = forward[y * w + x];
                let weight = (mv.x.abs() + mv.y.abs()).min(1.0);
                if weight <= 0.0 {
                    band[d..d + BPP].copy_from_slice(&src[o..o + BPP]);
                    continue;
                }
                let gain4 = gain4 * f32x4::splat(weight);
                if BPP == 4 {
                    let c = load_bytes4(&src[o..o + 4]);
                    let l = load_bytes4(&src[o_l..o_l + 4]);
                    let rr = load_bytes4(&src[o_r..o_r + 4]);
                    let u = load_bytes4(&src[o_u..o_u + 4]);
                    let dn4 = load_bytes4(&src[o_d..o_d + 4]);
                    let grad = (rr - l).abs() + (dn4 - u).abs();
                    let mask = grad.simd_gt(threshold);
                    if mask.none() {
                        write_u8x4(&mut band[d..d + 4], c);
                        continue;
                    }
                    if !thin_feature4(src, row_len, w, h, x, y, o_l, o_r, o_u, o_d) {
                        write_u8x4(&mut band[d..d + 4], c);
                        continue;
                    }
                    let lap = c * two - (l + rr + u + dn4) * half;
                    let lo = l.min(rr).min(u).min(dn4).min(c);
                    let hi = l.max(rr).max(u).max(dn4).max(c);
                    let sharp = (c + lap * gain4).max(lo).min(hi);
                    write_u8x4(&mut band[d..d + 4], mask.select(sharp, c));
                } else {
                    let o_xm3 = y * row_len + x.saturating_sub(3) * BPP;
                    let o_xp3 = y * row_len + (x + 3).min(w - 1) * BPP;
                    let o_ym3 = y.saturating_sub(3) * row_len + x * BPP;
                    let o_yp3 = (y + 3).min(h - 1) * row_len + x * BPP;
                    for ch in 0..BPP {
                        let c = f32::from(src[o + ch]);
                        let l = f32::from(src[o_l + ch]);
                        let rr = f32::from(src[o_r + ch]);
                        let u = f32::from(src[o_u + ch]);
                        let dn1 = f32::from(src[o_d + ch]);
                        let grad = (rr - l).abs() + (dn1 - u).abs();
                        let thin = if grad > SHARP_THRESHOLD {
                            let gx = rr - l;
                            let gy = dn1 - u;
                            flip(gx, l - f32::from(src[o_xm3 + ch]))
                                || flip(gx, f32::from(src[o_xp3 + ch]) - rr)
                                || flip(gy, u - f32::from(src[o_ym3 + ch]))
                                || flip(gy, f32::from(src[o_yp3 + ch]) - dn1)
                        } else {
                            false
                        };
                        let v = if thin {
                            // Clamp to the local neighbourhood range so a hard
                            // step edge cannot gain a halo ring.
                            let lo = l.min(rr).min(u).min(dn1).min(c);
                            let hi = l.max(rr).max(u).max(dn1).max(c);
                            (c + gain * weight * (c * 2.0 - (l + rr + u + dn1) * 0.5)).clamp(lo, hi)
                        } else {
                            c
                        };
                        band[d + ch] = round_u8(v);
                    }
                }
            }
        }
    });
}

/// Upper bound on worker threads used by one frame-sized stage.
const MAX_WORKER_THREADS: usize = 16;

/// Rows per parallel band for the synthesiser. Fixed so the aggregate metrics
/// are bit-identical for every worker count.
const SYNTH_BAND_ROWS: usize = 16;

/// Escape hatch that disables the edge-preserving unsharp pass, for A/B
/// measurements and for feeds where sharpening is not wanted.
fn sharpening_disabled() -> bool {
    std::env::var("FLUID_NO_SHARP")
        .ok()
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

/// Worker count requested through the FLUID_THREADS environment variable.
///
/// Mirrors the upscaler convention: a missing, unparsable or zero value means
/// "use every available core".
fn env_thread_limit() -> Option<usize> {
    std::env::var("FLUID_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
}

/// Worker count for a frame-sized stage. Tiny images stay serial because the
/// fork/join overhead would dominate the work.
fn worker_threads(pixels: usize) -> usize {
    if pixels < 65_536 {
        return 1;
    }
    let avail = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let limit = env_thread_limit().unwrap_or(avail);
    limit.min(avail).clamp(1, MAX_WORKER_THREADS)
}

/// Rows per band: aim for roughly four bands per worker so hybrid CPUs with
/// fast and slow cores stay balanced.
fn band_rows(total_rows: usize, threads: usize) -> usize {
    total_rows.div_ceil(threads.max(1) * 4).max(1)
}

/// Apply `f(band, first_index)` to every fixed-size band of `data`, spreading
/// the bands over `threads` workers through a small local queue.
///
/// Band boundaries depend only on `band_len` and `data.len()`, never on the
/// thread count, and each band writes only its own elements, so the output is
/// independent of the worker count.
fn parallel_bands<T: Send>(
    data: &mut [T],
    band_len: usize,
    threads: usize,
    f: impl Fn(&mut [T], usize) + Sync,
) {
    if data.is_empty() {
        return;
    }
    let band_len = band_len.max(1);
    let mut jobs: Vec<(usize, &mut [T])> = Vec::new();
    let mut rest: &mut [T] = data;
    let mut first = 0usize;
    while !rest.is_empty() {
        let take = band_len.min(rest.len());
        let (band, tail) = rest.split_at_mut(take);
        rest = tail;
        jobs.push((first, band));
        first += take;
    }
    let threads = threads.clamp(1, jobs.len());
    if threads == 1 {
        for (first, band) in jobs {
            f(band, first);
        }
        return;
    }
    let queue = std::sync::Mutex::new(jobs);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let queue = &queue;
            let f = &f;
            scope.spawn(move || {
                loop {
                    let job = queue.lock().expect("band queue poisoned").pop();
                    let Some((first, band)) = job else {
                        break;
                    };
                    f(band, first);
                }
            });
        }
    });
}

/// Row-major wrapper around `parallel_bands`: `f(rows, first_row)`.
fn parallel_row_bands<T: Send>(
    data: &mut [T],
    row_len: usize,
    rows_per_band: usize,
    threads: usize,
    f: impl Fn(&mut [T], usize) + Sync,
) {
    let row_len = row_len.max(1);
    debug_assert_eq!(data.len() % row_len, 0);
    let band_len = rows_per_band.max(1) * row_len;
    parallel_bands(data, band_len, threads, move |band, first| {
        f(band, first / row_len)
    });
}

/// Like `parallel_bands` but walks a stats slice and a pixel buffer in lockstep:
/// `f(stats_of_band, pixel_bytes_of_band, first_byte)`.
fn parallel_pairs<S: Send, T: Send>(
    stats: &mut [S],
    data: &mut [T],
    band_len: usize,
    threads: usize,
    f: impl Fn(&mut S, &mut [T], usize) + Sync,
) {
    if data.is_empty() {
        return;
    }
    let band_len = band_len.max(1);
    let mut jobs: Vec<(&mut S, &mut [T], usize)> = Vec::new();
    let mut slots: &mut [S] = stats;
    let mut rest: &mut [T] = data;
    let mut first = 0usize;
    while !rest.is_empty() {
        let take = band_len.min(rest.len());
        let (band, tail) = rest.split_at_mut(take);
        rest = tail;
        let (slot, slots_tail) = slots.split_first_mut().expect("stats slice too short");
        slots = slots_tail;
        jobs.push((slot, band, first));
        first += take;
    }
    let threads = threads.clamp(1, jobs.len());
    if threads == 1 {
        for (slot, band, first) in jobs {
            f(slot, band, first);
        }
        return;
    }
    let queue = std::sync::Mutex::new(jobs);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let queue = &queue;
            let f = &f;
            scope.spawn(move || {
                loop {
                    let job = queue.lock().expect("band queue poisoned").pop();
                    let Some((slot, band, first)) = job else {
                        break;
                    };
                    f(slot, band, first);
                }
            });
        }
    });
}

/// Accumulators for one synthesis band, reduced in band order afterwards.
#[derive(Clone, Copy, Debug, Default)]
struct BandStats {
    blended: u64,
    occluded: u64,
    sum_conf: f64,
    sum_flow: f64,
}

/// Precomputed trigonometry of the global rotation, evaluated per output
/// pixel by the rotation-aware synthesis path.
///
/// `ct`/`st` are the sine and cosine of the full angle, `ca`/`sa` of
/// `angle * alpha` and `cb`/`sb` of `angle * (1 - alpha)`. Rotating the sample
/// positions by the temporal fraction puts the two gather points on the
/// geodesic between the endpoints, so alpha = 0 samples `prev` exactly and
/// alpha = 1 samples `next` exactly.
#[derive(Clone, Copy, Debug)]
struct RotKernel {
    cx: f32,
    cy: f32,
    ct: f32,
    st: f32,
    ca: f32,
    sa: f32,
    cb: f32,
    sb: f32,
}

/// Immutable inputs shared by every synthesis band.
#[derive(Clone, Copy)]
struct SynthCtx<'a> {
    prev: &'a [u8],
    next: &'a [u8],
    w: usize,
    h: usize,
    forward: &'a [Vec2],
    confidence: &'a [f32],
    alpha: f32,
    one_minus_alpha: f32,
    blend: f32,
    min_conf: f32,
    occlusion_fill: bool,
    /// Set when the flow estimate carries a significant global rotation.
    rot: Option<RotKernel>,
}

/// Synthesise a run of full rows of one pixel format.
fn synth_band<const BPP: usize>(
    band: &mut [u8],
    first_row: usize,
    stats: &mut BandStats,
    ctx: &SynthCtx<'_>,
) {
    let w = ctx.w;
    let h = ctx.h;
    let row_bytes = w * BPP;
    let alpha = ctx.alpha;
    let rows = band.len() / row_bytes;
    if ctx.rot.is_some() {
        if BPP == 4 {
            synth_band_rot4(band, first_row, stats, ctx);
        } else {
            synth_band_rot::<BPP>(band, first_row, stats, ctx);
        }
        return;
    }
    if BPP == 4 {
        synth_band4(band, first_row, stats, ctx);
        return;
    }
    for r in 0..rows {
        let y = first_row + r;
        let row = &mut band[r * row_bytes..(r + 1) * row_bytes];
        for x in 0..w {
            let i = y * w + x;
            let v = ctx.forward[i];
            let c = ctx.confidence.get(i).copied().unwrap_or(0.0);
            stats.sum_conf += f64::from(c);
            stats.sum_flow += f64::from(v.len());
            let o = x * BPP;
            let occluded = c < ctx.min_conf;
            if occluded {
                stats.occluded += 1;
            } else {
                stats.blended += 1;
            }
            if occluded && !ctx.occlusion_fill {
                // Disoccluded pixel with no fill requested: plain temporal blend.
                for ch in 0..BPP {
                    let a = f32::from(ctx.prev[i * BPP + ch]);
                    let b = f32::from(ctx.next[i * BPP + ch]);
                    row[o + ch] = round_u8(a + (b - a) * alpha);
                }
            } else {
                // Motion-compensated sample of both frames along the field.
                //
                // A disoccluded pixel's own motion is not reliable, but the
                // energy-minimising field around it still follows the surface behind
                // it, so this sample is a better fill than a raw nearer-frame copy: it
                // does not switch hard at alpha = 0.5 and it does not duplicate the
                // occluder into the hole.
                let xp = x as f32 + alpha * v.x;
                let yp = y as f32 + alpha * v.y;
                let pa = sample_px::<BPP>(ctx.prev, w, h, xp, yp);
                let xn = x as f32 - ctx.one_minus_alpha * v.x;
                let yn = y as f32 - ctx.one_minus_alpha * v.y;
                let pb = sample_px::<BPP>(ctx.next, w, h, xn, yn);
                for ch in 0..BPP {
                    let mc = pa[ch] + (pb[ch] - pa[ch]) * alpha;
                    let value = if ctx.blend >= 1.0 {
                        mc
                    } else {
                        let ca = f32::from(ctx.prev[i * BPP + ch]);
                        let cb = f32::from(ctx.next[i * BPP + ch]);
                        let cf = ca + (cb - ca) * alpha;
                        cf + (mc - cf) * ctx.blend
                    };
                    row[o + ch] = round_u8(value);
                }
            }
        }
    }
}

/// Rotation-aware synthesis for a run of full rows.
///
/// The flow field already contains the rotation, but blending two linear
/// displacements between frames that differ by a rotation cuts the corner of
/// the motion arc. Here the rotation is applied analytically with the temporal
/// fraction and only the residual (non-rotational) flow is lerped linearly:
/// prev is sampled at c + R(-angle*alpha) v - 0.5 + alpha*r and next at
/// c + R(angle*(1-alpha)) v - 0.5 + (alpha-1)*r, so both gather points land on
/// the same arc position at every alpha.
fn synth_band_rot<const BPP: usize>(
    band: &mut [u8],
    first_row: usize,
    stats: &mut BandStats,
    ctx: &SynthCtx<'_>,
) {
    let w = ctx.w;
    let h = ctx.h;
    let row_bytes = w * BPP;
    let alpha = ctx.alpha;
    let k = match ctx.rot {
        Some(k) => k,
        None => {
            synth_band::<BPP>(band, first_row, stats, ctx);
            return;
        }
    };
    let rows = band.len() / row_bytes;
    for r in 0..rows {
        let y = first_row + r;
        let row = &mut band[r * row_bytes..(r + 1) * row_bytes];
        let vy = y as f32 + 0.5 - k.cy;
        for x in 0..w {
            let i = y * w + x;
            let v = ctx.forward[i];
            let c = ctx.confidence.get(i).copied().unwrap_or(0.0);
            stats.sum_conf += f64::from(c);
            stats.sum_flow += f64::from(v.len());
            let o = x * BPP;
            let occluded = c < ctx.min_conf;
            if occluded {
                stats.occluded += 1;
            } else {
                stats.blended += 1;
            }
            if occluded && !ctx.occlusion_fill {
                for ch in 0..BPP {
                    let a = f32::from(ctx.prev[i * BPP + ch]);
                    let b = f32::from(ctx.next[i * BPP + ch]);
                    row[o + ch] = round_u8(a + (b - a) * alpha);
                }
                continue;
            }
            let vx = x as f32 + 0.5 - k.cx;
            // Displacement the rotation alone explains at this pixel.
            let frx = k.ct * vx + k.st * vy - vx;
            let fry = -k.st * vx + k.ct * vy - vy;
            let rx = v.x - frx;
            let ry = v.y - fry;
            let xp = k.cx + k.ca * vx + k.sa * vy - 0.5 + alpha * rx;
            let yp = k.cy - k.sa * vx + k.ca * vy - 0.5 + alpha * ry;
            let pa = sample_px::<BPP>(ctx.prev, w, h, xp, yp);
            let xn = k.cx + k.cb * vx - k.sb * vy - 0.5 + (alpha - 1.0) * rx;
            let yn = k.cy + k.sb * vx + k.cb * vy - 0.5 + (alpha - 1.0) * ry;
            let pb = sample_px::<BPP>(ctx.next, w, h, xn, yn);
            for ch in 0..BPP {
                let mc = pa[ch] + (pb[ch] - pa[ch]) * alpha;
                let value = if ctx.blend >= 1.0 {
                    mc
                } else {
                    let ca = f32::from(ctx.prev[i * BPP + ch]);
                    let cb = f32::from(ctx.next[i * BPP + ch]);
                    let cf = ca + (cb - ca) * alpha;
                    cf + (mc - cf) * ctx.blend
                };
                row[o + ch] = round_u8(value);
            }
        }
    }
}

/// Four-channel rotation-aware synthesis: the rotation trigonometry is shared
/// by all four lanes, so the two geodesic gathers run through the same
/// `f32x4` bilinear kernel as the plain path. Per-lane arithmetic order matches
/// `synth_band_rot`, so the bytes are identical.
fn synth_band_rot4(band: &mut [u8], first_row: usize, stats: &mut BandStats, ctx: &SynthCtx<'_>) {
    const BPP: usize = 4;
    let w = ctx.w;
    let h = ctx.h;
    let row_bytes = w * BPP;
    let alpha = ctx.alpha;
    let alpha4 = f32x4::splat(alpha);
    let blend4 = f32x4::splat(ctx.blend);
    let k = match ctx.rot {
        Some(k) => k,
        None => {
            synth_band4(band, first_row, stats, ctx);
            return;
        }
    };
    let rows = band.len() / row_bytes;
    for r in 0..rows {
        let y = first_row + r;
        let row = &mut band[r * row_bytes..(r + 1) * row_bytes];
        let vy = y as f32 + 0.5 - k.cy;
        for x in 0..w {
            let i = y * w + x;
            let v = ctx.forward[i];
            let c = ctx.confidence.get(i).copied().unwrap_or(0.0);
            stats.sum_conf += f64::from(c);
            stats.sum_flow += f64::from(v.len());
            let o = x * BPP;
            let occluded = c < ctx.min_conf;
            if occluded {
                stats.occluded += 1;
            } else {
                stats.blended += 1;
            }
            if occluded && !ctx.occlusion_fill {
                let ca = load_bytes4(&ctx.prev[i * BPP..]);
                let cb = load_bytes4(&ctx.next[i * BPP..]);
                write_u8x4(&mut row[o..o + BPP], ca + (cb - ca) * alpha4);
                continue;
            }
            let vx = x as f32 + 0.5 - k.cx;
            let frx = k.ct * vx + k.st * vy - vx;
            let fry = -k.st * vx + k.ct * vy - vy;
            let rx = v.x - frx;
            let ry = v.y - fry;
            let xp = k.cx + k.ca * vx + k.sa * vy - 0.5 + alpha * rx;
            let yp = k.cy - k.sa * vx + k.ca * vy - 0.5 + alpha * ry;
            let pa = sample_px4(ctx.prev, w, h, xp, yp);
            let xn = k.cx + k.cb * vx - k.sb * vy - 0.5 + (alpha - 1.0) * rx;
            let yn = k.cy + k.sb * vx + k.cb * vy - 0.5 + (alpha - 1.0) * ry;
            let pb = sample_px4(ctx.next, w, h, xn, yn);
            let mc = pa + (pb - pa) * alpha4;
            let value = if ctx.blend >= 1.0 {
                mc
            } else {
                let ca = load_bytes4(&ctx.prev[i * BPP..]);
                let cb = load_bytes4(&ctx.next[i * BPP..]);
                let cf = ca + (cb - ca) * alpha4;
                cf + (mc - cf) * blend4
            };
            write_u8x4(&mut row[o..o + BPP], value);
        }
    }
}

/// Four-channel synthesis: the four colour channels of one pixel fill one
/// `f32x4`, so both bilinear gathers, the temporal blend and the optional
/// crossfade mix run four channels per instruction. Per-lane arithmetic order
/// matches the scalar path, so the produced bytes are identical.
fn synth_band4(band: &mut [u8], first_row: usize, stats: &mut BandStats, ctx: &SynthCtx<'_>) {
    const BPP: usize = 4;
    let w = ctx.w;
    let h = ctx.h;
    let row_bytes = w * BPP;
    let alpha = ctx.alpha;
    let alpha4 = f32x4::splat(alpha);
    let blend4 = f32x4::splat(ctx.blend);
    let rows = band.len() / row_bytes;
    for r in 0..rows {
        let y = first_row + r;
        let row = &mut band[r * row_bytes..(r + 1) * row_bytes];
        for x in 0..w {
            let i = y * w + x;
            let v = ctx.forward[i];
            let c = ctx.confidence.get(i).copied().unwrap_or(0.0);
            stats.sum_conf += f64::from(c);
            stats.sum_flow += f64::from(v.len());
            let o = x * BPP;
            let occluded = c < ctx.min_conf;
            if occluded {
                stats.occluded += 1;
            } else {
                stats.blended += 1;
            }
            if occluded && !ctx.occlusion_fill {
                // Disoccluded pixel with no fill requested: plain temporal blend.
                let ca = load_bytes4(&ctx.prev[i * BPP..]);
                let cb = load_bytes4(&ctx.next[i * BPP..]);
                write_u8x4(&mut row[o..o + BPP], ca + (cb - ca) * alpha4);
            } else {
                // Motion-compensated sample of both frames along the field; see
                // `synth_band` for why this is the disocclusion fill.
                let xp = x as f32 + alpha * v.x;
                let yp = y as f32 + alpha * v.y;
                let pa = sample_px4(ctx.prev, w, h, xp, yp);
                let xn = x as f32 - ctx.one_minus_alpha * v.x;
                let yn = y as f32 - ctx.one_minus_alpha * v.y;
                let pb = sample_px4(ctx.next, w, h, xn, yn);
                let mc = pa + (pb - pa) * alpha4;
                let value = if ctx.blend >= 1.0 {
                    mc
                } else {
                    let ca = load_bytes4(&ctx.prev[i * BPP..]);
                    let cb = load_bytes4(&ctx.next[i * BPP..]);
                    let cf = ca + (cb - ca) * alpha4;
                    cf + (mc - cf) * blend4
                };
                write_u8x4(&mut row[o..o + BPP], value);
            }
        }
    }
}
/// Whole-frame synthesis for a compile-time pixel size.
fn synth_all<const BPP: usize>(
    stats: &mut [BandStats],
    data: &mut [u8],
    rows_per_band: usize,
    threads: usize,
    ctx: &SynthCtx<'_>,
) {
    let row_bytes = ctx.w * BPP;
    debug_assert_eq!(data.len(), ctx.w * ctx.h * BPP);
    debug_assert_eq!(stats.len(), ctx.h.div_ceil(rows_per_band));
    parallel_pairs(
        stats,
        data,
        rows_per_band * row_bytes,
        threads,
        |st, band, first| {
            synth_band::<BPP>(band, first / row_bytes, st, ctx);
        },
    );
}

#[inline]
fn rgb_to_luma(r: u8, g: u8, b: u8) -> f32 {
    (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32) / 255.0
}

/// Row-parallel luma conversion, bit-identical to `Frame::to_luma`.
fn luma_plane_par(frame: &Frame, threads: usize, out: &mut LumaPlane) {
    let w = frame.dims.width as usize;
    let h = frame.dims.height as usize;
    if out.dims != frame.dims {
        *out = LumaPlane::zeros(frame.dims);
    }
    let rows_per_band = band_rows(h, threads);
    match frame.fmt {
        fluid_core::PixelFormat::Rgba8 => {
            fill_luma::<4>(&frame.data, w, 0, 1, 2, out, rows_per_band, threads);
        }
        fluid_core::PixelFormat::Bgra8 => {
            fill_luma::<4>(&frame.data, w, 2, 1, 0, out, rows_per_band, threads);
        }
        fluid_core::PixelFormat::Rgb8 => {
            fill_luma::<3>(&frame.data, w, 0, 1, 2, out, rows_per_band, threads);
        }
        fluid_core::PixelFormat::Luma8 => {
            parallel_row_bands(&mut out.data, w, rows_per_band, threads, |band, first_y| {
                let rows = band.len() / w;
                for r in 0..rows {
                    let base = (first_y + r) * w;
                    let row = &mut band[r * w..(r + 1) * w];
                    for (x, slot) in row.iter_mut().enumerate() {
                        *slot = f32::from(frame.data[base + x]) / 255.0;
                    }
                }
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_luma<const BPP: usize>(
    src: &[u8],
    w: usize,
    r_off: usize,
    g_off: usize,
    b_off: usize,
    out: &mut LumaPlane,
    rows_per_band: usize,
    threads: usize,
) {
    let c1 = f32x4::splat(0.2126);
    let c2 = f32x4::splat(0.7152);
    let c3 = f32x4::splat(0.0722);
    let inv255 = f32x4::splat(255.0);
    parallel_row_bands(&mut out.data, w, rows_per_band, threads, |band, first_y| {
        let rows = band.len() / w;
        // Four pixels per iteration: the channel gathers fill three f32x4
        // registers and the weights, the /255 and the store stay vectorised.
        // The expression order matches rgb_to_luma, so the plane is
        // bit-identical to Frame::to_luma.
        let mut cr = [0.0f32; 4];
        let mut cg = [0.0f32; 4];
        let mut cb = [0.0f32; 4];
        for r in 0..rows {
            let y = first_y + r;
            let row = &mut band[r * w..(r + 1) * w];
            let base = y * w;
            let mut x = 0usize;
            while x + 4 <= w {
                for lane in 0..4 {
                    let i = (base + x + lane) * BPP;
                    cr[lane] = f32::from(src[i + r_off]);
                    cg[lane] = f32::from(src[i + g_off]);
                    cb[lane] = f32::from(src[i + b_off]);
                }
                let luma = (f32x4::from(cr) * c1 + f32x4::from(cg) * c2) + f32x4::from(cb) * c3;
                row[x..x + 4].copy_from_slice(&(luma / inv255).to_array());
                x += 4;
            }
            while x < w {
                let i = (base + x) * BPP;
                row[x] = rgb_to_luma(src[i + r_off], src[i + g_off], src[i + b_off]);
                x += 1;
            }
        }
    });
}

/// Stateful frame generator.
#[derive(Clone, Debug)]
pub struct Interpolator {
    params: InterpParams,
    multiplier: u32,
    detector: SceneCutDetector,
    last: InterpMetrics,
    /// Reusable luma planes for `generate` so steady-state calls do not allocate.
    prev_luma: LumaPlane,
    next_luma: LumaPlane,
}

impl Interpolator {
    /// Build from config; invalid multiplier falls back to 2.
    pub fn new(cfg: &InterpolatorConfig) -> Self {
        let multiplier = u32::from(cfg.multiplier.clamp(1, 8));
        Self {
            params: InterpParams::from(cfg),
            multiplier,
            detector: SceneCutDetector::new(cfg.scene_cut_threshold),
            last: InterpMetrics::default(),
            prev_luma: LumaPlane::zeros(Dims::new(1, 1)),
            next_luma: LumaPlane::zeros(Dims::new(1, 1)),
        }
    }

    pub fn params(&self) -> InterpParams {
        self.params
    }

    /// 1 disables generation, 2 inserts one frame, ...
    pub fn multiplier(&self) -> u32 {
        self.multiplier
    }

    pub fn set_multiplier(&mut self, m: u32) -> Result<()> {
        if !(1..=8).contains(&m) {
            return Err(FluidError::invalid(format!(
                "interpolation multiplier {m} out of range 1..=8"
            )));
        }
        self.multiplier = m;
        Ok(())
    }

    pub fn last_metrics(&self) -> InterpMetrics {
        self.last
    }

    /// Synthesise one frame at position `alpha` in (0,1) using a precomputed flow.
    pub fn interpolate(
        &self,
        prev: &Frame,
        next: &Frame,
        alpha: f32,
        flow: &FlowEstimate,
    ) -> Result<(Frame, InterpMetrics)> {
        let threads = worker_threads(prev.dims.area() as usize);
        self.interpolate_with(prev, next, alpha, flow, threads)
    }

    /// Parallel implementation of `interpolate`. Bands are fixed at
    /// `SYNTH_BAND_ROWS` rows, so pixels and metrics are bit-identical for every
    /// worker count; the reference single-threaded path is `threads == 1`.
    fn interpolate_with(
        &self,
        prev: &Frame,
        next: &Frame,
        alpha: f32,
        flow: &FlowEstimate,
        threads: usize,
    ) -> Result<(Frame, InterpMetrics)> {
        if !prev.same_shape(next) {
            return Err(FluidError::invalid(format!(
                "interpolate frame mismatch: {} {} vs {} {}",
                prev.dims, prev.fmt, next.dims, next.fmt
            )));
        }
        if flow.dims() != prev.dims {
            return Err(FluidError::invalid(format!(
                "interpolate flow mismatch: {} vs frame {}",
                flow.dims(),
                prev.dims
            )));
        }
        let alpha = if alpha.is_nan() {
            0.0
        } else {
            alpha.clamp(0.0, 1.0)
        };
        let w = prev.dims.width as usize;
        let h = prev.dims.height as usize;
        let bpp = prev.fmt.bytes_per_pixel();
        let blend = if self.params.blend.is_nan() {
            1.0
        } else {
            self.params.blend.clamp(0.0, 1.0)
        };
        let min_conf = FlowParams::default().confidence_min;
        let mut out = Frame::zeroed(prev.dims, prev.fmt);
        out.index = prev.index;
        out.pts_ns = lerp_i64(prev.pts_ns, next.pts_ns, alpha);
        // A global rotation needs the geodesic path, not a linear lerp of two
        // rotated displacements; precompute its trigonometry once per frame.
        let rot = flow.rotation.filter(|r| r.is_significant()).map(|r| {
            let (st, ct) = r.cos_sin();
            let (sa, ca) = (r.angle * alpha).sin_cos();
            let (sb, cb) = (r.angle * (1.0 - alpha)).sin_cos();
            RotKernel {
                cx: r.center_x,
                cy: r.center_y,
                ct,
                st,
                ca,
                sa,
                cb,
                sb,
            }
        });
        let ctx = SynthCtx {
            prev: &prev.data,
            next: &next.data,
            w,
            h,
            forward: flow.forward.vectors(),
            confidence: &flow.confidence,
            alpha,
            one_minus_alpha: 1.0 - alpha,
            blend,
            min_conf,
            occlusion_fill: self.params.occlusion_fill,
            rot,
        };
        let rows_per_band = SYNTH_BAND_ROWS;
        let band_count = h.div_ceil(rows_per_band);
        let mut stats = vec![BandStats::default(); band_count];
        match bpp {
            1 => synth_all::<1>(&mut stats, &mut out.data, rows_per_band, threads, &ctx),
            3 => synth_all::<3>(&mut stats, &mut out.data, rows_per_band, threads, &ctx),
            _ => synth_all::<4>(&mut stats, &mut out.data, rows_per_band, threads, &ctx),
        }
        // Reduce in band order: this is the same raster order as the serial
        // path, so the floating point sums are identical for any worker count.
        let mut blended = 0u64;
        let mut occluded = 0u64;
        let mut sum_conf = 0.0f64;
        let mut sum_flow = 0.0f64;
        for band in &stats {
            blended += band.blended;
            occluded += band.occluded;
            sum_conf += band.sum_conf;
            sum_flow += band.sum_flow;
        }
        let count = (w * h) as f64;
        let metrics = InterpMetrics {
            frames: 1,
            blended_px: blended,
            occluded_px: occluded,
            mean_confidence: (sum_conf / count) as f32,
            mean_flow_px: (sum_flow / count) as f32,
            scene_cut: false,
        };
        if blend <= 0.0 {
            return Ok((out, metrics));
        }
        let sharpened = sharpen_frame(&out, flow.forward.vectors(), alpha, threads);
        Ok((sharpened, metrics))
    }

    /// Blend two frames without motion (scene cuts, fallback, multiplier on video).
    pub fn crossfade(&self, prev: &Frame, next: &Frame, alpha: f32) -> Frame {
        let alpha = if alpha.is_nan() {
            0.0
        } else {
            alpha.clamp(0.0, 1.0)
        };
        if !prev.same_shape(next) {
            return prev.clone();
        }
        let mut out = Frame::zeroed(prev.dims, prev.fmt);
        out.index = prev.index;
        out.pts_ns = lerp_i64(prev.pts_ns, next.pts_ns, alpha);
        let w = prev.dims.width as usize;
        let h = prev.dims.height as usize;
        let row_len = w * prev.fmt.bytes_per_pixel();
        let threads = worker_threads(prev.dims.area() as usize);
        let rows_per_band = band_rows(h, threads);
        parallel_row_bands(
            &mut out.data,
            row_len,
            rows_per_band,
            threads,
            |band, first| {
                let base = first * row_len;
                let t = f32x4::splat(alpha);
                let mut k = 0usize;
                while k + 4 <= band.len() {
                    let i = base + k;
                    let mixed = lerp_bytes4(&prev.data[i..], &next.data[i..], t);
                    write_u8x4(&mut band[k..], mixed);
                    k += 4;
                }
                while k < band.len() {
                    let i = base + k;
                    let a = f32::from(prev.data[i]);
                    let b = f32::from(next.data[i]);
                    band[k] = round_u8(a + (b - a) * alpha);
                    k += 1;
                }
            },
        );
        out
    }

    /// Generate all intermediate frames between `prev` and `next@.
    ///
    /// Returns `multiplier - 1` frames ordered by increasing alpha, or an empty
    /// vector when the scenes cut (callers then repeat the source frames).
    pub fn generate(
        &mut self,
        prev: &Frame,
        next: &Frame,
        flow_cfg: &FlowConfig,
    ) -> Result<Vec<Frame>> {
        if !prev.same_shape(next) {
            return Err(FluidError::invalid(format!(
                "generate frame mismatch: {} {} vs {} {}",
                prev.dims, prev.fmt, next.dims, next.fmt
            )));
        }
        if self.multiplier <= 1 {
            self.last = InterpMetrics::default();
            return Ok(Vec::new());
        }
        let threads = worker_threads(prev.dims.area() as usize);
        luma_plane_par(prev, threads, &mut self.prev_luma);
        luma_plane_par(next, threads, &mut self.next_luma);
        if self.detector.is_cut(&self.prev_luma, &self.next_luma) {
            self.last = InterpMetrics {
                scene_cut: true,
                ..InterpMetrics::default()
            };
            return Ok(Vec::new());
        }
        let params = FlowParams::from(flow_cfg);
        let flow = fluid_flow::estimate(&self.prev_luma, &self.next_luma, &params)?;
        self.generate_with_flow(prev, next, &flow)
    }

    /// Generate intermediates using an already computed flow estimate.
    pub fn generate_with_flow(
        &mut self,
        prev: &Frame,
        next: &Frame,
        flow: &FlowEstimate,
    ) -> Result<Vec<Frame>> {
        if !prev.same_shape(next) {
            return Err(FluidError::invalid(format!(
                "generate frame mismatch: {} {} vs {} {}",
                prev.dims, prev.fmt, next.dims, next.fmt
            )));
        }
        if flow.dims() != prev.dims {
            return Err(FluidError::invalid(format!(
                "generate flow mismatch: {} vs frame {}",
                flow.dims(),
                prev.dims
            )));
        }
        let m = self.multiplier.max(1);
        if m <= 1 {
            self.last = InterpMetrics::default();
            return Ok(Vec::new());
        }
        let mut frames = Vec::with_capacity((m - 1) as usize);
        let mut total_conf = 0.0f64;
        let mut total_flow = 0.0f64;
        let mut total_occluded = 0u64;
        let mut total_blended = 0u64;
        for k in 1..m {
            let alpha = k as f32 / m as f32;
            let (frame, metrics) = self.interpolate(prev, next, alpha, flow)?;
            total_conf += f64::from(metrics.mean_confidence);
            total_flow += f64::from(metrics.mean_flow_px);
            total_occluded += metrics.occluded_px;
            total_blended += metrics.blended_px;
            frames.push(frame);
        }
        let n = f64::from(m - 1);
        self.last = InterpMetrics {
            frames: m - 1,
            blended_px: total_blended,
            occluded_px: total_occluded,
            mean_confidence: (total_conf / n) as f32,
            mean_flow_px: (total_flow / n) as f32,
            scene_cut: false,
        };
        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluid_core::{Dims, PixelFormat};
    use fluid_flow::{FlowField, GlobalRotation, Vec2};

    fn gray_frame(dims: Dims, fmt: PixelFormat, value: u8) -> Frame {
        let mut f = Frame::zeroed(dims, fmt);
        f.data.fill(value);
        f
    }

    fn zero_flow(dims: Dims, confidence: f32) -> FlowEstimate {
        let n = dims.area() as usize;
        FlowEstimate::new(
            FlowField::new(dims),
            FlowField::new(dims),
            vec![confidence; n],
        )
    }

    fn constant_flow(dims: Dims, v: Vec2, confidence: f32) -> FlowEstimate {
        let n = dims.area() as usize;
        let mut forward = FlowField::new(dims);
        forward.vectors_mut().fill(v);
        let mut backward = FlowField::new(dims);
        backward.vectors_mut().fill(v.scaled(-1.0));
        FlowEstimate::new(forward, backward, vec![confidence; n])
    }

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

    fn blob_value(uf: f32, vf: f32) -> f32 {
        let mut v = 0.12f32;
        for &(bx, by, sigma, amp) in BLOBS.iter() {
            let dx = uf - bx;
            let dy = vf - by;
            let r2 = (dx * dx + dy * dy) / (sigma * sigma);
            if r2 < 9.0 {
                v += amp * (-r2).exp();
            }
        }
        v.clamp(0.0, 1.0)
    }

    fn blob_frame(dims: Dims, dx: f32) -> Frame {
        let mut f = Frame::zeroed(dims, PixelFormat::Rgba8);
        let w = dims.width as usize;
        for y in 0..dims.height as usize {
            for x in 0..w {
                let uf = (x as f32 + 0.5 - dx) / dims.width as f32;
                let vf = (y as f32 + 0.5) / dims.height as f32;
                let v = (blob_value(uf, vf) * 255.0 + 0.5) as u8;
                let o = (y * w + x) * 4;
                f.data[o] = v;
                f.data[o + 1] = v;
                f.data[o + 2] = v;
                f.data[o + 3] = 255;
            }
        }
        f
    }

    fn expected_gray(dims: Dims, x: f32, y: f32, dx: f32) -> f32 {
        let uf = (x + 0.5 - dx) / dims.width as f32;
        let vf = (y + 0.5) / dims.height as f32;
        blob_value(uf, vf) * 255.0
    }

    #[test]
    fn crossfade_midpoint_is_average() {
        let d = Dims::new(4, 4);
        let a = Frame::new(d, PixelFormat::Rgba8, vec![0u8; d.area() as usize * 4]).unwrap();
        let b = Frame::new(d, PixelFormat::Rgba8, vec![200u8; d.area() as usize * 4]).unwrap();
        let it = Interpolator::new(&InterpolatorConfig::default());
        let out = it.crossfade(&a, &b, 0.5);
        assert_eq!(out.pixel(1, 1)[0], 100);
    }

    #[test]
    fn interpolation_preserves_pixel_format() {
        let d = Dims::new(8, 6);
        let it = Interpolator::new(&InterpolatorConfig::default());
        for fmt in [
            PixelFormat::Rgba8,
            PixelFormat::Bgra8,
            PixelFormat::Rgb8,
            PixelFormat::Luma8,
        ] {
            let a = gray_frame(d, fmt, 10);
            let b = gray_frame(d, fmt, 200);
            let flow = zero_flow(d, 1.0);
            let (out, m) = it.interpolate(&a, &b, 0.5, &flow).unwrap();
            assert_eq!(out.fmt, fmt);
            assert_eq!(out.data.len(), fmt.buffer_len(d));
            assert_eq!(out.pixel(3, 2).len(), fmt.bytes_per_pixel());
            assert_eq!(m.blended_px, d.area());
            assert_eq!(m.occluded_px, 0);
            assert_eq!(out.pixel(0, 0)[0], 105);
        }
    }

    #[test]
    fn motion_compensation_follows_translation() {
        let d = Dims::new(96, 64);
        let prev = blob_frame(d, 0.0);
        let next = blob_frame(d, 2.0);
        let cfg = InterpolatorConfig {
            multiplier: 2,
            ..InterpolatorConfig::default()
        };
        let mut it = Interpolator::new(&cfg);
        let frames = it.generate(&prev, &next, &FlowConfig::default()).unwrap();
        assert_eq!(frames.len(), 1);
        let mid = &frames[0];
        let mut err = 0.0f64;
        let mut n = 0.0f64;
        for y in 16..48 {
            for x in 16..80 {
                let got = f32::from(mid.pixel(x, y)[0]);
                let want = expected_gray(d, x as f32, y as f32, 1.0);
                err += f64::from((got - want).abs());
                n += 1.0;
            }
        }
        let mae = err / n;
        assert!(mae < 15.0, "motion compensated MAE {mae} too high");
    }

    /// Quality guard for the ground-truth harness (`examples/quality.rs`): a
    /// 2.5 px translation of a smooth blob texture must be synthesised almost
    /// exactly (measured 58.4 dB at 720p). A raw nearer-frame copy or an
    /// unrefined field lands well below the bound, so it guards the flow
    /// estimator and the disocclusion fill together.
    #[test]
    fn translation_quality_guard() {
        let d = Dims::new(1280, 720);
        let motion = 2.5f32;
        let prev = blob_frame(d, 0.0);
        let next = blob_frame(d, motion);
        let prev_l = prev.to_luma();
        let next_l = next.to_luma();
        let mut it = Interpolator::new(&InterpolatorConfig::default());
        let frames = it.generate(&prev, &next, &FlowConfig::default()).unwrap();
        assert_eq!(frames.len(), 1);
        let mid = frames[0].to_luma();
        let (mut mse, mut cross_mse, mut count) = (0.0f64, 0.0f64, 0.0f64);
        for y in 16..d.height - 16 {
            for x in 16..d.width - 16 {
                let truth = f64::from(expected_gray(d, x as f32, y as f32, motion * 0.5) / 255.0);
                mse += (f64::from(mid.get(x, y)) - truth).powi(2);
                let cross = 0.5 * (f64::from(prev_l.get(x, y)) + f64::from(next_l.get(x, y)));
                cross_mse += (cross - truth).powi(2);
                count += 1.0;
            }
        }
        let psnr = -10.0 * (mse / count).log10();
        let cross_psnr = -10.0 * (cross_mse / count).log10();
        println!("translation guard: psnr {psnr:.2} dB, crossfade {cross_psnr:.2} dB");
        // Raw nearer-frame copying and a plain crossfade both land 15+ dB lower on
        // this texture, so the bound detects a regression in flow or in the fill.
        assert!(
            psnr > 55.0,
            "motion-compensated PSNR too low: {psnr:.2} dB (crossfade {cross_psnr:.2} dB)"
        );
    }
    #[test]
    fn generate_produces_multiplier_minus_one_frames() {
        let d = Dims::new(64, 48);
        let prev = blob_frame(d, 0.0);
        let next = blob_frame(d, 2.0);
        let cfg = InterpolatorConfig {
            multiplier: 3,
            ..InterpolatorConfig::default()
        };
        let mut it = Interpolator::new(&cfg);
        let frames = it.generate(&prev, &next, &FlowConfig::default()).unwrap();
        assert_eq!(frames.len(), 2);
        let m = it.last_metrics();
        assert_eq!(m.frames, 2);
        assert!(!m.scene_cut);
        assert!(m.blended_px > 0);
        assert!(m.mean_flow_px > 1.0);
        assert_eq!(frames[0].dims, d);
        assert_eq!(it.multiplier(), 3);
        it.set_multiplier(4).unwrap();
        assert_eq!(it.multiplier(), 4);
        assert!(it.set_multiplier(0).is_err());
        assert!(it.set_multiplier(9).is_err());
    }

    #[test]
    fn occlusion_fill_uses_motion_compensated_samples() {
        // Occluded pixels are filled along the field, not by copying raw pixels
        // (which used to jump at alpha = 0.5 and duplicate the occluder) and not by a
        // plain crossfade. A quadratic ramp separates the three behaviours: a raw
        // copy and a crossfade of identical frames both give v(x), while the
        // motion-compensated mean gives (v(x+1) + v(x-1)) / 2 = v(x) + 2.
        let d = Dims::new(16, 4);
        let value = |x: u32| {
            let xf = x as f32;
            (2.0 * xf * xf).min(255.0) as u8
        };
        let mut a = Frame::zeroed(d, PixelFormat::Rgba8);
        let mut b = Frame::zeroed(d, PixelFormat::Rgba8);
        for y in 0..d.height {
            for x in 0..d.width {
                let v = value(x);
                let o = ((y * d.width + x) as usize) * 4;
                a.data[o..o + 4].copy_from_slice(&[v, v, v, 255]);
                b.data[o..o + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        let it = Interpolator::new(&InterpolatorConfig::default());
        // Forward field (2, 0) at alpha = 0.5 gathers prev(x + 1) and next(x - 1).
        let flow = constant_flow(d, Vec2::new(2.0, 0.0), 0.0);
        let (out, m) = it.interpolate(&a, &b, 0.5, &flow).unwrap();
        assert_eq!(m.occluded_px, d.area());
        assert_eq!(m.blended_px, 0);
        let mut different_from_raw_copy = 0;
        for x in 2..d.width - 2 {
            let want = round_u8(0.5 * (f32::from(value(x + 1)) + f32::from(value(x - 1))));
            assert_eq!(out.pixel(x, 1)[0], want, "pixel {x}");
            if want != value(x) {
                different_from_raw_copy += 1;
            }
        }
        assert!(
            different_from_raw_copy > 0,
            "fill did not move off the raw never-motion sample"
        );
        // Without the fill the occluded pixels fall back to a plain temporal blend.
        let cfg = InterpolatorConfig {
            occlusion_fill: false,
            ..InterpolatorConfig::default()
        };
        let plain = Interpolator::new(&cfg);
        let (out2, m2) = plain.interpolate(&a, &b, 0.5, &flow).unwrap();
        assert_eq!(m2.occluded_px, d.area());
        for x in 0..d.width {
            assert_eq!(out2.pixel(x, 1)[0], value(x), "pixel {x}");
        }
    }

    #[test]
    fn occlusion_fill_disabled_crossfades() {
        let d = Dims::new(8, 8);
        let a = gray_frame(d, PixelFormat::Rgba8, 10);
        let b = gray_frame(d, PixelFormat::Rgba8, 250);
        let cfg = InterpolatorConfig {
            occlusion_fill: false,
            ..InterpolatorConfig::default()
        };
        let it = Interpolator::new(&cfg);
        let flow = zero_flow(d, 0.0);
        let (out, m) = it.interpolate(&a, &b, 0.25, &flow).unwrap();
        assert_eq!(m.occluded_px, d.area());
        assert_eq!(out.pixel(1, 1)[0], 70);
    }

    #[test]
    fn blend_zero_matches_crossfade() {
        let d = Dims::new(16, 12);
        let a = blob_frame(d, 0.0);
        let b = blob_frame(d, 2.0);
        let cfg = InterpolatorConfig {
            blend: 0.0,
            ..InterpolatorConfig::default()
        };
        let it = Interpolator::new(&cfg);
        let flow = constant_flow(d, Vec2::new(2.0, 0.0), 1.0);
        let (out, m) = it.interpolate(&a, &b, 0.5, &flow).unwrap();
        assert_eq!(m.blended_px, d.area());
        let expected = it.crossfade(&a, &b, 0.5);
        assert_eq!(out.data, expected.data);
    }

    #[test]
    fn scene_cut_yields_no_frames() {
        let d = Dims::new(32, 32);
        let a = gray_frame(d, PixelFormat::Rgba8, 0);
        let b = gray_frame(d, PixelFormat::Rgba8, 255);
        let mut it = Interpolator::new(&InterpolatorConfig::default());
        let frames = it.generate(&a, &b, &FlowConfig::default()).unwrap();
        assert!(frames.is_empty());
        assert!(it.last_metrics().scene_cut);
        assert_eq!(it.last_metrics().frames, 0);
    }

    #[test]
    fn shape_mismatches_are_rejected() {
        let a = gray_frame(Dims::new(8, 8), PixelFormat::Rgba8, 0);
        let b = gray_frame(Dims::new(8, 4), PixelFormat::Rgba8, 0);
        let mut it = Interpolator::new(&InterpolatorConfig::default());
        let flow = zero_flow(Dims::new(8, 8), 1.0);
        assert!(it.interpolate(&a, &b, 0.5, &flow).is_err());
        let bad = zero_flow(Dims::new(4, 4), 1.0);
        assert!(it.interpolate(&a, &a, 0.5, &bad).is_err());
        assert!(it.generate(&a, &b, &FlowConfig::default()).is_err());
        assert!(it.generate_with_flow(&a, &a, &bad).is_err());
    }

    #[test]
    fn timestamps_and_metrics_are_reported() {
        let d = Dims::new(8, 8);
        let a = gray_frame(d, PixelFormat::Rgba8, 20)
            .with_pts(1000)
            .with_index(7);
        let b = gray_frame(d, PixelFormat::Rgba8, 220)
            .with_pts(2000)
            .with_index(8);
        let it = Interpolator::new(&InterpolatorConfig::default());
        let flow = zero_flow(d, 1.0);
        let (out, m) = it.interpolate(&a, &b, 0.5, &flow).unwrap();
        assert_eq!(out.index, 7);
        assert_eq!(out.pts_ns, 1500);
        assert_eq!(m.frames, 1);
        assert!((m.mean_confidence - 1.0).abs() < 1.0e-6);
        assert_eq!(m.mean_flow_px, 0.0);
        assert!(!m.scene_cut);
        assert!(m.to_string().contains("frames=1"));
    }

    #[test]
    fn identical_frames_interpolate_to_itself() {
        let d = Dims::new(24, 16);
        let a = blob_frame(d, 0.0);
        let mut it = Interpolator::new(&InterpolatorConfig::default());
        let frames = it.generate(&a, &a, &FlowConfig::default()).unwrap();
        assert_eq!(frames.len(), 1);
        for (got, want) in frames[0].data.iter().zip(a.data.iter()) {
            let diff = i32::from(*got) - i32::from(*want);
            assert!(diff.abs() <= 2, "identical frames differ by {diff}");
        }
    }

    #[test]
    fn multiplier_one_generates_nothing() {
        let d = Dims::new(8, 8);
        let a = gray_frame(d, PixelFormat::Rgba8, 0);
        let cfg = InterpolatorConfig {
            multiplier: 1,
            ..InterpolatorConfig::default()
        };
        let mut it = Interpolator::new(&cfg);
        let flow = zero_flow(d, 1.0);
        assert!(it.generate_with_flow(&a, &a, &flow).unwrap().is_empty());
        assert!(
            it.generate(&a, &a, &FlowConfig::default())
                .unwrap()
                .is_empty()
        );
    }

    /// Deterministic flow with a mix of confident and low-confidence pixels, used
    /// to exercise both synthesis branches in the parallel equivalence test.
    fn patterned_flow(d: Dims) -> FlowEstimate {
        let w = d.width as usize;
        let h = d.height as usize;
        let mut fwd = FlowField::new(d);
        let mut bwd = FlowField::new(d);
        let mut conf = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let v = Vec2::new(
                    1.5 + ((i % 5) as f32) * 0.25,
                    -0.75 + ((i % 3) as f32) * 0.5,
                );
                fwd.set(x as u32, y as u32, v);
                bwd.set(x as u32, y as u32, Vec2::new(-v.x, -v.y));
                conf[i] = if i.is_multiple_of(7) { 0.1 } else { 0.9 };
            }
        }
        FlowEstimate::new(fwd, bwd, conf)
    }

    #[test]
    fn parallel_synthesis_matches_serial() {
        let d = Dims::new(320, 240);
        let a = blob_frame(d, 0.0);
        let b = blob_frame(d, 2.0);
        let flow = patterned_flow(d);
        let it = Interpolator::new(&InterpolatorConfig::default());
        let (serial, m_serial) = it.interpolate_with(&a, &b, 0.5, &flow, 1).unwrap();
        assert!(
            m_serial.occluded_px > 0,
            "test data must exercise occlusion"
        );
        assert!(m_serial.blended_px > 0);
        for tid in [2usize, 4, 8, 16] {
            let (par, m) = it.interpolate_with(&a, &b, 0.5, &flow, tid).unwrap();
            assert_eq!(serial.data, par.data, "pixels differ at {tid} threads");
            assert_eq!(m_serial, m, "metrics differ at {tid} threads");
        }
        let (public, _) = it.interpolate(&a, &b, 0.5, &flow).unwrap();
        assert_eq!(serial.data, public.data);
    }

    #[test]
    fn luma_conversion_matches_core() {
        let d = Dims::new(64, 48);
        for fmt in [
            PixelFormat::Rgba8,
            PixelFormat::Bgra8,
            PixelFormat::Rgb8,
            PixelFormat::Luma8,
        ] {
            let mut frame = Frame::zeroed(d, fmt);
            for (i, byte) in frame.data.iter_mut().enumerate() {
                *byte = (i.wrapping_mul(37) % 251) as u8;
            }
            let want = frame.to_luma();
            let mut got = LumaPlane::zeros(Dims::new(1, 1));
            luma_plane_par(&frame, 8, &mut got);
            assert_eq!(got, want, "luma differs for {fmt}");
        }
    }

    #[test]
    #[ignore = "performance measurement"]
    fn perf_1080p_interp() {
        for d in [Dims::new(1920, 1080), Dims::new(1280, 720)] {
            let prev = blob_frame(d, 0.0);
            let next = blob_frame(d, 2.0);
            let lp = prev.to_luma();
            let ln = next.to_luma();
            let t = std::time::Instant::now();
            let flow = fluid_flow::estimate(&lp, &ln, &FlowParams::default()).unwrap();
            let flow_ms = t.elapsed().as_secs_f64() * 1000.0;
            let it = Interpolator::new(&InterpolatorConfig::default());
            let t = std::time::Instant::now();
            let (serial, _m) = it.interpolate_with(&prev, &next, 0.5, &flow, 1).unwrap();
            let serial_ms = t.elapsed().as_secs_f64() * 1000.0;
            let t = std::time::Instant::now();
            let (par, _m) = it.interpolate_with(&prev, &next, 0.5, &flow, 16).unwrap();
            let par_ms = t.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(serial.data, par.data);
            let mut it2 = Interpolator::new(&InterpolatorConfig::default());
            let t = std::time::Instant::now();
            let frames = it2.generate(&prev, &next, &FlowConfig::default()).unwrap();
            let total2 = t.elapsed().as_secs_f64() * 1000.0;
            let cfg3 = InterpolatorConfig {
                multiplier: 3,
                ..InterpolatorConfig::default()
            };
            let mut it3 = Interpolator::new(&cfg3);
            let t = std::time::Instant::now();
            let frames3 = it3.generate(&prev, &next, &FlowConfig::default()).unwrap();
            let total3 = t.elapsed().as_secs_f64() * 1000.0;
            println!(
                "interp {}x{} Rgba8: flow {flow_ms:.2} ms, synthesis serial {serial_ms:.2} ms / \
parallel {par_ms:.2} ms ({:.1}x), generate(x2) {total2:.2} ms ({} frame), \
generate(x3) {total3:.2} ms ({} frames)",
                d.width,
                d.height,
                serial_ms / par_ms.max(0.001),
                frames.len(),
                frames3.len()
            );
        }
    }

    /// Scalar reference of the edge-preserving unsharp pass: the four-channel
    /// kernel calls the "thin feature" test once per pixel and applies it to all
    /// channels, so this mirrors that pixel-level decision.
    fn reference_sharpen(src: &Frame, alpha: f32) -> Frame {
        let w = src.dims.width as usize;
        let h = src.dims.height as usize;
        let bpp = src.fmt.bytes_per_pixel();
        let row = w * bpp;
        let gain = SHARP_GAIN * 4.0 * alpha * (1.0 - alpha);
        let at =
            |x: usize, y: usize, c: usize| -> f32 { f32::from(src.data[y * row + x * bpp + c]) };
        let mut out = Frame::zeroed(src.dims, src.fmt);
        for y in 0..h {
            for x in 0..w {
                let xl = x.saturating_sub(1);
                let xr = (x + 1).min(w - 1);
                let yu = y.saturating_sub(1);
                let yd = (y + 1).min(h - 1);
                let thin = (0..bpp).any(|c| {
                    let gx = at(xr, y, c) - at(xl, y, c);
                    let gy = at(x, yd, c) - at(x, yu, c);
                    flip(gx, at(xl, y, c) - at(x.saturating_sub(3), y, c))
                        || flip(gx, at((x + 3).min(w - 1), y, c) - at(xr, y, c))
                        || flip(gy, at(x, yu, c) - at(x, y.saturating_sub(3), c))
                        || flip(gy, at(x, (y + 3).min(h - 1), c) - at(x, yd, c))
                });
                for c in 0..bpp {
                    let cv = at(x, y, c);
                    let (l, r) = (at(xl, y, c), at(xr, y, c));
                    let (u, dn) = (at(x, yu, c), at(x, yd, c));
                    let grad = (r - l).abs() + (dn - u).abs();
                    let v = if thin && grad > SHARP_THRESHOLD {
                        let lo = l.min(r).min(u).min(dn).min(cv);
                        let hi = l.max(r).max(u).max(dn).max(cv);
                        (cv + gain * (cv * 2.0 - (l + r + u + dn) * 0.5)).clamp(lo, hi)
                    } else {
                        cv
                    };
                    out.data[y * row + x * bpp + c] = round_u8(v);
                }
            }
        }
        out
    }

    /// Harness-style thin line: blob-texture background with a 3 px bright rule
    /// moving by `line_dx` pixels, sampled at temporal fraction `t`.
    fn textured_thin_line(d: Dims, line_dx: f32, t: f32) -> Frame {
        let w = d.width as usize;
        let xc = 0.3 * d.width as f32 + line_dx * t;
        let mut f = Frame::zeroed(d, PixelFormat::Rgba8);
        for y in 0..d.height as usize {
            for x in 0..w {
                let bg = blob_value(
                    (x as f32 + 0.5) / d.width as f32,
                    (y as f32 + 0.5) / d.height as f32,
                ) * 0.45
                    + 0.12;
                let cov = (1.5 - (x as f32 - xc).abs()).clamp(0.0, 1.0);
                let v = ((bg + (0.95 - bg) * cov) * 255.0 + 0.5) as u8;
                let o = (y * w + x) * 4;
                f.data[o] = v;
                f.data[o + 1] = v;
                f.data[o + 2] = v;
                f.data[o + 3] = 255;
            }
        }
        f
    }

    /// Where the flow is zero the gather is an exact copy, so the pass must not
    /// touch a single byte (this is what keeps static frames bit-identical), and
    /// the result must not depend on the worker count.
    #[test]
    fn sharpening_skips_static_pixels_and_is_deterministic() {
        let d = Dims::new(96, 32);
        let frame = blob_frame(d, 0.0);
        let area = d.area() as usize;
        let still: Vec<Vec2> = vec![Vec2::ZERO; area];
        let moving: Vec<Vec2> = vec![Vec2::new(3.0, 0.5); area];
        let threads = worker_threads(area);
        let untouched = sharpen_frame(&frame, &still, 0.5, threads);
        assert_eq!(untouched.data, frame.data, "static flow must not sharpen");
        let sharpened = sharpen_frame(&frame, &moving, 0.5, threads);
        assert_eq!(sharpened.data, sharpen_frame(&frame, &moving, 0.5, 1).data);
        assert_eq!(
            sharpened.data,
            sharpen_frame(&frame, &moving, 0.5, 8).data,
            "sharpening must not depend on the worker count"
        );
        assert_ne!(
            sharpened.data, frame.data,
            "moving pixels should be sharpened"
        );
    }

    /// Harness-like thin line: a 3 px bright rule over the blob texture. This is
    /// the scene that loses ~20 dB without the unsharp pass, so it guards the
    /// pass end to end (and would catch a regression in the flow estimate).
    #[test]
    fn thin_line_quality_guard() {
        let d = Dims::new(480, 270);
        let (line_dx, t) = (3.0f32, 0.5f32);
        let prev = textured_thin_line(d, line_dx, 0.0);
        let next = textured_thin_line(d, line_dx, 1.0);
        let truth = textured_thin_line(d, line_dx, t).to_luma();
        let prev_l = prev.to_luma();
        let next_l = next.to_luma();
        let mut it = Interpolator::new(&InterpolatorConfig::default());
        let frames = it.generate(&prev, &next, &FlowConfig::default()).unwrap();
        assert_eq!(frames.len(), 1);
        let mid = frames[0].to_luma();
        let (mut mse, mut cross_mse, mut count) = (0.0f64, 0.0f64, 0.0f64);
        for y in 8..d.height - 8 {
            for x in 8..d.width - 8 {
                let want = f64::from(truth.get(x, y));
                mse += (f64::from(mid.get(x, y)) - want).powi(2);
                let cross = 0.5 * (f64::from(prev_l.get(x, y)) + f64::from(next_l.get(x, y)));
                cross_mse += (cross - want).powi(2);
                count += 1.0;
            }
        }
        let psnr = -10.0 * (mse / count).log10();
        let cross_psnr = -10.0 * (cross_mse / count).log10();
        println!("thin line guard: psnr {psnr:.2} dB, crossfade {cross_psnr:.2} dB");
        // Without the unsharp pass this scene lands at 34.7 dB, so the bound
        // below fails if the pass stops firing or stops helping.
        assert!(
            psnr > cross_psnr + 9.0 && psnr > 36.0,
            "thin line synthesis {psnr:.2} dB is too weak (crossfade {cross_psnr:.2} dB)"
        );
    }

    #[test]
    #[ignore = "performance measurement"]
    fn perf_sharpen_cost() {
        for d in [Dims::new(1920, 1080), Dims::new(1280, 720)] {
            let frame = blob_frame(d, 0.0);
            let threads = worker_threads(d.area() as usize);
            let moving: Vec<Vec2> = vec![Vec2::new(3.0, 0.0); d.area() as usize];
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                let out = sharpen_frame(&frame, &moving, 0.5, threads);
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                std::hint::black_box(&out);
                best = best.min(ms);
            }
            println!(
                "sharpen {}x{} ({} threads): {best:.2} ms",
                d.width, d.height, threads
            );
        }
    }

    #[test]
    fn rgba_vector_synthesis_matches_scalar_reference() {
        // The four-channel kernel must stay indistinguishable from the scalar
        // per-channel arithmetic it replaced; recompute every pixel with the
        // scalar helpers (sample_px + round_u8) and compare exactly.
        let d = Dims::new(37, 23);
        let mut a = Frame::zeroed(d, PixelFormat::Rgba8);
        let mut b = Frame::zeroed(d, PixelFormat::Rgba8);
        for (i, byte) in a.data.iter_mut().enumerate() {
            *byte = (i.wrapping_mul(53) % 256) as u8;
        }
        for (i, byte) in b.data.iter_mut().enumerate() {
            *byte = (i.wrapping_mul(29).wrapping_add(11) % 256) as u8;
        }
        let w = d.width as usize;
        let h = d.height as usize;
        let alpha = 0.35f32;
        let flow = patterned_flow(d);
        let min_conf = FlowParams::default().confidence_min;
        for fill in [true, false] {
            let cfg = InterpolatorConfig {
                occlusion_fill: fill,
                ..InterpolatorConfig::default()
            };
            let it = Interpolator::new(&cfg);
            let (out, _) = it.interpolate(&a, &b, alpha, &flow).unwrap();
            // Reference synthesis, then the reference sharpening pass; both are
            // recomputed with the scalar helpers the vector kernel replaced.
            let mut synth = Frame::zeroed(d, PixelFormat::Rgba8);
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let (px, py) = (x as f32, y as f32);
                    let mut want = [0u8; 4];
                    if !fill && flow.confidence[i] < min_conf {
                        // No fill requested: the pixel is a plain temporal blend.
                        for (ch, slot) in want.iter_mut().enumerate() {
                            let ca = f32::from(a.data[i * 4 + ch]);
                            let cb = f32::from(b.data[i * 4 + ch]);
                            *slot = round_u8(ca + (cb - ca) * alpha);
                        }
                    } else {
                        let v = flow.forward.at(x as u32, y as u32);
                        let pa = sample_px::<4>(&a.data, w, h, px + alpha * v.x, py + alpha * v.y);
                        let pb = sample_px::<4>(
                            &b.data,
                            w,
                            h,
                            px - (1.0 - alpha) * v.x,
                            py - (1.0 - alpha) * v.y,
                        );
                        for (ch, slot) in want.iter_mut().enumerate() {
                            *slot = round_u8(pa[ch] + (pb[ch] - pa[ch]) * alpha);
                        }
                    }
                    synth.data[i * 4..i * 4 + 4].copy_from_slice(&want);
                }
            }
            let want = reference_sharpen(&synth, alpha);
            assert_eq!(out.data, want.data, "frame differs, fill {fill}");
        }
    }

    /// Frame whose content is rotated by `angle` about the centre:
    /// content(p) = texture(c + R(-angle) * (p + 0.5 - c)).
    fn rotate_frame(dims: Dims, angle: f32) -> Frame {
        let mut f = Frame::zeroed(dims, PixelFormat::Rgba8);
        let w = dims.width as usize;
        let (cx, cy) = (dims.width as f32 * 0.5, dims.height as f32 * 0.5);
        let (sa, ca) = angle.sin_cos();
        for y in 0..dims.height as usize {
            for x in 0..w {
                let vx = x as f32 + 0.5 - cx;
                let vy = y as f32 + 0.5 - cy;
                let sx = cx + ca * vx + sa * vy;
                let sy = cy - sa * vx + ca * vy;
                let v = (blob_value(sx / w as f32, sy / dims.height as f32) * 255.0 + 0.5) as u8;
                let o = (y * w + x) * 4;
                f.data[o] = v;
                f.data[o + 1] = v;
                f.data[o + 2] = v;
                f.data[o + 3] = 255;
            }
        }
        f
    }

    fn rotation_flow(dims: Dims, rot: GlobalRotation) -> FlowEstimate {
        let n = dims.area() as usize;
        let mut forward = FlowField::new(dims);
        let mut backward = FlowField::new(dims);
        for y in 0..dims.height {
            for x in 0..dims.width {
                let (xf, yf) = (x as f32, y as f32);
                forward.set(x, y, rot.forward_at(xf, yf));
                backward.set(x, y, rot.backward_at(xf, yf));
            }
        }
        FlowEstimate::new(forward, backward, vec![1.0; n]).with_rotation(Some(rot))
    }

    #[test]
    fn rotation_aware_synthesis_follows_the_arc() {
        let d = Dims::new(256, 144);
        // A large angle keeps the arc clearly separated from a straight blend.
        let angle = 20.0f32.to_radians();
        let prev = rotate_frame(d, 0.0);
        let next = rotate_frame(d, angle);
        let rot = GlobalRotation::new(angle, d.width as f32 * 0.5, d.height as f32 * 0.5);
        let flow = rotation_flow(d, rot);
        let it = Interpolator::new(&InterpolatorConfig::default());

        // The true intermediate of a rotation is the half-angle frame.
        let want = rotate_frame(d, angle * 0.5);
        let (mid, metrics) = it.interpolate(&prev, &next, 0.5, &flow).unwrap();
        assert!(!metrics.scene_cut);
        assert_eq!(metrics.blended_px, d.area());
        let mut err = 0.0f64;
        let mut n = 0u64;
        for y in 6..d.height - 6 {
            for x in 6..d.width - 6 {
                let got = i32::from(mid.pixel(x, y)[0]);
                let exp = i32::from(want.pixel(x, y)[0]);
                err += f64::from((got - exp).abs());
                n += 1;
            }
        }
        let mae = err / n as f64;
        assert!(mae < 3.0, "mid-frame MAE {mae:.2} over rotation");

        // Without the rotation hint the same fields are interpolated linearly,
        // which leaves the arc: the pictures must differ.
        let linear_flow = flow.clone().with_rotation(None);
        let (linear, _) = it.interpolate(&prev, &next, 0.5, &linear_flow).unwrap();
        let differing = (0..d.area() as usize)
            .filter(|&i| linear.data[i * 4].abs_diff(mid.data[i * 4]) > 2)
            .count();
        assert!(
            differing > 100,
            "rotation kernel unused: only {differing} pixels differ"
        );

        // The geodesic reduces to the input frames at the endpoints.
        let (start, _) = it.interpolate(&prev, &next, 0.0, &flow).unwrap();
        let (end, _) = it.interpolate(&prev, &next, 1.0, &flow).unwrap();
        for (a, b) in start.data.iter().zip(prev.data.iter()) {
            assert!(a.abs_diff(*b) <= 1);
        }
        for (a, b) in end.data.iter().zip(next.data.iter()) {
            assert!(a.abs_diff(*b) <= 1);
        }

        // Band partitioning must not change a byte of the rotated synthesis.
        let (serial, _) = it.interpolate_with(&prev, &next, 0.5, &flow, 1).unwrap();
        assert_eq!(serial.data, mid.data);
        for tid in [2usize, 4, 8] {
            let (par, _) = it.interpolate_with(&prev, &next, 0.5, &flow, tid).unwrap();
            assert_eq!(par.data, mid.data, "rotation synthesis differs at {tid}");
        }
    }
}
