//! Contrast-adaptive unsharp mask (RCAS-ish) and the shared detail-restore
//! kernel used by the adaptive upscaler.
//!
//! The mask is a 3x3 cross: the high pass response is
//! `hp = 4*c - (l + r + t + b)`. The delta is scaled by the strength and the
//! result is clamped to the min/max envelope of the full 3x3 neighbourhood
//! (centre, four edges and four corners). Clamping to that envelope is what
//! makes the filter halo free: a pixel can never move past a neighbouring value
//! and no new local extreme can be created, so a hard step is left untouched and
//! no overshoot rings appear. Only curvature (soft edges, ringing
//! already present in the base upscale, natural texture) moves, which is exactly
//! where sharpening helps; flat and linear regions have a zero response.
//!
//! A gradient estimate decides whether the response is taken from the full
//! cross or from the axis perpendicular to the dominant edge: along an edge the
//! pixels are already correlated, so sharpening across the edge (and not along
//! it) avoids accentuating diagonal aliasing.
//!
//! Everything is integer fixed point (strength in Q8) and touches luma only;
//! the same delta is applied to the red, green and blue bytes so hue is
//! preserved, and alpha is never modified.

use fluid_core::Frame;

use crate::pixel::{Layout, luma_row_f32};
use crate::resample::{row_chunks, threads_for};
use crate::simd::dispatch;

/// Detail-restore strength applied by the adaptive upscaler itself (Q8, ~0.2).
/// Deliberately gentle: user sharpening is a separate, opt-in stage.
pub(crate) const REFINE_STRENGTH_Q8: i32 = 52;

/// Quantise amount in 0..=1 to Q8 strength.
#[inline]
fn strength_q8(amount: f32) -> i32 {
    if !amount.is_finite() || amount <= 0.0 {
        return 0;
    }
    let a = amount.clamp(0.0, 1.0);
    (a * 256.0 + 0.5) as i32
}

/// The fixed point delta of one interior pixel, computed in the f32 domain.
///
/// `top`, `mid` and `bot` are the converted luma rows of the frame. The
/// expression mirrors the integer reference: every intermediate value is a small
/// integer and the single quotient is either an integer or at least 1/511 away
/// from one, so the f32 division rounds to within 6e-5 of the true quotient and
/// truncating it gives the same integer as integer division does.
#[inline(always)]
fn luma_delta(top: &[f32], mid: &[f32], bot: &[f32], x: usize, s: f32) -> i32 {
    let c = mid[x];
    let lf = mid[x - 1];
    let rg = mid[x + 1];
    let t = top[x];
    let b = bot[x];
    // Envelope of the 3x3 neighbourhood (centre, edges and corners).
    let mn = c
        .min(lf)
        .min(rg)
        .min(t)
        .min(b)
        .min(top[x - 1])
        .min(top[x + 1])
        .min(bot[x - 1])
        .min(bot[x + 1]);
    let mx = c
        .max(lf)
        .max(rg)
        .max(t)
        .max(b)
        .max(top[x - 1])
        .max(top[x + 1])
        .max(bot[x - 1])
        .max(bot[x + 1]);
    // Cross high pass, steered along the dominant gradient axis.
    let gx = (lf - rg).abs();
    let gy = (t - b).abs();
    let hp_cross = c * 4.0 - (lf + rg + t + b);
    let hp_axis = if gy > gx {
        c * 2.0 - (t + b)
    } else {
        c * 2.0 - (lf + rg)
    };
    let w_den = gx + gy + 1.0;
    let w_num = (gx - gy).abs();
    let hp = ((hp_cross * (w_den - w_num) + hp_axis * w_num) / w_den).trunc();
    let delta = ((hp * s + 128.0) * (1.0 / 256.0)).floor();
    // Clamp inside the envelope: no new local extreme, so no halo.
    ((c + delta).max(mn).min(mx) - c) as i32
}

/// Like [`luma_delta`] for a pixel at the row end, where the missing
/// neighbour is clamped to the pixel itself.
#[inline]
fn luma_delta_clamped(top: &[f32], mid: &[f32], bot: &[f32], x: usize, s: f32) -> i32 {
    let xl = x.saturating_sub(1);
    let xr = (x + 1).min(mid.len() - 1);
    let c = mid[x];
    let lf = mid[xl];
    let rg = mid[xr];
    let t = top[x];
    let b = bot[x];
    let mn = c
        .min(lf)
        .min(rg)
        .min(t)
        .min(b)
        .min(top[xl])
        .min(top[xr])
        .min(bot[xl])
        .min(bot[xr]);
    let mx = c
        .max(lf)
        .max(rg)
        .max(t)
        .max(b)
        .max(top[xl])
        .max(top[xr])
        .max(bot[xl])
        .max(bot[xr]);
    let gx = (lf - rg).abs();
    let gy = (t - b).abs();
    let hp_cross = c * 4.0 - (lf + rg + t + b);
    let hp_axis = if gy > gx {
        c * 2.0 - (t + b)
    } else {
        c * 2.0 - (lf + rg)
    };
    let w_den = gx + gy + 1.0;
    let w_num = (gx - gy).abs();
    let hp = ((hp_cross * (w_den - w_num) + hp_axis * w_num) / w_den).trunc();
    let delta = ((hp * s + 128.0) * (1.0 / 256.0)).floor();
    ((c + delta).max(mn).min(mx) - c) as i32
}

/// Add the sharpened delta to one luma row (and the matching colour bytes).
///
/// The three luma rows are the f32 planes kept by the caller, so no byte to
/// float conversion happens here at all: each frame row is converted once when
/// it enters the ring, instead of nine times per pixel for the nine taps of the
/// kernel. The per-pixel response loop is a plain array loop that the dispatched
/// codegen widens to the machine's vector width; `deltas` is caller scratch of
/// at least `mid.len()` entries, so a row allocates nothing.
#[inline(always)]
pub(crate) fn apply_delta_row(
    row: &mut [u8],
    top: &[f32],
    mid: &[f32],
    bot: &[f32],
    deltas: &mut [i32],
    l: &Layout,
    s_q8: i32,
) {
    let w = mid.len();
    if s_q8 <= 0 || w < 3 {
        return;
    }
    let s = s_q8 as f32;
    let (top, bot) = (&top[..w], &bot[..w]);
    let d = &mut deltas[..w];
    d[0] = luma_delta_clamped(top, mid, bot, 0, s);
    // Interior pixels: no clamping, so this loop has no branches at all.
    for (x, slot) in d.iter_mut().enumerate().take(w - 1).skip(1) {
        *slot = luma_delta(top, mid, bot, x, s);
    }
    d[w - 1] = luma_delta_clamped(top, mid, bot, w - 1, s);
    // One delta drives up to three colour bytes, so the application stays a
    // scalar loop over the row; the arithmetic above is the expensive part.
    for (x, delta) in d.iter().enumerate() {
        if *delta != 0 {
            apply_delta_bytes(row, x, l, *delta);
        }
    }
}

/// Add one luma delta to the colour bytes of a pixel; alpha is never touched.
#[inline(always)]
fn apply_delta_bytes(row: &mut [u8], x: usize, l: &Layout, d: i32) {
    let base = x * l.bpp;
    if l.bpp == 1 {
        row[base] = (row[base] as i32 + d).clamp(0, 255) as u8;
    } else {
        for off in [l.r, l.g, l.b] {
            let v = row[base + off] as i32 + d;
            row[base + off] = v.clamp(0, 255) as u8;
        }
    }
}

/// Streaming contrast-adaptive unsharp mask over a whole frame.
///
/// Only three luma rows are kept in flight, so the working set is O(width) and
/// the frame is traversed row by row (cache friendly, no per-pixel allocation).
///
/// Rows are independent once every row's luma is taken from the *unsharpened*
/// frame, so the pass splits into contiguous row bands. A band needs the luma of
/// the two rows just outside it; those are captured before any worker starts, so
/// a band never reads a row another worker is writing. The arithmetic is
/// unchanged, hence the parallel and serial outputs are bit-identical.
pub(crate) fn sharpen_frame(frame: &mut Frame, amount: f32) {
    let threads = threads_for(frame.dims.area(), frame.dims.height as usize);
    sharpen_frame_threads(frame, amount, threads);
}

/// Sharpen with an explicit worker count (1 is the serial path).
///
/// Exposed inside the crate so tests can compare parallel and serial output
/// byte for byte.
pub(crate) fn sharpen_frame_threads(frame: &mut Frame, amount: f32, threads: usize) {
    let s = strength_q8(amount);
    if s == 0 {
        return;
    }
    let l = Layout::of(frame.fmt);
    let w = frame.dims.width as usize;
    let h = frame.dims.height as usize;
    if w == 0 || h == 0 {
        return;
    }
    let threads = threads.max(1).min(h);
    let chunks = row_chunks(h, threads);
    if chunks.len() <= 1 {
        sharpen_serial(frame, s, &l);
        return;
    }
    // Luma of the rows just outside every band, taken from the untouched frame
    // before the workers run. Only two rows per band, so the copy is negligible.
    let mut tops: Vec<Option<Vec<f32>>> = Vec::with_capacity(chunks.len());
    let mut bots: Vec<Option<Vec<f32>>> = Vec::with_capacity(chunks.len());
    for &(y0, y1) in &chunks {
        let top = if y0 > 0 {
            let mut row = vec![0.0f32; w];
            luma_row_f32(frame.row((y0 - 1) as u32), &mut row, &l);
            Some(row)
        } else {
            None
        };
        let bot = if y1 < h {
            let mut row = vec![0.0f32; w];
            luma_row_f32(frame.row(y1 as u32), &mut row, &l);
            Some(row)
        } else {
            None
        };
        tops.push(top);
        bots.push(bot);
    }
    let stride = frame.stride();
    let spec = BandSpec {
        w,
        h,
        stride,
        layout: &l,
        s_q8: s,
    };
    let data: &mut [u8] = &mut frame.data[..h * stride];
    std::thread::scope(|scope| {
        let mut rest: &mut [u8] = data;
        for (ci, &(y0, y1)) in chunks.iter().enumerate() {
            let len = (y1 - y0) * stride;
            let (head, tail) = rest.split_at_mut(len);
            rest = tail;
            let top = tops[ci].take();
            let bot = bots[ci].take();
            let spec = &spec;
            scope.spawn(move || sharpen_band(head, y0, y1, spec, top, bot));
        }
    });
}

/// Shared, read-only description of the frame geometry a band needs.
#[derive(Clone, Copy)]
struct BandSpec<'a> {
    w: usize,
    h: usize,
    stride: usize,
    layout: &'a Layout,
    s_q8: i32,
}

/// Sharpen one contiguous band of rows (rows y0..y1 of the frame are in head,
/// starting at its first byte).
fn sharpen_band(
    head: &mut [u8],
    y0: usize,
    y1: usize,
    spec: &BandSpec<'_>,
    top_luma: Option<Vec<f32>>,
    bot_luma: Option<Vec<f32>>,
) {
    let BandSpec {
        w,
        h,
        stride,
        layout,
        s_q8,
    } = *spec;
    let mut deltas = vec![0i32; w];
    let mut mid = vec![0.0f32; w];
    luma_row_f32(&head[..stride], &mut mid, layout);
    // Row 0 of the frame clamps its top neighbour to itself.
    let mut top = match top_luma {
        Some(row) => row,
        None => mid.clone(),
    };
    let mut bot = vec![0.0f32; w];
    // The whole band runs inside one dispatch, so the row kernel is compiled
    // with the best instruction set this CPU supports.
    dispatch(move || {
        for ty in y0..y1 {
            if ty + 1 < y1 {
                let row = &head[(ty + 1 - y0) * stride..(ty + 2 - y0) * stride];
                luma_row_f32(row, &mut bot, layout);
            } else if ty + 1 < h {
                bot.copy_from_slice(bot_luma.as_ref().expect("boundary luma row"));
            } else {
                // Last row of the frame clamps its bottom neighbour to itself.
                bot.copy_from_slice(&mid);
            }
            let row = &mut head[(ty - y0) * stride..(ty - y0 + 1) * stride];
            apply_delta_row(row, &top, &mid, &bot, &mut deltas, layout, s_q8);
            std::mem::swap(&mut top, &mut mid);
            std::mem::swap(&mut mid, &mut bot);
        }
    });
}

/// Serial streaming implementation, used for small frames and one worker.
fn sharpen_serial(frame: &mut Frame, s: i32, l: &Layout) {
    let w = frame.dims.width as usize;
    let h = frame.dims.height as usize;
    // Luma is captured before a row is modified, so the ring always holds the
    // response of the unsharpened input.
    let mut ring: [Vec<f32>; 3] = [vec![0.0f32; w], vec![0.0f32; w], vec![0.0f32; w]];
    let mut deltas = vec![0i32; w];
    luma_row_f32(frame.row(0), &mut ring[0], l);
    dispatch(move || {
        for y in 1..h {
            let slot = y % 3;
            luma_row_f32(frame.row(y as u32), &mut ring[slot], l);
            let mid = (y - 1) % 3;
            let top = y.saturating_sub(2) % 3;
            let (t, m, b) = (&ring[top], &ring[mid], &ring[slot]);
            apply_delta_row(frame.row_mut((y - 1) as u32), t, m, b, &mut deltas, l, s);
        }
        // Last row: the bottom neighbour is clamped to itself.
        let y = h - 1;
        let mid = y % 3;
        let top = if y >= 1 { (y - 1) % 3 } else { mid };
        let (t, m) = (&ring[top], &ring[mid]);
        apply_delta_row(frame.row_mut(y as u32), t, m, m, &mut deltas, l, s);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluid_core::{Dims, PixelFormat};

    fn luma_frame(vals: &[u8], w: u32) -> Frame {
        let h = vals.len() as u32 / w;
        Frame::new(Dims::new(w, h), PixelFormat::Luma8, vals.to_vec()).unwrap()
    }

    /// A frame whose every row is the given profile.
    fn rows_of_profile(profile: &[u8], h: u32) -> Frame {
        let w = profile.len() as u32;
        let mut data = Vec::with_capacity(profile.len() * h as usize);
        for _ in 0..h {
            data.extend_from_slice(profile);
        }
        Frame::new(Dims::new(w, h), PixelFormat::Luma8, data).unwrap()
    }

    #[test]
    fn flat_frame_is_unchanged() {
        let mut f = luma_frame(&[80u8; 64], 8);
        let before = f.data.clone();
        sharpen_frame(&mut f, 1.0);
        assert_eq!(f.data, before);
    }

    #[test]
    fn zero_or_invalid_amount_is_noop() {
        let vals: Vec<u8> = (0..64u32).map(|i| (i * 7 % 251) as u8).collect();
        let mut f = luma_frame(&vals, 8);
        let before = f.data.clone();
        sharpen_frame(&mut f, 0.0);
        assert_eq!(f.data, before);
        sharpen_frame(&mut f, f32::NAN);
        assert_eq!(f.data, before);
        sharpen_frame(&mut f, -3.0);
        assert_eq!(f.data, before);
        assert_eq!(strength_q8(0.5), 128);
        assert_eq!(strength_q8(2.0), 256);
    }

    #[test]
    fn hard_step_is_left_alone_no_halos() {
        // A vertical step is already maximally sharp; a halo free mask must not
        // touch it (no overshoot, no undershoot).
        let profile = [50u8, 50, 50, 50, 200, 200, 200, 200];
        let mut f = rows_of_profile(&profile, 8);
        let before = f.data.clone();
        sharpen_frame(&mut f, 1.0);
        assert_eq!(f.data, before, "hard step must not gain halos");
    }

    #[test]
    fn soft_edge_gains_contrast_without_new_extremes() {
        // Bilinear-style ramp between two plateaus.
        let profile = [100u8, 100, 110, 130, 150, 160, 160, 160];
        let mut f = rows_of_profile(&profile, 8);
        let before = f.data.clone();
        sharpen_frame(&mut f, 1.0);
        let mn = *before.iter().min().unwrap();
        let mx = *before.iter().max().unwrap();
        for v in &f.data {
            assert!(*v >= mn && *v <= mx, "overshoot: {v} outside {mn}..={mx}");
        }
        // Variance must grow: that is what sharpening means for a soft edge.
        let var = |d: &[u8]| -> f64 {
            let mean = d.iter().map(|v| *v as f64).sum::<f64>() / d.len() as f64;
            d.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / d.len() as f64
        };
        assert!(var(&f.data) > var(&before), "soft edge should steepen");
    }

    #[test]
    fn monotone_strength_increases_response() {
        let profile = [100u8, 100, 110, 130, 150, 160, 160, 160];
        let base = rows_of_profile(&profile, 4);
        let mut weak = base.clone();
        let mut strong = base.clone();
        sharpen_frame(&mut weak, 0.25);
        sharpen_frame(&mut strong, 1.0);
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
    fn alpha_channel_is_never_touched() {
        let dims = Dims::new(4, 4);
        let mut data = Vec::with_capacity(16 * 4);
        for i in 0..16u32 {
            let v = if i % 2 == 0 { 10 } else { 240 };
            data.extend_from_slice(&[v, v, v, 77]);
        }
        let mut f = Frame::new(dims, PixelFormat::Rgba8, data).unwrap();
        sharpen_frame(&mut f, 1.0);
        for i in 0..16usize {
            assert_eq!(f.data[i * 4 + 3], 77);
        }
    }

    #[test]
    fn channels_keep_hue() {
        // Grey ramp in RGB: the same delta must hit all three channels.
        let profile: [u8; 8] = [100, 100, 110, 130, 150, 160, 160, 160];
        let dims = Dims::new(8, 4);
        let mut data = Vec::with_capacity(8 * 4 * 3);
        for _ in 0..4 {
            for v in profile {
                data.extend_from_slice(&[v, v, v]);
            }
        }
        let mut f = Frame::new(dims, PixelFormat::Rgb8, data).unwrap();
        sharpen_frame(&mut f, 1.0);
        for px in f.data.as_chunks::<3>().0 {
            assert_eq!(px[0], px[1]);
            assert_eq!(px[1], px[2]);
        }
    }

    /// Deterministic pseudo-random frame: every byte differs, so a bad band
    /// boundary changes the output.
    fn noise(dims: Dims, fmt: PixelFormat, seed: u32) -> Frame {
        let bpp = fmt.bytes_per_pixel();
        let mut data = vec![0u8; dims.area() as usize * bpp];
        let mut s = seed | 1;
        for (i, b) in data.iter_mut().enumerate() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *b = ((s >> 9) as u8).wrapping_add(i as u8).wrapping_mul(29);
        }
        Frame::new(dims, fmt, data).unwrap()
    }

    #[test]
    fn parallel_matches_serial_for_every_format() {
        for fmt in [
            PixelFormat::Rgba8,
            PixelFormat::Bgra8,
            PixelFormat::Rgb8,
            PixelFormat::Luma8,
        ] {
            let src = noise(Dims::new(97, 61), fmt, 17);
            let mut serial = src.clone();
            sharpen_frame_threads(&mut serial, 0.6, 1);
            for threads in [2usize, 3, 4, 16] {
                let mut parallel = src.clone();
                sharpen_frame_threads(&mut parallel, 0.6, threads);
                assert_eq!(
                    serial.data, parallel.data,
                    "{fmt:?} threads={threads} parallel != serial"
                );
            }
        }
    }

    #[test]
    fn parallel_matches_serial_for_uneven_bands() {
        // 203 rows over 5 workers gives uneven bands; both frame edges and the
        // band edges exercise the clamped-luma cases.
        let src = noise(Dims::new(50, 203), PixelFormat::Rgba8, 3);
        for amount in [0.05f32, 0.3, 1.0] {
            let mut serial = src.clone();
            sharpen_frame_threads(&mut serial, amount, 1);
            for threads in [2usize, 3, 5, 7] {
                let mut parallel = src.clone();
                sharpen_frame_threads(&mut parallel, amount, threads);
                assert_eq!(
                    serial.data, parallel.data,
                    "amount={amount} threads={threads}"
                );
            }
        }
    }

    #[test]
    fn parallel_matches_serial_on_tall_and_wide_frames() {
        // One row and one column are the degenerate shapes for the cross kernel.
        for dims in [Dims::new(64, 1), Dims::new(1, 64), Dims::new(2, 2)] {
            let src = noise(dims, PixelFormat::Luma8, 21);
            let mut serial = src.clone();
            sharpen_frame_threads(&mut serial, 1.0, 1);
            for threads in [2usize, 4, 16] {
                let mut parallel = src.clone();
                sharpen_frame_threads(&mut parallel, 1.0, threads);
                assert_eq!(serial.data, parallel.data, "{dims} threads={threads}");
            }
        }
    }

    #[test]
    fn tiled_kernel_matches_integer_reference() {
        let mut state = 0x9e37_79b9u32;
        fn next(state: &mut u32) -> u8 {
            *state ^= *state << 13;
            *state ^= *state >> 17;
            *state ^= *state << 5;
            *state as u8
        }
        // Integer reference of the kernel, one pixel at a time. The f32 tile
        // kernel must reproduce this bit for bit (integer division truncates
        // toward zero, and so does the truncated f32 quotient).
        #[allow(clippy::too_many_arguments)]
        fn reference_delta(
            c: i32,
            lf: i32,
            rg: i32,
            t: i32,
            b: i32,
            mn: i32,
            mx: i32,
            s_q8: i32,
        ) -> i32 {
            if s_q8 <= 0 || mn == mx {
                return 0;
            }
            let hp_cross = 4 * c - (lf + rg + t + b);
            let gx = (lf - rg).abs();
            let gy = (t - b).abs();
            let (hp_axis, w_num, w_den) = if gy > gx {
                (2 * c - t - b, gy - gx, gy + gx + 1)
            } else {
                (2 * c - lf - rg, gx - gy, gy + gx + 1)
            };
            let hp = (hp_cross * (w_den - w_num) + hp_axis * w_num) / w_den;
            let delta = (hp * s_q8 + 128) >> 8;
            if delta == 0 {
                return 0;
            }
            (c + delta).clamp(mn, mx) - c
        }
        let w = 97usize;
        for fmt in [
            PixelFormat::Rgba8,
            PixelFormat::Bgra8,
            PixelFormat::Rgb8,
            PixelFormat::Luma8,
        ] {
            let l = Layout::of(fmt);
            let stride = w * l.bpp;
            let mid: Vec<u8> = (0..w).map(|_| next(&mut state)).collect();
            let top: Vec<u8> = (0..w).map(|_| next(&mut state)).collect();
            let bot: Vec<u8> = (0..w).map(|_| next(&mut state)).collect();
            let row: Vec<u8> = (0..stride).map(|_| next(&mut state)).collect();
            let midf: Vec<f32> = mid.iter().map(|v| *v as f32).collect();
            let topf: Vec<f32> = top.iter().map(|v| *v as f32).collect();
            let botf: Vec<f32> = bot.iter().map(|v| *v as f32).collect();
            let mut deltas = vec![0i32; w];
            let mut fast = row.clone();
            apply_delta_row(&mut fast, &topf, &midf, &botf, &mut deltas, &l, 90);
            let mut slow = row.clone();
            for x in 0..w {
                let xl = x.saturating_sub(1);
                let xr = (x + 1).min(w - 1);
                let c = mid[x] as i32;
                let lf = mid[xl] as i32;
                let rg = mid[xr] as i32;
                let t = top[x] as i32;
                let b = bot[x] as i32;
                let env = [
                    c,
                    lf,
                    rg,
                    t,
                    b,
                    top[xl] as i32,
                    top[xr] as i32,
                    bot[xl] as i32,
                    bot[xr] as i32,
                ];
                let mn = env.iter().copied().min().unwrap();
                let mx = env.iter().copied().max().unwrap();
                let d = reference_delta(c, lf, rg, t, b, mn, mx, 90);
                if d != 0 {
                    apply_delta_bytes(&mut slow, x, &l, d);
                }
            }
            assert_eq!(fast, slow, "{fmt:?}: tile kernel differs from reference");
        }
    }
}
