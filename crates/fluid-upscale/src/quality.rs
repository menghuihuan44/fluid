//! Analytic quality harness for the upscaling filters.
//!
//! This module is analysis-only: the pipeline does not call it, and it has no
//! effect on runtime behaviour. It exists so the filters can be scored against a
//! ground truth that is known exactly instead of a photograph.
//!
//! Ground truth: [`render_reference`] draws a six-band chart from pure maths at
//! the native output size (1280x720 or 1920x1080): hard edges at three angles,
//! thin 1-2 px lines, a one- and two-pixel checkerboard, a zone plate with
//! radial spokes, text-like glyph blocks and smooth colour gradients. Because
//! every pixel is computed analytically, no detail has to be estimated: it is
//! the ideal image an upscaler is trying to reconstruct.
//!
//! Input: [`downscale_box`] produces the low-resolution frames by an exact
//! integer-ratio area average, which is the mathematical inverse of what the
//! filter has to undo. Upscaling the box average back to the reference size and
//! comparing against the reference is therefore a fair test.
//!
//! Metrics:
//! - PSNR over the RGB channels in u8 units. A zero-error frame is reported as
//!   99 dB instead of infinity so the value stays JSON-friendly. The mean is
//!   computed from global MSE; the minimum is the worst 64x64 tile, which is
//!   what exposes ringing and lost edges that a global average hides.
//! - SSIM: the standard 11x11 Gaussian window (sigma 1.5) variant, evaluated on
//!   BT.709 luma with the usual C1 = (0.01 * 255)^2 and C2 = (0.03 * 255)^2.
//! - Gradient energy ratio: mean |dx| + |dy| of the upscaled luma divided by the
//!   same on the reference. Below 1 means detail was lost, above 1 means the
//!   kernel invented contrast (ringing or sharpening).
//! - Flicker: mean absolute difference between consecutive upscaled outputs of a
//!   static chart whose low-resolution input carries a deterministic dither of up
//!   to three LSB. It measures how much of the input noise survives to the
//!   screen; a temporal filter scores below the spatial filters.
//!
//! Two chart bands are deliberately at or near the Nyquist limit of the
//! low-resolution input (the one-pixel checkerboard and the one- and two-pixel
//! lines). The box downscale averages those bands to a near-constant value, so no
//! upscaler can recover them: they show up as the low per-band PSNR and as the
//! worst-tile minimum (around 7.5 dB) for every spatial filter. That is expected
//! behaviour of the measurement, not a filter defect, and it is why the per-band
//! breakdown is reported next to the global mean.
//!
//! [`default_scenarios`] enumerates the static reported matrix: each reference
//! size (1280x720, 1920x1080) at 2x, plus 3x for 1920x1080, every spatial filter
//! (nearest, bilinear, bicubic, lanczos, adaptive) at sharpness 0.0, 0.3 and 0.6,
//! and the temporal filter over a three-frame moving-shape sequence at all three
//! sharpness levels. [`default_motion_cases`] adds the ghosting matrix: a disc
//! translated by 0, 2, 6, 12 or 24 reference pixels per frame over six frames, at
//! 2x from both reference sizes. [`run_scenario`] and [`run_motion_case`]
//! measure one cell; the `quality` example prints both tables and writes the
//! JSON, whose motion_cases array carries every per-frame PSNR.

use std::time::Instant;

use fluid_core::{Dims, Frame, PixelFormat, TemporalConfig, UpscaleFilter, UpscalerConfig};

use crate::Upscaler;

/// PSNR assigned to a pixel-exact match (dB), instead of infinity.
pub const PSNR_CAP_DB: f64 = 99.0;

/// Number of content bands in the analytic chart.
pub const CHART_BANDS: usize = 6;

/// Human-readable names of the chart bands, in vertical order.
pub const BAND_NAMES: [&str; CHART_BANDS] = [
    "hard edges",
    "thin lines",
    "checkerboard",
    "zone plate",
    "glyph blocks",
    "gradients",
];

/// Frames in each moving-content ghosting sequence.
pub const MOTION_FRAMES: u32 = 6;

/// Horizontal disc speeds of the ghosting matrix, in reference pixels per frame.
pub const MOTION_SPEEDS: [u32; 5] = [0, 2, 6, 12, 24];

/// Frames in the moving-shape sequence used for the temporal scenarios.
pub const TEMPORAL_FRAMES: u32 = 3;

/// Sharpening strengths every filter is measured at.
const SHARPNESS_LEVELS: [f32; 3] = [0.0, 0.3, 0.6];

/// Timing passes per scenario. The fastest pass wins, which is what keeps a busy
/// shared machine from dominating the reported call time.
const TIMING_ROUNDS: usize = 3;

#[inline]
fn clamp_f32(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp_f32((x - e0) / (e1 - e0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn to_u8(v: f32) -> u8 {
    (clamp_f32(v, 0.0, 255.0) + 0.5) as u8
}

/// Five by seven glyph bitmaps for the text-like band. One row per entry, five
/// significant bits, most significant bit is the leftmost column.
const GLYPHS: [[u8; 7]; 5] = [
    [
        0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
    ],
    [
        0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
    ],
    [
        0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
    ],
    [
        0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111,
    ],
    [
        0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
    ],
];

#[inline]
fn glyph_bit(glyph: usize, x: u32, y: u32) -> bool {
    if x >= 5 || y >= 7 {
        return false;
    }
    GLYPHS[glyph % GLYPHS.len()][y as usize] & (0b10000 >> x) != 0
}

/// Analytic chart sample at one pixel, in linear 0..255 RGB.
fn sample_chart(w: u32, h: u32, x: u32, y: u32) -> [f32; 3] {
    let band = ((y as u64 * CHART_BANDS as u64) / h as u64) as usize;
    let band = band.min(CHART_BANDS - 1);
    let band_top = (band as u64 * h as u64 / CHART_BANDS as u64) as u32;
    let band_bottom = ((band as u64 + 1) * h as u64 / CHART_BANDS as u64) as u32;
    let band_h = band_bottom.saturating_sub(band_top).max(1);
    let by = y.saturating_sub(band_top).min(band_h - 1);
    match band {
        0 => {
            // Three hard edges at 0, 20 and 40 degrees.
            let third = (w / 3).max(1);
            let sub = ((x / third) as usize).min(2);
            let lx = (x % third) as f32 / third as f32;
            let ly = by as f32 / band_h as f32;
            let angle = [0.0f32, 0.35, 0.7][sub];
            let d = (lx - 0.5) * angle.cos() + (ly - 0.5) * angle.sin();
            let v = if d > 0.0 { 235.0 } else { 25.0 };
            [v, v, v]
        }
        1 => {
            // 1 px vertical lines and 2 px horizontal lines on a dark field.
            let v = if x.is_multiple_of(13) {
                240.0
            } else if y % 17 < 2 {
                150.0
            } else {
                30.0
            };
            [v, v * 0.98, v * 0.94]
        }
        2 => {
            // One-pixel checkerboard on the left, two-pixel on the right.
            let block: u32 = if x < w / 2 { 1 } else { 2 };
            let v = if ((x / block) + (y / block)).is_multiple_of(2) {
                235.0
            } else {
                20.0
            };
            [v, v, v]
        }
        3 => {
            // Zone plate plus radial spokes: frequency rises with radius.
            let cx = w as f32 * 0.5;
            let cy = band_h as f32 * 0.5;
            let dx = (x as f32 - cx) / h as f32;
            let dy = (by as f32 - cy) / h as f32;
            let r2 = dx * dx + dy * dy;
            let ring = 0.5 + 0.5 * (r2 * 400.0).cos();
            let spoke = 0.5 + 0.5 * (dy.atan2(dx) * 10.0).cos();
            let v = 255.0 * (0.75 * ring + 0.25 * spoke);
            [v, v * 0.85, v]
        }
        4 => {
            // Text-like glyph blocks: repeated F L U I D cells.
            let cell_w = (w / 40).max(6);
            let cell_h = (band_h / 12).max(8);
            let gx = x / cell_w;
            let gy = by / cell_h;
            let glyph = ((gx * 7 + gy * 3) % 5) as usize;
            let fx = (x % cell_w) * 5 / cell_w;
            let fy = (by % cell_h) * 7 / cell_h;
            let v = if glyph_bit(glyph, fx, fy) {
                230.0
            } else {
                20.0
            };
            [v, v, v * 0.9]
        }
        _ => {
            // Smooth colour gradients with a low-frequency modulation.
            let u = x as f32 / w as f32;
            let v = by as f32 / band_h as f32;
            let r = 20.0 + 215.0 * u;
            let g = 30.0 + 190.0 * v;
            let b = 128.0 + 90.0 * ((u * 3.0).cos() * (v * 2.0).cos());
            [r, g, b]
        }
    }
}

/// Render the analytic reference chart at the requested size.
///
/// The same composition is produced at every size because the bands are laid
/// out in normalised coordinates, while the features that must stay at pixel
/// scale (thin lines, checkerboard, glyph strokes) are drawn in pixel units.
pub fn render_reference(dims: Dims) -> Frame {
    let w = dims.width.max(1);
    let h = dims.height.max(1);
    let mut data = vec![0u8; (w as usize) * (h as usize) * 4];
    for y in 0..h {
        for x in 0..w {
            let [r, g, b] = sample_chart(w, h, x, y);
            let i = ((y as usize) * (w as usize) + x as usize) * 4;
            data[i] = to_u8(r);
            data[i + 1] = to_u8(g);
            data[i + 2] = to_u8(b);
            data[i + 3] = 255;
        }
    }
    Frame::new(dims, PixelFormat::Rgba8, data).expect("analytic chart dimensions")
}

/// Render one frame of the moving-shape sequence used by the temporal filter.
///
/// The scene has a static gradient background, a static grid and border for
/// alignment, and one bright disc that moves horizontally by about seven percent
/// of the width per frame. A temporal filter that over-blends history will show
/// trailing ghosts on the disc, which the PSNR against the per-frame reference
/// records honestly.
pub fn render_temporal_frame(dims: Dims, frame: u32) -> Frame {
    let w = dims.width.max(1);
    let h = dims.height.max(1);
    let min_side = w.min(h) as f32;
    let mut data = vec![0u8; (w as usize) * (h as usize) * 4];
    for y in 0..h {
        for x in 0..w {
            let u = x as f32 / w.max(1) as f32;
            let v = y as f32 / h.max(1) as f32;
            let mut r = 16.0 + 55.0 * u;
            let mut g = 20.0 + 45.0 * v;
            let mut b = 28.0 + 35.0 * (0.5 + 0.5 * (u * 6.0 + v * 4.0).sin());
            // Static alignment grid.
            if x % 32 == 0 || y % 32 == 0 {
                r = 120.0;
                g = 120.0;
                b = 132.0;
            }
            // Moving disc with a soft one-pixel edge.
            let cx = (0.14 + 0.07 * frame as f32) * w as f32;
            let cy = 0.5 * h as f32;
            let radius = 0.09 * min_side;
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let mask = 1.0 - smoothstep(radius - 1.5, radius + 1.5, dist);
            r += (250.0 - r) * mask;
            g += (215.0 - g) * mask;
            b += (40.0 - b) * mask;
            // Static frame border, one pixel wide.
            if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                r = 235.0;
                g = 235.0;
                b = 235.0;
            }
            let i = ((y as usize) * (w as usize) + x as usize) * 4;
            data[i] = to_u8(r);
            data[i + 1] = to_u8(g);
            data[i + 2] = to_u8(b);
            data[i + 3] = 255;
        }
    }
    Frame::new(dims, PixelFormat::Rgba8, data).expect("temporal frame dimensions")
}

/// Render one frame of the moving-content ghosting sequence.
///
/// A bright disc slides right by speed_px reference pixels per frame across a
/// static textured background (soft two-axis gradient plus a 48 px grid), so
/// every output frame has an exact analytic ground truth and the area behind the
/// disc is known: a temporal filter that over-weights history leaves a visible
/// trail there. Speed 0 is the static control.
pub fn render_motion_frame(dims: Dims, speed_px: u32, frame: u32) -> Frame {
    let w = dims.width.max(1) as usize;
    let h = dims.height.max(1) as usize;
    let mut data = vec![0u8; w * h * 4];
    let min_side = w.min(h) as f32;
    let radius = 0.07 * min_side;
    let cx = 0.10 * w as f32 + speed_px as f32 * frame as f32;
    let cy = 0.5 * h as f32;
    for y in 0..h {
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            let v = (y as f32 + 0.5) / h as f32;
            let mut r = 16.0 + 38.0 * u;
            let mut g = 20.0 + 32.0 * v;
            let mut b = 28.0 + 26.0 * (0.5 + 0.5 * (u * 5.0 + v * 7.0).sin());
            if x % 48 == 0 || y % 48 == 0 {
                r = 92.0;
                g = 100.0;
                b = 118.0;
            }
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let mask = 1.0 - smoothstep(radius - 1.5, radius + 1.5, dist);
            if mask > 0.0 {
                r += (250.0 - r) * mask;
                g += (198.0 - g) * mask;
                b += (52.0 - b) * mask;
            }
            let i = (y * w + x) * 4;
            data[i] = to_u8(r);
            data[i + 1] = to_u8(g);
            data[i + 2] = to_u8(b);
            data[i + 3] = 255;
        }
    }
    Frame::new(dims, PixelFormat::Rgba8, data).expect("motion frame dimensions")
}

/// Exact integer-ratio box (area average) downscale: the inverse of upscaling.
///
/// Every destination pixel is the rounded mean of the factor x factor block of
/// source pixels, which is the ideal anti-aliased low-resolution observation for
/// an integer scale factor. The source dimensions must be an exact multiple of
/// the destination dimensions.
pub fn downscale_box(src: &Frame, dst: Dims) -> Frame {
    let factor = src.dims.width / dst.width.max(1);
    assert!(factor >= 1, "downscale factor must be at least one");
    assert_eq!(
        src.dims.width,
        dst.width * factor,
        "downscale width must be an exact multiple"
    );
    assert_eq!(
        src.dims.height,
        dst.height * factor,
        "downscale height must be an exact multiple"
    );
    let bpp = src.fmt.bytes_per_pixel();
    assert!(bpp <= 4, "box downscale supports up to four channels");
    let f = factor as usize;
    let src_stride = src.dims.width as usize * bpp;
    let dst_stride = dst.width as usize * bpp;
    let mut data = vec![0u8; src.fmt.buffer_len(dst)];
    let n = (f * f) as u32;
    for dy in 0..dst.height as usize {
        for dx in 0..dst.width as usize {
            let mut sums = [0u32; 4];
            for sy in 0..f {
                let row = (dy * f + sy) * src_stride;
                for sx in 0..f {
                    let i = row + (dx * f + sx) * bpp;
                    for (sum, byte) in sums.iter_mut().zip(&src.data[i..i + bpp]) {
                        *sum += *byte as u32;
                    }
                }
            }
            let o = dy * dst_stride + dx * bpp;
            for c in 0..bpp {
                data[o + c] = ((sums[c] + n / 2) / n) as u8;
            }
        }
    }
    Frame::new(dst, src.fmt, data).expect("downscale dimensions")
}

/// Deterministic small dither used by the flicker metric.
///
/// Each frame gets a different offset of up to three LSB per pixel, which models
/// the sensor and quantisation noise a temporal filter is supposed to damp. It is
/// a fixed hash, not a random number generator, so runs are reproducible.
pub fn add_dither(frame: &mut Frame, seed: u32) {
    let bpp = frame.fmt.bytes_per_pixel();
    let w = frame.dims.width;
    for y in 0..frame.dims.height {
        for x in 0..w {
            let h = hash3(x, y, seed);
            let delta = (h % 7) as i32 - 3;
            if delta == 0 {
                continue;
            }
            let i = ((y as usize) * (w as usize) + x as usize) * bpp;
            for c in 0..bpp.min(3) {
                let v = frame.data[i + c] as i32 + delta;
                frame.data[i + c] = v.clamp(0, 255) as u8;
            }
        }
    }
}

#[inline]
fn hash3(x: u32, y: u32, seed: u32) -> u32 {
    let mut h = x
        .wrapping_mul(374_761_393)
        .wrapping_add(y.wrapping_mul(668_265_263))
        .wrapping_add(seed.wrapping_mul(2_246_822_519));
    h ^= h >> 13;
    h = h.wrapping_mul(1_274_126_177);
    h ^ (h >> 16)
}

#[inline]
fn luma_at(data: &[u8], bpp: usize, i: usize) -> f32 {
    if bpp == 1 {
        return data[i] as f32;
    }
    0.2126 * data[i] as f32 + 0.7152 * data[i + 1] as f32 + 0.0722 * data[i + 2] as f32
}

/// BT.709 luma plane in 0..255.
pub fn luma_plane(frame: &Frame) -> Vec<f32> {
    let bpp = frame.fmt.bytes_per_pixel();
    let n = (frame.dims.width as usize) * (frame.dims.height as usize);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(luma_at(&frame.data, bpp, i * bpp));
    }
    out
}

fn check_pair(a: &Frame, b: &Frame) {
    assert_eq!(a.dims, b.dims, "metric needs equal dimensions");
    assert_eq!(a.fmt, b.fmt, "metric needs equal pixel formats");
}

/// PSNR over the RGB channels in u8 units, capped at [`PSNR_CAP_DB`].
pub fn psnr(a: &Frame, b: &Frame) -> f64 {
    check_pair(a, b);
    let bpp = a.fmt.bytes_per_pixel();
    let channels = bpp.min(3);
    let mut sum = 0.0f64;
    let mut count = 0u64;
    for (pa, pb) in a.data.chunks_exact(bpp).zip(b.data.chunks_exact(bpp)) {
        for c in 0..channels {
            let d = pa[c] as f64 - pb[c] as f64;
            sum += d * d;
        }
        count += channels as u64;
    }
    if count == 0 {
        return PSNR_CAP_DB;
    }
    let mse = sum / count as f64;
    if mse <= 1.0e-12 {
        PSNR_CAP_DB
    } else {
        (10.0 * (255.0f64 * 255.0 / mse).log10()).min(PSNR_CAP_DB)
    }
}

/// PSNR restricted to a horizontal band of rows.
pub fn psnr_rows(a: &Frame, b: &Frame, y0: u32, y1: u32) -> f64 {
    check_pair(a, b);
    let bpp = a.fmt.bytes_per_pixel();
    let channels = bpp.min(3);
    let stride = a.dims.width as usize * bpp;
    let mut sum = 0.0f64;
    let mut count = 0u64;
    for y in y0..y1.min(a.dims.height) {
        let start = y as usize * stride;
        let end = start + stride;
        for (pa, pb) in a.data[start..end]
            .chunks_exact(bpp)
            .zip(b.data[start..end].chunks_exact(bpp))
        {
            for c in 0..channels {
                let d = pa[c] as f64 - pb[c] as f64;
                sum += d * d;
            }
            count += channels as u64;
        }
    }
    if count == 0 {
        return PSNR_CAP_DB;
    }
    let mse = sum / count as f64;
    if mse <= 1.0e-12 {
        PSNR_CAP_DB
    } else {
        (10.0 * (255.0f64 * 255.0 / mse).log10()).min(PSNR_CAP_DB)
    }
}

/// Mean and worst-tile PSNR over 64x64 pixel tiles.
///
/// The worst tile is the useful number: a global average can look fine while a
/// whole edge block is destroyed.
pub fn psnr_tiles(a: &Frame, b: &Frame, tile: u32) -> (f64, f64) {
    check_pair(a, b);
    let tile = tile.max(1);
    let mut sum = 0.0f64;
    let mut tiles = 0u64;
    let mut worst = PSNR_CAP_DB;
    let mut y = 0;
    while y < a.dims.height {
        let y1 = (y + tile).min(a.dims.height);
        let mut x = 0;
        while x < a.dims.width {
            let x1 = (x + tile).min(a.dims.width);
            let p = psnr_rows_cols(a, b, x, x1, y, y1);
            sum += p;
            tiles += 1;
            worst = worst.min(p);
            x = x1;
        }
        y = y1;
    }
    if tiles == 0 {
        return (PSNR_CAP_DB, PSNR_CAP_DB);
    }
    (sum / tiles as f64, worst)
}

fn psnr_rows_cols(a: &Frame, b: &Frame, x0: u32, x1: u32, y0: u32, y1: u32) -> f64 {
    let bpp = a.fmt.bytes_per_pixel();
    let channels = bpp.min(3);
    let stride = a.dims.width as usize * bpp;
    let mut sum = 0.0f64;
    let mut count = 0u64;
    for y in y0..y1 {
        let base = y as usize * stride;
        for x in x0..x1 {
            let i = base + x as usize * bpp;
            for c in 0..channels {
                let d = a.data[i + c] as f64 - b.data[i + c] as f64;
                sum += d * d;
            }
            count += channels as u64;
        }
    }
    if count == 0 {
        return PSNR_CAP_DB;
    }
    let mse = sum / count as f64;
    if mse <= 1.0e-12 {
        PSNR_CAP_DB
    } else {
        (10.0 * (255.0f64 * 255.0 / mse).log10()).min(PSNR_CAP_DB)
    }
}

/// Mean absolute RGB difference in u8 units, used for the flicker metric.
pub fn mean_abs_diff(a: &Frame, b: &Frame) -> f64 {
    check_pair(a, b);
    let bpp = a.fmt.bytes_per_pixel();
    let channels = bpp.min(3);
    let mut sum = 0.0f64;
    let mut count = 0u64;
    for (pa, pb) in a.data.chunks_exact(bpp).zip(b.data.chunks_exact(bpp)) {
        for c in 0..channels {
            sum += (pa[c] as f64 - pb[c] as f64).abs();
        }
        count += channels as u64;
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

fn mean_gradient(m: &[f32], w: usize, h: usize) -> f64 {
    if w < 2 || h < 2 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            let i = y * w + x;
            sum += (m[i + 1] - m[i]).abs() as f64 + (m[i + w] - m[i]).abs() as f64;
        }
    }
    sum / (((w - 1) * (h - 1)) as f64)
}

/// Ratio of mean gradient energy (upscaled over reference).
///
/// One is a perfect match, below one is lost detail, above one is invented
/// contrast.
pub fn gradient_energy_ratio(out: &Frame, reference: &Frame) -> f64 {
    check_pair(out, reference);
    let w = out.dims.width as usize;
    let h = out.dims.height as usize;
    let mine = mean_gradient(&luma_plane(out), w, h);
    let truth = mean_gradient(&luma_plane(reference), w, h);
    if truth <= 1.0e-12 {
        if mine <= 1.0e-12 { 1.0 } else { f64::INFINITY }
    } else {
        mine / truth
    }
}

#[inline]
fn gaussian_kernel_11() -> [f32; 11] {
    const SIGMA: f32 = 1.5;
    let mut k = [0.0f32; 11];
    let mut sum = 0.0f32;
    for (i, slot) in k.iter_mut().enumerate() {
        let x = i as f32 - 5.0;
        let v = (-(x * x) / (2.0 * SIGMA * SIGMA)).exp();
        *slot = v;
        sum += v;
    }
    for slot in k.iter_mut() {
        *slot /= sum;
    }
    k
}

/// Separable 11-tap Gaussian blur with clamped edges.
fn blur(src: &[f32], w: usize, h: usize, k: &[f32; 11]) -> Vec<f32> {
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let mut acc = 0.0f32;
            for (t, weight) in k.iter().enumerate() {
                let sx = (x as i32 + t as i32 - 5).clamp(0, w as i32 - 1) as usize;
                acc += src[row + sx] * weight;
            }
            tmp[row + x] = acc;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (t, weight) in k.iter().enumerate() {
                let sy = (y as i32 + t as i32 - 5).clamp(0, h as i32 - 1) as usize;
                acc += tmp[sy * w + x] * weight;
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// Standard 11x11 Gaussian-window SSIM (sigma 1.5) on BT.709 luma.
///
/// Identical frames score 1.0 up to float rounding.
pub fn ssim(a: &Frame, b: &Frame) -> f64 {
    check_pair(a, b);
    let w = a.dims.width as usize;
    let h = a.dims.height as usize;
    if w == 0 || h == 0 {
        return 1.0;
    }
    let x = luma_plane(a);
    let y = luma_plane(b);
    let k = gaussian_kernel_11();
    let mu_x = blur(&x, w, h, &k);
    let mu_y = blur(&y, w, h, &k);
    let x2: Vec<f32> = x.iter().map(|v| v * v).collect();
    let y2: Vec<f32> = y.iter().map(|v| v * v).collect();
    let xy: Vec<f32> = x.iter().zip(y.iter()).map(|(p, q)| p * q).collect();
    let ex2 = blur(&x2, w, h, &k);
    let ey2 = blur(&y2, w, h, &k);
    let exy = blur(&xy, w, h, &k);
    let c1 = (0.01f64 * 255.0) * (0.01f64 * 255.0);
    let c2 = (0.03f64 * 255.0) * (0.03f64 * 255.0);
    let mut sum = 0.0f64;
    for i in 0..w * h {
        let mx = mu_x[i] as f64;
        let my = mu_y[i] as f64;
        let vx = (ex2[i] as f64 - mx * mx).max(0.0);
        let vy = (ey2[i] as f64 - my * my).max(0.0);
        let vxy = exy[i] as f64 - mx * my;
        let num = (2.0 * mx * my + c1) * (2.0 * vxy + c2);
        let den = (mx * mx + my * my + c1) * (vx + vy + c2);
        sum += if den > 0.0 { num / den } else { 1.0 };
    }
    sum / (w * h) as f64
}

/// One cell of the quality matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scenario {
    /// Native size the reference and the upscaled output share.
    pub ref_dims: Dims,
    /// Size of the downscaled input fed to the upscaler.
    pub low_dims: Dims,
    /// Filter under test (Temporal when `temporal` is set).
    pub filter: UpscaleFilter,
    /// Sharpening strength passed to the upscaler config.
    pub sharpness: f32,
    /// Run the three-frame moving-shape sequence instead of the static chart.
    pub temporal: bool,
}

impl Scenario {
    /// Integer scale factor between the input and the output.
    pub fn scale_factor(&self) -> u32 {
        self.ref_dims.width / self.low_dims.width.max(1)
    }

    /// One-line description for tables and logs.
    pub fn describe(&self) -> String {
        let sharp = if self.sharpness > 0.0 {
            format!("sharp {:.1}", self.sharpness)
        } else {
            "sharp 0.0".to_string()
        };
        if self.temporal {
            format!(
                "{}x{} from {} ({}x) temporal {} frames {}",
                self.ref_dims.width,
                self.ref_dims.height,
                self.low_dims,
                self.scale_factor(),
                TEMPORAL_FRAMES,
                sharp
            )
        } else {
            format!(
                "{}x{} from {} ({}x) {:?} {}",
                self.ref_dims.width,
                self.ref_dims.height,
                self.low_dims,
                self.scale_factor(),
                self.filter,
                sharp
            )
        }
    }
}

/// One moving-content ghosting case: temporal upscale of a sliding disc.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionCase {
    /// Native size the output is compared against.
    pub ref_dims: Dims,
    /// Size of the downscaled input fed to the upscaler.
    pub low_dims: Dims,
    /// Disc travel in reference pixels per frame (0 is the static control).
    pub speed_px: u32,
    /// Sharpening strength passed to the upscaler config.
    pub sharpness: f32,
}

impl MotionCase {
    /// Integer scale factor between the input and the output.
    pub fn scale_factor(&self) -> u32 {
        self.ref_dims.width / self.low_dims.width.max(1)
    }

    /// One-line description for tables and logs.
    pub fn describe(&self) -> String {
        let sharp = if self.sharpness > 0.0 {
            format!(" sharp {:.1}", self.sharpness)
        } else {
            String::new()
        };
        format!(
            "{}x{} from {} ({}x) speed {} px/frame{}",
            self.ref_dims.width,
            self.ref_dims.height,
            self.low_dims,
            self.scale_factor(),
            self.speed_px,
            sharp
        )
    }
}

/// The moving-content matrix: 2x from 1920x1080 and 1280x720 at sharpness 0.3,
/// for disc speeds of 0 (static), 2, 6, 12 and 24 reference pixels per frame.
pub fn default_motion_cases() -> Vec<MotionCase> {
    let mut cases = Vec::new();
    for ref_dims in [Dims::new(1920, 1080), Dims::new(1280, 720)] {
        let low_dims = Dims::new(ref_dims.width / 2, ref_dims.height / 2);
        for speed_px in MOTION_SPEEDS {
            cases.push(MotionCase {
                ref_dims,
                low_dims,
                speed_px,
                sharpness: 0.3,
            });
        }
    }
    cases
}

/// The full reported matrix: two reference sizes at 2x, 1920x1080 at 3x, every
/// spatial filter at sharpness 0.0, 0.3 and 0.6, and the temporal filter over
/// three frames at all three sharpness levels.
pub fn default_scenarios() -> Vec<Scenario> {
    let cases = [
        (Dims::new(1280, 720), 2u32),
        (Dims::new(1920, 1080), 2u32),
        (Dims::new(1920, 1080), 3u32),
    ];
    let spatial = [
        UpscaleFilter::Nearest,
        UpscaleFilter::Bilinear,
        UpscaleFilter::Bicubic,
        UpscaleFilter::Lanczos,
        UpscaleFilter::Adaptive,
    ];
    let mut out = Vec::new();
    for (ref_dims, factor) in cases {
        let low_dims = Dims::new(ref_dims.width / factor, ref_dims.height / factor);
        for filter in spatial {
            for sharpness in SHARPNESS_LEVELS {
                out.push(Scenario {
                    ref_dims,
                    low_dims,
                    filter,
                    sharpness,
                    temporal: false,
                });
            }
        }
        for sharpness in SHARPNESS_LEVELS {
            out.push(Scenario {
                ref_dims,
                low_dims,
                filter: UpscaleFilter::Temporal,
                sharpness,
                temporal: true,
            });
        }
    }
    out
}

/// Measured metrics for one scenario.
#[derive(Clone, Debug)]
pub struct ScenarioMetrics {
    /// Number of frames compared (1 for the static chart, 3 for temporal).
    pub frames: u32,
    /// Mean PSNR over the compared frames, in dB.
    pub psnr_mean: f64,
    /// Worst 64x64-tile PSNR over all compared frames, in dB.
    pub psnr_min: f64,
    /// Mean SSIM over the compared frames.
    pub ssim: f64,
    /// Gradient energy ratio, upscaled over reference.
    pub gradient_ratio: f64,
    /// Mean absolute difference between consecutive static-scene outputs.
    pub flicker: f64,
    /// Upscale call time in milliseconds: fastest of three timing passes per
    /// frame, median across the frames of the sequence.
    pub ms: f64,
    /// Mean PSNR per content band, in chart order.
    pub band_psnr: Vec<f64>,
    /// Fingerprint of every compared output frame, so a before/after run can
    /// prove a filter's pixels did not move.
    pub checksum: u64,
}

/// Measured metrics for one moving-content case.
#[derive(Clone, Debug)]
pub struct MotionMetrics {
    /// Frames processed (MOTION_FRAMES).
    pub frames: u32,
    /// PSNR of each output frame against its own analytic reference.
    pub psnr_per_frame: Vec<f64>,
    /// Mean of psnr_per_frame.
    pub psnr_mean: f64,
    /// Minimum of psnr_per_frame: the ghosting/trailing headline number.
    pub psnr_min_sequence: f64,
    /// Mean absolute RGB error in the trail behind the disc, in u8 units.
    pub ghosting: f64,
    /// Upscale call time in milliseconds: median of the per-frame calls.
    pub ms: f64,
}

fn motion_config_for(case: &MotionCase) -> UpscalerConfig {
    UpscalerConfig {
        filter: UpscaleFilter::Temporal,
        sharpness: case.sharpness,
        temporal: TemporalConfig {
            enabled: true,
            blend: 0.12,
            history_frames: 2,
        },
        ..UpscalerConfig::default()
    }
}

fn config_for(scenario: &Scenario) -> UpscalerConfig {
    UpscalerConfig {
        filter: scenario.filter,
        sharpness: scenario.sharpness,
        temporal: TemporalConfig {
            enabled: true,
            blend: 0.12,
            history_frames: 2,
        },
        ..UpscalerConfig::default()
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

/// Flicker metric: run three static-chart frames whose input carries a
/// deterministic one-LSB dither and average the absolute difference between
/// consecutive outputs. Lower is more stable.
pub fn flicker_metric(scenario: &Scenario) -> f64 {
    let cfg = config_for(scenario);
    let reference = render_reference(scenario.ref_dims);
    let low = downscale_box(&reference, scenario.low_dims);
    let mut up = Upscaler::new(&cfg, scenario.ref_dims);
    let mut prev: Option<Frame> = None;
    let mut sum = 0.0f64;
    let mut pairs = 0u32;
    for t in 0..TEMPORAL_FRAMES {
        let mut input = low.clone();
        add_dither(&mut input, t + 1);
        let out = up.upscale(&input).expect("flicker upscale");
        if let Some(p) = prev {
            sum += mean_abs_diff(&p, &out);
            pairs += 1;
        }
        prev = Some(out);
    }
    if pairs == 0 { 0.0 } else { sum / pairs as f64 }
}

/// Measure one scenario: build ground truth, downscale, upscale, compare.
pub fn run_scenario(scenario: &Scenario) -> ScenarioMetrics {
    let cfg = config_for(scenario);
    let references: Vec<Frame> = if scenario.temporal {
        (0..TEMPORAL_FRAMES)
            .map(|t| render_temporal_frame(scenario.ref_dims, t))
            .collect()
    } else {
        vec![render_reference(scenario.ref_dims)]
    };
    let inputs: Vec<Frame> = references
        .iter()
        .map(|r| downscale_box(r, scenario.low_dims))
        .collect();

    // Warm up plan tables, scratch and page faults, then reset the history so the
    // timed sequence still starts from a clean state.
    let mut upscaler = Upscaler::new(&cfg, scenario.ref_dims);
    for input in &inputs {
        let _ = upscaler.upscale(input).expect("warmup upscale");
    }
    upscaler.reset();

    // Time three passes over the sequence. A shared machine can stall any single
    // call, so per frame keep the fastest pass and then take the median across the
    // sequence. Only the first pass keeps its outputs (history must start clean for
    // every timed pass).
    let mut outputs = Vec::with_capacity(inputs.len());
    let mut samples: Vec<Vec<f64>> = (0..TIMING_ROUNDS)
        .map(|_| Vec::with_capacity(inputs.len()))
        .collect();
    for (round, round_samples) in samples.iter_mut().enumerate() {
        if round > 0 {
            upscaler.reset();
        }
        for input in &inputs {
            let start = Instant::now();
            let out = upscaler.upscale(input).expect("upscale");
            round_samples.push(start.elapsed().as_secs_f64() * 1000.0);
            if round == 0 {
                outputs.push(out);
            }
        }
    }
    let mut per_frame: Vec<f64> = (0..inputs.len())
        .map(|frame| {
            samples
                .iter()
                .map(|round| round[frame])
                .fold(f64::INFINITY, f64::min)
        })
        .collect();

    let mut psnrs = Vec::with_capacity(outputs.len());
    let mut ssims = Vec::with_capacity(outputs.len());
    let mut ratios = Vec::with_capacity(outputs.len());
    let mut worst_tile = PSNR_CAP_DB;
    for (out, reference) in outputs.iter().zip(references.iter()) {
        psnrs.push(psnr(out, reference));
        let (_, tile_min) = psnr_tiles(out, reference, 64);
        worst_tile = worst_tile.min(tile_min);
        ssims.push(ssim(out, reference));
        ratios.push(gradient_energy_ratio(out, reference));
    }
    let height = scenario.ref_dims.height;
    let mut band_psnr = Vec::with_capacity(CHART_BANDS);
    for band in 0..CHART_BANDS {
        let y0 = (band as u32 * height) / CHART_BANDS as u32;
        let y1 = ((band as u32 + 1) * height) / CHART_BANDS as u32;
        let mut values = Vec::with_capacity(outputs.len());
        for (out, reference) in outputs.iter().zip(references.iter()) {
            values.push(psnr_rows(out, reference, y0, y1));
        }
        band_psnr.push(mean(&values));
    }

    let checksum = outputs.iter().fold(0xcbf2_9ce4_8422_2325u64, |acc, out| {
        acc.rotate_left(7) ^ out.checksum()
    });

    ScenarioMetrics {
        frames: outputs.len() as u32,
        psnr_mean: mean(&psnrs),
        psnr_min: worst_tile,
        ssim: mean(&ssims),
        gradient_ratio: mean(&ratios),
        flicker: flicker_metric(scenario),
        ms: median(&mut per_frame),
        band_psnr,
        checksum,
    }
}

/// Mean absolute RGB error over the trail the disc leaves behind it.
///
/// The mask covers the two previous disc footprints (the object has moved away
/// from them) but excludes the current disc, so it measures exactly the stale
/// history a temporal filter can leave behind. Returns the sum of absolute
/// channel differences and the number of compared channels.
fn trail_ghosting(out: &Frame, reference: &Frame, speed_px: u32, frame: u32) -> (f64, u64) {
    let w = out.dims.width as usize;
    let h = out.dims.height as usize;
    let bpp = out.fmt.bytes_per_pixel();
    let channels = bpp.min(3);
    let min_side = out.dims.width.min(out.dims.height) as f32;
    let radius = 0.07 * min_side;
    let cx = 0.10 * out.dims.width as f32 + speed_px as f32 * frame as f32;
    let cy = 0.5 * out.dims.height as f32;
    let trail2 = (radius * 0.9) * (radius * 0.9);
    let keep_out2 = (radius * 1.1) * (radius * 1.1);
    let mut sum = 0.0f64;
    let mut count = 0u64;
    for y in 0..h {
        let dy = y as f32 + 0.5 - cy;
        for x in 0..w {
            let dx = x as f32 + 0.5 - cx;
            if dx * dx + dy * dy < keep_out2 {
                continue;
            }
            let mut on_trail = false;
            let mut k = 1u32;
            while k <= 2 && k <= frame {
                let prev = x as f32 + 0.5 - (cx - speed_px as f32 * k as f32);
                if prev * prev + dy * dy <= trail2 {
                    on_trail = true;
                    break;
                }
                k += 1;
            }
            if !on_trail {
                continue;
            }
            let base = (y * w + x) * bpp;
            for c in 0..channels {
                sum += (out.data[base + c] as f64 - reference.data[base + c] as f64).abs();
            }
            count += channels as u64;
        }
    }
    (sum, count)
}

/// Measure one moving-content case: render the ground-truth sequence, downscale
/// it, run it through a stateful temporal upscaler and score every output frame
/// against its own reference.
pub fn run_motion_case(case: &MotionCase) -> MotionMetrics {
    let config = motion_config_for(case);
    let references: Vec<Frame> = (0..MOTION_FRAMES)
        .map(|frame| render_motion_frame(case.ref_dims, case.speed_px, frame))
        .collect();
    let inputs: Vec<Frame> = references
        .iter()
        .map(|reference| downscale_box(reference, case.low_dims))
        .collect();
    let mut upscaler = Upscaler::new(&config, case.ref_dims);
    // Warm every input once so kernel tables and scratch are amortised, then
    // start the measured pass from the same temporal state the harness uses.
    for input in &inputs {
        let _ = upscaler.upscale(input).expect("motion warm-up upscale");
    }
    upscaler.reset();
    let mut outputs = Vec::with_capacity(inputs.len());
    let mut per_frame = Vec::with_capacity(inputs.len());
    for input in &inputs {
        let started = Instant::now();
        let out = upscaler.upscale(input).expect("motion upscale");
        per_frame.push(started.elapsed().as_secs_f64() * 1000.0);
        outputs.push(out);
    }
    let mut psnrs = Vec::with_capacity(outputs.len());
    let mut ghost_sum = 0.0f64;
    let mut ghost_count = 0u64;
    for (index, (out, reference)) in outputs.iter().zip(references.iter()).enumerate() {
        check_pair(out, reference);
        psnrs.push(psnr(out, reference));
        if index > 0 {
            let (sum, count) = trail_ghosting(out, reference, case.speed_px, index as u32);
            ghost_sum += sum;
            ghost_count += count;
        }
    }
    let psnr_min_sequence = psnrs.iter().copied().fold(f64::INFINITY, f64::min);
    MotionMetrics {
        frames: outputs.len() as u32,
        psnr_mean: mean(&psnrs),
        psnr_min_sequence,
        psnr_per_frame: psnrs,
        ghosting: if ghost_count == 0 {
            0.0
        } else {
            ghost_sum / ghost_count as f64
        },
        ms: median(&mut per_frame),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resize;

    #[test]
    fn ssim_of_identical_images_is_one() {
        let frame = render_reference(Dims::new(96, 72));
        let score = ssim(&frame, &frame);
        assert!((score - 1.0).abs() < 1.0e-6, "identical SSIM {score}");
        let second = render_temporal_frame(Dims::new(96, 72), 1);
        assert!(ssim(&second, &second) > 0.999_999);
    }

    #[test]
    fn box_downscale_then_nearest_round_trip_has_finite_psnr() {
        let reference = render_reference(Dims::new(128, 72));
        let low = downscale_box(&reference, Dims::new(64, 36));
        let up = resize(&low, reference.dims, UpscaleFilter::Nearest).expect("nearest");
        let score = psnr(&up, &reference);
        assert!(score.is_finite(), "PSNR must be finite, got {score}");
        assert!(
            score > 10.0 && score < PSNR_CAP_DB,
            "nearest round trip PSNR {score:.3} dB is implausible"
        );
        let ratio = gradient_energy_ratio(&up, &reference);
        assert!(ratio.is_finite() && ratio > 0.0, "gradient ratio {ratio}");
        let (mean_tile, worst_tile) = psnr_tiles(&up, &reference, 16);
        assert!(mean_tile.is_finite() && worst_tile <= mean_tile);
    }

    /// Sanity check at 2x: every filter reconstructs the box-downscaled chart
    /// within a couple of dB of nearest, and the higher-order kernels beat
    /// bilinear, which is the ordering real content shows.
    ///
    /// Nearest is a strong baseline at an exact 2x factor with a box downscaler:
    /// duplicating the block average is the area reconstruction, so it keeps hard
    /// edges and smooth ramps very accurate and even wins some bands (see the
    /// bands in out/upscale-quality.json). The test therefore bounds the gap to
    /// nearest instead of asserting a strict order over it, and checks the
    /// ordering the kernels do have to keep: sharper than bilinear.
    #[test]
    fn spatial_filters_reconstruct_near_nearest_at_2x() {
        let reference = render_reference(Dims::new(512, 288));
        let low = downscale_box(&reference, Dims::new(256, 144));
        let height = reference.dims.height;
        let band = |a: &Frame, b: &Frame, index: usize| {
            let y0 = (index as u32 * height) / CHART_BANDS as u32;
            let y1 = ((index as u32 + 1) * height) / CHART_BANDS as u32;
            psnr_rows(a, b, y0, y1)
        };
        let filters = [
            UpscaleFilter::Nearest,
            UpscaleFilter::Bilinear,
            UpscaleFilter::Bicubic,
            UpscaleFilter::Lanczos,
            UpscaleFilter::Adaptive,
        ];
        let mut nearest = 0.0;
        let mut bilinear = 0.0;
        let mut bilinear_ratio = 0.0;
        for filter in filters {
            let up = resize(&low, reference.dims, filter).expect("filter");
            let score = psnr(&up, &reference);
            let ratio = gradient_energy_ratio(&up, &reference);
            let bands: Vec<f64> = (0..CHART_BANDS)
                .map(|index| band(&up, &reference, index))
                .collect();
            assert!(
                score.is_finite() && ratio.is_finite(),
                "{filter:?} produced non-finite metrics: psnr {score} grad {ratio}"
            );
            match filter {
                UpscaleFilter::Nearest => nearest = score,
                UpscaleFilter::Bilinear => {
                    bilinear = score;
                    bilinear_ratio = ratio;
                }
                _ => {
                    assert!(
                        score + 0.2 >= bilinear,
                        "{filter:?} PSNR {score:.3} dB is worse than bilinear {bilinear:.3} dB"
                    );
                    assert!(
                        ratio > bilinear_ratio,
                        "{filter:?} gradient ratio {ratio:.3} is not above bilinear {bilinear_ratio:.3}"
                    );
                }
            }
            for (index, value) in bands.iter().enumerate() {
                assert!(
                    value.is_finite(),
                    "{filter:?} band {index} PSNR is not finite"
                );
            }
        }
        assert!(
            bilinear + 2.0 >= nearest,
            "bilinear {bilinear:.3} dB is far below nearest {nearest:.3} dB"
        );
    }

    #[test]
    fn motion_frame_translates_the_disc_only() {
        let dims = Dims::new(256, 144);
        let first = render_motion_frame(dims, 8, 0);
        let same = render_motion_frame(dims, 8, 0);
        assert_eq!(first.data, same.data, "same frame must render identically");
        let second = render_motion_frame(dims, 8, 1);
        // The background is static: a corner far from both discs is identical.
        let far = (20 * dims.width + 220) as usize * 4;
        assert_eq!(
            &first.data[far..far + 4],
            &second.data[far..far + 4],
            "static background moved"
        );
        let changed = first
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .zip(second.data.as_chunks::<4>().0)
            .filter(|(a, b)| a != b)
            .count();
        assert!(changed > 0, "disc did not move");
    }

    #[test]
    fn trail_ghosting_is_zero_between_identical_frames() {
        let reference = render_motion_frame(Dims::new(192, 108), 12, 2);
        let (sum, count) = trail_ghosting(&reference, &reference, 12, 2);
        assert_eq!(sum, 0.0);
        assert!(count > 0, "trail mask is empty");
    }

    #[test]
    fn motion_case_reports_finite_metrics() {
        let case = MotionCase {
            ref_dims: Dims::new(160, 90),
            low_dims: Dims::new(80, 45),
            speed_px: 12,
            sharpness: 0.0,
        };
        let metrics = run_motion_case(&case);
        assert_eq!(metrics.frames, MOTION_FRAMES);
        assert_eq!(metrics.psnr_per_frame.len() as u32, MOTION_FRAMES);
        for (index, value) in metrics.psnr_per_frame.iter().enumerate() {
            assert!(
                value.is_finite() && *value > 5.0,
                "frame {index} psnr {value}"
            );
        }
        assert!(metrics.psnr_mean.is_finite());
        assert!(metrics.psnr_min_sequence <= metrics.psnr_mean + 1e-9);
        assert!(metrics.ghosting.is_finite() && metrics.ghosting >= 0.0);
        assert!(metrics.ms.is_finite());
        assert!(case.describe().contains("speed 12"));
        assert_eq!(case.scale_factor(), 2);
    }
}
