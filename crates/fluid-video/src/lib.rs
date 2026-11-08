//! Video and image-sequence frame sources/sinks.
//!
//! Fluid never shells out to a codec library: it speaks the simple, lossless
//! formats that any tool can convert to/from (y4m, PNG sequences) plus raw
//! pipes. That keeps the release binary small and dependency-light while
//! still covering the offline (video enhancement) use case:
//!
//! - fluid video --input clip.y4m --output out.y4m --multiplier 4 --scale 2
//! - ffmpeg -i clip.mp4 clip.y4m   (conversion done outside Fluid)
//!
//! Streams can also flow through pipes: [Y4mReader::from_reader] and
//! [Y4mReader::new_stdin] accept any boxed reader, while
//! [Y4mWriter::from_writer] and [Y4mWriter::new_stdout] accept any boxed
//! writer. That is what lets the CLI run as the middle of an ffmpeg pipeline
//! (ffmpeg | fluid video -i - -o - | ffmpeg) without linking a codec itself.
//!
//! # What lives here
//!
//! * [Y4mReader] / [Y4mWriter] - the unfiltered YUV4MPEG2 container, which is
//!   a one-line header followed by a FRAME marker line and the raw Y, Cb and
//!   Cr planes.
//! * [ImageSequence] / [PngWriter] - directories of numbered still images
//!   (png/jpeg/bmp) for frame-interpolation experiments.
//! * [synthetic_pattern] / [SyntheticSource] - a deterministic test clip with
//!   a precisely known per-frame displacement, used by tests, demos and
//!   benchmarks to measure interpolation quality.
//! * [AviWriter] - a self-contained Motion-JPEG AVI file (one baseline JPEG
//!   per frame plus a standard RIFF index) for machines that have no ffmpeg.
//! * [AviReader] - reads those MJPEG AVI files back (RIFF header parse plus
//!   JPEG chunks located through idx1 or a sequential movi walk), so the video
//!   pipeline accepts .avi input without ffmpeg too.
//!
//! # Colour conversion
//!
//! y4m stores BT.709 limited-range YCbCr (luma 16..235, chroma 16..240 with
//! centre 128). Conversion both ways uses Q16 fixed point integer arithmetic
//! with chroma coefficient sets whose sum is zero, so neutral grey maps to
//! exactly 128 and a round trip through the container stays within +-2 code
//! values per channel (enforced by the round-trip tests). Reading 4:2:0 or
//! 4:2:2 replicates the coarser chroma sample over the luma grid; writing
//! averages the RGB of each chroma footprint before converting, which is the
//! cheapest correct subsampling filter.
//!
//! 10-bit y4m input (C420p10/C422p10/C444p10/mono10) is accepted and narrowed
//! to 8 bits per sample before conversion: samples are stored little-endian in
//! the stream and get their low 2 bits dropped with rounding (0 -> 0,
//! 512 -> 128, 1023 -> 255). There is no HDR tone mapping - a 10-bit HDR
//! source therefore loses highlight detail. Writers always emit 8-bit streams.
//!
//! # Performance
//!
//! Every hot path is a row-wise loop over a pre-allocated plane buffer; no
//! allocation happens per pixel. A reader/writer allocates exactly one output
//! [Frame] (which owns its pixels) per frame plus fixed scratch buffers, all
//! conversion helpers are #[inline], and there is no trig or floating point
//! in the per-pixel paths.

#![forbid(unsafe_code)]

mod avi;
mod avi_read;

pub use avi::{AviHeader, AviWriter, DEFAULT_JPEG_QUALITY};
pub use avi_read::AviReader;

use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use fluid_core::{Dims, FluidError, Frame, FrameSink, FrameSource, LumaPlane, PixelFormat, Result};

/// Magic that starts every y4m stream.
const Y4M_MAGIC: &str = "YUV4MPEG2";

/// Chroma subsampling of a y4m stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Y4mFormat {
    C420,
    C422,
    C444,
    Mono,
}

impl Y4mFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Y4mFormat::C420 => "420jpeg",
            Y4mFormat::C422 => "422",
            Y4mFormat::C444 => "444",
            Y4mFormat::Mono => "mono",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "420" | "420jpeg" | "420mpeg2" | "420paldv" => Some(Y4mFormat::C420),
            "422" => Some(Y4mFormat::C422),
            "444" | "444alpha" => Some(Y4mFormat::C444),
            "mono" => Some(Y4mFormat::Mono),
            _ => None,
        }
    }

    /// Horizontal and vertical chroma subsampling factors.
    pub fn subsampling(&self) -> (u32, u32) {
        match self {
            Y4mFormat::C420 => (2, 2),
            Y4mFormat::C422 => (2, 1),
            Y4mFormat::C444 | Y4mFormat::Mono => (1, 1),
        }
    }
}

/// Parsed y4m stream header.
#[derive(Clone, Debug, PartialEq)]
pub struct Y4mHeader {
    pub dims: Dims,
    pub fps_num: u32,
    pub fps_den: u32,
    pub format: Y4mFormat,
    /// Bits per sample as stored in the stream: 8 for plain y4m, 10 for the
    /// C420p10/C422p10/C444p10/mono10 variants. Private so headers can only be
    /// built through [Y4mHeader::new] or by parsing a stream; frames handed out
    /// by readers are always 8-bit.
    bit_depth: u32,
}

impl Y4mHeader {
    /// Build a header with 4:2:0 chroma, snapping the frame rate to a rational
    /// (60.0 -> 60/1, 29.97 -> 30000/1001, 7.5 -> 15/2).
    pub fn new(dims: Dims, fps: f32) -> Self {
        let (fps_num, fps_den) = rational_fps(fps);
        Self {
            dims,
            fps_num,
            fps_den,
            format: Y4mFormat::C420,
            bit_depth: 8,
        }
    }

    pub fn fps(&self) -> f32 {
        if self.fps_den == 0 {
            0.0
        } else {
            self.fps_num as f32 / self.fps_den as f32
        }
    }

    /// Bits per sample as stored in the stream: 8 for plain y4m, 10 for
    /// C420p10/C422p10/C444p10/mono10. Readers narrow 10-bit input to 8 bits,
    /// so frames are always 8-bit.
    pub fn bit_depth(&self) -> u32 {
        self.bit_depth
    }

    /// Number of samples (luma plus chroma) in one raw frame.
    pub fn sample_count(&self) -> usize {
        let (sx, sy) = self.format.subsampling();
        let w = self.dims.width as usize;
        let h = self.dims.height as usize;
        let y = w * h;
        if self.format == Y4mFormat::Mono {
            y
        } else {
            y + (w / sx as usize) * (h / sy as usize) * 2
        }
    }

    /// Bytes per sample on disk: 2 for 10-bit streams, 1 otherwise.
    fn bytes_per_sample(&self) -> usize {
        if self.bit_depth > 8 { 2 } else { 1 }
    }

    /// Number of bytes occupied by one raw frame on disk (planes only, no
    /// FRAME marker). For 10-bit headers this is twice [Self::sample_count].
    pub fn frame_bytes(&self) -> usize {
        self.sample_count() * self.bytes_per_sample()
    }

    /// The `C` token describing this header, e.g. "420jpeg" or "420p10".
    pub fn chroma_tag(&self) -> &'static str {
        match (self.format, self.bit_depth) {
            (Y4mFormat::C420, 10) => "420p10",
            (Y4mFormat::C422, 10) => "422p10",
            (Y4mFormat::C444, 10) => "444p10",
            (Y4mFormat::Mono, 10) => "mono10",
            _ => self.format.as_str(),
        }
    }
}

impl std::fmt::Display for Y4mHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}x{} @{:.3}fps {}",
            self.dims.width,
            self.dims.height,
            self.fps(),
            self.chroma_tag()
        )
    }
}

// ---------------------------------------------------------------------------
// Frame rate helpers
// ---------------------------------------------------------------------------

/// Well known broadcast/integer frame rates, checked before the generic
/// rational approximation so 29.97 lands on 30000/1001 rather than 2997/100.
const COMMON_FPS: &[(f32, u32, u32)] = &[
    (23.976, 24000, 1001),
    (23.98, 24000, 1001),
    (29.97, 30000, 1001),
    (29.976, 30000, 1001),
    (59.94, 60000, 1001),
    (59.976, 60000, 1001),
    (119.88, 120000, 1001),
    (24.0, 24, 1),
    (25.0, 25, 1),
    (30.0, 30, 1),
    (48.0, 48, 1),
    (50.0, 50, 1),
    (60.0, 60, 1),
    (90.0, 90, 1),
    (100.0, 100, 1),
    (120.0, 120, 1),
    (144.0, 144, 1),
    (165.0, 165, 1),
    (240.0, 240, 1),
];

/// Rational representation of a frame rate. Non-positive / non-finite rates
/// become 0/1 (i.e. "unknown"), which Y4mHeader::fps reports as 0.0.
fn rational_fps(fps: f32) -> (u32, u32) {
    if !fps.is_finite() || fps <= 0.0 {
        return (0, 1);
    }
    for &(rate, num, den) in COMMON_FPS {
        if (fps - rate).abs() <= 0.01 {
            return (num, den);
        }
    }
    if fps > 1000.0 {
        return (fps.round() as u32, 1);
    }
    let x = fps as f64;
    let mut best = (x.round().max(1.0) as u32, 1u32);
    let mut best_err = (x - best.0 as f64).abs();
    for den in 2..=1000u32 {
        let num = (x * den as f64).round() as u32;
        if num == 0 {
            continue;
        }
        let err = (x - num as f64 / den as f64).abs();
        if err < best_err - 1e-9 {
            best = (num, den);
            best_err = err;
        }
    }
    let g = gcd(best.0, best.1);
    (best.0 / g, best.1 / g)
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

// ---------------------------------------------------------------------------
// BT.709 limited-range colour conversion (Q16 fixed point)
// ---------------------------------------------------------------------------
//
// Rgb8 -> YCbCr (limited):
//   Y  = 16  + (219/255) * (0.2126 R + 0.7152 G + 0.0722 B)
//   Cb = 128 + (224/255) * (-0.1146 R - 0.3854 G + 0.5 B)
//   Cr = 128 + (224/255) * (0.5 R - 0.4542 G - 0.0458 B)
//
// The chroma coefficient sets sum to exactly zero, so R = G = B (neutral grey)
// maps to Cb = Cr = 128 for every level, which keeps the 4:2:0 round trip
// honest on grey ramps.

/// 219/255 * 0.2126 * 65536
const Y_R: i32 = 11966;
/// 219/255 * 0.7152 * 65536
const Y_G: i32 = 40252;
/// 219/255 * 0.0722 * 65536
const Y_B: i32 = 4066;
/// 224/255 * -0.1146 * 65536
const CB_R: i32 = -6596;
/// 224/255 * -0.3854 * 65536
const CB_G: i32 = -22182;
/// 224/255 * 0.5 * 65536
const CB_B: i32 = 28778;
/// 224/255 * 0.5 * 65536
const CR_R: i32 = 28785;
/// 224/255 * -0.4542 * 65536
const CR_G: i32 = -26148;
/// 224/255 * -0.0458 * 65536
const CR_B: i32 = -2637;
/// 255/219 * 65536 - limited luma to 0..255 luma.
const Y_SCALE: i32 = 76309;
/// 255/224 * 1.8556 * 65536 - Cb to blue.
const B_CB: i32 = 138446;
/// 255/224 * 1.5748 * 65536 - Cr to red.
const R_CR: i32 = 117483;
/// 255/224 * -0.1873 * 65536 - Cb to green.
const G_CB: i32 = -13973;
/// 255/224 * -0.4681 * 65536 - Cr to green.
const G_CR: i32 = -34925;

#[inline]
fn clip_u8(v: i32) -> u8 {
    if v < 0 {
        0
    } else if v > 255 {
        255
    } else {
        v as u8
    }
}

/// Limited-range BT.709 luma for one RGB pixel.
#[inline]
fn rgb_to_y_limited(r: u8, g: u8, b: u8) -> u8 {
    let y = Y_R * r as i32 + Y_G * g as i32 + Y_B * b as i32;
    16 + ((y + 32768) >> 16) as u8
}

/// Limited-range BT.709 chroma for one RGB pixel (or an averaged footprint).
#[inline]
fn rgb_to_chroma(r: u8, g: u8, b: u8) -> (u8, u8) {
    let (ri, gi, bi) = (r as i32, g as i32, b as i32);
    let cb = 128 + ((CB_R * ri + CB_G * gi + CB_B * bi + 32768) >> 16);
    let cr = 128 + ((CR_R * ri + CR_G * gi + CR_B * bi + 32768) >> 16);
    (clip_u8(cb), clip_u8(cr))
}

/// One YCbCr sample -> RGB, BT.709 limited range.
#[inline]
fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let yv = (y as i32 - 16) * Y_SCALE;
    let cbv = cb as i32 - 128;
    let crv = cr as i32 - 128;
    let r = (yv + R_CR * crv + 32768) >> 16;
    let g = (yv + G_CB * cbv + G_CR * crv + 32768) >> 16;
    let b = (yv + B_CB * cbv + 32768) >> 16;
    (clip_u8(r), clip_u8(g), clip_u8(b))
}

/// Limited-range luma -> 0..255 grey (used for mono y4m).
#[inline]
fn limited_luma_to_grey(y: u8) -> u8 {
    clip_u8(((y as i32 - 16) * Y_SCALE + 32768) >> 16)
}

/// Average two RGB pixels (rounded).
#[inline]
fn avg2(a: &[u8], b: &[u8]) -> (u8, u8, u8) {
    (
        ((a[0] as u16 + b[0] as u16 + 1) >> 1) as u8,
        ((a[1] as u16 + b[1] as u16 + 1) >> 1) as u8,
        ((a[2] as u16 + b[2] as u16 + 1) >> 1) as u8,
    )
}

/// Average four RGB pixels (rounded).
#[inline]
fn avg4(a: &[u8], b: &[u8], c: &[u8], d: &[u8]) -> (u8, u8, u8) {
    (
        ((a[0] as u16 + b[0] as u16 + c[0] as u16 + d[0] as u16 + 2) >> 2) as u8,
        ((a[1] as u16 + b[1] as u16 + c[1] as u16 + d[1] as u16 + 2) >> 2) as u8,
        ((a[2] as u16 + b[2] as u16 + c[2] as u16 + d[2] as u16 + 2) >> 2) as u8,
    )
}

// ---------------------------------------------------------------------------
// y4m header parsing
// ---------------------------------------------------------------------------

fn parse_u32(token: &str, what: &str) -> Result<u32> {
    token
        .parse::<u32>()
        .map_err(|_| FluidError::format(format!("invalid y4m {what} {token:?}")))
}

/// Map a y4m `C` token to a chroma format and bit depth.
///
/// Only 10-bit variants are recognised beyond the plain 8-bit tags
/// (C420p10/C422p10/C444p10/mono10); other depths are unsupported.
fn parse_chroma(value: &str) -> Option<(Y4mFormat, u32)> {
    let tag = value.to_ascii_lowercase();
    match tag.as_str() {
        "420p10" => Some((Y4mFormat::C420, 10)),
        "422p10" => Some((Y4mFormat::C422, 10)),
        "444p10" => Some((Y4mFormat::C444, 10)),
        "mono10" => Some((Y4mFormat::Mono, 10)),
        _ => Y4mFormat::parse(&tag).map(|format| (format, 8)),
    }
}

/// Parse a single YUV4MPEG2 header line (a trailing newline may be present).
fn parse_header_line(line: &str) -> Result<Y4mHeader> {
    let line = line.trim_end_matches(['\n', '\r']);
    let mut tokens = line.split_ascii_whitespace();
    let magic = tokens
        .next()
        .ok_or_else(|| FluidError::format("empty y4m header"))?;
    if magic != Y4M_MAGIC {
        return Err(FluidError::format(format!(
            "not a y4m stream: expected {Y4M_MAGIC}, got {magic:?}"
        )));
    }
    let mut width = None;
    let mut height = None;
    let mut fps_num = 0u32;
    let mut fps_den = 1u32;
    let mut format = Y4mFormat::C420;
    let mut bit_depth = 8u32;
    for token in tokens {
        let bytes = token.as_bytes();
        if bytes.len() < 2 || !bytes[0].is_ascii_uppercase() {
            continue;
        }
        let (tag, value) = token.split_at(1);
        match tag {
            "W" => width = Some(parse_u32(value, "width")?),
            "H" => height = Some(parse_u32(value, "height")?),
            "F" => {
                let (num, den) = value.split_once(':').ok_or_else(|| {
                    FluidError::format(format!("malformed y4m frame rate token {token:?}"))
                })?;
                fps_num = parse_u32(num, "frame rate numerator")?;
                fps_den = parse_u32(den, "frame rate denominator")?;
                if fps_den == 0 {
                    return Err(FluidError::format("y4m frame rate denominator is zero"));
                }
            }
            "C" => {
                if value.eq_ignore_ascii_case("444alpha") {
                    return Err(FluidError::format("y4m 444alpha streams are not supported"));
                }
                let parsed = parse_chroma(value).ok_or_else(|| {
                    FluidError::format(format!("unsupported y4m chroma format {value:?}"))
                })?;
                format = parsed.0;
                bit_depth = parsed.1;
            }
            // "I" (interlacing) and "A" (pixel aspect) do not change the frame
            // layout we care about. Unknown tags (including "X" comments) are
            // ignored for forward compatibility.
            _ => {}
        }
    }
    let dims = Dims::new(
        width.ok_or_else(|| FluidError::format("y4m header is missing the W (width) token"))?,
        height.ok_or_else(|| FluidError::format("y4m header is missing the H (height) token"))?,
    );
    if dims.is_empty() {
        return Err(FluidError::format("y4m header has zero width or height"));
    }
    Ok(Y4mHeader {
        dims,
        fps_num,
        fps_den,
        format,
        bit_depth,
    })
}

fn read_header(reader: &mut impl BufRead) -> Result<Y4mHeader> {
    let mut line = Vec::with_capacity(128);
    let n = reader.read_until(b'\n', &mut line)?;
    if n == 0 {
        return Err(FluidError::format("empty y4m stream (no YUV4MPEG2 header)"));
    }
    let text = std::str::from_utf8(&line)
        .map_err(|_| FluidError::format("y4m header is not valid ASCII/UTF-8"))?;
    parse_header_line(text)
}

/// Map a short read of frame data to InvalidFormat: a stream that ends inside a
/// frame is truncated, not a generic IO failure.
fn map_truncated_frame(err: std::io::Error) -> FluidError {
    match err.kind() {
        std::io::ErrorKind::UnexpectedEof => {
            FluidError::format("truncated y4m frame data (plane bytes missing)")
        }
        _ => FluidError::from(err),
    }
}

// ---------------------------------------------------------------------------
// 10-bit input narrowing
// ---------------------------------------------------------------------------

/// Narrow one 10-bit sample to 8 bits by dropping the low 2 bits, rounding half
/// up: 0 -> 0, 512 -> 128, 1023 -> 255.
#[inline]
fn ten_to_eight(sample: u16) -> u8 {
    (((sample as u32 + 2) >> 2).min(255)) as u8
}

/// Narrow a frame of packed little-endian 10-bit samples (2 bytes each) to one
/// 8-bit sample per value. Plane order and subsampling are unchanged, so the
/// result feeds the normal 8-bit conversion path.
fn ten_bit_planes_to_eight(raw: &[u8], out: &mut [u8]) {
    debug_assert_eq!(raw.len(), out.len() * 2);
    for (dst, src) in out.iter_mut().zip(raw.as_chunks::<2>().0.iter()) {
        *dst = ten_to_eight(u16::from_le_bytes(*src));
    }
}

// ---------------------------------------------------------------------------
// y4m plane <-> RGB conversion
// ---------------------------------------------------------------------------

/// Convert packed y4m planes into an Rgb8 buffer.
fn yuv_to_rgb(header: &Y4mHeader, planes: &[u8], out: &mut [u8]) {
    let dims = header.dims;
    let w = dims.width as usize;
    let h = dims.height as usize;
    let ylen = w * h;
    let stride = w * 3;
    debug_assert_eq!(planes.len(), header.sample_count());
    debug_assert!(out.len() >= stride * h);

    if header.format == Y4mFormat::Mono {
        for y in 0..h {
            let yrow = &planes[y * w..y * w + w];
            let orow = &mut out[y * stride..y * stride + stride];
            for (px, &yv) in orow.as_chunks_mut::<3>().0.iter_mut().zip(yrow.iter()) {
                let l = limited_luma_to_grey(yv);
                px[0] = l;
                px[1] = l;
                px[2] = l;
            }
        }
        return;
    }

    let (sx, sy) = header.format.subsampling();
    let cw = w / sx as usize;
    let ch = h / sy as usize;
    if cw == 0 || ch == 0 {
        // Degenerate header (e.g. a one pixel wide 4:2:0 stream): no chroma
        // samples exist, so fall back to the luma channel as grey.
        for y in 0..h {
            let yrow = &planes[y * w..y * w + w];
            let orow = &mut out[y * stride..y * stride + stride];
            for (px, &yv) in orow.as_chunks_mut::<3>().0.iter_mut().zip(yrow.iter()) {
                let l = limited_luma_to_grey(yv);
                px[0] = l;
                px[1] = l;
                px[2] = l;
            }
        }
        return;
    }

    let uplane = &planes[ylen..ylen + cw * ch];
    let vplane = &planes[ylen + cw * ch..ylen + cw * ch * 2];
    let sx_usize = sx as usize;
    for y in 0..h {
        let cy = (y / sy as usize).min(ch - 1);
        let yrow = &planes[y * w..y * w + w];
        let crow = cy * cw;
        let urow = &uplane[crow..crow + cw];
        let vrow = &vplane[crow..crow + cw];
        let orow = &mut out[y * stride..y * stride + stride];
        for (x, (px, &yv)) in orow
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(yrow.iter())
            .enumerate()
        {
            let cx = (x / sx_usize).min(cw - 1);
            let (r, g, b) = ycbcr_to_rgb(yv, urow[cx], vrow[cx]);
            px[0] = r;
            px[1] = g;
            px[2] = b;
        }
    }
}

/// Convert an Rgb8 buffer into packed y4m planes.
fn rgb_to_yuv(header: &Y4mHeader, rgb: &[u8], out: &mut [u8]) {
    let dims = header.dims;
    let w = dims.width as usize;
    let h = dims.height as usize;
    let ylen = w * h;
    let stride = w * 3;
    debug_assert_eq!(out.len(), header.sample_count());
    debug_assert!(rgb.len() >= stride * h);

    // Luma first: it is the largest plane and this loop is the hot one.
    for y in 0..h {
        let src = &rgb[y * stride..y * stride + stride];
        let dst = &mut out[y * w..y * w + w];
        for (px, dv) in src.as_chunks::<3>().0.iter().zip(dst.iter_mut()) {
            *dv = rgb_to_y_limited(px[0], px[1], px[2]);
        }
    }
    if header.format == Y4mFormat::Mono {
        return;
    }

    let (sx, sy) = header.format.subsampling();
    let cw = w / sx as usize;
    let ch = h / sy as usize;
    let u_off = ylen;
    let v_off = ylen + cw * ch;
    match header.format {
        Y4mFormat::C444 => {
            for i in 0..ylen {
                let px = &rgb[i * 3..i * 3 + 3];
                let (cb, cr) = rgb_to_chroma(px[0], px[1], px[2]);
                out[u_off + i] = cb;
                out[v_off + i] = cr;
            }
        }
        Y4mFormat::C422 => {
            for y in 0..h {
                let row = &rgb[y * stride..y * stride + stride];
                for cx in 0..cw {
                    let (r, g, b) = avg2(&row[cx * 6..cx * 6 + 3], &row[cx * 6 + 3..cx * 6 + 6]);
                    let (cb, cr) = rgb_to_chroma(r, g, b);
                    out[u_off + y * cw + cx] = cb;
                    out[v_off + y * cw + cx] = cr;
                }
            }
        }
        Y4mFormat::C420 => {
            for cy in 0..ch {
                let y0 = cy * 2;
                let row0 = &rgb[y0 * stride..y0 * stride + stride];
                let row1 = &rgb[(y0 + 1) * stride..(y0 + 1) * stride + stride];
                for cx in 0..cw {
                    let x = cx * 6;
                    let (r, g, b) = avg4(
                        &row0[x..x + 3],
                        &row0[x + 3..x + 6],
                        &row1[x..x + 3],
                        &row1[x + 3..x + 6],
                    );
                    let (cb, cr) = rgb_to_chroma(r, g, b);
                    out[u_off + cy * cw + cx] = cb;
                    out[v_off + cy * cw + cx] = cr;
                }
            }
        }
        Y4mFormat::Mono => unreachable!(),
    }
}

/// Presentation timestamp of frame "index" on a stream running at "fps".
#[inline]
fn pts_for(index: u64, fps: f32) -> i64 {
    if fps > 0.0 && fps.is_finite() {
        let ns = index as f64 * 1_000_000_000.0 / fps as f64;
        if ns <= i64::MAX as f64 { ns as i64 } else { 0 }
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// y4m reader
// ---------------------------------------------------------------------------

/// Reads y4m streams into Rgb8 frames.
///
/// The byte source is any Read + Send stream: a file, a pipe, or an in-memory
/// buffer. Use [Y4mReader::open] for paths, [Y4mReader::from_reader] for a
/// boxed reader and [Y4mReader::new_stdin] for the process standard input.
pub struct Y4mReader {
    header: Y4mHeader,
    inner: BufReader<Box<dyn Read + Send>>,
    index: u64,
    finished: bool,
    /// Reusable packed 8-bit plane buffer, header.sample_count() long.
    planes: Vec<u8>,
    /// Reusable on-disk frame buffer for 10-bit input (2 bytes per sample);
    /// empty for 8-bit streams.
    raw: Vec<u8>,
    /// Reusable line buffer for the per-frame FRAME marker.
    marker: Vec<u8>,
}

impl Y4mReader {
    /// Open a y4m file on disk.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        Self::from_reader(Box::new(file))
    }

    /// Read a y4m stream from an arbitrary byte source (pipe, socket, memory).
    ///
    /// The y4m header is parsed immediately, so malformed streams fail here
    /// rather than on the first next_frame call. The reader must be Send
    /// because FrameSource is.
    pub fn from_reader(reader: Box<dyn Read + Send>) -> Result<Self> {
        let mut inner = BufReader::with_capacity(1 << 16, reader);
        let header = read_header(&mut inner)?;
        let planes = vec![0u8; header.sample_count()];
        let raw = if header.bytes_per_sample() == 2 {
            vec![0u8; header.frame_bytes()]
        } else {
            Vec::new()
        };
        Ok(Self {
            header,
            inner,
            index: 0,
            finished: false,
            planes,
            raw,
            marker: Vec::with_capacity(64),
        })
    }

    /// Read a y4m stream from standard input, e.g. an ffmpeg pipe.
    pub fn new_stdin() -> Result<Self> {
        Self::from_reader(Box::new(std::io::stdin()))
    }

    pub fn header(&self) -> &Y4mHeader {
        &self.header
    }
}

impl FrameSource for Y4mReader {
    fn dims(&self) -> Dims {
        self.header.dims
    }

    fn fps(&self) -> f32 {
        self.header.fps()
    }

    fn backend_name(&self) -> &'static str {
        "y4m-reader"
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        if self.finished {
            return Ok(None);
        }
        let Self {
            header,
            inner,
            index,
            finished,
            planes,
            raw,
            marker,
        } = self;

        marker.clear();
        let n = inner.read_until(b'\n', marker)?;
        if n == 0 {
            *finished = true;
            return Ok(None);
        }
        if marker.last() != Some(&b'\n') {
            *finished = true;
            return Err(FluidError::format("truncated y4m frame marker"));
        }
        let text = std::str::from_utf8(marker)
            .map_err(|_| FluidError::format("y4m frame marker is not ASCII"))?;
        let text = text.trim_end_matches(['\n', '\r']);
        let rest = text
            .strip_prefix("FRAME")
            .ok_or_else(|| FluidError::format(format!("expected a FRAME marker, got {text:?}")))?;
        if !rest.is_empty() && !rest.starts_with(' ') {
            return Err(FluidError::format(format!(
                "expected a FRAME marker, got {text:?}"
            )));
        }

        if raw.is_empty() {
            // 8-bit stream: the on-disk frame already is the sample buffer.
            inner.read_exact(planes).map_err(map_truncated_frame)?;
        } else {
            inner.read_exact(raw).map_err(map_truncated_frame)?;
            ten_bit_planes_to_eight(raw, planes);
        }

        let mut rgb = vec![0u8; header.dims.area() as usize * 3];
        yuv_to_rgb(header, planes, &mut rgb);
        let mut frame = Frame::new(header.dims, PixelFormat::Rgb8, rgb)?;
        frame.index = *index;
        frame.pts_ns = pts_for(*index, header.fps());
        *index += 1;
        Ok(Some(frame))
    }
}

// ---------------------------------------------------------------------------
// y4m writer
// ---------------------------------------------------------------------------

/// Writes Rgb8 frames as a y4m stream.
///
/// The destination is any Write + Send sink. Use [Y4mWriter::create] for
/// paths, [Y4mWriter::from_writer] for a boxed writer and
/// [Y4mWriter::new_stdout] for the process standard output (for example piping
/// into an ffmpeg encoder).
pub struct Y4mWriter {
    header: Y4mHeader,
    inner: BufWriter<Box<dyn Write + Send>>,
    frames: u64,
    /// Reusable packed-plane buffer, header.frame_bytes() long.
    scratch: Vec<u8>,
}

/// Reject headers a writer cannot represent (non-8-bit, empty or odd
/// subsampled dims).
fn validate_header_for_write(header: &Y4mHeader) -> Result<()> {
    let dims = header.dims;
    if header.bit_depth != 8 {
        return Err(FluidError::unsupported(format!(
            "writing {}-bit y4m is not supported; Fluid writes 8-bit y4m only",
            header.bit_depth
        )));
    }
    if dims.is_empty() {
        return Err(FluidError::invalid(
            "cannot write a y4m stream with empty dimensions",
        ));
    }
    let (sx, sy) = header.format.subsampling();
    if sx == 2 && !dims.width.is_multiple_of(2) {
        return Err(FluidError::invalid(format!(
            "{} y4m requires an even width, got {dims}",
            header.format.as_str()
        )));
    }
    if sy == 2 && !dims.height.is_multiple_of(2) {
        return Err(FluidError::invalid(format!(
            "{} y4m requires an even height, got {dims}",
            header.format.as_str()
        )));
    }
    Ok(())
}

impl Y4mWriter {
    /// Create a y4m file on disk.
    pub fn create(path: impl AsRef<Path>, header: Y4mHeader) -> Result<Self> {
        validate_header_for_write(&header)?;
        let file = File::create(path.as_ref())?;
        Self::from_writer(Box::new(file), header)
    }

    /// Write a y4m stream to an arbitrary byte sink (pipe, socket, memory).
    ///
    /// The header line is written and flushed immediately, so a downstream
    /// reader can parse it before the first frame arrives.
    pub fn from_writer(writer: Box<dyn Write + Send>, header: Y4mHeader) -> Result<Self> {
        validate_header_for_write(&header)?;
        let dims = header.dims;
        let mut inner = BufWriter::with_capacity(1 << 16, writer);
        let (fps_num, fps_den) = if header.fps_den == 0 {
            (0, 1)
        } else {
            (header.fps_num, header.fps_den)
        };
        writeln!(
            inner,
            "{Y4M_MAGIC} W{} H{} F{}:{} Ip A1:1 C{}",
            dims.width,
            dims.height,
            fps_num,
            fps_den,
            header.chroma_tag()
        )?;
        inner.flush()?;
        let scratch = vec![0u8; header.frame_bytes()];
        Ok(Self {
            header,
            inner,
            frames: 0,
            scratch,
        })
    }

    /// Write a y4m stream to standard output, e.g. into an ffmpeg pipe.
    pub fn new_stdout(header: Y4mHeader) -> Result<Self> {
        Self::from_writer(Box::new(std::io::stdout()), header)
    }

    pub fn frames(&self) -> u64 {
        self.frames
    }

    pub fn finish(&mut self) -> Result<()> {
        self.inner.flush()?;
        Ok(())
    }
}

impl FrameSink for Y4mWriter {
    fn dims(&self) -> Dims {
        self.header.dims
    }

    fn backend_name(&self) -> &'static str {
        "y4m-writer"
    }

    fn present(&mut self, frame: &Frame) -> Result<()> {
        let Self {
            header,
            inner,
            frames,
            scratch,
        } = self;
        if frame.dims != header.dims {
            return Err(FluidError::invalid(format!(
                "y4m writer expects {} frames, got {}",
                header.dims, frame.dims
            )));
        }
        let converted;
        let rgb: &[u8] = if frame.fmt == PixelFormat::Rgb8 {
            &frame.data
        } else {
            converted = frame.convert(PixelFormat::Rgb8);
            &converted.data
        };
        rgb_to_yuv(header, rgb, scratch);
        inner.write_all(b"FRAME\n")?;
        inner.write_all(scratch)?;
        *frames += 1;
        Ok(())
    }

    fn close(&mut self) {
        // The pipeline calls close() when a run ends; flush so a pipe receives
        // every buffered frame even if the caller forgets finish().
        let _ = self.finish();
    }
}

// ---------------------------------------------------------------------------
// Image sequence reader
// ---------------------------------------------------------------------------

fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "bmp"
            )
        })
        .unwrap_or(false)
}

/// Read a digit run as an integer, ignoring leading zeros. Returns the value
/// and the index one past the run.
fn read_number(bytes: &[u8], mut i: usize) -> (u64, usize) {
    while i < bytes.len() && bytes[i] == b'0' {
        i += 1;
    }
    let start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if start == i {
        return (0, i);
    }
    let mut v: u64 = 0;
    for &d in &bytes[start..i] {
        v = v.saturating_mul(10).saturating_add((d - b'0') as u64);
    }
    (v, i)
}

/// Natural ("human") ordering: digit runs compare by numeric value, everything
/// else compares case-insensitively byte by byte. shot2.png sorts before
/// shot10.png.
fn natural_cmp(a: &str, b: &str) -> Ordering {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        if a[i].is_ascii_digit() && b[j].is_ascii_digit() {
            let (na, ni) = read_number(a, i);
            let (nb, nj) = read_number(b, j);
            match na.cmp(&nb) {
                Ordering::Equal => {
                    i = ni;
                    j = nj;
                }
                other => return other,
            }
        } else {
            let ca = a[i].to_ascii_lowercase();
            let cb = b[j].to_ascii_lowercase();
            if ca != cb {
                return ca.cmp(&cb);
            }
            i += 1;
            j += 1;
        }
    }
    (a.len() - i).cmp(&(b.len() - j))
}

/// Reads a directory of PNG/JPEG/BMP files in natural name order.
pub struct ImageSequence {
    files: Vec<PathBuf>,
    index: usize,
    dims: Dims,
    fps: f32,
}

impl ImageSequence {
    pub fn open_dir(dir: impl AsRef<Path>, fps: f32) -> Result<Self> {
        let dir = dir.as_ref();
        let mut files = Vec::new();
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && is_image_path(&path) {
                files.push(path);
            }
        }
        if files.is_empty() {
            return Err(FluidError::invalid(format!(
                "no png/jpeg/bmp images found in {}",
                dir.display()
            )));
        }
        files.sort_by(|a, b| {
            let an = a
                .file_name()
                .map(|s| s.to_string_lossy())
                .unwrap_or_default();
            let bn = b
                .file_name()
                .map(|s| s.to_string_lossy())
                .unwrap_or_default();
            natural_cmp(&an, &bn).then_with(|| a.cmp(b))
        });
        let dims = decode_image(&files[0])?.0;
        Ok(Self {
            files,
            index: 0,
            dims,
            fps,
        })
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    pub fn dims(&self) -> Dims {
        self.dims
    }
}

/// Decode one image file into (dims, Rgb8 bytes).
fn decode_image(path: &Path) -> Result<(Dims, Vec<u8>)> {
    let img = image::open(path)
        .map_err(|e| FluidError::format(format!("failed to decode {}: {e}", path.display())))?;
    let rgb = img.to_rgb8();
    let dims = Dims::new(rgb.width(), rgb.height());
    Ok((dims, rgb.into_raw()))
}

impl FrameSource for ImageSequence {
    fn dims(&self) -> Dims {
        self.dims
    }

    fn fps(&self) -> f32 {
        self.fps
    }

    fn backend_name(&self) -> &'static str {
        "image-sequence"
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        let Some(path) = self.files.get(self.index) else {
            return Ok(None);
        };
        let (dims, data) = decode_image(path)?;
        if dims != self.dims {
            return Err(FluidError::format(format!(
                "image {} is {dims} but the sequence started at {}",
                path.display(),
                self.dims
            )));
        }
        let mut frame = Frame::new(dims, PixelFormat::Rgb8, data)?;
        frame.index = self.index as u64;
        frame.pts_ns = pts_for(self.index as u64, self.fps);
        self.index += 1;
        Ok(Some(frame))
    }
}

// ---------------------------------------------------------------------------
// PNG writer
// ---------------------------------------------------------------------------

/// Writes frames as numbered PNG files.
pub struct PngWriter {
    dir: PathBuf,
    dims: Dims,
    frames: u64,
}

impl PngWriter {
    pub fn create(dir: impl AsRef<Path>, dims: Dims) -> Result<Self> {
        if dims.is_empty() {
            return Err(FluidError::invalid(
                "cannot write a PNG sequence with empty dimensions",
            ));
        }
        fs::create_dir_all(dir.as_ref())?;
        Ok(Self {
            dir: dir.as_ref().to_path_buf(),
            dims,
            frames: 0,
        })
    }

    pub fn frames(&self) -> u64 {
        self.frames
    }
}

impl FrameSink for PngWriter {
    fn dims(&self) -> Dims {
        self.dims
    }

    fn backend_name(&self) -> &'static str {
        "png-writer"
    }

    fn present(&mut self, frame: &Frame) -> Result<()> {
        if frame.dims != self.dims {
            return Err(FluidError::invalid(format!(
                "png writer expects {} frames, got {}",
                self.dims, frame.dims
            )));
        }
        let converted;
        let rgb: &[u8] = if frame.fmt == PixelFormat::Rgb8 {
            &frame.data
        } else {
            converted = frame.convert(PixelFormat::Rgb8);
            &converted.data
        };
        let path = self.dir.join(format!("frame_{:06}.png", self.frames));
        image::save_buffer(
            &path,
            rgb,
            self.dims.width,
            self.dims.height,
            image::ExtendedColorType::Rgb8,
        )
        .map_err(|e| FluidError::format(format!("failed to write {}: {e}", path.display())))?;
        self.frames += 1;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Synthetic pattern
// ---------------------------------------------------------------------------

/// Per-frame displacement (pixels per frame) of the synthetic pattern. This is
/// the ground truth an interpolator is evaluated against.
pub const SYNTHETIC_VELOCITY: (i32, i32) = (4, 2);

/// Smooth animated test pattern with a scrolling gradient, a bouncing square
/// and per-frame marker bars so interpolation artefacts are easy to spot.
///
/// The whole scene is a pure translation: frame index + 1 equals frame index
/// moved by [SYNTHETIC_VELOCITY] pixels, modulo the frame size. That makes the
/// ground-truth motion exactly known, so an interpolator's error can be
/// measured against the frame it should have produced. Contents:
///
/// * a 24..164 diagonal luma ramp tinted blue-ish that also scrolls with t,
/// * 1 px wide vertical bars every 16 px (aliasing / ringing detector),
/// * a bright 8..64 px square with a 1 px dark border, moving 4 px right and
///   2 px down per frame (wrapping),
/// * a fixed 8x4 cell strip in the top-left corner encoding the frame index as
///   bits, so a duplicated or dropped frame is immediately visible.
///
/// It is genuinely deterministic (no clock, no RNG) and costs a handful of
/// integer ops per pixel: no trig, no per-pixel allocation.
pub fn synthetic_pattern(dims: Dims, t: f32, index: u64) -> Frame {
    if dims.is_empty() {
        return Frame::zeroed(dims, PixelFormat::Rgb8);
    }
    let w = dims.width as i32;
    let h = dims.height as i32;
    let stride = dims.width as usize * 3;
    let mut data = vec![0u8; stride * dims.height as usize];

    let (vx, vy) = SYNTHETIC_VELOCITY;
    let shift_x = (index as i64).wrapping_mul(vx as i64).rem_euclid(w as i64) as i32;
    let shift_y = (index as i64).wrapping_mul(vy as i64).rem_euclid(h as i64) as i32;
    let span = w + h;
    // t only advances the decorative ramp; it is wrapped into one span so a
    // silly value cannot overflow the per-pixel arithmetic.
    let phase = ((t * 24.0) as i32).rem_euclid(span);
    let square = (w.min(h) / 8).clamp(8, 64);
    let sq_x = w / 4;
    let sq_y = h / 4;
    let u_start = (-shift_x).rem_euclid(w);

    for y in 0..h {
        let v = (y - shift_y).rem_euclid(h);
        let mut u = u_start;
        let row_start = y as usize * stride;
        let row = &mut data[row_start..row_start + stride];
        for x in 0..w {
            let ramp = (u + v + phase).rem_euclid(span);
            let l = 24 + ramp * 140 / span;
            let (mut r, mut g, mut b) = (l, l, 40 + ramp * 110 / span);
            if (u & 15) == 0 {
                r = 8;
                g = 8;
                b = 22;
            }
            if u >= sq_x && u < sq_x + square && v >= sq_y && v < sq_y + square {
                let edge =
                    u == sq_x || u == sq_x + square - 1 || v == sq_y || v == sq_y + square - 1;
                if edge {
                    r = 12;
                    g = 12;
                    b = 30;
                } else {
                    r = 250;
                    g = 246;
                    b = 236;
                }
            }
            // Fixed screen-space frame counter: 8 columns x 4 rows of bits.
            if x < 8 && y < 4 {
                let bit = (index >> (x as u32 + y as u32 * 8)) & 1;
                let c = if bit == 1 { 200 } else { 6 };
                r = c;
                g = c;
                b = c;
            }
            let o = x as usize * 3;
            row[o] = r as u8;
            row[o + 1] = g as u8;
            row[o + 2] = b as u8;
            u += 1;
            if u == w {
                u = 0;
            }
        }
    }

    Frame {
        dims,
        fmt: PixelFormat::Rgb8,
        data,
        pts_ns: 0,
        index,
    }
}

/// Luma-only variant of [synthetic_pattern].
pub fn synthetic_luma(dims: Dims, t: f32, index: u64) -> LumaPlane {
    synthetic_pattern(dims, t, index).to_luma()
}

/// Infinite synthetic source for benchmarks and demos.
pub struct SyntheticSource {
    dims: Dims,
    fps: f32,
    index: u64,
}

impl SyntheticSource {
    pub fn new(dims: Dims, fps: f32) -> Self {
        Self {
            dims,
            fps,
            index: 0,
        }
    }
}

impl FrameSource for SyntheticSource {
    fn dims(&self) -> Dims {
        self.dims
    }

    fn fps(&self) -> f32 {
        self.fps
    }

    fn backend_name(&self) -> &'static str {
        "synthetic"
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        let t = self.index as f32 / self.fps.max(1.0);
        let mut f = synthetic_pattern(self.dims, t, self.index);
        f.index = self.index;
        f.pts_ns = pts_for(self.index, self.fps);
        self.index += 1;
        Ok(Some(f))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
    use std::sync::{Arc, Mutex};

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(tag: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, AtomicOrdering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "fluid-video-{}-{}-{}",
                std::process::id(),
                tag,
                n
            ));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }

        fn join(&self, name: &str) -> PathBuf {
            self.path.join(name)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn test_header(dims: Dims, format: Y4mFormat) -> Y4mHeader {
        let mut h = Y4mHeader::new(dims, 30.0);
        h.format = format;
        h
    }

    fn assert_invalid_format<T>(what: &str, r: Result<T>) {
        match r {
            Err(FluidError::InvalidFormat(_)) => {}
            Err(e) => panic!("{what}: expected InvalidFormat, got {e:?}"),
            Ok(_) => panic!("{what}: expected InvalidFormat, got Ok"),
        }
    }

    /// Chroma-neutral test frame (R = G = B per pixel), so 4:2:0/4:2:2
    /// subsampling itself is lossless and only YCbCr quantisation can show up.
    fn grey_frame(dims: Dims) -> Frame {
        let mut frame = Frame::zeroed(dims, PixelFormat::Rgb8);
        for y in 0..dims.height as usize {
            for x in 0..dims.width as usize {
                let v = ((x * 13 + y * 7) % 256) as u8;
                let o = (y * dims.width as usize + x) * 3;
                frame.data[o] = v;
                frame.data[o + 1] = v;
                frame.data[o + 2] = v;
            }
        }
        frame
    }

    fn assert_close(a: &[u8], b: &[u8], tolerance: i32, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let d = (x as i32 - y as i32).abs();
            assert!(
                d <= tolerance,
                "{what}: byte {i} differs by {d} ({x} vs {y})"
            );
        }
    }

    #[test]
    fn y4m_sizes() {
        let h = Y4mHeader::new(Dims::new(64, 48), 60.0);
        assert_eq!(h.dims, Dims::new(64, 48));
        assert!((h.fps() - 60.0).abs() < 0.1);
        assert_eq!(h.frame_bytes(), 64 * 48 + 32 * 24 * 2);
    }

    #[test]
    fn y4m_frame_bytes_per_format() {
        let dims = Dims::new(64, 48);
        let (w, h) = (64usize, 48usize);
        let mut header = test_header(dims, Y4mFormat::C444);
        assert_eq!(header.frame_bytes(), w * h * 3);
        header.format = Y4mFormat::C422;
        assert_eq!(header.frame_bytes(), w * h + (w / 2) * h * 2);
        header.format = Y4mFormat::C420;
        assert_eq!(header.frame_bytes(), w * h + (w / 2) * (h / 2) * 2);
        header.format = Y4mFormat::Mono;
        assert_eq!(header.frame_bytes(), w * h);
    }

    #[test]
    fn header_fps_is_rational() {
        let h = Y4mHeader::new(Dims::new(64, 48), 60.0);
        assert_eq!((h.fps_num, h.fps_den), (60, 1));
        let h = Y4mHeader::new(Dims::new(64, 48), 29.97);
        assert_eq!((h.fps_num, h.fps_den), (30000, 1001));
        let h = Y4mHeader::new(Dims::new(64, 48), 59.94);
        assert_eq!((h.fps_num, h.fps_den), (60000, 1001));
        let h = Y4mHeader::new(Dims::new(64, 48), 23.976);
        assert_eq!((h.fps_num, h.fps_den), (24000, 1001));
        let h = Y4mHeader::new(Dims::new(64, 48), 7.5);
        assert_eq!((h.fps_num, h.fps_den), (15, 2));
        let h = Y4mHeader::new(Dims::new(64, 48), 0.0);
        assert_eq!((h.fps_num, h.fps_den), (0, 1));
        assert_eq!(h.fps(), 0.0);
        let h = Y4mHeader::new(Dims::new(64, 48), f32::NAN);
        assert_eq!(h.fps(), 0.0);
    }

    #[test]
    fn y4m_header_parse_variants() {
        let h = parse_header_line("YUV4MPEG2 W1920 H1080 F60000:1001 Ip A1:1 C420jpeg\n").unwrap();
        assert_eq!(h.dims, Dims::new(1920, 1080));
        assert_eq!((h.fps_num, h.fps_den), (60000, 1001));
        assert_eq!(h.format, Y4mFormat::C420);
        assert!((h.fps() - 59.94).abs() < 0.01);

        let h = parse_header_line("YUV4MPEG2 W640 H480 F30:1 It C422").unwrap();
        assert_eq!(h.format, Y4mFormat::C422);
        assert_eq!(h.dims, Dims::new(640, 480));

        let h = parse_header_line("YUV4MPEG2 W1280 H720 F25:1 C444").unwrap();
        assert_eq!(h.format, Y4mFormat::C444);

        let h = parse_header_line("YUV4MPEG2 W384 H216 F30000:1001 Cmono").unwrap();
        assert_eq!(h.format, Y4mFormat::Mono);
        assert_eq!(h.frame_bytes(), 384 * 216);

        // Missing chroma tag defaults to 4:2:0, missing F means unknown fps.
        let h = parse_header_line("YUV4MPEG2 W64 H48").unwrap();
        assert_eq!(h.format, Y4mFormat::C420);
        assert_eq!(h.fps(), 0.0);

        // Odd dimensions are tolerated when parsing (some encoders round oddly);
        // the frame size uses the truncated plane sizes.
        let h = parse_header_line("YUV4MPEG2 W1279 H719 F24:1 C422").unwrap();
        assert_eq!(h.dims, Dims::new(1279, 719));
        assert_eq!(h.frame_bytes(), 1279 * 719 + 639 * 719 * 2);

        // Unknown X tags are ignored.
        let h = parse_header_line("YUV4MPEG2 W64 H48 F30:1 Xsome C444").unwrap();
        assert_eq!(h.format, Y4mFormat::C444);
    }

    #[test]
    fn y4m_header_parse_errors() {
        assert_invalid_format("empty", parse_header_line(""));
        assert_invalid_format("bad magic", parse_header_line("AVI4MPEG2 W64 H48 F30:1"));
        assert_invalid_format("missing width", parse_header_line("YUV4MPEG2 H48 F30:1"));
        assert_invalid_format("missing height", parse_header_line("YUV4MPEG2 W64 F30:1"));
        assert_invalid_format("bad width", parse_header_line("YUV4MPEG2 Wabc H48 F30:1"));
        assert_invalid_format("bad fps", parse_header_line("YUV4MPEG2 W64 H48 F30"));
        assert_invalid_format("zero fps den", parse_header_line("YUV4MPEG2 W64 H48 F30:0"));
        assert_invalid_format("bad chroma", parse_header_line("YUV4MPEG2 W64 H48 C999"));
        assert_invalid_format("zero dims", parse_header_line("YUV4MPEG2 W0 H48 F30:1"));
        assert_invalid_format(
            "444alpha",
            parse_header_line("YUV4MPEG2 W64 H48 F30:1 C444alpha"),
        );
    }

    #[test]
    fn y4m_roundtrip_c444() {
        let temp = TempDir::new("y4m444");
        let path = temp.join("clip.y4m");
        let dims = Dims::new(64, 48);
        let header = test_header(dims, Y4mFormat::C444);
        let frames: Vec<Frame> = (0..3)
            .map(|i| synthetic_pattern(dims, i as f32 / 30.0, i))
            .collect();

        let mut writer = Y4mWriter::create(&path, header.clone()).unwrap();
        for f in &frames {
            writer.present(f).unwrap();
        }
        assert_eq!(writer.frames(), 3);
        writer.finish().unwrap();

        let mut reader = Y4mReader::open(&path).unwrap();
        assert_eq!(reader.header(), &header);
        assert_eq!(reader.dims(), dims);
        assert!((reader.fps() - 30.0).abs() < 0.01);
        for f in &frames {
            let back = reader.next_frame().unwrap().unwrap();
            assert_eq!(back.dims, dims);
            assert_eq!(back.fmt, PixelFormat::Rgb8);
            assert_close(&f.data, &back.data, 2, "c444 roundtrip");
        }
        assert!(reader.next_frame().unwrap().is_none());
        // End of stream is sticky.
        assert!(reader.next_frame().unwrap().is_none());
    }

    #[test]
    fn y4m_roundtrip_c420_grey() {
        // Neutral grey has exactly Cb = Cr = 128, so 4:2:0 subsampling is
        // lossless and only the luma quantisation can show up.
        let temp = TempDir::new("y4m420");
        let path = temp.join("grey.y4m");
        let dims = Dims::new(32, 24);
        let mut frame = Frame::zeroed(dims, PixelFormat::Rgb8);
        for y in 0..dims.height {
            for x in 0..dims.width {
                let v = ((x * 7 + y * 5) % 256) as u8;
                let o = (y as usize * dims.width as usize + x as usize) * 3;
                frame.data[o] = v;
                frame.data[o + 1] = v;
                frame.data[o + 2] = v;
            }
        }
        let header = test_header(dims, Y4mFormat::C420);
        let mut writer = Y4mWriter::create(&path, header.clone()).unwrap();
        writer.present(&frame).unwrap();
        writer.finish().unwrap();

        let mut reader = Y4mReader::open(&path).unwrap();
        assert_eq!(reader.header(), &header);
        let back = reader.next_frame().unwrap().unwrap();
        // Compare against the original grey level, not the converted frame.
        for (i, &orig) in frame.data.iter().enumerate() {
            let d = (orig as i32 - back.data[i] as i32).abs();
            assert!(d <= 2, "grey roundtrip byte {i} differs by {d}");
        }
        assert!(reader.next_frame().unwrap().is_none());
    }

    #[test]
    fn y4m_mono_roundtrip() {
        let temp = TempDir::new("y4mmono");
        let path = temp.join("mono.y4m");
        let dims = Dims::new(24, 16);
        let mut frame = Frame::zeroed(dims, PixelFormat::Rgb8);
        for y in 0..dims.height {
            for x in 0..dims.width {
                let v = ((x * 9 + y * 3) % 256) as u8;
                let o = (y as usize * dims.width as usize + x as usize) * 3;
                frame.data[o] = v;
                frame.data[o + 1] = v;
                frame.data[o + 2] = v;
            }
        }
        let header = test_header(dims, Y4mFormat::Mono);
        let mut writer = Y4mWriter::create(&path, header.clone()).unwrap();
        writer.present(&frame).unwrap();
        writer.finish().unwrap();

        let mut reader = Y4mReader::open(&path).unwrap();
        assert_eq!(reader.header(), &header);
        let back = reader.next_frame().unwrap().unwrap();
        assert_eq!(back.dims, dims);
        for (i, &orig) in frame.data.iter().enumerate() {
            let d = (orig as i32 - back.data[i] as i32).abs();
            assert!(d <= 2, "mono roundtrip byte {i} differs by {d}");
        }
    }

    #[test]
    fn y4m_writer_rejects_odd_dims_for_subsampled_formats() {
        let temp = TempDir::new("odd");
        assert!(
            Y4mWriter::create(
                temp.join("a.y4m"),
                test_header(Dims::new(31, 17), Y4mFormat::C420)
            )
            .is_err()
        );
        let mut writer = Y4mWriter::create(
            temp.join("b.y4m"),
            test_header(Dims::new(31, 17), Y4mFormat::C444),
        )
        .unwrap();
        let frame = Frame::black(Dims::new(31, 17), PixelFormat::Rgb8);
        writer.present(&frame).unwrap();
        writer.finish().unwrap();
    }

    #[test]
    fn y4m_truncated_files_are_invalid_format() {
        let temp = TempDir::new("trunc");

        // Truncated header.
        let path = temp.join("header.y4m");
        fs::write(&path, b"YUV4MPEG2 W8 H").unwrap();
        assert_invalid_format("truncated header", Y4mReader::open(&path));

        // Header but no frames at all is a valid empty stream.
        let path = temp.join("empty.y4m");
        fs::write(&path, b"YUV4MPEG2 W8 H8 F30:1 Ip C420\n").unwrap();
        let mut reader = Y4mReader::open(&path).unwrap();
        assert!(reader.next_frame().unwrap().is_none());

        // Truncated frame marker.
        let path = temp.join("marker.y4m");
        let mut bytes = b"YUV4MPEG2 W8 H8 F30:1 Ip C420\n".to_vec();
        bytes.extend_from_slice(b"FRA");
        fs::write(&path, &bytes).unwrap();
        let mut reader = Y4mReader::open(&path).unwrap();
        assert_invalid_format("truncated marker", reader.next_frame());

        // Marker present but plane data missing (8x8 4:2:0 needs 96 bytes).
        let path = temp.join("planes.y4m");
        let mut bytes = b"YUV4MPEG2 W8 H8 F30:1 Ip C420\n".to_vec();
        bytes.extend_from_slice(b"FRAME\n");
        bytes.extend_from_slice(&[16u8; 40]);
        fs::write(&path, &bytes).unwrap();
        let mut reader = Y4mReader::open(&path).unwrap();
        assert_invalid_format("truncated planes", reader.next_frame());

        // Wrong magic.
        let path = temp.join("magic.y4m");
        fs::write(&path, b"hello world\n").unwrap();
        assert_invalid_format("bad magic", Y4mReader::open(&path));
    }

    #[test]
    fn natural_order_of_file_names() {
        use std::cmp::Ordering as Cmp;
        assert_eq!(natural_cmp("shot1.png", "shot2.png"), Cmp::Less);
        assert_eq!(natural_cmp("shot2.png", "shot10.png"), Cmp::Less);
        assert_eq!(natural_cmp("shot10.png", "shot010.png"), Cmp::Equal);
        assert_eq!(
            natural_cmp("frame_000001.png", "frame_000002.png"),
            Cmp::Less
        );
        assert_eq!(natural_cmp("a.png", "b.png"), Cmp::Less);
        assert_eq!(natural_cmp("a2.png", "a10.png"), Cmp::Less);
        assert_eq!(natural_cmp("shot9.png", "shot10.png"), Cmp::Less);
    }

    #[test]
    fn image_sequence_natural_order_and_eos() {
        let temp = TempDir::new("seq");
        let dims = Dims::new(4, 4);
        // Write in an order whose lexicographic sort would be wrong.
        for (name, level) in [("shot10.png", 10u8), ("shot2.png", 2), ("shot1.png", 1)] {
            let mut rgb = vec![0u8; (dims.area() as usize) * 3];
            for px in rgb.as_chunks_mut::<3>().0.iter_mut() {
                px[0] = level;
                px[1] = level;
                px[2] = level;
            }
            image::save_buffer(
                temp.join(name),
                &rgb,
                dims.width,
                dims.height,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        }
        // Non-image files are ignored.
        fs::write(temp.join("notes.txt"), b"hello").unwrap();

        let mut seq = ImageSequence::open_dir(temp.path(), 24.0).unwrap();
        assert_eq!(seq.len(), 3);
        assert!(!seq.is_empty());
        assert_eq!(seq.dims(), dims);
        assert_eq!(seq.fps(), 24.0);
        for (i, expected) in [1u8, 2, 10].iter().enumerate() {
            let f = seq.next_frame().unwrap().unwrap();
            assert_eq!(f.index, i as u64);
            assert_eq!(&f.data[0..3], &[*expected, *expected, *expected]);
            let expected_pts = (i as f64 * 1e9 / 24.0) as i64;
            assert!((f.pts_ns - expected_pts).abs() <= 1);
        }
        assert!(seq.next_frame().unwrap().is_none());
        assert!(seq.next_frame().unwrap().is_none());
    }

    #[test]
    fn image_sequence_empty_dir_errors() {
        let temp = TempDir::new("emptyseq");
        assert!(ImageSequence::open_dir(temp.path(), 30.0).is_err());
        fs::write(temp.join("readme.txt"), b"nothing").unwrap();
        assert!(ImageSequence::open_dir(temp.path(), 30.0).is_err());
        assert!(ImageSequence::open_dir(temp.join("does-not-exist"), 30.0).is_err());
    }

    #[test]
    fn png_writer_roundtrip_via_image_sequence() {
        let temp = TempDir::new("png");
        let out = temp.join("frames");
        let dims = Dims::new(40, 24);
        let frames: Vec<Frame> = (0..2)
            .map(|i| synthetic_pattern(dims, i as f32 / 10.0, i))
            .collect();
        let mut writer = PngWriter::create(&out, dims).unwrap();
        for f in &frames {
            writer.present(f).unwrap();
        }
        assert_eq!(writer.frames(), 2);
        assert!(out.join("frame_000000.png").is_file());
        assert!(out.join("frame_000001.png").is_file());

        let mut seq = ImageSequence::open_dir(&out, 10.0).unwrap();
        assert_eq!(seq.len(), 2);
        for f in &frames {
            let back = seq.next_frame().unwrap().unwrap();
            assert_eq!(back.dims, dims);
            // PNG is lossless for Rgb8.
            assert_eq!(back.data, f.data);
        }
        assert!(seq.next_frame().unwrap().is_none());
    }

    #[test]
    fn png_writer_rejects_wrong_dims() {
        let temp = TempDir::new("pngdims");
        let mut writer = PngWriter::create(temp.join("out"), Dims::new(8, 8)).unwrap();
        let wrong = Frame::black(Dims::new(8, 9), PixelFormat::Rgb8);
        assert!(writer.present(&wrong).is_err());
        assert_eq!(writer.frames(), 0);
    }

    #[test]
    fn synthetic_is_deterministic() {
        let dims = Dims::new(48, 32);
        let a = synthetic_pattern(dims, 0.25, 7);
        let b = synthetic_pattern(dims, 0.25, 7);
        assert_eq!(a.dims, dims);
        assert_eq!(a.fmt, PixelFormat::Rgb8);
        assert_eq!(a.data.len(), (dims.area() as usize) * 3);
        assert_eq!(a.checksum(), b.checksum());

        let newer = synthetic_pattern(dims, 0.25, 8);
        assert_ne!(a.checksum(), newer.checksum());
        let other_time = synthetic_pattern(dims, 0.30, 7);
        assert_ne!(a.checksum(), other_time.checksum());

        // Luma variant tracks the pattern.
        let luma = synthetic_luma(dims, 0.25, 7);
        assert_eq!(luma.dims, dims);
        assert!(luma.data.iter().all(|v| (0.0..=1.0).contains(v)));
        let darkest = luma.data.iter().cloned().fold(0.0f32, f32::min);
        let brightest = luma.data.iter().cloned().fold(0.0f32, f32::max);
        assert!(brightest > 0.8, "pattern should contain the bright square");
        assert!(darkest < 0.3, "pattern should contain dark bars");
    }

    #[test]
    fn synthetic_motion_is_known() {
        // Frame(index + 1) is frame(index) translated by SYNTHETIC_VELOCITY,
        // modulo the frame size. Verify by finding the SSD-minimising shift.
        let dims = Dims::new(64, 64);
        let w = dims.width as i32;
        let h = dims.height as i32;
        let a = synthetic_pattern(dims, 0.0, 0);
        let b = synthetic_pattern(dims, 0.0, 1);
        let (vx, vy) = SYNTHETIC_VELOCITY;

        // Skip the fixed frame-counter strip so the scene is a pure translation.
        let skip = |x: i32, y: i32| x < 12 && y < 8;

        let mut best = (0i32, 0i32);
        let mut best_err = i64::MAX;
        for dy in -6i32..=6 {
            for dx in -6i32..=6 {
                let mut err = 0i64;
                for y in 0..h {
                    for x in 0..w {
                        if skip(x, y) {
                            continue;
                        }
                        let ax = (x + dx).rem_euclid(w) as usize;
                        let ay = (y + dy).rem_euclid(h) as usize;
                        for c in 0..3 {
                            let d = b.data[(y as usize * w as usize + x as usize) * 3 + c] as i64
                                - a.data[(ay * w as usize + ax) * 3 + c] as i64;
                            err += d * d;
                        }
                    }
                }
                if err < best_err {
                    best_err = err;
                    best = (dx, dy);
                }
            }
        }
        // b(x, y) = a(x - vx, y - vy), so the matching shift is (-vx, -vy).
        assert_eq!(best, (-vx, -vy), "best SSD shift was {best:?}");
        assert_eq!(best_err, 0, "a pure translation should match exactly");
    }

    #[test]
    fn synthetic_source_is_infinite_and_advances() {
        let dims = Dims::new(32, 24);
        let mut source = SyntheticSource::new(dims, 60.0);
        assert_eq!(source.dims(), dims);
        assert_eq!(source.fps(), 60.0);
        let first = source.next_frame().unwrap().unwrap();
        assert_eq!(first.index, 0);
        assert_eq!(first.pts_ns, 0);
        let second = source.next_frame().unwrap().unwrap();
        assert_eq!(second.index, 1);
        assert!((second.pts_ns as f64 - 1e9 / 60.0).abs() < 2.0);
        assert_ne!(first.checksum(), second.checksum());
        assert_eq!(first.dims, dims);
    }

    #[test]
    fn synthetic_changes_over_time() {
        let a = synthetic_pattern(Dims::new(32, 32), 0.0, 0);
        let b = synthetic_pattern(Dims::new(32, 32), 0.5, 5);
        assert_eq!(a.fmt, PixelFormat::Rgb8);
        assert_ne!(a.checksum(), b.checksum());
    }

    #[test]
    fn yuv_conversion_roundtrips_primaries() {
        for &(r, g, b) in &[
            (0u8, 0u8, 0u8),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (16, 128, 200),
            (250, 246, 236),
        ] {
            let y = rgb_to_y_limited(r, g, b);
            let (cb, cr) = rgb_to_chroma(r, g, b);
            let (rr, gg, bb) = ycbcr_to_rgb(y, cb, cr);
            assert!((rr as i32 - r as i32).abs() <= 2, "{r},{g},{b} -> r {rr}");
            assert!((gg as i32 - g as i32).abs() <= 2, "{r},{g},{b} -> g {gg}");
            assert!((bb as i32 - b as i32).abs() <= 2, "{r},{g},{b} -> b {bb}");
        }
    }

    /// Write adapter that appends to a shared byte vector: lets a y4m stream be
    /// produced in memory and then read back, exactly like a pipe.
    #[derive(Clone)]
    struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

    impl SharedBuffer {
        fn new() -> Self {
            Self(Arc::new(Mutex::new(Vec::new())))
        }

        fn bytes(&self) -> Vec<u8> {
            self.0.lock().expect("buffer lock").clone()
        }
    }

    impl Write for SharedBuffer {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("buffer lock").extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn y4m_pipe_roundtrip_through_memory() {
        let dims = Dims::new(48, 32);
        let header = test_header(dims, Y4mFormat::C444);
        let sink = SharedBuffer::new();
        let mut writer = Y4mWriter::from_writer(Box::new(sink.clone()), header.clone()).unwrap();
        assert_eq!(writer.frames(), 0);
        let frames: Vec<Frame> = (0..4)
            .map(|i| synthetic_pattern(dims, i as f32 / 30.0, i))
            .collect();
        for frame in &frames {
            writer.present(frame).unwrap();
        }
        assert_eq!(writer.frames(), 4);
        writer.finish().unwrap();
        let bytes = sink.bytes();
        assert!(
            bytes.starts_with(b"YUV4MPEG2 W48 H32 F30:1"),
            "header missing from the stream: {:?}",
            String::from_utf8_lossy(&bytes[..bytes.len().min(40)])
        );

        let mut reader = Y4mReader::from_reader(Box::new(Cursor::new(bytes))).unwrap();
        assert_eq!(reader.header(), &header);
        for (i, expected) in frames.iter().enumerate() {
            let got = reader.next_frame().unwrap().unwrap();
            assert_eq!(got.index, i as u64);
            assert_eq!(got.fmt, PixelFormat::Rgb8);
            assert_close(&got.data, &expected.data, 2, "pipe roundtrip");
            assert_eq!(got.pts_ns, pts_for(i as u64, 30.0));
        }
        assert!(reader.next_frame().unwrap().is_none());
    }

    #[test]
    fn y4m_from_writer_flushes_header_immediately() {
        let header = test_header(Dims::new(16, 16), Y4mFormat::C420);
        let sink = SharedBuffer::new();
        let writer = Y4mWriter::from_writer(Box::new(sink.clone()), header).unwrap();
        let bytes = sink.bytes();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.starts_with("YUV4MPEG2 W16 H16 F30:1"), "got {text:?}");
        assert!(text.ends_with('\n'));
        drop(writer);
    }

    #[test]
    fn y4m_sink_close_flushes_the_pipe() {
        let dims = Dims::new(16, 16);
        let sink = SharedBuffer::new();
        let mut writer =
            Y4mWriter::from_writer(Box::new(sink.clone()), test_header(dims, Y4mFormat::C444))
                .unwrap();
        writer.present(&synthetic_pattern(dims, 0.0, 0)).unwrap();
        let as_sink: &mut dyn FrameSink = &mut writer;
        as_sink.close();
        let bytes = sink.bytes();
        assert!(bytes.len() > 16 * 16 * 3);
        let mut reader = Y4mReader::from_reader(Box::new(Cursor::new(bytes))).unwrap();
        assert!(reader.next_frame().unwrap().is_some());
    }

    #[test]
    fn y4m_from_reader_rejects_a_garbage_header() {
        let r = Y4mReader::from_reader(Box::new(Cursor::new(b"not a y4m stream\n".to_vec())));
        assert_invalid_format("garbage header", r);
    }

    /// Frames whose dimensions are not multiples of the chroma subsampling
    /// factor must survive a boxed-reader/writer round trip: real files from
    /// croppers and phone recordings have odd widths and heights all the time.
    #[test]
    fn y4m_pipe_roundtrip_odd_dimensions() {
        // 4:4:4 keeps a coloured synthetic frame; only the +-2 YCbCr
        // quantisation error is allowed.
        let dims = Dims::new(1279, 719);
        let header = test_header(dims, Y4mFormat::C444);
        let sink = SharedBuffer::new();
        let mut writer = Y4mWriter::from_writer(Box::new(sink.clone()), header.clone()).unwrap();
        let frames: Vec<Frame> = (0..2)
            .map(|i| synthetic_pattern(dims, i as f32 / 24.0, i))
            .collect();
        for frame in &frames {
            writer.present(frame).unwrap();
        }
        writer.finish().unwrap();

        let mut reader = Y4mReader::from_reader(Box::new(Cursor::new(sink.bytes()))).unwrap();
        assert_eq!(reader.header(), &header);
        assert_eq!(reader.dims(), dims);
        for (i, expected) in frames.iter().enumerate() {
            let got = reader.next_frame().unwrap().unwrap();
            assert_eq!(got.index, i as u64);
            assert_close(&got.data, &expected.data, 2, "odd 4:4:4 pipe roundtrip");
        }
        assert!(reader.next_frame().unwrap().is_none());

        // 4:2:2 only requires an even width (an odd height is legal) and mono
        // accepts both odd dimensions. Use a grey frame so subsampling is
        // lossless and the tolerances stay tight.
        for (format, dims) in [
            (Y4mFormat::C422, Dims::new(1280, 719)),
            (Y4mFormat::Mono, Dims::new(321, 97)),
        ] {
            let grey = grey_frame(dims);
            let header = test_header(dims, format);
            let sink = SharedBuffer::new();
            let mut writer =
                Y4mWriter::from_writer(Box::new(sink.clone()), header.clone()).unwrap();
            writer.present(&grey).unwrap();
            writer.finish().unwrap();

            let mut reader = Y4mReader::from_reader(Box::new(Cursor::new(sink.bytes()))).unwrap();
            assert_eq!(reader.header(), &header);
            let got = reader.next_frame().unwrap().unwrap();
            assert_eq!(got.dims, dims);
            assert_close(&got.data, &grey.data, 2, "odd grey pipe roundtrip");
            assert!(reader.next_frame().unwrap().is_none());
        }
    }

    /// Every standard frame rate Fluid supports must come back out of the
    /// container with the exact same rational numerator/denominator, so the
    /// output file plays at input_fps * multiplier, not at a rounded rate.
    #[test]
    fn y4m_pipe_roundtrip_standard_fps_ratios() {
        let cases: &[(f32, u32, u32)] = &[
            (23.976, 24000, 1001),
            (24.0, 24, 1),
            (25.0, 25, 1),
            (29.97, 30000, 1001),
            (30.0, 30, 1),
            (50.0, 50, 1),
            (59.94, 60000, 1001),
            (60.0, 60, 1),
            (120.0, 120, 1),
            (144.0, 144, 1),
        ];
        let dims = Dims::new(32, 16);
        for &(fps, num, den) in cases {
            let header = Y4mHeader::new(dims, fps);
            assert_eq!((header.fps_num, header.fps_den), (num, den), "fps {fps}");

            let sink = SharedBuffer::new();
            let mut writer =
                Y4mWriter::from_writer(Box::new(sink.clone()), header.clone()).unwrap();
            writer.present(&synthetic_pattern(dims, 0.0, 0)).unwrap();
            writer.finish().unwrap();

            let mut reader = Y4mReader::from_reader(Box::new(Cursor::new(sink.bytes()))).unwrap();
            assert_eq!(reader.header(), &header, "fps {fps} header roundtrip");
            assert!(
                (reader.fps() - fps).abs() < 0.01,
                "fps {fps} read back as {}",
                reader.fps()
            );
            let frame = reader.next_frame().unwrap().unwrap();
            assert_eq!(frame.pts_ns, pts_for(0, header.fps()), "fps {fps} pts");
        }
    }

    /// Malformed header tokens and non-UTF-8 / empty streams are always
    /// FluidError::InvalidFormat, never a panic or an Io error.
    #[test]
    fn y4m_header_parse_errors_extra() {
        for (what, line) in [
            ("whitespace only", "   "),
            ("width overflow", "YUV4MPEG2 W99999999999 H8 F30:1"),
            ("negative width", "YUV4MPEG2 W-8 H8 F30:1"),
            ("empty fps numerator", "YUV4MPEG2 W8 H8 F:1"),
            ("empty fps denominator", "YUV4MPEG2 W8 H8 F30:"),
            ("extra fps colon", "YUV4MPEG2 W8 H8 F30:1:2"),
            ("fps not a number", "YUV4MPEG2 W8 H8 Fx:1"),
            ("unknown chroma token", "YUV4MPEG2 W8 H8 F30:1 C420p12"),
            ("444alpha chroma", "YUV4MPEG2 W8 H8 F30:1 C444alpha"),
            ("zero height", "YUV4MPEG2 W8 H0 F30:1"),
        ] {
            assert_invalid_format(what, parse_header_line(line));
        }

        assert_invalid_format(
            "non-utf8 header",
            Y4mReader::from_reader(Box::new(Cursor::new(vec![0xffu8, 0xfe, b'\n']))),
        );
        assert_invalid_format(
            "empty stream",
            Y4mReader::from_reader(Box::new(Cursor::new(Vec::new()))),
        );
    }

    /// A stream that ends in the middle of the FRAME marker or in the middle of
    /// any plane must report InvalidFormat instead of returning a short frame.
    #[test]
    fn y4m_truncated_streams_are_invalid_format() {
        let dims = Dims::new(16, 16);
        let sink = SharedBuffer::new();
        let mut writer =
            Y4mWriter::from_writer(Box::new(sink.clone()), test_header(dims, Y4mFormat::C444))
                .unwrap();
        writer.present(&synthetic_pattern(dims, 0.0, 0)).unwrap();
        writer.present(&synthetic_pattern(dims, 0.1, 1)).unwrap();
        writer.finish().unwrap();
        let full = sink.bytes();

        let header_len = full.iter().position(|&b| b == b'\n').unwrap() + 1;
        let frame_len = 6 + dims.area() as usize * 3; // FRAME\n + 4:4:4 planes
        assert_eq!(full.len(), header_len + 2 * frame_len);

        // Truncated first frame marker: no newline, then a partial marker.
        for cut in [header_len + 1, header_len + 3, header_len + 5] {
            let mut reader =
                Y4mReader::from_reader(Box::new(Cursor::new(full[..cut].to_vec()))).unwrap();
            assert_invalid_format("truncated first frame", reader.next_frame());
        }

        // Complete first frame, then the second frame stops early: 1 byte into
        // the marker, 5 bytes (FRAME without newline), 1 plane byte past the
        // marker, in the middle of the chroma planes, and one byte short.
        let second = header_len + frame_len;
        for cut in [
            second + 1,
            second + 5,
            second + 7,
            second + 6 + dims.area() as usize,
            second + frame_len - 1,
        ] {
            let mut reader =
                Y4mReader::from_reader(Box::new(Cursor::new(full[..cut].to_vec()))).unwrap();
            assert!(
                reader.next_frame().unwrap().is_some(),
                "cut at {cut} should still yield the first frame"
            );
            assert_invalid_format("truncated second frame", reader.next_frame());
        }
    }

    /// PNG sequences keep odd, non-square sizes exactly: PNG is lossless, so the
    /// bytes read back must equal the bytes written.
    #[test]
    fn png_sequence_roundtrip_odd_dimensions() {
        let temp = TempDir::new("pngodd");
        let out = temp.join("frames");
        let dims = Dims::new(1279, 719);
        let frames: Vec<Frame> = (0..2)
            .map(|i| synthetic_pattern(dims, i as f32 / 24.0, i))
            .collect();
        let mut writer = PngWriter::create(&out, dims).unwrap();
        for frame in &frames {
            writer.present(frame).unwrap();
        }
        assert_eq!(writer.frames(), 2);

        let mut seq = ImageSequence::open_dir(&out, 24.0).unwrap();
        assert_eq!(seq.len(), 2);
        assert!(!seq.is_empty());
        assert_eq!(seq.dims(), dims);
        for expected in &frames {
            let got = seq.next_frame().unwrap().unwrap();
            assert_eq!(got.dims, dims);
            assert_eq!(got.fmt, PixelFormat::Rgb8);
            assert_eq!(got.data, expected.data);
        }
        assert!(seq.next_frame().unwrap().is_none());
    }

    // -----------------------------------------------------------------------
    // 10-bit y4m input
    // -----------------------------------------------------------------------

    /// A header for a 10-bit stream of the given chroma format.
    fn ten_bit_header(dims: Dims, format: Y4mFormat) -> Y4mHeader {
        let mut header = test_header(dims, format);
        header.bit_depth = 10;
        header
    }

    /// Build an in-memory y4m stream from a header. Samples come from
    /// `sample(frame, sample_index)`; 10-bit streams get two little-endian
    /// bytes per sample, 8-bit streams one byte.
    fn y4m_stream_with(
        header: &Y4mHeader,
        frames: usize,
        sample: impl Fn(usize, usize) -> u16,
    ) -> Vec<u8> {
        let dims = header.dims;
        let mut out = Vec::new();
        out.extend_from_slice(
            format!(
                "{Y4M_MAGIC} W{} H{} F{}:{} Ip A1:1 C{}",
                dims.width,
                dims.height,
                header.fps_num,
                header.fps_den,
                header.chroma_tag()
            )
            .as_bytes(),
        );
        out.push(b'\n');
        for f in 0..frames {
            out.extend_from_slice(b"FRAME\n");
            for i in 0..header.sample_count() {
                let v = sample(f, i);
                if header.bit_depth > 8 {
                    out.extend_from_slice(&v.to_le_bytes());
                } else {
                    out.push(v as u8);
                }
            }
        }
        out
    }

    /// All four 10-bit chroma modes parse, expose the right geometry and hand
    /// out ordinary 8-bit Rgb8 frames; the public 8-bit parser stays 8-bit.
    #[test]
    fn y4m_10bit_header_parse_and_frame_geometry() {
        for (tag, format) in [
            ("420p10", Y4mFormat::C420),
            ("422p10", Y4mFormat::C422),
            ("444p10", Y4mFormat::C444),
            ("mono10", Y4mFormat::Mono),
        ] {
            let dims = Dims::new(64, 48);
            let mut header = ten_bit_header(dims, format);
            header.fps_num = 60000;
            header.fps_den = 1001;
            let stream = y4m_stream_with(&header, 2, |_, i| (i % 1024) as u16);
            let mut reader = Y4mReader::from_reader(Box::new(Cursor::new(stream))).unwrap();
            {
                let header = reader.header();
                assert_eq!(header.format, format, "{tag}");
                assert_eq!(header.bit_depth(), 10, "{tag}");
                assert_eq!(header.dims, dims, "{tag}");
                assert_eq!((header.fps_num, header.fps_den), (60000, 1001), "{tag}");
                assert!((header.fps() - 59.94).abs() < 0.01, "{tag}");
                assert_eq!(header.chroma_tag(), tag, "{tag}");
                assert_eq!(header.frame_bytes(), header.sample_count() * 2, "{tag}");
                assert!(header.to_string().contains(tag), "{tag} display");
            }
            let first = reader.next_frame().unwrap().unwrap();
            assert_eq!(first.fmt, PixelFormat::Rgb8, "{tag}");
            assert_eq!(first.data.len(), dims.area() as usize * 3, "{tag}");
            assert_eq!(reader.next_frame().unwrap().unwrap().index, 1, "{tag}");
            assert!(reader.next_frame().unwrap().is_none(), "{tag}");
        }

        assert!(Y4mFormat::parse("420p10").is_none());
        assert_eq!(Y4mHeader::new(Dims::new(8, 8), 30.0).bit_depth(), 8);
    }

    /// 10-bit samples narrow with rounding: 0 -> 0, 512 -> 128, 1023 -> 255.
    #[test]
    fn y4m_10bit_samples_narrow_with_known_values() {
        assert_eq!(ten_to_eight(0), 0);
        assert_eq!(ten_to_eight(2), 1);
        assert_eq!(ten_to_eight(512), 128);
        assert_eq!(ten_to_eight(1022), 255);
        assert_eq!(ten_to_eight(1023), 255);

        let dims = Dims::new(16, 16);
        for (ten, expected) in [(0u16, 0u8), (512, 128), (1023, 255)] {
            let header = ten_bit_header(dims, Y4mFormat::Mono);
            let stream = y4m_stream_with(&header, 1, move |_, _| ten);
            let mut reader = Y4mReader::from_reader(Box::new(Cursor::new(stream))).unwrap();
            let frame = reader.next_frame().unwrap().unwrap();
            assert_eq!(
                reader.planes,
                vec![expected; dims.area() as usize],
                "10-bit {ten} narrows to {expected}"
            );
            let grey = limited_luma_to_grey(expected);
            assert!(
                frame
                    .data
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .all(|px| *px == [grey, grey, grey]),
                "10-bit {ten} frame should be uniform grey {grey}"
            );
        }
    }

    /// A 10-bit stream whose samples are narrowed by hand must decode exactly
    /// like the equivalent 8-bit stream.
    #[test]
    fn y4m_10bit_matches_pre_narrowed_8bit_stream() {
        let dims = Dims::new(32, 24);
        let ten_header = ten_bit_header(dims, Y4mFormat::C444);
        let eight_header = test_header(dims, Y4mFormat::C444);
        let ten_stream = y4m_stream_with(&ten_header, 2, |f, i| ((i * 7 + f * 131) % 1024) as u16);
        let eight_stream = y4m_stream_with(&eight_header, 2, |f, i| {
            ten_to_eight(((i * 7 + f * 131) % 1024) as u16) as u16
        });
        let mut ten_reader = Y4mReader::from_reader(Box::new(Cursor::new(ten_stream))).unwrap();
        let mut eight_reader = Y4mReader::from_reader(Box::new(Cursor::new(eight_stream))).unwrap();
        for frame in 0..2 {
            let a = ten_reader.next_frame().unwrap().unwrap();
            let b = eight_reader.next_frame().unwrap().unwrap();
            assert_eq!(a.data, b.data, "frame {frame}");
        }
        assert!(ten_reader.next_frame().unwrap().is_none());
        assert!(eight_reader.next_frame().unwrap().is_none());
    }

    /// Odd dimensions are fine for 10-bit input: they only limit subsampled
    /// writers, and readers floor the chroma plane size.
    #[test]
    fn y4m_10bit_odd_dimensions_parse_and_read() {
        for (tag, format, dims) in [
            ("444p10", Y4mFormat::C444, Dims::new(1279, 719)),
            ("422p10", Y4mFormat::C422, Dims::new(1279, 719)),
            ("420p10", Y4mFormat::C420, Dims::new(1278, 718)),
            ("mono10", Y4mFormat::Mono, Dims::new(321, 97)),
        ] {
            let header = ten_bit_header(dims, format);
            let stream = y4m_stream_with(&header, 1, |_, i| ((i * 3) % 1024) as u16);
            let mut reader = Y4mReader::from_reader(Box::new(Cursor::new(stream))).unwrap();
            assert_eq!(
                reader.header().frame_bytes(),
                reader.header().sample_count() * 2,
                "{tag}"
            );
            let frame = reader.next_frame().unwrap().unwrap();
            assert_eq!(frame.dims, dims, "{tag}");
            assert_eq!(frame.data.len(), dims.area() as usize * 3, "{tag}");
            assert!(reader.next_frame().unwrap().is_none(), "{tag}");
        }
    }

    /// A 10-bit stream that ends inside a sample stays InvalidFormat.
    #[test]
    fn y4m_10bit_truncated_streams_are_invalid_format() {
        let dims = Dims::new(16, 16);
        let samples = dims.area() as usize * 3;
        let header_line = b"YUV4MPEG2 W16 H16 F30:1 Ip A1:1 C444p10\n";
        let mut full = header_line.to_vec();
        full.extend_from_slice(b"FRAME\n");
        for i in 0..samples {
            full.extend_from_slice(&((i % 1024) as u16).to_le_bytes());
        }

        // One full frame, then EOF: valid.
        let mut reader = Y4mReader::from_reader(Box::new(Cursor::new(full.clone()))).unwrap();
        assert!(reader.next_frame().unwrap().is_some());
        assert!(reader.next_frame().unwrap().is_none());

        for cut in [
            full.len() - 1,        // one byte short of the last sample
            full.len() - 2,        // missing the final sample
            header_line.len() + 6, // FRAME marker, no sample bytes
            header_line.len() + 3, // interrupted marker ("FRA")
        ] {
            let mut reader =
                Y4mReader::from_reader(Box::new(Cursor::new(full[..cut].to_vec()))).unwrap();
            assert_invalid_format("truncated 10-bit frame", reader.next_frame());
        }
    }

    /// Writers keep emitting 8-bit streams and reject a 10-bit header.
    #[test]
    fn y4m_writer_rejects_10bit_headers() {
        let header = Y4mHeader {
            dims: Dims::new(16, 16),
            fps_num: 30,
            fps_den: 1,
            format: Y4mFormat::C444,
            bit_depth: 10,
        };
        match Y4mWriter::from_writer(Box::new(Vec::new()), header) {
            Err(FluidError::Unsupported(_)) => {}
            Err(err) => panic!("expected Unsupported for 10-bit writer, got {err:?}"),
            Ok(_) => panic!("expected Unsupported for 10-bit writer, got Ok"),
        }
    }

    /// End-to-end: read a 10-bit y4m file, write it as ordinary 8-bit 4:4:4
    /// y4m, read that back. Colour stays within +-2 per channel.
    #[test]
    fn y4m_10bit_roundtrip_through_8bit_writer() {
        let temp = TempDir::new("tenbit");
        let src_path = temp.join("src10.y4m");
        let out_path = temp.join("out8.y4m");
        let dims = Dims::new(64, 48);
        let fps = 30.0;
        let frames = 3usize;
        let src_header = ten_bit_header(dims, Y4mFormat::C444);
        let src = y4m_stream_with(&src_header, frames, |f, i| {
            ((i * 11 + f * 97) % 1024) as u16
        });
        fs::write(&src_path, &src).unwrap();

        let mut reader = Y4mReader::open(&src_path).unwrap();
        assert_eq!(reader.header().bit_depth(), 10);
        let mut out_header = Y4mHeader::new(dims, fps);
        out_header.format = Y4mFormat::C444;
        let mut writer = Y4mWriter::create(&out_path, out_header).unwrap();
        let mut decoded = Vec::new();
        while let Some(frame) = reader.next_frame().unwrap() {
            decoded.push(frame.clone());
            writer.present(&frame).unwrap();
        }
        writer.finish().unwrap();
        assert_eq!(decoded.len(), frames);
        assert_eq!(writer.frames(), frames as u64);

        let mut out_reader = Y4mReader::open(&out_path).unwrap();
        assert_eq!(out_reader.header().bit_depth(), 8);
        assert_eq!(out_reader.header().format, Y4mFormat::C444);
        for (i, expected) in decoded.iter().enumerate() {
            let got = out_reader.next_frame().unwrap().unwrap();
            assert_eq!(got.dims, dims, "frame {i}");
            assert_close(&got.data, &expected.data, 2, "10-bit -> 8-bit roundtrip");
        }
        assert!(out_reader.next_frame().unwrap().is_none());
    }
}
