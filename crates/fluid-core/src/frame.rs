//! Frame container, pixel formats, luma plane and rectangle helpers.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{FluidError, Result};

/// Width/height pair.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Serialize, Deserialize)]
pub struct Dims {
    pub width: u32,
    pub height: u32,
}

impl Dims {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub const fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    pub fn aspect(&self) -> f32 {
        if self.height == 0 || self.width == 0 {
            0.0
        } else {
            self.width as f32 / self.height as f32
        }
    }

    /// Scale both axes by factor (clamped to at least 1x1).
    pub fn scaled(&self, factor: f32) -> Dims {
        let f = if factor.is_finite() && factor > 0.0 {
            factor
        } else {
            1.0
        };
        Dims::new(
            ((self.width as f32) * f).round().max(1.0) as u32,
            ((self.height as f32) * f).round().max(1.0) as u32,
        )
    }

    /// Scale by an exact integer ratio, e.g. 3/2.
    pub fn scaled_ratio(&self, num: u32, den: u32) -> Dims {
        let den = den.max(1);
        Dims::new(
            ((self.width as u64 * num as u64) / den as u64).max(1) as u32,
            ((self.height as u64 * num as u64) / den as u64).max(1) as u32,
        )
    }

    /// Round both axes down to even values (required by many yuv/encoder paths).
    pub fn even(&self) -> Dims {
        Dims::new(self.width & !1, self.height & !1)
    }

    pub fn fits(&self, other: Dims) -> bool {
        self.width <= other.width && self.height <= other.height
    }

    pub fn to_tuple(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl fmt::Display for Dims {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

impl From<(u32, u32)> for Dims {
    fn from(v: (u32, u32)) -> Self {
        Dims::new(v.0, v.1)
    }
}

/// Pixel layout of a Frame. All formats are tightly packed (no row padding).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PixelFormat {
    Rgba8,
    Bgra8,
    Rgb8,
    Luma8,
}

impl PixelFormat {
    pub const fn bytes_per_pixel(&self) -> usize {
        match self {
            PixelFormat::Rgba8 | PixelFormat::Bgra8 => 4,
            PixelFormat::Rgb8 => 3,
            PixelFormat::Luma8 => 1,
        }
    }

    pub const fn channels(&self) -> usize {
        match self {
            PixelFormat::Rgba8 | PixelFormat::Bgra8 => 4,
            PixelFormat::Rgb8 => 3,
            PixelFormat::Luma8 => 1,
        }
    }

    pub const fn has_alpha(&self) -> bool {
        matches!(self, PixelFormat::Rgba8 | PixelFormat::Bgra8)
    }

    /// Bytes needed for a dims sized frame of this format.
    pub fn buffer_len(&self, dims: Dims) -> usize {
        dims.width as usize * dims.height as usize * self.bytes_per_pixel()
    }
}

impl fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PixelFormat::Rgba8 => "rgba8",
            PixelFormat::Bgra8 => "bgra8",
            PixelFormat::Rgb8 => "rgb8",
            PixelFormat::Luma8 => "luma8",
        };
        f.write_str(s)
    }
}

/// A single tightly-packed CPU frame plus presentation timestamps.
#[derive(Clone, Debug)]
pub struct Frame {
    pub dims: Dims,
    pub fmt: PixelFormat,
    pub data: Vec<u8>,
    /// Presentation timestamp in nanoseconds on the capture/stream clock.
    pub pts_ns: i64,
    /// Monotonic source frame index (used for telemetry and ordering).
    pub index: u64,
}

impl Frame {
    pub fn new(dims: Dims, fmt: PixelFormat, data: Vec<u8>) -> Result<Self> {
        if dims.is_empty() {
            return Err(FluidError::invalid("frame dimensions must be non-zero"));
        }
        let want = fmt.buffer_len(dims);
        if data.len() != want {
            return Err(FluidError::format(format!(
                "frame buffer has {} bytes, expected {want} for {dims} {fmt}",
                data.len()
            )));
        }
        Ok(Self {
            dims,
            fmt,
            data,
            pts_ns: 0,
            index: 0,
        })
    }

    /// Convenience allocator: zeroed frame with the requested shape.
    pub fn zeroed(dims: Dims, fmt: PixelFormat) -> Self {
        Self {
            dims,
            fmt,
            data: vec![0u8; fmt.buffer_len(dims)],
            pts_ns: 0,
            index: 0,
        }
    }

    pub fn black(dims: Dims, fmt: PixelFormat) -> Self {
        let mut f = Frame::zeroed(dims, fmt);
        if fmt.has_alpha() {
            let (chunks, _) = f.data.as_chunks_mut::<4>();
            for px in chunks {
                px[3] = 255;
            }
        }
        f
    }

    pub fn with_pts(mut self, pts_ns: i64) -> Self {
        self.pts_ns = pts_ns;
        self
    }

    pub fn with_index(mut self, index: u64) -> Self {
        self.index = index;
        self
    }

    pub fn stride(&self) -> usize {
        self.dims.width as usize * self.fmt.bytes_per_pixel()
    }

    pub fn row(&self, y: u32) -> &[u8] {
        let s = self.stride();
        let start = y as usize * s;
        &self.data[start..start + s]
    }

    pub fn row_mut(&mut self, y: u32) -> &mut [u8] {
        let s = self.stride();
        let start = y as usize * s;
        &mut self.data[start..start + s]
    }

    /// Pixel accessor returning the channel bytes.
    pub fn pixel(&self, x: u32, y: u32) -> &[u8] {
        let bpp = self.fmt.bytes_per_pixel();
        let s = self.stride();
        let start = y as usize * s + x as usize * bpp;
        &self.data[start..start + bpp]
    }

    pub fn same_shape(&self, other: &Frame) -> bool {
        self.dims == other.dims && self.fmt == other.fmt
    }

    /// Convert between tightly packed formats.
    pub fn convert(&self, dst: PixelFormat) -> Frame {
        if dst == self.fmt {
            return self.clone();
        }
        let mut out = Frame::zeroed(self.dims, dst);
        let n = self.dims.area() as usize;
        match (self.fmt, dst) {
            (PixelFormat::Rgb8, PixelFormat::Rgba8) => {
                for i in 0..n {
                    let (r, g, b) = (self.data[i * 3], self.data[i * 3 + 1], self.data[i * 3 + 2]);
                    out.data[i * 4..i * 4 + 4].copy_from_slice(&[r, g, b, 255]);
                }
            }
            (PixelFormat::Rgb8, PixelFormat::Bgra8) => {
                for i in 0..n {
                    let (r, g, b) = (self.data[i * 3], self.data[i * 3 + 1], self.data[i * 3 + 2]);
                    out.data[i * 4..i * 4 + 4].copy_from_slice(&[b, g, r, 255]);
                }
            }
            (PixelFormat::Rgba8, PixelFormat::Bgra8) | (PixelFormat::Bgra8, PixelFormat::Rgba8) => {
                for i in 0..n {
                    let p = [
                        self.data[i * 4],
                        self.data[i * 4 + 1],
                        self.data[i * 4 + 2],
                        self.data[i * 4 + 3],
                    ];
                    out.data[i * 4..i * 4 + 4].copy_from_slice(&[p[2], p[1], p[0], p[3]]);
                }
            }
            (PixelFormat::Rgba8, PixelFormat::Rgb8) | (PixelFormat::Bgra8, PixelFormat::Rgb8) => {
                let (r_off, g_off, b_off) = if self.fmt == PixelFormat::Rgba8 {
                    (0usize, 1usize, 2usize)
                } else {
                    (2usize, 1usize, 0usize)
                };
                for i in 0..n {
                    out.data[i * 3] = self.data[i * 4 + r_off];
                    out.data[i * 3 + 1] = self.data[i * 4 + g_off];
                    out.data[i * 3 + 2] = self.data[i * 4 + b_off];
                }
            }
            (PixelFormat::Rgba8, PixelFormat::Rgba8)
            | (PixelFormat::Bgra8, PixelFormat::Bgra8)
            | (PixelFormat::Rgb8, PixelFormat::Rgb8) => {
                // Same format; handled by the early return above.
            }
            (_, PixelFormat::Luma8) => {
                let luma = self.to_luma();
                for (i, v) in luma.data.iter().enumerate() {
                    out.data[i] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
            }
            (PixelFormat::Luma8, other) => {
                for i in 0..n {
                    let v = self.data[i];
                    match other {
                        PixelFormat::Rgb8 => out.data[i * 3..i * 3 + 3].copy_from_slice(&[v, v, v]),
                        PixelFormat::Rgba8 | PixelFormat::Bgra8 => {
                            out.data[i * 4..i * 4 + 4].copy_from_slice(&[v, v, v, 255])
                        }
                        PixelFormat::Luma8 => unreachable!(),
                    }
                }
            }
        }
        out.pts_ns = self.pts_ns;
        out.index = self.index;
        out
    }

    /// BT.709 luma in the 0..1 range.
    pub fn to_luma(&self) -> LumaPlane {
        let n = self.dims.area() as usize;
        let mut data = vec![0f32; n];
        match self.fmt {
            PixelFormat::Rgba8 | PixelFormat::Bgra8 => {
                let (r_off, g_off, b_off) = if self.fmt == PixelFormat::Rgba8 {
                    (0usize, 1usize, 2usize)
                } else {
                    (2usize, 1usize, 0usize)
                };
                let (px_chunks, _) = self.data.as_chunks::<4>();
                for (out, px) in data.iter_mut().zip(px_chunks) {
                    *out = rgb_to_luma(px[r_off], px[g_off], px[b_off]);
                }
            }
            PixelFormat::Rgb8 => {
                let (px_chunks, _) = self.data.as_chunks::<3>();
                for (out, px) in data.iter_mut().zip(px_chunks) {
                    *out = rgb_to_luma(px[0], px[1], px[2]);
                }
            }
            PixelFormat::Luma8 => {
                for (out, &byte) in data.iter_mut().zip(self.data.iter()) {
                    *out = byte as f32 / 255.0;
                }
            }
        }
        LumaPlane {
            dims: self.dims,
            data,
        }
    }

    /// Copy a sub-rectangle into a new frame of the same pixel format.
    pub fn crop(&self, rect: Rect) -> Result<Frame> {
        let r = rect.intersect(self.dims);
        if r.is_empty() {
            return Err(FluidError::invalid(
                "crop rectangle does not intersect the frame",
            ));
        }
        let bpp = self.fmt.bytes_per_pixel();
        let mut out = Frame::zeroed(Dims::new(r.w, r.h), self.fmt);
        for y in 0..r.h {
            let src_off = (r.y + y) as usize * self.stride() + r.x as usize * bpp;
            let dst_off = y as usize * out.stride();
            out.data[dst_off..dst_off + r.w as usize * bpp]
                .copy_from_slice(&self.data[src_off..src_off + r.w as usize * bpp]);
        }
        out.pts_ns = self.pts_ns;
        out.index = self.index;
        Ok(out)
    }

    /// FNV-1a checksum, useful for tests and deterministic caching.
    pub fn checksum(&self) -> u64 {
        crate::math::fnv1a_u64(&self.data)
    }
}

#[inline]
fn rgb_to_luma(r: u8, g: u8, b: u8) -> f32 {
    (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32) / 255.0
}

/// Single channel 32-bit float luma plane in the 0..1 range.
#[derive(Clone, Debug, PartialEq)]
pub struct LumaPlane {
    pub dims: Dims,
    pub data: Vec<f32>,
}

impl LumaPlane {
    pub fn new(dims: Dims, data: Vec<f32>) -> Result<Self> {
        if dims.is_empty() {
            return Err(FluidError::invalid(
                "luma plane dimensions must be non-zero",
            ));
        }
        if data.len() != dims.area() as usize {
            return Err(FluidError::format(format!(
                "luma buffer has {} values, expected {} for {dims}",
                data.len(),
                dims.area()
            )));
        }
        Ok(Self { dims, data })
    }

    pub fn zeros(dims: Dims) -> Self {
        Self {
            dims,
            data: vec![0.0; dims.area() as usize],
        }
    }

    pub fn from_frame(frame: &Frame) -> Self {
        frame.to_luma()
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32) -> f32 {
        self.data[y as usize * self.dims.width as usize + x as usize]
    }

    #[inline]
    pub fn set(&mut self, x: u32, y: u32, v: f32) {
        self.data[y as usize * self.dims.width as usize + x as usize] = v;
    }

    #[inline]
    pub fn get_clamped(&self, x: i32, y: i32) -> f32 {
        let x = x.clamp(0, self.dims.width as i32 - 1) as u32;
        let y = y.clamp(0, self.dims.height as i32 - 1) as u32;
        self.get(x, y)
    }

    /// Bilinear sample with edge clamping; coordinates are pixel centers.
    pub fn sample_bilinear(&self, x: f32, y: f32) -> f32 {
        let max_x = self.dims.width as f32 - 1.0;
        let max_y = self.dims.height as f32 - 1.0;
        let x = x.clamp(0.0, max_x);
        let y = y.clamp(0.0, max_y);
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let x1 = (x0 + 1).min(self.dims.width as i32 - 1);
        let y1 = (y0 + 1).min(self.dims.height as i32 - 1);
        let x0 = x0.max(0);
        let y0 = y0.max(0);
        let a = self.get(x0 as u32, y0 as u32);
        let b = self.get(x1 as u32, y0 as u32);
        let c = self.get(x0 as u32, y1 as u32);
        let d = self.get(x1 as u32, y1 as u32);
        let top = a + (b - a) * fx;
        let bot = c + (d - c) * fx;
        top + (bot - top) * fy
    }

    /// 2x box downscale.
    pub fn downscale2(&self) -> LumaPlane {
        let dw = (self.dims.width / 2).max(1);
        let dh = (self.dims.height / 2).max(1);
        let mut out = LumaPlane::zeros(Dims::new(dw, dh));
        for y in 0..dh {
            let sy = (y * 2).min(self.dims.height.saturating_sub(1));
            let sy1 = (sy + 1).min(self.dims.height.saturating_sub(1));
            for x in 0..dw {
                let sx = (x * 2).min(self.dims.width.saturating_sub(1));
                let sx1 = (sx + 1).min(self.dims.width.saturating_sub(1));
                let v =
                    (self.get(sx, sy) + self.get(sx1, sy) + self.get(sx, sy1) + self.get(sx1, sy1))
                        * 0.25;
                out.set(x, y, v);
            }
        }
        out
    }

    /// Bilinear resample to dims.
    pub fn resample_to(&self, dims: Dims) -> LumaPlane {
        let mut out = LumaPlane::zeros(dims);
        let sx = self.dims.width as f32 / dims.width as f32;
        let sy = self.dims.height as f32 / dims.height as f32;
        for y in 0..dims.height {
            let fy = (y as f32 + 0.5) * sy - 0.5;
            for x in 0..dims.width {
                let fx = (x as f32 + 0.5) * sx - 0.5;
                out.set(x, y, self.sample_bilinear(fx, fy));
            }
        }
        out
    }

    pub fn mean(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }

    pub fn variance(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let mean = self.mean();
        let mut acc = 0f32;
        for v in &self.data {
            let d = v - mean;
            acc += d * d;
        }
        acc / self.data.len() as f32
    }

    /// Row-major access to one row (clamped).
    pub fn row(&self, y: u32) -> &[f32] {
        let w = self.dims.width as usize;
        let y = y.min(self.dims.height.saturating_sub(1)) as usize;
        &self.data[y * w..(y + 1) * w]
    }
}

/// Axis aligned integer rectangle.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    pub const fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    pub const fn from_dims(d: Dims) -> Self {
        Self {
            x: 0,
            y: 0,
            w: d.width,
            h: d.height,
        }
    }

    pub const fn is_empty(&self) -> bool {
        self.w == 0 || self.h == 0
    }

    pub fn right(&self) -> u32 {
        self.x.saturating_add(self.w)
    }

    pub fn bottom(&self) -> u32 {
        self.y.saturating_add(self.h)
    }

    pub fn dims(&self) -> Dims {
        Dims::new(self.w, self.h)
    }

    pub fn intersect(&self, dims: Dims) -> Rect {
        let x0 = self.x.min(dims.width);
        let y0 = self.y.min(dims.height);
        let x1 = self.right().min(dims.width);
        let y1 = self.bottom().min(dims.height);
        Rect::new(x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0))
    }

    pub fn contains(&self, x: u32, y: u32) -> bool {
        x >= self.x && y >= self.y && x < self.right() && y < self.bottom()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_math() {
        let d = Dims::new(1920, 1080);
        assert_eq!(d.scaled(0.5), Dims::new(960, 540));
        assert_eq!(d.scaled_ratio(3, 2), Dims::new(2880, 1620));
        assert!((d.aspect() - 1.7777778).abs() < 1e-4);
        assert_eq!(Dims::new(1919, 1079).even(), Dims::new(1918, 1078));
    }

    #[test]
    fn frame_roundtrip_and_convert() {
        let dims = Dims::new(4, 3);
        let mut f = Frame::black(dims, PixelFormat::Rgba8);
        f.data[0] = 255;
        f.data[3] = 255;
        let bgra = f.convert(PixelFormat::Bgra8);
        assert_eq!(bgra.data[0], 0);
        assert_eq!(bgra.data[2], 255);
        let rgb = bgra.convert(PixelFormat::Rgb8);
        assert_eq!(rgb.data.len(), 36);
        let luma = f.to_luma();
        assert!((luma.get(0, 0) - 0.2126).abs() < 1e-3);
    }

    #[test]
    fn crop_and_bounds() {
        let f = Frame::black(Dims::new(10, 10), PixelFormat::Rgb8);
        let c = f.crop(Rect::new(2, 3, 4, 5)).unwrap();
        assert_eq!(c.dims, Dims::new(4, 5));
        assert!(f.crop(Rect::new(50, 50, 2, 2)).is_err());
    }

    #[test]
    fn luma_downscale_and_resample() {
        let dims = Dims::new(8, 8);
        let mut p = LumaPlane::zeros(dims);
        for y in 0..8 {
            for x in 0..8 {
                p.set(x, y, if x < 4 { 0.0 } else { 1.0 });
            }
        }
        let d = p.downscale2();
        assert_eq!(d.dims, Dims::new(4, 4));
        assert!((d.get(1, 1) - 0.0).abs() < 1e-6);
        assert!((d.get(2, 1) - 1.0).abs() < 1e-6);
        let up = p.resample_to(Dims::new(16, 16));
        assert_eq!(up.dims, Dims::new(16, 16));
    }
}
