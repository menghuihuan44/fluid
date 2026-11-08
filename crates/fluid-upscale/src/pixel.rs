//! Pixel layout helpers shared by the resampling and sharpening passes.
//!
//! Fluid frames are tightly packed, so one pixel is `bpp` contiguous bytes and a
//! layout records where the red/green/blue channels live (BGRA stores them in
//! reverse order) and whether an alpha channel must be preserved. Luma is the
//! BT.709 weighted sum quantised to 0..=255 with integer weights that add up to
//! 256, which is exact enough for the sharpening masks and cheap to evaluate.

use fluid_core::PixelFormat;

/// Channel offsets for one pixel format.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Layout {
    /// Bytes per pixel.
    pub bpp: usize,
    /// Offset of the red (or luma) channel.
    pub r: usize,
    /// Offset of the green channel.
    pub g: usize,
    /// Offset of the blue channel.
    pub b: usize,
}

impl Layout {
    pub(crate) fn of(fmt: PixelFormat) -> Self {
        match fmt {
            PixelFormat::Rgba8 => Self {
                bpp: 4,
                r: 0,
                g: 1,
                b: 2,
            },
            PixelFormat::Bgra8 => Self {
                bpp: 4,
                r: 2,
                g: 1,
                b: 0,
            },
            PixelFormat::Rgb8 => Self {
                bpp: 3,
                r: 0,
                g: 1,
                b: 2,
            },
            PixelFormat::Luma8 => Self {
                bpp: 1,
                r: 0,
                g: 0,
                b: 0,
            },
        }
    }
}

/// BT.709 luma of one pixel, quantised to 0..=255.
///
/// The integer weights (54, 183, 19) sum to 256, i.e. 0.2109/0.7148/0.0742,
/// close to 0.2126/0.7152/0.0722 and division free.
#[inline]
pub(crate) fn luma8(px: &[u8], l: &Layout) -> u8 {
    if l.bpp == 1 {
        px[0]
    } else {
        let y = 54 * px[l.r] as u32 + 183 * px[l.g] as u32 + 19 * px[l.b] as u32;
        ((y + 128) >> 8) as u8
    }
}

/// Quantise a float accumulator to a clamped 0..=255 byte.
#[inline]
pub(crate) fn store_u8(v: f32) -> u8 {
    // Rust float-to-int casts saturate: NaN -> 0, negatives -> 0, > 255 -> 255.
    // The explicit +0.5 makes this round-half-up and keeps the loop branch free
    // so LLVM can vectorise it.
    (v + 0.5) as u8
}

/// Fill `out` with the luma of every pixel of `row`.
#[inline]
pub(crate) fn luma_row(row: &[u8], out: &mut [u8], l: &Layout) {
    if l.bpp == 1 {
        let n = out.len().min(row.len());
        out[..n].copy_from_slice(&row[..n]);
        return;
    }
    for (o, px) in out.iter_mut().zip(row.chunks_exact(l.bpp)) {
        *o = luma8(px, l);
    }
}

/// Fill `out` with the luma of every pixel of `row` as f32.
///
/// The value is exactly [`luma8`] widened to f32, so a caller that keeps luma
/// planes in f32 computes the same integer deltas as one converting per tap.
#[inline(always)]
pub(crate) fn luma_row_f32(row: &[u8], out: &mut [f32], l: &Layout) {
    if l.bpp == 1 {
        let n = out.len().min(row.len());
        for (o, v) in out[..n].iter_mut().zip(&row[..n]) {
            *o = *v as f32;
        }
        return;
    }
    for (o, px) in out.iter_mut().zip(row.chunks_exact(l.bpp)) {
        *o = luma8(px, l) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluid_core::{Dims, Frame};

    #[test]
    fn luma_weights_are_sane() {
        let l = Layout::of(PixelFormat::Luma8);
        assert_eq!(luma8(&[77], &l), 77);
        let l = Layout::of(PixelFormat::Rgb8);
        assert_eq!(luma8(&[255, 255, 255], &l), 255);
        assert_eq!(luma8(&[0, 0, 0], &l), 0);
        assert_eq!(luma8(&[255, 0, 0], &l), 54);
        let l = Layout::of(PixelFormat::Bgra8);
        // BGRA: blue first, red last.
        assert_eq!(luma8(&[0, 0, 255, 255], &l), 54);
    }

    #[test]
    fn store_u8_rounds_and_clamps() {
        assert_eq!(store_u8(-4.0), 0);
        assert_eq!(store_u8(0.0), 0);
        assert_eq!(store_u8(0.4), 0);
        assert_eq!(store_u8(0.5), 1);
        assert_eq!(store_u8(254.6), 255);
        assert_eq!(store_u8(300.0), 255);
        assert_eq!(store_u8(f32::NAN), 0);
    }

    #[test]
    fn luma_row_matches_pixel_access() {
        let dims = Dims::new(3, 1);
        let f = Frame::new(
            dims,
            PixelFormat::Rgba8,
            vec![10, 20, 30, 255, 40, 50, 60, 0, 70, 80, 90, 7],
        )
        .unwrap();
        let l = Layout::of(PixelFormat::Rgba8);
        let mut out = [0u8; 3];
        luma_row(f.row(0), &mut out, &l);
        for (x, v) in out.iter().enumerate() {
            assert_eq!(*v, luma8(f.pixel(x as u32, 0), &l));
        }
    }
}
