//! CPU-side conversion between the caller's pixel formats and the canonical
//! RGBA8 texel format the compute shaders operate on.
//!
//! The GPU pipeline always works on RGBA8 texels because that lets every format
//! share one set of kernels. The conversion is byte exact and lossless in both
//! directions for all four formats: BGRA8 is a channel swizzle, RGB8 gains an
//! opaque alpha byte that is dropped on the way back, LUMA8 broadcasts its
//! sample across RGB and is read back from red. Nearest-neighbour scaling and
//! the Off identity copy are therefore byte-identical to the CPU path, which
//! hands the raw bytes through untouched.
//!
//! These helpers are pure safe Rust; the whole crate is `#![forbid(unsafe_code)]`
//! and wgpu owns all of the device lifetime and mapping work.

use fluid_core::PixelFormat;

/// Expand one source row into canonical RGBA8 texels (`4 * pixels` bytes).
///
/// `src` must hold exactly `pixels * fmt.bytes_per_pixel()` bytes and `dst`
/// exactly `pixels * 4` bytes.
pub(crate) fn expand_row(src: &[u8], fmt: PixelFormat, dst: &mut [u8]) {
    match fmt {
        PixelFormat::Rgba8 => dst.copy_from_slice(src),
        PixelFormat::Bgra8 => {
            for (d, s) in dst
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<4>().0)
            {
                d[0] = s[2];
                d[1] = s[1];
                d[2] = s[0];
                d[3] = s[3];
            }
        }
        PixelFormat::Rgb8 => {
            for (d, s) in dst
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<3>().0)
            {
                d[0] = s[0];
                d[1] = s[1];
                d[2] = s[2];
                d[3] = 255;
            }
        }
        PixelFormat::Luma8 => {
            for (d, &v) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src) {
                d[0] = v;
                d[1] = v;
                d[2] = v;
                d[3] = 255;
            }
        }
    }
}

/// Shrink a canonical RGBA8 row back into the caller's pixel format.
///
/// `src` must hold exactly `pixels * 4` bytes and `dst` exactly
/// `pixels * fmt.bytes_per_pixel()` bytes.
pub(crate) fn shrink_row(src: &[u8], fmt: PixelFormat, dst: &mut [u8]) {
    match fmt {
        PixelFormat::Rgba8 => dst.copy_from_slice(src),
        PixelFormat::Bgra8 => {
            for (d, s) in dst
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<4>().0)
            {
                d[0] = s[2];
                d[1] = s[1];
                d[2] = s[0];
                d[3] = s[3];
            }
        }
        PixelFormat::Rgb8 => {
            for (d, s) in dst
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<4>().0)
            {
                d[0] = s[0];
                d[1] = s[1];
                d[2] = s[2];
            }
        }
        PixelFormat::Luma8 => {
            for (d, s) in dst.iter_mut().zip(src.as_chunks::<4>().0) {
                *d = s[0];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FORMATS: [PixelFormat; 4] = [
        PixelFormat::Rgba8,
        PixelFormat::Bgra8,
        PixelFormat::Rgb8,
        PixelFormat::Luma8,
    ];

    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 37 + 11) as u8).collect()
    }

    #[test]
    fn round_trips_every_format_byte_exactly() {
        for fmt in FORMATS {
            let bpp = fmt.bytes_per_pixel();
            let pixels = 7;
            let src = pattern(pixels * bpp);
            let mut canon = vec![0u8; pixels * 4];
            expand_row(&src, fmt, &mut canon);
            let mut back = vec![0u8; pixels * bpp];
            shrink_row(&canon, fmt, &mut back);
            assert_eq!(back, src, "{fmt:?} round trip");
        }
    }

    #[test]
    fn canonical_pixels_are_opaque_and_expansions_are_correct() {
        let mut canon = vec![0u8; 8];
        expand_row(
            &[10, 20, 30, 5, 40, 50, 60, 6],
            PixelFormat::Rgba8,
            &mut canon,
        );
        assert_eq!(canon, [10, 20, 30, 5, 40, 50, 60, 6]);

        expand_row(&[1, 2, 3, 4], PixelFormat::Bgra8, &mut canon[..4]);
        assert_eq!(&canon[..4], &[3, 2, 1, 4]);

        expand_row(&[7, 8, 9], PixelFormat::Rgb8, &mut canon[..4]);
        assert_eq!(&canon[..4], &[7, 8, 9, 255]);

        expand_row(&[123], PixelFormat::Luma8, &mut canon[..4]);
        assert_eq!(&canon[..4], &[123, 123, 123, 255]);
    }

    #[test]
    fn shrink_drops_alpha_and_broadcasts_luma() {
        let canon = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut out = [0u8; 6];
        shrink_row(&canon, PixelFormat::Rgb8, &mut out[..6]);
        assert_eq!(out[..6], [1, 2, 3, 5, 6, 7]);
        let mut luma = [0u8; 2];
        shrink_row(&canon, PixelFormat::Luma8, &mut luma[..2]);
        assert_eq!(luma[..2], [1, 5]);
    }
}
