//! Pixel conversion for window surfaces.
//!
//! Both presentation backends end up with a buffer whose row pitch is given by
//! the platform (Win32 DIB and Android ANativeWindow alike), so the hot path is
//! always the same: convert one source frame row at a time into the destination
//! row, honouring the destination stride. The window format is always
//! RGBA_8888 / 32-bit BGRA-with-alpha, which means every source format needs a
//! small swizzle:
//!
//! | source | conversion |
//! |--------|------------|
//! | Rgba8  | copy, alpha forced to 255 |
//! | Bgra8  | swap red and blue, alpha 255 |
//! | Rgb8   | expand to 4 bytes, alpha 255 |
//! | Luma8  | replicate grey to RGB, alpha 255 |
//!
//! Alpha is always written as 255: window surfaces ignore alpha for opaque
//! windows, and a zeroed alpha would show up as a transparent window on some
//! Android compositors.

use fluid_core::{FluidError, Frame, PixelFormat, Result};

/// Bytes per pixel on the window surface (RGBA_8888).
pub(crate) const WINDOW_PIXEL_BYTES: usize = 4;

/// Copy a frame into a window surface buffer with stride_px pixels per row.
///
/// The destination must hold at least stride_px * frame height * 4 bytes. Rows
/// past the frame width (padding at the end of each surface row) are left
/// untouched, which is what both the Win32 DIB and the ANativeWindow lock
/// expect.
// Source strides are producer-defined: `chunks_exact` tolerates a ragged tail
// where `as_chunks` would panic in the middle of the presentation thread.
#[allow(clippy::chunks_exact_to_as_chunks)]
pub(crate) fn write_frame_rows(frame: &Frame, dst: &mut [u8], stride_px: usize) -> Result<()> {
    let dims = frame.dims;
    if dims.is_empty() {
        return Err(FluidError::invalid(
            "android present: frame dimensions must be non-zero",
        ));
    }
    if stride_px < dims.width as usize {
        return Err(FluidError::invalid(format!(
            "android present: window stride {stride_px} is smaller than the {}-pixel frame width",
            dims.width
        )));
    }
    let row_bytes = stride_px * WINDOW_PIXEL_BYTES;
    let needed = row_bytes
        .checked_mul(dims.height as usize)
        .ok_or_else(|| FluidError::platform("android present: window buffer is too large"))?;
    if dst.len() < needed {
        return Err(FluidError::invalid(format!(
            "android present: window buffer is {} bytes, needs {needed} for {stride_px}x{} RGBA_8888",
            dst.len(),
            dims.height
        )));
    }

    let width = dims.width as usize;
    let src_bpp = frame.fmt.bytes_per_pixel();
    let src_stride = frame.stride();
    for y in 0..dims.height as usize {
        let src_start = y * src_stride;
        let src = &frame.data[src_start..src_start + width * src_bpp];
        let dst_start = y * row_bytes;
        let out = &mut dst[dst_start..dst_start + width * WINDOW_PIXEL_BYTES];
        match frame.fmt {
            PixelFormat::Rgba8 => {
                for (o, s) in out.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                    o.copy_from_slice(s);
                    o[3] = 255;
                }
            }
            PixelFormat::Bgra8 => {
                for (o, s) in out.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                    o[0] = s[2];
                    o[1] = s[1];
                    o[2] = s[0];
                    o[3] = 255;
                }
            }
            PixelFormat::Rgb8 => {
                for (o, s) in out.chunks_exact_mut(4).zip(src.chunks_exact(3)) {
                    o[0] = s[0];
                    o[1] = s[1];
                    o[2] = s[2];
                    o[3] = 255;
                }
            }
            PixelFormat::Luma8 => {
                for (o, s) in out.chunks_exact_mut(4).zip(src.iter()) {
                    let l = *s;
                    o[0] = l;
                    o[1] = l;
                    o[2] = l;
                    o[3] = 255;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluid_core::{Dims, Frame, PixelFormat};

    fn frame_from(dims: Dims, fmt: PixelFormat, data: Vec<u8>) -> Frame {
        Frame {
            dims,
            fmt,
            data,
            pts_ns: 0,
            index: 0,
        }
    }

    #[test]
    fn copies_rgba_rows_and_preserves_padding() {
        let dims = Dims::new(2, 2);
        let frame = frame_from(
            dims,
            PixelFormat::Rgba8,
            vec![
                1, 2, 3, 0, 4, 5, 6, 0, //
                7, 8, 9, 0, 10, 11, 12, 0,
            ],
        );
        let mut dst = vec![0xAAu8; 3 * 2 * 4];
        write_frame_rows(&frame, &mut dst, 3).unwrap();
        assert_eq!(&dst[0..4], &[1, 2, 3, 255]);
        assert_eq!(&dst[4..8], &[4, 5, 6, 255]);
        assert_eq!(&dst[8..12], &[0xAA; 4]);
        assert_eq!(&dst[12..16], &[7, 8, 9, 255]);
        assert_eq!(&dst[16..20], &[10, 11, 12, 255]);
        assert_eq!(&dst[20..24], &[0xAA; 4]);
    }

    #[test]
    fn swaps_blue_and_red_for_bgra() {
        let frame = frame_from(Dims::new(1, 1), PixelFormat::Bgra8, vec![10, 20, 30, 40]);
        let mut dst = vec![0u8; 4];
        write_frame_rows(&frame, &mut dst, 1).unwrap();
        assert_eq!(dst, vec![30, 20, 10, 255]);
    }

    #[test]
    fn expands_rgb_to_opaque_rgba() {
        let frame = frame_from(Dims::new(2, 1), PixelFormat::Rgb8, vec![1, 2, 3, 4, 5, 6]);
        let mut dst = vec![0u8; 2 * 4];
        write_frame_rows(&frame, &mut dst, 2).unwrap();
        assert_eq!(dst, vec![1, 2, 3, 255, 4, 5, 6, 255]);
    }

    #[test]
    fn replicates_luma_as_grey() {
        let frame = frame_from(Dims::new(2, 1), PixelFormat::Luma8, vec![0, 200]);
        let mut dst = vec![0u8; 2 * 4];
        write_frame_rows(&frame, &mut dst, 2).unwrap();
        assert_eq!(dst, vec![0, 0, 0, 255, 200, 200, 200, 255]);
    }

    #[test]
    fn rejects_short_stride_and_buffer() {
        let frame = frame_from(Dims::new(2, 2), PixelFormat::Rgba8, vec![0u8; 16]);
        let mut dst = vec![0u8; 2 * 2 * 4];
        let err = write_frame_rows(&frame, &mut dst, 1).unwrap_err();
        assert!(err.to_string().contains("stride 1"), "{err}");
        let err = write_frame_rows(&frame, &mut dst, 3).unwrap_err();
        assert!(err.to_string().contains("needs 24"), "{err}");
    }

    #[test]
    fn rejects_empty_frames() {
        let frame = frame_from(Dims::new(0, 0), PixelFormat::Rgba8, Vec::new());
        let mut dst = vec![0u8; 16];
        assert!(write_frame_rows(&frame, &mut dst, 2).is_err());
    }
}
