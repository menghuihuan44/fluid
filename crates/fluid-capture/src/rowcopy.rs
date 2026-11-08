//! Pure row-copy helpers for packed image planes.
//!
//! Capture backends never hand us tightly packed pixels: the producer (DXGI
//! staging textures on Windows, AHardwareBuffer / Image.Plane buffers on
//! Android) pads every row to some hardware alignment and may store more than
//! one byte per channel. This module owns the single tested implementation of
//! "copy an arbitrary window out of such a buffer into a tight RGBA8 frame",
//! kept free of platform APIs so the stride arithmetic is unit-testable on any
//! host.
//!
//! The output is always RGBA8 with the alpha channel forced to 255: desktop
//! and MediaProjection captures are opaque by definition and some producers
//! leave the alpha byte undefined (DXGI) or premultiplied (SurfaceFlinger).

use fluid_core::{Dims, FluidError, Rect, Result};

/// Bytes copied per output pixel.
pub(crate) const OUT_PIXEL_BYTES: usize = 4;

/// Copy a crop window from a strided source image into a tight RGBA8 buffer.
///
/// The source holds at least row_stride * image_dims.height bytes. Rows start
/// row_stride bytes apart, and pixels inside a row are pixel_stride bytes
/// apart. The following source layouts are understood:
///
/// - pixel_stride == 4: RGBA8 (alpha is replaced with 255),
/// - pixel_stride == 3: RGB8 (alpha is set to 255),
/// - pixel_stride == 1: 8-bit luminance replicated to RGB (opaque),
/// - any larger stride: the first three bytes of every pixel are used.
///
/// The destination must be exactly crop.w * crop.h * 4 bytes long. The crop
/// must be fully inside image_dims; callers clamp it before calling (see
/// Rect::intersect). Returns an invalid-configuration error on a size
/// mismatch instead of panicking so a broken producer cannot take the engine
/// down.
// The source rows are producer-defined strides: `chunks_exact` keeps the
// ragged-tail cases explicit, `as_chunks` would silently panic on them.
#[allow(clippy::chunks_exact_to_as_chunks)]
pub(crate) fn copy_window_to_rgba8(
    src: &[u8],
    row_stride: usize,
    pixel_stride: usize,
    image_dims: Dims,
    crop: Rect,
    dst: &mut [u8],
) -> Result<()> {
    let dst_len = crop.w as usize * crop.h as usize * OUT_PIXEL_BYTES;
    if dst.len() != dst_len {
        return Err(FluidError::invalid(format!(
            "android capture: destination buffer is {} bytes, expected {dst_len} for a {}x{} crop",
            dst.len(),
            crop.w,
            crop.h
        )));
    }
    if pixel_stride == 0 {
        return Err(FluidError::invalid(
            "android capture: image plane reports a zero pixel stride",
        ));
    }
    if crop.is_empty() {
        return Ok(());
    }
    let crop_right = crop.x as u64 + crop.w as u64;
    let crop_bottom = crop.y as u64 + crop.h as u64;
    if crop_right > image_dims.width as u64 || crop_bottom > image_dims.height as u64 {
        return Err(FluidError::invalid(format!(
            "android capture: crop {}x{} at ({}, {}) exceeds the {}x{} image",
            crop.w, crop.h, crop.x, crop.y, image_dims.width, image_dims.height
        )));
    }
    // Bytes needed by the last pixel of the widest row we read.
    let row_bytes = (image_dims.width as usize - 1)
        .saturating_mul(pixel_stride)
        .saturating_add(OUT_PIXEL_BYTES.min(pixel_stride.max(1)));
    if row_stride < row_bytes {
        return Err(FluidError::invalid(format!(
            "android capture: row stride {row_stride} is smaller than the {row_bytes} bytes one row of pixels needs"
        )));
    }
    let needed = row_stride
        .saturating_mul(image_dims.height as usize - 1)
        .saturating_add(row_bytes);
    if src.len() < needed {
        return Err(FluidError::invalid(format!(
            "android capture: image plane is {} bytes, needs at least {needed}",
            src.len()
        )));
    }

    let crop_w = crop.w as usize;
    for out_y in 0..crop.h as usize {
        let src_y = crop.y as usize + out_y;
        let row_start = src_y * row_stride + crop.x as usize * pixel_stride;
        let src_row = &src[row_start..row_start + crop_w * pixel_stride];
        let out_row =
            &mut dst[out_y * crop_w * OUT_PIXEL_BYTES..(out_y + 1) * crop_w * OUT_PIXEL_BYTES];
        match pixel_stride {
            4 => {
                for (s, d) in src_row.chunks_exact(4).zip(out_row.chunks_exact_mut(4)) {
                    d[0] = s[0];
                    d[1] = s[1];
                    d[2] = s[2];
                    d[3] = 255;
                }
            }
            3 => {
                for (s, d) in src_row.chunks_exact(3).zip(out_row.chunks_exact_mut(4)) {
                    d[0] = s[0];
                    d[1] = s[1];
                    d[2] = s[2];
                    d[3] = 255;
                }
            }
            1 => {
                for (s, d) in src_row.iter().zip(out_row.chunks_exact_mut(4)) {
                    d[0] = *s;
                    d[1] = *s;
                    d[2] = *s;
                    d[3] = 255;
                }
            }
            stride => {
                for (x, d) in out_row.chunks_exact_mut(4).enumerate() {
                    let p = &src_row[x * stride..x * stride + 3];
                    d[0] = p[0];
                    d[1] = p[1];
                    d[2] = p[2];
                    d[3] = 255;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluid_core::PixelFormat;

    fn rgba_pixels(width: usize, pixel: [u8; 4]) -> Vec<u8> {
        (0..width).flat_map(|_| pixel).collect()
    }

    #[test]
    fn copies_rgba_with_row_padding() {
        let image = Dims::new(3, 2);
        let mut src = Vec::new();
        // Row 0: red, green, blue + one pixel of padding.
        src.extend(rgba_pixels(1, [255, 0, 0, 7]));
        src.extend(rgba_pixels(1, [0, 255, 0, 7]));
        src.extend(rgba_pixels(1, [0, 0, 255, 7]));
        src.extend(rgba_pixels(1, [9, 9, 9, 9]));
        // Row 1: white, black, grey + padding.
        src.extend(rgba_pixels(1, [255, 255, 255, 0]));
        src.extend(rgba_pixels(1, [0, 0, 0, 0]));
        src.extend(rgba_pixels(1, [128, 128, 128, 3]));
        src.extend(rgba_pixels(1, [9, 9, 9, 9]));
        let crop = Rect::new(0, 0, 3, 2);
        let mut dst = vec![0u8; 3 * 2 * 4];
        copy_window_to_rgba8(&src, 16, 4, image, crop, &mut dst).unwrap();
        assert_eq!(dst[0..4], [255, 0, 0, 255]);
        assert_eq!(dst[4..8], [0, 255, 0, 255]);
        assert_eq!(dst[8..12], [0, 0, 255, 255]);
        assert_eq!(dst[12..16], [255, 255, 255, 255]);
        assert_eq!(dst[16..20], [0, 0, 0, 255]);
        assert_eq!(dst[20..24], [128, 128, 128, 255]);
    }

    #[test]
    fn crops_offset_window() {
        let image = Dims::new(4, 3);
        let row_stride = 4 * 4 + 8;
        let mut src = Vec::new();
        for y in 0..3u8 {
            for x in 0..4u8 {
                src.extend([x, y, x ^ y, 0]);
            }
            src.extend([0; 8]);
        }
        let crop = Rect::new(1, 1, 2, 2);
        let mut dst = vec![0u8; 2 * 2 * 4];
        copy_window_to_rgba8(&src, row_stride, 4, image, crop, &mut dst).unwrap();
        assert_eq!(dst[0..4], [1, 1, 0, 255]);
        assert_eq!(dst[4..8], [2, 1, 3, 255]);
        assert_eq!(dst[8..12], [1, 2, 3, 255]);
        assert_eq!(dst[12..16], [2, 2, 0, 255]);
    }

    #[test]
    fn expands_three_byte_pixels() {
        let image = Dims::new(2, 1);
        let src = [10u8, 20, 30, 40, 50, 60];
        let mut dst = vec![0u8; 2 * 4];
        copy_window_to_rgba8(&src, 6, 3, image, Rect::new(0, 0, 2, 1), &mut dst).unwrap();
        assert_eq!(dst, [10, 20, 30, 255, 40, 50, 60, 255]);
    }

    #[test]
    fn expands_luminance_to_grey() {
        let image = Dims::new(3, 1);
        let src = [0u8, 128, 255, 7];
        let mut dst = vec![0u8; 3 * 4];
        copy_window_to_rgba8(&src, 4, 1, image, Rect::new(0, 0, 3, 1), &mut dst).unwrap();
        assert_eq!(dst, [0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255]);
    }

    #[test]
    fn rejects_buffer_size_mismatch() {
        let image = Dims::new(2, 2);
        let src = vec![0u8; 16];
        let mut dst = vec![0u8; 3];
        assert!(copy_window_to_rgba8(&src, 8, 4, image, Rect::new(0, 0, 2, 2), &mut dst).is_err());
    }

    #[test]
    fn rejects_crop_outside_image() {
        let image = Dims::new(2, 2);
        let src = vec![0u8; 16];
        let mut dst = vec![0u8; 2 * 4];
        // Crop is one pixel too wide for the image.
        assert!(copy_window_to_rgba8(&src, 8, 4, image, Rect::new(1, 0, 2, 1), &mut dst).is_err());
    }

    #[test]
    fn rejects_short_source_buffer() {
        let image = Dims::new(2, 2);
        let src = vec![0u8; 10];
        let mut dst = vec![0u8; 4 * 4];
        assert!(copy_window_to_rgba8(&src, 8, 4, image, Rect::new(0, 0, 2, 2), &mut dst).is_err());
    }

    #[test]
    fn empty_crop_is_a_noop() {
        let src = [0u8; 4];
        let mut dst = [0u8; 0];
        copy_window_to_rgba8(&src, 4, 4, Dims::new(1, 1), Rect::new(0, 0, 0, 0), &mut dst).unwrap();
    }

    /// The frame the engine receives must be a tight RGBA8 buffer.
    #[test]
    fn output_is_tight_rgba8() {
        let image = Dims::new(2, 1);
        let src = [1u8, 2, 3, 0, 4, 5, 6, 0, 9, 9, 9, 9];
        let mut dst = vec![0u8; 2 * 4];
        copy_window_to_rgba8(&src, 12, 4, image, Rect::new(0, 0, 2, 1), &mut dst).unwrap();
        assert_eq!(dst, [1, 2, 3, 255, 4, 5, 6, 255]);
        let frame = fluid_core::Frame {
            dims: image,
            fmt: PixelFormat::Rgba8,
            data: dst,
            pts_ns: 0,
            index: 0,
        };
        assert_eq!(frame.row(0), &[1, 2, 3, 255, 4, 5, 6, 255]);
    }
}
