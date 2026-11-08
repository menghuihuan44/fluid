//! Mouse cursor compositing for the desktop duplication backend.
//!
//! DXGI Desktop Duplication does not render the pointer into the desktop
//! image; it reports the pointer separately through
//! `DXGI_OUTDUPL_FRAME_INFO` / `IDXGIOutputDuplication::GetFramePointerShape`.
//! The shape arrives in one of three encodings, all handled here without any
//! GDI/COM dependency so the logic is unit testable on the target machine:
//!
//! * `COLOR` - 32 bit BGRA pixels with straight (non-premultiplied) alpha.
//! * `MASKED_COLOR` - 32 bit BGRA where a non-zero alpha means "replace",
//!   used for invert/xor style cursors.
//! * `MONOCHROME` - two 1 bpp masks stacked vertically: first the AND mask,
//!   then the XOR mask. The classic Windows cursor rules apply:
//!   AND=1 keeps the desktop, AND=0/XOR=1 paints white, AND=0/XOR=0 paints
//!   black.
//!
//! Compositing is clipped to the frame and tolerant of short shape buffers, so
//! a truncated cursor can never panic or corrupt a frame.

use fluid_core::{Frame, PixelFormat};

/// Raw `DXGI_OUTDUPL_POINTER_SHAPE_TYPE` values.
pub(crate) const KIND_MONOCHROME: u32 = 1;
pub(crate) const KIND_COLOR: u32 = 2;
pub(crate) const KIND_MASKED_COLOR: u32 = 4;

/// Cursor shape encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CursorKind {
    Monochrome,
    Color,
    MaskedColor,
}

impl CursorKind {
    /// Map a raw DXGI shape type to a kind; unknown types are ignored.
    pub(crate) fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            KIND_MONOCHROME => Some(Self::Monochrome),
            KIND_COLOR => Some(Self::Color),
            KIND_MASKED_COLOR => Some(Self::MaskedColor),
            _ => None,
        }
    }
}

/// A decoded pointer shape in desktop pixels.
#[derive(Clone, Debug)]
pub(crate) struct CursorShape {
    pub kind: CursorKind,
    pub width: u32,
    pub height: u32,
    /// Bytes per shape row as reported by the OS (may be padded).
    pub pitch: u32,
    /// Hot spot offset from the shape's top-left corner.
    pub hotspot_x: i32,
    pub hotspot_y: i32,
    /// Raw shape bytes, length at least `pitch * rows`.
    pub data: Vec<u8>,
}

#[inline]
fn rows_for(shape: &CursorShape) -> usize {
    match shape.kind {
        CursorKind::Monochrome => shape.height as usize * 2,
        _ => shape.height as usize,
    }
}

/// Blend one BGRA source pixel over one RGBA destination pixel (src-over).
#[inline]
fn blend_pixel(dst: &mut [u8], src: &[u8]) {
    let alpha = src[3] as u32;
    if alpha == 0 {
        return;
    }
    if alpha == 255 {
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
        dst[3] = 255;
        return;
    }
    let inv = 255 - alpha;
    for channel in 0..3 {
        let src_c = src[2 - channel] as u32;
        dst[channel] = ((src_c * alpha + dst[channel] as u32 * inv + 127) / 255) as u8;
    }
    dst[3] = 255;
}

/// Composite `shape` onto `frame` so that the hot spot lands on
/// `(pos_x, pos_y)` in frame coordinates. Out-of-bounds shapes are clipped;
/// non-RGBA frames are left untouched.
pub(crate) fn composite_cursor(frame: &mut Frame, shape: &CursorShape, pos_x: i32, pos_y: i32) {
    if frame.fmt != PixelFormat::Rgba8 || shape.width == 0 || shape.height == 0 {
        return;
    }
    let pitch = shape.pitch as usize;
    if pitch == 0 || shape.data.len() < pitch * rows_for(shape) {
        return;
    }

    let origin_x = pos_x - shape.hotspot_x;
    let origin_y = pos_y - shape.hotspot_y;
    let frame_w = frame.dims.width as i32;
    let frame_h = frame.dims.height as i32;

    // Clip the shape against the frame before touching any pixels.
    let x0 = origin_x.max(0);
    let y0 = origin_y.max(0);
    let x1 = (origin_x + shape.width as i32).min(frame_w);
    let y1 = (origin_y + shape.height as i32).min(frame_h);
    if x0 >= x1 || y0 >= y1 {
        return;
    }

    let frame_stride = frame_w as usize * 4;
    for sy in y0..y1 {
        let data_row = (sy - origin_y) as usize;
        let frame_row =
            &mut frame.data[sy as usize * frame_stride..(sy as usize + 1) * frame_stride];
        for sx in x0..x1 {
            let shape_x = (sx - origin_x) as usize;
            // Destination uses the frame column, source addressing uses the
            // shape column: they differ wherever clipping trimmed the shape.
            let dst = &mut frame_row[sx as usize * 4..sx as usize * 4 + 4];
            match shape.kind {
                CursorKind::Color => {
                    let off = data_row * pitch + shape_x * 4;
                    blend_pixel(dst, &shape.data[off..off + 4]);
                }
                CursorKind::MaskedColor => {
                    let off = data_row * pitch + shape_x * 4;
                    let src = &shape.data[off..off + 4];
                    if src[3] != 0 {
                        dst[0] = src[2];
                        dst[1] = src[1];
                        dst[2] = src[0];
                        dst[3] = 255;
                    }
                }
                CursorKind::Monochrome => {
                    let byte = shape_x / 8;
                    let bit = 7 - (shape_x % 8);
                    let and_row = data_row;
                    let xor_row = data_row + shape.height as usize;
                    let and_on = shape.data[and_row * pitch + byte] >> bit & 1 == 1;
                    let xor_on = shape.data[xor_row * pitch + byte] >> bit & 1 == 1;
                    if !and_on {
                        let value = if xor_on { 255 } else { 0 };
                        dst[0] = value;
                        dst[1] = value;
                        dst[2] = value;
                        dst[3] = 255;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluid_core::{Dims, Frame, PixelFormat};

    fn frame() -> Frame {
        Frame::zeroed(Dims::new(4, 4), PixelFormat::Rgba8)
    }

    fn shape_of(kind: CursorKind, pixel: [u8; 4]) -> CursorShape {
        CursorShape {
            kind,
            width: 1,
            height: 1,
            pitch: 4,
            hotspot_x: 0,
            hotspot_y: 0,
            data: pixel.to_vec(),
        }
    }

    #[test]
    fn color_cursor_blends_half_alpha() {
        let mut f = frame();
        composite_cursor(&mut f, &shape_of(CursorKind::Color, [0, 0, 255, 128]), 1, 1);
        let px = f.pixel(1, 1);
        assert_eq!(px, &[128u8, 0, 0, 255][..]);
    }

    #[test]
    fn color_cursor_is_opaque_when_alpha_is_opaque() {
        let mut f = frame();
        composite_cursor(
            &mut f,
            &shape_of(CursorKind::Color, [10, 20, 30, 255]),
            0,
            0,
        );
        assert_eq!(f.pixel(0, 0), &[30u8, 20, 10, 255][..]);
    }

    #[test]
    fn masked_color_cursor_replaces_pixels() {
        let mut f = frame();
        composite_cursor(
            &mut f,
            &shape_of(CursorKind::MaskedColor, [10, 20, 30, 255]),
            2,
            2,
        );
        assert_eq!(f.pixel(2, 2), &[30u8, 20, 10, 255][..]);
        // A transparent masked-colour pixel leaves the desktop alone.
        composite_cursor(
            &mut f,
            &shape_of(CursorKind::MaskedColor, [9, 9, 9, 0]),
            0,
            0,
        );
        assert_eq!(f.pixel(0, 0), &[0u8, 0, 0, 0][..]);
    }

    #[test]
    fn monochrome_cursor_applies_and_xor_masks() {
        let shape = CursorShape {
            kind: CursorKind::Monochrome,
            width: 8,
            height: 1,
            pitch: 1,
            hotspot_x: 0,
            hotspot_y: 0,
            // AND mask 1000_0000, XOR mask 0100_0000.
            data: vec![0x80, 0x40],
        };
        let mut f = frame();
        composite_cursor(&mut f, &shape, 0, 0);
        assert_eq!(
            f.pixel(0, 0),
            &[0u8, 0, 0, 0][..],
            "AND=1 must keep the frame"
        );
        assert_eq!(
            f.pixel(1, 0),
            &[255u8, 255, 255, 255][..],
            "AND=0/XOR=1 paints white"
        );
        assert_eq!(
            f.pixel(2, 0),
            &[0u8, 0, 0, 255][..],
            "AND=0/XOR=0 paints black"
        );
    }

    #[test]
    fn cursor_is_clipped_at_frame_edges() {
        let shape = CursorShape {
            kind: CursorKind::Color,
            width: 2,
            height: 2,
            pitch: 8,
            hotspot_x: 0,
            hotspot_y: 0,
            data: vec![
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            ],
        };
        let mut f = frame();
        // Draw at (-1, -1): only the bottom-right shape pixel lands on (0, 0).
        composite_cursor(&mut f, &shape, -1, -1);
        assert_eq!(f.pixel(0, 0), &[255u8, 255, 255, 255][..]);
        // Entirely off-screen is a no-op and must not panic.
        composite_cursor(&mut f, &shape, -100, 50);
        composite_cursor(&mut f, &shape, 100, -50);
    }
}
