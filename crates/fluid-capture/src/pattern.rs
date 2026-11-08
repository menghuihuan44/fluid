//! Deterministic synthetic pattern used by the mock capture backend.
//!
//! The frame is built in three passes so that the most important property for
//! tests - the encoded frame counter - is written last and can never be
//! covered by the moving box:
//!
//! 1. background: a 256-period diagonal gradient
//!    `r = (x + scroll) & 0xFF`, `g = y & 0xFF`, `b = r ^ g`,
//!    where `scroll = floor(fract(phase) * 256)`;
//! 2. a solid box of about 1/6 of the frame travelling on a Lissajous curve in
//!    `phase`, giving the interpolator genuine motion to track;
//! 3. the bottom row: 32 pixels holding bits 0..31 of `index` (bit 0 leftmost),
//!    white for a set bit and black for a clear bit, alpha 255.
//!
//! Everything is integer math on purpose: the same `(dims, phase, index)`
//! always produces the same bytes, which makes mock captures usable as golden
//! data without shipping fixtures. Alpha is always 255 so frames can be shown
//! on any sink.

use fluid_core::{Dims, Frame, PixelFormat};

/// Colour of the moving box (opaque orange).
const BOX_COLOR: [u8; 4] = [250, 96, 48, 255];

/// Render one synthetic frame. Empty dimensions produce an empty frame.
pub(crate) fn render(dims: Dims, phase: f64, index: u64) -> Frame {
    if dims.is_empty() {
        return Frame {
            dims,
            fmt: PixelFormat::Rgba8,
            data: Vec::new(),
            pts_ns: 0,
            index,
        };
    }

    let width = dims.width;
    let height = dims.height;
    let row_len = width as usize * 4;
    let mut data = vec![0u8; dims.area() as usize * 4];

    // Pass 1: scrolling diagonal gradient.
    let fract = phase.rem_euclid(1.0);
    let scroll = (fract * 256.0) as u32 & 0xFF;
    for y in 0..height {
        let green = (y & 0xFF) as u8;
        let row = &mut data[y as usize * row_len..(y as usize + 1) * row_len];
        for (x, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let red = ((x as u32).wrapping_add(scroll) & 0xFF) as u8;
            px[0] = red;
            px[1] = green;
            px[2] = red ^ green;
            px[3] = 255;
        }
    }

    // Pass 2: moving box on a Lissajous path.
    let box_w = (width / 6).max(2).min(width);
    let box_h = (height / 6).max(2).min(height);
    let travel_x = (width - box_w) as f64;
    let travel_y = (height - box_h) as f64;
    let angle = std::f64::consts::TAU * fract;
    let box_x = (travel_x * (0.5 + 0.5 * angle.cos())).round() as u32;
    let box_y = (travel_y * (0.5 + 0.5 * angle.sin())).round() as u32;
    let box_x = box_x.min(width - box_w);
    let box_y = box_y.min(height - box_h);
    for y in box_y..box_y + box_h {
        let row = &mut data[y as usize * row_len..(y as usize + 1) * row_len];
        for px in row
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .skip(box_x as usize)
            .take(box_w as usize)
        {
            px.copy_from_slice(&BOX_COLOR);
        }
    }

    // Pass 3: bottom-row counter, drawn last so tests can read it back.
    let counter = (width as usize).min(32);
    if counter > 0 {
        let last = (height - 1) as usize;
        let row = &mut data[last * row_len..(last + 1) * row_len];
        for (bit, px) in row
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .take(counter)
            .enumerate()
        {
            let on = (index >> bit) & 1 == 1;
            let value = if on { 255 } else { 0 };
            px.copy_from_slice(&[value, value, value, 255]);
        }
    }

    Frame {
        dims,
        fmt: PixelFormat::Rgba8,
        data,
        pts_ns: 0,
        index,
    }
}
