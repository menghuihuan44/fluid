//! Per-output-coordinate filter weights, computed on the CPU.
//!
//! The GPU resampling passes are a short dot product per output coordinate, so
//! the weights and edge-clamped source indices are built here once at
//! `GpuUpscaler` construction and uploaded as structured buffers. This module is
//! a faithful port of `fluid-upscale`'s `coeff` module: the pixel-centre
//! mapping, the kernel shapes, the edge clamp and the float normalisation are
//! identical, which is what makes the GPU output match the CPU reference to the
//! last bit for RGBA/BGRA sources.
//!
//! Tables are intentionally tiny: at most 6 taps per destination coordinate, so
//! a 4K Lanczos table is `3840 * 6` floats per axis. They are uploaded once and
//! never rebuilt per frame.

#[cfg(test)]
use fluid_core::{Dims, Result};

/// Windowed reconstruction kernels used by the separable passes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kernel {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos3,
}

impl Kernel {
    /// Number of source taps per output coordinate.
    pub(crate) const fn taps(self) -> u32 {
        match self {
            Kernel::Nearest => 1,
            Kernel::Bilinear => 2,
            Kernel::Bicubic => 4,
            Kernel::Lanczos3 => 6,
        }
    }

    /// First tap offset relative to `floor(pos)`.
    const fn first_offset(self) -> i32 {
        match self {
            Kernel::Nearest => 0,
            Kernel::Bilinear => 0,
            Kernel::Bicubic => -1,
            Kernel::Lanczos3 => -2,
        }
    }
}

/// One axis of a resize: `n` taps for each of the output coordinates.
///
/// Output coordinate `d` owns `idx[d * n .. d * n + n]` and `w[d * n ..]`.
#[derive(Clone, Debug, Default)]
pub(crate) struct Taps {
    /// Taps per output coordinate.
    pub n: u32,
    /// Source index per tap, clamped to the source axis.
    pub idx: Vec<u32>,
    /// Normalised weight per tap.
    pub w: Vec<f32>,
}

/// Build the weight table for one axis, matching `fluid-upscale`.
pub(crate) fn build(kernel: Kernel, src_len: u32, dst_len: u32) -> Taps {
    let sl = src_len.max(1) as usize;
    let dl = dst_len.max(1) as usize;
    let last = (sl - 1) as i32;
    let ratio = sl as f32 / dl as f32;
    let n = kernel.taps();
    let nu = n as usize;
    let mut taps = Taps {
        n,
        idx: Vec::with_capacity(dl * nu),
        w: Vec::with_capacity(dl * nu),
    };
    for d in 0..dl {
        // Pixel-centre mapping: destination centres land on source centres.
        let pos = (d as f32 + 0.5) * ratio - 0.5;
        let base = match kernel {
            Kernel::Nearest => pos.round() as i32,
            _ => pos.floor() as i32,
        };
        let mut sum = 0.0f32;
        for t in 0..nu {
            let si = base + kernel.first_offset() + t as i32;
            let x = pos - si as f32;
            let weight = match kernel {
                Kernel::Nearest => 1.0,
                Kernel::Bilinear => 1.0 - x.abs(),
                Kernel::Bicubic => catmull_rom(x),
                Kernel::Lanczos3 => lanczos3(x),
            };
            taps.idx.push(si.clamp(0, last) as u32);
            taps.w.push(weight);
            sum += weight;
        }
        // Normalise so a flat source is reproduced exactly.
        if sum.abs() > 1e-6 {
            let inv = 1.0 / sum;
            for k in 0..nu {
                taps.w[d * nu + k] *= inv;
            }
        } else {
            for k in 0..nu {
                taps.w[d * nu + k] = 0.0;
            }
            taps.w[d * nu] = 1.0;
        }
    }
    taps
}

/// Catmull-Rom cubic kernel (a = -0.5), support 2.
#[inline]
pub(crate) fn catmull_rom(x: f32) -> f32 {
    let x = x.abs();
    if x < 1.0 {
        (1.5 * x - 2.5) * x * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

/// Lanczos-3 windowed sinc, support 3.
#[inline]
pub(crate) fn lanczos3(x: f32) -> f32 {
    if x.abs() < 3.0 {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

#[inline]
fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-5 {
        1.0
    } else {
        let p = std::f32::consts::PI * x;
        p.sin() / p
    }
}

/// Build the two axis tables for a resize.
#[cfg(test)]
pub(crate) fn plan(kernel: Kernel, src: Dims, dst: Dims) -> Result<(Taps, Taps)> {
    if src.is_empty() || dst.is_empty() {
        return Err(fluid_core::FluidError::invalid(
            "gpu upscale: tap tables need non-empty dimensions",
        ));
    }
    Ok((
        build(kernel, src.width, dst.width),
        build(kernel, src.height, dst.height),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Kernel; 4] = [
        Kernel::Nearest,
        Kernel::Bilinear,
        Kernel::Bicubic,
        Kernel::Lanczos3,
    ];

    #[test]
    fn weights_sum_to_one_and_indices_are_in_range() {
        for k in ALL {
            for (sl, dl) in [
                (1u32, 1u32),
                (1, 7),
                (7, 1),
                (4, 8),
                (8, 4),
                (13, 7),
                (320, 1920),
            ] {
                let t = build(k, sl, dl);
                assert_eq!(t.idx.len(), dl as usize * t.n as usize);
                assert_eq!(t.w.len(), dl as usize * t.n as usize);
                for d in 0..dl as usize {
                    let mut sum = 0.0f32;
                    for j in 0..t.n as usize {
                        let i = t.idx[d * t.n as usize + j];
                        assert!(i < sl);
                        sum += t.w[d * t.n as usize + j];
                    }
                    assert!((sum - 1.0).abs() < 1e-3, "{k:?} {sl}->{dl}");
                }
            }
        }
    }

    #[test]
    fn nearest_tables_are_identity_when_sizes_match() {
        let t = build(Kernel::Nearest, 97, 97);
        assert_eq!(t.n, 1);
        for d in 0..97 {
            assert_eq!(t.idx[d], d as u32);
            assert_eq!(t.w[d], 1.0);
        }
    }

    #[test]
    fn nearest_rounds_half_away_from_zero_like_the_cpu() {
        // 2x upscale of the pixel-centre mapping lands on exact half pixels for
        // the odd destination columns; pos.round() goes away from zero.
        let t = build(Kernel::Nearest, 2, 5);
        assert_eq!(t.idx, vec![0, 0, 1, 1, 1]);
    }

    #[test]
    fn bilinear_weights_are_a_triangle() {
        let t = build(Kernel::Bilinear, 2, 4);
        // ratio 0.5: pos = -0.25, 0.25, 0.75, 1.25, so the edge taps clamp.
        assert_eq!(t.idx, vec![0, 0, 0, 1, 0, 1, 1, 1]);
        assert!((t.w[0] - 0.25).abs() < 1e-6);
        assert!((t.w[1] - 0.75).abs() < 1e-6);
        assert!((t.w[2] - 0.75).abs() < 1e-6);
        assert!((t.w[3] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn plan_rejects_empty_axes() {
        assert!(plan(Kernel::Bilinear, Dims::new(0, 4), Dims::new(8, 8)).is_err());
        assert!(plan(Kernel::Bilinear, Dims::new(4, 4), Dims::new(8, 0)).is_err());
    }
}
