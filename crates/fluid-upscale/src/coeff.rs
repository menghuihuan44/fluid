//! Precomputed per-output-coordinate filter weights.
//!
//! Every resize is separable, so it is described by two [`Taps`] tables: one
//! mapping destination columns to source columns and one mapping destination
//! rows to source rows. For each output coordinate the table stores `n` source
//! indices (already edge clamped) and the matching `f32` weights, so the inner
//! loops are a short dot product with no coordinate arithmetic.
//!
//! Kernels: nearest (1 tap), bilinear (2 taps, triangle), bicubic (4 taps,
//! Catmull-Rom with a = -0.5) and Lanczos-3 (6 taps, sinc(x) * sinc(x / 3)).
//! Weights are normalised to sum to exactly one, so a flat source reconstructs
//! exactly even where clamping repeats an index.

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
    pub(crate) const fn taps(self) -> usize {
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

/// One axis of a resize: n taps for each of the output coordinates.
///
/// Output coordinate `d` owns `idx[d * n .. d * n + n]` and `w[d * n ..]`.
#[derive(Clone, Debug, Default)]
pub(crate) struct Taps {
    /// Taps per output coordinate.
    pub n: usize,
    /// Source index per tap, clamped to the source axis.
    pub idx: Vec<u32>,
    /// Normalised weight per tap.
    pub w: Vec<f32>,
}

/// Build the weight table for one axis.
pub(crate) fn build(kernel: Kernel, src_len: u32, dst_len: u32) -> Taps {
    let sl = src_len.max(1) as usize;
    let dl = dst_len.max(1) as usize;
    let last = (sl - 1) as i32;
    let ratio = sl as f32 / dl as f32;
    let n = kernel.taps();
    let mut taps = Taps {
        n,
        idx: Vec::with_capacity(dl * n),
        w: Vec::with_capacity(dl * n),
    };
    for d in 0..dl {
        // Pixel-centre mapping: destination centres land on source centres.
        let pos = (d as f32 + 0.5) * ratio - 0.5;
        let base = match kernel {
            Kernel::Nearest => pos.round() as i32,
            _ => pos.floor() as i32,
        };
        let mut sum = 0.0f32;
        for t in 0..n {
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
            for k in 0..n {
                taps.w[d * n + k] *= inv;
            }
        } else {
            // Degenerate response (cannot happen for these kernels); fall back
            // to a box tap so the output is still defined.
            for k in 0..n {
                taps.w[d * n + k] = 0.0;
            }
            taps.w[d * n] = 1.0;
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
    let x = x.abs();
    if x < 3.0 {
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
                assert_eq!(t.idx.len(), dl as usize * t.n);
                assert_eq!(t.w.len(), dl as usize * t.n);
                for d in 0..dl as usize {
                    let mut sum = 0.0f32;
                    for j in 0..t.n {
                        let i = t.idx[d * t.n + j];
                        assert!(i < sl, "{k:?} {sl}->{dl}: index {i} out of range");
                        sum += t.w[d * t.n + j];
                    }
                    assert!(
                        (sum - 1.0).abs() < 1e-3,
                        "{k:?} {sl}->{dl} d={d}: sum={sum}"
                    );
                }
            }
        }
    }

    #[test]
    fn flat_source_reconstructs_exactly() {
        for k in ALL {
            let t = build(k, 11, 29);
            for d in 0..29usize {
                let mut acc = 0.0f32;
                for j in 0..t.n {
                    acc += 137.0 * t.w[d * t.n + j];
                }
                assert!((acc - 137.0).abs() < 0.5, "{k:?} d={d} acc={acc}");
            }
        }
    }

    #[test]
    fn bilinear_identity_is_a_delta() {
        let t = build(Kernel::Bilinear, 8, 8);
        for d in 0..8usize {
            assert_eq!(t.idx[d * 2], d as u32);
            assert!((t.w[d * 2] - 1.0).abs() < 1e-6);
            assert!(t.w[d * 2 + 1].abs() < 1e-6);
        }
    }

    #[test]
    fn nearest_samples_nearest_centre() {
        let t = build(Kernel::Nearest, 2, 4);
        // Destination centres map to source -0.25, 0.25, 0.75, 1.25.
        assert_eq!(t.idx, vec![0, 0, 1, 1]);
    }

    #[test]
    fn kernel_shapes() {
        assert!((catmull_rom(0.0) - 1.0).abs() < 1e-6);
        assert!(catmull_rom(1.0).abs() < 1e-6);
        assert!(catmull_rom(2.0).abs() < 1e-6);
        assert!(catmull_rom(3.0).abs() < 1e-6);
        assert!((catmull_rom(0.5) - 0.5625).abs() < 1e-6);
        assert!((lanczos3(0.0) - 1.0).abs() < 1e-6);
        assert!(lanczos3(3.0).abs() < 1e-6);
        assert!(lanczos3(1.5) < 0.0, "lanczos rings (negative lobe)");
    }
}
