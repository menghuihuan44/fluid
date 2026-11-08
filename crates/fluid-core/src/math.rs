//! Small dependency-free math helpers shared by the imaging crates.

/// Clamp to the 0..1 range.
#[inline]
pub fn clamp01(v: f32) -> f32 {
    if v.is_nan() {
        // NaN has no meaningful position in 0..1; treat it as black.
        0.0
    } else {
        v.clamp(0.0, 1.0)
    }
}

#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Hermite smoothstep between edge0 and edge1.
#[inline]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < f32::EPSILON {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = clamp01((x - edge0) / (edge1 - edge0));
    t * t * (3.0 - 2.0 * t)
}

/// Catmull-Rom cubic interpolation between p1 and p2 at t in 0..1.
#[inline]
pub fn cubic_catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    let a1 = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    let a2 = -0.5 * p0 + 0.5 * p2;
    ((a0 * t + a1) * t + a2) * t + p1
}

/// FNV-1a 64 bit hash.
pub fn fnv1a_u64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Linear interpolation of a sorted ascending slice of f32.
pub fn percentile_f32(sorted_asc: &[f32], p: f32) -> f32 {
    if sorted_asc.is_empty() {
        return 0.0;
    }
    let p = clamp01(p);
    let idx = p * (sorted_asc.len() - 1) as f32;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f32;
    lerp(
        sorted_asc[lo],
        sorted_asc[hi.min(sorted_asc.len() - 1)],
        frac,
    )
}

/// Linear interpolation of a sorted ascending slice of i64.
pub fn percentile_i64(sorted_asc: &[i64], p: f32) -> i64 {
    if sorted_asc.is_empty() {
        return 0;
    }
    let p = clamp01(p);
    let idx = p * (sorted_asc.len() - 1) as f32;
    let lo = idx.floor() as usize;
    let hi = (idx.ceil() as usize).min(sorted_asc.len() - 1);
    let frac = idx - lo as f32;
    let a = sorted_asc[lo] as f32;
    let b = sorted_asc[hi] as f32;
    (a + (b - a) * frac) as i64
}

/// Round a float to the nearest integer, deterministically (no libm surprises).
#[inline]
pub fn round_u32(v: f32) -> u32 {
    // NaN and non-positive values both round to zero.
    if v.is_nan() || v <= 0.0 {
        0
    } else if v >= u32::MAX as f32 {
        u32::MAX
    } else {
        (v + 0.5) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation_helpers() {
        assert!((lerp(0.0, 10.0, 0.25) - 2.5).abs() < 1e-6);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);
        assert!((cubic_catmull_rom(0.0, 1.0, 2.0, 3.0, 0.0) - 1.0).abs() < 1e-5);
        assert!((cubic_catmull_rom(0.0, 1.0, 2.0, 3.0, 1.0) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn percentiles() {
        let v = [1.0f32, 2.0, 3.0, 4.0];
        assert!((percentile_f32(&v, 0.0) - 1.0).abs() < 1e-6);
        assert!((percentile_f32(&v, 1.0) - 4.0).abs() < 1e-6);
        assert!((percentile_f32(&v, 0.5) - 2.5).abs() < 1e-6);
        let i = [10i64, 20, 30];
        assert_eq!(percentile_i64(&i, 0.5), 20);
    }

    #[test]
    fn hash_is_stable() {
        assert_eq!(fnv1a_u64(b""), 0xcbf2_9ce4_8422_2325);
        assert_ne!(fnv1a_u64(b"a"), fnv1a_u64(b"b"));
    }
}
