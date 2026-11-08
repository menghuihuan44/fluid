//! Small SIMD helpers shared by the resampling and sharpening inner loops.
//!
//! Everything is built on the SSE2-baseline `f32x4` from the `wide` crate,
//! which also lowers to NEON on aarch64 and keeps this crate free of `unsafe`.
//! Four bytes are loaded with a plain array initialiser: LLVM turns that into a
//! zero-extend plus a packed int-to-float conversion, which is the widest byte
//! -> float load the SSE2 baseline offers. Stores work the other way round: the
//! lanes are quantised one by one with the scalar `store_u8` (clamp + round
//! half up) and written as four bytes, which LLVM fuses into a packed
//! convert-and-pack sequence.
//!
//! The helpers are deliberately tiny and `#[inline]` so the call sites keep
//! their row structure and the per-output-coordinate tap tables are still read
//! once per output pixel; only the arithmetic becomes four lanes wide.
//!
//! Runtime dispatch: passes wrap their row loops in [`dispatch`], which asks
//! `pulp` for the best instruction set the CPU supports (AVX2 + FMA on this
//! target, NEON on aarch64, plain scalar elsewhere) and instantiates the whole
//! closure once per supported instruction set. The code inside is ordinary safe
//! Rust; it is simply compiled with that target's features, so LLVM widens the
//! scalar row arithmetic to 256-bit vectors. `Arch::new()` is called once per
//! pass, never per row, and the helper functions in this module are
//! `#[inline]` so they are specialised into the dispatched body.
//!
//! The floating point operations inside a dispatched body are the same IEEE
//! operations, in the same order, as the scalar code: the crate does not enable
//! fused multiply-add contraction, so widening the loop to eight lanes cannot
//! change a single output byte.

use wide::{f32x4, u8x16};

use crate::pixel::store_u8;

/// Run `f` compiled for the best instruction set this CPU supports.
///
/// `pulp` instantiates the closure separately for every instruction set the
/// binary may see at run time and picks one with a single CPUID check done once
/// per pass. The closure must be self contained (it is `FnOnce`); everything it
/// calls that is `#[inline]` is compiled into the dispatched body as well.
#[inline]
pub(crate) fn dispatch<R>(f: impl FnOnce() -> R) -> R {
    // The inherent `vectorize` method of a concrete instruction set token
    // applies that instruction set's `#[target_feature]` attributes to the
    // closure body, so LLVM emits 256-bit AVX2 code for it. `Arch::dispatch`
    // wraps the closure in an extra generic layer and loses the attributes, so
    // match on the token type directly. Each architecture names its token
    // differently, and the fallback arm covers plain CPUs (and flat maps).
    #[cfg(target_arch = "x86_64")]
    {
        match pulp::Arch::new() {
            pulp::Arch::V3(v3) => v3.vectorize(f),
            _ => f(),
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        match pulp::Arch::new() {
            pulp::Arch::Neon(neon) => neon.vectorize(f),
            _ => f(),
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        f()
    }
}

/// Load four consecutive bytes as four f32 lanes.
#[inline]
pub(crate) fn load4(src: &[u8], off: usize) -> f32x4 {
    let mut lanes = [0.0f32; 4];
    for (lane, byte) in lanes.iter_mut().zip(&src[off..off + 4]) {
        *lane = *byte as f32;
    }
    f32x4::from(lanes)
}

/// Quantise four f32 lanes and store them to four consecutive bytes.
#[inline]
pub(crate) fn store4(v: f32x4, dst: &mut [u8], off: usize) {
    for (byte, lane) in dst[off..off + 4].iter_mut().zip(v.to_array()) {
        *byte = store_u8(lane);
    }
}

/// Load sixteen bytes as one u8x16 lane group.
#[inline]
pub(crate) fn load16(src: &[u8], off: usize) -> u8x16 {
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&src[off..off + 16]);
    u8x16::from(bytes)
}
