// Shared parameter block and byte helpers for the fluid-gpu kernels.
//
// Pixels move through the pipeline as packed u32 texels in canonical RGBA8
// order (r | g<<8 | b<<16 | a<<24). The kernels deliberately run the same
// fixed-point byte arithmetic as the CPU reference in fluid-upscale, so the two
// backends agree byte for byte on nearest/identity and to within one LSB on the
// floating point filters.

struct Params {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    nx: u32,
    ny: u32,
    sharp_q8: i32,
    blend_s: i32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
};

// Canonical RGBA8 texel -> four f32 channel values (no normalisation: the CPU
// accumulates raw byte values and rounds at the end).
fn unpack4(px: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(px & 0xFFu),
        f32((px >> 8u) & 0xFFu),
        f32((px >> 16u) & 0xFFu),
        f32(px >> 24u),
    );
}

// Four accumulated f32 channels -> canonical texel, rounding half up exactly
// like the CPU's (v + 0.5) as u8 saturating cast.
fn pack4(v: vec4<f32>) -> u32 {
    let q = clamp(v + vec4<f32>(0.5), vec4<f32>(0.0), vec4<f32>(255.0));
    return u32(q.x) | (u32(q.y) << 8u) | (u32(q.z) << 16u) | (u32(q.w) << 24u);
}

// Integer luma used by the detail-restore and sharpen kernels.
fn luma(px: u32) -> i32 {
    let r = i32(px & 0xFFu);
    let g = i32((px >> 8u) & 0xFFu);
    let b = i32((px >> 16u) & 0xFFu);
    return ((54 * r + 183 * g + 19 * b) + 128) >> 8;
}

// Floor division, matching an arithmetic right shift on negative values (the
// CPU divides with >> 8 and >> 12 in signed arithmetic).
fn floor_div(a: i32, b: i32) -> i32 {
    if a >= 0 {
        return a / b;
    }
    return -((-a + b - 1) / b);
}

// Clamped previous index: prev(0) == 0.
fn prev(v: u32) -> u32 {
    return select(v - 1u, 0u, v == 0u);
}

// Clamped next index: next(last, last) == last.
fn next(v: u32, last: u32) -> u32 {
    return select(v + 1u, last, v >= last);
}

// Contrast-adaptive cross response, identical to fluid-upscale's cross_delta.
//
// The admissible range is the min/max envelope of the full 3x3 neighbourhood
// (centre, four edge taps and four corners), so the filter can never create a
// new local extreme and cannot ring diagonally either.
fn cross_delta(
    l: i32,
    r: i32,
    t: i32,
    b: i32,
    c: i32,
    tl: i32,
    tr: i32,
    bl: i32,
    br: i32,
    s_q8: i32,
) -> i32 {
    if s_q8 <= 0 {
        return 0;
    }
    let hp_cross = 4 * c - (l + r + t + b);
    let gx = abs(l - r);
    let gy = abs(t - b);
    let mn = min(c, min(min(l, r), min(min(t, b), min(min(tl, tr), min(bl, br)))));
    let mx = max(c, max(max(l, r), max(max(t, b), max(max(tl, tr), max(bl, br)))));
    if mn == mx {
        return 0;
    }
    var hp_axis = 2 * c - l - r;
    var w_num = gx - gy;
    if gy > gx {
        hp_axis = 2 * c - t - b;
        w_num = gy - gx;
    }
    let w_den = gy + gx + 1;
    let hp = (hp_cross * (w_den - w_num) + hp_axis * w_num) / w_den;
    let delta = floor_div(hp * s_q8 + 128, 256);
    if delta == 0 {
        return 0;
    }
    // 'target' is a reserved WGSL keyword, so the clamped result is 'result'.
    let result = clamp(c + delta, mn, mx);
    return result - c;
}

// Add one luma delta to the colour bytes, leaving alpha alone.
fn add_delta(px: u32, d: i32) -> u32 {
    let r = clamp(i32(px & 0xFFu) + d, 0, 255);
    let g = clamp(i32((px >> 8u) & 0xFFu) + d, 0, 255);
    let b = clamp(i32((px >> 16u) & 0xFFu) + d, 0, 255);
    let a = i32(px >> 24u);
    return u32(r) | (u32(g) << 8u) | (u32(b) << 16u) | (u32(a) << 24u);
}

// Fixed point Q12 temporal blend of one byte.
fn blend_byte(cur: u32, hist: u32, s: i32) -> u32 {
    let c = i32(cur);
    let h = i32(hist);
    let v = h + floor_div((c - h) * s + 2048, 4096);
    return u32(clamp(v, 0, 255));
}
