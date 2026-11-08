// Shared prelude for the GPU optical-flow and interpolation kernels.
//
// The flow mirrors 'fluid-flow' exactly: the same 2x box pyramid, the same
// decimated SAD coarse search, the same 3x3 refinement with a separable
// parabolic fit, the same damped Gauss-Newton sub-pixel pass, the same dense
// field expansion and the same confidence formula. Keeping the arithmetic
// identical is what lets the GPU intermediate frames match the CPU synthesised
// frames within a fraction of a decibel.

struct Params {
    // Level width, level height, block-grid columns, block-grid rows.
    dims: vec4<u32>,
    // Block size, search radius, patch step, spare.
    cfg: vec4<u32>,
    // Old grid columns, old grid rows, spare, spare.
    extra: vec4<u32>,
    // Spare.
    extra2: vec4<u32>,
    // Alpha, blend, confidence threshold, one minus alpha.
    flt: vec4<f32>,
    // Dense-field scale, inverse block pitch, spare, spare.
    flt2: vec4<f32>,
};

// Distance penalty of the coarse search, identical to 'fluid-flow'.
const TIE_BREAK: f32 = 1.0e-4;
// Confidence consistency constant, identical to 'fluid-flow'.
const CONF_K: f32 = 0.5;

// Four packed RGBA bytes as float channel values in 0..=255.
fn unpack4(p: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(p & 0xFFu),
        f32((p >> 8u) & 0xFFu),
        f32((p >> 16u) & 0xFFu),
        f32(p >> 24u),
    );
}

// One channel back to a packed byte, rounded like Rust's '(v + 0.5) as u8':
// NaN becomes zero, everything else saturates to 0..=255 before truncation.
fn q8(v: f32) -> u32 {
    if (v != v) {
        return 0u;
    }
    return u32(clamp(v + 0.5, 0.0, 255.0));
}

fn pack4(v: vec4<f32>) -> u32 {
    let r = q8(v.x);
    let g = q8(v.y);
    let b = q8(v.z);
    let a = q8(v.w);
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

// Rec.709 luma of one packed RGBA pixel, exactly 'fluid-core's rgb_to_luma.
fn luma_of(p: u32) -> f32 {
    let r = f32(p & 0xFFu);
    let g = f32((p >> 8u) & 0xFFu);
    let b = f32((p >> 16u) & 0xFFu);
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
}

// Rust's f32::round (half away from zero); WGSL round() is half to even.
fn round_away(v: f32) -> f32 {
    if (v >= 0.0) {
        return floor(v + 0.5);
    }
    return ceil(v - 0.5);
}

// Edge-clamped bilinear sample coordinates of one plane.
struct Coords {
    i00: u32,
    i10: u32,
    i01: u32,
    i11: u32,
    fx: f32,
    fy: f32,
};

// Bilinear coordinates shared by every plane gather in the flow kernels.
// Non-finite inputs become zero, matching the CPU helpers.
fn coords(x: f32, y: f32, dims: vec2<u32>) -> Coords {
    let max_x = f32(dims.x) - 1.0;
    let max_y = f32(dims.y) - 1.0;
    var cx = x;
    var cy = y;
    if (cx != cx) {
        cx = 0.0;
    } else {
        cx = clamp(cx, 0.0, max_x);
    }
    if (cy != cy) {
        cy = 0.0;
    } else {
        cy = clamp(cy, 0.0, max_y);
    }
    let x0 = u32(floor(cx));
    let y0 = u32(floor(cy));
    let x1 = min(x0 + 1u, dims.x - 1u);
    let y1 = min(y0 + 1u, dims.y - 1u);
    let row0 = y0 * dims.x;
    let row1 = y1 * dims.x;
    return Coords(
        row0 + x0,
        row0 + x1,
        row1 + x0,
        row1 + x1,
        cx - f32(x0),
        cy - f32(y0),
    );
}

fn bilerp(v00: f32, v10: f32, v01: f32, v11: f32, c: Coords) -> f32 {
    let top = v00 + (v10 - v00) * c.fx;
    let bot = v01 + (v11 - v01) * c.fx;
    return top + (bot - top) * c.fy;
}

fn bilerp2(v00: vec2<f32>, v10: vec2<f32>, v01: vec2<f32>, v11: vec2<f32>, c: Coords) -> vec2<f32> {
    let top = v00 + (v10 - v00) * c.fx;
    let bot = v01 + (v11 - v01) * c.fx;
    return top + (bot - top) * c.fy;
}

fn bilerp4(v00: vec4<f32>, v10: vec4<f32>, v01: vec4<f32>, v11: vec4<f32>, c: Coords) -> vec4<f32> {
    let top = v00 + (v10 - v00) * c.fx;
    let bot = v01 + (v11 - v01) * c.fx;
    return top + (bot - top) * c.fy;
}

// Truncating float to index conversion with the CPU's NaN fallback.
fn to_index(v: f32, max: f32) -> u32 {
    if (v != v) {
        return 0u;
    }
    return u32(clamp(v, 0.0, max));
}

// Separable parabolic sub-pixel fit, identical to 'fluid-flow'.
fn parabola(sm: f32, s0: f32, sp: f32) -> f32 {
    let denom = sm - 2.0 * s0 + sp;
    if (denom > 1.0e-6) {
        return clamp((sm - sp) / (2.0 * denom), -0.5, 0.5);
    }
    return 0.0;
}
