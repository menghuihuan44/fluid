// Adaptive detail restore: the same edge-directed RCAS-ish cross response the
// CPU resampler applies to its bicubic base, at the fixed refine strength.

@group(0) @binding(0) var<storage, read> base: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

const REFINE_STRENGTH_Q8: i32 = 52;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = params.dst_w;
    let h = params.dst_h;
    if x >= w || y >= h {
        return;
    }
    let row = y * w;
    let xl = prev(x);
    let xr = next(x, w - 1u);
    let up = prev(y) * w;
    let dn = next(y, h - 1u) * w;
    let c = luma(base[row + x]);
    let l = luma(base[row + xl]);
    let r = luma(base[row + xr]);
    let t = luma(base[up + x]);
    let b = luma(base[dn + x]);
    let tl = luma(base[up + xl]);
    let tr = luma(base[up + xr]);
    let bl = luma(base[dn + xl]);
    let br = luma(base[dn + xr]);
    let d = cross_delta(l, r, t, b, c, tl, tr, bl, br, REFINE_STRENGTH_Q8);
    var out = base[row + x];
    if d != 0 {
        out = add_delta(out, d);
    }
    dst[row + x] = out;
}
