// Contrast-adaptive unsharp mask over the luma of the unsharpened frame.
// Reads and writes different buffers, so every neighbour is the pre-sharpen
// pixel exactly like the CPU's row-ring implementation.

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

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
    let c = luma(src[row + x]);
    let l = luma(src[row + xl]);
    let r = luma(src[row + xr]);
    let t = luma(src[up + x]);
    let b = luma(src[dn + x]);
    let tl = luma(src[up + xl]);
    let tr = luma(src[up + xr]);
    let bl = luma(src[dn + xl]);
    let br = luma(src[dn + xr]);
    let d = cross_delta(l, r, t, b, c, tl, tr, bl, br, params.sharp_q8);
    var out = src[row + x];
    if d != 0 {
        out = add_delta(out, d);
    }
    dst[row + x] = out;
}
