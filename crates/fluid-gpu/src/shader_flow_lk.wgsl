// Damped Gauss-Newton sub-pixel refinement of one block against the gradient
// planes of the destination level ('fluid-flow' lk_refine_block).
@group(0) @binding(0) var<storage, read> src_luma: array<f32>;
@group(0) @binding(1) var<storage, read> dst_luma: array<f32>;
@group(0) @binding(2) var<storage, read> grad_x: array<f32>;
@group(0) @binding(3) var<storage, read> grad_y: array<f32>;
@group(0) @binding(4) var<storage, read> grid_in: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> grid_out: array<vec2<f32>>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cols = params.dims.z;
    let rows = params.dims.w;
    if (gid.x >= cols || gid.y >= rows) {
        return;
    }
    let w = params.dims.x;
    let h = params.dims.y;
    let level = vec2<u32>(w, h);
    let bs = params.cfg.x;
    let limit = f32(params.cfg.y);
    let x0 = gid.x * bs;
    let y0 = gid.y * bs;
    var o = grid_in[gid.y * cols + gid.x];
    var iter = 0u;
    while (iter < 2u) {
        var a11 = 0.0;
        var a12 = 0.0;
        var a22 = 0.0;
        var b1 = 0.0;
        var b2 = 0.0;
        var used = 0u;
        var py = 0u;
        while (py < bs) {
            let y = y0 + py;
            var px = 0u;
            while (px < bs) {
                let x = x0 + px;
                px = px + 1u;
                if (x >= w || y >= h) {
                    continue;
                }
                let c = coords(f32(x) + o.x, f32(y) + o.y, level);
                let gxv = bilerp(grad_x[c.i00], grad_x[c.i10], grad_x[c.i01], grad_x[c.i11], c);
                let gyv = bilerp(grad_y[c.i00], grad_y[c.i10], grad_y[c.i01], grad_y[c.i11], c);
                if (gxv * gxv + gyv * gyv < 4.0e-4) {
                    continue;
                }
                let dv = bilerp(dst_luma[c.i00], dst_luma[c.i10], dst_luma[c.i01], dst_luma[c.i11], c);
                let e = src_luma[y * w + x] - dv;
                a11 = a11 + gxv * gxv;
                a12 = a12 + gxv * gyv;
                a22 = a22 + gyv * gyv;
                b1 = b1 + gxv * e;
                b2 = b2 + gyv * e;
                used = used + 1u;
            }
            py = py + 1u;
        }
        if (used < 4u) {
            break;
        }
        let det = a11 * a22 - a12 * a12;
        if (det != det || det <= 1.0e-9) {
            break;
        }
        let dx = (a22 * b1 - a12 * b2) / det;
        let dy = (a11 * b2 - a12 * b1) / det;
        if (dx != dx || dy != dy) {
            break;
        }
        o = vec2<f32>(
            clamp(o.x + clamp(dx, -0.5, 0.5), -limit, limit),
            clamp(o.y + clamp(dy, -0.5, 0.5), -limit, limit),
        );
        iter = iter + 1u;
    }
    grid_out[gid.y * cols + gid.x] = o;
}
