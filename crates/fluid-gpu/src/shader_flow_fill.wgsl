// Expand the block grid into a dense per-pixel gather field ('fluid-flow'
// fill_dense). The stored vector is the negated content displacement, so the
// interpolation kernel samples 'prev' at x + alpha * v.
@group(0) @binding(0) var<storage, read> grid: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dense: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.dims.x;
    let h = params.dims.y;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let cols = params.dims.z;
    let rows = params.dims.w;
    let inv = params.flt2.x;
    let scale = params.flt2.y;
    let gx = clamp((f32(gid.x) + 0.5) * inv - 0.5, 0.0, f32(cols) - 1.0);
    let gy = clamp((f32(gid.y) + 0.5) * inv - 0.5, 0.0, f32(rows) - 1.0);
    let c = coords(gx, gy, vec2<u32>(cols, rows));
    let v = bilerp2(grid[c.i00], grid[c.i10], grid[c.i01], grid[c.i11], c);
    dense[gid.y * w + gid.x] = v * scale;
}
