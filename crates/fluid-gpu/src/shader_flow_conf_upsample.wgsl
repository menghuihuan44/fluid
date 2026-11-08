// Bilinear upsample of the half-resolution confidence grid to the full
// confidence map ('fluid-flow' fill_and_confidence confidence branch).
@group(0) @binding(0) var<storage, read> conf_grid: array<f32>;
@group(0) @binding(1) var<storage, read_write> confidence: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.dims.x;
    let h = params.dims.y;
    let gw = params.dims.z;
    let gh = params.dims.w;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let c = coords(f32(gid.x) * 0.5, f32(gid.y) * 0.5, vec2<u32>(gw, gh));
    let v = bilerp(
        conf_grid[c.i00],
        conf_grid[c.i10],
        conf_grid[c.i01],
        conf_grid[c.i11],
        c,
    );
    confidence[gid.y * w + gid.x] = clamp(v, 0.0, 1.0);
}
