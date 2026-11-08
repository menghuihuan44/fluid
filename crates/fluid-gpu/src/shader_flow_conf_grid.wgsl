// Half-resolution forward-backward consistency grid ('fluid-flow'
// fill_and_confidence confidence branch).
//
// Each cell covers a 2x2 block of full-resolution pixels: the grids are sampled
// at every second pixel, the consistency and texture terms are evaluated there
// and one value is stored. A separate bilinear pass upsamples the grid to the
// full confidence map, which is exactly what the CPU reference does.
@group(0) @binding(0) var<storage, read> grid_f: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> grid_b: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> luma: array<f32>;
@group(0) @binding(3) var<storage, read_write> conf_grid: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.dims.x;
    let h = params.dims.y;
    let cols = params.dims.z;
    let rows = params.dims.w;
    let gw = params.cfg.x;
    let gh = params.cfg.y;
    if (gid.x >= gw || gid.y >= gh) {
        return;
    }
    let inv = params.flt2.x;
    let gather = params.flt2.y;
    let xpix = min(2u * gid.x, w - 1u);
    let ypix = min(2u * gid.y, h - 1u);
    let fxy = coords(
        clamp((f32(xpix) + 0.5) * inv - 0.5, 0.0, f32(cols) - 1.0),
        clamp((f32(ypix) + 0.5) * inv - 0.5, 0.0, f32(rows) - 1.0),
        vec2<u32>(cols, rows),
    );
    let f = bilerp2(grid_f[fxy.i00], grid_f[fxy.i10], grid_f[fxy.i01], grid_f[fxy.i11], fxy);
    let fx = f.x * gather;
    let fy = f.y * gather;
    let px = clamp(f32(xpix) + fx, 0.0, f32(w) - 1.0);
    let py = clamp(f32(ypix) + fy, 0.0, f32(h) - 1.0);
    let bxy = coords(
        clamp((px + 0.5) * inv - 0.5, 0.0, f32(cols) - 1.0),
        clamp((py + 0.5) * inv - 0.5, 0.0, f32(rows) - 1.0),
        vec2<u32>(cols, rows),
    );
    let b = bilerp2(grid_b[bxy.i00], grid_b[bxy.i10], grid_b[bxy.i01], grid_b[bxy.i11], bxy);
    let ex = fx + b.x * gather;
    let ey = fy + b.y * gather;
    let consistency = 1.0 / (1.0 + CONF_K * (ex * ex + ey * ey));
    let xl = xpix - min(xpix, 1u);
    let xr = min(xpix + 1u, w - 1u);
    let yu = ypix - min(ypix, 1u);
    let yd = min(ypix + 1u, h - 1u);
    let gx = luma[ypix * w + xr] - luma[ypix * w + xl];
    let gy = luma[yd * w + xpix] - luma[yu * w + xpix];
    let texture = min(0.2 + 3.0 * (abs(gx) + abs(gy)), 1.0);
    conf_grid[gid.y * gw + gid.x] = clamp(consistency * texture, 0.0, 1.0);
}
