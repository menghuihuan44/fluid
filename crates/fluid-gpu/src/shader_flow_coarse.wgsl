// Coarsest pyramid level: decimated SAD search over the full radius followed by
// one 3x3 refinement with a separable parabolic fit ('fluid-flow' coarse_search
// plus the first refine_once of match_flow).
@group(0) @binding(0) var<storage, read> src_luma: array<f32>;
@group(0) @binding(1) var<storage, read> dst_luma: array<f32>;
@group(0) @binding(2) var<storage, read_write> grid: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

fn block_sad(x0: u32, y0: u32, dx: i32, dy: i32, step: u32) -> f32 {
    let sw = params.dims.x;
    let sh = params.dims.y;
    let x0c = min(x0, sw);
    let y0c = min(y0, sh);
    let bw = min(params.cfg.x, sw - x0c);
    let bh = min(params.cfg.x, sh - y0c);
    let stride = max(step, 1u);
    let max_x = i32(sw) - 1;
    let max_y = i32(sh) - 1;
    var total = 0.0;
    var y = 0u;
    while (y < bh) {
        let py = y0c + y;
        let sy = u32(clamp(i32(py) + dy, 0, max_y));
        var x = 0u;
        while (x < bw) {
            let px = x0c + x;
            let sx = u32(clamp(i32(px) + dx, 0, max_x));
            total = total + abs(src_luma[py * sw + px] - dst_luma[sy * sw + sx]);
            x = x + stride;
        }
        y = y + stride;
    }
    return total;
}

fn refine_once(x0: u32, y0: u32, start: vec2<f32>, patch_step: u32) -> vec2<f32> {
    let base_x = i32(round_away(start.x));
    let base_y = i32(round_away(start.y));
    var best = vec2<f32>(f32(base_x), f32(base_y));
    var best_cost = 3.0e38;
    var oy = -1;
    while (oy <= 1) {
        var ox = -1;
        while (ox <= 1) {
            let dx = base_x + ox;
            let dy = base_y + oy;
            let cost = block_sad(x0, y0, dx, dy, patch_step) + TIE_BREAK * f32(dx * dx + dy * dy);
            if (cost < best_cost - 1.0e-6) {
                best_cost = cost;
                best = vec2<f32>(f32(dx), f32(dy));
            }
            ox = ox + 1;
        }
        oy = oy + 1;
    }
    let ix = i32(best.x);
    let iy = i32(best.y);
    let s0 = block_sad(x0, y0, ix, iy, patch_step);
    let sm = block_sad(x0, y0, ix - 1, iy, patch_step);
    let sp = block_sad(x0, y0, ix + 1, iy, patch_step);
    let um = block_sad(x0, y0, ix, iy - 1, patch_step);
    let up = block_sad(x0, y0, ix, iy + 1, patch_step);
    return vec2<f32>(best.x + parabola(sm, s0, sp), best.y + parabola(um, s0, up));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cols = params.dims.z;
    let rows = params.dims.w;
    if (gid.x >= cols || gid.y >= rows) {
        return;
    }
    let bs = params.cfg.x;
    let x0 = gid.x * bs;
    let y0 = gid.y * bs;
    let radius = i32(params.cfg.y);
    let step = select(1u, 2u, radius >= 4);
    var best = vec2<f32>(0.0, 0.0);
    var best_cost = block_sad(x0, y0, 0, 0, 2u);
    var dy = -radius;
    while (dy <= radius) {
        var dx = -radius;
        while (dx <= radius) {
            if (dx != 0 || dy != 0) {
                let cost = block_sad(x0, y0, dx, dy, 2u) + TIE_BREAK * f32(dx * dx + dy * dy);
                if (cost < best_cost - 1.0e-6) {
                    best_cost = cost;
                    best = vec2<f32>(f32(dx), f32(dy));
                }
            }
            dx = dx + i32(step);
        }
        dy = dy + i32(step);
    }
    grid[gid.y * cols + gid.x] = refine_once(x0, y0, best, 1u);
}
