// One pyramid level of the coarse-to-fine loop: the parent grid is bilinearly
// resampled at the new block centres, scaled, refined with a 3x3 patch search
// and finally polished by a damped Gauss-Newton pass when it is the target
// level ('fluid-flow' match_flow level loop).
@group(0) @binding(0) var<storage, read> src_luma: array<f32>;
@group(0) @binding(1) var<storage, read> dst_luma: array<f32>;
@group(0) @binding(2) var<storage, read> grid_in: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> grid_out: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> params: Params;

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

// Bilinear gather of a block-grid vector, edge clamped like sample_grid.
fn sample_grid(g: vec2<f32>, cols: u32, rows: u32) -> vec2<f32> {
    let c = coords(g.x, g.y, vec2<u32>(cols, rows));
    return bilerp2(grid_in[c.i00], grid_in[c.i10], grid_in[c.i01], grid_in[c.i11], c);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cols = params.dims.z;
    let rows = params.dims.w;
    if (gid.x >= cols || gid.y >= rows) {
        return;
    }
    let bs = params.cfg.x;
    let patch_step = max(params.cfg.y, 1u);
    // On a repeated pass at the target level the input grid already has the
    // output resolution: the CPU simply refines its previous result, so read
    // the same block instead of bilinearly resampling a parent grid.
    var v = vec2<f32>(0.0, 0.0);
    if (params.cfg.z != 0u) {
        v = grid_in[gid.y * cols + gid.x];
    } else {
        let old_cols = params.extra.x;
        let old_rows = params.extra.y;
        let gx = (f32(gid.x) + 0.5) * (f32(old_cols) / f32(cols)) - 0.5;
        let gy = (f32(gid.y) + 0.5) * (f32(old_rows) / f32(rows)) - 0.5;
        let up = sample_grid(vec2<f32>(gx, gy), old_cols, old_rows);
        v = vec2<f32>(up.x * params.flt.x, up.y * params.flt.y);
    }
    let x0 = gid.x * bs;
    let y0 = gid.y * bs;
    grid_out[gid.y * cols + gid.x] = refine_once(x0, y0, v, patch_step);
}
