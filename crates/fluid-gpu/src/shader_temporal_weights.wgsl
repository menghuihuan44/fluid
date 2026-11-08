// Per-tile motion weights for the motion-adaptive temporal blend.
//
// fluid-upscale's temporal stage no longer applies one global blend factor: it
// splits the destination frame into 16 x 16 pixel tiles, measures the mean
// absolute luma difference between the current resize and the history inside
// each tile, and moves the tile all the way to the current frame when that
// difference is large enough that the history would ghost. Static tiles keep the
// configured weight, so the dither the temporal filter exists to damp still gets
// smoothed.
//
// This kernel reproduces motion_weights_row from crates/fluid-upscale/src/lib.rs
// exactly: the same 4 x 4 sample grid, the same BT.709 luma, the same smoothstep
// ramp from MOTION_LO to MOTION_HI and the same twelve-bit rounding.

@group(0) @binding(0) var<storage, read> current: array<u32>;
@group(0) @binding(1) var<storage, read> history: array<u32>;
@group(0) @binding(2) var<storage, read_write> weights: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

/// Destination tile size of the motion grid, in pixels.
const MOTION_TILE: u32 = 16u;
/// Number of luma samples per tile axis (a 16 x 16 tile becomes 4 x 4 samples).
const MOTION_SAMPLES: u32 = 4u;
/// Tile luma difference below which the configured weight is kept.
const MOTION_LO: f32 = 8.0;
/// Tile luma difference at or above which the tile follows the current frame.
const MOTION_HI: f32 = 40.0;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // params.pad1/pad2 carry the motion grid dimensions (cols, rows).
    let cols = params.pad1;
    if gid.x >= cols || gid.y >= params.pad2 {
        return;
    }
    let step = MOTION_TILE / MOTION_SAMPLES;
    let half = step / 2u;
    let tile_x = gid.x * MOTION_TILE;
    let tile_y = gid.y * MOTION_TILE;
    var sum = 0u;
    for (var sy = 0u; sy < MOTION_SAMPLES; sy = sy + 1u) {
        let y = min(tile_y + sy * step + half, params.dst_h - 1u);
        let row = y * params.dst_w;
        for (var sx = 0u; sx < MOTION_SAMPLES; sx = sx + 1u) {
            let x = min(tile_x + sx * step + half, params.dst_w - 1u);
            let i = row + x;
            sum = sum + u32(abs(luma(current[i]) - luma(history[i])));
        }
    }
    let mean = f32(sum) / f32(MOTION_SAMPLES * MOTION_SAMPLES);
    let t = clamp((mean - MOTION_LO) / (MOTION_HI - MOTION_LO), 0.0, 1.0);
    let ramp = t * t * (3.0 - 2.0 * t);
    let base = f32(params.blend_s);
    let ticks = base + (4096.0 - base) * ramp;
    weights[gid.y * cols + gid.x] = u32(clamp(ticks + 0.5, 0.0, 4096.0));
}
