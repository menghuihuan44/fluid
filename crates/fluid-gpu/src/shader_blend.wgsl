// Temporal EMA blend, written in place over the history buffer. Every byte
// (alpha included) follows the CPU's fixed point Q12 blend, so no separate
// history copy is needed.
//
// The weight is per 16 x 16 pixel tile, not global: fluid-upscale's temporal
// stage estimates how much each tile moved (see shader_temporal_weights.wgsl)
// and biases the blend towards the current frame where history would ghost.
// Static tiles get exactly the configured weight.
//
// Tiled 16x16 for the same reason as the copy kernel: a linear 64-wide
// dispatch overflows Vulkan's 65535 workgroup limit on 4K frames.

@group(0) @binding(0) var<storage, read> spatial: array<u32>;
@group(0) @binding(1) var<storage, read_write> history: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

/// Tile size of the motion grid, in destination pixels.
const MOTION_TILE: u32 = 16u;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.dst_w || y >= params.dst_h {
        return;
    }
    let i = y * params.dst_w + x;
    // params.pad1 carries the motion grid width in tiles.
    let tile = (y / MOTION_TILE) * params.pad1 + x / MOTION_TILE;
    let s = i32(weights[tile]);
    let cur = spatial[i];
    let hist = history[i];
    let r = blend_byte(cur & 0xFFu, hist & 0xFFu, s);
    let g = blend_byte((cur >> 8u) & 0xFFu, (hist >> 8u) & 0xFFu, s);
    let b = blend_byte((cur >> 16u) & 0xFFu, (hist >> 16u) & 0xFFu, s);
    let a = blend_byte(cur >> 24u, hist >> 24u, s);
    history[i] = r | (g << 8u) | (b << 16u) | (a << 24u);
}
