// 2x box downscale of one luma pyramid level ('fluid-flow' downscale2).
@group(0) @binding(0) var<storage, read> src_luma: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst_luma: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.dims.x;
    let h = params.dims.y;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let sw = params.extra.x;
    let sh = params.extra.y;
    let sx = min(gid.x * 2u, sw - 1u);
    let sx1 = min(sx + 1u, sw - 1u);
    let sy = min(gid.y * 2u, sh - 1u);
    let sy1 = min(sy + 1u, sh - 1u);
    let v = (src_luma[sy * sw + sx] + src_luma[sy * sw + sx1] + src_luma[sy1 * sw + sx]
        + src_luma[sy1 * sw + sx1]) * 0.25;
    dst_luma[gid.y * w + gid.x] = v;
}
