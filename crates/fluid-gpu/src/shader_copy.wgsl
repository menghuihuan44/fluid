// Identity copy: source and destination dimensions match, so the spatial step
// is the CPU's exact pass-through. Also used for the Off pass-through fast path
// when the driver keeps it on the GPU.
//
// Tiled 16x16: a linear 64-wide dispatch would exceed Vulkan's 65535 workgroup
// limit on 4K frames (3840 * 2160 / 64 = 129600 groups).

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.dst_w || y >= params.dst_h {
        return;
    }
    let i = y * params.dst_w + x;
    dst[i] = src[i];
}
