// Nearest-neighbour resize in one pass: both axes are pure index selections.

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read> xidx: array<u32>;
@group(0) @binding(2) var<storage, read> yidx: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.dst_w || y >= params.dst_h {
        return;
    }
    let sx = xidx[x];
    let sy = yidx[y];
    dst[y * params.dst_w + x] = src[sy * params.src_w + sx];
}
