// Horizontal pass: one invocation per (destination column, source row).
// Accumulation order matches the CPU: the first tap is a multiply, the rest are
// multiply-and-add onto the running accumulator.

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.dst_w || y >= params.src_h {
        return;
    }
    let row = y * params.src_w;
    let base = x * params.nx;
    var acc = unpack4(src[row + idx[base]]) * weights[base];
    for (var j = 1u; j < params.nx; j = j + 1u) {
        acc = unpack4(src[row + idx[base + j]]) * weights[base + j] + acc;
    }
    scratch[y * params.dst_w + x] = pack4(acc);
}
