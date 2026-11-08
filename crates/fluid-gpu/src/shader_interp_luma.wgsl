// Canonical RGBA textures to luma planes plus per-workgroup sums of the
// absolute luma difference used by the scene-cut detector. The sums are scaled
// by 64 so a u32 stays exact for 8K frames; the CPU side divides by
// 64 * area to obtain the mean absolute difference.
@group(0) @binding(0) var<storage, read> prev_px: array<u32>;
@group(0) @binding(1) var<storage, read> next_px: array<u32>;
@group(0) @binding(2) var<storage, read_write> luma_prev: array<f32>;
@group(0) @binding(3) var<storage, read_write> luma_next: array<f32>;
@group(0) @binding(4) var<storage, read_write> partials: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> reduce_buf: array<u32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let w = params.dims.x;
    let h = params.dims.y;
    var scaled = 0u;
    if (gid.x < w && gid.y < h) {
        let i = gid.y * w + gid.x;
        let a = luma_of(prev_px[i]);
        let b = luma_of(next_px[i]);
        luma_prev[i] = a;
        luma_next[i] = b;
        scaled = u32(abs(a - b) * 64.0 + 0.5);
    }
    reduce_buf[lid] = scaled;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
        if (lid < stride) {
            reduce_buf[lid] = reduce_buf[lid] + reduce_buf[lid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid == 0u) {
        partials[wid.y * params.extra.x + wid.x] = reduce_buf[0];
    }
}
