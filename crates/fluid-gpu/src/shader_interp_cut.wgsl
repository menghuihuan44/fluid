// Reduce the per-workgroup luma difference sums into a single scene-cut metric.
@group(0) @binding(0) var<storage, read> partials: array<u32>;
@group(0) @binding(1) var<storage, read_write> cut: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> reduce_buf: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
    let n = params.extra.x;
    var acc = 0u;
    var i = lid;
    while (i < n) {
        acc = acc + partials[i];
        i = i + 256u;
    }
    reduce_buf[lid] = acc;
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
        cut[0] = reduce_buf[0];
    }
}
