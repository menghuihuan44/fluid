// Vertical pass: one invocation per destination pixel, resampling the
// horizontally resampled scratch rows.
//
// Bicubic, Lanczos3 and the adaptive base ring on hard edges because their
// kernels have negative lobes, so this pass also applies the CPU's anti-ringing
// clamp: the rounded luma of the reconstructed sample is clamped to the min/max
// luma of every source pixel inside the horizontal x vertical tap window, and
// the resulting delta is added to the three colour bytes (alpha untouched). The
// window is the same set of edge-clamped taps the CPU combines across its two
// passes, so the clamp is bit-identical.

@group(0) @binding(0) var<storage, read> scratch: array<u32>;
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> src: array<u32>;
@group(0) @binding(6) var<storage, read> h_idx: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.dst_w || y >= params.dst_h {
        return;
    }
    let base = y * params.ny;
    var acc = unpack4(scratch[idx[base] * params.dst_w + x]) * weights[base];
    for (var j = 1u; j < params.ny; j = j + 1u) {
        acc = unpack4(scratch[idx[base + j] * params.dst_w + x]) * weights[base + j] + acc;
    }
    var px = pack4(acc);
    // pad0 carries the per-pipeline flags: bit 0 enables the anti-ringing clamp.
    if (params.pad0 & 1u) != 0u {
        var lo: i32 = 255;
        var hi: i32 = 0;
        let hbase = x * params.nx;
        for (var j = 0u; j < params.ny; j = j + 1u) {
            let srow = idx[base + j] * params.src_w;
            for (var d = 0u; d < params.nx; d = d + 1u) {
                let v = luma(src[srow + h_idx[hbase + d]]);
                lo = min(lo, v);
                hi = max(hi, v);
            }
        }
        let lum = luma(px);
        if lum < lo || lum > hi {
            px = add_delta(px, clamp(lum, lo, hi) - lum);
        }
    }
    dst[y * params.dst_w + x] = px;
}
