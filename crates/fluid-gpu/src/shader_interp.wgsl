// Motion-compensated intermediate frame synthesis, matching 'fluid-interp'
// synth exactly: compensate both frames by the scaled dense forward flow,
// bilinearly sample both, average them by alpha and optionally mix the result
// with a plain cross-fade by the configured blend weight.
@group(0) @binding(0) var<storage, read> prev_px: array<u32>;
@group(0) @binding(1) var<storage, read> next_px: array<u32>;
@group(0) @binding(2) var<storage, read> forward: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> confidence: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_px: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.dims.x;
    let h = params.dims.y;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let i = gid.y * w + gid.x;
    let alpha = params.flt.x;
    let one_minus = params.flt.w;
    let blend = params.flt.y;
    let min_conf = params.flt.z;
    let dims = vec2<u32>(w, h);
    let v = forward[i];
    let c = confidence[i];
    let p = unpack4(prev_px[i]);
    let n = unpack4(next_px[i]);
    var value = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (c < min_conf && params.extra.x == 0u) {
        value = p + (n - p) * alpha;
    } else {
        let x = f32(gid.x);
        let y = f32(gid.y);
        let cp = coords(x + alpha * v.x, y + alpha * v.y, dims);
        let pa = bilerp4(
            unpack4(prev_px[cp.i00]), unpack4(prev_px[cp.i10]),
            unpack4(prev_px[cp.i01]), unpack4(prev_px[cp.i11]), cp,
        );
        let cn = coords(x - one_minus * v.x, y - one_minus * v.y, dims);
        let pb = bilerp4(
            unpack4(next_px[cn.i00]), unpack4(next_px[cn.i10]),
            unpack4(next_px[cn.i01]), unpack4(next_px[cn.i11]), cn,
        );
        let mc = pa + (pb - pa) * alpha;
        if (blend >= 1.0) {
            value = mc;
        } else {
            let cf = p + (n - p) * alpha;
            value = cf + (mc - cf) * blend;
        }
    }
    out_px[i] = pack4(value);
}
