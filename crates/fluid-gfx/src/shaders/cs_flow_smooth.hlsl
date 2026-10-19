// Bilateral smoothing of the block motion field.
//
// Block matching produces a piecewise-constant vector field (one vector per
// 8x8 source pixels). Warping with it shows block seams that read as ghosting
// or doubling after the temporal blend. A confidence- and vector-aware 3x3
// filter removes the seams while refusing to smooth across motion
// discontinuities (object edges), which is where real occlusion lives.
//
// Cost and confidence are carried through unchanged from the estimator; only
// the vector is filtered.

Texture2D flow_in : register(t0);
RWTexture2D<float4> flow_out : register(u0);

cbuffer FlowParams : register(b1) {
    float4 src_rect;
    float4 source_size;
    float4 flow_size;
    float4 search_cfg;
};

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    int2 bounds = int2(max(flow_size.xy, float2(1.0, 1.0)));
    if (tid.x >= (uint)bounds.x || tid.y >= (uint)bounds.y) {
        return;
    }

    float4 c = flow_in[int2(tid.xy)];
    float2 acc = c.xy;
    float wsum = 1.0;

    [unroll]
    for (int dy = -1; dy <= 1; ++dy) {
        [unroll]
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            int2 q = clamp(int2(tid.xy) + int2(dx, dy), int2(0, 0), bounds - 1);
            float4 n = flow_in[q];
            // Gaussian-ish spatial kernel; diagonal neighbours weigh less.
            float spatial = (dx == 0 || dy == 0) ? 0.6 : 0.3;
            // Ignore neighbours that could not measure motion reliably.
            float conf = saturate(n.w);
            // Refuse to average across a motion discontinuity: a neighbour
            // whose vector differs strongly from the center describes a
            // different moving surface.
            float2 dv = n.xy - c.xy;
            float sim = exp(-dot(dv, dv) * 0.02);
            float w = spatial * conf * sim;
            acc += n.xy * w;
            wsum += w;
        }
    }

    flow_out[int2(tid.xy)] = float4(acc / wsum, c.z, c.w);
}
