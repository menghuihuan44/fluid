// Coarse-to-fine block-motion estimator.
// Runs at 1/8 source resolution. Flow is stored in source pixels:
//   rg = motion vector, b = residual, a = confidence.
Texture2D prev_tex : register(t0);
Texture2D curr_tex : register(t1);
SamplerState linear_clamp : register(s0);
RWTexture2D<float4> flow_out : register(u0);

cbuffer FlowParams : register(b1) {
    float4 src_rect;    // normalized source sub-rect in the captured monitor
    float4 source_size; // xy = source window size in physical pixels
    float4 flow_size;   // xy = flow texture dimensions
    float4 search_cfg;  // x = coarse step, y = refine step, z = motion penalty
};

float luma(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

float2 pixel_to_uv(float2 p) {
    float2 size = max(source_size.xy, 1.0);
    p = clamp(p, 0.0, size - 1.0);
    return src_rect.xy + (p / size) * (src_rect.zw - src_rect.xy);
}

float luma_at(Texture2D tex, float2 p) {
    return luma(tex.SampleLevel(linear_clamp, pixel_to_uv(p), 0).rgb);
}

// Compare a center-plus-cross patch after applying a central-time motion warp.
// Five taps keep the estimator cheap enough for high-refresh 1440p/4K output.
float motion_sad(Texture2D a, Texture2D b, float2 p, float2 motion) {
    float sum = 0.0;
    [unroll]
    for (int i = 0; i < 5; ++i) {
        float2 q = p;
        if (i == 1) q += float2(2.0, 0.0);
        if (i == 2) q += float2(-2.0, 0.0);
        if (i == 3) q += float2(0.0, 2.0);
        if (i == 4) q += float2(0.0, -2.0);
        float la = luma_at(a, q - motion * 0.5);
        float lb = luma_at(b, q + motion * 0.5);
        sum += abs(la - lb);
    }
    return sum * 0.2;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (float(tid.x) >= flow_size.x || float(tid.y) >= flow_size.y) {
        return;
    }

    float2 p = (float2(tid.xy) + 0.5) / flow_size.xy * source_size.xy;
    float2 best_motion = float2(0.0, 0.0);
    float best_cost = motion_sad(prev_tex, curr_tex, p, best_motion) +
                      length(best_motion) * search_cfg.z;

    // Coarse search: 5x5 candidates. search_cfg.x controls the radius.
    // Keep this as a real loop: fully unrolling 34 block matches makes shader
    // compilation take seconds on some systems, while the GPU executes this
    // uniform loop cheaply at the reduced flow resolution.
    [loop]
    for (int cy = -2; cy <= 2; ++cy) {
        [loop]
        for (int cx = -2; cx <= 2; ++cx) {
            float2 d = float2(cx, cy) * search_cfg.x;
            float cost = motion_sad(prev_tex, curr_tex, p, d) + length(d) * search_cfg.z;
            if (cost < best_cost) {
                best_cost = cost;
                best_motion = d;
            }
        }
    }

    // Refinement: 3x3 candidates around the coarse winner.
    float2 coarse = best_motion;
    [loop]
    for (int ry = -1; ry <= 1; ++ry) {
        [loop]
        for (int rx = -1; rx <= 1; ++rx) {
            float2 d = coarse + float2(rx, ry) * search_cfg.y;
            float cost = motion_sad(prev_tex, curr_tex, p, d) + length(d) * search_cfg.z;
            if (cost < best_cost) {
                best_cost = cost;
                best_motion = d;
            }
        }
    }

    float zero_cost = motion_sad(prev_tex, curr_tex, p, float2(0.0, 0.0));
    float improvement = saturate((zero_cost - best_cost) / max(zero_cost, 0.04));
    float residual_conf = saturate(1.0 - best_cost * 3.0);
    float confidence = saturate(improvement * 2.5) * residual_conf;

    // If the static hypothesis is already as good as the search winner,
    // prefer the zero vector. This avoids flicker on static UI/text.
    if (best_cost >= zero_cost * 0.98) {
        best_motion = float2(0.0, 0.0);
        confidence = 1.0;
    }

    flow_out[tid.xy] = float4(best_motion, best_cost, confidence);
}
