// Hierarchical block-motion estimator.
//
// Runs on a 1/8-resolution grid and uses the mip pyramid of the endpoint
// frames so a small candidate grid covers large motion: the search starts at
// LOD 3 (one texel = 8x8 source pixels), then refines the winner down to
// sub-pixel accuracy at LOD 0. Total candidates stay near the old single-scale
// estimator while the covered range grows from ~±22 to ~±80 source pixels on
// a 1080p/4K capture, which is where fast camera/aim motion used to fail.
//
// Every refinement level re-baselines the winner at that level's smoothing so
// costs stay comparable across the pyramid, and falls back to a fresh 5x5
// search around zero motion when the inherited winner cannot beat the static
// hypothesis at that level. That keeps flat-at-coarse content (sky, walls,
// UI panels) from inheriting a garbage coarse vector: a level that cannot
// justify the inherited motion re-decides locally instead.
//
// Flow is stored in source pixels:
//   rg = motion vector (prev -> curr), b = residual cost, a = confidence.
Texture2D prev_tex : register(t0);
Texture2D curr_tex : register(t1);
SamplerState linear_clamp : register(s0);
RWTexture2D<float4> flow_out : register(u0);

cbuffer FlowParams : register(b1) {
    float4 src_rect;    // normalized source sub-rect in the captured monitor
    float4 source_size; // xy = source window size in physical pixels
    float4 flow_size;   // xy = flow texture dimensions
    float4 search_cfg;  // x = coarse step (source px), y = motion penalty
};

float luma(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

float2 pixel_to_uv(float2 p) {
    float2 size = max(source_size.xy, 1.0);
    p = clamp(p, 0.0, size - 1.0);
    return src_rect.xy + (p / size) * (src_rect.zw - src_rect.xy);
}

// Sample the frame luma at a source-pixel position and pyramid level.
float luma_at(Texture2D tex, float2 p, float lod) {
    return luma(tex.SampleLevel(linear_clamp, pixel_to_uv(p), lod).rgb);
}

// Compare a center-plus-cross patch after applying a central-time motion warp.
// Tap spacing grows with the pyramid level so every level sees comparable
// spatial context while each tap still costs one bilinear fetch.
float motion_sad(Texture2D a, Texture2D b, float2 p, float2 motion, float lod) {
    float tap = max(1.5, exp2(lod) * 1.5);
    float sum = 0.0;
    [unroll]
    for (int i = 0; i < 5; ++i) {
        float2 o = float2(0.0, 0.0);
        if (i == 1) o = float2( tap, 0.0);
        if (i == 2) o = float2(-tap, 0.0);
        if (i == 3) o = float2(0.0,  tap);
        if (i == 4) o = float2(0.0, -tap);
        float la = luma_at(a, p + o - motion * 0.5, lod);
        float lb = luma_at(b, p + o + motion * 0.5, lod);
        sum += abs(la - lb);
    }
    return sum * 0.2;
}

float cost_of(Texture2D a, Texture2D b, float2 p, float2 d, float lod, float penalty) {
    return motion_sad(a, b, p, d, lod) + length(d) * penalty;
}

// Re-baseline the inherited winner at `lod`, then either refine it with a 3x3
// neighborhood or, when it cannot beat the static hypothesis at this level,
// re-search a fresh 5x5 grid around zero motion.
void refine_level(Texture2D a, Texture2D b, float2 p, float lod, float step,
                  float penalty, inout float2 best_motion, inout float best_cost) {
    float inherited_cost = cost_of(a, b, p, best_motion, lod, penalty);
    float zero_cost = cost_of(a, b, p, float2(0.0, 0.0), lod, penalty);
    bool reliable = inherited_cost <= zero_cost * 1.02;
    float2 base;
    int r;
    if (reliable) {
        base = best_motion;
        best_cost = inherited_cost;
        r = 1;
    } else {
        // Do not keep patching a coarse vector that has already failed its
        // re-baseline check. Resetting the incumbent to the static hypothesis
        // before the local search prevents a bad coarse match from surviving
        // just because all of its neighbours are even worse.
        base = float2(0.0, 0.0);
        best_motion = float2(0.0, 0.0);
        best_cost = zero_cost;
        r = 2;
    }
    [loop]
    for (int cy = -r; cy <= r; ++cy) {
        [loop]
        for (int cx = -r; cx <= r; ++cx) {
            if (cx == 0 && cy == 0) {
                continue;
            }
            float2 d = base + float2(cx, cy) * step;
            float cost = cost_of(a, b, p, d, lod, penalty);
            if (cost < best_cost) {
                best_cost = cost;
                best_motion = d;
            }
        }
    }
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (float(tid.x) >= flow_size.x || float(tid.y) >= flow_size.y) {
        return;
    }

    float2 p = (float2(tid.xy) + 0.5) / flow_size.xy * source_size.xy;
    float penalty = search_cfg.y;

    // Zero-motion reference at full resolution. A block whose content did not
    // change needs no search at all: emitting an exact zero vector keeps
    // static UI and text perfectly stable and skips the whole pyramid.
    float zero_cost = motion_sad(prev_tex, curr_tex, p, float2(0.0, 0.0), 0.0);
    if (zero_cost <= 0.02) {
        flow_out[tid.xy] = float4(0.0, 0.0, zero_cost, 1.0);
        return;
    }

    float2 best_motion = float2(0.0, 0.0);
    // Costs are only comparable at the same pyramid level. Compare the coarse
    // candidates against a coarse zero-motion hypothesis, not the sharper
    // full-resolution cost above.
    float step3 = search_cfg.x;
    float best_cost = cost_of(prev_tex, curr_tex, p, float2(0.0, 0.0), 3.0, penalty);

    // Level 3 (1/8 linear size): 5x5 candidates. With the coarse step S this
    // covers +/-2*S source pixels around zero.
    [loop]
    for (int cy = -2; cy <= 2; ++cy) {
        [loop]
        for (int cx = -2; cx <= 2; ++cx) {
            float2 d = float2(cx, cy) * step3;
            float cost = cost_of(prev_tex, curr_tex, p, d, 3.0, penalty);
            if (cost < best_cost) {
                best_cost = cost;
                best_motion = d;
            }
        }
    }

    // Refinement ladder: each level halves the granularity of the previous one
    // so the window always spans the remaining error, ending with a fixed
    // sub-pixel lock at LOD 0 (bilinear sampling gives the sub-pixel response).
    refine_level(prev_tex, curr_tex, p, 2.0, max(step3 * 0.5, 2.0), penalty,
                 best_motion, best_cost);
    refine_level(prev_tex, curr_tex, p, 1.0, max(step3 * 0.25, 1.0), penalty,
                 best_motion, best_cost);
    refine_level(prev_tex, curr_tex, p, 0.0, max(step3 * 0.125, 0.5), penalty,
                 best_motion, best_cost);
    refine_level(prev_tex, curr_tex, p, 0.0, 0.25, penalty,
                 best_motion, best_cost);

    // If the full ladder cannot clearly beat the static hypothesis (noise,
    // repeated patterns), prefer the zero vector. This avoids flicker on
    // static UI and keeps noisy flat areas artifact-free.
    float improvement = saturate((zero_cost - best_cost) / max(zero_cost, 0.04));
    float residual_conf = saturate(1.0 - best_cost * 3.0);
    float confidence = saturate(improvement * 2.5) * residual_conf;
    if (best_cost >= zero_cost * 0.98) {
        best_motion = float2(0.0, 0.0);
        confidence = 1.0;
    }

    flow_out[tid.xy] = float4(best_motion, best_cost, confidence);
}
