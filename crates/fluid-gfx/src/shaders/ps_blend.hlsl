// Temporal blend at source resolution.
//
// Warps both endpoint frames with the smoothed motion field and evaluates a
// per-pixel confidence built from four independent gates:
//   1. endpoint agreement  - the two warped samples must have the same colour;
//                            this is the per-pixel occlusion/disocclusion test;
//   2. flow consistency    - the flow field must agree with itself along the
//                            motion path (catches occlusion boundaries the
//                            block estimator quantized away);
//   3. estimator trust     - the block matcher's own confidence;
//   4. global scene gate   - the average motion-compensated residual over the
//                            whole frame, read from the raw flow mip pyramid.
//                            A hard cut or exposure flash cannot be
//                            motion-compensated and shows up as a large
//                            average residual, so the blend is disabled.
// Note gate 4 is motion-aware, unlike a raw previous-vs-current comparison,
// which would report every moving scene as a "cut" and disable interpolation
// entirely.
//
// The result is written at source resolution; ps_upscale reconstructs it at
// output resolution afterwards.

Texture2D prev_tex : register(t0);
Texture2D curr_tex : register(t1);
Texture2D flow_tex : register(t2);    // smoothed motion field (mip 0)
Texture2D flow_stats : register(t3);  // raw flow with mips, for the global gate
SamplerState linear_clamp : register(s0);

cbuffer PerFrame : register(b0) {
    float4 uv_rect;     // source window rectangle in captured monitor UV
    float4 hud_rect;    // unused here; keeps the constant layout shared
    float4 output_size; // unused here
    float4 source_size; // xy = source window pixels in the captured frame
    float4 params0;     // x phase, y scale mode, z sharpness, w fg enabled
    float4 params1;     // x motion enabled, y HDR, z aspect lock, w fg strength
    float4 params2;     // x scene threshold, y flow mip LOD for the global gate
};

struct PSIn {
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

float2 pixel_to_uv(float2 p) {
    float2 size = max(source_size.xy, 1.0);
    p = clamp(p, 0.0, size - 1.0);
    return uv_rect.xy + (p / size) * (uv_rect.zw - uv_rect.xy);
}

float3 sample_src(Texture2D tex, float2 p) {
    return tex.SampleLevel(linear_clamp, pixel_to_uv(p), 0).rgb;
}

float match_of(float3 a, float3 b) {
    float3 delta = abs(a - b);
    // Per-channel maximum, not just luminance: a luminance test passes when
    // the warped samples disagree in hue, which reads as colour ghosting.
    float d = max(delta.r, max(delta.g, delta.b));
    return saturate(1.0 - d * 3.5);
}

float4 main(PSIn i) : SV_Target {
    float2 px = i.uv * max(source_size.xy, 1.0);
    float phase = saturate(params0.x);
    bool fg_active = (params0.w > 0.5) && (phase > 0.001) && (phase < 0.999);
    float3 current = sample_src(curr_tex, px);
    if (!fg_active) {
        return float4(current, 1.0);
    }

    float3 prev_color;
    float3 curr_color;
    float3 mixed;
    float confidence;

    float strength = saturate(params1.w);
    float2 size = max(source_size.xy, 1.0);

    if (params1.x > 0.5) {
        // Motion-compensated path.
        float2 flow_uv = px / size;
        float4 flow_sample = flow_tex.SampleLevel(linear_clamp, flow_uv, 0);
        float2 motion = flow_sample.xy;
        float search_confidence = saturate(flow_sample.w);

        float2 prev_px = px - motion * phase;
        float2 curr_px = px + motion * (1.0 - phase);
        prev_color = sample_src(prev_tex, prev_px);
        curr_color = sample_src(curr_tex, curr_px);
        mixed = lerp(prev_color, curr_color, phase);

        float match_conf = match_of(prev_color, curr_color);

        // Occlusion via flow self-consistency: sample the field at both warped
        // positions. On a covering/revealing boundary the field disagrees with
        // itself and the blend cannot be trusted at this pixel.
        float4 f1 = flow_tex.SampleLevel(linear_clamp, prev_px / size, 0);
        float4 f2 = flow_tex.SampleLevel(linear_clamp, curr_px / size, 0);
        float divergence = length(f1.xy - motion) + length(f2.xy - motion);
        float occ_conf = saturate(1.0 - divergence / 24.0);

        // Global scene gate from the coarsest flow mip (~2x2 average of the
        // motion-compensated residual across the frame).
        float4 stats = flow_stats.SampleLevel(linear_clamp, flow_uv, params2.y);
        float threshold = params2.x;
        float scene_gate = 1.0 - smoothstep(threshold, threshold + 0.18, saturate(stats.z));

        float flow_trust = smoothstep(0.05, 0.40, search_confidence);
        confidence = match_conf * occ_conf * flow_trust * scene_gate;
    } else {
        // Blend without motion vectors (weak-GPU mode): only endpoint
        // agreement gates the result.
        prev_color = sample_src(prev_tex, px);
        curr_color = sample_src(curr_tex, px);
        mixed = lerp(prev_color, curr_color, phase);
        confidence = match_of(prev_color, curr_color);
    }

    // Never introduce colours that are not present in either endpoint.
    mixed = clamp(mixed, min(prev_color, curr_color), max(prev_color, curr_color));
    // Blend confidence is a gate, not a permanent 15% ghost of the current
    // frame. High-confidence motion produces the pure interpolated result;
    // low-confidence pixels stay exactly on the real current frame. The
    // strength slider moves the gate threshold: lower values demand more
    // confidence before interpolation is allowed.
    float gate_lo = 1.0 - strength;
    float gate_hi = min(1.0, 1.0 - 0.35 * strength);
    float trust = (strength <= 0.001)
        ? 0.0
        : smoothstep(gate_lo, max(gate_hi, gate_lo + 0.05), confidence);
    float3 result = lerp(current, mixed, trust);
    return float4(result, 1.0);
}
