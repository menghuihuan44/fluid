// Frame composition + spatially adaptive upscale.
// t0 = previous captured frame, t1 = current captured frame,
// t2 = motion field produced by cs_flow.
Texture2D prev_tex : register(t0);
Texture2D curr_tex : register(t1);
Texture2D flow_tex : register(t2);
SamplerState linear_clamp : register(s0);

cbuffer PerFrame : register(b0) {
    float4 uv_rect;     // source window rectangle in captured monitor UV
    float4 hud_rect;
    float4 output_size; // xy = output pixels
    float4 source_size; // xy = source window pixels in the captured frame
    float4 params0;     // x phase, y scale mode, z sharpness, w fg enabled
    float4 params1;     // x motion enabled, y HDR, z aspect lock, w fg strength
    float4 params2;     // x scene threshold
};

struct PSIn {
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

float luminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

float3 to_output(float3 c) {
    // FP16 desktop duplication usually carries linear scRGB. Convert to an
    // approximate SDR display signal; precise Windows HDR metadata handling is
    // a future improvement.
    if (params1.y > 0.5) {
        c = max(c, 0.0);
        c = c / (1.0 + c);
        c = pow(c, 1.0 / 2.2);
    }
    return saturate(c);
}

float2 pixel_to_uv(float2 p) {
    float2 size = max(source_size.xy, 1.0);
    p = clamp(p, 0.0, size - 1.0);
    return uv_rect.xy + (p / size) * (uv_rect.zw - uv_rect.xy);
}

float3 sample_bilinear(Texture2D tex, float2 p) {
    return tex.SampleLevel(linear_clamp, pixel_to_uv(p), 0).rgb;
}

float4 cubic_weights(float x) {
    // Catmull-Rom reconstruction weights.
    float x2 = x * x;
    float x3 = x2 * x;
    return float4(
        -0.5 * x + x2 - 0.5 * x3,
        1.0 - 2.5 * x2 + 1.5 * x3,
        0.5 * x + 2.0 * x2 - 1.5 * x3,
        -0.5 * x2 + 0.5 * x3);
}

float3 sample_adaptive(Texture2D tex, float2 p) {
    float2 base = floor(p - 0.5);
    float2 f = p - 0.5 - base;
    float4 wx = cubic_weights(f.x);
    float4 wy = cubic_weights(f.y);

    float3 acc = 0.0;
    float3 lo = 1.0;
    float3 hi = 0.0;

    [unroll]
    for (int y = 0; y < 4; ++y) {
        [unroll]
        for (int x = 0; x < 4; ++x) {
            float3 c = sample_bilinear(tex, base + float2(x, y));
            float w = wx[x] * wy[y];
            acc += c * w;
            lo = min(lo, c);
            hi = max(hi, c);
        }
    }

    // Anti-ringing: a Catmull-Rom kernel can overshoot around hard edges.
    // Clamping to the local 4x4 color range keeps text and HUD edges clean.
    return clamp(acc, lo, hi);
}

float3 sample_frame(Texture2D tex, float2 p) {
    // At 1:1 output, Catmull-Rom would cost 16 samples per pixel for no
    // benefit. Bilinear at exact pixel centers is both exact and cheap.
    if (abs(output_size.x - source_size.x) < 0.5 &&
        abs(output_size.y - source_size.y) < 0.5) {
        return sample_bilinear(tex, p);
    }
    int mode = (int)(params0.y + 0.5);
    if (mode == 2) {
        return sample_adaptive(tex, p);
    }
    if (mode == 0) {
        float2 nearest = floor(p) + 0.5;
        return sample_bilinear(tex, nearest);
    }
    return sample_bilinear(tex, p);
}

float4 main(PSIn i) : SV_Target {
    float2 sample_uv = i.uv;
    if (params1.z > 0.5) {
        float2 output_px = i.uv * output_size.xy;
        float source_aspect = source_size.x / max(source_size.y, 1.0);
        float output_aspect = output_size.x / max(output_size.y, 1.0);
        float2 content_size = output_size.xy;
        if (output_aspect > source_aspect) {
            content_size.x = output_size.y * source_aspect;
        } else {
            content_size.y = output_size.x / source_aspect;
        }
        float2 content_origin = (output_size.xy - content_size) * 0.5;
        float2 local = output_px - content_origin;
        if (any(local < 0.0) || any(local >= content_size)) {
            return float4(0.0, 0.0, 0.0, 1.0);
        }
        sample_uv = local / content_size;
    }

    sample_uv = saturate(sample_uv);
    float2 px = sample_uv * source_size.xy;
    float3 current = sample_frame(curr_tex, px);

    float phase = params0.x;
    bool fg_active = (params0.w > 0.5) && (phase > 0.001) && (phase < 0.999);
    if (!fg_active) {
        return float4(to_output(current), 1.0);
    }

    float4 flow_sample = flow_tex.SampleLevel(linear_clamp, sample_uv, 0);
    float2 motion = flow_sample.xy;
    float search_confidence = saturate(flow_sample.w);
    if (params1.x < 0.5) {
        motion = float2(0.0, 0.0);
        search_confidence = 1.0;
    }

    float2 prev_px = px - motion * phase;
    float2 curr_px = px + motion * (1.0 - phase);
    float3 prev_color = sample_frame(prev_tex, prev_px);
    float3 curr_color = sample_frame(curr_tex, curr_px);
    float3 mixed = lerp(prev_color, curr_color, phase);

    // Suppress colour synthesis when the two warped samples disagree:
    // this catches occlusions, exposure changes, and scene cuts.
    float warped_delta = abs(luminance(prev_color) - luminance(curr_color));
    float match_conf = saturate(1.0 - warped_delta * 4.2);

    float3 static_prev = sample_bilinear(prev_tex, px);
    float static_delta = abs(luminance(static_prev) - luminance(current));
    float scene_conf = 1.0 - smoothstep(params2.x, params2.x + 0.18, static_delta);

    float strength = saturate(params1.w);
    // When the motion field is weak, favour the real current frame over a
    // warped blend. This trades a little synthetic smoothness for far less
    // ghosting/double-image artefacts.
    float flow_trust = smoothstep(0.04, 0.38, search_confidence);
    float confidence = match_conf * scene_conf * flow_trust * strength;

    // Never introduce colours that are not present in either endpoint.
    mixed = clamp(mixed, min(prev_color, curr_color), max(prev_color, curr_color));
    float3 result = lerp(current, mixed, confidence);
    return float4(to_output(result), 1.0);
}
