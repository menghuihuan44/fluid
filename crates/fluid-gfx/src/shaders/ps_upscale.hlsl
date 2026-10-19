// Spatial reconstruction at output resolution.
//
// Input is the source-resolution blended frame written by ps_blend (or the
// current captured frame when interpolation is off). Splitting the temporal
// and spatial passes means the reconstruction kernel runs once on the final
// image instead of once per warped endpoint, which both costs ~3x fewer
// texture fetches and lets the anti-ringing clamp operate on the frame that
// is actually displayed.
//
// Magnification uses adaptive Catmull-Rom reconstruction. For large output
// surfaces the renderer runs the separable form in two passes: 4 horizontal
// taps into a compact intermediate, then 4 vertical taps into the final frame.
// That costs 8 texture fetches per output pixel instead of 16 for a direct
// 4x4 kernel, with a much smaller working set. A local range clamp on each axis
// keeps text/HUD edges free of cubic overshoot. Minification uses a 4-tap area
// filter so shrinking shimmers far less than bilinear.
//
// params2.z selects the pass:
//   0 = direct source -> output (also handles minification/bilinear)
//   1 = horizontal half of the separable reconstruction
//   2 = vertical/final half, including letterboxing and HDR conversion

Texture2D src_tex : register(t0);
SamplerState linear_clamp : register(s0);

cbuffer PerFrame : register(b0) {
    float4 uv_rect;     // unused: the input already contains only the source window
    float4 hud_rect;
    float4 output_size; // xy = output pixels
    float4 source_size; // xy = input (source window) pixels
    float4 params0;     // x phase, y scale mode, z sharpness, w fg enabled
    float4 params1;     // x motion enabled, y HDR, z aspect lock, w fg strength
    float4 params2;     // x scene threshold, y flow LOD, z pass, w source aspect
};

struct PSIn {
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

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

float3 sample_uv(float2 u) {
    return src_tex.SampleLevel(linear_clamp, saturate(u), 0).rgb;
}

// `p` is in continuous pixel coordinates: source texel i lives at i + 0.5.
// Clamping to texel centers avoids smearing the outermost edge when a cubic
// kernel asks for an out-of-bounds neighbour.
float3 sample_px(float2 p) {
    float2 size = max(source_size.xy, 1.0);
    p = clamp(p, 0.5, size - 0.5);
    return sample_uv(p / size);
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

float3 sample_adaptive(float2 p) {
    float2 q = p - 0.5;
    float2 base = floor(q);
    float2 f = q - base;
    float4 wx = cubic_weights(f.x);
    float4 wy = cubic_weights(f.y);

    float3 acc = 0.0;
    float3 lo = 1.0e9;
    float3 hi = -1.0e9;

    [unroll]
    for (int y = 0; y < 4; ++y) {
        [unroll]
        for (int x = 0; x < 4; ++x) {
            // base is the nearest texel index below the sample point; tap
            // indices are base-1 .. base+2 and centers are index+0.5.
            float3 c = sample_px(base + float2((float)x - 0.5, (float)y - 0.5));
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

// One axis of the separable Catmull-Rom reconstruction. The other axis keeps
// its texel-center coordinate so the intermediate is not resampled twice.
float3 sample_cubic_horizontal(float2 p) {
    float q = p.x - 0.5;
    float base = floor(q);
    float f = q - base;
    float4 w = cubic_weights(f);
    float3 acc = 0.0;
    float3 lo = 1.0e9;
    float3 hi = -1.0e9;
    [unroll]
    for (int x = 0; x < 4; ++x) {
        float3 c = sample_px(float2(base + (float)x - 0.5, p.y));
        acc += c * w[x];
        lo = min(lo, c);
        hi = max(hi, c);
    }
    return clamp(acc, lo, hi);
}

float3 sample_cubic_vertical(float2 p) {
    float q = p.y - 0.5;
    float base = floor(q);
    float f = q - base;
    float4 w = cubic_weights(f);
    float3 acc = 0.0;
    float3 lo = 1.0e9;
    float3 hi = -1.0e9;
    [unroll]
    for (int y = 0; y < 4; ++y) {
        float3 c = sample_px(float2(p.x, base + (float)y - 0.5));
        acc += c * w[y];
        lo = min(lo, c);
        hi = max(hi, c);
    }
    return clamp(acc, lo, hi);
}

// `destination_content` is the actual image rectangle, excluding letterbox
// bars. At 1:1 content inside a larger canvas this keeps us on the cheap
// bilinear/exact path instead of doing a 4x4 cubic reconstruction for nothing.
float3 sample_frame(float2 px, float2 destination_content) {
    float2 out_size = max(destination_content, 1.0);
    float2 src_size = max(source_size.xy, 1.0);

    // At 1:1 output, Catmull-Rom would cost 16 samples per pixel for no
    // benefit. Bilinear at exact pixel centers is both exact and cheap.
    if (abs(out_size.x - src_size.x) < 0.5 && abs(out_size.y - src_size.y) < 0.5) {
        return sample_px(px);
    }

    // Minification: a bilinear tap misses most of the source footprint and
    // shimmers. Four taps spread over the footprint approximate a box filter
    // (well behaved up to ~2x reduction, which covers window->monitor fits).
    if (out_size.x < src_size.x - 0.5 || out_size.y < src_size.y - 0.5) {
        float2 step = src_size / out_size;
        float3 acc =
            sample_px(px + float2( 0.25,  0.25) * step) +
            sample_px(px + float2(-0.25,  0.25) * step) +
            sample_px(px + float2( 0.25, -0.25) * step) +
            sample_px(px + float2(-0.25, -0.25) * step);
        return acc * 0.25;
    }

    int mode = (int)(params0.y + 0.5);
    if (mode == 2) {
        return sample_adaptive(px);
    }
    if (mode == 0) {
        float2 nearest = floor(px) + 0.5;
        return sample_px(nearest);
    }
    return sample_px(px);
}

float4 main(PSIn i) : SV_Target {
    int pass_mode = (int)(params2.z + 0.5);

    // First separable pass: upscale horizontally only. The destination has
    // the final content width and the source height, so y maps one-to-one and
    // remains at a texel center.
    if (pass_mode == 1) {
        float2 src_size = max(source_size.xy, 1.0);
        float2 p = float2(i.uv.x * src_size.x, i.uv.y * src_size.y);
        return float4(sample_cubic_horizontal(p), 1.0);
    }

    float2 input_uv = i.uv;
    float2 out_size = max(output_size.xy, 1.0);
    float2 destination_size = out_size;
    if (params1.z > 0.5) {
        float2 src_size = max(source_size.xy, 1.0);
        float source_aspect = params2.w > 0.01
            ? params2.w
            : src_size.x / max(src_size.y, 1.0);
        float output_aspect = out_size.x / max(out_size.y, 1.0);
        destination_size = out_size;
        if (output_aspect > source_aspect) {
            destination_size.x = out_size.y * source_aspect;
        } else {
            destination_size.y = out_size.x / source_aspect;
        }
        float2 content_origin = (out_size - destination_size) * 0.5;
        float2 local = i.uv * out_size - content_origin;
        if (any(local < 0.0) || any(local >= destination_size)) {
            return float4(0.0, 0.0, 0.0, 1.0);
        }
        input_uv = local / destination_size;
    }

    float2 px = saturate(input_uv) * max(source_size.xy, 1.0);
    float3 color = (pass_mode == 2)
        ? sample_cubic_vertical(px)
        : sample_frame(px, destination_size);
    return float4(to_output(color), 1.0);
}
