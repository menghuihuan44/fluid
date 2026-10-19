// Robust Contrast Adaptive Sharpening (RCAS).
// Ported from AMD FidelityFX-FSR v1 (ffx_fsr1.h, MIT license,
// Copyright (c) 2021 Advanced Micro Devices, Inc.).
//
// The negative lobe is bounded by the local 3x3 min/max so halos stay inside
// the local color range, and a high-pass noise gate keeps film grain from
// being sharpened. Sharpening is scalar (applied to the RGB triplet) which
// avoids the color fringing of per-channel unsharp masks on game UI text.

Texture2D src_tex : register(t0);
SamplerState linear_clamp : register(s0);

cbuffer PerFrame : register(b0) {
    float4 uv_rect;
    float4 hud_rect;
    float4 output_size;
    float4 source_size;
    float4 params0; // z = sharpness 0..1
    float4 params1;
};

struct PSIn {
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

#define FSR_RCAS_LIMIT (0.25 - (1.0 / 16.0))

float3 fetch(int2 p) {
    int2 size = int2(max(output_size.xy, float2(1.0, 1.0)));
    return src_tex.Load(int3(clamp(p, int2(0, 0), size - 1), 0)).rgb;
}

float4 main(PSIn i) : SV_Target {
    int2 ip = int2(i.pos.xy);
    float3 e = fetch(ip);
    float amount = saturate(params0.z);
    if (amount <= 0.0001) {
        return float4(e, 1.0);
    }

    //    b
    //  d e f
    //    h
    float3 b = fetch(ip + int2(0, -1));
    float3 d = fetch(ip + int2(-1, 0));
    float3 f = fetch(ip + int2(1, 0));
    float3 h = fetch(ip + int2(0, 1));

    // FSR "luma times 2": 0.5*r + g + 0.5*b.
    float bL = b.b * 0.5 + (b.r * 0.5 + b.g);
    float dL = d.b * 0.5 + (d.r * 0.5 + d.g);
    float eL = e.b * 0.5 + (e.r * 0.5 + e.g);
    float fL = f.b * 0.5 + (f.r * 0.5 + f.g);
    float hL = h.b * 0.5 + (h.r * 0.5 + h.g);

    // Noise detection: gate the lobe by the local high-pass relative to the
    // local range, so grain and dithering are not amplified.
    float nz = 0.25 * bL + 0.25 * dL + 0.25 * fL + 0.25 * hL - eL;
    float local_max = max(max(max(bL, dL), eL), max(fL, hL));
    float local_min = min(min(min(bL, dL), eL), min(fL, hL));
    nz = saturate(abs(nz) / max(local_max - local_min, 1.0e-4));
    nz = -0.5 * nz + 1.0;

    // Local range limiters (per channel).
    float3 mn4 = min(min(b, d), min(f, h));
    float3 mx4 = max(max(b, d), max(f, h));
    float3 hitMin = min(mn4, e) / max(4.0 * mx4, 1.0e-4);
    float3 hitMax = (1.0 - max(mx4, e)) / max(4.0 * mn4 - 4.0, -1.0);
    float3 lobeRGB = max(-hitMin, hitMax);
    float lobe = max(-FSR_RCAS_LIMIT, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0));

    // Sharpness mapping: slider 0..1 maps to 0..4 "stops"; every stop halves
    // the sharpening response.
    float stops = (1.0 - amount) * 4.0;
    lobe *= exp2(-stops);
    lobe *= nz;

    float rcpL = 1.0 / (4.0 * lobe + 1.0);
    float3 pix = (lobe * (b + d + h + f) + e) * rcpL;
    return float4(pix, 1.0);
}
