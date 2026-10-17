// Contrast-adaptive sharpening pass.
// Textures are sampled at native output resolution, so this pass is cheap and
// operates on the already-composited frame rather than on the compressed source.
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

float luminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

float4 main(PSIn i) : SV_Target {
    float2 uv = saturate(i.uv);
    float3 c = src_tex.SampleLevel(linear_clamp, uv, 0).rgb;
    float amount = saturate(params0.z);
    if (amount <= 0.0001) {
        return float4(c, 1.0);
    }

    float2 tx = 1.0 / max(output_size.xy, 1.0);
    float3 n = src_tex.SampleLevel(linear_clamp, uv + float2(0.0, -tx.y), 0).rgb;
    float3 s = src_tex.SampleLevel(linear_clamp, uv + float2(0.0,  tx.y), 0).rgb;
    float3 e = src_tex.SampleLevel(linear_clamp, uv + float2( tx.x, 0.0), 0).rgb;
    float3 w = src_tex.SampleLevel(linear_clamp, uv + float2(-tx.x, 0.0), 0).rgb;
    float3 blur = (n + s + e + w) * 0.25;
    float3 lo = min(min(n, s), min(e, w));
    float3 hi = max(max(n, s), max(e, w));

    // RCAS-like local contrast adaptation: high-contrast edges receive less
    // boost than soft texture detail, reducing halos and shimmering.
    float local_contrast = luminance(hi) - luminance(lo);
    float adaptive = amount * lerp(1.25, 0.65, saturate(local_contrast * 4.0));
    float3 sharpened = c + (c - blur) * adaptive;
    sharpened = clamp(sharpened, lo, hi);
    return float4(sharpened, 1.0);
}
