// HUD 通道：预乘 alpha 的 CPU 光栅化文字纹理，叠加在主画面左上角。
Texture2D hud_tex : register(t0);
SamplerState linear_clamp : register(s0);

cbuffer PerFrame : register(b0) {
    float4 uv_rect;
    float4 hud_rect;
};

struct PSIn {
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

float4 main(PSIn i) : SV_Target {
    float2 suv = lerp(hud_rect.xy, hud_rect.zw, i.uv);
    return hud_tex.Sample(linear_clamp, suv);
}
