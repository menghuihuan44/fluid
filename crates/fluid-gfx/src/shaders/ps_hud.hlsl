// Real-time monitor HUD: premultiplied-alpha text/background, scissored by the
// renderer so the full-screen pass only touches the panel rectangle.
Texture2D hud_tex : register(t0);
SamplerState linear_clamp : register(s0);

cbuffer HudParams : register(b1) {
    float4 hud_tex_rect;  // source UV subrect of the HUD texture
    float4 hud_dest_rect; // destination rectangle in normalized screen coordinates
};

struct PSIn {
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

float4 main(PSIn i) : SV_Target {
    float2 size = hud_dest_rect.zw - hud_dest_rect.xy;
    if (size.x <= 0.0 || size.y <= 0.0) {
        return float4(0.0, 0.0, 0.0, 0.0);
    }
    float2 local = (i.uv - hud_dest_rect.xy) / size;
    if (any(local < 0.0) || any(local > 1.0)) {
        return float4(0.0, 0.0, 0.0, 0.0);
    }
    float2 uv = lerp(hud_tex_rect.xy, hud_tex_rect.zw, local);
    return hud_tex.Sample(linear_clamp, uv);
}
