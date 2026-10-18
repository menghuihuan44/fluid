//! Minimal renderer for the independent real-time HUD window.
//!
//! This renderer intentionally avoids the capture/composition pipeline: it only
//! updates a small CPU-rasterized text texture and draws one full-window quad.

use crate::error::GfxError;
use crate::hud::HudTexture;
use crate::shaders;
use windows::Win32::Graphics::Direct3D::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11BlendState, ID3D11Buffer, ID3D11Device, ID3D11DeviceContext, ID3D11PixelShader,
    ID3D11RasterizerState, ID3D11RenderTargetView, ID3D11SamplerState, ID3D11VertexShader,
    D3D11_BIND_CONSTANT_BUFFER, D3D11_BLEND_DESC, D3D11_BLEND_INV_SRC_ALPHA, D3D11_BLEND_ONE,
    D3D11_BLEND_OP_ADD, D3D11_BUFFER_DESC, D3D11_COLOR_WRITE_ENABLE_ALL, D3D11_COMPARISON_NEVER,
    D3D11_CULL_NONE, D3D11_FILL_SOLID, D3D11_FILTER_MIN_MAG_MIP_LINEAR, D3D11_RASTERIZER_DESC,
    D3D11_RENDER_TARGET_BLEND_DESC, D3D11_SAMPLER_DESC, D3D11_TEXTURE_ADDRESS_CLAMP,
    D3D11_USAGE_DEFAULT, D3D11_VIEWPORT,
};

#[repr(C)]
#[derive(Clone, Copy)]
struct HudParams {
    tex_rect: [f32; 4],
    dest_rect: [f32; 4],
}

pub struct HudRenderer {
    ctx: ID3D11DeviceContext,
    vs: ID3D11VertexShader,
    ps: ID3D11PixelShader,
    sampler: ID3D11SamplerState,
    cbuf: ID3D11Buffer,
    blend: ID3D11BlendState,
    rasterizer: ID3D11RasterizerState,
    hud: HudTexture,
    clear_color: [f32; 4],
}

impl HudRenderer {
    pub fn new(device: &ID3D11Device, ctx: &ID3D11DeviceContext) -> Result<Self, GfxError> {
        let bins = shaders::compile_hud_only()?;
        unsafe {
            let mut vs = None;
            let mut ps = None;
            device.CreateVertexShader(&bins.vs, None, Some(&mut vs))?;
            device.CreatePixelShader(&bins.ps_hud, None, Some(&mut ps))?;

            let sdesc = D3D11_SAMPLER_DESC {
                Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                MipLODBias: 0.0,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D11_COMPARISON_NEVER,
                BorderColor: [0.0; 4],
                MinLOD: 0.0,
                MaxLOD: f32::MAX,
            };
            let mut sampler = None;
            device.CreateSamplerState(&sdesc, Some(&mut sampler))?;

            let mut cbuf = None;
            device.CreateBuffer(
                &D3D11_BUFFER_DESC {
                    ByteWidth: std::mem::size_of::<HudParams>() as u32,
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: (D3D11_BIND_CONSTANT_BUFFER.0) as u32,
                    CPUAccessFlags: 0,
                    MiscFlags: 0,
                    StructureByteStride: 0,
                },
                None,
                Some(&mut cbuf),
            )?;

            let premult = D3D11_RENDER_TARGET_BLEND_DESC {
                BlendEnable: true.into(),
                SrcBlend: D3D11_BLEND_ONE,
                DestBlend: D3D11_BLEND_INV_SRC_ALPHA,
                BlendOp: D3D11_BLEND_OP_ADD,
                SrcBlendAlpha: D3D11_BLEND_ONE,
                DestBlendAlpha: D3D11_BLEND_INV_SRC_ALPHA,
                BlendOpAlpha: D3D11_BLEND_OP_ADD,
                RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8,
            };
            let blend_desc = D3D11_BLEND_DESC {
                AlphaToCoverageEnable: false.into(),
                IndependentBlendEnable: false.into(),
                RenderTarget: [premult; 8],
            };
            let mut blend = None;
            device.CreateBlendState(&blend_desc, Some(&mut blend))?;

            let raster_desc = D3D11_RASTERIZER_DESC {
                FillMode: D3D11_FILL_SOLID,
                CullMode: D3D11_CULL_NONE,
                FrontCounterClockwise: false.into(),
                DepthBias: 0,
                DepthBiasClamp: 0.0,
                SlopeScaledDepthBias: 0.0,
                DepthClipEnable: true.into(),
                ScissorEnable: false.into(),
                MultisampleEnable: false.into(),
                AntialiasedLineEnable: false.into(),
            };
            let mut rasterizer = None;
            device.CreateRasterizerState(&raster_desc, Some(&mut rasterizer))?;

            let hud = HudTexture::new(device, ctx)?;
            Ok(Self {
                ctx: ctx.clone(),
                vs: vs.unwrap(),
                ps: ps.unwrap(),
                sampler: sampler.unwrap(),
                cbuf: cbuf.unwrap(),
                blend: blend.unwrap(),
                rasterizer: rasterizer.unwrap(),
                hud,
                clear_color: [0.02, 0.025, 0.035, 1.0],
            })
        }
    }

    /// Upload text and return the panel size in pixels.
    pub fn update_text(
        &mut self,
        text: &str,
        font_scale: u8,
        background_color: [u8; 3],
    ) -> (u32, u32) {
        self.clear_color = [
            background_color[0] as f32 / 255.0,
            background_color[1] as f32 / 255.0,
            background_color[2] as f32 / 255.0,
            1.0,
        ];
        self.hud
            .update_colored(&self.ctx, text, font_scale, background_color, 255)
    }

    pub fn draw(
        &self,
        rtv: &ID3D11RenderTargetView,
        width: u32,
        height: u32,
    ) -> Result<(), GfxError> {
        let (content_w, content_h) = self.hud.content_size();
        let (texture_w, texture_h) = self.hud.size();
        if content_w == 0 || content_h == 0 || width == 0 || height == 0 {
            return Ok(());
        }
        let params = HudParams {
            tex_rect: [
                0.0,
                0.0,
                content_w as f32 / texture_w.max(1) as f32,
                content_h as f32 / texture_h.max(1) as f32,
            ],
            dest_rect: [0.0, 0.0, 1.0, 1.0],
        };
        unsafe {
            self.ctx.UpdateSubresource(
                &self.cbuf,
                0,
                None,
                &params as *const HudParams as *const core::ffi::c_void,
                0,
                0,
            );
            self.ctx
                .OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);
            self.ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            self.ctx.ClearRenderTargetView(rtv, &self.clear_color);
            self.ctx
                .IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            self.ctx.VSSetShader(&self.vs, None);
            self.ctx.PSSetShader(&self.ps, None);
            self.ctx
                .PSSetConstantBuffers(1, Some(&[Some(self.cbuf.clone())]));
            self.ctx
                .PSSetShaderResources(0, Some(&[Some(self.hud.srv().clone())]));
            self.ctx
                .PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
            self.ctx
                .OMSetBlendState(Some(&self.blend), None, 0xffff_ffff);
            self.ctx.RSSetState(Some(&self.rasterizer));
            self.ctx.Draw(3, 0);
        }
        Ok(())
    }
}
