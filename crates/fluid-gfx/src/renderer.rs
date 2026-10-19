//! GPU render pipeline:
//! captured frame ring (with lazy mip pyramid) -> optional hierarchical motion
//! field + bilateral smoothing -> source-resolution temporal blend -> optional
//! separable output-resolution reconstruction -> optional contrast-adaptive
//! sharpening -> HUD -> waitable swapchain.
//!
//! Everything stays on the GPU. The CPU never copies pixels and only writes a
//! few constant-buffer values per presented frame.
//!
//! The pipeline is split so each pass does one job:
//!   ps_blend    (source res)   motion-compensated temporal blend + occlusion,
//!                              confidence and scene-cut gating;
//!   ps_upscale  (output res)   two-pass separable Catmull-Rom for magnification,
//!                              single-pass area filter for minification,
//!                              letterboxing, HDR conversion;
//!   ps_rcas     (output res)   FSR1-style robust contrast-adaptive sharpening.
//! Native 1:1 output bypasses the blend/scale intermediates, and a direct RCAS
//! path avoids two full-frame passes when only sharpening is requested.

use crate::error::GfxError;
use crate::hud::HudTexture;
use crate::shaders;
use std::time::{Duration, Instant};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11BlendState, ID3D11Buffer, ID3D11ComputeShader, ID3D11Device, ID3D11DeviceContext,
    ID3D11PixelShader, ID3D11Query, ID3D11RasterizerState, ID3D11RenderTargetView,
    ID3D11SamplerState, ID3D11ShaderResourceView, ID3D11Texture2D, ID3D11UnorderedAccessView,
    ID3D11VertexShader, D3D11_BIND_CONSTANT_BUFFER, D3D11_BIND_RENDER_TARGET,
    D3D11_BIND_SHADER_RESOURCE, D3D11_BIND_UNORDERED_ACCESS, D3D11_BLEND_DESC,
    D3D11_BLEND_INV_SRC_ALPHA, D3D11_BLEND_ONE, D3D11_BLEND_OP_ADD, D3D11_BLEND_ZERO,
    D3D11_BUFFER_DESC, D3D11_COLOR_WRITE_ENABLE_ALL, D3D11_COMPARISON_NEVER, D3D11_CULL_NONE,
    D3D11_FILL_SOLID, D3D11_FILTER_MIN_MAG_MIP_LINEAR, D3D11_QUERY,
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT, D3D11_QUERY_DESC, D3D11_QUERY_TIMESTAMP,
    D3D11_QUERY_TIMESTAMP_DISJOINT, D3D11_RASTERIZER_DESC, D3D11_RENDER_TARGET_BLEND_DESC,
    D3D11_RESOURCE_MISC_GENERATE_MIPS, D3D11_SAMPLER_DESC, D3D11_TEXTURE2D_DESC,
    D3D11_TEXTURE_ADDRESS_CLAMP, D3D11_USAGE_DEFAULT, D3D11_VIEWPORT,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_SAMPLE_DESC,
};

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct HudParams {
    tex_rect: [f32; 4],
    dest_rect: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PerFrame {
    uv_rect: [f32; 4],
    hud_rect: [f32; 4],
    output_size: [f32; 4],
    source_size: [f32; 4],
    params0: [f32; 4],
    params1: [f32; 4],
    params2: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FlowParams {
    src_rect: [f32; 4],
    source_size: [f32; 4],
    flow_size: [f32; 4],
    search_cfg: [f32; 4],
}

#[derive(Clone, Copy)]
pub struct RenderOptions {
    /// 0 = nearest, 1 = bilinear, 2 = adaptive Catmull-Rom.
    pub scale_mode: u32,
    /// 0..=1.
    pub sharpness: f32,
    pub fg_enabled: bool,
    pub motion_compensation: bool,
    /// Preserve source aspect ratio and letterbox when output aspect differs.
    pub preserve_aspect: bool,
    /// 0..=1 interpolation strength.
    pub fg_strength: f32,
    /// 0..=1 scene-cut protection; higher = fall back to the real frame sooner.
    pub scene_protection: f32,
    /// 0..=1 interpolation position between previous and current capture.
    pub fg_phase: f32,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            scale_mode: 2,
            sharpness: 0.3,
            fg_enabled: false,
            motion_compensation: true,
            preserve_aspect: true,
            fg_strength: 0.85,
            scene_protection: 0.6,
            fg_phase: 1.0,
        }
    }
}

#[derive(Clone, Copy)]
pub struct HudOptions {
    /// 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right, 4=custom.
    pub anchor: u32,
    pub margin: f32,
    pub custom_x: f32,
    pub custom_y: f32,
    pub font_scale: u8,
    pub background_alpha: u8,
}

impl Default for HudOptions {
    fn default() -> Self {
        Self {
            anchor: 0,
            margin: 12.0,
            custom_x: 0.5,
            custom_y: 0.5,
            font_scale: 2,
            background_alpha: 160,
        }
    }
}

pub struct RenderTargetInfo<'a> {
    pub rtv: &'a ID3D11RenderTargetView,
    pub width: f32,
    pub height: f32,
    pub uv_rect: [f32; 4],
}

pub struct RenderRequest<'a> {
    pub target: RenderTargetInfo<'a>,
    pub previous: &'a FrameSnapshot,
    pub current: &'a FrameSnapshot,
    pub options: RenderOptions,
    pub hud_text: Option<&'a str>,
    pub hud: HudOptions,
}

#[derive(Clone)]
pub struct FrameSnapshot {
    srv: ID3D11ShaderResourceView,
    tex: ID3D11Texture2D,
    slot: usize,
    pub qpc: i64,
    pub seq: u64,
}

pub struct FrameSlot {
    tex: ID3D11Texture2D,
    srv: ID3D11ShaderResourceView,
    pub qpc: i64,
    pub seq: u64,
    /// True once the mip pyramid of this slot has been generated for the
    /// hierarchical motion estimator. Mips are built lazily so a session with
    /// frame generation disabled never pays for them.
    mips_valid: bool,
}

struct QueryPair {
    disjoint: ID3D11Query,
    start: ID3D11Query,
    end: ID3D11Query,
    valid: bool,
}

struct OutputTarget {
    _tex: ID3D11Texture2D,
    srv: ID3D11ShaderResourceView,
    rtv: ID3D11RenderTargetView,
    w: u32,
    h: u32,
}

struct FlowTarget {
    _tex: ID3D11Texture2D,
    srv: ID3D11ShaderResourceView,
    uav: ID3D11UnorderedAccessView,
    w: u32,
    h: u32,
}

struct BlendTarget {
    _tex: ID3D11Texture2D,
    srv: ID3D11ShaderResourceView,
    rtv: ID3D11RenderTargetView,
    w: u32,
    h: u32,
    format: DXGI_FORMAT,
}

pub struct Renderer {
    device: ID3D11Device,
    ctx: ID3D11DeviceContext,
    vs: ID3D11VertexShader,
    ps_blend: ID3D11PixelShader,
    ps_upscale: ID3D11PixelShader,
    ps_rcas: ID3D11PixelShader,
    ps_hud: ID3D11PixelShader,
    cs_flow: ID3D11ComputeShader,
    cs_flow_smooth: ID3D11ComputeShader,
    sampler: ID3D11SamplerState,
    cbuf: ID3D11Buffer,
    flow_cbuf: ID3D11Buffer,
    hud_cbuf: ID3D11Buffer,
    blend_opaque: ID3D11BlendState,
    blend_premult: ID3D11BlendState,
    rasterizer: ID3D11RasterizerState,
    rasterizer_scissor: ID3D11RasterizerState,
    hud: HudTexture,
    hud_last_update: Option<Instant>,
    hud_last_style: Option<(u8, u8)>,
    ring: Vec<FrameSlot>,
    ring_w: u32,
    ring_h: u32,
    ring_format: Option<DXGI_FORMAT>,
    ring_len: usize,
    head: usize,
    seq: u64,
    output: Option<OutputTarget>,
    blend: Option<BlendTarget>,
    /// Intermediate for the two-pass separable reconstruction. It stores the
    /// horizontally scaled image at source height and never leaves the GPU.
    scale: Option<BlendTarget>,
    flow: Option<FlowTarget>,
    flow_smooth: Option<FlowTarget>,
    flow_key: Option<(u64, u64)>,
    /// Mip level of the raw flow texture whose ~2x2 average approximates a
    /// whole-frame statistic for the scene-cut gate.
    flow_lod: f32,
    queries: [QueryPair; 2],
    q_slot: usize,
    gpu_us: f32,
}

fn sample_desc() -> DXGI_SAMPLE_DESC {
    DXGI_SAMPLE_DESC {
        Count: 1,
        Quality: 0,
    }
}

fn create_2d(
    device: &ID3D11Device,
    w: u32,
    h: u32,
    format: DXGI_FORMAT,
    bind_flags: u32,
    mips: u32,
    misc: u32,
) -> Result<ID3D11Texture2D, GfxError> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: w.max(1),
        Height: h.max(1),
        MipLevels: mips,
        ArraySize: 1,
        Format: format,
        SampleDesc: sample_desc(),
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: bind_flags,
        CPUAccessFlags: 0,
        MiscFlags: misc,
    };
    let mut tex = None;
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut tex)) }.map_err(|e| {
        GfxError::Other(format!(
            "CreateTexture2D {w}x{h} format={format:?} bind={bind_flags:#x}: {e}"
        ))
    })?;
    Ok(tex.unwrap())
}

fn rt_srv(
    device: &ID3D11Device,
    tex: &ID3D11Texture2D,
) -> Result<(ID3D11ShaderResourceView, ID3D11RenderTargetView), GfxError> {
    let mut srv = None;
    let mut rtv = None;
    unsafe {
        device
            .CreateShaderResourceView(tex, None, Some(&mut srv))
            .map_err(|e| GfxError::Other(format!("CreateShaderResourceView: {e}")))?;
        device
            .CreateRenderTargetView(tex, None, Some(&mut rtv))
            .map_err(|e| GfxError::Other(format!("CreateRenderTargetView: {e}")))?;
    }
    Ok((srv.unwrap(), rtv.unwrap()))
}

/// Output content rectangle after optional aspect preservation. Returning
/// integers here keeps every render target allocation exact and lets the
/// renderer decide whether a two-pass separable scaler is worthwhile.
fn content_dimensions(
    source_w: u32,
    source_h: u32,
    out_w: u32,
    out_h: u32,
    preserve_aspect: bool,
) -> (u32, u32) {
    if !preserve_aspect {
        return (out_w.max(1), out_h.max(1));
    }
    let src_aspect = source_w.max(1) as f32 / source_h.max(1) as f32;
    let out_aspect = out_w.max(1) as f32 / out_h.max(1) as f32;
    let (w, h) = if out_aspect > src_aspect {
        (out_h as f32 * src_aspect, out_h as f32)
    } else {
        (out_w as f32, out_w as f32 / src_aspect)
    };
    (
        (w.round() as u32).clamp(1, out_w.max(1)),
        (h.round() as u32).clamp(1, out_h.max(1)),
    )
}

/// Ring slots carry a full mip pyramid so the hierarchical motion estimator
/// can search coarse levels cheaply. Mips are generated lazily per slot.
fn ring_texture(
    device: &ID3D11Device,
    w: u32,
    h: u32,
    format: DXGI_FORMAT,
) -> Result<(ID3D11Texture2D, ID3D11ShaderResourceView), GfxError> {
    let tex = create_2d(
        device,
        w,
        h,
        format,
        (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0) as u32,
        0,
        D3D11_RESOURCE_MISC_GENERATE_MIPS.0 as u32,
    )?;
    let mut srv = None;
    unsafe { device.CreateShaderResourceView(&tex, None, Some(&mut srv)) }
        .map_err(|e| GfxError::Other(format!("CreateShaderResourceView for ring: {e}")))?;
    Ok((tex, srv.unwrap()))
}

impl Renderer {
    pub fn new(
        device: &ID3D11Device,
        ctx: &ID3D11DeviceContext,
        ring_len: usize,
    ) -> Result<Self, GfxError> {
        let ring_len = ring_len.clamp(2, 8);
        let bins = shaders::compile_all()?;
        unsafe {
            let mut vs = None;
            let mut ps_blend = None;
            let mut ps_upscale = None;
            let mut ps_rcas = None;
            let mut ps_hud = None;
            let mut cs_flow = None;
            let mut cs_flow_smooth = None;
            device.CreateVertexShader(&bins.vs, None, Some(&mut vs))?;
            device.CreatePixelShader(&bins.ps_blend, None, Some(&mut ps_blend))?;
            device.CreatePixelShader(&bins.ps_upscale, None, Some(&mut ps_upscale))?;
            device.CreatePixelShader(&bins.ps_rcas, None, Some(&mut ps_rcas))?;
            device.CreatePixelShader(&bins.ps_hud, None, Some(&mut ps_hud))?;
            device.CreateComputeShader(&bins.cs_flow, None, Some(&mut cs_flow))?;
            device.CreateComputeShader(&bins.cs_flow_smooth, None, Some(&mut cs_flow_smooth))?;

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

            let make_cbuf = |size: u32| -> windows::core::Result<ID3D11Buffer> {
                let mut cb = None;
                device.CreateBuffer(
                    &D3D11_BUFFER_DESC {
                        ByteWidth: size,
                        Usage: D3D11_USAGE_DEFAULT,
                        BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                        CPUAccessFlags: 0,
                        MiscFlags: 0,
                        StructureByteStride: 0,
                    },
                    None,
                    Some(&mut cb),
                )?;
                Ok(cb.unwrap())
            };
            let cbuf = make_cbuf(std::mem::size_of::<PerFrame>() as u32)?;
            let flow_cbuf = make_cbuf(std::mem::size_of::<FlowParams>() as u32)?;
            let hud_cbuf = make_cbuf(std::mem::size_of::<HudParams>() as u32)?;

            let opaque = D3D11_RENDER_TARGET_BLEND_DESC {
                BlendEnable: false.into(),
                SrcBlend: D3D11_BLEND_ONE,
                DestBlend: D3D11_BLEND_ZERO,
                BlendOp: D3D11_BLEND_OP_ADD,
                SrcBlendAlpha: D3D11_BLEND_ONE,
                DestBlendAlpha: D3D11_BLEND_ZERO,
                BlendOpAlpha: D3D11_BLEND_OP_ADD,
                RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8,
            };
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
            let blend_desc_opaque = D3D11_BLEND_DESC {
                AlphaToCoverageEnable: false.into(),
                IndependentBlendEnable: false.into(),
                RenderTarget: [opaque; 8],
            };
            let blend_desc_premult = D3D11_BLEND_DESC {
                AlphaToCoverageEnable: false.into(),
                IndependentBlendEnable: false.into(),
                RenderTarget: [premult; 8],
            };
            let mut blend_opaque = None;
            device.CreateBlendState(&blend_desc_opaque, Some(&mut blend_opaque))?;
            let mut blend_premult = None;
            device.CreateBlendState(&blend_desc_premult, Some(&mut blend_premult))?;

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

            let scissor_desc = D3D11_RASTERIZER_DESC {
                ScissorEnable: true.into(),
                ..raster_desc
            };
            let mut rasterizer_scissor = None;
            device.CreateRasterizerState(&scissor_desc, Some(&mut rasterizer_scissor))?;

            let mk_query = |q: D3D11_QUERY| -> windows::core::Result<ID3D11Query> {
                let mut query = None;
                device.CreateQuery(
                    &D3D11_QUERY_DESC {
                        Query: q,
                        MiscFlags: 0,
                    },
                    Some(&mut query),
                )?;
                Ok(query.unwrap())
            };
            let queries = [
                QueryPair {
                    disjoint: mk_query(D3D11_QUERY_TIMESTAMP_DISJOINT)?,
                    start: mk_query(D3D11_QUERY_TIMESTAMP)?,
                    end: mk_query(D3D11_QUERY_TIMESTAMP)?,
                    valid: false,
                },
                QueryPair {
                    disjoint: mk_query(D3D11_QUERY_TIMESTAMP_DISJOINT)?,
                    start: mk_query(D3D11_QUERY_TIMESTAMP)?,
                    end: mk_query(D3D11_QUERY_TIMESTAMP)?,
                    valid: false,
                },
            ];

            let hud = HudTexture::new(device, ctx)?;

            Ok(Self {
                device: device.clone(),
                ctx: ctx.clone(),
                vs: vs.unwrap(),
                ps_blend: ps_blend.unwrap(),
                ps_upscale: ps_upscale.unwrap(),
                ps_rcas: ps_rcas.unwrap(),
                ps_hud: ps_hud.unwrap(),
                cs_flow: cs_flow.unwrap(),
                cs_flow_smooth: cs_flow_smooth.unwrap(),
                sampler: sampler.unwrap(),
                cbuf,
                flow_cbuf,
                hud_cbuf,
                blend_opaque: blend_opaque.unwrap(),
                blend_premult: blend_premult.unwrap(),
                rasterizer: rasterizer.unwrap(),
                rasterizer_scissor: rasterizer_scissor.unwrap(),
                hud,
                hud_last_update: None,
                hud_last_style: None,
                ring: Vec::new(),
                ring_w: 0,
                ring_h: 0,
                ring_format: None,
                ring_len,
                head: 0,
                seq: 0,
                output: None,
                blend: None,
                scale: None,
                flow: None,
                flow_smooth: None,
                flow_key: None,
                flow_lod: 0.0,
                queries,
                q_slot: 0,
                gpu_us: 0.0,
            })
        }
    }

    /// Allocate the ring at the captured monitor resolution and pixel format.
    pub fn ensure_ring(&mut self, w: u32, h: u32, format: DXGI_FORMAT) -> Result<(), GfxError> {
        if !self.ring.is_empty()
            && self.ring_w == w
            && self.ring_h == h
            && self.ring_format == Some(format)
        {
            return Ok(());
        }
        let len = self.ring_len;
        let mut ring = Vec::with_capacity(len);
        for _ in 0..len {
            let (tex, srv) = ring_texture(&self.device, w, h, format)?;
            ring.push(FrameSlot {
                tex,
                srv,
                qpc: 0,
                seq: 0,
                mips_valid: false,
            });
        }
        self.ring = ring;
        self.ring_w = w;
        self.ring_h = h;
        self.ring_format = Some(format);
        self.head = 0;
        self.seq = 0;
        self.flow_key = None;
        tracing::info!("frame ring allocated: {len} x {w}x{h}, format={format:?} (mip chain)");
        Ok(())
    }

    pub fn clear_frames(&mut self) {
        for slot in &mut self.ring {
            slot.qpc = 0;
            slot.seq = 0;
            slot.mips_valid = false;
        }
        self.head = 0;
        self.seq = 0;
        self.flow_key = None;
    }

    pub fn ring_len(&self) -> usize {
        self.ring.len()
    }

    /// Allocate a BGRA8 ring for the Windows Graphics Capture backend.
    pub fn ensure_bgra8_ring(&mut self, w: u32, h: u32) -> Result<(), GfxError> {
        self.ensure_ring(w, h, DXGI_FORMAT_B8G8R8A8_UNORM)
    }

    /// Copy a newly acquired DDA/WGC texture into the ring (GPU-side only).
    /// Only mip 0 is copied; the rest of the pyramid is generated lazily when
    /// the motion estimator first needs this slot.
    pub fn push_frame(&mut self, src: &ID3D11Texture2D, qpc: i64) {
        if self.ring.is_empty() {
            return;
        }
        let head = self.head;
        {
            let slot = &mut self.ring[head];
            unsafe {
                self.ctx
                    .CopySubresourceRegion(&slot.tex, 0, 0, 0, 0, src, 0, None);
            }
            slot.qpc = qpc;
            slot.mips_valid = false;
        }
        self.seq += 1;
        self.ring[head].seq = self.seq;
        self.head = (head + 1) % self.ring.len();
    }

    fn snapshot(&self, idx: usize) -> FrameSnapshot {
        let slot = &self.ring[idx];
        FrameSnapshot {
            srv: slot.srv.clone(),
            tex: slot.tex.clone(),
            slot: idx,
            qpc: slot.qpc,
            seq: slot.seq,
        }
    }

    pub fn latest_snapshot(&self) -> Option<FrameSnapshot> {
        if self.ring.is_empty() {
            return None;
        }
        let idx = (self.head + self.ring.len() - 1) % self.ring.len();
        (self.ring[idx].seq > 0).then(|| self.snapshot(idx))
    }

    /// Select the captured pair that brackets `target_qpc` and return the
    /// interpolation phase. The pipeline targets a scheduled presentation
    /// instant minus an attainable-multiplier lookback, so two endpoint frames
    /// are available for every interior phase and the N-1 generated slots of an
    /// exact N-times schedule are not skipped.
    pub fn pair_at_time(&self, target_qpc: i64) -> Option<(FrameSnapshot, FrameSnapshot, f32)> {
        if self.ring.is_empty() {
            return None;
        }
        // Stack buffer instead of a heap Vec: this runs once per presented
        // frame and the ring is at most 8 slots.
        let mut entries: [(u64, i64, usize); 8] = [(0, 0, 0); 8];
        let mut n = 0usize;
        for (i, f) in self.ring.iter().enumerate() {
            if f.seq > 0 {
                entries[n] = (f.seq, f.qpc, i);
                n += 1;
                if n == 8 {
                    break;
                }
            }
        }
        if n == 0 {
            return None;
        }
        for i in 1..n {
            let key = entries[i];
            let mut j = i;
            while j > 0 && entries[j - 1].0 > key.0 {
                entries[j] = entries[j - 1];
                j -= 1;
            }
            entries[j] = key;
        }
        if n == 1 {
            let s = self.snapshot(entries[0].2);
            return Some((s.clone(), s, 1.0));
        }

        for pair in entries[..n].windows(2) {
            let (_, a_qpc, a_idx) = pair[0];
            let (_, b_qpc, b_idx) = pair[1];
            if b_qpc > a_qpc {
                if target_qpc <= a_qpc {
                    return Some((self.snapshot(a_idx), self.snapshot(b_idx), 0.0));
                }
                if target_qpc <= b_qpc {
                    let phase = ((target_qpc - a_qpc) as f64 / (b_qpc - a_qpc) as f64) as f32;
                    return Some((
                        self.snapshot(a_idx),
                        self.snapshot(b_idx),
                        phase.clamp(0.0, 1.0),
                    ));
                }
            }
        }

        let a_idx = entries[n - 2].2;
        let b_idx = entries[n - 1].2;
        Some((self.snapshot(a_idx), self.snapshot(b_idx), 1.0))
    }

    pub fn gpu_us(&self) -> f32 {
        self.gpu_us
    }

    fn hud_draw_rects(
        &self,
        bb_w: f32,
        bb_h: f32,
        hud: &HudOptions,
    ) -> Option<([f32; 4], [f32; 4], RECT)> {
        let (content_w, content_h) = self.hud.content_size();
        if content_w == 0 || content_h == 0 {
            return None;
        }
        let (texture_w, texture_h) = self.hud.size();
        let bb_w = bb_w.max(1.0);
        let bb_h = bb_h.max(1.0);
        let margin = hud.margin.max(0.0);
        // If the target window is smaller than the full HUD panel, scale the
        // panel down so corner anchors remain geometrically correct instead of
        // collapsing to the left/top edge.
        let available_w = (bb_w - margin * 2.0).max(1.0);
        let available_h = (bb_h - margin * 2.0).max(1.0);
        let raw_width = content_w as f32;
        let raw_height = content_h as f32;
        let fit = (available_w / raw_width)
            .min(available_h / raw_height)
            .min(1.0);
        let width = raw_width * fit;
        let height = raw_height * fit;
        let max_x = (bb_w - width).max(0.0);
        let max_y = (bb_h - height).max(0.0);
        let (x, y) = match hud.anchor {
            1 => ((max_x - margin).max(0.0), margin.min(max_y)),
            2 => (margin.min(max_x), (max_y - margin).max(0.0)),
            3 => ((max_x - margin).max(0.0), (max_y - margin).max(0.0)),
            4 => (
                hud.custom_x.clamp(0.0, 1.0) * max_x,
                hud.custom_y.clamp(0.0, 1.0) * max_y,
            ),
            _ => (margin.min(max_x), margin.min(max_y)),
        };
        let left = x.floor().max(0.0);
        let top = y.floor().max(0.0);
        let right = (x + width).ceil().min(bb_w);
        let bottom = (y + height).ceil().min(bb_h);
        if right <= left || bottom <= top {
            return None;
        }
        let tex_rect = [
            0.0,
            0.0,
            content_w as f32 / texture_w.max(1) as f32,
            content_h as f32 / texture_h.max(1) as f32,
        ];
        let dest_rect = [x / bb_w, y / bb_h, (x + width) / bb_w, (y + height) / bb_h];
        Some((
            tex_rect,
            dest_rect,
            RECT {
                left: left as i32,
                top: top as i32,
                right: right as i32,
                bottom: bottom as i32,
            },
        ))
    }

    fn ensure_output(&mut self, w: u32, h: u32) -> Result<(), GfxError> {
        if let Some(out) = &self.output {
            if out.w == w && out.h == h {
                return Ok(());
            }
        }
        let tex = create_2d(
            &self.device,
            w,
            h,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
            1,
            0,
        )?;
        let (srv, rtv) = rt_srv(&self.device, &tex)?;
        self.output = Some(OutputTarget {
            _tex: tex,
            srv,
            rtv,
            w,
            h,
        });
        Ok(())
    }

    /// Source-resolution intermediate that receives the temporal blend.
    fn ensure_blend(&mut self, w: u32, h: u32, format: DXGI_FORMAT) -> Result<(), GfxError> {
        if let Some(blend) = &self.blend {
            if blend.w == w && blend.h == h && blend.format == format {
                return Ok(());
            }
        }
        let tex = create_2d(
            &self.device,
            w,
            h,
            format,
            (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
            1,
            0,
        )?;
        let (srv, rtv) = rt_srv(&self.device, &tex)?;
        self.blend = Some(BlendTarget {
            _tex: tex,
            srv,
            rtv,
            w,
            h,
            format,
        });
        tracing::debug!("blend target allocated: {w}x{h} format={format:?}");
        Ok(())
    }

    /// Allocate the horizontal half of the separable adaptive upscaler.
    fn ensure_scale(&mut self, w: u32, h: u32, format: DXGI_FORMAT) -> Result<(), GfxError> {
        if let Some(scale) = &self.scale {
            if scale.w == w && scale.h == h && scale.format == format {
                return Ok(());
            }
        }
        let tex = create_2d(
            &self.device,
            w,
            h,
            format,
            (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
            1,
            0,
        )?;
        let (srv, rtv) = rt_srv(&self.device, &tex)?;
        self.scale = Some(BlendTarget {
            _tex: tex,
            srv,
            rtv,
            w,
            h,
            format,
        });
        tracing::debug!("separable scale target allocated: {w}x{h} format={format:?}");
        Ok(())
    }

    fn ensure_flow(&mut self, w: u32, h: u32) -> Result<(), GfxError> {
        if let Some(flow) = &self.flow {
            if flow.w == w && flow.h == h {
                return Ok(());
            }
        }
        // Raw field: full mip pyramid so the blend pass can read a whole-frame
        // statistic for the scene-cut gate.
        let raw = create_2d(
            &self.device,
            w,
            h,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            (D3D11_BIND_UNORDERED_ACCESS.0
                | D3D11_BIND_SHADER_RESOURCE.0
                | D3D11_BIND_RENDER_TARGET.0) as u32,
            0,
            D3D11_RESOURCE_MISC_GENERATE_MIPS.0 as u32,
        )?;
        let mut raw_srv = None;
        let mut raw_uav = None;
        unsafe {
            self.device
                .CreateShaderResourceView(&raw, None, Some(&mut raw_srv))
                .map_err(|e| GfxError::Other(format!("CreateShaderResourceView for flow: {e}")))?;
            self.device
                .CreateUnorderedAccessView(&raw, None, Some(&mut raw_uav))
                .map_err(|e| GfxError::Other(format!("CreateUnorderedAccessView for flow: {e}")))?;
        }
        self.flow = Some(FlowTarget {
            _tex: raw,
            srv: raw_srv.unwrap(),
            uav: raw_uav.unwrap(),
            w,
            h,
        });

        let smooth = create_2d(
            &self.device,
            w,
            h,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            (D3D11_BIND_UNORDERED_ACCESS.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
            1,
            0,
        )?;
        let mut sm_srv = None;
        let mut sm_uav = None;
        unsafe {
            self.device
                .CreateShaderResourceView(&smooth, None, Some(&mut sm_srv))
                .map_err(|e| {
                    GfxError::Other(format!("CreateShaderResourceView for flow smooth: {e}"))
                })?;
            self.device
                .CreateUnorderedAccessView(&smooth, None, Some(&mut sm_uav))
                .map_err(|e| {
                    GfxError::Other(format!("CreateUnorderedAccessView for flow smooth: {e}"))
                })?;
        }
        self.flow_smooth = Some(FlowTarget {
            _tex: smooth,
            srv: sm_srv.unwrap(),
            uav: sm_uav.unwrap(),
            w,
            h,
        });
        self.flow_lod = ((w.max(h) as f32).log2() - 1.0).max(0.0);
        Ok(())
    }

    fn ensure_mips(&mut self, slot: usize) {
        let make_mips = {
            let s = &self.ring[slot];
            s.seq > 0 && !s.mips_valid
        };
        if make_mips {
            let srv = self.ring[slot].srv.clone();
            unsafe { self.ctx.GenerateMips(&srv) };
            self.ring[slot].mips_valid = true;
        }
    }

    fn compute_flow(
        &mut self,
        prev: &FrameSnapshot,
        curr: &FrameSnapshot,
        src_rect: [f32; 4],
        source_w: u32,
        source_h: u32,
    ) -> Result<(), GfxError> {
        let Some(flow) = self.flow.as_ref() else {
            return Ok(());
        };
        let (flow_w, flow_h) = (flow.w, flow.h);
        let raw_srv = flow.srv.clone();
        let raw_uav = flow.uav.clone();
        let Some(smooth) = self.flow_smooth.as_ref() else {
            return Ok(());
        };
        let (smooth_w, smooth_h) = (smooth.w, smooth.h);
        let smooth_uav = smooth.uav.clone();
        self.ensure_mips(prev.slot);
        self.ensure_mips(curr.slot);

        // Coarse step covers +/-2*step source pixels at LOD 3; one LOD-3 texel
        // spans 8 source pixels, so 40 px step ~= 160 px/frame coverage range
        // before the refinement ladder narrows it back to sub-pixel accuracy.
        let max_dim = source_w.max(source_h) as f32;
        let coarse_step = (max_dim / 48.0).clamp(8.0, 40.0);
        let params = FlowParams {
            src_rect,
            source_size: [
                source_w.max(1) as f32,
                source_h.max(1) as f32,
                1.0 / source_w.max(1) as f32,
                1.0 / source_h.max(1) as f32,
            ],
            flow_size: [flow_w as f32, flow_h as f32, 0.0, 0.0],
            search_cfg: [coarse_step, 0.0008, 0.0, 0.0],
        };
        unsafe {
            self.ctx.UpdateSubresource(
                &self.flow_cbuf,
                0,
                None,
                &params as *const FlowParams as *const core::ffi::c_void,
                0,
                0,
            );
            self.ctx
                .CSSetConstantBuffers(0, Some(&[Some(self.flow_cbuf.clone())]));
            self.ctx
                .CSSetSamplers(0, Some(&[Some(self.sampler.clone())]));

            // Pass 1: hierarchical block matching into the raw field.
            self.ctx.CSSetShader(&self.cs_flow, None);
            self.ctx
                .CSSetShaderResources(0, Some(&[Some(prev.srv.clone()), Some(curr.srv.clone())]));
            let raw_uav = [Some(raw_uav)];
            self.ctx
                .CSSetUnorderedAccessViews(0, 1, Some(raw_uav.as_ptr()), None);
            self.ctx.Dispatch(flow_w.div_ceil(8), flow_h.div_ceil(8), 1);
            let null_uav: Option<ID3D11UnorderedAccessView> = None;
            self.ctx
                .CSSetUnorderedAccessViews(0, 1, Some(&null_uav), None);

            // Build the raw-field pyramid for the scene-cut gate in ps_blend.
            self.ctx.GenerateMips(&raw_srv);

            // Pass 2: bilateral smoothing into the field the blend pass reads.
            self.ctx.CSSetShader(&self.cs_flow_smooth, None);
            let null_srvs: [Option<ID3D11ShaderResourceView>; 2] = [None, None];
            self.ctx.CSSetShaderResources(0, Some(&null_srvs));
            self.ctx.CSSetShaderResources(0, Some(&[Some(raw_srv)]));
            let sm_uav = [Some(smooth_uav)];
            self.ctx
                .CSSetUnorderedAccessViews(0, 1, Some(sm_uav.as_ptr()), None);
            self.ctx
                .Dispatch(smooth_w.div_ceil(8), smooth_h.div_ceil(8), 1);

            // Unbind compute resources before the pixel passes use the fields.
            self.ctx
                .CSSetUnorderedAccessViews(0, 1, Some(&null_uav), None);
            self.ctx.CSSetShaderResources(0, Some(&null_srvs));
            self.ctx.CSSetShader(None, None);
        }
        Ok(())
    }

    /// Render one display frame. `previous` and `current` are snapshots from
    /// the capture ring; `options.fg_phase` selects the temporal position.
    pub fn render(&mut self, request: RenderRequest<'_>) -> Result<(), GfxError> {
        let RenderRequest {
            target,
            previous,
            current,
            options,
            hud_text,
            hud,
        } = request;
        let rtv = target.rtv;
        let bb_w = target.width;
        let bb_h = target.height;
        let uv_rect = target.uv_rect;

        // Release SRVs from the previous pass before a render target may be
        // reallocated. D3D11 requires a resource not to be bound as RTV and SRV
        // at the same time.
        let null_srvs: [Option<ID3D11ShaderResourceView>; 4] = [None, None, None, None];
        unsafe {
            self.ctx.PSSetShaderResources(0, Some(&null_srvs));
        }

        let out_w = bb_w.max(1.0).round() as u32;
        let out_h = bb_h.max(1.0).round() as u32;
        let uv_w = (uv_rect[2] - uv_rect[0]).max(1.0e-6);
        let uv_h = (uv_rect[3] - uv_rect[1]).max(1.0e-6);
        let source_w = (uv_w * self.ring_w.max(1) as f32).max(1.0).round() as u32;
        let source_h = (uv_h * self.ring_h.max(1) as f32).max(1.0).round() as u32;
        let fg_active = options.fg_enabled
            && options.fg_phase > 0.001
            && options.fg_phase < 0.999
            && previous.seq != current.seq;
        let use_motion = fg_active && options.motion_compensation;
        let sharpen = options.sharpness.clamp(0.0, 1.0) > 0.0001;

        let uv_full =
            uv_rect[0] == 0.0 && uv_rect[1] == 0.0 && uv_rect[2] == 1.0 && uv_rect[3] == 1.0;
        let ring_format = self.ring_format.unwrap_or(DXGI_FORMAT_B8G8R8A8_UNORM);
        let hud_rect = [0.0, 0.0, 1.0, 1.0];
        let cb = PerFrame {
            uv_rect,
            hud_rect,
            output_size: [
                out_w as f32,
                out_h as f32,
                1.0 / out_w as f32,
                1.0 / out_h as f32,
            ],
            source_size: [
                source_w as f32,
                source_h as f32,
                1.0 / source_w.max(1) as f32,
                1.0 / source_h.max(1) as f32,
            ],
            params0: [
                options.fg_phase.clamp(0.0, 1.0),
                options.scale_mode as f32,
                options.sharpness.clamp(0.0, 1.0),
                if fg_active { 1.0 } else { 0.0 },
            ],
            params1: [
                if use_motion { 1.0 } else { 0.0 },
                if ring_format == DXGI_FORMAT_R16G16B16A16_FLOAT {
                    1.0
                } else {
                    0.0
                },
                if options.preserve_aspect { 1.0 } else { 0.0 },
                options.fg_strength.clamp(0.0, 1.0),
            ],
            params2: [
                0.45 - options.scene_protection.clamp(0.0, 1.0) * 0.35,
                self.flow_lod,
                0.0,
                0.0,
            ],
        };

        // Native-resolution fast paths. They are by far the most common case
        // with MatchWindow output: no temporal blend, no intermediate render
        // target and no spatial reconstruction. Without frame generation the
        // capture already has exactly the pixels the swapchain needs.
        let native_direct = !fg_active
            && hud_text.is_none()
            && uv_full
            && out_w == source_w
            && out_h == source_h
            && ring_format == DXGI_FORMAT_B8G8R8A8_UNORM;
        if native_direct {
            if !sharpen {
                let resource = unsafe { rtv.GetResource() }.ok();
                if let Some(dst) = resource {
                    unsafe {
                        self.ctx
                            .CopySubresourceRegion(&dst, 0, 0, 0, 0, &current.tex, 0, None);
                    }
                    return Ok(());
                }
            } else {
                // Sharpening does not need a full-resolution copy through the
                // adaptive scaler. RCAS can read the captured texture directly,
                // removing two full-frame passes from the native path.
                let qp_slot = self.q_slot;
                unsafe {
                    self.ctx.Begin(&self.queries[qp_slot].disjoint);
                    self.ctx.End(&self.queries[qp_slot].start);
                    self.ctx.UpdateSubresource(
                        &self.cbuf,
                        0,
                        None,
                        &cb as *const PerFrame as *const core::ffi::c_void,
                        0,
                        0,
                    );
                    self.ctx
                        .IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    self.ctx.VSSetShader(&self.vs, None);
                    self.ctx.RSSetState(Some(&self.rasterizer));
                    self.ctx
                        .OMSetBlendState(Some(&self.blend_opaque), None, 0xffff_ffff);
                    self.ctx
                        .OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);
                    self.ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                        TopLeftX: 0.0,
                        TopLeftY: 0.0,
                        Width: bb_w,
                        Height: bb_h,
                        MinDepth: 0.0,
                        MaxDepth: 1.0,
                    }]));
                    self.ctx
                        .PSSetConstantBuffers(0, Some(&[Some(self.cbuf.clone())]));
                    self.ctx
                        .PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
                    self.ctx.PSSetShader(&self.ps_rcas, None);
                    self.ctx
                        .PSSetShaderResources(0, Some(&[Some(current.srv.clone())]));
                    self.ctx.Draw(3, 0);
                    self.ctx.End(&self.queries[qp_slot].end);
                    let null_srvs: [Option<ID3D11ShaderResourceView>; 1] = [None];
                    self.ctx.PSSetShaderResources(0, Some(&null_srvs));
                }
                self.queries[qp_slot].valid = true;
                self.resolve((qp_slot + 1) % 2);
                self.q_slot = (qp_slot + 1) % 2;
                return Ok(());
            }
        }

        let (content_w, content_h) =
            content_dimensions(source_w, source_h, out_w, out_h, options.preserve_aspect);
        // The separable kernel is only a win for true magnification. Mixed
        // scale directions and minification stay on the single-pass area path.
        let use_separable = options.scale_mode == 2 && content_w > source_w && content_h > source_h;
        // Frame generation at native output resolution does not need any
        // spatial reconstruction. The blend result can go straight to the
        // swapchain (or to the RCAS intermediate when sharpening), saving a
        // full-resolution blend target and the upscale draw every frame.
        let direct_blend_native = fg_active
            && uv_full
            && out_w == source_w
            && out_h == source_h
            && ring_format == DXGI_FORMAT_B8G8R8A8_UNORM;
        if sharpen {
            self.ensure_output(out_w, out_h)?;
        }
        if use_separable {
            self.ensure_scale(content_w, source_h, ring_format)?;
        }
        if use_motion {
            let flow_w = source_w.div_ceil(8).max(1);
            let flow_h = source_h.div_ceil(8).max(1);
            let flow_key = (previous.seq, current.seq);
            let needs_flow = self.flow_key != Some(flow_key)
                || self
                    .flow
                    .as_ref()
                    .is_none_or(|flow| flow.w != flow_w || flow.h != flow_h);
            if needs_flow {
                self.ensure_flow(flow_w, flow_h)?;
                self.compute_flow(previous, current, uv_rect, source_w, source_h)?;
                self.flow_key = Some(flow_key);
            }
        }
        if !direct_blend_native {
            self.ensure_blend(source_w, source_h, ring_format)?;
        }

        let output_rtv = if sharpen {
            Some(
                self.output
                    .as_ref()
                    .ok_or_else(|| GfxError::Other("output target missing".into()))?
                    .rtv
                    .clone(),
            )
        } else {
            None
        };
        let output_srv = if sharpen {
            Some(
                self.output
                    .as_ref()
                    .ok_or_else(|| GfxError::Other("output target missing".into()))?
                    .srv
                    .clone(),
            )
        } else {
            None
        };
        let blend = if direct_blend_native {
            None
        } else {
            Some(
                self.blend
                    .as_ref()
                    .ok_or_else(|| GfxError::Other("blend target missing".into()))?,
            )
        };
        let scale_srv = if use_separable {
            Some(
                self.scale
                    .as_ref()
                    .ok_or_else(|| GfxError::Other("scale target missing".into()))?
                    .srv
                    .clone(),
            )
        } else {
            None
        };
        let scale_rtv = if use_separable {
            Some(
                self.scale
                    .as_ref()
                    .ok_or_else(|| GfxError::Other("scale target missing".into()))?
                    .rtv
                    .clone(),
            )
        } else {
            None
        };
        let flow_srv = self
            .flow_smooth
            .as_ref()
            .map(|f| f.srv.clone())
            .unwrap_or_else(|| current.srv.clone());
        let flow_stats_srv = self
            .flow
            .as_ref()
            .map(|f| f.srv.clone())
            .unwrap_or_else(|| current.srv.clone());

        let mut cb_horizontal = cb;
        cb_horizontal.output_size = [
            content_w as f32,
            source_h as f32,
            1.0 / content_w.max(1) as f32,
            1.0 / source_h.max(1) as f32,
        ];
        cb_horizontal.source_size[0] = source_w as f32;
        cb_horizontal.source_size[1] = source_h as f32;
        cb_horizontal.source_size[2] = 1.0 / source_w.max(1) as f32;
        cb_horizontal.source_size[3] = 1.0 / source_h.max(1) as f32;
        cb_horizontal.params2[2] = 1.0;

        let mut cb_vertical = cb;
        cb_vertical.source_size = [
            content_w as f32,
            source_h as f32,
            1.0 / content_w.max(1) as f32,
            1.0 / source_h.max(1) as f32,
        ];
        cb_vertical.params2[2] = 2.0;
        // The intermediate is stretched horizontally but has the original
        // source height, so its own aspect ratio no longer describes the
        // letterbox geometry. Carry the original source aspect explicitly.
        cb_vertical.params2[3] = source_w.max(1) as f32 / source_h.max(1) as f32;

        let qp_slot = self.q_slot;
        unsafe {
            self.ctx.Begin(&self.queries[qp_slot].disjoint);
            self.ctx.End(&self.queries[qp_slot].start);

            self.ctx.UpdateSubresource(
                &self.cbuf,
                0,
                None,
                &cb as *const PerFrame as *const core::ffi::c_void,
                0,
                0,
            );
            self.ctx
                .IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            self.ctx.VSSetShader(&self.vs, None);
            self.ctx.RSSetState(Some(&self.rasterizer));
            self.ctx
                .OMSetBlendState(Some(&self.blend_opaque), None, 0xffff_ffff);
            self.ctx
                .PSSetConstantBuffers(0, Some(&[Some(self.cbuf.clone())]));
            self.ctx
                .PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));

            // Pass 1: temporal blend at source resolution. At native output
            // this writes directly into the final path; otherwise it writes
            // the source-resolution intermediate used by the scaler.
            let blend_pass_rtv = if direct_blend_native {
                if sharpen {
                    output_rtv
                        .clone()
                        .ok_or_else(|| GfxError::Other("output target missing".into()))?
                } else {
                    rtv.clone()
                }
            } else {
                blend
                    .as_ref()
                    .ok_or_else(|| GfxError::Other("blend target missing".into()))?
                    .rtv
                    .clone()
            };
            self.ctx
                .OMSetRenderTargets(Some(&[Some(blend_pass_rtv)]), None);
            self.ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: source_w as f32,
                Height: source_h as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            self.ctx.PSSetShader(&self.ps_blend, None);
            self.ctx.PSSetShaderResources(
                0,
                Some(&[
                    Some(previous.srv.clone()),
                    Some(current.srv.clone()),
                    Some(flow_srv),
                    Some(flow_stats_srv),
                ]),
            );
            self.ctx.Draw(3, 0);

            // Pass 2: spatial reconstruction. The source is already in
            // `blend`, never bind its RTV and SRV at the same time.
            self.ctx.PSSetShader(&self.ps_upscale, None);
            if direct_blend_native {
                // Already at the final resolution. Skip the reconstruction
                // pass entirely; RCAS below (if enabled) reads the blend
                // intermediate directly.
            } else if use_separable {
                // Horizontal pass -> compact intermediate at source height.
                self.ctx.UpdateSubresource(
                    &self.cbuf,
                    0,
                    None,
                    &cb_horizontal as *const PerFrame as *const core::ffi::c_void,
                    0,
                    0,
                );
                self.ctx.OMSetRenderTargets(
                    Some(&[Some(scale_rtv.clone().ok_or_else(|| {
                        GfxError::Other("scale target missing".into())
                    })?)]),
                    None,
                );
                self.ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: content_w as f32,
                    Height: source_h as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]));
                self.ctx.PSSetShaderResources(
                    0,
                    Some(&[Some(
                        blend
                            .as_ref()
                            .ok_or_else(|| GfxError::Other("blend target missing".into()))?
                            .srv
                            .clone(),
                    )]),
                );
                self.ctx.Draw(3, 0);

                // Vertical/final pass. Binding the next target first releases
                // the scale RTV; only then is scale.srv allowed to be read.
                self.ctx.UpdateSubresource(
                    &self.cbuf,
                    0,
                    None,
                    &cb_vertical as *const PerFrame as *const core::ffi::c_void,
                    0,
                    0,
                );
                self.ctx.OMSetRenderTargets(
                    Some(&[Some(if sharpen {
                        output_rtv
                            .clone()
                            .ok_or_else(|| GfxError::Other("output target missing".into()))?
                    } else {
                        rtv.clone()
                    })]),
                    None,
                );
                self.ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: bb_w,
                    Height: bb_h,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]));
                self.ctx.PSSetShaderResources(
                    0,
                    Some(&[Some(scale_srv.clone().ok_or_else(|| {
                        GfxError::Other("scale target missing".into())
                    })?)]),
                );
                self.ctx.Draw(3, 0);
            } else {
                self.ctx.UpdateSubresource(
                    &self.cbuf,
                    0,
                    None,
                    &cb as *const PerFrame as *const core::ffi::c_void,
                    0,
                    0,
                );
                self.ctx.OMSetRenderTargets(
                    Some(&[Some(if sharpen {
                        output_rtv
                            .clone()
                            .ok_or_else(|| GfxError::Other("output target missing".into()))?
                    } else {
                        rtv.clone()
                    })]),
                    None,
                );
                self.ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: bb_w,
                    Height: bb_h,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]));
                self.ctx.PSSetShaderResources(
                    0,
                    Some(&[Some(
                        blend
                            .as_ref()
                            .ok_or_else(|| GfxError::Other("blend target missing".into()))?
                            .srv
                            .clone(),
                    )]),
                );
                self.ctx.Draw(3, 0);
            }

            // Final contrast-adaptive sharpening into the swapchain. The RCAS
            // target must be bound before output.srv as input for the same
            // simultaneous-view reason as above.
            if sharpen {
                self.ctx
                    .OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);
                self.ctx.PSSetShader(&self.ps_rcas, None);
                self.ctx.PSSetShaderResources(
                    0,
                    Some(&[Some(output_srv.clone().ok_or_else(|| {
                        GfxError::Other("output target missing".into())
                    })?)]),
                );
                self.ctx.Draw(3, 0);
            }

            // Pass 4: premultiplied-alpha, scissored real-time HUD.
            if let Some(text) = hud_text {
                let style_key = (hud.font_scale, hud.background_alpha);
                let style_changed = self.hud_last_style != Some(style_key);
                let should_update = style_changed
                    || self
                        .hud_last_update
                        .is_none_or(|last| last.elapsed() >= Duration::from_millis(250));
                if should_update {
                    self.hud
                        .update(&self.ctx, text, hud.font_scale, hud.background_alpha);
                    self.hud_last_update = Some(Instant::now());
                    self.hud_last_style = Some(style_key);
                }
                if let Some((tex_rect, dest_rect, scissor)) = self.hud_draw_rects(bb_w, bb_h, &hud)
                {
                    let hud_params = HudParams {
                        tex_rect,
                        dest_rect,
                    };
                    self.ctx.UpdateSubresource(
                        &self.hud_cbuf,
                        0,
                        None,
                        &hud_params as *const HudParams as *const core::ffi::c_void,
                        0,
                        0,
                    );
                    self.ctx.PSSetShader(&self.ps_hud, None);
                    self.ctx
                        .PSSetConstantBuffers(1, Some(&[Some(self.hud_cbuf.clone())]));
                    self.ctx
                        .PSSetShaderResources(0, Some(&[Some(self.hud.srv().clone())]));
                    self.ctx
                        .OMSetBlendState(Some(&self.blend_premult), None, 0xffff_ffff);
                    self.ctx.RSSetState(Some(&self.rasterizer_scissor));
                    self.ctx.RSSetScissorRects(Some(&[scissor]));
                    self.ctx.Draw(3, 0);
                    self.ctx.RSSetState(Some(&self.rasterizer));
                }
            }

            self.ctx.End(&self.queries[qp_slot].end);
            let null_srvs: [Option<ID3D11ShaderResourceView>; 4] = [None, None, None, None];
            self.ctx.PSSetShaderResources(0, Some(&null_srvs));
        }
        self.queries[qp_slot].valid = true;

        self.resolve((qp_slot + 1) % 2);
        self.q_slot = (qp_slot + 1) % 2;
        Ok(())
    }

    fn resolve(&mut self, slot: usize) {
        if !self.queries[slot].valid {
            return;
        }
        let mut disjoint = D3D11_QUERY_DATA_TIMESTAMP_DISJOINT::default();
        unsafe {
            let ok = self
                .ctx
                .GetData(
                    &self.queries[slot].disjoint,
                    Some(&mut disjoint as *mut _ as *mut core::ffi::c_void),
                    std::mem::size_of_val(&disjoint) as u32,
                    0,
                )
                .is_ok();
            if !ok || disjoint.Disjoint.as_bool() {
                return;
            }
            let mut start = 0u64;
            let mut end = 0u64;
            let s_ok = self
                .ctx
                .GetData(
                    &self.queries[slot].start,
                    Some(&mut start as *mut _ as *mut core::ffi::c_void),
                    std::mem::size_of_val(&start) as u32,
                    0,
                )
                .is_ok();
            let e_ok = self
                .ctx
                .GetData(
                    &self.queries[slot].end,
                    Some(&mut end as *mut _ as *mut core::ffi::c_void),
                    std::mem::size_of_val(&end) as u32,
                    0,
                )
                .is_ok();
            if s_ok && e_ok && disjoint.Frequency > 0 && end > start {
                self.gpu_us = ((end - start) as f32 / disjoint.Frequency as f32) * 1e6;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::content_dimensions;

    #[test]
    fn same_aspect_preserves_the_whole_output() {
        assert_eq!(
            content_dimensions(1920, 1080, 2560, 1440, true),
            (2560, 1440)
        );
        assert_eq!(content_dimensions(1920, 1080, 1280, 720, true), (1280, 720));
    }

    #[test]
    fn letterbox_preserves_source_aspect() {
        // 4:3 content in a square output: 128x96 centered vertically.
        assert_eq!(content_dimensions(32, 24, 128, 128, true), (128, 96));
        // Ultrawide content in 16:9: 2560x1080 content plus bars.
        assert_eq!(
            content_dimensions(2560, 1080, 2560, 1440, true),
            (2560, 1080)
        );
        assert_eq!(
            content_dimensions(2560, 1080, 2560, 1440, false),
            (2560, 1440)
        );
    }
}
