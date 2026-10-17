//! GPU render pipeline:
//! captured frame ring -> optional motion field -> frame composition/upscale
//! -> contrast-adaptive sharpening -> HUD overlay -> waitable swapchain.
//!
//! Everything stays on the GPU. The CPU never copies pixels and only writes a
//! few constant-buffer values per presented frame.

use crate::error::GfxError;
use crate::hud::HudTexture;
use crate::shaders;
use std::time::{Duration, Instant};
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
    D3D11_SAMPLER_DESC, D3D11_TEXTURE2D_DESC, D3D11_TEXTURE_ADDRESS_CLAMP, D3D11_USAGE_DEFAULT,
    D3D11_VIEWPORT,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_SAMPLE_DESC,
};

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
}

#[derive(Clone)]
pub struct FrameSnapshot {
    srv: ID3D11ShaderResourceView,
    pub qpc: i64,
    pub seq: u64,
}

pub struct FrameSlot {
    tex: ID3D11Texture2D,
    srv: ID3D11ShaderResourceView,
    pub qpc: i64,
    pub seq: u64,
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

pub struct Renderer {
    device: ID3D11Device,
    ctx: ID3D11DeviceContext,
    vs: ID3D11VertexShader,
    ps_compose: ID3D11PixelShader,
    ps_rcas: ID3D11PixelShader,
    ps_hud: ID3D11PixelShader,
    cs_flow: ID3D11ComputeShader,
    sampler: ID3D11SamplerState,
    cbuf: ID3D11Buffer,
    flow_cbuf: ID3D11Buffer,
    blend_opaque: ID3D11BlendState,
    blend_premult: ID3D11BlendState,
    rasterizer: ID3D11RasterizerState,
    hud: HudTexture,
    hud_last_update: Option<Instant>,
    ring: Vec<FrameSlot>,
    ring_w: u32,
    ring_h: u32,
    ring_format: Option<DXGI_FORMAT>,
    ring_len: usize,
    head: usize,
    seq: u64,
    output: Option<OutputTarget>,
    flow: Option<FlowTarget>,
    flow_key: Option<(u64, u64)>,
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
) -> Result<ID3D11Texture2D, GfxError> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: w.max(1),
        Height: h.max(1),
        MipLevels: 1,
        ArraySize: 1,
        Format: format,
        SampleDesc: sample_desc(),
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: bind_flags,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let mut tex = None;
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut tex)) }.map_err(|e| {
        GfxError::Other(format!(
            "CreateTexture2D {w}x{h} format={format:?} bind={bind_flags:#x}: {e}"
        ))
    })?;
    Ok(tex.unwrap())
}

fn ring_texture(
    device: &ID3D11Device,
    w: u32,
    h: u32,
    format: DXGI_FORMAT,
) -> Result<(ID3D11Texture2D, ID3D11ShaderResourceView), GfxError> {
    let tex = create_2d(device, w, h, format, D3D11_BIND_SHADER_RESOURCE.0 as u32)?;
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
        let ring_len = ring_len.max(2);
        let bins = shaders::compile_all()?;
        unsafe {
            let mut vs = None;
            let mut ps_compose = None;
            let mut ps_rcas = None;
            let mut ps_hud = None;
            let mut cs_flow = None;
            device.CreateVertexShader(&bins.vs, None, Some(&mut vs))?;
            device.CreatePixelShader(&bins.ps_compose, None, Some(&mut ps_compose))?;
            device.CreatePixelShader(&bins.ps_rcas, None, Some(&mut ps_rcas))?;
            device.CreatePixelShader(&bins.ps_hud, None, Some(&mut ps_hud))?;
            device.CreateComputeShader(&bins.cs_flow, None, Some(&mut cs_flow))?;

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
                ps_compose: ps_compose.unwrap(),
                ps_rcas: ps_rcas.unwrap(),
                ps_hud: ps_hud.unwrap(),
                cs_flow: cs_flow.unwrap(),
                sampler: sampler.unwrap(),
                cbuf,
                flow_cbuf,
                blend_opaque: blend_opaque.unwrap(),
                blend_premult: blend_premult.unwrap(),
                rasterizer: rasterizer.unwrap(),
                hud,
                hud_last_update: None,
                ring: Vec::new(),
                ring_w: 0,
                ring_h: 0,
                ring_format: None,
                ring_len,
                head: 0,
                seq: 0,
                output: None,
                flow: None,
                flow_key: None,
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
            });
        }
        self.ring = ring;
        self.ring_w = w;
        self.ring_h = h;
        self.ring_format = Some(format);
        self.head = 0;
        self.seq = 0;
        self.flow_key = None;
        tracing::info!("frame ring allocated: {len} x {w}x{h}, format={format:?}");
        Ok(())
    }

    pub fn ring_len(&self) -> usize {
        self.ring.len()
    }

    /// Allocate a BGRA8 ring for the Windows Graphics Capture backend.
    pub fn ensure_bgra8_ring(&mut self, w: u32, h: u32) -> Result<(), GfxError> {
        self.ensure_ring(w, h, DXGI_FORMAT_B8G8R8A8_UNORM)
    }

    /// Copy a newly acquired DDA texture into the ring (GPU-side only).
    pub fn push_frame(&mut self, src: &ID3D11Texture2D, qpc: i64) {
        if self.ring.is_empty() {
            return;
        }
        let head = self.head;
        {
            let slot = &mut self.ring[head];
            unsafe { self.ctx.CopyResource(&slot.tex, src) };
            slot.qpc = qpc;
        }
        self.seq += 1;
        self.ring[head].seq = self.seq;
        self.head = (head + 1) % self.ring.len();
    }

    fn snapshot(&self, idx: usize) -> FrameSnapshot {
        let slot = &self.ring[idx];
        FrameSnapshot {
            srv: slot.srv.clone(),
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
    /// interpolation phase. The render loop targets `now - half_capture_interval`
    /// so two endpoint frames are always available before we display the blend.
    pub fn pair_at_time(&self, target_qpc: i64) -> Option<(FrameSnapshot, FrameSnapshot, f32)> {
        if self.ring.is_empty() {
            return None;
        }
        let mut entries: Vec<(u64, i64, usize)> = self
            .ring
            .iter()
            .enumerate()
            .filter(|(_, f)| f.seq > 0)
            .map(|(i, f)| (f.seq, f.qpc, i))
            .collect();
        if entries.is_empty() {
            return None;
        }
        entries.sort_by_key(|(seq, _, _)| *seq);
        if entries.len() == 1 {
            let s = self.snapshot(entries[0].2);
            return Some((s.clone(), s, 1.0));
        }

        for pair in entries.windows(2) {
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

        let a_idx = entries[entries.len() - 2].2;
        let b_idx = entries[entries.len() - 1].2;
        Some((self.snapshot(a_idx), self.snapshot(b_idx), 1.0))
    }

    pub fn gpu_us(&self) -> f32 {
        self.gpu_us
    }

    fn hud_rect(&self, bb_w: f32, bb_h: f32) -> [f32; 4] {
        let (hw, hh) = self.hud.size();
        [
            0.0,
            0.0,
            (hw as f32 / bb_w.max(1.0)).min(1.0),
            (hh as f32 / bb_h.max(1.0)).min(1.0),
        ]
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
        )?;
        let mut srv = None;
        let mut rtv = None;
        unsafe {
            self.device
                .CreateShaderResourceView(&tex, None, Some(&mut srv))
                .map_err(|e| {
                    GfxError::Other(format!("CreateShaderResourceView for output: {e}"))
                })?;
            self.device
                .CreateRenderTargetView(&tex, None, Some(&mut rtv))
                .map_err(|e| GfxError::Other(format!("CreateRenderTargetView for output: {e}")))?;
        }
        self.output = Some(OutputTarget {
            _tex: tex,
            srv: srv.unwrap(),
            rtv: rtv.unwrap(),
            w,
            h,
        });
        Ok(())
    }

    fn ensure_flow(&mut self, w: u32, h: u32) -> Result<(), GfxError> {
        if let Some(flow) = &self.flow {
            if flow.w == w && flow.h == h {
                return Ok(());
            }
        }
        let tex = create_2d(
            &self.device,
            w,
            h,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            (D3D11_BIND_UNORDERED_ACCESS.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
        )?;
        let mut srv = None;
        let mut uav = None;
        unsafe {
            self.device
                .CreateShaderResourceView(&tex, None, Some(&mut srv))
                .map_err(|e| GfxError::Other(format!("CreateShaderResourceView for flow: {e}")))?;
            self.device
                .CreateUnorderedAccessView(&tex, None, Some(&mut uav))
                .map_err(|e| GfxError::Other(format!("CreateUnorderedAccessView for flow: {e}")))?;
        }
        self.flow = Some(FlowTarget {
            _tex: tex,
            srv: srv.unwrap(),
            uav: uav.unwrap(),
            w,
            h,
        });
        Ok(())
    }

    fn compute_flow(
        &self,
        prev: &FrameSnapshot,
        curr: &FrameSnapshot,
        src_rect: [f32; 4],
        source_w: u32,
        source_h: u32,
    ) -> Result<(), GfxError> {
        let Some(flow) = &self.flow else {
            return Ok(());
        };
        let radius = (source_w.min(source_h) as f32 * 0.018).clamp(4.0, 24.0);
        let coarse_step = (radius * 0.5).max(1.0);
        let refine_step = (coarse_step * 0.25).clamp(1.0, 4.0);
        let params = FlowParams {
            src_rect,
            source_size: [
                source_w.max(1) as f32,
                source_h.max(1) as f32,
                1.0 / source_w.max(1) as f32,
                1.0 / source_h.max(1) as f32,
            ],
            flow_size: [flow.w as f32, flow.h as f32, 0.0, 0.0],
            search_cfg: [coarse_step, refine_step, 0.0008, 0.0],
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
            self.ctx.CSSetShader(&self.cs_flow, None);
            self.ctx
                .CSSetConstantBuffers(0, Some(&[Some(self.flow_cbuf.clone())]));
            self.ctx
                .CSSetShaderResources(0, Some(&[Some(prev.srv.clone()), Some(curr.srv.clone())]));
            self.ctx
                .CSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
            let uavs = [Some(flow.uav.clone())];
            self.ctx
                .CSSetUnorderedAccessViews(0, 1, Some(uavs.as_ptr()), None);
            self.ctx.Dispatch(flow.w.div_ceil(8), flow.h.div_ceil(8), 1);

            // Unbind compute resources before the pixel pass uses the flow SRV.
            let null_uav: Option<ID3D11UnorderedAccessView> = None;
            self.ctx
                .CSSetUnorderedAccessViews(0, 1, Some(&null_uav), None);
            let null_srvs: [Option<ID3D11ShaderResourceView>; 2] = [None, None];
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
        } = request;
        let rtv = target.rtv;
        let bb_w = target.width;
        let bb_h = target.height;
        let uv_rect = target.uv_rect;

        // Release SRVs from the previous pass before a render target may be
        // reallocated. D3D11 requires a resource not to be bound as RTV and SRV
        // at the same time.
        let null_srvs: [Option<ID3D11ShaderResourceView>; 3] = [None, None, None];
        unsafe {
            self.ctx.PSSetShaderResources(0, Some(&null_srvs));
        }

        let out_w = bb_w.max(1.0).round() as u32;
        let out_h = bb_h.max(1.0).round() as u32;
        self.ensure_output(out_w, out_h)?;

        let uv_w = (uv_rect[2] - uv_rect[0]).max(1.0e-6);
        let uv_h = (uv_rect[3] - uv_rect[1]).max(1.0e-6);
        let source_w = (uv_w * self.ring_w.max(1) as f32).max(1.0).round() as u32;
        let source_h = (uv_h * self.ring_h.max(1) as f32).max(1.0).round() as u32;
        let fg_active = options.fg_enabled
            && options.fg_phase > 0.001
            && options.fg_phase < 0.999
            && previous.seq != current.seq;
        let use_motion = fg_active && options.motion_compensation;
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

        let hud_rect = self.hud_rect(bb_w, bb_h);
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
                if self.ring_format == Some(DXGI_FORMAT_R16G16B16A16_FLOAT) {
                    1.0
                } else {
                    0.0
                },
                if options.preserve_aspect { 1.0 } else { 0.0 },
                options.fg_strength.clamp(0.0, 1.0),
            ],
            params2: [
                0.45 - options.scene_protection.clamp(0.0, 1.0) * 0.35,
                0.0,
                0.0,
                0.0,
            ],
        };

        let output_srv = self
            .output
            .as_ref()
            .map(|o| o.srv.clone())
            .ok_or_else(|| GfxError::Other("output target missing".into()))?;
        let output_rtv = self
            .output
            .as_ref()
            .map(|o| o.rtv.clone())
            .ok_or_else(|| GfxError::Other("output target missing".into()))?;
        let flow_srv = if use_motion {
            self.flow
                .as_ref()
                .map(|f| f.srv.clone())
                .unwrap_or_else(|| current.srv.clone())
        } else {
            current.srv.clone()
        };

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

            // Pass 1: temporal composition + spatial reconstruction.
            self.ctx
                .OMSetRenderTargets(Some(&[Some(output_rtv.clone())]), None);
            self.ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: bb_w,
                Height: bb_h,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            self.ctx
                .ClearRenderTargetView(&output_rtv, &[0.0, 0.0, 0.0, 1.0]);
            self.ctx.PSSetShader(&self.ps_compose, None);
            self.ctx.PSSetShaderResources(
                0,
                Some(&[
                    Some(previous.srv.clone()),
                    Some(current.srv.clone()),
                    Some(flow_srv),
                ]),
            );
            self.ctx.Draw(3, 0);

            // Pass 2: contrast-adaptive sharpening into the swapchain.
            self.ctx
                .OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);
            self.ctx.PSSetShader(&self.ps_rcas, None);
            self.ctx
                .PSSetShaderResources(0, Some(&[Some(output_srv.clone())]));
            self.ctx.Draw(3, 0);

            // Pass 3: premultiplied-alpha HUD.
            if let Some(text) = hud_text {
                let should_update = self
                    .hud_last_update
                    .is_none_or(|last| last.elapsed() >= Duration::from_millis(250));
                if should_update {
                    self.hud.update(&self.ctx, text);
                    self.hud_last_update = Some(Instant::now());
                }
                self.ctx.PSSetShader(&self.ps_hud, None);
                self.ctx
                    .PSSetShaderResources(0, Some(&[Some(self.hud.srv().clone())]));
                self.ctx
                    .OMSetBlendState(Some(&self.blend_premult), None, 0xffff_ffff);
                self.ctx.Draw(3, 0);
            }

            self.ctx.End(&self.queries[qp_slot].end);
            let null_srvs: [Option<ID3D11ShaderResourceView>; 3] = [None, None, None];
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
