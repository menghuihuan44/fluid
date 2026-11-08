//! DXGI flip-model presentation backend (cfg(windows), backend name
//! dxgi-swapchain).
//!
//! This is the low-latency Windows path: a D3D11 device owns an
//! IDXGISwapChain1 created for the same overlay window the GDI backend uses
//! (window.rs), and every frame is copied into the current flip-model back
//! buffer. Flip model presentaion lets DWM composite the frame without a copy
//! through the GDI redirection surface, and with vsync off plus tearing
//! support it can present immediately instead of waiting for the next vblank.
//!
//! Data path per frame:
//! 1. rows of the incoming frame are translated to BGRA (Rgba8 swaps channels,
//!    Bgra8 is copied, Rgb8 and Luma8 are expanded) straight into a
//!    D3D11_USAGE_DYNAMIC texture mapped once with WRITE_DISCARD, honouring
//!    the driver row pitch, so there is exactly one upload and no allocation
//!    per frame;
//! 2. the upload texture is copied into the swap chain back buffer with
//!    CopyResource (same size, same format);
//! 3. Present(1, 0) when vsync is on, otherwise Present(0,
//!    DXGI_PRESENT_ALLOW_TEARING) when the adapter reported tearing support,
//!    else Present(0, 0).
//!
//! The swap chain stays at the frame size; the window box (window.rs) already
//! fits the frame into the monitor work area, and DXGI_SCALING_STRETCH lets
//! the compositor scale the buffer to the window when the user changes the
//! size of a decorated window. If the frame size changes the back buffers are
//! resized with ResizeBuffers and the upload texture is recreated.
//!
//! Failure handling: a removed or reset display device turns into a clear
//! FluidError::platform that tells the caller to reopen the sink; nothing here
//! panics on driver errors. Vsync is provided by DXGI itself (Present with
//! sync interval 1), there is no separate frame limiter in this backend.
//!
//! Threading: the device, swap chain and window are created on the calling
//! thread and must be used from that one thread afterwards (documented, not
//! enforced). Send is implemented only because the engine moves the sink to
//! its render thread before the first present.

use std::ffi::c_void;
use std::mem::size_of;

use fluid_core::{Dims, FluidError, Frame, FrameSink, PixelFormat, PresentConfig, Result};
use windows::Win32::Foundation::{BOOL, HMODULE, RECT};
use windows::Win32::Graphics::Direct3D::{
    D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL, D3D_FEATURE_LEVEL_10_0, D3D_FEATURE_LEVEL_10_1,
    D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1,
};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BIND_SHADER_RESOURCE, D3D11_CPU_ACCESS_WRITE, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
    D3D11_MAP_WRITE_DISCARD, D3D11_MAPPED_SUBRESOURCE, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_DYNAMIC, D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_ALPHA_MODE_IGNORE, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_RESET, DXGI_FEATURE_PRESENT_ALLOW_TEARING,
    DXGI_PRESENT, DXGI_PRESENT_ALLOW_TEARING, DXGI_SCALING_STRETCH, DXGI_SWAP_CHAIN_DESC1,
    DXGI_SWAP_CHAIN_FLAG, DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING, DXGI_SWAP_EFFECT_FLIP_DISCARD,
    DXGI_USAGE_RENDER_TARGET_OUTPUT, IDXGIDevice, IDXGIFactory2, IDXGIFactory5, IDXGISwapChain1,
};
use windows::Win32::UI::WindowsAndMessaging::GetClientRect;
use windows::core::Interface;

use super::window::OverlayWindow;

/// Number of back buffers: two is the flip-model minimum, enough to keep one
/// frame being presented while the next is written.
const BUFFER_COUNT: u32 = 2;

fn platform(what: &str, err: windows::core::Error) -> FluidError {
    FluidError::platform(format!("dxgi-swapchain: {what}: {err}"))
}

/// Map a DXGI failure to a caller-actionable error. Device removal and reset
/// are the two failures a caller can recover from by reopening presentation.
fn dxgi_error(what: &str, err: windows::core::Error) -> FluidError {
    let code = err.code();
    if code == DXGI_ERROR_DEVICE_REMOVED {
        FluidError::platform(format!(
            "dxgi-swapchain: {what}: the display device was removed; reopen presentation"
        ))
    } else if code == DXGI_ERROR_DEVICE_RESET {
        FluidError::platform(format!(
            "dxgi-swapchain: {what}: the display device was reset; reopen presentation"
        ))
    } else {
        platform(what, err)
    }
}

/// Swap chain description for a frame of the given size. Flip discard with two
/// buffers is the lowest-latency configuration available on Windows 10+; the
/// tearing flag is only advertised when the caller may present without vsync.
pub(crate) fn swapchain_desc(dims: Dims, allow_tearing: bool) -> DXGI_SWAP_CHAIN_DESC1 {
    DXGI_SWAP_CHAIN_DESC1 {
        Width: dims.width.max(1),
        Height: dims.height.max(1),
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        Stereo: BOOL(0),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: BUFFER_COUNT,
        Scaling: DXGI_SCALING_STRETCH,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
        AlphaMode: DXGI_ALPHA_MODE_IGNORE,
        Flags: if allow_tearing {
            DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0 as u32
        } else {
            0
        },
    }
}

/// Present arguments: (sync interval, flags). Vsync waits for the next vblank;
/// without vsync a tearing-capable chain may present immediately, otherwise
/// sync interval 0 still lets the compositor pace the queue.
pub(crate) fn present_flags(vsync: bool, allow_tearing: bool) -> (u32, DXGI_PRESENT) {
    if vsync {
        (1, DXGI_PRESENT(0))
    } else if allow_tearing {
        (0, DXGI_PRESENT_ALLOW_TEARING)
    } else {
        (0, DXGI_PRESENT(0))
    }
}

/// Translate one frame row into the BGRA layout the swap chain expects.
/// Writes exactly width * 4 bytes into dst; the rest of dst is untouched so a
/// caller can hand over the mapped row directly, padding included.
// Frame rows come from producers with their own strides: chunks_exact keeps a
// ragged tail harmless, while as_chunks would panic mid-presentation.
#[allow(clippy::chunks_exact_to_as_chunks)]
#[inline]
pub(crate) fn row_to_bgra(frame: &Frame, y: u32, dst: &mut [u8]) -> Result<()> {
    let width = frame.dims.width as usize;
    let row_bytes = width * 4;
    if y >= frame.dims.height {
        return Err(FluidError::invalid(format!(
            "dxgi-swapchain: row {y} is outside a {}x{} frame",
            frame.dims.width, frame.dims.height
        )));
    }
    if dst.len() < row_bytes {
        return Err(FluidError::invalid(format!(
            "dxgi-swapchain: row buffer is {} bytes, need {row_bytes}",
            dst.len()
        )));
    }
    let bpp = frame.fmt.bytes_per_pixel();
    let offset = y as usize * frame.stride();
    let end = offset + width * bpp;
    let src = frame.data.get(offset..end).ok_or_else(|| {
        FluidError::invalid("dxgi-swapchain: frame buffer is smaller than its dimensions")
    })?;
    let out = &mut dst[..row_bytes];
    match frame.fmt {
        PixelFormat::Rgba8 => {
            for (s, d) in src.chunks_exact(4).zip(out.chunks_exact_mut(4)) {
                d[0] = s[2];
                d[1] = s[1];
                d[2] = s[0];
                d[3] = 255;
            }
        }
        PixelFormat::Bgra8 => {
            out.copy_from_slice(src);
            // ALPHA_MODE_IGNORE means the blend value is unused, but keeping
            // it opaque avoids confusing debuggers and screenshot tools.
            for d in out.chunks_exact_mut(4) {
                d[3] = 255;
            }
        }
        PixelFormat::Rgb8 => {
            for (s, d) in src.chunks_exact(3).zip(out.chunks_exact_mut(4)) {
                d[0] = s[2];
                d[1] = s[1];
                d[2] = s[0];
                d[3] = 255;
            }
        }
        PixelFormat::Luma8 => {
            for (s, d) in src.iter().zip(out.chunks_exact_mut(4)) {
                d[0] = *s;
                d[1] = *s;
                d[2] = *s;
                d[3] = 255;
            }
        }
    }
    Ok(())
}

/// Create a hardware D3D11 device with BGRA support. Feature level 11.1 may be
/// rejected on older runtimes (E_INVALIDARG), in which case we retry from 11.0.
fn create_device() -> Result<(ID3D11Device, ID3D11DeviceContext)> {
    const LEVELS: [D3D_FEATURE_LEVEL; 4] = [
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    ];
    let mut device: Option<ID3D11Device> = None;
    let mut context: Option<ID3D11DeviceContext> = None;
    // SAFETY: all out parameters point at live Options; no adapter is passed,
    // so a hardware adapter is chosen by the runtime with no DXGI enum helper.
    let first = unsafe {
        D3D11CreateDevice(
            None::<&windows::Win32::Graphics::Dxgi::IDXGIAdapter>,
            D3D_DRIVER_TYPE_HARDWARE,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            Some(&LEVELS),
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )
    };
    if let Err(err) = first {
        let mut retry_device: Option<ID3D11Device> = None;
        let mut retry_context: Option<ID3D11DeviceContext> = None;
        // SAFETY: same contract as above, feature levels start at 11.0.
        unsafe {
            D3D11CreateDevice(
                None::<&windows::Win32::Graphics::Dxgi::IDXGIAdapter>,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                Some(&LEVELS[1..]),
                D3D11_SDK_VERSION,
                Some(&mut retry_device),
                None,
                Some(&mut retry_context),
            )
        }
        .map_err(|retry_err| {
            FluidError::platform(format!(
                "dxgi-swapchain: D3D11CreateDevice failed: {err} (retry without 11.1: {retry_err})"
            ))
        })?;
        device = retry_device;
        context = retry_context;
    }
    let device = device.ok_or_else(|| {
        FluidError::platform("dxgi-swapchain: D3D11CreateDevice returned no device")
    })?;
    let context = context.ok_or_else(|| {
        FluidError::platform("dxgi-swapchain: D3D11CreateDevice returned no context")
    })?;
    Ok((device, context))
}

/// Create the dynamic BGRA texture that receives the CPU frame.
fn create_upload(device: &ID3D11Device, dims: Dims) -> Result<ID3D11Texture2D> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: dims.width.max(1),
        Height: dims.height.max(1),
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DYNAMIC,
        // A texture with no bind flags at all is rejected by at least the
        // Intel driver (E_INVALIDARG): D3D11 wants a bind flag on every
        // resource. Shader-resource is the cheapest valid choice here; the
        // texture is only ever mapped and copied, never sampled.
        BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
        MiscFlags: 0,
    };
    let mut texture: Option<ID3D11Texture2D> = None;
    // SAFETY: desc is a valid description and no initial data is supplied.
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture)) }
        .map_err(|e| dxgi_error("CreateTexture2D", e))?;
    texture
        .ok_or_else(|| FluidError::platform("dxgi-swapchain: CreateTexture2D returned no texture"))
}

/// Current back buffer of the swap chain.
fn get_backbuffer(swapchain: &IDXGISwapChain1) -> Result<ID3D11Texture2D> {
    // SAFETY: buffer 0 always exists on a chain created with BUFFER_COUNT >= 1.
    unsafe { swapchain.GetBuffer::<ID3D11Texture2D>(0) }.map_err(|e| dxgi_error("GetBuffer", e))
}

/// Ask the adapter whether it supports variable-refresh tearing presents.
fn check_tearing(factory: &IDXGIFactory2) -> bool {
    let Ok(factory5) = factory.cast::<IDXGIFactory5>() else {
        return false;
    };
    let mut supported = BOOL(0);
    // SAFETY: the feature data pointer and size describe a live BOOL, which is
    // what DXGI_FEATURE_PRESENT_ALLOW_TEARING expects.
    let ok = unsafe {
        factory5.CheckFeatureSupport(
            DXGI_FEATURE_PRESENT_ALLOW_TEARING,
            &mut supported as *mut BOOL as *mut c_void,
            size_of::<BOOL>() as u32,
        )
    }
    .is_ok();
    ok && supported.as_bool()
}

/// Flip-model window sink. Keep every call on the thread that opened it.
pub(crate) struct DxgiSwapSink {
    window: OverlayWindow,
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    swapchain: IDXGISwapChain1,
    backbuffer: Option<ID3D11Texture2D>,
    upload: Option<ID3D11Texture2D>,
    dims: Dims,
    vsync: bool,
    tearing: bool,
    closed: bool,
}

// SAFETY: the fields are COM interfaces and a window handle, none of which
// implement Send. The engine constructs the sink on one thread and moves it to
// the render thread before the first present; from then on one thread uses it
// (documented in the module docs).
unsafe impl Send for DxgiSwapSink {}

impl DxgiSwapSink {
    /// Rebuild the back buffers and upload texture for a new frame size.
    fn reconfigure(&mut self, dims: Dims) -> Result<()> {
        if dims.is_empty() {
            return Err(FluidError::invalid(
                "present: frame dimensions must be non-zero",
            ));
        }
        // Drop our references before resizing; DXGI refuses to resize while a
        // back buffer is still referenced.
        self.backbuffer = None;
        self.upload = None;
        let flags = if self.tearing {
            DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
        } else {
            DXGI_SWAP_CHAIN_FLAG(0)
        };
        // SAFETY: resize to the new frame size in the format the chain was
        // created with; pass 0 buffers to keep the existing count.
        unsafe {
            self.swapchain.ResizeBuffers(
                BUFFER_COUNT,
                dims.width.max(1),
                dims.height.max(1),
                DXGI_FORMAT_B8G8R8A8_UNORM,
                flags,
            )
        }
        .map_err(|e| dxgi_error("ResizeBuffers", e))?;
        self.dims = dims;
        self.upload = Some(create_upload(&self.device, dims)?);
        self.backbuffer = Some(get_backbuffer(&self.swapchain)?);
        Ok(())
    }

    /// Upload one frame into the dynamic texture, converting rows to BGRA.
    fn write_frame(&mut self, frame: &Frame) -> Result<()> {
        let upload = self
            .upload
            .as_ref()
            .ok_or_else(|| FluidError::platform("dxgi-swapchain: upload texture is missing"))?;
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        // SAFETY: a full WRITE_DISCARD map of the whole dynamic texture; the
        // mapped pointer stays valid until Unmap below.
        unsafe {
            self.context
                .Map(upload, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut mapped))
        }
        .map_err(|e| dxgi_error("Map", e))?;
        let result = (|| {
            let base = mapped.pData as *mut u8;
            let pitch = mapped.RowPitch as usize;
            let row_bytes = frame.dims.width as usize * 4;
            for y in 0..frame.dims.height {
                // SAFETY: the mapped region covers RowPitch * Height bytes, and
                // row_bytes never exceeds RowPitch for a BGRA texture.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(base.add(y as usize * pitch), row_bytes)
                };
                row_to_bgra(frame, y, dst)?;
            }
            Ok(())
        })();
        // SAFETY: unmaps the resource mapped just above; Unmap has no result.
        unsafe { self.context.Unmap(upload, 0) };
        result
    }

    fn shutdown(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        self.backbuffer = None;
        self.upload = None;
        self.window.destroy();
    }
}

impl Drop for DxgiSwapSink {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl FrameSink for DxgiSwapSink {
    fn dims(&self) -> Dims {
        self.dims
    }

    fn backend_name(&self) -> &'static str {
        "dxgi-swapchain"
    }

    fn present(&mut self, frame: &Frame) -> Result<()> {
        if self.closed {
            return Err(FluidError::cancelled());
        }
        self.window.pump();
        if self.window.close_requested() {
            return Err(FluidError::cancelled());
        }
        // A resize notification only matters for input handling here; the
        // swap chain buffers follow the frame size, not the window size.
        let _ = self.window.take_resized();

        let mut client = RECT::default();
        // SAFETY: reads the client rectangle of our own window.
        unsafe { GetClientRect(self.window.hwnd(), &mut client) }
            .map_err(|e| platform("GetClientRect", e))?;
        if client.right <= client.left || client.bottom <= client.top {
            // Minimised: report success so the engine keeps rendering.
            return Ok(());
        }

        if frame.dims != self.dims {
            self.reconfigure(frame.dims)?;
        }
        self.write_frame(frame)?;

        let backbuffer = self
            .backbuffer
            .as_ref()
            .ok_or_else(|| FluidError::platform("dxgi-swapchain: back buffer is missing"))?;
        let upload = self
            .upload
            .as_ref()
            .ok_or_else(|| FluidError::platform("dxgi-swapchain: upload texture is missing"))?;
        // SAFETY: both textures were created with the same size and format, so
        // a full-resource copy is valid; no render targets are bound.
        unsafe { self.context.CopyResource(backbuffer, upload) };

        let (sync_interval, flags) = present_flags(self.vsync, self.tearing);
        // SAFETY: presents the current back buffer; the HRESULT carries failure
        // information instead of panicking.
        let hr = unsafe { self.swapchain.Present(sync_interval, flags) };
        if hr.is_err() {
            return Err(dxgi_error(
                "Present",
                windows::core::Error::from_hresult(hr),
            ));
        }
        Ok(())
    }

    fn should_close(&self) -> bool {
        self.window.pump();
        self.closed || self.window.close_requested()
    }

    fn close(&mut self) {
        self.shutdown();
    }
}

/// Whether this build can create the D3D11 device and factory a flip-model
/// swap chain needs. Used by backend_report; no window is created.
pub(crate) fn available() -> bool {
    let device = match create_device() {
        Ok((device, _context)) => device,
        Err(_) => return false,
    };
    let Ok(dxgi_device) = device.cast::<IDXGIDevice>() else {
        return false;
    };
    // SAFETY: the DXGI device is live for the duration of the queries, and
    // the adapter it returns is owned by that device.
    let Ok(adapter) = (unsafe { dxgi_device.GetAdapter() }) else {
        return false;
    };
    // SAFETY: GetParent only reads the adapter's creation factory.
    unsafe { adapter.GetParent::<IDXGIFactory2>() }.is_ok()
}

/// Open the flip-model swap chain sink for the configured monitor.
pub(crate) fn open(config: &PresentConfig, dims: Dims) -> Result<Box<dyn FrameSink>> {
    if dims.is_empty() {
        return Err(FluidError::invalid(
            "present: frame dimensions must be non-zero",
        ));
    }
    let (device, context) = create_device()?;
    let dxgi_device: IDXGIDevice = device
        .cast()
        .map_err(|e| platform("QueryInterface(IDXGIDevice)", e))?;
    // SAFETY: the DXGI device is live for the duration of the queries.
    let adapter = unsafe { dxgi_device.GetAdapter() }.map_err(|e| dxgi_error("GetAdapter", e))?;
    // SAFETY: GetParent only reads the adapter's creation factory.
    let factory: IDXGIFactory2 =
        unsafe { adapter.GetParent() }.map_err(|e| dxgi_error("GetParent(IDXGIFactory2)", e))?;
    let tearing = check_tearing(&factory);
    let window = OverlayWindow::create(config, dims)?;
    let desc = swapchain_desc(dims, tearing);
    // SAFETY: the device, window and description are all live for the call; no
    // fullscreen description means windowed presentation to our own HWND.
    let swapchain =
        unsafe { factory.CreateSwapChainForHwnd(&device, window.hwnd(), &desc, None, None) }
            .map_err(|e| dxgi_error("CreateSwapChainForHwnd", e))?;
    let upload = create_upload(&device, dims)?;
    let backbuffer = get_backbuffer(&swapchain)?;
    Ok(Box::new(DxgiSwapSink {
        window,
        device,
        context,
        swapchain,
        backbuffer: Some(backbuffer),
        upload: Some(upload),
        dims,
        vsync: config.vsync,
        tearing,
        closed: false,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::core::HRESULT;

    fn frame_with(fmt: PixelFormat, dims: Dims, data: Vec<u8>) -> Frame {
        Frame {
            dims,
            fmt,
            data,
            pts_ns: 0,
            index: 0,
        }
    }

    #[test]
    fn row_to_bgra_swaps_rgba_channels() {
        let frame = frame_with(
            PixelFormat::Rgba8,
            Dims::new(2, 1),
            vec![10, 20, 30, 255, 1, 2, 3, 0],
        );
        let mut dst = [0u8; 8];
        row_to_bgra(&frame, 0, &mut dst).unwrap();
        assert_eq!(dst, [30, 20, 10, 255, 3, 2, 1, 255]);
    }

    #[test]
    fn row_to_bgra_handles_odd_width_and_padding() {
        // Three pixels: dst is the full padded mapped row (12 usable bytes plus
        // four bytes of driver padding that must stay untouched).
        let frame = frame_with(
            PixelFormat::Rgba8,
            Dims::new(3, 2),
            vec![
                1, 2, 3, 9, 4, 5, 6, 9, 7, 8, 9, 9, // row 0
                11, 12, 13, 9, 14, 15, 16, 9, 17, 18, 19, 9, // row 1
            ],
        );
        let mut dst = [0xAAu8; 16];
        row_to_bgra(&frame, 1, &mut dst).unwrap();
        assert_eq!(
            &dst[..12],
            &[13, 12, 11, 255, 16, 15, 14, 255, 19, 18, 17, 255]
        );
        assert_eq!(&dst[12..], &[0xAA, 0xAA, 0xAA, 0xAA]);
    }

    #[test]
    fn row_to_bgra_copies_bgra_and_forces_opaque() {
        let frame = frame_with(
            PixelFormat::Bgra8,
            Dims::new(2, 1),
            vec![10, 20, 30, 0, 40, 50, 60, 7],
        );
        let mut dst = [0u8; 8];
        row_to_bgra(&frame, 0, &mut dst).unwrap();
        assert_eq!(dst, [10, 20, 30, 255, 40, 50, 60, 255]);
    }

    #[test]
    fn row_to_bgra_expands_rgb8_and_luma8() {
        let rgb = frame_with(
            PixelFormat::Rgb8,
            Dims::new(2, 1),
            vec![10, 20, 30, 40, 50, 60],
        );
        let mut dst = [0u8; 8];
        row_to_bgra(&rgb, 0, &mut dst).unwrap();
        assert_eq!(dst, [30, 20, 10, 255, 60, 50, 40, 255]);

        let luma = frame_with(PixelFormat::Luma8, Dims::new(2, 1), vec![7, 200]);
        let mut dst = [0u8; 8];
        row_to_bgra(&luma, 0, &mut dst).unwrap();
        assert_eq!(dst, [7, 7, 7, 255, 200, 200, 200, 255]);
    }

    #[test]
    fn row_to_bgra_rejects_bad_rows_and_short_buffers() {
        let frame = frame_with(PixelFormat::Rgba8, Dims::new(2, 1), vec![0u8; 8]);
        let mut dst = [0u8; 8];
        assert!(row_to_bgra(&frame, 1, &mut dst).is_err());
        let mut short = [0u8; 4];
        assert!(row_to_bgra(&frame, 0, &mut short).is_err());
        // A frame whose data does not match its dimensions is rejected instead
        // of reading out of bounds.
        let truncated = frame_with(PixelFormat::Rgba8, Dims::new(2, 1), vec![0u8; 4]);
        assert!(row_to_bgra(&truncated, 0, &mut dst).is_err());
    }

    #[test]
    fn swapchain_desc_uses_flip_discard_and_two_buffers() {
        let desc = swapchain_desc(Dims::new(1920, 1080), false);
        assert_eq!(desc.Width, 1920);
        assert_eq!(desc.Height, 1080);
        assert_eq!(desc.Format, DXGI_FORMAT_B8G8R8A8_UNORM);
        assert_eq!(desc.BufferCount, BUFFER_COUNT);
        assert_eq!(desc.SwapEffect, DXGI_SWAP_EFFECT_FLIP_DISCARD);
        assert_eq!(desc.Scaling, DXGI_SCALING_STRETCH);
        assert_eq!(desc.AlphaMode, DXGI_ALPHA_MODE_IGNORE);
        assert!(!desc.Stereo.as_bool());
        assert_eq!(desc.SampleDesc.Count, 1);
        assert_eq!(desc.Flags, 0);
    }

    #[test]
    fn swapchain_desc_sets_tearing_flag_and_clamps_size() {
        let desc = swapchain_desc(Dims::new(0, 0), true);
        assert_eq!(desc.Width, 1);
        assert_eq!(desc.Height, 1);
        assert_eq!(desc.Flags, DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0 as u32);
    }

    #[test]
    fn present_flags_cover_vsync_tearing_and_fallback() {
        assert_eq!(present_flags(true, false), (1, DXGI_PRESENT(0)));
        // Vsync wins even when tearing is available.
        assert_eq!(present_flags(true, true), (1, DXGI_PRESENT(0)));
        assert_eq!(present_flags(false, true), (0, DXGI_PRESENT_ALLOW_TEARING));
        assert_eq!(present_flags(false, false), (0, DXGI_PRESENT(0)));
    }

    #[test]
    fn device_error_mapping_mentions_removal() {
        let removed = dxgi_error(
            "Present",
            windows::core::Error::from_hresult(DXGI_ERROR_DEVICE_REMOVED),
        );
        let text = removed.to_string();
        assert!(text.contains("dxgi-swapchain"));
        assert!(text.contains("removed"));
        let reset = dxgi_error(
            "Present",
            windows::core::Error::from_hresult(DXGI_ERROR_DEVICE_RESET),
        );
        assert!(reset.to_string().contains("reset"));
        let other = dxgi_error(
            "Present",
            windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32)),
        );
        assert!(other.to_string().contains("dxgi-swapchain: Present"));
    }
}
