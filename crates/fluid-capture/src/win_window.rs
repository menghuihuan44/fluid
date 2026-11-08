//! Window capture backend (Windows 10 1903+ / Windows 11).
//!
//! Captures one application window instead of a whole monitor.
//!
//! Preferred path: Windows.Graphics.Capture (WGC). A "GraphicsCaptureItem" for
//! the selected window handle feeds a free-threaded capture frame pool (no
//! DispatcherQueue and no message pump needed) and a capture session. Frames are
//! pulled with "TryGetNextFrame" inside the "CaptureConfig::timeout_ms" budget,
//! copied into a CPU readable staging texture and converted BGRA -> RGBA row by
//! row. WGC runs inside the desktop compositor, so the window is captured even
//! when another window covers it, and it needs no hooks, no injection and no game
//! memory access. The yellow capture border is turned off best effort; recent
//! Windows builds may ignore that and still draw a thin border.
//!
//! Fallback path: when WGC cannot be created (older build, remote session, some
//! protected windows) the backend uses DXGI desktop duplication of the monitor
//! the window currently sits on and crops every desktop frame to the window
//! rectangle. This path is honest about its limits: occluded parts of the window
//! show whatever is on top of them, the mouse pointer is not composited, and the
//! title bar / border is part of the captured content.
//!
//! Geometry: the output size is fixed when the source is opened (window content
//! size, or the configured region inside it) because the engine builds its
//! pipeline for one size. Moving the window is followed every frame. Resizing
//! keeps the pipeline size: content that no longer fits is cropped, content that
//! became smaller is padded with black. Reopen the source to adopt a new size.
//! "CaptureConfig::region" is a crop inside the window, not a desktop region.
//!
//! Liveness: a minimized window, a closed window, a lost duplication or a frame
//! timeout repeats the previous frame with a fresh timestamp and sequence number.
//! A live source never returns "Ok(None)"; when the window closes the last frame
//! keeps repeating until the sink closes.
//!
//! Threading: create, use and drop the source on one thread at a time. The type
//! is "Send" so the engine can move it to its capture thread; the COM objects
//! underneath are free-threaded but the capture session is meant to be used
//! serially. Calls block up to "timeout_ms".

use std::ffi::c_void;
use std::mem::size_of;
use std::time::{Duration, Instant};

use fluid_core::{
    CaptureConfig, Dims, FluidError, Frame, FrameSource, PixelFormat, Rect, Result, now_ns,
    sleep_until,
};

use windows::Graphics::Capture::{
    Direct3D11CaptureFrame, Direct3D11CaptureFramePool, GraphicsCaptureItem, GraphicsCaptureSession,
};
use windows::Graphics::DirectX::Direct3D11::IDirect3DDevice;
use windows::Graphics::DirectX::DirectXPixelFormat;
use windows::Win32::Foundation::{BOOL, CloseHandle, HMODULE, HWND, LPARAM, RECT};
use windows::Win32::Graphics::Direct3D::{
    D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_10_0, D3D_FEATURE_LEVEL_10_1,
    D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1,
};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_CPU_ACCESS_READ, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_MAP_READ,
    D3D11_MAPPED_SUBRESOURCE, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING,
    D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dwm::{
    DWMWA_CLOAKED, DWMWA_EXTENDED_FRAME_BOUNDS, DwmGetWindowAttribute,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{
    DXGI_ERROR_ACCESS_LOST, DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_WAIT_TIMEOUT, IDXGIAdapter,
    IDXGIDevice, IDXGIOutputDuplication, IDXGIResource,
};
use windows::Win32::Graphics::Gdi::{HMONITOR, MONITOR_DEFAULTTONEAREST, MonitorFromWindow};
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx};
use windows::Win32::System::Threading::{
    OpenProcess, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION, QueryFullProcessImageNameW,
};
use windows::Win32::System::WinRT::Direct3D11::{
    CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess,
};
use windows::Win32::System::WinRT::Graphics::Capture::IGraphicsCaptureItemInterop;
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GWL_EXSTYLE, GetClassNameW, GetForegroundWindow, GetWindowLongW, GetWindowRect,
    GetWindowTextLengthW, GetWindowTextW, GetWindowThreadProcessId, IsIconic, IsWindow,
    IsWindowVisible, WS_EX_TOOLWINDOW,
};
use windows::core::{Interface, PWSTR};

use crate::format::convert_row;
use crate::win::{
    DesktopLayout, bgra_row_to_rgba, blank_frame, create_device, create_staging_with_format,
    desktop_layout, duplicate_output, live_outputs,
};
use crate::window::{
    PixelRect, WindowInfo, WindowSelector, capture_region, replay, screen_rect, select_window,
};

/// Hard cap so a desktop with hundreds of helper windows cannot produce a huge
/// list; the enumeration stops before the vector grows without bound.
const MAX_WINDOWS: usize = 512;

/// Wrap a failed Windows call, keeping it a platform error.
fn platform(what: &str, err: windows::core::Error) -> FluidError {
    FluidError::platform(format!("window capture: {what}: {err}"))
}

/// The window frame rectangle in physical desktop pixels.
///
/// "DwmGetWindowAttribute(DWMWA_EXTENDED_FRAME_BOUNDS)" is preferred because it
/// reports physical pixels even for a DPI aware process, while "GetWindowRect"
/// can be virtualized. The DWM call fails for some windows; fall back then.
fn window_frame_rect(hwnd: HWND) -> Option<RECT> {
    let mut rect = RECT::default();
    // SAFETY: hwnd is a live window handle and rect is a local out parameter of
    // the advertised size.
    let dwm = unsafe {
        DwmGetWindowAttribute(
            hwnd,
            DWMWA_EXTENDED_FRAME_BOUNDS,
            &mut rect as *mut RECT as *mut c_void,
            size_of::<RECT>() as u32,
        )
    };
    if dwm.is_ok() && rect.right > rect.left && rect.bottom > rect.top {
        return Some(rect);
    }
    let mut fallback = RECT::default();
    // SAFETY: same as above, plain GetWindowRect query.
    unsafe { GetWindowRect(hwnd, &mut fallback) }.ok()?;
    if fallback.right > fallback.left && fallback.bottom > fallback.top {
        Some(fallback)
    } else {
        None
    }
}

/// Convert a Win32 rectangle into the signed pixel rectangle helper.
fn pixel_rect(rect: RECT) -> PixelRect {
    PixelRect::new(
        rect.left,
        rect.top,
        (rect.right - rect.left).max(0) as u32,
        (rect.bottom - rect.top).max(0) as u32,
    )
}

/// True when DWM reports the window as cloaked (hidden UWP shell windows,
/// windows on another virtual desktop).
fn is_cloaked(hwnd: HWND) -> bool {
    let mut cloaked = 0u32;
    // SAFETY: hwnd is a live window handle; cloaked is a u32 out parameter.
    let result = unsafe {
        DwmGetWindowAttribute(
            hwnd,
            DWMWA_CLOAKED,
            &mut cloaked as *mut u32 as *mut c_void,
            size_of::<u32>() as u32,
        )
    };
    result.is_ok() && cloaked != 0
}

fn window_title(hwnd: HWND) -> String {
    // SAFETY: read-only queries on a live window handle.
    let length = unsafe { GetWindowTextLengthW(hwnd) };
    if length <= 0 {
        return String::new();
    }
    let mut buffer = vec![0u16; length as usize + 1];
    // SAFETY: the buffer has room for the text plus a terminator.
    let written = unsafe { GetWindowTextW(hwnd, &mut buffer) }.max(0) as usize;
    String::from_utf16_lossy(&buffer[..written.min(buffer.len())])
}

fn window_class(hwnd: HWND) -> String {
    let mut buffer = [0u16; 256];
    // SAFETY: the buffer is a writable array of the length passed to the API.
    let written = unsafe { GetClassNameW(hwnd, &mut buffer) }.max(0) as usize;
    String::from_utf16_lossy(&buffer[..written.min(buffer.len())])
}

/// File name part of a full executable path, used for "WindowInfo::process_name".
fn process_file_name(path: &str) -> String {
    if let Some(name) = std::path::Path::new(path).file_name() {
        return name.to_string_lossy().into_owned();
    }
    path.to_string()
}

fn window_process_name(hwnd: HWND) -> Option<String> {
    let mut pid = 0u32;
    // SAFETY: read-only query; pid points at a local.
    unsafe { GetWindowThreadProcessId(hwnd, Some(&mut pid)) };
    if pid == 0 {
        return None;
    }
    // SAFETY: PROCESS_QUERY_LIMITED_INFORMATION is the minimum access needed to
    // read the image name, and the handle is closed below.
    let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) }.ok()?;
    let mut buffer = [0u16; 1024];
    let mut length = buffer.len() as u32;
    // SAFETY: the buffer is writable and length advertises its capacity.
    let result = unsafe {
        QueryFullProcessImageNameW(
            process,
            PROCESS_NAME_WIN32,
            PWSTR(buffer.as_mut_ptr()),
            &mut length,
        )
    };
    // SAFETY: the handle came from OpenProcess and is not used afterwards.
    let _ = unsafe { CloseHandle(process) };
    result.ok()?;
    let full = String::from_utf16_lossy(&buffer[..(length as usize).min(buffer.len())]);
    Some(process_file_name(&full))
}

/// Build one "WindowInfo" for a live window, or None when it is not capturable.
fn window_info(hwnd: HWND) -> Option<WindowInfo> {
    // SAFETY: read-only window queries; an invalid handle simply returns false.
    if !unsafe { IsWindowVisible(hwnd) }.as_bool() {
        return None;
    }
    let title = window_title(hwnd);
    // Untitled windows are shell helpers, tooltips or hidden hosts; they are not
    // useful capture targets and would flood the list.
    if title.is_empty() {
        return None;
    }
    if is_cloaked(hwnd) {
        return None;
    }
    // SAFETY: read-only style query.
    let ex_style = unsafe { GetWindowLongW(hwnd, GWL_EXSTYLE) };
    if ex_style & WS_EX_TOOLWINDOW.0 as i32 != 0 {
        return None;
    }
    let rect = window_frame_rect(hwnd)?;
    let width = (rect.right - rect.left).max(0) as u32;
    let height = (rect.bottom - rect.top).max(0) as u32;
    if width == 0 || height == 0 {
        return None;
    }
    // SAFETY: read-only query.
    let minimized = unsafe { IsIconic(hwnd) }.as_bool();
    Some(WindowInfo {
        id: hwnd.0 as usize as u64,
        title,
        class_name: window_class(hwnd),
        process_name: window_process_name(hwnd).unwrap_or_default(),
        dims: Dims::new(width, height),
        left: rect.left,
        top: rect.top,
        visible: true,
        minimized,
    })
}

/// State threaded through the "EnumWindows" callback.
struct EnumState {
    windows: Vec<WindowInfo>,
    truncated: bool,
}

/// "EnumWindows" callback: appends one entry per capturable window.
///
/// SAFETY: the caller passes a pointer to an "EnumState" that outlives the whole
/// enumeration, as required by the "EnumWindows" contract.
unsafe extern "system" fn enum_window_proc(hwnd: HWND, lparam: LPARAM) -> BOOL {
    // SAFETY: lparam is the pointer supplied by "enumerate_windows".
    let state = unsafe { &mut *(lparam.0 as usize as *mut EnumState) };
    if state.windows.len() >= MAX_WINDOWS {
        state.truncated = true;
        return BOOL(0);
    }
    if let Some(info) = window_info(hwnd) {
        state.windows.push(info);
    }
    BOOL(1)
}

/// Enumerate capturable top-level windows: foreground first, then z-order.
pub(crate) fn enumerate_windows() -> Result<Vec<WindowInfo>> {
    let mut state = EnumState {
        windows: Vec::new(),
        truncated: false,
    };
    let lparam = LPARAM(&mut state as *mut EnumState as isize);
    // SAFETY: the callback and the state pointer stay valid for the duration of
    // the call; EnumWindows does not keep either afterwards.
    unsafe { EnumWindows(Some(enum_window_proc), lparam) }
        .map_err(|e| platform("EnumWindows", e))?;
    // EnumWindows returns z-order, which is not always foreground first.
    let foreground = unsafe { GetForegroundWindow() };
    if !foreground.0.is_null()
        && let Some(position) = state
            .windows
            .iter()
            .position(|window| window.id == foreground.0 as usize as u64)
    {
        let window = state.windows.remove(position);
        state.windows.insert(0, window);
    }
    let _ = state.truncated;
    Ok(state.windows)
}

/// Resolve a selector to a live window handle.
fn resolve_window(selector: &WindowSelector) -> Result<HWND> {
    if let Some(id) = selector.id {
        let hwnd = HWND(id as usize as *mut c_void);
        // SAFETY: read-only validity check; a stale id is reported below.
        if unsafe { IsWindow(hwnd) }.as_bool() {
            return Ok(hwnd);
        }
        return Err(FluidError::invalid(format!(
            "window id {id} is not a live window (it may have been closed); enumerate the windows again"
        )));
    }
    let windows = enumerate_windows()?;
    let picked = select_window(&windows, selector)?;
    Ok(HWND(picked.id as usize as *mut c_void))
}

/// The current vertical refresh rate of the monitor a window is on.
fn window_monitor_refresh(hwnd: HWND) -> f32 {
    // SAFETY: plain monitor lookup for a live window handle.
    let monitor = unsafe { MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST) };
    match live_outputs() {
        Ok(outputs) => outputs
            .iter()
            .find(|output| output.entry.monitor == monitor)
            .map(|output| output.entry.refresh_hz)
            .unwrap_or(60.0),
        Err(_) => 60.0,
    }
}

/// Create the shared D3D11 device used by the WGC path.
fn create_shared_device() -> Result<(ID3D11Device, ID3D11DeviceContext, IDirect3DDevice)> {
    let mut levels = [
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    ];
    let mut device: Option<ID3D11Device> = None;
    let mut context: Option<ID3D11DeviceContext> = None;
    // SAFETY: no adapter is passed (the default adapter is used), every out
    // parameter points at a local, and the feature level slice is valid.
    let mut result = unsafe {
        D3D11CreateDevice(
            None::<&IDXGIAdapter>,
            D3D_DRIVER_TYPE_HARDWARE,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            Some(&levels),
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )
    };
    if result.is_err() {
        // Older runtimes reject D3D_FEATURE_LEVEL_11_1 with E_INVALIDARG; retry
        // starting at 11_0, as Microsoft recommends.
        levels[0] = D3D_FEATURE_LEVEL_11_0;
        device = None;
        context = None;
        // SAFETY: same valid inputs as above.
        result = unsafe {
            D3D11CreateDevice(
                None::<&IDXGIAdapter>,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                Some(&levels[1..]),
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
        };
    }
    result.map_err(|e| platform("D3D11CreateDevice", e))?;
    let (Some(device), Some(context)) = (device, context) else {
        return Err(FluidError::platform(
            "window capture: D3D11CreateDevice succeeded but returned no device/context",
        ));
    };
    let dxgi: IDXGIDevice = device
        .cast()
        .map_err(|e| platform("ID3D11Device -> IDXGIDevice", e))?;
    // SAFETY: the DXGI device is a valid interface that outlives the call and the
    // wrapper takes its own reference.
    let inspectable = unsafe { CreateDirect3D11DeviceFromDXGIDevice(&dxgi) }
        .map_err(|e| platform("CreateDirect3D11DeviceFromDXGIDevice", e))?;
    let d3d_device: IDirect3DDevice = inspectable
        .cast()
        .map_err(|e| platform("IInspectable -> IDirect3DDevice", e))?;
    Ok((device, context, d3d_device))
}

/// CPU readable staging texture matching the captured surface descriptor.
fn create_staging_texture(
    device: &ID3D11Device,
    dims: Dims,
    format: DXGI_FORMAT,
) -> Result<ID3D11Texture2D> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: dims.width,
        Height: dims.height,
        MipLevels: 1,
        ArraySize: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_STAGING,
        BindFlags: 0,
        CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
        MiscFlags: 0,
    };
    let mut texture: Option<ID3D11Texture2D> = None;
    // SAFETY: desc is fully initialised, no initial data is supplied and the out
    // parameter points at a local.
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture)) }
        .map_err(|e| platform("ID3D11Device::CreateTexture2D", e))?;
    texture.ok_or_else(|| {
        FluidError::platform("window capture: CreateTexture2D succeeded but returned no texture")
    })
}

/// Windows.Graphics.Capture window source.
struct WgcWindowSource {
    hwnd: HWND,
    /// Fixed output size, decided when the source was opened.
    dims: Dims,
    /// Crop inside the captured content, in content pixels.
    region: Rect,
    fps_cap: f32,
    refresh_hz: f32,
    timeout_ms: u32,
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    d3d_device: IDirect3DDevice,
    item: GraphicsCaptureItem,
    pool: Direct3D11CaptureFramePool,
    session: GraphicsCaptureSession,
    staging: Option<ID3D11Texture2D>,
    staging_dims: Dims,
    /// Last delivered image, replayed when the window produces nothing new.
    cached: Frame,
    index: u64,
    delivered: u64,
    start: Instant,
    closed: bool,
    /// Item size seen last; a change means the window was resized.
    last_item_dims: Dims,
}

// SAFETY: the COM/WinRT objects used here are free-threaded and the source is
// only ever used by whichever thread owns it. FrameSource requires Send so the
// engine can move it to its capture thread before pulling frames.
unsafe impl Send for WgcWindowSource {}

impl Drop for WgcWindowSource {
    fn drop(&mut self) {
        // Stop capture before the pool and the device are released; failures are
        // harmless at this point.
        let _ = self.session.Close();
        let _ = self.pool.Close();
    }
}

impl WgcWindowSource {
    /// Try to start WGC capture for one window. Configuration level problems are
    /// reported as invalid arguments and must not be retried on the fallback.
    fn open(hwnd: HWND, config: &CaptureConfig) -> Result<Self> {
        let (device, context, d3d_device) = create_shared_device()?;
        // SAFETY: activating the well known WGC activation factory; the returned
        // interface is owned by us.
        let factory: IGraphicsCaptureItemInterop =
            windows::core::factory::<GraphicsCaptureItem, IGraphicsCaptureItemInterop>()
                .map_err(|e| platform("Windows.Graphics.Capture activation", e))?;
        // SAFETY: hwnd is a live window handle; WGC reads it through the
        // compositor, no memory of the target process is touched.
        let item: GraphicsCaptureItem =
            unsafe { factory.CreateForWindow::<HWND, GraphicsCaptureItem>(hwnd) }
                .map_err(|e| platform("IGraphicsCaptureItemInterop::CreateForWindow", e))?;
        let size = item
            .Size()
            .map_err(|e| platform("GraphicsCaptureItem::Size", e))?;
        let content = Dims::new(size.Width.max(0) as u32, size.Height.max(0) as u32);
        let (region, dims) = capture_region(content, config.region.map(Rect::from))?;
        let pool = Direct3D11CaptureFramePool::CreateFreeThreaded(
            &d3d_device,
            DirectXPixelFormat::B8G8R8A8UIntNormalized,
            2,
            size,
        )
        .map_err(|e| platform("Direct3D11CaptureFramePool::CreateFreeThreaded", e))?;
        let session = pool
            .CreateCaptureSession(&item)
            .map_err(|e| platform("Direct3D11CaptureFramePool::CreateCaptureSession", e))?;
        // WGC composites the pointer itself; newer builds may refuse the border
        // toggle, which only means the yellow capture border stays visible.
        let _ = session.SetIsCursorCaptureEnabled(config.include_cursor);
        let _ = session.SetIsBorderRequired(false);
        session
            .StartCapture()
            .map_err(|e| platform("GraphicsCaptureSession::StartCapture", e))?;
        Ok(Self {
            hwnd,
            dims,
            region,
            fps_cap: config.max_fps,
            refresh_hz: window_monitor_refresh(hwnd),
            timeout_ms: config.timeout_ms,
            device,
            context,
            d3d_device,
            item,
            pool,
            session,
            staging: None,
            staging_dims: Dims::new(0, 0),
            cached: blank_frame(dims),
            index: 0,
            delivered: 0,
            start: Instant::now(),
            closed: false,
            last_item_dims: content,
        })
    }

    /// Rate limiter: makes frame n wait until start + n / fps.
    fn pace(&self) {
        if self.fps_cap <= 0.0 || self.fps_cap.is_nan() {
            return;
        }
        let period = Duration::from_secs_f64(1.0 / self.fps_cap as f64);
        let deadline = self.start + period.mul_f64(self.delivered as f64);
        sleep_until(deadline);
    }

    fn window_alive(&self) -> bool {
        // SAFETY: read-only validity check; stale handles return false.
        unsafe { IsWindow(self.hwnd) }.as_bool()
    }

    fn minimized(&self) -> bool {
        // SAFETY: read-only window state query.
        unsafe { IsIconic(self.hwnd) }.as_bool()
    }

    /// Recreate the frame pool when the captured window changed size.
    fn adapt_to_size(&mut self) {
        let Ok(size) = self.item.Size() else {
            return;
        };
        let dims = Dims::new(size.Width.max(0) as u32, size.Height.max(0) as u32);
        if dims.is_empty() || dims == self.last_item_dims {
            return;
        }
        self.last_item_dims = dims;
        // A frame pool sized for the old content stops delivering frames after a
        // resize, so recreate it with the new item size (best effort).
        let _ = self.pool.Recreate(
            &self.d3d_device,
            DirectXPixelFormat::B8G8R8A8UIntNormalized,
            2,
            size,
        );
        self.staging = None;
        self.staging_dims = Dims::new(0, 0);
    }

    /// Pull the next frame within the timeout budget.
    ///
    /// Returns "Ok(None)" when no frame arrived; the caller replays the cached
    /// frame then (a live source never ends).
    fn wait_for_frame(&mut self) -> Result<Option<Frame>> {
        let deadline = Instant::now() + Duration::from_millis(self.timeout_ms as u64);
        loop {
            match self.pool.TryGetNextFrame() {
                Ok(frame) => return self.convert_frame(&frame),
                Err(_) => {
                    if Instant::now() >= deadline {
                        return Ok(None);
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
    }

    /// Copy one WGC frame into the fixed output frame.
    ///
    /// Returns "Ok(None)" when the frame has no usable content (minimized).
    fn convert_frame(&mut self, frame: &Direct3D11CaptureFrame) -> Result<Option<Frame>> {
        let size = frame
            .ContentSize()
            .map_err(|e| platform("Direct3D11CaptureFrame::ContentSize", e))?;
        if size.Width <= 0 || size.Height <= 0 {
            return Ok(None);
        }
        let surface = frame
            .Surface()
            .map_err(|e| platform("Direct3D11CaptureFrame::Surface", e))?;
        let access: IDirect3DDxgiInterfaceAccess = surface
            .cast()
            .map_err(|e| platform("Direct3DSurface -> IDirect3DDxgiInterfaceAccess", e))?;
        // SAFETY: the surface is D3D11 backed; the returned texture reference is
        // owned by us and keeps the image valid after the frame is closed.
        let texture: ID3D11Texture2D = unsafe { access.GetInterface() }
            .map_err(|e| platform("IDirect3DDxgiInterfaceAccess::GetInterface", e))?;
        // Release the WinRT frame before doing CPU work.
        let _ = frame.Close();
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: texture is valid and desc is a local out parameter.
        unsafe { texture.GetDesc(&mut desc) };
        let content = Dims::new(desc.Width.max(1), desc.Height.max(1));
        if self.staging_dims != content {
            self.staging = Some(create_staging_texture(&self.device, content, desc.Format)?);
            self.staging_dims = content;
        }
        let Some(staging) = self.staging.as_ref() else {
            return Err(FluidError::platform(
                "window capture: staging texture is not initialised",
            ));
        };
        // SAFETY: staging matches the captured texture in size and format and is
        // not bound to the pipeline.
        unsafe { self.context.CopyResource(staging, &texture) };
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        // SAFETY: staging is a staging texture with CPU read access; mapped is a
        // local out parameter.
        unsafe {
            self.context
                .Map(staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
        }
        .map_err(|e| platform("ID3D11DeviceContext::Map", e))?;
        let mut output = Frame {
            dims: self.dims,
            fmt: PixelFormat::Rgba8,
            data: vec![0u8; self.dims.area() as usize * 4],
            pts_ns: now_ns(),
            index: 0,
        };
        let region = self.region;
        let dst_row_bytes = self.dims.width as usize * 4;
        // Content that no longer fits is cropped; missing content stays black.
        let copy_w = region
            .w
            .min(content.width.saturating_sub(region.x))
            .min(self.dims.width) as usize;
        let copy_h = region
            .h
            .min(content.height.saturating_sub(region.y))
            .min(self.dims.height) as usize;
        if copy_w > 0 && copy_h > 0 {
            // SAFETY: Map succeeded, so pData points at RowPitch * height bytes of
            // readable staging memory valid until Unmap; every source row read is
            // inside the staging texture and every destination row is inside the
            // output buffer.
            unsafe {
                let base = mapped.pData as *const u8;
                let pitch = mapped.RowPitch as usize;
                for y in 0..copy_h {
                    let src_offset = (region.y as usize + y) * pitch + region.x as usize * 4;
                    let src = std::slice::from_raw_parts(base.add(src_offset), copy_w * 4);
                    let dst_start = y * dst_row_bytes;
                    let dst = &mut output.data[dst_start..dst_start + copy_w * 4];
                    bgra_row_to_rgba(src, dst);
                }
            }
        }
        // SAFETY: staging was mapped above and is unmapped exactly once.
        unsafe { self.context.Unmap(staging, 0) };
        Ok(Some(output))
    }

    /// Give the frame its sequence number and timestamp.
    fn stamp(&mut self, mut frame: Frame) -> Frame {
        frame.pts_ns = now_ns();
        frame.index = self.index;
        self.index += 1;
        self.delivered += 1;
        frame
    }

    /// Re-emit the previous frame (minimized, closed or nothing new).
    fn emit_cached(&mut self) -> Frame {
        let frame = replay(&self.cached, self.dims, now_ns(), self.index);
        self.index += 1;
        self.delivered += 1;
        frame
    }
}

impl FrameSource for WgcWindowSource {
    fn dims(&self) -> Dims {
        self.dims
    }

    fn fps(&self) -> f32 {
        if self.fps_cap > 0.0 {
            self.fps_cap
        } else {
            self.refresh_hz
        }
    }

    fn backend_name(&self) -> &'static str {
        "window-wgc"
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        if self.closed {
            return Ok(None);
        }
        self.pace();
        if !self.window_alive() || self.minimized() {
            return Ok(Some(self.emit_cached()));
        }
        self.adapt_to_size();
        match self.wait_for_frame()? {
            Some(frame) => {
                let frame = self.stamp(frame);
                self.cached = frame.clone();
                Ok(Some(frame))
            }
            None => Ok(Some(self.emit_cached())),
        }
    }

    fn close(&mut self) {
        self.closed = true;
    }
}

/// DXGI desktop duplication window source (fallback when WGC is unavailable).
struct DxgiWindowSource {
    hwnd: HWND,
    /// Fixed output size, decided when the source was opened.
    dims: Dims,
    /// Crop inside the window frame rectangle, in window local pixels.
    region: Rect,
    /// Desktop rectangle of the monitor currently holding the window.
    screen: PixelRect,
    hmonitor: Option<HMONITOR>,
    refresh_hz: f32,
    fps_cap: f32,
    timeout_ms: u32,
    device: Option<ID3D11Device>,
    context: Option<ID3D11DeviceContext>,
    duplication: Option<IDXGIOutputDuplication>,
    staging: Option<ID3D11Texture2D>,
    /// Desktop pixel format and matching staging texture format.
    layout: Option<DesktopLayout>,
    cached: Frame,
    index: u64,
    delivered: u64,
    start: Instant,
    closed: bool,
}

// SAFETY: same reasoning as the DXGI monitor source: D3D11 objects are
// free-threaded, the duplication object is used by one thread at a time, and the
// engine moves the source to its capture thread before pulling frames.
unsafe impl Send for DxgiWindowSource {}

impl DxgiWindowSource {
    fn open(hwnd: HWND, config: &CaptureConfig) -> Result<Self> {
        let rect = window_frame_rect(hwnd).ok_or_else(|| {
            FluidError::invalid(
                "window has no frame rectangle (restore or un-minimize the window before capturing it)",
            )
        })?;
        let frame = pixel_rect(rect);
        // The fallback approximates the window content by its frame rectangle, so
        // the configured region is relative to the frame (title bar included).
        let (region, dims) = capture_region(frame.dims(), config.region.map(Rect::from))?;
        let mut source = Self {
            hwnd,
            dims,
            region,
            screen: PixelRect::new(0, 0, 0, 0),
            hmonitor: None,
            refresh_hz: 60.0,
            fps_cap: config.max_fps,
            timeout_ms: config.timeout_ms,
            device: None,
            context: None,
            duplication: None,
            staging: None,
            layout: None,
            cached: blank_frame(dims),
            index: 0,
            delivered: 0,
            start: Instant::now(),
            closed: false,
        };
        source.rebuild()?;
        Ok(source)
    }

    /// Rate limiter, same policy as the monitor source.
    fn pace(&self) {
        if self.fps_cap <= 0.0 || self.fps_cap.is_nan() {
            return;
        }
        let period = Duration::from_secs_f64(1.0 / self.fps_cap as f64);
        let deadline = self.start + period.mul_f64(self.delivered as f64);
        sleep_until(deadline);
    }

    fn window_alive(&self) -> bool {
        // SAFETY: read-only validity check; stale handles return false.
        unsafe { IsWindow(self.hwnd) }.as_bool()
    }

    fn minimized(&self) -> bool {
        // SAFETY: read-only window state query.
        unsafe { IsIconic(self.hwnd) }.as_bool()
    }

    /// (Re)acquire desktop duplication for the monitor under the window.
    fn rebuild(&mut self) -> Result<()> {
        let outputs = live_outputs()?;
        if outputs.is_empty() {
            return Err(FluidError::platform(
                "window capture: no attached desktop output found",
            ));
        }
        // SAFETY: plain monitor lookup for the live window handle.
        let hmonitor = unsafe { MonitorFromWindow(self.hwnd, MONITOR_DEFAULTTONEAREST) };
        let frame = window_frame_rect(self.hwnd);
        let index = outputs
            .iter()
            .position(|output| output.entry.monitor == hmonitor)
            .or_else(|| {
                frame.and_then(|rect| {
                    outputs.iter().position(|output| {
                        output.entry.rect.left <= rect.left
                            && rect.right <= output.entry.rect.right
                            && output.entry.rect.top <= rect.top
                            && rect.bottom <= output.entry.rect.bottom
                    })
                })
            })
            .unwrap_or(0);
        let output = &outputs[index];
        let screen = pixel_rect(output.entry.rect);
        if screen.is_empty() {
            return Err(FluidError::platform(
                "window capture: the monitor under the window reported a zero sized rectangle",
            ));
        }
        let (device, context) = create_device(&output.adapter)?;
        let duplication = duplicate_output(&device, &output.output)?;
        // The desktop may be HDR/10-bit; the staging texture must use the same
        // format as the desktop image for "CopyResource" to accept it.
        // SAFETY: the duplication object was just created and is owned here;
        // GetDesc only reads its description.
        let duplicated = unsafe { duplication.GetDesc() };
        let layout = desktop_layout(duplicated.ModeDesc.Format)?;
        let staging = create_staging_with_format(&device, screen.dims(), layout.dxgi)?;
        self.screen = screen;
        self.hmonitor = Some(hmonitor);
        self.refresh_hz = output.entry.refresh_hz;
        self.device = Some(device);
        self.context = Some(context);
        self.duplication = Some(duplication);
        self.staging = Some(staging);
        self.layout = Some(layout);
        Ok(())
    }

    /// Copy the window rectangle out of the last desktop image.
    fn read_window(&mut self, texture: &ID3D11Texture2D, window: PixelRect) -> Result<Frame> {
        let dims = self.dims;
        let region = self.region;
        let screen = self.screen;
        let (Some(staging), Some(context), Some(layout)) = (
            self.staging.as_ref(),
            self.context.as_ref(),
            self.layout.as_ref(),
        ) else {
            return Err(FluidError::platform(
                "window capture: DXGI crop pipeline is not initialised",
            ));
        };
        let source_format = layout.format;
        let source_bytes_per_pixel = source_format.bytes_per_pixel();
        let mut output = Frame {
            dims,
            fmt: PixelFormat::Rgba8,
            data: vec![0u8; dims.area() as usize * 4],
            pts_ns: now_ns(),
            index: 0,
        };
        // SAFETY: both textures are valid, the staging texture was created with
        // the monitor size and format, and it is not bound to the pipeline.
        unsafe { context.CopyResource(staging, texture) };
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        // SAFETY: the staging texture has CPU read access and mapped is a local.
        unsafe { context.Map(staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped)) }
            .map_err(|e| platform("ID3D11DeviceContext::Map", e))?;
        let dst_row_bytes = dims.width as usize * 4;
        // Destination rectangle in desktop pixels (window frame origin plus the
        // crop offset) and the part of it that is actually on the monitor. A
        // window can hang off any edge of its monitor, including negative desktop
        // coordinates on a left hand secondary monitor, so both axes are clipped
        // and the parts outside the monitor stay black.
        let dst = screen_rect(window, region);
        let visible = dst.intersect(&screen);
        let copy_w = visible.width as usize;
        let copy_h = visible.height as usize;
        let mut convert_error = None;
        if copy_w > 0 && copy_h > 0 {
            let dst_offset = (visible.left - dst.left) as usize;
            let first_row = (visible.top - dst.top) as usize;
            let src_x = (visible.left - screen.left) as usize;
            let src_y = (visible.top - screen.top) as usize;
            // SAFETY: Map succeeded, so pData points at RowPitch * height bytes of
            // readable staging memory valid until Unmap. Only the visible sub
            // rectangle of the staging texture is read; rows hold at least
            // copy_w * source_bytes_per_pixel bytes from src_x. Failures are
            // recorded and returned after Unmap.
            unsafe {
                let base = mapped.pData as *const u8;
                let pitch = mapped.RowPitch as usize;
                for y in 0..copy_h {
                    let src_offset = (src_y + y) * pitch + src_x * source_bytes_per_pixel;
                    let src = std::slice::from_raw_parts(
                        base.add(src_offset),
                        copy_w * source_bytes_per_pixel,
                    );
                    let dst_start = (first_row + y) * dst_row_bytes + dst_offset * 4;
                    let dst = &mut output.data[dst_start..dst_start + copy_w * 4];
                    if let Err(err) = convert_row(src, dst, copy_w, source_format) {
                        convert_error = Some(err);
                        break;
                    }
                }
            }
        }
        // SAFETY: the staging texture was mapped above and is unmapped once.
        unsafe { context.Unmap(staging, 0) };
        if let Some(err) = convert_error {
            return Err(err);
        }
        Ok(output)
    }

    fn stamp(&mut self, mut frame: Frame) -> Frame {
        frame.pts_ns = now_ns();
        frame.index = self.index;
        self.index += 1;
        self.delivered += 1;
        frame
    }

    fn emit_cached(&mut self) -> Frame {
        let frame = replay(&self.cached, self.dims, now_ns(), self.index);
        self.index += 1;
        self.delivered += 1;
        frame
    }

    fn release_frame(&self) {
        if let Some(duplication) = self.duplication.as_ref() {
            // SAFETY: only called after a successful AcquireNextFrame, one release
            // per acquired frame.
            let _ = unsafe { duplication.ReleaseFrame() };
        }
    }
}

impl FrameSource for DxgiWindowSource {
    fn dims(&self) -> Dims {
        self.dims
    }

    fn fps(&self) -> f32 {
        if self.fps_cap > 0.0 {
            self.fps_cap
        } else {
            self.refresh_hz
        }
    }

    fn backend_name(&self) -> &'static str {
        "window-dxgi-crop"
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        if self.closed {
            return Ok(None);
        }
        self.pace();
        if !self.window_alive() || self.minimized() {
            return Ok(Some(self.emit_cached()));
        }
        let Some(rect) = window_frame_rect(self.hwnd) else {
            return Ok(Some(self.emit_cached()));
        };
        let window = pixel_rect(rect);
        // A window can be dragged to another monitor; duplication is per output,
        // so rebuild for the new monitor. The frame origin is re-read every frame,
        // which also follows plain movement within one monitor.
        // SAFETY: plain monitor lookup for the live window handle.
        let hmonitor = unsafe { MonitorFromWindow(self.hwnd, MONITOR_DEFAULTTONEAREST) };
        if self.hmonitor != Some(hmonitor) {
            self.rebuild()?;
        }
        enum Step {
            Acquired(Option<IDXGIResource>),
            Timeout,
            Retry,
        }
        let mut attempt = 0u32;
        loop {
            if self.duplication.is_none() {
                self.rebuild()?;
            }
            let step = {
                let Some(duplication) = self.duplication.as_ref() else {
                    return Err(FluidError::platform(
                        "window capture: DXGI crop pipeline is not initialised",
                    ));
                };
                let mut info = windows::Win32::Graphics::Dxgi::DXGI_OUTDUPL_FRAME_INFO::default();
                let mut resource: Option<IDXGIResource> = None;
                // SAFETY: duplication is a valid duplication object and both out
                // parameters point at locals.
                match unsafe {
                    duplication.AcquireNextFrame(self.timeout_ms, &mut info, &mut resource)
                } {
                    Ok(()) => Step::Acquired(resource),
                    Err(e) if e.code() == DXGI_ERROR_WAIT_TIMEOUT => Step::Timeout,
                    Err(e)
                        if e.code() == DXGI_ERROR_ACCESS_LOST
                            || e.code() == DXGI_ERROR_DEVICE_REMOVED =>
                    {
                        if attempt == 0 {
                            Step::Retry
                        } else {
                            return Err(platform("IDXGIOutputDuplication::AcquireNextFrame", e));
                        }
                    }
                    Err(e) => return Err(platform("IDXGIOutputDuplication::AcquireNextFrame", e)),
                }
            };
            let resource = match step {
                Step::Timeout => return Ok(Some(self.emit_cached())),
                Step::Retry => {
                    attempt = 1;
                    self.duplication = None;
                    self.staging = None;
                    self.device = None;
                    self.context = None;
                    continue;
                }
                Step::Acquired(resource) => resource,
            };
            let texture = match resource.map(|resource| resource.cast::<ID3D11Texture2D>()) {
                Some(Ok(texture)) => Some(texture),
                Some(Err(e)) => {
                    self.release_frame();
                    return Err(platform("IDXGIResource -> ID3D11Texture2D", e));
                }
                None => None,
            };
            let Some(texture) = texture else {
                self.release_frame();
                return Ok(Some(self.emit_cached()));
            };
            let read = self.read_window(&texture, window);
            self.release_frame();
            let frame = self.stamp(read?);
            self.cached = frame.clone();
            return Ok(Some(frame));
        }
    }

    fn close(&mut self) {
        self.closed = true;
        self.duplication = None;
        self.staging = None;
        self.device = None;
        self.context = None;
    }
}

/// Open a window capture source, preferring Windows.Graphics.Capture.
pub(crate) fn open(
    selector: &WindowSelector,
    config: &CaptureConfig,
) -> Result<Box<dyn FrameSource>> {
    config.validate()?;
    // WGC activation is WinRT; COM must be initialized on the calling thread.
    // RPC_E_CHANGED_MODE just means COM already exists with another model.
    // SAFETY: CoInitializeEx only initializes per-thread COM state and the result
    // is deliberately ignored.
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
    }
    let hwnd = resolve_window(selector)?;
    match WgcWindowSource::open(hwnd, config) {
        Ok(source) => Ok(Box::new(source)),
        Err(FluidError::InvalidArgument(message)) => Err(FluidError::InvalidArgument(message)),
        Err(wgc_error) => match DxgiWindowSource::open(hwnd, config) {
            Ok(source) => Ok(Box::new(source)),
            Err(fallback_error @ FluidError::InvalidArgument(_)) => Err(fallback_error),
            Err(fallback_error) => Err(FluidError::platform(format!(
                "window capture failed with Windows.Graphics.Capture ({wgc_error}) and the DXGI crop fallback ({fallback_error})"
            ))),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_rect_handles_negative_origins() {
        let rect = RECT {
            left: -1920,
            top: 120,
            right: -320,
            bottom: 840,
        };
        let pixel = pixel_rect(rect);
        assert_eq!(pixel.left, -1920);
        assert_eq!(pixel.top, 120);
        assert_eq!(pixel.width, 1600);
        assert_eq!(pixel.height, 720);
        assert_eq!(pixel.dims(), Dims::new(1600, 720));
        // Degenerate rectangles never produce a negative width.
        let empty = pixel_rect(RECT {
            left: 10,
            top: 10,
            right: 10,
            bottom: 4,
        });
        assert!(empty.is_empty());
    }

    #[test]
    fn process_file_name_strips_directories() {
        assert_eq!(
            process_file_name("C:\\Windows\\System32\\notepad.exe"),
            "notepad.exe"
        );
        assert_eq!(process_file_name("notepad.exe"), "notepad.exe");
        assert_eq!(process_file_name(""), "");
    }

    #[test]
    fn desktop_crop_is_clipped_to_the_monitor() {
        // Window frame starts left of the monitor: the first 100 columns are off
        // screen and the remaining 220 columns are copied.
        let frame = PixelRect::new(-100, 120, 320, 240);
        let dst = screen_rect(frame, Rect::from_dims(frame.dims()));
        assert_eq!(dst, PixelRect::new(-100, 120, 320, 240));
        let screen = PixelRect::new(0, 0, 1920, 1080);
        assert_eq!(dst.intersect(&screen), PixelRect::new(0, 120, 220, 240));
        // Secondary monitor placed left of the primary one: negative desktop
        // coordinates are clipped on the left edge.
        let screen = PixelRect::new(-1920, 0, 1920, 1080);
        let frame = PixelRect::new(-2000, 100, 320, 240);
        let visible = screen_rect(frame, Rect::from_dims(frame.dims())).intersect(&screen);
        assert_eq!(visible, PixelRect::new(-1920, 100, 240, 240));
        // A window fully outside its monitor contributes no pixels.
        let screen = PixelRect::new(0, 0, 1920, 1080);
        let frame = PixelRect::new(4000, 0, 320, 240);
        assert!(
            screen_rect(frame, Rect::from_dims(frame.dims()))
                .intersect(&screen)
                .is_empty()
        );
        // A region is an offset inside the window frame.
        let frame = PixelRect::new(100, 50, 640, 480);
        let region = Rect::new(10, 20, 32, 16);
        assert_eq!(screen_rect(frame, region), PixelRect::new(110, 70, 32, 16));
    }
}
