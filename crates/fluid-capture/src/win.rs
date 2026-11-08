//! DXGI Desktop Duplication capture backend (Windows 8+).
//!
//! Pipeline per frame:
//!
//! 1. `IDXGIOutputDuplication::AcquireNextFrame(timeout_ms)` blocks until the
//!    desktop compositor produces a new image (or the timeout expires).
//! 2. The returned `IDXGIResource` is copied into a CPU readable staging
//!    `ID3D11Texture2D` with `CopyResource` (whole monitor) or
//!    `CopySubresourceRegion` (configured region).
//! 3. `Map` gives a pointer plus the real row pitch; every row is converted
//!    BGRA -> RGBA into an owned `Vec<u8>` (`PixelFormat::Rgba8`, the format
//!    this backend always produces).
//! 4. The optional mouse pointer is composited in software from the pointer
//!    shape DXGI reports (see `cursor.rs`); the desktop image itself never
//!    contains the cursor.
//! 5. `ReleaseFrame` is called before the frame is handed to the caller.
//!
//! Failure handling: `DXGI_ERROR_WAIT_TIMEOUT` (no desktop update within
//! `timeout_ms`) returns a clone of the previous frame instead of ending the
//! stream - a live capture source never returns `Ok(None)`. Access loss
//! (Ctrl+Alt+Del, driver reset, mode change) tears the device and duplication
//! object down and recreates them once; if the geometry changed the call fails
//! with a message telling the caller to reopen capture.
//!
//! Geometry: outputs are ordered by desktop position (left, top) so that
//! `CaptureConfig::monitor` matches `enumerate_monitors()` order and the
//! present backend's monitor order. An out of range index is clamped to the
//! last output. `CaptureConfig::region` is relative to the selected monitor;
//! it is clipped to the monitor, and `FrameSource::dims` reports the clipped
//! region size.
//!
//! Threading: `open` creates the D3D11 device and the duplication object and
//! the source must then be used from one thread at a time. `FrameSource`
//! requires `Send`, so the type declares `Send`; the engine may create the
//! source on its main thread and move it to the capture thread before the
//! first `next_frame` call. Calls are blocking (bounded by `timeout_ms`).

use std::ffi::c_void;
use std::mem::size_of;
use std::time::{Duration, Instant};

use fluid_core::{
    CaptureConfig, Dims, FluidError, Frame, FrameSource, PixelFormat, Rect, Result, now_ns,
    sleep_until,
};

use windows::Win32::Foundation::{HMODULE, POINT, RECT};
use windows::Win32::Graphics::Direct3D::{
    D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL_10_0, D3D_FEATURE_LEVEL_10_1,
    D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1,
};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BOX, D3D11_CPU_ACCESS_READ, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_MAP_READ,
    D3D11_MAPPED_SUBRESOURCE, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING,
    D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM,
    DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_ERROR_ACCESS_LOST, DXGI_ERROR_DEVICE_REMOVED,
    DXGI_ERROR_NOT_CURRENTLY_AVAILABLE, DXGI_ERROR_NOT_FOUND, DXGI_ERROR_WAIT_TIMEOUT,
    DXGI_OUTDUPL_FRAME_INFO, DXGI_OUTDUPL_POINTER_SHAPE_INFO, DXGI_OUTPUT_DESC, IDXGIAdapter,
    IDXGIAdapter1, IDXGIFactory1, IDXGIOutput1, IDXGIOutputDuplication, IDXGIResource,
};
use windows::Win32::Graphics::Gdi::{
    DEVMODEW, ENUM_CURRENT_SETTINGS, EnumDisplaySettingsW, GetMonitorInfoW, HMONITOR,
    MONITOR_DEFAULTTOPRIMARY, MONITORINFO, MonitorFromPoint,
};
use windows::core::{Interface, PCWSTR};

use crate::MonitorInfo;
use crate::cursor::{CursorKind, CursorShape, composite_cursor};
use crate::format::{DesktopFormat, convert_row, format_label};

/// Wrap a failed COM call with the call site, keeping it a platform error.
fn platform(what: &str, err: windows::core::Error) -> FluidError {
    FluidError::platform(format!("dxgi: {what}: {err}"))
}

/// The desktop format desktop duplication reports plus the matching DXGI format
/// used for the CPU staging texture. The two must agree, otherwise
/// "CopyResource" rejects the copy (E_INVALIDARG) because the formats differ.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DesktopLayout {
    pub(crate) format: DesktopFormat,
    pub(crate) dxgi: DXGI_FORMAT,
}

/// Negotiate the desktop duplication source format.
///
/// HDR and 10-bit desktops commonly use R16G16B16A16_FLOAT or R10G10B10A2; both
/// are converted to 8-bit RGBA downstream. Anything else is reported as an
/// unsupported format naming the DXGI value instead of failing later with an
/// unhelpful "CopyResource" or map error.
pub(crate) fn desktop_layout(dxgi: DXGI_FORMAT) -> Result<DesktopLayout> {
    let format = match dxgi {
        DXGI_FORMAT_B8G8R8A8_UNORM => DesktopFormat::Bgra8,
        DXGI_FORMAT_R8G8B8A8_UNORM => DesktopFormat::Rgba8,
        DXGI_FORMAT_R10G10B10A2_UNORM => DesktopFormat::Rgb10a2,
        DXGI_FORMAT_R16G16B16A16_FLOAT => DesktopFormat::Rgba16F,
        other => {
            return Err(FluidError::unsupported(format!(
                "dxgi: desktop duplication returned {} for the desktop image; Fluid captures \
                 B8G8R8A8_UNORM, R8G8B8A8_UNORM, R10G10B10A2_UNORM and R16G16B16A16_FLOAT only",
                format_label(other.0 as u32)
            )));
        }
    };
    Ok(DesktopLayout { format, dxgi })
}

/// A DXGI output together with where it sits on the virtual desktop.
#[derive(Clone, Debug)]
pub(crate) struct OutputEntry {
    /// Desktop coordinates of the output.
    pub(crate) rect: RECT,
    /// Human readable monitor name, e.g. "DISPLAY1".
    pub(crate) name: String,
    pub(crate) refresh_hz: f32,
    pub(crate) primary: bool,
    /// The Win32 monitor handle, used by the window capture fallback.
    pub(crate) monitor: HMONITOR,
}

/// A live COM reference to an output (kept so `open` can duplicate it).
pub(crate) struct LiveOutput {
    pub(crate) adapter: IDXGIAdapter1,
    pub(crate) output: IDXGIOutput1,
    pub(crate) entry: OutputEntry,
}

pub(crate) fn screen_dims(entry: &OutputEntry) -> Dims {
    Dims::new(
        (entry.rect.right - entry.rect.left).max(0) as u32,
        (entry.rect.bottom - entry.rect.top).max(0) as u32,
    )
}

/// Convert a DXGI device name ("\\\\.\\DISPLAY1") into a short monitor name.
fn output_name(raw: &[u16; 32], fallback_index: u32) -> String {
    let len = raw.iter().position(|&c| c == 0).unwrap_or(raw.len());
    let text = String::from_utf16_lossy(&raw[..len]);
    let trimmed = text.trim_start_matches("\\\\.\\").trim();
    if trimmed.is_empty() {
        format!("output-{fallback_index}")
    } else {
        trimmed.to_string()
    }
}

/// Query the current vertical refresh rate for an output device, if known.
///
/// `dmDisplayFrequency` may report 0 or 1 for "hardware default"; those are
/// filtered out so the caller can fall back to 60 Hz.
fn query_refresh_hz(device_name: &[u16; 32]) -> Option<f32> {
    let mut mode = DEVMODEW {
        dmSize: size_of::<DEVMODEW>() as u16,
        ..Default::default()
    };
    // SAFETY: `device_name` is the NUL padded name from DXGI_OUTPUT_DESC and
    // stays alive for the call; `mode` is a fully initialised DEVMODEW whose
    // dmSize field advertises its size as the API requires.
    let ok = unsafe {
        EnumDisplaySettingsW(
            PCWSTR(device_name.as_ptr()),
            ENUM_CURRENT_SETTINGS,
            &mut mode,
        )
        .as_bool()
    };
    if !ok {
        return None;
    }
    let hz = mode.dmDisplayFrequency as f32;
    if (1.0..=1000.0).contains(&hz) {
        Some(hz)
    } else {
        None
    }
}

/// The desktop rectangle of the primary monitor, used to flag it.
fn primary_monitor_rect() -> Option<RECT> {
    // SAFETY: MonitorFromPoint and GetMonitorInfoW take plain handles plus a
    // pointer to a MONITORINFO that we fully initialise before the call.
    unsafe {
        let monitor: HMONITOR = MonitorFromPoint(POINT { x: 0, y: 0 }, MONITOR_DEFAULTTOPRIMARY);
        let mut info = MONITORINFO {
            cbSize: size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        if GetMonitorInfoW(monitor, &mut info).as_bool() {
            Some(info.rcMonitor)
        } else {
            None
        }
    }
}

/// Enumerate attached outputs, sorted by desktop position.
pub(crate) fn live_outputs() -> Result<Vec<LiveOutput>> {
    // SAFETY: no parameters; returns a new factory we own.
    let factory: IDXGIFactory1 =
        unsafe { CreateDXGIFactory1() }.map_err(|e| platform("CreateDXGIFactory1", e))?;
    let primary = primary_monitor_rect();
    let mut outputs: Vec<LiveOutput> = Vec::new();
    let mut adapter_index = 0u32;
    loop {
        // SAFETY: `factory` is a valid IDXGIFactory1 for the whole function.
        let adapter = match unsafe { factory.EnumAdapters1(adapter_index) } {
            Ok(adapter) => adapter,
            Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => break,
            Err(e) => return Err(platform("IDXGIFactory1::EnumAdapters1", e)),
        };
        let adapter_parent: IDXGIAdapter = adapter
            .cast()
            .map_err(|e| platform("IDXGIAdapter1 -> IDXGIAdapter", e))?;
        let mut output_index = 0u32;
        loop {
            // SAFETY: `adapter_parent` is a valid IDXGIAdapter alive here.
            let output = match unsafe { adapter_parent.EnumOutputs(output_index) } {
                Ok(output) => output,
                Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => break,
                Err(e) => return Err(platform("IDXGIAdapter::EnumOutputs", e)),
            };
            // SAFETY: `output` is a valid IDXGIOutput alive here.
            let desc: DXGI_OUTPUT_DESC =
                unsafe { output.GetDesc() }.map_err(|e| platform("IDXGIOutput::GetDesc", e))?;
            let rect = desc.DesktopCoordinates;
            let width = (rect.right - rect.left).max(0) as u32;
            let height = (rect.bottom - rect.top).max(0) as u32;
            if !desc.AttachedToDesktop.as_bool() || width == 0 || height == 0 {
                output_index += 1;
                continue;
            }
            if let Ok(output) = output.cast::<IDXGIOutput1>() {
                let name = output_name(&desc.DeviceName, output_index);
                let refresh_hz = query_refresh_hz(&desc.DeviceName).unwrap_or(60.0);
                let primary = primary
                    .map(|p| {
                        p.left == rect.left
                            && p.top == rect.top
                            && p.right == rect.right
                            && p.bottom == rect.bottom
                    })
                    .unwrap_or(false);
                outputs.push(LiveOutput {
                    adapter: adapter.clone(),
                    output,
                    entry: OutputEntry {
                        rect,
                        name,
                        refresh_hz,
                        primary,
                        monitor: desc.Monitor,
                    },
                });
            }
            output_index += 1;
        }
        adapter_index += 1;
    }
    // Order by desktop position so monitor indices are stable and match the
    // presentation backend.
    outputs.sort_by_key(|o| (o.entry.rect.left, o.entry.rect.top));
    Ok(outputs)
}

/// Create a hardware D3D11 device on `adapter`.
///
/// D3D11 requires `D3D_DRIVER_TYPE_UNKNOWN` whenever an explicit adapter is
/// passed; `D3D_DRIVER_TYPE_HARDWARE` with an adapter fails with E_INVALIDARG
/// on some drivers. `UNKNOWN` means "use the adapter you were given".
pub(crate) fn create_device(
    adapter: &IDXGIAdapter1,
) -> Result<(ID3D11Device, ID3D11DeviceContext)> {
    // D3D11CreateDevice wants the base IDXGIAdapter interface.
    let adapter: IDXGIAdapter = adapter
        .cast()
        .map_err(|e| platform("IDXGIAdapter1 -> IDXGIAdapter", e))?;
    let mut levels = [
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    ];
    let mut device: Option<ID3D11Device> = None;
    let mut context: Option<ID3D11DeviceContext> = None;
    // SAFETY: `adapter` is a valid DXGI adapter that outlives the call; every
    // out parameter points at a local that starts out None.
    let mut result = unsafe {
        D3D11CreateDevice(
            Some(&adapter),
            D3D_DRIVER_TYPE_UNKNOWN,
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
        // from 11_0 down, as recommended by Microsoft.
        levels[0] = D3D_FEATURE_LEVEL_11_0;
        device = None;
        context = None;
        // SAFETY: same valid inputs as above.
        result = unsafe {
            D3D11CreateDevice(
                Some(&adapter),
                D3D_DRIVER_TYPE_UNKNOWN,
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
    match (device, context) {
        (Some(device), Some(context)) => Ok((device, context)),
        _ => Err(FluidError::platform(
            "dxgi: D3D11CreateDevice succeeded but returned no device/context",
        )),
    }
}

/// Open desktop duplication for `output`.
pub(crate) fn duplicate_output(
    device: &ID3D11Device,
    output: &IDXGIOutput1,
) -> Result<IDXGIOutputDuplication> {
    // SAFETY: `device` and `output` are valid interfaces that outlive the call.
    unsafe { output.DuplicateOutput(device) }.map_err(|e| {
        if e.code() == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE {
            FluidError::platform(
                "dxgi: DuplicateOutput is not currently available (too many concurrent \
                 duplication clients, or the desktop is not ready); close other capture apps",
            )
        } else {
            platform("IDXGIOutput1::DuplicateOutput", e)
        }
    })
}

/// Create the CPU readable staging texture every frame is copied through.
///
/// The texture must use the same DXGI format as the desktop image, so the
/// format is negotiated per desktop (see desktop_layout).
pub(crate) fn create_staging_with_format(
    device: &ID3D11Device,
    dims: Dims,
    dxgi: DXGI_FORMAT,
) -> Result<ID3D11Texture2D> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: dims.width,
        Height: dims.height,
        MipLevels: 1,
        ArraySize: 1,
        Format: dxgi,
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
    // SAFETY: `desc` is fully initialised, no initial data is supplied and the
    // out parameter points at a local Option.
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture)) }
        .map_err(|e| platform("ID3D11Device::CreateTexture2D", e))?;
    texture.ok_or_else(|| {
        FluidError::platform("dxgi: CreateTexture2D succeeded but returned no texture")
    })
}

/// Re-exported so the window-capture backend keeps sharing one BGRA path.
pub(crate) use crate::format::bgra_row_to_rgba;

/// An all-transparent frame used until the first desktop image arrives.
pub(crate) fn blank_frame(dims: Dims) -> Frame {
    Frame {
        dims,
        fmt: PixelFormat::Rgba8,
        data: vec![0u8; dims.area() as usize * 4],
        pts_ns: 0,
        index: 0,
    }
}

/// The Windows desktop duplication capture source.
pub(crate) struct DxgiSource {
    /// Position in the sorted output list, used to re-acquire after access loss.
    monitor_index: usize,
    /// Configured sub-rectangle of the monitor, in monitor coordinates.
    region: Option<Rect>,
    dims: Dims,
    /// Optional capture rate cap from `CaptureConfig::max_fps` (0 = uncapped).
    fps_cap: f32,
    refresh_hz: f32,
    timeout_ms: u32,
    include_cursor: bool,
    device: Option<ID3D11Device>,
    context: Option<ID3D11DeviceContext>,
    duplication: Option<IDXGIOutputDuplication>,
    staging: Option<ID3D11Texture2D>,
    /// Pixel format of the desktop image and of the staging texture.
    layout: Option<DesktopLayout>,
    /// Last delivered image, re-emitted when the desktop does not change.
    cached: Frame,
    index: u64,
    delivered: u64,
    start: Instant,
    closed: bool,
    cursor_shape: Option<CursorShape>,
    shape_buffer: Vec<u8>,
}

// SAFETY: D3D11 device/context objects are free-threaded and the duplication
// object is only used from whichever thread currently owns the FrameSource;
// the engine moves the source to its capture thread before pulling frames.
unsafe impl Send for DxgiSource {}

impl DxgiSource {
    /// Rate limiter: makes frame `n` wait until `start + n / fps`.
    fn pace(&self) {
        if self.fps_cap <= 0.0 || self.fps_cap.is_nan() {
            return;
        }
        let period = Duration::from_secs_f64(1.0 / self.fps_cap as f64);
        let deadline = self.start + period.mul_f64(self.delivered as f64);
        sleep_until(deadline);
    }

    /// Release the device, duplication object and staging texture.
    fn drop_pipeline(&mut self) {
        self.duplication = None;
        self.staging = None;
        self.layout = None;
        self.device = None;
        self.context = None;
        self.cursor_shape = None;
    }

    /// Re-acquire the whole pipeline after access loss or device removal.
    fn rebuild(&mut self) -> Result<()> {
        let outputs = live_outputs()?;
        if outputs.is_empty() {
            return Err(FluidError::platform(
                "dxgi: no attached desktop output found while recovering capture",
            ));
        }
        let monitor_index = self.monitor_index.min(outputs.len() - 1);
        let output = &outputs[monitor_index];
        let screen = screen_dims(&output.entry);
        let region = match self.region {
            None => None,
            Some(region) => {
                let clipped = region.intersect(screen);
                if clipped.is_empty() {
                    return Err(FluidError::platform(format!(
                        "dxgi: capture region is outside monitor {monitor_index} after a display change"
                    )));
                }
                Some(clipped)
            }
        };
        let dims = region.map(|r| r.dims()).unwrap_or(screen);
        if dims.width != self.dims.width || dims.height != self.dims.height {
            return Err(FluidError::platform(format!(
                "dxgi: display geometry changed while capturing ({} -> {dims}); reopen capture",
                self.dims
            )));
        }
        let (device, context) = create_device(&output.adapter)?;
        let duplication = duplicate_output(&device, &output.output)?;
        // SAFETY: the duplication object is live and owned by this pipeline;
        // GetDesc only reads its description.
        let duplicated = unsafe { duplication.GetDesc() };
        let layout = desktop_layout(duplicated.ModeDesc.Format)?;
        let staging = create_staging_with_format(&device, dims, layout.dxgi)?;
        self.monitor_index = monitor_index;
        self.region = region;
        self.device = Some(device);
        self.context = Some(context);
        self.duplication = Some(duplication);
        self.staging = Some(staging);
        self.layout = Some(layout);
        Ok(())
    }

    /// Copy the acquired desktop image into a fresh RGBA frame.
    fn read_back(&mut self, texture: &ID3D11Texture2D) -> Result<Frame> {
        let dims = self.dims;
        let (Some(staging), Some(context), Some(layout)) = (
            self.staging.as_ref(),
            self.context.as_ref(),
            self.layout.as_ref(),
        ) else {
            return Err(FluidError::platform(
                "dxgi: capture pipeline is not initialised",
            ));
        };
        let source_format = layout.format;
        let mut frame = Frame {
            dims,
            fmt: PixelFormat::Rgba8,
            data: vec![0u8; dims.area() as usize * 4],
            pts_ns: now_ns(),
            index: 0,
        };
        let row_bytes = dims.width as usize * 4;
        match self.region {
            None => {
                // SAFETY: both textures are valid, same format and size; the
                // CPU staging texture is not bound to the pipeline.
                unsafe { context.CopyResource(staging, texture) };
            }
            Some(region) => {
                let clip = D3D11_BOX {
                    left: region.x,
                    top: region.y,
                    front: 0,
                    right: region.x + region.w,
                    bottom: region.y + region.h,
                    back: 1,
                };
                // SAFETY: the box lies inside the desktop texture (the region
                // was clipped to the monitor) and inside the staging texture.
                unsafe {
                    context.CopySubresourceRegion(
                        staging,
                        0,
                        0,
                        0,
                        0,
                        texture,
                        0,
                        Some(&clip as *const D3D11_BOX),
                    );
                }
            }
        }
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        // SAFETY: `staging` is a staging texture created with CPU read access;
        // `mapped` is a local out parameter.
        unsafe { context.Map(staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped)) }
            .map_err(|e| platform("ID3D11DeviceContext::Map", e))?;
        let source_row_bytes = dims.width as usize * source_format.bytes_per_pixel();
        // SAFETY: Map succeeded, so `pData` points at `RowPitch * height` bytes
        // of readable staging memory valid until Unmap. Each source row holds at
        // least `width * bytes_per_pixel` bytes of the negotiated desktop format
        // (the staging texture has that size); destination rows are exactly
        // `row_bytes` wide. A conversion error is stashed and returned only after
        // Unmap, so the staging texture is never left mapped.
        let convert_error = unsafe {
            let base = mapped.pData as *const u8;
            let pitch = mapped.RowPitch as usize;
            let mut failure = None;
            for y in 0..dims.height as usize {
                let src =
                    std::slice::from_raw_parts(base.add(y * pitch), source_row_bytes.min(pitch));
                let dst = &mut frame.data[y * row_bytes..(y + 1) * row_bytes];
                if let Err(err) = convert_row(src, dst, dims.width as usize, source_format) {
                    failure = Some(err);
                    break;
                }
            }
            context.Unmap(staging, 0);
            failure
        };
        if let Some(err) = convert_error {
            return Err(err);
        }
        Ok(frame)
    }

    /// Refresh the cached pointer shape from the frame metadata.
    fn update_cursor_shape(&mut self, info: &DXGI_OUTDUPL_FRAME_INFO) {
        let size = info.PointerShapeBufferSize;
        if size == 0 {
            return; // No new shape this frame; keep the cached one.
        }
        let Some(duplication) = self.duplication.as_ref() else {
            return;
        };
        if self.shape_buffer.len() < size as usize {
            self.shape_buffer.resize(size as usize, 0);
        }
        let mut required = 0u32;
        let mut shape_info = DXGI_OUTDUPL_POINTER_SHAPE_INFO::default();
        // SAFETY: the buffer holds at least `size` writable bytes and the out
        // parameters point at locals.
        let ok = unsafe {
            duplication.GetFramePointerShape(
                size,
                self.shape_buffer.as_mut_ptr() as *mut c_void,
                &mut required,
                &mut shape_info,
            )
        };
        if ok.is_err() {
            return;
        }
        let Some(kind) = CursorKind::from_raw(shape_info.Type) else {
            return;
        };
        let rows = match kind {
            CursorKind::Monochrome => shape_info.Height as usize * 2,
            _ => shape_info.Height as usize,
        };
        let needed = (shape_info.Pitch as usize)
            .saturating_mul(rows)
            .min(self.shape_buffer.len())
            .min(required as usize);
        if needed == 0 {
            return;
        }
        self.cursor_shape = Some(CursorShape {
            kind,
            width: shape_info.Width,
            height: shape_info.Height,
            pitch: shape_info.Pitch,
            hotspot_x: shape_info.HotSpot.x,
            hotspot_y: shape_info.HotSpot.y,
            data: self.shape_buffer[..needed].to_vec(),
        });
    }

    /// Composite the cached pointer at the position reported this frame.
    fn paint_cursor(&self, frame: &mut Frame, info: &DXGI_OUTDUPL_FRAME_INFO) {
        if !self.include_cursor || !info.PointerPosition.Visible.as_bool() {
            return;
        }
        let Some(shape) = self.cursor_shape.as_ref() else {
            return;
        };
        let (offset_x, offset_y) = match self.region {
            Some(region) => (region.x as i32, region.y as i32),
            None => (0, 0),
        };
        let position = info.PointerPosition.Position;
        composite_cursor(frame, shape, position.x - offset_x, position.y - offset_y);
    }

    /// Give the frame its sequence number and timestamp.
    fn stamp(&mut self, mut frame: Frame) -> Frame {
        frame.pts_ns = now_ns();
        frame.index = self.index;
        self.index += 1;
        self.delivered += 1;
        frame
    }

    /// Re-emit the previous frame (desktop unchanged or no image available).
    fn emit_cached(&mut self) -> Frame {
        let frame = self.cached.clone();
        self.stamp(frame)
    }

    /// Release the desktop frame acquired by the last successful
    /// AcquireNextFrame call; failures are harmless.
    fn release_frame(&self) {
        if let Some(duplication) = self.duplication.as_ref() {
            // SAFETY: only called after a successful AcquireNextFrame, and each
            // acquired frame is released exactly once.
            let _ = unsafe { duplication.ReleaseFrame() };
        }
    }
}

impl FrameSource for DxgiSource {
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
        "dxgi"
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        if self.closed {
            return Ok(None);
        }
        self.pace();
        // Outcome of a single acquisition attempt.
        enum Step {
            Acquired(DXGI_OUTDUPL_FRAME_INFO, Option<IDXGIResource>),
            Timeout,
            Retry,
        }
        let mut attempt = 0u32;
        loop {
            if self.duplication.is_none() {
                self.rebuild()?;
            }
            // Acquire in a scope that ends the immutable borrow of self before
            // the frame is copied into an owned buffer below.
            let step = {
                let Some(duplication) = self.duplication.as_ref() else {
                    return Err(FluidError::platform(
                        "dxgi: capture pipeline is not initialised",
                    ));
                };
                let mut info = DXGI_OUTDUPL_FRAME_INFO::default();
                let mut resource: Option<IDXGIResource> = None;
                // SAFETY: duplication is a valid duplication object; both out
                // parameters point at locals initialised just above.
                match unsafe {
                    duplication.AcquireNextFrame(self.timeout_ms, &mut info, &mut resource)
                } {
                    Ok(()) => Step::Acquired(info, resource),
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
            let (info, resource) = match step {
                // No new desktop image within the timeout: repeat the previous
                // frame so a live source never ends.
                Step::Timeout => return Ok(Some(self.emit_cached())),
                Step::Retry => {
                    // Lost the desktop (secure desktop, mode change, driver
                    // reset): rebuild the whole pipeline once and retry.
                    attempt = 1;
                    self.drop_pipeline();
                    continue;
                }
                Step::Acquired(info, resource) => (info, resource),
            };

            // Copy while the frame is still acquired, then always release it.
            let texture = match resource.map(|r| r.cast::<ID3D11Texture2D>()) {
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
            self.update_cursor_shape(&info);
            let read = self.read_back(&texture);
            self.release_frame();
            let mut frame = read?;
            self.paint_cursor(&mut frame, &info);
            let frame = self.stamp(frame);
            self.cached = frame.clone();
            return Ok(Some(frame));
        }
    }

    fn close(&mut self) {
        self.closed = true;
        self.drop_pipeline();
    }
}

/// Open the DXGI backend for `config`.
pub(crate) fn open(config: &CaptureConfig) -> Result<Box<dyn FrameSource>> {
    config.validate()?;
    let outputs = live_outputs()?;
    if outputs.is_empty() {
        return Err(FluidError::platform(
            "dxgi: no attached desktop output found (is a display connected and the session interactive?)",
        ));
    }
    // An out of range index is clamped to the last output (the CLI lists the
    // monitors, so this is only a safety net).
    let monitor_index = config.monitor.min(outputs.len() - 1);
    let output = &outputs[monitor_index];
    let screen = screen_dims(&output.entry);
    let region = match config.region {
        None => None,
        Some(requested) => {
            let requested: Rect = requested.into();
            let clipped = requested.intersect(screen);
            if clipped.is_empty() {
                return Err(FluidError::invalid(format!(
                    "capture.region {}x{} at ({}, {}) does not intersect monitor {monitor_index} ({}x{})",
                    requested.w, requested.h, requested.x, requested.y, screen.width, screen.height
                )));
            }
            Some(clipped)
        }
    };
    let dims = region.map(|r| r.dims()).unwrap_or(screen);
    let (device, context) = create_device(&output.adapter)?;
    let duplication = duplicate_output(&device, &output.output)?;
    // The desktop image may be HDR/10-bit; negotiate the format before creating
    // the staging texture so the two match exactly.
    // SAFETY: the duplication object was just created and is owned here;
    // GetDesc only reads its description.
    let duplicated = unsafe { duplication.GetDesc() };
    let layout = desktop_layout(duplicated.ModeDesc.Format)?;
    let staging = create_staging_with_format(&device, dims, layout.dxgi)?;
    Ok(Box::new(DxgiSource {
        monitor_index,
        region,
        dims,
        fps_cap: config.max_fps,
        refresh_hz: output.entry.refresh_hz,
        timeout_ms: config.timeout_ms,
        include_cursor: config.include_cursor,
        device: Some(device),
        context: Some(context),
        duplication: Some(duplication),
        staging: Some(staging),
        layout: Some(layout),
        cached: blank_frame(dims),
        index: 0,
        delivered: 0,
        start: Instant::now(),
        closed: false,
        cursor_shape: None,
        shape_buffer: Vec::new(),
    }))
}

/// Enumerate desktop outputs as `MonitorInfo`, ordered by desktop position.
pub(crate) fn enumerate_monitors() -> Result<Vec<MonitorInfo>> {
    let outputs = live_outputs()?;
    Ok(outputs
        .iter()
        .enumerate()
        .map(|(index, output)| MonitorInfo {
            index,
            name: output.entry.name.clone(),
            dims: screen_dims(&output.entry),
            refresh_hz: output.entry.refresh_hz,
            primary: output.entry.primary,
            left: output.entry.rect.left,
            top: output.entry.rect.top,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wide_name(text: &str) -> [u16; 32] {
        let mut raw = [0u16; 32];
        for (slot, unit) in raw.iter_mut().zip(text.encode_utf16()) {
            *slot = unit;
        }
        raw
    }

    #[test]
    fn monitor_names_are_shortened() {
        assert_eq!(output_name(&wide_name("\\\\.\\DISPLAY1"), 0), "DISPLAY1");
        assert_eq!(output_name(&wide_name("\\\\.\\DISPLAY2"), 1), "DISPLAY2");
        assert_eq!(output_name(&wide_name(""), 3), "output-3");
        // A name that is only the device prefix falls back to the ordinal.
        assert_eq!(output_name(&wide_name("\\\\.\\"), 4), "output-4");
    }

    #[test]
    fn bgra_rows_convert_to_opaque_rgba() {
        let src = [10u8, 20, 30, 0, 1, 2, 3, 0];
        let mut dst = [0u8; 8];
        bgra_row_to_rgba(&src, &mut dst);
        assert_eq!(dst, [30, 20, 10, 255, 3, 2, 1, 255]);
    }

    #[test]
    fn blank_frames_are_zeroed_rgba() {
        let frame = blank_frame(Dims::new(4, 3));
        assert_eq!(frame.fmt, PixelFormat::Rgba8);
        assert_eq!(frame.data.len(), 4 * 3 * 4);
        assert!(frame.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn desktop_formats_are_recognised() {
        let cases = [
            (DXGI_FORMAT_B8G8R8A8_UNORM, DesktopFormat::Bgra8),
            (DXGI_FORMAT_R8G8B8A8_UNORM, DesktopFormat::Rgba8),
            (DXGI_FORMAT_R10G10B10A2_UNORM, DesktopFormat::Rgb10a2),
            (DXGI_FORMAT_R16G16B16A16_FLOAT, DesktopFormat::Rgba16F),
        ];
        for (dxgi, expected) in cases {
            let layout = desktop_layout(dxgi).unwrap();
            assert_eq!(layout.format, expected);
            assert_eq!(layout.dxgi, dxgi);
        }
    }

    #[test]
    fn unsupported_desktop_formats_are_reported() {
        for raw in [0u32, 115u32] {
            let err = desktop_layout(DXGI_FORMAT(raw as i32)).unwrap_err();
            let text = err.to_string();
            assert!(text.contains("unsupported"), "unexpected error: {text}");
            assert!(text.contains("DXGI_FORMAT("), "unexpected error: {text}");
            assert!(
                text.contains("R16G16B16A16_FLOAT"),
                "unexpected error: {text}"
            );
        }
    }
}
