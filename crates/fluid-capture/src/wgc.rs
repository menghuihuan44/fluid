//! Windows Graphics Capture (WGC) backend for per-window capture.
//!
//! Capturing a single HWND avoids Desktop Duplication's feedback loop: the
//! fluid overlay is a separate top-level window, so presenting it does not
//! generate new frames in the captured window. This keeps idle GPU load low.

use crate::clock::{qpc_now, system_relative_100ns_to_qpc};
use crate::duplication::CaptureError;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, OnceLock};
use windows::core::{factory, Interface};
use windows::Graphics::Capture::{
    Direct3D11CaptureFrame, Direct3D11CaptureFramePool, GraphicsCaptureItem, GraphicsCaptureSession,
};
use windows::Graphics::DirectX::Direct3D11::{IDirect3DDevice, IDirect3DSurface};
use windows::Graphics::DirectX::DirectXPixelFormat;
use windows::Graphics::SizeInt32;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D11::{ID3D11Device, ID3D11Texture2D};
use windows::Win32::Graphics::Dxgi::IDXGIDevice;
use windows::Win32::Graphics::Gdi::HMONITOR;
use windows::Win32::System::WinRT::Direct3D11::{
    CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess,
};
use windows::Win32::System::WinRT::Graphics::Capture::IGraphicsCaptureItemInterop;
use windows::Win32::System::WinRT::{RoInitialize, RO_INIT_MULTITHREADED};

pub struct WgcCapture {
    // Put frame/texture before the session/pool so explicit Close() calls from
    // drop cannot block the pipeline when the target window disappears.
    frame: Option<Direct3D11CaptureFrame>,
    texture: Option<ID3D11Texture2D>,
    _session: GraphicsCaptureSession,
    frame_pool: Direct3D11CaptureFramePool,
    direct3d_device: IDirect3DDevice,
    width: u32,
    height: u32,
    timestamp_normalizer: TimestampNormalizer,
    last_frame_qpc: i64,
}

pub struct WgcFrame<'a> {
    pub texture: &'a ID3D11Texture2D,
    pub qpc: i64,
    /// Raw WinRT timestamp for diagnostics; zero when unavailable.
    pub system_relative_100ns: i64,
}

const TIMESTAMP_WARMUP_SAMPLES: u32 = 16;

/// Normalize the compositor timestamp to QPC without erasing its inter-frame
/// intervals. During warmup the largest observed delivery lead is retained;
/// afterwards the correction is frozen and only widened if a frame would
/// otherwise be stamped in the future.
#[derive(Default)]
struct TimestampNormalizer {
    bias: Option<i64>,
    samples: u32,
    last: i64,
}

impl TimestampNormalizer {
    fn normalize(&mut self, raw_qpc: i64, received_qpc: i64) -> i64 {
        let lead = (received_qpc as i128 - raw_qpc as i128)
            .clamp(i64::MIN as i128, i64::MAX as i128) as i64;
        let mut bias = match self.bias {
            None => lead,
            Some(b) if self.samples < TIMESTAMP_WARMUP_SAMPLES => b.min(lead),
            Some(b) => b,
        };
        self.samples = self.samples.saturating_add(1);
        self.bias = Some(bias);
        let mut corrected = raw_qpc.saturating_add(bias);
        if corrected > received_qpc {
            bias = bias.min(lead);
            self.bias = Some(bias);
            corrected = raw_qpc.saturating_add(bias).min(received_qpc);
        }
        // Timestamps must never move backwards when a bias refinement occurs.
        let monotonic = if self.last > 0 {
            corrected.max(self.last.saturating_add(1))
        } else {
            corrected
        };
        self.last = monotonic;
        monotonic.max(1)
    }
}

// WGC was created with Direct3D11CaptureFramePool::CreateFreeThreaded and the
// pipeline COM apartment is MTA. Disposing the COM references from a cleanup
// thread is safe and prevents a dead target from blocking the render thread.
unsafe impl Send for WgcCapture {}

static WGC_CLEANUP_TX: OnceLock<mpsc::Sender<WgcCapture>> = OnceLock::new();
static WGC_CLEANUP_PENDING: AtomicUsize = AtomicUsize::new(0);

fn cleanup_sender() -> Option<&'static mpsc::Sender<WgcCapture>> {
    if let Some(tx) = WGC_CLEANUP_TX.get() {
        return Some(tx);
    }
    let (tx, rx) = mpsc::channel::<WgcCapture>();
    let builder = std::thread::Builder::new().name("fluid-wgc-cleanup".into());
    if let Err(e) = builder.spawn(move || {
        // Serialize WinRT/D3D teardown and give the pipeline a short head start
        // on the new target before releasing the previous capture objects.
        while let Ok(capture) = rx.recv() {
            std::thread::sleep(std::time::Duration::from_millis(120));
            drop(capture);
            WGC_CLEANUP_PENDING.fetch_sub(1, Ordering::Relaxed);
        }
    }) {
        tracing::warn!("failed to start WGC cleanup worker: {e}");
        return None;
    }
    let _ = WGC_CLEANUP_TX.set(tx);
    WGC_CLEANUP_TX.get()
}

impl WgcCapture {
    pub fn dispose_async(self) {
        let Some(tx) = cleanup_sender() else {
            drop(self);
            return;
        };
        WGC_CLEANUP_PENDING.fetch_add(1, Ordering::Relaxed);
        if tx.send(self).is_err() {
            WGC_CLEANUP_PENDING.fetch_sub(1, Ordering::Relaxed);
            return;
        }
        let pending = WGC_CLEANUP_PENDING.load(Ordering::Relaxed);
        if pending == 8 || pending == 32 {
            tracing::warn!("WGC cleanup queue is backing up: pending={pending}");
        }
    }
}

pub fn wgc_cleanup_pending() -> usize {
    WGC_CLEANUP_PENDING.load(Ordering::Relaxed)
}

impl WgcCapture {
    fn prepare_device(device: &ID3D11Device) -> Result<IDirect3DDevice, CaptureError> {
        unsafe {
            RoInitialize(RO_INIT_MULTITHREADED)?;
        }
        if !GraphicsCaptureSession::IsSupported().unwrap_or(false) {
            return Err(CaptureError::Other(
                "Windows Graphics Capture is not supported on this system".into(),
            ));
        }
        let dxgi_device: IDXGIDevice = device.cast()?;
        let inspectable = unsafe { CreateDirect3D11DeviceFromDXGIDevice(&dxgi_device)? };
        Ok(inspectable.cast()?)
    }

    fn from_item(
        direct3d_device: IDirect3DDevice,
        item: GraphicsCaptureItem,
        label: &str,
    ) -> Result<Self, CaptureError> {
        let size = item.Size()?;
        if size.Width <= 0 || size.Height <= 0 {
            return Err(CaptureError::Other("capture item has an empty size".into()));
        }

        let frame_pool = Direct3D11CaptureFramePool::CreateFreeThreaded(
            &direct3d_device,
            DirectXPixelFormat::B8G8R8A8UIntNormalized,
            2,
            size,
        )?;
        let session = frame_pool.CreateCaptureSession(&item)?;
        if let Err(e) = session.SetIsCursorCaptureEnabled(false) {
            tracing::debug!("WGC cursor setting unavailable: {e}");
        }
        if let Err(e) = session.SetIsBorderRequired(false) {
            tracing::debug!("WGC border setting unavailable: {e}");
        }
        session.StartCapture()?;
        tracing::info!(
            "Windows Graphics Capture ready: {}x{} {}",
            size.Width,
            size.Height,
            label
        );

        Ok(Self {
            frame: None,
            texture: None,
            _session: session,
            frame_pool,
            direct3d_device,
            width: size.Width as u32,
            height: size.Height as u32,
            timestamp_normalizer: TimestampNormalizer::default(),
            last_frame_qpc: 0,
        })
    }

    pub fn new(device: &ID3D11Device, hwnd: HWND) -> Result<Self, CaptureError> {
        let direct3d_device = Self::prepare_device(device)?;
        let interop: IGraphicsCaptureItemInterop =
            factory::<GraphicsCaptureItem, IGraphicsCaptureItemInterop>()?;
        let item: GraphicsCaptureItem = unsafe { interop.CreateForWindow(hwnd)? };
        Self::from_item(direct3d_device, item, &format!("hwnd={hwnd:?}"))
    }

    pub fn new_monitor(device: &ID3D11Device, monitor: HMONITOR) -> Result<Self, CaptureError> {
        let direct3d_device = Self::prepare_device(device)?;
        let interop: IGraphicsCaptureItemInterop =
            factory::<GraphicsCaptureItem, IGraphicsCaptureItemInterop>()?;
        let item: GraphicsCaptureItem = unsafe { interop.CreateForMonitor(monitor)? };
        Self::from_item(direct3d_device, item, &format!("monitor={monitor:?}"))
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), CaptureError> {
        if width == 0 || height == 0 {
            return Err(CaptureError::Other("cannot resize WGC to zero".into()));
        }
        self.release_frame();
        self.frame_pool.Recreate(
            &self.direct3d_device,
            DirectXPixelFormat::B8G8R8A8UIntNormalized,
            2,
            SizeInt32 {
                Width: width as i32,
                Height: height as i32,
            },
        )?;
        self.width = width;
        self.height = height;
        Ok(())
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn acquire_latest(&mut self) -> Result<Option<WgcFrame<'_>>, CaptureError> {
        if self.frame.is_some() {
            return Ok(None);
        }
        let frame = match self.frame_pool.TryGetNextFrame() {
            Ok(frame) => frame,
            // Polling an empty frame pool commonly returns a method-call error.
            Err(_) => return Ok(None),
        };
        let content = frame.ContentSize().unwrap_or(SizeInt32 {
            Width: self.width as i32,
            Height: self.height as i32,
        });
        if content.Width <= 0 || content.Height <= 0 {
            drop(frame);
            return Ok(None);
        }
        self.width = content.Width as u32;
        self.height = content.Height as u32;

        let surface: IDirect3DSurface = frame.Surface()?;
        let access: IDirect3DDxgiInterfaceAccess = surface.cast()?;
        let texture: ID3D11Texture2D = unsafe { access.GetInterface()? };
        // WGC carries the compositor's frame timestamp. Using the polling
        // instant here would quantize every source interval to our output
        // cadence and make frame-generation phases jitter. Fall back to QPC
        // only when the platform does not provide the timestamp.
        let received_qpc = qpc_now();
        let raw_timestamp = frame.SystemRelativeTime().ok().map(|t| t.Duration);
        let raw_qpc = raw_timestamp
            .map(system_relative_100ns_to_qpc)
            .filter(|&v| v > 0);
        let frame_qpc = if let Some(raw_qpc) = raw_qpc {
            let normalized = self.timestamp_normalizer.normalize(raw_qpc, received_qpc);
            let monotonic = if self.last_frame_qpc > 0 {
                normalized.max(self.last_frame_qpc.saturating_add(1))
            } else {
                normalized
            };
            self.last_frame_qpc = monotonic;
            (monotonic, raw_timestamp.unwrap_or(0))
        } else {
            self.last_frame_qpc = received_qpc.max(self.last_frame_qpc);
            (self.last_frame_qpc, 0)
        };
        self.texture = Some(texture);
        self.frame = Some(frame);
        Ok(Some(WgcFrame {
            texture: self.texture.as_ref().unwrap(),
            qpc: frame_qpc.0,
            system_relative_100ns: frame_qpc.1,
        }))
    }

    pub fn release_frame(&mut self) {
        // Do not call Close() while polling. A target can disappear between
        // TryGetNextFrame and release; synchronous WinRT teardown can then block
        // the pipeline indefinitely. Dropping the COM reference is non-blocking.
        self.texture = None;
        self.frame = None;
    }
}

impl Drop for WgcCapture {
    fn drop(&mut self) {
        // This Drop runs on the dedicated cleanup worker, never on the pipeline
        // thread. Explicit Close() is needed to tear down the WinRT capture
        // session worker; if it blocks, the pipeline remains responsive.
        self.texture = None;
        self.frame = None;
        tracing::debug!("WGC cleanup: closing capture session");
        let _ = self._session.Close();
        let _ = self.frame_pool.Close();
        tracing::debug!("WGC cleanup: capture session closed");
    }
}

#[cfg(test)]
mod tests {
    use super::TimestampNormalizer;

    #[test]
    fn warmup_then_frozen_bias_preserves_source_intervals() {
        let mut n = TimestampNormalizer::default();
        let raw_step = 5_000_000i64; // 500 ms at QPC/10 MHz scale
        let mut raw = 100_000_000i64;
        let mut received = raw - 35_000; // delivered 3.5 ms early
        for _ in 0..20 {
            let corrected = n.normalize(raw, received);
            assert!(corrected <= received);
            raw += raw_step;
            received = raw - 35_000;
        }

        // After the correction is frozen, raw timestamp deltas must survive;
        // they are what the interpolation phase depends on.
        let a = n.normalize(raw, received);
        raw += raw_step;
        received = raw - 35_000;
        let b = n.normalize(raw, received);
        assert_eq!(b - a, raw_step);
    }

    #[test]
    fn a_larger_lead_never_creates_a_future_timestamp() {
        let mut n = TimestampNormalizer::default();
        for i in 0..20 {
            let raw = 1_000_000_000 + i * 5_000_000;
            let received = raw - 2_000_000;
            assert!(n.normalize(raw, received) <= received);
        }
        let raw = 1_000_000_000 + 21 * 5_000_000;
        let received = raw - 8_000_000;
        assert!(n.normalize(raw, received) <= received);
    }
}
