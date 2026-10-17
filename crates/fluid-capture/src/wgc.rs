//! Windows Graphics Capture (WGC) backend for per-window capture.
//!
//! Capturing a single HWND avoids Desktop Duplication's feedback loop: the
//! fluid overlay is a separate top-level window, so presenting it does not
//! generate new frames in the captured window. This keeps idle GPU load low.

use crate::clock::qpc_now;
use crate::duplication::CaptureError;
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
use windows::Win32::System::WinRT::Direct3D11::{
    CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess,
};
use windows::Win32::System::WinRT::Graphics::Capture::IGraphicsCaptureItemInterop;
use windows::Win32::System::WinRT::{RoInitialize, RO_INIT_MULTITHREADED};

pub struct WgcCapture {
    _item: GraphicsCaptureItem,
    frame_pool: Direct3D11CaptureFramePool,
    session: GraphicsCaptureSession,
    frame: Option<Direct3D11CaptureFrame>,
    texture: Option<ID3D11Texture2D>,
    width: u32,
    height: u32,
}

pub struct WgcFrame<'a> {
    pub texture: &'a ID3D11Texture2D,
    pub qpc: i64,
}

impl WgcCapture {
    pub fn new(device: &ID3D11Device, hwnd: HWND) -> Result<Self, CaptureError> {
        unsafe {
            // The pipeline thread already initializes COM as MTA. RoInitialize
            // returns S_FALSE when the apartment is already initialized, which
            // windows-rs treats as success.
            RoInitialize(RO_INIT_MULTITHREADED)?;
        }

        if !GraphicsCaptureSession::IsSupported().unwrap_or(false) {
            return Err(CaptureError::Other(
                "Windows Graphics Capture is not supported on this system".into(),
            ));
        }

        let dxgi_device: IDXGIDevice = device.cast()?;
        let inspectable = unsafe { CreateDirect3D11DeviceFromDXGIDevice(&dxgi_device)? };
        let direct3d_device: IDirect3DDevice = inspectable.cast()?;

        let interop: IGraphicsCaptureItemInterop =
            factory::<GraphicsCaptureItem, IGraphicsCaptureItemInterop>()?;
        let item: GraphicsCaptureItem = unsafe { interop.CreateForWindow(hwnd)? };
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
            "Windows Graphics Capture ready: {}x{} hwnd={:?}",
            size.Width,
            size.Height,
            hwnd
        );

        Ok(Self {
            _item: item,
            frame_pool,
            session,
            frame: None,
            texture: None,
            width: size.Width as u32,
            height: size.Height as u32,
        })
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
            let _ = frame.Close();
            return Ok(None);
        }
        self.width = content.Width as u32;
        self.height = content.Height as u32;

        let surface: IDirect3DSurface = frame.Surface()?;
        let access: IDirect3DDxgiInterfaceAccess = surface.cast()?;
        let texture: ID3D11Texture2D = unsafe { access.GetInterface()? };
        self.texture = Some(texture);
        self.frame = Some(frame);
        Ok(Some(WgcFrame {
            texture: self.texture.as_ref().unwrap(),
            qpc: qpc_now(),
        }))
    }

    pub fn release_frame(&mut self) {
        self.texture = None;
        if let Some(frame) = self.frame.take() {
            let _ = frame.Close();
        }
    }
}

impl Drop for WgcCapture {
    fn drop(&mut self) {
        self.release_frame();
        let _ = self.session.Close();
        let _ = self.frame_pool.Close();
    }
}
