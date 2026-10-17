//! 覆盖窗口：置顶、免激活、鼠标点击穿透（输入直达下层游戏），
//! 内容经 DirectComposition 承载 FLIP_DISCARD 翻转交换链，
//! 带可等待对象（waitable swapchain）实现 MaximumFrameLatency=1 的最低延迟呈现。
//! 窗口用 WDA_EXCLUDEFROMCAPTURE 从系统采集（含我们自己的 DDA 采集）中排除，
//! 避免自捕获递归，也让本程序永不出现在录屏/直播画面里。

use crate::error::GfxError;
use windows::core::{w, IUnknown, Interface};
use windows::Win32::Foundation::{BOOL, COLORREF, HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D11::{ID3D11Device, ID3D11RenderTargetView, ID3D11Texture2D};
use windows::Win32::Graphics::DirectComposition::{
    DCompositionCreateDevice, IDCompositionDevice, IDCompositionTarget, IDCompositionVisual,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_ALPHA_MODE_PREMULTIPLIED, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_UNKNOWN,
    DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    IDXGIDevice, IDXGIFactory2, IDXGISwapChain1, IDXGISwapChain2, DXGI_PRESENT,
    DXGI_SCALING_STRETCH, DXGI_SWAP_CHAIN_DESC1,
    DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT, DXGI_SWAP_EFFECT_FLIP_DISCARD,
    DXGI_USAGE_RENDER_TARGET_OUTPUT,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::WaitForSingleObject;
use windows::Win32::UI::WindowsAndMessaging::*;

unsafe extern "system" fn overlay_wndproc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    DefWindowProcW(hwnd, msg, wparam, lparam)
}

fn register_class() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let hinstance = unsafe { GetModuleHandleW(None) }.unwrap();
        let hinstance = windows::Win32::Foundation::HINSTANCE(hinstance.0);
        let wc = WNDCLASSW {
            lpfnWndProc: Some(overlay_wndproc),
            hInstance: hinstance,
            lpszClassName: w!("FluidOverlayWnd"),
            ..Default::default()
        };
        let atom = unsafe { RegisterClassW(&wc) };
        debug_assert!(atom != 0);
    });
}

pub struct OverlayWindow {
    hwnd: HWND,
    device: ID3D11Device,
    swapchain: IDXGISwapChain1,
    waitable: HANDLE,
    dcomp_device: IDCompositionDevice,
    _dcomp_target: IDCompositionTarget,
    _dcomp_visual: IDCompositionVisual,
    rtv: Option<ID3D11RenderTargetView>,
    width: u32,
    height: u32,
    visible: bool,
    last_rect: Option<(i32, i32, i32, i32)>,
}

impl OverlayWindow {
    pub fn new(
        device: &ID3D11Device,
        factory2: &IDXGIFactory2,
        w: u32,
        h: u32,
    ) -> Result<Self, GfxError> {
        register_class();
        let hinstance = unsafe { GetModuleHandleW(None)? };
        let hinstance = windows::Win32::Foundation::HINSTANCE(hinstance.0);
        let ex_style = WINDOW_EX_STYLE(
            WS_EX_TOPMOST.0
                | WS_EX_TOOLWINDOW.0
                | WS_EX_NOACTIVATE.0
                | WS_EX_LAYERED.0
                | WS_EX_TRANSPARENT.0,
        );
        let hwnd = unsafe {
            CreateWindowExW(
                ex_style,
                w!("FluidOverlayWnd"),
                w!("fluid"),
                WINDOW_STYLE(WS_POPUP.0),
                0,
                0,
                w as i32,
                h as i32,
                None,
                None,
                hinstance,
                None,
            )?
        };

        unsafe {
            // 完全不透明，但保留鼠标穿透（WS_EX_TRANSPARENT 需要 LAYERED 才生效）
            let _ = SetLayeredWindowAttributes(hwnd, COLORREF(0), 255, LWA_ALPHA);
            match SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE) {
                Ok(()) => tracing::info!("覆盖窗口已从系统采集中排除 (WDA_EXCLUDEFROMCAPTURE)"),
                Err(e) => tracing::warn!(
                    "SetWindowDisplayAffinity 失败（需 Win10 2004+，可能出现自捕获）: {e}"
                ),
            }
        }

        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: w,
            Height: h,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            Stereo: false.into(),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: 2,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: DXGI_ALPHA_MODE_PREMULTIPLIED,
            Flags: DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT.0 as u32,
        };
        let swapchain = unsafe { factory2.CreateSwapChainForComposition(device, &desc, None)? };
        let swapchain2 = swapchain.cast::<IDXGISwapChain2>()?;
        unsafe {
            swapchain2.SetMaximumFrameLatency(1)?;
        }
        let waitable = unsafe { swapchain2.GetFrameLatencyWaitableObject() };

        let dcomp_device = unsafe {
            let dxgi_device = device.cast::<IDXGIDevice>()?;
            let dcomp: IDCompositionDevice = DCompositionCreateDevice(&dxgi_device)?;
            dcomp
        };
        let (_dcomp_target, _dcomp_visual) = unsafe {
            let target = dcomp_device.CreateTargetForHwnd(hwnd, BOOL(1))?;
            let visual = dcomp_device.CreateVisual()?;
            let content: IUnknown = swapchain.cast()?;
            visual.SetContent(&content)?;
            target.SetRoot(&visual)?;
            dcomp_device.Commit()?;
            (target, visual)
        };

        let mut this = Self {
            hwnd,
            device: device.clone(),
            swapchain,
            waitable,
            dcomp_device,
            _dcomp_target,
            _dcomp_visual,
            rtv: None,
            width: w,
            height: h,
            visible: false,
            last_rect: None,
        };
        this.rtv = Some(this.create_rtv()?);
        Ok(this)
    }

    fn create_rtv(&self) -> Result<ID3D11RenderTargetView, GfxError> {
        let tex: ID3D11Texture2D = unsafe { self.swapchain.GetBuffer(0)? };
        let mut rtv = None;
        unsafe {
            self.device
                .CreateRenderTargetView(&tex, None, Some(&mut rtv))?;
        }
        Ok(rtv.unwrap())
    }

    pub fn hwnd(&self) -> HWND {
        self.hwnd
    }

    pub fn size(&self) -> (f32, f32) {
        (self.width as f32, self.height as f32)
    }

    pub fn rtv(&self) -> Option<&ID3D11RenderTargetView> {
        self.rtv.as_ref()
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    /// 移动/缩放覆盖窗口到目标矩形（桌面物理像素），尺寸变化时重建交换链缓冲。
    pub fn place(&mut self, rect: &RECT) -> Result<(), GfxError> {
        let w = (rect.right - rect.left).max(1) as u32;
        let h = (rect.bottom - rect.top).max(1) as u32;
        let key = (rect.left, rect.top, rect.right, rect.bottom);
        if w != self.width || h != self.height {
            self.resize(w, h)?;
        }
        if self.last_rect == Some(key) {
            if !self.visible {
                self.show(true);
            }
            return Ok(());
        }
        unsafe {
            SetWindowPos(
                self.hwnd,
                HWND_TOPMOST,
                rect.left,
                rect.top,
                w as i32,
                h as i32,
                SWP_NOACTIVATE | SWP_SHOWWINDOW,
            )?;
        }
        self.last_rect = Some(key);
        if !self.visible {
            self.show(true);
        }
        Ok(())
    }

    fn resize(&mut self, w: u32, h: u32) -> Result<(), GfxError> {
        // 等待帧槽排空后再 ResizeBuffers
        self.wait_present_slot(1000);
        self.rtv = None;
        unsafe {
            self.swapchain.ResizeBuffers(
                0,
                w,
                h,
                DXGI_FORMAT_UNKNOWN,
                DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT,
            )?;
        }
        self.rtv = Some(self.create_rtv()?);
        self.width = w;
        self.height = h;
        tracing::debug!("覆盖窗口缓冲调整为 {w}x{h}");
        Ok(())
    }

    pub fn show(&mut self, visible: bool) {
        unsafe {
            let _ = ShowWindow(self.hwnd, if visible { SW_SHOWNOACTIVATE } else { SW_HIDE });
        }
        self.visible = visible;
    }

    /// 等待交换链帧槽空闲（MaximumFrameLatency=1 的节拍来源）。
    pub fn wait_present_slot(&self, timeout_ms: u32) {
        unsafe {
            let _ = WaitForSingleObject(self.waitable, timeout_ms);
        }
    }

    pub fn present(&self, sync_interval: i32) {
        // Present 返回 HRESULT（翻转模型下 DXGI_ERROR_OCCLUDED 等可安全忽略）
        let hr = unsafe {
            self.swapchain
                .Present(sync_interval as u32, DXGI_PRESENT(0))
        };
        if hr.is_err() {
            tracing::trace!("Present: {hr:?}");
        }
    }

    pub fn commit_dcomp(&self) {
        let _ = unsafe { self.dcomp_device.Commit() };
    }
}

impl Drop for OverlayWindow {
    fn drop(&mut self) {
        unsafe {
            let _ = DestroyWindow(self.hwnd);
        }
    }
}
