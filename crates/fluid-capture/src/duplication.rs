//! DXGI Desktop Duplication：只读采集整块显示器的合成画面。
//! 这是 OBS/腾讯会议等软件同款的系统级 API——不接触任何目标进程，
//! 这也是 fluid 对反作弊安全的根本原因。

use crate::clock::qpc_now;
use windows::core::Interface;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D11::{ID3D11Device, ID3D11Texture2D};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIFactory1, IDXGIOutput1, IDXGIOutputDuplication, IDXGIResource,
    DXGI_ERROR_ACCESS_LOST, DXGI_ERROR_WAIT_TIMEOUT, DXGI_OUTDUPL_FRAME_INFO,
};
use windows::Win32::Graphics::Gdi::HMONITOR;

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("桌面复制已失效（显示模式切换或权限变化），需要重建")]
    AccessLost,
    #[error("Windows API 错误: {0}")]
    Windows(#[from] windows::core::Error),
    #[error("{0}")]
    Other(String),
}

fn is_timeout(e: &windows::core::Error) -> bool {
    e.code() == DXGI_ERROR_WAIT_TIMEOUT
}

fn is_access_lost(e: &windows::core::Error) -> bool {
    e.code() == DXGI_ERROR_ACCESS_LOST
}

pub struct Duplication {
    dup: IDXGIOutputDuplication,
    output_rect: RECT,
    #[allow(dead_code)]
    monitor: HMONITOR,
    desktop_tex: Option<ID3D11Texture2D>,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
}

pub struct AcquiredFrame<'a> {
    pub texture: &'a ID3D11Texture2D,
    /// 该帧被 DWM 合成时刻的 QPC 计数（用于插帧相位与延迟统计）。
    pub qpc: i64,
    /// 自上次采集以来合成的总帧数（>1 表示发生了积压/丢帧）。
    pub accumulated: u32,
    /// True when LastPresentTime indicates that desktop image content changed.
    /// Pointer-only updates have this false and should not trigger frame
    /// generation or a full GPU frame copy.
    pub desktop_updated: bool,
}

impl Duplication {
    pub fn new_for_monitor(device: &ID3D11Device, monitor: HMONITOR) -> Result<Self, CaptureError> {
        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1()? };
        let mut ai = 0u32;
        loop {
            let adapter = match unsafe { factory.EnumAdapters1(ai) } {
                Ok(a) => a,
                Err(_) => break,
            };
            let mut oi = 0u32;
            loop {
                let output = match unsafe { adapter.EnumOutputs(oi) } {
                    Ok(o) => o,
                    Err(_) => break,
                };
                let desc = match unsafe { output.GetDesc() } {
                    Ok(d) => d,
                    Err(_) => {
                        oi += 1;
                        continue;
                    }
                };
                if desc.Monitor != monitor {
                    oi += 1;
                    continue;
                }
                let out1 = output.cast::<IDXGIOutput1>()?;
                let dup = unsafe { out1.DuplicateOutput(device)? };
                let d = unsafe { dup.GetDesc() };
                tracing::info!("桌面复制就绪: {}x{}", d.ModeDesc.Width, d.ModeDesc.Height);
                return Ok(Self {
                    dup,
                    output_rect: desc.DesktopCoordinates,
                    monitor,
                    desktop_tex: None,
                    width: d.ModeDesc.Width,
                    height: d.ModeDesc.Height,
                    format: d.ModeDesc.Format,
                });
            }
            ai += 1;
        }
        Err(CaptureError::Other(
            "未在 DXGI 中找到该显示器对应的输出".into(),
        ))
    }

    pub fn output_rect(&self) -> RECT {
        self.output_rect
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// The shader pipeline currently assumes an SDR BGRA8 desktop image.
    pub fn is_supported_format(&self) -> bool {
        self.format == DXGI_FORMAT_B8G8R8A8_UNORM
    }

    pub fn format(&self) -> DXGI_FORMAT {
        self.format
    }

    /// 以超时 0 反复获取，丢弃积压帧，只保留最新一帧（低延迟的关键）。
    /// 成功后调用方必须在使用完纹理后调用 [`Self::release_frame`]。
    pub fn acquire_latest(&mut self) -> Result<Option<AcquiredFrame<'_>>, CaptureError> {
        let mut info = DXGI_OUTDUPL_FRAME_INFO::default();
        let mut resource: Option<IDXGIResource> = None;
        if let Err(e) = unsafe { self.dup.AcquireNextFrame(0, &mut info, &mut resource) } {
            if is_timeout(&e) {
                return Ok(None);
            }
            if is_access_lost(&e) {
                return Err(CaptureError::AccessLost);
            }
            return Err(e.into());
        }
        let Some(resource) = resource else {
            unsafe {
                let _ = self.dup.ReleaseFrame();
            }
            return Err(CaptureError::Other(
                "AcquireNextFrame returned no desktop resource".into(),
            ));
        };

        // DDA itself only exposes the newest composed desktop image. The
        // AccumulatedFrames field tells us how many updates were coalesced, but
        // a second AcquireNextFrame call is not allowed until ReleaseFrame. The
        // caller copies this texture to the ring, then calls release_frame.
        let tex = match resource.cast::<ID3D11Texture2D>() {
            Ok(tex) => tex,
            Err(e) => {
                unsafe {
                    let _ = self.dup.ReleaseFrame();
                }
                return Err(e.into());
            }
        };
        self.desktop_tex = Some(tex);
        Ok(Some(AcquiredFrame {
            texture: self.desktop_tex.as_ref().unwrap(),
            qpc: if info.LastPresentTime > 0 {
                info.LastPresentTime
            } else {
                // Pointer-only updates and some static desktops report zero.
                qpc_now()
            },
            accumulated: info.AccumulatedFrames.max(1),
            desktop_updated: info.LastPresentTime > 0,
        }))
    }

    pub fn release_frame(&mut self) {
        unsafe {
            let _ = self.dup.ReleaseFrame();
        }
        self.desktop_tex = None;
    }
}
