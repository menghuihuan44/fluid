use crate::error::GfxError;
use windows::core::Interface;
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::{
    D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_UNKNOWN, D3D_DRIVER_TYPE_WARP,
    D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1,
};
use windows::Win32::Graphics::Direct3D11::{
    D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
    D3D11_CREATE_DEVICE_DEBUG, D3D11_CREATE_DEVICE_FLAG, D3D11_SDK_VERSION,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIAdapter, IDXGIAdapter1, IDXGIFactory1, IDXGIFactory2,
};

/// 一套 D3D11 设备 + 立即上下文 + 工厂。整个管线生命周期内只创建一次。
pub struct DeviceBundle {
    pub adapter: Option<IDXGIAdapter1>,
    pub device: ID3D11Device,
    pub ctx: ID3D11DeviceContext,
    pub factory2: IDXGIFactory2,
    /// True when the device fell back to the WARP software rasterizer.
    pub is_warp: bool,
}

#[derive(Debug, Clone)]
pub struct AdapterInfo {
    pub adapter: IDXGIAdapter1,
    pub name: String,
    pub dedicated_vram_mb: u64,
}

pub fn enumerate_adapters() -> Result<Vec<AdapterInfo>, GfxError> {
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1()? };
    let mut out = Vec::new();
    let mut i = 0u32;
    while let Ok(adapter) = unsafe { factory.EnumAdapters1(i) } {
        let desc = unsafe { adapter.GetDesc1()? };
        let len = desc
            .Description
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(desc.Description.len());
        out.push(AdapterInfo {
            adapter,
            name: String::from_utf16_lossy(&desc.Description[..len]),
            dedicated_vram_mb: desc.DedicatedVideoMemory as u64 / (1024 * 1024),
        });
        i += 1;
    }
    Ok(out)
}

/// 用默认适配器创建硬件设备（失败时回退 WARP，保证任何机器都能跑起来）。
pub fn create_device() -> Result<DeviceBundle, GfxError> {
    match create_device_with_driver(None, D3D_DRIVER_TYPE_HARDWARE) {
        Ok(b) => Ok(b),
        Err(e) => {
            tracing::warn!("硬件设备创建失败，回退 WARP: {e}");
            create_device_with_driver(None, D3D_DRIVER_TYPE_WARP)
        }
    }
}

/// 指定适配器与驱动类型创建设备。
pub fn create_device_with_driver(
    adapter: Option<&IDXGIAdapter1>,
    driver: windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE,
) -> Result<DeviceBundle, GfxError> {
    make_device(adapter, driver, D3D11_CREATE_DEVICE_BGRA_SUPPORT).or_else(|e| {
        // 调试层可能未安装（缺 Graphics Tools），去掉 DEBUG 重试
        if cfg!(debug_assertions) {
            tracing::warn!("带调试层创建失败，重试: {e}");
            make_device(adapter, driver, D3D11_CREATE_DEVICE_BGRA_SUPPORT)
        } else {
            Err(e)
        }
    })
}

fn make_device(
    adapter: Option<&IDXGIAdapter1>,
    driver: windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE,
    mut flags: D3D11_CREATE_DEVICE_FLAG,
) -> Result<DeviceBundle, GfxError> {
    if cfg!(not(debug_assertions)) {
        flags = D3D11_CREATE_DEVICE_FLAG(flags.0 & !D3D11_CREATE_DEVICE_DEBUG.0);
    }
    let levels = [D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0];
    let driver = if adapter.is_some() {
        D3D_DRIVER_TYPE_UNKNOWN
    } else {
        driver
    };
    let mut device = None;
    let mut ctx = None;
    let mut feature_level = D3D_FEATURE_LEVEL_11_0;
    // D3D11CreateDevice 的 adapter 参数是基础接口 IDXGIAdapter
    let adapter_base: Option<IDXGIAdapter> = match adapter {
        Some(a) => Some(a.cast::<IDXGIAdapter>()?),
        None => None,
    };
    unsafe {
        D3D11CreateDevice(
            adapter_base.as_ref(),
            driver,
            HMODULE::default(),
            flags,
            Some(&levels),
            D3D11_SDK_VERSION,
            Some(&mut device),
            Some(&mut feature_level),
            Some(&mut ctx),
        )?;
    }
    let device = device.ok_or_else(|| GfxError::Other("D3D11CreateDevice 未返回设备".into()))?;
    let ctx = ctx.ok_or_else(|| GfxError::Other("D3D11CreateDevice 未返回上下文".into()))?;
    let factory1: IDXGIFactory1 = unsafe { CreateDXGIFactory1()? };
    let factory2 = factory1.cast::<IDXGIFactory2>()?;
    tracing::info!("D3D11 设备创建完成: feature level 0x{:x}", feature_level.0);
    let is_warp = driver.0 == D3D_DRIVER_TYPE_WARP.0;
    Ok(DeviceBundle {
        adapter: adapter.cloned(),
        device,
        ctx,
        factory2,
        is_warp,
    })
}
