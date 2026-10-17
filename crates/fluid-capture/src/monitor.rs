use windows::Win32::Foundation::{HWND, RECT};
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};
use windows::Win32::Graphics::Gdi::{
    EnumDisplaySettingsW, GetMonitorInfoW, MonitorFromWindow, DEVMODEW, ENUM_CURRENT_SETTINGS,
    HMONITOR, MONITORINFOEXW, MONITOR_DEFAULTTONEAREST,
};

#[derive(Debug, Clone)]
pub struct MonitorInfo {
    pub hmonitor: isize,
    pub name: String,
    pub rect: (i32, i32, i32, i32),
    pub refresh_hz: f32,
}

pub fn monitor_from_window(hwnd: HWND) -> HMONITOR {
    unsafe { MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST) }
}

fn refresh_hz_of(device_name: &[u16; 32]) -> f32 {
    unsafe {
        let mut dm = DEVMODEW {
            dmSize: std::mem::size_of::<DEVMODEW>() as u16,
            ..Default::default()
        };
        if EnumDisplaySettingsW(
            windows::core::PCWSTR(device_name.as_ptr()),
            ENUM_CURRENT_SETTINGS,
            &mut dm,
        )
        .as_bool()
            && dm.dmDisplayFrequency > 0
        {
            return dm.dmDisplayFrequency as f32;
        }
    }
    60.0
}

pub fn enumerate_monitors() -> Vec<MonitorInfo> {
    let mut out = Vec::new();
    let factory: IDXGIFactory1 = match unsafe { CreateDXGIFactory1() } {
        Ok(f) => f,
        Err(_) => return out,
    };
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
            {
                let len = desc
                    .DeviceName
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(desc.DeviceName.len());
                let r = desc.DesktopCoordinates;
                out.push(MonitorInfo {
                    hmonitor: desc.Monitor.0 as isize,
                    name: format!(
                        "{}#{}",
                        String::from_utf16_lossy(&desc.DeviceName[..len]),
                        oi
                    ),
                    rect: (r.left, r.top, r.right, r.bottom),
                    refresh_hz: refresh_hz_of(&desc.DeviceName),
                });
            }
            oi += 1;
        }
        ai += 1;
    }
    out
}

pub fn monitor_refresh_hz(monitor: HMONITOR) -> f32 {
    unsafe {
        let mut info = MONITORINFOEXW::default();
        info.monitorInfo.cbSize = std::mem::size_of::<MONITORINFOEXW>() as u32;
        if GetMonitorInfoW(monitor, &mut info as *mut MONITORINFOEXW as *mut _).as_bool() {
            return refresh_hz_of(&info.szDevice);
        }
    }
    60.0
}

pub fn monitor_rect(monitor: HMONITOR) -> Option<RECT> {
    unsafe {
        let mut info = MONITORINFOEXW::default();
        info.monitorInfo.cbSize = std::mem::size_of::<MONITORINFOEXW>() as u32;
        if GetMonitorInfoW(monitor, &mut info as *mut MONITORINFOEXW as *mut _).as_bool() {
            Some(info.monitorInfo.rcMonitor)
        } else {
            None
        }
    }
}
