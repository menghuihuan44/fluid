//! Win32 GDI presentation backend (cfg(windows)).
//!
//! A borderless, always-on-top window is created by the shared window module
//! (window.rs) on the calling thread, and every frame is blitted into the
//! window device context with StretchDIBits, which the driver usually maps to a
//! single DWM composition surface. That keeps latency low without owning a swap
//! chain: there is no queue to fill, every present is one blit.
//!
//! Vsync: the GDI blit is composited by DWM, so it cannot tear. Vsync is
//! enforced as a frame limiter rather than with DwmFlush: the sink asks DWM for
//! the display refresh interval (DwmGetCompositionTimingInfo.rateRefresh) and
//! never starts a present earlier than one refresh after the previous one.
//! That caps the rate at the panel refresh without the extra composition wait
//! DwmFlush added (QA defect D2: vsync used to cost two vblanks per frame).
//! The wait is bounded by one refresh interval, so a late frame never delays
//! the next one. If the timing query fails the sink assumes 60 Hz.
//!
//! Window box and message handling live in window.rs. Threading: the window
//! class, window, device context and message pump are all created and used on
//! one thread. The sink implements Send because the engine moves it into the
//! render thread before the first present; all later calls must stay on the
//! thread that called open. low_latency is accepted as a hint but the GDI blit
//! path has no extra queue to bypass.

use std::ffi::c_void;
use std::mem::size_of;
use std::time::{Duration, Instant};

use fluid_core::{Dims, FluidError, Frame, FrameSink, PixelFormat, PresentConfig, Result};
use windows::Win32::Foundation::{HWND, RECT};
use windows::Win32::Graphics::Dwm::{DWM_TIMING_INFO, DwmGetCompositionTimingInfo};
use windows::Win32::Graphics::Gdi::{
    BI_RGB, BITMAPINFO, BITMAPINFOHEADER, COLORONCOLOR, DEVMODEW, DIB_RGB_COLORS,
    ENUM_CURRENT_SETTINGS, EnumDisplaySettingsW, GetDC, HDC, ReleaseDC, SRCCOPY, SetStretchBltMode,
    StretchDIBits,
};
use windows::Win32::UI::WindowsAndMessaging::GetClientRect;
use windows::core::PCWSTR;

use super::window::OverlayWindow;

fn platform(what: &str, err: windows::core::Error) -> FluidError {
    FluidError::platform(format!("win32-window: {what}: {err}"))
}

/// Durations come from the ratio DWM reports, so NTSC rates such as
/// 60000/1001 stay exact instead of rounding to 60 Hz.
fn refresh_period_from_ratio(numerator: u32, denominator: u32) -> Option<Duration> {
    if numerator == 0 || denominator == 0 {
        return None;
    }
    let hz = numerator as f64 / denominator as f64;
    if !(hz.is_finite() && (1.0..=1000.0).contains(&hz)) {
        return None;
    }
    Some(Duration::from_secs_f64(1.0 / hz))
}

/// Refresh interval of the display mode configured for a monitor.
fn query_mode_refresh(device: &[u16; 32]) -> Option<Duration> {
    let mut mode = DEVMODEW {
        dmSize: size_of::<DEVMODEW>() as u16,
        ..Default::default()
    };
    // SAFETY: device is a NUL-terminated name filled in by GetMonitorInfoW and
    // mode is a correctly sized DEVMODEW.
    let ok =
        unsafe { EnumDisplaySettingsW(PCWSTR(device.as_ptr()), ENUM_CURRENT_SETTINGS, &mut mode) }
            .as_bool();
    if !ok {
        return None;
    }
    refresh_period_from_ratio(mode.dmDisplayFrequency, 1)
}

/// Refresh interval of the monitor the window is on. DWM's composition timing
/// is the most accurate source but is not available on every session, so fall
/// back to the configured display mode and finally to 60 Hz.
fn query_refresh_period(hwnd: HWND, device: &[u16; 32]) -> Duration {
    let mut info = DWM_TIMING_INFO {
        cbSize: size_of::<DWM_TIMING_INFO>() as u32,
        ..Default::default()
    };
    // SAFETY: info is a correctly sized DWM_TIMING_INFO for our own window.
    if unsafe { DwmGetCompositionTimingInfo(hwnd, &mut info) }.is_ok() {
        // Copy the ratio out of the packed struct before touching its fields.
        let rate = info.rateRefresh;
        if let Some(period) = refresh_period_from_ratio(rate.uiNumerator, rate.uiDenominator) {
            return period;
        }
    }
    if let Some(period) = query_mode_refresh(device) {
        return period;
    }
    Duration::from_secs_f64(1.0 / 60.0)
}

/// Sleep until a deadline without overshooting by a whole timer tick: sleep in
/// coarse chunks while more than 1.5 ms remain, then yield-spin for the rest.
/// Only present() calls that are ahead of the display use this, and the spin is
/// at most 1.5 ms.
fn wait_until(deadline: Instant) {
    loop {
        let now = Instant::now();
        if now >= deadline {
            return;
        }
        let remain = deadline - now;
        if remain > Duration::from_micros(1500) {
            std::thread::sleep(remain - Duration::from_micros(1500));
        } else {
            std::thread::yield_now();
        }
    }
}

/// Convert one rgba frame into the BGRA layout GDI wants. The alpha byte is
/// written opaque because a display window is never blended.
#[inline]
fn rgba_to_bgra_into(src: &[u8], dst: &mut [u8]) {
    for (s, d) in src
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<4>().0.iter_mut())
    {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
        d[3] = 255;
    }
}

/// Top-down 32bpp BITMAPINFO describing the blitted buffer.
fn bitmap_info(dims: Dims) -> BITMAPINFO {
    let mut info = BITMAPINFO::default();
    info.bmiHeader.biSize = size_of::<BITMAPINFOHEADER>() as u32;
    info.bmiHeader.biWidth = dims.width as i32;
    info.bmiHeader.biHeight = -(dims.height as i32);
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB.0;
    info
}

/// Windows GDI window sink. Keep every call on the thread that opened it.
pub(crate) struct Win32Sink {
    window: OverlayWindow,
    hdc: HDC,
    dims: Dims,
    vsync: bool,
    refresh: Duration,
    last_present: Option<Instant>,
    bgra: Vec<u8>,
    info: BITMAPINFO,
    closed: bool,
}

// SAFETY: all fields are plain data plus Win32 handles. The engine moves the
// sink to the render thread before its first present, and the Win32 objects
// are only touched from that one thread afterwards (documented above).
unsafe impl Send for Win32Sink {}

impl Win32Sink {
    /// Wait until at least one display refresh has passed since the previous
    /// present. The wait is bounded by one refresh interval: when the engine is
    /// already late this returns immediately, so a slow frame never makes the
    /// following frame wait for two vblanks (QA defect D2).
    fn pace_refresh(&mut self) {
        let now = Instant::now();
        if let Some(last) = self.last_present {
            let deadline = last + self.refresh;
            if deadline > now {
                wait_until(deadline);
            }
        }
    }

    fn fill_bgra(&mut self, frame: &Frame) -> Result<()> {
        let needed = self.dims.area() as usize * 4;
        if self.bgra.len() != needed {
            self.bgra.resize(needed, 0);
        }
        match frame.fmt {
            PixelFormat::Rgba8 => rgba_to_bgra_into(&frame.data, &mut self.bgra),
            PixelFormat::Bgra8 => self.bgra.copy_from_slice(&frame.data),
            _ => {
                let rgba = frame.convert(PixelFormat::Rgba8);
                rgba_to_bgra_into(&rgba.data, &mut self.bgra);
            }
        }
        Ok(())
    }

    fn shutdown(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        if !self.hdc.is_invalid() {
            // SAFETY: releases the DC acquired by GetDC in open; the window is
            // still alive at this point.
            unsafe {
                let _ = ReleaseDC(self.window.hwnd(), self.hdc);
            }
            self.hdc = HDC::default();
        }
        self.window.destroy();
    }
}

impl Drop for Win32Sink {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl FrameSink for Win32Sink {
    fn dims(&self) -> Dims {
        self.dims
    }

    fn backend_name(&self) -> &'static str {
        "win32-window"
    }

    fn present(&mut self, frame: &Frame) -> Result<()> {
        if self.closed {
            return Err(FluidError::cancelled());
        }
        self.window.pump();
        if self.window.close_requested() {
            return Err(FluidError::cancelled());
        }
        if frame.dims != self.dims {
            if frame.dims.is_empty() {
                return Err(FluidError::invalid(
                    "present: frame dimensions must be non-zero",
                ));
            }
            self.dims = frame.dims;
            self.info = bitmap_info(self.dims);
            self.bgra.clear();
        }
        self.fill_bgra(frame)?;

        let mut client = RECT::default();
        // SAFETY: reads the client rectangle of our own window.
        unsafe { GetClientRect(self.window.hwnd(), &mut client) }
            .map_err(|e| platform("GetClientRect", e))?;
        let width = client.right - client.left;
        let height = client.bottom - client.top;
        if width <= 0 || height <= 0 {
            // Minimised: report success so the engine keeps rendering.
            return Ok(());
        }

        if self.vsync {
            self.pace_refresh();
        } else {
            self.last_present = None;
        }

        // SAFETY: hdc is our window DC, the buffer and BITMAPINFO outlive the
        // call, and the source size matches the frame we copied.
        let copied = unsafe {
            StretchDIBits(
                self.hdc,
                0,
                0,
                width,
                height,
                0,
                0,
                self.dims.width as i32,
                self.dims.height as i32,
                Some(self.bgra.as_ptr() as *const c_void),
                &self.info,
                DIB_RGB_COLORS,
                SRCCOPY,
            )
        };
        if copied == 0 || copied == -1 {
            return Err(FluidError::platform(
                "win32-window: StretchDIBits failed (device context lost?)",
            ));
        }
        if self.vsync {
            self.last_present = Some(Instant::now());
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

/// Open the Win32 window sink for the configured monitor.
pub(crate) fn open(config: &PresentConfig, dims: Dims) -> Result<Box<dyn FrameSink>> {
    if dims.is_empty() {
        return Err(FluidError::invalid(
            "present: frame dimensions must be non-zero",
        ));
    }
    let window = OverlayWindow::create(config, dims)?;
    // SAFETY: acquires the window DC; released in shutdown.
    let hdc = unsafe { GetDC(window.hwnd()) };
    if hdc.is_invalid() {
        return Err(FluidError::platform(
            "win32-window: GetDC returned no device context",
        ));
    }
    // SAFETY: COLORONCOLOR makes StretchDIBits use nearest-neighbour-ish
    // colour reduction instead of halftoning, which keeps text crisp.
    unsafe { SetStretchBltMode(hdc, COLORONCOLOR) };

    let refresh = query_refresh_period(window.hwnd(), window.device());
    Ok(Box::new(Win32Sink {
        window,
        hdc,
        dims,
        vsync: config.vsync,
        refresh,
        last_present: None,
        bgra: Vec::new(),
        info: bitmap_info(dims),
        closed: false,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refresh_period_uses_dwm_ratio() {
        let ntsc = refresh_period_from_ratio(60000, 1001).unwrap();
        assert!((ntsc.as_secs_f64() - 1001.0 / 60000.0).abs() < 1e-9);
        let hz90 = refresh_period_from_ratio(90, 1).unwrap();
        assert!((hz90.as_secs_f64() - 1.0 / 90.0).abs() < 1e-9);
        assert!(refresh_period_from_ratio(0, 1).is_none());
        assert!(refresh_period_from_ratio(1, 0).is_none());
        assert!(refresh_period_from_ratio(1, 1000).is_none());
    }

    #[test]
    fn rgba_rows_convert_to_bgra() {
        let src = [10u8, 20, 30, 255, 1, 2, 3, 40];
        let mut dst = [0u8; 8];
        rgba_to_bgra_into(&src, &mut dst);
        assert_eq!(dst, [30, 20, 10, 255, 3, 2, 1, 255]);
    }

    #[test]
    fn bitmap_info_is_top_down_32bpp() {
        let info = bitmap_info(Dims::new(640, 360));
        assert_eq!(info.bmiHeader.biWidth, 640);
        assert_eq!(info.bmiHeader.biHeight, -360);
        assert_eq!(info.bmiHeader.biPlanes, 1);
        assert_eq!(info.bmiHeader.biBitCount, 32);
        assert_eq!(info.bmiHeader.biCompression, BI_RGB.0);
        assert_eq!(
            info.bmiHeader.biSize as usize,
            size_of::<BITMAPINFOHEADER>()
        );
    }
}
