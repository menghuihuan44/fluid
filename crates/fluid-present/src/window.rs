//! Shared Win32 overlay-window plumbing for the presentation backends.
//!
//! Both the GDI sink (win.rs) and the DXGI flip-model sink (swapchain.rs) show
//! the same kind of window: borderless and always-on-top by default, placed on
//! the configured monitor, sized to fit the frame into the monitor work area
//! (never enlarged, aspect preserved), closing on WM_CLOSE or on VK_ESCAPE
//! when close_on_escape is set. This module owns that shared piece so the two
//! backends only differ in how pixels reach the screen.
//!
//! Threading: create, pump, destroy and every other method must run on the
//! thread that opened the sink. The window class is registered per process and
//! reused by later sinks through the ERROR_CLASS_ALREADY_EXISTS path.

use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, Ordering};

use fluid_core::{Dims, FluidError, PresentConfig, Result};
use windows::Win32::Foundation::{
    BOOL, ERROR_CLASS_ALREADY_EXISTS, HINSTANCE, HWND, LPARAM, LRESULT, RECT, WPARAM,
};
use windows::Win32::Graphics::Gdi::{
    EnumDisplayMonitors, GetMonitorInfoW, HDC, HMONITOR, MONITORINFO, MONITORINFOEXW,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::HiDpi::{
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext,
};
use windows::Win32::UI::WindowsAndMessaging::{
    AdjustWindowRectEx, CS_HREDRAW, CS_VREDRAW, CreateWindowExW, DefWindowProcW, DestroyWindow,
    DispatchMessageW, GWLP_USERDATA, GetWindowLongPtrW, HMENU, MSG, PM_REMOVE, PeekMessageW,
    RegisterClassW, SW_SHOW, SetWindowLongPtrW, ShowWindow, TranslateMessage, WM_CLOSE, WM_DESTROY,
    WM_ERASEBKGND, WM_KEYDOWN, WM_QUIT, WM_SIZE, WNDCLASSW, WS_EX_TOPMOST, WS_OVERLAPPEDWINDOW,
    WS_POPUP, WS_VISIBLE,
};
use windows::core::{HRESULT, PCWSTR};

/// Virtual-key code for Escape (the keyboard feature is not enabled, so name
/// the one constant we need locally).
const VK_ESCAPE: usize = 0x1B;
/// Window class shared by every presentation window in this process.
const WINDOW_CLASS: &str = "FluidPresentWindow";

fn platform(what: &str, err: windows::core::Error) -> FluidError {
    FluidError::platform(format!("present window: {what}: {err}"))
}

/// Monitor rectangle in virtual-desktop coordinates, plus its work area (the
/// part not covered by the taskbar or other app bars) and its GDI device name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MonitorRect {
    pub(crate) left: i32,
    pub(crate) top: i32,
    pub(crate) width: i32,
    pub(crate) height: i32,
    pub(crate) work_left: i32,
    pub(crate) work_top: i32,
    pub(crate) work_width: i32,
    pub(crate) work_height: i32,
    /// NUL-terminated GDI device name such as DISPLAY1, used to query the
    /// configured display mode (and therefore the refresh rate).
    pub(crate) device: [u16; 32],
}

/// Shrink a w x h box to fit max_w x max_h while preserving its aspect ratio.
/// Never enlarges: a box that already fits is returned unchanged.
pub(crate) fn fit_aspect(w: i64, h: i64, max_w: i64, max_h: i64) -> (i32, i32) {
    if w <= 0 || h <= 0 || max_w <= 0 || max_h <= 0 {
        return (1, 1);
    }
    let (mut width, mut height) = (w, h);
    if w > max_w || h > max_h {
        // Compare aspect ratios without floating point: width-bound when
        // w/h >= max_w/max_h, which is the same as w*max_h >= max_w*h.
        if w * max_h >= max_w * h {
            width = max_w;
            height = (h * max_w / w).max(1);
        } else {
            height = max_h;
            width = (w * max_h / h).max(1);
        }
    }
    (width as i32, height as i32)
}

/// Window box for a frame. Fullscreen covers the whole monitor; windowed mode
/// fits the frame into the monitor work area at its native aspect ratio,
/// shrinking only when needed, and centres it there.
pub(crate) fn window_box(
    monitor: MonitorRect,
    fullscreen: bool,
    dims: Dims,
) -> (i32, i32, i32, i32) {
    if fullscreen {
        return (
            monitor.left,
            monitor.top,
            monitor.width.max(1),
            monitor.height.max(1),
        );
    }
    let work_w = monitor.work_width.max(1);
    let work_h = monitor.work_height.max(1);
    let (width, height) = fit_aspect(
        dims.width as i64,
        dims.height as i64,
        work_w as i64,
        work_h as i64,
    );
    let x = monitor.work_left + (work_w - width) / 2;
    let y = monitor.work_top + (work_h - height) / 2;
    (x, y, width, height)
}

pub(crate) fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0u16)).collect()
}

unsafe extern "system" fn collect_monitor(
    monitor: HMONITOR,
    _hdc: HDC,
    _clip: *mut RECT,
    data: LPARAM,
) -> BOOL {
    // SAFETY: EnumDisplayMonitors passes back the pointer we gave it, which
    // points at a live Vec<MonitorRect> for the duration of the enumeration.
    let rects = unsafe { &mut *(data.0 as *mut Vec<MonitorRect>) };
    let mut info = MONITORINFOEXW {
        monitorInfo: MONITORINFO {
            cbSize: size_of::<MONITORINFOEXW>() as u32,
            ..Default::default()
        },
        szDevice: [0u16; 32],
    };
    // SAFETY: MONITORINFOEXW starts with a MONITORINFO, which is what
    // GetMonitorInfoW writes; cbSize announces the larger struct so the device
    // name is filled in too.
    let ok = unsafe {
        GetMonitorInfoW(
            monitor,
            &mut info as *mut MONITORINFOEXW as *mut MONITORINFO,
        )
    }
    .as_bool();
    if ok {
        let monitor_rect = info.monitorInfo.rcMonitor;
        let work = info.monitorInfo.rcWork;
        let work_valid = work.right > work.left && work.bottom > work.top;
        rects.push(MonitorRect {
            left: monitor_rect.left,
            top: monitor_rect.top,
            width: monitor_rect.right - monitor_rect.left,
            height: monitor_rect.bottom - monitor_rect.top,
            work_left: if work_valid {
                work.left
            } else {
                monitor_rect.left
            },
            work_top: if work_valid {
                work.top
            } else {
                monitor_rect.top
            },
            work_width: if work_valid {
                work.right - work.left
            } else {
                monitor_rect.right - monitor_rect.left
            },
            work_height: if work_valid {
                work.bottom - work.top
            } else {
                monitor_rect.bottom - monitor_rect.top
            },
            device: info.szDevice,
        });
    }
    BOOL(1)
}

/// All monitors ordered by desktop position, so monitor indices agree with the
/// capture crate's enumeration.
pub(crate) fn monitor_rects() -> Vec<MonitorRect> {
    let mut rects: Vec<MonitorRect> = Vec::with_capacity(4);
    // SAFETY: the callback only writes through the pointer we pass and lives
    // for the duration of the EnumDisplayMonitors call.
    let _ = unsafe {
        EnumDisplayMonitors(
            HDC::default(),
            None,
            Some(collect_monitor),
            LPARAM(&mut rects as *mut Vec<MonitorRect> as isize),
        )
    };
    rects.sort_by_key(|m| (m.left, m.top));
    rects
}

/// Shared mutable state behind the window user-data pointer.
pub(crate) struct WindowState {
    closed: AtomicBool,
    close_on_escape: bool,
    resized: AtomicBool,
}

fn state_of(hwnd: HWND) -> *mut WindowState {
    // SAFETY: reads the user-data slot of our own window; null until open
    // installs the state, which wndproc tolerates.
    unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut WindowState }
}

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    let state = state_of(hwnd);
    match msg {
        WM_CLOSE => {
            if !state.is_null() {
                // SAFETY: state points at the WindowState owned by the sink.
                unsafe { (*state).closed.store(true, Ordering::Release) };
            }
            LRESULT(0)
        }
        WM_DESTROY => {
            if !state.is_null() {
                // SAFETY: state points at the WindowState owned by the sink.
                unsafe { (*state).closed.store(true, Ordering::Release) };
            }
            // No PostQuitMessage here on purpose: WM_QUIT is queued per thread,
            // so a window that is torn down (for example after a failed backend
            // attempt) would otherwise poison the next window's message pump.
            // Closing is signalled through the state flag, which the pump and
            // should_close read.
            LRESULT(0)
        }
        WM_KEYDOWN => {
            let escape = wparam.0 == VK_ESCAPE;
            // SAFETY: state points at the WindowState owned by the sink.
            let enabled = !state.is_null() && unsafe { (*state).close_on_escape };
            if escape && enabled {
                // SAFETY: state points at the WindowState owned by the sink.
                unsafe { (*state).closed.store(true, Ordering::Release) };
                LRESULT(0)
            } else {
                // SAFETY: forwarding an unhandled message to the default proc.
                unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
            }
        }
        WM_SIZE => {
            if !state.is_null() {
                // SAFETY: state points at the WindowState owned by the sink.
                unsafe { (*state).resized.store(true, Ordering::Release) };
            }
            // SAFETY: forwarding an unhandled message to the default proc.
            unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
        }
        WM_ERASEBKGND => LRESULT(1),
        _ => {
            // SAFETY: forwarding an unhandled message to the default proc.
            unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
        }
    }
}

/// An owned Win32 overlay window plus its shared state.
pub(crate) struct OverlayWindow {
    hwnd: HWND,
    state: *mut WindowState,
    device: [u16; 32],
}

impl OverlayWindow {
    /// Create and show the overlay window for the configured monitor.
    pub(crate) fn create(config: &PresentConfig, dims: Dims) -> Result<Self> {
        // Best effort: per-monitor DPI awareness keeps the window from being
        // bitmap-stretched by the OS. It fails harmlessly if already set.
        // SAFETY: process-wide configuration call with no memory arguments.
        let _ =
            unsafe { SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) };

        let monitors = monitor_rects();
        if monitors.is_empty() {
            return Err(FluidError::platform(
                "present window: EnumDisplayMonitors reported no monitors (headless session?)",
            ));
        }
        let monitor = monitors[config.monitor.min(monitors.len() - 1)];
        let (x, y, width, height) = window_box(monitor, config.fullscreen, dims);
        let style = if config.fullscreen || config.borderless {
            WS_POPUP | WS_VISIBLE
        } else {
            WS_OVERLAPPEDWINDOW | WS_VISIBLE
        };
        let ex_style = WS_EX_TOPMOST;
        let mut rect = RECT {
            left: x,
            top: y,
            right: x + width,
            bottom: y + height,
        };
        if !config.fullscreen && !config.borderless {
            // Grow the requested client box to include the frame decorations.
            // SAFETY: rect is a valid RECT; the style values are the ones
            // passed to CreateWindowExW below.
            unsafe {
                let _ = AdjustWindowRectEx(&mut rect, style, BOOL(0), ex_style);
            };
        }

        let class = wide(WINDOW_CLASS);
        let title = wide(&config.title);
        // SAFETY: fetching this process's module handle takes no arguments.
        let module = unsafe { GetModuleHandleW(PCWSTR::null()) }
            .map_err(|e| platform("GetModuleHandleW", e))?;
        let instance = HINSTANCE(module.0);
        let window_class = WNDCLASSW {
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(wndproc),
            hInstance: instance,
            lpszClassName: PCWSTR(class.as_ptr()),
            ..Default::default()
        };
        // SAFETY: window_class and the class string outlive the call.
        let atom = unsafe { RegisterClassW(&window_class) };
        if atom == 0 {
            let err = windows::core::Error::from_win32();
            // A second sink in the same process reuses the registered class.
            if err.code() != HRESULT::from_win32(ERROR_CLASS_ALREADY_EXISTS.0) {
                return Err(platform("RegisterClassW", err));
            }
        }

        let state = Box::into_raw(Box::new(WindowState {
            closed: AtomicBool::new(false),
            close_on_escape: config.close_on_escape,
            resized: AtomicBool::new(false),
        }));
        // SAFETY: class, title and instance are valid for the call; the state
        // pointer is reclaimed by destroy or by the error path below.
        let hwnd = match unsafe {
            CreateWindowExW(
                ex_style,
                PCWSTR(class.as_ptr()),
                PCWSTR(title.as_ptr()),
                style,
                rect.left,
                rect.top,
                rect.right - rect.left,
                rect.bottom - rect.top,
                None::<&HWND>,
                None::<&HMENU>,
                instance,
                Some(state as *const c_void),
            )
        } {
            Ok(hwnd) => hwnd,
            Err(e) => {
                // SAFETY: reclaims the allocation made just above; no window
                // owns it because creation failed.
                unsafe { drop(Box::from_raw(state)) };
                return Err(platform("CreateWindowExW", e));
            }
        };
        // SAFETY: installs our state pointer in our own window's user-data
        // slot; the pointer is reclaimed by destroy.
        unsafe { SetWindowLongPtrW(hwnd, GWLP_USERDATA, state as isize) };
        // SAFETY: shows the window we just created.
        unsafe {
            let _ = ShowWindow(hwnd, SW_SHOW);
        }
        Ok(Self {
            hwnd,
            state,
            device: monitor.device,
        })
    }

    pub(crate) fn hwnd(&self) -> HWND {
        self.hwnd
    }

    /// GDI device name of the monitor this window was placed on.
    pub(crate) fn device(&self) -> &[u16; 32] {
        &self.device
    }

    pub(crate) fn request_close(&self) {
        if !self.state.is_null() {
            // SAFETY: state points at the WindowState owned by this window.
            unsafe { (*self.state).closed.store(true, Ordering::Release) };
        }
    }

    pub(crate) fn close_requested(&self) -> bool {
        if self.state.is_null() {
            return true;
        }
        // SAFETY: state points at the WindowState owned by this window.
        unsafe { (*self.state).closed.load(Ordering::Acquire) }
    }

    /// Consume the pending WM_SIZE notification, if any.
    pub(crate) fn take_resized(&self) -> bool {
        if self.state.is_null() {
            return false;
        }
        // SAFETY: state points at the WindowState owned by this window.
        unsafe { (*self.state).resized.swap(false, Ordering::AcqRel) }
    }

    /// Drain the thread's message queue without blocking.
    pub(crate) fn pump(&self) {
        let mut msg = MSG::default();
        loop {
            // SAFETY: msg is a valid MSG; a null HWND filter reads the thread
            // queue, and PM_REMOVE takes each message out of it.
            let has = unsafe { PeekMessageW(&mut msg, None::<&HWND>, 0, 0, PM_REMOVE) };
            if !has.as_bool() {
                break;
            }
            if msg.message == WM_QUIT {
                self.request_close();
                break;
            }
            // SAFETY: translating and dispatching a message we just removed.
            unsafe {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    }

    /// Destroy the window and free the shared state. Idempotent.
    pub(crate) fn destroy(&mut self) {
        if !self.hwnd.is_invalid() {
            // SAFETY: destroys our own window; its wndproc tolerates the state
            // pointer still being alive during WM_DESTROY.
            unsafe {
                let _ = DestroyWindow(self.hwnd);
            }
            self.hwnd = HWND::default();
        }
        if !self.state.is_null() {
            // SAFETY: reclaims the Box::into_raw allocation made in create.
            unsafe { drop(Box::from_raw(self.state)) };
            self.state = null_mut();
        }
    }
}

impl Drop for OverlayWindow {
    fn drop(&mut self) {
        self.destroy();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_monitor() -> MonitorRect {
        MonitorRect {
            left: 1920,
            top: 0,
            width: 2560,
            height: 1440,
            work_left: 1920,
            work_top: 0,
            work_width: 2560,
            work_height: 1400,
            device: [0u16; 32],
        }
    }

    #[test]
    fn window_box_fullscreen_covers_monitor() {
        assert_eq!(
            window_box(test_monitor(), true, Dims::new(640, 360)),
            (1920, 0, 2560, 1440)
        );
    }

    #[test]
    fn window_box_fits_work_area_and_centres() {
        // Fits: native size, centred inside the work area (not the monitor).
        assert_eq!(
            window_box(test_monitor(), false, Dims::new(1280, 720)),
            (2560, 340, 1280, 720)
        );
        // Small frames are never enlarged.
        assert_eq!(
            window_box(test_monitor(), false, Dims::new(320, 180)),
            (3040, 610, 320, 180)
        );
        // 4K on a 2880x1800 screen: scaled down to the work area, aspect kept.
        assert_eq!(
            window_box(test_monitor(), false, Dims::new(3840, 2160)),
            (1956, 0, 2488, 1400)
        );
        // 4:3 oversized request is height-bound.
        assert_eq!(
            window_box(test_monitor(), false, Dims::new(4000, 3000)),
            (2267, 0, 1866, 1400)
        );
        // Degenerate frame still gets a valid 1x1 box.
        let small = MonitorRect {
            left: 0,
            top: 0,
            width: 100,
            height: 50,
            work_left: 0,
            work_top: 0,
            work_width: 100,
            work_height: 50,
            device: [0u16; 32],
        };
        assert_eq!(window_box(small, false, Dims::new(0, 0)), (49, 24, 1, 1));
    }

    #[test]
    fn fit_aspect_rejects_degenerate_input() {
        assert_eq!(fit_aspect(0, 10, 100, 100), (1, 1));
        assert_eq!(fit_aspect(10, 0, 100, 100), (1, 1));
        assert_eq!(fit_aspect(10, 10, 0, 100), (1, 1));
        assert_eq!(fit_aspect(10, 10, 100, 100), (10, 10));
    }

    #[test]
    fn wide_strings_are_nul_terminated() {
        let w = wide("Fluid");
        assert_eq!(
            w,
            vec![
                b'F' as u16,
                b'l' as u16,
                b'u' as u16,
                b'i' as u16,
                b'd' as u16,
                0
            ]
        );
    }
}
