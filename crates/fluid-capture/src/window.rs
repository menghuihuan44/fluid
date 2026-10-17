//! Window discovery and target-rectangle tracking.
//!
//! Only public Win32 window metadata is used. No target process handle is
//! opened and no process memory is read.

use fluid_core::config::WindowRef;
use std::sync::atomic::{AtomicIsize, Ordering};
use windows::Win32::Foundation::{BOOL, HWND, LPARAM, POINT, RECT};
use windows::Win32::Graphics::Dwm::{
    DwmGetWindowAttribute, DWMWA_CLOAKED, DWMWA_EXTENDED_FRAME_BOUNDS,
};
use windows::Win32::Graphics::Gdi::ClientToScreen;
use windows::Win32::System::Threading::GetCurrentProcessId;
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GetAncestor, GetClassNameW, GetClientRect, GetForegroundWindow, GetWindowLongPtrW,
    GetWindowRect, GetWindowTextW, GetWindowThreadProcessId, IsIconic, IsWindow, IsWindowVisible,
    GA_ROOT, GWL_EXSTYLE, WS_EX_TOOLWINDOW,
};

#[derive(Debug, Clone)]
pub struct WinEnum {
    pub hwnd: isize,
    pub title: String,
    pub class: String,
    pub rect: (i32, i32, i32, i32),
}

impl WinEnum {
    pub fn width(&self) -> i32 {
        self.rect.2 - self.rect.0
    }

    pub fn height(&self) -> i32 {
        self.rect.3 - self.rect.1
    }

    pub fn area(&self) -> i64 {
        self.width() as i64 * self.height() as i64
    }
}

pub fn hwnd_from_isize(v: isize) -> HWND {
    HWND(v as *mut core::ffi::c_void)
}

pub fn is_window(hwnd: HWND) -> bool {
    unsafe { IsWindow(hwnd).as_bool() }
}

pub fn is_iconic(hwnd: HWND) -> bool {
    unsafe { IsIconic(hwnd).as_bool() }
}

/// DWM extended frame bounds, falling back to GetWindowRect for windows that
/// do not expose DWM attributes (some games, launchers, and child render windows).
pub fn window_bounds(hwnd: HWND) -> Option<RECT> {
    let mut rect = RECT::default();
    let dwm_ok = unsafe {
        DwmGetWindowAttribute(
            hwnd,
            DWMWA_EXTENDED_FRAME_BOUNDS,
            &mut rect as *mut RECT as *mut core::ffi::c_void,
            std::mem::size_of::<RECT>() as u32,
        )
        .is_ok()
    };
    if !dwm_ok || rect.right <= rect.left || rect.bottom <= rect.top {
        rect = RECT::default();
        if unsafe { GetWindowRect(hwnd, &mut rect) }.is_err() {
            return None;
        }
    }
    (rect.right > rect.left && rect.bottom > rect.top).then_some(rect)
}

/// Kept as the public compatibility name used by the pipeline.
pub fn extended_frame_bounds(hwnd: HWND) -> Option<RECT> {
    window_bounds(hwnd)
}

fn is_cloaked(hwnd: HWND) -> bool {
    let mut cloaked: u32 = 0;
    unsafe {
        DwmGetWindowAttribute(
            hwnd,
            DWMWA_CLOAKED,
            &mut cloaked as *mut u32 as *mut core::ffi::c_void,
            std::mem::size_of::<u32>() as u32,
        )
        .is_ok()
            && cloaked != 0
    }
}

pub fn window_title(hwnd: HWND) -> String {
    let mut buf = [0u16; 512];
    let n = unsafe { GetWindowTextW(hwnd, &mut buf) };
    let n = n.max(0) as usize;
    String::from_utf16_lossy(&buf[..n.min(buf.len())])
}

pub fn window_class(hwnd: HWND) -> String {
    let mut buf = [0u16; 256];
    let n = unsafe { GetClassNameW(hwnd, &mut buf) };
    let n = n.max(0) as usize;
    String::from_utf16_lossy(&buf[..n.min(buf.len())])
}

/// Client-area rectangle in physical desktop coordinates. WGC captures window
/// content rather than the non-client frame, so this is the correct overlay rect.
pub fn client_rect_on_screen(hwnd: HWND) -> Option<RECT> {
    let mut client = RECT::default();
    if unsafe { GetClientRect(hwnd, &mut client) }.is_err() {
        return None;
    }
    let mut origin = POINT { x: 0, y: 0 };
    if !unsafe { ClientToScreen(hwnd, &mut origin) }.as_bool() {
        return None;
    }
    Some(RECT {
        left: origin.x,
        top: origin.y,
        right: origin.x + client.right - client.left,
        bottom: origin.y + client.bottom - client.top,
    })
}

pub fn window_process_id(hwnd: HWND) -> u32 {
    let mut pid = 0u32;
    unsafe { GetWindowThreadProcessId(hwnd, Some(&mut pid)) };
    pid
}

pub fn is_own_process_window(hwnd: HWND) -> bool {
    window_process_id(hwnd) == unsafe { GetCurrentProcessId() }
}

/// Foreground window normalized to its top-level root.
pub fn foreground_window() -> Option<HWND> {
    let fg = unsafe { GetForegroundWindow() };
    if fg.0.is_null() {
        return None;
    }
    let root = unsafe { GetAncestor(fg, GA_ROOT) };
    let target = if root.0.is_null() { fg } else { root };
    (is_window(target) && unsafe { IsWindowVisible(target).as_bool() }).then_some(target)
}

/// Build a selectable entry for one window, even when it was not produced by
/// the regular enumeration (for example, "use last foreground window").
pub fn window_info(hwnd: HWND) -> Option<WinEnum> {
    if !is_window(hwnd) || !unsafe { IsWindowVisible(hwnd).as_bool() } {
        return None;
    }
    if is_own_process_window(hwnd) {
        return None;
    }
    let rect = window_bounds(hwnd)?;
    let (w, h) = (rect.right - rect.left, rect.bottom - rect.top);
    if w < 80 || h < 60 {
        return None;
    }
    Some(WinEnum {
        hwnd: hwnd.0 as isize,
        title: window_title(hwnd),
        class: window_class(hwnd),
        rect: (rect.left, rect.top, rect.right, rect.bottom),
    })
}

unsafe extern "system" fn enum_cb(hwnd: HWND, lparam: LPARAM) -> BOOL {
    let list = &mut *(lparam.0 as *mut Vec<HWND>);
    list.push(hwnd);
    true.into()
}

/// Enumerate visible top-level windows. The filter is intentionally permissive:
/// games and video players often have empty titles, non-standard classes, or
/// tool-window styles. The UI sorts large/foreground windows first.
pub fn enumerate_windows() -> Vec<WinEnum> {
    let mut hwnds: Vec<HWND> = Vec::new();
    unsafe {
        let _ = EnumWindows(Some(enum_cb), LPARAM(&mut hwnds as *mut _ as isize));
    }

    let foreground = foreground_window();
    let mut out: Vec<(WinEnum, bool)> = Vec::new();
    for hwnd in hwnds {
        if is_own_process_window(hwnd) {
            continue;
        }
        if !unsafe { IsWindowVisible(hwnd).as_bool() } || is_iconic(hwnd) {
            continue;
        }
        if is_cloaked(hwnd) {
            continue;
        }

        let Some(rect) = window_bounds(hwnd) else {
            continue;
        };
        let w = rect.right - rect.left;
        let h = rect.bottom - rect.top;
        if w < 120 || h < 80 {
            continue;
        }

        let class = window_class(hwnd);
        let title = window_title(hwnd);
        if title.is_empty() && class.is_empty() {
            continue;
        }

        if matches!(
            class.as_str(),
            "Progman" | "WorkerW" | "Shell_TrayWnd" | "Shell_SecondaryTrayWnd"
        ) {
            continue;
        }
        if h < 120 {
            continue;
        }

        let ex_style = unsafe { GetWindowLongPtrW(hwnd, GWL_EXSTYLE) } as u32;
        let is_tool = ex_style & WS_EX_TOOLWINDOW.0 != 0;
        if is_tool && w < 640 && h < 360 {
            continue;
        }

        out.push((
            WinEnum {
                hwnd: hwnd.0 as isize,
                title,
                class,
                rect: (rect.left, rect.top, rect.right, rect.bottom),
            },
            foreground == Some(hwnd),
        ));
    }

    // Foreground first, then large windows. This makes fullscreen games and
    // videos appear near the top even if their title is empty.
    out.sort_by(|(a, a_fg), (b, b_fg)| {
        b_fg.cmp(a_fg)
            .then_with(|| b.area().cmp(&a.area()))
            .then_with(|| a.title.to_lowercase().cmp(&b.title.to_lowercase()))
    });
    out.into_iter().map(|(w, _)| w).collect()
}

/// Re-locate a remembered window by title + class, preferring the largest match.
pub fn find_window_by_ref(win: &WindowRef) -> Option<HWND> {
    let all = enumerate_windows();
    let mut best: Option<(&WinEnum, i64)> = None;
    for w in &all {
        let title_matches = win.title.is_empty() || w.title == win.title;
        if title_matches && (win.class.is_empty() || w.class == win.class) {
            let area = w.area();
            if best.is_none_or(|(_, a)| area > a) {
                best = Some((w, area));
            }
        }
    }
    best.map(|(w, _)| hwnd_from_isize(w.hwnd))
}

/// Return the currently active foreground root window if it is not fluid.
pub fn external_foreground_window() -> Option<HWND> {
    foreground_window().filter(|hwnd| !is_own_process_window(*hwnd))
}

static LAST_EXTERNAL_FOREGROUND: AtomicIsize = AtomicIsize::new(0);

/// Lightweight background watcher so foreground-window mode still knows the
/// last external window after the fluid settings window takes focus.
pub fn start_foreground_watcher() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("fluid-foreground-watch".into())
            .spawn(|| loop {
                if let Some(hwnd) = external_foreground_window() {
                    LAST_EXTERNAL_FOREGROUND.store(hwnd.0 as isize, Ordering::Relaxed);
                }
                std::thread::sleep(std::time::Duration::from_millis(150));
            });
    });
}

pub fn last_external_foreground_window() -> Option<HWND> {
    let value = LAST_EXTERNAL_FOREGROUND.load(Ordering::Relaxed);
    if value == 0 {
        return None;
    }
    let hwnd = hwnd_from_isize(value);
    window_info(hwnd).map(|info| hwnd_from_isize(info.hwnd))
}

pub fn monitor_of(hwnd: HWND) -> isize {
    crate::monitor::monitor_from_window(hwnd).0 as isize
}

/// Compute the target-window rectangle inside the captured output frame.
pub fn uv_rect(output: &RECT, window: &RECT) -> [f32; 4] {
    let ow = (output.right - output.left).max(1) as f32;
    let oh = (output.bottom - output.top).max(1) as f32;
    let l = (window.left - output.left).clamp(0, output.right - output.left) as f32;
    let t = (window.top - output.top).clamp(0, output.bottom - output.top) as f32;
    let r = (window.right - output.left).clamp(0, output.right - output.left) as f32;
    let b = (window.bottom - output.top).clamp(0, output.bottom - output.top) as f32;
    [l / ow, t / oh, r / ow, b / oh]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(l: i32, t: i32, r: i32, b: i32) -> RECT {
        RECT {
            left: l,
            top: t,
            right: r,
            bottom: b,
        }
    }

    #[test]
    fn uv_rect_identity_on_same_origin() {
        let out = rect(0, 0, 1920, 1080);
        let win = rect(0, 0, 1920, 1080);
        assert_eq!(uv_rect(&out, &win), [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn uv_rect_subwindow() {
        let out = rect(0, 0, 1920, 1080);
        let win = rect(960, 540, 1920, 1080);
        assert_eq!(uv_rect(&out, &win), [0.5, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn uv_rect_offset_output_and_clamp() {
        let out = rect(1920, 0, 3840, 1080);
        let win = rect(1600, -100, 3840, 1080);
        assert_eq!(uv_rect(&out, &win), [0.0, 0.0, 1.0, 1.0]);
    }
}
