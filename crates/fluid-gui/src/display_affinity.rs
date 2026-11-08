//! Keep the fullscreen output window invisible to screen capture.
//!
//! The 全屏输出 viewport shows the enhanced picture full screen while the game
//! keeps rendering on the desktop. On a single monitor the DXGI desktop
//! duplication backend used for capture would see that window and feed it back
//! into the pipeline - the classic "infinite mirror" (hall of mirrors) effect.
//!
//! Windows 10 2004 and newer know SetWindowDisplayAffinity(hwnd,
//! WDA_EXCLUDEFROMCAPTURE = 0x11): the window is still displayed on the
//! physical monitor, but every screen-capture API (DXGI duplication, BitBlt,
//! PrintWindow, ...) treats it as if it were not there. Older Windows builds
//! reject the call; in that case we report the failure and the UI keeps the
//! "use a second monitor / borderless window" advice.
//!
//! The lookup is a plain EnumWindows + title match scoped to our own PID,
//! because eframe does not hand us the winit window of a second viewport.

/// Window title of the second, chrome-less fullscreen viewport.
pub(crate) const OUTPUT_WINDOW_TITLE: &str = "Fluid 全屏输出";

/// Result of trying to exclude the output window from screen capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Affinity {
    /// The window is excluded from capture: the mirror loop is gone.
    Excluded,
    /// The viewport window was not found (yet); the caller may retry.
    NotFound,
    /// Windows rejected the call, usually because it predates 10 2004.
    Unsupported(u32),
    /// Built for a non-Windows target: there is nothing to exclude.
    NotWindows,
}

impl Affinity {
    /// True when the exclusion is in place.
    pub(crate) fn is_excluded(self) -> bool {
        matches!(self, Affinity::Excluded)
    }

    /// One-line log record, e.g. "output window excluded from capture: true".
    pub(crate) fn log_line(self) -> String {
        match self {
            Affinity::Excluded => "output window excluded from capture: true".to_string(),
            Affinity::NotFound => {
                "output window excluded from capture: false (window not found)".to_string()
            }
            Affinity::Unsupported(code) => {
                format!("output window excluded from capture: false (error 0x{code:08X})")
            }
            Affinity::NotWindows => {
                "output window excluded from capture: false (not windows)".to_string()
            }
        }
    }

    /// Beginner-friendly, grey hint shown under the 全屏输出 checkbox.
    pub(crate) fn hint(self) -> &'static str {
        match self {
            Affinity::Excluded => {
                "已阻止屏幕捕获：屏幕捕获和录屏软件都看不到这个全屏窗口，因此不会再出现「无限镜像」。"
            }
            _ => {
                "这个系统不允许把全屏窗口从屏幕捕获中隐藏（需要 Windows 10 2004 或更新）。可以把全屏输出窗口拖到第二块屏幕，或让游戏用「无边框窗口」模式运行。"
            }
        }
    }
}

/// True when a window caption contains the needle (case sensitive).
pub(crate) fn matches_title(title: &str, needle: &str) -> bool {
    !needle.is_empty() && title.contains(needle)
}

/// Pick the first window whose PID and title match; returns the raw handle bits.
///
/// Split out from the Win32 code so it can be unit tested with fake windows.
pub(crate) fn pick_window(
    candidates: impl Iterator<Item = (u32, usize, String)>,
    pid: u32,
    needle: &str,
) -> Option<usize> {
    candidates
        .filter(|(owner, _, _)| *owner == pid)
        .find(|(_, _, title)| matches_title(title, needle))
        .map(|(_, handle, _)| handle)
}

/// Find the fullscreen output window and exclude it from screen capture.
///
/// Safe to call repeatedly: returns Affinity::NotFound until the viewport
/// window exists, and the excluded/failed result once it does.
#[cfg(windows)]
pub(crate) fn exclude_output_window() -> Affinity {
    use windows::Win32::Foundation::{BOOL, HWND, LPARAM};
    use windows::Win32::UI::WindowsAndMessaging::{
        EnumWindows, GetWindowDisplayAffinity, GetWindowTextW, GetWindowThreadProcessId,
        IsWindowVisible, SetWindowDisplayAffinity, WDA_EXCLUDEFROMCAPTURE,
    };

    /// Scratch space for the enumeration callback.
    struct Collect {
        found: Vec<(u32, usize, String)>,
    }

    unsafe extern "system" fn collect(hwnd: HWND, lparam: LPARAM) -> BOOL {
        // SAFETY: EnumWindows calls this synchronously with the lparam we
        // passed in, which points at a live Collect on this thread's stack.
        let state = unsafe { &mut *(lparam.0 as *mut Collect) };
        // SAFETY: read-only query for the handle we were just given.
        if unsafe { IsWindowVisible(hwnd) } == BOOL(0) {
            return BOOL(1);
        }
        let mut buffer = [0u16; 256];
        // SAFETY: buffer is a valid writable slice for the caption.
        let len = unsafe { GetWindowTextW(hwnd, &mut buffer) };
        if len <= 0 {
            return BOOL(1);
        }
        let title = String::from_utf16_lossy(&buffer[..len as usize]);
        let mut pid = 0u32;
        // SAFETY: pid is a live local written by the call.
        unsafe { GetWindowThreadProcessId(hwnd, Some(&mut pid as *mut u32)) };
        state.found.push((pid, hwnd.0 as usize, title));
        BOOL(1)
    }

    let mut state = Collect { found: Vec::new() };
    // SAFETY: the callback and the context pointer stay valid for the call.
    let enumerated =
        unsafe { EnumWindows(Some(collect), LPARAM(&mut state as *mut Collect as isize)) }.is_ok();
    if !enumerated {
        return Affinity::NotFound;
    }

    let pid = std::process::id();
    let Some(handle) = pick_window(state.found.into_iter(), pid, OUTPUT_WINDOW_TITLE) else {
        return Affinity::NotFound;
    };
    let hwnd = HWND(handle as *mut core::ffi::c_void);

    // SAFETY: the handle came from EnumWindows a moment ago; the call only
    // updates window metadata and fails cleanly for stale handles.
    match unsafe { SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE) } {
        Ok(()) => {
            let mut current = 0u32;
            // SAFETY: current is a live local written by the call.
            let verified = unsafe { GetWindowDisplayAffinity(hwnd, &mut current as *mut u32) }
                .is_ok()
                && current == WDA_EXCLUDEFROMCAPTURE.0;
            if verified {
                Affinity::Excluded
            } else {
                Affinity::Unsupported(0)
            }
        }
        Err(err) => Affinity::Unsupported(err.code().0 as u32),
    }
}

/// Non-Windows stub: there is no WDA_EXCLUDEFROMCAPTURE equivalent to call.
#[cfg(not(windows))]
pub(crate) fn exclude_output_window() -> Affinity {
    Affinity::NotWindows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_match_is_contains_based() {
        assert!(matches_title("Fluid 全屏输出", "Fluid 全屏输出"));
        assert!(matches_title("Fluid 全屏输出 - 1024x768", "全屏输出"));
        assert!(!matches_title("Fluid · 帧生成与画质增强", "全屏输出"));
        assert!(!matches_title("Fluid 全屏输出", ""));
    }

    #[test]
    fn window_picker_filters_by_pid() {
        let candidates = vec![
            (7u32, 0x11usize, "Fluid 全屏输出".to_string()),
            (42, 0x22, "Fluid 全屏输出".to_string()),
        ];
        assert_eq!(
            pick_window(candidates.into_iter(), 42, "全屏输出"),
            Some(0x22)
        );
    }

    #[test]
    fn window_picker_ignores_other_titles() {
        let candidates = vec![
            (42u32, 0x11usize, "Fluid · 帧生成与画质增强".to_string()),
            (42, 0x22, "记事本".to_string()),
        ];
        assert_eq!(pick_window(candidates.into_iter(), 42, "全屏输出"), None);
    }

    #[test]
    fn window_picker_is_first_match() {
        let candidates = vec![
            (42u32, 0x11usize, "Fluid 全屏输出".to_string()),
            (42, 0x22, "Fluid 全屏输出".to_string()),
        ];
        assert_eq!(
            pick_window(candidates.into_iter(), 42, "全屏输出"),
            Some(0x11)
        );
    }

    #[test]
    fn log_lines_are_greppable() {
        assert_eq!(
            Affinity::Excluded.log_line(),
            "output window excluded from capture: true"
        );
        assert!(
            Affinity::Unsupported(0x8007_0057).log_line()
                == "output window excluded from capture: false (error 0x80070057)"
        );
        assert!(
            Affinity::NotFound
                .log_line()
                .starts_with("output window excluded from capture: false")
        );
        assert!(Affinity::Excluded.is_excluded());
        assert!(!Affinity::Unsupported(5).is_excluded());
    }

    #[test]
    fn hints_explain_the_fallback() {
        assert!(Affinity::Excluded.hint().contains("无限镜像"));
        assert!(Affinity::Unsupported(5).hint().contains("第二块屏幕"));
        assert!(Affinity::NotWindows.hint().contains("第二块屏幕"));
    }
}
