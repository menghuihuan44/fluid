//! Window-capture helpers: filter the capturable window list, render one
//! list row and build the fluid-core target that gets persisted.
//!
//! Everything here is pure so it can be unit tested without a desktop; the
//! win32 enumeration itself lives in fluid-capture.

use fluid_capture::WindowInfo;
use fluid_core::WindowTarget;

/// Friendly message when the selected window disappeared between refreshes.
pub const MISSING_WINDOW: &str = "选中的窗口已经关闭或不可见，请点“刷新列表”重新选择。";
/// Friendly message when the run starts without any window selected.
pub const NO_WINDOW: &str = "还没有选择窗口：先点“刷新列表”，再在列表里点一个要捕获的窗口。";
/// Shown on 实时增强 while a window capture keeps running after its window
/// closed or was minimized (QA defect D3): the pipeline keeps repeating the
/// last frame, so the user must be told instead of staring at a frozen picture.
pub const WINDOW_LOST: &str = "被捕获的窗口已经关闭或最小化：Fluid 还会继续重复最后一帧画面。请先停止增强，再点“刷新列表”重新选择窗口。";

/// Case-insensitive substring filter over the window title and process name.
///
/// An empty (or whitespace-only) needle means "show everything", so the list
/// is never empty just because the search box contains a stray space.
pub fn filter_windows<'a>(windows: &'a [WindowInfo], needle: &str) -> Vec<&'a WindowInfo> {
    let needle = needle.trim().to_lowercase();
    windows
        .iter()
        .filter(|info| {
            needle.is_empty()
                || info.title.to_lowercase().contains(&needle)
                || info.process_name.to_lowercase().contains(&needle)
        })
        .collect()
}

/// One selection-list row: title, process, size, and a minimized marker.
pub fn row_text(info: &WindowInfo) -> String {
    format!(
        "{} · {} · {}x{}{}",
        info.title,
        info.process_name,
        info.dims.width,
        info.dims.height,
        if info.minimized {
            "（已最小化）"
        } else {
            ""
        }
    )
}

/// One-line summary of the currently selected window.
pub fn summary_text(info: &WindowInfo) -> String {
    format!(
        "{} · {} · {}x{}",
        info.title, info.process_name, info.dims.width, info.dims.height
    )
}

/// Capture target for a window id.
///
/// Selection is always by id: title matches would be ambiguous when an app
/// (browser, editor, game launcher) has several equally named windows.
pub fn target_for(id: u64) -> WindowTarget {
    WindowTarget {
        id,
        ..WindowTarget::default()
    }
}

/// True when the id is still present in the enumerated list.
pub fn contains_id(windows: &[WindowInfo], id: u64) -> bool {
    windows.iter().any(|info| info.id == id)
}

/// Look up a window by id (used for the "已选" summary line).
pub fn find_by_id(windows: &[WindowInfo], id: u64) -> Option<&WindowInfo> {
    windows.iter().find(|info| info.id == id)
}

/// Notice to show on 实时增强 while a window-capture run lost its window.
///
/// A minimized window keeps the last frame, so it is treated like a closed one:
/// either way the preview freezes and the user has to re-select a window.
pub fn loss_notice(
    running: bool,
    selected: Option<u64>,
    windows: &[WindowInfo],
) -> Option<&'static str> {
    if !running {
        return None;
    }
    let id = selected?;
    match find_by_id(windows, id) {
        None => Some(WINDOW_LOST),
        Some(info) if info.minimized => Some(WINDOW_LOST),
        Some(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluid_core::Dims;

    fn info(id: u64, title: &str, process: &str, minimized: bool) -> WindowInfo {
        WindowInfo {
            id,
            title: title.to_owned(),
            class_name: "TestClass".to_owned(),
            process_name: process.to_owned(),
            dims: Dims::new(1280, 720),
            left: 0,
            top: 0,
            visible: true,
            minimized,
        }
    }

    fn sample() -> Vec<WindowInfo> {
        vec![
            info(0x11, "记事本 - 未命名", "notepad.exe", false),
            info(0x22, "Fluid · 帧生成与画质增强", "fluid-gui.exe", false),
            info(0x33, "Chrome", "chrome.exe", true),
        ]
    }

    #[test]
    fn empty_filter_keeps_every_window() {
        let windows = sample();
        assert_eq!(filter_windows(&windows, "").len(), 3);
        assert_eq!(filter_windows(&windows, "   ").len(), 3);
    }

    #[test]
    fn filter_matches_the_title_case_insensitively() {
        let windows = sample();
        let hits = filter_windows(&windows, "记事本");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 0x11);

        let upper = filter_windows(&windows, "FLUID");
        assert_eq!(upper.len(), 1);
        assert_eq!(upper[0].id, 0x22);
    }

    #[test]
    fn filter_matches_the_process_name() {
        let windows = sample();
        let hits = filter_windows(&windows, "chrome.exe");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 0x33);
        // The ".exe" suffix is optional for users typing a process name.
        assert_eq!(filter_windows(&windows, "notepad").len(), 1);
    }

    #[test]
    fn filter_without_matches_is_empty() {
        assert!(filter_windows(&sample(), "no such window").is_empty());
    }

    #[test]
    fn rows_show_title_process_and_size() {
        let windows = sample();
        let text = row_text(&windows[0]);
        assert!(text.contains("记事本"));
        assert!(text.contains("notepad.exe"));
        assert!(text.contains("1280x720"));
        assert!(!text.contains("已最小化"));
        assert!(row_text(&windows[2]).contains("已最小化"));
    }

    #[test]
    fn target_is_by_id_only() {
        let target = target_for(0x11);
        assert_eq!(target.id, 0x11);
        assert!(target.title.is_empty());
        assert!(target.process.is_empty());
        assert!(target.is_set());
        assert!(target.validate().is_ok());
    }

    #[test]
    fn contains_id_follows_the_list() {
        let windows = sample();
        assert!(contains_id(&windows, 0x22));
        assert!(!contains_id(&windows, 0x99));
        assert_eq!(
            find_by_id(&windows, 0x22).map(|w| w.title.clone()),
            Some("Fluid · 帧生成与画质增强".to_owned())
        );
        assert!(find_by_id(&windows, 0x99).is_none());
    }

    #[test]
    fn lost_window_raises_a_notice_while_running() {
        let windows = sample();
        let notice = loss_notice(true, Some(0x99), &windows);
        assert_eq!(notice, Some(WINDOW_LOST));
        assert!(WINDOW_LOST.contains("重新选择窗口"));
        assert!(WINDOW_LOST.contains("最后一帧"));
    }

    #[test]
    fn minimized_window_counts_as_lost() {
        let windows = sample();
        assert_eq!(loss_notice(true, Some(0x33), &windows), Some(WINDOW_LOST));
    }

    #[test]
    fn present_window_and_idle_runs_have_no_notice() {
        let windows = sample();
        assert_eq!(loss_notice(true, Some(0x11), &windows), None);
        // Idle: the 已选 row already says the window is gone, no banner needed.
        assert_eq!(loss_notice(false, Some(0x99), &windows), None);
        assert_eq!(loss_notice(true, None, &windows), None);
    }

    #[test]
    fn summary_names_the_window() {
        let windows = sample();
        let summary = summary_text(&windows[1]);
        assert!(summary.contains("Fluid"));
        assert!(summary.contains("fluid-gui.exe"));
    }
}
