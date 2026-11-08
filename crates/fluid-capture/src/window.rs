//! Window capture types and platform independent helpers.
//!
//! A window source captures one application window instead of a whole monitor.
//! The public entry points live in the crate root ("enumerate_windows" and
//! "open_window"); this module owns the data types plus the pure selection and
//! geometry logic so it can be unit tested on every platform, including CI.
//!
//! Geometry: "WindowInfo" reports the window frame rectangle in physical
//! desktop pixels (negative origins are normal on a secondary monitor that
//! sits left of or above the primary one). "CaptureConfig::region" is
//! interpreted as a crop *inside* the window: it is clipped to the captured
//! content and "FrameSource::dims" reports the clipped size. Window sources
//! always produce "Rgba8" frames.
//!
//! Follow behaviour: a window source tracks its target window. Moving the
//! window does not change the captured content. Resizing the window keeps the
//! output dimensions fixed at the size reported when the source was opened
//! (the engine builds its pipeline from that size); content that no longer
//! fits is cropped, content that became smaller is padded black. Reopen the
//! source to adopt a new size.
//!
//! Liveness: a live window source never returns "Ok(None)". When the window is
//! minimized, occluded, not repainting or already closed, the previous frame
//! is replayed with a fresh timestamp and sequence number (see "replay").

use std::fmt;

// "FrameSource" is used by the fake window source in the test module below, so it
// is only needed when tests are compiled; allow the import in library builds.
#[cfg_attr(not(test), allow(unused_imports))]
use fluid_core::{Dims, FluidError, Frame, FrameSource, PixelFormat, Rect, Result};

/// A top-level application window that can be captured.
///
/// "id" is the native window handle stored as an integer; it is stable for the
/// lifetime of the window and is what "WindowSelector::id" expects.
/// "dims", "left" and "top" describe the window frame rectangle in physical
/// desktop pixels (the frame includes the title bar and borders; the captured
/// content can be slightly smaller, ask the opened source for its exact size).
/// "process_name" is the executable file name, for example "notepad.exe".
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowInfo {
    pub id: u64,
    pub title: String,
    pub class_name: String,
    pub process_name: String,
    pub dims: Dims,
    pub left: i32,
    pub top: i32,
    pub visible: bool,
    pub minimized: bool,
}

impl fmt::Display for WindowInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} {}x{} at ({}, {}) ({}){}",
            self.id,
            self.title,
            self.dims.width,
            self.dims.height,
            self.left,
            self.top,
            self.process_name,
            if self.minimized { " (minimized)" } else { "" }
        )
    }
}

/// Which window "open_window" should capture.
///
/// "id" always wins when set. Otherwise every set field must match ("and"):
/// "title_contains" is a case insensitive substring of the window title and
/// "process" matches the executable name, with or without the ".exe" suffix.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WindowSelector {
    pub id: Option<u64>,
    pub title_contains: Option<String>,
    pub process: Option<String>,
}

/// Rectangle in signed desktop or window local pixel coordinates.
///
/// "fluid_core::Rect" uses unsigned coordinates because it describes a frame
/// buffer. Window geometry can be negative (a monitor placed left of the
/// primary one), so window capture uses this signed variant internally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PixelRect {
    pub(crate) left: i32,
    pub(crate) top: i32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl PixelRect {
    pub(crate) const fn new(left: i32, top: i32, width: u32, height: u32) -> Self {
        Self {
            left,
            top,
            width,
            height,
        }
    }

    pub(crate) const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    pub(crate) fn right(&self) -> i64 {
        self.left as i64 + self.width as i64
    }

    pub(crate) fn bottom(&self) -> i64 {
        self.top as i64 + self.height as i64
    }

    pub(crate) fn dims(&self) -> Dims {
        Dims::new(self.width, self.height)
    }

    /// Largest rectangle contained in both rectangles; zero sized when they do
    /// not overlap. Uses 64 bit intermediates so huge virtual desktops cannot
    /// wrap around.
    pub(crate) fn intersect(&self, other: &PixelRect) -> PixelRect {
        let left = self.left.max(other.left);
        let top = self.top.max(other.top);
        let right = self.right().min(other.right());
        let bottom = self.bottom().min(other.bottom());
        if right <= left as i64 || bottom <= top as i64 {
            return PixelRect::new(left, top, 0, 0);
        }
        PixelRect::new(
            left,
            top,
            (right - left as i64) as u32,
            (bottom - top as i64) as u32,
        )
    }
}

/// Resolve the capture rectangle inside a window (window local coordinates).
///
/// Returns the clipped crop and its dimensions. A region that does not overlap
/// the captured content is a configuration error naming both rectangles.
pub(crate) fn capture_region(content: Dims, region: Option<Rect>) -> Result<(Rect, Dims)> {
    if content.is_empty() {
        return Err(FluidError::invalid(
            "window has no capturable content (it may be minimized or fully off screen)",
        ));
    }
    match region {
        None => Ok((Rect::from_dims(content), content)),
        Some(requested) => {
            let clipped = requested.intersect(content);
            if clipped.is_empty() {
                return Err(FluidError::invalid(format!(
                    "window capture region {}x{} at ({}, {}) does not intersect the {} window content",
                    requested.w, requested.h, requested.x, requested.y, content
                )));
            }
            Ok((clipped, clipped.dims()))
        }
    }
}

/// Map a window local crop to desktop coordinates: frame origin plus offset.
pub(crate) fn screen_rect(frame: PixelRect, region: Rect) -> PixelRect {
    PixelRect::new(
        frame.left.saturating_add(region.x as i32),
        frame.top.saturating_add(region.y as i32),
        region.w,
        region.h,
    )
}

/// Re-emit "cached" with a fresh timestamp and sequence number.
///
/// Used when the window produced no new content (minimized, static, closed or
/// temporarily unreadable) so the engine sees a normal frame instead of the end
/// of the stream. A cached frame of the wrong shape is replaced by black to
/// keep the invariant that every returned frame has "dims".
pub(crate) fn replay(cached: &Frame, dims: Dims, pts_ns: i64, index: u64) -> Frame {
    let expected = dims.area() as usize * 4;
    let data = if cached.dims == dims && cached.data.len() == expected {
        cached.data.clone()
    } else {
        vec![0u8; expected]
    };
    Frame {
        dims,
        fmt: PixelFormat::Rgba8,
        data,
        pts_ns,
        index,
    }
}

/// Case insensitive substring match used by "WindowSelector::title_contains".
fn title_matches(title: &str, needle: &str) -> bool {
    let title = title.to_lowercase();
    let needle = needle.to_lowercase();
    title.contains(&needle)
}

/// Executable name comparison used by "WindowSelector::process".
///
/// Accepts the file name with or without the ".exe" suffix in either side, so
/// "notepad" matches "notepad.exe" and "Notepad.EXE" matches "notepad".
fn process_matches(candidate: &str, needle: &str) -> bool {
    let candidate = candidate.trim();
    let needle = needle.trim();
    if candidate.eq_ignore_ascii_case(needle) {
        return true;
    }
    let strip = |name: &str| {
        let name = name.to_lowercase();
        name.strip_suffix(".exe").unwrap_or(&name).to_string()
    };
    !candidate.is_empty() && strip(candidate) == strip(needle)
}

/// Pick the single window described by a selector.
///
/// With an explicit id the lookup is exact. With title/process filters every
/// set filter must match; exactly one match is returned, an exact title match
/// wins over partial matches, and several remaining candidates produce an
/// error that lists them so the caller can retry with an id.
pub(crate) fn select_window<'a>(
    windows: &'a [WindowInfo],
    selector: &WindowSelector,
) -> Result<&'a WindowInfo> {
    if let Some(id) = selector.id {
        return windows.iter().find(|window| window.id == id).ok_or_else(|| {
            FluidError::invalid(format!(
                "window id {id} is not in the current window list ({} windows visible); list the windows and pick another id",
                windows.len()
            ))
        });
    }
    let title = selector
        .title_contains
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty());
    let process = selector
        .process
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty());
    if title.is_none() && process.is_none() {
        return Err(FluidError::invalid(
            "window selector needs an id, a title substring or a process name",
        ));
    }
    let candidates: Vec<&WindowInfo> = windows
        .iter()
        .filter(|window| title.is_none_or(|needle| title_matches(&window.title, needle)))
        .filter(|window| process.is_none_or(|needle| process_matches(&window.process_name, needle)))
        .collect();
    match candidates.len() {
        0 => Err(FluidError::invalid(format!(
            "no visible window matches title {:?} / process {:?} ({} windows enumerated)",
            title.unwrap_or(""),
            process.unwrap_or(""),
            windows.len()
        ))),
        1 => Ok(candidates[0]),
        _ => {
            // A full title match is an unambiguous choice even if other titles
            // merely contain the same text.
            if let Some(needle) = title
                && let Some(exact) = candidates
                    .iter()
                    .find(|window| window.title.eq_ignore_ascii_case(needle))
            {
                return Ok(exact);
            }
            let listing = candidates
                .iter()
                .take(8)
                .map(|window| {
                    format!(
                        "#{} {:?} ({})",
                        window.id, window.title, window.process_name
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            Err(FluidError::invalid(format!(
                "{} windows match the selector; retry with an explicit window id: {listing}",
                candidates.len()
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::mock_frame;

    fn info(id: u64, title: &str, process: &str) -> WindowInfo {
        WindowInfo {
            id,
            title: title.to_string(),
            class_name: "TestClass".to_string(),
            process_name: process.to_string(),
            dims: Dims::new(320, 240),
            left: 0,
            top: 0,
            visible: true,
            minimized: false,
        }
    }

    fn selector(id: Option<u64>, title: Option<&str>, process: Option<&str>) -> WindowSelector {
        WindowSelector {
            id,
            title_contains: title.map(str::to_string),
            process: process.map(str::to_string),
        }
    }

    #[test]
    fn window_info_formats_for_listings() {
        let mut window = info(4660, "Untitled - Notepad", "notepad.exe");
        window.left = -1920;
        window.top = 64;
        assert_eq!(
            window.to_string(),
            "[4660] Untitled - Notepad 320x240 at (-1920, 64) (notepad.exe)"
        );
        window.minimized = true;
        assert!(window.to_string().ends_with("(minimized)"));
    }

    #[test]
    fn selector_requires_a_criterion() {
        let windows = [info(1, "A", "a.exe")];
        assert!(select_window(&windows, &WindowSelector::default()).is_err());
        let blank = selector(None, Some("   "), Some(""));
        assert!(select_window(&windows, &blank).is_err());
    }

    #[test]
    fn selector_by_id_wins_over_filters() {
        let windows = [info(10, "Alpha", "a.exe"), info(20, "Beta", "b.exe")];
        let picked = select_window(&windows, &selector(Some(20), Some("Alpha"), None)).unwrap();
        assert_eq!(picked.id, 20);
        // A missing id is reported clearly.
        let error = select_window(&windows, &selector(Some(99), None, None)).unwrap_err();
        assert!(error.to_string().contains("99"), "{error}");
    }

    #[test]
    fn selector_matches_titles_case_insensitively() {
        let windows = [
            info(1, "Untitled - Notepad", "notepad.exe"),
            info(2, "Visual Studio Code", "Code.exe"),
        ];
        let picked = select_window(&windows, &selector(None, Some("notepad"), None)).unwrap();
        assert_eq!(picked.id, 1);
        let picked = select_window(&windows, &selector(None, Some("visual studio"), None)).unwrap();
        assert_eq!(picked.id, 2);
    }

    #[test]
    fn selector_process_matches_with_or_without_exe_suffix() {
        assert!(process_matches("notepad.exe", "notepad"));
        assert!(process_matches("notepad", "Notepad.EXE"));
        assert!(process_matches("Code.exe", "code.exe"));
        assert!(!process_matches("notepad.exe", "code"));
        let windows = [info(7, "x", "Notepad.exe")];
        assert_eq!(
            select_window(&windows, &selector(None, None, Some("notepad")))
                .unwrap()
                .id,
            7
        );
        // Title and process filters are combined with AND.
        assert!(select_window(&windows, &selector(None, Some("x"), Some("code"))).is_err());
    }

    #[test]
    fn selector_prefers_an_exact_title_and_reports_ambiguity() {
        let windows = [
            info(1, "Notes", "one.exe"),
            info(2, "Notes - Extra", "two.exe"),
        ];
        let picked = select_window(&windows, &selector(None, Some("notes"), None)).unwrap();
        assert_eq!(picked.id, 1, "exact title wins over a partial match");

        let windows = [info(1, "Notes", "one.exe"), info(2, "notes", "two.exe")];
        let error = select_window(&windows, &selector(None, Some("note"), None)).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("#1"), "{message}");
        assert!(message.contains("#2"), "{message}");
    }

    #[test]
    fn capture_region_defaults_to_the_whole_window() {
        let dims = Dims::new(128, 72);
        let (region, size) = capture_region(dims, None).unwrap();
        assert_eq!(region, Rect::new(0, 0, 128, 72));
        assert_eq!(size, dims);
    }

    #[test]
    fn capture_region_clips_odd_offsets_inside_the_window() {
        let (region, size) =
            capture_region(Dims::new(100, 60), Some(Rect::new(90, 50, 30, 20))).unwrap();
        assert_eq!(region, Rect::new(90, 50, 10, 10));
        assert_eq!(size, Dims::new(10, 10));

        let (region, size) =
            capture_region(Dims::new(101, 61), Some(Rect::new(3, 5, 7, 9))).unwrap();
        assert_eq!(region, Rect::new(3, 5, 7, 9));
        assert_eq!(size, Dims::new(7, 9));
    }

    #[test]
    fn capture_region_outside_the_window_is_an_error() {
        let error =
            capture_region(Dims::new(100, 60), Some(Rect::new(200, 0, 10, 10))).unwrap_err();
        assert!(error.to_string().contains("does not intersect"), "{error}");
        let empty = capture_region(Dims::new(0, 0), None).unwrap_err();
        assert!(
            empty.to_string().contains("no capturable content"),
            "{empty}"
        );
    }

    #[test]
    fn pixel_rects_handle_negative_secondary_monitor_origins() {
        let window = PixelRect::new(-1920, 64, 800, 600);
        let screen = PixelRect::new(-1920, 0, 1920, 1080);
        let overlap = window.intersect(&screen);
        assert_eq!(overlap, window);

        // Partially off the left edge: only 700 columns remain visible.
        let window = PixelRect::new(-2000, 64, 800, 600);
        let overlap = window.intersect(&screen);
        assert_eq!(overlap, PixelRect::new(-1920, 64, 720, 600));

        // No overlap at all yields an empty rectangle at the clamped origin.
        let window = PixelRect::new(4000, 0, 100, 100);
        assert!(window.intersect(&screen).is_empty());
    }

    #[test]
    fn pixel_rect_intersection_does_not_overflow() {
        let huge = PixelRect::new(i32::MAX - 10, i32::MAX - 10, u32::MAX, u32::MAX);
        let other = PixelRect::new(0, 0, 100, 100);
        assert!(huge.intersect(&other).is_empty());
    }

    #[test]
    fn screen_rect_maps_a_window_crop_to_the_desktop() {
        let frame = PixelRect::new(-1920, 64, 800, 600);
        let screen = screen_rect(frame, Rect::new(10, 20, 640, 360));
        assert_eq!(screen, PixelRect::new(-1910, 84, 640, 360));
        assert_eq!(screen.dims(), Dims::new(640, 360));
    }

    #[test]
    fn replay_keeps_pixels_and_updates_identity() {
        let dims = Dims::new(16, 8);
        let cached = mock_frame(dims, 0.25, 3);
        let repeated = replay(&cached, dims, 1234, 9);
        assert_eq!(repeated.dims, dims);
        assert_eq!(repeated.fmt, PixelFormat::Rgba8);
        assert_eq!(repeated.data, cached.data);
        assert_eq!(repeated.pts_ns, 1234);
        assert_eq!(repeated.index, 9);

        // A stale buffer of another shape is replaced by black rather than
        // handed to the engine with a mismatched length.
        let repeated = replay(&cached, Dims::new(8, 8), 1, 2);
        assert_eq!(repeated.dims, Dims::new(8, 8));
        assert_eq!(repeated.data.len(), 8 * 8 * 4);
        assert!(repeated.data.iter().all(|&byte| byte == 0));
    }

    /// Minimal stand-in for a window source: either produces a fresh pattern
    /// frame or replays the previous one while the fake window is minimized.
    struct FakeWindowSource {
        dims: Dims,
        cached: Frame,
        minimized: bool,
        index: u64,
    }

    impl FrameSource for FakeWindowSource {
        fn dims(&self) -> Dims {
            self.dims
        }

        fn backend_name(&self) -> &'static str {
            "fake-window"
        }

        fn next_frame(&mut self) -> Result<Option<Frame>> {
            let pts_ns = 100 + self.index as i64 * 10;
            let frame = if self.minimized {
                replay(&self.cached, self.dims, pts_ns, self.index)
            } else {
                mock_frame(
                    self.dims,
                    (self.index as f64 / 120.0).rem_euclid(1.0),
                    self.index,
                )
            };
            self.cached = frame.clone();
            self.index += 1;
            Ok(Some(frame))
        }
    }

    #[test]
    fn minimized_windows_replay_the_previous_frame_forever() {
        let dims = Dims::new(32, 16);
        let mut source = FakeWindowSource {
            dims,
            cached: mock_frame(dims, 0.0, 0),
            minimized: false,
            index: 0,
        };
        let first = source.next_frame().unwrap().unwrap();
        source.minimized = true;
        let mut previous = first.checksum();
        for expected_index in 1..6 {
            let frame = source
                .next_frame()
                .unwrap()
                .expect("a window source never ends");
            assert_eq!(frame.dims, dims);
            assert_eq!(frame.index, expected_index);
            assert!(frame.pts_ns > 0);
            assert_eq!(frame.checksum(), previous, "pixels are replayed unchanged");
            previous = frame.checksum();
        }
        // Restoring the window resumes fresh content.
        source.minimized = false;
        let resumed = source.next_frame().unwrap().unwrap();
        assert_ne!(resumed.checksum(), previous);
    }
}
