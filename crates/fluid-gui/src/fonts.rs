//! CJK font discovery and installation for egui.
//!
//! egui's built-in fonts do not contain CJK glyphs, so Chinese labels would show
//! up as empty boxes. We look for a system font that has them and register it as
//! a fallback in both the proportional and the monospace font families.

use std::sync::Arc;

use eframe::egui;

/// Candidate font files, in order of preference.
pub const CANDIDATES: &[&str] = &[
    "C:\\Windows\\Fonts\\msyh.ttc",
    "C:\\Windows\\Fonts\\msyhl.ttc",
    "C:\\Windows\\Fonts\\simhei.ttf",
    "C:\\Windows\\Fonts\\simsun.ttc",
    "/system/fonts/NotoSansCJK-Regular.ttc",
    "/system/fonts/NotoSansSC-Regular.otf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
];

/// Pick the first candidate for which `exists` returns true.
///
/// This is a pure function so that the search logic can be unit-tested without
/// touching the filesystem.
pub fn pick_existing<'a, F>(candidates: &'a [&'a str], exists: F) -> Option<&'a str>
where
    F: Fn(&str) -> bool,
{
    candidates.iter().copied().find(|path| exists(path))
}

/// Find a CJK font on this machine, load it and install it into `ctx`.
///
/// Returns the path of the font that was installed, or `None` if no candidate
/// was found (the built-in fonts are kept in that case).
pub fn install(ctx: &egui::Context) -> Option<&'static str> {
    let path = pick_existing(CANDIDATES, |path| std::path::Path::new(path).is_file())?;
    let bytes = std::fs::read(path).ok()?;
    if bytes.is_empty() {
        return None;
    }

    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "cjk".to_owned(),
        Arc::new(egui::FontData::from_owned(bytes)),
    );

    // Appended last = used as a fallback after the default fonts, so Latin text
    // keeps the familiar egui look while CJK text still renders.
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push("cjk".to_owned());
    }

    ctx.set_fonts(fonts);
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FAKE: &[&str] = &["/first.ttf", "/second.ttf", "/third.ttf"];

    #[test]
    fn picks_the_first_existing_candidate() {
        let picked = pick_existing(FAKE, |path| path == "/second.ttf" || path == "/third.ttf");
        assert_eq!(picked, Some("/second.ttf"));
    }

    #[test]
    fn returns_none_when_nothing_exists() {
        assert_eq!(pick_existing(FAKE, |_| false), None);
    }

    #[test]
    fn returns_none_for_empty_candidate_list() {
        assert_eq!(pick_existing(&[], |_| true), None);
    }

    #[test]
    fn candidate_list_is_non_empty_and_ordered() {
        assert!(!CANDIDATES.is_empty());
        assert!(CANDIDATES[0].contains("msyh"));
    }
}
