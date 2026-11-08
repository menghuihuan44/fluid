//! Config file location, loading and saving.
//!
//! The GUI stores its settings in the same TOML format that `fluid-core`
//! already uses for the CLI, so a config written by the GUI can be reused by
//! `fluid-cli` and vice versa.

use std::path::{Path, PathBuf};

use fluid_core::FluidConfig;

/// Resolve the config file path from the environment values handed in.
///
/// Pure function so the resolution rules can be tested without touching the
/// real environment: on Windows the app data directory wins, then the XDG
/// config home, then the home directory, and finally the current directory.
pub fn resolve_path(appdata: Option<&str>, xdg: Option<&str>, home: Option<&str>) -> PathBuf {
    if let Some(dir) = appdata.filter(|s| !s.is_empty()) {
        return PathBuf::from(dir).join("Fluid").join("gui.toml");
    }
    if let Some(dir) = xdg.filter(|s| !s.is_empty()) {
        return PathBuf::from(dir).join("fluid").join("gui.toml");
    }
    if let Some(dir) = home.filter(|s| !s.is_empty()) {
        return PathBuf::from(dir)
            .join(".config")
            .join("fluid")
            .join("gui.toml");
    }
    PathBuf::from("fluid-gui.toml")
}

/// Locate the config file for the current process.
pub fn default_config_path() -> PathBuf {
    // Android has no APPDATA/XDG layout; keep the config next to the app data.
    if cfg!(target_os = "android") {
        return PathBuf::from("fluid-gui.toml");
    }
    let appdata = std::env::var("APPDATA").ok();
    let xdg = std::env::var("XDG_CONFIG_HOME").ok();
    let home = std::env::var("HOME").ok();
    resolve_path(appdata.as_deref(), xdg.as_deref(), home.as_deref())
}

/// Result of loading the config: the effective config plus an optional warning
/// that should be shown in the UI (bad config file, permissions, ...).
pub struct LoadedConfig {
    pub config: FluidConfig,
    pub warning: Option<String>,
}

/// Load the config from `path`, never failing: on any error we fall back to
/// the defaults and surface a warning instead of panicking.
pub fn load(path: &Path) -> LoadedConfig {
    match FluidConfig::load_or_default(path) {
        Ok(config) => LoadedConfig {
            config,
            warning: None,
        },
        Err(err) => LoadedConfig {
            config: FluidConfig::default(),
            warning: Some(format!("配置文件无法读取，已使用默认设置：{err}")),
        },
    }
}

/// Save the config, returning a human-readable error message on failure.
pub fn save(config: &FluidConfig, path: &Path) -> Result<(), String> {
    config
        .save(path)
        .map_err(|err| format!("配置保存失败：{err}"))
}

/// Which capture source the 实时增强 tab starts with.
///
/// The window itself (its native id) lives in the FluidConfig TOML because the
/// capture backend needs it; the choice of source is a UI preference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SourceChoice {
    /// Capture a whole monitor through DXGI desktop duplication.
    Screen,
    /// Capture one application window (Windows only).
    Window,
    /// Built-in moving test pattern, so the app is usable without a game.
    #[default]
    Synthetic,
}

impl SourceChoice {
    pub const ALL: [Self; 3] = [Self::Screen, Self::Window, Self::Synthetic];

    /// Stable identifier written to the UI preference file.
    pub fn name(self) -> &'static str {
        match self {
            Self::Screen => "screen",
            Self::Window => "window",
            Self::Synthetic => "synthetic",
        }
    }

    /// Chinese label used by the source selector.
    pub fn label(self) -> &'static str {
        match self {
            Self::Screen => "屏幕捕获",
            Self::Window => "窗口捕获",
            Self::Synthetic => "内置测试画面",
        }
    }

    /// Parse the stored identifier; unknown values mean "not set".
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "screen" | "monitor" | "desktop" => Some(Self::Screen),
            "window" => Some(Self::Window),
            "synthetic" | "builtin" | "test" => Some(Self::Synthetic),
            _ => None,
        }
    }
}

/// UI-only preferences that egui must remember between runs.
///
/// They cannot live in the FluidConfig TOML because that struct belongs to
/// fluid-core, so the GUI keeps a tiny side-car file next to gui.toml.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UiFlags {
    /// Whether the 高级设置 section is expanded; None means "follow the
    /// layout default" (open on desktop, collapsed on phones).
    pub advanced_open: Option<bool>,
    /// Last used capture source; None means "pick a sensible default".
    pub source: Option<SourceChoice>,
    /// Apply the last used configuration profile on the next start.
    pub apply_last_profile: Option<bool>,
    /// Name of the profile that was active when the app last closed.
    pub last_profile: Option<String>,
    /// Silently switch profiles when the captured window matches a rule.
    pub auto_apply: Option<bool>,
    /// The first-run guide card was dismissed.
    pub guide_dismissed: Option<bool>,
}

/// Path of the UI preference file: same directory as the config file.
pub fn ui_flags_path(config_path: &Path) -> PathBuf {
    config_path.with_file_name("gui-ui.conf")
}

/// Parse the side-car file. Unknown lines are ignored and garbage values mean
/// "not set", so a hand-edited or corrupted file never breaks the UI.
pub fn parse_ui_flags(text: &str) -> UiFlags {
    let mut flags = UiFlags::default();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let (key, value) = (key.trim(), value.trim());
        match key {
            "advanced_open" => flags.advanced_open = parse_bool(value),
            "source" => flags.source = SourceChoice::parse(value),
            "apply_last" => flags.apply_last_profile = parse_bool(value),
            "auto_apply" => flags.auto_apply = parse_bool(value),
            "guide_dismissed" => flags.guide_dismissed = parse_bool(value),
            "last_profile" => {
                let name = value.trim();
                flags.last_profile = if name.is_empty() {
                    None
                } else {
                    Some(name.to_owned())
                };
            }
            _ => {}
        }
    }
    flags
}

/// Parse a yes/no flag; unknown text means "not set".
fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Serialize the flags in the same simple key=value format.
pub fn encode_ui_flags(flags: &UiFlags) -> String {
    let mut text = String::new();
    if let Some(open) = flags.advanced_open {
        text.push_str(&format!("advanced_open={open}\n"));
    }
    if let Some(source) = flags.source {
        text.push_str(&format!("source={}\n", source.name()));
    }
    if let Some(apply) = flags.apply_last_profile {
        text.push_str(&format!("apply_last={apply}\n"));
    }
    if let Some(name) = &flags.last_profile {
        text.push_str(&format!("last_profile={name}\n"));
    }
    if let Some(auto) = flags.auto_apply {
        text.push_str(&format!("auto_apply={auto}\n"));
    }
    if let Some(dismissed) = flags.guide_dismissed {
        text.push_str(&format!("guide_dismissed={dismissed}\n"));
    }
    text
}

/// Load the flags; a missing or unreadable file simply means "no preferences".
pub fn load_ui_flags(path: &Path) -> UiFlags {
    std::fs::read_to_string(path)
        .map(|text| parse_ui_flags(&text))
        .unwrap_or_default()
}

/// Persist the flags, creating the parent directory if needed.
pub fn save_ui_flags(flags: &UiFlags, path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|err| format!("界面设置目录创建失败：{err}"))?;
    }
    std::fs::write(path, encode_ui_flags(flags)).map_err(|err| format!("界面设置保存失败：{err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appdata_wins_on_windows_layout() {
        let path = resolve_path(
            Some("C:\\Users\\u\\AppData\\Roaming"),
            Some("/xdg"),
            Some("/home/u"),
        );
        assert_eq!(
            path,
            PathBuf::from("C:\\Users\\u\\AppData\\Roaming")
                .join("Fluid")
                .join("gui.toml")
        );
    }

    #[test]
    fn xdg_is_the_second_choice() {
        let path = resolve_path(None, Some("/xdg"), Some("/home/u"));
        assert_eq!(path, PathBuf::from("/xdg").join("fluid").join("gui.toml"));
    }

    #[test]
    fn home_is_the_third_choice() {
        let path = resolve_path(None, None, Some("/home/u"));
        assert_eq!(
            path,
            PathBuf::from("/home/u")
                .join(".config")
                .join("fluid")
                .join("gui.toml")
        );
    }

    #[test]
    fn current_directory_is_the_last_resort() {
        assert_eq!(
            resolve_path(None, None, None),
            PathBuf::from("fluid-gui.toml")
        );
        assert_eq!(
            resolve_path(Some(""), Some(""), Some("")),
            PathBuf::from("fluid-gui.toml")
        );
    }

    #[test]
    fn bad_config_falls_back_to_defaults_with_warning() {
        let dir = std::env::temp_dir().join(format!("fluid-gui-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gui.toml");
        std::fs::write(&path, "this is not => valid toml [").unwrap();
        let loaded = load(&path);
        assert!(loaded.warning.is_some());
        assert_eq!(loaded.config.profile, FluidConfig::default().profile);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_config_loads_defaults_without_warning() {
        let path = std::env::temp_dir()
            .join("fluid-gui-definitely-missing-dir")
            .join("gui.toml");
        let loaded = load(&path);
        assert!(loaded.warning.is_none());
    }

    #[test]
    fn save_then_load_roundtrips() {
        let dir = std::env::temp_dir().join(format!("fluid-gui-roundtrip-{}", std::process::id()));
        let path = dir.join("nested").join("gui.toml");
        let mut config = FluidConfig::default();
        config.upscaler.sharpness = 0.42;
        config.upscaler.backend = fluid_core::UpscaleBackendKind::Gpu;
        save(&config, &path).expect("save works");
        let loaded = load(&path);
        assert!(loaded.warning.is_none());
        assert!((loaded.config.upscaler.sharpness - 0.42).abs() < 1e-4);
        assert_eq!(
            loaded.config.upscaler.backend,
            fluid_core::UpscaleBackendKind::Gpu
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ui_flags_roundtrip_through_text() {
        let flags = UiFlags {
            advanced_open: Some(false),
            ..UiFlags::default()
        };
        let text = encode_ui_flags(&flags);
        assert_eq!(text, "advanced_open=false\n");
        assert_eq!(parse_ui_flags(&text), flags);
        assert_eq!(
            parse_ui_flags("advanced_open=true"),
            UiFlags {
                advanced_open: Some(true),
                ..UiFlags::default()
            }
        );
    }

    #[test]
    fn source_choice_roundtrips_through_the_flag_file() {
        for choice in SourceChoice::ALL {
            let flags = UiFlags {
                advanced_open: Some(true),
                source: Some(choice),
                ..UiFlags::default()
            };
            let text = encode_ui_flags(&flags);
            assert!(
                text.contains(&format!("source={}", choice.name())),
                "{}",
                text
            );
            assert_eq!(parse_ui_flags(&text), flags);
        }
        assert_eq!(
            parse_ui_flags("source=window\n"),
            UiFlags {
                source: Some(SourceChoice::Window),
                ..UiFlags::default()
            }
        );
        // Old files written before the source option existed must not fail.
        assert_eq!(
            parse_ui_flags("advanced_open=true\n"),
            UiFlags {
                advanced_open: Some(true),
                ..UiFlags::default()
            }
        );
        // Garbage values mean "not set" instead of an error.
        assert_eq!(parse_ui_flags("source=webcam\n").source, None);
    }

    #[test]
    fn source_choice_labels_cover_the_three_sources() {
        assert_eq!(SourceChoice::Screen.label(), "屏幕捕获");
        assert_eq!(SourceChoice::Window.label(), "窗口捕获");
        assert_eq!(SourceChoice::Synthetic.label(), "内置测试画面");
        assert_eq!(SourceChoice::default(), SourceChoice::Synthetic);
    }

    #[test]
    fn unknown_ui_flag_lines_are_ignored() {
        let parsed = parse_ui_flags(
            "# written by an older build\nadvanced_open=1\ntheme=dark\nmystery line without key\n",
        );
        assert_eq!(parsed.advanced_open, Some(true));
        let garbage = parse_ui_flags("advanced_open=maybe\n");
        assert_eq!(garbage, UiFlags::default());
    }

    #[test]
    fn missing_ui_flag_file_means_defaults() {
        let path = std::env::temp_dir().join("fluid-gui-no-ui-flags.conf");
        let _ = std::fs::remove_file(&path);
        assert_eq!(load_ui_flags(&path), UiFlags::default());
    }

    #[test]
    fn ui_flags_are_saved_next_to_the_config() {
        let dir = std::env::temp_dir().join(format!("fluid-gui-uiflags-{}", std::process::id()));
        let path = ui_flags_path(&dir.join("gui.toml"));
        assert_eq!(path, dir.join("gui-ui.conf"));
        let flags = UiFlags {
            advanced_open: Some(true),
            source: Some(SourceChoice::Screen),
            apply_last_profile: Some(true),
            last_profile: Some("均衡".to_owned()),
            auto_apply: Some(true),
            guide_dismissed: Some(true),
        };
        save_ui_flags(&flags, &path).expect("save works");
        assert_eq!(load_ui_flags(&path), flags);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn profile_and_guide_flags_roundtrip() {
        let flags = UiFlags {
            apply_last_profile: Some(true),
            last_profile: Some("Genshin 流畅".to_owned()),
            auto_apply: Some(false),
            guide_dismissed: Some(true),
            ..UiFlags::default()
        };
        let text = encode_ui_flags(&flags);
        assert!(text.contains("apply_last=true"), "{text}");
        assert!(text.contains("last_profile=Genshin 流畅"), "{text}");
        assert!(text.contains("auto_apply=false"), "{text}");
        assert!(text.contains("guide_dismissed=true"), "{text}");
        assert_eq!(parse_ui_flags(&text), flags);
    }

    #[test]
    fn old_flag_files_without_profile_keys_still_load() {
        // gui-ui.conf files written before profiles existed must keep working.
        let parsed = parse_ui_flags("advanced_open=true\nsource=window\n");
        assert_eq!(parsed.advanced_open, Some(true));
        assert_eq!(parsed.source, Some(SourceChoice::Window));
        assert_eq!(parsed.apply_last_profile, None);
        assert_eq!(parsed.last_profile, None);
        assert_eq!(parsed.auto_apply, None);
        assert_eq!(parsed.guide_dismissed, None);
        // Broken values mean "not set" instead of an error.
        let garbage = parse_ui_flags("apply_last=maybe\nlast_profile=\nauto_apply=2\n");
        assert_eq!(garbage, UiFlags::default());
    }

    #[test]
    fn old_config_without_a_backend_field_still_loads() {
        let dir = std::env::temp_dir().join(format!("fluid-gui-oldcfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gui.toml");
        // gui.toml files written before the backend option existed have no
        // "backend" key; loading them must keep working and default to auto.
        let text = FluidConfig::default()
            .to_toml_string()
            .expect("the default config serializes");
        let stripped = text
            .lines()
            .filter(|line| !line.trim_start().starts_with("backend"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, stripped).unwrap();
        let loaded = load(&path);
        assert!(loaded.warning.is_none());
        assert_eq!(
            loaded.config.upscaler.backend,
            fluid_core::UpscaleBackendKind::Auto
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
