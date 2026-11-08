//! Named configuration profiles.
//!
//! A profile is a full FluidConfig saved as TOML under
//! <app config dir>/profiles/<name>.toml, so the files stay readable and can be
//! copied between machines. The fluid-core config schema is not touched.

use std::path::{Path, PathBuf};

use fluid_core::FluidConfig;

/// Longest allowed profile name, in characters.
pub const NAME_MAX_CHARS: usize = 24;

/// Windows reserved device names (compared case insensitively).
const RESERVED_NAMES: [&str; 22] = [
    "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8",
    "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
];

/// Characters Windows does not allow in a file name.
const BAD_CHARS: [char; 9] = ['/', '\\', ':', '*', '?', '"', '<', '>', '|'];

/// Directory that holds the profile files, next to the main config file.
pub fn profiles_dir(config_path: &Path) -> PathBuf {
    config_path.with_file_name("profiles")
}

/// Path of one profile file inside the profile directory.
pub fn profile_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.toml"))
}

/// Check a user typed profile name; the error text is shown in the dialog.
pub fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("名字不能为空。".to_owned());
    }
    if name.trim() != name {
        return Err("名字开头和结尾不能有空格。".to_owned());
    }
    if name.chars().count() > NAME_MAX_CHARS {
        return Err(format!("名字太长，最多 {NAME_MAX_CHARS} 个字符。"));
    }
    if name.chars().any(char::is_control) {
        return Err("名字里不能有看不见的控制字符。".to_owned());
    }
    if let Some(bad) = name.chars().find(|ch| BAD_CHARS.contains(ch)) {
        return Err(format!("名字里不能包含 {bad} 这个字符。"));
    }
    if name == "." || name == ".." || name.ends_with('.') {
        return Err("名字不能以点结尾。".to_owned());
    }
    let lower = name.to_ascii_lowercase();
    if RESERVED_NAMES.contains(&lower.as_str()) {
        return Err("这是系统保留的文件名，换一个名字吧。".to_owned());
    }
    Ok(())
}

/// Names of the profiles stored in dir, sorted case insensitively.
///
/// Files that fail to parse still show up in the list, so the user can delete
/// them from the dialog instead of hunting for the folder.
pub fn list_profiles(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return names;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        if stem.is_empty() {
            continue;
        }
        names.push(stem.to_owned());
    }
    names.sort_by_key(|name| name.to_lowercase());
    names
}

/// Load one profile. Missing file or bad TOML becomes a friendly message.
pub fn load_profile(dir: &Path, name: &str) -> Result<FluidConfig, String> {
    validate_name(name)?;
    FluidConfig::load(profile_path(dir, name))
        .map_err(|err| format!("配置档「{name}」读取失败：{err}"))
}

/// Save a profile. Without overwrite an existing name is refused, so the user
/// has to confirm in the dialog before replacing a profile.
pub fn save_profile(
    dir: &Path,
    name: &str,
    config: &FluidConfig,
    overwrite: bool,
) -> Result<(), String> {
    validate_name(name)?;
    let path = profile_path(dir, name);
    if !overwrite && path.exists() {
        return Err(format!(
            "已经有叫「{name}」的配置档了，换个名字或先删除它。"
        ));
    }
    config
        .save(&path)
        .map_err(|err| format!("配置档「{name}」保存失败：{err}"))
}

/// Rename a profile file. Refuses to clobber an existing name.
pub fn rename_profile(dir: &Path, old: &str, new: &str) -> Result<(), String> {
    validate_name(new)?;
    if old == new {
        return Ok(());
    }
    let from = profile_path(dir, old);
    let to = profile_path(dir, new);
    if !from.exists() {
        return Err(format!("找不到配置档「{old}」。"));
    }
    if to.exists() {
        return Err(format!("已经有叫「{new}」的配置档了。"));
    }
    std::fs::rename(&from, &to).map_err(|err| format!("重命名失败：{err}"))
}

/// Delete a profile file.
pub fn delete_profile(dir: &Path, name: &str) -> Result<(), String> {
    let path = profile_path(dir, name);
    if !path.exists() {
        return Err(format!("找不到配置档「{name}」。"));
    }
    std::fs::remove_file(&path).map_err(|err| format!("删除失败：{err}"))
}

/// Suggest a free name based on base, e.g. 均衡 -> 均衡 2 when 均衡 exists.
pub fn unique_name(base: &str, existing: &[String]) -> String {
    let trimmed = base.trim();
    let base = if trimmed.is_empty() {
        "新配置档"
    } else {
        trimmed
    };
    if !existing.iter().any(|name| name == base) {
        return base.to_owned();
    }
    for index in 2..=99 {
        let candidate = format!("{base} {index}");
        if !existing.iter().any(|name| name == &candidate) {
            return candidate;
        }
    }
    format!("{base} 99")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    fn temp_dir() -> PathBuf {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("fluid-gui-profiles-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn accepts_friendly_names() {
        for name in ["均衡", "Genshin 流畅", "4K 高画质", "a", "我的配置档 1"] {
            assert!(
                validate_name(name).is_ok(),
                "{name} should be accepted, got {:?}",
                validate_name(name)
            );
        }
    }

    #[test]
    fn rejects_names_that_are_not_file_names() {
        for name in [
            "", " 名字", "名字 ", "名字.", ".", "..", "a/b", "a\\b", "a:b", "a*b", "a?b", "a<b",
            "a>b", "a|b", "a\"b", "CON", "nul", "Com1",
        ] {
            assert!(validate_name(name).is_err(), "{name:?} should be rejected");
        }
        let long = "很".repeat(NAME_MAX_CHARS + 1);
        assert!(validate_name(&long).is_err());
        assert!(validate_name(&"很".repeat(NAME_MAX_CHARS)).is_ok());
    }

    #[test]
    fn profiles_live_next_to_the_config() {
        let dir = profiles_dir(Path::new("C:/cfg/gui.toml"));
        assert_eq!(dir, Path::new("C:/cfg/profiles"));
        assert_eq!(
            profile_path(&dir, "均衡"),
            Path::new("C:/cfg/profiles/均衡.toml")
        );
    }

    #[test]
    fn save_list_and_load_roundtrip() {
        let dir = temp_dir();
        let mut config = FluidConfig::default();
        config.upscaler.sharpness = 0.42;
        save_profile(&dir, "均衡", &config, false).expect("save works");
        save_profile(&dir, "流畅", &config, false).expect("save works");
        assert_eq!(
            list_profiles(&dir),
            vec!["均衡".to_owned(), "流畅".to_owned()]
        );
        let loaded = load_profile(&dir, "均衡").expect("load works");
        assert!((loaded.upscaler.sharpness - 0.42).abs() < 1e-4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_refuses_to_clobber_without_permission() {
        let dir = temp_dir();
        let config = FluidConfig::default();
        save_profile(&dir, "均衡", &config, false).expect("first save works");
        let error = save_profile(&dir, "均衡", &config, false).expect_err("second save must fail");
        assert!(error.contains("已经有叫"), "{error}");
        save_profile(&dir, "均衡", &config, true).expect("overwrite allowed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rename_guards_existing_and_missing_names() {
        let dir = temp_dir();
        let config = FluidConfig::default();
        save_profile(&dir, "均衡", &config, false).unwrap();
        save_profile(&dir, "流畅", &config, false).unwrap();
        assert!(rename_profile(&dir, "均衡", "流畅").is_err());
        assert!(rename_profile(&dir, "不存在", "新名字").is_err());
        rename_profile(&dir, "均衡", "高画质").expect("rename works");
        assert_eq!(
            list_profiles(&dir),
            vec!["流畅".to_owned(), "高画质".to_owned()]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_reports_missing_profiles() {
        let dir = temp_dir();
        let config = FluidConfig::default();
        save_profile(&dir, "均衡", &config, false).unwrap();
        delete_profile(&dir, "均衡").expect("delete works");
        assert!(list_profiles(&dir).is_empty());
        assert!(delete_profile(&dir, "均衡").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_directory_lists_nothing() {
        let dir = std::env::temp_dir().join("fluid-gui-profiles-does-not-exist");
        assert!(list_profiles(&dir).is_empty());
    }

    #[test]
    fn unique_name_adds_a_counter() {
        let existing = vec!["均衡".to_owned(), "均衡 2".to_owned()];
        assert_eq!(unique_name("流畅", &existing), "流畅");
        assert_eq!(unique_name("均衡", &existing), "均衡 3");
        assert_eq!(unique_name("  ", &existing), "新配置档");
    }
}
