//! Per-application profile rules.
//!
//! The mapping "process name -> profile name" lives in a small text file next
//! to the GUI config, so users can edit it by hand:
//!
//! ```text
//! # 进程名=配置档名
//! genshinimpact.exe=流畅
//! ```
//!
//! It is a GUI-local preference file; the fluid-core config schema is untouched.

use std::path::{Path, PathBuf};

/// One "if this program is captured, use this profile" rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppRule {
    /// Lowercased process image name, e.g. genshinimpact.exe.
    pub process: String,
    /// Profile name inside the profiles directory.
    pub profile: String,
}

impl AppRule {
    /// Build a rule from user input; the process name is normalized.
    pub fn new(process: &str, profile: &str) -> Self {
        Self {
            process: normalize_process(process),
            profile: profile.trim().to_owned(),
        }
    }
}

/// Process names are compared case insensitively and without outer spaces.
pub fn normalize_process(process: &str) -> String {
    process.trim().to_ascii_lowercase()
}

/// Path of the rules file: same directory as the config file.
pub fn rules_path(config_path: &Path) -> PathBuf {
    config_path.with_file_name("gui-app-rules.conf")
}

/// Parse the rules file. Comments, blank lines and broken lines are skipped and
/// a later line for the same process replaces the earlier one.
pub fn parse_rules(text: &str) -> Vec<AppRule> {
    let mut rules: Vec<AppRule> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((process, profile)) = line.split_once('=') else {
            continue;
        };
        let rule = AppRule::new(process, profile);
        if rule.process.is_empty() || rule.profile.is_empty() {
            continue;
        }
        rules.retain(|existing| existing.process != rule.process);
        rules.push(rule);
    }
    rules
}

/// Serialize the rules sorted by process name, with a header comment.
pub fn encode_rules(rules: &[AppRule]) -> String {
    let mut sorted: Vec<&AppRule> = rules.iter().collect();
    sorted.sort_by(|a, b| a.process.cmp(&b.process));
    let mut text = String::from("# Fluid：按程序自动套用的配置档（进程名=配置档名）\n");
    for rule in sorted {
        text.push_str(&format!("{}={}\n", rule.process, rule.profile));
    }
    text
}

/// Rule that matches this process name, if any.
pub fn rule_for_process<'a>(rules: &'a [AppRule], process: &str) -> Option<&'a AppRule> {
    let needle = normalize_process(process);
    if needle.is_empty() {
        return None;
    }
    rules.iter().find(|rule| rule.process == needle)
}

/// Add or replace the rule for a process. Returns true when a new rule was
/// added, false when an existing one was updated.
pub fn upsert_rule(rules: &mut Vec<AppRule>, process: &str, profile: &str) -> bool {
    let rule = AppRule::new(process, profile);
    if rule.process.is_empty() || rule.profile.is_empty() {
        return false;
    }
    match rules
        .iter_mut()
        .find(|existing| existing.process == rule.process)
    {
        Some(existing) => {
            existing.profile = rule.profile;
            false
        }
        None => {
            rules.push(rule);
            true
        }
    }
}

/// Remove the rule for a process; true when something was removed.
pub fn remove_rule(rules: &mut Vec<AppRule>, process: &str) -> bool {
    let needle = normalize_process(process);
    let before = rules.len();
    rules.retain(|rule| rule.process != needle);
    rules.len() != before
}

/// Load the rules; a missing or unreadable file means "no rules".
pub fn load_rules(path: &Path) -> Vec<AppRule> {
    std::fs::read_to_string(path)
        .map(|text| parse_rules(&text))
        .unwrap_or_default()
}

/// Persist the rules, creating the parent directory if needed.
pub fn save_rules(rules: &[AppRule], path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|err| format!("规则目录创建失败：{err}"))?;
    }
    std::fs::write(path, encode_rules(rules)).map_err(|err| format!("规则保存失败：{err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_comments_and_ignores_broken_lines() {
        let text =
            "# 注释\n\ngenshinimpact.exe=流畅\nbroken line\n=空进程\n空档位=\n原神.exe = 高画质 \n";
        let rules = parse_rules(text);
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].process, "genshinimpact.exe");
        assert_eq!(rules[0].profile, "流畅");
        assert_eq!(rules[1].process, "原神.exe");
        assert_eq!(rules[1].profile, "高画质");
    }

    #[test]
    fn later_lines_win_for_the_same_process() {
        let rules = parse_rules("game.exe=流畅\nGAME.EXE=高画质\n");
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].profile, "高画质");
        assert_eq!(rules[0].process, "game.exe");
    }

    #[test]
    fn encode_sorts_and_roundtrips() {
        let rules = vec![AppRule::new("b.exe", "流畅"), AppRule::new("a.exe", "均衡")];
        let text = encode_rules(&rules);
        assert!(text.starts_with("# "), "{text}");
        assert!(text.find("a.exe=").unwrap() < text.find("b.exe=").unwrap());
        // The file is sorted by process name, so the round trip comes back sorted.
        assert_eq!(
            parse_rules(&text),
            vec![AppRule::new("a.exe", "均衡"), AppRule::new("b.exe", "流畅")]
        );
        assert_eq!(encode_rules(&[]).lines().count(), 1);
    }

    #[test]
    fn matching_is_case_insensitive() {
        let rules = parse_rules("GenshinImpact.exe=流畅\n");
        let rule = rule_for_process(&rules, " genshinimpact.EXE ").expect("rule matches");
        assert_eq!(rule.profile, "流畅");
        assert!(rule_for_process(&rules, "other.exe").is_none());
        assert!(rule_for_process(&rules, "  ").is_none());
    }

    #[test]
    fn upsert_adds_then_updates() {
        let mut rules = Vec::new();
        assert!(upsert_rule(&mut rules, "game.exe", "流畅"));
        assert!(!upsert_rule(&mut rules, "GAME.EXE", "高画质"));
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].profile, "高画质");
        assert!(!upsert_rule(&mut rules, "   ", "流畅"));
    }

    #[test]
    fn remove_drops_the_rule() {
        let mut rules = vec![AppRule::new("a.exe", "均衡"), AppRule::new("b.exe", "流畅")];
        assert!(remove_rule(&mut rules, "A.EXE"));
        assert_eq!(rules.len(), 1);
        assert!(!remove_rule(&mut rules, "a.exe"));
    }

    #[test]
    fn rules_file_roundtrips() {
        let dir = std::env::temp_dir().join(format!("fluid-gui-rules-{}", std::process::id()));
        let path = rules_path(&dir.join("gui.toml"));
        assert_eq!(path, dir.join("gui-app-rules.conf"));
        let rules = vec![AppRule::new("game.exe", "流畅")];
        save_rules(&rules, &path).expect("save works");
        assert_eq!(load_rules(&path), rules);
        let _ = std::fs::remove_dir_all(&dir);
        assert!(load_rules(Path::new("C:/definitely/missing/gui-app-rules.conf")).is_empty());
    }
}
