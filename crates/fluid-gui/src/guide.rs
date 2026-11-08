//! First-run guide card for the 实时增强 tab.

/// One numbered step of the welcome card.
pub struct Step {
    /// Short title, e.g. 选择画面来源.
    pub title: &'static str,
    /// One sentence aimed at a beginner.
    pub text: &'static str,
}

/// The three steps, in the order a new user should follow them.
pub const STEPS: [Step; 3] = [
    Step {
        title: "① 选择画面来源",
        text: "想增强正在玩的游戏就选「窗口捕获」，再点「选择窗口」；第一次试用可以选「内置测试画面」。",
    },
    Step {
        title: "② 选好预设",
        text: "不确定就保持「均衡」，它会自动配好插帧倍率和放大强度。",
    },
    Step {
        title: "③ 点开始增强",
        text: "点屏幕底部的大按钮；游戏用「无边框窗口」模式运行效果最好。",
    },
];

/// The card is hidden after the user dismisses it, and never shown during
/// automated --smoke runs (they must stay deterministic).
pub fn should_show(dismissed: bool, smoke: bool) -> bool {
    !dismissed && !smoke
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_run_shows_the_card() {
        assert!(should_show(false, false));
    }

    #[test]
    fn dismissed_or_smoke_runs_hide_the_card() {
        assert!(!should_show(true, false));
        assert!(!should_show(false, true));
        assert!(!should_show(true, true));
    }

    #[test]
    fn there_are_three_numbered_steps() {
        assert_eq!(STEPS.len(), 3);
        for (index, step) in STEPS.iter().enumerate() {
            assert!(
                step.title
                    .starts_with(char::from_u32(0x2460 + index as u32).unwrap())
            );
            assert!(!step.text.is_empty());
        }
    }
}
