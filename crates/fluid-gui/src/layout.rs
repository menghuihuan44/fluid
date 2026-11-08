//! Adaptive layout decisions for desktop and phone.
//!
//! Everything here is pure so it can be unit tested without an egui context:
//! the app asks for a LayoutPlan once per frame and then draws accordingly.
//! Sizes are in egui points (already DPI aware: one point is scaled by
//! ctx.pixels_per_point() when painted).

use eframe::egui;

/// Minimum touch target size in points (Apple/Material guidelines: 44-48 pt).
pub const TOUCH_MIN: f32 = 44.0;
/// Windows narrower than this use the phone layout.
pub const COMPACT_MAX_WIDTH: f32 = 700.0;
/// Windows narrower than this use a medium (stacked but desktop spacing) layout.
pub const MEDIUM_MAX_WIDTH: f32 = 1050.0;

/// How much room the window gives us.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScreenClass {
    /// Phone / very narrow desktop window: one column, big touch targets.
    Compact,
    /// Tablet or small desktop window: stacked layout, normal spacing.
    Medium,
    /// Normal desktop window: preview and controls side by side.
    Wide,
}

impl ScreenClass {
    pub const ALL: [ScreenClass; 3] =
        [ScreenClass::Compact, ScreenClass::Medium, ScreenClass::Wide];

    /// Classify a window by its width in points.
    pub fn from_width(width_points: f32) -> Self {
        if width_points < COMPACT_MAX_WIDTH {
            ScreenClass::Compact
        } else if width_points < MEDIUM_MAX_WIDTH {
            ScreenClass::Medium
        } else {
            ScreenClass::Wide
        }
    }

    pub fn is_touch(self) -> bool {
        self == ScreenClass::Compact
    }
}

/// Spacing values that make every widget reachable with a finger.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TouchMetrics {
    /// Minimum size of any clickable widget (44 pt is the accessibility floor).
    pub interact_size: egui::Vec2,
    pub button_padding: egui::Vec2,
    pub item_spacing: egui::Vec2,
    pub slider_width: f32,
    pub slider_rail_height: f32,
    /// Height of primary action buttons (开始增强 / 停止增强).
    pub primary_button_height: f32,
}

/// Metrics for a screen class: compact pads everything to 44 pt, desktop stays
/// dense so the existing desktop layout does not change.
pub fn touch_metrics(class: ScreenClass) -> TouchMetrics {
    match class {
        ScreenClass::Compact => TouchMetrics {
            interact_size: egui::vec2(44.0, TOUCH_MIN),
            button_padding: egui::vec2(14.0, 10.0),
            item_spacing: egui::vec2(10.0, 10.0),
            slider_width: 220.0,
            slider_rail_height: 14.0,
            primary_button_height: 56.0,
        },
        ScreenClass::Medium => TouchMetrics {
            interact_size: egui::vec2(40.0, 32.0),
            button_padding: egui::vec2(12.0, 8.0),
            item_spacing: egui::vec2(9.0, 8.0),
            slider_width: 160.0,
            slider_rail_height: 10.0,
            primary_button_height: 50.0,
        },
        ScreenClass::Wide => TouchMetrics {
            interact_size: egui::vec2(40.0, 24.0),
            button_padding: egui::vec2(12.0, 7.0),
            item_spacing: egui::vec2(8.0, 8.0),
            slider_width: 120.0,
            slider_rail_height: 8.0,
            primary_button_height: 56.0,
        },
    }
}

/// Everything the drawing code needs to know about the current window size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayoutPlan {
    pub class: ScreenClass,
    /// Preview on the left, controls on the right (desktop only).
    pub side_by_side: bool,
    /// Preview height when stacked (points); 0 when side by side.
    pub preview_height: f32,
    /// Width of the right-hand control column (side by side only).
    pub control_width: f32,
    /// Width used for content on the stacked layout.
    pub content_width: f32,
    /// Horizontal scroll bar for the tab row instead of a fixed top row.
    pub tabs_scroll: bool,
    /// Actions live in a fixed bottom bar, always reachable with a thumb.
    pub bottom_bar: bool,
    /// Pad all widgets for fingers and prefer buttons over checkboxes.
    pub touch: bool,
    /// 高级设置 section starts expanded (desktop) or collapsed (phone).
    pub advanced_default_open: bool,
    /// Text scale (kept at 1.0: compact relies on padding, not tiny fonts).
    pub text_scale: f32,
    /// Very short or very narrow window (QA defect D2): the wrapped tab bar
    /// and the action-bar hints would eat the whole page, so they collapse
    /// into a single picker row and the buttons stay unclipped.
    pub tight: bool,
}

impl LayoutPlan {
    /// Plan the layout for the given available size in points.
    pub fn new(available: egui::Vec2) -> Self {
        let width = available.x.max(1.0);
        let height = available.y.max(1.0);
        let class = ScreenClass::from_width(width);
        let side_by_side = class == ScreenClass::Wide && width >= MEDIUM_MAX_WIDTH;

        let content_width = (width - 24.0).clamp(240.0, 760.0);
        let control_width = if side_by_side {
            (width * 0.30).clamp(300.0, 380.0)
        } else {
            content_width
        };

        let preview_height = if side_by_side {
            0.0
        } else {
            match class {
                ScreenClass::Compact => (height * 0.38).clamp(150.0, 300.0),
                _ => (height * 0.45).clamp(200.0, 420.0),
            }
        };

        Self {
            class,
            side_by_side,
            preview_height,
            control_width,
            content_width,
            tabs_scroll: class == ScreenClass::Compact,
            bottom_bar: class == ScreenClass::Compact,
            touch: class.is_touch(),
            advanced_default_open: class != ScreenClass::Compact,
            text_scale: 1.0,
            // Below roughly 560 points of height the top and bottom chrome
            // leave no room for the page itself (360x740 / 480x820 windows).
            tight: class == ScreenClass::Compact && height < 560.0,
        }
    }

    /// Height of primary action buttons.
    pub fn primary_button_height(&self) -> f32 {
        touch_metrics(self.class).primary_button_height
    }

    /// Apply spacing to a UI style: this is what makes every control at least
    /// 44x44 points on a phone.
    pub fn apply_style(&self, style: &mut egui::Style) {
        let metrics = touch_metrics(self.class);
        style.spacing.interact_size = metrics.interact_size;
        style.spacing.button_padding = metrics.button_padding;
        style.spacing.item_spacing = metrics.item_spacing;
        style.spacing.slider_width = metrics.slider_width;
        style.spacing.slider_rail_height = metrics.slider_rail_height;
        // Bigger icons for checkboxes so the whole row is easy to hit.
        if self.touch {
            style.spacing.icon_width = 22.0;
            style.spacing.icon_width_inner = 12.0;
            style.spacing.icon_spacing = 8.0;
            style.spacing.combo_height = 320.0;
        }
    }
}

/// Human readable description of the current layout, used in 关于 / 性能体检.
pub fn class_label(class: ScreenClass) -> &'static str {
    match class {
        ScreenClass::Compact => "手机 / 窄窗口布局",
        ScreenClass::Medium => "平板 / 小窗口布局",
        ScreenClass::Wide => "电脑宽屏布局",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_thresholds_are_stable() {
        assert_eq!(ScreenClass::from_width(320.0), ScreenClass::Compact);
        assert_eq!(ScreenClass::from_width(699.9), ScreenClass::Compact);
        assert_eq!(ScreenClass::from_width(700.0), ScreenClass::Medium);
        assert_eq!(ScreenClass::from_width(1049.9), ScreenClass::Medium);
        assert_eq!(ScreenClass::from_width(1050.0), ScreenClass::Wide);
        assert_eq!(ScreenClass::from_width(1180.0), ScreenClass::Wide);
    }

    #[test]
    fn desktop_keeps_the_side_by_side_layout() {
        let plan = LayoutPlan::new(egui::vec2(1180.0, 760.0));
        assert_eq!(plan.class, ScreenClass::Wide);
        assert!(plan.side_by_side);
        assert!(!plan.bottom_bar);
        assert!(!plan.tabs_scroll);
        assert!(plan.advanced_default_open);
        assert!((plan.control_width - 354.0).abs() < 1.0);
        assert_eq!(plan.preview_height, 0.0);
        assert!(!plan.touch);
    }

    #[test]
    fn phone_uses_one_column_and_a_bottom_bar() {
        let plan = LayoutPlan::new(egui::vec2(420.0, 800.0));
        assert_eq!(plan.class, ScreenClass::Compact);
        assert!(!plan.side_by_side);
        assert!(plan.bottom_bar);
        assert!(plan.tabs_scroll);
        assert!(plan.touch);
        assert!(!plan.advanced_default_open);
        assert!(plan.preview_height >= 150.0 && plan.preview_height <= 300.0);
        assert!(plan.control_width <= 700.0);
    }

    #[test]
    fn medium_is_stacked_without_the_touch_padding() {
        let plan = LayoutPlan::new(egui::vec2(820.0, 700.0));
        assert_eq!(plan.class, ScreenClass::Medium);
        assert!(!plan.side_by_side);
        assert!(!plan.bottom_bar);
        assert!(!plan.tabs_scroll);
        assert!(plan.advanced_default_open);
        assert!(plan.preview_height > 0.0);
        assert!(!plan.touch);
    }

    #[test]
    fn narrow_windows_never_overflow_the_preview() {
        for width in [320.0, 360.0, 420.0, 600.0] {
            let plan = LayoutPlan::new(egui::vec2(width, 640.0));
            assert!(
                plan.preview_height < 640.0 * 0.5,
                "preview too tall at {width}"
            );
            assert!(plan.content_width <= width.max(240.0));
        }
    }

    #[test]
    fn touch_metrics_meet_the_44_point_floor() {
        let metrics = touch_metrics(ScreenClass::Compact);
        assert!(metrics.interact_size.x >= TOUCH_MIN);
        assert!(metrics.interact_size.y >= TOUCH_MIN);
        assert!(metrics.primary_button_height >= 52.0);
        assert!(metrics.slider_rail_height > 8.0);
        // Desktop stays denser than the phone.
        let desktop = touch_metrics(ScreenClass::Wide);
        assert!(desktop.interact_size.y < metrics.interact_size.y);
    }

    #[test]
    fn apply_style_sets_touch_sizes_only_when_compact() {
        let mut compact_style = egui::Style::default();
        LayoutPlan::new(egui::vec2(400.0, 800.0)).apply_style(&mut compact_style);
        assert!(compact_style.spacing.interact_size.y >= TOUCH_MIN);
        assert!(compact_style.spacing.slider_width >= 180.0);

        let mut desktop_style = egui::Style::default();
        LayoutPlan::new(egui::vec2(1180.0, 760.0)).apply_style(&mut desktop_style);
        assert_eq!(
            desktop_style.spacing.interact_size,
            touch_metrics(ScreenClass::Wide).interact_size
        );
    }

    #[test]
    fn class_labels_are_human_readable() {
        assert!(class_label(ScreenClass::Compact).contains("手机"));
        assert!(class_label(ScreenClass::Wide).contains("电脑"));
    }

    #[test]
    fn tiny_windows_use_the_tight_chrome() {
        // 480x820 and 360x740 physical windows on a 200% display are only
        // 240x410 and 180x370 points: QA defect D2.
        let tiny = LayoutPlan::new(egui::vec2(240.0, 410.0));
        assert_eq!(tiny.class, ScreenClass::Compact);
        assert!(tiny.tight);
        assert!(tiny.bottom_bar);

        let tiny_360 = LayoutPlan::new(egui::vec2(180.0, 370.0));
        assert!(tiny_360.tight);

        // A normal phone window keeps the fixed bottom bar and hints.
        let phone = LayoutPlan::new(egui::vec2(433.0, 790.0));
        assert_eq!(phone.class, ScreenClass::Compact);
        assert!(!phone.tight);

        // Desktop never uses the tight chrome.
        let desktop = LayoutPlan::new(egui::vec2(1180.0, 760.0));
        assert!(!desktop.tight);
    }
}
