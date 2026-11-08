//! Pure status-strip and label helpers.
//!
//! Keeping this logic free of egui makes it unit-testable, and guarantees the
//! UI never shows a misleading "0 fps" just because a measurement window is
//! short or the pipeline paused.

use fluid_core::Dims;

/// Texts for the live status strip.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FpsTexts {
    /// Value shown next to "输入".
    pub input: String,
    /// Value shown next to "输出".
    pub output: String,
    /// Optional warning shown after the numbers.
    pub note: Option<String>,
}

fn fps_value(fps: f32) -> String {
    if fps >= 100.0 {
        format!("{fps:.0} fps")
    } else {
        format!("{fps:.1} fps")
    }
}

fn pair(out_fps: f32, older: bool, multiplier: u32) -> (String, String) {
    let multiplier = multiplier.max(1) as f32;
    let input = out_fps / multiplier;
    if older {
        (
            format!("上次 {}", fps_value(input)),
            format!("上次 {}", fps_value(out_fps)),
        )
    } else {
        (fps_value(input), fps_value(out_fps))
    }
}

/// Decide what the status strip should say.
///
/// * `running`: a pipeline run is active.
/// * `live_out_fps`: what the GUI measured live (0.0 until it has two samples).
/// * `last_out_fps`: last known output frame rate (from an earlier measurement or report).
/// * `stalled_for`: seconds since the last preview frame, only meaningful while running.
pub fn fps_texts(
    running: bool,
    live_out_fps: f32,
    last_out_fps: Option<f32>,
    multiplier: u32,
    stalled_for: Option<f32>,
) -> FpsTexts {
    if running {
        if let Some(stalled) = stalled_for
            && stalled >= 2.0
        {
            return FpsTexts {
                input: "—".to_owned(),
                output: "停滞".to_owned(),
                note: Some(format!("已 {stalled:.1} 秒没有新画面，可能卡住了")),
            };
        }
        if live_out_fps > 0.05 {
            let (input, output) = pair(live_out_fps, false, multiplier);
            return FpsTexts {
                input,
                output,
                note: None,
            };
        }
        if let Some(last) = last_out_fps.filter(|fps| *fps > 0.05) {
            let (input, output) = pair(last, true, multiplier);
            return FpsTexts {
                input,
                output,
                note: Some("正在测量…".to_owned()),
            };
        }
        return FpsTexts {
            input: "—".to_owned(),
            output: "测量中…".to_owned(),
            note: None,
        };
    }

    match last_out_fps.filter(|fps| *fps > 0.05) {
        Some(last) => {
            let (input, output) = pair(last, true, multiplier);
            FpsTexts {
                input,
                output,
                note: None,
            }
        }
        None => FpsTexts {
            input: "—".to_owned(),
            output: "—".to_owned(),
            note: None,
        },
    }
}

/// Label for the built-in synthetic source, stating the real frame size in use.
pub fn synthetic_label(dims: Dims) -> String {
    format!("{}x{} 移动测试图案（无需游戏）", dims.width, dims.height)
}

/// Friendly name for a capture-source backend name reported by fluid-capture.
///
/// The names match `FrameSource::backend_name` ("dxgi", "window-wgc",
/// "window-dxgi-crop", "synthetic", "mock", "android-mediaprojection").
pub fn source_backend_label(name: &str) -> String {
    match name.trim() {
        "" => "—".to_owned(),
        "synthetic" | "mock" => "内置画面".to_owned(),
        "dxgi" => "屏幕（DXGI）".to_owned(),
        "window-wgc" => "窗口（WGC）".to_owned(),
        "window-dxgi-crop" => "窗口（DXGI 裁剪）".to_owned(),
        "android-mediaprojection" => "屏幕（MediaProjection）".to_owned(),
        other => other.to_owned(),
    }
}

/// What the status strip says about the capture source.
///
/// While a run is active the pipeline has not reported yet, so the backend
/// name the worker captured at spawn time is shown; afterwards the finished
/// report is authoritative (window capture can fall back from WGC to a DXGI
/// crop, and the strip must show which one actually ran).
pub fn source_text(
    running: bool,
    live_backend: Option<&str>,
    report_source: Option<&str>,
) -> String {
    if running {
        return match live_backend {
            Some(name) => format!("{}（运行中）", source_backend_label(name)),
            None => "—".to_owned(),
        };
    }
    match report_source {
        Some(name) => source_backend_label(name),
        None => "—".to_owned(),
    }
}

/// Whether the extra fullscreen output viewport should be shown.
pub fn output_visible(enabled: bool, running: bool) -> bool {
    enabled && running
}

/// What the status strip says about the upscaling backend.
///
/// While a run is active the engine has not reported yet, so the strip shows
/// the backend the run was *started* with (QA defect D4): picking another
/// backend mid-run must not look like the running pipeline already switched,
/// so the new choice is marked as taking effect next time. Afterwards the
/// finished report is authoritative, including "未启用放大" when upscaling
/// was off.
pub fn backend_text(
    running: bool,
    run_label: &str,
    current_label: &str,
    report_backend: Option<&str>,
) -> String {
    if running {
        return running_backend_label(run_label, current_label);
    }
    match report_backend {
        Some(name) => crate::mapping::backend_display_name(name),
        None => "—".to_owned(),
    }
}

/// Label used while a run is active: what it actually started with, plus a
/// "next run" marker when the user has already picked something else.
fn running_backend_label(run_label: &str, current_label: &str) -> String {
    if run_label == current_label {
        format!("{run_label}（运行中）")
    } else {
        format!("{run_label}（运行中） · 下次生效：{current_label}")
    }
}

/// What the status strip says about the frame generation (插帧) backend.
///
/// Same contract as backend_text: while running we can only show the request,
/// afterwards the engine report is authoritative because Auto may fall back
/// from the GPU flow path to the CPU path.
pub fn interp_backend_text(
    running: bool,
    run_label: &str,
    current_label: &str,
    report_backend: Option<&str>,
) -> String {
    if running {
        return running_backend_label(run_label, current_label);
    }
    match report_backend {
        Some(name) => crate::mapping::interp_backend_display_name(name),
        None => "—".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(width: u32, height: u32) -> Dims {
        Dims::new(width, height)
    }

    #[test]
    fn stopped_run_shows_last_measured_values() {
        let texts = fps_texts(false, 0.0, Some(27.5), 2, None);
        assert_eq!(texts.output, "上次 27.5 fps");
        assert_eq!(texts.input, "上次 13.8 fps");
        assert!(texts.note.is_none());
    }

    #[test]
    fn stopped_run_without_history_shows_dashes() {
        let texts = fps_texts(false, 0.0, None, 1, None);
        assert_eq!(texts.output, "—");
        assert_eq!(texts.input, "—");
    }

    #[test]
    fn live_value_is_shown_while_running() {
        let texts = fps_texts(true, 29.4, Some(12.0), 2, Some(0.1));
        assert_eq!(texts.output, "29.4 fps");
        assert_eq!(texts.input, "14.7 fps");
        assert!(texts.note.is_none());
    }

    #[test]
    fn short_measurement_falls_back_to_last_value() {
        let texts = fps_texts(true, 0.0, Some(30.0), 3, Some(0.2));
        assert_eq!(texts.output, "上次 30.0 fps");
        assert_eq!(texts.input, "上次 10.0 fps");
        assert_eq!(texts.note.as_deref(), Some("正在测量…"));
    }

    #[test]
    fn fresh_run_without_numbers_never_shows_zero() {
        let texts = fps_texts(true, 0.0, None, 2, None);
        assert_eq!(texts.output, "测量中…");
        assert!(!texts.output.contains('0'));
    }

    #[test]
    fn stalled_pipeline_is_surfaced_as_warning() {
        let texts = fps_texts(true, 29.0, Some(29.0), 2, Some(5.5));
        assert_eq!(texts.output, "停滞");
        assert_eq!(
            texts.note.as_deref(),
            Some("已 5.5 秒没有新画面，可能卡住了")
        );
    }

    #[test]
    fn synthetic_label_states_the_real_size() {
        assert_eq!(
            synthetic_label(dims(640, 360)),
            "640x360 移动测试图案（无需游戏）"
        );
        assert_eq!(
            synthetic_label(dims(1280, 720)),
            "1280x720 移动测试图案（无需游戏）"
        );
    }

    #[test]
    fn output_viewport_needs_both_toggle_and_run() {
        assert!(!output_visible(false, true));
        assert!(!output_visible(true, false));
        assert!(output_visible(true, true));
    }

    #[test]
    fn backend_text_shows_the_request_while_running() {
        let text = backend_text(true, "自动（推荐）", "自动（推荐）", Some("cpu"));
        assert_eq!(text, "自动（推荐）（运行中）");
    }

    #[test]
    fn backend_text_reports_what_actually_ran() {
        assert_eq!(
            backend_text(false, "自动（推荐）", "自动（推荐）", Some("cpu")),
            "CPU（处理器）"
        );
        assert_eq!(
            backend_text(false, "GPU", "GPU", Some("d3d11-compute")),
            "GPU（Direct3D 11 计算）"
        );
        assert_eq!(
            backend_text(false, "自动（推荐）", "自动（推荐）", Some("none")),
            "未启用放大"
        );
    }

    #[test]
    fn backend_text_is_dash_before_the_first_run() {
        assert_eq!(
            backend_text(false, "自动（推荐）", "自动（推荐）", None),
            "—"
        );
    }

    #[test]
    fn source_backend_labels_are_human_readable() {
        assert_eq!(source_backend_label("dxgi"), "屏幕（DXGI）");
        assert_eq!(source_backend_label("window-wgc"), "窗口（WGC）");
        assert_eq!(
            source_backend_label("window-dxgi-crop"),
            "窗口（DXGI 裁剪）"
        );
        assert_eq!(source_backend_label("synthetic"), "内置画面");
        assert_eq!(source_backend_label("mock"), "内置画面");
        assert_eq!(
            source_backend_label("android-mediaprojection"),
            "屏幕（MediaProjection）"
        );
        assert_eq!(source_backend_label(""), "—");
        // Unknown future backends stay visible instead of turning into "—".
        assert_eq!(source_backend_label("kms"), "kms");
    }

    #[test]
    fn source_text_shows_the_live_backend_while_running() {
        assert_eq!(
            source_text(true, Some("window-wgc"), Some("dxgi")),
            "窗口（WGC）（运行中）"
        );
        assert_eq!(source_text(true, None, None), "—");
    }

    #[test]
    fn source_text_reports_the_backend_that_actually_ran() {
        // Window capture can fall back to a DXGI crop; the report wins.
        assert_eq!(
            source_text(false, Some("window-wgc"), Some("window-dxgi-crop")),
            "窗口（DXGI 裁剪）"
        );
        assert_eq!(source_text(false, None, None), "—");
    }

    #[test]
    fn interp_backend_text_shows_the_request_while_running() {
        assert_eq!(
            interp_backend_text(true, "自动（推荐）", "自动（推荐）", Some("cpu")),
            "自动（推荐）（运行中）"
        );
    }

    #[test]
    fn interp_backend_text_reports_what_actually_ran() {
        assert_eq!(
            interp_backend_text(false, "自动（推荐）", "自动（推荐）", Some("cpu")),
            "CPU（处理器）"
        );
        assert_eq!(
            interp_backend_text(false, "自动（推荐）", "自动（推荐）", Some("wgpu-flow")),
            "GPU（计算）"
        );
        assert_eq!(
            interp_backend_text(false, "自动（推荐）", "自动（推荐）", Some("none")),
            "未启用插帧"
        );
        assert_eq!(
            interp_backend_text(false, "自动（推荐）", "自动（推荐）", None),
            "—"
        );
    }

    #[test]
    fn backend_picked_mid_run_is_marked_as_taking_effect_next_time() {
        // QA defect D4: changing the dropdown while running must not claim the
        // running pipeline already switched backends.
        let text = backend_text(true, "自动（推荐）", "GPU", Some("cpu"));
        assert_eq!(text, "自动（推荐）（运行中） · 下次生效：GPU");
        assert!(!text.starts_with("GPU"));
        let interp = interp_backend_text(true, "CPU", "GPU", Some("cpu"));
        assert_eq!(interp, "CPU（运行中） · 下次生效：GPU");
    }

    #[test]
    fn unchanged_backend_while_running_has_no_next_run_marker() {
        let text = backend_text(true, "CPU", "CPU", Some("d3d11-compute"));
        assert_eq!(text, "CPU（运行中）");
        assert!(!text.contains("下次生效"));
    }
}
