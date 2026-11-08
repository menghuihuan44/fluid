//! Fluid GUI library: entry points for the desktop binary and the Android shell.
//!
//! The GUI is a thin shell around the fluid-engine pipeline: all capture,
//! interpolation and upscaling work happens on a worker thread while the
//! eframe/egui thread only draws and polls.
//!
//! Layout:
//! * `run()` is the desktop entry point used by src/main.rs (the thin binary).
//! * `android_main()` is the cdylib entry point the Android shell loads;
//!   `android_jni` exports the MediaProjection / Surface install calls.
//! * Everything else (app, worker, sink, fonts, config, layout, ...) is a module
//!   of this library so both targets compile exactly the same UI code.
//!
//! Extra flags exist for automated checks:
//!   --smoke 4                 run the window and auto close after 4 seconds
//!   --smoke-seconds 4          alias of --smoke
//!   --config <path>            use an explicit configuration file
//!   --json-summary <path>      write a machine readable smoke summary
//!   --fullscreen-output        start with the second fullscreen output viewport on
//!   --width / --height <pt>    initial window size (default 1180x760; phones shrink)
//!   --source <screen|window|synthetic>  capture source to select on startup
//!   --window-id <id>           select a window by native id (automation / QA)
//!
//! The window size is clamped to 92% of the smallest monitor and centred, so it
//! never hangs over the taskbar at 200% DPI scaling.
// The Win32 screen-capture exclusion is the only unsafe code in the GUI; it is
// confined to display_affinity.rs where every block carries a SAFETY comment.
#![cfg_attr(not(any(windows, target_os = "android")), forbid(unsafe_code))]
#![cfg_attr(windows, deny(unsafe_code))]
#![allow(dead_code)]

#[cfg(target_os = "android")]
mod android_jni;
mod app;
mod auto_apply;
mod config_io;
#[allow(unsafe_code)]
mod display_affinity;
mod doctor;
mod file_picker;
mod fonts;
mod guide;
mod layout;
mod mapping;
mod prewarm;
mod profiles;
mod sink;
mod status;
mod video_format;
mod video_job;
mod window_source;
mod worker;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::app::{AppOptions, FluidApp, SmokeSnapshot};

const APP_TITLE: &str = "Fluid · 帧生成与画质增强";

#[derive(Clone, Debug, Default)]
struct Args {
    config: Option<PathBuf>,
    smoke_seconds: Option<f32>,
    json_summary: Option<PathBuf>,
    fullscreen_output: bool,
    width: Option<f32>,
    height: Option<f32>,
    source: Option<config_io::SourceChoice>,
    window_id: Option<u64>,
    help: bool,
    version: bool,
}

fn parse_args_from<I>(items: I) -> Args
where
    I: IntoIterator<Item = String>,
{
    let mut args = Args::default();
    let mut iter = items.into_iter().peekable();
    while let Some(item) = iter.next() {
        match item.as_str() {
            "--smoke" | "--smoke-seconds" => {
                let mut seconds = None;
                if iter.peek().is_some_and(|next| !next.starts_with("--"))
                    && let Some(value) = iter.next()
                {
                    seconds = value.parse::<f32>().ok();
                }
                args.smoke_seconds = Some(seconds.unwrap_or(4.0).max(0.5));
            }
            "--config" => {
                if let Some(value) = iter.next() {
                    args.config = Some(PathBuf::from(value));
                }
            }
            "--json-summary" => {
                if let Some(value) = iter.next() {
                    args.json_summary = Some(PathBuf::from(value));
                }
            }
            "--fullscreen-output" => args.fullscreen_output = true,
            "--width" => {
                if let Some(value) = iter.next() {
                    args.width = value.parse::<f32>().ok();
                }
            }
            "--height" => {
                if let Some(value) = iter.next() {
                    args.height = value.parse::<f32>().ok();
                }
            }
            "--source" => {
                if let Some(value) = iter.next() {
                    match config_io::SourceChoice::parse(&value) {
                        Some(choice) => args.source = Some(choice),
                        None => eprintln!("未知来源：{value}（可用 screen / window / synthetic）"),
                    }
                }
            }
            "--window-id" => {
                if let Some(value) = iter.next() {
                    match parse_window_id(&value) {
                        Some(id) => args.window_id = Some(id),
                        None => {
                            eprintln!("窗口 id 无效：{value}（可用十进制或 0x 开头的十六进制）")
                        }
                    }
                }
            }
            "--help" | "-h" => args.help = true,
            "--version" | "-V" => args.version = true,
            other => {
                eprintln!("未知参数：{other}（使用 --help 查看可用参数）");
            }
        }
    }
    args
}

fn parse_args() -> Args {
    parse_args_from(std::env::args().skip(1))
}

/// Parse a native window id, accepting decimal (as printed by
/// `fluid devices --windows`) and 0x-prefixed hexadecimal handles.
fn parse_window_id(text: &str) -> Option<u64> {
    let text = text.trim();
    if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        text.parse::<u64>().ok()
    }
}

fn print_help() {
    println!("Fluid GUI - 帧生成与画质增强");
    println!();
    println!("用法: fluid-gui [选项]");
    println!();
    println!("选项:");
    println!("  --smoke <秒>          启动窗口并在指定秒数后自动关闭（自动化测试用）");
    println!("  --smoke-seconds <秒>  --smoke 的别名");
    println!("  --config <路径>       指定配置文件路径");
    println!("  --json-summary <路径> 退出或停止增强时写出运行摘要 JSON（自动化测试用）");
    println!("  --fullscreen-output   启动时默认开启“全屏输出”第二个窗口");
    println!("  --width <点>          启动窗口宽度（默认 1180，自动限制在当前屏幕的 92% 内）");
    println!("  --height <点>         启动窗口高度（默认 760，自动限制在当前屏幕的 92% 内）");
    println!("  --source <来源>       启动时选择画面来源：screen / window / synthetic");
    println!("  --window-id <id>      启动时按窗口 id 选中“窗口捕获”（十进制或 0x 十六进制）");
    println!("  -h, --help            显示本帮助");
    println!("  -V, --version         显示版本");
}

/// Default window size in points: phones get a portrait window, desktops keep
/// the size the UI was designed for.
fn default_window_size() -> (f32, f32) {
    if cfg!(target_os = "android") {
        (420.0, 800.0)
    } else {
        (1180.0, 760.0)
    }
}

const MIN_WINDOW_W: f32 = 320.0;
const MIN_WINDOW_H: f32 = 480.0;
/// Windows may occupy at most this share of the monitor, so the title bar and
/// the taskbar never clip the window at high DPI scaling.
const MONITOR_FRACTION: f32 = 0.92;

/// Pure clamp: keep the requested window size inside 92% of the smallest
/// monitor, never shrinking below the minimum usable size. Monitor sizes are
/// physical pixels; the caller passes them in the same unit as requested
/// (points at 100% scaling matches pixels on normal displays).
fn fit_to_monitors(requested: (f32, f32), monitors: &[(f32, f32)]) -> (f32, f32) {
    let smallest = monitors
        .iter()
        .copied()
        .filter(|(w, h)| *w > 0.0 && *h > 0.0)
        .reduce(|a, b| if a.0 * a.1 <= b.0 * b.1 { a } else { b });
    let Some((monitor_w, monitor_h)) = smallest else {
        return (requested.0.max(MIN_WINDOW_W), requested.1.max(MIN_WINDOW_H));
    };
    let max_w = (monitor_w * MONITOR_FRACTION).max(MIN_WINDOW_W);
    let max_h = (monitor_h * MONITOR_FRACTION).max(MIN_WINDOW_H);
    (
        requested.0.clamp(MIN_WINDOW_W, max_w),
        requested.1.clamp(MIN_WINDOW_H, max_h),
    )
}

/// Requested size, clamped to the screens reported by the capture backend so
/// the window fits the work area. NativeOptions.centered then centres it.
fn window_size(args: &Args) -> (f32, f32) {
    let (default_w, default_h) = default_window_size();
    let requested = (
        args.width.unwrap_or(default_w),
        args.height.unwrap_or(default_h),
    );
    let monitors: Vec<(f32, f32)> = fluid_capture::enumerate_monitors()
        .unwrap_or_default()
        .into_iter()
        .map(|monitor| (monitor.dims.width as f32, monitor.dims.height as f32))
        .collect();
    fit_to_monitors(requested, &monitors)
}

fn native_options(args: &Args) -> eframe::NativeOptions {
    let (width, height) = window_size(args);
    eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title(APP_TITLE)
            .with_inner_size([width, height])
            .with_min_inner_size([MIN_WINDOW_W, MIN_WINDOW_H]),
        centered: true,
        ..Default::default()
    }
}

fn make_app_options(args: &Args, snapshot: Arc<Mutex<Option<SmokeSnapshot>>>) -> AppOptions {
    AppOptions {
        config_path: args
            .config
            .clone()
            .unwrap_or_else(config_io::default_config_path),
        smoke_seconds: args.smoke_seconds,
        snapshot,
        fullscreen_output: args.fullscreen_output,
        source: args.source,
        window_id: args.window_id,
        json_summary: args.json_summary.clone(),
    }
}

pub(crate) fn write_json_summary(
    path: &PathBuf,
    snapshot: &Option<SmokeSnapshot>,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let body = match snapshot {
        Some(snapshot) => match &snapshot.report {
            Some(report) => format!(
                "{{\"ok\":true,\"frames\":{},\"fps\":{:.2},\"report\":{}}}",
                snapshot.frames,
                snapshot.fps,
                report.to_json()
            ),
            None => format!(
                "{{\"ok\":true,\"frames\":{},\"fps\":{:.2},\"report\":null}}",
                snapshot.frames, snapshot.fps
            ),
        },
        None => "{\"ok\":false,\"frames\":0,\"fps\":0.0,\"report\":null}".to_owned(),
    };
    std::fs::write(path, body)
}

/// Desktop entry point: parse the command line, run the app, report the result.
///
/// Returns the process exit code so src/main.rs stays a one line binary.
pub fn run() -> i32 {
    let args = parse_args();
    if args.help {
        print_help();
        return 0;
    }
    if args.version {
        println!("{}", fluid_core::version_string());
        return 0;
    }

    let snapshot: Arc<Mutex<Option<SmokeSnapshot>>> = Arc::new(Mutex::new(None));
    let snapshot_for_app = Arc::clone(&snapshot);
    let options = native_options(&args);
    let args_for_app = args.clone();

    let run_result = eframe::run_native(
        "fluid-gui",
        options,
        Box::new(move |cc| {
            Ok(Box::new(FluidApp::new(
                cc,
                make_app_options(&args_for_app, snapshot_for_app),
            )))
        }),
    );

    if let Err(err) = run_result {
        eprintln!("GUI 启动失败：{err}");
        return 1;
    }

    if args.smoke_seconds.is_some() {
        let taken = snapshot.lock().ok().and_then(|slot| slot.as_ref().cloned());
        let frames = taken.as_ref().map(|snapshot| snapshot.frames).unwrap_or(0);
        let fps = taken.as_ref().map(|snapshot| snapshot.fps).unwrap_or(0.0);
        println!("GUI OK");
        println!("smoke: frames={frames} fps={fps:.1}");
        if let Some(path) = &args.json_summary {
            if let Err(err) = write_json_summary(path, &taken) {
                eprintln!("写入 JSON 摘要失败：{err}");
            } else {
                println!("summary: {}", path.display());
            }
        }
    }
    0
}

#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
pub extern "C" fn android_main(app: winit::platform::android::activity::AndroidApp) {
    let args = parse_args();
    let snapshot: Arc<Mutex<Option<SmokeSnapshot>>> = Arc::new(Mutex::new(None));
    let snapshot_for_app = Arc::clone(&snapshot);
    let mut options = native_options(&args);
    options.android_app = Some(app);
    let _ = eframe::run_native(
        "fluid-gui",
        options,
        Box::new(move |cc| {
            Ok(Box::new(FluidApp::new(
                cc,
                make_app_options(&args, snapshot_for_app),
            )))
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(items: &[&str]) -> Args {
        parse_args_from(items.iter().map(|item| (*item).to_owned()))
    }

    #[test]
    fn smoke_flag_takes_seconds() {
        let parsed = args(&["--smoke", "4"]);
        assert_eq!(parsed.smoke_seconds, Some(4.0));
        assert!(parsed.json_summary.is_none());
    }

    #[test]
    fn smoke_flag_defaults_to_four_seconds() {
        let parsed = args(&["--smoke"]);
        assert_eq!(parsed.smoke_seconds, Some(4.0));
    }

    #[test]
    fn smoke_seconds_alias_is_accepted() {
        let parsed = args(&["--smoke-seconds", "2.5"]);
        assert_eq!(parsed.smoke_seconds, Some(2.5));
    }

    #[test]
    fn config_and_summary_paths_are_parsed() {
        let parsed = args(&[
            "--config",
            "C:\\tmp\\gui.toml",
            "--json-summary",
            "C:\\tmp\\summary.json",
        ]);
        assert_eq!(parsed.config, Some(PathBuf::from("C:\\tmp\\gui.toml")));
        assert_eq!(
            parsed.json_summary,
            Some(PathBuf::from("C:\\tmp\\summary.json"))
        );
    }

    #[test]
    fn width_and_height_flags_are_parsed() {
        let parsed = args(&["--width", "420", "--height", "800"]);
        assert_eq!(parsed.width, Some(420.0));
        assert_eq!(parsed.height, Some(800.0));
    }

    #[test]
    fn window_size_is_clamped_to_the_smallest_monitor() {
        // A 2880x1800 screen next to a 1366x768 laptop: the laptop limits us.
        let (width, height) =
            fit_to_monitors((1180.0, 760.0), &[(2880.0, 1800.0), (1366.0, 768.0)]);
        assert!((width - 1180.0).abs() < 0.01);
        assert!((height - 768.0 * 0.92).abs() < 0.01);
    }

    #[test]
    fn desktop_default_size_is_kept_on_big_screens() {
        assert_eq!(
            fit_to_monitors((1180.0, 760.0), &[(2880.0, 1800.0)]),
            (1180.0, 760.0)
        );
    }

    #[test]
    fn window_size_respects_the_minimum_on_tiny_screens() {
        assert_eq!(
            fit_to_monitors((1180.0, 760.0), &[(320.0, 240.0)]),
            (320.0, 480.0)
        );
    }

    #[test]
    fn missing_monitors_still_enforce_the_minimum() {
        assert_eq!(fit_to_monitors((100.0, 100.0), &[]), (320.0, 480.0));
    }

    #[test]
    fn fullscreen_output_flag_is_parsed() {
        assert!(args(&["--fullscreen-output"]).fullscreen_output);
        assert!(!args(&[]).fullscreen_output);
    }

    #[test]
    fn source_flag_is_parsed() {
        assert_eq!(
            args(&["--source", "window"]).source,
            Some(config_io::SourceChoice::Window)
        );
        assert_eq!(
            args(&["--source", "synthetic"]).source,
            Some(config_io::SourceChoice::Synthetic)
        );
        assert_eq!(
            args(&["--source", "screen"]).source,
            Some(config_io::SourceChoice::Screen)
        );
        // An unknown value must not crash the app or pick a surprise source.
        assert_eq!(args(&["--source", "chromium"]).source, None);
        assert_eq!(args(&[]).source, None);
    }

    #[test]
    fn window_id_accepts_decimal_and_hex() {
        assert_eq!(args(&["--window-id", "462856"]).window_id, Some(462_856));
        assert_eq!(args(&["--window-id", "0x710c8"]).window_id, Some(0x710c8));
        assert_eq!(args(&["--window-id", "0xZZ"]).window_id, None);
        assert_eq!(args(&["--window-id", "not-a-number"]).window_id, None);
    }

    #[test]
    fn help_and_version_flags_are_recognised() {
        assert!(args(&["--help"]).help);
        assert!(args(&["-V"]).version);
        assert!(!args(&[]).help);
    }

    #[test]
    fn json_summary_without_snapshot_reports_failure() {
        let dir = std::env::temp_dir().join(format!("fluid-gui-smoke-{}", std::process::id()));
        let file = dir.join("summary.json");
        write_json_summary(&file, &None).expect("write summary");
        let text = std::fs::read_to_string(&file).expect("read summary");
        assert!(text.contains("\"ok\":false"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
