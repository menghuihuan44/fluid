//! The egui application: four tabs, a non-blocking pipeline worker and CJK fonts.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use egui::{Color32, RichText};

use fluid_core::{Dims, FluidConfig, FpsMeter, FrameSource, PixelFormat, ScaleConfig, now_ns};
use fluid_engine::PipelineReport;

use crate::auto_apply::{self, AppRule};
use crate::config_io::{self, SourceChoice};
use crate::display_affinity;
use crate::doctor::{DoctorHandle, DoctorState};
use crate::file_picker;
use crate::fonts;
use crate::guide;
use crate::layout::{self, LayoutPlan, TOUCH_MIN};
use crate::mapping::{
    BackendChoice, FilterChoice, InterpBackendChoice, InterpChoice, PresetChoice, SYNTHETIC_DIMS,
    ScaleChoice, UiQuality, backend_display_name, interp_backend_display_name,
};
use crate::prewarm::{self, Prewarm, PrewarmState};
use crate::profiles;
use crate::status;
use crate::video_format::{self, VideoFormat};
use crate::video_job::{self, VideoInputInfo, VideoJob, VideoState};
use crate::window_source;
use crate::worker::RunHandle;

/// The window may occupy at most this share of the monitor (QA defect D5).
const MONITOR_FRACTION: f32 = 0.92;
/// Minimum usable window size in points (also used as the window minimum).
const MIN_WINDOW_W: f32 = 320.0;
const MIN_WINDOW_H: f32 = 480.0;

/// Command line options handed from main to the app.
pub struct AppOptions {
    pub config_path: PathBuf,
    pub smoke_seconds: Option<f32>,
    pub snapshot: Arc<Mutex<Option<SmokeSnapshot>>>,
    /// Start with the 全屏输出 second viewport enabled (used by `--fullscreen-output`).
    pub fullscreen_output: bool,
    /// Source forced by `--source` (overrides the saved preference).
    pub source: Option<SourceChoice>,
    /// Window id forced by `--window-id` (implies the 窗口捕获 source).
    pub window_id: Option<u64>,
    /// Write the json summary after every run, not only in smoke mode (QA D5).
    pub json_summary: Option<PathBuf>,
}

/// Written back to the json summary file after a smoke run.
#[derive(Clone, Debug, Default)]
pub struct SmokeSnapshot {
    pub frames: u64,
    pub fps: f32,
    pub report: Option<PipelineReport>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tab {
    Live,
    Video,
    Doctor,
    About,
}

impl Tab {
    const ALL: [Tab; 4] = [Tab::Live, Tab::Video, Tab::Doctor, Tab::About];

    fn label(self) -> &'static str {
        match self {
            Tab::Live => "实时增强",
            Tab::Video => "视频处理",
            Tab::Doctor => "性能体检",
            Tab::About => "关于",
        }
    }
}

/// Keys that switch tabs, in the same order as [`Tab::ALL`]. Both Ctrl and Alt
/// are accepted: on some machines (including the development one) an IME or a
/// shell helper swallows Ctrl+1 before it ever reaches the window, so Alt+1..4
/// keeps every tab reachable from the keyboard.
const TAB_SHORTCUTS: [(egui::Key, Tab); 4] = [
    (egui::Key::Num1, Tab::Live),
    (egui::Key::Num2, Tab::Video),
    (egui::Key::Num3, Tab::Doctor),
    (egui::Key::Num4, Tab::About),
];

/// Application state. The GUI thread only ever draws and polls.
pub struct FluidApp {
    tab: Tab,

    config_path: PathBuf,
    config: FluidConfig,
    quality: UiQuality,
    saved_quality: UiQuality,

    source: SourceChoice,
    monitors: Vec<fluid_capture::MonitorInfo>,
    monitor_error: Option<String>,
    monitor_idx: usize,

    /// Windows that can be captured (Windows only; empty until the first refresh).
    windows: Vec<fluid_capture::WindowInfo>,
    /// Error from the last window enumeration, shown inline.
    windows_error: Option<String>,
    /// True once the window list has been fetched, so it is not refetched per frame.
    windows_loaded: bool,
    /// Search box for the window list.
    window_filter: String,
    /// Native id of the selected window (also persisted as config.capture.window).
    window_selected: Option<u64>,
    /// Whether the window list is expanded.
    window_panel_open: bool,
    /// Friendly inline error for 窗口捕获 (no window picked, window gone).
    window_error: Option<String>,
    /// True while a window-capture run has lost its window (closed/minimized).
    window_lost: bool,
    /// Last time the captured window was re-checked (throttles enumeration).
    window_watch_at: Option<Instant>,
    /// True while `--source`/`--window-id` forced a source (smoke keeps it).
    forced_source: bool,
    /// Set when something outside 画质设置 needs to be persisted (window choice).
    config_dirty: bool,

    // ------------------------------------------------------------ profiles
    /// Directory with named profiles (profiles/<name>.toml next to gui.toml).
    profiles_dir: PathBuf,
    /// Profile names on disk, refreshed after every create/rename/delete.
    profile_names: Vec<String>,
    /// Profile currently applied; None means the settings were edited by hand.
    active_profile: Option<String>,
    /// Whether the 配置档 dialog is open.
    profiles_open: bool,
    /// Text field for 新建 / 重命名.
    profile_name: String,
    /// Profile being renamed (None means create mode).
    profile_rename: Option<String>,
    /// Inline error shown inside the profile dialog.
    profile_error: Option<String>,
    /// Inline success message shown inside the profile dialog.
    profile_notice: Option<String>,
    /// Apply the last used profile on the next start.
    apply_last_profile: bool,
    /// Silently switch profiles when the captured window matches a rule.
    auto_apply: bool,
    /// "process name -> profile" rules loaded from gui-app-rules.conf.
    rules: Vec<AppRule>,
    /// Path of the per-application rule file.
    rules_path: PathBuf,
    /// Text field for adding a rule (program name).
    rule_process: String,
    /// Profile picked for the new rule.
    rule_profile: Option<String>,
    /// Window id a rule was already applied for (avoids re-applying each refresh).
    auto_applied_for: Option<u64>,
    /// Short hint shown after an automatic profile switch.
    auto_apply_hint: Option<String>,
    /// Whether the first-run guide card was dismissed.
    guide_dismissed: bool,

    font_path: Option<&'static str>,
    warning: Option<String>,
    error_message: Option<String>,

    run: Option<RunHandle>,
    run_started: Instant,
    run_multiplier: u32,
    /// Upscaling backend label chosen when the current run started (QA D4).
    run_backend_label: String,
    /// Frame-generation backend label chosen when the current run started.
    run_interp_label: String,
    last_generation: u64,
    frames: u64,
    fps: FpsMeter,
    /// Dimensions the built-in synthetic source will use next time it starts.
    synthetic_dims: Dims,
    /// When the last preview frame arrived (None until the first one).
    last_frame_at: Option<Instant>,
    /// Last measured output frame rate, so the strip can say "上次 X fps".
    last_out_fps: Option<f32>,
    /// User toggle for the extra fullscreen output viewport.
    fullscreen_output: bool,
    /// When the fullscreen output viewport was opened, used to show a short Esc hint.
    output_opened_at: Option<Instant>,
    /// Result of hiding the fullscreen output window from screen capture.
    output_affinity: Option<display_affinity::Affinity>,
    /// 高级设置 expanded state; None until the first frame picks the layout default.
    advanced_open: Option<bool>,
    /// Set when the advanced-open preference has to be written to disk.
    ui_flags_dirty: bool,
    /// Side-car file for UI-only preferences (the TOML config belongs to fluid-core).
    ui_flags_path: PathBuf,
    /// Layout class of the last frame, shown in 关于 / 性能体检.
    layout_class: layout::ScreenClass,
    /// Set once the first frame has clamped/centred the window to the monitor.
    window_fitted: bool,
    texture: Option<egui::TextureHandle>,
    texture_dims: (u32, u32),
    last_report: Option<PipelineReport>,
    stopping_since: Option<Instant>,
    forced_stop_note: bool,

    video_input: String,
    video_output: String,
    video_job: Option<VideoJob>,
    video_state: VideoState,
    video_error: Option<String>,
    /// JPEG quality for the AVI output (40..=100, default 88).
    video_quality: u8,
    /// Container details of the chosen input file, shown before starting.
    video_input_info: Option<VideoInputInfo>,
    video_input_error: Option<String>,
    /// Last finished output file, for the 打开输出文件夹 button.
    video_output_path: Option<PathBuf>,
    video_reveal_error: Option<String>,
    /// Background GPU warm-up (task-45).
    prewarm: Option<Prewarm>,
    prewarm_state: PrewarmState,

    doctor: Option<DoctorHandle>,
    doctor_state: DoctorState,
    doctor_ran: bool,

    smoke_seconds: Option<f32>,
    /// Set when the first frame is presented, so --smoke measures real run time.
    smoke_deadline: Option<Instant>,
    /// Upper bound on waiting for that first frame (never hang in automation).
    smoke_hard_deadline: Option<Instant>,
    smoke_started: bool,
    smoke_stop_sent: bool,
    smoke_done: bool,
    snapshot: Arc<Mutex<Option<SmokeSnapshot>>>,

    /// Where the json summary is written after each run (QA D5).
    json_summary: Option<PathBuf>,

    threads: usize,
    display: String,
}

impl FluidApp {
    pub fn new(cc: &eframe::CreationContext<'_>, options: AppOptions) -> Self {
        let font_path = fonts::install(&cc.egui_ctx);
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        cc.egui_ctx.all_styles_mut(|style| {
            style.spacing.item_spacing = egui::vec2(8.0, 8.0);
            style.spacing.button_padding = egui::vec2(12.0, 7.0);
        });

        let mut loaded = config_io::load(&options.config_path);
        let ui_flags_path = config_io::ui_flags_path(&options.config_path);
        let ui_flags = config_io::load_ui_flags(&ui_flags_path);

        // Named profiles (task-39): the profile directory and the per-program
        // rules live next to gui.toml. Applying the last profile on startup is
        // skipped during --smoke runs, which must stay deterministic.
        let profiles_dir = profiles::profiles_dir(&options.config_path);
        let profile_names = profiles::list_profiles(&profiles_dir);
        let rules_path = auto_apply::rules_path(&options.config_path);
        let rules = auto_apply::load_rules(&rules_path);
        let apply_last_profile = ui_flags.apply_last_profile.unwrap_or(false);
        let auto_apply = ui_flags.auto_apply.unwrap_or(true);
        let guide_dismissed = ui_flags.guide_dismissed.unwrap_or(false);
        let mut active_profile = None;
        let mut profile_notice = None;
        if options.smoke_seconds.is_none()
            && apply_last_profile
            && let Some(name) = ui_flags.last_profile.clone()
            && profile_names.contains(&name)
        {
            match profiles::load_profile(&profiles_dir, &name) {
                Ok(config) => {
                    loaded.config = config;
                    active_profile = Some(name.clone());
                    profile_notice = Some(format!("已应用上次的配置档「{name}」。"));
                }
                Err(err) => {
                    loaded.warning = Some(match loaded.warning.take() {
                        Some(existing) => format!("{existing}；{err}"),
                        None => err,
                    });
                }
            }
        }

        let (monitors, monitor_error) = match fluid_capture::enumerate_monitors() {
            Ok(monitors) => (monitors, None),
            Err(err) => (Vec::new(), Some(format!("无法列出显示器：{err}"))),
        };
        let display = monitors
            .iter()
            .find(|monitor| monitor.primary)
            .or_else(|| monitors.first())
            .map(|monitor| {
                format!(
                    "{}（{}x{}，{:.0} Hz）",
                    monitor.name, monitor.dims.width, monitor.dims.height, monitor.refresh_hz
                )
            })
            .unwrap_or_else(|| "未检测到显示器".to_owned());
        let monitor_idx = monitors
            .iter()
            .position(|monitor| monitor.index == loaded.config.capture.monitor)
            .unwrap_or(0);

        let quality = UiQuality::from_config(&loaded.config);

        // Source: command line wins, then the remembered choice, then "the last
        // window that was captured", otherwise the built-in pattern.
        let mut source = options.source.or(ui_flags.source).unwrap_or_else(|| {
            if loaded
                .config
                .capture
                .window
                .as_ref()
                .is_some_and(|window| window.id != 0)
            {
                SourceChoice::Window
            } else {
                SourceChoice::Synthetic
            }
        });
        let mut window_selected = loaded
            .config
            .capture
            .window
            .as_ref()
            .map(|window| window.id)
            .filter(|id| *id != 0);
        if let Some(id) = options.window_id {
            window_selected = Some(id);
            if options.source.is_none() {
                source = SourceChoice::Window;
            }
        }
        // Phones only get whole-screen MediaProjection capture, so 窗口捕获 is
        // Windows-only; fall back quietly instead of failing when a run starts.
        if cfg!(target_os = "android") && source == SourceChoice::Window {
            source = SourceChoice::Synthetic;
        }
        let forced_source = options.source.is_some() || options.window_id.is_some();
        let window_panel_open = source == SourceChoice::Window && window_selected.is_none();

        let threads = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(0);

        // The smoke clock starts when the pipeline presents its first frame
        // (a cold GPU backend can take seconds to initialise); this hard bound
        // makes sure a broken pipeline still exits instead of hanging.
        let smoke_hard_deadline = options.smoke_seconds.map(|seconds| {
            Instant::now() + Duration::from_secs_f32(seconds.max(0.5)) + Duration::from_secs(25)
        });

        // Warm the GPU on a worker thread so the first 开始增强 does not sit
        // several seconds without frames (task-45). Skipped in --smoke runs so
        // automation stays deterministic.
        let prewarm = if options.smoke_seconds.is_none() {
            Some(Prewarm::spawn())
        } else {
            None
        };

        Self {
            tab: Tab::Live,
            config_path: options.config_path,
            config: loaded.config,
            quality,
            saved_quality: quality,
            source,
            monitors,
            monitor_error,
            monitor_idx,
            windows: Vec::new(),
            windows_error: None,
            windows_loaded: false,
            window_filter: String::new(),
            window_selected,
            window_panel_open,
            window_error: None,
            window_lost: false,
            window_watch_at: None,
            forced_source,
            config_dirty: false,
            profiles_dir,
            profile_names,
            active_profile,
            profiles_open: false,
            profile_name: String::new(),
            profile_rename: None,
            profile_error: None,
            profile_notice,
            apply_last_profile,
            auto_apply,
            rules,
            rules_path,
            rule_process: String::new(),
            rule_profile: None,
            auto_applied_for: None,
            auto_apply_hint: None,
            guide_dismissed,
            font_path,
            warning: loaded.warning,
            error_message: None,
            run: None,
            run_started: Instant::now(),
            run_multiplier: 1,
            run_backend_label: quality.backend.label().to_owned(),
            run_interp_label: quality.interp_backend.label().to_owned(),
            last_generation: 0,
            frames: 0,
            fps: FpsMeter::one_second(),
            synthetic_dims: SYNTHETIC_DIMS,
            last_frame_at: None,
            last_out_fps: None,
            fullscreen_output: options.fullscreen_output,
            output_opened_at: None,
            output_affinity: None,
            advanced_open: ui_flags.advanced_open,
            ui_flags_dirty: false,
            ui_flags_path,
            layout_class: layout::ScreenClass::Wide,
            window_fitted: false,
            texture: None,
            texture_dims: (0, 0),
            last_report: None,
            stopping_since: None,
            forced_stop_note: false,
            video_input: String::new(),
            video_output: String::new(),
            video_job: None,
            video_state: VideoState::default(),
            video_error: None,
            video_quality: video_format::QUALITY_DEFAULT,
            video_input_info: None,
            video_input_error: None,
            video_output_path: None,
            video_reveal_error: None,
            prewarm,
            prewarm_state: PrewarmState::default(),
            doctor: None,
            doctor_state: DoctorState::default(),
            doctor_ran: false,
            smoke_seconds: options.smoke_seconds,
            smoke_deadline: None,
            smoke_hard_deadline,
            smoke_started: false,
            smoke_stop_sent: false,
            smoke_done: false,
            snapshot: options.snapshot,
            json_summary: options.json_summary,
            threads,
            display,
        }
    }

    // ---------------------------------------------------------- window fit

    /// Desktop window fit (QA defect D5). eframe centres the window using the
    /// monitor size, but at 200% DPI scaling the title bar can still hang over
    /// the taskbar. Once the viewport reports real sizes, clamp the window to
    /// 92% of the monitor (minus chrome) and centre the *outer* rect.
    fn fit_window(&mut self, ctx: &egui::Context) {
        if self.window_fitted {
            return;
        }
        let (monitor, inner, outer) = ctx.input(|input| {
            let viewport = input.viewport();
            (
                viewport.monitor_size,
                viewport.inner_rect,
                viewport.outer_rect,
            )
        });
        let Some(monitor) = monitor else {
            return;
        };
        let Some(inner) = inner else {
            return;
        };
        let chrome = outer
            .map(|outer| (outer.size() - inner.size()).max(egui::Vec2::ZERO))
            .unwrap_or(egui::Vec2::ZERO);
        let max_inner =
            (monitor * MONITOR_FRACTION - chrome).max(egui::vec2(MIN_WINDOW_W, MIN_WINDOW_H));
        let desired = inner
            .size()
            .min(max_inner)
            .max(egui::vec2(MIN_WINDOW_W, MIN_WINDOW_H));
        if (desired - inner.size()).length() > 0.5 {
            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(desired));
        }
        let new_outer = desired + chrome;
        let position = ((monitor - new_outer) * 0.5).max(egui::Vec2::ZERO);
        ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(position.to_pos2()));
        self.window_fitted = true;
    }

    // ------------------------------------------------------------- polling

    fn poll(&mut self, ctx: &egui::Context) {
        self.smoke_start();
        self.prewarm_tick();
        self.windows_tick();
        self.window_watch_tick();
        self.worker_tick(ctx);
        self.video_tick();
        self.doctor_tick();
        self.config_tick();
        self.smoke_lifecycle(ctx);

        let busy = self.run.is_some()
            || self.video_job.is_some()
            || self
                .prewarm
                .as_ref()
                .is_some_and(|prewarm| !prewarm.is_finished())
            || self
                .doctor
                .as_ref()
                .map(|doctor| doctor.is_running())
                .unwrap_or(false)
            || self.smoke_deadline.is_some()
            || (self.smoke_seconds.is_some() && !self.smoke_done);
        if busy {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
    }

    fn worker_tick(&mut self, ctx: &egui::Context) {
        if self.run.is_none() {
            return;
        }

        let mut finished = false;
        let mut error = None;
        let mut pending_frame = None;
        if let Some(run) = &self.run {
            let status = run.status();
            finished = status.finished;
            error = status.error;
            if status.generation != self.last_generation
                && let Some((generation, frame)) = run.take_frame()
            {
                self.last_generation = generation;
                self.frames = status.frames;
                pending_frame = Some(frame);
            }
        }

        if let Some(frame) = pending_frame {
            self.fps.tick(now_ns());
            self.last_frame_at = Some(Instant::now());
            let live = self.fps.fps();
            if live > 0.05 {
                self.last_out_fps = Some(live);
            }
            self.update_texture(ctx, &frame);
        }
        if let Some(message) = error {
            self.error_message = Some(message);
        }

        if finished {
            if let Some(mut run) = self.run.take() {
                run.join();
                if let Some(report) = run.take_report() {
                    self.last_out_fps = Some(report.stats.output_fps_value());
                    self.last_report = Some(report);
                }
            }
            self.stopping_since = None;
            self.forced_stop_note = false;
            self.window_lost = false;
            self.window_watch_at = None;
            // QA defect D5: interactive runs write the same summary file as
            // --smoke, as soon as the worker published its report.
            self.save_json_summary();
        }
    }

    fn video_tick(&mut self) {
        if let Some(job) = self.video_job.as_ref() {
            self.video_state = job.state();
        }
        let finished = self
            .video_job
            .as_ref()
            .map(|job| job.is_finished())
            .unwrap_or(false);
        if finished && let Some(mut job) = self.video_job.take() {
            job.join();
            self.video_state = job.state();
        }
    }

    /// Poll the background GPU warm-up; it never blocks the UI thread.
    fn prewarm_tick(&mut self) {
        if let Some(prewarm) = self.prewarm.as_ref() {
            self.prewarm_state = prewarm.state();
        }
        let finished = self
            .prewarm
            .as_ref()
            .map(Prewarm::is_finished)
            .unwrap_or(true);
        if finished && let Some(mut prewarm) = self.prewarm.take() {
            prewarm.join();
            self.prewarm_state = prewarm.state();
        }
    }

    fn doctor_tick(&mut self) {
        if let Some(doctor) = self.doctor.as_ref() {
            self.doctor_state = doctor.snapshot();
        }
        let finished = self
            .doctor
            .as_ref()
            .map(|doctor| !doctor.is_running())
            .unwrap_or(false);
        if finished && let Some(mut doctor) = self.doctor.take() {
            doctor.join();
        }
    }

    fn config_tick(&mut self) {
        if self.quality == self.saved_quality && !self.config_dirty {
            return;
        }
        self.quality.apply(&mut self.config);
        if self.smoke_deadline.is_none()
            && let Err(err) = config_io::save(&self.config, &self.config_path)
        {
            self.warning = Some(format!("配置保存失败：{err}"));
        }
        self.saved_quality = self.quality;
        self.config_dirty = false;
    }

    // ------------------------------------------------- profiles & auto-apply

    /// Persist the UI side-car file (advanced state, source, profile prefs, guide).
    fn persist_ui_flags(&mut self) {
        let flags = config_io::UiFlags {
            advanced_open: self.advanced_open,
            source: Some(self.source),
            apply_last_profile: Some(self.apply_last_profile),
            last_profile: self.active_profile.clone(),
            auto_apply: Some(self.auto_apply),
            guide_dismissed: Some(self.guide_dismissed),
        };
        if let Err(err) = config_io::save_ui_flags(&flags, &self.ui_flags_path) {
            self.warning = Some(err);
        }
    }

    /// Re-read the profile directory; call after every create/rename/delete.
    fn refresh_profiles(&mut self) {
        self.profile_names = profiles::list_profiles(&self.profiles_dir);
    }

    /// Apply a profile to the settings (takes effect on the next run).
    fn apply_profile(&mut self, name: &str) {
        self.apply_profile_inner(name, true);
    }

    /// Load one profile into the live settings. Returns false on failure so the
    /// automatic per-window switch can report a deleted profile.
    fn apply_profile_inner(&mut self, name: &str, announce: bool) -> bool {
        match profiles::load_profile(&self.profiles_dir, name) {
            Ok(config) => {
                self.config = config;
                self.quality = UiQuality::from_config(&self.config);
                self.saved_quality = self.quality;
                self.monitor_idx = self
                    .monitors
                    .iter()
                    .position(|monitor| monitor.index == self.config.capture.monitor)
                    .unwrap_or(0);
                self.active_profile = Some(name.to_owned());
                // The TOML config follows the profile, so the next launch starts
                // from the same settings even with 启动时应用上次的配置 turned off.
                self.config_dirty = true;
                self.profile_error = None;
                if announce {
                    self.profile_notice =
                        Some(format!("已应用配置档「{name}」，下一次开始增强时生效。"));
                }
                self.persist_ui_flags();
                true
            }
            Err(err) => {
                self.profile_error = Some(err);
                false
            }
        }
    }

    /// Save the current settings as a named profile.
    fn create_profile_from_current(&mut self, overwrite: bool) {
        let name = self.profile_name.trim().to_owned();
        // Make the TOML match what the sliders show right now.
        self.quality.apply(&mut self.config);
        match profiles::save_profile(&self.profiles_dir, &name, &self.config, overwrite) {
            Ok(()) => {
                self.refresh_profiles();
                self.active_profile = Some(name.clone());
                self.config_dirty = true;
                self.profile_error = None;
                self.profile_notice = Some(if overwrite {
                    format!("已覆盖配置档「{name}」。")
                } else {
                    format!("已保存配置档「{name}」。")
                });
                self.profile_name.clear();
                self.persist_ui_flags();
            }
            Err(err) => self.profile_error = Some(err),
        }
    }

    /// Rename the profile picked with the 重命名 button.
    fn rename_named_profile(&mut self) {
        let Some(old) = self.profile_rename.clone() else {
            return;
        };
        let new = self.profile_name.trim().to_owned();
        match profiles::rename_profile(&self.profiles_dir, &old, &new) {
            Ok(()) => {
                self.refresh_profiles();
                if self.active_profile.as_deref() == Some(old.as_str()) {
                    self.active_profile = Some(new.clone());
                }
                // Rules that pointed at the old name follow the rename.
                for rule in self.rules.iter_mut() {
                    if rule.profile == old {
                        rule.profile = new.clone();
                    }
                }
                self.save_rules();
                self.profile_rename = None;
                self.profile_name.clear();
                self.profile_error = None;
                self.profile_notice = Some(format!("已把「{old}」改名为「{new}」。"));
                self.persist_ui_flags();
            }
            Err(err) => self.profile_error = Some(err),
        }
    }

    /// Delete a profile, together with the rules that used it.
    fn delete_named_profile(&mut self, name: &str) {
        match profiles::delete_profile(&self.profiles_dir, name) {
            Ok(()) => {
                self.refresh_profiles();
                if self.profile_rename.as_deref() == Some(name) {
                    self.profile_rename = None;
                    self.profile_name.clear();
                }
                let before = self.rules.len();
                self.rules.retain(|rule| rule.profile != name);
                if self.rules.len() != before {
                    self.save_rules();
                }
                if self.active_profile.as_deref() == Some(name) {
                    self.active_profile = None;
                    self.persist_ui_flags();
                }
                self.profile_error = None;
                self.profile_notice = Some(format!(
                    "已删除配置档「{name}」，当前设置保留为「自定义」。"
                ));
            }
            Err(err) => self.profile_error = Some(err),
        }
    }

    /// Persist the per-application rules; errors stay inside the dialog.
    fn save_rules(&mut self) {
        if let Err(err) = auto_apply::save_rules(&self.rules, &self.rules_path) {
            self.profile_error = Some(err);
        }
    }

    /// Process name of the selected window, if the list still knows it.
    fn window_process(&self, id: u64) -> Option<String> {
        window_source::find_by_id(&self.windows, id).map(|info| info.process_name.clone())
    }

    /// Silently apply the profile saved for the selected window's program.
    fn maybe_auto_apply(&mut self) {
        if !self.auto_apply || self.source != SourceChoice::Window {
            return;
        }
        let Some(id) = self.window_selected else {
            return;
        };
        let Some(process) = self.window_process(id) else {
            return;
        };
        let Some(rule) = auto_apply::rule_for_process(&self.rules, &process).cloned() else {
            return;
        };
        if self.auto_applied_for == Some(id) {
            return;
        }
        self.auto_applied_for = Some(id);
        let profile = rule.profile.clone();
        if self.apply_profile_inner(&profile, false) {
            self.auto_apply_hint =
                Some(format!("检测到 {process}：已自动套用配置档「{profile}」。"));
        } else {
            self.auto_apply_hint = Some(format!(
                "检测到 {process}：配置档「{profile}」不可用（可能已删除）。"
            ));
        }
    }

    /// One-line offer shown under the selected window when a rule matches.
    fn auto_apply_row(&mut self, ui: &mut egui::Ui, plan: &LayoutPlan) {
        let Some(id) = self.window_selected else {
            self.auto_apply_hint = None;
            return;
        };
        let Some(process) = self.window_process(id) else {
            return;
        };
        let rule = auto_apply::rule_for_process(&self.rules, &process).cloned();
        if let Some(rule) = rule {
            let applied = self.active_profile.as_deref() == Some(rule.profile.as_str());
            ui.horizontal_wrapped(|ui| {
                ui.label(format!(
                    "检测到 {}：保存过配置档「{}」",
                    rule.process, rule.profile
                ));
                if applied {
                    ui.label(RichText::new("已套用").color(Color32::from_rgb(120, 200, 140)));
                } else if dialog_button(ui, plan.touch, "一键套用") {
                    self.apply_profile(&rule.profile);
                }
            });
        }
        if let Some(message) = self.auto_apply_hint.clone() {
            ui.label(
                RichText::new(message)
                    .small()
                    .color(Color32::from_rgb(120, 200, 140)),
            );
        }
    }

    /// 配置档 entry point in the 实时增强 settings column.
    fn profiles_section(&mut self, ui: &mut egui::Ui, plan: &LayoutPlan) {
        ui.heading("配置档");
        hint(
            ui,
            "把一个画质组合存成配置档（例如给某个游戏单独一份），以后一键切换。",
        );
        let label = self.active_profile.as_deref().unwrap_or("自定义");
        let text = format!("配置档：{label}…");
        let button = if plan.touch {
            egui::Button::new(text).min_size(egui::vec2(ui.available_width(), TOUCH_MIN))
        } else {
            egui::Button::new(text)
        };
        if ui.add(button).clicked() {
            self.profiles_open = true;
            self.profile_error = None;
            self.refresh_profiles();
            if self.profile_name.trim().is_empty() {
                self.profile_name = profiles::unique_name(
                    self.active_profile.as_deref().unwrap_or("新配置档"),
                    &self.profile_names,
                );
            }
            if self.rule_process.trim().is_empty()
                && let Some(id) = self.window_selected
                && let Some(process) = self.window_process(id)
            {
                self.rule_process = process;
            }
        }
        hint(
            ui,
            "“启动时应用上次的配置”和“按窗口自动切换档位”在配置档窗口里设置。",
        );
    }

    /// The 配置档 window (profiles + per-program rules). Centered and scrollable
    /// so it also fits a 360x740 phone window.
    fn profiles_dialog(&mut self, ctx: &egui::Context, plan: &LayoutPlan) {
        if !self.profiles_open {
            return;
        }
        let screen = ctx.content_rect();
        let width = if plan.touch {
            (screen.width() - 24.0).clamp(240.0, 460.0)
        } else {
            (screen.width() * 0.5).clamp(360.0, 560.0)
        };
        let height = (screen.height() - 90.0).max(200.0);
        let mut open = true;
        egui::Window::new("配置档")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(width)
            .default_height(height)
            .max_width(width)
            .max_height(height)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.set_width(width - 24.0);
                egui::ScrollArea::vertical()
                    .id_salt("profiles_scroll")
                    .max_height(height - 60.0)
                    .show(ui, |ui| {
                        self.profiles_body(ui, plan);
                    });
            });
        if !open {
            self.profiles_open = false;
        }
    }

    /// Contents of the 配置档 window.
    fn profiles_body(&mut self, ui: &mut egui::Ui, plan: &LayoutPlan) {
        let touch = plan.touch;

        ui.label(RichText::new("已保存的配置档").strong());
        if self.profile_names.is_empty() {
            hint(
                ui,
                "还没有配置档。取个名字，点下面的“新建（从当前设置复制）”即可保存当前画质。",
            );
        } else {
            let mut apply: Option<String> = None;
            let mut rename: Option<String> = None;
            let mut delete: Option<String> = None;
            for name in &self.profile_names {
                let active = self.active_profile.as_deref() == Some(name.as_str());
                ui.horizontal_wrapped(|ui| {
                    let mut text = RichText::new(name);
                    if active {
                        text = text.strong().color(Color32::from_rgb(120, 200, 255));
                    }
                    ui.label(text);
                    if active {
                        ui.label(RichText::new("（当前）").weak());
                    } else if dialog_button(ui, touch, "应用") {
                        apply = Some(name.clone());
                    }
                    if dialog_button(ui, touch, "重命名") {
                        rename = Some(name.clone());
                    }
                    if dialog_button(ui, touch, "删除") {
                        delete = Some(name.clone());
                    }
                });
            }
            if let Some(name) = apply {
                self.apply_profile(&name);
            }
            if let Some(name) = rename {
                self.profile_name = name.clone();
                self.profile_rename = Some(name);
                self.profile_error = None;
                self.profile_notice = None;
            }
            if let Some(name) = delete {
                self.delete_named_profile(&name);
            }
        }

        ui.add_space(6.0);
        ui.separator();
        let renaming = self.profile_rename.clone();
        let exists = self
            .profile_names
            .iter()
            .any(|existing| existing.eq_ignore_ascii_case(self.profile_name.trim()));
        let action_label = match (&renaming, exists) {
            (Some(_), _) => "确认重命名",
            (None, true) => "覆盖这个配置档",
            (None, false) => "新建（从当前设置复制）",
        };
        ui.label(
            RichText::new(if renaming.is_some() {
                "新的名字"
            } else {
                "配置档名字"
            })
            .strong(),
        );
        ui.add(
            egui::TextEdit::singleline(&mut self.profile_name)
                .hint_text("例如 某游戏的流畅档")
                .desired_width(ui.available_width()),
        );
        ui.horizontal_wrapped(|ui| {
            if dialog_button(ui, touch, action_label) {
                if renaming.is_some() {
                    self.rename_named_profile();
                } else {
                    self.create_profile_from_current(exists);
                }
            }
            if renaming.is_some() && dialog_button(ui, touch, "取消重命名") {
                self.profile_rename = None;
                self.profile_name.clear();
                self.profile_error = None;
            }
        });
        if let Some(err) = &self.profile_error {
            ui.colored_label(Color32::from_rgb(235, 150, 120), err);
        }
        if let Some(notice) = &self.profile_notice {
            ui.colored_label(Color32::from_rgb(130, 205, 150), notice);
        }

        ui.add_space(6.0);
        ui.separator();
        if ui
            .checkbox(&mut self.apply_last_profile, "启动时应用上次的配置")
            .changed()
        {
            self.ui_flags_dirty = true;
        }
        hint(ui, "重新打开 Fluid 时自动恢复上次用的配置档。");
        if ui
            .checkbox(&mut self.auto_apply, "按窗口自动切换档位")
            .changed()
        {
            self.ui_flags_dirty = true;
            self.auto_apply_hint = None;
        }
        hint(
            ui,
            "选中保存过的程序窗口时自动换到对应配置档，并在画面上方提示一句。",
        );
        ui.label(
            RichText::new(format!("规则文件：{}", self.rules_path.display()))
                .small()
                .weak(),
        );

        ui.add_space(6.0);
        ui.separator();
        ui.label(RichText::new("按程序自动套用").strong());
        if self.rules.is_empty() {
            hint(
                ui,
                "还没有规则。添加后，选择这个程序的窗口时会自动套用对应配置档。",
            );
        } else {
            let mut remove: Option<String> = None;
            for rule in &self.rules {
                ui.horizontal_wrapped(|ui| {
                    ui.label(format!("{} → {}", rule.process, rule.profile));
                    if dialog_button(ui, touch, "删除规则") {
                        remove = Some(rule.process.clone());
                    }
                });
            }
            if let Some(process) = remove {
                auto_apply::remove_rule(&mut self.rules, &process);
                self.save_rules();
            }
        }
        ui.add_space(4.0);
        ui.label(
            RichText::new("程序名（例如 genshinimpact.exe）")
                .small()
                .weak(),
        );
        ui.add(
            egui::TextEdit::singleline(&mut self.rule_process)
                .hint_text("game.exe")
                .desired_width(ui.available_width()),
        );
        let mut add_pressed = false;
        ui.horizontal_wrapped(|ui| {
            egui::ComboBox::from_id_salt("rule_profile_combo")
                .selected_text(
                    self.rule_profile
                        .clone()
                        .unwrap_or_else(|| "选择配置档".to_owned()),
                )
                .show_ui(ui, |ui| {
                    for name in &self.profile_names {
                        ui.selectable_value(&mut self.rule_profile, Some(name.clone()), name);
                    }
                });
            add_pressed = dialog_button(ui, touch, "添加 / 更新");
        });
        if add_pressed {
            let process = self.rule_process.trim().to_owned();
            let profile = self.rule_profile.clone().unwrap_or_default();
            if process.is_empty() || profile.is_empty() {
                self.profile_error = Some("先填程序名，并选择要套用的配置档。".to_owned());
            } else if !self.profile_names.iter().any(|name| name == &profile) {
                self.profile_error = Some(format!("配置档「{profile}」不存在。"));
            } else {
                auto_apply::upsert_rule(&mut self.rules, &process, &profile);
                self.rule_process.clear();
                self.save_rules();
                self.profile_error = None;
                self.profile_notice = Some(format!(
                    "已保存：{} 使用配置档「{profile}」。",
                    auto_apply::normalize_process(&process)
                ));
            }
        }
    }

    /// Dismissible three-step card for people who just installed the app.
    fn first_run_card(&mut self, ui: &mut egui::Ui, plan: &LayoutPlan) {
        if !guide::should_show(self.guide_dismissed, self.smoke_seconds.is_some()) {
            return;
        }
        let style = ui.style().clone();
        let mut dismiss = false;
        egui::Frame::group(&style)
            .fill(Color32::from_rgb(24, 38, 54))
            .show(ui, |ui| {
                ui.label(
                    RichText::new("三步开始")
                        .strong()
                        .size(16.0)
                        .color(Color32::from_rgb(150, 200, 255)),
                );
                for step in guide::STEPS.iter() {
                    ui.label(RichText::new(step.title).strong());
                    hint(ui, step.text);
                }
                ui.horizontal_wrapped(|ui| {
                    if dialog_button(ui, plan.touch, "知道了") {
                        dismiss = true;
                    }
                    if ui
                        .checkbox(&mut self.guide_dismissed, "别再提示我")
                        .changed()
                    {
                        self.ui_flags_dirty = true;
                    }
                });
            });
        if dismiss {
            self.guide_dismissed = true;
            self.ui_flags_dirty = true;
        }
        ui.add_space(6.0);
    }

    // ------------------------------------------------------------ windows

    /// Fetch the window list the first time 窗口捕获 is shown (or after a refresh).
    fn windows_tick(&mut self) {
        if self.source == SourceChoice::Window && !self.windows_loaded {
            self.refresh_windows();
        }
    }

    /// Re-enumerate top-level windows; errors stay inline instead of blocking.
    fn refresh_windows(&mut self) {
        match fluid_capture::enumerate_windows() {
            Ok(windows) => {
                self.windows = windows;
                self.windows_error = None;
            }
            Err(err) => {
                self.windows.clear();
                self.windows_error = Some(format!("无法列出窗口：{err}"));
            }
        }
        self.windows_loaded = true;
        // A refresh can reveal a window whose program has a saved profile.
        if self.source == SourceChoice::Window {
            self.maybe_auto_apply();
        }
    }

    /// While a window capture is running, check once a second that the captured
    /// window still exists (QA defect D3). A closed or minimized window leaves
    /// the preview frozen on the last frame, so the UI has to say why.
    fn window_watch_tick(&mut self) {
        if self.source != SourceChoice::Window || self.run.is_none() {
            self.window_lost = false;
            self.window_watch_at = None;
            return;
        }
        let now = Instant::now();
        if let Some(last) = self.window_watch_at
            && now.duration_since(last) < Duration::from_millis(1000)
        {
            return;
        }
        self.window_watch_at = Some(now);
        if let Ok(windows) = fluid_capture::enumerate_windows() {
            self.window_lost =
                window_source::loss_notice(true, self.window_selected, &windows).is_some();
            // Refresh the list too, so the 已选 row says "窗口已关闭或不可见".
            self.windows = windows;
            self.windows_loaded = true;
            self.windows_error = None;
        }
    }

    fn select_window(&mut self, id: u64) {
        self.window_selected = Some(id);
        self.window_error = None;
        self.window_panel_open = false;
        self.window_lost = false;
        self.window_watch_at = None;
        self.config.capture.window = Some(window_source::target_for(id));
        self.config_dirty = true;
        self.auto_apply_hint = None;
        // Picking a window is the moment the per-program profile should kick in.
        self.maybe_auto_apply();
    }

    fn smoke_start(&mut self) {
        if self.smoke_seconds.is_none() || self.smoke_started {
            return;
        }
        self.smoke_started = true;
        if !self.forced_source {
            self.source = SourceChoice::Synthetic;
        }
        // Smoke runs a small frame so the whole path stays cheap and deterministic.
        self.synthetic_dims = Dims::new(640, 360);
        // Keep the on-screen controls in sync with the forced smoke pipeline (2x, native size).
        self.quality.interp = InterpChoice::X2;
        self.quality.scale = ScaleChoice::Off;
        self.quality.filter = FilterChoice::Adaptive;
        self.start_run(true);
    }

    fn smoke_lifecycle(&mut self, ctx: &egui::Context) {
        let Some(seconds) = self.smoke_seconds else {
            return;
        };
        // Start the clock on the first presented frame, so --smoke 4 really
        // exercises four seconds of the pipeline even on a cold GPU init.
        if self.smoke_deadline.is_none() {
            let produced = self.frames > 0 || self.last_report.is_some();
            if produced {
                self.smoke_deadline =
                    Some(Instant::now() + Duration::from_secs_f32(seconds.max(0.5)));
            } else if self.run.is_none()
                || self
                    .smoke_hard_deadline
                    .map(|hard| Instant::now() >= hard)
                    .unwrap_or(false)
            {
                // Nothing was ever produced: stop now instead of hanging.
                self.smoke_deadline = Some(Instant::now());
            } else {
                ctx.request_repaint_after(Duration::from_millis(16));
                return;
            }
        }
        let Some(deadline) = self.smoke_deadline else {
            return;
        };
        let now = Instant::now();
        if !self.smoke_stop_sent && now >= deadline {
            self.smoke_stop_sent = true;
            self.stop_run();
        }
        let grace_over = now >= deadline + Duration::from_millis(1500);
        if self.smoke_stop_sent && (self.run.is_none() || grace_over) {
            self.finish_smoke(ctx);
        }
    }

    fn finish_smoke(&mut self, ctx: &egui::Context) {
        if self.smoke_done {
            return;
        }
        self.smoke_done = true;
        // Give the worker one last chance to publish its report.
        if let Some(mut run) = self.run.take() {
            if run.is_finished() {
                run.join();
            }
            if self.last_report.is_none() {
                self.last_report = run.take_report();
            }
        }
        self.snapshot_now();
        self.smoke_deadline = None;
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
    }

    fn snapshot_now(&mut self) {
        // Prefer the pipeline report: it counts every frame the sink presented,
        // including the ones the UI did not get to pull, and its fps is the
        // measured output rate rather than the polled meter.
        let (frames, fps) = match &self.last_report {
            Some(report) => (report.stats.frames_out, report.stats.output_fps_value()),
            None => (self.frames, self.fps.fps()),
        };
        let snapshot = SmokeSnapshot {
            frames,
            fps,
            report: self.last_report.clone(),
        };
        if let Ok(mut slot) = self.snapshot.lock() {
            *slot = Some(snapshot);
        }
    }

    /// Write the json summary after an interactive run too (QA defect D5).
    ///
    /// The --smoke epilogue in lib.rs is unchanged, but a normal run must also
    /// leave {"ok":true,"frames":N,"fps":F,"report":{...}} behind when the
    /// user stops enhancement or closes the window.
    fn save_json_summary(&mut self) {
        let Some(path) = self.json_summary.clone() else {
            return;
        };
        self.snapshot_now();
        let snapshot = self.snapshot.lock().ok().and_then(|slot| slot.clone());
        if let Err(err) = crate::write_json_summary(&path, &snapshot) {
            self.warning = Some(format!("运行摘要保存失败：{err}"));
        }
    }

    // ------------------------------------------------------------- live run

    fn selected_monitor_index(&self) -> usize {
        self.monitors
            .get(self.monitor_idx)
            .map(|monitor| monitor.index)
            .unwrap_or(self.monitor_idx)
    }

    fn start_run(&mut self, smoke: bool) {
        if self.run.is_some() {
            return;
        }
        self.error_message = None;
        self.last_report = None;
        self.frames = 0;
        self.last_generation = 0;
        self.last_frame_at = None;
        self.fps.reset();
        self.stopping_since = None;
        self.forced_stop_note = false;
        self.window_lost = false;
        self.window_watch_at = None;

        let mut config = self.config.clone();
        if smoke {
            config.interpolator.mode = fluid_core::InterpMode::Fixed;
            config.interpolator.multiplier = 2;
            config.upscaler.scale = ScaleConfig::default();
            config.upscaler.filter = fluid_core::UpscaleFilter::Adaptive;
        } else {
            self.quality.apply(&mut config);
        }
        config.capture.monitor = self.selected_monitor_index();
        // Only 窗口捕获 sets a window target; the other sources must clear it or
        // fluid_capture would quietly open a window instead of the screen.
        config.capture.window = None;

        if self.source == SourceChoice::Window {
            if cfg!(target_os = "android") {
                self.error_message = Some(
                    "手机版不能单独捕获一个窗口：Android 只提供整屏录制（MediaProjection）权限。请在电脑上使用窗口捕获。"
                        .to_owned(),
                );
                return;
            }
            // Re-enumerate so a window that closed since the last refresh gives a
            // friendly inline message instead of a raw pipeline error.
            match fluid_capture::enumerate_windows() {
                Ok(windows) => {
                    self.windows = windows;
                    self.windows_loaded = true;
                    self.windows_error = None;
                }
                Err(err) => {
                    self.window_error = Some(format!("无法列出窗口：{err}"));
                    self.window_panel_open = true;
                    return;
                }
            }
            match self.window_selected {
                Some(id) if window_source::contains_id(&self.windows, id) => {
                    config.capture.window = Some(window_source::target_for(id));
                    self.window_error = None;
                }
                Some(_) => {
                    self.window_error = Some(window_source::MISSING_WINDOW.to_owned());
                    self.window_panel_open = true;
                    return;
                }
                None => {
                    self.window_error = Some(window_source::NO_WINDOW.to_owned());
                    self.window_panel_open = true;
                    return;
                }
            }
        }

        let source: Option<Box<dyn FrameSource>> = match self.source {
            SourceChoice::Synthetic => {
                let dims = if smoke {
                    Dims::new(640, 360)
                } else {
                    SYNTHETIC_DIMS
                };
                Some(Box::new(fluid_video::SyntheticSource::new(dims, 60.0)))
            }
            SourceChoice::Screen => match fluid_capture::open(&config.capture) {
                Ok(source) => Some(source),
                Err(err) => {
                    self.error_message = Some(format!(
                        "屏幕捕获启动失败：{err}。可以先切换到“内置测试画面”体验。"
                    ));
                    None
                }
            },
            SourceChoice::Window => match fluid_capture::open(&config.capture) {
                Ok(source) => Some(source),
                Err(err) => {
                    self.error_message = Some(format!(
                        "窗口捕获启动失败：{err}。窗口可能已经关闭；点“刷新列表”重新选择，或改用内置测试画面。"
                    ));
                    self.window_panel_open = true;
                    None
                }
            },
        };
        let Some(source) = source else {
            return;
        };

        self.run_multiplier = if config.interpolator.mode == fluid_core::InterpMode::Off {
            1
        } else {
            u32::from(config.interpolator.multiplier)
        };

        if !smoke {
            self.config = config.clone();
            if let Err(err) = config_io::save(&self.config, &self.config_path) {
                self.warning = Some(format!("配置保存失败：{err}"));
            }
            self.saved_quality = self.quality;
        }

        // QA defect D4: remember the backends this run really started with, so
        // editing the dropdowns mid-run cannot relabel the running pipeline.
        self.run_backend_label = BackendChoice::from_backend(config.upscaler.backend)
            .label()
            .to_owned();
        self.run_interp_label = InterpBackendChoice::from_backend(config.interpolator.backend)
            .label()
            .to_owned();

        self.run_started = Instant::now();
        self.run = Some(RunHandle::spawn(source, config));
    }

    fn stop_run(&mut self) {
        let live = self.fps.fps();
        if live > 0.05 {
            self.last_out_fps = Some(live);
        }
        if let Some(run) = &self.run {
            run.stop();
            self.stopping_since = Some(Instant::now());
        }
    }

    fn force_stop(&mut self) {
        if let Some(run) = self.run.take() {
            run.stop();
        }
        self.stopping_since = None;
        self.forced_stop_note = true;
    }

    fn update_texture(&mut self, ctx: &egui::Context, frame: &fluid_core::Frame) {
        let rgba = if frame.fmt == PixelFormat::Rgba8 {
            frame.clone()
        } else {
            frame.convert(PixelFormat::Rgba8)
        };
        let size = [rgba.dims.width as usize, rgba.dims.height as usize];
        let image = egui::ColorImage::from_rgba_unmultiplied(size, &rgba.data);
        let dims = (rgba.dims.width, rgba.dims.height);
        match &mut self.texture {
            Some(texture) if self.texture_dims == dims => {
                texture.set(image, egui::TextureOptions::LINEAR);
            }
            _ => {
                self.texture =
                    Some(ctx.load_texture("fluid-preview", image, egui::TextureOptions::LINEAR));
            }
        }
        self.texture_dims = dims;
    }

    // ---------------------------------------------------------- video start

    /// Read resolution / fps / frame count of the chosen input file so the
    /// 视频处理 tab can show it before the job starts.
    fn refresh_video_info(&mut self) {
        self.video_input_info = None;
        self.video_input_error = None;
        let text = self.video_input.trim().to_owned();
        if text.is_empty() {
            return;
        }
        let format = match video_format::detect_format(&text) {
            Ok(format) => format,
            Err(err) => {
                self.video_input_error = Some(err);
                return;
            }
        };
        let path = PathBuf::from(&text);
        if !path.exists() {
            self.video_input_error = Some(format!("找不到输入文件：{}", path.display()));
            return;
        }
        match video_job::inspect_input(&path, format) {
            Ok(info) => self.video_input_info = Some(info),
            Err(err) => self.video_input_error = Some(err),
        }
    }

    /// Native "choose the video to process" dialog.
    fn pick_video_input(&mut self) {
        let Some(path) = file_picker::pick_input_video() else {
            return;
        };
        self.video_input = path.to_string_lossy().into_owned();
        if self.video_output.trim().is_empty() {
            self.video_output = video_format::suggest_output(&self.video_input);
        }
        self.video_error = None;
        self.refresh_video_info();
    }

    /// Native "save the result as" dialog; the extension decides the container.
    fn pick_video_output(&mut self) {
        let suggested = if self.video_output.trim().is_empty() {
            video_format::suggest_output(&self.video_input)
        } else {
            self.video_output.clone()
        };
        let Some(path) = file_picker::pick_output_video(&suggested) else {
            return;
        };
        let path = file_picker::with_default_extension(path, VideoFormat::Y4m);
        self.video_output = path.to_string_lossy().into_owned();
        self.video_error = None;
    }

    /// One path row with an optional native picker button.
    fn video_path_row(
        &mut self,
        ui: &mut egui::Ui,
        plan: &LayoutPlan,
        label: &str,
        hint_text: &str,
        input: bool,
        dialogs: bool,
    ) {
        ui.label(RichText::new(label).strong());
        let button = if input {
            "选择视频…"
        } else {
            "另存为…"
        };
        let field = if input {
            &mut self.video_input
        } else {
            &mut self.video_output
        };
        let mut pick = false;
        let mut changed = false;
        if plan.touch {
            changed = ui
                .add(
                    egui::TextEdit::singleline(field)
                        .hint_text(hint_text)
                        .desired_width(ui.available_width()),
                )
                .changed();
            if dialogs {
                pick = dialog_button(ui, true, button);
            }
        } else {
            let width = (ui.available_width() - 170.0).max(120.0);
            ui.horizontal(|ui| {
                changed = ui
                    .add(
                        egui::TextEdit::singleline(field)
                            .hint_text(hint_text)
                            .desired_width(width),
                    )
                    .changed();
                if dialogs {
                    pick = dialog_button(ui, false, button);
                }
            });
        }
        let _ = plan;
        if pick {
            if input {
                self.pick_video_input();
            } else {
                self.pick_video_output();
            }
        } else if input && changed {
            self.refresh_video_info();
        }
    }

    fn start_video(&mut self) {
        self.video_error = None;
        self.video_reveal_error = None;
        let input_text = self.video_input.trim().to_owned();
        let input_format = match video_format::detect_format(&input_text) {
            Ok(format) => format,
            Err(err) => {
                self.video_input_error = Some(err);
                return;
            }
        };
        let input_path = PathBuf::from(&input_text);
        if !input_path.exists() {
            self.video_input_error = Some(format!("找不到输入文件：{}", input_path.display()));
            return;
        }
        let info = match video_job::inspect_input(&input_path, input_format) {
            Ok(info) => info,
            Err(err) => {
                self.video_input_error = Some(err);
                return;
            }
        };
        self.video_input_info = Some(info.clone());
        self.video_input_error = None;

        let output_text = if self.video_output.trim().is_empty() {
            video_format::suggest_output(&input_text)
        } else {
            self.video_output.trim().to_owned()
        };
        let output_format = match video_format::detect_format(&output_text) {
            Ok(format) => format,
            Err(err) => {
                self.video_error = Some(err);
                return;
            }
        };
        self.video_output = output_text.clone();

        let mut config = self.config.clone();
        self.quality.apply(&mut config);
        self.config = config.clone();
        self.saved_quality = self.quality;
        let _ = config_io::save(&self.config, &self.config_path);

        let quality = if video_format::quality_applies(output_format) {
            video_format::clamp_quality(self.video_quality)
        } else {
            video_format::QUALITY_DEFAULT
        };

        self.video_output_path = Some(PathBuf::from(&output_text));
        self.video_state = VideoState::default();
        self.video_job = Some(VideoJob::spawn(
            PathBuf::from(input_text),
            PathBuf::from(output_text),
            output_format,
            info,
            config,
            quality,
        ));
    }

    // ------------------------------------------------------------ shortcuts

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        // Ctrl+1..4 / Alt+1..4 switch tabs. This runs before the text-field
        // check and consumes the key, so a focused widget (or a highlighted
        // tab) cannot swallow the shortcut.
        let tab_shortcut = ctx.input_mut(|input| {
            let mut found = None;
            for modifiers in [egui::Modifiers::CTRL, egui::Modifiers::ALT] {
                for (key, tab) in TAB_SHORTCUTS {
                    if input.consume_key(modifiers, key) {
                        found = Some(tab);
                    }
                }
            }
            found
        });
        if let Some(tab) = tab_shortcut {
            self.tab = tab;
        }

        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let pressed = |key: egui::Key| ctx.input(|input| input.key_pressed(key));

        if pressed(egui::Key::Space) && self.tab == Tab::Live {
            if self.run.is_some() {
                self.stop_run();
            } else {
                self.start_run(false);
            }
        }
        if pressed(egui::Key::Escape) {
            self.stop_run();
            if let Some(job) = &self.video_job {
                job.stop();
            }
        }
    }

    // ---------------------------------------------------------------- layout

    fn top_bar(&mut self, ui: &mut egui::Ui, plan: &LayoutPlan) {
        if plan.tight {
            // Very short / very narrow window (QA defect D2): a wrapped tab
            // row would eat the whole page height, so collapse the navigation
            // into a single picker row that is always fully visible.
            let width = ui.available_width().max(120.0);
            let mut chosen = self.tab;
            egui::ComboBox::from_id_salt("fluid_tab_combo")
                .width(width)
                .selected_text(RichText::new(chosen.label()).size(16.0))
                .show_ui(ui, |ui| {
                    for tab in Tab::ALL {
                        ui.selectable_value(
                            &mut chosen,
                            tab,
                            RichText::new(tab.label()).size(16.0),
                        );
                    }
                });
            self.tab = chosen;
            return;
        }
        if plan.tabs_scroll {
            // Phone / very narrow window: the title gets its own line and the
            // tabs wrap (QA defect D2), so no tab is ever clipped or hidden
            // behind a scrollbar the user cannot see.
            ui.horizontal(|ui| {
                ui.heading(RichText::new("Fluid").size(22.0));
                ui.label(RichText::new("帧生成与画质增强").weak());
            });
            ui.add_space(2.0);
            let narrow = ui.available_width() < 380.0;
            let tab_size = if narrow { 15.0 } else { 17.0 };
            ui.horizontal_wrapped(|ui| {
                for tab in Tab::ALL {
                    if ui
                        .selectable_label(
                            self.tab == tab,
                            RichText::new(tab.label()).size(tab_size),
                        )
                        .clicked()
                    {
                        self.tab = tab;
                    }
                }
            });
        } else {
            ui.horizontal(|ui| {
                ui.heading(RichText::new("Fluid").size(24.0));
                ui.label(RichText::new("帧生成与画质增强").weak());
                ui.add_space(20.0);
                for tab in Tab::ALL {
                    if ui
                        .selectable_label(self.tab == tab, RichText::new(tab.label()).size(16.0))
                        .clicked()
                    {
                        self.tab = tab;
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new(format!("{} 线程", self.threads)).weak());
                });
            });
        }
    }

    /// Fixed bottom action bar for phones: the primary action always stays
    /// under the thumb instead of scrolling away with the settings.
    fn bottom_bar(&mut self, ui: &mut egui::Ui, plan: &LayoutPlan) {
        let running = self.run.is_some();
        let height = plan.primary_button_height();
        let total = ui.available_width().max(160.0);
        let (label, fill) = if running {
            ("停止增强", Color32::from_rgb(150, 62, 62))
        } else {
            ("开始增强", Color32::from_rgb(46, 120, 86))
        };
        let fullscreen_label = if self.fullscreen_output {
            "全屏输出：开"
        } else {
            "全屏输出：关"
        };
        if total < 360.0 {
            // QA defect D2: in a 240-180 point wide window two buttons side by
            // side get clipped, so stack them and let each use the full width.
            ui.vertical(|ui| {
                let button = egui::Button::new(RichText::new(label).size(20.0).strong())
                    .min_size(egui::vec2(total, height))
                    .fill(fill);
                if ui.add(button).clicked() {
                    if running {
                        self.stop_run();
                    } else {
                        self.start_run(false);
                    }
                }
                let toggle = egui::Button::new(RichText::new(fullscreen_label).size(15.0))
                    .min_size(egui::vec2(total, TOUCH_MIN));
                if ui.add(toggle).clicked() {
                    self.fullscreen_output = !self.fullscreen_output;
                }
            });
        } else {
            let toggle_width = 132.0_f32.min(total * 0.38);
            let main_width = (total - toggle_width - 10.0).max(120.0);
            ui.horizontal(|ui| {
                let button = egui::Button::new(RichText::new(label).size(22.0).strong())
                    .min_size(egui::vec2(main_width, height))
                    .fill(fill);
                if ui.add(button).clicked() {
                    if running {
                        self.stop_run();
                    } else {
                        self.start_run(false);
                    }
                }
                let toggle = egui::Button::new(RichText::new(fullscreen_label).size(15.0))
                    .min_size(egui::vec2(toggle_width, height));
                if ui.add(toggle).clicked() {
                    self.fullscreen_output = !self.fullscreen_output;
                }
            });
        }
        if !plan.tight {
            hint(ui, "触屏：点大按钮即可开始 / 停止；电脑上也可以按空格键。");
        }
    }

    fn banners(&mut self, ui: &mut egui::Ui) {
        if let Some(message) = self.error_message.clone() {
            ui.horizontal(|ui| {
                ui.colored_label(Color32::from_rgb(235, 120, 120), format!("错误：{message}"));
                if ui.small_button("知道了").clicked() {
                    self.error_message = None;
                }
            });
        }
        if let Some(message) = self.warning.clone() {
            ui.horizontal(|ui| {
                ui.colored_label(Color32::from_rgb(230, 180, 90), format!("提示：{message}"));
                if ui.small_button("知道了").clicked() {
                    self.warning = None;
                }
            });
        }
        if self.forced_stop_note {
            ui.horizontal(|ui| {
                ui.colored_label(
                    Color32::from_rgb(230, 180, 90),
                    "上次运行没有及时退出，已放弃等待。可以重新开始。",
                );
                if ui.small_button("知道了").clicked() {
                    self.forced_stop_note = false;
                }
            });
        }
    }

    fn live_tab(&mut self, ui: &mut egui::Ui, plan: &LayoutPlan) {
        if plan.touch {
            // Phone layout: the whole page scrolls, so a very short window can
            // never clip the header, the source row or the preview, and the
            // fixed bottom bar keeps the main action reachable.
            egui::ScrollArea::vertical()
                .id_salt("live_scroll_compact")
                .show(ui, |ui| {
                    self.first_run_card(ui, plan);
                    self.live_header(ui, plan);
                    let height = plan.preview_height;
                    self.preview(ui, height);
                    ui.add_space(6.0);
                    self.control_panel(ui, plan);
                });
            return;
        }
        self.first_run_card(ui, plan);
        self.live_header(ui, plan);

        if plan.side_by_side {
            let available = ui.available_size();
            let control_width = plan.control_width.min((available.x * 0.45).max(260.0));
            let preview_width = (available.x - control_width - 16.0).max(240.0);
            let preview_height = (available.y - 8.0).max(200.0);
            ui.horizontal(|ui| {
                ui.allocate_ui_with_layout(
                    egui::vec2(preview_width, preview_height),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        self.preview(ui, preview_height);
                    },
                );
                ui.add_space(8.0);
                ui.vertical(|ui| {
                    ui.set_width(control_width);
                    egui::ScrollArea::vertical()
                        .id_salt("live_controls")
                        .show(ui, |ui| {
                            self.control_panel(ui, plan);
                        });
                });
            });
        } else {
            // Phone / narrow window: preview first, settings below, and only
            // the settings area scrolls so the action bar stays fixed.
            let width = ui.available_width().max(200.0);
            let height = plan
                .preview_height
                .min((ui.available_height() - 90.0).max(120.0));
            ui.allocate_ui_with_layout(
                egui::vec2(width, height),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    self.preview(ui, height);
                },
            );
            ui.add_space(6.0);
            egui::ScrollArea::vertical()
                .id_salt("live_controls_stacked")
                .show(ui, |ui| {
                    self.control_panel(ui, plan);
                });
        }
    }

    fn live_header(&mut self, ui: &mut egui::Ui, plan: &LayoutPlan) {
        ui.heading("实时增强");
        hint(
            ui,
            if plan.bottom_bar {
                "选好来源后点屏幕底部的大按钮开始。没有游戏也没关系，内置测试画面可以立刻看到插帧和放大效果。"
            } else {
                "选择画面来源后点击右侧的“开始增强”。没有游戏也没关系，内置测试画面可以立刻看到插帧和放大效果。"
            },
        );
        ui.add_space(4.0);
        self.source_row(ui, plan);
        if self.window_lost {
            self.window_lost_notice(ui);
        }
        ui.add_space(4.0);
        self.status_strip(ui, plan);
        ui.add_space(6.0);
    }

    /// Amber notice shown while a window capture lost its window (QA defect D3).
    fn window_lost_notice(&mut self, ui: &mut egui::Ui) {
        let style = ui.style().clone();
        egui::Frame::group(&style)
            .fill(Color32::from_rgb(52, 42, 24))
            .show(ui, |ui| {
                ui.label(
                    RichText::new("被捕获的窗口已经关闭或最小化")
                        .strong()
                        .color(Color32::from_rgb(235, 190, 110)),
                );
                hint(
                    ui,
                    "Fluid 会继续重复最后一帧画面，看起来就像卡住了。停止增强后点“刷新列表”重新选一个窗口即可恢复。",
                );
                ui.horizontal_wrapped(|ui| {
                    if ui.button("停止增强").clicked() {
                        self.stop_run();
                    }
                    if ui.button("重新选择窗口").clicked() {
                        self.refresh_windows();
                        self.window_panel_open = true;
                    }
                });
            });
        ui.add_space(4.0);
    }

    fn source_row(&mut self, ui: &mut egui::Ui, plan: &LayoutPlan) {
        let style = ui.style().clone();
        egui::Frame::group(&style).show(ui, |ui| {
            if plan.touch {
                // Touch layout: label on its own line, then full-width buttons
                // and a full-width picker (no cramped horizontal row).
                ui.label(RichText::new("画面来源").strong());
                ui.horizontal_wrapped(|ui| self.source_selector(ui));
                match self.source {
                    SourceChoice::Screen => self.monitor_combo(ui),
                    SourceChoice::Synthetic => {
                        ui.label(RichText::new(status::synthetic_label(self.synthetic_dims)).weak());
                    }
                    SourceChoice::Window => {}
                }
            } else {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("画面来源").strong());
                    ui.add_space(4.0);
                    self.source_selector(ui);
                    ui.add_space(12.0);
                    match self.source {
                        SourceChoice::Screen => self.monitor_combo(ui),
                        SourceChoice::Synthetic => {
                            ui.label(
                                RichText::new(status::synthetic_label(self.synthetic_dims)).weak(),
                            );
                        }
                        SourceChoice::Window => {}
                    }
                });
            }
            if self.source == SourceChoice::Window {
                self.window_summary(ui, plan);
                self.auto_apply_row(ui, plan);
                if self.window_panel_open {
                    self.window_source_panel(ui, plan);
                }
            }
            if let Some(message) = &self.monitor_error {
                ui.colored_label(Color32::from_rgb(230, 180, 90), message);
            }
            match self.source {
                SourceChoice::Screen => hint(
                    ui,
                    "屏幕捕获通过 DXGI 桌面复制读取整个屏幕，只在 Fluid 自己的窗口里显示，不会进入游戏进程。",
                ),
                SourceChoice::Window => hint(
                    ui,
                    "窗口捕获只读取选中的那一个窗口（优先用 Windows Graphics Capture，失败时退回 DXGI 裁剪），其他窗口不会出现在画面里。",
                ),
                SourceChoice::Synthetic => hint(
                    ui,
                    "内置测试画面是程序自己生成的一段移动图案，适合确认效果和测试性能。",
                ),
            }
        });
    }

    /// The 屏幕捕获 / 窗口捕获 / 内置测试画面 switch (窗口捕获 is hidden on Android).
    fn source_selector(&mut self, ui: &mut egui::Ui) {
        let before = self.source;
        for choice in SourceChoice::ALL {
            // Phones only have whole-screen MediaProjection capture.
            if cfg!(target_os = "android") && choice == SourceChoice::Window {
                continue;
            }
            ui.selectable_value(&mut self.source, choice, choice.label());
        }
        if self.source != before {
            self.window_error = None;
            self.ui_flags_dirty = true;
            if self.source == SourceChoice::Window {
                self.window_panel_open = self.window_selected.is_none();
                self.refresh_windows();
            }
        }
    }

    /// Selected-window line plus the "选择窗口" toggle for the list below it.
    fn window_summary(&mut self, ui: &mut egui::Ui, plan: &LayoutPlan) {
        let selected = self
            .window_selected
            .and_then(|id| window_source::find_by_id(&self.windows, id).cloned());
        let text = match (self.window_selected, selected) {
            (Some(_), Some(info)) => format!("已选：{}", window_source::summary_text(&info)),
            (Some(_), None) => "已选：窗口已关闭或不可见".to_owned(),
            (None, _) => "尚未选择窗口".to_owned(),
        };
        let toggle = if self.window_panel_open {
            "收起窗口列表"
        } else {
            "选择窗口"
        };
        if plan.touch {
            ui.label(RichText::new(text));
            let button =
                egui::Button::new(toggle).min_size(egui::vec2(ui.available_width(), TOUCH_MIN));
            if ui.add(button).clicked() {
                self.window_panel_open = !self.window_panel_open;
            }
        } else {
            ui.horizontal(|ui| {
                ui.label(RichText::new(text));
                ui.add_space(8.0);
                if ui.button(toggle).clicked() {
                    self.window_panel_open = !self.window_panel_open;
                }
            });
        }
        if let Some(message) = &self.window_error {
            ui.colored_label(Color32::from_rgb(235, 120, 120), message);
        }
    }

    /// 刷新列表 + search box + a scrollable list of capturable windows.
    fn window_source_panel(&mut self, ui: &mut egui::Ui, plan: &LayoutPlan) {
        ui.add_space(4.0);
        let row_height = if plan.touch { TOUCH_MIN } else { 30.0 };
        let mut refresh = false;
        ui.horizontal(|ui| {
            if plan.touch {
                let width = (ui.available_width() * 0.42).max(120.0);
                if ui
                    .add_sized([width, TOUCH_MIN], egui::Button::new("刷新列表"))
                    .clicked()
                {
                    refresh = true;
                }
            } else if ui.button("刷新列表").clicked() {
                refresh = true;
            }
            ui.add_space(8.0);
            match &self.windows_error {
                Some(message) => {
                    ui.colored_label(Color32::from_rgb(230, 180, 90), message);
                }
                None => {
                    ui.label(RichText::new(format!("共 {} 个可选窗口", self.windows.len())).weak());
                }
            }
        });
        if refresh {
            self.refresh_windows();
        }

        let filter_width = if plan.touch {
            ui.available_width()
        } else {
            260.0
        };
        ui.add_sized(
            [filter_width, row_height.max(24.0)],
            egui::TextEdit::singleline(&mut self.window_filter).hint_text("搜索窗口标题或进程名"),
        );

        let selected = self.window_selected;
        let rows: Vec<(u64, String)> =
            window_source::filter_windows(&self.windows, &self.window_filter)
                .into_iter()
                .map(|info| (info.id, window_source::row_text(info)))
                .collect();
        let list_is_empty = self.windows.is_empty();
        let mut clicked = None;
        egui::ScrollArea::vertical()
            .id_salt("window_select_list")
            .max_height(if plan.touch { 260.0 } else { 180.0 })
            .auto_shrink([false, true])
            .show(ui, |ui| {
                if rows.is_empty() {
                    let message = if list_is_empty {
                        "没有找到可以捕获的窗口（窗口必须出现在任务栏里）。"
                    } else {
                        "没有匹配的窗口，换个关键词试试。"
                    };
                    ui.label(RichText::new(message).weak());
                    return;
                }
                for (id, text) in &rows {
                    let mut button = egui::Button::new(text).selected(selected == Some(*id));
                    if plan.touch {
                        button = button.min_size(egui::vec2(ui.available_width(), TOUCH_MIN));
                    }
                    if ui.add(button).clicked() {
                        clicked = Some(*id);
                    }
                }
            });
        if let Some(id) = clicked {
            self.select_window(id);
        }
        hint(
            ui,
            "列表只显示任务栏里的窗口；选中的窗口要保持打开。窗口最小化时仍会显示，但画面会停在最后一帧。",
        );
    }

    /// Monitor picker; fills the row on phones, fixed width on desktop.
    fn monitor_combo(&mut self, ui: &mut egui::Ui) {
        let selected = self
            .monitors
            .get(self.monitor_idx)
            .map(|monitor| monitor.name.clone())
            .unwrap_or_else(|| "默认显示器".to_owned());
        let width = ui.available_width().clamp(200.0, 320.0);
        egui::ComboBox::from_id_salt("monitor_combo")
            .width(width)
            .selected_text(selected)
            .show_ui(ui, |ui| {
                for (index, monitor) in self.monitors.iter().enumerate() {
                    let text = format!(
                        "{} · {}x{} · {:.0} Hz{}",
                        monitor.name,
                        monitor.dims.width,
                        monitor.dims.height,
                        monitor.refresh_hz,
                        if monitor.primary { " · 主屏" } else { "" }
                    );
                    ui.selectable_value(&mut self.monitor_idx, index, text);
                }
            });
    }

    fn status_strip(&mut self, ui: &mut egui::Ui, plan: &LayoutPlan) {
        let running = self.run.is_some();
        let stalled_for = if running {
            self.last_frame_at
                .map(|instant| instant.elapsed().as_secs_f32())
        } else {
            None
        };
        let texts = status::fps_texts(
            running,
            self.fps.fps(),
            self.last_out_fps,
            self.run_multiplier,
            stalled_for,
        );
        let resolution = if self.texture_dims.0 > 0 {
            format!("{}x{}", self.texture_dims.0, self.texture_dims.1)
        } else {
            "等待画面".to_owned()
        };
        // QA defect D4: while running show the backend the run was STARTED
        // with (a mid-run change takes effect next time and is marked as such);
        // only after stopping does the report tell what actually ran.
        let backend = status::backend_text(
            running,
            &self.run_backend_label,
            self.quality.backend.label(),
            self.last_report
                .as_ref()
                .map(|report| report.backend.as_str()),
        );
        // Frame generation backend: Auto may silently fall back to the CPU path.
        let interp_backend = status::interp_backend_text(
            running,
            &self.run_interp_label,
            self.quality.interp_backend.label(),
            self.last_report
                .as_ref()
                .map(|report| report.interp_backend.as_str()),
        );
        // Show the real capture backend ("window-wgc", "dxgi", ...) once known.
        let source_backend = status::source_text(
            running,
            self.run.as_ref().map(|run| run.source_backend()),
            self.last_report
                .as_ref()
                .map(|report| report.source.as_str()),
        );
        let elapsed = self
            .run
            .as_ref()
            .map(|run| format!(" · 已运行 {:.0} 秒", run.elapsed().as_secs_f32()))
            .unwrap_or_default();
        if plan.touch {
            // Two compact lines instead of one wide row, so nothing is clipped
            // at 320-360 point widths.
            ui.label(
                RichText::new(format!("输入 {} · 输出 {}", texts.input, texts.output)).size(15.0),
            );
            ui.label(
                RichText::new(format!("帧数 {} · 分辨率 {resolution}", self.frames)).size(15.0),
            );
            ui.label(
                RichText::new(format!(
                    "来源 {source_backend} · 放大后端 {backend} · 插帧后端 {interp_backend}{elapsed}"
                ))
                .size(15.0),
            );
        } else {
            // QA defect O1: when a mid-run change adds " · 下次生效：X" to the
            // backend labels, a single wide row overflowed the default window
            // width and the tail was clipped. Split the strip over two rows so
            // the backend text always has a full row to itself.
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("输入 {}", texts.input)).size(15.0));
                ui.label(RichText::new("·").weak());
                ui.label(RichText::new(format!("输出 {}", texts.output)).size(15.0));
                ui.label(RichText::new("·").weak());
                ui.label(RichText::new(format!("帧数 {}", self.frames)).size(15.0));
                ui.label(RichText::new("·").weak());
                ui.label(RichText::new(format!("分辨率 {resolution}")).size(15.0));
                ui.label(RichText::new("·").weak());
                ui.label(RichText::new(format!("来源 {source_backend}")).size(15.0));
            });
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("放大后端 {backend}")).size(15.0));
                ui.label(RichText::new("·").weak());
                ui.label(RichText::new(format!("插帧后端 {interp_backend}")).size(15.0));
                if !elapsed.is_empty() {
                    ui.label(RichText::new(elapsed.trim_start_matches(" · ")).weak());
                }
            });
        }
        if let Some(note) = texts.note {
            ui.label(
                RichText::new(note)
                    .size(13.0)
                    .color(Color32::from_rgb(230, 180, 90)),
            );
        }
        if let Some(line) = prewarm::status_text(&self.prewarm_state, running) {
            ui.label(RichText::new(line).size(13.0).weak());
        }
    }

    fn preview(&mut self, ui: &mut egui::Ui, height: f32) {
        let width = ui.available_width().max(200.0);
        let (rect, _response) =
            ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
        ui.painter()
            .rect_filled(rect, 10.0, Color32::from_rgb(16, 17, 22));
        let Some(texture) = &self.texture else {
            ui.put(
                rect,
                egui::Label::new(
                    RichText::new("点击“开始增强”后，这里会显示处理后的画面")
                        .size(17.0)
                        .weak(),
                ),
            );
            return;
        };
        let size = fit_size(rect.size() - egui::vec2(24.0, 24.0), self.texture_dims);
        let image_rect = egui::Rect::from_center_size(rect.center(), size);
        let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        ui.painter()
            .image(texture.id(), image_rect, uv, Color32::WHITE);
    }

    /// Draw the optional second, chrome-less fullscreen viewport.
    ///
    /// This is how a gamer sees the enhanced picture: the game keeps rendering
    /// on the desktop, Fluid captures it and shows the interpolated/upscaled
    /// result in this window. Esc, unchecking 全屏输出, or stopping the run all
    /// close it again; the viewport is simply not requested on the next frame.
    fn fullscreen_output_viewport(&mut self, ctx: &egui::Context) {
        if !status::output_visible(self.fullscreen_output, self.run.is_some()) {
            self.output_opened_at = None;
            self.output_affinity = None;
            return;
        }
        let texture = self.texture.clone();
        let dims = self.texture_dims;
        let opened_at = *self.output_opened_at.get_or_insert_with(Instant::now);
        let show_hint = opened_at.elapsed().as_secs_f32() < 5.0;
        let viewport_id = egui::ViewportId::from_hash_of("fluid_fullscreen_output");
        let builder = egui::ViewportBuilder::default()
            .with_title(display_affinity::OUTPUT_WINDOW_TITLE)
            .with_decorations(false)
            .with_resizable(false)
            .with_fullscreen(true);
        let close_requested = ctx.show_viewport_immediate(viewport_id, builder, |ui, _class| {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(16));
            let rect = ui.max_rect();
            ui.painter().rect_filled(rect, 0.0, Color32::BLACK);
            if let Some(texture) = &texture {
                let size = fit_size(rect.size(), dims);
                let image_rect = egui::Rect::from_center_size(rect.center(), size);
                let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                ui.painter()
                    .image(texture.id(), image_rect, uv, Color32::WHITE);
            }
            if show_hint {
                ui.painter().text(
                    egui::pos2(rect.center().x, rect.bottom() - 26.0),
                    egui::Align2::CENTER_BOTTOM,
                    "按 Esc 关闭全屏输出",
                    egui::FontId::proportional(16.0),
                    Color32::from_white_alpha(170),
                );
            }
            let escape = ui.input(|input| input.key_pressed(egui::Key::Escape));
            let closed = ui.input(|input| input.viewport().close_requested());
            escape || closed
        });

        // The viewport window exists now: hide it from screen capture so DXGI
        // desktop duplication does not feed Fluid's own output back into the
        // pipeline (the "infinite mirror" on a single monitor). Retried while
        // the window is still being created, then latched.
        if self.output_affinity.is_none() {
            let outcome = display_affinity::exclude_output_window();
            let give_up = opened_at.elapsed().as_secs_f32() >= 5.0;
            if !matches!(outcome, display_affinity::Affinity::NotFound) || give_up {
                eprintln!("{}", outcome.log_line());
                self.output_affinity = Some(outcome);
            }
        }

        if close_requested {
            self.fullscreen_output = false;
            self.output_opened_at = None;
            self.output_affinity = None;
        }
    }

    fn control_panel(&mut self, ui: &mut egui::Ui, plan: &LayoutPlan) {
        let running = self.run.is_some();
        if plan.bottom_bar {
            hint(
                ui,
                "开始 / 停止和“全屏输出”在屏幕底部的大按钮上，手指更容易按到。",
            );
        } else {
            let width = ui.available_width().max(200.0);
            let label = if running {
                "停止增强"
            } else {
                "开始增强"
            };
            let fill = if running {
                Color32::from_rgb(150, 62, 62)
            } else {
                Color32::from_rgb(46, 120, 86)
            };
            let button = egui::Button::new(RichText::new(label).size(22.0).strong())
                .min_size(egui::vec2(width, plan.primary_button_height()))
                .fill(fill);
            if ui.add(button).clicked() {
                if running {
                    self.stop_run();
                } else {
                    self.start_run(false);
                }
            }
            hint(
                ui,
                "空格键也可以开始 / 停止。第一次使用建议先试内置测试画面。",
            );
            ui.add_space(6.0);
            ui.checkbox(&mut self.fullscreen_output, "全屏输出");
        }
        hint(
            ui,
            "开启后，处理好的画面会另外显示在一个无边框全屏窗口里：游戏照常在桌面运行，Fluid 把插帧和放大后的结果铺满屏幕。按 Esc 可以关闭全屏窗口。",
        );
        hint(
            ui,
            "建议：游戏用「无边框窗口」运行；如果有两块屏幕，用 Win+Shift+方向键 把全屏输出移到第二块屏幕，避免屏幕捕获把 Fluid 自己的全屏画面也拍进去。",
        );
        if let Some(affinity) = self.output_affinity {
            hint(ui, affinity.hint());
        }

        if let Some(since) = self.stopping_since {
            let waited = since.elapsed().as_secs_f32();
            ui.label(RichText::new(format!("正在等待旧线程退出（{waited:.1} 秒）……")).weak());
            if waited > 3.0 {
                if ui.button("强制停止").clicked() {
                    self.force_stop();
                }
                hint(ui, "强制停止会放弃旧线程的结果，一般不需要使用。");
            }
        }

        ui.add_space(8.0);
        ui.separator();
        self.profiles_section(ui, plan);

        ui.add_space(8.0);
        ui.separator();
        ui.heading("画质设置");
        hint(
            ui,
            "不确定时保持“均衡”即可，下面的选项会影响下一次开始增强的效果。",
        );
        ui.add_space(4.0);

        let style = ui.style().clone();
        let mut advanced_open = self.advanced_open;
        let toggled = egui::Frame::group(&style)
            .show(ui, |ui| {
                quality_controls(ui, &mut self.quality, &mut advanced_open, plan.touch)
            })
            .inner;
        if toggled || advanced_open != self.advanced_open {
            self.advanced_open = advanced_open;
            self.ui_flags_dirty = true;
        }

        if let Some(report) = self.last_report.clone() {
            ui.add_space(8.0);
            ui.separator();
            ui.label(RichText::new("上次运行报告").strong());
            report_grid(ui, &report);
        }
    }

    fn video_tab(&mut self, ui: &mut egui::Ui, plan: &LayoutPlan) {
        ui.heading("视频处理");
        hint(
            ui,
            "把录制好的视频离线插帧、放大，导出成新文件。支持 .y4m 和 .avi：输入两种格式都可以，输出格式由输出文件名的扩展名决定（.y4m 无损、体积大；.avi 是 MJPEG、体积小）。处理在后台线程进行，界面不会卡住。",
        );
        ui.add_space(6.0);

        let style = ui.style().clone();
        let dialogs = file_picker::dialogs_available();
        egui::Frame::group(&style).show(ui, |ui| {
            self.video_path_row(
                ui,
                plan,
                "输入文件",
                "例如 C:\\videos\\clip.y4m 或 clip.avi",
                true,
                dialogs,
            );
            self.video_path_row(
                ui,
                plan,
                "输出文件",
                "留空则自动写成 输入文件.fluid.y4m",
                false,
                dialogs,
            );
            if !dialogs {
                hint(ui, file_picker::MANUAL_HINT);
            }
            if let Some(info) = self.video_input_info.clone() {
                ui.label(RichText::new(format!("输入：{}", info.summary())).size(14.0));
            }
            if let Some(err) = self.video_input_error.clone() {
                ui.colored_label(Color32::from_rgb(235, 120, 120), err);
            } else if self.video_input_info.is_none() {
                hint(ui, "选好文件后，这里会显示分辨率、帧率和帧数。");
            }
            let output_text = self.video_output.trim().to_owned();
            if !output_text.is_empty() {
                match video_format::detect_format(&output_text) {
                    Ok(format) => {
                        ui.label(RichText::new(format!("输出格式：{}", format.label())).size(14.0));
                        if video_format::quality_applies(format) {
                            ui.horizontal_wrapped(|ui| {
                                ui.label("JPEG 质量");
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.video_quality,
                                        video_format::QUALITY_MIN..=video_format::QUALITY_MAX,
                                    )
                                    .integer(),
                                );
                            });
                            hint(ui, "只对 .avi 生效：越高越清晰、文件越大，88 是默认值。");
                        }
                    }
                    Err(err) => {
                        ui.colored_label(Color32::from_rgb(235, 120, 120), err);
                    }
                }
            }
        });

        ui.add_space(6.0);
        ui.heading("画质设置");
        let style = ui.style().clone();
        let mut advanced_open = self.advanced_open;
        let toggled = egui::Frame::group(&style)
            .show(ui, |ui| {
                quality_controls(ui, &mut self.quality, &mut advanced_open, plan.touch)
            })
            .inner;
        if toggled || advanced_open != self.advanced_open {
            self.advanced_open = advanced_open;
            self.ui_flags_dirty = true;
        }

        ui.add_space(6.0);
        let processing = self.video_job.is_some();
        let label = if processing {
            "停止处理"
        } else {
            "开始处理"
        };
        if plan.touch {
            let width = ui.available_width().max(200.0);
            if ui
                .add_sized(
                    egui::vec2(width, plan.primary_button_height()),
                    egui::Button::new(RichText::new(label).size(20.0)),
                )
                .clicked()
            {
                if processing {
                    if let Some(job) = &self.video_job {
                        job.stop();
                    }
                } else {
                    self.start_video();
                }
            }
            if processing {
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new());
                    ui.label("正在后台处理……");
                });
            } else {
                hint(ui, "输出文件留空时，会自动写成“输入文件.fluid.y4m”。");
            }
        } else {
            ui.horizontal(|ui| {
                if ui
                    .add_sized(
                        egui::vec2(170.0, plan.primary_button_height()),
                        egui::Button::new(RichText::new(label).size(18.0)),
                    )
                    .clicked()
                {
                    if processing {
                        if let Some(job) = &self.video_job {
                            job.stop();
                        }
                    } else {
                        self.start_video();
                    }
                }
                ui.add_space(8.0);
                if processing {
                    ui.add(egui::Spinner::new());
                    ui.label("正在后台处理……");
                } else {
                    hint(ui, "输出文件留空时，会自动写成“输入文件.fluid.y4m”。");
                }
            });
        }

        let state = self.video_state.clone();
        if self.video_job.is_some() || state.done || state.error.is_some() {
            ui.add_space(6.0);
            let fraction = state.fraction();
            let text = if state.total_estimate > 0 {
                format!(
                    "{} / {} 帧（{:.0}%）",
                    state.frames_out,
                    state.total_estimate,
                    fraction * 100.0
                )
            } else {
                format!("已输出 {} 帧", state.frames_out)
            };
            let width = ui.available_width().max(200.0);
            ui.add(
                egui::ProgressBar::new(fraction)
                    .desired_width(width)
                    .text(text),
            );
            if state.src_dims.width > 0 && state.out_dims.width > 0 {
                let mut line = format!(
                    "分辨率：{}x{} → {}x{}",
                    state.src_dims.width,
                    state.src_dims.height,
                    state.out_dims.width,
                    state.out_dims.height
                );
                if state.running {
                    line.push_str(&format!(" · 已用 {:.0} 秒", state.elapsed_s));
                }
                ui.label(line);
            }
            if state.running {
                ui.label(RichText::new(state.eta_text()).weak());
            }
            if state.done && !state.cancelled {
                ui.colored_label(Color32::from_rgb(120, 200, 140), "处理完成。");
            }
            if let Some(note) = state.cancelled_text() {
                ui.colored_label(Color32::from_rgb(235, 190, 110), note);
            }
            if let Some(message) = &state.error {
                ui.colored_label(
                    Color32::from_rgb(235, 120, 120),
                    format!("处理失败：{message}"),
                );
            }
            if let Some(report) = state.report.clone() {
                ui.add_space(4.0);
                report_grid(ui, &report);
            }
            if let Some(path) = state.output_path.clone() {
                ui.label(
                    RichText::new(format!("输出文件：{}", path.display()))
                        .size(13.0)
                        .weak(),
                );
                if state.done
                    && state.error.is_none()
                    && file_picker::dialogs_available()
                    && dialog_button(ui, plan.touch, "打开输出文件夹")
                {
                    match file_picker::reveal_output(&path) {
                        Ok(()) => self.video_reveal_error = None,
                        Err(err) => self.video_reveal_error = Some(err),
                    }
                }
            }
        }
        if let Some(message) = self.video_reveal_error.clone() {
            ui.colored_label(Color32::from_rgb(235, 190, 110), message);
        }
        if let Some(message) = self.video_error.clone() {
            ui.colored_label(Color32::from_rgb(235, 120, 120), message);
        }
    }

    fn doctor_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("性能体检");
        hint(
            ui,
            "用内置测试画面跑几个小测试，估算这台电脑适合多高的画质设置。整个过程约 5-10 秒，期间界面仍可正常操作。",
        );
        ui.add_space(6.0);

        let running = self.doctor_state.running;
        ui.horizontal(|ui| {
            let button = egui::Button::new(RichText::new("开始体检").size(18.0))
                .min_size(egui::vec2(150.0, 44.0));
            if ui.add_enabled(!running, button).clicked() {
                let mut config = self.config.clone();
                self.quality.apply(&mut config);
                self.config = config.clone();
                self.saved_quality = self.quality;
                self.doctor_state = DoctorState::default();
                self.doctor_ran = true;
                self.doctor = Some(DoctorHandle::spawn(config));
            }
            if running {
                ui.add(egui::Spinner::new());
                ui.label("正在测试……");
            } else if !self.doctor_ran {
                hint(
                    ui,
                    "测试会按上面的放大后端和插帧后端设置运行，不会影响正在运行的游戏。",
                );
            }
        });

        if !self.doctor_state.results.is_empty() {
            ui.add_space(10.0);
            let style = ui.style().clone();
            egui::Frame::group(&style).show(ui, |ui| {
                egui::Grid::new("doctor_grid")
                    .num_columns(7)
                    .spacing(egui::vec2(16.0, 8.0))
                    .show(ui, |ui| {
                        for header in [
                            "测试项目",
                            "输入",
                            "输出",
                            "放大后端",
                            "插帧后端",
                            "平均耗时",
                            "输出帧率",
                        ] {
                            ui.label(RichText::new(header).strong());
                        }
                        ui.end_row();
                        for result in &self.doctor_state.results {
                            ui.label(&result.label);
                            ui.label(format!(
                                "{}x{}",
                                result.src_dims.width, result.src_dims.height
                            ));
                            ui.label(format!(
                                "{}x{}",
                                result.out_dims.width, result.out_dims.height
                            ));
                            ui.label(backend_display_name(&result.backend));
                            ui.label(interp_backend_display_name(&result.interp_backend));
                            ui.label(format!("{:.1} ms", result.mean_ms));
                            ui.label(format!("{:.0} fps", result.fps));
                            ui.end_row();
                        }
                    });
            });
            if !self.doctor_state.recommendation.is_empty() {
                ui.add_space(10.0);
                ui.label(
                    RichText::new(self.doctor_state.recommendation.clone())
                        .size(20.0)
                        .strong(),
                );
                hint(
                    ui,
                    "这只是根据 CPU 处理能力做的估算，实际帧率还会受到游戏、驱动和屏幕刷新率的影响。",
                );
            }
        }
        if let Some(message) = &self.doctor_state.error {
            ui.colored_label(
                Color32::from_rgb(235, 120, 120),
                format!("体检失败：{message}"),
            );
        }

        ui.add_space(10.0);
        let style = ui.style().clone();
        egui::Frame::group(&style).show(ui, |ui| {
            ui.label(RichText::new("本机信息").strong());
            ui.label(format!("CPU 线程：{}", self.threads));
            ui.label(format!("显示器：{}", self.display));
            ui.label(format!(
                "界面布局：{}",
                layout::class_label(self.layout_class)
            ));
            ui.label(format!("配置文件：{}", self.config_path.display()));
            if let Some(path) = self.font_path {
                ui.label(format!("中文字体：{path}"));
            } else {
                ui.colored_label(
                    Color32::from_rgb(230, 180, 90),
                    "未找到系统中文字体，界面中文可能显示为方块。",
                );
            }
        });
    }

    fn about_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("关于 Fluid");
        ui.label(RichText::new(fluid_core::version_string()).size(18.0));
        hint(
            ui,
            "Fluid 是一个开源的帧生成（插帧）与画质增强工具，纯 CPU 实现，类似 DLSS / FSR 的思路，但不需要显卡的专用单元。",
        );
        ui.add_space(8.0);

        let style = ui.style().clone();
        egui::Frame::group(&style).show(ui, |ui| {
            ui.label(RichText::new("它是什么，不是什么").strong());
            ui.label("Fluid 只读取屏幕画面，在自己的窗口里处理后显示。");
            ui.label("它不会注入游戏进程、不读游戏内存、不修改游戏文件，所以不是作弊工具。");
            ui.label("效果相当于在显示端做后期处理：游戏本身不受影响，录制软件也可能拍不到插帧后的画面。");
        });

        ui.add_space(8.0);
        egui::Frame::group(&style).show(ui, |ui| {
            ui.label(RichText::new("文件与配置").strong());
            ui.label(format!("配置文件：{}", self.config_path.display()));
            ui.label(format!("配置档目录：{}", self.profiles_dir.display()));
            ui.label(format!("自动套用规则：{}", self.rules_path.display()));
            ui.label(format!(
                "项目文档：{}\\README.md 与 {}\\docs\\ARCHITECTURE.md",
                env!("CARGO_MANIFEST_DIR"),
                env!("CARGO_MANIFEST_DIR")
            ));
            ui.label("许可证：AGPL-3.0-or-later（开源，可自由使用和修改，衍生作品需同样开源）");
        });

        ui.add_space(8.0);
        egui::Frame::group(&style).show(ui, |ui| {
            ui.label(RichText::new("全屏输出怎么用").strong());
            ui.label("1. 在“实时增强”里勾选“全屏输出”，然后点“开始增强”。");
            ui.label("2. 游戏用无边框窗口或全屏运行在桌面上，Fluid 会读取屏幕并在另一个全屏窗口里显示插帧后的画面。");
            ui.label("3. 在全屏窗口里按 Esc 只会关闭这个全屏窗口（不会退出 Fluid）；想停止增强请回到主窗口点“停止增强”，或取消勾选“全屏输出”。");
            ui.label("提示：如果游戏独占全屏、Fluid 窗口拿不到画面，请把游戏改成「无边框窗口」模式。");
            ui.label("提示：只有一块屏幕时，全屏输出会被屏幕捕获再拍一次（画面会变糊）；建议用两块屏幕，并用 Win+Shift+方向键 把全屏输出移到第二块屏幕。");
        });

        ui.add_space(8.0);
        egui::Frame::group(&style).show(ui, |ui| {
            ui.label(RichText::new("键盘快捷键（只在电脑上）").strong());
            ui.label("空格：开始 / 停止实时增强");
            ui.label("Ctrl+1 / 2 / 3 / 4 或 Alt+1 / 2 / 3 / 4：切换标签页（部分输入法会占用 Ctrl+1，这时用 Alt+1）");
            ui.label("Esc：在主窗口里停止增强；全屏输出窗口里只关闭该窗口");
            hint(
                ui,
                "触屏设备没有键盘也能完成所有操作：每个功能都有对应的按钮。",
            );
        });

        ui.add_space(8.0);
        egui::Frame::group(&style).show(ui, |ui| {
            ui.label(RichText::new("配置档与自动套用").strong());
            ui.label("配置档把整套画质设置存成一个名字（例如“某游戏的流畅档”），之后在“实时增强 → 配置档”里一键切换。");
            ui.label("“启动时应用上次的配置”打开后，重新打开 Fluid 会自动恢复上次用的配置档。");
            ui.label("“按窗口自动切换档位”打开后，选中保存过规则的窗口时自动换到对应配置档；规则在配置档窗口里添加。");
            ui.label(format!("当前配置档：{}", self.active_profile.as_deref().unwrap_or("自定义")));
            ui.label(format!("已保存 {} 个配置档，{} 条自动套用规则。", self.profile_names.len(), self.rules.len()));
            hint(ui, "配置档只是把同一套 FluidConfig 存成文件，没有新增任何配置项，命令行和配置文件仍然兼容。");
        });

        ui.add_space(8.0);
        egui::Frame::group(&style).show(ui, |ui| {
            ui.label(RichText::new("手机 / 触屏").strong());
            ui.label("同一套界面在电脑和手机上都可用：窄窗口或手机竖屏时自动变成上下布局，预览在上、设置在下面滚动，开始 / 停止按钮固定在屏幕底部。");
            ui.label("两个平台的设置项完全一样，没有只在电脑上才能改的选项；高级选项（放大算法、放大后端、插帧后端）在手机默认收起，点“高级设置”展开。");
            ui.label("放大和插帧都可以选“自动（推荐）”：自动会先用 GPU，显卡不可用或出错时自动回退 CPU，运行报告里的“放大后端 / 插帧后端”显示实际用了哪个。");
            ui.label("触屏上按钮至少 44 点高、滑动条更粗，手指更容易操作。");
            ui.label("键盘快捷键只是电脑上的加速方式，不会成为某个功能的唯一入口。");
        });

        ui.add_space(8.0);
        egui::Frame::group(&style).show(ui, |ui| {
            ui.label(RichText::new("中文字体").strong());
            if let Some(path) = self.font_path {
                ui.label(format!("已加载：{path}"));
            } else {
                ui.colored_label(
                    Color32::from_rgb(230, 180, 90),
                    "未找到可用的中文字体。请安装微软雅黑（msyh.ttc）或 Noto Sans CJK 后重启 Fluid。",
                );
            }
        });

        ui.add_space(8.0);
        hint(
            ui,
            "电脑上可以把画面来源切成“窗口捕获”，只读取一个游戏窗口；手机上没有这个选项，因为 Android 只提供整屏录制（MediaProjection）权限。",
        );
        hint(
            ui,
            "Android 版本需要系统授予屏幕录制（MediaProjection）权限；没有权限时会自动退回内置测试画面。",
        );
    }
}

impl eframe::App for FluidApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll(&ctx);
        self.handle_shortcuts(&ctx);
        self.fit_window(&ctx);

        // Responsive layout, recomputed from the window size every frame:
        // wide desktop -> preview left / settings right, phone or narrow
        // window -> stacked preview-first with a fixed bottom action bar.
        let plan = LayoutPlan::new(ui.available_size());
        self.layout_class = plan.class;
        plan.apply_style(ui.style_mut());
        if self.advanced_open.is_none() {
            self.advanced_open = Some(plan.advanced_default_open);
        }
        if self.ui_flags_dirty {
            self.ui_flags_dirty = false;
            self.persist_ui_flags();
        }

        egui::Panel::top("fluid_top_bar").show(ui, |ui| {
            ui.add_space(4.0);
            self.top_bar(ui, &plan);
            ui.add_space(4.0);
        });

        if plan.bottom_bar {
            egui::Panel::bottom("fluid_bottom_bar").show(ui, |ui| {
                ui.add_space(4.0);
                self.bottom_bar(ui, &plan);
                ui.add_space(2.0);
            });
        }

        egui::CentralPanel::default().show(ui, |ui| {
            self.banners(ui);
            match self.tab {
                Tab::Live => self.live_tab(ui, &plan),
                Tab::Video => {
                    if plan.side_by_side {
                        self.video_tab(ui, &plan);
                    } else {
                        egui::ScrollArea::vertical()
                            .id_salt("video_tab_scroll")
                            .show(ui, |ui| self.video_tab(ui, &plan));
                    }
                }
                Tab::Doctor => {
                    if plan.side_by_side {
                        self.doctor_tab(ui);
                    } else {
                        egui::ScrollArea::vertical()
                            .id_salt("doctor_tab_scroll")
                            .show(ui, |ui| self.doctor_tab(ui));
                    }
                }
                Tab::About => {
                    if plan.side_by_side {
                        self.about_tab(ui);
                    } else {
                        egui::ScrollArea::vertical()
                            .id_salt("about_tab_scroll")
                            .show(ui, |ui| self.about_tab(ui));
                    }
                }
            }
        });

        self.profiles_dialog(&ctx, &plan);
        self.fullscreen_output_viewport(&ctx);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if let Some(mut run) = self.run.take() {
            run.stop();
            // Closing the window while a run is active should still leave a
            // real summary behind (QA defect D5), so give the worker a short,
            // bounded moment to publish its report before the process dies.
            let deadline = Instant::now() + Duration::from_millis(500);
            while !run.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            if run.is_finished() {
                run.join();
                if let Some(report) = run.take_report() {
                    self.last_out_fps = Some(report.stats.output_fps_value());
                    self.last_report = Some(report);
                }
            }
        }
        if let Some(job) = &self.video_job {
            job.stop();
        }
        if let Some(doctor) = &self.doctor {
            let _ = doctor;
        }
        self.quality.apply(&mut self.config);
        self.save_json_summary();
        let _ = config_io::save(&self.config, &self.config_path);
    }
}

// ---------------------------------------------------------------- helpers

fn hint(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).small().weak());
}

/// A small button next to a label, padded to the 44 pt touch minimum on phones.
fn dialog_button(ui: &mut egui::Ui, touch: bool, label: &str) -> bool {
    if touch {
        ui.add(egui::Button::new(label).min_size(egui::vec2(TOUCH_MIN * 1.6, TOUCH_MIN)))
            .clicked()
    } else {
        ui.button(label).clicked()
    }
}

fn fit_size(available: egui::Vec2, dims: (u32, u32)) -> egui::Vec2 {
    let ratio = if dims.0 == 0 || dims.1 == 0 {
        16.0 / 9.0
    } else {
        dims.0 as f32 / dims.1 as f32
    };
    let max_width = available.x.max(1.0);
    let max_height = available.y.max(1.0);
    let mut width = max_width;
    let mut height = width / ratio;
    if height > max_height {
        height = max_height;
        width = height * ratio;
    }
    egui::vec2(width.max(1.0), height.max(1.0))
}

fn quality_controls(
    ui: &mut egui::Ui,
    quality: &mut UiQuality,
    advanced_open: &mut Option<bool>,
    touch: bool,
) -> bool {
    let mut toggled = false;

    ui.horizontal(|ui| {
        ui.label(RichText::new("预设").strong());
        let before = quality.preset;
        egui::ComboBox::from_id_salt("preset_combo")
            .width(if touch { 190.0 } else { 120.0 })
            .selected_text(quality.preset.label())
            .show_ui(ui, |ui| {
                for preset in PresetChoice::ALL {
                    ui.selectable_value(&mut quality.preset, preset, preset.label());
                }
            });
        if quality.preset != before {
            quality.select_preset(quality.preset);
        }
    });
    hint(ui, quality.preset.hint());
    ui.add_space(6.0);

    ui.label(RichText::new("插帧倍率").strong());
    ui.horizontal_wrapped(|ui| {
        for choice in InterpChoice::ALL {
            ui.selectable_value(&mut quality.interp, choice, choice.label());
        }
    });
    hint(
        ui,
        "把 60 帧的画面补到 120 / 180 / 240 帧，动作更顺滑。关掉就只做放大。",
    );
    ui.add_space(6.0);

    ui.label(RichText::new("分辨率提升").strong());
    ui.horizontal_wrapped(|ui| {
        for choice in ScaleChoice::ALL {
            ui.selectable_value(&mut quality.scale, choice, choice.label());
        }
    });
    hint(
        ui,
        "把画面放大到更高分辨率。“4K”适合 4K 显示器，1080p 屏幕选 1.5x 或 2x 即可。",
    );
    ui.add_space(8.0);

    // 高级选项：放大算法和计算后端。桌面上默认展开，手机 / 窄窗口默认收起，
    // 收起时也能看到当前选择了什么。
    let open = advanced_open.unwrap_or(!touch);
    let header_text = if open {
        "高级设置：已展开（点这里收起）"
    } else {
        "高级设置：已收起（点这里展开）"
    };
    let mut header = egui::Button::new(RichText::new(header_text).size(15.0)).frame(false);
    if touch {
        header = header.min_size(egui::vec2(ui.available_width(), TOUCH_MIN));
    }
    if ui.add(header).clicked() {
        *advanced_open = Some(!open);
        toggled = true;
    }
    let open = advanced_open.unwrap_or(!touch);
    if open {
        ui.label(RichText::new("放大算法").strong());
        let filter_width = if touch {
            ui.available_width().min(320.0)
        } else {
            160.0
        };
        egui::ComboBox::from_id_salt("filter_combo")
            .width(filter_width)
            .selected_text(quality.filter.label())
            .show_ui(ui, |ui| {
                for filter in FilterChoice::ALL {
                    ui.selectable_value(&mut quality.filter, filter, filter.label());
                }
            });
        hint(ui, quality.filter.hint());
        ui.add_space(6.0);

        ui.label(RichText::new("插帧后端").strong());
        let interp_off = quality.interp == InterpChoice::Off;
        ui.add_enabled_ui(!interp_off, |ui| {
            ui.horizontal_wrapped(|ui| {
                for choice in InterpBackendChoice::ALL {
                    ui.selectable_value(&mut quality.interp_backend, choice, choice.label());
                }
            });
        });
        hint(
            ui,
            interp_backend_hint(quality.interp, quality.interp_backend),
        );
        ui.add_space(6.0);

        ui.label(RichText::new("计算后端").strong());
        ui.horizontal_wrapped(|ui| {
            for choice in BackendChoice::ALL {
                ui.selectable_value(&mut quality.backend, choice, choice.label());
            }
        });
        hint(ui, quality.backend.hint());
        ui.add_space(6.0);
    } else {
        hint(
            ui,
            &format!(
                "放大算法 {} · 插帧后端 {} · 计算后端 {}（点上面一行展开）",
                quality.filter.label(),
                quality.interp_backend.label(),
                quality.backend.label()
            ),
        );
    }

    ui.add_space(6.0);
    ui.label(RichText::new("锐化").strong());
    let slider = egui::Slider::new(&mut quality.sharpness, 0.0..=1.0)
        .step_by(0.01)
        .text("强度");
    ui.add(slider);
    hint(ui, "0 表示不锐化；0.3 左右比较自然；过高会出现白边。");
    ui.add_space(6.0);

    ui.checkbox(&mut quality.vsync, "垂直同步");
    hint(ui, "减少画面撕裂，可能会稍微降低帧率。");
    ui.checkbox(&mut quality.borderless, "无边框全屏");
    hint(ui, "开始增强后占满整个屏幕，按 Esc 退出。");

    toggled
}

/// Hint shown under the 插帧后端 row: greyed-out explanation when 插帧倍率 is off.
fn interp_backend_hint(interp: InterpChoice, backend: InterpBackendChoice) -> &'static str {
    if interp == InterpChoice::Off {
        "“插帧倍率”现在是关，这个设置不会生效；选 2x / 3x / 4x 之后才用得到。"
    } else {
        backend.hint()
    }
}

fn report_grid(ui: &mut egui::Ui, report: &PipelineReport) {
    let stats = &report.stats;
    let latency_p50 = stats.frame_latency_ns.p50_ns() as f64 / 1e6;
    let latency_p99 = stats.frame_latency_ns.p99_ns() as f64 / 1e6;
    let rows = vec![
        (
            "处理链路".to_owned(),
            format!(
                "{} → {}（{}x 插帧，{}）",
                report.source, report.sink, report.multiplier, report.filter
            ),
        ),
        ("放大后端".to_owned(), backend_display_name(&report.backend)),
        (
            "插帧后端".to_owned(),
            interp_backend_display_name(&report.interp_backend),
        ),
        (
            "输出分辨率".to_owned(),
            format!("{}x{}", report.out_dims.width, report.out_dims.height),
        ),
        (
            "输入 / 输出帧数".to_owned(),
            format!("{} / {}", stats.frames_in, stats.frames_out),
        ),
        (
            "插帧 / 场景切换".to_owned(),
            format!("{} / {}", stats.interpolated, stats.scene_cuts),
        ),
        (
            "静态帧 / 重复帧".to_owned(),
            format!("{} / {}", stats.static_frames, stats.duplicates),
        ),
        (
            "源等待 / 呈现耗时".to_owned(),
            format!(
                "{:.2} / {:.2} ms",
                stats.source_wait_ns.mean_ns() as f64 / 1e6,
                stats.present_ns.mean_ns() as f64 / 1e6
            ),
        ),
        (
            "输入 / 输出帧率".to_owned(),
            format!(
                "{:.1} / {:.1} fps",
                stats.input_fps_value(),
                stats.output_fps_value()
            ),
        ),
        (
            "单帧耗时 平均 / P99".to_owned(),
            format!(
                "{:.2} / {:.2} ms",
                stats.mean_process_ms(),
                stats.p99_process_ms()
            ),
        ),
        (
            "端到端延迟 P50 / P99".to_owned(),
            format!("{latency_p50:.1} / {latency_p99:.1} ms"),
        ),
        (
            "运行时长".to_owned(),
            format!(
                "{:.1} 秒（实时系数 {:.2}x）",
                report.elapsed_s(),
                report.realtime_factor()
            ),
        ),
    ];

    egui::Grid::new("fluid_report_grid")
        .num_columns(2)
        .spacing(egui::vec2(16.0, 6.0))
        .show(ui, |ui| {
            for (key, value) in rows {
                ui.label(RichText::new(key).weak());
                ui.label(value);
                ui.end_row();
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pure check: every tab has exactly one number shortcut and the order
    /// matches the tab bar, so Ctrl/Alt+N always selects the Nth tab.
    #[test]
    fn tab_shortcuts_cover_every_tab_in_order() {
        let expected_keys = [
            egui::Key::Num1,
            egui::Key::Num2,
            egui::Key::Num3,
            egui::Key::Num4,
        ];
        assert_eq!(TAB_SHORTCUTS.len(), Tab::ALL.len());
        for (index, (key, tab)) in TAB_SHORTCUTS.iter().enumerate() {
            assert_eq!(*tab, Tab::ALL[index]);
            assert_eq!(*key, expected_keys[index]);
        }
    }

    /// The 插帧后端 row must explain that it is inert while 插帧倍率 is 关.
    #[test]
    fn interp_backend_hint_switches_with_the_multiplier() {
        let off = interp_backend_hint(InterpChoice::Off, InterpBackendChoice::Gpu);
        assert!(off.contains("不会生效"), "unexpected hint: {off}");
        assert!(!off.contains("显卡做插帧"));

        let on = interp_backend_hint(InterpChoice::X2, InterpBackendChoice::Gpu);
        assert_eq!(on, InterpBackendChoice::Gpu.hint());
    }

    /// Every 插帧后端 choice must be reachable from the UI.
    #[test]
    fn interp_backend_choosers_cover_all_choices() {
        assert_eq!(InterpBackendChoice::ALL.len(), 3);
        for choice in InterpBackendChoice::ALL {
            assert!(!choice.label().is_empty());
            assert!(!choice.hint().is_empty());
        }
    }
}
