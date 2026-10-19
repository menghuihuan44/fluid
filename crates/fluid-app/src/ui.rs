//! egui settings window: capture source, quality, frame generation, stats.

use crate::monitor_hud::HudService;
use crate::pipeline::{CaptureRequest, Command, Pipeline};
use eframe::egui;
use fluid_capture::monitor::{enumerate_monitors, MonitorInfo};
use fluid_capture::window::{
    enumerate_windows, external_foreground_window, hwnd_from_isize, is_window, window_info, WinEnum,
};
use fluid_core::config::WindowRef;
use fluid_core::{AppConfig, CaptureMode, HudAnchor, OutputMode, PresentMode, RunState, ScaleMode};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

pub struct FluidApp {
    cfg: AppConfig,
    applied: AppConfig,
    config_snapshot: AppConfig,
    config_edit_at: std::time::Instant,
    windows: Vec<WinEnum>,
    selected: Option<usize>,
    monitors: Vec<MonitorInfo>,
    selected_monitor: Option<usize>,
    last_foreground: Option<WinEnum>,
    pipe: Pipeline,
    hud: HudService,
    shared: Arc<fluid_core::SharedState>,
    auto_start_pending: bool,
    quitting: bool,
    hidden: bool,
    saw_minimized: bool,
    notice: String,
    ui_frames: u64,
    ui_last_report: std::time::Instant,
}

impl FluidApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        shared: Arc<fluid_core::SharedState>,
        pipe: Pipeline,
        hud: HudService,
        cfg: AppConfig,
        initial_foreground: Option<WinEnum>,
    ) -> Self {
        install_cjk_fonts(&cc.egui_ctx);
        crate::wake::set(&cc.egui_ctx);
        let windows = enumerate_windows();
        let selected = locate_window(&windows, cfg.target.window.as_ref());
        let monitors = enumerate_monitors();
        let selected_monitor = locate_monitor(&monitors, cfg.target.monitor.as_deref());
        let last_foreground =
            initial_foreground.or_else(|| external_foreground_window().and_then(window_info));
        let auto_start_pending = cfg.target.auto_start;
        let config_snapshot = cfg.clone();
        Self {
            cfg,
            applied: config_snapshot.clone(),
            config_snapshot,
            config_edit_at: std::time::Instant::now(),
            windows,
            selected,
            monitors,
            selected_monitor,
            last_foreground,
            pipe,
            hud,
            shared,
            auto_start_pending,
            quitting: false,
            hidden: false,
            saw_minimized: false,
            notice: String::new(),
            ui_frames: 0,
            ui_last_report: std::time::Instant::now(),
        }
    }

    fn refresh_windows(&mut self) {
        let previous = self
            .selected
            .and_then(|i| self.windows.get(i))
            .map(|w| w.hwnd);
        self.windows = enumerate_windows();
        self.selected = previous
            .and_then(|hwnd| self.windows.iter().position(|w| w.hwnd == hwnd))
            .or_else(|| locate_window(&self.windows, self.cfg.target.window.as_ref()));
    }

    fn refresh_monitors(&mut self) {
        let previous = self
            .selected_monitor
            .and_then(|i| self.monitors.get(i))
            .map(|m| m.name.clone());
        self.monitors = enumerate_monitors();
        self.selected_monitor = previous
            .and_then(|name| self.monitors.iter().position(|m| m.name == name))
            .or_else(|| locate_monitor(&self.monitors, self.cfg.target.monitor.as_deref()));
    }

    fn refresh_foreground(&mut self) {
        if let Some(hwnd) =
            external_foreground_window().or_else(fluid_capture::last_external_foreground_window)
        {
            let hwnd_value = hwnd.0 as isize;
            if self.last_foreground.as_ref().map(|w| w.hwnd) != Some(hwnd_value) {
                if let Some(info) = window_info(hwnd) {
                    if !self.windows.iter().any(|w| w.hwnd == info.hwnd) {
                        self.windows.push(info.clone());
                    }
                    self.last_foreground = Some(info);
                }
            }
        } else if let Some(info) = &self.last_foreground {
            if !is_window(hwnd_from_isize(info.hwnd)) {
                self.last_foreground = None;
            }
        }
    }

    fn remember_last_foreground_for_window_mode(&mut self) {
        let Some(info) = self.last_foreground.clone() else {
            self.notice = "尚未捕获到外部前台窗口，请先切换到游戏/视频窗口。".into();
            return;
        };
        if !self.windows.iter().any(|w| w.hwnd == info.hwnd) {
            self.windows.push(info.clone());
        }
        self.selected = self.windows.iter().position(|w| w.hwnd == info.hwnd);
        self.cfg.target.mode = CaptureMode::Window;
    }

    fn build_request(&mut self) -> Option<CaptureRequest> {
        match self.cfg.target.mode {
            CaptureMode::Monitor => {
                let index = self.selected_monitor?;
                let monitor = self.monitors.get(index)?;
                self.cfg.target.monitor = Some(monitor.name.clone());
                Some(CaptureRequest::Monitor {
                    hmonitor: monitor.hmonitor,
                })
            }
            CaptureMode::Window => {
                let index = self.selected?;
                let window = self.windows.get(index)?;
                self.cfg.target.window = Some(WindowRef {
                    title: window.title.clone(),
                    class: window.class.clone(),
                });
                Some(CaptureRequest::Window { hwnd: window.hwnd })
            }
            CaptureMode::ForegroundWindow => {
                let index = self.selected_monitor?;
                let monitor = self.monitors.get(index)?;
                self.cfg.target.monitor = Some(monitor.name.clone());
                Some(CaptureRequest::Foreground {
                    fallback_monitor: monitor.hmonitor,
                })
            }
        }
    }

    fn start_capture(&mut self) -> bool {
        let Some(request) = self.build_request() else {
            self.notice = "没有可用的捕获目标，请选择显示器或窗口。".into();
            return false;
        };
        self.flush_config();
        self.notice.clear();
        self.pipe.send(Command::Start { request });
        true
    }

    fn stop_capture(&mut self) {
        self.pipe.send(Command::Stop);
    }

    fn minimize_for_capture(&mut self, ctx: &egui::Context) {
        // The capture overlay is topmost. Keeping a visible settings window
        // underneath it is confusing and makes the app look black/frozen.
        ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
        self.hidden = true;
        self.saw_minimized = false;
    }

    fn request_quit(&mut self, ctx: &egui::Context) {
        self.quitting = true;
        self.flush_config();
        self.stop_capture();
        self.hud.shutdown();
        crate::tray::shutdown();
        crate::hotkeys::shutdown();
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
    }

    fn publish_hotkey_plan(&self) {
        let fallback_monitor = self
            .selected_monitor
            .and_then(|index| self.monitors.get(index))
            .map(|monitor| monitor.hmonitor);
        let plan = match self.cfg.target.mode {
            CaptureMode::Monitor => {
                fallback_monitor.map(|hmonitor| crate::hotkeys::HotkeyPlan::Monitor { hmonitor })
            }
            CaptureMode::Window => self
                .selected
                .and_then(|index| self.windows.get(index))
                .map(|window| crate::hotkeys::HotkeyPlan::Window { hwnd: window.hwnd })
                .or(fallback_monitor.map(|fallback_monitor| {
                    crate::hotkeys::HotkeyPlan::Foreground { fallback_monitor }
                })),
            CaptureMode::ForegroundWindow => fallback_monitor.map(|fallback_monitor| {
                crate::hotkeys::HotkeyPlan::Foreground { fallback_monitor }
            }),
        };
        crate::hotkeys::set_plan(plan);
    }

    fn commit_config(&mut self) {
        if let Err(e) = self.cfg.save() {
            tracing::warn!("failed to save config: {e}");
        }
        self.pipe.send(Command::Config(self.cfg.clone()));
        self.hud.update_config(&self.cfg);
        self.applied = self.cfg.clone();
        self.config_snapshot = self.cfg.clone();
        self.config_edit_at = std::time::Instant::now();
    }

    /// Coalesce slider drags and rapid clicks into one save/config update.
    fn apply_config_debounced(&mut self) {
        if self.cfg != self.config_snapshot {
            self.config_snapshot = self.cfg.clone();
            self.config_edit_at = std::time::Instant::now();
        }
        if self.cfg != self.applied && self.config_edit_at.elapsed() >= Duration::from_millis(250) {
            self.commit_config();
        }
    }

    fn flush_config(&mut self) {
        if self.cfg != self.applied {
            self.commit_config();
        }
    }
}

fn locate_window(windows: &[WinEnum], target: Option<&WindowRef>) -> Option<usize> {
    let target = target?;
    windows.iter().position(|window| {
        (target.title.is_empty() || window.title == target.title)
            && (target.class.is_empty() || window.class == target.class)
    })
}

fn locate_monitor(monitors: &[MonitorInfo], target: Option<&str>) -> Option<usize> {
    if monitors.is_empty() {
        return None;
    }
    target
        .and_then(|name| monitors.iter().position(|monitor| monitor.name == name))
        .or(Some(0))
}

fn window_label(window: &WinEnum) -> String {
    let title = if window.title.is_empty() {
        format!("[{}]", window.class)
    } else {
        window.title.clone()
    };
    format!(
        "{}  ·  {}x{}  ·  {}",
        title,
        window.width(),
        window.height(),
        window.class
    )
}

fn monitor_label(monitor: &MonitorInfo) -> String {
    let width = monitor.rect.2 - monitor.rect.0;
    let height = monitor.rect.3 - monitor.rect.1;
    format!(
        "{}  ·  {}x{} @ {:.0}Hz",
        monitor.name, width, height, monitor.refresh_hz
    )
}

impl eframe::App for FluidApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.shared.ui_heartbeat.fetch_add(1, Ordering::Relaxed);
        self.refresh_foreground();
        crate::hotkeys::set_enabled(self.cfg.display.global_hotkeys);
        self.publish_hotkey_plan();
        if self
            .shared
            .minimize_ui_requested
            .swap(false, Ordering::Relaxed)
        {
            self.minimize_for_capture(ctx);
        }
        let minimized = ctx.input(|input| input.viewport().minimized);
        if minimized == Some(true) {
            self.saw_minimized = true;
        }
        if minimized == Some(false) {
            if self.saw_minimized
                && matches!(
                    *self.shared.state.lock(),
                    RunState::Starting | RunState::Running | RunState::Paused
                )
            {
                // Pause only the topmost capture overlay. The capture session,
                // HUD and pipeline stay alive, and the overlay resumes as soon as
                // the settings window is minimized or another window gets focus.
                self.pipe.send(Command::SetPaused(true));
                self.notice = "设置窗口已置顶：捕获画面暂停，设置窗口失焦后自动恢复。".into();
            }
            self.saw_minimized = false;
            self.hidden = false;
        }
        self.ui_frames += 1;
        if self.ui_last_report.elapsed() >= std::time::Duration::from_secs(2) {
            let elapsed = self.ui_last_report.elapsed().as_secs_f64();
            tracing::debug!(
                "ui update rate: {:.1} fps ({} frames)",
                self.ui_frames as f64 / elapsed,
                self.ui_frames
            );
            self.ui_frames = 0;
            self.ui_last_report = std::time::Instant::now();
        }
        let state = self.shared.state.lock().clone();
        let msg = self.shared.message.lock().clone();
        // The X button exits completely by default. Users who prefer a tray
        // resident tool can enable close_to_tray in settings.
        if ctx.input(|i| i.viewport().close_requested()) {
            if self.cfg.display.close_to_tray && !self.quitting {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
                self.hidden = true;
                self.saw_minimized = false;
                self.notice = "已隐藏到托盘；点击托盘图标 →「打开设置」可恢复。".into();
            } else {
                self.quitting = true;
                self.flush_config();
                self.stop_capture();
                crate::tray::shutdown();
                crate::hotkeys::shutdown();
                return;
            }
        }

        for id in crate::tray::take_events() {
            match id.as_str() {
                crate::tray::ID_OPEN => {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(false));
                    ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
                    ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
                    self.hidden = false;
                    self.saw_minimized = false;
                }
                crate::tray::ID_QUIT => {
                    self.request_quit(ctx);
                    return;
                }
                _ => {}
            }
        }

        if self.quitting {
            return;
        }

        if self.auto_start_pending {
            self.auto_start_pending = false;
            if matches!(state, RunState::Idle | RunState::Failed) && self.start_capture() {
                self.minimize_for_capture(ctx);
            }
        }

        let mut start_clicked = false;
        let mut stop_clicked = false;
        let mut hide_clicked = false;
        let mut quit_clicked = false;

        egui::TopBottomPanel::top("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let (label, color) = match state {
                    RunState::Idle => ("待机", egui::Color32::GRAY),
                    RunState::Starting => ("启动中", egui::Color32::YELLOW),
                    RunState::Running => ("运行中", egui::Color32::LIGHT_GREEN),
                    RunState::Paused => ("已暂停", egui::Color32::YELLOW),
                    RunState::Failed => ("失败", egui::Color32::RED),
                };
                ui.colored_label(color, format!("● {label}"));
                if !msg.is_empty() {
                    ui.weak(&msg);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("完全退出").clicked() {
                        quit_clicked = true;
                    }
                    if ui.button("隐藏").clicked() {
                        hide_clicked = true;
                    }
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.heading("fluid — 帧生成 + 画质提升");
                    ui.weak(format!(
                        "v{} · 纯外部采集 · 零注入 · 轻量 GPU 管线",
                        env!("CARGO_PKG_VERSION")
                    ));
                    if !self.notice.is_empty() {
                        ui.colored_label(egui::Color32::YELLOW, &self.notice);
                    }
                    ui.separator();

                    ui.group(|ui| {
                        ui.strong("捕获来源");
                        ui.radio_value(
                            &mut self.cfg.target.mode,
                            CaptureMode::ForegroundWindow,
                            "最近前台窗口（推荐：低占用，无需找窗口）",
                        );
                        ui.radio_value(
                            &mut self.cfg.target.mode,
                            CaptureMode::Monitor,
                            "整屏显示器（WGC 监视器捕获，适合全屏无边框游戏/视频）",
                        );
                        ui.radio_value(
                            &mut self.cfg.target.mode,
                            CaptureMode::Window,
                            "指定窗口（高级；适合窗口化游戏/视频）",
                        );

                        match self.cfg.target.mode {
                            CaptureMode::Monitor | CaptureMode::ForegroundWindow => {
                                ui.horizontal(|ui| {
                                    ui.label("显示器:");
                                    let text = self
                                        .selected_monitor
                                        .and_then(|i| self.monitors.get(i))
                                        .map(monitor_label)
                                        .unwrap_or_else(|| "未找到显示器".into());
                                    egui::ComboBox::from_id_salt("monitor_select")
                                        .selected_text(text)
                                        .width(380.0)
                                        .show_ui(ui, |ui| {
                                            for (i, monitor) in self.monitors.iter().enumerate() {
                                                ui.selectable_value(
                                                    &mut self.selected_monitor,
                                                    Some(i),
                                                    monitor_label(monitor),
                                                );
                                            }
                                        });
                                    if ui.button("刷新显示器").clicked() {
                                        self.refresh_monitors();
                                    }
                                });
                            }
                            CaptureMode::Window => {
                                ui.horizontal(|ui| {
                                    ui.label("窗口:");
                                    let text = self
                                        .selected
                                        .and_then(|i| self.windows.get(i))
                                        .map(window_label)
                                        .unwrap_or_else(|| "未选择".into());
                                    egui::ComboBox::from_id_salt("target_window")
                                        .selected_text(text)
                                        .width(430.0)
                                        .show_ui(ui, |ui| {
                                            for (i, window) in self.windows.iter().enumerate() {
                                                ui.selectable_value(
                                                    &mut self.selected,
                                                    Some(i),
                                                    window_label(window),
                                                );
                                            }
                                        });
                                    if ui.button("刷新窗口").clicked() {
                                        self.refresh_windows();
                                    }
                                });
                                ui.horizontal(|ui| {
                                    if ui.button("使用最近前台窗口").clicked() {
                                        self.remember_last_foreground_for_window_mode();
                                    }
                                    if let Some(info) = &self.last_foreground {
                                        ui.weak(format!("最近前台: {}", window_label(info)));
                                    } else {
                                        ui.weak("尚未记录外部前台窗口");
                                    }
                                });
                            }
                        }

                        if self.cfg.target.mode == CaptureMode::ForegroundWindow {
                            ui.colored_label(
                                egui::Color32::LIGHT_GREEN,
                                "开始后会持续跟随当前前台窗口；切换窗口会自动重定向。",
                            );
                            ui.colored_label(
                                egui::Color32::YELLOW,
                                "独占全屏游戏无法在画面上方显示外部覆盖层/HUD；请改用无边框窗口化或窗口模式。",
                            );
                            if let Some(info) = &self.last_foreground {
                                ui.weak(format!("最近记录: {}", window_label(info)));
                            } else {
                                ui.weak("暂时没有前台窗口记录，将先使用所选显示器作为兜底。");
                            }
                        }
                        ui.checkbox(&mut self.cfg.target.auto_start, "应用启动时自动开始");
                    });

                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("快捷预设:");
                        if ui.button("低功耗").clicked() {
                            self.cfg.scale.mode = ScaleMode::Bilinear;
                            self.cfg.scale.sharpness = 5;
                            self.cfg.fg.enabled = false;
                        }
                        if ui.button("均衡").clicked() {
                            self.cfg.scale.mode = ScaleMode::Adaptive;
                            self.cfg.scale.sharpness = 25;
                            self.cfg.fg.enabled = true;
                            self.cfg.fg.motion_compensation = true;
                            self.cfg.fg.strength = 80;
                            self.cfg.fg.scene_protection = 60;
                        }
                        if ui.button("画质优先").clicked() {
                            self.cfg.scale.mode = ScaleMode::Adaptive;
                            self.cfg.scale.sharpness = 48;
                            self.cfg.fg.enabled = true;
                            self.cfg.fg.motion_compensation = true;
                            self.cfg.fg.strength = 92;
                            self.cfg.fg.scene_protection = 55;
                        }
                    });

                    ui.group(|ui| {
                        ui.strong("画质与输出");
                        ui.horizontal(|ui| {
                            ui.label("缩放:");
                            egui::ComboBox::from_id_salt("scale_mode")
                                .selected_text(match self.cfg.scale.mode {
                                    ScaleMode::Adaptive => "自适应边缘重建 + 锐化",
                                    ScaleMode::Bilinear => "双线性（低功耗）",
                                    ScaleMode::Passthrough => "像素/直通",
                                })
                                .width(280.0)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut self.cfg.scale.mode,
                                        ScaleMode::Adaptive,
                                        "自适应边缘重建 + 锐化（推荐）",
                                    );
                                    ui.selectable_value(
                                        &mut self.cfg.scale.mode,
                                        ScaleMode::Bilinear,
                                        "双线性（低功耗）",
                                    );
                                    ui.selectable_value(
                                        &mut self.cfg.scale.mode,
                                        ScaleMode::Passthrough,
                                        "像素/直通",
                                    );
                                });
                        });
                        ui.add(
                            egui::Slider::new(&mut self.cfg.scale.sharpness, 0..=100)
                                .text("锐化强度"),
                        );
                        ui.checkbox(
                            &mut self.cfg.scale.preserve_aspect,
                            "保持宽高比（必要时加黑边）",
                        );

                        let can_output_scale = matches!(
                            self.cfg.target.mode,
                            CaptureMode::Window | CaptureMode::ForegroundWindow
                        );
                        ui.add_enabled_ui(can_output_scale, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("窗口输出:");
                                egui::ComboBox::from_id_salt("output_mode")
                                    .selected_text(match self.cfg.scale.output {
                                        OutputMode::MatchWindow => "与目标窗口同尺寸",
                                        OutputMode::Monitor => "铺满所在显示器（外置放大）",
                                    })
                                    .width(300.0)
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut self.cfg.scale.output,
                                            OutputMode::MatchWindow,
                                            "与目标窗口同尺寸",
                                        );
                                        ui.selectable_value(
                                            &mut self.cfg.scale.output,
                                            OutputMode::Monitor,
                                            "铺满所在显示器（外置放大）",
                                        );
                                    });
                            });
                        });
                        if !can_output_scale {
                            ui.weak("整屏捕获时，输出固定为所选显示器全屏。");
                        } else if matches!(self.cfg.scale.output, OutputMode::Monitor) {
                            ui.weak("会把目标窗口画面放大到整块显示器；适合低分辨率无边框游戏。");
                        }
                        let (sw, sh) = self.shared.stats.source_size();
                        let (ow, oh) = self.shared.stats.output_size();
                        if sw > 0
                            && sh > 0
                            && sw.abs_diff(ow) <= 1
                            && sh.abs_diff(oh) <= 1
                        {
                            ui.colored_label(
                                egui::Color32::YELLOW,
                                "当前源与输出尺寸相同：不会发生分辨率放大，只会进行锐化/抗锯齿。",
                            );
                        }
                    });

                    ui.group(|ui| {
                        ui.strong("帧生成");
                        ui.checkbox(&mut self.cfg.fg.enabled, "启用运动补偿插帧");
                        ui.add_enabled(
                            self.cfg.fg.enabled,
                            egui::Checkbox::new(
                                &mut self.cfg.fg.motion_compensation,
                                "运动补偿（关闭则退化为双向混合）",
                            ),
                        );
                        ui.add_enabled(
                            self.cfg.fg.enabled,
                            egui::Slider::new(&mut self.cfg.fg.strength, 0..=100).text("插帧强度"),
                        );
                        ui.add_enabled(
                            self.cfg.fg.enabled,
                            egui::Slider::new(&mut self.cfg.fg.scene_protection, 0..=100)
                                .text("场景切换保护"),
                        );
                        ui.add_enabled(
                            self.cfg.fg.enabled,
                            egui::Slider::new(&mut self.cfg.fg.max_multiplier, 2..=8)
                                .text("最高插帧倍率（相对源帧率）"),
                        );
                        ui.weak("只有当源帧率明显低于显示器刷新率时才会真正插值。");
                        let refresh = self.shared.stats.refresh_hz_x100.load(Ordering::Relaxed) as f64 / 100.0;
                        let capture_fps = self.shared.stats.capture_fps();
                        if refresh > 1.0 {
                            let theoretical = if capture_fps > 0.5 {
                                (refresh / capture_fps).max(1.0)
                            } else {
                                1.0
                            };
                            ui.weak(format!(
                                "显示器 {refresh:.0} Hz，当前源约 {capture_fps:.1} fps：理论上限约 {theoretical:.1}x。"
                            ));
                            if self.cfg.fg.max_multiplier as f64 > theoretical + 0.15 {
                                ui.colored_label(
                                    egui::Color32::YELLOW,
                                    "所选倍率超过刷新率允许值，超出的部分不会产生真实新帧。",
                                );
                            }
                        }
                        ui.weak(
                            "提高强度会更积极采用插值；低置信度、遮挡和场景切换像素始终回退真实当前帧。",
                        );
                        ui.weak("纯外部插帧需要等待前后两帧；倍率越高，为凑齐相位所需的回看延迟越大。");
                        ui.weak("实际倍率看 HUD 的 target / effective 值（target 已按刷新率封顶）。");
                    });

                    ui.group(|ui| {
                        ui.strong("呈现与窗口");
                        ui.radio_value(
                            &mut self.cfg.display.present_mode,
                            PresentMode::LowLatency,
                            "低延迟（推荐）",
                        );
                        ui.radio_value(
                            &mut self.cfg.display.present_mode,
                            PresentMode::Vsync,
                            "锁定刷新率（更稳，但延迟更高）",
                        );
                        let mut always_minimize = true;
                        ui.add_enabled(
                            false,
                            egui::Checkbox::new(
                                &mut always_minimize,
                                "捕获运行时自动最小化设置窗口（避免置顶覆盖层挡住界面）",
                            ),
                        );
                        ui.checkbox(
                            &mut self.cfg.display.close_to_tray,
                            "关闭设置窗口时最小化到托盘（默认关闭 = 直接退出）",
                        );
                        ui.checkbox(
                            &mut self.cfg.display.global_hotkeys,
                            "启用全局热键 F9 开始/停止、F10 显示/隐藏 HUD、F8 返回设置",
                        );
                    });

                    ui.group(|ui| {
                        ui.strong("实时监控 HUD");
                        ui.checkbox(
                            &mut self.cfg.display.show_hud,
                            "启用独立的点击穿透实时 HUD 小窗（不拦截鼠标/键盘）",
                        );
                        let hud_enabled = self.cfg.display.show_hud;
                        ui.add_enabled_ui(hud_enabled, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("位置:");
                                egui::ComboBox::from_id_salt("hud_anchor")
                                    .selected_text(match self.cfg.hud.anchor {
                                        HudAnchor::TopLeft => "左上",
                                        HudAnchor::TopRight => "右上",
                                        HudAnchor::BottomLeft => "左下",
                                        HudAnchor::BottomRight => "右下",
                                        HudAnchor::Custom => "自定义",
                                    })
                                    .width(160.0)
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut self.cfg.hud.anchor,
                                            HudAnchor::TopLeft,
                                            "左上",
                                        );
                                        ui.selectable_value(
                                            &mut self.cfg.hud.anchor,
                                            HudAnchor::TopRight,
                                            "右上",
                                        );
                                        ui.selectable_value(
                                            &mut self.cfg.hud.anchor,
                                            HudAnchor::BottomLeft,
                                            "左下",
                                        );
                                        ui.selectable_value(
                                            &mut self.cfg.hud.anchor,
                                            HudAnchor::BottomRight,
                                            "右下",
                                        );
                                        ui.selectable_value(
                                            &mut self.cfg.hud.anchor,
                                            HudAnchor::Custom,
                                            "自定义",
                                        );
                                    });
                            });
                            if self.cfg.hud.anchor == HudAnchor::Custom {
                                ui.add(
                                    egui::Slider::new(&mut self.cfg.hud.custom_x, 0..=100)
                                        .text("水平位置 %"),
                                );
                                ui.add(
                                    egui::Slider::new(&mut self.cfg.hud.custom_y, 0..=100)
                                        .text("垂直位置 %"),
                                );
                            }
                            ui.add(
                                egui::Slider::new(&mut self.cfg.hud.margin, 0..=80)
                                    .text("边距 px"),
                            );
                            ui.add(
                                egui::Slider::new(&mut self.cfg.hud.font_scale, 1..=3)
                                    .text("字体倍率"),
                            );
                            ui.horizontal(|ui| {
                                ui.label("背景颜色:");
                                ui.color_edit_button_srgb(&mut self.cfg.hud.background_color);
                                ui.add(
                                    egui::Slider::new(&mut self.cfg.hud.opacity, 10..=100)
                                        .text("HUD 不透明度 %"),
                                );
                            });
                            ui.horizontal_wrapped(|ui| {
                                ui.checkbox(&mut self.cfg.hud.show_source, "来源/尺寸");
                                ui.checkbox(&mut self.cfg.hud.show_performance, "实时性能");
                                ui.checkbox(&mut self.cfg.hud.show_quality, "画质参数");
                                ui.checkbox(&mut self.cfg.hud.show_framegen, "插帧参数");
                                ui.checkbox(&mut self.cfg.hud.show_system, "系统指标");
                            });
                            let preview_alpha =
                                (self.cfg.hud.opacity.clamp(10, 100) as u32 * 255 / 100) as u8;
                            let preview_color = egui::Color32::from_rgba_unmultiplied(
                                self.cfg.hud.background_color[0],
                                self.cfg.hud.background_color[1],
                                self.cfg.hud.background_color[2],
                                preview_alpha,
                            );
                            let (preview_rect, _) = ui.allocate_exact_size(
                                egui::vec2(ui.available_width().min(420.0), 56.0),
                                egui::Sense::hover(),
                            );
                            ui.painter().rect_filled(preview_rect, 4.0, preview_color);
                            ui.painter().text(
                                preview_rect.center(),
                                egui::Align2::CENTER_CENTER,
                                "HUD 背景 / 不透明度预览",
                                egui::FontId::proportional(16.0),
                                egui::Color32::WHITE,
                            );
                            ui.weak(
                                "HUD 使用独立小窗；背景颜色可任意调整，不透明度作用于整个 HUD 小窗（包括文字）。",
                            );
                            ui.weak(
                                "HUD 不参与捕获、插帧或缩放，停止捕获后自动隐藏。",
                            );
                        });
                    });

                    ui.separator();
                    ui.horizontal(|ui| {
                        let can_start = matches!(state, RunState::Idle | RunState::Failed);
                        if ui
                            .add_enabled(can_start, egui::Button::new("▶ 开始"))
                            .clicked()
                        {
                            start_clicked = true;
                        }
                        if ui
                            .add_enabled(
                                matches!(
                                    state,
                                    RunState::Running | RunState::Paused | RunState::Starting
                                ),
                                egui::Button::new("■ 停止"),
                            )
                            .clicked()
                        {
                            stop_clicked = true;
                        }
                        if ui.button("隐藏到托盘").clicked() {
                            hide_clicked = true;
                        }
                        if ui.button("完全退出").clicked() {
                            quit_clicked = true;
                        }
                    });

                    ui.separator();
                    ui.group(|ui| {
                        ui.strong("实时指标");
                        let stats = &self.shared.stats;
                        egui::Grid::new("stats")
                            .num_columns(2)
                            .spacing([18.0, 4.0])
                            .show(ui, |ui| {
                                let (source_w, source_h) = stats.source_size();
                                let (output_w, output_h) = stats.output_size();
                                let backend = match stats
                                    .capture_backend
                                    .load(std::sync::atomic::Ordering::Relaxed)
                                {
                                    0 => "Desktop Duplication",
                                    1 => "Windows Graphics Capture",
                                    _ => "unknown",
                                };
                                ui.label("采集来源");
                                ui.label(backend);
                                ui.end_row();
                                ui.label("源分辨率");
                                ui.label(format!("{source_w}x{source_h}"));
                                ui.end_row();
                                ui.label("输出分辨率");
                                ui.label(format!("{output_w}x{output_h}"));
                                ui.end_row();
                                ui.label("显示器刷新率");
                                ui.label(format!(
                                    "{:.1} Hz",
                                    stats
                                        .refresh_hz_x100
                                        .load(std::sync::atomic::Ordering::Relaxed)
                                        as f64
                                        / 100.0
                                ));
                                ui.end_row();
                                ui.label("采集 FPS");
                                ui.label(format!("{:.1}", stats.capture_fps()));
                                ui.end_row();
                                ui.label("呈现 FPS");
                                ui.label(format!("{:.1}", stats.present_fps()));
                                ui.end_row();
                                ui.label("GPU 时间戳");
                                ui.label(if stats.gpu_ms() > 0.001 {
                                    format!("{:.2} ms", stats.gpu_ms())
                                } else {
                                    "n/a（查询未就绪）".to_string()
                                });
                                ui.end_row();
                                ui.label("有效内容延迟");
                                ui.label(format!("{:.1} ms", stats.pipeline_latency_ms()));
                                ui.end_row();
                                ui.label("插帧状态");
                                ui.label(if stats.fg_active() {
                                    "运行中"
                                } else {
                                    "未启用 / 无空余刷新"
                                });
                                ui.end_row();
                                ui.label("生成帧率");
                                ui.label(format!("{:.1} fps", stats.generated_fps()));
                                ui.end_row();
                                ui.label("目标倍率 / 实际倍率");
                                let target = stats
                                    .fg_target_multiplier_x100
                                    .load(std::sync::atomic::Ordering::Relaxed)
                                    as f64
                                    / 100.0;
                                let actual = if stats.capture_fps() > 0.1 {
                                    stats.present_fps() / stats.capture_fps()
                                } else {
                                    1.0
                                };
                                ui.label(format!("{target:.1}x / {actual:.1}x"));
                                ui.end_row();
                                ui.label("已生成帧");
                                ui.label(format!(
                                    "{}",
                                    stats
                                        .generated_frames
                                        .load(std::sync::atomic::Ordering::Relaxed)
                                ));
                                ui.end_row();
                                ui.label("插帧相位");
                                ui.label(format!(
                                    "{:.2}",
                                    stats
                                        .fg_phase_x1000
                                        .load(std::sync::atomic::Ordering::Relaxed)
                                        as f64
                                        / 1000.0
                                ));
                                ui.end_row();
                                ui.label("本进程 CPU");
                                ui.label(format!("{:.1}%", stats.process_cpu_percent()));
                                ui.end_row();
                                ui.label("本进程内存");
                                ui.label(format!("{} MB", stats.working_set_mb()));
                                ui.end_row();
                                ui.label("总输出帧");
                                ui.label(format!(
                                    "{}",
                                    stats
                                        .frame_index
                                        .load(std::sync::atomic::Ordering::Relaxed)
                                ));
                                ui.end_row();
                                ui.label("积压丢弃帧");
                                ui.label(format!(
                                    "{}",
                                    stats
                                        .dropped_frames
                                        .load(std::sync::atomic::Ordering::Relaxed)
                                ));
                                ui.end_row();
                            });
                    });

                    ui.separator();
                    ui.weak("· 默认 X 关闭 = 完全退出；如启用托盘模式，请从托盘菜单完全退出。");
                    ui.weak("· 游戏请使用无边框窗口/窗口化；独占全屏无法被桌面复制捕获。");
                    ui.weak("· 整屏模式最稳定；窗口/遮挡场景建议让目标窗口保持无遮挡。");
                    ui.weak("· 鼠标点击会穿透覆盖层直达游戏；覆盖层自身不会出现在录屏/采集中。");
                    ui.weak(
                        "· 纯外部采集：不注入、不修改游戏/视频、不读取目标进程内存；\
                         反作弊兼容性请以具体游戏规则为准。",
                    );
                });
        });

        if start_clicked && self.start_capture() {
            self.minimize_for_capture(ctx);
        }
        if stop_clicked {
            self.stop_capture();
        }
        if hide_clicked {
            ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
            self.hidden = true;
            self.saw_minimized = false;
            self.notice = "已隐藏到托盘；点击托盘图标 →「打开设置」可恢复。".into();
        }
        if quit_clicked {
            self.request_quit(ctx);
            return;
        }

        self.apply_config_debounced();
        let interval = if self.hidden { 1500 } else { 500 };
        ctx.request_repaint_after(Duration::from_millis(interval));
    }
}

/// egui's default font set has no CJK glyphs. Load a system Chinese font.
fn install_cjk_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    for name in ["msyh.ttc", "msyh.ttf", "simhei.ttf", "simsun.ttc"] {
        let path = std::path::PathBuf::from("C:\\Windows\\Fonts").join(name);
        if let Ok(bytes) = std::fs::read(&path) {
            fonts
                .font_data
                .insert("cjk".into(), egui::FontData::from_owned(bytes));
            if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
                family.push("cjk".into());
            }
            tracing::info!("loaded CJK font: {}", path.display());
            break;
        }
    }
    ctx.set_fonts(fonts);
}
