//! Independent real-time HUD window.
//!
//! This runs on its own lightweight thread, uses its own tiny swapchain, and
//! never participates in the capture overlay or frame-generation pipeline.

use fluid_core::{AppConfig, HudAnchor, OutputMode, PresentMode, RunState, ScaleMode, SharedState};
use fluid_gfx::{create_device, DeviceBundle, HudRenderer, OverlayWindow};
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};
use windows::Win32::Foundation::RECT;

const HUD_REFRESH_MS: u64 = 250;
const POLL_MS: u64 = 80;

pub enum HudCommand {
    Config(AppConfig),
    Shutdown,
}

pub struct HudService {
    tx: mpsc::Sender<HudCommand>,
}

impl HudService {
    pub fn spawn(shared: Arc<SharedState>, initial_cfg: AppConfig) -> Self {
        let (tx, rx) = mpsc::channel();
        let shared_for_thread = shared.clone();
        if let Err(e) = std::thread::Builder::new()
            .name("fluid-hud".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    hud_thread(shared_for_thread, rx, initial_cfg)
                }));
                if result.is_err() {
                    tracing::error!("independent HUD thread panicked");
                }
            })
        {
            tracing::error!("failed to start independent HUD thread: {e}");
        }
        Self { tx }
    }

    pub fn update_config(&self, cfg: &AppConfig) {
        let _ = self.tx.send(HudCommand::Config(cfg.clone()));
    }

    pub fn shutdown(&self) {
        let _ = self.tx.send(HudCommand::Shutdown);
    }
}

fn hud_thread(shared: Arc<SharedState>, rx: mpsc::Receiver<HudCommand>, initial_cfg: AppConfig) {
    let mut cfg = initial_cfg;
    let mut resources: Option<(DeviceBundle, OverlayWindow, HudRenderer)> = None;
    let mut visible = false;
    let mut last_render = Instant::now() - Duration::from_millis(HUD_REFRESH_MS);
    let mut last_text = String::new();
    let mut last_style = (0u8, [0u8; 3], 0u8);
    let mut last_window: Option<RECT> = None;
    let mut last_window_alpha: Option<u8> = None;
    let mut last_hud_raise: Option<Instant> = None;
    let mut has_presented = false;
    let mut raise_burst_until: Option<Instant> = None;
    let mut last_raise_key: Option<(u32, isize, u32, u32)> = None;

    loop {
        fluid_gfx::pump_thread_messages();
        shared.hud_heartbeat.fetch_add(1, Ordering::Relaxed);
        match rx.recv_timeout(Duration::from_millis(POLL_MS)) {
            Ok(HudCommand::Config(new_cfg)) => {
                cfg = new_cfg;
            }
            Ok(HudCommand::Shutdown) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
        while let Ok(command) = rx.try_recv() {
            match command {
                HudCommand::Config(new_cfg) => cfg = new_cfg,
                HudCommand::Shutdown => return,
            }
        }

        let state = shared.state.lock().clone();
        let enabled = shared.hud_enabled.load(Ordering::Relaxed)
            && matches!(
                state,
                RunState::Running | RunState::Paused | RunState::Starting
            );
        if !enabled {
            if visible {
                if let Some((_, overlay, _)) = resources.as_mut() {
                    overlay.show(false);
                }
                visible = false;
            }
            // A hidden swapchain can retire its waitable in an unsignaled
            // state. The next shown frame must bypass the latency wait once.
            has_presented = false;
            last_hud_raise = None;
            raise_burst_until = None;
            last_raise_key = None;
            // Keep the HUD device/overlay alive for the process lifetime. Creating
            // and destroying a second hardware D3D/DComp device whenever the HUD
            // idles for a while has proven to deadlock Intel display drivers under
            // mixed OpenGL/D3D load.
            continue;
        }
        // The capture overlay is also topmost. WGC target switches call
        // SetWindowPos(HWND_TOPMOST) on the overlay, so the HUD must re-assert
        // itself. For DDA, use a lower keepalive rate to avoid needless desktop
        // updates/feedback while still recovering from any overlay raise race.
        let backend = shared.stats.capture_backend.load(Ordering::Relaxed);
        let source_id = shared.stats.source_id.load(Ordering::Relaxed);
        let (source_w, source_h) = shared.stats.source_size();
        let raise_key = (backend, source_id, source_w, source_h);
        let now = Instant::now();
        if last_raise_key != Some(raise_key) {
            last_raise_key = Some(raise_key);
            // A target/size switch is when the capture overlay is most likely to
            // be raised above the HUD. Re-assert quickly for a short burst, then
            // fall back to a low keepalive rate instead of churning the topmost
            // window z-order 12 times per second forever.
            raise_burst_until = Some(now + Duration::from_millis(900));
        }
        let raise_interval = if raise_burst_until.is_some_and(|until| now < until) {
            Duration::from_millis(80)
        } else {
            Duration::from_millis(1500)
        };
        if visible && last_hud_raise.is_none_or(|last| last.elapsed() >= raise_interval) {
            if let Some((_, overlay, _)) = resources.as_mut() {
                overlay.raise();
            }
            last_hud_raise = Some(Instant::now());
        }
        if last_render.elapsed() < Duration::from_millis(HUD_REFRESH_MS) {
            continue;
        }
        last_render = Instant::now();

        if resources.is_none() {
            match create_hud_resources() {
                Ok(value) => {
                    resources = Some(value);
                    last_window_alpha = None;
                }
                Err(e) => {
                    tracing::error!("failed to create independent HUD resources: {e}");
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }
            }
        }

        let text = build_hud_text(&cfg, &shared);
        let style = (
            cfg.hud.font_scale.clamp(1, 3),
            cfg.hud.background_color,
            cfg.hud.opacity.clamp(10, 100),
        );
        let (Some((_bundle, overlay, renderer)),) = (resources.as_mut(),) else {
            continue;
        };
        let (content_w, content_h) = renderer.update_text(&text, style.0, style.1);
        let global_alpha = (style.2 as u16 * 255 / 100) as u8;
        if last_window_alpha != Some(global_alpha) {
            overlay.set_alpha(global_alpha);
            last_window_alpha = Some(global_alpha);
        }
        if content_w == 0 || content_h == 0 {
            overlay.show(false);
            visible = false;
            has_presented = false;
            continue;
        }
        let monitor = active_monitor_rect(&shared);
        let (window_rect, draw_w, draw_h) = hud_window_rect(&cfg, monitor, content_w, content_h);
        let rect_changed = last_window != Some(window_rect);
        if rect_changed {
            if let Err(e) = overlay.place(&window_rect) {
                tracing::warn!("HUD window placement failed: {e}");
                overlay.show(false);
                visible = false;
                has_presented = false;
                continue;
            }
            last_window = Some(window_rect);
        }
        let Some(rtv) = overlay.rtv().cloned() else {
            overlay.show(false);
            visible = false;
            has_presented = false;
            continue;
        };
        // The first frame must be submitted after the window is visible. After
        // that, cap the frame-latency wait so a stuck DWM queue cannot block the
        // HUD thread (and its Win32 message pump) for hundreds of milliseconds.
        if has_presented && !overlay.wait_present_slot(8) {
            tracing::debug!("HUD present slot wait timed out; retrying next tick");
            continue;
        }
        if let Err(e) = renderer.draw(&rtv, draw_w, draw_h) {
            tracing::warn!("HUD draw failed: {e}");
            overlay.show(false);
            visible = false;
            has_presented = false;
            continue;
        }
        if !visible {
            overlay.show(true);
            visible = true;
        }
        overlay.present(0);
        has_presented = true;
        if rect_changed {
            overlay.raise();
            last_hud_raise = Some(Instant::now());
        }
        if text != last_text || style != last_style {
            last_text = text;
            last_style = style;
        }
    }

    if let Some((_, mut overlay, _)) = resources {
        overlay.show(false);
    }
}

fn create_hud_resources() -> Result<(DeviceBundle, OverlayWindow, HudRenderer), String> {
    let bundle = create_device().map_err(|e| format!("D3D11 device: {e}"))?;
    let overlay = OverlayWindow::new(&bundle.device, &bundle.factory2, 320, 200)
        .map_err(|e| format!("HUD overlay: {e}"))?;
    let renderer =
        HudRenderer::new(&bundle.device, &bundle.ctx).map_err(|e| format!("HUD renderer: {e}"))?;
    tracing::info!("independent real-time HUD initialized");
    Ok((bundle, overlay, renderer))
}

fn active_monitor_rect(shared: &SharedState) -> RECT {
    let left = shared.stats.hud_monitor_left.load(Ordering::Relaxed);
    let top = shared.stats.hud_monitor_top.load(Ordering::Relaxed);
    let right = shared.stats.hud_monitor_right.load(Ordering::Relaxed);
    let bottom = shared.stats.hud_monitor_bottom.load(Ordering::Relaxed);
    if right > left && bottom > top {
        return RECT {
            left,
            top,
            right,
            bottom,
        };
    }
    if let Some(monitor) = fluid_capture::enumerate_monitors().first() {
        let (l, t, r, b) = (
            monitor.rect.0,
            monitor.rect.1,
            monitor.rect.2,
            monitor.rect.3,
        );
        return RECT {
            left: l,
            top: t,
            right: r,
            bottom: b,
        };
    }
    RECT {
        left: 0,
        top: 0,
        right: 1920,
        bottom: 1080,
    }
}

fn hud_window_rect(
    cfg: &AppConfig,
    monitor: RECT,
    content_w: u32,
    content_h: u32,
) -> (RECT, u32, u32) {
    let monitor_w = (monitor.right - monitor.left).max(1);
    let monitor_h = (monitor.bottom - monitor.top).max(1);
    let margin = cfg.hud.margin as i32;
    let available_w = (monitor_w - margin * 2).max(1);
    let available_h = (monitor_h - margin * 2).max(1);
    let width = (content_w as i32).min(available_w).max(1);
    let height = (content_h as i32).min(available_h).max(1);
    let max_x = (monitor_w - width).max(0);
    let max_y = (monitor_h - height).max(0);
    let (x, y) = match cfg.hud.anchor {
        HudAnchor::TopLeft => (margin.min(max_x), margin.min(max_y)),
        HudAnchor::TopRight => ((max_x - margin).max(0), margin.min(max_y)),
        HudAnchor::BottomLeft => (margin.min(max_x), (max_y - margin).max(0)),
        HudAnchor::BottomRight => ((max_x - margin).max(0), (max_y - margin).max(0)),
        HudAnchor::Custom => (
            (cfg.hud.custom_x.min(100) as f32 / 100.0 * max_x as f32).round() as i32,
            (cfg.hud.custom_y.min(100) as f32 / 100.0 * max_y as f32).round() as i32,
        ),
    };
    (
        RECT {
            left: monitor.left + x,
            top: monitor.top + y,
            right: monitor.left + x + width,
            bottom: monitor.top + y + height,
        },
        width as u32,
        height as u32,
    )
}

fn build_hud_text(cfg: &AppConfig, shared: &SharedState) -> String {
    let stats = &shared.stats;
    let fg_state = if !cfg.fg.enabled {
        "off"
    } else if stats.fg_active() {
        "interpolating"
    } else {
        "idle"
    };
    let (source_w, source_h) = stats.source_size();
    let (output_w, output_h) = stats.output_size();
    let native_size = source_w.abs_diff(output_w) <= 1 && source_h.abs_diff(output_h) <= 1;
    let quality_label = if native_size {
        "native+sharpen"
    } else {
        match cfg.scale.mode {
            ScaleMode::Adaptive => "adaptive-upscale",
            ScaleMode::Bilinear => "bilinear-upscale",
            ScaleMode::Passthrough => "nearest",
        }
    };
    let output_label = match cfg.scale.output {
        OutputMode::MatchWindow => "match window",
        OutputMode::Monitor => "fill monitor",
    };
    let present_label = match cfg.display.present_mode {
        PresentMode::LowLatency => "low-latency",
        PresentMode::Vsync => "vsync",
    };
    let (source_w, source_h) = stats.source_size();
    let (output_w, output_h) = stats.output_size();
    let source_id = stats.source_id.load(Ordering::Relaxed);
    let backend = stats.capture_backend.load(Ordering::Relaxed);
    let source_label = match backend {
        1 => format!("WGC window 0x{source_id:x}"),
        2 => format!("WGC monitor 0x{source_id:x}"),
        _ => format!("DDA monitor 0x{source_id:x}"),
    };
    let refresh_hz = stats.refresh_hz_x100.load(Ordering::Relaxed) as f64 / 100.0;
    let generated_fps = stats.generated_fps();
    let capture_fps = stats.capture_fps();
    let present_fps = stats.present_fps();
    let effective_multiplier = if capture_fps > 0.1 {
        present_fps / capture_fps
    } else {
        1.0
    };
    let process_cpu = stats.process_cpu_percent();
    let working_set = stats.working_set_mb();
    let frame_index = stats.frame_index.load(Ordering::Relaxed);
    let gpu_ms = stats.gpu_ms();
    let gpu_label = if gpu_ms > 0.001 {
        format!("{gpu_ms:.2} ms")
    } else {
        "n/a".to_string()
    };

    let mut lines = vec![format!(
        "fluid v{}  realtime monitor",
        env!("CARGO_PKG_VERSION")
    )];
    if cfg.hud.show_source {
        lines.push(format!("capture: {source_label}"));
        lines.push(format!(
            "source: {source_w}x{source_h} @ {refresh_hz:.0} Hz"
        ));
        lines.push(format!("output: {output_w}x{output_h} | {output_label}"));
    }
    if cfg.hud.show_performance {
        lines.push(format!(
            "capture {:.1} fps | output {:.1} fps",
            stats.capture_fps(),
            stats.present_fps()
        ));
        lines.push(format!(
            "gpu {gpu_label} | latency {:.1} ms | drops {}",
            stats.pipeline_latency_ms(),
            stats.dropped_frames.load(Ordering::Relaxed)
        ));
        lines.push(format!(
            "generated {generated_fps:.1} fps | effective {effective_multiplier:.1}x | total {}",
            stats.generated_frames.load(Ordering::Relaxed)
        ));
    }
    if cfg.hud.show_quality {
        lines.push(format!(
            "scale: {quality_label} | sharpness: {} | aspect: {}",
            cfg.scale.sharpness.min(100),
            if cfg.scale.preserve_aspect {
                "on"
            } else {
                "off"
            }
        ));
        lines.push(format!("present: {present_label} | output: {output_label}"));
    }
    if cfg.hud.show_framegen {
        lines.push(format!(
            "frame gen: {} | state: {fg_state} | motion: {}",
            if cfg.fg.enabled { "on" } else { "off" },
            if stats.fg_perf_fallback.load(Ordering::Relaxed) != 0 {
                "perf-fallback"
            } else if cfg.fg.motion_compensation {
                "on"
            } else {
                "off"
            }
        ));
        lines.push(format!(
            "strength: {} | protection: {} | max: {}x",
            cfg.fg.strength.min(100),
            cfg.fg.scene_protection.min(100),
            cfg.fg.max_multiplier.clamp(2, 8)
        ));
        lines.push(format!(
            "phase: {:.2}",
            stats.fg_phase_x1000.load(Ordering::Relaxed) as f64 / 1000.0
        ));
    }
    if cfg.hud.show_system {
        lines.push(format!(
            "process CPU: {process_cpu:.1}% | RAM: {working_set} MB"
        ));
        lines.push(format!(
            "frame: {frame_index} | backend: {}",
            match backend {
                1 => "WGC-window",
                2 => "WGC-monitor",
                _ => "DDA-monitor",
            }
        ));
    }
    lines.join("\n")
}
