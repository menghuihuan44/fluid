//! Pipeline thread: capture -> frame ring -> frame generation / upscale ->
//! low-latency presentation. The UI thread only exchanges commands and atomic
//! statistics with this thread; no locks are taken on the hot path.

use fluid_capture::clock::{qpc_freq, qpc_now};
use fluid_capture::duplication::{CaptureError, Duplication};
use fluid_capture::monitor::{monitor_rect, monitor_refresh_hz};
use fluid_capture::wgc::WgcCapture;
use fluid_capture::window::{client_rect_on_screen, is_iconic, is_window, monitor_of, uv_rect};
use fluid_core::time::qpc_diff_us;
use fluid_core::{AppConfig, OutputMode, PresentMode, RunState, ScaleMode, SharedState, Stats};
use fluid_gfx::{
    create_device, DeviceBundle, OverlayWindow, RenderOptions, RenderRequest, RenderTargetInfo,
    Renderer,
};
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};
use tracing::{error, info, warn};
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Gdi::HMONITOR;

#[derive(Debug, Clone, Copy)]
pub enum CaptureRequest {
    /// Capture a complete monitor. This needs no window enumeration and is the
    /// most reliable mode for borderless fullscreen games and video players.
    Monitor { hmonitor: isize },
    /// Capture (and track) one window rectangle.
    Window { hwnd: isize },
}

pub enum Command {
    Start {
        request: CaptureRequest,
    },
    Stop,
    #[allow(dead_code)]
    SetPaused(bool),
    ToggleHud,
    Config(AppConfig),
}

pub struct Pipeline {
    tx: mpsc::Sender<Command>,
    #[allow(dead_code)]
    shared: Arc<SharedState>,
}

impl Pipeline {
    pub fn spawn(shared: Arc<SharedState>, init_cfg: AppConfig) -> Self {
        let (tx, rx) = mpsc::channel::<Command>();
        let shared2 = shared.clone();
        std::thread::Builder::new()
            .name("fluid-pipeline".into())
            .spawn(move || thread_main(shared2, rx, init_cfg))
            .expect("failed to start pipeline thread");
        Self { tx, shared }
    }

    pub fn send(&self, cmd: Command) {
        let _ = self.tx.send(cmd);
    }

    pub fn command_sender(&self) -> mpsc::Sender<Command> {
        self.tx.clone()
    }
}

enum TickErr {
    Fatal(String),
    Recoverable(String),
}

fn thread_main(shared: Arc<SharedState>, rx: mpsc::Receiver<Command>, init_cfg: AppConfig) {
    unsafe {
        let _ = windows::Win32::System::Com::CoInitializeEx(
            None,
            windows::Win32::System::Com::COINIT_MULTITHREADED,
        );
    }
    let thread_id = unsafe { windows::Win32::System::Threading::GetCurrentThreadId() };
    info!("pipeline thread started (id={thread_id})");
    let mut cfg = init_cfg;
    let mut session: Option<Session> = None;
    let mut paused = false;
    shared.set_state(RunState::Idle, "");

    loop {
        let cmd = if session.is_none() {
            rx.recv().ok()
        } else {
            rx.try_recv().ok()
        };
        match cmd {
            Some(Command::Stop) => {
                if session.is_some() {
                    info!("pipeline session stopped by command");
                }
                session = None;
                paused = false;
                shared.stats.fg_active.store(0, Ordering::Relaxed);
                shared.set_state(RunState::Idle, "");
                continue;
            }
            Some(Command::Config(c)) => {
                cfg = c;
                if let Some(s) = session.as_mut() {
                    s.mark_dirty();
                }
            }
            Some(Command::SetPaused(p)) => paused = p,
            Some(Command::ToggleHud) => {
                cfg.display.show_hud = !cfg.display.show_hud;
                if let Some(s) = session.as_mut() {
                    s.mark_dirty();
                }
            }
            Some(Command::Start { request }) => {
                shared.set_state(RunState::Starting, "");
                match Session::new(request, shared.clone()) {
                    Ok(s) => {
                        info!("pipeline session started: {request:?}");
                        session = Some(s);
                        paused = false;
                        shared.set_state(RunState::Running, "");
                    }
                    Err(e) => {
                        error!("session start failed: {e}");
                        shared.set_state(RunState::Failed, &e);
                    }
                }
                continue;
            }
            None => {}
        }

        let Some(s) = session.as_mut() else { continue };
        if paused {
            s.hide_overlay();
            shared.stats.fg_active.store(0, Ordering::Relaxed);
            shared.set_state(RunState::Paused, "");
            std::thread::sleep(Duration::from_millis(50));
            continue;
        }
        match s.tick(&cfg) {
            Ok(()) => shared.set_state(RunState::Running, ""),
            Err(TickErr::Fatal(m)) => {
                error!("pipeline terminated: {m}");
                session = None;
                shared.stats.fg_active.store(0, Ordering::Relaxed);
                shared.set_state(RunState::Failed, &m);
            }
            Err(TickErr::Recoverable(m)) => {
                warn!("recoverable pipeline error: {m}");
                shared.set_state(RunState::Running, &m);
                s.invalidate_capture();
                std::thread::sleep(Duration::from_millis(200));
            }
        }
    }
}

struct Session {
    request: CaptureRequest,
    shared: Arc<SharedState>,
    device_bundle: DeviceBundle,
    overlay: OverlayWindow,
    renderer: Renderer,
    dup: Option<Duplication>,
    wgc: Option<WgcCapture>,
    wgc_target_size: Option<(u32, u32)>,
    monitor: isize,
    next_tick: Option<Instant>,
    refresh_hz: f32,
    refresh_checked: Instant,
    last_capture_qpc: i64,
    capture_interval_us: f64,
    last_present_tick: Option<Instant>,
    last_output_rect: Option<(i32, i32, i32, i32)>,
    dirty: bool,
    has_frame: bool,
    last_present_qpc: i64,
    last_debug_log: Instant,
    last_debug_generated: u64,
    freq: u64,
}

impl Session {
    fn new(request: CaptureRequest, shared: Arc<SharedState>) -> Result<Self, String> {
        let bundle = create_device().map_err(|e| format!("D3D11 device creation failed: {e}"))?;
        if bundle.is_warp {
            return Err(
                "only the WARP software renderer is available; desktop duplication needs a real GPU. \
                 Enable hardware acceleration or run in a normal interactive desktop session."
                    .into(),
            );
        }
        let overlay = OverlayWindow::new(&bundle.device, &bundle.factory2, 64, 64)
            .map_err(|e| format!("overlay creation failed: {e}"))?;
        let renderer = Renderer::new(&bundle.device, &bundle.ctx, 2)
            .map_err(|e| format!("renderer initialization failed: {e}"))?;
        Ok(Self {
            request,
            shared,
            device_bundle: bundle,
            overlay,
            renderer,
            dup: None,
            wgc: None,
            wgc_target_size: None,
            monitor: -1,
            next_tick: None,
            refresh_hz: 60.0,
            refresh_checked: Instant::now() - Duration::from_secs(10),
            last_capture_qpc: 0,
            capture_interval_us: 1_000_000.0 / 60.0,
            last_present_tick: None,
            last_output_rect: None,
            dirty: true,
            has_frame: false,
            last_present_qpc: 0,
            last_debug_log: Instant::now(),
            last_debug_generated: 0,
            freq: qpc_freq(),
        })
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    fn hide_overlay(&mut self) {
        if self.overlay.is_visible() {
            self.overlay.show(false);
        }
    }

    fn invalidate_capture(&mut self) {
        self.dup = None;
        self.wgc = None;
        self.wgc_target_size = None;
        self.dirty = true;
        self.has_frame = false;
    }

    fn tick(&mut self, cfg: &AppConfig) -> Result<(), TickErr> {
        if self.last_debug_log.elapsed() >= Duration::from_secs(2) {
            let generated = self.shared.stats.generated_frames.load(Ordering::Relaxed);
            tracing::debug!(
                "pipeline stats: cap={:.1} out={:.1} gpu={:.2}ms fg={} generated_delta={} dirty={}",
                self.shared.stats.capture_fps(),
                self.shared.stats.present_fps(),
                self.shared.stats.gpu_ms(),
                self.shared.stats.fg_active(),
                generated.saturating_sub(self.last_debug_generated),
                self.dirty
            );
            self.last_debug_generated = generated;
            self.last_debug_log = Instant::now();
        }

        let (target_rect, source_monitor, use_wgc) = match self.request {
            CaptureRequest::Monitor { hmonitor } => {
                let hmon = HMONITOR(hmonitor as *mut core::ffi::c_void);
                let rect = monitor_rect(hmon).ok_or_else(|| {
                    TickErr::Fatal("selected monitor is no longer available".into())
                })?;
                (rect, hmonitor, false)
            }
            CaptureRequest::Window { hwnd } => {
                let hwnd = HWND(hwnd as *mut core::ffi::c_void);
                if !is_window(hwnd) {
                    return Err(TickErr::Fatal("target window was closed".into()));
                }
                if is_iconic(hwnd) {
                    self.hide_overlay();
                    return Ok(());
                }
                let Some(rect) = client_rect_on_screen(hwnd) else {
                    self.hide_overlay();
                    return Ok(());
                };
                let (w, h) = (rect.right - rect.left, rect.bottom - rect.top);
                if w < 8 || h < 8 {
                    self.hide_overlay();
                    return Ok(());
                }
                (rect, monitor_of(hwnd), true)
            }
        };

        if self.refresh_checked.elapsed() > Duration::from_secs(2) {
            self.refresh_hz =
                monitor_refresh_hz(HMONITOR(source_monitor as *mut core::ffi::c_void));
            self.refresh_checked = Instant::now();
        }

        if use_wgc {
            let hwnd = match self.request {
                CaptureRequest::Window { hwnd } => HWND(hwnd as *mut core::ffi::c_void),
                CaptureRequest::Monitor { .. } => {
                    return Err(TickErr::Fatal("invalid capture backend".into()));
                }
            };
            let target_w = (target_rect.right - target_rect.left).max(1) as u32;
            let target_h = (target_rect.bottom - target_rect.top).max(1) as u32;
            let need_recreate =
                self.wgc.is_none() || self.wgc_target_size != Some((target_w, target_h));
            if need_recreate {
                match WgcCapture::new(&self.device_bundle.device, hwnd) {
                    Ok(capture) => {
                        self.renderer
                            .ensure_bgra8_ring(capture.width(), capture.height())
                            .map_err(|e| {
                                TickErr::Recoverable(format!(
                                    "WGC frame ring allocation failed: {e}"
                                ))
                            })?;
                        self.wgc = Some(capture);
                        self.wgc_target_size = Some((target_w, target_h));
                        self.monitor = source_monitor;
                        self.dirty = true;
                        self.has_frame = false;
                    }
                    Err(e) => {
                        self.hide_overlay();
                        return Err(TickErr::Recoverable(format!(
                            "Windows Graphics Capture initialization failed: {e}. \
                             Use full-screen capture mode if the game cannot be captured as a window."
                        )));
                    }
                }
            }
        } else if self.dup.is_none() || source_monitor != self.monitor {
            match Duplication::new_for_monitor(
                &self.device_bundle.device,
                HMONITOR(source_monitor as *mut core::ffi::c_void),
            ) {
                Ok(d) => {
                    let (dw, dh) = (d.width(), d.height());
                    self.renderer.ensure_ring(dw, dh, d.format()).map_err(|e| {
                        TickErr::Recoverable(format!("frame ring allocation failed: {e}"))
                    })?;
                    self.dup = Some(d);
                    self.monitor = source_monitor;
                    self.dirty = true;
                    self.has_frame = false;
                }
                Err(e) => {
                    self.hide_overlay();
                    return Err(TickErr::Recoverable(format!(
                        "desktop duplication initialization failed: {e}"
                    )));
                }
            }
        }

        // Acquire at most one frame per tick. DDA coalesces updates; WGC
        // produces frames only when the captured window content changes.
        let mut got_new = false;
        let mut frame_qpc = 0i64;

        if use_wgc {
            let mut acquired_qpc = None;
            {
                let capture = self.wgc.as_mut().unwrap();
                match capture.acquire_latest() {
                    Ok(Some(frame)) => {
                        let qpc = frame.qpc;
                        self.renderer.push_frame(frame.texture, qpc);
                        acquired_qpc = Some(qpc);
                    }
                    Ok(None) => {}
                    Err(e) => {
                        self.wgc = None;
                        return Err(TickErr::Recoverable(format!("WGC capture failed: {e}")));
                    }
                }
            }
            if let Some(qpc) = acquired_qpc {
                self.wgc.as_mut().unwrap().release_frame();
                self.has_frame = true;
                frame_qpc = qpc;
                got_new = true;

                if self.last_capture_qpc != 0 {
                    let dt_us = qpc_diff_us(self.last_capture_qpc, qpc, self.freq);
                    if dt_us > 100 && dt_us < 100_000 {
                        Stats::ema_x100(
                            &self.shared.stats.capture_fps_x100,
                            1e6 / dt_us as f64,
                            0.08,
                        );
                        let measured = dt_us as f64;
                        self.capture_interval_us =
                            self.capture_interval_us * 0.85 + measured * 0.15;
                    }
                }
                self.last_capture_qpc = qpc;
            }
        } else {
            let dup = self.dup.as_mut().unwrap();
            match dup.acquire_latest() {
                Ok(Some(frame)) => {
                    let should_consume = frame.desktop_updated || !self.has_frame;
                    if should_consume {
                        let qpc = frame.qpc;
                        if frame.accumulated > 1 {
                            self.shared
                                .stats
                                .dropped_frames
                                .fetch_add(frame.accumulated - 1, Ordering::Relaxed);
                        }
                        if self.last_present_qpc > 0 && qpc > 0 {
                            let delta_us = qpc_diff_us(self.last_present_qpc, qpc, self.freq);
                            tracing::trace!(
                                "DDA content update: delta_from_last_present={:.2}ms accumulated={}",
                                delta_us as f64 / 1000.0,
                                frame.accumulated
                            );
                        }
                        self.renderer.push_frame(frame.texture, qpc);
                        self.has_frame = true;

                        if self.last_capture_qpc != 0 {
                            let dt_us = qpc_diff_us(self.last_capture_qpc, qpc, self.freq);
                            if dt_us > 100 && dt_us < 100_000 {
                                Stats::ema_x100(
                                    &self.shared.stats.capture_fps_x100,
                                    1e6 / dt_us as f64,
                                    0.08,
                                );
                                let measured = dt_us as f64;
                                self.capture_interval_us =
                                    self.capture_interval_us * 0.85 + measured * 0.15;
                            }
                        }
                        self.last_capture_qpc = qpc;
                        frame_qpc = qpc;
                        got_new = true;
                    }
                    dup.release_frame();
                }
                Ok(None) => {}
                Err(CaptureError::AccessLost) => {
                    self.dup = None;
                    return Err(TickErr::Recoverable(
                        "desktop duplication lost, rebuilding".into(),
                    ));
                }
                Err(e) => {
                    self.dup = None;
                    return Err(TickErr::Recoverable(format!("capture failed: {e}")));
                }
            }
        }

        let output_rect = match (&self.request, cfg.scale.output) {
            (CaptureRequest::Monitor { .. }, _) => target_rect,
            (_, OutputMode::Monitor) => {
                monitor_rect(HMONITOR(source_monitor as *mut core::ffi::c_void))
                    .unwrap_or(target_rect)
            }
            (_, OutputMode::MatchWindow) => target_rect,
        };
        let output_key = (
            output_rect.left,
            output_rect.top,
            output_rect.right,
            output_rect.bottom,
        );
        let rect_changed = self.last_output_rect != Some(output_key);

        self.overlay
            .place(&output_rect)
            .map_err(|e| TickErr::Recoverable(format!("overlay placement failed: {e}")))?;
        self.last_output_rect = Some(output_key);

        let refresh_interval_us = 1_000_000.0 / self.refresh_hz.max(24.0) as f64;
        let frame_period = Duration::from_secs_f64(1.0 / self.refresh_hz.max(24.0) as f64);

        // Always pace the loop before deciding whether to render. Without this,
        // skipping unchanged frames would spin the CPU at full speed.
        if let Some(next) = self.next_tick {
            let now = Instant::now();
            if now < next {
                std::thread::sleep((next - now).saturating_sub(Duration::from_millis(1)));
            }
        }

        let latest_now = self.renderer.latest_snapshot();
        let now_before = qpc_now();
        let expected_interval_ticks =
            (self.capture_interval_us * 1.0e-6 * self.freq as f64).max(1.0) as i64;
        let source_age_ticks = latest_now
            .as_ref()
            .filter(|frame| frame.qpc > 0)
            .map(|frame| now_before.saturating_sub(frame.qpc))
            .unwrap_or(0);
        let stall_limit_ticks = (expected_interval_ticks * 2).max((self.freq as f64 * 0.05) as i64);
        let source_stalled = source_age_ticks > stall_limit_ticks;
        let enough_frames = latest_now.as_ref().is_some_and(|frame| frame.seq >= 2);
        if latest_now.is_none() {
            // WGC can legitimately be silent until the target window presents a
            // new frame. Wait without tearing down the capture session.
            self.hide_overlay();
            self.next_tick = Some(Instant::now() + frame_period);
            return Ok(());
        }

        let capture_ratio = self.capture_interval_us / refresh_interval_us;
        let fg_eligible = cfg.fg.enabled
            && cfg.fg.strength > 0
            && cfg.fg.max_multiplier >= 2
            && capture_ratio >= 1.35
            && enough_frames
            && !source_stalled;
        self.shared
            .stats
            .fg_active
            .store(u32::from(fg_eligible), Ordering::Relaxed);

        // If nothing changed and no interpolated output is due, keep the last
        // presented swapchain image. This avoids pointless GPU work on a static
        // source and when interpolation is disabled.
        if !got_new && !fg_eligible && !self.dirty && !rect_changed {
            self.next_tick = Some(Instant::now() + frame_period);
            return Ok(());
        }

        // Generated frames are capped relative to source FPS. This avoids
        // spending 144 Hz worth of GPU work on a 45 FPS source unless the user
        // explicitly raises the multiplier.
        if fg_eligible && !got_new && !self.dirty && !rect_changed {
            let multiplier = cfg.fg.max_multiplier.clamp(2, 8) as f64;
            let desired_period =
                Duration::from_secs_f64((self.capture_interval_us * 1.0e-6) / multiplier);
            let minimum_period = frame_period.max(desired_period);
            let since_present = self
                .last_present_tick
                .map_or(minimum_period, |last| last.elapsed());
            if since_present < minimum_period.saturating_sub(Duration::from_millis(1)) {
                self.next_tick = Some(Instant::now() + frame_period);
                return Ok(());
            }
        }

        let rtv = match self.overlay.rtv() {
            Some(r) => r.clone(),
            None => return Err(TickErr::Recoverable("swapchain RTV unavailable".into())),
        };

        self.overlay.wait_present_slot(200);
        let t0 = Instant::now();
        let now_qpc = qpc_now();

        let latest =
            latest_now.ok_or_else(|| TickErr::Recoverable("no captured frame available".into()))?;

        let mut fg_phase = 1.0f32;
        let (previous, current) = if fg_eligible {
            let half_interval_ticks =
                (self.capture_interval_us * 1.0e-6 * self.freq as f64 * 0.5) as i64;
            let target_qpc = now_qpc.saturating_sub(half_interval_ticks);
            match self.renderer.pair_at_time(target_qpc) {
                Some((prev, curr, phase)) => {
                    fg_phase = phase;
                    (prev, curr)
                }
                None => (latest.clone(), latest),
            }
        } else {
            (latest.clone(), latest)
        };

        let previous = &previous;
        let current = &current;
        let interpolating =
            fg_eligible && previous.seq != current.seq && fg_phase > 0.001 && fg_phase < 0.999;
        if interpolating {
            self.shared
                .stats
                .generated_frames
                .fetch_add(1, Ordering::Relaxed);
        }
        let scale_mode = match cfg.scale.mode {
            ScaleMode::Passthrough => 0,
            ScaleMode::Bilinear => 1,
            ScaleMode::Adaptive => 2,
        };
        let options = RenderOptions {
            scale_mode,
            sharpness: cfg.scale.sharpness.min(100) as f32 / 100.0,
            fg_enabled: fg_eligible,
            motion_compensation: cfg.fg.motion_compensation,
            preserve_aspect: cfg.scale.preserve_aspect,
            fg_strength: cfg.fg.strength.min(100) as f32 / 100.0,
            scene_protection: cfg.fg.scene_protection.min(100) as f32 / 100.0,
            fg_phase,
        };

        let hud_text = if cfg.display.show_hud {
            let fg_label = if interpolating {
                "fg:interp"
            } else if fg_eligible {
                "fg:ready"
            } else if cfg.fg.enabled {
                "fg:idle"
            } else {
                "fg:off"
            };
            let source_label = match self.request {
                CaptureRequest::Monitor { .. } => "monitor",
                CaptureRequest::Window { .. } => "window",
            };
            Some(format!(
                "fluid v{ver} | {source_label} | cap {cap:.0} | out {pres:.0} fps | gpu {gpu:.2} ms | lat {lat:.1} ms | {fg_label}",
                ver = env!("CARGO_PKG_VERSION"),
                source_label = source_label,
                cap = self.shared.stats.capture_fps(),
                pres = self.shared.stats.present_fps(),
                gpu = self.shared.stats.gpu_ms(),
                lat = self.shared.stats.pipeline_latency_ms(),
                fg_label = fg_label,
            ))
        } else {
            None
        };

        let sync = match cfg.display.present_mode {
            PresentMode::LowLatency => 0,
            PresentMode::Vsync => 1,
        };
        let uv = if use_wgc {
            [0.0, 0.0, 1.0, 1.0]
        } else {
            let output_rect_desc = self.dup.as_ref().unwrap().output_rect();
            uv_rect(&output_rect_desc, &target_rect)
        };
        let (bbw, bbh) = self.overlay.size();
        self.renderer
            .render(RenderRequest {
                target: RenderTargetInfo {
                    rtv: &rtv,
                    width: bbw,
                    height: bbh,
                    uv_rect: uv,
                },
                previous,
                current,
                options,
                hud_text: hud_text.as_deref(),
            })
            .map_err(|e| TickErr::Recoverable(format!("render failed: {e}")))?;
        self.overlay.present(sync);
        self.last_present_qpc = qpc_now();
        self.dirty = false;

        self.next_tick = Some(t0 + frame_period);
        self.shared
            .stats
            .frame_index
            .fetch_add(1, Ordering::Relaxed);

        let now_instant = Instant::now();
        if let Some(last) = self.last_present_tick {
            let dt = now_instant.duration_since(last).as_secs_f64();
            if dt > 0.0005 {
                Stats::ema_x100(&self.shared.stats.present_fps_x100, 1.0 / dt, 0.1);
            }
        } else {
            self.shared.stats.present_fps_x100.store(
                (self.refresh_hz.max(24.0) * 100.0) as u32,
                Ordering::Relaxed,
            );
        }
        self.last_present_tick = Some(now_instant);
        Stats::ema_x100(
            &self.shared.stats.gpu_us,
            self.renderer.gpu_us() as f64,
            0.1,
        );

        let effective_qpc = if interpolating && current.qpc > previous.qpc {
            previous.qpc as f64 + (current.qpc - previous.qpc) as f64 * fg_phase as f64
        } else if got_new && frame_qpc != 0 {
            frame_qpc as f64
        } else {
            current.qpc as f64
        };
        if effective_qpc > 0.0 {
            let lat_us = ((now_qpc as f64 - effective_qpc) / self.freq as f64 * 1e6).max(0.0);
            Stats::ema_x100(&self.shared.stats.pipeline_latency_us, lat_us, 0.1);
        }
        Ok(())
    }
}
