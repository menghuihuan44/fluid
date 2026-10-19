//! Pipeline thread: capture -> frame ring -> frame generation / upscale ->
//! low-latency presentation. The UI thread only exchanges commands and atomic
//! statistics with this thread; no locks are taken on the hot path.

use fluid_capture::clock::{qpc_freq, qpc_now};
use fluid_capture::duplication::{CaptureError, Duplication};
use fluid_capture::monitor::{enumerate_monitors, monitor_rect, monitor_refresh_hz};
use fluid_capture::wgc::WgcCapture;
use fluid_capture::window::{
    client_rect_on_screen, is_foreground_capture_candidate, is_iconic, is_window, monitor_of,
    uv_rect,
};
use fluid_core::time::qpc_diff_us;
use fluid_core::{AppConfig, OutputMode, PresentMode, RunState, ScaleMode, SharedState, Stats};
use fluid_gfx::{
    create_device, DeviceBundle, HudOptions, OverlayWindow, RenderOptions, RenderRequest,
    RenderTargetInfo, Renderer,
};
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};
use tracing::{error, info, warn};
use windows::Win32::Foundation::{FILETIME, HWND};
use windows::Win32::Graphics::Gdi::HMONITOR;
use windows::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
use windows::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

const STAGE_TICK: u32 = 0;
const STAGE_SELECT: u32 = 1;
const STAGE_GRACE: u32 = 2;
const STAGE_WGC_DISCARD: u32 = 3;
const STAGE_WGC_NEW: u32 = 4;
const STAGE_RING: u32 = 5;
const STAGE_DDA_NEW: u32 = 6;
const STAGE_CAPTURE: u32 = 7;
const STAGE_OVERLAY_PLACE: u32 = 8;
const STAGE_PACE: u32 = 9;
const STAGE_RENDER: u32 = 10;
const STAGE_PRESENT: u32 = 11;
const STAGE_SUSPEND: u32 = 12;

pub(crate) fn stage_name(stage: u32) -> &'static str {
    match stage {
        STAGE_TICK => "tick-start",
        STAGE_SELECT => "select-target",
        STAGE_GRACE => "target-grace",
        STAGE_WGC_DISCARD => "wgc-discard",
        STAGE_WGC_NEW => "wgc-new",
        STAGE_RING => "ring-allocate",
        STAGE_DDA_NEW => "dda-new",
        STAGE_CAPTURE => "capture-acquire",
        STAGE_OVERLAY_PLACE => "overlay-place",
        STAGE_PACE => "frame-pacing",
        STAGE_RENDER => "renderer-render",
        STAGE_PRESENT => "overlay-present",
        STAGE_SUSPEND => "session-suspend",
        _ => "unknown",
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CaptureRequest {
    /// Capture a complete monitor. This needs no window enumeration and is the
    /// most reliable mode for borderless fullscreen games and video players.
    Monitor { hmonitor: isize },
    /// Capture (and track) one window rectangle.
    Window { hwnd: isize },
    /// Dynamically follow the current external foreground window with WGC.
    /// If no foreground window is available, use the fallback monitor with DDA
    /// until a window appears.
    Foreground { fallback_monitor: isize },
}

pub enum Command {
    Start { request: CaptureRequest },
    Stop,
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
        let shared_for_panic = shared.clone();
        std::thread::Builder::new()
            .name("fluid-pipeline".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    thread_main(shared2, rx, init_cfg)
                }));
                if result.is_err() {
                    shared_for_panic.set_state(
                        RunState::Failed,
                        "pipeline thread panicked; please restart capture",
                    );
                }
            })
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
    shared
        .hud_enabled
        .store(init_cfg.display.show_hud, Ordering::Relaxed);
    let mut cfg = init_cfg;
    let mut session: Option<Session> = None;
    let mut active = false;
    let mut paused = false;
    shared.set_state(RunState::Idle, "");

    loop {
        fluid_gfx::pump_thread_messages();
        let cmd = if !active {
            match rx.recv() {
                Ok(command) => Some(command),
                Err(_) => break,
            }
        } else {
            rx.try_recv().ok()
        };
        match cmd {
            Some(Command::Stop) => {
                if let Some(s) = session.as_mut() {
                    info!("pipeline session suspended by command");
                    s.suspend();
                }
                active = false;
                paused = false;
                shared.stats.fg_active.store(0, Ordering::Relaxed);
                shared
                    .stats
                    .fg_target_multiplier_x100
                    .store(0, Ordering::Relaxed);
                shared.set_state(RunState::Idle, "");
                continue;
            }
            Some(Command::Config(c)) => {
                shared
                    .hud_enabled
                    .store(c.display.show_hud, Ordering::Relaxed);
                cfg = c;
                if let Some(s) = session.as_mut() {
                    s.mark_dirty();
                }
            }
            Some(Command::SetPaused(p)) => {
                if paused != p {
                    paused = p;
                    if let Some(s) = session.as_mut() {
                        if p {
                            tracing::info!("capture overlay paused while settings is foreground");
                            s.suspend();
                        } else {
                            tracing::info!("capture overlay resumed after settings lost focus");
                            s.prepare_resume();
                        }
                    }
                }
            }
            Some(Command::ToggleHud) => {
                cfg.display.show_hud = !cfg.display.show_hud;
                shared
                    .hud_enabled
                    .store(cfg.display.show_hud, Ordering::Relaxed);
                if let Some(s) = session.as_mut() {
                    s.mark_dirty();
                }
            }
            Some(Command::Start { request }) => {
                shared.set_state(RunState::Starting, "");
                if let Some(s) = session.as_mut() {
                    info!("pipeline session resumed: {request:?}");
                    s.reconfigure(request);
                    active = true;
                    paused = false;
                    shared.set_state(RunState::Running, "");
                } else {
                    match Session::new(request, shared.clone()) {
                        Ok(s) => {
                            info!("pipeline session started: {request:?}");
                            session = Some(s);
                            active = true;
                            paused = false;
                            shared.set_state(RunState::Running, "");
                        }
                        Err(e) => {
                            error!("session start failed: {e}");
                            active = false;
                            shared.set_state(RunState::Failed, &e);
                        }
                    }
                }
                continue;
            }
            None => {}
        }

        let Some(s) = session.as_mut() else {
            active = false;
            continue;
        };
        if !active {
            continue;
        }
        if paused {
            s.hide_overlay();
            shared.stats.fg_active.store(0, Ordering::Relaxed);
            shared
                .stats
                .fg_target_multiplier_x100
                .store(0, Ordering::Relaxed);
            shared.set_state(RunState::Paused, "");
            std::thread::sleep(Duration::from_millis(50));
            continue;
        }
        match s.tick(&cfg) {
            Ok(()) => shared.set_state(RunState::Running, ""),
            Err(TickErr::Recoverable(m)) => {
                warn!("recoverable pipeline error: {m}");
                s.hide_overlay();
                shared.set_state(RunState::Running, &m);
                s.invalidate_capture();
                std::thread::sleep(Duration::from_millis(200));
            }
        }
    }
}

fn filetime_ticks(time: FILETIME) -> u64 {
    ((time.dwHighDateTime as u64) << 32) | time.dwLowDateTime as u64
}

fn sleep_until(deadline: Instant) {
    loop {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        let remaining = deadline - now;
        if remaining > Duration::from_millis(1) {
            std::thread::sleep(remaining - Duration::from_millis(1));
        } else {
            std::thread::yield_now();
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FgPacing {
    /// Number of output presents per source frame that the display and the
    /// user multiplier actually permit. It is <= the configured multiplier.
    effective_multiplier: f64,
    /// Time between generated/real output presents.
    output_period: Duration,
    /// How far behind real time the interpolation clock runs. For an exact
    /// N-times schedule this makes the N-1 interpolated slots land at phases
    /// 1/N .. (N-1)/N instead of repeatedly hitting an endpoint.
    lookback: Duration,
}

impl FgPacing {
    fn new(capture_interval_us: f64, refresh_hz: f32, requested_multiplier: u8) -> Self {
        let requested = requested_multiplier.clamp(2, 8) as f64;
        let capture_interval =
            Duration::from_secs_f64((capture_interval_us.max(1_000.0) * 1.0e-6).min(1.0));
        let display_period = Duration::from_secs_f64(1.0 / refresh_hz.clamp(24.0, 1_000.0) as f64);
        let requested_period = capture_interval.div_f64(requested);
        let output_period = display_period.max(requested_period);
        let effective_multiplier = (capture_interval.as_secs_f64()
            / output_period.as_secs_f64().max(1.0e-9))
        .clamp(1.0, requested);
        // At the effective multiplier M, delay the interpolation clock by
        // (M-1)/M of a source interval. A fresh capture then exposes phases
        // 1/M, 2/M, ... and uses every one of the M-1 interior slots.
        let lookback = capture_interval.mul_f64((1.0 - 1.0 / effective_multiplier).max(0.0));
        Self {
            effective_multiplier,
            output_period,
            lookback,
        }
    }

    fn allows_generation(self) -> bool {
        self.effective_multiplier >= 1.35
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
    wgc_hwnd: Option<isize>,
    wgc_monitor: Option<isize>,
    wgc_target_size: Option<(u32, u32)>,
    pending_target_size: Option<((u32, u32), Instant)>,
    current_window: Option<HWND>,
    pending_window: Option<(HWND, Instant)>,
    failed_window: Option<(HWND, Instant)>,
    fallback_grace_until: Option<Instant>,
    dda_retry_after: Option<Instant>,
    monitor_wgc_disabled: bool,
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
    last_metrics_sample: Instant,
    last_generated_count: u64,
    last_process_cpu_100ns: u64,
    fg_backoff_until: Option<Instant>,
    overload_count: u32,
    fg_underperf_count: u32,
    fg_low_gain_count: u32,
    /// Graduated performance fallback: 0 = full quality, 1 = sharpening off,
    /// 2 = sharpening + motion search off (bilinear scale).
    fg_perf_stage: u32,
    fg_perf_until: Option<Instant>,
    /// Last (prev seq, curr seq) pair that produced an interpolated present,
    /// and the phase it was shown with. Display time must never move backwards
    /// between presents of the same pair.
    last_fg_pair: Option<(u64, u64)>,
    last_fg_phase: f32,
    keep_overlay_below_taskbar: bool,
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
        let renderer = Renderer::new(&bundle.device, &bundle.ctx, 3)
            .map_err(|e| format!("renderer initialization failed: {e}"))?;
        Ok(Self {
            request,
            shared,
            device_bundle: bundle,
            overlay,
            renderer,
            dup: None,
            wgc: None,
            wgc_hwnd: None,
            wgc_monitor: None,
            wgc_target_size: None,
            pending_target_size: None,
            current_window: None,
            pending_window: None,
            failed_window: None,
            fallback_grace_until: None,
            dda_retry_after: None,
            monitor_wgc_disabled: false,
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
            last_metrics_sample: Instant::now(),
            last_generated_count: 0,
            last_process_cpu_100ns: 0,
            fg_backoff_until: None,
            overload_count: 0,
            fg_underperf_count: 0,
            fg_low_gain_count: 0,
            fg_perf_stage: 0,
            fg_perf_until: None,
            last_fg_pair: None,
            last_fg_phase: 0.0,
            keep_overlay_below_taskbar: false,
            freq: qpc_freq(),
        })
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    fn set_stage(&self, stage: u32) {
        self.shared.pipeline_stage.store(stage, Ordering::Relaxed);
    }

    fn suspend(&mut self) {
        self.set_stage(STAGE_SUSPEND);
        self.discard_wgc();
        self.dup = None;
        self.current_window = None;
        self.pending_window = None;
        self.pending_target_size = None;
        self.fg_backoff_until = None;
        self.overload_count = 0;
        self.fg_underperf_count = 0;
        self.fg_low_gain_count = 0;
        self.fg_perf_stage = 0;
        self.fg_perf_until = None;
        self.last_fg_pair = None;
        self.last_fg_phase = 0.0;
        self.has_frame = false;
        self.dirty = true;
        self.next_tick = None;
        self.hide_overlay();
    }

    fn prepare_resume(&mut self) {
        self.renderer.clear_frames();
        self.current_window = None;
        self.pending_window = None;
        self.failed_window = None;
        self.has_frame = false;
        self.dirty = true;
        self.next_tick = None;
        self.last_capture_qpc = 0;
        self.capture_interval_us = 1_000_000.0 / 60.0;
        self.last_present_qpc = 0;
        self.last_present_tick = None;
    }

    fn reconfigure(&mut self, request: CaptureRequest) {
        self.request = request;
        self.discard_wgc();
        self.dup = None;
        self.current_window = None;
        self.pending_window = None;
        self.pending_target_size = None;
        self.failed_window = None;
        self.fallback_grace_until = None;
        self.dda_retry_after = None;
        self.monitor_wgc_disabled = false;
        self.fg_backoff_until = None;
        self.overload_count = 0;
        self.fg_underperf_count = 0;
        self.fg_low_gain_count = 0;
        self.fg_perf_stage = 0;
        self.fg_perf_until = None;
        self.last_fg_pair = None;
        self.last_fg_phase = 0.0;
        self.monitor = -1;
        self.next_tick = None;
        self.last_capture_qpc = 0;
        self.capture_interval_us = 1_000_000.0 / 60.0;
        self.last_present_tick = None;
        self.last_present_qpc = 0;
        self.last_output_rect = None;
        self.has_frame = false;
        self.dirty = true;
        self.renderer.clear_frames();
        self.hide_overlay();
    }

    fn sample_system_metrics(&mut self) {
        let elapsed = self.last_metrics_sample.elapsed();
        if elapsed < Duration::from_secs(1) {
            return;
        }
        let generated = self.shared.stats.generated_frames.load(Ordering::Relaxed);
        let generated_delta = generated.saturating_sub(self.last_generated_count);
        Stats::ema_x100(
            &self.shared.stats.generated_fps_x100,
            generated_delta as f64 / elapsed.as_secs_f64(),
            0.5,
        );
        self.last_generated_count = generated;

        unsafe {
            let process = GetCurrentProcess();
            let mut creation = FILETIME::default();
            let mut exit = FILETIME::default();
            let mut kernel = FILETIME::default();
            let mut user = FILETIME::default();
            if GetProcessTimes(process, &mut creation, &mut exit, &mut kernel, &mut user).is_ok() {
                let total = filetime_ticks(kernel) + filetime_ticks(user);
                if self.last_process_cpu_100ns > 0 {
                    let cpu_delta = total.saturating_sub(self.last_process_cpu_100ns);
                    let elapsed_100ns = (elapsed.as_secs_f64() * 1.0e7) as u64;
                    if elapsed_100ns > 0 {
                        let cpu_percent = cpu_delta as f64 / elapsed_100ns as f64 * 100.0;
                        Stats::ema_x100(&self.shared.stats.process_cpu_x100, cpu_percent, 0.35);
                    }
                }
                self.last_process_cpu_100ns = total;
            }

            let mut counters = PROCESS_MEMORY_COUNTERS {
                cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
                ..Default::default()
            };
            if GetProcessMemoryInfo(
                process,
                &mut counters,
                std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            )
            .is_ok()
            {
                self.shared.stats.working_set_mb.store(
                    (counters.WorkingSetSize / (1024 * 1024)) as u32,
                    Ordering::Relaxed,
                );
            }
        }
        self.last_metrics_sample = Instant::now();
    }

    fn hide_overlay(&mut self) {
        if self.overlay.is_visible() {
            self.overlay.show(false);
        }
    }

    fn discard_wgc(&mut self) {
        self.set_stage(STAGE_WGC_DISCARD);
        self.wgc_hwnd = None;
        self.wgc_monitor = None;
        self.pending_target_size = None;
        self.wgc_target_size = None;
        if let Some(capture) = self.wgc.take() {
            capture.dispose_async();
        }
    }

    fn invalidate_capture(&mut self) {
        self.dup = None;
        self.discard_wgc();
        self.renderer.clear_frames();
        self.shared.stats.fg_active.store(0, Ordering::Relaxed);
        self.shared
            .stats
            .fg_target_multiplier_x100
            .store(0, Ordering::Relaxed);
        self.last_fg_pair = None;
        self.last_fg_phase = 0.0;
        self.dirty = true;
        self.has_frame = false;
    }

    fn reset_for_target_change(&mut self) {
        self.discard_wgc();
        self.renderer.clear_frames();
        self.shared.stats.fg_active.store(0, Ordering::Relaxed);
        self.shared
            .stats
            .fg_target_multiplier_x100
            .store(0, Ordering::Relaxed);
        self.last_fg_pair = None;
        self.last_fg_phase = 0.0;
        self.has_frame = false;
        self.dirty = true;
        self.last_capture_qpc = 0;
        self.capture_interval_us = 1_000_000.0 / 60.0;
        self.pending_window = None;
        self.hide_overlay();
    }

    fn wait_for_capture_target(&mut self) -> Result<(), TickErr> {
        self.invalidate_capture();
        self.hide_overlay();
        self.next_tick = None;
        std::thread::sleep(Duration::from_millis(100));
        Ok(())
    }

    fn window_blacklisted(&self, hwnd: HWND) -> bool {
        self.failed_window.is_some_and(|(failed, since)| {
            failed == hwnd && since.elapsed() < Duration::from_secs(15)
        })
    }

    fn mark_window_failed(&mut self, hwnd: HWND) {
        self.failed_window = Some((hwnd, Instant::now()));
    }

    fn mark_current_wgc_failed(&mut self) {
        if let Some(hwnd) = self.wgc_hwnd {
            self.mark_window_failed(HWND(hwnd as *mut core::ffi::c_void));
        }
        if self.wgc_monitor.is_some() {
            self.monitor_wgc_disabled = true;
        }
        self.fallback_grace_until = Some(Instant::now() + Duration::from_millis(650));
        if matches!(self.request, CaptureRequest::Foreground { .. }) {
            self.current_window = None;
            self.pending_window = None;
            self.reset_for_target_change();
        } else {
            self.discard_wgc();
            self.has_frame = false;
            self.dirty = true;
            self.hide_overlay();
        }
    }

    fn select_foreground_window(&mut self) -> Option<HWND> {
        if self
            .failed_window
            .is_some_and(|(_, since)| since.elapsed() >= Duration::from_secs(15))
        {
            self.failed_window = None;
        }
        let candidate = fluid_capture::external_foreground_window()
            .or_else(fluid_capture::last_external_foreground_window)
            .filter(|hwnd| {
                is_foreground_capture_candidate(*hwnd) && !self.window_blacklisted(*hwnd)
            });
        if let Some(current) = self.current_window {
            if !is_foreground_capture_candidate(current) || self.window_blacklisted(current) {
                tracing::info!("foreground capture target is no longer capturable");
                self.current_window = None;
                self.pending_window = None;
                self.fallback_grace_until = Some(Instant::now() + Duration::from_millis(650));
                self.reset_for_target_change();
            }
        }
        let Some(candidate) = candidate else {
            self.pending_window = None;
            return self.current_window;
        };
        if self.current_window == Some(candidate) {
            self.pending_window = None;
            return self.current_window;
        }
        if self.current_window.is_none() {
            tracing::info!("following foreground window 0x{:x}", candidate.0 as isize);
            self.current_window = Some(candidate);
            self.pending_window = None;
            self.fallback_grace_until = None;
            self.reset_for_target_change();
            return self.current_window;
        }
        match self.pending_window {
            Some((pending, since)) if pending == candidate => {
                if since.elapsed() >= Duration::from_millis(260) {
                    tracing::info!(
                        "switching foreground capture to 0x{:x}",
                        candidate.0 as isize
                    );
                    self.current_window = Some(candidate);
                    self.pending_window = None;
                    self.reset_for_target_change();
                }
            }
            _ => self.pending_window = Some((candidate, Instant::now())),
        }
        self.current_window
    }

    fn tick(&mut self, cfg: &AppConfig) -> Result<(), TickErr> {
        fluid_gfx::pump_thread_messages();
        self.shared
            .pipeline_heartbeat
            .fetch_add(1, Ordering::Relaxed);
        self.shared
            .stats
            .pipeline_ticks
            .fetch_add(1, Ordering::Relaxed);
        self.set_stage(STAGE_TICK);
        if self.last_debug_log.elapsed() >= Duration::from_secs(2) {
            let generated = self.shared.stats.generated_frames.load(Ordering::Relaxed);
            let capture_age_ms = if self.last_capture_qpc > 0 {
                qpc_diff_us(self.last_capture_qpc, qpc_now(), self.freq) as f64 / 1000.0
            } else {
                0.0
            };
            tracing::debug!(
                "pipeline stats: cap={:.1} out={:.1} refresh={:.1}Hz backend={} gpu={:.2}ms latency={:.1}ms capture_age={:.1}ms wait={:.2}ms present={:.2}ms render={:.2}ms fg={} generated_delta={} frames={} ticks={} acq={:.2}ms dirty={}",
                self.shared.stats.capture_fps(),
                self.shared.stats.present_fps(),
                self.shared.stats.refresh_hz_x100.load(Ordering::Relaxed) as f64 / 100.0,
                self.shared.stats.capture_backend.load(Ordering::Relaxed),
                self.shared.stats.gpu_ms(),
                self.shared.stats.pipeline_latency_ms(),
                capture_age_ms,
                self.shared.stats.present_wait_us.load(Ordering::Relaxed) as f64 / 1000.0,
                self.shared.stats.present_call_us.load(Ordering::Relaxed) as f64 / 1000.0,
                self.shared.stats.render_call_us.load(Ordering::Relaxed) as f64 / 1000.0,
                self.shared.stats.fg_active(),
                generated.saturating_sub(self.last_debug_generated),
                self.shared.stats.frame_index.load(Ordering::Relaxed),
                self.shared.stats.pipeline_ticks.load(Ordering::Relaxed),
                self.shared.stats.acquire_call_us.load(Ordering::Relaxed) as f64 / 1000.0,
                self.dirty
            );
            self.last_debug_generated = generated;
            self.last_debug_log = Instant::now();
            if self.shared.stats.fg_active() {
                let cap = self.shared.stats.capture_fps();
                let out = self.shared.stats.present_fps();
                let refresh =
                    self.shared.stats.refresh_hz_x100.load(Ordering::Relaxed) as f64 / 100.0;
                let requested = cfg.fg.max_multiplier.clamp(2, 8) as f64;
                let target = (cap * requested).min(refresh.max(1.0));
                if cap > 5.0 && target > cap * 1.35 && out < target * 0.85 {
                    self.fg_underperf_count = self.fg_underperf_count.saturating_add(1);
                } else {
                    self.fg_underperf_count = 0;
                }
                // Graduated degradation: first give up sharpening (cheap pass),
                // then the motion search, before pausing generation entirely.
                // Each stage holds for 10 s so a transient spike does not
                // permanently downgrade quality.
                if self.fg_underperf_count >= 2
                    && self
                        .fg_perf_until
                        .is_none_or(|until| Instant::now() >= until)
                {
                    self.fg_perf_until = Some(Instant::now() + Duration::from_secs(10));
                    self.fg_underperf_count = 0;
                    match self.fg_perf_stage {
                        0 => {
                            self.fg_perf_stage = 1;
                            tracing::warn!(
                                "frame generation performance fallback stage 1: sharpening disabled for 10s to approach the refresh-rate target"
                            );
                        }
                        _ => {
                            self.fg_perf_stage = 2;
                            tracing::warn!(
                                "frame generation performance fallback stage 2: motion search and sharpening disabled for 10s to approach the refresh-rate target"
                            );
                        }
                    }
                }
                if cap > 5.0 && out > 0.0 && out / cap < 1.15 {
                    self.fg_low_gain_count = self.fg_low_gain_count.saturating_add(1);
                } else {
                    self.fg_low_gain_count = 0;
                }
                if self.fg_low_gain_count >= 3 {
                    self.fg_backoff_until = Some(Instant::now() + Duration::from_secs(5));
                    self.fg_low_gain_count = 0;
                    tracing::warn!(
                        "frame generation paused for 5s: output is not increasing enough to justify its GPU cost"
                    );
                }
            } else {
                self.fg_underperf_count = 0;
                self.fg_low_gain_count = 0;
            }
            if self
                .fg_perf_until
                .is_some_and(|until| Instant::now() >= until)
            {
                self.fg_perf_stage = 0;
            }
            self.shared
                .stats
                .fg_perf_fallback
                .store(self.fg_perf_stage, Ordering::Relaxed);
        }

        self.sample_system_metrics();

        // Pace before polling the capture API, not after. Sleeping first means
        // the WGC/DDA acquire immediately before rendering observes the newest
        // completed source frame instead of a frame from before the sleep.
        // That removes up to one output period of avoidable input latency and
        // gives the frame-generation pair selector fresher endpoints.
        if let Some(next) = self.next_tick {
            self.set_stage(STAGE_PACE);
            sleep_until(next);
        }

        let mut active_window: Option<HWND> = None;
        let mut fallback_monitor: Option<isize> = None;
        match self.request {
            CaptureRequest::Monitor { hmonitor } => {
                fallback_monitor = Some(hmonitor);
            }
            CaptureRequest::Window { hwnd } => {
                let hwnd = HWND(hwnd as *mut core::ffi::c_void);
                if is_window(hwnd) && !is_iconic(hwnd) && !self.window_blacklisted(hwnd) {
                    active_window = Some(hwnd);
                } else {
                    // Never leave a selected-window session permanently black or
                    // spinning on an uncapturable target. Fall back to the monitor
                    // that contains the window (or the primary monitor).
                    let mut monitor = monitor_of(hwnd);
                    if monitor == 0 {
                        monitor = enumerate_monitors()
                            .first()
                            .map(|m| m.hmonitor)
                            .unwrap_or(-1);
                    }
                    if monitor > 0 {
                        tracing::debug!(
                            "selected window unavailable/uncapturable; falling back to monitor 0x{monitor:x}"
                        );
                        fallback_monitor = Some(monitor);
                        self.fallback_grace_until =
                            Some(Instant::now() + Duration::from_millis(650));
                    } else {
                        return self.wait_for_capture_target();
                    }
                }
            }
            CaptureRequest::Foreground {
                fallback_monitor: monitor,
            } => {
                fallback_monitor = Some(monitor);
                self.set_stage(STAGE_SELECT);
                active_window = self.select_foreground_window();
            }
        }

        if active_window.is_none()
            && self
                .fallback_grace_until
                .is_some_and(|until| Instant::now() < until)
        {
            // Opening/closing windows can briefly leave no valid foreground
            // target. Do not immediately flash a full-monitor DDA overlay across
            // the desktop; wait briefly for the next real window.
            self.set_stage(STAGE_GRACE);
            tracing::trace!("retarget grace active; hiding overlay temporarily");
            self.hide_overlay();
            self.has_frame = false;
            self.next_tick = None;
            std::thread::sleep(Duration::from_millis(80));
            return Ok(());
        }
        if self
            .fallback_grace_until
            .is_some_and(|until| Instant::now() >= until)
        {
            tracing::info!("retarget grace expired; allowing monitor fallback");
            self.fallback_grace_until = None;
        }

        let (mut target_rect, source_monitor, use_wgc) = if let Some(hwnd) = active_window {
            if !is_window(hwnd) || is_iconic(hwnd) {
                if matches!(self.request, CaptureRequest::Foreground { .. }) {
                    self.current_window = None;
                    self.reset_for_target_change();
                }
                return self.wait_for_capture_target();
            }
            let Some(rect) = client_rect_on_screen(hwnd) else {
                if matches!(self.request, CaptureRequest::Foreground { .. }) {
                    self.current_window = None;
                    self.reset_for_target_change();
                }
                return self.wait_for_capture_target();
            };
            let (w, h) = (rect.right - rect.left, rect.bottom - rect.top);
            if w < 16 || h < 16 {
                return self.wait_for_capture_target();
            }
            (rect, monitor_of(hwnd), true)
        } else if let Some(monitor) = fallback_monitor {
            let hmon = HMONITOR(monitor as *mut core::ffi::c_void);
            let Some(rect) = monitor_rect(hmon) else {
                return Err(TickErr::Recoverable(
                    "fallback monitor is no longer available".into(),
                ));
            };
            let wgc_monitor = !self.monitor_wgc_disabled;
            (rect, monitor, wgc_monitor)
        } else {
            return self.wait_for_capture_target();
        };

        if !use_wgc && self.wgc.is_some() {
            self.discard_wgc();
            self.has_frame = false;
            self.dirty = true;
        }

        self.shared.stats.source_id.store(
            if use_wgc {
                active_window
                    .map(|hwnd| hwnd.0 as isize)
                    .unwrap_or(source_monitor)
            } else {
                source_monitor
            },
            Ordering::Relaxed,
        );
        if let Some(rect) = monitor_rect(HMONITOR(source_monitor as *mut core::ffi::c_void)) {
            self.shared
                .stats
                .hud_monitor_left
                .store(rect.left, Ordering::Relaxed);
            self.shared
                .stats
                .hud_monitor_top
                .store(rect.top, Ordering::Relaxed);
            self.shared
                .stats
                .hud_monitor_right
                .store(rect.right, Ordering::Relaxed);
            self.shared
                .stats
                .hud_monitor_bottom
                .store(rect.bottom, Ordering::Relaxed);
        }

        if self.refresh_checked.elapsed() > Duration::from_secs(2) {
            self.refresh_hz =
                monitor_refresh_hz(HMONITOR(source_monitor as *mut core::ffi::c_void));
            self.refresh_checked = Instant::now();
        }

        if use_wgc {
            let is_monitor = matches!(self.request, CaptureRequest::Monitor { .. });
            let target_hwnd = if is_monitor {
                None
            } else {
                Some(
                    active_window
                        .ok_or_else(|| TickErr::Recoverable("missing WGC target window".into()))?,
                )
            };
            let target_w = (target_rect.right - target_rect.left).max(1) as u32;
            let target_h = (target_rect.bottom - target_rect.top).max(1) as u32;
            let same_target = if is_monitor {
                self.wgc_monitor == Some(source_monitor)
            } else {
                self.wgc_hwnd == target_hwnd.map(|hwnd| hwnd.0 as isize)
            };
            let size_changed = self
                .wgc_target_size
                .is_none_or(|(w, h)| w.abs_diff(target_w) > 2 || h.abs_diff(target_h) > 2);
            // A window resize changes the client size on every mouse-move tick.
            // Recreating a WinRT capture session and swapchain for each one can
            // deadlock the Intel capture/DComp driver. Keep using the old frame
            // pool while the window is moving, then resize it once it is stable.
            let resize_stable = if same_target && size_changed {
                match self.pending_target_size {
                    Some((pending, since)) if pending == (target_w, target_h) => {
                        since.elapsed() >= Duration::from_millis(500)
                    }
                    _ => {
                        self.pending_target_size = Some(((target_w, target_h), Instant::now()));
                        false
                    }
                }
            } else {
                self.pending_target_size = None;
                false
            };

            if self.wgc.is_none() || !same_target {
                self.discard_wgc();
                self.set_stage(STAGE_WGC_NEW);
                let created = if is_monitor {
                    WgcCapture::new_monitor(
                        &self.device_bundle.device,
                        HMONITOR(source_monitor as *mut core::ffi::c_void),
                    )
                } else {
                    WgcCapture::new(&self.device_bundle.device, target_hwnd.unwrap())
                };
                match created {
                    Ok(capture) => {
                        self.set_stage(STAGE_RING);
                        self.renderer
                            .ensure_bgra8_ring(capture.width(), capture.height())
                            .map_err(|e| {
                                TickErr::Recoverable(format!(
                                    "WGC frame ring allocation failed: {e}"
                                ))
                            })?;
                        self.renderer.clear_frames();
                        self.last_fg_pair = None;
                        self.last_fg_phase = 0.0;
                        self.wgc_target_size = Some((capture.width(), capture.height()));
                        self.wgc = Some(capture);
                        if is_monitor {
                            self.wgc_monitor = Some(source_monitor);
                            self.wgc_hwnd = None;
                        } else {
                            self.wgc_hwnd = target_hwnd.map(|hwnd| hwnd.0 as isize);
                            self.wgc_monitor = None;
                        }
                        self.monitor = source_monitor;
                        self.failed_window = None;
                        self.fallback_grace_until = None;
                        self.dirty = true;
                        self.has_frame = false;
                    }
                    Err(e) => {
                        if is_monitor {
                            tracing::warn!(
                                "WGC monitor capture failed for 0x{source_monitor:x}; falling back to desktop duplication: {e}"
                            );
                            self.monitor_wgc_disabled = true;
                            self.discard_wgc();
                            self.has_frame = false;
                            self.hide_overlay();
                            return Ok(());
                        }
                        let hwnd = target_hwnd.unwrap();
                        tracing::warn!(
                            "WGC cannot capture window 0x{:x}; backing off for 15s: {e}",
                            hwnd.0 as isize
                        );
                        self.mark_window_failed(hwnd);
                        self.fallback_grace_until =
                            Some(Instant::now() + Duration::from_millis(650));
                        self.discard_wgc();
                        self.has_frame = false;
                        if matches!(self.request, CaptureRequest::Foreground { .. }) {
                            self.current_window = None;
                            self.pending_window = None;
                            self.reset_for_target_change();
                        } else {
                            self.hide_overlay();
                        }
                        // Return successfully so the next tick can immediately fall
                        // back to desktop duplication instead of hammering WGC.
                        return Ok(());
                    }
                }
            } else if same_target && size_changed && resize_stable {
                self.set_stage(STAGE_RING);
                let resized = self
                    .wgc
                    .as_mut()
                    .map(|capture| capture.resize(target_w, target_h))
                    .unwrap_or(Ok(()));
                match resized {
                    Ok(()) => {
                        tracing::info!(
                            "WGC frame pool resized after stabilization: {}x{}",
                            target_w,
                            target_h
                        );
                        self.renderer
                            .ensure_bgra8_ring(target_w, target_h)
                            .map_err(|e| {
                                TickErr::Recoverable(format!("WGC frame ring resize failed: {e}"))
                            })?;
                        self.renderer.clear_frames();
                        self.last_fg_pair = None;
                        self.last_fg_phase = 0.0;
                        self.wgc_target_size = Some((target_w, target_h));
                        self.pending_target_size = None;
                        self.has_frame = false;
                        self.dirty = true;
                    }
                    Err(e) => {
                        tracing::warn!("WGC frame pool resize failed: {e}; recreating session");
                        if is_monitor {
                            self.monitor_wgc_disabled = true;
                        }
                        self.discard_wgc();
                        self.has_frame = false;
                    }
                }
            }
        } else if self
            .dda_retry_after
            .is_some_and(|retry| Instant::now() < retry)
        {
            self.hide_overlay();
            self.next_tick = None;
            std::thread::sleep(Duration::from_millis(100));
            return Ok(());
        } else if self.dup.is_none() || source_monitor != self.monitor {
            tracing::debug!(
                "initializing Desktop Duplication fallback for monitor 0x{source_monitor:x}"
            );
            self.set_stage(STAGE_DDA_NEW);
            match Duplication::new_for_monitor(
                &self.device_bundle.device,
                HMONITOR(source_monitor as *mut core::ffi::c_void),
            ) {
                Ok(d) => {
                    let (dw, dh) = (d.width(), d.height());
                    self.renderer.ensure_ring(dw, dh, d.format()).map_err(|e| {
                        TickErr::Recoverable(format!("frame ring allocation failed: {e}"))
                    })?;
                    self.renderer.clear_frames();
                    self.last_fg_pair = None;
                    self.last_fg_phase = 0.0;
                    self.dup = Some(d);
                    self.monitor = source_monitor;
                    self.dda_retry_after = None;
                    self.fallback_grace_until = None;
                    self.dirty = true;
                    self.has_frame = false;
                }
                Err(e) => {
                    tracing::warn!(
                        "desktop duplication initialization failed; retrying in 2s: {e}"
                    );
                    self.dda_retry_after = Some(Instant::now() + Duration::from_secs(2));
                    self.hide_overlay();
                    return Ok(());
                }
            }
        }

        // Acquire at most one frame per tick. DDA coalesces updates; WGC
        // produces frames only when the captured window content changes.
        let mut got_new = false;
        let mut frame_qpc = 0i64;

        if use_wgc {
            let mut acquired_qpc = None;
            let mut acquire_error = None;
            {
                self.set_stage(STAGE_CAPTURE);
                let capture = self.wgc.as_mut().unwrap();
                let acquire_started = Instant::now();
                let acquire_result = capture.acquire_latest();
                self.shared.stats.acquire_call_us.store(
                    acquire_started.elapsed().as_micros().min(u32::MAX as u128) as u32,
                    Ordering::Relaxed,
                );
                match acquire_result {
                    Ok(Some(frame)) => {
                        let qpc = frame.qpc;
                        self.renderer.push_frame(frame.texture, qpc);
                        acquired_qpc = Some(qpc);
                    }
                    Ok(None) => {}
                    Err(e) => acquire_error = Some(e),
                }
            }
            if let Some(e) = acquire_error {
                tracing::warn!(
                    "WGC capture failed on hwnd 0x{:x}; switching to fallback/backoff: {e}",
                    self.wgc_hwnd.unwrap_or(0)
                );
                self.mark_current_wgc_failed();
                self.hide_overlay();
                return Ok(());
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
            self.set_stage(STAGE_CAPTURE);
            let dup = self.dup.as_mut().unwrap();
            match dup.acquire_latest() {
                Ok(Some(frame)) => {
                    // In DDA mode a visible overlay can feed its own presents back
                    // into the duplicator. Those frames arrive within a few ms of
                    // our previous overlay present and otherwise create a full-rate
                    // feedback loop and high GPU/CPU load. Ignore them once the
                    // capture has a real frame; genuine source frames arrive later
                    // in the display interval.
                    let self_generated = self.has_frame
                        && self.overlay.is_visible()
                        && self.last_present_qpc > 0
                        && frame.qpc > 0
                        && {
                            let freq = self.freq.max(1) as i128;
                            let delta_us = (frame.qpc as i128 - self.last_present_qpc as i128)
                                * 1_000_000
                                / freq;
                            delta_us < 10_000
                        };
                    let should_consume =
                        (frame.desktop_updated || !self.has_frame) && !self_generated;
                    if self_generated {
                        tracing::trace!("ignored DDA update caused by the capture overlay present");
                    }
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
                    tracing::warn!("desktop duplication access lost; retrying in 2s");
                    self.dup = None;
                    self.dda_retry_after = Some(Instant::now() + Duration::from_secs(2));
                    self.hide_overlay();
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!("desktop duplication capture failed; retrying in 2s: {e}");
                    self.dup = None;
                    self.dda_retry_after = Some(Instant::now() + Duration::from_secs(2));
                    self.hide_overlay();
                    return Ok(());
                }
            }
        }

        if use_wgc {
            if let Some((captured_w, captured_h)) = self.wgc_target_size {
                target_rect.right = target_rect.left + captured_w as i32;
                target_rect.bottom = target_rect.top + captured_h as i32;
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
        self.shared.stats.source_width.store(
            (target_rect.right - target_rect.left).max(1) as u32,
            Ordering::Relaxed,
        );
        self.shared.stats.source_height.store(
            (target_rect.bottom - target_rect.top).max(1) as u32,
            Ordering::Relaxed,
        );
        self.shared.stats.output_width.store(
            (output_rect.right - output_rect.left).max(1) as u32,
            Ordering::Relaxed,
        );
        self.shared.stats.output_height.store(
            (output_rect.bottom - output_rect.top).max(1) as u32,
            Ordering::Relaxed,
        );
        self.shared
            .stats
            .refresh_hz_x100
            .store((self.refresh_hz.max(1.0) * 100.0) as u32, Ordering::Relaxed);
        let backend_code = if use_wgc {
            if matches!(self.request, CaptureRequest::Monitor { .. }) {
                2
            } else {
                1
            }
        } else {
            0
        };
        self.shared
            .stats
            .capture_backend
            .store(backend_code, Ordering::Relaxed);

        let keep_below_taskbar = matches!(self.request, CaptureRequest::Monitor { .. });
        if keep_below_taskbar != self.keep_overlay_below_taskbar {
            self.overlay.set_keep_below_taskbar(keep_below_taskbar);
            self.keep_overlay_below_taskbar = keep_below_taskbar;
        }
        self.set_stage(STAGE_OVERLAY_PLACE);
        self.overlay
            .place(&output_rect)
            .map_err(|e| TickErr::Recoverable(format!("overlay placement failed: {e}")))?;
        self.last_output_rect = Some(output_key);

        let refresh_interval_us = 1_000_000.0 / self.refresh_hz.max(24.0) as f64;
        let frame_period = Duration::from_secs_f64(1.0 / self.refresh_hz.max(24.0) as f64);
        let fg_pacing = FgPacing::new(
            self.capture_interval_us,
            self.refresh_hz,
            cfg.fg.max_multiplier,
        );

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
        if !self.has_frame || latest_now.is_none() {
            // WGC can legitimately be silent until the target window presents a
            // new frame, and switching targets discards the old frame snapshot.
            // Wait without tearing down the capture session.
            self.hide_overlay();
            self.next_tick = Some(Instant::now() + frame_period);
            return Ok(());
        }

        let capture_ratio = self.capture_interval_us / refresh_interval_us;
        let fg_eligible = cfg.fg.enabled
            && cfg.fg.strength > 0
            && cfg.fg.max_multiplier >= 2
            && capture_ratio >= 1.35
            && fg_pacing.allows_generation()
            && enough_frames
            && !source_stalled
            && self
                .fg_backoff_until
                .is_none_or(|until| Instant::now() >= until);
        self.shared
            .stats
            .fg_active
            .store(u32::from(fg_eligible), Ordering::Relaxed);
        self.shared.stats.fg_target_multiplier_x100.store(
            if fg_eligible {
                (fg_pacing.effective_multiplier * 100.0).round() as u32
            } else {
                0
            },
            Ordering::Relaxed,
        );
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
        let rtv = match self.overlay.rtv() {
            Some(r) => r.clone(),
            None => return Err(TickErr::Recoverable("swapchain RTV unavailable".into())),
        };

        // MaximumFrameLatency=1 is only an upper bound; if the compositor is
        // late, presenting again immediately queues work and raises latency.
        // Wait for a free frame slot only once this tick is known to present,
        // so static/paused ticks never consume the waitable signal needlessly.
        if self.overlay.is_visible() {
            let budget_us =
                (1_000_000.0 / self.refresh_hz.max(24.0) as f64 * 0.5).clamp(500.0, 4_000.0) as u32;
            let wait_started = Instant::now();
            let ready = self.overlay.wait_present_slot(budget_us);
            let waited_us = wait_started.elapsed().as_micros().min(u32::MAX as u128) as u32;
            self.shared
                .stats
                .present_wait_us
                .store(waited_us, Ordering::Relaxed);
            if !ready {
                tracing::trace!("present slot wait timed out after {waited_us}us");
            }
        }
        // Use the post-wait presentation instant as the phase reference rather
        // than the moment frame acquisition happened to finish. `now_qpc` is
        // sampled separately and still reports true end-to-end latency.
        let scheduled_present_qpc = qpc_now();
        let now_qpc = qpc_now();

        let latest =
            latest_now.ok_or_else(|| TickErr::Recoverable("no captured frame available".into()))?;

        let mut fg_phase = 1.0f32;
        let (previous, current) = if fg_eligible {
            let lookback_ticks =
                (fg_pacing.lookback.as_secs_f64() * self.freq as f64).round() as i64;
            let target_qpc = scheduled_present_qpc
                .min(now_qpc)
                .saturating_sub(lookback_ticks);
            match self.renderer.pair_at_time(target_qpc) {
                Some((prev, curr, phase)) => {
                    // Display time must never move backwards between presents
                    // of the same source pair: a late tick or a jittery frame
                    // arrival would otherwise show a briefly "younger" frame
                    // and read as a stutter.
                    let pair_key = (prev.seq, curr.seq);
                    if self.last_fg_pair == Some(pair_key) && phase < self.last_fg_phase {
                        fg_phase = self.last_fg_phase.min(0.999);
                    } else {
                        fg_phase = phase;
                    }
                    self.last_fg_pair = Some(pair_key);
                    self.last_fg_phase = fg_phase;
                    (prev, curr)
                }
                None => (latest.clone(), latest),
            }
        } else {
            self.last_fg_pair = None;
            (latest.clone(), latest)
        };

        self.shared.stats.fg_phase_x1000.store(
            (fg_phase.clamp(0.0, 1.0) * 1000.0) as u32,
            Ordering::Relaxed,
        );

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
        let perf_stage = if self
            .fg_perf_until
            .is_some_and(|until| Instant::now() < until)
        {
            self.fg_perf_stage
        } else {
            0
        };
        let scale_mode = if perf_stage >= 2 {
            1
        } else {
            match cfg.scale.mode {
                ScaleMode::Passthrough => 0,
                ScaleMode::Bilinear => 1,
                ScaleMode::Adaptive => 2,
            }
        };
        let options = RenderOptions {
            scale_mode,
            sharpness: if perf_stage >= 1 {
                0.0
            } else {
                cfg.scale.sharpness.min(100) as f32 / 100.0
            },
            fg_enabled: fg_eligible,
            motion_compensation: cfg.fg.motion_compensation && perf_stage < 2,
            preserve_aspect: cfg.scale.preserve_aspect,
            fg_strength: cfg.fg.strength.min(100) as f32 / 100.0,
            scene_protection: cfg.fg.scene_protection.min(100) as f32 / 100.0,
            fg_phase,
        };

        // The real-time HUD is now a separate lightweight window. The capture
        // overlay no longer draws any HUD content.
        let hud_options = HudOptions::default();
        let hud_text: Option<&str> = None;

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
        if !self.overlay.is_visible() {
            self.overlay.show(true);
        }
        let (bbw, bbh) = self.overlay.size();
        self.set_stage(STAGE_RENDER);
        let workload_started = Instant::now();
        let render_started = Instant::now();
        let render_result = self.renderer.render(RenderRequest {
            target: RenderTargetInfo {
                rtv: &rtv,
                width: bbw,
                height: bbh,
                uv_rect: uv,
            },
            previous,
            current,
            options,
            hud_text,
            hud: hud_options,
        });
        self.shared.stats.render_call_us.store(
            render_started.elapsed().as_micros().min(u32::MAX as u128) as u32,
            Ordering::Relaxed,
        );
        render_result.map_err(|e| TickErr::Recoverable(format!("render failed: {e}")))?;
        self.set_stage(STAGE_PRESENT);
        let present_started = Instant::now();
        self.overlay.present(sync);
        self.shared.stats.present_call_us.store(
            present_started.elapsed().as_micros().min(u32::MAX as u128) as u32,
            Ordering::Relaxed,
        );
        self.last_present_qpc = qpc_now();
        self.dirty = false;

        let workload = workload_started.elapsed();
        let overload_limit = frame_period.saturating_mul(2);
        if fg_eligible && workload > overload_limit {
            self.overload_count = self.overload_count.saturating_add(1);
        } else if workload < frame_period {
            self.overload_count = self.overload_count.saturating_sub(1);
        }
        if self.overload_count >= 45 {
            self.fg_backoff_until = Some(Instant::now() + Duration::from_secs(5));
            self.overload_count = 0;
            tracing::warn!(
                "frame generation paused for 5s: GPU cannot sustain the requested output rate"
            );
        }

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
        let next_period = if fg_eligible {
            fg_pacing.output_period
        } else {
            frame_period
        };
        // Absolute output grid: the schedule advances in whole periods from the
        // previous planned instant. A late present no longer pushes every
        // future present later (the old `now + period` re-anchor accumulated
        // drift and quietly reduced the output rate); fully missed slots are
        // accounted as dropped output frames instead.
        let cadence_base = self.next_tick.unwrap_or(now_instant);
        let mut next = cadence_base + next_period;
        if next <= now_instant {
            let late_periods =
                ((now_instant - next).as_secs_f64() / next_period.as_secs_f64()).floor() as u32;
            if late_periods > 0 {
                self.shared
                    .stats
                    .dropped_frames
                    .fetch_add(late_periods.min(8), Ordering::Relaxed);
            }
            next += next_period * (late_periods.min(8) + 1);
            if next <= now_instant {
                next = now_instant + next_period;
            }
        }
        self.next_tick = Some(next);
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
            self.shared
                .stats
                .pipeline_latency_us
                .store(lat_us.min(u32::MAX as f64) as u32, Ordering::Relaxed);
            tracing::trace!("pipeline effective frame age: {:.2}ms", lat_us / 1000.0);
        }
        Ok(())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.discard_wgc();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phase_after_arrival(pacing: FgPacing, source_period: Duration, slots: usize) -> Vec<f32> {
        // A capture arrives at t=source_period; output slots advance from
        // there. The interpolation clock is target = t - lookback.
        (0..slots)
            .map(|slot| {
                let now = source_period + pacing.output_period.mul_f64(slot as f64);
                let target = now.saturating_sub(pacing.lookback);
                (target.as_secs_f64() / source_period.as_secs_f64()).clamp(0.0, 1.0) as f32
            })
            .collect()
    }

    #[test]
    fn exact_2x_schedule_uses_one_interior_slot_per_source_frame() {
        let source_us = 1_000_000.0 / 60.0;
        let p = FgPacing::new(source_us, 120.0, 2);
        assert!((p.effective_multiplier - 2.0).abs() < 0.01);
        let phases = phase_after_arrival(p, Duration::from_secs_f64(1.0 / 60.0), 2);
        assert!((phases[0] - 0.5).abs() < 0.01, "{phases:?}");
        assert!((phases[1] - 1.0).abs() < 0.01, "{phases:?}");
    }

    #[test]
    fn exact_3x_schedule_uses_two_interior_slots_per_source_frame() {
        let source_us = 1_000_000.0 / 60.0;
        let p = FgPacing::new(source_us, 180.0, 3);
        assert!((p.effective_multiplier - 3.0).abs() < 0.01);
        let phases = phase_after_arrival(p, Duration::from_secs_f64(1.0 / 60.0), 3);
        for (i, expected) in [1.0 / 3.0, 2.0 / 3.0, 1.0].into_iter().enumerate() {
            assert!((phases[i] - expected).abs() < 0.01, "{phases:?}");
        }
    }

    #[test]
    fn each_stable_source_interval_emits_all_interior_slots() {
        // Simulate the pair selector directly: a source frame arrives every
        // interval, and M output slots follow it. With the lookback from
        // FgPacing the phases must be 1/M .. M/M, i.e. exactly M-1 generated
        // frames plus one real endpoint per interval.
        for multiplier in 2..=4u8 {
            let source_us = 1_000_000.0 / 60.0;
            let source_period = Duration::from_secs_f64(source_us * 1.0e-6);
            let p = FgPacing::new(source_us, (60 * multiplier) as f32, multiplier);
            let mut generated = 0u32;
            for interval in 1..=120u32 {
                let interval_start = source_period.mul_f64(interval as f64);
                for slot in 0..multiplier as u32 {
                    let now = interval_start + p.output_period.mul_f64(slot as f64);
                    let target = now.saturating_sub(p.lookback);
                    let phase = target.as_secs_f64() / source_period.as_secs_f64();
                    let phase_f = phase - phase.floor();
                    if phase_f > 0.001 && phase_f < 0.999 {
                        generated += 1;
                    }
                }
            }
            assert_eq!(
                generated,
                120 * (multiplier as u32 - 1),
                "multiplier {multiplier}"
            );
        }
    }

    #[test]
    fn refresh_rate_caps_the_requested_multiplier() {
        // 60 FPS source on a 144 Hz display can present at most 2.4x even if
        // the user asks for 4x. The schedule must report the attainable rate
        // instead of pretending that dropped slots were generated.
        let p = FgPacing::new(1_000_000.0 / 60.0, 144.0, 4);
        assert!((p.effective_multiplier - 2.4).abs() < 0.01);
        assert_eq!(p.output_period, Duration::from_secs_f64(1.0 / 144.0));
    }
}
