use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicIsize, AtomicU32, AtomicU64, Ordering};

/// 管线热路径统计。全部用原子量，管线线程写、UI 线程 4Hz 读，零锁竞争。
/// 浮点值以 ×100 定点存储。
#[derive(Debug, Default)]
pub struct Stats {
    pub capture_fps_x100: AtomicU32,
    pub present_fps_x100: AtomicU32,
    pub gpu_us: AtomicU32,
    pub pipeline_latency_us: AtomicU32,
    pub present_wait_us: AtomicU32,
    pub present_call_us: AtomicU32,
    pub render_call_us: AtomicU32,
    pub dropped_frames: AtomicU32,
    pub generated_frames: AtomicU64,
    pub fg_active: AtomicU32,
    pub fg_perf_fallback: AtomicU32,
    pub frame_index: AtomicU64,
    pub pipeline_ticks: AtomicU64,
    pub acquire_call_us: AtomicU32,
    pub source_width: AtomicU32,
    pub source_height: AtomicU32,
    pub output_width: AtomicU32,
    pub output_height: AtomicU32,
    pub refresh_hz_x100: AtomicU32,
    /// 0 = Desktop Duplication monitor, 1 = Windows Graphics Capture window.
    pub capture_backend: AtomicU32,
    pub generated_fps_x100: AtomicU32,
    pub process_cpu_x100: AtomicU32,
    pub working_set_mb: AtomicU32,
    pub fg_phase_x1000: AtomicU32,
    /// HWND for WGC, or monitor handle for DDA.
    pub source_id: AtomicIsize,
    pub hud_monitor_left: AtomicI32,
    pub hud_monitor_top: AtomicI32,
    pub hud_monitor_right: AtomicI32,
    pub hud_monitor_bottom: AtomicI32,
}

impl Stats {
    /// 对 x100 定点原子量做指数滑动平均。
    pub fn ema_x100(atomic: &AtomicU32, value: f64, alpha: f64) {
        let old = atomic.load(Ordering::Relaxed) as f64 / 100.0;
        let ema = old * (1.0 - alpha) + value * alpha;
        atomic.store((ema * 100.0) as u32, Ordering::Relaxed);
    }

    pub fn capture_fps(&self) -> f64 {
        self.capture_fps_x100.load(Ordering::Relaxed) as f64 / 100.0
    }

    pub fn present_fps(&self) -> f64 {
        self.present_fps_x100.load(Ordering::Relaxed) as f64 / 100.0
    }

    pub fn gpu_ms(&self) -> f64 {
        self.gpu_us.load(Ordering::Relaxed) as f64 / 1000.0
    }

    pub fn pipeline_latency_ms(&self) -> f64 {
        self.pipeline_latency_us.load(Ordering::Relaxed) as f64 / 1000.0
    }

    pub fn fg_active(&self) -> bool {
        self.fg_active.load(Ordering::Relaxed) != 0
    }

    pub fn generated_fps(&self) -> f64 {
        self.generated_fps_x100.load(Ordering::Relaxed) as f64 / 100.0
    }

    pub fn process_cpu_percent(&self) -> f64 {
        self.process_cpu_x100.load(Ordering::Relaxed) as f64 / 100.0
    }

    pub fn working_set_mb(&self) -> u32 {
        self.working_set_mb.load(Ordering::Relaxed)
    }

    pub fn source_size(&self) -> (u32, u32) {
        (
            self.source_width.load(Ordering::Relaxed),
            self.source_height.load(Ordering::Relaxed),
        )
    }

    pub fn output_size(&self) -> (u32, u32) {
        (
            self.output_width.load(Ordering::Relaxed),
            self.output_height.load(Ordering::Relaxed),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum RunState {
    #[default]
    Idle,
    Starting,
    Running,
    Paused,
    Failed,
}

#[derive(Debug, Default)]
pub struct SharedState {
    pub state: Mutex<RunState>,
    pub message: Mutex<String>,
    pub stats: Stats,
    /// Runtime HUD master switch. The hotkey can toggle this independently of
    /// the settings-window repaint rate.
    pub hud_enabled: AtomicBool,
    /// Set by hotkey-start so a visible settings window can minimize before the
    /// focus guard stops capture.
    pub minimize_ui_requested: AtomicBool,
    pub ui_heartbeat: AtomicU64,
    pub pipeline_heartbeat: AtomicU64,
    pub hud_heartbeat: AtomicU64,
    pub pipeline_stage: AtomicU32,
}

impl SharedState {
    pub fn set_state(&self, state: RunState, new_message: &str) {
        {
            let mut current = self.state.lock();
            if *current != state {
                *current = state;
            }
        }
        {
            let mut message = self.message.lock();
            if message.as_str() != new_message {
                message.clear();
                message.push_str(new_message);
            }
        }
    }
}
