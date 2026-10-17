use parking_lot::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// 管线热路径统计。全部用原子量，管线线程写、UI 线程 4Hz 读，零锁竞争。
/// 浮点值以 ×100 定点存储。
#[derive(Debug, Default)]
pub struct Stats {
    pub capture_fps_x100: AtomicU32,
    pub present_fps_x100: AtomicU32,
    pub gpu_us: AtomicU32,
    pub pipeline_latency_us: AtomicU32,
    pub dropped_frames: AtomicU32,
    pub generated_frames: AtomicU64,
    pub fg_active: AtomicU32,
    pub frame_index: AtomicU64,
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
