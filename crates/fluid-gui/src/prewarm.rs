//! Background GPU warm-up.
//!
//! Creating the wgpu device is a one-time cost that can take several seconds on
//! a cold driver (measured ~6 s on this machine). The engine keeps one
//! process-wide device, so building a tiny pipeline on a worker thread ahead of
//! time means the first press of 开始增强 already has a device and no longer
//! sits there without frames. Nothing here blocks the UI thread.

use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use fluid_core::{
    Dims, FluidConfig, InterpMode, ScaleConfig, ScaleMode, UpscaleBackendKind, UpscaleFilter,
};
use fluid_engine::{Pipeline, PipelineConfig};

/// Shown in the status strip while the device is being created.
pub const WARMING_TEXT: &str = "正在初始化 GPU…（第一次会多等几秒）";

/// Outcome of the background warm-up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrewarmOutcome {
    /// A compute device exists; the string is its description.
    Ready(String),
    /// No usable device; the pipeline will use the CPU backend.
    NoGpu(String),
}

impl PrewarmOutcome {
    pub fn text(&self) -> String {
        match self {
            Self::Ready(name) => format!("GPU 已就绪：{name}"),
            Self::NoGpu(reason) => format!("没有可用的 GPU，将用 CPU 计算（{reason}）"),
        }
    }
}

/// Snapshot of the warm-up, cheap to clone every frame.
#[derive(Clone, Debug, Default)]
pub struct PrewarmState {
    pub running: bool,
    pub outcome: Option<PrewarmOutcome>,
}

/// The background warm-up thread.
pub struct Prewarm {
    state: Arc<Mutex<PrewarmState>>,
    join: Option<JoinHandle<()>>,
}

impl Prewarm {
    /// Start warming the GPU on a worker thread. On Android the wgpu compute
    /// backend is not wired up yet, so nothing is spawned there.
    pub fn spawn() -> Self {
        let state = Arc::new(Mutex::new(PrewarmState::default()));
        if cfg!(target_os = "android") {
            return Self { state, join: None };
        }
        {
            let mut guard = lock(&state);
            guard.running = true;
        }
        let thread_state = Arc::clone(&state);
        let join = thread::Builder::new()
            .name("fluid-gui-prewarm".to_owned())
            .spawn(move || {
                let outcome = warm_gpu();
                let mut guard = lock(&thread_state);
                guard.running = false;
                guard.outcome = Some(outcome);
            })
            .ok();
        Self { state, join }
    }

    pub fn state(&self) -> PrewarmState {
        lock(&self.state).clone()
    }

    pub fn is_finished(&self) -> bool {
        match &self.join {
            Some(handle) => handle.is_finished(),
            None => true,
        }
    }

    pub fn join(&mut self) {
        if let Some(handle) = self.join.take() {
            let _ = handle.join();
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// What the status strip should say, if anything. The warm-up line is only
/// shown while it runs, and the result only while a run is not in progress.
pub fn status_text(state: &PrewarmState, run_active: bool) -> Option<String> {
    if state.running {
        return Some(WARMING_TEXT.to_owned());
    }
    if run_active {
        return None;
    }
    state.outcome.as_ref().map(PrewarmOutcome::text)
}

/// Probe the adapter, then build a tiny GPU pipeline: that call creates the
/// process-wide device the real runs reuse, so the cost is paid here instead of
/// on the first frame.
fn warm_gpu() -> PrewarmOutcome {
    let device = match fluid_gpu::probe() {
        Ok(info) => info.to_string(),
        Err(err) => return PrewarmOutcome::NoGpu(err.to_string()),
    };
    let mut config = FluidConfig::default();
    config.interpolator.mode = InterpMode::Off;
    config.interpolator.multiplier = 1;
    config.upscaler.backend = UpscaleBackendKind::Gpu;
    config.upscaler.filter = UpscaleFilter::Lanczos;
    config.upscaler.scale = ScaleConfig {
        mode: ScaleMode::Factor,
        factor: 2.0,
        ..Default::default()
    };
    // 64x36 -> 128x72: small enough to be free, big enough that the upscaler
    // really goes through the GPU path instead of the 1:1 shortcut.
    match Pipeline::with_config(&config, Dims::new(64, 36), PipelineConfig::default()) {
        Ok(_) => PrewarmOutcome::Ready(device),
        Err(err) => PrewarmOutcome::NoGpu(err.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_status_line_explains_the_wait_while_warming() {
        let state = PrewarmState {
            running: true,
            outcome: None,
        };
        assert_eq!(status_text(&state, false), Some(WARMING_TEXT.to_owned()));
        assert_eq!(status_text(&state, true), Some(WARMING_TEXT.to_owned()));
    }

    #[test]
    fn a_ready_gpu_is_named_after_the_warm_up() {
        let state = PrewarmState {
            running: false,
            outcome: Some(PrewarmOutcome::Ready(
                "Intel Iris Xe (integrated, vulkan)".to_owned(),
            )),
        };
        let text = status_text(&state, false).expect("text");
        assert!(text.contains("GPU 已就绪"));
        assert!(text.contains("Intel Iris Xe"));
        assert_eq!(status_text(&state, true), None);
    }

    #[test]
    fn a_missing_gpu_points_at_the_cpu_fallback() {
        let state = PrewarmState {
            running: false,
            outcome: Some(PrewarmOutcome::NoGpu("no adapter".to_owned())),
        };
        let text = status_text(&state, false).expect("text");
        assert!(text.contains("CPU"));
        assert!(text.contains("no adapter"));
    }

    #[test]
    fn nothing_is_shown_before_the_warm_up_starts() {
        assert_eq!(status_text(&PrewarmState::default(), false), None);
    }
}
