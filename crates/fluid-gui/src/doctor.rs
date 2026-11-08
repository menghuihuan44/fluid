//! Performance check ("性能体检"): runs a couple of short synthetic probes and
//! turns the measured timings into a beginner friendly recommendation.

use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use fluid_core::{Dims, FluidConfig};
use fluid_engine::{Pipeline, PipelineConfig, PipelineReport};
use fluid_present::null_sink;
use fluid_video::SyntheticSource;

use crate::mapping::{
    InterpChoice, ScaleChoice, backend_display_name, interp_backend_display_name,
};

/// One measured probe run.
#[derive(Clone, Debug)]
pub struct ProbeResult {
    /// Human readable label, e.g. "720p".
    pub label: String,
    pub src_dims: Dims,
    pub out_dims: Dims,
    pub frames: u64,
    /// Mean processing time per input frame.
    pub mean_ms: f32,
    /// Output frames per second the machine can sustain with these settings.
    pub fps: f32,
    /// Upscaling backend the probe actually used ("cpu", "d3d11-compute", "none").
    pub backend: String,
    /// Frame generation backend the probe actually used ("cpu", "wgpu-flow", "none").
    pub interp_backend: String,
}

impl ProbeResult {
    pub fn summary(&self) -> String {
        format!(
            "{}: {}x{} -> {}x{}, {} 帧, {:.2} ms/帧, {:.0} fps, 放大后端 {}, 插帧后端 {}",
            self.label,
            self.src_dims.width,
            self.src_dims.height,
            self.out_dims.width,
            self.out_dims.height,
            self.frames,
            self.mean_ms,
            self.fps,
            backend_display_name(&self.backend),
            interp_backend_display_name(&self.interp_backend)
        )
    }
}

/// The state shared with the UI while a check is running.
#[derive(Clone, Debug, Default)]
pub struct DoctorState {
    pub running: bool,
    pub results: Vec<ProbeResult>,
    pub recommendation: String,
    pub error: Option<String>,
}

/// Run one probe: a synthetic source through a real pipeline into a null sink.
pub fn run_probe(
    label: &str,
    src_dims: Dims,
    config: &FluidConfig,
) -> fluid_core::Result<ProbeResult> {
    let pipeline_config = PipelineConfig {
        pace: false,
        max_output_frames: Some(24),
        close_on_finish: true,
        stats_interval: None,
        verbose: false,
        ..Default::default()
    };
    let mut pipeline = Pipeline::with_config(config, src_dims, pipeline_config)?;
    let out_dims = pipeline.out_dims();
    let mut source = SyntheticSource::new(src_dims, 60.0);
    let mut sink = null_sink(out_dims);
    let report: PipelineReport = pipeline.run(&mut source, &mut sink)?;
    Ok(result_from_report(label, src_dims, out_dims, &report))
}

fn result_from_report(
    label: &str,
    src_dims: Dims,
    out_dims: Dims,
    report: &PipelineReport,
) -> ProbeResult {
    let frames = report.stats.frames_out;
    let mean_ms = report.stats.mean_process_ms();
    let elapsed_s = report.elapsed_s();
    let fps = if elapsed_s > 0.0 {
        frames as f32 / elapsed_s as f32
    } else {
        0.0
    };
    ProbeResult {
        label: label.to_owned(),
        src_dims,
        out_dims,
        frames,
        mean_ms,
        fps,
        backend: report.backend.clone(),
        interp_backend: report.interp_backend.clone(),
    }
}

/// Turn the measured 1080p numbers into a concrete recommendation.
///
/// Pure function so the decision logic is unit-testable: `mean_ms` is the
/// processing time of one input frame at 1080p with the current settings.
pub fn recommendation(mean_ms: f32) -> Recommendation {
    let capacity = if mean_ms > 0.0 {
        1000.0 / mean_ms
    } else {
        1000.0
    };

    // Keep the output below ~144 fps (a typical high refresh display) and only
    // interpolate when the machine can already produce at least 20 fps: below
    // that, frame generation makes artifacts worse instead of helping.
    let interp = if capacity < 20.0 || capacity * 2.0 > 144.0 {
        InterpChoice::Off
    } else if capacity * 3.0 > 144.0 {
        InterpChoice::X2
    } else if capacity * 4.0 > 144.0 {
        InterpChoice::X3
    } else {
        InterpChoice::X4
    };

    let scale = if capacity >= 30.0 {
        ScaleChoice::X2
    } else if capacity >= 15.0 {
        ScaleChoice::X1_5
    } else {
        ScaleChoice::Off
    };

    let predicted = capacity * interp.multiplier() as f32;
    Recommendation {
        interp,
        scale,
        predicted_fps: predicted,
    }
}

/// A concrete suggestion shown to the user.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Recommendation {
    pub interp: InterpChoice,
    pub scale: ScaleChoice,
    pub predicted_fps: f32,
}

impl Recommendation {
    pub fn text(&self) -> String {
        let interp = if self.interp == InterpChoice::Off {
            "关闭插帧".to_owned()
        } else {
            format!("{} 插帧", self.interp.label())
        };
        let scale = if self.scale == ScaleChoice::Off {
            "关闭放大".to_owned()
        } else {
            format!("{} 放大", self.scale.label())
        };
        format!(
            "推荐: {} + {}，预计约 {:.0} fps",
            interp, scale, self.predicted_fps
        )
    }
}

/// Handle for a running performance check.
pub struct DoctorHandle {
    state: Arc<Mutex<DoctorState>>,
    join: Option<JoinHandle<()>>,
}

impl DoctorHandle {
    /// Start the two probes (720p and 1080p) on a worker thread.
    pub fn spawn(config: FluidConfig) -> Self {
        let state = Arc::new(Mutex::new(DoctorState {
            running: true,
            ..Default::default()
        }));
        let thread_state = Arc::clone(&state);

        let join = thread::Builder::new()
            .name("fluid-gui-doctor".to_owned())
            .spawn(move || {
                let mut results = Vec::new();
                let mut error = None;
                for (label, dims) in [
                    ("720p", Dims::new(1280, 720)),
                    ("1080p", Dims::new(1920, 1080)),
                ] {
                    match run_probe(label, dims, &config) {
                        Ok(result) => results.push(result),
                        Err(err) => {
                            error = Some(format!("{label} 测试失败：{err}"));
                            break;
                        }
                    }
                }

                let recommendation = results
                    .iter()
                    .find(|result| result.label == "1080p")
                    .or_else(|| results.last())
                    .map(|result| recommendation(result.mean_ms).text())
                    .unwrap_or_else(|| "无法给出建议：测试没有产生结果".to_owned());

                if let Ok(mut guard) = thread_state.lock() {
                    guard.results = results;
                    guard.recommendation = recommendation;
                    guard.error = error;
                    guard.running = false;
                }
            })
            .ok();

        Self { state, join }
    }

    /// Snapshot of the current state (cheap: results are only a couple rows).
    pub fn snapshot(&self) -> DoctorState {
        match self.state.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    pub fn is_running(&self) -> bool {
        match self.state.lock() {
            Ok(guard) => guard.running,
            Err(poisoned) => poisoned.into_inner().running,
        }
    }

    pub fn join(&mut self) {
        if let Some(handle) = self.join.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slow_machine_gets_no_extra_multiplier() {
        // 80 ms per frame at 1080p -> 12.5 fps, cannot even hold 60 output fps.
        let rec = recommendation(80.0);
        assert_eq!(rec.interp, InterpChoice::Off);
        assert_eq!(rec.scale, ScaleChoice::Off);
    }

    #[test]
    fn mid_range_machine_gets_4x_and_15x() {
        // 40 ms per frame -> 25 fps input, 4x brings it to ~100 output fps.
        let rec = recommendation(40.0);
        assert_eq!(rec.interp, InterpChoice::X4);
        assert_eq!(rec.scale, ScaleChoice::X1_5);
        assert!((rec.predicted_fps - 100.0).abs() < 0.5);
    }

    #[test]
    fn fast_machine_does_not_need_interpolation() {
        // 10 ms per frame -> 100 fps input already, no interpolation needed.
        let rec = recommendation(10.0);
        assert_eq!(rec.interp, InterpChoice::Off);
        assert_eq!(rec.scale, ScaleChoice::X2);
    }

    #[test]
    fn recommendation_text_is_chinese_and_contains_fps() {
        let text = recommendation(22.0).text();
        assert!(text.starts_with("推荐:"), "unexpected text: {text}");
        assert!(text.contains("fps"), "unexpected text: {text}");
        assert!(text.contains("插帧"), "unexpected text: {text}");
    }

    #[test]
    fn very_slow_machine_is_told_to_turn_everything_off() {
        // 200 ms per frame -> 5 fps; interpolation cannot save this.
        let rec = recommendation(200.0);
        assert_eq!(rec.interp, InterpChoice::Off);
        assert_eq!(rec.scale, ScaleChoice::Off);
        assert!((rec.predicted_fps - 5.0).abs() < 0.5);
    }

    #[test]
    fn probe_result_summary_lists_resolutions_and_timings() {
        let result = ProbeResult {
            label: "1080p".to_owned(),
            src_dims: Dims::new(1920, 1080),
            out_dims: Dims::new(2880, 1620),
            frames: 24,
            mean_ms: 12.5,
            fps: 80.0,
            backend: "cpu".to_owned(),
            interp_backend: "wgpu-flow".to_owned(),
        };
        let summary = result.summary();
        assert!(summary.contains("1920x1080"));
        assert!(summary.contains("2880x1620"));
        assert!(summary.contains("12.50 ms"));
        assert!(summary.contains("CPU（处理器）"));
        // The summary must name both stages so a fallback is visible.
        assert!(summary.contains("GPU（计算）"));
    }
}
