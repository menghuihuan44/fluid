//! The render pipeline: pull a frame from a source, process it (generate +
//! upscale), present every produced frame, pace the output, collect telemetry.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use fluid_core::{
    Dims, FluidConfig, FluidError, Frame, FrameSink, FrameSource, Result, fluid_debug, fluid_info,
    now_ns,
};
use fluid_core::{PacingMode, UpscaleFilter};

use crate::pacing::Pacer;
use crate::processor::{FrameProcessor, ProcessOutcome};
use crate::stats::{PipelineReport, PipelineStats};

/// Knobs that only affect the run loop, not image quality.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Stop after this many *source* frames (None = until source ends).
    pub max_input_frames: Option<u64>,
    /// Stop after this many *presented* frames (None = no limit).
    pub max_output_frames: Option<u64>,
    /// How often to emit a progress line while running.
    pub stats_interval: Option<Duration>,
    /// Print a progress line every stats_interval.
    pub verbose: bool,
    /// Sleep to keep output on cadence. Turn off for offline file processing.
    pub pace: bool,
    /// Override the target output framerate (used when pacing).
    pub target_fps: Option<f32>,
    /// Clear the signal on success only; on failure the run still closes sinks.
    pub close_on_finish: bool,
    /// Cooperative cancellation token (for example one flipped by a Ctrl+C
    /// handler). Checked once per frame; when set, the loop stops and `run`
    /// returns `Ok(report)` with the frames processed so far.
    pub cancel: Option<Arc<AtomicBool>>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_input_frames: None,
            max_output_frames: None,
            stats_interval: Some(Duration::from_millis(1000)),
            verbose: false,
            pace: true,
            target_fps: None,
            close_on_finish: true,
            cancel: None,
        }
    }
}

impl PipelineConfig {
    pub fn offline() -> Self {
        Self {
            pace: false,
            stats_interval: None,
            verbose: false,
            ..Default::default()
        }
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn with_max_output_frames(mut self, n: u64) -> Self {
        self.max_output_frames = Some(n);
        self
    }

    pub fn with_max_input_frames(mut self, n: u64) -> Self {
        self.max_input_frames = Some(n);
        self
    }

    pub fn with_pace(mut self, pace: bool) -> Self {
        self.pace = pace;
        self
    }

    /// Attach a cancellation flag checked once per frame.
    pub fn with_cancel(mut self, cancel: Arc<AtomicBool>) -> Self {
        self.cancel = Some(cancel);
        self
    }
}

/// Owns the processor plus run-loop state.
pub struct Pipeline {
    processor: FrameProcessor,
    pcfg: PipelineConfig,
    stats: PipelineStats,
    started: Option<Instant>,
}

impl Pipeline {
    pub fn new(cfg: &FluidConfig, src_dims: Dims) -> Result<Self> {
        Self::with_config(cfg, src_dims, PipelineConfig::default())
    }

    pub fn with_config(cfg: &FluidConfig, src_dims: Dims, pcfg: PipelineConfig) -> Result<Self> {
        let processor = FrameProcessor::new(cfg, src_dims)?;
        Ok(Self {
            processor,
            pcfg,
            stats: PipelineStats::default(),
            started: None,
        })
    }

    pub fn processor(&self) -> &FrameProcessor {
        &self.processor
    }

    pub fn processor_mut(&mut self) -> &mut FrameProcessor {
        &mut self.processor
    }

    pub fn config(&self) -> &FluidConfig {
        self.processor.config()
    }

    pub fn run_config(&self) -> &PipelineConfig {
        &self.pcfg
    }

    pub fn set_run_config(&mut self, pcfg: PipelineConfig) {
        self.pcfg = pcfg;
    }

    pub fn src_dims(&self) -> Dims {
        self.processor.src_dims()
    }

    pub fn out_dims(&self) -> Dims {
        self.processor.out_dims()
    }

    pub fn multiplier(&self) -> u32 {
        self.processor.multiplier()
    }

    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    pub fn reset(&mut self) {
        self.processor.reset();
        self.stats.reset();
        self.started = None;
    }

    /// Process a single frame and update counters. Stateless w.r.t. run loop.
    pub fn process_frame(&mut self, frame: Frame) -> Result<ProcessOutcome> {
        let now = now_ns();
        let outcome = self.processor.process(frame)?;
        self.stats.record_outcome(&outcome, now);
        Ok(outcome)
    }

    /// Target output rate for pacing, or None to leave pacing disabled.
    pub fn resolve_target_fps(&self, source: &dyn FrameSource) -> Option<f32> {
        if !self.pcfg.pace {
            return None;
        }
        if let Some(fps) = self.pcfg.target_fps
            && fps > 1.0
        {
            return Some(fps);
        }
        let configured = self.config().latency.target_fps;
        if configured > 0.0 {
            return Some(configured);
        }
        let src = source.fps();
        if src > 1.0 && self.config().latency.pacing != PacingMode::Unlimited {
            let mult = self.multiplier() as f32;
            return Some(src * mult.max(1.0));
        }
        None
    }

    /// Run until the source ends, the sink closes, a frame limit is hit, or
    /// the cancel token is set. Cancellation is cooperative: the frame in
    /// flight is finished and a normal report is returned.
    pub fn run(
        &mut self,
        source: &mut dyn FrameSource,
        sink: &mut dyn FrameSink,
    ) -> Result<PipelineReport> {
        if sink.dims() != self.processor.out_dims() {
            return Err(FluidError::invalid(format!(
                "sink is {:?} but pipeline produces {:?} (check upscaler scale settings)",
                sink.dims(),
                self.processor.out_dims()
            )));
        }
        // Adaptive interpolation needs the source cadence to size its budget.
        self.processor.set_source_fps(source.fps());
        let src_name = source.backend_name().to_string();
        let sink_name = sink.backend_name().to_string();
        let target_fps = self.resolve_target_fps(source);
        let mut pacer = match target_fps {
            Some(fps) => Pacer::new(fps),
            None => Pacer::disabled(),
        };
        if self.pcfg.verbose {
            fluid_info!(
                "pipeline start: {} {:?} -> {} {:?} ({}x, filter {})",
                src_name,
                self.processor.src_dims(),
                sink_name,
                self.processor.out_dims(),
                self.multiplier(),
                self.config().upscaler.filter.as_str()
            );
        }
        let started = Instant::now();
        self.started = Some(started);
        self.stats.reset();
        let mut last_stats = Instant::now();
        let mut last_pts = i64::MIN;
        let mut last_index = u64::MAX;
        let mut have_last = false;
        let mut presented = 0u64;
        // Clone the token so the loop does not borrow the config while the
        // pipeline itself is mutated.
        let cancel = self.pcfg.cancel.clone();

        'outer: loop {
            if let Some(flag) = cancel.as_ref()
                && flag.load(Ordering::Relaxed)
            {
                fluid_debug!(
                    "cancel requested after {} input frames",
                    self.stats.frames_in
                );
                break;
            }
            if sink.should_close() {
                fluid_debug!("sink requested close");
                break;
            }
            if let Some(max) = self.pcfg.max_input_frames
                && self.stats.frames_in >= max
            {
                break;
            }
            let captured_ns = now_ns();
            let fetch_started = Instant::now();
            let frame = match source.next_frame() {
                Ok(Some(f)) => f,
                Ok(None) => break,
                Err(e) => {
                    if self.pcfg.close_on_finish {
                        sink.close();
                    }
                    return Err(e);
                }
            };
            self.stats
                .record_source_wait(fetch_started.elapsed().as_nanos() as i64);
            if have_last && frame.pts_ns == last_pts && frame.index == last_index {
                self.stats.duplicates += 1;
                // A repeated frame means the compositor had nothing new; do not
                // re-process and do not busy-spin.
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            have_last = true;
            last_pts = frame.pts_ns;
            last_index = frame.index;

            let outcome = self.process_frame(frame)?;
            for out in outcome.frames.iter() {
                let present_started = Instant::now();
                let presented_ok = sink.present(out);
                self.stats
                    .record_present_time(present_started.elapsed().as_nanos() as i64);
                if let Err(e) = presented_ok {
                    if matches!(e, FluidError::Cancelled) || sink.should_close() {
                        fluid_debug!("sink cancelled");
                        break 'outer;
                    }
                    if self.pcfg.close_on_finish {
                        sink.close();
                    }
                    return Err(e);
                }
                let now = now_ns();
                self.stats.record_present(now - captured_ns, now);
                presented += 1;
                pacer.wait();
                if let Some(max) = self.pcfg.max_output_frames
                    && presented >= max
                {
                    break 'outer;
                }
            }
            if let Some(interval) = self.pcfg.stats_interval
                && last_stats.elapsed() >= interval
            {
                last_stats = Instant::now();
                if self.pcfg.verbose {
                    fluid_info!("{}", self.stats.summary());
                }
            }
        }

        if self.pcfg.close_on_finish {
            sink.close();
            source.close();
        }
        self.stats.finish();
        if self.pcfg.verbose {
            fluid_info!("pipeline done in {:.3}s", started.elapsed().as_secs_f64());
        }
        let report = self.report_with(&src_name, &sink_name);
        Ok(report)
    }

    /// Build a report from current state (source/sink names already recorded).
    pub fn report(&self) -> PipelineReport {
        self.report_with("", "")
    }

    fn report_with(&self, source: &str, sink: &str) -> PipelineReport {
        let cfg = self.processor.config();
        PipelineReport {
            stats: self.stats.clone(),
            source: source.to_string(),
            sink: sink.to_string(),
            src_dims: self.processor.src_dims(),
            out_dims: self.processor.out_dims(),
            multiplier: self.multiplier(),
            filter: cfg.upscaler.filter.as_str().to_string(),
            backend: self
                .processor
                .upscale_backend()
                .unwrap_or("none")
                .to_string(),
            interp_backend: self
                .processor
                .interp_backend()
                .unwrap_or("none")
                .to_string(),
            preset: cfg.profile.as_str().to_string(),
        }
    }

    /// Convenience: describe what a given source would do without running it.
    pub fn plan(cfg: &FluidConfig, src_dims: Dims) -> Result<String> {
        let p = FrameProcessor::new(cfg, src_dims)?;
        let filter = cfg.upscaler.filter;
        Ok(format!(
            "source {} -> output {}; upscale {} ({}); interpolation {} at {}x; preset {}",
            p.src_dims(),
            p.out_dims(),
            if filter == UpscaleFilter::Off {
                "off"
            } else {
                filter.as_str()
            },
            filter_cost_label(filter),
            if p.interpolation_enabled() {
                "on"
            } else {
                "off"
            },
            p.multiplier(),
            cfg.profile.as_str()
        ))
    }
}

fn filter_cost_label(filter: UpscaleFilter) -> &'static str {
    match filter {
        UpscaleFilter::Off => "free",
        UpscaleFilter::Nearest => "cheapest",
        UpscaleFilter::Bilinear => "cheap",
        UpscaleFilter::Bicubic => "moderate",
        UpscaleFilter::Lanczos => "expensive",
        UpscaleFilter::Adaptive => "adaptive",
        UpscaleFilter::Temporal => "temporal",
    }
}

/// Run one source into one sink with the given configuration.
pub fn run(
    cfg: &FluidConfig,
    source: &mut dyn FrameSource,
    sink: &mut dyn FrameSink,
) -> Result<PipelineReport> {
    let dims = source.dims();
    let mut pipeline = Pipeline::new(cfg, dims)?;
    pipeline.run(source, sink)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluid_core::PixelFormat;
    use fluid_video::SyntheticSource;

    fn small() -> Dims {
        Dims::new(32, 18)
    }

    #[test]
    fn rejects_mismatched_sink_dims() {
        let cfg = FluidConfig::default();
        let mut p = Pipeline::new(&cfg, small()).unwrap();
        let mut sink = fluid_present::null_sink(Dims::new(4, 4));
        let mut src = SyntheticSource::new(small(), 30.0);
        let err = p.run(&mut src, &mut sink).unwrap_err();
        assert!(matches!(err, FluidError::InvalidArgument(_)));
    }

    #[test]
    fn runs_synthetic_through_null_sink() {
        let mut cfg = FluidConfig::default();
        cfg.interpolator.mode = fluid_core::InterpMode::Off;
        cfg.upscaler.filter = UpscaleFilter::Off;
        let dims = small();
        let run_cfg = PipelineConfig::offline().with_max_input_frames(8);
        let mut p = Pipeline::with_config(&cfg, dims, run_cfg).unwrap();
        let mut src = SyntheticSource::new(dims, 30.0);
        let mut sink = fluid_present::null_sink(dims);
        let report = p.run(&mut src, &mut sink).unwrap();
        assert!(
            report.stats.frames_in > 4,
            "frames_in={}",
            report.stats.frames_in
        );
        assert_eq!(report.stats.frames_out, report.stats.frames_in);
        assert!(report.stats.process_ns.count() > 0);
    }

    /// A source that delivers the exact same pixels every frame (a static
    /// desktop or a paused game) while still advancing pts/index.
    struct StaticSource {
        dims: Dims,
        served: u64,
        limit: u64,
    }

    impl FrameSource for StaticSource {
        fn dims(&self) -> Dims {
            self.dims
        }

        fn fps(&self) -> f32 {
            30.0
        }

        fn backend_name(&self) -> &'static str {
            "static-test"
        }

        fn next_frame(&mut self) -> Result<Option<Frame>> {
            if self.served >= self.limit {
                return Ok(None);
            }
            let index = self.served;
            self.served += 1;
            Ok(Some(
                Frame::zeroed(self.dims, PixelFormat::Rgba8)
                    .with_pts((index as i64) * 33_000_000)
                    .with_index(index),
            ))
        }

        fn close(&mut self) {}
    }

    #[test]
    fn static_source_frames_are_counted_and_fast_pathed() {
        let mut cfg = FluidConfig::default();
        cfg.interpolator.mode = fluid_core::InterpMode::Fixed;
        cfg.interpolator.multiplier = 2;
        cfg.upscaler.filter = UpscaleFilter::Off;
        let dims = small();
        let run_cfg = PipelineConfig::offline().with_max_input_frames(5);
        let mut p = Pipeline::with_config(&cfg, dims, run_cfg).unwrap();
        let mut src = StaticSource {
            dims,
            served: 0,
            limit: 5,
        };
        let mut sink = fluid_present::null_sink(dims);
        let report = p.run(&mut src, &mut sink).unwrap();
        assert_eq!(report.stats.frames_in, 5);
        assert_eq!(report.stats.frames_out, 10);
        assert_eq!(
            report.stats.static_frames, 4,
            "every frame after the first is pixel-identical to its predecessor"
        );
        assert_eq!(report.stats.process_ns.count(), 5);
    }
    #[test]
    fn frame_limit_stops_the_run() {
        let cfg = FluidConfig::default();
        let dims = small();
        let run_cfg = PipelineConfig::offline().with_max_output_frames(5);
        let mut p = Pipeline::with_config(&cfg, dims, run_cfg).unwrap();
        let mut src = SyntheticSource::new(dims, 30.0);
        let mut sink = fluid_present::null_sink(dims);
        let report = p.run(&mut src, &mut sink).unwrap();
        assert_eq!(report.stats.frames_out, 5);
    }

    #[test]
    fn plan_mentions_dims() {
        let cfg = FluidConfig::default();
        let plan = Pipeline::plan(&cfg, small()).unwrap();
        assert!(plan.contains("32x18"), "{plan}");
    }

    /// A source that flips the cancel token after serving N frames and records
    /// whether close() was called.
    struct CancellingSource {
        inner: SyntheticSource,
        cancel: Arc<AtomicBool>,
        cancel_after: u64,
        served: u64,
        closed: Arc<AtomicBool>,
    }

    impl FrameSource for CancellingSource {
        fn dims(&self) -> Dims {
            self.inner.dims()
        }

        fn fps(&self) -> f32 {
            self.inner.fps()
        }

        fn backend_name(&self) -> &'static str {
            "cancelling-test"
        }

        fn next_frame(&mut self) -> Result<Option<Frame>> {
            let frame = self.inner.next_frame()?;
            if frame.is_some() {
                self.served += 1;
                if self.served >= self.cancel_after {
                    self.cancel.store(true, Ordering::Relaxed);
                }
            }
            Ok(frame)
        }

        fn close(&mut self) {
            self.closed.store(true, Ordering::Relaxed);
        }
    }

    /// Null sink that records close().
    struct CloseTrackingSink {
        dims: Dims,
        closed: Arc<AtomicBool>,
    }

    impl FrameSink for CloseTrackingSink {
        fn dims(&self) -> Dims {
            self.dims
        }

        fn backend_name(&self) -> &'static str {
            "close-tracking"
        }

        fn present(&mut self, _frame: &Frame) -> Result<()> {
            Ok(())
        }

        fn close(&mut self) {
            self.closed.store(true, Ordering::Relaxed);
        }
    }

    #[test]
    fn cancel_token_stops_the_run_with_a_report() {
        let mut cfg = FluidConfig::default();
        cfg.interpolator.mode = fluid_core::InterpMode::Off;
        cfg.upscaler.filter = UpscaleFilter::Off;
        let dims = small();
        let cancel = Arc::new(AtomicBool::new(false));
        let sink_closed = Arc::new(AtomicBool::new(false));
        let source_closed = Arc::new(AtomicBool::new(false));

        let run_cfg = PipelineConfig::offline().with_cancel(cancel.clone());
        let mut p = Pipeline::with_config(&cfg, dims, run_cfg).unwrap();
        let mut src = CancellingSource {
            inner: SyntheticSource::new(dims, 30.0),
            cancel: cancel.clone(),
            cancel_after: 6,
            served: 0,
            closed: source_closed.clone(),
        };
        let mut sink = CloseTrackingSink {
            dims,
            closed: sink_closed.clone(),
        };

        let report = p.run(&mut src, &mut sink).unwrap();
        assert!(cancel.load(Ordering::Relaxed), "token was not set");
        assert_eq!(report.stats.frames_in, 6);
        assert_eq!(report.stats.frames_out, report.stats.frames_in);
        assert_eq!(src.served, 6);
        assert!(sink_closed.load(Ordering::Relaxed), "sink was not closed");
        assert!(
            source_closed.load(Ordering::Relaxed),
            "source was not closed"
        );
    }
}
