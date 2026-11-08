//! Live pipeline worker.
//!
//! The GUI thread only draws; a single worker thread owns the `FrameSource`
//! and the `GuiSink`, runs the `Pipeline` and publishes results through
//! shared state.  Starting / stopping never blocks the UI.

use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use fluid_core::{Dims, FluidConfig, FrameSource};
use fluid_engine::{Pipeline, PipelineConfig, PipelineReport};

use crate::sink::{GuiSink, PreviewShared};

/// A running (or finished) live session.
pub struct RunHandle {
    shared: Arc<PreviewShared>,
    error: Arc<Mutex<Option<String>>>,
    report: Arc<Mutex<Option<PipelineReport>>>,
    join: Option<JoinHandle<()>>,
    started: Instant,
    /// Backend name the source reported when the run started, e.g. "dxgi" or
    /// "window-wgc", so the status strip can name the real capture source
    /// before the pipeline report exists.
    source_backend: &'static str,
}

/// Snapshot of the worker state, cheap to read every UI frame.
#[derive(Clone, Debug, Default)]
pub struct WorkerStatus {
    pub generation: u64,
    pub frames: u64,
    pub finished: bool,
    pub error: Option<String>,
}

impl RunHandle {
    /// Spawn the pipeline on a worker thread.
    ///
    /// The pipeline is built *inside* the thread so that a config problem can
    /// never block or fail the UI thread; it is reported through
    /// `WorkerStatus::error` instead.
    pub fn spawn(source: Box<dyn FrameSource>, config: FluidConfig) -> Self {
        let shared = Arc::new(PreviewShared::new());
        let error = Arc::new(Mutex::new(None));
        let report = Arc::new(Mutex::new(None));

        let thread_shared = Arc::clone(&shared);
        let thread_error = Arc::clone(&error);
        let thread_report = Arc::clone(&report);
        let src_dims: Dims = source.dims();
        let source_backend = source.backend_name();

        let join = thread::Builder::new()
            .name("fluid-gui-pipeline".to_owned())
            .spawn(move || {
                run_pipeline(
                    source,
                    src_dims,
                    &config,
                    thread_shared,
                    thread_error,
                    thread_report,
                );
            })
            .ok();

        Self {
            shared,
            error,
            report,
            join,
            started: Instant::now(),
            source_backend,
        }
    }

    /// Ask the pipeline to stop.  Returns immediately, the UI keeps drawing.
    pub fn stop(&self) {
        self.shared.request_stop();
    }

    pub fn stop_requested(&self) -> bool {
        self.shared.stop_requested()
    }

    /// How long ago the pipeline was started.
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    /// Name of the capture backend this run was started with.
    pub fn source_backend(&self) -> &'static str {
        self.source_backend
    }

    /// Read the worker state; never blocks for long.
    pub fn status(&self) -> WorkerStatus {
        let (generation, frames) = self.shared.progress();
        let error = self.error.lock().ok().and_then(|guard| guard.clone());
        WorkerStatus {
            generation,
            frames,
            finished: self.is_finished(),
            error,
        }
    }

    /// Newest frame together with its generation, if one was produced.
    pub fn take_frame(&self) -> Option<(u64, Arc<fluid_core::Frame>)> {
        self.shared.take_latest()
    }

    /// The pipeline report, available once the worker has finished.
    pub fn take_report(&self) -> Option<PipelineReport> {
        match self.report.lock() {
            Ok(mut guard) => guard.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        }
    }

    /// True once the worker thread has exited.
    pub fn is_finished(&self) -> bool {
        match &self.join {
            Some(handle) => handle.is_finished(),
            None => false,
        }
    }

    /// Join the worker thread.  Only call this once `is_finished()` is true (or
    /// after `stop()`), otherwise the UI thread would block.
    pub fn join(&mut self) {
        if let Some(handle) = self.join.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for RunHandle {
    fn drop(&mut self) {
        // Never leave a pipeline running if the UI drops the handle.
        self.shared.request_stop();
    }
}

fn run_pipeline(
    mut source: Box<dyn FrameSource>,
    src_dims: Dims,
    config: &FluidConfig,
    shared: Arc<PreviewShared>,
    error: Arc<Mutex<Option<String>>>,
    report: Arc<Mutex<Option<PipelineReport>>>,
) {
    let pipeline_config = PipelineConfig {
        pace: true,
        stats_interval: None,
        verbose: false,
        close_on_finish: true,
        ..Default::default()
    };

    let mut pipeline = match Pipeline::with_config(config, src_dims, pipeline_config) {
        Ok(pipeline) => pipeline,
        Err(err) => {
            set_error(&error, format!("无法创建处理管线：{err}"));
            return;
        }
    };

    let sink_dims = pipeline.out_dims();
    let mut sink = GuiSink::new(sink_dims, Arc::clone(&shared));

    match pipeline.run(&mut *source, &mut sink) {
        Ok(pipeline_report) => {
            if let Some(mut guard) = report_lock(&report) {
                *guard = Some(pipeline_report);
            }
        }
        Err(err) => set_error(&error, format!("处理过程中出错：{err}")),
    }
}

fn report_lock(
    report: &Arc<Mutex<Option<PipelineReport>>>,
) -> Option<std::sync::MutexGuard<'_, Option<PipelineReport>>> {
    match report.lock() {
        Ok(guard) => Some(guard),
        Err(poisoned) => Some(poisoned.into_inner()),
    }
}

fn set_error(slot: &Arc<Mutex<Option<String>>>, message: String) {
    match slot.lock() {
        Ok(mut guard) => *guard = Some(message),
        Err(poisoned) => *poisoned.into_inner() = Some(message),
    }
}
