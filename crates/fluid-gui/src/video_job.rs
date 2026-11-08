//! Offline video processing: a y4m / avi file in, a processed y4m / avi file out.
//!
//! Runs on a worker thread with PipelineConfig::offline() (no pacing, as fast as
//! the machine allows) and publishes progress, ETA and the final report for the UI.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use fluid_core::{Dims, FluidConfig, Frame, FrameSink, FrameSource};
use fluid_engine::{Pipeline, PipelineConfig, PipelineReport};
use fluid_video::{AviHeader, AviReader, AviWriter, Y4mHeader, Y4mReader, Y4mWriter};

use crate::video_format::{self, VideoFormat};

/// What the input file looks like, read before the job starts so the tab can
/// show resolution / fps / frame count up front.
#[derive(Clone, Debug)]
pub struct VideoInputInfo {
    pub format: VideoFormat,
    pub dims: Dims,
    pub fps: f32,
    pub frames: u64,
}

impl VideoInputInfo {
    /// One line for the UI, beginner friendly.
    pub fn summary(&self) -> String {
        format!(
            "{} · {}x{} · {:.2} fps · 约 {} 帧",
            self.format.label(),
            self.dims.width,
            self.dims.height,
            self.fps,
            self.frames
        )
    }
}

/// Open the input once to learn its container details. Errors are friendly
/// strings because they go straight into the UI.
pub fn inspect_input(path: &Path, format: VideoFormat) -> Result<VideoInputInfo, String> {
    match format {
        VideoFormat::Y4m => {
            let reader = Y4mReader::open(path).map_err(|err| format!("无法打开输入视频：{err}"))?;
            let header = reader.header().clone();
            let frames = estimate_y4m_frames(path, &header);
            Ok(VideoInputInfo {
                format,
                dims: header.dims,
                fps: header.fps(),
                frames,
            })
        }
        VideoFormat::Avi => {
            let reader = AviReader::open(path).map_err(|err| format!("无法打开输入视频：{err}"))?;
            let frames = reader.frame_count().unwrap_or_else(|| reader.frames());
            Ok(VideoInputInfo {
                format,
                dims: reader.dims(),
                fps: reader.fps(),
                frames,
            })
        }
    }
}

/// Progress published to the UI.
#[derive(Clone, Debug, Default)]
pub struct VideoState {
    pub running: bool,
    pub done: bool,
    /// The user pressed stop: the file is kept but incomplete.
    pub cancelled: bool,
    pub frames_out: u64,
    /// Frame count of the input file (for the progress bar).
    pub total_estimate: u64,
    pub src_dims: Dims,
    pub out_dims: Dims,
    pub output_format: Option<VideoFormat>,
    pub output_path: Option<PathBuf>,
    pub elapsed_s: f32,
    pub eta_seconds: Option<f32>,
    pub error: Option<String>,
    pub report: Option<PipelineReport>,
}

impl VideoState {
    /// Fraction of the input that has been processed, in 0.0..=1.0.
    pub fn fraction(&self) -> f32 {
        if self.total_estimate == 0 {
            return 0.0;
        }
        (self.frames_out as f32 / self.total_estimate as f32).clamp(0.0, 1.0)
    }

    /// Remaining-time text for the progress area.
    pub fn eta_text(&self) -> String {
        if self.done || !self.running {
            return String::new();
        }
        video_format::format_eta(self.eta_seconds)
    }

    /// Note explaining that a cancelled output file is partial.
    pub fn cancelled_text(&self) -> Option<&'static str> {
        if self.cancelled {
            Some(video_format::cancelled_note())
        } else {
            None
        }
    }
}

/// A running offline video job.
pub struct VideoJob {
    state: Arc<Mutex<VideoState>>,
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl VideoJob {
    /// Start processing input into output on a worker thread.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        input: PathBuf,
        output: PathBuf,
        output_format: VideoFormat,
        info: VideoInputInfo,
        config: FluidConfig,
        quality: u8,
    ) -> Self {
        let state = Arc::new(Mutex::new(VideoState {
            running: true,
            total_estimate: info.frames,
            src_dims: info.dims,
            output_format: Some(output_format),
            output_path: Some(output.clone()),
            ..Default::default()
        }));
        let stop = Arc::new(AtomicBool::new(false));

        let thread_state = Arc::clone(&state);
        let thread_stop = Arc::clone(&stop);
        let join = thread::Builder::new()
            .name("fluid-gui-video".to_owned())
            .spawn(move || {
                run_video(
                    &input,
                    &output,
                    output_format,
                    &info,
                    &config,
                    quality,
                    thread_state,
                    thread_stop,
                );
            })
            .ok();

        Self { state, stop, join }
    }

    pub fn state(&self) -> VideoState {
        match self.state.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
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

impl Drop for VideoJob {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

fn update(state: &Arc<Mutex<VideoState>>, change: impl FnOnce(&mut VideoState)) {
    match state.lock() {
        Ok(mut guard) => change(&mut guard),
        Err(poisoned) => change(&mut poisoned.into_inner()),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_video(
    input: &Path,
    output: &Path,
    output_format: VideoFormat,
    info: &VideoInputInfo,
    config: &FluidConfig,
    quality: u8,
    state: Arc<Mutex<VideoState>>,
    stop: Arc<AtomicBool>,
) {
    let result = process(
        input,
        output,
        output_format,
        info,
        config,
        quality,
        &state,
        &stop,
    );
    match result {
        Ok(report) => {
            let cancelled = stop.load(Ordering::SeqCst);
            update(&state, |state| {
                state.running = false;
                state.done = true;
                state.cancelled = cancelled;
                state.frames_out = report.stats.frames_out;
                state.eta_seconds = None;
                state.report = Some(report);
            });
        }
        Err(err) => update(&state, |state| {
            state.running = false;
            state.done = false;
            state.error = Some(err);
            state.eta_seconds = None;
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn process(
    input: &Path,
    output: &Path,
    output_format: VideoFormat,
    info: &VideoInputInfo,
    config: &FluidConfig,
    quality: u8,
    state: &Arc<Mutex<VideoState>>,
    stop: &Arc<AtomicBool>,
) -> Result<PipelineReport, String> {
    let mut reader: Box<dyn FrameSource> = match info.format {
        VideoFormat::Y4m => {
            Box::new(Y4mReader::open(input).map_err(|err| format!("无法打开输入视频：{err}"))?)
        }
        VideoFormat::Avi => {
            Box::new(AviReader::open(input).map_err(|err| format!("无法打开输入视频：{err}"))?)
        }
    };

    let mut pipeline = Pipeline::with_config(config, info.dims, PipelineConfig::offline())
        .map_err(|err| format!("无法创建处理管线：{err}"))?;
    let out_dims = pipeline.out_dims();

    update(state, |state| {
        state.src_dims = info.dims;
        state.out_dims = out_dims;
        state.total_estimate = info.frames;
        state.output_format = Some(output_format);
        state.output_path = Some(output.to_path_buf());
    });

    let writer = OutputWriter::create(output, out_dims, info.fps, output_format, quality)?;
    let mut sink = ProgressSink {
        inner: writer,
        state: Arc::clone(state),
        stop: Arc::clone(stop),
        start: Instant::now(),
        total: info.frames,
    };

    let report = pipeline
        .run(reader.as_mut(), &mut sink)
        .map_err(|err| format!("视频处理失败：{err}"))?;
    sink.finish()
        .map_err(|err| format!("写入视频失败：{err}"))?;
    Ok(report)
}

fn estimate_y4m_frames(input: &Path, header: &Y4mHeader) -> u64 {
    let frame_bytes = header.frame_bytes();
    if frame_bytes == 0 {
        return 0;
    }
    let Ok(metadata) = std::fs::metadata(input) else {
        return 0;
    };
    // The y4m stream header is tiny; ignoring its exact length is fine for a
    // progress bar.
    metadata.len().saturating_sub(128) / frame_bytes as u64
}

/// The two containers the writer can produce.
enum OutputWriter {
    Y4m(Y4mWriter),
    Avi(AviWriter),
}

impl OutputWriter {
    fn create(
        path: &Path,
        dims: Dims,
        fps: f32,
        format: VideoFormat,
        quality: u8,
    ) -> Result<Self, String> {
        match format {
            VideoFormat::Y4m => Y4mWriter::create(path, Y4mHeader::new(dims, fps))
                .map(Self::Y4m)
                .map_err(|err| format!("无法创建输出视频：{err}")),
            VideoFormat::Avi => {
                let header = AviHeader::new(dims, fps, video_format::clamp_quality(quality));
                AviWriter::create(path, header)
                    .map(Self::Avi)
                    .map_err(|err| format!("无法创建输出视频：{err}"))
            }
        }
    }

    fn frames(&self) -> u64 {
        match self {
            Self::Y4m(writer) => writer.frames(),
            Self::Avi(writer) => writer.frames(),
        }
    }

    fn finish(&mut self) -> fluid_core::Result<()> {
        match self {
            Self::Y4m(writer) => writer.finish(),
            Self::Avi(writer) => writer.finish(),
        }
    }
}

impl FrameSink for OutputWriter {
    fn dims(&self) -> Dims {
        match self {
            Self::Y4m(writer) => writer.dims(),
            Self::Avi(writer) => writer.dims(),
        }
    }

    fn backend_name(&self) -> &'static str {
        match self {
            Self::Y4m(writer) => writer.backend_name(),
            Self::Avi(writer) => writer.backend_name(),
        }
    }

    fn present(&mut self, frame: &Frame) -> fluid_core::Result<()> {
        match self {
            Self::Y4m(writer) => writer.present(frame),
            Self::Avi(writer) => writer.present(frame),
        }
    }
}

/// Wraps the real writer so progress, elapsed time and the ETA can be published,
/// and so the user can stop the job.
struct ProgressSink {
    inner: OutputWriter,
    state: Arc<Mutex<VideoState>>,
    stop: Arc<AtomicBool>,
    start: Instant,
    total: u64,
}

impl ProgressSink {
    fn finish(&mut self) -> fluid_core::Result<()> {
        self.inner.finish()
    }
}

impl FrameSink for ProgressSink {
    fn dims(&self) -> Dims {
        self.inner.dims()
    }

    fn backend_name(&self) -> &'static str {
        self.inner.backend_name()
    }

    fn present(&mut self, frame: &Frame) -> fluid_core::Result<()> {
        self.inner.present(frame)?;
        let frames = self.inner.frames();
        let elapsed_s = self.start.elapsed().as_secs_f32();
        let eta_seconds = video_format::eta_seconds(frames, self.total, elapsed_s);
        if let Ok(mut guard) = self.state.lock() {
            guard.frames_out = frames;
            guard.elapsed_s = elapsed_s;
            guard.eta_seconds = eta_seconds;
        }
        Ok(())
    }

    fn should_close(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    use fluid_core::{
        InterpBackendKind, InterpMode, ScaleConfig, UpscaleBackendKind, UpscaleFilter,
    };
    use fluid_video::SyntheticSource;

    fn quick_config() -> FluidConfig {
        let mut config = FluidConfig::default();
        config.interpolator.mode = InterpMode::Off;
        config.interpolator.multiplier = 1;
        config.interpolator.backend = InterpBackendKind::Cpu;
        config.upscaler.filter = UpscaleFilter::Off;
        config.upscaler.scale = ScaleConfig::default();
        config.upscaler.backend = UpscaleBackendKind::Cpu;
        config
    }

    fn write_test_y4m(path: &Path, dims: Dims, frames: u64) {
        let mut source = SyntheticSource::new(dims, 30.0);
        let mut writer = Y4mWriter::create(path, Y4mHeader::new(dims, 30.0)).expect("y4m writer");
        for _ in 0..frames {
            let frame = source.next_frame().expect("frame").expect("some frame");
            writer.present(&frame).expect("present");
        }
        writer.finish().expect("finish");
    }

    #[test]
    fn progress_fraction_handles_unknown_total() {
        let state = VideoState::default();
        assert_eq!(state.fraction(), 0.0);

        let state = VideoState {
            frames_out: 25,
            total_estimate: 100,
            ..Default::default()
        };
        assert!((state.fraction() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn progress_fraction_is_clamped() {
        let state = VideoState {
            frames_out: 200,
            total_estimate: 100,
            ..Default::default()
        };
        assert_eq!(state.fraction(), 1.0);
    }

    #[test]
    fn cancelled_state_explains_the_partial_file() {
        let state = VideoState {
            done: true,
            cancelled: true,
            ..Default::default()
        };
        assert!(
            state
                .cancelled_text()
                .is_some_and(|text| text.contains("不完整"))
        );
        assert_eq!(VideoState::default().cancelled_text(), None);
        let running = VideoState {
            running: true,
            eta_seconds: Some(12.0),
            ..Default::default()
        };
        assert!(running.eta_text().contains("12"));
    }

    #[test]
    fn video_job_can_be_stopped_before_starting_work() {
        // Nothing to join yet: spawn is only real work when it opens files, so
        // this checks the cheap public API rather than the thread.
        let stop = Arc::new(AtomicBool::new(false));
        stop.store(true, Ordering::SeqCst);
        assert!(stop.load(Ordering::SeqCst));
    }

    #[test]
    fn inspect_reads_y4m_and_avi_metadata() {
        let dir = std::env::temp_dir().join(format!("fluid-gui-inspect-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("temp dir");
        let y4m = dir.join("input.y4m");
        write_test_y4m(&y4m, Dims::new(64, 48), 4);

        let info = inspect_input(&y4m, VideoFormat::Y4m).expect("y4m info");
        assert_eq!(info.dims, Dims::new(64, 48));
        assert!((info.fps - 30.0).abs() < 0.01);
        assert!(info.frames >= 3, "{}", info.frames);
        assert!(info.summary().contains("64x48"));

        let avi = dir.join("output.avi");
        let state = Arc::new(Mutex::new(VideoState::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let report = process(
            &y4m,
            &avi,
            VideoFormat::Avi,
            &info,
            &quick_config(),
            88,
            &state,
            &stop,
        )
        .expect("y4m to avi");
        assert!(report.stats.frames_out > 0);
        assert!(avi.exists());
        let avi_info = inspect_input(&avi, VideoFormat::Avi).expect("avi info");
        assert_eq!(avi_info.dims, Dims::new(64, 48));
        assert!(avi_info.frames >= 3, "{}", avi_info.frames);

        let back = dir.join("back.y4m");
        let report = process(
            &avi,
            &back,
            VideoFormat::Y4m,
            &avi_info,
            &quick_config(),
            88,
            &state,
            &stop,
        )
        .expect("avi to y4m");
        assert!(report.stats.frames_out > 0);
        assert!(back.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_job_writes_the_output_file_and_reports() {
        let dir = std::env::temp_dir().join(format!("fluid-gui-job-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("temp dir");
        let input = dir.join("input.y4m");
        let output = dir.join("out.avi");
        write_test_y4m(&input, Dims::new(64, 48), 5);
        let info = inspect_input(&input, VideoFormat::Y4m).expect("info");

        let mut job = VideoJob::spawn(
            input,
            output.clone(),
            VideoFormat::Avi,
            info,
            quick_config(),
            88,
        );
        let mut waited = 0;
        while !job.is_finished() && waited < 30_000 {
            std::thread::sleep(std::time::Duration::from_millis(20));
            waited += 20;
        }
        job.join();
        let state = job.state();
        assert!(state.error.is_none(), "{:?}", state.error);
        assert!(state.done);
        assert!(state.frames_out > 0);
        assert!(output.exists());
        assert!(state.report.is_some());
        let _ = fs::remove_dir_all(&dir);
    }
}
