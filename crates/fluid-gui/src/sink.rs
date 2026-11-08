//! The GUI's own `FrameSink`: it keeps only the newest frame for the preview
//! and counts presented frames / output fps.  All shared state is behind
//! `Arc<Mutex<_>>` or atomics so the UI thread can read it without blocking
//! the pipeline for more than a moment.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use fluid_core::{Dims, FluidError, FpsMeter, Frame, FrameSink, now_ns};

/// State shared between the pipeline worker thread and the UI thread.
pub struct PreviewShared {
    /// Newest presented frame (the UI clones the `Arc` while holding the lock).
    pub latest: Mutex<Option<Arc<Frame>>>,
    /// Incremented every time a new frame is stored.
    pub generation: AtomicU64,
    /// Number of frames presented since the worker started.
    pub presented: AtomicU64,
    /// Set by the UI (or by the pipeline when it wants to stop the preview).
    pub stop: AtomicBool,
}

impl PreviewShared {
    pub fn new() -> Self {
        Self {
            latest: Mutex::new(None),
            generation: AtomicU64::new(0),
            presented: AtomicU64::new(0),
            stop: AtomicBool::new(false),
        }
    }

    /// Stop the pipeline at its next check.
    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    pub fn stop_requested(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    /// Current frame generation and presented frame count.
    pub fn progress(&self) -> (u64, u64) {
        (
            self.generation.load(Ordering::SeqCst),
            self.presented.load(Ordering::SeqCst),
        )
    }

    /// Clone the newest frame, if any.
    pub fn take_latest(&self) -> Option<(u64, Arc<Frame>)> {
        let generation = self.generation.load(Ordering::SeqCst);
        let guard = match self.latest.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.as_ref().map(|frame| (generation, Arc::clone(frame)))
    }

    pub fn clear(&self) {
        if let Ok(mut guard) = self.latest.lock() {
            *guard = None;
        }
        self.generation.store(0, Ordering::SeqCst);
        self.presented.store(0, Ordering::SeqCst);
        self.stop.store(false, Ordering::SeqCst);
    }
}

impl Default for PreviewShared {
    fn default() -> Self {
        Self::new()
    }
}

/// `FrameSink` implementation that feeds the GUI preview.
pub struct GuiSink {
    dims: Dims,
    shared: Arc<PreviewShared>,
    fps: FpsMeter,
}

impl GuiSink {
    pub fn new(dims: Dims, shared: Arc<PreviewShared>) -> Self {
        Self {
            dims,
            shared,
            fps: FpsMeter::one_second(),
        }
    }

    /// Output fps, measured on the frames this sink has seen.
    pub fn fps(&self) -> f32 {
        self.fps.fps()
    }

    pub fn frames_presented(&self) -> u64 {
        self.shared.presented.load(Ordering::SeqCst)
    }
}

impl FrameSink for GuiSink {
    fn dims(&self) -> Dims {
        self.dims
    }

    fn backend_name(&self) -> &'static str {
        "gui-preview"
    }

    fn present(&mut self, frame: &Frame) -> fluid_core::Result<()> {
        if frame.dims != self.dims {
            return Err(FluidError::InvalidArgument(format!(
                "gui-preview: frame is {}x{} but the sink expects {}x{}",
                frame.dims.width, frame.dims.height, self.dims.width, self.dims.height
            )));
        }
        self.fps.tick(now_ns());
        let frame = Arc::new(frame.clone());
        {
            let mut guard = match self.shared.latest.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            *guard = Some(frame);
        }
        self.shared.generation.fetch_add(1, Ordering::SeqCst);
        self.shared.presented.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn should_close(&self) -> bool {
        self.shared.stop_requested()
    }

    fn close(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluid_core::PixelFormat;

    fn frame(dims: Dims) -> Frame {
        Frame::new(
            dims,
            PixelFormat::Rgb8,
            vec![7; dims.width as usize * dims.height as usize * 3],
        )
        .unwrap()
    }

    #[test]
    fn backend_name_is_gui_preview() {
        let sink = GuiSink::new(Dims::new(4, 4), Arc::new(PreviewShared::new()));
        assert_eq!(sink.backend_name(), "gui-preview");
        assert_eq!(sink.dims(), Dims::new(4, 4));
    }

    #[test]
    fn present_stores_the_newest_frame_and_counts() {
        let shared = Arc::new(PreviewShared::new());
        let mut sink = GuiSink::new(Dims::new(4, 4), Arc::clone(&shared));
        sink.present(&frame(Dims::new(4, 4))).unwrap();
        sink.present(&frame(Dims::new(4, 4))).unwrap();

        let (generation, presented) = shared.progress();
        assert_eq!(generation, 2);
        assert_eq!(presented, 2);
        assert_eq!(sink.frames_presented(), 2);

        let (stored_generation, stored) = shared.take_latest().expect("a frame was stored");
        assert_eq!(stored_generation, 2);
        assert_eq!(stored.dims, Dims::new(4, 4));
    }

    #[test]
    fn wrong_dimensions_are_rejected() {
        let shared = Arc::new(PreviewShared::new());
        let mut sink = GuiSink::new(Dims::new(4, 4), Arc::clone(&shared));
        let err = sink.present(&frame(Dims::new(5, 5))).unwrap_err();
        assert!(matches!(err, FluidError::InvalidArgument(_)));
        assert_eq!(shared.progress(), (0, 0));
    }

    #[test]
    fn should_close_follows_the_stop_flag() {
        let shared = Arc::new(PreviewShared::new());
        let sink = GuiSink::new(Dims::new(2, 2), Arc::clone(&shared));
        assert!(!sink.should_close());
        shared.request_stop();
        assert!(sink.should_close());
    }

    #[test]
    fn clear_resets_the_counters() {
        let shared = Arc::new(PreviewShared::new());
        let mut sink = GuiSink::new(Dims::new(2, 2), Arc::clone(&shared));
        sink.present(&frame(Dims::new(2, 2))).unwrap();
        shared.clear();
        assert_eq!(shared.progress(), (0, 0));
        assert!(shared.take_latest().is_none());
        assert!(!shared.stop_requested());
    }
}
