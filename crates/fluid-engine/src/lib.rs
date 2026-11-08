//! Engine glue for Fluid: connect a frame source to a frame sink, synthesise
//! intermediate frames, upscale, pace the output and collect telemetry.
//!
//! The engine is deliberately policy-light: everything visual is configured
//! through fluid_core::FluidConfig and implemented by the flow / interp /
//! upscale crates. This crate only sequences work and measures it.
//!
//! Typical use: build a Pipeline for a source resolution, then call run with
//! any FrameSource / FrameSink pair and print the returned report.

pub mod framegen;
pub mod pacing;
pub mod pipeline;
pub mod processor;
pub mod stats;

pub use framegen::{
    CpuFrameGen, FrameGenBackend, Generated, GpuFusedFrameGen, make_framegen, make_framegen_for,
};
pub use pacing::{Pacer, fps_period};
pub use pipeline::{Pipeline, PipelineConfig, run};
pub use processor::{FrameProcessor, ProcessOutcome, StageTimings};
pub use stats::{PipelineReport, PipelineStats};
