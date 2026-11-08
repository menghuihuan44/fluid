//! Telemetry for the render pipeline: counters, per-stage timings, FPS and
//! a printable / JSON-serialisable report.

use std::fmt;
use std::time::Duration;

use fluid_core::{Dims, FpsMeter, LatencyStats, now_ns};

use crate::processor::ProcessOutcome;

/// Rolling counters and latency histograms for one pipeline run.
#[derive(Clone, Debug)]
pub struct PipelineStats {
    pub frames_in: u64,
    pub frames_out: u64,
    pub interpolated: u64,
    pub scene_cuts: u64,
    pub duplicates: u64,
    pub dropped: u64,
    /// Source frames that were pixel-identical to the previous one and reused
    /// the already generated output (static content fast path).
    pub static_frames: u64,
    pub input_fps: FpsMeter,
    pub output_fps: FpsMeter,
    pub process_ns: LatencyStats,
    pub flow_ns: LatencyStats,
    pub interp_ns: LatencyStats,
    pub upscale_ns: LatencyStats,
    /// Time spent blocked in `FrameSource::next_frame` (capture wait).
    pub source_wait_ns: LatencyStats,
    /// Time spent inside `FrameSink::present` (blit / blit + vsync).
    pub present_ns: LatencyStats,
    pub frame_latency_ns: LatencyStats,
    pub started_ns: i64,
    pub last_frame_ns: i64,
    pub elapsed_ns: u64,
}

impl Default for PipelineStats {
    fn default() -> Self {
        Self::new(1024)
    }
}

impl PipelineStats {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(16);
        Self {
            frames_in: 0,
            frames_out: 0,
            interpolated: 0,
            scene_cuts: 0,
            duplicates: 0,
            dropped: 0,
            static_frames: 0,
            input_fps: FpsMeter::one_second(),
            output_fps: FpsMeter::one_second(),
            process_ns: LatencyStats::new(cap),
            flow_ns: LatencyStats::new(cap),
            interp_ns: LatencyStats::new(cap),
            upscale_ns: LatencyStats::new(cap),
            source_wait_ns: LatencyStats::new(cap),
            present_ns: LatencyStats::new(cap),
            frame_latency_ns: LatencyStats::new(cap),
            started_ns: now_ns(),
            last_frame_ns: 0,
            elapsed_ns: 0,
        }
    }

    pub fn reset(&mut self) {
        // Keep the rolling window size stable across resets: deriving it from the
        // sample count would shrink it to the 16-sample minimum on a fresh run and
        // distort every mean afterwards (the window would be much shorter than the
        // run, while the running sum still covered the whole run).
        let cap = self.process_ns.capacity();
        *self = PipelineStats::new(cap);
    }

    /// Record one processed source frame.
    pub fn record_outcome(&mut self, outcome: &ProcessOutcome, now: i64) {
        self.frames_in += 1;
        self.interpolated += u64::from(outcome.interpolated);
        if outcome.static_frame {
            self.static_frames += 1;
        }
        if outcome.scene_cut {
            self.scene_cuts += 1;
        }
        self.input_fps.tick(now);
        self.last_frame_ns = now;
        self.process_ns.push(outcome.timings.total_ns as i64);
        self.flow_ns.push(outcome.timings.flow_ns as i64);
        self.interp_ns.push(outcome.timings.interp_ns as i64);
        self.upscale_ns.push(outcome.timings.upscale_ns as i64);
    }

    /// Record how long the run blocked waiting for the next source frame.
    pub fn record_source_wait(&mut self, ns: i64) {
        if ns >= 0 {
            self.source_wait_ns.push(ns);
        }
    }

    /// Record how long one sink `present` call took.
    pub fn record_present_time(&mut self, ns: i64) {
        if ns >= 0 {
            self.present_ns.push(ns);
        }
    }

    /// Record one presented frame and its end-to-end latency from capture.
    pub fn record_present(&mut self, latency_ns: i64, now: i64) {
        self.frames_out += 1;
        self.output_fps.tick(now);
        if latency_ns > 0 {
            self.frame_latency_ns.push(latency_ns);
        }
    }

    pub fn finish(&mut self) {
        let now = now_ns();
        self.elapsed_ns = (now - self.started_ns).max(0) as u64;
    }

    pub fn elapsed(&self) -> Duration {
        Duration::from_nanos(self.elapsed_ns)
    }

    pub fn input_fps_value(&self) -> f32 {
        self.input_fps.fps()
    }

    pub fn output_fps_value(&self) -> f32 {
        self.output_fps.fps()
    }

    pub fn mean_process_ms(&self) -> f32 {
        self.process_ns.mean_ns() / 1.0e6
    }

    pub fn p99_process_ms(&self) -> f32 {
        self.process_ns.p99_ns() as f32 / 1.0e6
    }

    /// One-line human readable summary of the run so far.
    pub fn summary(&self) -> String {
        format!(
            "in {:.1} fps | out {:.1} fps | frames {}/{} | interp {} | cuts {} | process {:.2}ms mean, {:.2}ms p99 | latency p50 {:.2}ms p99 {:.2}ms",
            self.input_fps_value(),
            self.output_fps_value(),
            self.frames_in,
            self.frames_out,
            self.interpolated,
            self.scene_cuts,
            self.mean_process_ms(),
            self.p99_process_ms(),
            self.frame_latency_ns.p50_ns() as f32 / 1.0e6,
            self.frame_latency_ns.p99_ns() as f32 / 1.0e6,
        )
    }
}

impl fmt::Display for PipelineStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary())
    }
}

/// Immutable end-of-run report.
#[derive(Clone, Debug)]
pub struct PipelineReport {
    pub stats: PipelineStats,
    pub source: String,
    pub sink: String,
    pub src_dims: Dims,
    pub out_dims: Dims,
    pub multiplier: u32,
    pub filter: String,
    /// Active upscaling backend name ("cpu", "wgpu", "none").
    pub backend: String,
    /// Active frame generation backend name ("cpu", "wgpu-flow", "none").
    pub interp_backend: String,
    pub preset: String,
}

impl PipelineReport {
    pub fn elapsed_s(&self) -> f64 {
        self.stats.elapsed_ns as f64 / 1.0e9
    }

    pub fn realtime_factor(&self) -> f32 {
        let secs = self.elapsed_s();
        if secs <= 0.0 {
            return 0.0;
        }
        self.stats.frames_out as f32 / secs as f32
    }

    /// Render as machine-readable JSON (stable field names for tooling).
    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"source\":\"{}\",",
                "\"sink\":\"{}\",",
                "\"src\":[{},{}],",
                "\"out\":[{},{}],",
                "\"multiplier\":{},",
                "\"filter\":\"{}\",",
                "\"backend\":\"{}\",",
                "\"interp_backend\":\"{}\",",
                "\"preset\":\"{}\",",
                "\"elapsed_s\":{:.6},",
                "\"frames_in\":{},",
                "\"frames_out\":{},",
                "\"interpolated\":{},",
                "\"scene_cuts\":{},",
                "\"duplicates\":{},",
                "\"static_frames\":{},",
                "\"input_fps\":{:.3},",
                "\"output_fps\":{:.3},",
                "\"process_mean_ms\":{:.4},",
                "\"process_p99_ms\":{:.4},",
                "\"flow_mean_ms\":{:.4},",
                "\"interp_mean_ms\":{:.4},",
                "\"upscale_mean_ms\":{:.4},",
                "\"source_wait_mean_ms\":{:.4},",
                "\"present_mean_ms\":{:.4},",
                "\"latency_p50_ms\":{:.4},",
                "\"latency_p99_ms\":{:.4}",
                "}}"
            ),
            escape(&self.source),
            escape(&self.sink),
            self.src_dims.width,
            self.src_dims.height,
            self.out_dims.width,
            self.out_dims.height,
            self.multiplier,
            escape(&self.filter),
            escape(&self.backend),
            escape(&self.interp_backend),
            escape(&self.preset),
            self.elapsed_s(),
            self.stats.frames_in,
            self.stats.frames_out,
            self.stats.interpolated,
            self.stats.scene_cuts,
            self.stats.duplicates,
            self.stats.static_frames,
            self.stats.input_fps_value(),
            self.stats.output_fps_value(),
            self.stats.process_ns.mean_ns() as f64 / 1.0e6,
            self.stats.process_ns.p99_ns() as f64 / 1.0e6,
            self.stats.flow_ns.mean_ns() as f64 / 1.0e6,
            self.stats.interp_ns.mean_ns() as f64 / 1.0e6,
            self.stats.upscale_ns.mean_ns() as f64 / 1.0e6,
            self.stats.source_wait_ns.mean_ns() as f64 / 1.0e6,
            self.stats.present_ns.mean_ns() as f64 / 1.0e6,
            self.stats.frame_latency_ns.p50_ns() as f64 / 1.0e6,
            self.stats.frame_latency_ns.p99_ns() as f64 / 1.0e6,
        )
    }
}

impl fmt::Display for PipelineReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "fluid pipeline report")?;
        writeln!(f, "  source      : {} {:?}", self.source, self.src_dims)?;
        writeln!(f, "  sink        : {} {:?}", self.sink, self.out_dims)?;
        writeln!(f, "  preset      : {}", self.preset)?;
        writeln!(
            f,
            "  upscale     : {} -> {:?} ({})",
            self.src_dims, self.out_dims, self.filter
        )?;
        writeln!(f, "  backend     : {}", self.backend)?;
        writeln!(
            f,
            "  interp      : {}x ({})",
            self.multiplier, self.interp_backend
        )?;
        writeln!(f, "  elapsed     : {:.3} s", self.elapsed_s())?;
        writeln!(
            f,
            "  frames      : {} in, {} out ({} interpolated)",
            self.stats.frames_in, self.stats.frames_out, self.stats.interpolated
        )?;
        writeln!(
            f,
            "  framerate   : in {:.1} fps, out {:.1} fps",
            self.stats.input_fps_value(),
            self.stats.output_fps_value()
        )?;
        writeln!(f, "  scene cuts  : {}", self.stats.scene_cuts)?;
        writeln!(f, "  duplicates  : {}", self.stats.duplicates)?;
        writeln!(
            f,
            "  static      : {} frame(s) reused (identical to the previous frame)",
            self.stats.static_frames
        )?;
        writeln!(
            f,
            "  process     : {:.3} ms mean / {:.3} ms p99",
            self.stats.process_ns.mean_ns() as f64 / 1.0e6,
            self.stats.process_ns.p99_ns() as f64 / 1.0e6
        )?;
        writeln!(
            f,
            "  flow        : {:.3} ms mean",
            self.stats.flow_ns.mean_ns() as f64 / 1.0e6
        )?;
        writeln!(
            f,
            "  interp      : {:.3} ms mean",
            self.stats.interp_ns.mean_ns() as f64 / 1.0e6
        )?;
        writeln!(
            f,
            "  upscale     : {:.3} ms mean",
            self.stats.upscale_ns.mean_ns() as f64 / 1.0e6
        )?;
        writeln!(
            f,
            "  io          : source wait {:.3} ms, present {:.3} ms (mean)",
            self.stats.source_wait_ns.mean_ns() as f64 / 1.0e6,
            self.stats.present_ns.mean_ns() as f64 / 1.0e6
        )?;
        write!(
            f,
            "  latency     : {:.3} ms p50 / {:.3} ms p99",
            self.stats.frame_latency_ns.p50_ns() as f64 / 1.0e6,
            self.stats.frame_latency_ns.p99_ns() as f64 / 1.0e6
        )
    }
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push(' '),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_is_printable() {
        let s = PipelineStats::new(32);
        assert!(s.summary().contains("fps"));
        assert!(s.to_string().contains("frames"));
    }

    #[test]
    fn reset_keeps_window_capacity_and_means_stay_consistent() {
        let mut s = PipelineStats::new(16);
        let cap = s.process_ns.capacity();
        s.process_ns.push(5_000_000);
        s.reset();
        assert_eq!(s.process_ns.capacity(), cap);
        // 100 samples with a window of 16: the mean must be the mean of the last
        // 16 samples and can never exceed the largest sample.
        for i in 0..100i64 {
            s.process_ns.push(i * 1_000_000);
        }
        let mean = s.process_ns.mean_ns();
        let start = 100i64 - cap as i64;
        let expected: f32 = (start..100).map(|i| i as f32 * 1_000_000.0).sum::<f32>() / cap as f32;
        assert!(
            (mean - expected).abs() < 1.0,
            "mean {mean} expected {expected}"
        );
        assert!(mean <= s.process_ns.max_ns() as f32);
    }

    #[test]
    fn report_json_roundtrips_roughly() {
        let stats = PipelineStats::new(8);
        let r = PipelineReport {
            stats,
            source: "mock\"quoted\"".to_string(),
            sink: "null".to_string(),
            src_dims: Dims::new(320, 180),
            out_dims: Dims::new(640, 360),
            multiplier: 2,
            filter: "bilinear".to_string(),
            backend: "cpu".to_string(),
            interp_backend: "cpu".to_string(),
            preset: "balanced".to_string(),
        };
        let js = r.to_json();
        assert!(js.starts_with('{') && js.ends_with('}'));
        assert!(js.contains("\"src\":[320,180]"));
        assert!(js.contains("mock\\\"quoted\\\""));
        let parsed: serde_json::Value = serde_json::from_str(&js).expect("valid json");
        assert_eq!(parsed["multiplier"], 2);
    }
}
