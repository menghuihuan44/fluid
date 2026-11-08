//! Frame, frame-rate and latency metrics.

use std::collections::VecDeque;

use crate::frame::{Frame, LumaPlane};

/// Simple statistics over a luma plane, used for scene cut detection and tests.
#[derive(Clone, Debug)]
pub struct FrameStats {
    pub mean: f32,
    pub variance: f32,
    pub min: u8,
    pub max: u8,
    pub histogram: [u32; 256],
}

impl FrameStats {
    pub fn from_luma(plane: &LumaPlane) -> Self {
        let mut histogram = [0u32; 256];
        let mut min = 255u8;
        let mut max = 0u8;
        let mut sum = 0f64;
        for &v in &plane.data {
            let q = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            histogram[q as usize] += 1;
            min = min.min(q);
            max = max.max(q);
            sum += v as f64;
        }
        let n = plane.data.len().max(1);
        let mean = (sum / n as f64) as f32;
        let mut var = 0f64;
        for &v in &plane.data {
            let d = v as f64 - sum / n as f64;
            var += d * d;
        }
        Self {
            mean,
            variance: (var / n as f64) as f32,
            min,
            max,
            histogram,
        }
    }

    pub fn from_frame(frame: &Frame) -> Self {
        Self::from_luma(&frame.to_luma())
    }

    /// Mean absolute difference between two luma planes (same dimensions).
    pub fn mean_abs_diff(a: &LumaPlane, b: &LumaPlane) -> f32 {
        let n = a.data.len().min(b.data.len());
        if n == 0 {
            return 0.0;
        }
        let mut acc = 0f64;
        for i in 0..n {
            acc += (a.data[i] - b.data[i]).abs() as f64;
        }
        (acc / n as f64) as f32
    }

    /// Peak signal to noise ratio in dB between two luma planes.
    pub fn psnr(a: &LumaPlane, b: &LumaPlane) -> f32 {
        let n = a.data.len().min(b.data.len());
        if n == 0 {
            return f32::INFINITY;
        }
        let mut mse = 0f64;
        for i in 0..n {
            let d = (a.data[i] - b.data[i]) as f64;
            mse += d * d;
        }
        mse /= n as f64;
        if mse <= 1e-12 {
            return f32::INFINITY;
        }
        (-10.0 * (mse.log10())) as f32
    }
}

/// Rolling frame-rate meter.
#[derive(Clone, Debug)]
pub struct FpsMeter {
    window_ns: i64,
    stamps: VecDeque<i64>,
}

impl FpsMeter {
    pub fn new(window_ns: i64) -> Self {
        Self {
            window_ns: window_ns.max(1),
            stamps: VecDeque::new(),
        }
    }

    /// One second rolling window.
    pub fn one_second() -> Self {
        Self::new(1_000_000_000)
    }

    pub fn tick(&mut self, now_ns: i64) {
        self.stamps.push_back(now_ns);
        let horizon = now_ns - self.window_ns;
        while let Some(&front) = self.stamps.front() {
            if front < horizon {
                self.stamps.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn fps(&self) -> f32 {
        if self.stamps.len() < 2 {
            return 0.0;
        }
        let first = *self.stamps.front().unwrap();
        let last = *self.stamps.back().unwrap();
        let span = last - first;
        if span <= 0 {
            return 0.0;
        }
        (self.stamps.len() - 1) as f32 * 1e9 / span as f32
    }

    pub fn reset(&mut self) {
        self.stamps.clear();
    }

    pub fn samples(&self) -> usize {
        self.stamps.len()
    }
}

/// Percentile latency tracker (nanoseconds) over a fixed ring buffer.
///
/// The buffer keeps the most recent `capacity` samples; `count()` keeps counting
/// every push since the last `reset()`. `mean_ns()` is the mean of the *window*
/// (not of all history), which is why evicted samples are subtracted from the
/// running sum on every push.
#[derive(Clone, Debug)]
pub struct LatencyStats {
    capacity: usize,
    samples: std::collections::VecDeque<i64>,
    total: u64,
    sum_ns: i128,
    max_ns: i64,
}

impl LatencyStats {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(2);
        Self {
            capacity: cap,
            samples: std::collections::VecDeque::with_capacity(cap),
            total: 0,
            sum_ns: 0,
            max_ns: 0,
        }
    }

    /// Number of samples the rolling window can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn push(&mut self, ns: i64) {
        let ns = ns.max(0);
        if self.samples.len() == self.capacity
            && let Some(evicted) = self.samples.pop_front()
        {
            self.sum_ns -= evicted as i128;
        }
        self.samples.push_back(ns);
        self.sum_ns += ns as i128;
        self.total += 1;
        self.max_ns = self.max_ns.max(ns);
    }

    pub fn count(&self) -> u64 {
        self.total
    }

    pub fn mean_ns(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        (self.sum_ns as f64 / self.samples.len() as f64) as f32
    }

    pub fn percentile_ns(&self, p: f32) -> i64 {
        let mut sorted: Vec<i64> = self.samples.iter().copied().collect();
        sorted.sort_unstable();
        crate::math::percentile_i64(&sorted, p / 100.0)
    }

    pub fn p50_ns(&self) -> i64 {
        self.percentile_ns(50.0)
    }

    pub fn p95_ns(&self) -> i64 {
        self.percentile_ns(95.0)
    }

    pub fn p99_ns(&self) -> i64 {
        self.percentile_ns(99.0)
    }

    pub fn max_ns(&self) -> i64 {
        self.max_ns
    }

    pub fn reset(&mut self) {
        self.samples.clear();
        self.total = 0;
        self.sum_ns = 0;
        self.max_ns = 0;
    }

    /// One line human readable summary.
    pub fn summary(&self) -> String {
        format!(
            "mean {:.2}ms  p50 {:.2}ms  p95 {:.2}ms  p99 {:.2}ms  max {:.2}ms",
            self.mean_ns() / 1e6,
            self.p50_ns() as f64 / 1e6,
            self.p95_ns() as f64 / 1e6,
            self.p99_ns() as f64 / 1e6,
            self.max_ns() as f64 / 1e6
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Dims;

    #[test]
    fn frame_stats_basic() {
        let mut p = LumaPlane::zeros(Dims::new(4, 4));
        for y in 0..4 {
            for x in 0..4 {
                p.set(x, y, if x == 0 { 0.0 } else { 1.0 });
            }
        }
        let s = FrameStats::from_luma(&p);
        assert_eq!(s.min, 0);
        assert_eq!(s.max, 255);
        assert!((s.mean - 0.75).abs() < 1e-5);
        assert_eq!(s.histogram[255], 12);
    }

    #[test]
    fn fps_meter_counts() {
        let mut m = FpsMeter::one_second();
        for i in 0..=60 {
            m.tick(i as i64 * 16_666_667);
        }
        let fps = m.fps();
        assert!((fps - 60.0).abs() < 1.0, "fps was {fps}");
    }

    #[test]
    fn latency_rolling_mean_uses_window() {
        let mut l = LatencyStats::new(4);
        for i in 1..=10 {
            l.push(i * 1_000_000);
        }
        // Window holds 7,8,9,10 ms => mean 8.5 ms, not the 5.5 ms of all history.
        assert_eq!(l.count(), 10);
        assert!(
            (l.mean_ns() - 8_500_000.0).abs() < 1.0,
            "mean {}",
            l.mean_ns()
        );
        assert!(l.mean_ns() <= l.max_ns() as f32);
        assert_eq!(l.p50_ns(), 8_500_000);
    }

    #[test]
    fn latency_percentiles() {
        let mut l = LatencyStats::new(100);
        for i in 1..=100 {
            l.push(i * 1_000_000);
        }
        assert_eq!(l.count(), 100);
        assert!((l.mean_ns() - 50_500_000.0).abs() < 1.0);
        assert!(l.p99_ns() >= 99_000_000);
        assert_eq!(l.max_ns(), 100_000_000);
    }
}
