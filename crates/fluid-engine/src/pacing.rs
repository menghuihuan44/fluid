//! Output pacing: keep presented frames on a steady cadence so generated
//! frames do not arrive in bursts, while never accumulating latency debt.

use std::time::{Duration, Instant};

use fluid_core::sleep_until;

/// A simple deadline pacer. A target of 0 fps disables pacing entirely.
#[derive(Clone, Debug)]
pub struct Pacer {
    target_fps: f32,
    next: Option<Instant>,
    slack: Duration,
}

impl Pacer {
    pub fn new(target_fps: f32) -> Self {
        let fps = if target_fps.is_finite() && target_fps > 1.0 {
            target_fps
        } else {
            0.0
        };
        Self {
            target_fps: fps,
            next: None,
            slack: Duration::ZERO,
        }
    }

    /// Disabled pacer: wait() is a no-op.
    pub fn disabled() -> Self {
        Self::new(0.0)
    }

    pub fn target_fps(&self) -> f32 {
        self.target_fps
    }

    pub fn enabled(&self) -> bool {
        self.target_fps > 0.0
    }

    pub fn period(&self) -> Option<Duration> {
        if self.target_fps > 0.0 {
            Some(Duration::from_secs_f64(1.0 / self.target_fps as f64))
        } else {
            None
        }
    }

    /// Time budget per output frame that the rest of the pipeline may spend.
    pub fn slack(&self) -> Duration {
        self.slack
    }

    pub fn reset(&mut self) {
        self.next = None;
    }

    /// Wait until the next output frame slot. Call once per presented frame.
    pub fn wait(&mut self) {
        let Some(period) = self.period() else {
            return;
        };
        let now = Instant::now();
        let deadline = match self.next {
            None => now + period,
            Some(d) => {
                let candidate = d + period;
                // If we are more than two periods behind, resync instead of
                // trying to catch up with a burst (which would add latency).
                if candidate + period < now + period {
                    now + period
                } else {
                    candidate
                }
            }
        };
        self.next = Some(deadline);
        if deadline > now {
            sleep_until(deadline);
        }
        self.slack = deadline.saturating_duration_since(now);
    }

    /// How far behind schedule the pacer is right now.
    pub fn lag(&self) -> Duration {
        match self.next {
            Some(d) => Instant::now().saturating_duration_since(d),
            None => Duration::ZERO,
        }
    }
}

/// Convert an FPS target to a frame period, ignoring invalid values.
pub fn fps_period(fps: f32) -> Option<Duration> {
    if fps.is_finite() && fps > 0.1 && fps <= 1000.0 {
        Some(Duration::from_secs_f64(1.0 / fps as f64))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_pacer_never_waits() {
        let mut p = Pacer::disabled();
        assert!(!p.enabled());
        let t = Instant::now();
        for _ in 0..5 {
            p.wait();
        }
        assert!(t.elapsed() < Duration::from_millis(50));
    }

    #[test]
    fn enabled_pacer_spaces_calls() {
        let mut p = Pacer::new(100.0);
        assert!(p.enabled());
        assert!(p.period().is_some());
        let t = Instant::now();
        for _ in 0..3 {
            p.wait();
        }
        // Two paced gaps of 10 ms each (the first call only arms the timer).
        assert!(t.elapsed() >= Duration::from_millis(15));
    }

    #[test]
    fn fps_period_rejects_garbage() {
        assert!(fps_period(0.0).is_none());
        assert!(fps_period(f32::NAN).is_none());
        assert!(fps_period(60.0).is_some());
    }
}
