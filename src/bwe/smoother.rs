use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::rtp_::Bitrate;

/// Time window for averaging estimate changes.
const ESTIMATE_WINDOW: Duration = Duration::from_secs(3);

type TimeBitrate = (Instant, Bitrate);

const TOLERANCE: f64 = 0.05;

/// Smooths BWE estimates by averaging over a time window.
pub struct EstimateSmoother {
    estimates: VecDeque<TimeBitrate>,
    maybe_emit: bool,
    emitted: Option<Bitrate>,
}

impl EstimateSmoother {
    pub fn new() -> Self {
        Self {
            estimates: VecDeque::new(),
            maybe_emit: false,
            emitted: None,
        }
    }

    /// Record a new estimate and update the smoothed average.
    pub fn record(&mut self, now: Instant, estimate: Bitrate) {
        self.record_for(now, estimate, ESTIMATE_WINDOW);
    }

    pub fn record_for(&mut self, now: Instant, estimate: Bitrate, window: Duration) {
        // Did value change from previous?
        let do_update = self.estimates.back().map(|b| b.1) != Some(estimate);

        if do_update {
            // Keep increases smoothed. A material drop starts a new window
            // so an encoder can follow the reduced pacing rate immediately.
            if self
                .emitted
                .is_some_and(|rate| estimate < rate * (1.0 - TOLERANCE))
            {
                self.estimates.clear();
            }
            self.maybe_emit = true;
            self.estimates.push_back((now, estimate));
        }

        // Remove entries older than the window.
        while let Some((time, _)) = self.estimates.front() {
            if now.duration_since(*time) > window {
                // Keep last entry
                if self.estimates.len() == 1 {
                    break;
                }

                self.maybe_emit = true;
                self.estimates.pop_front();
            } else {
                break;
            }
        }
    }

    /// Poll for an estimate to emit. Returns Some only when there's a new value to emit.
    pub fn poll(&mut self) -> Option<Bitrate> {
        if !self.maybe_emit {
            return None;
        }

        if self.estimates.is_empty() {
            return None;
        }

        let total: f64 = self.estimates.iter().map(|b| b.1.as_f64()).sum();
        let avg = total / self.estimates.len() as f64;
        let rate: Bitrate = avg.into();

        // This forces emitting if we have the first ever value, or a last
        // where the rest of the window is gone (estimates stop coming).
        let force = self.estimates.len() == 1 && self.emitted != Some(rate);

        // Emit if we deviate enough from previously emitted.
        let deviate = if let Some(emitted) = self.emitted {
            !in_tolerance(emitted, rate)
        } else {
            true
        };

        self.maybe_emit = false;

        // Are we not to emit?
        if !force && !deviate {
            return None;
        }

        self.emitted = Some(rate);
        Some(rate)
    }
}

fn in_tolerance(b1: Bitrate, b2: Bitrate) -> bool {
    let min = b1 * (1.0 - TOLERANCE);
    let max = b1 * (1.0 + TOLERANCE);
    b2 > min && b2 < max
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_decreases_emit_before_the_old_window_expires() {
        let mut smoother = EstimateSmoother::new();
        let now = Instant::now();
        smoother.record(now, Bitrate::bps(56_000_000));
        assert_eq!(smoother.poll(), Some(Bitrate::bps(56_000_000)));
        smoother.record(now + Duration::from_millis(100), Bitrate::bps(64_000_000));
        assert_eq!(smoother.poll(), Some(Bitrate::bps(60_000_000)));
        smoother.record(now + Duration::from_millis(200), Bitrate::bps(9_000_000));
        assert_eq!(smoother.poll(), Some(Bitrate::bps(9_000_000)));
        smoother.record(now + Duration::from_millis(300), Bitrate::bps(15_000_000));
        assert_eq!(smoother.poll(), Some(Bitrate::bps(12_000_000)));
    }
}
