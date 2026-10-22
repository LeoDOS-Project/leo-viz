use crate::config::{BatteryConfig, DutyLimit};
use std::collections::VecDeque;

/// Energy update shared with the viewer. Apply net generation/load before clipping.
pub fn integrate_energy(
    energy: f64,
    capacity: f64,
    generation_w: f64,
    load_w: f64,
    dt: f64,
) -> f64 {
    (energy + (generation_w - load_w) * dt).clamp(0.0, capacity)
}
#[derive(Clone, Debug)]
pub struct RollingDuty {
    config: DutyLimit,
    segments: VecDeque<(f64, f64, f64)>,
}
impl RollingDuty {
    pub fn new(config: DutyLimit) -> Self {
        let mut segments = VecDeque::new();
        if config.initial_utilization > 0.0 {
            segments.push_back((-config.window_s, 0.0, config.initial_utilization));
        }
        Self { config, segments }
    }
    pub fn used(&self, at: f64) -> f64 {
        let cutoff = at - self.config.window_s;
        self.segments
            .iter()
            .map(|(a, b, u)| (b.min(at) - a.max(cutoff)).max(0.0) * u)
            .sum()
    }
    /// For grid-aligned intervals the rolling integral is linear within each step.
    /// Checking both boundary values therefore enforces the cap at every instant.
    pub fn allowance(&self, end: f64, dt: f64) -> f64 {
        (self.config.max_fraction * self.config.window_s - self.used(end)).clamp(0.0, dt)
    }
    pub fn record(&mut self, start: f64, end: f64, busy_s: f64) {
        let cutoff = end - self.config.window_s;
        while self.segments.front().is_some_and(|(_, b, _)| *b <= cutoff) {
            self.segments.pop_front();
        }
        if busy_s <= 0.0 {
            return;
        }
        let u = busy_s / (end - start);
        if let Some((_, last_end, last_u)) = self.segments.back_mut() {
            if *last_end == start && (*last_u - u).abs() < 1e-12 {
                *last_end = end;
                return;
            }
        }
        self.segments.push_back((start, end, u));
    }
}
/// One step's shared energy and average-load budget. All optional loads draw here.
#[derive(Clone, Debug)]
pub struct PowerBudget {
    pub unbounded_end_j: f64,
    pub available_j: f64,
    pub spent_j: f64,
    pub reserve_deficit_j: f64,
}
impl PowerBudget {
    pub fn new(energy: f64, c: &BatteryConfig, sunlit: bool, dt: f64) -> Self {
        let end = energy + ((if sunlit { c.solar_w } else { 0.0 }) - c.idle_w) * dt;
        Self {
            unbounded_end_j: end,
            available_j: (end - c.reserve_j)
                .max(0.0)
                .min((c.max_average_load_w - c.idle_w) * dt),
            spent_j: 0.0,
            reserve_deficit_j: (c.reserve_j - end).max(0.0),
        }
    }
    pub fn permitted_seconds(&self, watts: f64, requested: f64) -> f64 {
        if watts == 0.0 {
            requested
        } else {
            requested.min(self.available_j / watts)
        }
    }
    pub fn spend(&mut self, watts: f64, seconds: f64) {
        let j = watts * seconds;
        debug_assert!(j <= self.available_j + 1e-6);
        self.available_j = (self.available_j - j).max(0.0);
        self.spent_j += j;
    }
    pub fn finish(&self, capacity: f64) -> f64 {
        (self.unbounded_end_j - self.spent_j).clamp(0.0, capacity)
    }
}
