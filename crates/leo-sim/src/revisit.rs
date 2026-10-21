//! Geometric pushbroom opportunities at sampled ground points, independent of
//! rendering and propagation. Short great-circle arcs interpolate ground tracks
//! so a narrow sweep cannot fall entirely between temporal samples.
use crate::geometry::{dot, ground_position, norm, scale, sub, Vector};

fn cross(a: Vector, b: Vector) -> Vector {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub fn unit(p: Vector) -> Vector {
    scale(p, 1.0 / norm(p))
}

#[derive(Clone, Copy, Debug)]
pub struct GroundPoint {
    pub lat: f64,
    pub lon: f64,
}
impl GroundPoint {
    pub fn xyz(self) -> Vector {
        ground_position(self.lat, self.lon, 1.0, 0.0)
    }
}

/// Return the fraction of the arc when its perpendicular scanline crosses p.
/// No rounded end caps: those would incorrectly add exposures between steps.
pub fn crossing(a: Vector, b: Vector, p: Vector, half_width: f64) -> Option<f64> {
    let normal = cross(a, b);
    let n = norm(normal);
    if n < 1e-12 {
        return None;
    }
    let normal = scale(normal, 1.0 / n);
    if dot(p, normal).abs() > half_width.sin() + 1e-12 {
        return None;
    }
    let tangent = cross(normal, a);
    let arc = dot(b, tangent).atan2(dot(b, a));
    let angle = dot(p, tangent).atan2(dot(p, a));
    (angle >= -1e-10 && angle <= arc + 1e-10).then(|| (angle / arc).clamp(0.0, 1.0))
}

const SIDE: usize = 32;
fn cell(v: f64) -> usize {
    (((v + 1.0) * 0.5 * SIDE as f64).floor() as usize).min(SIDE - 1)
}
fn key(x: usize, y: usize, z: usize) -> usize {
    (x * SIDE + y) * SIDE + z
}

pub struct Accumulator {
    points: Vec<Vector>,
    cells: Vec<Vec<usize>>,
    pub events: Vec<Vec<f64>>,
}
impl Accumulator {
    pub fn new(points: &[GroundPoint]) -> Self {
        let points: Vec<_> = points.iter().map(|p| p.xyz()).collect();
        let mut cells = vec![Vec::new(); SIDE * SIDE * SIDE];
        for (i, p) in points.iter().enumerate() {
            cells[key(cell(p[0]), cell(p[1]), cell(p[2]))].push(i);
        }
        Self {
            events: vec![Vec::new(); points.len()],
            points,
            cells,
        }
    }

    pub fn segment(&mut self, a: Vector, b: Vector, start: f64, end: f64, half_width: f64) {
        let middle = unit(std::array::from_fn(|i| a[i] + b[i]));
        // A cube enclosing the complete spherical strip, also at poles/dateline.
        let extent = norm(sub(a, middle)).max(norm(sub(b, middle))) + half_width + 1e-10;
        let lo: [usize; 3] = std::array::from_fn(|i| cell(middle[i] - extent));
        let hi: [usize; 3] = std::array::from_fn(|i| cell(middle[i] + extent));
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    for &i in &self.cells[key(x, y, z)] {
                        if let Some(t) = crossing(a, b, self.points[i], half_width) {
                            self.events[i].push(start + t * (end - start));
                        }
                    }
                }
            }
        }
    }
    pub fn finish(self, duration: f64) -> Vec<PointStats> {
        self.events
            .into_iter()
            .map(|e| PointStats::new(e, duration))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct PointStats {
    pub count: usize,
    pub min_gap: Option<f64>,
    pub max_gap: Option<f64>,
    /// Includes the censored start/end intervals. This is only a measured span
    /// inside the experiment, not a bound on gaps outside the experiment.
    pub longest_silence: f64,
}
impl PointStats {
    pub fn new(mut events: Vec<f64>, duration: f64) -> Self {
        events.retain(|t| t.is_finite() && *t >= 0.0 && *t <= duration);
        events.sort_unstable_by(f64::total_cmp);
        // Shared segment endpoints and truly simultaneous satellites count once.
        events.dedup_by(|a, b| (*a - *b).abs() < 0.001);
        let gaps: Vec<_> = events.windows(2).map(|w| w[1] - w[0]).collect();
        let min_gap = gaps.iter().copied().reduce(f64::min);
        let max_gap = gaps.iter().copied().reduce(f64::max);
        let longest_silence = max_gap
            .unwrap_or(0.0)
            .max(events.first().copied().unwrap_or(duration))
            .max(duration - events.last().copied().unwrap_or(0.0));
        Self {
            count: events.len(),
            min_gap,
            max_gap,
            longest_silence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn p(lat: f64, lon: f64) -> Vector {
        GroundPoint { lat, lon }.xyz()
    }
    #[test]
    fn continuous_scan_detects_narrow_pass_between_samples_without_end_caps() {
        let width = 0.5 / 6371.0;
        assert!(
            (crossing(p(0.0, -1.0), p(0.0, 1.0), p(0.0, 0.0), width).unwrap() - 0.5).abs() < 1e-9
        );
        assert!(crossing(p(0.0, -1.0), p(0.0, 1.0), p(0.01, 0.0), width).is_none());
        assert!(crossing(p(0.0, -1.0), p(0.0, 1.0), p(0.0, 1.001), width).is_none());
        assert!(crossing(p(0.0, 179.0), p(0.0, -179.0), p(0.0, 180.0), width).is_some());
        assert!(crossing(p(89.0, 0.0), p(89.0, 180.0), p(90.0, 0.0), width).is_some());
    }
    #[test]
    fn missed_locations_and_window_edges_are_never_reported_as_finite_revisit() {
        let empty = PointStats::new(vec![], 100.0);
        assert_eq!(empty.count, 0);
        assert_eq!(empty.max_gap, None);
        assert_eq!(empty.longest_silence, 100.0);
        let one = PointStats::new(vec![25.0], 100.0);
        assert_eq!(one.max_gap, None);
        assert_eq!(one.longest_silence, 75.0);
        let many = PointStats::new(vec![80.0, 10.0, 10.00001, 30.0], 100.0);
        assert_eq!(many.count, 3);
        assert!((many.min_gap.unwrap() - 20.0).abs() < 0.001);
        assert_eq!(many.max_gap, Some(50.0));
        assert_eq!(many.longest_silence, 50.0);
    }
    #[test]
    fn spatial_index_matches_exhaustive_scan_including_poles() {
        let points: Vec<_> = (-90..=90)
            .step_by(5)
            .flat_map(|lat| {
                (-180..180).step_by(5).map(move |lon| GroundPoint {
                    lat: lat as f64,
                    lon: lon as f64,
                })
            })
            .collect();
        for (a, b) in [
            (p(0.0, 179.0), p(0.0, -179.0)),
            (p(89.0, 0.0), p(89.0, 180.0)),
            (p(-45.0, 20.0), p(-43.0, 21.0)),
        ] {
            let mut acc = Accumulator::new(&points);
            acc.segment(a, b, 0.0, 10.0, 0.04);
            for (i, point) in points.iter().enumerate() {
                assert_eq!(
                    !acc.events[i].is_empty(),
                    crossing(a, b, point.xyz(), 0.04).is_some()
                );
            }
        }
    }
}
