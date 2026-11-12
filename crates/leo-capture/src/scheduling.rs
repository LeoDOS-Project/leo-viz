//! Finite acquisition DAGs, not a scheduler. Each action has a skip edge.
//! Crop polygons retain exact overlap information within the numerical geometry
//! model. No centroid coverage, storage, processing or bandwidth constraints.
mod export;
mod graph;

pub use export::{compact, generate, load_preset, Config, Summary};
pub use graph::{Action, Topology};

use crate::{
    baseline, geometry,
    workload::{Illumination, Job},
};
use leo_sim::{
    geometry::{
        dot, gmst, ground_position, local_solar_time_hours, sun_direction, Vector,
        EARTH_RADIUS_KM as R,
    },
    orbit::Elements,
    worldview,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub duration_s: f64,
    pub step_s: f64,
    pub width_km: f64,
    pub reach_km: f64,
    pub cross_spacing_km: f64,
    pub along_offsets_km: Vec<f64>,
    pub settle_s: f64,
    pub edge_km: f64,
    pub cell_deg: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            duration_s: 86400.0,
            step_s: 30.0,
            width_km: worldview::IMAGING_SWATH_KM,
            reach_km: worldview::ACCESS_WIDTH_KM,
            cross_spacing_km: worldview::IMAGING_SWATH_KM * 0.9,
            along_offsets_km: vec![-150.0, 0.0, 150.0],
            settle_s: 2.0,
            edge_km: 2.0,
            cell_deg: 2.0,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<(), String> {
        if !self.duration_s.is_finite()
            || self.duration_s <= 0.0
            || self.duration_s > 30.0 * 86400.0
            || !self.step_s.is_finite()
            || !(1.0..=30.0).contains(&self.step_s)
            || (self.duration_s / self.step_s - (self.duration_s / self.step_s).round()).abs()
                > 1e-8
            || !self.width_km.is_finite()
            || self.width_km <= 0.0
            || !self.reach_km.is_finite()
            || self.reach_km < self.width_km
            || self.reach_km > 500.0
            || !self.cross_spacing_km.is_finite()
            || self.cross_spacing_km < 1.0
            || self.cross_spacing_km > self.width_km
            || !self.settle_s.is_finite()
            || !(0.0..12.0).contains(&self.settle_s)
            || self.along_offsets_km.is_empty()
            || self.along_offsets_km.len() > 31
            || self
                .along_offsets_km
                .iter()
                .any(|x| !x.is_finite() || x.abs() > (self.reach_km - self.width_km) / 2.0)
            || !self.along_offsets_km.contains(&0.0)
        {
            return Err("Invalid graph horizon, strip sampling, reach, or agility settings".into());
        }
        let mut offsets = self.along_offsets_km.clone();
        offsets.sort_by(f64::total_cmp);
        offsets.dedup();
        if offsets.len() != self.along_offsets_km.len() {
            return Err("Duplicate along-track offsets".into());
        }
        self.baseline().validate()
    }
    fn baseline(&self) -> baseline::Options {
        baseline::Options {
            duration_s: self.duration_s,
            step_s: self.step_s,
            edge_km: self.edge_km,
            cell_deg: self.cell_deg,
            ..Default::default()
        }
    }
    pub fn ticks(&self) -> u32 {
        (self.duration_s / self.step_s).round() as u32
    }
    /// A single published 200 km / 12 s example, interpreted as a nadir-to-target
    /// maneuver at 617 km. The settling allocation is assumed, not published.
    pub fn slew_rate_rad_s(&self) -> f64 {
        let arc = 200.0 / R;
        (R * arc.sin()).atan2(worldview::ALTITUDE_KM + R * (1.0 - arc.cos()))
            / (12.0 - self.settle_s)
    }
    pub fn modes(&self) -> Vec<Mode> {
        let limit = (self.reach_km - self.width_km) / 2.0;
        let mut modes = Vec::new();
        let mut along = self.along_offsets_km.clone();
        along.sort_by(f64::total_cmp);
        for x in along {
            let count =
                ((limit * limit - x * x).max(0.0).sqrt() / self.cross_spacing_km).floor() as i32;
            for i in -count..=count {
                modes.push(Mode::new(x, f64::from(i) * self.cross_spacing_km));
            }
        }
        modes
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mode {
    pub along_km: f64,
    pub cross_km: f64,
    /// Minimum-rotation quaternion (w,x,y,z), relative to the orbit-following
    /// nadir frame. This surrogate ignores attitude-controller dynamics.
    pub attitude: [f64; 4],
}
impl Mode {
    fn new(along_km: f64, cross_km: f64) -> Self {
        let distance = along_km.hypot(cross_km);
        let attitude = if distance < 1e-12 {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            let angle = distance / R;
            let look = geometry::unit([
                R * angle.sin() * along_km / distance,
                R * angle.sin() * cross_km / distance,
                worldview::ALTITUDE_KM + R * (1.0 - angle.cos()),
            ]);
            let w = ((1.0 + look[2]) / 2.0).sqrt();
            [w, -look[1] / (2.0 * w), look[0] / (2.0 * w), 0.0]
        };
        Self {
            along_km,
            cross_km,
            attitude,
        }
    }
    pub fn transition_s(&self, other: &Self, options: &Options) -> f64 {
        if self.along_km == other.along_km && self.cross_km == other.cross_km {
            return 0.0;
        }
        let cosine = self
            .attitude
            .iter()
            .zip(other.attitude)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .abs()
            .clamp(0.0, 1.0);
        options.settle_s + 2.0 * cosine.acos() / options.slew_rate_rad_s()
    }
    fn offset(&self, q: Vector, n: Vector) -> (Vector, Vector) {
        let distance = self.along_km.hypot(self.cross_km);
        if distance < 1e-12 {
            return (q, n);
        }
        let tangent = geometry::cross(n, q);
        let direction =
            std::array::from_fn(|i| (tangent[i] * self.along_km + n[i] * self.cross_km) / distance);
        let axis = geometry::cross(q, direction);
        let angle = distance / R;
        let center = std::array::from_fn(|i| q[i] * angle.cos() + direction[i] * angle.sin());
        let axis_cross_n = geometry::cross(axis, n);
        let normal = std::array::from_fn(|i| {
            n[i] * angle.cos()
                + axis_cross_n[i] * angle.sin()
                + axis[i] * dot(axis, n) * (1.0 - angle.cos())
        });
        (center, normal)
    }
}

fn sample(elements: &Elements, epoch: f64, t: f64) -> (Vector, Vector) {
    let fixed = |time: f64| {
        let p = elements.sample(time).xyz;
        let (s, c) = gmst(epoch + time).sin_cos();
        geometry::unit([c * p[0] - s * p[2], p[1], s * p[0] + c * p[2]])
    };
    let q = fixed(t);
    let n = geometry::unit(geometry::cross(fixed(t - 0.25), fixed(t + 0.25)));
    (q, geometry::normal_between(n, n, q, 0.0))
}

/// Reject only if *every* cadence-clipped subinterval fails the same anchor
/// time checks as the baseline. Polygon-vertex elevation checks still run later.
fn eligible(job: &Job, epoch: f64, start: f64, end: f64) -> bool {
    let phase = job.acquisition.phase_hours * 3600.0;
    let cadence = job.cadence_hours * 3600.0;
    let mut left = start.max(phase);
    while left < end - 1e-9 {
        let slot = ((left - phase) / cadence + 1e-10).floor();
        let right = end.min(phase + (slot + 1.0) * cadence);
        if right <= left {
            break;
        }
        if job.acquisition.illumination == Illumination::AnyTime {
            return true;
        }
        if [left, (left + right) / 2.0, right].iter().all(|t| {
            let sun = sun_direction(epoch + t);
            let anchor = ground_position(job.aoi.latitude, job.aoi.longitude, 1.0, gmst(epoch + t));
            job.acquisition
                .accepts(local_solar_time_hours(anchor, sun), 90.0)
        }) {
            return true;
        }
        left = right;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use leo_sim::geometry::norm;
    #[test]
    fn modes_reach_and_calibrated_agility() {
        let o = Options::default();
        o.validate().unwrap();
        let modes = o.modes();
        assert_eq!(modes.len(), 91);
        for m in &modes {
            assert!(m.along_km.hypot(m.cross_km) + o.width_km / 2.0 <= o.reach_km / 2.0 + 1e-9);
        }
        assert!(
            (Mode::new(0.0, 0.0).transition_s(&Mode::new(0.0, 200.0), &o) - 12.0).abs() < 1e-10
        );
        assert_eq!(modes[0].transition_s(&modes[0], &o), 0.0);
        for m in &modes {
            let (q, n) = m.offset([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
            assert!((norm(q) - 1.0).abs() < 1e-12 && dot(q, n).abs() < 1e-12);
            assert!((q[0].acos() * R - m.along_km.hypot(m.cross_km)).abs() < 1e-7);
        }
    }
    #[test]
    fn phase_and_cadence_prefilter() {
        let mut j = crate::workload::Generator::default()
            .generate([crate::workload::AoiRef {
                source: "test".into(),
                id: "phase".into(),
                area_km2: 1.0,
                longitude: 0.0,
                latitude: 0.0,
                geometry_hash: 42,
            }])
            .unwrap()
            .remove(0);
        j.acquisition.illumination = Illumination::AnyTime;
        j.acquisition.phase_hours = 0.5;
        assert!(!eligible(&j, 0.0, 0.0, 30.0));
        assert!(eligible(&j, 0.0, 1790.0, 1820.0));
    }
}
