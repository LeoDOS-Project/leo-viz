//! Placement-independent, saturated-window workload accounting.
//! All byte profiles are linear in AOI area, so totals do not depend on a
//! future subdivision into observation units. No acquisition schedule is implied.
use leo_sim::acquisition::CaptureProfile;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fulfilment {
    #[default]
    Independent,
    Mosaic,
}
impl Fulfilment {
    pub fn label(self) -> &'static str {
        match self {
            Self::Independent => "Independent crops",
            Self::Mosaic => "Complete mosaic",
        }
    }
}

/// Cadence slots start at run start + phase; all pieces must be acquired within
/// that slot. No observation is carried into another slot. Local solar time is
/// circular (e.g. 23:30 ± 1 h includes 00:15), not civil time or satellite eclipse.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Acquisition {
    pub fulfilment: Fulfilment,
    /// Larger values have strictly higher admission priority. Ties use job ID.
    pub priority: u16,
    pub phase_hours: f64,
    pub solar_time_hours: f64,
    pub solar_tolerance_hours: f64,
    pub min_solar_elevation_deg: f64,
}
impl Default for Acquisition {
    fn default() -> Self {
        Self {
            fulfilment: Fulfilment::Independent,
            priority: 1,
            phase_hours: 0.0,
            solar_time_hours: 12.0,
            solar_tolerance_hours: 1.0,
            min_solar_elevation_deg: 0.0,
        }
    }
}
impl Acquisition {
    pub fn validate(&self, cadence: f64) -> Result<(), String> {
        if !self.phase_hours.is_finite()
            || self.phase_hours < 0.0
            || self.phase_hours >= cadence
            || !self.solar_time_hours.is_finite()
            || !(0.0..24.0).contains(&self.solar_time_hours)
            || !self.solar_tolerance_hours.is_finite()
            || !(0.0..=12.0).contains(&self.solar_tolerance_hours)
            || !self.min_solar_elevation_deg.is_finite()
            || !(0.0..=90.0).contains(&self.min_solar_elevation_deg)
        {
            return Err(
                "Invalid acquisition phase, local solar time, tolerance or minimum elevation."
                    .into(),
            );
        }
        Ok(())
    }
    pub fn accepts(&self, solar_hours: f64, elevation_deg: f64) -> bool {
        let distance = ((solar_hours - self.solar_time_hours + 12.0).rem_euclid(24.0) - 12.0).abs();
        elevation_deg >= self.min_solar_elevation_deg && distance <= self.solar_tolerance_hours
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AcquisitionGenerator {
    pub mosaic_fraction: f64,
    pub solar_stddev_hours: f64,
    pub solar_tolerance_hours: f64,
    pub min_solar_elevation_deg: f64,
    pub priority_min: u16,
    pub priority_max: u16,
}
impl Default for AcquisitionGenerator {
    fn default() -> Self {
        Self {
            mosaic_fraction: 0.5,
            solar_stddev_hours: 4.0,
            solar_tolerance_hours: 1.0,
            min_solar_elevation_deg: 0.0,
            priority_min: 1,
            priority_max: 5,
        }
    }
}
impl AcquisitionGenerator {
    pub fn validate(&self) -> Result<(), String> {
        Acquisition {
            solar_tolerance_hours: self.solar_tolerance_hours,
            min_solar_elevation_deg: self.min_solar_elevation_deg,
            ..Default::default()
        }
        .validate(1.0)?;
        if !self.mosaic_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.mosaic_fraction)
            || !self.solar_stddev_hours.is_finite()
            || !(0.0..=12.0).contains(&self.solar_stddev_hours)
            || self.priority_min > self.priority_max
        {
            return Err(
                "Invalid acquisition mixture, normal standard deviation or priority range.".into(),
            );
        }
        Ok(())
    }
    fn generate(&self, rng: &mut Random) -> Acquisition {
        // Rejection sampling, not clamping: no artificial mass at midnight.
        let solar_time_hours = loop {
            let z = (-2.0 * (1.0 - rng.unit()).ln()).sqrt()
                * (std::f64::consts::TAU * rng.unit()).cos();
            let time = 12.0 + z * self.solar_stddev_hours;
            if (0.0..24.0).contains(&time) {
                break time;
            }
        };
        Acquisition {
            fulfilment: if rng.unit() < self.mosaic_fraction {
                Fulfilment::Mosaic
            } else {
                Fulfilment::Independent
            },
            priority: self.priority_min
                + (rng.unit()
                    * (u32::from(self.priority_max) - u32::from(self.priority_min) + 1) as f64)
                    as u16,
            solar_time_hours,
            solar_tolerance_hours: self.solar_tolerance_hours,
            min_solar_elevation_deg: self.min_solar_elevation_deg,
            phase_hours: 0.0,
        }
    }
}

/// Future overload policy: remove whole jobs in this order, never exchange a
/// higher-priority job for any number of lower-priority jobs. Baseline never drops.
pub fn lowest_priority_first(jobs: &[Job]) -> Vec<usize> {
    let mut order: Vec<_> = (0..jobs.len()).collect();
    order.sort_by(|&a, &b| {
        jobs[a]
            .acquisition
            .priority
            .cmp(&jobs[b].acquisition.priority)
            .then_with(|| jobs[a].aoi.key().cmp(&jobs[b].aoi.key()))
    });
    order
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AoiRef {
    pub source: String,
    pub id: String,
    pub area_km2: f64,
    pub longitude: f64,
    pub latitude: f64,
    pub geometry_hash: u64,
}
impl AoiRef {
    pub fn key(&self) -> String {
        format!("{}:{}", self.source, self.id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pipeline {
    Map,
    MapWindow,
    Window,
}
impl Pipeline {
    pub const ALL: [Self; 3] = [Self::Map, Self::MapWindow, Self::Window];
    pub fn label(self) -> &'static str {
        match self {
            Self::Map => "Map",
            Self::MapWindow => "Map → Window",
            Self::Window => "Window",
        }
    }
    pub fn has_window(self) -> bool {
        self != Self::Map
    }
    pub fn has_map(self) -> bool {
        self != Self::Window
    }
    pub fn index(self) -> usize {
        Self::ALL.iter().position(|p| *p == self).unwrap()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TemporalProfile {
    pub name: String,
    pub weight: f64,
    pub cadence_hours: f64,
    pub samples: u32,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bounds {
    pub cadence_min_hours: f64,
    pub cadence_max_hours: f64,
    pub span_min_hours: f64,
    pub span_max_hours: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Generator {
    pub seed: u32,
    pub pipeline_weights: [f64; 3],
    pub temporal: Vec<TemporalProfile>,
    pub bounds: Bounds,
    /// Medians of log-uniform output density distributions, independent of
    /// image encoding/compression (records are not compressed image samples).
    pub map_bytes_per_km2: f64,
    pub window_bytes_per_km2: f64,
    pub output_size_range: f64,
    #[serde(default)]
    pub acquisition: AcquisitionGenerator,
}
impl Default for Generator {
    fn default() -> Self {
        Self {
            seed: 42,
            pipeline_weights: [1.0, 1.0, 1.0],
            temporal: vec![
                TemporalProfile {
                    name: "Daily history".into(),
                    weight: 1.0,
                    cadence_hours: 2.0,
                    samples: 13,
                },
                TemporalProfile {
                    name: "Monthly history".into(),
                    weight: 1.0,
                    cadence_hours: 24.0,
                    samples: 31,
                },
                TemporalProfile {
                    name: "Yearly history".into(),
                    weight: 1.0,
                    cadence_hours: 168.0,
                    samples: 53,
                },
            ],
            bounds: Bounds {
                cadence_min_hours: 2.0,
                cadence_max_hours: 168.0,
                span_min_hours: 24.0,
                span_max_hours: 8760.0,
            },
            map_bytes_per_km2: 8192.0,
            window_bytes_per_km2: 2048.0,
            output_size_range: 100.0,
            acquisition: AcquisitionGenerator::default(),
        }
    }
}
fn positive(v: f64) -> bool {
    v.is_finite() && v > 0.0
}
fn nonnegative(v: f64) -> bool {
    v.is_finite() && v >= 0.0
}
fn weights(values: &[f64]) -> bool {
    !values.is_empty() && values.iter().all(|v| nonnegative(*v)) && positive(values.iter().sum())
}
impl Bounds {
    pub fn validate(&self) -> Result<(), String> {
        if !positive(self.cadence_min_hours)
            || !positive(self.cadence_max_hours)
            || self.cadence_min_hours > self.cadence_max_hours
            || !positive(self.span_min_hours)
            || !positive(self.span_max_hours)
            || self.span_min_hours > self.span_max_hours
        {
            return Err("Cadence and history bounds must be positive and ordered.".into());
        }
        Ok(())
    }
    pub fn contains(&self, cadence_hours: f64, samples: u32) -> bool {
        let span = f64::from(samples.saturating_sub(1)) * cadence_hours;
        cadence_hours >= self.cadence_min_hours
            && cadence_hours <= self.cadence_max_hours
            && (samples == 1 || (span >= self.span_min_hours && span <= self.span_max_hours))
    }
}
impl Generator {
    pub fn acquisition_for(&self, aoi: &AoiRef) -> Acquisition {
        let mut rng = Random(
            hash(aoi.key().as_bytes())
                ^ aoi.geometry_hash
                ^ u64::from(self.seed)
                ^ 0x6163717569736974,
        );
        self.acquisition.generate(&mut rng)
    }
    pub fn validate(&self) -> Result<(), String> {
        self.bounds.validate()?;
        self.acquisition.validate()?;
        if !weights(&self.pipeline_weights)
            || !weights(&self.temporal.iter().map(|p| p.weight).collect::<Vec<_>>())
        {
            return Err(
                "Each mixture needs at least one positive weight; weights cannot be negative."
                    .into(),
            );
        }
        for p in &self.temporal {
            if p.samples < 2
                || p.samples > 100_000
                || !positive(p.cadence_hours)
                || !self.bounds.contains(p.cadence_hours, p.samples)
            {
                return Err(format!(
                    "{}: cadence or full-window span is outside the global bounds.",
                    p.name
                ));
            }
        }
        if !nonnegative(self.map_bytes_per_km2)
            || !nonnegative(self.window_bytes_per_km2)
            || !self.output_size_range.is_finite()
            || !(1.0..=1e6).contains(&self.output_size_range)
        {
            return Err(
                "Output densities must be nonnegative; size range must be 1–1,000,000×.".into(),
            );
        }
        Ok(())
    }
    pub fn generate(&self, aois: impl IntoIterator<Item = AoiRef>) -> Result<Vec<Job>, String> {
        self.validate()?;
        let mut keys = std::collections::HashSet::new();
        let mut jobs = Vec::new();
        let temporal_weights: Vec<_> = self.temporal.iter().map(|p| p.weight).collect();
        for aoi in aois {
            if !keys.insert(aoi.key()) {
                return Err("AOI identifiers must be unique within each source.".into());
            }
            let mut rng =
                Random(hash(aoi.key().as_bytes()) ^ aoi.geometry_hash ^ u64::from(self.seed));
            let pipeline = Pipeline::ALL[rng.choose(&self.pipeline_weights)];
            let profile = &self.temporal[rng.choose(&temporal_weights)];
            let job = Job {
                aoi,
                acquisition: Acquisition::default(),
                pipeline,
                cadence_hours: profile.cadence_hours,
                samples: if pipeline.has_window() {
                    profile.samples
                } else {
                    1
                },
                map_bytes_per_km2: self.map_bytes_per_km2 * rng.spread(self.output_size_range),
                window_bytes_per_km2: self.window_bytes_per_km2
                    * rng.spread(self.output_size_range),
            };
            let mut job = job;
            job.acquisition = self.acquisition_for(&job.aoi);
            job.validate()?;
            jobs.push(job);
        }
        Ok(jobs)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Job {
    pub aoi: AoiRef,
    pub pipeline: Pipeline,
    #[serde(default)]
    pub acquisition: Acquisition,
    /// Reference cadence for the steady average, not a geometric opportunity.
    pub cadence_hours: f64,
    pub samples: u32,
    pub map_bytes_per_km2: f64,
    pub window_bytes_per_km2: f64,
}
impl Job {
    pub fn span_hours(&self) -> f64 {
        f64::from(self.samples.saturating_sub(1)) * self.cadence_hours
    }
    pub fn validate(&self) -> Result<(), String> {
        self.acquisition.validate(self.cadence_hours)?;
        if self.aoi.id.is_empty()
            || self.aoi.source.is_empty()
            || !positive(self.aoi.area_km2)
            || !self.aoi.latitude.is_finite()
            || self.aoi.latitude.abs() > 90.0
            || !self.aoi.longitude.is_finite()
            || self.aoi.longitude.abs() > 180.0
            || !positive(self.cadence_hours)
            || !self.span_hours().is_finite()
            || self.samples > 100_000
            || (self.pipeline.has_window() && self.samples < 2)
            || (!self.pipeline.has_window() && self.samples != 1)
            || !nonnegative(self.map_bytes_per_km2)
            || !nonnegative(self.window_bytes_per_km2)
        {
            return Err(format!("Invalid job configuration for {}.", self.aoi.key()));
        }
        Ok(())
    }
    pub fn estimate(&self, capture: CaptureProfile) -> Result<Estimate, String> {
        self.validate()?;
        let input = capture
            .stored_bytes_for_area(self.aoi.area_km2)
            .ok_or("Invalid image encoding or compression.")?;
        let mapped = self.aoi.area_km2 * self.map_bytes_per_km2;
        let entry = match self.pipeline {
            Pipeline::Map => 0.0,
            Pipeline::MapWindow => mapped,
            Pipeline::Window => input,
        };
        let output = if self.pipeline.has_window() {
            self.aoi.area_km2 * self.window_bytes_per_km2
        } else {
            mapped
        };
        let updates_per_day = 24.0 / self.cadence_hours;
        let estimate = Estimate {
            input_per_update: input,
            window_entry_bytes: entry,
            output_per_update: output,
            input_per_day: input * updates_per_day,
            history_bytes: entry * f64::from(self.samples),
            output_per_day: output * updates_per_day,
        };
        if [
            estimate.input_per_update,
            estimate.window_entry_bytes,
            estimate.output_per_update,
            estimate.input_per_day,
            estimate.history_bytes,
            estimate.output_per_day,
        ]
        .iter()
        .any(|v| !nonnegative(*v))
        {
            return Err("Byte estimate exceeds the supported numeric range.".into());
        }
        Ok(estimate)
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Estimate {
    pub input_per_update: f64,
    pub window_entry_bytes: f64,
    pub output_per_update: f64,
    pub input_per_day: f64,
    pub history_bytes: f64,
    pub output_per_day: f64,
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Totals {
    pub count: usize,
    pub input_per_day: f64,
    pub history_bytes: f64,
    pub output_per_day: f64,
}
impl Totals {
    pub fn add(&mut self, estimate: Estimate) {
        self.count += 1;
        self.input_per_day += estimate.input_per_day;
        self.history_bytes += estimate.history_bytes;
        self.output_per_day += estimate.output_per_day;
    }
}
pub fn totals(jobs: &[Job], capture: CaptureProfile) -> Result<(Totals, [Totals; 3]), String> {
    let mut total = Totals::default();
    let mut groups = [Totals::default(); 3];
    for job in jobs {
        let estimate = job.estimate(capture)?;
        total.add(estimate);
        groups[job.pipeline.index()].add(estimate);
    }
    if [
        total.input_per_day,
        total.history_bytes,
        total.output_per_day,
    ]
    .iter()
    .any(|v| !nonnegative(*v))
    {
        return Err("Total byte estimate exceeds the supported numeric range.".into());
    }
    Ok((total, groups))
}

/// Stable hashing/randomness makes assignment independent of AOI traversal order.
pub fn hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    })
}
struct Random(u64);
impl Random {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        (((z ^ (z >> 31)) >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn choose(&mut self, weights: &[f64]) -> usize {
        let mut value = self.unit() * weights.iter().sum::<f64>();
        for (i, weight) in weights.iter().enumerate() {
            value -= weight;
            if value < 0.0 {
                return i;
            }
        }
        weights.iter().rposition(|w| *w > 0.0).unwrap()
    }
    fn spread(&mut self, range: f64) -> f64 {
        ((self.unit() - 0.5) * range.ln()).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn aoi(id: &str) -> AoiRef {
        AoiRef {
            source: "gdp".into(),
            id: id.into(),
            area_km2: 1.0,
            longitude: 0.0,
            latitude: 0.0,
            geometry_hash: 1,
        }
    }
    fn job(pipeline: Pipeline) -> Job {
        Job {
            aoi: aoi("a"),
            acquisition: Acquisition::default(),
            pipeline,
            cadence_hours: 2.0,
            samples: if pipeline.has_window() { 13 } else { 1 },
            map_bytes_per_km2: 1000.0,
            window_bytes_per_km2: 100.0,
        }
    }
    fn capture() -> CaptureProfile {
        CaptureProfile {
            channels: 4,
            resolution_m: 1.0,
            bits_per_sample: 8,
            compression_ratio: 2.0,
        }
    }
    #[test]
    fn independent_oracle_for_all_three_pipelines_and_reusable_history() {
        let map = job(Pipeline::Map).estimate(capture()).unwrap();
        assert_eq!(
            (map.input_per_day, map.history_bytes, map.output_per_day),
            (24e6, 0.0, 12000.0)
        );
        let mw = job(Pipeline::MapWindow).estimate(capture()).unwrap();
        assert_eq!(
            (mw.input_per_day, mw.history_bytes, mw.output_per_day),
            (24e6, 13000.0, 1200.0)
        );
        let window = job(Pipeline::Window).estimate(capture()).unwrap();
        assert_eq!(
            (
                window.input_per_day,
                window.history_bytes,
                window.output_per_day
            ),
            (24e6, 26e6, 1200.0)
        );
        let mut bigger = job(Pipeline::Window);
        bigger.samples *= 2;
        let doubled = bigger.estimate(capture()).unwrap();
        assert_eq!(doubled.history_bytes, window.history_bytes * 2.0);
        assert_eq!(doubled.input_per_day, window.input_per_day);
        assert_eq!(doubled.output_per_day, window.output_per_day);
    }
    #[test]
    fn cadence_resolution_encoding_and_compression_have_distinct_effects() {
        let mut j = job(Pipeline::MapWindow);
        let before = j.estimate(capture()).unwrap();
        j.cadence_hours *= 2.0;
        let after = j.estimate(capture()).unwrap();
        assert_eq!(after.input_per_day, before.input_per_day / 2.0);
        assert_eq!(after.output_per_day, before.output_per_day / 2.0);
        assert_eq!(after.history_bytes, before.history_bytes);
        let finer = CaptureProfile {
            resolution_m: 0.5,
            ..capture()
        };
        assert_eq!(
            j.estimate(finer).unwrap().input_per_day,
            after.input_per_day * 4.0
        );
        assert_eq!(
            j.estimate(finer).unwrap().history_bytes,
            after.history_bytes
        );
        let uncompressed = CaptureProfile {
            compression_ratio: 1.0,
            ..capture()
        };
        assert_eq!(
            job(Pipeline::Window)
                .estimate(uncompressed)
                .unwrap()
                .history_bytes,
            52e6
        );
    }
    #[test]
    fn overlaps_count_independently_and_tiling_preserves_area_totals() {
        let j = job(Pipeline::Window);
        let (single, _) = totals(&[j.clone()], capture()).unwrap();
        let (overlap, _) = totals(&[j.clone(), j.clone()], capture()).unwrap();
        assert_eq!(overlap.input_per_day, single.input_per_day * 2.0);
        assert_eq!(overlap.history_bytes, single.history_bytes * 2.0);
        let mut tile = j;
        tile.aoi.area_km2 = 0.25;
        let (tiled, _) = totals(&vec![tile; 4], capture()).unwrap();
        assert_eq!(tiled.input_per_day, single.input_per_day);
        assert_eq!(tiled.history_bytes, single.history_bytes);
        assert_eq!(tiled.output_per_day, single.output_per_day);
    }
    #[test]
    fn stable_assignments_joint_profiles_and_invalid_configurations() {
        let generator = Generator::default();
        let jobs = generator
            .generate((0..1000).map(|i| aoi(&i.to_string())))
            .unwrap();
        let reverse = generator
            .generate((0..1000).rev().map(|i| aoi(&i.to_string())))
            .unwrap();
        assert_eq!(jobs, reverse.into_iter().rev().collect::<Vec<_>>());
        for j in &jobs {
            assert!(generator.bounds.contains(j.cadence_hours, j.samples));
        }
        assert!(Pipeline::ALL
            .iter()
            .all(|p| jobs.iter().any(|j| j.pipeline == *p)));
        let mut changed = generator.clone();
        changed.seed += 1;
        assert_ne!(
            jobs,
            changed
                .generate((0..1000).map(|i| aoi(&i.to_string())))
                .unwrap()
        );
        changed.pipeline_weights = [0.0; 3];
        assert!(changed.validate().is_err());
        let mut bad = generator;
        bad.temporal[0].samples = 10000;
        assert!(bad.validate().is_err());
        let mut bad_job = job(Pipeline::Window);
        bad_job.cadence_hours = 0.0;
        assert!(bad_job.estimate(capture()).is_err());
    }
}
