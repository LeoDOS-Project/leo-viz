use crate::{
    geometry::{self, Atlas, Shape},
    workload::{Fulfilment, Job},
};
use geo::{Area, BooleanOps, BoundingRect, Intersects, MultiPolygon};
use leo_sim::{
    acquisition::CaptureProfile,
    geometry::{
        dot, gmst, ground_position, local_solar_time_hours, norm, sun_direction, Vector,
        EARTH_RADIUS_KM as R,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const DEFAULT_STORAGE_BYTES: u64 = 32_000_000_000_000;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub duration_s: f64,
    pub step_s: f64,
    pub cell_deg: f64,
    pub edge_km: f64,
    pub storage_bytes_per_satellite: u64,
    /// Relative geometric area tolerance for polygon clipping roundoff.
    pub mosaic_area_tolerance: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            duration_s: 86400.0,
            step_s: 5.0,
            cell_deg: 2.0,
            edge_km: 1.0,
            storage_bytes_per_satellite: DEFAULT_STORAGE_BYTES,
            mosaic_area_tolerance: 1e-6,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<(), String> {
        if !self.duration_s.is_finite()
            || !(0.0..=30.0 * 86400.0).contains(&self.duration_s)
            || self.duration_s == 0.0
            || !self.step_s.is_finite()
            || !(0.05..=30.0).contains(&self.step_s)
            || self.storage_bytes_per_satellite == 0
            || !self.mosaic_area_tolerance.is_finite()
            || !(0.0..=1e-4).contains(&self.mosaic_area_tolerance)
        {
            return Err("Invalid baseline duration (≤30 days), step (0.05–30 s), storage or mosaic tolerance.".into());
        }
        Ok(())
    }
}

pub struct Crop {
    pub job: usize,
    pub slot: u64,
    pub start_s: f64,
    pub end_s: f64,
    pub geometry: Shape,
}
pub struct Opportunity {
    pub satellite: usize,
    pub patch: usize,
    pub start_s: f64,
    pub end_s: f64,
    pub crops: Vec<Crop>,
}
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct OpportunityStats {
    pub strip_patch_opportunities: u64,
    pub eligible_job_crops: u64,
    pub geometrically_intersecting_job_segments: u64,
    pub timing_or_daylight_rejected_segments: u64,
    pub eligible_physical_area_km2: f64,
    pub eligible_logical_area_km2: f64,
}

/// Reusable streaming generator. `position(satellite, seconds)` returns an
/// inertial position at seconds from this run's start; epoch is that start's UTC.
/// It supports any propagator adapter, including the GUI's Kepler/J2/SGP4 paths.
pub fn opportunities(
    jobs: &[Job],
    atlas: &Atlas,
    satellite_count: usize,
    epoch_unix_s: f64,
    swath_km: f64,
    options: &Options,
    position: impl FnMut(usize, f64) -> Result<Vector, String>,
    visit: impl FnMut(Opportunity) -> Result<(), String>,
    progress: impl FnMut(f32) -> bool,
) -> Result<OpportunityStats, String> {
    options.validate()?;
    for j in jobs {
        j.validate()?;
    }
    opportunities_range(
        jobs,
        atlas,
        satellite_count,
        epoch_unix_s,
        swath_km,
        options,
        0,
        (options.duration_s / options.step_s).ceil() as u64,
        OpportunityStats::default(),
        position,
        visit,
        progress,
    )
}

/// Advance on the original run's fixed tick grid. Chunking never changes strip
/// boundaries, slot origins or the order in which satellites acquire coverage.
pub(crate) fn opportunities_range(
    jobs: &[Job],
    atlas: &Atlas,
    satellite_count: usize,
    epoch_unix_s: f64,
    swath_km: f64,
    options: &Options,
    first_tick: u64,
    last_tick: u64,
    mut stats: OpportunityStats,
    mut position: impl FnMut(usize, f64) -> Result<Vector, String>,
    mut visit: impl FnMut(Opportunity) -> Result<(), String>,
    mut progress: impl FnMut(f32) -> bool,
) -> Result<OpportunityStats, String> {
    options.validate()?;
    if satellite_count == 0
        || !epoch_unix_s.is_finite()
        || !swath_km.is_finite()
        || !(0.0..=500.0).contains(&swath_km)
        || swath_km == 0.0
    {
        return Err("Finite epoch, satellites and a 0–500 km positive swath are required.".into());
    }
    let half = swath_km / (2.0 * R);
    let fixed = |p: Vector, t: f64| -> Result<Vector, String> {
        let radius = norm(p);
        if !radius.is_finite() || radius <= R || half > (R / radius).acos() {
            return Err("Invalid propagated position or swath beyond the horizon.".into());
        }
        let (s, c) = gmst(epoch_unix_s + t).sin_cos();
        Ok(geometry::unit([
            c * p[0] - s * p[2],
            p[1],
            s * p[0] + c * p[2],
        ]))
    };
    // Centered ground velocity defines a shared scanline at each boundary.
    // Using each segment's own great-circle normal creates artificial wedge gaps
    // where successive segments turn. The 0.5 s derivative is independent of step.
    let mut sample = |s: usize, t: f64| -> Result<(Vector, Vector), String> {
        let q = fixed(position(s, t)?, t)?;
        let before = fixed(position(s, t - 0.25)?, t - 0.25)?;
        let after = fixed(position(s, t + 0.25)?, t + 0.25)?;
        let n = geometry::cross(before, after);
        if norm(n) < 1e-14 {
            return Err("Ground track has no resolved heading.".into());
        }
        let n = geometry::normal_between(n, n, q, 0.0);
        Ok((q, n))
    };
    let mut previous = (0..satellite_count)
        .map(|s| sample(s, first_tick as f64 * options.step_s))
        .collect::<Result<Vec<_>, _>>()?;
    let steps = last_tick.min((options.duration_s / options.step_s).ceil() as u64);
    for tick in first_tick..steps {
        if !progress((tick - first_tick) as f32 / (steps - first_tick) as f32) {
            return Err("Cancelled.".into());
        }
        let start = tick as f64 * options.step_s;
        let end = ((tick + 1) as f64 * options.step_s).min(options.duration_s);
        for (satellite, (a, na)) in previous.iter_mut().enumerate() {
            let (b, nb) = sample(satellite, end)?;
            if dot(*a, b) < 0.1_f64.cos() {
                return Err(
                    "Ground-track segment exceeds 637 km; reduce step or check propagation.".into(),
                );
            }
            for patch_id in atlas.candidates(*a, b, half) {
                let patch = &atlas.patches[patch_id];
                // BVH leaves are conservative; filter each patch before projecting.
                let middle = geometry::unit(std::array::from_fn(|i| a[i] + b[i]));
                if norm(leo_sim::geometry::sub(patch.sphere.center, middle))
                    > patch.sphere.radius + norm(leo_sim::geometry::sub(*a, middle)) + half + 1e-8
                {
                    continue;
                }
                let full_strip =
                    geometry::strip(&patch.chart, *a, b, *na, nb, half, options.edge_km);
                let Some(bounds) = full_strip.bounding_rect() else {
                    continue;
                };
                let mut crops = Vec::new();
                for piece in &patch.pieces {
                    if !piece.bounds.intersects(&bounds) {
                        continue;
                    }
                    let intersect = piece.shape.intersection(&full_strip);
                    if intersect.unsigned_area() <= 1e-10 {
                        continue;
                    }
                    stats.geometrically_intersecting_job_segments += 1;
                    let job = &jobs[piece.job];
                    let cadence = job.cadence_hours * 3600.0;
                    let phase = job.acquisition.phase_hours * 3600.0;
                    let mut left = start.max(phase);
                    let mut accepted = false;
                    while left < end - 1e-9 {
                        let slot = ((left - phase) / cadence + 1e-10).floor() as u64;
                        let right = end.min(phase + (slot + 1) as f64 * cadence);
                        if right <= left {
                            break;
                        }
                        let shape = if left == start && right == end {
                            intersect.clone()
                        } else {
                            let aa = geometry::mix(*a, b, (left - start) / (end - start));
                            let bb = geometry::mix(*a, b, (right - start) / (end - start));
                            let naa = geometry::normal_between(
                                *na,
                                nb,
                                aa,
                                (left - start) / (end - start),
                            );
                            let nbb = geometry::normal_between(
                                *na,
                                nb,
                                bb,
                                (right - start) / (end - start),
                            );
                            piece.shape.intersection(&geometry::strip(
                                &patch.chart,
                                aa,
                                bb,
                                naa,
                                nbb,
                                half,
                                options.edge_km,
                            ))
                        };
                        if shape.unsigned_area() > 1e-10
                            && usable(job, &shape, &patch.chart, epoch_unix_s, left, right)
                        {
                            accepted = true;
                            crops.push(Crop {
                                job: piece.job,
                                slot,
                                start_s: left,
                                end_s: right,
                                geometry: shape,
                            });
                        }
                        left = right;
                    }
                    if !accepted {
                        stats.timing_or_daylight_rejected_segments += 1;
                    }
                }
                if !crops.is_empty() {
                    let mut union = MultiPolygon::new(vec![]);
                    for c in &crops {
                        stats.eligible_logical_area_km2 += c.geometry.unsigned_area();
                        union = union.union(&c.geometry);
                    }
                    stats.eligible_physical_area_km2 += union.unsigned_area();
                    stats.strip_patch_opportunities += 1;
                    stats.eligible_job_crops += crops.len() as u64;
                    visit(Opportunity {
                        satellite,
                        patch: patch_id,
                        start_s: start,
                        end_s: end,
                        crops,
                    })?;
                }
            }
            *a = b;
            *na = nb;
        }
    }
    progress(1.0);
    Ok(stats)
}

fn usable(
    job: &Job,
    shape: &Shape,
    chart: &geometry::Chart,
    epoch: f64,
    start: f64,
    end: f64,
) -> bool {
    // Local-time preference is defined at the job's geographic anchor. Solar
    // elevation is conservatively checked over crop vertices at all three times.
    // Near a boundary this can reject a sub-step valid sliver: halve step/edge
    // spacing to measure convergence. There is no cloud/random usability filter.
    [start, (start + end) * 0.5, end].iter().all(|t| {
        let sun = sun_direction(epoch + t);
        let rotation = gmst(epoch + t);
        let anchor = ground_position(job.aoi.latitude, job.aoi.longitude, 1.0, rotation);
        let local_time = local_solar_time_hours(anchor, sun);
        let (s, c) = rotation.sin_cos();
        let min_elevation = shape
            .0
            .iter()
            .flat_map(|p| std::iter::once(p.exterior()).chain(p.interiors()))
            .flat_map(|r| r.0.iter())
            .map(|p| {
                let v = chart.inverse(*p);
                let inertial = [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]];
                dot(inertial, sun).clamp(-1.0, 1.0).asin().to_degrees()
            })
            .fold(90.0, f64::min);
        job.acquisition.accepts(local_time, min_elevation)
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobReport {
    pub job_key: String,
    pub priority: u16,
    pub fulfilment: Fulfilment,
    pub declared_area_km2: f64,
    pub geometric_area_km2: f64,
    pub due_updates: u64,
    pub complete_due_updates: u64,
    pub partial_due_updates: u64,
    pub missed_due_updates: u64,
    pub pending_updates: u64,
    pub complete_pending_updates: u64,
    pub captured_area_km2: f64,
    pub ready_area_km2: f64,
    pub due_captured_area_km2: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SatelliteReport {
    pub satellite: usize,
    pub retained_crop_bytes: f64,
    pub storage_capacity_bytes: u64,
    pub first_capacity_exceeded_s: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub schema_version: u32,
    pub mode: String,
    pub epoch_unix_s: f64,
    pub swath_km: f64,
    pub capture: CaptureProfile,
    pub options: Options,
    pub jobs: Vec<JobReport>,
    pub satellites: Vec<SatelliteReport>,
    pub opportunities: OpportunityStats,
    pub retained_physical_bytes: f64,
    pub fulfilled_logical_bytes: f64,
    pub reuse_saved_bytes: f64,
    pub reused_physical_bytes: f64,
    pub dropped_jobs: u64,
    pub assumptions: Vec<String>,
}
#[derive(Clone, Serialize, Deserialize)]
struct Request {
    patches: BTreeMap<usize, Shape>,
    area: f64,
    complete: bool,
}
/// Persistent capture state. Geometry is in the deterministic atlas charts.
#[derive(Clone, Serialize, Deserialize)]
pub struct BaselineState {
    #[serde(with = "request_map")]
    active: BTreeMap<(usize, u64), Request>,
    expiry: BTreeSet<(u64, usize, u64)>,
    pub report: Report,
    pub captures: Vec<RetainedCapture>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Allocation {
    pub job: usize,
    pub slot: u64,
    pub geometry: Shape,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct RetainedCapture {
    pub satellite: usize,
    pub patch: usize,
    pub start_s: f64,
    pub end_s: f64,
    pub bytes: f64,
    pub geometry: Shape,
    pub allocations: Vec<Allocation>,
}
mod request_map {
    use super::*;
    pub fn serialize<S: serde::Serializer>(
        map: &BTreeMap<(usize, u64), Request>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        map.iter().collect::<Vec<_>>().serialize(s)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        d: D,
    ) -> Result<BTreeMap<(usize, u64), Request>, D::Error> {
        let values = Vec::<((usize, u64), Request)>::deserialize(d)?;
        let len = values.len();
        let map: BTreeMap<_, _> = values.into_iter().collect();
        if map.len() != len {
            return Err(serde::de::Error::custom("Duplicate coverage state"));
        }
        Ok(map)
    }
}
pub struct Baseline<'a> {
    jobs: &'a [Job],
    atlas: &'a Atlas,
    options: &'a Options,
    bytes_per_km2: f64,
    state: BaselineState,
    retain_ledger: bool,
}
impl std::ops::Deref for Baseline<'_> {
    type Target = BaselineState;
    fn deref(&self) -> &Self::Target {
        &self.state
    }
}
impl std::ops::DerefMut for Baseline<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}
impl<'a> Baseline<'a> {
    pub fn new(
        jobs: &'a [Job],
        atlas: &'a Atlas,
        satellites: usize,
        epoch: f64,
        swath: f64,
        capture: CaptureProfile,
        options: &'a Options,
    ) -> Result<Self, String> {
        options.validate()?;
        for job in jobs {
            job.validate()?;
        }
        if jobs.is_empty() || jobs.len() != atlas.job_areas_km2.len() || satellites == 0 {
            return Err("Baseline requires jobs, matching geometry and satellites.".into());
        }
        let bytes_per_km2 = capture
            .stored_bytes_for_area(1.0)
            .ok_or("Invalid capture profile")?;
        let reports = jobs
            .iter()
            .enumerate()
            .map(|(i, j)| {
                let available = (options.duration_s - j.acquisition.phase_hours * 3600.0).max(0.0);
                let intervals = available / (j.cadence_hours * 3600.0);
                let due = (intervals + 1e-10).floor() as u64;
                let pending = u64::from(intervals > due as f64 + 1e-10);
                JobReport {
                    job_key: j.aoi.key(),
                    priority: j.acquisition.priority,
                    fulfilment: j.acquisition.fulfilment,
                    declared_area_km2: j.aoi.area_km2,
                    geometric_area_km2: atlas.job_areas_km2[i],
                    due_updates: due,
                    complete_due_updates: 0,
                    partial_due_updates: 0,
                    missed_due_updates: due,
                    pending_updates: pending,
                    complete_pending_updates: 0,
                    captured_area_km2: 0.0,
                    ready_area_km2: 0.0,
                    due_captured_area_km2: 0.0,
                }
            })
            .collect();
        Ok(Self { jobs,atlas,options,bytes_per_km2,retain_ledger:false,state:BaselineState {captures:Vec::new(),active:BTreeMap::new(),expiry:BTreeSet::new(),
            report:Report {schema_version:1,mode:"unconstrained_capture_baseline".into(),epoch_unix_s:epoch,
                swath_km:swath,capture,options:options.clone(),jobs:reports,
                satellites:(0..satellites).map(|satellite|SatelliteReport {satellite,retained_crop_bytes:0.0,
                    storage_capacity_bytes:options.storage_bytes_per_satellite,first_capacity_exceeded_s:None}).collect(),
                opportunities:OpportunityStats::default(),retained_physical_bytes:0.0,fulfilled_logical_bytes:0.0,
                reuse_saved_bytes:0.0,reused_physical_bytes:0.0,dropped_jobs:0,
                assumptions:vec![
                    "No admission or execution: configured finite storage is measured, never enforced in this baseline.".into(),
                    "Only newly useful cropped polygons are retained; all retained crops remain for the full run, without downlink, deletion or processing.".into(),
                    "A physical capture may satisfy multiple jobs only within their acquisition slot and local solar-time/daylight constraints.".into(),
                    "Slots are [phase+n*cadence, phase+(n+1)*cadence); no pre-run history or carry-over. Incomplete final slots are censored.".into(),
                    "Independent crops become ready incrementally; mosaics require full geometric coverage within a slot, up to the reported area tolerance.".into(),
                    "Spherical Earth, interpolated swept scanlines and densified equal-area polygons. Halve step and edge spacing for convergence.".into(),
                    "Job local time is evaluated at its anchor. Crop solar elevation and timing are checked at interval start/midpoint/end; boundary slivers may be conservatively rejected.".into(),
                    "Priority: larger values win; future overload drops whole jobs strictly lowest priority first, ties by job key. No drops here.".into(),
                    "Clouds, write throughput, energy, compute capacity, processing histories and delivery are not modeled.".into(),
                ]} } })
    }
    fn expire(&mut self, time: f64) {
        // Nanosecond keys avoid float ordering; no expiry is earlier than its deadline.
        let now = (time * 1e9).floor() as u64;
        while self.state.expiry.first().is_some_and(|(t, _, _)| *t <= now) {
            let (_, job, slot) = self.state.expiry.pop_first().unwrap();
            if let Some(request) = self.state.active.remove(&(job, slot)) {
                let r = &mut self.state.report.jobs[job];
                r.due_captured_area_km2 += request.area.min(r.geometric_area_km2);
                if request.complete {
                    r.complete_due_updates += 1;
                } else if request.area > 0.0 {
                    r.partial_due_updates += 1;
                }
            }
        }
    }
    pub fn accept(&mut self, opportunity: &Opportunity) -> Result<(), String> {
        self.expire(opportunity.start_s);
        let mut physical = MultiPolygon::new(vec![]);
        let mut shared = MultiPolygon::new(vec![]);
        let mut logical = 0.0;
        let mut allocations = Vec::new();
        for crop in &opportunity.crops {
            let job = &self.jobs[crop.job];
            let key = (crop.job, crop.slot);
            if !self.state.active.contains_key(&key) {
                let deadline = (job.acquisition.phase_hours
                    + (crop.slot + 1) as f64 * job.cadence_hours)
                    * 3600.0;
                self.state
                    .expiry
                    .insert(((deadline * 1e9).ceil() as u64, crop.job, crop.slot));
                self.state.active.insert(
                    key,
                    Request {
                        patches: BTreeMap::new(),
                        area: 0.0,
                        complete: false,
                    },
                );
            }
            let request = self.state.active.get_mut(&key).unwrap();
            if request.complete {
                continue;
            }
            let covered = request
                .patches
                .entry(opportunity.patch)
                .or_insert_with(|| MultiPolygon::new(vec![]));
            let new = crop.geometry.difference(covered);
            let area = new.unsigned_area();
            if area <= 1e-10 {
                continue;
            }
            *covered = covered.union(&new);
            request.area += area;
            logical += area;
            shared = shared.union(&physical.intersection(&new));
            physical = physical.union(&new);
            if self.retain_ledger {
                allocations.push(Allocation {
                    job: crop.job,
                    slot: crop.slot,
                    geometry: new,
                });
            }
            let r = &mut self.state.report.jobs[crop.job];
            r.captured_area_km2 += area;
            if job.acquisition.fulfilment == Fulfilment::Independent {
                r.ready_area_km2 += area;
            }
            if request.area
                >= self.atlas.job_areas_km2[crop.job] * (1.0 - self.options.mosaic_area_tolerance)
            {
                request.complete = true;
                request.patches.clear();
                if job.acquisition.fulfilment == Fulfilment::Mosaic {
                    r.ready_area_km2 += self.atlas.job_areas_km2[crop.job];
                }
            }
        }
        let physical_bytes = physical.unsigned_area() * self.bytes_per_km2;
        self.state.report.retained_physical_bytes += physical_bytes;
        self.state.report.fulfilled_logical_bytes += logical * self.bytes_per_km2;
        self.state.report.reused_physical_bytes += shared.unsigned_area() * self.bytes_per_km2;
        let s = self
            .report
            .satellites
            .get_mut(opportunity.satellite)
            .ok_or("Unknown satellite")?;
        s.retained_crop_bytes += physical_bytes;
        if s.retained_crop_bytes > s.storage_capacity_bytes as f64
            && s.first_capacity_exceeded_s.is_none()
        {
            s.first_capacity_exceeded_s = Some(opportunity.end_s);
        }
        if self.retain_ledger && physical_bytes > 0.0 {
            self.state.captures.push(RetainedCapture {
                satellite: opportunity.satellite,
                patch: opportunity.patch,
                start_s: opportunity.start_s,
                end_s: opportunity.end_s,
                bytes: physical_bytes,
                geometry: physical,
                allocations,
            });
        }
        Ok(())
    }
    pub fn into_state(self) -> BaselineState {
        self.state
    }
    pub(crate) fn resume(
        jobs: &'a [Job],
        atlas: &'a Atlas,
        options: &'a Options,
        state: BaselineState,
    ) -> Self {
        let bytes_per_km2 = state.report.capture.stored_bytes_for_area(1.0).unwrap();
        Self {
            jobs,
            atlas,
            options,
            bytes_per_km2,
            state,
            retain_ledger: true,
        }
    }
    /// Set reports to the committed clock, including deadlines with no captures.
    pub(crate) fn update_clock(&mut self, time: f64) {
        self.expire(time);
        for (job, r) in self.jobs.iter().zip(&mut self.state.report.jobs) {
            let intervals = (time - job.acquisition.phase_hours * 3600.0).max(0.0)
                / (job.cadence_hours * 3600.0);
            r.due_updates = (intervals + 1e-10).floor() as u64;
            r.pending_updates = u64::from(intervals > r.due_updates as f64 + 1e-10);
            r.complete_pending_updates = 0;
            r.missed_due_updates = r
                .due_updates
                .saturating_sub(r.complete_due_updates + r.partial_due_updates);
        }
        for ((job, _), request) in &self.state.active {
            if request.complete {
                self.state.report.jobs[*job].complete_pending_updates += 1;
            }
        }
        self.state.report.reuse_saved_bytes = (self.state.report.fulfilled_logical_bytes
            - self.state.report.retained_physical_bytes)
            .max(0.0);
    }
    pub fn finish(mut self, opportunities: OpportunityStats) -> Report {
        self.expire(self.options.duration_s);
        for ((job, _), request) in &self.state.active {
            if request.complete {
                self.state.report.jobs[*job].complete_pending_updates += 1;
            }
        }
        for r in &mut self.state.report.jobs {
            r.missed_due_updates = r.due_updates - r.complete_due_updates - r.partial_due_updates;
        }
        self.state.report.reuse_saved_bytes = (self.state.report.fulfilled_logical_bytes
            - self.state.report.retained_physical_bytes)
            .max(0.0);
        self.state.report.opportunities = opportunities;
        self.state.report
    }
}

/// Optional JSONL output retains actual crop polygons and timing, making the
/// opportunity stream reusable by a later scheduler without GUI state.
pub fn opportunity_json(op: &Opportunity, jobs: &[Job], atlas: &Atlas) -> serde_json::Value {
    let chart = &atlas.patches[op.patch].chart;
    let crops:Vec<_>=op.crops.iter().map(|crop| {
        let mut polygons=geometry::coordinates(&crop.geometry);
        for p in polygons.iter_mut().flatten().flatten() {
            let v=chart.inverse(geo::Coord {x:p[0],y:p[1]});
            *p=[{
                        let origin = (-chart.center[2]).atan2(chart.center[0]).to_degrees();
                        let longitude = (-v[2]).atan2(v[0]).to_degrees();
                        (origin + (longitude-origin+180.0).rem_euclid(360.0)-180.0).clamp(-180.0,180.0)
                    },v[1].clamp(-1.0,1.0).asin().to_degrees()];
        }
        serde_json::json!({"job_key":jobs[crop.job].aoi.key(),"slot":crop.slot,"start_s":crop.start_s,
            "end_s":crop.end_s,"area_km2":crop.geometry.unsigned_area(),"polygons_lon_lat":polygons})
    }).collect();
    serde_json::json!({"satellite":op.satellite,"patch":op.patch,"start_s":op.start_s,"end_s":op.end_s,"crops":crops})
}

impl BaselineState {
    pub(crate) fn validate(
        &self,
        jobs: &[Job],
        atlas: &Atlas,
        satellites: usize,
        now: f64,
    ) -> Result<(), String> {
        use geo::{CoordsIter, Validation};
        let valid_shape = |shape: &Shape| {
            shape
                .coords_iter()
                .all(|p| p.x.is_finite() && p.y.is_finite())
                && shape.is_valid()
        };
        let invalid = || "Inconsistent capture checkpoint state".to_string();
        if self.report.jobs.len() != jobs.len() || self.report.satellites.len() != satellites {
            return Err(invalid());
        }
        for (i, (j, r)) in jobs.iter().zip(&self.report.jobs).enumerate() {
            let due = (((now - j.acquisition.phase_hours * 3600.0).max(0.0)
                / (j.cadence_hours * 3600.0))
                + 1e-10)
                .floor() as u64;
            if r.job_key != j.aoi.key()
                || r.geometric_area_km2 != atlas.job_areas_km2[i]
                || r.due_updates != due
                || r.complete_due_updates + r.partial_due_updates + r.missed_due_updates != due
                || [
                    r.captured_area_km2,
                    r.ready_area_km2,
                    r.due_captured_area_km2,
                ]
                .iter()
                .any(|x| !x.is_finite() || *x < 0.0)
            {
                return Err(invalid());
            }
        }
        let mut expiry = BTreeSet::new();
        for (&(job, slot), r) in &self.active {
            let j = jobs.get(job).ok_or_else(invalid)?;
            let deadline =
                (j.acquisition.phase_hours + (slot + 1) as f64 * j.cadence_hours) * 3600.0;
            if deadline <= now
                || !r.area.is_finite()
                || r.area < 0.0
                || r.patches
                    .iter()
                    .any(|(p, s)| *p >= atlas.patches.len() || !valid_shape(s))
            {
                return Err(invalid());
            }
            expiry.insert(((deadline * 1e9).ceil() as u64, job, slot));
        }
        if expiry != self.expiry {
            return Err(invalid());
        }
        let mut totals = vec![0.0; satellites];
        let mut last = 0.0;
        for c in &self.captures {
            if c.satellite >= satellites
                || c.patch >= atlas.patches.len()
                || !c.start_s.is_finite()
                || !c.end_s.is_finite()
                || c.start_s < 0.0
                || c.start_s >= c.end_s
                || c.end_s > now
                || c.end_s < last
                || !c.bytes.is_finite()
                || c.bytes <= 0.0
                || !valid_shape(&c.geometry)
                || c.allocations.is_empty()
                || c.allocations
                    .iter()
                    .any(|a| a.job >= jobs.len() || !valid_shape(&a.geometry))
            {
                return Err(invalid());
            }
            last = c.end_s;
            totals[c.satellite] += c.bytes;
        }
        let close = |a: f64, b: f64| {
            a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-8 * a.abs().max(1.0)
        };
        for (i, (r, total)) in self.report.satellites.iter().zip(totals.iter()).enumerate() {
            if r.satellite != i
                || !close(r.retained_crop_bytes, *total)
                || r.storage_capacity_bytes != self.report.options.storage_bytes_per_satellite
            {
                return Err(invalid());
            }
        }
        if !close(self.report.retained_physical_bytes, totals.iter().sum()) {
            return Err(invalid());
        }
        Ok(())
    }
}
