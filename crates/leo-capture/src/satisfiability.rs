//! Finite-forecast geometric access and revisit checks over complete AOI polygons.
//! No capture allocation, storage, data-volume accounting or execution schedule.
use crate::{
    baseline::{self, Opportunity},
    geometry::{Atlas, Shape},
    session::Inputs,
};
use geo::{Area, BooleanOps, BoundingRect, Intersects, Rect};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub duration_s: f64,
    pub step_s: f64,
    pub cell_deg: f64,
    pub edge_km: f64,
    pub area_tolerance: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self::from_baseline(&baseline::Options::default())
    }
}
impl Options {
    fn from_baseline(o: &baseline::Options) -> Self {
        Self {
            duration_s: o.duration_s,
            step_s: o.step_s,
            cell_deg: o.cell_deg,
            edge_km: o.edge_km,
            area_tolerance: o.mosaic_area_tolerance,
        }
    }
}
impl From<&Options> for baseline::Options {
    fn from(o: &Options) -> Self {
        Self {
            duration_s: o.duration_s,
            step_s: o.step_s,
            cell_deg: o.cell_deg,
            edge_km: o.edge_km,
            mosaic_area_tolerance: o.area_tolerance,
            ..Self::default()
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Validate,
    Geometry,
    Forecast,
}
#[derive(Clone, Copy, Debug)]
pub struct Progress {
    pub stage: Stage,
    pub fraction: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Satisfied,
    Uncovered,
    TooSlow,
    ResolutionLimited,
    Untested,
}
impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Self::Satisfied => "Satisfied",
            Self::Uncovered => "Incomplete coverage",
            Self::TooSlow => "Revisit too slow",
            Self::ResolutionLimited => "Refine time step",
            Self::Untested => "Forecast too short",
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct Counts {
    pub satisfied: usize,
    pub uncovered: usize,
    pub too_slow: usize,
    pub resolution_limited: usize,
    pub untested: usize,
}
impl Counts {
    fn add(&mut self, verdict: Verdict) {
        match verdict {
            Verdict::Satisfied => self.satisfied += 1,
            Verdict::Uncovered => self.uncovered += 1,
            Verdict::TooSlow => self.too_slow += 1,
            Verdict::ResolutionLimited => self.resolution_limited += 1,
            Verdict::Untested => self.untested += 1,
        }
    }
    pub fn all_satisfied(&self) -> bool {
        self.satisfied > 0
            && self.uncovered + self.too_slow + self.resolution_limited + self.untested == 0
    }
}
pub struct JobReport {
    pub job_key: String,
    pub priority: u16,
    pub cadence_hours: f64,
    pub coverage_fraction: f64,
    /// Bounds on the longest silence anywhere in the AOI, including forecast
    /// edges. A swept crop's acquisition time is bounded by its track segment.
    pub silence_min_s: f64,
    pub silence_max_s: f64,
    pub verdict: Verdict,
}
pub struct Assessment {
    pub jobs: Vec<JobReport>,
    pub satellite_count: usize,
    pub epoch_unix_s: f64,
    pub reach_km: f64,
    pub options: Options,
    pub counts: Counts,
    pub cadences: Vec<(f64, Counts)>,
    pub details: Vec<usize>,
}

struct Seen {
    shape: Shape,
    bounds: Rect<f64>,
    start: f64,
    end: f64,
}
#[derive(Default)]
struct Silence {
    min: f64,
    max: f64,
}
impl Silence {
    fn include(&mut self, min: f64, max: f64) {
        self.min = self.min.max(min.max(0.0));
        self.max = self.max.max(max.max(0.0));
    }
}
/// A spatial partition tagged with only the most recent access interval.
/// Older geometry is replaced on overlap; no history/retained-data ledger grows
/// with the number of passes. Distinct parts of an AOI keep distinct timestamps.
struct Revisit {
    pieces: BTreeMap<(usize, usize), Vec<Seen>>,
    silence: Vec<Silence>,
}
impl Revisit {
    fn new(jobs: usize) -> Self {
        Self {
            pieces: BTreeMap::new(),
            silence: (0..jobs).map(|_| Silence::default()).collect(),
        }
    }
    fn accept(&mut self, opportunity: Opportunity) -> Result<(), String> {
        for crop in opportunity.crops {
            let seen = self
                .pieces
                .entry((opportunity.patch, crop.job))
                .or_default();
            let Some(bounds) = crop.geometry.bounding_rect() else {
                continue;
            };
            let mut fresh = crop.geometry.unsigned_area();
            let mut next = Vec::with_capacity(seen.len() + 1);
            // Use the shared tick interval, not a job-slot fragment's time. All
            // satellites in one tick are processed together, so intervals are
            // monotonic even when a cadence boundary splits their crops.
            let (start, end) = (opportunity.start_s, opportunity.end_s);
            for old in seen.drain(..) {
                if !old.bounds.intersects(&bounds) {
                    next.push(old);
                    continue;
                }
                let overlap = old.shape.intersection(&crop.geometry).unsigned_area();
                if overlap <= 1e-10 {
                    next.push(old);
                    continue;
                }
                fresh -= overlap;
                self.silence[crop.job].include(start - old.end, end - old.start);
                let remainder = old.shape.difference(&crop.geometry);
                if remainder.unsigned_area() > 1e-10 {
                    if let Some(bounds) = remainder.bounding_rect() {
                        next.push(Seen {
                            shape: remainder,
                            bounds,
                            start: old.start,
                            end: old.end,
                        });
                    }
                }
            }
            if fresh > 1e-10 {
                self.silence[crop.job].include(start, end);
            }
            next.push(Seen {
                shape: crop.geometry,
                bounds,
                start,
                end,
            });
            *seen = next;
        }
        Ok(())
    }
    fn finish(mut self, inputs: &Inputs, atlas: &Atlas, options: Options) -> Assessment {
        let mut area = vec![0.0; inputs.jobs.len()];
        for ((_, job), pieces) in self.pieces {
            for piece in pieces {
                area[job] += piece.shape.unsigned_area();
                self.silence[job].include(
                    options.duration_s - piece.end,
                    options.duration_s - piece.start,
                );
            }
        }
        let jobs: Vec<_> = inputs
            .jobs
            .iter()
            .enumerate()
            .map(|(i, job)| {
                let fraction = (area[i] / atlas.job_areas_km2[i]).clamp(0.0, 1.0);
                let full = fraction >= 1.0 - options.area_tolerance;
                let mut silence = std::mem::take(&mut self.silence[i]);
                if !full {
                    silence.include(options.duration_s, options.duration_s);
                }
                let cadence = job.cadence_hours * 3600.0;
                let verdict = if options.duration_s + 1e-8 < cadence {
                    Verdict::Untested
                } else if !full {
                    Verdict::Uncovered
                } else if silence.min > cadence + 1e-8 {
                    Verdict::TooSlow
                } else if silence.max > cadence + 1e-8 {
                    Verdict::ResolutionLimited
                } else {
                    Verdict::Satisfied
                };
                JobReport {
                    job_key: job.aoi.key(),
                    priority: job.acquisition.priority,
                    cadence_hours: job.cadence_hours,
                    coverage_fraction: fraction,
                    silence_min_s: silence.min,
                    silence_max_s: silence.max,
                    verdict,
                }
            })
            .collect();
        let mut counts = Counts::default();
        let mut cadences = BTreeMap::<u64, Counts>::new();
        for job in &jobs {
            counts.add(job.verdict);
            cadences
                .entry(job.cadence_hours.to_bits())
                .or_default()
                .add(job.verdict);
        }
        let mut details: Vec<_> = (0..jobs.len()).collect();
        details.sort_by_key(|&i| {
            (
                match jobs[i].verdict {
                    Verdict::Uncovered => 0,
                    Verdict::TooSlow => 1,
                    Verdict::ResolutionLimited => 2,
                    Verdict::Untested => 3,
                    Verdict::Satisfied => 4,
                },
                std::cmp::Reverse(jobs[i].priority),
                i,
            )
        });
        Assessment {
            jobs,
            satellite_count: inputs.satellites.len(),
            epoch_unix_s: inputs.epoch(),
            reach_km: inputs.swath_km,
            options,
            counts,
            cadences: cadences
                .into_iter()
                .map(|(h, c)| (f64::from_bits(h), c))
                .collect(),
            details,
        }
    }
}

/// Uses the input orbit/job geometry and access width. Sensor byte settings and
/// satellite storage are intentionally irrelevant to geometric satisfiability.
pub fn evaluate(
    inputs: &Inputs,
    mut progress: impl FnMut(Progress) -> Result<(), String>,
) -> Result<Assessment, String> {
    progress(Progress {
        stage: Stage::Validate,
        fraction: 0.0,
    })?;
    let options = Options::from_baseline(&inputs.options);
    let geometric: baseline::Options = (&options).into();
    geometric.validate()?;
    if !inputs.epoch().is_finite()
        || !inputs.start_s.is_finite()
        || !inputs.orbit_epoch_unix_s.is_finite()
        || !(0.0..=500.0).contains(&inputs.swath_km)
        || inputs.swath_km == 0.0
        || inputs.jobs.is_empty()
        || inputs.satellites.is_empty()
    {
        return Err("Provide jobs, matching AOIs, satellites, a finite epoch and positive observation reach up to 500 km.".into());
    }
    // Cadence is a maximum allowed gap, not an arbitrary partition into slots.
    // Phase/fulfilment affect scheduling; solar constraints still filter usable access.
    let mut jobs = inputs.jobs.clone();
    for job in &mut jobs {
        job.validate()?;
        job.acquisition.phase_hours = 0.0;
    }
    let mut stopped = None;
    let mut update = |stage, fraction| match progress(Progress { stage, fraction }) {
        Ok(()) => true,
        Err(error) => {
            stopped = Some(error);
            false
        }
    };
    let atlas = Atlas::new(
        &jobs,
        &inputs.regions,
        options.cell_deg,
        options.edge_km,
        |f| update(Stage::Geometry, f),
    );
    let atlas = match atlas {
        Ok(a) => a,
        Err(e) => return Err(stopped.unwrap_or(e)),
    };
    let mut revisit = Revisit::new(jobs.len());
    let result = baseline::opportunities(
        &jobs,
        &atlas,
        inputs.satellites.len(),
        inputs.epoch(),
        inputs.swath_km,
        &geometric,
        |s, t| {
            inputs.satellites[s]
                .orbit
                .position(inputs.start_s + t, inputs.orbit_epoch_unix_s)
        },
        |op| revisit.accept(op),
        |f| update(Stage::Forecast, f),
    );
    if let Err(e) = result {
        return Err(stopped.unwrap_or(e));
    }
    Ok(revisit.finish(inputs, &atlas, options))
}

#[cfg(test)]
mod tests;
