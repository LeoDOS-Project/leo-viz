//! Finite-horizon acquisition feasibility, using the capture model's exact job
//! windows and cumulative polygons. This is not a constrained schedule or an
//! all-time revisit certificate. No retained-capture ledger is allocated.
use crate::{
    baseline::{self, Baseline, Report},
    geometry::Atlas,
    session::Inputs,
};

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

#[derive(Clone, Debug, Default)]
pub struct Counts {
    pub satisfied: usize,
    /// At least one due slot had no usable coverage.
    pub missed: usize,
    /// No empty slots, but at least one slot had incomplete AOI coverage.
    pub partial: usize,
    pub untested: usize,
    pub due_updates: u64,
    pub complete_updates: u64,
}
impl Counts {
    fn add(&mut self, job: &baseline::JobReport) {
        self.due_updates += job.due_updates;
        self.complete_updates += job.complete_due_updates;
        if job.due_updates == 0 {
            self.untested += 1;
        } else if job.complete_due_updates == job.due_updates {
            self.satisfied += 1;
        } else if job.missed_due_updates > 0 {
            self.missed += 1;
        } else {
            self.partial += 1;
        }
    }
    pub fn all_satisfied(&self) -> bool {
        self.satisfied > 0 && self.missed == 0 && self.partial == 0 && self.untested == 0
    }
}
pub struct Assessment {
    pub report: Report,
    pub counts: Counts,
    pub cadences: Vec<(f64, Counts)>,
    /// Index into the original job order; failures and untested jobs first.
    pub details: Vec<usize>,
}
impl Assessment {
    fn new(report: Report, inputs: &Inputs) -> Self {
        let mut counts = Counts::default();
        let mut cadences = std::collections::BTreeMap::<u64, Counts>::new();
        for (job, result) in inputs.jobs.iter().zip(&report.jobs) {
            counts.add(result);
            cadences
                .entry(job.cadence_hours.to_bits())
                .or_default()
                .add(result);
        }
        let mut details: Vec<_> = (0..report.jobs.len()).collect();
        details.sort_by_key(|&i| {
            let j = &report.jobs[i];
            (
                if j.missed_due_updates > 0 {
                    0
                } else if j.partial_due_updates > 0 {
                    1
                } else if j.due_updates == 0 {
                    2
                } else {
                    3
                },
                std::cmp::Reverse(j.priority),
                i,
            )
        });
        Self {
            report,
            counts,
            cadences: cadences
                .into_iter()
                .map(|(h, c)| (f64::from_bits(h), c))
                .collect(),
            details,
        }
    }
}

pub fn evaluate(
    inputs: &Inputs,
    mut progress: impl FnMut(Progress) -> Result<(), String>,
) -> Result<Assessment, String> {
    progress(Progress {
        stage: Stage::Validate,
        fraction: 0.0,
    })?;
    inputs.validate()?;
    // Preserve the caller's cancellation/resource error rather than replacing
    // it with the geometry engine's generic cancellation message.
    let mut stopped = None;
    let mut update = |stage, fraction| match progress(Progress { stage, fraction }) {
        Ok(()) => true,
        Err(error) => {
            stopped = Some(error);
            false
        }
    };
    let atlas = Atlas::new(
        &inputs.jobs,
        &inputs.regions,
        inputs.options.cell_deg,
        inputs.options.edge_km,
        |f| update(Stage::Geometry, f),
    );
    let atlas = match atlas {
        Ok(a) => a,
        Err(e) => return Err(stopped.unwrap_or(e)),
    };
    let mut baseline = Baseline::new(
        &inputs.jobs,
        &atlas,
        inputs.satellites.len(),
        inputs.epoch(),
        inputs.swath_km,
        inputs.capture,
        &inputs.options,
    )?;
    let opportunities = baseline::opportunities(
        &inputs.jobs,
        &atlas,
        inputs.satellites.len(),
        inputs.epoch(),
        inputs.swath_km,
        &inputs.options,
        |s, t| {
            inputs.satellites[s]
                .orbit
                .position(inputs.start_s + t, inputs.orbit_epoch_unix_s)
        },
        |op| baseline.accept(&op),
        |f| update(Stage::Forecast, f),
    );
    let opportunities = match opportunities {
        Ok(o) => o,
        Err(e) => return Err(stopped.unwrap_or(e)),
    };
    debug_assert!(baseline.captures.is_empty());
    Ok(Assessment::new(baseline.finish(opportunities), inputs))
}

#[cfg(test)]
mod tests;
