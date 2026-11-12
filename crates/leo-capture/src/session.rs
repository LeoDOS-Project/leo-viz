//! Resumable, deterministic capture-only sessions. No execution or admission.
use crate::{
    baseline::{self, Baseline, BaselineState, Options, Report},
    geometry::{Atlas, Polygons, Region},
    workload::Job,
};
use leo_sim::{
    acquisition::CaptureProfile,
    geometry::{gmst, norm, Vector},
};
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    io::{Read, Write},
    path::Path,
    sync::Arc,
};

/// Preparation has several distinct stages; percentages apply only to the
/// named stage, never to an invented overall completion estimate.
#[derive(Clone, Debug, PartialEq)]
pub enum PreparationStage {
    ReadCheckpoint,
    ValidateInputs,
    BuildGeometry,
    BuildIndex,
    InitializeCoverage,
    ValidateCheckpoint,
    VerifyPositions,
    Ready,
}
#[derive(Clone, Debug)]
pub struct PreparationProgress {
    pub stage: PreparationStage,
    pub completed: usize,
    pub total: usize,
}
impl PreparationProgress {
    fn report(
        callback: &mut impl FnMut(Self) -> bool,
        stage: PreparationStage,
        completed: usize,
        total: usize,
    ) -> Result<(), String> {
        if callback(Self {
            stage,
            completed,
            total,
        }) {
            Ok(())
        } else {
            Err("Cancelled.".into())
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Orbit {
    Analytical {
        elements: leo_sim::orbit::Elements,
    },
    Sgp4 {
        constants: sgp4::Constants,
        epoch_minutes: f64,
    },
}
impl Orbit {
    pub fn position(&self, time: f64, epoch: f64) -> Result<Vector, String> {
        let p = match self {
            Self::Analytical { elements } => elements.sample(time).xyz,
            Self::Sgp4 {
                constants,
                epoch_minutes,
            } => {
                let p = constants
                    .propagate(sgp4::MinutesSinceEpoch(
                        (epoch + time) / 60.0 - epoch_minutes,
                    ))
                    .map_err(|e| format!("SGP4: {e:?}"))?;
                [p.position[0], p.position[2], -p.position[1]]
            }
        };
        if p.iter().any(|v| !v.is_finite()) || norm(p) <= leo_sim::geometry::EARTH_RADIUS_KM {
            return Err("Invalid satellite position".into());
        }
        Ok(p)
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Satellite {
    pub name: String,
    pub constellation: String,
    pub orbit: Orbit,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Inputs {
    pub orbit_epoch_unix_s: f64,
    pub start_s: f64,
    pub satellites: Vec<Satellite>,
    pub jobs: Vec<Job>,
    pub regions: Vec<Region>,
    pub capture: CaptureProfile,
    pub swath_km: f64,
    pub options: Options,
}
impl Inputs {
    fn validate(&self) -> Result<(), String> {
        self.options.validate()?;
        if !self.epoch().is_finite()
            || !self.orbit_epoch_unix_s.is_finite()
            || !self.start_s.is_finite()
            || !self.swath_km.is_finite()
            || !(0.0..=500.0).contains(&self.swath_km)
            || self.swath_km == 0.0
            || self.jobs.is_empty()
            || self.satellites.is_empty()
        {
            return Err("Invalid session inputs".into());
        }
        self.capture
            .stored_bytes_for_area(1.0)
            .ok_or("Invalid sensor profile")?;
        for j in &self.jobs {
            j.validate()?;
        }
        for s in &self.satellites {
            s.orbit.position(self.start_s, self.orbit_epoch_unix_s)?;
            s.orbit.position(
                self.start_s + self.options.duration_s,
                self.orbit_epoch_unix_s,
            )?;
        }
        Ok(())
    }
    pub fn epoch(&self) -> f64 {
        self.orbit_epoch_unix_s + self.start_s
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    schema_version: u32,
    inputs: Inputs,
    next_tick: u64,
    /// Explicit inertial positions at the committed clock, verified on restore.
    positions_km: Vec<Vector>,
    state: BaselineState,
}
pub struct Session {
    inputs: Arc<Inputs>,
    atlas: Atlas,
    state: Option<BaselineState>,
    next_tick: u64,
    poisoned: bool,
    meter_cache: RefCell<Option<(u64, Meters)>>,
    window: RefCell<CaptureWindow>,
}
#[derive(Default)]
struct CaptureWindow {
    first: usize,
    seen: usize,
    bytes: Vec<f64>,
    counts: Vec<usize>,
}
#[derive(Clone)]
pub struct Meters {
    pub elapsed_s: f64,
    pub captured_last_24h_bytes: Vec<f64>,
    pub captures_last_24h: u64,
    pub captures_total: u64,
    pub due_updates: u64,
    pub complete_updates: u64,
    pub partial_updates: u64,
    pub failed_updates: u64,
    pub affected_jobs: u64,
    pub jobs_meeting_all_due_updates: u64,
    pub jobs_without_due_updates: u64,
}
impl Meters {
    pub fn constellation_last_24h_bytes(&self) -> f64 {
        self.captured_last_24h_bytes.iter().sum()
    }
}
impl Session {
    pub fn new(inputs: Inputs, mut progress: impl FnMut(f32) -> bool) -> Result<Self, String> {
        Self::prepare(inputs, |p| match p.stage {
            PreparationStage::BuildGeometry => progress(p.completed as f32 / p.total.max(1) as f32),
            PreparationStage::Ready => progress(1.0),
            _ => true,
        })
    }
    pub fn prepare(
        inputs: Inputs,
        mut progress: impl FnMut(PreparationProgress) -> bool,
    ) -> Result<Self, String> {
        PreparationProgress::report(
            &mut progress,
            PreparationStage::ValidateInputs,
            0,
            inputs.jobs.len(),
        )?;
        inputs.validate()?;
        let total = inputs.regions.len();
        let atlas = Atlas::new(
            &inputs.jobs,
            &inputs.regions,
            inputs.options.cell_deg,
            inputs.options.edge_km,
            |f| {
                progress(PreparationProgress {
                    stage: if f >= 1.0 {
                        PreparationStage::BuildIndex
                    } else {
                        PreparationStage::BuildGeometry
                    },
                    completed: (f * total as f32).round() as usize,
                    total,
                })
            },
        )?;
        PreparationProgress::report(
            &mut progress,
            PreparationStage::InitializeCoverage,
            0,
            inputs.jobs.len(),
        )?;
        let mut baseline = Baseline::new(
            &inputs.jobs,
            &atlas,
            inputs.satellites.len(),
            inputs.epoch(),
            inputs.swath_km,
            inputs.capture,
            &inputs.options,
        )?;
        baseline.update_clock(0.0);
        let state = baseline.into_state();
        PreparationProgress::report(&mut progress, PreparationStage::Ready, total, total)?;
        Ok(Self {
            inputs: Arc::new(inputs),
            atlas,
            state: Some(state),
            next_tick: 0,
            poisoned: false,
            meter_cache: RefCell::new(None),
            window: RefCell::new(CaptureWindow::default()),
        })
    }
    pub fn inputs(&self) -> &Inputs {
        &self.inputs
    }
    /// Immutable setup shared with viewers without duplicating all AOI geometry.
    pub fn shared_inputs(&self) -> Arc<Inputs> {
        self.inputs.clone()
    }
    pub fn elapsed_s(&self) -> f64 {
        (self.next_tick as f64 * self.inputs.options.step_s).min(self.inputs.options.duration_s)
    }
    pub fn finished(&self) -> bool {
        self.elapsed_s() >= self.inputs.options.duration_s
    }
    pub fn state(&self) -> &BaselineState {
        self.state.as_ref().unwrap()
    }
    pub fn report(&self) -> &Report {
        &self.state().report
    }
    pub fn positions(&self) -> Result<Vec<Vector>, String> {
        self.inputs
            .satellites
            .iter()
            .map(|s| {
                s.orbit.position(
                    self.inputs.start_s + self.elapsed_s(),
                    self.inputs.orbit_epoch_unix_s,
                )
            })
            .collect()
    }
    pub fn ground_positions(&self) -> Result<Vec<[f64; 2]>, String> {
        let angle = gmst(self.inputs.epoch() + self.elapsed_s());
        Ok(self
            .positions()?
            .into_iter()
            .map(|p| {
                [
                    (-p[2].atan2(p[0]) - angle).to_degrees().rem_euclid(360.0),
                    (p[1] / norm(p)).asin().to_degrees(),
                ]
            })
            .map(|[lon, lat]| [(lon + 180.0).rem_euclid(360.0) - 180.0, lat])
            .collect())
    }
    /// Advance whole fixed ticks. Callers poll cancellation between these units;
    /// a checkpoint is only ever taken at a fully committed tick boundary.
    pub fn advance(&mut self, ticks: u64) -> Result<(), String> {
        if self.poisoned {
            return Err(
                "Session stopped after a propagation/geometry error; reload the last checkpoint."
                    .into(),
            );
        }
        if self.finished() || ticks == 0 {
            return Ok(());
        }
        let last = (self.next_tick.saturating_add(ticks))
            .min((self.inputs.options.duration_s / self.inputs.options.step_s).ceil() as u64);
        let mut baseline = Baseline::resume(
            &self.inputs.jobs,
            &self.atlas,
            &self.inputs.options,
            self.state.take().unwrap(),
        );
        let result = baseline::opportunities_range(
            &self.inputs.jobs,
            &self.atlas,
            self.inputs.satellites.len(),
            self.inputs.epoch(),
            self.inputs.swath_km,
            &self.inputs.options,
            self.next_tick,
            last,
            std::mem::take(&mut baseline.report.opportunities),
            if self.inputs.jobs.len() >= 1000 {
                crate::parallel::worker_count(self.inputs.satellites.len())
            } else {
                1
            },
            |s, t| {
                self.inputs.satellites[s]
                    .orbit
                    .position(self.inputs.start_s + t, self.inputs.orbit_epoch_unix_s)
            },
            |op| baseline.accept(&op),
            |_| true,
        );
        match result {
            Ok(s) => {
                self.next_tick = last;
                let t =
                    (last as f64 * self.inputs.options.step_s).min(self.inputs.options.duration_s);
                baseline.update_clock(t);
                baseline.report.opportunities = s;
                self.state = Some(baseline.into_state());
                Ok(())
            }
            Err(e) => {
                self.state = Some(baseline.into_state());
                self.poisoned = true;
                Err(e)
            }
        }
    }
    pub fn meters(&self) -> Meters {
        if let Some((tick, meters)) = &*self.meter_cache.borrow() {
            if *tick == self.next_tick {
                return meters.clone();
            }
        }
        let now = self.elapsed_s();
        let captures = &self.state().captures;
        let mut window = self.window.borrow_mut();
        window.bytes.resize(self.inputs.satellites.len(), 0.0);
        window.counts.resize(self.inputs.satellites.len(), 0);
        while window.first < window.seen && captures[window.first].end_s <= now - 86400.0 {
            let c = &captures[window.first];
            window.bytes[c.satellite] = (window.bytes[c.satellite] - c.bytes).max(0.0);
            window.counts[c.satellite] -= 1;
            if window.counts[c.satellite] == 0 {
                window.bytes[c.satellite] = 0.0;
            }
            window.first += 1;
        }
        // A restored/fast-forwarded session may contain many days of history.
        // Skip newly observed records already outside the window instead of
        // adding and subtracting huge totals (and losing precision).
        let first_new =
            window.seen + captures[window.seen..].partition_point(|c| c.end_s <= now - 86400.0);
        if first_new > window.seen {
            window.first = first_new;
        }
        for c in &captures[first_new..] {
            window.bytes[c.satellite] += c.bytes;
            window.counts[c.satellite] += 1;
        }
        window.seen = captures.len();
        let mut meters = Meters {
            elapsed_s: now,
            captured_last_24h_bytes: window.bytes.clone(),
            captures_last_24h: (window.seen - window.first) as u64,
            captures_total: self.state().captures.len() as u64,
            due_updates: 0,
            complete_updates: 0,
            partial_updates: 0,
            failed_updates: 0,
            affected_jobs: 0,
            jobs_meeting_all_due_updates: 0,
            jobs_without_due_updates: 0,
        };
        for j in &self.report().jobs {
            meters.due_updates += j.due_updates;
            meters.complete_updates += j.complete_due_updates;
            meters.partial_updates += j.partial_due_updates;
            let failures = j.due_updates - j.complete_due_updates;
            meters.failed_updates += failures;
            meters.affected_jobs += u64::from(failures > 0);
            meters.jobs_meeting_all_due_updates += u64::from(j.due_updates > 0 && failures == 0);
            meters.jobs_without_due_updates += u64::from(j.due_updates == 0);
        }
        *self.meter_cache.borrow_mut() = Some((self.next_tick, meters.clone()));
        meters
    }
    pub fn footprint(&self, capture: &baseline::RetainedCapture) -> Polygons {
        let chart = &self.atlas.patches[capture.patch].chart;
        let mut polygons = crate::geometry::coordinates(&capture.geometry);
        for p in polygons.iter_mut().flatten().flatten() {
            let v = chart.inverse(geo::Coord { x: p[0], y: p[1] });
            *p = [
                (-v[2]).atan2(v[0]).to_degrees(),
                v[1].clamp(-1.0, 1.0).asin().to_degrees(),
            ];
        }
        polygons
    }
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if self.poisoned {
            return Err("Cannot save a session after a failed step".into());
        }
        // Stream directly to a sibling temporary file; never overwrite a valid
        // checkpoint with an incomplete serialization or interrupted write.
        let tmp = path.with_extension(format!(
            "{}.{}.tmp",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let result = (|| -> Result<(), String> {
            let file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp)
                .map_err(|e| e.to_string())?;
            let mut gzip = flate2::write::GzEncoder::new(
                std::io::BufWriter::new(file),
                flate2::Compression::fast(),
            );
            // Serialize references to avoid duplicating the complete retained ledger.
            #[derive(Serialize)]
            struct Saved<'a> {
                schema_version: u32,
                inputs: &'a Inputs,
                next_tick: u64,
                positions_km: Vec<Vector>,
                state: &'a BaselineState,
            }
            serde_json::to_writer(
                &mut gzip,
                &Saved {
                    schema_version: 2,
                    inputs: &self.inputs,
                    next_tick: self.next_tick,
                    positions_km: self.positions()?,
                    state: self.state(),
                },
            )
            .map_err(|e| e.to_string())?;
            let mut writer = gzip.finish().map_err(|e| e.to_string())?;
            writer.flush().map_err(|e| e.to_string())?;
            writer.get_ref().sync_all().map_err(|e| e.to_string())?;
            std::fs::rename(&tmp, path).map_err(|e| e.to_string())?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(tmp);
        }
        result
    }
    pub fn load(path: &Path, mut progress: impl FnMut(f32) -> bool) -> Result<Self, String> {
        Self::load_with_progress(path, |p| match p.stage {
            PreparationStage::BuildGeometry => progress(p.completed as f32 / p.total.max(1) as f32),
            PreparationStage::Ready => progress(1.0),
            _ => true,
        })
    }
    pub fn load_with_progress(
        path: &Path,
        mut progress: impl FnMut(PreparationProgress) -> bool,
    ) -> Result<Self, String> {
        PreparationProgress::report(&mut progress, PreparationStage::ReadCheckpoint, 0, 0)?;
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let decoder = flate2::read::GzDecoder::new(std::io::BufReader::new(file));
        struct ProgressReader<R, F> {
            inner: R,
            progress: F,
            read: usize,
            next: usize,
        }
        impl<R: Read, F: FnMut(PreparationProgress) -> bool> Read for ProgressReader<R, F> {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                if self.read >= self.next {
                    if !(self.progress)(PreparationProgress {
                        stage: PreparationStage::ReadCheckpoint,
                        completed: self.read,
                        total: 0,
                    }) {
                        return Err(std::io::Error::other("Cancelled while reading checkpoint"));
                    }
                    self.next = self.read.saturating_add(1024 * 1024);
                }
                let n = self.inner.read(buffer)?;
                self.read += n;
                Ok(n)
            }
        }
        let reader = ProgressReader {
            inner: decoder.take(2 * 1024 * 1024 * 1024),
            progress: &mut progress,
            read: 0,
            next: 0,
        };
        let checkpoint: Checkpoint = serde_json::from_reader(std::io::BufReader::new(reader))
            .map_err(|e| format!("Invalid capture checkpoint: {e}"))?;
        if !(1..=2).contains(&checkpoint.schema_version) {
            return Err("Unsupported capture checkpoint version".into());
        }
        let mut session = Self::prepare(checkpoint.inputs, |p| {
            if p.stage == PreparationStage::Ready {
                true
            } else {
                progress(p)
            }
        })?;
        if checkpoint.next_tick
            > (session.inputs.options.duration_s / session.inputs.options.step_s).ceil() as u64
        {
            return Err("Checkpoint clock exceeds horizon".into());
        }
        session.next_tick = checkpoint.next_tick;
        PreparationProgress::report(
            &mut progress,
            PreparationStage::ValidateCheckpoint,
            0,
            checkpoint.state.captures.len(),
        )?;
        session.validate_state(&checkpoint.state, &mut progress)?;
        PreparationProgress::report(
            &mut progress,
            PreparationStage::VerifyPositions,
            0,
            checkpoint.positions_km.len(),
        )?;
        let positions = session.positions()?;
        if positions.len() != checkpoint.positions_km.len()
            || positions
                .iter()
                .zip(&checkpoint.positions_km)
                .any(|(a, b)| {
                    a.iter()
                        .zip(b)
                        .any(|(x, y)| !y.is_finite() || (x - y).abs() > 1e-6)
                })
        {
            return Err("Checkpoint positions do not match saved constellation and clock".into());
        }
        session.state = Some(checkpoint.state);
        PreparationProgress::report(
            &mut progress,
            PreparationStage::Ready,
            session.inputs.jobs.len(),
            session.inputs.jobs.len(),
        )?;
        Ok(session)
    }
    fn validate_state(
        &self,
        state: &BaselineState,
        progress: &mut impl FnMut(PreparationProgress) -> bool,
    ) -> Result<(), String> {
        state.validate(
            &self.inputs.jobs,
            &self.atlas,
            self.inputs.satellites.len(),
            self.elapsed_s(),
            |completed, total| {
                progress(PreparationProgress {
                    stage: PreparationStage::ValidateCheckpoint,
                    completed,
                    total,
                })
            },
        )?;
        let r = &state.report;
        if r.options != self.inputs.options
            || r.capture != self.inputs.capture
            || r.epoch_unix_s != self.inputs.epoch()
            || r.swath_km != self.inputs.swath_km
        {
            return Err("Checkpoint sensor/configuration mismatch".into());
        }
        Ok(())
    }
}
