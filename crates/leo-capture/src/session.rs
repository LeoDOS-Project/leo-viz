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
    io::{Read, Write},
    path::Path,
};

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
    inputs: Inputs,
    atlas: Atlas,
    state: Option<BaselineState>,
    next_tick: u64,
    poisoned: bool,
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
}
impl Meters {
    pub fn constellation_last_24h_bytes(&self) -> f64 {
        self.captured_last_24h_bytes.iter().sum()
    }
}
impl Session {
    pub fn new(inputs: Inputs, progress: impl FnMut(f32) -> bool) -> Result<Self, String> {
        inputs.validate()?;
        let atlas = Atlas::new(
            &inputs.jobs,
            &inputs.regions,
            inputs.options.cell_deg,
            inputs.options.edge_km,
            progress,
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
        Ok(Self {
            inputs,
            atlas,
            state: Some(state),
            next_tick: 0,
            poisoned: false,
        })
    }
    pub fn inputs(&self) -> &Inputs {
        &self.inputs
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
        let now = self.elapsed_s();
        let mut meters = Meters {
            elapsed_s: now,
            captured_last_24h_bytes: vec![0.0; self.inputs.satellites.len()],
            captures_last_24h: 0,
            captures_total: self.state().captures.len() as u64,
            due_updates: 0,
            complete_updates: 0,
            partial_updates: 0,
            failed_updates: 0,
            affected_jobs: 0,
        };
        for c in self
            .state()
            .captures
            .iter()
            .rev()
            .take_while(|c| c.end_s > now - 86400.0)
        {
            meters.captured_last_24h_bytes[c.satellite] += c.bytes;
            meters.captures_last_24h += 1;
        }
        for j in &self.report().jobs {
            meters.due_updates += j.due_updates;
            meters.complete_updates += j.complete_due_updates;
            meters.partial_updates += j.partial_due_updates;
            let failures = j.due_updates - j.complete_due_updates;
            meters.failed_updates += failures;
            meters.affected_jobs += u64::from(failures > 0);
        }
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
                    schema_version: 1,
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
    pub fn load(path: &Path, progress: impl FnMut(f32) -> bool) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let decoder = flate2::read::GzDecoder::new(std::io::BufReader::new(file));
        let checkpoint: Checkpoint = serde_json::from_reader(decoder.take(2 * 1024 * 1024 * 1024))
            .map_err(|e| format!("Invalid capture checkpoint: {e}"))?;
        if checkpoint.schema_version != 1 {
            return Err("Unsupported capture checkpoint version".into());
        }
        let mut session = Self::new(checkpoint.inputs, progress)?;
        if checkpoint.next_tick
            > (session.inputs.options.duration_s / session.inputs.options.step_s).ceil() as u64
        {
            return Err("Checkpoint clock exceeds horizon".into());
        }
        session.next_tick = checkpoint.next_tick;
        session.validate_state(&checkpoint.state)?;
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
        Ok(session)
    }
    fn validate_state(&self, state: &BaselineState) -> Result<(), String> {
        state.validate(
            &self.inputs.jobs,
            &self.atlas,
            self.inputs.satellites.len(),
            self.elapsed_s(),
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
