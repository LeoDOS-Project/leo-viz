//! Read-only acquisition forecast. Uses the same windows/geometry as capture,
//! with no session, retained ledger, checkpoint writes or simulation clock edits.
#[cfg(not(target_arch = "wasm32"))]
use crate::aoi::preview::Preview;
use crate::{config::PlanetConfig, revisit};
use chrono::{DateTime, Utc};
use eframe::egui;
#[cfg(not(target_arch = "wasm32"))]
use leo_capture::session::Inputs;
use leo_capture::{
    baseline::Options,
    satisfiability::{Assessment, Progress, Stage},
};
use leo_sim::acquisition::CaptureProfile;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    time::{Duration, Instant},
};

#[derive(Clone, PartialEq)]
struct Key {
    orbits: u64,
    revision: u64,
    capture: CaptureProfile,
    options: Options,
}
struct Worker {
    key: Key,
    cancel: Arc<AtomicBool>,
    progress: Arc<Mutex<Progress>>,
    rx: mpsc::Receiver<Result<Assessment, String>>,
    started: Instant,
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}
#[cfg(not(target_arch = "wasm32"))]
impl Worker {
    fn start(mut inputs: Inputs, layers: Vec<Arc<Preview>>, key: Key, ctx: egui::Context) -> Self {
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = cancel.clone();
        let progress = Arc::new(Mutex::new(Progress {
            stage: Stage::Validate,
            fraction: 0.0,
        }));
        let shared = progress.clone();
        let (tx, rx) = mpsc::sync_channel(1);
        std::thread::spawn(move || {
            leo_capture::resources::background_thread();
            let run = || {
                leo_capture::resources::check_memory()?;
                for layer in layers {
                    if stop.load(Ordering::Relaxed) {
                        return Err("Cancelled.".into());
                    }
                    inputs.regions.extend(layer.capture_regions());
                    leo_capture::resources::check_memory()?;
                }
                let mut last = Instant::now();
                leo_capture::satisfiability::evaluate(&inputs, |p| {
                    if stop.load(Ordering::Relaxed) {
                        return Err("Cancelled.".into());
                    }
                    if last.elapsed() >= Duration::from_millis(100)
                        || p.fraction == 0.0
                        || p.fraction == 1.0
                    {
                        leo_capture::resources::check_memory()?;
                        *shared.lock().unwrap() = p;
                        ctx.request_repaint();
                        let duty = leo_capture::resources::duty_percent() as f64 / 100.0;
                        let delay = last.elapsed().mul_f64((1.0 - duty) / duty);
                        let until = Instant::now() + delay;
                        while Instant::now() < until {
                            if stop.load(Ordering::Relaxed) {
                                return Err("Cancelled.".into());
                            }
                            std::thread::sleep(
                                until
                                    .saturating_duration_since(Instant::now())
                                    .min(Duration::from_millis(25)),
                            );
                        }
                        last = Instant::now();
                    }
                    Ok(())
                })
            };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run))
                .unwrap_or_else(|payload| {
                    let message = payload
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "unexpected geometry error".into());
                    Err(format!(
                        "Assessment stopped: {message}. No satisfiability result was produced."
                    ))
                });
            let _ = tx.send(result);
            ctx.request_repaint();
        });
        Self {
            key,
            cancel,
            progress,
            rx,
            started: Instant::now(),
        }
    }
}

pub(super) struct Panel {
    options: Options,
    worker: Option<Worker>,
    result: Option<(Key, Assessment)>,
    message: Option<String>,
}
impl Default for Panel {
    fn default() -> Self {
        Self {
            options: Options {
                duration_s: 30.0 * 86400.0,
                ..Default::default()
            },
            worker: None,
            result: None,
            message: None,
        }
    }
}
impl Panel {
    fn poll(&mut self, key: &Key, compatible: bool, ctx: &egui::Context) {
        if self
            .worker
            .as_ref()
            .is_some_and(|w| w.key != *key || !compatible)
        {
            self.worker = None;
            self.message = Some("Setup changed; assessment cancelled. Check again for the current jobs and constellation.".into());
        }
        if let Some(worker) = &self.worker {
            match worker.rx.try_recv() {
                Ok(result) => {
                    let worker = self.worker.take().unwrap();
                    match result {
                        Ok(report) => {
                            self.result = Some((worker.key.clone(), report));
                            self.message = None;
                        }
                        Err(error) => self.message = Some(error),
                    }
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.worker = None;
                    self.message =
                        Some("Assessment worker stopped. No new result was produced.".into());
                }
                Err(mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(Duration::from_millis(100))
                }
            }
        }
    }
}

impl super::Panel {
    pub(crate) fn satisfiability_ui(
        &mut self,
        ui: &mut egui::Ui,
        planet: &PlanetConfig,
        capture: CaptureProfile,
        width: f64,
        access: bool,
        start: f64,
        epoch: DateTime<Utc>,
    ) {
        #[cfg(target_arch = "wasm32")]
        let _ = start;
        self.satisfiability.options.storage_bytes_per_satellite =
            self.baseline.storage_capacity_bytes();
        let key = Key {
            orbits: revisit::fingerprint(planet, width, epoch),
            revision: self.revision,
            capture,
            options: self.satisfiability.options.clone(),
        };
        self.satisfiability.poll(&key, self.compatible, ui.ctx());
        egui::CollapsingHeader::new("Satisfiability").id_salt("job-satisfiability").default_open(false).show(ui, |ui| {
            ui.small("Checks cumulative AOI coverage in every job cadence window, including phase, local solar time and daylight. Starts at the current simulation time; playback does not move an existing forecast.");
            if access { ui.small("Uses the observable corridor as continuous access. Pointing schedules and simultaneous imaging capacity are not modeled."); }
            let busy = self.satisfiability.worker.is_some();
            ui.add_enabled_ui(!busy, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Forecast");
                    let mut days = self.satisfiability.options.duration_s / 86400.0;
                    if ui.add(egui::DragValue::new(&mut days).range(0.001..=30.0).speed(0.25).suffix(" days")).changed() {
                        self.satisfiability.options.duration_s = days * 86400.0;
                    }
                    if ui.button("Fit job cadences (up to 30 days)").clicked() {
                        let hours = self.jobs.iter().map(|j| j.cadence_hours + j.acquisition.phase_hours).fold(0.0_f64, f64::max);
                        self.satisfiability.options.duration_s = (hours * 3600.0).clamp(86.4, 30.0 * 86400.0);
                    }
                });
                ui.collapsing("Accuracy", |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Track step");
                        ui.add(egui::DragValue::new(&mut self.satisfiability.options.step_s).range(0.05..=30.0).speed(0.5).suffix(" s"));
                        ui.label("Polygon edge spacing");
                        ui.add(egui::DragValue::new(&mut self.satisfiability.options.edge_km).range(0.05..=10.0).speed(0.1).suffix(" km"));
                    });
                    ui.small("Halve both spacings to check convergence. Full coverage uses the capture model's one-part-per-million area tolerance.");
                });
            });
            let untested = self.jobs.iter().filter(|j| (j.cadence_hours + j.acquisition.phase_hours) * 3600.0 > self.satisfiability.options.duration_s + 1e-6).count();
            if untested > 0 { ui.colored_label(egui::Color32::YELLOW, format!("{untested} jobs have no deadline within this forecast and will remain untested (maximum 30 days).")); }
            if self.jobs.is_empty() { ui.label("Load or assign jobs first."); }
            if !self.compatible { ui.colored_label(egui::Color32::YELLOW, "AOIs and jobs do not match. Reassign jobs or load a matching workload."); }
            #[cfg(not(target_arch = "wasm32"))]
            if ui.add_enabled(!busy && self.compatible && self.assignment.is_none() && !self.jobs.is_empty(), egui::Button::new("Check current jobs")).clicked() {
                match revisit::capture_satellites(planet) {
                    Err(error) => self.satisfiability.message = Some(error),
                    Ok(satellites) => {
                        let mut key = key.clone();
                        key.options = self.satisfiability.options.clone();
                        let inputs = Inputs { orbit_epoch_unix_s: epoch.timestamp_millis() as f64 / 1000.0, start_s: start,
                            satellites, jobs: self.jobs.clone(), regions: Vec::new(), capture, swath_km: width, options: key.options.clone() };
                        self.satisfiability.worker = Some(Worker::start(inputs, self.layers.clone(), key, ui.ctx().clone()));
                        self.satisfiability.message = None;
                        self.satisfiability.result = None;
                    }
                }
            }
            #[cfg(target_arch = "wasm32")]
            ui.small("Acquisition satisfiability checks require the native app.");
            if let Some(worker) = &self.satisfiability.worker {
                let p = *worker.progress.lock().unwrap();
                let label = match p.stage { Stage::Validate => "Preparing jobs and AOI geometry", Stage::Geometry => "Building spatial coverage index", Stage::Forecast => "Checking acquisition windows" };
                ui.label(format!("{label} · {:.0} s elapsed", worker.started.elapsed().as_secs_f64()));
                ui.add(egui::ProgressBar::new(p.fraction).show_percentage());
                if p.stage == Stage::Forecast { ui.small(format!("{:.2} / {:.2} forecast days", p.fraction as f64 * worker.key.options.duration_s / 86400.0, worker.key.options.duration_s / 86400.0)); }
                if ui.button("Cancel assessment").clicked() {
                    self.satisfiability.worker = None;
                    self.satisfiability.message = Some("Assessment cancelled; no partial result is treated as a verdict.".into());
                }
            }
            if let Some(message) = &self.satisfiability.message { ui.colored_label(egui::Color32::YELLOW, message); }
            if let Some((previous, report)) = &self.satisfiability.result {
                let stale = previous != &Key { options: self.satisfiability.options.clone(), ..key } || !self.compatible;
                if stale { ui.colored_label(egui::Color32::YELLOW, "OUTDATED — setup changed. Run the check again."); }
                results(ui, report, stale);
            }
            ui.small("A satisfied job has full AOI coverage in every completed window. Independent crops can still produce useful partial output; mosaics wait for completion. Final unfinished windows are untested.");
            ui.small("This is a finite-forecast acquisition check, not an all-time guarantee or a constrained schedule. No clouds, energy, compute, downlink or job dropping. Shared captures are reused. Native worker, memory and CPU-duty limits apply.");
        });
    }
}

fn results(ui: &mut egui::Ui, result: &Assessment, stale: bool) {
    let c = &result.counts;
    let r = &result.report;
    let start = DateTime::from_timestamp_millis((r.epoch_unix_s * 1000.0) as i64)
        .map(|d| d.format("%Y-%m-%d %H:%M UTC").to_string())
        .unwrap_or_default();
    ui.separator();
    ui.small(format!(
        "{} jobs · {} satellites · {:.1} km observation width · {:.2} days from {start}",
        r.jobs.len(),
        r.satellites.len(),
        r.swath_km,
        r.options.duration_s / 86400.0
    ));
    let (color, verdict) = if stale {
        (egui::Color32::GRAY, "Previous forecast result")
    } else if c.missed + c.partial > 0 {
        (
            egui::Color32::LIGHT_RED,
            "Acquisition requirements not met in this forecast",
        )
    } else if c.all_satisfied() {
        (
            egui::Color32::LIGHT_GREEN,
            "All acquisition requirements met in this forecast",
        )
    } else {
        (
            egui::Color32::YELLOW,
            "Insufficient forecast: some jobs remain untested",
        )
    };
    ui.colored_label(color, verdict);
    ui.label(format!(
        "{} satisfied · {} partially covered · {} with missed windows · {} untested",
        c.satisfied, c.partial, c.missed, c.untested
    ));
    ui.label(format!(
        "{} / {} due acquisition updates complete",
        c.complete_updates, c.due_updates
    ));
    egui::Grid::new("satisfiability-cadences")
        .striped(true)
        .show(ui, |ui| {
            for label in ["Cadence", "Satisfied", "Partial", "Missed", "Untested"] {
                ui.strong(label);
            }
            ui.end_row();
            for (hours, counts) in &result.cadences {
                ui.label(super::duration(*hours));
                for n in [
                    counts.satisfied,
                    counts.partial,
                    counts.missed,
                    counts.untested,
                ] {
                    ui.label(n.to_string());
                }
                ui.end_row();
            }
        });
    let overloaded = r
        .satellites
        .iter()
        .filter(|s| s.first_capacity_exceeded_s.is_some())
        .count();
    if overloaded > 0 {
        ui.colored_label(egui::Color32::YELLOW, format!("Storage: {overloaded} satellites exceed {:.1} TB in this baseline. A storage-feasible allocation has not been established.", r.options.storage_bytes_per_satellite as f64 / 1e12));
    } else {
        ui.small(
            "Baseline retained crops fit the configured satellite storage over this forecast.",
        );
    }
    ui.small("Storage counts every retained crop with no deletion or downlink. Baseline allocation is not optimized, so overload alone is not proof that no feasible schedule exists.");
    ui.collapsing("Job details — failures first", |ui| {
        ui.small(format!("Showing up to 100 of {} jobs. Complete / due counts full AOI updates; partial counts windows with some coverage.", r.jobs.len()));
        egui::ScrollArea::both().max_height(300.0).show(ui, |ui| {
            egui::Grid::new("satisfiability-jobs").striped(true).show(ui, |ui| {
                for label in ["Job", "Mode", "Priority", "Complete / due", "Partial", "Missed"] { ui.strong(label); } ui.end_row();
                for &i in result.details.iter().take(100) {
                    let j = &r.jobs[i];
                    ui.label(&j.job_key); ui.label(j.fulfilment.label()); ui.label(j.priority.to_string());
                    ui.label(format!("{} / {}", j.complete_due_updates, j.due_updates));
                    ui.label(j.partial_due_updates.to_string()); ui.label(j.missed_due_updates.to_string()); ui.end_row();
                }
            });
        });
    });
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn text_position(shapes: &[egui::epaint::ClippedShape], needle: &str) -> Option<egui::Pos2> {
        fn find(shape: &egui::Shape, needle: &str) -> Option<egui::Pos2> {
            match shape {
                egui::Shape::Text(t) if t.galley.job.text == needle => {
                    Some(t.pos + t.galley.size() / 2.0)
                }
                egui::Shape::Vec(v) => v.iter().find_map(|s| find(s, needle)),
                _ => None,
            }
        }
        shapes.iter().find_map(|s| find(&s.shape, needle))
    }

    #[test]
    fn button_checks_loaded_jobs_in_background_and_setup_edits_mark_result_outdated() {
        let preview = Arc::new(Preview::from_json(br#"{"schema_version":1,"config":{"seed":42,"count":1},"aois":[{"id":"a","area_km2":100,"anchor_lon":-90,"anchor_lat":0,"geometry":{"type":"Polygon","coordinates":[[[-90.1,-0.1],[-89.9,-0.1],[-89.9,0.1],[-90.1,0.1],[-90.1,-0.1]]]}}]}"#).unwrap());
        let mut jobs = super::super::Panel::default();
        jobs.sync(&[preview]);
        jobs.assign().unwrap();
        jobs.jobs[0].cadence_hours = 0.01;
        jobs.jobs[0].acquisition.illumination = leo_capture::workload::Illumination::AnyTime;
        jobs.satisfiability.options.duration_s = 86.4;
        let mut planet = PlanetConfig::new("Earth".into());
        let mut constellation = crate::config::ConstellationConfig::new(0);
        constellation.num_planes = 1;
        constellation.sats_per_plane = 1;
        planet.constellations = vec![constellation];
        let epoch = DateTime::from_timestamp(946728000, 0).unwrap();
        let ctx = egui::Context::default();
        let frame = |jobs: &mut super::super::Panel, events| {
            ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1200.0, 1600.0),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| {
                    jobs.satisfiability_ui(
                        ui,
                        &planet,
                        CaptureProfile::default(),
                        20.0,
                        false,
                        0.0,
                        epoch,
                    );
                },
            )
        };
        for _ in 0..3 {
            let _ = frame(&mut jobs, vec![]);
        }
        let output = frame(&mut jobs, vec![]);
        let header = text_position(&output.shapes, "Satisfiability").unwrap();
        for pressed in [true, false] {
            let _ = frame(
                &mut jobs,
                vec![
                    egui::Event::PointerMoved(header),
                    egui::Event::PointerButton {
                        pos: header,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: Default::default(),
                    },
                ],
            );
        }
        for _ in 0..3 {
            let _ = frame(&mut jobs, vec![]);
        }
        let output = frame(&mut jobs, vec![]);
        let pos = text_position(&output.shapes, "Check current jobs").unwrap();
        for pressed in [true, false] {
            let _ = frame(
                &mut jobs,
                vec![
                    egui::Event::PointerMoved(pos),
                    egui::Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: Default::default(),
                    },
                ],
            );
        }
        assert!(jobs.satisfiability.worker.is_some());
        let deadline = Instant::now() + Duration::from_secs(10);
        while jobs.satisfiability.worker.is_some() && Instant::now() < deadline {
            let _ = frame(&mut jobs, vec![]);
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(jobs.satisfiability.worker.is_none());
        assert!(
            jobs.satisfiability.message.is_none(),
            "{:?}",
            jobs.satisfiability.message
        );
        let (_, result) = jobs.satisfiability.result.as_ref().unwrap();
        assert_eq!(result.report.jobs.len(), 1);
        assert_eq!(result.counts.due_updates, 2);
        let output = frame(&mut jobs, vec![]);
        assert!(text_position(
            &output.shapes,
            "OUTDATED — setup changed. Run the check again."
        )
        .is_none());
        jobs.revision += 1;
        let output = frame(&mut jobs, vec![]);
        assert!(text_position(
            &output.shapes,
            "OUTDATED — setup changed. Run the check again."
        )
        .is_some());
    }

    #[test]
    fn changed_jobs_cancel_pending_check_even_if_the_result_is_ready() {
        let mut panel = Panel::default();
        let key = Key {
            orbits: 1,
            revision: 1,
            capture: CaptureProfile::default(),
            options: panel.options.clone(),
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        tx.send(Err("Old result must not overwrite cancellation".into()))
            .unwrap();
        panel.worker = Some(Worker {
            key: key.clone(),
            cancel: cancel.clone(),
            progress: Arc::new(Mutex::new(Progress {
                stage: Stage::Forecast,
                fraction: 1.0,
            })),
            rx,
            started: Instant::now(),
        });
        panel.poll(&Key { revision: 2, ..key }, true, &egui::Context::default());
        assert!(cancel.load(Ordering::Relaxed));
        assert!(panel.worker.is_none() && panel.result.is_none());
        assert!(panel.message.unwrap().contains("Setup changed"));
    }
}
