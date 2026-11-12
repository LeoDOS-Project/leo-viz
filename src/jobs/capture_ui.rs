use crate::{aoi::preview::Preview, config::PlanetConfig};
use chrono::{DateTime, Utc};
use eframe::egui;
use leo_capture::{
    baseline::Options,
    workload::{Acquisition, AcquisitionGenerator, Fulfilment, Illumination, Job},
};
use leo_sim::acquisition::CaptureProfile;
use std::sync::{atomic::Ordering, Arc};

pub(super) fn generator_controls(ui: &mut egui::Ui, g: &mut AcquisitionGenerator) {
    ui.collapsing("Acquisition requirements",|ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label("Complete mosaic fraction");ui.add(egui::DragValue::new(&mut g.mosaic_fraction).range(0.0..=1.0).speed(0.01));
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Priority range (higher wins)");
            ui.add(egui::DragValue::new(&mut g.priority_min));ui.label("to");ui.add(egui::DragValue::new(&mut g.priority_max));
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Noon-centred normal σ");ui.add(egui::DragValue::new(&mut g.solar_stddev_hours).range(0.0..=12.0).speed(0.25).suffix(" h"));
        });
        solar_controls(ui,&mut g.solar_tolerance_hours,&mut g.min_solar_elevation_deg);
        ui.small("Cadence < 24 h: any-time capture, with no solar-time or elevation filter. Cadence ≥ 24 h: daylight plus a preferred local solar time. This assumes frequent jobs can use data acquired at night.");
        ui.small("Daylight preferences use a noon-centred truncated normal. Draws stay within 06:00–18:00 and within the configured tolerance of noon, so every generated window includes noon. Latitude, season and orbital coverage still determine feasibility. Clouds are not modeled.");
        ui.small("Cadence slots begin at run start. Mosaic pieces must all come from the same slot; independent jobs release crops as captured. Priorities are uniform integers in this range.");
    });
}
fn solar_controls(ui: &mut egui::Ui, tolerance: &mut f64, elevation: &mut f64) {
    ui.horizontal_wrapped(|ui| {
        ui.label("Local solar-time tolerance ±");
        ui.add(
            egui::DragValue::new(tolerance)
                .range(0.0..=12.0)
                .speed(0.25)
                .suffix(" h"),
        );
    });
    ui.horizontal_wrapped(|ui| {
        ui.label("Minimum solar elevation");
        ui.add(
            egui::DragValue::new(elevation)
                .range(0.0..=90.0)
                .speed(1.0)
                .suffix("°"),
        );
    });
}
pub(super) fn job_controls(ui: &mut egui::Ui, a: &mut Acquisition, cadence: f64) {
    ui.collapsing("Acquisition and priority",|ui| {
        egui::ComboBox::from_id_salt("job-fulfilment").selected_text(a.fulfilment.label()).show_ui(ui,|ui| {
            for value in [Fulfilment::Independent,Fulfilment::Mosaic] {ui.selectable_value(&mut a.fulfilment,value,value.label());}
        });
        ui.horizontal_wrapped(|ui| {ui.label("Priority (higher wins)");ui.add(egui::DragValue::new(&mut a.priority));});
        ui.horizontal_wrapped(|ui| {ui.label("First slot starts after");ui.add(egui::DragValue::new(&mut a.phase_hours).range(0.0..=(cadence-1e-6).max(0.0)).suffix(" h"));});
        egui::ComboBox::from_id_salt("job-illumination").selected_text(a.illumination.label()).show_ui(ui, |ui| {
            ui.selectable_value(&mut a.illumination, Illumination::AnyTime, Illumination::AnyTime.label());
            ui.add_enabled_ui(cadence >= 24.0, |ui| {
                ui.selectable_value(&mut a.illumination, Illumination::Daylight, Illumination::Daylight.label());
            });
        });
        if cadence < 24.0 { ui.small("Subdaily jobs must allow capture at any time of day."); }
        ui.add_enabled_ui(a.illumination == Illumination::Daylight, |ui| {
            ui.horizontal_wrapped(|ui| {ui.label("Preferred local solar time");ui.add(egui::DragValue::new(&mut a.solar_time_hours).range(0.0..=23.999).speed(0.25).suffix(" h"));});
            solar_controls(ui,&mut a.solar_tolerance_hours,&mut a.min_solar_elevation_deg);
        });
        ui.small("Cumulative captures within each cadence slot. Overlapping jobs can use the same physical capture if both accept its time. Overload policy: drop lowest-priority whole jobs first.");
    });
}

use super::capture_worker::{Command, Event, Initial, View, Worker};
use leo_capture::session::{Inputs, PreparationStage};

pub(super) struct Panel {
    options: Options,
    worker: Option<Worker>,
    pending: Option<Worker>,
    view: Option<View>,
    error: Option<String>,
    notice: Option<String>,
    path: String,
    speed: f64,
    forward_hours: f64,
    show_scene: bool,
    stopped: bool,
}
impl Default for Panel {
    fn default() -> Self {
        Self {
            options: Options::default(),
            worker: None,
            pending: None,
            view: None,
            error: None,
            notice: None,
            path: std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("capture-session.json.gz")
                .display()
                .to_string(),
            speed: 600.0,
            forward_hours: 24.0,
            show_scene: false,
            stopped: false,
        }
    }
}
impl Panel {
    fn poll(&mut self, ctx: &egui::Context) {
        if let Some(p) = &self.pending {
            if let Some(view) = p.take_view() {
                self.view = Some(view);
                self.worker = self.pending.take();
                self.stopped = false;
                self.notice=Some("Session ready. Checkpoints include all AOIs, jobs, orbits, retained footprints and partial coverage.".into());
            } else {
                let error = p.events.try_iter().find_map(|event| match event {
                    Event::Error(e) | Event::Stopped(e) => Some(e),
                    _ => None,
                });
                if let Some(e) = error {
                    self.error = Some(e);
                    self.pending = None;
                }
            }
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        if let Some(w) = &self.worker {
            if let Some(view) = w.take_view() {
                self.view = Some(view);
            }
            while let Ok(event) = w.events.try_recv() {
                match event {
                    Event::Error(e) | Event::Paused(e) => self.error = Some(e),
                    Event::Stopped(e) => {
                        self.stopped = true;
                        if let Some(v) = &mut self.view {
                            v.running = false;
                            v.fast_target = None;
                        }
                        self.error = Some(e);
                    }
                    Event::Saved {
                        path,
                        elapsed_s,
                        epoch_unix_s,
                    } => {
                        let utc = DateTime::<Utc>::from_timestamp(
                            (epoch_unix_s + elapsed_s).floor() as i64,
                            0,
                        )
                        .map(|t| t.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                        .unwrap_or_default();
                        self.notice=Some(format!("Last successful checkpoint: {path} · {utc} · {:.2} h into the run. Later progress is not saved automatically.",elapsed_s/3600.0));
                    }
                }
            }
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
    fn command(&mut self, command: Command) {
        if !self.stopped && matches!(command, Command::Run(true) | Command::FastForward(_)) {
            self.error = None;
        }
        if let Some(w) = &self.worker {
            if w.commands.send(command).is_err() {
                w.fast.store(false, Ordering::Relaxed);
                self.error = Some(
                    "Capture worker is no longer running. Load a checkpoint or start a new run."
                        .into(),
                );
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn start(&mut self, initial: Initial) {
        // Keep the old checkpoint/session available if loading or preparation fails.
        self.command(Command::Run(false));
        let worker = Worker::spawn(initial, self.speed);
        let _ = worker.commands.send(Command::Scene(self.show_scene));
        self.pending = Some(worker);
        self.error = None;
        self.notice = None;
        // Starting or loading a run must not replace the user’s current globe view.
    }
    fn controls(&mut self, ui: &mut egui::Ui) {
        let ready = self.view.is_some() && !self.stopped;
        let running = self.view.as_ref().is_some_and(|v| v.running);
        let finished = self.view.as_ref().is_some_and(|v| v.finished);
        let fast = self
            .worker
            .as_ref()
            .is_some_and(|w| w.fast.load(Ordering::Relaxed));
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(
                    ready && !finished,
                    egui::Button::new(if running || fast { "Pause" } else { "Run" }),
                )
                .clicked()
            {
                self.command(Command::Run(!(running || fast)));
            }
            ui.label("Speed");
            if ui
                .add(
                    egui::DragValue::new(&mut self.speed)
                        .range(1.0..=86400.0)
                        .speed(60.0)
                        .suffix("×"),
                )
                .changed()
            {
                self.command(Command::Speed(self.speed));
            }
            if ui.checkbox(&mut self.show_scene, "Open capture diagnostic map")
                .on_hover_text("Optional longitude/latitude map of the frozen capture session. Replaces the globe while open; closing it returns to the setup view.").changed() {
                self.command(Command::Scene(self.show_scene));
                if let Some(p) = &self.pending { let _ = p.commands.send(Command::Scene(self.show_scene)); }
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.forward_hours)
                    .range(0.001..=720.0)
                    .speed(1.0)
                    .suffix(" h"),
            );
            if ui
                .add_enabled(
                    ready && !finished && !fast,
                    egui::Button::new("Fast-forward"),
                )
                .clicked()
            {
                if let Some(w) = &self.worker {
                    w.fast.store(true, Ordering::Relaxed);
                }
                self.command(Command::FastForward(self.forward_hours * 3600.0));
            }
        });
        ui.small("The setup globe is a separate orbit preview; it does not display captured data or follow the capture session clock. Open the diagnostic map to inspect actual session positions and retained crops.");
        if self.stopped {
            ui.colored_label(egui::Color32::LIGHT_RED,"Worker stopped — these are the last completed results. Run and save are disabled until a session is loaded or restarted.");
        }
        ui.small("Fast-forward skips scene rendering, advances the same capture ticks, then restores the previous Run/Pause state. Target rounds up to the next track step, capped by the run horizon.");
        ui.add(egui::TextEdit::singleline(&mut self.path).desired_width(f32::INFINITY));
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(ready, egui::Button::new("Save checkpoint"))
                .clicked()
            {
                self.notice = Some("Saving at the next committed step…".into());
                self.command(Command::Save(self.path.clone()));
            }
            #[cfg(not(target_arch = "wasm32"))]
            if ui.button("Load checkpoint").clicked() {
                self.start(Initial::Load(self.path.clone()));
            }
            if ui.add_enabled(self.worker.is_some() || self.pending.is_some(), egui::Button::new("Unload session"))
                .on_hover_text("Release this session's memory. Stops capture and discards unsaved progress; save a checkpoint first.").clicked() {
                self.pending = None;
                self.worker = None;
                self.view = None;
                self.stopped = false;
                self.error = None;
                self.notice = Some("Session unloaded; unsaved progress was discarded. Load a checkpoint to resume.".into());
            }
        });
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::LIGHT_RED, e);
        }
        if let Some(n) = &self.notice {
            ui.small(n);
        }
    }
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        jobs: &[Job],
        layers: &[Arc<Preview>],
        compatible: bool,
        _revision: u64,
        planet: &PlanetConfig,
        capture: CaptureProfile,
        swath: f64,
        start: f64,
        epoch: DateTime<Utc>,
    ) {
        self.poll(ui.ctx());
        egui::CollapsingHeader::new("Capture simulation").id_salt("capture-baseline").default_open(true).show(ui,|ui| {
            ui.small("Unconstrained capture baseline. Storage is measured against capacity; no storage enforcement, processing, downlink, clouds or job dropping yet.");
            ui.small("Use this baseline to test the workload against the selected constellation. Consistent timing alone does not guarantee revisit or full AOI coverage; polar night can also prevent daylight jobs.");
            ui.collapsing("New run settings",|ui| {
                ui.horizontal_wrapped(|ui| {
                    let mut days=self.options.duration_s/86400.0;
                    ui.label("Run horizon");if ui.add(egui::DragValue::new(&mut days).range(0.001..=30.0).speed(0.1).suffix(" days")).changed(){self.options.duration_s=days*86400.0;}
                    let mut tb=self.options.storage_bytes_per_satellite as f64/1e12;
                    ui.label("Storage/satellite");if ui.add(egui::DragValue::new(&mut tb).range(0.001..=10000.0).suffix(" TB")).changed(){self.options.storage_bytes_per_satellite=(tb*1e12).round() as u64;}
                });
                ui.horizontal_wrapped(|ui| {ui.label("Track step");ui.add(egui::DragValue::new(&mut self.options.step_s).range(0.05..=30.0).speed(0.5).suffix(" s"));});
                ui.horizontal_wrapped(|ui| {ui.label("Polygon edge spacing");ui.add(egui::DragValue::new(&mut self.options.edge_km).range(0.05..=10.0).speed(0.1).suffix(" km"));});
                ui.small("Halve step and edge spacing to check convergence. Mosaic tolerance: 1 part per million of geometric area.");
                let no_deadline = jobs.iter().filter(|j| (j.acquisition.phase_hours+j.cadence_hours)*3600.0 > self.options.duration_s).count();
                if no_deadline > 0 {
                    ui.colored_label(egui::Color32::YELLOW, format!("{no_deadline} jobs have no deadline within this horizon. Their cadence feasibility will remain untested. Weekly jobs need at least 7 days; monthly jobs use a fixed 30-day cadence."));
                }
            });
            #[cfg(not(target_arch="wasm32"))]
            if ui.add_enabled(compatible&&!jobs.is_empty(),egui::Button::new("Start new capture run from setup")).clicked() {
                match crate::revisit::capture_satellites(planet) {
                    Err(e)=>self.error=Some(e),
                    Ok(satellites)=> {
                        self.start(Initial::New(Inputs {orbit_epoch_unix_s:epoch.timestamp_millis() as f64/1000.0,start_s:start,satellites,jobs:jobs.to_vec(),
                            regions:layers.iter().flat_map(|p|p.capture_regions()).collect(),capture,swath_km:swath,options:self.options.clone()}));
                    }
                }
            }
            #[cfg(target_arch="wasm32")]
            ui.small("Resumable capture runs require the native app.");
            ui.small("New runs freeze the selected constellation, current epoch/positions, AOIs, jobs and sensor. Setup edits affect the next run. Loading restores the saved session, paused. The optional diagnostic map uses its own saved constellation and clock.");
            #[cfg(not(target_arch="wasm32"))]
            self.resource_controls(ui);
            self.preparation_ui(ui);
            self.controls(ui);
            if let Some(v)=&self.view {meters(ui,v,self.stopped);}
        });
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn resource_controls(&self, ui: &mut egui::Ui) {
        use leo_capture::resources as r;
        ui.collapsing("Native resource limits", |ui| {
            let mut workers = r::worker_limit();
            ui.horizontal_wrapped(|ui| {
                ui.label(format!("Background workers ({} CPUs detected)", r::cpu_count()));
                if ui.add(egui::DragValue::new(&mut workers).range(1..=r::cpu_count())).changed() { r::set_worker_limit(workers); }
            });
            let mut duty = r::duty_percent();
            ui.horizontal_wrapped(|ui| {
                ui.label("Simulation CPU activity");
                if ui.add(egui::Slider::new(&mut duty, 10..=100).suffix("%")).changed() { r::set_duty_percent(duty); }
            });
            let mut gib = r::memory_limit() as f64 / r::GIB as f64;
            ui.horizontal_wrapped(|ui| {
                ui.label("Client memory pause threshold");
                if ui.add(egui::DragValue::new(&mut gib).range(0.5..=1024.0).speed(0.5).suffix(" GiB")).changed() { r::set_memory_limit((gib * r::GIB as f64) as u64); }
            });
            if let Some(bytes) = self.view.as_ref().and_then(|v| v.memory_bytes) {
                ui.label(format!("Client memory at last update: {:.2} GiB", bytes as f64 / r::GIB as f64));
            } else if r::process_memory().is_none() {
                ui.small("Process memory monitoring is unavailable on this platform; automatic memory pausing is unavailable.");
            }
            ui.small("Background worker capacity is shared by AOI generation, job assignment and capture preparation. Defaults leave two CPUs free and use at most eight workers. Changes apply as current work batches finish.");
            ui.small("Simulation CPU activity controls calculation/rest time, not total system CPU. The memory pause threshold defaults to 16 GiB. These settings apply for this app session.");
            ui.small("The memory threshold covers the whole client. Capture pauses between completed ticks and remains saveable; preparation cancels. An operation already in progress can overshoot. Retained captures are never deleted to meet the limit. Save, then Unload session to release its memory before loading another checkpoint.");
        });
    }
    fn preparation_ui(&mut self, ui: &mut egui::Ui) {
        if let Some(worker) = &self.pending {
            let p = worker.progress.lock().unwrap().clone();
            let message = preparation_message(&p);
            ui.strong(message);
            if p.stage == PreparationStage::BuildGeometry {
                ui.add(
                    egui::ProgressBar::new(p.completed as f32 / p.total.max(1) as f32)
                        .show_percentage(),
                );
                ui.small("Checking polygon boundaries, splitting them into geographic cells, and preparing coverage geometry. Detailed environmental AOIs can take longer than rectangles.");
            } else {
                ui.spinner();
            }
            ui.small(format!(
                "Preparation elapsed: {:.0} s. Percentages apply to the named stage.",
                worker.started.elapsed().as_secs_f64()
            ));
            if ui.button("Cancel preparation").clicked() {
                self.pending = None;
            }
        }
    }
    fn scene(&mut self, ctx: &egui::Context, fast_only: bool) -> bool {
        let fast = self
            .worker
            .as_ref()
            .is_some_and(|w| w.fast.load(Ordering::Relaxed));
        if fast_only && !fast {
            return false;
        }
        if !fast_only && (!self.show_scene || self.view.is_none()) {
            return false;
        }
        #[allow(deprecated)]
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(if fast {
                "Fast-forwarding capture simulation"
            } else {
                "Capture diagnostic map"
            });
            self.controls(ui);
            if let Some(v) = &self.view {
                ui.label(format!(
                    "Elapsed {:.2} h · last 24 h {} · unfulfilled updates {} · retained {}",
                    v.meters.elapsed_s / 3600.0,
                    super::bytes(v.meters.constellation_last_24h_bytes()),
                    v.meters.failed_updates,
                    super::bytes(v.report.retained_physical_bytes)
                ));
                if fast {
                    let target = v.fast_target.unwrap_or(v.report.options.duration_s);
                    ui.add(
                        egui::ProgressBar::new((v.meters.elapsed_s / target.max(1.0)) as f32)
                            .show_percentage(),
                    );
                    ui.label(format!(
                        "At {:.2} h; target {:.2} h. Scene rendering is suspended.",
                        v.meters.elapsed_s / 3600.0,
                        target / 3600.0
                    ));
                    meters(ui, v, self.stopped);
                } else {
                    map(ui, v);
                }
            } else {
                ui.spinner();
            }
        });
        true
    }
}
fn preparation_message(p: &leo_capture::session::PreparationProgress) -> String {
    match p.stage {
        PreparationStage::ReadCheckpoint => format!(
            "Reading and decompressing checkpoint… {:.1} MiB decoded",
            p.completed as f64 / (1024.0 * 1024.0)
        ),
        PreparationStage::ValidateInputs => {
            format!("Validating sensor, constellation and {} jobs…", p.total)
        }
        PreparationStage::BuildGeometry => {
            format!("Preparing AOI geometry: {} / {}", p.completed, p.total)
        }
        PreparationStage::BuildIndex => {
            format!("Building spatial lookup index for {} AOIs…", p.total)
        }
        PreparationStage::InitializeCoverage => {
            format!("Initializing coverage and counters for {} jobs…", p.total)
        }
        PreparationStage::ValidateCheckpoint => format!(
            "Validating restored partial coverage and {} retained crop records…",
            p.total
        ),
        PreparationStage::VerifyPositions => {
            format!("Checking {} saved satellite positions…", p.total)
        }
        PreparationStage::Ready => "Session ready".into(),
    }
}
fn meters(ui: &mut egui::Ui, v: &View, stopped: bool) {
    let m = &v.meters;
    let r = &v.report;
    ui.separator();
    if let Some(time) =
        DateTime::<Utc>::from_timestamp((r.epoch_unix_s + m.elapsed_s).floor() as i64, 0)
    {
        ui.label(format!(
            "Simulation time: {}",
            time.format("%Y-%m-%d %H:%M:%S UTC")
        ));
    }
    ui.label(format!(
        "{} jobs · {} satellites · {:.3}/{:.3} days {}",
        v.scene.anchors.len(),
        r.satellites.len(),
        m.elapsed_s / 86400.0,
        r.options.duration_s / 86400.0,
        if stopped {
            "· worker stopped (last completed results)"
        } else if v.finished {
            "· finished"
        } else if v.running {
            "· running"
        } else {
            "· paused"
        }
    ));
    ui.strong(format!(
        "Last 24 h: {} · {} retained crop records",
        super::bytes(m.constellation_last_24h_bytes()),
        m.captures_last_24h
    ));
    ui.label(format!(
        "Retained total: {} · reuse saved: {}",
        super::bytes(r.retained_physical_bytes),
        super::bytes(r.reuse_saved_bytes)
    ));
    ui.strong(format!(
        "Unfulfilled acquisition updates: {} / {} due · {} jobs affected",
        m.failed_updates, m.due_updates, m.affected_jobs
    ));
    ui.label(format!(
        "{} complete · {} partial · {} with no accepted coverage",
        m.complete_updates,
        m.partial_updates,
        m.failed_updates - m.partial_updates
    ));
    ui.label(format!("Baseline coverage by job: {} met every elapsed deadline · {} fell short · {} have no deadline yet",
        m.jobs_meeting_all_due_updates, m.affected_jobs, m.jobs_without_due_updates));
    ui.small("This tests full AOI coverage at the requested cadence, including for independent-crop jobs. A shortfall in this unconstrained baseline cannot be fixed by resource scheduling. Results apply to elapsed deadlines; they do not certify other dates or future deadlines.");
    ui.small("Unfulfilled means a cadence deadline passed without full accepted AOI coverage. This counter is separate from worker/software errors. Missing orbital coverage, daylight restrictions and the allowed local solar-time window can all contribute; individual deadline causes are not yet classified. Partial independent crops remain usable.");
    ui.small("All cadence slots count, including slots outside a job’s permitted time of day. Storage is not enforced in this baseline. The 24 h meter uses simulation time and credits crops at their end time.");
    ui.collapsing("Acquisition eligibility diagnostics",|ui| {
        let o=&r.opportunities;
        ui.label(format!("{} intersecting job/track segments checked · {} rejected by daylight or solar-time requirements",o.geometrically_intersecting_job_segments,o.timing_or_daylight_rejected_segments));
        ui.small("These are track-segment counts across the run, not a breakdown of failed deadlines. A job can have many segments, or none.");
    });
    ui.collapsing("Per-satellite capture and storage meters", |ui| {
        egui::ScrollArea::vertical()
            .id_salt("satellite-capture-meters")
            .max_height(300.0)
            .show_rows(ui, 28.0, r.satellites.len(), |ui, range| {
                for i in range {
                    let s = &r.satellites[i];
                    let fraction = s.retained_crop_bytes / s.storage_capacity_bytes as f64;
                    ui.add(
                        egui::ProgressBar::new(fraction.min(1.0) as f32).text(format!(
                            "{} · 24h {} · {}/{} ({:.1}%){}",
                            v.scene.satellites[i],
                            super::bytes(m.captured_last_24h_bytes[i]),
                            super::bytes(s.retained_crop_bytes),
                            super::bytes(s.storage_capacity_bytes as f64),
                            fraction * 100.0,
                            if fraction > 1.0 {
                                " · OVER CAPACITY"
                            } else {
                                ""
                            }
                        )),
                    );
                }
            });
    });
    ui.collapsing("Job results (first 100; checkpoint contains all)", |ui| {
        for j in &r.jobs {
            ui.label(format!(
                "{} · P{} · {} · {}/{} due complete · ready {:.2} km²",
                j.job_key,
                j.priority,
                j.fulfilment.label(),
                j.complete_due_updates,
                j.due_updates,
                j.ready_area_km2
            ));
        }
    });
}
fn map(ui: &mut egui::Ui, v: &View) {
    use egui_plot::{Line, Plot, Points};
    ui.label("This is a longitude/latitude map of the capture session, not the setup globe.");
    ui.small("Blue points mark requested AOIs (their boundaries appear when zoomed in, up to 50,000 vertices per frame). Amber points are satellite ground positions at the displayed simulation time. Green outlines are crops actually retained in the last 10 simulated minutes, limited to the latest 2,000 records and 100,000 vertices.");
    ui.small("Green outlines are recent acquisition activity, not cumulative job completion or a full sensor swath. They disappear from this map after 10 simulated minutes; the retained data stays in storage. Scroll to zoom; drag to pan.");
    Plot::new("capture-session-map")
        .include_x(-180.0)
        .include_x(180.0)
        .include_y(-90.0)
        .include_y(90.0)
        .x_axis_label("Longitude (°)")
        .y_axis_label("Latitude (°)")
        .data_aspect(1.0)
        .height(ui.available_height().max(180.0))
        .show(ui, |plot| {
            let bounds = plot.plot_bounds();
            plot.points(
                Points::new("AOIs", v.scene.anchors.as_slice())
                    .radius(1.5)
                    .color(egui::Color32::from_rgb(80, 130, 210)),
            );
            if bounds.width() < 30.0 {
                let mut vertices = 0;
                for ring in v.scene.visible_rings(bounds.min(), bounds.max()) {
                    if vertices + ring.len() > 50_000 {
                        break;
                    }
                    vertices += ring.len();
                    plot.line(
                        Line::new("", ring.clone())
                            .color(egui::Color32::from_rgb(80, 130, 210))
                            .width(1.0),
                    );
                }
            }
            for footprint in &v.footprints {
                for polygon in footprint {
                    for ring in polygon {
                        // Never connect across the dateline in this geographic map.
                        let mut segment = Vec::new();
                        for &p in ring {
                            if segment
                                .last()
                                .is_some_and(|q: &[f64; 2]| (q[0] - p[0]).abs() > 180.0)
                            {
                                plot.line(
                                    Line::new("", std::mem::take(&mut segment))
                                        .color(egui::Color32::LIGHT_GREEN),
                                );
                            }
                            segment.push(p);
                        }
                        plot.line(
                            Line::new("", segment)
                                .color(egui::Color32::LIGHT_GREEN)
                                .width(2.0),
                        );
                    }
                }
            }
            plot.points(
                Points::new("Satellites", v.positions.clone())
                    .radius(3.0)
                    .color(egui::Color32::GOLD),
            );
        });
}

impl super::Panel {
    pub(crate) fn poll_capture(&mut self, ctx: &egui::Context) {
        self.baseline.poll(ctx);
    }
    pub(crate) fn capture_fast_forward_ui(&mut self, ctx: &egui::Context) -> bool {
        self.baseline.scene(ctx, true)
    }
    pub(crate) fn capture_scene(&mut self, ctx: &egui::Context) -> bool {
        self.baseline.scene(ctx, false)
    }

    pub(crate) fn capture_ui(
        &mut self,
        ui: &mut egui::Ui,
        planet: &PlanetConfig,
        capture: CaptureProfile,
        swath: f64,
        start: f64,
        epoch: DateTime<Utc>,
    ) {
        self.baseline.ui(
            ui,
            &self.jobs,
            &self.layers,
            self.compatible,
            self.revision,
            planet,
            capture,
            swath,
            start,
            epoch,
        );
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};
    fn text(shape: &egui::Shape, out: &mut String) {
        match shape {
            egui::Shape::Text(t) => {
                out.push_str(&t.galley.job.text);
                out.push('\n');
            }
            egui::Shape::Vec(shapes) => {
                for s in shapes {
                    text(s, out);
                }
            }
            _ => {}
        }
    }
    #[test]
    #[allow(deprecated)] // Exercise the same top-level panel API as the application.
    fn capture_dashboard_shows_meters_and_bad_load_preserves_current_session() {
        let ctx = egui::Context::default();
        let mut p = Panel::default();
        p.speed = 1.0;
        p.start(Initial::New(super::super::capture_worker::tests::inputs()));
        let deadline = Instant::now() + Duration::from_secs(15);
        while p.view.is_none() && Instant::now() < deadline {
            p.poll(&ctx);
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(p.view.is_some(), "{:?}", p.error);
        assert!(
            !p.show_scene,
            "Starting capture must preserve the current view"
        );
        p.show_scene = true;
        p.command(Command::Run(false));
        let mut labels = String::new();
        for _ in 0..3 {
            let output = ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1600.0, 1000.0),
                    )),
                    ..Default::default()
                },
                |ctx| {
                    assert!(p.scene(ctx, false));
                },
            );
            labels.clear();
            for shape in output.shapes {
                text(&shape.shape, &mut labels);
            }
        }
        assert!(labels.contains("Capture diagnostic map"));
        assert!(labels.contains("last 24 h"));
        assert!(labels.contains("unfulfilled updates"));
        assert!(labels.contains("Save checkpoint"));
        p.stopped = true;
        let output = ctx.run(egui::RawInput::default(), |ctx| {
            p.scene(ctx, false);
        });
        labels.clear();
        for shape in output.shapes {
            text(&shape.shape, &mut labels);
        }
        assert!(labels.contains("Worker stopped"));
        assert!(labels.contains("Run and save are disabled"));
        p.stopped = false;

        let count = p.view.as_ref().unwrap().scene.anchors.len();
        p.start(Initial::Load(
            "/this-path-does-not-exist/capture.json.gz".into(),
        ));
        let deadline = Instant::now() + Duration::from_secs(10);
        while p.pending.is_some() && Instant::now() < deadline {
            p.poll(&ctx);
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(p.error.is_some());
        assert!(p.worker.is_some());
        assert_eq!(p.view.as_ref().unwrap().scene.anchors.len(), count);
        // Headless fast-forward path returns a progress screen instead of the map.
        p.worker
            .as_ref()
            .unwrap()
            .fast
            .store(true, Ordering::Relaxed);
        let output = ctx.run(egui::RawInput::default(), |ctx| {
            assert!(p.scene(ctx, true));
        });
        labels.clear();
        for shape in output.shapes {
            text(&shape.shape, &mut labels);
        }
        assert!(labels.contains("Fast-forwarding capture simulation"));
        assert!(!labels.contains("Scroll to zoom"));
    }
}
