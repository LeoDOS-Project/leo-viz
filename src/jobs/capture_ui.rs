use crate::{aoi::preview::Preview, config::PlanetConfig};
use chrono::{DateTime, Utc};
use eframe::egui;
use leo_capture::{
    baseline::Options,
    workload::{Acquisition, AcquisitionGenerator, Fulfilment, Job},
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
        ui.small("Preferred local solar times follow a truncated normal centred at 12:00, bounded by 00:00–24:00. Independent of geographic source and pipeline. Clouds are not modeled.");
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
        ui.horizontal_wrapped(|ui| {ui.label("Preferred local solar time");ui.add(egui::DragValue::new(&mut a.solar_time_hours).range(0.0..=23.999).speed(0.25).suffix(" h"));});
        solar_controls(ui,&mut a.solar_tolerance_hours,&mut a.min_solar_elevation_deg);
        ui.small("Cumulative captures within each cadence slot. Overlapping jobs can use the same physical capture if both accept its time. Overload policy: drop lowest-priority whole jobs first.");
    });
}

use super::capture_worker::{Command, Event, Initial, View, Worker};
use leo_capture::session::{Inputs, Orbit, Satellite};

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
        }
    }
}
impl Panel {
    fn poll(&mut self, ctx: &egui::Context) {
        if let Some(p) = &self.pending {
            if let Ok(view) = p.views.try_recv() {
                self.view = Some(view);
                self.worker = self.pending.take();
                self.notice=Some("Session ready. Checkpoints include all AOIs, jobs, orbits, retained footprints and partial coverage.".into());
            } else {
                let error = p.events.try_iter().find_map(|event| match event {
                    Event::Error(e) => Some(e),
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
            while let Ok(view) = w.views.try_recv() {
                self.view = Some(view);
            }
            while let Ok(event) = w.events.try_recv() {
                match event {
                    Event::Error(e) => self.error = Some(e),
                    Event::Saved(path) => self.notice = Some(format!("Checkpoint saved: {path}")),
                }
            }
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
    fn command(&mut self, command: Command) {
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
        self.pending = Some(Worker::spawn(initial, self.speed));
        self.error = None;
        self.notice = None;
        self.show_scene = true;
    }
    fn controls(&mut self, ui: &mut egui::Ui) {
        let ready = self.view.is_some();
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
            ui.checkbox(&mut self.show_scene, "Show capture map");
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
            });
            #[cfg(not(target_arch="wasm32"))]
            if ui.add_enabled(compatible&&!jobs.is_empty(),egui::Button::new("Start new capture run from setup")).clicked() {
                match crate::revisit::orbit_groups(planet) {
                    Err(e)=>self.error=Some(e),
                    Ok(groups)=> {
                        let satellites=groups.into_iter().flat_map(|(name,orbits)|orbits.into_iter().enumerate().map(move |(i,o)| {
                            let (label,orbit)=match o {
                                crate::revisit::Orbit::Analytical(elements)=>(format!("Satellite {}",i+1),Orbit::Analytical{elements}),
                                crate::revisit::Orbit::Tle(t)=>(format!("{} ({})",t.name,t.norad_id),Orbit::Sgp4{constants:t.constants,epoch_minutes:t.epoch_minutes}),
                            };
                            Satellite {name:label,constellation:name.clone(),orbit}
                        })).collect();
                        self.start(Initial::New(Inputs {orbit_epoch_unix_s:epoch.timestamp_millis() as f64/1000.0,start_s:start,satellites,jobs:jobs.to_vec(),
                            regions:layers.iter().flat_map(|p|p.capture_regions()).collect(),capture,swath_km:swath,options:self.options.clone()}));
                    }
                }
            }
            #[cfg(target_arch="wasm32")]
            ui.small("Resumable capture runs require the native app.");
            ui.small("New runs freeze the selected constellation, current epoch/positions, AOIs, jobs and sensor. Setup edits affect the next run. Loading restores the saved session, paused, in its capture map.");
            if self.pending.is_some() {
                ui.label("Preparing capture session…");
                ui.add(egui::ProgressBar::new(self.pending.as_ref().unwrap().progress.load(Ordering::Relaxed) as f32/1000.0));
                if ui.button("Cancel preparation").clicked(){self.pending=None;}
            }
            self.controls(ui);
            if let Some(v)=&self.view {meters(ui,v);}
        });
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
                "Capture simulation"
            });
            self.controls(ui);
            if let Some(v) = &self.view {
                ui.label(format!(
                    "Elapsed {:.2} h · last 24 h {} · failed updates {} · retained {}",
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
                    meters(ui, v);
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
fn meters(ui: &mut egui::Ui, v: &View) {
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
        if v.finished {
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
        "Failed acquisition updates: {} / {} due · {} jobs affected",
        m.failed_updates, m.due_updates, m.affected_jobs
    ));
    ui.label(format!(
        "{} complete · {} partial · {} with no coverage",
        m.complete_updates,
        m.partial_updates,
        m.failed_updates - m.partial_updates
    ));
    ui.small("A failure is a cadence deadline without full AOI coverage; a recurring job can fail multiple updates. Partial independent crops remain usable. No processing failures are modeled. Last 24 h uses simulation time and credits each crop at its end time.");
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
    ui.small("Blue: AOIs · amber: satellite ground positions · green: retained crop outlines from the last 10 simulated minutes (latest 2,000). Scroll to zoom; drag to pan. AOI outlines appear when zoomed in.");
    Plot::new("capture-session-map")
        .include_x(-180.0)
        .include_x(180.0)
        .include_y(-90.0)
        .include_y(90.0)
        .data_aspect(1.0)
        .height(ui.available_height().max(180.0))
        .show(ui, |plot| {
            let bounds = plot.plot_bounds();
            plot.points(
                Points::new("AOIs", v.scene.anchors.clone())
                    .radius(1.5)
                    .color(egui::Color32::from_rgb(80, 130, 210)),
            );
            if bounds.width() < 30.0 {
                for region in &v.scene.regions {
                    for polygon in &region.polygons {
                        for ring in polygon {
                            if ring.iter().any(|p| {
                                p[0] >= bounds.min()[0]
                                    && p[0] <= bounds.max()[0]
                                    && p[1] >= bounds.min()[1]
                                    && p[1] <= bounds.max()[1]
                            }) {
                                plot.line(
                                    Line::new("", ring.clone())
                                        .color(egui::Color32::from_rgb(80, 130, 210))
                                        .width(1.0),
                                );
                            }
                        }
                    }
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
        assert!(labels.contains("Capture simulation"));
        assert!(labels.contains("last 24 h"));
        assert!(labels.contains("failed updates"));
        assert!(labels.contains("Save checkpoint"));
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
