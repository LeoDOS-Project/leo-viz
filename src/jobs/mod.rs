//! Job configuration and steady averages, separate from infrastructure scheduling.
mod capture_ui;
mod capture_worker;
pub(crate) mod model;
mod storage;

use crate::aoi::preview::{Hit, Preview};
use eframe::egui;
use leo_sim::acquisition::CaptureProfile;
use model::{Generator, Job, Pipeline, Totals};
use std::{collections::HashMap, sync::Arc};

pub(crate) struct Panel {
    config: Generator,
    applied: Option<Generator>,
    jobs: Vec<Job>,
    lookup: HashMap<String, usize>,
    layers: Vec<Arc<Preview>>,
    compatible: bool,
    hits: Vec<Hit>,
    selected: Option<Hit>,
    edit: Option<Job>,
    reveal: bool,
    capture: Option<CaptureProfile>,
    report: Option<Result<(Totals, [Totals; 3]), String>>,
    path: String,
    message: Option<String>,
    revision: u64,
    baseline: capture_ui::Panel,
}
impl Default for Panel {
    fn default() -> Self {
        Self {
            revision: 0,
            baseline: capture_ui::Panel::default(),
            config: Generator::default(),
            applied: None,
            jobs: Vec::new(),
            lookup: HashMap::new(),
            layers: Vec::new(),
            compatible: true,
            hits: Vec::new(),
            selected: None,
            edit: None,
            reveal: false,
            capture: None,
            report: None,
            path: std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("assets/jobs/default-workload.json.gz")
                .to_string_lossy()
                .into_owned(),
            message: None,
        }
    }
}
/// Decimal byte formatting, shared with sensor acquisition estimates.
pub(crate) fn bytes(mut value: f64) -> String {
    let units = ["B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"];
    let mut i = 0;
    while value >= 1000.0 && i < units.len() - 1 {
        value /= 1000.0;
        i += 1;
    }
    format!("{value:.2} {}", units[i])
}
fn duration(hours: f64) -> String {
    if hours >= 24.0 {
        format!("{:.2} days", hours / 24.0)
    } else {
        format!("{hours:.2} hours")
    }
}
fn layers_match(jobs: &[Job], layers: &[Arc<Preview>]) -> bool {
    let regions: HashMap<_, _> = layers
        .iter()
        .flat_map(|p| &p.aois)
        .map(|a| (a.key(), a))
        .collect();
    regions.len() == jobs.len()
        && jobs
            .iter()
            .all(|j| regions.get(&j.aoi.key()) == Some(&&j.aoi))
}
impl Panel {
    pub fn selected(&self) -> Option<(&Arc<Preview>, usize)> {
        self.selected.as_ref().map(|hit| (&hit.preview, hit.index))
    }
    pub fn pick(&mut self, mut hits: Vec<Hit>) {
        if hits.is_empty() {
            return;
        }
        hits.sort_by(|a, b| {
            a.preview.aois[a.index]
                .area_km2
                .total_cmp(&b.preview.aois[b.index].area_km2)
        });
        self.selected = hits.first().cloned();
        self.hits = hits;
        self.reset_edit();
        self.reveal = true;
    }
    fn reset_edit(&mut self) {
        self.edit = self
            .selected
            .as_ref()
            .and_then(|hit| self.lookup.get(&hit.preview.aois[hit.index].key()))
            .and_then(|i| self.jobs.get(*i))
            .cloned();
    }
    fn sync(&mut self, layers: &[Arc<Preview>]) {
        if layers.len() == self.layers.len()
            && layers
                .iter()
                .zip(&self.layers)
                .all(|(a, b)| Arc::ptr_eq(a, b))
        {
            return;
        }
        self.revision += 1;
        self.layers = layers.to_vec();
        self.compatible = self.jobs.is_empty() || layers_match(&self.jobs, layers);
        self.hits
            .retain(|hit| layers.iter().any(|p| Arc::ptr_eq(p, &hit.preview)));
        if self
            .selected
            .as_ref()
            .is_some_and(|hit| !layers.iter().any(|p| Arc::ptr_eq(p, &hit.preview)))
        {
            self.selected = None;
            self.edit = None;
        }
    }
    fn assign(&mut self) -> Result<(), String> {
        let jobs = self
            .config
            .generate(self.layers.iter().flat_map(|p| p.aois.iter().cloned()))?;
        if jobs.is_empty() {
            return Err("Load or generate AOIs first.".into());
        }
        self.install(jobs, self.config.clone());
        Ok(())
    }
    fn install(&mut self, jobs: Vec<Job>, generator: Generator) {
        self.revision += 1;
        self.jobs = jobs;
        self.lookup = self
            .jobs
            .iter()
            .enumerate()
            .map(|(i, j)| (j.aoi.key(), i))
            .collect();
        self.applied = Some(generator);
        self.report = None;
        self.compatible = true;
        self.reset_edit();
    }
    fn report(&mut self, capture: CaptureProfile) -> Result<(Totals, [Totals; 3]), String> {
        if self.capture != Some(capture) || self.report.is_none() {
            self.report = Some(model::totals(&self.jobs, capture));
            self.capture = Some(capture);
        }
        self.report.clone().unwrap()
    }
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        layers: Vec<Arc<Preview>>,
        capture: &mut CaptureProfile,
    ) {
        self.sync(&layers);
        let reveal = std::mem::take(&mut self.reveal);
        egui::CollapsingHeader::new("Jobs")
            .id_salt("sidebar-jobs").default_open(true).open(reveal.then_some(true)).show(ui, |ui| {
                let intro = ui.label("Steady averages · all jobs fulfilled · full sliding windows");
                if reveal { intro.scroll_to_me(Some(egui::Align::TOP)); }
                ui.small("One job per loaded AOI. Overlaps count separately; overlay visibility does not change demand.");
                ui.label("Click an AOI on the Earth globe to inspect its job.");
                if !self.compatible {
                    ui.colored_label(egui::Color32::YELLOW, "AOIs changed. Assign jobs again, or load a workload matching these AOIs.");
                }
                self.inspector(ui, *capture);
                if !self.jobs.is_empty() && self.compatible {
                    ui.separator();
                    self.summary(ui, *capture);
                }
                let count: usize = self.layers.iter().map(|p| p.count).sum();
                ui.separator();
                egui::CollapsingHeader::new(format!("Job generator · {count} loaded AOIs"))
                    .id_salt("job-generator").default_open(self.jobs.is_empty()).show(ui, |ui| {
                        self.generator_ui(ui);
                        let validation = self.config.validate();
                        if let Err(error) = &validation { ui.colored_label(egui::Color32::LIGHT_RED, error); }
                        if ui.add_enabled(count > 0 && validation.is_ok(), egui::Button::new(
                            if self.jobs.is_empty() { "Assign jobs to AOIs" } else { "Reassign all jobs" })).clicked() {
                            self.message = match self.assign() {
                                Ok(()) => Some(format!("Assigned {} jobs with seed {}.", self.jobs.len(), self.config.seed)),
                                Err(error) => Some(error),
                            };
                        }
                        if self.applied.as_ref().is_some_and(|applied| applied != &self.config) {
                            ui.small("Generator changes apply when you reassign. Existing jobs and totals are unchanged.");
                        }
                    });
                #[cfg(not(target_arch = "wasm32"))]
                self.storage_ui(ui, capture);
                if let Some(message) = &self.message { ui.small(message); }
                ui.collapsing("Calculation assumptions", |ui| {
                    ui.label("Input is new useful imagery over each AOI, using the Data pane's channels, resolution, packed encoding and compression. It excludes unused swath pixels and repeated reads of history.");
                    ui.label("Each location is updated at its job's reference cadence. Area-linear byte profiles let us sum a large AOI without pretending it is captured instantaneously or choosing observation-unit sizes yet.");
                    ui.label("Map: no history. Map → Window: k mapper outputs. Window: k images. Histories are fully populated and reused; there is one copy per job. Temporary working memory and assembly buffers are excluded.");
                    ui.label("One result per update; output density is configured in bytes/km²/update. Image compression does not change record output sizes. Outputs are the data offered for delivery to Earth, without transport overhead.");
                    ui.label("No failures, daylight/cloud gating, acquisition feasibility, sharing, placement, ISLs, or downlink scheduling. Totals are workload requirements, not a claim that a constellation can serve them.");
                    ui.monospace("input/day = image bytes × 24 / cadence hours\nstate     = samples × bytes per history entry\noutput/day = result bytes × 24 / cadence hours");
                    ui.small("Decimal bytes. One day = 24 hours. Window span = (samples − 1) × cadence.");
                });
            });
    }
    fn generator_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Seed");
            ui.add(egui::DragValue::new(&mut self.config.seed));
        });
        ui.small("Synthetic parameter choices, independent of geographic source. Mixture weights are relative.");
        egui::Grid::new("job-mix").num_columns(2).show(ui, |ui| {
            for (pipeline, weight) in Pipeline::ALL.iter().zip(&mut self.config.pipeline_weights) {
                ui.label(pipeline.label());
                ui.add(egui::DragValue::new(weight).range(0.0..=1000.0).speed(0.1));
                ui.end_row();
            }
        });
        capture_ui::generator_controls(ui, &mut self.config.acquisition);
        ui.label("Joint cadence and window profiles");
        for (i, profile) in self.config.temporal.iter_mut().enumerate() {
            ui.push_id(i, |ui| {
                ui.strong(&profile.name);
                ui.horizontal_wrapped(|ui| {
                    ui.label("Weight");
                    ui.add(
                        egui::DragValue::new(&mut profile.weight)
                            .range(0.0..=1000.0)
                            .speed(0.1),
                    );
                    ui.label("Cadence");
                    ui.add(
                        egui::DragValue::new(&mut profile.cadence_hours)
                            .range(0.01..=87600.0)
                            .speed(0.25)
                            .suffix(" h"),
                    );
                    ui.label("Samples");
                    ui.add(egui::DragValue::new(&mut profile.samples).range(2..=100_000));
                });
                ui.small(format!(
                    "Full-window span: {}",
                    duration(f64::from(profile.samples.saturating_sub(1)) * profile.cadence_hours)
                ));
            });
        }
        ui.small("Map jobs use the selected cadence with one sample and no history.");
        ui.collapsing("Global cadence and history bounds", |ui| {
            let b = &mut self.config.bounds;
            for (label, lo, hi) in [
                (
                    "Cadence",
                    &mut b.cadence_min_hours,
                    &mut b.cadence_max_hours,
                ),
                ("Window span", &mut b.span_min_hours, &mut b.span_max_hours),
            ] {
                ui.horizontal_wrapped(|ui| {
                    ui.label(label);
                    ui.add(
                        egui::DragValue::new(lo)
                            .range(0.01..=876000.0)
                            .suffix(" h min"),
                    );
                    ui.add(
                        egui::DragValue::new(hi)
                            .range(0.01..=876000.0)
                            .suffix(" h max"),
                    );
                });
            }
        });
        ui.collapsing("Output size distributions", |ui| {
            ui.small("Log-uniform byte densities. Values below are medians; these are tunable workload assumptions.");
            output_control(ui, "Mapper output median", &mut self.config.map_bytes_per_km2);
            output_control(ui, "Window output median", &mut self.config.window_bytes_per_km2);
            ui.horizontal_wrapped(|ui| {
                ui.label("Maximum/minimum density");
                ui.add(egui::DragValue::new(&mut self.config.output_size_range).range(1.0..=1e6).suffix("×"));
            });
            ui.small("Range 100× draws between median/10 and median×10. Result size scales with AOI area.");
        });
    }
    fn summary(&mut self, ui: &mut egui::Ui, capture: CaptureProfile) {
        let (total, groups) = match self.report(capture) {
            Ok(report) => report,
            Err(error) => {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
                return;
            }
        };
        ui.strong(format!("Workload total · {} jobs", total.count));
        egui::Grid::new("workload-totals")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                metric(ui, "Fresh input / day", total.input_per_day);
                metric(ui, "Retained window history", total.history_bytes);
                metric(ui, "Final output / day", total.output_per_day);
            });
        ui.small(format!(
            "Average input {}/s · output {}/s",
            bytes(total.input_per_day / 86400.0),
            bytes(total.output_per_day / 86400.0)
        ));
        ui.collapsing("By pipeline", |ui| {
            for (pipeline, total) in Pipeline::ALL.iter().zip(groups) {
                ui.strong(format!("{} · {} jobs", pipeline.label(), total.count));
                ui.label(format!(
                    "Input {}/day · history {}",
                    bytes(total.input_per_day),
                    bytes(total.history_bytes)
                ));
                ui.label(format!("Output {}/day", bytes(total.output_per_day)));
            }
        });
        ui.small(format!(
            "{} channels · {} m/px · {} bits/sample · {:.2}:1 compression (shared Data settings)",
            capture.channels,
            capture.resolution_m,
            capture.bits_per_sample,
            capture.compression_ratio
        ));
    }
    fn inspector(&mut self, ui: &mut egui::Ui, capture: CaptureProfile) {
        let Some(selected) = self.selected.clone() else {
            return;
        };
        ui.separator();
        if self.hits.len() > 1 {
            let mut index = self
                .hits
                .iter()
                .position(|h| {
                    h.index == selected.index && Arc::ptr_eq(&h.preview, &selected.preview)
                })
                .unwrap_or(0);
            let before = index;
            egui::ComboBox::from_id_salt("job-hit-choice")
                .width(ui.available_width().min(380.0))
                .selected_text(format!("{} AOIs here · choose one", self.hits.len()))
                .show_ui(ui, |ui| {
                    for (i, hit) in self.hits.iter().enumerate() {
                        let a = &hit.preview.aois[hit.index];
                        ui.selectable_value(
                            &mut index,
                            i,
                            format!("{} · {:.1} km²", a.key(), a.area_km2),
                        );
                    }
                });
            if before != index {
                self.selected = Some(self.hits[index].clone());
                self.reset_edit();
            }
        }
        let selected = self.selected.as_ref().unwrap();
        let aoi = &selected.preview.aois[selected.index];
        ui.strong(format!("Selected AOI · {}", aoi.key()));
        ui.label(format!(
            "{:.2} km² · anchor {:.4}°, {:.4}°",
            aoi.area_km2, aoi.latitude, aoi.longitude
        ));
        if !self.compatible || self.edit.is_none() {
            ui.label("Assign jobs to inspect this AOI's pipeline and data demand.");
            return;
        }
        let job = self.edit.as_mut().unwrap();
        let old = job.clone();
        ui.horizontal_wrapped(|ui| {
            ui.label("Pipeline");
            egui::ComboBox::from_id_salt("selected-job-pipeline")
                .selected_text(job.pipeline.label())
                .show_ui(ui, |ui| {
                    for p in Pipeline::ALL {
                        ui.selectable_value(&mut job.pipeline, p, p.label());
                    }
                });
        });
        let bounds = &self.applied.as_ref().unwrap().bounds;
        if job.pipeline != old.pipeline {
            job.samples = if job.pipeline.has_window() && old.pipeline.has_window() {
                old.samples
            } else if job.pipeline.has_window() {
                ((bounds.span_min_hours / job.cadence_hours).ceil() as u32 + 1).max(2)
            } else {
                1
            };
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("Capture cadence");
            ui.add(
                egui::DragValue::new(&mut job.cadence_hours)
                    .range(bounds.cadence_min_hours..=bounds.cadence_max_hours)
                    .speed(0.25)
                    .suffix(" h"),
            );
            if job.pipeline.has_window() {
                ui.label("Window samples");
                ui.add(egui::DragValue::new(&mut job.samples).range(2..=100_000));
            }
        });
        if job.pipeline.has_window() {
            ui.label(format!(
                "{} samples · {} history span · slide 1",
                job.samples,
                duration(job.span_hours())
            ));
        }
        ui.collapsing("Operator byte profiles", |ui| {
            if job.pipeline.has_map() {
                output_control(ui, "Mapper output", &mut job.map_bytes_per_km2);
            }
            if job.pipeline.has_window() {
                output_control(ui, "Window output", &mut job.window_bytes_per_km2);
            }
        });
        capture_ui::job_controls(ui, &mut job.acquisition, job.cadence_hours);
        let result = job.estimate(capture).and_then(|e| if bounds.contains(job.cadence_hours, job.samples) { Ok(e) }
            else { Err("This cadence/window combination is outside the applied global bounds. Adjust it to apply the edit.".into()) });
        let estimate = match result {
            Ok(estimate) => estimate,
            Err(error) => {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
                return;
            }
        };
        if *job != old {
            let index = self.lookup[&job.aoi.key()];
            self.revision += 1;
            self.jobs[index] = job.clone();
            self.report = None;
        }
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.label(format!(
                "Observation · {} / AOI-equivalent update",
                bytes(estimate.input_per_update)
            ));
            if job.pipeline.has_map() {
                ui.label(format!(
                    "↓ Mapper · {} output",
                    bytes(job.aoi.area_km2 * job.map_bytes_per_km2)
                ));
            }
            if job.pipeline.has_window() {
                ui.label(format!(
                    "↓ Window · {} × {} retained",
                    job.samples,
                    bytes(estimate.window_entry_bytes)
                ));
            }
            ui.label(format!(
                "↓ Final output · {} / update",
                bytes(estimate.output_per_update)
            ));
        });
        egui::Grid::new("selected-job-metrics")
            .num_columns(2)
            .show(ui, |ui| {
                metric(ui, "Fresh input / day", estimate.input_per_day);
                metric(ui, "Retained window history", estimate.history_bytes);
                metric(ui, "Final output / day", estimate.output_per_day);
            });
        ui.small(
            "Cadence applies throughout the AOI; its parts need not be acquired simultaneously.",
        );
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn storage_ui(&mut self, ui: &mut egui::Ui, capture: &mut CaptureProfile) {
        ui.collapsing("Save / load workload", |ui| {
            ui.label("Compressed job file (.json.gz)");
            ui.add(egui::TextEdit::singleline(&mut self.path).desired_width(f32::INFINITY));
            ui.horizontal(|ui| {
                if ui.add_enabled(!self.jobs.is_empty() && self.compatible, egui::Button::new("Save workload")).clicked() {
                    let document = storage::Document { schema_version: 2, generator: self.applied.clone().unwrap(), capture: *capture, jobs: self.jobs.clone() };
                    self.message = Some(match document.save(std::path::Path::new(&self.path)) {
                        Ok(()) => format!("Saved {} job configurations.", self.jobs.len()), Err(error) => format!("Could not save: {error}"),
                    });
                }
                if ui.button("Load workload").clicked() {
                    let result = std::fs::read(&self.path).map_err(|e| e.to_string()).and_then(|bytes| storage::Document::decode(&bytes));
                    self.message = Some(match result {
                        Ok(document) if layers_match(&document.jobs, &self.layers) => {
                            *capture = document.capture;
                            self.config = document.generator.clone();
                            self.install(document.jobs, document.generator);
                            format!("Loaded {} jobs and their shared image settings.", self.jobs.len())
                        }
                        Ok(_) => "AOIs do not match this workload. Load the same AOI datasets first.".into(),
                        Err(error) => format!("Could not load: {error}"),
                    });
                }
            });
            ui.small("Saves assignments, seed, parameters and shared image settings. References existing AOI geometry; loading verifies IDs and shapes. Load restores Data settings.");
        });
    }
}
fn output_control(ui: &mut egui::Ui, label: &str, value: &mut f64) {
    ui.horizontal_wrapped(|ui| {
        ui.label(label);
        ui.add(
            egui::DragValue::new(value)
                .range(0.0..=1e12)
                .speed(100.0)
                .suffix(" B/km²/update"),
        );
    });
}
fn metric(ui: &mut egui::Ui, label: &str, value: f64) {
    ui.label(label);
    ui.strong(bytes(value));
    ui.end_row();
}

#[cfg(test)]
mod tests {
    use super::*;
    fn preview() -> Arc<Preview> {
        Arc::new(Preview::from_json(br#"{"schema_version":1,"config":{"seed":42,"count":1},"aois":[{"id":"a","area_km2":100,"anchor_lon":-90,"anchor_lat":0,"geometry":{"type":"Polygon","coordinates":[[[-91,-1],[-89,-1],[-89,1],[-91,1],[-91,-1]]]}}]}"#).unwrap())
    }
    #[test]
    fn assign_button_generates_jobs() {
        let p = preview();
        let mut panel = Panel::default();
        let mut capture = CaptureProfile::default();
        let ctx = egui::Context::default();
        let frame = |panel: &mut Panel, capture: &mut CaptureProfile, events| {
            ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(550., 1400.),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| panel.ui(ui, vec![p.clone()], capture),
            )
        };
        let _ = frame(&mut panel, &mut capture, vec![]);
        let out = frame(&mut panel, &mut capture, vec![]);
        fn text(shape: &egui::Shape) -> Option<egui::Pos2> {
            match shape {
                egui::Shape::Text(t) if t.galley.job.text == "Assign jobs to AOIs" => {
                    Some(t.pos + t.galley.size() / 2.)
                }
                egui::Shape::Vec(v) => v.iter().find_map(text),
                _ => None,
            }
        }
        let pos = out
            .shapes
            .iter()
            .find_map(|s| text(&s.shape))
            .expect("Assignment button is visible");
        for pressed in [true, false] {
            let _ = frame(
                &mut panel,
                &mut capture,
                vec![
                    egui::Event::PointerMoved(pos),
                    egui::Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
            );
        }
        assert_eq!(panel.jobs.len(), 1);
        assert_eq!(panel.report(capture).unwrap().0.count, 1);
    }

    #[test]
    #[ignore = "full 50,000-AOI assignment, persistence and cached-summary check"]
    fn workload_50k_roundtrip_and_cached_summary() {
        use std::io::Read;
        let mut json = Vec::new();
        flate2::read::GzDecoder::new(
            &include_bytes!("../../assets/aois/default-jobs-oriented.json.gz")[..],
        )
        .read_to_end(&mut json)
        .unwrap();
        let preview = Arc::new(Preview::from_json(&json).unwrap());
        let mut panel = Panel::default();
        panel.sync(&[preview.clone()]);
        let start = std::time::Instant::now();
        panel.assign().unwrap();
        let assign_time = start.elapsed();
        let capture = CaptureProfile::default();
        let start = std::time::Instant::now();
        let report = panel.report(capture).unwrap();
        let summary_time = start.elapsed();
        assert_eq!(report.0.count, 50_000);
        assert_eq!(report.1.iter().map(|g| g.count).sum::<usize>(), 50_000);
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            assert_eq!(panel.report(capture).unwrap(), report);
        }
        let cached_time = start.elapsed();
        let document = storage::Document {
            schema_version: 2,
            generator: panel.applied.clone().unwrap(),
            capture,
            jobs: panel.jobs.clone(),
        };
        let encoded = document.encode().unwrap();
        let loaded = storage::Document::decode(&encoded).unwrap();
        assert!(layers_match(&loaded.jobs, &[preview]));
        assert_eq!(model::totals(&loaded.jobs, loaded.capture).unwrap(), report);
        eprintln!("50k jobs: assignment {assign_time:?}; summary {summary_time:?}; 1000 cached summaries {cached_time:?}; compressed {} bytes", encoded.len());
        eprintln!(
            "Input {}/day; retained history {}; output {}/day",
            bytes(report.0.input_per_day),
            bytes(report.0.history_bytes),
            bytes(report.0.output_per_day)
        );
    }

    #[test]
    fn totals_invalidate_on_encoding_changes_and_geometry_replacement() {
        let mut panel = Panel::default();
        panel.sync(&[preview()]);
        panel.assign().unwrap();
        let capture = CaptureProfile::default();
        let before = panel.report(capture).unwrap().0;
        let after = panel
            .report(CaptureProfile {
                channels: 8,
                ..capture
            })
            .unwrap()
            .0;
        assert_eq!(after.input_per_day, 2.0 * before.input_per_day);
        panel.sync(&[preview()]);
        assert!(panel.compatible); // Same geometry reloaded.
        panel.sync(&[]);
        assert!(!panel.compatible); // Stale totals must be hidden.
    }
    #[test]
    fn inspector_selects_generated_job_and_renders_steady_metrics() {
        let p = preview();
        let mut panel = Panel::default();
        panel.sync(&[p.clone()]);
        panel.assign().unwrap();
        panel.pick(vec![Hit {
            preview: p.clone(),
            index: 0,
        }]);
        assert_eq!(panel.edit.as_ref().unwrap().aoi.id, "a");
        let ctx = egui::Context::default();
        let mut capture = CaptureProfile::default();
        let mut shapes = Vec::new();
        for _ in 0..3 {
            shapes = ctx
                .run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(600.0, 1600.0),
                        )),
                        ..Default::default()
                    },
                    |ui| panel.ui(ui, vec![p.clone()], &mut capture),
                )
                .shapes;
        }
        fn text(shape: &egui::Shape, needle: &str) -> bool {
            match shape {
                egui::Shape::Text(t) => t.galley.job.text.contains(needle),
                egui::Shape::Vec(v) => v.iter().any(|s| text(s, needle)),
                _ => false,
            }
        }
        for label in [
            "Selected AOI",
            "Capture cadence",
            "Fresh input / day",
            "Retained window history",
            "Final output / day",
            "Workload total",
        ] {
            assert!(
                shapes.iter().any(|s| text(&s.shape, label)),
                "Missing {label}"
            );
        }
    }
}
