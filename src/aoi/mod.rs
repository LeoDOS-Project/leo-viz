//! Native and browser AOI generation. Generated AOIs stay separate from
//! the circular routing-demo targets and do not acquire application semantics.

mod compact;
mod data;
mod environment;
mod fill;
mod generate;
mod geometry;
pub(crate) mod gpu;
mod presets;
pub(crate) mod preview;
mod spatial;
#[cfg(not(target_arch = "wasm32"))]
mod storage;
mod worker;

use eframe::egui;
use std::sync::Arc;

pub(crate) struct AoiGenerator {
    pub jobs: crate::jobs::Panel,
    pub visible: bool,
    pub opacity: f32,
    fill_opacity: f32,
    legacy_preset: bool,
    shape: String,
    random_rotation: bool,
    max_aspect: f64,
    pub preview: Option<Arc<preview::Preview>>,
    page: usize,
    sources: [environment::Source; 3],
    pending_source: Option<usize>,
    progress: String,
    count: usize,
    mean_area: f64,
    size_ratio: f64,
    seed: u32,
    uniform: bool,
    hotspot_count: usize,
    hotspot_spread: f64,
    hotspot_fraction: f64,
    dataset: data::Input,
    error: Option<String>,
    preview_name: &'static str,
    pending_name: &'static str,
    pending: Option<std::sync::mpsc::Receiver<worker::Event>>,
    #[cfg(not(target_arch = "wasm32"))]
    storage: storage::Panel,
}

impl Default for AoiGenerator {
    fn default() -> Self {
        let defaults = presets::default_settings();
        Self {
            jobs: crate::jobs::Panel::default(),
            visible: true,
            opacity: 0.8,
            fill_opacity: 0.1,
            legacy_preset: false,
            shape: defaults.generation.shape.clone(),
            random_rotation: defaults.generation.random_rotation,
            max_aspect: defaults.generation.max_aspect,
            preview: None,
            page: 0,
            sources: [
                environment::Source::new("forest", "Forests"),
                environment::Source::new("river", "Rivers"),
                environment::Source::new("coast", "Coasts"),
            ],
            pending_source: None,
            progress: String::new(),
            count: defaults.generation.count,
            mean_area: defaults.mean_area_km2,
            size_ratio: defaults.size_ratio,
            seed: defaults.generation.seed,
            uniform: false,
            hotspot_count: defaults.generation.hotspot_count,
            hotspot_spread: defaults.generation.hotspot_spread_km,
            hotspot_fraction: defaults.generation.hotspot_fraction,
            dataset: data::Input::default(),
            error: None,
            preview_name: "Generated AOIs",
            pending_name: "Generated AOIs",
            pending: None,
            #[cfg(not(target_arch = "wasm32"))]
            storage: storage::Panel::default(),
        }
    }
}

impl AoiGenerator {
    fn busy(&self) -> bool {
        let busy = self.pending.is_some() || self.jobs.assigning();
        #[cfg(not(target_arch = "wasm32"))]
        let busy = busy || self.storage.busy();
        busy
    }

    pub fn jobs_ui(
        &mut self,
        ui: &mut egui::Ui,
        capture: &mut leo_sim::acquisition::CaptureProfile,
    ) {
        let layers = self
            .preview
            .iter()
            .chain(self.sources.iter().filter_map(|s| s.preview.as_ref()))
            .cloned()
            .collect();
        self.jobs.ui(ui, layers, capture);
    }

    pub fn overlay(&self) -> Option<(&Arc<preview::Preview>, f32, f32)> {
        self.preview
            .as_ref()
            .filter(|_| self.visible)
            .map(|p| (p, self.opacity, self.fill_opacity))
    }

    fn bounds(&self) -> (f64, f64) {
        let minimum = if self.size_ratio == 1.0 {
            self.mean_area
        } else {
            self.mean_area * self.size_ratio.ln() / (self.size_ratio - 1.0)
        };
        (minimum, minimum * self.size_ratio)
    }

    pub fn overlays(&self) -> Vec<(&Arc<preview::Preview>, f32, f32)> {
        let mut layers: Vec<_> = self.overlay().into_iter().collect();
        for source in &self.sources {
            if source.visible {
                if let Some(preview) = &source.preview {
                    layers.push((preview, source.opacity, self.fill_opacity));
                }
            }
        }
        layers
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        self.poll(ui.ctx());
        let busy = self.busy();
        if !busy {
            if let Some(file) = ui.ctx().input(|i| i.raw.dropped_files.first().cloned()) {
                if self.page == 0 {
                    self.dataset.accept_drop(&file);
                } else {
                    self.sources[self.page - 1].accept_drop(&file);
                }
            }
        }
        ui.horizontal_wrapped(|ui| {
            for (index, label) in ["GDP rectangles", "Forests", "Rivers", "Coasts"]
                .iter()
                .enumerate()
            {
                ui.selectable_value(&mut self.page, index, *label);
            }
        });
        ui.separator();
        if self.page == 0 {
            self.gdp_ui(ui);
        } else {
            let index = self.page - 1;
            if let Some(action) = self.sources[index].ui(ui, busy) {
                self.error = None;
                self.progress = "Preparing geographic source…".into();
                self.pending_source = Some(index);
                self.pending = Some(match action {
                    environment::Action::Generate(settings, dataset) => {
                        worker::start(settings, dataset, ui.ctx().clone())
                    }
                    environment::Action::Compact(preview) => {
                        self.progress = "Simplifying stored AOI boundaries…".into();
                        worker::compact(preview, ui.ctx().clone())
                    }
                });
            }
            if let Some(error) = &self.error {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
            }
        }
        if busy {
            ui.horizontal_wrapped(|ui| {
                ui.spinner();
                ui.label(&self.progress);
            });
        }
        ui.separator();
        ui.label("Visible overlays");
        ui.horizontal_wrapped(|ui| {
            ui.add_enabled(
                self.preview.is_some(),
                egui::Checkbox::new(&mut self.visible, "GDP"),
            );
            for source in &mut self.sources {
                ui.add_enabled(
                    source.preview.is_some(),
                    egui::Checkbox::new(&mut source.visible, source.name),
                );
            }
        });
        ui.add(
            egui::Slider::new(&mut self.fill_opacity, 0.0..=0.3)
                .text("Fill opacity")
                .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
        );
        ui.small("Off-screen regions are skipped. Distant AOIs use simpler outlines or grouped dots; zoom in for detail.");
    }

    pub fn presets_ui(
        &mut self,
        ui: &mut egui::Ui,
        capture: &mut leo_sim::acquisition::CaptureProfile,
    ) {
        #[cfg(target_arch = "wasm32")]
        let _ = capture;
        self.poll(ui.ctx());
        #[cfg(not(target_arch = "wasm32"))]
        self.jobs.apply_restored_capture(capture);
        egui::CollapsingHeader::new("AOIs & jobs · presets")
            .id_salt("aoi-library").default_open(true).show(ui, |ui| {
                #[cfg(target_arch = "wasm32")]
                ui.add_enabled_ui(!self.busy(), |ui| {
                    ui.horizontal_wrapped(|ui| {
                        egui::ComboBox::from_id_salt("aoi-preset-choice")
                            .selected_text(if self.legacy_preset { presets::LEGACY_JOBS } else { presets::DEFAULT_JOBS })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.legacy_preset, false, presets::DEFAULT_JOBS);
                                ui.selectable_value(&mut self.legacy_preset, true, presets::LEGACY_JOBS);
                            });
                        if ui.button("Load").clicked() { self.load_default_jobs(ui.ctx()); }
                    });
                    ui.small(if self.legacy_preset { "50,000 original GDP rectangles · 1,000 km² average." } else { "53,000 AOIs: 50,000 GDP rectangles + 1,000 each of forests, rivers and coasts. Assign job settings separately below." });
                    ui.small("Presets contain saved geometry. Loading does not regenerate AOIs.");
                });
                #[cfg(not(target_arch = "wasm32"))]
                self.storage_ui(ui, capture);
                if self.pending.is_some() {
                    ui.horizontal_wrapped(|ui| { ui.spinner(); ui.label(&self.progress); });
                }
                if let Some(error) = &self.error { ui.colored_label(egui::Color32::LIGHT_RED, error); }
            });
    }

    fn gdp_ui(&mut self, ui: &mut egui::Ui) {
        let context = ui.ctx().clone();
        let ctx = &context;
        self.poll(ctx);
        let busy = self.busy();

        ui.push_id("aoi-generator", |ui| {
                ui.label("Generate rectangular AOIs on Earth. Overlaps are allowed.");
                ui.add_enabled_ui(!busy, |ui| {
                    egui::Grid::new("aoi-generator-controls").num_columns(2).spacing([16.0, 9.0])
                        .show(ui, |ui| {
                            ui.label("Number of AOIs");
                            ui.add(egui::DragValue::new(&mut self.count).range(1..=50_000).speed(100));
                            ui.end_row();
                            ui.label("Average area");
                            ui.add(egui::DragValue::new(&mut self.mean_area).range(1.0..=1_000_000.0)
                                .speed(1.0).suffix(" km²"));
                            ui.end_row();
                            ui.label("Size range (max/min)");
                            ui.add(egui::DragValue::new(&mut self.size_ratio).range(1.0..=10_000.0)
                                .speed(10.0).suffix("×"));
                            ui.end_row();
                            ui.label("Locations");
                            egui::ComboBox::from_id_salt("aoi-location-source")
                                .selected_text(if self.uniform { "Uniform globe" } else { "GDP / demand dataset" })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut self.uniform, false, "GDP / demand dataset");
                                    ui.selectable_value(&mut self.uniform, true, "Uniform globe");
                                });
                            ui.end_row();
                        });
                    let (minimum, maximum) = self.bounds();
                    ui.small(format!("Requested size range: {minimum:.2}–{maximum:.2} km²"));
                    ui.small(if self.shape == "oriented_rectangle" { "Log-uniform areas and aspect ratios. Dimensions use local ground distances." } else { "Legacy grid dimensions vary in ground aspect ratio with latitude." });
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Geometry");
                        egui::ComboBox::from_id_salt("aoi-rectangle-shape")
                            .selected_text(if self.shape == "oriented_rectangle" { "Ground rectangles" } else { "Grid rectangles (legacy)" })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.shape, "oriented_rectangle".into(), "Ground rectangles");
                                ui.selectable_value(&mut self.shape, "rectangle".into(), "Grid rectangles (legacy)");
                            });
                    });
                    ui.add_enabled(self.shape == "oriented_rectangle", egui::Checkbox::new(&mut self.random_rotation, "Random rotation (0–180°)"));
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Maximum elongation");
                        ui.add(egui::DragValue::new(&mut self.max_aspect).range(1.0..=10.0).speed(0.1).suffix(":1"));
                    });
                    ui.collapsing("Clustering and reproducibility", |ui| {
                        ui.horizontal(|ui| { ui.label("Seed"); ui.add(egui::DragValue::new(&mut self.seed)); });
                        ui.horizontal(|ui| {
                            ui.label("Hotspots (0 = independent)");
                            ui.add(egui::DragValue::new(&mut self.hotspot_count).range(0..=10_000).speed(10));
                        });
                        ui.add_enabled_ui(self.hotspot_count > 0, |ui| {
                            ui.add(egui::Slider::new(&mut self.hotspot_fraction, 0.0..=1.0).text("Clustered fraction"));
                            ui.horizontal(|ui| {
                                ui.label("Hotspot spread");
                                ui.add(egui::DragValue::new(&mut self.hotspot_spread).range(0.0..=2000.0)
                                    .speed(1.0).suffix(" km"));
                            });
                        });
                    });
                    ui.collapsing("Dataset", |ui| {
                        ui.add_enabled_ui(!self.uniform, |ui| self.dataset.ui(ui));
                        if self.dataset.is_default() && !self.uniform { ui.label("GDP 2024"); }
                    });
                    // Only legacy grid rectangles require at least one tile.
                    let valid_size = self.shape == "oriented_rectangle" || minimum >= 0.01;
                    if !valid_size {
                        ui.colored_label(egui::Color32::YELLOW,
                            "Increase average area or reduce the size range: smallest AOI is below one tile (0.01 km²).");
                    }
                    if ui.add_enabled(valid_size, egui::Button::new("Generate")).clicked() {
                        self.start(ctx);
                    }
                });
                if let Some(error) = &self.error { ui.colored_label(egui::Color32::LIGHT_RED, error); }
                if let Some(preview) = &self.preview {
                    ui.separator();
                    ui.strong(self.preview_name);
                    ui.label(format!("{} AOIs · actual average {:.1} km²", preview.count, preview.mean_area));
                    ui.small(format!("Area {:.2}–{:.2} km² · seed {}", preview.min_area, preview.max_area, preview.seed));
                    preview.complexity_ui(ui);
                    ui.checkbox(&mut self.visible, "Show AOIs on Earth globes");
                    ui.add(egui::Slider::new(&mut self.opacity, 0.05..=1.0).text("Outline opacity"));
                    ui.small("Zoom in for outlines. Subpixel AOIs appear as dots.");
                    if ui.add_enabled(!busy, egui::Button::new("Clear overlay")).clicked() { self.preview = None; }
                }
            });
    }

    pub(crate) fn load_default_jobs(&mut self, _ctx: &egui::Context) {
        if self.busy() { return; }
        let defaults = presets::settings(self.legacy_preset);
        self.shape = defaults.generation.shape.clone();
        self.random_rotation = defaults.generation.random_rotation;
        self.max_aspect = defaults.generation.max_aspect;
        self.count = defaults.generation.count;
        self.mean_area = defaults.mean_area_km2;
        self.size_ratio = defaults.size_ratio;
        self.seed = defaults.generation.seed;
        self.uniform = false;
        self.hotspot_count = defaults.generation.hotspot_count;
        self.hotspot_spread = defaults.generation.hotspot_spread_km;
        self.hotspot_fraction = defaults.generation.hotspot_fraction;
        self.dataset = data::Input::default();
        self.error = None;
        self.pending_source = None;
        self.pending_name = if self.legacy_preset {
            presets::LEGACY_JOBS
        } else {
            presets::DEFAULT_JOBS
        };
        self.progress = format!("Loading {}…", self.pending_name);
        self.pending = Some(worker::load_preset(_ctx.clone(), self.legacy_preset));
    }

    fn start(&mut self, ctx: &egui::Context) {
        let settings = generate::Settings::Gdp(generate::Gdp {
            count: self.count,
            seed: self.seed,
            mean_area: self.mean_area,
            size_ratio: self.size_ratio,
            uniform: self.uniform,
            hotspot_count: self.hotspot_count,
            hotspot_spread: self.hotspot_spread,
            hotspot_fraction: self.hotspot_fraction,
            shape: self.shape.clone(),
            random_rotation: self.random_rotation,
            max_aspect: self.max_aspect,
        });
        self.error = None;
        self.pending_source = None;
        self.progress = "Generating GDP rectangles…".into();
        self.pending_name = "Generated AOIs";
        self.pending = Some(worker::start(settings, self.dataset.clone(), ctx.clone()));
    }

    pub(crate) fn poll(&mut self, ctx: &egui::Context) {
        self.poll_aois(ctx);
        let layers: Vec<_> = self
            .preview
            .iter()
            .chain(self.sources.iter().filter_map(|s| s.preview.as_ref()))
            .cloned()
            .collect();
        self.jobs.poll(ctx, &layers);
    }

    fn poll_aois(&mut self, ctx: &egui::Context) {
        use std::sync::mpsc::TryRecvError;
        #[cfg(not(target_arch = "wasm32"))]
        self.poll_storage();
        loop {
            let Some(rx) = &self.pending else {
                return;
            };
            match rx.try_recv() {
                Ok(worker::Event::Progress(message)) => {
                    self.progress = message;
                }
                Ok(worker::Event::PresetFinished(result)) => {
                    self.pending = None;
                    match result {
                        Ok(layers) => {
                            self.jobs.restore_workload(None);
                            let mut layers = layers.into_iter();
                            self.preview = layers.next();
                            self.visible = true;
                            self.preview_name = self.pending_name;
                            for source in &mut self.sources {
                                *source = environment::Source::new(source.kind, source.name);
                                source.preview = layers.next();
                                source.visible = source.preview.is_some();
                            }
                        }
                        Err(error) => self.error = Some(error),
                    }
                    return;
                }
                Ok(worker::Event::Finished(Ok(preview))) => {
                    if let Some(index) = self.pending_source.take() {
                        self.sources[index].preview = Some(Arc::new(preview));
                        self.sources[index].visible = true;
                    } else {
                        self.preview = Some(Arc::new(preview));
                        self.visible = true;
                        self.preview_name = self.pending_name;
                    }
                    self.pending = None;
                    return;
                }
                Ok(worker::Event::Finished(Err(error))) => {
                    self.error = Some(error);
                    self.pending_source = None;
                    self.pending = None;
                    return;
                }
                Err(TryRecvError::Disconnected) => {
                    self.error =
                        Some("AOI worker stopped unexpectedly. Try generating again.".into());
                    self.pending_source = None;
                    self.pending = None;
                    return;
                }
                Err(TryRecvError::Empty) => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_preset_keeps_every_existing_layer() {
        let preview = Arc::new(preview::Preview::empty(0, 42).finish());
        let mut panel = AoiGenerator::default();
        panel.preview = Some(preview.clone());
        for source in &mut panel.sources { source.preview = Some(preview.clone()); }
        let (tx, rx) = std::sync::mpsc::channel();
        panel.pending = Some(rx);
        tx.send(worker::Event::PresetFinished(Err("Missing forest source".into()))).unwrap();
        panel.poll(&egui::Context::default());
        assert!(Arc::ptr_eq(panel.preview.as_ref().unwrap(), &preview));
        for source in &panel.sources { assert!(Arc::ptr_eq(source.preview.as_ref().unwrap(), &preview)); }
        assert!(panel.error.as_ref().unwrap().contains("Missing forest"));
        assert!(panel.pending.is_none());
    }

    fn text_position(shapes: &[egui::epaint::ClippedShape], needle: &str) -> Option<egui::Pos2> {
        fn find(shape: &egui::Shape, needle: &str) -> Option<egui::Pos2> {
            match shape {
                egui::Shape::Text(t) if t.galley.job.text == needle => {
                    Some(t.pos + t.galley.size() * 0.5)
                }
                egui::Shape::Vec(v) => v.iter().find_map(|s| find(s, needle)),
                _ => None,
            }
        }
        shapes.iter().find_map(|s| find(&s.shape, needle))
    }

    #[test]
    fn panel_has_generation_controls_and_valid_size_bounds() {
        let ctx = egui::Context::default();
        let mut panel = AoiGenerator::default();
        let mut output = None;
        for _ in 0..3 {
            output = Some(ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1000.0, 800.0),
                    )),
                    ..Default::default()
                },
                |ui| panel.ui(ui),
            ));
        }
        let shapes = &output.unwrap().shapes;
        assert!(text_position(shapes, "Number of AOIs").is_some());
        assert!(text_position(shapes, "Default AOIs (bundled)").is_none());
        assert!(text_position(shapes, "Load").is_none());
        assert!(text_position(shapes, "Average area").is_some());
        assert!(text_position(shapes, "Generate").is_some());
        let (lo, hi) = panel.bounds();
        assert!(((hi - lo) / (hi / lo).ln() - panel.mean_area).abs() < 1e-9);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn default_jobs_loads_without_python_or_demand_files() {
        let ctx = egui::Context::default();
        let mut panel = AoiGenerator::default();
        panel.storage.selected_saved = false;
        assert_eq!(panel.count, 50_000);
        assert_eq!(panel.mean_area, 100.0);
        assert_eq!(panel.size_ratio, 1000.0);
        assert_eq!(panel.seed, 42);
        assert_eq!(panel.hotspot_count, 600);
        assert_eq!(panel.hotspot_spread, 100.0);
        assert_eq!(panel.hotspot_fraction, 0.4);
        panel.dataset.path = "/nonexistent/demand.aoi-data.json.gz".into();
        panel.count = 1;
        panel.visible = false;
        let frame = |panel: &mut AoiGenerator, events| {
            ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1000.0, 800.0),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| panel.presets_ui(ui, &mut Default::default()),
            )
        };
        let _ = frame(&mut panel, vec![]);
        let _ = frame(&mut panel, vec![]);
        let output = frame(&mut panel, vec![]);
        let pos = text_position(&output.shapes, "Load").unwrap();
        for pressed in [true, false] {
            let _ = frame(
                &mut panel,
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
        assert!(panel.pending.is_some());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(180);
        while panel.pending.is_some() && std::time::Instant::now() < deadline {
            panel.poll(&ctx);
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(panel.pending.is_none(), "preset loading timed out");
        assert!(panel.error.is_none(), "{:?}", panel.error);
        let preview = panel.preview.as_ref().unwrap();
        assert_eq!(preview.count, 50_000);
        assert_eq!(preview.seed, 42);
        assert_eq!(panel.preview_name, "Default AOIs (bundled)");
        assert_eq!(panel.count, 50_000);
        assert!(panel.overlay().is_some());
        let (lo, hi) = panel.bounds();
        assert!(preview.min_area >= lo && preview.max_area <= hi);
        assert_eq!(panel.overlays().len(), 4);
        for source in &panel.sources {
            let layer = source.preview.as_ref().unwrap();
            assert_eq!(layer.count, 1000);
            assert!(layer.max_edges <= compact::EDGE_LIMIT);
            assert!(layer.aois.iter().all(|a| a.source == source.kind));
        }
        assert_eq!(panel.overlays().iter().map(|(p, _, _)| p.count).sum::<usize>(), 53_000);
        // A second click reuses prepared geometry and GPU buffers for every layer.
        let previous: Vec<_> = panel.overlays().iter().map(|(p, _, _)| Arc::clone(p)).collect();
        panel.load_default_jobs(&ctx);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while panel.pending.is_some() && std::time::Instant::now() < deadline {
            panel.poll(&ctx);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(panel.pending.is_none(), "cached preset loading timed out");
        assert!(panel.error.is_none(), "{:?}", panel.error);
        for (old, (current, _, _)) in previous.iter().zip(panel.overlays()) {
            assert!(Arc::ptr_eq(old, current));
        }
        assert!(panel.random_rotation);
        assert_eq!(panel.fill_opacity, 0.1);
        assert_eq!(
            presets::default_settings().generation.shape,
            "oriented_rectangle"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn clicking_generate_finishes_in_background_and_closing_keeps_overlay() {
        let ctx = egui::Context::default();
        let mut panel = AoiGenerator::default();
        panel.uniform = true;
        panel.count = 30;
        let frame = |panel: &mut AoiGenerator, events| {
            ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1000.0, 800.0),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| panel.ui(ui),
            )
        };
        let _ = frame(&mut panel, vec![]);
        let _ = frame(&mut panel, vec![]);
        let output = frame(&mut panel, vec![]);
        let pos = text_position(&output.shapes, "Generate").unwrap();
        for pressed in [true, false] {
            let _ = frame(
                &mut panel,
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
        assert!(panel.pending.is_some());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while panel.pending.is_some() && std::time::Instant::now() < deadline {
            panel.poll(&ctx);
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(panel.error.is_none(), "{:?}", panel.error);
        assert_eq!(panel.preview.as_ref().unwrap().count, 30);
        assert!(panel.overlay().is_some());
        panel.visible = false;
        assert!(panel.overlay().is_none());
    }
}
