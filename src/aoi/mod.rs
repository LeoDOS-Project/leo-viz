//! Interactive client of tools/leo-aoi. Generated AOIs stay separate from
//! the circular routing-demo targets and do not acquire application semantics.

mod environment;
mod gpu;
mod presets;
pub(crate) mod preview;
mod spatial;
#[cfg(not(target_arch = "wasm32"))]
mod worker;

use eframe::egui;
use std::sync::Arc;

pub(crate) struct AoiGenerator {
    pub jobs: crate::jobs::Panel,
    pub visible: bool,
    pub opacity: f32,
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
    python: String,
    demand_path: String,
    error: Option<String>,
    preview_name: &'static str,
    pending_name: &'static str,
    #[cfg(not(target_arch = "wasm32"))]
    pending: Option<std::sync::mpsc::Receiver<worker::Event>>,
}

impl Default for AoiGenerator {
    fn default() -> Self {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/leo-aoi");
        let venv = root.join(if cfg!(windows) {
            ".venv/Scripts/python.exe"
        } else {
            ".venv/bin/python"
        });
        let defaults = presets::default_settings();
        Self {
            jobs: crate::jobs::Panel::default(),
            visible: true,
            opacity: 0.8,
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
            python: std::env::var("LEO_AOI_PYTHON").unwrap_or_else(|_| {
                if venv.is_file() {
                    venv.to_string_lossy().into_owned()
                } else {
                    "python3".into()
                }
            }),
            demand_path: root
                .join("data/gdp-demand.npz")
                .to_string_lossy()
                .into_owned(),
            error: None,
            preview_name: "Generated AOIs",
            pending_name: "Generated AOIs",
            #[cfg(not(target_arch = "wasm32"))]
            pending: None,
        }
    }
}

impl AoiGenerator {
    pub fn jobs_ui(&mut self, ui: &mut egui::Ui, capture: &mut leo_sim::acquisition::CaptureProfile) {
        let layers = self.preview.iter().chain(self.sources.iter().filter_map(|s| s.preview.as_ref())).cloned().collect();
        self.jobs.ui(ui, layers, capture);
    }

    pub fn overlay(&self) -> Option<(&Arc<preview::Preview>, f32)> {
        self.preview
            .as_ref()
            .filter(|_| self.visible)
            .map(|p| (p, self.opacity))
    }

    fn bounds(&self) -> (f64, f64) {
        let minimum = if self.size_ratio == 1.0 {
            self.mean_area
        } else {
            self.mean_area * self.size_ratio.ln() / (self.size_ratio - 1.0)
        };
        (minimum, minimum * self.size_ratio)
    }

    pub fn overlays(&self) -> Vec<(&Arc<preview::Preview>, f32)> {
        let mut layers: Vec<_> = self.overlay().into_iter().collect();
        for source in &self.sources {
            if source.visible {
                if let Some(preview) = &source.preview {
                    layers.push((preview, source.opacity));
                }
            }
        }
        layers
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        #[cfg(not(target_arch = "wasm32"))]
        self.poll(ui.ctx());
        #[cfg(not(target_arch = "wasm32"))]
        let busy = self.pending.is_some();
        #[cfg(target_arch = "wasm32")]
        let busy = false;
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
            if let Some(request) = self.sources[index].ui(ui, busy) {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.error = None;
                    self.progress = "Preparing geographic source…".into();
                    self.pending_source = Some(index);
                    self.pending = Some(worker::start(
                        self.python.clone(),
                        request,
                        ui.ctx().clone(),
                    ));
                }
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
        ui.small("Off-screen regions are skipped. Distant AOIs use simpler outlines or grouped dots; zoom in for detail.");
    }

    fn gdp_ui(&mut self, ui: &mut egui::Ui) {
        let context = ui.ctx().clone();
        let ctx = &context;
        #[cfg(not(target_arch = "wasm32"))]
        self.poll(ctx);
        #[cfg(not(target_arch = "wasm32"))]
        let busy = self.pending.is_some();
        #[cfg(target_arch = "wasm32")]
        let busy = false;

        ui.push_id("aoi-generator", |ui| {
                ui.add_enabled_ui(!busy, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("AOI preset:");
                        ui.strong(presets::DEFAULT_JOBS);
                        if ui.button("Load preset").clicked() { self.load_default_jobs(ctx); }
                    });
                    ui.small("50,000 saved GDP-weighted rectangles. Loads without Python.");
                });
                ui.separator();
                ui.label("Generate rectangular AOIs on Earth. Overlaps are allowed.");
                ui.add_enabled_ui(!busy, |ui| {
                    egui::Grid::new("aoi-generator-controls").num_columns(2).spacing([16.0, 9.0])
                        .show(ui, |ui| {
                            ui.label("Number of AOIs");
                            ui.add(egui::DragValue::new(&mut self.count).range(1..=50_000).speed(100));
                            ui.end_row();
                            ui.label("Average area");
                            ui.add(egui::DragValue::new(&mut self.mean_area).range(1.0..=1_000_000.0)
                                .speed(100.0).suffix(" km²"));
                            ui.end_row();
                            ui.label("Size range (max/min)");
                            ui.add(egui::DragValue::new(&mut self.size_ratio).range(1.0..=10_000.0)
                                .speed(10.0).suffix("×"));
                            ui.end_row();
                            ui.label("Locations");
                            egui::ComboBox::from_id_salt("aoi-location-source")
                                .selected_text(if self.uniform { "Uniform globe" } else { "Demand dataset" })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut self.uniform, false, "Demand dataset");
                                    ui.selectable_value(&mut self.uniform, true, "Uniform globe");
                                });
                            ui.end_row();
                        });
                    let (minimum, maximum) = self.bounds();
                    ui.small(format!("Requested size range: {minimum:.2}–{maximum:.2} km²"));
                    ui.small("Log-uniform sizes; rectangle dimensions are rounded to the grid.");
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
                    ui.collapsing("Data and runtime", |ui| {
                        ui.label("Demand field (.npz)");
                        ui.add_enabled(!self.uniform, egui::TextEdit::singleline(&mut self.demand_path)
                            .desired_width(f32::INFINITY));
                        ui.small("Defaults to the ingested GDP grid. Any leo-aoi demand field works.");
                        ui.label("Python executable");
                        ui.add(egui::TextEdit::singleline(&mut self.python).desired_width(f32::INFINITY));
                        ui.small("Setup instructions: tools/leo-aoi/README.md");
                    });
                    // Keep the batch generator's 100 m projected tile grid.
                    let valid_size = minimum >= 0.01;
                    if !valid_size {
                        ui.colored_label(egui::Color32::YELLOW,
                            "Increase average area or reduce the size range: smallest AOI is below one tile (0.01 km²).");
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    if ui.add_enabled(valid_size, egui::Button::new("Generate")).clicked() {
                        self.start(ctx);
                    }
                    #[cfg(target_arch = "wasm32")]
                    ui.label("Generation requires the native desktop app and the Python AOI tool.");
                });
                if let Some(error) = &self.error { ui.colored_label(egui::Color32::LIGHT_RED, error); }
                if let Some(preview) = &self.preview {
                    ui.separator();
                    ui.strong(self.preview_name);
                    ui.label(format!("{} AOIs · actual average {:.1} km²", preview.count, preview.mean_area));
                    ui.small(format!("Area {:.2}–{:.2} km² · seed {}", preview.min_area, preview.max_area, preview.seed));
                    ui.checkbox(&mut self.visible, "Show AOIs on Earth globes");
                    ui.add(egui::Slider::new(&mut self.opacity, 0.05..=1.0).text("Opacity"));
                    ui.small("Zoom in for outlines. Subpixel AOIs appear as dots.");
                    if ui.add_enabled(!busy, egui::Button::new("Clear overlay")).clicked() { self.preview = None; }
                }
            });
    }

    fn load_default_jobs(&mut self, _ctx: &egui::Context) {
        let defaults = presets::default_settings();
        self.count = defaults.generation.count;
        self.mean_area = defaults.mean_area_km2;
        self.size_ratio = defaults.size_ratio;
        self.seed = defaults.generation.seed;
        self.uniform = false;
        self.hotspot_count = defaults.generation.hotspot_count;
        self.hotspot_spread = defaults.generation.hotspot_spread_km;
        self.hotspot_fraction = defaults.generation.hotspot_fraction;
        self.demand_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tools/leo-aoi/data/gdp-demand.npz")
            .to_string_lossy()
            .into_owned();
        self.error = None;
        self.pending_source = None;
        self.progress = "Loading Default Jobs…".into();
        self.pending_name = presets::DEFAULT_JOBS;
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.pending = Some(worker::load_default_jobs(_ctx.clone()));
        }
        #[cfg(target_arch = "wasm32")]
        match presets::load_default_jobs() {
            Ok(preview) => {
                self.preview = Some(Arc::new(preview));
                self.preview_name = presets::DEFAULT_JOBS;
                self.visible = true;
            }
            Err(error) => self.error = Some(error),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn start(&mut self, ctx: &egui::Context) {
        let request = serde_json::json!({
            "schema_version": 1,
            "demand_path": if self.uniform { None } else { Some(&self.demand_path) },
            "mean_area_km2": self.mean_area, "size_ratio": self.size_ratio,
            "generation": {"count": self.count, "seed": self.seed, "tile_edge_km": 0.1,
                "shape": presets::default_settings().generation.shape,
                "hotspot_count": self.hotspot_count, "hotspot_spread_km": self.hotspot_spread,
                "hotspot_fraction": self.hotspot_fraction}
        });
        self.error = None;
        self.pending_source = None;
        self.progress = "Generating GDP rectangles…".into();
        self.pending_name = "Generated AOIs";
        self.pending = Some(worker::start(self.python.clone(), request, ctx.clone()));
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn poll(&mut self, ctx: &egui::Context) {
        use std::sync::mpsc::TryRecvError;
        loop {
            let Some(rx) = &self.pending else {
                return;
            };
            match rx.try_recv() {
                Ok(worker::Event::Progress(message)) => {
                    self.progress = message;
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
        assert!(text_position(shapes, "Default Jobs").is_some());
        assert!(text_position(shapes, "Load preset").is_some());
        assert!(text_position(shapes, "Average area").is_some());
        #[cfg(not(target_arch = "wasm32"))]
        assert!(text_position(shapes, "Generate").is_some());
        let (lo, hi) = panel.bounds();
        assert!(((hi - lo) / (hi / lo).ln() - panel.mean_area).abs() < 1e-9);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn default_jobs_loads_without_python_or_demand_files() {
        let ctx = egui::Context::default();
        let mut panel = AoiGenerator::default();
        assert_eq!(panel.count, 50_000);
        assert_eq!(panel.mean_area, 1000.0);
        assert_eq!(panel.size_ratio, 1000.0);
        assert_eq!(panel.seed, 42);
        assert_eq!(panel.hotspot_count, 600);
        assert_eq!(panel.hotspot_spread, 100.0);
        assert_eq!(panel.hotspot_fraction, 0.4);
        panel.python = "/nonexistent/python".into();
        panel.demand_path = "/nonexistent/demand.npz".into();
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
                |ui| panel.ui(ui),
            )
        };
        let _ = frame(&mut panel, vec![]);
        let _ = frame(&mut panel, vec![]);
        let output = frame(&mut panel, vec![]);
        let pos = text_position(&output.shapes, "Load preset").unwrap();
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
        assert!(panel.pending.is_none(), "preset loading timed out");
        assert!(panel.error.is_none(), "{:?}", panel.error);
        let preview = panel.preview.as_ref().unwrap();
        assert_eq!(preview.count, 50_000);
        assert_eq!(preview.seed, 42);
        assert_eq!(panel.preview_name, "Default Jobs");
        assert_eq!(panel.count, 50_000);
        assert!(panel.overlay().is_some());
        // Rounding one rectangle's dimensions can change its area by half
        // a tile column (aspect >= 0.5, projected tile edge 0.1 km).
        let (lo, hi) = panel.bounds();
        let tolerance = 0.1 * hi.sqrt() + 0.01;
        assert!(preview.min_area > 0.0 && preview.min_area >= lo - tolerance);
        assert!(preview.max_area <= hi + tolerance);
        assert_eq!(presets::default_settings().generation.shape, "rectangle");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "requires the leo-aoi Python environment"]
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
