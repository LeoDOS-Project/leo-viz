//! Independent settings and overlays for geographic sampling sources.
use super::preview::Preview;
use eframe::egui;
use std::sync::Arc;

pub(super) struct Source {
    pub kind: &'static str,
    pub name: &'static str,
    pub preview: Option<Arc<Preview>>,
    pub visible: bool,
    pub opacity: f32,
    count: usize,
    seed: u32,
    mean_area: f64,
    size_ratio: f64,
    mean_length: f64,
    length_ratio: f64,
    buffer: f64,
    inland: f64,
    offshore: f64,
    flow: f64,
    compactness: f64,
    latitude: f64,
    tile_edge: f64,
    data_dir: String,
}

impl Source {
    pub fn new(kind: &'static str, name: &'static str) -> Self {
        Self {
            kind,
            name,
            preview: None,
            visible: true,
            opacity: 0.7,
            count: 1000,
            seed: 42,
            mean_area: 1000.0,
            size_ratio: 100.0,
            mean_length: 100.0,
            length_ratio: 20.0,
            buffer: 2.0,
            inland: 2.0,
            offshore: 1.0,
            flow: 10.0,
            compactness: 0.7,
            latitude: 70.0,
            tile_edge: 0.1,
            data_dir: std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tools/leo-aoi/data/environment")
                .to_string_lossy()
                .into_owned(),
        }
    }

    fn request(&self) -> serde_json::Value {
        serde_json::json!({"schema_version":1, "environment": {
            "kind":self.kind, "count":self.count, "seed":self.seed,
            "mean_area_km2":self.mean_area, "size_ratio":self.size_ratio,
            "mean_length_km":self.mean_length, "length_ratio":self.length_ratio,
            "buffer_km":self.buffer, "inland_km":self.inland, "offshore_km":self.offshore,
            "min_discharge":self.flow, "compactness":self.compactness,
            "max_latitude":self.latitude, "tile_edge_km":self.tile_edge,
            "data_dir":self.data_dir}})
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, busy: bool) -> Option<serde_json::Value> {
        let mut request = None;
        ui.push_id(self.kind, |ui| {
            ui.heading(self.name);
            match self.kind {
                "forest" => {
                    ui.hyperlink_to("ESA WorldCover 2021", "https://esa-worldcover.org/en/data-access");
                    ui.label("Area-weighted tree-cover locations. Connected patches grow to a sampled area, preferring forested cells.");
                    ui.small("Coarse tree-cover proxy: 0.025° cells (about 2.8 km north–south). This does not set imagery resolution.");
                }
                "river" => {
                    ui.hyperlink_to("HydroRIVERS v1.0", "https://www.hydrosheds.org/products/hydrorivers");
                    ui.label("Length-weighted river locations. Follow connected reaches downstream and buffer both banks.");
                    ui.small("Paths end at outlets or missing connections. The buffer is a monitoring corridor, not a flood prediction.");
                }
                _ => {
                    ui.hyperlink_to("OpenStreetMap coastlines", "https://osmdata.openstreetmap.de/data/coastlines.html");
                    ui.label("Length-weighted coastline locations. Follow connected shoreline segments with land and sea buffers.");
                    ui.small("Paths stop at gaps or after one circuit of an island. © OpenStreetMap contributors, ODbL.");
                }
            }
            ui.add_enabled_ui(!busy, |ui| {
                egui::Grid::new("settings").num_columns(2).spacing([18.0, 9.0]).show(ui, |ui| {
                    ui.label("Number of AOIs");
                    ui.add(egui::DragValue::new(&mut self.count).range(1..=50_000).speed(100)); ui.end_row();
                    ui.label("Seed"); ui.add(egui::DragValue::new(&mut self.seed)); ui.end_row();
                    if self.kind == "forest" {
                        ui.label("Average area"); ui.add(egui::DragValue::new(&mut self.mean_area).range(10.0..=100_000.0).speed(100.0).suffix(" km²")); ui.end_row();
                        ui.label("Area range (max/min)"); ui.add(egui::DragValue::new(&mut self.size_ratio).range(1.0..=1000.0).suffix("×")); ui.end_row();
                        ui.label("Compactness"); ui.add(egui::Slider::new(&mut self.compactness, 0.0..=1.0)); ui.end_row();
                    } else {
                        ui.label("Average path length"); ui.add(egui::DragValue::new(&mut self.mean_length).range(1.0..=1000.0).speed(5.0).suffix(" km")); ui.end_row();
                        ui.label("Length range (max/min)"); ui.add(egui::DragValue::new(&mut self.length_ratio).range(1.0..=100.0).suffix("×")); ui.end_row();
                        if self.kind == "river" {
                            ui.label("Buffer on each bank"); ui.add(egui::DragValue::new(&mut self.buffer).range(0.1..=50.0).speed(0.1).suffix(" km")); ui.end_row();
                            ui.label("Minimum discharge"); ui.add(egui::DragValue::new(&mut self.flow).range(1.0..=1e6).speed(1.0).suffix(" m³/s")); ui.end_row();
                        } else {
                            ui.label("Inland buffer"); ui.add(egui::DragValue::new(&mut self.inland).range(0.1..=50.0).speed(0.1).suffix(" km")); ui.end_row();
                            ui.label("Offshore buffer"); ui.add(egui::DragValue::new(&mut self.offshore).range(0.1..=50.0).speed(0.1).suffix(" km")); ui.end_row();
                        }
                    }
                    ui.label("Anchor latitude band"); ui.add(egui::DragValue::new(&mut self.latitude).range(1.0..=85.0).prefix("±").suffix("°")); ui.end_row();
                });
                ui.small("Sizes are log-uniform; averages are expected values. Overlapping jobs are retained.");
                ui.collapsing("Dataset cache and geometry", |ui| {
                    ui.label("Local geographic datasets");
                    ui.text_edit_singleline(&mut self.data_dir);
                    ui.horizontal(|ui| { ui.label("Projected tile edge");
                        ui.add(egui::DragValue::new(&mut self.tile_edge).range(0.05..=10.0).speed(0.05).suffix(" km")); });
                    ui.small("Downloads are cached outside Git. Generation reuses the local cache; finer tiles cost more time and memory.");
                });
                let ready = std::path::Path::new(&self.data_dir).join(format!("{}.npz", self.kind)).is_file();
                ui.small(if ready { "Geographic dataset is ready locally." } else { "First generation downloads and prepares this global dataset." });
                #[cfg(not(target_arch = "wasm32"))]
                if ui.button(format!("Generate {}", self.name.to_lowercase())).clicked() { request = Some(self.request()); }
                #[cfg(target_arch = "wasm32")]
                ui.label("Generation requires the native app.");
            });
            if let Some(preview) = &self.preview {
                ui.separator();
                ui.label(format!("{} AOIs · actual average {:.1} km²", preview.count, preview.mean_area));
                ui.small(format!("Area {:.2}–{:.2} km² · seed {}", preview.min_area, preview.max_area, preview.seed));
                if let Some(length) = preview.mean_path_length {
                    ui.small(format!("Actual average path {length:.1} km · {} paths ended early at outlets, gaps or closed coastlines", preview.truncated_paths));
                }
                ui.checkbox(&mut self.visible, "Show on Earth globes");
                ui.add(egui::Slider::new(&mut self.opacity, 0.05..=1.0).text("Opacity"));
                if ui.button("Clear this overlay").clicked() { self.preview = None; }
            }
        });
        request
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "requires prepared environmental datasets and the Python environment"]
    fn environmental_generate_buttons_keep_independent_layers() {
        let ctx = egui::Context::default();
        let mut panel = super::super::AoiGenerator::default();
        let frame = |panel: &mut super::super::AoiGenerator, events| {
            ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(800.0, 1100.0),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| panel.ui(ui),
            )
        };
        fn text(shape: &egui::Shape, needle: &str) -> Option<egui::Pos2> {
            match shape {
                egui::Shape::Text(t) if t.galley.job.text == needle => {
                    Some(t.pos + t.galley.size() * 0.5)
                }
                egui::Shape::Vec(v) => v.iter().find_map(|s| text(s, needle)),
                _ => None,
            }
        }
        for index in 0..3 {
            panel.page = index + 1;
            panel.sources[index].count = 3;
            panel.sources[index].seed = 42 + index as u32;
            let _ = frame(&mut panel, vec![]);
            let _ = frame(&mut panel, vec![]);
            let out = frame(&mut panel, vec![]);
            let label = format!("Generate {}", panel.sources[index].name.to_lowercase());
            let pos = out
                .shapes
                .iter()
                .find_map(|s| text(&s.shape, &label))
                .expect("Generate control is visible");
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
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
            while panel.pending.is_some() && std::time::Instant::now() < deadline {
                panel.poll(&ctx);
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            assert!(panel.pending.is_none(), "Timed out: {}", panel.progress);
            assert!(panel.error.is_none(), "{:?}", panel.error);
            let result = panel.sources[index].preview.as_ref().unwrap();
            assert_eq!(result.count, 3);
            assert_eq!(result.seed, 42 + index as u32);
            assert_eq!(panel.overlays().len(), index + 1);
        }
        assert_eq!(panel.sources[0].count, 3);
        assert_eq!(panel.sources[0].seed, 42);
        panel.sources[0].visible = false;
        assert_eq!(panel.overlays().len(), 2);
        assert!(panel.preview.is_none());
    }
}
