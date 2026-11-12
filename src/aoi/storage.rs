//! Native AOI geometry persistence, independent of workloads and checkpoints.
use super::{
    preview::{Config, Document, Preview},
    AoiGenerator,
};
use eframe::egui;
use serde::{ser::SerializeSeq, Serialize};
use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
    sync::{mpsc, Arc},
};

const MAX_JSON_BYTES: u64 = 256 * 1024 * 1024;
const KINDS: [&str; 4] = ["gdp", "forest", "river", "coast"];
type Layers = [Option<Arc<Preview>>; 4];

struct Records<'a>(&'a [Arc<Preview>]);
impl Serialize for Records<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.0.iter().map(|p| p.count).sum()))?;
        for layer in self.0 {
            layer.serialize_records(&mut seq)?;
        }
        seq.end()
    }
}

// Enforce the same decoded limit on write and read: a successful save must be
// loadable, including when highly compressed geometry is very large.
struct LimitedWriter<W> {
    inner: W,
    remaining: u64,
}
impl<W: Write> Write for LimitedWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() as u64 > self.remaining {
            return Err(std::io::Error::other(
                "AOIs exceed the 256 MB decoded limit",
            ));
        }
        let n = self.inner.write(bytes)?;
        self.remaining -= n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn save(path: &Path, layers: &[Arc<Preview>]) -> Result<usize, String> {
    let count = layers.iter().map(|p| p.count).sum();
    if count == 0 {
        return Err("Load or generate AOIs first.".into());
    }
    let mut source_seeds = BTreeMap::new();
    let mut source_counts = [0; 4];
    let mut keys = std::collections::HashSet::new();
    for layer in layers {
        for aoi in &layer.aois {
            let i = KINDS
                .iter()
                .position(|k| *k == aoi.source)
                .ok_or_else(|| format!("Unsupported AOI source: {}", aoi.source))?;
            source_counts[i] += 1;
            if source_counts[i] > 50_000 || !keys.insert(aoi.key()) {
                return Err(
                    "AOI IDs must be unique within each source, with at most 50,000 per source."
                        .into(),
                );
            }
            source_seeds.insert(aoi.source.clone(), layer.seed);
        }
    }
    #[derive(Serialize)]
    struct Saved<'a> {
        schema_version: u32,
        config: Config,
        aois: Records<'a>,
    }
    let saved = Saved {
        schema_version: 1,
        config: Config {
            seed: layers[0].seed,
            count,
            source_seeds,
        },
        aois: Records(layers),
    };
    let tmp = path.with_extension(format!(
        "{}.{}.tmp",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let result = (|| -> Result<(), String> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|e| e.to_string())?;
        let mut writer = BufWriter::new(file);
        if path.extension().is_some_and(|e| e == "gz") {
            let mut gzip = flate2::write::GzEncoder::new(&mut writer, flate2::Compression::fast());
            serde_json::to_writer(
                &mut LimitedWriter {
                    inner: &mut gzip,
                    remaining: MAX_JSON_BYTES,
                },
                &saved,
            )
            .map_err(|e| e.to_string())?;
            gzip.finish().map_err(|e| e.to_string())?;
        } else {
            serde_json::to_writer(
                &mut LimitedWriter {
                    inner: &mut writer,
                    remaining: MAX_JSON_BYTES,
                },
                &saved,
            )
            .map_err(|e| e.to_string())?;
        }
        writer.flush().map_err(|e| e.to_string())?;
        writer.get_ref().sync_all().map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(tmp);
    }
    result.map(|()| count)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn preview(kind: &str, seed: u32, dateline: bool) -> Arc<Preview> {
        let coordinates = if dateline {
            serde_json::json!([
                [[
                    [179.12345678901235, 1.0],
                    [180.0, 1.0],
                    [180.0, 2.0],
                    [179.12345678901235, 2.0],
                    [179.12345678901235, 1.0]
                ]],
                [[
                    [-180.0, 1.0],
                    [-179.0, 1.0],
                    [-179.0, 2.0],
                    [-180.0, 2.0],
                    [-180.0, 1.0]
                ]]
            ])
        } else {
            serde_json::json!([[
                [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [2.0, 1.0], [1.0, 1.0]]
            ]])
        };
        Arc::new(Preview::from_json(&serde_json::to_vec(&serde_json::json!({
            "schema_version":1,"config":{"seed":seed,"count":1},
            "aois":[{"id":"same-id","source_kind":kind,"anchor_lon":if dateline {179.5} else {0.5},
                "anchor_lat":1.5,"area_km2":123.45678901234567,
                "path_truncated":kind=="river","path_length_km":if kind=="river" {Some(12.345678901234567)} else {None},
                "geometry":{"type":"MultiPolygon","coordinates":coordinates}}]
        })).unwrap()).unwrap())
    }
    #[test]
    fn all_sources_round_trip_exact_geometry_identity_and_workload_references() {
        let originals: Vec<_> = KINDS
            .iter()
            .enumerate()
            .map(|(i, k)| preview(k, 40 + i as u32, i == 0))
            .collect();
        let refs: Vec<_> = originals.iter().flat_map(|p| p.aois.clone()).collect();
        let jobs = leo_capture::workload::Generator::default()
            .generate(refs.clone())
            .unwrap();
        for extension in ["json", "json.gz"] {
            let path = std::env::temp_dir().join(format!(
                "leo-aoi-roundtrip-{}.{}",
                std::process::id(),
                extension
            ));
            assert_eq!(save(&path, &originals).unwrap(), 4);
            let restored = load(&path, |_| {}).unwrap();
            for (a, b) in originals.iter().zip(restored.iter()) {
                let b = b.as_ref().unwrap();
                assert_eq!(a.aois, b.aois);
                assert_eq!(a.seed, b.seed);
                assert_eq!(a.truncated_paths, b.truncated_paths);
                assert_eq!(a.mean_path_length, b.mean_path_length);
                assert_eq!(
                    serde_json::to_value(a.capture_regions()).unwrap(),
                    serde_json::to_value(b.capture_regions()).unwrap()
                );
            }
            // Export is also a standard leo-aoi dataset readable by the headless simulator.
            let headless = leo_capture::dataset::decode(&std::fs::read(&path).unwrap()).unwrap();
            assert_eq!(
                headless.into_iter().map(|(a, _)| a).collect::<Vec<_>>(),
                refs
            );
            assert_eq!(
                restored
                    .iter()
                    .flatten()
                    .flat_map(|p| p.aois.iter())
                    .collect::<Vec<_>>(),
                jobs.iter().map(|j| &j.aoi).collect::<Vec<_>>()
            );
            std::fs::remove_file(path).unwrap();
        }
    }
    #[test]
    fn simplified_layer_round_trips_geometry_area_and_reduction_metadata() {
        use super::super::{compact, preview::Geometry, worker};
        let mut record = preview("forest", 43, false).record(0);
        let mut ring: Vec<_> = (0..500)
            .map(|i| {
                let angle = std::f64::consts::TAU * i as f64 / 500.0;
                [angle.cos(), angle.sin()]
            })
            .collect();
        ring.push(ring[0]);
        let polygons = vec![vec![ring]];
        record.area_km2 = compact::area(&polygons);
        record.anchor_lon = 0.0;
        record.anchor_lat = 0.0;
        record.geometry = Geometry::MultiPolygon(polygons);
        let mut original = Preview::empty(1, 43);
        original.push_record(record).unwrap();
        let original = Arc::new(original.finish());
        let rx = worker::compact(Arc::clone(&original), egui::Context::default());
        let reduced = loop {
            match rx.recv_timeout(std::time::Duration::from_secs(10)).unwrap() {
                worker::Event::Finished(result) => break Arc::new(result.unwrap()),
                worker::Event::PresetFinished(_) => unreachable!(),
                worker::Event::Progress(_) => (),
            }
        };
        assert_eq!(original.edge_count, 500); // Replacement preserves the old snapshot.
        assert_eq!(reduced.max_edges, 100);
        assert_eq!(reduced.original_edges, 500);
        assert_eq!(original.aois[0].key(), reduced.aois[0].key());
        assert_ne!(
            original.aois[0].geometry_hash,
            reduced.aois[0].geometry_hash
        );
        assert!(reduced.gpu_geometry().segments.len() < original.gpu_geometry().segments.len());
        for extension in ["json", "json.gz"] {
            let path = std::env::temp_dir().join(format!(
                "leo-aoi-simplified-{}.{}",
                std::process::id(),
                extension
            ));
            save(&path, &[Arc::clone(&reduced)]).unwrap();
            let loaded = load(&path, |_| {}).unwrap();
            let loaded = loaded[1].as_ref().unwrap();
            assert_eq!(loaded.edge_count, reduced.edge_count);
            assert_eq!(loaded.original_edges, reduced.original_edges);
            assert_eq!(loaded.max_area_change, reduced.max_area_change);
            assert_eq!(loaded.aois, reduced.aois);
            assert_eq!(
                serde_json::to_value(loaded.record(0)).unwrap(),
                serde_json::to_value(reduced.record(0)).unwrap()
            );
            let headless = leo_capture::dataset::decode(&std::fs::read(&path).unwrap()).unwrap();
            assert_eq!(headless[0].0, reduced.aois[0]);
            assert_eq!(
                headless[0].1.polygons,
                reduced.capture_regions()[0].polygons
            );
            std::fs::remove_file(path).unwrap();
        }
    }
    fn frame(
        panel: &mut AoiGenerator,
        ctx: &egui::Context,
        events: Vec<egui::Event>,
    ) -> egui::FullOutput {
        ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(900.0, 700.0),
                )),
                events,
                ..Default::default()
            },
            |ui| panel.storage_ui(ui),
        )
    }
    fn click(panel: &mut AoiGenerator, ctx: &egui::Context, label: &str) {
        for _ in 0..2 {
            let _ = frame(panel, ctx, vec![]);
        }
        let output = frame(panel, ctx, vec![]);
        let pos = output
            .shapes
            .iter()
            .find_map(|s| match &s.shape {
                egui::Shape::Text(t) if t.galley.job.text == label => {
                    Some(t.pos + t.galley.size() * 0.5)
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("Missing button: {label}"));
        for pressed in [true, false] {
            let _ = frame(
                panel,
                ctx,
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
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while panel.storage.busy() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
            // Completion also works when the EO sidebar has been closed.
            panel.poll(ctx);
        }
        assert!(!panel.storage.busy(), "AOI file operation timed out");
    }
    #[test]
    fn save_load_buttons_include_hidden_layers_and_bad_load_preserves_current_aois() {
        let path =
            std::env::temp_dir().join(format!("leo-aoi-buttons-{}.json.gz", std::process::id()));
        let mut panel = AoiGenerator::default();
        panel.storage.path = path.display().to_string();
        panel.preview = Some(preview("gdp", 42, true));
        panel.sources[0].preview = Some(preview("forest", 43, false));
        panel.sources[0].visible = false;
        let expected = panel.sources[0].preview.as_ref().unwrap().aois.clone();
        let ctx = egui::Context::default();
        click(&mut panel, &ctx, "Save AOIs");
        assert!(panel.storage.error.is_none(), "{:?}", panel.storage.error);
        assert!(panel
            .storage
            .message
            .as_ref()
            .unwrap()
            .contains("Saved 2 AOIs"));
        panel.preview = None;
        panel.sources[0].preview = None;
        panel.sources[2].preview = Some(preview("coast", 99, false));
        click(&mut panel, &ctx, "Load AOIs");
        assert!(panel.storage.error.is_none(), "{:?}", panel.storage.error);
        assert!(panel.preview.is_some());
        assert_eq!(panel.sources[0].preview.as_ref().unwrap().aois, expected);
        assert!(panel.sources[2].preview.is_none()); // Replace all, don't leave stale AOIs.
        let prior = panel.preview.as_ref().unwrap().clone();
        std::fs::write(&path, b"invalid file").unwrap();
        click(&mut panel, &ctx, "Load AOIs");
        assert!(panel.storage.error.is_some());
        assert!(Arc::ptr_eq(panel.preview.as_ref().unwrap(), &prior));
        assert_eq!(panel.sources[0].preview.as_ref().unwrap().aois, expected);
        std::fs::remove_file(path).unwrap();
    }
}

fn load(path: &Path, mut progress: impl FnMut(String)) -> Result<Layers, String> {
    let mut file = BufReader::new(std::fs::File::open(path).map_err(|e| e.to_string())?);
    let compressed = file
        .fill_buf()
        .map_err(|e| e.to_string())?
        .starts_with(&[0x1f, 0x8b]);
    let reader: Box<dyn Read> = if compressed {
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut reader = reader.take(MAX_JSON_BYTES + 1);
    let doc: Document =
        serde_json::from_reader(&mut reader).map_err(|e| format!("Invalid AOI file: {e}"))?;
    if reader.limit() == 0 {
        return Err("AOIs exceed the 256 MB decoded limit.".into());
    }
    if doc.schema_version != 1
        || doc.aois.is_empty()
        || doc.aois.len() != doc.config.count
        || doc.aois.len() > 200_000
    {
        return Err("Invalid AOI version or count (1–200,000 required).".into());
    }
    let mut groups: [Vec<_>; 4] = std::array::from_fn(|_| Vec::new());
    for (index, mut aoi) in doc.aois.into_iter().enumerate() {
        if aoi.source_kind.is_empty() {
            aoi.source_kind = "gdp".into();
        }
        // Resolve legacy fallback IDs before splitting or changing record order.
        if aoi.id.is_empty() {
            aoi.id = format!("aoi-{index:08}");
        }
        let i = KINDS
            .iter()
            .position(|k| *k == aoi.source_kind)
            .ok_or_else(|| format!("Unsupported AOI source: {}", aoi.source_kind))?;
        groups[i].push(aoi);
    }
    let mut layers: Layers = std::array::from_fn(|_| None);
    for (i, aois) in groups.into_iter().enumerate() {
        if aois.is_empty() {
            continue;
        }
        progress(format!(
            "Preparing {} layer · {} AOIs…",
            KINDS[i],
            aois.len()
        ));
        let config = Config {
            seed: *doc
                .config
                .source_seeds
                .get(KINDS[i])
                .unwrap_or(&doc.config.seed),
            count: aois.len(),
            source_seeds: BTreeMap::new(),
        };
        layers[i] = Some(Arc::new(Preview::from_document(Document {
            schema_version: 1,
            config,
            aois,
        })?));
    }
    Ok(layers)
}

enum Outcome {
    Saved(usize),
    Loaded(Layers),
}
enum Event {
    Progress(String),
    Finished(Result<Outcome, String>),
}
pub(super) struct Panel {
    path: String,
    pending: Option<mpsc::Receiver<Event>>,
    message: Option<String>,
    error: Option<String>,
    progress: String,
}
impl Default for Panel {
    fn default() -> Self {
        Self {
            path: Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("assets/aois/saved-aois.json.gz")
                .display()
                .to_string(),
            pending: None,
            message: None,
            error: None,
            progress: String::new(),
        }
    }
}
impl Panel {
    pub fn busy(&self) -> bool {
        self.pending.is_some()
    }
    fn start(&mut self, layers: Option<Vec<Arc<Preview>>>, ctx: egui::Context) {
        let (tx, rx) = mpsc::channel();
        self.pending = Some(rx);
        self.message = None;
        self.error = None;
        self.progress = if layers.is_some() {
            "Saving all loaded AOI layers…"
        } else {
            "Reading AOI geometry…"
        }
        .into();
        let path = self.path.clone();
        std::thread::spawn(move || {
            let result = if let Some(layers) = layers {
                save(Path::new(&path), &layers).map(Outcome::Saved)
            } else {
                load(Path::new(&path), |p| {
                    let _ = tx.send(Event::Progress(p));
                    ctx.request_repaint();
                })
                .map(Outcome::Loaded)
            };
            let _ = tx.send(Event::Finished(result));
            ctx.request_repaint();
        });
    }
}
impl AoiGenerator {
    fn install_aois(&mut self, mut layers: Layers) {
        self.error = None;
        self.preview = layers[0].take();
        self.preview_name = "Loaded AOIs";
        self.visible = self.preview.is_some();
        for (source, layer) in self.sources.iter_mut().zip(layers.into_iter().skip(1)) {
            source.preview = layer;
            source.visible = source.preview.is_some();
        }
        // The job panel checks identity/geometry against these new layers on its
        // next UI call. Existing assignments and running capture sessions persist.
    }
    pub(super) fn poll_storage(&mut self) {
        while let Some(rx) = &self.storage.pending {
            match rx.try_recv() {
                Ok(Event::Progress(p)) => self.storage.progress = p,
                Ok(Event::Finished(result)) => {
                    self.storage.pending = None;
                    match result {
                        Ok(Outcome::Saved(count)) => {
                            self.storage.message =
                                Some(format!("Saved {count} AOIs to {}", self.storage.path))
                        }
                        Ok(Outcome::Loaded(layers)) => {
                            let count: usize = layers.iter().flatten().map(|p| p.count).sum();
                            self.install_aois(layers);
                            self.storage.message = Some(format!(
                                "Loaded {count} AOIs. You can now load their matching workload."
                            ));
                        }
                        Err(e) => self.storage.error = Some(e),
                    }
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.storage.pending = None;
                    self.storage.error =
                        Some("AOI file worker stopped. Current layers were preserved.".into());
                }
            }
        }
    }
    pub(super) fn storage_ui(&mut self, ui: &mut egui::Ui) {
        self.poll_storage();
        ui.push_id("aoi-storage", |ui| {
            let busy = self.storage.busy() || self.pending.is_some();
            ui.add_enabled_ui(!busy, |ui| {
                ui.add(egui::TextEdit::singleline(&mut self.storage.path).desired_width(f32::INFINITY));
                ui.horizontal_wrapped(|ui| {
                    let has_aois = self.preview.is_some() || self.sources.iter().any(|s| s.preview.is_some());
                    if ui.add_enabled(has_aois, egui::Button::new("Save AOIs")).clicked() {
                        let layers = self.preview.iter().chain(self.sources.iter().filter_map(|s| s.preview.as_ref())).cloned().collect();
                        self.storage.start(Some(layers), ui.ctx().clone());
                    }
                    if ui.button("Load AOIs").on_hover_text("Replace all AOI layers with the file's layers. Workloads and capture sessions are separate.").clicked() {
                        self.storage.start(None, ui.ctx().clone());
                    }
                });
            });
            ui.small("Saves all loaded layers, including hidden AOIs, with exact boundaries and IDs. Use .json.gz for compressed JSON or .json. Save job settings separately with Save workload. Loading replaces the AOI layers after the entire file validates.");
            if self.storage.busy() {
                ui.horizontal_wrapped(|ui| {ui.spinner(); ui.label(&self.storage.progress);});
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
            }
            if let Some(message) = &self.storage.message { ui.small(message); }
            if let Some(error) = &self.storage.error { ui.colored_label(egui::Color32::LIGHT_RED, error); }
        });
    }
}
