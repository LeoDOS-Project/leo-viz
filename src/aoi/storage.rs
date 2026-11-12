//! One native preset file: exact AOI geometry and optional assigned workload.
use super::{
    preview::{Config, Document, Preview, Record},
    AoiGenerator,
};
use eframe::egui;
use leo_capture::storage::Document as Workload;
use leo_sim::acquisition::CaptureProfile;
use serde::{ser::SerializeSeq, Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
    sync::{mpsc, Arc},
};

const MAX_JSON_BYTES: u64 = 256 * 1024 * 1024;
const KINDS: [&str; 4] = ["gdp", "forest", "river", "coast"];
type Layers = [Option<Arc<Preview>>; 4];

struct Snapshot {
    layers: Vec<Arc<Preview>>,
    workload: Option<Workload>,
}
struct Loaded {
    layers: Layers,
    workload: Option<Workload>,
}
#[derive(Deserialize)]
struct SavedDocument {
    schema_version: u32,
    #[serde(default)]
    preset_version: Option<u32>,
    config: Config,
    aois: Vec<Record>,
    #[serde(default)]
    workload: Option<Workload>,
}

fn validate_workload(workload: &Option<Workload>, layers: &[Arc<Preview>]) -> Result<(), String> {
    if let Some(workload) = workload {
        workload.validate()?;
        if !crate::jobs::layers_match(&workload.jobs, layers) {
            return Err("Saved jobs do not match the AOI geometry. No changes were made.".into());
        }
    }
    Ok(())
}

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
                "Preset exceeds the 256 MB decoded limit",
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

fn save(
    path: &Path,
    layers: &[Arc<Preview>],
    workload: Option<Workload>,
) -> Result<(usize, usize), String> {
    validate_workload(&workload, layers)?;
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
        preset_version: u32,
        config: Config,
        aois: Records<'a>,
        #[serde(skip_serializing_if = "Option::is_none")]
        workload: Option<&'a Workload>,
    }
    let saved = Saved {
        schema_version: 1,
        preset_version: 1,
        config: Config {
            seed: layers[0].seed,
            count,
            source_seeds,
        },
        aois: Records(layers),
        workload: workload.as_ref(),
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
    result.map(|()| (count, workload.as_ref().map_or(0, |w| w.jobs.len())))
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
            assert_eq!(save(&path, &originals, None).unwrap(), (4, 0));
            let restored = load(&path, std::array::from_fn(|_| None), |_| {})
                .unwrap()
                .layers;
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
            save(&path, &[Arc::clone(&reduced)], None).unwrap();
            let loaded = load(&path, std::array::from_fn(|_| None), |_| {})
                .unwrap()
                .layers;
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
        capture: &mut CaptureProfile,
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
            |ui| panel.storage_ui(ui, capture),
        )
    }
    fn click(
        panel: &mut AoiGenerator,
        ctx: &egui::Context,
        capture: &mut CaptureProfile,
        label: &str,
    ) {
        for _ in 0..2 {
            let _ = frame(panel, ctx, capture, vec![]);
        }
        let output = frame(panel, ctx, capture, vec![]);
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
                capture,
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
        assert!(!panel.storage.busy(), "Preset file operation timed out");
        let _ = frame(panel, ctx, capture, vec![]);
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
        let mut capture = CaptureProfile::default();
        click(&mut panel, &ctx, &mut capture, "Save");
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
        click(&mut panel, &ctx, &mut capture, "Load");
        assert!(panel.storage.error.is_none(), "{:?}", panel.storage.error);
        assert!(panel.preview.is_some());
        assert_eq!(panel.sources[0].preview.as_ref().unwrap().aois, expected);
        assert!(panel.sources[2].preview.is_none()); // Replace all, don't leave stale AOIs.
        let prior = panel.preview.as_ref().unwrap().clone();
        std::fs::write(&path, b"invalid file").unwrap();
        click(&mut panel, &ctx, &mut capture, "Load");
        assert!(panel.storage.error.is_some());
        assert!(Arc::ptr_eq(panel.preview.as_ref().unwrap(), &prior));
        assert_eq!(panel.sources[0].preview.as_ref().unwrap().aois, expected);
        std::fs::remove_file(path).unwrap();
    }

    fn workload(layers: &[Arc<Preview>]) -> Workload {
        let generator = leo_capture::workload::Generator::default();
        let mut jobs = generator
            .generate(layers.iter().flat_map(|p| p.aois.clone()))
            .unwrap();
        jobs[0].acquisition.priority = 987;
        jobs[0].acquisition.solar_time_hours = 10.25;
        Workload {
            schema_version: 3,
            generator,
            jobs,
            capture: CaptureProfile {
                channels: 8,
                resolution_m: 1.5,
                bits_per_sample: 16,
                compression_ratio: 2.5,
            },
        }
    }

    #[test]
    fn unified_buttons_restore_jobs_and_image_settings_and_aoi_only_clears_jobs() {
        let path =
            std::env::temp_dir().join(format!("leo-preset-buttons-{}.json.gz", std::process::id()));
        let mut panel = AoiGenerator::default();
        panel.storage.path = path.display().to_string();
        panel.preview = Some(preview("gdp", 42, true));
        panel.sources[0].preview = Some(preview("forest", 43, false));
        panel.sources[0].visible = false;
        let layers = vec![
            panel.preview.clone().unwrap(),
            panel.sources[0].preview.clone().unwrap(),
        ];
        let expected = workload(&layers);
        let expected_jobs = expected.jobs.clone();
        let expected_generator = expected.generator.clone();
        let expected_capture = expected.capture;
        panel.jobs.restore_workload(Some(expected));
        let mut capture = CaptureProfile::default();
        let ctx = egui::Context::default();
        click(&mut panel, &ctx, &mut capture, "Save");
        assert!(panel.storage.error.is_none(), "{:?}", panel.storage.error);
        assert!(panel.storage.selected_saved);
        assert!(panel
            .storage
            .message
            .as_ref()
            .unwrap()
            .contains("2 AOIs and 2 assigned jobs"));

        panel.preview = None;
        panel.sources[0].preview = None;
        panel.sources[2].preview = Some(preview("coast", 99, false));
        panel.jobs.restore_workload(None);
        capture = CaptureProfile::default();
        click(&mut panel, &ctx, &mut capture, "Load");
        assert!(panel.storage.error.is_none(), "{:?}", panel.storage.error);
        assert!(panel.sources[2].preview.is_none());
        let restored_layers = vec![
            panel.preview.clone().unwrap(),
            panel.sources[0].preview.clone().unwrap(),
        ];
        let restored = panel
            .jobs
            .saved_workload(&restored_layers, capture)
            .unwrap()
            .unwrap();
        assert_eq!(restored.jobs, expected_jobs);
        assert_eq!(restored.generator, expected_generator);
        assert_eq!(capture, expected_capture);
        assert_eq!(restored_layers[0].aois, layers[0].aois);
        assert_eq!(restored_layers[1].aois, layers[1].aois);

        // Corrupt only the job reference. Neither geometry, assignments nor Data
        // settings may change when validation fails late in a load.
        let mut json = Vec::new();
        flate2::read::GzDecoder::new(std::fs::File::open(&path).unwrap())
            .read_to_end(&mut json)
            .unwrap();
        let mut invalid: serde_json::Value = serde_json::from_slice(&json).unwrap();
        invalid["workload"]["jobs"][0]["aoi"]["geometry_hash"] = 0.into();
        std::fs::write(&path, serde_json::to_vec(&invalid).unwrap()).unwrap();
        click(&mut panel, &ctx, &mut capture, "Load");
        assert!(panel
            .storage
            .error
            .as_ref()
            .unwrap()
            .contains("do not match"));
        assert!(Arc::ptr_eq(
            panel.preview.as_ref().unwrap(),
            &restored_layers[0]
        ));
        assert_eq!(
            panel
                .jobs
                .saved_workload(&restored_layers, capture)
                .unwrap()
                .unwrap()
                .jobs,
            expected_jobs
        );
        assert_eq!(capture, expected_capture);

        // Also accept old AOI-only JSON, without a preset version or workload.
        invalid.as_object_mut().unwrap().remove("workload");
        invalid.as_object_mut().unwrap().remove("preset_version");
        std::fs::write(&path, serde_json::to_vec(&invalid).unwrap()).unwrap();
        click(&mut panel, &ctx, &mut capture, "Load");
        assert!(panel.storage.error.is_none(), "{:?}", panel.storage.error);
        let current = vec![
            panel.preview.clone().unwrap(),
            panel.sources[0].preview.clone().unwrap(),
        ];
        assert!(panel
            .jobs
            .saved_workload(&current, capture)
            .unwrap()
            .is_none());
        assert_eq!(capture, expected_capture); // AOI-only files have no Data settings.
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn combined_files_and_legacy_workloads_validate_jobs_before_replacement() {
        let layers = vec![preview("gdp", 42, true), preview("forest", 43, false)];
        let expected = workload(&layers);
        let current = [Some(layers[0].clone()), Some(layers[1].clone()), None, None];
        for extension in ["json", "json.gz"] {
            let path = std::env::temp_dir().join(format!(
                "leo-preset-roundtrip-{}.{}",
                std::process::id(),
                extension
            ));
            assert_eq!(
                save(&path, &layers, Some(workload(&layers))).unwrap(),
                (2, 2)
            );
            let restored = load(&path, [None, None, None, None], |_| {}).unwrap();
            assert_eq!(restored.workload.unwrap().jobs, expected.jobs);
            let before = std::fs::read(&path).unwrap();
            let mut wrong = workload(&layers);
            wrong.jobs[0].aoi.geometry_hash = 0;
            assert!(save(&path, &layers, Some(wrong)).is_err());
            assert_eq!(std::fs::read(&path).unwrap(), before);
            let legacy = if extension == "json.gz" {
                expected.encode().unwrap()
            } else {
                serde_json::to_vec(&expected).unwrap()
            };
            std::fs::write(&path, legacy).unwrap();
            assert!(load(&path, [None, None, None, None], |_| {}).is_err());
            let restored = load(&path, current.clone(), |_| {}).unwrap();
            assert!(Arc::ptr_eq(
                restored.layers[0].as_ref().unwrap(),
                &layers[0]
            ));
            assert_eq!(restored.workload.unwrap().jobs, expected.jobs);
            std::fs::remove_file(path).unwrap();
        }
        let mut jobs = crate::jobs::Panel::default();
        jobs.restore_workload(Some(expected));
        assert!(jobs
            .saved_workload(&[layers[0].clone()], CaptureProfile::default())
            .is_err());
    }
}

fn load(path: &Path, current: Layers, mut progress: impl FnMut(String)) -> Result<Loaded, String> {
    progress("Reading saved AOIs and jobs…".into());
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
    let mut json = Vec::new();
    reader
        .take(MAX_JSON_BYTES + 1)
        .read_to_end(&mut json)
        .map_err(|e| e.to_string())?;
    if json.len() as u64 > MAX_JSON_BYTES {
        return Err("Preset exceeds the 256 MB decoded limit.".into());
    }
    // Probe without allocating a JSON value tree. Older workload-only files can
    // still be imported through the same Load button, against the current AOIs.
    #[derive(Deserialize)]
    struct Kind {
        aois: Option<serde::de::IgnoredAny>,
    }
    let kind: Kind = serde_json::from_slice(&json).map_err(|e| format!("Invalid preset: {e}"))?;
    if kind.aois.is_none() {
        let workload = Some(Workload::from_json(&json)?);
        validate_workload(
            &workload,
            &current.iter().flatten().cloned().collect::<Vec<_>>(),
        )?;
        return Ok(Loaded {
            layers: current,
            workload,
        });
    }
    let doc: SavedDocument =
        serde_json::from_slice(&json).map_err(|e| format!("Invalid preset: {e}"))?;
    drop(json);
    if doc.schema_version != 1
        || doc.preset_version.is_some_and(|v| v != 1)
        || doc.aois.is_empty()
        || doc.aois.len() != doc.config.count
        || doc.aois.len() > 200_000
    {
        return Err("Invalid preset version or AOI count (1–200,000 required).".into());
    }
    let workload = doc.workload.map(Workload::into_current).transpose()?;
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
        let config = Config {
            seed: *doc
                .config
                .source_seeds
                .get(KINDS[i])
                .unwrap_or(&doc.config.seed),
            count: aois.len(),
            source_seeds: BTreeMap::new(),
        };
        let workers = leo_capture::parallel::worker_count(aois.len());
        let mut last = std::time::Instant::now();
        layers[i] = Some(Arc::new(Preview::from_document_parallel(
            Document {
                schema_version: 1,
                config,
                aois,
            },
            workers,
            |stage, done, total| {
                if done == 0 || done == total || last.elapsed().as_millis() >= 100 {
                    progress(format!(
                        "{} · {stage}: {done} / {total} · {workers} CPU workers",
                        KINDS[i]
                    ));
                    last = std::time::Instant::now();
                }
            },
        )?));
    }
    progress("Validating saved job assignments…".into());
    validate_workload(
        &workload,
        &layers.iter().flatten().cloned().collect::<Vec<_>>(),
    )?;
    Ok(Loaded { layers, workload })
}

enum Outcome {
    Saved(usize, usize),
    Loaded(Loaded),
}
enum Event {
    Progress(String),
    Finished(Result<Outcome, String>),
}
pub(super) struct Panel {
    path: String,
    pub(super) selected_saved: bool,
    pending: Option<mpsc::Receiver<Event>>,
    message: Option<String>,
    error: Option<String>,
    progress: String,
}
impl Default for Panel {
    fn default() -> Self {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/presets/default-preset.json.gz");
        Self {
            selected_saved: path.is_file(),
            path: path.display().to_string(),
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
    fn start(&mut self, snapshot: Option<Snapshot>, current: Layers, ctx: egui::Context) {
        let (tx, rx) = mpsc::channel();
        self.pending = Some(rx);
        self.message = None;
        self.error = None;
        self.progress = if snapshot.is_some() {
            "Saving AOIs and assigned jobs…"
        } else {
            "Reading saved preset…"
        }
        .into();
        let path = self.path.clone();
        std::thread::spawn(move || {
            leo_capture::resources::background_thread();
            let result = leo_capture::resources::check_memory().and_then(|()| {
                if let Some(snapshot) = snapshot {
                    save(Path::new(&path), &snapshot.layers, snapshot.workload)
                        .map(|(aois, jobs)| Outcome::Saved(aois, jobs))
                } else {
                    load(Path::new(&path), current, |p| {
                        let _ = tx.send(Event::Progress(p));
                        ctx.request_repaint();
                    })
                    .map(Outcome::Loaded)
                }
            });
            let _ = tx.send(Event::Finished(result));
            ctx.request_repaint();
        });
    }
}
impl AoiGenerator {
    fn install_aois(&mut self, mut layers: Layers) {
        self.error = None;
        self.preview = layers[0].take();
        self.preview_name = "Loaded preset";
        self.visible = self.preview.is_some();
        for (source, layer) in self.sources.iter_mut().zip(layers.into_iter().skip(1)) {
            source.preview = layer;
            source.visible = source.preview.is_some();
        }
    }
    pub(super) fn poll_storage(&mut self) {
        while let Some(rx) = &self.storage.pending {
            match rx.try_recv() {
                Ok(Event::Progress(p)) => self.storage.progress = p,
                Ok(Event::Finished(result)) => {
                    self.storage.pending = None;
                    match result {
                        Ok(Outcome::Saved(aois, jobs)) => {
                            self.storage.selected_saved = true;
                            self.storage.message = Some(format!(
                                "Saved {aois} AOIs and {jobs} assigned jobs to {}",
                                self.storage.path
                            ));
                        }
                        Ok(Outcome::Loaded(loaded)) => {
                            let aois: usize = loaded.layers.iter().flatten().map(|p| p.count).sum();
                            let jobs = loaded.workload.as_ref().map_or(0, |w| w.jobs.len());
                            self.install_aois(loaded.layers);
                            self.jobs.restore_workload(loaded.workload);
                            self.storage.message =
                                Some(format!("Loaded {aois} AOIs and {jobs} assigned jobs."));
                        }
                        Err(e) => self.storage.error = Some(e),
                    }
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.storage.pending = None;
                    self.storage.error = Some(
                        "Preset file worker stopped. Current AOIs and jobs were preserved.".into(),
                    );
                }
            }
        }
    }
    pub(super) fn storage_ui(&mut self, ui: &mut egui::Ui, capture: &mut CaptureProfile) {
        self.poll_storage();
        self.jobs.apply_restored_capture(capture);
        ui.push_id("aoi-storage", |ui| {
            let busy = self.busy() || self.jobs.assigning();
            ui.add_enabled_ui(!busy, |ui| {
                egui::ComboBox::from_id_salt("preset-choice")
                    .selected_text(if self.storage.selected_saved { "Saved preset" }
                        else if self.legacy_preset { super::presets::LEGACY_JOBS }
                        else { super::presets::DEFAULT_JOBS })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.storage.selected_saved, true, "Saved preset");
                        for (legacy, name) in [(false, super::presets::DEFAULT_JOBS), (true, super::presets::LEGACY_JOBS)] {
                            if ui.selectable_label(!self.storage.selected_saved && self.legacy_preset == legacy, name).clicked() {
                                self.storage.selected_saved = false;
                                self.legacy_preset = legacy;
                            }
                        }
                    });
                ui.label("Saved preset file (.json.gz or .json)");
                if ui.add(egui::TextEdit::singleline(&mut self.storage.path).desired_width(f32::INFINITY)).changed() {
                    self.storage.selected_saved = true;
                }
                ui.horizontal_wrapped(|ui| {
                    let current: Layers = std::array::from_fn(|i| if i == 0 { self.preview.clone() } else { self.sources[i - 1].preview.clone() });
                    let has_aois = current.iter().any(Option::is_some);
                    if ui.add_enabled(has_aois, egui::Button::new("Save")).on_hover_text("Save every loaded AOI layer and its assigned jobs to the file above.").clicked() {
                        let layers: Vec<_> = current.iter().flatten().cloned().collect();
                        match self.jobs.saved_workload(&layers, *capture) {
                            Ok(workload) => self.storage.start(Some(Snapshot { layers, workload }), current.clone(), ui.ctx().clone()),
                            Err(e) => { self.storage.error = Some(e); self.storage.message = None; }
                        }
                    }
                    if ui.button("Load").on_hover_text("Load the selected preset, replacing AOIs and assigned jobs together.").clicked() {
                        self.storage.message = None;
                        self.storage.error = None;
                        if self.storage.selected_saved {
                            self.storage.start(None, current, ui.ctx().clone());
                        } else {
                            self.load_default_jobs(ui.ctx());
                        }
                    }
                });
            });
            ui.small("Save includes hidden AOIs and current job assignments. Load restores exactly the saved contents; AOI-only presets clear existing jobs. Built-in samples contain AOIs only.");
            ui.collapsing("Older files", |ui| {
                ui.small("Select Saved preset and enter an existing AOI or workload file above. Workload-only files require matching AOIs already loaded.");
            });
            if self.storage.busy() {
                ui.horizontal_wrapped(|ui| {ui.spinner(); ui.label(&self.storage.progress);});
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
            }
            if let Some(message) = &self.storage.message { ui.small(message); }
            if let Some(error) = &self.storage.error { ui.colored_label(egui::Color32::LIGHT_RED, error); }
        });
    }
}
