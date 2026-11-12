//! Frozen AOI geometry, independent of Python and local demand data.
use super::preview::{Document, Preview};
use serde::Deserialize;
use std::{
    io::Read,
    sync::{Arc, Mutex, Weak},
};

pub(super) const DEFAULT_JOBS: &str = "Default AOIs (bundled)";
pub(super) const LEGACY_JOBS: &str = "Legacy rectangles (1,000 km²)";
const SETTINGS: &str = include_str!("../../assets/aois/default-jobs-oriented.settings.json");
const DATA: &[u8] = include_bytes!("../../assets/aois/default-jobs-oriented.json.gz");
const LEGACY_SETTINGS: &str = include_str!("../../assets/aois/default-jobs.settings.json");
const LEGACY_DATA: &[u8] = include_bytes!("../../assets/aois/default-jobs.json.gz");

pub(super) struct Layer {
    pub name: &'static str,
    data: &'static [u8],
    // Do not retain an unused preset (or another copy of its GPU buffers).
    cache: Mutex<Weak<Preview>>,
}
impl Layer {
    const fn new(name: &'static str, data: &'static [u8]) -> Self {
        Self {
            name,
            data,
            cache: Mutex::new(Weak::new()),
        }
    }
    pub fn cached(&self) -> Option<Arc<Preview>> {
        self.cache.lock().unwrap().upgrade()
    }
    pub fn remember(&self, preview: Preview) -> Arc<Preview> {
        let preview = Arc::new(preview);
        *self.cache.lock().unwrap() = Arc::downgrade(&preview);
        preview
    }
    pub fn document(&self, progress: impl FnMut(&str)) -> Result<Document, String> {
        decode(self.data, progress).map_err(|e| format!("{}: {e}", self.name))
    }
}

pub(super) fn layers(legacy: bool) -> &'static [Layer] {
    static DEFAULT: [Layer; 4] = [
        Layer::new("GDP rectangles", DATA),
        Layer::new(
            "Forests",
            include_bytes!("../../assets/aois/default-forest.json.gz"),
        ),
        Layer::new(
            "Rivers",
            include_bytes!("../../assets/aois/default-river.json.gz"),
        ),
        Layer::new(
            "Coasts",
            include_bytes!("../../assets/aois/default-coast.json.gz"),
        ),
    ];
    static LEGACY: [Layer; 1] = [Layer::new("Legacy GDP rectangles", LEGACY_DATA)];
    if legacy {
        &LEGACY
    } else {
        &DEFAULT
    }
}

fn decode(data: &[u8], mut progress: impl FnMut(&str)) -> Result<Document, String> {
    progress("Decompressing saved geometry…");
    let mut json = Vec::new();
    flate2::read::GzDecoder::new(data)
        .read_to_end(&mut json)
        .map_err(|e| format!("Could not decompress AOI preset: {e}"))?;
    progress("Reading saved geometry…");
    // Release the decoded JSON before allocating render geometry.
    serde_json::from_slice(&json).map_err(|e| format!("Invalid AOI preset: {e}"))
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn load_native(
    legacy: bool,
    mut progress: impl FnMut(String),
) -> Result<Vec<Arc<Preview>>, String> {
    let bundled = layers(legacy);
    let mut loaded = Vec::with_capacity(bundled.len());
    for (index, layer) in bundled.iter().enumerate() {
        let label = format!("Layer {}/{} · {}", index + 1, bundled.len(), layer.name);
        if let Some(cached) = layer.cached() {
            progress(format!("{label} · Reusing loaded geometry"));
            loaded.push(cached);
            continue;
        }
        leo_capture::resources::check_memory()?;
        let document = layer.document(|stage| progress(format!("{label} · {stage}")))?;
        let workers = leo_capture::parallel::worker_count(document.config.count);
        let mut last = std::time::Instant::now();
        let mut previous_stage = "";
        let preview = Preview::from_document_parallel(document, workers, |stage, done, total| {
            if stage != previous_stage || done == total || last.elapsed().as_millis() >= 100 {
                progress(format!(
                    "{label} · {stage}: {done} / {total} · {workers} CPU workers"
                ));
                last = std::time::Instant::now();
                previous_stage = stage;
            }
        })?;
        loaded.push(layer.remember(preview));
    }
    Ok(loaded)
}

#[derive(Deserialize)]
pub(super) struct Settings {
    pub mean_area_km2: f64,
    pub size_ratio: f64,
    pub generation: Generation,
}
#[derive(Deserialize)]
pub(super) struct Generation {
    pub count: usize,
    pub shape: String,
    pub seed: u32,
    pub hotspot_count: usize,
    pub hotspot_spread_km: f64,
    pub hotspot_fraction: f64,
    #[serde(default)]
    pub random_rotation: bool,
    #[serde(default = "default_aspect")]
    pub max_aspect: f64,
}
fn default_aspect() -> f64 {
    2.0
}
pub(super) fn default_settings() -> Settings {
    settings(false)
}
pub(super) fn settings(legacy: bool) -> Settings {
    serde_json::from_str(if legacy { LEGACY_SETTINGS } else { SETTINGS })
        .expect("bundled AOI settings must be valid")
}
#[cfg(test)]
pub(super) fn load_default_jobs() -> Result<Preview, String> {
    load(false)
}
#[cfg(test)]
fn load(legacy: bool) -> Result<Preview, String> {
    Preview::from_document(decode(if legacy { LEGACY_DATA } else { DATA }, |_| {})?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preset_cache_reuses_buffers_but_does_not_retain_unused_geometry() {
        let layer = Layer::new("test", &[]);
        assert!(layer.cached().is_none());
        let preview = layer.remember(Preview::empty(0, 42).finish());
        let cached = layer.cached().unwrap();
        assert!(Arc::ptr_eq(&preview, &cached));
        assert!(Arc::ptr_eq(&preview.gpu_geometry(), &cached.gpu_geometry()));
        drop(preview);
        assert!(layer.cached().is_some());
        drop(cached);
        assert!(layer.cached().is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn frozen_environment_matches_default_settings_and_generated_geometry_identity() {
        let manifest: serde_json::Value = serde_json::from_str(include_str!(
            "../../assets/aois/default-environment.manifest.json"
        ))
        .unwrap();
        for (index, kind) in ["forest", "river", "coast"].into_iter().enumerate() {
            let layer = &layers(false)[index + 1];
            let expected = &manifest["layers"][index];
            let (settings, _) = crate::aoi::environment::Source::new(kind, layer.name).request();
            let crate::aoi::generate::Settings::Environment(settings) = settings else {
                unreachable!()
            };
            assert_eq!(
                expected["settings"],
                serde_json::to_value(settings).unwrap()
            );
            assert_eq!(
                expected["file_hash"].as_u64(),
                Some(crate::jobs::model::hash(layer.data))
            );
            let document = layer.document(|_| {}).unwrap();
            assert_eq!(document.config.count, 1000);
            let preview = Preview::from_document_parallel(
                document,
                leo_capture::parallel::worker_count(1000),
                |_, _, _| {},
            )
            .unwrap();
            assert!(preview.aois.iter().all(|a| a.source == kind));
            assert!(preview.max_edges <= crate::aoi::compact::EDGE_LIMIT);
            assert_eq!(
                expected["edge_count"].as_u64(),
                Some(preview.edge_count as u64)
            );
            assert_eq!(
                expected["geometry_identity_hash"].as_u64(),
                Some(crate::jobs::model::hash(
                    &serde_json::to_vec(&preview.aois).unwrap()
                ))
            );
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "measures complete cold and cached preset loads"]
    fn benchmark_preset_loading() {
        use std::time::Instant;
        let start = Instant::now();
        let gdp = load(false).unwrap();
        eprintln!(
            "Sequential saved GDP: {:.3}s",
            start.elapsed().as_secs_f64()
        );
        drop(gdp);
        let start = Instant::now();
        let loaded = load_native(false, |message| {
            if message.contains("Building display") {
                eprintln!("{:.3}s · {message}", start.elapsed().as_secs_f64());
            }
        })
        .unwrap();
        let cold = start.elapsed();
        assert_eq!(loaded.iter().map(|p| p.count).sum::<usize>(), 53_000);
        let start = Instant::now();
        let cached = load_native(false, |_| {}).unwrap();
        let warm = start.elapsed();
        for (a, b) in loaded.iter().zip(&cached) {
            assert!(Arc::ptr_eq(a, b));
        }
        eprintln!(
            "Full preset: cold {:.3}s, cached {:.3}ms · {} workers",
            cold.as_secs_f64(),
            warm.as_secs_f64() * 1000.0,
            leo_capture::parallel::worker_count(50_000)
        );
    }
    /// Explicit maintainer command: freeze the default environmental sample.
    /// Normal loading never calls the generator or reads geographic source data.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "regenerates bundled preset assets; run explicitly after changing defaults"]
    fn rebuild_environment_presets() {
        use crate::aoi::{
            data,
            environment::Source,
            generate::Job,
            preview::{Config, Document},
        };
        use std::{io::Write, time::Instant};
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/aois");
        let workers = leo_capture::parallel::worker_count(1000);
        let start = Instant::now();
        let gdp = load(false).unwrap();
        eprintln!("Old GDP load: {:.3}s", start.elapsed().as_secs_f64());
        drop(gdp);
        let mut generation_seconds = 0.0;
        let mut saved_seconds = 0.0;
        let mut manifest = Vec::new();
        for (kind, name) in [
            ("forest", "Forests"),
            ("river", "Rivers"),
            ("coast", "Coasts"),
        ] {
            let start = Instant::now();
            let (settings, input) = Source::new(kind, name).request();
            let config = match &settings {
                crate::aoi::generate::Settings::Environment(c) => serde_json::to_value(c).unwrap(),
                _ => unreachable!(),
            };
            let dataset = data::load(&input, kind).unwrap();
            let preview = Job::new(settings, Some(dataset))
                .unwrap()
                .run_parallel(workers, |_, _, _| {})
                .unwrap();
            let generated = start.elapsed().as_secs_f64();
            generation_seconds += generated;
            assert_eq!(preview.count, 1000);
            assert!(preview.max_edges <= crate::aoi::compact::EDGE_LIMIT);
            let doc = Document {
                schema_version: 1,
                config: Config {
                    count: preview.count,
                    seed: preview.seed,
                    source_seeds: Default::default(),
                },
                aois: (0..preview.count).map(|i| preview.record(i)).collect(),
            };
            let json = serde_json::to_vec(&doc).unwrap();
            let mut gzip =
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            gzip.write_all(&json).unwrap();
            let bytes = gzip.finish().unwrap();
            let file = format!("default-{kind}.json.gz");
            std::fs::write(root.join(&file), &bytes).unwrap();
            let start = Instant::now();
            let mut decoded = Vec::new();
            flate2::read::GzDecoder::new(bytes.as_slice())
                .read_to_end(&mut decoded)
                .unwrap();
            let restored = Preview::from_document_parallel(
                serde_json::from_slice(&decoded).unwrap(),
                workers,
                |_, _, _| {},
            )
            .unwrap();
            let loaded = start.elapsed().as_secs_f64();
            saved_seconds += loaded;
            assert_eq!(preview.aois, restored.aois);
            assert_eq!(
                json,
                serde_json::to_vec(&Document {
                    schema_version: 1,
                    config: Config {
                        count: restored.count,
                        seed: restored.seed,
                        source_seeds: Default::default()
                    },
                    aois: (0..restored.count).map(|i| restored.record(i)).collect(),
                })
                .unwrap()
            );
            eprintln!(
                "{name}: generate {generated:.3}s, load frozen {loaded:.3}s, {} bytes",
                bytes.len()
            );
            manifest.push(serde_json::json!({
                "file": file, "settings": config, "count": preview.count,
                "edge_count": preview.edge_count, "max_edges": preview.max_edges,
                "geometry_identity_hash": crate::jobs::model::hash(&serde_json::to_vec(&preview.aois).unwrap()),
                "source": format!("{kind}.aoi-data.json.gz"),
                "source_hash": crate::jobs::model::hash(&std::fs::read(root.join(format!("{kind}.aoi-data.json.gz"))).unwrap()),
                "file_hash": crate::jobs::model::hash(&bytes)
            }));
        }
        std::fs::write(root.join("default-environment.manifest.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1, "generator": "src/aoi/generate.rs + compact.rs (100-edge cap)",
            "rebuild": "cargo test --release rebuild_environment_presets -- --ignored --nocapture",
            "hash_algorithm": "leo_capture::workload::hash (FNV-1a 64)", "layers": manifest
        })).unwrap()).unwrap();
        eprintln!("Environmental total: generate {generation_seconds:.3}s, load frozen {saved_seconds:.3}s ({workers} workers)");
    }
    #[test]
    fn legacy_preset_remains_available_for_existing_workloads() {
        let old = load(true).unwrap();
        assert_eq!(old.count, 50_000);
        assert!((old.mean_area - 994.343171430485).abs() < 1e-8);
        assert_eq!(settings(true).generation.shape, "rectangle");
        assert_eq!(default_settings().mean_area_km2, 100.0);
    }
}
