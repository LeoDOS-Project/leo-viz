//! Portable, prepared geographic data. Raw GeoTIFF/shapefile ingestion stays offline.
use base64::{engine::general_purpose::STANDARD, Engine};
use serde::Deserialize;
use std::{io::Read, sync::Arc};

const MAX_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Clone, Default)]
pub(super) struct Input {
    pub path: String,
    pub bytes: Option<Arc<[u8]>>,
}

impl Input {
    pub fn is_default(&self) -> bool {
        self.path.trim().is_empty() && self.bytes.is_none()
    }
    pub fn ui(&mut self, ui: &mut eframe::egui::Ui) {
        ui.label("Dataset file or URL");
        if ui.text_edit_singleline(&mut self.path).changed() {
            self.bytes = None;
        }
    }

    pub fn accept_drop(&mut self, file: &eframe::egui::DroppedFile) {
        self.path = file
            .path
            .as_ref()
            .map_or_else(|| file.name.clone(), |p| p.display().to_string());
        self.bytes = file.bytes.clone();
    }
}

#[derive(Deserialize)]
struct Document {
    schema_version: u32,
    #[serde(flatten)]
    source: Source,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Source {
    Demand {
        columns: usize,
        rows: usize,
        cells: Vec<(usize, f64)>,
        eligible: Option<String>,
    },
    Forest {
        columns: usize,
        rows: usize,
        mask: String,
    },
    River {
        network: NetworkDocument,
    },
    Coast {
        network: NetworkDocument,
    },
}

#[derive(Deserialize)]
#[serde(untagged)]
enum NetworkDocument {
    Arrays(Network),
    Packed {
        points: String,
        offsets: String,
        next_index: String,
        length: String,
        ids: String,
        flow: String,
    },
}

fn unpack<const N: usize, T>(encoded: &str, read: impl Fn([u8; N]) -> T) -> Result<Vec<T>, String> {
    let bytes = STANDARD
        .decode(encoded)
        .map_err(|e| format!("Invalid geographic array: {e}"))?;
    if bytes.len() % N != 0 {
        return Err("Invalid geographic array byte length".into());
    }
    Ok(bytes
        .chunks_exact(N)
        .map(|c| read(c.try_into().unwrap()))
        .collect())
}

impl NetworkDocument {
    fn decode(self) -> Result<Network, String> {
        let network = match self {
            Self::Arrays(n) => n,
            Self::Packed {
                points,
                offsets,
                next_index,
                length,
                ids,
                flow,
            } => Network {
                points: unpack(&points, |b: [u8; 8]| {
                    [
                        f32::from_le_bytes(b[..4].try_into().unwrap()) as f64,
                        f32::from_le_bytes(b[4..].try_into().unwrap()) as f64,
                    ]
                })?,
                offsets: unpack(&offsets, u64::from_le_bytes)?
                    .into_iter()
                    .map(|n| {
                        usize::try_from(n)
                            .map_err(|_| "Geographic index exceeds platform limits".to_owned())
                    })
                    .collect::<Result<_, _>>()?,
                next_index: unpack(&next_index, i32::from_le_bytes)?,
                length: unpack(&length, f64::from_le_bytes)?,
                ids: unpack(&ids, i64::from_le_bytes)?,
                flow: unpack(&flow, f32::from_le_bytes)?
                    .into_iter()
                    .map(f64::from)
                    .collect(),
            },
        };
        network.validate()?;
        Ok(network)
    }
}

pub(super) enum Dataset {
    Demand(Demand),
    Forest(Forest),
    River(Network),
    Coast(Network),
}

pub(super) struct Demand {
    pub columns: usize,
    pub rows: usize,
    pub cells: Vec<(usize, f64)>,
    pub eligible: Option<Vec<u8>>,
}

pub(super) struct Forest {
    pub columns: usize,
    pub rows: usize,
    pub mask: Vec<u8>,
}

#[derive(Deserialize)]
pub(super) struct Network {
    pub points: Vec<[f64; 2]>,
    pub offsets: Vec<usize>,
    pub next_index: Vec<i32>,
    pub length: Vec<f64>,
    pub ids: Vec<i64>,
    pub flow: Vec<f64>,
}

pub(super) fn bit(bits: &[u8], index: usize) -> bool {
    bits[index / 8] & (1 << (index % 8)) != 0
}

fn mask(encoded: &str, count: usize) -> Result<Vec<u8>, String> {
    let bytes = STANDARD
        .decode(encoded)
        .map_err(|e| format!("Invalid dataset mask: {e}"))?;
    if bytes.len() != count.div_ceil(8) {
        return Err("Dataset mask does not match its grid".into());
    }
    Ok(bytes)
}

fn dimensions(columns: usize, rows: usize, limit: usize) -> Result<usize, String> {
    columns
        .checked_mul(rows)
        .filter(|&n| columns > 0 && rows > 0 && n <= limit)
        .ok_or_else(|| "Invalid or oversized dataset grid".into())
}

impl Dataset {
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.starts_with(b"version https://git-lfs.github.com/spec/v1") {
            return Err("Dataset is a Git LFS pointer, not geographic data. Run git lfs pull --include=\"assets/aois/**\"; web deployments must publish the resolved LFS files.".into());
        }
        if bytes.len() as u64 > MAX_BYTES {
            return Err("AOI dataset exceeds 512 MiB".into());
        }
        let mut decoded = Vec::new();
        if bytes.starts_with(&[0x1f, 0x8b]) {
            flate2::read::GzDecoder::new(bytes)
                .take(MAX_BYTES + 1)
                .read_to_end(&mut decoded)
                .map_err(|e| format!("Could not decompress AOI dataset: {e}"))?;
        } else if bytes.len() as u64 <= MAX_BYTES {
            decoded.extend_from_slice(bytes);
        }
        if decoded.len() as u64 > MAX_BYTES || decoded.is_empty() {
            return Err("AOI dataset is empty or exceeds 512 MiB".into());
        }
        let doc: Document = serde_json::from_slice(&decoded)
            .map_err(|e| format!("Invalid AOI dataset: {e}. Use a prepared .aoi-data.json.gz file; see tools/leo-aoi/README.md for the Rust exporter."))?;
        if doc.schema_version != 1 {
            return Err("Unsupported AOI dataset version".into());
        }
        Ok(match doc.source {
            Source::Demand {
                columns,
                rows,
                cells,
                eligible,
            } => {
                let count = dimensions(columns, rows, 20_000_000)?;
                let eligible = eligible.map(|s| mask(&s, count)).transpose()?;
                let mut previous = None;
                if cells.is_empty()
                    || cells.iter().any(|&(i, w)| {
                        let valid = i < count
                            && w.is_finite()
                            && w > 0.0
                            && previous.is_none_or(|p| i > p)
                            && eligible.as_ref().is_none_or(|m| bit(m, i));
                        previous = Some(i);
                        !valid
                    })
                {
                    return Err("Invalid demand cells, weights, or eligibility mask".into());
                }
                Self::Demand(Demand {
                    columns,
                    rows,
                    cells,
                    eligible,
                })
            }
            Source::Forest {
                columns,
                rows,
                mask: encoded,
            } => {
                let count = dimensions(columns, rows, 200_000_000)?;
                if columns != rows * 2 {
                    return Err("Forest grid must cover 360 by 180 degrees".into());
                }
                let mask = mask(&encoded, count)?;
                if !mask.iter().any(|&b| b != 0) {
                    return Err("Forest dataset has no tree-cover cells".into());
                }
                Self::Forest(Forest {
                    columns,
                    rows,
                    mask,
                })
            }
            Source::River { network } => Self::River(network.decode()?),
            Source::Coast { network } => Self::Coast(network.decode()?),
        })
    }
}

fn cache(kind: &str) -> Result<&'static std::sync::OnceLock<Arc<Dataset>>, String> {
    static GDP: std::sync::OnceLock<Arc<Dataset>> = std::sync::OnceLock::new();
    static FOREST: std::sync::OnceLock<Arc<Dataset>> = std::sync::OnceLock::new();
    static RIVER: std::sync::OnceLock<Arc<Dataset>> = std::sync::OnceLock::new();
    static COAST: std::sync::OnceLock<Arc<Dataset>> = std::sync::OnceLock::new();
    match kind {
        "gdp" => Ok(&GDP),
        "forest" => Ok(&FOREST),
        "river" => Ok(&RIVER),
        "coast" => Ok(&COAST),
        _ => Err("Unknown geographic dataset".into()),
    }
}

fn asset_path(kind: &str) -> String {
    format!("assets/aois/{kind}.aoi-data.json.gz")
}

fn publish(result: Arc<Dataset>, kind: &str, defaults: bool) -> Result<Arc<Dataset>, String> {
    let matches = matches!(
        (kind, result.as_ref()),
        ("gdp", Dataset::Demand(_))
            | ("forest", Dataset::Forest(_))
            | ("river", Dataset::River(_))
            | ("coast", Dataset::Coast(_))
    );
    if !matches {
        return Err(format!("Dataset does not match the selected {kind} source"));
    }
    if defaults {
        let cached = cache(kind)?;
        let _ = cached.set(Arc::clone(&result));
        return Ok(Arc::clone(cached.get().unwrap()));
    }
    Ok(result)
}

impl Network {
    fn validate(&self) -> Result<(), String> {
        let n = self.length.len();
        if n == 0
            || self.offsets.len() != n + 1
            || self.next_index.len() != n
            || self.ids.len() != n
            || self.flow.len() != n
            || self.offsets[0] != 0
            || self.offsets[n] != self.points.len()
            || self
                .offsets
                .windows(2)
                .any(|w| w[1] < w[0].saturating_add(2))
            || self
                .next_index
                .iter()
                .any(|&i| i < -1 || i as i64 >= n as i64)
            || self.length.iter().any(|l| !l.is_finite() || *l <= 0.0)
            || self.flow.iter().any(|f| !f.is_finite() || *f < 0.0)
            || self.points.iter().any(|p| {
                !p[0].is_finite()
                    || !p[1].is_finite()
                    || p[1].abs() > 90.0
                    || p[0].abs() > 1_000_000.0
            })
        {
            return Err("Invalid geographic network arrays or connections".into());
        }
        Ok(())
    }

    pub fn points_for(&self, index: usize) -> &[[f64; 2]] {
        &self.points[self.offsets[index]..self.offsets[index + 1]]
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn load(input: &Input, kind: &str) -> Result<Arc<Dataset>, String> {
    if let Some(bytes) = &input.bytes {
        return publish(Arc::new(Dataset::decode(bytes)?), kind, false);
    }
    let defaults = input.is_default();
    if defaults {
        if let Some(data) = cache(kind)?.get() {
            return Ok(Arc::clone(data));
        }
    }
    let path = if defaults {
        asset_path(kind)
    } else {
        input.path.trim().to_owned()
    };
    let bytes = if path.starts_with("https://") || path.starts_with("http://") {
        let response = ureq::get(&path)
            .timeout(std::time::Duration::from_secs(120))
            .call()
            .map_err(|e| format!("Could not download AOI dataset: {e}"))?;
        let mut bytes = Vec::new();
        response
            .into_reader()
            .take(MAX_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        bytes
    } else {
        let path = std::path::Path::new(&path);
        let path = if path.is_absolute() {
            path.to_owned()
        } else if defaults {
            let deployed = std::env::current_exe()
                .ok()
                .and_then(|e| e.parent().map(|p| p.join(path)));
            deployed
                .filter(|p| p.is_file())
                .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path))
        } else {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
        };
        let file = std::fs::File::open(&path).map_err(|e| format!("Could not open {}: {e}. Run git lfs pull --include=\"assets/aois/**\" or select another dataset.", path.display()))?;
        let mut bytes = Vec::new();
        file.take(MAX_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        bytes
    };
    if bytes.len() as u64 > MAX_BYTES {
        return Err("AOI dataset exceeds 512 MiB".into());
    }
    publish(Arc::new(Dataset::decode(&bytes)?), kind, defaults)
}

#[cfg(target_arch = "wasm32")]
pub(super) async fn load(input: &Input, kind: &str) -> Result<Arc<Dataset>, String> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    if let Some(bytes) = &input.bytes {
        return publish(Arc::new(Dataset::decode(bytes)?), kind, false);
    }
    let defaults = input.is_default();
    if defaults {
        if let Some(data) = cache(kind)?.get() {
            return Ok(Arc::clone(data));
        }
    }
    let path = if defaults {
        asset_path(kind)
    } else {
        input.path.trim().to_owned()
    };
    let window = web_sys::window().ok_or("No browser window")?;
    let response: web_sys::Response = JsFuture::from(window.fetch_with_str(&path))
        .await
        .map_err(|e| format!("Dataset download failed: {e:?}"))?
        .dyn_into()
        .map_err(|_| "Invalid dataset response")?;
    if !response.ok() {
        return Err(format!(
            "Dataset download returned HTTP {}",
            response.status()
        ));
    }
    let buffer = JsFuture::from(response.array_buffer().map_err(|e| format!("{e:?}"))?)
        .await
        .map_err(|e| format!("{e:?}"))?;
    let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
    publish(Arc::new(Dataset::decode(&bytes)?), kind, defaults)
}

#[cfg(test)]
pub(super) fn fixture(kind: &str) -> Input {
    let value = if kind == "forest" {
        let (columns, rows) = (1440usize, 720usize);
        let mut bits = vec![0u8; (columns * rows).div_ceil(8)];
        for y in 356..364 {
            for x in 716..724 {
                let i = y * columns + x;
                bits[i / 8] |= 1 << (i % 8);
            }
        }
        serde_json::json!({"schema_version":1,"kind":"forest","columns":columns,"rows":rows,"mask":STANDARD.encode(bits)})
    } else {
        serde_json::json!({"schema_version":1,"kind":kind,"network":{
            "points":[[0.0,0.0],[1.0,0.0],[1.0,0.0],[2.0,0.0]],
            "offsets":[0,2,4],"next_index":[1,-1],"length":[111.19492664455873,111.19492664455873],
            "ids":[1,2],"flow":[100.0,100.0]}})
    };
    Input {
        path: format!("{kind}.aoi-data.json"),
        bytes: Some(Arc::from(serde_json::to_vec(&value).unwrap())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn default_demand_is_valid_and_cached() {
        let a = load(&Input::default(), "gdp").unwrap();
        assert!(Arc::ptr_eq(&a, &load(&Input::default(), "gdp").unwrap()));
        let Dataset::Demand(d) = a.as_ref() else {
            panic!()
        };
        assert_eq!((d.columns, d.rows), (1440, 720));
        assert_eq!(d.cells.len(), 264199);
    }

    #[test]
    fn lfs_pointers_return_an_actionable_error() {
        let error = Dataset::decode(
            b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 12\n",
        )
        .err()
        .unwrap();
        assert!(error.contains("git lfs pull"));
    }

    #[test]
    fn wrong_source_does_not_enter_the_default_cache() {
        let river = Arc::new(Dataset::decode(fixture("river").bytes.as_ref().unwrap()).unwrap());
        assert!(publish(river, "forest", true).is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn compact_environmental_datasets_generate_aois_and_are_cached() {
        use super::super::generate::{Environment, Job, Settings};
        for kind in ["forest", "river", "coast"] {
            let dataset = load(&Input::default(), kind).unwrap();
            assert!(Arc::ptr_eq(
                &dataset,
                &load(&Input::default(), kind).unwrap()
            ));
            match dataset.as_ref() {
                Dataset::Forest(f) => assert_eq!((f.columns, f.rows), (7200, 3600)),
                Dataset::River(n) => assert_eq!(n.ids.len(), 857210),
                Dataset::Coast(n) => assert_eq!(n.ids.len(), 6847),
                _ => panic!("Wrong geographic source"),
            }
            let mut job = Job::new(
                Settings::Environment(Environment {
                    kind,
                    count: 5,
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
                }),
                Some(dataset),
            )
            .unwrap();
            let preview = loop {
                if let Some(p) = job.step().unwrap() {
                    break p;
                }
            };
            assert_eq!(preview.count, 5);
        }
    }
    #[test]
    fn invalid_datasets_return_errors() {
        for value in [
            serde_json::json!({"schema_version":2,"kind":"demand","columns":1,"rows":1,"cells":[[0,1]]}),
            serde_json::json!({"schema_version":1,"kind":"demand","columns":1,"rows":1,"cells":[[1,1]]}),
            serde_json::json!({"schema_version":1,"kind":"demand","columns":1,"rows":1,"cells":[[0,1]],"eligible":"AA=="}),
            serde_json::json!({"schema_version":1,"kind":"forest","columns":2,"rows":1,"mask":""}),
            serde_json::json!({"schema_version":1,"kind":"river","network":{"points":[[0,0],[1,0]],"offsets":[0,3],"next_index":[-1],"length":[1],"ids":[1],"flow":[10]}}),
        ] {
            assert!(Dataset::decode(&serde_json::to_vec(&value).unwrap()).is_err());
        }
        assert!(Dataset::decode(b"not JSON").is_err());
    }

    #[test]
    fn packed_networks_match_plain_arrays() {
        let points: Vec<u8> = [0.0_f32, 0.0, 1.0, 0.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let offsets: Vec<u8> = [0_u64, 2].into_iter().flat_map(u64::to_le_bytes).collect();
        let value = serde_json::json!({"schema_version":1,"kind":"river","network":{
            "points":STANDARD.encode(points),"offsets":STANDARD.encode(offsets),
            "next_index":STANDARD.encode((-1_i32).to_le_bytes()),"length":STANDARD.encode(111.0_f64.to_le_bytes()),
            "ids":STANDARD.encode(9_i64.to_le_bytes()),"flow":STANDARD.encode(100.0_f32.to_le_bytes())}});
        let Dataset::River(network) =
            Dataset::decode(&serde_json::to_vec(&value).unwrap()).unwrap()
        else {
            panic!()
        };
        assert_eq!(network.points, vec![[0.0, 0.0], [1.0, 0.0]]);
        assert_eq!(network.offsets, vec![0, 2]);
        assert_eq!(network.next_index, vec![-1]);
        assert_eq!(network.ids, vec![9]);
    }
}
