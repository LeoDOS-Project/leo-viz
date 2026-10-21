//! Frozen AOI geometry, independent of the Python runtime and local demand data.
use super::preview::Preview;
use serde::Deserialize;
use std::io::Read;

pub(super) const DEFAULT_JOBS: &str = "Default Jobs";
const SETTINGS: &str = include_str!("../../assets/aois/default-jobs.settings.json");
const DATA: &[u8] = include_bytes!("../../assets/aois/default-jobs.json.gz");

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
}

pub(super) fn default_settings() -> Settings {
    serde_json::from_str(SETTINGS).expect("bundled AOI settings must be valid")
}

pub(super) fn load_default_jobs() -> Result<Preview, String> {
    let mut json = Vec::new();
    flate2::read::GzDecoder::new(DATA)
        .read_to_end(&mut json)
        .map_err(|e| format!("Could not decompress Default Jobs: {e}"))?;
    Preview::from_json(&json)
}
