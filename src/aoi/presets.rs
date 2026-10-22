//! Frozen AOI geometry, independent of Python and local demand data.
use super::preview::Preview;
use serde::Deserialize;
use std::io::Read;

pub(super) const DEFAULT_JOBS: &str = "Default Jobs";
pub(super) const LEGACY_JOBS: &str = "Legacy rectangles (1,000 km²)";
const SETTINGS: &str = include_str!("../../assets/aois/default-jobs-oriented.settings.json");
const DATA: &[u8] = include_bytes!("../../assets/aois/default-jobs-oriented.json.gz");
const LEGACY_SETTINGS: &str = include_str!("../../assets/aois/default-jobs.settings.json");
const LEGACY_DATA: &[u8] = include_bytes!("../../assets/aois/default-jobs.json.gz");

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
pub(super) fn load(legacy: bool) -> Result<Preview, String> {
    let mut json = Vec::new();
    flate2::read::GzDecoder::new(if legacy { LEGACY_DATA } else { DATA })
        .read_to_end(&mut json)
        .map_err(|e| format!("Could not decompress AOI preset: {e}"))?;
    Preview::from_json(&json)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn legacy_preset_remains_available_for_existing_workloads() {
        let old = load(true).unwrap();
        assert_eq!(old.count, 50_000);
        assert!((old.mean_area - 994.343171430485).abs() < 1e-8);
        assert_eq!(settings(true).generation.shape, "rectangle");
        assert_eq!(default_settings().mean_area_km2, 100.0);
    }
}
