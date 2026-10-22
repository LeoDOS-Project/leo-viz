//! Headless adapter for the same versioned leo-aoi GeoJSON documents as the GUI.
use crate::{
    geometry::{Polygons, Region},
    workload::{hash, AoiRef},
};
use serde::Deserialize;
use std::io::Read;

#[derive(Deserialize)]
struct Document {
    schema_version: u32,
    config: Config,
    aois: Vec<Record>,
}
#[derive(Deserialize)]
struct Config {
    count: usize,
}
#[derive(Deserialize)]
struct Record {
    #[serde(default)]
    id: String,
    #[serde(default)]
    source_kind: String,
    anchor_lon: f64,
    anchor_lat: f64,
    area_km2: f64,
    geometry: Geometry,
}
#[derive(Deserialize)]
#[serde(tag = "type", content = "coordinates")]
enum Geometry {
    Polygon(Vec<Vec<[f64; 2]>>),
    MultiPolygon(Polygons),
}

pub fn decode(bytes: &[u8]) -> Result<Vec<(AoiRef, Region)>, String> {
    let mut data = Vec::new();
    if bytes.starts_with(&[0x1f, 0x8b]) {
        flate2::read::GzDecoder::new(bytes)
            .take(256 * 1024 * 1024 + 1)
            .read_to_end(&mut data)
            .map_err(|e| e.to_string())?;
    } else {
        data.extend_from_slice(bytes);
    }
    if data.len() > 256 * 1024 * 1024 {
        return Err("AOI dataset exceeds 256 MB decoded limit".into());
    }
    let document: Document = serde_json::from_slice(&data).map_err(|e| e.to_string())?;
    if document.schema_version != 1
        || document.aois.is_empty()
        || document.aois.len() != document.config.count
    {
        return Err("Invalid AOI version or count".into());
    }
    document
        .aois
        .into_iter()
        .enumerate()
        .map(|(i, a)| {
            let polygons = match a.geometry {
                Geometry::Polygon(p) => vec![p],
                Geometry::MultiPolygon(p) => p,
            };
            let aoi = AoiRef {
                source: if a.source_kind.is_empty() {
                    "gdp".into()
                } else {
                    a.source_kind
                },
                id: if a.id.is_empty() {
                    format!("aoi-{i:08}")
                } else {
                    a.id
                },
                area_km2: a.area_km2,
                longitude: a.anchor_lon,
                latitude: a.anchor_lat,
                geometry_hash: hash(&serde_json::to_vec(&polygons).map_err(|e| e.to_string())?),
            };
            let region = Region {
                job_key: aoi.key(),
                polygons,
            };
            Ok((aoi, region))
        })
        .collect()
}
