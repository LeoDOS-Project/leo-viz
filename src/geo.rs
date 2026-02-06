pub struct CityLabel {
    pub lat: f64,
    pub lon: f64,
    pub name: String,
    pub population: f64,
}

pub struct GeoOverlayData {
    pub borders: Vec<Vec<(f64, f64)>>,
    pub cities: Vec<CityLabel>,
}

pub enum GeoLoadState {
    NotLoaded,
    Loading,
    Loaded(GeoOverlayData),
    Failed,
}

pub fn parse_geojson_borders(json: &str) -> Result<Vec<Vec<(f64, f64)>>, String> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| format!("{}", e))?;
    let features = v["features"].as_array().ok_or("no features")?;
    let mut polylines = Vec::new();
    for feat in features {
        let geom = &feat["geometry"];
        match geom["type"].as_str() {
            Some("LineString") => {
                if let Some(line) = extract_coord_line(&geom["coordinates"]) {
                    polylines.push(line);
                }
            }
            Some("MultiLineString") => {
                if let Some(arrs) = geom["coordinates"].as_array() {
                    for arr in arrs {
                        if let Some(line) = extract_coord_line(arr) {
                            polylines.push(line);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(polylines)
}

fn extract_coord_line(arr: &serde_json::Value) -> Option<Vec<(f64, f64)>> {
    let points = arr.as_array()?;
    let coords: Vec<(f64, f64)> = points.iter().filter_map(|p| {
        let a = p.as_array()?;
        Some((a.get(1)?.as_f64()?, a.first()?.as_f64()?))
    }).collect();
    if coords.is_empty() { None } else { Some(coords) }
}

pub fn parse_geojson_cities(json: &str) -> Result<Vec<CityLabel>, String> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| format!("{}", e))?;
    let features = v["features"].as_array().ok_or("no features")?;
    let mut cities = Vec::new();
    for feat in features {
        let props = &feat["properties"];
        let geom = &feat["geometry"];
        if let Some(coords) = geom["coordinates"].as_array() {
            let lon = coords[0].as_f64().unwrap_or(0.0);
            let lat = coords[1].as_f64().unwrap_or(0.0);
            let name = props["name"].as_str().unwrap_or("").to_string();
            let pop = props["pop_max"].as_f64()
                .or_else(|| props["pop_min"].as_f64())
                .unwrap_or(0.0);
            if !name.is_empty() {
                cities.push(CityLabel { lat, lon, name, population: pop });
            }
        }
    }
    cities.sort_by(|a, b| b.population.partial_cmp(&a.population).unwrap_or(std::cmp::Ordering::Equal));
    Ok(cities)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn dirs_cache() -> std::path::PathBuf {
    if let Some(dir) = dirs_sys_cache() {
        dir
    } else {
        std::path::PathBuf::from(".")
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn dirs_sys_cache() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache"))
}

#[cfg(not(target_arch = "wasm32"))]
fn fetch_or_cache_geojson(filename: &str, url: &str) -> Result<String, String> {
    let cache_dir = dirs_cache().join("leo-viz").join("geodata");
    let _ = std::fs::create_dir_all(&cache_dir);
    let path = cache_dir.join(filename);
    if path.exists() {
        return std::fs::read_to_string(&path).map_err(|e| format!("{}", e));
    }
    let resp = ureq::get(url).call().map_err(|e| format!("{}", e))?;
    let data = resp.into_string().map_err(|e| format!("{}", e))?;
    let _ = std::fs::write(&path, &data);
    Ok(data)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_geo_overlay() -> Result<GeoOverlayData, String> {
    let borders_json = fetch_or_cache_geojson(
        "ne_110m_admin_0_boundary_lines_land.geojson",
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_boundary_lines_land.geojson",
    )?;
    let cities_json = fetch_or_cache_geojson(
        "ne_110m_populated_places_simple.geojson",
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_populated_places_simple.geojson",
    )?;
    let borders = parse_geojson_borders(&borders_json)?;
    let cities = parse_geojson_cities(&cities_json)?;
    Ok(GeoOverlayData { borders, cities })
}
