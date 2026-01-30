use eframe::egui;
use egui_plot::{Line, Plot, PlotImage, PlotPoints, PlotPoint, Points, Polygon, Text};
use nalgebra::{Matrix3, Vector3};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, mpsc};
use sgp4::Constants;
use chrono::{DateTime, Utc, Duration};

#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::JsCast;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum CelestialBody {
    Earth,
    Moon,
    Mars,
    Mercury,
    Venus,
    Jupiter,
    Saturn,
    Sun,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Skin {
    Default,
    HellOnEarth,
    Terraformed,
    Civilized,
}

impl Skin {
    fn label(&self) -> &'static str {
        match self {
            Skin::Default => "Default",
            Skin::HellOnEarth => "Hell on Earth",
            Skin::Terraformed => "Terraformed",
            Skin::Civilized => "Civilized",
        }
    }

    fn filename(&self, body: CelestialBody) -> Option<&'static str> {
        match (body, self) {
            (CelestialBody::Earth, Skin::Default) => Some("textures/earth_2k.jpg"),
            (CelestialBody::Earth, Skin::HellOnEarth) => Some("textures/hell_on_earth_2k.png"),
            (CelestialBody::Moon, Skin::Default) => Some("textures/moon_2k.jpg"),
            (CelestialBody::Mars, Skin::Default) => Some("textures/mars_2k.jpg"),
            (CelestialBody::Mars, Skin::Terraformed) => Some("textures/mars_terraformed.png"),
            (CelestialBody::Mars, Skin::Civilized) => Some("textures/mars_civilized.png"),
            (CelestialBody::Mercury, Skin::Default) => Some("textures/mercury_2k.jpg"),
            (CelestialBody::Venus, Skin::Default) => Some("textures/venus_2k.jpg"),
            (CelestialBody::Jupiter, Skin::Default) => Some("textures/jupiter_2k.jpg"),
            (CelestialBody::Saturn, Skin::Default) => Some("textures/saturn_2k.jpg"),
            (CelestialBody::Sun, Skin::Default) => Some("textures/sun_2k.jpg"),
            _ => None,
        }
    }
}

impl CelestialBody {
    fn label(&self) -> &'static str {
        match self {
            CelestialBody::Earth => "Earth",
            CelestialBody::Moon => "Moon",
            CelestialBody::Mars => "Mars",
            CelestialBody::Mercury => "Mercury",
            CelestialBody::Venus => "Venus",
            CelestialBody::Jupiter => "Jupiter",
            CelestialBody::Saturn => "Saturn",
            CelestialBody::Sun => "Sun",
        }
    }

    fn available_skins(&self) -> &'static [Skin] {
        match self {
            CelestialBody::Earth => &[Skin::Default, Skin::HellOnEarth],
            CelestialBody::Mars => &[Skin::Default, Skin::Terraformed, Skin::Civilized],
            _ => &[Skin::Default],
        }
    }

    const ALL: [CelestialBody; 8] = [
        CelestialBody::Earth,
        CelestialBody::Moon,
        CelestialBody::Mars,
        CelestialBody::Mercury,
        CelestialBody::Venus,
        CelestialBody::Jupiter,
        CelestialBody::Saturn,
        CelestialBody::Sun,
    ];

    fn radius_km(&self) -> f64 {
        match self {
            CelestialBody::Earth => 6371.0,
            CelestialBody::Moon => 1737.4,
            CelestialBody::Mars => 3389.5,
            CelestialBody::Mercury => 2439.7,
            CelestialBody::Venus => 6051.8,
            CelestialBody::Jupiter => 69911.0,
            CelestialBody::Saturn => 58232.0,
            CelestialBody::Sun => 696340.0,
        }
    }

    fn mu(&self) -> f64 {
        match self {
            CelestialBody::Earth => 398600.4418,
            CelestialBody::Moon => 4902.8,
            CelestialBody::Mars => 42828.37,
            CelestialBody::Mercury => 22032.0,
            CelestialBody::Venus => 324859.0,
            CelestialBody::Jupiter => 126686534.0,
            CelestialBody::Saturn => 37931187.0,
            CelestialBody::Sun => 132712440018.0,
        }
    }
}

#[derive(Clone)]
#[allow(dead_code)]
enum TextureLoadState {
    Loading,
    Loaded(Arc<EarthTexture>),
    Failed(String),
}

const EARTH_TEXTURE_BYTES: &[u8] = include_bytes!("../earth.jpg");

const COLOR_ASCENDING: egui::Color32 = egui::Color32::from_rgb(200, 120, 50);
const COLOR_DESCENDING: egui::Color32 = egui::Color32::from_rgb(50, 100, 180);

struct EarthTexture {
    width: u32,
    height: u32,
    pixels: Vec<[u8; 3]>,
}

impl EarthTexture {
    fn load() -> Self {
        Self::from_bytes(EARTH_TEXTURE_BYTES).expect("Failed to load built-in Earth texture")
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| format!("Failed to decode image: {}", e))?
            .to_rgb8();
        let width = img.width();
        let height = img.height();
        let pixels: Vec<[u8; 3]> = img.pixels().map(|p| p.0).collect();
        Ok(Self { width, height, pixels })
    }

    fn sample(&self, u: f64, v: f64) -> [u8; 3] {
        let x = ((u * self.width as f64) as u32).min(self.width - 1);
        let y = ((v * self.height as f64) as u32).min(self.height - 1);
        self.pixels[(y * self.width + x) as usize]
    }

    fn render_sphere(&self, size: usize, rot: &Matrix3<f64>) -> egui::ColorImage {
        let mut pixels = vec![egui::Color32::TRANSPARENT; size * size];
        let center = size as f64 / 2.0;
        let radius = center * 0.95;
        let inv_rot = rot.transpose();

        for py in 0..size {
            for px in 0..size {
                let dx = px as f64 - center;
                let dy = py as f64 - center;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < radius * radius {
                    let z = (radius * radius - dist_sq).sqrt();
                    let x = dx / radius;
                    let y = -dy / radius;
                    let z = z / radius;

                    let v = inv_rot * Vector3::new(x, y, z);

                    let lat = v.y.asin();
                    let lon = v.z.atan2(-v.x);

                    let u = (lon + PI) / (2.0 * PI);
                    let vt = (PI / 2.0 - lat) / PI;

                    let [r, g, b] = self.sample(u, vt);

                    let shade = (0.3 + 0.7 * z.max(0.0)) as f32;
                    let r = (r as f32 * shade) as u8;
                    let g = (g as f32 * shade) as u8;
                    let b = (b as f32 * shade) as u8;

                    pixels[py * size + px] = egui::Color32::from_rgb(r, g, b);
                }
            }
        }

        egui::ColorImage {
            size: [size, size],
            pixels,
            source_size: egui::Vec2::ZERO,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum WalkerType {
    Delta,
    Star,
}

struct WalkerConstellation {
    walker_type: WalkerType,
    total_sats: usize,
    num_planes: usize,
    altitude_km: f64,
    inclination_deg: f64,
    phasing: f64,
    planet_radius: f64,
    planet_mu: f64,
}

impl WalkerConstellation {
    fn sats_per_plane(&self) -> usize {
        self.total_sats / self.num_planes
    }

    fn raan_spread(&self) -> f64 {
        match self.walker_type {
            WalkerType::Delta => 2.0 * PI,
            WalkerType::Star => PI,
        }
    }

    fn satellite_positions(&self, time: f64) -> Vec<SatelliteState> {
        let mut positions = Vec::with_capacity(self.total_sats);
        let sats_per_plane = self.sats_per_plane();
        let orbit_radius = self.planet_radius + self.altitude_km;
        let period = 2.0 * PI * (orbit_radius.powi(3) / self.planet_mu).sqrt();
        let mean_motion = 2.0 * PI / period;
        let raan_spread = self.raan_spread();
        let inc = self.inclination_deg.to_radians();
        let inc_cos = inc.cos();
        let inc_sin = inc.sin();
        let raan_step = raan_spread / self.num_planes as f64;
        let sat_step = 2.0 * PI / sats_per_plane as f64;
        let is_star = self.walker_type == WalkerType::Star;

        let phase_step = self.phasing * 2.0 * PI / self.total_sats as f64;

        for plane in 0..self.num_planes {
            let raan = raan_step * plane as f64;
            let raan_cos = raan.cos();
            let raan_sin = raan.sin();
            let phase_offset = phase_step * plane as f64;

            for sat in 0..sats_per_plane {
                let true_anomaly = sat_step * sat as f64 + mean_motion * time + phase_offset;
                let ascending = true_anomaly.cos() > 0.0;

                let x_orbital = orbit_radius * true_anomaly.cos();
                let y_orbital = orbit_radius * true_anomaly.sin();

                let x = x_orbital * raan_cos - y_orbital * inc_cos * raan_sin;
                let z = x_orbital * raan_sin + y_orbital * inc_cos * raan_cos;
                let y = y_orbital * inc_sin;

                let lat = (y / orbit_radius).asin().to_degrees();
                let lon = z.atan2(x).to_degrees();

                positions.push(SatelliteState {
                    plane,
                    sat_index: sat,
                    x,
                    y,
                    z,
                    lat,
                    lon,
                    ascending,
                    neighbor_idx: None,
                });
            }
        }

        for i in 0..positions.len() {
            let sat = &positions[i];
            if is_star && sat.plane == self.num_planes - 1 {
                continue;
            }
            let next_plane = (sat.plane + 1) % self.num_planes;
            let next_plane_start = next_plane * sats_per_plane;
            let next_plane_end = next_plane_start + sats_per_plane;
            let target_idx = sat.sat_index;
            let target_ascending = sat.ascending;
            for j in next_plane_start..next_plane_end {
                let other = &positions[j];
                if other.sat_index == target_idx && other.ascending == target_ascending {
                    positions[i].neighbor_idx = Some(j);
                    break;
                }
            }
        }

        positions
    }

    fn orbit_points_3d(&self, plane: usize) -> Vec<(f64, f64, f64)> {
        let orbit_radius = self.planet_radius + self.altitude_km;
        let raan = (self.raan_spread() / self.num_planes as f64) * plane as f64;
        let inc = self.inclination_deg.to_radians();
        let inc_cos = inc.cos();
        let inc_sin = inc.sin();
        let raan_cos = raan.cos();
        let raan_sin = raan.sin();

        (0..=200)
            .map(|i| {
                let theta = 2.0 * PI * i as f64 / 200.0;
                let x_orbital = orbit_radius * theta.cos();
                let y_orbital = orbit_radius * theta.sin();

                let x = x_orbital * raan_cos - y_orbital * inc_cos * raan_sin;
                let z = x_orbital * raan_sin + y_orbital * inc_cos * raan_cos;
                let y = y_orbital * inc_sin;

                (x, y, z)
            })
            .collect()
    }
}

struct SatelliteState {
    plane: usize,
    sat_index: usize,
    x: f64,
    y: f64,
    z: f64,
    lat: f64,
    lon: f64,
    ascending: bool,
    neighbor_idx: Option<usize>,
}

fn rotate_point_matrix(x: f64, y: f64, z: f64, rot: &Matrix3<f64>) -> (f64, f64, f64) {
    let v = rot * Vector3::new(x, y, z);
    (v.x, v.y, v.z)
}

fn rotation_from_drag(dx: f64, dy: f64) -> Matrix3<f64> {
    let rot_y = Matrix3::new(
        dx.cos(), 0.0, dx.sin(),
        0.0, 1.0, 0.0,
        -dx.sin(), 0.0, dx.cos(),
    );
    let rot_x = Matrix3::new(
        1.0, 0.0, 0.0,
        0.0, dy.cos(), -dy.sin(),
        0.0, dy.sin(), dy.cos(),
    );
    rot_x * rot_y
}

#[derive(Clone, Copy, PartialEq)]
enum Preset {
    None,
    Starlink,
    OneWeb,
    Iridium,
    Kuiper,
    Iris2,
    Telesat,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum TlePreset {
    Starlink,
    OneWeb,
    Iridium,
    IridiumNext,
    Globalstar,
    Orbcomm,
    Gps,
    Galileo,
    Glonass,
    Beidou,
}

impl TlePreset {
    fn label(&self) -> &'static str {
        match self {
            TlePreset::Starlink => "Starlink",
            TlePreset::OneWeb => "OneWeb",
            TlePreset::Iridium => "Iridium",
            TlePreset::IridiumNext => "Iridium NEXT",
            TlePreset::Globalstar => "Globalstar",
            TlePreset::Orbcomm => "Orbcomm",
            TlePreset::Gps => "GPS",
            TlePreset::Galileo => "Galileo",
            TlePreset::Glonass => "GLONASS",
            TlePreset::Beidou => "Beidou",
        }
    }

    fn url(&self) -> &'static str {
        match self {
            TlePreset::Starlink => "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
            TlePreset::OneWeb => "https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle",
            TlePreset::Iridium => "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium&FORMAT=tle",
            TlePreset::IridiumNext => "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-NEXT&FORMAT=tle",
            TlePreset::Globalstar => "https://celestrak.org/NORAD/elements/gp.php?GROUP=globalstar&FORMAT=tle",
            TlePreset::Orbcomm => "https://celestrak.org/NORAD/elements/gp.php?GROUP=orbcomm&FORMAT=tle",
            TlePreset::Gps => "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle",
            TlePreset::Galileo => "https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle",
            TlePreset::Glonass => "https://celestrak.org/NORAD/elements/gp.php?GROUP=glo-ops&FORMAT=tle",
            TlePreset::Beidou => "https://celestrak.org/NORAD/elements/gp.php?GROUP=beidou&FORMAT=tle",
        }
    }

    const ALL: [TlePreset; 10] = [
        TlePreset::Starlink,
        TlePreset::OneWeb,
        TlePreset::Iridium,
        TlePreset::IridiumNext,
        TlePreset::Globalstar,
        TlePreset::Orbcomm,
        TlePreset::Gps,
        TlePreset::Galileo,
        TlePreset::Glonass,
        TlePreset::Beidou,
    ];
}

#[derive(Clone)]
struct TleSatellite {
    #[allow(dead_code)]
    name: String,
    constants: Constants,
    epoch_minutes: f64,
}

#[derive(Clone)]
#[allow(dead_code)]
enum TleLoadState {
    NotLoaded,
    Loading,
    Loaded { satellites: Vec<TleSatellite>, loaded_at: std::time::Instant },
    Failed(String),
}

fn current_utc_minutes() -> f64 {
    let now = std::time::SystemTime::now();
    let since_epoch = now.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    since_epoch.as_secs_f64() / 60.0
}

fn datetime_to_minutes(dt: &sgp4::chrono::NaiveDateTime) -> f64 {
    dt.and_utc().timestamp() as f64 / 60.0
}

fn greenwich_mean_sidereal_time(timestamp: DateTime<Utc>) -> f64 {
    let j2000 = DateTime::parse_from_rfc3339("2000-01-01T12:00:00Z")
        .unwrap()
        .with_timezone(&Utc);
    let days_since_j2000 = (timestamp - j2000).num_milliseconds() as f64 / (1000.0 * 86400.0);
    let centuries = days_since_j2000 / 36525.0;
    let gmst_degrees = 280.46061837
        + 360.98564736629 * days_since_j2000
        + 0.000387933 * centuries * centuries
        - centuries * centuries * centuries / 38710000.0;
    let gmst_normalized = gmst_degrees.rem_euclid(360.0);
    gmst_normalized.to_radians()
}

#[cfg(not(target_arch = "wasm32"))]
fn fetch_tle_data(url: &str) -> Result<Vec<TleSatellite>, String> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("HTTP error: {}", e))?;

    let body = response.into_string()
        .map_err(|e| format!("Read error: {}", e))?;

    parse_tle_data(&body)
}

#[cfg(target_arch = "wasm32")]
fn fetch_tle_data(_url: &str) -> Result<Vec<TleSatellite>, String> {
    Err("WASM fetch not yet implemented".to_string())
}

fn parse_tle_data(data: &str) -> Result<Vec<TleSatellite>, String> {
    let lines: Vec<&str> = data.lines().collect();
    let mut satellites = Vec::new();

    let mut i = 0;
    while i + 2 < lines.len() {
        let name_line = lines[i].trim();
        let line1 = lines[i + 1].trim();
        let line2 = lines[i + 2].trim();

        if !line1.starts_with('1') || !line2.starts_with('2') {
            i += 1;
            continue;
        }

        let tle = format!("{}\n{}\n{}", name_line, line1, line2);

        match sgp4::parse_3les(&tle) {
            Ok(elements_vec) => {
                for elements in elements_vec {
                    match Constants::from_elements(&elements) {
                        Ok(constants) => {
                            let epoch_minutes = datetime_to_minutes(&elements.datetime);
                            satellites.push(TleSatellite {
                                name: elements.object_name.unwrap_or_default(),
                                constants,
                                epoch_minutes,
                            });
                        }
                        Err(_) => continue,
                    }
                }
            }
            Err(_) => {}
        }

        i += 3;
    }

    if satellites.is_empty() {
        Err("No valid TLE data found".to_string())
    } else {
        Ok(satellites)
    }
}

#[derive(Clone)]
struct ConstellationConfig {
    sats_per_plane: usize,
    num_planes: usize,
    altitude_km: f64,
    inclination: f64,
    walker_type: WalkerType,
    phasing: f64,
    preset: Preset,
    color_offset: usize,
    hidden: bool,
}

impl ConstellationConfig {
    fn new(color_offset: usize) -> Self {
        Self {
            sats_per_plane: 11,
            num_planes: 6,
            altitude_km: 780.0,
            inclination: 86.4,
            walker_type: WalkerType::Star,
            phasing: 2.0,
            preset: Preset::Iridium,
            color_offset,
            hidden: false,
        }
    }

    fn total_sats(&self) -> usize {
        self.sats_per_plane * self.num_planes
    }

    fn constellation(&self, planet_radius: f64, planet_mu: f64) -> WalkerConstellation {
        WalkerConstellation {
            walker_type: self.walker_type,
            total_sats: self.sats_per_plane * self.num_planes,
            num_planes: self.num_planes,
            altitude_km: self.altitude_km,
            inclination_deg: self.inclination,
            phasing: self.phasing,
            planet_radius,
            planet_mu,
        }
    }

    fn preset_name(&self) -> &'static str {
        match self.preset {
            Preset::None => "Custom",
            Preset::Starlink => "Starlink",
            Preset::OneWeb => "OneWeb",
            Preset::Iridium => "Iridium",
            Preset::Kuiper => "Kuiper",
            Preset::Iris2 => "Iris²",
            Preset::Telesat => "Telesat",
        }
    }
}

#[derive(Clone)]
struct TabConfig {
    name: String,
    constellations: Vec<ConstellationConfig>,
    constellation_counter: usize,
    celestial_body: CelestialBody,
    skin: Skin,
    satellite_cameras: Vec<SatelliteCamera>,
    pending_cameras: Vec<SatelliteCamera>,
    cameras_to_remove: Vec<usize>,
    show_stats: bool,
    show_tle_window: bool,
    tle_selections: HashMap<TlePreset, (bool, TleLoadState)>,
}

impl TabConfig {
    fn new(name: String) -> Self {
        let mut tle_selections = HashMap::new();
        for preset in TlePreset::ALL {
            tle_selections.insert(preset, (false, TleLoadState::NotLoaded));
        }
        Self {
            name,
            constellations: Vec::new(),
            constellation_counter: 0,
            celestial_body: CelestialBody::Earth,
            skin: Skin::Default,
            satellite_cameras: Vec::new(),
            pending_cameras: Vec::new(),
            cameras_to_remove: Vec::new(),
            show_stats: false,
            show_tle_window: false,
            tle_selections,
        }
    }

    fn add_constellation(&mut self) {
        self.constellations.push(ConstellationConfig::new(self.constellation_counter));
        self.constellation_counter += 1;
    }

    fn tle_satellite_positions(&self, time: f64) -> Vec<SatelliteState> {
        let now_minutes = current_utc_minutes();
        let time_offset_minutes = time / 60.0;
        let propagation_minutes = now_minutes + time_offset_minutes;

        let mut positions = Vec::new();

        for (preset_idx, preset) in TlePreset::ALL.iter().enumerate() {
            let Some((selected, state)) = self.tle_selections.get(preset) else { continue };
            if !*selected { continue; }
            let TleLoadState::Loaded { satellites, .. } = state else { continue };

            for (idx, sat) in satellites.iter().enumerate() {
                let minutes_since_epoch = propagation_minutes - sat.epoch_minutes;
                let prediction = match sat.constants.propagate(sgp4::MinutesSinceEpoch(minutes_since_epoch)) {
                    Ok(p) => p,
                    Err(_) => continue,
                };

                let x = prediction.position[0];
                let y = prediction.position[2];
                let z = prediction.position[1];

                let r = (x * x + y * y + z * z).sqrt();
                let lat = (y / r).asin().to_degrees();
                let lon = z.atan2(x).to_degrees();

                let ascending = prediction.velocity[2] > 0.0;

                positions.push(SatelliteState {
                    plane: preset_idx,
                    sat_index: idx,
                    x, y, z,
                    lat, lon,
                    ascending,
                    neighbor_idx: None,
                });
            }
        }

        positions
    }

    #[allow(dead_code)]
    fn tle_total_sats(&self) -> usize {
        self.tle_selections.values()
            .filter(|(selected, _)| *selected)
            .map(|(_, state)| {
                if let TleLoadState::Loaded { satellites, .. } = state {
                    satellites.len()
                } else {
                    0
                }
            })
            .sum()
    }
}

#[derive(Clone)]
struct SatelliteCamera {
    id: usize,
    label: String,
    constellation_idx: usize,
    plane: usize,
    sat_index: usize,
    screen_pos: Option<egui::Pos2>,
}

struct App {
    tabs: Vec<TabConfig>,
    active_tab: usize,
    camera_id_counter: usize,
    tab_counter: usize,
    time: f64,
    speed: f64,
    animate: bool,
    show_orbits: bool,
    show_links: bool,
    show_intra_links: bool,
    show_ground_track: bool,
    show_torus: bool,
    show_axes: bool,
    show_coverage: bool,
    coverage_angle: f64,
    hide_behind_earth: bool,
    single_color_per_constellation: bool,
    menu_open: bool,
    zoom: f64,
    torus_zoom: f64,
    vertical_split: f32,
    sat_radius: f32,
    rotation: Matrix3<f64>,
    torus_rotation: Matrix3<f64>,
    planet_textures: HashMap<(CelestialBody, Skin), Arc<EarthTexture>>,
    planet_image_handles: HashMap<(CelestialBody, Skin), egui::TextureHandle>,
    last_rotation: Option<Matrix3<f64>>,
    earth_resolution: usize,
    last_resolution: usize,
    texture_load_state: TextureLoadState,
    pending_body: Option<(CelestialBody, Skin)>,
    dark_mode: bool,
    show_info: bool,
    follow_satellite: bool,
    show_routing_paths: bool,
    show_manhattan_path: bool,
    show_shortest_path: bool,
    show_camera_windows: bool,
    render_planet: bool,
    last_max_planet_radius: f64,
    real_time: f64,
    start_timestamp: DateTime<Utc>,
    show_side_panel: bool,
    #[cfg(not(target_arch = "wasm32"))]
    tle_fetch_tx: mpsc::Sender<(TlePreset, Result<Vec<TleSatellite>, String>)>,
    #[cfg(not(target_arch = "wasm32"))]
    tle_fetch_rx: mpsc::Receiver<(TlePreset, Result<Vec<TleSatellite>, String>)>,
}

impl Default for App {
    fn default() -> Self {
        let torus_initial = Matrix3::new(
            1.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
            0.0, 1.0, 0.0,
        );
        let builtin_texture = Arc::new(EarthTexture::load());
        #[cfg(not(target_arch = "wasm32"))]
        let (tle_fetch_tx, tle_fetch_rx) = mpsc::channel();
        Self {
            tabs: vec![TabConfig::new("Planet 1".to_string())],
            active_tab: 0,
            camera_id_counter: 0,
            tab_counter: 1,
            time: 0.0,
            speed: 1.0,
            animate: true,
            show_orbits: true,
            show_links: true,
            show_intra_links: false,
            show_ground_track: false,
            show_torus: false,
            show_axes: false,
            show_coverage: false,
            coverage_angle: 25.0,
            hide_behind_earth: true,
            single_color_per_constellation: false,
            menu_open: false,
            zoom: 1.0,
            torus_zoom: 1.0,
            vertical_split: 0.6,
            sat_radius: 5.0,
            rotation: Matrix3::identity(),
            torus_rotation: torus_initial,
            planet_textures: {
                let mut map = HashMap::new();
                map.insert((CelestialBody::Earth, Skin::Default), builtin_texture.clone());
                map
            },
            planet_image_handles: HashMap::new(),
            last_rotation: None,
            earth_resolution: 512,
            last_resolution: 0,
            texture_load_state: TextureLoadState::Loaded(builtin_texture),
            pending_body: None,
            dark_mode: true,
            show_info: false,
            follow_satellite: false,
            show_routing_paths: false,
            show_manhattan_path: true,
            show_shortest_path: true,
            show_camera_windows: false,
            render_planet: true,
            last_max_planet_radius: CelestialBody::Earth.radius_km(),
            real_time: 0.0,
            start_timestamp: Utc::now(),
            show_side_panel: true,
            #[cfg(not(target_arch = "wasm32"))]
            tle_fetch_tx,
            #[cfg(not(target_arch = "wasm32"))]
            tle_fetch_rx,
        }
    }
}


impl App {
    fn add_tab(&mut self) {
        self.tab_counter += 1;
        let tab = TabConfig::new(format!("Planet {}", self.tab_counter));
        self.tabs.push(tab);
        self.active_tab = self.tabs.len() - 1;
    }

    fn remove_tab(&mut self, idx: usize) {
        if self.tabs.len() > 1 {
            self.tabs.remove(idx);
            if self.active_tab >= self.tabs.len() {
                self.active_tab = self.tabs.len() - 1;
            }
        }
    }

    fn render_tab_ui(&mut self, ui: &mut egui::Ui, tab_idx: usize, single_tab: bool) {

        {
            let tab = &mut self.tabs[tab_idx];
            for camera in std::mem::take(&mut tab.pending_cameras) {
                tab.satellite_cameras.push(camera);
            }
            tab.satellite_cameras.retain(|c| !tab.cameras_to_remove.contains(&c.id));
            tab.cameras_to_remove.clear();
        }

        let tab = &mut self.tabs[tab_idx];
        ui.horizontal(|ui| {
            ui.label("Planet:");
            egui::ComboBox::from_id_salt(format!("planet_{}", tab.name))
                .selected_text(tab.celestial_body.label())
                .show_ui(ui, |ui| {
                    for body in CelestialBody::ALL {
                        if ui.selectable_value(&mut tab.celestial_body, body, body.label()).changed() {
                            tab.skin = Skin::Default;
                        }
                    }
                });

            let available_skins = tab.celestial_body.available_skins();
            if available_skins.len() > 1 {
                ui.label("Skin:");
                egui::ComboBox::from_id_salt(format!("skin_{}", tab.name))
                    .selected_text(tab.skin.label())
                    .show_ui(ui, |ui| {
                        for skin in available_skins {
                            ui.selectable_value(&mut tab.skin, *skin, skin.label());
                        }
                    });
            }

            if ui.button("Stats").clicked() {
                tab.show_stats = !tab.show_stats;
            }
            ui.checkbox(&mut tab.show_tle_window, "Live");
        });

        let tab = &mut self.tabs[tab_idx];
        if tab.show_stats {
            let tab_name = tab.name.clone();
            let celestial_body = tab.celestial_body;
            let planet_radius = celestial_body.radius_km();
            let mu = celestial_body.mu();
            let constellations: Vec<_> = tab.constellations.iter().map(|c| c.clone()).collect();
            let tle_selections = tab.tle_selections.clone();

            egui::Window::new(format!("Stats - {}", tab_name))
                .open(&mut self.tabs[tab_idx].show_stats)
                .show(ui.ctx(), |ui| {
                    const SPEED_OF_LIGHT_KM_S: f64 = 299792.0;

                    ui.heading(celestial_body.label());
                    ui.label(format!("  Radius: {:.0} km", planet_radius));
                    ui.label(format!("  μ: {:.0} km³/s²", mu));
                    let surface_gravity = mu / (planet_radius * planet_radius);
                    ui.label(format!("  Surface gravity: {:.2} m/s²", surface_gravity * 1000.0));
                    let escape_velocity = (2.0 * mu * 1e9 / (planet_radius * 1000.0)).sqrt() / 1000.0;
                    ui.label(format!("  Escape velocity: {:.2} km/s", escape_velocity));
                    let geo_orbit = (mu * (86400.0 / (2.0 * PI)).powi(2)).powf(1.0/3.0);
                    let geo_altitude = geo_orbit - planet_radius;
                    if geo_altitude > 0.0 {
                        ui.label(format!("  Geostationary alt: {:.0} km", geo_altitude));
                    }
                    ui.separator();

                    if !constellations.is_empty() {
                        ui.heading("Walker Constellations");
                        for cons in &constellations {
                            ui.strong(cons.preset_name());
                            ui.label(format!("  Satellites: {}", cons.total_sats()));
                            {
                                let orbit_radius = planet_radius + cons.altitude_km;
                                let orbit_radius_m = orbit_radius * 1000.0;
                                let velocity_ms = (mu * 1e9 / orbit_radius_m).sqrt();
                                let velocity_kmh = velocity_ms * 3.6;

                                let intra_plane_dist = orbit_radius * (2.0 * (1.0 - (2.0 * PI / cons.sats_per_plane as f64).cos())).sqrt();
                                let inc_rad = cons.inclination.to_radians();
                                let base_inter = orbit_radius * (2.0 * (1.0 - (2.0 * PI / cons.num_planes as f64).cos())).sqrt();
                                let inter_plane_dist = base_inter * inc_rad.sin().abs().max(0.1);
                                let ground_dist = cons.altitude_km;

                                let intra_latency_ms = intra_plane_dist / SPEED_OF_LIGHT_KM_S * 1000.0;
                                let inter_latency_ms = inter_plane_dist / SPEED_OF_LIGHT_KM_S * 1000.0;
                                let ground_latency_ms = ground_dist / SPEED_OF_LIGHT_KM_S * 1000.0;

                                ui.label(format!("  Velocity: {:.0} km/h", velocity_kmh));
                                ui.label(format!("  Intra-plane: {:.0} km ({:.2} ms)", intra_plane_dist, intra_latency_ms));
                                ui.label(format!("  Inter-plane: {:.0} km ({:.2} ms)", inter_plane_dist, inter_latency_ms));
                                ui.label(format!("  Ground: {:.0} km ({:.2} ms)", ground_dist, ground_latency_ms));
                            }
                        }
                        ui.separator();
                    }

                    let live_data: Vec<_> = TlePreset::ALL.iter()
                        .filter_map(|preset| {
                            if let Some((selected, state)) = tle_selections.get(preset) {
                                if *selected {
                                    if let TleLoadState::Loaded { satellites, .. } = state {
                                        return Some((preset.label(), satellites.len()));
                                    }
                                }
                            }
                            None
                        })
                        .collect();

                    if !live_data.is_empty() {
                        ui.heading("Live Data (TLE)");
                        let mut total = 0;
                        for (name, count) in &live_data {
                            ui.label(format!("  {}: {} satellites", name, count));
                            total += count;
                        }
                        ui.label(format!("  Total: {} satellites", total));
                    }
                });
        }

        ui.separator();

        let mut const_to_remove: Option<usize> = None;
        let tab = &mut self.tabs[tab_idx];
        let num_constellations = tab.constellations.len();
        let tab_name = tab.name.clone();
        let show_tle = tab.show_tle_window;

        #[cfg(not(target_arch = "wasm32"))]
        let tle_fetch_tx = self.tle_fetch_tx.clone();

        let tab = &mut self.tabs[tab_idx];
        ui.horizontal(|ui| {
            if show_tle {
                ui.vertical(|ui| {
                    let mut fetch_requested = false;
                    ui.horizontal(|ui| {
                        ui.label("TLE");
                        if ui.small_button("All").clicked() {
                            for (selected, _) in tab.tle_selections.values_mut() {
                                *selected = true;
                            }
                        }
                        if ui.small_button("None").clicked() {
                            for (selected, _) in tab.tle_selections.values_mut() {
                                *selected = false;
                            }
                        }
                        if ui.small_button("Fetch").clicked() {
                            fetch_requested = true;
                        }
                        if ui.small_button("x").clicked() {
                            tab.show_tle_window = false;
                        }
                    });

                    ui.horizontal(|ui| {
                        for col in 0..2 {
                            ui.vertical(|ui| {
                                for row in 0..5 {
                                    let preset_idx = col * 5 + row;
                                    if preset_idx < TlePreset::ALL.len() {
                                        let preset = &TlePreset::ALL[preset_idx];
                                        if let Some((selected, state)) = tab.tle_selections.get_mut(preset) {
                                            ui.horizontal(|ui| {
                                                let color = plane_color(preset_idx);
                                                let rect = ui.allocate_space(egui::vec2(10.0, 10.0)).1;
                                                ui.painter().rect_filled(rect, 2.0, color);

                                                let is_loading = matches!(state, TleLoadState::Loading);
                                                ui.checkbox(selected, preset.label());
                                                if is_loading {
                                                    ui.spinner();
                                                }

                                                #[cfg(not(target_arch = "wasm32"))]
                                                if fetch_requested && *selected {
                                                    if matches!(state, TleLoadState::NotLoaded | TleLoadState::Failed(_)) {
                                                        *state = TleLoadState::Loading;
                                                        let url = preset.url().to_string();
                                                        let preset_copy = *preset;
                                                        let tx = tle_fetch_tx.clone();
                                                        std::thread::spawn(move || {
                                                            let result = fetch_tle_data(&url);
                                                            let _ = tx.send((preset_copy, result));
                                                        });
                                                    }
                                                }
                                            });
                                        }
                                    }
                                }
                            });
                        }
                    });
                });
                ui.separator();
            }

            for (cidx, cons) in tab.constellations.iter_mut().enumerate() {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(cons.preset_name());
                        let hide_btn = if cons.hidden {
                            egui::Button::new(egui::RichText::new("+").color(egui::Color32::WHITE))
                                .fill(egui::Color32::from_rgb(60, 140, 60)).small()
                        } else {
                            egui::Button::new(egui::RichText::new("−").color(egui::Color32::WHITE))
                                .fill(egui::Color32::from_rgb(100, 100, 100)).small()
                        };
                        if ui.add(hide_btn).clicked() {
                            cons.hidden = !cons.hidden;
                        }
                        if num_constellations > 0 {
                            let btn = egui::Button::new(
                                egui::RichText::new("x").color(egui::Color32::WHITE)
                            ).fill(egui::Color32::from_rgb(180, 60, 60)).small();
                            if ui.add(btn).clicked() {
                                const_to_remove = Some(cidx);
                            }
                        }
                        if self.single_color_per_constellation {
                            ui.label(format!("({})", color_name(cons.color_offset)));
                        }
                    });

                    {
                    ui.horizontal(|ui| {
                        let mut sats = cons.sats_per_plane as i32;
                        let mut planes = cons.num_planes as i32;
                        ui.label("Sats:");
                        let sats_resp = ui.add(egui::DragValue::new(&mut sats).range(1..=100));
                        ui.label("Orbits:");
                        let planes_resp = ui.add(egui::DragValue::new(&mut planes).range(1..=100));
                        if sats > 0 && planes > 0 {
                            cons.sats_per_plane = sats as usize;
                            cons.num_planes = planes as usize;
                        }
                        if sats_resp.changed() || planes_resp.changed() {
                            cons.preset = Preset::None;
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Alt:");
                        let alt_resp = ui.add(egui::DragValue::new(&mut cons.altitude_km).range(100.0..=50000.0).suffix(" km"));
                        let orbit_label = if cons.altitude_km < 450.0 { "VLEO" }
                            else if cons.altitude_km < 2000.0 { "LEO" }
                            else if cons.altitude_km < 35000.0 { "MEO" }
                            else { "GEO" };
                        egui::ComboBox::from_id_salt(format!("orbit_{}_{}", tab_name, cidx))
                            .selected_text(orbit_label)
                            .width(50.0)
                            .show_ui(ui, |ui| {
                                if ui.selectable_label(orbit_label == "VLEO", "VLEO").clicked() {
                                    cons.altitude_km = 350.0;
                                    cons.preset = Preset::None;
                                }
                                if ui.selectable_label(orbit_label == "LEO", "LEO").clicked() {
                                    cons.altitude_km = 1080.0;
                                    cons.preset = Preset::None;
                                }
                                if ui.selectable_label(orbit_label == "MEO", "MEO").clicked() {
                                    cons.altitude_km = 18893.0;
                                    cons.preset = Preset::None;
                                }
                                if ui.selectable_label(orbit_label == "GEO", "GEO").clicked() {
                                    cons.altitude_km = 35786.0;
                                    cons.preset = Preset::None;
                                }
                            });
                        if alt_resp.changed() {
                            cons.preset = Preset::None;
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Inc:");
                        let inc_resp = ui.add(egui::DragValue::new(&mut cons.inclination).range(0.0..=180.0).suffix("°"));
                        if inc_resp.changed() {
                            cons.preset = Preset::None;
                        }
                        ui.label("F:");
                        let max_f = (cons.num_planes - 1).max(1) as f64;
                        let phase_resp = ui.add(egui::DragValue::new(&mut cons.phasing).range(0.0..=max_f).speed(0.1));
                        if phase_resp.changed() {
                            cons.preset = Preset::None;
                        }
                    });

                    ui.horizontal(|ui| {
                        let old_type = cons.walker_type;
                        ui.selectable_value(&mut cons.walker_type, WalkerType::Delta, "Delta");
                        ui.selectable_value(&mut cons.walker_type, WalkerType::Star, "Star");
                        ui.label(format!("({} sats)", cons.total_sats()));
                        if cons.walker_type != old_type {
                            cons.preset = Preset::None;
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Preset:");
                        egui::ComboBox::from_id_salt(format!("preset_{}_{}", tab_name, cidx))
                            .selected_text(cons.preset_name())
                            .show_ui(ui, |ui| {
                                if ui.selectable_label(cons.preset == Preset::Starlink, "Starlink").clicked() {
                                    cons.sats_per_plane = 22; cons.num_planes = 72;
                                    cons.altitude_km = 550.0; cons.inclination = 53.0;
                                    cons.walker_type = WalkerType::Delta; cons.phasing = 1.0;
                                    cons.preset = Preset::Starlink;
                                }
                                if ui.selectable_label(cons.preset == Preset::OneWeb, "OneWeb").clicked() {
                                    cons.sats_per_plane = 54; cons.num_planes = 12;
                                    cons.altitude_km = 1200.0; cons.inclination = 87.9;
                                    cons.walker_type = WalkerType::Star; cons.phasing = 1.0;
                                    cons.preset = Preset::OneWeb;
                                }
                                if ui.selectable_label(cons.preset == Preset::Iridium, "Iridium").clicked() {
                                    cons.sats_per_plane = 11; cons.num_planes = 6;
                                    cons.altitude_km = 780.0; cons.inclination = 86.4;
                                    cons.walker_type = WalkerType::Star; cons.phasing = 2.0;
                                    cons.preset = Preset::Iridium;
                                }
                                if ui.selectable_label(cons.preset == Preset::Kuiper, "Kuiper").clicked() {
                                    cons.sats_per_plane = 34; cons.num_planes = 34;
                                    cons.altitude_km = 630.0; cons.inclination = 51.9;
                                    cons.walker_type = WalkerType::Delta; cons.phasing = 1.0;
                                    cons.preset = Preset::Kuiper;
                                }
                                if ui.selectable_label(cons.preset == Preset::Iris2, "Iris²").clicked() {
                                    cons.sats_per_plane = 22; cons.num_planes = 12;
                                    cons.altitude_km = 1200.0; cons.inclination = 87.0;
                                    cons.walker_type = WalkerType::Star; cons.phasing = 1.0;
                                    cons.preset = Preset::Iris2;
                                }
                                if ui.selectable_label(cons.preset == Preset::Telesat, "Telesat").clicked() {
                                    cons.sats_per_plane = 13; cons.num_planes = 6;
                                    cons.altitude_km = 1015.0; cons.inclination = 98.98;
                                    cons.walker_type = WalkerType::Star; cons.phasing = 1.0;
                                    cons.preset = Preset::Telesat;
                                }
                            });
                    });
                    }
                });
                ui.separator();
            }

            if ui.button("[+] Add constellation").clicked() {
                const_to_remove = Some(usize::MAX);
            }
        });

        if let Some(cidx) = const_to_remove {
            if cidx == usize::MAX {
                self.tabs[tab_idx].add_constellation();
            } else {
                self.tabs[tab_idx].constellations.remove(cidx);
            }
        }

        ui.separator();

        let tab = &self.tabs[tab_idx];
        let planet_radius = tab.celestial_body.radius_km();
        let planet_mu = tab.celestial_body.mu();
        let celestial_body = tab.celestial_body;
        let skin = tab.skin;
        let tab_name = tab.name.clone();

        let mut constellations_data: Vec<_> = tab.constellations.iter()
            .enumerate()
            .filter(|(_, c)| !c.hidden)
            .map(|(orig_idx, c)| {
                let wc = c.constellation(planet_radius, planet_mu);
                let pos = wc.satellite_positions(self.time);
                (wc, pos, c.color_offset, false, orig_idx)
            })
            .collect();

        let tle_positions = tab.tle_satellite_positions(self.time);
        if !tle_positions.is_empty() {
            let tle_wc = WalkerConstellation {
                walker_type: WalkerType::Delta,
                total_sats: tle_positions.len(),
                num_planes: 1,
                altitude_km: 550.0,
                inclination_deg: 0.0,
                phasing: 0.0,
                planet_radius,
                planet_mu,
            };
            let tle_color_offset = 0;
            constellations_data.push((tle_wc, tle_positions, tle_color_offset, true, usize::MAX));
        }

        let available = ui.available_size();
        let use_horizontal = single_tab && self.show_torus && !self.show_ground_track;

        if use_horizontal {
            let half_width = (available.x - 15.0) / 2.0;
            let view_height = available.y - 20.0;
            let view_size = half_width.min(view_height);

            let show_orbits = self.show_orbits;
            let show_axes = self.show_axes;
            let show_coverage = self.show_coverage;
            let coverage_angle = self.coverage_angle;
            let rotation = self.rotation;
            let zoom = self.zoom;
            let sat_radius = self.sat_radius;
            let show_links = self.show_links;
            let show_intra_links = self.show_intra_links;
            let hide_behind_earth = self.hide_behind_earth;
            let single_color = self.single_color_per_constellation;
            let dark_mode = self.dark_mode;
            let show_routing_paths = self.show_routing_paths;
            let show_manhattan_path = self.show_manhattan_path;
            let show_shortest_path = self.show_shortest_path;
            let render_planet = self.render_planet;
            let planet_handle = self.planet_image_handles.get(&(celestial_body, skin));
            let time = self.time;
            let torus_rotation = self.torus_rotation;
            let torus_zoom = self.torus_zoom;

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    let tab = &mut self.tabs[tab_idx];
                    let (rot, new_zoom) = draw_3d_view(
                        ui,
                        &tab_name,
                        &constellations_data,
                        show_orbits,
                        show_axes,
                        show_coverage,
                        coverage_angle,
                        rotation,
                        half_width,
                        view_size,
                        planet_handle,
                        zoom,
                        sat_radius,
                        show_links,
                        show_intra_links,
                        hide_behind_earth,
                        single_color,
                        dark_mode,
                        &mut tab.pending_cameras,
                        &mut self.camera_id_counter,
                        &mut tab.satellite_cameras,
                        &mut tab.cameras_to_remove,
                        show_routing_paths,
                        show_manhattan_path,
                        show_shortest_path,
                        planet_radius,
                        render_planet,
                    );
                    self.rotation = rot;
                    self.zoom = new_zoom;
                });

                ui.add_space(5.0);

                ui.vertical(|ui| {
                    let tab = &mut self.tabs[tab_idx];
                    let (trot, tzoom) = draw_torus(
                        ui,
                        &format!("torus_{}", tab_name),
                        &constellations_data,
                        time,
                        torus_rotation,
                        half_width,
                        view_size,
                        sat_radius,
                        show_links,
                        single_color,
                        torus_zoom,
                        &mut tab.satellite_cameras,
                        show_routing_paths,
                        show_manhattan_path,
                        show_shortest_path,
                        planet_radius,
                        &mut tab.pending_cameras,
                        &mut self.camera_id_counter,
                        &mut tab.cameras_to_remove,
                    );
                    self.torus_rotation = trot;
                    self.torus_zoom = tzoom;
                });
            });
        } else {
            let viz_width = available.x - 10.0;
            let available_for_views = available.y - 20.0;

            let has_secondary = self.show_torus || self.show_ground_track;
            let separator_height = if has_secondary { 8.0 } else { 0.0 };

            let earth_height = if has_secondary {
                (available_for_views - separator_height) * self.vertical_split
            } else {
                available_for_views
            }.min(viz_width);

            let secondary_height = if has_secondary {
                (available_for_views - separator_height) * (1.0 - self.vertical_split)
            } else {
                0.0
            };

            let show_orbits = self.show_orbits;
            let show_axes = self.show_axes;
            let show_coverage = self.show_coverage;
            let coverage_angle = self.coverage_angle;
            let rotation = self.rotation;
            let zoom = self.zoom;
            let sat_radius = self.sat_radius;
            let show_links = self.show_links;
            let show_intra_links = self.show_intra_links;
            let hide_behind_earth = self.hide_behind_earth;
            let single_color = self.single_color_per_constellation;
            let dark_mode = self.dark_mode;
            let show_routing_paths = self.show_routing_paths;
            let show_manhattan_path = self.show_manhattan_path;
            let show_shortest_path = self.show_shortest_path;
            let render_planet = self.render_planet;
            let planet_handle = self.planet_image_handles.get(&(celestial_body, skin));

            let tab = &mut self.tabs[tab_idx];
            let (rot, new_zoom) = draw_3d_view(
                ui,
                &tab_name,
                &constellations_data,
                show_orbits,
                show_axes,
                show_coverage,
                coverage_angle,
                rotation,
                viz_width,
                earth_height,
                planet_handle,
                zoom,
                sat_radius,
                show_links,
                show_intra_links,
                hide_behind_earth,
                single_color,
                dark_mode,
                &mut tab.pending_cameras,
                &mut self.camera_id_counter,
                &mut tab.satellite_cameras,
                &mut tab.cameras_to_remove,
                show_routing_paths,
                show_manhattan_path,
                show_shortest_path,
                planet_radius,
                render_planet,
            );
            self.rotation = rot;
            self.zoom = new_zoom;

            if has_secondary {
                let separator_rect = ui.available_rect_before_wrap();
                let separator_rect = egui::Rect::from_min_size(
                    separator_rect.min,
                    egui::vec2(viz_width, separator_height),
                );
                let response = ui.allocate_rect(separator_rect, egui::Sense::drag());

                ui.painter().rect_filled(
                    separator_rect,
                    0.0,
                    if response.hovered() || response.dragged() {
                        egui::Color32::WHITE
                    } else {
                        egui::Color32::from_rgb(200, 200, 200)
                    },
                );
                ui.painter().line_segment(
                    [separator_rect.center() - egui::vec2(20.0, 0.0),
                     separator_rect.center() + egui::vec2(20.0, 0.0)],
                    egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 100, 100)),
                );

                if response.dragged() {
                    let delta = response.drag_delta().y / available_for_views;
                    self.vertical_split = (self.vertical_split + delta).clamp(0.2, 0.9);
                }

                if response.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
                }
            }

            if self.show_torus && self.show_ground_track {
                let torus_height = secondary_height * 0.6;
                let time = self.time;
                let torus_rotation = self.torus_rotation;
                let sat_radius = self.sat_radius;
                let show_links = self.show_links;
                let single_color = self.single_color_per_constellation;
                let torus_zoom = self.torus_zoom;
                let show_routing_paths = self.show_routing_paths;
                let show_manhattan_path = self.show_manhattan_path;
                let show_shortest_path = self.show_shortest_path;
                let tab = &mut self.tabs[tab_idx];
                let (trot, tzoom) = draw_torus(
                    ui,
                    &format!("torus_{}", tab_name),
                    &constellations_data,
                    time,
                    torus_rotation,
                    viz_width,
                    torus_height,
                    sat_radius,
                    show_links,
                    single_color,
                    torus_zoom,
                    &mut tab.satellite_cameras,
                    show_routing_paths,
                    show_manhattan_path,
                    show_shortest_path,
                    planet_radius,
                    &mut tab.pending_cameras,
                    &mut self.camera_id_counter,
                    &mut tab.cameras_to_remove,
                );
                self.torus_rotation = trot;
                self.torus_zoom = tzoom;

                let ground_height = secondary_height * 0.4;
                draw_ground_track(
                    ui,
                    &format!("ground_{}", tab_name),
                    &constellations_data,
                    viz_width,
                    ground_height,
                    self.sat_radius,
                    self.single_color_per_constellation,
                );
            } else if self.show_torus {
                let time = self.time;
                let torus_rotation = self.torus_rotation;
                let sat_radius = self.sat_radius;
                let show_links = self.show_links;
                let single_color = self.single_color_per_constellation;
                let torus_zoom = self.torus_zoom;
                let show_routing_paths = self.show_routing_paths;
                let show_manhattan_path = self.show_manhattan_path;
                let show_shortest_path = self.show_shortest_path;
                let tab = &mut self.tabs[tab_idx];
                let (trot, tzoom) = draw_torus(
                    ui,
                    &format!("torus_{}", tab_name),
                    &constellations_data,
                    time,
                    torus_rotation,
                    viz_width,
                    secondary_height,
                    sat_radius,
                    show_links,
                    single_color,
                    torus_zoom,
                    &mut tab.satellite_cameras,
                    show_routing_paths,
                    show_manhattan_path,
                    show_shortest_path,
                    planet_radius,
                    &mut tab.pending_cameras,
                    &mut self.camera_id_counter,
                    &mut tab.cameras_to_remove,
                );
                self.torus_rotation = trot;
                self.torus_zoom = tzoom;
            } else if self.show_ground_track {
                draw_ground_track(
                    ui,
                    &format!("ground_{}", tab_name),
                    &constellations_data,
                    viz_width,
                    secondary_height,
                    self.sat_radius,
                    self.single_color_per_constellation,
                );
            }
        }
    }

    fn show_settings(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.dark_mode, "Dark mode");
        let mut stop_time = !self.animate;
        ui.checkbox(&mut stop_time, "Stop time");
        self.animate = !stop_time;

        ui.horizontal(|ui| {
            ui.label("Speed:");
            ui.add(egui::DragValue::new(&mut self.speed).range(0.1..=1000.0).speed(1.0));
        });
        let start = self.start_timestamp;
        let real_timestamp = start + Duration::seconds(self.real_time as i64);
        ui.horizontal(|ui| {
            ui.label("Time:");
            ui.add(
                egui::DragValue::new(&mut self.time)
                    .speed(1.0)
                    .custom_formatter(|secs, _| {
                        let ts = start + Duration::seconds(secs as i64);
                        ts.format("%H:%M:%S %d/%m/%Y").to_string()
                    })
                    .custom_parser(|input| {
                        if let Ok(secs) = input.parse::<f64>() {
                            return Some(secs);
                        }
                        let formats = [
                            "%H:%M:%S %d/%m/%Y",
                            "%H:%M %d/%m/%Y",
                            "%d/%m/%Y %H:%M:%S",
                            "%d/%m/%Y %H:%M",
                            "%d/%m/%Y",
                        ];
                        for fmt in formats {
                            if let Ok(parsed) = chrono::NaiveDateTime::parse_from_str(input, fmt) {
                                let parsed_utc = parsed.and_utc();
                                let diff = parsed_utc.signed_duration_since(start);
                                return Some(diff.num_seconds() as f64);
                            }
                        }
                        if let Ok(parsed) = chrono::NaiveDate::parse_from_str(input, "%d/%m/%Y") {
                            let parsed_utc = parsed.and_hms_opt(0, 0, 0).unwrap().and_utc();
                            let diff = parsed_utc.signed_duration_since(start);
                            return Some(diff.num_seconds() as f64);
                        }
                        None
                    })
            );
        });
        ui.label(format!("Real: {}", real_timestamp.format("%H:%M:%S %d/%m/%Y")));
        if ui.button("Sync time").clicked() {
            self.time = self.real_time;
        }

        ui.checkbox(&mut self.show_axes, "Show axes");
        ui.checkbox(&mut self.show_coverage, "Show coverage");
        if self.show_coverage {
            ui.horizontal(|ui| {
                ui.label("Angle:");
                ui.add(egui::DragValue::new(&mut self.coverage_angle)
                    .range(0.5..=70.0)
                    .speed(0.1)
                    .max_decimals(1)
                    .suffix("°"));
            });
        }
        ui.checkbox(&mut self.show_ground_track, "Show ground");
        ui.checkbox(&mut self.show_camera_windows, "Show camera windows");
        ui.checkbox(&mut self.hide_behind_earth, "Hide behind Earth");
        ui.checkbox(&mut self.render_planet, "Render planet");
        ui.checkbox(&mut self.single_color_per_constellation, "Monochrome");
        ui.checkbox(&mut self.follow_satellite, "Follow satellite");

        ui.add_space(5.0);
        ui.label(egui::RichText::new("Simulation options").strong());
        ui.checkbox(&mut self.show_orbits, "Show orbits");
        ui.checkbox(&mut self.show_intra_links, "Intra-plane links");
        ui.checkbox(&mut self.show_links, "Inter-plane links");
        ui.checkbox(&mut self.show_routing_paths, "Show routing paths");
        if self.show_routing_paths {
            ui.indent("routing_opts", |ui| {
                ui.checkbox(&mut self.show_manhattan_path, "Manhattan (red)");
                ui.checkbox(&mut self.show_shortest_path, "Shortest distance (green)");
            });
        }
        ui.checkbox(&mut self.show_torus, "Show torus");

        ui.add_space(10.0);

        ui.horizontal(|ui| {
            ui.label("Zoom:");
            ui.add(egui::DragValue::new(&mut self.zoom).range(0.01..=20.0).speed(0.1));
            ui.label("Sat:");
            ui.add(egui::DragValue::new(&mut self.sat_radius).range(1.0..=15.0).speed(0.1));
        });

        ui.add_space(10.0);

        ui.horizontal(|ui| {
            if ui.button("N/S view").clicked() {
                self.rotation = Matrix3::identity();
            }
            if ui.button("E/W view").clicked() {
                self.rotation = Matrix3::new(
                    1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0,
                    0.0, -1.0, 0.0,
                );
            }
        });

        if ui.button("Reset view").clicked() {
            self.rotation = Matrix3::identity();
            self.torus_rotation = Matrix3::new(
                1.0, 0.0, 0.0,
                0.0, 0.0, -1.0,
                0.0, 1.0, 0.0,
            );
            self.zoom = 1.0;
        }

        ui.add_space(10.0);
        ui.separator();
        ui.label("Delta: RAAN spread 360°");
        ui.label("Star: RAAN spread 180°");
        ui.add_space(5.0);
        ui.label("Drag 3D views to rotate");
        ui.add_space(5.0);
        ui.label("Earth textures: Solar System Scope (CC-BY)");
    }

    #[allow(unused_variables)]
    fn load_texture_for_body(&mut self, body: CelestialBody, skin: Skin, ctx: &egui::Context) {
        let key = (body, skin);
        if self.planet_textures.contains_key(&key) {
            return;
        }

        let filename = match skin.filename(body) {
            Some(f) => f,
            None => return,
        };
        self.texture_load_state = TextureLoadState::Loading;
        self.pending_body = Some(key);

        #[cfg(not(target_arch = "wasm32"))]
        {
            match std::fs::read(filename) {
                Ok(bytes) => match EarthTexture::from_bytes(&bytes) {
                    Ok(texture) => {
                        let texture = Arc::new(texture);
                        self.planet_textures.insert(key, texture.clone());
                        self.texture_load_state = TextureLoadState::Loaded(texture);
                        self.planet_image_handles.remove(&key);
                    }
                    Err(e) => self.texture_load_state = TextureLoadState::Failed(e),
                },
                Err(e) => self.texture_load_state = TextureLoadState::Failed(e.to_string()),
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            let ctx = ctx.clone();
            let filename = filename.to_string();
            wasm_bindgen_futures::spawn_local(async move {
                let result = fetch_texture(&filename).await;
                TEXTURE_RESULT.with(|cell| {
                    *cell.borrow_mut() = Some(result);
                });
                ctx.request_repaint();
            });
        }
    }
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static TEXTURE_RESULT: std::cell::RefCell<Option<Result<EarthTexture, String>>> = std::cell::RefCell::new(None);
}

#[cfg(target_arch = "wasm32")]
async fn fetch_texture(url: &str) -> Result<EarthTexture, String> {
    use wasm_bindgen::JsCast as _;
    use web_sys::{Request, RequestInit, RequestMode, Response};

    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);

    let request = Request::new_with_str_and_init(url, &opts)
        .map_err(|e| format!("Failed to create request: {:?}", e))?;

    let window = web_sys::window().ok_or("No window")?;
    let resp_value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|e| format!("Fetch failed: {:?}", e))?;

    let resp: Response = resp_value.dyn_into()
        .map_err(|_| "Response is not a Response")?;

    if !resp.ok() {
        return Err(format!("HTTP error: {}", resp.status()));
    }

    let array_buffer = wasm_bindgen_futures::JsFuture::from(
        resp.array_buffer().map_err(|e| format!("Failed to get array buffer: {:?}", e))?
    )
    .await
    .map_err(|e| format!("Failed to read response: {:?}", e))?;

    let uint8_array = js_sys::Uint8Array::new(&array_buffer);
    let bytes: Vec<u8> = uint8_array.to_vec();

    EarthTexture::from_bytes(&bytes)
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_visuals(if self.dark_mode {
            egui::Visuals::dark()
        } else {
            egui::Visuals::light()
        });

        let bodies_needed: Vec<(CelestialBody, Skin)> = self.tabs.iter()
            .map(|tab| (tab.celestial_body, tab.skin))
            .collect();
        for (body, skin) in &bodies_needed {
            self.load_texture_for_body(*body, *skin, ctx);
        }

        #[cfg(not(target_arch = "wasm32"))]
        while let Ok((preset, result)) = self.tle_fetch_rx.try_recv() {
            for tab in &mut self.tabs {
                if let Some((_, state)) = tab.tle_selections.get_mut(&preset) {
                    if matches!(state, TleLoadState::Loading) {
                        *state = match result.clone() {
                            Ok(satellites) => TleLoadState::Loaded {
                                satellites,
                                loaded_at: std::time::Instant::now(),
                            },
                            Err(e) => TleLoadState::Failed(e),
                        };
                    }
                }
            }
        }

        let earth_radius = CelestialBody::Earth.radius_km();
        let max_planet_radius = bodies_needed.iter()
            .map(|(b, _)| b.radius_km())
            .fold(earth_radius, |a, b| a.max(b));
        if max_planet_radius > self.last_max_planet_radius {
            let ideal_zoom = earth_radius / max_planet_radius;
            self.zoom = ideal_zoom.clamp(0.01, 1.0);
            self.last_max_planet_radius = max_planet_radius;
        } else if max_planet_radius < self.last_max_planet_radius {
            self.last_max_planet_radius = max_planet_radius;
        }

        let dt = ctx.input(|i| i.stable_dt) as f64;
        self.real_time += dt;

        ctx.request_repaint();
        if self.animate {
            self.time += dt * self.speed;
        }

        let sim_time = self.start_timestamp + Duration::seconds(self.time as i64);
        let gmst = greenwich_mean_sidereal_time(sim_time);
        let cos_a = gmst.cos();
        let sin_a = gmst.sin();
        let planet_y_rotation = Matrix3::new(
            cos_a, 0.0, sin_a,
            0.0, 1.0, 0.0,
            -sin_a, 0.0, cos_a,
        );
        let combined_rotation = self.rotation * planet_y_rotation;

        if self.follow_satellite {
            if let Some(tab) = self.tabs.get(self.active_tab) {
                if tab.satellite_cameras.len() == 1 {
                    let cam = &tab.satellite_cameras[0];
                    if let Some(cons) = tab.constellations.get(cam.constellation_idx) {
                        let planet_radius = tab.celestial_body.radius_km();
                        let planet_mu = tab.celestial_body.mu();
                        let wc = cons.constellation(planet_radius, planet_mu);
                        let positions = wc.satellite_positions(self.time);
                        if let Some(sat) = positions.iter().find(|s| s.plane == cam.plane && s.sat_index == cam.sat_index) {
                            let forward: Vector3<f64> = Vector3::new(sat.x, sat.y, sat.z).normalize();

                            let raan_spread = match cons.walker_type {
                                WalkerType::Delta => 2.0 * PI,
                                WalkerType::Star => PI,
                            };
                            let raan = raan_spread * cam.plane as f64 / cons.num_planes as f64;
                            let inc = cons.inclination.to_radians();
                            let orbital_normal: Vector3<f64> = Vector3::new(
                                raan.sin() * inc.sin(),
                                inc.cos(),
                                -raan.cos() * inc.sin(),
                            );
                            let velocity_dir = orbital_normal.cross(&forward).normalize();
                            let up = -velocity_dir;
                            let right = up.cross(&forward).normalize();

                            self.rotation = Matrix3::new(
                                right.x, right.y, right.z,
                                up.x, up.y, up.z,
                                forward.x, forward.y, forward.z,
                            );
                        }
                    }
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        TEXTURE_RESULT.with(|cell| {
            if let Some(result) = cell.borrow_mut().take() {
                if let Some(body) = self.pending_body {
                    match result {
                        Ok(texture) => {
                            let texture = Arc::new(texture);
                            self.planet_textures.insert(body, texture.clone());
                            self.texture_load_state = TextureLoadState::Loaded(texture);
                            self.planet_image_handles.remove(&body);
                        }
                        Err(e) => {
                            self.texture_load_state = TextureLoadState::Failed(e);
                        }
                    }
                }
            }
        });

        let rotation_changed = self.last_rotation.map_or(true, |r| r != combined_rotation);
        let resolution_changed = self.last_resolution != self.earth_resolution;

        for key in &bodies_needed {
            let texture_missing = !self.planet_image_handles.contains_key(key);
            let need_rerender = rotation_changed || resolution_changed || texture_missing;
            if need_rerender {
                if let Some(texture) = self.planet_textures.get(key) {
                    let image = texture.render_sphere(self.earth_resolution, &combined_rotation);
                    let handle = ctx.load_texture(
                        &format!("planet_{:?}_{:?}", key.0, key.1),
                        image,
                        egui::TextureOptions::LINEAR,
                    );
                    self.planet_image_handles.insert(*key, handle);
                }
            }
        }
        if rotation_changed {
            self.last_rotation = Some(combined_rotation);
        }
        if resolution_changed {
            self.last_resolution = self.earth_resolution;
        }

        let is_mobile = ctx.input(|i| i.viewport().inner_rect.map_or(600.0, |r| r.width())) < 600.0;

        if is_mobile {
            egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let menu_label = if self.menu_open { "Settings \u{25B2}" } else { "Settings \u{25BC}" };
                    if ui.button(menu_label).clicked() {
                        self.menu_open = !self.menu_open;
                    }
                    ui.heading("LEO Viz");
                });
                if self.menu_open {
                    ui.separator();
                    self.show_settings(ui);
                }
            });
        } else if self.show_side_panel {
            egui::SidePanel::left("global_controls").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("LEO Viz");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("-").clicked() {
                            self.show_side_panel = false;
                        }
                    });
                });
                if ui.button("Info").clicked() {
                    self.show_info = !self.show_info;
                }
                ui.separator();
                ui.heading("Display Settings");
                ui.separator();
                self.show_settings(ui);
            });
        } else {
            egui::SidePanel::left("collapsed_panel")
                .max_width(30.0)
                .resizable(false)
                .show(ctx, |ui| {
                    if ui.button("+").clicked() {
                        self.show_side_panel = true;
                    }
                });
        }

        if self.show_info {
            egui::Window::new("Info")
                .open(&mut self.show_info)
                .default_width(400.0)
                .show(ctx, |ui| {
                    ui.heading("Celestial Bodies");
                    egui::Grid::new("bodies_grid").striped(true).show(ui, |ui| {
                        ui.strong("Body");
                        ui.strong("Radius (km)");
                        ui.strong("μ (km³/s²)");
                        ui.end_row();
                        for body in CelestialBody::ALL {
                            ui.label(body.label());
                            ui.label(format!("{:.0}", body.radius_km()));
                            ui.label(format!("{:.0}", body.mu()));
                            ui.end_row();
                        }
                    });

                    ui.add_space(10.0);
                    ui.heading("Orbital Mechanics");
                    ui.label("μ = G × M (standard gravitational parameter)");
                    ui.separator();
                    ui.label("Orbital velocity:");
                    ui.monospace("  v = √(μ / r)");
                    ui.separator();
                    ui.label("Orbital period:");
                    ui.monospace("  T = 2π √(r³ / μ)");
                    ui.separator();
                    ui.label("Where:");
                    ui.label("  r = orbital radius (planet radius + altitude)");
                    ui.label("  μ = gravitational parameter");

                    ui.add_space(10.0);
                    ui.heading("Walker Constellation");
                    ui.label("Notation: i:T/P/F");
                    ui.label("  i = inclination (degrees)");
                    ui.label("  T = total satellites");
                    ui.label("  P = number of orbital planes");
                    ui.label("  F = phasing factor (0 to P-1)");
                    ui.separator();
                    ui.label("Types:");
                    ui.label("  Delta: planes spread 360° (co-rotating)");
                    ui.label("  Star: planes spread 180° (counter-rotating seam)");
                    ui.separator();
                    ui.label("Phasing offset per plane:");
                    ui.monospace("  Δ = F × 360° / T");

                    ui.add_space(10.0);
                    ui.heading("Link Latency");
                    ui.label("Speed of light: 299,792 km/s");
                    ui.separator();
                    ui.label("One-way latency:");
                    ui.monospace("  t = distance / c");
                    ui.separator();
                    ui.label("Intra-plane distance (between adjacent sats):");
                    ui.monospace("  d = 2r × sin(π / sats_per_plane)");
                    ui.separator();
                    ui.label("Inter-plane distance depends on inclination");
                    ui.label("and relative orbital positions.");

                    ui.add_space(10.0);
                    ui.heading("Satellite Constellations");
                    egui::Grid::new("constellations_grid").striped(true).show(ui, |ui| {
                        ui.strong("Name");
                        ui.strong("Sats");
                        ui.strong("Planes");
                        ui.strong("Alt (km)");
                        ui.strong("Inc (°)");
                        ui.end_row();

                        ui.label("Starlink");
                        ui.label("22×72");
                        ui.label("72");
                        ui.label("550");
                        ui.label("53");
                        ui.end_row();

                        ui.label("OneWeb");
                        ui.label("49×36");
                        ui.label("36");
                        ui.label("1200");
                        ui.label("87.9");
                        ui.end_row();

                        ui.label("Iridium");
                        ui.label("11×6");
                        ui.label("6");
                        ui.label("780");
                        ui.label("86.4");
                        ui.end_row();

                        ui.label("Kuiper");
                        ui.label("34×34");
                        ui.label("34");
                        ui.label("630");
                        ui.label("51.9");
                        ui.end_row();

                        ui.label("Iris²");
                        ui.label("22×12");
                        ui.label("12");
                        ui.label("7800");
                        ui.label("75");
                        ui.end_row();

                        ui.label("Telesat");
                        ui.label("13×6");
                        ui.label("6");
                        ui.label("1015");
                        ui.label("98.98");
                        ui.end_row();
                    });

                    ui.add_space(5.0);
                    ui.label("Live TLE sources from CelesTrak:");
                    ui.label("  Starlink, OneWeb, Iridium, Globalstar,");
                    ui.label("  Orbcomm, GPS, Galileo, GLONASS, Beidou");
                });
        }

        let num_tabs = self.tabs.len();
        let single_tab = num_tabs == 1;
        let mut tab_to_remove: Option<usize> = None;
        let mut add_tab = false;

        egui::CentralPanel::default().show(ctx, |ui| {
            let available_rect = ui.available_rect_before_wrap();
            let tab_width = available_rect.width() / num_tabs as f32;
            let separator_width = 4.0;

            for tab_idx in 0..num_tabs {
                let x_offset = tab_idx as f32 * tab_width;
                let inner_width = if tab_idx < num_tabs - 1 {
                    tab_width - separator_width
                } else {
                    tab_width
                };

                let tab_rect = egui::Rect::from_min_size(
                    egui::pos2(available_rect.min.x + x_offset, available_rect.min.y),
                    egui::vec2(inner_width, available_rect.height()),
                );

                let tab_name = self.tabs[tab_idx].name.clone();

                let is_last_tab = tab_idx == num_tabs - 1;
                ui.scope_builder(egui::UiBuilder::new().max_rect(tab_rect), |ui| {
                    ui.horizontal(|ui| {
                        ui.strong(&tab_name);
                        if is_last_tab {
                            if ui.small_button("+").clicked() {
                                add_tab = true;
                            }
                        }
                        if !single_tab {
                            let btn = egui::Button::new(
                                egui::RichText::new("×").color(egui::Color32::WHITE)
                            ).fill(egui::Color32::from_rgb(180, 60, 60)).small();
                            if ui.add(btn).clicked() {
                                tab_to_remove = Some(tab_idx);
                            }
                        }
                    });
                    ui.separator();
                    self.render_tab_ui(ui, tab_idx, single_tab);
                });

                if tab_idx < num_tabs - 1 {
                    let sep_rect = egui::Rect::from_min_size(
                        egui::pos2(available_rect.min.x + x_offset + inner_width, available_rect.min.y),
                        egui::vec2(separator_width, available_rect.height()),
                    );
                    ui.painter().rect_filled(
                        sep_rect,
                        0.0,
                        egui::Color32::from_rgb(60, 60, 60),
                    );
                }
            }
        });

        if let Some(idx) = tab_to_remove {
            self.remove_tab(idx);
        }
        if add_tab {
            self.add_tab();
        }

        if self.show_camera_windows {
            for tab in &self.tabs {
                let pr = tab.celestial_body.radius_km();
                let pm = tab.celestial_body.mu();
                let texture = self.planet_textures.get(&(tab.celestial_body, tab.skin));

                for camera in &tab.satellite_cameras {
                    let sat_data = tab.constellations.get(camera.constellation_idx).map(|cons| {
                        let wc = cons.constellation(pr, pm);
                        let positions = wc.satellite_positions(self.time);
                        positions.iter()
                            .find(|s| s.plane == camera.plane && s.sat_index == camera.sat_index)
                            .map(|s| (s.lat, s.lon, cons.altitude_km, texture, pr))
                    }).flatten();

                    if let Some((lat, lon, altitude_km, texture, planet_radius)) = sat_data {
                        let win_response = egui::Window::new(format!("{}: {}", tab.name, camera.label))
                            .id(egui::Id::new(format!("sat_cam_{}_{}", tab.name, camera.id)))
                            .title_bar(true)
                            .collapsible(false)
                            .default_size([200.0, 220.0])
                            .show(ctx, |ui| {
                                if let Some(tex) = texture {
                                    draw_satellite_camera(
                                        ui,
                                        camera.id,
                                        lat,
                                        lon,
                                        altitude_km,
                                        self.coverage_angle,
                                        tex,
                                        planet_radius,
                                    );
                                }
                            });

                        if let (Some(screen_pos), Some(win_resp)) = (camera.screen_pos, win_response) {
                            let win_rect = win_resp.response.rect;
                            let win_center = win_rect.left_center();
                            ctx.layer_painter(egui::LayerId::new(
                                egui::Order::Middle,
                                egui::Id::new("sat_lines"),
                            ))
                            .line_segment(
                                [screen_pos, win_center],
                                egui::Stroke::new(1.5, egui::Color32::WHITE),
                            );
                        }
                    }
                }
            }
        }
    }
}

fn draw_satellite_camera(
    ui: &mut egui::Ui,
    camera_id: usize,
    lat: f64,
    lon: f64,
    altitude_km: f64,
    coverage_angle: f64,
    earth_texture: &EarthTexture,
    planet_radius: f64,
) {
    let size = ui.available_size();
    let img_size = size.x.min(size.y - 40.0) as usize;
    if img_size < 10 {
        return;
    }

    let lat_rad = lat.to_radians();
    let lon_rad = lon.to_radians();
    let cone_half_angle = coverage_angle.to_radians();
    let orbit_radius = planet_radius + altitude_km;
    let max_earth_angle = (planet_radius / orbit_radius).acos();
    let earth_central_angle = (orbit_radius * cone_half_angle.sin() / planet_radius).asin();
    let angular_radius = earth_central_angle.min(max_earth_angle);

    let mut pixels = vec![egui::Color32::BLACK; img_size * img_size];

    for py in 0..img_size {
        for px in 0..img_size {
            let nx = (px as f64 / img_size as f64 - 0.5) * 2.0;
            let ny = (py as f64 / img_size as f64 - 0.5) * 2.0;

            let dist = (nx * nx + ny * ny).sqrt();
            if dist > 1.0 {
                continue;
            }

            let angle_from_nadir = dist * angular_radius;
            let azimuth = ny.atan2(nx);

            let clat = (lat_rad.sin() * angle_from_nadir.cos()
                + lat_rad.cos() * angle_from_nadir.sin() * (-azimuth).cos())
            .asin();
            let clon = lon_rad
                + (angle_from_nadir.sin() * (-azimuth).sin())
                    .atan2(lat_rad.cos() * angle_from_nadir.cos()
                        - lat_rad.sin() * angle_from_nadir.sin() * (-azimuth).cos());

            let u = (clon + PI) / (2.0 * PI);
            let v = (PI / 2.0 - clat) / PI;

            let [r, g, b] = earth_texture.sample(u, v);
            pixels[py * img_size + px] = egui::Color32::from_rgb(r, g, b);
        }
    }

    let image = egui::ColorImage {
        size: [img_size, img_size],
        pixels,
        source_size: egui::Vec2::ZERO,
    };
    let texture = ui.ctx().load_texture(
        format!("sat_cam_tex_{}", camera_id),
        image,
        egui::TextureOptions::LINEAR,
    );
    ui.image(&texture);

    ui.horizontal(|ui| {
        ui.label(format!("Lat: {:.1}°", lat));
        ui.label(format!("Lon: {:.1}°", lon));
    });
    ui.label(format!("Alt: {:.0} km", altitude_km));
}

fn compute_manhattan_path(
    src_plane: usize, src_sat: usize,
    dst_plane: usize, dst_sat: usize,
    num_planes: usize, sats_per_plane: usize,
    is_star: bool,
) -> Vec<(usize, usize)> {
    let mut path = vec![(src_plane, src_sat)];

    let (plane_dir, plane_steps) = if is_star {
        if dst_plane >= src_plane {
            (1i32, dst_plane - src_plane)
        } else {
            (-1i32, src_plane - dst_plane)
        }
    } else {
        let plane_diff_fwd = (dst_plane + num_planes - src_plane) % num_planes;
        let plane_diff_bwd = (src_plane + num_planes - dst_plane) % num_planes;
        if plane_diff_fwd <= plane_diff_bwd {
            (1i32, plane_diff_fwd)
        } else {
            (-1i32, plane_diff_bwd)
        }
    };

    let sat_diff_fwd = (dst_sat + sats_per_plane - src_sat) % sats_per_plane;
    let sat_diff_bwd = (src_sat + sats_per_plane - dst_sat) % sats_per_plane;
    let (sat_dir, sat_steps) = if sat_diff_fwd <= sat_diff_bwd {
        (1i32, sat_diff_fwd)
    } else {
        (-1i32, sat_diff_bwd)
    };

    let mut cur_plane = src_plane;
    for _ in 0..plane_steps {
        cur_plane = ((cur_plane as i32 + plane_dir + num_planes as i32) % num_planes as i32) as usize;
        path.push((cur_plane, src_sat));
    }

    let mut cur_sat = src_sat;
    for _ in 0..sat_steps {
        cur_sat = ((cur_sat as i32 + sat_dir + sats_per_plane as i32) % sats_per_plane as i32) as usize;
        path.push((dst_plane, cur_sat));
    }

    path
}

fn compute_shortest_path(
    src_plane: usize, src_sat: usize,
    dst_plane: usize, dst_sat: usize,
    num_planes: usize, sats_per_plane: usize,
    positions: &[SatelliteState],
    is_star: bool,
) -> Vec<(usize, usize)> {
    let mut path = vec![(src_plane, src_sat)];

    let (plane_dir, mut plane_steps_remaining) = if is_star {
        if dst_plane >= src_plane {
            (1i32, dst_plane - src_plane)
        } else {
            (-1i32, src_plane - dst_plane)
        }
    } else {
        let plane_diff_fwd = (dst_plane + num_planes - src_plane) % num_planes;
        let plane_diff_bwd = (src_plane + num_planes - dst_plane) % num_planes;
        if plane_diff_fwd <= plane_diff_bwd {
            (1i32, plane_diff_fwd)
        } else {
            (-1i32, plane_diff_bwd)
        }
    };

    let sat_diff_fwd = (dst_sat + sats_per_plane - src_sat) % sats_per_plane;
    let sat_diff_bwd = (src_sat + sats_per_plane - dst_sat) % sats_per_plane;
    let (sat_dir, mut sat_steps_remaining) = if sat_diff_fwd <= sat_diff_bwd {
        (1i32, sat_diff_fwd)
    } else {
        (-1i32, sat_diff_bwd)
    };

    let get_pos = |plane: usize, sat_idx: usize| -> Option<(f64, f64, f64)> {
        positions.iter()
            .find(|s| s.plane == plane && s.sat_index == sat_idx)
            .map(|s| (s.x, s.y, s.z))
    };

    let distance = |p1: (f64, f64, f64), p2: (f64, f64, f64)| -> f64 {
        let dx = p1.0 - p2.0;
        let dy = p1.1 - p2.1;
        let dz = p1.2 - p2.2;
        (dx * dx + dy * dy + dz * dz).sqrt()
    };

    let mut cur_plane = src_plane;
    let mut cur_sat = src_sat;

    while plane_steps_remaining > 0 || sat_steps_remaining > 0 {
        if plane_steps_remaining == 0 {
            cur_sat = ((cur_sat as i32 + sat_dir + sats_per_plane as i32) % sats_per_plane as i32) as usize;
            sat_steps_remaining -= 1;
            path.push((cur_plane, cur_sat));
            continue;
        }
        if sat_steps_remaining == 0 {
            cur_plane = ((cur_plane as i32 + plane_dir + num_planes as i32) % num_planes as i32) as usize;
            plane_steps_remaining -= 1;
            path.push((cur_plane, cur_sat));
            continue;
        }

        let next_plane = ((cur_plane as i32 + plane_dir + num_planes as i32) % num_planes as i32) as usize;
        let next_sat = ((cur_sat as i32 + sat_dir + sats_per_plane as i32) % sats_per_plane as i32) as usize;

        let cur_pos = get_pos(cur_plane, cur_sat);
        let cross_plane_pos = get_pos(next_plane, cur_sat);
        let within_plane_pos = get_pos(cur_plane, next_sat);
        let cross_plane_after_within = get_pos(next_plane, next_sat);

        match (cur_pos, cross_plane_pos, within_plane_pos, cross_plane_after_within) {
            (Some(cur), Some(cross), Some(within), Some(cross_after)) => {
                let cross_now = distance(cur, cross);
                let cross_after_within = distance(within, cross_after);

                if cross_now <= cross_after_within {
                    cur_plane = next_plane;
                    plane_steps_remaining -= 1;
                } else {
                    cur_sat = next_sat;
                    sat_steps_remaining -= 1;
                }
            }
            _ => {
                cur_plane = next_plane;
                plane_steps_remaining -= 1;
            }
        }
        path.push((cur_plane, cur_sat));
    }

    path
}

fn draw_routing_path(
    plot_ui: &mut egui_plot::PlotUi,
    path: &[(usize, usize)],
    positions: &[SatelliteState],
    rotation: &Matrix3<f64>,
    color: egui::Color32,
    width: f32,
    hide_behind_earth: bool,
    earth_r_sq: f64,
) {
    if path.len() < 2 {
        return;
    }

    for i in 0..(path.len() - 1) {
        let (plane1, sat1) = path[i];
        let (plane2, sat2) = path[i + 1];

        let pos1 = positions.iter().find(|s| s.plane == plane1 && s.sat_index == sat1);
        let pos2 = positions.iter().find(|s| s.plane == plane2 && s.sat_index == sat2);

        if let (Some(p1), Some(p2)) = (pos1, pos2) {
            let (rx1, ry1, rz1) = rotate_point_matrix(p1.x, p1.y, p1.z, rotation);
            let (rx2, ry2, rz2) = rotate_point_matrix(p2.x, p2.y, p2.z, rotation);

            let visible1 = rz1 >= 0.0 || (rx1 * rx1 + ry1 * ry1) >= earth_r_sq;
            let visible2 = rz2 >= 0.0 || (rx2 * rx2 + ry2 * ry2) >= earth_r_sq;

            if hide_behind_earth && !visible1 && !visible2 {
                continue;
            }

            let line_color = if visible1 && visible2 {
                color
            } else {
                egui::Color32::from_rgba_unmultiplied(
                    color.r() / 2, color.g() / 2, color.b() / 2, 150,
                )
            };

            plot_ui.line(
                Line::new("", PlotPoints::new(vec![[rx1, ry1], [rx2, ry2]]))
                    .color(line_color)
                    .width(width),
            );
        }
    }
}

fn draw_3d_view(
    ui: &mut egui::Ui,
    id: &str,
    constellations: &[(WalkerConstellation, Vec<SatelliteState>, usize, bool, usize)],
    show_orbits: bool,
    show_axes: bool,
    show_coverage: bool,
    coverage_angle: f64,
    mut rotation: Matrix3<f64>,
    width: f32,
    height: f32,
    earth_texture: Option<&egui::TextureHandle>,
    mut zoom: f64,
    sat_radius: f32,
    show_links: bool,
    show_intra_links: bool,
    hide_behind_earth: bool,
    single_color: bool,
    dark_mode: bool,
    pending_cameras: &mut Vec<SatelliteCamera>,
    camera_id_counter: &mut usize,
    satellite_cameras: &mut [SatelliteCamera],
    cameras_to_remove: &mut Vec<usize>,
    show_routing_paths: bool,
    show_manhattan_path: bool,
    show_shortest_path: bool,
    planet_radius: f64,
    render_planet: bool,
) -> (Matrix3<f64>, f64) {
    let max_altitude = constellations.iter()
        .map(|(c, _, _, _, _)| c.altitude_km)
        .fold(550.0_f64, |a, b| a.max(b));
    let orbit_radius = planet_radius + max_altitude;
    let axis_len = orbit_radius * 1.05;
    let planet_view_reference = planet_radius * 1.15;
    let margin = planet_view_reference / zoom;

    let plot = Plot::new(id)
        .data_aspect(1.0)
        .width(width)
        .height(height)
        .show_axes(false)
        .show_grid(false)
        .show_x(false)
        .show_y(false)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .cursor_color(egui::Color32::TRANSPARENT);

    let response = plot.show(ui, |plot_ui| {
        plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
            [-margin, -margin],
            [margin, margin],
        ));

        let visual_earth_r = planet_radius * 0.95;
        let earth_r_sq = visual_earth_r * visual_earth_r;

        if show_orbits && !hide_behind_earth {
            for (constellation, _, color_offset, is_tle, _) in constellations {
                if *is_tle { continue; }
                for plane in 0..constellation.num_planes {
                    let orbit_pts = constellation.orbit_points_3d(plane);
                    let color = plane_color(if single_color { *color_offset } else { plane + color_offset });

                    let mut behind_segment: Vec<[f64; 2]> = Vec::new();
                    for &(x, y, z) in &orbit_pts {
                        let (rx, ry, rz) = rotate_point_matrix(x, y, z, &rotation);
                        let occluded = rz < 0.0 && (rx * rx + ry * ry) < earth_r_sq;
                        if occluded {
                            behind_segment.push([rx, ry]);
                        } else if !behind_segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut behind_segment)))
                                    .color(dim_color(color))
                                    .width(1.0),
                            );
                        }
                    }
                    if !behind_segment.is_empty() {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(behind_segment))
                                .color(dim_color(color))
                                .width(1.0),
                        );
                    }
                }
            }
        }

        if !hide_behind_earth {
            for (constellation, positions, color_offset, _is_tle, _) in constellations {
                for plane in 0..constellation.num_planes {
                    let color = plane_color(if single_color { *color_offset } else { plane + color_offset });
                    let pts: PlotPoints = positions
                        .iter()
                        .filter_map(|s| {
                            if s.plane != plane {
                                return None;
                            }
                            let (rx, ry, rz) = rotate_point_matrix(s.x, s.y, s.z, &rotation);
                            if rz < 0.0 {
                                Some([rx, ry])
                            } else {
                                None
                            }
                        })
                        .collect();
                    plot_ui.points(
                        Points::new("", pts)
                            .color(dim_color(color))
                            .radius(sat_radius * 0.8)
                            .filled(true),
                    );
                }
            }
        }

        if render_planet {
            if let Some(tex) = earth_texture {
                let size = egui::Vec2::splat(planet_radius as f32 * 2.0);
                plot_ui.image(PlotImage::new(
                    "",
                    tex,
                    PlotPoint::new(0.0, 0.0),
                    size,
                ));
            } else {
                let earth_pts: PlotPoints = (0..=100)
                    .map(|i| {
                        let theta = 2.0 * PI * i as f64 / 100.0;
                        [planet_radius * theta.cos(), planet_radius * theta.sin()]
                    })
                    .collect();
                plot_ui.polygon(
                    Polygon::new("", earth_pts)
                        .fill_color(egui::Color32::from_rgb(30, 60, 120))
                        .stroke(egui::Stroke::new(2.0, egui::Color32::from_rgb(70, 130, 180))),
                );
            }

            if dark_mode {
                let border_radius = planet_radius * 0.95;
                let border_pts: PlotPoints = (0..=100)
                    .map(|i| {
                        let theta = 2.0 * PI * i as f64 / 100.0;
                        [border_radius * theta.cos(), border_radius * theta.sin()]
                    })
                    .collect();
                plot_ui.line(Line::new("", border_pts).color(egui::Color32::WHITE).width(1.0));
            }
        } else {
            let earth_pts: PlotPoints = (0..=100)
                .map(|i| {
                    let theta = 2.0 * PI * i as f64 / 100.0;
                    [planet_radius * theta.cos(), planet_radius * theta.sin()]
                })
                .collect();
            plot_ui.polygon(
                Polygon::new("", earth_pts)
                    .fill_color(egui::Color32::from_rgb(30, 60, 120))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(50, 100, 150))),
            );
        }

        if show_coverage {
            for (constellation, positions, color_offset, _is_tle, _) in constellations {
                let orbit_radius = planet_radius + constellation.altitude_km;
                let cone_half_angle = coverage_angle.to_radians();
                let max_earth_angle = (planet_radius / orbit_radius).acos();
                let earth_central_angle = (orbit_radius * cone_half_angle.sin() / planet_radius).asin();
                let angular_radius = earth_central_angle.min(max_earth_angle);
                for sat in positions {
                    let lat = sat.lat.to_radians();
                    let lon = sat.lon.to_radians();

                    let coverage_pts: Vec<([f64; 2], bool)> = (0..=32)
                        .map(|i| {
                            let angle = 2.0 * PI * i as f64 / 32.0;

                            let clat = (lat.sin() * angular_radius.cos()
                                + lat.cos() * angular_radius.sin() * angle.cos())
                            .asin();
                            let clon = lon
                                + (angular_radius.sin() * angle.sin())
                                    .atan2(lat.cos() * angular_radius.cos()
                                        - lat.sin() * angular_radius.sin() * angle.cos());

                            let x = planet_radius * clat.cos() * clon.cos();
                            let y = planet_radius * clat.sin();
                            let z = planet_radius * clat.cos() * clon.sin();

                            let (rx, ry, rz) = rotate_point_matrix(x, y, z, &rotation);
                            ([rx, ry], rz >= 0.0)
                        })
                        .collect();

                    let all_visible = coverage_pts.iter().all(|(_, vis)| *vis);
                    let color = plane_color(sat.plane + color_offset);

                    if all_visible {
                        let pts: Vec<[f64; 2]> = coverage_pts.iter().map(|(p, _)| *p).collect();
                        let fill = egui::Color32::from_rgba_unmultiplied(
                            color.r(), color.g(), color.b(), 60
                        );
                        plot_ui.polygon(
                            Polygon::new("", PlotPoints::new(pts))
                                .fill_color(fill)
                                .stroke(egui::Stroke::new(1.0, color)),
                        );
                    } else {
                        let mut segment: Vec<[f64; 2]> = Vec::new();
                        for (pt, visible) in &coverage_pts {
                            if *visible {
                                segment.push(*pt);
                            } else if !segment.is_empty() {
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(std::mem::take(&mut segment)))
                                        .color(color)
                                        .width(1.0),
                                );
                            }
                        }
                        if !segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(segment))
                                    .color(color)
                                    .width(1.0),
                            );
                        }
                    }
                }
            }
        }

        if show_axes {
            let (ep_x, ep_y, _) = rotate_point_matrix(axis_len, 0.0, 0.0, &rotation);
            let (wn_x, wn_y, _) = rotate_point_matrix(-axis_len, 0.0, 0.0, &rotation);
            plot_ui.line(
                Line::new("", PlotPoints::new(vec![[wn_x, wn_y], [ep_x, ep_y]]))
                    .color(egui::Color32::from_rgb(255, 100, 100))
                    .width(1.5),
            );

            let (np_x, np_y, _) = rotate_point_matrix(0.0, axis_len, 0.0, &rotation);
            let (sn_x, sn_y, _) = rotate_point_matrix(0.0, -axis_len, 0.0, &rotation);
            plot_ui.line(
                Line::new("", PlotPoints::new(vec![[sn_x, sn_y], [np_x, np_y]]))
                    .color(egui::Color32::from_rgb(100, 100, 255))
                    .width(1.5),
            );

            let label_offset = axis_len * 1.15;
            let (n_x, n_y, _) = rotate_point_matrix(0.0, label_offset, 0.0, &rotation);
            let (s_x, s_y, _) = rotate_point_matrix(0.0, -label_offset, 0.0, &rotation);
            let (e_x, e_y, _) = rotate_point_matrix(label_offset, 0.0, 0.0, &rotation);
            let (w_x, w_y, _) = rotate_point_matrix(-label_offset, 0.0, 0.0, &rotation);

            plot_ui.text(Text::new("", PlotPoint::new(n_x, n_y), "N").color(egui::Color32::WHITE));
            plot_ui.text(Text::new("", PlotPoint::new(s_x, s_y), "S").color(egui::Color32::WHITE));
            plot_ui.text(Text::new("", PlotPoint::new(e_x, e_y), "E").color(egui::Color32::WHITE));
            plot_ui.text(Text::new("", PlotPoint::new(w_x, w_y), "W").color(egui::Color32::WHITE));
        }

        if show_orbits {
            for (constellation, _, color_offset, is_tle, _) in constellations {
                if *is_tle { continue; }
                for plane in 0..constellation.num_planes {
                    let orbit_pts = constellation.orbit_points_3d(plane);
                    let color = if show_routing_paths {
                        egui::Color32::from_rgb(80, 80, 80)
                    } else {
                        plane_color(if single_color { *color_offset } else { plane + color_offset })
                    };

                    let mut front_segment: Vec<[f64; 2]> = Vec::new();
                    for &(x, y, z) in &orbit_pts {
                        let (rx, ry, rz) = rotate_point_matrix(x, y, z, &rotation);
                        let visible = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;
                        if visible {
                            front_segment.push([rx, ry]);
                        } else if !front_segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut front_segment)))
                                    .color(color)
                                    .width(1.5),
                            );
                        }
                    }
                    if !front_segment.is_empty() {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(front_segment))
                                .color(color)
                                .width(1.5),
                        );
                    }
                }
            }
        }

        if show_links {
            let base_link_color = if show_routing_paths {
                egui::Color32::from_rgb(80, 80, 80)
            } else {
                egui::Color32::from_rgb(200, 200, 200)
            };
            let link_dim = egui::Color32::from_rgba_unmultiplied(50, 50, 60, 80);
            for (_, positions, _, _, _) in constellations {
                for sat in positions {
                    if let Some(neighbor_idx) = sat.neighbor_idx {
                        let neighbor = &positions[neighbor_idx];
                        let (rx1, ry1, rz1) = rotate_point_matrix(sat.x, sat.y, sat.z, &rotation);
                        let (rx2, ry2, rz2) = rotate_point_matrix(neighbor.x, neighbor.y, neighbor.z, &rotation);
                        let visible1 = rz1 >= 0.0 || (rx1 * rx1 + ry1 * ry1) >= earth_r_sq;
                        let visible2 = rz2 >= 0.0 || (rx2 * rx2 + ry2 * ry2) >= earth_r_sq;
                        let both_visible = visible1 && visible2;
                        if hide_behind_earth && !both_visible {
                            continue;
                        }
                        let color = if both_visible { base_link_color } else { link_dim };
                        plot_ui.line(
                            Line::new("", PlotPoints::new(vec![[rx1, ry1], [rx2, ry2]]))
                                .color(color)
                                .width(1.0),
                        );
                    }
                }
            }
        }

        if show_intra_links {
            let base_link_color = if show_routing_paths {
                egui::Color32::from_rgb(80, 80, 80)
            } else {
                egui::Color32::from_rgb(200, 200, 200)
            };
            let link_dim = egui::Color32::from_rgba_unmultiplied(50, 50, 60, 80);
            for (constellation, positions, _, _, _) in constellations {
                let sats_per_plane = constellation.sats_per_plane();
                for plane in 0..constellation.num_planes {
                    let plane_sats: Vec<_> = positions.iter()
                        .filter(|s| s.plane == plane)
                        .collect();
                    for i in 0..plane_sats.len() {
                        let sat = plane_sats[i];
                        let next = plane_sats[(i + 1) % sats_per_plane];
                        let (rx1, ry1, rz1) = rotate_point_matrix(sat.x, sat.y, sat.z, &rotation);
                        let (rx2, ry2, rz2) = rotate_point_matrix(next.x, next.y, next.z, &rotation);
                        let visible1 = rz1 >= 0.0 || (rx1 * rx1 + ry1 * ry1) >= earth_r_sq;
                        let visible2 = rz2 >= 0.0 || (rx2 * rx2 + ry2 * ry2) >= earth_r_sq;
                        let both_visible = visible1 && visible2;
                        if hide_behind_earth && !both_visible {
                            continue;
                        }
                        let color = if both_visible { base_link_color } else { link_dim };
                        plot_ui.line(
                            Line::new("", PlotPoints::new(vec![[rx1, ry1], [rx2, ry2]]))
                                .color(color)
                                .width(1.0),
                        );
                    }
                }
            }
        }

        if show_routing_paths && !satellite_cameras.is_empty() {
            let manhattan_color = egui::Color32::from_rgb(255, 100, 100);
            let shortest_color = egui::Color32::from_rgb(100, 255, 100);

            for (cidx, (constellation, positions, _, _, _)) in constellations.iter().enumerate() {
                let tracked: Vec<_> = satellite_cameras.iter()
                    .filter(|c| c.constellation_idx == cidx)
                    .collect();

                if tracked.len() < 2 {
                    continue;
                }

                let num_planes = constellation.num_planes;
                let sats_per_plane = constellation.sats_per_plane();

                let is_star = constellation.walker_type == WalkerType::Star;

                for i in 0..tracked.len() {
                    for j in (i + 1)..tracked.len() {
                        let src = tracked[i];
                        let dst = tracked[j];

                        let src_sat = positions.iter().find(|s| s.plane == src.plane && s.sat_index == src.sat_index);
                        let dst_sat = positions.iter().find(|s| s.plane == dst.plane && s.sat_index == dst.sat_index);

                        let can_route = match (src_sat, dst_sat) {
                            (Some(_), Some(_)) => {
                                if is_star {
                                    let plane_diff_fwd = (dst.plane + num_planes - src.plane) % num_planes;
                                    let plane_diff_bwd = (src.plane + num_planes - dst.plane) % num_planes;
                                    let crosses_seam = plane_diff_fwd > num_planes / 2 && plane_diff_bwd > num_planes / 2;
                                    !crosses_seam
                                } else {
                                    true
                                }
                            }
                            _ => false,
                        };

                        if !can_route {
                            continue;
                        }

                        if show_manhattan_path {
                            let path = compute_manhattan_path(
                                src.plane, src.sat_index,
                                dst.plane, dst.sat_index,
                                num_planes, sats_per_plane,
                                is_star,
                            );
                            draw_routing_path(
                                plot_ui, &path, positions, &rotation,
                                manhattan_color, 2.5, hide_behind_earth, earth_r_sq,
                            );
                        }

                        if show_shortest_path {
                            let path = compute_shortest_path(
                                src.plane, src.sat_index,
                                dst.plane, dst.sat_index,
                                num_planes, sats_per_plane,
                                positions,
                                is_star,
                            );
                            draw_routing_path(
                                plot_ui, &path, positions, &rotation,
                                shortest_color, 2.0, hide_behind_earth, earth_r_sq,
                            );
                        }
                    }
                }
            }
        }

        for (constellation, positions, color_offset, is_tle, orig_idx) in constellations {
            if *is_tle {
                for sat in positions {
                    let color = plane_color(color_offset + sat.plane);
                    let dim_col = egui::Color32::from_rgba_unmultiplied(
                        color.r() / 2, color.g() / 2, color.b() / 2, 80,
                    );

                    let (rx, ry, rz) = rotate_point_matrix(sat.x, sat.y, sat.z, &rotation);
                    let in_front = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;

                    let bg_color = if dark_mode {
                        egui::Color32::from_rgb(30, 30, 30)
                    } else {
                        egui::Color32::from_rgb(240, 240, 240)
                    };

                    if !hide_behind_earth && !in_front {
                        plot_ui.points(
                            Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                .color(dim_col)
                                .radius(sat_radius * 0.8)
                                .filled(true),
                        );
                        plot_ui.points(
                            Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                .color(bg_color)
                                .radius(sat_radius * 0.4)
                                .filled(true),
                        );
                    }
                    if in_front {
                        plot_ui.points(
                            Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                .color(color)
                                .radius(sat_radius)
                                .filled(true),
                        );
                        plot_ui.points(
                            Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                .color(bg_color)
                                .radius(sat_radius * 0.5)
                                .filled(true),
                        );
                    }
                }
                continue;
            }
            for plane in 0..constellation.num_planes {
                let base_color = plane_color(if single_color { *color_offset } else { plane + color_offset });

                for sat in positions.iter().filter(|s| s.plane == plane) {
                    let is_tracked = satellite_cameras.iter().any(|c|
                        c.constellation_idx == *orig_idx && c.plane == sat.plane && c.sat_index == sat.sat_index
                    );
                    let color = if show_routing_paths {
                        if is_tracked {
                            if sat.ascending { COLOR_ASCENDING } else { COLOR_DESCENDING }
                        } else {
                            if sat.ascending {
                                egui::Color32::from_rgb(180, 140, 80)
                            } else {
                                egui::Color32::from_rgb(80, 120, 180)
                            }
                        }
                    } else {
                        base_color
                    };
                    let dim_col = egui::Color32::from_rgba_unmultiplied(
                        color.r() / 2, color.g() / 2, color.b() / 2, 80,
                    );

                    let (rx, ry, rz) = rotate_point_matrix(sat.x, sat.y, sat.z, &rotation);
                    let in_front = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;

                    let bg_color = if dark_mode {
                        egui::Color32::from_rgb(30, 30, 30)
                    } else {
                        egui::Color32::from_rgb(240, 240, 240)
                    };

                    if !hide_behind_earth && !in_front {
                        plot_ui.points(
                            Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                .color(dim_col)
                                .radius(sat_radius * 0.8)
                                .filled(true),
                        );
                        if *is_tle {
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(bg_color)
                                    .radius(sat_radius * 0.4)
                                    .filled(true),
                            );
                        }
                    }
                    if in_front {
                        plot_ui.points(
                            Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                .color(color)
                                .radius(sat_radius)
                                .filled(true),
                        );
                        if *is_tle {
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(bg_color)
                                    .radius(sat_radius * 0.5)
                                    .filled(true),
                            );
                        }
                    }
                }
            }
        }
    });

    for (_constellation, positions, color_offset, _is_tle, orig_idx) in constellations {
        for sat in positions {
            for cam in satellite_cameras.iter_mut() {
                if cam.constellation_idx == *orig_idx && cam.plane == sat.plane && cam.sat_index == sat.sat_index {
                    let (rx, ry, _) = rotate_point_matrix(sat.x, sat.y, sat.z, &rotation);
                    let plot_pt = egui_plot::PlotPoint::new(rx, ry);
                    let screen_pos = response.transform.position_from_point(&plot_pt);
                    cam.screen_pos = Some(screen_pos);

                    let color = plane_color(if single_color { *color_offset } else { sat.plane + color_offset });
                    ui.painter().circle_stroke(
                        screen_pos,
                        sat_radius * 2.5,
                        egui::Stroke::new(2.0, color),
                    );
                }
            }
        }
    }

    if let Some(hover_pos) = response.response.hover_pos() {
        let plot_pos = response.transform.value_from_position(hover_pos);
        let hover_threshold = margin * 0.025;

        'hover: for (_constellation, positions, color_offset, _, _) in constellations {
            for sat in positions {
                let (rx, ry, rz) = rotate_point_matrix(sat.x, sat.y, sat.z, &rotation);
                let earth_r_sq = (planet_radius * 0.95).powi(2) as f64;
                let visible = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;
                if !visible && hide_behind_earth {
                    continue;
                }
                let dx = rx - plot_pos.x;
                let dy = ry - plot_pos.y;
                if dx * dx + dy * dy < hover_threshold * hover_threshold {
                    let plot_pt = egui_plot::PlotPoint::new(rx, ry);
                    let screen_pt = response.transform.position_from_point(&plot_pt);
                    let color = plane_color(if single_color { *color_offset } else { sat.plane + color_offset });
                    ui.painter().circle_stroke(
                        screen_pt,
                        sat_radius * 2.0,
                        egui::Stroke::new(2.0, color),
                    );
                    break 'hover;
                }
            }
        }
    }

    if response.response.dragged() && !response.response.drag_started() {
        let drag = response.response.drag_delta();
        let delta_rot = rotation_from_drag(drag.x as f64 * 0.01, drag.y as f64 * 0.01);
        rotation = delta_rot * rotation;
    }

    if response.response.clicked() {
        if let Some(pos) = response.response.interact_pointer_pos() {
            let plot_pos = response.transform.value_from_position(pos);
            let click_x = plot_pos.x;
            let click_y = plot_pos.y;
            let click_threshold = margin * 0.03;

            'outer: for (_constellation, positions, _color_offset, _, orig_idx) in constellations {
                for sat in positions {
                    let (rx, ry, rz) = rotate_point_matrix(sat.x, sat.y, sat.z, &rotation);
                    let earth_r_sq = (planet_radius * 0.95).powi(2) as f64;
                    let visible = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;
                    if !visible && hide_behind_earth {
                        continue;
                    }
                    let dx = rx - click_x;
                    let dy = ry - click_y;
                    if dx * dx + dy * dy < click_threshold * click_threshold {
                        let existing = satellite_cameras.iter().find(|c|
                            c.constellation_idx == *orig_idx && c.plane == sat.plane && c.sat_index == sat.sat_index
                        );
                        if let Some(cam) = existing {
                            cameras_to_remove.push(cam.id);
                        } else {
                            let in_pending = pending_cameras.iter().any(|c|
                                c.constellation_idx == *orig_idx && c.plane == sat.plane && c.sat_index == sat.sat_index
                            );
                            if !in_pending {
                                *camera_id_counter += 1;
                                pending_cameras.push(SatelliteCamera {
                                    id: *camera_id_counter,
                                    label: format!("Sat {}-{}", sat.plane + 1, sat.sat_index + 1),
                                    constellation_idx: *orig_idx,
                                    plane: sat.plane,
                                    sat_index: sat.sat_index,
                                    screen_pos: None,
                                });
                            }
                        }
                        break 'outer;
                    }
                }
            }
        }
    }

    if response.response.hovered() {
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0 {
            let factor = 1.0 + scroll as f64 * 0.001;
            zoom = (zoom * factor).clamp(0.01, 20.0);
        }
        if let Some(touch) = ui.input(|i| i.multi_touch()) {
            let factor = touch.zoom_delta as f64;
            zoom = (zoom * factor).clamp(0.01, 20.0);
        }
    }

    (rotation, zoom)
}

fn draw_ground_track(
    ui: &mut egui::Ui,
    id: &str,
    constellations: &[(WalkerConstellation, Vec<SatelliteState>, usize, bool, usize)],
    width: f32,
    height: f32,
    sat_radius: f32,
    single_color: bool,
) {
    let plot = Plot::new(id)
        .width(width)
        .height(height)
        .include_x(-180.0)
        .include_x(180.0)
        .include_y(-90.0)
        .include_y(90.0)
        .show_axes([true, true]);

    plot.show(ui, |plot_ui| {
        for (constellation, positions, color_offset, _is_tle, _) in constellations {
            for plane in 0..constellation.num_planes {
                let color = plane_color(if single_color { *color_offset } else { plane + color_offset });
                let pts: PlotPoints = positions
                    .iter()
                    .filter(|s| s.plane == plane)
                    .map(|s| [s.lon, s.lat])
                    .collect();
                plot_ui.points(
                    Points::new("", pts)
                        .color(color)
                        .radius(sat_radius)
                        .filled(true),
                );
            }
        }

        plot_ui.line(
            Line::new("", PlotPoints::new(vec![[-180.0, 0.0], [180.0, 0.0]]))
                .color(egui::Color32::DARK_GRAY)
                .width(0.5),
        );
        plot_ui.line(
            Line::new("", PlotPoints::new(vec![[0.0, -90.0], [0.0, 90.0]]))
                .color(egui::Color32::DARK_GRAY)
                .width(0.5),
        );
    });
}

fn draw_torus(
    ui: &mut egui::Ui,
    id: &str,
    constellations: &[(WalkerConstellation, Vec<SatelliteState>, usize, bool, usize)],
    time: f64,
    rotation: Matrix3<f64>,
    width: f32,
    height: f32,
    sat_radius: f32,
    show_links: bool,
    single_color: bool,
    mut zoom: f64,
    satellite_cameras: &mut [SatelliteCamera],
    show_routing_paths: bool,
    show_manhattan_path: bool,
    show_shortest_path: bool,
    planet_radius: f64,
    pending_cameras: &mut Vec<SatelliteCamera>,
    camera_id_counter: &mut usize,
    cameras_to_remove: &mut Vec<usize>,
) -> (Matrix3<f64>, f64) {
    let (major_radius, minor_radius) = if let Some((constellation, _, _, _, _)) = constellations.first() {
        let inclination_rad = constellation.inclination_deg.to_radians();
        let cos_i = inclination_rad.cos().abs();
        let major = 1.0;
        let minor = (major * (1.0 - cos_i) / (1.0 + cos_i)).max(0.05);
        (major, minor)
    } else {
        (1.0, 0.8)
    };

    let margin = (major_radius + minor_radius) * 1.3 / zoom;

    let mut user_rotation = rotation;
    let display_rotation = rotation;

    let plot = Plot::new(id)
        .data_aspect(1.0)
        .width(width)
        .height(height)
        .show_axes(false)
        .show_grid(false)
        .show_x(false)
        .show_y(false)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .cursor_color(egui::Color32::TRANSPARENT);

    let response = plot.show(ui, |plot_ui| {
        plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
            [-margin, -margin],
            [margin, margin],
        ));

        let is_facing_camera = |theta: f64, phi: f64| -> bool {
            let nx = phi.cos() * theta.cos();
            let ny = phi.sin();
            let nz = phi.cos() * theta.sin();
            let (_, _, nz_rot) = rotate_point_matrix(nx, ny, nz, &rotation);
            nz_rot >= 0.0
        };

        let torus_point = |theta: f64, phi: f64| -> (f64, f64, f64) {
            let r = major_radius + minor_radius * phi.cos();
            let y = minor_radius * phi.sin();
            let x = r * theta.cos();
            let z = r * theta.sin();
            rotate_point_matrix(x, y, z, &display_rotation)
        };

        for (_cidx, (constellation, positions, color_offset, _is_tle, orig_idx)) in constellations.iter().enumerate() {
            let sats_per_plane = constellation.total_sats / constellation.num_planes;
            let orbit_radius = constellation.planet_radius + constellation.altitude_km;
            let period = 2.0 * PI * (orbit_radius.powi(3) / constellation.planet_mu).sqrt();
            let mean_motion = 2.0 * PI / period;

            let torus_pos = |plane: usize, sat_idx: usize| -> (f64, f64, f64) {
                let angle = 2.0 * PI * plane as f64 / constellation.num_planes as f64;
                let sat_spacing = 2.0 * PI * sat_idx as f64 / sats_per_plane as f64;
                let phase = sat_spacing + mean_motion * time;
                torus_point(angle, phase)
            };

            for plane in 0..constellation.num_planes {
                let angle = 2.0 * PI * plane as f64 / constellation.num_planes as f64;
                let color = if show_routing_paths {
                    egui::Color32::from_rgb(80, 80, 80)
                } else {
                    plane_color(if single_color { *color_offset } else { plane + color_offset })
                };
                let dim_col = egui::Color32::from_rgba_unmultiplied(
                    color.r(), color.g(), color.b(), 180,
                );

                let mut front_segment: Vec<[f64; 2]> = Vec::new();
                let mut back_segment: Vec<[f64; 2]> = Vec::new();

                for i in 0..=50 {
                    let phase = 2.0 * PI * i as f64 / 50.0;
                    let (rx, ry, _) = torus_point(angle, phase);
                    let facing = is_facing_camera(angle, phase);

                    if facing {
                        front_segment.push([rx, ry]);
                        if !back_segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut back_segment)))
                                    .color(dim_col)
                                    .width(1.0),
                            );
                        }
                    } else {
                        back_segment.push([rx, ry]);
                        if !front_segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut front_segment)))
                                    .color(color)
                                    .width(1.5),
                            );
                        }
                    }
                }
                if !front_segment.is_empty() {
                    plot_ui.line(Line::new("", PlotPoints::new(front_segment)).color(color).width(1.5));
                }
                if !back_segment.is_empty() {
                    plot_ui.line(Line::new("", PlotPoints::new(back_segment)).color(dim_col).width(1.0));
                }
            }

            if show_links {
                let base_link_color = if show_routing_paths {
                    egui::Color32::from_rgb(80, 80, 80)
                } else {
                    egui::Color32::from_rgb(150, 150, 150)
                };
                let link_dim = egui::Color32::from_rgba_unmultiplied(50, 50, 60, 100);
                for sat in positions {
                    if let Some(neighbor_idx) = sat.neighbor_idx {
                        let neighbor = &positions[neighbor_idx];
                        let angle1 = 2.0 * PI * sat.plane as f64 / constellation.num_planes as f64;
                        let angle2 = 2.0 * PI * neighbor.plane as f64 / constellation.num_planes as f64;
                        let phase1 = 2.0 * PI * sat.sat_index as f64 / sats_per_plane as f64 + mean_motion * time;
                        let phase2 = 2.0 * PI * neighbor.sat_index as f64 / sats_per_plane as f64 + mean_motion * time;

                        let (x1, y1, _) = torus_pos(sat.plane, sat.sat_index);
                        let (x2, y2, _) = torus_pos(neighbor.plane, neighbor.sat_index);
                        let facing1 = is_facing_camera(angle1, phase1);
                        let facing2 = is_facing_camera(angle2, phase2);
                        let color = if facing1 && facing2 { base_link_color } else { link_dim };
                        plot_ui.line(
                            Line::new("", PlotPoints::new(vec![[x1, y1], [x2, y2]]))
                                .color(color)
                                .width(1.0),
                        );
                    }
                }
            }

            for plane in 0..constellation.num_planes {
                let base_color = plane_color(if single_color { *color_offset } else { plane + color_offset });
                let angle = 2.0 * PI * plane as f64 / constellation.num_planes as f64;

                for sat in positions.iter().filter(|s| s.plane == plane) {
                    let is_tracked = satellite_cameras.iter().any(|c|
                        c.constellation_idx == *orig_idx && c.plane == sat.plane && c.sat_index == sat.sat_index
                    );
                    let color = if show_routing_paths {
                        if is_tracked {
                            if sat.ascending { COLOR_ASCENDING } else { COLOR_DESCENDING }
                        } else {
                            if sat.ascending {
                                egui::Color32::from_rgb(180, 140, 80)
                            } else {
                                egui::Color32::from_rgb(80, 120, 180)
                            }
                        }
                    } else {
                        base_color
                    };
                    let dim_col = egui::Color32::from_rgba_unmultiplied(
                        color.r(), color.g(), color.b(), 140,
                    );

                    let phase = 2.0 * PI * sat.sat_index as f64 / sats_per_plane as f64 + mean_motion * time;
                    let (x, y, _) = torus_pos(sat.plane, sat.sat_index);
                    let facing = is_facing_camera(angle, phase);
                    let (c, r) = if facing { (color, sat_radius) } else { (dim_col, sat_radius * 0.8) };
                    plot_ui.points(
                        Points::new("", PlotPoints::new(vec![[x, y]]))
                            .color(c)
                            .radius(r)
                            .filled(true),
                    );

                    if is_tracked {
                        plot_ui.points(
                            Points::new("", PlotPoints::new(vec![[x, y]]))
                                .color(base_color)
                                .radius(sat_radius * 2.5)
                                .filled(false),
                        );
                    }
                }
            }

            if show_routing_paths {
                let tracked: Vec<_> = satellite_cameras.iter()
                    .filter(|c| c.constellation_idx == *orig_idx)
                    .collect();

                if tracked.len() >= 2 {
                    let manhattan_color = egui::Color32::from_rgb(255, 100, 100);
                    let shortest_color = egui::Color32::from_rgb(100, 255, 100);
                    let is_star = constellation.walker_type == WalkerType::Star;
                    let num_planes = constellation.num_planes;

                    for i in 0..tracked.len() {
                        for j in (i + 1)..tracked.len() {
                            let src = tracked[i];
                            let dst = tracked[j];

                            let src_sat = positions.iter().find(|s| s.plane == src.plane && s.sat_index == src.sat_index);
                            let dst_sat = positions.iter().find(|s| s.plane == dst.plane && s.sat_index == dst.sat_index);

                            let can_route = match (src_sat, dst_sat) {
                                (Some(_), Some(_)) => {
                                    if is_star {
                                        let plane_diff_fwd = (dst.plane + num_planes - src.plane) % num_planes;
                                        let plane_diff_bwd = (src.plane + num_planes - dst.plane) % num_planes;
                                        let crosses_seam = plane_diff_fwd > num_planes / 2 && plane_diff_bwd > num_planes / 2;
                                        !crosses_seam
                                    } else {
                                        true
                                    }
                                }
                                _ => false,
                            };

                            if !can_route {
                                continue;
                            }

                            if show_manhattan_path {
                                let path = compute_manhattan_path(
                                    src.plane, src.sat_index,
                                    dst.plane, dst.sat_index,
                                    num_planes, sats_per_plane,
                                    is_star,
                                );
                                for k in 0..(path.len() - 1) {
                                    let (p1, s1) = path[k];
                                    let (p2, s2) = path[k + 1];
                                    let (x1, y1, _) = torus_pos(p1, s1);
                                    let (x2, y2, _) = torus_pos(p2, s2);
                                    plot_ui.line(
                                        Line::new("", PlotPoints::new(vec![[x1, y1], [x2, y2]]))
                                            .color(manhattan_color)
                                            .width(2.5),
                                    );
                                }
                            }

                            if show_shortest_path {
                                let path = compute_shortest_path(
                                    src.plane, src.sat_index,
                                    dst.plane, dst.sat_index,
                                    num_planes, sats_per_plane,
                                    positions,
                                    is_star,
                                );
                                for k in 0..(path.len() - 1) {
                                    let (p1, s1) = path[k];
                                    let (p2, s2) = path[k + 1];
                                    let (x1, y1, _) = torus_pos(p1, s1);
                                    let (x2, y2, _) = torus_pos(p2, s2);
                                    plot_ui.line(
                                        Line::new("", PlotPoints::new(vec![[x1, y1], [x2, y2]]))
                                            .color(shortest_color)
                                            .width(2.0),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    });

    if response.response.dragged() && !response.response.drag_started() {
        let drag = response.response.drag_delta();
        let delta_rot = rotation_from_drag(drag.x as f64 * 0.01, drag.y as f64 * 0.01);
        user_rotation = delta_rot * user_rotation;
    }

    if response.response.hovered() {
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0 {
            let factor = 1.0 + scroll as f64 * 0.001;
            zoom = (zoom * factor).clamp(0.01, 20.0);
        }
        if let Some(touch) = ui.input(|i| i.multi_touch()) {
            let factor = touch.zoom_delta as f64;
            zoom = (zoom * factor).clamp(0.01, 20.0);
        }
    }

    if let Some(pos) = response.response.interact_pointer_pos() {
        if response.response.clicked() {
            let click_x = response.transform.value_from_position(pos).x;
            let click_y = response.transform.value_from_position(pos).y;
            let (major_radius, minor_radius) = if let Some((constellation, _, _, _, _)) = constellations.first() {
                let sats_per_plane = constellation.sats_per_plane();
                let orbit_radius = planet_radius + constellation.altitude_km;
                let inclination_rad = constellation.inclination_deg.to_radians();
                let inclination_factor = inclination_rad.sin().abs().max(0.1);
                let altitude_factor = orbit_radius / (planet_radius + 500.0);
                let major = altitude_factor * (sats_per_plane as f64 / constellation.num_planes as f64);
                let minor_base = altitude_factor * inclination_factor;
                let minor = minor_base.max(major * inclination_factor);
                let scale = 2.0 / (major + minor).max(1.0);
                (major * scale, minor * scale)
            } else {
                (2.0, 0.8)
            };
            let margin = (major_radius + minor_radius) * 1.3 / zoom;
            let click_threshold = margin * 0.05;

            let torus_point = |theta: f64, phi: f64| -> (f64, f64, f64) {
                let r = major_radius + minor_radius * phi.cos();
                let y = minor_radius * phi.sin();
                let x = r * theta.cos();
                let z = r * theta.sin();
                rotate_point_matrix(x, y, z, &display_rotation)
            };

            'outer: for (_cidx, (constellation, positions, _, _, orig_idx)) in constellations.iter().enumerate() {
                let sats_per_plane = constellation.total_sats / constellation.num_planes;
                let orbit_radius = constellation.planet_radius + constellation.altitude_km;
                let period = 2.0 * PI * (orbit_radius.powi(3) / constellation.planet_mu).sqrt();
                let mean_motion = 2.0 * PI / period;

                for sat in positions {
                    let angle = 2.0 * PI * sat.plane as f64 / constellation.num_planes as f64;
                    let phase = 2.0 * PI * sat.sat_index as f64 / sats_per_plane as f64 + mean_motion * time;
                    let (tx, ty, _) = torus_point(angle, phase);

                    let dx = tx - click_x;
                    let dy = ty - click_y;
                    if dx * dx + dy * dy < click_threshold * click_threshold {
                        let existing = satellite_cameras.iter().find(|c|
                            c.constellation_idx == *orig_idx && c.plane == sat.plane && c.sat_index == sat.sat_index
                        );
                        if let Some(cam) = existing {
                            cameras_to_remove.push(cam.id);
                        } else {
                            let in_pending = pending_cameras.iter().any(|c|
                                c.constellation_idx == *orig_idx && c.plane == sat.plane && c.sat_index == sat.sat_index
                            );
                            if !in_pending {
                                *camera_id_counter += 1;
                                pending_cameras.push(SatelliteCamera {
                                    id: *camera_id_counter,
                                    label: format!("Sat {}-{}", sat.plane + 1, sat.sat_index + 1),
                                    constellation_idx: *orig_idx,
                                    plane: sat.plane,
                                    sat_index: sat.sat_index,
                                    screen_pos: None,
                                });
                            }
                        }
                        break 'outer;
                    }
                }
            }
        }
    }

    (user_rotation, zoom)
}

fn plane_color(plane: usize) -> egui::Color32 {
    COLORS[plane % COLORS.len()]
}

fn color_name(idx: usize) -> &'static str {
    COLOR_NAMES[idx % COLOR_NAMES.len()]
}

const COLORS: [egui::Color32; 8] = [
    egui::Color32::from_rgb(255, 99, 71),
    egui::Color32::from_rgb(50, 205, 50),
    egui::Color32::from_rgb(30, 144, 255),
    egui::Color32::from_rgb(255, 215, 0),
    egui::Color32::from_rgb(238, 130, 238),
    egui::Color32::from_rgb(0, 206, 209),
    egui::Color32::from_rgb(255, 140, 0),
    egui::Color32::from_rgb(147, 112, 219),
];

const COLOR_NAMES: [&str; 8] = [
    "Red", "Green", "Blue", "Gold", "Violet", "Cyan", "Orange", "Purple",
];

fn dim_color(color: egui::Color32) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(
        (color.r() as f32 * 0.4) as u8,
        (color.g() as f32 * 0.4) as u8,
        (color.b() as f32 * 0.4) as u8,
        200,
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1600.0, 1000.0]),
        ..Default::default()
    };

    eframe::run_native(
        "LEO Viz",
        options,
        Box::new(|_cc| Ok(Box::new(App::default()))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("No window")
            .document()
            .expect("No document");

        let canvas = document
            .get_element_by_id("canvas")
            .expect("No canvas element")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("Not a canvas");

        let web_options = eframe::WebOptions::default();
        eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|_cc| Ok(Box::new(App::default()))),
            )
            .await
            .expect("Failed to start eframe");
    });
}
