//! Configuration types for tabs, planets, constellations, and ground elements.
//!
//! Defines TabConfig, PlanetConfig, ConstellationConfig, presets, ground
//! stations, areas of interest, device layers, and per-tab view settings.

use crate::celestial::{CelestialBody, Skin};
use crate::kessler::DebrisFragment;
use crate::math::lat_lon_to_matrix;
use crate::tle::{TlePreset, TleLoadState, TleShell};
use crate::walker::{WalkerType, WalkerConstellation};
use eframe::egui;
use nalgebra::Matrix3;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, PartialEq)]
pub enum Propagator {
    Keplerian,
    J2,
    Numerical,
    #[cfg(not(target_arch = "wasm32"))]
    Lib42,
}

#[derive(Clone)]
pub struct NumericalSatState {
    pub pos: [f64; 3],
    pub vel: [f64; 3],
}

#[derive(Clone)]
pub struct NumericalState {
    pub sats: Vec<NumericalSatState>,
    pub time: f64,
    pub config_hash: u64,
}


#[derive(Clone, Copy, PartialEq)]
pub enum Preset {
    None,
    Starlink,
    OneWeb,
    Iridium,
    Kuiper,
    Iris2,
    Telesat,
}

#[derive(Clone)]
pub struct ConstellationConfig {
    pub sats_per_plane: usize,
    pub num_planes: usize,
    pub altitude_km: f64,
    pub inclination: f64,
    pub sso: bool,
    pub walker_type: WalkerType,
    pub phasing: f64,
    pub raan_offset: f64,
    pub raan_spacing: Option<f64>,
    pub sat_spacing_km: Option<f64>,
    pub eccentricity: f64,
    pub arg_periapsis: f64,
    pub isl_neighbors: usize,
    pub propagator: Propagator,
    pub drag_enabled: bool,
    pub ballistic_coeff: f64,
    pub preset: Preset,
    pub label: Option<String>,
    pub color_offset: usize,
    pub hidden: bool,
    pub show_physics_ui: bool,
    pub physics: crate::physics::PhysicsConfig,
    pub physics_state: Vec<crate::physics::SatellitePhysics>,
    pub numerical: Option<NumericalState>,
}

impl ConstellationConfig {
    pub fn new(color_offset: usize) -> Self {
        Self {
            sats_per_plane: 30,
            num_planes: 30,
            altitude_km: 200.0,
            inclination: 90.0,
            sso: false,
            walker_type: WalkerType::Delta,
            phasing: 0.0,
            raan_offset: 0.0,
            raan_spacing: None,
            sat_spacing_km: None,
            eccentricity: 0.0,
            arg_periapsis: 0.0,
            isl_neighbors: 4,
            propagator: Propagator::Keplerian,
            drag_enabled: false,
            ballistic_coeff: 100.0,
            preset: Preset::None,
            label: None,
            color_offset,
            hidden: false,
            show_physics_ui: false,
            physics: crate::physics::PhysicsConfig::default(),
            physics_state: Vec::new(),
            numerical: None,
        }
    }

    pub fn total_sats(&self) -> usize {
        self.sats_per_plane * self.num_planes
    }

    pub fn orbital_config_hash(&self) -> u64 {
        let mut h = 0u64;
        h = h.wrapping_mul(31).wrapping_add(self.sats_per_plane as u64);
        h = h.wrapping_mul(31).wrapping_add(self.num_planes as u64);
        h = h.wrapping_mul(31).wrapping_add(self.altitude_km.to_bits());
        h = h.wrapping_mul(31).wrapping_add(self.inclination.to_bits());
        h = h.wrapping_mul(31).wrapping_add(self.eccentricity.to_bits());
        h = h.wrapping_mul(31).wrapping_add(self.arg_periapsis.to_bits());
        h = h.wrapping_mul(31).wrapping_add(self.phasing.to_bits());
        h = h.wrapping_mul(31).wrapping_add(self.raan_offset.to_bits());
        h
    }

    pub fn sso_inclination(altitude_km: f64, eccentricity: f64, planet_mu: f64, planet_j2: f64, planet_mean_radius: f64, planet_eq_radius: f64, planet_year_days: f64) -> Option<f64> {
        // a uses mean radius (matches walker.rs's semi_major calculation).
        // The J2 (Re/a)² scaling uses equatorial radius (matches walker.rs's r_ratio).
        let a = planet_mean_radius + altitude_km;
        let n = (planet_mu / (a * a * a)).sqrt();
        let rate_required = 2.0 * std::f64::consts::PI / (planet_year_days * 86400.0);
        let e2 = 1.0 - eccentricity * eccentricity;
        let cos_i = -rate_required * e2 * e2 * a * a / (1.5 * n * planet_j2 * planet_eq_radius * planet_eq_radius);
        if cos_i.abs() <= 1.0 {
            Some(cos_i.acos().to_degrees())
        } else {
            None
        }
    }

    pub fn constellation(&self, planet_radius: f64, planet_mu: f64, planet_j2: f64, planet_equatorial_radius: f64) -> WalkerConstellation {
        WalkerConstellation {
            walker_type: self.walker_type,
            total_sats: self.sats_per_plane * self.num_planes,
            num_planes: self.num_planes,
            altitude_km: self.altitude_km,
            inclination_deg: self.inclination,
            phasing: self.phasing,
            raan_offset_deg: self.raan_offset,
            raan_spacing_deg: self.raan_spacing,
            sat_spacing_km: self.sat_spacing_km,
            isl_neighbors: self.isl_neighbors,
            propagator: self.propagator,
            eccentricity: self.eccentricity,
            arg_periapsis_deg: self.arg_periapsis,
            planet_radius,
            planet_mu,
            planet_j2,
            planet_equatorial_radius,
        }
    }

    pub fn preset_name(&self) -> &str {
        if let Some(label) = &self.label {
            return label;
        }
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
pub struct GroundStation {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub radius_km: f64,
    pub color: egui::Color32,
    pub selected: bool,
}

#[derive(Clone, Copy, PartialEq)]
pub enum AoiJobMode {
    Route,
    SpaceComp,
}

#[derive(Clone)]
pub struct AreaOfInterest {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub radius_km: f64,
    pub color: egui::Color32,
    pub ground_station_idx: Option<usize>,
    pub job_mode: AoiJobMode,
    pub job_n: usize,
    pub selected: bool,
}

#[derive(Clone)]
pub struct DeviceLayer {
    pub name: String,
    pub color: egui::Color32,
    pub devices: Vec<(f64, f64)>,
}

#[derive(Clone)]
pub struct PlanetConfig {
    pub name: String,
    pub constellations: Vec<ConstellationConfig>,
    pub constellation_counter: usize,
    pub celestial_body: CelestialBody,
    pub skin: Skin,
    pub abstract_ocean: egui::Color32,
    pub abstract_land: egui::Color32,
    pub abstract_ice: egui::Color32,
    pub abstract_colors_dirty: bool,
    pub satellite_cameras: Vec<SatelliteCamera>,
    pub pending_cameras: Vec<SatelliteCamera>,
    pub cameras_to_remove: Vec<usize>,
    pub show_tle_window: bool,
    pub show_gs_aoi_window: bool,
    pub show_config_window: bool,
    pub tle_selections: HashMap<TlePreset, (bool, TleLoadState, Option<Vec<TleShell>>)>,
    pub ground_stations: Vec<GroundStation>,
    pub areas_of_interest: Vec<AreaOfInterest>,
    pub device_layers: Vec<DeviceLayer>,
    pub pass_cache: PassPredictionCache,
    pub conjunction_cache: ConjunctionCache,
    pub conjunction_prev_positions: HashMap<(usize, usize), [f64; 3]>,
    pub show_conjunction_window: bool,
    pub show_conjunction_lines: bool,
    pub kessler: KesslerSimulation,
    pub radiation: RadiationConfig,
    pub show_radiation_window: bool,
    pub show_moons_window: bool,
    pub enabled_moons: HashSet<CelestialBody>,
    pub moon_inclination_override: Option<f64>,
    pub auto_cluster_tle: bool,
    pub tle_isl_k: usize,
    /// Drop ISLs whose endpoint latitude exceeds this absolute value (degrees).
    /// Models the high-latitude cross-plane laser tear-down. `90.0` disables.
    pub tle_isl_max_lat_deg: f64,
    /// Accumulated sub-satellite points per tracked satellite, keyed by
    /// (constellation_idx, plane, sat_index). Stored as (lat_deg, lon_deg, sim_time_s).
    pub ground_track_history: HashMap<(usize, usize, usize), Vec<(f64, f64, f64)>>,
    /// Optional per-planet projection override. When `None`, the tab's
    /// `TabSettings.planet_projection` is used.
    pub projection_override: Option<crate::projection::ProjectionKind>,
}

impl PlanetConfig {
    pub fn new(name: String) -> Self {
        let mut tle_selections = HashMap::new();
        for preset in TlePreset::ALL {
            tle_selections.insert(preset, (false, TleLoadState::NotLoaded, None));
        }
        Self {
            name,
            constellations: Vec::new(),
            constellation_counter: 0,
            celestial_body: CelestialBody::Earth,
            skin: Skin::Default,
            abstract_ocean: egui::Color32::from_rgb(25, 40, 80),
            abstract_land: egui::Color32::from_rgb(60, 75, 85),
            abstract_ice: egui::Color32::from_rgb(140, 150, 160),
            abstract_colors_dirty: false,
            satellite_cameras: Vec::new(),
            pending_cameras: Vec::new(),
            cameras_to_remove: Vec::new(),
            show_tle_window: false,
            show_gs_aoi_window: false,
            show_config_window: true,
            tle_selections,
            ground_stations: Vec::new(),
            areas_of_interest: Vec::new(),
            device_layers: vec![DeviceLayer {
                name: "CandyTron".to_string(),
                color: egui::Color32::from_rgb(80, 140, 255),
                devices: vec![(59.40481807006525, 17.949657783197082), (59.41, 17.96)],
            }],
            pass_cache: PassPredictionCache::default(),
            conjunction_cache: ConjunctionCache::default(),
            conjunction_prev_positions: HashMap::new(),
            show_conjunction_window: false,
            show_conjunction_lines: false,
            kessler: KesslerSimulation::default(),
            radiation: RadiationConfig::default(),
            show_radiation_window: false,
            show_moons_window: false,
            enabled_moons: HashSet::new(),
            moon_inclination_override: None,
            auto_cluster_tle: false,
            tle_isl_k: 0,
            tle_isl_max_lat_deg: 70.0,
            ground_track_history: HashMap::new(),
            projection_override: None,
        }
    }

    pub fn add_constellation(&mut self) {
        self.constellations.push(ConstellationConfig::new(self.constellation_counter));
        self.constellation_counter += 1;
    }
}

#[derive(Clone)]
pub struct View3DFlags {
    pub show_orbits: bool,
    pub show_axes: bool,
    pub show_magnetic_axis: bool,
    pub show_coverage: bool,
    pub show_links: bool,
    pub hide_behind_earth: bool,
    pub single_color: bool,
    pub dark_mode: bool,
    pub show_routing_paths: bool,
    pub show_manhattan_path: bool,
    pub show_shortest_path: bool,
    pub show_radiation_path: bool,
    pub radiation_weight: f64,
    pub routing_width: f32,
    pub routing_node_scale: f32,
    pub show_asc_desc_colors: bool,
    pub color_ascending: egui::Color32,
    pub color_descending: egui::Color32,
    pub color_links: egui::Color32,
    pub show_sat_labels: bool,
    pub show_altitude_lines: bool,
    pub altitude_line_width: f32,
    pub show_inclination_bounds: bool,
    pub render_planet: bool,
    pub fixed_sizes: bool,
    pub show_sat_border: bool,
    pub show_polar_circle: bool,
    pub show_equator: bool,
    pub show_graticule: bool,
    pub show_crosshairs: bool,
    pub show_terminator: bool,
    pub show_eclipse: bool,
    pub show_sun: bool,
    pub earth_fixed_camera: bool,
    pub use_gpu_rendering: bool,
    pub show_clouds: bool,
    pub show_day_night: bool,
    pub show_city_lights: bool,
    pub show_stars: bool,
    pub show_borders: bool,
    pub show_cities: bool,
    pub trackpad_rotate: bool,
    pub north_up: bool,
    pub enabled_moons: HashSet<CelestialBody>,
    pub moon_inclination_override: Option<f64>,
    pub show_moon_orbits: bool,
    pub show_moon_lines: bool,
    pub show_moon_labels: bool,
    pub moon_camera_distance_km: f64,
    pub tle_monochrome: bool,
    pub show_ground_tracks: bool,
}

#[derive(Clone, Copy, PartialEq, Default)]
pub enum CameraMode {
    #[default]
    Unlocked,
    TrackSatellite,
}

#[derive(Clone, Copy, PartialEq, Default)]
pub enum ViewMode {
    #[default]
    Planet,
    SolarSystem,
    PlanetSizes,
}

impl ViewMode {
    pub fn label(&self) -> &'static str {
        match self {
            ViewMode::Planet => "Planet",
            ViewMode::SolarSystem => "Solar System",
            ViewMode::PlanetSizes => "Planet Sizes",
        }
    }
}

#[derive(Clone)]
pub struct TabSettings {
    pub time: f64,
    pub speed: f64,
    pub animate: bool,
    pub zoom: f64,
    pub rotation: Matrix3<f64>,
    pub earth_fixed_camera: bool,
    pub camera_mode: CameraMode,
    pub show_camera_windows: bool,
    pub show_orbits: bool,
    pub show_links: bool,
    pub show_coverage: bool,
    pub coverage_angle: f64,
    pub show_routing_paths: bool,
    pub show_manhattan_path: bool,
    pub show_shortest_path: bool,
    pub show_radiation_path: bool,
    pub radiation_weight: f64,
    pub routing_width: f32,
    pub routing_node_scale: f32,
    pub show_asc_desc_colors: bool,
    pub color_ascending: egui::Color32,
    pub color_descending: egui::Color32,
    pub color_links: egui::Color32,
    pub show_sat_labels: bool,
    pub single_color: bool,
    pub show_torus: bool,
    pub planet_projection: crate::projection::ProjectionKind,
    pub show_axes: bool,
    pub show_magnetic_axis: bool,
    pub hide_behind_earth: bool,
    pub show_altitude_lines: bool,
    pub show_devices: bool,
    pub show_polar_circle: bool,
    pub show_equator: bool,
    pub show_graticule: bool,
    pub show_crosshairs: bool,
    pub show_borders: bool,
    pub show_cities: bool,
    pub show_day_night: bool,
    pub show_city_lights: bool,
    pub show_terminator: bool,
    pub show_eclipse: bool,
    pub show_sun: bool,
    pub show_clouds: bool,
    pub show_stars: bool,
    pub show_radiation_belts: bool,
    pub view_mode: ViewMode,
    pub show_hohmann: bool,
    pub show_ss_labels: bool,
    pub solar_system_hide_bodies: bool,
    pub solar_system_log_power: f64,
    pub sat_radius: f32,
    pub link_width: f32,
    pub fixed_sizes: bool,
    pub show_sat_border: bool,
    pub trackpad_rotate: bool,
    pub north_up: bool,
    pub show_moon_orbits: bool,
    pub show_moon_lines: bool,
    pub show_moon_labels: bool,
    pub moon_camera_distance_km: f64,
    pub show_circular_calendar: bool,
    pub auto_zoom: bool,
    pub auto_zoom_min_alt: f64,
    pub auto_zoom_max_alt: f64,
    pub auto_zoom_duration: f64,
    pub auto_zoom_time: f64,
    pub auto_rotate: bool,
    pub auto_rotate_speed: f64,
    pub auto_rotate_axis_lat: f64,
    pub auto_rotate_axis_lon: f64,
    pub camera_roll: f64,
    pub initial_rotation: Option<nalgebra::Matrix3<f64>>,
    pub tle_monochrome: bool,
    pub reset_time_on_switch: bool,
    pub sun_fixed_camera: bool,
    pub show_ground_tracks: bool,
    pub altitude_line_width: f32,
    pub show_inclination_bounds: bool,
}

impl Default for TabSettings {
    fn default() -> Self {
        Self {
            time: 0.0,
            speed: 50.0,
            animate: true,
            zoom: 1.0,
            rotation: lat_lon_to_matrix(0.0, 0.0),
            earth_fixed_camera: false,
            camera_mode: CameraMode::Unlocked,
            show_camera_windows: false,
            show_orbits: true,
            show_links: true,
            show_coverage: false,
            coverage_angle: 50.0,
            show_routing_paths: false,
            show_manhattan_path: true,
            show_shortest_path: true,
            show_radiation_path: false,
            radiation_weight: 5.0,
            routing_width: 1.5,
            routing_node_scale: 1.0,
            show_asc_desc_colors: false,
            color_ascending: egui::Color32::from_rgb(230, 150, 70),
            color_descending: egui::Color32::from_rgb(70, 130, 210),
            color_links: egui::Color32::from_rgb(80, 80, 80),
            show_sat_labels: true,
            single_color: false,
            show_torus: false,
            planet_projection: crate::projection::ProjectionKind::Orthographic,
            show_axes: false,
            show_magnetic_axis: false,
            hide_behind_earth: true,
            show_altitude_lines: false,
            show_devices: false,
            show_polar_circle: false,
            show_equator: false,
            show_graticule: false,
            show_crosshairs: false,
            show_borders: false,
            show_cities: false,
            show_day_night: false,
            show_city_lights: false,
            show_terminator: false,
            show_eclipse: false,
            show_sun: false,
            show_clouds: false,
            show_stars: false,
            show_radiation_belts: false,
            view_mode: ViewMode::Planet,
            show_hohmann: false,
            show_ss_labels: true,
            solar_system_hide_bodies: false,
            solar_system_log_power: 0.4,
            sat_radius: 1.5,
            link_width: 0.25,
            fixed_sizes: false,
            show_sat_border: false,
            trackpad_rotate: true,
            north_up: true,
            show_moon_orbits: true,
            show_moon_lines: false,
            show_moon_labels: true,
            moon_camera_distance_km: 1_000_000.0,
            show_circular_calendar: false,
            auto_zoom: false,
            auto_zoom_min_alt: 1000.0,
            auto_zoom_max_alt: 60000.0,
            auto_zoom_duration: 20.0,
            auto_zoom_time: 0.0,
            auto_rotate: false,
            auto_rotate_speed: 5.0,
            auto_rotate_axis_lat: 0.0,
            auto_rotate_axis_lon: 0.0,
            camera_roll: 0.0,
            initial_rotation: None,
            tle_monochrome: false,
            reset_time_on_switch: false,
            sun_fixed_camera: false,
            show_ground_tracks: false,
            altitude_line_width: 0.5,
            show_inclination_bounds: false,
        }
    }
}

pub struct TabConfig {
    pub name: String,
    pub title: String,
    pub description: String,
    pub planets: Vec<PlanetConfig>,
    pub planet_counter: usize,
    pub show_stats: bool,
    pub show_sat_list: bool,
    pub show_fps: bool,
    pub settings: TabSettings,
}

/// Build an egui `LayoutJob` from text with `**bold**` markdown inline spans.
/// Non-bold runs use `base_color`; bold runs are white and bold-weight.
pub fn description_layout_job(
    text: &str,
    font_size: f32,
    base_color: egui::Color32,
) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::default();
    let regular = egui::FontId::proportional(font_size);
    let bold = egui::FontId::new(font_size, egui::FontFamily::Proportional);
    let bold_color = egui::Color32::WHITE;

    let mut rest = text;
    while let Some(start) = rest.find("**") {
        if start > 0 {
            job.append(
                &rest[..start],
                0.0,
                egui::TextFormat {
                    font_id: regular.clone(),
                    color: base_color,
                    ..Default::default()
                },
            );
        }
        let after = &rest[start + 2..];
        if let Some(end) = after.find("**") {
            let bold_text = &after[..end];
            job.append(
                bold_text,
                0.0,
                egui::TextFormat {
                    font_id: bold.clone(),
                    color: bold_color,
                    italics: false,
                    underline: egui::Stroke::new(1.5, bold_color),
                    ..Default::default()
                },
            );
            rest = &after[end + 2..];
        } else {
            // Unmatched — render as-is.
            job.append(
                &rest[start..],
                0.0,
                egui::TextFormat {
                    font_id: regular.clone(),
                    color: base_color,
                    ..Default::default()
                },
            );
            rest = "";
        }
    }
    if !rest.is_empty() {
        job.append(
            rest,
            0.0,
            egui::TextFormat {
                font_id: regular,
                color: base_color,
                ..Default::default()
            },
        );
    }
    job
}

/// Strip `**...**` markdown markers from plain text (for renderers that
/// don't support styled runs).
pub fn strip_bold_markers(text: &str) -> String {
    text.replace("**", "")
}

impl TabConfig {
    pub fn new(name: String) -> Self {
        let mut tab = Self::new_empty(name);
        tab.add_planet();
        tab
    }

    pub fn new_empty(name: String) -> Self {
        Self {
            name,
            title: String::new(),
            description: String::new(),
            planets: Vec::new(),
            planet_counter: 0,
            show_stats: false,
            show_sat_list: false,
            show_fps: false,
            settings: TabSettings::default(),
        }
    }

    pub fn add_planet(&mut self) {
        self.planet_counter += 1;
        let planet = PlanetConfig::new(format!("Planet {}", self.planet_counter));
        self.planets.push(planet);
    }
}

#[derive(Clone)]
pub struct SatelliteCamera {
    pub id: usize,
    pub label: String,
    pub constellation_idx: usize,
    pub plane: usize,
    pub sat_index: usize,
    pub screen_pos: Option<egui::Pos2>,
}

#[derive(Clone)]
pub struct PassInfo {
    pub constellation_idx: usize,
    pub sat_plane: usize,
    pub sat_index: usize,
    pub sat_name: String,
    pub time_to_aos: f64,
    pub max_elevation: f64,
    pub duration: f64,
    pub ascending: bool,
    pub altitude_km: f64,
}

#[derive(Clone)]
pub struct PassPredictionCache {
    pub passes: HashMap<usize, Vec<PassInfo>>,
    pub last_compute_time: f64,
    pub prediction_window_min: f64,
}

impl Default for PassPredictionCache {
    fn default() -> Self {
        Self {
            passes: HashMap::new(),
            last_compute_time: f64::NEG_INFINITY,
            prediction_window_min: 10080.0,
        }
    }
}

#[derive(Clone)]
pub struct ConjunctionInfo {
    pub name_a: String,
    pub name_b: String,
    pub source_a: String,
    pub source_b: String,
    pub distance_km: f64,
    pub pos_a: [f64; 3],
    pub pos_b: [f64; 3],
    pub tca_seconds: f64,
    pub min_distance_km: f64,
}

#[derive(Clone)]
pub struct PredictedConjunction {
    pub name_a: String,
    pub name_b: String,
    pub source_a: String,
    pub source_b: String,
    pub time_until: f64,
    pub min_distance_km: f64,
}

#[derive(Clone)]
pub struct ConjunctionCache {
    pub conjunctions: Vec<ConjunctionInfo>,
    pub threshold_km: f64,
    pub show_heatmap: bool,
    pub predictions: Vec<PredictedConjunction>,
    pub prediction_window_min: f64,
    pub last_prediction_time: f64,
}

impl Default for ConjunctionCache {
    fn default() -> Self {
        Self {
            conjunctions: Vec::new(),
            threshold_km: 50.0,
            show_heatmap: false,
            predictions: Vec::new(),
            prediction_window_min: 10.0,
            last_prediction_time: f64::NEG_INFINITY,
        }
    }
}

#[derive(Clone)]
pub struct CourseCorrection {
    pub sat_name: String,
    pub start_time: f64,
    pub end_time: f64,
    pub altitude_offset_km: f64,
}

impl CourseCorrection {
    pub fn offset_at(&self, time: f64) -> f64 {
        if time < self.start_time || time > self.end_time {
            return 0.0;
        }
        let frac = (time - self.start_time) / (self.end_time - self.start_time);
        self.altitude_offset_km * (frac * std::f64::consts::PI).sin()
    }
}

#[derive(Clone)]
pub struct KesslerSimulation {
    pub enabled: bool,
    pub debris: Vec<DebrisFragment>,
    pub collision_threshold_km: f64,
    pub fragments_per_collision: usize,
    pub max_debris: usize,
    pub collision_count: usize,
    pub collision_id_counter: u64,
    pub collided_pairs: HashSet<(String, String)>,
    pub course_correction_enabled: bool,
    pub active_corrections: Vec<CourseCorrection>,
    pub correction_altitude_km: f64,
    pub corrections_made: usize,
}

#[derive(Clone, Copy, PartialEq)]
pub enum HeatmapMode {
    Radiation,
    FieldStrength,
    IgrfField,
    IgrfRadiation,
}

pub(crate) const GEOMAGNETIC_PALETTE: [[u8; 3]; 18] = [
    [9,33,52], [12,40,79], [16,47,113], [51,52,144],
    [77,60,148], [100,72,143], [120,82,139], [140,90,133],
    [160,100,131], [180,108,125], [202,118,114], [222,128,102],
    [240,143,90], [242,164,81], [244,184,80], [247,203,86],
    [250,222,96], [245,240,111],
];

pub fn heatmap_color(t: f64, smooth: bool) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0) as f32;
    let last = (GEOMAGNETIC_PALETTE.len() - 1) as f32;
    if smooth {
        let pos = t * last;
        let lo = (pos as usize).min(GEOMAGNETIC_PALETTE.len() - 2);
        let frac = pos - lo as f32;
        let [r0, g0, b0] = GEOMAGNETIC_PALETTE[lo];
        let [r1, g1, b1] = GEOMAGNETIC_PALETTE[lo + 1];
        egui::Color32::from_rgb(
            (r0 as f32 + (r1 as f32 - r0 as f32) * frac) as u8,
            (g0 as f32 + (g1 as f32 - g0 as f32) * frac) as u8,
            (b0 as f32 + (b1 as f32 - b0 as f32) * frac) as u8,
        )
    } else {
        let idx = (t * last) as usize;
        let [r, g, b] = GEOMAGNETIC_PALETTE[idx.min(GEOMAGNETIC_PALETTE.len() - 1)];
        egui::Color32::from_rgb(r, g, b)
    }
}


#[derive(Clone)]
pub struct RadiationConfig {
    pub kp_index: f64,
    pub show_belts: bool,
    pub show_magnetopause: bool,
    pub show_sat_exposure: bool,
    pub num_meridians: usize,
    pub num_shells: usize,
    pub shell_phasing: f64,
    pub num_links: usize,
    pub dipole_tilt: f64,
    pub show_lines: bool,
    pub show_dots: bool,
    pub dots_per_line: usize,
    pub connect_along_shell: bool,
    pub connect_across_shells: bool,
    pub show_fill: bool,
    pub show_heatmap_sphere: bool,
    pub heatmap_altitude_km: f64,
    pub heatmap_resolution: usize,
    pub dipole_offset_km: f64,
    pub heatmap_mode: HeatmapMode,
    pub show_protons: bool,
    pub show_electrons: bool,
    pub smooth_colors: bool,
    pub igrf_grid_cache: Option<(f64, crate::igrf::IgrfGrid)>,
    pub igrf_rad_cache: Option<(f64, f64, crate::igrf::IgrfRadGrid)>,
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(dead_code)]
    pub igrf_rad_pending: Option<(f64, f64, std::sync::Arc<std::sync::Mutex<Option<crate::igrf::IgrfRadGrid>>>)>,
}

impl Default for RadiationConfig {
    fn default() -> Self {
        Self {
            kp_index: 5.0,
            show_belts: true,
            show_magnetopause: false,
            show_sat_exposure: true,
            num_meridians: 2,
            num_shells: 32,
            shell_phasing: 0.0,
            num_links: 0,
            dipole_tilt: 11.0,
            show_lines: false,
            show_dots: false,
            dots_per_line: 12,
            connect_along_shell: false,
            connect_across_shells: false,
            show_fill: false,
            show_heatmap_sphere: true,
            heatmap_altitude_km: 200.0,
            heatmap_resolution: 36,
            dipole_offset_km: 450.0,
            heatmap_mode: HeatmapMode::Radiation,
            show_protons: true,
            show_electrons: true,
            smooth_colors: false,
            igrf_grid_cache: None,
            igrf_rad_cache: None,
            #[cfg(not(target_arch = "wasm32"))]
            igrf_rad_pending: None,
        }
    }
}

impl Default for KesslerSimulation {
    fn default() -> Self {
        Self {
            enabled: false,
            debris: Vec::new(),
            collision_threshold_km: 1.0,
            fragments_per_collision: 5,
            max_debris: 10000,
            collision_count: 0,
            collision_id_counter: 0,
            collided_pairs: HashSet::new(),
            course_correction_enabled: false,
            active_corrections: Vec::new(),
            correction_altitude_km: 2.0,
            corrections_made: 0,
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod shareable {
    use super::*;
    use base64::Engine;

    #[derive(serde::Serialize, serde::Deserialize)]
    pub struct ShareableShell {
        pub s: usize,
        pub p: usize,
        pub a: f64,
        pub i: f64,
        pub w: WalkerType,
        pub ph: f64,
        pub ro: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub rs: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub ss: Option<f64>,
        #[serde(default)]
        pub e: f64,
        #[serde(default)]
        pub ap: f64,
    }

    #[derive(serde::Serialize, serde::Deserialize)]
    pub struct ShareableConfig {
        pub body: CelestialBody,
        pub shells: Vec<ShareableShell>,
    }

    impl ShareableConfig {
        pub fn from_planet(planet: &PlanetConfig) -> Self {
            Self {
                body: planet.celestial_body,
                shells: planet.constellations.iter().map(|c| ShareableShell {
                    s: c.sats_per_plane,
                    p: c.num_planes,
                    a: c.altitude_km,
                    i: c.inclination,
                    w: c.walker_type,
                    ph: c.phasing,
                    ro: c.raan_offset,
                    rs: c.raan_spacing,
                    ss: c.sat_spacing_km,
                    e: c.eccentricity,
                    ap: c.arg_periapsis,
                }).collect(),
            }
        }

        pub fn apply_to_planet(&self, planet: &mut PlanetConfig) {
            planet.celestial_body = self.body;
            planet.constellations.clear();
            planet.constellation_counter = 0;
            for shell in &self.shells {
                let mut c = ConstellationConfig::new(planet.constellation_counter);
                planet.constellation_counter += 1;
                c.sats_per_plane = shell.s;
                c.num_planes = shell.p;
                c.altitude_km = shell.a;
                c.inclination = shell.i;
                c.walker_type = shell.w;
                c.phasing = shell.ph;
                c.raan_offset = shell.ro;
                c.raan_spacing = shell.rs;
                c.sat_spacing_km = shell.ss;
                c.eccentricity = shell.e;
                c.arg_periapsis = shell.ap;
                planet.constellations.push(c);
            }
        }

        pub fn to_url_hash(&self) -> String {
            let json = serde_json::to_string(self).unwrap_or_default();
            let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json.as_bytes());
            format!("#c={encoded}")
        }

        pub fn from_url_hash(hash: &str) -> Option<Self> {
            let data = hash.strip_prefix("#c=").or_else(|| hash.strip_prefix("c="))?;
            let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(data).ok()?;
            serde_json::from_slice(&bytes).ok()
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use shareable::ShareableConfig;
