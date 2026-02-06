//! Configuration types for tabs, planets, constellations, and ground elements.
//!
//! Defines TabConfig, PlanetConfig, ConstellationConfig, presets, ground
//! stations, areas of interest, device layers, and per-tab view settings.

use crate::celestial::{CelestialBody, Skin};
use crate::math::lat_lon_to_matrix;
use crate::tle::{TlePreset, TleLoadState, TleShell};
use crate::walker::{WalkerType, WalkerConstellation};
use eframe::egui;
use nalgebra::Matrix3;
use std::collections::HashMap;

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
    pub walker_type: WalkerType,
    pub phasing: f64,
    pub raan_offset: f64,
    pub raan_spacing: Option<f64>,
    pub eccentricity: f64,
    pub arg_periapsis: f64,
    pub drag_enabled: bool,
    pub ballistic_coeff: f64,
    pub preset: Preset,
    pub color_offset: usize,
    pub hidden: bool,
}

impl ConstellationConfig {
    pub fn new(color_offset: usize) -> Self {
        Self {
            sats_per_plane: 30,
            num_planes: 30,
            altitude_km: 200.0,
            inclination: 90.0,
            walker_type: WalkerType::Delta,
            phasing: 0.0,
            raan_offset: 0.0,
            raan_spacing: None,
            eccentricity: 0.0,
            arg_periapsis: 0.0,
            drag_enabled: false,
            ballistic_coeff: 100.0,
            preset: Preset::None,
            color_offset,
            hidden: false,
        }
    }

    pub fn total_sats(&self) -> usize {
        self.sats_per_plane * self.num_planes
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
            eccentricity: self.eccentricity,
            arg_periapsis_deg: self.arg_periapsis,
            planet_radius,
            planet_mu,
            planet_j2,
            planet_equatorial_radius,
        }
    }

    pub fn preset_name(&self) -> &'static str {
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
}

#[derive(Clone)]
pub struct AreaOfInterest {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub radius_km: f64,
    pub color: egui::Color32,
    pub ground_station_idx: Option<usize>,
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
        }
    }

    pub fn add_constellation(&mut self) {
        self.constellations.push(ConstellationConfig::new(self.constellation_counter));
        self.constellation_counter += 1;
    }
}

#[derive(Clone, Copy)]
pub struct View3DFlags {
    pub show_orbits: bool,
    pub show_axes: bool,
    pub show_coverage: bool,
    pub show_links: bool,
    pub show_intra_links: bool,
    pub hide_behind_earth: bool,
    pub single_color: bool,
    pub dark_mode: bool,
    pub show_routing_paths: bool,
    pub show_manhattan_path: bool,
    pub show_shortest_path: bool,
    pub show_asc_desc_colors: bool,
    pub show_altitude_lines: bool,
    pub render_planet: bool,
    pub fixed_sizes: bool,
    pub show_polar_circle: bool,
    pub show_equator: bool,
    pub show_terminator: bool,
    pub earth_fixed_camera: bool,
    pub use_gpu_rendering: bool,
    pub show_clouds: bool,
    pub show_day_night: bool,
    pub show_stars: bool,
    pub show_milky_way: bool,
    pub show_borders: bool,
    pub show_cities: bool,
}

#[derive(Clone)]
pub struct TabSettings {
    pub time: f64,
    pub speed: f64,
    pub animate: bool,
    pub zoom: f64,
    pub rotation: Matrix3<f64>,
    pub earth_fixed_camera: bool,
    pub follow_satellite: bool,
    pub show_camera_windows: bool,
    pub show_orbits: bool,
    pub show_links: bool,
    pub show_intra_links: bool,
    pub show_coverage: bool,
    pub coverage_angle: f64,
    pub show_routing_paths: bool,
    pub show_manhattan_path: bool,
    pub show_shortest_path: bool,
    pub show_asc_desc_colors: bool,
    pub single_color: bool,
    pub show_torus: bool,
    pub show_ground_track: bool,
    pub show_axes: bool,
    pub hide_behind_earth: bool,
    pub render_planet: bool,
    pub show_altitude_lines: bool,
    pub show_devices: bool,
    pub show_polar_circle: bool,
    pub show_equator: bool,
    pub show_borders: bool,
    pub show_cities: bool,
    pub show_day_night: bool,
    pub show_terminator: bool,
    pub show_clouds: bool,
    pub show_stars: bool,
    pub show_milky_way: bool,
    pub sat_radius: f32,
    pub link_width: f32,
    pub fixed_sizes: bool,
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
            follow_satellite: false,
            show_camera_windows: false,
            show_orbits: true,
            show_links: true,
            show_intra_links: false,
            show_coverage: false,
            coverage_angle: 25.0,
            show_routing_paths: false,
            show_manhattan_path: true,
            show_shortest_path: true,
            show_asc_desc_colors: false,
            single_color: false,
            show_torus: false,
            show_ground_track: false,
            show_axes: false,
            hide_behind_earth: true,
            render_planet: true,
            show_altitude_lines: false,
            show_devices: false,
            show_polar_circle: false,
            show_equator: false,
            show_borders: false,
            show_cities: false,
            show_day_night: false,
            show_terminator: false,
            show_clouds: false,
            show_stars: false,
            show_milky_way: false,
            sat_radius: 1.5,
            link_width: 0.25,
            fixed_sizes: false,
        }
    }
}

pub struct TabConfig {
    pub name: String,
    pub planets: Vec<PlanetConfig>,
    pub planet_counter: usize,
    pub show_stats: bool,
    pub settings: TabSettings,
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
            planets: Vec::new(),
            planet_counter: 0,
            show_stats: false,
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
