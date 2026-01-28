use eframe::egui;
use egui_dock::{DockArea, DockState, TabViewer};
use egui_plot::{Line, Plot, PlotImage, PlotPoints, PlotPoint, Points, Polygon, Text};
use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;
use std::sync::Arc;

#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::JsCast;

#[derive(Clone, Copy, PartialEq, Debug)]
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

    fn filename(&self) -> &'static str {
        match self {
            CelestialBody::Earth => "textures/earth_2k.jpg",
            CelestialBody::Moon => "textures/moon_2k.jpg",
            CelestialBody::Mars => "textures/mars_2k.jpg",
            CelestialBody::Mercury => "textures/mercury_2k.jpg",
            CelestialBody::Venus => "textures/venus_2k.jpg",
            CelestialBody::Jupiter => "textures/jupiter_2k.jpg",
            CelestialBody::Saturn => "textures/saturn_2k.jpg",
            CelestialBody::Sun => "textures/sun_2k.jpg",
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
}

#[derive(Clone)]
#[allow(dead_code)]
enum TextureLoadState {
    Loading,
    Loaded(Arc<EarthTexture>),
    Failed(String),
}

const EARTH_RADIUS_KM: f64 = 6371.0;
const EARTH_TEXTURE_BYTES: &[u8] = include_bytes!("../earth.jpg");

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
        let orbit_radius = EARTH_RADIUS_KM + self.altitude_km;
        let period = 2.0 * PI * (orbit_radius.powi(3) / 398600.4418_f64).sqrt();
        let mean_motion = 2.0 * PI / period;
        let raan_spread = self.raan_spread();
        let inc = self.inclination_deg.to_radians();
        let inc_cos = inc.cos();
        let inc_sin = inc.sin();
        let raan_step = raan_spread / self.num_planes as f64;
        let sat_step = 2.0 * PI / sats_per_plane as f64;
        let is_star = self.walker_type == WalkerType::Star;

        for plane in 0..self.num_planes {
            let raan = raan_step * plane as f64;
            let raan_cos = raan.cos();
            let raan_sin = raan.sin();

            for sat in 0..sats_per_plane {
                let true_anomaly = sat_step * sat as f64 + mean_motion * time;
                let normalized_anomaly = true_anomaly.rem_euclid(2.0 * PI);
                let ascending = normalized_anomaly < PI;

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
        let orbit_radius = EARTH_RADIUS_KM + self.altitude_km;
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

#[derive(Clone)]
struct ConstellationConfig {
    sats_per_plane: usize,
    num_planes: usize,
    altitude_km: f64,
    inclination: f64,
    walker_type: WalkerType,
    preset: Preset,
    color_offset: usize,
}

impl ConstellationConfig {
    fn new(color_offset: usize) -> Self {
        Self {
            sats_per_plane: 11,
            num_planes: 6,
            altitude_km: 780.0,
            inclination: 86.4,
            walker_type: WalkerType::Star,
            preset: Preset::Iridium,
            color_offset,
        }
    }

    fn total_sats(&self) -> usize {
        self.sats_per_plane * self.num_planes
    }

    fn constellation(&self) -> WalkerConstellation {
        WalkerConstellation {
            walker_type: self.walker_type,
            total_sats: self.total_sats(),
            num_planes: self.num_planes,
            altitude_km: self.altitude_km,
            inclination_deg: self.inclination,
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
}

impl TabConfig {
    fn new(name: String) -> Self {
        Self {
            name,
            constellations: vec![ConstellationConfig::new(0)],
            constellation_counter: 1,
        }
    }

    fn add_constellation(&mut self) {
        self.constellations.push(ConstellationConfig::new(self.constellation_counter));
        self.constellation_counter += 1;
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
    dock_state: DockState<TabConfig>,
    satellite_cameras: Vec<SatelliteCamera>,
    camera_id_counter: usize,
    tab_counter: usize,
    time: f64,
    speed: f64,
    animate: bool,
    show_orbits: bool,
    show_links: bool,
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
    earth_texture: Arc<EarthTexture>,
    earth_image_handle: Option<egui::TextureHandle>,
    last_rotation: Option<Matrix3<f64>>,
    earth_resolution: usize,
    last_resolution: usize,
    celestial_body: CelestialBody,
    texture_load_state: TextureLoadState,
    pending_body: Option<CelestialBody>,
    dark_mode: bool,
    pending_cameras: Vec<SatelliteCamera>,
}

impl Default for App {
    fn default() -> Self {
        let torus_initial = Matrix3::new(
            1.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
            0.0, 1.0, 0.0,
        );
        let builtin_texture = Arc::new(EarthTexture::load());
        Self {
            dock_state: DockState::new(vec![TabConfig::new("Config 1".to_string())]),
            satellite_cameras: Vec::new(),
            camera_id_counter: 0,
            tab_counter: 1,
            time: 0.0,
            speed: 1.0,
            animate: true,
            show_orbits: true,
            show_links: true,
            show_ground_track: false,
            show_torus: false,
            show_axes: true,
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
            earth_texture: builtin_texture.clone(),
            earth_image_handle: None,
            last_rotation: None,
            earth_resolution: 512,
            last_resolution: 0,
            celestial_body: CelestialBody::Earth,
            texture_load_state: TextureLoadState::Loading,
            pending_body: Some(CelestialBody::Earth),
            dark_mode: true,
            pending_cameras: Vec::new(),
        }
    }
}

struct ConstellationTabViewer<'a> {
    time: f64,
    show_orbits: bool,
    show_links: bool,
    show_axes: bool,
    show_coverage: bool,
    coverage_angle: f64,
    show_torus: bool,
    show_ground: bool,
    hide_behind_earth: bool,
    single_color: bool,
    single_tab: bool,
    dark_mode: bool,
    pending_cameras: &'a mut Vec<SatelliteCamera>,
    camera_id_counter: &'a mut usize,
    satellite_cameras: &'a mut Vec<SatelliteCamera>,
    zoom: &'a mut f64,
    torus_zoom: &'a mut f64,
    vertical_split: &'a mut f32,
    sat_radius: f32,
    rotation: &'a mut Matrix3<f64>,
    torus_rotation: &'a mut Matrix3<f64>,
    earth_handle: Option<&'a egui::TextureHandle>,
}

impl<'a> TabViewer for ConstellationTabViewer<'a> {
    type Tab = TabConfig;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        tab.name.as_str().into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        let mut const_to_remove: Option<usize> = None;
        let num_constellations = tab.constellations.len();

        ui.horizontal(|ui| {
            for (cidx, cons) in tab.constellations.iter_mut().enumerate() {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        let label = if self.single_color {
                            format!("{} ({}):", cons.preset_name(), color_name(cons.color_offset))
                        } else {
                            format!("{}:", cons.preset_name())
                        };
                        ui.label(label);
                        if num_constellations > 1 {
                            if ui.small_button("x").clicked() {
                                const_to_remove = Some(cidx);
                            }
                        }
                    });

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
                        let alt_resp = ui.add(egui::DragValue::new(&mut cons.altitude_km).range(100.0..=40000.0).suffix(" km"));
                        if ui.small_button("LEO").clicked() {
                            cons.altitude_km = 160.0;
                            cons.preset = Preset::None;
                        }
                        if ui.small_button("MEO").clicked() {
                            cons.altitude_km = 2000.0;
                            cons.preset = Preset::None;
                        }
                        if ui.small_button("GEO").clicked() {
                            cons.altitude_km = 36000.0;
                            cons.preset = Preset::None;
                        }
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
                        // Note: Do not delete these comments
                        // https://en.wikipedia.org/wiki/Starlink
                        if ui.selectable_label(cons.preset == Preset::Starlink, "Starlink").clicked() {
                            cons.sats_per_plane = 22; cons.num_planes = 72;
                            cons.altitude_km = 550.0; cons.inclination = 53.0;
                            cons.walker_type = WalkerType::Delta;
                            cons.preset = Preset::Starlink;
                        }
                        // https://www.eoportal.org/satellite-missions/oneweb
                        if ui.selectable_label(cons.preset == Preset::OneWeb, "OneWeb").clicked() {
                            cons.sats_per_plane = 54; cons.num_planes = 12;
                            cons.altitude_km = 1200.0; cons.inclination = 87.9;
                            cons.walker_type = WalkerType::Star;
                            cons.preset = Preset::OneWeb;
                        }
                        // https://en.wikipedia.org/wiki/Iridium_satellite_constellation
                        if ui.selectable_label(cons.preset == Preset::Iridium, "Iridium").clicked() {
                            cons.sats_per_plane = 11; cons.num_planes = 6;
                            cons.altitude_km = 780.0; cons.inclination = 86.4;
                            cons.walker_type = WalkerType::Star;
                            cons.preset = Preset::Iridium;
                        }
                    });

                    ui.horizontal(|ui| {
                        // https://www.eoportal.org/satellite-missions/projectkuiper
                        if ui.selectable_label(cons.preset == Preset::Kuiper, "Kuiper").clicked() {
                            cons.sats_per_plane = 34; cons.num_planes = 34;
                            cons.altitude_km = 630.0; cons.inclination = 51.9;
                            cons.walker_type = WalkerType::Delta;
                            cons.preset = Preset::Kuiper;
                        }
                        // https://en.wikipedia.org/wiki/IRIS%C2%B2
                        if ui.selectable_label(cons.preset == Preset::Iris2, "Iris²").clicked() {
                            cons.sats_per_plane = 22; cons.num_planes = 12;
                            cons.altitude_km = 1200.0; cons.inclination = 87.0;
                            cons.walker_type = WalkerType::Star;
                            cons.preset = Preset::Iris2;
                        }
                        // https://www.eoportal.org/satellite-missions/telesat-lightspeed
                        if ui.selectable_label(cons.preset == Preset::Telesat, "Telesat").clicked() {
                            cons.sats_per_plane = 13; cons.num_planes = 6;
                            cons.altitude_km = 1015.0; cons.inclination = 98.98;
                            cons.walker_type = WalkerType::Star;
                            cons.preset = Preset::Telesat;
                        }
                    });

                    let orbit_radius = EARTH_RADIUS_KM + cons.altitude_km;
                    let intra_plane_dist = orbit_radius * (2.0 * (1.0 - (2.0 * PI / cons.sats_per_plane as f64).cos())).sqrt();
                    let inc_rad = cons.inclination.to_radians();
                    let base_inter = orbit_radius * (2.0 * (1.0 - (2.0 * PI / cons.num_planes as f64).cos())).sqrt();
                    let inter_plane_dist = base_inter * inc_rad.sin().abs().max(0.1);
                    let ground_dist = cons.altitude_km;

                    const SPEED_OF_LIGHT_KM_S: f64 = 299792.0;
                    let intra_latency_ms = intra_plane_dist / SPEED_OF_LIGHT_KM_S * 1000.0;
                    let inter_latency_ms = inter_plane_dist / SPEED_OF_LIGHT_KM_S * 1000.0;
                    let ground_latency_ms = ground_dist / SPEED_OF_LIGHT_KM_S * 1000.0;

                    ui.label(format!(
                        "Latency: intra {:.2}ms, inter {:.2}ms, ground {:.2}ms",
                        intra_latency_ms, inter_latency_ms, ground_latency_ms
                    ));
                });
                ui.separator();
            }

            if ui.small_button("+").clicked() {
                const_to_remove = Some(usize::MAX);
            }
        });

        if let Some(cidx) = const_to_remove {
            if cidx == usize::MAX {
                tab.add_constellation();
            } else {
                tab.constellations.remove(cidx);
            }
        }

        ui.separator();

        let constellations_data: Vec<_> = tab.constellations.iter()
            .map(|c| {
                let wc = c.constellation();
                let pos = wc.satellite_positions(self.time);
                (wc, pos, c.color_offset)
            })
            .collect();

        let available = ui.available_size();
        let use_horizontal = self.single_tab && self.show_torus && !self.show_ground;

        if use_horizontal {
            let half_width = (available.x - 15.0) / 2.0;
            let view_height = available.y - 20.0;
            let view_size = half_width.min(view_height);

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    let (rot, new_zoom) = draw_3d_view(
                        ui,
                        &tab.name,
                        &constellations_data,
                        self.show_orbits,
                        self.show_axes,
                        self.show_coverage,
                        self.coverage_angle,
                        *self.rotation,
                        half_width,
                        view_size,
                        self.earth_handle,
                        *self.zoom,
                        self.sat_radius,
                        self.show_links,
                        self.hide_behind_earth,
                        self.single_color,
                        self.dark_mode,
                        self.pending_cameras,
                        self.camera_id_counter,
                        self.satellite_cameras,
                    );
                    *self.rotation = rot;
                    *self.zoom = new_zoom;
                });

                ui.add_space(5.0);

                ui.vertical(|ui| {
                    let (trot, tzoom) = draw_torus(
                        ui,
                        &format!("torus_{}", tab.name),
                        &constellations_data,
                        self.time,
                        *self.torus_rotation,
                        half_width,
                        view_size,
                        self.sat_radius,
                        self.show_links,
                        self.single_color,
                        *self.torus_zoom,
                    );
                    *self.torus_rotation = trot;
                    *self.torus_zoom = tzoom;
                });
            });
        } else {
            let viz_width = available.x - 10.0;
            let available_for_views = available.y - 20.0;

            let has_secondary = self.show_torus || self.show_ground;
            let separator_height = if has_secondary { 8.0 } else { 0.0 };

            let earth_height = if has_secondary {
                (available_for_views - separator_height) * *self.vertical_split
            } else {
                available_for_views
            }.min(viz_width);

            let secondary_height = if has_secondary {
                (available_for_views - separator_height) * (1.0 - *self.vertical_split)
            } else {
                0.0
            };

            let (rot, new_zoom) = draw_3d_view(
                ui,
                &tab.name,
                &constellations_data,
                self.show_orbits,
                self.show_axes,
                self.show_coverage,
                self.coverage_angle,
                *self.rotation,
                viz_width,
                earth_height,
                self.earth_handle,
                *self.zoom,
                self.sat_radius,
                self.show_links,
                self.hide_behind_earth,
                self.single_color,
                self.dark_mode,
                self.pending_cameras,
                self.camera_id_counter,
                self.satellite_cameras,
            );
            *self.rotation = rot;
            *self.zoom = new_zoom;

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
                    *self.vertical_split = (*self.vertical_split + delta).clamp(0.2, 0.9);
                }

                if response.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
                }
            }

            if self.show_torus && self.show_ground {
                let torus_height = secondary_height * 0.6;
                let (trot, tzoom) = draw_torus(
                    ui,
                    &format!("torus_{}", tab.name),
                    &constellations_data,
                    self.time,
                    *self.torus_rotation,
                    viz_width,
                    torus_height,
                    self.sat_radius,
                    self.show_links,
                    self.single_color,
                    *self.torus_zoom,
                );
                *self.torus_rotation = trot;
                *self.torus_zoom = tzoom;

                let ground_height = secondary_height * 0.4;
                draw_ground_track(
                    ui,
                    &format!("ground_{}", tab.name),
                    &constellations_data,
                    viz_width,
                    ground_height,
                    self.sat_radius,
                    self.single_color,
                );
            } else if self.show_torus {
                let (trot, tzoom) = draw_torus(
                    ui,
                    &format!("torus_{}", tab.name),
                    &constellations_data,
                    self.time,
                    *self.torus_rotation,
                    viz_width,
                    secondary_height,
                    self.sat_radius,
                    self.show_links,
                    self.single_color,
                    *self.torus_zoom,
                );
                *self.torus_rotation = trot;
                *self.torus_zoom = tzoom;
            } else if self.show_ground {
                draw_ground_track(
                    ui,
                    &format!("ground_{}", tab.name),
                    &constellations_data,
                    viz_width,
                    secondary_height,
                    self.sat_radius,
                    self.single_color,
                );
            }
        }
    }

}

impl App {
    fn add_tab(&mut self) {
        self.tab_counter += 1;
        let tab = TabConfig::new(format!("Config {}", self.tab_counter));
        let tree = self.dock_state.main_surface_mut();
        let n = tree.num_tabs();
        let fraction = n as f32 / (n + 1) as f32;
        tree.split_right(egui_dock::NodeIndex::root(), fraction, vec![tab]);
    }

    fn balance_tabs(&mut self) {
        let tabs: Vec<TabConfig> = self.dock_state
            .main_surface_mut()
            .tabs()
            .cloned()
            .collect();

        if tabs.is_empty() {
            return;
        }

        let first = tabs[0].clone();
        self.dock_state = DockState::new(vec![first]);

        for tab in tabs.into_iter().skip(1) {
            let tree = self.dock_state.main_surface_mut();
            let n = tree.num_tabs();
            let fraction = n as f32 / (n + 1) as f32;
            tree.split_right(egui_dock::NodeIndex::root(), fraction, vec![tab]);
        }
    }

    fn show_settings(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.dark_mode, "Dark mode");
        ui.checkbox(&mut self.animate, "Animate");
        ui.checkbox(&mut self.show_orbits, "Show orbits");
        ui.checkbox(&mut self.show_links, "Show links");
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
        ui.checkbox(&mut self.show_torus, "Show torus");
        ui.checkbox(&mut self.show_ground_track, "Show ground");
        ui.checkbox(&mut self.hide_behind_earth, "Hide behind Earth");
        ui.checkbox(&mut self.single_color_per_constellation, "Single color per constellation");

        ui.add_space(10.0);

        ui.horizontal(|ui| {
            ui.label("Speed:");
            ui.add(egui::Slider::new(&mut self.speed, 0.1..=10.0).logarithmic(true));
        });

        ui.horizontal(|ui| {
            ui.label("Zoom:");
            ui.add(egui::Slider::new(&mut self.zoom, 0.5..=3.0).logarithmic(true));
        });

        ui.horizontal(|ui| {
            ui.label("Sat size:");
            ui.add(egui::Slider::new(&mut self.sat_radius, 1.0..=15.0));
        });

        ui.add_space(10.0);

        ui.horizontal(|ui| {
            ui.label("Body:");
            let current_label = match &self.texture_load_state {
                TextureLoadState::Loading => format!("{} (loading...)", self.celestial_body.label()),
                TextureLoadState::Failed(_) => format!("{} (failed)", self.celestial_body.label()),
                _ => self.celestial_body.label().to_string(),
            };
            egui::ComboBox::from_id_salt("celestial_body")
                .selected_text(current_label)
                .show_ui(ui, |ui| {
                    for body in CelestialBody::ALL {
                        let is_selected = self.celestial_body == body;
                        if ui.selectable_label(is_selected, body.label()).clicked() && !is_selected {
                            self.pending_body = Some(body);
                        }
                    }
                });
        });

        if let TextureLoadState::Failed(err) = &self.texture_load_state {
            ui.colored_label(egui::Color32::RED, format!("Error: {}", err));
        }

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

        if ui.button("Reset time").clicked() {
            self.time = 0.0;
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
    fn switch_texture(&mut self, body: CelestialBody, ctx: &egui::Context) {
        self.celestial_body = body;
        let filename = body.filename();
        self.texture_load_state = TextureLoadState::Loading;

        #[cfg(not(target_arch = "wasm32"))]
        {
            match std::fs::read(filename) {
                Ok(bytes) => match EarthTexture::from_bytes(&bytes) {
                    Ok(texture) => {
                        let texture = Arc::new(texture);
                        self.earth_texture = texture.clone();
                        self.texture_load_state = TextureLoadState::Loaded(texture);
                        self.last_rotation = None;
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

        if let Some(body) = self.pending_body.take() {
            self.switch_texture(body, ctx);
        }

        if self.animate {
            self.time += self.speed;
            ctx.request_repaint();
        }

        #[cfg(target_arch = "wasm32")]
        TEXTURE_RESULT.with(|cell| {
            if let Some(result) = cell.borrow_mut().take() {
                match result {
                    Ok(texture) => {
                        let texture = Arc::new(texture);
                        self.earth_texture = texture.clone();
                        self.texture_load_state = TextureLoadState::Loaded(texture);
                        self.last_rotation = None;
                    }
                    Err(e) => {
                        self.texture_load_state = TextureLoadState::Failed(e);
                    }
                }
            }
        });

        let rotation_changed = self.last_rotation.map_or(true, |r| r != self.rotation);
        let resolution_changed = self.last_resolution != self.earth_resolution;
        if self.earth_image_handle.is_none() || rotation_changed || resolution_changed {
            let earth_image = self.earth_texture.render_sphere(self.earth_resolution, &self.rotation);
            self.earth_image_handle = Some(ctx.load_texture(
                "earth",
                earth_image,
                egui::TextureOptions::LINEAR,
            ));
            self.last_rotation = Some(self.rotation);
            self.last_resolution = self.earth_resolution;
        }

        let is_mobile = ctx.screen_rect().width() < 600.0;

        if !is_mobile {
            egui::SidePanel::left("global_controls").show(ctx, |ui| {
                ui.heading("Display Settings");
                ui.separator();
                self.show_settings(ui);
            });
        }

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if is_mobile {
                    let menu_label = if self.menu_open { "Settings \u{25B2}" } else { "Settings \u{25BC}" };
                    if ui.button(menu_label).clicked() {
                        self.menu_open = !self.menu_open;
                    }
                }
                ui.heading("LEO Viz");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("+ Add Config").clicked() {
                        self.add_tab();
                    }
                    if ui.button("Balance").clicked() {
                        self.balance_tabs();
                    }
                });
            });

            if is_mobile && self.menu_open {
                ui.separator();
                self.show_settings(ui);
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let single_tab = self.dock_state.main_surface().num_tabs() == 1;
            let mut tab_viewer = ConstellationTabViewer {
                time: self.time,
                show_orbits: self.show_orbits,
                show_links: self.show_links,
                show_axes: self.show_axes,
                show_coverage: self.show_coverage,
                coverage_angle: self.coverage_angle,
                show_torus: self.show_torus,
                show_ground: self.show_ground_track,
                hide_behind_earth: self.hide_behind_earth,
                single_color: self.single_color_per_constellation,
                single_tab,
                zoom: &mut self.zoom,
                torus_zoom: &mut self.torus_zoom,
                vertical_split: &mut self.vertical_split,
                sat_radius: self.sat_radius,
                rotation: &mut self.rotation,
                torus_rotation: &mut self.torus_rotation,
                earth_handle: self.earth_image_handle.as_ref(),
                dark_mode: self.dark_mode,
                pending_cameras: &mut self.pending_cameras,
                camera_id_counter: &mut self.camera_id_counter,
                satellite_cameras: &mut self.satellite_cameras,
            };

            DockArea::new(&mut self.dock_state)
                .style(egui_dock::Style::from_egui(ui.style().as_ref()))
                .show_close_buttons(true)
                .show_inside(ui, &mut tab_viewer);
        });

        for camera in std::mem::take(&mut self.pending_cameras) {
            self.satellite_cameras.push(camera);
        }

        let all_tabs: Vec<_> = self.dock_state.main_surface().tabs().cloned().collect();

        let mut cameras_to_remove = Vec::new();
        for camera in &self.satellite_cameras {
            let mut open = true;

            let sat_data = all_tabs.first().and_then(|tab| {
                tab.constellations.get(camera.constellation_idx).map(|cons| {
                    let wc = cons.constellation();
                    let positions = wc.satellite_positions(self.time);
                    positions.iter()
                        .find(|s| s.plane == camera.plane && s.sat_index == camera.sat_index)
                        .map(|s| (s.lat, s.lon, cons.altitude_km))
                })
            }).flatten();

            if let Some((lat, lon, altitude_km)) = sat_data {
                let win_response = egui::Window::new(&camera.label)
                    .id(egui::Id::new(format!("sat_cam_{}", camera.id)))
                    .open(&mut open)
                    .default_size([200.0, 220.0])
                    .show(ctx, |ui| {
                        draw_satellite_camera(
                            ui,
                            camera.id,
                            lat,
                            lon,
                            altitude_km,
                            self.coverage_angle,
                            &self.earth_texture,
                        );
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
            if !open {
                cameras_to_remove.push(camera.id);
            }
        }
        self.satellite_cameras.retain(|c| !cameras_to_remove.contains(&c.id));
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
) {
    let size = ui.available_size();
    let img_size = size.x.min(size.y - 40.0) as usize;
    if img_size < 10 {
        return;
    }

    let lat_rad = lat.to_radians();
    let lon_rad = lon.to_radians();
    let fov_rad = coverage_angle.to_radians();

    let mut pixels = vec![egui::Color32::BLACK; img_size * img_size];

    for py in 0..img_size {
        for px in 0..img_size {
            let nx = (px as f64 / img_size as f64 - 0.5) * 2.0;
            let ny = (py as f64 / img_size as f64 - 0.5) * 2.0;

            let dist = (nx * nx + ny * ny).sqrt();
            if dist > 1.0 {
                continue;
            }

            let angle_from_nadir = dist * fov_rad;
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

fn draw_3d_view(
    ui: &mut egui::Ui,
    id: &str,
    constellations: &[(WalkerConstellation, Vec<SatelliteState>, usize)],
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
    hide_behind_earth: bool,
    single_color: bool,
    dark_mode: bool,
    pending_cameras: &mut Vec<SatelliteCamera>,
    camera_id_counter: &mut usize,
    satellite_cameras: &mut [SatelliteCamera],
) -> (Matrix3<f64>, f64) {
    let max_orbit_radius = constellations.iter()
        .map(|(c, _, _)| EARTH_RADIUS_KM + c.altitude_km)
        .fold(EARTH_RADIUS_KM, |a, b| a.max(b));
    let axis_len = max_orbit_radius * 1.05;
    let label_offset = axis_len * 1.1;
    let margin = (max_orbit_radius.max(label_offset) * 1.08) / zoom;

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

        let visual_earth_r = EARTH_RADIUS_KM * 0.95;
        let earth_r_sq = visual_earth_r * visual_earth_r;

        if show_orbits && !hide_behind_earth {
            for (constellation, _, color_offset) in constellations {
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
                                Line::new(PlotPoints::new(std::mem::take(&mut behind_segment)))
                                    .color(dim_color(color))
                                    .width(1.0),
                            );
                        }
                    }
                    if !behind_segment.is_empty() {
                        plot_ui.line(
                            Line::new(PlotPoints::new(behind_segment))
                                .color(dim_color(color))
                                .width(1.0),
                        );
                    }
                }
            }
        }

        if !hide_behind_earth {
            for (constellation, positions, color_offset) in constellations {
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
                        Points::new(pts)
                            .color(dim_color(color))
                            .radius(sat_radius * 0.8)
                            .filled(true),
                    );
                }
            }
        }

        if let Some(tex) = earth_texture {
            let size = egui::Vec2::splat(EARTH_RADIUS_KM as f32 * 2.0);
            plot_ui.image(PlotImage::new(
                tex,
                PlotPoint::new(0.0, 0.0),
                size,
            ));
        } else {
            let earth_pts: PlotPoints = (0..=100)
                .map(|i| {
                    let theta = 2.0 * PI * i as f64 / 100.0;
                    [EARTH_RADIUS_KM * theta.cos(), EARTH_RADIUS_KM * theta.sin()]
                })
                .collect();
            plot_ui.polygon(
                Polygon::new(earth_pts)
                    .fill_color(egui::Color32::from_rgb(30, 60, 120))
                    .stroke(egui::Stroke::new(2.0, egui::Color32::from_rgb(70, 130, 180))),
            );
        }

        if dark_mode {
            let border_radius = EARTH_RADIUS_KM * 0.95;
            let border_pts: PlotPoints = (0..=100)
                .map(|i| {
                    let theta = 2.0 * PI * i as f64 / 100.0;
                    [border_radius * theta.cos(), border_radius * theta.sin()]
                })
                .collect();
            plot_ui.line(Line::new(border_pts).color(egui::Color32::WHITE).width(1.0));
        }

        if show_coverage {
            for (constellation, positions, color_offset) in constellations {
                let orbit_radius = EARTH_RADIUS_KM + constellation.altitude_km;
                let cone_half_angle = coverage_angle.to_radians();
                let max_earth_angle = (EARTH_RADIUS_KM / orbit_radius).acos();
                let earth_central_angle = (orbit_radius * cone_half_angle.sin() / EARTH_RADIUS_KM).asin();
                let angular_radius = earth_central_angle.min(max_earth_angle);
                for sat in positions {
                    let lat = sat.lat.to_radians();
                    let lon = sat.lon.to_radians();

                    let coverage_pts: Vec<[f64; 2]> = (0..=32)
                        .filter_map(|i| {
                            let angle = 2.0 * PI * i as f64 / 32.0;

                            let clat = (lat.sin() * angular_radius.cos()
                                + lat.cos() * angular_radius.sin() * angle.cos())
                            .asin();
                            let clon = lon
                                + (angular_radius.sin() * angle.sin())
                                    .atan2(lat.cos() * angular_radius.cos()
                                        - lat.sin() * angular_radius.sin() * angle.cos());

                            let x = EARTH_RADIUS_KM * clat.cos() * clon.cos();
                            let y = EARTH_RADIUS_KM * clat.sin();
                            let z = EARTH_RADIUS_KM * clat.cos() * clon.sin();

                            let (rx, ry, rz) = rotate_point_matrix(x, y, z, &rotation);
                            if rz >= 0.0 {
                                Some([rx, ry])
                            } else {
                                None
                            }
                        })
                        .collect();

                    if coverage_pts.len() >= 3 {
                        let color = plane_color(sat.plane + color_offset);
                        let fill = egui::Color32::from_rgba_unmultiplied(
                            color.r(), color.g(), color.b(), 60
                        );
                        plot_ui.polygon(
                            Polygon::new(PlotPoints::new(coverage_pts))
                                .fill_color(fill)
                                .stroke(egui::Stroke::new(1.0, color)),
                        );
                    }
                }
            }
        }

        if show_axes {
            let (ep_x, ep_y, _) = rotate_point_matrix(axis_len, 0.0, 0.0, &rotation);
            let (wn_x, wn_y, _) = rotate_point_matrix(-axis_len, 0.0, 0.0, &rotation);
            plot_ui.line(
                Line::new(PlotPoints::new(vec![[wn_x, wn_y], [ep_x, ep_y]]))
                    .color(egui::Color32::from_rgb(255, 100, 100))
                    .width(1.5),
            );

            let (np_x, np_y, _) = rotate_point_matrix(0.0, axis_len, 0.0, &rotation);
            let (sn_x, sn_y, _) = rotate_point_matrix(0.0, -axis_len, 0.0, &rotation);
            plot_ui.line(
                Line::new(PlotPoints::new(vec![[sn_x, sn_y], [np_x, np_y]]))
                    .color(egui::Color32::from_rgb(100, 100, 255))
                    .width(1.5),
            );

            let label_offset = axis_len * 1.15;
            let (n_x, n_y, _) = rotate_point_matrix(0.0, label_offset, 0.0, &rotation);
            let (s_x, s_y, _) = rotate_point_matrix(0.0, -label_offset, 0.0, &rotation);
            let (e_x, e_y, _) = rotate_point_matrix(label_offset, 0.0, 0.0, &rotation);
            let (w_x, w_y, _) = rotate_point_matrix(-label_offset, 0.0, 0.0, &rotation);

            plot_ui.text(Text::new(PlotPoint::new(n_x, n_y), "N").color(egui::Color32::WHITE));
            plot_ui.text(Text::new(PlotPoint::new(s_x, s_y), "S").color(egui::Color32::WHITE));
            plot_ui.text(Text::new(PlotPoint::new(e_x, e_y), "E").color(egui::Color32::WHITE));
            plot_ui.text(Text::new(PlotPoint::new(w_x, w_y), "W").color(egui::Color32::WHITE));
        }

        if show_orbits {
            for (constellation, _, color_offset) in constellations {
                for plane in 0..constellation.num_planes {
                    let orbit_pts = constellation.orbit_points_3d(plane);
                    let color = plane_color(if single_color { *color_offset } else { plane + color_offset });

                    let mut front_segment: Vec<[f64; 2]> = Vec::new();
                    for &(x, y, z) in &orbit_pts {
                        let (rx, ry, rz) = rotate_point_matrix(x, y, z, &rotation);
                        let visible = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;
                        if visible {
                            front_segment.push([rx, ry]);
                        } else if !front_segment.is_empty() {
                            plot_ui.line(
                                Line::new(PlotPoints::new(std::mem::take(&mut front_segment)))
                                    .color(color)
                                    .width(1.5),
                            );
                        }
                    }
                    if !front_segment.is_empty() {
                        plot_ui.line(
                            Line::new(PlotPoints::new(front_segment))
                                .color(color)
                                .width(1.5),
                        );
                    }
                }
            }
        }

        if show_links {
            let link_color = egui::Color32::from_rgb(200, 200, 200);
            let link_dim = egui::Color32::from_rgba_unmultiplied(80, 80, 100, 100);
            for (_, positions, _) in constellations {
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
                        let color = if both_visible { link_color } else { link_dim };
                        plot_ui.line(
                            Line::new(PlotPoints::new(vec![[rx1, ry1], [rx2, ry2]]))
                                .color(color)
                                .width(1.0),
                        );
                    }
                }
            }
        }

        for (constellation, positions, color_offset) in constellations {
            for plane in 0..constellation.num_planes {
                let color = plane_color(if single_color { *color_offset } else { plane + color_offset });
                let dim_col = egui::Color32::from_rgba_unmultiplied(
                    color.r() / 2, color.g() / 2, color.b() / 2, 80,
                );

                let front_pts: PlotPoints = positions
                    .iter()
                    .filter(|s| s.plane == plane)
                    .filter_map(|s| {
                        let (rx, ry, rz) = rotate_point_matrix(s.x, s.y, s.z, &rotation);
                        let in_front = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;
                        if in_front { Some([rx, ry]) } else { None }
                    })
                    .collect();

                if !hide_behind_earth {
                    let back_pts: PlotPoints = positions
                        .iter()
                        .filter(|s| s.plane == plane)
                        .filter_map(|s| {
                            let (rx, ry, rz) = rotate_point_matrix(s.x, s.y, s.z, &rotation);
                            let behind = rz < 0.0 && (rx * rx + ry * ry) < earth_r_sq;
                            if behind { Some([rx, ry]) } else { None }
                        })
                        .collect();

                    plot_ui.points(
                        Points::new(back_pts)
                            .color(dim_col)
                            .radius(sat_radius * 0.8)
                            .filled(true),
                    );
                }
                plot_ui.points(
                    Points::new(front_pts)
                        .color(color)
                        .radius(sat_radius)
                        .filled(true),
                );
            }
        }
    });

    for (cidx, (_constellation, positions, color_offset)) in constellations.iter().enumerate() {
        for sat in positions {
            for cam in satellite_cameras.iter_mut() {
                if cam.constellation_idx == cidx && cam.plane == sat.plane && cam.sat_index == sat.sat_index {
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

        'hover: for (_constellation, positions, color_offset) in constellations {
            for sat in positions {
                let (rx, ry, rz) = rotate_point_matrix(sat.x, sat.y, sat.z, &rotation);
                let earth_r_sq = (EARTH_RADIUS_KM * 0.95).powi(2) as f64;
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

            'outer: for (cidx, (_constellation, positions, _color_offset)) in constellations.iter().enumerate() {
                for sat in positions {
                    let (rx, ry, rz) = rotate_point_matrix(sat.x, sat.y, sat.z, &rotation);
                    let earth_r_sq = (EARTH_RADIUS_KM * 0.95).powi(2) as f64;
                    let visible = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;
                    if !visible && hide_behind_earth {
                        continue;
                    }
                    let dx = rx - click_x;
                    let dy = ry - click_y;
                    if dx * dx + dy * dy < click_threshold * click_threshold {
                        *camera_id_counter += 1;
                        pending_cameras.push(SatelliteCamera {
                            id: *camera_id_counter,
                            label: format!("Sat {}-{}", sat.plane + 1, sat.sat_index + 1),
                            constellation_idx: cidx,
                            plane: sat.plane,
                            sat_index: sat.sat_index,
                            screen_pos: None,
                        });
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
            zoom = (zoom * factor).clamp(0.5, 3.0);
        }
        if let Some(touch) = ui.input(|i| i.multi_touch()) {
            let factor = touch.zoom_delta as f64;
            zoom = (zoom * factor).clamp(0.5, 3.0);
        }
    }

    (rotation, zoom)
}

fn draw_ground_track(
    ui: &mut egui::Ui,
    id: &str,
    constellations: &[(WalkerConstellation, Vec<SatelliteState>, usize)],
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
        for (constellation, positions, color_offset) in constellations {
            for plane in 0..constellation.num_planes {
                let color = plane_color(if single_color { *color_offset } else { plane + color_offset });
                let pts: PlotPoints = positions
                    .iter()
                    .filter(|s| s.plane == plane)
                    .map(|s| [s.lon, s.lat])
                    .collect();
                plot_ui.points(
                    Points::new(pts)
                        .color(color)
                        .radius(sat_radius)
                        .filled(true),
                );
            }
        }

        plot_ui.line(
            Line::new(PlotPoints::new(vec![[-180.0, 0.0], [180.0, 0.0]]))
                .color(egui::Color32::DARK_GRAY)
                .width(0.5),
        );
        plot_ui.line(
            Line::new(PlotPoints::new(vec![[0.0, -90.0], [0.0, 90.0]]))
                .color(egui::Color32::DARK_GRAY)
                .width(0.5),
        );
    });
}

fn draw_torus(
    ui: &mut egui::Ui,
    id: &str,
    constellations: &[(WalkerConstellation, Vec<SatelliteState>, usize)],
    time: f64,
    mut rotation: Matrix3<f64>,
    width: f32,
    height: f32,
    sat_radius: f32,
    show_links: bool,
    single_color: bool,
    mut zoom: f64,
) -> (Matrix3<f64>, f64) {
    let (major_radius, minor_radius) = if let Some((constellation, positions, _)) = constellations.first() {
        let sats_per_plane = constellation.sats_per_plane();
        let orbit_radius = EARTH_RADIUS_KM + constellation.altitude_km;

        let intra_plane_dist = 2.0 * orbit_radius * (PI / sats_per_plane as f64).sin();

        let mut total_link_dist = 0.0;
        let mut link_count = 0;
        for sat in positions {
            if let Some(neighbor_idx) = sat.neighbor_idx {
                let neighbor = &positions[neighbor_idx];
                let dx = sat.x - neighbor.x;
                let dy = sat.y - neighbor.y;
                let dz = sat.z - neighbor.z;
                total_link_dist += (dx*dx + dy*dy + dz*dz).sqrt();
                link_count += 1;
            }
        }

        let avg_inter_plane_dist = if link_count > 0 {
            total_link_dist / link_count as f64
        } else {
            intra_plane_dist
        };

        let ratio = avg_inter_plane_dist / intra_plane_dist;

        let inclination_rad = constellation.inclination_deg.to_radians();
        let inclination_factor = inclination_rad.sin().abs().max(0.1);

        let altitude_factor = orbit_radius / (EARTH_RADIUS_KM + 500.0);
        let major = altitude_factor * ratio * (sats_per_plane as f64 / constellation.num_planes as f64);
        let minor_base = altitude_factor * inclination_factor;
        let minor = minor_base.max(major * inclination_factor);

        let scale = 2.0 / (major + minor).max(1.0);
        (major * scale, minor * scale)
    } else {
        (2.0, 0.8)
    };

    let margin = (major_radius + minor_radius) * 1.3 / zoom;
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
            rotate_point_matrix(x, y, z, &rotation)
        };

        for (constellation, positions, color_offset) in constellations {
            let sats_per_plane = constellation.total_sats / constellation.num_planes;
            let orbit_radius = EARTH_RADIUS_KM + constellation.altitude_km;
            let period = 2.0 * PI * (orbit_radius.powi(3) / 398600.4418_f64).sqrt();
            let mean_motion = 2.0 * PI / period;

            let torus_pos = |plane: usize, sat_idx: usize| -> (f64, f64, f64) {
                let angle = 2.0 * PI * plane as f64 / constellation.num_planes as f64;
                let sat_spacing = 2.0 * PI * sat_idx as f64 / sats_per_plane as f64;
                let phase = sat_spacing + mean_motion * time;
                torus_point(angle, phase)
            };

            for plane in 0..constellation.num_planes {
                let angle = 2.0 * PI * plane as f64 / constellation.num_planes as f64;
                let color = plane_color(if single_color { *color_offset } else { plane + color_offset });
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
                                Line::new(PlotPoints::new(std::mem::take(&mut back_segment)))
                                    .color(dim_col)
                                    .width(1.0),
                            );
                        }
                    } else {
                        back_segment.push([rx, ry]);
                        if !front_segment.is_empty() {
                            plot_ui.line(
                                Line::new(PlotPoints::new(std::mem::take(&mut front_segment)))
                                    .color(color)
                                    .width(1.5),
                            );
                        }
                    }
                }
                if !front_segment.is_empty() {
                    plot_ui.line(Line::new(PlotPoints::new(front_segment)).color(color).width(1.5));
                }
                if !back_segment.is_empty() {
                    plot_ui.line(Line::new(PlotPoints::new(back_segment)).color(dim_col).width(1.0));
                }
            }

            if show_links {
                let link_color = egui::Color32::from_rgb(150, 150, 150);
                let link_dim = egui::Color32::from_rgba_unmultiplied(150, 150, 150, 140);
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
                        let color = if facing1 && facing2 { link_color } else { link_dim };
                        plot_ui.line(
                            Line::new(PlotPoints::new(vec![[x1, y1], [x2, y2]]))
                                .color(color)
                                .width(1.0),
                        );
                    }
                }
            }

            for plane in 0..constellation.num_planes {
                let color = plane_color(if single_color { *color_offset } else { plane + color_offset });
                let dim_col = egui::Color32::from_rgba_unmultiplied(
                    color.r(), color.g(), color.b(), 140,
                );
                let angle = 2.0 * PI * plane as f64 / constellation.num_planes as f64;

                for sat in positions.iter().filter(|s| s.plane == plane) {
                    let phase = 2.0 * PI * sat.sat_index as f64 / sats_per_plane as f64 + mean_motion * time;
                    let (x, y, _) = torus_pos(sat.plane, sat.sat_index);
                    let facing = is_facing_camera(angle, phase);
                    let (c, r) = if facing { (color, sat_radius) } else { (dim_col, sat_radius * 0.8) };
                    plot_ui.points(
                        Points::new(PlotPoints::new(vec![[x, y]]))
                            .color(c)
                            .radius(r)
                            .filled(true),
                    );
                }
            }
        }
    });

    if response.response.dragged() && !response.response.drag_started() {
        let drag = response.response.drag_delta();
        let delta_rot = rotation_from_drag(drag.x as f64 * 0.01, drag.y as f64 * 0.01);
        rotation = delta_rot * rotation;
    }

    if response.response.hovered() {
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0 {
            let factor = 1.0 + scroll as f64 * 0.001;
            zoom = (zoom * factor).clamp(0.5, 3.0);
        }
        if let Some(touch) = ui.input(|i| i.multi_touch()) {
            let factor = touch.zoom_delta as f64;
            zoom = (zoom * factor).clamp(0.5, 3.0);
        }
    }

    (rotation, zoom)
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
