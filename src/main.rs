use eframe::egui;
use egui_plot::{Line, Plot, PlotImage, PlotPoints, PlotPoint, Points, Polygon, Text};
use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;
use std::sync::Arc;

#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::JsCast;

const EARTH_RADIUS_KM: f64 = 6371.0;
const EARTH_TEXTURE_BYTES: &[u8] = include_bytes!("../earth.jpg");

struct EarthTexture {
    width: u32,
    height: u32,
    pixels: Vec<[u8; 3]>,
}

impl EarthTexture {
    fn load() -> Self {
        let img = image::load_from_memory(EARTH_TEXTURE_BYTES)
            .expect("Failed to load Earth texture")
            .to_rgb8();
        let width = img.width();
        let height = img.height();
        let pixels: Vec<[u8; 3]> = img.pixels().map(|p| p.0).collect();
        Self { width, height, pixels }
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
        let mut positions = Vec::new();
        let sats_per_plane = self.sats_per_plane();
        let orbit_radius = EARTH_RADIUS_KM + self.altitude_km;
        let period = 2.0 * PI * (orbit_radius.powi(3) / 398600.4418_f64).sqrt();
        let mean_motion = 2.0 * PI / period;
        let raan_spread = self.raan_spread();

        for plane in 0..self.num_planes {
            let raan = (raan_spread / self.num_planes as f64) * plane as f64;

            for sat in 0..sats_per_plane {
                let sat_spacing = (2.0 * PI / sats_per_plane as f64) * sat as f64;
                let true_anomaly = sat_spacing + mean_motion * time;
                let normalized_anomaly = true_anomaly.rem_euclid(2.0 * PI);
                let ascending = normalized_anomaly < PI;

                let inc = self.inclination_deg.to_radians();

                let x_orbital = orbit_radius * true_anomaly.cos();
                let y_orbital = orbit_radius * true_anomaly.sin();

                let x = x_orbital * raan.cos() - y_orbital * inc.cos() * raan.sin();
                let z = x_orbital * raan.sin() + y_orbital * inc.cos() * raan.cos();
                let y = y_orbital * inc.sin();

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
                });
            }
        }
        positions
    }

    fn orbit_points_3d(&self, plane: usize) -> Vec<(f64, f64, f64)> {
        let orbit_radius = EARTH_RADIUS_KM + self.altitude_km;
        let raan_spread = self.raan_spread();
        let raan = (raan_spread / self.num_planes as f64) * plane as f64;
        let inc = self.inclination_deg.to_radians();

        (0..=200)
            .map(|i| {
                let theta = 2.0 * PI * i as f64 / 200.0;
                let x_orbital = orbit_radius * theta.cos();
                let y_orbital = orbit_radius * theta.sin();

                let x = x_orbital * raan.cos() - y_orbital * inc.cos() * raan.sin();
                let z = x_orbital * raan.sin() + y_orbital * inc.cos() * raan.cos();
                let y = y_orbital * inc.sin();

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

struct App {
    tabs: Vec<TabConfig>,
    tab_counter: usize,
    time: f64,
    speed: f64,
    animate: bool,
    show_orbits: bool,
    show_links: bool,
    show_ground_track: bool,
    show_torus: bool,
    hide_behind_earth: bool,
    single_color_per_constellation: bool,
    zoom: f64,
    sat_radius: f32,
    rotation: Matrix3<f64>,
    torus_rotation: Matrix3<f64>,
    earth_texture: Arc<EarthTexture>,
    earth_image_handle: Option<egui::TextureHandle>,
    last_rotation: Option<Matrix3<f64>>,
    earth_resolution: usize,
    last_resolution: usize,
}

impl Default for App {
    fn default() -> Self {
        let torus_initial = Matrix3::new(
            1.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
            0.0, 1.0, 0.0,
        );
        Self {
            tabs: vec![TabConfig::new("Config 1".to_string())],
            tab_counter: 1,
            time: 0.0,
            speed: 1.0,
            animate: true,
            show_orbits: true,
            show_links: true,
            show_ground_track: false,
            show_torus: false,
            hide_behind_earth: true,
            single_color_per_constellation: false,
            zoom: 1.0,
            sat_radius: 5.0,
            rotation: Matrix3::identity(),
            torus_rotation: torus_initial,
            earth_texture: Arc::new(EarthTexture::load()),
            earth_image_handle: None,
            last_rotation: None,
            earth_resolution: 512,
            last_resolution: 0,
        }
    }
}

impl App {
    fn add_tab(&mut self) {
        self.tab_counter += 1;
        self.tabs.push(TabConfig::new(format!("Config {}", self.tab_counter)));
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.animate {
            self.time += self.speed;
            ctx.request_repaint();
        }

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

        egui::SidePanel::left("global_controls").show(ctx, |ui| {
            ui.heading("Display Settings");
            ui.separator();

            ui.checkbox(&mut self.animate, "Animate");
            ui.checkbox(&mut self.show_orbits, "Show orbits");
            ui.checkbox(&mut self.show_links, "Show links");
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

            ui.add_space(20.0);
            ui.separator();
            ui.label("Delta: RAAN spread 360°");
            ui.label("Star: RAAN spread 180°");
            ui.add_space(5.0);
            ui.label("Drag 3D views to rotate");
        });

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("LEO Viz");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("+ Add Config").clicked() {
                        self.add_tab();
                    }
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let time = self.time;
            let show_orbits = self.show_orbits;
            let show_links = self.show_links;
            let show_torus = self.show_torus;
            let show_ground = self.show_ground_track;
            let hide_behind_earth = self.hide_behind_earth;
            let single_color = self.single_color_per_constellation;
            let zoom = self.zoom;
            let sat_radius = self.sat_radius;
            let earth_handle = self.earth_image_handle.clone();

            let mut new_rotation = self.rotation;
            let mut new_torus_rotation = self.torus_rotation;

            let available_width = ui.available_width();
            let available_height = ui.available_height();
            let num_tabs = self.tabs.len().max(1) as f32;
            let separator_space = (num_tabs - 1.0).max(0.0) * 10.0;

            let controls_height = 120.0;
            let height_ratio = 0.8
                + if show_torus { 0.5 } else { 0.0 }
                + if show_ground { 0.35 } else { 0.0 };
            let max_width_from_height = (available_height - controls_height) / height_ratio;

            let width_based = (available_width - separator_space) / num_tabs;
            let panel_width = width_based.min(max_width_from_height).max(200.0);

            let mut tab_to_remove: Option<usize> = None;

            ui.horizontal(|ui| {
                for (idx, tab) in self.tabs.iter_mut().enumerate() {
                    ui.vertical(|ui| {
                        ui.set_min_width(panel_width);
                        ui.set_max_width(panel_width);

                            ui.horizontal(|ui| {
                                ui.strong(&tab.name);
                                if ui.small_button("X").clicked() {
                                    tab_to_remove = Some(idx);
                                }
                            });

                            let mut const_to_remove: Option<usize> = None;
                            let num_constellations = tab.constellations.len();

                            ui.horizontal(|ui| {
                                for (cidx, cons) in tab.constellations.iter_mut().enumerate() {
                                    ui.vertical(|ui| {
                                        ui.horizontal(|ui| {
                                            let label = if single_color {
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
                                            let alt_resp = ui.add(egui::DragValue::new(&mut cons.altitude_km).range(200.0..=36000.0).suffix(" km"));
                                            ui.label("Inc:");
                                            let inc_resp = ui.add(egui::DragValue::new(&mut cons.inclination).range(0.0..=180.0).suffix("°"));
                                            if alt_resp.changed() || inc_resp.changed() {
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
                                    let pos = wc.satellite_positions(time);
                                    (wc, pos, c.color_offset)
                                })
                                .collect();

                            let viz_width = panel_width - 10.0;
                            let viz_height = viz_width * 0.8;

                            let rot = draw_3d_view(
                                ui,
                                &format!("earth_3d_{}", idx),
                                &constellations_data,
                                show_orbits,
                                new_rotation,
                                viz_width,
                                viz_height,
                                earth_handle.as_ref(),
                                zoom,
                                sat_radius,
                                show_links,
                                hide_behind_earth,
                                single_color,
                            );
                            if rot != new_rotation {
                                new_rotation = rot;
                            }

                            if show_torus {
                                ui.add_space(5.0);
                                let trot = draw_torus(
                                    ui,
                                    &format!("torus_{}", idx),
                                    &constellations_data,
                                    time,
                                    new_torus_rotation,
                                    viz_width,
                                    viz_width * 0.5,
                                    sat_radius,
                                    show_links,
                                    single_color,
                                );
                                if trot != new_torus_rotation {
                                    new_torus_rotation = trot;
                                }
                            }

                            if show_ground {
                                ui.add_space(5.0);
                                draw_ground_track(
                                    ui,
                                    &format!("ground_{}", idx),
                                    &constellations_data,
                                    viz_width,
                                    viz_width * 0.35,
                                    sat_radius,
                                    single_color,
                                );
                            }
                        });

                    ui.separator();
                }
            });

            if let Some(idx) = tab_to_remove {
                if self.tabs.len() > 1 {
                    self.tabs.remove(idx);
                }
            }

            self.rotation = new_rotation;
            self.torus_rotation = new_torus_rotation;
        });
    }
}

fn draw_3d_view(
    ui: &mut egui::Ui,
    id: &str,
    constellations: &[(WalkerConstellation, Vec<SatelliteState>, usize)],
    show_orbits: bool,
    mut rotation: Matrix3<f64>,
    width: f32,
    height: f32,
    earth_texture: Option<&egui::TextureHandle>,
    zoom: f64,
    sat_radius: f32,
    show_links: bool,
    hide_behind_earth: bool,
    single_color: bool,
) -> Matrix3<f64> {
    let max_orbit_radius = constellations.iter()
        .map(|(c, _, _)| EARTH_RADIUS_KM + c.altitude_km)
        .fold(0.0_f64, |a, b| a.max(b));
    let axis_len = EARTH_RADIUS_KM * 1.1;
    let label_offset = axis_len * 1.15;
    let margin = (max_orbit_radius.max(label_offset) * 1.08) / zoom;

    let plot = Plot::new(id)
        .data_aspect(1.0)
        .width(width)
        .height(height)
        .show_axes(false)
        .show_grid(false)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false);

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
                                Line::new(PlotPoints::new(behind_segment.clone()))
                                    .color(dim_color(color))
                                    .width(1.0),
                            );
                            behind_segment.clear();
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

        let (ep_x, ep_y, _) = rotate_point_matrix(axis_len, 0.0, 0.0, &rotation);
        let (ei_x, ei_y, _) = rotate_point_matrix(EARTH_RADIUS_KM, 0.0, 0.0, &rotation);
        plot_ui.line(
            Line::new(PlotPoints::new(vec![[ei_x, ei_y], [ep_x, ep_y]]))
                .color(egui::Color32::from_rgb(255, 100, 100))
                .width(1.5),
        );
        let (wn_x, wn_y, _) = rotate_point_matrix(-axis_len, 0.0, 0.0, &rotation);
        let (wi_x, wi_y, _) = rotate_point_matrix(-EARTH_RADIUS_KM, 0.0, 0.0, &rotation);
        plot_ui.line(
            Line::new(PlotPoints::new(vec![[wn_x, wn_y], [wi_x, wi_y]]))
                .color(egui::Color32::from_rgb(255, 100, 100))
                .width(1.5),
        );

        let (np_x, np_y, _) = rotate_point_matrix(0.0, axis_len, 0.0, &rotation);
        let (ni_x, ni_y, _) = rotate_point_matrix(0.0, EARTH_RADIUS_KM, 0.0, &rotation);
        plot_ui.line(
            Line::new(PlotPoints::new(vec![[ni_x, ni_y], [np_x, np_y]]))
                .color(egui::Color32::from_rgb(100, 100, 255))
                .width(1.5),
        );
        let (sn_x, sn_y, _) = rotate_point_matrix(0.0, -axis_len, 0.0, &rotation);
        let (si_x, si_y, _) = rotate_point_matrix(0.0, -EARTH_RADIUS_KM, 0.0, &rotation);
        plot_ui.line(
            Line::new(PlotPoints::new(vec![[sn_x, sn_y], [si_x, si_y]]))
                .color(egui::Color32::from_rgb(100, 100, 255))
                .width(1.5),
        );

        let label_offset = axis_len * 1.15;
        let (n_x, n_y, _) = rotate_point_matrix(0.0, label_offset, 0.0, &rotation);
        let (s_x, s_y, _) = rotate_point_matrix(0.0, -label_offset, 0.0, &rotation);
        let (e_x, e_y, _) = rotate_point_matrix(label_offset, 0.0, 0.0, &rotation);
        let (w_x, w_y, _) = rotate_point_matrix(-label_offset, 0.0, 0.0, &rotation);

        plot_ui.text(Text::new(PlotPoint::new(n_x, n_y), "N").color(egui::Color32::BLACK));
        plot_ui.text(Text::new(PlotPoint::new(s_x, s_y), "S").color(egui::Color32::BLACK));
        plot_ui.text(Text::new(PlotPoint::new(e_x, e_y), "E").color(egui::Color32::BLACK));
        plot_ui.text(Text::new(PlotPoint::new(w_x, w_y), "W").color(egui::Color32::BLACK));

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
                                Line::new(PlotPoints::new(front_segment.clone()))
                                    .color(color)
                                    .width(1.5),
                            );
                            front_segment.clear();
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
            for (constellation, positions, _) in constellations {
                let is_star = constellation.walker_type == WalkerType::Star;
                for sat in positions {
                    if is_star && sat.plane == constellation.num_planes - 1 {
                        continue;
                    }
                    let next_plane = (sat.plane + 1) % constellation.num_planes;
                    if let Some(neighbor) = positions.iter().find(|s| {
                        s.plane == next_plane && s.sat_index == sat.sat_index && s.ascending == sat.ascending
                    }) {
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

    if response.response.dragged() && !response.response.drag_started() {
        let drag = response.response.drag_delta();
        let delta_rot = rotation_from_drag(drag.x as f64 * 0.01, drag.y as f64 * 0.01);
        rotation = delta_rot * rotation;
    }

    rotation
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
) -> Matrix3<f64> {
    let major_radius = 2.0;
    let minor_radius = 0.8;

    let margin = (major_radius + minor_radius) * 1.3;
    let plot = Plot::new(id)
        .data_aspect(1.0)
        .width(width)
        .height(height)
        .show_axes(false)
        .show_grid(false)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false);

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
            let is_star = constellation.walker_type == WalkerType::Star;

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
                                Line::new(PlotPoints::new(back_segment.clone()))
                                    .color(dim_col)
                                    .width(1.0),
                            );
                            back_segment.clear();
                        }
                    } else {
                        back_segment.push([rx, ry]);
                        if !front_segment.is_empty() {
                            plot_ui.line(
                                Line::new(PlotPoints::new(front_segment.clone()))
                                    .color(color)
                                    .width(1.5),
                            );
                            front_segment.clear();
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
                    if is_star && sat.plane == constellation.num_planes - 1 {
                        continue;
                    }
                    let next_plane = (sat.plane + 1) % constellation.num_planes;
                    if let Some(neighbor) = positions.iter().find(|s| {
                        s.plane == next_plane && s.sat_index == sat.sat_index && s.ascending == sat.ascending
                    }) {
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

    rotation
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
