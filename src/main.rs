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
                    let lon = v.z.atan2(v.x);

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

struct App {
    sats_per_plane: usize,
    num_planes: usize,
    altitude_km: f64,
    delta_inclination: f64,
    star_inclination: f64,
    time: f64,
    speed: f64,
    animate: bool,
    show_orbits: bool,
    show_ground_track: bool,
    show_torus: bool,
    show_star: bool,
    rotation: Matrix3<f64>,
    torus_rotation: Matrix3<f64>,
    earth_texture: Arc<EarthTexture>,
    earth_image_handle: Option<egui::TextureHandle>,
    last_rotation: Option<Matrix3<f64>>,
    last_resolution: usize,
    zoom: f64,
    earth_resolution: usize,
}

impl Default for App {
    fn default() -> Self {
        let torus_initial = Matrix3::new(
            1.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
            0.0, 1.0, 0.0,
        );
        Self {
            sats_per_plane: 11,
            num_planes: 6,
            altitude_km: 780.0,
            delta_inclination: 53.0,
            star_inclination: 86.4,
            time: 0.0,
            speed: 1.0,
            animate: true,
            show_orbits: true,
            show_ground_track: false,
            show_torus: false,
            show_star: false,
            rotation: Matrix3::identity(),
            torus_rotation: torus_initial,
            earth_texture: Arc::new(EarthTexture::load()),
            earth_image_handle: None,
            last_rotation: None,
            last_resolution: 0,
            zoom: 1.0,
            earth_resolution: 256,
        }
    }
}

impl App {
    fn total_sats(&self) -> usize {
        self.sats_per_plane * self.num_planes
    }

    fn delta_constellation(&self) -> WalkerConstellation {
        WalkerConstellation {
            walker_type: WalkerType::Delta,
            total_sats: self.total_sats(),
            num_planes: self.num_planes,
            altitude_km: self.altitude_km,
            inclination_deg: self.delta_inclination,
        }
    }

    fn star_constellation(&self) -> WalkerConstellation {
        WalkerConstellation {
            walker_type: WalkerType::Star,
            total_sats: self.total_sats(),
            num_planes: self.num_planes,
            altitude_km: self.altitude_km,
            inclination_deg: self.star_inclination,
        }
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

        let delta = self.delta_constellation();
        let star = self.star_constellation();
        let delta_positions = delta.satellite_positions(self.time);
        let star_positions = star.satellite_positions(self.time);

        egui::SidePanel::left("controls").show(ctx, |ui| {
            ui.heading("Walker Constellations");

            ui.add_space(10.0);
            ui.label("Configuration");
            ui.separator();

            let mut sats = self.sats_per_plane as i32;
            let mut planes = self.num_planes as i32;

            ui.horizontal(|ui| {
                ui.label("Sats per orbit:");
                ui.add(egui::DragValue::new(&mut sats).range(1..=50));
            });

            ui.horizontal(|ui| {
                ui.label("Orbital planes:");
                ui.add(egui::DragValue::new(&mut planes).range(1..=20));
            });

            if sats > 0 && planes > 0 {
                self.sats_per_plane = sats as usize;
                self.num_planes = planes as usize;
            }

            ui.horizontal(|ui| {
                ui.label("Altitude (km):");
                ui.add(egui::DragValue::new(&mut self.altitude_km).range(200.0..=36000.0));
            });

            ui.add_space(10.0);
            ui.separator();
            ui.label("Inclinations");

            ui.horizontal(|ui| {
                ui.label("Delta (°):");
                ui.add(egui::DragValue::new(&mut self.delta_inclination).range(0.0..=180.0));
            });

            ui.horizontal(|ui| {
                ui.label("Star (°):");
                ui.add(egui::DragValue::new(&mut self.star_inclination).range(0.0..=180.0));
            });

            ui.add_space(10.0);
            ui.separator();

            ui.checkbox(&mut self.animate, "Animate");
            ui.checkbox(&mut self.show_orbits, "Show orbits");
            ui.checkbox(&mut self.show_ground_track, "Show ground track");
            ui.checkbox(&mut self.show_torus, "Show torus");
            ui.checkbox(&mut self.show_star, "Show Walker Star");

            ui.horizontal(|ui| {
                ui.label("Speed:");
                ui.add(egui::Slider::new(&mut self.speed, 0.1..=10.0).logarithmic(true));
            });

            ui.horizontal(|ui| {
                ui.label("Zoom:");
                ui.add(egui::Slider::new(&mut self.zoom, 0.5..=3.0).logarithmic(true));
            });

            ui.horizontal(|ui| {
                ui.label("Earth res:");
                let mut res = self.earth_resolution as i32;
                if ui.add(egui::Slider::new(&mut res, 64..=512).step_by(64.0)).changed() {
                    self.earth_resolution = res as usize;
                }
            });

            if ui.button("Reset time").clicked() {
                self.time = 0.0;
            }

            if ui.button("Reset view").clicked() {
                self.rotation = Matrix3::identity();
                self.torus_rotation = Matrix3::new(
                    1.0, 0.0, 0.0,
                    0.0, 0.0, -1.0,
                    0.0, 1.0, 0.0,
                );
                self.zoom = 1.0;
            }

            ui.add_space(20.0);
            ui.label(format!("Total: {} satellites", self.total_sats()));

            ui.add_space(10.0);
            ui.separator();
            ui.label("Delta: RAAN spread 360°");
            ui.label("Star: RAAN spread 180°");
            ui.add_space(5.0);
            ui.label("Drag 3D views to rotate");
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let show_orbits = self.show_orbits;
            let show_ground = self.show_ground_track;
            let show_torus = self.show_torus;
            let show_star = self.show_star;

            let available_width = ui.available_width();
            let available_height = ui.available_height();
            let plot_width = if show_star {
                (available_width - 30.0) / 2.0
            } else {
                available_width - 20.0
            };
            let (plot_3d_height, plot_ground_height, plot_torus_height) = match (show_ground, show_torus) {
                (true, true) => {
                    let h = available_height - 80.0;
                    (h * 0.4, h * 0.25, h * 0.35)
                }
                (true, false) => {
                    let h = available_height - 60.0;
                    (h * 0.6, h * 0.4, 0.0)
                }
                (false, true) => {
                    let h = available_height - 60.0;
                    (h * 0.5, 0.0, h * 0.5)
                }
                (false, false) => {
                    let h = available_height - 40.0;
                    (h, 0.0, 0.0)
                }
            };

            let mut rotation = self.rotation;
            let mut torus_rotation = self.torus_rotation;

            let earth_handle = self.earth_image_handle.clone();

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.heading("Walker Delta");
                    rotation = draw_3d_view(
                        ui,
                        "delta_3d",
                        &delta,
                        &delta_positions,
                        show_orbits,
                        rotation,
                        plot_width,
                        plot_3d_height,
                        earth_handle.as_ref(),
                        self.zoom,
                    );
                    if show_ground {
                        ui.add_space(5.0);
                        draw_ground_track(ui, "delta_ground", &delta_positions, delta.num_planes, plot_width, plot_ground_height);
                    }
                    if show_torus {
                        ui.add_space(5.0);
                        torus_rotation = draw_torus(ui, "delta_torus", &delta, &delta_positions, self.time, torus_rotation, plot_width, plot_torus_height);
                    }
                });

                if show_star {
                    ui.add_space(20.0);

                    ui.vertical(|ui| {
                        ui.heading("Walker Star");
                        rotation = draw_3d_view(
                            ui,
                            "star_3d",
                            &star,
                            &star_positions,
                            show_orbits,
                            rotation,
                            plot_width,
                            plot_3d_height,
                            earth_handle.as_ref(),
                            self.zoom,
                        );
                        if show_ground {
                            ui.add_space(5.0);
                            draw_ground_track(ui, "star_ground", &star_positions, star.num_planes, plot_width, plot_ground_height);
                        }
                        if show_torus {
                            ui.add_space(5.0);
                            torus_rotation = draw_torus(ui, "star_torus", &star, &star_positions, self.time, torus_rotation, plot_width, plot_torus_height);
                        }
                    });
                }
            });

            self.rotation = rotation;
            self.torus_rotation = torus_rotation;
        });
    }
}

fn draw_3d_view(
    ui: &mut egui::Ui,
    id: &str,
    constellation: &WalkerConstellation,
    positions: &[SatelliteState],
    show_orbits: bool,
    mut rotation: Matrix3<f64>,
    width: f32,
    height: f32,
    earth_texture: Option<&egui::TextureHandle>,
    zoom: f64,
) -> Matrix3<f64> {
    let orbit_radius = EARTH_RADIUS_KM + constellation.altitude_km;
    let axis_len = EARTH_RADIUS_KM * 1.5;
    let margin = (orbit_radius.max(axis_len) * 1.25) / zoom;
    let total_sats = constellation.total_sats as f32;
    let sat_radius = ((width.min(height) / 100.0) / (total_sats / 100.0).sqrt()).clamp(2.0, 10.0);

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

        if show_orbits {
            for plane in 0..constellation.num_planes {
                let orbit_pts = constellation.orbit_points_3d(plane);
                let color = plane_color(plane);

                let mut behind_segment: Vec<[f64; 2]> = Vec::new();
                for &(x, y, z) in &orbit_pts {
                    let (rx, ry, rz) = rotate_point_matrix(x, y, z, &rotation);
                    if rz < 0.0 {
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

        for plane in 0..constellation.num_planes {
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
                    .color(dim_color(plane_color(plane)))
                    .radius(sat_radius * 0.8)
                    .filled(true),
            );
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
            for plane in 0..constellation.num_planes {
                let orbit_pts = constellation.orbit_points_3d(plane);
                let color = plane_color(plane);

                let mut front_segment: Vec<[f64; 2]> = Vec::new();
                for &(x, y, z) in &orbit_pts {
                    let (rx, ry, rz) = rotate_point_matrix(x, y, z, &rotation);
                    if rz >= 0.0 {
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

        let link_color = egui::Color32::from_rgb(200, 200, 200);
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
                if rz1 >= 0.0 && rz2 >= 0.0 {
                    plot_ui.line(
                        Line::new(PlotPoints::new(vec![[rx1, ry1], [rx2, ry2]]))
                            .color(link_color)
                            .width(1.0),
                    );
                }
            }
        }

        for plane in 0..constellation.num_planes {
            let pts: PlotPoints = positions
                .iter()
                .filter_map(|s| {
                    if s.plane != plane {
                        return None;
                    }
                    let (rx, ry, rz) = rotate_point_matrix(s.x, s.y, s.z, &rotation);
                    if rz >= 0.0 {
                        Some([rx, ry])
                    } else {
                        None
                    }
                })
                .collect();
            plot_ui.points(
                Points::new(pts)
                    .color(plane_color(plane))
                    .radius(sat_radius)
                    .filled(true),
            );
        }
    });

    if response.response.dragged() && !response.response.drag_started() {
        let drag = response.response.drag_delta();
        let delta_rot = rotation_from_drag(drag.x as f64 * 0.01, drag.y as f64 * 0.01);
        rotation = delta_rot * rotation;
    }

    rotation
}

fn draw_ground_track(ui: &mut egui::Ui, id: &str, positions: &[SatelliteState], num_planes: usize, width: f32, height: f32) {
    let total_sats = positions.len() as f32;
    let sat_radius = ((width.min(height) / 100.0) / (total_sats / 100.0).sqrt()).clamp(2.0, 10.0);
    let plot = Plot::new(id)
        .width(width)
        .height(height)
        .include_x(-180.0)
        .include_x(180.0)
        .include_y(-90.0)
        .include_y(90.0)
        .show_axes([true, true]);

    plot.show(ui, |plot_ui| {
        for plane in 0..num_planes {
            let pts: PlotPoints = positions
                .iter()
                .filter(|s| s.plane == plane)
                .map(|s| [s.lon, s.lat])
                .collect();
            plot_ui.points(
                Points::new(pts)
                    .color(plane_color(plane))
                    .radius(sat_radius)
                    .filled(true),
            );
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
    constellation: &WalkerConstellation,
    positions: &[SatelliteState],
    time: f64,
    mut rotation: Matrix3<f64>,
    width: f32,
    height: f32,
) -> Matrix3<f64> {
    let major_radius = 2.0;
    let minor_radius = 0.8;
    let total_sats = constellation.total_sats as f32;
    let sat_radius = ((width.min(height) / 100.0) / (total_sats / 100.0).sqrt()).clamp(2.0, 10.0);
    let sats_per_plane = constellation.total_sats / constellation.num_planes;
    let orbit_radius = EARTH_RADIUS_KM + constellation.altitude_km;
    let period = 2.0 * PI * (orbit_radius.powi(3) / 398600.4418_f64).sqrt();
    let mean_motion = 2.0 * PI / period;

    let torus_pos = |plane: usize, sat_idx: usize| -> (f64, f64, f64) {
        let angle = 2.0 * PI * plane as f64 / constellation.num_planes as f64;
        let sat_spacing = 2.0 * PI * sat_idx as f64 / sats_per_plane as f64;
        let phase = sat_spacing + mean_motion * time;
        let r = major_radius + minor_radius * phase.cos();
        let y = minor_radius * phase.sin();
        let x = r * angle.cos();
        let z = r * angle.sin();
        rotate_point_matrix(x, y, z, &rotation)
    };

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

        let is_star = constellation.walker_type == WalkerType::Star;

        for plane in 0..constellation.num_planes {
            let angle = 2.0 * PI * plane as f64 / constellation.num_planes as f64;
            let orbit: PlotPoints = (0..=50)
                .map(|i| {
                    let phase = 2.0 * PI * i as f64 / 50.0;
                    let r = major_radius + minor_radius * phase.cos();
                    let y = minor_radius * phase.sin();
                    let x = r * angle.cos();
                    let z = r * angle.sin();
                    let (rx, ry, _) = rotate_point_matrix(x, y, z, &rotation);
                    [rx, ry]
                })
                .collect();
            plot_ui.line(
                Line::new(orbit)
                    .color(plane_color(plane))
                    .width(1.0),
            );
        }

        let link_color = egui::Color32::from_rgb(150, 150, 150);
        for sat in positions {
            if is_star && sat.plane == constellation.num_planes - 1 {
                continue;
            }
            let next_plane = (sat.plane + 1) % constellation.num_planes;
            if let Some(neighbor) = positions.iter().find(|s| {
                s.plane == next_plane && s.sat_index == sat.sat_index && s.ascending == sat.ascending
            }) {
                let (x1, y1, _) = torus_pos(sat.plane, sat.sat_index);
                let (x2, y2, _) = torus_pos(neighbor.plane, neighbor.sat_index);
                plot_ui.line(
                    Line::new(PlotPoints::new(vec![[x1, y1], [x2, y2]]))
                        .color(link_color)
                        .width(1.0),
                );
            }
        }

        for plane in 0..constellation.num_planes {
            let pts: PlotPoints = positions
                .iter()
                .filter(|s| s.plane == plane)
                .map(|s| {
                    let (x, y, _) = torus_pos(s.plane, s.sat_index);
                    [x, y]
                })
                .collect();
            plot_ui.points(
                Points::new(pts)
                    .color(plane_color(plane))
                    .radius(sat_radius)
                    .filled(true),
            );
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
    let colors = [
        egui::Color32::from_rgb(255, 99, 71),
        egui::Color32::from_rgb(50, 205, 50),
        egui::Color32::from_rgb(30, 144, 255),
        egui::Color32::from_rgb(255, 215, 0),
        egui::Color32::from_rgb(238, 130, 238),
        egui::Color32::from_rgb(0, 206, 209),
        egui::Color32::from_rgb(255, 140, 0),
        egui::Color32::from_rgb(147, 112, 219),
    ];
    colors[plane % colors.len()]
}

fn dim_color(color: egui::Color32) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(
        (color.r() as f32 * 0.3) as u8,
        (color.g() as f32 * 0.3) as u8,
        (color.b() as f32 * 0.3) as u8,
        150,
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 900.0]),
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
