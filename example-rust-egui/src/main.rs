use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, PlotPoint, Points, Polygon, Text};
use std::f64::consts::PI;

const EARTH_RADIUS_KM: f64 = 6371.0;

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

fn rotate_point(x: f64, y: f64, z: f64, rot_x: f64, rot_y: f64) -> (f64, f64, f64) {
    let (sin_x, cos_x) = rot_x.sin_cos();
    let (sin_y, cos_y) = rot_y.sin_cos();

    let y1 = y * cos_x - z * sin_x;
    let z1 = y * sin_x + z * cos_x;

    let x2 = x * cos_y + z1 * sin_y;
    let z2 = -x * sin_y + z1 * cos_y;

    (x2, y1, z2)
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
    rot_x: f64,
    rot_y: f64,
    torus_rot_x: f64,
    torus_rot_y: f64,
}

impl Default for App {
    fn default() -> Self {
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
            rot_x: 0.0,
            rot_y: 0.0,
            torus_rot_x: PI / 2.0,
            torus_rot_y: 0.0,
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

            ui.horizontal(|ui| {
                ui.label("Speed:");
                ui.add(egui::Slider::new(&mut self.speed, 0.1..=10.0).logarithmic(true));
            });

            if ui.button("Reset time").clicked() {
                self.time = 0.0;
            }

            if ui.button("Reset view").clicked() {
                self.rot_x = 0.0;
                self.rot_y = 0.0;
                self.torus_rot_x = PI / 2.0;
                self.torus_rot_y = 0.0;
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

            let available_width = ui.available_width();
            let available_height = ui.available_height();
            let plot_width = (available_width - 30.0) / 2.0;
            let plot_3d_height = (available_height - 80.0) * 0.4;
            let plot_ground_height = (available_height - 80.0) * 0.25;
            let plot_torus_height = (available_height - 80.0) * 0.35;

            let mut rot_x = self.rot_x;
            let mut rot_y = self.rot_y;
            let mut torus_rot_x = self.torus_rot_x;
            let mut torus_rot_y = self.torus_rot_y;

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.heading("Walker Delta");
                    let (new_rot_x, new_rot_y) = draw_3d_view(
                        ui,
                        "delta_3d",
                        &delta,
                        &delta_positions,
                        show_orbits,
                        rot_x,
                        rot_y,
                        plot_width,
                        plot_3d_height,
                    );
                    rot_x = new_rot_x;
                    rot_y = new_rot_y;
                    ui.add_space(5.0);
                    draw_ground_track(ui, "delta_ground", &delta_positions, delta.num_planes, plot_width, plot_ground_height);
                    ui.add_space(5.0);
                    let (new_torus_rot_x, new_torus_rot_y) = draw_torus(ui, "delta_torus", &delta, &delta_positions, self.time, torus_rot_x, torus_rot_y, plot_width, plot_torus_height);
                    torus_rot_x = new_torus_rot_x;
                    torus_rot_y = new_torus_rot_y;
                });

                ui.add_space(20.0);

                ui.vertical(|ui| {
                    ui.heading("Walker Star");
                    let (new_rot_x, new_rot_y) = draw_3d_view(
                        ui,
                        "star_3d",
                        &star,
                        &star_positions,
                        show_orbits,
                        rot_x,
                        rot_y,
                        plot_width,
                        plot_3d_height,
                    );
                    rot_x = new_rot_x;
                    rot_y = new_rot_y;
                    ui.add_space(5.0);
                    draw_ground_track(ui, "star_ground", &star_positions, star.num_planes, plot_width, plot_ground_height);
                    ui.add_space(5.0);
                    let (new_torus_rot_x, new_torus_rot_y) = draw_torus(ui, "star_torus", &star, &star_positions, self.time, torus_rot_x, torus_rot_y, plot_width, plot_torus_height);
                    torus_rot_x = new_torus_rot_x;
                    torus_rot_y = new_torus_rot_y;
                });
            });

            self.rot_x = rot_x;
            self.rot_y = rot_y;
            self.torus_rot_x = torus_rot_x;
            self.torus_rot_y = torus_rot_y;
        });
    }
}

fn draw_3d_view(
    ui: &mut egui::Ui,
    id: &str,
    constellation: &WalkerConstellation,
    positions: &[SatelliteState],
    show_orbits: bool,
    mut rot_x: f64,
    mut rot_y: f64,
    width: f32,
    height: f32,
) -> (f64, f64) {
    let orbit_radius = EARTH_RADIUS_KM + constellation.altitude_km;
    let axis_len = EARTH_RADIUS_KM * 1.5;
    let margin = orbit_radius.max(axis_len) * 1.25;
    let sat_radius = (width.min(height) / 80.0).max(2.0);

    let plot = Plot::new(id)
        .data_aspect(1.0)
        .width(width)
        .height(height)
        .show_axes(false)
        .show_grid(false)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .include_x(-margin)
        .include_x(margin)
        .include_y(-margin)
        .include_y(margin);

    let response = plot.show(ui, |plot_ui| {
        if show_orbits {
            for plane in 0..constellation.num_planes {
                let orbit_pts = constellation.orbit_points_3d(plane);
                let color = plane_color(plane);

                let mut behind_segment: Vec<[f64; 2]> = Vec::new();
                for &(x, y, z) in &orbit_pts {
                    let (rx, ry, rz) = rotate_point(x, y, z, rot_x, rot_y);
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
                    let (rx, ry, rz) = rotate_point(s.x, s.y, s.z, rot_x, rot_y);
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

        let (x1, y1, _) = rotate_point(axis_len, 0.0, 0.0, rot_x, rot_y);
        let (x2, y2, _) = rotate_point(-axis_len, 0.0, 0.0, rot_x, rot_y);
        plot_ui.line(
            Line::new(PlotPoints::new(vec![[x2, y2], [x1, y1]]))
                .color(egui::Color32::from_rgb(255, 100, 100))
                .width(1.5),
        );

        let (yp_x, yp_y, _) = rotate_point(0.0, axis_len, 0.0, rot_x, rot_y);
        let (yn_x, yn_y, _) = rotate_point(0.0, -axis_len, 0.0, rot_x, rot_y);
        plot_ui.line(
            Line::new(PlotPoints::new(vec![[yn_x, yn_y], [yp_x, yp_y]]))
                .color(egui::Color32::from_rgb(100, 100, 255))
                .width(1.5),
        );

        let label_offset = axis_len * 1.15;
        let (n_x, n_y, _) = rotate_point(0.0, label_offset, 0.0, rot_x, rot_y);
        let (s_x, s_y, _) = rotate_point(0.0, -label_offset, 0.0, rot_x, rot_y);
        let (e_x, e_y, _) = rotate_point(label_offset, 0.0, 0.0, rot_x, rot_y);
        let (w_x, w_y, _) = rotate_point(-label_offset, 0.0, 0.0, rot_x, rot_y);

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
                    let (rx, ry, rz) = rotate_point(x, y, z, rot_x, rot_y);
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
                let (rx1, ry1, rz1) = rotate_point(sat.x, sat.y, sat.z, rot_x, rot_y);
                let (rx2, ry2, rz2) = rotate_point(neighbor.x, neighbor.y, neighbor.z, rot_x, rot_y);
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
                    let (rx, ry, rz) = rotate_point(s.x, s.y, s.z, rot_x, rot_y);
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

    if response.response.dragged() {
        let drag = response.response.drag_delta();
        rot_y += drag.x as f64 * 0.01;
        rot_x += drag.y as f64 * 0.01;
    }

    (rot_x, rot_y)
}

fn draw_ground_track(ui: &mut egui::Ui, id: &str, positions: &[SatelliteState], num_planes: usize, width: f32, height: f32) {
    let sat_radius = (width.min(height) / 80.0).max(2.0);
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
    mut rot_x: f64,
    mut rot_y: f64,
    width: f32,
    height: f32,
) -> (f64, f64) {
    let major_radius = 2.0;
    let minor_radius = 0.8;
    let sat_radius = (width.min(height) / 80.0).max(2.0);
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
        rotate_point(x, y, z, rot_x, rot_y)
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
        .allow_boxed_zoom(false)
        .include_x(-margin)
        .include_x(margin)
        .include_y(-margin)
        .include_y(margin);

    let response = plot.show(ui, |plot_ui| {
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
                    let (rx, ry, _) = rotate_point(x, y, z, rot_x, rot_y);
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

    if response.response.dragged() {
        let drag = response.response.drag_delta();
        rot_y += drag.x as f64 * 0.01;
        rot_x += drag.y as f64 * 0.01;
    }

    (rot_x, rot_y)
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

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 900.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Walker Constellations",
        options,
        Box::new(|_cc| Ok(Box::new(App::default()))),
    )
}
