use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Points, Polygon};
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

                let inc = self.inclination_deg.to_radians();

                let x_orbital = orbit_radius * true_anomaly.cos();
                let y_orbital = orbit_radius * true_anomaly.sin();

                let x = x_orbital * raan.cos() - y_orbital * inc.cos() * raan.sin();
                let y = x_orbital * raan.sin() + y_orbital * inc.cos() * raan.cos();
                let z = y_orbital * inc.sin();

                let lat = (z / orbit_radius).asin().to_degrees();
                let lon = y.atan2(x).to_degrees();

                positions.push(SatelliteState {
                    plane,
                    x,
                    y,
                    z,
                    lat,
                    lon,
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
                let y = x_orbital * raan.sin() + y_orbital * inc.cos() * raan.cos();
                let z = y_orbital * inc.sin();

                (x, y, z)
            })
            .collect()
    }

    fn name(&self) -> &'static str {
        match self.walker_type {
            WalkerType::Delta => "Walker Delta",
            WalkerType::Star => "Walker Star",
        }
    }
}

struct SatelliteState {
    plane: usize,
    x: f64,
    y: f64,
    z: f64,
    lat: f64,
    lon: f64,
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
            ui.label("Shared Configuration");
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

            ui.add_space(20.0);
            ui.label(format!("Total: {} satellites", self.total_sats()));

            ui.add_space(10.0);
            ui.separator();
            ui.label("Delta: RAAN spread 360°");
            ui.label("Star: RAAN spread 180°");
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let show_orbits = self.show_orbits;

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.heading("Walker Delta");
                    draw_3d_view(ui, "delta_3d", &delta, &delta_positions, show_orbits);
                    ui.add_space(5.0);
                    draw_ground_track(ui, "delta_ground", &delta_positions, delta.num_planes);
                });

                ui.add_space(20.0);

                ui.vertical(|ui| {
                    ui.heading("Walker Star");
                    draw_3d_view(ui, "star_3d", &star, &star_positions, show_orbits);
                    ui.add_space(5.0);
                    draw_ground_track(ui, "star_ground", &star_positions, star.num_planes);
                });
            });
        });
    }
}

fn draw_3d_view(
    ui: &mut egui::Ui,
    id: &str,
    constellation: &WalkerConstellation,
    positions: &[SatelliteState],
    show_orbits: bool,
) {
    let plot = Plot::new(id)
        .data_aspect(1.0)
        .width(350.0)
        .height(300.0)
        .show_axes(false)
        .show_grid(false);

    plot.show(ui, |plot_ui| {
        if show_orbits {
            for plane in 0..constellation.num_planes {
                let orbit_pts = constellation.orbit_points_3d(plane);
                let color = plane_color(plane);

                let mut behind_segment: Vec<[f64; 2]> = Vec::new();
                for &(x, y, z) in &orbit_pts {
                    if z < 0.0 {
                        behind_segment.push([x, y]);
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
                .filter(|s| s.plane == plane && s.z < 0.0)
                .map(|s| [s.x, s.y])
                .collect();
            plot_ui.points(
                Points::new(pts)
                    .color(dim_color(plane_color(plane)))
                    .radius(4.0)
                    .filled(true),
            );
        }

        let earth_fill: PlotPoints = (0..=100)
            .map(|i| {
                let theta = 2.0 * PI * i as f64 / 100.0;
                [
                    EARTH_RADIUS_KM * theta.cos(),
                    EARTH_RADIUS_KM * theta.sin(),
                ]
            })
            .collect();
        plot_ui.polygon(
            Polygon::new(earth_fill)
                .fill_color(egui::Color32::from_rgb(30, 60, 120))
                .stroke(egui::Stroke::new(2.0, egui::Color32::from_rgb(70, 130, 180))),
        );

        if show_orbits {
            for plane in 0..constellation.num_planes {
                let orbit_pts = constellation.orbit_points_3d(plane);
                let color = plane_color(plane);

                let mut front_segment: Vec<[f64; 2]> = Vec::new();
                for &(x, y, z) in &orbit_pts {
                    if z >= 0.0 {
                        front_segment.push([x, y]);
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

        for plane in 0..constellation.num_planes {
            let pts: PlotPoints = positions
                .iter()
                .filter(|s| s.plane == plane && s.z >= 0.0)
                .map(|s| [s.x, s.y])
                .collect();
            plot_ui.points(
                Points::new(pts)
                    .color(plane_color(plane))
                    .radius(5.0)
                    .filled(true),
            );
        }
    });
}

fn draw_ground_track(ui: &mut egui::Ui, id: &str, positions: &[SatelliteState], num_planes: usize) {
    let plot = Plot::new(id)
        .width(350.0)
        .height(150.0)
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
                    .radius(4.0)
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
        viewport: egui::ViewportBuilder::default().with_inner_size([1000.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Walker Constellations",
        options,
        Box::new(|_cc| Ok(Box::new(App::default()))),
    )
}
