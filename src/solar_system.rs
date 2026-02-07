use crate::celestial::CelestialBody;
use egui_plot::{Line, PlotImage, PlotPoint, Text};
use std::collections::HashMap;
use std::f64::consts::PI;

pub const SCALE_OFFSET: f64 = 1.0;

struct MoonOrbit {
    parent: CelestialBody,
    distance_au: f64,
    period_days: f64,
}

fn moon_orbit(body: CelestialBody) -> Option<MoonOrbit> {
    match body {
        CelestialBody::Moon => Some(MoonOrbit {
            parent: CelestialBody::Earth, distance_au: 0.00257, period_days: 27.32,
        }),
        CelestialBody::Ganymede => Some(MoonOrbit {
            parent: CelestialBody::Jupiter, distance_au: 0.00716, period_days: 7.155,
        }),
        CelestialBody::Callisto => Some(MoonOrbit {
            parent: CelestialBody::Jupiter, distance_au: 0.01258, period_days: 16.689,
        }),
        CelestialBody::Io => Some(MoonOrbit {
            parent: CelestialBody::Jupiter, distance_au: 0.00282, period_days: 1.769,
        }),
        CelestialBody::Europa => Some(MoonOrbit {
            parent: CelestialBody::Jupiter, distance_au: 0.00449, period_days: 3.551,
        }),
        CelestialBody::Titan => Some(MoonOrbit {
            parent: CelestialBody::Saturn, distance_au: 0.00817, period_days: 15.945,
        }),
        CelestialBody::Triton => Some(MoonOrbit {
            parent: CelestialBody::Neptune, distance_au: 0.00237, period_days: -5.877,
        }),
        CelestialBody::Charon => Some(MoonOrbit {
            parent: CelestialBody::Pluto, distance_au: 0.000131, period_days: 6.387,
        }),
        CelestialBody::Enceladus => Some(MoonOrbit {
            parent: CelestialBody::Saturn, distance_au: 0.00159, period_days: 1.370,
        }),
        CelestialBody::Mimas => Some(MoonOrbit {
            parent: CelestialBody::Saturn, distance_au: 0.00124, period_days: 0.942,
        }),
        CelestialBody::Iapetus => Some(MoonOrbit {
            parent: CelestialBody::Saturn, distance_au: 0.0238, period_days: 79.32,
        }),
        CelestialBody::Phobos => Some(MoonOrbit {
            parent: CelestialBody::Mars, distance_au: 0.0000628, period_days: 0.319,
        }),
        _ => None,
    }
}

pub struct Asteroid {
    pub name: String,
    pub a: f64,
    pub e: f64,
    pub om_rad: f64,
    pub w_rad: f64,
    pub ma_rad: f64,
    pub n_rad_per_day: f64,
    pub epoch_jd: f64,
    pub i_rad: f64,
}

pub enum AsteroidLoadState {
    NotLoaded,
    Loading,
    Loaded(Vec<Asteroid>),
    #[allow(dead_code)]
    Failed(String),
}

const J2000_JD: f64 = 2451545.0;

#[cfg(not(target_arch = "wasm32"))]
pub fn fetch_asteroids() -> Result<Vec<Asteroid>, String> {
    let url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api?\
        fields=full_name,a,e,i,om,w,ma,n,epoch&\
        sb-cdata=%7B%22AND%22%3A%5B%22a%7CRG%7C2.1%7C3.3%22%5D%7D&\
        sb-kind=a&limit=2000&full-prec=true";
    let body = ureq::get(url)
        .call()
        .map_err(|e| format!("HTTP error: {e}"))?
        .into_string()
        .map_err(|e| format!("Read error: {e}"))?;
    let json: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("JSON error: {e}"))?;
    let data = json["data"]
        .as_array()
        .ok_or("Missing data field")?;
    let mut asteroids = Vec::with_capacity(data.len());
    for row in data {
        let arr = row.as_array().ok_or("Row not array")?;
        let name = arr.get(0)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let f = |idx: usize| -> Result<f64, String> {
            arr.get(idx)
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok())
                .ok_or_else(|| format!("Bad field {idx}"))
        };
        let deg = PI / 180.0;
        asteroids.push(Asteroid {
            name,
            a: f(1)?,
            e: f(2)?,
            i_rad: f(3)? * deg,
            om_rad: f(4)? * deg,
            w_rad: f(5)? * deg,
            ma_rad: f(6)? * deg,
            n_rad_per_day: f(7)? * deg,
            epoch_jd: f(8)?,
        });
    }
    Ok(asteroids)
}

fn solve_kepler(ma: f64, e: f64) -> f64 {
    let mut ea = ma;
    for _ in 0..8 {
        ea = ea - (ea - e * ea.sin() - ma) / (1.0 - e * ea.cos());
    }
    ea
}

pub fn asteroid_position(ast: &Asteroid, j2000_days: f64) -> [f64; 2] {
    let jd = j2000_days + J2000_JD;
    let dt = jd - ast.epoch_jd;
    let ma = (ast.ma_rad + ast.n_rad_per_day * dt) % (2.0 * PI);
    let ea = solve_kepler(ma, ast.e);
    let cos_ea = ea.cos();
    let sin_ea = ea.sin();
    let x_orb = ast.a * (cos_ea - ast.e);
    let y_orb = ast.a * (1.0 - ast.e * ast.e).sqrt() * sin_ea;
    let cos_w = ast.w_rad.cos();
    let sin_w = ast.w_rad.sin();
    let cos_om = ast.om_rad.cos();
    let sin_om = ast.om_rad.sin();
    let cos_i = ast.i_rad.cos();
    let x = (cos_om * cos_w - sin_om * sin_w * cos_i) * x_orb
        + (-cos_om * sin_w - sin_om * cos_w * cos_i) * y_orb;
    let y = (sin_om * cos_w + cos_om * sin_w * cos_i) * x_orb
        + (-sin_om * sin_w + cos_om * cos_w * cos_i) * y_orb;
    [x, y]
}

static J2000_EPOCH: std::sync::LazyLock<chrono::DateTime<chrono::Utc>> =
    std::sync::LazyLock::new(|| {
        chrono::DateTime::parse_from_rfc3339("2000-01-01T12:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc)
    });

fn scale_position(x: f64, y: f64, power: f64) -> [f64; 2] {
    let r = (x * x + y * y).sqrt();
    if r < 1e-10 {
        return [0.0, 0.0];
    }
    let r_scaled = (r + SCALE_OFFSET).powf(power) - SCALE_OFFSET.powf(power);
    let s = r_scaled / r;
    [x * s, y * s]
}

pub fn compute_body_position_au(
    body: CelestialBody,
    j2000_days: f64,
) -> Option<[f64; 2]> {
    if body == CelestialBody::Sun {
        return Some([0.0, 0.0]);
    }

    if let Some(orbit) = moon_orbit(body) {
        let parent_pos = compute_body_position_au(orbit.parent, j2000_days)?;
        let angle = 2.0 * PI * j2000_days / orbit.period_days;
        let mx = orbit.distance_au * angle.cos();
        let my = orbit.distance_au * angle.sin();
        return Some([parent_pos[0] + mx, parent_pos[1] + my]);
    }

    let sma = body.semi_major_axis_au()?;
    let period = body.orbital_period_days()?;

    let angle = body.mean_longitude_j2000_deg().to_radians() + 2.0 * PI * j2000_days / period;

    let x = sma * angle.cos();
    let y = sma * angle.sin();

    Some([x, y])
}

fn circle_points(cx: f64, cy: f64, r: f64, n: usize) -> Vec<[f64; 2]> {
    (0..=n)
        .map(|i| {
            let a = 2.0 * PI * i as f64 / n as f64;
            [cx + r * a.cos(), cy + r * a.sin()]
        })
        .collect()
}

pub fn draw_solar_system_view(
    plot_ui: &mut egui_plot::PlotUi,
    focused_body: CelestialBody,
    timestamp: chrono::DateTime<chrono::Utc>,
    sphere_handles: &HashMap<CelestialBody, eframe::egui::TextureHandle>,
    dark_mode: bool,
    log_power: f64,
    asteroids: &[Asteroid],
    _asteroid_sprite: Option<&eframe::egui::TextureHandle>,
) -> Option<CelestialBody> {
    let j2000_days = (timestamp - *J2000_EPOCH).num_seconds() as f64 / 86400.0;

    let label_color = if dark_mode {
        eframe::egui::Color32::WHITE
    } else {
        eframe::egui::Color32::BLACK
    };

    let bounds = plot_ui.plot_bounds();
    let view_size = (bounds.max()[0] - bounds.min()[0])
        .max(bounds.max()[1] - bounds.min()[1]);
    let inflation = (1.0 - log_power).max(0.0);
    let min_radius = view_size * 0.004 * inflation;

    let mercury_scaled = (0.387 + SCALE_OFFSET).powf(log_power) - SCALE_OFFSET.powf(log_power);
    let sun_visual_radius = 0.25 * mercury_scaled;
    let sun_km = CelestialBody::Sun.radius_km();

    let mut bodies: Vec<(CelestialBody, f64, f64, f64)> = Vec::new();

    for &body in &CelestialBody::ALL {
        let pos = match compute_body_position_au(body, j2000_days) {
            Some(p) => p,
            None => continue,
        };

        let scaled = scale_position(pos[0], pos[1], log_power);

        let visual_radius = (sun_visual_radius * (body.radius_km() / sun_km).powf(log_power)).max(min_radius);

        bodies.push((body, scaled[0], scaled[1], visual_radius));
    }

    for &body in &CelestialBody::ALL {
        if body == CelestialBody::Sun || body == CelestialBody::Moon {
            continue;
        }

        let sma = match body.semi_major_axis_au() {
            Some(s) => s,
            None => continue,
        };
        let period = match body.orbital_period_days() {
            Some(p) => p,
            None => continue,
        };
        let mean_lon = body.mean_longitude_j2000_deg().to_radians();

        let num_points = 200;
        let mut points: Vec<[f64; 2]> = Vec::with_capacity(num_points + 1);
        for i in 0..=num_points {
            let t = i as f64 / num_points as f64;
            let angle = mean_lon + 2.0 * PI * (j2000_days / period + t);
            let ox = sma * angle.cos();
            let oy = sma * angle.sin();
            points.push(scale_position(ox, oy, log_power));
        }

        let orbit_color = if body == focused_body {
            body.display_color()
        } else {
            body.display_color().gamma_multiply(0.4)
        };
        let orbit_width = if body == focused_body { 2.0 } else { 1.0 };

        plot_ui.line(
            Line::new("", points)
                .color(orbit_color)
                .width(orbit_width),
        );
    }

    for &moon_body in &CelestialBody::ALL {
        let orbit = match moon_orbit(moon_body) {
            Some(o) => o,
            None => continue,
        };
        let parent_pos = match compute_body_position_au(orbit.parent, j2000_days) {
            Some(p) => p,
            None => continue,
        };
        let parent_scaled = scale_position(parent_pos[0], parent_pos[1], log_power);
        let edge_pos = [parent_pos[0] + orbit.distance_au, parent_pos[1]];
        let edge_scaled = scale_position(edge_pos[0], edge_pos[1], log_power);
        let orbit_r = ((edge_scaled[0] - parent_scaled[0]).powi(2)
            + (edge_scaled[1] - parent_scaled[1]).powi(2))
            .sqrt();
        if orbit_r > view_size * 0.005 {
            let orbit_pts = circle_points(parent_scaled[0], parent_scaled[1], orbit_r, 200);
            let orbit_color = if focused_body == moon_body {
                moon_body.display_color()
            } else {
                moon_body.display_color().gamma_multiply(0.4)
            };
            plot_ui.line(
                Line::new("", orbit_pts)
                    .color(orbit_color)
                    .width(1.0),
            );
        }
    }

    let mut ast_positions: Vec<(usize, f64, f64)> = Vec::new();
    if !asteroids.is_empty() {
        let asteroid_color = if dark_mode {
            eframe::egui::Color32::from_rgba_unmultiplied(180, 160, 140, 120)
        } else {
            eframe::egui::Color32::from_rgba_unmultiplied(120, 100, 80, 140)
        };
        let mut pts: Vec<[f64; 2]> = Vec::with_capacity(asteroids.len());
        for (idx, ast) in asteroids.iter().enumerate() {
            let pos = asteroid_position(ast, j2000_days);
            let scaled = scale_position(pos[0], pos[1], log_power);
            ast_positions.push((idx, scaled[0], scaled[1]));
            pts.push([scaled[0], scaled[1]]);
        }
        plot_ui.points(
            egui_plot::Points::new("", pts)
                .color(asteroid_color)
                .radius(1.5),
        );
    }

    let base_label_size = (90.0 / view_size.max(0.01)).clamp(8.0, 16.0) as f32;

    for &(body, x, y, visual_radius) in &bodies {
        if let Some(handle) = sphere_handles.get(&body) {
            let ring_scale = body.ring_params().map(|(_, _, o)| o as f64).unwrap_or(1.0).max(1.0);
            let img_size = (visual_radius * 2.0 * ring_scale) as f32;
            plot_ui.image(PlotImage::new(
                "",
                handle.id(),
                PlotPoint::new(x, y),
                [img_size, img_size],
            ));
        }

        if body.parent_body().is_none() || body == focused_body {
            let name_color = if body == focused_body {
                body.display_color()
            } else {
                label_color
            };

            let dist_from_center = (x * x + y * y).sqrt();
            let edge_frac = (dist_from_center / (view_size * 0.45)).clamp(0.0, 1.0) as f32;
            let label_font_size = base_label_size + edge_frac * 4.0;

            let label_text = if body != CelestialBody::Sun {
                let offset_p = SCALE_OFFSET.powf(log_power);
                let au = if dist_from_center > 1e-6 { (dist_from_center + offset_p).powf(1.0 / log_power) - SCALE_OFFSET } else { 0.0 };
                let km = au * 149_597_870.7;
                if km >= 1_000_000.0 {
                    format!("{} ({:.1}M km)", body.label(), km / 1_000_000.0)
                } else {
                    format!("{} ({:.0} km)", body.label(), km)
                }
            } else {
                body.label().to_string()
            };

            plot_ui.text(
                Text::new(
                    "",
                    PlotPoint::new(x, y + visual_radius + view_size * 0.015),
                    eframe::egui::RichText::new(label_text).size(label_font_size),
                )
                .color(name_color),
            );
        }
    }

    if plot_ui.response().hovered() {
        if let Some(pointer) = plot_ui.pointer_coordinate() {
            let line_color = if dark_mode {
                eframe::egui::Color32::from_rgba_unmultiplied(255, 255, 255, 40)
            } else {
                eframe::egui::Color32::from_rgba_unmultiplied(0, 0, 0, 40)
            };
            plot_ui.line(
                Line::new("", vec![[0.0, 0.0], [pointer.x, pointer.y]])
                    .color(line_color)
                    .width(1.0),
            );

            let sr = (pointer.x.powi(2) + pointer.y.powi(2)).sqrt();
            let offset_p = SCALE_OFFSET.powf(log_power);
            let real_au = if sr > 1e-6 { (sr + offset_p).powf(1.0 / log_power) - SCALE_OFFSET } else { 0.0 };
            let screen_pos = plot_ui.screen_from_plot(PlotPoint::new(pointer.x, pointer.y));
            let offset_screen = eframe::egui::Pos2::new(screen_pos.x + 12.0, screen_pos.y - 12.0);
            let offset_plot = plot_ui.plot_from_screen(offset_screen);
            plot_ui.text(
                Text::new(
                    "",
                    offset_plot,
                    eframe::egui::RichText::new(format!("{:.2} AU", real_au))
                        .size(12.0),
                )
                .color(label_color)
                .anchor(eframe::egui::Align2::LEFT_BOTTOM),
            );

            let mut best: Option<(CelestialBody, f64, f64, f64, f64)> = None;
            for &(body, x, y, visual_radius) in &bodies {
                let dx = pointer.x - x;
                let dy = pointer.y - y;
                let dist = (dx * dx + dy * dy).sqrt();
                let hit = visual_radius.max(view_size * 0.015) * 2.0;
                if dist <= hit {
                    let dominated = best.map_or(false, |(_, _, _, _, bd)| bd < dist);
                    if !dominated {
                        best = Some((body, x, y, visual_radius, dist));
                    }
                }
            }
            if let Some((body, x, y, visual_radius, _)) = best {
                let ring_r = visual_radius * 1.15;
                let ring_pts = circle_points(x, y, ring_r, 64);
                plot_ui.line(
                    Line::new("", ring_pts)
                        .color(body.display_color())
                        .width(2.0),
                );

                eframe::egui::Tooltip::always_open(
                    plot_ui.ctx().clone(),
                    eframe::egui::LayerId::background(),
                    eframe::egui::Id::new("ss_tooltip"),
                    eframe::egui::PopupAnchor::Pointer,
                )
                .gap(12.0)
                .show(|ui| {
                    ui.set_min_width(200.0);
                    ui.label(eframe::egui::RichText::new(body.label()).strong().size(16.0));
                    ui.separator();
                    eframe::egui::Grid::new("ss_tooltip_grid")
                        .num_columns(2)
                        .spacing([12.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("Radius:");
                            ui.label(format!("{:.0} km", body.radius_km()));
                            ui.end_row();
                            if let Some(sma) = body.semi_major_axis_au() {
                                ui.label("Orbit:");
                                ui.label(format!("{:.3} AU", sma));
                                ui.end_row();
                            }
                            if let Some(period) = body.orbital_period_days() {
                                let years = period / 365.25;
                                ui.label("Period:");
                                if years >= 1.0 {
                                    ui.label(format!("{:.2} years", years));
                                } else {
                                    ui.label(format!("{:.1} days", period));
                                }
                                ui.end_row();
                            }
                            ui.label("Rotation:");
                            let rot = body.rotation_period_hours();
                            if rot.abs() > 100.0 {
                                ui.label(format!("{:.0} hours", rot));
                            } else {
                                ui.label(format!("{:.1} hours", rot));
                            }
                            ui.end_row();
                            ui.label("Mass param:");
                            ui.label(format!("{:.0} km³/s²", body.mu()));
                            ui.end_row();
                        });
                });
            } else {
                let hit_r = sun_visual_radius * 0.15;
                let mut best_ast: Option<(usize, f64)> = None;
                for &(idx, ax, ay) in &ast_positions {
                    let dx = pointer.x - ax;
                    let dy = pointer.y - ay;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist <= hit_r {
                        if best_ast.map_or(true, |(_, bd)| dist < bd) {
                            best_ast = Some((idx, dist));
                        }
                    }
                }
                if let Some((idx, _)) = best_ast {
                    let (_, ax, ay) = ast_positions[ast_positions.iter().position(|&(i, _, _)| i == idx).unwrap()];
                    let ring_r = view_size * 0.015;
                    let ring_pts = circle_points(ax, ay, ring_r, 32);
                    plot_ui.line(
                        Line::new("", ring_pts)
                            .color(eframe::egui::Color32::from_rgb(200, 170, 120))
                            .width(2.0),
                    );
                    let ast = &asteroids[idx];
                    eframe::egui::Tooltip::always_open(
                        plot_ui.ctx().clone(),
                        eframe::egui::LayerId::background(),
                        eframe::egui::Id::new("ss_tooltip"),
                        eframe::egui::PopupAnchor::Pointer,
                    )
                    .gap(12.0)
                    .show(|ui| {
                        ui.label(eframe::egui::RichText::new(&ast.name).strong().size(14.0));
                        ui.separator();
                        eframe::egui::Grid::new("ast_tooltip_grid")
                            .num_columns(2)
                            .spacing([12.0, 4.0])
                            .show(ui, |ui| {
                                let km = ast.a * 149_597_870.7;
                                ui.label("Orbit:");
                                if km >= 1_000_000.0 {
                                    ui.label(format!("{:.3} AU\n({:.1}M km)", ast.a, km / 1_000_000.0));
                                } else {
                                    ui.label(format!("{:.3} AU\n({:.0} km)", ast.a, km));
                                }
                                ui.end_row();
                                ui.label("Eccentricity:");
                                ui.label(format!("{:.4}", ast.e));
                                ui.end_row();
                                ui.label("Inclination:");
                                ui.label(format!("{:.2}\u{00b0}", ast.i_rad.to_degrees()));
                                ui.end_row();
                                let period = 2.0 * PI / ast.n_rad_per_day;
                                let years = period / 365.25;
                                ui.label("Period:");
                                ui.label(format!("{:.2} years", years));
                                ui.end_row();
                            });
                    });
                }
            }
        }
    }

    if plot_ui.response().clicked() {
        if let Some(pointer) = plot_ui.pointer_coordinate() {
            let mut closest: Option<(CelestialBody, f64)> = None;
            for &(body, x, y, visual_radius) in &bodies {
                let dx = pointer.x - x;
                let dy = pointer.y - y;
                let dist = (dx * dx + dy * dy).sqrt();
                let hit = visual_radius.max(view_size * 0.015) * 2.0;
                if dist <= hit {
                    if closest.is_none() || dist < closest.unwrap().1 {
                        closest = Some((body, dist));
                    }
                }
            }
            if let Some((body, _)) = closest {
                return Some(body);
            }
        }
    }

    None
}

pub struct AutoZoomState {
    pub enabled: bool,
    pub total_duration: f32,
    pub stay_duration: f32,
    pub time: f64,
}

pub fn draw_planet_sizes(
    ui: &mut eframe::egui::Ui,
    sphere_handles: &HashMap<CelestialBody, eframe::egui::TextureHandle>,
    zoom_t: &mut f64,
    auto_zoom: &mut AutoZoomState,
) -> Option<CelestialBody> {
    use eframe::egui;

    let mut sorted: Vec<CelestialBody> = CelestialBody::ALL.to_vec();
    sorted.sort_by(|a, b| b.radius_km().partial_cmp(&a.radius_km()).unwrap());

    let available = ui.available_size();
    let mut clicked_body = None;

    {
        let n = sorted.len();

        let text_color = ui.visuals().text_color();
        let weak_color = ui.visuals().weak_text_color();

        let (response, painter) = ui.allocate_painter(
            egui::Vec2::new(available.x, available.y),
            egui::Sense::click().union(egui::Sense::hover()),
        );
        let rect = response.rect;
        let painter = painter.with_clip_rect(rect);

        if !auto_zoom.enabled && response.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll.abs() > 0.1 {
                *zoom_t = (*zoom_t - scroll as f64 * 0.005)
                    .clamp(0.0, (n - 1) as f64);
                ui.ctx().request_repaint();
            }
        }

        let view_h = rect.height() as f64;
        let view_w = rect.width() as f64;

        let margin = 12.0;
        let compute_layout = |k: usize| -> Vec<(f64, f64)> {
            let r_focus = sorted[k].radius_km();
            let h_scale = view_h / (2.5 * r_focus);
            let num_gaps = (n - k).saturating_sub(1).max(1) as f64;
            let usable_w = view_w - 2.0 * margin;
            let body_ext = |b: &CelestialBody| -> f64 {
                b.ring_params().map(|(_, _, o)| (o as f64).max(1.0)).unwrap_or(1.0)
            };
            let gap = 8.0;
            let total_extent: f64 = sorted[k..].iter()
                .map(|b| 2.0 * b.radius_km() * body_ext(b)).sum();
            let w_scale = (usable_w - num_gaps * gap) / total_extent;
            let scale = h_scale.min(w_scale);

            let mut result = vec![(0.0, 0.0); n];
            let mut sx = margin;
            for j in k..n {
                let r_px = sorted[j].radius_km() * scale;
                let ext = body_ext(&sorted[j]);
                sx += r_px * ext;
                result[j] = (sx, r_px);
                sx += r_px * ext + gap;
            }
            for j in (0..k).rev() {
                let r_px = sorted[j].radius_km() * scale;
                let ext = body_ext(&sorted[j]);
                let (next_x, next_r) = result[j + 1];
                let next_ext = body_ext(&sorted[j + 1]);
                result[j] = (next_x - next_r * next_ext - gap - r_px * ext, r_px);
            }
            if k > 0 {
                let (prev_x, prev_r) = result[k - 1];
                let prev_ext = body_ext(&sorted[k - 1]);
                let right_edge = prev_x + prev_r * prev_ext;
                if right_edge > -margin {
                    let shift = right_edge + margin;
                    for j in 0..k {
                        result[j].0 -= shift;
                    }
                }
            }
            result
        };

        if auto_zoom.enabled {
            let dt = ui.ctx().input(|i| i.stable_dt) as f64;
            auto_zoom.time += dt;

            let mut cum = vec![0.0f64; n];
            for k in 0..n - 1 {
                let la = compute_layout(k);
                let lb = compute_layout(k + 1);
                let dist = (la[k + 1].0 - lb[k + 1].0).abs();
                cum[k + 1] = cum[k] + dist;
            }
            let total = cum[n - 1];

            let scroll = auto_zoom.total_duration as f64;
            let stay = auto_zoom.stay_duration as f64;
            let cycle = 2.0 * (stay + scroll);
            let t = auto_zoom.time % cycle;
            let target = if t < stay {
                0.0
            } else if t < stay + scroll {
                (t - stay) / scroll * total
            } else if t < 2.0 * stay + scroll {
                total
            } else {
                (2.0 - (t - 2.0 * stay) / scroll) * total
            };

            let seg = cum.partition_point(|&d| d <= target)
                .saturating_sub(1).min(n - 2);
            let frac = (target - cum[seg]) / (cum[seg + 1] - cum[seg]);
            *zoom_t = (seg as f64 + frac).clamp(0.0, (n - 1) as f64);
            ui.ctx().request_repaint();
        }

        let i = (*zoom_t as usize).min(n - 2);
        let frac = *zoom_t - i as f64;

        let layout_a = compute_layout(i);
        let layout_b = compute_layout((i + 1).min(n - 1));
        let center_y = rect.center().y;

        let base_name = (view_h as f32 * 0.04).clamp(14.0, 36.0);
        let base_km = (view_h as f32 * 0.03).clamp(10.0, 27.0);

        let mut screen_bodies: Vec<(CelestialBody, f32, f32)> = Vec::new();

        for j in 0..n {
            let (xa, ra) = layout_a[j];
            let (xb, rb) = layout_b[j];
            let cx = (xa * (1.0 - frac) + xb * frac) as f32 + rect.left();
            let r_px = (ra * (1.0 - frac) + rb * frac) as f32;

            let body = sorted[j];
            let ring_scale = body.ring_params().map(|(_, _, o)| (o as f32).max(1.0)).unwrap_or(1.0);
            let vis_extent = r_px * ring_scale;

            if cx + vis_extent < rect.left() - 10.0
                || cx - vis_extent > rect.right() + 10.0
            {
                continue;
            }

            screen_bodies.push((body, cx, r_px));

            if let Some(handle) = sphere_handles.get(&body) {
                let img_rect = egui::Rect::from_center_size(
                    egui::Pos2::new(cx, center_y),
                    egui::Vec2::splat(r_px * 2.0 * ring_scale),
                );
                painter.image(
                    handle.id(),
                    img_rect,
                    egui::Rect::from_min_max(
                        egui::pos2(0.0, 0.0),
                        egui::pos2(1.0, 1.0),
                    ),
                    egui::Color32::WHITE,
                );
            }

            if r_px > 3.0 {
                let decay = ((*zoom_t - j as f64) * 0.3).exp() as f32;
                let name_size = (base_name * decay).max(7.0);
                let km_size = (base_km * decay).max(5.0);
                let sub_size = (base_km * decay * 0.9).max(5.0);
                let ring_scale = body.ring_params().map(|(_, _, o)| (o as f32).max(1.0)).unwrap_or(1.0);
                let vert_extent = r_px * (ring_scale * 0.5).max(1.0);
                let mut label_y = center_y + vert_extent + 8.0;
                painter.text(
                    egui::Pos2::new(cx, label_y),
                    egui::Align2::CENTER_TOP,
                    body.label(),
                    egui::FontId::proportional(name_size),
                    text_color,
                );
                label_y += name_size + 2.0;
                let subtitle = if let Some(parent) = body.parent_body() {
                    Some(format!("Moon of {}", parent.label()))
                } else {
                    match body.category() {
                        "Dwarf Planets" => Some("Dwarf Planet".to_string()),
                        "Asteroids" => Some("Asteroid".to_string()),
                        _ => None,
                    }
                };
                if let Some(sub) = subtitle {
                    painter.text(
                        egui::Pos2::new(cx, label_y),
                        egui::Align2::CENTER_TOP,
                        sub,
                        egui::FontId::proportional(sub_size),
                        weak_color,
                    );
                    label_y += sub_size + 2.0;
                }
                painter.text(
                    egui::Pos2::new(cx, label_y),
                    egui::Align2::CENTER_TOP,
                    format!("{:.0} km", body.radius_km()),
                    egui::FontId::proportional(km_size),
                    weak_color,
                );
            }
        }

        let show_tooltip = |ui: &egui::Ui, body: CelestialBody, anchor: egui::PopupAnchor| {
            egui::Tooltip::always_open(
                ui.ctx().clone(),
                egui::LayerId::background(),
                egui::Id::new("planet_size_tooltip"),
                anchor,
            )
            .gap(12.0)
            .show(|ui| {
                ui.label(egui::RichText::new(body.label()).strong().size(14.0));
                ui.separator();
                egui::Grid::new("ps_tip")
                    .num_columns(2)
                    .spacing([12.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Radius:");
                        ui.label(format!("{:.0} km", body.radius_km()));
                        ui.end_row();
                        if let Some(sma) = body.semi_major_axis_au() {
                            ui.label("Orbit:");
                            let km = sma * 149_597_870.7;
                            if km >= 1_000_000.0 {
                                ui.label(format!("{:.3} AU\n({:.1}M km)", sma, km / 1_000_000.0));
                            } else {
                                ui.label(format!("{:.3} AU\n({:.0} km)", sma, km));
                            }
                            ui.end_row();
                        }
                        if let Some(period) = body.orbital_period_days() {
                            let years = period / 365.25;
                            ui.label("Period:");
                            if years >= 1.0 {
                                ui.label(format!("{:.2} years", years));
                            } else {
                                ui.label(format!("{:.1} days", period));
                            }
                            ui.end_row();
                        }
                        ui.label("Rotation:");
                        let rot = body.rotation_period_hours();
                        if rot.abs() > 100.0 {
                            ui.label(format!("{:.0} hours", rot));
                        } else {
                            ui.label(format!("{:.1} hours", rot));
                        }
                        ui.end_row();
                    });
            });
        };

        let draw_highlight = |painter: &egui::Painter, body: CelestialBody, cx: f32, r_px: f32| {
            let ring_r = r_px * 1.15;
            let n_pts = 64;
            for i in 0..n_pts {
                let a0 = std::f32::consts::TAU * i as f32 / n_pts as f32;
                let a1 = std::f32::consts::TAU * (i + 1) as f32 / n_pts as f32;
                painter.line_segment(
                    [
                        egui::Pos2::new(cx + ring_r * a0.cos(), center_y + ring_r * a0.sin()),
                        egui::Pos2::new(cx + ring_r * a1.cos(), center_y + ring_r * a1.sin()),
                    ],
                    egui::Stroke::new(2.0, body.display_color()),
                );
            }
        };

        if response.hovered() || response.clicked() {
            if let Some(pointer) = ui.ctx().pointer_hover_pos() {
                for &(body, cx, r_px) in &screen_bodies {
                    let dx = pointer.x - cx;
                    let dy = pointer.y - center_y;
                    let dist = (dx * dx + dy * dy).sqrt();
                    let hit = r_px.max(12.0);
                    if dist <= hit {
                        draw_highlight(&painter, body, cx, r_px);
                        show_tooltip(ui, body, egui::PopupAnchor::Pointer);
                        if response.clicked() {
                            clicked_body = Some(body);
                        }
                        break;
                    }
                }
            }
        }
    }
    clicked_body
}
