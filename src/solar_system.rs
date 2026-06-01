use crate::celestial::CelestialBody;
use egui_plot::{Line, PlotImage, PlotPoint, Text};
use std::collections::HashMap;
use std::f64::consts::PI;

pub const SCALE_OFFSET: f64 = 1.0;

const MU_AU3_DAY2: f64 = (2.0 * PI) * (2.0 * PI) / (365.25 * 365.25);
const AU_KM: f64 = 149_597_870.7;

pub const HOHMANN_PLANETS: [CelestialBody; 8] = [
    CelestialBody::Mercury,
    CelestialBody::Venus,
    CelestialBody::Earth,
    CelestialBody::Mars,
    CelestialBody::Jupiter,
    CelestialBody::Saturn,
    CelestialBody::Uranus,
    CelestialBody::Neptune,
];

#[allow(dead_code)]
pub struct HohmannParams {
    pub transfer_sma: f64,
    pub eccentricity: f64,
    pub transfer_time_days: f64,
    pub departure_dv_km_s: f64,
    pub arrival_dv_km_s: f64,
    pub phase_angle_rad: f64,
    pub synodic_period_days: f64,
}

pub fn hohmann_transfer_params(
    origin: CelestialBody,
    dest: CelestialBody,
) -> Option<HohmannParams> {
    let r1 = origin.semi_major_axis_au()?;
    let r2 = dest.semi_major_axis_au()?;
    let t1 = origin.orbital_period_days()?;
    let t2 = dest.orbital_period_days()?;

    let a_t = (r1 + r2) / 2.0;
    let e = (r2 - r1).abs() / (r1 + r2);
    let transfer_time = PI * (a_t.powi(3) / MU_AU3_DAY2).sqrt();

    let mu_km3_s2 = 1.32712440018e11;
    let r1_km = r1 * AU_KM;
    let r2_km = r2 * AU_KM;
    let v_circ1 = (mu_km3_s2 / r1_km).sqrt();
    let v_circ2 = (mu_km3_s2 / r2_km).sqrt();
    let a_t_km = a_t * AU_KM;
    let v_dep = (mu_km3_s2 * (2.0 / r1_km - 1.0 / a_t_km)).sqrt();
    let v_arr = (mu_km3_s2 * (2.0 / r2_km - 1.0 / a_t_km)).sqrt();
    let departure_dv = (v_dep - v_circ1).abs();
    let arrival_dv = (v_circ2 - v_arr).abs();

    let ratio = ((r1 / r2 + 1.0).powi(3) / 8.0).sqrt();
    let phase_angle = PI * (1.0 - ratio);

    let synodic = (1.0 / t1 - 1.0 / t2).abs();
    let synodic_period = if synodic > 1e-12 {
        1.0 / synodic
    } else {
        f64::INFINITY
    };

    Some(HohmannParams {
        transfer_sma: a_t,
        eccentricity: e,
        transfer_time_days: transfer_time,
        departure_dv_km_s: departure_dv,
        arrival_dv_km_s: arrival_dv,
        phase_angle_rad: phase_angle,
        synodic_period_days: synodic_period,
    })
}

pub fn next_launch_window_days(
    origin: CelestialBody,
    dest: CelestialBody,
    j2000_days: f64,
) -> Option<f64> {
    let params = hohmann_transfer_params(origin, dest)?;
    let pos_o = compute_body_position_au(origin, j2000_days)?;
    let pos_d = compute_body_position_au(dest, j2000_days)?;
    let angle_o = pos_o[1].atan2(pos_o[0]);
    let angle_d = pos_d[1].atan2(pos_d[0]);
    let mut current_phase = angle_d - angle_o;
    while current_phase < -PI {
        current_phase += 2.0 * PI;
    }
    while current_phase > PI {
        current_phase -= 2.0 * PI;
    }

    let required = params.phase_angle_rad;
    let t1 = origin.orbital_period_days()?;
    let t2 = dest.orbital_period_days()?;
    let rel_rate = 2.0 * PI / t2 - 2.0 * PI / t1;
    if rel_rate.abs() < 1e-15 {
        return None;
    }

    let mut diff = required - current_phase;
    if rel_rate > 0.0 {
        while diff < 0.0 {
            diff += 2.0 * PI;
        }
    } else {
        while diff > 0.0 {
            diff -= 2.0 * PI;
        }
    }
    let wait = diff / rel_rate;
    Some(wait.max(0.0))
}

pub fn heliocentric_longitude(body: CelestialBody, j2000_days: f64) -> Option<f64> {
    let pos = compute_body_position_au(body, j2000_days)?;
    Some(pos[1].atan2(pos[0]))
}

pub fn next_conjunction_days(a: CelestialBody, b: CelestialBody, j2000_days: f64) -> Option<f64> {
    let t_a = a.orbital_period_days()?;
    let t_b = b.orbital_period_days()?;
    let rate_a = 2.0 * PI / t_a;
    let rate_b = 2.0 * PI / t_b;
    let rel_rate = rate_b - rate_a;
    if rel_rate.abs() < 1e-15 {
        return None;
    }

    let ang_a = heliocentric_longitude(a, j2000_days)?;
    let ang_b = heliocentric_longitude(b, j2000_days)?;
    let mut diff = ang_b - ang_a;
    while diff > PI {
        diff -= 2.0 * PI;
    }
    while diff < -PI {
        diff += 2.0 * PI;
    }

    let mut wait = -diff / rel_rate;
    if wait < 1.0 {
        wait += (2.0 * PI / rel_rate.abs()).abs();
    }

    for _ in 0..8 {
        let t = j2000_days + wait;
        let la = heliocentric_longitude(a, t)?;
        let lb = heliocentric_longitude(b, t)?;
        let mut err = lb - la;
        while err > PI {
            err -= 2.0 * PI;
        }
        while err < -PI {
            err += 2.0 * PI;
        }
        wait -= err / rel_rate;
        if wait < 0.0 {
            wait += (2.0 * PI / rel_rate.abs()).abs();
        }
    }

    Some(wait.max(0.0))
}

pub fn next_opposition_days(a: CelestialBody, b: CelestialBody, j2000_days: f64) -> Option<f64> {
    let t_a = a.orbital_period_days()?;
    let t_b = b.orbital_period_days()?;
    let rate_a = 2.0 * PI / t_a;
    let rate_b = 2.0 * PI / t_b;
    let rel_rate = rate_b - rate_a;
    if rel_rate.abs() < 1e-15 {
        return None;
    }

    let ang_a = heliocentric_longitude(a, j2000_days)?;
    let ang_b = heliocentric_longitude(b, j2000_days)?;
    let mut diff = ang_b - ang_a - PI;
    while diff > PI {
        diff -= 2.0 * PI;
    }
    while diff < -PI {
        diff += 2.0 * PI;
    }

    let mut wait = -diff / rel_rate;
    if wait < 1.0 {
        wait += (2.0 * PI / rel_rate.abs()).abs();
    }

    for _ in 0..8 {
        let t = j2000_days + wait;
        let la = heliocentric_longitude(a, t)?;
        let lb = heliocentric_longitude(b, t)?;
        let mut err = lb - la - PI;
        while err > PI {
            err -= 2.0 * PI;
        }
        while err < -PI {
            err += 2.0 * PI;
        }
        wait -= err / rel_rate;
        if wait < 0.0 {
            wait += (2.0 * PI / rel_rate.abs()).abs();
        }
    }

    Some(wait.max(0.0))
}

pub fn planet_angular_spread(bodies: &[CelestialBody], j2000_days: f64) -> Option<f64> {
    let mut angles: Vec<f64> = Vec::new();
    for &body in bodies {
        if let Some(a) = heliocentric_longitude(body, j2000_days) {
            angles.push(a);
        }
    }
    if angles.len() < 2 {
        return None;
    }
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut max_gap = 0.0_f64;
    for i in 0..angles.len() {
        let next = if i + 1 < angles.len() {
            angles[i + 1] - angles[i]
        } else {
            (angles[0] + 2.0 * PI) - angles[i]
        };
        max_gap = max_gap.max(next);
    }
    Some(2.0 * PI - max_gap)
}

pub fn next_alignment_days(
    bodies: &[CelestialBody],
    j2000_days: f64,
    search_years: f64,
) -> Option<(f64, f64)> {
    if bodies.len() < 2 {
        return None;
    }

    let mut best_spread = f64::MAX;
    let mut best_day = 0.0;

    let search_range = search_years * 365.25;
    let coarse_step = (search_range / 10_000.0).max(1.0);
    let mut t = j2000_days;
    while t < j2000_days + search_range {
        if let Some(spread) = planet_angular_spread(bodies, t) {
            if spread < best_spread {
                best_spread = spread;
                best_day = t;
            }
        }
        t += coarse_step;
    }

    let mid_start = (best_day - 2000.0).max(j2000_days);
    let mid_end = best_day + 2000.0;
    let mut t = mid_start;
    while t < mid_end {
        if let Some(spread) = planet_angular_spread(bodies, t) {
            if spread < best_spread {
                best_spread = spread;
                best_day = t;
            }
        }
        t += 10.0;
    }

    let fine_start = (best_day - 20.0).max(j2000_days);
    let fine_end = best_day + 20.0;
    let mut t = fine_start;
    while t < fine_end {
        if let Some(spread) = planet_angular_spread(bodies, t) {
            if spread < best_spread {
                best_spread = spread;
                best_day = t;
            }
        }
        t += 0.5;
    }

    let wait = best_day - j2000_days;
    Some((wait.max(0.0), best_spread.to_degrees()))
}

pub fn next_equinox_solstice(j2000_days: f64) -> (f64, &'static str) {
    let ts_epoch = *J2000_EPOCH;
    let current = ts_epoch + chrono::Duration::seconds((j2000_days * 86400.0) as i64);
    use chrono::Datelike;
    let doy = current.ordinal() as f64;
    let year = current.year();

    let events: [(f64, &str); 4] = [
        (80.0, "Vernal Equinox"),
        (172.0, "Summer Solstice"),
        (266.0, "Autumnal Equinox"),
        (355.0, "Winter Solstice"),
    ];

    let mut best_wait = f64::MAX;
    let mut best_name = events[0].1;

    for &(event_doy, name) in &events {
        let mut wait = event_doy - doy;
        if wait < 1.0 {
            wait += 365.0;
        }
        if wait < best_wait {
            best_wait = wait;
            best_name = name;
        }
    }

    let _ = year;
    (best_wait, best_name)
}

pub fn draw_circular_calendar(
    plot_ui: &mut egui_plot::PlotUi,
    j2000_days: f64,
    log_power: f64,
    dark_mode: bool,
) {
    let earth_sma = match CelestialBody::Earth.semi_major_axis_au() {
        Some(a) => a,
        None => return,
    };

    let earth_period = match CelestialBody::Earth.orbital_period_days() {
        Some(p) => p,
        None => return,
    };
    let mean_lon = CelestialBody::Earth.mean_longitude_j2000_deg().to_radians();

    let month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let month_colors: [eframe::egui::Color32; 12] = [
        eframe::egui::Color32::from_rgba_unmultiplied(135, 206, 235, 80),
        eframe::egui::Color32::from_rgba_unmultiplied(175, 238, 238, 80),
        eframe::egui::Color32::from_rgba_unmultiplied(144, 238, 144, 80),
        eframe::egui::Color32::from_rgba_unmultiplied(152, 251, 152, 80),
        eframe::egui::Color32::from_rgba_unmultiplied(255, 255, 150, 80),
        eframe::egui::Color32::from_rgba_unmultiplied(255, 218, 185, 80),
        eframe::egui::Color32::from_rgba_unmultiplied(255, 182, 135, 80),
        eframe::egui::Color32::from_rgba_unmultiplied(255, 160, 122, 80),
        eframe::egui::Color32::from_rgba_unmultiplied(244, 164, 96, 80),
        eframe::egui::Color32::from_rgba_unmultiplied(210, 180, 140, 80),
        eframe::egui::Color32::from_rgba_unmultiplied(176, 196, 222, 80),
        eframe::egui::Color32::from_rgba_unmultiplied(173, 216, 230, 80),
    ];

    let ts_epoch = *J2000_EPOCH;
    let current_dt = ts_epoch + chrono::Duration::seconds((j2000_days * 86400.0) as i64);
    use chrono::Datelike;
    let current_year = current_dt.year();

    let outer_r = earth_sma;

    let mut month_boundaries = Vec::with_capacity(13);
    for m in 0..12 {
        let date = chrono::NaiveDate::from_ymd_opt(current_year, m as u32 + 1, 1);
        if let Some(date) = date {
            let dt = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
            let boundary_j2000 = (dt - ts_epoch).num_seconds() as f64 / 86400.0;
            let angle = mean_lon + 2.0 * PI * boundary_j2000 / earth_period;
            month_boundaries.push(angle);
        }
    }
    if !month_boundaries.is_empty() {
        month_boundaries.push(month_boundaries[0] + 2.0 * PI);
    }

    let mut painter = plot_ui
        .ctx()
        .layer_painter(eframe::egui::LayerId::background());
    let clip = plot_ui.response().rect;
    painter.set_clip_rect(clip);

    let to_screen = |px: f64, py: f64| -> eframe::egui::Pos2 {
        plot_ui.screen_from_plot(PlotPoint::new(px, py))
    };

    let arc_segments = 20;
    for m in 0..12 {
        if m + 1 >= month_boundaries.len() {
            break;
        }
        let a_start = month_boundaries[m];
        let a_end = month_boundaries[m + 1];

        let mut screen_pts: Vec<eframe::egui::Pos2> = Vec::with_capacity(arc_segments + 3);
        let [cx, cy] = scale_position(0.0, 0.0, log_power);
        screen_pts.push(to_screen(cx, cy));
        for i in 0..=arc_segments {
            let frac = i as f64 / arc_segments as f64;
            let a = a_start + (a_end - a_start) * frac;
            let ox = outer_r * a.cos();
            let oy = outer_r * a.sin();
            let [sx, sy] = scale_position(ox, oy, log_power);
            screen_pts.push(to_screen(sx, sy));
        }

        let stroke_color = eframe::egui::Color32::from_rgba_unmultiplied(
            month_colors[m].r(),
            month_colors[m].g(),
            month_colors[m].b(),
            30,
        );
        let shape = eframe::egui::Shape::convex_polygon(
            screen_pts,
            month_colors[m],
            eframe::egui::Stroke::new(1.0, stroke_color),
        );
        painter.add(shape);

        let mid_angle = (a_start + a_end) / 2.0;
        let label_r = earth_sma * 0.65;
        let lx = label_r * mid_angle.cos();
        let ly = label_r * mid_angle.sin();
        let [sx, sy] = scale_position(lx, ly, log_power);
        let label_screen = to_screen(sx, sy);

        let label_color = if dark_mode {
            eframe::egui::Color32::from_rgba_unmultiplied(255, 255, 255, 160)
        } else {
            eframe::egui::Color32::from_rgba_unmultiplied(0, 0, 0, 160)
        };
        painter.text(
            label_screen,
            eframe::egui::Align2::CENTER_CENTER,
            month_names[m],
            eframe::egui::FontId::proportional(16.0),
            label_color,
        );
    }

    if let Some(earth_pos) = compute_body_position_au(CelestialBody::Earth, j2000_days) {
        let earth_angle = earth_pos[1].atan2(earth_pos[0]);
        let [s1x, s1y] = scale_position(0.0, 0.0, log_power);
        let marker_outer = earth_sma * 1.04;
        let p2x = marker_outer * earth_angle.cos();
        let p2y = marker_outer * earth_angle.sin();
        let [s2x, s2y] = scale_position(p2x, p2y, log_power);

        painter.line_segment(
            [to_screen(s1x, s1y), to_screen(s2x, s2y)],
            eframe::egui::Stroke::new(2.5, eframe::egui::Color32::from_rgb(255, 220, 50)),
        );
    }
}

pub struct HohmannState {
    pub origin: CelestialBody,
    pub dest: CelestialBody,
    pub launched: bool,
    pub launch_j2000_days: f64,
    pub mission_elapsed_days: f64,
    pub trail: Vec<[f64; 2]>,
    pub departure_angle: f64,
    pub arrival_angle: f64,
    pub arrived: bool,
}

impl Default for HohmannState {
    fn default() -> Self {
        Self {
            origin: CelestialBody::Earth,
            dest: CelestialBody::Mars,
            launched: false,
            launch_j2000_days: 0.0,
            mission_elapsed_days: 0.0,
            trail: Vec::new(),
            departure_angle: 0.0,
            arrival_angle: 0.0,
            arrived: false,
        }
    }
}

fn transfer_point(
    fraction: f64,
    a_t: f64,
    e: f64,
    going_outward: bool,
    departure_angle: f64,
) -> [f64; 2] {
    let m = if going_outward {
        PI * fraction
    } else {
        PI * (1.0 + fraction)
    };
    let ea = solve_kepler(m, e);
    let r = a_t * (1.0 - e * ea.cos());
    let true_anomaly =
        2.0 * ((1.0 + e).sqrt() * (ea / 2.0).sin()).atan2((1.0 - e).sqrt() * (ea / 2.0).cos());
    let angle = if going_outward {
        departure_angle + true_anomaly
    } else {
        departure_angle - PI + true_anomaly
    };
    [r * angle.cos(), r * angle.sin()]
}

pub fn hohmann_spacecraft_position_au(state: &HohmannState, j2000_days: f64) -> Option<[f64; 2]> {
    if state.arrived {
        return compute_body_position_au(state.dest, j2000_days);
    }

    let params = hohmann_transfer_params(state.origin, state.dest)?;
    let r1 = state.origin.semi_major_axis_au()?;
    let r2 = state.dest.semi_major_axis_au()?;
    let going_outward = r2 >= r1;

    let fraction = (state.mission_elapsed_days / params.transfer_time_days).clamp(0.0, 1.0);

    Some(transfer_point(
        fraction,
        params.transfer_sma,
        params.eccentricity,
        going_outward,
        state.departure_angle,
    ))
}

pub fn draw_hohmann_overlay(
    plot_ui: &mut egui_plot::PlotUi,
    state: &HohmannState,
    j2000_days: f64,
    log_power: f64,
    dark_mode: bool,
) {
    let params = match hohmann_transfer_params(state.origin, state.dest) {
        Some(p) => p,
        None => return,
    };
    let r1 = match state.origin.semi_major_axis_au() {
        Some(r) => r,
        None => return,
    };
    let r2 = match state.dest.semi_major_axis_au() {
        Some(r) => r,
        None => return,
    };

    if !state.launched {
        return;
    }

    let depart_angle = state.departure_angle;
    let going_outward = r2 >= r1;

    let n_pts = 200;
    let mut ellipse_pts: Vec<[f64; 2]> = Vec::with_capacity(n_pts + 1);
    for i in 0..=n_pts {
        let frac = i as f64 / n_pts as f64;
        let pt = transfer_point(
            frac,
            params.transfer_sma,
            params.eccentricity,
            going_outward,
            depart_angle,
        );
        ellipse_pts.push(scale_position(pt[0], pt[1], log_power));
    }
    let ellipse_color = if dark_mode {
        eframe::egui::Color32::from_rgba_unmultiplied(100, 255, 100, 120)
    } else {
        eframe::egui::Color32::from_rgba_unmultiplied(0, 180, 0, 140)
    };
    plot_ui.line(
        Line::new("", ellipse_pts)
            .color(ellipse_color)
            .width(2.0)
            .style(egui_plot::LineStyle::dashed_dense()),
    );

    if let Some(pos) = hohmann_spacecraft_position_au(state, j2000_days) {
        let sp = scale_position(pos[0], pos[1], log_power);

        if state.trail.len() >= 2 {
            let max_trail = 500;
            let start = if state.trail.len() > max_trail {
                state.trail.len() - max_trail
            } else {
                0
            };
            let trail_pts: Vec<[f64; 2]> = state.trail[start..]
                .iter()
                .map(|p| scale_position(p[0], p[1], log_power))
                .collect();
            let trail_color = if dark_mode {
                eframe::egui::Color32::from_rgba_unmultiplied(255, 200, 50, 100)
            } else {
                eframe::egui::Color32::from_rgba_unmultiplied(200, 150, 0, 120)
            };
            plot_ui.line(Line::new("", trail_pts).color(trail_color).width(2.5));
        }

        let bounds = plot_ui.plot_bounds();
        let view_size = (bounds.max()[0] - bounds.min()[0]).max(bounds.max()[1] - bounds.min()[1]);
        let marker_r = view_size * 0.008;
        let marker_pts = circle_points(sp[0], sp[1], marker_r, 16);
        let sc_color = eframe::egui::Color32::from_rgb(255, 220, 50);
        plot_ui.line(Line::new("", marker_pts).color(sc_color).width(3.0));
        plot_ui.points(
            egui_plot::Points::new("", vec![sp])
                .color(sc_color)
                .radius(5.0),
        );

        let dep_pos = scale_position(r1 * depart_angle.cos(), r1 * depart_angle.sin(), log_power);
        let dv1_text = format!("\u{0394}v1: {:.2} km/s", params.departure_dv_km_s);
        plot_ui.text(
            Text::new(
                "",
                PlotPoint::new(dep_pos[0], dep_pos[1] - view_size * 0.03),
                eframe::egui::RichText::new(dv1_text)
                    .size(11.0)
                    .color(eframe::egui::Color32::from_rgb(100, 255, 100)),
            )
            .color(eframe::egui::Color32::from_rgb(100, 255, 100)),
        );

        let arr_angle = depart_angle + PI;
        let arr_pos = scale_position(r2 * arr_angle.cos(), r2 * arr_angle.sin(), log_power);
        let dv2_text = format!("\u{0394}v2: {:.2} km/s", params.arrival_dv_km_s);
        plot_ui.text(
            Text::new(
                "",
                PlotPoint::new(arr_pos[0], arr_pos[1] - view_size * 0.03),
                eframe::egui::RichText::new(dv2_text)
                    .size(11.0)
                    .color(eframe::egui::Color32::from_rgb(100, 255, 100)),
            )
            .color(eframe::egui::Color32::from_rgb(100, 255, 100)),
        );

        let met_days = state.mission_elapsed_days;
        let info_x = bounds.min()[0] + view_size * 0.02;
        let info_y = bounds.max()[1] - view_size * 0.02;
        let total_dv = params.departure_dv_km_s + params.arrival_dv_km_s;
        let progress = (met_days / params.transfer_time_days * 100.0).min(100.0);
        let info = format!(
            "MET: {:.1} days  ({:.0}%)\n\u{0394}v total: {:.2} km/s",
            met_days, progress, total_dv,
        );
        let info_color = if dark_mode {
            eframe::egui::Color32::from_rgb(200, 200, 200)
        } else {
            eframe::egui::Color32::from_rgb(40, 40, 40)
        };
        plot_ui.text(
            Text::new(
                "",
                PlotPoint::new(info_x, info_y),
                eframe::egui::RichText::new(info).size(12.0),
            )
            .color(info_color)
            .anchor(eframe::egui::Align2::LEFT_TOP),
        );
    }
}

struct MoonOrbit {
    parent: CelestialBody,
    distance_au: f64,
    period_days: f64,
}

fn moon_orbit(body: CelestialBody) -> Option<MoonOrbit> {
    match body {
        CelestialBody::Moon => Some(MoonOrbit {
            parent: CelestialBody::Earth,
            distance_au: 0.00257,
            period_days: 27.32,
        }),
        CelestialBody::Ganymede => Some(MoonOrbit {
            parent: CelestialBody::Jupiter,
            distance_au: 0.00716,
            period_days: 7.155,
        }),
        CelestialBody::Callisto => Some(MoonOrbit {
            parent: CelestialBody::Jupiter,
            distance_au: 0.01258,
            period_days: 16.689,
        }),
        CelestialBody::Io => Some(MoonOrbit {
            parent: CelestialBody::Jupiter,
            distance_au: 0.00282,
            period_days: 1.769,
        }),
        CelestialBody::Europa => Some(MoonOrbit {
            parent: CelestialBody::Jupiter,
            distance_au: 0.00449,
            period_days: 3.551,
        }),
        CelestialBody::Titan => Some(MoonOrbit {
            parent: CelestialBody::Saturn,
            distance_au: 0.00817,
            period_days: 15.945,
        }),
        CelestialBody::Triton => Some(MoonOrbit {
            parent: CelestialBody::Neptune,
            distance_au: 0.00237,
            period_days: -5.877,
        }),
        CelestialBody::Charon => Some(MoonOrbit {
            parent: CelestialBody::Pluto,
            distance_au: 0.000131,
            period_days: 6.387,
        }),
        CelestialBody::Enceladus => Some(MoonOrbit {
            parent: CelestialBody::Saturn,
            distance_au: 0.00159,
            period_days: 1.370,
        }),
        CelestialBody::Mimas => Some(MoonOrbit {
            parent: CelestialBody::Saturn,
            distance_au: 0.00124,
            period_days: 0.942,
        }),
        CelestialBody::Iapetus => Some(MoonOrbit {
            parent: CelestialBody::Saturn,
            distance_au: 0.0238,
            period_days: 79.32,
        }),
        CelestialBody::Phobos => Some(MoonOrbit {
            parent: CelestialBody::Mars,
            distance_au: 0.0000628,
            period_days: 0.319,
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
    let data = json["data"].as_array().ok_or("Missing data field")?;
    let mut asteroids = Vec::with_capacity(data.len());
    for row in data {
        let arr = row.as_array().ok_or("Row not array")?;
        let name = arr
            .get(0)
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

pub static J2000_EPOCH_PUB: std::sync::LazyLock<chrono::DateTime<chrono::Utc>> =
    std::sync::LazyLock::new(|| *J2000_EPOCH);

fn scale_position(x: f64, y: f64, power: f64) -> [f64; 2] {
    let r = (x * x + y * y).sqrt();
    if r < 1e-10 {
        return [0.0, 0.0];
    }
    let r_scaled = (r + SCALE_OFFSET).powf(power) - SCALE_OFFSET.powf(power);
    let s = r_scaled / r;
    [x * s, y * s]
}

pub fn compute_body_position_au(body: CelestialBody, j2000_days: f64) -> Option<[f64; 2]> {
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
    show_labels: bool,
    show_calendar: bool,
    hide_bodies: bool,
) -> Option<CelestialBody> {
    let j2000_days = (timestamp - *J2000_EPOCH).num_seconds() as f64 / 86400.0;

    let label_color = if dark_mode {
        eframe::egui::Color32::WHITE
    } else {
        eframe::egui::Color32::BLACK
    };

    let bounds = plot_ui.plot_bounds();
    let view_size = (bounds.max()[0] - bounds.min()[0]).max(bounds.max()[1] - bounds.min()[1]);
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

        let visual_radius =
            (sun_visual_radius * (body.radius_km() / sun_km).powf(log_power)).max(min_radius);

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

        let orbit_color = if dark_mode {
            eframe::egui::Color32::from_rgb(80, 220, 120)
        } else {
            eframe::egui::Color32::from_rgb(30, 140, 60)
        };
        let orbit_width = if body == focused_body { 5.0 } else { 4.0 };

        plot_ui.line(Line::new("", points).color(orbit_color).width(orbit_width));
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
            plot_ui.line(Line::new("", orbit_pts).color(orbit_color).width(1.0));
        }
    }

    // Asteroid belt rendered as a filled annulus between 2.2 AU and 3.3 AU
    // (the canonical inner/outer edges of the main belt). Built from triangle
    // strips because egui_plot Polygons don't support holes. Edges are outlined
    // for clarity.
    {
        let fill_color = if dark_mode {
            eframe::egui::Color32::from_rgba_unmultiplied(210, 190, 160, 70)
        } else {
            eframe::egui::Color32::from_rgba_unmultiplied(110, 85, 55, 90)
        };
        let edge_color = if dark_mode {
            eframe::egui::Color32::from_rgba_unmultiplied(220, 200, 170, 200)
        } else {
            eframe::egui::Color32::from_rgba_unmultiplied(110, 85, 55, 220)
        };
        let inner_scaled = (2.2_f64 + SCALE_OFFSET).powf(log_power) - SCALE_OFFSET.powf(log_power);
        let outer_scaled = (3.3_f64 + SCALE_OFFSET).powf(log_power) - SCALE_OFFSET.powf(log_power);
        let segments = 128;
        for i in 0..segments {
            let t0 = 2.0 * std::f64::consts::PI * i as f64 / segments as f64;
            let t1 = 2.0 * std::f64::consts::PI * (i + 1) as f64 / segments as f64;
            let quad: Vec<[f64; 2]> = vec![
                [inner_scaled * t0.cos(), inner_scaled * t0.sin()],
                [outer_scaled * t0.cos(), outer_scaled * t0.sin()],
                [outer_scaled * t1.cos(), outer_scaled * t1.sin()],
                [inner_scaled * t1.cos(), inner_scaled * t1.sin()],
            ];
            plot_ui.polygon(
                egui_plot::Polygon::new("", egui_plot::PlotPoints::new(quad))
                    .fill_color(fill_color)
                    .stroke(eframe::egui::Stroke::NONE),
            );
        }
        // Inner and outer edge outlines.
        for edge in [inner_scaled, outer_scaled] {
            let pts: Vec<[f64; 2]> = (0..=segments)
                .map(|i| {
                    let t = 2.0 * std::f64::consts::PI * i as f64 / segments as f64;
                    [edge * t.cos(), edge * t.sin()]
                })
                .collect();
            plot_ui.line(Line::new("", pts).color(edge_color).width(1.5));
        }
        // Belt label, placed just outside the outer edge on the +y axis so it
        // doesn't overlap with Ceres' orbit (which lies inside the belt).
        let label_size = ((90.0 / view_size.max(0.01)).clamp(12.0, 22.0) as f32).round();
        plot_ui.text(
            Text::new(
                "",
                PlotPoint::new(0.0, outer_scaled + view_size * 0.03),
                eframe::egui::RichText::new("Asteroid Belt").size(label_size),
            )
            .color(label_color),
        );
    }

    let mut ast_positions: Vec<(usize, f64, f64)> = Vec::new();
    if !asteroids.is_empty() && !hide_bodies {
        let belt_scaled = (3.0 + SCALE_OFFSET).powf(log_power) - SCALE_OFFSET.powf(log_power);
        // Opaque from normal-zoom down to moderate zoom-out, then fade to zero
        // when the view reaches outer-planet scales (Uranus ~19 AU, Neptune
        // ~30 AU), so the belt doesn't clutter wide-system views. `ratio` is
        // the fraction of the view that the belt occupies; 1.0 = belt fills
        // the view, small values = zoomed way out.
        let ratio = (belt_scaled / view_size) as f32;
        let alpha_f = ((ratio - 0.06) / 0.10).clamp(0.0, 1.0);
        if alpha_f > 0.0 {
            let alpha = (alpha_f * 255.0) as u8;
            let asteroid_color = if dark_mode {
                eframe::egui::Color32::from_rgba_unmultiplied(230, 210, 180, alpha)
            } else {
                eframe::egui::Color32::from_rgba_unmultiplied(90, 70, 50, alpha)
            };
            let mut pts: Vec<[f64; 2]> = Vec::with_capacity(asteroids.len());
            for (idx, ast) in asteroids.iter().enumerate() {
                let pos = asteroid_position(ast, j2000_days);
                let scaled = scale_position(pos[0], pos[1], log_power);
                ast_positions.push((idx, scaled[0], scaled[1]));
                pts.push([scaled[0], scaled[1]]);
            }
            let ast_radius = (ratio * 6.0).clamp(1.2, 5.0);
            plot_ui.points(
                egui_plot::Points::new("", pts)
                    .color(asteroid_color)
                    .radius(ast_radius),
            );
        } else {
            // Still collect positions for hover/picking even when not drawn.
            for (idx, ast) in asteroids.iter().enumerate() {
                let pos = asteroid_position(ast, j2000_days);
                let scaled = scale_position(pos[0], pos[1], log_power);
                ast_positions.push((idx, scaled[0], scaled[1]));
            }
        }
    }

    if show_calendar {
        draw_circular_calendar(plot_ui, j2000_days, log_power, dark_mode);
    }

    // Snap label size so the glyph atlas stays hot while view_size sweeps
    // continuously during auto-zoom — otherwise each frame asks egui to
    // rasterize a slightly-different pixel size and the CPU glyph-cache
    // churn shows up as per-frame lag spikes.
    let base_label_size = ((90.0 / view_size.max(0.01)).clamp(12.0, 22.0) as f32).round();

    for &(body, x, y, visual_radius) in &bodies {
        // In `hide_bodies` mode every planet body image is skipped — only the
        // Sun's texture is drawn. Orbits and labels remain so you can still
        // see where each planet is without the rendered sphere.
        let draw_body = !hide_bodies || body == CelestialBody::Sun;
        if draw_body {
            if let Some(handle) = sphere_handles.get(&body) {
                let ring_scale = body
                    .ring_params()
                    .map(|(_, _, o)| o as f64)
                    .unwrap_or(1.0)
                    .max(1.0);
                let img_size = (visual_radius * 2.0 * ring_scale) as f32;
                plot_ui.image(PlotImage::new(
                    "",
                    handle.id(),
                    PlotPoint::new(x, y),
                    [img_size, img_size],
                ));
            }
        } else if body != CelestialBody::Sun && body.parent_body().is_none() {
            // Mark the planet's position with a small filled dot so the viewer
            // can still see where it is (same style as satellite markers).
            plot_ui.points(
                egui_plot::Points::new("", vec![[x, y]])
                    .color(label_color)
                    .radius(4.0)
                    .filled(true),
            );
        }

        if show_labels && (body.parent_body().is_none() || body == focused_body) {
            // Use the same label color for every planet (including the focused
            // one) so Earth doesn't end up tinted blue when it's the focus.
            let name_color = label_color;

            let dist_from_center = (x * x + y * y).sqrt();
            let edge_frac = (dist_from_center / (view_size * 0.45)).clamp(0.0, 1.0) as f32;
            let label_font_size = (base_label_size + edge_frac * 4.0).round();

            let label_text = if body != CelestialBody::Sun {
                let offset_p = SCALE_OFFSET.powf(log_power);
                let au = if dist_from_center > 1e-6 {
                    (dist_from_center + offset_p).powf(1.0 / log_power) - SCALE_OFFSET
                } else {
                    0.0
                };
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
            let real_au = if sr > 1e-6 {
                (sr + offset_p).powf(1.0 / log_power) - SCALE_OFFSET
            } else {
                0.0
            };
            let screen_pos = plot_ui.screen_from_plot(PlotPoint::new(pointer.x, pointer.y));
            let offset_screen = eframe::egui::Pos2::new(screen_pos.x + 12.0, screen_pos.y - 12.0);
            let offset_plot = plot_ui.plot_from_screen(offset_screen);
            plot_ui.text(
                Text::new(
                    "",
                    offset_plot,
                    eframe::egui::RichText::new(format!("{:.2} AU", real_au)).size(12.0),
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
                    ui.label(
                        eframe::egui::RichText::new(body.label())
                            .strong()
                            .size(16.0),
                    );
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
                    let (_, ax, ay) = ast_positions[ast_positions
                        .iter()
                        .position(|&(i, _, _)| i == idx)
                        .unwrap()];
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
                                    ui.label(format!(
                                        "{:.3} AU\n({:.1}M km)",
                                        ast.a,
                                        km / 1_000_000.0
                                    ));
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
    enabled: &mut std::collections::HashSet<CelestialBody>,
) -> Option<CelestialBody> {
    use eframe::egui;

    ui.horizontal(|ui| {
        if ui
            .selectable_label(enabled.len() == CelestialBody::ALL.len(), "All")
            .clicked()
        {
            *enabled = CelestialBody::ALL.iter().copied().collect();
        }
        for cat in &["Star", "Planets", "Dwarf Planets", "Asteroids", "Moons"] {
            let in_cat: Vec<CelestialBody> = CelestialBody::ALL
                .iter()
                .copied()
                .filter(|b| b.category() == *cat)
                .collect();
            let all_on = in_cat.iter().all(|b| enabled.contains(b));
            if ui.selectable_label(all_on, *cat).clicked() {
                for b in &in_cat {
                    if all_on {
                        enabled.remove(b);
                    } else {
                        enabled.insert(*b);
                    }
                }
            }
        }
    });

    let mut sorted: Vec<CelestialBody> = CelestialBody::ALL
        .iter()
        .copied()
        .filter(|b| enabled.contains(b))
        .collect();
    sorted.sort_by(|a, b| b.radius_km().partial_cmp(&a.radius_km()).unwrap());
    if sorted.is_empty() {
        return None;
    }

    let available = ui.available_size();
    let mut clicked_body = None;

    {
        let n = sorted.len();
        *zoom_t = zoom_t.clamp(0.0, (n - 1) as f64);

        let text_color = ui.visuals().text_color();
        let weak_color = ui.visuals().weak_text_color();

        let (response, painter) = ui.allocate_painter(
            egui::Vec2::new(available.x, available.y),
            egui::Sense::click().union(egui::Sense::hover()),
        );
        let rect = response.rect;
        let painter = painter.with_clip_rect(rect);

        if !auto_zoom.enabled && response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.1 {
                *zoom_t = (*zoom_t - scroll as f64 * 0.005).clamp(0.0, (n - 1) as f64);
                ui.ctx().request_repaint();
            }
        }

        let view_h = rect.height() as f64;
        let view_w = rect.width() as f64;

        let base_name_pre = (view_h as f32 * 0.06).clamp(18.0, 48.0);
        let base_km_pre = (view_h as f32 * 0.045).clamp(14.0, 36.0);
        let label_reserve_f64 = (base_name_pre + base_km_pre * 0.9 + base_km_pre + 16.0) as f64;
        let body_h = view_h - label_reserve_f64;

        let margin = 12.0;
        let compute_layout = |k: usize| -> Vec<(f64, f64)> {
            let r_focus = sorted[k].radius_km();
            let h_scale = body_h / (2.0 * r_focus);
            let num_gaps = (n - k).saturating_sub(1).max(1) as f64;
            let usable_w = view_w - 2.0 * margin;
            let body_ext = |_b: &CelestialBody| -> f64 { 1.0 };
            let total_extent: f64 = sorted[k..]
                .iter()
                .map(|b| 2.0 * b.radius_km() * body_ext(b))
                .sum();
            let prelim_scale = h_scale.min(usable_w / total_extent);
            let min_r_px = sorted[k..]
                .last()
                .map(|b| b.radius_km() * prelim_scale)
                .unwrap_or(1.0);
            let gap = (min_r_px * 0.5).clamp(1.0, 8.0);
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
            let frac = if t < stay {
                0.0
            } else if t < stay + scroll {
                (t - stay) / scroll
            } else if t < 2.0 * stay + scroll {
                1.0
            } else {
                1.0 - (t - 2.0 * stay - scroll) / scroll
            };
            // Classic smoothstep: 3t^2 - 2t^3. Gentler than the rational ease.
            let s = frac;
            let eased = s * s * (3.0 - 2.0 * s);
            let target = eased * total;

            let seg = cum
                .partition_point(|&d| d <= target)
                .saturating_sub(1)
                .min(n - 2);
            let frac = (target - cum[seg]) / (cum[seg + 1] - cum[seg]);
            *zoom_t = (seg as f64 + frac).clamp(0.0, (n - 1) as f64);
            ui.ctx().request_repaint();
        }

        let i = (*zoom_t as usize).min(n - 2);
        let frac = *zoom_t - i as f64;

        let layout_a = compute_layout(i);
        let layout_b = compute_layout((i + 1).min(n - 1));
        let base_name = (view_h as f32 * 0.06).clamp(18.0, 48.0);
        let base_km = (view_h as f32 * 0.045).clamp(14.0, 36.0);
        let label_reserve = base_name + base_km * 0.9 + base_km + 16.0;
        let baseline_y = rect.bottom() - label_reserve;

        let mut screen_bodies: Vec<(CelestialBody, f32, f32, f32)> = Vec::new();

        for j in 0..n {
            let (xa, ra) = layout_a[j];
            let (xb, rb) = layout_b[j];
            let cx = (view_w - (xa * (1.0 - frac) + xb * frac)) as f32 + rect.left();
            let r_px = (ra * (1.0 - frac) + rb * frac) as f32;

            let body = sorted[j];
            let vis_extent = r_px;

            if cx + vis_extent < rect.left() - 10.0 || cx - vis_extent > rect.right() + 10.0 {
                continue;
            }

            let body_cy = baseline_y - r_px;
            screen_bodies.push((body, cx, r_px, body_cy));
            if let Some(handle) = sphere_handles.get(&body) {
                let img_rect = egui::Rect::from_center_size(
                    egui::Pos2::new(cx, body_cy),
                    egui::Vec2::splat(r_px * 2.0),
                );
                painter.image(
                    handle.id(),
                    img_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            }

            if r_px > 3.0 {
                let decay = ((*zoom_t - j as f64) * 0.3).exp() as f32;
                // Smooth scaling: rasterize each label once at the max font
                // size (base_name / base_km) and then scale the tessellated
                // mesh by `decay`. The glyph atlas stays hot across frames
                // and the GPU interpolates between frames — no per-frame
                // rasterization wobble.
                let scale = decay.clamp(0.2, 1.0);
                let total_label_h = (base_name + base_km * 0.9 + base_km + 6.0) * scale;
                let below_y = baseline_y + 8.0;
                let label_y_start = if below_y + total_label_h > rect.bottom() - 4.0 {
                    (rect.bottom() - total_label_h - 4.0).max(baseline_y)
                } else {
                    below_y
                };

                let subtitle = if let Some(parent) = body.parent_body() {
                    Some(format!("Moon of {}", parent.label()))
                } else {
                    match body.category() {
                        "Dwarf Planets" => Some("Dwarf Planet".to_string()),
                        "Asteroids" => Some("Asteroid".to_string()),
                        _ => None,
                    }
                };

                let mut entries: Vec<(String, f32, egui::Color32)> = Vec::new();
                entries.push((body.label().to_string(), base_name, text_color));
                if let Some(sub) = subtitle {
                    entries.push((sub, base_km * 0.9, weak_color));
                }
                entries.push((format!("{:.0} km", body.radius_km()), base_km, weak_color));

                let ppp = ui.ctx().pixels_per_point();
                let font_tex_size = ui.ctx().fonts(|f| f.font_image_size());
                let mut tess = egui::epaint::Tessellator::new(
                    ppp,
                    egui::epaint::TessellationOptions::default(),
                    font_tex_size,
                    Vec::new(),
                );

                let mut local_y = 0.0f32;
                for (text, size, color) in entries {
                    let galley =
                        painter.layout_no_wrap(text, egui::FontId::proportional(size), color);
                    let gw = galley.size().x;
                    let gh = galley.size().y;
                    let text_shape = egui::epaint::TextShape::new(
                        egui::Pos2::new(-gw * 0.5, local_y),
                        galley,
                        color,
                    );
                    let mut mesh = egui::epaint::Mesh::default();
                    tess.tessellate_text(&text_shape, &mut mesh);
                    for v in &mut mesh.vertices {
                        v.pos =
                            egui::Pos2::new(cx + v.pos.x * scale, label_y_start + v.pos.y * scale);
                    }
                    painter.add(egui::Shape::Mesh(std::sync::Arc::new(mesh)));
                    local_y += gh + 2.0;
                }
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

        let draw_highlight =
            |painter: &egui::Painter, body: CelestialBody, cx: f32, cy: f32, r_px: f32| {
                let ring_r = r_px * 1.15;
                let n_pts = 64;
                for i in 0..n_pts {
                    let a0 = std::f32::consts::TAU * i as f32 / n_pts as f32;
                    let a1 = std::f32::consts::TAU * (i + 1) as f32 / n_pts as f32;
                    painter.line_segment(
                        [
                            egui::Pos2::new(cx + ring_r * a0.cos(), cy + ring_r * a0.sin()),
                            egui::Pos2::new(cx + ring_r * a1.cos(), cy + ring_r * a1.sin()),
                        ],
                        egui::Stroke::new(2.0, body.display_color()),
                    );
                }
            };

        if response.hovered() || response.clicked() {
            if let Some(pointer) = ui.ctx().pointer_hover_pos() {
                for &(body, cx, r_px, cy) in &screen_bodies {
                    let dx = pointer.x - cx;
                    let dy = pointer.y - cy;
                    let dist = (dx * dx + dy * dy).sqrt();
                    let hit = r_px.max(12.0);
                    if dist <= hit {
                        draw_highlight(&painter, body, cx, cy, r_px);
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
