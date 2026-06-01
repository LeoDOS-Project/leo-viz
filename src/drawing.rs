//! 2D and 3D drawing routines for satellite visualizations.
//!
//! Renders the 3D globe view, torus topology view, ground track map, and
//! satellite camera projections. Handles orbit lines, coverage cones,
//! inter-satellite links, routing paths, and place markers.

use crate::celestial::{CelestialBody, Skin, TextureResolution};
use crate::config::{
    AoiJobMode, AreaOfInterest, ConjunctionInfo, DeviceLayer, GroundStation, RadiationConfig,
    SatelliteCamera, View3DFlags,
};
use crate::geo::CityLabel;
use crate::math::{rotate_point_matrix, rotation_from_drag};
use crate::renderer::{
    mat3_to_padded_cols, HeatmapPaintCallback, HeatmapUniforms, MapPaintCallback,
    PlanetPaintCallback, PlanetUniforms, SunPaintCallback, SunUniforms,
};
use crate::texture::EarthTexture;
use crate::walker::{SatelliteState, WalkerConstellation, WalkerType};
use eframe::egui;
use egui::mutex::Mutex;
use egui_plot::{Line, Plot, PlotImage, PlotPoint, PlotPoints, Points, Polygon, Text};
use nalgebra::{Matrix3, Vector3};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::sync::Arc;

use crate::EARTH_VISUAL_SCALE;

fn normalize_field_nt(f: f64, r_earth_radii: f64) -> f64 {
    let ref_field = 30000.0 / r_earth_radii.powi(3);
    let lo = ref_field * 0.67;
    let hi = ref_field * 2.2;
    ((f - lo) / (hi - lo)).clamp(0.0, 1.0)
}

fn blend_proton_electron(
    p: f64,
    e: f64,
    show_p: bool,
    show_e: bool,
    smooth: bool,
) -> egui::Color32 {
    let pv = if show_p { p } else { 0.0 };
    let ev = if show_e { e } else { 0.0 };
    crate::config::heatmap_color(pv.max(ev), smooth)
}

#[allow(dead_code)]
fn igrf_rad_to_rgba(grid: &crate::igrf::IgrfRadGrid) -> Vec<u8> {
    let w = 181;
    let h = 91;
    let mut data = vec![0u8; w * h * 4];
    for ci in 0..h {
        for li in 0..w {
            let idx = (ci * w + li) * 4;
            let p = (grid.protons[ci * w + li] * 255.0).round() as u8;
            let e = (grid.electrons[ci * w + li] * 255.0).round() as u8;
            data[idx] = p;
            data[idx + 1] = e;
            data[idx + 2] = 0;
            data[idx + 3] = 255;
        }
    }
    data
}

/// Rough atmospheric decay lifetime for a circular orbit at `altitude_km`,
/// assuming an exponential isothermal atmosphere with the same reference
/// values used by `app.rs` (ρ₀=2.8e-12 kg/m³ at 400 km, H=60 km). Closed-form
/// integral of `dt = B / (ρ·v·a) dh` from 100 km up to the current altitude,
/// holding v and a fixed at their current values. Returns seconds.
fn orbital_lifetime_seconds(
    altitude_km: f64,
    ballistic_coeff: f64,
    planet_radius_km: f64,
    planet_mu: f64,
) -> f64 {
    if altitude_km <= 100.0 || ballistic_coeff <= 0.0 {
        return 0.0;
    }
    let scale_height = 60.0_f64;
    let rho_ref = 2.8e-12_f64;
    let h_ref = 400.0_f64;
    let r_km = planet_radius_km + altitude_km;
    let v_ms = (planet_mu / r_km).sqrt() * 1000.0;
    let a_m = r_km * 1000.0;
    let f_hi = ((altitude_km - h_ref) / scale_height).exp();
    let f_lo = ((100.0 - h_ref) / scale_height).exp();
    1000.0 * ballistic_coeff * scale_height * (f_hi - f_lo) / (rho_ref * v_ms * a_m)
}

fn format_duration(seconds: f64) -> String {
    if !seconds.is_finite() || seconds <= 0.0 {
        return "decayed".to_string();
    }
    let minutes = seconds / 60.0;
    let hours = minutes / 60.0;
    let days = hours / 24.0;
    let years = days / 365.25;
    if years >= 1000.0 {
        format!("{:.0} kyr", years / 1000.0)
    } else if years >= 1.0 {
        format!("{:.1} yr", years)
    } else if days >= 1.0 {
        format!("{:.1} days", days)
    } else if hours >= 1.0 {
        format!("{:.1} h", hours)
    } else if minutes >= 1.0 {
        format!("{:.1} min", minutes)
    } else {
        format!("{:.0} s", seconds)
    }
}

fn dist_pos2_to_segment(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let ab_len_sq = ab.length_sq();
    if ab_len_sq < 1e-6 {
        return ap.length();
    }
    let t = (ap.dot(ab) / ab_len_sq).clamp(0.0, 1.0);
    let proj = a + ab * t;
    (p - proj).length()
}

fn clip_link_at_earth(
    rx1: f64,
    ry1: f64,
    rz1: f64,
    visible1: bool,
    rx2: f64,
    ry2: f64,
    rz2: f64,
    visible2: bool,
    earth_r_sq: f64,
) -> Option<([f64; 2], [f64; 2])> {
    if visible1 && visible2 {
        return Some(([rx1, ry1], [rx2, ry2]));
    }
    if !visible1 && !visible2 {
        return None;
    }
    let (vx, vy, vz, hx, hy, hz) = if visible1 {
        (rx1, ry1, rz1, rx2, ry2, rz2)
    } else {
        (rx2, ry2, rz2, rx1, ry1, rz1)
    };
    let dx = hx - vx;
    let dy = hy - vy;
    let dz = hz - vz;
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    for _ in 0..20 {
        let mid = (lo + hi) * 0.5;
        let mx = vx + mid * dx;
        let my = vy + mid * dy;
        let mz = vz + mid * dz;
        if mz >= 0.0 || (mx * mx + my * my) >= earth_r_sq {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let cx = vx + lo * dx;
    let cy = vy + lo * dy;
    if visible1 {
        Some(([rx1, ry1], [cx, cy]))
    } else {
        Some(([cx, cy], [rx2, ry2]))
    }
}

pub fn draw_satellite_camera(
    ui: &mut egui::Ui,
    camera_id: usize,
    lat: f64,
    lon: f64,
    altitude_km: f64,
    coverage_angle: f64,
    earth_texture: &EarthTexture,
    planet_radius: f64,
    heading_rad: f64,
) {
    let size = ui.available_size();
    let img_size = size.x.min(size.y - 40.0) as usize;
    if img_size < 10 {
        return;
    }

    let lat_rad = lat.to_radians();
    let lon_rad = lon.to_radians();
    let cone_half_angle = (coverage_angle / 2.0).to_radians();
    let orbit_radius = planet_radius + altitude_km;
    let max_earth_angle = (planet_radius / orbit_radius).acos();
    let sin_beta = orbit_radius * cone_half_angle.sin() / planet_radius;
    let angular_radius = if sin_beta >= 1.0 {
        max_earth_angle
    } else {
        (sin_beta.asin() - cone_half_angle).min(max_earth_angle)
    };

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
            let azimuth = -heading_rad - nx.atan2(-ny);

            let clat = (lat_rad.sin() * angle_from_nadir.cos()
                + lat_rad.cos() * angle_from_nadir.sin() * (-azimuth).cos())
            .asin();
            let clon = lon_rad
                + (angle_from_nadir.sin() * (-azimuth).sin()).atan2(
                    lat_rad.cos() * angle_from_nadir.cos()
                        - lat_rad.sin() * angle_from_nadir.sin() * (-azimuth).cos(),
                );

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

pub fn wrap_index(current: usize, direction: i32, modulus: usize) -> usize {
    ((current as i32 + direction + modulus as i32) % modulus as i32) as usize
}

pub fn compute_path_direction(
    src: usize,
    dst: usize,
    modulus: usize,
    is_star: bool,
) -> (i32, usize) {
    if is_star {
        if dst >= src {
            (1, dst - src)
        } else {
            (-1, src - dst)
        }
    } else {
        let diff_fwd = (dst + modulus - src) % modulus;
        let diff_bwd = (src + modulus - dst) % modulus;
        if diff_fwd <= diff_bwd {
            (1, diff_fwd)
        } else {
            (-1, diff_bwd)
        }
    }
}

pub fn compute_manhattan_path(
    src_plane: usize,
    src_sat: usize,
    dst_plane: usize,
    dst_sat: usize,
    num_planes: usize,
    sats_per_plane: usize,
    is_star: bool,
    positions: &[SatelliteState],
) -> Vec<(usize, usize)> {
    let (plane_dir, plane_steps) =
        compute_path_direction(src_plane, dst_plane, num_planes, is_star);
    let (sat_dir, sat_steps) = compute_path_direction(src_sat, dst_sat, sats_per_plane, false);

    let build_path = |planes_first: bool| -> Vec<(usize, usize)> {
        let mut p = vec![(src_plane, src_sat)];
        if planes_first {
            let mut cur = src_plane;
            for _ in 0..plane_steps {
                cur = wrap_index(cur, plane_dir, num_planes);
                p.push((cur, src_sat));
            }
            let mut cur = src_sat;
            for _ in 0..sat_steps {
                cur = wrap_index(cur, sat_dir, sats_per_plane);
                p.push((dst_plane, cur));
            }
        } else {
            let mut cur = src_sat;
            for _ in 0..sat_steps {
                cur = wrap_index(cur, sat_dir, sats_per_plane);
                p.push((src_plane, cur));
            }
            let mut cur = src_plane;
            for _ in 0..plane_steps {
                cur = wrap_index(cur, plane_dir, num_planes);
                p.push((cur, dst_sat));
            }
        }
        p
    };

    let path_distance = |path: &[(usize, usize)]| -> f64 {
        path.windows(2)
            .map(|w| {
                let a = positions
                    .iter()
                    .find(|s| s.plane == w[0].0 && s.sat_index == w[0].1);
                let b = positions
                    .iter()
                    .find(|s| s.plane == w[1].0 && s.sat_index == w[1].1);
                match (a, b) {
                    (Some(a), Some(b)) => {
                        let dx = a.x - b.x;
                        let dy = a.y - b.y;
                        let dz = a.z - b.z;
                        (dx * dx + dy * dy + dz * dz).sqrt()
                    }
                    _ => 0.0,
                }
            })
            .sum()
    };

    let planes_first = build_path(true);
    let sats_first = build_path(false);
    let d1 = path_distance(&planes_first);
    let d2 = path_distance(&sats_first);
    if d1 >= d2 {
        planes_first
    } else {
        sats_first
    }
}

pub fn compute_shortest_path_graph(
    src_idx: usize,
    dst_idx: usize,
    positions: &[SatelliteState],
) -> Vec<(usize, usize)> {
    use std::collections::BinaryHeap;

    let n = positions.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, sat) in positions.iter().enumerate() {
        for &j in &sat.neighbors {
            if j < n {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }

    let mut cost = vec![f64::INFINITY; n];
    let mut prev = vec![usize::MAX; n];
    cost[src_idx] = 0.0;

    #[derive(PartialEq)]
    struct State(f64, usize);
    impl Eq for State {}
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for State {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            other
                .0
                .partial_cmp(&self.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    }

    let mut heap: BinaryHeap<State> = BinaryHeap::new();
    heap.push(State(0.0, src_idx));

    while let Some(State(c, u)) = heap.pop() {
        if u == dst_idx {
            break;
        }
        if c > cost[u] {
            continue;
        }
        let (ux, uy, uz) = (positions[u].x, positions[u].y, positions[u].z);
        for &v in &adj[u] {
            let (vx, vy, vz) = (positions[v].x, positions[v].y, positions[v].z);
            let dx = ux - vx;
            let dy = uy - vy;
            let dz = uz - vz;
            let new_cost = cost[u] + (dx * dx + dy * dy + dz * dz).sqrt();
            if new_cost < cost[v] {
                cost[v] = new_cost;
                prev[v] = u;
                heap.push(State(new_cost, v));
            }
        }
    }

    let mut path = Vec::new();
    let mut cur = dst_idx;
    while cur != usize::MAX {
        path.push((positions[cur].plane, positions[cur].sat_index));
        if cur == src_idx {
            break;
        }
        cur = prev[cur];
    }
    path.reverse();
    path
}

pub fn build_bidirectional_adj(positions: &[SatelliteState]) -> Vec<Vec<usize>> {
    let n = positions.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, sat) in positions.iter().enumerate() {
        for &j in &sat.neighbors {
            if j < n {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    adj
}

pub fn graph_hop_count(src_idx: usize, dst_idx: usize, adj: &[Vec<usize>]) -> usize {
    use std::collections::VecDeque;

    if src_idx == dst_idx {
        return 0;
    }
    let n = adj.len();
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();
    visited[src_idx] = true;
    queue.push_back((src_idx, 0usize));
    while let Some((u, dist)) = queue.pop_front() {
        for &v in &adj[u] {
            if v == dst_idx {
                return dist + 1;
            }
            if !visited[v] {
                visited[v] = true;
                queue.push_back((v, dist + 1));
            }
        }
    }
    usize::MAX
}

pub fn compute_shortest_path(
    src_plane: usize,
    src_sat: usize,
    dst_plane: usize,
    dst_sat: usize,
    num_planes: usize,
    sats_per_plane: usize,
    positions: &[SatelliteState],
    is_star: bool,
) -> Vec<(usize, usize)> {
    use std::collections::BinaryHeap;

    let n = num_planes * sats_per_plane;
    let node = |p: usize, s: usize| p * sats_per_plane + s;
    let src = node(src_plane, src_sat);
    let dst = node(dst_plane, dst_sat);

    let get_pos = |plane: usize, sat_idx: usize| -> Option<(f64, f64, f64)> {
        positions
            .iter()
            .find(|s| s.plane == plane && s.sat_index == sat_idx)
            .map(|s| (s.x, s.y, s.z))
    };

    let dist3d = |a: (f64, f64, f64), b: (f64, f64, f64)| -> f64 {
        let dx = a.0 - b.0;
        let dy = a.1 - b.1;
        let dz = a.2 - b.2;
        (dx * dx + dy * dy + dz * dz).sqrt()
    };

    let mut cost = vec![f64::INFINITY; n];
    let mut prev = vec![usize::MAX; n];
    cost[src] = 0.0;

    #[derive(PartialEq)]
    struct State(f64, usize);
    impl Eq for State {}
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for State {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            other
                .0
                .partial_cmp(&self.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    }

    let mut heap: BinaryHeap<State> = BinaryHeap::new();
    heap.push(State(0.0, src));

    while let Some(State(c, u)) = heap.pop() {
        if u == dst {
            break;
        }
        if c > cost[u] {
            continue;
        }

        let up = u / sats_per_plane;
        let us = u % sats_per_plane;
        let cur_pos = match get_pos(up, us) {
            Some(p) => p,
            None => continue,
        };

        let mut neighbors = Vec::with_capacity(4);
        neighbors.push((up, (us + 1) % sats_per_plane));
        neighbors.push((up, (us + sats_per_plane - 1) % sats_per_plane));
        let fwd_plane = (up + 1) % num_planes;
        let bck_plane = (up + num_planes - 1) % num_planes;
        if !is_star || fwd_plane != 0 || up != num_planes - 1 {
            neighbors.push((fwd_plane, us));
        }
        if !is_star || up != 0 || bck_plane != num_planes - 1 {
            neighbors.push((bck_plane, us));
        }

        for (np, ns) in neighbors {
            let v = node(np, ns);
            if let Some(npos) = get_pos(np, ns) {
                let new_cost = cost[u] + dist3d(cur_pos, npos);
                if new_cost < cost[v] {
                    cost[v] = new_cost;
                    prev[v] = u;
                    heap.push(State(new_cost, v));
                }
            }
        }
    }

    let mut path = Vec::new();
    let mut cur = dst;
    while cur != usize::MAX {
        let p = cur / sats_per_plane;
        let s = cur % sats_per_plane;
        path.push((p, s));
        if cur == src {
            break;
        }
        cur = prev[cur];
    }
    path.reverse();
    path
}

pub fn compute_radiation_path(
    src_plane: usize,
    src_sat: usize,
    dst_plane: usize,
    dst_sat: usize,
    num_planes: usize,
    sats_per_plane: usize,
    positions: &[SatelliteState],
    is_star: bool,
    body_rotation: &Matrix3<f64>,
    igrf_rad_cache: Option<&crate::igrf::IgrfRadGrid>,
    rad_weight: f64,
) -> Vec<(usize, usize)> {
    use std::collections::BinaryHeap;

    let n = num_planes * sats_per_plane;
    let node = |p: usize, s: usize| p * sats_per_plane + s;
    let src = node(src_plane, src_sat);
    let dst = node(dst_plane, dst_sat);

    let inv_rot = body_rotation.transpose();
    let mut rad_vals = vec![0.0_f64; n];
    if let Some(grid) = igrf_rad_cache {
        for sat in positions.iter() {
            let bp = inv_rot * Vector3::new(sat.x, sat.y, sat.z);
            let r = (bp.x * bp.x + bp.y * bp.y + bp.z * bp.z).sqrt();
            let colat = (bp.y / r).acos();
            let elon = (-bp.z).atan2(bp.x);
            let (p, e) = grid.lookup(colat, elon);
            let idx = node(sat.plane, sat.sat_index);
            if idx < n {
                rad_vals[idx] = p.max(e);
            }
        }
    }

    let get_pos = |plane: usize, sat_idx: usize| -> Option<(f64, f64, f64)> {
        positions
            .iter()
            .find(|s| s.plane == plane && s.sat_index == sat_idx)
            .map(|s| (s.x, s.y, s.z))
    };

    let dist3d = |a: (f64, f64, f64), b: (f64, f64, f64)| -> f64 {
        let dx = a.0 - b.0;
        let dy = a.1 - b.1;
        let dz = a.2 - b.2;
        (dx * dx + dy * dy + dz * dz).sqrt()
    };

    let mut cost = vec![f64::INFINITY; n];
    let mut prev = vec![usize::MAX; n];
    cost[src] = 0.0;

    #[derive(PartialEq)]
    struct State(f64, usize);
    impl Eq for State {}
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for State {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            other
                .0
                .partial_cmp(&self.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    }

    let mut heap: BinaryHeap<State> = BinaryHeap::new();
    heap.push(State(0.0, src));

    while let Some(State(c, u)) = heap.pop() {
        if u == dst {
            break;
        }
        if c > cost[u] {
            continue;
        }

        let up = u / sats_per_plane;
        let us = u % sats_per_plane;
        let cur_pos = match get_pos(up, us) {
            Some(p) => p,
            None => continue,
        };

        let mut neighbors = Vec::with_capacity(4);
        neighbors.push((up, (us + 1) % sats_per_plane));
        neighbors.push((up, (us + sats_per_plane - 1) % sats_per_plane));
        let fwd_plane = (up + 1) % num_planes;
        let bck_plane = (up + num_planes - 1) % num_planes;
        if !is_star || fwd_plane != 0 || up != num_planes - 1 {
            neighbors.push((fwd_plane, us));
        }
        if !is_star || up != 0 || bck_plane != num_planes - 1 {
            neighbors.push((bck_plane, us));
        }

        for (np, ns) in neighbors {
            let v = node(np, ns);
            if let Some(npos) = get_pos(np, ns) {
                let d = dist3d(cur_pos, npos);
                let rad = rad_vals[u].max(rad_vals[v]);
                let edge_cost = d * (1.0 + rad_weight * rad);
                let new_cost = cost[u] + edge_cost;
                if new_cost < cost[v] {
                    cost[v] = new_cost;
                    prev[v] = u;
                    heap.push(State(new_cost, v));
                }
            }
        }
    }

    let mut path = Vec::new();
    let mut cur = dst;
    while cur != usize::MAX {
        let p = cur / sats_per_plane;
        let s = cur % sats_per_plane;
        path.push((p, s));
        if cur == src {
            break;
        }
        cur = prev[cur];
    }
    path.reverse();
    path
}

pub fn draw_routing_path(
    plot_ui: &mut egui_plot::PlotUi,
    path: &[(usize, usize)],
    positions: &[SatelliteState],
    rotation: &Matrix3<f64>,
    color: egui::Color32,
    width: f32,
    hide_behind_earth: bool,
    earth_r_sq: f64,
    show_distance: bool,
    path_distance_labels: &mut Vec<([f64; 2], String)>,
) {
    if path.len() < 2 {
        return;
    }

    for i in 0..(path.len() - 1) {
        let (plane1, sat1) = path[i];
        let (plane2, sat2) = path[i + 1];

        let pos1 = positions
            .iter()
            .find(|s| s.plane == plane1 && s.sat_index == sat1);
        let pos2 = positions
            .iter()
            .find(|s| s.plane == plane2 && s.sat_index == sat2);

        if let (Some(p1), Some(p2)) = (pos1, pos2) {
            let (rx1, ry1, rz1) = rotate_point_matrix(p1.x, p1.y, p1.z, rotation);
            let (rx2, ry2, rz2) = rotate_point_matrix(p2.x, p2.y, p2.z, rotation);

            let visible1 = rz1 >= 0.0 || (rx1 * rx1 + ry1 * ry1) >= earth_r_sq;
            let visible2 = rz2 >= 0.0 || (rx2 * rx2 + ry2 * ry2) >= earth_r_sq;

            if hide_behind_earth {
                if let Some((p1, p2)) =
                    clip_link_at_earth(rx1, ry1, rz1, visible1, rx2, ry2, rz2, visible2, earth_r_sq)
                {
                    plot_ui.line(
                        Line::new("", PlotPoints::new(vec![p1, p2]))
                            .color(color)
                            .width(width),
                    );
                }
            } else {
                let line_color = if visible1 && visible2 {
                    color
                } else {
                    egui::Color32::from_rgba_unmultiplied(
                        color.r() / 2,
                        color.g() / 2,
                        color.b() / 2,
                        150,
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

    if show_distance {
        if let Some(label) = path_distance_label(path, positions, rotation) {
            path_distance_labels.push(label);
        }
    }
}

fn path_distance_label(
    path: &[(usize, usize)],
    positions: &[SatelliteState],
    rotation: &Matrix3<f64>,
) -> Option<([f64; 2], String)> {
    let Some((distance_km, [x, y])) = path_distance_and_midpoint(path, positions, rotation) else {
        return None;
    };
    let label = format!("{:.0} km", distance_km);
    Some(([x, y], label))
}

fn path_distance_and_midpoint(
    path: &[(usize, usize)],
    positions: &[SatelliteState],
    rotation: &Matrix3<f64>,
) -> Option<(f64, [f64; 2])> {
    let mut segments: Vec<(f64, [f64; 2], [f64; 2])> = Vec::new();
    let mut total = 0.0;
    for w in path.windows(2) {
        let a = positions
            .iter()
            .find(|s| s.plane == w[0].0 && s.sat_index == w[0].1)?;
        let b = positions
            .iter()
            .find(|s| s.plane == w[1].0 && s.sat_index == w[1].1)?;
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let dz = b.z - a.z;
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len <= 1e-9 {
            continue;
        }
        let (ax, ay, _) = rotate_point_matrix(a.x, a.y, a.z, rotation);
        let (bx, by, _) = rotate_point_matrix(b.x, b.y, b.z, rotation);
        segments.push((len, [ax, ay], [bx, by]));
        total += len;
    }
    if total <= 1e-9 {
        return None;
    }
    let mut remaining = total * 0.5;
    for (len, a, b) in segments {
        if remaining <= len {
            let t = remaining / len;
            return Some((total, [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]));
        }
        remaining -= len;
    }
    None
}

pub fn draw_3d_view(
    ui: &mut egui::Ui,
    id: &str,
    constellations: &[(
        WalkerConstellation,
        Vec<SatelliteState>,
        usize,
        u8,
        usize,
        String,
    )],
    flags: View3DFlags,
    coverage_angle: f64,
    mut rotation: Matrix3<f64>,
    satellite_rotation: Matrix3<f64>,
    width: f32,
    height: f32,
    earth_texture: Option<&egui::TextureHandle>,
    mut zoom: f64,
    sat_radius: f32,
    link_width: f32,
    pending_cameras: &mut Vec<SatelliteCamera>,
    camera_id_counter: &mut usize,
    satellite_cameras: &mut [SatelliteCamera],
    cameras_to_remove: &mut Vec<usize>,
    pinned_isls: &mut HashSet<crate::config::PinnedIsl>,
    planet_radius: f64,
    flattening: f64,
    gpu_available: bool,
    body_key: (CelestialBody, Skin, TextureResolution),
    body_rotation: &Matrix3<f64>,
    sun_dir: [f32; 3],
    time: f64,
    ground_stations: &mut [GroundStation],
    ground_stations_locked: bool,
    areas_of_interest: &mut [AreaOfInterest],
    device_layers: &[DeviceLayer],
    body_rot_angle: f64,
    dragging_place: &mut Option<(usize, usize, bool, usize)>,
    drag_tab_planet: (usize, usize),
    detail_bounds: Option<[f32; 4]>,
    geo_borders: &[Vec<(f64, f64)>],
    geo_cities: &[CityLabel],
    conjunction_lines: &[(ConjunctionInfo, f64)],
    conjunction_heatmap: &[(ConjunctionInfo, f64)],
    correcting_sats: &HashSet<String>,
    hit_sats: &HashSet<String>,
    radiation: Option<&RadiationConfig>,
    moon_handles: &HashMap<CelestialBody, egui::TextureHandle>,
    context_menu_request: &mut Option<(egui::Pos2, f64, f64)>,
    label_click_request: &mut Option<(bool, usize, egui::Pos2)>,
    physics_colors: &HashMap<(usize, usize), egui::Color32>,
    physics_info: &HashMap<(usize, usize), (f64, f64, bool)>,
    ground_tracks: &[Vec<(f64, f64)>],
    flash_intensities: &[HashMap<u32, f32>],
) -> (Matrix3<f64>, f64) {
    let View3DFlags {
        show_orbits,
        show_axes,
        show_magnetic_axis,
        show_coverage,
        show_links,
        show_gs_links,
        hide_behind_earth,
        single_color,
        dark_mode,
        show_routing_paths,
        show_proxy_links,
        show_path_distance,
        show_manhattan_path,
        show_shortest_path,
        show_radiation_path,
        radiation_weight,
        routing_width,
        routing_node_scale,
        show_asc_desc_colors,
        color_ascending,
        color_descending,
        color_links,
        show_sat_labels,
        show_altitude_lines,
        altitude_line_width,
        show_inclination_bounds,
        render_planet,
        fixed_sizes,
        show_sat_border,
        show_polar_circle,
        show_equator,
        show_graticule,
        show_crosshairs,
        show_terminator,
        show_eclipse,
        show_sun,
        earth_fixed_camera,
        use_gpu_rendering,
        show_clouds,
        show_day_night,
        show_city_lights,
        show_stars,
        show_borders,
        show_cities,
        trackpad_rotate,
        north_up,
        ref enabled_moons,
        moon_inclination_override,
        show_moon_orbits,
        show_moon_lines,
        show_moon_labels,
        moon_camera_distance_km,
        tle_monochrome,
        show_ground_tracks,
    } = flags;
    let max_altitude = constellations
        .iter()
        .map(|(c, _, _, _, _, _)| c.altitude_km)
        .fold(550.0_f64, |a, b| a.max(b));
    let orbit_radius = planet_radius + max_altitude;
    let has_simulated = constellations.iter().any(|(_, _, _, tk, _, _)| *tk == 0);
    let axis_len = orbit_radius * 1.05;
    let planet_view_reference = planet_radius * 1.15;
    let margin = planet_view_reference / zoom;
    let zoom_factor = if fixed_sizes { 1.0 } else { zoom as f32 };
    let scaled_sat_radius = sat_radius * zoom_factor;
    let scaled_link_width = (link_width * zoom_factor).max(0.5);
    let scaled_routing_width = (routing_width * zoom_factor).max(1.0);
    let active_sat_radius = if show_routing_paths {
        scaled_sat_radius * routing_node_scale
    } else {
        scaled_sat_radius
    };

    let use_gpu = gpu_available && render_planet && use_gpu_rendering;

    if use_gpu {
        let rect = egui::Rect::from_min_size(ui.cursor().min, egui::Vec2::new(width, height));
        let combined_rotation = if earth_fixed_camera {
            rotation
        } else {
            rotation * body_rotation
        };
        let inv_rotation = combined_rotation.transpose();
        let star_inv_rotation = if earth_fixed_camera {
            body_rotation * rotation.transpose()
        } else {
            rotation.transpose()
        };
        let flat = flattening as f32;
        let key = body_key;
        let scale = (planet_radius / margin) as f32;
        let atmosphere = match body_key.0 {
            CelestialBody::Earth if body_key.1 != crate::celestial::Skin::Abstract => 1.0_f32,
            _ => 0.0,
        };

        let bg_c = ui.visuals().extreme_bg_color;
        let bg_color = [
            bg_c.r() as f32 / 255.0,
            bg_c.g() as f32 / 255.0,
            bg_c.b() as f32 / 255.0,
        ];
        let logical_aspect = width / height.max(1.0);

        let rot_cols = mat3_to_padded_cols(&inv_rotation);
        let star_cols = mat3_to_padded_cols(&star_inv_rotation);

        let ring_params = key.0.ring_params();
        let (ring_inner, ring_outer) = ring_params.map(|(_, i, o)| (i, o)).unwrap_or((0.0, 0.0));
        let has_rings_f = if ring_params.is_some() { 1.0f32 } else { 0.0 };
        let adams = if has_rings_f > 0.5 && key.0 == CelestialBody::Neptune {
            1.0f32
        } else {
            0.0
        };
        let eps = if has_rings_f > 0.5 && key.0 == CelestialBody::Uranus {
            1.0f32
        } else {
            0.0
        };

        let has_detail = detail_bounds.is_some();
        let db = detail_bounds.unwrap_or([0.0; 4]);

        let uniforms = PlanetUniforms {
            inv_rot_0: rot_cols[0],
            inv_rot_1: rot_cols[1],
            inv_rot_2: rot_cols[2],
            star_rot_0: star_cols[0],
            star_rot_1: star_cols[1],
            star_rot_2: star_cols[2],
            sun_dir_flat: [sun_dir[0], sun_dir[1], sun_dir[2], flat],
            bg_aspect: [bg_color[0], bg_color[1], bg_color[2], logical_aspect],
            detail_bounds: db,
            uv_etc: [1.0, 1.0, scale, atmosphere],
            flags_a: [
                if show_clouds { 1.0 } else { 0.0 },
                if show_day_night { 1.0 } else { 0.0 },
                if show_city_lights { 1.0 } else { 0.0 },
                if show_stars { 1.0 } else { 0.0 },
            ],
            flags_b: [
                if has_detail { 1.0 } else { 0.0 },
                0.0,
                has_rings_f,
                ring_inner,
            ],
            flags_c: [ring_outer, adams, eps, 0.0],
        };

        let callback = egui_wgpu::Callback::new_paint_callback(
            rect,
            PlanetPaintCallback::new(uniforms, key, show_stars, has_detail),
        );
        ui.painter().add(callback);

        if let Some(rad) = radiation {
            if rad.show_heatmap_sphere {
                let dipole_tilt = rad.dipole_tilt.to_radians();
                let tilt_lon = (-287.3_f64).to_radians();
                let hm_mag_axis = Vector3::new(
                    dipole_tilt.sin() * tilt_lon.cos(),
                    dipole_tilt.cos(),
                    dipole_tilt.sin() * tilt_lon.sin(),
                )
                .normalize();
                let offset_lat = 22.0_f64.to_radians();
                let offset_lon = (-140.0_f64).to_radians();
                let sphere_r_km = planet_radius + rad.heatmap_altitude_km;
                let hm_dipole_ox = (rad.dipole_offset_km * offset_lat.cos() * offset_lon.cos()
                    / sphere_r_km) as f32;
                let hm_dipole_oy = (rad.dipole_offset_km * offset_lat.sin() / sphere_r_km) as f32;
                let hm_dipole_oz = (rad.dipole_offset_km * offset_lat.cos() * offset_lon.sin()
                    / sphere_r_km) as f32;

                let hm_mode = match rad.heatmap_mode {
                    crate::config::HeatmapMode::Radiation => 0i32,
                    crate::config::HeatmapMode::FieldStrength => 1,
                    crate::config::HeatmapMode::IgrfField => 2,
                    crate::config::HeatmapMode::IgrfRadiation => 3,
                };

                let hm_kp = rad.kp_index as f32;
                let hm_planet_r = (planet_radius / sphere_r_km) as f32;
                let hm_show_p = if rad.show_protons { 1.0f32 } else { 0.0 };
                let hm_show_e = if rad.show_electrons { 1.0f32 } else { 0.0 };
                let hm_smooth = rad.smooth_colors;
                let hm_mag_f = [
                    hm_mag_axis.x as f32,
                    hm_mag_axis.y as f32,
                    hm_mag_axis.z as f32,
                ];
                let hm_dipole_f = [hm_dipole_ox, hm_dipole_oy, hm_dipole_oz];
                let hm_sphere_r_km = sphere_r_km as f32;
                let hm_scale = (sphere_r_km / margin) as f32;
                let (hm_rad_data, hm_compute_rad): (Option<(u64, Vec<u8>)>, Option<f32>) =
                    if rad.heatmap_mode == crate::config::HeatmapMode::IgrfRadiation {
                        (None, Some(hm_sphere_r_km))
                    } else {
                        (None, None)
                    };
                let hm_inv_rotation = inv_rotation;
                let hm_rot_cols = mat3_to_padded_cols(&hm_inv_rotation);
                let hm_uniforms = HeatmapUniforms {
                    inv_rot_0: hm_rot_cols[0],
                    inv_rot_1: hm_rot_cols[1],
                    inv_rot_2: hm_rot_cols[2],
                    mag_aspect: [hm_mag_f[0], hm_mag_f[1], hm_mag_f[2], logical_aspect],
                    dipole_scale: [hm_dipole_f[0], hm_dipole_f[1], hm_dipole_f[2], hm_scale],
                    uv_mode_smooth: [1.0, 1.0, hm_mode as f32, if hm_smooth { 1.0 } else { 0.0 }],
                    kp_pr_sr_sp: [hm_kp, hm_planet_r, hm_sphere_r_km, hm_show_p],
                    se_pad: [hm_show_e, 0.0, 0.0, 0.0],
                };

                let hm_callback = egui_wgpu::Callback::new_paint_callback(
                    rect,
                    HeatmapPaintCallback::new(hm_uniforms, hm_rad_data, hm_compute_rad),
                );
                ui.painter().add(hm_callback);
            }
        }

        if show_sun {
            let sun_v =
                Vector3::new(sun_dir[0] as f64, sun_dir[1] as f64, sun_dir[2] as f64).normalize();
            let sun_rot = if earth_fixed_camera {
                rotation
            } else {
                rotation * *body_rotation
            };
            let (sx, sy, sz) = rotate_point_matrix(sun_v.x, sun_v.y, sun_v.z, &sun_rot);
            if sz <= 0.0 {
                let sun_pos_uv = [(sx * 1.3) as f32, (sy * 1.3) as f32];
                let sun_dist_au = body_key.0.semi_major_axis_au().unwrap_or(1.0) as f32;
                let sun_intensity = (1.0 / (sun_dist_au * sun_dist_au)).min(2.0);
                let zoom_dil = (1.0 / zoom as f32).sqrt();
                let cam_ratio = (moon_camera_distance_km / 1_000_000.0) as f32;

                let sun_uniforms = SunUniforms {
                    uv_aspect_ps: [1.0, 1.0, logical_aspect, scale],
                    sun_cam_int: [sun_pos_uv[0], sun_pos_uv[1], cam_ratio, sun_intensity],
                    zoom_pad: [zoom_dil, 0.0, 0.0, 0.0],
                };

                let sun_callback = egui_wgpu::Callback::new_paint_callback(
                    rect,
                    SunPaintCallback::new(sun_uniforms),
                );
                ui.painter().add(sun_callback);
            }
        }
    }

    let plot = Plot::new(id)
        .data_aspect(1.0)
        .width(width)
        .height(height)
        .show_axes(false)
        .show_grid(false)
        .show_x(false)
        .show_y(false)
        .show_background(!gpu_available || !use_gpu_rendering)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .cursor_color(egui::Color32::TRANSPARENT);

    let egui_ctx = ui.ctx().clone();
    let mut surface_labels: Vec<([f64; 2], String, egui::Color32, bool, usize)> = Vec::new();
    let mut device_cluster_labels: Vec<([f64; 2], usize, egui::Color32)> = Vec::new();
    let mut spacecomp_role_labels: Vec<([f64; 2], &'static str, egui::Color32)> = Vec::new();
    let mut path_distance_labels: Vec<([f64; 2], String)> = Vec::new();
    let mut hover_isl_segments: Vec<(
        [f64; 2],
        [f64; 2],
        f64,
        crate::config::LinkBudget,
        crate::config::PinnedIsl,
    )> = Vec::new();
    let mut pinned_isl_overlays: Vec<([f64; 2], [f64; 2])> = Vec::new();

    let response = plot.show(ui, |plot_ui| {
        let ground_stations = &*ground_stations;
        let areas_of_interest = &*areas_of_interest;
        plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
            [-margin, -margin],
            [margin, margin],
        ));

        let visual_earth_r = planet_radius;
        let earth_r_sq = visual_earth_r * visual_earth_r;

        if show_orbits && !hide_behind_earth {
            for (constellation, _, color_offset, tle_kind, _, _) in constellations {
                if *tle_kind != 0 {
                    continue;
                }
                for plane in 0..constellation.num_planes {
                    let orbit_pts = constellation.orbit_points_3d(plane, time);
                    let color = plane_color(if single_color {
                        *color_offset
                    } else {
                        plane + color_offset
                    });

                    let mut behind_segment: Vec<[f64; 2]> = Vec::new();
                    for &(x, y, z) in &orbit_pts {
                        let (rx, ry, rz) = rotate_point_matrix(x, y, z, &satellite_rotation);
                        let occluded = rz < 0.0 && (rx * rx + ry * ry) < earth_r_sq;
                        if occluded {
                            behind_segment.push([rx, ry]);
                        } else if !behind_segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut behind_segment)))
                                    .color(dim_color(color))
                                    .width(scaled_link_width),
                            );
                        }
                    }
                    if !behind_segment.is_empty() {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(behind_segment))
                                .color(dim_color(color))
                                .width(scaled_link_width),
                        );
                    }
                }
            }
        }

        if !hide_behind_earth {
            for (constellation, positions, color_offset, _tle_kind, _, _) in constellations {
                for plane in 0..constellation.num_planes {
                    let color = plane_color(if single_color {
                        *color_offset
                    } else {
                        plane + color_offset
                    });
                    let pts: PlotPoints = positions
                        .iter()
                        .filter_map(|s| {
                            if s.plane != plane {
                                return None;
                            }
                            let (rx, ry, rz) =
                                rotate_point_matrix(s.x, s.y, s.z, &satellite_rotation);
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
                            .radius(scaled_sat_radius * 0.8)
                            .filled(true),
                    );
                }
            }
        }

        if show_sun && !use_gpu {
            let sun =
                Vector3::new(sun_dir[0] as f64, sun_dir[1] as f64, sun_dir[2] as f64).normalize();
            let sun_rot = if earth_fixed_camera {
                rotation
            } else {
                rotation * *body_rotation
            };
            let dist = margin * 1.3;
            let (sx, sy, sz) =
                rotate_point_matrix(sun.x * dist, sun.y * dist, sun.z * dist, &sun_rot);
            if sz <= 0.0 {
                let clip_r = visual_earth_r * 1.01;
                let pr_sq = clip_r * clip_r;
                let sun_dist_au = body_key.0.semi_major_axis_au().unwrap_or(1.0);
                let max_glow = planet_radius * 0.2 / sun_dist_au;
                let intensity = 1.0 / (sun_dist_au * sun_dist_au);
                let zoom_dilution = (1.0 / zoom).sqrt();
                let num_rings = 40usize;
                let ring_width = (max_glow / num_rings as f64 * 1.8) as f32 * zoom_factor;
                let n = 72;
                for ring in 0..num_rings {
                    let t = ring as f64 / (num_rings - 1) as f64;
                    let r = max_glow * (1.0 - t * t);
                    let brightness = t * t * t;
                    let base_alpha = 8.0 + 247.0 * brightness;
                    let outer_fade = if brightness < 0.5 { zoom_dilution } else { 1.0 };
                    let alpha =
                        (base_alpha * intensity.min(1.0) * outer_fade).clamp(1.0, 255.0) as u8;
                    let g = (200.0 + 55.0 * brightness) as u8;
                    let b = (80.0 + 170.0 * brightness) as u8;
                    let color = egui::Color32::from_rgba_unmultiplied(255, g, b, alpha);
                    let mut seg: Vec<[f64; 2]> = Vec::new();
                    for i in 0..=n {
                        let theta = 2.0 * PI * i as f64 / n as f64;
                        let px = sx + r * theta.cos();
                        let py = sy + r * theta.sin();
                        if px * px + py * py >= pr_sq {
                            seg.push([px, py]);
                        } else {
                            if seg.len() >= 2 {
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(std::mem::take(&mut seg)))
                                        .color(color)
                                        .width(ring_width),
                                );
                            }
                            seg.clear();
                        }
                    }
                    if seg.len() >= 2 {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(seg))
                                .color(color)
                                .width(ring_width),
                        );
                    }
                }
                let ray_len = max_glow * 2.5;
                let ray_alpha = (40.0 * intensity.min(1.0) * zoom_dilution).clamp(1.0, 255.0) as u8;
                let ray_color = egui::Color32::from_rgba_unmultiplied(255, 240, 200, ray_alpha);
                let spike_w = 2.0_f32;
                for spike in 0..4 {
                    let angle = PI / 4.0 * spike as f64;
                    for &dir in &[1.0_f64, -1.0] {
                        let mut pts = Vec::new();
                        let steps = 30;
                        for i in 0..=steps {
                            let frac = i as f64 / steps as f64;
                            let px = sx + dir * ray_len * frac * angle.cos();
                            let py = sy + dir * ray_len * frac * angle.sin();
                            if px * px + py * py >= pr_sq {
                                pts.push([px, py]);
                            } else {
                                if pts.len() >= 2 {
                                    plot_ui.line(
                                        Line::new("", PlotPoints::new(std::mem::take(&mut pts)))
                                            .color(ray_color)
                                            .width(spike_w),
                                    );
                                }
                                pts.clear();
                            }
                        }
                        if pts.len() >= 2 {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(pts))
                                    .color(ray_color)
                                    .width(spike_w),
                            );
                        }
                    }
                }
            }
        }

        if render_planet {
            if use_gpu {
                // GPU rendering is handled by paint callback before the plot
            } else if let Some(tex) = earth_texture {
                let size = egui::Vec2::splat(planet_radius as f32 * 2.0);
                plot_ui.image(PlotImage::new("", tex, PlotPoint::new(0.0, 0.0), size));
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
                        .stroke(egui::Stroke::new(
                            2.0,
                            egui::Color32::from_rgb(70, 130, 180),
                        )),
                );
            }

            if dark_mode && !use_gpu {
                let equatorial_r = planet_radius;
                let polar_r = planet_radius * (1.0 - flattening);
                let border_pts: PlotPoints = (0..=100)
                    .map(|i| {
                        let theta = 2.0 * PI * i as f64 / 100.0;
                        [equatorial_r * theta.cos(), polar_r * theta.sin()]
                    })
                    .collect();
                plot_ui.line(
                    Line::new("", border_pts)
                        .color(egui::Color32::WHITE)
                        .width(1.0),
                );
            }

            if show_polar_circle {
                let polar_r = planet_radius * (1.0 - flattening);
                let circle_pts: PlotPoints = (0..=100)
                    .map(|i| {
                        let theta = 2.0 * PI * i as f64 / 100.0;
                        [polar_r * theta.cos(), polar_r * theta.sin()]
                    })
                    .collect();
                plot_ui.line(
                    Line::new("", circle_pts)
                        .color(egui::Color32::from_rgb(255, 200, 50))
                        .width(1.0),
                );
            }

            if show_terminator {
                let sun = Vector3::new(sun_dir[0] as f64, sun_dir[1] as f64, sun_dir[2] as f64)
                    .normalize();
                let up = if sun.y.abs() > 0.9 {
                    Vector3::new(1.0, 0.0, 0.0)
                } else {
                    Vector3::new(0.0, 1.0, 0.0)
                };
                let u = sun.cross(&up).normalize();
                let v = sun.cross(&u).normalize();
                let term_rotation = if earth_fixed_camera {
                    rotation
                } else {
                    rotation * *body_rotation
                };

                // Build the terminator great circle as (x, y, is_front) samples,
                // then split into contiguous front/back segments so we can render
                // the hidden half dimmer (or skip it entirely when
                // hide_behind_earth is set).
                const TERM_SEGMENTS: usize = 200;
                let pts: Vec<(f64, f64, bool)> = (0..TERM_SEGMENTS)
                    .map(|i| {
                        let theta = 2.0 * PI * i as f64 / TERM_SEGMENTS as f64;
                        let x = planet_radius * (u.x * theta.cos() + v.x * theta.sin());
                        let y = planet_radius * (u.y * theta.cos() + v.y * theta.sin());
                        let z = planet_radius * (u.z * theta.cos() + v.z * theta.sin());
                        let (sx, sy, sz) = rotate_point_matrix(x, y, z, &term_rotation);
                        (sx, sy, sz >= 0.0)
                    })
                    .collect();

                let bright = egui::Color32::from_rgb(255, 180, 0);
                let dim = egui::Color32::from_rgba_unmultiplied(255, 180, 0, 70);

                let mut seg: Vec<[f64; 2]> = Vec::new();
                let mut seg_front = pts[0].2;
                let flush =
                    |plot_ui: &mut egui_plot::PlotUi, seg: &mut Vec<[f64; 2]>, front: bool| {
                        if seg.len() >= 2 {
                            let color = if front {
                                bright
                            } else if !hide_behind_earth {
                                dim
                            } else {
                                return;
                            };
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(seg)))
                                    .color(color)
                                    .width(2.0),
                            );
                        } else {
                            seg.clear();
                        }
                    };
                // Iterate through the loop including a wrap back to index 0 so
                // the final segment closes cleanly.
                for i in 0..=TERM_SEGMENTS {
                    let (sx, sy, in_front) = pts[i % TERM_SEGMENTS];
                    if in_front != seg_front && !seg.is_empty() {
                        // Boundary crossing — emit the accumulated segment and
                        // start a new one. Append the boundary point to both
                        // sides so segments meet at the earth limb.
                        seg.push([sx, sy]);
                        flush(plot_ui, &mut seg, seg_front);
                        seg_front = in_front;
                    }
                    seg.push([sx, sy]);
                }
                flush(plot_ui, &mut seg, seg_front);
            }
        }

        if let Some(rad_config) = radiation {
            use crate::radiation::belt_profile_r;

            let belt_rotation = if earth_fixed_camera {
                rotation
            } else {
                rotation * *body_rotation
            };

            let dipole_tilt = rad_config.dipole_tilt.to_radians();
            let tilt_lon = (-287.3_f64).to_radians();
            let mag_axis = Vector3::new(
                dipole_tilt.sin() * tilt_lon.cos(),
                dipole_tilt.cos(),
                dipole_tilt.sin() * tilt_lon.sin(),
            )
            .normalize();
            let mag_u = if mag_axis.x.abs() < 0.9 {
                Vector3::new(1.0, 0.0, 0.0).cross(&mag_axis).normalize()
            } else {
                Vector3::new(0.0, 0.0, 1.0).cross(&mag_axis).normalize()
            };
            let mag_v = mag_axis.cross(&mag_u);

            let offset_lat = 22.0_f64.to_radians();
            let offset_lon = (-140.0_f64).to_radians();
            let dipole_ox = rad_config.dipole_offset_km * offset_lat.cos() * offset_lon.cos();
            let dipole_oy = rad_config.dipole_offset_km * offset_lat.sin();
            let dipole_oz = rad_config.dipole_offset_km * offset_lat.cos() * offset_lon.sin();

            if rad_config.show_heatmap_sphere && !use_gpu {
                let sphere_r = planet_radius + rad_config.heatmap_altitude_km;
                let sphere_r_sq = sphere_r * sphere_r;
                let ma = mag_axis;
                let inv_rot = belt_rotation.transpose();
                let tex_size: usize = rad_config.heatmap_resolution.max(12) as usize * 16;

                let texel_size = 2.0 * sphere_r / tex_size as f64;
                let aa_width = texel_size * 3.0;
                let igrf_grid = rad_config.igrf_grid_cache.as_ref().map(|(_, g)| g);
                let igrf_rad_grid = rad_config.igrf_rad_cache.as_ref().map(|(_, _, g)| g);
                let mut pixels = Vec::with_capacity(tex_size * tex_size);
                for ty in 0..tex_size {
                    let py = sphere_r * (1.0 - 2.0 * ty as f64 / (tex_size - 1) as f64);
                    for tx in 0..tex_size {
                        let px = sphere_r * (-1.0 + 2.0 * tx as f64 / (tex_size - 1) as f64);
                        let d = (px * px + py * py).sqrt();
                        let edge_alpha = ((sphere_r - d) / aa_width + 0.5).clamp(0.0, 1.0);
                        if edge_alpha <= 0.0 {
                            pixels.push(egui::Color32::TRANSPARENT);
                            continue;
                        }
                        let pz = (sphere_r_sq - (d * d).min(sphere_r_sq)).sqrt();
                        let gp = inv_rot * Vector3::new(px, py, pz);
                        let dx = gp.x - dipole_ox;
                        let dy = gp.y - dipole_oy;
                        let dz = gp.z - dipole_oz;
                        let r_d = (dx * dx + dy * dy + dz * dz).sqrt();
                        let mag_dot = dx * ma.x + dy * ma.y + dz * ma.z;
                        let sin_ml = mag_dot / r_d;
                        let r_d_er = r_d / planet_radius;
                        let alpha = (180.0 * edge_alpha) as u8;
                        let smooth = rad_config.smooth_colors;
                        let c = if rad_config.heatmap_mode
                            == crate::config::HeatmapMode::IgrfRadiation
                        {
                            let colat =
                                (gp.y / (gp.x * gp.x + gp.y * gp.y + gp.z * gp.z).sqrt()).acos();
                            let elon = (-gp.z).atan2(gp.x);
                            let (p, e) = igrf_rad_grid.unwrap().lookup(colat, elon);
                            blend_proton_electron(
                                p,
                                e,
                                rad_config.show_protons,
                                rad_config.show_electrons,
                                smooth,
                            )
                        } else {
                            let intensity = match rad_config.heatmap_mode {
                                crate::config::HeatmapMode::Radiation => {
                                    let r_c = (gp.x * gp.x + gp.y * gp.y + gp.z * gp.z).sqrt();
                                    let saa_factor = (r_d / r_c).powi(12);
                                    let cos_ml_sq = 1.0 - sin_ml * sin_ml;
                                    let l = if cos_ml_sq > 1e-6 {
                                        r_d_er / cos_ml_sq
                                    } else {
                                        r_d_er * 1e6
                                    };
                                    (belt_profile_r(l, rad_config.kp_index) * saa_factor)
                                        .clamp(0.0, 1.0)
                                }
                                crate::config::HeatmapMode::FieldStrength => {
                                    let b0 = 30115.0;
                                    let f =
                                        b0 / r_d_er.powi(3) * (1.0 + 3.0 * sin_ml * sin_ml).sqrt();
                                    normalize_field_nt(f, r_d_er)
                                }
                                crate::config::HeatmapMode::IgrfField => {
                                    let r_km = (gp.x * gp.x + gp.y * gp.y + gp.z * gp.z).sqrt();
                                    let colat = (gp.y / r_km).acos();
                                    let elon = (-gp.z).atan2(gp.x);
                                    let g = igrf_grid.unwrap();
                                    g.normalize(g.lookup(colat, elon))
                                }
                                _ => 0.0,
                            };
                            crate::config::heatmap_color(intensity, smooth)
                        };
                        pixels.push(egui::Color32::from_rgba_unmultiplied(
                            c.r(),
                            c.g(),
                            c.b(),
                            alpha,
                        ));
                    }
                }
                let image = egui::ColorImage::new([tex_size, tex_size], pixels);
                let tex_handle =
                    egui_ctx.load_texture("rad_heatmap", image, egui::TextureOptions::LINEAR);
                plot_ui.image(PlotImage::new(
                    "",
                    tex_handle.id(),
                    PlotPoint::new(0.0, 0.0),
                    egui::Vec2::splat(2.0 * sphere_r as f32),
                ));
            }

            if rad_config.show_belts {
                let num_meridians = rad_config.num_meridians.max(1);
                let num_l = rad_config.num_shells.max(2);
                let l_min = 1.02_f64;
                let l_max = 7.0_f64;
                let num_lat = 24;
                let num_dots = rad_config.dots_per_line.max(2);

                let shell_pt3 = |l: f64, phi: f64, lam: f64| -> (f64, f64, f64) {
                    let dir = mag_u * phi.cos() + mag_v * phi.sin();
                    let r = l * lam.cos().powi(2) * planet_radius;
                    let px = mag_axis.x * r * lam.sin() + dir.x * r * lam.cos();
                    let py = mag_axis.y * r * lam.sin() + dir.y * r * lam.cos();
                    let pz = mag_axis.z * r * lam.sin() + dir.z * r * lam.cos();
                    rotate_point_matrix(px, py, pz, &belt_rotation)
                };

                let visible = |p: (f64, f64, f64)| -> bool {
                    !hide_behind_earth || p.2 >= 0.0 || (p.0 * p.0 + p.1 * p.1) >= earth_r_sq
                };

                let draw_clipped_line = |plot_ui: &mut egui_plot::PlotUi,
                                         pts: &[(f64, f64, f64)],
                                         color: egui::Color32,
                                         width: f32| {
                    if !hide_behind_earth {
                        let xy: Vec<[f64; 2]> = pts.iter().map(|p| [p.0, p.1]).collect();
                        if xy.len() >= 2 {
                            plot_ui
                                .line(Line::new("", PlotPoints::new(xy)).color(color).width(width));
                        }
                        return;
                    }
                    let mut seg: Vec<[f64; 2]> = Vec::new();
                    for i in 0..pts.len() {
                        let (rx2, ry2, rz2) = pts[i];
                        let v2 = rz2 >= 0.0 || (rx2 * rx2 + ry2 * ry2) >= earth_r_sq;
                        if i == 0 {
                            if v2 {
                                seg.push([rx2, ry2]);
                            }
                            continue;
                        }
                        let (rx1, ry1, rz1) = pts[i - 1];
                        let v1 = rz1 >= 0.0 || (rx1 * rx1 + ry1 * ry1) >= earth_r_sq;
                        if v1 && v2 {
                            if seg.is_empty() {
                                seg.push([rx1, ry1]);
                            }
                            seg.push([rx2, ry2]);
                        } else if let Some((p1, p2)) =
                            clip_link_at_earth(rx1, ry1, rz1, v1, rx2, ry2, rz2, v2, earth_r_sq)
                        {
                            if v1 {
                                if seg.is_empty() {
                                    seg.push(p1);
                                }
                                seg.push(p2);
                                if seg.len() >= 2 {
                                    plot_ui.line(
                                        Line::new("", PlotPoints::new(std::mem::take(&mut seg)))
                                            .color(color)
                                            .width(width),
                                    );
                                }
                                seg.clear();
                            } else {
                                if seg.len() >= 2 {
                                    plot_ui.line(
                                        Line::new("", PlotPoints::new(std::mem::take(&mut seg)))
                                            .color(color)
                                            .width(width),
                                    );
                                }
                                seg.clear();
                                seg.push(p1);
                                seg.push(p2);
                            }
                        } else {
                            if seg.len() >= 2 {
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(std::mem::take(&mut seg)))
                                        .color(color)
                                        .width(width),
                                );
                            }
                            seg.clear();
                        }
                    }
                    if seg.len() >= 2 {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(seg))
                                .color(color)
                                .width(width),
                        );
                    }
                };

                let shell_color = |l: f64| -> egui::Color32 {
                    let intensity = belt_profile_r(l, rad_config.kp_index);
                    let t = (0.15 + intensity * 0.85).clamp(0.15, 1.0);
                    egui::Color32::from_rgba_unmultiplied(
                        (255.0 * t) as u8,
                        (255.0 * (1.0 - t)) as u8,
                        0,
                        180,
                    )
                };

                let mut prev_dot_grid: Option<Vec<Vec<(f64, f64, f64)>>> = None;

                for li in 0..num_l {
                    let l = l_min + (l_max - l_min) * li as f64 / (num_l - 1) as f64;
                    let mirror_lat = (1.0_f64 / l.sqrt()).acos();
                    let color = shell_color(l);
                    let phi_offset = rad_config.shell_phasing * PI * li as f64 / num_l as f64;
                    let shell_num_lat = (num_lat as f64 * (l / l_min).sqrt()) as usize;

                    let mut dot_grid: Vec<Vec<(f64, f64, f64)>> = Vec::new();

                    for mi in 0..num_meridians {
                        let phi = phi_offset + 2.0 * PI * mi as f64 / num_meridians as f64;

                        if rad_config.show_lines {
                            let pts3: Vec<(f64, f64, f64)> = (0..=shell_num_lat)
                                .map(|j| {
                                    let lam = -mirror_lat
                                        + 2.0 * mirror_lat * j as f64 / shell_num_lat as f64;
                                    shell_pt3(l, phi, lam)
                                })
                                .collect();
                            draw_clipped_line(plot_ui, &pts3, color, 1.0);
                        }

                        let dots: Vec<(f64, f64, f64)> = (0..num_dots)
                            .map(|j| {
                                let lam = -mirror_lat
                                    + 2.0 * mirror_lat * j as f64 / (num_dots - 1) as f64;
                                shell_pt3(l, phi, lam)
                            })
                            .collect();

                        if rad_config.show_dots {
                            let vis_dots: Vec<[f64; 2]> = dots
                                .iter()
                                .filter(|&&p| visible(p))
                                .map(|p| [p.0, p.1])
                                .collect();
                            if !vis_dots.is_empty() {
                                plot_ui.points(
                                    Points::new("", PlotPoints::new(vis_dots))
                                        .color(color)
                                        .radius(2.0),
                                );
                            }
                        }

                        dot_grid.push(dots);
                    }

                    if rad_config.connect_along_shell {
                        for j in 0..num_dots {
                            let ring: Vec<(f64, f64, f64)> = (0..=num_meridians)
                                .map(|mi| dot_grid[mi % num_meridians][j])
                                .collect();
                            draw_clipped_line(plot_ui, &ring, color, 0.5);
                        }
                    }

                    if rad_config.connect_across_shells {
                        if let Some(prev) = &prev_dot_grid {
                            let prev_color = shell_color(
                                l_min + (l_max - l_min) * (li as f64 - 1.0) / (num_l - 1) as f64,
                            );
                            let steps = 8;
                            let n = prev[0].len().min(dot_grid[0].len());
                            for mi in 0..num_meridians.min(prev.len()).min(dot_grid.len()) {
                                for j in 0..n {
                                    let a = prev[mi][j];
                                    let b = dot_grid[mi][j];
                                    let seg_pts: Vec<(f64, f64, f64)> = (0..=steps)
                                        .map(|s| {
                                            let t = s as f64 / steps as f64;
                                            (
                                                a.0 + (b.0 - a.0) * t,
                                                a.1 + (b.1 - a.1) * t,
                                                a.2 + (b.2 - a.2) * t,
                                            )
                                        })
                                        .collect();
                                    for s in 0..steps {
                                        let tm = (s as f64 + 0.5) / steps as f64;
                                        let p0 = seg_pts[s];
                                        let p1 = seg_pts[s + 1];
                                        let v0 = !hide_behind_earth
                                            || p0.2 >= 0.0
                                            || (p0.0 * p0.0 + p0.1 * p0.1) >= earth_r_sq;
                                        let v1 = !hide_behind_earth
                                            || p1.2 >= 0.0
                                            || (p1.0 * p1.0 + p1.1 * p1.1) >= earth_r_sq;
                                        if !v0 && !v1 {
                                            continue;
                                        }
                                        let seg_color = egui::Color32::from_rgba_unmultiplied(
                                            (prev_color.r() as f64
                                                + (color.r() as f64 - prev_color.r() as f64) * tm)
                                                as u8,
                                            (prev_color.g() as f64
                                                + (color.g() as f64 - prev_color.g() as f64) * tm)
                                                as u8,
                                            (prev_color.b() as f64
                                                + (color.b() as f64 - prev_color.b() as f64) * tm)
                                                as u8,
                                            (prev_color.a() as f64
                                                + (color.a() as f64 - prev_color.a() as f64) * tm)
                                                as u8,
                                        );
                                        if let Some((cp0, cp1)) = clip_link_at_earth(
                                            p0.0, p0.1, p0.2, v0, p1.0, p1.1, p1.2, v1, earth_r_sq,
                                        ) {
                                            plot_ui.line(
                                                Line::new("", PlotPoints::new(vec![cp0, cp1]))
                                                    .color(seg_color)
                                                    .width(0.5),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }

                    prev_dot_grid = Some(dot_grid);
                }

                if rad_config.show_fill {
                    for li in 0..num_l.saturating_sub(1) {
                        let l0 = l_min + (l_max - l_min) * li as f64 / (num_l - 1) as f64;
                        let l1 = l_min + (l_max - l_min) * (li + 1) as f64 / (num_l - 1) as f64;
                        let mirror0 = (1.0_f64 / l0.sqrt()).acos();
                        let mirror1 = (1.0_f64 / l1.sqrt()).acos();
                        let c0 = shell_color(l0);
                        let c1 = shell_color(l1);
                        let fill_color = egui::Color32::from_rgba_unmultiplied(
                            ((c0.r() as u16 + c1.r() as u16) / 2) as u8,
                            ((c0.g() as u16 + c1.g() as u16) / 2) as u8,
                            ((c0.b() as u16 + c1.b() as u16) / 2) as u8,
                            90,
                        );
                        let phi_off0 = rad_config.shell_phasing * PI * li as f64 / num_l as f64;
                        let phi_off1 =
                            rad_config.shell_phasing * PI * (li + 1) as f64 / num_l as f64;
                        let fill_lat_steps = (num_lat as f64 * (l1 / l_min).sqrt()) as usize;

                        for mi in 0..num_meridians {
                            let phi0 = phi_off0 + 2.0 * PI * mi as f64 / num_meridians as f64;
                            let phi1 = phi_off1 + 2.0 * PI * mi as f64 / num_meridians as f64;

                            for j in 0..fill_lat_steps {
                                let frac_a = j as f64 / fill_lat_steps as f64;
                                let frac_b = (j + 1) as f64 / fill_lat_steps as f64;
                                let p0 = shell_pt3(l0, phi0, -mirror0 + 2.0 * mirror0 * frac_a);
                                let p1 = shell_pt3(l0, phi0, -mirror0 + 2.0 * mirror0 * frac_b);
                                let p2 = shell_pt3(l1, phi1, -mirror1 + 2.0 * mirror1 * frac_b);
                                let p3 = shell_pt3(l1, phi1, -mirror1 + 2.0 * mirror1 * frac_a);
                                if !visible(p0) && !visible(p1) && !visible(p2) && !visible(p3) {
                                    continue;
                                }
                                let quad =
                                    vec![[p0.0, p0.1], [p1.0, p1.1], [p2.0, p2.1], [p3.0, p3.1]];
                                plot_ui.polygon(
                                    Polygon::new("", PlotPoints::new(quad))
                                        .fill_color(fill_color)
                                        .stroke(egui::Stroke::NONE),
                                );
                            }
                        }
                    }
                }
            }
        }

        if show_coverage {
            for (constellation, positions, color_offset, _tle_kind, _, _) in constellations {
                let orbit_radius = planet_radius + constellation.altitude_km;
                let cone_half_angle = (coverage_angle / 2.0).to_radians();
                let max_earth_angle = (planet_radius / orbit_radius).acos();
                let sin_beta = orbit_radius * cone_half_angle.sin() / planet_radius;
                let angular_radius = if sin_beta >= 1.0 {
                    max_earth_angle
                } else {
                    (sin_beta.asin() - cone_half_angle).min(max_earth_angle)
                };
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
                                + (angular_radius.sin() * angle.sin()).atan2(
                                    lat.cos() * angular_radius.cos()
                                        - lat.sin() * angular_radius.sin() * angle.cos(),
                                );

                            let x = planet_radius * clat.cos() * clon.cos();
                            let y = planet_radius * clat.sin();
                            let z = -planet_radius * clat.cos() * clon.sin();

                            let (rx, ry, rz) = rotate_point_matrix(x, y, z, &satellite_rotation);
                            ([rx, ry], rz >= 0.0)
                        })
                        .collect();

                    let all_visible = coverage_pts.iter().all(|(_, vis)| *vis);
                    let color = plane_color(sat.plane + color_offset);

                    if all_visible {
                        let pts: Vec<[f64; 2]> = coverage_pts.iter().map(|(p, _)| *p).collect();
                        let fill = egui::Color32::from_rgba_unmultiplied(
                            color.r(),
                            color.g(),
                            color.b(),
                            60,
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
                                        .width(scaled_link_width),
                                );
                            }
                        }
                        if !segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(segment))
                                    .color(color)
                                    .width(scaled_link_width),
                            );
                        }
                    }
                }
            }
        }

        let surface_rotation = if earth_fixed_camera {
            rotation
        } else {
            rotation * *body_rotation
        };

        if show_equator {
            let n_pts = 200;
            let eq_color = egui::Color32::from_rgb(255, 100, 100);
            let dim_eq = egui::Color32::from_rgba_unmultiplied(255, 100, 100, 60);
            let mut front_seg: Vec<[f64; 2]> = Vec::new();
            let mut back_seg: Vec<[f64; 2]> = Vec::new();
            for i in 0..=n_pts {
                let theta = 2.0 * PI * i as f64 / n_pts as f64;
                let x = planet_radius * theta.cos();
                let z = planet_radius * theta.sin();
                let (rx, ry, rz) = rotate_point_matrix(x, 0.0, z, &surface_rotation);
                let occluded = rz < 0.0 && (rx * rx + ry * ry) < earth_r_sq;
                if occluded {
                    if !front_seg.is_empty() {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(std::mem::take(&mut front_seg)))
                                .color(eq_color)
                                .width(1.5),
                        );
                    }
                    back_seg.push([rx, ry]);
                } else {
                    if !back_seg.is_empty() {
                        if !hide_behind_earth {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut back_seg)))
                                    .color(dim_eq)
                                    .width(1.0),
                            );
                        } else {
                            back_seg.clear();
                        }
                    }
                    front_seg.push([rx, ry]);
                }
            }
            if !front_seg.is_empty() {
                plot_ui.line(
                    Line::new("", PlotPoints::new(front_seg))
                        .color(eq_color)
                        .width(1.5),
                );
            }
            if !back_seg.is_empty() && !hide_behind_earth {
                plot_ui.line(
                    Line::new("", PlotPoints::new(back_seg))
                        .color(dim_eq)
                        .width(1.0),
                );
            }
        }

        if show_graticule {
            let grat_color = egui::Color32::from_rgb(100, 100, 100);
            let grat_dim = egui::Color32::from_rgba_unmultiplied(100, 100, 100, 30);
            let n_pts = 200;

            for lat_deg in (-60..=60).step_by(30) {
                if lat_deg == 0 && show_equator {
                    continue;
                }
                let phi = (lat_deg as f64).to_radians();
                let cos_phi = phi.cos();
                let sin_phi = phi.sin();
                let mut front_seg: Vec<[f64; 2]> = Vec::new();
                let mut back_seg: Vec<[f64; 2]> = Vec::new();
                for i in 0..=n_pts {
                    let theta = 2.0 * PI * i as f64 / n_pts as f64;
                    let x = planet_radius * cos_phi * theta.cos();
                    let y = planet_radius * sin_phi;
                    let z = planet_radius * cos_phi * theta.sin();
                    let (rx, ry, rz) = rotate_point_matrix(x, y, z, &surface_rotation);
                    let occluded = rz < 0.0 && (rx * rx + ry * ry) < earth_r_sq;
                    if occluded {
                        if !front_seg.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut front_seg)))
                                    .color(grat_color)
                                    .width(0.5),
                            );
                        }
                        back_seg.push([rx, ry]);
                    } else {
                        if !back_seg.is_empty() {
                            if !hide_behind_earth {
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(std::mem::take(&mut back_seg)))
                                        .color(grat_dim)
                                        .width(0.5),
                                );
                            } else {
                                back_seg.clear();
                            }
                        }
                        front_seg.push([rx, ry]);
                    }
                }
                if !front_seg.is_empty() {
                    plot_ui.line(
                        Line::new("", PlotPoints::new(front_seg))
                            .color(grat_color)
                            .width(0.5),
                    );
                }
                if !back_seg.is_empty() && !hide_behind_earth {
                    plot_ui.line(
                        Line::new("", PlotPoints::new(back_seg))
                            .color(grat_dim)
                            .width(0.5),
                    );
                }
            }

            for lon_deg in (-150..=180).step_by(30) {
                let lambda = (lon_deg as f64).to_radians();
                let cos_l = lambda.cos();
                let sin_l = lambda.sin();
                let mut front_seg: Vec<[f64; 2]> = Vec::new();
                let mut back_seg: Vec<[f64; 2]> = Vec::new();
                for i in 0..=n_pts {
                    let phi = -PI / 2.0 + PI * i as f64 / n_pts as f64;
                    let cos_phi = phi.cos();
                    let x = planet_radius * cos_phi * cos_l;
                    let y = planet_radius * phi.sin();
                    let z = planet_radius * cos_phi * sin_l;
                    let (rx, ry, rz) = rotate_point_matrix(x, y, z, &surface_rotation);
                    let occluded = rz < 0.0 && (rx * rx + ry * ry) < earth_r_sq;
                    if occluded {
                        if !front_seg.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut front_seg)))
                                    .color(grat_color)
                                    .width(0.5),
                            );
                        }
                        back_seg.push([rx, ry]);
                    } else {
                        if !back_seg.is_empty() {
                            if !hide_behind_earth {
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(std::mem::take(&mut back_seg)))
                                        .color(grat_dim)
                                        .width(0.5),
                                );
                            } else {
                                back_seg.clear();
                            }
                        }
                        front_seg.push([rx, ry]);
                    }
                }
                if !front_seg.is_empty() {
                    plot_ui.line(
                        Line::new("", PlotPoints::new(front_seg))
                            .color(grat_color)
                            .width(0.5),
                    );
                }
                if !back_seg.is_empty() && !hide_behind_earth {
                    plot_ui.line(
                        Line::new("", PlotPoints::new(back_seg))
                            .color(grat_dim)
                            .width(0.5),
                    );
                }
            }
        }

        if show_inclination_bounds {
            // Draw the two latitude circles at ±inclination for each non-TLE
            // constellation, using that constellation's colour. This visualises
            // the maximum latitude reachable by satellites in that orbit.
            let n_pts = 200;
            for (_, _, color_offset, tle_kind, _, _) in constellations {
                if *tle_kind != 0 {
                    continue;
                }
                let inc = match constellations
                    .iter()
                    .find(|(_, _, co, _, _, _)| co == color_offset)
                    .map(|(c, _, _, _, _, _)| c.inclination_deg.abs().min(90.0))
                {
                    Some(v) => v,
                    None => continue,
                };
                let base_color = plane_color(*color_offset);
                let dim_color = egui::Color32::from_rgba_unmultiplied(
                    base_color.r(),
                    base_color.g(),
                    base_color.b(),
                    90,
                );
                for sign in [1.0_f64, -1.0_f64] {
                    let lat = (sign * inc).to_radians();
                    let cos_lat = lat.cos();
                    let sin_lat = lat.sin();
                    let mut front_seg: Vec<[f64; 2]> = Vec::new();
                    let mut back_seg: Vec<[f64; 2]> = Vec::new();
                    for i in 0..=n_pts {
                        let theta = 2.0 * PI * i as f64 / n_pts as f64;
                        let x = planet_radius * cos_lat * theta.cos();
                        let y = planet_radius * sin_lat;
                        let z = planet_radius * cos_lat * theta.sin();
                        let (rx, ry, rz) = rotate_point_matrix(x, y, z, &surface_rotation);
                        let occluded = rz < 0.0 && (rx * rx + ry * ry) < earth_r_sq;
                        if occluded {
                            if !front_seg.is_empty() {
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(std::mem::take(&mut front_seg)))
                                        .color(base_color)
                                        .width(2.0),
                                );
                            }
                            back_seg.push([rx, ry]);
                        } else {
                            if !back_seg.is_empty() {
                                if !hide_behind_earth {
                                    plot_ui.line(
                                        Line::new(
                                            "",
                                            PlotPoints::new(std::mem::take(&mut back_seg)),
                                        )
                                        .color(dim_color)
                                        .width(1.0),
                                    );
                                } else {
                                    back_seg.clear();
                                }
                            }
                            front_seg.push([rx, ry]);
                        }
                    }
                    if !front_seg.is_empty() {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(front_seg))
                                .color(base_color)
                                .width(2.0),
                        );
                    }
                    if !back_seg.is_empty() && !hide_behind_earth {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(back_seg))
                                .color(dim_color)
                                .width(1.0),
                        );
                    }
                }
            }
        }

        if show_ground_tracks {
            // Ground tracks live in Earth-fixed lat/lon, so they're drawn with
            // the surface rotation (which rotates with the body). Consecutive
            // points whose longitude jumps by more than 180° are a wrap around
            // the ±180° meridian — we break the segment there so the line
            // doesn't cut straight across the globe.
            for (track_idx, track) in ground_tracks.iter().enumerate() {
                if track.len() < 2 {
                    continue;
                }
                let base_color = plane_color(track_idx);
                let dim_color = egui::Color32::from_rgba_unmultiplied(
                    base_color.r(),
                    base_color.g(),
                    base_color.b(),
                    90,
                );
                let mut front_seg: Vec<[f64; 2]> = Vec::new();
                let mut back_seg: Vec<[f64; 2]> = Vec::new();
                let mut last_lon: Option<f64> = None;
                let flush_segs = |plot_ui: &mut egui_plot::PlotUi,
                                  front: &mut Vec<[f64; 2]>,
                                  back: &mut Vec<[f64; 2]>| {
                    if front.len() >= 2 {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(std::mem::take(front)))
                                .color(base_color)
                                .width(2.0),
                        );
                    } else {
                        front.clear();
                    }
                    if back.len() >= 2 && !hide_behind_earth {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(std::mem::take(back)))
                                .color(dim_color)
                                .width(1.0),
                        );
                    } else {
                        back.clear();
                    }
                };
                for &(lat_deg, lon_deg) in track {
                    // Break segment on longitude wrap (>180° jump).
                    if let Some(prev) = last_lon {
                        if (lon_deg - prev).abs() > 180.0 {
                            flush_segs(plot_ui, &mut front_seg, &mut back_seg);
                        }
                    }
                    last_lon = Some(lon_deg);
                    let lat = lat_deg.to_radians();
                    let lon = lon_deg.to_radians();
                    // Surface point in body frame using the same convention as the
                    // satellite positions: lon = -atan2(z, x), so x = R*cos(lat)*cos(-lon),
                    // z = R*cos(lat)*sin(-lon).
                    let x = planet_radius * lat.cos() * (-lon).cos();
                    let y = planet_radius * lat.sin();
                    let z = planet_radius * lat.cos() * (-lon).sin();
                    let (rx, ry, rz) = rotate_point_matrix(x, y, z, &surface_rotation);
                    let occluded = rz < 0.0 && (rx * rx + ry * ry) < earth_r_sq;
                    if occluded {
                        if !front_seg.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut front_seg)))
                                    .color(base_color)
                                    .width(2.0),
                            );
                        }
                        back_seg.push([rx, ry]);
                    } else {
                        if !back_seg.is_empty() {
                            if !hide_behind_earth {
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(std::mem::take(&mut back_seg)))
                                        .color(dim_color)
                                        .width(1.0),
                                );
                            } else {
                                back_seg.clear();
                            }
                        }
                        front_seg.push([rx, ry]);
                    }
                }
                flush_segs(plot_ui, &mut front_seg, &mut back_seg);
            }
        }

        if show_crosshairs {
            if let Some(ptr) = plot_ui.pointer_coordinate() {
                let px = ptr.x;
                let py = ptr.y;
                if px * px + py * py <= earth_r_sq {
                    let pz = (earth_r_sq - px * px - py * py).sqrt();
                    let inv = surface_rotation.transpose();
                    let orig = inv * Vector3::new(px, py, pz);
                    let lat = (orig.y / planet_radius).asin();
                    let lon = -(orig.z.atan2(orig.x));
                    let cursor_color = egui::Color32::from_rgba_unmultiplied(200, 200, 200, 100);
                    let n_pts = 200;
                    let cos_lat = lat.cos();
                    let sin_lat = lat.sin();
                    let mut front_seg: Vec<[f64; 2]> = Vec::new();
                    for i in 0..=n_pts {
                        let theta = 2.0 * PI * i as f64 / n_pts as f64;
                        let x = planet_radius * cos_lat * theta.cos();
                        let y = planet_radius * sin_lat;
                        let z = planet_radius * cos_lat * theta.sin();
                        let (rx, ry, rz) = rotate_point_matrix(x, y, z, &surface_rotation);
                        if rz < 0.0 && (rx * rx + ry * ry) < earth_r_sq {
                            if front_seg.len() >= 2 {
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(std::mem::take(&mut front_seg)))
                                        .color(cursor_color)
                                        .width(0.5),
                                );
                            } else {
                                front_seg.clear();
                            }
                        } else {
                            front_seg.push([rx, ry]);
                        }
                    }
                    if front_seg.len() >= 2 {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(front_seg))
                                .color(cursor_color)
                                .width(0.5),
                        );
                    }
                    let cos_lon = lon.cos();
                    let sin_lon = lon.sin();
                    let mut front_seg: Vec<[f64; 2]> = Vec::new();
                    for i in 0..=n_pts {
                        let phi = -PI / 2.0 + PI * i as f64 / n_pts as f64;
                        let cp = phi.cos();
                        let x = planet_radius * cp * cos_lon;
                        let y = planet_radius * phi.sin();
                        let z = -planet_radius * cp * sin_lon;
                        let (rx, ry, rz) = rotate_point_matrix(x, y, z, &surface_rotation);
                        if rz < 0.0 && (rx * rx + ry * ry) < earth_r_sq {
                            if front_seg.len() >= 2 {
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(std::mem::take(&mut front_seg)))
                                        .color(cursor_color)
                                        .width(0.5),
                                );
                            } else {
                                front_seg.clear();
                            }
                        } else {
                            front_seg.push([rx, ry]);
                        }
                    }
                    if front_seg.len() >= 2 {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(front_seg))
                                .color(cursor_color)
                                .width(0.5),
                        );
                    }
                }
            }
        }

        if show_borders && body_key.0 == CelestialBody::Earth {
            let border_color = egui::Color32::from_rgb(0, 0, 0);
            let dim_border = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 40);
            for polyline in geo_borders {
                let mut front_seg: Vec<[f64; 2]> = Vec::new();
                let mut back_seg: Vec<[f64; 2]> = Vec::new();
                for &(lat_deg, lon_deg) in polyline {
                    let lat = lat_deg.to_radians();
                    let lon = (-lon_deg).to_radians();
                    let x = planet_radius * lat.cos() * lon.cos();
                    let y = planet_radius * lat.sin();
                    let z = planet_radius * lat.cos() * lon.sin();
                    let (rx, ry, rz) = rotate_point_matrix(x, y, z, &surface_rotation);
                    let occluded = rz < 0.0 && (rx * rx + ry * ry) < earth_r_sq;
                    if occluded {
                        if !front_seg.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut front_seg)))
                                    .color(border_color)
                                    .width(1.0),
                            );
                        }
                        back_seg.push([rx, ry]);
                    } else {
                        if !back_seg.is_empty() {
                            if !hide_behind_earth {
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(std::mem::take(&mut back_seg)))
                                        .color(dim_border)
                                        .width(0.5),
                                );
                            } else {
                                back_seg.clear();
                            }
                        }
                        front_seg.push([rx, ry]);
                    }
                }
                if !front_seg.is_empty() {
                    plot_ui.line(
                        Line::new("", PlotPoints::new(front_seg))
                            .color(border_color)
                            .width(1.0),
                    );
                }
                if !back_seg.is_empty() && !hide_behind_earth {
                    plot_ui.line(
                        Line::new("", PlotPoints::new(back_seg))
                            .color(dim_border)
                            .width(0.5),
                    );
                }
            }
        }

        if show_cities && body_key.0 == CelestialBody::Earth {
            let min_pop = if zoom >= 8.0 {
                0.0
            } else if zoom >= 4.0 {
                500_000.0
            } else if zoom >= 2.0 {
                2_000_000.0
            } else {
                5_000_000.0
            };
            let max_cities = if zoom >= 8.0 {
                200
            } else if zoom >= 4.0 {
                80
            } else if zoom >= 2.0 {
                30
            } else {
                15
            };
            let city_color = egui::Color32::from_rgb(220, 220, 200);
            let mut count = 0usize;
            for city in geo_cities {
                if city.population < min_pop {
                    continue;
                }
                if count >= max_cities {
                    break;
                }
                let lat = city.lat.to_radians();
                let lon = (-city.lon).to_radians();
                let x = planet_radius * lat.cos() * lon.cos();
                let y = planet_radius * lat.sin();
                let z = planet_radius * lat.cos() * lon.sin();
                let (rx, ry, rz) = rotate_point_matrix(x, y, z, &surface_rotation);
                if !hide_behind_earth || rz >= 0.0 {
                    surface_labels.push((
                        [rx, ry],
                        city.name.clone(),
                        city_color,
                        false,
                        500_000 + count,
                    ));
                    count += 1;
                }
            }
        }

        for (aoi_idx, aoi) in areas_of_interest.iter().enumerate() {
            let lat = aoi.lat.to_radians();
            let lon = (-aoi.lon).to_radians();
            let angular_radius = aoi.radius_km / planet_radius;

            let aoi_pts: Vec<([f64; 2], bool)> = (0..=32)
                .map(|i| {
                    let angle = 2.0 * PI * i as f64 / 32.0;
                    let clat = (lat.sin() * angular_radius.cos()
                        + lat.cos() * angular_radius.sin() * angle.cos())
                    .asin();
                    let clon = lon
                        + (angular_radius.sin() * angle.sin()).atan2(
                            lat.cos() * angular_radius.cos()
                                - lat.sin() * angular_radius.sin() * angle.cos(),
                        );

                    let x = planet_radius * clat.cos() * clon.cos();
                    let y = planet_radius * clat.sin();
                    let z = planet_radius * clat.cos() * clon.sin();

                    let (rx, ry, rz) = rotate_point_matrix(x, y, z, &surface_rotation);
                    ([rx, ry], rz >= 0.0)
                })
                .collect();

            let all_visible = aoi_pts.iter().all(|(_, vis)| *vis);
            if all_visible {
                let pts: Vec<[f64; 2]> = aoi_pts.iter().map(|(p, _)| *p).collect();
                let fill = egui::Color32::from_rgba_unmultiplied(
                    aoi.color.r(),
                    aoi.color.g(),
                    aoi.color.b(),
                    aoi.color.a(),
                );
                plot_ui.polygon(
                    Polygon::new("", PlotPoints::new(pts))
                        .fill_color(fill)
                        .stroke(egui::Stroke::new(2.0, aoi.color)),
                );
            } else {
                let mut segment: Vec<[f64; 2]> = Vec::new();
                for (pt, visible) in &aoi_pts {
                    if *visible {
                        segment.push(*pt);
                    } else if !segment.is_empty() {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(std::mem::take(&mut segment)))
                                .color(aoi.color)
                                .width(2.0),
                        );
                    }
                }
                if !segment.is_empty() {
                    plot_ui.line(
                        Line::new("", PlotPoints::new(segment))
                            .color(aoi.color)
                            .width(2.0),
                    );
                }
            }

            let cx = planet_radius * lat.cos() * lon.cos();
            let cy = planet_radius * lat.sin();
            let cz = planet_radius * lat.cos() * lon.sin();
            let (crx, cry, crz) = rotate_point_matrix(cx, cy, cz, &surface_rotation);
            if !hide_behind_earth || crz >= 0.0 {
                surface_labels.push(([crx, cry], aoi.name.clone(), aoi.color, false, aoi_idx));
            }
        }

        for (gs_idx, gs) in ground_stations.iter().enumerate() {
            let lat = gs.lat.to_radians();
            let lon = (-gs.lon).to_radians();
            let angular_radius = gs.radius_km / planet_radius;

            let gs_pts: Vec<([f64; 2], bool)> = (0..=32)
                .map(|i| {
                    let angle = 2.0 * PI * i as f64 / 32.0;
                    let clat = (lat.sin() * angular_radius.cos()
                        + lat.cos() * angular_radius.sin() * angle.cos())
                    .asin();
                    let clon = lon
                        + (angular_radius.sin() * angle.sin()).atan2(
                            lat.cos() * angular_radius.cos()
                                - lat.sin() * angular_radius.sin() * angle.cos(),
                        );

                    let x = planet_radius * clat.cos() * clon.cos();
                    let y = planet_radius * clat.sin();
                    let z = planet_radius * clat.cos() * clon.sin();

                    let (rx, ry, rz) = rotate_point_matrix(x, y, z, &surface_rotation);
                    ([rx, ry], rz >= 0.0)
                })
                .collect();

            let all_visible = gs_pts.iter().all(|(_, vis)| *vis);
            if all_visible {
                let pts: Vec<[f64; 2]> = gs_pts.iter().map(|(p, _)| *p).collect();
                let fill = egui::Color32::from_rgba_unmultiplied(
                    gs.color.r(),
                    gs.color.g(),
                    gs.color.b(),
                    50,
                );
                plot_ui.polygon(
                    Polygon::new("", PlotPoints::new(pts))
                        .fill_color(fill)
                        .stroke(egui::Stroke::new(2.0, gs.color)),
                );
            } else {
                let mut segment: Vec<[f64; 2]> = Vec::new();
                for (pt, visible) in &gs_pts {
                    if *visible {
                        segment.push(*pt);
                    } else if !segment.is_empty() {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(std::mem::take(&mut segment)))
                                .color(gs.color)
                                .width(2.0),
                        );
                    }
                }
                if !segment.is_empty() {
                    plot_ui.line(
                        Line::new("", PlotPoints::new(segment))
                            .color(gs.color)
                            .width(2.0),
                    );
                }
            }

            let x = planet_radius * lat.cos() * lon.cos();
            let y = planet_radius * lat.sin();
            let z = planet_radius * lat.cos() * lon.sin();
            let (rx, ry, rz) = rotate_point_matrix(x, y, z, &surface_rotation);

            if !hide_behind_earth || rz >= 0.0 {
                surface_labels.push(([rx, ry], gs.name.clone(), gs.color, true, gs_idx));
            }
        }

        let show_device_dots = zoom >= 2000.0;

        for (layer_idx, layer) in device_layers.iter().enumerate() {
            if layer.devices.is_empty() {
                continue;
            }

            let mut projected: Vec<([f64; 2], bool)> = Vec::new();
            for &(lat_deg, lon_deg) in &layer.devices {
                let lat = lat_deg.to_radians();
                let lon = (-lon_deg).to_radians();
                let x = planet_radius * lat.cos() * lon.cos();
                let y = planet_radius * lat.sin();
                let z = planet_radius * lat.cos() * lon.sin();
                let (rx, ry, rz) = rotate_point_matrix(x, y, z, &surface_rotation);
                let visible = !hide_behind_earth || rz >= 0.0;
                projected.push(([rx, ry], visible));
            }

            if show_device_dots {
                for (dev_idx, (pos, vis)) in projected.iter().enumerate() {
                    if !vis {
                        continue;
                    }
                    plot_ui.points(
                        Points::new("", PlotPoints::new(vec![*pos]))
                            .color(layer.color)
                            .radius(scaled_sat_radius * 0.8),
                    );
                    let label = format!("{} #{}", layer.name, dev_idx + 1);
                    surface_labels.push((
                        *pos,
                        label,
                        layer.color,
                        false,
                        1000000 + layer_idx * 10000 + dev_idx,
                    ));
                }
            } else {
                let cell_size = planet_radius * 0.15 / zoom.max(0.1);
                let min_circle_r = 2.0;
                let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> =
                    std::collections::HashMap::new();
                for (i, (pos, vis)) in projected.iter().enumerate() {
                    if !vis {
                        continue;
                    }
                    let gx = (pos[0] / cell_size).floor() as i64;
                    let gy = (pos[1] / cell_size).floor() as i64;
                    grid.entry((gx, gy)).or_default().push(i);
                }

                for indices in grid.values() {
                    let count = indices.len();
                    let cx: f64 =
                        indices.iter().map(|&i| projected[i].0[0]).sum::<f64>() / count as f64;
                    let cy: f64 =
                        indices.iter().map(|&i| projected[i].0[1]).sum::<f64>() / count as f64;

                    let circle_r = (cell_size * 0.35).max(min_circle_r);
                    let fill = egui::Color32::from_rgba_unmultiplied(
                        layer.color.r(),
                        layer.color.g(),
                        layer.color.b(),
                        60,
                    );
                    let n = 24;
                    let pts: Vec<[f64; 2]> = (0..=n)
                        .map(|i| {
                            let a = 2.0 * PI * i as f64 / n as f64;
                            [cx + circle_r * a.cos(), cy + circle_r * a.sin()]
                        })
                        .collect();
                    plot_ui.polygon(
                        Polygon::new("", PlotPoints::new(pts))
                            .fill_color(fill)
                            .stroke(egui::Stroke::new(1.5, layer.color)),
                    );

                    device_cluster_labels.push(([cx, cy], count, layer.color));
                }
            }
        }

        if show_axes {
            let (ep_x, ep_y, _) = rotate_point_matrix(axis_len, 0.0, 0.0, &satellite_rotation);
            let (wn_x, wn_y, _) = rotate_point_matrix(-axis_len, 0.0, 0.0, &satellite_rotation);
            plot_ui.line(
                Line::new("", PlotPoints::new(vec![[wn_x, wn_y], [ep_x, ep_y]]))
                    .color(egui::Color32::from_rgb(255, 100, 100))
                    .width(1.5),
            );

            let (np_x, np_y, _) = rotate_point_matrix(0.0, axis_len, 0.0, &satellite_rotation);
            let (sn_x, sn_y, _) = rotate_point_matrix(0.0, -axis_len, 0.0, &satellite_rotation);
            plot_ui.line(
                Line::new("", PlotPoints::new(vec![[sn_x, sn_y], [np_x, np_y]]))
                    .color(egui::Color32::from_rgb(100, 100, 255))
                    .width(1.5),
            );

            let label_offset = axis_len * 1.15;
            let (n_x, n_y, _) = rotate_point_matrix(0.0, label_offset, 0.0, &satellite_rotation);
            let (s_x, s_y, _) = rotate_point_matrix(0.0, -label_offset, 0.0, &satellite_rotation);
            let (e_x, e_y, _) = rotate_point_matrix(label_offset, 0.0, 0.0, &satellite_rotation);
            let (w_x, w_y, _) = rotate_point_matrix(-label_offset, 0.0, 0.0, &satellite_rotation);

            plot_ui.text(Text::new("", PlotPoint::new(n_x, n_y), "N").color(egui::Color32::WHITE));
            plot_ui.text(Text::new("", PlotPoint::new(s_x, s_y), "S").color(egui::Color32::WHITE));
            plot_ui.text(Text::new("", PlotPoint::new(e_x, e_y), "E").color(egui::Color32::WHITE));
            plot_ui.text(Text::new("", PlotPoint::new(w_x, w_y), "W").color(egui::Color32::WHITE));
        }

        if show_magnetic_axis {
            let mag_lat = 80.65_f64.to_radians();
            let mag_lon = 72.68_f64.to_radians();
            let mx = mag_lat.cos() * mag_lon.cos();
            let my = mag_lat.sin();
            let mz = mag_lat.cos() * mag_lon.sin();
            let (np_x, np_y, _) = rotate_point_matrix(
                mx * axis_len,
                my * axis_len,
                mz * axis_len,
                &surface_rotation,
            );
            let (sp_x, sp_y, _) = rotate_point_matrix(
                -mx * axis_len,
                -my * axis_len,
                -mz * axis_len,
                &surface_rotation,
            );
            plot_ui.line(
                Line::new("", PlotPoints::new(vec![[sp_x, sp_y], [np_x, np_y]]))
                    .color(egui::Color32::from_rgb(255, 100, 255))
                    .width(1.5),
            );
            let label_offset = axis_len * 1.15;
            let (gmn_x, gmn_y, _) = rotate_point_matrix(
                mx * label_offset,
                my * label_offset,
                mz * label_offset,
                &surface_rotation,
            );
            let (gms_x, gms_y, _) = rotate_point_matrix(
                -mx * label_offset,
                -my * label_offset,
                -mz * label_offset,
                &surface_rotation,
            );
            let mag_color = egui::Color32::from_rgb(255, 150, 255);
            plot_ui.text(Text::new("", PlotPoint::new(gmn_x, gmn_y), "GM-N").color(mag_color));
            plot_ui.text(Text::new("", PlotPoint::new(gms_x, gms_y), "GM-S").color(mag_color));
        }

        if show_orbits {
            for (constellation, _, color_offset, tle_kind, _, _) in constellations {
                if *tle_kind != 0 {
                    continue;
                }
                for plane in 0..constellation.num_planes {
                    let orbit_pts = constellation.orbit_points_3d(plane, time);
                    let color = if show_routing_paths || show_asc_desc_colors {
                        color_links
                    } else {
                        plane_color(if single_color {
                            *color_offset
                        } else {
                            plane + color_offset
                        })
                    };

                    let mut front_segment: Vec<[f64; 2]> = Vec::new();
                    for &(x, y, z) in &orbit_pts {
                        let (rx, ry, rz) = rotate_point_matrix(x, y, z, &satellite_rotation);
                        let visible = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;
                        if visible {
                            front_segment.push([rx, ry]);
                        } else if !front_segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut front_segment)))
                                    .color(color)
                                    .width(scaled_link_width),
                            );
                        }
                    }
                    if !front_segment.is_empty() {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(front_segment))
                                .color(color)
                                .width(scaled_link_width),
                        );
                    }
                }
            }
        }

        if show_links {
            let base_link_color = if show_routing_paths || show_asc_desc_colors {
                color_links
            } else {
                egui::Color32::from_rgb(200, 200, 200)
            };
            let link_dim = egui::Color32::from_rgba_unmultiplied(50, 50, 60, 80);
            let cursor_over_plot = plot_ui.pointer_coordinate().is_some();
            for (wc, positions, _, _, orig_idx, _) in constellations {
                let lb = wc.link_budget;
                for sat in positions {
                    for &neighbor_idx in &sat.neighbors {
                        let neighbor = &positions[neighbor_idx];
                        let (rx1, ry1, rz1) =
                            rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                        let (rx2, ry2, rz2) = rotate_point_matrix(
                            neighbor.x,
                            neighbor.y,
                            neighbor.z,
                            &satellite_rotation,
                        );
                        let visible1 = rz1 >= 0.0 || (rx1 * rx1 + ry1 * ry1) >= earth_r_sq;
                        let visible2 = rz2 >= 0.0 || (rx2 * rx2 + ry2 * ry2) >= earth_r_sq;
                        let dist_km = {
                            let dx = sat.x - neighbor.x;
                            let dy = sat.y - neighbor.y;
                            let dz = sat.z - neighbor.z;
                            (dx * dx + dy * dy + dz * dz).sqrt()
                        };
                        let id = crate::config::PinnedIsl::canonical(
                            *orig_idx,
                            sat.plane,
                            sat.sat_index,
                            neighbor.plane,
                            neighbor.sat_index,
                        );
                        let is_pinned = pinned_isls.contains(&id);
                        if hide_behind_earth {
                            if let Some((p1, p2)) = clip_link_at_earth(
                                rx1, ry1, rz1, visible1, rx2, ry2, rz2, visible2, earth_r_sq,
                            ) {
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(vec![p1, p2]))
                                        .color(base_link_color)
                                        .width(scaled_link_width),
                                );
                                if cursor_over_plot {
                                    hover_isl_segments.push((p1, p2, dist_km, lb, id));
                                }
                                if is_pinned {
                                    pinned_isl_overlays.push((p1, p2));
                                }
                            }
                        } else {
                            let color = if visible1 && visible2 {
                                base_link_color
                            } else {
                                link_dim
                            };
                            plot_ui.line(
                                Line::new("", PlotPoints::new(vec![[rx1, ry1], [rx2, ry2]]))
                                    .color(color)
                                    .width(scaled_link_width),
                            );
                            if cursor_over_plot {
                                hover_isl_segments.push(([rx1, ry1], [rx2, ry2], dist_km, lb, id));
                            }
                            if is_pinned {
                                pinned_isl_overlays.push(([rx1, ry1], [rx2, ry2]));
                            }
                        }
                    }
                }
            }
        }

        if show_gs_links && !ground_stations.is_empty() {
            let gs_link_color = egui::Color32::from_rgb(120, 220, 255);
            let gs_unit: Vec<Vector3<f64>> = ground_stations
                .iter()
                .map(|gs| {
                    let lat = gs.lat.to_radians();
                    let lon = (-gs.lon).to_radians();
                    let body =
                        Vector3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin());
                    body_rotation * body
                })
                .collect();

            for (constellation, positions, _, _, _, _) in constellations {
                let orbit_radius = planet_radius + constellation.altitude_km;
                let cone_half_angle = (coverage_angle / 2.0).to_radians();
                let max_earth_angle = (planet_radius / orbit_radius).acos();
                let sin_beta = orbit_radius * cone_half_angle.sin() / planet_radius;
                let angular_radius = if sin_beta >= 1.0 {
                    max_earth_angle
                } else {
                    (sin_beta.asin() - cone_half_angle).min(max_earth_angle)
                };
                let cos_thr = angular_radius.cos();
                for sat in positions {
                    let r_sat = (sat.x * sat.x + sat.y * sat.y + sat.z * sat.z).sqrt();
                    if r_sat < 1e-6 {
                        continue;
                    }
                    let sat_unit = Vector3::new(sat.x / r_sat, sat.y / r_sat, sat.z / r_sat);
                    for gs_inertial in &gs_unit {
                        if sat_unit.dot(gs_inertial) < cos_thr {
                            continue;
                        }
                        let gx = planet_radius * gs_inertial.x;
                        let gy = planet_radius * gs_inertial.y;
                        let gz = planet_radius * gs_inertial.z;
                        let (rsx, rsy, rsz) =
                            rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                        let (rgx, rgy, rgz) = rotate_point_matrix(gx, gy, gz, &satellite_rotation);
                        let sat_visible = rsz >= 0.0 || (rsx * rsx + rsy * rsy) >= earth_r_sq;
                        let gs_visible = rgz >= 0.0 || (rgx * rgx + rgy * rgy) >= earth_r_sq;
                        if hide_behind_earth {
                            if let Some((p1, p2)) = clip_link_at_earth(
                                rsx,
                                rsy,
                                rsz,
                                sat_visible,
                                rgx,
                                rgy,
                                rgz,
                                gs_visible,
                                earth_r_sq,
                            ) {
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(vec![p1, p2]))
                                        .color(gs_link_color)
                                        .width(scaled_link_width),
                                );
                            }
                        } else {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(vec![[rsx, rsy], [rgx, rgy]]))
                                    .color(gs_link_color)
                                    .width(scaled_link_width),
                            );
                        }
                    }
                }
            }
        }

        for (conj, threshold) in conjunction_lines {
            let (rx1, ry1, rz1) = rotate_point_matrix(
                conj.pos_a[0],
                conj.pos_a[1],
                conj.pos_a[2],
                &satellite_rotation,
            );
            let (rx2, ry2, rz2) = rotate_point_matrix(
                conj.pos_b[0],
                conj.pos_b[1],
                conj.pos_b[2],
                &satellite_rotation,
            );
            let urgency = 1.0 - (conj.distance_km / threshold).clamp(0.0, 1.0);
            let r = (255.0 * urgency) as u8;
            let g = (255.0 * (1.0 - urgency)) as u8;
            let color = egui::Color32::from_rgb(r, g, 0);
            let width = (scaled_link_width * (1.0 + 2.0 * urgency as f32)).max(1.0);
            let visible1 = rz1 >= 0.0 || (rx1 * rx1 + ry1 * ry1) >= earth_r_sq;
            let visible2 = rz2 >= 0.0 || (rx2 * rx2 + ry2 * ry2) >= earth_r_sq;
            if hide_behind_earth {
                if let Some((p1, p2)) =
                    clip_link_at_earth(rx1, ry1, rz1, visible1, rx2, ry2, rz2, visible2, earth_r_sq)
                {
                    plot_ui.line(
                        Line::new("", PlotPoints::new(vec![p1, p2]))
                            .color(color)
                            .width(width),
                    );
                }
            } else {
                let c = if visible1 && visible2 {
                    color
                } else {
                    egui::Color32::from_rgba_unmultiplied(r, g, 0, 80)
                };
                plot_ui.line(
                    Line::new("", PlotPoints::new(vec![[rx1, ry1], [rx2, ry2]]))
                        .color(c)
                        .width(width),
                );
            }
        }

        if show_routing_paths && !satellite_cameras.is_empty() {
            let manhattan_color = egui::Color32::from_rgb(255, 100, 100);
            let shortest_color = egui::Color32::from_rgb(100, 255, 100);
            let radiation_color = egui::Color32::from_rgb(100, 220, 255);
            let rad_grid = radiation.and_then(|r| r.igrf_rad_cache.as_ref().map(|(_, _, g)| g));

            for (cidx, (constellation, positions, _, _, _, _)) in constellations.iter().enumerate()
            {
                let tracked: Vec<_> = satellite_cameras
                    .iter()
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

                        let src_sat = positions
                            .iter()
                            .find(|s| s.plane == src.plane && s.sat_index == src.sat_index);
                        let dst_sat = positions
                            .iter()
                            .find(|s| s.plane == dst.plane && s.sat_index == dst.sat_index);

                        let can_route = match (src_sat, dst_sat) {
                            (Some(_), Some(_)) => {
                                if is_star {
                                    let plane_diff_fwd =
                                        (dst.plane + num_planes - src.plane) % num_planes;
                                    let plane_diff_bwd =
                                        (src.plane + num_planes - dst.plane) % num_planes;
                                    let crosses_seam = plane_diff_fwd > num_planes / 2
                                        && plane_diff_bwd > num_planes / 2;
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
                                src.plane,
                                src.sat_index,
                                dst.plane,
                                dst.sat_index,
                                num_planes,
                                sats_per_plane,
                                is_star,
                                positions,
                            );
                            draw_routing_path(
                                plot_ui,
                                &path,
                                positions,
                                &satellite_rotation,
                                manhattan_color,
                                scaled_routing_width,
                                hide_behind_earth,
                                earth_r_sq,
                                show_path_distance,
                                &mut path_distance_labels,
                            );
                        }

                        if show_shortest_path {
                            let path = compute_shortest_path(
                                src.plane,
                                src.sat_index,
                                dst.plane,
                                dst.sat_index,
                                num_planes,
                                sats_per_plane,
                                positions,
                                is_star,
                            );
                            draw_routing_path(
                                plot_ui,
                                &path,
                                positions,
                                &satellite_rotation,
                                shortest_color,
                                scaled_routing_width,
                                hide_behind_earth,
                                earth_r_sq,
                                show_path_distance,
                                &mut path_distance_labels,
                            );
                        }

                        if show_radiation_path {
                            let path = compute_radiation_path(
                                src.plane,
                                src.sat_index,
                                dst.plane,
                                dst.sat_index,
                                num_planes,
                                sats_per_plane,
                                positions,
                                is_star,
                                body_rotation,
                                rad_grid,
                                radiation_weight,
                            );
                            draw_routing_path(
                                plot_ui,
                                &path,
                                positions,
                                &satellite_rotation,
                                radiation_color,
                                scaled_routing_width,
                                hide_behind_earth,
                                earth_r_sq,
                                show_path_distance,
                                &mut path_distance_labels,
                            );
                        }
                    }
                }
            }
        }

        if show_proxy_links && satellite_cameras.len() >= 2 {
            let cone_half_angle = (coverage_angle / 2.0).to_radians();
            let cos_cone = cone_half_angle.cos();
            let r2 = planet_radius * planet_radius;
            let mut tracked: Vec<(usize, Vector3<f64>)> = Vec::new();
            for (cidx, (_, positions, _, _, _, _)) in constellations.iter().enumerate() {
                for cam in satellite_cameras
                    .iter()
                    .filter(|c| c.constellation_idx == cidx)
                {
                    if let Some(sat) = positions
                        .iter()
                        .find(|s| s.plane == cam.plane && s.sat_index == cam.sat_index)
                    {
                        tracked.push((cidx, Vector3::new(sat.x, sat.y, sat.z)));
                    }
                }
            }
            let proxy_color = egui::Color32::from_rgb(255, 180, 100);
            for i in 0..tracked.len() {
                for j in (i + 1)..tracked.len() {
                    let (ci_a, pa) = tracked[i];
                    let (ci_b, pb) = tracked[j];
                    if ci_a == ci_b {
                        continue;
                    }
                    let (outer, inner) = if pa.norm() >= pb.norm() {
                        (pa, pb)
                    } else {
                        (pb, pa)
                    };
                    let s_mag = outer.norm();
                    let sp = inner - outer;
                    let sp_mag = sp.norm();
                    if sp_mag < 1e-6 {
                        continue;
                    }
                    let cos_off_nadir = (s_mag * s_mag - outer.dot(&inner)) / (s_mag * sp_mag);
                    if cos_off_nadir < cos_cone {
                        continue;
                    }
                    let t = (-outer.dot(&sp) / (sp_mag * sp_mag)).clamp(0.0, 1.0);
                    let closest = outer + sp * t;
                    if closest.norm_squared() < r2 {
                        continue;
                    }
                    let (rx1, ry1, rz1) =
                        rotate_point_matrix(outer.x, outer.y, outer.z, &satellite_rotation);
                    let (rx2, ry2, rz2) =
                        rotate_point_matrix(inner.x, inner.y, inner.z, &satellite_rotation);
                    let v1 = rz1 >= 0.0 || (rx1 * rx1 + ry1 * ry1) >= earth_r_sq;
                    let v2 = rz2 >= 0.0 || (rx2 * rx2 + ry2 * ry2) >= earth_r_sq;
                    let w = (scaled_link_width * 1.5).max(1.5);
                    if hide_behind_earth {
                        if let Some((p1, p2)) =
                            clip_link_at_earth(rx1, ry1, rz1, v1, rx2, ry2, rz2, v2, earth_r_sq)
                        {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(vec![p1, p2]))
                                    .color(proxy_color)
                                    .width(w),
                            );
                        }
                    } else {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(vec![[rx1, ry1], [rx2, ry2]]))
                                .color(proxy_color)
                                .width(w),
                        );
                    }
                }
            }
        }

        for aoi in areas_of_interest {
            if let Some(gs_idx) = aoi.ground_station_idx {
                if let Some(gs) = ground_stations.get(gs_idx) {
                    match aoi.job_mode {
                        AoiJobMode::Route => {
                            let find_nearest_sat = |center_lat: f64,
                                                    center_lon: f64,
                                                    radius_km: f64,
                                                    ascending_filter: Option<bool>|
                             -> Option<(
                                usize,
                                &WalkerConstellation,
                                &Vec<SatelliteState>,
                                &SatelliteState,
                                u8,
                            )> {
                                let center_lat_rad = center_lat.to_radians();
                                let center_lon_rad = center_lon.to_radians() + body_rot_angle;
                                let max_angular_dist = radius_km / planet_radius;

                                let haversine_dist = |sat: &SatelliteState| -> f64 {
                                    let sat_lat_rad = sat.lat.to_radians();
                                    let sat_lon_rad = sat.lon.to_radians();
                                    let dlat = sat_lat_rad - center_lat_rad;
                                    let dlon = sat_lon_rad - center_lon_rad;
                                    let a = (dlat / 2.0).sin().powi(2)
                                        + center_lat_rad.cos()
                                            * sat_lat_rad.cos()
                                            * (dlon / 2.0).sin().powi(2);
                                    2.0 * a.sqrt().asin()
                                };

                                let mut best: Option<(
                                    usize,
                                    &WalkerConstellation,
                                    &Vec<SatelliteState>,
                                    &SatelliteState,
                                    u8,
                                    f64,
                                )> = None;

                                for (cidx, (cons, positions, _, tle_kind, _, _)) in
                                    constellations.iter().enumerate()
                                {
                                    let is_tle = *tle_kind != 0;
                                    let has_neighbors =
                                        is_tle && positions.iter().any(|s| !s.neighbors.is_empty());
                                    if is_tle && !has_neighbors {
                                        continue;
                                    }
                                    for sat in positions.iter() {
                                        if let Some(asc) = ascending_filter {
                                            if sat.ascending != asc {
                                                continue;
                                            }
                                        }
                                        let dist = haversine_dist(sat);
                                        if dist <= max_angular_dist
                                            && (best.is_none() || dist < best.as_ref().unwrap().5)
                                        {
                                            best =
                                                Some((cidx, cons, positions, sat, *tle_kind, dist));
                                        }
                                    }
                                }

                                best.map(|(cidx, cons, positions, sat, tk, _)| {
                                    (cidx, cons, positions, sat, tk)
                                })
                            };

                            let aoi_asc =
                                find_nearest_sat(aoi.lat, aoi.lon, aoi.radius_km, Some(true));
                            let gs_asc = find_nearest_sat(gs.lat, gs.lon, gs.radius_km, Some(true));
                            let (aoi_result, gs_result) = if aoi_asc.is_some() && gs_asc.is_some() {
                                (aoi_asc, gs_asc)
                            } else {
                                let aoi_desc =
                                    find_nearest_sat(aoi.lat, aoi.lon, aoi.radius_km, Some(false));
                                let gs_desc =
                                    find_nearest_sat(gs.lat, gs.lon, gs.radius_km, Some(false));
                                (aoi_desc, gs_desc)
                            };

                            if let (
                                Some((gs_cidx, gs_cons, gs_positions, gs_sat, gs_tk)),
                                Some((aoi_cidx, _, _, aoi_sat, _)),
                            ) = (gs_result, aoi_result)
                            {
                                let path_color = egui::Color32::from_rgb(255, 255, 0);
                                let routing_width = scaled_routing_width;

                                if gs_cidx == aoi_cidx {
                                    let path = if gs_tk != 0 {
                                        let gs_idx = gs_positions
                                            .iter()
                                            .position(|s| s.sat_index == gs_sat.sat_index)
                                            .unwrap_or(0);
                                        let aoi_idx = gs_positions
                                            .iter()
                                            .position(|s| s.sat_index == aoi_sat.sat_index)
                                            .unwrap_or(0);
                                        compute_shortest_path_graph(gs_idx, aoi_idx, gs_positions)
                                    } else {
                                        compute_shortest_path(
                                            gs_sat.plane,
                                            gs_sat.sat_index,
                                            aoi_sat.plane,
                                            aoi_sat.sat_index,
                                            gs_cons.num_planes,
                                            gs_cons.sats_per_plane(),
                                            gs_positions,
                                            gs_cons.walker_type == WalkerType::Star,
                                        )
                                    };
                                    draw_routing_path(
                                        plot_ui,
                                        &path,
                                        gs_positions,
                                        &satellite_rotation,
                                        path_color,
                                        routing_width,
                                        hide_behind_earth,
                                        earth_r_sq,
                                        show_path_distance,
                                        &mut path_distance_labels,
                                    );
                                } else {
                                    let (rx1, ry1, rz1) = rotate_point_matrix(
                                        gs_sat.x,
                                        gs_sat.y,
                                        gs_sat.z,
                                        &satellite_rotation,
                                    );
                                    let (rx2, ry2, rz2) = rotate_point_matrix(
                                        aoi_sat.x,
                                        aoi_sat.y,
                                        aoi_sat.z,
                                        &satellite_rotation,
                                    );

                                    let visible1 =
                                        rz1 >= 0.0 || (rx1 * rx1 + ry1 * ry1) >= earth_r_sq;
                                    let visible2 =
                                        rz2 >= 0.0 || (rx2 * rx2 + ry2 * ry2) >= earth_r_sq;

                                    if !hide_behind_earth || (visible1 && visible2) {
                                        plot_ui.line(
                                            Line::new(
                                                "",
                                                PlotPoints::new(vec![[rx1, ry1], [rx2, ry2]]),
                                            )
                                            .color(path_color)
                                            .width(routing_width),
                                        );
                                    }
                                }

                                let dot_size = scaled_sat_radius as f64 * 1.2;
                                let (rx1, ry1, _) = rotate_point_matrix(
                                    gs_sat.x,
                                    gs_sat.y,
                                    gs_sat.z,
                                    &satellite_rotation,
                                );
                                let (rx2, ry2, _) = rotate_point_matrix(
                                    aoi_sat.x,
                                    aoi_sat.y,
                                    aoi_sat.z,
                                    &satellite_rotation,
                                );
                                plot_ui.points(
                                    Points::new("", PlotPoints::new(vec![[rx1, ry1], [rx2, ry2]]))
                                        .radius(dot_size as f32)
                                        .color(path_color),
                                );
                            }
                        }
                        AoiJobMode::SpaceComp => {
                            let routing_width = scaled_routing_width;
                            let color_collector = egui::Color32::from_rgb(100, 200, 255);
                            let color_mapper = egui::Color32::from_rgb(255, 165, 0);
                            let color_reducer = egui::Color32::from_rgb(220, 50, 220);
                            let color_gs = egui::Color32::from_rgb(255, 255, 0);

                            for (_cidx, (cons, positions, _, tle_kind, _, _)) in
                                constellations.iter().enumerate()
                            {
                                let is_tle = *tle_kind != 0;
                                let has_neighbors =
                                    is_tle && positions.iter().any(|s| !s.neighbors.is_empty());
                                if is_tle && !has_neighbors {
                                    continue;
                                }
                                let is_star = cons.walker_type == WalkerType::Star;
                                let job = if is_tle {
                                    crate::spacecomp::compute_spacecomp_job_graph(
                                        aoi.lat,
                                        aoi.lon,
                                        aoi.radius_km,
                                        gs.lat,
                                        gs.lon,
                                        gs.radius_km,
                                        positions,
                                        planet_radius,
                                        body_rot_angle,
                                        aoi.job_n,
                                        aoi.reducer_placement,
                                    )
                                } else {
                                    crate::spacecomp::compute_spacecomp_job(
                                        aoi.lat,
                                        aoi.lon,
                                        aoi.radius_km,
                                        gs.lat,
                                        gs.lon,
                                        gs.radius_km,
                                        positions,
                                        cons,
                                        is_star,
                                        planet_radius,
                                        body_rot_angle,
                                        aoi.job_n,
                                        aoi.reducer_placement,
                                    )
                                };
                                let Some(job) = job else { continue };

                                let px_to_world_off = 2.0 * margin / width as f64;
                                let offset_unit = routing_width as f64 * px_to_world_off * 1.3;

                                let sat_to_arr = |sat_idx: usize| -> usize {
                                    positions
                                        .iter()
                                        .position(|s| s.sat_index == sat_idx)
                                        .unwrap_or(0)
                                };

                                let mut all_paths: Vec<(Vec<(usize, usize)>, egui::Color32)> =
                                    Vec::new();

                                for &(ci, mi) in &job.assignments {
                                    let (cp, cs) = job.collectors[ci];
                                    let (mp, ms) = job.mappers[mi];
                                    let path = if is_tle {
                                        compute_shortest_path_graph(
                                            sat_to_arr(cs),
                                            sat_to_arr(ms),
                                            positions,
                                        )
                                    } else {
                                        compute_shortest_path(
                                            cp,
                                            cs,
                                            mp,
                                            ms,
                                            cons.num_planes,
                                            cons.sats_per_plane(),
                                            positions,
                                            is_star,
                                        )
                                    };
                                    all_paths.push((path, color_collector));
                                }

                                let mut drawn_mappers = std::collections::HashSet::new();
                                for &(_ci, mi) in &job.assignments {
                                    if drawn_mappers.insert(mi) {
                                        let (mp, ms) = job.mappers[mi];
                                        let (rp, rs) = job.reducer;
                                        let path = if is_tle {
                                            compute_shortest_path_graph(
                                                sat_to_arr(ms),
                                                sat_to_arr(rs),
                                                positions,
                                            )
                                        } else {
                                            compute_shortest_path(
                                                mp,
                                                ms,
                                                rp,
                                                rs,
                                                cons.num_planes,
                                                cons.sats_per_plane(),
                                                positions,
                                                is_star,
                                            )
                                        };
                                        all_paths.push((path, color_mapper));
                                    }
                                }

                                {
                                    let (rp, rs) = job.reducer;
                                    let (gp, gsi) = job.gs_sat;
                                    let path = if is_tle {
                                        compute_shortest_path_graph(
                                            sat_to_arr(rs),
                                            sat_to_arr(gsi),
                                            positions,
                                        )
                                    } else {
                                        compute_shortest_path(
                                            rp,
                                            rs,
                                            gp,
                                            gsi,
                                            cons.num_planes,
                                            cons.sats_per_plane(),
                                            positions,
                                            is_star,
                                        )
                                    };
                                    all_paths.push((path, color_reducer));
                                }

                                type Edge = ((usize, usize), (usize, usize));
                                let norm_edge = |a: (usize, usize), b: (usize, usize)| -> Edge {
                                    if a <= b {
                                        (a, b)
                                    } else {
                                        (b, a)
                                    }
                                };
                                let mut edge_count: std::collections::HashMap<Edge, usize> =
                                    std::collections::HashMap::new();
                                for (path, _) in &all_paths {
                                    for w in path.windows(2) {
                                        let e = norm_edge(w[0], w[1]);
                                        *edge_count.entry(e).or_insert(0) += 1;
                                    }
                                }
                                let mut edge_idx: std::collections::HashMap<Edge, usize> =
                                    std::collections::HashMap::new();

                                let mut edge_perp: std::collections::HashMap<Edge, (f64, f64)> =
                                    std::collections::HashMap::new();
                                for (path, _) in &all_paths {
                                    for w in path.windows(2) {
                                        let e = norm_edge(w[0], w[1]);
                                        if edge_perp.contains_key(&e) {
                                            continue;
                                        }
                                        let pa = positions
                                            .iter()
                                            .find(|s| s.plane == e.0 .0 && s.sat_index == e.0 .1);
                                        let pb = positions
                                            .iter()
                                            .find(|s| s.plane == e.1 .0 && s.sat_index == e.1 .1);
                                        if let (Some(a), Some(b)) = (pa, pb) {
                                            let (ax, ay, _) = rotate_point_matrix(
                                                a.x,
                                                a.y,
                                                a.z,
                                                &satellite_rotation,
                                            );
                                            let (bx, by, _) = rotate_point_matrix(
                                                b.x,
                                                b.y,
                                                b.z,
                                                &satellite_rotation,
                                            );
                                            let dx = bx - ax;
                                            let dy = by - ay;
                                            let len = (dx * dx + dy * dy).sqrt();
                                            if len > 1e-12 {
                                                edge_perp.insert(e, (-dy / len, dx / len));
                                            } else {
                                                edge_perp.insert(e, (0.0, 0.0));
                                            }
                                        }
                                    }
                                }

                                for (path, color) in &all_paths {
                                    for w in path.windows(2) {
                                        let e = norm_edge(w[0], w[1]);
                                        let total = edge_count[&e];
                                        let idx = edge_idx.entry(e).or_insert(0);
                                        let spread =
                                            (*idx as f64 - (total - 1) as f64 / 2.0) * offset_unit;
                                        *idx += 1;

                                        let &(px, py) = edge_perp.get(&e).unwrap_or(&(0.0, 0.0));
                                        let nx = px * spread;
                                        let ny = py * spread;

                                        let pos1 = positions
                                            .iter()
                                            .find(|s| s.plane == w[0].0 && s.sat_index == w[0].1);
                                        let pos2 = positions
                                            .iter()
                                            .find(|s| s.plane == w[1].0 && s.sat_index == w[1].1);
                                        if let (Some(p1), Some(p2)) = (pos1, pos2) {
                                            let (rx1, ry1, rz1) = rotate_point_matrix(
                                                p1.x,
                                                p1.y,
                                                p1.z,
                                                &satellite_rotation,
                                            );
                                            let (rx2, ry2, rz2) = rotate_point_matrix(
                                                p2.x,
                                                p2.y,
                                                p2.z,
                                                &satellite_rotation,
                                            );

                                            let visible1 =
                                                rz1 >= 0.0 || (rx1 * rx1 + ry1 * ry1) >= earth_r_sq;
                                            let visible2 =
                                                rz2 >= 0.0 || (rx2 * rx2 + ry2 * ry2) >= earth_r_sq;

                                            if hide_behind_earth {
                                                if let Some((cp1, cp2)) = clip_link_at_earth(
                                                    rx1, ry1, rz1, visible1, rx2, ry2, rz2,
                                                    visible2, earth_r_sq,
                                                ) {
                                                    plot_ui.line(
                                                        Line::new(
                                                            "",
                                                            PlotPoints::new(vec![
                                                                [cp1[0] + nx, cp1[1] + ny],
                                                                [cp2[0] + nx, cp2[1] + ny],
                                                            ]),
                                                        )
                                                        .color(*color)
                                                        .width(routing_width),
                                                    );
                                                }
                                            } else {
                                                let line_color = if visible1 && visible2 {
                                                    *color
                                                } else {
                                                    egui::Color32::from_rgba_unmultiplied(
                                                        color.r() / 2,
                                                        color.g() / 2,
                                                        color.b() / 2,
                                                        150,
                                                    )
                                                };
                                                plot_ui.line(
                                                    Line::new(
                                                        "",
                                                        PlotPoints::new(vec![
                                                            [rx1 + nx, ry1 + ny],
                                                            [rx2 + nx, ry2 + ny],
                                                        ]),
                                                    )
                                                    .color(line_color)
                                                    .width(routing_width),
                                                );
                                            }
                                        }
                                    }
                                }

                                if show_path_distance {
                                    for (path, _) in &all_paths {
                                        if let Some(label) = path_distance_label(
                                            path,
                                            positions,
                                            &satellite_rotation,
                                        ) {
                                            path_distance_labels.push(label);
                                        }
                                    }
                                }

                                let px_to_world = 2.0 * margin / width as f64;
                                let circle_r = scaled_sat_radius as f64 * px_to_world * 2.5;
                                let circle_segs = 24;
                                let make_circle = |cx: f64, cy: f64, r: f64| -> Vec<[f64; 2]> {
                                    (0..=circle_segs)
                                        .map(|i| {
                                            let angle = 2.0 * PI * i as f64 / circle_segs as f64;
                                            [cx + r * angle.cos(), cy + r * angle.sin()]
                                        })
                                        .collect()
                                };

                                let mut circles: Vec<(f64, f64, f64, egui::Color32)> = Vec::new();
                                for &(cp, cs) in &job.collectors {
                                    if let Some(s) = positions
                                        .iter()
                                        .find(|s| s.plane == cp && s.sat_index == cs)
                                    {
                                        let (rx, ry, _) =
                                            rotate_point_matrix(s.x, s.y, s.z, &satellite_rotation);
                                        circles.push((rx, ry, circle_r, color_collector));
                                    }
                                }
                                for &(mp, ms) in &job.mappers {
                                    if let Some(s) = positions
                                        .iter()
                                        .find(|s| s.plane == mp && s.sat_index == ms)
                                    {
                                        let (rx, ry, _) =
                                            rotate_point_matrix(s.x, s.y, s.z, &satellite_rotation);
                                        circles.push((rx, ry, circle_r, color_mapper));
                                    }
                                }
                                {
                                    let (rp, rs) = job.reducer;
                                    if let Some(s) = positions
                                        .iter()
                                        .find(|s| s.plane == rp && s.sat_index == rs)
                                    {
                                        let (rx, ry, _) =
                                            rotate_point_matrix(s.x, s.y, s.z, &satellite_rotation);
                                        circles.push((rx, ry, circle_r * 1.5, color_reducer));
                                    }
                                }
                                {
                                    let (gp, gsi) = job.gs_sat;
                                    if let Some(s) = positions
                                        .iter()
                                        .find(|s| s.plane == gp && s.sat_index == gsi)
                                    {
                                        let (rx, ry, _) =
                                            rotate_point_matrix(s.x, s.y, s.z, &satellite_rotation);
                                        circles.push((rx, ry, circle_r * 1.2, color_gs));
                                    }
                                }

                                for &(cx, cy, r, color) in &circles {
                                    let pts = make_circle(cx, cy, r);
                                    plot_ui.line(
                                        Line::new("", PlotPoints::new(pts))
                                            .color(color)
                                            .width(routing_width * 0.6),
                                    );
                                }

                                let push_label =
                                    |labels: &mut Vec<([f64; 2], &'static str, egui::Color32)>,
                                     pos: &SatelliteState,
                                     text: &'static str,
                                     color: egui::Color32| {
                                        let (rx, ry, _) = rotate_point_matrix(
                                            pos.x,
                                            pos.y,
                                            pos.z,
                                            &satellite_rotation,
                                        );
                                        labels.push(([rx, ry], text, color));
                                    };
                                for &(cp, cs) in &job.collectors {
                                    if let Some(s) = positions
                                        .iter()
                                        .find(|s| s.plane == cp && s.sat_index == cs)
                                    {
                                        push_label(
                                            &mut spacecomp_role_labels,
                                            s,
                                            "Collector",
                                            color_collector,
                                        );
                                    }
                                }
                                for &(mp, ms) in &job.mappers {
                                    if let Some(s) = positions
                                        .iter()
                                        .find(|s| s.plane == mp && s.sat_index == ms)
                                    {
                                        push_label(
                                            &mut spacecomp_role_labels,
                                            s,
                                            "Mapper",
                                            color_mapper,
                                        );
                                    }
                                }
                                {
                                    let (rp, rs) = job.reducer;
                                    if let Some(s) = positions
                                        .iter()
                                        .find(|s| s.plane == rp && s.sat_index == rs)
                                    {
                                        push_label(
                                            &mut spacecomp_role_labels,
                                            s,
                                            "Reducer",
                                            color_reducer,
                                        );
                                    }
                                }
                                {
                                    let (gp, gsi) = job.gs_sat;
                                    if let Some(s) = positions
                                        .iter()
                                        .find(|s| s.plane == gp && s.sat_index == gsi)
                                    {
                                        push_label(
                                            &mut spacecomp_role_labels,
                                            s,
                                            "Line of Sight",
                                            color_gs,
                                        );
                                    }
                                }

                                break;
                            }
                        }
                    }
                }
            }
        }

        let sun_inertial = if show_eclipse {
            let sd =
                Vector3::new(sun_dir[0] as f64, sun_dir[1] as f64, sun_dir[2] as f64).normalize();
            Some((body_rotation * sd).normalize())
        } else {
            None
        };

        for (ci, (constellation, positions, color_offset, tle_kind, orig_idx, cons_label)) in
            constellations.iter().enumerate()
        {
            if *tle_kind != 0 {
                let is_tle_debris = *tle_kind == 2;
                let is_kessler = *tle_kind == 3;
                for sat in positions {
                    let (rx, ry, rz) =
                        rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                    let in_front = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;

                    if is_kessler {
                        if hide_behind_earth && !in_front {
                            continue;
                        }
                        let alpha = if in_front { 220u8 } else { 100 };
                        let c = egui::Color32::from_rgba_unmultiplied(255, 60, 60, alpha);
                        // Same cross rendering as TLE debris, so Kessler
                        // fragments read as "debris" at a glance.
                        let d = scaled_sat_radius as f64 * 1.5 * margin / (width as f64 * 0.5);
                        let w = if in_front { 1.5 } else { 0.8 };
                        plot_ui.line(
                            egui_plot::Line::new(
                                "",
                                PlotPoints::new(vec![[rx - d, ry - d], [rx + d, ry + d]]),
                            )
                            .color(c)
                            .width(w),
                        );
                        plot_ui.line(
                            egui_plot::Line::new(
                                "",
                                PlotPoints::new(vec![[rx - d, ry + d], [rx + d, ry - d]]),
                            )
                            .color(c)
                            .width(w),
                        );
                        continue;
                    }

                    let mut color = if let Some(&pc) = physics_colors.get(&(
                        *orig_idx,
                        sat.plane * constellation.sats_per_plane() + sat.sat_index,
                    )) {
                        pc
                    } else if is_tle_debris {
                        // Bright, distinct palette for debris X marks. Order:
                        // Fengyun 1C = red, Cosmos 2251 = green, Iridium 33 =
                        // blue, Cosmos 1408 = yellow. Anything else falls back
                        // to the normal plane color.
                        if cons_label.starts_with("Fengyun") {
                            egui::Color32::from_rgb(255, 70, 70)
                        } else if cons_label.starts_with("Cosmos 2251") {
                            egui::Color32::from_rgb(70, 230, 90)
                        } else if cons_label.starts_with("Iridium 33") {
                            egui::Color32::from_rgb(90, 160, 255)
                        } else if cons_label.starts_with("Cosmos 1408") {
                            egui::Color32::from_rgb(255, 220, 70)
                        } else {
                            plane_color(color_offset + sat.plane)
                        }
                    } else {
                        plane_color(color_offset + sat.plane)
                    };
                    if let Some(rc) = radiation {
                        if rc.show_sat_exposure {
                            let bp = body_rotation.transpose() * Vector3::new(sat.x, sat.y, sat.z);
                            let tilt = rc.dipole_tilt.to_radians();
                            let tl = (-287.3_f64).to_radians();
                            let ma = Vector3::new(
                                tilt.sin() * tl.cos(),
                                tilt.cos(),
                                tilt.sin() * tl.sin(),
                            );
                            let o_lat = 22.0_f64.to_radians();
                            let o_lon = (-140.0_f64).to_radians();
                            let ox = rc.dipole_offset_km * o_lat.cos() * o_lon.cos();
                            let oy = rc.dipole_offset_km * o_lat.sin();
                            let oz = rc.dipole_offset_km * o_lat.cos() * o_lon.sin();
                            let dx = bp.x - ox;
                            let dy = bp.y - oy;
                            let dz = bp.z - oz;
                            let r_d = (dx * dx + dy * dy + dz * dz).sqrt();
                            let r_c = (bp.x * bp.x + bp.y * bp.y + bp.z * bp.z).sqrt();
                            let saa_factor = (r_d / r_c).powi(12);
                            let mag_dot = dx * ma.x + dy * ma.y + dz * ma.z;
                            let sin_ml = mag_dot / r_d;
                            let cos_ml_sq = 1.0 - sin_ml * sin_ml;
                            let r_d_er = r_d / planet_radius;
                            let l = if cos_ml_sq > 1e-6 {
                                r_d_er / cos_ml_sq
                            } else {
                                r_d_er * 1e6
                            };
                            let exp = match rc.heatmap_mode {
                                crate::config::HeatmapMode::Radiation => {
                                    (crate::radiation::belt_profile_r(l, rc.kp_index) * saa_factor)
                                        .clamp(0.0, 1.0)
                                }
                                crate::config::HeatmapMode::FieldStrength => {
                                    let b0 = 30115.0;
                                    let f =
                                        b0 / r_d_er.powi(3) * (1.0 + 3.0 * sin_ml * sin_ml).sqrt();
                                    normalize_field_nt(f, r_d_er)
                                }
                                crate::config::HeatmapMode::IgrfField => {
                                    let r_km = (bp.x * bp.x + bp.y * bp.y + bp.z * bp.z).sqrt();
                                    let colat = (bp.y / r_km).acos();
                                    let elon = (-bp.z).atan2(bp.x);
                                    let f = crate::igrf::igrf_field_nt(r_km, colat, elon);
                                    if let Some((_, ref g)) = rc.igrf_grid_cache {
                                        g.normalize(f)
                                    } else {
                                        normalize_field_nt(f, r_km / 6371.0)
                                    }
                                }
                                crate::config::HeatmapMode::IgrfRadiation => 0.0,
                            };
                            if rc.heatmap_mode == crate::config::HeatmapMode::IgrfRadiation {
                                let colat = (bp.y
                                    / (bp.x * bp.x + bp.y * bp.y + bp.z * bp.z).sqrt())
                                .acos();
                                let elon = (-bp.z).atan2(bp.x);
                                if let Some((_, _, ref grid)) = rc.igrf_rad_cache {
                                    let (p, e) = grid.lookup(colat, elon);
                                    color = blend_proton_electron(
                                        p,
                                        e,
                                        rc.show_protons,
                                        rc.show_electrons,
                                        rc.smooth_colors,
                                    );
                                }
                            } else {
                                color = crate::config::heatmap_color(exp, rc.smooth_colors);
                            }
                        }
                    }
                    let color = if let Some(ref sun) = sun_inertial {
                        let sp = Vector3::new(sat.x, sat.y, sat.z);
                        let proj = sp.dot(sun);
                        if proj < 0.0 && (sp.dot(&sp) - proj * proj) < planet_radius * planet_radius
                        {
                            egui::Color32::from_rgba_unmultiplied(
                                color.r() / 3,
                                color.g() / 3,
                                color.b() / 3,
                                color.a(),
                            )
                        } else {
                            color
                        }
                    } else {
                        color
                    };
                    let dim_col = egui::Color32::from_rgba_unmultiplied(
                        color.r() / 2,
                        color.g() / 2,
                        color.b() / 2,
                        80,
                    );

                    let bg_color = if dark_mode {
                        egui::Color32::from_rgb(30, 30, 30)
                    } else {
                        egui::Color32::from_rgb(240, 240, 240)
                    };

                    if is_tle_debris {
                        let size_mul = if tle_monochrome { 2.2 } else { 1.5 };
                        let d = scaled_sat_radius as f64 * size_mul * margin / (width as f64 * 0.5);
                        let c = if in_front {
                            color
                        } else if !hide_behind_earth {
                            dim_col
                        } else {
                            continue;
                        };
                        let w_base = if tle_monochrome { 2.5 } else { 1.0 };
                        let w = if in_front { w_base } else { w_base * 0.5 };
                        plot_ui.line(
                            egui_plot::Line::new(
                                "",
                                PlotPoints::new(vec![[rx - d, ry - d], [rx + d, ry + d]]),
                            )
                            .color(c)
                            .width(w),
                        );
                        plot_ui.line(
                            egui_plot::Line::new(
                                "",
                                PlotPoints::new(vec![[rx - d, ry + d], [rx + d, ry - d]]),
                            )
                            .color(c)
                            .width(w),
                        );
                    } else {
                        if !hide_behind_earth && !in_front {
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(dim_col)
                                    .radius(scaled_sat_radius * 0.8)
                                    .filled(true),
                            );
                            if has_simulated {
                                plot_ui.points(
                                    Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                        .color(bg_color)
                                        .radius(scaled_sat_radius * 0.4)
                                        .filled(true),
                                );
                            }
                        }
                        if in_front {
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(color)
                                    .radius(scaled_sat_radius)
                                    .filled(true),
                            );
                            if has_simulated {
                                plot_ui.points(
                                    Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                        .color(bg_color)
                                        .radius(scaled_sat_radius * 0.5)
                                        .filled(true),
                                );
                            }
                        }
                    }
                }
                continue;
            }
            for plane in 0..constellation.num_planes {
                let base_color = plane_color(if single_color {
                    *color_offset
                } else {
                    plane + color_offset
                });

                for sat in positions.iter().filter(|s| s.plane == plane) {
                    let flat_idx = sat.plane * constellation.sats_per_plane() + sat.sat_index;
                    let color = if let Some(&pc) = physics_colors.get(&(*orig_idx, flat_idx)) {
                        pc
                    } else if show_asc_desc_colors {
                        if sat.ascending {
                            color_ascending
                        } else {
                            color_descending
                        }
                    } else {
                        base_color
                    };
                    let color = if let Some(rc) = radiation {
                        if rc.show_sat_exposure {
                            let bp = body_rotation.transpose() * Vector3::new(sat.x, sat.y, sat.z);
                            let tilt = rc.dipole_tilt.to_radians();
                            let tl = (-287.3_f64).to_radians();
                            let ma = Vector3::new(
                                tilt.sin() * tl.cos(),
                                tilt.cos(),
                                tilt.sin() * tl.sin(),
                            );
                            let o_lat = 22.0_f64.to_radians();
                            let o_lon = (-140.0_f64).to_radians();
                            let ox = rc.dipole_offset_km * o_lat.cos() * o_lon.cos();
                            let oy = rc.dipole_offset_km * o_lat.sin();
                            let oz = rc.dipole_offset_km * o_lat.cos() * o_lon.sin();
                            let dx = bp.x - ox;
                            let dy = bp.y - oy;
                            let dz = bp.z - oz;
                            let r_d = (dx * dx + dy * dy + dz * dz).sqrt();
                            let r_c = (bp.x * bp.x + bp.y * bp.y + bp.z * bp.z).sqrt();
                            let saa_factor = (r_d / r_c).powi(12);
                            let mag_dot = dx * ma.x + dy * ma.y + dz * ma.z;
                            let sin_ml = mag_dot / r_d;
                            let cos_ml_sq = 1.0 - sin_ml * sin_ml;
                            let r_d_er = r_d / planet_radius;
                            let l = if cos_ml_sq > 1e-6 {
                                r_d_er / cos_ml_sq
                            } else {
                                r_d_er * 1e6
                            };
                            let exp = match rc.heatmap_mode {
                                crate::config::HeatmapMode::Radiation => {
                                    (crate::radiation::belt_profile_r(l, rc.kp_index) * saa_factor)
                                        .clamp(0.0, 1.0)
                                }
                                crate::config::HeatmapMode::FieldStrength => {
                                    let b0 = 30115.0;
                                    let f =
                                        b0 / r_d_er.powi(3) * (1.0 + 3.0 * sin_ml * sin_ml).sqrt();
                                    normalize_field_nt(f, r_d_er)
                                }
                                crate::config::HeatmapMode::IgrfField => {
                                    let r_km = (bp.x * bp.x + bp.y * bp.y + bp.z * bp.z).sqrt();
                                    let colat = (bp.y / r_km).acos();
                                    let elon = (-bp.z).atan2(bp.x);
                                    let f = crate::igrf::igrf_field_nt(r_km, colat, elon);
                                    if let Some((_, ref g)) = rc.igrf_grid_cache {
                                        g.normalize(f)
                                    } else {
                                        normalize_field_nt(f, r_km / 6371.0)
                                    }
                                }
                                crate::config::HeatmapMode::IgrfRadiation => 0.0,
                            };
                            if rc.heatmap_mode == crate::config::HeatmapMode::IgrfRadiation {
                                let colat = (bp.y
                                    / (bp.x * bp.x + bp.y * bp.y + bp.z * bp.z).sqrt())
                                .acos();
                                let elon = (-bp.z).atan2(bp.x);
                                if let Some((_, _, ref grid)) = rc.igrf_rad_cache {
                                    let (p, e) = grid.lookup(colat, elon);
                                    blend_proton_electron(
                                        p,
                                        e,
                                        rc.show_protons,
                                        rc.show_electrons,
                                        rc.smooth_colors,
                                    )
                                } else {
                                    color
                                }
                            } else {
                                crate::config::heatmap_color(exp, rc.smooth_colors)
                            }
                        } else {
                            color
                        }
                    } else {
                        color
                    };
                    let color = if let Some(ref sun) = sun_inertial {
                        let sp = Vector3::new(sat.x, sat.y, sat.z);
                        let proj = sp.dot(sun);
                        if proj < 0.0 && (sp.dot(&sp) - proj * proj) < planet_radius * planet_radius
                        {
                            egui::Color32::from_rgba_unmultiplied(
                                color.r() / 3,
                                color.g() / 3,
                                color.b() / 3,
                                color.a(),
                            )
                        } else {
                            color
                        }
                    } else {
                        color
                    };
                    let dim_col = egui::Color32::from_rgba_unmultiplied(
                        color.r() / 2,
                        color.g() / 2,
                        color.b() / 2,
                        80,
                    );

                    let intensity = flash_intensities
                        .get(ci)
                        .and_then(|m| m.get(&(flat_idx as u32)))
                        .copied()
                        .unwrap_or(0.0);
                    let color = if intensity > 0.0 {
                        let m = intensity.clamp(0.0, 1.0);
                        let r = (color.r() as f32 * (1.0 - m) + 255.0 * m) as u8;
                        let g = (color.g() as f32 * (1.0 - m) + 255.0 * m) as u8;
                        let b = (color.b() as f32 * (1.0 - m) + 255.0 * m) as u8;
                        egui::Color32::from_rgba_unmultiplied(r, g, b, color.a())
                    } else {
                        color
                    };
                    let active_sat_radius = active_sat_radius * (1.0 + intensity * 1.2);

                    let (rx, ry, rz) =
                        rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                    let in_front = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;

                    let bg_color = if dark_mode {
                        egui::Color32::from_rgb(30, 30, 30)
                    } else {
                        egui::Color32::from_rgb(240, 240, 240)
                    };

                    if !hide_behind_earth && !in_front {
                        if show_sat_border {
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(bg_color)
                                    .radius(active_sat_radius * 0.8 + 1.0)
                                    .filled(false),
                            );
                        }
                        plot_ui.points(
                            Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                .color(dim_col)
                                .radius(active_sat_radius * 0.8)
                                .filled(true),
                        );
                        if *tle_kind != 0 && has_simulated {
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(bg_color)
                                    .radius(active_sat_radius * 0.4)
                                    .filled(true),
                            );
                        }
                    }
                    if in_front {
                        if show_sat_border {
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(bg_color)
                                    .radius(active_sat_radius + 1.0)
                                    .filled(false),
                            );
                        }
                        plot_ui.points(
                            Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                .color(color)
                                .radius(active_sat_radius)
                                .filled(true),
                        );
                        if *tle_kind != 0 && has_simulated {
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(bg_color)
                                    .radius(active_sat_radius * 0.5)
                                    .filled(true),
                            );
                        }
                    }

                    if in_front && !correcting_sats.is_empty() {
                        let sat_name = sat.name.clone().unwrap_or_else(|| {
                            format!("{}#{} P{}:S{}", cons_label, ci, sat.plane, sat.sat_index)
                        });
                        if correcting_sats.contains(&sat_name) {
                            let green = egui::Color32::from_rgb(0, 255, 120);
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(green)
                                    .radius(scaled_sat_radius * 2.0)
                                    .filled(false),
                            );
                        }
                    }

                    if (in_front || !hide_behind_earth) && !hit_sats.is_empty() {
                        let sat_name = sat.name.clone().unwrap_or_else(|| {
                            format!("{}#{} P{}:S{}", cons_label, ci, sat.plane, sat.sat_index)
                        });
                        if hit_sats.contains(&sat_name) {
                            let d = scaled_sat_radius as f64 * 3.0 * margin / (width as f64 * 0.5);
                            let red = egui::Color32::from_rgb(255, 60, 60);
                            let w = if in_front {
                                2.5 * zoom_factor
                            } else {
                                1.5 * zoom_factor
                            };
                            plot_ui.line(
                                egui_plot::Line::new(
                                    "",
                                    PlotPoints::new(vec![[rx - d, ry - d], [rx + d, ry + d]]),
                                )
                                .color(red)
                                .width(w),
                            );
                            plot_ui.line(
                                egui_plot::Line::new(
                                    "",
                                    PlotPoints::new(vec![[rx - d, ry + d], [rx + d, ry - d]]),
                                )
                                .color(red)
                                .width(w),
                            );
                        }
                    }

                    if constellation.altitude_km < 100.0 && (in_front || !hide_behind_earth) {
                        let d = scaled_sat_radius as f64 * 3.0 * margin / (width as f64 * 0.5);
                        let red = egui::Color32::from_rgb(255, 60, 60);
                        plot_ui.line(
                            egui_plot::Line::new(
                                "",
                                PlotPoints::new(vec![[rx - d, ry - d], [rx + d, ry + d]]),
                            )
                            .color(red)
                            .width(2.0 * zoom_factor),
                        );
                        plot_ui.line(
                            egui_plot::Line::new(
                                "",
                                PlotPoints::new(vec![[rx - d, ry + d], [rx + d, ry - d]]),
                            )
                            .color(red)
                            .width(2.0 * zoom_factor),
                        );
                    }

                    if show_altitude_lines && (in_front || !hide_behind_earth) {
                        let r = (sat.x * sat.x + sat.y * sat.y + sat.z * sat.z).sqrt();
                        let scale = planet_radius / r;
                        let (gx, gy, _) = rotate_point_matrix(
                            sat.x * scale,
                            sat.y * scale,
                            sat.z * scale,
                            &satellite_rotation,
                        );
                        let alt_color = egui::Color32::from_rgba_unmultiplied(
                            color.r(),
                            color.g(),
                            color.b(),
                            180,
                        );
                        plot_ui.line(
                            egui_plot::Line::new("", PlotPoints::new(vec![[rx, ry], [gx, gy]]))
                                .color(alt_color)
                                .width(altitude_line_width * zoom_factor),
                        );
                    }
                }
            }
        }
    });

    if !conjunction_heatmap.is_empty() {
        let earth_r_sq = (planet_radius * EARTH_VISUAL_SCALE).powi(2);
        for (conj, _threshold) in conjunction_heatmap {
            let (rx1, ry1, rz1) = rotate_point_matrix(
                conj.pos_a[0],
                conj.pos_a[1],
                conj.pos_a[2],
                &satellite_rotation,
            );
            let (rx2, ry2, rz2) = rotate_point_matrix(
                conj.pos_b[0],
                conj.pos_b[1],
                conj.pos_b[2],
                &satellite_rotation,
            );
            let visible1 = rz1 >= 0.0 || (rx1 * rx1 + ry1 * ry1) >= earth_r_sq;
            let visible2 = rz2 >= 0.0 || (rx2 * rx2 + ry2 * ry2) >= earth_r_sq;
            let color = egui::Color32::from_rgb(255, 0, 0);
            let dim_color = egui::Color32::from_rgba_unmultiplied(255, 0, 0, 80);
            let dot_radius = 5.0_f32;
            if hide_behind_earth {
                if let Some((p1, p2)) =
                    clip_link_at_earth(rx1, ry1, rz1, visible1, rx2, ry2, rz2, visible2, earth_r_sq)
                {
                    let sp1 = response
                        .transform
                        .position_from_point(&egui_plot::PlotPoint::new(p1[0], p1[1]));
                    let sp2 = response
                        .transform
                        .position_from_point(&egui_plot::PlotPoint::new(p2[0], p2[1]));
                    ui.painter()
                        .line_segment([sp1, sp2], egui::Stroke::new(1.5, color));
                    if visible1 {
                        ui.painter().circle_filled(sp1, dot_radius, color);
                    }
                    if visible2 {
                        ui.painter().circle_filled(sp2, dot_radius, color);
                    }
                }
            } else {
                let c = if visible1 && visible2 {
                    color
                } else {
                    dim_color
                };
                let sp1 = response
                    .transform
                    .position_from_point(&egui_plot::PlotPoint::new(rx1, ry1));
                let sp2 = response
                    .transform
                    .position_from_point(&egui_plot::PlotPoint::new(rx2, ry2));
                ui.painter()
                    .line_segment([sp1, sp2], egui::Stroke::new(1.5, c));
                if visible1 {
                    ui.painter().circle_filled(sp1, dot_radius, color);
                }
                if visible2 {
                    ui.painter().circle_filled(sp2, dot_radius, c);
                }
            }
        }
    }

    if !enabled_moons.is_empty() {
        let plot_rect = response.response.rect;
        let px_per_km = width as f64 * 0.5 / margin;
        let camera_alt = moon_camera_distance_km;
        let earth_r_sq = planet_radius * planet_radius;
        for &(moon_body, orbit_km, period_days, default_incl_rad) in body_key.0.moons() {
            if !enabled_moons.contains(&moon_body) {
                continue;
            }
            let incl_rad = moon_inclination_override
                .map(|d| d.to_radians())
                .unwrap_or(default_incl_rad);
            let color = moon_body.display_color();

            let moon_r_km = moon_body.radius_km();
            let angle = 2.0 * PI * time / (period_days * 86400.0);
            let x = orbit_km * angle.cos();
            let y_orbit = orbit_km * angle.sin();
            let y = y_orbit * incl_rad.cos();
            let z = y_orbit * incl_rad.sin();
            let (rx, ry, rz) = rotate_point_matrix(x, y, z, &satellite_rotation);

            let dist = camera_alt - rz;
            let moon_behind = dist <= 0.0;

            let altitude = 10000.0 / zoom;
            let fade = if rz > 0.0 && altitude < 30000.0 {
                ((altitude - 10000.0) / 20000.0).clamp(0.0, 1.0)
            } else {
                1.0
            };
            if fade <= 0.0 {
                continue;
            }
            let faded_color = egui::Color32::from_rgba_unmultiplied(
                color.r(),
                color.g(),
                color.b(),
                (color.a() as f64 * fade) as u8,
            );

            if show_moon_orbits {
                let steps = 512;
                let mut prev: Option<([f64; 2], bool, f64, f64)> = None;
                for i in 0..=steps {
                    let a = 2.0 * PI * i as f64 / steps as f64;
                    let ox = orbit_km * a.cos();
                    let oy_orbit = orbit_km * a.sin();
                    let oy = oy_orbit * incl_rad.cos();
                    let oz = oy_orbit * incl_rad.sin();
                    let (orx, ory, orz) = rotate_point_matrix(ox, oy, oz, &satellite_rotation);
                    let d = camera_alt - orz;
                    if d <= 0.0 {
                        prev = None;
                        continue;
                    }
                    let pt_fade = if orz > 0.0 && altitude < 30000.0 {
                        ((altitude - 10000.0) / 20000.0).clamp(0.0, 1.0)
                    } else {
                        1.0
                    };
                    let behind_planet = orz < 0.0 && (orx * orx + ory * ory) < earth_r_sq;
                    let scale = camera_alt / d;
                    let px = orx * scale;
                    let py = ory * scale;
                    if let Some((p, p_planet, pd, pf)) = prev {
                        let seg_fade = (pf + pt_fade) * 0.5;
                        if seg_fade > 0.0 {
                            let s1 = response
                                .transform
                                .position_from_point(&egui_plot::PlotPoint::new(p[0], p[1]));
                            let s2 = response
                                .transform
                                .position_from_point(&egui_plot::PlotPoint::new(px, py));
                            if let Some((c1, c2)) = clip_line_to_rect(s1, s2, plot_rect) {
                                let avg_d = (pd + d) * 0.5;
                                let w = (camera_alt / avg_d).max(0.5) as f32;
                                if behind_planet || p_planet {
                                    let a = (60.0 * seg_fade) as u8;
                                    let c = egui::Color32::from_rgba_unmultiplied(
                                        color.r(),
                                        color.g(),
                                        color.b(),
                                        a,
                                    );
                                    ui.painter().line_segment(
                                        [c1, c2],
                                        egui::Stroke::new((w * 0.5).max(0.3), c),
                                    );
                                } else {
                                    let a = (color.a() as f64 * seg_fade) as u8;
                                    let c = egui::Color32::from_rgba_unmultiplied(
                                        color.r(),
                                        color.g(),
                                        color.b(),
                                        a,
                                    );
                                    ui.painter().line_segment([c1, c2], egui::Stroke::new(w, c));
                                }
                            }
                        }
                    }
                    prev = Some(([px, py], behind_planet, d, pt_fade));
                }
            }

            if moon_behind {
                continue;
            }
            let proj_r_sq = rx * rx + ry * ry;
            if rz < 0.0 && proj_r_sq < earth_r_sq {
                continue;
            }
            let moon_scale = camera_alt / dist;
            let mrx = rx * moon_scale;
            let mry = ry * moon_scale;
            let sp = response
                .transform
                .position_from_point(&egui_plot::PlotPoint::new(mrx, mry));
            if !plot_rect.contains(sp) {
                continue;
            }

            if show_moon_lines {
                let d = (rx * rx + ry * ry + rz * rz).sqrt();
                if d > planet_radius {
                    let t = planet_radius / d;
                    let sx = rx * t;
                    let sy = ry * t;
                    let surface = response
                        .transform
                        .position_from_point(&egui_plot::PlotPoint::new(sx, sy));
                    let line_a = (120.0 * fade) as u8;
                    let line_color = egui::Color32::from_rgba_unmultiplied(
                        color.r(),
                        color.g(),
                        color.b(),
                        line_a,
                    );
                    ui.painter()
                        .line_segment([surface, sp], egui::Stroke::new(0.5, line_color));
                }
            }

            let dot_r = (moon_r_km * camera_alt / dist * px_per_km) as f32;
            if let Some(handle) = moon_handles.get(&moon_body) {
                let rect = egui::Rect::from_center_size(sp, egui::Vec2::splat(dot_r * 2.0));
                let tint =
                    egui::Color32::from_rgba_unmultiplied(255, 255, 255, (255.0 * fade) as u8);
                ui.painter().image(
                    handle.id(),
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    tint,
                );
            } else {
                ui.painter().circle_filled(sp, dot_r, faded_color);
            }
            if show_moon_labels {
                let label_a = (255.0 * fade) as u8;
                let label_color =
                    egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), label_a);
                let font = egui::FontId::proportional(11.0);
                let galley = ui.painter().layout_no_wrap(
                    format!("{} ({:.0} km)", moon_body.label(), orbit_km),
                    font,
                    label_color,
                );
                let text_pos = sp + egui::Vec2::new(dot_r + 4.0, -galley.size().y * 0.5);
                let bg_a = (160.0 * fade) as u8;
                let bg = egui::Rect::from_min_size(text_pos, galley.size()).expand(2.0);
                ui.painter().rect_filled(
                    bg,
                    2.0,
                    egui::Color32::from_rgba_unmultiplied(0, 0, 0, bg_a),
                );
                ui.painter().galley(text_pos, galley, label_color);
            }
        }
    }

    let label_font_size = (14.0 * zoom as f32).clamp(10.0, 28.0);
    for (pos, text) in &path_distance_labels {
        let plot_pt = egui_plot::PlotPoint::new(pos[0], pos[1]);
        let screen_pos = response.transform.position_from_point(&plot_pt);
        let galley = ui.painter().layout_no_wrap(
            text.clone(),
            egui::FontId::proportional(12.0),
            egui::Color32::WHITE,
        );
        let text_pos = screen_pos - galley.size() * 0.5;
        let bg_rect = egui::Rect::from_min_size(text_pos, galley.size()).expand(4.0);
        ui.painter().rect_filled(
            bg_rect,
            4.0,
            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 200),
        );
        ui.painter().galley(text_pos, galley, egui::Color32::WHITE);
    }

    let mut label_rects: Vec<(egui::Rect, bool, usize)> = Vec::new();
    for (pos, name, color, is_gs, idx) in &surface_labels {
        let plot_pt = egui_plot::PlotPoint::new(pos[0], pos[1]);
        let screen_pos = response.transform.position_from_point(&plot_pt);
        let galley = ui.painter().layout_no_wrap(
            name.clone(),
            egui::FontId::proportional(label_font_size),
            *color,
        );
        let text_pos =
            screen_pos + egui::Vec2::new(-(galley.size().x / 2.0), -galley.size().y - 4.0);
        let bg_rect = egui::Rect::from_min_size(text_pos, galley.size()).expand(3.0);
        ui.painter().rect_filled(
            bg_rect,
            3.0,
            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180),
        );
        ui.painter().galley(text_pos, galley, *color);
        label_rects.push((bg_rect, *is_gs, *idx));
    }

    for (pos, text, color) in &spacecomp_role_labels {
        let plot_pt = egui_plot::PlotPoint::new(pos[0], pos[1]);
        let screen_pos = response.transform.position_from_point(&plot_pt);
        let galley = ui.painter().layout_no_wrap(
            text.to_string(),
            egui::FontId::proportional(12.0),
            egui::Color32::WHITE,
        );
        let text_pos =
            screen_pos + egui::Vec2::new(scaled_sat_radius * 3.0, -galley.size().y / 2.0);
        let bg_rect = egui::Rect::from_min_size(text_pos, galley.size()).expand(4.0);
        ui.painter().rect_filled(
            bg_rect,
            4.0,
            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 200),
        );
        ui.painter().galley(text_pos, galley, *color);
    }

    for (pos, count, color) in &device_cluster_labels {
        let plot_pt = egui_plot::PlotPoint::new(pos[0], pos[1]);
        let screen_pos = response.transform.position_from_point(&plot_pt);
        let text = format!("{}", count);
        let galley = ui.painter().layout_no_wrap(
            text,
            egui::FontId::proportional(11.0),
            egui::Color32::WHITE,
        );
        let text_pos =
            screen_pos + egui::Vec2::new(-(galley.size().x / 2.0), -(galley.size().y / 2.0));
        let bg_rect = egui::Rect::from_min_size(text_pos, galley.size()).expand(2.0);
        ui.painter().rect_filled(
            bg_rect,
            3.0,
            egui::Color32::from_rgba_unmultiplied(color.r() / 3, color.g() / 3, color.b() / 3, 200),
        );
        ui.painter().galley(text_pos, galley, egui::Color32::WHITE);
    }

    if constellations.len() > 1 {
        let plot_rect = response.response.rect;
        let mut y = plot_rect.min.y + 8.0;
        let x = plot_rect.min.x + 8.0;
        let font = egui::FontId::proportional(12.0);
        let square_size = 10.0;
        let mut seen = std::collections::HashSet::new();
        for (_, _, color_offset, tle_kind, _, name) in constellations {
            if !seen.insert((name.as_str(), *color_offset)) {
                continue;
            }
            let color = if *tle_kind == 3 {
                egui::Color32::from_rgb(255, 60, 60)
            } else if *tle_kind == 2 {
                // Mirror the debris X palette defined in the satellite rendering
                // block so the legend swatch matches the on-globe colour.
                if name.starts_with("Fengyun") {
                    egui::Color32::from_rgb(255, 70, 70)
                } else if name.starts_with("Cosmos 2251") {
                    egui::Color32::from_rgb(70, 230, 90)
                } else if name.starts_with("Iridium 33") {
                    egui::Color32::from_rgb(90, 160, 255)
                } else if name.starts_with("Cosmos 1408") {
                    egui::Color32::from_rgb(255, 220, 70)
                } else {
                    plane_color(*color_offset)
                }
            } else {
                plane_color(*color_offset)
            };
            let square_rect = egui::Rect::from_min_size(
                egui::pos2(x, y + 1.0),
                egui::vec2(square_size, square_size),
            );
            ui.painter().rect_filled(square_rect, 2.0, color);
            let galley =
                ui.painter()
                    .layout_no_wrap(name.clone(), font.clone(), egui::Color32::WHITE);
            let text_pos = egui::pos2(x + square_size + 4.0, y - 1.0);
            let bg_rect = egui::Rect::from_min_max(
                egui::pos2(x - 2.0, y - 2.0),
                egui::pos2(
                    text_pos.x + galley.size().x + 2.0,
                    y + galley.size().y + 2.0,
                ),
            );
            ui.painter().rect_filled(
                bg_rect,
                3.0,
                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 160),
            );
            ui.painter().rect_filled(square_rect, 2.0, color);
            if *tle_kind == 1 {
                let inset = 2.5;
                let inner = square_rect.shrink(inset);
                ui.painter().rect_filled(inner, 1.0, egui::Color32::BLACK);
            } else if *tle_kind == 2 {
                let c = square_rect.center();
                let h = square_size * 0.35;
                ui.painter().line_segment(
                    [c - egui::vec2(h, h), c + egui::vec2(h, h)],
                    egui::Stroke::new(1.5, egui::Color32::BLACK),
                );
                ui.painter().line_segment(
                    [c + egui::vec2(-h, h), c + egui::vec2(h, -h)],
                    egui::Stroke::new(1.5, egui::Color32::BLACK),
                );
            } else if *tle_kind == 3 {
                let c = square_rect.center();
                ui.painter().circle_filled(c, 3.0, egui::Color32::BLACK);
            }
            ui.painter().galley(text_pos, galley, egui::Color32::WHITE);
            y += 16.0;
        }
    }

    for (constellation, positions, color_offset, _tle_kind, orig_idx, _) in constellations {
        for sat in positions {
            for cam in satellite_cameras.iter_mut() {
                if cam.constellation_idx == *orig_idx
                    && cam.plane == sat.plane
                    && cam.sat_index == sat.sat_index
                {
                    let (rx, ry, _) = rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                    let plot_pt = egui_plot::PlotPoint::new(rx, ry);
                    let screen_pos = response.transform.position_from_point(&plot_pt);
                    cam.screen_pos = Some(screen_pos);

                    let color = plane_color(if single_color {
                        *color_offset
                    } else {
                        sat.plane + color_offset
                    });
                    ui.painter().circle_stroke(
                        screen_pos,
                        scaled_sat_radius * 2.5,
                        egui::Stroke::new(2.0, color),
                    );

                    if show_sat_labels {
                        let r = (sat.x * sat.x + sat.y * sat.y + sat.z * sat.z).sqrt();
                        let alt_km = r - constellation.planet_radius;
                        let vel_km_s = (constellation.planet_mu / r).sqrt();
                        let inv_body = body_rotation.transpose();
                        let body_pos = inv_body * Vector3::new(sat.x, sat.y, sat.z);
                        let ground_lat = (body_pos.y / r).asin().to_degrees();
                        let ground_lon = (-body_pos.z).atan2(body_pos.x).to_degrees();
                        let id = if !cam.label.is_empty() {
                            cam.label.clone()
                        } else if let Some(name) = &sat.name {
                            name.clone()
                        } else {
                            format!("P{}S{}", sat.plane, sat.sat_index)
                        };
                        let mut text = if let (Some(inc), Some(mm)) =
                            (sat.tle_inclination_deg, sat.tle_mean_motion)
                        {
                            let revs_per_day = mm;
                            let period_min = 1440.0 / revs_per_day;
                            format!(
                                "{}  {:.1}° {:.1}°\n{:.0} km  {:.2} km/s\nInc {:.1}°  {:.2} rev/day\nPeriod {:.1} min",
                                id,
                                ground_lat, ground_lon,
                                alt_km, vel_km_s,
                                inc, revs_per_day,
                                period_min,
                            )
                        } else {
                            format!(
                                "{}  {:.1}° {:.1}°\n{:.0} km  {:.2} km/s",
                                id, ground_lat, ground_lon, alt_km, vel_km_s,
                            )
                        };
                        let flat_idx = sat.plane * constellation.sats_per_plane() + sat.sat_index;
                        if let Some(&(soc, temp_k, is_dead)) =
                            physics_info.get(&(*orig_idx, flat_idx))
                        {
                            if is_dead {
                                text.push_str("\n⚠ DEAD");
                            } else {
                                text.push_str(&format!(
                                    "\nBattery: {:.0}%  T: {:.0} K",
                                    soc * 100.0,
                                    temp_k
                                ));
                            }
                        }
                        let font = egui::FontId::proportional(12.0);
                        let galley = ui
                            .painter()
                            .layout_no_wrap(text, font, egui::Color32::WHITE);
                        let text_pos = screen_pos
                            + egui::Vec2::new(scaled_sat_radius * 3.0, -galley.size().y / 2.0);
                        let bg_rect =
                            egui::Rect::from_min_size(text_pos, galley.size()).expand(4.0);
                        ui.painter().rect_filled(
                            bg_rect,
                            4.0,
                            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 200),
                        );
                        ui.painter().galley(text_pos, galley, egui::Color32::WHITE);
                    }
                }
            }
        }
    }

    if !pinned_isl_overlays.is_empty() {
        let pin_w = (scaled_link_width + 3.0).max(3.5);
        let pin_halo = egui::Color32::from_rgba_unmultiplied(120, 220, 255, 90);
        let pin_core = egui::Color32::from_rgb(180, 240, 255);
        for (p1, p2) in &pinned_isl_overlays {
            let s1 = response
                .transform
                .position_from_point(&egui_plot::PlotPoint::new(p1[0], p1[1]));
            let s2 = response
                .transform
                .position_from_point(&egui_plot::PlotPoint::new(p2[0], p2[1]));
            ui.painter()
                .line_segment([s1, s2], egui::Stroke::new(pin_w + 2.0, pin_halo));
            ui.painter()
                .line_segment([s1, s2], egui::Stroke::new(pin_w, pin_core));
        }
    }

    let mut hovering_satellite = false;
    if let Some(hover_pos) = response.response.hover_pos() {
        let plot_pos = response.transform.value_from_position(hover_pos);
        let hover_threshold = margin * 0.025;

        'hover: for (constellation, positions, color_offset, _, orig_idx, _label) in constellations
        {
            for sat in positions {
                let (rx, ry, rz) = rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                let earth_r_sq = (planet_radius * EARTH_VISUAL_SCALE).powi(2);
                let visible = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;
                if !visible && hide_behind_earth {
                    continue;
                }
                let dx = rx - plot_pos.x;
                let dy = ry - plot_pos.y;
                if dx * dx + dy * dy < hover_threshold * hover_threshold {
                    let plot_pt = egui_plot::PlotPoint::new(rx, ry);
                    let screen_pt = response.transform.position_from_point(&plot_pt);
                    let color = plane_color(if single_color {
                        *color_offset
                    } else {
                        sat.plane + color_offset
                    });
                    ui.painter().circle_stroke(
                        screen_pt,
                        scaled_sat_radius * 2.0,
                        egui::Stroke::new(2.0, color),
                    );
                    if show_sat_labels {
                        let id = match &sat.name {
                            Some(name) => name.clone(),
                            None => format!("P{}S{}", sat.plane, sat.sat_index),
                        };
                        let r = (sat.x * sat.x + sat.y * sat.y + sat.z * sat.z).sqrt();
                        let alt_km = r - planet_radius;
                        let vel_km_s = (constellation.planet_mu / r).sqrt();
                        let inv_body = body_rotation.transpose();
                        let body_pos = inv_body * Vector3::new(sat.x, sat.y, sat.z);
                        let ground_lat = (body_pos.y / r).asin().to_degrees();
                        let ground_lon = (-body_pos.z).atan2(body_pos.x).to_degrees();
                        let mut tip = if let (Some(inc), Some(mm)) =
                            (sat.tle_inclination_deg, sat.tle_mean_motion)
                        {
                            let period_min = 1440.0 / mm;
                            format!(
                                "{}  {:.1}° {:.1}°\n{:.0} km  {:.2} km/s\nInc {:.1}°  {:.2} rev/day\nPeriod {:.1} min",
                                id, ground_lat, ground_lon, alt_km, vel_km_s, inc, mm, period_min,
                            )
                        } else {
                            format!(
                                "{}  {:.1}° {:.1}°\n{:.0} km  {:.2} km/s",
                                id, ground_lat, ground_lon, alt_km, vel_km_s,
                            )
                        };
                        let lifetime_s = orbital_lifetime_seconds(
                            alt_km,
                            constellation.ballistic_coeff,
                            planet_radius,
                            constellation.planet_mu,
                        );
                        tip.push_str(&format!("\nLifetime: {}", format_duration(lifetime_s)));
                        let flat_idx = sat.plane * constellation.sats_per_plane() + sat.sat_index;
                        if let Some(&(soc, temp_k, is_dead)) =
                            physics_info.get(&(*orig_idx, flat_idx))
                        {
                            if is_dead {
                                tip.push_str("\n⚠ DEAD");
                            } else {
                                tip.push_str(&format!(
                                    "\nBattery: {:.0}%  T: {:.0} K",
                                    soc * 100.0,
                                    temp_k
                                ));
                            }
                        }
                        let font = egui::FontId::proportional(12.0);
                        let galley = ui.painter().layout_no_wrap(tip, font, egui::Color32::WHITE);
                        let tip_pos = screen_pt
                            - egui::Vec2::new(
                                galley.size().x * 0.5,
                                galley.size().y + scaled_sat_radius * 2.0 + 8.0,
                            );
                        let rect = egui::Rect::from_min_size(tip_pos, galley.size()).expand(4.0);
                        ui.painter().rect_filled(
                            rect,
                            3.0,
                            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 200),
                        );
                        ui.painter().galley(tip_pos, galley, egui::Color32::WHITE);
                    }
                    hovering_satellite = true;
                    break 'hover;
                }
            }
        }

        let mut hovering_isl = false;
        if !hovering_satellite && !hover_isl_segments.is_empty() {
            let threshold_px = 6.0_f32;
            let mut closest: Option<(f32, f64, crate::config::LinkBudget, egui::Pos2, egui::Pos2)> =
                None;
            for (p1, p2, dist_km, lb, _id) in &hover_isl_segments {
                let s1 = response
                    .transform
                    .position_from_point(&egui_plot::PlotPoint::new(p1[0], p1[1]));
                let s2 = response
                    .transform
                    .position_from_point(&egui_plot::PlotPoint::new(p2[0], p2[1]));
                let d = dist_pos2_to_segment(hover_pos, s1, s2);
                if d < threshold_px && closest.as_ref().map_or(true, |(d2, _, _, _, _)| d < *d2) {
                    closest = Some((d, *dist_km, *lb, s1, s2));
                }
            }
            if let Some((_, dist_km, lb, s1, s2)) = closest {
                let highlight_w = (scaled_link_width + 3.0).max(3.5);
                let halo = egui::Color32::from_rgba_unmultiplied(255, 220, 100, 90);
                let core = egui::Color32::from_rgb(255, 240, 160);
                ui.painter()
                    .line_segment([s1, s2], egui::Stroke::new(highlight_w + 2.0, halo));
                ui.painter()
                    .line_segment([s1, s2], egui::Stroke::new(highlight_w, core));

                let latency_ms = dist_km / 299.792458;
                let capacity_gbps = lb.capacity_bps(dist_km) / 1e9;
                let tip = format!(
                    "ISL (laser)\nDistance: {:.0} km\nOne-way latency: {:.2} ms\nShannon C: {:.1} Gbps\nB = {:.1} GHz, P = {:.1} W, G = {:.1} dBi, λ = {} nm",
                    dist_km, latency_ms, capacity_gbps,
                    lb.bandwidth_ghz, lb.tx_power_w, lb.antenna_gain_dbi, lb.wavelength_nm as i32,
                );
                let font = egui::FontId::proportional(12.0);
                let galley = ui.painter().layout_no_wrap(tip, font, egui::Color32::WHITE);
                let tip_pos = hover_pos + egui::Vec2::new(15.0, -15.0 - galley.size().y);
                let rect = egui::Rect::from_min_size(tip_pos, galley.size()).expand(4.0);
                ui.painter().rect_filled(
                    rect,
                    3.0,
                    egui::Color32::from_rgba_unmultiplied(0, 0, 0, 200),
                );
                ui.painter().galley(tip_pos, galley, egui::Color32::WHITE);
                hovering_isl = true;
            }
        }

        let px = plot_pos.x;
        let py = plot_pos.y;
        let r_sq = planet_radius * planet_radius;
        if !hovering_isl && px * px + py * py <= r_sq {
            let pz = (r_sq - px * px - py * py).sqrt();
            let surface_rot = if earth_fixed_camera {
                rotation
            } else {
                rotation * *body_rotation
            };
            let inv = surface_rot.transpose();
            let orig = inv * Vector3::new(px, py, pz);
            let lat = (orig.y / planet_radius).asin().to_degrees();
            let lon = -(orig.z.atan2(orig.x)).to_degrees();
            let text = format!("{:.1}° {:.1}°", lat, lon);
            let font = egui::FontId::proportional(12.0);
            let text_pos = hover_pos + egui::Vec2::new(15.0, -15.0);
            let galley = ui
                .painter()
                .layout_no_wrap(text, font, egui::Color32::WHITE);
            let rect = egui::Rect::from_min_size(
                text_pos - egui::Vec2::new(0.0, galley.size().y),
                galley.size(),
            )
            .expand(3.0);
            ui.painter().rect_filled(
                rect,
                3.0,
                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180),
            );
            ui.painter().galley(
                text_pos - egui::Vec2::new(0.0, galley.size().y),
                galley,
                egui::Color32::WHITE,
            );
        } else if !hovering_satellite && !hovering_isl {
            let dist = (px * px + py * py).sqrt();
            let alt_km = (dist - planet_radius) / planet_radius * 6371.0;
            if alt_km > 0.0 {
                let text = format!("{:.0} km", alt_km);
                let font = egui::FontId::proportional(12.0);
                let text_pos = hover_pos + egui::Vec2::new(15.0, -15.0);
                let galley = ui
                    .painter()
                    .layout_no_wrap(text, font, egui::Color32::WHITE);
                let rect = egui::Rect::from_min_size(
                    text_pos - egui::Vec2::new(0.0, galley.size().y),
                    galley.size(),
                )
                .expand(3.0);
                ui.painter().rect_filled(
                    rect,
                    3.0,
                    egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180),
                );
                ui.painter().galley(
                    text_pos - egui::Vec2::new(0.0, galley.size().y),
                    galley,
                    egui::Color32::WHITE,
                );
            }
        }
    }

    if !hovering_satellite {
        if let (Some(hover_pos), Some(rc)) = (response.response.hover_pos(), radiation) {
            let plot_pos = response.transform.value_from_position(hover_pos);
            let px = plot_pos.x;
            let py = plot_pos.y;
            let r_sq = planet_radius * planet_radius;
            if px * px + py * py <= r_sq {
                let pz = (r_sq - px * px - py * py).sqrt();
                let surface_rot = if earth_fixed_camera {
                    rotation
                } else {
                    rotation * *body_rotation
                };
                let inv = surface_rot.transpose();
                let orig = inv * Vector3::new(px, py, pz);
                let lat = (orig.y / planet_radius).asin().to_degrees();
                let lon = -(orig.z.atan2(orig.x)).to_degrees();
                let colat = (orig.y / planet_radius).acos();
                let elon = (-orig.z).atan2(orig.x);
                let sphere_r = planet_radius + rc.heatmap_altitude_km;

                let tip = match rc.heatmap_mode {
                    crate::config::HeatmapMode::IgrfField => {
                        let f = crate::igrf::igrf_field_nt(sphere_r, colat, elon);
                        Some(format!("{:.1}° {:.1}°  F: {:.0} nT", lat, lon, f))
                    }
                    crate::config::HeatmapMode::IgrfRadiation => {
                        if let Some((_, _, ref g)) = rc.igrf_rad_cache {
                            let (p, e) = g.lookup(colat, elon);
                            Some(format!(
                                "{:.1}° {:.1}°\nProtons: {:.3}  Electrons: {:.3}",
                                lat, lon, p, e
                            ))
                        } else {
                            None
                        }
                    }
                    crate::config::HeatmapMode::FieldStrength => {
                        let b0 = 30115.0;
                        let r_er = sphere_r / 6371.2;
                        let sin_ml = orig.y / planet_radius;
                        let f = b0 / r_er.powi(3) * (1.0 + 3.0 * sin_ml * sin_ml).sqrt();
                        Some(format!("{:.1}° {:.1}°  F: {:.0} nT (dipole)", lat, lon, f))
                    }
                    _ => None,
                };

                if let Some(text) = tip {
                    let galley = ui.painter().layout_no_wrap(
                        text,
                        egui::FontId::monospace(12.0),
                        egui::Color32::WHITE,
                    );
                    let text_pos = hover_pos + egui::Vec2::new(15.0, -15.0);
                    let rect = egui::Rect::from_min_size(
                        text_pos - egui::Vec2::new(0.0, galley.size().y),
                        galley.size(),
                    )
                    .expand(4.0);
                    ui.painter().rect_filled(
                        rect,
                        4.0,
                        egui::Color32::from_rgba_unmultiplied(0, 0, 0, 200),
                    );
                    ui.painter().galley(
                        text_pos - egui::Vec2::new(0.0, galley.size().y),
                        galley,
                        egui::Color32::WHITE,
                    );
                }
            }
        }
    }

    if response.response.is_pointer_button_down_on() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
    } else if !hovering_satellite {
        if response.response.hover_pos().is_some() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
        }
    }

    if response.response.double_clicked() {
        let up_screen = rotation * Vector3::new(0.0, 1.0, 0.0);
        let bearing = up_screen.x.atan2(up_screen.y);
        if bearing.abs() > 1e-6 {
            let cb = bearing.cos();
            let sb = bearing.sin();
            rotation = Matrix3::new(cb, sb, 0.0, -sb, cb, 0.0, 0.0, 0.0, 1.0) * rotation;
        }
    }

    if response.response.drag_started() {
        if let Some(pos) = response.response.interact_pointer_pos() {
            let mut found = false;
            for (rect, is_gs, idx) in &label_rects {
                if rect.contains(pos) {
                    if *is_gs && ground_stations_locked {
                        continue;
                    }
                    *dragging_place = Some((drag_tab_planet.0, drag_tab_planet.1, *is_gs, *idx));
                    found = true;
                    break;
                }
            }
            if !found {
                *dragging_place = None;
            }
        }
    }

    let is_dragging_place = dragging_place.map_or(false, |(t, p, _, _)| {
        t == drag_tab_planet.0 && p == drag_tab_planet.1
    });

    if response.response.dragged() && !response.response.drag_started() {
        if is_dragging_place {
            if let Some(pos) = response.response.interact_pointer_pos() {
                let plot_pos = response.transform.value_from_position(pos);
                let px = plot_pos.x;
                let py = plot_pos.y;
                let r_sq = planet_radius * planet_radius;
                if px * px + py * py <= r_sq {
                    let pz = (r_sq - px * px - py * py).sqrt();
                    let surface_rot = if earth_fixed_camera {
                        rotation
                    } else {
                        rotation * *body_rotation
                    };
                    let inv = surface_rot.transpose();
                    let orig = inv * Vector3::new(px, py, pz);
                    let lat = (orig.y / planet_radius).asin().to_degrees();
                    let lon = -(orig.z.atan2(orig.x)).to_degrees();
                    if let Some((_, _, is_gs, idx)) = *dragging_place {
                        if is_gs {
                            if !ground_stations_locked {
                                if let Some(gs) = ground_stations.get_mut(idx) {
                                    gs.lat = lat;
                                    gs.lon = lon;
                                }
                            }
                        } else if let Some(aoi) = areas_of_interest.get_mut(idx) {
                            aoi.lat = lat;
                            aoi.lon = lon;
                        }
                    }
                }
            }
        } else if let Some(pos) = response.response.interact_pointer_pos() {
            let drag = response.response.drag_delta();
            let prev_pos = pos - drag;
            let cur = response.transform.value_from_position(pos);
            let prev = response.transform.value_from_position(prev_pos);
            let r = planet_radius;
            let r_sq = r * r;
            let to_sphere = |px: f64, py: f64| -> Vector3<f64> {
                let d_sq = px * px + py * py;
                if d_sq <= r_sq {
                    Vector3::new(px, py, (r_sq - d_sq).sqrt())
                } else {
                    let s = r / d_sq.sqrt();
                    Vector3::new(px * s, py * s, 0.0)
                }
            };
            let a = to_sphere(prev.x, prev.y).normalize();
            let b = to_sphere(cur.x, cur.y).normalize();
            let cross = a.cross(&b);
            let cross_len = cross.norm();
            if cross_len > 1e-12 {
                let axis = cross / cross_len;
                let angle = cross_len.atan2(a.dot(&b));
                let c = angle.cos();
                let s = angle.sin();
                let t = 1.0 - c;
                let (x, y, z) = (axis.x, axis.y, axis.z);
                let rot = Matrix3::new(
                    t * x * x + c,
                    t * x * y - s * z,
                    t * x * z + s * y,
                    t * x * y + s * z,
                    t * y * y + c,
                    t * y * z - s * x,
                    t * x * z - s * y,
                    t * y * z + s * x,
                    t * z * z + c,
                );
                rotation = rot * rotation;
            }
        }
    }

    if !response.response.dragged()
        && dragging_place
            .is_some_and(|(t, p, _, _)| t == drag_tab_planet.0 && p == drag_tab_planet.1)
    {
        *dragging_place = None;
    }

    if response.response.clicked() {
        if let Some(pos) = response.response.interact_pointer_pos() {
            let mut handled = false;
            for (rect, is_gs, idx) in &label_rects {
                if rect.contains(pos) {
                    *label_click_request = Some((*is_gs, *idx, pos));
                    handled = true;
                    break;
                }
            }
            if !handled {
                let plot_pos = response.transform.value_from_position(pos);
                let click_x = plot_pos.x;
                let click_y = plot_pos.y;
                let click_threshold = margin * 0.03;

                let mut sat_hit = false;
                'outer: for (_constellation, positions, _color_offset, _, orig_idx, _) in
                    constellations
                {
                    for sat in positions {
                        let (rx, ry, rz) =
                            rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                        let earth_r_sq = (planet_radius * EARTH_VISUAL_SCALE).powi(2);
                        let visible = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;
                        if !visible && hide_behind_earth {
                            continue;
                        }
                        let dx = rx - click_x;
                        let dy = ry - click_y;
                        if dx * dx + dy * dy < click_threshold * click_threshold {
                            let existing = satellite_cameras.iter().find(|c| {
                                c.constellation_idx == *orig_idx
                                    && c.plane == sat.plane
                                    && c.sat_index == sat.sat_index
                            });
                            if let Some(cam) = existing {
                                cameras_to_remove.push(cam.id);
                            } else {
                                let in_pending = pending_cameras.iter().any(|c| {
                                    c.constellation_idx == *orig_idx
                                        && c.plane == sat.plane
                                        && c.sat_index == sat.sat_index
                                });
                                if !in_pending {
                                    *camera_id_counter += 1;
                                    pending_cameras.push(SatelliteCamera {
                                        id: *camera_id_counter,
                                        label: format!(
                                            "Sat {}-{}",
                                            sat.plane + 1,
                                            sat.sat_index + 1
                                        ),
                                        constellation_idx: *orig_idx,
                                        plane: sat.plane,
                                        sat_index: sat.sat_index,
                                        screen_pos: None,
                                    });
                                }
                            }
                            sat_hit = true;
                            break 'outer;
                        }
                    }
                }

                if !sat_hit && !hover_isl_segments.is_empty() {
                    let threshold_px = 6.0_f32;
                    let mut closest: Option<(f32, crate::config::PinnedIsl)> = None;
                    for (p1, p2, _dist_km, _lb, id) in &hover_isl_segments {
                        let s1 = response
                            .transform
                            .position_from_point(&egui_plot::PlotPoint::new(p1[0], p1[1]));
                        let s2 = response
                            .transform
                            .position_from_point(&egui_plot::PlotPoint::new(p2[0], p2[1]));
                        let d = dist_pos2_to_segment(pos, s1, s2);
                        if d < threshold_px && closest.as_ref().map_or(true, |(d2, _)| d < *d2) {
                            closest = Some((d, *id));
                        }
                    }
                    if let Some((_, id)) = closest {
                        if !pinned_isls.remove(&id) {
                            pinned_isls.insert(id);
                        }
                    }
                }
            }
        }
    }

    if response.response.secondary_clicked() {
        if let Some(pos) = response.response.interact_pointer_pos() {
            let plot_pos = response.transform.value_from_position(pos);
            let px = plot_pos.x;
            let py = plot_pos.y;
            let r_sq = planet_radius * planet_radius;
            if px * px + py * py <= r_sq {
                let pz = (r_sq - px * px - py * py).sqrt();
                let surface_rot = if earth_fixed_camera {
                    rotation
                } else {
                    rotation * *body_rotation
                };
                let inv = surface_rot.transpose();
                let orig = inv * Vector3::new(px, py, pz);
                let lat = (orig.y / planet_radius).asin().to_degrees();
                let lon = -(orig.z.atan2(orig.x)).to_degrees();
                *context_menu_request = Some((pos, lat, lon));
            }
        }
    }

    if response.response.hovered() {
        let north_up = north_up && (10000.0 / zoom) <= 30000.0;
        let scroll_delta = ui.input(|i| i.smooth_scroll_delta);
        let zd = ui.input(|i| i.zoom_delta());
        let rot_delta = ui.input(|i| i.rotation_delta()) as f64;
        if rot_delta.abs() > 0.001 {
            let cr = rot_delta.cos();
            let sr = rot_delta.sin();
            rotation = Matrix3::new(cr, sr, 0.0, -sr, cr, 0.0, 0.0, 0.0, 1.0) * rotation;
        }
        let is_pinching = (zd - 1.0).abs() > 0.001;
        if trackpad_rotate && !is_pinching {
            let sx = scroll_delta.x as f64;
            let sy = scroll_delta.y as f64;
            if sx.abs() > 0.1 || sy.abs() > 0.1 {
                let sensitivity = 0.002 / (1.0 + zoom.ln().max(0.0));
                let pitch = -sy * sensitivity;
                let yaw = sx * sensitivity;
                let cp = pitch.cos();
                let sp = pitch.sin();
                let rx = Matrix3::new(1.0, 0.0, 0.0, 0.0, cp, sp, 0.0, -sp, cp);
                if north_up {
                    rotation = rx * rotation;
                    let cy = yaw.cos();
                    let s_y = yaw.sin();
                    let ry_world = Matrix3::new(cy, 0.0, s_y, 0.0, 1.0, 0.0, -s_y, 0.0, cy);
                    rotation = rotation * ry_world;
                    let up_screen = rotation * Vector3::new(0.0, 1.0, 0.0);
                    let bearing = up_screen.x.atan2(up_screen.y);
                    if bearing.abs() > 1e-6 {
                        let max_corr = 0.05;
                        let corr = (-bearing * 0.4).clamp(-max_corr, max_corr);
                        let cc = corr.cos();
                        let sc = corr.sin();
                        rotation =
                            Matrix3::new(cc, sc, 0.0, -sc, cc, 0.0, 0.0, 0.0, 1.0) * rotation;
                    }
                } else {
                    let cy = (-yaw).cos();
                    let s_y = (-yaw).sin();
                    let ry = Matrix3::new(cy, 0.0, -s_y, 0.0, 1.0, 0.0, s_y, 0.0, cy);
                    rotation = rx * ry * rotation;
                }
            }
        } else {
            let scroll = scroll_delta.y;
            if scroll != 0.0 {
                let old_zoom = zoom;
                let factor = 1.0 + scroll as f64 * 0.001;
                zoom = (zoom * factor).clamp(0.01, 20000.0);

                let alt_km = 10000.0 / zoom;
                if alt_km <= 30000.0 {
                    if let Some(hover_pos) = response.response.hover_pos() {
                        let plot_pos = response.transform.value_from_position(hover_pos);
                        let cx = plot_pos.x;
                        let cy = plot_pos.y;
                        let r_sq = planet_radius * planet_radius;
                        if cx * cx + cy * cy <= r_sq {
                            let ratio = old_zoom / zoom;
                            let tx = cx * ratio;
                            let ty = cy * ratio;
                            if tx * tx + ty * ty <= r_sq {
                                let a = Vector3::new(cx, cy, (r_sq - cx * cx - cy * cy).sqrt())
                                    .normalize();
                                let b = Vector3::new(tx, ty, (r_sq - tx * tx - ty * ty).sqrt())
                                    .normalize();
                                let cross = a.cross(&b);
                                let cross_len = cross.norm();
                                if cross_len > 1e-12 {
                                    let axis = cross / cross_len;
                                    let angle = cross_len.atan2(a.dot(&b));
                                    let ca = angle.cos();
                                    let sa = angle.sin();
                                    let t = 1.0 - ca;
                                    let (x, y, z) = (axis.x, axis.y, axis.z);
                                    let rot = Matrix3::new(
                                        t * x * x + ca,
                                        t * x * y - sa * z,
                                        t * x * z + sa * y,
                                        t * x * y + sa * z,
                                        t * y * y + ca,
                                        t * y * z - sa * x,
                                        t * x * z - sa * y,
                                        t * y * z + sa * x,
                                        t * z * z + ca,
                                    );
                                    rotation = rot * rotation;
                                }
                            }

                            let center = rotation.transpose() * Vector3::new(0.0, 0.0, 1.0);
                            let lat_limit = 85.0_f64.to_radians().sin();
                            let clamped_y = center.y.clamp(-lat_limit, lat_limit);
                            let needs_clamp = (center.y - clamped_y).abs() > 1e-8;
                            if needs_clamp {
                                let horiz = (center.x * center.x + center.z * center.z).sqrt();
                                let new_horiz = (1.0 - clamped_y * clamped_y).sqrt();
                                let scale = if horiz > 1e-10 {
                                    new_horiz / horiz
                                } else {
                                    1.0
                                };
                                let clamped =
                                    Vector3::new(center.x * scale, clamped_y, center.z * scale)
                                        .normalize();
                                let right_raw = Vector3::new(clamped.z, 0.0, -clamped.x);
                                let right_len = right_raw.norm();
                                if right_len > 0.01 {
                                    let right = right_raw / right_len;
                                    let up = clamped.cross(&right);
                                    let r0 = Matrix3::new(
                                        right.x, right.y, right.z, up.x, up.y, up.z, clamped.x,
                                        clamped.y, clamped.z,
                                    );
                                    let up_screen = rotation * Vector3::new(0.0, 1.0, 0.0);
                                    let bearing = up_screen.x.atan2(up_screen.y);
                                    let cb = bearing.cos();
                                    let sb = bearing.sin();
                                    let rz = Matrix3::new(cb, sb, 0.0, -sb, cb, 0.0, 0.0, 0.0, 1.0);
                                    rotation = rz * r0;
                                }
                            }
                            if north_up {
                                let north_blend = (zoom.log2() / 4.0).clamp(0.0, 1.0);
                                if north_blend > 0.0 {
                                    let up_screen = rotation * Vector3::new(0.0, 1.0, 0.0);
                                    let bearing = up_screen.x.atan2(up_screen.y);
                                    let zoom_octaves =
                                        (zoom / old_zoom).ln().abs() / (2.0_f64).ln();
                                    let decay = (-north_blend * zoom_octaves * 1.5).exp();
                                    let correction = bearing * (decay - 1.0);
                                    let ca = correction.cos();
                                    let sa = correction.sin();
                                    rotation =
                                        Matrix3::new(ca, sa, 0.0, -sa, ca, 0.0, 0.0, 0.0, 1.0)
                                            * rotation;
                                }
                            }
                        }
                    }
                }
            }
        }
        if is_pinching {
            let old_zoom = zoom;
            zoom = (zoom * zd as f64).clamp(0.01, 20000.0);

            if let Some(hover_pos) = response.response.hover_pos() {
                let plot_pos = response.transform.value_from_position(hover_pos);
                let cx = plot_pos.x;
                let cy = plot_pos.y;
                let r_sq = planet_radius * planet_radius;
                if cx * cx + cy * cy <= r_sq {
                    let ratio = old_zoom / zoom;
                    let tx = cx * ratio;
                    let ty = cy * ratio;
                    if tx * tx + ty * ty <= r_sq {
                        let a = Vector3::new(cx, cy, (r_sq - cx * cx - cy * cy).sqrt()).normalize();
                        let b = Vector3::new(tx, ty, (r_sq - tx * tx - ty * ty).sqrt()).normalize();
                        let cross = a.cross(&b);
                        let cross_len = cross.norm();
                        if cross_len > 1e-12 {
                            let axis = cross / cross_len;
                            let angle = cross_len.atan2(a.dot(&b));
                            let ca = angle.cos();
                            let sa = angle.sin();
                            let t = 1.0 - ca;
                            let (x, y, z) = (axis.x, axis.y, axis.z);
                            let rot = Matrix3::new(
                                t * x * x + ca,
                                t * x * y - sa * z,
                                t * x * z + sa * y,
                                t * x * y + sa * z,
                                t * y * y + ca,
                                t * y * z - sa * x,
                                t * x * z - sa * y,
                                t * y * z + sa * x,
                                t * z * z + ca,
                            );
                            rotation = rot * rotation;
                        }
                    }

                    let center = rotation.transpose() * Vector3::new(0.0, 0.0, 1.0);
                    let lat_limit = 85.0_f64.to_radians().sin();
                    let clamped_y = center.y.clamp(-lat_limit, lat_limit);
                    let needs_clamp = (center.y - clamped_y).abs() > 1e-8;
                    if needs_clamp {
                        let horiz = (center.x * center.x + center.z * center.z).sqrt();
                        let new_horiz = (1.0 - clamped_y * clamped_y).sqrt();
                        let scale = if horiz > 1e-10 {
                            new_horiz / horiz
                        } else {
                            1.0
                        };
                        let clamped =
                            Vector3::new(center.x * scale, clamped_y, center.z * scale).normalize();
                        let right_raw = Vector3::new(clamped.z, 0.0, -clamped.x);
                        let right_len = right_raw.norm();
                        if right_len > 0.01 {
                            let right = right_raw / right_len;
                            let up = clamped.cross(&right);
                            let r0 = Matrix3::new(
                                right.x, right.y, right.z, up.x, up.y, up.z, clamped.x, clamped.y,
                                clamped.z,
                            );
                            let up_screen = rotation * Vector3::new(0.0, 1.0, 0.0);
                            let bearing = up_screen.x.atan2(up_screen.y);
                            let cb = bearing.cos();
                            let sb = bearing.sin();
                            let rz = Matrix3::new(cb, sb, 0.0, -sb, cb, 0.0, 0.0, 0.0, 1.0);
                            rotation = rz * r0;
                        }
                    }
                    if north_up {
                        let north_blend = (zoom.log2() / 4.0).clamp(0.0, 1.0);
                        if north_blend > 0.0 {
                            let up_screen = rotation * Vector3::new(0.0, 1.0, 0.0);
                            let bearing = up_screen.x.atan2(up_screen.y);
                            let zoom_octaves = (zoom / old_zoom).ln().abs() / (2.0_f64).ln();
                            let decay = (-north_blend * zoom_octaves * 1.5).exp();
                            let correction = bearing * (decay - 1.0);
                            let ca = correction.cos();
                            let sa = correction.sin();
                            rotation =
                                Matrix3::new(ca, sa, 0.0, -sa, ca, 0.0, 0.0, 0.0, 1.0) * rotation;
                        }
                    }
                }
            }
        }
    }

    (rotation, zoom)
}

fn project_segments(
    proj: &dyn crate::projection::Projection,
    points: &[(f64, f64)],
) -> Vec<Vec<[f64; 2]>> {
    let mut result = Vec::new();
    let mut seg: Vec<[f64; 2]> = Vec::new();
    for &(lat, lon) in points {
        if let Some((x, y)) = proj.project(lat, lon) {
            if let Some(last) = seg.last() {
                if (x - last[0]).abs() > 180.0 {
                    if seg.len() >= 2 {
                        result.push(std::mem::take(&mut seg));
                    } else {
                        seg.clear();
                    }
                }
            }
            seg.push([x, y]);
        } else {
            if seg.len() >= 2 {
                result.push(std::mem::take(&mut seg));
            } else {
                seg.clear();
            }
        }
    }
    if seg.len() >= 2 {
        result.push(seg);
    }
    result
}

pub fn draw_map_view(
    ui: &mut egui::Ui,
    id: &str,
    constellations: &[(
        WalkerConstellation,
        Vec<SatelliteState>,
        usize,
        u8,
        usize,
        String,
    )],
    proj: &dyn crate::projection::Projection,
    width: f32,
    height: f32,
    sat_radius: f32,
    single_color: bool,
    link_width: f32,
    show_orbits: bool,
    show_links: bool,
    show_coverage: bool,
    coverage_angle: f64,
    show_routing_paths: bool,
    show_manhattan_path: bool,
    show_shortest_path: bool,
    show_radiation_path: bool,
    radiation_weight: f64,
    show_graticule: bool,
    show_crosshairs: bool,
    satellite_cameras: &[SatelliteCamera],
    planet_radius: f64,
    geo_borders: &[Vec<(f64, f64)>],
    geo_cities: &[crate::geo::CityLabel],
    ground_stations: &[GroundStation],
    radiation: Option<&RadiationConfig>,
    body_rotation: &Matrix3<f64>,
    time: f64,
    gpu_available: bool,
    proj_shader_id: i32,
) {
    let (xmin, xmax) = proj.x_range();
    let (ymin, ymax) = proj.y_range();

    let use_gpu = gpu_available;

    let shared_bounds = Arc::new(Mutex::new([xmin, xmax, ymin, ymax]));

    if use_gpu {
        let rect = egui::Rect::from_min_size(ui.cursor().min, egui::Vec2::new(width, height));
        let callback = egui_wgpu::Callback::new_paint_callback(
            rect,
            MapPaintCallback::new(proj_shader_id, shared_bounds.clone()),
        );
        ui.painter().add(callback);
    }

    let plot = Plot::new(id)
        .width(width)
        .height(height)
        .include_x(xmin)
        .include_x(xmax)
        .include_y(ymin)
        .include_y(ymax)
        .data_aspect(1.0)
        .show_axes([!use_gpu, !use_gpu])
        .show_grid(!use_gpu)
        .show_x(!use_gpu)
        .show_y(!use_gpu)
        .show_background(!use_gpu);

    let mut crosshair_tooltip: Option<(egui::Pos2, f64, f64)> = None;
    plot.show(ui, |plot_ui| {
        let pb = plot_ui.plot_bounds();
        *shared_bounds.lock() = [pb.min()[0], pb.max()[0], pb.min()[1], pb.max()[1]];

        if show_graticule {
            let grid_color = egui::Color32::from_rgba_unmultiplied(120, 120, 120, 120);
            for lat in (-60..=60).step_by(30) {
                let latlon: Vec<(f64, f64)> = (-180..=180)
                    .step_by(2)
                    .map(|lon| (lat as f64, lon as f64))
                    .collect();
                for seg in project_segments(proj, &latlon) {
                    plot_ui.line(
                        Line::new("", PlotPoints::new(seg))
                            .color(grid_color)
                            .width(0.5),
                    );
                }
            }
            for lon in (-180..=180).step_by(30) {
                let latlon: Vec<(f64, f64)> = (-90..=90)
                    .step_by(2)
                    .map(|lat| (lat as f64, lon as f64))
                    .collect();
                for seg in project_segments(proj, &latlon) {
                    plot_ui.line(
                        Line::new("", PlotPoints::new(seg))
                            .color(grid_color)
                            .width(0.5),
                    );
                }
            }
            let equator: Vec<(f64, f64)> = (-180..=180)
                .step_by(2)
                .map(|lon| (0.0, lon as f64))
                .collect();
            for seg in project_segments(proj, &equator) {
                plot_ui.line(
                    Line::new("", PlotPoints::new(seg))
                        .color(grid_color)
                        .width(0.5),
                );
            }
            let prime: Vec<(f64, f64)> =
                (-90..=90).step_by(2).map(|lat| (lat as f64, 0.0)).collect();
            for seg in project_segments(proj, &prime) {
                plot_ui.line(
                    Line::new("", PlotPoints::new(seg))
                        .color(grid_color)
                        .width(0.5),
                );
            }
        }

        if show_crosshairs {
            if let Some(ptr) = plot_ui.pointer_coordinate() {
                if let Some((lat, lon)) = proj.inverse(ptr.x, ptr.y) {
                    let cursor_color = egui::Color32::from_rgba_unmultiplied(200, 200, 200, 100);
                    let lat_line: Vec<(f64, f64)> =
                        (-180..=180).step_by(2).map(|lo| (lat, lo as f64)).collect();
                    for seg in project_segments(proj, &lat_line) {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(seg))
                                .color(cursor_color)
                                .width(0.5),
                        );
                    }
                    let lon_line: Vec<(f64, f64)> =
                        (-90..=90).step_by(2).map(|la| (la as f64, lon)).collect();
                    for seg in project_segments(proj, &lon_line) {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(seg))
                                .color(cursor_color)
                                .width(0.5),
                        );
                    }
                    if let Some(screen_pos) = plot_ui.response().hover_pos() {
                        crosshair_tooltip = Some((screen_pos, lat, lon));
                    }
                }
            }
        }

        for polyline in geo_borders {
            let border_color = egui::Color32::from_rgb(80, 120, 80);
            for seg in project_segments(proj, polyline) {
                plot_ui.line(
                    Line::new("", PlotPoints::new(seg))
                        .color(border_color)
                        .width(0.8),
                );
            }
        }

        for city in geo_cities {
            if let Some((cx, cy)) = proj.project(city.lat, city.lon) {
                plot_ui.text(
                    Text::new("", PlotPoint::new(cx, cy), &city.name)
                        .color(egui::Color32::from_rgb(160, 160, 160)),
                );
            }
        }

        for gs in ground_stations {
            if let Some((gx, gy)) = proj.project(gs.lat, gs.lon) {
                plot_ui.points(
                    Points::new("", PlotPoints::new(vec![[gx, gy]]))
                        .color(gs.color)
                        .radius(4.0)
                        .filled(true),
                );
            }
        }

        for (constellation, positions, color_offset, _, _, _) in constellations {
            if show_orbits {
                for plane in 0..constellation.num_planes {
                    let color = plane_color(if single_color {
                        *color_offset
                    } else {
                        plane + color_offset
                    });
                    let dim =
                        egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 80);
                    let orbit_pts = constellation.orbit_points_3d(plane, time);
                    let latlon: Vec<(f64, f64)> = orbit_pts
                        .iter()
                        .map(|&(x, y, z)| {
                            let r = (x * x + y * y + z * z).sqrt();
                            (y / r).asin().to_degrees().clamp(-90.0, 90.0)
                        })
                        .zip(
                            orbit_pts
                                .iter()
                                .map(|&(x, _, z)| (-z).atan2(x).to_degrees()),
                        )
                        .map(|(lat, lon)| (lat, lon))
                        .collect();
                    for seg in project_segments(proj, &latlon) {
                        plot_ui.line(Line::new("", PlotPoints::new(seg)).color(dim).width(1.0));
                    }
                }
            }

            if show_links {
                for sat in positions.iter() {
                    for &ni in &sat.neighbors {
                        if let Some(neigh) = positions.get(ni) {
                            let link_color =
                                egui::Color32::from_rgba_unmultiplied(120, 120, 120, 60);
                            if let (Some((x1, y1)), Some((x2, y2))) = (
                                proj.project(sat.lat, sat.lon),
                                proj.project(neigh.lat, neigh.lon),
                            ) {
                                if (x1 - x2).abs() < 180.0 {
                                    plot_ui.line(
                                        Line::new("", PlotPoints::new(vec![[x1, y1], [x2, y2]]))
                                            .color(link_color)
                                            .width(link_width),
                                    );
                                }
                            }
                        }
                    }
                }
            }

            if show_coverage {
                let orbit_radius = planet_radius + constellation.altitude_km;
                let cone_half_angle = (coverage_angle / 2.0).to_radians();
                let max_earth_angle = (planet_radius / orbit_radius).acos();
                let sin_beta = orbit_radius * cone_half_angle.sin() / planet_radius;
                let angular_radius = if sin_beta >= 1.0 {
                    max_earth_angle
                } else {
                    (sin_beta.asin() - cone_half_angle).min(max_earth_angle)
                };

                for sat in positions.iter() {
                    let color = plane_color(if single_color {
                        *color_offset
                    } else {
                        sat.plane + color_offset
                    });
                    let fill =
                        egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 30);
                    let lat = sat.lat.to_radians();
                    let lon = sat.lon.to_radians();
                    let circle_pts: Vec<(f64, f64)> = (0..=32)
                        .map(|i| {
                            let angle = 2.0 * PI * i as f64 / 32.0;
                            let clat = (lat.sin() * angular_radius.cos()
                                + lat.cos() * angular_radius.sin() * angle.cos())
                            .asin();
                            let clon = lon
                                + (angular_radius.sin() * angle.sin()).atan2(
                                    lat.cos() * angular_radius.cos()
                                        - lat.sin() * angular_radius.sin() * angle.cos(),
                                );
                            (clat.to_degrees(), clon.to_degrees())
                        })
                        .collect();
                    let projected: Vec<[f64; 2]> = circle_pts
                        .iter()
                        .filter_map(|&(la, lo)| proj.project(la, lo).map(|(x, y)| [x, y]))
                        .collect();
                    if projected.len() >= 3 {
                        plot_ui
                            .polygon(Polygon::new("", PlotPoints::new(projected)).fill_color(fill));
                    }
                }
            }

            for plane in 0..constellation.num_planes {
                let color = plane_color(if single_color {
                    *color_offset
                } else {
                    plane + color_offset
                });
                let pts: Vec<[f64; 2]> = positions
                    .iter()
                    .filter(|s| s.plane == plane)
                    .filter_map(|s| proj.project(s.lat, s.lon).map(|(x, y)| [x, y]))
                    .collect();
                plot_ui.points(
                    Points::new("", PlotPoints::new(pts))
                        .color(color)
                        .radius(sat_radius)
                        .filled(true),
                );
            }
        }

        if show_routing_paths && !satellite_cameras.is_empty() {
            let manhattan_color = egui::Color32::from_rgb(255, 100, 100);
            let shortest_color = egui::Color32::from_rgb(100, 255, 100);
            let rad_color = egui::Color32::from_rgb(100, 220, 255);
            let rad_grid = radiation.and_then(|r| r.igrf_rad_cache.as_ref().map(|(_, _, g)| g));

            for (cidx, (constellation, positions, _, _, _, _)) in constellations.iter().enumerate()
            {
                let tracked: Vec<_> = satellite_cameras
                    .iter()
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

                        let draw_path =
                            |path: &[(usize, usize)], color: egui::Color32, w: f32| -> Vec<Line> {
                                let mut lines = Vec::new();
                                for k in 0..(path.len() - 1) {
                                    let s1 = positions
                                        .iter()
                                        .find(|s| s.plane == path[k].0 && s.sat_index == path[k].1);
                                    let s2 = positions.iter().find(|s| {
                                        s.plane == path[k + 1].0 && s.sat_index == path[k + 1].1
                                    });
                                    if let (Some(a), Some(b)) = (s1, s2) {
                                        if let (Some((x1, y1)), Some((x2, y2))) =
                                            (proj.project(a.lat, a.lon), proj.project(b.lat, b.lon))
                                        {
                                            if (x1 - x2).abs() < 180.0 {
                                                lines.push(
                                                    Line::new(
                                                        "",
                                                        PlotPoints::new(vec![[x1, y1], [x2, y2]]),
                                                    )
                                                    .color(color)
                                                    .width(w),
                                                );
                                            }
                                        }
                                    }
                                }
                                lines
                            };

                        if show_manhattan_path {
                            let path = compute_manhattan_path(
                                src.plane,
                                src.sat_index,
                                dst.plane,
                                dst.sat_index,
                                num_planes,
                                sats_per_plane,
                                is_star,
                                positions,
                            );
                            for line in draw_path(&path, manhattan_color, 2.5) {
                                plot_ui.line(line);
                            }
                        }
                        if show_shortest_path {
                            let path = compute_shortest_path(
                                src.plane,
                                src.sat_index,
                                dst.plane,
                                dst.sat_index,
                                num_planes,
                                sats_per_plane,
                                positions,
                                is_star,
                            );
                            for line in draw_path(&path, shortest_color, 2.0) {
                                plot_ui.line(line);
                            }
                        }
                        if show_radiation_path {
                            let path = compute_radiation_path(
                                src.plane,
                                src.sat_index,
                                dst.plane,
                                dst.sat_index,
                                num_planes,
                                sats_per_plane,
                                positions,
                                is_star,
                                body_rotation,
                                rad_grid,
                                radiation_weight,
                            );
                            for line in draw_path(&path, rad_color, 2.0) {
                                plot_ui.line(line);
                            }
                        }
                    }
                }
            }
        }
    });

    if let Some((screen_pos, lat, lon)) = crosshair_tooltip {
        let text = format!("{:.1}° {:.1}°", lat, lon);
        let font = egui::FontId::proportional(12.0);
        let text_pos = screen_pos + egui::Vec2::new(15.0, -15.0);
        let galley = ui
            .painter()
            .layout_no_wrap(text, font, egui::Color32::WHITE);
        let rect = egui::Rect::from_min_size(
            text_pos - egui::Vec2::new(0.0, galley.size().y),
            galley.size(),
        )
        .expand(3.0);
        ui.painter().rect_filled(
            rect,
            3.0,
            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180),
        );
        ui.painter().galley(
            text_pos - egui::Vec2::new(0.0, galley.size().y),
            galley,
            egui::Color32::WHITE,
        );
    }
}

pub fn draw_torus(
    ui: &mut egui::Ui,
    id: &str,
    constellations: &[(
        WalkerConstellation,
        Vec<SatelliteState>,
        usize,
        u8,
        usize,
        String,
    )],
    time: f64,
    rotation: Matrix3<f64>,
    width: f32,
    height: f32,
    sat_radius: f32,
    show_links: bool,
    show_orbits: bool,
    single_color: bool,
    mut zoom: f64,
    satellite_cameras: &mut [SatelliteCamera],
    show_routing_paths: bool,
    show_manhattan_path: bool,
    show_shortest_path: bool,
    show_radiation_path: bool,
    radiation_weight: f64,
    show_asc_desc_colors: bool,
    color_ascending: egui::Color32,
    color_descending: egui::Color32,
    color_links: egui::Color32,
    planet_radius: f64,
    pending_cameras: &mut Vec<SatelliteCamera>,
    camera_id_counter: &mut usize,
    cameras_to_remove: &mut Vec<usize>,
    link_width: f32,
    fixed_sizes: bool,
    body_rotation: &Matrix3<f64>,
    igrf_rad_cache: Option<&crate::igrf::IgrfRadGrid>,
) -> (Matrix3<f64>, f64) {
    let (major_radius, minor_radius) =
        if let Some((constellation, _, _, _, _, _)) = constellations.first() {
            let inclination_rad = constellation.inclination_deg.to_radians();
            let cos_i = inclination_rad.cos().abs();
            let major = 1.0;
            let minor = (major * (1.0 - cos_i) / (1.0 + cos_i)).max(0.002);
            (major, minor)
        } else {
            (1.0, 0.8)
        };

    let margin = (major_radius + minor_radius) * 1.3 / zoom;
    let zoom_factor = if fixed_sizes { 1.0 } else { zoom as f32 };
    let scaled_sat_radius = sat_radius * zoom_factor;
    let scaled_link_width = (link_width * zoom_factor).max(0.5);

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
        .show_background(!ui.visuals().dark_mode)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .cursor_color(egui::Color32::TRANSPARENT);

    let old_plot_bg_stroke = ui.visuals().widgets.noninteractive.bg_stroke;
    if !ui.visuals().dark_mode {
        ui.visuals_mut().widgets.noninteractive.bg_stroke = egui::Stroke::NONE;
    }

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

        let torus_point = |theta: f64, phi: f64, ecc: f64, omega: f64| -> (f64, f64, f64) {
            let r_orbit = if ecc > 0.001 {
                minor_radius * (1.0 - ecc * ecc) / (1.0 + ecc * phi.cos())
            } else {
                minor_radius
            };
            let angle = phi + omega;
            let r = major_radius + r_orbit * angle.cos();
            let y = r_orbit * angle.sin();
            let x = r * theta.cos();
            let z = r * theta.sin();
            rotate_point_matrix(x, y, z, &display_rotation)
        };

        for (constellation, positions, color_offset, _tle_kind, orig_idx, _) in
            constellations.iter()
        {
            let sats_per_plane = constellation.total_sats / constellation.num_planes;
            let orbit_radius = constellation.planet_radius + constellation.altitude_km;
            let period = 2.0 * PI * (orbit_radius.powi(3) / constellation.planet_mu).sqrt();
            let mean_motion = 2.0 * PI / period;
            let ecc = constellation.eccentricity;
            let omega = constellation.arg_periapsis_deg.to_radians();
            let raan_step = constellation.raan_step();
            let raan_offset = constellation.raan_offset_deg.to_radians();
            let plane_theta = |plane: usize| -> f64 { raan_offset + raan_step * plane as f64 };

            let phase_step = constellation.phasing * 2.0 * PI / constellation.total_sats as f64;
            let torus_pos = |plane: usize, sat_idx: usize| -> (f64, f64, f64) {
                let angle = plane_theta(plane);
                let sat_spacing = 2.0 * PI * sat_idx as f64 / sats_per_plane as f64;
                let phase = sat_spacing + mean_motion * time + phase_step * plane as f64;
                torus_point(angle, phase, ecc, omega)
            };

            if show_orbits {
                for plane in 0..constellation.num_planes {
                    let angle = plane_theta(plane);
                    let color = if show_routing_paths || show_asc_desc_colors {
                        color_links
                    } else {
                        plane_color(if single_color {
                            *color_offset
                        } else {
                            plane + color_offset
                        })
                    };
                    let dim_col =
                        egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 180);

                    let mut front_segment: Vec<[f64; 2]> = Vec::new();
                    let mut back_segment: Vec<[f64; 2]> = Vec::new();

                    for i in 0..=50 {
                        let phase = 2.0 * PI * i as f64 / 50.0;
                        let (rx, ry, _) = torus_point(angle, phase, ecc, omega);
                        let facing = is_facing_camera(angle, phase);

                        if facing {
                            front_segment.push([rx, ry]);
                            if !back_segment.is_empty() {
                                plot_ui.line(
                                    Line::new(
                                        "",
                                        PlotPoints::new(std::mem::take(&mut back_segment)),
                                    )
                                    .color(dim_col)
                                    .width(scaled_link_width),
                                );
                            }
                        } else {
                            back_segment.push([rx, ry]);
                            if !front_segment.is_empty() {
                                plot_ui.line(
                                    Line::new(
                                        "",
                                        PlotPoints::new(std::mem::take(&mut front_segment)),
                                    )
                                    .color(color)
                                    .width(scaled_link_width * 1.5),
                                );
                            }
                        }
                    }
                    if !front_segment.is_empty() {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(front_segment))
                                .color(color)
                                .width(scaled_link_width * 1.5),
                        );
                    }
                    if !back_segment.is_empty() {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(back_segment))
                                .color(dim_col)
                                .width(scaled_link_width),
                        );
                    }
                }
            }

            if show_links {
                let base_link_color = if show_routing_paths || show_asc_desc_colors {
                    color_links
                } else {
                    egui::Color32::from_rgb(150, 150, 150)
                };
                let link_dim = egui::Color32::from_rgba_unmultiplied(50, 50, 60, 100);
                for sat in positions {
                    for &neighbor_idx in &sat.neighbors {
                        let neighbor = &positions[neighbor_idx];
                        let angle1 = plane_theta(sat.plane);
                        let angle2 = plane_theta(neighbor.plane);
                        let phase1 = 2.0 * PI * sat.sat_index as f64 / sats_per_plane as f64
                            + mean_motion * time;
                        let phase2 = 2.0 * PI * neighbor.sat_index as f64 / sats_per_plane as f64
                            + mean_motion * time;

                        let (x1, y1, _) = torus_pos(sat.plane, sat.sat_index);
                        let (x2, y2, _) = torus_pos(neighbor.plane, neighbor.sat_index);
                        let facing1 = is_facing_camera(angle1, phase1);
                        let facing2 = is_facing_camera(angle2, phase2);
                        let color = if facing1 && facing2 {
                            base_link_color
                        } else {
                            link_dim
                        };
                        plot_ui.line(
                            Line::new("", PlotPoints::new(vec![[x1, y1], [x2, y2]]))
                                .color(color)
                                .width(scaled_link_width),
                        );
                    }
                }
            }

            for plane in 0..constellation.num_planes {
                let base_color = plane_color(if single_color {
                    *color_offset
                } else {
                    plane + color_offset
                });
                let angle = plane_theta(plane);

                for sat in positions.iter().filter(|s| s.plane == plane) {
                    let is_tracked = satellite_cameras.iter().any(|c| {
                        c.constellation_idx == *orig_idx
                            && c.plane == sat.plane
                            && c.sat_index == sat.sat_index
                    });
                    let color = if show_asc_desc_colors {
                        if sat.ascending {
                            color_ascending
                        } else {
                            color_descending
                        }
                    } else {
                        base_color
                    };
                    let dim_col =
                        egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 140);

                    let phase = 2.0 * PI * sat.sat_index as f64 / sats_per_plane as f64
                        + mean_motion * time;
                    let (x, y, _) = torus_pos(sat.plane, sat.sat_index);
                    let facing = is_facing_camera(angle, phase);
                    let (c, r) = if facing {
                        (color, scaled_sat_radius)
                    } else {
                        (dim_col, scaled_sat_radius * 0.8)
                    };
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
                                .radius(scaled_sat_radius * 2.5)
                                .filled(false),
                        );
                    }
                }
            }

            if show_routing_paths {
                let tracked: Vec<_> = satellite_cameras
                    .iter()
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

                            let src_sat = positions
                                .iter()
                                .find(|s| s.plane == src.plane && s.sat_index == src.sat_index);
                            let dst_sat = positions
                                .iter()
                                .find(|s| s.plane == dst.plane && s.sat_index == dst.sat_index);

                            let can_route = match (src_sat, dst_sat) {
                                (Some(_), Some(_)) => {
                                    if is_star {
                                        let plane_diff_fwd =
                                            (dst.plane + num_planes - src.plane) % num_planes;
                                        let plane_diff_bwd =
                                            (src.plane + num_planes - dst.plane) % num_planes;
                                        let crosses_seam = plane_diff_fwd > num_planes / 2
                                            && plane_diff_bwd > num_planes / 2;
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
                                    src.plane,
                                    src.sat_index,
                                    dst.plane,
                                    dst.sat_index,
                                    num_planes,
                                    sats_per_plane,
                                    is_star,
                                    positions,
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
                                    src.plane,
                                    src.sat_index,
                                    dst.plane,
                                    dst.sat_index,
                                    num_planes,
                                    sats_per_plane,
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

                            if show_radiation_path {
                                let path = compute_radiation_path(
                                    src.plane,
                                    src.sat_index,
                                    dst.plane,
                                    dst.sat_index,
                                    num_planes,
                                    sats_per_plane,
                                    positions,
                                    is_star,
                                    body_rotation,
                                    igrf_rad_cache,
                                    radiation_weight,
                                );
                                let rad_color = egui::Color32::from_rgb(100, 220, 255);
                                for k in 0..(path.len() - 1) {
                                    let (p1, s1) = path[k];
                                    let (p2, s2) = path[k + 1];
                                    let (x1, y1, _) = torus_pos(p1, s1);
                                    let (x2, y2, _) = torus_pos(p2, s2);
                                    plot_ui.line(
                                        Line::new("", PlotPoints::new(vec![[x1, y1], [x2, y2]]))
                                            .color(rad_color)
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

    if !ui.visuals().dark_mode {
        ui.visuals_mut().widgets.noninteractive.bg_stroke = old_plot_bg_stroke;
    }

    if response.response.dragged() && !response.response.drag_started() {
        let drag = response.response.drag_delta();
        let sens = 0.01 / zoom.max(1.0);
        let delta_rot = rotation_from_drag(drag.x as f64 * sens, drag.y as f64 * sens);
        user_rotation = delta_rot * user_rotation;
    }

    if response.response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            let factor = 1.0 + scroll as f64 * 0.001;
            zoom = (zoom * factor).clamp(0.01, 20000.0);
        }
        let zd = ui.input(|i| i.zoom_delta());
        if (zd - 1.0).abs() > 0.001 {
            zoom = (zoom * zd as f64).clamp(0.01, 20000.0);
        }
    }

    if let Some(pos) = response.response.interact_pointer_pos() {
        if response.response.clicked() {
            let click_x = response.transform.value_from_position(pos).x;
            let click_y = response.transform.value_from_position(pos).y;
            let (major_radius, minor_radius) =
                if let Some((constellation, _, _, _, _, _)) = constellations.first() {
                    let sats_per_plane = constellation.sats_per_plane();
                    let orbit_radius = planet_radius + constellation.altitude_km;
                    let inclination_rad = constellation.inclination_deg.to_radians();
                    let inclination_factor = inclination_rad.sin().abs().max(0.002);
                    let altitude_factor = orbit_radius / (planet_radius + 500.0);
                    let major =
                        altitude_factor * (sats_per_plane as f64 / constellation.num_planes as f64);
                    let minor_base = altitude_factor * inclination_factor;
                    let minor = minor_base.max(major * inclination_factor);
                    let scale = 2.0 / (major + minor).max(1.0);
                    (major * scale, minor * scale)
                } else {
                    (2.0, 0.8)
                };
            let margin = (major_radius + minor_radius) * 1.3 / zoom;
            let click_threshold = margin * 0.05;

            let torus_point_click =
                |theta: f64, phi: f64, ecc: f64, omega: f64| -> (f64, f64, f64) {
                    let r_orbit = if ecc > 0.001 {
                        minor_radius * (1.0 - ecc * ecc) / (1.0 + ecc * phi.cos())
                    } else {
                        minor_radius
                    };
                    let angle = phi + omega;
                    let r = major_radius + r_orbit * angle.cos();
                    let y = r_orbit * angle.sin();
                    let x = r * theta.cos();
                    let z = r * theta.sin();
                    rotate_point_matrix(x, y, z, &display_rotation)
                };

            'outer: for (constellation, positions, _, _, orig_idx, _) in constellations.iter() {
                let sats_per_plane = constellation.total_sats / constellation.num_planes;
                let orbit_radius = constellation.planet_radius + constellation.altitude_km;
                let period = 2.0 * PI * (orbit_radius.powi(3) / constellation.planet_mu).sqrt();
                let mean_motion = 2.0 * PI / period;
                let ecc = constellation.eccentricity;
                let omega = constellation.arg_periapsis_deg.to_radians();
                let raan_step = constellation.raan_step();
                let raan_offset = constellation.raan_offset_deg.to_radians();

                for sat in positions {
                    let angle = raan_offset + raan_step * sat.plane as f64;
                    let phase = 2.0 * PI * sat.sat_index as f64 / sats_per_plane as f64
                        + mean_motion * time;
                    let (tx, ty, _) = torus_point_click(angle, phase, ecc, omega);

                    let dx = tx - click_x;
                    let dy = ty - click_y;
                    if dx * dx + dy * dy < click_threshold * click_threshold {
                        let existing = satellite_cameras.iter().find(|c| {
                            c.constellation_idx == *orig_idx
                                && c.plane == sat.plane
                                && c.sat_index == sat.sat_index
                        });
                        if let Some(cam) = existing {
                            cameras_to_remove.push(cam.id);
                        } else {
                            let in_pending = pending_cameras.iter().any(|c| {
                                c.constellation_idx == *orig_idx
                                    && c.plane == sat.plane
                                    && c.sat_index == sat.sat_index
                            });
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

fn clip_line_to_rect(
    mut p0: egui::Pos2,
    mut p1: egui::Pos2,
    r: egui::Rect,
) -> Option<(egui::Pos2, egui::Pos2)> {
    const LEFT: u8 = 1;
    const RIGHT: u8 = 2;
    const BOT: u8 = 4;
    const TOP: u8 = 8;
    let outcode = |p: egui::Pos2| -> u8 {
        let mut c = 0u8;
        if p.x < r.min.x {
            c |= LEFT;
        } else if p.x > r.max.x {
            c |= RIGHT;
        }
        if p.y < r.min.y {
            c |= TOP;
        } else if p.y > r.max.y {
            c |= BOT;
        }
        c
    };
    let mut c0 = outcode(p0);
    let mut c1 = outcode(p1);
    loop {
        if (c0 | c1) == 0 {
            return Some((p0, p1));
        }
        if (c0 & c1) != 0 {
            return None;
        }
        let co = if c0 != 0 { c0 } else { c1 };
        let dx = p1.x - p0.x;
        let dy = p1.y - p0.y;
        let p = if co & TOP != 0 {
            egui::pos2(p0.x + dx * (r.min.y - p0.y) / dy, r.min.y)
        } else if co & BOT != 0 {
            egui::pos2(p0.x + dx * (r.max.y - p0.y) / dy, r.max.y)
        } else if co & RIGHT != 0 {
            egui::pos2(r.max.x, p0.y + dy * (r.max.x - p0.x) / dx)
        } else {
            egui::pos2(r.min.x, p0.y + dy * (r.min.x - p0.x) / dx)
        };
        if co == c0 {
            p0 = p;
            c0 = outcode(p0);
        } else {
            p1 = p;
            c1 = outcode(p1);
        }
    }
}

pub fn plane_color(plane: usize) -> egui::Color32 {
    if plane < COLORS.len() {
        return COLORS[plane];
    }
    // Beyond the base palette, generate distinct colors using golden-ratio
    // hue spacing. Saturation and value are kept high so generated colors
    // stay readable on dark backgrounds (the old formula could drop value
    // to ~0.7, producing near-black purples and blues).
    let golden = 0.6180339887498949_f32;
    let hue = ((plane as f32) * golden).fract();
    let sat = 0.65 + 0.2 * (((plane / COLORS.len()) as f32) * golden).fract();
    let val = 0.9 + 0.1 * (((plane / (COLORS.len() * 3)) as f32) * golden).fract();
    hsv_to_color32(hue, sat, val.clamp(0.85, 1.0))
}

fn hsv_to_color32(h: f32, s: f32, v: f32) -> egui::Color32 {
    let h = h.fract();
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match i as i32 % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

pub const COLORS: [egui::Color32; 16] = [
    egui::Color32::from_rgb(255, 99, 71),
    egui::Color32::from_rgb(50, 205, 50),
    egui::Color32::from_rgb(30, 144, 255),
    egui::Color32::from_rgb(255, 215, 0),
    egui::Color32::from_rgb(238, 130, 238),
    egui::Color32::from_rgb(0, 206, 209),
    egui::Color32::from_rgb(255, 140, 0),
    egui::Color32::from_rgb(190, 150, 255),
    egui::Color32::from_rgb(0, 255, 127),
    egui::Color32::from_rgb(255, 105, 180),
    egui::Color32::from_rgb(100, 149, 237),
    egui::Color32::from_rgb(240, 230, 140),
    egui::Color32::from_rgb(60, 179, 113),
    egui::Color32::from_rgb(233, 150, 122),
    egui::Color32::from_rgb(186, 85, 211),
    egui::Color32::from_rgb(135, 206, 235),
];

pub fn dim_color(color: egui::Color32) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(
        (color.r() as f32 * 0.4) as u8,
        (color.g() as f32 * 0.4) as u8,
        (color.b() as f32 * 0.4) as u8,
        200,
    )
}
