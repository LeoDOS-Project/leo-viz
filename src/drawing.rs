//! 2D and 3D drawing routines for satellite visualizations.
//!
//! Renders the 3D globe view, torus topology view, ground track map, and
//! satellite camera projections. Handles orbit lines, coverage cones,
//! inter-satellite links, routing paths, and place markers.

use crate::celestial::{CelestialBody, Skin, TextureResolution};
use crate::config::{
    AreaOfInterest, DeviceLayer, GroundStation, SatelliteCamera, View3DFlags,
};
use crate::geo::CityLabel;
use crate::math::{rotate_point_matrix, rotation_from_drag};
use crate::renderer::SphereRenderer;
use crate::texture::EarthTexture;
use crate::tile::{DetailBounds, DetailTexture};
use crate::walker::{WalkerType, WalkerConstellation, SatelliteState};
use eframe::{egui, egui_glow, glow};
use egui::mutex::Mutex;
use egui_plot::{Line, Plot, PlotImage, PlotPoints, PlotPoint, Points, Polygon, Text};
use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;
use std::sync::Arc;

use crate::EARTH_VISUAL_SCALE;

pub const COLOR_ASCENDING: egui::Color32 = egui::Color32::from_rgb(200, 120, 50);
pub const COLOR_DESCENDING: egui::Color32 = egui::Color32::from_rgb(50, 100, 180);

pub fn draw_satellite_camera(
    ui: &mut egui::Ui,
    camera_id: usize,
    lat: f64,
    lon: f64,
    altitude_km: f64,
    coverage_angle: f64,
    earth_texture: &EarthTexture,
    planet_radius: f64,
) {
    let size = ui.available_size();
    let img_size = size.x.min(size.y - 40.0) as usize;
    if img_size < 10 {
        return;
    }

    let lat_rad = lat.to_radians();
    let lon_rad = lon.to_radians();
    let cone_half_angle = coverage_angle.to_radians();
    let orbit_radius = planet_radius + altitude_km;
    let max_earth_angle = (planet_radius / orbit_radius).acos();
    let earth_central_angle = (orbit_radius * cone_half_angle.sin() / planet_radius).asin();
    let angular_radius = earth_central_angle.min(max_earth_angle);

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

pub fn compute_path_direction(src: usize, dst: usize, modulus: usize, is_star: bool) -> (i32, usize) {
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
    src_plane: usize, src_sat: usize,
    dst_plane: usize, dst_sat: usize,
    num_planes: usize, sats_per_plane: usize,
    is_star: bool,
) -> Vec<(usize, usize)> {
    let mut path = vec![(src_plane, src_sat)];

    let (plane_dir, plane_steps) = compute_path_direction(src_plane, dst_plane, num_planes, is_star);
    let (sat_dir, sat_steps) = compute_path_direction(src_sat, dst_sat, sats_per_plane, false);

    let mut cur_plane = src_plane;
    for _ in 0..plane_steps {
        cur_plane = wrap_index(cur_plane, plane_dir, num_planes);
        path.push((cur_plane, src_sat));
    }

    let mut cur_sat = src_sat;
    for _ in 0..sat_steps {
        cur_sat = wrap_index(cur_sat, sat_dir, sats_per_plane);
        path.push((dst_plane, cur_sat));
    }

    path
}

pub fn compute_shortest_path(
    src_plane: usize, src_sat: usize,
    dst_plane: usize, dst_sat: usize,
    num_planes: usize, sats_per_plane: usize,
    positions: &[SatelliteState],
    is_star: bool,
) -> Vec<(usize, usize)> {
    let mut path = vec![(src_plane, src_sat)];

    let (plane_dir, mut plane_steps_remaining) = compute_path_direction(src_plane, dst_plane, num_planes, is_star);
    let (sat_dir, mut sat_steps_remaining) = compute_path_direction(src_sat, dst_sat, sats_per_plane, false);

    let get_pos = |plane: usize, sat_idx: usize| -> Option<(f64, f64, f64)> {
        positions.iter()
            .find(|s| s.plane == plane && s.sat_index == sat_idx)
            .map(|s| (s.x, s.y, s.z))
    };

    let distance = |p1: (f64, f64, f64), p2: (f64, f64, f64)| -> f64 {
        let dx = p1.0 - p2.0;
        let dy = p1.1 - p2.1;
        let dz = p1.2 - p2.2;
        (dx * dx + dy * dy + dz * dz).sqrt()
    };

    let mut cur_plane = src_plane;
    let mut cur_sat = src_sat;

    while plane_steps_remaining > 0 || sat_steps_remaining > 0 {
        if plane_steps_remaining == 0 {
            cur_sat = wrap_index(cur_sat, sat_dir, sats_per_plane);
            sat_steps_remaining -= 1;
            path.push((cur_plane, cur_sat));
            continue;
        }
        if sat_steps_remaining == 0 {
            cur_plane = wrap_index(cur_plane, plane_dir, num_planes);
            plane_steps_remaining -= 1;
            path.push((cur_plane, cur_sat));
            continue;
        }

        let next_plane = wrap_index(cur_plane, plane_dir, num_planes);
        let next_sat = wrap_index(cur_sat, sat_dir, sats_per_plane);

        let cur_pos = get_pos(cur_plane, cur_sat);
        let cross_plane_pos = get_pos(next_plane, cur_sat);
        let within_plane_pos = get_pos(cur_plane, next_sat);
        let cross_plane_after_within = get_pos(next_plane, next_sat);

        match (cur_pos, cross_plane_pos, within_plane_pos, cross_plane_after_within) {
            (Some(cur), Some(cross), Some(within), Some(cross_after)) => {
                let cross_now = distance(cur, cross);
                let cross_after_within = distance(within, cross_after);

                if cross_now <= cross_after_within {
                    cur_plane = next_plane;
                    plane_steps_remaining -= 1;
                } else {
                    cur_sat = next_sat;
                    sat_steps_remaining -= 1;
                }
            }
            _ => {
                cur_plane = next_plane;
                plane_steps_remaining -= 1;
            }
        }
        path.push((cur_plane, cur_sat));
    }

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
) {
    if path.len() < 2 {
        return;
    }

    for i in 0..(path.len() - 1) {
        let (plane1, sat1) = path[i];
        let (plane2, sat2) = path[i + 1];

        let pos1 = positions.iter().find(|s| s.plane == plane1 && s.sat_index == sat1);
        let pos2 = positions.iter().find(|s| s.plane == plane2 && s.sat_index == sat2);

        if let (Some(p1), Some(p2)) = (pos1, pos2) {
            let (rx1, ry1, rz1) = rotate_point_matrix(p1.x, p1.y, p1.z, rotation);
            let (rx2, ry2, rz2) = rotate_point_matrix(p2.x, p2.y, p2.z, rotation);

            let visible1 = rz1 >= 0.0 || (rx1 * rx1 + ry1 * ry1) >= earth_r_sq;
            let visible2 = rz2 >= 0.0 || (rx2 * rx2 + ry2 * ry2) >= earth_r_sq;

            if hide_behind_earth && !visible1 && !visible2 {
                continue;
            }

            let line_color = if visible1 && visible2 {
                color
            } else {
                egui::Color32::from_rgba_unmultiplied(
                    color.r() / 2, color.g() / 2, color.b() / 2, 150,
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

pub fn draw_3d_view(
    ui: &mut egui::Ui,
    id: &str,
    constellations: &[(WalkerConstellation, Vec<SatelliteState>, usize, u8, usize, String)],
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
    planet_radius: f64,
    flattening: f64,
    sphere_renderer: Option<&Arc<Mutex<SphereRenderer>>>,
    body_key: (CelestialBody, Skin, TextureResolution),
    body_rotation: &Matrix3<f64>,
    sun_dir: [f32; 3],
    time: f64,
    ground_stations: &mut [GroundStation],
    areas_of_interest: &mut [AreaOfInterest],
    device_layers: &[DeviceLayer],
    body_rot_angle: f64,
    dragging_place: &mut Option<(usize, usize, bool, usize)>,
    drag_tab_planet: (usize, usize),
    detail_gl_info: Option<(glow::Texture, [f32; 4])>,
    geo_borders: &[Vec<(f64, f64)>],
    geo_cities: &[CityLabel],
) -> (Matrix3<f64>, f64) {
    let View3DFlags {
        show_orbits, show_axes, show_coverage, show_links, show_intra_links,
        hide_behind_earth, single_color, dark_mode, show_routing_paths,
        show_manhattan_path, show_shortest_path, show_asc_desc_colors,
        show_altitude_lines, render_planet, fixed_sizes, show_polar_circle,
        show_equator, show_terminator, earth_fixed_camera, use_gpu_rendering,
        show_clouds, show_day_night, show_stars, show_milky_way, show_borders, show_cities,
    } = flags;
    let max_altitude = constellations.iter()
        .map(|(c, _, _, _, _, _)| c.altitude_km)
        .fold(550.0_f64, |a, b| a.max(b));
    let orbit_radius = planet_radius + max_altitude;
    let axis_len = orbit_radius * 1.05;
    let planet_view_reference = planet_radius * 1.15;
    let margin = planet_view_reference / zoom;
    let zoom_factor = if fixed_sizes { 1.0 } else { zoom as f32 };
    let scaled_sat_radius = sat_radius * zoom_factor;
    let scaled_link_width = (link_width * zoom_factor).max(0.5);

    let use_gpu = sphere_renderer.is_some() && render_planet && use_gpu_rendering;

    // Draw sphere FIRST (before plot) so it renders behind
    if use_gpu {
        let rect = egui::Rect::from_min_size(ui.cursor().min, egui::Vec2::new(width, height));
        let renderer = sphere_renderer.unwrap().clone();
        let combined_rotation = if earth_fixed_camera {
            rotation
        } else {
            rotation * body_rotation
        };
        let inv_rotation = combined_rotation.transpose();
        let flat = flattening as f32;
        let aspect = width / height;
        let key = body_key;
        let scale = (planet_radius / margin) as f32;
        let atmosphere = match body_key.0 {
            CelestialBody::Earth => 1.0_f32,
            _ => 0.0,
        };

        let bg = ui.visuals().extreme_bg_color;
        let bg_color = [bg.r() as f32 / 255.0, bg.g() as f32 / 255.0, bg.b() as f32 / 255.0];
        let detail_info = detail_gl_info;
        let callback = egui::PaintCallback {
            rect,
            callback: Arc::new(egui_glow::CallbackFn::new(move |_info, painter| {
                let gl = painter.gl();
                let r = renderer.lock();
                if let Some((detail_tex, detail_bounds)) = detail_info {
                    let dt = DetailTexture {
                        width: 0,
                        height: 0,
                        bounds: DetailBounds {
                            min_lon: detail_bounds[0] as f64,
                            max_lon: detail_bounds[1] as f64,
                            min_lat: detail_bounds[2] as f64,
                            max_lat: detail_bounds[3] as f64,
                        },
                        gl_texture: Some(detail_tex),
                    };
                    r.paint(gl, key, &inv_rotation, flat as f64, aspect, scale, atmosphere, show_clouds, show_day_night, sun_dir, Some(&dt), show_stars, show_milky_way, bg_color);
                } else {
                    r.paint(gl, key, &inv_rotation, flat as f64, aspect, scale, atmosphere, show_clouds, show_day_night, sun_dir, None, show_stars, show_milky_way, bg_color);
                }
            })),
        };
        ui.painter().add(callback);
    }

    let plot = Plot::new(id)
        .data_aspect(1.0)
        .width(width)
        .height(height)
        .show_axes(false)
        .show_grid(false)
        .show_x(false)
        .show_y(false)
        .show_background(sphere_renderer.is_none() || !use_gpu_rendering)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .cursor_color(egui::Color32::TRANSPARENT);

    let mut surface_labels: Vec<([f64; 2], String, egui::Color32, bool, usize)> = Vec::new();
    let mut device_cluster_labels: Vec<([f64; 2], usize, egui::Color32)> = Vec::new();

    let response = plot.show(ui, |plot_ui| {
        let ground_stations = &*ground_stations;
        let areas_of_interest = &*areas_of_interest;
        plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
            [-margin, -margin],
            [margin, margin],
        ));

        let visual_earth_r = planet_radius * 0.95;
        let earth_r_sq = visual_earth_r * visual_earth_r;

        if show_orbits && !hide_behind_earth {
            for (constellation, _, color_offset, tle_kind, _, _) in constellations {
                if *tle_kind != 0 { continue; }
                for plane in 0..constellation.num_planes {
                    let orbit_pts = constellation.orbit_points_3d(plane, time);
                    let color = plane_color(if single_color { *color_offset } else { plane + color_offset });

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
                    let color = plane_color(if single_color { *color_offset } else { plane + color_offset });
                    let pts: PlotPoints = positions
                        .iter()
                        .filter_map(|s| {
                            if s.plane != plane {
                                return None;
                            }
                            let (rx, ry, rz) = rotate_point_matrix(s.x, s.y, s.z, &satellite_rotation);
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

        if render_planet {
            if use_gpu {
                // GPU rendering is handled by paint callback before the plot
            } else if let Some(tex) = earth_texture {
                let size = egui::Vec2::splat(planet_radius as f32 * 2.0);
                plot_ui.image(PlotImage::new(
                    "",
                    tex,
                    PlotPoint::new(0.0, 0.0),
                    size,
                ));
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
                        .stroke(egui::Stroke::new(2.0, egui::Color32::from_rgb(70, 130, 180))),
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
                plot_ui.line(Line::new("", border_pts).color(egui::Color32::WHITE).width(1.0));
            }

            if show_polar_circle {
                let polar_r = planet_radius * (1.0 - flattening);
                let circle_pts: PlotPoints = (0..=100)
                    .map(|i| {
                        let theta = 2.0 * PI * i as f64 / 100.0;
                        [polar_r * theta.cos(), polar_r * theta.sin()]
                    })
                    .collect();
                plot_ui.line(Line::new("", circle_pts)
                    .color(egui::Color32::from_rgb(255, 200, 50))
                    .width(1.0));
            }

            if show_terminator {
                let sun = Vector3::new(sun_dir[0] as f64, sun_dir[1] as f64, sun_dir[2] as f64).normalize();
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
                let terminator_pts: PlotPoints = (0..=100)
                    .map(|i| {
                        let theta = 2.0 * PI * i as f64 / 100.0;
                        let x = planet_radius * (u.x * theta.cos() + v.x * theta.sin());
                        let y = planet_radius * (u.y * theta.cos() + v.y * theta.sin());
                        let z = planet_radius * (u.z * theta.cos() + v.z * theta.sin());
                        let (sx, sy, _) = rotate_point_matrix(x, y, z, &term_rotation);
                        [sx, sy]
                    })
                    .collect();
                plot_ui.line(Line::new("", terminator_pts)
                    .color(egui::Color32::from_rgb(255, 180, 0))
                    .width(2.0));
            }
        }

        if show_coverage {
            for (constellation, positions, color_offset, _tle_kind, _, _) in constellations {
                let orbit_radius = planet_radius + constellation.altitude_km;
                let cone_half_angle = coverage_angle.to_radians();
                let max_earth_angle = (planet_radius / orbit_radius).acos();
                let earth_central_angle = (orbit_radius * cone_half_angle.sin() / planet_radius).asin();
                let angular_radius = earth_central_angle.min(max_earth_angle);
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
                                + (angular_radius.sin() * angle.sin())
                                    .atan2(lat.cos() * angular_radius.cos()
                                        - lat.sin() * angular_radius.sin() * angle.cos());

                            let x = planet_radius * clat.cos() * clon.cos();
                            let y = planet_radius * clat.sin();
                            let z = planet_radius * clat.cos() * clon.sin();

                            let (rx, ry, rz) = rotate_point_matrix(x, y, z, &satellite_rotation);
                            ([rx, ry], rz >= 0.0)
                        })
                        .collect();

                    let all_visible = coverage_pts.iter().all(|(_, vis)| *vis);
                    let color = plane_color(sat.plane + color_offset);

                    if all_visible {
                        let pts: Vec<[f64; 2]> = coverage_pts.iter().map(|(p, _)| *p).collect();
                        let fill = egui::Color32::from_rgba_unmultiplied(
                            color.r(), color.g(), color.b(), 60
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
                        plot_ui.line(Line::new("", PlotPoints::new(std::mem::take(&mut front_seg)))
                            .color(eq_color).width(1.5));
                    }
                    back_seg.push([rx, ry]);
                } else {
                    if !back_seg.is_empty() {
                        if !hide_behind_earth {
                            plot_ui.line(Line::new("", PlotPoints::new(std::mem::take(&mut back_seg)))
                                .color(dim_eq).width(1.0));
                        } else {
                            back_seg.clear();
                        }
                    }
                    front_seg.push([rx, ry]);
                }
            }
            if !front_seg.is_empty() {
                plot_ui.line(Line::new("", PlotPoints::new(front_seg))
                    .color(eq_color).width(1.5));
            }
            if !back_seg.is_empty() && !hide_behind_earth {
                plot_ui.line(Line::new("", PlotPoints::new(back_seg))
                    .color(dim_eq).width(1.0));
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
                            plot_ui.line(Line::new("", PlotPoints::new(std::mem::take(&mut front_seg)))
                                .color(border_color).width(1.0));
                        }
                        back_seg.push([rx, ry]);
                    } else {
                        if !back_seg.is_empty() {
                            if !hide_behind_earth {
                                plot_ui.line(Line::new("", PlotPoints::new(std::mem::take(&mut back_seg)))
                                    .color(dim_border).width(0.5));
                            } else {
                                back_seg.clear();
                            }
                        }
                        front_seg.push([rx, ry]);
                    }
                }
                if !front_seg.is_empty() {
                    plot_ui.line(Line::new("", PlotPoints::new(front_seg))
                        .color(border_color).width(1.0));
                }
                if !back_seg.is_empty() && !hide_behind_earth {
                    plot_ui.line(Line::new("", PlotPoints::new(back_seg))
                        .color(dim_border).width(0.5));
                }
            }
        }

        if show_cities && body_key.0 == CelestialBody::Earth {
            let min_pop = if zoom >= 8.0 { 0.0 }
                else if zoom >= 4.0 { 500_000.0 }
                else if zoom >= 2.0 { 2_000_000.0 }
                else { 5_000_000.0 };
            let max_cities = if zoom >= 8.0 { 200 }
                else if zoom >= 4.0 { 80 }
                else if zoom >= 2.0 { 30 }
                else { 15 };
            let city_color = egui::Color32::from_rgb(220, 220, 200);
            let mut count = 0usize;
            for city in geo_cities {
                if city.population < min_pop { continue; }
                if count >= max_cities { break; }
                let lat = city.lat.to_radians();
                let lon = (-city.lon).to_radians();
                let x = planet_radius * lat.cos() * lon.cos();
                let y = planet_radius * lat.sin();
                let z = planet_radius * lat.cos() * lon.sin();
                let (rx, ry, rz) = rotate_point_matrix(x, y, z, &surface_rotation);
                if !hide_behind_earth || rz >= 0.0 {
                    surface_labels.push(([rx, ry], city.name.clone(), city_color, false, 500_000 + count));
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
                        + (angular_radius.sin() * angle.sin())
                            .atan2(lat.cos() * angular_radius.cos()
                                - lat.sin() * angular_radius.sin() * angle.cos());

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
                    aoi.color.r(), aoi.color.g(), aoi.color.b(), aoi.color.a()
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
                        + (angular_radius.sin() * angle.sin())
                            .atan2(lat.cos() * angular_radius.cos()
                                - lat.sin() * angular_radius.sin() * angle.cos());

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
                    gs.color.r(), gs.color.g(), gs.color.b(), 50
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
            if layer.devices.is_empty() { continue; }

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
                    if !vis { continue; }
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
                let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> = std::collections::HashMap::new();
                for (i, (pos, vis)) in projected.iter().enumerate() {
                    if !vis { continue; }
                    let gx = (pos[0] / cell_size).floor() as i64;
                    let gy = (pos[1] / cell_size).floor() as i64;
                    grid.entry((gx, gy)).or_default().push(i);
                }

                for indices in grid.values() {
                    let count = indices.len();
                    let cx: f64 = indices.iter().map(|&i| projected[i].0[0]).sum::<f64>() / count as f64;
                    let cy: f64 = indices.iter().map(|&i| projected[i].0[1]).sum::<f64>() / count as f64;

                    let circle_r = (cell_size * 0.35).max(min_circle_r);
                    let fill = egui::Color32::from_rgba_unmultiplied(
                        layer.color.r(), layer.color.g(), layer.color.b(), 60,
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

        if show_orbits {
            for (constellation, _, color_offset, tle_kind, _, _) in constellations {
                if *tle_kind != 0 { continue; }
                for plane in 0..constellation.num_planes {
                    let orbit_pts = constellation.orbit_points_3d(plane, time);
                    let color = if show_routing_paths || show_asc_desc_colors {
                        egui::Color32::from_rgb(80, 80, 80)
                    } else {
                        plane_color(if single_color { *color_offset } else { plane + color_offset })
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
                egui::Color32::from_rgb(80, 80, 80)
            } else {
                egui::Color32::from_rgb(200, 200, 200)
            };
            let link_dim = egui::Color32::from_rgba_unmultiplied(50, 50, 60, 80);
            for (_, positions, _, _, _, _) in constellations {
                for sat in positions {
                    if let Some(neighbor_idx) = sat.neighbor_idx {
                        let neighbor = &positions[neighbor_idx];
                        let (rx1, ry1, rz1) = rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                        let (rx2, ry2, rz2) = rotate_point_matrix(neighbor.x, neighbor.y, neighbor.z, &satellite_rotation);
                        let visible1 = rz1 >= 0.0 || (rx1 * rx1 + ry1 * ry1) >= earth_r_sq;
                        let visible2 = rz2 >= 0.0 || (rx2 * rx2 + ry2 * ry2) >= earth_r_sq;
                        let both_visible = visible1 && visible2;
                        if hide_behind_earth && !both_visible {
                            continue;
                        }
                        let color = if both_visible { base_link_color } else { link_dim };
                        plot_ui.line(
                            Line::new("", PlotPoints::new(vec![[rx1, ry1], [rx2, ry2]]))
                                .color(color)
                                .width(scaled_link_width),
                        );
                    }
                }
            }
        }

        if show_intra_links {
            let base_link_color = if show_routing_paths || show_asc_desc_colors {
                egui::Color32::from_rgb(80, 80, 80)
            } else {
                egui::Color32::from_rgb(200, 200, 200)
            };
            let link_dim = egui::Color32::from_rgba_unmultiplied(50, 50, 60, 80);
            for (constellation, positions, _, _, _, _) in constellations {
                let sats_per_plane = constellation.sats_per_plane();
                for plane in 0..constellation.num_planes {
                    let plane_sats: Vec<_> = positions.iter()
                        .filter(|s| s.plane == plane)
                        .collect();
                    for i in 0..plane_sats.len() {
                        let sat = plane_sats[i];
                        let next = plane_sats[(i + 1) % sats_per_plane];
                        let (rx1, ry1, rz1) = rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                        let (rx2, ry2, rz2) = rotate_point_matrix(next.x, next.y, next.z, &satellite_rotation);
                        let visible1 = rz1 >= 0.0 || (rx1 * rx1 + ry1 * ry1) >= earth_r_sq;
                        let visible2 = rz2 >= 0.0 || (rx2 * rx2 + ry2 * ry2) >= earth_r_sq;
                        let both_visible = visible1 && visible2;
                        if hide_behind_earth && !both_visible {
                            continue;
                        }
                        let color = if both_visible { base_link_color } else { link_dim };
                        plot_ui.line(
                            Line::new("", PlotPoints::new(vec![[rx1, ry1], [rx2, ry2]]))
                                .color(color)
                                .width(scaled_link_width),
                        );
                    }
                }
            }
        }

        if show_routing_paths && !satellite_cameras.is_empty() {
            let manhattan_color = egui::Color32::from_rgb(255, 100, 100);
            let shortest_color = egui::Color32::from_rgb(100, 255, 100);

            for (cidx, (constellation, positions, _, _, _, _)) in constellations.iter().enumerate() {
                let tracked: Vec<_> = satellite_cameras.iter()
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

                        let src_sat = positions.iter().find(|s| s.plane == src.plane && s.sat_index == src.sat_index);
                        let dst_sat = positions.iter().find(|s| s.plane == dst.plane && s.sat_index == dst.sat_index);

                        let can_route = match (src_sat, dst_sat) {
                            (Some(_), Some(_)) => {
                                if is_star {
                                    let plane_diff_fwd = (dst.plane + num_planes - src.plane) % num_planes;
                                    let plane_diff_bwd = (src.plane + num_planes - dst.plane) % num_planes;
                                    let crosses_seam = plane_diff_fwd > num_planes / 2 && plane_diff_bwd > num_planes / 2;
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
                                src.plane, src.sat_index,
                                dst.plane, dst.sat_index,
                                num_planes, sats_per_plane,
                                is_star,
                            );
                            draw_routing_path(
                                plot_ui, &path, positions, &satellite_rotation,
                                manhattan_color, (scaled_link_width + 3.0).max(3.0), hide_behind_earth, earth_r_sq,
                            );
                        }

                        if show_shortest_path {
                            let path = compute_shortest_path(
                                src.plane, src.sat_index,
                                dst.plane, dst.sat_index,
                                num_planes, sats_per_plane,
                                positions,
                                is_star,
                            );
                            draw_routing_path(
                                plot_ui, &path, positions, &satellite_rotation,
                                shortest_color, (scaled_link_width + 3.0).max(3.0), hide_behind_earth, earth_r_sq,
                            );
                        }
                    }
                }
            }
        }

        for aoi in areas_of_interest {
            if let Some(gs_idx) = aoi.ground_station_idx {
                if let Some(gs) = ground_stations.get(gs_idx) {
                    let find_nearest_sat = |center_lat: f64, center_lon: f64, radius_km: f64, ascending_filter: Option<bool>|
                        -> Option<(usize, &WalkerConstellation, &Vec<SatelliteState>, &SatelliteState)>
                    {
                        let center_lat_rad = center_lat.to_radians();
                        let center_lon_rad = center_lon.to_radians() + body_rot_angle;
                        let max_angular_dist = radius_km / planet_radius;

                        let haversine_dist = |sat: &SatelliteState| -> f64 {
                            let sat_lat_rad = sat.lat.to_radians();
                            let sat_lon_rad = sat.lon.to_radians();
                            let dlat = sat_lat_rad - center_lat_rad;
                            let dlon = sat_lon_rad - center_lon_rad;
                            let a = (dlat / 2.0).sin().powi(2)
                                + center_lat_rad.cos() * sat_lat_rad.cos() * (dlon / 2.0).sin().powi(2);
                            2.0 * a.sqrt().asin()
                        };

                        let mut best: Option<(usize, &WalkerConstellation, &Vec<SatelliteState>, &SatelliteState, f64)> = None;

                        for (cidx, (cons, positions, _, tle_kind, _, _)) in constellations.iter().enumerate() {
                            if *tle_kind != 0 { continue; }
                            for sat in positions.iter() {
                                if let Some(asc) = ascending_filter {
                                    if sat.ascending != asc { continue; }
                                }
                                let dist = haversine_dist(sat);
                                if dist <= max_angular_dist && (best.is_none() || dist < best.as_ref().unwrap().4) {
                                    best = Some((cidx, cons, positions, sat, dist));
                                }
                            }
                        }

                        best.map(|(cidx, cons, positions, sat, _)| (cidx, cons, positions, sat))
                    };

                    let aoi_asc = find_nearest_sat(aoi.lat, aoi.lon, aoi.radius_km, Some(true));
                    let gs_asc = find_nearest_sat(gs.lat, gs.lon, gs.radius_km, Some(true));
                    let (aoi_result, gs_result) = if aoi_asc.is_some() && gs_asc.is_some() {
                        (aoi_asc, gs_asc)
                    } else {
                        let aoi_desc = find_nearest_sat(aoi.lat, aoi.lon, aoi.radius_km, Some(false));
                        let gs_desc = find_nearest_sat(gs.lat, gs.lon, gs.radius_km, Some(false));
                        (aoi_desc, gs_desc)
                    };

                    if let (Some((gs_cidx, gs_cons, gs_positions, gs_sat)),
                            Some((aoi_cidx, _, _, aoi_sat))) = (gs_result, aoi_result)
                    {
                        let path_color = egui::Color32::from_rgb(255, 255, 0);
                        let routing_width = (scaled_link_width + 3.0).max(3.0);

                        if gs_cidx == aoi_cidx {
                            let path = compute_shortest_path(
                                gs_sat.plane, gs_sat.sat_index,
                                aoi_sat.plane, aoi_sat.sat_index,
                                gs_cons.num_planes, gs_cons.sats_per_plane(),
                                gs_positions,
                                gs_cons.walker_type == WalkerType::Star,
                            );
                            draw_routing_path(
                                plot_ui, &path, gs_positions, &satellite_rotation,
                                path_color, routing_width, hide_behind_earth, earth_r_sq,
                            );
                        } else {
                            let (rx1, ry1, rz1) = rotate_point_matrix(gs_sat.x, gs_sat.y, gs_sat.z, &satellite_rotation);
                            let (rx2, ry2, rz2) = rotate_point_matrix(aoi_sat.x, aoi_sat.y, aoi_sat.z, &satellite_rotation);

                            let visible1 = rz1 >= 0.0 || (rx1 * rx1 + ry1 * ry1) >= earth_r_sq;
                            let visible2 = rz2 >= 0.0 || (rx2 * rx2 + ry2 * ry2) >= earth_r_sq;

                            if !hide_behind_earth || (visible1 && visible2) {
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(vec![[rx1, ry1], [rx2, ry2]]))
                                        .color(path_color)
                                        .width(routing_width),
                                );
                            }
                        }

                        let dot_size = scaled_sat_radius as f64 * 1.2;
                        let (rx1, ry1, _) = rotate_point_matrix(gs_sat.x, gs_sat.y, gs_sat.z, &satellite_rotation);
                        let (rx2, ry2, _) = rotate_point_matrix(aoi_sat.x, aoi_sat.y, aoi_sat.z, &satellite_rotation);
                        plot_ui.points(
                            Points::new("", PlotPoints::new(vec![[rx1, ry1], [rx2, ry2]]))
                                .radius(dot_size as f32)
                                .color(path_color),
                        );
                    }
                }
            }
        }

        for (constellation, positions, color_offset, tle_kind, orig_idx, _) in constellations {
            if *tle_kind != 0 {
                let is_debris = *tle_kind == 2;
                for sat in positions {
                    let color = plane_color(color_offset + sat.plane);
                    let dim_col = egui::Color32::from_rgba_unmultiplied(
                        color.r() / 2, color.g() / 2, color.b() / 2, 80,
                    );

                    let (rx, ry, rz) = rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                    let in_front = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;

                    let bg_color = if dark_mode {
                        egui::Color32::from_rgb(30, 30, 30)
                    } else {
                        egui::Color32::from_rgb(240, 240, 240)
                    };

                    if is_debris {
                        let d = scaled_sat_radius as f64 * 1.5 * margin / (width as f64 * 0.5);
                        let c = if in_front { color } else if !hide_behind_earth { dim_col } else { continue };
                        let w = if in_front { 1.0 } else { 0.5 };
                        plot_ui.line(
                            egui_plot::Line::new("", PlotPoints::new(vec![[rx - d, ry - d], [rx + d, ry + d]]))
                                .color(c).width(w),
                        );
                        plot_ui.line(
                            egui_plot::Line::new("", PlotPoints::new(vec![[rx - d, ry + d], [rx + d, ry - d]]))
                                .color(c).width(w),
                        );
                    } else {
                        if !hide_behind_earth && !in_front {
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(dim_col)
                                    .radius(scaled_sat_radius * 0.8)
                                    .filled(true),
                            );
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(bg_color)
                                    .radius(scaled_sat_radius * 0.4)
                                    .filled(true),
                            );
                        }
                        if in_front {
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(color)
                                    .radius(scaled_sat_radius)
                                    .filled(true),
                            );
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(bg_color)
                                    .radius(scaled_sat_radius * 0.5)
                                    .filled(true),
                            );
                        }
                    }
                }
                continue;
            }
            for plane in 0..constellation.num_planes {
                let base_color = plane_color(if single_color { *color_offset } else { plane + color_offset });

                for sat in positions.iter().filter(|s| s.plane == plane) {
                    let is_tracked = satellite_cameras.iter().any(|c|
                        c.constellation_idx == *orig_idx && c.plane == sat.plane && c.sat_index == sat.sat_index
                    );
                    let color = if show_asc_desc_colors {
                        if is_tracked {
                            if sat.ascending { COLOR_ASCENDING } else { COLOR_DESCENDING }
                        } else if sat.ascending {
                            egui::Color32::from_rgb(180, 140, 80)
                        } else {
                            egui::Color32::from_rgb(80, 120, 180)
                        }
                    } else {
                        base_color
                    };
                    let dim_col = egui::Color32::from_rgba_unmultiplied(
                        color.r() / 2, color.g() / 2, color.b() / 2, 80,
                    );

                    let (rx, ry, rz) = rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                    let in_front = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;

                    let bg_color = if dark_mode {
                        egui::Color32::from_rgb(30, 30, 30)
                    } else {
                        egui::Color32::from_rgb(240, 240, 240)
                    };

                    if !hide_behind_earth && !in_front {
                        plot_ui.points(
                            Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                .color(dim_col)
                                .radius(scaled_sat_radius * 0.8)
                                .filled(true),
                        );
                        if *tle_kind != 0 {
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
                        if *tle_kind != 0 {
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(bg_color)
                                    .radius(scaled_sat_radius * 0.5)
                                    .filled(true),
                            );
                        }
                    }

                    if constellation.altitude_km < 100.0 && (in_front || !hide_behind_earth) {
                        let d = scaled_sat_radius as f64 * 3.0 * margin / (width as f64 * 0.5);
                        let red = egui::Color32::from_rgb(255, 60, 60);
                        plot_ui.line(
                            egui_plot::Line::new("", PlotPoints::new(vec![[rx - d, ry - d], [rx + d, ry + d]]))
                                .color(red)
                                .width(2.0 * zoom_factor),
                        );
                        plot_ui.line(
                            egui_plot::Line::new("", PlotPoints::new(vec![[rx - d, ry + d], [rx + d, ry - d]]))
                                .color(red)
                                .width(2.0 * zoom_factor),
                        );
                    }

                    if show_altitude_lines && (in_front || !hide_behind_earth) {
                        let r = (sat.x * sat.x + sat.y * sat.y + sat.z * sat.z).sqrt();
                        let scale = planet_radius / r;
                        let (gx, gy, _) = rotate_point_matrix(sat.x * scale, sat.y * scale, sat.z * scale, &satellite_rotation);
                        let alt_color = egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 100);
                        plot_ui.line(
                            egui_plot::Line::new("", PlotPoints::new(vec![[rx, ry], [gx, gy]]))
                                .color(alt_color)
                                .width(0.5 * zoom_factor),
                        );
                    }
                }
            }
        }
    });

    let label_font_size = (14.0 * zoom as f32).clamp(10.0, 28.0);
    let mut label_rects: Vec<(egui::Rect, bool, usize)> = Vec::new();
    for (pos, name, color, is_gs, idx) in &surface_labels {
        let plot_pt = egui_plot::PlotPoint::new(pos[0], pos[1]);
        let screen_pos = response.transform.position_from_point(&plot_pt);
        let galley = ui.painter().layout_no_wrap(
            name.clone(),
            egui::FontId::proportional(label_font_size),
            *color,
        );
        let text_pos = screen_pos + egui::Vec2::new(-(galley.size().x / 2.0), -galley.size().y - 4.0);
        let bg_rect = egui::Rect::from_min_size(text_pos, galley.size()).expand(3.0);
        ui.painter().rect_filled(bg_rect, 3.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180));
        ui.painter().galley(text_pos, galley, *color);
        label_rects.push((bg_rect, *is_gs, *idx));
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
        let text_pos = screen_pos + egui::Vec2::new(-(galley.size().x / 2.0), -(galley.size().y / 2.0));
        let bg_rect = egui::Rect::from_min_size(text_pos, galley.size()).expand(2.0);
        ui.painter().rect_filled(bg_rect, 3.0, egui::Color32::from_rgba_unmultiplied(
            color.r() / 3, color.g() / 3, color.b() / 3, 200,
        ));
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
            if !seen.insert((name.as_str(), *color_offset)) { continue; }
            let color = plane_color(*color_offset);
            let square_rect = egui::Rect::from_min_size(
                egui::pos2(x, y + 1.0),
                egui::vec2(square_size, square_size),
            );
            ui.painter().rect_filled(square_rect, 2.0, color);
            let galley = ui.painter().layout_no_wrap(
                name.clone(),
                font.clone(),
                egui::Color32::WHITE,
            );
            let text_pos = egui::pos2(x + square_size + 4.0, y - 1.0);
            let bg_rect = egui::Rect::from_min_max(
                egui::pos2(x - 2.0, y - 2.0),
                egui::pos2(text_pos.x + galley.size().x + 2.0, y + galley.size().y + 2.0),
            );
            ui.painter().rect_filled(bg_rect, 3.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 160));
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
            }
            ui.painter().galley(text_pos, galley, egui::Color32::WHITE);
            y += 16.0;
        }
    }

    for (constellation, positions, color_offset, _tle_kind, orig_idx, _) in constellations {
        for sat in positions {
            for cam in satellite_cameras.iter_mut() {
                if cam.constellation_idx == *orig_idx && cam.plane == sat.plane && cam.sat_index == sat.sat_index {
                    let (rx, ry, _) = rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                    let plot_pt = egui_plot::PlotPoint::new(rx, ry);
                    let screen_pos = response.transform.position_from_point(&plot_pt);
                    cam.screen_pos = Some(screen_pos);

                    let color = plane_color(if single_color { *color_offset } else { sat.plane + color_offset });
                    ui.painter().circle_stroke(
                        screen_pos,
                        scaled_sat_radius * 2.5,
                        egui::Stroke::new(2.0, color),
                    );

                    let r = (sat.x * sat.x + sat.y * sat.y + sat.z * sat.z).sqrt();
                    let alt_km = r - constellation.planet_radius;
                    let vel_km_s = (constellation.planet_mu / r).sqrt();
                    let inv_body = body_rotation.transpose();
                    let body_pos = inv_body * Vector3::new(sat.x, sat.y, sat.z);
                    let ground_lat = (body_pos.y / r).asin().to_degrees();
                    let ground_lon = (-body_pos.z).atan2(body_pos.x).to_degrees();
                    let id = match &sat.name {
                        Some(name) => name.clone(),
                        None => format!("P{}S{}", sat.plane, sat.sat_index),
                    };
                    let text = if let (Some(inc), Some(mm)) = (sat.tle_inclination_deg, sat.tle_mean_motion) {
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
                            id,
                            ground_lat, ground_lon,
                            alt_km, vel_km_s,
                        )
                    };
                    let font = egui::FontId::proportional(12.0);
                    let galley = ui.painter().layout_no_wrap(text, font, egui::Color32::WHITE);
                    let text_pos = screen_pos + egui::Vec2::new(scaled_sat_radius * 3.0, -galley.size().y / 2.0);
                    let bg_rect = egui::Rect::from_min_size(text_pos, galley.size()).expand(4.0);
                    ui.painter().rect_filled(bg_rect, 4.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 200));
                    ui.painter().galley(text_pos, galley, egui::Color32::WHITE);
                }
            }
        }
    }

    let mut hovering_satellite = false;
    if let Some(hover_pos) = response.response.hover_pos() {
        let plot_pos = response.transform.value_from_position(hover_pos);
        let hover_threshold = margin * 0.025;

        'hover: for (_constellation, positions, color_offset, _, _, _) in constellations {
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
                    let color = plane_color(if single_color { *color_offset } else { sat.plane + color_offset });
                    ui.painter().circle_stroke(
                        screen_pt,
                        scaled_sat_radius * 2.0,
                        egui::Stroke::new(2.0, color),
                    );
                    hovering_satellite = true;
                    break 'hover;
                }
            }
        }

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
            let text = format!("{:.1}° {:.1}°", lat, lon);
            let font = egui::FontId::proportional(12.0);
            let text_pos = hover_pos + egui::Vec2::new(15.0, -15.0);
            let galley = ui.painter().layout_no_wrap(text, font, egui::Color32::WHITE);
            let rect = egui::Rect::from_min_size(
                text_pos - egui::Vec2::new(0.0, galley.size().y),
                galley.size(),
            ).expand(3.0);
            ui.painter().rect_filled(rect, 3.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180));
            ui.painter().galley(text_pos - egui::Vec2::new(0.0, galley.size().y), galley, egui::Color32::WHITE);
        }
    }

    if response.response.is_pointer_button_down_on() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
    } else if !hovering_satellite {
        if let Some(hover_pos) = response.response.hover_pos() {
            let plot_pos = response.transform.value_from_position(hover_pos);
            let on_label = label_rects.iter().any(|(rect, _, _)| rect.contains(hover_pos));
            let on_earth = plot_pos.x * plot_pos.x + plot_pos.y * plot_pos.y <= planet_radius * planet_radius;
            if on_label || on_earth {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }
        }
    }

    if response.response.drag_started() {
        if let Some(pos) = response.response.interact_pointer_pos() {
            let mut found = false;
            for (rect, is_gs, idx) in &label_rects {
                if rect.contains(pos) {
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

    let is_dragging_place = dragging_place.map_or(false, |(t, p, _, _)| t == drag_tab_planet.0 && p == drag_tab_planet.1);

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
                            if let Some(gs) = ground_stations.get_mut(idx) {
                                gs.lat = lat;
                                gs.lon = lon;
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
                    t*x*x + c,   t*x*y - s*z, t*x*z + s*y,
                    t*x*y + s*z, t*y*y + c,   t*y*z - s*x,
                    t*x*z - s*y, t*y*z + s*x, t*z*z + c,
                );
                rotation = rot * rotation;
            }

        }
    }

    if !response.response.dragged() && dragging_place.is_some_and(|(t, p, _, _)| t == drag_tab_planet.0 && p == drag_tab_planet.1) {
        *dragging_place = None;
    }

    if response.response.clicked() {
        if let Some(pos) = response.response.interact_pointer_pos() {
            let plot_pos = response.transform.value_from_position(pos);
            let click_x = plot_pos.x;
            let click_y = plot_pos.y;
            let click_threshold = margin * 0.03;

            'outer: for (_constellation, positions, _color_offset, _, orig_idx, _) in constellations {
                for sat in positions {
                    let (rx, ry, rz) = rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                    let earth_r_sq = (planet_radius * EARTH_VISUAL_SCALE).powi(2);
                    let visible = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;
                    if !visible && hide_behind_earth {
                        continue;
                    }
                    let dx = rx - click_x;
                    let dy = ry - click_y;
                    if dx * dx + dy * dy < click_threshold * click_threshold {
                        let existing = satellite_cameras.iter().find(|c|
                            c.constellation_idx == *orig_idx && c.plane == sat.plane && c.sat_index == sat.sat_index
                        );
                        if let Some(cam) = existing {
                            cameras_to_remove.push(cam.id);
                        } else {
                            let in_pending = pending_cameras.iter().any(|c|
                                c.constellation_idx == *orig_idx && c.plane == sat.plane && c.sat_index == sat.sat_index
                            );
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

    if response.response.hovered() {
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0 {
            let old_zoom = zoom;
            let factor = 1.0 + scroll as f64 * 0.001;
            zoom = (zoom * factor).clamp(0.01, 20000.0);

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
                        let a = Vector3::new(cx, cy, (r_sq - cx*cx - cy*cy).sqrt()).normalize();
                        let b = Vector3::new(tx, ty, (r_sq - tx*tx - ty*ty).sqrt()).normalize();
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
                                t*x*x+ca,    t*x*y-sa*z, t*x*z+sa*y,
                                t*x*y+sa*z,  t*y*y+ca,   t*y*z-sa*x,
                                t*x*z-sa*y,  t*y*z+sa*x, t*z*z+ca,
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
                        let scale = if horiz > 1e-10 { new_horiz / horiz } else { 1.0 };
                        let clamped = Vector3::new(center.x * scale, clamped_y, center.z * scale).normalize();
                        let right_raw = Vector3::new(clamped.z, 0.0, -clamped.x);
                        let right_len = right_raw.norm();
                        if right_len > 0.01 {
                            let right = right_raw / right_len;
                            let up = clamped.cross(&right);
                            let r0 = Matrix3::new(
                                right.x, right.y, right.z,
                                up.x, up.y, up.z,
                                clamped.x, clamped.y, clamped.z,
                            );
                            let up_screen = rotation * Vector3::new(0.0, 1.0, 0.0);
                            let bearing = up_screen.x.atan2(up_screen.y);
                            let cb = bearing.cos();
                            let sb = bearing.sin();
                            let rz = Matrix3::new(
                                 cb, sb, 0.0,
                                -sb, cb, 0.0,
                                0.0, 0.0, 1.0,
                            );
                            rotation = rz * r0;
                        }
                    }
                    let north_blend = (zoom.log2() / 4.0).clamp(0.0, 1.0);
                    if north_blend > 0.0 {
                        let up_screen = rotation * Vector3::new(0.0, 1.0, 0.0);
                        let bearing = up_screen.x.atan2(up_screen.y);
                        let zoom_octaves = (zoom / old_zoom).ln().abs() / (2.0_f64).ln();
                        let decay = (-north_blend * zoom_octaves * 1.5).exp();
                        let correction = bearing * (decay - 1.0);
                        let ca = correction.cos();
                        let sa = correction.sin();
                        rotation = Matrix3::new(
                            ca, sa, 0.0,
                            -sa, ca, 0.0,
                            0.0, 0.0, 1.0,
                        ) * rotation;
                    }
                }
            }
        }
        if let Some(touch) = ui.input(|i| i.multi_touch()) {
            let factor = touch.zoom_delta as f64;
            zoom = (zoom * factor).clamp(0.01, 20000.0);
        }
    }

    (rotation, zoom)
}

pub fn draw_ground_track(
    ui: &mut egui::Ui,
    id: &str,
    constellations: &[(WalkerConstellation, Vec<SatelliteState>, usize, u8, usize, String)],
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
        for (constellation, positions, color_offset, _tle_kind, _, _) in constellations {
            for plane in 0..constellation.num_planes {
                let color = plane_color(if single_color { *color_offset } else { plane + color_offset });
                let pts: PlotPoints = positions
                    .iter()
                    .filter(|s| s.plane == plane)
                    .map(|s| [s.lon, s.lat])
                    .collect();
                plot_ui.points(
                    Points::new("", pts)
                        .color(color)
                        .radius(sat_radius)
                        .filled(true),
                );
            }
        }

        plot_ui.line(
            Line::new("", PlotPoints::new(vec![[-180.0, 0.0], [180.0, 0.0]]))
                .color(egui::Color32::DARK_GRAY)
                .width(0.5),
        );
        plot_ui.line(
            Line::new("", PlotPoints::new(vec![[0.0, -90.0], [0.0, 90.0]]))
                .color(egui::Color32::DARK_GRAY)
                .width(0.5),
        );
    });
}

pub fn draw_torus(
    ui: &mut egui::Ui,
    id: &str,
    constellations: &[(WalkerConstellation, Vec<SatelliteState>, usize, u8, usize, String)],
    time: f64,
    rotation: Matrix3<f64>,
    width: f32,
    height: f32,
    sat_radius: f32,
    show_links: bool,
    single_color: bool,
    mut zoom: f64,
    satellite_cameras: &mut [SatelliteCamera],
    show_routing_paths: bool,
    show_manhattan_path: bool,
    show_shortest_path: bool,
    show_asc_desc_colors: bool,
    planet_radius: f64,
    pending_cameras: &mut Vec<SatelliteCamera>,
    camera_id_counter: &mut usize,
    cameras_to_remove: &mut Vec<usize>,
    link_width: f32,
    fixed_sizes: bool,
) -> (Matrix3<f64>, f64) {
    let (major_radius, minor_radius) = if let Some((constellation, _, _, _, _, _)) = constellations.first() {
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

        if let Some((constellation, _, _, _, _, _)) = constellations.first() {
            let orbit_radius = constellation.planet_radius + constellation.altitude_km;
            let earth_scale = planet_radius / orbit_radius;
            let earth_major = major_radius * earth_scale;
            let earth_minor = minor_radius * earth_scale;
            let earth_color = egui::Color32::from_rgb(60, 100, 140);
            let earth_dim = egui::Color32::from_rgba_unmultiplied(40, 70, 100, 150);

            let earth_point = |theta: f64, phi: f64| -> (f64, f64, f64) {
                let r = earth_major + earth_minor * phi.cos();
                let y = earth_minor * phi.sin();
                let x = r * theta.cos();
                let z = r * theta.sin();
                rotate_point_matrix(x, y, z, &display_rotation)
            };

            let earth_facing = |theta: f64, phi: f64| -> bool {
                let nx = phi.cos() * theta.cos();
                let ny = phi.sin();
                let nz = phi.cos() * theta.sin();
                let (_, _, nz_rot) = rotate_point_matrix(nx, ny, nz, &rotation);
                nz_rot >= 0.0
            };

            for ring in 0..12 {
                let phi = 2.0 * PI * ring as f64 / 12.0;
                let mut front_segment: Vec<[f64; 2]> = Vec::new();
                let mut back_segment: Vec<[f64; 2]> = Vec::new();
                for i in 0..=50 {
                    let theta = 2.0 * PI * i as f64 / 50.0;
                    let (rx, ry, _) = earth_point(theta, phi);
                    let facing = earth_facing(theta, phi);
                    if facing {
                        front_segment.push([rx, ry]);
                        if !back_segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut back_segment)))
                                    .color(earth_dim).width(1.0),
                            );
                        }
                    } else {
                        back_segment.push([rx, ry]);
                        if !front_segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut front_segment)))
                                    .color(earth_color).width(1.5),
                            );
                        }
                    }
                }
                if !front_segment.is_empty() {
                    plot_ui.line(Line::new("", PlotPoints::new(front_segment)).color(earth_color).width(1.5));
                }
                if !back_segment.is_empty() {
                    plot_ui.line(Line::new("", PlotPoints::new(back_segment)).color(earth_dim).width(1.0));
                }
            }

            for ring in 0..16 {
                let theta = 2.0 * PI * ring as f64 / 16.0;
                let mut front_segment: Vec<[f64; 2]> = Vec::new();
                let mut back_segment: Vec<[f64; 2]> = Vec::new();
                for i in 0..=50 {
                    let phi = 2.0 * PI * i as f64 / 50.0;
                    let (rx, ry, _) = earth_point(theta, phi);
                    let facing = earth_facing(theta, phi);
                    if facing {
                        front_segment.push([rx, ry]);
                        if !back_segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut back_segment)))
                                    .color(earth_dim).width(1.0),
                            );
                        }
                    } else {
                        back_segment.push([rx, ry]);
                        if !front_segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut front_segment)))
                                    .color(earth_color).width(1.5),
                            );
                        }
                    }
                }
                if !front_segment.is_empty() {
                    plot_ui.line(Line::new("", PlotPoints::new(front_segment)).color(earth_color).width(1.5));
                }
                if !back_segment.is_empty() {
                    plot_ui.line(Line::new("", PlotPoints::new(back_segment)).color(earth_dim).width(1.0));
                }
            }
        }

        for (constellation, positions, color_offset, _tle_kind, orig_idx, _) in constellations.iter() {
            let sats_per_plane = constellation.total_sats / constellation.num_planes;
            let orbit_radius = constellation.planet_radius + constellation.altitude_km;
            let period = 2.0 * PI * (orbit_radius.powi(3) / constellation.planet_mu).sqrt();
            let mean_motion = 2.0 * PI / period;
            let ecc = constellation.eccentricity;
            let omega = constellation.arg_periapsis_deg.to_radians();
            let raan_step = constellation.raan_step();
            let raan_offset = constellation.raan_offset_deg.to_radians();
            let plane_theta = |plane: usize| -> f64 {
                raan_offset + raan_step * plane as f64
            };

            let torus_pos = |plane: usize, sat_idx: usize| -> (f64, f64, f64) {
                let angle = plane_theta(plane);
                let sat_spacing = 2.0 * PI * sat_idx as f64 / sats_per_plane as f64;
                let phase = sat_spacing + mean_motion * time;
                torus_point(angle, phase, ecc, omega)
            };

            for plane in 0..constellation.num_planes {
                let angle = plane_theta(plane);
                let color = if show_routing_paths || show_asc_desc_colors {
                    egui::Color32::from_rgb(80, 80, 80)
                } else {
                    plane_color(if single_color { *color_offset } else { plane + color_offset })
                };
                let dim_col = egui::Color32::from_rgba_unmultiplied(
                    color.r(), color.g(), color.b(), 180,
                );

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
                                Line::new("", PlotPoints::new(std::mem::take(&mut back_segment)))
                                    .color(dim_col)
                                    .width(scaled_link_width),
                            );
                        }
                    } else {
                        back_segment.push([rx, ry]);
                        if !front_segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut front_segment)))
                                    .color(color)
                                    .width(scaled_link_width * 1.5),
                            );
                        }
                    }
                }
                if !front_segment.is_empty() {
                    plot_ui.line(Line::new("", PlotPoints::new(front_segment)).color(color).width(scaled_link_width * 1.5));
                }
                if !back_segment.is_empty() {
                    plot_ui.line(Line::new("", PlotPoints::new(back_segment)).color(dim_col).width(scaled_link_width));
                }
            }

            if show_links {
                let base_link_color = if show_routing_paths || show_asc_desc_colors {
                    egui::Color32::from_rgb(80, 80, 80)
                } else {
                    egui::Color32::from_rgb(150, 150, 150)
                };
                let link_dim = egui::Color32::from_rgba_unmultiplied(50, 50, 60, 100);
                for sat in positions {
                    if let Some(neighbor_idx) = sat.neighbor_idx {
                        let neighbor = &positions[neighbor_idx];
                        let angle1 = plane_theta(sat.plane);
                        let angle2 = plane_theta(neighbor.plane);
                        let phase1 = 2.0 * PI * sat.sat_index as f64 / sats_per_plane as f64 + mean_motion * time;
                        let phase2 = 2.0 * PI * neighbor.sat_index as f64 / sats_per_plane as f64 + mean_motion * time;

                        let (x1, y1, _) = torus_pos(sat.plane, sat.sat_index);
                        let (x2, y2, _) = torus_pos(neighbor.plane, neighbor.sat_index);
                        let facing1 = is_facing_camera(angle1, phase1);
                        let facing2 = is_facing_camera(angle2, phase2);
                        let color = if facing1 && facing2 { base_link_color } else { link_dim };
                        plot_ui.line(
                            Line::new("", PlotPoints::new(vec![[x1, y1], [x2, y2]]))
                                .color(color)
                                .width(scaled_link_width),
                        );
                    }
                }
            }

            for plane in 0..constellation.num_planes {
                let base_color = plane_color(if single_color { *color_offset } else { plane + color_offset });
                let angle = plane_theta(plane);

                for sat in positions.iter().filter(|s| s.plane == plane) {
                    let is_tracked = satellite_cameras.iter().any(|c|
                        c.constellation_idx == *orig_idx && c.plane == sat.plane && c.sat_index == sat.sat_index
                    );
                    let color = if show_asc_desc_colors {
                        if is_tracked {
                            if sat.ascending { COLOR_ASCENDING } else { COLOR_DESCENDING }
                        } else if sat.ascending {
                            egui::Color32::from_rgb(180, 140, 80)
                        } else {
                            egui::Color32::from_rgb(80, 120, 180)
                        }
                    } else {
                        base_color
                    };
                    let dim_col = egui::Color32::from_rgba_unmultiplied(
                        color.r(), color.g(), color.b(), 140,
                    );

                    let phase = 2.0 * PI * sat.sat_index as f64 / sats_per_plane as f64 + mean_motion * time;
                    let (x, y, _) = torus_pos(sat.plane, sat.sat_index);
                    let facing = is_facing_camera(angle, phase);
                    let (c, r) = if facing { (color, scaled_sat_radius) } else { (dim_col, scaled_sat_radius * 0.8) };
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
                let tracked: Vec<_> = satellite_cameras.iter()
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

                            let src_sat = positions.iter().find(|s| s.plane == src.plane && s.sat_index == src.sat_index);
                            let dst_sat = positions.iter().find(|s| s.plane == dst.plane && s.sat_index == dst.sat_index);

                            let can_route = match (src_sat, dst_sat) {
                                (Some(_), Some(_)) => {
                                    if is_star {
                                        let plane_diff_fwd = (dst.plane + num_planes - src.plane) % num_planes;
                                        let plane_diff_bwd = (src.plane + num_planes - dst.plane) % num_planes;
                                        let crosses_seam = plane_diff_fwd > num_planes / 2 && plane_diff_bwd > num_planes / 2;
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
                                    src.plane, src.sat_index,
                                    dst.plane, dst.sat_index,
                                    num_planes, sats_per_plane,
                                    is_star,
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
                                    src.plane, src.sat_index,
                                    dst.plane, dst.sat_index,
                                    num_planes, sats_per_plane,
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
                        }
                    }
                }
            }
        }
    });

    if response.response.dragged() && !response.response.drag_started() {
        let drag = response.response.drag_delta();
        let sens = 0.01 / zoom.max(1.0);
        let delta_rot = rotation_from_drag(drag.x as f64 * sens, drag.y as f64 * sens);
        user_rotation = delta_rot * user_rotation;
    }

    if response.response.hovered() {
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0 {
            let factor = 1.0 + scroll as f64 * 0.001;
            zoom = (zoom * factor).clamp(0.01, 20000.0);
        }
        if let Some(touch) = ui.input(|i| i.multi_touch()) {
            let factor = touch.zoom_delta as f64;
            zoom = (zoom * factor).clamp(0.01, 20000.0);
        }
    }

    if let Some(pos) = response.response.interact_pointer_pos() {
        if response.response.clicked() {
            let click_x = response.transform.value_from_position(pos).x;
            let click_y = response.transform.value_from_position(pos).y;
            let (major_radius, minor_radius) = if let Some((constellation, _, _, _, _, _)) = constellations.first() {
                let sats_per_plane = constellation.sats_per_plane();
                let orbit_radius = planet_radius + constellation.altitude_km;
                let inclination_rad = constellation.inclination_deg.to_radians();
                let inclination_factor = inclination_rad.sin().abs().max(0.002);
                let altitude_factor = orbit_radius / (planet_radius + 500.0);
                let major = altitude_factor * (sats_per_plane as f64 / constellation.num_planes as f64);
                let minor_base = altitude_factor * inclination_factor;
                let minor = minor_base.max(major * inclination_factor);
                let scale = 2.0 / (major + minor).max(1.0);
                (major * scale, minor * scale)
            } else {
                (2.0, 0.8)
            };
            let margin = (major_radius + minor_radius) * 1.3 / zoom;
            let click_threshold = margin * 0.05;

            let torus_point_click = |theta: f64, phi: f64, ecc: f64, omega: f64| -> (f64, f64, f64) {
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
                    let phase = 2.0 * PI * sat.sat_index as f64 / sats_per_plane as f64 + mean_motion * time;
                    let (tx, ty, _) = torus_point_click(angle, phase, ecc, omega);

                    let dx = tx - click_x;
                    let dy = ty - click_y;
                    if dx * dx + dy * dy < click_threshold * click_threshold {
                        let existing = satellite_cameras.iter().find(|c|
                            c.constellation_idx == *orig_idx && c.plane == sat.plane && c.sat_index == sat.sat_index
                        );
                        if let Some(cam) = existing {
                            cameras_to_remove.push(cam.id);
                        } else {
                            let in_pending = pending_cameras.iter().any(|c|
                                c.constellation_idx == *orig_idx && c.plane == sat.plane && c.sat_index == sat.sat_index
                            );
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

pub fn plane_color(plane: usize) -> egui::Color32 {
    COLORS[plane % COLORS.len()]
}

pub const COLORS: [egui::Color32; 16] = [
    egui::Color32::from_rgb(255, 99, 71),
    egui::Color32::from_rgb(50, 205, 50),
    egui::Color32::from_rgb(30, 144, 255),
    egui::Color32::from_rgb(255, 215, 0),
    egui::Color32::from_rgb(238, 130, 238),
    egui::Color32::from_rgb(0, 206, 209),
    egui::Color32::from_rgb(255, 140, 0),
    egui::Color32::from_rgb(147, 112, 219),
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
