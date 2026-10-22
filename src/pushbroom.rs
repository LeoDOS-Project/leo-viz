//! Illustrative continuous nadir imaging using the displayed satellite positions.
//! Trails are observed simulation history, not completed acquisition jobs.
use crate::drawing::{plane_color, DrawConstellationData};
use crate::globe_overlay::{front_polygon, front_segment, line_quad, project_surface};
use eframe::egui::{self, Color32, Mesh, Rect, Shape};
use egui_plot::{PlotBounds, PlotGeometry, PlotItem, PlotItemBase, PlotPoint, PlotTransform};
use leo_sim::geometry::{norm, scale, sub, Vector};
use nalgebra::{Matrix3, Vector3};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    hash::{Hash, Hasher},
    ops::RangeInclusive,
};

#[derive(Clone, Copy)]
pub(crate) struct Settings {
    pub enabled: bool,
    pub swath_km: f64,
    pub trail_s: f64,
    pub show_fan: bool,
    pub show_images: bool,
    pub image_length_km: f64,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            enabled: false,
            swath_km: 20.0,
            trail_s: 60.0,
            show_fan: false,
            show_images: false,
            image_length_km: 50.0,
        }
    }
}
impl Settings {
    pub fn ui(&mut self, ui: &mut egui::Ui, enabled: bool) {
        ui.add_enabled(
            enabled,
            egui::Checkbox::new(&mut self.enabled, "Pushbroom sweeps"),
        )
        .on_hover_text("Show each satellite's ground scanline and recent swept strip on Earth");
        ui.add_enabled_ui(enabled, |ui| {
            ui.horizontal(|ui| {
                ui.label("Swath width");
                ui.add(egui::DragValue::new(&mut self.swath_km).range(1.0..=500.0).speed(1.0).suffix(" km"));
            });
            ui.add_enabled_ui(self.enabled, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Recent trail");
                    ui.add(egui::DragValue::new(&mut self.trail_s).range(5.0..=300.0).speed(5.0).suffix(" s"));
                });
                ui.checkbox(&mut self.show_fan, "Show sensor fan");
                ui.add_enabled(cfg!(not(target_arch = "wasm32")), egui::Checkbox::new(&mut self.show_images, "Pushbroom images"))
                    .on_hover_text("Click a satellite to open its swept-ground image from Esri World Imagery; desktop only");
                ui.add_enabled_ui(self.show_images && cfg!(not(target_arch = "wasm32")), |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Image length");
                        ui.add(egui::DragValue::new(&mut self.image_length_km).range(1.0..=2000.0).speed(1.0).suffix(" km"));
                    });
                });
            });
        });
    }
}

type Key = (u64, usize, usize);
#[derive(Clone, Copy)]
struct Sample {
    time: f64,
    centre: Vector,
    radius_ratio: f64,
}
#[derive(Clone)]
struct Track {
    samples: VecDeque<Sample>,
    head: Sample,
    color: Color32,
}
#[derive(Clone, Default)]
pub(crate) struct History {
    tracks: HashMap<Key, Track>,
    time: Option<f64>,
    sources: HashMap<usize, Vec<u64>>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub(crate) struct ImageLine {
    pub edges: [Vector; 2],
    pub distance_km: f64,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub(crate) struct ImageStrip {
    pub lines: Vec<ImageLine>,
    pub time: f64,
    pub source: u64,
}

#[cfg(not(target_arch = "wasm32"))]
impl History {
    pub fn image_strip(
        &self,
        source_index: usize,
        plane: usize,
        satellite: usize,
        settings: Settings,
        radius: f64,
    ) -> Option<ImageStrip> {
        let source = *self
            .sources
            .get(&source_index)?
            .iter()
            .find(|&&source| self.tracks.contains_key(&(source, plane, satellite)))?;
        let track = self.tracks.get(&(source, plane, satellite))?;
        let mut samples: Vec<_> = track.samples.iter().copied().collect();
        if samples.last().is_some_and(|s| s.time != track.head.time) {
            samples.push(track.head);
        }
        if samples.len() < 2 {
            return None;
        }
        let cutoff = track.head.time - settings.trail_s;
        if samples[0].time < cutoff && samples[1].time > cutoff {
            let fraction = (cutoff - samples[0].time) / (samples[1].time - samples[0].time);
            samples[0].centre = normalized(std::array::from_fn(|i| {
                samples[0].centre[i] * (1.0 - fraction) + samples[1].centre[i] * fraction
            }))?;
            samples[0].time = cutoff;
        }
        let mut lines: Vec<ImageLine> = Vec::new();
        let mut distance = 0.0;
        for i in (0..samples.len()).rev() {
            if i + 1 < samples.len() {
                let a = samples[i].centre;
                let b = samples[i + 1].centre;
                distance +=
                    norm(cross(a, b)).atan2(a.iter().zip(b).map(|(a, b)| a * b).sum()) * radius;
            }
            let direction = sub(
                samples[(i + 1).min(samples.len() - 1)].centre,
                samples[i.saturating_sub(1)].centre,
            );
            let mut edges = scanline(
                samples[i].centre,
                direction,
                settings.swath_km,
                radius,
                samples[i].radius_ratio,
            )?;
            if distance > settings.image_length_km {
                let previous = lines.last()?;
                let fraction = (settings.image_length_km - previous.distance_km)
                    / (distance - previous.distance_km);
                for side in 0..2 {
                    edges[side] = normalized(std::array::from_fn(|axis| {
                        previous.edges[side][axis] * (1.0 - fraction) + edges[side][axis] * fraction
                    }))?;
                }
                lines.push(ImageLine {
                    edges,
                    distance_km: settings.image_length_km,
                });
                break;
            }
            lines.push(ImageLine {
                edges,
                distance_km: distance,
            });
        }
        (distance > 0.001).then_some(ImageStrip {
            lines,
            time: track.head.time,
            source,
        })
    }
}

fn cross(a: Vector, b: Vector) -> Vector {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn normalized(a: Vector) -> Option<Vector> {
    let n = norm(a);
    (n.is_finite() && n > 1e-12).then(|| scale(a, 1.0 / n))
}

/// Endpoints of a spherical cross-track scanline; clamp to the local horizon.
fn scanline(
    centre: Vector,
    direction: Vector,
    width: f64,
    radius: f64,
    radius_ratio: f64,
) -> Option<[Vector; 2]> {
    let side = normalized(cross(centre, direction))?;
    let half = (width / (2.0 * radius)).min((1.0 / radius_ratio).clamp(0.0, 1.0).acos());
    let (s, c) = half.sin_cos();
    Some([
        std::array::from_fn(|i| centre[i] * c + side[i] * s),
        std::array::from_fn(|i| centre[i] * c - side[i] * s),
    ])
}

impl History {
    pub fn clear(&mut self) {
        self.tracks.clear();
        self.time = None;
        self.sources.clear();
    }

    pub fn update(
        &mut self,
        time: f64,
        rotation: &Matrix3<f64>,
        radius: f64,
        constellations: &[DrawConstellationData],
        settings: Settings,
    ) {
        if !settings.enabled || !time.is_finite() {
            self.clear();
            return;
        }
        if self
            .time
            .is_some_and(|last| time < last || time - last >= settings.trail_s)
        {
            self.clear();
        }
        let mut seen = HashSet::new();
        self.sources.clear();
        for (draw_index, (orbit, positions, color_offset, kind, source_index, label)) in
            constellations.iter().enumerate()
        {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            (
                draw_index,
                kind,
                source_index,
                label,
                orbit.total_sats,
                orbit.num_planes,
            )
                .hash(&mut hasher);
            for v in [
                orbit.altitude_km,
                orbit.inclination_deg,
                orbit.phasing,
                orbit.raan_offset_deg,
                orbit.eccentricity,
                orbit.arg_periapsis_deg,
                orbit.raan_spacing_deg.unwrap_or(0.0),
                orbit.sat_spacing_km.unwrap_or(0.0),
            ] {
                v.to_bits().hash(&mut hasher);
            }
            let source = hasher.finish();
            self.sources.entry(*source_index).or_default().push(source);
            for sat in positions {
                let body = rotation.transpose() * Vector3::new(sat.x, sat.y, sat.z);
                let xyz = [body.x, body.y, body.z];
                let r = norm(xyz);
                if !r.is_finite() || r <= radius {
                    continue;
                }
                let sample = Sample {
                    time,
                    centre: scale(xyz, 1.0 / r),
                    radius_ratio: r / radius,
                };
                let key = (source, sat.plane, sat.sat_index);
                seen.insert(key);
                self.record(
                    key,
                    sample,
                    plane_color(color_offset + sat.plane),
                    settings.trail_s,
                    radius,
                );
            }
        }
        self.tracks.retain(|key, _| seen.contains(key));
        self.time = Some(time);
    }

    fn record(&mut self, key: Key, sample: Sample, color: Color32, trail: f64, radius: f64) {
        let track = self.tracks.entry(key).or_insert_with(|| Track {
            samples: VecDeque::new(),
            head: sample,
            color,
        });
        let elapsed = sample.time - track.head.time;
        let jump_km = norm(sub(sample.centre, track.head.centre)) * radius;
        // Also handle orbit edits/teleports while paused. Do not draw a strip
        // through places that were never visited by the displayed spacecraft.
        if elapsed < 0.0 || jump_km > elapsed.max(0.0) * 15.0 + 0.001 {
            track.samples.clear();
        }
        let spacing = (trail / 48.0).max(0.25);
        if track
            .samples
            .back()
            .is_none_or(|last| sample.time - last.time >= spacing)
        {
            track.samples.push_back(sample);
        }
        track.head = sample;
        track.color = color;
        // Keep one sample before the cutoff so drawing can clip at the exact age.
        while track.samples.len() > 1 && track.samples[1].time <= sample.time - trail {
            track.samples.pop_front();
        }
        while track.samples.len() > 64 {
            track.samples.pop_front();
        }
    }
}

pub(crate) struct Overlay<'a> {
    base: PlotItemBase,
    history: &'a History,
    settings: Settings,
    rotation: Matrix3<f64>,
    radius: f64,
    polar_scale: f64,
}

/// Retain the full history, but draw fewer segments when zoomed out. Bound
/// angular spacing for curvature and temporal spacing for the fading gradient.
fn drawing_samples(samples: &[Sample], pixel_radius: f64, trail_s: f64) -> Vec<usize> {
    if samples.len() < 2 {
        return (0..samples.len()).collect();
    }
    let angle = (2.0 / pixel_radius.max(1.0))
        .sqrt()
        .min(2.0_f64.to_radians());
    let chord = 2.0 * (angle / 2.0).sin();
    let mut indices = vec![0];
    let mut last = 0;
    for i in 1..samples.len() - 1 {
        let next = samples[i + 1];
        if norm(sub(next.centre, samples[last].centre)) > chord
            || next.time - samples[last].time > trail_s / 8.0
        {
            indices.push(i);
            last = i;
        }
    }
    indices.push(samples.len() - 1);
    indices
}

impl<'a> Overlay<'a> {
    pub fn new(
        history: &'a History,
        settings: Settings,
        rotation: Matrix3<f64>,
        radius: f64,
        flattening: f64,
    ) -> Self {
        Self {
            base: PlotItemBase::new("Pushbroom sweeps".into()),
            history,
            settings,
            rotation,
            radius,
            polar_scale: 1.0 - flattening,
        }
    }

    fn mesh(&self, transform: &PlotTransform) -> Mesh {
        let mut mesh = Mesh::default();
        let Some(now) = self.history.time else {
            return mesh;
        };
        let project = |p: Vector| {
            project_surface(
                Vector3::from(p),
                &self.rotation,
                self.radius,
                self.polar_scale,
            )
        };
        let screen = |p: Vector3<f64>| transform.position_from_point(&PlotPoint::new(p.x, p.y));
        let pixel_radius = self.radius
            * transform
                .dpos_dvalue_x()
                .abs()
                .max(transform.dpos_dvalue_y().abs());
        for track in self.history.tracks.values() {
            let mut samples: Vec<_> = track.samples.iter().copied().collect();
            if samples.last().is_some_and(|s| s.time != track.head.time) {
                samples.push(track.head);
            }
            if samples.len() < 2 {
                continue;
            }
            // The head-to-history chord plus half the swath bounds every
            // endpoint. Avoid constructing geometry entirely behind Earth.
            let extent = samples
                .iter()
                .map(|s| norm(sub(s.centre, track.head.centre)))
                .fold(0.0_f64, f64::max)
                + self.settings.swath_km / (2.0 * self.radius);
            let head = project(track.head.centre);
            if head.1 + extent / self.polar_scale < 0.0 {
                continue;
            }
            if !self.settings.show_fan
                && !transform.frame().intersects(Rect::from_center_size(
                    screen(head.0),
                    egui::Vec2::splat((2.0 * extent * pixel_radius + 6.0) as f32),
                ))
            {
                continue;
            }
            let cutoff = now - self.settings.trail_s;
            if samples[0].time < cutoff && samples[1].time > cutoff {
                let t = (cutoff - samples[0].time) / (samples[1].time - samples[0].time);
                samples[0].centre = normalized(std::array::from_fn(|i| {
                    samples[0].centre[i] * (1.0 - t) + samples[1].centre[i] * t
                }))
                .unwrap_or(samples[1].centre);
                samples[0].time = cutoff;
            }
            let indices = drawing_samples(&samples, pixel_radius, self.settings.trail_s);
            let edges: Vec<_> = indices
                .iter()
                .map(|&i| {
                    let s = samples[i];
                    let before = samples[i.saturating_sub(1)].centre;
                    let after = samples[(i + 1).min(samples.len() - 1)].centre;
                    scanline(
                        s.centre,
                        sub(after, before),
                        self.settings.swath_km,
                        self.radius,
                        s.radius_ratio,
                    )
                    .map(|edge| edge.map(project))
                })
                .collect();
            let rgba = |alpha| {
                Color32::from_rgba_unmultiplied(
                    track.color.r(),
                    track.color.g(),
                    track.color.b(),
                    alpha,
                )
            };
            for i in 1..indices.len() {
                let (Some(a), Some(b)) = (edges[i - 1], edges[i]) else {
                    continue;
                };
                let fade = (1.0
                    - (now - (samples[indices[i - 1]].time + samples[indices[i]].time) / 2.0)
                        / self.settings.trail_s)
                    .clamp(0.0, 1.0);
                let points = [a[0], a[1], b[1], b[0]];
                let polygon = front_polygon(&points);
                if polygon.len() >= 3 {
                    let pts: Vec<_> = polygon.into_iter().map(screen).collect();
                    let bounds = Rect::from_points(&pts);
                    if transform.frame().intersects(bounds) {
                        let first = mesh.vertices.len() as u32;
                        for p in pts.iter().copied() {
                            mesh.colored_vertex(p, rgba((65.0 * fade) as u8));
                        }
                        for j in 1..pts.len() - 1 {
                            mesh.add_triangle(first, first + j as u32, first + j as u32 + 1);
                        }
                    }
                }
                for side in 0..2 {
                    if let Some((a, b)) = front_segment(a[side], b[side]) {
                        line_quad(
                            &mut mesh,
                            screen(a),
                            screen(b),
                            rgba((150.0 * fade) as u8),
                            0.8,
                        );
                    }
                }
            }
            if let Some(Some(edge)) = edges.last() {
                if let Some((a, b)) = front_segment(edge[0], edge[1]) {
                    line_quad(&mut mesh, screen(a), screen(b), rgba(255), 2.5);
                }
                if self.settings.show_fan && project(track.head.centre).1 >= 0.0 {
                    let satellite = project_surface(
                        Vector3::from(scale(track.head.centre, track.head.radius_ratio)),
                        &self.rotation,
                        self.radius,
                        1.0,
                    )
                    .0;
                    if satellite.z >= 0.0 {
                        for &p in edge {
                            line_quad(&mut mesh, screen(satellite), screen(p.0), rgba(65), 0.8);
                        }
                    }
                }
            }
        }
        mesh
    }
}
impl PlotItem for Overlay<'_> {
    fn shapes(&self, _: &egui::Ui, transform: &PlotTransform, shapes: &mut Vec<Shape>) {
        shapes.push(Shape::mesh(self.mesh(transform)));
    }
    fn initialize(&mut self, _: RangeInclusive<f64>) {}
    fn color(&self) -> Color32 {
        Color32::LIGHT_GREEN
    }
    fn allow_hover(&self) -> bool {
        false
    }
    fn geometry(&self) -> PlotGeometry<'_> {
        PlotGeometry::None
    }
    fn bounds(&self) -> PlotBounds {
        PlotBounds::NOTHING
    }
    fn base(&self) -> &PlotItemBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut PlotItemBase {
        &mut self.base
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use leo_sim::geometry::{angular_distance, dot, ground_position};
    const R: f64 = leo_sim::geometry::EARTH_RADIUS_KM;
    fn sample(time: f64) -> Sample {
        Sample {
            time,
            centre: ground_position(0.0, -90.0 + time * 0.05, 1.0, 0.0),
            radius_ratio: 1.08,
        }
    }
    fn transform() -> PlotTransform {
        PlotTransform::new(
            Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 800.0)),
            PlotBounds::new_symmetrical(7000.0),
            [false, false],
        )
    }
    #[test]
    fn width_is_physical_and_perpendicular_to_motion_at_equator_poles_and_dateline() {
        for (lat, lon) in [(0.0, 0.0), (89.9, 179.9), (-89.9, -179.9), (45.0, 180.0)] {
            let centre = ground_position(lat, lon, 1.0, 0.0);
            let direction = sub(ground_position(lat + 0.01, lon + 0.01, 1.0, 0.0), centre);
            let [left, right] = scanline(centre, direction, 20.0, R, 1.08).unwrap();
            assert!((angular_distance(left, right) * R - 20.0).abs() < 1e-6);
            assert!((norm(left) - 1.0).abs() < 1e-12 && (norm(right) - 1.0).abs() < 1e-12);
            assert!(dot(sub(left, right), direction).abs() < 1e-12);
        }
        assert!(scanline([1.0, 0.0, 0.0], [0.0; 3], 20.0, R, 1.08).is_none());
    }
    #[test]
    fn history_is_bounded_pausing_does_not_duplicate_and_teleports_clear() {
        let mut h = History::default();
        for i in 0..2000 {
            h.record((0, 0, 0), sample(i as f64 * 0.1), Color32::GREEN, 60.0, R);
        }
        let n = h.tracks[&(0, 0, 0)].samples.len();
        assert!(n <= 64);
        let head = h.tracks[&(0, 0, 0)].head;
        for _ in 0..100 {
            h.record((0, 0, 0), head, Color32::GREEN, 60.0, R);
        }
        assert_eq!(h.tracks[&(0, 0, 0)].samples.len(), n);
        let teleport = Sample {
            centre: scale(head.centre, -1.0),
            ..head
        };
        h.record((0, 0, 0), teleport, Color32::GREEN, 60.0, R);
        assert_eq!(h.tracks[&(0, 0, 0)].samples.len(), 1);
    }

    #[test]
    fn image_strip_uses_recorded_scanlines_and_clips_its_ground_length() {
        let mut history = History::default();
        history.sources.insert(2, vec![0]);
        let settings = Settings {
            enabled: true,
            image_length_km: 10.0,
            ..Default::default()
        };
        assert!(history.image_strip(2, 0, 0, settings, R).is_none());
        for time in 0..20 {
            history.record((0, 0, 0), sample(time as f64), Color32::GREEN, 60.0, R);
        }
        let strip = history.image_strip(2, 0, 0, settings, R).unwrap();
        assert_eq!(strip.time, 19.0);
        assert_eq!(strip.lines[0].distance_km, 0.0);
        assert!((strip.lines.last().unwrap().distance_km - 10.0).abs() < 1e-8);
        assert!(strip
            .lines
            .windows(2)
            .all(|lines| lines[1].distance_km > lines[0].distance_km));
        for line in strip.lines {
            assert!(
                (angular_distance(line.edges[0], line.edges[1]) * R - settings.swath_km).abs()
                    < 0.01
            );
        }
        assert!(history.image_strip(3, 0, 0, settings, R).is_none());
    }
    #[test]
    fn globe_mesh_clips_hidden_side_and_fan_adds_geometry() {
        let mut h = History::default();
        for i in 0..60 {
            h.record((0, 0, 0), sample(i as f64), Color32::GREEN, 60.0, R);
        }
        h.time = Some(59.0);
        let s = Settings {
            enabled: true,
            ..Default::default()
        };
        let front = Overlay::new(&h, s, Matrix3::identity(), R, 0.0).mesh(&transform());
        assert!(front.is_valid() && !front.vertices.is_empty());
        assert!(front
            .vertices
            .iter()
            .all(|v| v.pos.x.is_finite() && v.pos.y.is_finite()));
        let fan = Overlay::new(
            &h,
            Settings {
                show_fan: true,
                ..s
            },
            Matrix3::identity(),
            R,
            0.0,
        )
        .mesh(&transform());
        assert!(fan.vertices.len() > front.vertices.len());
        let rotation = crate::math::rotation_from_drag(std::f64::consts::PI, 0.0);
        assert!(Overlay::new(&h, s, rotation, R, 0.0)
            .mesh(&transform())
            .vertices
            .is_empty());
    }
    #[test]
    fn horizon_clipping_retains_only_the_visible_patch() {
        let points = [
            (Vector3::new(-1.0, 0.0, -1.0), -1.0),
            (Vector3::new(1.0, 0.0, -1.0), -1.0),
            (Vector3::new(1.0, 1.0, 1.0), 1.0),
            (Vector3::new(-1.0, 1.0, 1.0), 1.0),
        ];
        let clipped = front_polygon(&points);
        assert_eq!(clipped.len(), 4);
        assert!(clipped.iter().all(|p| p.z >= 0.0));
    }
    #[test]
    fn toggle_starts_off_and_exposes_sensor_controls_when_enabled() {
        let mut s = Settings::default();
        assert!(!s.enabled);
        let ctx = egui::Context::default();
        let output = ctx.run_ui(Default::default(), |ui| s.ui(ui, true));
        fn text(shape: &Shape, needle: &str) -> bool {
            match shape {
                Shape::Text(t) => t.galley.job.text == needle,
                Shape::Vec(v) => v.iter().any(|s| text(s, needle)),
                _ => false,
            }
        }
        assert!(output
            .shapes
            .iter()
            .any(|s| text(&s.shape, "Pushbroom sweeps")));
        s.enabled = true;
        let output = ctx.run_ui(Default::default(), |ui| s.ui(ui, true));
        assert!(output.shapes.iter().any(|s| text(&s.shape, "Swath width")));
        assert!(output.shapes.iter().any(|s| text(&s.shape, "Recent trail")));
    }
    #[test]
    fn displayed_positions_follow_earth_rotation_and_reset_with_time_or_toggle() {
        let mut config = crate::config::ConstellationConfig::new(0);
        config.sats_per_plane = 1;
        config.num_planes = 1;
        let orbit = config.constellation(R, leo_sim::geometry::EARTH_MU, 0.0, R);
        let positions = orbit.satellite_positions(0.0);
        let data = vec![(orbit, positions, 0, 0, 0, "test".into())];
        let body = crate::math::rotation_from_drag(0.7, 0.0);
        let mut history = History::default();
        let settings = Settings {
            enabled: true,
            ..Default::default()
        };
        history.update(0.0, &body, R, &data, settings);
        let actual = history.tracks.values().next().unwrap().head.centre;
        let sat = &data[0].1[0];
        let expected = (body.transpose() * Vector3::new(sat.x, sat.y, sat.z)).normalize();
        assert!((Vector3::from(actual) - expected).norm() < 1e-12);
        history.update(2.0, &body, R, &data, settings);
        assert_eq!(history.tracks.values().next().unwrap().samples.len(), 2);
        history.update(-1.0, &body, R, &data, settings);
        assert_eq!(history.tracks.values().next().unwrap().samples.len(), 1);
        history.update(200.0, &body, R, &data, settings);
        assert_eq!(history.tracks.values().next().unwrap().samples.len(), 1);
        history.update(201.0, &body, R, &data, Settings::default());
        assert!(history.tracks.is_empty());
        history.update(202.0, &body, R, &data, settings);
        history.update(203.0, &body, R, &[], settings);
        assert!(history.tracks.is_empty());
    }

    #[test]
    fn drawing_detail_increases_with_zoom_and_preserves_the_current_scanline() {
        let samples: Vec<_> = (0..60).map(|i| sample(i as f64)).collect();
        let overview = drawing_samples(&samples, 400.0, 60.0);
        let closeup = drawing_samples(&samples, 1_000_000.0, 60.0);
        assert!(overview.len() < closeup.len());
        for indices in [&overview, &closeup] {
            assert_eq!(indices.first(), Some(&0));
            assert_eq!(indices.last(), Some(&59));
            assert!(indices
                .windows(2)
                .all(|pair| samples[pair[1]].time - samples[pair[0]].time <= 7.5));
        }
    }
    #[test]
    fn thousand_satellite_overlay_is_bounded_and_valid() {
        let mut h = History::default();
        for id in 0..1000 {
            for t in 0..60 {
                let s = Sample {
                    time: t as f64,
                    centre: ground_position(
                        (id % 120) as f64 - 60.0,
                        id as f64 * 0.36 - 180.0 + t as f64 * 0.05,
                        1.0,
                        0.0,
                    ),
                    radius_ratio: 1.08,
                };
                h.record((id, 0, 0), s, Color32::GREEN, 60.0, R);
            }
        }
        h.time = Some(59.0);
        let started = std::time::Instant::now();
        let mesh = Overlay::new(
            &h,
            Settings {
                enabled: true,
                ..Default::default()
            },
            Matrix3::identity(),
            R,
            0.0,
        )
        .mesh(&transform());
        assert!(mesh.is_valid());
        assert!(!mesh.vertices.is_empty());
        eprintln!(
            "1,000 sweep overlay: {:?}, {} vertices",
            started.elapsed(),
            mesh.vertices.len()
        );
    }
}
