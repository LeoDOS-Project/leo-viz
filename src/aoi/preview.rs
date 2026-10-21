//! Cached spherical boundaries and one batched plot item for thousands of AOIs.
use crate::globe_overlay::{front_segment, line_quad, project_surface};
use eframe::egui::{self, Color32, Mesh, Rect, Shape};
use egui_plot::{PlotBounds, PlotGeometry, PlotItem, PlotItemBase, PlotPoint, PlotTransform};
use nalgebra::{Matrix3, Vector3};
use serde::Deserialize;
use std::{
    ops::RangeInclusive,
    sync::{Arc, Mutex},
};

#[derive(Deserialize)]
struct Document {
    schema_version: u32,
    config: Config,
    aois: Vec<Record>,
}
#[derive(Deserialize)]
struct Config {
    seed: u32,
    count: usize,
}
#[derive(Deserialize)]
struct Record {
    #[serde(default)]
    id: String,
    anchor_lon: f64,
    anchor_lat: f64,
    area_km2: f64,
    geometry: Geometry,
    #[serde(default)]
    source_kind: String,
    #[serde(default)]
    path_truncated: bool,
    #[serde(default)]
    path_length_km: Option<f64>,
}
#[derive(Deserialize)]
#[serde(tag = "type", content = "coordinates")]
enum Geometry {
    Polygon(Vec<Vec<[f64; 2]>>),
    MultiPolygon(Vec<Vec<Vec<[f64; 2]>>>),
}

struct Detail {
    levels: Vec<(f64, Vec<Vector3<f64>>)>,
}

impl Detail {
    fn new(points: Vec<Vector3<f64>>) -> Self {
        let mut levels = vec![];
        for tolerance in [1e-6, 1e-5, 1e-4, 1e-3] {
            let simplified = simplify(&points, tolerance);
            let previous = levels
                .last()
                .map_or(points.len(), |(_, p): &(f64, Vec<Vector3<f64>>)| p.len());
            if simplified.len() >= 4 && simplified.len() < previous {
                levels.push((tolerance, simplified));
            }
        }
        levels.insert(0, (0.0, points));
        Self { levels }
    }
    fn at_scale(&self, scale: f64) -> &[Vector3<f64>] {
        &self
            .levels
            .iter()
            .rev()
            .find(|(error, _)| error * scale <= 0.7)
            .unwrap()
            .1
    }
}

fn simplify(points: &[Vector3<f64>], tolerance: f64) -> Vec<Vector3<f64>> {
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    let mut stack = vec![(0, points.len() - 1)];
    while let Some((first, last)) = stack.pop() {
        if last <= first + 1 {
            continue;
        }
        let a = points[first];
        let d = points[last] - a;
        let length = d.norm_squared();
        let mut furthest = first;
        let mut error = tolerance * tolerance;
        for (i, p) in points.iter().enumerate().take(last).skip(first + 1) {
            let t = if length > 0.0 {
                ((p - a).dot(&d) / length).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let distance = (p - a - d * t).norm_squared();
            if distance > error {
                error = distance;
                furthest = i;
            }
        }
        if furthest != first {
            keep[furthest] = true;
            stack.push((first, furthest));
            stack.push((furthest, last));
        }
    }
    points
        .iter()
        .zip(keep)
        .filter_map(|(p, keep)| keep.then_some(*p))
        .collect()
}

struct Boundary {
    anchor: Vector3<f64>,
    radius: f64,
    rings: Vec<Detail>,
    polygons: Vec<Vec<Vec<[f64; 2]>>>,
    color: [u8; 3],
}

pub(crate) struct Preview {
    pub aois: Vec<crate::jobs::model::AoiRef>,
    pub count: usize,
    pub mean_area: f64,
    pub min_area: f64,
    pub max_area: f64,
    pub seed: u32,
    pub truncated_paths: usize,
    pub mean_path_length: Option<f64>,
    boundaries: Vec<Boundary>,
    spatial: super::spatial::Index,
    mesh_cache: Mutex<Vec<(Vec<f64>, Arc<Mesh>)>>,
}

fn unit([lon, lat]: [f64; 2]) -> Vector3<f64> {
    let (lon, lat) = (-lon.to_radians(), lat.to_radians());
    Vector3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin())
}

fn ring(points: Vec<[f64; 2]>) -> Result<Vec<Vector3<f64>>, String> {
    if points.len() < 4
        || points.first() != points.last()
        || points.iter().any(|p| {
            !p[0].is_finite() || !p[1].is_finite() || p[0].abs() > 180.0 || p[1].abs() > 90.0
        })
    {
        return Err("Invalid AOI boundary returned by generator".into());
    }
    let mut result = Vec::new();
    for edge in points.windows(2) {
        let a = edge[0];
        let b = edge[1];
        // GeoJSON is already split at the dateline. Interpolate lon/lat along
        // parallels/meridians, not a great circle across a rectangular AOI.
        let steps = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()) / 0.5)
            .ceil()
            .max(1.0) as usize;
        for i in 0..steps {
            let t = i as f64 / steps as f64;
            result.push(unit([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]));
        }
    }
    result.push(result[0]);
    Ok(result)
}

impl Preview {
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let doc: Document =
            serde_json::from_slice(bytes).map_err(|e| format!("Invalid AOI response: {e}"))?;
        if doc.schema_version != 1
            || doc.aois.is_empty()
            || doc.aois.len() > 50_000
            || doc.aois.len() != doc.config.count
        {
            return Err("Invalid AOI response version or count".into());
        }
        let mut preview = Self {
            aois: Vec::with_capacity(doc.aois.len()),
            count: doc.aois.len(),
            mean_area: 0.0,
            min_area: f64::INFINITY,
            max_area: 0.0,
            seed: doc.config.seed,
            truncated_paths: 0,
            mean_path_length: None,
            boundaries: Vec::with_capacity(doc.aois.len()),
            spatial: super::spatial::Index::default(),
            mesh_cache: Mutex::new(Vec::new()),
        };
        const COLORS: [[u8; 3]; 6] = [
            [61, 222, 230],
            [255, 190, 86],
            [159, 173, 255],
            [130, 231, 158],
            [244, 142, 189],
            [235, 226, 136],
        ];
        let mut identifiers = std::collections::HashSet::new();
        for (index, aoi) in doc.aois.into_iter().enumerate() {
            if !aoi.area_km2.is_finite()
                || aoi.area_km2 <= 0.0
                || !aoi.anchor_lon.is_finite()
                || !aoi.anchor_lat.is_finite()
                || aoi.anchor_lon.abs() > 180.0
                || aoi.anchor_lat.abs() > 90.0
            {
                return Err("Invalid AOI area or anchor".into());
            }
            let anchor = unit([aoi.anchor_lon, aoi.anchor_lat]);
            let polygons = match aoi.geometry {
                Geometry::Polygon(p) => vec![p],
                Geometry::MultiPolygon(p) => p,
            };
            let rings = polygons
                .iter()
                .flatten()
                .cloned()
                .map(ring)
                .collect::<Result<Vec<_>, _>>()?;
            if rings.is_empty() {
                return Err("AOI has no boundary".into());
            }
            let radius = rings
                .iter()
                .flatten()
                .map(|p| (p - anchor).norm())
                .fold(0.0, f64::max);
            let metadata = crate::jobs::model::AoiRef {
                source: if aoi.source_kind.is_empty() {
                    "gdp".into()
                } else {
                    aoi.source_kind.clone()
                },
                id: if aoi.id.is_empty() {
                    format!("aoi-{index:08}")
                } else {
                    aoi.id
                },
                area_km2: aoi.area_km2,
                longitude: aoi.anchor_lon,
                latitude: aoi.anchor_lat,
                geometry_hash: crate::jobs::model::hash(&serde_json::to_vec(&polygons).unwrap()),
            };
            if !identifiers.insert(metadata.key()) {
                return Err("Duplicate AOI identifier".into());
            }
            preview.aois.push(metadata);
            preview.boundaries.push(Boundary {
                anchor,
                radius,
                rings: rings.into_iter().map(Detail::new).collect(),
                polygons,
                color: match aoi.source_kind.as_str() {
                    "forest" => [130, 231, 158],
                    "river" => [61, 222, 230],
                    "coast" => [255, 190, 86],
                    _ => COLORS[index % COLORS.len()],
                },
            });
            preview.truncated_paths += usize::from(aoi.path_truncated);
            if let Some(length) = aoi.path_length_km {
                if !length.is_finite() || length <= 0.0 {
                    return Err("Invalid AOI path length".into());
                }
                *preview.mean_path_length.get_or_insert(0.0) += length / preview.count as f64;
            }
            preview.mean_area += aoi.area_km2 / preview.count as f64;
            preview.min_area = preview.min_area.min(aoi.area_km2);
            preview.max_area = preview.max_area.max(aoi.area_km2);
        }
        preview.spatial = super::spatial::Index::build(
            &preview
                .boundaries
                .iter()
                .map(|boundary| super::spatial::Sphere {
                    center: boundary.anchor,
                    radius: boundary.radius + 1e-5,
                })
                .collect::<Vec<_>>(),
        );
        Ok(preview)
    }
}

#[derive(Clone)]
pub(crate) struct Hit {
    pub preview: Arc<Preview>,
    pub index: usize,
}

fn contains_ring(points: &[[f64; 2]], x: f64, y: f64) -> bool {
    let mut inside = false;
    for edge in points.windows(2) {
        let ([ax, ay], [bx, by]) = (edge[0], edge[1]);
        if (ay > y) != (by > y) && x < ax + (y - ay) * (bx - ax) / (by - ay) {
            inside = !inside;
        }
    }
    inside
}
fn contains(polygons: &[Vec<Vec<[f64; 2]>>], lon: f64, lat: f64) -> bool {
    polygons.iter().any(|polygon| {
        polygon.first().is_some_and(|outer| {
            contains_ring(outer, lon, lat)
                && !polygon[1..]
                    .iter()
                    .any(|hole| contains_ring(hole, lon, lat))
        })
    })
}

pub(crate) struct Overlay {
    base: PlotItemBase,
    preview: Arc<Preview>,
    rotation: Matrix3<f64>,
    radius: f64,
    polar_scale: f64,
    opacity: f32,
    selected: Option<usize>,
}

impl Overlay {
    pub fn new(
        preview: Arc<Preview>,
        rotation: Matrix3<f64>,
        radius: f64,
        flattening: f64,
        opacity: f32,
    ) -> Self {
        Self {
            base: PlotItemBase::new("Generated AOIs".into()),
            preview,
            rotation,
            radius,
            polar_scale: 1.0 - flattening,
            opacity,
            selected: None,
        }
    }

    pub fn with_selection(mut self, selected: Option<usize>) -> Self {
        self.selected = selected;
        self
    }

    /// Exact polygon picking (including holes), with a small screen-space
    /// fallback for AOIs represented by dots. Uses the existing spatial index.
    pub fn pick(&self, pos: egui::Pos2, transform: &PlotTransform) -> Vec<Hit> {
        let point = transform.value_from_position(pos);
        let Some((lat, lon)) = crate::globe_overlay::pick_surface(
            point.x,
            point.y,
            &self.rotation,
            self.radius,
            self.polar_scale,
        ) else {
            return Vec::new();
        };
        let point = unit([lon, lat]);
        let scale = transform
            .dpos_dvalue_x()
            .abs()
            .max(transform.dpos_dvalue_y().abs())
            * self.radius;
        let candidates = self.preview.spatial.query(|sphere| {
            let (p, normal) = self.project(sphere.center);
            let screen = transform.position_from_point(&PlotPoint::new(p.x, p.y));
            normal >= -sphere.radius / self.polar_scale
                && screen.distance(pos) as f64 <= sphere.radius * scale + 6.0
        });
        let mut found = Vec::new();
        for index in candidates {
            let boundary = &self.preview.boundaries[index];
            let (p, normal) = self.project(boundary.anchor);
            let screen = transform.position_from_point(&PlotPoint::new(p.x, p.y));
            let exact = (point - boundary.anchor).norm() <= boundary.radius + 1e-5
                && contains(&boundary.polygons, lon, lat);
            let dot = boundary.radius * scale < 3.0 && normal >= 0.0 && screen.distance(pos) <= 6.0;
            if exact || dot {
                found.push(Hit {
                    preview: self.preview.clone(),
                    index,
                });
            }
        }
        found.sort_by(|a, b| {
            self.preview.aois[a.index]
                .area_km2
                .total_cmp(&self.preview.aois[b.index].area_km2)
        });
        found
    }

    fn selection_mesh(&self, transform: &PlotTransform) -> Mesh {
        let mut mesh = Mesh::default();
        let Some(boundary) = self.selected.and_then(|i| self.preview.boundaries.get(i)) else {
            return mesh;
        };
        let color = Color32::YELLOW;
        for detail in &boundary.rings {
            let ring = &detail.levels[0].1;
            for edge in ring.windows(2) {
                if let Some((a, b)) = front_segment(self.project(edge[0]), self.project(edge[1])) {
                    let a = transform.position_from_point(&PlotPoint::new(a.x, a.y));
                    let b = transform.position_from_point(&PlotPoint::new(b.x, b.y));
                    if transform
                        .frame()
                        .intersects(Rect::from_two_pos(a, b).expand(2.0))
                    {
                        line_quad(&mut mesh, a, b, color, 2.5);
                    }
                }
            }
        }
        let (anchor, normal) = self.project(boundary.anchor);
        let anchor = transform.position_from_point(&PlotPoint::new(anchor.x, anchor.y));
        if normal >= 0.0 && transform.frame().contains(anchor) {
            mesh.add_colored_rect(Rect::from_center_size(anchor, egui::vec2(5.0, 5.0)), color);
        }
        mesh
    }

    fn project(&self, p: Vector3<f64>) -> (Vector3<f64>, f64) {
        project_surface(p, &self.rotation, self.radius, self.polar_scale)
    }

    fn cached_mesh(&self, transform: &PlotTransform) -> Arc<Mesh> {
        let mut key = self.rotation.as_slice().to_vec();
        let frame = transform.frame();
        key.extend([
            frame.min.x as f64,
            frame.min.y as f64,
            frame.max.x as f64,
            frame.max.y as f64,
        ]);
        key.extend(transform.bounds().min());
        key.extend(transform.bounds().max());
        key.extend([self.radius, self.polar_scale, self.opacity as f64]);
        let scale = transform
            .dpos_dvalue_x()
            .abs()
            .max(transform.dpos_dvalue_y().abs())
            * self.radius;
        if let Some((_, mesh)) = self
            .preview
            .mesh_cache
            .lock()
            .unwrap()
            .iter()
            .find(|(k, _)| {
                k[9..] == key[9..]
                    && k[..9]
                        .iter()
                        .zip(&key[..9])
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>()
                        .sqrt()
                        * scale
                        <= 0.25
            })
        {
            return Arc::clone(mesh);
        }
        let mesh = Arc::new(self.mesh(transform));
        let mut cache = self.preview.mesh_cache.lock().unwrap();
        if cache.len() >= 4 {
            cache.remove(0);
        }
        cache.push((key, Arc::clone(&mesh)));
        mesh
    }

    fn mesh(&self, transform: &PlotTransform) -> Mesh {
        let mut mesh = Mesh::default();
        let scale = transform
            .dpos_dvalue_x()
            .abs()
            .max(transform.dpos_dvalue_y().abs())
            * self.radius;
        // Precompute the affine plot mapping once, instead of dividing for
        // every projected vertex. Keep f64 until the final screen position.
        let origin = transform.bounds().min();
        let corner = transform.position_from_point(&PlotPoint::new(origin[0], origin[1]));
        let sx = transform.dpos_dvalue_x();
        let sy = transform.dpos_dvalue_y();
        let screen = |p: Vector3<f64>| {
            egui::pos2(
                (corner.x as f64 + (p.x - origin[0]) * sx) as f32,
                (corner.y as f64 + (p.y - origin[1]) * sy) as f32,
            )
        };
        let frame = transform.frame();
        let candidates = self.preview.spatial.query(|sphere| {
            let (center, normal) = self.project(sphere.center);
            normal >= -sphere.radius / self.polar_scale
                && frame
                    .expand((sphere.radius * scale) as f32 + 3.0)
                    .contains(screen(center))
        });
        // Many overlapping subpixel AOIs contribute no useful extra detail.
        // One representative per 3px cell bounds overdraw without changing data.
        let columns = (frame.width() / 3.0).ceil().max(1.0) as usize;
        let rows = (frame.height() / 3.0).ceil().max(1.0) as usize;
        let mut dots = vec![false; columns * rows];
        for index in candidates {
            let boundary = &self.preview.boundaries[index];
            let color = Color32::from_rgba_unmultiplied(
                boundary.color[0],
                boundary.color[1],
                boundary.color[2],
                (self.opacity * 255.0).round() as u8,
            );
            let (anchor, normal) = self.project(boundary.anchor);
            let pixel_radius = boundary.radius * scale;
            // Conservative bounds: retain any AOI that could cross the horizon
            // or viewport, including continental and dateline-crossing shapes.
            if normal < -boundary.radius / self.polar_scale
                || !transform
                    .frame()
                    .expand(pixel_radius as f32 + 2.0)
                    .contains(screen(anchor))
            {
                continue;
            }
            if pixel_radius < 3.0 {
                if normal >= 0.0 && transform.frame().contains(screen(anchor)) {
                    let point = screen(anchor);
                    let x = ((point.x - frame.left()) / 3.0).floor().max(0.0) as usize;
                    let y = ((point.y - frame.top()) / 3.0).floor().max(0.0) as usize;
                    let cell = y.min(rows - 1) * columns + x.min(columns - 1);
                    if dots[cell] {
                        continue;
                    }
                    dots[cell] = true;
                    mesh.add_colored_rect(
                        Rect::from_center_size(point, egui::vec2(2.5, 2.5)),
                        color,
                    );
                }
                continue;
            }
            for detail in &boundary.rings {
                let ring = detail.at_scale(scale);
                let mut previous = self.project(ring[0]);
                for point in &ring[1..] {
                    let current = self.project(*point);
                    if let Some((a, b)) = front_segment(previous, current) {
                        let (a, b) = (screen(a), screen(b));
                        if transform
                            .frame()
                            .intersects(Rect::from_two_pos(a, b).expand(1.0))
                        {
                            line_quad(&mut mesh, a, b, color, 1.2);
                        }
                    }
                    previous = current;
                }
            }
        }
        mesh
    }
    #[cfg(test)]
    fn reference_mesh(&self, transform: &PlotTransform) -> Mesh {
        let mut mesh = Mesh::default();
        let scale = transform
            .dpos_dvalue_x()
            .abs()
            .max(transform.dpos_dvalue_y().abs())
            * self.radius;
        let screen = |p: Vector3<f64>| transform.position_from_point(&PlotPoint::new(p.x, p.y));
        for boundary in &self.preview.boundaries {
            let color = Color32::from_rgba_unmultiplied(
                boundary.color[0],
                boundary.color[1],
                boundary.color[2],
                (self.opacity * 255.0).round() as u8,
            );
            let (anchor, normal) = self.project(boundary.anchor);
            let pixel_radius = boundary.radius * scale;
            // Conservative bounds: retain any AOI that could cross the horizon
            // or viewport, including continental and dateline-crossing shapes.
            if normal < -boundary.radius / self.polar_scale
                || !transform
                    .frame()
                    .expand(pixel_radius as f32 + 2.0)
                    .contains(screen(anchor))
            {
                continue;
            }
            if pixel_radius < 1.5 {
                if normal >= 0.0 && transform.frame().contains(screen(anchor)) {
                    mesh.add_colored_rect(
                        Rect::from_center_size(screen(anchor), egui::vec2(2.5, 2.5)),
                        color,
                    );
                }
                continue;
            }
            for detail in &boundary.rings {
                let ring = &detail.levels[0].1;
                let mut previous = self.project(ring[0]);
                for point in &ring[1..] {
                    let current = self.project(*point);
                    if let Some((a, b)) = front_segment(previous, current) {
                        let (a, b) = (screen(a), screen(b));
                        if transform
                            .frame()
                            .intersects(Rect::from_two_pos(a, b).expand(1.0))
                        {
                            line_quad(&mut mesh, a, b, color, 1.2);
                        }
                    }
                    previous = current;
                }
            }
        }
        mesh
    }
}

impl PlotItem for Overlay {
    fn shapes(&self, _ui: &egui::Ui, transform: &PlotTransform, shapes: &mut Vec<Shape>) {
        shapes.push(Shape::Mesh(self.cached_mesh(transform)));
        if self.selected.is_some() {
            shapes.push(Shape::Mesh(Arc::new(self.selection_mesh(transform))));
        }
    }
    fn initialize(&mut self, _: RangeInclusive<f64>) {}
    fn color(&self) -> Color32 {
        Color32::LIGHT_BLUE
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
    use egui::Pos2;
    #[test]
    fn picks_interiors_holes_overlaps_and_dateline_without_backside_hits() {
        let geometry = serde_json::json!({"type":"Polygon", "coordinates":[
            [[-94.,-4.],[-86.,-4.],[-86.,4.],[-94.,4.],[-94.,-4.]],
            [[-91.,-1.],[-89.,-1.],[-89.,1.],[-91.,1.],[-91.,-1.]]
        ]});
        let doc = serde_json::json!({"schema_version":1,"config":{"seed":42,"count":2},"aois":[
            {"id":"large","area_km2":1000,"anchor_lon":-90,"anchor_lat":0,"geometry":geometry.clone()},
            {"id":"small","area_km2":100,"anchor_lon":-90,"anchor_lat":0,"geometry":geometry}
        ]});
        let preview = Arc::new(Preview::from_json(&serde_json::to_vec(&doc).unwrap()).unwrap());
        let transform = PlotTransform::new(
            Rect::from_min_size(Pos2::ZERO, egui::vec2(800., 800.)),
            PlotBounds::new_symmetrical(7000.),
            [false, false],
        );
        let overlay = Overlay::new(preview.clone(), Matrix3::identity(), 6371., 0., 0.8);
        let screen = |lon, lat| {
            let (p, _) = overlay.project(unit([lon, lat]));
            transform.position_from_point(&PlotPoint::new(p.x, p.y))
        };
        assert!(
            overlay.pick(screen(-90., 0.), &transform).is_empty(),
            "A hole is not an AOI"
        );
        let hits = overlay.pick(screen(-92., 0.), &transform);
        assert_eq!(hits.iter().map(|h| h.index).collect::<Vec<_>>(), vec![1, 0]);
        let turn = crate::math::rotation_from_drag(std::f64::consts::PI, 0.);
        assert!(Overlay::new(preview.clone(), turn, 6371., 0., 0.8)
            .pick(screen(-92., 0.), &transform)
            .is_empty());
        assert!(overlay
            .with_selection(Some(0))
            .selection_mesh(&transform)
            .is_valid());
        let seam = vec![
            vec![vec![
                [179., -2.],
                [180., -2.],
                [180., 2.],
                [179., 2.],
                [179., -2.],
            ]],
            vec![vec![
                [-180., -2.],
                [-179., -2.],
                [-179., 2.],
                [-180., 2.],
                [-180., -2.],
            ]],
        ];
        assert!(contains(&seam, 179.5, 0.));
        assert!(contains(&seam, -179.5, 0.));
        assert!(!contains(&seam, 0., 0.));
    }

    #[test]
    fn clips_crossing_but_omits_hidden_segments() {
        let a = (Vector3::new(1.0, 0.0, -1.0), -1.0);
        let b = (Vector3::new(1.0, 0.0, 1.0), 1.0);
        assert!(front_segment(a, a).is_none());
        assert_eq!(front_segment(a, b).unwrap().0.z, 0.0);
        assert_eq!(front_segment(b, a).unwrap().1.z, 0.0);
    }
    #[test]
    fn dateline_parts_remain_separate_and_follow_parallels() {
        let points = ring(vec![
            [179.0, 30.0],
            [180.0, 30.0],
            [180.0, 31.0],
            [179.0, 30.0],
        ])
        .unwrap();
        assert!(points.iter().all(|p| p.x < -0.8));
        assert!((points[1].y - 0.5).abs() < 1e-12);
        assert!((points[1].norm() - 1.0).abs() < 1e-12);
    }
    #[test]
    fn invalid_response_and_open_rings_are_rejected() {
        assert!(Preview::from_json(
            br#"{"schema_version":2,"config":{"count":0,"seed":1},"aois":[]}"#
        )
        .is_err());
        assert!(ring(vec![[0.0, 0.0]; 3]).is_err());
        assert!(ring(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]).is_err());
    }
    #[test]
    fn front_and_back_globes_render_different_geometry() {
        let preview = Preview::from_json(br#"{"schema_version":1,"config":{"seed":42,"count":1},"aois":[{"area_km2":100,"anchor_lon":-90,"anchor_lat":0,"geometry":{"type":"Polygon","coordinates":[[[-91,-1],[-89,-1],[-89,1],[-91,1],[-91,-1]]]}}]}"#).unwrap();
        let transform = PlotTransform::new(
            Rect::from_min_size(Pos2::ZERO, egui::vec2(800.0, 800.0)),
            PlotBounds::new_symmetrical(7000.0),
            [false, false],
        );
        let preview = Arc::new(preview);
        let front =
            Overlay::new(preview.clone(), Matrix3::identity(), 6371.0, 0.0, 0.8).mesh(&transform);
        assert!(!front.vertices.is_empty());
        assert!(front.is_valid());
        let turn = crate::math::rotation_from_drag(std::f64::consts::PI, 0.0);
        let back = Overlay::new(preview, turn, 6371.0, 0.0, 0.8).mesh(&transform);
        assert!(back.vertices.is_empty());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "requires the leo-aoi Python environment and ingested GDP example"]
    fn real_gdp_generation_and_batched_render() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/leo-aoi");
        let request = serde_json::json!({"schema_version":1, "demand_path":root.join("data/gdp-demand.npz"),
            "mean_area_km2":1000, "size_ratio":100,
            "generation":{"count":10000,"seed":42,"hotspot_count":256,"hotspot_spread_km":10,"hotspot_fraction":0.8}});
        let start = std::time::Instant::now();
        let preview =
            super::super::worker::run(root.join(".venv/bin/python").to_str().unwrap(), &request)
                .unwrap();
        assert_eq!(preview.count, 10000);
        assert!((preview.mean_area - 1000.0).abs() < 50.0);
        let generation_time = start.elapsed();
        let transform = PlotTransform::new(
            Rect::from_min_size(Pos2::ZERO, egui::vec2(1000.0, 1000.0)),
            PlotBounds::new_symmetrical(1500.0),
            [false, false],
        );
        let preview = Arc::new(preview);
        let start = std::time::Instant::now();
        let rotation = crate::math::lat_lon_to_matrix(50.0_f64.to_radians(), 10.0_f64.to_radians());
        let overlay = Overlay::new(preview, rotation, 6371.0, 0.0, 0.8);
        let mesh = overlay.cached_mesh(&transform);
        assert!(mesh.is_valid() && !mesh.vertices.is_empty());
        assert!(mesh
            .vertices
            .iter()
            .all(|v| v.pos.x.is_finite() && v.pos.y.is_finite()));
        eprintln!(
            "10,000 GDP AOIs: generation {:?}; zoomed render {:?}; {} mesh vertices",
            generation_time,
            start.elapsed(),
            mesh.vertices.len()
        );
        let cached_start = std::time::Instant::now();
        let cached = overlay.cached_mesh(&transform);
        assert!(Arc::ptr_eq(&cached, &mesh));
        eprintln!("unchanged camera: {:?}", cached_start.elapsed());
    }
    #[test]
    #[ignore = "render benchmark using the bundled 50k AOIs"]
    fn benchmark_default_jobs_render() {
        let preview = Arc::new(super::super::presets::load_default_jobs().unwrap());
        for zoom in [7000.0, 1000.0, 250.0] {
            let transform = PlotTransform::new(
                Rect::from_min_size(Pos2::ZERO, egui::vec2(1000.0, 900.0)),
                PlotBounds::new_symmetrical(zoom),
                [false, false],
            );
            for reference in [true, false] {
                let start = std::time::Instant::now();
                let mut vertices = 0;
                for frame in 0..120 {
                    let rotation = crate::math::lat_lon_to_matrix(
                        50.0_f64.to_radians(),
                        (10.0 + frame as f64 * 0.05).to_radians(),
                    );
                    let overlay = Overlay::new(preview.clone(), rotation, 6371.0, 0.0, 0.8);
                    vertices += if reference {
                        overlay.reference_mesh(&transform)
                    } else {
                        overlay.mesh(&transform)
                    }
                    .vertices
                    .len();
                }
                eprintln!(
                    "50k AOIs zoom={zoom} reference={reference}: {:.3} ms/frame, {} vertices/frame",
                    start.elapsed().as_secs_f64() * 1000.0 / 120.0,
                    vertices / 120
                );
            }
        }
    }

    #[test]
    fn spatial_culling_keeps_visible_boundaries_and_reduces_zoomed_candidates() {
        let preview = Arc::new(super::super::presets::load_default_jobs().unwrap());
        for (lat, lon) in [(50.0_f64, 10.0_f64), (0.0, 179.0), (-70.0, -90.0)] {
            let rotation = crate::math::lat_lon_to_matrix(lat.to_radians(), lon.to_radians());
            let overlay = Overlay::new(preview.clone(), rotation, 6371.0, 0.004, 0.8);
            let transform = PlotTransform::new(
                Rect::from_min_size(Pos2::ZERO, egui::vec2(1000.0, 900.0)),
                PlotBounds::new_symmetrical(250.0),
                [false, false],
            );
            let scale = transform
                .dpos_dvalue_x()
                .abs()
                .max(transform.dpos_dvalue_y().abs())
                * overlay.radius;
            let candidates = preview.spatial.query(|sphere| {
                let (p, normal) = overlay.project(sphere.center);
                normal >= -sphere.radius / overlay.polar_scale
                    && transform
                        .frame()
                        .expand((sphere.radius * scale) as f32 + 3.0)
                        .contains(transform.position_from_point(&PlotPoint::new(p.x, p.y)))
            });
            assert!(candidates.len() < preview.count / 4);
            let set: std::collections::HashSet<_> = candidates.into_iter().collect();
            for (index, boundary) in preview.boundaries.iter().enumerate() {
                let visible = boundary
                    .rings
                    .iter()
                    .flat_map(|r| r.levels[0].1.iter())
                    .any(|p| {
                        let (p, normal) = overlay.project(*p);
                        normal >= 0.0
                            && transform
                                .frame()
                                .contains(transform.position_from_point(&PlotPoint::new(p.x, p.y)))
                    });
                if visible {
                    assert!(set.contains(&index), "Culled a visible AOI at {lat},{lon}");
                }
            }
        }
    }

    #[test]
    fn detail_retains_closed_rings_and_restores_full_geometry_on_zoom() {
        let points: Vec<_> = (0..=1000)
            .map(|i| {
                let angle = (i % 1000) as f64 * std::f64::consts::TAU / 1000.0;
                unit([10.0 + angle.cos(), 50.0 + angle.sin()])
            })
            .collect();
        let detail = Detail::new(points.clone());
        assert!(detail.at_scale(500.0).len() < points.len() / 10);
        assert_eq!(detail.at_scale(1e9), points.as_slice());
        for (_, simplified) in &detail.levels {
            assert_eq!(simplified.first(), simplified.last());
        }
    }
}
