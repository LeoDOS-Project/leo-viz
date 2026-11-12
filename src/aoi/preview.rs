//! Cached spherical boundaries and one batched plot item for thousands of AOIs.
use crate::globe_overlay::{front_segment, line_quad, project_surface};
use eframe::egui::{self, Color32, Mesh, Rect, Shape};
use egui_plot::{PlotBounds, PlotGeometry, PlotItem, PlotItemBase, PlotPoint, PlotTransform};
use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};
use std::{
    ops::RangeInclusive,
    sync::{Arc, Mutex, OnceLock},
};

#[derive(Deserialize)]
pub(super) struct Document {
    pub schema_version: u32,
    pub config: Config,
    pub aois: Vec<Record>,
}
#[derive(Deserialize, Serialize)]
pub(super) struct Config {
    pub seed: u32,
    pub count: usize,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub source_seeds: std::collections::BTreeMap<String, u32>,
}
#[derive(Deserialize, Serialize)]
pub(super) struct Record {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reduction: Option<super::compact::Reduction>,
    #[serde(default)]
    pub id: String,
    pub anchor_lon: f64,
    pub anchor_lat: f64,
    pub area_km2: f64,
    pub geometry: Geometry,
    #[serde(default)]
    pub source_kind: String,
    #[serde(default)]
    pub path_truncated: bool,
    #[serde(default)]
    pub path_length_km: Option<f64>,
}
#[derive(Deserialize, Serialize)]
#[serde(tag = "type", content = "coordinates")]
pub(super) enum Geometry {
    Polygon(Vec<Vec<[f64; 2]>>),
    MultiPolygon(Vec<Vec<Vec<[f64; 2]>>>),
}

struct Detail {
    levels: Vec<(f64, Vec<Vector3<f64>>)>,
}

impl Detail {
    fn new(points: Vec<Vector3<f64>>) -> Self {
        Self::with_unit(points, 1.0)
    }

    fn with_unit(points: Vec<Vector3<f64>>, unit: f64) -> Self {
        let mut levels = vec![];
        for tolerance in [1e-6, 1e-5, 1e-4, 1e-3] {
            let tolerance = tolerance * unit;
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
    fill: super::fill::Fill,
    path_truncated: bool,
    path_length_km: Option<f64>,
    reduction: Option<super::compact::Reduction>,
}

/// Expensive per-AOI render preparation can be done independently on workers.
pub(super) struct PreparedRecord {
    aoi: crate::jobs::model::AoiRef,
    boundary: Boundary,
}

pub(crate) struct Preview {
    pub aois: Vec<crate::jobs::model::AoiRef>,
    pub count: usize,
    pub edge_count: usize,
    pub max_edges: usize,
    pub original_edges: usize,
    pub removed_holes: usize,
    pub removed_parts: usize,
    pub max_area_change: f64,
    pub mean_area: f64,
    pub min_area: f64,
    pub max_area: f64,
    pub seed: u32,
    pub truncated_paths: usize,
    pub mean_path_length: Option<f64>,
    boundaries: Vec<Boundary>,
    spatial: super::spatial::Index,
    mesh_cache: Mutex<Vec<(Vec<f64>, Arc<Mesh>)>>,
    gpu_geometry: OnceLock<Arc<super::gpu::Geometry>>,
    projected_geometry: Mutex<Vec<(i32, Arc<super::gpu::Geometry>)>>,
    #[cfg(not(target_arch = "wasm32"))]
    pending_projections: Mutex<std::collections::HashSet<i32>>,
}

pub(super) fn unit([lon, lat]: [f64; 2]) -> Vector3<f64> {
    let (lon, lat) = (-lon.to_radians(), lat.to_radians());
    Vector3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin())
}

fn validate_ring(points: &[[f64; 2]]) -> Result<(), String> {
    if points.len() < 4
        || points.first() != points.last()
        || points.iter().any(|p| {
            !p[0].is_finite() || !p[1].is_finite() || p[0].abs() > 180.0 || p[1].abs() > 90.0
        })
    {
        return Err("Invalid AOI boundary returned by generator".into());
    }
    Ok(())
}

fn ring(points: Vec<[f64; 2]>) -> Result<Vec<Vector3<f64>>, String> {
    validate_ring(&points)?;
    let mut result = Vec::new();
    for edge in points.windows(2) {
        let a = edge[0];
        let b = edge[1];
        let longitude_delta = (b[0] - a[0] + 180.0).rem_euclid(360.0) - 180.0;
        // GeoJSON is already split at the dateline. Interpolate lon/lat along
        // parallels/meridians, not a great circle across a rectangular AOI.
        let steps = (longitude_delta.abs().max((b[1] - a[1]).abs()) / 0.5)
            .ceil()
            .max(1.0) as usize;
        for i in 0..steps {
            let t = i as f64 / steps as f64;
            result.push(unit([a[0] + t * longitude_delta, a[1] + t * (b[1] - a[1])]));
        }
    }
    result.push(result[0]);
    Ok(result)
}

impl Preview {
    /// Serialize original coordinates, never the simplified display outlines.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn serialize_records<S: serde::ser::SerializeSeq>(
        &self,
        seq: &mut S,
    ) -> Result<(), S::Error> {
        #[derive(Serialize)]
        struct GeometryRef<'a> {
            r#type: &'static str,
            coordinates: &'a Vec<Vec<Vec<[f64; 2]>>>,
        }
        #[derive(Serialize)]
        struct RecordRef<'a> {
            id: &'a str,
            source_kind: &'a str,
            anchor_lon: f64,
            anchor_lat: f64,
            area_km2: f64,
            geometry: GeometryRef<'a>,
            path_truncated: bool,
            path_length_km: Option<f64>,
            #[serde(skip_serializing_if = "Option::is_none")]
            reduction: &'a Option<super::compact::Reduction>,
        }
        for (a, b) in self.aois.iter().zip(&self.boundaries) {
            seq.serialize_element(&RecordRef {
                id: &a.id,
                source_kind: &a.source,
                anchor_lon: a.longitude,
                anchor_lat: a.latitude,
                area_km2: a.area_km2,
                geometry: GeometryRef {
                    r#type: "MultiPolygon",
                    coordinates: &b.polygons,
                },
                path_truncated: b.path_truncated,
                path_length_km: b.path_length_km,
                reduction: &b.reduction,
            })?;
        }
        Ok(())
    }
    pub(super) fn record(&self, index: usize) -> Record {
        let a = &self.aois[index];
        let b = &self.boundaries[index];
        Record {
            id: a.id.clone(),
            source_kind: a.source.clone(),
            anchor_lon: a.longitude,
            anchor_lat: a.latitude,
            area_km2: a.area_km2,
            geometry: Geometry::MultiPolygon(b.polygons.clone()),
            path_truncated: b.path_truncated,
            path_length_km: b.path_length_km,
            reduction: b.reduction.clone(),
        }
    }
    pub(super) fn complexity_ui(&self, ui: &mut egui::Ui) {
        ui.small(format!(
            "Boundary edges: {} · average {:.1} · maximum {} per AOI",
            self.edge_count,
            self.edge_count as f64 / self.count.max(1) as f64,
            self.max_edges
        ));
        if self.original_edges > self.edge_count {
            ui.small(format!(
                "Reduced from {} edges · largest area change {:.1}% · {} holes / {} pieces removed",
                self.original_edges,
                self.max_area_change * 100.0,
                self.removed_holes,
                self.removed_parts
            ));
        }
        let gpu = self.gpu_geometry();
        ui.small(format!(
            "GPU geometry: {} outline segments across detail levels · {} fill triangles",
            gpu.segments.len(),
            gpu.triangles.len()
        ));
    }
    pub(crate) fn capture_regions(&self) -> Vec<leo_capture::geometry::Region> {
        self.aois
            .iter()
            .zip(&self.boundaries)
            .map(|(a, b)| leo_capture::geometry::Region {
                job_key: a.key(),
                polygons: b.polygons.clone(),
            })
            .collect()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn from_areas(
        areas: &[crate::config::AreaOfInterest],
        radius: f64,
    ) -> Result<Self, String> {
        let mut preview = Self::empty(areas.len(), 0);
        for (index, area) in areas.iter().enumerate() {
            let lat = area.lat.to_radians();
            let lon = area.lon.to_radians();
            let angle = area.radius_km / radius;
            let mut points: Vec<_> = (0..64)
                .map(|i| {
                    let bearing = std::f64::consts::TAU * i as f64 / 64.0;
                    let y = (lat.sin() * angle.cos() + lat.cos() * angle.sin() * bearing.cos())
                        .clamp(-1.0, 1.0)
                        .asin();
                    let x = lon
                        + (angle.sin() * bearing.sin()).atan2(
                            lat.cos() * angle.cos() - lat.sin() * angle.sin() * bearing.cos(),
                        );
                    [
                        (x.to_degrees() + 180.0).rem_euclid(360.0) - 180.0,
                        y.to_degrees(),
                    ]
                })
                .collect();
            points.push(points[0]);
            preview.push_record(Record {
                reduction: None,
                id: format!("manual-{index}"),
                anchor_lon: area.lon,
                anchor_lat: area.lat,
                area_km2: std::f64::consts::PI * area.radius_km.powi(2),
                geometry: Geometry::Polygon(vec![points]),
                source_kind: "manual".into(),
                path_truncated: false,
                path_length_km: None,
            })?;
            preview.boundaries.last_mut().unwrap().color =
                [area.color.r(), area.color.g(), area.color.b()];
        }
        Ok(preview.finish())
    }

    pub fn from_geojson(
        geometry: serde_json::Value,
        anchor: [f64; 2],
        color: [u8; 3],
    ) -> Result<Self, String> {
        use geo::GeodesicArea;
        let geometry: Geometry = serde_json::from_value(geometry)
            .map_err(|e| format!("Invalid mapped boundary: {e}"))?;
        let polygons = match &geometry {
            Geometry::Polygon(polygon) => std::slice::from_ref(polygon),
            Geometry::MultiPolygon(polygons) => polygons.as_slice(),
        };
        let count: usize = polygons.iter().flatten().map(Vec::len).sum();
        if count > 250_000 || polygons.is_empty() || polygons.iter().any(Vec::is_empty) {
            return Err("Mapped boundary is empty or too detailed.".into());
        }
        for points in polygons.iter().flatten() {
            validate_ring(points)?;
        }
        let dense_count: usize = polygons
            .iter()
            .flatten()
            .flat_map(|points| points.windows(2))
            .map(|edge| {
                let longitude = (edge[1][0] - edge[0][0] + 180.0).rem_euclid(360.0) - 180.0;
                (longitude.abs().max((edge[1][1] - edge[0][1]).abs()) / 0.5)
                    .ceil()
                    .max(1.0) as usize
            })
            .sum();
        if dense_count > 1_000_000 {
            return Err("Mapped boundary is too detailed.".into());
        }
        let mut shapes = Vec::with_capacity(polygons.len());
        for polygon in polygons {
            let line = |points: &Vec<[f64; 2]>| {
                geo::LineString::new(points.iter().map(|&[x, y]| geo::Coord { x, y }).collect())
            };
            shapes.push(geo::Polygon::new(
                line(&polygon[0]),
                polygon[1..].iter().map(line).collect(),
            ));
        }
        let area_km2 = shapes
            .iter()
            .map(|polygon| polygon.geodesic_area_signed().abs())
            .sum::<f64>()
            / 1e6;
        let mut preview = Self::empty(1, 0);
        preview.push_record(Record {
            reduction: None,
            id: "mapped-search-boundary".into(),
            anchor_lon: anchor[0],
            anchor_lat: anchor[1],
            area_km2,
            geometry,
            source_kind: "osm".into(),
            path_truncated: false,
            path_length_km: None,
        })?;
        preview.boundaries[0].color = color;
        Ok(preview.finish())
    }

    pub fn geographic_bounds(&self) -> Option<[f64; 4]> {
        let mut bounds = [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
        ];
        for [lon, lat] in self
            .boundaries
            .iter()
            .flat_map(|b| b.polygons.iter().flatten().flatten())
        {
            bounds[0] = bounds[0].min(*lon);
            bounds[1] = bounds[1].max(*lat);
            bounds[2] = bounds[2].max(*lon);
            bounds[3] = bounds[3].min(*lat);
        }
        bounds.iter().all(|v| v.is_finite()).then_some(bounds)
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let doc: Document =
            serde_json::from_slice(bytes).map_err(|e| format!("Invalid AOI response: {e}"))?;
        Self::from_document(doc)
    }

    pub(super) fn from_document(doc: Document) -> Result<Self, String> {
        if doc.schema_version != 1
            || doc.aois.is_empty()
            || doc.aois.len() > 50_000
            || doc.aois.len() != doc.config.count
        {
            return Err("Invalid AOI response version or count".into());
        }
        let mut preview = Self::empty(doc.config.count, doc.config.seed);
        let mut identifiers = std::collections::HashSet::new();
        for aoi in doc.aois {
            preview.push_record(aoi)?;
            if !identifiers.insert(preview.aois.last().unwrap().key()) {
                return Err("Duplicate AOI identifier".into());
            }
        }
        Ok(preview.finish())
    }

    pub(super) fn empty(count: usize, seed: u32) -> Self {
        Self {
            aois: Vec::with_capacity(count),
            edge_count: 0,
            max_edges: 0,
            original_edges: 0,
            removed_holes: 0,
            removed_parts: 0,
            max_area_change: 0.0,
            count,
            mean_area: 0.0,
            min_area: f64::INFINITY,
            max_area: 0.0,
            seed,
            truncated_paths: 0,
            mean_path_length: None,
            boundaries: Vec::with_capacity(count),
            spatial: super::spatial::Index::default(),
            mesh_cache: Mutex::new(Vec::new()),
            gpu_geometry: OnceLock::new(),
            projected_geometry: Mutex::new(Vec::new()),
            #[cfg(not(target_arch = "wasm32"))]
            pending_projections: Mutex::new(std::collections::HashSet::new()),
        }
    }

    pub(super) fn push_record(&mut self, aoi: Record) -> Result<(), String> {
        let prepared = Self::prepare_record(aoi, self.aois.len())?;
        self.push_prepared(prepared);
        Ok(())
    }
    pub(super) fn prepare_record(aoi: Record, index: usize) -> Result<PreparedRecord, String> {
        const COLORS: [[u8; 3]; 6] = [
            [61, 222, 230],
            [255, 190, 86],
            [159, 173, 255],
            [130, 231, 158],
            [244, 142, 189],
            [235, 226, 136],
        ];
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
        if let Some(r) = &aoi.reduction {
            if !r.original_area_km2.is_finite()
                || r.original_area_km2 <= 0.0
                || r.original_edges < super::compact::edges(&polygons)
            {
                return Err("Invalid AOI simplification metadata".into());
            }
        }
        if aoi
            .path_length_km
            .is_some_and(|length| !length.is_finite() || length <= 0.0)
        {
            return Err("Invalid AOI path length".into());
        }
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
        let boundary = Boundary {
            anchor,
            radius,
            rings: rings.into_iter().map(Detail::new).collect(),
            fill: super::fill::Fill::new(&polygons)?,
            path_truncated: aoi.path_truncated,
            path_length_km: aoi.path_length_km,
            reduction: aoi.reduction,
            polygons,
            color: match aoi.source_kind.as_str() {
                "forest" => [130, 231, 158],
                "river" => [61, 222, 230],
                "coast" => [255, 190, 86],
                _ => COLORS[index % COLORS.len()],
            },
        };
        Ok(PreparedRecord {
            aoi: metadata,
            boundary,
        })
    }
    pub(super) fn push_prepared(&mut self, prepared: PreparedRecord) {
        let PreparedRecord { aoi, boundary } = prepared;
        let edges = super::compact::edges(&boundary.polygons);
        self.edge_count += edges;
        self.max_edges = self.max_edges.max(edges);
        self.original_edges += boundary
            .reduction
            .as_ref()
            .map_or(edges, |r| r.original_edges);
        if let Some(r) = &boundary.reduction {
            self.removed_holes += r.removed_holes;
            self.removed_parts += r.removed_parts;
            self.max_area_change = self
                .max_area_change
                .max((aoi.area_km2 / r.original_area_km2 - 1.0).abs());
        }
        self.truncated_paths += usize::from(boundary.path_truncated);
        if let Some(length) = boundary.path_length_km {
            *self.mean_path_length.get_or_insert(0.0) += length / self.count as f64;
        }
        self.mean_area += aoi.area_km2 / self.count as f64;
        self.min_area = self.min_area.min(aoi.area_km2);
        self.max_area = self.max_area.max(aoi.area_km2);
        self.aois.push(aoi);
        self.boundaries.push(boundary);
    }

    pub(super) fn finish(mut self) -> Self {
        let preview = &mut self;
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
        // Build render data while generation/loading still runs off the native UI thread.
        self.gpu_geometry();
        self
    }

    pub(super) fn gpu_geometry(&self) -> Arc<super::gpu::Geometry> {
        Arc::clone(self.gpu_geometry.get_or_init(|| {
            let mut segments = Vec::new();
            let mut triangles = Vec::new();
            let mut dots = Vec::with_capacity(self.boundaries.len());
            for boundary in &self.boundaries {
                let pack = |p: Vector3<f64>| [p.x as f32, p.y as f32, p.z as f32, 0.0];
                let anchor = pack(boundary.anchor);
                let color = [boundary.color[0], boundary.color[1], boundary.color[2], 255];
                triangles.extend(
                    boundary
                        .fill
                        .triangles
                        .iter()
                        .map(|t| super::gpu::Triangle {
                            points: t.map(pack),
                            anchor,
                            radius: boundary.radius as f32,
                            color,
                        }),
                );
                dots.push(super::gpu::Dot {
                    anchor,
                    radius: boundary.radius as f32,
                    color,
                });
                for detail in &boundary.rings {
                    for (level, (error, points)) in detail.levels.iter().enumerate() {
                        let min_scale = detail
                            .levels
                            .get(level + 1)
                            .map_or(0.0, |(next_error, _)| (0.7 / next_error) as f32);
                        let max_scale = if *error > 0.0 {
                            (0.7 / error) as f32
                        } else {
                            f32::MAX
                        };
                        for edge in points.windows(2) {
                            segments.push(super::gpu::Segment {
                                start: pack(edge[0]),
                                end: pack(edge[1]),
                                anchor,
                                radius: boundary.radius as f32,
                                color,
                                scales: [min_scale, max_scale],
                            });
                        }
                    }
                }
            }
            let mut geometry = super::gpu::Geometry::new(segments, dots);
            geometry.triangles = triangles;
            Arc::new(geometry)
        }))
    }

    pub(super) fn projected_geometry(
        &self,
        projection: crate::projection::ProjectionKind,
    ) -> Arc<super::gpu::Geometry> {
        let key = projection.shader_id();
        if let Some(geometry) = self.cached_projected_geometry(projection) {
            return geometry;
        }
        let mut segments = Vec::new();
        let mut dots = Vec::new();
        for boundary in &self.boundaries {
            let paths = projected_boundary(boundary, projection);
            let Some(first) = paths.first().and_then(|path| path.first()) else {
                continue;
            };
            let lon = (-boundary.anchor.z).atan2(boundary.anchor.x).to_degrees();
            let lat = boundary.anchor.y.clamp(-1.0, 1.0).asin().to_degrees();
            let anchor = projection
                .instance()
                .project(lat, lon)
                .map_or([first.x, first.y], |(x, y)| [x, y]);
            let radius = paths
                .iter()
                .flatten()
                .map(|p| (p.x - anchor[0]).hypot(p.y - anchor[1]))
                .fold(0.0_f64, f64::max)
                .max(1e-9) as f32;
            let anchor = super::gpu::split_point(anchor);
            let color = [boundary.color[0], boundary.color[1], boundary.color[2], 255];
            dots.push(super::gpu::Dot {
                anchor,
                radius,
                color,
            });
            for path in paths {
                let detail = Detail::with_unit(path, 180.0 / std::f64::consts::PI);
                for (level, (error, points)) in detail.levels.iter().enumerate() {
                    let min_scale = detail
                        .levels
                        .get(level + 1)
                        .map_or(0.0, |(next_error, _)| (0.7 / next_error) as f32);
                    let max_scale = if *error > 0.0 {
                        (0.7 / error) as f32
                    } else {
                        f32::MAX
                    };
                    for edge in points.windows(2) {
                        segments.push(super::gpu::Segment {
                            start: super::gpu::split_point([edge[0].x, edge[0].y]),
                            end: super::gpu::split_point([edge[1].x, edge[1].y]),
                            anchor,
                            radius,
                            color,
                            scales: [min_scale, max_scale],
                        });
                    }
                }
            }
        }
        let geometry = Arc::new(super::gpu::Geometry::new(segments, dots));
        let mut cache = self.projected_geometry.lock().unwrap();
        if cache.len() >= 4 {
            cache.remove(0);
        }
        cache.push((key, Arc::clone(&geometry)));
        geometry
    }

    fn cached_projected_geometry(
        &self,
        projection: crate::projection::ProjectionKind,
    ) -> Option<Arc<super::gpu::Geometry>> {
        let mut cache = self.projected_geometry.lock().unwrap();
        let index = cache
            .iter()
            .position(|(id, _)| *id == projection.shader_id())?;
        let entry = cache.remove(index);
        let geometry = Arc::clone(&entry.1);
        cache.push(entry);
        Some(geometry)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn projected_geometry_ready(
        &self,
        projection: crate::projection::ProjectionKind,
    ) -> bool {
        self.projected_geometry
            .lock()
            .unwrap()
            .iter()
            .any(|(id, _)| *id == projection.shader_id())
    }

    pub(super) fn projected_geometry_for_view(
        self: &Arc<Self>,
        projection: crate::projection::ProjectionKind,
        _ctx: &egui::Context,
    ) -> Option<Arc<super::gpu::Geometry>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(geometry) = self.cached_projected_geometry(projection) {
                return Some(geometry);
            }
            let key = projection.shader_id();
            let mut pending = self.pending_projections.lock().unwrap();
            if pending.len() < 2 && pending.insert(key) {
                let preview = Arc::clone(self);
                let ctx = _ctx.clone();
                std::thread::spawn(move || {
                    preview.projected_geometry(projection);
                    preview.pending_projections.lock().unwrap().remove(&key);
                    ctx.request_repaint();
                });
            }
            None
        }
        #[cfg(target_arch = "wasm32")]
        {
            Some(self.projected_geometry(projection))
        }
    }
}

fn projected_boundary(
    boundary: &Boundary,
    projection: crate::projection::ProjectionKind,
) -> Vec<Vec<Vector3<f64>>> {
    boundary
        .polygons
        .iter()
        .flat_map(|polygon| polygon.iter())
        .flat_map(|ring| crate::projection::project_path(projection.instance(), ring))
        .map(|path| {
            path.into_iter()
                .map(|[x, y]| Vector3::new(x, y, 0.0))
                .collect()
        })
        .collect()
}

#[derive(Clone)]
pub(crate) struct Hit {
    pub preview: Arc<Preview>,
    pub index: usize,
}

fn contains_ring(points: &[[f64; 2]], x: f64, y: f64) -> bool {
    if points
        .windows(2)
        .any(|edge| (edge[1][0] - edge[0][0]).abs() > 180.0)
    {
        let Some(first) = points.first() else {
            return false;
        };
        let x = x + ((first[0] - x) / 360.0).round() * 360.0;
        let mut ax = first[0];
        let mut inside = false;
        for edge in points.windows(2) {
            let (ay, by) = (edge[0][1], edge[1][1]);
            let bx = ax + (edge[1][0] - edge[0][0] + 180.0).rem_euclid(360.0) - 180.0;
            if (ay > y) != (by > y) && x < ax + (y - ay) * (bx - ax) / (by - ay) {
                inside = !inside;
            }
            ax = bx;
        }
        return inside;
    }
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
    fill_opacity: f32,
    selected: Option<usize>,
    use_gpu: bool,
    projection: Option<crate::projection::ProjectionKind>,
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
            fill_opacity: 0.1,
            selected: None,
            use_gpu: false,
            projection: None,
        }
    }

    pub fn with_fill(mut self, opacity: f32) -> Self {
        self.fill_opacity = opacity.clamp(0.0, 1.0);
        self
    }

    pub fn projected(
        preview: Arc<Preview>,
        projection: crate::projection::ProjectionKind,
        opacity: f32,
    ) -> Self {
        let mut overlay = Self::new(preview, Matrix3::identity(), 1.0, 0.0, opacity);
        overlay.projection = Some(projection);
        overlay
    }

    pub fn with_selection(mut self, selected: Option<usize>) -> Self {
        self.selected = selected;
        self
    }

    pub fn with_gpu(mut self, enabled: bool) -> Self {
        self.use_gpu = enabled;
        self
    }

    /// Exact polygon picking (including holes), with a small screen-space
    /// fallback for AOIs represented by dots. Uses the existing spatial index.
    pub fn pick(&self, pos: egui::Pos2, transform: &PlotTransform) -> Vec<Hit> {
        if let Some(projection) = self.projection {
            return self.pick_projected(pos, transform, projection);
        }
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
        if let Some(projection) = self.projection {
            for path in projected_boundary(boundary, projection) {
                for edge in path.windows(2) {
                    let a = transform.position_from_point(&PlotPoint::new(edge[0].x, edge[0].y));
                    let b = transform.position_from_point(&PlotPoint::new(edge[1].x, edge[1].y));
                    if transform
                        .frame()
                        .intersects(Rect::from_two_pos(a, b).expand(2.0))
                    {
                        line_quad(&mut mesh, a, b, color, 2.5);
                    }
                }
            }
            return mesh;
        }
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

    fn pick_projected(
        &self,
        pos: egui::Pos2,
        transform: &PlotTransform,
        projection: crate::projection::ProjectionKind,
    ) -> Vec<Hit> {
        let proj = projection.instance();
        let cursor = transform.value_from_position(pos);
        let Some((lat, lon)) = proj.inverse(cursor.x, cursor.y) else {
            return Vec::new();
        };
        let point = unit([lon, lat]);
        let mut margin = 0.0_f64;
        for offset in [
            egui::vec2(6.0, 0.0),
            egui::vec2(-6.0, 0.0),
            egui::vec2(0.0, 6.0),
            egui::vec2(0.0, -6.0),
        ] {
            let nearby = transform.value_from_position(pos + offset);
            if let Some((lat, lon)) = proj.inverse(nearby.x, nearby.y) {
                margin = margin.max((unit([lon, lat]) - point).norm());
            }
        }
        let scale = transform
            .dpos_dvalue_x()
            .abs()
            .max(transform.dpos_dvalue_y().abs())
            * 180.0
            / std::f64::consts::PI;
        let mut found = Vec::new();
        for index in self
            .preview
            .spatial
            .query(|sphere| (sphere.center - point).norm() <= sphere.radius + margin + 1e-5)
        {
            let boundary = &self.preview.boundaries[index];
            let exact = contains(&boundary.polygons, lon, lat);
            let anchor_lon = (-boundary.anchor.z).atan2(boundary.anchor.x).to_degrees();
            let anchor_lat = boundary.anchor.y.clamp(-1.0, 1.0).asin().to_degrees();
            let dot = boundary.radius * scale < 3.0
                && proj.project(anchor_lat, anchor_lon).is_some_and(|(x, y)| {
                    transform
                        .position_from_point(&PlotPoint::new(x, y))
                        .distance(pos)
                        <= 6.0
                });
            if exact || dot {
                found.push(Hit {
                    preview: Arc::clone(&self.preview),
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

    pub(super) fn projected_mesh(
        &self,
        transform: &PlotTransform,
        projection: crate::projection::ProjectionKind,
    ) -> Mesh {
        let geometry = self.preview.projected_geometry(projection);
        let scale = transform
            .dpos_dvalue_x()
            .abs()
            .max(transform.dpos_dvalue_y().abs());
        let screen = |p| {
            let [x, y] = super::gpu::join_point(p);
            transform.position_from_point(&PlotPoint::new(x, y))
        };
        let color = |c: [u8; 4]| {
            Color32::from_rgba_unmultiplied(
                c[0],
                c[1],
                c[2],
                (c[3] as f32 * self.opacity).round() as u8,
            )
        };
        let mut mesh = Mesh::default();
        for edge in &geometry.segments {
            if f64::from(edge.radius) * scale < 3.0
                || scale <= f64::from(edge.scales[0])
                || scale > f64::from(edge.scales[1])
            {
                continue;
            }
            let (a, b) = (screen(edge.start), screen(edge.end));
            if transform
                .frame()
                .intersects(Rect::from_two_pos(a, b).expand(1.0))
            {
                line_quad(&mut mesh, a, b, color(edge.color), 1.2);
            }
        }
        for dot in &geometry.dots {
            let anchor = screen(dot.anchor);
            if f64::from(dot.radius) * scale < 3.0 && transform.frame().contains(anchor) {
                mesh.add_colored_rect(
                    Rect::from_center_size(anchor, egui::vec2(2.5, 2.5)),
                    color(dot.color),
                );
            }
        }
        mesh
    }

    fn cached_mesh(&self, transform: &PlotTransform) -> Arc<Mesh> {
        if let Some(projection) = self.projection {
            return Arc::new(self.projected_mesh(transform, projection));
        }
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
        key.extend([
            self.radius,
            self.polar_scale,
            self.opacity as f64,
            self.fill_opacity as f64,
        ]);
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

    pub(super) fn mesh(&self, transform: &PlotTransform) -> Mesh {
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
            if self.fill_opacity > 0.0 {
                let fill_color = Color32::from_rgba_unmultiplied(
                    boundary.color[0],
                    boundary.color[1],
                    boundary.color[2],
                    (self.fill_opacity * 255.0).round() as u8,
                );
                boundary
                    .fill
                    .paint(&mut mesh, fill_color, *frame, |p| self.project(p), &screen);
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
    fn shapes(&self, ui: &egui::Ui, transform: &PlotTransform, shapes: &mut Vec<Shape>) {
        if self.use_gpu {
            let callback = if let Some(projection) = self.projection {
                let Some(geometry) = self
                    .preview
                    .projected_geometry_for_view(projection, ui.ctx())
                else {
                    return;
                };
                super::gpu::PaintCallback::projected(ui.id(), geometry, self.opacity, transform)
            } else {
                super::gpu::PaintCallback::new(
                    ui.id(),
                    self.preview.gpu_geometry(),
                    &self.rotation,
                    self.radius,
                    self.polar_scale,
                    self.opacity,
                    transform,
                )
            }
            .with_fill(self.fill_opacity);
            shapes
                .push(egui_wgpu::Callback::new_paint_callback(*transform.frame(), callback).into());
        } else {
            shapes.push(Shape::Mesh(self.cached_mesh(transform)));
        }
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
    #[cfg(not(target_arch = "wasm32"))]
    fn manual_areas_keep_their_spherical_boundaries_and_colors() {
        use crate::config::{AoiJobMode, AreaOfInterest, SpaceCompReducerPlacement};
        for (lat, lon) in [(0.0, 0.0), (60.0, 179.99), (-70.0, -179.99)] {
            let area = AreaOfInterest {
                name: "Test".into(),
                lat,
                lon,
                radius_km: 10.0,
                color: Color32::from_rgb(255, 70, 90),
                ground_station_idx: None,
                job_mode: AoiJobMode::Route,
                job_n: 3,
                reducer_placement: SpaceCompReducerPlacement::NearMappers,
                show_spacecomp_flow: false,
                selected: false,
            };
            let preview = Preview::from_areas(&[area], 6371.0).unwrap();
            assert_eq!(preview.count, 1);
            let boundary = &preview.boundaries[0];
            assert_eq!(boundary.color, [255, 70, 90]);
            for point in &boundary.rings[0].levels[0].1 {
                let distance = boundary
                    .anchor
                    .cross(point)
                    .norm()
                    .atan2(boundary.anchor.dot(point))
                    * 6371.0;
                assert!((distance - 10.0).abs() < 1e-7);
            }
            assert!(!preview
                .projected_geometry(crate::projection::ProjectionKind::Mercator)
                .segments
                .is_empty());
        }
    }
    #[test]
    fn gpu_geometry_is_cached_and_keeps_the_cpu_fallback_and_picking() {
        let preview = Arc::new(Preview::from_json(br#"{"schema_version":1,"config":{"seed":42,"count":1},"aois":[{"area_km2":100,"anchor_lon":-90,"anchor_lat":0,"geometry":{"type":"Polygon","coordinates":[[[-91,-1],[-89,-1],[-89,1],[-91,1],[-91,-1]]]}}]}"#).unwrap());
        let geometry = preview.gpu_geometry();
        let transform = PlotTransform::new(
            Rect::from_min_size(Pos2::ZERO, egui::vec2(800.0, 600.0)),
            PlotBounds::new_symmetrical(7000.0),
            [false, false],
        );
        let context = egui::Context::default();
        let mut output = context.run_ui(Default::default(), |ui| {
            for angle in [0.0, 0.2, 0.5] {
                let overlay = Overlay::new(
                    preview.clone(),
                    crate::math::rotation_from_drag(angle, 0.0),
                    6371.0,
                    0.0,
                    0.8,
                );
                let mut shapes = Vec::new();
                overlay.with_gpu(true).shapes(ui, &transform, &mut shapes);
                assert_eq!(shapes.len(), 1);
                assert!(matches!(shapes[0], Shape::Callback(_)));
                assert!(Arc::ptr_eq(&geometry, &preview.gpu_geometry()));
            }
            let overlay = Overlay::new(preview.clone(), Matrix3::identity(), 6371.0, 0.0, 0.8);
            let mut shapes = Vec::new();
            overlay.shapes(ui, &transform, &mut shapes);
            assert!(matches!(shapes[0], Shape::Mesh(_)));
            let center = transform.position_from_point(&PlotPoint::new(0.0, 0.0));
            assert_eq!(overlay.pick(center, &transform).len(), 1);
        });
        output.textures_delta.clear();
    }
    #[test]
    fn fill_opacity_is_independent_cached_and_clipped_at_the_horizon() {
        let preview = Arc::new(Preview::from_json(br#"{"schema_version":1,"config":{"seed":42,"count":1},"aois":[{"area_km2":100,"anchor_lon":-90,"anchor_lat":0,"geometry":{"type":"Polygon","coordinates":[[[-91,-1],[-89,-1],[-89,1],[-91,1],[-91,-1]]]}}]}"#).unwrap());
        let transform = PlotTransform::new(
            Rect::from_min_size(Pos2::ZERO, egui::vec2(800., 800.)),
            PlotBounds::new_symmetrical(1000.),
            [false, false],
        );
        let overlay = Overlay::new(preview.clone(), Matrix3::identity(), 6371., 0., 0.8);
        let mesh = overlay.cached_mesh(&transform);
        assert!(mesh.is_valid());
        assert!(mesh.vertices.iter().any(|v| v.color.a() == 26)); // 10% fill
        assert!(mesh.vertices.iter().any(|v| v.color.a() == 204)); // 80% outline
        let unfilled = overlay.with_fill(0.).cached_mesh(&transform);
        assert!(!Arc::ptr_eq(&mesh, &unfilled));
        assert!(!unfilled.vertices.iter().any(|v| v.color.a() == 26));
        assert!(mesh.indices.len() > unfilled.indices.len());
        let turn = crate::math::rotation_from_drag(std::f64::consts::PI, 0.);
        assert!(Overlay::new(preview.clone(), turn, 6371., 0., 0.8)
            .mesh(&transform)
            .vertices
            .is_empty());
        let limb = crate::math::rotation_from_drag(std::f64::consts::FRAC_PI_2, 0.);
        let globe = PlotTransform::new(
            Rect::from_min_size(Pos2::ZERO, egui::vec2(800., 800.)),
            PlotBounds::new_symmetrical(7000.),
            [false, false],
        );
        let clipped = Overlay::new(preview, limb, 6371., 0., 0.8).mesh(&globe);
        assert!(clipped.is_valid() && !clipped.vertices.is_empty());
    }

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
    fn projected_overlays_render_pick_holes_and_reuse_geometry() {
        use crate::projection::ProjectionKind::*;
        let preview = Arc::new(
            Preview::from_json(
                &serde_json::to_vec(&serde_json::json!({
                    "schema_version": 1, "config": {"count": 1, "seed": 42},
                    "aois": [{"area_km2": 1000, "anchor_lon": 25.5, "anchor_lat": 48.7,
                        "geometry": {"type": "Polygon", "coordinates": [
                            [[22,45],[29,45],[29,52],[22,52],[22,45]],
                            [[25,48],[26,48],[26,49],[25,49],[25,48]]
                        ]}}]
                }))
                .unwrap(),
            )
            .unwrap(),
        );
        for projection in [
            Equirectangular,
            Mercator,
            Mollweide,
            Sinusoidal,
            AzimuthalEquidistant,
            Hammer,
            HEALPix,
            Cassini,
            TransverseMercator,
            LambertAzimuthalEqualArea,
            GallPeters,
            PeirceQuincuncial,
        ] {
            let proj = projection.instance();
            let (x, y) = proj.project(48.7, 25.5).unwrap();
            let transform = PlotTransform::new(
                Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)),
                PlotBounds::from_min_max([x - 10.0, y - 10.0], [x + 10.0, y + 10.0]),
                [false, false],
            );
            let overlay = Overlay::projected(Arc::clone(&preview), projection, 0.3);
            let geometry = preview.projected_geometry(projection);
            assert!(Arc::ptr_eq(
                &geometry,
                &preview.projected_geometry(projection)
            ));
            let mesh = overlay.projected_mesh(&transform, projection);
            assert!(!mesh.vertices.is_empty(), "{}", projection.label());
            assert!(mesh.is_valid());
            assert!(mesh.vertices.iter().all(|v| v.color.a() <= 77));
            let screen = |lat, lon| {
                let (x, y) = proj.project(lat, lon).unwrap();
                transform.position_from_point(&PlotPoint::new(x, y))
            };
            assert!(
                overlay.pick(screen(48.7, 25.5), &transform).is_empty(),
                "{} must preserve holes",
                projection.label()
            );
            assert_eq!(
                overlay.pick(screen(48.0, 24.0), &transform).len(),
                1,
                "{}",
                projection.label()
            );
            let selection = overlay.with_selection(Some(0)).selection_mesh(&transform);
            assert!(!selection.vertices.is_empty());
            assert!(selection.is_valid());
        }
        assert!(preview.projected_geometry.lock().unwrap().len() <= 4);
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
    fn real_gdp_generation_and_batched_render() {
        let start = std::time::Instant::now();
        let mut job = super::super::generate::Job::new(
            super::super::generate::Settings::Gdp(super::super::generate::Gdp {
                count: 10000,
                seed: 42,
                mean_area: 1000.0,
                size_ratio: 100.0,
                uniform: false,
                hotspot_count: 256,
                hotspot_spread: 10.0,
                hotspot_fraction: 0.8,
                shape: "oriented_rectangle".into(),
                random_rotation: true,
                max_aspect: 3.0,
            }),
            Some(super::super::data::load(&super::super::data::Input::default(), "gdp").unwrap()),
        )
        .unwrap();
        let preview = loop {
            if let Some(preview) = job.step().unwrap() {
                break preview;
            }
        };
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
