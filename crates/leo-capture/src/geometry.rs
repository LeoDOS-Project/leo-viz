//! Common local equal-area charts preserve polygon holes and overlap identity.
//! AOIs are split at fixed geographic cell boundaries; cells are NOT point
//! samples. All coverage/union/difference operations act on clipped polygons.
use crate::{
    spatial::{Index, Sphere},
    workload::Job,
};
use geo::{
    Area, BooleanOps, BoundingRect, Coord, LineString, MultiPolygon, Polygon, Rect, Validation,
};
use leo_sim::geometry::{dot, ground_position, norm, scale, sub, Vector, EARTH_RADIUS_KM as R};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub type Polygons = Vec<Vec<Vec<[f64; 2]>>>;
pub type Shape = MultiPolygon<f64>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Region {
    pub job_key: String,
    /// GeoJSON order: polygon, ring (exterior first), vertex [longitude, latitude].
    /// Input polygons must be valid and split at the dateline, as leo-aoi exports.
    pub polygons: Polygons,
}

pub(crate) fn unit(v: Vector) -> Vector {
    scale(v, 1.0 / norm(v))
}
pub(crate) fn cross(a: Vector, b: Vector) -> Vector {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub(crate) fn mix(a: Vector, b: Vector, u: f64) -> Vector {
    let angle = dot(a, b).clamp(-1.0, 1.0).acos();
    if angle < 1e-10 {
        return a;
    }
    std::array::from_fn(|i| {
        (a[i] * ((1.0 - u) * angle).sin() + b[i] * (u * angle).sin()) / angle.sin()
    })
}
pub(crate) fn normal_between(a: Vector, b: Vector, q: Vector, u: f64) -> Vector {
    let n: Vector = std::array::from_fn(|i| a[i] * (1.0 - u) + b[i] * u);
    unit(std::array::from_fn(|i| n[i] - q[i] * dot(n, q)))
}
fn point([lon, lat]: [f64; 2]) -> Vector {
    ground_position(lat, lon, 1.0, 0.0)
}

#[derive(Clone)]
pub(crate) struct Chart {
    pub center: Vector,
    east: Vector,
    north: Vector,
}
impl Chart {
    fn new(lon: f64, lat: f64) -> Self {
        let center = point([lon, lat]);
        let east = point([lon + 90.0, 0.0]);
        let north = cross(center, east);
        Self {
            center,
            east,
            north,
        }
    }
    pub fn project(&self, v: Vector) -> Coord<f64> {
        let k = R * (2.0 / (1.0 + dot(self.center, v))).sqrt();
        Coord {
            x: k * dot(v, self.east),
            y: k * dot(v, self.north),
        }
    }
    pub fn inverse(&self, p: Coord<f64>) -> Vector {
        let rho = p.x.hypot(p.y);
        if rho < 1e-12 {
            return self.center;
        }
        let c = 2.0 * (rho / (2.0 * R)).clamp(0.0, 1.0).asin();
        std::array::from_fn(|i| {
            c.cos() * self.center[i] + c.sin() / rho * (p.x * self.east[i] + p.y * self.north[i])
        })
    }
    fn ring(&self, ring: &LineString<f64>, edge_km: f64) -> LineString<f64> {
        let mut out = Vec::new();
        for edge in ring.0.windows(2) {
            let (a, b) = (edge[0], edge[1]);
            // Preserve the GeoJSON straight lon/lat edges, including parallels.
            let steps = ((a.x - b.x).abs().max((a.y - b.y).abs()).to_radians() * R / edge_km)
                .ceil()
                .max(1.0) as usize;
            for i in 0..steps {
                let t = i as f64 / steps as f64;
                out.push(self.project(point([a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t])));
            }
        }
        if let Some(p) = out.first().copied() {
            out.push(p);
        }
        LineString::new(out)
    }
    fn shape(&self, shape: &Shape, edge: f64) -> Shape {
        MultiPolygon::new(
            shape
                .0
                .iter()
                .map(|p| {
                    Polygon::new(
                        self.ring(p.exterior(), edge),
                        p.interiors().iter().map(|r| self.ring(r, edge)).collect(),
                    )
                })
                .collect(),
        )
    }
}

pub(crate) struct Piece {
    pub job: usize,
    pub shape: Shape,
    pub bounds: Rect<f64>,
}
pub(crate) struct Patch {
    pub chart: Chart,
    pub sphere: Sphere,
    pub pieces: Vec<Piece>,
}
pub struct Atlas {
    pub(crate) patches: Vec<Patch>,
    index: Index,
    pub job_areas_km2: Vec<f64>,
    pub polygon_pieces: usize,
}
impl Atlas {
    pub fn new(
        jobs: &[Job],
        regions: &[Region],
        cell_deg: f64,
        edge_km: f64,
        mut progress: impl FnMut(f32) -> bool,
    ) -> Result<Self, String> {
        if !cell_deg.is_finite()
            || !(0.25..=10.0).contains(&cell_deg)
            || (180.0 / cell_deg - (180.0 / cell_deg).round()).abs() > 1e-8
            || !edge_km.is_finite()
            || !(0.05..=10.0).contains(&edge_km)
        {
            return Err(
                "Cell size must divide 180° (0.25–10°); edge spacing must be 0.05–10 km.".into(),
            );
        }
        let keys: BTreeMap<_, _> = jobs
            .iter()
            .enumerate()
            .map(|(i, j)| (j.aoi.key(), i))
            .collect();
        if keys.len() != jobs.len() || regions.len() != jobs.len() {
            return Err("Provide exactly one geometry per unique job.".into());
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut patches = BTreeMap::<(i32, i32), Patch>::new();
        let mut areas = vec![0.0; jobs.len()];
        for (n, region) in regions.iter().enumerate() {
            if !progress(n as f32 / regions.len().max(1) as f32) {
                return Err("Cancelled.".into());
            }
            let job = *keys
                .get(&region.job_key)
                .ok_or("Geometry has no matching job")?;
            if crate::workload::hash(
                &serde_json::to_vec(&region.polygons).map_err(|e| e.to_string())?,
            ) != jobs[job].aoi.geometry_hash
            {
                return Err(format!("Geometry hash mismatch for {}", region.job_key));
            }
            if !seen.insert(job) {
                return Err("Duplicate geometry for job.".into());
            }
            let mut per_cell = BTreeMap::<(i32, i32), Shape>::new();
            for poly in &region.polygons {
                if poly.is_empty()
                    || poly.iter().any(|ring| {
                        ring.len() < 4
                            || ring.first() != ring.last()
                            || ring.iter().any(|p| {
                                !p[0].is_finite()
                                    || !p[1].is_finite()
                                    || p[0].abs() > 180.0
                                    || p[1].abs() > 90.0
                            })
                            || ring.windows(2).any(|w| (w[0][0] - w[1][0]).abs() > 180.0)
                    })
                {
                    return Err(
                        "Invalid polygon; rings must be closed and split at the dateline.".into(),
                    );
                }
                let rings: Vec<LineString<_>> = poly
                    .iter()
                    .map(|r| r.iter().map(|p| (p[0], p[1])).collect())
                    .collect();
                let shape =
                    MultiPolygon::new(vec![Polygon::new(rings[0].clone(), rings[1..].to_vec())]);
                if !shape.is_valid() {
                    return Err(format!("Invalid polygon topology for {}", region.job_key));
                }
                let bounds = shape.bounding_rect().ok_or("Empty polygon")?;
                let nx = (360.0 / cell_deg).round() as i32;
                let ny = (180.0 / cell_deg).round() as i32;
                let ix = |x: f64| (((x + 180.0) / cell_deg).floor() as i32).clamp(0, nx - 1);
                let iy = |y: f64| (((y + 90.0) / cell_deg).floor() as i32).clamp(0, ny - 1);
                for y in iy(bounds.min().y)..=iy(bounds.max().y) {
                    for x in ix(bounds.min().x)..=ix(bounds.max().x) {
                        let lon = -180.0 + x as f64 * cell_deg;
                        let lat = -90.0 + y as f64 * cell_deg;
                        let clip = shape.intersection(
                            &Rect::new((lon, lat), (lon + cell_deg, lat + cell_deg)).to_polygon(),
                        );
                        if clip.unsigned_area() <= 1e-18 {
                            continue;
                        }
                        let patch = patches.entry((x, y)).or_insert_with(|| {
                            let chart = Chart::new(lon + cell_deg / 2.0, lat + cell_deg / 2.0);
                            // Angular distance from center bounded by half latitude + half longitude span.
                            let sphere = Sphere {
                                center: chart.center,
                                radius: cell_deg.to_radians() + 1e-8,
                            };
                            Patch {
                                chart,
                                sphere,
                                pieces: vec![],
                            }
                        });
                        let projected = patch.chart.shape(&clip, edge_km);
                        let previous = per_cell
                            .entry((x, y))
                            .or_insert_with(|| MultiPolygon::new(vec![]));
                        *previous = previous.union(&projected);
                    }
                }
            }
            for (key, shape) in per_cell {
                let area = shape.unsigned_area();
                if area > 0.0 {
                    areas[job] += area;
                    let bounds = shape.bounding_rect().unwrap();
                    patches
                        .get_mut(&key)
                        .unwrap()
                        .pieces
                        .push(Piece { job, shape, bounds });
                }
            }
            if areas[job] <= 0.0 {
                return Err(format!("{} has no positive-area geometry", region.job_key));
            }
        }
        let patches: Vec<_> = patches.into_values().collect();
        let polygon_pieces = patches.iter().map(|p| p.pieces.len()).sum();
        let index = Index::build(&patches.iter().map(|p| p.sphere).collect::<Vec<_>>());
        Ok(Self {
            patches,
            index,
            job_areas_km2: areas,
            polygon_pieces,
        })
    }
    pub fn patch_count(&self) -> usize {
        self.patches.len()
    }
    pub(crate) fn candidates(&self, a: Vector, b: Vector, width: f64) -> Vec<usize> {
        let center = unit(std::array::from_fn(|i| a[i] + b[i]));
        let radius = norm(sub(a, center)).max(norm(sub(b, center))) + width + 1e-8;
        self.index
            .query(|s| norm(sub(s.center, center)) <= s.radius + radius)
    }
}

/// A swept scanline has flat temporal boundaries, never rounded end caps.
pub(crate) fn strip(
    chart: &Chart,
    a: Vector,
    b: Vector,
    na: Vector,
    nb: Vector,
    half_width: f64,
    edge_km: f64,
) -> Shape {
    if norm(cross(a, b)) < 1e-14 {
        return MultiPolygon::new(vec![]);
    }
    let angle = dot(a, b).clamp(-1.0, 1.0).acos();
    let along = (angle * R / edge_km).ceil().max(1.0) as usize;
    let across = (2.0 * half_width * R / edge_km).ceil().max(1.0) as usize;
    let offset = |u: f64, w: f64| {
        let q = mix(a, b, u);
        let n = normal_between(na, nb, q, u);
        std::array::from_fn(|i| q[i] * w.cos() + n[i] * w.sin())
    };
    let mut ring = Vec::with_capacity(2 * (along + across) + 1);
    for i in 0..along {
        ring.push(chart.project(offset(i as f64 / along as f64, half_width)));
    }
    for i in 0..across {
        ring.push(chart.project(offset(
            1.0,
            half_width * (1.0 - 2.0 * i as f64 / across as f64),
        )));
    }
    for i in 0..along {
        ring.push(chart.project(offset(1.0 - i as f64 / along as f64, -half_width)));
    }
    for i in 0..across {
        ring.push(chart.project(offset(
            0.0,
            half_width * (-1.0 + 2.0 * i as f64 / across as f64),
        )));
    }
    ring.push(ring[0]);
    MultiPolygon::new(vec![Polygon::new(LineString::new(ring), vec![])])
}

pub fn coordinates(shape: &Shape) -> Polygons {
    shape
        .0
        .iter()
        .map(|p| {
            std::iter::once(p.exterior())
                .chain(p.interiors())
                .map(|r| r.0.iter().map(|p| [p.x, p.y]).collect())
                .collect()
        })
        .collect()
}
