//! Compact equal-area tile footprints; work is bounded by local geometry, not Earth-sized bitmaps.
use super::preview::Geometry;
use geo::{
    algorithm::unary_union, BooleanOps, BoundingRect, Coord, LineString, MapCoords, MultiPolygon,
    Polygon, Rect,
};
use std::{collections::BTreeMap, f64::consts::PI};

pub(super) const RADIUS: f64 = leo_sim::geometry::EARTH_RADIUS_KM;

#[derive(Clone, Copy)]
pub(super) struct Grid {
    pub columns: i64,
    pub rows: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TileRect {
    pub col: i64,
    pub row: i64,
    pub width: i64,
    pub height: i64,
}

impl Grid {
    pub fn new(edge: f64) -> Self {
        Self {
            columns: (2.0 * PI * RADIUS / edge).round() as i64,
            rows: (2.0 * RADIUS / edge).round() as i64,
        }
    }
    pub fn area(self) -> f64 {
        4.0 * PI * RADIUS.powi(2) / (self.columns * self.rows) as f64
    }
    pub fn lonlat(self, x: f64, y: f64) -> [f64; 2] {
        [
            360.0 * x / self.columns as f64 - 180.0,
            (1.0 - 2.0 * y / self.rows as f64)
                .clamp(-1.0, 1.0)
                .asin()
                .to_degrees(),
        ]
    }
    pub fn project(self, p: [f64; 2], center_lon: f64) -> Coord<f64> {
        let lon = center_lon + (p[0] - center_lon + 180.0).rem_euclid(360.0) - 180.0;
        Coord {
            x: (lon + 180.0) / 360.0 * self.columns as f64,
            y: (1.0 - p[1].to_radians().sin()) / 2.0 * self.rows as f64,
        }
    }
    pub fn rectangle(
        self,
        anchor: [f64; 2],
        tiles: i64,
        aspect: f64,
    ) -> Result<Vec<TileRect>, String> {
        let col =
            (((anchor[0] + 180.0).rem_euclid(360.0)) / 360.0 * self.columns as f64).floor() as i64;
        let row = ((1.0 - anchor[1].to_radians().sin()) / 2.0 * self.rows as f64).floor() as i64;
        let height = ((tiles as f64 / aspect).sqrt().round() as i64).clamp(1, self.rows.min(tiles));
        let width = (tiles as f64 / height as f64).round().max(1.0) as i64;
        self.split(TileRect {
            col: col - width / 2,
            row: (row - height / 2).clamp(0, self.rows - height),
            width,
            height,
        })
    }
    fn split(self, rect: TileRect) -> Result<Vec<TileRect>, String> {
        if rect.width < 1
            || rect.width > self.columns
            || rect.height < 1
            || rect.row < 0
            || rect.row + rect.height > self.rows
        {
            return Err("AOI footprint exceeds its tile grid; reduce its size".into());
        }
        let start = rect.col.rem_euclid(self.columns);
        let width = rect.width.min(self.columns - start);
        let mut result = vec![TileRect {
            col: start,
            width,
            ..rect
        }];
        if width < rect.width {
            result.push(TileRect {
                col: 0,
                width: rect.width - width,
                ..rect
            });
        }
        Ok(result)
    }

    pub fn cover(self, polygon: &MultiPolygon<f64>) -> Result<Vec<TileRect>, String> {
        let bounds = polygon
            .bounding_rect()
            .ok_or("Empty geographic footprint")?;
        if ![
            bounds.min().x,
            bounds.min().y,
            bounds.max().x,
            bounds.max().y,
        ]
        .iter()
        .all(|v| v.is_finite())
        {
            return Err("Invalid geographic footprint".into());
        }
        let y0 = (bounds.min().y.floor() as i64).max(0);
        let y1 = (bounds.max().y.ceil() as i64).min(self.rows);
        let width = bounds.max().x.ceil() - bounds.min().x.floor();
        if width <= 0.0
            || y1 <= y0
            || width > self.columns as f64
            || width * (y1 - y0) as f64 > 100_000_000.0
        {
            return Err(
                "Footprint exceeds 100 million tiles; reduce its size or increase the tile edge"
                    .into(),
            );
        }
        let edges: Vec<_> = polygon
            .0
            .iter()
            .flat_map(|p| std::iter::once(p.exterior()).chain(p.interiors()))
            .flat_map(|ring| ring.0.windows(2).map(|w| (w[0], w[1])))
            .collect();
        let mut active: BTreeMap<(i64, i64), TileRect> = BTreeMap::new();
        let mut result = Vec::new();
        for row in y0..y1 {
            let y = row as f64 + 0.5;
            let mut crossings = Vec::new();
            let mut spans = Vec::new();
            for &(a, b) in &edges {
                if (a.y > y) != (b.y > y) {
                    crossings.push(a.x + (b.x - a.x) * (y - a.y) / (b.y - a.y));
                }
                // Include cells touched by a boundary, including thin corridors.
                if a.y.max(b.y) < row as f64 || a.y.min(b.y) > row as f64 + 1.0 {
                    continue;
                }
                let (mut lo, mut hi) = (a.x, b.x);
                if (b.y - a.y).abs() > 1e-12 {
                    let t0 = ((row as f64 - a.y) / (b.y - a.y)).clamp(0.0, 1.0);
                    let t1 = ((row as f64 + 1.0 - a.y) / (b.y - a.y)).clamp(0.0, 1.0);
                    lo = a.x + (b.x - a.x) * t0;
                    hi = a.x + (b.x - a.x) * t1;
                }
                let start = lo.min(hi).floor() as i64;
                let end = (lo.max(hi).ceil() as i64).max(start + 1);
                spans.push((start, end));
            }
            crossings.sort_by(f64::total_cmp);
            for pair in crossings.chunks_exact(2) {
                spans.push((pair[0].floor() as i64, pair[1].ceil() as i64));
            }
            spans.sort_unstable();
            let mut runs: Vec<(i64, i64)> = Vec::new();
            for (start, end) in spans {
                if end <= start {
                    continue;
                }
                if let Some(last) = runs.last_mut().filter(|r| start <= r.1) {
                    last.1 = last.1.max(end);
                } else {
                    runs.push((start, end));
                }
            }
            let mut current = BTreeMap::new();
            for (start, end) in runs {
                let key = (start, end - start);
                let mut rect = active.remove(&key).unwrap_or(TileRect {
                    col: start,
                    row,
                    width: end - start,
                    height: 0,
                });
                rect.height += 1;
                current.insert(key, rect);
            }
            result.extend(active.into_values());
            active = current;
        }
        result.extend(active.into_values());
        result
            .into_iter()
            .map(|r| self.split(r))
            .collect::<Result<Vec<_>, _>>()
            .map(|v| v.into_iter().flatten().collect())
    }

    pub fn geometry(self, rects: &[TileRect]) -> Geometry {
        let pieces: Vec<_> = rects
            .iter()
            .map(|r| {
                Rect::new(
                    Coord {
                        x: r.col as f64,
                        y: r.row as f64,
                    },
                    Coord {
                        x: (r.col + r.width) as f64,
                        y: (r.row + r.height) as f64,
                    },
                )
                .to_polygon()
            })
            .collect();
        let polygons = unary_union(&pieces).map_coords(|c| {
            let p = self.lonlat(c.x, c.y);
            Coord { x: p[0], y: p[1] }
        });
        let coordinates: Vec<_> = polygons
            .0
            .into_iter()
            .map(|p| {
                std::iter::once(p.exterior())
                    .chain(p.interiors())
                    .map(|r| r.0.iter().map(|c| [c.x, c.y]).collect())
                    .collect()
            })
            .collect();
        Geometry::MultiPolygon(coordinates)
    }
}

pub(super) fn rectangle(x0: f64, y0: f64, x1: f64, y1: f64) -> Polygon<f64> {
    Rect::new(Coord { x: x0, y: y0 }, Coord { x: x1, y: y1 }).to_polygon()
}

/// Ground dimensions in a local Lambert equal-area plane. Sample curved edges
/// before splitting at the dateline, including rectangles enclosing a pole.
pub(super) fn oriented_rectangle(
    anchor: [f64; 2],
    area: f64,
    aspect: f64,
    bearing: f64,
) -> Result<Geometry, String> {
    let (width, height) = ((area * aspect).sqrt(), (area / aspect).sqrt());
    if width.hypot(height) >= 4.0 * RADIUS {
        return Err("Rectangle exceeds its local equal-area projection".into());
    }
    let (slon, clon) = anchor[0].to_radians().sin_cos();
    let (slat, clat) = anchor[1].to_radians().sin_cos();
    let center = [clat * clon, clat * slon, slat];
    let east = [-slon, clon, 0.0];
    let north = [-slat * clon, -slat * slon, clat];
    let (s, c) = bearing.sin_cos();
    let geographic = |p: [f64; 2]| {
        let (x, y) = (p[0] * c + p[1] * s, -p[0] * s + p[1] * c);
        let squared = (x * x + y * y) / RADIUS.powi(2);
        let xyz: [f64; 3] = std::array::from_fn(|i| {
            (1.0 - squared / 2.0) * center[i]
                + (1.0 - squared / 4.0).max(0.0).sqrt() * (x * east[i] + y * north[i]) / RADIUS
        });
        [
            xyz[1].atan2(xyz[0]).to_degrees(),
            xyz[2].clamp(-1.0, 1.0).asin().to_degrees(),
        ]
    };
    let corners = [
        [-width / 2.0, -height / 2.0],
        [width / 2.0, -height / 2.0],
        [width / 2.0, height / 2.0],
        [-width / 2.0, height / 2.0],
    ];
    let mut points = Vec::new();
    for i in 0..4 {
        let (a, b) = (corners[i], corners[(i + 1) % 4]);
        let steps = ((b[0] - a[0]).hypot(b[1] - a[1]) / 5.0).ceil().max(1.0) as usize;
        for j in 0..steps {
            let t = j as f64 / steps as f64;
            points.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
        }
    }
    points.push(points[0]);
    for _ in 0..12 {
        let mut refined = Vec::new();
        let mut changed = false;
        for pair in points.windows(2) {
            refined.push(pair[0]);
            let delta =
                (geographic(pair[1])[0] - geographic(pair[0])[0] + 180.0).rem_euclid(360.0) - 180.0;
            if delta.abs() > 1.0 {
                refined.push([
                    (pair[0][0] + pair[1][0]) / 2.0,
                    (pair[0][1] + pair[1][1]) / 2.0,
                ]);
                changed = true;
            }
        }
        refined.push(points[0]);
        points = refined;
        if !changed {
            break;
        }
    }
    let mut ring: Vec<Coord<f64>> = Vec::with_capacity(points.len() + 3);
    for p in points {
        let [mut x, y] = geographic(p);
        if let Some(previous) = ring.last() {
            x = previous.x + (x - previous.x + 180.0).rem_euclid(360.0) - 180.0;
        }
        ring.push(Coord { x, y });
    }
    let first = ring[0];
    let last = *ring.last().unwrap();
    if (last.x - first.x).abs() > 180.0 {
        let pole = if anchor[1] >= 0.0 { 90.0 } else { -90.0 };
        ring.extend([
            Coord { x: last.x, y: pole },
            Coord {
                x: first.x,
                y: pole,
            },
            first,
        ]);
    } else {
        *ring.last_mut().unwrap() = first;
    }
    let polygon = Polygon::new(LineString(ring), vec![]);
    let bounds = polygon
        .bounding_rect()
        .ok_or("Rectangle has no geographic area")?;
    let mut parts = Vec::new();
    let lo = ((bounds.min().x + 180.0) / 360.0).floor() as i32;
    let hi = ((bounds.max().x + 180.0) / 360.0).floor() as i32;
    for turn in lo..=hi {
        let offset = f64::from(turn) * 360.0;
        let clipped =
            polygon.intersection(&rectangle(-180.0 + offset, -90.0, 180.0 + offset, 90.0));
        parts.extend(
            clipped
                .map_coords(|p| Coord {
                    x: (p.x - offset).clamp(-180.0, 180.0),
                    y: p.y,
                })
                .0,
        );
    }
    if parts.is_empty() {
        return Err("Rectangle has no geographic area".into());
    }
    // Wrapped polar pieces can meet along the original start meridian.
    let polygons = unary_union(&parts);
    Ok(Geometry::MultiPolygon(
        polygons
            .0
            .iter()
            .map(|p| {
                std::iter::once(p.exterior())
                    .chain(p.interiors())
                    .map(|r| {
                        let mut points = Vec::new();
                        for edge in r.0.windows(2) {
                            // A polar cap's closing edge can span a whole turn.
                            // Explicit intermediate meridians keep the capture
                            // engine's split-at-dateline contract unambiguous.
                            let steps =
                                ((edge[1].x - edge[0].x).abs() / 180.0).ceil().max(1.0) as usize;
                            for i in 0..steps {
                                let t = i as f64 / steps as f64;
                                points.push([
                                    (edge[0].x + t * (edge[1].x - edge[0].x)).clamp(-180.0, 180.0),
                                    (edge[0].y + t * (edge[1].y - edge[0].y)).clamp(-90.0, 90.0),
                                ]);
                            }
                        }
                        points.push(points[0]);
                        points
                    })
                    .collect()
            })
            .collect(),
    ))
}

pub(super) fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    let p = a[1].to_radians();
    let q = b[1].to_radians();
    let h = ((q - p) / 2.0).sin().powi(2)
        + p.cos() * q.cos() * ((b[0] - a[0]).to_radians() / 2.0).sin().powi(2);
    2.0 * RADIUS * h.clamp(0.0, 1.0).sqrt().asin()
}

// Spherical azimuthal equidistant coordinates in km, matching the AOI tile sphere.
pub(super) fn local(p: [f64; 2], anchor: [f64; 2]) -> Coord<f64> {
    let (phi, origin) = (p[1].to_radians(), anchor[1].to_radians());
    let d = (p[0] - anchor[0]).to_radians();
    let c = (origin.sin() * phi.sin() + origin.cos() * phi.cos() * d.cos())
        .clamp(-1.0, 1.0)
        .acos();
    let k = if c.abs() < 1e-10 { 1.0 } else { c / c.sin() };
    Coord {
        x: RADIUS * k * phi.cos() * d.sin(),
        y: RADIUS * k * (origin.cos() * phi.sin() - origin.sin() * phi.cos() * d.cos()),
    }
}

pub(super) fn geographic(p: Coord<f64>, anchor: [f64; 2]) -> [f64; 2] {
    let rho = p.x.hypot(p.y);
    if rho < 1e-10 {
        return anchor;
    }
    let c = rho / RADIUS;
    let phi = anchor[1].to_radians();
    let lat = (c.cos() * phi.sin() + p.y * c.sin() * phi.cos() / rho)
        .clamp(-1.0, 1.0)
        .asin();
    let lon = anchor[0].to_radians()
        + (p.x * c.sin()).atan2(rho * phi.cos() * c.cos() - p.y * phi.sin() * c.sin());
    [lon.to_degrees(), lat.to_degrees()]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ground_rectangles_preserve_area_at_seams_and_poles() {
        use geo::{Contains, Point, Validation};
        for anchor in [
            [0.0, 0.0],
            [12.0, 60.0],
            [120.0, 80.0],
            [179.99, 45.0],
            [-179.99, -45.0],
            [30.0, 89.99],
            [30.0, -89.99],
        ] {
            for angle in [0.0_f64, 23.0, 90.0, 145.0] {
                let Geometry::MultiPolygon(parts) =
                    oriented_rectangle(anchor, 100.0, 2.0, angle.to_radians()).unwrap()
                else {
                    panic!()
                };
                let polygons = MultiPolygon(
                    parts
                        .iter()
                        .map(|p| {
                            let lines: Vec<LineString<f64>> = p
                                .iter()
                                .map(|r| r.iter().map(|p| (p[0], p[1])).collect())
                                .collect();
                            Polygon::new(lines[0].clone(), lines[1..].to_vec())
                        })
                        .collect(),
                );
                assert!(
                    polygons.is_valid(),
                    "{anchor:?}, {angle}: {:?}",
                    polygons.validation_errors()
                );
                assert!(
                    polygons.contains(&Point::new(anchor[0], anchor[1])),
                    "{anchor:?}, {angle}"
                );
                // Integrate the GeoJSON's straight lon/lat edges on the same sphere.
                let area: f64 = parts
                    .iter()
                    .map(|p| {
                        p.iter()
                            .map(|ring| {
                                ring.windows(2)
                                    .map(|w| {
                                        let a = w[0][1].to_radians();
                                        let b = w[1][1].to_radians();
                                        let half = (b - a) / 2.0;
                                        let sinc = if half.abs() < 1e-12 {
                                            1.0
                                        } else {
                                            half.sin() / half
                                        };
                                        (w[1][0] - w[0][0]).to_radians()
                                            * ((a + b) / 2.0).sin()
                                            * sinc
                                    })
                                    .sum::<f64>()
                            })
                            .sum::<f64>()
                            .abs()
                            * RADIUS.powi(2)
                    })
                    .sum();
                assert!((area - 100.0).abs() < 0.02, "{anchor:?}, {angle}: {area}");
                assert!(parts
                    .iter()
                    .flatten()
                    .flatten()
                    .all(|p| p[0].abs() <= 180.0 && p[1].abs() <= 90.0));
                assert!(parts
                    .iter()
                    .flatten()
                    .all(|r| r.windows(2).all(|w| (w[1][0] - w[0][0]).abs() <= 180.0)));
            }
        }
    }
    #[test]
    fn rectangles_wrap_without_losing_tiles() {
        let grid = Grid::new(0.1);
        let rects = grid.rectangle([180.0, 89.9], 10000, 1.0).unwrap();
        assert_eq!(rects.len(), 2);
        assert_eq!(rects.iter().map(|r| r.width * r.height).sum::<i64>(), 10000);
        assert!(rects
            .iter()
            .all(|r| r.col >= 0 && r.col + r.width <= grid.columns && r.row >= 0));
    }
    #[test]
    fn projection_round_trips_at_seam_and_high_latitude() {
        for anchor in [[179.5, 70.0], [-120.0, -60.0], [0.0, 0.0]] {
            let p = [anchor[0] + 2.0, anchor[1] + 1.0];
            let actual = geographic(local(p, anchor), anchor);
            assert!(distance(p, actual) < 1e-7);
        }
    }
    #[test]
    fn cover_is_compact_and_preserves_holes_and_thin_lines() {
        let grid = Grid {
            columns: 360,
            rows: 180,
        };
        let outer = rectangle(10.2, 10.2, 20.8, 20.8);
        let polygon = Polygon::new(
            outer.exterior().clone(),
            vec![rectangle(13.0, 13.0, 18.0, 18.0).exterior().clone()],
        );
        let rects = grid.cover(&MultiPolygon(vec![polygon])).unwrap();
        assert!(!rects
            .iter()
            .any(|r| r.col <= 15 && r.col + r.width > 15 && r.row <= 15 && r.row + r.height > 15));
        assert!(rects.len() < 20);
        assert!(!grid
            .cover(&MultiPolygon(vec![rectangle(1.01, 1.0, 1.02, 5.0)]))
            .unwrap()
            .is_empty());
        assert!(grid
            .cover(&MultiPolygon(vec![rectangle(0.0, 0.0, 400.0, 1.0)]))
            .is_err());
    }
}
