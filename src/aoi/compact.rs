//! Bound stored geometry before triangulation, LOD construction and serialization.
//! Remove low-area boundary vertices with a spatially indexed topology check.
use super::{
    geometry::RADIUS,
    preview::{Geometry, Record},
};
use geo::{
    Area, BoundingRect, Coord, Intersects, LineString, MultiPolygon, Point, Polygon, Triangle,
    Validation,
};
use rstar::{RTree, RTreeObject, AABB};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::BinaryHeap};

pub(super) const EDGE_LIMIT: usize = 100;
type Polygons = Vec<Vec<Vec<[f64; 2]>>>;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct Reduction {
    pub original_edges: usize,
    pub original_area_km2: f64,
    pub removed_holes: usize,
    pub removed_parts: usize,
}

pub(super) fn edges(polygons: &[Vec<Vec<[f64; 2]>>]) -> usize {
    polygons
        .iter()
        .flatten()
        .map(|r| r.len().saturating_sub(1))
        .sum()
}

fn ring_area(ring: &[[f64; 2]]) -> f64 {
    signed_area(ring).abs()
}
fn segment_area(a: [f64; 2], b: [f64; 2]) -> f64 {
    // Exact integral along the stored straight lon/lat segment on our sphere.
    let half = (b[1] - a[1]).to_radians() * 0.5;
    let sinc = if half.abs() < 1e-12 {
        1.0
    } else {
        half.sin() / half
    };
    (b[0] - a[0]).to_radians() * ((a[1] + b[1]) * 0.5).to_radians().sin() * sinc * RADIUS.powi(2)
}
fn signed_area(ring: &[[f64; 2]]) -> f64 {
    ring.windows(2).map(|w| segment_area(w[0], w[1])).sum()
}
pub(super) fn area(polygons: &[Vec<Vec<[f64; 2]>>]) -> f64 {
    polygons
        .iter()
        .map(|p| ring_area(&p[0]) - p[1..].iter().map(|r| ring_area(r)).sum::<f64>())
        .sum()
}

#[derive(Clone, PartialEq)]
struct Site {
    point: [f64; 2],
    node: usize,
}
impl RTreeObject for Site {
    type Envelope = AABB<[f64; 2]>;
    fn envelope(&self) -> Self::Envelope {
        AABB::from_point(self.point)
    }
}
struct Node {
    point: Coord<f64>,
    prev: usize,
    next: usize,
    ring: usize,
    alive: bool,
    fixed: bool,
    version: usize,
}
struct Ring {
    head: usize,
    count: usize,
    part: usize,
    hole: bool,
    alive: bool,
    area: f64,
}
#[derive(PartialEq)]
struct Candidate {
    area: f64,
    node: usize,
    version: usize,
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .area
            .total_cmp(&self.area)
            .then_with(|| other.node.cmp(&self.node))
            .then_with(|| other.version.cmp(&self.version))
    }
}

struct Simplifier {
    nodes: Vec<Node>,
    rings: Vec<Ring>,
    tree: RTree<Site>,
    heap: BinaryHeap<Candidate>,
    remaining: usize,
    removed_holes: usize,
    removed_parts: usize,
}
impl Simplifier {
    fn new(polygons: &[Vec<Vec<[f64; 2]>>], anchor: [f64; 2]) -> Result<Self, String> {
        let mut s = Self {
            nodes: Vec::new(),
            rings: Vec::new(),
            tree: RTree::new(),
            heap: BinaryHeap::new(),
            remaining: 0,
            removed_holes: 0,
            removed_parts: 0,
        };
        for (part, polygon) in polygons.iter().enumerate() {
            for (hole, ring) in polygon.iter().enumerate() {
                if ring.len() < 4
                    || ring.first() != ring.last()
                    || ring.iter().any(|p| {
                        !p[0].is_finite()
                            || !p[1].is_finite()
                            || p[0].abs() > 180.0
                            || p[1].abs() > 90.0
                    })
                {
                    return Err("Cannot simplify an invalid AOI boundary".into());
                }
                let head = s.nodes.len();
                let count = ring.len() - 1;
                let id = s.rings.len();
                for i in 0..count {
                    let p = ring[i];
                    let a = ring[(i + count - 1) % count];
                    let c = ring[(i + 1) % count];
                    // Preserve endpoints of dateline/polar cuts. Collinear points
                    // within a cut remain removable, without opening seam gaps.
                    let fixed = (p[0].abs() == 180.0 && (a[0] != p[0] || c[0] != p[0]))
                        || (p[1].abs() == 90.0 && (a[1] != p[1] || c[1] != p[1]));
                    s.nodes.push(Node {
                        point: Coord { x: p[0], y: p[1] },
                        prev: head + (i + count - 1) % count,
                        next: head + (i + 1) % count,
                        ring: id,
                        alive: true,
                        fixed,
                        version: 0,
                    });
                }
                s.rings.push(Ring {
                    head,
                    count,
                    part,
                    hole: hole != 0,
                    alive: true,
                    area: signed_area(ring),
                });
            }
        }
        s.remaining = s.nodes.len();
        let mut sites: Vec<_> = s
            .nodes
            .iter()
            .enumerate()
            .map(|(node, n)| Site {
                point: [n.point.x, n.point.y],
                node,
            })
            .collect();
        // Keep the sample location on the same side of every retained boundary.
        sites.push(Site {
            point: anchor,
            node: usize::MAX,
        });
        s.tree = RTree::bulk_load(sites);
        Ok(s)
    }
    fn enqueue(&mut self, i: usize) {
        let b = &self.nodes[i];
        if !b.alive || b.fixed || self.rings[b.ring].count <= 3 {
            return;
        }
        let (a, c) = (self.nodes[b.prev].point, self.nodes[b.next].point);
        let triangle = Triangle::new(a, b.point, c);
        // Local ground area for ordering; topology is checked in stored coordinates.
        let area =
            triangle.unsigned_area() * ((a.y + b.point.y + c.y) / 3.0).to_radians().cos().abs();
        self.heap.push(Candidate {
            area,
            node: i,
            version: b.version,
        });
    }
    fn refill(&mut self) {
        self.heap.clear();
        for i in 0..self.nodes.len() {
            self.enqueue(i);
        }
    }
    fn removable(&self, i: usize) -> bool {
        let b = &self.nodes[i];
        let (a, c) = (self.nodes[b.prev].point, self.nodes[b.next].point);
        if (a.x - c.x).abs() > 180.0 {
            return false;
        }
        let triangle = Triangle::new(a, b.point, c);
        let bounds = triangle.bounding_rect();
        let envelope = AABB::from_corners(
            [bounds.min().x, bounds.min().y],
            [bounds.max().x, bounds.max().y],
        );
        // On a valid planar polygon, replacing AB/BC by AC is topology-safe
        // when the swept triangle contains no other boundary vertex. This also
        // prevents swallowing holes or joining originally separate components.
        !self
            .tree
            .locate_in_envelope_intersecting(&envelope)
            .any(|site| {
                site.node != i
                    && site.node != b.prev
                    && site.node != b.next
                    && triangle.intersects(&Point::new(site.point[0], site.point[1]))
            })
    }
    fn remove(&mut self, i: usize) {
        let b = &self.nodes[i];
        let (prev, next, ring) = (b.prev, b.next, b.ring);
        let point = |n: &Node| [n.point.x, n.point.y];
        let (a, b, c) = (point(&self.nodes[prev]), point(b), point(&self.nodes[next]));
        self.rings[ring].area += segment_area(a, c) - segment_area(a, b) - segment_area(b, c);
        self.tree.remove(&Site { point: b, node: i });
        self.nodes[i].alive = false;
        self.nodes[prev].next = next;
        self.nodes[next].prev = prev;
        self.rings[ring].count -= 1;
        if self.rings[ring].head == i {
            self.rings[ring].head = next;
        }
        self.remaining -= 1;
        for n in [prev, next] {
            self.nodes[n].version += 1;
            self.enqueue(n);
        }
    }
    fn points(&self, id: usize) -> Vec<[f64; 2]> {
        let r = &self.rings[id];
        let mut result = Vec::with_capacity(r.count + 1);
        let mut i = r.head;
        loop {
            let n = &self.nodes[i];
            result.push([n.point.x, n.point.y]);
            i = n.next;
            if i == r.head {
                break;
            }
        }
        result.push(result[0]);
        result
    }
    fn smallest_feature(&self) -> Option<(usize, f64, usize)> {
        let parts = self.rings.iter().filter(|r| r.alive && !r.hole).count();
        self.rings
            .iter()
            .enumerate()
            .filter(|(_, r)| r.alive && (r.hole || parts > 1))
            .map(|(id, r)| {
                let mut area = r.area.abs();
                let mut edges = r.count;
                if !r.hole {
                    for h in self
                        .rings
                        .iter()
                        .filter(|h| h.alive && h.hole && h.part == r.part)
                    {
                        area -= h.area.abs();
                        edges += h.count;
                    }
                }
                (id, area.max(0.0), edges)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)))
    }
    fn discard_smallest_feature(&mut self) -> bool {
        let Some((id, _, _)) = self.smallest_feature() else {
            return false;
        };
        let (part, hole, head) = (
            self.rings[id].part,
            self.rings[id].hole,
            self.rings[id].head,
        );
        for r in &mut self.rings {
            if !r.alive || if hole { r.head != head } else { r.part != part } {
                continue;
            }
            let mut i = r.head;
            loop {
                let n = &mut self.nodes[i];
                n.alive = false;
                self.tree.remove(&Site {
                    point: [n.point.x, n.point.y],
                    node: i,
                });
                i = n.next;
                if i == r.head {
                    break;
                }
            }
            self.remaining -= r.count;
            r.alive = false;
            if r.hole {
                self.removed_holes += 1;
            }
        }
        if !hole {
            self.removed_parts += 1;
        }
        true
    }
    fn simplify(mut self, limit: usize) -> Result<(Polygons, usize, usize), String> {
        // Drop unavoidable excess features before spending the vertex budget.
        // Otherwise every ring gets reduced to a triangle before the smallest
        // ones are discarded, leaving no detail available to restore elsewhere.
        while self
            .rings
            .iter()
            .filter(|r| r.alive)
            .map(|r| {
                let mut fixed = 0;
                let mut i = r.head;
                loop {
                    fixed += usize::from(self.nodes[i].fixed);
                    i = self.nodes[i].next;
                    if i == r.head {
                        break;
                    }
                }
                fixed.max(3)
            })
            .sum::<usize>()
            > limit
        {
            if !self.discard_smallest_feature() {
                return Err(format!(
                    "Cannot fit AOI within {limit} edges while preserving its dateline cuts"
                ));
            }
        }
        self.refill();
        let mut last_refill = self.remaining;
        while self.remaining > limit {
            if let Some(candidate) = self.heap.pop() {
                let n = &self.nodes[candidate.node];
                if !n.alive || n.version != candidate.version || self.rings[n.ring].count <= 3 {
                    continue;
                }
                if self.removable(candidate.node) {
                    // Spend the remaining edge budget where it preserves the
                    // most area. A tiny hole can cost less to remove entirely
                    // than a corner of a much larger retained boundary.
                    let drop_feature = self.smallest_feature().is_some_and(|(_, area, edges)| {
                        area / (edges.min(self.remaining - limit) as f64)
                            < candidate.area * RADIUS.to_radians().powi(2)
                    });
                    if drop_feature {
                        self.discard_smallest_feature();
                        self.refill();
                        last_refill = self.remaining;
                    } else {
                        self.remove(candidate.node);
                    }
                }
            } else {
                // Non-neighbor removals can unblock a previously rejected ear.
                if self.remaining == last_refill && !self.discard_smallest_feature() {
                    return Err(format!(
                        "Cannot fit AOI within {limit} edges while preserving its dateline cuts"
                    ));
                }
                last_refill = self.remaining;
                self.refill();
            }
        }
        let mut result = Vec::new();
        for (id, r) in self
            .rings
            .iter()
            .enumerate()
            .filter(|(_, r)| r.alive && !r.hole)
        {
            let mut polygon = vec![self.points(id)];
            polygon.extend(
                self.rings
                    .iter()
                    .enumerate()
                    .filter(|(_, h)| h.alive && h.hole && h.part == r.part)
                    .map(|(id, _)| self.points(id)),
            );
            result.push(polygon);
        }
        let shape = MultiPolygon(
            result
                .iter()
                .map(|p| {
                    let ring = |r: &Vec<[f64; 2]>| {
                        LineString::from(r.iter().map(|v| (v[0], v[1])).collect::<Vec<_>>())
                    };
                    Polygon::new(ring(&p[0]), p[1..].iter().map(ring).collect())
                })
                .collect(),
        );
        if !shape.is_valid() {
            return Err("Simplified AOI has invalid topology".into());
        }
        Ok((result, self.removed_holes, self.removed_parts))
    }
}

pub(super) fn record(mut record: Record) -> Result<Record, String> {
    if !["forest", "river", "coast"].contains(&record.source_kind.as_str()) {
        return Ok(record);
    }
    let polygons = match &record.geometry {
        Geometry::Polygon(p) => std::slice::from_ref(p),
        Geometry::MultiPolygon(p) => p.as_slice(),
    };
    let original_edges = edges(polygons);
    if original_edges <= EDGE_LIMIT {
        return Ok(record);
    }
    let (polygons, holes, parts) =
        Simplifier::new(polygons, [record.anchor_lon, record.anchor_lat])?.simplify(EDGE_LIMIT)?;
    let new_area = area(&polygons);
    if !new_area.is_finite() || new_area <= 0.0 {
        return Err("Simplification produced an empty AOI".into());
    }
    let reduction = record.reduction.get_or_insert(Reduction {
        original_edges,
        original_area_km2: record.area_km2,
        removed_holes: 0,
        removed_parts: 0,
    });
    reduction.removed_holes += holes;
    reduction.removed_parts += parts;
    record.area_km2 = new_area;
    record.geometry = Geometry::MultiPolygon(polygons);
    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn circle(x: f64, y: f64, radius: f64, n: usize) -> Vec<[f64; 2]> {
        let mut r: Vec<_> = (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / n as f64;
                [x + radius * a.cos(), y + radius * a.sin()]
            })
            .collect();
        r.push(r[0]);
        r
    }
    fn sample(polygons: Polygons) -> Record {
        Record {
            id: "forest-1".into(),
            source_kind: "forest".into(),
            anchor_lon: 0.0,
            anchor_lat: 0.0,
            area_km2: area(&polygons),
            geometry: Geometry::MultiPolygon(polygons),
            reduction: None,
            path_length_km: None,
            path_truncated: false,
        }
    }
    fn polygons(record: &Record) -> &Polygons {
        let Geometry::MultiPolygon(p) = &record.geometry else {
            panic!()
        };
        p
    }
    #[test]
    fn bounded_geometry_preserves_holes_area_identity_and_determinism() {
        let p = vec![vec![
            circle(0.0, 0.0, 2.0, 1500),
            circle(0.5, 0.0, 0.2, 200),
        ]];
        let a = record(sample(p.clone())).unwrap();
        let b = record(sample(p)).unwrap();
        assert_eq!(edges(polygons(&a)), 100);
        assert_eq!(polygons(&a)[0].len(), 2);
        assert_eq!(
            serde_json::to_vec(&a).unwrap(),
            serde_json::to_vec(&b).unwrap()
        );
        assert_eq!(a.id, "forest-1");
        assert!((a.area_km2 / a.reduction.as_ref().unwrap().original_area_km2 - 1.0).abs() < 0.01);
        let bytes = serde_json::to_vec(&a).unwrap();
        assert_eq!(
            bytes,
            serde_json::to_vec(&record(a).unwrap()).unwrap(),
            "already capped geometry is unchanged"
        );
    }
    #[test]
    fn excess_holes_and_separate_parts_are_removed_to_meet_the_hard_cap() {
        let mut p = vec![circle(0.0, 0.0, 10.0, 200)];
        for y in 0..7 {
            for x in 0..7 {
                p.push(circle(x as f64 - 3.0, y as f64 - 3.0, 0.1, 12));
            }
        }
        let a = record(sample(vec![p])).unwrap();
        assert!(edges(polygons(&a)) <= 100);
        assert!(a.reduction.unwrap().removed_holes > 0);
        let p: Vec<_> = (0..40)
            .map(|i| vec![circle(i as f64 * 0.5, 0.0, 0.1, 20)])
            .collect();
        let a = record(sample(p)).unwrap();
        assert!(edges(polygons(&a)) <= 100);
        assert!(a.reduction.unwrap().removed_parts > 0);
    }
    #[test]
    fn tiny_clearings_do_not_consume_the_budget_for_larger_features() {
        let mut p = vec![circle(0.0, 0.0, 10.0, 200), circle(3.0, 0.0, 2.0, 100)];
        for y in 0..6 {
            for x in 0..5 {
                p.push(circle(x as f64 - 6.0, y as f64 - 3.0, 0.1, 12));
            }
        }
        let a = record(sample(vec![p])).unwrap();
        assert!(edges(polygons(&a)) <= EDGE_LIMIT);
        assert!(
            polygons(&a)[0][1..]
                .iter()
                .any(|r| r.iter().any(|v| v[0] > 0.0)),
            "the large clearing must survive"
        );
        let r = a.reduction.as_ref().unwrap();
        assert!(r.removed_holes > 0);
        assert!(
            (a.area_km2 / r.original_area_km2 - 1.0).abs() < 0.01,
            "tiny holes must not force the exterior or large clearing into triangles"
        );
    }
    #[test]
    fn seam_endpoints_and_narrow_corridors_survive_simplification() {
        let mut ring: Vec<_> = (0..1000)
            .map(|i| {
                let x = i as f64 / 100.0;
                [x, (x * 3.0).sin() + 0.02]
            })
            .collect();
        ring.extend((0..1000).rev().map(|i| {
            let x = i as f64 / 100.0;
            [x, (x * 3.0).sin() - 0.02]
        }));
        ring.push(ring[0]);
        let mut input = sample(vec![vec![ring]]);
        input.source_kind = "river".into();
        input.path_length_km = Some(123.0);
        input.path_truncated = true;
        let a = record(input).unwrap();
        assert!(edges(polygons(&a)) <= 100);
        assert_eq!(a.path_length_km, Some(123.0));
        assert!(a.path_truncated);
        let mut pieces = Vec::new();
        for sign in [-1.0_f64, 1.0] {
            let mut r: Vec<_> = (0..300)
                .map(|i| {
                    let y = -1.0 + 2.0 * i as f64 / 299.0;
                    [sign * (179.0 + 0.1 * (y * 20.0).sin()), y]
                })
                .collect();
            r.extend([[sign * 180.0, 1.0], [sign * 180.0, -1.0], r[0]]);
            pieces.push(vec![r]);
        }
        let a = record(sample(pieces)).unwrap();
        assert!(edges(polygons(&a)) <= 100);
        assert_eq!(polygons(&a).len(), 2);
        for p in polygons(&a) {
            for y in [-1.0, 1.0] {
                assert!(p[0].iter().any(|v| v[0].abs() == 180.0 && v[1] == y));
            }
        }
    }
    #[test]
    fn gdp_and_small_environmental_shapes_are_byte_for_byte_unchanged() {
        for (kind, count) in [("gdp", 200), ("forest", 20)] {
            let mut a = sample(vec![vec![circle(0.0, 0.0, 1.0, count)]]);
            a.source_kind = kind.into();
            let before = serde_json::to_vec(&a).unwrap();
            assert_eq!(before, serde_json::to_vec(&record(a).unwrap()).unwrap());
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "read-only benchmark; set LEO_AOI_BENCH_DATASET to a saved AOI file"]
    fn benchmark_saved_dataset() {
        use std::io::Read;
        let path = std::env::var("LEO_AOI_BENCH_DATASET").expect("Set LEO_AOI_BENCH_DATASET");
        let mut json = Vec::new();
        flate2::read::GzDecoder::new(std::fs::File::open(path).unwrap())
            .read_to_end(&mut json)
            .unwrap();
        let doc: super::super::preview::Document = serde_json::from_slice(&json).unwrap();
        drop(json);
        let started = std::time::Instant::now();
        let mut totals =
            std::collections::BTreeMap::<String, (usize, usize, usize, usize, usize, f64)>::new();
        let mut outliers = std::collections::BTreeMap::<String, Vec<(f64, String)>>::new();
        let source = std::env::var("LEO_AOI_BENCH_SOURCE").ok();
        for a in doc.aois {
            if source.as_ref().is_some_and(|kind| kind != &a.source_kind) {
                continue;
            }
            let before = match &a.geometry {
                Geometry::Polygon(p) => edges(std::slice::from_ref(p)),
                Geometry::MultiPolygon(p) => edges(p),
            };
            let measured_before = match &a.geometry {
                Geometry::Polygon(p) => area(std::slice::from_ref(p)),
                Geometry::MultiPolygon(p) => area(p),
            };
            let a = record(a).unwrap();
            let after = match &a.geometry {
                Geometry::Polygon(p) => edges(std::slice::from_ref(p)),
                Geometry::MultiPolygon(p) => edges(p),
            };
            if a.source_kind != "gdp" {
                assert!(after <= EDGE_LIMIT, "{}", a.id);
            }
            let t = totals.entry(a.source_kind.clone()).or_default();
            t.0 += 1;
            t.1 += before;
            t.2 += after;
            if let Some(r) = a.reduction {
                t.3 += r.removed_holes;
                t.4 += r.removed_parts;
                t.5 = t.5.max((a.area_km2 / r.original_area_km2 - 1.0).abs());
                let change = (a.area_km2 / measured_before - 1.0).abs();
                outliers.entry(a.source_kind).or_default().push((change,
                    format!("{}: {before} -> {after} edges; recorded area {}, measured before {}, after {}; removed {} holes / {} parts", a.id, r.original_area_km2, measured_before, a.area_km2, r.removed_holes, r.removed_parts)));
            }
        }
        eprintln!("Simplification: {:?}; source: (AOIs, original edges, final edges, removed holes, removed parts, max relative area change)\n{totals:#?}", started.elapsed());
        for (kind, mut values) in outliers {
            values.sort_by(|a, b| a.0.total_cmp(&b.0));
            eprintln!(
                "{kind}: relative geometric area change p50={}, p95={}, p99={}",
                values[values.len() / 2].0,
                values[values.len() * 95 / 100].0,
                values[values.len() * 99 / 100].0
            );
            for v in values.iter().rev().take(3) {
                eprintln!("{v:?}");
            }
        }
    }
}
