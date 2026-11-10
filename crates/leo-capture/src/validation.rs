//! Preserve geo's polygon validation rules while indexing the expensive
//! all-pairs edge check for large environmental boundaries, including polygons with holes.
use crate::geometry::Shape;
use geo::{
    coordinate_position::CoordPos, dimensions::Dimensions, Intersects, Line, LineString, Polygon,
    PreparedGeometry, Relate, RemoveRepeatedPoints, Validation,
};
use rstar::{RTree, RTreeObject, AABB};

struct Edge {
    line: Line<f64>,
    index: usize,
}
impl RTreeObject for Edge {
    type Envelope = AABB<[f64; 2]>;
    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners(
            [
                self.line.start.x.min(self.line.end.x),
                self.line.start.y.min(self.line.end.y),
            ],
            [
                self.line.start.x.max(self.line.end.x),
                self.line.start.y.max(self.line.end.y),
            ],
        )
    }
}
pub(crate) fn valid_shape(shape: &Shape) -> bool {
    if shape.0.len() != 1 {
        return shape.is_valid();
    }
    let polygon = &shape.0[0];
    if std::iter::once(polygon.exterior())
        .chain(polygon.interiors())
        .all(|ring| ring.0.len() < 64)
    {
        return shape.is_valid();
    }
    for ring in std::iter::once(polygon.exterior()).chain(polygon.interiors()) {
        if !valid_ring(ring) {
            return false;
        }
    }
    if polygon.interiors().is_empty() {
        return true;
    }
    let exterior = Polygon::new(polygon.exterior().clone(), vec![]);
    let prepared = PreparedGeometry::from(&exterior);
    for (i, ring) in polygon.interiors().iter().enumerate() {
        if ring.0.is_empty() {
            continue;
        }
        let hole = Polygon::new(ring.clone(), vec![]);
        let hole = PreparedGeometry::from(&hole);
        let relation = prepared.relate(&hole);
        if !relation.is_contains()
            || relation.get(CoordPos::OnBoundary, CoordPos::OnBoundary)
                == Dimensions::OneDimensional
        {
            return false;
        }
        for other in &polygon.interiors()[i + 1..] {
            if other.0.is_empty() {
                continue;
            }
            let other = Polygon::new(other.clone(), vec![]);
            let relation = hole.relate(&other);
            if relation.get(CoordPos::Inside, CoordPos::Inside) == Dimensions::TwoDimensional
                || relation.get(CoordPos::OnBoundary, CoordPos::OnBoundary)
                    == Dimensions::OneDimensional
            {
                return false;
            }
        }
    }
    true
}
fn valid_ring(ring: &LineString<f64>) -> bool {
    if ring.0.is_empty() {
        return true;
    }
    if ring.0.iter().any(|c| !c.x.is_finite() || !c.y.is_finite())
        || ring.remove_repeated_points().0.len() < 4
    {
        return false;
    }
    let edges = RTree::bulk_load(
        ring.lines()
            .enumerate()
            .map(|(index, line)| Edge { line, index })
            .collect(),
    );
    for a in edges.iter() {
        for b in edges.locate_in_envelope_intersecting(&a.envelope()) {
            // Same intersection and shared-endpoint rules as geo 0.33's
            // validation::utils::linestring_has_self_intersection. The index
            // only prunes disjoint bounding boxes; it never simplifies rings.
            if a.index < b.index
                && a.line.start != b.line.end
                && a.line.end != b.line.start
                && a.line.intersects(&b.line)
            {
                return false;
            }
        }
    }
    true
}
#[cfg(test)]
mod tests {
    use super::*;
    use geo::{LineString, MultiPolygon, Polygon};
    #[test]
    fn indexed_validation_matches_geo_for_detailed_valid_and_crossing_boundaries() {
        for count in [100, 1000] {
            let mut points: Vec<_> = (0..count)
                .map(|i| {
                    let a = i as f64 / count as f64 * std::f64::consts::TAU;
                    (a.cos(), a.sin())
                })
                .collect();
            points.push(points[0]);
            let shape =
                |p: Vec<(f64, f64)>| MultiPolygon(vec![Polygon::new(LineString::from(p), vec![])]);
            let valid = shape(points.clone());
            assert!(valid_shape(&valid));
            assert_eq!(valid_shape(&valid), valid.is_valid());
            points.swap(10, count / 2);
            let crossed = shape(points);
            assert!(!valid_shape(&crossed));
            assert_eq!(valid_shape(&crossed), crossed.is_valid());
        }
    }
    #[test]
    fn indexed_hole_validation_matches_geo_for_containment_overlap_and_touching() {
        let outer: LineString<_> = (0..=100)
            .map(|i| {
                let a = i as f64 / 100.0 * std::f64::consts::TAU;
                (10.0 * a.cos(), 10.0 * a.sin())
            })
            .collect();
        let square = |x: f64, y: f64| {
            LineString::from(vec![
                (x, y),
                (x + 2.0, y),
                (x + 2.0, y + 2.0),
                (x, y + 2.0),
                (x, y),
            ])
        };
        for holes in [
            vec![square(0.0, 0.0)],
            vec![square(20.0, 0.0)],
            vec![square(0.0, 0.0), square(1.0, 1.0)],
            vec![square(0.0, 0.0), square(2.0, 2.0)],
            vec![square(0.0, 0.0), square(2.0, 0.0)],
        ] {
            let shape = MultiPolygon(vec![Polygon::new(outer.clone(), holes)]);
            assert_eq!(valid_shape(&shape), shape.is_valid());
        }
    }
}
