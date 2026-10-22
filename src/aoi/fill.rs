//! Triangulate once on the worker; preserve holes and clip fills at the globe limb.
use super::preview::unit;
use crate::globe_overlay::{front_polygon, SurfacePoint};
use eframe::egui::{Color32, Mesh, Pos2, Rect};
use nalgebra::Vector3;

pub(super) struct Fill {
    triangles: Vec<[Vector3<f64>; 3]>,
}
impl Fill {
    pub fn new(polygons: &[Vec<Vec<[f64; 2]>>]) -> Result<Self, String> {
        let mut triangles = Vec::new();
        for polygon in polygons {
            let mut points = Vec::new();
            let mut holes = Vec::new();
            for (i, ring) in polygon.iter().enumerate() {
                if i > 0 {
                    holes.push(points.len());
                }
                points.extend_from_slice(&ring[..ring.len() - 1]);
            }
            let coords: Vec<_> = points.iter().flatten().copied().collect();
            let indices = earcutr::earcut(&coords, &holes, 2)
                .map_err(|e| format!("Invalid AOI fill: {e:?}"))?;
            for triangle in indices.chunks_exact(3) {
                let mut stack = vec![(
                    [
                        points[triangle[0]],
                        points[triangle[1]],
                        points[triangle[2]],
                    ],
                    0,
                )];
                while let Some((p, depth)) = stack.pop() {
                    let v = p.map(unit);
                    // Long geographic edges need intermediate surface points,
                    // otherwise the filled triangle cuts across curved parallels.
                    let mut split = None;
                    let mut error = 1e-5;
                    for i in 0..3 {
                        let j = (i + 1) % 3;
                        let mid = [(p[i][0] + p[j][0]) * 0.5, (p[i][1] + p[j][1]) * 0.5];
                        let distance = (unit(mid) - (v[i] + v[j]) * 0.5).norm();
                        if distance > error {
                            error = distance;
                            split = Some((i, j, mid));
                        }
                    }
                    if let Some((i, j, mid)) = split.filter(|_| depth < 10) {
                        let k = (j + 1) % 3;
                        stack.push(([p[i], mid, p[k]], depth + 1));
                        stack.push(([mid, p[j], p[k]], depth + 1));
                    } else if (v[1] - v[0]).cross(&(v[2] - v[0])).norm_squared() > 1e-28 {
                        triangles.push(v);
                    }
                }
            }
        }
        Ok(Self { triangles })
    }
    pub fn paint(
        &self,
        mesh: &mut Mesh,
        color: Color32,
        frame: Rect,
        project: impl Fn(Vector3<f64>) -> SurfacePoint,
        screen: impl Fn(Vector3<f64>) -> Pos2,
    ) {
        for triangle in &self.triangles {
            let projected = triangle.map(&project);
            if projected.iter().all(|p| p.1 < 0.0) {
                continue;
            }
            if projected.iter().all(|p| p.1 >= 0.0) {
                append(mesh, &projected.map(|p| screen(p.0)), color, frame);
            } else {
                let clipped: Vec<_> = front_polygon(&projected).into_iter().map(&screen).collect();
                append(mesh, &clipped, color, frame);
            }
        }
    }
}
fn append(mesh: &mut Mesh, points: &[Pos2], color: Color32, frame: Rect) {
    if points.len() < 3 {
        return;
    }
    let mut bounds = Rect::NOTHING;
    for p in points {
        bounds.extend_with(*p);
    }
    if !frame.intersects(bounds) {
        return;
    }
    let base = mesh.vertices.len() as u32;
    for p in points {
        mesh.colored_vertex(*p, color);
    }
    for i in 1..points.len() - 1 {
        mesh.add_triangle(base, base + i as u32, base + i as u32 + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn triangulation_preserves_holes_and_concavities() {
        let polygons = vec![vec![
            vec![[0., 0.], [4., 0.], [4., 4.], [0., 4.], [0., 0.]],
            vec![[1., 1.], [1., 3.], [3., 3.], [3., 1.], [1., 1.]],
        ]];
        let fill = Fill::new(&polygons).unwrap();
        let mut area = 0.;
        for t in fill.triangles {
            let ll = t.map(|p| [-p.z.atan2(p.x).to_degrees(), p.y.asin().to_degrees()]);
            let c = [
                (ll[0][0] + ll[1][0] + ll[2][0]) / 3.,
                (ll[0][1] + ll[1][1] + ll[2][1]) / 3.,
            ];
            assert!(!(c[0] > 1. && c[0] < 3. && c[1] > 1. && c[1] < 3.));
            area += ((ll[1][0] - ll[0][0]) * (ll[2][1] - ll[0][1])
                - (ll[2][0] - ll[0][0]) * (ll[1][1] - ll[0][1]))
                .abs()
                / 2.;
        }
        assert!((area - 12.).abs() < 1e-9);
        let l_shape = vec![vec![vec![
            [0., 0.],
            [3., 0.],
            [3., 1.],
            [1., 1.],
            [1., 3.],
            [0., 3.],
            [0., 0.],
        ]]];
        for t in Fill::new(&l_shape).unwrap().triangles {
            let lon = t.iter().map(|p| -p.z.atan2(p.x).to_degrees()).sum::<f64>() / 3.;
            let lat = t.iter().map(|p| p.y.asin().to_degrees()).sum::<f64>() / 3.;
            assert!(lon <= 1. + 1e-9 || lat <= 1. + 1e-9);
        }
    }
}
