//! Shared projection and clipping for ground overlays on the globe.
use eframe::egui::{self, Color32, Mesh, Pos2};
use nalgebra::{Matrix3, Vector3};

pub(crate) type SurfacePoint = (Vector3<f64>, f64);

/// Inverse of project_surface for the front intersection of an orthographic
/// ray with the displayed oblate globe. Longitude follows the viewer's -Z convention.
pub(crate) fn pick_surface(
    x: f64,
    y: f64,
    r: &Matrix3<f64>,
    radius: f64,
    polar_scale: f64,
) -> Option<(f64, f64)> {
    if radius <= 0.0 || polar_scale <= 0.0 {
        return None;
    }
    let inv = r.transpose();
    let mut a = inv * Vector3::new(x / radius, y / radius, 0.0);
    let mut b = inv * Vector3::new(0.0, 0.0, 1.0);
    a.y /= polar_scale;
    b.y /= polar_scale;
    let aa = b.dot(&b);
    let ab = a.dot(&b);
    let cc = a.dot(&a) - 1.0;
    let discriminant = ab * ab - aa * cc;
    if discriminant < 0.0 || !discriminant.is_finite() {
        return None;
    }
    let p = (a + b * ((-ab + discriminant.sqrt()) / aa)).normalize();
    Some((
        p.y.clamp(-1.0, 1.0).asin().to_degrees(),
        -p.z.atan2(p.x).to_degrees(),
    ))
}

pub(crate) fn project_surface(
    p: Vector3<f64>,
    r: &Matrix3<f64>,
    radius: f64,
    polar_scale: f64,
) -> SurfacePoint {
    let y = p.y * polar_scale;
    let x_rot = r.m11 * p.x + r.m12 * y + r.m13 * p.z;
    let y_rot = r.m21 * p.x + r.m22 * y + r.m23 * p.z;
    let z_rot = r.m31 * p.x + r.m32 * y + r.m33 * p.z;
    let normal_z = r.m31 * p.x + r.m32 * p.y / polar_scale + r.m33 * p.z;
    (
        Vector3::new(x_rot * radius, y_rot * radius, z_rot * radius),
        normal_z,
    )
}

pub(crate) fn front_segment(
    a: SurfacePoint,
    b: SurfacePoint,
) -> Option<(Vector3<f64>, Vector3<f64>)> {
    if a.1 < 0.0 && b.1 < 0.0 {
        return None;
    }
    if a.1 >= 0.0 && b.1 >= 0.0 {
        return Some((a.0, b.0));
    }
    let crossing = a.0 + (b.0 - a.0) * (a.1 / (a.1 - b.1));
    Some(if a.1 < 0.0 {
        (crossing, b.0)
    } else {
        (a.0, crossing)
    })
}

/// Clip a short convex surface patch against the visible hemisphere.
pub(crate) fn front_polygon(points: &[SurfacePoint]) -> Vec<Vector3<f64>> {
    let mut result = Vec::with_capacity(points.len() + 2);
    let Some(mut previous) = points.last().copied() else {
        return result;
    };
    for &current in points {
        if (previous.1 < 0.0) != (current.1 < 0.0) {
            result.push(
                previous.0 + (current.0 - previous.0) * (previous.1 / (previous.1 - current.1)),
            );
        }
        if current.1 >= 0.0 {
            result.push(current.0);
        }
        previous = current;
    }
    result
}

pub(crate) fn line_quad(mesh: &mut Mesh, a: Pos2, b: Pos2, color: Color32, width: f32) {
    let d = b - a;
    if d.length_sq() < 0.0001 {
        return;
    }
    let offset = egui::vec2(-d.y, d.x).normalized() * (width / 2.0);
    let first = mesh.vertices.len() as u32;
    for pos in [a + offset, a - offset, b + offset, b - offset] {
        mesh.colored_vertex(pos, color);
    }
    mesh.add_triangle(first, first + 1, first + 2);
    mesh.add_triangle(first + 2, first + 1, first + 3);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn picking_round_trips_rotated_oblate_globe_including_poles_and_dateline() {
        for (lat, lon) in [
            (0.0_f64, 0.0_f64),
            (57.7, 11.97),
            (89.99, 179.99),
            (-89.99, -179.99),
            (0.0, 180.0),
        ] {
            let (s, c) = lat.to_radians().sin_cos();
            let (sl, cl) = lon.to_radians().sin_cos();
            let p = Vector3::new(c * cl, s, -c * sl);
            // Orient this location towards the camera, then add a small tilt.
            let target = Vector3::new(0.2, 0.1, 0.97).normalize();
            let r = nalgebra::Rotation3::rotation_between(&p, &target)
                .unwrap()
                .into_inner();
            for polar_scale in [1.0, 0.996647, 0.8] {
                let (projected, front) = project_surface(p, &r, 6371.0, polar_scale);
                assert!(front > 0.0);
                let (a, b) =
                    pick_surface(projected.x, projected.y, &r, 6371.0, polar_scale).unwrap();
                assert!((a - lat).abs() < 1e-7);
                assert!(((b - lon + 180.0).rem_euclid(360.0) - 180.0).abs() < 1e-7);
            }
        }
        assert!(pick_surface(7000.0, 0.0, &Matrix3::identity(), 6371.0, 1.0).is_none());
        assert!(pick_surface(0.0, 6360.0, &Matrix3::identity(), 6371.0, 0.99).is_none());
    }
}
