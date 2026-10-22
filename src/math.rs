//! 3D rotation and coordinate transformations.
//!
//! Matrix operations for camera rotation, lat/lon conversions,
//! and drag-based rotation from mouse input.

use nalgebra::{Matrix3, Vector3};
use std::f64::consts::{FRAC_PI_2, PI};

pub const MAX_CAMERA_ZOOM: f64 = 1_000_000.0;

pub fn rotate_point_matrix(x: f64, y: f64, z: f64, rot: &Matrix3<f64>) -> (f64, f64, f64) {
    let v = rot * Vector3::new(x, y, z);
    (v.x, v.y, v.z)
}

pub fn matrix_to_lat_lon(m: &Matrix3<f64>) -> (f64, f64) {
    let lat = m[(2, 1)].asin().clamp(-FRAC_PI_2, FRAC_PI_2);
    let mut lon = (-m[(0, 2)]).atan2(m[(0, 0)]) - FRAC_PI_2;
    if lon < -PI {
        lon += 2.0 * PI;
    }
    if lon > PI {
        lon -= 2.0 * PI;
    }
    (lat, lon)
}

pub fn lat_lon_to_matrix(lat: f64, lon: f64) -> Matrix3<f64> {
    let lon = -lon - FRAC_PI_2;
    let (sl, cl) = (lat.sin(), lat.cos());
    let (sn, cn) = (lon.sin(), lon.cos());
    Matrix3::new(cn, 0.0, sn, sl * sn, cl, -sl * cn, -cl * sn, sl, cl * cn)
}

pub fn rotation_from_drag(dx: f64, dy: f64) -> Matrix3<f64> {
    let rot_y = Matrix3::new(
        dx.cos(),
        0.0,
        dx.sin(),
        0.0,
        1.0,
        0.0,
        -dx.sin(),
        0.0,
        dx.cos(),
    );
    let rot_x = Matrix3::new(
        1.0,
        0.0,
        0.0,
        0.0,
        dy.cos(),
        -dy.sin(),
        0.0,
        dy.sin(),
        dy.cos(),
    );
    rot_x * rot_y
}

pub fn camera_pan_sensitivity(zoom: f64) -> f64 {
    // Cancel the camera's magnification so screen movement stays consistent near the ground.
    0.001 / zoom.max(1.0)
}

pub fn rotation_for_zoom(
    mut rotation: Matrix3<f64>,
    point: [f64; 2],
    planet_radius: f64,
    old_zoom: f64,
    zoom: f64,
    north_up: bool,
) -> Matrix3<f64> {
    if old_zoom == zoom {
        return rotation;
    }
    let [cx, cy] = point;
    let r_sq = planet_radius * planet_radius;
    if cx * cx + cy * cy > r_sq {
        return rotation;
    }
    let ratio = old_zoom / zoom;
    let tx = cx * ratio;
    let ty = cy * ratio;
    if tx * tx + ty * ty <= r_sq {
        // Preserve the ground point under the pointer as magnification changes.
        let a = Vector3::new(cx, cy, (r_sq - cx * cx - cy * cy).sqrt()).normalize();
        let b = Vector3::new(tx, ty, (r_sq - tx * tx - ty * ty).sqrt()).normalize();
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
                t * x * x + ca,
                t * x * y - sa * z,
                t * x * z + sa * y,
                t * x * y + sa * z,
                t * y * y + ca,
                t * y * z - sa * x,
                t * x * z - sa * y,
                t * y * z + sa * x,
                t * z * z + ca,
            );
            rotation = rot * rotation;
        }
    }

    let center = rotation.transpose() * Vector3::new(0.0, 0.0, 1.0);
    let lat_limit = 85.0_f64.to_radians().sin();
    let clamped_y = center.y.clamp(-lat_limit, lat_limit);
    if (center.y - clamped_y).abs() > 1e-8 {
        let horiz = (center.x * center.x + center.z * center.z).sqrt();
        let new_horiz = (1.0 - clamped_y * clamped_y).sqrt();
        let scale = if horiz > 1e-10 {
            new_horiz / horiz
        } else {
            1.0
        };
        let clamped = Vector3::new(center.x * scale, clamped_y, center.z * scale).normalize();
        let right_raw = Vector3::new(clamped.z, 0.0, -clamped.x);
        let right_len = right_raw.norm();
        if right_len > 0.01 {
            let right = right_raw / right_len;
            let up = clamped.cross(&right);
            let r0 = Matrix3::new(
                right.x, right.y, right.z, up.x, up.y, up.z, clamped.x, clamped.y, clamped.z,
            );
            let up_screen = rotation * Vector3::new(0.0, 1.0, 0.0);
            let bearing = up_screen.x.atan2(up_screen.y);
            let cb = bearing.cos();
            let sb = bearing.sin();
            rotation = Matrix3::new(cb, sb, 0.0, -sb, cb, 0.0, 0.0, 0.0, 1.0) * r0;
        }
    }
    if north_up {
        let north_blend = (zoom.log2() / 4.0).clamp(0.0, 1.0);
        if north_blend > 0.0 {
            let up_screen = rotation * Vector3::new(0.0, 1.0, 0.0);
            let bearing = up_screen.x.atan2(up_screen.y);
            let zoom_octaves = (zoom / old_zoom).ln().abs() / 2.0_f64.ln();
            let decay = (-north_blend * zoom_octaves * 1.5).exp();
            let correction = bearing * (decay - 1.0);
            let ca = correction.cos();
            let sa = correction.sin();
            rotation = Matrix3::new(ca, sa, 0.0, -sa, ca, 0.0, 0.0, 0.0, 1.0) * rotation;
        }
    }
    rotation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_keeps_the_point_under_the_pointer_at_every_altitude() {
        let rotation = lat_lon_to_matrix(30_f64.to_radians(), 25_f64.to_radians());
        let radius = 6371.0;
        for old_zoom in [0.5_f64, 3.0, 100.0, 20_000.0, 500_000.0] {
            let point = [
                radius * 0.3 / old_zoom.max(1.0),
                radius * 0.2 / old_zoom.max(1.0),
            ];
            let surface = Vector3::new(
                point[0],
                point[1],
                (radius * radius - point[0] * point[0] - point[1] * point[1]).sqrt(),
            );
            let world = rotation.transpose() * surface;
            for factor in [0.9, 1.1, 1.0002] {
                let zoom = old_zoom * factor;
                let updated = rotation_for_zoom(rotation, point, radius, old_zoom, zoom, false);
                let projected = updated * world;
                assert!((projected.x * zoom - point[0] * old_zoom).abs() < 1e-5);
                assert!((projected.y * zoom - point[1] * old_zoom).abs() < 1e-5);
                assert!((updated.transpose() * updated - Matrix3::identity()).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn centered_clamped_and_off_globe_zooms_do_not_move_the_camera() {
        let rotation = lat_lon_to_matrix(30_f64.to_radians(), 25_f64.to_radians());
        for (point, zoom) in [([0.0, 0.0], 12.0), ([2.0, 0.0], 12.0), ([0.3, 0.2], 10.0)] {
            assert_eq!(
                rotation_for_zoom(rotation, point, 1.0, 10.0, zoom, false),
                rotation
            );
        }
    }

    #[test]
    fn zoom_preserves_the_existing_north_up_and_polar_limits() {
        let bearing = |rotation: Matrix3<f64>| {
            let up = rotation * Vector3::new(0.0, 1.0, 0.0);
            up.x.atan2(up.y).abs()
        };
        let rotation = rotation_from_drag(0.1, 0.2)
            * lat_lon_to_matrix(30_f64.to_radians(), 25_f64.to_radians());
        let updated = rotation_for_zoom(rotation, [0.0, 0.0], 1.0, 8.0, 16.0, true);
        assert!(bearing(updated) < bearing(rotation));
        let polar = lat_lon_to_matrix(89_f64.to_radians(), 25_f64.to_radians());
        let updated = rotation_for_zoom(polar, [0.0, 0.0], 1.0, 8.0, 16.0, false);
        let center = updated.transpose() * Vector3::new(0.0, 0.0, 1.0);
        assert!(center.y.abs() <= 85_f64.to_radians().sin() + 1e-12);
    }

    #[test]
    fn camera_panning_does_not_accelerate_above_the_overview_altitude() {
        for zoom in [0.01, 0.1, 0.5, 1.0] {
            assert_eq!(camera_pan_sensitivity(zoom), 0.001);
        }
    }

    #[test]
    fn camera_panning_has_consistent_screen_speed_across_altitudes() {
        for altitude_km in [10_000.0, 1000.0, 100.0, 39.0, 10.0, 1.0, 0.5] {
            let zoom: f64 = 10_000.0 / altitude_km;
            assert!((camera_pan_sensitivity(zoom) * zoom - 0.001).abs() < 1e-12);
        }
    }

    #[test]
    fn camera_panning_is_slower_at_the_overview_and_faster_at_low_altitude() {
        for zoom in [100.0_f64, 256.0, 1000.0, 20_000.0] {
            let previous = 0.002 / (1.0 + zoom.ln()) / zoom;
            assert!(camera_pan_sensitivity(zoom) > previous);
        }
        assert!(camera_pan_sensitivity(1.0) < 0.002);
    }
}
