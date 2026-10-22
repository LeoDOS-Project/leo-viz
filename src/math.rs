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

#[cfg(test)]
mod tests {
    use super::*;

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
