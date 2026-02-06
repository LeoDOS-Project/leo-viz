//! 3D rotation and coordinate transformations.
//!
//! Matrix operations for camera rotation, lat/lon conversions,
//! and drag-based rotation from mouse input.

use nalgebra::{Matrix3, Vector3};
use std::f64::consts::{PI, FRAC_PI_2};

pub fn rotate_point_matrix(x: f64, y: f64, z: f64, rot: &Matrix3<f64>) -> (f64, f64, f64) {
    let v = rot * Vector3::new(x, y, z);
    (v.x, v.y, v.z)
}

pub fn matrix_to_lat_lon(m: &Matrix3<f64>) -> (f64, f64) {
    let lat = m[(2, 1)].asin().clamp(-FRAC_PI_2, FRAC_PI_2);
    let mut lon = (-m[(0, 2)]).atan2(m[(0, 0)]) - FRAC_PI_2;
    if lon < -PI { lon += 2.0 * PI; }
    if lon > PI { lon -= 2.0 * PI; }
    (lat, lon)
}

pub fn lat_lon_to_matrix(lat: f64, lon: f64) -> Matrix3<f64> {
    let lon = -lon - FRAC_PI_2;
    let (sl, cl) = (lat.sin(), lat.cos());
    let (sn, cn) = (lon.sin(), lon.cos());
    Matrix3::new(
        cn, 0.0, sn,
        sl * sn, cl, -sl * cn,
        -cl * sn, sl, cl * cn,
    )
}

pub fn rotation_from_drag(dx: f64, dy: f64) -> Matrix3<f64> {
    let rot_y = Matrix3::new(
        dx.cos(), 0.0, dx.sin(),
        0.0, 1.0, 0.0,
        -dx.sin(), 0.0, dx.cos(),
    );
    let rot_x = Matrix3::new(
        1.0, 0.0, 0.0,
        0.0, dy.cos(), -dy.sin(),
        0.0, dy.sin(), dy.cos(),
    );
    rot_x * rot_y
}
