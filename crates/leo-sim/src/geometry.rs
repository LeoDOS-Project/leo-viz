//! Geometry uses leo-viz's coordinates: +y north, -z east. Lengths are km.
use std::f64::consts::PI;
pub type Vector = [f64; 3];
pub const EARTH_RADIUS_KM: f64 = 6371.0;
pub const EARTH_EQUATORIAL_RADIUS_KM: f64 = 6378.137;
pub const EARTH_MU: f64 = 398600.4418;
pub const EARTH_J2: f64 = 1.08263e-3;
pub const LIGHT_KM_S: f64 = 299792.458;
pub fn dot(a: Vector, b: Vector) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
pub fn sub(a: Vector, b: Vector) -> Vector {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn norm(a: Vector) -> f64 {
    dot(a, a).sqrt()
}
pub fn scale(a: Vector, s: f64) -> Vector {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub fn distance(a: Vector, b: Vector) -> f64 {
    norm(sub(a, b))
}
pub fn gmst(unix_s: f64) -> f64 {
    let days = (unix_s - 946728000.0) / 86400.0;
    let c = days / 36525.0;
    (280.46061837 + 360.98564736629 * days + 0.000387933 * c * c - c * c * c / 38710000.0)
        .rem_euclid(360.0)
        .to_radians()
}
pub fn ground_position(lat_deg: f64, lon_deg: f64, radius: f64, rotation: f64) -> Vector {
    let lat = lat_deg.to_radians();
    let lon = -lon_deg.to_radians() - rotation;
    [
        radius * lat.cos() * lon.cos(),
        radius * lat.sin(),
        radius * lat.cos() * lon.sin(),
    ]
}
pub fn elevation(ground: Vector, sat: Vector) -> f64 {
    let ray = sub(sat, ground);
    if norm(ray) < 1e-9 {
        return 90.0;
    }
    (dot(ground, ray) / (norm(ground) * norm(ray)))
        .clamp(-1.0, 1.0)
        .asin()
        .to_degrees()
}
pub fn clear_line(a: Vector, b: Vector, radius: f64) -> bool {
    let d = sub(b, a);
    let dd = dot(d, d);
    if dd == 0.0 {
        return false;
    }
    let u = (-dot(a, d) / dd).clamp(0.0, 1.0);
    norm([a[0] + u * d[0], a[1] + u * d[1], a[2] + u * d[2]]) > radius
}
/// Approximate solar direction from mean longitude/anomaly; no daily reset.
pub fn sun_direction(unix_s: f64) -> Vector {
    let d = (unix_s - 946728000.0) / 86400.0;
    let g = (357.529 + 0.98560028 * d).to_radians();
    let l = (280.459 + 0.98564736 * d + 1.915 * g.sin() + 0.020 * (2.0 * g).sin()).to_radians();
    let e = (23.439 - 0.00000036 * d).to_radians();
    [l.cos(), e.sin() * l.sin(), -e.cos() * l.sin()]
}
pub fn eclipsed(pos: Vector, sun: Vector, radius: f64) -> bool {
    let projection = dot(pos, sun);
    projection < 0.0 && dot(pos, pos) - projection * projection < radius * radius
}
pub fn footprint_radius_rad(altitude: f64, full_fov_deg: f64, radius: f64) -> f64 {
    let r = radius + altitude;
    let half = full_fov_deg.to_radians() / 2.0;
    let horizon = (radius / r).acos();
    let v = r / radius * half.sin();
    if v >= 1.0 {
        horizon
    } else {
        (v.asin() - half).min(horizon).max(0.0)
    }
}
pub fn angular_distance(a: Vector, b: Vector) -> f64 {
    (dot(a, b) / (norm(a) * norm(b))).clamp(-1.0, 1.0).acos()
}
pub fn local_solar_time_hours(ground: Vector, sun: Vector) -> f64 {
    let lon = (-ground[2]).atan2(ground[0]);
    let solar_lon = (-sun[2]).atan2(sun[0]);
    (12.0 + (lon - solar_lon) * 12.0 / PI).rem_euclid(24.0)
}
