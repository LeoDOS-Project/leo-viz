//! Time and astronomical calculations.
//!
//! Provides Greenwich Mean Sidereal Time (GMST) calculation and
//! planetary rotation angles for accurate Earth-fixed positioning.

use std::f64::consts::PI;
use chrono::{DateTime, Utc};
use crate::celestial::CelestialBody;
use crate::tle::SECONDS_PER_DAY;

pub const DAYS_PER_JULIAN_CENTURY: f64 = 36525.0;
pub const GMST_BASE_DEG: f64 = 280.46061837;
pub const GMST_ROTATION_PER_DAY: f64 = 360.98564736629;
pub const GMST_CORRECTION: f64 = 0.000387933;
pub const SOLAR_DECLINATION_MAX: f64 = -23.45;
pub const DAYS_PER_YEAR: f64 = 365.0;

pub fn greenwich_mean_sidereal_time(timestamp: DateTime<Utc>) -> f64 {
    let j2000 = DateTime::parse_from_rfc3339("2000-01-01T12:00:00Z")
        .unwrap()
        .with_timezone(&Utc);
    let days_since_j2000 = (timestamp - j2000).num_milliseconds() as f64 / (1000.0 * SECONDS_PER_DAY);
    let centuries = days_since_j2000 / DAYS_PER_JULIAN_CENTURY;
    let gmst_degrees = GMST_BASE_DEG
        + GMST_ROTATION_PER_DAY * days_since_j2000
        + GMST_CORRECTION * centuries * centuries
        - centuries * centuries * centuries / 38710000.0;
    let gmst_normalized = gmst_degrees.rem_euclid(360.0);
    gmst_normalized.to_radians()
}

pub fn body_rotation_angle(body: CelestialBody, sim_time_seconds: f64, gmst: f64) -> f64 {
    if body == CelestialBody::Earth {
        gmst
    } else {
        let period_hours = body.rotation_period_hours();
        let period_seconds = period_hours * 3600.0;
        let rotations = sim_time_seconds / period_seconds;
        (rotations * 2.0 * PI).rem_euclid(2.0 * PI)
    }
}
