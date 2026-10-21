//! Time and astronomical calculations.
//!
//! Provides Greenwich Mean Sidereal Time (GMST) calculation and
//! planetary rotation angles for accurate Earth-fixed positioning.

use crate::celestial::CelestialBody;
use chrono::{DateTime, Utc};
use std::f64::consts::PI;

pub const SOLAR_DECLINATION_MAX: f64 = -23.45;
pub const DAYS_PER_YEAR: f64 = 365.0;

pub fn greenwich_mean_sidereal_time(timestamp: DateTime<Utc>) -> f64 {
    leo_sim::geometry::gmst(timestamp.timestamp_millis() as f64 / 1000.0)
}

/// Compute a continuous "day of year" value that advances linearly with sim
/// time, without the January 1 reset and without the 366-day roll of
/// `chrono::Datelike::ordinal()` in leap years. This keeps `sun_ra` perfectly
/// linear in sim seconds so the J2 RAAN rate (also linear) stays locked.
pub fn continuous_day_of_year(start: DateTime<Utc>, sim_time_seconds: f64) -> f64 {
    use chrono::{Datelike, Timelike};
    let start_day = start.ordinal() as f64
        + (start.hour() as f64 * 3600.0 + start.minute() as f64 * 60.0 + start.second() as f64)
            / 86400.0;
    start_day + sim_time_seconds / 86400.0
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
