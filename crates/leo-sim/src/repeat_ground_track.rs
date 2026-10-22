//! Repeat-ground-track design for circular Keplerian orbits, with constant
//! Earth rotation. Revolutions and sidereal days are distinct from satellites.
use crate::observation::EARTH_RATE;
use std::f64::consts::TAU;

pub const SIDEREAL_DAY: f64 = TAU / EARTH_RATE;
pub const MAX_REPEAT_DAYS: u32 = 30;
pub const REPEAT_PHASE_TOLERANCE: f64 = 1e-9;

pub fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RepeatRatio {
    pub revolutions: u32,
    pub days: u32,
}
impl RepeatRatio {
    pub fn new(revolutions: u32, days: u32) -> Result<Self, String> {
        if revolutions == 0 || days == 0 {
            return Err("Repeat integers must be positive.".into());
        }
        let divisor = gcd(revolutions, days);
        Ok(Self {
            revolutions: revolutions / divisor,
            days: days / divisor,
        })
    }
    pub fn duration(self) -> f64 {
        self.days as f64 * SIDEREAL_DAY
    }
    pub fn orbital_period(self) -> f64 {
        self.duration() / self.revolutions as f64
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub ratio: RepeatRatio,
    pub altitude_km: f64,
}

pub fn design(ratio: RepeatRatio, mu: f64, radius_km: f64) -> Result<Candidate, String> {
    let ratio = RepeatRatio::new(ratio.revolutions, ratio.days)?;
    if !mu.is_finite() || mu <= 0.0 || !radius_km.is_finite() || radius_km <= 0.0 {
        return Err("Positive finite gravitational parameter and radius required.".into());
    }
    let altitude_km = (mu * (ratio.orbital_period() / TAU).powi(2)).cbrt() - radius_km;
    if altitude_km <= 0.0 {
        return Err("Repeat orbit is below the surface.".into());
    }
    Ok(Candidate { ratio, altitude_km })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Search {
    pub nominal_altitude_km: f64,
    pub tolerance_km: f64,
    pub max_days: u32,
}
impl Default for Search {
    fn default() -> Self {
        Self {
            nominal_altitude_km: 550.0,
            tolerance_km: 100.0,
            max_days: 7,
        }
    }
}

/// Enumerate reduced ratios in the requested altitude range, ordered by
/// distance from the nominal altitude, then by repeat cycle. No altitude snap
/// happens until a caller explicitly applies a selected candidate.
pub fn candidates(search: Search, mu: f64, radius_km: f64) -> Result<Vec<Candidate>, String> {
    if !search.nominal_altitude_km.is_finite()
        || !(100.0..=2000.0).contains(&search.nominal_altitude_km)
        || !search.tolerance_km.is_finite()
        || !(0.0..=1900.0).contains(&search.tolerance_km)
        || !(1..=MAX_REPEAT_DAYS).contains(&search.max_days)
    {
        return Err("Use a 100–2000 km nominal altitude, a 0–1900 km tolerance and a 1–30 sidereal-day limit.".into());
    }
    design(RepeatRatio::new(15, 1)?, mu, radius_km)?; // validate physical constants
    let lo = (search.nominal_altitude_km - search.tolerance_km).max(100.0);
    let hi = (search.nominal_altitude_km + search.tolerance_km).min(2000.0);
    let orbit_rate = |h: f64| (mu / (radius_km + h).powi(3)).sqrt() / EARTH_RATE;
    let mut out = Vec::new();
    for days in 1..=search.max_days {
        let min_m = (orbit_rate(hi) * days as f64).floor() as u32;
        let max_m = (orbit_rate(lo) * days as f64).ceil() as u32;
        for revolutions in min_m.max(1)..=max_m {
            if gcd(revolutions, days) != 1 {
                continue;
            }
            let c = design(RepeatRatio { revolutions, days }, mu, radius_km)?;
            if c.altitude_km >= lo - 1e-9 && c.altitude_km <= hi + 1e-9 {
                out.push(c);
            }
        }
    }
    out.sort_by(|a, b| {
        (a.altitude_km - search.nominal_altitude_km)
            .abs()
            .total_cmp(&(b.altitude_km - search.nominal_altitude_km).abs())
            .then(a.ratio.days.cmp(&b.ratio.days))
            .then(a.ratio.revolutions.cmp(&b.ratio.revolutions))
    });
    Ok(out)
}

pub fn detect(rate: f64) -> Option<RepeatRatio> {
    if !rate.is_finite() || rate <= 0.0 {
        return None;
    }
    (1..=MAX_REPEAT_DAYS).find_map(|days| {
        let cycles = rate * days as f64 * SIDEREAL_DAY / TAU;
        (cycles.round() >= 1.0
            && cycles.round() <= u32::MAX as f64
            && (cycles - cycles.round()).abs() * TAU <= REPEAT_PHASE_TOLERANCE)
            .then(|| RepeatRatio::new(cycles.round() as u32, days).unwrap())
    })
}

/// Sufficient (not optimal) count for evenly phased satellites in one plane.
/// Coprime d and N ensure all mN ascending longitudes are distinct for d>1.
pub fn baseline_count(ratio: RepeatRatio, swath_km: f64, radius_km: f64) -> Result<usize, String> {
    let ratio = RepeatRatio::new(ratio.revolutions, ratio.days)?;
    if !swath_km.is_finite() || swath_km <= 0.0 || !radius_km.is_finite() || radius_km <= 0.0 {
        return Err("Positive swath and radius required.".into());
    }
    let required = (TAU * radius_km / (ratio.revolutions as f64 * swath_km))
        .ceil()
        .max(1.0);
    if required > 100_000.0 {
        return Err(
            "Baseline would exceed 100,000 satellites; increase the swath or repeat length.".into(),
        );
    }
    let mut n = required as u32;
    while gcd(n, ratio.days) != 1 {
        n += 1;
    }
    if n > 100_000 {
        return Err("Baseline would exceed 100,000 satellites.".into());
    }
    Ok(n as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{EARTH_MU as MU, EARTH_RADIUS_KM as R};
    #[test]
    fn candidates_are_reduced_repeat_orbits_inside_the_requested_range() {
        let s = Search::default();
        let cs = candidates(s, MU, R).unwrap();
        assert!(!cs.is_empty());
        for c in &cs {
            assert!((c.altitude_km - s.nominal_altitude_km).abs() <= s.tolerance_km);
            assert_eq!(gcd(c.ratio.revolutions, c.ratio.days), 1);
            let n = (MU / (R + c.altitude_km).powi(3)).sqrt();
            assert_eq!(detect(n), Some(c.ratio));
        }
        assert!(cs
            .windows(2)
            .all(|p| (p[0].altitude_km - 550.0).abs() <= (p[1].altitude_km - 550.0).abs()));
        let daily = design(RepeatRatio::new(15, 1).unwrap(), MU, R).unwrap();
        assert!((daily.altitude_km - 561.385853).abs() < 0.001);
        assert_eq!(RepeatRatio::new(30, 2).unwrap(), daily.ratio);
        assert_eq!(baseline_count(daily.ratio, 20.0, R).unwrap(), 134);
        assert_eq!(
            baseline_count(RepeatRatio::new(31, 2).unwrap(), 21.0, R).unwrap() % 2,
            1
        );
    }
    #[test]
    fn empty_search_invalid_parameters_and_nonrepeat_are_distinct() {
        let search = Search {
            nominal_altitude_km: 550.0,
            tolerance_km: 0.0,
            max_days: 1,
        };
        assert!(candidates(search, MU, R).unwrap().is_empty());
        assert!(candidates(
            Search {
                max_days: 0,
                ..search
            },
            MU,
            R
        )
        .is_err());
        assert!(candidates(
            Search {
                tolerance_km: f64::NAN,
                ..search
            },
            MU,
            R
        )
        .is_err());
        assert!(RepeatRatio::new(0, 1).is_err());
        assert!(baseline_count(RepeatRatio::new(15, 1).unwrap(), 0.0, R).is_err());
        assert_eq!(detect((MU / (R + 550.0).powi(3)).sqrt()), None);
    }
}
