//! Point observations for a continuous nadir pushbroom on a spherical Earth.
//!
//! Circular Keplerian orbits and constant Earth rotation give three harmonics
//! for x·q(t). Observations are stationary points of that expression within
//! the swath. We isolate those events, rather than time-step satellite tracks.
//! Root times are numerical; this is not a certified regional coverage solver.
use crate::{
    geometry::Vector,
    orbit::{Elements, Propagation},
    repeat_ground_track::{detect, MAX_REPEAT_DAYS, REPEAT_PHASE_TOLERANCE, SIDEREAL_DAY},
};
use std::f64::consts::{PI, TAU};

pub const EARTH_RATE: f64 = 360.98564736629 * PI / 180.0 / 86400.0;
pub const TIME_TOLERANCE: f64 = 0.0001;

#[derive(Clone, Debug)]
pub struct Model {
    orbits: Vec<Elements>,
    pub half_width: f64,
    pub latitude_limit: f64,
    pub orbit_period_range: (f64, f64),
    /// A repeat within 30 sidereal days, to floating-point phase tolerance.
    pub repeat: Option<f64>,
    guarantees: Vec<CoverageGuarantee>,
}

#[derive(Clone, Copy, Debug)]
pub struct Coverage {
    /// Every point in |latitude| <= latitude_limit is observed in every P.
    pub period: f64,
    pub latitude_limit: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct CoverageGuarantee {
    /// Every point in the symmetric latitude band is observed in every interval
    /// of this length. `exact` distinguishes a minimum from a sufficient bound.
    pub interval_s: f64,
    pub latitude_limit: f64,
    pub exact: bool,
}

/// Sufficient coverage certificates for each common-rate, common-inclination
/// subset. Extra satellites cannot invalidate coverage provided by a subset.
fn track_guarantees(orbits: &[Elements], half_width: f64) -> Vec<CoverageGuarantee> {
    let mut groups = std::collections::BTreeMap::<(u64, u64), Vec<&Elements>>::new();
    for e in orbits {
        if e.inclination_rad.sin().abs() < 1e-12 || !(0.0..=PI).contains(&e.inclination_rad) {
            continue;
        }
        groups
            .entry((rate(e).to_bits(), e.inclination_rad.to_bits()))
            .or_default()
            .push(e);
    }
    let mut guarantees = Vec::new();
    for group in groups.values() {
        let Some(ratio) = detect(rate(group[0])) else {
            continue;
        };
        let cell = TAU / ratio.revolutions as f64;
        // In the viewer's coordinates positive RAAN rotation decreases
        // longitude. At an ascending equator crossing u=0, t=-u0/n+k Torb.
        // The m tracks tile longitude with spacing 2π/m because gcd(m,d)=1.
        // Other latitudes shift every ascending crossing by the same amount.
        let mut phases: Vec<_> = group
            .iter()
            .map(|e| {
                (-e.raan_rad
                    + ratio.days as f64 / ratio.revolutions as f64
                        * (e.mean_anomaly_rad + e.argument_rad))
                    .rem_euclid(cell)
            })
            .collect();
        let gap = circular_gap(&mut phases, cell);
        if gap + 2.0 * REPEAT_PHASE_TOLERANCE <= 2.0 * half_width {
            // Same-latitude distance to a crossing <= R*gap/2. The nearest
            // point on that closed track is at least as close and satisfies
            // x·q'=0, hence lies on a pushbroom scanline. Includes latitude
            // extrema by continuity; does not certify the swath's edge band.
            guarantees.push(CoverageGuarantee {
                interval_s: ratio.duration(),
                latitude_limit: group[0].inclination_rad.sin().abs().asin(),
                exact: false,
            });
        }
    }
    guarantees
}

#[derive(Clone, Debug)]
pub struct PointReport {
    /// Some(infinity) means geometrically unreachable, not just unseen.
    pub exact_gap: Option<f64>,
    pub events: usize,
    pub max_gap: Option<f64>,
    pub longest_silence: f64,
    pub duration: f64,
    pub cyclic: bool,
}

fn rate(e: &Elements) -> f64 {
    (e.mu / e.semi_major_km.powi(3)).sqrt()
}
fn equatorial_rate(orbits: &[Elements]) -> Option<(f64, f64)> {
    let first = orbits.first()?;
    let sign = first.inclination_rad.cos().signum();
    let speed = sign * rate(first) - EARTH_RATE;
    orbits
        .iter()
        .all(|e| {
            e.inclination_rad.sin().abs() <= 1e-12
                && e.inclination_rad.cos().signum() * rate(e) - EARTH_RATE == speed
        })
        .then_some((sign, speed))
}
fn circular_gap(phases: &mut [f64], period: f64) -> f64 {
    phases.sort_by(f64::total_cmp);
    phases
        .windows(2)
        .map(|p| p[1] - p[0])
        .fold(period + phases[0] - phases[phases.len() - 1], f64::max)
}

impl Model {
    pub fn new(orbits: Vec<Elements>, swath_km: f64, radius_km: f64) -> Result<Self, String> {
        if orbits.is_empty() {
            return Err("Show at least one Earth constellation.".into());
        }
        if !radius_km.is_finite()
            || radius_km <= 0.0
            || !swath_km.is_finite()
            || !(0.0..PI * radius_km).contains(&swath_km)
            || swath_km == 0.0
        {
            return Err("A positive sub-hemisphere swath and Earth radius are required.".into());
        }
        for e in &orbits {
            if e.propagation != Propagation::Keplerian || e.eccentricity.abs() > 1e-12 {
                return Err("Analytical events require circular Keplerian orbits. Use the sampled forecast for J2, eccentric or TLE/SGP4 orbits.".into());
            }
            if [
                e.semi_major_km,
                e.inclination_rad,
                e.raan_rad,
                e.argument_rad,
                e.mean_anomaly_rad,
                e.mu,
            ]
            .iter()
            .any(|v| !v.is_finite())
                || e.mu <= 0.0
                || e.semi_major_km <= radius_km
                || rate(e) <= EARTH_RATE * 1.01
            {
                return Err("Analytical events require finite LEO orbital parameters.".into());
            }
        }
        let half_width = swath_km / (2.0 * radius_km);
        if orbits
            .iter()
            .any(|e| half_width > (radius_km / e.semi_major_km).acos())
        {
            return Err("Swath extends beyond a satellite's horizon; reduce its width.".into());
        }
        let latitude_limit = (orbits
            .iter()
            .map(|e| e.inclination_rad.sin().abs().asin())
            .fold(0.0, f64::max)
            + half_width)
            .min(PI / 2.0);
        let periods: Vec<_> = orbits.iter().map(|e| TAU / rate(e)).collect();
        let orbit_period_range = (
            periods.iter().copied().fold(f64::INFINITY, f64::min),
            periods.iter().copied().fold(0.0, f64::max),
        );
        // Do not silently round arbitrary altitudes into repeating orbits.
        let repeat = equatorial_rate(&orbits)
            .map(|(_, speed)| TAU / speed.abs())
            .filter(|t| *t <= MAX_REPEAT_DAYS as f64 * SIDEREAL_DAY)
            .or_else(|| {
                (1..=MAX_REPEAT_DAYS)
                    .map(|days| days as f64 * SIDEREAL_DAY)
                    .find(|t| {
                        orbits.iter().all(|e| {
                            let cycles = rate(e) * t / TAU;
                            (cycles - cycles.round()).abs() * TAU <= REPEAT_PHASE_TOLERANCE
                        })
                    })
            });
        let guarantees = track_guarantees(&orbits, half_width);
        Ok(Self {
            orbits,
            half_width,
            latitude_limit,
            orbit_period_range,
            repeat,
            guarantees,
        })
    }

    pub fn coverage_guarantee(&self, latitude_limit: f64) -> Option<CoverageGuarantee> {
        if !latitude_limit.is_finite() || !(0.0..=PI / 2.0).contains(&latitude_limit) {
            return None;
        }
        let exact = self.coverage().map(|c| CoverageGuarantee {
            interval_s: c.period,
            latitude_limit: c.latitude_limit,
            exact: true,
        });
        self.guarantees
            .iter()
            .copied()
            .chain(exact)
            .filter(|c| latitude_limit <= c.latitude_limit + 16.0 * f64::EPSILON)
            .min_by(|a, b| a.interval_s.total_cmp(&b.interval_s))
    }

    /// Exact common-rate equatorial solution, allowing nonuniform phases and
    /// duplicate satellites. Delta/Star plane labels alone do not imply spacing.
    pub fn coverage(&self) -> Option<Coverage> {
        let (sign, speed) = equatorial_rate(&self.orbits)?;
        let mut phases: Vec<_> = self
            .orbits
            .iter()
            .map(|e| (sign * (e.mean_anomaly_rad + e.argument_rad) - e.raan_rad).rem_euclid(TAU))
            .collect();
        Some(Coverage {
            period: circular_gap(&mut phases, TAU) / speed.abs(),
            latitude_limit: self.half_width,
        })
    }

    fn exact_point_gap(&self, latitude: f64) -> Option<f64> {
        if latitude.abs() > self.latitude_limit + 1e-12 {
            return Some(f64::INFINITY);
        }
        if let Some(c) = self.coverage() {
            return Some(if latitude.abs() <= c.latitude_limit + 1e-12 {
                c.period
            } else {
                f64::INFINITY
            });
        }
        // At a geographic pole the Earth rotation drops out of x·q(t).
        if (latitude.abs() - PI / 2.0).abs() < 1e-12 {
            let eligible: Vec<_> = self
                .orbits
                .iter()
                .filter(|e| e.inclination_rad.sin().abs() >= self.half_width.cos() - 1e-14)
                .collect();
            if eligible.is_empty() {
                return Some(f64::INFINITY);
            }
            let n = rate(eligible[0]);
            if eligible.iter().all(|e| rate(e) == n) {
                let mut phases: Vec<_> = eligible
                    .iter()
                    .map(|e| {
                        (e.mean_anomaly_rad + e.argument_rad
                            - if latitude * e.inclination_rad.sin() > 0.0 {
                                PI / 2.0
                            } else {
                                3.0 * PI / 2.0
                            })
                        .rem_euclid(TAU)
                    })
                    .collect();
                return Some(circular_gap(&mut phases, TAU) / n);
            }
        }
        None
    }

    /// `start` is relative to the elements' epoch. `earth_angle` is GMST at
    /// that epoch; after it we use constant EARTH_RATE. Progress can cancel.
    pub fn point(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        start: f64,
        earth_angle: f64,
        horizon: f64,
        mut progress: impl FnMut(f32) -> bool,
    ) -> Result<PointReport, String> {
        if !lat_deg.is_finite()
            || lat_deg.abs() > 90.0
            || !lon_deg.is_finite()
            || !start.is_finite()
            || !earth_angle.is_finite()
            || !horizon.is_finite()
            || horizon <= 0.0
        {
            return Err("Invalid coordinates, epoch or analysis horizon.".into());
        }
        if let Some(gap) = self.exact_point_gap(lat_deg.to_radians()) {
            return Ok(PointReport {
                exact_gap: Some(gap),
                events: 0,
                max_gap: None,
                longest_silence: 0.0,
                duration: 0.0,
                cyclic: false,
            });
        }
        let duration = self.repeat.unwrap_or(horizon);
        let (s, c) = lat_deg.to_radians().sin_cos();
        let (sl, cl) = lon_deg.to_radians().sin_cos();
        let x = [c * cl, s, -c * sl];
        let mut events = Vec::new();
        for (index, e) in self.orbits.iter().enumerate() {
            let series = Series::new(e, x, start, earth_angle);
            let chunk = TAU / rate(e) / 4.0;
            let count = (duration / chunk).ceil() as usize;
            for j in 0..count {
                if j % 64 == 0
                    && !progress(
                        (index as f32 + j as f32 / count as f32) / self.orbits.len() as f32,
                    )
                {
                    return Err("Cancelled.".into());
                }
                series.roots(
                    j as f64 * chunk,
                    ((j + 1) as f64 * chunk).min(duration),
                    self.half_width.cos(),
                    &mut events,
                );
            }
        }
        events.sort_by(f64::total_cmp);
        events.dedup_by(|a, b| (*a - *b).abs() < 2.0 * TIME_TOLERANCE);
        let cyclic = self.repeat.is_some();
        if cyclic
            && events.len() > 1
            && duration + events[0] - events[events.len() - 1] < 2.0 * TIME_TOLERANCE
        {
            events.pop();
        }
        let mut max_gap = events.windows(2).map(|p| p[1] - p[0]).reduce(f64::max);
        let longest_silence = if let (Some(first), Some(last)) = (events.first(), events.last()) {
            if cyclic {
                max_gap = Some(max_gap.unwrap_or(0.0).max(duration + first - last));
            }
            max_gap.unwrap_or(0.0).max(*first).max(duration - last)
        } else {
            duration
        };
        if cyclic && events.is_empty() {
            max_gap = Some(f64::INFINITY);
        }
        Ok(PointReport {
            exact_gap: None,
            events: events.len(),
            max_gap,
            longest_silence,
            duration,
            cyclic,
        })
    }
}

#[derive(Clone, Copy)]
struct Harmonic {
    frequency: f64,
    phase: f64,
    cosine: f64,
    sine: f64,
}
struct Series {
    terms: [Harmonic; 3],
    bounds: [f64; 3],
}
impl Series {
    fn new(e: &Elements, x: Vector, start: f64, earth_angle: f64) -> Self {
        let n = rate(e);
        let u = e.mean_anomaly_rad + e.argument_rad + n * start;
        let v = e.raan_rad + earth_angle + EARTH_RATE * start;
        let a = (1.0 + e.inclination_rad.cos()) / 2.0;
        let b = (1.0 - e.inclination_rad.cos()) / 2.0;
        let terms = [
            Harmonic {
                frequency: n - EARTH_RATE,
                phase: (u - v).rem_euclid(TAU),
                cosine: a * x[0],
                sine: -a * x[2],
            },
            Harmonic {
                frequency: n + EARTH_RATE,
                phase: (u + v).rem_euclid(TAU),
                cosine: b * x[0],
                sine: b * x[2],
            },
            Harmonic {
                frequency: n,
                phase: u.rem_euclid(TAU),
                cosine: 0.0,
                sine: x[1] * e.inclination_rad.sin(),
            },
        ];
        let bounds = std::array::from_fn(|i| {
            terms
                .iter()
                .map(|h| h.cosine.hypot(h.sine) * h.frequency.abs().powi(i as i32 + 1))
                .sum()
        });
        Self { terms, bounds }
    }
    fn value(&self, t: f64) -> [f64; 3] {
        let mut v = [0.0; 3];
        for h in &self.terms {
            let (s, c) = (h.frequency * t + h.phase).sin_cos();
            let g = h.cosine * c + h.sine * s;
            v[0] += g;
            v[1] += h.frequency * (-h.cosine * s + h.sine * c);
            v[2] -= h.frequency * h.frequency * g;
        }
        v
    }
    fn roots(&self, left: f64, right: f64, threshold: f64, out: &mut Vec<f64>) {
        let mid = (left + right) / 2.0;
        let half = (right - left) / 2.0;
        let v = self.value(mid);
        // Lipschitz bounds exclude intervals without visibility or stationary
        // points. The third derivative bound can prove monotonicity of g'.
        if v[0] + self.bounds[0] * half < threshold - 1e-14
            || v[1].abs() > self.bounds[1] * half + 1e-16
        {
            return;
        }
        if v[2].abs() > self.bounds[2] * half + 1e-18 {
            let mut a = left;
            let mut b = right;
            let fa = self.value(a)[1];
            let fb = self.value(b)[1];
            if fa * fb > 0.0 {
                return;
            }
            while b - a > TIME_TOLERANCE {
                let m = (a + b) / 2.0;
                if self.value(m)[1] * fa > 0.0 {
                    a = m;
                } else {
                    b = m;
                }
            }
            let t = (a + b) / 2.0;
            if self.value(t)[0] >= threshold - 1e-14 {
                out.push(t);
            }
        } else if right - left <= TIME_TOLERANCE {
            // Degenerate/tangent event, localized to the reported tolerance.
            if v[0] >= threshold - 1e-14 {
                out.push(mid);
            }
        } else {
            self.roots(left, mid, threshold, out);
            self.roots(mid, right, threshold, out);
        }
    }
}

#[cfg(test)]
mod tests;
