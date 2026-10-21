//! Analytical orbit math extracted from leo-viz's Walker implementation.
use crate::geometry::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, TAU};
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Propagation {
    Keplerian,
    J2,
}
#[derive(Clone, Copy, Debug)]
pub struct Elements {
    pub semi_major_km: f64,
    pub eccentricity: f64,
    pub inclination_rad: f64,
    pub raan_rad: f64,
    pub argument_rad: f64,
    pub mean_anomaly_rad: f64,
    pub mu: f64,
    pub radius_km: f64,
    pub j2: f64,
    pub propagation: Propagation,
}
#[derive(Clone, Copy, Debug)]
pub struct Position {
    pub xyz: Vector,
    pub lat_deg: f64,
    pub inertial_lon_deg: f64,
    pub ascending: bool,
}
impl Elements {
    pub fn sample(&self, time: f64) -> Position {
        let e = self.eccentricity;
        let inc = self.inclination_rad;
        let n = (self.mu / self.semi_major_km.powi(3)).sqrt();
        let rr = self.radius_km / self.semi_major_km;
        let den = (1.0 - e * e).powi(2);
        let coeff = if self.propagation == Propagation::J2 {
            1.5 * self.j2 * rr * rr * n / den
        } else {
            0.0
        };
        // Positive rotation around +y decreases geographic longitude.
        let raan = self.raan_rad + coeff * inc.cos() * time;
        let arg = self.argument_rad + coeff * (2.0 - 2.5 * inc.sin().powi(2)) * time;
        let m_dot = n + 0.5 * coeff * (3.0 * inc.cos().powi(2) - 1.0) * (1.0 - e * e).sqrt();
        let m = (self.mean_anomaly_rad + m_dot * time).rem_euclid(TAU);
        let mut ea = if e > 0.8 { PI } else { m };
        for _ in 0..30 {
            let d = (ea - e * ea.sin() - m) / (1.0 - e * ea.cos());
            ea -= d;
            if d.abs() < 1e-13 {
                break;
            }
        }
        let v =
            2.0 * ((1.0 + e).sqrt() * (ea / 2.0).sin()).atan2((1.0 - e).sqrt() * (ea / 2.0).cos());
        let r = self.semi_major_km * (1.0 - e * ea.cos());
        let angle = v + arg;
        let xo = r * angle.cos();
        let yo = -r * angle.sin();
        let xyz = [
            xo * raan.cos() - yo * inc.cos() * raan.sin(),
            -yo * inc.sin(),
            xo * raan.sin() + yo * inc.cos() * raan.cos(),
        ];
        Position {
            xyz,
            lat_deg: (xyz[1] / r).clamp(-1.0, 1.0).asin().to_degrees(),
            inertial_lon_deg: -xyz[2].atan2(xyz[0]).to_degrees(),
            ascending: angle.cos() > 0.0,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WalkerShell {
    pub planes: usize,
    pub satellites_per_plane: usize,
    pub altitude_km: f64,
    pub inclination_deg: f64,
    pub phasing: usize,
    pub raan_offset_deg: f64,
    pub star: bool,
    pub eccentricity: f64,
    pub argument_deg: f64,
    pub propagation: Propagation,
}
impl Default for WalkerShell {
    fn default() -> Self {
        Self {
            planes: 4,
            satellites_per_plane: 8,
            altitude_km: 550.0,
            inclination_deg: 53.0,
            phasing: 1,
            raan_offset_deg: 0.0,
            star: false,
            eccentricity: 0.0,
            argument_deg: 0.0,
            propagation: Propagation::Keplerian,
        }
    }
}
impl WalkerShell {
    pub fn count(&self) -> usize {
        self.planes * self.satellites_per_plane
    }
    pub fn elements(&self, plane: usize, sat: usize) -> Elements {
        Elements {
            semi_major_km: (EARTH_RADIUS_KM + self.altitude_km) / (1.0 - self.eccentricity),
            eccentricity: self.eccentricity,
            inclination_rad: self.inclination_deg.to_radians(),
            raan_rad: -self.raan_offset_deg.to_radians()
                + (if self.star { PI } else { TAU }) * plane as f64 / self.planes as f64,
            argument_rad: self.argument_deg.to_radians(),
            mean_anomaly_rad: TAU * sat as f64 / self.satellites_per_plane as f64
                + TAU * self.phasing as f64 * plane as f64 / self.count() as f64,
            mu: EARTH_MU,
            radius_km: EARTH_EQUATORIAL_RADIUS_KM,
            j2: EARTH_J2,
            propagation: self.propagation,
        }
    }
    pub fn validate(&self) -> crate::Result<()> {
        if self.planes == 0
            || self.satellites_per_plane == 0
            || self.planes.checked_mul(self.satellites_per_plane).is_none()
        {
            return Err("Walker dimensions must be positive and fit usize".into());
        }
        if !self.altitude_km.is_finite()
            || self.altitude_km < 100.0
            || !self.inclination_deg.is_finite()
            || !(0.0..=180.0).contains(&self.inclination_deg)
            || !self.eccentricity.is_finite()
            || !(0.0..0.95).contains(&self.eccentricity)
            || !self.raan_offset_deg.is_finite()
            || !self.argument_deg.is_finite()
            || self.phasing >= self.planes
        {
            return Err("invalid Walker orbit or phasing (require F < planes)".into());
        }
        Ok(())
    }
}
