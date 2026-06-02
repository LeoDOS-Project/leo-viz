//! Walker constellation calculations.
//!
//! Implements Walker Delta and Walker Star satellite constellation patterns,
//! computing orbital positions, RAAN drift, and inter-satellite neighbor links.

use crate::config::{LinkBudget, NumericalSatState, NumericalState, Propagator};

use std::f64::consts::PI;

#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WalkerType {
    Delta,
    Star,
}

pub struct WalkerConstellation {
    pub walker_type: WalkerType,
    pub total_sats: usize,
    pub num_planes: usize,
    pub altitude_km: f64,
    pub inclination_deg: f64,
    pub phasing: f64,
    pub raan_offset_deg: f64,
    pub raan_spacing_deg: Option<f64>,
    pub sat_spacing_km: Option<f64>,
    pub isl_neighbors: usize,
    pub propagator: Propagator,
    pub eccentricity: f64,
    pub arg_periapsis_deg: f64,
    pub planet_radius: f64,
    pub planet_mu: f64,
    pub planet_j2: f64,
    pub planet_equatorial_radius: f64,
    pub link_budget: LinkBudget,
    pub show_isl_hover_info: bool,
    pub ballistic_coeff: f64,
}

pub struct SatelliteState {
    pub plane: usize,
    pub sat_index: usize,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub lat: f64,
    pub lon: f64,
    pub ascending: bool,
    pub neighbors: Vec<usize>,
    pub name: Option<String>,
    pub tle_inclination_deg: Option<f64>,
    pub tle_mean_motion: Option<f64>,
}

impl WalkerConstellation {
    pub fn sats_per_plane(&self) -> usize {
        self.total_sats / self.num_planes
    }

    pub fn raan_spread(&self) -> f64 {
        if let Some(spacing) = self.raan_spacing_deg {
            spacing.to_radians() * (self.num_planes - 1) as f64
        } else {
            match self.walker_type {
                WalkerType::Delta => 2.0 * PI,
                WalkerType::Star => PI,
            }
        }
    }

    pub fn raan_step(&self) -> f64 {
        if let Some(spacing) = self.raan_spacing_deg {
            spacing.to_radians()
        } else {
            self.raan_spread() / self.num_planes as f64
        }
    }

    fn sat_step(&self) -> f64 {
        if let Some(spacing_km) = self.sat_spacing_km {
            let orbit_radius = self.planet_radius + self.altitude_km;
            spacing_km / orbit_radius
        } else {
            2.0 * PI / self.sats_per_plane() as f64
        }
    }

    fn partial_sat_coverage(&self) -> bool {
        if self.sat_spacing_km.is_none() {
            return false;
        }
        (self.sat_step() * self.sats_per_plane() as f64) < 2.0 * PI - 1e-9
    }

    fn partial_coverage(&self) -> bool {
        if self.raan_spacing_deg.is_none() {
            return false;
        }
        let max_spread = match self.walker_type {
            WalkerType::Delta => 2.0 * PI,
            WalkerType::Star => PI,
        };
        (self.raan_step() * self.num_planes as f64) < max_spread - 1e-9
    }

    pub fn single_satellite_lat_lon(
        &self,
        plane: usize,
        sat: usize,
        time: f64,
    ) -> (f64, f64, [f64; 3]) {
        let sats_per_plane = self.sats_per_plane();
        let perigee_radius = self.planet_radius + self.altitude_km;
        let ecc = self.eccentricity;
        let semi_major = perigee_radius / (1.0 - ecc);
        let period = 2.0 * PI * (semi_major.powi(3) / self.planet_mu).sqrt();
        let mean_motion = 2.0 * PI / period;
        let inc = self.inclination_deg.to_radians();
        let inc_cos = inc.cos();
        let inc_sin = inc.sin();
        let raan_step = self.raan_step();
        let raan_offset = -self.raan_offset_deg.to_radians();
        let sat_step = self.sat_step();
        let omega = self.arg_periapsis_deg.to_radians();
        let phase_step = self.phasing * 2.0 * PI / self.total_sats as f64;
        let r_ratio = self.planet_equatorial_radius / semi_major;
        let raan_drift_rate = -1.5 * self.planet_j2 * r_ratio * r_ratio * mean_motion * inc_cos;
        let center_offset = if self.partial_coverage() {
            raan_step * (self.num_planes - 1) as f64 / 2.0
        } else {
            0.0
        };
        let sat_center_offset = if self.partial_sat_coverage() {
            sat_step * (sats_per_plane - 1) as f64 / 2.0
        } else {
            0.0
        };
        let dead = self.altitude_km < 100.0;

        let raan_initial = raan_offset + raan_step * plane as f64 - center_offset;
        let raan = raan_initial + if dead { 0.0 } else { raan_drift_rate * time };
        let raan_cos = raan.cos();
        let raan_sin = raan.sin();
        let phase_offset = phase_step * plane as f64;

        let mean_anomaly = sat_step * sat as f64 - sat_center_offset
            + if dead { 0.0 } else { mean_motion * time }
            + phase_offset;
        let true_anomaly = if ecc < 1e-8 {
            mean_anomaly
        } else {
            let mut ea = mean_anomaly;
            for _ in 0..10 {
                ea = ea - (ea - ecc * ea.sin() - mean_anomaly) / (1.0 - ecc * ea.cos());
            }
            2.0 * ((1.0 + ecc).sqrt() * (ea / 2.0).sin())
                .atan2((1.0 - ecc).sqrt() * (ea / 2.0).cos())
        };

        let r = semi_major * (1.0 - ecc * ecc) / (1.0 + ecc * true_anomaly.cos());
        let angle = true_anomaly + omega;
        let x_orbital = r * angle.cos();
        let y_orbital = -r * angle.sin();
        let x = x_orbital * raan_cos - y_orbital * inc_cos * raan_sin;
        let z = x_orbital * raan_sin + y_orbital * inc_cos * raan_cos;
        let y = -y_orbital * inc_sin;

        let lat = (y / r).asin().to_degrees();
        let lon = -z.atan2(x).to_degrees();
        (lat, lon, [x, y, z])
    }

    pub fn satellite_positions(&self, time: f64) -> Vec<SatelliteState> {
        let mut positions = Vec::with_capacity(self.total_sats);
        let sats_per_plane = self.sats_per_plane();
        let perigee_radius = self.planet_radius + self.altitude_km;
        let ecc = self.eccentricity;
        let semi_major = perigee_radius / (1.0 - ecc);
        let period = 2.0 * PI * (semi_major.powi(3) / self.planet_mu).sqrt();
        let mean_motion = 2.0 * PI / period;
        let inc = self.inclination_deg.to_radians();
        let raan_step = self.raan_step();
        let raan_offset = -self.raan_offset_deg.to_radians();
        let sat_step = self.sat_step();
        let is_star = self.walker_type == WalkerType::Star;
        let omega = self.arg_periapsis_deg.to_radians();

        let phase_step = self.phasing * 2.0 * PI / self.total_sats as f64;

        let center_offset = if self.partial_coverage() {
            raan_step * (self.num_planes - 1) as f64 / 2.0
        } else {
            0.0
        };
        let sat_center_offset = if self.partial_sat_coverage() {
            sat_step * (sats_per_plane - 1) as f64 / 2.0
        } else {
            0.0
        };
        let dead = self.altitude_km < 100.0;

        #[cfg(not(target_arch = "wasm32"))]
        let use_lib42 = self.propagator == Propagator::Lib42;
        #[cfg(target_arch = "wasm32")]
        let use_lib42 = false;

        if use_lib42 {
            #[cfg(not(target_arch = "wasm32"))]
            {
                // 42 uses meters internally
                let sma_m = semi_major * 1000.0;
                let mu_m = self.planet_mu * 1e9; // km³/s² → m³/s²
                let re_m = self.planet_equatorial_radius * 1000.0;
                let j2 = self.planet_j2;
                let slr_m = sma_m * (1.0 - ecc * ecc);
                let re_slr = re_m / slr_m;
                let n = (mu_m / (sma_m * sma_m * sma_m)).sqrt();
                let j2_rw2bya = j2 * re_m * re_m / sma_m;

                // J2 secular drift rates (same as 42's formulas)
                let raan_dot = -1.5 * j2 * re_slr * re_slr * n * inc.cos();
                let argp_dot = 1.5 * j2 * re_slr * re_slr * n * (2.0 - 2.5 * inc.sin().powi(2));

                for plane in 0..self.num_planes {
                    let raan_initial = raan_offset + raan_step * plane as f64 - center_offset;
                    let phase_offset = phase_step * plane as f64;

                    for sat in 0..sats_per_plane {
                        let m0 = sat_step * sat as f64 - sat_center_offset + phase_offset;

                        let mut orb = leodos_lib42::sim::OrbitType::default();
                        orb.Regime = 2; // ORB_CENTRAL
                        orb.World = 3; // EARTH
                        orb.Exists = 1;
                        orb.mu = mu_m;
                        orb.SMA = sma_m;
                        orb.MeanSMA = sma_m;
                        orb.ecc = ecc;
                        orb.inc = inc;
                        orb.RAAN = raan_initial;
                        orb.RAAN0 = raan_initial;
                        orb.ArgP = omega;
                        orb.ArgP0 = omega;
                        orb.MeanAnom = m0;
                        orb.MeanAnom0 = m0;
                        orb.MeanMotion = n;
                        orb.Period = 2.0 * PI / n;
                        orb.SLR = slr_m;
                        orb.alpha = 1.0 / sma_m;
                        orb.rmin = sma_m * (1.0 - ecc);
                        orb.Epoch = 0.0;
                        orb.J2DriftEnabled = if dead { 0 } else { 1 };
                        orb.RAANdot = raan_dot;
                        orb.ArgPdot = argp_dot;
                        orb.J2Rw2bya = j2_rw2bya;

                        // DynTime = simulation time in seconds
                        leodos_lib42::orbit::mean_eph_to_rv(&mut orb, time);

                        // PosN is in meters, convert to km
                        // 42 uses ECI: x,y equatorial, z north
                        // Walker uses: x,z equatorial, y north
                        let x = orb.PosN[0] / 1000.0;
                        let y = orb.PosN[2] / 1000.0;
                        let z = -orb.PosN[1] / 1000.0;

                        let r = (x * x + y * y + z * z).sqrt();
                        let ascending = (orb.anom + orb.ArgP).cos() > 0.0;
                        let lat = (y / r).asin().to_degrees();
                        let lon = -z.atan2(x).to_degrees();

                        positions.push(SatelliteState {
                            plane,
                            sat_index: sat,
                            x,
                            y,
                            z,
                            lat,
                            lon,
                            ascending,
                            neighbors: Vec::new(),
                            name: None,
                            tle_inclination_deg: None,
                            tle_mean_motion: None,
                        });
                    }
                }
            }
        } else {
            let inc_cos = inc.cos();
            let inc_sin = inc.sin();
            let r_ratio = self.planet_equatorial_radius / semi_major;
            let use_j2 = self.propagator == Propagator::J2;

            // J2 secular perturbation rates
            let e2 = ecc * ecc;
            let one_minus_e2 = 1.0 - e2;
            let p_factor = one_minus_e2 * one_minus_e2;
            let j2_coeff = 1.5 * self.planet_j2 * r_ratio * r_ratio * mean_motion;

            // Sign note: with this codebase's convention `lon = -atan2(z, x)`, a
            // positive rotation of the ascending node around +y corresponds to a
            // *decreasing* longitude. J2 physically gives prograde RAAN regression
            // (−cos(i) rate) in standard astronomy; we flip the sign here so the
            // drift direction matches this code's lon convention — retrograde i
            // produces RAAN motion eastward (matching the Sun's apparent motion).
            let raan_rate = if use_j2 {
                j2_coeff * inc_cos / p_factor
            } else {
                0.0
            };
            let omega_rate = if use_j2 {
                j2_coeff * (2.0 - 2.5 * inc_sin * inc_sin) / p_factor
            } else {
                0.0
            };
            let m_dot = if use_j2 {
                mean_motion
                    + 0.75
                        * self.planet_j2
                        * r_ratio
                        * r_ratio
                        * mean_motion
                        * (3.0 * inc_cos * inc_cos - 1.0)
                        * one_minus_e2.sqrt()
                        / p_factor
            } else {
                mean_motion
            };

            for plane in 0..self.num_planes {
                let raan_initial = raan_offset + raan_step * plane as f64 - center_offset;
                let raan = raan_initial + if dead { 0.0 } else { raan_rate * time };
                let raan_cos = raan.cos();
                let raan_sin = raan.sin();
                let omega_t = omega + if dead { 0.0 } else { omega_rate * time };
                let phase_offset = phase_step * plane as f64;

                for sat in 0..sats_per_plane {
                    let raw_mean_anomaly = sat_step * sat as f64 - sat_center_offset
                        + if dead { 0.0 } else { m_dot * time }
                        + phase_offset;
                    let mean_anomaly = raw_mean_anomaly.rem_euclid(2.0 * PI);

                    let true_anomaly = if ecc < 1e-8 {
                        mean_anomaly
                    } else {
                        let mut ea = mean_anomaly;
                        for _ in 0..10 {
                            ea -= (ea - ecc * ea.sin() - mean_anomaly) / (1.0 - ecc * ea.cos());
                        }
                        2.0 * ((1.0 + ecc).sqrt() * (ea / 2.0).sin())
                            .atan2((1.0 - ecc).sqrt() * (ea / 2.0).cos())
                    };

                    let r = semi_major * (1.0 - ecc * ecc) / (1.0 + ecc * true_anomaly.cos());
                    let ascending = (true_anomaly + omega_t).cos() > 0.0;

                    let angle = true_anomaly + omega_t;
                    let x_orbital = r * angle.cos();
                    let y_orbital = -r * angle.sin();

                    let x = x_orbital * raan_cos - y_orbital * inc_cos * raan_sin;
                    let z = x_orbital * raan_sin + y_orbital * inc_cos * raan_cos;
                    let y = -y_orbital * inc_sin;

                    let lat = (y / r).asin().to_degrees();
                    let lon = -z.atan2(x).to_degrees();

                    positions.push(SatelliteState {
                        plane,
                        sat_index: sat,
                        x,
                        y,
                        z,
                        lat,
                        lon,
                        ascending,
                        neighbors: Vec::new(),
                        name: None,
                        tle_inclination_deg: None,
                        tle_mean_motion: None,
                    });
                }
            }
        }

        let no_plane_wrap = is_star || self.partial_coverage();
        let no_sat_wrap = self.partial_sat_coverage();
        let offsets: &[(isize, isize)] = match self.isl_neighbors {
            4 => &[(0, 1), (1, 0), (0, -1), (-1, 0)],
            8 => &[
                (0, 1),
                (1, 0),
                (0, -1),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ],
            _ => &[],
        };
        let np = self.num_planes as isize;
        let sp = sats_per_plane as isize;
        for i in 0..positions.len() {
            let plane = positions[i].plane as isize;
            let sat_idx = positions[i].sat_index as isize;
            let mut nbrs = Vec::new();
            for &(dp, ds) in offsets {
                let tp = plane + dp;
                let ts = sat_idx + ds;
                let tp = if no_plane_wrap {
                    if tp < 0 || tp >= np {
                        continue;
                    } else {
                        tp
                    }
                } else {
                    ((tp % np) + np) % np
                };
                let ts = if no_sat_wrap {
                    if ts < 0 || ts >= sp {
                        continue;
                    } else {
                        ts
                    }
                } else {
                    ((ts % sp) + sp) % sp
                };
                let j = (tp * sp + ts) as usize;
                if j < positions.len() && j > i {
                    nbrs.push(j);
                }
            }
            positions[i].neighbors = nbrs;
        }

        positions
    }

    pub fn orbit_points_3d(&self, plane: usize, time: f64) -> Vec<(f64, f64, f64)> {
        let ecc = self.eccentricity;
        let semi_major = (self.planet_radius + self.altitude_km) / (1.0 - ecc);
        let period = 2.0 * PI * (semi_major.powi(3) / self.planet_mu).sqrt();
        let mean_motion = 2.0 * PI / period;
        let r_ratio = self.planet_equatorial_radius / semi_major;
        let inc = self.inclination_deg.to_radians();
        let use_j2 = self.propagator == Propagator::J2;
        let raan_drift_rate = if use_j2 {
            let e2 = ecc * ecc;
            let one_minus_e2 = 1.0 - e2;
            let p_factor = one_minus_e2 * one_minus_e2;
            1.5 * self.planet_j2 * r_ratio * r_ratio * mean_motion * inc.cos() / p_factor
        } else {
            0.0
        };
        let omega = self.arg_periapsis_deg.to_radians();

        let raan_step = self.raan_step();
        let center_offset = if self.partial_coverage() {
            raan_step * (self.num_planes - 1) as f64 / 2.0
        } else {
            0.0
        };
        let raan_initial =
            -self.raan_offset_deg.to_radians() + raan_step * plane as f64 - center_offset;
        let raan = raan_initial + raan_drift_rate * time;
        let inc_cos = inc.cos();
        let inc_sin = inc.sin();
        let raan_cos = raan.cos();
        let raan_sin = raan.sin();

        let apogee = semi_major * (1.0 + ecc);
        let perigee = semi_major * (1.0 - ecc);
        let size_ratio = apogee / perigee;
        let num_points = (200.0 * size_ratio).min(2000.0) as usize;

        (0..=num_points)
            .map(|i| {
                let true_anomaly = 2.0 * PI * i as f64 / num_points as f64;
                let r = semi_major * (1.0 - ecc * ecc) / (1.0 + ecc * true_anomaly.cos());
                let angle = true_anomaly + omega;
                let x_orbital = r * angle.cos();
                let y_orbital = -r * angle.sin();

                let x = x_orbital * raan_cos - y_orbital * inc_cos * raan_sin;
                let z = x_orbital * raan_sin + y_orbital * inc_cos * raan_cos;
                let y = -y_orbital * inc_sin;

                (x, y, z)
            })
            .collect()
    }

    /// Initialize numerical state from Keplerian elements at given time.
    pub fn initialize_numerical_state(&self, time: f64, config_hash: u64) -> NumericalState {
        let sats_per_plane = self.sats_per_plane();
        let perigee_radius = self.planet_radius + self.altitude_km;
        let ecc = self.eccentricity;
        let semi_major = perigee_radius / (1.0 - ecc);
        let period = 2.0 * PI * (semi_major.powi(3) / self.planet_mu).sqrt();
        let mean_motion = 2.0 * PI / period;
        let inc = self.inclination_deg.to_radians();
        let inc_cos = inc.cos();
        let inc_sin = inc.sin();
        let raan_step = self.raan_step();
        let raan_offset = -self.raan_offset_deg.to_radians();
        let sat_step = self.sat_step();
        let omega = self.arg_periapsis_deg.to_radians();
        let phase_step = self.phasing * 2.0 * PI / self.total_sats as f64;
        let mu = self.planet_mu;
        let p = semi_major * (1.0 - ecc * ecc);

        let r_ratio = self.planet_equatorial_radius / semi_major;
        let e2 = ecc * ecc;
        let p_factor = (1.0 - e2) * (1.0 - e2);
        let j2_coeff = 1.5 * self.planet_j2 * r_ratio * r_ratio * mean_motion;
        let raan_rate = j2_coeff * inc_cos / p_factor;
        let omega_rate = j2_coeff * (2.0 - 2.5 * inc_sin * inc_sin) / p_factor;
        let m_dot = mean_motion
            + 0.75
                * self.planet_j2
                * r_ratio
                * r_ratio
                * mean_motion
                * (3.0 * inc_cos * inc_cos - 1.0)
                * (1.0 - e2).sqrt()
                / p_factor;

        let center_offset = if self.partial_coverage() {
            raan_step * (self.num_planes - 1) as f64 / 2.0
        } else {
            0.0
        };
        let sat_center_offset = if self.partial_sat_coverage() {
            sat_step * (sats_per_plane - 1) as f64 / 2.0
        } else {
            0.0
        };
        let dead = self.altitude_km < 100.0;

        let mut sats = Vec::with_capacity(self.total_sats);
        for plane in 0..self.num_planes {
            let raan_initial = raan_offset + raan_step * plane as f64 - center_offset;
            let raan = raan_initial + if dead { 0.0 } else { raan_rate * time };
            let raan_cos = raan.cos();
            let raan_sin = raan.sin();
            let omega_t = omega + if dead { 0.0 } else { omega_rate * time };
            let phase_offset = phase_step * plane as f64;

            for sat in 0..sats_per_plane {
                let raw_mean_anomaly = sat_step * sat as f64 - sat_center_offset
                    + if dead { 0.0 } else { m_dot * time }
                    + phase_offset;
                let mean_anomaly = raw_mean_anomaly.rem_euclid(2.0 * PI);

                let true_anomaly = if ecc < 1e-8 {
                    mean_anomaly
                } else {
                    let mut ea = mean_anomaly;
                    for _ in 0..10 {
                        ea -= (ea - ecc * ea.sin() - mean_anomaly) / (1.0 - ecc * ea.cos());
                    }
                    2.0 * ((1.0 + ecc).sqrt() * (ea / 2.0).sin())
                        .atan2((1.0 - ecc).sqrt() * (ea / 2.0).cos())
                };

                let r = semi_major * (1.0 - ecc * ecc) / (1.0 + ecc * true_anomaly.cos());
                let u = true_anomaly + omega_t;

                // Position in orbital frame
                let x_o = r * u.cos();
                let y_o = -r * u.sin();

                // Position in walker coords
                let x = x_o * raan_cos - y_o * inc_cos * raan_sin;
                let z = x_o * raan_sin + y_o * inc_cos * raan_cos;
                let y = -y_o * inc_sin;

                // Velocity in orbital frame
                let h = (mu * p).sqrt();
                let r_dot = (mu / p).sqrt() * ecc * true_anomaly.sin();
                let u_dot = h / (r * r);
                let vx_o = r_dot * u.cos() - r * u_dot * u.sin();
                let vy_o = -(r_dot * u.sin() + r * u_dot * u.cos());

                // Velocity in walker coords (same rotation)
                let vx = vx_o * raan_cos - vy_o * inc_cos * raan_sin;
                let vz = vx_o * raan_sin + vy_o * inc_cos * raan_cos;
                let vy = -vy_o * inc_sin;

                sats.push(NumericalSatState {
                    pos: [x, y, z],
                    vel: [vx, vy, vz],
                });
            }
        }

        NumericalState {
            sats,
            time,
            config_hash,
        }
    }
}

/// J2 gravitational acceleration in walker coords (y = polar axis).
fn j2_acceleration(pos: &[f64; 3], mu: f64, j2: f64, re: f64) -> [f64; 3] {
    let x = pos[0];
    let y = pos[1]; // polar
    let z = pos[2];
    let r2 = x * x + y * y + z * z;
    let r = r2.sqrt();
    let r5 = r2 * r2 * r;
    let factor = -1.5 * j2 * mu * re * re / r5;
    let y2_r2 = 5.0 * y * y / r2;
    [
        factor * x * (1.0 - y2_r2),
        factor * y * (3.0 - y2_r2),
        factor * z * (1.0 - y2_r2),
    ]
}

/// Compute total acceleration (two-body + J2).
fn acceleration(pos: &[f64; 3], mu: f64, j2: f64, re: f64) -> [f64; 3] {
    let r2 = pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2];
    let r = r2.sqrt();
    let r3 = r2 * r;
    let mu_r3 = -mu / r3;
    let j2a = j2_acceleration(pos, mu, j2, re);
    [
        mu_r3 * pos[0] + j2a[0],
        mu_r3 * pos[1] + j2a[1],
        mu_r3 * pos[2] + j2a[2],
    ]
}

/// RK4 step for a single satellite. Updates pos and vel in-place.
fn rk4_step(pos: &mut [f64; 3], vel: &mut [f64; 3], dt: f64, mu: f64, j2: f64, re: f64) {
    let p0 = *pos;
    let v0 = *vel;
    let a0 = acceleration(&p0, mu, j2, re);

    let p1 = [
        p0[0] + 0.5 * dt * v0[0],
        p0[1] + 0.5 * dt * v0[1],
        p0[2] + 0.5 * dt * v0[2],
    ];
    let v1 = [
        v0[0] + 0.5 * dt * a0[0],
        v0[1] + 0.5 * dt * a0[1],
        v0[2] + 0.5 * dt * a0[2],
    ];
    let a1 = acceleration(&p1, mu, j2, re);

    let p2 = [
        p0[0] + 0.5 * dt * v1[0],
        p0[1] + 0.5 * dt * v1[1],
        p0[2] + 0.5 * dt * v1[2],
    ];
    let v2 = [
        v0[0] + 0.5 * dt * a1[0],
        v0[1] + 0.5 * dt * a1[1],
        v0[2] + 0.5 * dt * a1[2],
    ];
    let a2 = acceleration(&p2, mu, j2, re);

    let p3 = [p0[0] + dt * v2[0], p0[1] + dt * v2[1], p0[2] + dt * v2[2]];
    let v3 = [v0[0] + dt * a2[0], v0[1] + dt * a2[1], v0[2] + dt * a2[2]];
    let a3 = acceleration(&p3, mu, j2, re);

    for i in 0..3 {
        pos[i] = p0[i] + dt / 6.0 * (v0[i] + 2.0 * v1[i] + 2.0 * v2[i] + v3[i]);
        vel[i] = v0[i] + dt / 6.0 * (a0[i] + 2.0 * a1[i] + 2.0 * a2[i] + a3[i]);
    }
}

/// Step all satellites in a numerical state by sim_seconds.
/// Uses sub-steps of at most `max_step` seconds (10s for LEO).
/// Returns false if reinit is needed (time jump too large).
pub fn step_numerical_state(
    state: &mut NumericalState,
    sim_seconds: f64,
    mu: f64,
    j2: f64,
    re: f64,
) -> bool {
    if sim_seconds.abs() < 1e-12 {
        return true;
    }

    // For reverse time, signal reinit needed
    if sim_seconds < 0.0 {
        return false;
    }

    let max_step = 10.0; // seconds
    let max_total_steps = 10_000;
    let n_steps = ((sim_seconds / max_step).ceil() as usize).max(1);
    if n_steps > max_total_steps {
        return false; // time jump too large, need reinit
    }
    let dt = sim_seconds / n_steps as f64;

    for sat in &mut state.sats {
        for _ in 0..n_steps {
            rk4_step(&mut sat.pos, &mut sat.vel, dt, mu, j2, re);
        }
    }
    state.time += sim_seconds;
    true
}

/// Compute k-nearest neighbours (upper-triangular, j > i) using a uniform 3D
/// spatial grid. Falls back to a brute-force pass for tiny inputs where the
/// grid setup is not worthwhile.
///
/// `prev_neighbors`, when supplied and length-matched, applies hysteresis:
/// candidates that were already linked get a small effective-distance bonus,
/// so links don't flicker when two candidates sit at near-equal range.
pub fn compute_knn_neighbor_lists(
    positions: &[SatelliteState],
    k: usize,
    prev_neighbors: Option<&[Vec<usize>]>,
) -> Vec<Vec<usize>> {
    let n = positions.len();
    if k == 0 || n < 2 {
        return vec![Vec::new(); n];
    }

    let coords: Vec<(f64, f64, f64)> = positions.iter().map(|s| (s.x, s.y, s.z)).collect();
    let ascending: Vec<bool> = positions.iter().map(|s| s.ascending).collect();

    // 0.49 = 0.7² — a previously-linked candidate at 1.0× wins over a fresh
    // candidate up to 0.7× as close. Strong enough to pin stable links
    // through small geometric shifts at recompute boundaries.
    const HYSTERESIS_D2_FACTOR: f64 = 0.49;
    let prev_links: std::collections::HashSet<(usize, usize)> = prev_neighbors
        .filter(|p| p.len() == n)
        .map(|p| {
            let mut s = std::collections::HashSet::new();
            for (i, nbrs) in p.iter().enumerate() {
                for &j in nbrs {
                    let (a, b) = if i < j { (i, j) } else { (j, i) };
                    s.insert((a, b));
                }
            }
            s
        })
        .unwrap_or_default();
    let was_linked = |i: usize, j: usize| -> bool {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        prev_links.contains(&(a, b))
    };

    if n < 256 {
        return brute_force_knn(&coords, &ascending, k, &was_linked, HYSTERESIS_D2_FACTOR);
    }

    // Cell size targets the expected nearest-neighbour spacing on a sphere
    // covered by `n` points, so the typical 1-cell neighbourhood holds enough
    // candidates for k ≤ 8.
    let mut r = 0.0f64;
    for &(x, y, z) in &coords {
        let r2 = x * x + y * y + z * z;
        if r2 > r {
            r = r2;
        }
    }
    let r = r.sqrt().max(1.0);
    let cell = (4.0 * std::f64::consts::PI * r * r / n as f64)
        .sqrt()
        .max(1.0);

    let mut min = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    for &(x, y, z) in &coords {
        if x < min.0 {
            min.0 = x;
        }
        if y < min.1 {
            min.1 = y;
        }
        if z < min.2 {
            min.2 = z;
        }
    }

    let cell_of = |p: (f64, f64, f64)| -> (i32, i32, i32) {
        (
            ((p.0 - min.0) / cell).floor() as i32,
            ((p.1 - min.1) / cell).floor() as i32,
            ((p.2 - min.2) / cell).floor() as i32,
        )
    };

    let mut grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
        std::collections::HashMap::with_capacity(n);
    for i in 0..n {
        grid.entry(cell_of(coords[i])).or_default().push(i);
    }

    let mut out: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut scratch: Vec<(usize, f64)> = Vec::new();

    for i in 0..n {
        let (xi, yi, zi) = coords[i];
        let (cx, cy, cz) = cell_of(coords[i]);

        // Expand search radius until we have at least k candidates. Most
        // queries succeed at radius 1.
        let mut radius = 1i32;
        scratch.clear();
        loop {
            scratch.clear();
            for dx in -radius..=radius {
                for dy in -radius..=radius {
                    for dz in -radius..=radius {
                        let Some(bucket) = grid.get(&(cx + dx, cy + dy, cz + dz)) else {
                            continue;
                        };
                        for &j in bucket {
                            if j == i {
                                continue;
                            }
                            if ascending[j] != ascending[i] {
                                continue;
                            }
                            let (xj, yj, zj) = coords[j];
                            let ddx = xi - xj;
                            let ddy = yi - yj;
                            let ddz = zi - zj;
                            let mut d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                            if was_linked(i, j) {
                                d2 *= HYSTERESIS_D2_FACTOR;
                            }
                            scratch.push((j, d2));
                        }
                    }
                }
            }
            if scratch.len() >= k || radius >= 6 {
                break;
            }
            radius += 1;
        }

        let k_actual = k.min(scratch.len());
        if k_actual == 0 {
            continue;
        }
        scratch.select_nth_unstable_by(k_actual - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
        for &(j, _) in &scratch[..k_actual] {
            if j > i {
                out[i].push(j);
            }
        }
    }
    out
}

fn brute_force_knn(
    coords: &[(f64, f64, f64)],
    ascending: &[bool],
    k: usize,
    was_linked: &impl Fn(usize, usize) -> bool,
    hysteresis_d2_factor: f64,
) -> Vec<Vec<usize>> {
    let n = coords.len();
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        let (xi, yi, zi) = coords[i];
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i && ascending[j] == ascending[i])
            .map(|j| {
                let (xj, yj, zj) = coords[j];
                let dx = xi - xj;
                let dy = yi - yj;
                let dz = zi - zj;
                let mut d2 = dx * dx + dy * dy + dz * dz;
                if was_linked(i, j) {
                    d2 *= hysteresis_d2_factor;
                }
                (j, d2)
            })
            .collect();
        let k_actual = k.min(dists.len());
        if k_actual == 0 {
            continue;
        }
        dists.select_nth_unstable_by(k_actual - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
        for &(j, _) in &dists[..k_actual] {
            if j > i {
                out[i].push(j);
            }
        }
    }
    out
}
