use std::f64::consts::PI;

#[derive(Clone, Copy, PartialEq)]
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
    pub eccentricity: f64,
    pub arg_periapsis_deg: f64,
    pub planet_radius: f64,
    pub planet_mu: f64,
    pub planet_j2: f64,
    pub planet_equatorial_radius: f64,
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
    pub neighbor_idx: Option<usize>,
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

    pub fn satellite_positions(&self, time: f64) -> Vec<SatelliteState> {
        let mut positions = Vec::with_capacity(self.total_sats);
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
        let sat_step = 2.0 * PI / sats_per_plane as f64;
        let is_star = self.walker_type == WalkerType::Star;
        let omega = self.arg_periapsis_deg.to_radians();

        let phase_step = self.phasing * 2.0 * PI / self.total_sats as f64;

        let r_ratio = self.planet_equatorial_radius / semi_major;
        let raan_drift_rate = -1.5 * self.planet_j2 * r_ratio * r_ratio * mean_motion * inc_cos;

        let center_offset = if self.raan_spacing_deg.is_some() {
            raan_step * (self.num_planes - 1) as f64 / 2.0
        } else {
            0.0
        };
        let dead = self.altitude_km < 100.0;
        for plane in 0..self.num_planes {
            let raan_initial = raan_offset + raan_step * plane as f64 - center_offset;
            let raan = raan_initial + if dead { 0.0 } else { raan_drift_rate * time };
            let raan_cos = raan.cos();
            let raan_sin = raan.sin();
            let phase_offset = phase_step * plane as f64;

            for sat in 0..sats_per_plane {
                let mean_anomaly = sat_step * sat as f64 + if dead { 0.0 } else { mean_motion * time } + phase_offset;

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
                let ascending = (true_anomaly + omega).cos() > 0.0;

                let angle = true_anomaly + omega;
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
                    neighbor_idx: None,
                    name: None,
                    tle_inclination_deg: None,
                    tle_mean_motion: None,
                });
            }
        }

        let no_wrap = is_star || self.raan_spacing_deg.is_some();
        for i in 0..positions.len() {
            let sat = &positions[i];
            if no_wrap && sat.plane == self.num_planes - 1 {
                continue;
            }
            let next_plane = (sat.plane + 1) % self.num_planes;
            let next_plane_start = next_plane * sats_per_plane;
            let next_plane_end = next_plane_start + sats_per_plane;
            let target_idx = sat.sat_index;
            for j in next_plane_start..next_plane_end {
                let other = &positions[j];
                if other.sat_index == target_idx {
                    positions[i].neighbor_idx = Some(j);
                    break;
                }
            }
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
        let raan_drift_rate = -1.5 * self.planet_j2 * r_ratio * r_ratio * mean_motion * inc.cos();
        let omega = self.arg_periapsis_deg.to_radians();

        let raan_step = self.raan_step();
        let center_offset = if self.raan_spacing_deg.is_some() {
            raan_step * (self.num_planes - 1) as f64 / 2.0
        } else {
            0.0
        };
        let raan_initial = -self.raan_offset_deg.to_radians() + raan_step * plane as f64 - center_offset;
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
}
