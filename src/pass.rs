use crate::celestial::CelestialBody;
use crate::config::{ConstellationConfig, PassInfo};
use crate::time::{body_rotation_angle, greenwich_mean_sidereal_time};
use crate::walker::WalkerConstellation;
use chrono::{DateTime, Utc};
use std::f64::consts::PI;

fn true_anomaly_to_mean_anomaly(v: f64, ecc: f64) -> f64 {
    if ecc < 1e-8 { return v; }
    let e_anom = 2.0 * ((1.0 - ecc).sqrt() * (v / 2.0).sin())
        .atan2((1.0 + ecc).sqrt() * (v / 2.0).cos());
    (e_anom - ecc * e_anom.sin()).rem_euclid(2.0 * PI)
}

fn haversine_dist(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * a.sqrt().asin()
}

fn sat_angular_dist(
    wc: &WalkerConstellation,
    plane: usize,
    sat_index: usize,
    t: f64,
    gs_lat_deg: f64,
    gs_lon_deg: f64,
    body: CelestialBody,
    start_timestamp: DateTime<Utc>,
) -> f64 {
    let sim_time = start_timestamp + chrono::Duration::milliseconds((t * 1000.0) as i64);
    let gmst = greenwich_mean_sidereal_time(sim_time);
    let body_rot = body_rotation_angle(body, t, gmst);
    let gs_lat_rad = gs_lat_deg.to_radians();
    let gs_lon_rad = gs_lon_deg.to_radians() + body_rot;
    let (lat, lon, _) = wc.single_satellite_lat_lon(plane, sat_index, t);
    haversine_dist(gs_lat_rad, gs_lon_rad, lat.to_radians(), lon.to_radians())
}

fn sat_in_range(
    wc: &WalkerConstellation,
    plane: usize,
    sat_index: usize,
    t: f64,
    gs_lat_deg: f64,
    gs_lon_deg: f64,
    max_angular_dist: f64,
    body: CelestialBody,
    start_timestamp: DateTime<Utc>,
) -> bool {
    sat_angular_dist(wc, plane, sat_index, t, gs_lat_deg, gs_lon_deg, body, start_timestamp)
        <= max_angular_dist
}

fn bisect_exit(
    wc: &WalkerConstellation,
    plane: usize,
    sat_index: usize,
    t_in: f64,
    t_out: f64,
    gs_lat_deg: f64,
    gs_lon_deg: f64,
    max_angular_dist: f64,
    body: CelestialBody,
    start_timestamp: DateTime<Utc>,
) -> f64 {
    let mut lo = t_in;
    let mut hi = t_out;
    for _ in 0..20 {
        let mid = (lo + hi) * 0.5;
        if sat_in_range(wc, plane, sat_index, mid, gs_lat_deg, gs_lon_deg, max_angular_dist, body, start_timestamp) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) * 0.5
}

fn bisect_entry(
    wc: &WalkerConstellation,
    plane: usize,
    sat_index: usize,
    t_out: f64,
    t_in: f64,
    gs_lat_deg: f64,
    gs_lon_deg: f64,
    max_angular_dist: f64,
    body: CelestialBody,
    start_timestamp: DateTime<Utc>,
) -> f64 {
    let mut lo = t_out;
    let mut hi = t_in;
    for _ in 0..20 {
        let mid = (lo + hi) * 0.5;
        if sat_in_range(wc, plane, sat_index, mid, gs_lat_deg, gs_lon_deg, max_angular_dist, body, start_timestamp) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    (lo + hi) * 0.5
}

fn gs_eci_position(
    lat_deg: f64,
    lon_deg: f64,
    planet_radius: f64,
    body_rot: f64,
) -> [f64; 3] {
    let lat = lat_deg.to_radians();
    let lon = (-lon_deg).to_radians() - body_rot;
    let x = planet_radius * lat.cos() * lon.cos();
    let y = planet_radius * lat.sin();
    let z = planet_radius * lat.cos() * lon.sin();
    [x, y, z]
}

fn elevation_from_ground(gs_xyz: [f64; 3], sat_xyz: [f64; 3]) -> f64 {
    let r = (gs_xyz[0] * gs_xyz[0] + gs_xyz[1] * gs_xyz[1] + gs_xyz[2] * gs_xyz[2]).sqrt();
    let ux = gs_xyz[0] / r;
    let uy = gs_xyz[1] / r;
    let uz = gs_xyz[2] / r;
    let dx = sat_xyz[0] - gs_xyz[0];
    let dy = sat_xyz[1] - gs_xyz[1];
    let dz = sat_xyz[2] - gs_xyz[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    if dist < 1e-9 { return 90.0; }
    let dot = ux * dx + uy * dy + uz * dz;
    (dot / dist).asin().to_degrees()
}

fn refine_pass_details(
    wc: &WalkerConstellation,
    plane: usize,
    sat_index: usize,
    t_start: f64,
    t_end: f64,
    ground_lat: f64,
    ground_lon: f64,
    planet_radius: f64,
    body: CelestialBody,
    start_timestamp: DateTime<Utc>,
) -> (f64, f64) {
    let steps = 20;
    let dt = (t_end - t_start) / steps as f64;
    let mut max_elev = f64::NEG_INFINITY;
    let mut alt_at_max = 0.0;
    for i in 0..=steps {
        let t = t_start + i as f64 * dt;
        let sim_time = start_timestamp + chrono::Duration::milliseconds((t * 1000.0) as i64);
        let gmst = greenwich_mean_sidereal_time(sim_time);
        let body_rot = body_rotation_angle(body, t, gmst);
        let gs_xyz = gs_eci_position(ground_lat, ground_lon, planet_radius, body_rot);
        let (_, _, sat_xyz) = wc.single_satellite_lat_lon(plane, sat_index, t);
        let elev = elevation_from_ground(gs_xyz, sat_xyz);
        if elev > max_elev {
            max_elev = elev;
            let r = (sat_xyz[0] * sat_xyz[0] + sat_xyz[1] * sat_xyz[1] + sat_xyz[2] * sat_xyz[2]).sqrt();
            alt_at_max = r - planet_radius;
        }
    }
    (max_elev, alt_at_max)
}

fn find_pass_around(
    wc: &WalkerConstellation,
    plane: usize,
    sat_index: usize,
    t_cross: f64,
    gs_lat_deg: f64,
    gs_lon_deg: f64,
    max_angular_dist: f64,
    body: CelestialBody,
    start_timestamp: DateTime<Utc>,
    t_min: f64,
    t_max: f64,
) -> Option<(f64, f64)> {
    let scan_step = 2.0;
    let scan_half = 600.0;
    let lo = (t_cross - scan_half).max(t_min);
    let hi = (t_cross + scan_half).min(t_max);

    let mut in_pass = false;
    let mut aos = 0.0;
    let mut result: Option<(f64, f64)> = None;
    let mut prev_t = lo;
    let mut t = lo;
    while t <= hi {
        let ir = sat_in_range(
            wc, plane, sat_index, t,
            gs_lat_deg, gs_lon_deg, max_angular_dist,
            body, start_timestamp,
        );
        if ir && !in_pass {
            in_pass = true;
            aos = if t == lo {
                t
            } else {
                bisect_entry(wc, plane, sat_index, prev_t, t,
                    gs_lat_deg, gs_lon_deg, max_angular_dist, body, start_timestamp)
            };
        } else if !ir && in_pass {
            let los = bisect_exit(wc, plane, sat_index, prev_t, t,
                gs_lat_deg, gs_lon_deg, max_angular_dist, body, start_timestamp);
            result = Some((aos, los));
            in_pass = false;
            break;
        }
        prev_t = t;
        t += scan_step;
    }
    if in_pass {
        result = Some((aos, hi));
    }
    result
}

pub(crate) fn compute_passes(
    ground_lat: f64,
    ground_lon: f64,
    radius_km: f64,
    constellations: &[ConstellationConfig],
    selected_sats: &[(usize, usize, usize)],
    current_time: f64,
    prediction_window_sec: f64,
    body: CelestialBody,
    start_timestamp: DateTime<Utc>,
) -> Vec<PassInfo> {
    let planet_radius = body.radius_km();
    let planet_mu = body.mu();
    let planet_j2 = body.j2();
    let planet_eq_radius = body.equatorial_radius_km();
    let max_angular_dist = radius_km / planet_radius;
    let gs_lat_rad = ground_lat.to_radians();
    let t_end = current_time + prediction_window_sec;

    let wcs: Vec<WalkerConstellation> = constellations.iter()
        .map(|c| c.constellation(planet_radius, planet_mu, planet_j2, planet_eq_radius))
        .collect();

    let mut passes = Vec::new();

    for &(ci, plane, sat_idx) in selected_sats {
        let c = &constellations[ci];
        let wc = &wcs[ci];
        let inc = c.inclination.to_radians();
        let omega = c.arg_periapsis.to_radians();
        let ecc = c.eccentricity;

        if inc.abs() < 1e-10 { continue; }

        let k = gs_lat_rad.sin() / inc.sin();
        if k.abs() > 1.0 && gs_lat_rad.abs() > inc.abs() + max_angular_dist {
            continue;
        }

        let perigee_radius = planet_radius + c.altitude_km;
        let semi_major = perigee_radius / (1.0 - ecc);
        let period = 2.0 * PI * (semi_major.powi(3) / planet_mu).sqrt();
        let mean_motion = 2.0 * PI / period;
        let spp = c.sats_per_plane;
        let sat_step_angle = 2.0 * PI / spp as f64;
        let phase_step = c.phasing * 2.0 * PI / c.total_sats() as f64;
        let m0 = sat_step_angle * sat_idx as f64 + phase_step * plane as f64;

        let u_checks: Vec<f64> = if k.abs() <= 1.0 {
            vec![k.asin(), PI - k.asin()]
        } else {
            if gs_lat_rad > 0.0 { vec![PI / 2.0] } else { vec![3.0 * PI / 2.0] }
        };

        for &u_check in &u_checks {
            let v_cross = (u_check - omega).rem_euclid(2.0 * PI);
            let m_cross = true_anomaly_to_mean_anomaly(v_cross, ecc);
            let dm = (m_cross - m0).rem_euclid(2.0 * PI);
            let t_first = dm / mean_motion;

            let mut t_cross = t_first;
            if t_cross < current_time {
                let orbits = ((current_time - t_cross) / period).ceil();
                t_cross += orbits * period;
            }

            while t_cross < t_end {
                let dist = sat_angular_dist(
                    wc, plane, sat_idx, t_cross,
                    ground_lat, ground_lon, body, start_timestamp,
                );

                if dist <= max_angular_dist * 3.0 {
                    if let Some((aos, los)) = find_pass_around(
                        wc, plane, sat_idx, t_cross,
                        ground_lat, ground_lon, max_angular_dist,
                        body, start_timestamp,
                        current_time, t_end,
                    ) {
                        let (lat, _, _) = wc.single_satellite_lat_lon(plane, sat_idx, aos);
                        let sin_ratio = lat.to_radians().sin() / inc.sin();
                        let ascending = if sin_ratio.abs() > 1.0 {
                            true
                        } else {
                            (sin_ratio.asin() + omega).cos() > 0.0
                        };
                        let (max_elev, altitude_km) = refine_pass_details(
                            wc, plane, sat_idx, aos, los,
                            ground_lat, ground_lon, planet_radius,
                            body, start_timestamp,
                        );
                        passes.push(PassInfo {
                            constellation_idx: ci,
                            sat_plane: plane,
                            sat_index: sat_idx,
                            sat_name: format!("{} P{}:S{}", c.preset_name(), plane, sat_idx),
                            time_to_aos: aos - current_time,
                            max_elevation: max_elev,
                            duration: los - aos,
                            ascending,
                            altitude_km,
                        });
                    }
                }

                t_cross += period;
            }
        }
    }

    passes.retain(|p| {
        let c = &constellations[p.constellation_idx];
        if c.eccentricity < 0.01 {
            return true;
        }
        let perigee_alt = c.altitude_km;
        let perigee_radius = planet_radius + perigee_alt;
        let semi_major = perigee_radius / (1.0 - c.eccentricity);
        let apogee_alt = semi_major * (1.0 + c.eccentricity) - planet_radius;
        let threshold = (perigee_alt + apogee_alt) / 2.0;
        p.altitude_km <= threshold
    });

    passes.sort_by(|a, b| a.time_to_aos.partial_cmp(&b.time_to_aos).unwrap());
    let mut seen = std::collections::HashSet::new();
    passes.retain(|p| seen.insert((p.constellation_idx, p.sat_plane, p.sat_index)));
    passes
}
