use crate::celestial::CelestialBody;
use crate::config::{ConstellationConfig, PassInfo};
use crate::time::{body_rotation_angle, greenwich_mean_sidereal_time};
use crate::walker::WalkerConstellation;
use chrono::{DateTime, Utc};

fn haversine_dist(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * a.sqrt().asin()
}

fn elevation_from_ground(
    gs_xyz: [f64; 3],
    sat_xyz: [f64; 3],
) -> f64 {
    let r = (gs_xyz[0] * gs_xyz[0] + gs_xyz[1] * gs_xyz[1] + gs_xyz[2] * gs_xyz[2]).sqrt();
    let ux = gs_xyz[0] / r;
    let uy = gs_xyz[1] / r;
    let uz = gs_xyz[2] / r;
    let dx = sat_xyz[0] - gs_xyz[0];
    let dy = sat_xyz[1] - gs_xyz[1];
    let dz = sat_xyz[2] - gs_xyz[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    if dist < 1e-9 {
        return 90.0;
    }
    let dot = ux * dx + uy * dy + uz * dz;
    (dot / dist).asin().to_degrees()
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
    let sim_time = start_timestamp + chrono::Duration::milliseconds((t * 1000.0) as i64);
    let gmst = greenwich_mean_sidereal_time(sim_time);
    let body_rot = body_rotation_angle(body, t, gmst);
    let gs_lat_rad = gs_lat_deg.to_radians();
    let gs_lon_rad = gs_lon_deg.to_radians() + body_rot;

    let (lat, lon) = wc.single_satellite_lat_lon(plane, sat_index, t);
    let dist = haversine_dist(gs_lat_rad, gs_lon_rad, lat.to_radians(), lon.to_radians());
    dist <= max_angular_dist
}

fn bisect_transition(
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
    for _ in 0..15 {
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
    for _ in 0..15 {
        let mid = (lo + hi) * 0.5;
        if sat_in_range(wc, plane, sat_index, mid, gs_lat_deg, gs_lon_deg, max_angular_dist, body, start_timestamp) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    (lo + hi) * 0.5
}

pub(crate) fn compute_passes(
    ground_lat: f64,
    ground_lon: f64,
    radius_km: f64,
    constellations: &[ConstellationConfig],
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

    let step = 30.0_f64;
    let steps = (prediction_window_sec / step).ceil() as usize;

    struct SatTracker {
        constellation_idx: usize,
        plane: usize,
        sat_index: usize,
        in_pass: bool,
        entry_step_time: f64,
        prev_step_time: f64,
        aos_time: f64,
        max_elev: f64,
        ascending_at_aos: bool,
    }

    let total_sats: usize = constellations.iter().map(|c| c.total_sats()).sum();
    let mut trackers: Vec<SatTracker> = Vec::with_capacity(total_sats);
    for (ci, c) in constellations.iter().enumerate() {
        for p in 0..c.num_planes {
            for s in 0..c.sats_per_plane {
                trackers.push(SatTracker {
                    constellation_idx: ci,
                    plane: p,
                    sat_index: s,
                    in_pass: false,
                    entry_step_time: 0.0,
                    prev_step_time: 0.0,
                    aos_time: 0.0,
                    max_elev: 0.0,
                    ascending_at_aos: false,
                });
            }
        }
    }

    let wcs: Vec<WalkerConstellation> = constellations.iter()
        .map(|c| c.constellation(planet_radius, planet_mu, planet_j2, planet_eq_radius))
        .collect();

    let mut passes = Vec::new();

    for si in 0..=steps {
        let t = current_time + si as f64 * step;
        let sim_time = start_timestamp + chrono::Duration::seconds(t as i64);
        let gmst = greenwich_mean_sidereal_time(sim_time);
        let body_rot = body_rotation_angle(body, t, gmst);

        let gs_lat_rad = ground_lat.to_radians();
        let gs_lon_rad = ground_lon.to_radians() + body_rot;
        let gs_xyz = gs_eci_position(ground_lat, ground_lon, planet_radius, body_rot);

        let mut tracker_idx = 0;
        for (ci, c) in constellations.iter().enumerate() {
            let positions = wcs[ci].satellite_positions(t);
            for sat in &positions {
                let sat_lat_rad = sat.lat.to_radians();
                let sat_lon_rad = sat.lon.to_radians();
                let dist = haversine_dist(gs_lat_rad, gs_lon_rad, sat_lat_rad, sat_lon_rad);
                let in_range = dist <= max_angular_dist;

                let tracker = &mut trackers[tracker_idx];

                if in_range {
                    let elev = elevation_from_ground(gs_xyz, [sat.x, sat.y, sat.z]);
                    if !tracker.in_pass {
                        tracker.in_pass = true;
                        tracker.entry_step_time = t;
                        if si == 0 {
                            tracker.aos_time = t - current_time;
                        } else {
                            let precise = bisect_entry(
                                &wcs[ci], tracker.plane, tracker.sat_index,
                                tracker.prev_step_time, t,
                                ground_lat, ground_lon, max_angular_dist,
                                body, start_timestamp,
                            );
                            tracker.aos_time = precise - current_time;
                        }
                        tracker.max_elev = elev;
                        tracker.ascending_at_aos = sat.ascending;
                    } else if elev > tracker.max_elev {
                        tracker.max_elev = elev;
                    }
                } else if tracker.in_pass {
                    let precise_los = bisect_transition(
                        &wcs[ci], tracker.plane, tracker.sat_index,
                        tracker.prev_step_time, t,
                        ground_lat, ground_lon, max_angular_dist,
                        body, start_timestamp,
                    );
                    let name = format!(
                        "{} P{}:S{}",
                        c.preset_name(),
                        tracker.plane,
                        tracker.sat_index
                    );
                    passes.push(PassInfo {
                        constellation_idx: tracker.constellation_idx,
                        sat_plane: tracker.plane,
                        sat_index: tracker.sat_index,
                        sat_name: name,
                        time_to_aos: tracker.aos_time,
                        max_elevation: tracker.max_elev,
                        duration: (precise_los - current_time) - tracker.aos_time,
                        ascending: tracker.ascending_at_aos,
                    });
                    tracker.in_pass = false;
                }
                tracker.prev_step_time = t;
                tracker_idx += 1;
            }
        }
    }

    for tracker in &trackers {
        if tracker.in_pass {
            let c = &constellations[tracker.constellation_idx];
            let name = format!(
                "{} P{}:S{}",
                c.preset_name(),
                tracker.plane,
                tracker.sat_index
            );
            passes.push(PassInfo {
                constellation_idx: tracker.constellation_idx,
                sat_plane: tracker.plane,
                sat_index: tracker.sat_index,
                sat_name: name,
                time_to_aos: tracker.aos_time,
                max_elevation: tracker.max_elev,
                duration: prediction_window_sec - tracker.aos_time,
                ascending: tracker.ascending_at_aos,
            });
        }
    }

    passes.sort_by(|a, b| a.time_to_aos.partial_cmp(&b.time_to_aos).unwrap());
    passes
}
