use crate::config::{ConjunctionInfo, PredictedConjunction};
use crate::tle::TleSatellite;
use crate::walker::{SatelliteState, WalkerConstellation};
use std::collections::HashMap;

const HALF_NEIGHBORS: [(i32, i32, i32); 13] = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, -1, 0),
    (1, 0, 1),
    (1, 0, -1),
    (0, 1, 1),
    (0, 1, -1),
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, 1),
    (1, -1, -1),
];

struct SatRef {
    ci: usize,
    si: usize,
    x: f64,
    y: f64,
    z: f64,
    name: String,
    source: String,
}

struct SpatialGrid {
    inv_cell_size: f64,
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
    sats: Vec<SatRef>,
}

impl SpatialGrid {
    fn new(cell_size: f64) -> Self {
        Self {
            inv_cell_size: 1.0 / cell_size,
            cells: HashMap::new(),
            sats: Vec::new(),
        }
    }

    fn cell_key(&self, x: f64, y: f64, z: f64) -> (i32, i32, i32) {
        (
            (x * self.inv_cell_size).floor() as i32,
            (y * self.inv_cell_size).floor() as i32,
            (z * self.inv_cell_size).floor() as i32,
        )
    }

    fn insert(&mut self, sat: SatRef) {
        let key = self.cell_key(sat.x, sat.y, sat.z);
        let idx = self.sats.len();
        self.sats.push(sat);
        self.cells.entry(key).or_default().push(idx);
    }

    fn find_close_pairs(&self, threshold_km: f64) -> Vec<(&SatRef, &SatRef, f64)> {
        let thresh_sq = threshold_km * threshold_km;
        let mut results = Vec::new();

        for (&(cx, cy, cz), indices) in &self.cells {
            for i in 0..indices.len() {
                let a = &self.sats[indices[i]];
                for j in (i + 1)..indices.len() {
                    let b = &self.sats[indices[j]];
                    let d_sq = dist_sq(a, b);
                    if d_sq < thresh_sq {
                        results.push((a, b, d_sq.sqrt()));
                    }
                }
            }

            for &(dx, dy, dz) in &HALF_NEIGHBORS {
                let nk = (cx + dx, cy + dy, cz + dz);
                if let Some(neighbor_indices) = self.cells.get(&nk) {
                    for &ai in indices {
                        let a = &self.sats[ai];
                        for &bi in neighbor_indices {
                            let b = &self.sats[bi];
                            let d_sq = dist_sq(a, b);
                            if d_sq < thresh_sq {
                                results.push((a, b, d_sq.sqrt()));
                            }
                        }
                    }
                }
            }
        }
        results
    }
}

fn dist_sq(a: &SatRef, b: &SatRef) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    dx * dx + dy * dy + dz * dz
}

fn estimate_tca(
    pos_a: [f64; 3],
    pos_b: [f64; 3],
    prev_a: [f64; 3],
    prev_b: [f64; 3],
    dt: f64,
) -> (f64, f64) {
    if dt.abs() < 1e-6 {
        let dx = pos_a[0] - pos_b[0];
        let dy = pos_a[1] - pos_b[1];
        let dz = pos_a[2] - pos_b[2];
        return (0.0, (dx * dx + dy * dy + dz * dz).sqrt());
    }

    let rx = pos_a[0] - pos_b[0];
    let ry = pos_a[1] - pos_b[1];
    let rz = pos_a[2] - pos_b[2];

    let vx = (rx - (prev_a[0] - prev_b[0])) / dt;
    let vy = (ry - (prev_a[1] - prev_b[1])) / dt;
    let vz = (rz - (prev_a[2] - prev_b[2])) / dt;

    let r_dot_v = rx * vx + ry * vy + rz * vz;
    let v_dot_v = vx * vx + vy * vy + vz * vz;

    if v_dot_v < 1e-12 {
        return (0.0, (rx * rx + ry * ry + rz * rz).sqrt());
    }

    let t_min = -(r_dot_v / v_dot_v).clamp(-300.0, 300.0);

    let px = rx + vx * t_min;
    let py = ry + vy * t_min;
    let pz = rz + vz * t_min;
    (t_min, (px * px + py * py + pz * pz).sqrt())
}

pub(crate) type ConstellationData = (WalkerConstellation, Vec<SatelliteState>, usize, u8, usize, String);

pub(crate) fn compute_conjunctions(
    conjunctions: &mut Vec<ConjunctionInfo>,
    threshold_km: f64,
    constellations_data: &[ConstellationData],
    prev_positions: &mut HashMap<(usize, usize), [f64; 3]>,
    dt: f64,
) {
    let mut grid = SpatialGrid::new(threshold_km.max(1.0));

    for (ci, (_, positions, _, _, _, label)) in constellations_data.iter().enumerate() {
        for (idx, sat) in positions.iter().enumerate() {
            let name = sat.name.clone().unwrap_or_else(|| {
                format!("{} P{}:S{}", label, sat.plane, sat.sat_index)
            });
            grid.insert(SatRef {
                ci,
                si: idx,
                x: sat.x,
                y: sat.y,
                z: sat.z,
                name,
                source: label.clone(),
            });
        }
    }

    let pairs = grid.find_close_pairs(threshold_km);

    conjunctions.clear();
    for (a, b, dist) in pairs {
        if a.name == b.name { continue; }
        let key_a = (a.ci, a.si);
        let key_b = (b.ci, b.si);
        let pos_a = [a.x, a.y, a.z];
        let pos_b = [b.x, b.y, b.z];

        let (tca, min_dist) = if let (Some(&pa), Some(&pb)) = (
            prev_positions.get(&key_a),
            prev_positions.get(&key_b),
        ) {
            estimate_tca(pos_a, pos_b, pa, pb, dt)
        } else {
            (0.0, dist)
        };

        conjunctions.push(ConjunctionInfo {
            name_a: a.name.clone(),
            name_b: b.name.clone(),
            source_a: a.source.clone(),
            source_b: b.source.clone(),
            distance_km: dist,
            pos_a,
            pos_b,
            tca_seconds: tca,
            min_distance_km: min_dist,
        });
    }

    conjunctions.sort_by(|a, b| {
        a.source_a.cmp(&b.source_a)
            .then_with(|| a.source_b.cmp(&b.source_b))
            .then_with(|| a.distance_km.partial_cmp(&b.distance_km).unwrap())
            .then_with(|| a.name_a.cmp(&b.name_a))
            .then_with(|| a.name_b.cmp(&b.name_b))
    });

    let mut new_prev = HashMap::new();
    for (ci, (_, positions, _, _, _, _)) in constellations_data.iter().enumerate() {
        for (idx, sat) in positions.iter().enumerate() {
            new_prev.insert((ci, idx), [sat.x, sat.y, sat.z]);
        }
    }
    *prev_positions = new_prev;
}

pub(crate) struct TleGroup<'a> {
    pub label: String,
    pub satellites: &'a [TleSatellite],
    pub propagation_minutes: f64,
}

pub(crate) fn predict_conjunctions(
    walker_data: &[(WalkerConstellation, String)],
    tle_groups: &[TleGroup],
    current_time: f64,
    threshold_km: f64,
    window_seconds: f64,
) -> Vec<PredictedConjunction> {
    let step = 10.0_f64;
    let steps = (window_seconds / step).ceil() as usize;
    let mut best: HashMap<(String, String), (f64, f64)> = HashMap::new();
    let mut names_sources: HashMap<(String, String), (String, String)> = HashMap::new();

    for s in 0..=steps {
        let dt = s as f64 * step;
        let future_time = current_time + dt;

        let mut grid = SpatialGrid::new(threshold_km.max(1.0));

        for (wc, label) in walker_data {
            let positions = wc.satellite_positions(future_time);
            for (idx, sat) in positions.iter().enumerate() {
                let name = sat.name.clone().unwrap_or_else(|| {
                    format!("{} P{}:S{}", label, sat.plane, sat.sat_index)
                });
                grid.insert(SatRef {
                    ci: 0,
                    si: idx,
                    x: sat.x,
                    y: sat.y,
                    z: sat.z,
                    name,
                    source: label.clone(),
                });
            }
        }

        for group in tle_groups {
            let future_prop_min = group.propagation_minutes + dt / 60.0;
            for (idx, sat) in group.satellites.iter().enumerate() {
                let mins_since_epoch = future_prop_min - sat.epoch_minutes;
                let prediction = match sat.constants.propagate(
                    sgp4::MinutesSinceEpoch(mins_since_epoch)
                ) {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                let x = prediction.position[0];
                let y = prediction.position[2];
                let z = -prediction.position[1];
                grid.insert(SatRef {
                    ci: 0,
                    si: 1_000_000 + idx,
                    x, y, z,
                    name: sat.name.clone(),
                    source: group.label.clone(),
                });
            }
        }

        let pairs = grid.find_close_pairs(threshold_km);
        for (a, b, dist) in pairs {
            if a.name == b.name { continue; }
            let key = if a.name < b.name {
                (a.name.clone(), b.name.clone())
            } else {
                (b.name.clone(), a.name.clone())
            };
            let entry = best.entry(key.clone()).or_insert((f64::MAX, f64::MAX));
            if dist < entry.1 {
                *entry = (dt, dist);
                names_sources.insert(key, (a.source.clone(), b.source.clone()));
            }
        }
    }

    let mut results: Vec<PredictedConjunction> = best
        .into_iter()
        .filter_map(|((na, nb), (time_offset, min_dist))| {
            let (sa, sb) = names_sources.get(&(na.clone(), nb.clone()))?;
            Some(PredictedConjunction {
                name_a: na,
                name_b: nb,
                source_a: sa.clone(),
                source_b: sb.clone(),
                time_until: time_offset,
                min_distance_km: min_dist,
            })
        })
        .collect();

    results.sort_by(|a, b| {
        a.time_until.partial_cmp(&b.time_until).unwrap()
            .then_with(|| a.min_distance_km.partial_cmp(&b.min_distance_km).unwrap())
    });
    results
}
