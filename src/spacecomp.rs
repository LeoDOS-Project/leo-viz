use crate::drawing::compute_path_direction;
use crate::walker::{WalkerConstellation, SatelliteState};

pub struct SpaceCompResult {
    pub collectors: Vec<(usize, usize)>,
    pub mappers: Vec<(usize, usize)>,
    pub assignments: Vec<(usize, usize)>,
    pub reducer: (usize, usize),
    pub gs_sat: (usize, usize),
}

fn manhattan_hop_count(
    p1: usize, s1: usize,
    p2: usize, s2: usize,
    num_planes: usize, sats_per_plane: usize,
    is_star: bool,
) -> usize {
    let (_, plane_hops) = compute_path_direction(p1, p2, num_planes, is_star);
    let (_, sat_hops) = compute_path_direction(s1, s2, sats_per_plane, false);
    plane_hops + sat_hops
}

fn lapjv_assignment(cost: &[Vec<usize>]) -> Vec<(usize, usize)> {
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }

    const UNASSIGNED: usize = usize::MAX;
    let mut col_for_row = vec![UNASSIGNED; n];
    let mut row_for_col = vec![UNASSIGNED; n];
    let mut u = vec![0i64; n];
    let mut v = vec![0i64; n];

    for j in 0..n {
        v[j] = cost[0][j] as i64;
        for i in 1..n {
            v[j] = v[j].min(cost[i][j] as i64);
        }
    }
    for i in 0..n {
        u[i] = cost[i][0] as i64 - v[0];
        for j in 1..n {
            u[i] = u[i].min(cost[i][j] as i64 - v[j]);
        }
    }

    for i in 0..n {
        for j in 0..n {
            if col_for_row[i] == UNASSIGNED
                && row_for_col[j] == UNASSIGNED
                && cost[i][j] as i64 == u[i] + v[j]
            {
                col_for_row[i] = j;
                row_for_col[j] = i;
                break;
            }
        }
    }

    for free_row in 0..n {
        if col_for_row[free_row] != UNASSIGNED {
            continue;
        }

        let mut dist = vec![i64::MAX; n];
        let mut pred = vec![UNASSIGNED; n];
        let mut visited = vec![false; n];

        for j in 0..n {
            dist[j] = cost[free_row][j] as i64 - u[free_row] - v[j];
            pred[j] = free_row;
        }

        let mut end_col;
        loop {
            let mut min_dist = i64::MAX;
            end_col = UNASSIGNED;
            for j in 0..n {
                if !visited[j] && dist[j] < min_dist {
                    min_dist = dist[j];
                    end_col = j;
                }
            }

            visited[end_col] = true;
            if row_for_col[end_col] == UNASSIGNED {
                break;
            }

            let scan_row = row_for_col[end_col];
            for j in 0..n {
                if visited[j] { continue; }
                let new_dist = cost[scan_row][j] as i64
                    - u[scan_row] - v[j] + min_dist;
                if new_dist < dist[j] {
                    dist[j] = new_dist;
                    pred[j] = scan_row;
                }
            }
        }

        let final_min = dist[end_col];
        for j in 0..n {
            if visited[j] && j != end_col {
                v[j] += dist[j] - final_min;
                if row_for_col[j] != UNASSIGNED {
                    u[row_for_col[j]] = cost[row_for_col[j]][j] as i64 - v[j];
                }
            }
        }
        v[end_col] += dist[end_col] - final_min;

        let mut j = end_col;
        loop {
            let i = pred[j];
            row_for_col[j] = i;
            let prev_j = col_for_row[i];
            col_for_row[i] = j;
            if i == free_row { break; }
            j = prev_j;
        }
        u[free_row] = cost[free_row][col_for_row[free_row]] as i64
            - v[col_for_row[free_row]];
    }

    (0..n).map(|i| (i, col_for_row[i])).collect()
}

pub fn compute_spacecomp_job(
    aoi_lat: f64, aoi_lon: f64, aoi_radius_km: f64,
    gs_lat: f64, gs_lon: f64, gs_radius_km: f64,
    positions: &[SatelliteState],
    constellation: &WalkerConstellation,
    is_star: bool,
    planet_radius: f64, body_rot_angle: f64,
    n: usize,
) -> Option<SpaceCompResult> {
    let num_planes = constellation.num_planes;
    let sats_per_plane = constellation.sats_per_plane();

    let haversine_dist = |sat: &SatelliteState, center_lat: f64, center_lon: f64| -> f64 {
        let center_lat_rad = center_lat.to_radians();
        let center_lon_rad = center_lon.to_radians() + body_rot_angle;
        let sat_lat_rad = sat.lat.to_radians();
        let sat_lon_rad = sat.lon.to_radians();
        let dlat = sat_lat_rad - center_lat_rad;
        let dlon = sat_lon_rad - center_lon_rad;
        let a = (dlat / 2.0).sin().powi(2)
            + center_lat_rad.cos() * sat_lat_rad.cos() * (dlon / 2.0).sin().powi(2);
        2.0 * a.sqrt().asin()
    };

    let aoi_max_angular = aoi_radius_km / planet_radius;
    let gs_max_angular = gs_radius_km / planet_radius;

    let try_ascending = |asc_filter: bool| -> Option<SpaceCompResult> {
        let mut collector_candidates: Vec<((usize, usize), f64)> = positions.iter()
            .filter(|s| s.ascending == asc_filter)
            .filter(|s| haversine_dist(s, aoi_lat, aoi_lon) <= aoi_max_angular)
            .map(|s| ((s.plane, s.sat_index), haversine_dist(s, aoi_lat, aoi_lon)))
            .collect();

        if collector_candidates.is_empty() {
            return None;
        }

        collector_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let collectors: Vec<(usize, usize)> = collector_candidates.iter()
            .take(n)
            .map(|&(ps, _)| ps)
            .collect();
        let n = collectors.len();
        let collector_set: std::collections::HashSet<(usize, usize)> =
            collectors.iter().copied().collect();

        let gs_sat = positions.iter()
            .filter(|s| s.ascending == asc_filter)
            .filter(|s| haversine_dist(s, gs_lat, gs_lon) <= gs_max_angular)
            .min_by(|a, b| {
                let da = haversine_dist(a, aoi_lat, aoi_lon);
                let db = haversine_dist(b, aoi_lat, aoi_lon);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| (s.plane, s.sat_index))?;

        let (gp, gsi) = gs_sat;

        let pool_size = (n * 3).max(n + 4);
        let mut candidates: Vec<((usize, usize), (usize, usize))> = Vec::new();
        for max_hops in 1.. {
            candidates = positions.iter()
                .filter(|s| s.ascending == asc_filter)
                .filter(|s| !collector_set.contains(&(s.plane, s.sat_index)))
                .filter(|s| (s.plane, s.sat_index) != gs_sat)
                .filter(|s| {
                    collectors.iter().any(|&(cp, cs)| manhattan_hop_count(
                        s.plane, s.sat_index, cp, cs,
                        num_planes, sats_per_plane, is_star,
                    ) <= max_hops)
                })
                .map(|s| {
                    let to_collector = collectors.iter()
                        .map(|&(cp, cs)| manhattan_hop_count(
                            s.plane, s.sat_index, cp, cs,
                            num_planes, sats_per_plane, is_star,
                        ))
                        .min()
                        .unwrap();
                    let to_gs = manhattan_hop_count(
                        s.plane, s.sat_index, gp, gsi,
                        num_planes, sats_per_plane, is_star,
                    );
                    ((s.plane, s.sat_index), (to_collector, to_gs))
                })
                .collect();
            if candidates.len() >= pool_size { break; }
        }

        candidates.sort_by_key(|&(_, key)| key);
        let pool: Vec<(usize, usize)> = candidates.iter()
            .take(pool_size.min(candidates.len()))
            .map(|&(ps, _)| ps)
            .collect();

        if pool.len() < n {
            return None;
        }

        let m = pool.len();
        let big_cost: Vec<Vec<usize>> = collectors.iter()
            .map(|&(cp, cs)| {
                pool.iter()
                    .map(|&(mp, ms)| manhattan_hop_count(
                        cp, cs, mp, ms,
                        num_planes, sats_per_plane, is_star,
                    ))
                    .collect()
            })
            .collect();

        let pad_cost: Vec<Vec<usize>> = (0..m).map(|i| {
            (0..m).map(|j| {
                if i < n { big_cost[i][j] } else { 0 }
            }).collect()
        }).collect();

        let full_assignments = lapjv_assignment(&pad_cost);
        let mut used_mappers = std::collections::HashSet::new();
        let mut assignments: Vec<(usize, usize)> = Vec::new();
        let mut mapper_indices: Vec<usize> = Vec::new();
        for &(row, col) in &full_assignments {
            if row < n {
                used_mappers.insert(col);
                mapper_indices.push(col);
                assignments.push((row, assignments.len()));
            }
        }

        let mappers: Vec<(usize, usize)> = mapper_indices.iter()
            .map(|&ci| pool[ci])
            .collect();

        if mappers.len() < n {
            return None;
        }

        let mapper_set: std::collections::HashSet<(usize, usize)> =
            mappers.iter().copied().collect();

        let reducer = positions.iter()
            .filter(|s| s.ascending == asc_filter)
            .filter(|s| !collector_set.contains(&(s.plane, s.sat_index)))
            .filter(|s| !mapper_set.contains(&(s.plane, s.sat_index)))
            .filter(|s| (s.plane, s.sat_index) != gs_sat)
            .min_by_key(|s| {
                let to_mapper = mappers.iter()
                    .map(|&(mp, ms)| manhattan_hop_count(
                        s.plane, s.sat_index, mp, ms,
                        num_planes, sats_per_plane, is_star,
                    ))
                    .min()
                    .unwrap();
                let to_gs = manhattan_hop_count(
                    s.plane, s.sat_index, gp, gsi,
                    num_planes, sats_per_plane, is_star,
                );
                (to_mapper, to_gs)
            })
            .map(|s| (s.plane, s.sat_index))?;

        Some(SpaceCompResult {
            collectors,
            mappers,
            assignments,
            reducer,
            gs_sat,
        })
    };

    try_ascending(true).or_else(|| try_ascending(false))
}
