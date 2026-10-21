use crate::{
    config::*,
    geometry::*,
    orbit::{Elements, Position},
    Result,
};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone, Copy, Debug)]
pub struct Link {
    pub bytes_per_s: f64,
    pub propagation_s: f64,
}
/// Policies may substitute an environment, e.g. measured contact traces.
/// Rates and availability are sampled at a fixed step's start and held for that step.
pub trait Environment {
    fn satellite_count(&self) -> usize;
    fn link(&self, from: Node, to: Node, time: f64) -> Option<Link>;
    fn sunlit(&self, satellite: usize, time: f64) -> bool;
    fn can_observe(
        &self,
        satellite: usize,
        target: &Target,
        sensor: &SensorConfig,
        time: f64,
    ) -> bool;
}
#[derive(Clone, Debug)]
pub struct OrbitEnvironment {
    epoch: f64,
    elements: Vec<Elements>,
    neighbors: Vec<BTreeMap<usize, bool>>,
    network: NetworkConfig,
    stations: Vec<GroundStation>,
    downlink_bps: Vec<f64>,
}
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Contact {
    pub start_s: f64,
    pub end_s: f64,
    pub capacity_bytes: f64,
}
impl OrbitEnvironment {
    pub fn new(c: &Scenario) -> Result<Self> {
        c.validate()?;
        let mut elements = Vec::new();
        let mut neighbors = vec![BTreeMap::new(); c.satellite_count()];
        let mut offset = 0;
        for s in &c.shells {
            for p in 0..s.planes {
                for k in 0..s.satellites_per_plane {
                    elements.push(s.elements(p, k));
                    let a = offset + p * s.satellites_per_plane + k;
                    let b = offset + p * s.satellites_per_plane + (k + 1) % s.satellites_per_plane;
                    if a != b {
                        neighbors[a].insert(b, false);
                        neighbors[b].insert(a, false);
                    }
                    if s.planes > 1 && (p + 1 < s.planes || !s.star) {
                        let next = (p + 1) % s.planes;
                        // Across the Delta seam, continue Walker phasing into plane zero.
                        let next_k = if p + 1 == s.planes {
                            (k + s.phasing) % s.satellites_per_plane
                        } else {
                            k
                        };
                        let b = offset + next * s.satellites_per_plane + next_k;
                        neighbors[a].insert(b, true);
                        neighbors[b].insert(a, true);
                    }
                }
            }
            offset += s.count();
        }
        Ok(Self {
            epoch: c.epoch_unix_s,
            elements,
            neighbors,
            network: c.network.clone(),
            stations: c.ground_stations.clone(),
            downlink_bps: (0..c.satellite_count())
                .map(|s| c.resources(s).downlink_bps)
                .collect(),
        })
    }
    pub fn position(&self, id: usize, t: f64) -> Position {
        self.elements[id].sample(t)
    }
    pub fn neighbor_ids(&self, id: usize) -> impl Iterator<Item = usize> + '_ {
        self.neighbors[id].keys().copied()
    }
    pub fn geographic_position(&self, id: usize, t: f64) -> (f64, f64) {
        let p = self.position(id, t);
        (
            p.lat_deg,
            (p.inertial_lon_deg - gmst(self.epoch + t).to_degrees() + 180.0).rem_euclid(360.0)
                - 180.0,
        )
    }
    pub fn solar_elevation(&self, target: &Target, t: f64) -> f64 {
        let ground = ground_position(target.lat_deg, target.lon_deg, 1.0, gmst(self.epoch + t));
        dot(ground, sun_direction(self.epoch + t))
            .clamp(-1.0, 1.0)
            .asin()
            .to_degrees()
    }
    pub fn local_solar_time(&self, target: &Target, t: f64) -> f64 {
        local_solar_time_hours(
            ground_position(target.lat_deg, target.lon_deg, 1.0, gmst(self.epoch + t)),
            sun_direction(self.epoch + t),
        )
    }
    /// Contact intervals use exactly the same sampled link predicate as execution.
    pub fn contacts(
        &self,
        from: Node,
        to: Node,
        start: f64,
        end: f64,
        step: f64,
    ) -> Result<Vec<Contact>> {
        if !step.is_finite() || step <= 0.0 || !start.is_finite() || !end.is_finite() || end < start
        {
            return Err("invalid contact interval".into());
        }
        let mut out = Vec::new();
        let mut current: Option<Contact> = None;
        let mut i = 0u64;
        loop {
            let t = start + i as f64 * step;
            if t >= end {
                break;
            }
            let dt = (end - t).min(step);
            if let Some(l) = self.link(from, to, t) {
                let v = current.get_or_insert(Contact {
                    start_s: t,
                    end_s: t,
                    capacity_bytes: 0.0,
                });
                v.end_s = t + dt;
                v.capacity_bytes += l.bytes_per_s * dt;
            } else if let Some(v) = current.take() {
                out.push(v);
            }
            i += 1;
        }
        if let Some(v) = current {
            out.push(v);
        }
        Ok(out)
    }
}
impl Environment for OrbitEnvironment {
    fn satellite_count(&self) -> usize {
        self.elements.len()
    }
    fn link(&self, from: Node, to: Node, t: f64) -> Option<Link> {
        let (bps, range, extra) = match (from, to) {
            (Node::Satellite(a), Node::Satellite(b)) => {
                let inter = *self.neighbors.get(a)?.get(&b)?;
                let p = self.position(a, t);
                let q = self.position(b, t);
                let distance = distance(p.xyz, q.xyz);
                if (inter
                    && (p.lat_deg.abs() > self.network.interplane_max_lat_deg
                        || q.lat_deg.abs() > self.network.interplane_max_lat_deg))
                    || distance > self.network.isl_max_range_km
                    || !clear_line(p.xyz, q.xyz, EARTH_RADIUS_KM)
                {
                    return None;
                }
                (self.network.isl_bps, distance, 0.0)
            }
            (Node::Satellite(s), Node::Ground(g)) => {
                let station = self.stations.get(g)?;
                let p = self.elements.get(s)?.sample(t);
                let ground = ground_position(
                    station.lat_deg,
                    station.lon_deg,
                    EARTH_RADIUS_KM,
                    gmst(self.epoch + t),
                );
                if elevation(ground, p.xyz) < station.min_elevation_deg {
                    return None;
                }
                (
                    station.per_link_bps.min(self.downlink_bps[s]),
                    distance(ground, p.xyz),
                    station.delivery_latency_s,
                )
            }
            _ => return None,
        };
        if bps <= 0.0 {
            return None;
        }
        Some(Link {
            bytes_per_s: bps / 8.0,
            propagation_s: extra
                + if self.network.propagation_delay {
                    range / LIGHT_KM_S
                } else {
                    0.0
                },
        })
    }
    fn sunlit(&self, s: usize, t: f64) -> bool {
        !eclipsed(
            self.position(s, t).xyz,
            sun_direction(self.epoch + t),
            EARTH_RADIUS_KM,
        )
    }
    fn can_observe(&self, s: usize, target: &Target, sensor: &SensorConfig, t: f64) -> bool {
        if !sensor.enabled {
            return false;
        }
        let pos = self.position(s, t).xyz;
        let ground = ground_position(
            target.lat_deg,
            target.lon_deg,
            EARTH_RADIUS_KM,
            gmst(self.epoch + t),
        );
        let angular_radius = target.radius_km / EARTH_RADIUS_KM;
        let footprint = footprint_radius_rad(
            norm(pos) - EARTH_RADIUS_KM,
            sensor.full_fov_deg,
            EARTH_RADIUS_KM,
        );
        if angular_distance(pos, ground) + angular_radius > footprint + 1e-12 {
            return false;
        }
        // The cap's worst solar elevation differs from its center by at most its angular radius.
        target
            .min_solar_elevation_deg
            .is_none_or(|min| self.solar_elevation(target, t) - angular_radius.to_degrees() >= min)
    }
}
/// Deterministic FIFO allocation, full duplex ISLs, station and terminal sharing.
/// Terminal assignments last for a step; partially sent objects retain progress.
pub(crate) struct NetworkBudget {
    links: BTreeMap<(Node, Node), f64>,
    station_remaining: Vec<f64>,
    sat_remaining: Vec<f64>,
    station_peers: Vec<BTreeSet<usize>>,
    sat_ground_peers: Vec<BTreeSet<usize>>,
    isl_peers: Vec<BTreeSet<usize>>,
    dt: f64,
}
impl NetworkBudget {
    pub fn new(c: &Scenario, dt: f64) -> Self {
        Self {
            links: BTreeMap::new(),
            station_remaining: c
                .ground_stations
                .iter()
                .map(|g| g.aggregate_bps / 8.0 * dt)
                .collect(),
            sat_remaining: (0..c.satellite_count())
                .map(|s| c.resources(s).downlink_bps / 8.0 * dt)
                .collect(),
            station_peers: vec![BTreeSet::new(); c.ground_stations.len()],
            sat_ground_peers: vec![BTreeSet::new(); c.satellite_count()],
            isl_peers: vec![BTreeSet::new(); c.satellite_count()],
            dt,
        }
    }
    pub fn available(&self, c: &Scenario, a: Node, b: Node, l: Link) -> f64 {
        let base = self
            .links
            .get(&(a, b))
            .copied()
            .unwrap_or(l.bytes_per_s * self.dt);
        match (a, b) {
            (Node::Satellite(s), Node::Ground(g)) => {
                if (!self.station_peers[g].contains(&s)
                    && self.station_peers[g].len() >= c.ground_stations[g].max_contacts)
                    || (!self.sat_ground_peers[s].contains(&g)
                        && self.sat_ground_peers[s].len() >= c.resources(s).downlink_terminals)
                {
                    return 0.0;
                }
                base.min(self.station_remaining[g])
                    .min(self.sat_remaining[s])
            }
            (Node::Satellite(s), Node::Satellite(d)) => {
                if (!self.isl_peers[s].contains(&d)
                    && self.isl_peers[s].len() >= c.network.isl_terminals)
                    || (!self.isl_peers[d].contains(&s)
                        && self.isl_peers[d].len() >= c.network.isl_terminals)
                {
                    0.0
                } else {
                    base
                }
            }
            _ => 0.0,
        }
    }
    pub fn spend(&mut self, a: Node, b: Node, l: Link, bytes: f64) {
        *self.links.entry((a, b)).or_insert(l.bytes_per_s * self.dt) -= bytes;
        match (a, b) {
            (Node::Satellite(s), Node::Ground(g)) => {
                self.station_remaining[g] -= bytes;
                self.sat_remaining[s] -= bytes;
                self.station_peers[g].insert(s);
                self.sat_ground_peers[s].insert(g);
            }
            (Node::Satellite(s), Node::Satellite(d)) => {
                self.isl_peers[s].insert(d);
                self.isl_peers[d].insert(s);
            }
            _ => {}
        }
    }
}
