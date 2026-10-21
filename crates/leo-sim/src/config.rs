use crate::{orbit::WalkerShell, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Node {
    Satellite(usize),
    Ground(usize),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Object {
    pub id: String,
    pub bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredObject {
    pub node: Node,
    pub object: Object,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DutyLimit {
    pub window_s: f64,
    pub max_fraction: f64,
    /// Uniform utilization throughout [-window_s, 0), explicitly part of scenario state.
    pub initial_utilization: f64,
}
impl Default for DutyLimit {
    fn default() -> Self {
        Self {
            window_s: 86400.0,
            max_fraction: 0.2,
            initial_utilization: 0.0,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BatteryConfig {
    pub capacity_j: f64,
    pub initial_j: f64,
    pub reserve_j: f64,
    pub solar_w: f64,
    pub idle_w: f64,
    pub max_average_load_w: f64,
    pub compute_w: f64,
    pub sensing_w: f64,
    pub isl_tx_w: f64,
    pub isl_rx_w: f64,
    pub downlink_w: f64,
}
impl Default for BatteryConfig {
    fn default() -> Self {
        Self {
            capacity_j: 100000.0,
            initial_j: 100000.0,
            reserve_j: 10000.0,
            solar_w: 50.0,
            idle_w: 10.0,
            max_average_load_w: 100.0,
            compute_w: 30.0,
            sensing_w: 10.0,
            isl_tx_w: 10.0,
            isl_rx_w: 5.0,
            downlink_w: 20.0,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SensorConfig {
    pub full_fov_deg: f64,
    pub enabled: bool,
}
impl Default for SensorConfig {
    fn default() -> Self {
        Self {
            full_fov_deg: 20.0,
            enabled: true,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SatelliteResources {
    pub compute_units_per_s: f64,
    pub storage_bytes: Option<u64>,
    pub duty: Option<DutyLimit>,
    pub battery: Option<BatteryConfig>,
    pub sensor: SensorConfig,
    pub downlink_bps: f64,
    pub downlink_terminals: usize,
}
impl Default for SatelliteResources {
    fn default() -> Self {
        Self {
            compute_units_per_s: 1.0,
            storage_bytes: None,
            duty: Some(DutyLimit::default()),
            battery: None,
            sensor: SensorConfig::default(),
            downlink_bps: 100e6,
            downlink_terminals: 1,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GroundStation {
    pub name: String,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub min_elevation_deg: f64,
    pub per_link_bps: f64,
    pub aggregate_bps: f64,
    pub max_contacts: usize,
    pub delivery_latency_s: f64,
}
impl Default for GroundStation {
    fn default() -> Self {
        Self {
            name: "ground".into(),
            lat_deg: 0.0,
            lon_deg: 0.0,
            min_elevation_deg: 10.0,
            per_link_bps: 100e6,
            aggregate_bps: 100e6,
            max_contacts: 1,
            delivery_latency_s: 0.0,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NetworkConfig {
    pub isl_bps: f64,
    pub isl_max_range_km: f64,
    pub interplane_max_lat_deg: f64,
    pub isl_terminals: usize,
    pub propagation_delay: bool,
}
impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            isl_bps: 1e9,
            isl_max_range_km: 5000.0,
            interplane_max_lat_deg: 70.0,
            isl_terminals: 4,
            propagation_delay: true,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub lat_deg: f64,
    pub lon_deg: f64,
    #[serde(default)]
    pub radius_km: f64,
    #[serde(default)]
    pub min_solar_elevation_deg: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Action {
    Compute {
        satellite: usize,
        work: f64,
        #[serde(default)]
        inputs: Vec<String>,
        output: Object,
    },
    Transfer {
        object: String,
        path: Vec<Node>,
    },
    Acquire {
        satellite: usize,
        duration_s: f64,
        target: Target,
        output: Object,
    },
    Delete {
        node: Node,
        object: String,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScheduledAction {
    pub id: String,
    pub at_s: f64,
    #[serde(flatten)]
    pub action: Action,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Scenario {
    pub schema_version: u32,
    pub epoch_unix_s: f64,
    pub step_s: f64,
    pub duration_s: f64,
    pub seed: u64,
    pub shells: Vec<WalkerShell>,
    pub satellite: SatelliteResources,
    pub overrides: BTreeMap<usize, SatelliteResources>,
    pub ground_stations: Vec<GroundStation>,
    pub network: NetworkConfig,
    pub initial_objects: Vec<StoredObject>,
    pub actions: Vec<ScheduledAction>,
    pub event_capacity: usize,
}
impl Default for Scenario {
    fn default() -> Self {
        Self {
            schema_version: 1,
            epoch_unix_s: 1791158400.0,
            step_s: 1.0,
            duration_s: 3600.0,
            seed: 0,
            shells: vec![WalkerShell::default()],
            satellite: SatelliteResources::default(),
            overrides: BTreeMap::new(),
            ground_stations: vec![GroundStation::default()],
            network: NetworkConfig::default(),
            initial_objects: vec![],
            actions: vec![],
            event_capacity: 10000,
        }
    }
}
pub(crate) fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.0
}
pub(crate) fn nonnegative(x: f64) -> bool {
    x.is_finite() && x >= 0.0
}
pub fn aligned(x: f64, step: f64) -> bool {
    x.is_finite() && ((x / step).round() - x / step).abs() < 1e-7
}
impl Scenario {
    pub fn satellite_count(&self) -> usize {
        self.shells.iter().map(WalkerShell::count).sum()
    }
    pub fn resources(&self, id: usize) -> &SatelliteResources {
        self.overrides.get(&id).unwrap_or(&self.satellite)
    }
    pub fn valid_node(&self, n: Node) -> bool {
        match n {
            Node::Satellite(i) => i < self.satellite_count(),
            Node::Ground(i) => i < self.ground_stations.len(),
        }
    }
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || !self.epoch_unix_s.is_finite()
            || !positive(self.step_s)
            || !nonnegative(self.duration_s)
            || !aligned(self.duration_s, self.step_s)
            || self.duration_s / self.step_s > (1u64 << 53) as f64
        {
            return Err("invalid version, epoch, duration or step alignment".into());
        }
        if self.shells.is_empty() {
            return Err("at least one shell is required".into());
        }
        let mut count = 0usize;
        for s in &self.shells {
            s.validate()?;
            count = count
                .checked_add(s.count())
                .ok_or("satellite count overflow")?;
        }
        if count > 1_000_000 {
            return Err("satellite count exceeds supported configuration limit".into());
        }
        for (id, r) in
            std::iter::once((0, &self.satellite)).chain(self.overrides.iter().map(|(i, r)| (*i, r)))
        {
            if id >= count
                || !positive(r.compute_units_per_s)
                || !nonnegative(r.downlink_bps)
                || !positive(r.sensor.full_fov_deg)
                || r.sensor.full_fov_deg > 180.0
            {
                return Err("invalid satellite resources".into());
            }
            if let Some(d) = &r.duty {
                if !positive(d.window_s)
                    || !aligned(d.window_s, self.step_s)
                    || d.window_s < self.step_s
                    || !nonnegative(d.max_fraction)
                    || d.max_fraction > 1.0
                    || !nonnegative(d.initial_utilization)
                    || d.initial_utilization > d.max_fraction
                {
                    return Err("invalid rolling duty budget".into());
                }
            }
            if let Some(b) = &r.battery {
                if !positive(b.capacity_j)
                    || ![
                        b.initial_j,
                        b.reserve_j,
                        b.solar_w,
                        b.idle_w,
                        b.max_average_load_w,
                        b.compute_w,
                        b.sensing_w,
                        b.isl_tx_w,
                        b.isl_rx_w,
                        b.downlink_w,
                    ]
                    .iter()
                    .all(|x| nonnegative(*x))
                    || b.reserve_j > b.initial_j
                    || b.initial_j > b.capacity_j
                    || b.max_average_load_w < b.idle_w
                {
                    return Err("invalid battery configuration".into());
                }
            }
        }
        let n = &self.network;
        if !nonnegative(n.isl_bps)
            || !positive(n.isl_max_range_km)
            || !nonnegative(n.interplane_max_lat_deg)
            || n.interplane_max_lat_deg > 90.0
            || n.isl_terminals > 4
        {
            return Err("invalid grid network configuration".into());
        }
        for g in &self.ground_stations {
            validate_point(g.lat_deg, g.lon_deg)?;
            if !nonnegative(g.min_elevation_deg)
                || g.min_elevation_deg > 90.0
                || !nonnegative(g.per_link_bps)
                || !nonnegative(g.aggregate_bps)
                || !nonnegative(g.delivery_latency_s)
            {
                return Err("invalid ground station resources".into());
            }
        }
        let mut ids = std::collections::BTreeSet::new();
        for a in &self.actions {
            if a.id.is_empty()
                || !ids.insert(&a.id)
                || !nonnegative(a.at_s)
                || !aligned(a.at_s, self.step_s)
                || a.at_s >= self.duration_s
            {
                return Err(
                    "action IDs must be unique and times must be aligned within the run".into(),
                );
            }
            match &a.action {
                Action::Compute {
                    satellite,
                    work,
                    output,
                    ..
                } => {
                    if *satellite >= count || !positive(*work) {
                        return Err("invalid compute action".into());
                    }
                    validate_object(output)?;
                }
                Action::Acquire {
                    satellite,
                    duration_s,
                    target,
                    output,
                } => {
                    if *satellite >= count
                        || !positive(*duration_s)
                        || !aligned(*duration_s, self.step_s)
                        || !nonnegative(target.radius_km)
                    {
                        return Err("invalid acquisition action".into());
                    }
                    validate_point(target.lat_deg, target.lon_deg)?;
                    if target
                        .min_solar_elevation_deg
                        .is_some_and(|v| !v.is_finite() || !(-90.0..=90.0).contains(&v))
                    {
                        return Err("invalid illumination bound".into());
                    }
                    validate_object(output)?;
                }
                Action::Transfer { path, .. } => {
                    if path.len() < 2
                        || path.iter().any(|p| !self.valid_node(*p))
                        || path[..path.len() - 1]
                            .iter()
                            .any(|p| matches!(p, Node::Ground(_)))
                        || path.windows(2).any(|p| p[0] == p[1])
                    {
                        return Err("transfer path must have distinct adjacent nodes and ground only at its end".into());
                    }
                }
                Action::Delete { node, .. } => {
                    if !self.valid_node(*node) {
                        return Err("invalid deletion node".into());
                    }
                }
            }
        }
        for o in &self.initial_objects {
            if !self.valid_node(o.node) {
                return Err("invalid initial object node".into());
            }
            validate_object(&o.object)?;
        }
        Ok(())
    }
}
fn validate_object(o: &Object) -> Result<()> {
    if o.id.is_empty() || o.bytes == 0 || o.bytes > (1u64 << 53) {
        Err("object must have an ID and 1..2^53 bytes".into())
    } else {
        Ok(())
    }
}
fn validate_point(lat: f64, lon: f64) -> Result<()> {
    if !lat.is_finite()
        || !lon.is_finite()
        || !(-90.0..=90.0).contains(&lat)
        || !(-180.0..=180.0).contains(&lon)
    {
        Err("invalid latitude/longitude".into())
    } else {
        Ok(())
    }
}
