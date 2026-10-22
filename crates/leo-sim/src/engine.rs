use crate::{
    config::*,
    network::{Environment, NetworkBudget},
    resources::{PowerBudget, RollingDuty},
    storage::Storage,
    Result,
};
use serde::Serialize;
use std::collections::{BTreeSet, VecDeque};

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "state", content = "reason", rename_all = "snake_case")]
pub enum OperationStatus {
    Pending,
    Queued,
    Running,
    InTransit,
    Completed,
    Rejected(String),
}
#[derive(Clone, Debug, Serialize)]
pub struct OperationReport {
    pub id: String,
    pub status: OperationStatus,
    pub started_s: Option<f64>,
    pub finished_s: Option<f64>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Event {
    pub time_s: f64,
    pub action_id: String,
    pub kind: String,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct Metrics {
    pub compute_work: f64,
    pub compute_busy_s: f64,
    pub sensing_busy_s: f64,
    pub isl_byte_hops: f64,
    pub downlink_bytes: f64,
    pub energy_load_j: f64,
    pub reserve_deficit_j: f64,
    pub completed_actions: u64,
    pub rejected_actions: u64,
}
#[derive(Clone, Debug, Serialize)]
pub struct SatelliteReport {
    pub id: usize,
    pub stored_and_reserved_bytes: u64,
    pub compute_busy_s: f64,
    pub rolling_compute_busy_s: f64,
    pub energy_j: Option<f64>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub schema_version: u32,
    pub time_s: f64,
    pub metrics: Metrics,
    pub satellites: Vec<SatelliteReport>,
    pub objects: Vec<StoredObject>,
    pub operations: Vec<OperationReport>,
    pub recent_events: Vec<Event>,
    pub events_dropped: u64,
}
#[derive(Clone, Debug)]
struct Operation {
    report: OperationReport,
    remaining: f64,
    hop: usize,
    reserved: bool,
    arrival: Option<f64>,
}
impl Operation {
    fn pending(a: &ScheduledAction) -> Self {
        Self {
            report: OperationReport {
                id: a.id.clone(),
                status: OperationStatus::Pending,
                started_s: None,
                finished_s: None,
            },
            remaining: match &a.action {
                Action::Compute { work, .. } => *work,
                Action::Acquire { duration_s, .. } => *duration_s,
                _ => 0.0,
            },
            hop: 0,
            reserved: false,
            arrival: None,
        }
    }
}
#[derive(Clone, Debug)]
struct Satellite {
    duty: Option<RollingDuty>,
    energy_j: Option<f64>,
    compute_owner: Option<usize>,
    sensor_owner: Option<usize>,
    busy_s: f64,
}

pub struct Simulator<E: Environment> {
    config: Scenario,
    environment: E,
    tick: u64,
    storage: Storage,
    satellites: Vec<Satellite>,
    operations: Vec<Operation>,
    active: BTreeSet<usize>,
    schedule: Vec<usize>,
    next_action: usize,
    metrics: Metrics,
    events: VecDeque<Event>,
    events_dropped: u64,
}
impl<E: Environment> Simulator<E> {
    pub fn new(mut config: Scenario, environment: E) -> Result<Self> {
        config.validate()?;
        config.actions.sort_by(|a, b| a.at_s.total_cmp(&b.at_s));
        if environment.satellite_count() != config.satellite_count() {
            return Err("environment and scenario satellite counts differ".into());
        }
        let mut storage = object_registry(&config)?;
        let satellites = (0..config.satellite_count())
            .map(|id| {
                let c = config.resources(id);
                if let Some(limit) = c.storage_bytes {
                    storage.limit(Node::Satellite(id), limit);
                }
                Satellite {
                    duty: c.duty.clone().map(RollingDuty::new),
                    energy_j: c.battery.as_ref().map(|b| b.initial_j),
                    compute_owner: None,
                    sensor_owner: None,
                    busy_s: 0.0,
                }
            })
            .collect();
        for o in &config.initial_objects {
            if !storage.has(o.node, &o.object.id) {
                if !storage.reserve(o.node, &o.object.id) {
                    return Err("initial objects exceed storage capacity".into());
                }
                storage.commit(o.node, &o.object.id);
            }
        }
        let operations = config.actions.iter().map(Operation::pending).collect();
        let mut schedule: Vec<_> = (0..config.actions.len()).collect();
        schedule.sort_by(|a, b| {
            config.actions[*a]
                .at_s
                .total_cmp(&config.actions[*b].at_s)
                .then(a.cmp(b))
        });
        Ok(Self {
            config,
            environment,
            tick: 0,
            storage,
            satellites,
            operations,
            active: BTreeSet::new(),
            schedule,
            next_action: 0,
            metrics: Metrics::default(),
            events: VecDeque::new(),
            events_dropped: 0,
        })
    }
    /// Submit a batch at or after the current boundary. Validation is atomic.
    /// Policies choose placement/routes; the core enforces resource feasibility.
    pub fn submit_actions(&mut self, actions: Vec<ScheduledAction>) -> Result<()> {
        if actions.iter().any(|a| a.at_s < self.time_s()) {
            return Err("cannot submit an action in the past".into());
        }
        let mut candidate = self.config.clone();
        candidate.actions.extend(actions.iter().cloned());
        candidate.validate()?;
        object_registry(&candidate)?;
        for action in actions {
            if let Action::Compute { output, .. } | Action::Acquire { output, .. } = &action.action
            {
                self.storage.define(output)?;
            }
            let i = self.operations.len();
            self.operations.push(Operation::pending(&action));
            self.config.actions.push(action);
            self.schedule.push(i);
        }
        self.schedule[self.next_action..].sort_by(|a, b| {
            self.config.actions[*a]
                .at_s
                .total_cmp(&self.config.actions[*b].at_s)
                .then(a.cmp(b))
        });
        Ok(())
    }
    pub fn config(&self) -> &Scenario {
        &self.config
    }
    pub fn environment(&self) -> &E {
        &self.environment
    }
    pub fn time_s(&self) -> f64 {
        self.tick as f64 * self.config.step_s
    }
    pub fn has_object(&self, node: Node, id: &str) -> bool {
        self.storage.has(node, id)
    }
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }
    pub fn drain_events(&mut self) -> impl Iterator<Item = Event> + '_ {
        self.events.drain(..)
    }
    fn event(&mut self, i: usize, t: f64, kind: &str) {
        if self.config.event_capacity == 0 {
            self.events_dropped += 1;
            return;
        }
        if self.events.len() == self.config.event_capacity {
            self.events.pop_front();
            self.events_dropped += 1;
        }
        self.events.push_back(Event {
            time_s: t,
            action_id: self.operations[i].report.id.clone(),
            kind: kind.into(),
        });
    }
    fn start(&mut self, i: usize, t: f64) {
        if self.operations[i].report.started_s.is_none() {
            self.operations[i].report.started_s = Some(t);
            self.event(i, t, "started");
        }
        self.operations[i].report.status = OperationStatus::Running;
    }
    fn complete(&mut self, i: usize, t: f64) {
        self.operations[i].report.status = OperationStatus::Completed;
        self.operations[i].report.finished_s = Some(t);
        self.active.remove(&i);
        self.metrics.completed_actions += 1;
        self.event(i, t, "completed");
    }
    fn reject(&mut self, i: usize, t: f64, reason: &str) {
        self.operations[i].report.status = OperationStatus::Rejected(reason.into());
        self.operations[i].report.finished_s = Some(t);
        self.active.remove(&i);
        self.metrics.rejected_actions += 1;
        self.event(i, t, reason);
    }
    fn pinned(&self, node: Node, id: &str) -> bool {
        self.active.iter().any(|i| {
            let op = &self.operations[*i];
            match &self.config.actions[*i].action {
                Action::Compute {
                    satellite, inputs, ..
                } => {
                    self.satellites[*satellite].compute_owner == Some(*i)
                        && node == Node::Satellite(*satellite)
                        && inputs.iter().any(|s| s == id)
                }
                Action::Transfer { object, path } => {
                    object == id && op.reserved && op.arrival.is_none() && path[op.hop] == node
                }
                _ => false,
            }
        })
    }
    fn arrivals(&mut self, time: f64) {
        let due: Vec<_> = self
            .active
            .iter()
            .copied()
            .filter(|i| {
                self.operations[*i]
                    .arrival
                    .is_some_and(|t| t <= time + 1e-9)
            })
            .collect();
        for i in due {
            let Action::Transfer { object, path } = &self.config.actions[i].action else {
                unreachable!()
            };
            let object = object.clone();
            let dest = path[self.operations[i].hop + 1];
            let last = self.operations[i].hop + 2 == path.len();
            self.storage.commit(dest, &object);
            self.operations[i].hop += 1;
            self.operations[i].arrival = None;
            self.operations[i].reserved = false;
            self.operations[i].remaining = 0.0;
            self.event(i, time, "hop_arrived");
            if last {
                self.complete(i, time);
            } else {
                self.operations[i].report.status = OperationStatus::Queued;
            }
        }
    }
    /// Only grid-aligned observation boundaries are accepted, so caller chunking
    /// cannot change numerical integration or resource arbitration.
    pub fn run_until(&mut self, until: f64) -> Result<()> {
        if !aligned(until, self.config.step_s)
            || until < self.time_s()
            || until > self.config.duration_s
        {
            return Err("run_until must be monotonic, grid-aligned and within duration".into());
        }
        let end_tick = (until / self.config.step_s).round() as u64;
        while self.tick < end_tick {
            // Idle abstract-resource periods need no per-satellite updates.
            if self.active.is_empty() && self.satellites.iter().all(|s| s.energy_j.is_none()) {
                let next = self
                    .schedule
                    .get(self.next_action)
                    .map(|i| (self.config.actions[*i].at_s / self.config.step_s).round() as u64)
                    .unwrap_or(end_tick)
                    .min(end_tick);
                if next > self.tick {
                    self.tick = next;
                    continue;
                }
            }
            self.step()?;
        }
        Ok(())
    }
    pub fn step(&mut self) -> Result<()> {
        let t = self.time_s();
        let dt = self.config.step_s;
        let end = (self.tick + 1) as f64 * dt;
        if end > self.config.duration_s + 1e-9 {
            return Err("simulation duration exhausted".into());
        }
        self.arrivals(t);
        while let Some(&i) = self.schedule.get(self.next_action) {
            if self.config.actions[i].at_s > t + 1e-9 {
                break;
            }
            self.operations[i].report.status = OperationStatus::Queued;
            self.active.insert(i);
            self.next_action += 1;
            self.event(i, t, "queued");
        }
        let mut power: Vec<_> = self
            .satellites
            .iter()
            .enumerate()
            .map(|(s, state)| {
                self.config.resources(s).battery.as_ref().map(|c| {
                    PowerBudget::new(
                        state.energy_j.unwrap(),
                        c,
                        self.environment.sunlit(s, t + dt / 2.0),
                        dt,
                    )
                })
            })
            .collect();
        if let Some((s, _)) = power
            .iter()
            .enumerate()
            .find(|(_, p)| p.as_ref().is_some_and(|p| p.reserve_deficit_j > 1e-7))
        {
            return Err(format!(
                "baseline load breaches battery reserve on satellite {s} at {t}s"
            ));
        }
        let mut busy = vec![0.0; self.satellites.len()];
        let mut compute_used = BTreeSet::new();
        let mut sensor_used = BTreeSet::new();
        let mut finished = Vec::new();
        let mut network = NetworkBudget::new(&self.config, dt);
        let mut active: Vec<_> = self.active.iter().copied().collect();
        active.sort_by(|a, b| {
            self.config.actions[*a]
                .at_s
                .total_cmp(&self.config.actions[*b].at_s)
                .then(a.cmp(b))
        });
        for i in active {
            let action = self.config.actions[i].action.clone();
            match action {
                Action::Compute {
                    satellite: s,
                    work: _,
                    inputs,
                    output,
                } => {
                    if compute_used.contains(&s)
                        || self.satellites[s].compute_owner.is_some_and(|v| v != i)
                    {
                        continue;
                    }
                    if !inputs
                        .iter()
                        .all(|id| self.storage.has(Node::Satellite(s), id))
                    {
                        continue;
                    }
                    let c = self.config.resources(s);
                    let watts = c.battery.as_ref().map_or(0.0, |b| b.compute_w);
                    let mut seconds =
                        (self.operations[i].remaining / c.compute_units_per_s).min(dt);
                    if let Some(d) = &self.satellites[s].duty {
                        seconds = seconds.min(d.allowance(end, dt));
                    }
                    if let Some(p) = &power[s] {
                        seconds = p.permitted_seconds(watts, seconds);
                    }
                    if seconds <= 1e-12 {
                        continue;
                    }
                    let rate = c.compute_units_per_s;
                    if !self.operations[i].reserved {
                        if !self.storage.reserve(Node::Satellite(s), &output.id) {
                            continue;
                        }
                        self.operations[i].reserved = true;
                    }
                    self.start(i, t);
                    self.satellites[s].compute_owner = Some(i);
                    compute_used.insert(s);
                    if let Some(p) = &mut power[s] {
                        p.spend(watts, seconds);
                    }
                    let work = seconds * rate;
                    self.operations[i].remaining = (self.operations[i].remaining - work).max(0.0);
                    self.metrics.compute_work += work;
                    self.metrics.compute_busy_s += seconds;
                    busy[s] += seconds;
                    if self.operations[i].remaining <= 1e-9 {
                        finished.push((i, s, output.id, false));
                    }
                }
                Action::Acquire {
                    satellite: s,
                    duration_s: _,
                    target,
                    output,
                } => {
                    if sensor_used.contains(&s)
                        || self.satellites[s].sensor_owner.is_some_and(|v| v != i)
                    {
                        continue;
                    }
                    let c = self.config.resources(s);
                    let seconds = self.operations[i].remaining.min(dt);
                    let watts = c.battery.as_ref().map_or(0.0, |b| b.sensing_w);
                    let visible = [t, t + seconds / 2.0, t + seconds]
                        .iter()
                        .all(|time| self.environment.can_observe(s, &target, &c.sensor, *time));
                    let powered = power[s]
                        .as_ref()
                        .is_none_or(|p| p.permitted_seconds(watts, seconds) + 1e-9 >= seconds);
                    if !visible || !powered {
                        if self.satellites[s].sensor_owner == Some(i) {
                            self.storage.release(Node::Satellite(s), &output.id);
                            self.satellites[s].sensor_owner = None;
                            self.reject(i, t, "acquisition_interrupted");
                        }
                        continue;
                    }
                    if !self.operations[i].reserved {
                        if !self.storage.reserve(Node::Satellite(s), &output.id) {
                            continue;
                        }
                        self.operations[i].reserved = true;
                    }
                    self.start(i, t);
                    self.satellites[s].sensor_owner = Some(i);
                    sensor_used.insert(s);
                    if let Some(p) = &mut power[s] {
                        p.spend(watts, seconds);
                    }
                    self.operations[i].remaining =
                        (self.operations[i].remaining - seconds).max(0.0);
                    self.metrics.sensing_busy_s += seconds;
                    if self.operations[i].remaining <= 1e-9 {
                        finished.push((i, s, output.id, true));
                    }
                }
                Action::Transfer { object, path } => {
                    if self.operations[i].arrival.is_some() {
                        continue;
                    }
                    // Reuse an existing copy, including one delivered by another transfer.
                    while self.operations[i].hop + 1 < path.len()
                        && self.storage.has(path[self.operations[i].hop + 1], &object)
                    {
                        self.operations[i].hop += 1;
                    }
                    if self.operations[i].hop + 1 == path.len() {
                        self.complete(i, t);
                        continue;
                    }
                    let h = self.operations[i].hop;
                    let from = path[h];
                    let to = path[h + 1];
                    if !self.storage.has(from, &object) {
                        continue;
                    }
                    let Some(link) = self.environment.link(from, to, t) else {
                        continue;
                    };
                    if !link.bytes_per_s.is_finite()
                        || link.bytes_per_s <= 0.0
                        || !nonnegative(link.propagation_s)
                    {
                        return Err("environment returned invalid link".into());
                    }
                    let mut bytes = network.available(&self.config, from, to, link).min(
                        if self.operations[i].reserved {
                            self.operations[i].remaining
                        } else {
                            self.storage.bytes(&object) as f64
                        },
                    );
                    let Node::Satellite(src) = from else {
                        unreachable!()
                    };
                    let tx_w = self
                        .config
                        .resources(src)
                        .battery
                        .as_ref()
                        .map_or(0.0, |b| {
                            if matches!(to, Node::Ground(_)) {
                                b.downlink_w
                            } else {
                                b.isl_tx_w
                            }
                        });
                    if let Some(p) = &power[src] {
                        bytes = bytes.min(p.permitted_seconds(tx_w, dt) * link.bytes_per_s);
                    }
                    let rx_w = if let Node::Satellite(dst) = to {
                        let w = self
                            .config
                            .resources(dst)
                            .battery
                            .as_ref()
                            .map_or(0.0, |b| b.isl_rx_w);
                        if let Some(p) = &power[dst] {
                            bytes = bytes.min(p.permitted_seconds(w, dt) * link.bytes_per_s);
                        }
                        w
                    } else {
                        0.0
                    };
                    if bytes <= 1e-9 {
                        continue;
                    }
                    if !self.operations[i].reserved {
                        if !self.storage.reserve(to, &object) {
                            continue;
                        }
                        self.operations[i].reserved = true;
                        self.operations[i].remaining = self.storage.bytes(&object) as f64;
                    }
                    self.start(i, t);
                    network.spend(from, to, link, bytes);
                    let seconds = bytes / link.bytes_per_s;
                    if let Some(p) = &mut power[src] {
                        p.spend(tx_w, seconds);
                    }
                    if let Node::Satellite(dst) = to {
                        if let Some(p) = &mut power[dst] {
                            p.spend(rx_w, seconds);
                        }
                    }
                    match to {
                        Node::Satellite(_) => self.metrics.isl_byte_hops += bytes,
                        Node::Ground(_) => self.metrics.downlink_bytes += bytes,
                    }
                    self.operations[i].remaining = (self.operations[i].remaining - bytes).max(0.0);
                    if self.operations[i].remaining <= 1e-7 {
                        self.operations[i].arrival = Some(end + link.propagation_s);
                        self.operations[i].report.status = OperationStatus::InTransit;
                        self.event(i, end, "hop_sent");
                    }
                }
                Action::Delete { node, object } => {
                    if self.pinned(node, &object) {
                        self.reject(i, t, "object_in_use");
                    } else if self.storage.delete(node, &object) {
                        self.complete(i, t);
                    } else {
                        self.reject(i, t, "object_not_present");
                    }
                }
            }
        }
        for (i, s, id, sensing) in finished {
            self.storage.commit(Node::Satellite(s), &id);
            if sensing {
                self.satellites[s].sensor_owner = None;
            } else {
                self.satellites[s].compute_owner = None;
            }
            self.complete(i, end);
        }
        for (s, state) in self.satellites.iter_mut().enumerate() {
            state.busy_s += busy[s];
            if let Some(d) = &mut state.duty {
                d.record(t, end, busy[s]);
            }
            if let Some(p) = &power[s] {
                let c = self.config.resources(s).battery.as_ref().unwrap();
                state.energy_j = Some(p.finish(c.capacity_j));
                self.metrics.energy_load_j += p.spent_j + c.idle_w * dt;
                self.metrics.reserve_deficit_j += p.reserve_deficit_j;
            }
        }
        self.tick += 1;
        self.arrivals(end);
        Ok(())
    }
    pub fn report(&self) -> Report {
        Report {
            schema_version: 1,
            time_s: self.time_s(),
            metrics: self.metrics.clone(),
            satellites: self
                .satellites
                .iter()
                .enumerate()
                .map(|(id, s)| SatelliteReport {
                    id,
                    stored_and_reserved_bytes: self.storage.used(Node::Satellite(id)),
                    compute_busy_s: s.busy_s,
                    rolling_compute_busy_s: s.duty.as_ref().map_or(0.0, |d| d.used(self.time_s())),
                    energy_j: s.energy_j,
                })
                .collect(),
            objects: self
                .storage
                .copies()
                .map(|(node, id, bytes)| StoredObject {
                    node,
                    object: Object {
                        id: id.into(),
                        bytes,
                    },
                })
                .collect(),
            operations: self.operations.iter().map(|op| op.report.clone()).collect(),
            recent_events: self.events.iter().cloned().collect(),
            events_dropped: self.events_dropped,
        }
    }
}

fn object_registry(config: &Scenario) -> Result<Storage> {
    let mut storage = Storage::default();
    let mut producers = BTreeSet::new();
    for o in &config.initial_objects {
        storage.define(&o.object)?;
        producers.insert(o.object.id.clone());
    }
    for a in &config.actions {
        if let Action::Compute { output, .. } | Action::Acquire { output, .. } = &a.action {
            if !producers.insert(output.id.clone()) {
                return Err(format!("duplicate object producer {}", output.id));
            }
            storage.define(output)?;
        }
    }
    for a in &config.actions {
        match &a.action {
            Action::Compute { inputs, .. } if inputs.iter().any(|i| !storage.knows(i)) => {
                return Err(format!("undefined input for {}", a.id));
            }
            Action::Transfer { object, .. } | Action::Delete { object, .. }
                if !storage.knows(object) =>
            {
                return Err(format!("undefined object for {}", a.id));
            }
            _ => {}
        }
    }
    Ok(storage)
}
