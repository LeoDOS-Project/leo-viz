use leo_sim::*;
use leo_sim::{
    geometry::*,
    network::Link,
    orbit::{Propagation, WalkerShell},
    resources::{integrate_energy, RollingDuty},
};
use std::collections::BTreeMap;
const S0: Node = Node::Satellite(0);
const S1: Node = Node::Satellite(1);
const S2: Node = Node::Satellite(2);
const G0: Node = Node::Ground(0);
#[derive(Clone)]
struct Static {
    n: usize,
    links: BTreeMap<(Node, Node), (f64, f64)>,
    off: Vec<(f64, f64)>,
    sun: bool,
    visible: bool,
}
impl Static {
    fn new(n: usize) -> Self {
        Self {
            n,
            links: BTreeMap::new(),
            off: vec![],
            sun: true,
            visible: true,
        }
    }
    fn edge(mut self, a: Node, b: Node, rate: f64, delay: f64) -> Self {
        self.links.insert((a, b), (rate, delay));
        self
    }
}
impl Environment for Static {
    fn satellite_count(&self) -> usize {
        self.n
    }
    fn link(&self, a: Node, b: Node, t: f64) -> Option<Link> {
        if self.off.iter().any(|(x, y)| t >= *x && t < *y) {
            None
        } else {
            self.links.get(&(a, b)).map(|(rate, delay)| Link {
                bytes_per_s: *rate,
                propagation_s: *delay,
            })
        }
    }
    fn sunlit(&self, _: usize, _: f64) -> bool {
        self.sun
    }
    fn can_observe(&self, _: usize, _: &Target, s: &SensorConfig, _: f64) -> bool {
        self.visible && s.enabled
    }
}
fn config() -> Scenario {
    Scenario {
        shells: vec![WalkerShell {
            planes: 1,
            satellites_per_plane: 3,
            phasing: 0,
            ..Default::default()
        }],
        duration_s: 30.0,
        satellite: SatelliteResources {
            duty: None,
            downlink_bps: 800.0,
            ..Default::default()
        },
        ground_stations: vec![GroundStation {
            aggregate_bps: 800.0,
            per_link_bps: 800.0,
            max_contacts: 2,
            ..Default::default()
        }],
        ..Default::default()
    }
}
fn obj(id: &str, bytes: u64) -> Object {
    Object {
        id: id.into(),
        bytes,
    }
}
fn initial(c: &mut Scenario, node: Node, id: &str, bytes: u64) {
    c.initial_objects.push(StoredObject {
        node,
        object: obj(id, bytes),
    });
}
fn action(c: &mut Scenario, id: &str, t: f64, a: Action) {
    c.actions.push(ScheduledAction {
        id: id.into(),
        at_s: t,
        action: a,
    });
}
fn transfer(c: &mut Scenario, id: &str, object: &str, path: Vec<Node>) {
    action(
        c,
        id,
        0.0,
        Action::Transfer {
            object: object.into(),
            path,
        },
    );
}
fn compute(c: &mut Scenario, id: &str, s: usize, work: f64, inputs: Vec<&str>) {
    action(
        c,
        id,
        0.0,
        Action::Compute {
            satellite: s,
            work,
            inputs: inputs.into_iter().map(String::from).collect(),
            output: obj(id, 10),
        },
    );
}
#[test]
fn serde_roundtrip_and_reject_unknown_config() {
    let mut c = config();
    compute(&mut c, "x", 0, 2.0, vec![]);
    let json = serde_json::to_string(&c).unwrap();
    let read: Scenario = serde_json::from_str(&json).unwrap();
    read.validate().unwrap();
    assert!(serde_json::from_str::<Scenario>(r#"{"typo_capacity":2}"#).is_err());
}
#[test]
fn compute_work_and_dependencies_do_not_finish_early() {
    let mut c = config();
    compute(&mut c, "a", 0, 2.0, vec![]);
    compute(&mut c, "b", 0, 1.0, vec!["a"]);
    let mut s = Simulator::new(c, Static::new(3)).unwrap();
    s.run_until(1.0).unwrap();
    assert!(!s.has_object(S0, "a"));
    s.run_until(2.0).unwrap();
    assert!(s.has_object(S0, "a"));
    assert!(!s.has_object(S0, "b"));
    s.run_until(3.0).unwrap();
    assert!(s.has_object(S0, "b"));
    assert_eq!(s.metrics().compute_work, 3.0);
}
#[test]
fn shared_directed_link_and_no_partial_availability() {
    let mut c = config();
    initial(&mut c, S0, "a", 100);
    initial(&mut c, S0, "b", 100);
    transfer(&mut c, "txa", "a", vec![S0, S1]);
    transfer(&mut c, "txb", "b", vec![S0, S1]);
    let mut s = Simulator::new(c, Static::new(3).edge(S0, S1, 50.0, 0.0)).unwrap();
    s.run_until(1.0).unwrap();
    assert!(!s.has_object(S1, "a"));
    s.run_until(2.0).unwrap();
    assert!(s.has_object(S1, "a"));
    assert!(!s.has_object(S1, "b"));
    s.run_until(4.0).unwrap();
    assert!(s.has_object(S1, "b"));
    assert_eq!(s.metrics().isl_byte_hops, 200.0);
}
#[test]
fn transfer_resumes_after_contact_without_resending() {
    let mut c = config();
    initial(&mut c, S0, "a", 150);
    transfer(&mut c, "tx", "a", vec![S0, S1]);
    let mut env = Static::new(3).edge(S0, S1, 50.0, 0.0);
    env.off = vec![(1.0, 3.0)];
    let mut s = Simulator::new(c, env).unwrap();
    s.run_until(4.0).unwrap();
    assert!(!s.has_object(S1, "a"));
    assert_eq!(s.metrics().isl_byte_hops, 100.0);
    s.run_until(5.0).unwrap();
    assert!(s.has_object(S1, "a"));
    assert_eq!(s.metrics().isl_byte_hops, 150.0);
}
#[test]
fn propagation_prevents_instantaneous_hops() {
    let mut c = config();
    initial(&mut c, S0, "a", 50);
    transfer(&mut c, "tx", "a", vec![S0, S1, S2]);
    let env = Static::new(3)
        .edge(S0, S1, 50.0, 0.2)
        .edge(S1, S2, 50.0, 0.2);
    let mut s = Simulator::new(c, env).unwrap();
    s.run_until(1.0).unwrap();
    assert!(!s.has_object(S1, "a"));
    s.run_until(2.0).unwrap();
    assert!(s.has_object(S1, "a"));
    assert!(!s.has_object(S2, "a"));
    s.run_until(4.0).unwrap();
    assert!(s.has_object(S2, "a"));
    assert_eq!(s.metrics().isl_byte_hops, 100.0);
}
#[test]
fn downlinks_share_station_budget() {
    let mut c = config();
    initial(&mut c, S0, "a", 100);
    initial(&mut c, S1, "b", 100);
    transfer(&mut c, "a", "a", vec![S0, G0]);
    transfer(&mut c, "b", "b", vec![S1, G0]);
    let env = Static::new(3)
        .edge(S0, G0, 100.0, 0.0)
        .edge(S1, G0, 100.0, 0.0);
    let mut s = Simulator::new(c, env).unwrap();
    s.run_until(1.0).unwrap();
    assert_eq!(s.metrics().downlink_bytes, 100.0);
    assert!(s.has_object(G0, "a"));
    assert!(!s.has_object(G0, "b"));
    s.run_until(2.0).unwrap();
    assert!(s.has_object(G0, "b"));
}
#[test]
fn terminal_limit_is_separate_from_station_rate() {
    let mut c = config();
    c.ground_stations[0].max_contacts = 1;
    c.ground_stations[0].aggregate_bps = 8000.0;
    initial(&mut c, S0, "a", 50);
    initial(&mut c, S1, "b", 50);
    transfer(&mut c, "a", "a", vec![S0, G0]);
    transfer(&mut c, "b", "b", vec![S1, G0]);
    let env = Static::new(3)
        .edge(S0, G0, 100.0, 0.0)
        .edge(S1, G0, 100.0, 0.0);
    let mut s = Simulator::new(c, env).unwrap();
    s.run_until(1.0).unwrap();
    assert_eq!(s.metrics().downlink_bytes, 50.0);
    s.run_until(2.0).unwrap();
    assert_eq!(s.metrics().downlink_bytes, 100.0);
}
#[test]
fn full_duplex_uses_distinct_capacity() {
    let mut c = config();
    initial(&mut c, S0, "a", 100);
    initial(&mut c, S1, "b", 100);
    transfer(&mut c, "a", "a", vec![S0, S1]);
    transfer(&mut c, "b", "b", vec![S1, S0]);
    let env = Static::new(3)
        .edge(S0, S1, 100.0, 0.0)
        .edge(S1, S0, 100.0, 0.0);
    let mut s = Simulator::new(c, env).unwrap();
    s.run_until(1.0).unwrap();
    assert!(s.has_object(S0, "b") && s.has_object(S1, "a"));
    assert_eq!(s.metrics().isl_byte_hops, 200.0);
}
#[test]
fn rolling_budget_expires_without_midnight_reset() {
    let mut c = config();
    c.satellite.duty = Some(DutyLimit {
        window_s: 10.0,
        max_fraction: 0.2,
        initial_utilization: 0.0,
    });
    compute(&mut c, "a", 0, 20.0, vec![]);
    let mut s = Simulator::new(c, Static::new(3)).unwrap();
    s.run_until(10.0).unwrap();
    assert_eq!(s.metrics().compute_work, 2.0);
    s.run_until(11.0).unwrap();
    assert_eq!(s.metrics().compute_work, 3.0);
    assert_eq!(s.report().satellites[0].rolling_compute_busy_s, 2.0);
}
#[test]
fn rolling_budget_honors_initial_history_and_fractional_use() {
    let mut d = RollingDuty::new(DutyLimit {
        window_s: 10.0,
        max_fraction: 0.2,
        initial_utilization: 0.2,
    });
    for i in 0..30 {
        let t = i as f64;
        let allowed = d.allowance(t + 1.0, 1.0);
        assert!((allowed - 0.2).abs() < 1e-9);
        d.record(t, t + 1.0, allowed);
        assert!(d.used(t + 0.5) <= 2.0 + 1e-9);
        assert!(d.used(t + 1.0) <= 2.0 + 1e-9);
    }
}
#[test]
fn storage_reservation_limits_transfer_and_local_reuse() {
    let mut c = config();
    c.satellite.storage_bytes = Some(150);
    initial(&mut c, S0, "a", 100);
    initial(&mut c, S1, "b", 100);
    transfer(&mut c, "tx", "a", vec![S0, S1]);
    let mut s = Simulator::new(c, Static::new(3).edge(S0, S1, 100.0, 0.0)).unwrap();
    s.run_until(2.0).unwrap();
    assert_eq!(s.metrics().isl_byte_hops, 0.0);
    let mut c = config();
    initial(&mut c, S0, "a", 100);
    initial(&mut c, S1, "a", 100);
    transfer(&mut c, "tx", "a", vec![S0, S1]);
    let mut s = Simulator::new(c, Static::new(3)).unwrap();
    s.run_until(1.0).unwrap();
    assert_eq!(s.metrics().completed_actions, 1);
    assert_eq!(s.metrics().isl_byte_hops, 0.0);
}
#[test]
fn duplicate_transfers_do_not_duplicate_storage_or_bytes() {
    let mut c = config();
    initial(&mut c, S0, "a", 100);
    transfer(&mut c, "x", "a", vec![S0, S1]);
    transfer(&mut c, "y", "a", vec![S0, S1]);
    let mut s = Simulator::new(c, Static::new(3).edge(S0, S1, 100.0, 0.0)).unwrap();
    s.run_until(2.0).unwrap();
    assert_eq!(s.metrics().isl_byte_hops, 100.0);
    assert_eq!(s.report().satellites[1].stored_and_reserved_bytes, 100);
    assert_eq!(s.metrics().completed_actions, 2);
}
#[test]
fn in_use_inputs_cannot_be_deleted() {
    let mut c = config();
    initial(&mut c, S0, "input", 100);
    compute(&mut c, "out", 0, 3.0, vec!["input"]);
    action(
        &mut c,
        "del",
        1.0,
        Action::Delete {
            node: S0,
            object: "input".into(),
        },
    );
    let mut s = Simulator::new(c, Static::new(3)).unwrap();
    s.run_until(3.0).unwrap();
    assert_eq!(s.metrics().rejected_actions, 1);
    assert!(s.has_object(S0, "input"));
}
#[test]
fn power_uses_net_balance_and_throttles_optional_work() {
    assert_eq!(integrate_energy(100.0, 100.0, 50.0, 10.0, 1.0), 100.0);
    let mut c = config();
    c.satellite.battery = Some(BatteryConfig {
        capacity_j: 100.0,
        initial_j: 30.0,
        reserve_j: 20.0,
        solar_w: 0.0,
        idle_w: 0.0,
        compute_w: 20.0,
        ..Default::default()
    });
    compute(&mut c, "a", 0, 1.0, vec![]);
    let mut s = Simulator::new(c, Static::new(3)).unwrap();
    s.run_until(2.0).unwrap();
    assert_eq!(s.metrics().compute_work, 0.5);
    assert_eq!(s.report().satellites[0].energy_j, Some(20.0));
    assert!(!s.has_object(S0, "a"));
}
#[test]
fn baseline_energy_infeasibility_is_reported() {
    let mut c = config();
    c.satellite.battery = Some(BatteryConfig {
        capacity_j: 100.0,
        initial_j: 20.0,
        reserve_j: 20.0,
        solar_w: 0.0,
        idle_w: 1.0,
        ..Default::default()
    });
    let mut s = Simulator::new(c, Static::new(3)).unwrap();
    assert!(s.step().unwrap_err().contains("reserve"));
}
#[test]
fn acquisition_holds_sensor_and_creates_only_finished_data() {
    let mut c = config();
    action(
        &mut c,
        "capture",
        0.0,
        Action::Acquire {
            satellite: 0,
            duration_s: 2.0,
            target: Target {
                lat_deg: 0.0,
                lon_deg: 0.0,
                radius_km: 0.0,
                min_solar_elevation_deg: None,
            },
            output: obj("image", 100),
        },
    );
    let mut s = Simulator::new(c, Static::new(3)).unwrap();
    s.run_until(1.0).unwrap();
    assert!(!s.has_object(S0, "image"));
    s.run_until(2.0).unwrap();
    assert!(s.has_object(S0, "image"));
    assert_eq!(s.metrics().sensing_busy_s, 2.0);
}
#[test]
fn call_chunking_and_trace_consumption_do_not_change_physics() {
    let mut c = config();
    c.satellite.duty = Some(DutyLimit {
        window_s: 10.0,
        max_fraction: 0.3,
        initial_utilization: 0.0,
    });
    compute(&mut c, "a", 0, 10.0, vec![]);
    let mut a = Simulator::new(c.clone(), Static::new(3)).unwrap();
    let mut b = Simulator::new(c, Static::new(3)).unwrap();
    a.run_until(30.0).unwrap();
    for _ in 0..30 {
        b.step().unwrap();
        b.drain_events().for_each(drop);
    }
    assert_eq!(
        serde_json::to_value(a.metrics()).unwrap(),
        serde_json::to_value(b.metrics()).unwrap()
    );
    assert_eq!(
        a.report().satellites[0].compute_busy_s,
        b.report().satellites[0].compute_busy_s
    );
    assert!(b.run_until(30.1).is_err());
}
#[test]
fn invalid_scenarios_and_unknown_objects_are_rejected() {
    let mut c = config();
    c.step_s = 0.0;
    assert!(c.validate().is_err());
    let mut c = config();
    transfer(&mut c, "a", "unknown", vec![S0, S1]);
    assert!(Simulator::new(c, Static::new(3)).is_err());
    let mut c = config();
    c.shells[0].planes = 0;
    assert!(c.validate().is_err());
}
#[test]
fn geometry_earth_obstruction_and_ground_elevation() {
    assert!(!clear_line(
        [7000.0, 0.0, 0.0],
        [-7000.0, 0.0, 0.0],
        EARTH_RADIUS_KM
    ));
    assert!(clear_line(
        [7000.0, 0.0, 0.0],
        [6990.0, 100.0, 0.0],
        EARTH_RADIUS_KM
    ));
    assert!((elevation([EARTH_RADIUS_KM, 0.0, 0.0], [7000.0, 0.0, 0.0]) - 90.0).abs() < 1e-9);
    assert!(eclipsed(
        [-7000.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        EARTH_RADIUS_KM
    ));
}
#[test]
fn keplerian_period_and_j2_are_distinct() {
    let shell = WalkerShell::default();
    let e = shell.elements(0, 0);
    let period = std::f64::consts::TAU * (e.semi_major_km.powi(3) / e.mu).sqrt();
    assert!(distance(e.sample(0.0).xyz, e.sample(period).xyz) < 1e-7);
    let mut j = e;
    j.propagation = Propagation::J2;
    assert!(distance(e.sample(86400.0).xyz, j.sample(86400.0).xyz) > 1.0);
}
#[test]
fn seam_topology_is_symmetric_and_degree_bounded() {
    let c = Scenario {
        shells: vec![WalkerShell {
            planes: 4,
            satellites_per_plane: 8,
            phasing: 1,
            ..Default::default()
        }],
        ..Default::default()
    };
    let e = OrbitEnvironment::new(&c).unwrap();
    assert!(e.neighbor_ids(24).any(|n| n == 1));
    for s in 0..e.satellite_count() {
        let ns: Vec<_> = e.neighbor_ids(s).collect();
        assert!(ns.len() <= 4);
        for n in ns {
            assert!(e.neighbor_ids(n).any(|v| v == s));
        }
    }
}
#[test]
fn sensing_fov_and_ground_contact_are_independent() {
    let mut c = config();
    let temporary = OrbitEnvironment::new(&c).unwrap();
    let (lat, lon) = temporary.geographic_position(0, 0.0);
    c.ground_stations[0].lat_deg = lat;
    c.ground_stations[0].lon_deg = lon;
    let e = OrbitEnvironment::new(&c).unwrap();
    let target = Target {
        lat_deg: lat,
        lon_deg: lon,
        radius_km: 1.0,
        min_solar_elevation_deg: None,
    };
    assert!(e.link(S0, G0, 0.0).is_some());
    assert!(e.can_observe(0, &target, &SensorConfig::default(), 0.0));
    let disabled = SensorConfig {
        enabled: false,
        ..Default::default()
    };
    assert!(!e.can_observe(0, &target, &disabled, 0.0));
    assert!(e.link(S0, G0, 0.0).is_some());
}
#[test]
fn contact_plan_matches_sampled_execution_and_converges() {
    let mut c = config();
    let tmp = OrbitEnvironment::new(&c).unwrap();
    let (lat, lon) = tmp.geographic_position(0, 0.0);
    c.ground_stations[0].lat_deg = lat;
    c.ground_stations[0].lon_deg = lon;
    let e = OrbitEnvironment::new(&c).unwrap();
    let coarse = e.contacts(S0, G0, 0.0, 3600.0, 2.0).unwrap();
    let fine = e.contacts(S0, G0, 0.0, 3600.0, 1.0).unwrap();
    let total =
        |v: &Vec<leo_sim::network::Contact>| v.iter().map(|c| c.end_s - c.start_s).sum::<f64>();
    assert!((total(&coarse) - total(&fine)).abs() <= 4.0);
    let direct = (0..3600)
        .filter_map(|t| e.link(S0, G0, t as f64))
        .map(|l| l.bytes_per_s)
        .sum::<f64>();
    assert!((fine.iter().map(|c| c.capacity_bytes).sum::<f64>() - direct).abs() < 1e-6);
}

#[test]
fn dynamic_submission_is_atomic_and_replayable() {
    let c = config();
    let mut sim = Simulator::new(c, Static::new(3)).unwrap();
    sim.run_until(2.0).unwrap();
    let compute = ScheduledAction {
        id: "late".into(),
        at_s: 2.0,
        action: Action::Compute {
            satellite: 0,
            work: 2.0,
            inputs: vec![],
            output: obj("late-output", 10),
        },
    };
    let mut invalid = compute.clone();
    invalid.at_s = 1.0;
    assert!(sim.submit_actions(vec![invalid]).is_err());
    assert!(sim.config().actions.is_empty());
    sim.submit_actions(vec![compute]).unwrap();
    let replay_config = sim.config().clone();
    sim.run_until(5.0).unwrap();
    let mut replay = Simulator::new(replay_config, Static::new(3)).unwrap();
    replay.run_until(5.0).unwrap();
    assert_eq!(
        serde_json::to_value(sim.report()).unwrap(),
        serde_json::to_value(replay.report()).unwrap()
    );
}

#[test]
fn compute_and_transfers_share_the_same_energy_budget() {
    let mut c = config();
    c.satellite.battery = Some(BatteryConfig {
        capacity_j: 100.0,
        initial_j: 30.0,
        reserve_j: 10.0,
        solar_w: 0.0,
        idle_w: 0.0,
        compute_w: 10.0,
        isl_tx_w: 20.0,
        isl_rx_w: 0.0,
        ..Default::default()
    });
    initial(&mut c, S0, "image", 100);
    compute(&mut c, "cpu", 0, 1.0, vec![]);
    transfer(&mut c, "send", "image", vec![S0, S1]);
    let mut sim = Simulator::new(c, Static::new(3).edge(S0, S1, 100.0, 0.0)).unwrap();
    sim.step().unwrap();
    assert_eq!(sim.metrics().compute_work, 1.0);
    assert_eq!(sim.metrics().isl_byte_hops, 50.0);
    assert_eq!(sim.report().satellites[0].energy_j, Some(10.0));
}

#[test]
fn smoke_scenario_delivers_only_the_computed_result() {
    let c: Scenario = serde_json::from_str(include_str!("../scenarios/smoke.json")).unwrap();
    let env = OrbitEnvironment::new(&c).unwrap();
    let mut sim = Simulator::new(c, env).unwrap();
    sim.run_until(600.0).unwrap();
    assert_eq!(sim.metrics().completed_actions, 2);
    assert_eq!(sim.metrics().compute_work, 5.0);
    assert_eq!(sim.metrics().isl_byte_hops, 1_000_000.0);
    assert_eq!(sim.metrics().downlink_bytes, 1_000_000.0);
    assert!(sim.has_object(G0, "result"));
    assert!(!sim.has_object(G0, "input"));
}

#[test]
fn action_json_rejects_misspelled_fields() {
    let bad = r#"{"id":"x","at_s":0,"type":"compute","satellite":0,"work":1,"output":{"id":"x","bytes":1},"workk":1}"#;
    assert!(serde_json::from_str::<ScheduledAction>(bad).is_err());
}
