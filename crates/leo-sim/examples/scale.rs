//! Infrastructure stress fixture, not an application/workload model.
use leo_sim::{
    orbit::WalkerShell, Action, Object, OrbitEnvironment, Scenario, ScheduledAction, Simulator,
};
fn main() {
    let scenario = Scenario {
        step_s: 10.0,
        duration_s: 86400.0,
        event_capacity: 128,
        shells: vec![WalkerShell {
            planes: 16,
            satellites_per_plane: 64,
            ..Default::default()
        }],
        ..Default::default()
    };
    let n = scenario.satellite_count();
    let env = OrbitEnvironment::new(&scenario).unwrap();
    let mut sim = Simulator::new(scenario, env).unwrap();
    sim.submit_actions(
        (0..n)
            .map(|s| ScheduledAction {
                id: format!("compute-{s}"),
                at_s: 0.0,
                action: Action::Compute {
                    satellite: s,
                    work: 86400.0,
                    inputs: vec![],
                    output: Object {
                        id: format!("out-{s}"),
                        bytes: 1,
                    },
                },
            })
            .collect(),
    )
    .unwrap();
    let start = std::time::Instant::now();
    sim.run_until(86400.0).unwrap();
    assert!((sim.metrics().compute_work - n as f64 * 86400.0 * 0.2).abs() < 1e-4);
    println!("{n} satellites, 1 simulated day, 10 s steps, {:.3} wall seconds; {:.0} compute units, {} retained events",start.elapsed().as_secs_f64(),sim.metrics().compute_work,sim.report().recent_events.len());
}
