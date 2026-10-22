use leo_sim::{OrbitEnvironment, Scenario, Simulator};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    time::Instant,
};
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        return Err("Usage: leo-sim SCENARIO.json REPORT.json [EVENTS.jsonl]".into());
    }
    let scenario: Scenario = serde_json::from_reader(BufReader::new(File::open(&args[1])?))?;
    let environment = OrbitEnvironment::new(&scenario)?;
    let mut sim = Simulator::new(scenario.clone(), environment)?;
    let mut trace = if let Some(path) = args.get(3) {
        Some(BufWriter::new(File::create(path)?))
    } else {
        None
    };
    let start = Instant::now();
    if let Some(out) = &mut trace {
        // Drain each fixed step: bounded in-memory history. Overflow is reported.
        while sim.time_s() < scenario.duration_s {
            sim.step()?;
            for e in sim.drain_events() {
                serde_json::to_writer(&mut *out, &e)?;
                writeln!(out)?;
            }
        }
        out.flush()?;
    } else {
        sim.run_until(scenario.duration_s)?;
    }
    let report = sim.report();
    if trace.is_some() && report.events_dropped > 0 {
        eprintln!(
            "Warning: {} events dropped; increase event_capacity for a complete trace",
            report.events_dropped
        );
    }
    let envelope = serde_json::json!({"simulator_version":env!("CARGO_PKG_VERSION"),"source_revision":option_env!("LEO_SIM_REVISION").unwrap_or("unspecified"),"wall_seconds":start.elapsed().as_secs_f64(),"scenario":scenario,"report":report});
    let mut output = BufWriter::new(File::create(&args[2])?);
    serde_json::to_writer_pretty(&mut output, &envelope)?;
    output.flush()?;
    eprintln!(
        "Simulated {:.3} s in {:.3} s; {} completed, {} rejected",
        sim.time_s(),
        start.elapsed().as_secs_f64(),
        sim.metrics().completed_actions,
        sim.metrics().rejected_actions
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("leo-sim: {e}");
        std::process::exit(1);
    }
}
