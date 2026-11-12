//! Generate a reproducible candidate DAG from the GUI's combined AOI/jobs save.
use leo_capture::{
    resources,
    scheduling::{compact, generate, Config},
};
fn main() {
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let args: Vec<_> = std::env::args_os().collect();
        if args.len() == 3 && args[1] == "--compact" {
            let summary = compact(std::path::Path::new(&args[2]))?;
            println!(
                "Compacted {} satellite graphs to {} nodes / {} edges",
                summary.satellites.len(),
                summary.total_nodes,
                summary.total_edges
            );
            return Ok(());
        }
        if args.len() != 2 {
            return Err("Usage: leo-scheduling-graph CONFIG.json".into());
        }
        let path = std::fs::canonicalize(&args[1])?;
        let mut config: Config = serde_json::from_reader(std::fs::File::open(&path)?)?;
        let base = path.parent().unwrap_or(std::path::Path::new("."));
        config.preset = base.join(&config.preset);
        config.output = base.join(&config.output);
        resources::set_worker_limit(config.workers);
        resources::background_thread();
        let summary = generate(&config)?;
        println!(
            "{} satellites, {} actions, {} nodes, {} edges, {:.1} MiB compressed, {:.1} s wall; {}",
            summary.satellites.len(),
            summary.total_actions,
            summary.total_nodes,
            summary.total_edges,
            summary.graph_and_action_bytes as f64 / 1048576.0,
            summary.elapsed_s,
            config.output.join("manifest.json").display()
        );
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("Graph generation failed: {error}");
        std::process::exit(1);
    }
}
