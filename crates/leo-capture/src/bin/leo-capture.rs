use leo_capture::{
    baseline::{self, Baseline, Options},
    dataset,
    geometry::Atlas,
    storage::Document,
    workload::Generator,
};
use leo_sim::{acquisition::CaptureProfile, orbit::WalkerShell};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    epoch_unix_s: f64,
    #[serde(default)]
    start_s: f64,
    swath_km: f64,
    shells: Vec<WalkerShell>,
    aoi_files: Vec<PathBuf>,
    #[serde(default)]
    workload: Option<PathBuf>,
    #[serde(default)]
    generator: Generator,
    #[serde(default)]
    capture: Option<CaptureProfile>,
    #[serde(default)]
    options: Options,
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("Usage: leo-capture CONFIG.json REPORT.json [OPPORTUNITIES.jsonl]".into());
    }
    let config: Config = serde_json::from_reader(File::open(&args[1])?)?;
    config.options.validate()?;
    if !config.start_s.is_finite() || !config.epoch_unix_s.is_finite() {
        return Err("Invalid start or epoch".into());
    }
    let base = Path::new(&args[1]).parent().unwrap_or(Path::new("."));
    let mut aois = Vec::new();
    let mut regions = Vec::new();
    let mut sources = Vec::new();
    for path in &config.aoi_files {
        let bytes = std::fs::read(base.join(path))?;
        sources.push(
            serde_json::json!({"path":path,"content_hash":leo_capture::workload::hash(&bytes)}),
        );
        for (a, r) in dataset::decode(&bytes)? {
            aois.push(a);
            regions.push(r);
        }
    }
    let workload = if let Some(path) = &config.workload {
        Document::decode(&std::fs::read(base.join(path))?)?
    } else {
        Document {
            schema_version: 2,
            generator: config.generator.clone(),
            capture: config.capture.unwrap_or_default(),
            jobs: config.generator.generate(aois.clone())?,
        }
    };
    workload.validate()?;
    let input: std::collections::BTreeMap<_, _> = aois.iter().map(|a| (a.key(), a)).collect();
    if input.len() != aois.len()
        || input.len() != workload.jobs.len()
        || workload
            .jobs
            .iter()
            .any(|j| input.get(&j.aoi.key()) != Some(&&j.aoi))
    {
        return Err("Workload and AOI identities, geometry, areas or anchors differ".into());
    }
    let mut elements = Vec::new();
    for shell in &config.shells {
        shell.validate()?;
        for p in 0..shell.planes {
            for s in 0..shell.satellites_per_plane {
                elements.push(shell.elements(p, s));
            }
        }
    }
    let wall = std::time::Instant::now();
    eprintln!("Preparing {} polygon AOIs…", regions.len());
    let atlas = Atlas::new(
        &workload.jobs,
        &regions,
        config.options.cell_deg,
        config.options.edge_km,
        |_| true,
    )?;
    eprintln!(
        "{} satellites, {} geographic patches, {} AOI pieces",
        elements.len(),
        atlas.patch_count(),
        atlas.polygon_pieces
    );
    let epoch = config.epoch_unix_s + config.start_s;
    let capture = config.capture.unwrap_or(workload.capture);
    let mut baseline = Baseline::new(
        &workload.jobs,
        &atlas,
        elements.len(),
        epoch,
        config.swath_km,
        capture,
        &config.options,
    )?;
    let trace_temp = args.get(3).map(|p| PathBuf::from(format!("{p}.tmp")));
    let mut trace = trace_temp
        .as_ref()
        .map(|p| File::create(p).map(BufWriter::new))
        .transpose()?;
    if let Some(out) = &mut trace {
        writeln!(
            out,
            "{}",
            serde_json::json!({"type":"metadata","version":1,"epoch_unix_s":epoch,"config":config,"sources":sources,"workload":workload})
        )?;
    }
    let mut last_percent = 0;
    let stats = baseline::opportunities(
        &workload.jobs,
        &atlas,
        elements.len(),
        epoch,
        config.swath_km,
        &config.options,
        |s, t| Ok(elements[s].sample(config.start_s + t).xyz),
        |op| {
            baseline.accept(&op)?;
            if let Some(out) = &mut trace {
                serde_json::to_writer(
                    &mut *out,
                    &baseline::opportunity_json(&op, &workload.jobs, &atlas),
                )
                .map_err(|e| e.to_string())?;
                writeln!(out).map_err(|e| e.to_string())?;
            }
            Ok(())
        },
        |fraction| {
            let percent = (fraction * 100.0) as u32;
            if percent >= last_percent + 10 {
                last_percent = percent;
                eprintln!("{percent}%");
            }
            true
        },
    )?;
    let report = baseline.finish(stats);
    let envelope = serde_json::json!({"capture_version":env!("CARGO_PKG_VERSION"),
        "source_revision":option_env!("LEO_SIM_REVISION").unwrap_or("unspecified"),
        "wall_seconds":wall.elapsed().as_secs_f64(),"config":config,"workload":workload,"sources":sources,"report":report});
    let temp = format!("{}.tmp", args[2]);
    let mut output = BufWriter::new(File::create(&temp)?);
    serde_json::to_writer_pretty(&mut output, &envelope)?;
    output.flush()?;
    std::fs::rename(temp, &args[2])?;
    if let Some(out) = &mut trace {
        out.flush()?;
    }
    if let (Some(temp), Some(path)) = (trace_temp, args.get(3)) {
        std::fs::rename(temp, path)?;
    }
    eprintln!(
        "Captured {:.3} GB; reuse saved {:.3} GB; {:.2} s wall time",
        report.retained_physical_bytes / 1e9,
        report.reuse_saved_bytes / 1e9,
        wall.elapsed().as_secs_f64()
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("leo-capture: {error}");
        std::process::exit(1);
    }
}
