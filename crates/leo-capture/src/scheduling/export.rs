use super::{eligible, sample, Action, Mode, Options, Topology};
use crate::{
    baseline, dataset,
    geometry::{self, Atlas, Region},
    parallel, resources, storage,
    workload::{hash, Illumination, Job, TimingSummary},
};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use geo::Area;
use leo_sim::{
    geometry::{
        gmst, ground_position, local_solar_time_hours, sun_direction, EARTH_RADIUS_KM as R,
    },
    orbit::Elements,
    worldview::Fleet,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub preset: PathBuf,
    pub output: PathBuf,
    /// Orbit elements and the start of the graph share this absolute epoch.
    pub epoch_unix_s: f64,
    #[serde(default)]
    pub options: Options,
    #[serde(default = "default_workers")]
    pub workers: usize,
    #[serde(default)]
    pub first_satellite: usize,
    #[serde(default = "default_satellites")]
    pub satellite_count: usize,
}
fn default_workers() -> usize {
    resources::worker_limit()
}
fn default_satellites() -> usize {
    60
}

/// Read the existing combined save without changing or regenerating anything.
pub fn load_preset(path: &Path) -> Result<(Vec<Job>, Vec<Region>, u64), String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let checksum = hash(&bytes);
    let aois = dataset::decode(&bytes)?;
    #[derive(Deserialize)]
    struct Envelope {
        schema_version: u32,
        preset_version: Option<u32>,
        workload: Option<storage::Document>,
    }
    let reader: Box<dyn Read> = if bytes.starts_with(&[0x1f, 0x8b]) {
        Box::new(GzDecoder::new(bytes.as_slice()))
    } else {
        Box::new(bytes.as_slice())
    };
    let doc: Envelope = serde_json::from_reader(BufReader::new(reader.take(256 * 1024 * 1024 + 1)))
        .map_err(|e| e.to_string())?;
    if doc.schema_version != 1 || doc.preset_version.is_some_and(|v| v != 1) {
        return Err("Unsupported combined preset version".into());
    }
    let jobs = doc
        .workload
        .ok_or("The preset has no assigned jobs")?
        .into_current()?
        .jobs;
    TimingSummary::new(&jobs).validate()?;
    let keys: BTreeMap<_, _> = aois.iter().map(|(a, _)| (a.key(), a)).collect();
    if keys.len() != aois.len()
        || keys.len() != jobs.len()
        || jobs.iter().any(|j| keys.get(&j.aoi.key()) != Some(&&j.aoi))
    {
        return Err("Saved jobs and AOI geometries do not match".into());
    }
    Ok((jobs, aois.into_iter().map(|(_, r)| r).collect(), checksum))
}

#[derive(Serialize, Deserialize)]
pub struct SatelliteSummary {
    pub satellite: usize,
    pub actions: u64,
    pub nodes: u64,
    pub edges: u64,
    pub maneuver_edges: u64,
    pub crop_records: u64,
    pub jobs_with_candidates: usize,
    pub source_unreachable_actions: usize,
    pub actions_bytes: u64,
    pub graph_bytes: u64,
    pub elapsed_s: f64,
    pub observed_process_memory_bytes: u64,
    #[serde(skip)]
    job_hits: Vec<bool>,
}
#[derive(Serialize, Deserialize)]
pub struct Summary {
    pub schema_version: u32,
    pub status: String,
    pub config: Config,
    pub preset_hash_fnv1a64: u64,
    pub job_count: usize,
    pub jobs_with_due_deadlines: usize,
    pub due_observations: u64,
    pub jobs_with_no_due_deadline: usize,
    pub jobs_with_candidates: usize,
    pub cadence_job_counts: BTreeMap<String, usize>,
    pub modes: Vec<Mode>,
    pub slew_rate_deg_s: f64,
    pub total_actions: u64,
    pub total_nodes: u64,
    pub total_edges: u64,
    pub total_crop_records: u64,
    pub graph_and_action_bytes: u64,
    pub observed_process_memory_bytes: u64,
    pub elapsed_s: f64,
    pub satellites: Vec<SatelliteSummary>,
    pub logical_cpus: usize,
    pub effective_worker_limit: usize,
    pub duty_percent: usize,
    #[serde(default = "worldview_shell")]
    pub constellation: leo_sim::orbit::WalkerShell,
    pub assumptions: Vec<String>,
}
fn worldview_shell() -> leo_sim::orbit::WalkerShell {
    Fleet::WorldView60.shell()
}

type JsonLines = GzEncoder<BufWriter<File>>;
fn check_disk(path: &Path) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::ffi::OsStrExt;
        let path =
            std::ffi::CString::new(path.as_os_str().as_bytes()).map_err(|e| e.to_string())?;
        let mut info: libc::statvfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statvfs(path.as_ptr(), &mut info) } != 0 {
            return Err(format!(
                "Cannot check output disk space: {}",
                std::io::Error::last_os_error()
            ));
        }
        if info.f_bavail as u64 * (info.f_frsize as u64) < 1024 * 1024 * 1024 {
            return Err("Graph generation stopped with less than 1 GiB free disk space; output is incomplete".into());
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = path;
    Ok(())
}
fn writer(path: &Path) -> Result<JsonLines, String> {
    Ok(GzEncoder::new(
        BufWriter::new(File::create(path).map_err(|e| e.to_string())?),
        Compression::fast(),
    ))
}
fn line(w: &mut impl Write, v: &impl Serialize) -> Result<(), String> {
    serde_json::to_writer(&mut *w, v).map_err(|e| e.to_string())?;
    w.write_all(b"\n").map_err(|e| e.to_string())
}
fn finish(w: JsonLines, part: &Path, final_path: &Path) -> Result<u64, String> {
    let mut file = w.finish().map_err(|e| e.to_string())?;
    file.flush().map_err(|e| e.to_string())?;
    file.get_ref().sync_all().map_err(|e| e.to_string())?;
    drop(file);
    fs::rename(part, final_path).map_err(|e| e.to_string())?;
    Ok(fs::metadata(final_path).map_err(|e| e.to_string())?.len())
}

#[derive(Serialize)]
struct CropRecord {
    job: usize,
    slot: u64,
    patch: usize,
    start_s: f64,
    end_s: f64,
    area_km2: f64,
    polygons_xy_km: geometry::Polygons,
}
#[derive(Serialize)]
struct ActionRecord<'a> {
    id: u32,
    tick: u32,
    mode: usize,
    crops: &'a [CropRecord],
}
#[derive(Serialize)]
struct NodeRecord {
    action: u32,
    tick: u32,
    mode: usize,
    next_same: Option<u32>,
    maneuver_to: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    maneuver_hub: Option<usize>,
}

fn write_graph(
    config: &Config,
    id: usize,
    topology: &Topology,
) -> Result<(u64, u64, u64, u64), String> {
    let path = config
        .output
        .join(format!("satellite-{id:03}.graph.jsonl.gz"));
    let part = path.with_extension("gz.part");
    let mut graph = writer(&part)?;
    let source = topology.source_targets();
    let hubs = topology.hubs();
    let base = 2 * topology.actions.len() + 2;
    let nodes = base + hubs.as_ref().map_or(0, |h| h.hubs.len());
    line(
        &mut graph,
        &serde_json::json!({"schema_version":2,"satellite":id,"action_count":topology.actions.len(),
        "source":base-2,"sink":base-1,"source_to":source,"nodes":nodes,
        "transition_encoding":if hubs.is_some(){"shared_uniform_lag_hubs"}else{"direct"},
        "rules":"Action a: start=2a,end=2a+1. Parallel capture/skip start->end; next_same and maneuver_to connect end->target start. maneuver_hub connects end->hub node; a hub's targets are action IDs. Last action in each mode connects end->sink. Source also connects directly to sink. source_to contains action IDs."}),
    )?;
    let mut edges = 3 * topology.actions.len() as u64 + source.len() as u64 + 1;
    let mut maneuvers = 0;
    for (id, a) in topology.actions.iter().enumerate() {
        if id % 1024 == 0 {
            resources::check_memory()?;
        }
        let targets = if hubs.is_some() {
            Vec::new()
        } else {
            topology.targets(id)
        };
        let hub = hubs
            .as_ref()
            .and_then(|h| h.from_action[id])
            .map(|h| base + h);
        maneuvers += targets.len() as u64 + u64::from(hub.is_some());
        line(
            &mut graph,
            &NodeRecord {
                action: id as u32,
                tick: a.tick,
                mode: a.mode,
                next_same: topology.next_same[id],
                maneuver_to: targets,
                maneuver_hub: hub,
            },
        )?;
    }
    if let Some(hubs) = hubs {
        for (id, hub) in hubs.hubs.iter().enumerate() {
            maneuvers += hub.targets.len() as u64;
            line(
                &mut graph,
                &serde_json::json!({"hub":base+id,"ready_tick":hub.ready_tick,"targets":hub.targets}),
            )?;
        }
    }
    edges += maneuvers;
    let bytes = finish(graph, &part, &path)?;
    Ok((nodes as u64, edges, maneuvers, bytes))
}

fn masks(jobs: &[Job], config: &Config) -> Result<Vec<Vec<bool>>, String> {
    let bytes = jobs.len() as u64 * u64::from(config.options.ticks());
    if bytes > resources::memory_limit() / 3 {
        return Err(
            "Eligibility masks would exceed one third of the memory budget; shorten the horizon"
                .into(),
        );
    }
    parallel::map(
        (0..config.options.ticks()).collect(),
        config.workers.min(resources::worker_limit()),
        |tick| {
            let start = f64::from(tick) * config.options.step_s;
            let end = start + config.options.step_s;
            let hours = [start, (start + end) / 2.0, end].map(|t| {
                local_solar_time_hours(
                    ground_position(0.0, 0.0, 1.0, gmst(config.epoch_unix_s + t)),
                    sun_direction(config.epoch_unix_s + t),
                )
            });
            Ok(jobs
                .iter()
                .map(|j| {
                    let phase = j.acquisition.phase_hours * 3600.0;
                    let cadence = j.cadence_hours * 3600.0;
                    if start < phase
                        || end
                            > phase
                                + (((start - phase) / cadence + 1e-10).floor() + 1.0) * cadence
                                + 1e-9
                    {
                        return eligible(j, config.epoch_unix_s, start, end);
                    }
                    j.acquisition.illumination == Illumination::AnyTime
                        || hours.iter().all(|h| {
                            j.acquisition
                                .accepts((h + j.aoi.longitude / 15.0).rem_euclid(24.0), 90.0)
                        })
                })
                .collect())
        },
        |_, _| {},
    )
}

fn satellite(
    config: &Config,
    jobs: &[Job],
    atlas: &Atlas,
    modes: &[Mode],
    masks: &[Vec<bool>],
    id: usize,
    elements: Elements,
) -> Result<SatelliteSummary, String> {
    resources::background_thread();
    let started = Instant::now();
    let name = format!("satellite-{id:03}");
    let actions_path = config.output.join(format!("{name}.actions.jsonl.gz"));
    let actions_part = actions_path.with_extension("gz.part");
    let mut output = writer(&actions_part)?;
    let mut actions = Vec::new();
    let mut crop_count = 0;
    let mut job_hits = vec![false; jobs.len()];
    let mut peak = resources::process_memory().unwrap_or(0);
    let options = config.options.baseline();
    let mut previous = sample(&elements, config.epoch_unix_s, 0.0);
    let mut last_notice = Instant::now();
    for tick in 0..config.options.ticks() {
        resources::check_memory()?;
        let active = Instant::now();
        let start = f64::from(tick) * config.options.step_s;
        let end = start + config.options.step_s;
        let next = sample(&elements, config.epoch_unix_s, end);
        // No narrow-strip overlay if even the entire reach misses the atlas.
        if !atlas
            .candidates(previous.0, next.0, config.options.reach_km / (2.0 * R))
            .is_empty()
        {
            for (mode_id, mode) in modes.iter().enumerate() {
                let (a, na) = mode.offset(previous.0, previous.1);
                let (b, nb) = mode.offset(next.0, next.1);
                let segment = baseline::segment(
                    jobs,
                    atlas,
                    config.epoch_unix_s,
                    config.options.width_km / (2.0 * R),
                    &options,
                    id,
                    start,
                    end,
                    a,
                    na,
                    b,
                    nb,
                    Some(&masks[tick as usize]),
                )?;
                let mut crops = Vec::new();
                for op in segment.ops {
                    for c in op.crops {
                        job_hits[c.job] = true;
                        crops.push(CropRecord {
                            job: c.job,
                            slot: c.slot,
                            patch: op.patch,
                            start_s: c.start_s,
                            end_s: c.end_s,
                            area_km2: c.geometry.unsigned_area(),
                            polygons_xy_km: geometry::coordinates(&c.geometry),
                        });
                    }
                }
                if !crops.is_empty() {
                    let action = u32::try_from(actions.len()).map_err(|_| "Too many actions")?;
                    crop_count += crops.len() as u64;
                    line(
                        &mut output,
                        &ActionRecord {
                            id: action,
                            tick,
                            mode: mode_id,
                            crops: &crops,
                        },
                    )?;
                    actions.push(Action {
                        tick,
                        mode: mode_id,
                    });
                }
            }
        }
        previous = next;
        if tick % 32 == 0 {
            check_disk(&config.output)?;
            peak = peak.max(resources::process_memory().unwrap_or(0));
        }
        if last_notice.elapsed().as_secs() >= 30 {
            eprintln!(
                "satellite {id:02}: {:.1}/{:.1} simulated h, {} actions, {:.0} s wall",
                end / 3600.0,
                config.options.duration_s / 3600.0,
                actions.len(),
                started.elapsed().as_secs_f64()
            );
            last_notice = Instant::now();
        }
        let duty = resources::duty_percent() as f64 / 100.0;
        if duty < 1.0 {
            std::thread::sleep(active.elapsed().mul_f64((1.0 - duty) / duty));
        }
    }
    let actions_bytes = finish(output, &actions_part, &actions_path)?;
    let topology = Topology::new(actions, modes, &config.options)?;
    let (nodes, edges, maneuvers, graph_bytes) = write_graph(config, id, &topology)?;
    let unreachable = topology.reachable().iter().filter(|x| !**x).count();
    peak = peak.max(resources::process_memory().unwrap_or(0));
    Ok(SatelliteSummary {
        satellite: id,
        actions: topology.actions.len() as u64,
        nodes,
        edges,
        maneuver_edges: maneuvers,
        crop_records: crop_count,
        jobs_with_candidates: job_hits.iter().filter(|x| **x).count(),
        source_unreachable_actions: unreachable,
        actions_bytes,
        graph_bytes,
        elapsed_s: started.elapsed().as_secs_f64(),
        observed_process_memory_bytes: peak,
        job_hits,
    })
}

pub fn generate(config: &Config) -> Result<Summary, String> {
    config.options.validate()?;
    if config.output.exists() {
        return Err("Output must be a new directory; existing runs are never overwritten".into());
    }
    if !config.epoch_unix_s.is_finite()
        || config.workers == 0
        || config.satellite_count == 0
        || config.first_satellite >= 60
        || config.satellite_count > 60 - config.first_satellite
    {
        return Err("Invalid epoch, workers or satellite subset".into());
    }
    let started = Instant::now();
    eprintln!("Reading saved preset {}", config.preset.display());
    let (jobs, regions, checksum) = load_preset(&config.preset)?;
    let mut last = Instant::now();
    eprintln!(
        "Preparing {} AOI polygons with {} workers",
        jobs.len(),
        config.workers
    );
    let atlas = Atlas::with_workers(
        &jobs,
        &regions,
        config.options.cell_deg,
        config.options.edge_km,
        config.workers.min(resources::worker_limit()),
        |p| {
            if last.elapsed().as_secs() >= 5 {
                eprintln!("Geometry {:.0}%", p * 100.0);
                last = Instant::now();
            }
            true
        },
    )?;
    drop(regions);
    let modes = config.options.modes();
    eprintln!(
        "Building solar-time masks; {} pointing modes, {} ticks, {} patches",
        modes.len(),
        config.options.ticks(),
        atlas.patch_count()
    );
    let masks = masks(&jobs, config)?;
    // Refuse to overwrite an earlier run, including an incomplete one.
    if let Some(parent) = config.output.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::create_dir(&config.output).map_err(|e| format!("Output must be a new directory: {e}"))?;
    check_disk(&config.output)?;
    fs::write(
        config.output.join("INCOMPLETE"),
        b"Only manifest.json with status=complete marks a complete run.\n",
    )
    .map_err(|e| e.to_string())?;
    let saved_preset =
        config
            .output
            .join(if config.preset.extension().is_some_and(|x| x == "gz") {
                "preset.json.gz"
            } else {
                "preset.json"
            });
    fs::copy(&config.preset, &saved_preset).map_err(|e| e.to_string())?;
    if hash(&fs::read(&saved_preset).map_err(|e| e.to_string())?) != checksum {
        return Err("Preset changed during preparation; restart from the saved version".into());
    }
    let charts: Vec<_> = atlas
        .patches
        .iter()
        .enumerate()
        .map(|(id, p)| serde_json::json!({"patch":id,"center_unit":p.chart.center}))
        .collect();
    fs::write(
        config.output.join("charts.json"),
        serde_json::to_vec(&charts).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let shell = Fleet::WorldView60.shell();
    let ids: Vec<_> =
        (config.first_satellite..config.first_satellite + config.satellite_count).collect();
    let mut last_done = usize::MAX;
    let satellites = parallel::map(
        ids,
        config.workers.min(resources::worker_limit()),
        |id| {
            satellite(
                config,
                &jobs,
                &atlas,
                &modes,
                &masks,
                id,
                shell.elements(id / 3, id % 3),
            )
        },
        |done, total| {
            if done == last_done {
                return;
            }
            last_done = done;
            eprintln!(
                "Completed {done}/{total} satellite graphs; {:.1} min wall",
                started.elapsed().as_secs_f64() / 60.0
            )
        },
    )?;
    let mut seen = vec![false; jobs.len()];
    for s in &satellites {
        for (a, b) in seen.iter_mut().zip(&s.job_hits) {
            *a |= *b;
        }
    }
    let mut due = 0;
    let mut due_jobs = 0;
    let mut cadences = BTreeMap::new();
    for job in &jobs {
        let count = (((config.options.duration_s - job.acquisition.phase_hours * 3600.0).max(0.0)
            / (job.cadence_hours * 3600.0))
            + 1e-10)
            .floor() as u64;
        due += count;
        due_jobs += usize::from(count > 0);
        *cadences
            .entry(format!("{} h", job.cadence_hours))
            .or_insert(0) += 1;
    }
    let summary=Summary {schema_version:2,status:"complete".into(),config:config.clone(),preset_hash_fnv1a64:checksum,
        job_count:jobs.len(),jobs_with_due_deadlines:due_jobs,due_observations:due,jobs_with_no_due_deadline:jobs.len()-due_jobs,
        jobs_with_candidates:seen.iter().filter(|x|**x).count(),cadence_job_counts:cadences,
        slew_rate_deg_s:config.options.slew_rate_rad_s().to_degrees(),modes,
        total_actions:satellites.iter().map(|s|s.actions).sum(),total_nodes:satellites.iter().map(|s|s.nodes).sum(),
        total_edges:satellites.iter().map(|s|s.edges).sum(),total_crop_records:satellites.iter().map(|s|s.crop_records).sum(),
        graph_and_action_bytes:satellites.iter().map(|s|s.graph_bytes+s.actions_bytes).sum(),
        observed_process_memory_bytes:satellites.iter().map(|s|s.observed_process_memory_bytes).max().unwrap_or(0),
        elapsed_s:started.elapsed().as_secs_f64(),satellites,
        logical_cpus:resources::cpu_count(),effective_worker_limit:config.workers.min(resources::worker_limit()),duty_percent:resources::duty_percent(),
        constellation:shell,
        assumptions:vec![
            "WorldView-60: 20 planes x 3 satellites, F=14, 617 km, 97.9 degrees, J2; orbit epoch equals planning start. Not live spacecraft ephemerides.".into(),
            "Finite pointing modes follow the orbit in a nadir-relative frame, with constant along/cross ground offsets. Transitions use a time-invariant quaternion surrogate calibrated to the 200 km / 12 s example; the settling allowance is assumed. Tracking of the moving reference frame has no extra cost.".into(),
            "Candidate strips occupy one whole time step. Constant observation width and quality. Circular reach; modes conservatively reserve half the width at the boundary. Footprints use the existing swept-scanline polygon approximation.".into(),
            "Fixed phase/cadence windows, mosaic span=cadence, shared captures allowed. Weekly/monthly opportunities are included even without a deadline in this horizon. No initial imagery or committed actions; initial pointing is nadir.".into(),
            "Every action may be skipped without earning its crop coverage. A wait spine retains the pointing mode. End-to-earliest-start transitions preserve all action subsequences of the dense graph for this finite model.".into(),
            "When every mode change has the same rounded time-step lag, shared transition hubs preserve feasible acquisition sequences. Exact physical slew time is not an additive edge cost through those hubs; evaluate it from consecutive selected acquisitions if needed.".into(),
            "No storage, bandwidth, processing, energy, or cloud constraints. No paths have been optimized and no job or mosaic feasibility verdict is implied. Candidate union is only an upper bound. Finer time/pointing sampling may change results.".into(),
            "Crop coordinates are km in each patch's spherical Lambert azimuthal equal-area chart. Charts are identified in charts.json. Job indices refer to the saved preset's workload.jobs order. Sharing requires polygon union, never sums of crop areas.".into(),
        ]};
    let bytes = serde_json::to_vec_pretty(&summary).map_err(|e| e.to_string())?;
    let part = config.output.join("manifest.json.part");
    fs::write(&part, bytes).map_err(|e| e.to_string())?;
    fs::rename(part, config.output.join("manifest.json")).map_err(|e| e.to_string())?;
    fs::remove_file(config.output.join("INCOMPLETE")).map_err(|e| e.to_string())?;
    Ok(summary)
}

/// Upgrade already-generated topology without recomputing or modifying crops.
/// This also allows retry after an interrupted compaction; schema-1 and schema-2
/// graph records both contain the same ordered action stamps.
pub fn compact(output: &Path) -> Result<Summary, String> {
    let output = output.canonicalize().map_err(|e| e.to_string())?;
    let path = output.join("manifest.json");
    let mut summary: Summary =
        serde_json::from_reader(File::open(&path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if !["complete", "compacting"].contains(&summary.status.as_str()) {
        return Err("Only a fully generated graph can be compacted".into());
    }
    let start = Instant::now();
    summary.config.output = output.clone();
    summary.status = "compacting".into();
    let save = |summary: &Summary| -> Result<(), String> {
        let part = path.with_extension("json.part");
        fs::write(
            &part,
            serde_json::to_vec_pretty(summary).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        fs::rename(part, &path).map_err(|e| e.to_string())
    };
    save(&summary)?;
    for satellite in &mut summary.satellites {
        check_disk(&output)?;
        let graph = output.join(format!(
            "satellite-{:03}.graph.jsonl.gz",
            satellite.satellite
        ));
        let mut actions = Vec::new();
        let reader = BufReader::new(GzDecoder::new(
            File::open(&graph).map_err(|e| e.to_string())?,
        ));
        for line in reader.lines().skip(1) {
            let value: serde_json::Value = serde_json::from_str(&line.map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            if let Some(id) = value["action"].as_u64() {
                if id != actions.len() as u64 {
                    return Err("Unordered action IDs".into());
                }
                actions.push(Action {
                    tick: u32::try_from(value["tick"].as_u64().ok_or("Invalid tick")?)
                        .map_err(|e| e.to_string())?,
                    mode: usize::try_from(value["mode"].as_u64().ok_or("Invalid mode")?)
                        .map_err(|e| e.to_string())?,
                });
            }
        }
        if actions.len() as u64 != satellite.actions {
            return Err("Graph action count disagrees with manifest".into());
        }
        let topology = Topology::new(actions, &summary.modes, &summary.config.options)?;
        let (nodes, edges, maneuvers, bytes) =
            write_graph(&summary.config, satellite.satellite, &topology)?;
        satellite.nodes = nodes;
        satellite.edges = edges;
        satellite.maneuver_edges = maneuvers;
        satellite.graph_bytes = bytes;
    }
    summary.schema_version = 2;
    summary.status = "complete".into();
    summary.total_nodes = summary.satellites.iter().map(|s| s.nodes).sum();
    summary.total_edges = summary.satellites.iter().map(|s| s.edges).sum();
    summary.graph_and_action_bytes = summary
        .satellites
        .iter()
        .map(|s| s.graph_bytes + s.actions_bytes)
        .sum();
    summary.elapsed_s += start.elapsed().as_secs_f64();
    if !summary
        .assumptions
        .iter()
        .any(|a| a.contains("shared transition hubs"))
    {
        summary.assumptions.push("When every mode change has the same rounded time-step lag, shared transition hubs preserve feasible acquisition sequences. Exact physical slew time is not an additive edge cost through those hubs; evaluate it from consecutive selected acquisitions if needed.".into());
    }
    save(&summary)?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workload::{Acquisition, AoiRef, Generator, Pipeline};
    fn jobs() -> Vec<Job> {
        (-2..=2)
            .map(|i| Job {
                aoi: AoiRef {
                    source: "test".into(),
                    id: i.to_string(),
                    area_km2: 1.0,
                    longitude: f64::from(i) * 70.0,
                    latitude: f64::from(i) * 30.0,
                    geometry_hash: 1,
                },
                pipeline: Pipeline::Map,
                samples: 1,
                cadence_hours: 24.0,
                map_bytes_per_km2: 0.0,
                window_bytes_per_km2: 0.0,
                acquisition: Acquisition {
                    phase_hours: 0.004,
                    solar_time_hours: 12.0,
                    solar_tolerance_hours: 1.0,
                    ..Default::default()
                },
            })
            .collect()
    }
    #[test]
    fn cached_solar_masks_match_direct_geometry_including_phase_and_date_wrap() {
        let config = Config {
            preset: PathBuf::new(),
            output: PathBuf::new(),
            epoch_unix_s: 1794528000.0,
            options: Options {
                duration_s: 86400.0,
                ..Default::default()
            },
            workers: 1,
            first_satellite: 0,
            satellite_count: 1,
        };
        let jobs = jobs();
        let masks = masks(&jobs, &config).unwrap();
        for (tick, mask) in masks.iter().enumerate() {
            for (j, &result) in jobs.iter().zip(mask) {
                assert_eq!(
                    result,
                    eligible(
                        j,
                        config.epoch_unix_s,
                        tick as f64 * 30.0,
                        (tick + 1) as f64 * 30.0
                    ),
                    "tick {tick}, {}",
                    j.aoi.longitude
                );
            }
        }
    }
    #[test]
    fn exported_graph_has_shared_polygon_crops_and_preserves_saved_input() {
        let root = std::env::temp_dir().join(format!(
            "leo-graph-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let preset = root.join("preset.json");
        let epoch = 1794528000.0;
        let q = sample(&Fleet::WorldView60.shell().elements(0, 0), epoch, 15.0).0;
        let lon = (-q[2]).atan2(q[0]).to_degrees();
        let lat = q[1].asin().to_degrees();
        let polygons = vec![vec![vec![
            [lon - 0.2, lat - 0.2],
            [lon + 0.2, lat - 0.2],
            [lon + 0.2, lat + 0.2],
            [lon - 0.2, lat + 0.2],
            [lon - 0.2, lat - 0.2],
        ]]];
        let refs: Vec<_> = (0..2)
            .map(|i| AoiRef {
                source: "gdp".into(),
                id: i.to_string(),
                area_km2: 2000.0,
                longitude: lon,
                latitude: lat,
                geometry_hash: hash(&serde_json::to_vec(&polygons).unwrap()),
            })
            .collect();
        let generator = Generator::default();
        let mut jobs = generator.generate(refs.clone()).unwrap();
        for j in &mut jobs {
            j.cadence_hours = 4.0;
            j.samples = 1;
            j.pipeline = Pipeline::Map;
            j.acquisition.illumination = Illumination::AnyTime;
        }
        // A phase inside the first strip must clip spatially, rather than credit
        // the whole strip to a window that has not started yet.
        jobs[1].acquisition.phase_hours = 0.004;
        let aois:Vec<_>=refs.iter().map(|a|serde_json::json!({"id":a.id,"source_kind":a.source,"area_km2":a.area_km2,"anchor_lon":a.longitude,"anchor_lat":a.latitude,"geometry":{"type":"MultiPolygon","coordinates":polygons}})).collect();
        let doc = serde_json::json!({"schema_version":1,"preset_version":1,"config":{"count":2},"aois":aois,
            "workload":storage::Document{schema_version:3,generator,capture:Default::default(),jobs}});
        fs::write(&preset, serde_json::to_vec(&doc).unwrap()).unwrap();
        let before = fs::read(&preset).unwrap();
        let config = Config {
            preset: preset.clone(),
            output: root.join("graph"),
            epoch_unix_s: epoch,
            options: Options {
                duration_s: 60.0,
                reach_km: 40.0,
                along_offsets_km: vec![0.0],
                ..Default::default()
            },
            workers: 1,
            first_satellite: 0,
            satellite_count: 1,
        };
        let summary = generate(&config).unwrap();
        assert!(summary.total_actions > 0);
        assert_eq!(summary.jobs_with_candidates, 2);
        assert_eq!(summary.due_observations, 0);
        assert_eq!(summary.total_nodes, summary.total_actions * 2 + 2);
        let content = GzDecoder::new(
            File::open(config.output.join("satellite-000.actions.jsonl.gz")).unwrap(),
        );
        let records: Vec<serde_json::Value> = std::io::BufRead::lines(BufReader::new(content))
            .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
            .collect();
        assert_eq!(records.len() as u64, summary.total_actions);
        assert!(records.iter().any(|r| r["crops"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["job"] == 0)
            && r["crops"].as_array().unwrap().iter().any(|c| c["job"] == 1)));
        for record in &records {
            for crop in record["crops"].as_array().unwrap() {
                assert!(crop["area_km2"].as_f64().unwrap() > 0.0);
                assert!(!crop["polygons_xy_km"].as_array().unwrap().is_empty());
                if crop["job"] == 1 {
                    assert!(crop["start_s"].as_f64().unwrap() >= 14.4 - 1e-9);
                }
            }
        }
        assert_eq!(fs::read(&preset).unwrap(), before);
        let compacted = compact(&config.output).unwrap();
        assert_eq!(compacted.total_actions, summary.total_actions);
        assert_eq!(compacted.total_edges, summary.total_edges);
        assert!(generate(&config).err().unwrap().contains("new directory"));
        fs::remove_dir_all(root).unwrap();
    }
}
