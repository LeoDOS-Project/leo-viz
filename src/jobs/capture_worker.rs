//! The worker owns mutable capture state. Reports are published only after a
//! completed chunk; a worker failure never manufactures acquisition shortfalls.
use leo_capture::{
    baseline::Report,
    geometry::Polygons,
    session::{Inputs, Meters, PreparationProgress, PreparationStage, Session},
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::time::{Duration, Instant};

pub(super) enum Command {
    Run(bool),
    Scene(bool),
    Speed(f64),
    FastForward(f64),
    Save(String),
}
pub(super) enum Initial {
    New(Inputs),
    Load(String),
}
pub(super) enum Event {
    Error(String),
    Paused(String),
    Stopped(String),
    Saved {
        path: String,
        elapsed_s: f64,
        epoch_unix_s: f64,
    },
}
type RingEntry =
    rstar::primitives::GeomWithData<rstar::primitives::Rectangle<[f64; 2]>, (usize, usize, usize)>;
pub(super) struct Scene {
    pub inputs: Arc<Inputs>,
    pub anchors: Vec<egui_plot::PlotPoint>,
    pub satellites: Vec<String>,
    rings: rstar::RTree<RingEntry>,
}
impl Scene {
    fn new(inputs: Arc<Inputs>) -> Self {
        let anchors = inputs
            .jobs
            .iter()
            .map(|j| egui_plot::PlotPoint::new(j.aoi.longitude, j.aoi.latitude))
            .collect();
        let satellites = inputs
            .satellites
            .iter()
            .map(|s| format!("{} / {}", s.constellation, s.name))
            .collect();
        let mut rings = Vec::new();
        for (r, region) in inputs.regions.iter().enumerate() {
            for (p, polygon) in region.polygons.iter().enumerate() {
                for (i, ring) in polygon.iter().enumerate() {
                    let mut min = [f64::INFINITY; 2];
                    let mut max = [f64::NEG_INFINITY; 2];
                    for point in ring {
                        for d in 0..2 {
                            min[d] = min[d].min(point[d]);
                            max[d] = max[d].max(point[d]);
                        }
                    }
                    if !ring.is_empty() {
                        rings.push(RingEntry::new(
                            rstar::primitives::Rectangle::from_corners(min, max),
                            (r, p, i),
                        ));
                    }
                }
            }
        }
        Self {
            inputs,
            anchors,
            satellites,
            rings: rstar::RTree::bulk_load(rings),
        }
    }
    pub fn visible_rings(
        &self,
        min: [f64; 2],
        max: [f64; 2],
    ) -> impl Iterator<Item = &Vec<[f64; 2]>> {
        self.rings
            .locate_in_envelope_intersecting(&rstar::AABB::from_corners(min, max))
            .map(|entry| {
                let (r, p, i) = entry.data;
                &self.inputs.regions[r].polygons[p][i]
            })
    }
}
pub(super) struct View {
    pub scene: Arc<Scene>,
    pub report: Report,
    pub meters: Meters,
    pub positions: Vec<[f64; 2]>,
    pub footprints: Vec<Polygons>,
    pub running: bool,
    pub fast_target: Option<f64>,
    pub finished: bool,
    pub memory_bytes: Option<u64>,
}
pub(super) struct Worker {
    pub commands: mpsc::Sender<Command>,
    pub events: mpsc::Receiver<Event>,
    views: Arc<Mutex<Option<View>>>,
    pub cancel: Arc<AtomicBool>,
    pub progress: Arc<Mutex<PreparationProgress>>,
    pub started: Instant,
    pub fast: Arc<AtomicBool>,
}
impl Worker {
    pub fn take_view(&self) -> Option<View> {
        self.views.lock().unwrap().take()
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "panic without a text message".into())
}
fn stopped_message(context: &str, error: &str) -> String {
    format!("Capture worker stopped while {context}: {error}. Displayed counters are from the last completed update, not this worker failure. Reload a checkpoint to resume.")
}
#[cfg(not(target_arch = "wasm32"))]
impl Worker {
    pub fn spawn(initial: Initial, speed: f64) -> Self {
        Self::spawn_with_memory_check(initial, speed, leo_capture::resources::check_memory)
    }
    fn spawn_with_memory_check(
        initial: Initial,
        speed: f64,
        memory_check: impl Fn() -> Result<(), String> + Send + Sync + 'static,
    ) -> Self {
        let (commands, rx) = mpsc::channel();
        let (etx, events) = mpsc::channel();
        let views = Arc::new(Mutex::new(None));
        let vtx = views.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = cancel.clone();
        let progress = Arc::new(Mutex::new(PreparationProgress {
            stage: PreparationStage::ValidateInputs,
            completed: 0,
            total: 0,
        }));
        let worker_progress = progress.clone();
        let fast = Arc::new(AtomicBool::new(false));
        let worker_fast = fast.clone();
        std::thread::spawn(move || {
            leo_capture::resources::background_thread();
            let context = Mutex::new("preparing the session".to_string());
            let run = || {
                run(
                    initial,
                    speed,
                    rx,
                    &etx,
                    &vtx,
                    &stop,
                    &worker_fast,
                    &worker_progress,
                    &context,
                    &memory_check,
                )
            };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run))
                .unwrap_or_else(|payload| Err(panic_message(payload)));
            worker_fast.store(false, Ordering::Relaxed);
            if !stop.load(Ordering::Relaxed) {
                if let Err(e) = result {
                    let _ = etx.send(Event::Stopped(stopped_message(
                        &context.lock().unwrap(),
                        &e,
                    )));
                }
            }
        });
        Self {
            commands,
            events,
            views,
            cancel,
            progress,
            started: Instant::now(),
            fast,
        }
    }
}
#[cfg(not(target_arch = "wasm32"))]
fn run(
    initial: Initial,
    mut speed: f64,
    rx: mpsc::Receiver<Command>,
    events: &mpsc::Sender<Event>,
    views: &Mutex<Option<View>>,
    stop: &AtomicBool,
    fast: &AtomicBool,
    progress: &Mutex<PreparationProgress>,
    context: &Mutex<String>,
    memory_check: &dyn Fn() -> Result<(), String>,
) -> Result<(), String> {
    let mut running = matches!(&initial, Initial::New(_));
    let mut preparation_error = None;
    let mut preparation = |p: PreparationProgress| {
        *context.lock().unwrap() = format!("preparing {:?} ({}/{})", p.stage, p.completed, p.total);
        *progress.lock().unwrap() = p;
        if let Err(e) = memory_check() {
            preparation_error = Some(e);
            return false;
        }
        !stop.load(Ordering::Relaxed)
    };
    let prepared = match initial {
        Initial::New(i) => {
            leo_capture::workload::TimingSummary::new(&i.jobs).validate()?;
            Session::prepare(i, &mut preparation)
        }
        Initial::Load(path) => {
            Session::load_with_progress(std::path::Path::new(&path), &mut preparation)
        }
    };
    let mut session = prepared.map_err(|e| preparation_error.unwrap_or(e))?;
    let scene = Arc::new(Scene::new(session.shared_inputs()));
    let mut show_scene = false;
    let mut fast_target: Option<f64> = None;
    let mut resume_running = false;
    let mut last_wall = Instant::now();
    let mut debt = 0.0;
    let mut last_view = Instant::now() - Duration::from_secs(1);
    let mut dirty = true;
    let mut pending_command = None;
    loop {
        let work_start = Instant::now();
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        while let Some(command) = pending_command.take().or_else(|| rx.try_recv().ok()) {
            dirty = true;
            match command {
                Command::Scene(value) => show_scene = value,
                Command::Run(value) => {
                    running = value;
                    fast_target = None;
                    debt = 0.0;
                    fast.store(false, Ordering::Relaxed);
                }
                Command::Speed(value) => speed = value.clamp(1.0, 86400.0),
                Command::FastForward(seconds) => {
                    if seconds.is_finite() && seconds > 0.0 {
                        resume_running = running;
                        running = false;
                        debt = 0.0;
                        fast.store(true, Ordering::Relaxed);
                        fast_target = Some(
                            (session.elapsed_s() + seconds)
                                .min(session.inputs().options.duration_s),
                        );
                    }
                }
                Command::Save(path) => {
                    *context.lock().unwrap() = format!(
                        "saving the checkpoint at {:.3} simulated seconds",
                        session.elapsed_s()
                    );
                    match session.save(std::path::Path::new(&path)) {
                        Ok(()) => {
                            let _ = events.send(Event::Saved {
                                path,
                                elapsed_s: session.elapsed_s(),
                                epoch_unix_s: session.inputs().epoch(),
                            });
                        }
                        Err(e) => {
                            let _ = events.send(Event::Error(e));
                        }
                    }
                }
            }
            last_wall = Instant::now();
        }
        let now = Instant::now();
        let wall = now.duration_since(last_wall).as_secs_f64();
        last_wall = now;
        if !session.finished() {
            let steps = if let Some(target) = fast_target {
                ((target - session.elapsed_s()) / session.inputs().options.step_s)
                    .ceil()
                    .max(0.0) as u64
            } else if running {
                debt += wall * speed;
                (debt / session.inputs().options.step_s).floor() as u64
            } else {
                0
            };
            // One tick is the smallest safe interruption/checkpoint boundary.
            let steps = steps.min(1);
            if steps > 0 {
                if let Err(e) = memory_check() {
                    running = false;
                    fast_target = None;
                    fast.store(false, Ordering::Relaxed);
                    debt = 0.0;
                    dirty = true;
                    let _ = events.send(Event::Paused(format!(
                        "{e} Paused at a completed tick; the current session can still be saved."
                    )));
                } else {
                    *context.lock().unwrap() = format!(
                        "advancing capture from {:.3} to {:.3} simulated seconds",
                        session.elapsed_s(),
                        (session.elapsed_s() + steps as f64 * session.inputs().options.step_s)
                            .min(session.inputs().options.duration_s)
                    );
                    session.advance(steps)?;
                    dirty = true;
                    if fast_target.is_none() {
                        debt -= steps as f64 * session.inputs().options.step_s;
                    }
                }
            }
        }
        if fast_target.is_some_and(|target| session.elapsed_s() >= target) {
            fast_target = None;
            fast.store(false, Ordering::Relaxed);
            running = resume_running;
            debt = 0.0;
            last_wall = Instant::now();
            dirty = true;
        }
        if session.finished() {
            running = false;
            fast_target = None;
            fast.store(false, Ordering::Relaxed);
        }
        if dirty
            && (last_view.elapsed() >= Duration::from_millis(250)
                || (!running && fast_target.is_none()))
        {
            *context.lock().unwrap() = format!(
                "preparing the display at {:.3} simulated seconds",
                session.elapsed_s()
            );
            let report = session.report().summary(100);
            let footprints = if show_scene && fast_target.is_none() {
                let mut vertices = 0;
                session
                    .state()
                    .captures
                    .iter()
                    .rev()
                    .take_while(|c| c.end_s > session.elapsed_s() - 600.0)
                    .take(2000)
                    .take_while(|c| {
                        use geo::CoordsIter;
                        vertices += c.geometry.coords_count();
                        vertices <= 100_000
                    })
                    .map(|c| session.footprint(c))
                    .collect()
            } else {
                Vec::new()
            };
            let view = View {
                scene: scene.clone(),
                report,
                meters: session.meters(),
                positions: if show_scene && fast_target.is_none() {
                    session.ground_positions()?
                } else {
                    Vec::new()
                },
                footprints,
                running,
                fast_target,
                finished: session.finished(),
                memory_bytes: leo_capture::resources::process_memory(),
            };
            // Latest-only mailbox: no stale backlog or retry/rebuild spin when
            // the UI is busy. At most one unpublished snapshot is retained.
            *views.lock().unwrap() = Some(view);
            last_view = Instant::now();
            dirty = false;
        }
        let duty = leo_capture::resources::duty_percent() as f64 / 100.0;
        let delay = work_start.elapsed().mul_f64((1.0 - duty) / duty).max(
            if !running && fast_target.is_none() {
                Duration::from_millis(20)
            } else {
                Duration::from_millis(1)
            },
        );
        // Short sleeps keep pause/save/cancel responsive even after a slow tick.
        let until = Instant::now() + delay;
        while Instant::now() < until && !stop.load(Ordering::Relaxed) {
            match rx.recv_timeout(
                until
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(20)),
            ) {
                Ok(command) => {
                    pending_command = Some(command);
                    break;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(()),
                Err(mpsc::RecvTimeoutError::Timeout) => (),
            }
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(super) mod tests {
    use super::*;
    pub(crate) fn inputs() -> Inputs {
        let aois = leo_capture::dataset::decode(include_bytes!(
            "../../crates/leo-capture/scenarios/overlap-aois.json"
        ))
        .unwrap();
        let mut generator = leo_capture::workload::Generator::default();
        generator.acquisition.solar_tolerance_hours = 12.0;
        let jobs = generator
            .generate(aois.iter().map(|(a, _)| a.clone()))
            .unwrap();
        let shell = leo_sim::orbit::WalkerShell {
            planes: 1,
            satellites_per_plane: 1,
            ..Default::default()
        };
        Inputs {
            orbit_epoch_unix_s: 946728000.0,
            start_s: 0.0,
            satellites: vec![shell.elements(0, 0)]
                .into_iter()
                .enumerate()
                .map(|(i, e)| leo_capture::session::Satellite {
                    name: i.to_string(),
                    constellation: "test".into(),
                    orbit: leo_capture::session::Orbit::Analytical { elements: e },
                })
                .collect(),
            jobs,
            regions: aois.into_iter().map(|(_, r)| r).collect(),
            capture: Default::default(),
            swath_km: 20.0,
            options: leo_capture::baseline::Options {
                duration_s: 7200.0,
                ..Default::default()
            },
        }
    }
    fn view_until(w: &Worker, predicate: impl Fn(&View) -> bool) -> View {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            for event in w.events.try_iter() {
                if let Event::Error(e) | Event::Stopped(e) | Event::Paused(e) = event {
                    panic!("{e}");
                }
            }
            assert!(Instant::now() < deadline, "capture worker timeout");
            let Some(view) = w.take_view() else {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            };
            if predicate(&view) {
                return view;
            }
        }
    }
    #[test]
    fn memory_pause_preserves_save_and_resume_and_hidden_map_does_no_work() {
        let limited = Arc::new(AtomicBool::new(false));
        let flag = limited.clone();
        let w = Worker::spawn_with_memory_check(Initial::New(inputs()), 1.0, move || {
            if flag.load(Ordering::Relaxed) {
                Err("Test memory threshold".into())
            } else {
                Ok(())
            }
        });
        let initial = view_until(&w, |_| true);
        assert!(initial.positions.is_empty() && initial.footprints.is_empty());
        w.commands.send(Command::Run(false)).unwrap();
        view_until(&w, |v| !v.running);
        w.commands.send(Command::Scene(true)).unwrap();
        view_until(&w, |v| !v.positions.is_empty());
        std::thread::sleep(Duration::from_millis(300));
        assert!(
            w.take_view().is_none(),
            "paused sessions must not rebuild unchanged snapshots"
        );
        limited.store(true, Ordering::Relaxed);
        w.commands.send(Command::FastForward(60.0)).unwrap();
        let message = w.events.recv_timeout(Duration::from_secs(10)).unwrap();
        assert!(matches!(message, Event::Paused(_)));
        let paused = view_until(&w, |v| !v.running && v.fast_target.is_none());
        assert_eq!(paused.meters.elapsed_s, initial.meters.elapsed_s);
        let path =
            std::env::temp_dir().join(format!("leo-memory-pause-{}.json.gz", std::process::id()));
        w.commands
            .send(Command::Save(path.display().to_string()))
            .unwrap();
        assert!(matches!(
            w.events.recv_timeout(Duration::from_secs(10)).unwrap(),
            Event::Saved { .. }
        ));
        let restored = Session::load(&path, |_| true).unwrap();
        assert_eq!(restored.elapsed_s(), paused.meters.elapsed_s);
        limited.store(false, Ordering::Relaxed);
        w.commands.send(Command::FastForward(60.0)).unwrap();
        view_until(&w, |v| {
            v.meters.elapsed_s >= 60.0 && v.fast_target.is_none()
        });
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn fast_forward_restores_pause_state_and_checkpoint_loads_at_committed_clock() {
        let w = Worker::spawn(Initial::New(inputs()), 1.0);
        view_until(&w, |_| true);
        w.commands.send(Command::Run(false)).unwrap();
        view_until(&w, |v| !v.running);
        w.fast.store(true, Ordering::Relaxed);
        w.commands.send(Command::FastForward(120.0)).unwrap();
        let v = view_until(&w, |v| {
            v.meters.elapsed_s >= 120.0 && v.fast_target.is_none()
        });
        assert_eq!(v.meters.elapsed_s, 120.0);
        assert!(!v.running);
        assert!(!w.fast.load(Ordering::Relaxed));
        let path = std::env::temp_dir().join(format!("leo-worker-{}.json.gz", std::process::id()));
        w.commands
            .send(Command::Save(path.display().to_string()))
            .unwrap();
        match w.events.recv_timeout(Duration::from_secs(10)).unwrap() {
            Event::Saved { .. } => {}
            Event::Error(e) | Event::Stopped(e) | Event::Paused(e) => panic!("{e}"),
        }
        let loaded = Worker::spawn(Initial::Load(path.display().to_string()), 1.0);
        let restored = view_until(&loaded, |_| true);
        assert!(!restored.running);
        assert_eq!(restored.meters.elapsed_s, 120.0);
        assert_eq!(restored.positions, v.positions);
        loaded.commands.send(Command::Run(true)).unwrap();
        view_until(&loaded, |v| v.running);
        loaded.commands.send(Command::FastForward(120.0)).unwrap();
        let resumed = view_until(&loaded, |v| {
            v.meters.elapsed_s >= 240.0 && v.fast_target.is_none()
        });
        assert!(resumed.running);
        std::fs::remove_file(path).unwrap();
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod orbit_checkpoint_tests {
    use super::*;
    #[test]
    fn sgp4_constants_and_nonzero_run_offset_survive_checkpoint() {
        let satellites =
            crate::tle::parse_tle_data(include_str!("../../assets/tle/starlink.tle")).unwrap();
        let t = &satellites[0];
        let mut inputs = super::tests::inputs();
        inputs.orbit_epoch_unix_s = t.epoch_minutes * 60.0;
        inputs.start_s = 600.0;
        inputs.satellites = vec![leo_capture::session::Satellite {
            name: t.name.clone(),
            constellation: "Starlink".into(),
            orbit: leo_capture::session::Orbit::Sgp4 {
                constants: t.constants.clone(),
                epoch_minutes: t.epoch_minutes,
            },
        }];
        let mut session = Session::new(inputs, |_| true).unwrap();
        session.advance(12).unwrap();
        let path =
            std::env::temp_dir().join(format!("leo-session-tle-{}.json.gz", std::process::id()));
        session.save(&path).unwrap();
        let mut loaded = Session::load(&path, |_| true).unwrap();
        assert_eq!(session.positions().unwrap(), loaded.positions().unwrap());
        session.advance(12).unwrap();
        loaded.advance(12).unwrap();
        assert_eq!(session.positions().unwrap(), loaded.positions().unwrap());
        let expected = t
            .constants
            .propagate(sgp4::MinutesSinceEpoch(12.0))
            .unwrap();
        let actual = loaded.positions().unwrap()[0];
        assert!(
            leo_sim::geometry::norm(leo_sim::geometry::sub(
                actual,
                [
                    expected.position[0],
                    expected.position[2],
                    -expected.position[1]
                ]
            )) < 1e-6
        );
        std::fs::remove_file(path).unwrap();
    }
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;
    #[test]
    fn panic_diagnostics_preserve_the_actual_message_and_simulation_context() {
        let error = panic_message(Box::new("assertion failed: segment order".to_string()));
        let message = stopped_message(
            "advancing capture from 17535 to 17595 simulated seconds",
            &error,
        );
        assert!(message.contains("assertion failed: segment order"));
        assert!(message.contains("17535"));
        assert!(message.contains("not this worker failure"));
    }
}
