//! The worker owns all mutable capture state. UI snapshots are bounded and may
//! be skipped; capture ticks and checkpoint commands are never skipped.
use leo_capture::{
    baseline::Report,
    geometry::{Polygons, Region},
    session::{Inputs, Meters, Session},
};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc, Arc,
};
use std::time::{Duration, Instant};

pub(super) enum Command {
    Run(bool),
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
    Saved(String),
}
pub(super) struct Scene {
    pub regions: Vec<Region>,
    pub anchors: Vec<[f64; 2]>,
    pub satellites: Vec<String>,
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
}
pub(super) struct Worker {
    pub commands: mpsc::Sender<Command>,
    pub events: mpsc::Receiver<Event>,
    pub views: mpsc::Receiver<View>,
    pub cancel: Arc<AtomicBool>,
    pub progress: Arc<AtomicUsize>,
    pub fast: Arc<AtomicBool>,
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}
#[cfg(not(target_arch = "wasm32"))]
impl Worker {
    pub fn spawn(initial: Initial, speed: f64) -> Self {
        let (commands, rx) = mpsc::channel();
        let (etx, events) = mpsc::channel();
        let (vtx, views) = mpsc::sync_channel(1);
        let cancel = Arc::new(AtomicBool::new(false));
        let progress = Arc::new(AtomicUsize::new(0));
        let stop = cancel.clone();
        let p = progress.clone();
        let fast = Arc::new(AtomicBool::new(false));
        let worker_fast = fast.clone();
        std::thread::spawn(move || {
            let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||->Result<(),String> {
                let mut running=matches!(&initial,Initial::New(_));
                let progress=|f:f32| {p.store((f*1000.0) as usize,Ordering::Relaxed);!stop.load(Ordering::Relaxed)};
                let mut session=match initial {Initial::New(i)=>Session::new(i,progress)?,Initial::Load(path)=>Session::load(std::path::Path::new(&path),progress)?};
                let scene=Arc::new(Scene {regions:session.inputs().regions.clone(),anchors:session.inputs().jobs.iter().map(|j|[j.aoi.longitude,j.aoi.latitude]).collect(),
                    satellites:session.inputs().satellites.iter().map(|s|format!("{} / {}",s.constellation,s.name)).collect()});
                let mut speed=speed;
                let mut fast_target:Option<f64>=None;
                let mut resume_running=false;
                let mut last_wall=Instant::now();let mut debt=0.0;let mut last_view=Instant::now()-Duration::from_secs(1);
                loop {
                    if stop.load(Ordering::Relaxed) {return Ok(());}
                    let mut dirty=false;
                    while let Ok(command)=rx.try_recv() {
                        dirty=true;
                        match command {
                            Command::Run(value)=>{running=value;fast_target=None;debt=0.0;worker_fast.store(false,Ordering::Relaxed);},
                            Command::Speed(value)=>speed=value.clamp(1.0,86400.0),
                            Command::FastForward(seconds)=> {
                                if seconds.is_finite() && seconds>0.0 {
                                    resume_running=running;running=false;debt=0.0;
                                    worker_fast.store(true,Ordering::Relaxed);
                                    fast_target=Some((session.elapsed_s()+seconds).min(session.inputs().options.duration_s));
                                }
                            }
                            Command::Save(path)=>match session.save(std::path::Path::new(&path)) {
                                Ok(())=>{let _=etx.send(Event::Saved(path));},Err(e)=>{let _=etx.send(Event::Error(e));}
                            },
                        }
                        // Saving is an explicit simulation pause, not elapsed simulated time.
                        last_wall=Instant::now();
                    }
                    let now=Instant::now();let wall=now.duration_since(last_wall).as_secs_f64();last_wall=now;
                    if !session.finished() {
                        let steps=if let Some(target)=fast_target {
                            ((target-session.elapsed_s())/session.inputs().options.step_s).ceil().max(0.0) as u64
                        } else if running {
                            debt+=wall*speed;
                            (debt/session.inputs().options.step_s).floor() as u64
                        } else {0};
                        // Bounded work gives pause/save/cancel a chance between chunks.
                        let steps=steps.min(12);
                        if steps>0 {session.advance(steps)?;if fast_target.is_none(){debt-=steps as f64*session.inputs().options.step_s;}}
                    }
                    if fast_target.is_some_and(|target|session.elapsed_s()>=target) {
                        fast_target=None;worker_fast.store(false,Ordering::Relaxed);running=resume_running;debt=0.0;last_wall=Instant::now();dirty=true;
                    }
                    if session.finished() {running=false;fast_target=None;worker_fast.store(false,Ordering::Relaxed);}
                    if dirty || last_view.elapsed()>=Duration::from_millis(250) {
                        let mut report=session.report().clone();
                        // UI shows first 100; full reports and all jobs remain in checkpoints.
                        report.jobs.truncate(100);
                        let footprints=if fast_target.is_none() {
                            session.state().captures.iter().rev().take_while(|c|c.end_s>session.elapsed_s()-600.0).take(2000).map(|c|session.footprint(c)).collect()
                        } else {Vec::new()};
                        let view=View {scene:scene.clone(),report,meters:session.meters(),positions:if fast_target.is_none(){session.ground_positions()?}else{Vec::new()},footprints,
                            running,fast_target,finished:session.finished()};
                        match vtx.try_send(view) {Ok(())=>last_view=Instant::now(),Err(mpsc::TrySendError::Full(_))=>{},Err(mpsc::TrySendError::Disconnected(_))=>return Ok(())}
                    }
                    if fast_target.is_none(){std::thread::sleep(Duration::from_millis(20));}
                }
            })).unwrap_or_else(|_|Err("Capture worker stopped after an unexpected geometry error. Reload the last checkpoint.".into()));
            worker_fast.store(false, Ordering::Relaxed);
            if let Err(e) = result {
                let _ = etx.send(Event::Error(e));
            }
        });
        Self {
            commands,
            events,
            views,
            cancel,
            progress,
            fast,
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
                if let Event::Error(e) = event {
                    panic!("{e}");
                }
            }
            let view = w
                .views
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .expect("capture worker timeout");
            if predicate(&view) {
                return view;
            }
        }
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
            Event::Saved(_) => {}
            Event::Error(e) => panic!("{e}"),
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
