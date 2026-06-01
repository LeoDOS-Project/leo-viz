//! Per-constellation cFS instance handle.
//!
//! Owns a [`BridgeServer`] bound to an ephemeral loopback port and a
//! `docker` container running N cFS `core-cpu1` processes. Each process
//! connects back to the server, identifies itself with a [`Hello`] frame,
//! and receives state frames via [`BridgeServer::publish_tick`].
//!
//! Per-satellite stdout is redirected to `/tmp/leodos/sat-<scid>.log`
//! inside the container; a host directory is bind-mounted there so
//! leo-viz reads each file directly off the host filesystem — no
//! `docker exec`, no `tail`.

use crate::bridge_server::BridgeServer;
use crate::bridge_server::ConnectionTracker;
use crate::bridge_server::ReceivedEvent;
use crate::config::ConstellationConfig;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::fs;
use std::io::BufRead;
use std::io::BufReader;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

const IMAGE: &str = "cfs-build:latest";
const CONTAINER_LOG_DIR: &str = "/tmp/leodos";
const SPAWN_FIFO: &str = "/tmp/spawn";

/// Must match `ROUTER_GROUND_MAX_STATIONS` in
/// `apps/router/fsw/tables/router_ground.h`. The router's struct is
/// fixed-size; we always write `ROUTER_GROUND_MAX_STATIONS` entries
/// with the trailing `count` slots zero-padded.
const ROUTER_GROUND_MAX_STATIONS: usize = 4;
const ROUTER_GROUND_BIN_NAME: &str = "router_ground.bin";

/// Process-global registry of currently-running cFS container ids.
/// Populated by `Cfs::launch`, drained by `Drop`. The Ctrl-C handler
/// installed by `install_signal_handler` walks this on SIGINT and
/// runs `docker kill` for each so containers don't leak when leo-viz
/// is killed abruptly (graceful exit goes through `Drop`).
static RUNNING_CONTAINERS: std::sync::OnceLock<Mutex<Vec<String>>> = std::sync::OnceLock::new();

fn registry() -> &'static Mutex<Vec<String>> {
    RUNNING_CONTAINERS.get_or_init(|| Mutex::new(Vec::new()))
}

fn registry_add(id: &str) {
    if let Ok(mut g) = registry().lock() {
        g.push(id.to_string());
    }
}

fn registry_remove(id: &str) {
    if let Ok(mut g) = registry().lock() {
        g.retain(|x| x != id);
    }
}

/// Install a one-shot SIGINT handler that kills all registered cFS
/// containers and exits. Idempotent — safe to call multiple times,
/// only the first install wins.
pub fn install_signal_handler() {
    let _ = ctrlc::try_set_handler(|| {
        let ids: Vec<String> = registry().lock().map(|g| g.clone()).unwrap_or_default();
        for id in ids {
            log::info!("ctrl-c: killing container {}", id);
            let _ = Command::new("docker").args(["kill", &id]).output();
        }
        std::process::exit(130);
    });
}

/// Snapshot of a single ground station taken at `Cfs::launch` time.
/// Frozen for the lifetime of the cFS instance — both the router's
/// `router_ground.bin` and leo-viz's bridge LOS computation read from
/// the same `Vec<GroundStationSnapshot>`.
#[derive(Debug, Clone, Copy)]
pub struct GroundStationSnapshot {
    pub station_id: u8,
    pub lat_deg: f64,
    pub lon_deg: f64,
}
const LAUNCH_CONCURRENCY: usize = 10;
const PER_SAT_TIMEOUT: Duration = Duration::from_secs(20);
const LAUNCH_POLL: Duration = Duration::from_millis(50);

/// PID 1 inside the container: open a fifo for both reading and
/// writing (so the loop never sees EOF), then for each scid received
/// fork-and-exec a `core-cpu1` child. Forking from PID 1 ensures each
/// cFS process inherits PID 1's ulimits — `docker exec` does NOT
/// inherit `--ulimit` from `docker run`, so spawning via this fifo
/// is the only way to give every cFS the raised `nofile` cap.
const PID1_SCRIPT: &str = r#"
set -e
mkfifo /tmp/spawn
exec 3<>/tmp/spawn
while read -u 3 scid; do
    /cFS/build/leodos/exe/cpu1/core-cpu1 --scid "$scid" > /tmp/leodos/sat-${scid}.log 2>&1 &
done
"#;

const MAX_COMBINED_LINES: usize = 5000;
const MAX_PER_SAT_LINES: usize = 2000;
const TAIL_POLL: Duration = Duration::from_millis(100);

#[derive(Default)]
pub struct LogBuffer {
    pub combined: VecDeque<(u32, String)>,
    pub per_sat: Vec<VecDeque<String>>,
    pub apps: BTreeSet<String>,
}

impl LogBuffer {
    fn new(num_sats: usize) -> Self {
        Self {
            combined: VecDeque::new(),
            per_sat: (0..num_sats).map(|_| VecDeque::new()).collect(),
            apps: BTreeSet::new(),
        }
    }

    fn push(&mut self, scid: u32, line: String) {
        if let Some(app) = parse_line(&line).app {
            if !self.apps.contains(app) {
                self.apps.insert(app.to_string());
            }
        }
        if let Some(p) = self.per_sat.get_mut(scid as usize) {
            p.push_back(line.clone());
            while p.len() > MAX_PER_SAT_LINES {
                p.pop_front();
            }
        }
        self.combined.push_back((scid, line));
        while self.combined.len() > MAX_COMBINED_LINES {
            self.combined.pop_front();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KindFilter {
    All,
    Event,
    Log,
}

impl KindFilter {
    fn label(self) -> &'static str {
        match self {
            KindFilter::All => "All",
            KindFilter::Event => "Event (EVS)",
            KindFilter::Log => "Log (syslog)",
        }
    }
}

#[derive(Debug, Clone)]
pub enum CfsStatus {
    Launching,
    Running {
        #[allow(dead_code)]
        container_id: String,
    },
    Failed(String),
}

pub struct Cfs {
    server: BridgeServer,
    status: Arc<Mutex<CfsStatus>>,
    container_id: Arc<Mutex<Option<String>>>,
    host_log_dir: PathBuf,
    /// Frozen copy of the planet's ground stations as of launch.
    /// Same data the router reads from `router_ground.bin`.
    pub launched_stations: Vec<GroundStationSnapshot>,
    stop: Arc<AtomicBool>,
    pub logs: Arc<Mutex<LogBuffer>>,
    pub show_logs: bool,
    pub selected_tab: Option<u32>,
    pub app_filter: Option<String>,
    pub kind_filter: KindFilter,
    pub events_received: u64,
    pub last_event: Option<(String, u16, u32)>,
    pub last_event_per_sat: HashMap<u32, u64>,
    pub show_send: bool,
    pub ping_target: u32,
    pub ping_rto_ms: u32,
    pub ping_timeout_ms: u32,
    pub ping_output: Arc<Mutex<String>>,
    pub ping_pending: Arc<std::sync::atomic::AtomicBool>,
    /// Monotonic counter used as `PingRequestFrame.request_id`.
    /// Echoed back in the result EventFrame so the UI can correlate.
    pub ping_next_request_id: u32,
    /// Ground station the Send button targets. Today we only run one
    /// daemon per cFS; when multiple are launched the user picks here.
    pub ping_send_station_id: u8,
    _worker: Option<JoinHandle<()>>,
}

impl Cfs {
    pub fn launch(
        num_sats: usize,
        sats_per_plane: usize,
        altitude_km: f64,
        inclination_deg: f64,
        phasing: f64,
        ground_stations: Vec<GroundStationSnapshot>,
    ) -> std::io::Result<Self> {
        let server = BridgeServer::bind()?;
        let port = server.local_addr().port();
        let bridge_addr = format!("host.docker.internal:{}", port);

        let host_log_dir = std::env::temp_dir().join(format!("leodos-cfs-{}", port));
        let _ = fs::remove_dir_all(&host_log_dir);
        fs::create_dir_all(&host_log_dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&host_log_dir, fs::Permissions::from_mode(0o777))?;
        }

        let num_planes = if sats_per_plane == 0 {
            0
        } else {
            num_sats.div_ceil(sats_per_plane)
        };
        let table_path = host_log_dir.join(ROUTER_GROUND_BIN_NAME);
        let table_bytes = encode_ground_table(
            num_planes as u8,
            sats_per_plane as u8,
            (altitude_km * 1000.0) as f32,
            inclination_deg as f32,
            phasing as f32,
            &ground_stations,
        );
        fs::write(&table_path, &table_bytes)?;
        log::info!(
            "wrote router_ground.bin ({}x{}, alt={}km, incl={}°, F={}, {} stations) to {}",
            num_planes,
            sats_per_plane,
            altitude_km,
            inclination_deg,
            phasing,
            ground_stations.len(),
            table_path.display()
        );

        let status = Arc::new(Mutex::new(CfsStatus::Launching));
        let container_id = Arc::new(Mutex::new(None));
        let logs = Arc::new(Mutex::new(LogBuffer::new(num_sats)));
        let stop = Arc::new(AtomicBool::new(false));

        for scid in 0..num_sats as u32 {
            let path = host_log_dir.join(format!("sat-{}.log", scid));
            let logs = Arc::clone(&logs);
            let stop = Arc::clone(&stop);
            thread::Builder::new()
                .name(format!("cfs-tail-{}", scid))
                .spawn(move || tail_file(path, scid, logs, stop))?;
        }

        let worker = {
            let status = Arc::clone(&status);
            let container_id = Arc::clone(&container_id);
            let host_log_dir = host_log_dir.clone();
            let tracker = server.tracker();
            let stop = Arc::clone(&stop);
            let station_ids: Vec<u8> = ground_stations.iter().map(|s| s.station_id).collect();
            thread::Builder::new()
                .name("cfs-launch".into())
                .spawn(move || {
                    run_docker(
                        num_sats,
                        sats_per_plane,
                        bridge_addr,
                        host_log_dir,
                        station_ids,
                        status,
                        container_id,
                        tracker,
                        stop,
                    )
                })?
        };

        Ok(Self {
            server,
            status,
            container_id,
            host_log_dir,
            launched_stations: ground_stations,
            stop,
            logs,
            show_logs: false,
            selected_tab: None,
            app_filter: None,
            kind_filter: KindFilter::All,
            events_received: 0,
            last_event: None,
            last_event_per_sat: HashMap::new(),
            show_send: false,
            ping_target: 0,
            ping_rto_ms: 1000,
            ping_timeout_ms: 5000,
            ping_output: Arc::new(Mutex::new(String::new())),
            ping_pending: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            ping_next_request_id: 1,
            ping_send_station_id: 0,
            _worker: Some(worker),
        })
    }

    pub fn status(&self) -> CfsStatus {
        self.status
            .lock()
            .map(|s| s.clone())
            .unwrap_or(CfsStatus::Failed("status mutex poisoned".into()))
    }

    pub fn connected_scids(&self) -> Vec<u32> {
        self.server.connected_scids()
    }

    pub fn server_mut(&mut self) -> &mut BridgeServer {
        &mut self.server
    }

    /// Drain events the bridge has received from sim_clients since
    /// the last call. Updates the running counter and returns the
    /// drained events to the caller (renderer).
    pub fn drain_events(&mut self) -> Vec<ReceivedEvent> {
        let evs = self.server.drain_events();
        self.events_received = self.events_received.wrapping_add(evs.len() as u64);
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        for ev in &evs {
            self.last_event_per_sat.insert(ev.scid, now_ms);
            if ev.app_name == "GROUND" {
                let now = chrono::Local::now().format("%H:%M:%S");
                let line = format!("[{}] {}\n", now, ev.message);
                append_ping_line(&self.ping_output, line);
            }
        }
        if let Some(last) = evs.last() {
            self.last_event = Some((last.app_name.clone(), last.event_id, last.scid));
        }
        evs
    }
}

impl Drop for Cfs {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Ok(g) = self.container_id.lock() {
            if let Some(id) = g.as_ref() {
                let _ = Command::new("docker").args(["kill", id]).output();
                registry_remove(id);
            }
        }
        let _ = fs::remove_dir_all(&self.host_log_dir);
    }
}

pub fn render_cfs_log_window(ctx: &eframe::egui::Context, cons: &mut ConstellationConfig) {
    use eframe::egui;
    use egui_extras::Column;
    use egui_extras::TableBuilder;
    let Some(cfs_arc) = cons.cfs.as_ref() else {
        return;
    };
    let (mut open, logs_arc, mut selected, mut app_filter, mut kind_filter) = match cfs_arc.lock() {
        Ok(g) => (
            g.show_logs,
            Arc::clone(&g.logs),
            g.selected_tab,
            g.app_filter.clone(),
            g.kind_filter,
        ),
        Err(_) => return,
    };
    if !open {
        return;
    }
    let (num_sats, apps): (usize, Vec<String>) = logs_arc
        .lock()
        .map(|b| (b.per_sat.len(), b.apps.iter().cloned().collect()))
        .unwrap_or_default();
    let title = format!(
        "cFS logs — {}",
        cons.label.as_deref().unwrap_or("constellation"),
    );
    let id = egui::Id::new(("cfs_logs", cons.color_offset));
    let dim = egui::Color32::from_gray(140);
    let mono = egui::FontId::monospace(12.0);
    egui::Window::new(title)
        .id(id)
        .open(&mut open)
        .default_size([820.0, 480.0])
        .resizable(true)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("spacecraft id:");
                let current = match selected {
                    None => egui::RichText::new("All").monospace(),
                    Some(i) => egui::RichText::new(format!("{}", i))
                        .color(line_color(i))
                        .monospace(),
                };
                egui::ComboBox::from_id_salt(("cfs_log_sat", cons.color_offset))
                    .selected_text(current)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut selected,
                            None,
                            egui::RichText::new("All").monospace(),
                        );
                        for i in 0..num_sats as u32 {
                            let label = egui::RichText::new(format!("{}", i))
                                .color(line_color(i))
                                .monospace();
                            ui.selectable_value(&mut selected, Some(i), label);
                        }
                    });
                ui.add_space(12.0);
                ui.label("kind:");
                egui::ComboBox::from_id_salt(("cfs_log_kind", cons.color_offset))
                    .selected_text(egui::RichText::new(kind_filter.label()).monospace())
                    .show_ui(ui, |ui| {
                        for k in [KindFilter::All, KindFilter::Event, KindFilter::Log] {
                            ui.selectable_value(
                                &mut kind_filter,
                                k,
                                egui::RichText::new(k.label()).monospace(),
                            );
                        }
                    });
                ui.add_space(12.0);
                ui.label("event source:");
                let current_app = match &app_filter {
                    None => egui::RichText::new("All").monospace(),
                    Some(a) => egui::RichText::new(a.clone())
                        .color(app_color(a))
                        .monospace(),
                };
                egui::ComboBox::from_id_salt(("cfs_log_app", cons.color_offset))
                    .selected_text(current_app)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut app_filter,
                            None,
                            egui::RichText::new("All").monospace(),
                        );
                        for a in &apps {
                            let label = egui::RichText::new(a).color(app_color(a)).monospace();
                            ui.selectable_value(&mut app_filter, Some(a.clone()), label);
                        }
                    });
            });
            ui.separator();

            let rows: Vec<LogRow> = {
                let Ok(buf) = logs_arc.lock() else { return };
                let iter: Box<dyn Iterator<Item = LogRow>> = match selected {
                    None => Box::new(
                        buf.combined
                            .iter()
                            .map(|(scid, line)| LogRow::from_line(*scid, line)),
                    ),
                    Some(scid) => match buf.per_sat.get(scid as usize) {
                        Some(lines) => {
                            Box::new(lines.iter().map(move |line| LogRow::from_line(scid, line)))
                        }
                        None => Box::new(std::iter::empty()),
                    },
                };
                let kind_pred = move |r: &LogRow| match kind_filter {
                    KindFilter::All => true,
                    KindFilter::Event => r.eid.is_some(),
                    KindFilter::Log => r.eid.is_none(),
                };
                match &app_filter {
                    None => iter.filter(kind_pred).collect(),
                    Some(a) => iter
                        .filter(|r| r.app.as_deref() == Some(a.as_str()))
                        .filter(kind_pred)
                        .collect(),
                }
            };
            let last = rows.len().saturating_sub(1);

            let _ = last;
            let sats_per_plane = cons.sats_per_plane.max(1) as u32;
            let builder = TableBuilder::new(ui)
                .id_salt(("cfs_log_table", cons.color_offset))
                .striped(true)
                .resizable(true)
                .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                .column(Column::exact(80.0))
                .column(Column::exact(48.0))
                .column(Column::exact(80.0))
                .column(Column::exact(108.0))
                .column(Column::exact(110.0))
                .column(Column::exact(70.0))
                .column(Column::remainder())
                .auto_shrink([false; 2])
                .stick_to_bottom(true);
            builder
                .header(18.0, |mut header| {
                    header.col(|ui| {
                        ui.strong("spacecraft id");
                    });
                    header.col(|ui| {
                        ui.strong("orbit");
                    });
                    header.col(|ui| {
                        ui.strong("sat in orbit");
                    });
                    header.col(|ui| {
                        ui.strong("mission time")
                            .on_hover_text("cFS mission time, not wall clock");
                    });
                    header.col(|ui| {
                        ui.strong("event source");
                    });
                    header.col(|ui| {
                        ui.strong("event id");
                    });
                    header.col(|ui| {
                        ui.strong("message");
                    });
                })
                .body(|body| {
                    body.rows(16.0, rows.len(), |mut row| {
                        let r = &rows[row.index()];
                        let orbit = r.scid / sats_per_plane;
                        let sat_in_orbit = r.scid % sats_per_plane;
                        row.col(|ui| {
                            ui.label(
                                egui::RichText::new(r.scid.to_string())
                                    .font(mono.clone())
                                    .color(line_color(r.scid)),
                            );
                        });
                        row.col(|ui| {
                            ui.label(
                                egui::RichText::new(orbit.to_string())
                                    .font(mono.clone())
                                    .color(dim),
                            );
                        });
                        row.col(|ui| {
                            ui.label(
                                egui::RichText::new(sat_in_orbit.to_string())
                                    .font(mono.clone())
                                    .color(dim),
                            );
                        });
                        row.col(|ui| {
                            ui.label(
                                egui::RichText::new(r.time.as_deref().unwrap_or(""))
                                    .font(mono.clone())
                                    .color(dim),
                            );
                        });
                        row.col(|ui| {
                            if let Some(app) = r.app.as_deref() {
                                ui.label(
                                    egui::RichText::new(app)
                                        .font(mono.clone())
                                        .color(app_color(app)),
                                );
                            }
                        });
                        row.col(|ui| {
                            ui.label(
                                egui::RichText::new(r.eid.as_deref().unwrap_or(""))
                                    .font(mono.clone())
                                    .color(dim),
                            );
                        });
                        row.col(|ui| {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(&r.message).font(mono.clone()),
                                )
                                .selectable(true)
                                .truncate(),
                            );
                        });
                    });
                });
        });
    if let Ok(mut g) = cfs_arc.lock() {
        g.show_logs = open;
        g.selected_tab = selected;
        g.app_filter = app_filter;
        g.kind_filter = kind_filter;
    }
}

struct LogRow {
    scid: u32,
    time: Option<String>,
    app: Option<String>,
    eid: Option<String>,
    message: String,
}

impl LogRow {
    fn from_line(scid: u32, line: &str) -> Self {
        let p = parse_line(line);
        Self {
            scid,
            time: p.time.map(|s| s.to_string()),
            app: p.app.map(|s| s.to_string()),
            eid: p.eid.map(|s| s.to_string()),
            message: p.msg.to_string(),
        }
    }
}

/// Window for composing a message to send into the running
/// constellation via `leodos-ground`. Currently only Ping; intended
/// to host more message types over time (raw command, ColonyOS job,
/// etc.) without changing the bridge protocol.
pub fn render_cfs_send_window(ctx: &eframe::egui::Context, cons: &mut ConstellationConfig) {
    use eframe::egui;
    let Some(cfs_arc) = cons.cfs.as_ref() else {
        return;
    };
    let (
        mut open,
        mut target,
        mut rto_ms,
        mut timeout_ms,
        sats_per_plane,
        total_sats,
        output_arc,
        pending_arc,
    ) = match cfs_arc.lock() {
        Ok(g) => (
            g.show_send,
            g.ping_target,
            g.ping_rto_ms,
            g.ping_timeout_ms,
            cons.sats_per_plane.max(1) as u32,
            cons.total_sats() as u32,
            Arc::clone(&g.ping_output),
            Arc::clone(&g.ping_pending),
        ),
        Err(_) => return,
    };
    if !open {
        return;
    }
    let _ = pending_arc;
    let title = format!(
        "send — {}",
        cons.label.as_deref().unwrap_or("constellation"),
    );
    let id = egui::Id::new(("cfs_send", cons.color_offset));
    egui::Window::new(title)
        .id(id)
        .open(&mut open)
        .resizable(true)
        .default_size([520.0, 360.0])
        .show(ctx, |ui| {
            egui::CollapsingHeader::new("ping")
                .default_open(true)
                .show(ui, |ui| {
                    egui::Grid::new("ping_form").num_columns(2).show(ui, |ui| {
                        ui.label("target spacecraft id");
                        ui.add(
                            egui::DragValue::new(&mut target)
                                .range(0..=(total_sats.saturating_sub(1))),
                        );
                        ui.end_row();
                        ui.label("RTO (ms)");
                        ui.add(egui::DragValue::new(&mut rto_ms).range(100..=60_000));
                        ui.end_row();
                        ui.label("timeout (ms)");
                        ui.add(egui::DragValue::new(&mut timeout_ms).range(100..=60_000));
                        ui.end_row();
                    });
                    ui.add_space(4.0);
                    let send = ui.button("Send");
                    if send.clicked() {
                        let orb = (target / sats_per_plane) as u8;
                        let sat = (target % sats_per_plane) as u8;
                        let n = sats_per_plane as u8;
                        if let Ok(mut g) = cfs_arc.lock() {
                            let request_id = g.ping_next_request_id;
                            g.ping_next_request_id = g.ping_next_request_id.wrapping_add(1);
                            let station_id = g.ping_send_station_id;
                            let frame = crate::bridge::PingRequestFrame::new(
                                request_id, orb, sat, n, rto_ms, timeout_ms,
                            );
                            let sent = g.server_mut().send_ping_request(station_id as u32, &frame);
                            let now = chrono::Local::now().format("%H:%M:%S");
                            let line = if sent {
                                format!(
                                    "[{}] req={} sent ping orb={} sat={} via station_id={}\n",
                                    now, request_id, orb, sat, station_id,
                                )
                            } else {
                                format!(
                                    "[{}] req={} send failed: no ground daemon for station_id={}\n",
                                    now, request_id, station_id,
                                )
                            };
                            append_ping_line(&output_arc, line);
                        }
                    }
                });
            ui.separator();
            ui.label("output");
            let body = output_arc.lock().map(|g| g.clone()).unwrap_or_default();
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut body.as_str())
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY)
                            .desired_rows(12),
                    );
                });
        });
    if let Ok(mut g) = cfs_arc.lock() {
        g.show_send = open;
        g.ping_target = target;
        g.ping_rto_ms = rto_ms;
        g.ping_timeout_ms = timeout_ms;
    }
    ctx.request_repaint();
}

fn append_ping_line(output: &Arc<Mutex<String>>, line: String) {
    if let Ok(mut buf) = output.lock() {
        buf.push_str(&line);
        if buf.len() > 200_000 {
            let cut = buf.len() - 200_000;
            if let Some(nl) = buf[cut..].find('\n') {
                buf.drain(..cut + nl + 1);
            }
        }
    }
}

/// Returns the per-scid event-flash intensity (0..1) for the given
/// constellation, fading over `fade_ms` since last event. Caller
/// passes this to the planet renderer to brighten emitting sats.
pub fn flash_intensities(
    cons: &ConstellationConfig,
    fade_ms: u64,
) -> std::collections::HashMap<u32, f32> {
    let mut out = std::collections::HashMap::new();
    let Some(cfs_arc) = cons.cfs.as_ref() else {
        return out;
    };
    let Ok(g) = cfs_arc.lock() else { return out };
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    for (&scid, &t) in g.last_event_per_sat.iter() {
        let age = now_ms.saturating_sub(t);
        if age >= fade_ms {
            continue;
        }
        let i = 1.0 - (age as f32 / fade_ms as f32);
        out.insert(scid, i.clamp(0.0, 1.0));
    }
    out
}

/// Encode the constellation + ground-station snapshot as bytes
/// matching the `RouterGroundTable_t` C struct in
/// `apps/router/fsw/tables/router_ground.h`. Layout (16 + 4 + 52 B):
///   `num_orbs:u8, num_sats:u8, _pad:[u8;2]`,
///   `altitude_m:f32, inclination_deg:f32, phasing:f32`,
///   `count:u8, _pad:[u8;3]`,
///   then 4 entries each
///   `{station_id:u8, _pad:[u8;3], lat_deg:f32 LE, lon_deg:f32 LE}`.
/// Excess stations beyond `ROUTER_GROUND_MAX_STATIONS` are truncated.
fn encode_ground_table(
    num_orbs: u8,
    num_sats: u8,
    altitude_m: f32,
    inclination_deg: f32,
    phasing: f32,
    stations: &[GroundStationSnapshot],
) -> Vec<u8> {
    const ENTRY_SIZE: usize = 1 + 3 + 4 + 4;
    const HEADER_SIZE: usize = 2 + 2 + 4 + 4 + 4 + 1 + 3;
    const TABLE_SIZE: usize = HEADER_SIZE + ROUTER_GROUND_MAX_STATIONS * ENTRY_SIZE;
    let n = stations.len().min(ROUTER_GROUND_MAX_STATIONS);
    let mut out = Vec::with_capacity(TABLE_SIZE);
    out.push(num_orbs);
    out.push(num_sats);
    out.extend_from_slice(&[0u8; 2]);
    out.extend_from_slice(&altitude_m.to_le_bytes());
    out.extend_from_slice(&inclination_deg.to_le_bytes());
    out.extend_from_slice(&phasing.to_le_bytes());
    out.push(n as u8);
    out.extend_from_slice(&[0u8; 3]);
    for i in 0..ROUTER_GROUND_MAX_STATIONS {
        match stations.get(i) {
            Some(s) => {
                out.push(s.station_id);
                out.extend_from_slice(&[0u8; 3]);
                out.extend_from_slice(&(s.lat_deg as f32).to_le_bytes());
                out.extend_from_slice(&(s.lon_deg as f32).to_le_bytes());
            }
            None => out.extend_from_slice(&[0u8; ENTRY_SIZE]),
        }
    }
    debug_assert_eq!(out.len(), TABLE_SIZE);
    out
}

pub fn render_cfs_button(
    ui: &mut eframe::egui::Ui,
    cons: &mut ConstellationConfig,
    ground_stations: &[GroundStationSnapshot],
) {
    use eframe::egui;
    let status = cons
        .cfs
        .as_ref()
        .and_then(|c| c.lock().ok().map(|g| g.status()));
    match status {
        None => {
            let btn = egui::Button::new(egui::RichText::new("▶").color(egui::Color32::WHITE))
                .fill(egui::Color32::from_rgb(60, 140, 60))
                .small();
            let tooltip = "Launch cFS.\n\n\
                Sats, orbits, altitude, inclination, and phasing are passed to the\n\
                flight software and locked while it runs.\n\n\
                Walker geometry not represented on the flight side\n\
                (RAAN₀, Δ, d, Ecc, ω, propagator, walker type, drag)\n\
                will be snapped to defaults to keep leo-viz consistent\n\
                with the cFS view.";
            if ui.add(btn).on_hover_text(tooltip).clicked() {
                // cFS only knows about num_orbs/num_sats/altitude/inclination
                // (and reads phasing implicitly via the LOS frames leo-viz
                // sends). The rest of the Walker geometry isn't represented
                // on the flight side, so snap it to canonical values before
                // launch — keeps leo-viz consistent with what cFS expects.
                cons.raan_offset = 0.0;
                cons.raan_spacing = None;
                cons.sat_spacing_km = None;
                cons.eccentricity = 0.0;
                cons.arg_periapsis = 0.0;
                cons.walker_type = crate::walker::WalkerType::Delta;
                cons.propagator = crate::config::Propagator::Keplerian;
                cons.drag_enabled = false;

                let n = cons.total_sats();
                let spp = cons.sats_per_plane.max(1) as usize;
                let alt = cons.altitude_km;
                let incl = cons.inclination;
                let f = cons.phasing;
                match Cfs::launch(n, spp, alt, incl, f, ground_stations.to_vec()) {
                    Ok(c) => cons.cfs = Some(Arc::new(Mutex::new(c))),
                    Err(e) => log::warn!("cFS launch failed: {}", e),
                }
            }
        }
        Some(CfsStatus::Launching) => {
            ui.add(egui::Spinner::new().size(12.0));
            ui.weak("launching…");
        }
        Some(CfsStatus::Running { .. }) => {
            let connected = cons
                .cfs
                .as_ref()
                .and_then(|c| c.lock().ok().map(|g| g.connected_scids().len()))
                .unwrap_or(0);
            let btn = egui::Button::new(egui::RichText::new("■").color(egui::Color32::WHITE))
                .fill(egui::Color32::from_rgb(160, 100, 60))
                .small();
            if ui.add(btn).on_hover_text("Stop cFS").clicked() {
                cons.cfs = None;
            }
            let logs_btn = egui::Button::new(egui::RichText::new("📜").color(egui::Color32::WHITE))
                .fill(egui::Color32::from_rgb(80, 100, 140))
                .small();
            if ui.add(logs_btn).on_hover_text("Show cFS logs").clicked() {
                if let Some(c) = cons.cfs.as_ref() {
                    if let Ok(mut g) = c.lock() {
                        g.show_logs = !g.show_logs;
                    }
                }
            }
            let send_btn = egui::Button::new(egui::RichText::new("📨").color(egui::Color32::WHITE))
                .fill(egui::Color32::from_rgb(140, 60, 100))
                .small();
            if ui.add(send_btn).on_hover_text("Send a message").clicked() {
                if let Some(c) = cons.cfs.as_ref() {
                    if let Ok(mut g) = c.lock() {
                        g.show_send = !g.show_send;
                    }
                }
            }
            ui.weak(format!("{}/{} connected", connected, cons.total_sats()));
        }
        Some(CfsStatus::Failed(msg)) => {
            let btn = egui::Button::new(egui::RichText::new("↻").color(egui::Color32::WHITE))
                .fill(egui::Color32::from_rgb(160, 60, 60))
                .small();
            if ui
                .add(btn)
                .on_hover_text(format!("Failed: {}", msg))
                .clicked()
            {
                cons.cfs = None;
            }
        }
    }
}

fn line_color(scid: u32) -> eframe::egui::Color32 {
    use eframe::egui::Color32;
    const PALETTE: &[Color32] = &[
        Color32::from_rgb(255, 140, 100),
        Color32::from_rgb(120, 200, 255),
        Color32::from_rgb(180, 230, 130),
        Color32::from_rgb(255, 200, 100),
        Color32::from_rgb(200, 150, 255),
        Color32::from_rgb(255, 130, 200),
        Color32::from_rgb(130, 230, 220),
        Color32::from_rgb(255, 240, 130),
        Color32::from_rgb(170, 200, 230),
    ];
    PALETTE[(scid as usize) % PALETTE.len()]
}

struct ParsedLine<'a> {
    time: Option<&'a str>,
    app: Option<&'a str>,
    eid: Option<&'a str>,
    msg: &'a str,
}

fn parse_line(line: &str) -> ParsedLine<'_> {
    if let Some(p) = parse_evs(line) {
        return p;
    }
    if let Some(p) = parse_syslog(line) {
        return p;
    }
    ParsedLine {
        time: None,
        app: None,
        eid: None,
        msg: line,
    }
}

fn parse_evs(line: &str) -> Option<ParsedLine<'_>> {
    let rest = line.strip_prefix("EVS Port1 ")?;
    let (ts_full, rest) = rest.split_once(' ')?;
    let (path, rest) = rest.split_once(' ')?;
    let (eid_raw, msg) = rest.split_once(' ')?;
    let eid = eid_raw.strip_suffix(':')?;
    let time = trim_timestamp(ts_full)?;
    let app = path.split('/').nth(2)?;
    Some(ParsedLine {
        time: Some(time),
        app: Some(app),
        eid: Some(eid),
        msg,
    })
}

/// Parse cFE syslog lines like
/// `1980-001-09:19:49.91157 CFE_ES_Main: CFE_ES_Main entering CORE_STARTUP state`.
fn parse_syslog(line: &str) -> Option<ParsedLine<'_>> {
    let (ts_full, rest) = line.split_once(' ')?;
    let time = trim_timestamp(ts_full)?;
    let (app, msg) = match rest.split_once(": ") {
        Some((a, m)) if !a.contains(' ') => (Some(a), m),
        _ => (None, rest),
    };
    Some(ParsedLine {
        time: Some(time),
        app,
        eid: None,
        msg,
    })
}

/// Validate a `YYYY-DDD-HH:MM:SS.fffff` cFE timestamp and return the
/// `HH:MM:SS.fff` slice (year-of-day dropped, fractional truncated to ms).
fn trim_timestamp(ts: &str) -> Option<&str> {
    let mut parts = ts.splitn(3, '-');
    let year = parts.next()?;
    let doy = parts.next()?;
    let time_full = parts.next()?;
    if year.len() != 4 || !year.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if doy.len() != 3 || !doy.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(match time_full.split_once('.') {
        Some((hms, frac)) => &time_full[..hms.len() + 1 + frac.len().min(3)],
        None => time_full,
    })
}

fn app_color(app: &str) -> eframe::egui::Color32 {
    use eframe::egui::Color32;
    const PALETTE: &[Color32] = &[
        Color32::from_rgb(150, 200, 230),
        Color32::from_rgb(180, 220, 180),
        Color32::from_rgb(220, 200, 160),
        Color32::from_rgb(200, 180, 230),
        Color32::from_rgb(230, 180, 180),
        Color32::from_rgb(180, 220, 220),
    ];
    let mut h: u32 = 5381;
    for b in app.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u32);
    }
    PALETTE[(h as usize) % PALETTE.len()]
}

fn run_docker(
    num_sats: usize,
    sats_per_plane: usize,
    bridge_addr: String,
    host_log_dir: PathBuf,
    station_ids: Vec<u8>,
    status: Arc<Mutex<CfsStatus>>,
    container_id: Arc<Mutex<Option<String>>>,
    tracker: ConnectionTracker,
    stop: Arc<AtomicBool>,
) {
    let mount = format!("{}:{}", host_log_dir.display(), CONTAINER_LOG_DIR);
    let output = Command::new("docker")
        .args([
            "run",
            "-d",
            "--rm",
            "--sysctl",
            "fs.mqueue.msg_max=1000",
            "--sysctl",
            "fs.mqueue.queues_max=4096",
            "--ulimit",
            "nofile=65536:65536",
            "--ulimit",
            "msgqueue=1073741824:1073741824",
            "--ulimit",
            "rtprio=99:99",
            "--cap-add",
            "SYS_NICE",
            "-v",
            &mount,
            "-e",
            &format!("LEODOS_BRIDGE_ADDR={}", bridge_addr),
            IMAGE,
            "bash",
            "-c",
            PID1_SCRIPT,
        ])
        .output();

    let result = match output {
        Ok(o) if o.status.success() => {
            let id = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if id.is_empty() {
                Err("docker run returned no container id".to_string())
            } else {
                Ok(id)
            }
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr).to_string();
            Err(format!(
                "docker run failed: {}",
                stderr.lines().next().unwrap_or("")
            ))
        }
        Err(e) => Err(format!("docker invocation failed: {}", e)),
    };

    let id = match result {
        Ok(id) => id,
        Err(msg) => {
            if let Ok(mut s) = status.lock() {
                *s = CfsStatus::Failed(msg);
            }
            return;
        }
    };

    if let Ok(mut g) = container_id.lock() {
        *g = Some(id.clone());
    }
    if let Ok(mut s) = status.lock() {
        *s = CfsStatus::Running {
            container_id: id.clone(),
        };
    }
    registry_add(&id);

    run_launcher(id.clone(), num_sats, tracker, stop);
    spawn_ground_daemons(&id, &bridge_addr, sats_per_plane, &station_ids);
}

/// After all sats have connected, fork one ground-daemon per launched
/// station inside the same container. Each daemon connects back to
/// the leo-viz bridge over `host.docker.internal:<port>` and turns
/// PingRequestFrames into actual SRSPP pings.
fn spawn_ground_daemons(
    container_id: &str,
    bridge_addr: &str,
    sats_per_plane: usize,
    station_ids: &[u8],
) {
    for &id in station_ids {
        let cmd = format!(
            "/cFS/target/release/leodos-ground \
             --num-sats {n} bridge --bridge-addr {addr} --station-id {id} \
             > /tmp/leodos/ground-{id}.log 2>&1 &",
            n = sats_per_plane,
            addr = bridge_addr,
            id = id,
        );
        let _ = Command::new("docker")
            .args(["exec", "-d", container_id, "sh", "-c", &cmd])
            .output();
        log::info!("ground daemon spawned: station_id={}", id);
    }
}

fn run_launcher(
    container_id: String,
    num_sats: usize,
    tracker: ConnectionTracker,
    stop: Arc<AtomicBool>,
) {
    let mut next: u32 = 0;
    let mut in_flight: HashMap<u32, Instant> = HashMap::new();
    let mut done: HashSet<u32> = HashSet::new();

    while !stop.load(Ordering::Relaxed) && done.len() < num_sats {
        while in_flight.len() < LAUNCH_CONCURRENCY && (next as usize) < num_sats {
            spawn_sat(&container_id, next);
            in_flight.insert(next, Instant::now());
            next += 1;
        }
        thread::sleep(LAUNCH_POLL);
        let connected: HashSet<u32> = tracker.connected_scids().into_iter().collect();
        in_flight.retain(|scid, started| {
            if connected.contains(scid) {
                done.insert(*scid);
                false
            } else if started.elapsed() > PER_SAT_TIMEOUT {
                log::warn!("cfs launcher: scid={} timed out, advancing anyway", scid);
                done.insert(*scid);
                false
            } else {
                true
            }
        });
    }
    log::info!("cfs launcher: {}/{} sats up", done.len(), num_sats);
}

fn spawn_sat(container_id: &str, scid: u32) {
    let _ = Command::new("docker")
        .args([
            "exec",
            container_id,
            "sh",
            "-c",
            &format!("echo {} > {}", scid, SPAWN_FIFO),
        ])
        .output();
}

fn tail_file(path: PathBuf, scid: u32, logs: Arc<Mutex<LogBuffer>>, stop: Arc<AtomicBool>) {
    while !path.exists() {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        thread::sleep(TAIL_POLL);
    }
    let Ok(file) = std::fs::File::open(&path) else {
        return;
    };
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    while !stop.load(Ordering::Relaxed) {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => thread::sleep(TAIL_POLL),
            Ok(_) => {
                while line.ends_with('\n') || line.ends_with('\r') {
                    line.pop();
                }
                if !line.is_empty() {
                    if let Ok(mut buf) = logs.lock() {
                        buf.push(scid, std::mem::take(&mut line));
                    }
                }
            }
            Err(_) => break,
        }
    }
}
