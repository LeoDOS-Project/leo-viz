//! Per-constellation cFS instance handle.
//!
//! Owns a [`BridgeServer`] bound to an ephemeral loopback port and
//! a `docker` container running N cFS `core-cpu1` processes. Each
//! process connects back to the server, identifies itself with a
//! [`Hello`] frame, and receives state frames via
//! [`BridgeServer::publish_tick`].

use crate::bridge_server::BridgeServer;
use crate::config::ConstellationConfig;
use std::io::BufRead;
use std::io::BufReader;
use std::process::Command;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::thread::JoinHandle;

const IMAGE: &str = "cfs-build:latest";
const SAT_LAUNCH_CMD: &str = r#"
for i in $(seq 0 $((LEODOS_NUM_SATS - 1))); do
    /cFS/build/leodos/exe/cpu1/core-cpu1 --scid "$i" >/tmp/sat-$i.log 2>&1 &
done
wait
"#;

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
    logs: Arc<Mutex<String>>,
    show_logs: bool,
    _worker: Option<JoinHandle<()>>,
}

impl Cfs {
    pub fn launch(num_sats: usize) -> std::io::Result<Self> {
        let server = BridgeServer::bind()?;
        let port = server.local_addr().port();
        let bridge_addr = format!("host.docker.internal:{}", port);

        let status = Arc::new(Mutex::new(CfsStatus::Launching));
        let container_id = Arc::new(Mutex::new(None));
        let logs = Arc::new(Mutex::new(String::new()));

        let worker = {
            let status = Arc::clone(&status);
            let container_id = Arc::clone(&container_id);
            let logs = Arc::clone(&logs);
            thread::Builder::new()
                .name("cfs-launch".into())
                .spawn(move || run_docker(num_sats, bridge_addr, status, container_id, logs))?
        };

        Ok(Self {
            server,
            status,
            container_id,
            logs,
            show_logs: false,
            _worker: Some(worker),
        })
    }

    pub fn logs(&self) -> Arc<Mutex<String>> {
        Arc::clone(&self.logs)
    }

    pub fn show_logs_mut(&mut self) -> &mut bool {
        &mut self.show_logs
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
}

impl Drop for Cfs {
    fn drop(&mut self) {
        if let Ok(g) = self.container_id.lock() {
            if let Some(id) = g.as_ref() {
                let _ = Command::new("docker").args(["kill", id]).output();
            }
        }
    }
}

pub fn render_cfs_log_window(ctx: &eframe::egui::Context, cons: &mut ConstellationConfig) {
    use eframe::egui;
    let Some(cfs) = cons.cfs.as_ref() else { return };
    let (mut open, logs_arc) = match cfs.lock() {
        Ok(g) => (g.show_logs, g.logs()),
        Err(_) => return,
    };
    if !open {
        return;
    }
    let title = format!(
        "cFS logs — {}",
        cons.label.as_deref().unwrap_or("constellation"),
    );
    let id = egui::Id::new(("cfs_logs", cons.color_offset));
    egui::Window::new(title)
        .id(id)
        .open(&mut open)
        .default_size([720.0, 480.0])
        .resizable(true)
        .show(ctx, |ui| {
            let buf = logs_arc.lock().map(|g| g.clone()).unwrap_or_default();
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut buf.as_str())
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY)
                            .desired_rows(28),
                    );
                });
        });
    if let Ok(mut g) = cfs.lock() {
        *g.show_logs_mut() = open;
    }
}

pub fn render_cfs_button(ui: &mut eframe::egui::Ui, cons: &mut ConstellationConfig) {
    use eframe::egui;
    let status = cons.cfs.as_ref().and_then(|c| c.lock().ok().map(|g| g.status()));
    match status {
        None => {
            let btn = egui::Button::new(egui::RichText::new("▶").color(egui::Color32::WHITE))
                .fill(egui::Color32::from_rgb(60, 140, 60))
                .small();
            if ui.add(btn).on_hover_text("Launch cFS").clicked() {
                let n = cons.total_sats();
                match Cfs::launch(n) {
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
                        let s = g.show_logs_mut();
                        *s = !*s;
                    }
                }
            }
            ui.weak(format!("{}/{} connected", connected, cons.total_sats()));
        }
        Some(CfsStatus::Failed(msg)) => {
            let btn = egui::Button::new(egui::RichText::new("↻").color(egui::Color32::WHITE))
                .fill(egui::Color32::from_rgb(160, 60, 60))
                .small();
            if ui.add(btn).on_hover_text(format!("Failed: {}", msg)).clicked() {
                cons.cfs = None;
            }
        }
    }
}

fn run_docker(
    num_sats: usize,
    bridge_addr: String,
    status: Arc<Mutex<CfsStatus>>,
    container_id: Arc<Mutex<Option<String>>>,
    logs: Arc<Mutex<String>>,
) {
    let output = Command::new("docker")
        .args([
            "run",
            "-d",
            "--rm",
            "--sysctl",
            "fs.mqueue.msg_max=1000",
            "-e",
            &format!("LEODOS_NUM_SATS={}", num_sats),
            "-e",
            &format!("LEODOS_BRIDGE_ADDR={}", bridge_addr),
            IMAGE,
            "bash",
            "-c",
            SAT_LAUNCH_CMD,
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

    match result {
        Ok(id) => {
            if let Ok(mut g) = container_id.lock() {
                *g = Some(id.clone());
            }
            if let Ok(mut s) = status.lock() {
                *s = CfsStatus::Running { container_id: id.clone() };
            }
            let id_for_tail = id.clone();
            thread::Builder::new()
                .name("cfs-tail".into())
                .spawn(move || tail_logs(id_for_tail, num_sats, logs))
                .ok();
        }
        Err(msg) => {
            if let Ok(mut s) = status.lock() {
                *s = CfsStatus::Failed(msg);
            }
        }
    }
}

fn tail_logs(container_id: String, num_sats: usize, logs: Arc<Mutex<String>>) {
    let mut args: Vec<String> = vec![
        "exec".into(),
        container_id,
        "sh".into(),
        "-c".into(),
        format!(
            "while ! ls /tmp/sat-0.log >/dev/null 2>&1; do sleep 0.1; done; \
             tail -n 200 -F {}",
            (0..num_sats)
                .map(|i| format!("/tmp/sat-{}.log", i))
                .collect::<Vec<_>>()
                .join(" "),
        ),
    ];
    let Ok(mut child) = Command::new("docker")
        .args(args.drain(..))
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    else {
        return;
    };
    let Some(stdout) = child.stdout.take() else {
        return;
    };
    let reader = BufReader::new(stdout);
    for line in reader.lines() {
        let Ok(line) = line else { break };
        if let Ok(mut buf) = logs.lock() {
            buf.push_str(&line);
            buf.push('\n');
            if buf.len() > 200_000 {
                let cut = buf.len() - 200_000;
                if let Some(nl) = buf[cut..].find('\n') {
                    buf.drain(..cut + nl + 1);
                }
            }
        }
    }
    let _ = child.wait();
}
