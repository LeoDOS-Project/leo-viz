//! Process boundary: generation and first-use data preparation run off the GUI thread.
use super::preview::Preview;
use eframe::egui;
use std::{
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Command, Stdio},
    sync::mpsc,
};

pub(super) enum Event {
    Progress(String),
    Finished(Result<Preview, String>),
}

pub(super) fn start(
    python: String,
    request: serde_json::Value,
    ctx: egui::Context,
) -> mpsc::Receiver<Event> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let result = run_with_progress(&python, &request, Some(tx.clone()));
        let _ = tx.send(Event::Finished(result));
        ctx.request_repaint();
    });
    rx
}

pub(super) fn load_preset(ctx: egui::Context, legacy: bool) -> mpsc::Receiver<Event> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(Event::Finished(super::presets::load(legacy)));
        ctx.request_repaint();
    });
    rx
}

#[cfg(test)]
pub(super) fn run(python: &str, request: &serde_json::Value) -> Result<Preview, String> {
    run_with_progress(python, request, None)
}

fn run_with_progress(
    python: &str,
    request: &serde_json::Value,
    progress: Option<mpsc::Sender<Event>>,
) -> Result<Preview, String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/leo-aoi");
    let mut child = Command::new(python).args(["-m", "leo_aoi.viewer"])
        .current_dir(root).env("PYTHONDONTWRITEBYTECODE", "1")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().map_err(|e| format!("Could not start Python: {e}. Check Data and runtime; setup is in tools/leo-aoi/README.md."))?;
    let input = serde_json::to_vec(request).map_err(|e| e.to_string())?;
    if let Err(error) = child.stdin.take().unwrap().write_all(&input) {
        let _ = child.kill();
        let _ = child.wait();
        return Err(format!("Could not send AOI settings to Python: {error}"));
    }
    let stderr = child.stderr.take().unwrap();
    let reader = std::thread::spawn(move || {
        let mut tail = std::collections::VecDeque::new();
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if let Some(sender) = &progress {
                let _ = sender.send(Event::Progress(line.clone()));
            }
            tail.push_back(line);
            if tail.len() > 20 {
                tail.pop_front();
            }
        }
        tail.into_iter().collect::<Vec<_>>().join("\n")
    });
    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    let detail = reader
        .join()
        .unwrap_or_else(|_| "Could not read generator diagnostics".into());
    if !output.status.success() {
        return Err(format!(
            "{}\nCheck the Python environment and the selected source’s dataset cache.",
            detail.trim()
        ));
    }
    Preview::from_json(&output.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_runtime_returns_actionable_error() {
        let error = run("/nonexistent/leo-aoi-python", &serde_json::json!({}))
            .err()
            .unwrap();
        assert!(error.contains("Could not start Python"));
    }
}
