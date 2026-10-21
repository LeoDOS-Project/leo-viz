//! Background native work and cooperative browser generation; no external runtime.
use super::{
    data,
    generate::{Job, Settings},
    preview::Preview,
};
use eframe::egui;
use std::sync::mpsc;

pub(super) enum Event {
    Progress(String),
    Finished(Result<Preview, String>),
}

pub(super) fn start(
    settings: Settings,
    input: data::Input,
    ctx: egui::Context,
) -> mpsc::Receiver<Event> {
    let (tx, rx) = mpsc::channel();
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::spawn(move || {
        let result = (|| {
            settings.validate()?;
            let dataset = if settings.uniform() {
                None
            } else {
                Some(data::load(&input, matches!(settings, Settings::Gdp(_)))?)
            };
            let total = settings.count();
            let mut job = Job::new(settings, dataset)?;
            let mut last = std::time::Instant::now();
            loop {
                if let Some(preview) = job.step()? {
                    return Ok(preview);
                }
                if last.elapsed().as_millis() >= 100 {
                    let _ = tx.send(Event::Progress(format!(
                        "Generating AOIs: {} / {total}",
                        job.completed
                    )));
                    ctx.request_repaint();
                    last = std::time::Instant::now();
                }
            }
        })();
        let _ = tx.send(Event::Finished(result));
        ctx.request_repaint();
    });
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_futures::spawn_local(async move {
        crate::tle::yield_now().await;
        let result = async {
            settings.validate()?;
            let dataset = if settings.uniform() {
                None
            } else {
                Some(data::load(&input, matches!(settings, Settings::Gdp(_))).await?)
            };
            let total = settings.count();
            let mut job = Job::new(settings, dataset)?;
            loop {
                if let Some(preview) = job.step()? {
                    return Ok(preview);
                }
                let _ = tx.send(Event::Progress(format!(
                    "Generating AOIs: {} / {total}",
                    job.completed
                )));
                ctx.request_repaint();
                crate::tle::yield_now().await;
            }
        }
        .await;
        let _ = tx.send(Event::Finished(result));
        ctx.request_repaint();
    });
    rx
}

pub(super) fn load_default_jobs(ctx: egui::Context) -> mpsc::Receiver<Event> {
    let (tx, rx) = mpsc::channel();
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::spawn(move || {
        let _ = tx.send(Event::Finished(super::presets::load_default_jobs()));
        ctx.request_repaint();
    });
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_futures::spawn_local(async move {
        crate::tle::yield_now().await;
        let _ = tx.send(Event::Finished(super::presets::load_default_jobs()));
        ctx.request_repaint();
    });
    rx
}
