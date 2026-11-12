//! Background native work and cooperative browser generation; no external runtime.
use super::{
    data,
    generate::{Job, Settings},
    preview::Preview,
};
use eframe::egui;
use std::sync::{mpsc, Arc};

pub(super) fn compact(
    preview: std::sync::Arc<Preview>,
    ctx: egui::Context,
) -> mpsc::Receiver<Event> {
    let (tx, rx) = mpsc::channel();
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::spawn(move || {
        leo_capture::resources::background_thread();
        let workers = leo_capture::parallel::worker_count(preview.count);
        let mut last = std::time::Instant::now();
        let result = leo_capture::parallel::map(
            (0..preview.count).collect(),
            workers,
            |i| Preview::prepare_record(super::compact::record(preview.record(i))?, i),
            |done, total| {
                if done == 0 || done == total || last.elapsed().as_millis() >= 100 {
                    let _ = tx.send(Event::Progress(format!(
                        "Simplifying AOIs: {done} / {total} · {workers} CPU workers"
                    )));
                    ctx.request_repaint();
                    last = std::time::Instant::now();
                }
            },
        )
        .map(|prepared| {
            let mut result = Preview::empty(preview.count, preview.seed);
            for aoi in prepared {
                result.push_prepared(aoi);
            }
            let _ = tx.send(Event::Progress("Preparing AOI display buffers…".into()));
            ctx.request_repaint();
            result.finish()
        });
        let _ = tx.send(Event::Finished(result));
        ctx.request_repaint();
    });
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_futures::spawn_local(async move {
        let result = async {
            let mut result = Preview::empty(preview.count, preview.seed);
            let mut last = web_time::Instant::now();
            for i in 0..preview.count {
                result.push_record(super::compact::record(preview.record(i))?)?;
                if last.elapsed().as_millis() >= 4 {
                    let _ = tx.send(Event::Progress(format!(
                        "Simplifying AOIs: {} / {}",
                        i + 1,
                        preview.count
                    )));
                    ctx.request_repaint();
                    crate::tle::yield_now().await;
                    last = web_time::Instant::now();
                }
            }
            Ok(result.finish())
        }
        .await;
        let _ = tx.send(Event::Finished(result));
        ctx.request_repaint();
    });
    rx
}

pub(super) enum Event {
    Progress(String),
    Finished(Result<Preview, String>),
    PresetFinished(Result<Vec<Arc<Preview>>, String>),
}

pub(super) fn start(
    settings: Settings,
    input: data::Input,
    ctx: egui::Context,
) -> mpsc::Receiver<Event> {
    let (tx, rx) = mpsc::channel();
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::spawn(move || {
        leo_capture::resources::background_thread();
        let result = (|| {
            settings.validate()?;
            let workers = leo_capture::parallel::worker_count(settings.count());
            let _ = tx.send(Event::Progress(format!(
                "Loading geographic source · {workers} CPU workers…"
            )));
            ctx.request_repaint();
            let dataset = if settings.uniform() {
                None
            } else {
                Some(data::load(&input, settings.source_kind())?)
            };
            let mut last = std::time::Instant::now();
            let mut previous_stage = "";
            Job::new(settings, dataset)?.run_parallel(workers, |stage, done, total| {
                if stage != previous_stage || done == total || last.elapsed().as_millis() >= 100 {
                    let _ = tx.send(Event::Progress(format!(
                        "{stage}: {done} / {total} · {workers} CPU workers"
                    )));
                    ctx.request_repaint();
                    last = std::time::Instant::now();
                    previous_stage = stage;
                }
            })
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
                Some(data::load(&input, settings.source_kind()).await?)
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

pub(super) fn load_preset(ctx: egui::Context, legacy: bool) -> mpsc::Receiver<Event> {
    let (tx, rx) = mpsc::channel();
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::spawn(move || {
        leo_capture::resources::background_thread();
        let result = super::presets::load_native(legacy, |message| {
            let _ = tx.send(Event::Progress(message));
            ctx.request_repaint();
        });
        let _ = tx.send(Event::PresetFinished(result));
        ctx.request_repaint();
    });
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_futures::spawn_local(async move {
        crate::tle::yield_now().await;
        let result = async {
            let bundled = super::presets::layers(legacy);
            let mut loaded = Vec::with_capacity(bundled.len());
            for (index, layer) in bundled.iter().enumerate() {
                let label = format!("Layer {}/{} · {}", index + 1, bundled.len(), layer.name);
                if let Some(cached) = layer.cached() {
                    loaded.push(cached);
                    continue;
                }
                let document = layer.document(|stage| {
                    let _ = tx.send(Event::Progress(format!("{label} · {stage}")));
                    ctx.request_repaint();
                })?;
                document.validate()?;
                crate::tle::yield_now().await;
                let total = document.config.count;
                let mut preview = Preview::empty(total, document.config.seed);
                let mut last = web_time::Instant::now();
                for aoi in document.aois {
                    preview.push_record(aoi)?;
                    if last.elapsed().as_millis() >= 4 {
                        let _ = tx.send(Event::Progress(format!(
                            "{label} · Preparing saved geometry: {} / {total}",
                            preview.aois.len()
                        )));
                        ctx.request_repaint();
                        crate::tle::yield_now().await;
                        last = web_time::Instant::now();
                    }
                }
                let _ = tx.send(Event::Progress(format!(
                    "{label} · Building display buffers…"
                )));
                ctx.request_repaint();
                crate::tle::yield_now().await;
                loaded.push(layer.remember(preview.finish()));
            }
            Ok::<_, String>(loaded)
        }
        .await;
        let _ = tx.send(Event::PresetFinished(result));
        ctx.request_repaint();
    });
    rx
}
