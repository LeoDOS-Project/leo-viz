//! Assignment never runs on the desktop UI thread. Browser work yields in batches.
use super::model::{Generator, Job};
use crate::aoi::preview::Preview;
use eframe::egui;
use std::sync::{mpsc, Arc};

pub(super) enum Event {
    Progress(usize),
    Finished(Result<Vec<Job>, String>),
}

pub(super) fn start(
    generator: Generator,
    layers: Vec<Arc<Preview>>,
    ctx: egui::Context,
) -> mpsc::Receiver<Event> {
    let (tx, rx) = mpsc::channel();
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::spawn(move || {
        leo_capture::resources::background_thread();
        let mut last = std::time::Instant::now();
        let result = generator.generate_with_progress(
            layers.iter().flat_map(|p| p.aois.iter().cloned()),
            |done, total| {
                if done == 0 || done == total || last.elapsed().as_millis() >= 50 {
                    let _ = tx.send(Event::Progress(done));
                    ctx.request_repaint();
                    last = std::time::Instant::now();
                }
            },
        );
        let _ = tx.send(Event::Finished(result));
        ctx.request_repaint();
    });
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_futures::spawn_local(async move {
        crate::tle::yield_now().await;
        let result = async {
            let generation =
                generator.prepare(layers.iter().flat_map(|p| p.aois.iter().cloned()))?;
            let mut jobs = Vec::with_capacity(layers.iter().map(|p| p.count).sum());
            let mut last = web_time::Instant::now();
            for job in generation {
                jobs.push(job?);
                if last.elapsed().as_millis() >= 4 {
                    let _ = tx.send(Event::Progress(jobs.len()));
                    ctx.request_repaint();
                    crate::tle::yield_now().await;
                    last = web_time::Instant::now();
                }
            }
            Ok(jobs)
        }
        .await;
        let _ = tx.send(Event::Finished(result));
        ctx.request_repaint();
    });
    rx
}
