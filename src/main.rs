//! Entry point and module declarations for the LEO satellite constellation visualizer.

mod celestial;
mod config;
mod demo;
mod drawing;
mod geo;
mod math;
mod renderer;
mod settings;
mod texture;
mod tile;
mod time;
mod tle;
mod viewer;
mod app;
mod walker;
mod pass;
mod solar_system;

pub(crate) use viewer::ViewerState;
pub(crate) use app::App;
use eframe::egui;
#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::JsCast;

pub const EARTH_VISUAL_SCALE: f64 = 0.95;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1600.0, 1000.0]),
        ..Default::default()
    };

    eframe::run_native(
        "LEO Viz",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("No window")
            .document()
            .expect("No document");

        let canvas = document
            .get_element_by_id("canvas")
            .expect("No canvas element")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("Not a canvas");

        let web_options = eframe::WebOptions::default();
        eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(App::new(cc)))),
            )
            .await
            .expect("Failed to start eframe");
    });
}
