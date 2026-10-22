//! Entry point and module declarations for the LEO satellite constellation visualizer.

mod aep8;
mod aoi;
mod app;
#[cfg(not(target_arch = "wasm32"))]
mod bridge;
#[cfg(not(target_arch = "wasm32"))]
mod bridge_server;
mod celestial;
#[cfg(not(target_arch = "wasm32"))]
mod cfs;
mod config;
mod conjunction;
mod demo;
mod drawing;
mod earth_observation;
mod geo;
mod globe_overlay;
mod igrf;
mod jobs;
mod kessler;
mod math;
mod orbit_presets;
mod pass;
mod physics;
mod projection;
mod pushbroom;
#[cfg(not(target_arch = "wasm32"))]
mod pushbroom_image;
mod radiation;
mod renderer;
mod revisit;
mod settings;
mod simulation_ui;
mod slides;
mod solar_system;
mod spacecomp;
mod texture;
#[cfg(not(target_arch = "wasm32"))]
mod tile;
#[cfg(not(target_arch = "wasm32"))]
mod tile_diagnostics;
mod time;
mod tle;
mod viewer;
mod walker;

pub(crate) use app::App;
#[cfg(not(target_arch = "wasm32"))]
use eframe::egui;
#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::JsCast;
pub(crate) use viewer::ViewerState;

pub const EARTH_VISUAL_SCALE: f64 = 0.95;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    crate::cfs::install_signal_handler();
    let present_mode = match std::env::var("LEO_VIZ_PRESENT_MODE").as_deref() {
        Ok("auto-no-vsync") => egui_wgpu::wgpu::PresentMode::AutoNoVsync,
        Ok("immediate") => egui_wgpu::wgpu::PresentMode::Immediate,
        Ok("mailbox") => egui_wgpu::wgpu::PresentMode::Mailbox,
        Ok("fifo") => egui_wgpu::wgpu::PresentMode::Fifo,
        Ok("fifo-relaxed") => egui_wgpu::wgpu::PresentMode::FifoRelaxed,
        _ => egui_wgpu::wgpu::PresentMode::AutoVsync,
    };
    let frame_latency = std::env::var("LEO_VIZ_FRAME_LATENCY")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .or(Some(2));
    let options = eframe::NativeOptions {
        persist_window: !crate::tile_diagnostics::enabled(),
        persistence_path: crate::tile_diagnostics::enabled()
            .then(|| std::path::PathBuf::from("target/esri-diagnostics-eframe.ron")),
        viewport: egui::ViewportBuilder::default().with_inner_size([1600.0, 1000.0]),
        vsync: matches!(
            present_mode,
            egui_wgpu::wgpu::PresentMode::AutoVsync
                | egui_wgpu::wgpu::PresentMode::Fifo
                | egui_wgpu::wgpu::PresentMode::FifoRelaxed
        ),
        wgpu_options: egui_wgpu::WgpuConfiguration {
            present_mode,
            desired_maximum_frame_latency: frame_latency,
            ..Default::default()
        },
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
