//! Core viewer state and per-tab UI rendering.
//!
//! Owns the ViewerState struct (tabs, textures, camera state) and renders
//! each tab's planet views, constellation controls, TLE selection, and
//! satellite camera windows.

use crate::celestial::{CelestialBody, Skin, TextureResolution};
use crate::config::{
    AreaOfInterest, DeviceLayer, GroundStation, Preset, TabConfig, View3DFlags,
};
use crate::drawing::{
    draw_3d_view, draw_ground_track, draw_torus, plane_color,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::geo::{GeoLoadState, GeoOverlayData};
use crate::renderer::SphereRenderer;
use crate::texture::{TextureLoadState, EarthTexture, RingTexture};
use crate::tile::TileOverlayState;
use crate::time::{body_rotation_angle, DAYS_PER_YEAR, SOLAR_DECLINATION_MAX};
use crate::tle::{TlePreset, TleSatellite, TleShell, TleLoadState, mean_motion_to_altitude_km, SECONDS_PER_DAY};
#[cfg(not(target_arch = "wasm32"))]
use crate::tle::fetch_tle_data;
use crate::walker::{WalkerType, WalkerConstellation, SatelliteState};
use crate::texture::asset_path;
#[cfg(target_arch = "wasm32")]
use crate::texture::{
    fetch_texture,
    TEXTURE_RESULT, STAR_TEXTURE_RESULT, MILKY_WAY_TEXTURE_RESULT,
    NIGHT_TEXTURE_RESULT, CLOUD_TEXTURE_RESULT,
};
#[cfg(target_arch = "wasm32")]
use crate::tle::{
    fetch_tle_text, parse_tle_data_async, TLE_FETCH_RESULT,
};
use eframe::{egui, glow};
use egui::mutex::Mutex;
use egui_dock::{TabViewer, NodeIndex, SurfaceIndex};
use egui_dock::tab_viewer::OnCloseResponse;
use nalgebra::{Matrix3, Vector3};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, mpsc};
use chrono::{DateTime, Utc};

pub(crate) struct ViewerState {
    pub(crate) tabs: Vec<TabConfig>,
    pub(crate) camera_id_counter: usize,
    pub(crate) tab_counter: usize,
    pub(crate) torus_zoom: f64,
    pub(crate) torus_rotation: Matrix3<f64>,
    pub(crate) planet_textures: HashMap<(CelestialBody, Skin, TextureResolution), Arc<EarthTexture>>,
    pub(crate) ring_textures: HashMap<CelestialBody, Arc<RingTexture>>,
    pub(crate) cloud_textures: HashMap<TextureResolution, Arc<EarthTexture>>,
    pub(crate) planet_image_handles: HashMap<(CelestialBody, Skin, TextureResolution), egui::TextureHandle>,
    pub(crate) texture_resolution: TextureResolution,
    pub(crate) last_rotation: Option<Matrix3<f64>>,
    pub(crate) earth_resolution: usize,
    pub(crate) last_resolution: usize,
    pub(crate) texture_load_state: TextureLoadState,
    pub(crate) pending_body: Option<(CelestialBody, Skin, TextureResolution)>,
    pub(crate) dark_mode: bool,
    pub(crate) show_info: bool,
    pub(crate) real_time: f64,
    pub(crate) start_timestamp: DateTime<Utc>,
    pub(crate) show_side_panel: bool,
    pub(crate) pending_add_tab: Option<usize>,
    pub(crate) current_gmst: f64,
    pub(crate) auto_cycle_tabs: bool,
    pub(crate) auto_hide_tab_bar: bool,
    pub(crate) ui_visible: bool,
    pub(crate) cycle_interval: f64,
    pub(crate) last_cycle_time: f64,
    pub(crate) use_gpu_rendering: bool,
    pub(crate) show_borders: bool,
    pub(crate) show_cities: bool,
    pub(crate) active_tab_idx: usize,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) geo_data: GeoLoadState,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) geo_fetch_rx: Option<mpsc::Receiver<Result<GeoOverlayData, String>>>,
    pub(crate) dragging_place: Option<(usize, usize, bool, usize)>,
    pub(crate) night_texture: Option<Arc<EarthTexture>>,
    pub(crate) star_texture: Option<Arc<EarthTexture>>,
    pub(crate) milky_way_texture: Option<Arc<EarthTexture>>,
    #[allow(dead_code)]
    pub(crate) night_texture_loading: bool,
    #[allow(dead_code)]
    pub(crate) star_texture_loading: bool,
    #[allow(dead_code)]
    pub(crate) milky_way_texture_loading: bool,
    #[allow(dead_code)]
    pub(crate) cloud_texture_loading: bool,
    pub(crate) sphere_renderer: Option<Arc<Mutex<SphereRenderer>>>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) tle_fetch_tx: mpsc::Sender<(TlePreset, Result<Vec<TleSatellite>, String>)>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) tle_fetch_rx: mpsc::Receiver<(TlePreset, Result<Vec<TleSatellite>, String>)>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) tile_overlay: TileOverlayState,
    pub(crate) view_width: f32,
    pub(crate) view_height: f32,
    pub(crate) solar_system_handles: HashMap<CelestialBody, egui::TextureHandle>,
    pub(crate) ss_last_render_instant: Option<std::time::Instant>,
    pub(crate) show_planet_sizes: bool,
    pub(crate) planet_sizes_t: f64,
    pub(crate) planet_sizes_auto_zoom: bool,
    pub(crate) planet_sizes_zoom_duration: f32,
    pub(crate) planet_sizes_stay_duration: f32,
    pub(crate) planet_sizes_auto_time: f64,
    pub(crate) ss_auto_zoom: bool,
    pub(crate) ss_auto_zoom_duration: f32,
    pub(crate) ss_auto_zoom_stay: f32,
    pub(crate) ss_auto_zoom_time: f64,
    pub(crate) asteroid_sprite: Option<egui::TextureHandle>,
    pub(crate) asteroid_state: crate::solar_system::AsteroidLoadState,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) asteroid_rx: Option<mpsc::Receiver<Result<Vec<crate::solar_system::Asteroid>, String>>>,
    pub(crate) hohmann: crate::solar_system::HohmannState,
    #[cfg(target_arch = "wasm32")]
    pub(crate) last_url_hash: String,
}


impl TabViewer for ViewerState {
    type Tab = usize;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        self.tabs.get(*tab).map(|t| t.name.as_str()).unwrap_or("?").into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        if *tab < self.tabs.len() {
            self.active_tab_idx = *tab;
            self.render_tab_ui(ui, *tab);
        }
    }

    fn scroll_bars(&self, _tab: &Self::Tab) -> [bool; 2] {
        [false, false]
    }

    fn closeable(&mut self, _tab: &mut Self::Tab) -> bool {
        self.tabs.len() > 1
    }

    fn on_close(&mut self, tab: &mut Self::Tab) -> OnCloseResponse {
        if self.tabs.len() > 1 && *tab < self.tabs.len() {
            OnCloseResponse::Close
        } else {
            OnCloseResponse::Ignore
        }
    }

    fn on_add(&mut self, _surface: SurfaceIndex, _node: NodeIndex) {
        self.tab_counter += 1;
        let mut tab = TabConfig::new(format!("View {}", self.tab_counter));
        if let Some(last_tab) = self.tabs.last() {
            tab.planets = last_tab.planets.clone();
            tab.planet_counter = last_tab.planet_counter;
        }
        let new_idx = self.tabs.len();
        self.tabs.push(tab);
        self.pending_add_tab = Some(new_idx);
    }

    fn context_menu(&mut self, _ui: &mut egui::Ui, _tab: &mut Self::Tab, _surface: SurfaceIndex, _node: NodeIndex) {
    }
}


impl ViewerState {
    #[cfg(not(target_arch = "wasm32"))]
    fn tile_overlay_detail_gl_info(&self, body: CelestialBody) -> Option<(glow::Texture, [f32; 4])> {
        if !self.tile_overlay.enabled || body != CelestialBody::Earth {
            return None;
        }
        let dt = self.tile_overlay.detail_texture.as_ref()?;
        let gl_tex = dt.gl_texture?;
        Some((gl_tex, [
            dt.bounds.min_lon as f32,
            dt.bounds.max_lon as f32,
            dt.bounds.min_lat as f32,
            dt.bounds.max_lat as f32,
        ]))
    }

    #[cfg(target_arch = "wasm32")]
    fn tile_overlay_detail_gl_info(&self, _body: CelestialBody) -> Option<(glow::Texture, [f32; 4])> {
        None
    }

    fn render_tab_ui(&mut self, ui: &mut egui::Ui, tab_idx: usize) {
        for planet in &mut self.tabs[tab_idx].planets {
            for camera in std::mem::take(&mut planet.pending_cameras) {
                planet.satellite_cameras.push(camera);
            }
            planet.satellite_cameras.retain(|c| !planet.cameras_to_remove.contains(&c.id));
            planet.cameras_to_remove.clear();
        }

        let num_planets = self.tabs[tab_idx].planets.len();
        let available_rect = ui.available_rect_before_wrap();
        let gap = 4.0;
        let total_gap = gap * (num_planets.saturating_sub(1)) as f32;
        let planet_width = (available_rect.width() - total_gap) / num_planets as f32;

        let mut add_planet = false;
        let mut planet_to_remove: Option<usize> = None;

        for planet_idx in 0..num_planets {
            let x_offset = planet_idx as f32 * (planet_width + gap);
            let planet_rect = egui::Rect::from_min_size(
                egui::pos2(available_rect.min.x + x_offset, available_rect.min.y),
                egui::vec2(planet_width, available_rect.height()),
            );

            ui.scope_builder(egui::UiBuilder::new().max_rect(planet_rect), |ui| {
                let (should_add, should_remove) = self.render_planet_ui(ui, tab_idx, planet_idx, num_planets);
                if should_add { add_planet = true; }
                if should_remove { planet_to_remove = Some(planet_idx); }
            });

            if planet_idx < num_planets - 1 {
                let sep_x = available_rect.min.x + x_offset + planet_width + gap * 0.5;
                ui.painter().line_segment(
                    [
                        egui::pos2(sep_x, available_rect.min.y),
                        egui::pos2(sep_x, available_rect.max.y),
                    ],
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 80, 80)),
                );
            }
        }

        if let Some(idx) = planet_to_remove {
            self.tabs[tab_idx].planets.remove(idx);
        }
        if add_planet {
            self.tabs[tab_idx].add_planet();
        }
    }

    fn render_planet_ui(&mut self, ui: &mut egui::Ui, tab_idx: usize, planet_idx: usize, num_planets: usize) -> (bool, bool) {
        let mut add_planet = false;
        let mut remove_planet = false;

        let planet_name = self.tabs[tab_idx].planets[planet_idx].name.clone();
        let current_body = self.tabs[tab_idx].planets[planet_idx].celestial_body;
        let current_skin = self.tabs[tab_idx].planets[planet_idx].skin;
        let mut new_body = current_body;
        let mut new_skin = current_skin;
        let mut reset_skin = false;

        let show_stats = self.tabs[tab_idx].show_stats;
        let show_tle = self.tabs[tab_idx].planets[planet_idx].show_tle_window;
        let show_places = self.tabs[tab_idx].planets[planet_idx].show_gs_aoi_window;
        let show_config = self.tabs[tab_idx].planets[planet_idx].show_config_window;

        if self.ui_visible {
        ui.horizontal(|ui| {
            ui.strong(&planet_name);
            if ui.small_button("+").clicked() {
                add_planet = true;
            }
            if num_planets > 1 {
                let btn = egui::Button::new(
                    egui::RichText::new("×").color(egui::Color32::WHITE)
                ).fill(egui::Color32::from_rgb(180, 60, 60)).small();
                if ui.add(btn).clicked() {
                    remove_planet = true;
                }
            }

            ui.separator();

            egui::ComboBox::from_id_salt(format!("body_{}_{}", tab_idx, planet_idx))
                .selected_text(current_body.label())
                .show_ui(ui, |ui| {
                    let mut last_cat = "";
                    let mut last_parent = None::<CelestialBody>;
                    for body in CelestialBody::ALL {
                        let cat = body.category();
                        if cat != last_cat {
                            if !last_cat.is_empty() {
                                ui.separator();
                            }
                            ui.label(egui::RichText::new(cat).small().weak());
                            last_cat = cat;
                            last_parent = None;
                        }
                        if cat == "Moons" {
                            let parent = body.parent_body();
                            if parent != last_parent {
                                if last_parent.is_some() {
                                    ui.separator();
                                }
                                if let Some(p) = parent {
                                    ui.label(egui::RichText::new(
                                        format!("  {}", p.label())
                                    ).small().weak());
                                }
                                last_parent = parent;
                            }
                        }
                        if ui.selectable_value(&mut new_body, body, body.label()).changed() {
                            reset_skin = true;
                        }
                    }
                });
            if ui.small_button("▶").clicked() {
                let current_idx = CelestialBody::ALL.iter().position(|&b| b == current_body).unwrap_or(0);
                let next_idx = (current_idx + 1) % CelestialBody::ALL.len();
                new_body = CelestialBody::ALL[next_idx];
                reset_skin = true;
            }

            let available_skins = new_body.available_skins();
            if available_skins.len() > 1 {
                egui::ComboBox::from_id_salt(format!("skin_{}_{}", tab_idx, planet_idx))
                    .selected_text(new_skin.label())
                    .show_ui(ui, |ui| {
                        for skin in available_skins {
                            ui.selectable_value(&mut new_skin, *skin, skin.label());
                        }
                    });
            }

            ui.separator();

            if ui.selectable_label(show_stats, "Stats").clicked() {
                self.tabs[tab_idx].show_stats = !show_stats;
            }
            if ui.selectable_label(show_places, "Ground").clicked() {
                self.tabs[tab_idx].planets[planet_idx].show_gs_aoi_window = !show_places;
            }
            if ui.selectable_label(show_config, "Space").clicked() {
                self.tabs[tab_idx].planets[planet_idx].show_config_window = !show_config;
            }
            if ui.add_enabled(show_config, egui::Button::new("Live").selected(show_tle)).clicked() {
                self.tabs[tab_idx].planets[planet_idx].show_tle_window = !show_tle;
            }
        });
        } // ui_visible

        if remove_planet {
            return (add_planet, remove_planet);
        }

        {
            let tab = &mut self.tabs[tab_idx];
            let planet = &mut tab.planets[planet_idx];
            if new_body != planet.celestial_body {
                tab.settings.zoom = 1.0;
            }
            planet.celestial_body = new_body;
            if reset_skin {
                planet.skin = Skin::Default;
            } else {
                planet.skin = new_skin;
            }
        }

        if show_places {
            let planet = &self.tabs[tab_idx].planets[planet_idx];
            let planet_name = planet.name.clone();
            let mut ground_stations = planet.ground_stations.clone();
            let mut areas_of_interest = planet.areas_of_interest.clone();
            let mut device_layers = planet.device_layers.clone();
            let mut gs_changed = false;
            let mut aoi_changed = false;
            let mut dev_changed = false;

            egui::Window::new(format!("Ground - {}", planet_name))
                .open(&mut self.tabs[tab_idx].planets[planet_idx].show_gs_aoi_window)
                .show(ui.ctx(), |ui| {
                    ui.heading("Ground Stations");
                    let mut gs_to_remove = None;
                    for (idx, gs) in ground_stations.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            if ui.add_sized([80.0, 18.0], egui::TextEdit::singleline(&mut gs.name)).changed() {
                                gs_changed = true;
                            }
                            ui.label("Lat:");
                            if ui.add(egui::DragValue::new(&mut gs.lat).range(-90.0..=90.0).speed(0.5).suffix("°")).changed() {
                                gs_changed = true;
                            }
                            ui.label("Lon:");
                            if ui.add(egui::DragValue::new(&mut gs.lon).range(-180.0..=180.0).speed(0.5).suffix("°")).changed() {
                                gs_changed = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Radius:");
                            if ui.add(egui::DragValue::new(&mut gs.radius_km).range(1.0..=5000.0).speed(10.0).suffix(" km")).changed() {
                                gs_changed = true;
                            }
                            if ui.color_edit_button_srgba(&mut gs.color).changed() {
                                gs_changed = true;
                            }
                            if ui.small_button("×").clicked() {
                                gs_to_remove = Some(idx);
                            }
                        });
                    }
                    if let Some(idx) = gs_to_remove {
                        ground_stations.remove(idx);
                        gs_changed = true;
                    }
                    if ui.button("+ Add ground station").clicked() {
                        ground_stations.push(GroundStation {
                            name: format!("GS{}", ground_stations.len() + 1),
                            lat: 0.0,
                            lon: 0.0,
                            radius_km: 500.0,
                            color: egui::Color32::from_rgb(255, 100, 100),
                        });
                        gs_changed = true;
                    }

                    ui.separator();
                    ui.heading("Areas of Interest");
                    let mut aoi_to_remove = None;
                    for (idx, aoi) in areas_of_interest.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            if ui.add_sized([80.0, 18.0], egui::TextEdit::singleline(&mut aoi.name)).changed() {
                                aoi_changed = true;
                            }
                            ui.label("Lat:");
                            if ui.add(egui::DragValue::new(&mut aoi.lat).range(-90.0..=90.0).speed(0.5).suffix("°")).changed() {
                                aoi_changed = true;
                            }
                            ui.label("Lon:");
                            if ui.add(egui::DragValue::new(&mut aoi.lon).range(-180.0..=180.0).speed(0.5).suffix("°")).changed() {
                                aoi_changed = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Radius:");
                            if ui.add(egui::DragValue::new(&mut aoi.radius_km).range(1.0..=5000.0).speed(10.0).suffix(" km")).changed() {
                                aoi_changed = true;
                            }
                            if ui.color_edit_button_srgba(&mut aoi.color).changed() {
                                aoi_changed = true;
                            }
                            if ui.small_button("×").clicked() {
                                aoi_to_remove = Some(idx);
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("GS:");
                            let gs_label = aoi.ground_station_idx
                                .and_then(|i| ground_stations.get(i))
                                .map(|gs| gs.name.as_str())
                                .unwrap_or("None");
                            egui::ComboBox::from_id_salt(format!("aoi_gs_{}", idx))
                                .selected_text(gs_label)
                                .show_ui(ui, |ui| {
                                    if ui.selectable_label(aoi.ground_station_idx.is_none(), "None").clicked() {
                                        aoi.ground_station_idx = None;
                                        aoi_changed = true;
                                    }
                                    for (gs_idx, gs) in ground_stations.iter().enumerate() {
                                        if ui.selectable_label(aoi.ground_station_idx == Some(gs_idx), &gs.name).clicked() {
                                            aoi.ground_station_idx = Some(gs_idx);
                                            aoi_changed = true;
                                        }
                                    }
                                });
                        });
                    }
                    if let Some(idx) = aoi_to_remove {
                        areas_of_interest.remove(idx);
                        aoi_changed = true;
                    }
                    if ui.button("+ Add area of interest").clicked() {
                        areas_of_interest.push(AreaOfInterest {
                            name: format!("AOI{}", areas_of_interest.len() + 1),
                            lat: 0.0,
                            lon: 0.0,
                            radius_km: 500.0,
                            color: egui::Color32::from_rgba_unmultiplied(100, 200, 100, 100),
                            ground_station_idx: None,
                        });
                        aoi_changed = true;
                    }

                    ui.separator();
                    ui.heading("Devices");
                    let mut layer_to_remove = None;
                    for (li, layer) in device_layers.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            if ui.add_sized([80.0, 18.0], egui::TextEdit::singleline(&mut layer.name)).changed() {
                                dev_changed = true;
                            }
                            if ui.color_edit_button_srgba(&mut layer.color).changed() {
                                dev_changed = true;
                            }
                            ui.weak(format!("{} pts", layer.devices.len()));
                            if ui.small_button("×").clicked() {
                                layer_to_remove = Some(li);
                            }
                        });
                        let mut dev_to_remove = None;
                        egui::ScrollArea::vertical()
                            .id_salt(format!("devlayer_{}", li))
                            .max_height(120.0)
                            .show(ui, |ui| {
                                for (di, dev) in layer.devices.iter_mut().enumerate() {
                                    ui.horizontal(|ui| {
                                        ui.add_space(16.0);
                                        ui.label("Lat:");
                                        if ui.add(egui::DragValue::new(&mut dev.0).range(-90.0..=90.0).speed(0.5).suffix("°")).changed() {
                                            dev_changed = true;
                                        }
                                        ui.label("Lon:");
                                        if ui.add(egui::DragValue::new(&mut dev.1).range(-180.0..=180.0).speed(0.5).suffix("°")).changed() {
                                            dev_changed = true;
                                        }
                                        if ui.small_button("x").clicked() {
                                            dev_to_remove = Some(di);
                                        }
                                    });
                                }
                            });
                        if let Some(di) = dev_to_remove {
                            layer.devices.remove(di);
                            dev_changed = true;
                        }
                        if ui.small_button("+ Add device").clicked() {
                            layer.devices.push((0.0, 0.0));
                            dev_changed = true;
                        }
                        ui.separator();
                    }
                    if let Some(li) = layer_to_remove {
                        device_layers.remove(li);
                        dev_changed = true;
                    }
                    if ui.button("+ Add device layer").clicked() {
                        device_layers.push(DeviceLayer {
                            name: format!("Layer {}", device_layers.len() + 1),
                            color: egui::Color32::from_rgb(80, 140, 255),
                            devices: Vec::new(),
                        });
                        dev_changed = true;
                    }
                });

            if gs_changed {
                self.tabs[tab_idx].planets[planet_idx].ground_stations = ground_stations;
            }
            if aoi_changed {
                self.tabs[tab_idx].planets[planet_idx].areas_of_interest = areas_of_interest;
            }
            if dev_changed {
                self.tabs[tab_idx].planets[planet_idx].device_layers = device_layers;
            }
        }

        let show_stats = self.tabs[tab_idx].show_stats;
        if show_stats {
            let planet = &self.tabs[tab_idx].planets[planet_idx];
            let planet_name = planet.name.clone();
            let celestial_body = planet.celestial_body;
            let planet_radius = celestial_body.radius_km();
            let mu = celestial_body.mu();
            let constellations: Vec<_> = planet.constellations.to_vec();
            let tle_selections = planet.tle_selections.clone();

            egui::Window::new(format!("Stats - {}", planet_name))
                .open(&mut self.tabs[tab_idx].show_stats)
                .show(ui.ctx(), |ui| {
                    const SPEED_OF_LIGHT_KM_S: f64 = 299792.0;

                    ui.heading(celestial_body.label());
                    ui.label(format!("  Radius: {:.0} km", planet_radius));
                    ui.label(format!("  μ: {:.0} km³/s²", mu));
                    let surface_gravity = mu / (planet_radius * planet_radius);
                    ui.label(format!("  Surface gravity: {:.2} m/s²", surface_gravity * 1000.0));
                    let escape_velocity = (2.0 * mu * 1e9 / (planet_radius * 1000.0)).sqrt() / 1000.0;
                    ui.label(format!("  Escape velocity: {:.2} km/s", escape_velocity));
                    let geo_orbit = (mu * (SECONDS_PER_DAY / (2.0 * PI)).powi(2)).powf(1.0/3.0);
                    let geo_altitude = geo_orbit - planet_radius;
                    if geo_altitude > 0.0 {
                        ui.label(format!("  Geostationary alt: {:.0} km", geo_altitude));
                    }
                    ui.separator();

                    if !constellations.is_empty() {
                        ui.heading("Walker Constellations");
                        for cons in &constellations {
                            ui.strong(cons.preset_name());
                            ui.label(format!("  Satellites: {}", cons.total_sats()));
                            {
                                let orbit_radius = planet_radius + cons.altitude_km;
                                let orbit_radius_m = orbit_radius * 1000.0;
                                let velocity_ms = (mu * 1e9 / orbit_radius_m).sqrt();
                                let velocity_kmh = velocity_ms * 3.6;

                                let intra_plane_dist = orbit_radius * (2.0 * (1.0 - (2.0 * PI / cons.sats_per_plane as f64).cos())).sqrt();
                                let inc_rad = cons.inclination.to_radians();
                                let base_inter = orbit_radius * (2.0 * (1.0 - (2.0 * PI / cons.num_planes as f64).cos())).sqrt();
                                let inter_plane_dist = base_inter * inc_rad.sin().abs().max(0.1);
                                let ground_dist = cons.altitude_km;

                                let intra_latency_ms = intra_plane_dist / SPEED_OF_LIGHT_KM_S * 1000.0;
                                let inter_latency_ms = inter_plane_dist / SPEED_OF_LIGHT_KM_S * 1000.0;
                                let ground_latency_ms = ground_dist / SPEED_OF_LIGHT_KM_S * 1000.0;

                                ui.label(format!("  Velocity: {:.0} km/h", velocity_kmh));
                                ui.label(format!("  Intra-plane: {:.0} km ({:.2} ms)", intra_plane_dist, intra_latency_ms));
                                ui.label(format!("  Inter-plane: {:.0} km ({:.2} ms)", inter_plane_dist, inter_latency_ms));
                                ui.label(format!("  Ground: {:.0} km ({:.2} ms)", ground_dist, ground_latency_ms));
                            }
                        }
                        ui.separator();
                    }

                    let live_data: Vec<_> = TlePreset::ALL.iter()
                        .filter_map(|preset| {
                            if let Some((selected, state, _)) = tle_selections.get(preset) {
                                if *selected {
                                    if let TleLoadState::Loaded { satellites, .. } = state {
                                        return Some((preset.label(), satellites.len(), preset.is_debris()));
                                    }
                                }
                            }
                            None
                        })
                        .collect();

                    if !live_data.is_empty() {
                        ui.heading("Live Data (TLE)");
                        let mut total = 0;
                        let mut total_debris = 0;
                        for (name, count, is_debris) in &live_data {
                            let kind = if *is_debris { "debris" } else { "satellites" };
                            ui.label(format!("  {}: {} {}", name, count, kind));
                            if *is_debris { total_debris += count; } else { total += count; }
                        }
                        if total > 0 { ui.label(format!("  Satellites: {}", total)); }
                        if total_debris > 0 { ui.label(format!("  Debris: {}", total_debris)); }
                    }
                });
        }

        if show_config {
        ui.separator();

        let mut const_to_remove: Option<usize> = None;
        let mut cameras_to_clean: Vec<usize> = Vec::new();
        let planet = &mut self.tabs[tab_idx].planets[planet_idx];
        let num_constellations = planet.constellations.len();
        let show_tle = planet.show_tle_window;

        #[cfg(not(target_arch = "wasm32"))]
        let tle_fetch_tx = self.tle_fetch_tx.clone();

        let controls_height = 180.0;
        {
        let planet = &mut self.tabs[tab_idx].planets[planet_idx];
        ui.horizontal(|ui| {
            if show_tle {
                let selected_loaded: Vec<TlePreset> = planet.tle_selections.iter()
                    .filter(|(_, (sel, state, _))| *sel && matches!(state, TleLoadState::Loaded { .. }))
                    .map(|(p, _)| *p)
                    .collect();
                let can_split = selected_loaded.len() == 1;
                let split_active = can_split && planet.tle_selections.get(&selected_loaded[0])
                    .map(|(_, _, shells)| shells.is_some()).unwrap_or(false);

                ui.vertical(|ui| {
                    ui.set_min_height(controls_height);
                    let mut fetch_requested = false;
                    let mut split_preset: Option<TlePreset> = None;
                    let unsplit_preset: Option<TlePreset> = None;
                    ui.horizontal(|ui| {
                        ui.label("TLE");
                        if ui.small_button("All").clicked() {
                            for (preset, (selected, _, _)) in planet.tle_selections.iter_mut() {
                                *selected = !matches!(preset, TlePreset::Last30Days | TlePreset::Brightest100 | TlePreset::ActiveSats);
                            }
                        }
                        if ui.small_button("None").clicked() {
                            for (selected, _, _) in planet.tle_selections.values_mut() {
                                *selected = false;
                            }
                        }
                        if ui.small_button("Fetch").clicked() {
                            fetch_requested = true;
                        }
                        if can_split && !split_active && ui.small_button("Cluster").clicked() {
                            split_preset = Some(selected_loaded[0]);
                        }
                        if ui.small_button("x").clicked() {
                            planet.show_tle_window = false;
                        }
                    });

                    egui::ScrollArea::vertical()
                        .id_salt(format!("tle_scroll_{}_{}",tab_idx, planet_idx))
                        .max_height(controls_height)
                        .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        for category in ["Comms", "Navigation", "Observation", "Other", "Debris"] {
                            ui.vertical(|ui| {
                                ui.strong(category);
                                for preset in TlePreset::ALL.iter().filter(|p| p.category() == category) {
                                    if let Some((selected, state, _)) = planet.tle_selections.get_mut(preset) {
                                        let is_clustered_other = split_active && !selected_loaded.contains(preset);
                                        let is_clustered_selected = split_active && selected_loaded.contains(preset);
                                        ui.horizontal(|ui| {
                                            if !split_active {
                                                let color = plane_color(preset.color_index());
                                                let rect = ui.allocate_space(egui::vec2(10.0, 10.0)).1;
                                                ui.painter().rect_filled(rect, 2.0, color);
                                                ui.painter().rect_filled(rect.shrink(2.5), 1.0, egui::Color32::BLACK);
                                            }

                                            let is_loading = matches!(state, TleLoadState::Loading);
                                            if is_clustered_other {
                                                ui.add_enabled(false, egui::Button::new(preset.label()).selected(*selected));
                                            } else if is_clustered_selected {
                                                let _ = ui.selectable_label(true, preset.label());
                                            } else if ui.selectable_label(*selected, preset.label()).clicked() {
                                                *selected = !*selected;
                                            }
                                            if is_loading {
                                                ui.spinner();
                                            }

                                            #[cfg(not(target_arch = "wasm32"))]
                                            if fetch_requested && *selected && matches!(state, TleLoadState::NotLoaded | TleLoadState::Failed(_)) {
                                                *state = TleLoadState::Loading;
                                                let url = preset.url().to_string();
                                                let preset_copy = *preset;
                                                let tx = tle_fetch_tx.clone();
                                                std::thread::spawn(move || {
                                                    let result = fetch_tle_data(&url);
                                                    let _ = tx.send((preset_copy, result));
                                                });
                                            }

                                            #[cfg(target_arch = "wasm32")]
                                            if fetch_requested && *selected && matches!(state, TleLoadState::NotLoaded | TleLoadState::Failed(_)) {
                                                *state = TleLoadState::Loading;
                                                let url = preset.url().to_string();
                                                let preset_copy = *preset;
                                                let ctx = ui.ctx().clone();
                                                wasm_bindgen_futures::spawn_local(async move {
                                                    let result = match fetch_tle_text(&url).await {
                                                        Ok(text) => parse_tle_data_async(&text).await,
                                                        Err(e) => Err(e),
                                                    };
                                                    TLE_FETCH_RESULT.with(|cell| {
                                                        cell.borrow_mut().push((preset_copy, result));
                                                    });
                                                    ctx.request_repaint();
                                                });
                                            }
                                        });
                                    }
                                }
                            });
                        }

                    });
                    });

                    if let Some(preset) = split_preset {
                        if let Some((_, state, shells)) = planet.tle_selections.get_mut(&preset) {
                            if let TleLoadState::Loaded { satellites } = state {
                                let n = satellites.len();
                                let (inc_bin_size, alt_bin_size) = if n < 50 {
                                    (1.0, 10.0)
                                } else if n < 500 {
                                    (5.0, 50.0)
                                } else {
                                    (5.0, 100.0)
                                };
                                let mut groups: std::collections::HashMap<(i32, i32), Vec<usize>> = std::collections::HashMap::new();
                                for (i, sat) in satellites.iter().enumerate() {
                                    let alt = mean_motion_to_altitude_km(sat.mean_motion);
                                    let inc_bin = (sat.inclination_deg / inc_bin_size).round() as i32 * inc_bin_size as i32;
                                    let alt_bin = (alt / alt_bin_size).round() as i32 * alt_bin_size as i32;
                                    groups.entry((inc_bin, alt_bin)).or_default().push(i);
                                }
                                let mut sorted: Vec<_> = groups.into_iter().collect();
                                sorted.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
                                let min_sats = if n < 50 { 1 } else { (n / 50).max(10) };
                                let mut new_shells = Vec::new();
                                let mut other_indices = Vec::new();
                                for ((inc, alt), indices) in sorted {
                                    if indices.len() < min_sats {
                                        other_indices.extend(indices);
                                        continue;
                                    }
                                    let co = planet.constellation_counter;
                                    planet.constellation_counter += 1;
                                    new_shells.push(TleShell {
                                        label: format!("{}°/{}km", inc, alt),
                                        satellite_indices: indices,
                                        color_offset: co,
                                        selected: true,
                                    });
                                }
                                if !other_indices.is_empty() {
                                    let co = planet.constellation_counter;
                                    planet.constellation_counter += 1;
                                    new_shells.push(TleShell {
                                        label: "Other".to_string(),
                                        satellite_indices: other_indices,
                                        color_offset: co,
                                        selected: true,
                                    });
                                }
                                *shells = Some(new_shells);
                            }
                        }
                    }
                    if let Some(preset) = unsplit_preset {
                        if let Some((_, _, shells)) = planet.tle_selections.get_mut(&preset) {
                            *shells = None;
                        }
                    }
                });

                if split_active {
                    ui.separator();
                    let mut close_cluster = false;
                    ui.vertical(|ui| {
                        let preset = selected_loaded[0];
                        if let Some((_, _, Some(shells))) = planet.tle_selections.get_mut(&preset) {
                            ui.horizontal(|ui| {
                                ui.label(preset.label());
                                if ui.small_button("All").clicked() {
                                    for shell in shells.iter_mut() {
                                        shell.selected = true;
                                    }
                                }
                                if ui.small_button("None").clicked() {
                                    for shell in shells.iter_mut() {
                                        shell.selected = false;
                                    }
                                }
                                if ui.small_button("x").clicked() {
                                    close_cluster = true;
                                }
                            });
                            ui.horizontal(|ui| {
                                let num_cols = shells.len().div_ceil(7);
                                for col in 0..num_cols {
                                    ui.vertical(|ui| {
                                        for row in 0..7 {
                                            let idx = col * 7 + row;
                                            if idx < shells.len() {
                                                let shell = &mut shells[idx];
                                                ui.horizontal(|ui| {
                                                    let color = plane_color(shell.color_offset);
                                                    let rect = ui.allocate_space(egui::vec2(10.0, 10.0)).1;
                                                    ui.painter().rect_filled(rect, 2.0, color);
                                                    ui.painter().rect_filled(rect.shrink(2.5), 1.0, egui::Color32::BLACK);
                                                    if ui.selectable_label(shell.selected, &shell.label).clicked() {
                                                        shell.selected = !shell.selected;
                                                    }
                                                });
                                            }
                                        }
                                    });
                                }
                            });
                        }
                    });
                    if close_cluster {
                        let preset = selected_loaded[0];
                        if let Some((_, _, shells)) = planet.tle_selections.get_mut(&preset) {
                            *shells = None;
                        }
                    }
                }
                ui.separator();
            }

            for (cidx, cons) in planet.constellations.iter_mut().enumerate() {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        let color = plane_color(cons.color_offset);
                        let rect = ui.allocate_space(egui::vec2(10.0, 10.0)).1;
                        ui.painter().rect_filled(rect, 2.0, color);
                        ui.label(cons.preset_name());
                        let hide_btn = if cons.hidden {
                            egui::Button::new(egui::RichText::new("+").color(egui::Color32::WHITE))
                                .fill(egui::Color32::from_rgb(60, 140, 60)).small()
                        } else {
                            egui::Button::new(egui::RichText::new("−").color(egui::Color32::WHITE))
                                .fill(egui::Color32::from_rgb(100, 100, 100)).small()
                        };
                        if ui.add(hide_btn).clicked() {
                            cons.hidden = !cons.hidden;
                            if cons.hidden {
                                cameras_to_clean.push(cidx);
                            }
                        }
                        if num_constellations > 0 {
                            let btn = egui::Button::new(
                                egui::RichText::new("x").color(egui::Color32::WHITE)
                            ).fill(egui::Color32::from_rgb(180, 60, 60)).small();
                            if ui.add(btn).clicked() {
                                const_to_remove = Some(cidx);
                            }
                        }
                    });

                    {
                    ui.horizontal(|ui| {
                        let mut sats = cons.sats_per_plane as i32;
                        let mut planes = cons.num_planes as i32;
                        ui.label("Sats:");
                        let sats_resp = ui.add(egui::DragValue::new(&mut sats).range(1..=100));
                        ui.label("Orbits:");
                        let planes_resp = ui.add(egui::DragValue::new(&mut planes).range(1..=100));
                        if sats > 0 && planes > 0 {
                            cons.sats_per_plane = sats as usize;
                            cons.num_planes = planes as usize;
                        }
                        if sats_resp.changed() || planes_resp.changed() {
                            cons.preset = Preset::None;
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Alt:");
                        let alt_resp = ui.add(egui::DragValue::new(&mut cons.altitude_km).range(0.0..=50000.0).suffix(" km"));
                        let orbit_label = if cons.altitude_km < 450.0 { "VLEO" }
                            else if cons.altitude_km < 2000.0 { "LEO" }
                            else if cons.altitude_km < 35000.0 { "MEO" }
                            else { "GEO" };
                        egui::ComboBox::from_id_salt(format!("orbit_{}_{}_{}", tab_idx, planet_idx, cidx))
                            .selected_text(orbit_label)
                            .width(50.0)
                            .show_ui(ui, |ui| {
                                if ui.selectable_label(orbit_label == "VLEO", "VLEO").clicked() {
                                    cons.altitude_km = 350.0;
                                    cons.preset = Preset::None;
                                }
                                if ui.selectable_label(orbit_label == "LEO", "LEO").clicked() {
                                    cons.altitude_km = 1080.0;
                                    cons.preset = Preset::None;
                                }
                                if ui.selectable_label(orbit_label == "MEO", "MEO").clicked() {
                                    cons.altitude_km = 18893.0;
                                    cons.preset = Preset::None;
                                }
                                if ui.selectable_label(orbit_label == "GEO", "GEO").clicked() {
                                    cons.altitude_km = 35786.0;
                                    cons.preset = Preset::None;
                                }
                            });
                        if alt_resp.changed() {
                            cons.preset = Preset::None;
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Inc:");
                        let inc_resp = ui.add(egui::DragValue::new(&mut cons.inclination).range(0.0..=180.0).suffix("°"));
                        if inc_resp.changed() {
                            cons.preset = Preset::None;
                        }
                        ui.label("F:");
                        let max_f = (cons.num_planes - 1).max(1) as f64;
                        let phase_resp = ui.add(egui::DragValue::new(&mut cons.phasing).range(0.0..=max_f).speed(0.1));
                        if phase_resp.changed() {
                            cons.preset = Preset::None;
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("RAAN₀:");
                        if ui.add(egui::DragValue::new(&mut cons.raan_offset).range(-180.0..=180.0).suffix("°").speed(1.0)).changed() {
                            cons.preset = Preset::None;
                        }
                        let default_spacing = match cons.walker_type {
                            WalkerType::Delta => 360.0 / cons.num_planes as f64,
                            WalkerType::Star => 180.0 / cons.num_planes as f64,
                        };
                        let mut custom_spacing = cons.raan_spacing.is_some();
                        if ui.checkbox(&mut custom_spacing, "Δ:").changed() {
                            cons.raan_spacing = if custom_spacing { Some(default_spacing) } else { None };
                            cons.preset = Preset::None;
                        }
                        if let Some(ref mut spacing) = cons.raan_spacing {
                            if ui.add(egui::DragValue::new(spacing).range(0.1..=180.0).suffix("°").speed(0.5)).changed() {
                                cons.preset = Preset::None;
                            }
                        } else {
                            ui.weak(format!("{:.1}°", default_spacing));
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Ecc:");
                        if ui.add(egui::DragValue::new(&mut cons.eccentricity).range(0.0..=0.99).speed(0.001).max_decimals(4)).changed() {
                            cons.preset = Preset::None;
                        }
                        ui.label("ω:");
                        if ui.add(egui::DragValue::new(&mut cons.arg_periapsis).range(0.0..=360.0).suffix("°").speed(1.0)).changed() {
                            cons.preset = Preset::None;
                        }
                    });

                    ui.horizontal(|ui| {
                        let old_type = cons.walker_type;
                        ui.selectable_value(&mut cons.walker_type, WalkerType::Delta, "Delta");
                        ui.selectable_value(&mut cons.walker_type, WalkerType::Star, "Star");
                        if ui.checkbox(&mut cons.drag_enabled, "Drag:").changed() {
                            cons.preset = Preset::None;
                        }
                        if cons.drag_enabled {
                            if ui.add(egui::DragValue::new(&mut cons.ballistic_coeff).range(0.1..=500.0).suffix(" kg/m²").speed(1.0).max_decimals(1)).changed() {
                                cons.preset = Preset::None;
                            }
                        } else {
                            ui.weak("N/A");
                        }
                        if cons.walker_type != old_type {
                            cons.preset = Preset::None;
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Preset:");
                        egui::ComboBox::from_id_salt(format!("preset_{}_{}_{}", tab_idx, planet_idx, cidx))
                            .selected_text(cons.preset_name())
                            .show_ui(ui, |ui| {
                                if ui.selectable_label(cons.preset == Preset::Starlink, "Starlink").clicked() {
                                    cons.sats_per_plane = 22; cons.num_planes = 72;
                                    cons.altitude_km = 550.0; cons.inclination = 53.0;
                                    cons.walker_type = WalkerType::Delta; cons.phasing = 1.0;
                                    cons.preset = Preset::Starlink;
                                }
                                if ui.selectable_label(cons.preset == Preset::OneWeb, "OneWeb").clicked() {
                                    cons.sats_per_plane = 54; cons.num_planes = 12;
                                    cons.altitude_km = 1200.0; cons.inclination = 87.9;
                                    cons.walker_type = WalkerType::Star; cons.phasing = 1.0;
                                    cons.preset = Preset::OneWeb;
                                }
                                if ui.selectable_label(cons.preset == Preset::Iridium, "Iridium").clicked() {
                                    cons.sats_per_plane = 11; cons.num_planes = 6;
                                    cons.altitude_km = 780.0; cons.inclination = 86.4;
                                    cons.walker_type = WalkerType::Star; cons.phasing = 2.0;
                                    cons.preset = Preset::Iridium;
                                }
                                if ui.selectable_label(cons.preset == Preset::Kuiper, "Kuiper").clicked() {
                                    cons.sats_per_plane = 34; cons.num_planes = 34;
                                    cons.altitude_km = 630.0; cons.inclination = 51.9;
                                    cons.walker_type = WalkerType::Delta; cons.phasing = 1.0;
                                    cons.preset = Preset::Kuiper;
                                }
                                if ui.selectable_label(cons.preset == Preset::Iris2, "Iris²").clicked() {
                                    cons.sats_per_plane = 22; cons.num_planes = 12;
                                    cons.altitude_km = 1200.0; cons.inclination = 87.0;
                                    cons.walker_type = WalkerType::Star; cons.phasing = 1.0;
                                    cons.preset = Preset::Iris2;
                                }
                                if ui.selectable_label(cons.preset == Preset::Telesat, "Telesat").clicked() {
                                    cons.sats_per_plane = 13; cons.num_planes = 6;
                                    cons.altitude_km = 1015.0; cons.inclination = 98.98;
                                    cons.walker_type = WalkerType::Star; cons.phasing = 1.0;
                                    cons.preset = Preset::Telesat;
                                }
                            });
                    });
                    }
                });
                ui.separator();
            }

            let add_btn_text = if num_constellations == 0 { "[+] Add constellation" } else { "[+]" };
            if ui.button(add_btn_text).clicked() {
                const_to_remove = Some(usize::MAX);
            }
        });

        if let Some(cidx) = const_to_remove {
            if cidx == usize::MAX {
                self.tabs[tab_idx].planets[planet_idx].add_constellation();
            } else {
                cameras_to_clean.push(cidx);
                self.tabs[tab_idx].planets[planet_idx].constellations.remove(cidx);
                for cam in &mut self.tabs[tab_idx].planets[planet_idx].satellite_cameras {
                    if cam.constellation_idx != usize::MAX && cam.constellation_idx > cidx {
                        cam.constellation_idx -= 1;
                    }
                }
            }
        }
        {
            let p = &mut self.tabs[tab_idx].planets[planet_idx];
            p.satellite_cameras.retain(|c|
                c.constellation_idx == usize::MAX || !cameras_to_clean.contains(&c.constellation_idx)
            );
        }
        }

        ui.separator();
        }

        let planet = &self.tabs[tab_idx].planets[planet_idx];
        let planet_radius = planet.celestial_body.radius_km();
        let planet_mu = planet.celestial_body.mu();
        let planet_j2 = planet.celestial_body.j2();
        let planet_eq_radius = planet.celestial_body.equatorial_radius_km();
        let celestial_body = planet.celestial_body;
        let skin = planet.skin;
        let view_name = planet.name.clone();

        let hide_sats = self.tabs[tab_idx].settings.zoom > 100.0;
        let mut constellations_data: Vec<_> = if hide_sats {
            Vec::new()
        } else {
            planet.constellations.iter()
                .enumerate()
                .filter(|(_, c)| !c.hidden)
                .map(|(orig_idx, c)| {
                    let wc = c.constellation(planet_radius, planet_mu, planet_j2, planet_eq_radius);
                    let pos = wc.satellite_positions(self.tabs[tab_idx].settings.time);
                    let name = c.preset_name().to_string();
                    (wc, pos, c.color_offset, 0u8, orig_idx, name)
                })
                .collect()
        };

        if planet.show_tle_window {
            let propagation_minutes = self.start_timestamp.timestamp() as f64 / 60.0 + self.tabs[tab_idx].settings.time / 60.0;
            for preset in TlePreset::ALL.iter() {
                let Some((selected, state, shells)) = planet.tle_selections.get(preset) else { continue };
                if !*selected { continue; }
                let TleLoadState::Loaded { satellites, .. } = state else { continue };
                let mut all_positions: Vec<SatelliteState> = Vec::new();
                for (idx, sat) in satellites.iter().enumerate() {
                    let minutes_since_epoch = propagation_minutes - sat.epoch_minutes;
                    let prediction = match sat.constants.propagate(sgp4::MinutesSinceEpoch(minutes_since_epoch)) {
                        Ok(p) => p,
                        Err(_) => continue,
                    };
                    let x = prediction.position[0];
                    let y = prediction.position[2];
                    let z = -prediction.position[1];
                    let r = (x * x + y * y + z * z).sqrt();
                    let lat = (y / r).asin().to_degrees();
                    let lon = -z.atan2(x).to_degrees();
                    let ascending = prediction.velocity[2] > 0.0;
                    all_positions.push(SatelliteState {
                        plane: 0,
                        sat_index: idx,
                        x, y, z,
                        lat, lon,
                        ascending,
                        neighbor_idx: None,
                        name: Some(sat.name.clone()),
                        tle_inclination_deg: Some(sat.inclination_deg),
                        tle_mean_motion: Some(sat.mean_motion),
                    });
                }
                if all_positions.is_empty() { continue; }

                if let Some(shells) = shells {
                    let shell_indices: Vec<std::collections::HashSet<usize>> = shells.iter()
                        .map(|s| s.satellite_indices.iter().copied().collect())
                        .collect();
                    for (si, shell) in shells.iter().enumerate() {
                        if !shell.selected { continue; }
                        let positions: Vec<SatelliteState> = all_positions.iter()
                            .filter(|p| shell_indices[si].contains(&p.sat_index))
                            .map(|p| SatelliteState {
                                plane: p.plane, sat_index: p.sat_index,
                                x: p.x, y: p.y, z: p.z,
                                lat: p.lat, lon: p.lon,
                                ascending: p.ascending,
                                neighbor_idx: p.neighbor_idx,
                                name: p.name.clone(),
                                tle_inclination_deg: p.tle_inclination_deg,
                                tle_mean_motion: p.tle_mean_motion,
                            })
                            .collect();
                        if positions.is_empty() { continue; }
                        let tle_wc = WalkerConstellation {
                            walker_type: WalkerType::Delta,
                            total_sats: positions.len(),
                            num_planes: 1,
                            altitude_km: 550.0,
                            inclination_deg: 0.0,
                            phasing: 0.0,
                            raan_offset_deg: 0.0,
                            raan_spacing_deg: None,
                            eccentricity: 0.0,
                            arg_periapsis_deg: 0.0,
                            planet_radius,
                            planet_mu,
                            planet_j2,
                            planet_equatorial_radius: planet_eq_radius,
                        };
                        let label = format!("{} {}", preset.label(), shell.label);
                        let tle_kind = if preset.is_debris() { 2u8 } else { 1u8 };
                        constellations_data.push((tle_wc, positions, shell.color_offset, tle_kind, usize::MAX, label));
                    }
                } else {
                    let tle_wc = WalkerConstellation {
                        walker_type: WalkerType::Delta,
                        total_sats: all_positions.len(),
                        num_planes: 1,
                        altitude_km: 550.0,
                        inclination_deg: 0.0,
                        phasing: 0.0,
                        raan_offset_deg: 0.0,
                        raan_spacing_deg: None,
                        eccentricity: 0.0,
                        arg_periapsis_deg: 0.0,
                        planet_radius,
                        planet_mu,
                        planet_j2,
                        planet_equatorial_radius: planet_eq_radius,
                    };
                    let tle_kind = if preset.is_debris() { 2u8 } else { 1u8 };
                    constellations_data.push((tle_wc, all_positions, preset.color_index(), tle_kind, usize::MAX, preset.label().to_string()));
                }
            }
        }

        let available = ui.available_size();
        let settings = &self.tabs[tab_idx].settings;
        let render_planet = settings.render_planet;
        let show_torus = settings.show_torus;
        let show_ground_track = settings.show_ground_track;
        let show_solar_system = settings.show_solar_system;
        let show_orbits = settings.show_orbits;
        let show_axes = settings.show_axes;
        let show_coverage = settings.show_coverage;
        let coverage_angle = settings.coverage_angle;
        let time = settings.time;
        let rotation = settings.rotation;
        let zoom = settings.zoom;
        let earth_fixed_camera = settings.earth_fixed_camera;
        let body_rot_angle = body_rotation_angle(celestial_body, time, self.current_gmst);
        let cos_a = body_rot_angle.cos();
        let sin_a = body_rot_angle.sin();
        let body_y_rotation = Matrix3::new(
            cos_a, 0.0, sin_a,
            0.0, 1.0, 0.0,
            -sin_a, 0.0, cos_a,
        );
        let satellite_rotation = if earth_fixed_camera {
            rotation * body_y_rotation.transpose()
        } else {
            rotation
        };
        let sat_radius = settings.sat_radius;
        let show_links = settings.show_links;
        let show_intra_links = settings.show_intra_links;
        let hide_behind_earth = render_planet && settings.hide_behind_earth;
        let single_color = settings.single_color || constellations_data.len() > 1;
        let dark_mode = self.dark_mode;
        let show_routing_paths = settings.show_routing_paths;
        let show_manhattan_path = settings.show_manhattan_path;
        let show_shortest_path = settings.show_shortest_path;
        let show_asc_desc_colors = settings.show_asc_desc_colors;
        let show_altitude_lines = settings.show_altitude_lines;
        let tex_res = self.texture_resolution;
        let planet_handle = self.planet_image_handles.get(&(celestial_body, skin, tex_res));
        let torus_rotation = self.torus_rotation;
        let torus_zoom = self.torus_zoom;
        let link_width = settings.link_width;
        let fixed_sizes = settings.fixed_sizes;
        let flattening = celestial_body.flattening();
        let show_polar_circle = settings.show_polar_circle;
        let show_equator = settings.show_equator;
        let show_day_night = settings.show_day_night;
        let show_terminator = settings.show_terminator;
        let show_clouds = settings.show_clouds;
        let show_stars = settings.show_stars;
        let show_devices = settings.show_devices;
        let show_borders = settings.show_borders;
        let show_cities = settings.show_cities;
        let log_power = settings.solar_system_log_power;
        let detail_gl_info = self.tile_overlay_detail_gl_info(celestial_body);

        let show_torus = show_torus && render_planet;
        let show_planet_sizes = self.show_planet_sizes;

        let num_views = [render_planet, show_torus, show_solar_system, show_ground_track, show_planet_sizes]
            .iter().filter(|v| **v).count();

        if num_views > 0 {
            let separator_w = if num_views > 1 { 1.0 } else { 0.0 };
            let total_sep = separator_w * (num_views as f32 - 1.0);
            let view_width = (available.x - total_sep) / num_views as f32;
            let view_height = available.y - 20.0;
            self.view_width = view_width;
            self.view_height = view_height;

            ui.horizontal(|ui| {
                let mut needs_sep = false;
                if render_planet {
                    needs_sep = true;
                    ui.vertical(|ui| {
                        let planet = &mut self.tabs[tab_idx].planets[planet_idx];
                        let view_flags = View3DFlags {
                            show_orbits, show_axes, show_coverage, show_links, show_intra_links,
                            hide_behind_earth, single_color, dark_mode, show_routing_paths,
                            show_manhattan_path, show_shortest_path, show_asc_desc_colors,
                            show_altitude_lines, render_planet, fixed_sizes, show_polar_circle,
                            show_equator, show_terminator, earth_fixed_camera,
                            use_gpu_rendering: self.use_gpu_rendering, show_clouds, show_day_night,
                            show_stars, show_borders, show_cities,
                        };
                        let sun_dir = {
                            use chrono::Datelike;
                            let timestamp = self.start_timestamp + chrono::Duration::seconds(time as i64);
                            let day_of_year = timestamp.ordinal() as f64;
                            let declination: f64 = SOLAR_DECLINATION_MAX * ((360.0 / DAYS_PER_YEAR) * (day_of_year + 10.0)).to_radians().cos();
                            let decl_rad = declination.to_radians();
                            let sun_ra = ((day_of_year - 80.0) * 360.0 / 365.0).to_radians();
                            let sun_inertial = Vector3::new(
                                decl_rad.cos() * sun_ra.cos(),
                                decl_rad.sin(),
                                -decl_rad.cos() * sun_ra.sin(),
                            );
                            let sun_shader = body_y_rotation.transpose() * sun_inertial;
                            [sun_shader.x as f32, sun_shader.y as f32, sun_shader.z as f32]
                        };
                        let device_layers_ref: &[DeviceLayer] = if show_devices { &planet.device_layers } else { &[] };
                        let (rot, new_zoom) = draw_3d_view(
                            ui,
                            &view_name,
                            &constellations_data,
                            view_flags,
                            coverage_angle,
                            rotation,
                            satellite_rotation,
                            view_width,
                            view_height,
                            planet_handle,
                            zoom,
                            sat_radius,
                            link_width,
                            &mut planet.pending_cameras,
                            &mut self.camera_id_counter,
                            &mut planet.satellite_cameras,
                            &mut planet.cameras_to_remove,
                            planet_radius,
                            flattening,
                            self.sphere_renderer.as_ref(),
                            (celestial_body, skin, tex_res),
                            &body_y_rotation,
                            sun_dir,
                            time,
                            &mut planet.ground_stations,
                            &mut planet.areas_of_interest,
                            device_layers_ref,
                            body_rot_angle,
                            &mut self.dragging_place,
                            (tab_idx, planet_idx),
                            detail_gl_info,
                            #[cfg(not(target_arch = "wasm32"))]
                            { match &self.geo_data { GeoLoadState::Loaded(d) => if show_borders { d.borders.as_slice() } else { &[] }, _ => &[] } },
                            #[cfg(target_arch = "wasm32")]
                            &[],
                            #[cfg(not(target_arch = "wasm32"))]
                            { match &self.geo_data { GeoLoadState::Loaded(d) => if show_cities { d.cities.as_slice() } else { &[] }, _ => &[] } },
                            #[cfg(target_arch = "wasm32")]
                            &[],
                        );
                        self.tabs[tab_idx].settings.rotation = rot;
                        self.tabs[tab_idx].settings.zoom = new_zoom;
                    });
                }

                if show_torus {
                    if needs_sep { ui.separator(); }
                    needs_sep = true;
                    ui.vertical(|ui| {
                        let planet = &mut self.tabs[tab_idx].planets[planet_idx];
                        let (trot, tzoom) = draw_torus(
                            ui,
                            &format!("torus_{}", view_name),
                            &constellations_data,
                            time,
                            torus_rotation,
                            view_width,
                            view_height,
                            sat_radius,
                            show_links,
                            single_color,
                            torus_zoom,
                            &mut planet.satellite_cameras,
                            show_routing_paths,
                            show_manhattan_path,
                            show_shortest_path,
                            show_asc_desc_colors,
                            planet_radius,
                            &mut planet.pending_cameras,
                            &mut self.camera_id_counter,
                            &mut planet.cameras_to_remove,
                            link_width,
                            fixed_sizes,
                        );
                        self.torus_rotation = trot;
                        self.torus_zoom = tzoom;
                    });
                }

                if show_solar_system {
                    if needs_sep { ui.separator(); }
                    needs_sep = true;
                    ui.vertical(|ui| {
                        let ss_timestamp = self.start_timestamp + chrono::Duration::seconds(time as i64);
                        let lp = log_power;
                        let plot = egui_plot::Plot::new(format!("solar_{}", view_name))
                            .width(view_width)
                            .height(view_height)
                            .data_aspect(1.0)
                            .show_axes(false)
                            .show_grid(false)
                            .show_background(false)
                            .allow_boxed_zoom(false)
                            .allow_scroll(false)
                            .allow_zoom(false)
                            .show_x(false)
                            .show_y(false)
                            .coordinates_formatter(
                                egui_plot::Corner::RightBottom,
                                egui_plot::CoordinatesFormatter::new(move |pt, _| {
                                    let sr = (pt.x.powi(2) + pt.y.powi(2)).sqrt();
                                    let au = if sr > 1e-6 { sr.powf(1.0 / lp) } else { 0.0 };
                                    format!("{:.2} AU", au)
                                }),
                            );
                        let ss_handles = &self.solar_system_handles;
                        let ss_auto = self.ss_auto_zoom;
                        let ss_dur = self.ss_auto_zoom_duration;
                        let ss_stay = self.ss_auto_zoom_stay;
                        let ss_time = &mut self.ss_auto_zoom_time;
                        let ss_result = plot.show(ui, |plot_ui| {
                            if ss_auto {
                                let dt = plot_ui.ctx().input(|i| i.stable_dt) as f64;
                                *ss_time += dt;

                                let lp = log_power;
                                let sc = |au: f64| -> f64 {
                                    (au + crate::solar_system::SCALE_OFFSET).powf(lp)
                                        - crate::solar_system::SCALE_OFFSET.powf(lp)
                                };
                                let start = (sc(0.1) * 1.5).ln();
                                let end = (sc(460.0) * 1.4).ln();

                                let scroll = ss_dur as f64;
                                let stay = ss_stay as f64;
                                let cycle = 2.0 * (stay + scroll);
                                let t = *ss_time % cycle;
                                let frac = if t < stay {
                                    0.0
                                } else if t < stay + scroll {
                                    (t - stay) / scroll
                                } else if t < 2.0 * stay + scroll {
                                    1.0
                                } else {
                                    1.0 - (t - 2.0 * stay - scroll) / scroll
                                };

                                let half = (start + (end - start) * frac).exp();

                                plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
                                    [-half, -half],
                                    [half, half],
                                ));
                                plot_ui.ctx().request_repaint();
                            } else if plot_ui.response().hovered() {
                                let scroll = plot_ui.ctx().input(|i| i.smooth_scroll_delta.y);
                                if scroll.abs() > 0.0 {
                                    let bounds = plot_ui.plot_bounds();
                                    let factor = (-scroll as f64 * 0.002).exp();
                                    let cx = (bounds.min()[0] + bounds.max()[0]) / 2.0;
                                    let cy = (bounds.min()[1] + bounds.max()[1]) / 2.0;
                                    let hw = (bounds.max()[0] - bounds.min()[0]) / 2.0;
                                    let hh = (bounds.max()[1] - bounds.min()[1]) / 2.0;
                                    let (px, py) = plot_ui.pointer_coordinate()
                                        .map(|p| (p.x, p.y))
                                        .unwrap_or((cx, cy));
                                    let ncx = px + (cx - px) * factor;
                                    let ncy = py + (cy - py) * factor;
                                    let nhw = hw * factor;
                                    let nhh = hh * factor;
                                    plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
                                        [ncx - nhw, ncy - nhh],
                                        [ncx + nhw, ncy + nhh],
                                    ));
                                }
                            }
                            let ast_slice = match &self.asteroid_state {
                                crate::solar_system::AsteroidLoadState::Loaded(v) => v.as_slice(),
                                _ => &[],
                            };
                            let ss_click = crate::solar_system::draw_solar_system_view(
                                plot_ui,
                                celestial_body,
                                ss_timestamp,
                                ss_handles,
                                dark_mode,
                                log_power,
                                ast_slice,
                                self.asteroid_sprite.as_ref(),
                            );
                            if self.tabs[tab_idx].settings.show_hohmann {
                                let ss_j2000 = ss_timestamp.signed_duration_since(*crate::solar_system::J2000_EPOCH_PUB).num_seconds() as f64 / 86400.0;
                                if self.hohmann.launched {
                                    let elapsed = ss_j2000 - self.hohmann.launch_j2000_days;
                                    self.hohmann.mission_elapsed_days = elapsed.max(0.0);
                                    if !self.hohmann.arrived {
                                        let params = crate::solar_system::hohmann_transfer_params(self.hohmann.origin, self.hohmann.dest);
                                        if let Some(p) = params {
                                            if self.hohmann.mission_elapsed_days >= p.transfer_time_days {
                                                self.hohmann.mission_elapsed_days = p.transfer_time_days;
                                                self.hohmann.arrived = true;
                                            }
                                        }
                                    }
                                    if let Some(pos) = crate::solar_system::hohmann_spacecraft_position_au(&self.hohmann, ss_j2000) {
                                        let last = self.hohmann.trail.last();
                                        let dominated = last.map_or(false, |l| {
                                            (l[0] - pos[0]).powi(2) + (l[1] - pos[1]).powi(2) < 1e-8
                                        });
                                        if !dominated {
                                            self.hohmann.trail.push(pos);
                                        }
                                    }
                                }
                                crate::solar_system::draw_hohmann_overlay(
                                    plot_ui,
                                    &self.hohmann,
                                    ss_j2000,
                                    log_power,
                                    dark_mode,
                                );
                            }
                            ss_click
                        });
                        if let Some(new_body) = ss_result.inner {
                            self.tabs[tab_idx].planets[planet_idx].celestial_body = new_body;
                        }
                    });
                }

                if show_ground_track {
                    if needs_sep { ui.separator(); }
                    needs_sep = true;
                    ui.vertical(|ui| {
                        draw_ground_track(
                            ui,
                            &format!("ground_{}", view_name),
                            &constellations_data,
                            view_width,
                            view_height,
                            sat_radius,
                            single_color,
                        );
                    });
                }

                if show_planet_sizes {
                    if needs_sep { ui.separator(); }
                    ui.vertical(|ui| {
                        ui.set_width(view_width);
                        ui.set_height(view_height);
                        let mut az = crate::solar_system::AutoZoomState {
                            enabled: self.planet_sizes_auto_zoom,
                            total_duration: self.planet_sizes_zoom_duration,
                            stay_duration: self.planet_sizes_stay_duration,
                            time: self.planet_sizes_auto_time,
                        };
                        if let Some(body) = crate::solar_system::draw_planet_sizes(
                            ui,
                            &self.solar_system_handles,
                            &mut self.planet_sizes_t,
                            &mut az,
                        ) {
                            self.tabs[tab_idx].planets[planet_idx].celestial_body = body;
                        }
                        self.planet_sizes_auto_time = az.time;
                    });
                }
            });
        }

        (add_planet, remove_planet)
    }

    #[allow(unused_variables)]
    pub(crate) fn load_texture_for_body(&mut self, body: CelestialBody, skin: Skin, ctx: &egui::Context) {
        let res = self.texture_resolution;
        let key = (body, skin, res);
        if self.planet_textures.contains_key(&key) {
            return;
        }

        if skin == Skin::Abstract && body == CelestialBody::Earth {
            let src_key = (CelestialBody::Earth, Skin::Default, TextureResolution::R8192);
            if !self.planet_textures.contains_key(&src_key) {
                if let Some(path) = Skin::Default.filename(CelestialBody::Earth, TextureResolution::R8192) {
                    if let Ok(bytes) = std::fs::read(asset_path(path)) {
                        if let Ok(tex) = EarthTexture::from_bytes(&bytes) {
                            self.planet_textures.insert(src_key, Arc::new(tex));
                        }
                    }
                }
            }
            let texture = if let Some(src) = self.planet_textures.get(&src_key) {
                let ocean = [25u8, 40, 80];
                let land = [60u8, 75, 85];
                let ice = [140u8, 150, 160];
                let pixels: Vec<[u8; 3]> = src.pixels.iter().map(|&[r, g, b]| {
                    let brightness = (r as u16 + g as u16 + b as u16) / 3;
                    let is_ocean = b as u16 > (r as u16 + g as u16) / 2 + 20
                        && brightness < 180;
                    let is_ice = brightness > 200;
                    if is_ice { ice } else if is_ocean { ocean } else { land }
                }).collect();
                Arc::new(EarthTexture { width: src.width, height: src.height, pixels })
            } else {
                Arc::new(EarthTexture {
                    width: 2, height: 1,
                    pixels: vec![[25, 40, 80], [25, 40, 80]],
                })
            };
            self.planet_textures.insert(key, texture.clone());
            self.texture_load_state = TextureLoadState::Loaded(texture);
            self.planet_image_handles.remove(&key);
            return;
        }

        let filename = match skin.filename(body, res) {
            Some(f) => f,
            None => return,
        };
        self.texture_load_state = TextureLoadState::Loading;
        self.pending_body = Some(key);

        #[cfg(not(target_arch = "wasm32"))]
        {
            match std::fs::read(asset_path(filename)) {
                Ok(bytes) => match EarthTexture::from_bytes(&bytes) {
                    Ok(texture) => {
                        let mut factor = res.downscale_factor(body, skin);
                        let max_gpu_size = 16384u32;
                        while texture.width / factor > max_gpu_size || texture.height / factor > max_gpu_size {
                            factor += 1;
                        }
                        let texture = if factor > 1 {
                            texture.downscale(factor)
                        } else {
                            texture
                        };
                        let texture = Arc::new(texture);
                        self.planet_textures.insert(key, texture.clone());
                        self.texture_load_state = TextureLoadState::Loaded(texture);
                        self.planet_image_handles.remove(&key);
                        if let Some((ring_path, _, _)) = body.ring_params() {
                            if !self.ring_textures.contains_key(&body) {
                                if let Ok(ring_bytes) = std::fs::read(asset_path(ring_path)) {
                                    if let Ok(ring_tex) = RingTexture::from_bytes(&ring_bytes) {
                                        self.ring_textures.insert(body, Arc::new(ring_tex));
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => self.texture_load_state = TextureLoadState::Failed(e),
                },
                Err(e) => self.texture_load_state = TextureLoadState::Failed(e.to_string()),
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            let ctx = ctx.clone();
            let filename = filename.to_string();
            wasm_bindgen_futures::spawn_local(async move {
                let result = fetch_texture(&filename).await;
                TEXTURE_RESULT.with(|cell| {
                    cell.borrow_mut().push((key, result));
                });
                ctx.request_repaint();
            });
        }
    }

    pub(crate) fn load_cloud_texture(&mut self, _ctx: &egui::Context) {
        let res = self.texture_resolution;
        if self.cloud_textures.contains_key(&res) {
            return;
        }

        let filename = match res.cloud_filename() {
            Some(f) => f,
            None => return,
        };

        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Ok(bytes) = std::fs::read(asset_path(filename)) {
                if let Ok(texture) = EarthTexture::from_bytes(&bytes) {
                    self.cloud_textures.insert(res, Arc::new(texture));
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        if !self.cloud_texture_loading {
            self.cloud_texture_loading = true;
            let ctx = _ctx.clone();
            let filename = filename.to_string();
            wasm_bindgen_futures::spawn_local(async move {
                let result = fetch_texture(&filename).await;
                CLOUD_TEXTURE_RESULT.with(|cell| { *cell.borrow_mut() = Some((res, result)); });
                ctx.request_repaint();
            });
        }
    }

    pub(crate) fn load_night_texture(&mut self, _ctx: &egui::Context) {
        if self.night_texture.is_some() {
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let filename = "textures/earth/Earth_Сities_16k.png";
            if let Ok(bytes) = std::fs::read(asset_path(filename)) {
                if let Ok(texture) = EarthTexture::from_bytes(&bytes) {
                    self.night_texture = Some(Arc::new(texture));
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        if !self.night_texture_loading {
            self.night_texture_loading = true;
            let ctx = _ctx.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let result = fetch_texture("textures/earth/Earth_Сities_16k.png").await;
                NIGHT_TEXTURE_RESULT.with(|cell| { *cell.borrow_mut() = Some(result); });
                ctx.request_repaint();
            });
        }
    }

    pub(crate) fn load_star_textures(&mut self, _ctx: &egui::Context) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.star_texture.is_none() {
                let filename = "textures/stars/8k_stars.jpg";
                if let Ok(bytes) = std::fs::read(asset_path(filename)) {
                    if let Ok(texture) = EarthTexture::from_bytes(&bytes) {
                        self.star_texture = Some(Arc::new(texture));
                    }
                }
            }
            if self.milky_way_texture.is_none() {
                let filename = "textures/stars/8k_stars_milky_way.jpg";
                if let Ok(bytes) = std::fs::read(asset_path(filename)) {
                    if let Ok(texture) = EarthTexture::from_bytes(&bytes) {
                        self.milky_way_texture = Some(Arc::new(texture));
                    }
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            if self.star_texture.is_none() && !self.star_texture_loading {
                self.star_texture_loading = true;
                let ctx = _ctx.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let result = fetch_texture("textures/stars/8k_stars.jpg").await;
                    STAR_TEXTURE_RESULT.with(|cell| { *cell.borrow_mut() = Some(result); });
                    ctx.request_repaint();
                });
            }
            if self.milky_way_texture.is_none() && !self.milky_way_texture_loading {
                self.milky_way_texture_loading = true;
                let ctx = _ctx.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let result = fetch_texture("textures/stars/8k_stars_milky_way.jpg").await;
                    MILKY_WAY_TEXTURE_RESULT.with(|cell| { *cell.borrow_mut() = Some(result); });
                    ctx.request_repaint();
                });
            }
        }
    }
}
