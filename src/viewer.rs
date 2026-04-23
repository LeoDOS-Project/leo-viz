//! Core viewer state and per-tab UI rendering.
//!
//! Owns the ViewerState struct (tabs, textures, camera state) and renders
//! each tab's planet views, constellation controls, TLE selection, and
//! satellite camera windows.

use crate::celestial::{CelestialBody, Skin, TextureResolution};
use crate::config::{
    AreaOfInterest, ConstellationConfig, DeviceLayer, GroundStation, NumericalState, Preset, Propagator, TabConfig, View3DFlags,
};
use crate::drawing::{
    draw_3d_view, draw_map_view, draw_torus, plane_color,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::geo::{GeoLoadState, GeoOverlayData};
use crate::texture::{TextureLoadState, EarthTexture, RingTexture};
use crate::tile::TileOverlayState;
use crate::time::{body_rotation_angle, continuous_day_of_year, DAYS_PER_YEAR, SOLAR_DECLINATION_MAX};
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
use eframe::egui;
use egui_dock::{TabViewer, NodeIndex, SurfaceIndex};
use egui_dock::tab_viewer::OnCloseResponse;
use nalgebra::{Matrix3, Vector3};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::sync::{Arc, mpsc};
use chrono::{DateTime, Utc};

pub(crate) struct ContextMenuState {
    pub(crate) screen_pos: egui::Pos2,
    pub(crate) lat: f64,
    pub(crate) lon: f64,
    pub(crate) tab_idx: usize,
    pub(crate) planet_idx: usize,
}

pub(crate) struct EditingPlaceState {
    pub(crate) is_gs: bool,
    pub(crate) idx: usize,
    pub(crate) tab_idx: usize,
    pub(crate) planet_idx: usize,
    pub(crate) screen_pos: egui::Pos2,
    pub(crate) just_opened: bool,
}

fn numerical_state_to_positions(
    wc: &WalkerConstellation,
    ns: &NumericalState,
) -> Vec<SatelliteState> {
    let sats_per_plane = wc.sats_per_plane();
    let is_star = wc.walker_type == WalkerType::Star;
    let np = wc.num_planes as isize;
    let sp = sats_per_plane as isize;

    let partial_plane = wc.raan_spacing_deg.is_some() && {
        let max_spread = match wc.walker_type {
            WalkerType::Delta => 2.0 * std::f64::consts::PI,
            WalkerType::Star => std::f64::consts::PI,
        };
        (wc.raan_step() * wc.num_planes as f64) < max_spread - 1e-9
    };
    let no_plane_wrap = is_star || partial_plane;
    let no_sat_wrap = wc.sat_spacing_km.is_some() && {
        let orbit_radius = wc.planet_radius + wc.altitude_km;
        let step = wc.sat_spacing_km.unwrap() / orbit_radius;
        (step * sats_per_plane as f64) < 2.0 * std::f64::consts::PI - 1e-9
    };

    let offsets: &[(isize, isize)] = match wc.isl_neighbors {
        4 => &[(0, 1), (1, 0), (0, -1), (-1, 0)],
        8 => &[(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)],
        _ => &[],
    };

    let mut positions: Vec<SatelliteState> = ns.sats.iter().enumerate().map(|(i, sat)| {
        let plane = i / sats_per_plane;
        let sat_index = i % sats_per_plane;
        let [x, y, z] = sat.pos;
        let r = (x * x + y * y + z * z).sqrt();
        let lat = (y / r).asin().to_degrees();
        let lon = -z.atan2(x).to_degrees();
        // Ascending if moving northward (vy > 0)
        let ascending = sat.vel[1] > 0.0;
        SatelliteState {
            plane, sat_index, x, y, z, lat, lon, ascending,
            neighbors: Vec::new(),
            name: None,
            tle_inclination_deg: None,
            tle_mean_motion: None,
        }
    }).collect();

    // Build neighbor links (same logic as WalkerConstellation)
    for i in 0..positions.len() {
        let plane = positions[i].plane as isize;
        let sat_idx = positions[i].sat_index as isize;
        let mut nbrs = Vec::new();
        for &(dp, ds) in offsets {
            let tp = plane + dp;
            let ts = sat_idx + ds;
            let tp = if no_plane_wrap {
                if tp < 0 || tp >= np { continue; } else { tp }
            } else {
                ((tp % np) + np) % np
            };
            let ts = if no_sat_wrap {
                if ts < 0 || ts >= sp { continue; } else { ts }
            } else {
                ((ts % sp) + sp) % sp
            };
            let j = (tp * sp + ts) as usize;
            if j < positions.len() && j > i {
                nbrs.push(j);
            }
        }
        positions[i].neighbors = nbrs;
    }

    positions
}

pub(crate) struct ViewerState {
    pub(crate) tabs: Vec<TabConfig>,
    pub(crate) camera_id_counter: usize,
    pub(crate) tab_counter: usize,
    pub(crate) torus_zoom: f64,
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
    pub(crate) slideshow_mode: bool,
    pub(crate) show_tab_info: bool,
    pub(crate) slideshow_fade_alpha: f32,
    pub(crate) use_gpu_rendering: bool,
    pub(crate) show_borders: bool,
    pub(crate) show_cities: bool,
    pub(crate) active_tab_idx: usize,
    pub(crate) prev_active_tab_idx: usize,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) geo_data: GeoLoadState,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) geo_fetch_rx: Option<mpsc::Receiver<Result<GeoOverlayData, String>>>,
    pub(crate) dragging_place: Option<(usize, usize, bool, usize)>,
    pub(crate) context_menu: Option<ContextMenuState>,
    pub(crate) editing_place: Option<EditingPlaceState>,
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
    pub(crate) render_state: Option<egui_wgpu::RenderState>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) tle_fetch_tx: mpsc::Sender<(TlePreset, Result<Vec<TleSatellite>, String>)>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) tle_fetch_rx: mpsc::Receiver<(TlePreset, Result<Vec<TleSatellite>, String>)>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) tile_overlay: TileOverlayState,
    pub(crate) view_width: f32,
    pub(crate) view_height: f32,
    pub(crate) solar_system_handles: HashMap<CelestialBody, egui::TextureHandle>,
    pub(crate) planet_sizes_handles: HashMap<CelestialBody, egui::TextureHandle>,
    pub(crate) ss_last_render_instant: Option<web_time::Instant>,
    pub(crate) planet_sizes_t: f64,
    pub(crate) planet_sizes_auto_zoom: bool,
    pub(crate) planet_sizes_zoom_duration: f32,
    pub(crate) planet_sizes_stay_duration: f32,
    pub(crate) planet_sizes_auto_time: f64,
    pub(crate) planet_sizes_enabled: std::collections::HashSet<CelestialBody>,
    pub(crate) ss_auto_zoom: bool,
    pub(crate) ss_auto_zoom_duration: f32,
    pub(crate) ss_auto_zoom_stay: f32,
    pub(crate) ss_auto_zoom_time: f64,
    pub(crate) asteroid_sprite: Option<egui::TextureHandle>,
    pub(crate) asteroid_state: crate::solar_system::AsteroidLoadState,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) asteroid_rx: Option<mpsc::Receiver<Result<Vec<crate::solar_system::Asteroid>, String>>>,
    pub(crate) hohmann: crate::solar_system::HohmannState,
    pub(crate) conjunction_body_a: CelestialBody,
    pub(crate) conjunction_body_b: CelestialBody,
    pub(crate) opposition_body_a: CelestialBody,
    pub(crate) opposition_body_b: CelestialBody,
    pub(crate) alignment_planets: [bool; 8],
    pub(crate) alignment_search_years: f64,
    #[cfg(target_arch = "wasm32")]
    pub(crate) last_url_hash: String,
    pub(crate) last_frame_instant: Option<web_time::Instant>,
    pub(crate) fps_smooth: f64,
    pub(crate) moon_image_handles: HashMap<CelestialBody, egui::TextureHandle>,
    pub(crate) editing_tab: Option<usize>,
}


impl TabViewer for ViewerState {
    type Tab = usize;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        self.tabs.get(*tab).map(|t| t.name.as_str()).unwrap_or("?").into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        if *tab < self.tabs.len() {
            if self.prev_active_tab_idx != *tab && self.prev_active_tab_idx < self.tabs.len() {
                // Full reset on tab switch so each demo opens in its designed
                // initial state, regardless of whatever state the previous tab
                // accumulated.
                let tab_mut = &mut self.tabs[*tab];
                tab_mut.settings.time = 0.0;
                tab_mut.settings.auto_zoom_time = 0.0;
                if let Some(init) = tab_mut.settings.initial_rotation {
                    tab_mut.settings.rotation = init;
                }
                for planet in &mut tab_mut.planets {
                    planet.ground_track_history.clear();
                    planet.conjunction_prev_positions.clear();
                    planet.kessler.collided_pairs.clear();
                    planet.kessler.debris.clear();
                    planet.kessler.collision_count = 0;
                    planet.kessler.collision_id_counter = 0;
                    planet.kessler.active_corrections.clear();
                    planet.kessler.corrections_made = 0;
                    for cons in &mut planet.constellations {
                        cons.numerical = None;
                        cons.physics_state.clear();
                    }
                }
                // Auto-cycle countdown restarts so the new tab gets a full interval.
                self.last_cycle_time = 0.0;
                self.ss_auto_zoom_time = 0.0;
                self.planet_sizes_auto_time = 0.0;
            }
            self.prev_active_tab_idx = *tab;
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

    fn context_menu(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab, _surface: SurfaceIndex, _node: NodeIndex) {
        if ui.button("Edit tab info...").clicked() {
            self.editing_tab = Some(*tab);
            ui.close();
        }
    }
}


impl ViewerState {
    #[cfg(not(target_arch = "wasm32"))]
    fn tile_overlay_detail_bounds(&self, body: CelestialBody) -> Option<[f32; 4]> {
        if !self.tile_overlay.enabled || body != CelestialBody::Earth {
            return None;
        }
        let dt = self.tile_overlay.detail_texture.as_ref()?;
        Some([
            dt.bounds.min_lon as f32,
            dt.bounds.max_lon as f32,
            dt.bounds.min_lat as f32,
            dt.bounds.max_lat as f32,
        ])
    }

    #[cfg(target_arch = "wasm32")]
    fn tile_overlay_detail_bounds(&self, _body: CelestialBody) -> Option<[f32; 4]> {
        None
    }

    fn render_tab_ui(&mut self, ui: &mut egui::Ui, tab_idx: usize) {
        let now = web_time::Instant::now();
        if let Some(prev) = self.last_frame_instant {
            let dt = now.duration_since(prev).as_secs_f64();
            if dt > 0.0 {
                let instant_fps = 1.0 / dt;
                self.fps_smooth = self.fps_smooth * 0.9 + instant_fps * 0.1;
            }
        }
        self.last_frame_instant = Some(now);

        if self.tabs[tab_idx].show_fps {
            let fps_text = format!("{:.0} FPS", self.fps_smooth);
            let rect = ui.available_rect_before_wrap();
            let pos = egui::pos2(rect.right() - 70.0, rect.top() + 4.0);
            ui.painter().text(
                pos,
                egui::Align2::LEFT_TOP,
                fps_text,
                egui::FontId::monospace(14.0),
                egui::Color32::from_rgb(200, 200, 200),
            );
        }

        if self.auto_cycle_tabs && self.show_tab_info {
            let rect = ui.available_rect_before_wrap();
            let frac = (self.last_cycle_time / self.cycle_interval).min(1.0) as f32;
            let bar_rect = egui::Rect::from_min_size(
                egui::pos2(rect.left(), rect.bottom() - 3.0),
                egui::vec2(rect.width() * frac, 3.0),
            );
            ui.painter().rect_filled(bar_rect, 0.0, egui::Color32::from_rgba_unmultiplied(255, 255, 255, 80));
        }

        if self.auto_cycle_tabs && !self.show_tab_info {
            let tab = &self.tabs[tab_idx];
            let has_title = !tab.title.is_empty();
            let has_desc = !tab.description.is_empty();
            let rect = ui.available_rect_before_wrap();
            let painter = ui.painter();

            if self.slideshow_mode {
                if has_title || has_desc {
                    let shadow_offset = egui::vec2(2.0, 2.0);
                    let mut y = rect.top() + 30.0;
                    if has_title {
                        let pos = egui::pos2(rect.left() + 30.0, y);
                        let font = egui::FontId::proportional(32.0);
                        painter.text(
                            pos + shadow_offset,
                            egui::Align2::LEFT_TOP,
                            &tab.title,
                            font.clone(),
                            egui::Color32::BLACK,
                        );
                        painter.text(
                            pos,
                            egui::Align2::LEFT_TOP,
                            &tab.title,
                            font.clone(),
                            egui::Color32::WHITE,
                        );
                        let title_h = painter.layout_no_wrap(
                            tab.title.clone(),
                            font,
                            egui::Color32::WHITE,
                        ).rect.height();
                        y += title_h + 8.0;
                    }
                    if has_desc {
                        let pos = egui::pos2(rect.left() + 30.0, y);
                        let font = egui::FontId::proportional(20.0);
                        let desc_color = egui::Color32::from_rgba_unmultiplied(255, 255, 255, 200);
                        let plain = crate::config::strip_bold_markers(&tab.description);
                        painter.text(
                            pos + shadow_offset,
                            egui::Align2::LEFT_TOP,
                            &plain,
                            font.clone(),
                            egui::Color32::BLACK,
                        );
                        painter.text(
                            pos,
                            egui::Align2::LEFT_TOP,
                            &plain,
                            font,
                            desc_color,
                        );
                    }
                }

                let elapsed = self.last_cycle_time;
                let total = self.cycle_interval + 0.5;
                let frac = (elapsed / total).min(1.0) as f32;
                let bar_rect = egui::Rect::from_min_size(
                    egui::pos2(rect.left(), rect.bottom() - 4.0),
                    egui::vec2(rect.width() * frac, 4.0),
                );
                painter.rect_filled(
                    bar_rect,
                    0.0,
                    egui::Color32::from_rgba_unmultiplied(255, 255, 255, 100),
                );

                let fade_black = ((1.0 - self.slideshow_fade_alpha) * 255.0) as u8;
                if fade_black > 0 {
                    painter.rect_filled(
                        rect,
                        0.0,
                        egui::Color32::from_rgba_unmultiplied(0, 0, 0, fade_black),
                    );
                }
            } else if has_title || has_desc {
                let mut y = rect.bottom() - 20.0;
                if has_desc {
                    let plain = crate::config::strip_bold_markers(&tab.description);
                    y -= painter.layout_no_wrap(
                        plain.clone(),
                        egui::FontId::proportional(18.0),
                        egui::Color32::WHITE,
                    ).rect.height();
                    painter.text(
                        egui::pos2(rect.center().x, y),
                        egui::Align2::CENTER_BOTTOM,
                        &plain,
                        egui::FontId::proportional(18.0),
                        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 200),
                    );
                    y -= 8.0;
                }
                if has_title {
                    painter.text(
                        egui::pos2(rect.center().x, y),
                        egui::Align2::CENTER_BOTTOM,
                        &tab.title,
                        egui::FontId::proportional(28.0),
                        egui::Color32::WHITE,
                    );
                }
            }
        }

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
        // Up to 3 planets share a single row; 4+ planets break into a 3-column
        // grid that wraps onto additional rows.
        let cols = num_planets.min(3);
        let rows = (num_planets + cols - 1) / cols.max(1);
        let col_gap_total = gap * (cols.saturating_sub(1)) as f32;
        let row_gap_total = gap * (rows.saturating_sub(1)) as f32;
        let cell_width = (available_rect.width() - col_gap_total) / cols.max(1) as f32;
        let cell_height = (available_rect.height() - row_gap_total) / rows.max(1) as f32;

        let mut add_planet = false;
        let mut planet_to_remove: Option<usize> = None;

        for planet_idx in 0..num_planets {
            let col = planet_idx % cols;
            let row = planet_idx / cols;
            let x_offset = col as f32 * (cell_width + gap);
            let y_offset = row as f32 * (cell_height + gap);
            let planet_rect = egui::Rect::from_min_size(
                egui::pos2(
                    available_rect.min.x + x_offset,
                    available_rect.min.y + y_offset,
                ),
                egui::vec2(cell_width, cell_height),
            );

            ui.scope_builder(egui::UiBuilder::new().max_rect(planet_rect), |ui| {
                let (should_add, should_remove) = self.render_planet_ui(ui, tab_idx, planet_idx, num_planets);
                if should_add { add_planet = true; }
                if should_remove { planet_to_remove = Some(planet_idx); }
            });

            // Vertical separator between columns within the same row.
            if col < cols - 1 {
                let sep_x = available_rect.min.x + x_offset + cell_width + gap * 0.5;
                ui.painter().line_segment(
                    [
                        egui::pos2(sep_x, planet_rect.min.y),
                        egui::pos2(sep_x, planet_rect.max.y),
                    ],
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 80, 80)),
                );
            }
            // Horizontal separator between rows.
            if row < rows - 1 && col == cols - 1 {
                let sep_y = available_rect.min.y + y_offset + cell_height + gap * 0.5;
                ui.painter().line_segment(
                    [
                        egui::pos2(available_rect.min.x, sep_y),
                        egui::pos2(available_rect.max.x, sep_y),
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

        if self.dragging_place.is_some() {
            self.context_menu = None;
        }

        if let Some(ref cm) = self.context_menu {
            if cm.tab_idx == tab_idx {
                let cm_lat = cm.lat;
                let cm_lon = cm.lon;
                let cm_planet_idx = cm.planet_idx;
                let area_resp = egui::Area::new(egui::Id::new("planet_context_menu"))
                    .fixed_pos(cm.screen_pos)
                    .order(egui::Order::Foreground)
                    .show(ui.ctx(), |ui| {
                        egui::Frame::popup(ui.style()).show(ui, |ui| {
                            ui.label(format!(
                                "{:.2}°, {:.2}°",
                                cm_lat, cm_lon,
                            ));
                            ui.separator();
                            let add_gs = ui.button("Add Ground Station").on_hover_text("Place a ground station at this location").clicked();
                            let add_aoi = ui.button("Add Area of Interest").on_hover_text("Define an area of interest at this location").clicked();
                            (add_gs, add_aoi)
                        }).inner
                    });
                let (add_gs, add_aoi) = area_resp.inner;
                let clicked_outside = ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary))
                    && !area_resp.response.rect.contains(
                        ui.input(|i| i.pointer.interact_pos().unwrap_or_default()),
                    );
                if add_gs {
                    if let Some(planet) = self.tabs[tab_idx].planets.get_mut(cm_planet_idx) {
                        let n = planet.ground_stations.len() + 1;
                        planet.ground_stations.push(GroundStation {
                            name: format!("GS{}", n),
                            lat: cm_lat,
                            lon: cm_lon,
                            radius_km: 500.0,
                            color: egui::Color32::from_rgb(255, 100, 100),
                            selected: false,
                        });
                    }
                    self.context_menu = None;
                } else if add_aoi {
                    if let Some(planet) = self.tabs[tab_idx].planets.get_mut(cm_planet_idx) {
                        let n = planet.areas_of_interest.len() + 1;
                        planet.areas_of_interest.push(AreaOfInterest {
                            name: format!("AOI{}", n),
                            lat: cm_lat,
                            lon: cm_lon,
                            radius_km: 500.0,
                            color: egui::Color32::from_rgba_unmultiplied(100, 200, 100, 100),
                            ground_station_idx: None,
                            job_mode: crate::config::AoiJobMode::Route,
                            job_n: 3,
                            selected: false,
                        });
                    }
                    self.context_menu = None;
                } else if clicked_outside {
                    self.context_menu = None;
                }
            }
        }

        if let Some(ref ep) = self.editing_place {
            if ep.tab_idx == tab_idx {
                let ep_is_gs = ep.is_gs;
                let ep_idx = ep.idx;
                let ep_planet_idx = ep.planet_idx;
                let ep_screen_pos = ep.screen_pos;
                let ep_just_opened = ep.just_opened;
                let title = if ep_is_gs {
                    self.tabs[tab_idx].planets.get(ep_planet_idx)
                        .and_then(|p| p.ground_stations.get(ep_idx))
                        .map(|gs| format!("Edit GS: {}", gs.name))
                } else {
                    self.tabs[tab_idx].planets.get(ep_planet_idx)
                        .and_then(|p| p.areas_of_interest.get(ep_idx))
                        .map(|aoi| format!("Edit AOI: {}", aoi.name))
                };
                if let Some(title) = title {
                    let mut open = true;
                    let mut win = egui::Window::new(title)
                        .id(egui::Id::new("edit_place_window"))
                        .open(&mut open)
                        .title_bar(true)
                        .collapsible(false)
                        .default_width(200.0);
                    if ep_just_opened {
                        win = win.current_pos(ep_screen_pos);
                    }
                    let delete = win.show(ui.ctx(), |ui| {
                            let mut del = false;
                            if ep_is_gs {
                                if let Some(gs) = self.tabs[tab_idx].planets
                                    .get_mut(ep_planet_idx)
                                    .and_then(|p| p.ground_stations.get_mut(ep_idx))
                                {
                                    ui.horizontal(|ui| {
                                        ui.label("Name");
                                        ui.text_edit_singleline(&mut gs.name);
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Lat");
                                        ui.add(egui::DragValue::new(&mut gs.lat).speed(0.1).range(-90.0..=90.0));
                                        ui.label("Lon");
                                        ui.add(egui::DragValue::new(&mut gs.lon).speed(0.1).range(-180.0..=180.0));
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Radius (km)");
                                        ui.add(egui::DragValue::new(&mut gs.radius_km).speed(1.0).range(0.0..=f64::MAX));
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Color");
                                        let mut c = gs.color.to_array();
                                        if ui.color_edit_button_srgba_unmultiplied(&mut c).changed() {
                                            gs.color = egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]);
                                        }
                                    });
                                    ui.separator();
                                    if ui.button("Delete").clicked() { del = true; }
                                }
                            } else {
                                let gs_names: Vec<String> = self.tabs[tab_idx].planets
                                    .get(ep_planet_idx)
                                    .map(|p| p.ground_stations.iter().map(|gs| gs.name.clone()).collect())
                                    .unwrap_or_default();
                                if let Some(aoi) = self.tabs[tab_idx].planets
                                    .get_mut(ep_planet_idx)
                                    .and_then(|p| p.areas_of_interest.get_mut(ep_idx))
                                {
                                    ui.horizontal(|ui| {
                                        ui.label("Name");
                                        ui.text_edit_singleline(&mut aoi.name);
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Lat");
                                        ui.add(egui::DragValue::new(&mut aoi.lat).speed(0.1).range(-90.0..=90.0));
                                        ui.label("Lon");
                                        ui.add(egui::DragValue::new(&mut aoi.lon).speed(0.1).range(-180.0..=180.0));
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Radius (km)");
                                        ui.add(egui::DragValue::new(&mut aoi.radius_km).speed(1.0).range(0.0..=f64::MAX));
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Color");
                                        let mut c = aoi.color.to_array();
                                        if ui.color_edit_button_srgba_unmultiplied(&mut c).changed() {
                                            aoi.color = egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]);
                                        }
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("GS");
                                        let gs_label = aoi.ground_station_idx
                                            .and_then(|i| gs_names.get(i))
                                            .map(|s| s.as_str())
                                            .unwrap_or("None");
                                        egui::ComboBox::from_id_salt("edit_aoi_gs_link")
                                            .selected_text(gs_label)
                                            .show_ui(ui, |ui| {
                                                if ui.selectable_label(aoi.ground_station_idx.is_none(), "None").clicked() {
                                                    aoi.ground_station_idx = None;
                                                }
                                                for (gs_idx, name) in gs_names.iter().enumerate() {
                                                    if ui.selectable_label(aoi.ground_station_idx == Some(gs_idx), name).clicked() {
                                                        aoi.ground_station_idx = Some(gs_idx);
                                                    }
                                                }
                                            });
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Mode");
                                        egui::ComboBox::from_id_salt("edit_aoi_mode")
                                            .selected_text(match aoi.job_mode {
                                                crate::config::AoiJobMode::Route => "Route",
                                                crate::config::AoiJobMode::SpaceComp => "SpaceCoMP",
                                            })
                                            .show_ui(ui, |ui| {
                                                if ui.selectable_label(aoi.job_mode == crate::config::AoiJobMode::Route, "Route").clicked() {
                                                    aoi.job_mode = crate::config::AoiJobMode::Route;
                                                }
                                                if ui.selectable_label(aoi.job_mode == crate::config::AoiJobMode::SpaceComp, "SpaceCoMP").clicked() {
                                                    aoi.job_mode = crate::config::AoiJobMode::SpaceComp;
                                                }
                                            });
                                        if aoi.job_mode == crate::config::AoiJobMode::SpaceComp {
                                            ui.label("n");
                                            ui.add(egui::DragValue::new(&mut aoi.job_n).range(1..=50).speed(0.2));
                                        }
                                    });
                                    ui.separator();
                                    if ui.button("Delete").clicked() { del = true; }
                                }
                            }
                            del
                        });
                    let should_delete = delete.and_then(|r| r.inner).unwrap_or(false);
                    if !open || should_delete {
                        if should_delete {
                            if let Some(planet) = self.tabs[tab_idx].planets.get_mut(ep_planet_idx) {
                                if ep_is_gs {
                                    if ep_idx < planet.ground_stations.len() {
                                        planet.ground_stations.remove(ep_idx);
                                        for aoi in &mut planet.areas_of_interest {
                                            match aoi.ground_station_idx {
                                                Some(i) if i == ep_idx => aoi.ground_station_idx = None,
                                                Some(i) if i > ep_idx => aoi.ground_station_idx = Some(i - 1),
                                                _ => {}
                                            }
                                        }
                                    }
                                } else if ep_idx < planet.areas_of_interest.len() {
                                    planet.areas_of_interest.remove(ep_idx);
                                }
                            }
                        }
                        self.editing_place = None;
                    } else if ep_just_opened {
                        if let Some(ref mut ep) = self.editing_place {
                            ep.just_opened = false;
                        }
                    }
                } else {
                    self.editing_place = None;
                }
            }
        }
    }

    fn render_planet_ui(&mut self, ui: &mut egui::Ui, tab_idx: usize, planet_idx: usize, num_planets: usize) -> (bool, bool) {
        #[cfg(not(target_arch = "wasm32"))]
        let constrained_width = ui.available_width();
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

        if self.show_tab_info {
            ui.vertical_centered(|ui| {
                ui.add_space(4.0);
                ui.label(egui::RichText::new(&planet_name).strong().size(16.0));
            });
        } else if self.ui_visible {
        egui::ScrollArea::horizontal()
            .id_salt(("planet_hscroll", tab_idx, planet_idx))
            .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
            .show(ui, |ui| { ui.horizontal(|ui| {
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
            let show_conj = self.tabs[tab_idx].planets[planet_idx].show_conjunction_window;
            if ui.selectable_label(show_conj, "Conj").clicked() {
                self.tabs[tab_idx].planets[planet_idx].show_conjunction_window = !show_conj;
            }
            let show_rad = self.tabs[tab_idx].planets[planet_idx].show_radiation_window;
            if ui.selectable_label(show_rad, "Rad").clicked() {
                self.tabs[tab_idx].planets[planet_idx].show_radiation_window = !show_rad;
            }
            {
                let has_moons = !self.tabs[tab_idx].planets[planet_idx].celestial_body.moons().is_empty();
                let show_m = self.tabs[tab_idx].planets[planet_idx].show_moons_window;
                if ui.add_enabled(has_moons, egui::Button::new("Moons").selected(show_m)).clicked() {
                    self.tabs[tab_idx].planets[planet_idx].show_moons_window = !show_m;
                }
            }
            let show_fps = self.tabs[tab_idx].show_fps;
            if ui.selectable_label(show_fps, "FPS").clicked() {
                self.tabs[tab_idx].show_fps = !show_fps;
            }
        }); });
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

            let has_selected = ground_stations.iter().any(|gs| gs.selected)
                || areas_of_interest.iter().any(|a| a.selected);

            let current_time = self.tabs[tab_idx].settings.time;
            let body = planet.celestial_body;
            let constellations_clone: Vec<_> = planet.constellations.iter()
                .filter(|c| !c.hidden)
                .cloned()
                .collect();
            let start_ts = self.start_timestamp;
            let mut pass_cache = planet.pass_cache.clone();

            let any_constellations = !constellations_clone.is_empty();
            let selected_sats: Vec<(usize, usize, usize)> = planet.satellite_cameras.iter()
                .map(|c| (c.constellation_idx, c.plane, c.sat_index))
                .collect();

            let cache_stale = has_selected
                && any_constellations
                && (current_time - pass_cache.last_compute_time).abs() > 5.0;

            if cache_stale {
                pass_cache.passes.clear();
                let window_sec = pass_cache.prediction_window_min * 60.0;

                for (idx, gs) in ground_stations.iter().enumerate() {
                    if !gs.selected { continue; }
                    let p = crate::pass::compute_passes(
                        gs.lat, gs.lon, gs.radius_km,
                        &constellations_clone, &selected_sats,
                        current_time, window_sec, body, start_ts,
                    );
                    pass_cache.passes.insert(idx, p);
                }
                let gs_count = ground_stations.len();
                for (idx, aoi) in areas_of_interest.iter().enumerate() {
                    if !aoi.selected { continue; }
                    let p = crate::pass::compute_passes(
                        aoi.lat, aoi.lon, aoi.radius_km,
                        &constellations_clone, &selected_sats,
                        current_time, window_sec, body, start_ts,
                    );
                    pass_cache.passes.insert(gs_count + idx, p);
                }
                pass_cache.last_compute_time = current_time;
            }

            let pass_cache_for_ui = pass_cache.clone();
            let mut fast_forward_to: Option<(f64, f64, f64)> = None;

            egui::Window::new(format!("Ground - {}", planet_name))
                .open(&mut self.tabs[tab_idx].planets[planet_idx].show_gs_aoi_window)
                .default_width(if has_selected { 700.0 } else { 350.0 })
                .show(ui.ctx(), |ui| {
                    ui.columns(if has_selected { 2 } else { 1 }, |cols| {
                        let left = &mut cols[0];

                        left.heading("Ground Stations");
                        let mut gs_to_remove = None;
                        let mut gs_pass_clicked: Option<usize> = None;
                        for (idx, gs) in ground_stations.iter_mut().enumerate() {
                            left.horizontal(|ui| {
                                if ui.add_sized([70.0, 18.0], egui::TextEdit::singleline(&mut gs.name)).changed() {
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
                                ui.label("R:");
                                if ui.add(egui::DragValue::new(&mut gs.radius_km).range(1.0..=5000.0).speed(10.0).suffix(" km")).changed() {
                                    gs_changed = true;
                                }
                                if ui.small_button("×").clicked() {
                                    gs_to_remove = Some(idx);
                                }
                                if ui.checkbox(&mut gs.selected, "Track").on_hover_text("Show satellite passes over this station").changed() {
                                    gs_changed = true;
                                    if gs.selected {
                                        gs_pass_clicked = Some(idx);
                                    }
                                }
                            });
                        }
                        if let Some(clicked) = gs_pass_clicked {
                            for (i, gs) in ground_stations.iter_mut().enumerate() {
                                if i != clicked { gs.selected = false; }
                            }
                            for aoi in areas_of_interest.iter_mut() {
                                aoi.selected = false;
                            }
                            aoi_changed = true;
                        }
                        if let Some(idx) = gs_to_remove {
                            ground_stations.remove(idx);
                            gs_changed = true;
                        }
                        if left.button("+ Add ground station").on_hover_text("Add a new ground station").clicked() {
                            ground_stations.push(GroundStation {
                                name: format!("GS{}", ground_stations.len() + 1),
                                lat: 0.0,
                                lon: 0.0,
                                radius_km: 500.0,
                                color: egui::Color32::from_rgb(255, 100, 100),
                                selected: false,
                            });
                            gs_changed = true;
                        }

                        left.separator();
                        left.heading("Areas of Interest");
                        let mut aoi_to_remove = None;
                        let mut aoi_pass_clicked: Option<usize> = None;
                        for (idx, aoi) in areas_of_interest.iter_mut().enumerate() {
                            left.horizontal(|ui| {
                                if ui.add_sized([70.0, 18.0], egui::TextEdit::singleline(&mut aoi.name)).changed() {
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
                                ui.label("R:");
                                if ui.add(egui::DragValue::new(&mut aoi.radius_km).range(1.0..=5000.0).speed(10.0).suffix(" km")).changed() {
                                    aoi_changed = true;
                                }
                                if ui.small_button("×").clicked() {
                                    aoi_to_remove = Some(idx);
                                }
                                if ui.checkbox(&mut aoi.selected, "Track").on_hover_text("Show satellite passes over this area").changed() {
                                    aoi_changed = true;
                                    if aoi.selected {
                                        aoi_pass_clicked = Some(idx);
                                    }
                                }
                            });
                            left.horizontal(|ui| {
                                ui.add_space(22.0);
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
                                ui.label("Mode:");
                                egui::ComboBox::from_id_salt(format!("aoi_mode_{}", idx))
                                    .selected_text(match aoi.job_mode {
                                        crate::config::AoiJobMode::Route => "Route",
                                        crate::config::AoiJobMode::SpaceComp => "SpaceCoMP",
                                    })
                                    .show_ui(ui, |ui| {
                                        if ui.selectable_label(aoi.job_mode == crate::config::AoiJobMode::Route, "Route").clicked() {
                                            aoi.job_mode = crate::config::AoiJobMode::Route;
                                            aoi_changed = true;
                                        }
                                        if ui.selectable_label(aoi.job_mode == crate::config::AoiJobMode::SpaceComp, "SpaceCoMP").clicked() {
                                            aoi.job_mode = crate::config::AoiJobMode::SpaceComp;
                                            aoi_changed = true;
                                        }
                                    });
                                if aoi.job_mode == crate::config::AoiJobMode::SpaceComp {
                                    ui.label("n:");
                                    if ui.add(egui::DragValue::new(&mut aoi.job_n).range(1..=50).speed(0.2)).changed() {
                                        aoi_changed = true;
                                    }
                                }
                            });
                        }
                        if let Some(clicked) = aoi_pass_clicked {
                            for (i, aoi) in areas_of_interest.iter_mut().enumerate() {
                                if i != clicked { aoi.selected = false; }
                            }
                            for gs in ground_stations.iter_mut() {
                                gs.selected = false;
                            }
                            gs_changed = true;
                        }
                        if let Some(idx) = aoi_to_remove {
                            areas_of_interest.remove(idx);
                            aoi_changed = true;
                        }
                        if left.button("+ Add area of interest").clicked() {
                            areas_of_interest.push(AreaOfInterest {
                                name: format!("AOI{}", areas_of_interest.len() + 1),
                                lat: 0.0,
                                lon: 0.0,
                                radius_km: 500.0,
                                color: egui::Color32::from_rgba_unmultiplied(100, 200, 100, 100),
                                ground_station_idx: None,
                                job_mode: crate::config::AoiJobMode::Route,
                                job_n: 3,
                                selected: false,
                            });
                            aoi_changed = true;
                        }

                        left.separator();
                        left.heading("Devices");
                        let mut layer_to_remove = None;
                        for (li, layer) in device_layers.iter_mut().enumerate() {
                            left.horizontal(|ui| {
                                if ui.add_sized([80.0, 18.0], egui::TextEdit::singleline(&mut layer.name)).changed() {
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
                                .show(left, |ui| {
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
                            if left.small_button("+ Add device").clicked() {
                                layer.devices.push((0.0, 0.0));
                                dev_changed = true;
                            }
                            left.separator();
                        }
                        if let Some(li) = layer_to_remove {
                            device_layers.remove(li);
                            dev_changed = true;
                        }
                        if left.button("+ Add device layer").clicked() {
                            device_layers.push(DeviceLayer {
                                name: format!("Layer {}", device_layers.len() + 1),
                                color: egui::Color32::from_rgb(80, 140, 255),
                                devices: Vec::new(),
                            });
                            dev_changed = true;
                        }

                        if has_selected && cols.len() > 1 {
                            let right = &mut cols[1];
                            right.heading("Pass Predictions");

                            if !any_constellations {
                                right.weak("No constellations configured");
                            } else {
                                let gs_count = ground_stations.len();
                                let selected_items: Vec<(usize, String, f64, f64)> = ground_stations.iter()
                                    .enumerate()
                                    .filter(|(_, gs)| gs.selected)
                                    .map(|(i, gs)| (i, gs.name.clone(), gs.lat, gs.lon))
                                    .chain(
                                        areas_of_interest.iter()
                                            .enumerate()
                                            .filter(|(_, a)| a.selected)
                                            .map(|(i, a)| (gs_count + i, a.name.clone(), a.lat, a.lon))
                                    )
                                    .collect();

                                egui::ScrollArea::vertical()
                                    .id_salt("pass_scroll")
                                    .max_height(400.0)
                                    .show(right, |ui| {
                                        for (key, name, gs_lat, gs_lon) in &selected_items {
                                            ui.strong(name);
                                            if let Some(passes) = pass_cache_for_ui.passes.get(key) {
                                                if selected_sats.is_empty() {
                                                    ui.weak("Click satellites on globe to track");
                                                } else {
                                                    let window_min = pass_cache_for_ui.prediction_window_min;
                                                    let window_label = if window_min >= 1440.0 {
                                                        format!("{:.0}d", window_min / 1440.0)
                                                    } else {
                                                        format!("{:.0}h", window_min / 60.0)
                                                    };
                                                    egui::Grid::new(format!("pass_grid_{}", key))
                                                        .striped(true)
                                                        .show(ui, |ui| {
                                                            ui.strong("Satellite");
                                                            ui.strong("Arrival");
                                                            ui.strong("Remaining");
                                                            ui.strong("In Zone");
                                                            ui.strong("Max El");
                                                            ui.strong("Alt");
                                                            ui.strong("Dir");
                                                            ui.strong("");
                                                            ui.end_row();
                                                            let elapsed = current_time - pass_cache_for_ui.last_compute_time;
                                                            let dim = egui::Color32::from_rgb(120, 120, 120);
                                                            let bright = egui::Color32::from_rgb(230, 230, 230);
                                                            let very_dim = egui::Color32::from_rgb(80, 80, 80);
                                                            for pass in passes.iter() {
                                                                let adjusted_aos = (pass.time_to_aos - elapsed).max(0.0);
                                                                let in_zone = adjusted_aos < 0.1;
                                                                let color = if in_zone { bright } else { dim };
                                                                let rt = |s: String| {
                                                                    egui::RichText::new(s).color(color)
                                                                };
                                                                ui.label(rt(pass.sat_name.clone()));
                                                                let aos_time = start_ts + chrono::Duration::milliseconds(((current_time + adjusted_aos) * 1000.0) as i64);
                                                                let local_aos: chrono::DateTime<chrono::Local> = aos_time.into();
                                                                if in_zone {
                                                                    ui.label(egui::RichText::new("NOW").color(egui::Color32::GREEN));
                                                                    ui.label(egui::RichText::new("—").color(egui::Color32::GREEN));
                                                                } else {
                                                                    ui.label(rt(local_aos.format("%H:%M:%S %d/%m/%y").to_string()));
                                                                    let h = (adjusted_aos / 3600.0) as u64;
                                                                    let m = ((adjusted_aos % 3600.0) / 60.0) as u64;
                                                                    let s = adjusted_aos % 60.0;
                                                                    let text = if h > 0 {
                                                                        format!("{h}h{m}m{s:.0}s")
                                                                    } else if m > 0 {
                                                                        format!("{m}m{s:.0}s")
                                                                    } else {
                                                                        format!("{s:.1}s")
                                                                    };
                                                                    ui.label(rt(text));
                                                                }
                                                                let zone_time = if in_zone {
                                                                    (pass.time_to_aos + pass.duration - elapsed).max(0.0)
                                                                } else {
                                                                    pass.duration
                                                                };
                                                                ui.label(rt(format!("{:.1}s", zone_time)));
                                                                ui.label(rt(format!("{:.1}°", pass.max_elevation)));
                                                                ui.label(rt(format!("{:.0} km", pass.altitude_km)));
                                                                ui.label(rt(if pass.ascending { "Asc".into() } else { "Desc".into() }));
                                                                if !in_zone && ui.small_button("FF ⏩").clicked() {
                                                                    fast_forward_to = Some((current_time + adjusted_aos, *gs_lat, *gs_lon));
                                                                }
                                                                ui.end_row();
                                                            }
                                                            let sats_with_pass: std::collections::HashSet<(usize, usize, usize)> = passes.iter()
                                                                .map(|p| (p.constellation_idx, p.sat_plane, p.sat_index))
                                                                .collect();
                                                            for &(ci, plane, sat_idx) in &selected_sats {
                                                                if sats_with_pass.contains(&(ci, plane, sat_idx)) { continue; }
                                                                let name = if let Some(c) = constellations_clone.get(ci) {
                                                                    format!("{} P{}:S{}", c.preset_name(), plane, sat_idx)
                                                                } else {
                                                                    format!("P{}:S{}", plane, sat_idx)
                                                                };
                                                                let rt = |s: String| egui::RichText::new(s).color(very_dim);
                                                                ui.label(rt(name));
                                                                ui.label(rt(format!(">{window_label}")));
                                                                ui.label(rt("-".into()));
                                                                ui.label(rt("-".into()));
                                                                ui.label(rt("-".into()));
                                                                ui.label(rt("-".into()));
                                                                ui.label(rt("-".into()));
                                                                ui.label(rt("".into()));
                                                                ui.end_row();
                                                            }
                                                        });
                                                }
                                            } else {
                                                ui.weak("Computing...");
                                            }
                                            ui.add_space(8.0);
                                        }
                                    });
                            }
                        }
                    });
                });

            if gs_changed {
                self.tabs[tab_idx].planets[planet_idx].ground_stations = ground_stations;
                pass_cache.last_compute_time = f64::NEG_INFINITY;
            }
            if aoi_changed {
                self.tabs[tab_idx].planets[planet_idx].areas_of_interest = areas_of_interest;
                pass_cache.last_compute_time = f64::NEG_INFINITY;
            }
            if dev_changed {
                self.tabs[tab_idx].planets[planet_idx].device_layers = device_layers;
            }
            self.tabs[tab_idx].planets[planet_idx].pass_cache = pass_cache;
            if let Some((t, lat, lon)) = fast_forward_to {
                self.tabs[tab_idx].settings.time = t;
                let sim_time = self.start_timestamp + chrono::Duration::milliseconds((t * 1000.0) as i64);
                let gmst = crate::time::greenwich_mean_sidereal_time(sim_time);
                let body_rot = crate::time::body_rotation_angle(body, t, gmst);
                let target_lon = if self.tabs[tab_idx].settings.earth_fixed_camera {
                    lon.to_radians()
                } else {
                    lon.to_radians() + body_rot
                };
                self.tabs[tab_idx].settings.rotation = crate::math::lat_lon_to_matrix(lat.to_radians(), target_lon);
            }
        }

        let show_stats = self.tabs[tab_idx].show_stats;
        let mut open_sat_list = false;
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
                    ui.separator();
                    if ui.button("Satellite List").on_hover_text("Open the satellite list window").clicked() {
                        open_sat_list = true;
                    }
                });
        }
        if open_sat_list {
            self.tabs[tab_idx].show_sat_list = true;
        }

        if self.tabs[tab_idx].show_sat_list {
            let planet = &self.tabs[tab_idx].planets[planet_idx];
            let planet_name = planet.name.clone();
            let planet_radius = planet.celestial_body.radius_km();
            let mu = planet.celestial_body.mu();
            let tle_selections = planet.tle_selections.clone();
            let constellations = planet.constellations.clone();
            let mut show = self.tabs[tab_idx].show_sat_list;
            egui::Window::new(format!("Satellites - {}", planet_name))
                .open(&mut show)
                .default_width(600.0)
                .default_height(400.0)
                .show(ui.ctx(), |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("sat_list_scroll")
                        .show(ui, |ui| {
                            egui::Grid::new("sat_list_grid")
                                .striped(true)
                                .show(ui, |ui| {
                                    ui.strong("Name");
                                    ui.strong("Source");
                                    ui.strong("Alt (km)");
                                    ui.strong("Inc (°)");
                                    ui.strong("Period (min)");
                                    ui.end_row();

                                    for cons in &constellations {
                                        if cons.hidden { continue; }
                                        let label = cons.preset_name();
                                        let spp = cons.sats_per_plane;
                                        for p in 0..cons.num_planes {
                                            for s in 0..spp {
                                                ui.label(format!("{} P{}:S{}", label, p, s));
                                                ui.label(label);
                                                ui.label(format!("{:.0}", cons.altitude_km));
                                                ui.label(format!("{:.1}", cons.inclination));
                                                let r = planet_radius + cons.altitude_km;
                                                let period = 2.0 * std::f64::consts::PI * (r * r * r / mu).sqrt() / 60.0;
                                                ui.label(format!("{:.1}", period));
                                                ui.end_row();
                                            }
                                        }
                                    }

                                    for preset in TlePreset::ALL.iter() {
                                        if let Some((selected, state, _)) = tle_selections.get(preset) {
                                            if !*selected { continue; }
                                            if let TleLoadState::Loaded { satellites } = state {
                                                for sat in satellites {
                                                    ui.label(&sat.name);
                                                    ui.label(preset.label());
                                                    let alt = mean_motion_to_altitude_km(sat.mean_motion);
                                                    ui.label(format!("{:.0}", alt));
                                                    ui.label(format!("{:.1}", sat.inclination_deg));
                                                    let period = 1440.0 / sat.mean_motion;
                                                    ui.label(format!("{:.1}", period));
                                                    ui.end_row();
                                                }
                                            }
                                        }
                                    }
                                });
                        });
                });
            self.tabs[tab_idx].show_sat_list = show;
        }

        if show_config && !self.show_tab_info {
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
        egui::ScrollArea::horizontal()
            .id_salt(("planet_config_hscroll", tab_idx, planet_idx))
            .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
            .show(ui, |ui| { ui.horizontal(|ui| {
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
                        ui.separator();
                        ui.label("ISL k:");
                        ui.add(egui::DragValue::new(&mut planet.tle_isl_k).range(0..=8).speed(0.1));
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
                                                // Debris presets render with the dedicated bright
                                                // X palette in drawing.rs; mirror that here so the
                                                // legend swatch matches what the user sees on the
                                                // globe.
                                                let color = match preset {
                                                    TlePreset::Fengyun1cDebris => egui::Color32::from_rgb(255, 70, 70),
                                                    TlePreset::Cosmos2251Debris => egui::Color32::from_rgb(70, 230, 90),
                                                    TlePreset::Iridium33Debris => egui::Color32::from_rgb(90, 160, 255),
                                                    TlePreset::Cosmos1408Debris => egui::Color32::from_rgb(255, 220, 70),
                                                    _ => plane_color(preset.color_index()),
                                                };
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
                                                let preset_copy = *preset;
                                                let tx = tle_fetch_tx.clone();
                                                if let Some(owners) = preset.country_owners() {
                                                    std::thread::spawn(move || {
                                                        let result = crate::tle::fetch_tle_by_country(owners);
                                                        let _ = tx.send((preset_copy, result));
                                                    });
                                                } else {
                                                    let url = preset.url().to_string();
                                                    std::thread::spawn(move || {
                                                        let result = fetch_tle_data(&url);
                                                        let _ = tx.send((preset_copy, result));
                                                    });
                                                }
                                            }

                                            #[cfg(target_arch = "wasm32")]
                                            if fetch_requested && *selected && matches!(state, TleLoadState::NotLoaded | TleLoadState::Failed(_)) {
                                                *state = TleLoadState::Loading;
                                                let preset_copy = *preset;
                                                let ctx = ui.ctx().clone();
                                                if let Some(owners) = preset.country_owners() {
                                                    wasm_bindgen_futures::spawn_local(async move {
                                                        let result = crate::tle::fetch_tle_by_country_async(owners).await;
                                                        TLE_FETCH_RESULT.with(|cell| {
                                                            cell.borrow_mut().push((preset_copy, result));
                                                        });
                                                        ctx.request_repaint();
                                                    });
                                                } else {
                                                    let url = preset.url().to_string();
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
                        let phy_btn = if cons.show_physics_ui {
                            egui::Button::new(egui::RichText::new("⚡").color(egui::Color32::WHITE))
                                .fill(egui::Color32::from_rgb(60, 100, 160)).small()
                        } else {
                            egui::Button::new(egui::RichText::new("⚡"))
                                .fill(egui::Color32::from_rgb(80, 80, 80)).small()
                        };
                        if ui.add(phy_btn).on_hover_text("Toggle physics simulation").clicked() {
                            cons.show_physics_ui = !cons.show_physics_ui;
                            if !cons.show_physics_ui {
                                cons.physics.enabled = false;
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

                    ui.horizontal_top(|ui| {
                    ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        let mut sats = cons.sats_per_plane as i32;
                        let mut planes = cons.num_planes as i32;
                        ui.label("Sats:");
                        let sats_resp = ui.add(egui::DragValue::new(&mut sats).range(1..=100)).on_hover_text("Satellites per orbital plane");
                        ui.label("Orbits:");
                        let planes_resp = ui.add(egui::DragValue::new(&mut planes).range(1..=100)).on_hover_text("Number of orbital planes");
                        if sats > 0 && planes > 0 {
                            cons.sats_per_plane = sats as usize;
                            cons.num_planes = planes as usize;
                        }
                        if sats_resp.changed() || planes_resp.changed() {
                            cons.preset = Preset::None;
                        }
                        let orbit_radius = planet.celestial_body.radius_km() + cons.altitude_km;
                        let default_sat_spacing = 2.0 * std::f64::consts::PI * orbit_radius / cons.sats_per_plane as f64;
                        let mut custom_sat_spacing = cons.sat_spacing_km.is_some();
                        if ui.checkbox(&mut custom_sat_spacing, "d:").on_hover_text("Custom satellite spacing within each plane").changed() {
                            cons.sat_spacing_km = if custom_sat_spacing { Some(default_sat_spacing) } else { None };
                            cons.preset = Preset::None;
                        }
                        if let Some(ref mut spacing) = cons.sat_spacing_km {
                            if ui.add(egui::DragValue::new(spacing).range(0.001..=100000.0).suffix(" km").speed(0.1).max_decimals(3)).changed() {
                                cons.preset = Preset::None;
                            }
                        } else {
                            ui.weak(format!("{:.1} km", default_sat_spacing));
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Alt:");
                        let alt_resp = ui.add(egui::DragValue::new(&mut cons.altitude_km).range(0.0..=50000.0).suffix(" km")).on_hover_text("Orbit altitude above surface");
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
                        let cb = planet.celestial_body;
                        let sso_possible = cb.j2() > 0.0 && cb.orbital_period_days().is_some();
                        if sso_possible && cons.sso {
                            if let Some(inc) = ConstellationConfig::sso_inclination(
                                cons.altitude_km, cons.eccentricity,
                                cb.mu(), cb.j2(), cb.radius_km(), cb.equatorial_radius_km(),
                                cb.orbital_period_days().unwrap(),
                            ) {
                                cons.inclination = inc;
                            }
                        }
                        ui.add_enabled(sso_possible, egui::Checkbox::new(&mut cons.sso, "SSO"))
                            .on_hover_text("Sun-synchronous orbit (auto-compute inclination)");
                        ui.label("Inc:");
                        let inc_resp = ui.add_enabled(!cons.sso, egui::DragValue::new(&mut cons.inclination).range(0.0..=180.0).suffix("°")).on_hover_text("Orbital inclination angle");
                        if inc_resp.changed() {
                            cons.preset = Preset::None;
                        }
                        ui.label("F:");
                        let max_f = (cons.num_planes - 1).max(1) as f64;
                        let phase_resp = ui.add(egui::DragValue::new(&mut cons.phasing).range(0.0..=max_f).speed(0.1)).on_hover_text("Walker phasing parameter F");
                        if phase_resp.changed() {
                            cons.preset = Preset::None;
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("RAAN₀:");
                        if ui.add(egui::DragValue::new(&mut cons.raan_offset).range(-180.0..=180.0).suffix("°").speed(1.0)).on_hover_text("RAAN offset of the first plane").changed() {
                            cons.preset = Preset::None;
                        }
                        let default_spacing = match cons.walker_type {
                            WalkerType::Delta => 360.0 / cons.num_planes as f64,
                            WalkerType::Star => 180.0 / cons.num_planes as f64,
                        };
                        let mut custom_spacing = cons.raan_spacing.is_some();
                        if ui.checkbox(&mut custom_spacing, "Δ:").on_hover_text("Custom RAAN spacing between planes").changed() {
                            cons.raan_spacing = if custom_spacing { Some(default_spacing) } else { None };
                            cons.preset = Preset::None;
                        }
                        if let Some(ref mut spacing) = cons.raan_spacing {
                            if ui.add(egui::DragValue::new(spacing).range(0.0001..=180.0).suffix("°").speed(0.01).max_decimals(4)).changed() {
                                cons.preset = Preset::None;
                            }
                        } else {
                            ui.weak(format!("{:.1}°", default_spacing));
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Ecc:");
                        if ui.add(egui::DragValue::new(&mut cons.eccentricity).range(0.0..=0.99).speed(0.001).max_decimals(4)).on_hover_text("Orbital eccentricity").changed() {
                            cons.preset = Preset::None;
                        }
                        ui.label("ω:");
                        if ui.add(egui::DragValue::new(&mut cons.arg_periapsis).range(0.0..=360.0).suffix("°").speed(1.0)).on_hover_text("Argument of periapsis").changed() {
                            cons.preset = Preset::None;
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("ISLs/sat:");
                        ui.selectable_value(&mut cons.isl_neighbors, 0, "Off")
                            .on_hover_text("No inter-satellite links");
                        ui.selectable_value(&mut cons.isl_neighbors, 4, "4")
                            .on_hover_text("4 neighbors (cardinal: up/down/left/right)");
                        ui.selectable_value(&mut cons.isl_neighbors, 8, "8")
                            .on_hover_text("8 neighbors (cardinal + diagonal)");
                    });

                    ui.horizontal(|ui| {
                        ui.label("Propagator:");
                        ui.selectable_value(&mut cons.propagator, Propagator::Keplerian, "Keplerian")
                            .on_hover_text("Two-body Keplerian (RAAN drift only)");
                        ui.selectable_value(&mut cons.propagator, Propagator::J2, "J2")
                            .on_hover_text("J2 secular perturbations (RAAN drift, ω precession, mean motion correction)");
                        ui.selectable_value(&mut cons.propagator, Propagator::Numerical, "RK4")
                            .on_hover_text("Numerical RK4 integration with J2 (per-satellite state, forward-only)");
                        #[cfg(not(target_arch = "wasm32"))]
                        ui.selectable_value(&mut cons.propagator, Propagator::Lib42, "J2 Osc")
                            .on_hover_text("J2 osculating via NASA 42 (secular drifts + periodic SMA corrections)");
                    });

                    ui.horizontal(|ui| {
                        let old_type = cons.walker_type;
                        ui.selectable_value(&mut cons.walker_type, WalkerType::Delta, "Delta")
                            .on_hover_text("Walker-Delta: planes span 360° RAAN");
                        ui.selectable_value(&mut cons.walker_type, WalkerType::Star, "Star")
                            .on_hover_text("Walker-Star: planes span 180° RAAN");
                        if ui.checkbox(&mut cons.drag_enabled, "Drag:").on_hover_text("Enable atmospheric drag decay simulation").changed() {
                            cons.preset = Preset::None;
                        }
                        if cons.drag_enabled {
                            if ui.add(egui::DragValue::new(&mut cons.ballistic_coeff).range(0.1..=500.0).suffix(" kg/m²").speed(1.0).max_decimals(1)).on_hover_text("Ballistic coefficient (mass/drag area)").changed() {
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
                    }); // end left vertical

                    if cons.show_physics_ui {
                        ui.separator();
                        ui.vertical(|ui| {
                    egui::ScrollArea::vertical()
                        .id_salt(format!("phy_scroll_{}_{}_{}", tab_idx, planet_idx, cidx))
                        .max_height(200.0)
                        .show(ui, |ui| {
                    let phy = &mut cons.physics;
                    phy.enabled = true;
                    {
                        ui.horizontal(|ui| {
                            ui.label("Color:");
                            ui.selectable_value(&mut phy.color_mode, crate::physics::PhysicsColorMode::Normal, "Normal")
                                .on_hover_text("Use default plane colors");
                            ui.selectable_value(&mut phy.color_mode, crate::physics::PhysicsColorMode::Battery, "Battery")
                                .on_hover_text("Color by state of charge (green=full, red=empty)");
                            ui.selectable_value(&mut phy.color_mode, crate::physics::PhysicsColorMode::Temperature, "Temp")
                                .on_hover_text("Color by temperature (blue=cold, red=hot)");
                        });
                        ui.checkbox(&mut phy.power_enabled, "Power model")
                            .on_hover_text("Simulate battery charge/discharge with eclipse-gated solar panels");
                        if phy.power_enabled {
                            ui.indent("power_settings", |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Battery:");
                                    ui.add(egui::DragValue::new(&mut phy.max_battery_ws).range(100.0..=1_000_000.0).speed(100.0).suffix(" Ws"))
                                        .on_hover_text("Maximum battery capacity in watt-seconds (joules)");
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Charge rate:");
                                    ui.add(egui::DragValue::new(&mut phy.charging_rate_w).range(0.0..=1000.0).speed(1.0).suffix(" W"))
                                        .on_hover_text("Solar panel or RTG power generation rate");
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Idle power:");
                                    ui.add(egui::DragValue::new(&mut phy.idle_power_w).range(0.0..=500.0).speed(0.5).suffix(" W"))
                                        .on_hover_text("Constant power draw from satellite subsystems");
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Device:");
                                    ui.selectable_value(&mut phy.power_device_type, crate::physics::PowerDeviceType::SolarPanel, "Solar")
                                        .on_hover_text("Solar panels: only charge when not in eclipse");
                                    ui.selectable_value(&mut phy.power_device_type, crate::physics::PowerDeviceType::Rtg, "RTG")
                                        .on_hover_text("Radioisotope thermoelectric generator: charges regardless of eclipse");
                                });
                            });
                        }
                        ui.checkbox(&mut phy.thermal_enabled, "Thermal model")
                            .on_hover_text("Single-node heat balance: solar, albedo, body IR, emission, and activity heat");
                        if phy.thermal_enabled {
                            ui.indent("thermal_settings", |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Mass:");
                                    ui.add(egui::DragValue::new(&mut phy.mass_kg).range(1.0..=10000.0).speed(1.0).suffix(" kg"))
                                        .on_hover_text("Spacecraft mass affecting thermal inertia");
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Thermal cap:");
                                    ui.add(egui::DragValue::new(&mut phy.thermal_capacity).range(100.0..=5000.0).speed(10.0).suffix(" J/kgK"))
                                        .on_hover_text("Specific heat capacity of the spacecraft structure");
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Heat ratio:");
                                    ui.add(egui::DragValue::new(&mut phy.heat_ratio).range(0.0..=1.0).speed(0.01))
                                        .on_hover_text("Fraction of electrical power dissipated as heat (0=none, 1=all)");
                                });
                            });
                        }
                        ui.checkbox(&mut phy.radiation_enabled, "Radiation model")
                            .on_hover_text("Poisson-process radiation events: random restarts and permanent failures");
                        if phy.radiation_enabled {
                            ui.indent("radiation_settings", |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Failure rate:");
                                    ui.add(egui::DragValue::new(&mut phy.failure_rate).range(0.0..=1e-6).speed(1e-12).min_decimals(13))
                                        .on_hover_text("Probability per second of permanent satellite failure (single event latch-up)");
                                    ui.label("/s");
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Restart rate:");
                                    ui.add(egui::DragValue::new(&mut phy.restart_rate).range(0.0..=1e-4).speed(1e-10).min_decimals(11))
                                        .on_hover_text("Probability per second of transient restart event (single event upset)");
                                    ui.label("/s");
                                });
                            });
                        }
                        if !cons.physics_state.is_empty() {
                            let alive = cons.physics_state.iter().filter(|s| !s.is_dead).count();
                            let dead = cons.physics_state.len() - alive;
                            let avg_soc = if alive > 0 {
                                cons.physics_state.iter().filter(|s| !s.is_dead)
                                    .map(|s| s.state_of_charge(&cons.physics)).sum::<f64>() / alive as f64
                            } else { 0.0 };
                            let avg_temp = if alive > 0 {
                                cons.physics_state.iter().filter(|s| !s.is_dead)
                                    .map(|s| s.temperature_k).sum::<f64>() / alive as f64
                            } else { 0.0 };
                            ui.label(format!("Alive: {}  Dead: {}  Avg Battery: {:.0}%  Avg T: {:.0} K",
                                alive, dead, avg_soc * 100.0, avg_temp));
                            if ui.button("Reset physics").on_hover_text("Clear all physics state and reinitialize").clicked() {
                                cons.physics_state.clear();
                            }
                        }
                    }
                    }); // end scroll area
                    }); // end right vertical
                    } // end show_physics_ui
                    }); // end horizontal_top
                });
                ui.separator();
            }

            let add_btn_text = if num_constellations == 0 { "[+] Add constellation" } else { "[+]" };
            if ui.button(add_btn_text).on_hover_text("Add a new constellation to this planet").clicked() {
                const_to_remove = Some(usize::MAX);
            }
        }); });

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

        let hide_sats = self.tabs[tab_idx].settings.zoom > 100.0 && self.tile_overlay.enabled;
        let mut constellations_data: Vec<_> = if hide_sats {
            Vec::new()
        } else {
            planet.constellations.iter()
                .enumerate()
                .filter(|(_, c)| !c.hidden)
                .map(|(orig_idx, c)| {
                    let wc = c.constellation(planet_radius, planet_mu, planet_j2, planet_eq_radius);
                    let pos = if c.propagator == Propagator::Numerical {
                        if let Some(ns) = &c.numerical {
                            numerical_state_to_positions(&wc, ns)
                        } else {
                            wc.satellite_positions(self.tabs[tab_idx].settings.time)
                        }
                    } else {
                        wc.satellite_positions(self.tabs[tab_idx].settings.time)
                    };
                    let name = c.preset_name().to_string();
                    (wc, pos, c.color_offset, 0u8, orig_idx, name)
                })
                .collect()
        };

        // Render any selected TLE preset regardless of whether the TLE sidebar
        // is open — this lets demos include live satellites without forcing
        // the sidebar UI to be visible.
        let any_tle_selected = planet.tle_selections
            .iter()
            .any(|(_, (selected, _, _))| *selected);
        if planet.show_tle_window || any_tle_selected {
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
                        neighbors: Vec::new(),
                        name: Some(sat.name.clone()),
                        tle_inclination_deg: Some(sat.inclination_deg),
                        tle_mean_motion: Some(sat.mean_motion),
                    });
                }
                if all_positions.is_empty() { continue; }
                let tle_isl_k = planet.tle_isl_k;

                if let Some(shells) = shells {
                    let shell_indices: Vec<std::collections::HashSet<usize>> = shells.iter()
                        .map(|s| s.satellite_indices.iter().copied().collect())
                        .collect();
                    for (si, shell) in shells.iter().enumerate() {
                        if !shell.selected { continue; }
                        let mut positions: Vec<SatelliteState> = all_positions.iter()
                            .filter(|p| shell_indices[si].contains(&p.sat_index))
                            .map(|p| SatelliteState {
                                plane: p.plane, sat_index: p.sat_index,
                                x: p.x, y: p.y, z: p.z,
                                lat: p.lat, lon: p.lon,
                                ascending: p.ascending,
                                neighbors: p.neighbors.clone(),
                                name: p.name.clone(),
                                tle_inclination_deg: p.tle_inclination_deg,
                                tle_mean_motion: p.tle_mean_motion,
                            })
                            .collect();
                        if positions.is_empty() { continue; }
                        if tle_isl_k > 0 {
                            crate::walker::compute_knn_neighbors(&mut positions, tle_isl_k);
                        }
                        let tle_wc = WalkerConstellation {
                            walker_type: WalkerType::Delta,
                            total_sats: positions.len(),
                            num_planes: 1,
                            altitude_km: 550.0,
                            inclination_deg: 0.0,
                            phasing: 0.0,
                            raan_offset_deg: 0.0,
                            raan_spacing_deg: None,
                            sat_spacing_km: None,
                            isl_neighbors: 0,
                            propagator: Propagator::Keplerian,
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
                        sat_spacing_km: None,
                        isl_neighbors: 0,
                        propagator: Propagator::Keplerian,
                        eccentricity: 0.0,
                        arg_periapsis_deg: 0.0,
                        planet_radius,
                        planet_mu,
                        planet_j2,
                        planet_equatorial_radius: planet_eq_radius,
                    };
                    if tle_isl_k > 0 {
                        crate::walker::compute_knn_neighbors(&mut all_positions, tle_isl_k);
                    }
                    let tle_kind = if preset.is_debris() { 2u8 } else { 1u8 };
                    constellations_data.push((tle_wc, all_positions, preset.color_index(), tle_kind, usize::MAX, preset.label().to_string()));
                }
            }
        }

        {
            let current_time = self.tabs[tab_idx].settings.time;
            let planet = &self.tabs[tab_idx].planets[planet_idx];
            if planet.kessler.course_correction_enabled && !planet.kessler.active_corrections.is_empty() {
                let mut offset_map: HashMap<String, f64> = HashMap::new();
                for corr in &planet.kessler.active_corrections {
                    let off = corr.offset_at(current_time);
                    if off.abs() > 1e-6 {
                        offset_map.insert(corr.sat_name.clone(), off);
                    }
                }
                if !offset_map.is_empty() {
                    for (_, positions, _, tle_kind, _, label) in constellations_data.iter_mut() {
                        if *tle_kind == 3 { continue; }
                        for sat in positions.iter_mut() {
                            let name = sat.name.clone().unwrap_or_else(|| {
                                format!("{} P{}:S{}", label, sat.plane, sat.sat_index)
                            });
                            if let Some(&offset_km) = offset_map.get(&name) {
                                let r = (sat.x * sat.x + sat.y * sat.y + sat.z * sat.z).sqrt();
                                if r > 1.0 {
                                    let scale = (r + offset_km) / r;
                                    sat.x *= scale;
                                    sat.y *= scale;
                                    sat.z *= scale;
                                    sat.lat = (sat.y / (r + offset_km)).asin().to_degrees();
                                    sat.lon = -sat.z.atan2(sat.x).to_degrees();
                                }
                            }
                        }
                    }
                }
            }
        }

        {
            let planet = &self.tabs[tab_idx].planets[planet_idx];
            if planet.kessler.enabled && !planet.kessler.debris.is_empty() {
                let time = self.tabs[tab_idx].settings.time;
                let mut debris_positions = Vec::with_capacity(planet.kessler.debris.len());
                for (idx, frag) in planet.kessler.debris.iter().enumerate() {
                    let [x, y, z] = crate::kessler::propagate_fragment(frag, time);
                    let r = (x * x + y * y + z * z).sqrt();
                    if r < 1.0 { continue; }
                    debris_positions.push(SatelliteState {
                        plane: 0,
                        sat_index: idx,
                        x, y, z,
                        lat: (y / r).asin().to_degrees(),
                        lon: -z.atan2(x).to_degrees(),
                        ascending: false,
                        neighbors: Vec::new(),
                        name: Some(format!("Debris-{}", idx)),
                        tle_inclination_deg: None,
                        tle_mean_motion: None,
                    });
                }
                if !debris_positions.is_empty() {
                    let dummy_wc = WalkerConstellation {
                        walker_type: WalkerType::Delta,
                        total_sats: debris_positions.len(),
                        num_planes: 1,
                        altitude_km: 550.0,
                        inclination_deg: 0.0,
                        phasing: 0.0,
                        raan_offset_deg: 0.0,
                        raan_spacing_deg: None,
                        sat_spacing_km: None,
                        isl_neighbors: 0,
                        propagator: Propagator::Keplerian,
                        eccentricity: 0.0,
                        arg_periapsis_deg: 0.0,
                        planet_radius,
                        planet_mu,
                        planet_j2,
                        planet_equatorial_radius: planet_eq_radius,
                    };
                    constellations_data.push((dummy_wc, debris_positions, 0, 3, usize::MAX, "Kessler Debris".to_string()));
                }
            }
        }

        {
            use chrono::Datelike;
            let time = self.tabs[tab_idx].settings.time;
            let sim_speed = self.tabs[tab_idx].settings.speed;
            let dt = if self.tabs[tab_idx].settings.animate {
                ui.ctx().input(|i| i.stable_dt) as f64 * sim_speed
            } else {
                0.0
            };
            let timestamp = self.start_timestamp + chrono::Duration::seconds(time as i64);
            let day_of_year = timestamp.ordinal() as f64;
            let decl_rad = (crate::time::SOLAR_DECLINATION_MAX
                * ((360.0 / crate::time::DAYS_PER_YEAR) * (day_of_year + 10.0)).to_radians().cos())
                .to_radians();
            let sun_ra = ((day_of_year - 80.0) * 360.0 / 365.0).to_radians();
            let sun_inertial = Vector3::new(
                decl_rad.cos() * sun_ra.cos(),
                decl_rad.sin(),
                -decl_rad.cos() * sun_ra.sin(),
            ).normalize();
            let frame_seed = (time * 1000.0) as u64;
            let planet = &mut self.tabs[tab_idx].planets[planet_idx];
            for (_, positions, _, _, orig_idx, _) in &constellations_data {
                if *orig_idx == usize::MAX { continue; }
                let constellation = &mut planet.constellations[*orig_idx];
                if !constellation.physics.enabled { continue; }
                let total = constellation.sats_per_plane * constellation.num_planes;
                if constellation.physics_state.len() != total {
                    constellation.physics_state = (0..total)
                        .map(|_| crate::physics::SatellitePhysics::new(&constellation.physics))
                        .collect();
                }
                for (si, sat) in positions.iter().enumerate() {
                    if si >= constellation.physics_state.len() { break; }
                    let sat_pos = Vector3::new(sat.x, sat.y, sat.z);
                    let alt = (sat_pos.norm() - planet_radius).max(0.0);
                    let seed = frame_seed.wrapping_add(*orig_idx as u64 * 10000 + si as u64);
                    crate::physics::update_satellite(
                        &mut constellation.physics_state[si],
                        &constellation.physics,
                        dt,
                        &sat_pos,
                        &sun_inertial,
                        planet_radius,
                        alt,
                        seed,
                    );
                }
            }
        }

        {
            let sim_speed = self.tabs[tab_idx].settings.speed;
            let current_time = self.tabs[tab_idx].settings.time;
            let start_ts = self.start_timestamp.timestamp() as f64 / 60.0;
            let planet = &mut self.tabs[tab_idx].planets[planet_idx];
            let need_conj = planet.show_conjunction_window
                || planet.show_conjunction_lines
                || planet.conjunction_cache.show_heatmap
                || planet.kessler.enabled;
            if need_conj {
                let sim_dt = ui.ctx().input(|i| i.stable_dt) as f64 * sim_speed;
                let threshold = planet.conjunction_cache.threshold_km;
                let kessler_threshold = planet.kessler.collision_threshold_km;
                let detect_radius = if planet.conjunction_cache.show_heatmap {
                    threshold * 5.0
                } else if planet.kessler.enabled {
                    threshold.max(kessler_threshold)
                } else {
                    threshold
                };
                crate::conjunction::compute_conjunctions(
                    &mut planet.conjunction_cache.conjunctions,
                    detect_radius,
                    &constellations_data,
                    &mut planet.conjunction_prev_positions,
                    sim_dt,
                );

                if planet.kessler.enabled {
                    let coll_thresh = planet.kessler.collision_threshold_km;
                    let n_frags = planet.kessler.fragments_per_collision;
                    let max_debris = planet.kessler.max_debris;
                    let max_per_frame = (sim_dt * 0.2).max(1.0).min(20.0) as usize;
                    let mut already_hit: HashSet<String> = planet.kessler.collided_pairs.iter()
                        .flat_map(|(a, b)| [a.clone(), b.clone()])
                        .collect();
                    let candidates: Vec<_> = planet.conjunction_cache.conjunctions.iter()
                        .filter(|c| c.distance_km < coll_thresh)
                        .filter(|c| !(c.source_a == "Kessler Debris" && c.source_b == "Kessler Debris"))
                        .map(|c| (c.pos_a, c.pos_b, c.name_a.clone(), c.name_b.clone()))
                        .collect();
                    let mut collisions_this_frame = 0usize;
                    for (pos_a, pos_b, name_a, name_b) in candidates {
                        if collisions_this_frame >= max_per_frame { break; }
                        // One object can only be destroyed once — prevents a single
                        // physical collision event from being counted many times when
                        // several pairs fall within the detection threshold.
                        if already_hit.contains(&name_a) || already_hit.contains(&name_b) {
                            continue;
                        }
                        let key = if name_a < name_b {
                            (name_a.clone(), name_b.clone())
                        } else {
                            (name_b.clone(), name_a.clone())
                        };
                        already_hit.insert(name_a);
                        already_hit.insert(name_b);
                        planet.kessler.collided_pairs.insert(key);
                        planet.kessler.collision_count += 1;
                        planet.kessler.collision_id_counter += 1;
                        collisions_this_frame += 1;
                        if planet.kessler.debris.len() < max_debris {
                            let capped_frags = n_frags.min(15);
                            let new_debris = crate::kessler::generate_collision_debris(
                                pos_a, pos_b,
                                planet_mu, planet_radius,
                                current_time,
                                capped_frags,
                                planet.kessler.collision_id_counter,
                            );
                            let remaining = max_debris - planet.kessler.debris.len();
                            planet.kessler.debris.extend(new_debris.into_iter().take(remaining));
                        }
                    }
                }

                let recompute_interval = 2.0 * sim_speed.abs().max(1.0);
                if planet.show_conjunction_window
                    && (current_time - planet.conjunction_cache.last_prediction_time).abs() > recompute_interval
                {
                    let walker_data: Vec<_> = planet.constellations.iter()
                        .filter(|c| !c.hidden)
                        .map(|c| {
                            let wc = c.constellation(planet_radius, planet_mu, planet_j2, planet_eq_radius);
                            (wc, c.preset_name().to_string())
                        })
                        .collect();

                    let propagation_minutes = start_ts + current_time / 60.0;
                    let tle_groups: Vec<crate::conjunction::TleGroup> = crate::tle::TlePreset::ALL.iter()
                        .filter_map(|preset| {
                            let (selected, state, _) = planet.tle_selections.get(preset)?;
                            if !*selected { return None; }
                            if let crate::tle::TleLoadState::Loaded { satellites } = state {
                                Some(crate::conjunction::TleGroup {
                                    label: preset.label().to_string(),
                                    satellites: satellites.as_slice(),
                                    propagation_minutes,
                                })
                            } else {
                                None
                            }
                        })
                        .collect();

                    let window_sec = planet.conjunction_cache.prediction_window_min * 60.0;
                    planet.conjunction_cache.predictions = crate::conjunction::predict_conjunctions(
                        &walker_data,
                        &tle_groups,
                        current_time,
                        threshold,
                        window_sec,
                    );
                    planet.conjunction_cache.last_prediction_time = current_time;

                    if planet.kessler.course_correction_enabled {
                        let corr_alt = planet.kessler.correction_altitude_km;
                        let existing: HashSet<String> = planet.kessler.active_corrections.iter()
                            .map(|c| c.sat_name.clone())
                            .collect();
                        for pred in &planet.conjunction_cache.predictions {
                            let is_a_debris = pred.name_a.starts_with("Debris-")
                                || pred.source_a == "Kessler Debris";
                            let is_b_debris = pred.name_b.starts_with("Debris-")
                                || pred.source_b == "Kessler Debris";
                            let target = if is_a_debris && !is_b_debris {
                                Some(&pred.name_b)
                            } else if !is_a_debris && is_b_debris {
                                Some(&pred.name_a)
                            } else if !is_a_debris && !is_b_debris {
                                Some(&pred.name_a)
                            } else {
                                None
                            };
                            if let Some(name) = target {
                                if !existing.contains(name) {
                                    let tca_time = current_time + pred.time_until;
                                    let lead = pred.time_until.max(30.0);
                                    planet.kessler.active_corrections.push(
                                        crate::config::CourseCorrection {
                                            sat_name: name.clone(),
                                            start_time: tca_time - lead,
                                            end_time: tca_time + lead,
                                            altitude_offset_km: corr_alt,
                                        },
                                    );
                                    planet.kessler.corrections_made += 1;
                                }
                            }
                        }
                    }
                }

                planet.kessler.active_corrections.retain(|c| c.end_time > current_time);
            }
        }

        {
            let planet = &mut self.tabs[tab_idx].planets[planet_idx];
            let mut show_conj_window = planet.show_conjunction_window;
            let conj_cache = &mut planet.conjunction_cache;
            let kessler = &mut planet.kessler;
            let mut show_lines = planet.show_conjunction_lines;
            egui::Window::new(format!("Conjunctions - {}", planet_name))
                .open(&mut show_conj_window)
                .default_size([500.0, 500.0])
                .show(ui.ctx(), |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Threshold:");
                        ui.add(
                            egui::DragValue::new(&mut conj_cache.threshold_km)
                                .range(1.0..=500.0)
                                .speed(1.0)
                                .suffix(" km"),
                        ).on_hover_text("Distance threshold for conjunction alerts");
                        ui.checkbox(&mut show_lines, "Lines")
                            .on_hover_text("Draw lines between conjuncting objects");
                        ui.checkbox(&mut conj_cache.show_heatmap, "Heatmap")
                            .on_hover_text("Show conjunction risk as a heatmap");
                    });
                    ui.separator();
                    egui::ScrollArea::vertical()
                        .id_salt("conj_scroll")
                        .min_scrolled_height(400.0)
                        .max_height(400.0)
                        .show(ui, |ui| {
                            ui.strong("Kessler Simulation");
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut kessler.enabled, "Enable")
                                    .on_hover_text("Simulate cascading debris collisions");
                                if ui.button("Clear").on_hover_text("Remove all debris and reset counters").clicked() {
                                    kessler.debris.clear();
                                    kessler.collision_count = 0;
                                    kessler.collision_id_counter = 0;
                                    kessler.collided_pairs.clear();
                                }
                            });
                            if kessler.enabled {
                                ui.horizontal(|ui| {
                                    ui.label("Collision dist:");
                                    ui.add(
                                        egui::DragValue::new(&mut kessler.collision_threshold_km)
                                            .range(0.1..=50.0)
                                            .speed(0.1)
                                            .suffix(" km"),
                                    ).on_hover_text("Minimum distance to trigger a collision");
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Fragments/collision:");
                                    ui.add(
                                        egui::DragValue::new(&mut kessler.fragments_per_collision)
                                            .range(2..=50)
                                            .speed(1),
                                    ).on_hover_text("Debris fragments created per collision");
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Max debris:");
                                    ui.add(
                                        egui::DragValue::new(&mut kessler.max_debris)
                                            .range(100..=50000)
                                            .speed(100),
                                    ).on_hover_text("Maximum debris objects to simulate");
                                });
                                ui.label(format!(
                                    "Collisions: {}  Debris: {}",
                                    kessler.collision_count,
                                    kessler.debris.len()
                                ));

                                ui.separator();
                                ui.strong("Course Correction");
                                ui.checkbox(&mut kessler.course_correction_enabled, "Enable")
                                    .on_hover_text("Allow satellites to maneuver to avoid debris");
                                if kessler.course_correction_enabled {
                                    ui.horizontal(|ui| {
                                        ui.label("Maneuver altitude:");
                                        ui.add(
                                            egui::DragValue::new(&mut kessler.correction_altitude_km)
                                                .range(0.5..=50.0)
                                                .speed(0.1)
                                                .suffix(" km"),
                                        ).on_hover_text("Altitude change for collision avoidance maneuver");
                                    });
                                    ui.label(format!(
                                        "Corrections: {}  Active: {}",
                                        kessler.corrections_made,
                                        kessler.active_corrections.len()
                                    ));
                                }
                            }

                            let threshold = conj_cache.threshold_km;
                            let current: Vec<_> = conj_cache.conjunctions.iter()
                                .filter(|c| c.distance_km <= threshold)
                                .take(10)
                                .collect();
                            if !current.is_empty() {
                                ui.separator();
                                ui.strong("Current");
                                egui::Grid::new("conj_grid")
                                    .striped(true)
                                    .show(ui, |ui| {
                                        ui.strong("Dist");
                                        ui.strong("Object A");
                                        ui.strong("");
                                        ui.strong("Object B");
                                        ui.strong("TCA");
                                        ui.strong("Min Dist");
                                        ui.end_row();
                                        for conj in &current {
                                            let urgency = 1.0 - (conj.distance_km / threshold).clamp(0.0, 1.0);
                                            let r = (255.0 * urgency) as u8;
                                            let g = (255.0 * (1.0 - urgency)) as u8;
                                            let color = egui::Color32::from_rgb(r, g, 0);
                                            ui.colored_label(color, format!("{:.1} km", conj.distance_km));
                                            ui.label(format!("{} ({})", conj.name_a, conj.source_a));
                                            ui.label("↔");
                                            ui.label(format!("{} ({})", conj.name_b, conj.source_b));
                                            if conj.tca_seconds.abs() < 1.0 {
                                                ui.colored_label(egui::Color32::RED, "NOW");
                                            } else if conj.tca_seconds > 0.0 {
                                                ui.label(format!("{:.0}s", conj.tca_seconds));
                                            } else {
                                                ui.label(format!("{:.0}s ago", -conj.tca_seconds));
                                            }
                                            ui.label(format!("{:.1} km", conj.min_distance_km));
                                            ui.end_row();
                                        }
                                    });
                            }

                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.strong("Predicted");
                                ui.add(
                                    egui::DragValue::new(&mut conj_cache.prediction_window_min)
                                        .range(1.0..=60.0)
                                        .speed(1.0)
                                        .suffix(" min"),
                                ).on_hover_text("Look-ahead window for conjunction predictions");
                            });
                            if conj_cache.predictions.is_empty() {
                                ui.weak("No upcoming conjunctions predicted");
                            } else {
                                egui::Grid::new("conj_pred_grid")
                                    .striped(true)
                                    .show(ui, |ui| {
                                        ui.strong("In");
                                        ui.strong("Object A");
                                        ui.strong("");
                                        ui.strong("Object B");
                                        ui.strong("Min Dist");
                                        ui.end_row();
                                        for pred in conj_cache.predictions.iter().take(10) {
                                            let secs = pred.time_until;
                                            if secs < 60.0 {
                                                ui.colored_label(egui::Color32::RED, format!("{:.0}s", secs));
                                            } else {
                                                ui.label(format!("{:.0}m {:.0}s", (secs / 60.0).floor(), secs % 60.0));
                                            }
                                            ui.label(format!("{} ({})", pred.name_a, pred.source_a));
                                            ui.label("↔");
                                            ui.label(format!("{} ({})", pred.name_b, pred.source_b));
                                            ui.label(format!("{:.1} km", pred.min_distance_km));
                                            ui.end_row();
                                        }
                                    });
                            }
                        });
                });
            planet.show_conjunction_window = show_conj_window;
            planet.show_conjunction_lines = show_lines;
        }

        {
            let planet = &mut self.tabs[tab_idx].planets[planet_idx];
            let mut show_rad_window = planet.show_radiation_window;
            let rad = &mut planet.radiation;
            egui::Window::new(format!("Radiation - {}", planet_name))
                .open(&mut show_rad_window)
                .default_width(400.0)
                .show(ui.ctx(), |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Kp index:");
                        ui.add(
                            egui::DragValue::new(&mut rad.kp_index)
                                .range(0.0..=9.0)
                                .speed(0.1)
                                .max_decimals(1),
                        ).on_hover_text("Geomagnetic activity index (0=quiet, 9=storm)");
                        let kp_label = if rad.kp_index < 4.0 {
                            "Quiet"
                        } else if rad.kp_index < 6.0 {
                            "Active"
                        } else {
                            "Storm"
                        };
                        let kp_color = if rad.kp_index < 4.0 {
                            egui::Color32::GREEN
                        } else if rad.kp_index < 6.0 {
                            egui::Color32::YELLOW
                        } else {
                            egui::Color32::RED
                        };
                        ui.colored_label(kp_color, kp_label);
                    });
                    ui.checkbox(&mut rad.show_belts, "Show belt bands")
                        .on_hover_text("Draw Van Allen belt band outlines");
                    ui.checkbox(&mut rad.show_lines, "Show lines")
                        .on_hover_text("Draw field lines through the belts");
                    ui.checkbox(&mut rad.show_dots, "Show dots")
                        .on_hover_text("Show sample points along field lines");
                    ui.checkbox(&mut rad.connect_along_shell, "Connect along shell")
                        .on_hover_text("Connect points within the same drift shell");
                    ui.checkbox(&mut rad.connect_across_shells, "Connect across shells")
                        .on_hover_text("Connect points between adjacent drift shells");
                    ui.checkbox(&mut rad.show_fill, "Show fill")
                        .on_hover_text("Fill the belt regions with color");
                    ui.horizontal(|ui| {
                        ui.label("Dots per line:");
                        ui.add(egui::DragValue::new(&mut rad.dots_per_line).range(2..=100).speed(0.5))
                            .on_hover_text("Sample points per field line");
                    });
                    ui.checkbox(&mut rad.show_magnetopause, "Show magnetopause")
                        .on_hover_text("Draw the magnetopause boundary");
                    ui.checkbox(&mut rad.show_sat_exposure, "Satellite exposure coloring")
                        .on_hover_text("Color satellites by radiation dose");

                    ui.separator();
                    ui.strong("Heatmap Sphere");
                    ui.checkbox(&mut rad.show_heatmap_sphere, "Show heatmap sphere")
                        .on_hover_text("Render radiation data on a spherical surface");
                    ui.horizontal(|ui| {
                        ui.label("Altitude (km):");
                        ui.add(egui::DragValue::new(&mut rad.heatmap_altitude_km).range(0.0..=50000.0).speed(50.0).max_decimals(0))
                            .on_hover_text("Altitude of the heatmap sphere");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Resolution:");
                        ui.add(egui::DragValue::new(&mut rad.heatmap_resolution).range(12..=120).speed(0.5))
                            .on_hover_text("Grid resolution for heatmap rendering");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Mode:");
                        use crate::config::HeatmapMode;
                        ui.selectable_value(&mut rad.heatmap_mode, HeatmapMode::Radiation, "Radiation")
                            .on_hover_text("Simple dipole radiation model");
                        ui.selectable_value(&mut rad.heatmap_mode, HeatmapMode::IgrfRadiation, "IGRF Rad")
                            .on_hover_text("AE-8/AP-8 trapped particle model with IGRF field");
                        ui.selectable_value(&mut rad.heatmap_mode, HeatmapMode::FieldStrength, "Dipole (nT)")
                            .on_hover_text("Magnetic field strength from tilted dipole");
                        ui.selectable_value(&mut rad.heatmap_mode, HeatmapMode::IgrfField, "IGRF-14 (nT)")
                            .on_hover_text("Magnetic field strength from IGRF-14 model");
                    });
                    if rad.heatmap_mode == crate::config::HeatmapMode::IgrfRadiation {
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut rad.show_protons, "Protons")
                                .on_hover_text("Show trapped proton flux");
                            ui.checkbox(&mut rad.show_electrons, "Electrons")
                                .on_hover_text("Show trapped electron flux");
                        });
                    }
                    ui.checkbox(&mut rad.smooth_colors, "Smooth colors")
                        .on_hover_text("Interpolate colors smoothly across the heatmap");

                    ui.separator();
                    ui.strong("Belt Rendering");
                    ui.horizontal(|ui| {
                        ui.label("Drift shells:");
                        ui.add(egui::DragValue::new(&mut rad.num_shells).range(2..=100).speed(0.5))
                            .on_hover_text("Number of L-shells to render");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Meridians:");
                        ui.add(egui::DragValue::new(&mut rad.num_meridians).range(2..=64).speed(0.5))
                            .on_hover_text("Number of meridian slices per shell");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Shell phasing:");
                        ui.add(egui::DragValue::new(&mut rad.shell_phasing).range(0.0..=2.0).speed(0.05).max_decimals(2))
                            .on_hover_text("Rotational offset between adjacent shells");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Links:");
                        ui.add(egui::DragValue::new(&mut rad.num_links).range(0..=20).speed(0.5))
                            .on_hover_text("Cross-shell connection lines to draw");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Dipole offset (km):");
                        ui.add(egui::DragValue::new(&mut rad.dipole_offset_km).range(0.0..=2000.0).speed(10.0).max_decimals(0))
                            .on_hover_text("Offset of magnetic dipole from planet center");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Dipole tilt (°):");
                        ui.add(egui::DragValue::new(&mut rad.dipole_tilt).range(0.0..=90.0).speed(0.5).max_decimals(1))
                            .on_hover_text("Tilt angle of the magnetic dipole axis");
                    });

                    ui.separator();
                    ui.strong("Radiation Profile");

                    let kp = rad.kp_index;
                    let profile_line: Vec<[f64; 2]> = (0..200)
                        .map(|i| {
                            let alt = i as f64 * 300.0;
                            let intensity = crate::radiation::belt_profile(alt, kp);
                            [alt, intensity]
                        })
                        .collect();
                    let inner_color = egui::Color32::from_rgb(255, 120, 50);
                    let outer_color = egui::Color32::from_rgb(100, 130, 230);
                    let inner_line: Vec<[f64; 2]> = profile_line.iter()
                        .filter(|p| p[0] < 12000.0)
                        .copied()
                        .collect();
                    let outer_line: Vec<[f64; 2]> = profile_line.iter()
                        .filter(|p| p[0] >= 6000.0)
                        .copied()
                        .collect();
                    egui_plot::Plot::new("rad_profile")
                        .height(150.0)
                        .allow_drag(false)
                        .allow_zoom(false)
                        .allow_scroll(false)
                        .x_axis_label("Altitude (km)")
                        .y_axis_label("Intensity")
                        .show(ui, |plot_ui| {
                            plot_ui.line(
                                egui_plot::Line::new("Inner belt", egui_plot::PlotPoints::new(inner_line))
                                    .color(inner_color)
                                    .width(2.0),
                            );
                            plot_ui.line(
                                egui_plot::Line::new("Outer belt", egui_plot::PlotPoints::new(outer_line))
                                    .color(outer_color)
                                    .width(2.0),
                            );
                        });

                    ui.separator();
                    let r0 = 11.0 - 0.8 * kp;
                    ui.label(format!(
                        "Magnetopause standoff: {:.1} Re ({:.0} km)",
                        r0,
                        r0 * 6371.0
                    ));
                    ui.label(format!(
                        "Inner belt peak: ~3,200 km ({:.1} Re)",
                        (6371.0 + 3200.0) / 6371.0
                    ));
                    ui.label(format!(
                        "Outer belt peak: ~22,300 km ({:.1} Re)",
                        (6371.0 + 22300.0) / 6371.0
                    ));
                });
            planet.show_radiation_window = show_rad_window;
        }

        {
            let tab = &mut self.tabs[tab_idx];
            let planet = &mut tab.planets[planet_idx];
            let body = planet.celestial_body;
            let moons_list = body.moons();
            let mut show_moons_window = planet.show_moons_window;
            if !moons_list.is_empty() && show_moons_window {
                let settings = &mut tab.settings;
                egui::Window::new(format!("Moons - {}", planet_name))
                    .open(&mut show_moons_window)
                    .default_width(200.0)
                    .show(ui.ctx(), |ui| {
                        ui.horizontal(|ui| {
                            if ui.button("All").on_hover_text("Enable all moons").clicked() {
                                for &(moon_body, _, _, _) in moons_list {
                                    planet.enabled_moons.insert(moon_body);
                                }
                            }
                            if ui.button("None").on_hover_text("Disable all moons").clicked() {
                                planet.enabled_moons.clear();
                            }
                        });
                        ui.separator();
                        for &(moon_body, orbit_km, period_days, incl_rad) in moons_list {
                            let mut on = planet.enabled_moons.contains(&moon_body);
                            ui.horizontal(|ui| {
                                if ui.checkbox(&mut on, moon_body.label()).changed() {
                                    if on {
                                        planet.enabled_moons.insert(moon_body);
                                    } else {
                                        planet.enabled_moons.remove(&moon_body);
                                    }
                                }
                                if ui.button("View Orbit").on_hover_text("Zoom to show this moon's orbit").clicked() {
                                    settings.zoom = 10000.0 / (orbit_km * 1.3);
                                    let ci = incl_rad.cos();
                                    let si = incl_rad.sin();
                                    settings.rotation = nalgebra::Matrix3::new(
                                        1.0, 0.0, 0.0,
                                        0.0, ci, si,
                                        0.0, -si, ci,
                                    );
                                    planet.enabled_moons.insert(moon_body);
                                }
                                if ui.button("View Moon").on_hover_text("Zoom to the moon's current position").clicked() {
                                    let time = settings.time;
                                    let angle = 2.0 * PI * time / (period_days * 86400.0);
                                    let x = orbit_km * angle.cos();
                                    let y_orbit = orbit_km * angle.sin();
                                    let y = y_orbit * incl_rad.cos();
                                    let z = y_orbit * incl_rad.sin();
                                    let dist = (x * x + y * y + z * z).sqrt();
                                    let moon_r = moon_body.radius_km();
                                    settings.zoom = 10000.0 / (dist + moon_r * 5.0);
                                    let lat = (y / dist).asin();
                                    let lon = z.atan2(x);
                                    settings.rotation = crate::math::lat_lon_to_matrix(lat, lon);
                                    planet.enabled_moons.insert(moon_body);
                                }
                            });
                        }
                        ui.separator();
                        ui.checkbox(&mut settings.show_moon_orbits, "Show orbits")
                            .on_hover_text("Draw orbital paths for moons");
                        ui.checkbox(&mut settings.show_moon_lines, "Show lines to planet")
                            .on_hover_text("Draw lines from moons to the planet center");
                        ui.checkbox(&mut settings.show_moon_labels, "Show labels")
                            .on_hover_text("Display moon names");
                        ui.separator();
                        let mut has_override = planet.moon_inclination_override.is_some();
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut has_override, "Override inclination");
                            if has_override {
                                let val = planet.moon_inclination_override.get_or_insert(90.0);
                                ui.add(egui::DragValue::new(val).range(0.0..=180.0).speed(0.5).suffix("°"));
                            }
                        });
                        if !has_override {
                            planet.moon_inclination_override = None;
                        }
                    });
            }
            planet.show_moons_window = show_moons_window;
        }

        let mut available = ui.available_size();
        #[cfg(not(target_arch = "wasm32"))]
        {
            available.x = constrained_width.min(available.x);
            let clip = ui.clip_rect();
            let cursor = ui.cursor().min;
            available.x = available.x.min((clip.max.x - cursor.x).max(0.0));
            available.y = available.y.min((clip.max.y - cursor.y).max(0.0));
        }
        let settings = &self.tabs[tab_idx].settings;
        let view_mode = settings.view_mode;
        let render_planet = view_mode == crate::config::ViewMode::Planet;
        let show_torus = settings.show_torus && render_planet;
        // Projection can be overridden per-planet; fall back to the tab-wide
        // setting when no override is set.
        let planet_projection = self.tabs[tab_idx].planets[planet_idx]
            .projection_override
            .unwrap_or(settings.planet_projection);
        let show_solar_system = view_mode == crate::config::ViewMode::SolarSystem;
        let show_planet_sizes = view_mode == crate::config::ViewMode::PlanetSizes;
        let show_orbits = settings.show_orbits;
        let show_axes = settings.show_axes;
        let show_magnetic_axis = settings.show_magnetic_axis;
        let show_coverage = settings.show_coverage;
        let coverage_angle = settings.coverage_angle;
        let time = settings.time;
        let rotation = {
            let base = settings.rotation;
            let roll = settings.camera_roll.to_radians();
            if roll.abs() < 1e-9 {
                base
            } else {
                let c = roll.cos();
                let s = roll.sin();
                let roll_mat = nalgebra::Matrix3::new(
                    c, -s, 0.0,
                    s,  c, 0.0,
                    0.0, 0.0, 1.0,
                );
                roll_mat * base
            }
        };
        let zoom = settings.zoom;
        let earth_fixed_camera = settings.earth_fixed_camera;
        let sun_fixed_camera = settings.sun_fixed_camera;
        let body_rot_angle = body_rotation_angle(celestial_body, time, self.current_gmst);
        let cos_a = body_rot_angle.cos();
        let sin_a = body_rot_angle.sin();
        let body_y_rotation = Matrix3::new(
            cos_a, 0.0, sin_a,
            0.0, 1.0, 0.0,
            -sin_a, 0.0, cos_a,
        );
        // Sun-fixing rotation: cancels the Sun's RA drift around +y. This is
        // exactly what SSO locks onto — the RAAN–Sun_RA relationship stays
        // constant, so the orbit plane appears stationary in this frame.
        //
        // We intentionally DON'T cancel the Sun's declination component.
        // The Sun traces the ecliptic, moving ±23° in declination each year,
        // and that motion is shared by the Sun and the terminator but NOT by
        // a Keplerian orbit plane. Applying an R_z(-decl) correction would pin
        // the Sun perfectly but would wobble every other inertial vector
        // (including the SSO orbit) ~23° out of plane. The physical truth is
        // that SSO locks RA, not the 3D sun angle — so what the user sees
        // ("Sun slowly bobs up/down over the year, orbit stays fixed") is the
        // correct visualization.
        //
        // Uses `continuous_day_of_year` (not `ordinal()`) so sun_ra stays
        // exactly linear in sim time, matching the linear RAAN drift in walker.rs.
        let sun_y_rotation = if sun_fixed_camera {
            let day_of_year = continuous_day_of_year(self.start_timestamp, time);
            let sun_ra = ((day_of_year - 80.0) * 360.0 / DAYS_PER_YEAR).to_radians();
            let cy = sun_ra.cos();
            let sy = sun_ra.sin();
            Matrix3::new(
                cy, 0.0, -sy,
                0.0, 1.0, 0.0,
                sy, 0.0, cy,
            )
        } else {
            Matrix3::identity()
        };
        let rotation = rotation * sun_y_rotation;
        let satellite_rotation = if earth_fixed_camera && !sun_fixed_camera {
            rotation * body_y_rotation.transpose()
        } else {
            rotation
        };
        let sat_radius = settings.sat_radius;
        let show_links = settings.show_links;
        let hide_behind_earth = render_planet && settings.hide_behind_earth;
        let single_color = settings.single_color || constellations_data.len() > 1;
        let dark_mode = self.dark_mode;
        let show_routing_paths = settings.show_routing_paths;
        let show_manhattan_path = settings.show_manhattan_path;
        let show_shortest_path = settings.show_shortest_path;
        let show_radiation_path = settings.show_radiation_path;
        let radiation_weight = settings.radiation_weight;
        let routing_width = settings.routing_width;
        let routing_node_scale = settings.routing_node_scale;
        let show_asc_desc_colors = settings.show_asc_desc_colors;
        let color_ascending = settings.color_ascending;
        let color_descending = settings.color_descending;
        let color_links = settings.color_links;
        let show_sat_labels = settings.show_sat_labels;
        let show_altitude_lines = settings.show_altitude_lines;
        let altitude_line_width = settings.altitude_line_width;
        let show_inclination_bounds = settings.show_inclination_bounds;
        let show_ground_tracks = settings.show_ground_tracks;
        let tex_res = self.texture_resolution;
        let planet_handle = self.planet_image_handles.get(&(celestial_body, skin, tex_res));
        let torus_rotation = {
            let base = settings.rotation;
            let roll = settings.camera_roll.to_radians();
            if roll.abs() < 1e-9 {
                base
            } else {
                let c = roll.cos();
                let s = roll.sin();
                let roll_mat = nalgebra::Matrix3::new(
                    c, -s, 0.0,
                    s,  c, 0.0,
                    0.0, 0.0, 1.0,
                );
                roll_mat * base
            }
        };
        let torus_zoom = self.torus_zoom;
        let link_width = settings.link_width;
        let fixed_sizes = settings.fixed_sizes;
        let show_sat_border = settings.show_sat_border;
        let flattening = celestial_body.flattening();
        let show_polar_circle = settings.show_polar_circle;
        let show_equator = settings.show_equator;
        let show_graticule = settings.show_graticule;
        let show_crosshairs = settings.show_crosshairs;
        let show_day_night = settings.show_day_night;
        let show_city_lights = settings.show_city_lights;
        let show_terminator = settings.show_terminator;
        let show_eclipse = settings.show_eclipse;
        let show_sun = settings.show_sun;
        let show_clouds = settings.show_clouds;
        let show_stars = settings.show_stars;
        let show_devices = settings.show_devices;
        let show_borders = settings.show_borders;
        let show_cities = settings.show_cities;
        let show_radiation_belts = settings.show_radiation_belts;
        let trackpad_rotate = settings.trackpad_rotate;
        let north_up = settings.north_up;
        let enabled_moons = self.tabs[tab_idx].planets[planet_idx].enabled_moons.clone();
        let moon_inclination_override = self.tabs[tab_idx].planets[planet_idx].moon_inclination_override;
        let show_moon_orbits = settings.show_moon_orbits;
        let show_moon_lines = settings.show_moon_lines;
        let show_moon_labels = settings.show_moon_labels;
        let moon_camera_distance_km = settings.moon_camera_distance_km;
        let tle_monochrome = settings.tle_monochrome;

        let is_2d_projection = planet_projection != crate::projection::ProjectionKind::Orthographic;
        let log_power = settings.solar_system_log_power;
        let detail_bounds = self.tile_overlay_detail_bounds(celestial_body);
        let gpu_available = self.render_state.is_some();

        let num_views = [render_planet, show_torus, show_solar_system, show_planet_sizes]
            .iter().filter(|v| **v).count();

        if num_views > 0 {
            let view_height = available.y - 20.0;

            let view_width = available.x / num_views as f32;
            self.view_width = view_width;
            self.view_height = view_height;

            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                if render_planet && !is_2d_projection {
                    ui.vertical(|ui| {
                        let planet = &mut self.tabs[tab_idx].planets[planet_idx];
                        // Accumulate ground-track samples for any camera-tracked satellite.
                        // We record (geographic_lat, geographic_lon, sim_time) points as the
                        // sub-satellite point moves over the Earth-fixed frame. Points
                        // older than the orbit period × 6 are trimmed to keep history bounded.
                        if show_ground_tracks {
                            const GT_MIN_DT: f64 = 10.0;  // seconds of sim time between samples
                            const GT_MAX_POINTS: usize = 20000;
                            let tracked: Vec<(usize, usize, usize)> = planet.satellite_cameras
                                .iter()
                                .filter(|c| c.constellation_idx != usize::MAX)
                                .map(|c| (c.constellation_idx, c.plane, c.sat_index))
                                .collect();
                            for (cons_idx, plane, sat_index) in tracked {
                                let Some(cons) = planet.constellations.get(cons_idx) else { continue; };
                                let wc = cons.constellation(planet_radius, planet_mu, planet_j2, planet_eq_radius);
                                let positions = wc.satellite_positions(time);
                                let Some(sat) = positions.iter().find(|s| s.plane == plane && s.sat_index == sat_index) else { continue; };
                                let r = (sat.x * sat.x + sat.y * sat.y + sat.z * sat.z).sqrt();
                                if r < 1e-6 { continue; }
                                let lat = (sat.y / r).asin().to_degrees();
                                let bx = sat.x * cos_a - sat.z * sin_a;
                                let bz = sat.x * sin_a + sat.z * cos_a;
                                let lon = (-bz).atan2(bx).to_degrees();
                                let key = (cons_idx, plane, sat_index);
                                let entry = planet.ground_track_history.entry(key).or_default();
                                let should_push = entry.last()
                                    .map(|&(_, _, t)| (time - t).abs() >= GT_MIN_DT)
                                    .unwrap_or(true);
                                if should_push {
                                    entry.push((lat, lon, time));
                                    if entry.len() > GT_MAX_POINTS {
                                        entry.drain(..entry.len() - GT_MAX_POINTS);
                                    }
                                }
                            }
                        } else {
                            planet.ground_track_history.clear();
                        }
                        let view_flags = View3DFlags {
                            show_orbits, show_axes, show_magnetic_axis, show_coverage, show_links,
                            hide_behind_earth, single_color, dark_mode, show_routing_paths,
                            show_manhattan_path, show_shortest_path, show_radiation_path, radiation_weight,
                            routing_width, routing_node_scale,
                            show_asc_desc_colors,
                            color_ascending,
                            color_descending,
                            color_links,
                            show_sat_labels,
                            show_altitude_lines, altitude_line_width, show_inclination_bounds, render_planet, fixed_sizes, show_sat_border, show_polar_circle,
                            show_equator, show_graticule, show_crosshairs, show_terminator, show_eclipse, show_sun, earth_fixed_camera,
                            use_gpu_rendering: self.use_gpu_rendering, show_clouds, show_day_night, show_city_lights,
                            show_stars, show_borders, show_cities,
                            trackpad_rotate,
                            north_up,
                            enabled_moons: enabled_moons.clone(),
                            moon_inclination_override,
                            show_moon_orbits,
                            show_moon_lines,
                            show_moon_labels,
                            moon_camera_distance_km,
                            tle_monochrome,
                            show_ground_tracks,
                        };
                        let sun_dir = {
                            let day_of_year = continuous_day_of_year(self.start_timestamp, time);
                            // When sun-fixed camera is active, zero out the seasonal
                            // declination so the Sun traces the equator. This is
                            // unphysical (the Sun really moves on the ecliptic ±23.45°)
                            // but pedagogically necessary: SSO only locks RA, so with
                            // real declination the terminator would wobble ±23° over
                            // a year relative to the SSO plane. Flattening the Sun to
                            // the equator gives a clean demonstration where SSO stays
                            // exactly on the terminator forever.
                            let decl_rad: f64 = if sun_fixed_camera {
                                0.0
                            } else {
                                let declination = SOLAR_DECLINATION_MAX * ((360.0 / DAYS_PER_YEAR) * (day_of_year + 10.0)).to_radians().cos();
                                declination.to_radians()
                            };
                            let sun_ra = ((day_of_year - 80.0) * 360.0 / DAYS_PER_YEAR).to_radians();
                            let sun_inertial = Vector3::new(
                                decl_rad.cos() * sun_ra.cos(),
                                decl_rad.sin(),
                                -decl_rad.cos() * sun_ra.sin(),
                            );
                            let sun_shader = body_y_rotation.transpose() * sun_inertial;
                            [sun_shader.x as f32, sun_shader.y as f32, sun_shader.z as f32]
                        };
                        let device_layers_ref: &[DeviceLayer] = if show_devices { &planet.device_layers } else { &[] };
                        let conj_lines: Vec<_> = if planet.show_conjunction_lines && !planet.conjunction_cache.show_heatmap {
                            let t = planet.conjunction_cache.threshold_km;
                            planet.conjunction_cache.conjunctions.iter()
                                .filter(|c| c.distance_km <= t)
                                .map(|c| (c.clone(), t)).collect()
                        } else {
                            Vec::new()
                        };
                        let conj_heatmap: Vec<_> = if planet.conjunction_cache.show_heatmap {
                            let t = planet.conjunction_cache.threshold_km;
                            planet.conjunction_cache.conjunctions.iter()
                                .filter(|c| c.tca_seconds > 0.0 && c.min_distance_km < t)
                                .map(|c| (c.clone(), t))
                                .collect()
                        } else {
                            Vec::new()
                        };
                        let correcting_sats: HashSet<String> = if planet.kessler.course_correction_enabled {
                            planet.kessler.active_corrections.iter()
                                .filter(|c| c.offset_at(time).abs() > 1e-6)
                                .map(|c| c.sat_name.clone())
                                .collect()
                        } else {
                            HashSet::new()
                        };
                        let hit_sats: HashSet<String> = planet.kessler.collided_pairs.iter()
                            .flat_map(|(a, b)| [a.clone(), b.clone()])
                            .collect();
                        if show_radiation_belts || planet.show_radiation_window {
                            let sphere_r = planet_radius + planet.radiation.heatmap_altitude_km;
                            match planet.radiation.heatmap_mode {
                                crate::config::HeatmapMode::IgrfField => {
                                    if planet.radiation.igrf_grid_cache.as_ref().map(|(r, _)| *r) != Some(sphere_r) {
                                        planet.radiation.igrf_grid_cache = Some((sphere_r, crate::igrf::IgrfGrid::new(sphere_r)));
                                    }
                                }
                                crate::config::HeatmapMode::IgrfRadiation => {}
                                _ => {}
                            }
                        }
                        let physics_colors: HashMap<(usize, usize), egui::Color32> = {
                            let mut map = HashMap::new();
                            for c in &planet.constellations {
                                if !c.physics.enabled || c.physics_state.is_empty() { continue; }
                                let cidx = planet.constellations.iter().position(|x| std::ptr::eq(x, c)).unwrap_or(usize::MAX);
                                for (si, ps) in c.physics_state.iter().enumerate() {
                                    let color = if ps.is_dead {
                                        crate::physics::dead_color()
                                    } else {
                                        match c.physics.color_mode {
                                            crate::physics::PhysicsColorMode::Battery => {
                                                crate::physics::battery_color(ps.state_of_charge(&c.physics))
                                            }
                                            crate::physics::PhysicsColorMode::Temperature => {
                                                crate::physics::temperature_color(ps.temperature_k)
                                            }
                                            crate::physics::PhysicsColorMode::Normal => continue,
                                        }
                                    };
                                    map.insert((cidx, si), color);
                                }
                            }
                            map
                        };
                        let physics_info: HashMap<(usize, usize), (f64, f64, bool)> = {
                            let mut map = HashMap::new();
                            for c in &planet.constellations {
                                if !c.physics.enabled || c.physics_state.is_empty() { continue; }
                                let cidx = planet.constellations.iter().position(|x| std::ptr::eq(x, c)).unwrap_or(usize::MAX);
                                for (si, ps) in c.physics_state.iter().enumerate() {
                                    map.insert((cidx, si), (ps.state_of_charge(&c.physics), ps.temperature_k, ps.is_dead));
                                }
                            }
                            map
                        };
                        let mut ctx_menu_req: Option<(egui::Pos2, f64, f64)> = None;
                        let mut label_click_req: Option<(bool, usize, egui::Pos2)> = None;
                        let ground_tracks_vec: Vec<Vec<(f64, f64)>> = if show_ground_tracks {
                            planet.ground_track_history.values()
                                .map(|v| v.iter().map(|&(lat, lon, _)| (lat, lon)).collect())
                                .collect()
                        } else {
                            Vec::new()
                        };
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
                            gpu_available,
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
                            detail_bounds,
                            #[cfg(not(target_arch = "wasm32"))]
                            { match &self.geo_data { GeoLoadState::Loaded(d) => if show_borders { d.borders.as_slice() } else { &[] }, _ => &[] } },
                            #[cfg(target_arch = "wasm32")]
                            &[],
                            #[cfg(not(target_arch = "wasm32"))]
                            { match &self.geo_data { GeoLoadState::Loaded(d) => if show_cities { d.cities.as_slice() } else { &[] }, _ => &[] } },
                            #[cfg(target_arch = "wasm32")]
                            &[],
                            &conj_lines,
                            &conj_heatmap,
                            &correcting_sats,
                            &hit_sats,
                            if show_radiation_belts || planet.show_radiation_window {
                                Some(&planet.radiation)
                            } else {
                                None
                            },
                            &self.moon_image_handles,
                            &mut ctx_menu_req,
                            &mut label_click_req,
                            &physics_colors,
                            &physics_info,
                            &ground_tracks_vec,
                        );
                        {
                            // Strip the sun-fix rotation before saving so it
                            // doesn't compound across frames.
                            let rot = rot * sun_y_rotation.transpose();
                            let roll = self.tabs[tab_idx].settings.camera_roll.to_radians();
                            if roll.abs() < 1e-9 {
                                self.tabs[tab_idx].settings.rotation = rot;
                            } else {
                                let c = roll.cos();
                                let s = roll.sin();
                                let roll_inv = nalgebra::Matrix3::new(
                                    c, s, 0.0,
                                    -s, c, 0.0,
                                    0.0, 0.0, 1.0,
                                );
                                self.tabs[tab_idx].settings.rotation = roll_inv * rot;
                            }
                        }
                        self.tabs[tab_idx].settings.zoom = new_zoom;
                        if let Some((screen_pos, lat, lon)) = ctx_menu_req {
                            self.context_menu = Some(crate::viewer::ContextMenuState {
                                screen_pos, lat, lon,
                                tab_idx, planet_idx,
                            });
                            self.editing_place = None;
                        }
                        if let Some((is_gs, idx, click_pos)) = label_click_req {
                            self.editing_place = Some(crate::viewer::EditingPlaceState {
                                is_gs, idx,
                                tab_idx, planet_idx,
                                screen_pos: click_pos,
                                just_opened: true,
                            });
                            self.context_menu = None;
                        }
                    });
                } else if render_planet && is_2d_projection {
                    ui.vertical(|ui| {
                        let planet = &self.tabs[tab_idx].planets[planet_idx];
                        let proj = planet_projection.instance();
                        let rad_ref = if show_radiation_belts || planet.show_radiation_window {
                            Some(&planet.radiation)
                        } else {
                            None
                        };
                        draw_map_view(
                            ui,
                            &format!("ground_{}", view_name),
                            &constellations_data,
                            proj,
                            view_width,
                            view_height,
                            sat_radius,
                            single_color,
                            link_width,
                            show_orbits,
                            show_links,
                            show_coverage,
                            coverage_angle,
                            show_routing_paths,
                            show_manhattan_path,
                            show_shortest_path,
                            show_radiation_path,
                            radiation_weight,
                            show_graticule,
                            show_crosshairs,
                            &planet.satellite_cameras,
                            planet_radius,
                            #[cfg(not(target_arch = "wasm32"))]
                            { match &self.geo_data { GeoLoadState::Loaded(d) => if show_borders { d.borders.as_slice() } else { &[] }, _ => &[] } },
                            #[cfg(target_arch = "wasm32")]
                            &[],
                            #[cfg(not(target_arch = "wasm32"))]
                            { match &self.geo_data { GeoLoadState::Loaded(d) => if show_cities { d.cities.as_slice() } else { &[] }, _ => &[] } },
                            #[cfg(target_arch = "wasm32")]
                            &[],
                            &planet.ground_stations,
                            rad_ref,
                            &body_y_rotation,
                            time,
                            gpu_available,
                            planet_projection.shader_id(),
                        );
                    });
                }

                if show_torus {
                    ui.vertical(|ui| {
                        let planet = &mut self.tabs[tab_idx].planets[planet_idx];
                        let rad_grid: Option<&crate::igrf::IgrfRadGrid> = None;
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
                            show_orbits,
                            single_color,
                            torus_zoom,
                            &mut planet.satellite_cameras,
                            show_routing_paths,
                            show_manhattan_path,
                            show_shortest_path,
                            false,
                            radiation_weight,
                            show_asc_desc_colors,
                            color_ascending,
                            color_descending,
                            color_links,
                            planet_radius,
                            &mut planet.pending_cameras,
                            &mut self.camera_id_counter,
                            &mut planet.cameras_to_remove,
                            link_width,
                            fixed_sizes,
                            &body_y_rotation,
                            rad_grid,
                        );
                        let roll = self.tabs[tab_idx].settings.camera_roll.to_radians();
                        if roll.abs() < 1e-9 {
                            self.tabs[tab_idx].settings.rotation = trot;
                        } else {
                            let c = roll.cos();
                            let s = roll.sin();
                            let roll_inv = nalgebra::Matrix3::new(
                                c, s, 0.0,
                                -s, c, 0.0,
                                0.0, 0.0, 1.0,
                            );
                            self.tabs[tab_idx].settings.rotation = roll_inv * trot;
                        }
                        self.torus_zoom = tzoom;
                    });
                }

                if show_solar_system {
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
                                // Classic smoothstep: 3t^2 - 2t^3.
                                // Gentler ease than t^3/(t^3+(1-t)^3) — avoids
                                // the "slow-fast-slow" feel of sharper curves.
                                let s = frac;
                                let frac = s * s * (3.0 - 2.0 * s);

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
                            let show_ss_labels = self.tabs[tab_idx].settings.show_ss_labels;
                            let show_cal = self.tabs[tab_idx].settings.show_circular_calendar;
                            let hide_bodies = self.tabs[tab_idx].settings.solar_system_hide_bodies;
                            let ss_click = crate::solar_system::draw_solar_system_view(
                                plot_ui,
                                celestial_body,
                                ss_timestamp,
                                ss_handles,
                                dark_mode,
                                log_power,
                                ast_slice,
                                self.asteroid_sprite.as_ref(),
                                show_ss_labels,
                                show_cal,
                                hide_bodies,
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

                if show_planet_sizes {
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
                            &self.planet_sizes_handles,
                            &mut self.planet_sizes_t,
                            &mut az,
                            &mut self.planet_sizes_enabled,
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
                let (ocean, land, ice) = self.tabs.iter()
                    .flat_map(|t| t.planets.iter())
                    .find(|p| p.celestial_body == body && p.skin == Skin::Abstract)
                    .map(|p| (
                        [p.abstract_ocean.r(), p.abstract_ocean.g(), p.abstract_ocean.b()],
                        [p.abstract_land.r(), p.abstract_land.g(), p.abstract_land.b()],
                        [p.abstract_ice.r(), p.abstract_ice.g(), p.abstract_ice.b()],
                    ))
                    .unwrap_or(([25, 40, 80], [60, 75, 85], [140, 150, 160]));
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
