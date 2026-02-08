//! Application shell and eframe integration.
//!
//! Defines the App struct, initialization, and the main update loop that
//! drives texture loading, TLE polling, tile overlay compositing, and
//! the egui dock-based tab layout.

use crate::celestial::{CelestialBody, Skin, TextureResolution};
use crate::config::TabConfig;
#[cfg(not(target_arch = "wasm32"))]
use crate::geo::{GeoLoadState, dirs_cache, load_geo_overlay};
use crate::drawing::draw_satellite_camera;
use crate::renderer::SphereRenderer;
use crate::texture::TextureLoadState;
use crate::tile::{
    TileCoord, DetailBounds, DetailTexture, TileCacheEntry,
    TileQuadTree, TileOverlayState, TileFetchResult,
    lon_lat_to_tile, tile_to_lon_lat, camera_zoom_to_tile_zoom,
};
use crate::time::{greenwich_mean_sidereal_time, body_rotation_angle};
use crate::tle::{TlePreset, TleLoadState};
use crate::viewer::ViewerState;
use crate::texture::load_earth_texture;
#[cfg(not(target_arch = "wasm32"))]
use crate::texture::decode_jpeg_pixels;
#[cfg(target_arch = "wasm32")]
use crate::texture::{
    TEXTURE_RESULT, STAR_TEXTURE_RESULT, MILKY_WAY_TEXTURE_RESULT,
    NIGHT_TEXTURE_RESULT, CLOUD_TEXTURE_RESULT,
};
#[cfg(target_arch = "wasm32")]
use crate::tle::TLE_FETCH_RESULT;
use eframe::{egui, glow};
use egui::mutex::Mutex;
use egui_dock::{DockArea, DockState};
use nalgebra::{Matrix3, Vector3};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::sync::{Arc, mpsc};
use chrono::{Duration, Utc};
use glow::HasContext as _;

pub(crate) struct App {
    pub(crate) dock_state: DockState<usize>,
    pub(crate) viewer: ViewerState,
    first_frame: bool,
}

impl App {
    pub(crate) fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let gl = cc.gl.as_ref().expect("glow backend required");
        let sphere_renderer = Arc::new(Mutex::new(SphereRenderer::new(gl)));

        let torus_initial = Matrix3::new(
            1.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
            0.0, 1.0, 0.0,
        );
        let builtin_texture = Arc::new(load_earth_texture());
        #[cfg(not(target_arch = "wasm32"))]
        let (tle_fetch_tx, tle_fetch_rx) = mpsc::channel();

        {
            let mut renderer = sphere_renderer.lock();
            let builtin_key = if cfg!(target_arch = "wasm32") {
                (CelestialBody::Earth, Skin::Default, TextureResolution::R512)
            } else {
                (CelestialBody::Earth, Skin::Default, TextureResolution::R8192)
            };
            renderer.upload_texture(gl, builtin_key, &builtin_texture);
        }

        #[allow(unused_mut)]
        let mut app = Self {
            dock_state: DockState::new(vec![0]),
            viewer: ViewerState {
                tabs: vec![TabConfig::new("View 1".to_string())],
                camera_id_counter: 0,
                tab_counter: 1,
                torus_zoom: 1.0,
                torus_rotation: torus_initial,
                planet_textures: {
                    let mut map = HashMap::new();
                    let builtin_key = if cfg!(target_arch = "wasm32") {
                        (CelestialBody::Earth, Skin::Default, TextureResolution::R512)
                    } else {
                        (CelestialBody::Earth, Skin::Default, TextureResolution::R8192)
                    };
                    map.insert(builtin_key, builtin_texture.clone());
                    map
                },
                ring_textures: HashMap::new(),
                cloud_textures: HashMap::new(),
                planet_image_handles: HashMap::new(),
                texture_resolution: TextureResolution::R8192,
                last_rotation: None,
                earth_resolution: 512,
                last_resolution: 0,
                texture_load_state: TextureLoadState::Loaded(builtin_texture),
                pending_body: None,
                dark_mode: true,
                show_info: false,
                real_time: 0.0,
                start_timestamp: Utc::now(),
                show_side_panel: true,
                pending_add_tab: None,
                current_gmst: 0.0,
                auto_cycle_tabs: false,
                auto_hide_tab_bar: false,
                ui_visible: true,
                cycle_interval: 5.0,
                last_cycle_time: 0.0,
                use_gpu_rendering: true,
                show_borders: false,
                show_cities: false,
                active_tab_idx: 0,
                #[cfg(not(target_arch = "wasm32"))]
                geo_data: GeoLoadState::NotLoaded,
                #[cfg(not(target_arch = "wasm32"))]
                geo_fetch_rx: None,
                dragging_place: None,
                night_texture: None,
                star_texture: None,
                milky_way_texture: None,
                night_texture_loading: false,
                star_texture_loading: false,
                milky_way_texture_loading: false,
                cloud_texture_loading: false,
                sphere_renderer: Some(sphere_renderer),
                #[cfg(not(target_arch = "wasm32"))]
                tle_fetch_tx,
                #[cfg(not(target_arch = "wasm32"))]
                tle_fetch_rx,
                #[cfg(not(target_arch = "wasm32"))]
                tile_overlay: {
                    let (fetch_tx, fetch_rx) = mpsc::channel::<(TileCoord, std::path::PathBuf, u64)>();
                    let (result_tx, result_rx) = mpsc::channel::<TileFetchResult>();
                    let disk_cache_dir = dirs_cache().join("leo-viz").join("tiles");
                    let _ = std::fs::create_dir_all(&disk_cache_dir);

                    let fetch_generation = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
                    let fetch_rx = std::sync::Arc::new(std::sync::Mutex::new(fetch_rx));
                    for _ in 0..4 {
                        let rx = fetch_rx.clone();
                        let tx = result_tx.clone();
                        let gen = fetch_generation.clone();
                        std::thread::spawn(move || {
                            loop {
                                let msg = {
                                    let lock = rx.lock().unwrap();
                                    lock.recv()
                                };
                                let (coord, cache_dir, req_gen) = match msg {
                                    Ok(m) => m,
                                    Err(_) => break,
                                };
                                if coord.z > 6 && gen.load(std::sync::atomic::Ordering::Relaxed) != req_gen {
                                    let _ = tx.send(TileFetchResult { coord, pixels: Vec::new(), width: 0, height: 0 });
                                    continue;
                                }
                                let cache_path = cache_dir
                                    .join(coord.z.to_string())
                                    .join(coord.y.to_string())
                                    .join(format!("{}.jpg", coord.x));

                                let pixels_result = if cache_path.exists() {
                                    std::fs::read(&cache_path).ok().and_then(|bytes| decode_jpeg_pixels(&bytes))
                                } else {
                                    None
                                };

                                let fetched = if let Some(p) = pixels_result {
                                    Some(p)
                                } else {
                                    let url = format!(
                                        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{}/{}/{}",
                                        coord.z, coord.y, coord.x
                                    );
                                    match ureq::get(&url).call() {
                                        Ok(resp) => {
                                            let mut bytes = Vec::new();
                                            if std::io::Read::read_to_end(&mut resp.into_reader(), &mut bytes).is_ok() {
                                                if let Some(p) = decode_jpeg_pixels(&bytes) {
                                                    if let Some(parent) = cache_path.parent() {
                                                        let _ = std::fs::create_dir_all(parent);
                                                    }
                                                    let _ = std::fs::write(&cache_path, &bytes);
                                                    Some(p)
                                                } else {
                                                    None
                                                }
                                            } else {
                                                None
                                            }
                                        }
                                        Err(_) => None,
                                    }
                                };

                                if coord.z >= 10 {
                                    if let Some((ref px, _, _)) = fetched {
                                        let step = (px.len() / 256).max(1);
                                        let n = (px.len() / step).max(1) as u64;
                                        let (mut sr, mut sg, mut sb) = (0u64, 0u64, 0u64);
                                        for i in 0..n as usize {
                                            let p = px[i * step];
                                            sr += p[0] as u64; sg += p[1] as u64; sb += p[2] as u64;
                                        }
                                        let (ar, ag, ab) = (sr / n, sg / n, sb / n);
                                        let mut var = 0u64;
                                        for i in 0..n as usize {
                                            let p = px[i * step];
                                            let dr = p[0] as i64 - ar as i64;
                                            let dg = p[1] as i64 - ag as i64;
                                            let db = p[2] as i64 - ab as i64;
                                            var += (dr*dr + dg*dg + db*db) as u64;
                                        }
                                        if var / n < 100 {
                                            let _ = tx.send(TileFetchResult { coord, pixels: Vec::new(), width: 0, height: 0 });
                                            continue;
                                        }
                                    }
                                }
                                if let Some((pixels, w, h)) = fetched {
                                    let _ = tx.send(TileFetchResult { coord, pixels, width: w, height: h });
                                } else {
                                    let _ = tx.send(TileFetchResult { coord, pixels: Vec::new(), width: 0, height: 0 });
                                }
                            }
                        });
                    }

                    TileOverlayState {
                        enabled: false,
                        tile_tree: TileQuadTree::new(4096),
                        disk_cache_dir,
                        detail_texture: None,
                        fetch_tx,
                        result_rx,
                        last_zoom: 0,
                        fetch_generation: fetch_generation.clone(),
                        generation: 0,
                        tile_x_origin: 0,
                        pending_tiles: HashSet::new(),
                        needed_tiles: Vec::new(),
                        dirty: false,
                        last_compose: std::time::Instant::now(),
                        base_fetched: false,
                        compose_buffer: Vec::new(),
                    }
                },
                view_width: 800.0,
                view_height: 600.0,
                solar_system_handles: HashMap::new(),
                ss_last_render_instant: None,
                show_planet_sizes: false,
                planet_sizes_t: 0.0,
                planet_sizes_auto_zoom: false,
                planet_sizes_zoom_duration: 30.0,
                planet_sizes_stay_duration: 3.0,
                planet_sizes_auto_time: 0.0,
                ss_auto_zoom: false,
                ss_auto_zoom_duration: 30.0,
                ss_auto_zoom_stay: 3.0,
                ss_auto_zoom_time: 0.0,
                asteroid_sprite: None,
                asteroid_state: crate::solar_system::AsteroidLoadState::NotLoaded,
                #[cfg(not(target_arch = "wasm32"))]
                asteroid_rx: None,
                hohmann: crate::solar_system::HohmannState::default(),
            },
            first_frame: true,
        };

        #[cfg(target_arch = "wasm32")]
        {
            let loc = web_sys::window().and_then(|w| Some(w.location()));
            let path = loc.as_ref().and_then(|l| l.pathname().ok()).unwrap_or_default();
            let hash = loc.as_ref().and_then(|l| l.hash().ok()).unwrap_or_default();

            if path.ends_with("/demo") || path.ends_with("/demo/") {
                app.setup_demo();
            } else if hash.starts_with("#c=") {
                use crate::config::ShareableConfig;
                if let Some(cfg) = ShareableConfig::from_url_hash(&hash) {
                    cfg.apply_to_planet(&mut app.viewer.tabs[0].planets[0]);
                }
            }
            app.viewer.last_url_hash = String::new();
        }

        app
    }
}
impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let v = &mut self.viewer;

        ctx.set_visuals(if v.dark_mode {
            let mut vis = egui::Visuals::dark();
            let black = egui::Color32::BLACK;
            vis.window_fill = black;
            vis.panel_fill = black;
            vis.extreme_bg_color = black;
            vis.faint_bg_color = egui::Color32::from_gray(15);
            vis
        } else {
            egui::Visuals::light()
        });

        ctx.options_mut(|o| o.input_options.max_click_dist = 1.0);

        let focused = self.dock_state.focused_leaf();
        let active_tab_idx = self.dock_state.iter_all_tabs()
            .find(|((s, n), _)| focused == Some((*s, *n)))
            .map(|(_, tab)| *tab)
            .unwrap_or(0);

        let tex_res = v.texture_resolution;
        let bodies_needed: Vec<(CelestialBody, Skin, TextureResolution)> = {
            let mut seen = std::collections::HashSet::new();
            v.tabs.get(active_tab_idx)
                .into_iter()
                .flat_map(|tab| tab.planets.iter().map(|p| (p.celestial_body, p.skin, tex_res)))
                .filter(|key| seen.insert(*key))
                .collect()
        };
        for (body, skin, _) in &bodies_needed {
            v.load_texture_for_body(*body, *skin, ctx);
        }

        #[cfg(not(target_arch = "wasm32"))]
        for body in CelestialBody::ALL {
            let key = (body, Skin::Default, TextureResolution::R512);
            if v.planet_textures.contains_key(&key) {
                continue;
            }
            if let Some(filename) = Skin::Default.filename(body, TextureResolution::R512) {
                if let Ok(bytes) = std::fs::read(crate::texture::asset_path(filename)) {
                    if let Ok(tex) = crate::texture::EarthTexture::from_bytes(&bytes) {
                        let tex = tex.downscale((tex.width / 512).max(1));
                        v.planet_textures.insert(key, std::sync::Arc::new(tex));
                    }
                }
            }
        }

        let (show_clouds, show_day_night, show_stars) = v.tabs.get(active_tab_idx)
            .map(|t| (t.settings.show_clouds, t.settings.show_day_night, t.settings.show_stars))
            .unwrap_or((false, false, false));

        if show_clouds {
            v.load_cloud_texture(ctx);
        }

        if show_day_night {
            v.load_night_texture(ctx);
        }

        if show_stars {
            v.load_star_textures(ctx);
        }

        if let Some(gl) = frame.gl() {
            if let Some(ref sphere_renderer) = v.sphere_renderer {
                let mut renderer = sphere_renderer.lock();
                for (body, skin, res) in &bodies_needed {
                    if let Some(tex) = v.planet_textures.get(&(*body, *skin, *res)) {
                        renderer.upload_texture(gl, (*body, *skin, *res), tex);
                    }
                }
                if show_clouds {
                    if let Some(cloud_tex) = v.cloud_textures.get(&tex_res) {
                        renderer.upload_cloud_texture(gl, tex_res, cloud_tex);
                    }
                }
                if show_day_night {
                    if let Some(night_tex) = &v.night_texture {
                        renderer.upload_night_texture(gl, night_tex);
                    }
                }
                if show_stars {
                    if let Some(star_tex) = &v.star_texture {
                        renderer.upload_star_texture(gl, star_tex);
                    }
                    if let Some(mw_tex) = &v.milky_way_texture {
                        renderer.upload_milky_way_texture(gl, mw_tex);
                    }
                }
                for (body, ring_tex) in &v.ring_textures {
                    renderer.upload_ring_texture(gl, *body, ring_tex);
                }
                renderer.evict_unused_textures(gl, &bodies_needed);
            }
        }

        let bodies_set: std::collections::HashSet<_> = bodies_needed.iter().copied().collect();
        v.planet_textures.retain(|k, _| bodies_set.contains(k) || (k.1 == Skin::Default && k.2 == TextureResolution::R512));
        v.planet_image_handles.retain(|k, _| bodies_set.contains(k));
        let body_set: std::collections::HashSet<CelestialBody> = bodies_needed.iter().map(|k| k.0).collect();
        v.ring_textures.retain(|b, _| body_set.contains(b));

        #[cfg(not(target_arch = "wasm32"))]
        while let Ok((preset, result)) = v.tle_fetch_rx.try_recv() {
            for tab in &mut v.tabs {
                for planet in &mut tab.planets {
                    if let Some((_, state, _)) = planet.tle_selections.get_mut(&preset) {
                        if matches!(state, TleLoadState::Loading) {
                            *state = match result.clone() {
                                Ok(satellites) => TleLoadState::Loaded {
                                    satellites,
                                },
                                Err(e) => TleLoadState::Failed(e),
                            };
                        }
                    }
                }
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            if (v.show_borders || v.show_cities) && matches!(v.geo_data, GeoLoadState::NotLoaded) {
                let (tx, rx) = mpsc::channel();
                v.geo_fetch_rx = Some(rx);
                v.geo_data = GeoLoadState::Loading;
                std::thread::spawn(move || { let _ = tx.send(load_geo_overlay()); });
            }
            if let Some(ref rx) = v.geo_fetch_rx {
                if let Ok(result) = rx.try_recv() {
                    v.geo_data = match result {
                        Ok(data) => GeoLoadState::Loaded(data),
                        Err(_) => GeoLoadState::Failed,
                    };
                    v.geo_fetch_rx = None;
                }
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        if v.tile_overlay.enabled {
            while let Ok(result) = v.tile_overlay.result_rx.try_recv() {
                v.tile_overlay.pending_tiles.remove(&result.coord);
                if !result.pixels.is_empty() {
                    v.tile_overlay.tile_tree.insert(result.coord, TileCacheEntry {
                        pixels: result.pixels,
                        width: result.width,
                        height: result.height,
                    });
                    v.tile_overlay.dirty = true;
                }
            }

            if !v.tile_overlay.base_fetched {
                v.tile_overlay.base_fetched = true;
                for bz in 0u8..=3 {
                    let n = 1u32 << bz;
                    for bx in 0..n {
                        for by in 0..n {
                            let c = TileCoord { x: bx, y: by, z: bz };
                            if !v.tile_overlay.tile_tree.has_tile(&c) {
                                v.tile_overlay.pending_tiles.insert(c);
                                let _ = v.tile_overlay.fetch_tx.send((
                                    c,
                                    v.tile_overlay.disk_cache_dir.clone(),
                                    v.tile_overlay.generation,
                                ));
                            }
                        }
                    }
                }
            }

            let has_earth = v.tabs.get(active_tab_idx)
                .map(|t| t.planets.iter().any(|p| p.celestial_body == CelestialBody::Earth))
                .unwrap_or(false);

            if has_earth {
                let (tile_rotation, tile_time, tile_zoom, tile_earth_fixed) = v.tabs.get(active_tab_idx)
                    .map(|t| (t.settings.rotation, t.settings.time, t.settings.zoom, t.settings.earth_fixed_camera))
                    .unwrap_or((Matrix3::identity(), 0.0, 1.0, false));
                let surface_rotation = if tile_earth_fixed {
                    tile_rotation
                } else {
                    let body_rot = body_rotation_angle(CelestialBody::Earth, tile_time, v.current_gmst);
                    let (cb, sb) = (body_rot.cos(), body_rot.sin());
                    let body_mat = Matrix3::new(
                        cb, 0.0, sb,
                        0.0, 1.0, 0.0,
                        -sb, 0.0, cb,
                    );
                    tile_rotation * body_mat
                };
                let view_scale = tile_zoom / 1.15;
                let aspect = (v.view_width / v.view_height.max(1.0)) as f64;
                let inv_rot = surface_rotation.transpose();
                let b = 1.0 - CelestialBody::Earth.flattening();
                let b2 = b * b;
                let screen_to_lonlat = |sx: f64, sy: f64| -> Option<(f64, f64)> {
                    let cx = sx * aspect / view_scale;
                    let cy = sy / view_scale;
                    let o = inv_rot * nalgebra::Vector3::new(cx, cy, 0.0);
                    let d = inv_rot * nalgebra::Vector3::new(0.0, 0.0, -1.0);
                    let a_coef = d.x*d.x + d.y*d.y/b2 + d.z*d.z;
                    let b_coef = 2.0*(o.x*d.x + o.y*d.y/b2 + o.z*d.z);
                    let c_coef = o.x*o.x + o.y*o.y/b2 + o.z*o.z - 1.0;
                    let disc = b_coef*b_coef - 4.0*a_coef*c_coef;
                    if disc < 0.0 { return None; }
                    let t = (-b_coef - disc.sqrt()) / (2.0 * a_coef);
                    let wp = o + t * d;
                    let lat = (wp.y / b).clamp(-1.0, 1.0).asin();
                    let lon = (-wp.z).atan2(wp.x);
                    Some((lon.to_degrees(), lat.to_degrees()))
                };
                let mut samples: Vec<(f64, f64)> = vec![
                    (-1.0, -1.0), (0.0, -1.0), (1.0, -1.0),
                    (-1.0,  0.0), (0.0,  0.0), (1.0,  0.0),
                    (-1.0,  1.0), (0.0,  1.0), (1.0,  1.0),
                    (-0.7, -0.7), (0.7, -0.7), (-0.7, 0.7), (0.7, 0.7),
                ];
                let globe_rx = view_scale / aspect;
                let globe_ry = view_scale;
                if globe_rx < 1.0 || globe_ry < 1.0 {
                    for i in 0..24 {
                        let a = i as f64 * PI * 2.0 / 24.0;
                        let sx = a.cos() * globe_rx.min(1.0) * 0.95;
                        let sy = a.sin() * globe_ry.min(1.0) * 0.95;
                        samples.push((sx, sy));
                    }
                }
                let sample_pts: Vec<(f64, f64)> = samples.iter()
                    .filter_map(|&(sx, sy)| screen_to_lonlat(sx, sy)).collect();
                if !sample_pts.is_empty() {
                let (sin_sum, cos_sum) = sample_pts.iter().fold((0.0, 0.0), |(s, c), &(lon, _)| {
                    let r = lon.to_radians();
                    (s + r.sin(), c + r.cos())
                });
                let center_lon_avg = sin_sum.atan2(cos_sum).to_degrees();
                let mut min_lon = f64::MAX;
                let mut max_lon = f64::MIN;
                let mut min_lat = f64::MAX;
                let mut max_lat = f64::MIN;
                for &(lon, lat) in &sample_pts {
                    let mut dlon = lon - center_lon_avg;
                    if dlon > 180.0 { dlon -= 360.0; }
                    if dlon < -180.0 { dlon += 360.0; }
                    let adjusted_lon = center_lon_avg + dlon;
                    if adjusted_lon < min_lon { min_lon = adjusted_lon; }
                    if adjusted_lon > max_lon { max_lon = adjusted_lon; }
                    if lat < min_lat { min_lat = lat; }
                    if lat > max_lat { max_lat = lat; }
                }
                let margin = 1.5;
                let lon_center = (min_lon + max_lon) / 2.0;
                let lat_center = (min_lat + max_lat) / 2.0;
                let tile_deg = 360.0 / (1u64 << camera_zoom_to_tile_zoom(tile_zoom).clamp(2, 18)) as f64;
                let min_half = tile_deg * 3.0;
                let lon_half = ((max_lon - min_lon) / 2.0 * margin).max(min_half);
                let lat_half = ((max_lat - min_lat) / 2.0 * margin).max(min_half);
                min_lon = lon_center - lon_half;
                max_lon = lon_center + lon_half;
                min_lat = (lat_center - lat_half).max(-85.0);
                max_lat = (lat_center + lat_half).min(85.0);
                let lon_span = max_lon - min_lon;

                let mut tile_zoom = camera_zoom_to_tile_zoom(tile_zoom).max(2);
                let (needed, x_origin) = loop {
                    let n = 1u32 << tile_zoom;
                    let tl = lon_lat_to_tile(min_lon, max_lat, tile_zoom);
                    let br = lon_lat_to_tile(max_lon, min_lat, tile_zoom);
                    let tile_width_deg = 360.0 / n as f64;
                    let full_lon = lon_span >= 360.0 - tile_width_deg;
                    let (x_count, x_range_v, x_org) = if full_lon {
                        (n, (0..n).collect::<Vec<u32>>(), 0u32)
                    } else if tl.x <= br.x {
                        (br.x - tl.x + 1, (tl.x..=br.x).collect(), tl.x)
                    } else {
                        ((n - tl.x) + br.x + 1, (tl.x..n).chain(0..=br.x).collect(), tl.x)
                    };
                    let y_min = tl.y.min(br.y);
                    let y_max = tl.y.max(br.y);
                    let y_count = y_max - y_min + 1;
                    let total = x_count as usize * y_count as usize;
                    if total <= 256 || tile_zoom <= 2 {
                        let mut tiles = Vec::with_capacity(total);
                        for &tx in &x_range_v {
                            for ty in y_min..=y_max {
                                tiles.push(TileCoord { x: tx, y: ty, z: tile_zoom });
                            }
                        }
                        let cx = x_range_v.iter().map(|&x| x as f64).sum::<f64>() / x_range_v.len() as f64;
                        let cy = (y_min as f64 + y_max as f64) / 2.0;
                        tiles.sort_by(|a, b| {
                            let da = (a.x as f64 - cx).powi(2) + (a.y as f64 - cy).powi(2);
                            let db = (b.x as f64 - cx).powi(2) + (b.y as f64 - cy).powi(2);
                            da.partial_cmp(&db).unwrap()
                        });
                        break (tiles, x_org);
                    }
                    tile_zoom -= 1;
                };

                let bounds_changed = v.tile_overlay.last_zoom != tile_zoom
                    || v.tile_overlay.needed_tiles != needed;

                if bounds_changed && !needed.is_empty() {
                    let needed_set: HashSet<TileCoord> = needed.iter().copied().collect();
                    let stale_count = v.tile_overlay.pending_tiles.iter()
                        .filter(|c| !needed_set.contains(c)).count();
                    if stale_count > 8 {
                        v.tile_overlay.generation = v.tile_overlay.generation.wrapping_add(1);
                        v.tile_overlay.fetch_generation.store(
                            v.tile_overlay.generation,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                    v.tile_overlay.last_zoom = tile_zoom;
                    v.tile_overlay.tile_x_origin = x_origin;
                    v.tile_overlay.needed_tiles = needed.clone();
                    v.tile_overlay.dirty = true;
                    let mut keep_set = needed_set.clone();
                    for coord in &needed {
                        for step in &[2u8, 4] {
                            if coord.z > *step + 3 {
                                let az = coord.z - step;
                                keep_set.insert(TileCoord { x: coord.x >> step, y: coord.y >> step, z: az });
                            }
                        }
                    }
                    v.tile_overlay.pending_tiles.retain(|c| keep_set.contains(c));

                    for coord in &needed {
                        if !v.tile_overlay.tile_tree.has_tile(coord)
                            && !v.tile_overlay.pending_tiles.contains(coord)
                        {
                            v.tile_overlay.pending_tiles.insert(*coord);
                            let _ = v.tile_overlay.fetch_tx.send((
                                *coord,
                                v.tile_overlay.disk_cache_dir.clone(),
                                v.tile_overlay.generation,
                            ));
                        }
                        for step in &[2u8, 4] {
                            if coord.z > *step + 3 {
                                let az = coord.z - step;
                                let ac = TileCoord { x: coord.x >> step, y: coord.y >> step, z: az };
                                if !v.tile_overlay.tile_tree.has_tile(&ac)
                                    && !v.tile_overlay.pending_tiles.contains(&ac)
                                {
                                    v.tile_overlay.pending_tiles.insert(ac);
                                    let _ = v.tile_overlay.fetch_tx.send((
                                        ac,
                                        v.tile_overlay.disk_cache_dir.clone(),
                                        v.tile_overlay.generation,
                                    ));
                                }
                            }
                        }
                    }
                }

                let all_loaded = v.tile_overlay.pending_tiles.is_empty();
                let compose_elapsed = v.tile_overlay.last_compose.elapsed().as_millis() >= 200;
                if v.tile_overlay.dirty && !v.tile_overlay.needed_tiles.is_empty() && (bounds_changed || all_loaded || compose_elapsed) {
                    v.tile_overlay.dirty = false;
                    v.tile_overlay.last_compose = std::time::Instant::now();
                    let needed = v.tile_overlay.needed_tiles.clone();
                    let y_min = needed.iter().map(|c| c.y).min().unwrap();
                    let y_max = needed.iter().map(|c| c.y).max().unwrap();
                    let x_org = v.tile_overlay.tile_x_origin;
                    let z = needed[0].z;
                    let n = 1u32 << z;
                    let col_of = |x: u32| -> u32 {
                        if x >= x_org { x - x_org } else { n - x_org + x }
                    };
                    let cols = needed.iter().map(|c| col_of(c.x)).max().unwrap() + 1;
                    let rows = y_max - y_min + 1;
                    let tile_size = 256u32;
                    let tex_w = cols * tile_size;
                    let tex_h = rows * tile_size;
                    let pixel_count = (tex_w * tex_h) as usize;
                    v.tile_overlay.compose_buffer.resize(pixel_count, [0u8, 0, 0, 0]);
                    v.tile_overlay.compose_buffer.iter_mut().for_each(|p| *p = [0, 0, 0, 0]);
                    let pixels = &mut v.tile_overlay.compose_buffer;
                    for coord in &needed {
                        let dst_ox = (col_of(coord.x) * tile_size) as usize;
                        let dst_oy = ((coord.y - y_min) * tile_size) as usize;
                        if let Some(found_z) = v.tile_overlay.tile_tree.best_tile_zoom(coord) {
                            let d = coord.z - found_z;
                            if d == 0 {
                                let entry = v.tile_overlay.tile_tree.get_tile_at(coord).unwrap();
                                let tw = entry.width.min(tile_size) as usize;
                                let th = entry.height.min(tile_size) as usize;
                                for row in 0..th {
                                    for col in 0..tw {
                                        let src_idx = row * entry.width as usize + col;
                                        let dst_idx = (dst_oy + row) * tex_w as usize + (dst_ox + col);
                                        if src_idx < entry.pixels.len() && dst_idx < pixels.len() {
                                            let [r, g, b] = entry.pixels[src_idx];
                                            pixels[dst_idx] = [r, g, b, 255];
                                        }
                                    }
                                }
                            } else {
                                let anc = TileCoord { x: coord.x >> d, y: coord.y >> d, z: found_z };
                                let entry = v.tile_overlay.tile_tree.get_tile_at(&anc).unwrap();
                                let scale = 1u32 << d;
                                let sub_x = coord.x & (scale - 1);
                                let sub_y = coord.y & (scale - 1);
                                let src_ox = sub_x as f64 * entry.width as f64 / scale as f64;
                                let src_oy = sub_y as f64 * entry.height as f64 / scale as f64;
                                let src_w = entry.width as f64 / scale as f64;
                                let src_h = entry.height as f64 / scale as f64;
                                for row in 0..256usize {
                                    for col in 0..256usize {
                                        let sr = (src_oy + row as f64 * src_h / 256.0) as usize;
                                        let sc = (src_ox + col as f64 * src_w / 256.0) as usize;
                                        let ew = entry.width as usize;
                                        let eh = entry.height as usize;
                                        let si = sr.min(eh - 1) * ew + sc.min(ew - 1);
                                        let di = (dst_oy + row) * tex_w as usize + (dst_ox + col);
                                        if si < entry.pixels.len() && di < pixels.len() {
                                            let [r, g, b] = entry.pixels[si];
                                            pixels[di] = [r, g, b, 255];
                                        }
                                    }
                                }
                            }
                        }

                    }

                    let (top_left_lon, top_left_lat) = tile_to_lon_lat(&TileCoord { x: x_org, y: y_min, z });
                    let right_x = x_org + cols;
                    let (bot_right_lon, bot_right_lat) = if right_x <= n {
                        tile_to_lon_lat(&TileCoord { x: right_x, y: y_max + 1, z })
                    } else {
                        let (lon, lat) = tile_to_lon_lat(&TileCoord { x: right_x - n, y: y_max + 1, z });
                        (lon + 360.0, lat)
                    };

                    let new_bounds = DetailBounds {
                        min_lon: top_left_lon,
                        max_lon: bot_right_lon,
                        min_lat: bot_right_lat.to_radians(),
                        max_lat: top_left_lat.to_radians(),
                    };

                    if let Some(gl) = frame.gl() {
                        unsafe {
                            let flat_pixels: &[u8] = std::slice::from_raw_parts(
                                v.tile_overlay.compose_buffer.as_ptr() as *const u8,
                                v.tile_overlay.compose_buffer.len() * 4,
                            );

                            let reuse = v.tile_overlay.detail_texture.as_ref()
                                .and_then(|dt| dt.gl_texture.filter(|_| dt.width == tex_w && dt.height == tex_h));

                            if let Some(existing_tex) = reuse {
                                gl.bind_texture(glow::TEXTURE_2D, Some(existing_tex));
                                gl.tex_sub_image_2d(
                                    glow::TEXTURE_2D,
                                    0,
                                    0, 0,
                                    tex_w as i32, tex_h as i32,
                                    glow::RGBA,
                                    glow::UNSIGNED_BYTE,
                                    glow::PixelUnpackData::Slice(Some(flat_pixels)),
                                );
                                v.tile_overlay.detail_texture = Some(DetailTexture {
                                    width: tex_w,
                                    height: tex_h,
                                    bounds: new_bounds,
                                    gl_texture: Some(existing_tex),
                                });
                            } else {
                                if let Some(old) = &v.tile_overlay.detail_texture {
                                    if let Some(gl_tex) = old.gl_texture {
                                        gl.delete_texture(gl_tex);
                                    }
                                }
                                let texture = gl.create_texture().expect("create detail texture");
                                gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                                gl.tex_image_2d(
                                    glow::TEXTURE_2D,
                                    0,
                                    glow::RGBA as i32,
                                    tex_w as i32, tex_h as i32,
                                    0,
                                    glow::RGBA,
                                    glow::UNSIGNED_BYTE,
                                    glow::PixelUnpackData::Slice(Some(flat_pixels)),
                                );
                                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
                                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
                                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
                                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
                                v.tile_overlay.detail_texture = Some(DetailTexture {
                                    width: tex_w,
                                    height: tex_h,
                                    bounds: new_bounds,
                                    gl_texture: Some(texture),
                                });
                            }
                        }
                    }
                }
                }
            }
        }

        let dt = ctx.input(|i| i.stable_dt) as f64;
        v.real_time += dt;

        ctx.request_repaint();

        for tab in &mut v.tabs {
            let sim_seconds = if tab.settings.animate {
                tab.settings.time += dt * tab.settings.speed;
                dt * tab.settings.speed
            } else {
                0.0
            };
            if sim_seconds.abs() < 1e-9 { continue; }
            for planet in &mut tab.planets {
                let mu = planet.celestial_body.mu();
                let r_planet = planet.celestial_body.radius_km();
                for cons in &mut planet.constellations {
                    if cons.drag_enabled && cons.altitude_km > 0.0 {
                        let h = cons.altitude_km;
                        let r = r_planet + h;
                        let scale_height = 60.0;
                        let rho_ref = 2.8e-12;
                        let h_ref = 400.0;
                        let rho = rho_ref * ((h_ref - h) / scale_height).exp();
                        let v_ms = (mu / r).sqrt() * 1000.0;
                        let a_m = r * 1000.0;
                        let dh_ms = -rho * v_ms * a_m / cons.ballistic_coeff;
                        cons.altitude_km = (h + dh_ms * sim_seconds / 1000.0).max(0.0);
                    }
                }
            }
        }

        if v.auto_cycle_tabs && v.tabs.len() > 1 {
            v.last_cycle_time += dt;
            if v.last_cycle_time >= v.cycle_interval {
                v.last_cycle_time = 0.0;
                let tab_data: Vec<(egui_dock::SurfaceIndex, egui_dock::NodeIndex, usize)> = self.dock_state.iter_all_tabs()
                    .map(|((s, n), &idx)| (s, n, idx))
                    .collect();
                if let Some(current_pos) = tab_data.iter().position(|(_, _, idx)| *idx == active_tab_idx) {
                    let next_pos = (current_pos + 1) % tab_data.len();
                    let (surface, node, _) = tab_data[next_pos];
                    self.dock_state.set_focused_node_and_surface((surface, node));
                }
            }
        }

        let tab_time = v.tabs.get(active_tab_idx).map(|t| t.settings.time).unwrap_or(0.0);
        let sim_time = v.start_timestamp + Duration::seconds(tab_time as i64);
        let gmst = greenwich_mean_sidereal_time(sim_time);
        v.current_gmst = gmst;

        let new_follow_rotation: Option<Matrix3<f64>> = 'follow: {
            let Some(tab) = v.tabs.get(active_tab_idx) else { break 'follow None };
            if !tab.settings.follow_satellite { break 'follow None; }
            let Some(planet) = tab.planets.first() else { break 'follow None };
            let Some(cam) = planet.satellite_cameras.last() else { break 'follow None };

            let set_follow_rotation = |radial: Vector3<f64>, velocity_dir: Vector3<f64>| {
                let z_axis = radial;
                let vel_proj = velocity_dir - radial * velocity_dir.dot(&radial);
                let y_axis = vel_proj.normalize();
                let x_axis = y_axis.cross(&z_axis).normalize();
                Matrix3::new(
                    x_axis.x, x_axis.y, x_axis.z,
                    y_axis.x, y_axis.y, y_axis.z,
                    z_axis.x, z_axis.y, z_axis.z,
                )
            };

            if cam.constellation_idx == usize::MAX {
                let propagation_minutes = v.start_timestamp.timestamp() as f64 / 60.0 + tab_time / 60.0;
                for preset in TlePreset::ALL.iter() {
                    let Some((selected, state, _)) = planet.tle_selections.get(preset) else { continue };
                    if !*selected { continue; }
                    let TleLoadState::Loaded { satellites, .. } = state else { continue };
                    let Some(sat) = satellites.get(cam.sat_index) else { continue };
                    let minutes_since_epoch = propagation_minutes - sat.epoch_minutes;
                    let Ok(prediction) = sat.constants.propagate(sgp4::MinutesSinceEpoch(minutes_since_epoch)) else { continue };
                    let radial = Vector3::new(prediction.position[0], prediction.position[2], -prediction.position[1]).normalize();
                    let velocity_dir = Vector3::new(prediction.velocity[0], prediction.velocity[2], -prediction.velocity[1]).normalize();
                    break 'follow Some(set_follow_rotation(radial, velocity_dir));
                }
                None
            } else {
                let Some(cons) = planet.constellations.get(cam.constellation_idx) else { break 'follow None };
                let wc = cons.constellation(
                    planet.celestial_body.radius_km(),
                    planet.celestial_body.mu(),
                    planet.celestial_body.j2(),
                    planet.celestial_body.equatorial_radius_km(),
                );
                let pos_now = wc.satellite_positions(tab_time);
                let pos_next = wc.satellite_positions(tab_time + 0.1);
                let Some(sat) = pos_now.iter().find(|s| s.plane == cam.plane && s.sat_index == cam.sat_index) else { break 'follow None };
                let Some(sat2) = pos_next.iter().find(|s| s.plane == cam.plane && s.sat_index == cam.sat_index) else { break 'follow None };
                let radial = Vector3::new(sat.x, sat.y, sat.z).normalize();
                let velocity_dir = Vector3::new(sat2.x - sat.x, sat2.y - sat.y, sat2.z - sat.z).normalize();
                Some(set_follow_rotation(radial, velocity_dir))
            }
        };
        if let Some(new_rot) = new_follow_rotation {
            if let Some(tab) = v.tabs.get_mut(active_tab_idx) {
                tab.settings.rotation = new_rot;
            }
        }

        #[cfg(target_arch = "wasm32")]
        TEXTURE_RESULT.with(|cell| {
            for (body, result) in cell.borrow_mut().drain(..) {
                match result {
                    Ok(texture) => {
                        let texture = Arc::new(texture);
                        v.planet_textures.insert(body, texture.clone());
                        v.texture_load_state = TextureLoadState::Loaded(texture);
                        v.planet_image_handles.remove(&body);
                    }
                    Err(e) => {
                        v.texture_load_state = TextureLoadState::Failed(e);
                    }
                }
            }
        });

        #[cfg(target_arch = "wasm32")]
        {
            CLOUD_TEXTURE_RESULT.with(|cell| {
                if let Some((res, result)) = cell.borrow_mut().take() {
                    if let Ok(texture) = result {
                        v.cloud_textures.insert(res, Arc::new(texture));
                    }
                    v.cloud_texture_loading = false;
                }
            });
            STAR_TEXTURE_RESULT.with(|cell| {
                if let Some(result) = cell.borrow_mut().take() {
                    if let Ok(texture) = result {
                        v.star_texture = Some(Arc::new(texture));
                    }
                    v.star_texture_loading = false;
                }
            });
            MILKY_WAY_TEXTURE_RESULT.with(|cell| {
                if let Some(result) = cell.borrow_mut().take() {
                    if let Ok(texture) = result {
                        v.milky_way_texture = Some(Arc::new(texture));
                    }
                    v.milky_way_texture_loading = false;
                }
            });
            NIGHT_TEXTURE_RESULT.with(|cell| {
                if let Some(result) = cell.borrow_mut().take() {
                    if let Ok(texture) = result {
                        v.night_texture = Some(Arc::new(texture));
                    }
                    v.night_texture_loading = false;
                }
            });
            TLE_FETCH_RESULT.with(|cell| {
                for (preset, result) in cell.borrow_mut().drain(..) {
                    for tab in &mut v.tabs {
                        for planet in &mut tab.planets {
                            if let Some((_, state, _)) = planet.tle_selections.get_mut(&preset) {
                                if matches!(state, TleLoadState::Loading) {
                                    *state = match result.clone() {
                                        Ok(satellites) => TleLoadState::Loaded {
                                            satellites,
                                        },
                                        Err(e) => TleLoadState::Failed(e),
                                    };
                                }
                            }
                        }
                    }
                }
            });
        }

        if !v.use_gpu_rendering {
            let (tab_rotation, tab_time, tab_animate, tab_earth_fixed) = v.tabs.get(active_tab_idx)
                .map(|t| (t.settings.rotation, t.settings.time, t.settings.animate, t.settings.earth_fixed_camera))
                .unwrap_or((Matrix3::identity(), 0.0, false, false));

            let rotation_changed = v.last_rotation.is_none_or(|r| r != tab_rotation);
            let resolution_changed = v.last_resolution != v.earth_resolution;
            let time_changed = tab_animate;

            for key in &bodies_needed {
                let texture_missing = !v.planet_image_handles.contains_key(key);
                let need_rerender = rotation_changed || resolution_changed || texture_missing || time_changed;
                if need_rerender {
                    if let Some(texture) = v.planet_textures.get(key) {
                        let body_rotation = body_rotation_angle(key.0, tab_time, v.current_gmst);
                        let cos_a = body_rotation.cos();
                        let sin_a = body_rotation.sin();
                        let body_y_rotation = Matrix3::new(
                            cos_a, 0.0, sin_a,
                            0.0, 1.0, 0.0,
                            -sin_a, 0.0, cos_a,
                        );
                        let body_combined = if tab_earth_fixed {
                            tab_rotation
                        } else {
                            tab_rotation * body_y_rotation
                        };
                        let flattening = key.0.flattening();
                        let render_size = key.2.cpu_render_size();
                        let image = texture.render_sphere(render_size, &body_combined, flattening);
                        let handle = ctx.load_texture(
                            format!("planet_{:?}_{:?}", key.0, key.1),
                            image,
                            egui::TextureOptions::LINEAR,
                        );
                        v.planet_image_handles.insert(*key, handle);
                    }
                }
            }
            if rotation_changed {
                v.last_rotation = Some(tab_rotation);
            }
            if resolution_changed {
                v.last_resolution = v.earth_resolution;
            }
        }

        {
            let show_ss = v.tabs.get(active_tab_idx)
                .map(|t| t.settings.show_solar_system)
                .unwrap_or(false);
            let tab_time = v.tabs.get(active_tab_idx)
                .map(|t| t.settings.time)
                .unwrap_or(0.0);

            if show_ss || v.show_planet_sizes {
                #[cfg(not(target_arch = "wasm32"))]
                for &body in &CelestialBody::ALL {
                    let key = (body, Skin::Default, TextureResolution::R512);
                    if !v.planet_textures.contains_key(&key) {
                        if let Some(filename) = Skin::Default.filename(body, TextureResolution::R512) {
                            if let Ok(bytes) = std::fs::read(crate::texture::asset_path(filename)) {
                                if let Ok(tex) = crate::texture::EarthTexture::from_bytes(&bytes) {
                                    v.planet_textures.insert(key, Arc::new(tex));
                                }
                            }
                        }
                    }
                    if body.ring_params().is_some() && !v.ring_textures.contains_key(&body) {
                        if let Some((ring_path, _, _)) = body.ring_params() {
                            if let Ok(ring_bytes) = std::fs::read(crate::texture::asset_path(ring_path)) {
                                if let Ok(ring_tex) = crate::texture::RingTexture::from_bytes(&ring_bytes) {
                                    v.ring_textures.insert(body, Arc::new(ring_tex));
                                }
                            }
                        }
                    }
                }

                if v.asteroid_sprite.is_none() {
                    let size = 16usize;
                    let center = size as f64 / 2.0;
                    let radius = center - 1.0;
                    let mut pixels = vec![egui::Color32::TRANSPARENT; size * size];
                    for py in 0..size {
                        for px in 0..size {
                            let dx = px as f64 - center;
                            let dy = py as f64 - center;
                            let dist = (dx * dx + dy * dy).sqrt();
                            if dist < radius {
                                let z = (radius * radius - dist * dist).sqrt() / radius;
                                let noise = ((px * 7 + py * 13) % 17) as f64 / 34.0;
                                let shade = (0.25 + 0.55 * z + noise * 0.2).clamp(0.0, 1.0);
                                let base = (140.0 * shade) as u8;
                                let r = base.saturating_add(((px * 3 + py * 5) % 11) as u8);
                                let g = base.saturating_add(((px * 5 + py * 3) % 9) as u8);
                                let b = base;
                                let edge = ((radius - dist) / 1.5).clamp(0.0, 1.0);
                                let alpha = (edge * 255.0) as u8;
                                pixels[py * size + px] = egui::Color32::from_rgba_unmultiplied(r, g, b, alpha);
                            }
                        }
                    }
                    let image = egui::ColorImage {
                        size: [size, size],
                        pixels,
                        source_size: egui::Vec2::ZERO,
                    };
                    v.asteroid_sprite = Some(ctx.load_texture(
                        "asteroid_sprite",
                        image,
                        egui::TextureOptions::LINEAR,
                    ));
                }

                let render_interval = if v.show_planet_sizes { 200 } else if v.ss_auto_zoom { 5000 } else { 1000 };
                let needs_render = v.solar_system_handles.is_empty()
                    || v.ss_last_render_instant.map_or(true, |t| t.elapsed().as_millis() > render_interval);
                if needs_render {
                    v.ss_last_render_instant = Some(std::time::Instant::now());
                    let tilt = 30.0_f64.to_radians();
                    let cos_t = tilt.cos();
                    let sin_t = tilt.sin();
                    let tilt_mat = Matrix3::new(
                        1.0, 0.0, 0.0,
                        0.0, cos_t, -sin_t,
                        0.0, sin_t, cos_t,
                    );

                    let mut sorted_bodies: Vec<CelestialBody> = CelestialBody::ALL.to_vec();
                    sorted_bodies.sort_by(|a, b| b.radius_km().partial_cmp(&a.radius_km()).unwrap());
                    let focus_idx = (v.planet_sizes_t as usize).min(sorted_bodies.len().saturating_sub(1));
                    let focus_radius = sorted_bodies[focus_idx].radius_km();
                    let max_render = if v.show_planet_sizes { 192 } else { 64 };

                    let gpu_ok = frame.gl().is_some() && v.sphere_renderer.is_some();
                    if gpu_ok {
                        let gl = frame.gl().unwrap();
                        let sr = v.sphere_renderer.as_ref().unwrap();
                        let mut renderer = sr.lock();
                        for body in CelestialBody::ALL {
                            let key = (body, Skin::Default, TextureResolution::R512);
                            if let Some(tex) = v.planet_textures.get(&key) {
                                renderer.upload_texture(gl, key, tex);
                            }
                            if let Some(ring_tex) = v.ring_textures.get(&body) {
                                renderer.upload_ring_texture(gl, body, ring_tex);
                            }
                        }
                        for body in CelestialBody::ALL {
                            let key = (body, Skin::Default, TextureResolution::R512);
                            let ratio = (body.radius_km() / focus_radius).min(1.0);
                            let body_render_size = if ratio > 0.1 { max_render } else { ((max_render as f64 * ratio * 10.0) as usize).clamp(32, max_render) };
                            let body_rot = body_rotation_angle(body, tab_time, v.current_gmst)
                                + 30.0_f64.to_radians();
                            let cos_a = body_rot.cos();
                            let sin_a = body_rot.sin();
                            let y_rot = Matrix3::new(
                                cos_a, 0.0, sin_a,
                                0.0, 1.0, 0.0,
                                -sin_a, 0.0, cos_a,
                            );
                            let combined = tilt_mat * y_rot;
                            let inv_rotation = combined.transpose();
                            let image = renderer.render_to_image(gl, key, &inv_rotation, body.flattening(), body_render_size);
                            let handle = ctx.load_texture(
                                format!("ss_{:?}", body),
                                image,
                                egui::TextureOptions::LINEAR,
                            );
                            v.solar_system_handles.insert(body, handle);
                        }
                    } else {
                        for body in CelestialBody::ALL {
                            let key = (body, Skin::Default, TextureResolution::R512);
                            if let Some(texture) = v.planet_textures.get(&key) {
                                let ratio = (body.radius_km() / focus_radius).min(1.0);
                                let body_render_size = ((max_render as f64 * ratio) as usize).clamp(32, max_render);
                                let body_rot = body_rotation_angle(body, tab_time, v.current_gmst)
                                    + 30.0_f64.to_radians();
                                let cos_a = body_rot.cos();
                                let sin_a = body_rot.sin();
                                let y_rot = Matrix3::new(
                                    cos_a, 0.0, sin_a,
                                    0.0, 1.0, 0.0,
                                    -sin_a, 0.0, cos_a,
                                );
                                let combined = tilt_mat * y_rot;
                                let ring_tex = v.ring_textures.get(&body).map(|r| r.as_ref());
                                let image = texture.render_sphere_with_rings(
                                    body_render_size, &combined, body.flattening(), body, ring_tex,
                                );
                                let handle = ctx.load_texture(
                                    format!("ss_{:?}", body),
                                    image,
                                    egui::TextureOptions::LINEAR,
                                );
                                v.solar_system_handles.insert(body, handle);
                            }
                        }
                    }
                }
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            if matches!(v.asteroid_state, crate::solar_system::AsteroidLoadState::NotLoaded) {
                v.asteroid_state = crate::solar_system::AsteroidLoadState::Loading;
                let (tx, rx) = mpsc::channel();
                v.asteroid_rx = Some(rx);
                std::thread::spawn(move || {
                    let _ = tx.send(crate::solar_system::fetch_asteroids());
                });
            }
            if let Some(rx) = &v.asteroid_rx {
                if let Ok(result) = rx.try_recv() {
                    match result {
                        Ok(data) => v.asteroid_state = crate::solar_system::AsteroidLoadState::Loaded(data),
                        Err(e) => v.asteroid_state = crate::solar_system::AsteroidLoadState::Failed(e),
                    }
                    v.asteroid_rx = None;
                }
            }
        }

        if self.viewer.show_info {
            egui::Window::new("Info")
                .open(&mut self.viewer.show_info)
                .default_width(1000.0)
                .show(ctx, |ui| {
                    ui.columns(3, |cols| {
                        cols[0].heading("Celestial Bodies");
                        let mut bodies: Vec<_> = CelestialBody::ALL.iter().collect();
                        bodies.sort_by(|a, b| b.radius_km().partial_cmp(&a.radius_km()).unwrap());
                        egui::Grid::new("bodies_grid").striped(true).show(&mut cols[0], |ui| {
                            ui.strong("Body");
                            ui.strong("Radius (km)");
                            ui.strong("mu (km\u{00b3}/s\u{00b2})");
                            ui.strong("J2 (x10\u{207b}\u{00b3})");
                            ui.strong("Rotation (h)");
                            ui.end_row();
                            for body in bodies {
                                ui.label(body.label());
                                ui.label(format!("{:.0}", body.radius_km()));
                                ui.label(format!("{:.0}", body.mu()));
                                ui.label(format!("{:.4}", body.j2() * 1000.0));
                                let rot = body.rotation_period_hours();
                                if rot.abs() > 100.0 {
                                    ui.label(format!("{:.0}", rot));
                                } else {
                                    ui.label(format!("{:.1}", rot));
                                }
                                ui.end_row();
                            }
                        });

                        cols[0].add_space(10.0);
                        cols[0].heading("Walker Constellation");
                        cols[0].label("Notation: i:T/P/F");
                        egui::Grid::new("walker_grid").show(&mut cols[0], |ui| {
                            ui.monospace("i");
                            ui.label("Inclination (degrees from equator)");
                            ui.end_row();
                            ui.monospace("T");
                            ui.label("Total number of satellites");
                            ui.end_row();
                            ui.monospace("P");
                            ui.label("Number of orbital planes");
                            ui.end_row();
                            ui.monospace("F");
                            ui.label("Phasing factor (0 to P-1)");
                            ui.end_row();
                        });
                        cols[0].add_space(5.0);
                        cols[0].label("Types:");
                        cols[0].label("  Delta: 360\u{00b0} RAAN spread (co-rotating)");
                        cols[0].label("  Star: 180\u{00b0} RAAN spread (counter-rotating)");
                        cols[0].label("Phasing offset per plane:");
                        cols[0].monospace("  d = F x 360 / T");

                        cols[0].add_space(10.0);
                        cols[0].heading("Orbital Parameters");
                        egui::Grid::new("params_grid").show(&mut cols[0], |ui| {
                            ui.monospace("RAAN0");
                            ui.label("Right ascension of first plane");
                            ui.end_row();
                            ui.monospace("Delta");
                            ui.label("RAAN spacing between planes");
                            ui.end_row();
                            ui.monospace("Ecc");
                            ui.label("Eccentricity (0 = circular)");
                            ui.end_row();
                            ui.monospace("Omega");
                            ui.label("Argument of periapsis");
                            ui.end_row();
                            ui.monospace("Drag");
                            ui.label("Atmospheric drag coefficient");
                            ui.end_row();
                        });

                        cols[1].heading("Orbital Mechanics");
                        cols[1].label("Orbital velocity:");
                        cols[1].monospace("  v = sqrt(mu / r)");
                        cols[1].label("Orbital period:");
                        cols[1].monospace("  T = 2pi sqrt(r^3 / mu)");
                        cols[1].label("One-way latency:");
                        cols[1].monospace("  t = distance / c");
                        cols[1].label("J2 RAAN precession:");
                        cols[1].monospace("  dO/dt = -1.5 J2 (Re/a)^2 n cos(i)");
                        cols[1].label("where Re = equatorial radius, a = semi-major axis,");
                        cols[1].label("n = mean motion, i = inclination.");

                        cols[1].add_space(10.0);
                        cols[1].heading("Satellite Constellations");
                        egui::Grid::new("constellations_grid").striped(true).show(&mut cols[1], |ui| {
                            ui.strong("Name");
                            ui.strong("Config");
                            ui.strong("Alt");
                            ui.strong("Inc");
                            ui.end_row();
                            ui.label("Starlink");
                            ui.label("22x72");
                            ui.label("550km");
                            ui.label("53\u{00b0}");
                            ui.end_row();
                            ui.label("OneWeb");
                            ui.label("49x36");
                            ui.label("1200km");
                            ui.label("87.9\u{00b0}");
                            ui.end_row();
                            ui.label("Iridium");
                            ui.label("11x6");
                            ui.label("780km");
                            ui.label("86.4\u{00b0}");
                            ui.end_row();
                            ui.label("Kuiper");
                            ui.label("34x34");
                            ui.label("630km");
                            ui.label("51.9\u{00b0}");
                            ui.end_row();
                            ui.label("Telesat");
                            ui.label("13x6");
                            ui.label("1015km");
                            ui.label("99\u{00b0}");
                            ui.end_row();
                        });

                        cols[1].add_space(10.0);
                        cols[1].heading("Constants");
                        egui::Grid::new("constants_grid").show(&mut cols[1], |ui| {
                            ui.monospace("mu");
                            ui.label("Gravitational parameter = G x M");
                            ui.end_row();
                            ui.monospace("J2");
                            ui.label("Oblateness coefficient");
                            ui.end_row();
                            ui.monospace("c");
                            ui.label("Speed of light (299,792 km/s)");
                            ui.end_row();
                        });

                        cols[2].heading("Live TLE Data (CelesTrak)");
                        egui::ScrollArea::vertical().max_height(500.0).show(&mut cols[2], |ui| {
                            for (cat, entries) in [
                                ("Comms", vec![
                                    ("Starlink", "SpaceX LEO broadband", "Operational"),
                                    ("OneWeb", "LEO broadband", "Operational"),
                                    ("Kuiper", "Amazon LEO broadband", "Deploying"),
                                    ("GEO", "Geostationary satellites", "Operational"),
                                    ("Intelsat", "GEO comms operator", "Operational"),
                                    ("SES", "GEO/MEO comms operator", "Operational"),
                                    ("Iridium", "Original voice/data", "Decommissioned"),
                                    ("Iridium NEXT", "2nd-gen Iridium", "Operational"),
                                    ("Globalstar", "LEO voice/data", "Operational"),
                                    ("Orbcomm", "LEO IoT/M2M messaging", "Operational"),
                                    ("Molniya", "Russian HEO comms", "Decommissioned"),
                                    ("Swarm", "SpaceX IoT CubeSats", "Operational"),
                                    ("Amateur", "Amateur radio satellites", "Operational"),
                                    ("X-Comm", "Experimental comms", "Operational"),
                                    ("Other Comm", "Miscellaneous comms", "Operational"),
                                    ("SatNOGS", "Open-source ground stn network", "Operational"),
                                ]),
                                ("Navigation", vec![
                                    ("GPS", "US navigation (MEO)", "Operational"),
                                    ("Galileo", "EU navigation (MEO)", "Operational"),
                                    ("GLONASS", "Russian navigation (MEO)", "Operational"),
                                    ("Beidou", "Chinese navigation (MEO/GEO)", "Operational"),
                                    ("GNSS", "All GNSS combined", "Operational"),
                                    ("SBAS", "Augmentation systems (GEO)", "Operational"),
                                    ("NNSS", "Navy navigation (legacy)", "Decommissioned"),
                                    ("Musson", "Russian geodetic/nav", "Operational"),
                                ]),
                                ("Observation", vec![
                                    ("Weather", "Weather satellites", "Operational"),
                                    ("NOAA", "US weather (polar)", "Operational"),
                                    ("GOES", "US weather (GEO)", "Operational"),
                                    ("Earth Res.", "Earth resource imaging", "Operational"),
                                    ("SARSAT", "Search & rescue beacons", "Operational"),
                                    ("DMC", "Disaster monitoring", "Operational"),
                                    ("TDRSS", "NASA tracking & data relay", "Operational"),
                                    ("ARGOS", "Environmental data collection", "Operational"),
                                    ("Planet", "Earth-imaging CubeSats", "Operational"),
                                    ("Spire", "Weather/AIS CubeSats", "Operational"),
                                ]),
                                ("Other", vec![
                                    ("Stations", "ISS & space stations", "Operational"),
                                    ("Last 30 Days", "Recently launched", "Operational"),
                                    ("100 Brightest", "Visually brightest", "Operational"),
                                    ("Active", "All active satellites", "Operational"),
                                    ("Analyst", "Analyst-tracked objects", "Operational"),
                                    ("Science", "Scientific satellites", "Operational"),
                                    ("Geodetic", "Geodetic satellites", "Operational"),
                                    ("Engineering", "Engineering satellites", "Operational"),
                                    ("Education", "Educational satellites", "Operational"),
                                    ("Military", "Military satellites", "Operational"),
                                    ("Radar Cal.", "Radar calibration", "Operational"),
                                    ("CubeSats", "CubeSat catalog", "Operational"),
                                    ("Misc", "Uncategorized objects", "Operational"),
                                ]),
                                ("Debris", vec![
                                    ("Fengyun 1C", "2007 ASAT test (~1800)", ""),
                                    ("Cosmos 2251", "2009 collision (~580)", ""),
                                    ("Iridium 33", "2009 collision (~110)", ""),
                                    ("Cosmos 1408", "2021 ASAT test", ""),
                                ]),
                            ] {
                                ui.strong(cat);
                                egui::Grid::new(format!("tle_{}_grid", cat)).striped(true).show(ui, |ui| {
                                    for (name, desc, status) in &entries {
                                        ui.label(*name);
                                        ui.label(*desc);
                                        if !status.is_empty() {
                                            let color = match *status {
                                                "Operational" => egui::Color32::from_rgb(80, 200, 80),
                                                "Deploying" => egui::Color32::from_rgb(200, 200, 80),
                                                _ => egui::Color32::from_rgb(200, 80, 80),
                                            };
                                            ui.colored_label(color, *status);
                                        }
                                        ui.end_row();
                                    }
                                });
                                ui.add_space(4.0);
                            }
                        });
                    });
                });
        }


        if self.viewer.show_side_panel {
            egui::SidePanel::left("settings_panel")
                .resizable(true)
                .default_width(200.0)
                .show_separator_line(false)
                .frame(egui::Frame::side_top_panel(ctx.style().as_ref()).inner_margin(4.0).stroke(egui::Stroke::NONE))
                .show(ctx, |ui| {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.add_space(4.0);
                        ui.strong("Settings");
                        if ui.button("[Info]").clicked() {
                            self.viewer.show_info = !self.viewer.show_info;
                        }
                        if ui.button("[Demo]").clicked() {
                            self.setup_demo();
                        }
                        if ui.small_button("−").clicked() {
                            self.viewer.show_side_panel = false;
                        }
                    });
                    ui.separator();
                    let mut scroll = egui::ScrollArea::vertical().id_salt("settings_scroll");
                    if self.first_frame {
                        scroll = scroll.vertical_scroll_offset(0.0);
                    }
                    scroll.show(ui, |ui| {
                        self.viewer.show_settings(ui);
                    });
                });
        }

        let mut dock_style = egui_dock::Style::from_egui(ctx.style().as_ref());
        dock_style.main_surface_border_stroke = egui::Stroke::NONE;
        let full_tab_bar_height = dock_style.tab_bar.height;
        let ui_visible = if self.viewer.auto_hide_tab_bar {
            let hover_zone = full_tab_bar_height + 50.0;
            ctx.input(|i| {
                i.pointer.hover_pos().map_or(false, |p| p.y < hover_zone)
            })
        } else {
            true
        };
        self.viewer.ui_visible = ui_visible;
        if !ui_visible {
            dock_style.tab_bar.height = 0.0;
        }
        let tab_bar_height = dock_style.tab_bar.height;
        let mut dock = DockArea::new(&mut self.dock_state)
            .style(dock_style);
        if ui_visible {
            dock = dock.show_add_buttons(true);
        }
        dock.show(ctx, &mut self.viewer);

        if !self.viewer.show_side_panel && ui_visible {
            egui::Area::new(egui::Id::new("settings_btn"))
                .fixed_pos(egui::pos2(4.0, (tab_bar_height - 16.0) / 2.0))
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    if ui.small_button("+").clicked() {
                        self.viewer.show_side_panel = true;
                    }
                });
        }

        if let Some(new_idx) = self.viewer.pending_add_tab.take() {
            self.dock_state.push_to_focused_leaf(new_idx);
        }

        for tab in &self.viewer.tabs {
            if !tab.settings.show_camera_windows { continue; }
            let tab_time = tab.settings.time;
            let coverage_angle = tab.settings.coverage_angle;
            for planet in &tab.planets {
                let pr = planet.celestial_body.radius_km();
                let pm = planet.celestial_body.mu();
                let pj2 = planet.celestial_body.j2();
                let peq = planet.celestial_body.equatorial_radius_km();
                let texture = self.viewer.planet_textures.get(&(planet.celestial_body, planet.skin, self.viewer.texture_resolution));

                let body_rot = body_rotation_angle(planet.celestial_body, tab_time, self.viewer.current_gmst);
                let cos_a = body_rot.cos();
                let sin_a = body_rot.sin();
                for camera in &planet.satellite_cameras {
                    let sat_data = planet.constellations.get(camera.constellation_idx).and_then(|cons| {
                        let wc = cons.constellation(pr, pm, pj2, peq);
                        let positions = wc.satellite_positions(tab_time);
                        positions.iter()
                            .find(|s| s.plane == camera.plane && s.sat_index == camera.sat_index)
                            .map(|s| {
                                let bx = s.x * cos_a - s.z * sin_a;
                                let bz = s.x * sin_a + s.z * cos_a;
                                let ground_lon = (-bz).atan2(bx).to_degrees();
                                (s.lat, ground_lon, cons.altitude_km, texture, pr)
                            })
                    });

                    if let Some((lat, lon, altitude_km, texture, planet_radius)) = sat_data {
                        let win_response = egui::Window::new(format!("{}: {}", planet.name, camera.label))
                            .id(egui::Id::new(format!("sat_cam_{}_{}", planet.name, camera.id)))
                            .title_bar(true)
                            .collapsible(false)
                            .default_size([200.0, 220.0])
                            .show(ctx, |ui| {
                                if let Some(tex) = texture {
                                    draw_satellite_camera(
                                        ui,
                                        camera.id,
                                        lat,
                                        lon,
                                        altitude_km,
                                        coverage_angle,
                                        tex,
                                        planet_radius,
                                    );
                                }
                            });

                        if let (Some(screen_pos), Some(win_resp)) = (camera.screen_pos, win_response) {
                            let win_rect = win_resp.response.rect;
                            let win_center = win_rect.left_center();
                            ctx.layer_painter(egui::LayerId::new(
                                egui::Order::Middle,
                                egui::Id::new("sat_lines"),
                            ))
                            .line_segment(
                                [screen_pos, win_center],
                                egui::Stroke::new(1.5, egui::Color32::WHITE),
                            );
                        }
                    }
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            use crate::config::ShareableConfig;
            let active = self.viewer.active_tab_idx;
            if let Some(planet) = self.viewer.tabs.get(active).and_then(|t| t.planets.first()) {
                let hash = if planet.constellations.is_empty() {
                    String::new()
                } else {
                    ShareableConfig::from_planet(planet).to_url_hash()
                };
                if hash != self.viewer.last_url_hash {
                    self.viewer.last_url_hash = hash.clone();
                    if let Some(window) = web_sys::window() {
                        if let Ok(history) = window.history() {
                            let _ = history.replace_state_with_url(
                                &wasm_bindgen::JsValue::NULL,
                                "",
                                Some(if hash.is_empty() { "." } else { &hash }),
                            );
                        }
                    }
                }
            }
        }

        self.first_frame = false;
    }

    fn on_exit(&mut self, gl: Option<&glow::Context>) {
        if let Some(gl) = gl {
            if let Some(ref renderer) = self.viewer.sphere_renderer {
                renderer.lock().destroy(gl);
            }
        }
    }
}
