//! Opt-in replay and measurements for the native ESRI rendering path.

use crate::{tile::TileCoord, ViewerState};
use eframe::egui;
use serde::Serialize;
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

pub fn enabled() -> bool {
    std::env::var("LEO_VIZ_TILE_DIAGNOSTICS").is_ok_and(|v| v == "1")
}

#[derive(Clone, Serialize)]
struct Event {
    elapsed_ms: f64,
    step: usize,
    kind: &'static str,
    data: Value,
}

pub struct Trace {
    started: Instant,
    step: AtomicUsize,
    outstanding: AtomicUsize,
    events: Mutex<Vec<Event>>,
}

impl Trace {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            started: Instant::now(),
            step: AtomicUsize::new(0),
            outstanding: AtomicUsize::new(0),
            events: Mutex::new(Vec::new()),
        })
    }

    pub fn record(&self, kind: &'static str, data: Value) {
        self.record_for(self.step.load(Ordering::Relaxed), kind, data);
    }

    fn record_for(&self, step: usize, kind: &'static str, data: Value) {
        self.events.lock().unwrap().push(Event {
            elapsed_ms: self.started.elapsed().as_secs_f64() * 1000.0,
            step,
            kind,
            data,
        });
    }

    pub fn request(self: &Arc<Self>, coord: TileCoord, generation: u64) -> RequestTrace {
        self.outstanding.fetch_add(1, Ordering::Relaxed);
        let request = RequestTrace {
            trace: Arc::clone(self),
            step: self.step.load(Ordering::Relaxed),
            coord,
            generation,
            queued: Instant::now(),
        };
        request.record("queued", json!({}));
        request
    }
}

pub struct RequestTrace {
    trace: Arc<Trace>,
    step: usize,
    coord: TileCoord,
    generation: u64,
    queued: Instant,
}

impl RequestTrace {
    pub fn record(&self, kind: &'static str, mut data: Value) {
        data["tile"] = json!(self.coord);
        data["generation"] = json!(self.generation);
        self.trace.record_for(self.step, kind, data);
    }

    pub fn started(&self) {
        self.record(
            "fetch_started",
            json!({
                "queue_wait_ms": self.queued.elapsed().as_secs_f64() * 1000.0,
            }),
        );
    }
}

impl Drop for RequestTrace {
    fn drop(&mut self) {
        self.record("fetch_finished", json!({}));
        self.trace.outstanding.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, Serialize)]
struct Step {
    label: &'static str,
    altitude_km: f64,
    lat: f64,
    lon: f64,
    #[serde(skip)]
    projection: Option<crate::projection::ProjectionKind>,
}

pub struct Replay {
    pub trace: Arc<Trace>,
    steps: Vec<Step>,
    index: usize,
    started: Option<Instant>,
    quiet_since: Option<Instant>,
    screenshot_requested: Option<Instant>,
    results: Vec<Value>,
    report: std::path::PathBuf,
    warmup_frames: usize,
    pushbroom: bool,
    pushbroom_started: Option<Instant>,
    pushbroom_hold: Option<f64>,
    aois: bool,
}

impl Replay {
    pub fn from_env() -> Option<Self> {
        if !enabled() {
            return None;
        }
        let pushbroom =
            std::env::var("LEO_VIZ_PUSHBROOM_DIAGNOSTICS").is_ok_and(|value| value == "1");
        let high_resolution = std::env::var("LEO_VIZ_TILE_DIAGNOSTICS_HIGH_RESOLUTION")
            .is_ok_and(|value| value == "1");
        let lat = std::env::var("LEO_VIZ_TILE_DIAGNOSTICS_LAT")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && v.abs() <= 84.0)
            .unwrap_or(if pushbroom { 0.0 } else { 48.7 });
        let lon = std::env::var("LEO_VIZ_TILE_DIAGNOSTICS_LON")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && v.abs() <= 179.0)
            .unwrap_or(25.5);
        let mut steps: Vec<_> = [
            ("initial", 3000.0, lat, lon),
            ("zoom_in", 1000.0, lat, lon),
            ("return", 3000.0, lat, lon),
            ("zoom_in_again", 1000.0, lat, lon),
            ("intermediate", 2000.0, lat, lon),
            ("warm_3000", 3000.0, lat, lon),
            ("warm_1000", 1000.0, lat, lon),
            ("warm_2000", 2000.0, lat, lon),
            ("nearby_2000", 2000.0, lat + 0.1, lon + 0.1),
            ("adjacent_tiles", 2000.0, lat + 0.1, lon + 6.0),
            ("return_from_adjacent", 2000.0, lat, lon),
            ("nearby_3000", 3000.0, lat + 0.1, lon + 0.1),
        ]
        .into_iter()
        .map(|(label, altitude_km, lat, lon)| Step {
            label,
            altitude_km,
            lat,
            lon,
            projection: None,
        })
        .collect();
        if std::env::var("LEO_VIZ_TILE_DIAGNOSTICS_CLOSEUP").is_ok_and(|value| value == "1") {
            for (label, offset) in [
                ("maximum_zoom", 0.0),
                ("maximum_zoom_north", 0.000002),
                ("maximum_zoom_return", 0.0),
            ] {
                steps.push(Step {
                    label,
                    altitude_km: 0.5,
                    lat: lat + offset,
                    lon,
                    projection: None,
                });
            }
        }
        if high_resolution && !pushbroom {
            steps = [
                ("ukraine_level_23_fallback", 0.01, 48.7, 25.5),
                ("stockholm_level_19", 0.25, 59.3293, 18.0686),
                ("stockholm_level_20_fallback", 0.125, 59.3293, 18.0686),
                ("los_angeles_level_20", 0.125, 34.0522, -118.2437),
                ("los_angeles_level_21_fallback", 0.06, 34.0522, -118.2437),
                ("los_angeles_level_23_fallback", 0.01, 34.0522, -118.2437),
                ("los_angeles_level_20_return", 0.125, 34.0522, -118.2437),
                ("ukraine_level_23_return", 0.01, 48.7, 25.5),
            ]
            .into_iter()
            .map(|(label, altitude_km, lat, lon)| Step {
                label,
                altitude_km,
                lat,
                lon,
                projection: None,
            })
            .collect();
        }
        let projections =
            std::env::var("LEO_VIZ_TILE_DIAGNOSTICS_PROJECTIONS").is_ok_and(|value| value == "1");
        let aois = std::env::var("LEO_VIZ_TILE_DIAGNOSTICS_AOIS").is_ok_and(|value| value == "1");
        if projections {
            use crate::projection::ProjectionKind::*;
            steps = [
                Equirectangular,
                Mercator,
                Mollweide,
                Sinusoidal,
                AzimuthalEquidistant,
                Hammer,
                HEALPix,
                Cassini,
                TransverseMercator,
                LambertAzimuthalEqualArea,
                GallPeters,
                PeirceQuincuncial,
            ]
            .into_iter()
            .map(|projection| Step {
                label: projection.label(),
                altitude_km: 3000.0,
                lat,
                lon,
                projection: Some(projection),
            })
            .collect();
        }
        Some(Self {
            trace: Trace::new(),
            steps,
            index: 0,
            started: None,
            quiet_since: None,
            screenshot_requested: None,
            results: Vec::new(),
            report: if aois {
                "target/aoi-projections-diagnostics.json"
            } else if projections {
                "target/esri-projections-diagnostics.json"
            } else if pushbroom {
                "target/pushbroom-diagnostics.json"
            } else if high_resolution {
                "target/esri-high-resolution-diagnostics.json"
            } else {
                "target/esri-diagnostics.json"
            }
            .into(),
            warmup_frames: 0,
            pushbroom,
            pushbroom_started: None,
            pushbroom_hold: None,
            aois,
        })
    }

    pub fn before_frame(&mut self, viewer: &mut ViewerState, ctx: &egui::Context) {
        ctx.request_repaint();
        // Let the panels establish the viewport before selecting the first tiles.
        if self.warmup_frames < 3 {
            if self.warmup_frames == 0 && self.aois {
                viewer.aoi_generator.load_default_jobs(ctx);
                viewer.tabs[0].planets[0]
                    .areas_of_interest
                    .push(crate::config::AreaOfInterest {
                        name: "Manual AOI".into(),
                        lat: 48.7,
                        lon: 25.5,
                        radius_km: 1000.0,
                        color: egui::Color32::from_rgba_unmultiplied(255, 205, 40, 140),
                        ground_station_idx: None,
                        job_mode: crate::config::AoiJobMode::Route,
                        job_n: 3,
                        reducer_placement: crate::config::SpaceCompReducerPlacement::NearMappers,
                        show_spacecomp_flow: false,
                        selected: false,
                    });
            }
            self.warmup_frames += 1;
            return;
        }
        if self.aois && viewer.aoi_generator.preview.is_none() {
            return;
        }
        if self.pushbroom && self.screenshot_requested.is_none() {
            let tab = &mut viewer.tabs[0];
            if self.pushbroom_started.is_none() {
                let mut constellation = crate::config::ConstellationConfig::new(0);
                constellation.num_planes = 1;
                constellation.sats_per_plane = 1;
                constellation.altitude_km = 550.0;
                constellation.inclination = 0.0;
                constellation.raan_offset = 25.5
                    + crate::time::body_rotation_angle(
                        crate::celestial::CelestialBody::Earth,
                        0.0,
                        viewer.current_gmst,
                    )
                    .to_degrees();
                constellation.raan_offset =
                    (constellation.raan_offset + 180.0).rem_euclid(360.0) - 180.0;
                tab.planets[0].constellations = vec![constellation];
                tab.planets[0].satellite_cameras = vec![crate::config::SatelliteCamera {
                    id: 900_001,
                    label: "Diagnostic satellite".into(),
                    constellation_idx: 0,
                    plane: 0,
                    sat_index: 0,
                    screen_pos: None,
                }];
                tab.settings.pushbroom.enabled = true;
                tab.settings.pushbroom.show_images = true;
                self.pushbroom_started = Some(Instant::now());
            }
            tab.settings.time = self
                .pushbroom_hold
                .unwrap_or_else(|| self.pushbroom_started.unwrap().elapsed().as_secs_f64() * 4.0);
        }
        if self.started.is_some()
            || self.index >= self.steps.len()
            || self.trace.started.elapsed() < Duration::from_millis(500)
        {
            return;
        }
        let step = self.steps[self.index];
        if let Some(projection) = step.projection {
            viewer.tabs[0].settings.planet_projection = projection;
            viewer.tabs[0].planets[0].projection_override = Some(projection);
        }
        let settings = &mut viewer.tabs[0].settings;
        settings.animate = false;
        settings.auto_zoom = false;
        settings.auto_rotate = false;
        settings.sun_fixed_camera = false;
        settings.earth_fixed_camera = true;
        settings.camera_mode = crate::config::CameraMode::Unlocked;
        settings.zoom = 10_000.0 / step.altitude_km;
        settings.rotation =
            crate::math::lat_lon_to_matrix(step.lat.to_radians(), step.lon.to_radians());
        viewer.tile_overlay.enabled = true;
        self.trace.step.store(self.index, Ordering::Relaxed);
        self.trace.record("step_started", json!(step));
        self.started = Some(Instant::now());
        self.quiet_since = None;
        eprintln!(
            "ESRI diagnostic {}: {} km at {}, {}",
            self.index, step.altitude_km, step.lat, step.lon
        );
    }

    pub fn after_frame(&mut self, viewer: &ViewerState, ctx: &egui::Context) {
        let Some(started) = self.started else {
            return;
        };
        if let Some(requested) = self.screenshot_requested {
            let screenshot = ctx.input(|input| {
                input.events.iter().find_map(|event| {
                    if let egui::Event::Screenshot {
                        user_data, image, ..
                    } = event
                    {
                        if user_data
                            .data
                            .as_ref()
                            .and_then(|v| v.downcast_ref::<usize>())
                            == Some(&self.index)
                        {
                            return Some(Arc::clone(image));
                        }
                    }
                    None
                })
            });
            if let Some(image) = screenshot {
                let path = format!(
                    "{}-{:02}.png",
                    self.report.with_extension("").display(),
                    self.index
                );
                let pixels: Vec<_> = image.pixels.iter().flat_map(|p| p.to_array()).collect();
                match image::save_buffer(
                    &path,
                    &pixels,
                    image.size[0] as u32,
                    image.size[1] as u32,
                    image::ColorType::Rgba8,
                ) {
                    Ok(()) => self.results.last_mut().unwrap()["screenshot"] = json!(path),
                    Err(error) => {
                        self.results.last_mut().unwrap()["screenshot_error"] =
                            json!(error.to_string())
                    }
                }
                self.finish_step(viewer, ctx);
            } else if requested.elapsed() > Duration::from_secs(3) {
                self.results.last_mut().unwrap()["screenshot_error"] = json!("capture timed out");
                self.finish_step(viewer, ctx);
            }
            return;
        }
        let overlay = &viewer.tile_overlay;
        let mut source_levels = std::collections::BTreeMap::<u8, usize>::new();
        for &coord in &overlay.needed_tiles {
            if let Some(source) = std::iter::successors(Some(coord), |coord| coord.parent())
                .find(|coord| overlay.tile_tree.has_tile(coord))
            {
                *source_levels.entry(source.z).or_default() += 1;
            }
        }
        let exact = overlay
            .needed_tiles
            .iter()
            .filter(|c| overlay.tile_tree.has_tile(c))
            .count();
        let fading = overlay
            .detail_texture
            .as_ref()
            .is_some_and(|t| t.blend() < 1.0);
        self.trace.record(
            "frame",
            json!({
                "dirty": overlay.dirty, "fading": fading,
                "needed": overlay.needed_tiles.len(), "exact_in_ram": exact,
                "outstanding": self.trace.outstanding.load(Ordering::Relaxed),
            }),
        );
        let aois_ready = !self.aois
            || viewer
                .aoi_generator
                .preview
                .as_ref()
                .is_some_and(|preview| {
                    self.steps[self.index]
                        .projection
                        .is_none_or(|projection| preview.projected_geometry_ready(projection))
                });
        let quiet = aois_ready
            && self.trace.outstanding.load(Ordering::Relaxed) == 0
            && !overlay.dirty
            && !fading
            && !overlay.needed_tiles.is_empty()
            && overlay.detail_texture.is_some();
        if self.pushbroom && quiet {
            self.pushbroom_hold
                .get_or_insert(viewer.tabs[0].settings.time);
        }
        if !quiet {
            self.quiet_since = None;
        }
        let settled = quiet
            && self.quiet_since.get_or_insert_with(Instant::now).elapsed()
                >= Duration::from_millis(250);
        let timed_out = started.elapsed() >= Duration::from_secs(15);
        if settled || timed_out {
            self.results.push(json!({
                "step": self.index, "view": self.steps[self.index], "timed_out": timed_out,
                "settle_ms": started.elapsed().as_secs_f64() * 1000.0,
                "needed_tiles": overlay.needed_tiles.len(), "exact_in_ram": exact,
                "tile_level": overlay.last_zoom, "ram_tiles": overlay.tile_tree.tile_count,
                "source_levels": source_levels,
                "covered_tiles": source_levels.values().sum::<usize>(),
                "view_size": [viewer.view_width, viewer.view_height],
            }));
            self.screenshot_requested = Some(Instant::now());
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::new(
                self.index,
            )));
        }
    }

    fn finish_step(&mut self, viewer: &ViewerState, ctx: &egui::Context) {
        self.trace.record("step_finished", json!({}));
        self.index += 1;
        self.started = None;
        self.screenshot_requested = None;
        self.pushbroom_hold = None;
        if self.index != self.steps.len() {
            return;
        }
        let events = self.trace.events.lock().unwrap().clone();
        for result in &mut self.results {
            result["measurements"] = summarize(&events, result["step"].as_u64().unwrap() as usize);
        }
        let report = json!({
            "cache_directory": viewer.tile_overlay.disk_cache_dir,
            "cache_cleared": false,
            "ram_cache_limit_tiles": viewer.tile_overlay.tile_tree.max_tiles,
            "gpu_cache_limit_tiles": crate::tile::GPU_TILE_CACHE_LIMIT,
            "timing_notes": "Atlas timings measure CPU preparation and queue submission, not GPU execution. Settle time includes the 150 ms fade and 250 ms quiet verification.",
            "outstanding_at_exit": self.trace.outstanding.load(Ordering::Relaxed),
            "steps": self.results, "events": events,
        });
        let write = serde_json::to_vec_pretty(&report)
            .map_err(|e| e.to_string())
            .and_then(|bytes| std::fs::write(&self.report, bytes).map_err(|e| e.to_string()));
        match write {
            Ok(()) => eprintln!("ESRI diagnostic report: {}", self.report.display()),
            Err(error) => eprintln!("ESRI diagnostic report failed: {error}"),
        }
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
    }
}

fn summarize(events: &[Event], step: usize) -> Value {
    let events: Vec<_> = events.iter().filter(|e| e.step == step).collect();
    let count = |kind| events.iter().filter(|e| e.kind == kind).count();
    let sum = |kind, key: &str| {
        events
            .iter()
            .filter(|e| e.kind == kind)
            .filter_map(|e| e.data[key].as_f64())
            .sum::<f64>()
    };
    let max = |kind, key: &str| {
        events
            .iter()
            .filter(|e| e.kind == kind)
            .filter_map(|e| e.data[key].as_f64())
            .fold(0.0_f64, f64::max)
    };
    let initial = events.iter().find(|e| e.kind == "selection");
    json!({
        "initial_selection": initial.map(|e| &e.data),
        "queued_requests": count("queued"), "cancelled_requests": count("cancelled"),
        "disk_hits": events.iter().filter(|e| e.kind == "disk_lookup" && e.data["hit"] == true).count(),
        "http_requests": count("http_request"),
        "http_failures": events.iter().filter(|e| e.kind == "http_result" && e.data["success"] == false && e.data["missing"] != true).count(),
        "missing_tiles": events.iter().filter(|e| e.kind == "http_result" && e.data["missing"] == true).count(),
        "missing_cache_hits": count("missing_cache_hit"),
        "rejected_tiles": count("tile_rejected"),
        "ram_evictions": sum("ram_evicted", "count"),
        "max_queue_wait_ms": max("fetch_started", "queue_wait_ms"),
        "disk_lookup_decode_ms": sum("disk_lookup", "elapsed_ms"),
        "jpeg_decode_ms": sum("jpeg_decode", "elapsed_ms"),
        "http_total_ms": sum("http_result", "elapsed_ms"),
        "atlas_rebuilds": count("atlas_move"), "atlas_move_cpu_ms": sum("atlas_move", "elapsed_ms"),
        "upload_frames": events.iter().filter(|e| e.kind == "atlas_upload" && e.data["tiles"].as_u64().unwrap_or(0) > 0).count(),
        "uploaded_tiles": sum("atlas_upload", "tiles"),
        "uploaded_bytes": sum("atlas_upload", "bytes"),
        "gpu_reused_tiles": sum("atlas_upload", "gpu_reused_tiles"),
        "gpu_resampled_tiles": sum("atlas_upload", "gpu_resampled_tiles"),
        "gpu_evictions": sum("atlas_upload", "gpu_evicted_tiles"),
        "max_gpu_cached_tiles": max("atlas_upload", "gpu_cached_tiles"),
        "atlas_upload_cpu_ms": sum("atlas_upload", "elapsed_ms"),
        "max_atlas_upload_cpu_ms": max("atlas_upload", "elapsed_ms"),
        "publications": count("published"),
        "pushbroom_images": count("pushbroom_image"),
        "pushbroom_frames": count("pushbroom_frame"),
        "pushbroom_uploaded_tiles": sum("pushbroom_atlas_update", "tiles"),
        "max_pushbroom_cpu_ms": max("pushbroom_frame", "cpu_ms"),
        "first_publication_ms": events.iter().find(|e| e.kind == "published")
            .zip(events.iter().find(|e| e.kind == "step_started"))
            .map(|(published, started)| published.elapsed_ms - started.elapsed_ms),
        "frames": count("frame"),
        "fade_wait_frames": events.iter().filter(|e| e.kind == "frame" && e.data["fading"] == true).count(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_distinguish_queued_work_disk_hits_and_actual_http() {
        let trace = Trace::new();
        trace.record("selection", json!({"ram_hits": 42}));
        let request = trace.request(TileCoord { x: 1, y: 2, z: 3 }, 0);
        request.record("disk_lookup", json!({"hit": true, "elapsed_ms": 1.0}));
        drop(request);
        let request = trace.request(TileCoord { x: 2, y: 2, z: 3 }, 0);
        request.record("disk_lookup", json!({"hit": false, "elapsed_ms": 0.1}));
        trace.step.store(1, Ordering::Relaxed);
        request.record("http_request", json!({}));
        drop(request);
        let events = trace.events.lock().unwrap();
        let first = summarize(&events, 0);
        assert_eq!(first["queued_requests"], 2);
        assert_eq!(first["disk_hits"], 1);
        assert_eq!(first["http_requests"], 1);
        assert_eq!(summarize(&events, 1)["http_requests"], 0);
        assert_eq!(trace.outstanding.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn queued_or_cancelled_work_is_not_reported_as_a_download() {
        let trace = Trace::new();
        let request = trace.request(TileCoord { x: 0, y: 0, z: 0 }, 0);
        assert_eq!(trace.outstanding.load(Ordering::Relaxed), 1);
        request.record("cancelled", json!({}));
        drop(request);
        let events = trace.events.lock().unwrap();
        let report = summarize(&events, 0);
        assert_eq!(report["queued_requests"], 1);
        assert_eq!(report["cancelled_requests"], 1);
        assert_eq!(report["http_requests"], 0);
        assert_eq!(trace.outstanding.load(Ordering::Relaxed), 0);
    }
}
