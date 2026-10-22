//! GPU preview of the recorded ground strip, with an independent imagery cache.
use crate::pushbroom::ImageStrip;
use crate::tile::{
    fetch_tile, AtlasLayout, DetailAtlas, TileAtlasWriter, TileCacheEntry, TileCoord,
    TileFetchRequest, TileFetchResult, TileQuadTree, TileReuse, MAX_TILE_ZOOM, TILE_SIZE,
};
use eframe::egui;
use egui_wgpu::{wgpu, RenderState};
use std::collections::{HashMap, HashSet};
use std::sync::{mpsc, Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

const MAX_TILES: usize = 256;
const MAX_CAMERAS: usize = 4;
const UPLOADS_PER_FRAME: usize = 8;
const RETRY_INTERVAL: Duration = Duration::from_secs(5);
const MISSING_TTL: Duration = Duration::from_secs(24 * 60 * 60);

struct ImageCache {
    tree: TileQuadTree,
    directory: std::path::PathBuf,
    pending: HashSet<TileCoord>,
    missing: HashMap<TileCoord, Instant>,
    retried: HashMap<TileCoord, Instant>,
    wanted: Arc<RwLock<HashSet<TileCoord>>>,
    tx: mpsc::SyncSender<TileFetchRequest>,
    rx: mpsc::Receiver<TileFetchResult>,
    trace: Option<Arc<crate::tile_diagnostics::Trace>>,
}

impl ImageCache {
    fn new(ctx: &egui::Context, trace: Option<Arc<crate::tile_diagnostics::Trace>>) -> Self {
        let (tx, requests) = mpsc::sync_channel::<TileFetchRequest>(256);
        let (results, rx) = mpsc::channel();
        let requests = Arc::new(Mutex::new(requests));
        let wanted = Arc::new(RwLock::new(HashSet::<TileCoord>::new()));
        for _ in 0..4 {
            let requests = Arc::clone(&requests);
            let wanted = Arc::clone(&wanted);
            let results = results.clone();
            let ctx = ctx.clone();
            std::thread::spawn(move || {
                let agent = ureq::AgentBuilder::new()
                    .timeout(Duration::from_secs(8))
                    .build();
                loop {
                    let Ok(request) = requests.lock().unwrap().recv() else {
                        break;
                    };
                    if let Some(trace) = &request.trace {
                        trace.started();
                    }
                    let still_wanted = wanted.read().unwrap().contains(&request.coord);
                    let result = if still_wanted {
                        fetch_tile(&request, &agent)
                    } else {
                        if let Some(trace) = &request.trace {
                            trace.record("cancelled", serde_json::json!({"consumer": "pushbroom"}));
                        }
                        TileFetchResult::empty(&request)
                    };
                    if results.send(result).is_err() {
                        break;
                    }
                    ctx.request_repaint();
                }
            });
        }
        Self {
            tree: TileQuadTree::new(512),
            directory: crate::geo::dirs_cache()
                .join("leo-viz")
                .join("pushbroom-tiles"),
            pending: HashSet::new(),
            missing: HashMap::new(),
            retried: HashMap::new(),
            wanted,
            tx,
            rx,
            trace,
        }
    }

    fn poll(&mut self) {
        while let Ok(result) = self.rx.try_recv() {
            self.pending.remove(&result.coord);
            if result.missing {
                if self.missing.len() >= 16_384 {
                    self.missing.retain(|_, time| time.elapsed() < MISSING_TTL);
                    if let Some(oldest) = self
                        .missing
                        .iter()
                        .min_by_key(|(_, time)| **time)
                        .map(|(coord, _)| *coord)
                    {
                        self.missing.remove(&oldest);
                    }
                }
                self.missing.insert(result.coord, Instant::now());
            } else if !result.pixels.is_empty() {
                self.tree.insert(
                    result.coord,
                    TileCacheEntry {
                        pixels: result.pixels,
                        width: result.width,
                        height: result.height,
                    },
                );
            }
        }
    }

    fn request(&mut self, needed: &[TileCoord]) {
        let wanted: HashSet<_> = needed
            .iter()
            .flat_map(|coord| std::iter::successors(Some(*coord), |coord| coord.parent()))
            .collect();
        *self.wanted.write().unwrap() = wanted;
        self.retried
            .retain(|_, time| time.elapsed() < RETRY_INTERVAL);
        let mut attempted = HashSet::new();
        for &coord in needed {
            // Coarse coverage can be drawn immediately, then refined independently of motion.
            for shift in [4, 2, 0] {
                let shift = shift.min(coord.z);
                let mut source = TileCoord {
                    x: coord.x >> shift,
                    y: coord.y >> shift,
                    z: coord.z - shift,
                };
                while self
                    .missing
                    .get(&source)
                    .is_some_and(|time| time.elapsed() < MISSING_TTL)
                {
                    let Some(parent) = source.parent() else { break };
                    source = parent;
                }
                if self.tree.has_tile(&source)
                    || self.pending.contains(&source)
                    || self.retried.contains_key(&source)
                    || !attempted.insert(source)
                {
                    continue;
                }
                let request = TileFetchRequest {
                    coord: source,
                    cache_dir: self.directory.clone(),
                    generation: 0,
                    cancel_if_stale: false,
                    trace: self.trace.as_ref().map(|trace| trace.request(source, 0)),
                };
                match self.tx.try_send(request) {
                    Ok(()) => {
                        self.pending.insert(source);
                        self.retried.insert(source, Instant::now());
                    }
                    Err(mpsc::TrySendError::Full(_)) => return,
                    Err(mpsc::TrySendError::Disconnected(_)) => return,
                }
            }
        }
    }
}

struct PreviewAtlas {
    atlas: DetailAtlas,
    texture: wgpu::Texture,
    id: egui::TextureId,
    render: RenderState,
}

impl PreviewAtlas {
    fn texture(render: &RenderState, layout: AtlasLayout) -> wgpu::Texture {
        let [width, height] = layout.size();
        render.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pushbroom_atlas"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    fn new(render: &RenderState, layout: AtlasLayout) -> Self {
        let texture = Self::texture(render, layout);
        let id = render.renderer.write().register_native_texture(
            &render.device,
            &texture.create_view(&Default::default()),
            wgpu::FilterMode::Linear,
        );
        let mut atlas = DetailAtlas::default();
        atlas.set_layout(layout);
        Self {
            atlas,
            texture,
            id,
            render: render.clone(),
        }
    }

    fn update(
        &mut self,
        needed: &[TileCoord],
        tree: &mut TileQuadTree,
        budget: &mut usize,
    ) -> usize {
        let layout = self.atlas.layout.unwrap();
        if !needed.iter().all(|coord| layout.contains(*coord)) {
            let next = AtlasLayout::new(needed, needed[0].x);
            let texture = Self::texture(&self.render, next);
            let mut encoder = self
                .render
                .device
                .create_command_encoder(&Default::default());
            for copy in next.copies_from(layout) {
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        origin: wgpu::Origin3d {
                            x: copy.source[0],
                            y: copy.source[1],
                            z: 0,
                        },
                        ..self.texture.as_image_copy()
                    },
                    wgpu::TexelCopyTextureInfo {
                        origin: wgpu::Origin3d {
                            x: copy.destination[0],
                            y: copy.destination[1],
                            z: 0,
                        },
                        ..texture.as_image_copy()
                    },
                    wgpu::Extent3d {
                        width: copy.size[0],
                        height: copy.size[1],
                        depth_or_array_layers: 1,
                    },
                );
            }
            self.render.queue.submit([encoder.finish()]);
            self.render
                .renderer
                .write()
                .update_egui_texture_from_wgpu_texture(
                    &self.render.device,
                    &texture.create_view(&Default::default()),
                    wgpu::FilterMode::Linear,
                    self.id,
                );
            self.texture = texture;
            self.atlas.set_layout(next);
        }
        let before = *budget;
        self.atlas.update_tiles(
            tree,
            needed,
            AtlasWriter {
                texture: &self.texture,
                queue: &self.render.queue,
                budget,
            },
        );
        before - *budget
    }
}

impl Drop for PreviewAtlas {
    fn drop(&mut self) {
        self.render.renderer.write().free_texture(&self.id);
    }
}

struct AtlasWriter<'a> {
    texture: &'a wgpu::Texture,
    queue: &'a wgpu::Queue,
    budget: &'a mut usize,
}
impl TileAtlasWriter for AtlasWriter<'_> {
    fn reuse(&mut self, _: TileCoord, _: TileCoord, _: [u32; 2]) -> TileReuse {
        if *self.budget == 0 {
            TileReuse::Deferred
        } else {
            TileReuse::Missing
        }
    }
    fn upload(&mut self, _: TileCoord, _: TileCoord, origin: [u32; 2], rgba: &[[u8; 4]]) {
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                origin: wgpu::Origin3d {
                    x: origin[0],
                    y: origin[1],
                    z: 0,
                },
                ..self.texture.as_image_copy()
            },
            bytemuck::cast_slice(rgba),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TILE_SIZE * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: TILE_SIZE,
                height: TILE_SIZE,
                depth_or_array_layers: 1,
            },
        );
        *self.budget -= 1;
    }
}

#[derive(Default)]
struct Camera {
    atlas: Option<PreviewAtlas>,
    strip: Option<ImageStrip>,
    frozen: bool,
    epoch: Option<(u64, u64, u64, u64)>,
}
impl Camera {
    fn observe(&mut self, strip: Option<ImageStrip>, settings: crate::pushbroom::Settings) {
        if self.frozen {
            return;
        }
        let Some(strip) = strip else {
            self.atlas = None;
            self.strip = None;
            self.epoch = None;
            return;
        };
        let epoch = (
            strip.source,
            settings.swath_km.to_bits(),
            settings.trail_s.to_bits(),
            settings.image_length_km.to_bits(),
        );
        if self.epoch != Some(epoch) || self.strip.as_ref().is_some_and(|old| strip.time < old.time)
        {
            self.atlas = None;
        }
        self.epoch = Some(epoch);
        self.strip = Some(strip);
    }
}

#[derive(Default)]
pub(crate) struct Cameras {
    cameras: HashMap<usize, Camera>,
    cache: Option<ImageCache>,
}
impl Cameras {
    pub fn show(&mut self, viewer: &mut crate::viewer::ViewerState, ctx: &egui::Context) {
        if let Some(cache) = &mut self.cache {
            cache.poll();
        }
        let Some(tab) = viewer.tabs.get_mut(viewer.active_tab_idx) else {
            return;
        };
        let settings = tab.settings.pushbroom;
        if !settings.enabled || !settings.show_images {
            self.cameras.clear();
            if let Some(cache) = &mut self.cache {
                cache.request(&[]);
            }
            return;
        }
        let Some(render) = viewer.render_state.as_ref() else {
            return;
        };
        let trace = viewer.tile_overlay.diagnostics.clone();
        let cache = self
            .cache
            .get_or_insert_with(|| ImageCache::new(ctx, trace.clone()));
        let mut active = HashSet::new();
        let mut closed = HashSet::new();
        let mut requests = Vec::new();
        let mut budget = UPLOADS_PER_FRAME;
        for planet in &mut tab.planets {
            if planet.celestial_body != crate::celestial::CelestialBody::Earth {
                continue;
            }
            for selected in &planet.satellite_cameras {
                if active.len() == MAX_CAMERAS {
                    break;
                }
                active.insert(selected.id);
                let camera = self.cameras.entry(selected.id).or_default();
                let strip = planet.pushbroom_history.image_strip(
                    selected.constellation_idx,
                    selected.plane,
                    selected.sat_index,
                    settings,
                    planet.celestial_body.radius_km(),
                );
                let mut open = true;
                egui::Window::new(format!("Pushbroom: {}", selected.label))
                    .id(egui::Id::new(("pushbroom_image", selected.id)))
                    .open(&mut open).collapsible(false).default_size([420.0, 520.0])
                    .show(ctx, |ui| {
                        let started = Instant::now();
                        ui.checkbox(&mut camera.frozen, "Freeze image");
                        camera.observe(strip, settings);
                        let available = egui::vec2(ui.available_width().max(1.0), (ui.available_height() - 72.0).max(100.0));
                        let (rect, _) = ui.allocate_exact_size(available, egui::Sense::hover());
                        ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);
                        let Some(strip) = &camera.strip else {
                            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "Waiting for scan", egui::FontId::proportional(13.0), egui::Color32::GRAY);
                            return;
                        };
                        let swath = f64::from_bits(camera.epoch.unwrap().1);
                        let length = strip.lines.last().unwrap().distance_km;
                        let image_rect = fit_image(rect, swath, length);
                        let pixels = preview_pixels(image_rect.size(), ctx.pixels_per_point());
                        let needed = needed_tiles(strip, swath, pixels[0], render.device.limits().max_texture_dimension_2d);
                        let mut exact = 0;
                        let mut covered = 0;
                        for coord in &needed {
                            if let Some(z) = cache.tree.best_tile_zoom(coord) {
                                covered += 1;
                                exact += usize::from(z == coord.z);
                            }
                        }
                        if !needed.is_empty() {
                            if !camera.frozen {
                                requests.extend_from_slice(&needed);
                                let atlas = camera.atlas.get_or_insert_with(|| PreviewAtlas::new(render, AtlasLayout::new(&needed, needed[0].x)));
                                let uploaded = atlas.update(&needed, &mut cache.tree, &mut budget);
                                if uploaded > 0 {
                                    if let Some(trace) = &trace {
                                        trace.record("pushbroom_atlas_update", serde_json::json!({"camera":selected.id,"tiles":uploaded,"size":atlas.atlas.layout.unwrap().size(),"target_pixels":pixels}));
                                    }
                                }
                            }
                            if let Some(atlas) = &camera.atlas {
                                let mesh = strip_mesh(strip, image_rect, atlas.atlas.layout.unwrap(), atlas.id);
                                let vertices = mesh.vertices.len();
                                ui.painter().with_clip_rect(image_rect).add(egui::Shape::mesh(mesh));
                                if let Some(trace) = &trace {
                                    trace.record("pushbroom_frame", serde_json::json!({
                                        "camera":selected.id,"time":strip.time,"vertices":vertices,"size":pixels,
                                        "source_level":needed[0].z,"covered":covered,"needed":needed.len(),
                                        "frozen":camera.frozen,"cpu_ms":started.elapsed().as_secs_f64()*1000.0,
                                        "cache_directory":cache.directory,
                                    }));
                                }
                            }
                        } else {
                            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "Outside imagery coverage", egui::FontId::proportional(13.0), egui::Color32::GRAY);
                        }
                        ui.label(format!("{swath:.1} x {length:.1} km"));
                        ui.weak(format!("Tiles: {exact}/{}", needed.len()));
                        ui.small("Esri, Vantor, Earthstar Geographics, GIS User Community")
                            .on_hover_text("Esri World Imagery basemap, not live satellite imagery");
                    });
                if !open {
                    closed.insert(selected.id);
                }
            }
            planet
                .satellite_cameras
                .retain(|camera| !closed.contains(&camera.id));
        }
        self.cameras
            .retain(|id, _| active.contains(id) && !closed.contains(id));
        cache.request(&requests);
        if !active.is_empty() {
            ctx.request_repaint();
        }
    }
}

fn fit_image(rect: egui::Rect, swath: f64, length: f64) -> egui::Rect {
    let physical = egui::vec2(swath.max(0.001) as f32, length.max(0.001) as f32);
    let scale = (rect.width() / physical.x).min(rect.height() / physical.y);
    egui::Rect::from_center_size(rect.center(), physical * scale)
}

fn preview_pixels(size: egui::Vec2, pixels_per_point: f32) -> [usize; 2] {
    // Request extra source detail for filtering, not a fixed 320-pixel bitmap.
    let factor = pixels_per_point * 2.0;
    [size.x, size.y].map(|value| (value * factor).ceil().clamp(1.0, 2048.0) as usize)
}

fn uv(point: [f64; 3]) -> Option<[f64; 2]> {
    let norm = point.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= 0.0 {
        return None;
    }
    let mercator = (point[1] / norm).clamp(-1.0, 1.0).atanh();
    if mercator.abs() >= std::f64::consts::PI {
        return None;
    }
    Some([
        (((-point[2]).atan2(point[0]) / std::f64::consts::TAU) + 0.5).rem_euclid(1.0),
        0.5 - mercator / std::f64::consts::TAU,
    ])
}

fn needed_tiles(
    strip: &ImageStrip,
    swath_km: f64,
    width: usize,
    max_dimension: u32,
) -> Vec<TileCoord> {
    let points: Vec<_> = strip
        .lines
        .iter()
        .flat_map(|line| line.edges)
        .filter_map(uv)
        .collect();
    let Some(first) = points.first() else {
        return Vec::new();
    };
    let min_x = points
        .iter()
        .map(|p| first[0] + (p[0] - first[0] + 0.5).rem_euclid(1.0) - 0.5)
        .fold(f64::INFINITY, f64::min);
    let max_x = points
        .iter()
        .map(|p| first[0] + (p[0] - first[0] + 0.5).rem_euclid(1.0) - 0.5)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
    let max_y = points
        .iter()
        .map(|p| p[1])
        .fold(f64::NEG_INFINITY, f64::max);
    let latitude_cos = (std::f64::consts::TAU * (0.5 - first[1])).cosh().recip();
    let meters_per_pixel = swath_km * 1000.0 / width.max(1) as f64;
    let mut zoom = (40_075_016.686 * latitude_cos / (256.0 * meters_per_pixel))
        .log2()
        .ceil()
        .clamp(0.0, f64::from(MAX_TILE_ZOOM)) as u8;
    loop {
        let n = 1i64 << zoom;
        let left = (min_x * n as f64).floor() as i64;
        let right = (max_x * n as f64).floor() as i64;
        let top = ((min_y * n as f64).floor() as i64).clamp(0, n - 1);
        let bottom = ((max_y * n as f64).floor() as i64).clamp(0, n - 1);
        let cols = right - left + 1;
        let rows = bottom - top + 1;
        if (cols * rows <= MAX_TILES as i64
            && cols * 256 <= i64::from(max_dimension)
            && rows * 256 <= i64::from(max_dimension))
            || zoom == 0
        {
            return (top..=bottom)
                .flat_map(|y| {
                    (left..=right).map(move |x| TileCoord {
                        x: x.rem_euclid(n) as u32,
                        y: y as u32,
                        z: zoom,
                    })
                })
                .collect();
        }
        zoom -= 1;
    }
}

fn strip_mesh(
    strip: &ImageStrip,
    rect: egui::Rect,
    layout: AtlasLayout,
    texture_id: egui::TextureId,
) -> egui::Mesh {
    let mut mesh = egui::Mesh::with_texture(texture_id);
    let length = strip.lines.last().map_or(0.0, |line| line.distance_km);
    if length <= 0.0 {
        return mesh;
    }
    let n = f64::from(1u32 << layout.z);
    for lines in strip.lines.windows(2) {
        let a = &lines[0];
        let b = &lines[1];
        let rows = ((b.distance_km - a.distance_km) / 5.0)
            .ceil()
            .clamp(1.0, 16.0) as usize;
        let columns = 16;
        for row in 0..rows {
            for column in 0..columns {
                let base = mesh.vertices.len() as u32;
                let mut vertices = Vec::with_capacity(4);
                for (r, c) in [
                    (row, column),
                    (row, column + 1),
                    (row + 1, column + 1),
                    (row + 1, column),
                ] {
                    let along = r as f64 / rows as f64;
                    let across = c as f64 / columns as f64;
                    let point = std::array::from_fn(|axis| {
                        let left = a.edges[0][axis] * (1.0 - along) + b.edges[0][axis] * along;
                        let right = a.edges[1][axis] * (1.0 - along) + b.edges[1][axis] * along;
                        left * (1.0 - across) + right * across
                    });
                    let Some([u, v]) = uv(point) else { break };
                    let tex_u =
                        (u * n - f64::from(layout.x)).rem_euclid(n) / f64::from(layout.cols);
                    let tex_v = (v * n - f64::from(layout.y)) / f64::from(layout.rows);
                    let distance = a.distance_km * (1.0 - along) + b.distance_km * along;
                    vertices.push(egui::epaint::Vertex {
                        pos: egui::pos2(
                            rect.left() + across as f32 * rect.width(),
                            rect.top() + (distance / length) as f32 * rect.height(),
                        ),
                        uv: egui::pos2(tex_u as f32, tex_v as f32),
                        color: egui::Color32::WHITE,
                    });
                }
                if vertices.len() == 4 {
                    mesh.vertices.extend(vertices);
                    mesh.indices
                        .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
                }
            }
        }
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pushbroom::ImageLine;

    fn point(lat: f64, lon: f64) -> [f64; 3] {
        let (lat, lon) = (lat.to_radians(), lon.to_radians());
        [lat.cos() * lon.cos(), lat.sin(), -lat.cos() * lon.sin()]
    }
    fn strip(left: f64, right: f64) -> ImageStrip {
        ImageStrip {
            time: 5.0,
            source: 1,
            lines: vec![
                ImageLine {
                    edges: [point(0.2, left), point(0.2, right)],
                    distance_km: 0.0,
                },
                ImageLine {
                    edges: [point(-0.2, left), point(-0.2, right)],
                    distance_km: 45.0,
                },
            ],
        }
    }

    #[test]
    fn preview_resolution_tracks_window_size_and_display_density() {
        assert_eq!(preview_pixels(egui::vec2(420.0, 900.0), 2.0), [1680, 2048]);
        assert_eq!(preview_pixels(egui::vec2(420.0, 900.0), 1.0), [840, 1800]);
        let strip = strip(-0.2, 0.2);
        let low = needed_tiles(&strip, 44.0, 320, 8192);
        let high = needed_tiles(&strip, 44.0, 1260, 8192);
        assert!(high[0].z > low[0].z);
        assert!(high.len() <= MAX_TILES);
    }

    #[test]
    fn mesh_projects_the_recorded_strip_and_handles_the_dateline() {
        for (left, right) in [(-0.2, 0.2), (179.8, -179.8)] {
            let strip = strip(left, right);
            let needed = needed_tiles(&strip, 44.0, 640, 8192);
            let layout = AtlasLayout::new(&needed, needed[0].x);
            let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(440.0, 450.0));
            let mesh = strip_mesh(&strip, rect, layout, egui::TextureId::User(1));
            assert!(!mesh.indices.is_empty());
            assert!(mesh.is_valid());
            for vertex in &mesh.vertices {
                assert!(rect.contains(vertex.pos));
                assert!(vertex.uv.x >= 0.0 && vertex.uv.x <= 1.0, "{:?}", vertex.uv);
                assert!(vertex.uv.y >= 0.0 && vertex.uv.y <= 1.0, "{:?}", vertex.uv);
            }
            let n = f64::from(1u32 << layout.z);
            for (index, edge) in [(0, strip.lines[0].edges[0]), (1, strip.lines[0].edges[1])] {
                let vertex = if index == 0 {
                    mesh.vertices.first().unwrap()
                } else {
                    &mesh.vertices[15 * 4 + 1]
                };
                let expected = uv(edge).unwrap();
                let actual_u =
                    ((f64::from(vertex.uv.x) * f64::from(layout.cols) + f64::from(layout.x)) / n)
                        .rem_euclid(1.0);
                let actual_v =
                    (f64::from(vertex.uv.y) * f64::from(layout.rows) + f64::from(layout.y)) / n;
                assert!((actual_u - expected[0]).abs() < 1e-7);
                assert!((actual_v - expected[1]).abs() < 1e-7);
            }
        }
        assert!(uv(point(89.0, 0.0)).is_none());
    }

    #[test]
    fn preview_motion_is_not_quantized_to_ten_updates_per_second() {
        let a = strip(-0.2, 0.2);
        let mut b = a.clone();
        b.time += 1.0 / 60.0;
        for line in &mut b.lines {
            for edge in &mut line.edges {
                let longitude = (-edge[2]).atan2(edge[0]).to_degrees() + 0.0001;
                *edge = point(edge[1].asin().to_degrees(), longitude);
            }
        }
        let needed = needed_tiles(&a, 44.0, 640, 8192);
        let layout = AtlasLayout::new(&needed, needed[0].x);
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(440.0, 450.0));
        let old = strip_mesh(&a, rect, layout, egui::TextureId::User(1));
        let new = strip_mesh(&b, rect, layout, egui::TextureId::User(1));
        assert_eq!(old.indices, new.indices);
        assert_eq!(old.vertices[0].pos, new.vertices[0].pos);
        assert_ne!(old.vertices[0].uv, new.vertices[0].uv);
    }

    #[test]
    fn freezing_preserves_the_strip_until_resumed_and_rewinds_reset_it() {
        let settings = crate::pushbroom::Settings::default();
        let mut camera = Camera::default();
        camera.observe(Some(strip(-0.2, 0.2)), settings);
        camera.frozen = true;
        let mut newer = strip(-0.1, 0.3);
        newer.time = 6.0;
        camera.observe(Some(newer.clone()), settings);
        assert_eq!(camera.strip.as_ref().unwrap().time, 5.0);
        camera.observe(None, settings);
        assert!(camera.strip.is_some());
        camera.frozen = false;
        camera.observe(Some(newer), settings);
        assert_eq!(camera.strip.as_ref().unwrap().time, 6.0);
        camera.observe(None, settings);
        assert!(camera.strip.is_none());
    }

    #[test]
    fn tile_selection_obeys_texture_limits_and_has_no_polar_wrap() {
        let strip = strip(-0.2, 0.2);
        let needed = needed_tiles(&strip, 44.0, 2048, 1024);
        let layout = AtlasLayout::new(&needed, needed[0].x);
        assert!(layout.size().iter().all(|size| *size <= 1024));
        assert!(needed.len() <= MAX_TILES);
    }

    #[test]
    #[ignore = "requires a native GPU adapter"]
    fn gpu_preview_preserves_tiles_when_moving_and_releases_its_texture() {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let renderer =
            egui_wgpu::Renderer::new(&device, wgpu::TextureFormat::Rgba8Unorm, Default::default());
        let render = RenderState {
            adapter,
            available_adapters: Vec::new(),
            device,
            queue,
            target_format: wgpu::TextureFormat::Rgba8Unorm,
            renderer: Arc::new(egui::mutex::RwLock::new(renderer)),
        };
        let entry = |color| TileCacheEntry {
            pixels: vec![color; 256 * 256],
            width: 256,
            height: 256,
        };
        let a = TileCoord { x: 1, y: 1, z: 4 };
        let b = TileCoord { x: 2, ..a };
        let c = TileCoord { x: 3, ..a };
        let mut tree = TileQuadTree::new(10);
        tree.insert(a, entry([40, 80, 120]));
        tree.insert(b, entry([80, 120, 160]));
        tree.insert(c, entry([120, 160, 200]));
        let mut atlas = PreviewAtlas::new(&render, AtlasLayout::new(&[a, b], a.x));
        let id = atlas.id;
        let mut budget = 1;
        assert_eq!(atlas.update(&[a, b], &mut tree, &mut budget), 1);
        assert_eq!(budget, 0);
        assert!(!atlas.atlas.covers(&[a, b]));
        assert_eq!(atlas.update(&[a, b], &mut tree, &mut budget), 0);
        budget = 1;
        assert_eq!(atlas.update(&[a, b], &mut tree, &mut budget), 1);
        assert!(atlas.atlas.covers(&[a, b]));
        budget = 1;
        assert_eq!(atlas.update(&[b, c], &mut tree, &mut budget), 1);
        assert_eq!(atlas.id, id);
        let pixel = |x| {
            let buffer = render.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = render.device.create_command_encoder(&Default::default());
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    origin: wgpu::Origin3d { x, y: 0, z: 0 },
                    ..atlas.texture.as_image_copy()
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            render.queue.submit([encoder.finish()]);
            buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            render
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            let color: [u8; 4] = buffer.slice(..).get_mapped_range()[..4].try_into().unwrap();
            buffer.unmap();
            color
        };
        assert_eq!(pixel(0), [80, 120, 160, 255]);
        assert_eq!(pixel(256), [120, 160, 200, 255]);
        let separate = PreviewAtlas::new(&render, AtlasLayout::new(&[b, c], b.x));
        assert_ne!(separate.id, id);
        drop(atlas);
        assert!(render.renderer.read().texture(&id).is_none());
        assert!(render.renderer.read().texture(&separate.id).is_some());
    }
}
