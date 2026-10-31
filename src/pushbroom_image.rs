//! GPU preview of the recorded ground strip, with an independent imagery cache.
use crate::pushbroom::ImageStrip;
use crate::tile::{
    fetch_tile, AtlasLayout, DetailAtlas, TileAtlasWriter, TileCacheEntry, TileCoord,
    TileFetchRequest, TileFetchResult, TileQuadTree, TileReuse, MAX_TILE_ZOOM, TILE_SIZE,
};
use eframe::egui;
use egui_wgpu::{wgpu, RenderState};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{mpsc, Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

const MAX_TILES: usize = 256;
const MAX_CAMERAS: usize = 4;
const UPLOADS_PER_FRAME: usize = 8;
const MAX_PENDING_TILES: usize = 16;
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
        let (tx, requests) = mpsc::sync_channel::<TileFetchRequest>(MAX_PENDING_TILES);
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
        // Cover the whole view before queueing finer detail for individual cells.
        for shift in [MAX_TILE_ZOOM, 4, 2, 0] {
            let mut waiting = false;
            for &coord in needed {
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
                if self.tree.has_tile(&source) {
                    continue;
                }
                waiting = true;
                if self.pending.contains(&source)
                    || self.retried.contains_key(&source)
                    || !attempted.insert(source)
                {
                    continue;
                }
                if self.pending.len() >= MAX_PENDING_TILES {
                    return;
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
            if waiting && (shift == 4 || shift == 2) {
                return;
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
    fn from_previous(render: &RenderState, layout: AtlasLayout, previous: Option<&Self>) -> Self {
        let mut atlas = Self::new(render, layout);
        if let Some(previous) = previous.filter(|old| old.atlas.layout.unwrap().z == layout.z) {
            atlas.copy_from(previous);
            atlas.atlas = previous.atlas.clone();
            atlas.atlas.set_layout(layout);
        }
        atlas
    }

    fn copy_from(&self, old: &Self) {
        let mut encoder = self
            .render
            .device
            .create_command_encoder(&Default::default());
        for copy in self
            .atlas
            .layout
            .unwrap()
            .copies_from(old.atlas.layout.unwrap())
        {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    origin: wgpu::Origin3d {
                        x: copy.source[0],
                        y: copy.source[1],
                        z: 0,
                    },
                    ..old.texture.as_image_copy()
                },
                wgpu::TexelCopyTextureInfo {
                    origin: wgpu::Origin3d {
                        x: copy.destination[0],
                        y: copy.destination[1],
                        z: 0,
                    },
                    ..self.texture.as_image_copy()
                },
                wgpu::Extent3d {
                    width: copy.size[0],
                    height: copy.size[1],
                    depth_or_array_layers: 1,
                },
            );
        }
        self.render.queue.submit([encoder.finish()]);
    }

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
    image: ImageLayer,
    overview: ImageLayer,
    view: ImageView,
    level: Option<u8>,
    strip: Option<ImageStrip>,
    frozen: bool,
    show_aois: bool,
    aoi_texture: Option<crate::aoi::gpu::MapTexture>,
    epoch: Option<(u64, u64, u64, u64)>,
}
impl Camera {
    fn observe(&mut self, strip: Option<ImageStrip>, settings: crate::pushbroom::Settings) {
        if self.frozen {
            return;
        }
        let Some(strip) = strip else {
            self.image = ImageLayer::default();
            self.overview = ImageLayer::default();
            self.level = None;
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
            self.image = ImageLayer::default();
            self.overview = ImageLayer::default();
            self.level = None;
        }
        self.epoch = Some(epoch);
        self.strip = Some(strip);
    }
}

#[derive(Default)]
struct ImageLayer {
    atlas: Option<PreviewAtlas>,
    pending_atlas: Option<PreviewAtlas>,
}

impl ImageLayer {
    fn paint(&self, strip: &ImageStrip, rect: egui::Rect, painter: &egui::Painter) -> usize {
        let Some(atlas) = &self.atlas else { return 0 };
        let mesh = strip_mesh(strip, rect, atlas.atlas.layout.unwrap(), atlas.id);
        let vertices = mesh.vertices.len();
        painter.add(egui::Shape::mesh(mesh));
        vertices
    }

    fn update_atlas(
        &mut self,
        render: &RenderState,
        needed: &[TileCoord],
        requested: &[TileCoord],
        tree: &mut TileQuadTree,
        budget: &mut usize,
    ) -> usize {
        let before = *budget;
        let ordered: Vec<_> = needed
            .iter()
            .chain(requested.iter().filter(|c| !needed.contains(c)))
            .copied()
            .collect();
        if let Some(current) = &mut self.atlas {
            let layout = current.atlas.layout.unwrap();
            if needed.iter().all(|coord| layout.contains(*coord)) {
                let resident: Vec<_> = ordered
                    .iter()
                    .copied()
                    .filter(|c| layout.contains(*c))
                    .collect();
                current.update(&resident, tree, budget);
                if current.atlas.covers(needed) {
                    self.pending_atlas = None;
                    return before - *budget;
                }
            }
        }
        let layout = AtlasLayout::new(requested, requested[0].x);
        let pending = self.pending_atlas.get_or_insert_with(|| {
            PreviewAtlas::from_previous(render, layout, self.atlas.as_ref())
        });
        // Keep the padded layout even though visible cells are uploaded first.
        if !requested
            .iter()
            .all(|c| pending.atlas.layout.unwrap().contains(*c))
        {
            *pending = PreviewAtlas::from_previous(render, layout, Some(pending));
        }
        pending.update(&ordered, tree, budget);
        // Never replace a usable image with a partially uploaded texture.
        if pending.atlas.covers(needed) {
            self.atlas = self.pending_atlas.take();
        }
        before - *budget
    }
}

struct ImageView {
    zoom: f64,
    max_zoom: f64,
    center: [f64; 2],
}

impl Default for ImageView {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            max_zoom: f64::INFINITY,
            center: [0.5; 2],
        }
    }
}

impl ImageView {
    fn size(&self, fitted: egui::Rect) -> [f64; 2] {
        [
            f64::from(fitted.width()) * self.zoom,
            f64::from(fitted.height()) * self.zoom,
        ]
    }

    fn image_rect(&self, fitted: egui::Rect) -> egui::Rect {
        let size = self.size(fitted);
        let origin = [
            f64::from(fitted.center().x) - self.center[0] * size[0],
            f64::from(fitted.center().y) - self.center[1] * size[1],
        ];
        egui::Rect::from_min_max(
            egui::pos2(origin[0] as f32, origin[1] as f32),
            egui::pos2((origin[0] + size[0]) as f32, (origin[1] + size[1]) as f32),
        )
    }

    fn zoom_at(
        &mut self,
        fitted: egui::Rect,
        viewport: egui::Rect,
        anchor: egui::Pos2,
        factor: f64,
    ) {
        if factor == 1.0 {
            return;
        }
        let old_size = self.size(fitted);
        let delta = [
            f64::from(anchor.x - fitted.center().x),
            f64::from(anchor.y - fitted.center().y),
        ];
        self.zoom = (self.zoom * factor).clamp(1.0, self.max_zoom);
        let size = self.size(fitted);
        for axis in 0..2 {
            self.center[axis] += delta[axis] / old_size[axis] - delta[axis] / size[axis];
        }
        self.constrain(fitted, viewport);
    }

    fn constrain(&mut self, fitted: egui::Rect, viewport: egui::Rect) {
        let size = self.size(fitted);
        for (axis, window) in [viewport.width(), viewport.height()]
            .into_iter()
            .enumerate()
        {
            let center = &mut self.center[axis];
            let (image, window) = (size[axis], f64::from(window));
            let margin = (window / image * 0.5).min(0.5);
            *center = center.clamp(margin, 1.0 - margin);
        }
    }

    fn pan(&mut self, fitted: egui::Rect, viewport: egui::Rect, delta: egui::Vec2) {
        let size = self.size(fitted);
        self.center[0] -= f64::from(delta.x) / size[0];
        self.center[1] -= f64::from(delta.y) / size[1];
        self.constrain(fitted, viewport);
    }

    fn visible(&self, fitted: egui::Rect, viewport: egui::Rect) -> ImageRegion {
        let size = self.size(fitted);
        let delta = viewport.center() - fitted.center();
        let center = [
            self.center[0] + f64::from(delta.x) / size[0],
            self.center[1] + f64::from(delta.y) / size[1],
        ];
        let half = [
            f64::from(viewport.width()) / size[0] / 2.0,
            f64::from(viewport.height()) / size[1] / 2.0,
        ];
        ImageRegion {
            min: std::array::from_fn(|i| (center[i] - half[i]).max(0.0)),
            max: std::array::from_fn(|i| (center[i] + half[i]).min(1.0)),
        }
    }
}

#[derive(Clone, Copy)]
struct ImageRegion {
    min: [f64; 2],
    max: [f64; 2],
}
impl ImageRegion {
    fn width(&self) -> f64 {
        self.max[0] - self.min[0]
    }
    #[cfg(test)]
    fn height(&self) -> f64 {
        self.max[1] - self.min[1]
    }
}
impl From<egui::Rect> for ImageRegion {
    fn from(rect: egui::Rect) -> Self {
        Self {
            min: [f64::from(rect.left()), f64::from(rect.top())],
            max: [f64::from(rect.right()), f64::from(rect.bottom())],
        }
    }
}

fn native_zoom(
    strip: &ImageStrip,
    swath: f64,
    fitted: egui::Rect,
    viewport: egui::Rect,
    ppp: f32,
) -> f64 {
    let latitude_cos = strip
        .lines
        .first()
        .and_then(|line| uv(line.edges[0]))
        .map_or(1.0, |p| {
            (std::f64::consts::TAU * (0.5 - p[1])).cosh().recip()
        });
    let finest_pixel_m =
        40_075_016.686 * latitude_cos / (f64::from(TILE_SIZE) * f64::from(1u32 << MAX_TILE_ZOOM));
    let pixels = preview_pixels(viewport.size(), ppp)[0] as f64;
    (2.0 * swath * 1000.0 * f64::from(viewport.width())
        / (f64::from(fitted.width()) * pixels * finest_pixel_m))
        .max(1.0)
}

#[derive(Default)]
pub(crate) struct Cameras {
    cameras: HashMap<usize, Camera>,
    cache: Option<ImageCache>,
    aoi_renderer: Option<crate::aoi::gpu::MapRenderer>,
    manual_areas: HashMap<(usize, usize), (u64, Arc<crate::aoi::preview::Preview>)>,
}
impl Cameras {
    pub fn show(&mut self, viewer: &mut crate::viewer::ViewerState, ctx: &egui::Context) {
        let generated: Vec<_> = viewer
            .aoi_generator
            .overlays()
            .into_iter()
            .map(|(preview, opacity)| (Arc::clone(preview), opacity))
            .collect();
        let tab_index = viewer.active_tab_idx;
        if let Some(cache) = &mut self.cache {
            cache.poll();
        }
        let Some(tab) = viewer.tabs.get_mut(viewer.active_tab_idx) else {
            return;
        };
        let settings = tab.settings.pushbroom;
        if !settings.enabled || !settings.show_images {
            self.cameras.clear();
            self.aoi_renderer = None;
            self.manual_areas.clear();
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
        let mut overview_budget = UPLOADS_PER_FRAME;
        let mut used_manual = HashSet::new();
        for (planet_index, planet) in tab.planets.iter_mut().enumerate() {
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
                let mut window = egui::Window::new(format!("Pushbroom: {}", selected.label))
                    .id(egui::Id::new(("pushbroom_image", selected.id)))
                    .open(&mut open)
                    .collapsible(false)
                    .default_size([420.0, 520.0]);
                if let Some((size, zoom)) =
                    trace.as_ref().and_then(|trace| trace.pushbroom_preview())
                {
                    window = window.fixed_pos([24.0, 72.0]).fixed_size(size);
                    camera.view.zoom = f64::from(zoom);
                    if std::env::var("LEO_VIZ_PUSHBROOM_AOI_DIAGNOSTICS")
                        .is_ok_and(|value| value == "1")
                    {
                        camera.show_aois = true;
                    }
                }
                window
                    .show(ctx, |ui| {
                        let started = Instant::now();
                        let mut zoom_factor = 1.0;
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut camera.frozen, "Freeze image");
                            ui.checkbox(&mut camera.show_aois, "Show AOIs");
                        });
                        ui.horizontal(|ui| {
                            if image_button(ui, egui::include_image!("../assets/lucide-zoom-out.svg"), "Zoom out", camera.view.zoom > 1.0) {
                                zoom_factor = 1.0 / 1.5;
                            }
                            if image_button(ui, egui::include_image!("../assets/lucide-zoom-in.svg"), "Zoom in", camera.view.zoom < camera.view.max_zoom) {
                                zoom_factor = 1.5;
                            }
                            if image_button(ui, egui::include_image!("../assets/lucide-scan.svg"), "Fit image", camera.view.zoom != 1.0) {
                                camera.view = ImageView::default();
                            }
                            ui.label(format!("{:.0}%", camera.view.zoom * 100.0));
                        });
                        camera.observe(strip, settings);
                        let available = egui::vec2(ui.available_width().max(1.0), (ui.available_height() - 72.0).max(100.0));
                        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());
                        ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);
                        let Some(strip) = camera.strip.clone() else {
                            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "Waiting for scan", egui::FontId::proportional(13.0), egui::Color32::GRAY);
                            return;
                        };
                        let swath = f64::from_bits(camera.epoch.unwrap().1);
                        let length = strip.lines.last().unwrap().distance_km;
                        let fitted = fit_image(rect, swath, length);
                        camera.view.max_zoom = native_zoom(&strip, swath, fitted, rect, ctx.pixels_per_point());
                        camera.view.zoom = camera.view.zoom.min(camera.view.max_zoom);
                        camera.view.constrain(fitted, rect);
                        camera.view.zoom_at(fitted, rect, rect.center(), zoom_factor);
                        if response.double_clicked() {
                            camera.view = ImageView::default();
                        }
                        if response.hovered() {
                            let (zoom, pointer) = ui.input(|input| {
                                let scroll = if input.modifiers.ctrl || input.modifiers.command { 0.0 } else { input.smooth_scroll_delta.y };
                                (input.zoom_delta() * (scroll * 0.003).exp(), input.pointer.hover_pos())
                            });
                            if let Some(pointer) = pointer {
                                camera.view.zoom_at(fitted, rect, pointer, f64::from(zoom));
                            }
                        }
                        if response.dragged() {
                            camera.view.pan(fitted, rect, response.drag_delta());
                        }
                        if camera.view.zoom > 1.0 {
                            response.on_hover_cursor(egui::CursorIcon::Grab);
                        }
                        let image_rect = camera.view.image_rect(fitted);
                        let visible_rect = image_rect.intersect(rect);
                        let visible = camera.view.visible(fitted, rect);
                        let cropped = crop_strip(&strip, visible);
                        let pixels = preview_pixels(image_rect.intersect(rect).size(), ctx.pixels_per_point());
                        let max_dimension = render.device.limits().max_texture_dimension_2d;
                        let needed = select_tiles(&cropped, swath * visible.width(), pixels[0], max_dimension, camera.level);
                        let requested = padded_tiles(&needed, max_dimension);
                        let mut exact = 0;
                        let mut covered = 0;
                        for coord in &needed {
                            if let Some(z) = cache.tree.best_tile_zoom(coord) {
                                covered += 1;
                                exact += usize::from(z == coord.z);
                            }
                        }
                        if !needed.is_empty() {
                            camera.level = Some(needed[0].z);
                            requests.extend_from_slice(&requested);
                            let coarse = coarse_tiles(&needed);
                            let overview_uploaded = camera.overview.update_atlas(render, &coarse, &coarse, &mut cache.tree, &mut overview_budget);
                            let uploaded = overview_uploaded + camera.image.update_atlas(render, &needed, &requested, &mut cache.tree, &mut budget);
                            if uploaded > 0 {
                                if let Some(trace) = &trace {
                                    trace.record("pushbroom_atlas_update", serde_json::json!({"camera":selected.id,"tiles":uploaded,"target_pixels":pixels}));
                                }
                            }
                            let painter = ui.painter().with_clip_rect(rect.intersect(image_rect));
                            let overview_vertices = camera.overview.paint(&cropped, visible_rect, &painter);
                            let detail_vertices = camera.image.paint(&cropped, visible_rect, &painter);
                            let vertices = overview_vertices + detail_vertices;
                            if camera.show_aois {
                                let key = (tab_index, planet_index);
                                used_manual.insert(key);
                                let mut layers = generated.clone();
                                match manual_areas(&mut self.manual_areas, key, &planet.areas_of_interest, planet.celestial_body.radius_km()) {
                                    Ok(Some(preview)) => layers.push((preview, 1.0)),
                                    Ok(None) => {},
                                    Err(error) => { ui.colored_label(egui::Color32::LIGHT_RED, error); },
                                }
                                let renderer = self.aoi_renderer.get_or_insert_with(|| crate::aoi::gpu::MapRenderer::new(&render.device));
                                if renderer.update(&mut camera.aoi_texture, render, ctx, egui::Id::new(("pushbroom_aois", selected.id)), &layers, AtlasLayout::new(&needed, needed[0].x)) {
                                    if let Some(texture) = &camera.aoi_texture {
                                        painter.add(egui::Shape::mesh(strip_mesh(&cropped, visible_rect, texture.layout, texture.id)));
                                    }
                                }
                            } else {
                                camera.aoi_texture = None;
                            }
                            let bounds = image_rect.intersect(rect) * ctx.pixels_per_point();
                            if let Some(trace) = &trace {
                                    trace.record("pushbroom_frame", serde_json::json!({
                                        "camera":selected.id,"time":strip.time,"vertices":vertices,"size":pixels,
                                        "overview_vertices":overview_vertices,"detail_vertices":detail_vertices,
                                        "show_aois":camera.show_aois,
                                        "image_bounds":[bounds.left(),bounds.top(),bounds.right(),bounds.bottom()],
                                        "source_level":needed[0].z,"covered":covered,"needed":needed.len(),
                                        "frozen":camera.frozen,"zoom":camera.view.zoom,"pending":camera.image.pending_atlas.is_some(),
                                        "ready":camera.image.atlas.as_ref().is_some_and(|atlas| atlas.atlas.covers(&needed)),
                                        "atlas_level":camera.image.atlas.as_ref().map(|atlas| atlas.atlas.layout.unwrap().z),"cpu_ms":started.elapsed().as_secs_f64()*1000.0,
                                        "cache_directory":cache.directory,
                                    }));
                            }
                            if vertices == 0 {
                                ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "Loading imagery", egui::FontId::proportional(13.0), egui::Color32::GRAY);
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
        self.manual_areas.retain(|key, _| used_manual.contains(key));
        if !self.cameras.values().any(|camera| camera.show_aois) {
            self.aoi_renderer = None;
        }
        cache.request(&requests);
        if !active.is_empty() {
            ctx.request_repaint();
        }
    }
}

fn manual_areas(
    cache: &mut HashMap<(usize, usize), (u64, Arc<crate::aoi::preview::Preview>)>,
    key: (usize, usize),
    areas: &[crate::config::AreaOfInterest],
    radius: f64,
) -> Result<Option<Arc<crate::aoi::preview::Preview>>, String> {
    if areas.is_empty() {
        cache.remove(&key);
        return Ok(None);
    }
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    radius.to_bits().hash(&mut hash);
    for area in areas {
        [
            area.lat.to_bits(),
            area.lon.to_bits(),
            area.radius_km.to_bits(),
        ]
        .hash(&mut hash);
        area.color.to_array().hash(&mut hash);
    }
    let hash = hash.finish();
    if cache.get(&key).is_none_or(|(old, _)| *old != hash) {
        cache.insert(
            key,
            (
                hash,
                Arc::new(crate::aoi::preview::Preview::from_areas(areas, radius)?),
            ),
        );
    }
    Ok(cache.get(&key).map(|(_, preview)| Arc::clone(preview)))
}

fn image_button(
    ui: &mut egui::Ui,
    source: egui::ImageSource<'_>,
    tooltip: &str,
    enabled: bool,
) -> bool {
    ui.add_enabled(
        enabled,
        egui::Button::image(
            egui::Image::new(source)
                .fit_to_exact_size(egui::vec2(16.0, 16.0))
                .tint(ui.visuals().text_color()),
        ),
    )
    .on_hover_text(tooltip)
    .clicked()
}

fn crop_strip(strip: &ImageStrip, visible: impl Into<ImageRegion>) -> ImageStrip {
    let visible = visible.into();
    let length = strip.lines.last().map_or(0.0, |line| line.distance_km);
    let start = length * visible.min[1];
    let end = length * visible.max[1];
    let mut lines = Vec::new();
    for pair in strip.lines.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        for distance in [start.max(a.distance_km), end.min(b.distance_km)] {
            if distance < start
                || distance > end
                || distance < a.distance_km
                || distance > b.distance_km
                || lines
                    .last()
                    .is_some_and(|line: &crate::pushbroom::ImageLine| {
                        line.distance_km >= distance - start
                    })
            {
                continue;
            }
            let along =
                (distance - a.distance_km) / (b.distance_km - a.distance_km).max(f64::EPSILON);
            let edges = [visible.min[0], visible.max[0]].map(|across| {
                std::array::from_fn(|axis| {
                    let left = a.edges[0][axis] * (1.0 - along) + b.edges[0][axis] * along;
                    let right = a.edges[1][axis] * (1.0 - along) + b.edges[1][axis] * along;
                    left * (1.0 - across) + right * across
                })
            });
            lines.push(crate::pushbroom::ImageLine {
                edges,
                distance_km: distance - start,
            });
        }
    }
    ImageStrip {
        time: strip.time,
        source: strip.source,
        lines,
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

#[cfg(test)]
fn needed_tiles(
    strip: &ImageStrip,
    swath_km: f64,
    width: usize,
    max_dimension: u32,
) -> Vec<TileCoord> {
    select_tiles(strip, swath_km, width, max_dimension, None)
}

fn select_tiles(
    strip: &ImageStrip,
    swath_km: f64,
    width: usize,
    max_dimension: u32,
    previous: Option<u8>,
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
    let desired = (40_075_016.686 * latitude_cos / (256.0 * meters_per_pixel)).log2();
    // Avoid alternating levels when a moving strip crosses a resolution threshold.
    let mut zoom = previous
        .filter(|z| desired > f64::from(*z) - 1.1 && desired < f64::from(*z) + 0.1)
        .unwrap_or_else(|| desired.ceil().clamp(0.0, f64::from(MAX_TILE_ZOOM)) as u8);
    loop {
        let n = 1i64 << zoom;
        let left = (min_x * n as f64).floor() as i64;
        let right = (max_x * n as f64).floor() as i64;
        let top = ((min_y * n as f64).floor() as i64).clamp(0, n - 1);
        let bottom = ((max_y * n as f64).floor() as i64).clamp(0, n - 1);
        let cols = right - left + 1;
        let rows = bottom - top + 1;
        let limit = if previous == Some(zoom) {
            MAX_TILES
        } else {
            196
        };
        if (cols * rows <= limit as i64
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

fn padded_tiles(needed: &[TileCoord], max_dimension: u32) -> Vec<TileCoord> {
    let Some(first) = needed.first() else {
        return Vec::new();
    };
    let layout = AtlasLayout::new(needed, first.x);
    let n = 1 << layout.z;
    let cols = (layout.cols + 2).min(n);
    let x = if cols < n { (layout.x + n - 1) % n } else { 0 };
    let top = layout.y.saturating_sub(1);
    let bottom = (layout.y + layout.rows + 1).min(n);
    let rows = bottom - top;
    if cols * rows > MAX_TILES as u32
        || cols * TILE_SIZE > max_dimension
        || rows * TILE_SIZE > max_dimension
    {
        return needed.to_vec();
    }
    (top..bottom)
        .flat_map(|y| {
            (0..cols).map(move |col| TileCoord {
                x: (x + col) % n,
                y,
                z: layout.z,
            })
        })
        .collect()
}

fn coarse_tiles(needed: &[TileCoord]) -> Vec<TileCoord> {
    let mut seen = HashSet::new();
    needed
        .iter()
        .filter_map(|coord| {
            let shift = coord.z.min(4);
            let parent = TileCoord {
                x: coord.x >> shift,
                y: coord.y >> shift,
                z: coord.z - shift,
            };
            seen.insert(parent).then_some(parent)
        })
        .collect()
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
                    let center = f64::from(layout.x) + f64::from(layout.cols) * 0.5;
                    let unwrapped = center + (u * n - center + n * 0.5).rem_euclid(n) - n * 0.5;
                    let tex_u = (unwrapped - f64::from(layout.x)) / f64::from(layout.cols);
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
                    // An older ready atlas may cover only part of the new view.
                    // Clip geographically instead of stretching its edge pixels.
                    let vertices = clip_to_texture(vertices);
                    if vertices.len() >= 3 {
                        for index in 1..vertices.len() as u32 - 1 {
                            mesh.indices.extend([base, base + index, base + index + 1]);
                        }
                        mesh.vertices.extend(vertices);
                    }
                }
            }
        }
    }
    mesh
}

fn clip_to_texture(mut polygon: Vec<egui::epaint::Vertex>) -> Vec<egui::epaint::Vertex> {
    for (axis, bound, lower) in [
        (0, 0.0, true),
        (0, 1.0, false),
        (1, 0.0, true),
        (1, 1.0, false),
    ] {
        let value =
            |vertex: &egui::epaint::Vertex| if axis == 0 { vertex.uv.x } else { vertex.uv.y };
        let inside = |vertex: &egui::epaint::Vertex| {
            if lower {
                value(vertex) >= bound
            } else {
                value(vertex) <= bound
            }
        };
        if polygon.iter().all(inside) {
            continue;
        }
        let mut clipped = Vec::with_capacity(polygon.len() + 1);
        if let Some(mut previous) = polygon.last().copied() {
            for &current in &polygon {
                if inside(&previous) != inside(&current) {
                    let t = (bound - value(&previous)) / (value(&current) - value(&previous));
                    let mut intersection = egui::epaint::Vertex {
                        pos: previous.pos + (current.pos - previous.pos) * t,
                        uv: previous.uv + (current.uv - previous.uv) * t,
                        color: egui::Color32::WHITE,
                    };
                    if axis == 0 {
                        intersection.uv.x = bound
                    } else {
                        intersection.uv.y = bound
                    }
                    clipped.push(intersection);
                }
                if inside(&current) {
                    clipped.push(current);
                }
                previous = current;
            }
        }
        polygon = clipped;
    }
    polygon
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
    fn downloads_prioritize_complete_coverage_and_bound_the_backlog() {
        let (tx, requests) = mpsc::sync_channel(MAX_PENDING_TILES);
        let (_, rx) = mpsc::channel();
        let mut cache = ImageCache {
            tree: TileQuadTree::new(512),
            directory: Default::default(),
            pending: HashSet::new(),
            missing: HashMap::new(),
            retried: HashMap::new(),
            wanted: Arc::new(RwLock::new(HashSet::new())),
            tx,
            rx,
            trace: None,
        };
        let needed: Vec<_> = (2000..2016)
            .flat_map(|y| (1000..1016).map(move |x| TileCoord { x, y, z: 12 }))
            .collect();
        for max_level in [8, 10, 12] {
            cache.request(&needed);
            let queued: Vec<_> = requests.try_iter().collect();
            assert!(!queued.is_empty());
            assert!(queued.len() <= MAX_PENDING_TILES);
            assert!(queued.iter().all(|r| r.coord.z <= max_level));
            for request in queued {
                cache.pending.remove(&request.coord);
                cache.tree.insert(
                    request.coord,
                    TileCacheEntry {
                        pixels: vec![[1, 2, 3]],
                        width: 1,
                        height: 1,
                    },
                );
            }
        }
        cache.request(&[]);
        assert!(cache.wanted.read().unwrap().is_empty());
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
    fn zoom_keeps_the_point_under_the_cursor_and_pan_stays_in_bounds() {
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 600.0));
        let fitted = fit_image(rect, 20.0, 50.0);
        let mut view = ImageView::default();
        view.zoom_at(fitted, rect, rect.center(), 2.0);
        let anchor = rect.center() + egui::vec2(20.0, 40.0);
        let old = view.image_rect(fitted);
        let point = (anchor - old.min) / old.size();
        view.zoom_at(fitted, rect, anchor, 2.0);
        let new = view.image_rect(fitted);
        let after = (anchor - new.min) / new.size();
        assert!((after - point).length() < 1e-6);
        assert_eq!(view.zoom, 4.0);
        view.pan(fitted, rect, egui::vec2(1e6, -1e6));
        let visible = view.visible(fitted, rect);
        assert!(visible.min[0] >= -1e-6 && visible.min[1] >= -1e-6);
        assert!(visible.max[0] <= 1.000001 && visible.max[1] <= 1.000001);
        assert!(visible.width() < 0.5 && visible.height() < 0.5);
        view.zoom_at(fitted, rect, rect.center(), 0.0001);
        assert_eq!(view.zoom, 1.0);
        assert_eq!(view.center, [0.5; 2]);
    }

    #[test]
    fn zoom_requests_only_the_visible_area_at_higher_detail() {
        for (left, right) in [(-0.2, 0.2), (179.8, -179.8)] {
            let full = strip(left, right);
            let cropped = crop_strip(
                &full,
                egui::Rect::from_min_max(egui::pos2(0.4, 0.4), egui::pos2(0.6, 0.6)),
            );
            assert!((cropped.lines.last().unwrap().distance_km - 9.0).abs() < 1e-5);
            assert_eq!(cropped.lines[0].distance_km, 0.0);
            let fitted = needed_tiles(&full, 44.0, 1000, 8192);
            let zoomed = needed_tiles(&cropped, 8.8, 1000, 8192);
            assert!(zoomed[0].z > fitted[0].z);
            assert!(zoomed.len() <= MAX_TILES);
            let padded = padded_tiles(&zoomed, 8192);
            let layout = AtlasLayout::new(&padded, padded[0].x);
            assert!(zoomed.iter().all(|coord| layout.contains(*coord)));
            assert!(padded.len() <= MAX_TILES);
        }
    }

    #[test]
    fn tile_levels_do_not_oscillate_near_a_resolution_threshold() {
        let strip = strip(-0.2, 0.2);
        let before = select_tiles(&strip, 44.0, 1150, 8192, None);
        let after = select_tiles(&strip, 44.0, 1155, 8192, Some(before[0].z));
        assert_eq!(before[0].z, after[0].z);
        let back = select_tiles(&strip, 44.0, 1145, 8192, Some(after[0].z));
        assert_eq!(before[0].z, back[0].z);
    }

    #[test]
    fn preview_zoom_can_reach_the_highest_supported_tile_level() {
        let strip = strip(-0.2, 0.2);
        let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
        let fitted = fit_image(viewport, 44.0, 45.0);
        let mut view = ImageView {
            max_zoom: native_zoom(&strip, 44.0, fitted, viewport, 2.0),
            ..Default::default()
        };
        assert!(view.max_zoom > 32.0);
        view.zoom_at(fitted, viewport, viewport.center(), 1e9);
        assert_eq!(view.zoom, view.max_zoom);
        let visible = view.visible(fitted, viewport);
        let cropped = crop_strip(&strip, visible);
        let pixels = preview_pixels(viewport.size(), 2.0);
        let needed = needed_tiles(&cropped, 44.0 * visible.width(), pixels[0], 8192);
        assert_eq!(needed[0].z, MAX_TILE_ZOOM);
        assert!(needed.len() <= MAX_TILES);
        let layout = AtlasLayout::new(&needed, needed[0].x);
        let mesh = strip_mesh(&cropped, viewport, layout, egui::TextureId::User(1));
        assert!(mesh.is_valid() && !mesh.indices.is_empty());
        assert!(mesh
            .vertices
            .iter()
            .all(|v| viewport.expand(0.001).contains(v.pos)));
        let before = view.center;
        view.pan(fitted, viewport, egui::vec2(0.1, 0.0));
        assert!(view.center[0] < before[0]);
        assert_eq!(view.center[1], before[1]);
    }

    #[test]
    fn fallback_mesh_is_clipped_instead_of_stretching_atlas_edges() {
        let strip = strip(-0.2, 0.2);
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(440.0, 450.0));
        let needed = needed_tiles(&strip, 44.0, 640, 8192);
        let mut layout = AtlasLayout::new(&needed, needed[0].x);
        layout.cols = 1;
        let mesh = strip_mesh(&strip, rect, layout, egui::TextureId::User(1));
        assert!(!mesh.indices.is_empty());
        assert!(mesh
            .vertices
            .iter()
            .all(|v| v.uv.x >= 0.0 && v.uv.x <= 1.0 && v.uv.y >= 0.0 && v.uv.y <= 1.0));
        assert!(mesh.vertices.iter().all(|v| v.pos.x < rect.right() - 1.0));
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

    fn render_state() -> RenderState {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let renderer =
            egui_wgpu::Renderer::new(&device, wgpu::TextureFormat::Rgba8Unorm, Default::default());
        RenderState {
            adapter,
            available_adapters: Vec::new(),
            device,
            queue,
            target_format: wgpu::TextureFormat::Rgba8Unorm,
            renderer: Arc::new(egui::mutex::RwLock::new(renderer)),
        }
    }

    #[test]
    #[ignore = "requires a native GPU adapter"]
    fn gpu_preview_keeps_the_ready_image_during_partial_uploads_and_zoom_changes() {
        let render = render_state();
        let a = TileCoord { x: 1, y: 1, z: 4 };
        let mut tree = TileQuadTree::new(32);
        tree.insert(
            a,
            TileCacheEntry {
                pixels: vec![[40, 80, 120]; 256 * 256],
                width: 256,
                height: 256,
            },
        );
        let mut camera = ImageLayer::default();
        camera.update_atlas(&render, &[a], &[a], &mut tree, &mut 1);
        let id = camera.atlas.as_ref().unwrap().id;
        let children: Vec<_> = (2..4)
            .flat_map(|y| (2..4).map(move |x| TileCoord { x, y, z: 5 }))
            .collect();
        for _ in 0..2 {
            assert_eq!(
                camera.update_atlas(&render, &children, &children, &mut tree, &mut 1),
                1
            );
            assert_eq!(camera.atlas.as_ref().unwrap().id, id);
            assert!(camera.pending_atlas.is_some());
        }
        // Returning to the original level does not discard the displayed texture.
        camera.update_atlas(&render, &[a], &[a], &mut tree, &mut 1);
        assert_eq!(camera.atlas.as_ref().unwrap().id, id);
        assert!(camera.pending_atlas.is_none());
        for _ in 0..3 {
            camera.update_atlas(&render, &children, &children, &mut tree, &mut 1);
            assert_eq!(camera.atlas.as_ref().unwrap().id, id);
        }
        camera.update_atlas(&render, &children, &children, &mut tree, &mut 1);
        let current = camera.atlas.as_ref().unwrap();
        assert_ne!(current.id, id);
        assert!(current.atlas.covers(&children));
        assert!(camera.pending_atlas.is_none());
        let current_id = current.id;
        drop(camera);
        assert!(render.renderer.read().texture(&current_id).is_none());
        assert!(render.renderer.read().texture(&id).is_none());
    }

    #[test]
    #[ignore = "requires a native GPU adapter"]
    fn gpu_altitude_edits_reuse_the_loaded_preview_without_uploads() {
        use crate::{
            celestial::CelestialBody,
            config::ConstellationConfig,
            pushbroom::{History, Settings},
        };
        let render = render_state();
        let body = CelestialBody::Earth;
        let settings = Settings {
            enabled: true,
            ..Default::default()
        };
        let mut config = ConstellationConfig::new(0);
        config.sats_per_plane = 1;
        config.num_planes = 1;
        config.altitude_km = 550.0;
        config.inclination = 53.0;
        config.preset = crate::config::Preset::Starlink;
        let mut history = History::default();
        let update = |history: &mut History, config: &ConstellationConfig, time| {
            let orbit = config.constellation(
                body.radius_km(),
                body.mu(),
                body.j2(),
                body.equatorial_radius_km(),
            );
            let positions = orbit.satellite_positions(time);
            history.update(
                time,
                &nalgebra::Matrix3::identity(),
                body.radius_km(),
                &[(orbit, positions, 0, 0, 0, config.preset_name().into())],
                settings,
            );
        };
        for time in 0..=10 {
            update(&mut history, &config, time as f64);
        }
        let mut camera = Camera::default();
        camera.observe(
            history.image_strip(0, 0, 0, settings, body.radius_km()),
            settings,
        );
        let needed = needed_tiles(camera.strip.as_ref().unwrap(), settings.swath_km, 640, 8192);
        let mut tree = TileQuadTree::new(16);
        tree.insert(
            TileCoord { x: 0, y: 0, z: 0 },
            TileCacheEntry {
                pixels: vec![[40, 80, 120]; 256 * 256],
                width: 256,
                height: 256,
            },
        );
        for _ in 0..32 {
            camera
                .image
                .update_atlas(&render, &needed, &needed, &mut tree, &mut 8);
        }
        let id = camera.image.atlas.as_ref().unwrap().id;
        camera.level = Some(needed[0].z);
        camera.view.zoom = 2.0;
        for altitude in [550.1, 550.2, 550.1, 550.0, 549.9] {
            let previous = config.altitude_km;
            config.altitude_km = altitude;
            config.preset = crate::config::Preset::None;
            config.preserve_phase_after_altitude_change(previous, 10.0, body);
            update(&mut history, &config, 10.0);
            camera.observe(
                history.image_strip(0, 0, 0, settings, body.radius_km()),
                settings,
            );
            assert_eq!(camera.image.atlas.as_ref().unwrap().id, id);
            assert_eq!(camera.level, Some(needed[0].z));
            assert_eq!(camera.view.zoom, 2.0);
            let current =
                needed_tiles(camera.strip.as_ref().unwrap(), settings.swath_km, 640, 8192);
            assert_eq!(current, needed);
            assert_eq!(
                camera
                    .image
                    .update_atlas(&render, &current, &current, &mut tree, &mut 8),
                0
            );
        }
    }

    #[test]
    #[ignore = "requires a native GPU adapter"]
    fn gpu_overview_keeps_moving_imagery_visible_when_detail_falls_behind() {
        let render = render_state();
        let mut tree = TileQuadTree::new(32);
        tree.insert(
            TileCoord { x: 0, y: 0, z: 0 },
            TileCacheEntry {
                pixels: vec![[40, 80, 120]; 256 * 256],
                width: 256,
                height: 256,
            },
        );
        let original = strip(-0.2, 0.2);
        let needed = needed_tiles(&original, 44.0, 640, 8192);
        let coarse = coarse_tiles(&needed);
        let mut image = ImageLayer::default();
        let mut overview = ImageLayer::default();
        for _ in 0..32 {
            image.update_atlas(&render, &needed, &needed, &mut tree, &mut 8);
        }
        overview.update_atlas(&render, &coarse, &coarse, &mut tree, &mut 8);
        let moved = strip(0.7, 1.1);
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(440.0, 450.0));
        let mesh = |layer: &ImageLayer| {
            let atlas = layer.atlas.as_ref().unwrap();
            strip_mesh(&moved, rect, atlas.atlas.layout.unwrap(), atlas.id)
        };
        assert!(mesh(&image).indices.is_empty());
        let fallback = mesh(&overview);
        assert!(!fallback.indices.is_empty());
        assert!(fallback.is_valid());
        assert!(fallback.vertices.iter().any(|v| v.pos == rect.min));
        assert!(fallback.vertices.iter().any(|v| v.pos == rect.max));
    }

    #[test]
    #[ignore = "requires a native GPU adapter"]
    fn gpu_preview_preserves_tiles_when_moving_and_releases_its_texture() {
        let render = render_state();
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
