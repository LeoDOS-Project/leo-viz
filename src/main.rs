mod celestial;
mod geo;
mod tle;
mod walker;

use celestial::{CelestialBody, Skin, TextureResolution};
use geo::{CityLabel, GeoOverlayData, GeoLoadState};
#[cfg(not(target_arch = "wasm32"))]
use geo::{load_geo_overlay, dirs_cache};
use tle::{TlePreset, TleSatellite, TleShell, TleLoadState, mean_motion_to_altitude_km, SECONDS_PER_DAY};
#[cfg(not(target_arch = "wasm32"))]
use tle::fetch_tle_data;
use walker::{WalkerType, WalkerConstellation, SatelliteState};
use eframe::{egui, egui_glow, glow};
use egui::mutex::Mutex;
use egui_dock::{DockArea, DockState, NodeIndex, SurfaceIndex, TabViewer};
use egui_dock::tab_viewer::OnCloseResponse;
use egui_plot::{Line, Plot, PlotImage, PlotPoints, PlotPoint, Points, Polygon, Text};
use nalgebra::{Matrix3, Vector3};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::sync::{Arc, mpsc};
use chrono::{DateTime, Utc, Local, Duration};
use glow::HasContext as _;

const DAYS_PER_JULIAN_CENTURY: f64 = 36525.0;

const GMST_BASE_DEG: f64 = 280.46061837;
const GMST_ROTATION_PER_DAY: f64 = 360.98564736629;
const GMST_CORRECTION: f64 = 0.000387933;

const SOLAR_DECLINATION_MAX: f64 = -23.45;
const DAYS_PER_YEAR: f64 = 365.0;

const EARTH_VISUAL_SCALE: f64 = 0.95;

fn asset_path(relative: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

#[cfg(not(target_arch = "wasm32"))]
fn decode_jpeg_pixels(bytes: &[u8]) -> Option<(Vec<[u8; 3]>, u32, u32)> {
    use std::io::Cursor;
    let img = image::load(Cursor::new(bytes), image::ImageFormat::Jpeg).ok()?;
    let rgb = img.to_rgb8();
    let w = rgb.width();
    let h = rgb.height();
    let pixels: Vec<[u8; 3]> = rgb.pixels().map(|p| p.0).collect();
    Some((pixels, w, h))
}

#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::JsCast;

#[derive(Clone)]
#[allow(dead_code)]
enum TextureLoadState {
    Loading,
    Loaded(Arc<EarthTexture>),
    Failed(String),
}

const COLOR_ASCENDING: egui::Color32 = egui::Color32::from_rgb(200, 120, 50);
const COLOR_DESCENDING: egui::Color32 = egui::Color32::from_rgb(50, 100, 180);

#[derive(Clone)]
struct EarthTexture {
    width: u32,
    height: u32,
    pixels: Vec<[u8; 3]>,
}

impl EarthTexture {
    fn load() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let bytes = std::fs::read(asset_path("textures/earth/earth_8k.jpg"))
                .expect("Failed to read textures/earth/earth_8k.jpg");
            Self::from_bytes(&bytes).expect("Failed to load Earth texture")
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self { width: 2, height: 1, pixels: vec![[30, 60, 120], [30, 60, 120]] }
        }
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        use std::io::Cursor;
        let cursor = Cursor::new(bytes);
        let mut reader = image::ImageReader::new(cursor)
            .with_guessed_format()
            .map_err(|e| format!("Failed to guess format: {}", e))?;
        reader.no_limits();
        let img = reader.decode()
            .map_err(|e| format!("Failed to decode image: {}", e))?
            .to_rgb8();
        let width = img.width();
        let height = img.height();
        let pixels: Vec<[u8; 3]> = img.pixels().map(|p| p.0).collect();
        Ok(Self { width, height, pixels })
    }

    fn downscale(&self, factor: u32) -> Self {
        if factor <= 1 {
            return self.clone();
        }
        let new_width = self.width / factor;
        let new_height = self.height / factor;
        let mut pixels = Vec::with_capacity((new_width * new_height) as usize);

        for y in 0..new_height {
            for x in 0..new_width {
                let mut r_sum = 0u32;
                let mut g_sum = 0u32;
                let mut b_sum = 0u32;
                for dy in 0..factor {
                    for dx in 0..factor {
                        let sx = x * factor + dx;
                        let sy = y * factor + dy;
                        let idx = (sy * self.width + sx) as usize;
                        let [r, g, b] = self.pixels[idx];
                        r_sum += r as u32;
                        g_sum += g as u32;
                        b_sum += b as u32;
                    }
                }
                let count = factor * factor;
                pixels.push([
                    (r_sum / count) as u8,
                    (g_sum / count) as u8,
                    (b_sum / count) as u8,
                ]);
            }
        }
        Self { width: new_width, height: new_height, pixels }
    }

    fn sample(&self, u: f64, v: f64) -> [u8; 3] {
        let x = ((u * self.width as f64) as u32).min(self.width - 1);
        let y = ((v * self.height as f64) as u32).min(self.height - 1);
        self.pixels[(y * self.width + x) as usize]
    }

    fn render_sphere(&self, size: usize, rot: &Matrix3<f64>, flattening: f64) -> egui::ColorImage {
        let mut pixels = vec![egui::Color32::TRANSPARENT; size * size];
        let center = size as f64 / 2.0;
        let radius = center;
        let inv_rot = rot.transpose();
        let polar_scale = 1.0 - flattening;

        for py in 0..size {
            for px in 0..size {
                let dx = px as f64 - center;
                let dy = py as f64 - center;
                let dy_scaled = dy / polar_scale;
                let dist_sq = dx * dx + dy_scaled * dy_scaled;

                if dist_sq < radius * radius {
                    let z = (radius * radius - dist_sq).sqrt();
                    let x = dx / radius;
                    let y = -dy_scaled / radius;
                    let z = z / radius;

                    let v = inv_rot * Vector3::new(x, y, z);

                    let lat = v.y.asin();
                    let lon = (-v.z).atan2(v.x);

                    let u = (lon + PI) / (2.0 * PI);
                    let vt = (PI / 2.0 - lat) / PI;

                    let [r, g, b] = self.sample(u, vt);

                    let shade = (0.3 + 0.7 * z.max(0.0)) as f32;
                    let r = (r as f32 * shade) as u8;
                    let g = (g as f32 * shade) as u8;
                    let b = (b as f32 * shade) as u8;

                    pixels[py * size + px] = egui::Color32::from_rgb(r, g, b);
                }
            }
        }

        egui::ColorImage {
            size: [size, size],
            pixels,
            source_size: egui::Vec2::ZERO,
        }
    }
}

struct RingTexture {
    width: u32,
    height: u32,
    pixels: Vec<[u8; 4]>,
}

impl RingTexture {
    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        use std::io::Cursor;
        let cursor = Cursor::new(bytes);
        let mut reader = image::ImageReader::new(cursor)
            .with_guessed_format()
            .map_err(|e| format!("Failed to guess format: {}", e))?;
        reader.no_limits();
        let img = reader.decode()
            .map_err(|e| format!("Failed to decode image: {}", e))?
            .to_rgba8();
        let width = img.width();
        let height = img.height();
        let pixels: Vec<[u8; 4]> = img.pixels().map(|p| p.0).collect();
        Ok(Self { width, height, pixels })
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
struct TileCoord {
    x: u32,
    y: u32,
    z: u8,
}

#[derive(Clone)]
struct DetailBounds {
    min_lon: f64,
    max_lon: f64,
    min_lat: f64,
    max_lat: f64,
}

struct DetailTexture {
    width: u32,
    height: u32,
    bounds: DetailBounds,
    gl_texture: Option<glow::Texture>,
}

struct TileFetchResult {
    coord: TileCoord,
    pixels: Vec<[u8; 3]>,
    width: u32,
    height: u32,
}

struct TileCacheEntry {
    pixels: Vec<[u8; 3]>,
    width: u32,
    height: u32,
}

struct TileNode {
    tile: Option<TileCacheEntry>,
    children: [Option<Box<TileNode>>; 4],
    last_used: u64,
}

impl TileNode {
    fn new() -> Self {
        TileNode { tile: None, children: [None, None, None, None], last_used: 0 }
    }

    fn is_leaf(&self) -> bool {
        self.children.iter().all(|c| c.is_none())
    }
}

struct TileQuadTree {
    root: TileNode,
    tile_count: usize,
    max_tiles: usize,
    access_counter: u64,
}

impl TileQuadTree {
    fn new(max_tiles: usize) -> Self {
        TileQuadTree { root: TileNode::new(), tile_count: 0, max_tiles, access_counter: 0 }
    }

    fn child_index(x: u32, y: u32, z: u8, depth: u8) -> usize {
        let bit_x = ((x >> (z - 1 - depth)) & 1) as usize;
        let bit_y = ((y >> (z - 1 - depth)) & 1) as usize;
        bit_x | (bit_y << 1)
    }

    fn insert(&mut self, coord: TileCoord, entry: TileCacheEntry) {
        self.access_counter += 1;
        let mut node = &mut self.root;
        for depth in 0..coord.z {
            let idx = Self::child_index(coord.x, coord.y, coord.z, depth);
            node = node.children[idx].get_or_insert_with(|| Box::new(TileNode::new()));
        }
        if node.tile.is_none() {
            self.tile_count += 1;
        }
        node.tile = Some(entry);
        node.last_used = self.access_counter;
        self.evict_if_needed();
    }

    fn best_tile_zoom(&mut self, coord: &TileCoord) -> Option<u8> {
        self.access_counter += 1;
        let ac = self.access_counter;
        let mut best_z: Option<u8> = None;
        let mut node = &mut self.root;
        if node.tile.is_some() {
            node.last_used = ac;
            best_z = Some(0);
        }
        for depth in 0..coord.z {
            let idx = Self::child_index(coord.x, coord.y, coord.z, depth);
            match &mut node.children[idx] {
                Some(child) => {
                    node = child.as_mut();
                    if node.tile.is_some() {
                        node.last_used = ac;
                        best_z = Some(depth + 1);
                    }
                }
                None => break,
            }
        }
        best_z
    }

    fn get_tile_at(&self, coord: &TileCoord) -> Option<&TileCacheEntry> {
        let mut node = &self.root;
        for depth in 0..coord.z {
            let idx = Self::child_index(coord.x, coord.y, coord.z, depth);
            match &node.children[idx] {
                Some(child) => node = child,
                None => return None,
            }
        }
        node.tile.as_ref()
    }

    fn has_tile(&self, coord: &TileCoord) -> bool {
        self.get_tile_at(coord).is_some()
    }

    fn evict_if_needed(&mut self) {
        if self.tile_count <= self.max_tiles {
            return;
        }
        let target = self.max_tiles * 3 / 4;
        let mut candidates: Vec<(u64, Vec<usize>)> = Vec::new();
        Self::collect_evictable(&self.root, &mut Vec::new(), &mut candidates);
        candidates.sort_by_key(|(last_used, _)| *last_used);
        let to_remove = self.tile_count.saturating_sub(target);
        for (_, path) in candidates.iter().take(to_remove) {
            Self::remove_at(&mut self.root, path);
            self.tile_count -= 1;
        }
    }

    fn collect_evictable(node: &TileNode, path: &mut Vec<usize>, out: &mut Vec<(u64, Vec<usize>)>) {
        if node.is_leaf() && node.tile.is_some() {
            out.push((node.last_used, path.clone()));
            return;
        }
        for (i, child) in node.children.iter().enumerate() {
            if let Some(c) = child {
                path.push(i);
                Self::collect_evictable(c, path, out);
                path.pop();
            }
        }
    }

    fn remove_at(node: &mut TileNode, path: &[usize]) {
        if path.is_empty() {
            node.tile = None;
            return;
        }
        let idx = path[0];
        if let Some(child) = &mut node.children[idx] {
            Self::remove_at(child, &path[1..]);
            if child.tile.is_none() && child.is_leaf() {
                node.children[idx] = None;
            }
        }
    }
}

struct TileOverlayState {
    enabled: bool,
    tile_tree: TileQuadTree,
    #[cfg(not(target_arch = "wasm32"))]
    disk_cache_dir: std::path::PathBuf,
    detail_texture: Option<DetailTexture>,
    #[cfg(not(target_arch = "wasm32"))]
    fetch_tx: mpsc::Sender<(TileCoord, std::path::PathBuf, u64)>,
    result_rx: mpsc::Receiver<TileFetchResult>,
    last_zoom: u8,
    pending_tiles: HashSet<TileCoord>,
    needed_tiles: Vec<TileCoord>,
    dirty: bool,
    #[cfg(not(target_arch = "wasm32"))]
    fetch_generation: std::sync::Arc<std::sync::atomic::AtomicU64>,
    generation: u64,
    tile_x_origin: u32,
    last_compose: std::time::Instant,
    base_fetched: bool,
    compose_buffer: Vec<[u8; 4]>,
}

fn lon_lat_to_tile(lon: f64, lat: f64, z: u8) -> TileCoord {
    let n = (1u32 << z) as f64;
    let x = ((lon + 180.0) / 360.0 * n).floor() as i64;
    let ni = n as i64;
    let x = (((x % ni) + ni) % ni) as u32;
    let lat_rad = lat.to_radians();
    let y = ((1.0 - lat_rad.tan().asinh() / PI) / 2.0 * n).floor() as u32;
    TileCoord {
        x,
        y: y.min(n as u32 - 1),
        z,
    }
}

fn tile_to_lon_lat(t: &TileCoord) -> (f64, f64) {
    let n = (1u32 << t.z) as f64;
    let lon = t.x as f64 / n * 360.0 - 180.0;
    let lat = (PI * (1.0 - 2.0 * t.y as f64 / n)).sinh().atan().to_degrees();
    (lon, lat)
}

fn camera_zoom_to_tile_zoom(camera_zoom: f64) -> u8 {
    let z = (camera_zoom.log2() + 4.0).floor() as i32;
    z.clamp(0, 18) as u8
}

struct SphereRenderer {
    program: glow::Program,
    vertex_array: glow::VertexArray,
    textures: HashMap<(CelestialBody, Skin, TextureResolution), glow::Texture>,
    cloud_textures: HashMap<TextureResolution, glow::Texture>,
    night_texture: Option<glow::Texture>,
    star_texture: Option<glow::Texture>,
    milky_way_texture: Option<glow::Texture>,
    ring_textures: HashMap<CelestialBody, glow::Texture>,
}

impl SphereRenderer {
    fn new(gl: &glow::Context) -> Self {
        let shader_version = if cfg!(target_arch = "wasm32") {
            "#version 300 es"
        } else {
            "#version 330"
        };

        unsafe {
            let program = gl.create_program().expect("Cannot create program");

            let vertex_shader_source = r#"
                const vec2 verts[4] = vec2[4](
                    vec2(-1.0, -1.0),
                    vec2( 1.0, -1.0),
                    vec2(-1.0,  1.0),
                    vec2( 1.0,  1.0)
                );
                out vec2 v_uv;
                void main() {
                    v_uv = verts[gl_VertexID] * 0.5 + 0.5;
                    gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
                }
            "#;

            let fragment_shader_source = r#"
                precision highp float;
                in vec2 v_uv;
                out vec4 out_color;

                uniform sampler2D u_texture;
                uniform sampler2D u_clouds;
                uniform sampler2D u_night;
                uniform sampler2D u_detail;
                uniform sampler2D u_stars;
                uniform mat3 u_inv_rotation;
                uniform float u_flattening;
                uniform float u_aspect;
                uniform float u_scale;
                uniform float u_atmosphere;
                uniform float u_show_clouds;
                uniform float u_show_day_night;
                uniform vec3 u_sun_dir;
                uniform vec4 u_detail_bounds;
                uniform float u_use_detail;
                uniform float u_show_stars;
                uniform vec3 u_bg_color;
                uniform sampler2D u_ring_texture;
                uniform float u_has_rings;
                uniform float u_ring_inner;
                uniform float u_ring_outer;
                uniform float u_adams_arc;
                uniform float u_epsilon_wobble;

                const float PI = 3.14159265359;
                const vec3 ATMO_COLOR = vec3(0.4, 0.7, 1.0);
                const float ATMO_THICKNESS = 0.06;

                void main() {
                    vec2 centered = (v_uv - 0.5) * 2.0;
                    centered.x *= u_aspect;
                    centered /= u_scale;

                    float b = 1.0 - u_flattening;
                    float b2 = b * b;

                    vec3 O = u_inv_rotation * vec3(centered.x, centered.y, 0.0);
                    vec3 D = u_inv_rotation * vec3(0.0, 0.0, -1.0);

                    float A = D.x*D.x + D.y*D.y/b2 + D.z*D.z;
                    float B = 2.0 * (O.x*D.x + O.y*D.y/b2 + O.z*D.z);
                    float C = O.x*O.x + O.y*O.y/b2 + O.z*O.z - 1.0;

                    float discriminant = B*B - 4.0*A*C;

                    float screen_dist = length(centered);
                    float atmo_outer = 1.0 + ATMO_THICKNESS * u_atmosphere;

                    float lat_ortho = 0.0;
                    float lon_ortho = 0.0;
                    vec3 normal_ortho = vec3(0.0, 0.0, 1.0);
                    bool ortho_hit = discriminant >= 0.0;

                    if (ortho_hit) {
                        float t = (-B - sqrt(discriminant)) / (2.0 * A);
                        vec3 world_pt = O + t * D;
                        lat_ortho = asin(clamp(world_pt.y / b, -1.0, 1.0));
                        lon_ortho = atan(-world_pt.z, world_pt.x);
                        normal_ortho = normalize(vec3(world_pt.x, world_pt.y / b2, world_pt.z));
                    }

                    float lat, lon;
                    vec3 normal;
                    float alpha;

                    float t_sphere = ortho_hit ? (-B - sqrt(discriminant)) / (2.0 * A) : 1e10;

                    float ring_alpha = 0.0;
                    vec3 ring_color = vec3(0.0);
                    float t_ring = 1e10;
                    if (u_has_rings > 0.5 && abs(D.y) > 0.0001) {
                        float t_disc = -O.y / D.y;
                        vec3 rh = O + t_disc * D;
                        float r = length(vec2(rh.x, rh.z));
                        if (u_epsilon_wobble > 0.5) {
                            float theta = atan(rh.z, rh.x);
                            float eps_center = 2.017;
                            float eps_zone = 0.06;
                            float prox = 1.0 - smoothstep(0.0, eps_zone, abs(r - eps_center));
                            if (prox > 0.0) {
                                float radial_shift = eps_center * 0.025 * cos(theta);
                                r -= radial_shift * prox;
                                float width_scale = 1.0 + 0.7 * cos(theta);
                                float dr = r - eps_center;
                                r = eps_center + dr / mix(1.0, width_scale, prox);
                            }
                        }
                        if (r >= u_ring_inner && r <= u_ring_outer) {
                            float ru = (r - u_ring_inner) / (u_ring_outer - u_ring_inner);
                            vec4 rs = texture(u_ring_texture, vec2(ru, 0.5));
                            ring_color = rs.rgb;
                            ring_alpha = rs.a;
                            if (u_adams_arc > 0.5 && ru > 0.82) {
                                float ang = atan(rh.z, rh.x);
                                float deg = ang * 180.0 / PI;
                                if (deg < 0.0) deg += 360.0;
                                float arc = 0.0;
                                if (deg > 237.0 && deg < 257.0) arc = smoothstep(237.0, 240.0, deg) * (1.0 - smoothstep(254.0, 257.0, deg));
                                if (deg > 258.0 && deg < 265.0) arc = max(arc, smoothstep(258.0, 260.0, deg) * (1.0 - smoothstep(263.0, 265.0, deg)));
                                if (deg > 266.0 && deg < 273.0) arc = max(arc, smoothstep(266.0, 268.0, deg) * (1.0 - smoothstep(271.0, 273.0, deg)));
                                if (deg > 274.0 && deg < 290.0) arc = max(arc, smoothstep(274.0, 277.0, deg) * (1.0 - smoothstep(287.0, 290.0, deg)));
                                if (deg > 292.0 && deg < 320.0) arc = max(arc, smoothstep(292.0, 296.0, deg) * (1.0 - smoothstep(316.0, 320.0, deg)));
                                ring_alpha *= mix(0.15, 1.0, arc);
                            }
                            t_ring = t_disc;
                        }
                    }

                    bool ring_in_front = ring_alpha > 0.01 && t_ring < t_sphere;

                    if (!ortho_hit && ring_alpha < 0.01) {
                        vec3 bg = vec3(0.0);
                        float bg_alpha = 0.0;

                        if (u_show_stars > 0.5) {
                            vec2 sp = (v_uv - 0.5) * 2.0;
                            sp.x *= u_aspect;
                            vec3 dir = u_inv_rotation * normalize(vec3(sp, -2.0));
                            float slat = asin(clamp(dir.y, -1.0, 1.0));
                            float slon = atan(-dir.z, dir.x);
                            float su = (slon + PI) / (2.0 * PI);
                            float sv = (PI / 2.0 - slat) / PI;
                            bg = texture(u_stars, vec2(su, sv)).rgb;
                            bg_alpha = 1.0;
                        }

                        if (u_atmosphere > 0.0 && screen_dist < atmo_outer) {
                            float C_atmo = O.x*O.x + O.y*O.y + O.z*O.z - 1.0;
                            float disc_atmo = B*B - 4.0*A*C_atmo;
                            if (disc_atmo >= 0.0) {
                                float atmo_depth = (screen_dist - 1.0) / (ATMO_THICKNESS * u_atmosphere);
                                atmo_depth = clamp(atmo_depth, 0.0, 1.0);
                                float atmo_falloff = 1.0 - atmo_depth;
                                atmo_falloff = pow(atmo_falloff, 2.0);
                                float glow = atmo_falloff * 0.8;
                                bg = bg * (1.0 - glow) + ATMO_COLOR * glow;
                                bg_alpha = max(bg_alpha, glow);
                            }
                        }

                        out_color = vec4(mix(u_bg_color, bg, bg_alpha), 1.0);
                        return;
                    }

                    if (!ortho_hit && ring_alpha >= 0.01) {
                        vec3 bg = vec3(0.0);
                        float bg_alpha = 0.0;
                        if (u_show_stars > 0.5) {
                            vec2 sp = (v_uv - 0.5) * 2.0;
                            sp.x *= u_aspect;
                            vec3 dir = u_inv_rotation * normalize(vec3(sp, -2.0));
                            float slat = asin(clamp(dir.y, -1.0, 1.0));
                            float slon = atan(-dir.z, dir.x);
                            float su = (slon + PI) / (2.0 * PI);
                            float sv = (PI / 2.0 - slat) / PI;
                            bg = texture(u_stars, vec2(su, sv)).rgb;
                            bg_alpha = 1.0;
                        }
                        vec3 base = mix(u_bg_color, bg, bg_alpha);
                        vec3 final_color = mix(base, ring_color, ring_alpha);
                        out_color = vec4(final_color, 1.0);
                        return;
                    }
                    lat = lat_ortho;
                    lon = lon_ortho;
                    normal = normal_ortho;
                    alpha = 1.0;

                    float tex_u = (lon + PI) / (2.0 * PI);
                    float tex_v = (PI / 2.0 - lat) / PI;

                    vec3 day_color;
                    if (u_use_detail > 0.5) {
                        float lon_deg = lon * 180.0 / PI;
                        if (lon_deg < u_detail_bounds.x) lon_deg += 360.0;
                        float du = (lon_deg - u_detail_bounds.x) / (u_detail_bounds.y - u_detail_bounds.x);
                        float dv = (u_detail_bounds.w - lat) / (u_detail_bounds.w - u_detail_bounds.z);
                        if (du >= 0.0 && du <= 1.0 && dv >= 0.0 && dv <= 1.0) {
                            day_color = texture(u_detail, vec2(du, dv)).rgb;
                        } else {
                            day_color = texture(u_texture, vec2(tex_u, tex_v)).rgb;
                        }
                    } else {
                        day_color = texture(u_texture, vec2(tex_u, tex_v)).rgb;
                    }

                    if (u_show_clouds > 0.5 && u_use_detail < 0.5) {
                        float cloud = texture(u_clouds, vec2(tex_u, tex_v)).r;
                        day_color = mix(day_color, vec3(1.0), cloud);
                    }

                    vec3 color;
                    float sun_dot = dot(normal, u_sun_dir);
                    if (u_show_day_night > 0.5) {
                        float day_factor = smoothstep(-0.1, 0.1, sun_dot);
                        float shade = 0.2 + 0.8 * max(sun_dot, 0.0);
                        vec3 lit_day = day_color * shade;
                        vec3 night_lights = texture(u_night, vec2(tex_u, tex_v)).rgb;
                        color = mix(night_lights, lit_day, day_factor);
                    } else {
                        float shade = 0.3 + 0.7 * max(dot(normal, -D), 0.0);
                        color = day_color * shade;
                    }

                    if (u_atmosphere > 0.0) {
                        float fresnel = 1.0 - max(dot(normal, -D), 0.0);
                        fresnel = pow(fresnel, 3.0);
                        float rim = fresnel * 0.6 * u_atmosphere;
                        float atmo_sun = u_show_day_night > 0.5 ? max(sun_dot + 0.3, 0.0) : 1.0;
                        color = mix(color, ATMO_COLOR * atmo_sun, rim);
                    }

                    if (ring_in_front) {
                        color = mix(color, ring_color, ring_alpha);
                    }

                    out_color = vec4(mix(u_bg_color, color, alpha), 1.0);
                }
            "#;

            let shader_sources = [
                (glow::VERTEX_SHADER, vertex_shader_source),
                (glow::FRAGMENT_SHADER, fragment_shader_source),
            ];

            let shaders: Vec<_> = shader_sources
                .iter()
                .map(|(shader_type, shader_source)| {
                    let shader = gl.create_shader(*shader_type).expect("Cannot create shader");
                    gl.shader_source(shader, &format!("{shader_version}\n{shader_source}"));
                    gl.compile_shader(shader);
                    assert!(
                        gl.get_shader_compile_status(shader),
                        "Failed to compile shader: {}",
                        gl.get_shader_info_log(shader)
                    );
                    gl.attach_shader(program, shader);
                    shader
                })
                .collect();

            gl.link_program(program);
            assert!(
                gl.get_program_link_status(program),
                "Failed to link program: {}",
                gl.get_program_info_log(program)
            );

            for shader in shaders {
                gl.detach_shader(program, shader);
                gl.delete_shader(shader);
            }

            let vertex_array = gl.create_vertex_array().expect("Cannot create vertex array");

            Self {
                program,
                vertex_array,
                textures: HashMap::new(),
                cloud_textures: HashMap::new(),
                night_texture: None,
                star_texture: None,
                milky_way_texture: None,
                ring_textures: HashMap::new(),
            }
        }
    }

    fn upload_night_texture(&mut self, gl: &glow::Context, night_tex: &EarthTexture) {
        unsafe {
            if self.night_texture.is_some() {
                return;
            }

            let texture = gl.create_texture().expect("Cannot create texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));

            let pixels: Vec<u8> = night_tex.pixels.iter()
                .flat_map(|&[r, g, b]| [r, g, b])
                .collect();

            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGB as i32,
                night_tex.width as i32,
                night_tex.height as i32,
                0,
                glow::RGB,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&pixels)),
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            self.night_texture = Some(texture);
        }
    }

    fn upload_star_texture(&mut self, gl: &glow::Context, tex: &EarthTexture) {
        unsafe {
            if self.star_texture.is_some() {
                return;
            }
            let texture = gl.create_texture().expect("Cannot create texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            let pixels: Vec<u8> = tex.pixels.iter().flat_map(|&[r, g, b]| [r, g, b]).collect();
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::RGB as i32,
                tex.width as i32, tex.height as i32, 0,
                glow::RGB, glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&pixels)),
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            self.star_texture = Some(texture);
        }
    }

    fn upload_milky_way_texture(&mut self, gl: &glow::Context, tex: &EarthTexture) {
        unsafe {
            if self.milky_way_texture.is_some() {
                return;
            }
            let texture = gl.create_texture().expect("Cannot create texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            let pixels: Vec<u8> = tex.pixels.iter().flat_map(|&[r, g, b]| [r, g, b]).collect();
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::RGB as i32,
                tex.width as i32, tex.height as i32, 0,
                glow::RGB, glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&pixels)),
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            self.milky_way_texture = Some(texture);
        }
    }

    fn upload_ring_texture(&mut self, gl: &glow::Context, body: CelestialBody, tex: &RingTexture) {
        unsafe {
            if self.ring_textures.contains_key(&body) {
                return;
            }
            let texture = gl.create_texture().expect("Cannot create texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            let pixels: Vec<u8> = tex.pixels.iter().flat_map(|&[r, g, b, a]| [r, g, b, a]).collect();
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::RGBA as i32,
                tex.width as i32, tex.height as i32, 0,
                glow::RGBA, glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&pixels)),
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            self.ring_textures.insert(body, texture);
        }
    }

    fn upload_texture(&mut self, gl: &glow::Context, key: (CelestialBody, Skin, TextureResolution), earth_tex: &EarthTexture) {
        unsafe {
            if self.textures.contains_key(&key) {
                return;
            }

            let texture = gl.create_texture().expect("Cannot create texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));

            let pixels: Vec<u8> = earth_tex.pixels.iter()
                .flat_map(|&[r, g, b]| [r, g, b])
                .collect();

            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGB as i32,
                earth_tex.width as i32,
                earth_tex.height as i32,
                0,
                glow::RGB,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&pixels)),
            );

            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);

            self.textures.insert(key, texture);
        }
    }

    fn evict_unused_textures(&mut self, gl: &glow::Context, keep: &[(CelestialBody, Skin, TextureResolution)]) {
        let to_remove: Vec<_> = self.textures.keys()
            .filter(|k| !keep.contains(k))
            .copied()
            .collect();
        for key in to_remove {
            if let Some(tex) = self.textures.remove(&key) {
                unsafe { gl.delete_texture(tex); }
            }
        }
        let keep_bodies: std::collections::HashSet<CelestialBody> = keep.iter().map(|k| k.0).collect();
        let rings_to_remove: Vec<_> = self.ring_textures.keys()
            .filter(|b| !keep_bodies.contains(b))
            .copied()
            .collect();
        for body in rings_to_remove {
            if let Some(tex) = self.ring_textures.remove(&body) {
                unsafe { gl.delete_texture(tex); }
            }
        }
    }

    fn upload_cloud_texture(&mut self, gl: &glow::Context, res: TextureResolution, cloud_tex: &EarthTexture) {
        unsafe {
            if self.cloud_textures.contains_key(&res) {
                return;
            }

            let texture = gl.create_texture().expect("Cannot create cloud texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));

            let pixels: Vec<u8> = cloud_tex.pixels.iter()
                .flat_map(|&[r, g, b]| [r, g, b])
                .collect();

            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGB as i32,
                cloud_tex.width as i32,
                cloud_tex.height as i32,
                0,
                glow::RGB,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&pixels)),
            );

            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);

            self.cloud_textures.insert(res, texture);
        }
    }

    fn paint(
        &self,
        gl: &glow::Context,
        key: (CelestialBody, Skin, TextureResolution),
        inv_rotation: &Matrix3<f64>,
        flattening: f64,
        aspect: f32,
        scale: f32,
        atmosphere: f32,
        show_clouds: bool,
        show_day_night: bool,
        sun_dir: [f32; 3],
        detail_texture: Option<&DetailTexture>,
        show_stars: bool,
        show_milky_way: bool,
        bg_color: [f32; 3],
    ) {
        let Some(texture) = self.textures.get(&key) else { return };

        unsafe {
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vertex_array));

            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
            gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_texture").as_ref(), 0);

            gl.active_texture(glow::TEXTURE1);
            let cloud_tex = self.cloud_textures.get(&key.2);
            if let Some(ct) = cloud_tex {
                gl.bind_texture(glow::TEXTURE_2D, Some(*ct));
            } else {
                gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
            }
            gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_clouds").as_ref(), 1);

            gl.active_texture(glow::TEXTURE2);
            if let Some(nt) = self.night_texture {
                gl.bind_texture(glow::TEXTURE_2D, Some(nt));
            } else {
                gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
            }
            gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_night").as_ref(), 2);

            gl.active_texture(glow::TEXTURE3);
            let use_detail = if let Some(dt) = detail_texture {
                if let Some(gl_tex) = dt.gl_texture {
                    gl.bind_texture(glow::TEXTURE_2D, Some(gl_tex));
                    gl.uniform_4_f32(
                        gl.get_uniform_location(self.program, "u_detail_bounds").as_ref(),
                        dt.bounds.min_lon as f32,
                        dt.bounds.max_lon as f32,
                        dt.bounds.min_lat as f32,
                        dt.bounds.max_lat as f32,
                    );
                    true
                } else {
                    gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
                    false
                }
            } else {
                gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
                false
            };
            gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_detail").as_ref(), 3);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_use_detail").as_ref(), if use_detail { 1.0 } else { 0.0 });

            gl.active_texture(glow::TEXTURE4);
            let star_tex = if show_milky_way { self.milky_way_texture } else { self.star_texture };
            if let Some(st) = star_tex {
                gl.bind_texture(glow::TEXTURE_2D, Some(st));
            } else {
                gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
            }
            gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_stars").as_ref(), 4);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_show_stars").as_ref(), if show_stars && star_tex.is_some() { 1.0 } else { 0.0 });

            gl.active_texture(glow::TEXTURE5);
            let ring_params = key.0.ring_params();
            let has_rings = if let Some(rt) = self.ring_textures.get(&key.0) {
                gl.bind_texture(glow::TEXTURE_2D, Some(*rt));
                true
            } else {
                gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
                false
            };
            gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_ring_texture").as_ref(), 5);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_has_rings").as_ref(), if has_rings { 1.0 } else { 0.0 });
            let (ring_inner, ring_outer) = ring_params.map(|(_, i, o)| (i, o)).unwrap_or((0.0, 0.0));
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_ring_inner").as_ref(), ring_inner);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_ring_outer").as_ref(), ring_outer);
            let adams_arc = if has_rings && key.0 == CelestialBody::Neptune { 1.0f32 } else { 0.0f32 };
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_adams_arc").as_ref(), adams_arc);
            let epsilon_wobble = if has_rings && key.0 == CelestialBody::Uranus { 1.0f32 } else { 0.0f32 };
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_epsilon_wobble").as_ref(), epsilon_wobble);

            let rot_data: [f32; 9] = [
                inv_rotation[(0, 0)] as f32, inv_rotation[(1, 0)] as f32, inv_rotation[(2, 0)] as f32,
                inv_rotation[(0, 1)] as f32, inv_rotation[(1, 1)] as f32, inv_rotation[(2, 1)] as f32,
                inv_rotation[(0, 2)] as f32, inv_rotation[(1, 2)] as f32, inv_rotation[(2, 2)] as f32,
            ];
            gl.uniform_matrix_3_f32_slice(
                gl.get_uniform_location(self.program, "u_inv_rotation").as_ref(),
                false,
                &rot_data,
            );

            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_flattening").as_ref(), flattening as f32);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_aspect").as_ref(), aspect);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_scale").as_ref(), scale);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_atmosphere").as_ref(), atmosphere);
            let clouds_enabled = show_clouds && cloud_tex.is_some() && key.0 == CelestialBody::Earth;
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_show_clouds").as_ref(), if clouds_enabled { 1.0 } else { 0.0 });

            let day_night_enabled = show_day_night && self.night_texture.is_some() && key.0 == CelestialBody::Earth;
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_show_day_night").as_ref(), if day_night_enabled { 1.0 } else { 0.0 });
            gl.uniform_3_f32(gl.get_uniform_location(self.program, "u_sun_dir").as_ref(), sun_dir[0], sun_dir[1], sun_dir[2]);
            gl.uniform_3_f32(gl.get_uniform_location(self.program, "u_bg_color").as_ref(), bg_color[0], bg_color[1], bg_color[2]);

            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        }
    }

    fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_program(self.program);
            gl.delete_vertex_array(self.vertex_array);
            for texture in self.textures.values() {
                gl.delete_texture(*texture);
            }
            for texture in self.cloud_textures.values() {
                gl.delete_texture(*texture);
            }
            if let Some(nt) = self.night_texture {
                gl.delete_texture(nt);
            }
            if let Some(st) = self.star_texture {
                gl.delete_texture(st);
            }
            if let Some(mw) = self.milky_way_texture {
                gl.delete_texture(mw);
            }
            for rt in self.ring_textures.values() {
                gl.delete_texture(*rt);
            }
        }
    }
}

fn rotate_point_matrix(x: f64, y: f64, z: f64, rot: &Matrix3<f64>) -> (f64, f64, f64) {
    let v = rot * Vector3::new(x, y, z);
    (v.x, v.y, v.z)
}

fn matrix_to_lat_lon(m: &Matrix3<f64>) -> (f64, f64) {
    let lat = m[(2, 1)].asin().clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
    let mut lon = (-m[(0, 2)]).atan2(m[(0, 0)]) - std::f64::consts::FRAC_PI_2;
    if lon < -std::f64::consts::PI { lon += 2.0 * std::f64::consts::PI; }
    if lon > std::f64::consts::PI { lon -= 2.0 * std::f64::consts::PI; }
    (lat, lon)
}

fn lat_lon_to_matrix(lat: f64, lon: f64) -> Matrix3<f64> {
    let lon = -lon - std::f64::consts::FRAC_PI_2;
    let (sl, cl) = (lat.sin(), lat.cos());
    let (sn, cn) = (lon.sin(), lon.cos());
    Matrix3::new(
        cn, 0.0, sn,
        sl * sn, cl, -sl * cn,
        -cl * sn, sl, cl * cn,
    )
}

fn rotation_from_drag(dx: f64, dy: f64) -> Matrix3<f64> {
    let rot_y = Matrix3::new(
        dx.cos(), 0.0, dx.sin(),
        0.0, 1.0, 0.0,
        -dx.sin(), 0.0, dx.cos(),
    );
    let rot_x = Matrix3::new(
        1.0, 0.0, 0.0,
        0.0, dy.cos(), -dy.sin(),
        0.0, dy.sin(), dy.cos(),
    );
    rot_x * rot_y
}

#[derive(Clone, Copy, PartialEq)]
enum Preset {
    None,
    Starlink,
    OneWeb,
    Iridium,
    Kuiper,
    Iris2,
    Telesat,
}

fn greenwich_mean_sidereal_time(timestamp: DateTime<Utc>) -> f64 {
    let j2000 = DateTime::parse_from_rfc3339("2000-01-01T12:00:00Z")
        .unwrap()
        .with_timezone(&Utc);
    let days_since_j2000 = (timestamp - j2000).num_milliseconds() as f64 / (1000.0 * SECONDS_PER_DAY);
    let centuries = days_since_j2000 / DAYS_PER_JULIAN_CENTURY;
    let gmst_degrees = GMST_BASE_DEG
        + GMST_ROTATION_PER_DAY * days_since_j2000
        + GMST_CORRECTION * centuries * centuries
        - centuries * centuries * centuries / 38710000.0;
    let gmst_normalized = gmst_degrees.rem_euclid(360.0);
    gmst_normalized.to_radians()
}

fn body_rotation_angle(body: CelestialBody, sim_time_seconds: f64, gmst: f64) -> f64 {
    if body == CelestialBody::Earth {
        gmst
    } else {
        let period_hours = body.rotation_period_hours();
        let period_seconds = period_hours * 3600.0;
        let rotations = sim_time_seconds / period_seconds;
        (rotations * 2.0 * PI).rem_euclid(2.0 * PI)
    }
}

#[derive(Clone)]
struct ConstellationConfig {
    sats_per_plane: usize,
    num_planes: usize,
    altitude_km: f64,
    inclination: f64,
    walker_type: WalkerType,
    phasing: f64,
    raan_offset: f64,
    raan_spacing: Option<f64>,
    eccentricity: f64,
    arg_periapsis: f64,
    drag_enabled: bool,
    ballistic_coeff: f64,
    preset: Preset,
    color_offset: usize,
    hidden: bool,
}

impl ConstellationConfig {
    fn new(color_offset: usize) -> Self {
        Self {
            sats_per_plane: 30,
            num_planes: 30,
            altitude_km: 200.0,
            inclination: 90.0,
            walker_type: WalkerType::Delta,
            phasing: 0.0,
            raan_offset: 0.0,
            raan_spacing: None,
            eccentricity: 0.0,
            arg_periapsis: 0.0,
            drag_enabled: false,
            ballistic_coeff: 100.0,
            preset: Preset::None,
            color_offset,
            hidden: false,
        }
    }

    fn total_sats(&self) -> usize {
        self.sats_per_plane * self.num_planes
    }

    fn constellation(&self, planet_radius: f64, planet_mu: f64, planet_j2: f64, planet_equatorial_radius: f64) -> WalkerConstellation {
        WalkerConstellation {
            walker_type: self.walker_type,
            total_sats: self.sats_per_plane * self.num_planes,
            num_planes: self.num_planes,
            altitude_km: self.altitude_km,
            inclination_deg: self.inclination,
            phasing: self.phasing,
            raan_offset_deg: self.raan_offset,
            raan_spacing_deg: self.raan_spacing,
            eccentricity: self.eccentricity,
            arg_periapsis_deg: self.arg_periapsis,
            planet_radius,
            planet_mu,
            planet_j2,
            planet_equatorial_radius,
        }
    }

    fn preset_name(&self) -> &'static str {
        match self.preset {
            Preset::None => "Custom",
            Preset::Starlink => "Starlink",
            Preset::OneWeb => "OneWeb",
            Preset::Iridium => "Iridium",
            Preset::Kuiper => "Kuiper",
            Preset::Iris2 => "Iris²",
            Preset::Telesat => "Telesat",
        }
    }
}

#[derive(Clone)]
struct GroundStation {
    name: String,
    lat: f64,
    lon: f64,
    radius_km: f64,
    color: egui::Color32,
}

#[derive(Clone)]
struct AreaOfInterest {
    name: String,
    lat: f64,
    lon: f64,
    radius_km: f64,
    color: egui::Color32,
    ground_station_idx: Option<usize>,
}

#[derive(Clone)]
struct DeviceLayer {
    name: String,
    color: egui::Color32,
    devices: Vec<(f64, f64)>,
}

#[derive(Clone)]
struct PlanetConfig {
    name: String,
    constellations: Vec<ConstellationConfig>,
    constellation_counter: usize,
    celestial_body: CelestialBody,
    skin: Skin,
    satellite_cameras: Vec<SatelliteCamera>,
    pending_cameras: Vec<SatelliteCamera>,
    cameras_to_remove: Vec<usize>,
    show_tle_window: bool,
    show_gs_aoi_window: bool,
    show_config_window: bool,
    tle_selections: HashMap<TlePreset, (bool, TleLoadState, Option<Vec<TleShell>>)>,
    ground_stations: Vec<GroundStation>,
    areas_of_interest: Vec<AreaOfInterest>,
    device_layers: Vec<DeviceLayer>,
}

impl PlanetConfig {
    fn new(name: String) -> Self {
        let mut tle_selections = HashMap::new();
        for preset in TlePreset::ALL {
            tle_selections.insert(preset, (false, TleLoadState::NotLoaded, None));
        }
        Self {
            name,
            constellations: Vec::new(),
            constellation_counter: 0,
            celestial_body: CelestialBody::Earth,
            skin: Skin::Default,
            satellite_cameras: Vec::new(),
            pending_cameras: Vec::new(),
            cameras_to_remove: Vec::new(),
            show_tle_window: false,
            show_gs_aoi_window: false,
            show_config_window: true,
            tle_selections,
            ground_stations: Vec::new(),
            areas_of_interest: Vec::new(),
            device_layers: vec![DeviceLayer {
                name: "CandyTron".to_string(),
                color: egui::Color32::from_rgb(80, 140, 255),
                devices: vec![(59.40481807006525, 17.949657783197082), (59.41, 17.96)],
            }],
        }
    }

    fn add_constellation(&mut self) {
        self.constellations.push(ConstellationConfig::new(self.constellation_counter));
        self.constellation_counter += 1;
    }

}

#[derive(Clone, Copy)]
struct View3DFlags {
    show_orbits: bool,
    show_axes: bool,
    show_coverage: bool,
    show_links: bool,
    show_intra_links: bool,
    hide_behind_earth: bool,
    single_color: bool,
    dark_mode: bool,
    show_routing_paths: bool,
    show_manhattan_path: bool,
    show_shortest_path: bool,
    show_asc_desc_colors: bool,
    show_altitude_lines: bool,
    render_planet: bool,
    fixed_sizes: bool,
    show_polar_circle: bool,
    show_equator: bool,
    show_terminator: bool,
    earth_fixed_camera: bool,
    use_gpu_rendering: bool,
    show_clouds: bool,
    show_day_night: bool,
    show_stars: bool,
    show_milky_way: bool,
    show_borders: bool,
    show_cities: bool,
}

#[derive(Clone)]
struct TabSettings {
    time: f64,
    speed: f64,
    animate: bool,
    zoom: f64,
    rotation: Matrix3<f64>,
    earth_fixed_camera: bool,
    follow_satellite: bool,
    show_camera_windows: bool,
    show_orbits: bool,
    show_links: bool,
    show_intra_links: bool,
    show_coverage: bool,
    coverage_angle: f64,
    show_routing_paths: bool,
    show_manhattan_path: bool,
    show_shortest_path: bool,
    show_asc_desc_colors: bool,
    single_color: bool,
    show_torus: bool,
    show_ground_track: bool,
    show_axes: bool,
    hide_behind_earth: bool,
    render_planet: bool,
    show_altitude_lines: bool,
    show_devices: bool,
    show_polar_circle: bool,
    show_equator: bool,
    show_borders: bool,
    show_cities: bool,
    show_day_night: bool,
    show_terminator: bool,
    show_clouds: bool,
    show_stars: bool,
    show_milky_way: bool,
    sat_radius: f32,
    link_width: f32,
    fixed_sizes: bool,
}

impl Default for TabSettings {
    fn default() -> Self {
        Self {
            time: 0.0,
            speed: 50.0,
            animate: true,
            zoom: 1.0,
            rotation: lat_lon_to_matrix(0.0, 0.0),
            earth_fixed_camera: false,
            follow_satellite: false,
            show_camera_windows: false,
            show_orbits: true,
            show_links: true,
            show_intra_links: false,
            show_coverage: false,
            coverage_angle: 25.0,
            show_routing_paths: false,
            show_manhattan_path: true,
            show_shortest_path: true,
            show_asc_desc_colors: false,
            single_color: false,
            show_torus: false,
            show_ground_track: false,
            show_axes: false,
            hide_behind_earth: true,
            render_planet: true,
            show_altitude_lines: false,
            show_devices: false,
            show_polar_circle: false,
            show_equator: false,
            show_borders: false,
            show_cities: false,
            show_day_night: false,
            show_terminator: false,
            show_clouds: false,
            show_stars: false,
            show_milky_way: false,
            sat_radius: 1.5,
            link_width: 0.25,
            fixed_sizes: false,
        }
    }
}

struct TabConfig {
    name: String,
    planets: Vec<PlanetConfig>,
    planet_counter: usize,
    show_stats: bool,
    use_local_settings: bool,
    local_settings: TabSettings,
}

impl TabConfig {
    fn new(name: String) -> Self {
        let mut tab = Self::new_empty(name);
        tab.add_planet();
        tab
    }

    fn new_empty(name: String) -> Self {
        Self {
            name,
            planets: Vec::new(),
            planet_counter: 0,
            show_stats: false,
            use_local_settings: false,
            local_settings: TabSettings::default(),
        }
    }

    fn add_planet(&mut self) {
        self.planet_counter += 1;
        let planet = PlanetConfig::new(format!("Planet {}", self.planet_counter));
        self.planets.push(planet);
    }
}

#[derive(Clone)]
struct SatelliteCamera {
    id: usize,
    label: String,
    constellation_idx: usize,
    plane: usize,
    sat_index: usize,
    screen_pos: Option<egui::Pos2>,
}

struct ViewerState {
    tabs: Vec<TabConfig>,
    camera_id_counter: usize,
    tab_counter: usize,
    time: f64,
    speed: f64,
    animate: bool,
    show_orbits: bool,
    show_links: bool,
    show_intra_links: bool,
    show_ground_track: bool,
    show_torus: bool,
    show_axes: bool,
    show_coverage: bool,
    coverage_angle: f64,
    hide_behind_earth: bool,
    zoom: f64,
    torus_zoom: f64,
    vertical_split: f32,
    sat_radius: f32,
    rotation: Matrix3<f64>,
    torus_rotation: Matrix3<f64>,
    planet_textures: HashMap<(CelestialBody, Skin, TextureResolution), Arc<EarthTexture>>,
    ring_textures: HashMap<CelestialBody, Arc<RingTexture>>,
    cloud_textures: HashMap<TextureResolution, Arc<EarthTexture>>,
    planet_image_handles: HashMap<(CelestialBody, Skin, TextureResolution), egui::TextureHandle>,
    texture_resolution: TextureResolution,
    last_rotation: Option<Matrix3<f64>>,
    earth_resolution: usize,
    last_resolution: usize,
    texture_load_state: TextureLoadState,
    pending_body: Option<(CelestialBody, Skin, TextureResolution)>,
    dark_mode: bool,
    show_info: bool,
    follow_satellite: bool,
    show_routing_paths: bool,
    show_manhattan_path: bool,
    show_shortest_path: bool,
    show_asc_desc_colors: bool,
    single_color: bool,
    show_altitude_lines: bool,
    show_camera_windows: bool,
    render_planet: bool,
    show_polar_circle: bool,
    show_equator: bool,
    real_time: f64,
    start_timestamp: DateTime<Utc>,
    show_side_panel: bool,
    pending_add_tab: Option<usize>,
    link_width: f32,
    fixed_sizes: bool,
    earth_fixed_camera: bool,
    current_gmst: f64,
    auto_cycle_tabs: bool,
    cycle_interval: f64,
    last_cycle_time: f64,
    use_gpu_rendering: bool,
    show_clouds: bool,
    show_devices: bool,
    show_day_night: bool,
    show_terminator: bool,
    show_stars: bool,
    show_milky_way: bool,
    show_borders: bool,
    show_cities: bool,
    active_tab_idx: usize,
    #[cfg(not(target_arch = "wasm32"))]
    geo_data: GeoLoadState,
    #[cfg(not(target_arch = "wasm32"))]
    geo_fetch_rx: Option<mpsc::Receiver<Result<GeoOverlayData, String>>>,
    dragging_place: Option<(usize, usize, bool, usize)>,
    night_texture: Option<Arc<EarthTexture>>,
    star_texture: Option<Arc<EarthTexture>>,
    milky_way_texture: Option<Arc<EarthTexture>>,
    #[allow(dead_code)]
    night_texture_loading: bool,
    #[allow(dead_code)]
    star_texture_loading: bool,
    #[allow(dead_code)]
    milky_way_texture_loading: bool,
    #[allow(dead_code)]
    cloud_texture_loading: bool,
    sphere_renderer: Option<Arc<Mutex<SphereRenderer>>>,
    #[cfg(not(target_arch = "wasm32"))]
    tle_fetch_tx: mpsc::Sender<(TlePreset, Result<Vec<TleSatellite>, String>)>,
    #[cfg(not(target_arch = "wasm32"))]
    tle_fetch_rx: mpsc::Receiver<(TlePreset, Result<Vec<TleSatellite>, String>)>,
    #[cfg(not(target_arch = "wasm32"))]
    tile_overlay: TileOverlayState,
    view_width: f32,
    view_height: f32,
}

struct App {
    dock_state: DockState<usize>,
    viewer: ViewerState,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let gl = cc.gl.as_ref().expect("glow backend required");
        let sphere_renderer = Arc::new(Mutex::new(SphereRenderer::new(gl)));

        let torus_initial = Matrix3::new(
            1.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
            0.0, 1.0, 0.0,
        );
        let builtin_texture = Arc::new(EarthTexture::load());
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

        Self {
            dock_state: DockState::new(vec![0]),
            viewer: ViewerState {
                tabs: vec![TabConfig::new("View 1".to_string())],
                camera_id_counter: 0,
                tab_counter: 1,
                time: 0.0,
                speed: 50.0,
                animate: true,
                show_orbits: true,
                show_links: true,
                show_intra_links: false,
                show_ground_track: false,
                show_torus: false,
                show_axes: false,
                show_coverage: false,
                coverage_angle: 25.0,
                hide_behind_earth: true,
                zoom: 1.0,
                torus_zoom: 1.0,
                vertical_split: 0.6,
                sat_radius: 1.5,
                rotation: lat_lon_to_matrix(0.0, 0.0),
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
                follow_satellite: false,
                show_routing_paths: false,
                show_manhattan_path: true,
                show_shortest_path: true,
                show_asc_desc_colors: false,
                single_color: false,
                show_altitude_lines: false,
                show_camera_windows: false,
                render_planet: true,
                show_polar_circle: false,
                show_equator: false,
                real_time: 0.0,
                start_timestamp: Utc::now(),
                show_side_panel: true,
                pending_add_tab: None,
                link_width: 0.25,
                fixed_sizes: false,
                earth_fixed_camera: false,
                current_gmst: 0.0,
                auto_cycle_tabs: false,
                cycle_interval: 5.0,
                last_cycle_time: 0.0,
                use_gpu_rendering: true,
                show_clouds: false,
                show_devices: false,
                show_day_night: false,
                show_terminator: false,
                show_stars: false,
                show_milky_way: false,
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
            },
        }
    }
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

    fn context_menu(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab, _surface: SurfaceIndex, _node: NodeIndex) {
        if *tab < self.tabs.len() {
            ui.checkbox(&mut self.tabs[*tab].use_local_settings, "Override global settings");
        }
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
                    for body in CelestialBody::ALL {
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

        if remove_planet {
            return (add_planet, remove_planet);
        }

        {
            let planet = &mut self.tabs[tab_idx].planets[planet_idx];
            if new_body != planet.celestial_body {
                self.zoom = 1.0;
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

        let hide_sats = self.zoom > 100.0;
        let mut constellations_data: Vec<_> = if hide_sats {
            Vec::new()
        } else {
            planet.constellations.iter()
                .enumerate()
                .filter(|(_, c)| !c.hidden)
                .map(|(orig_idx, c)| {
                    let wc = c.constellation(planet_radius, planet_mu, planet_j2, planet_eq_radius);
                    let pos = wc.satellite_positions(self.time);
                    let name = c.preset_name().to_string();
                    (wc, pos, c.color_offset, 0u8, orig_idx, name)
                })
                .collect()
        };

        if planet.show_tle_window {
            let propagation_minutes = self.start_timestamp.timestamp() as f64 / 60.0 + self.time / 60.0;
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
        let use_local = self.tabs[tab_idx].use_local_settings;
        let local = &self.tabs[tab_idx].local_settings;
        let show_torus = if use_local { local.show_torus } else { self.show_torus };
        let show_ground_track = if use_local { local.show_ground_track } else { self.show_ground_track };
        let use_horizontal = show_torus && !show_ground_track;

        if use_horizontal {
            let half_width = available.x / 2.0;
            let view_height = available.y - 20.0;
            let view_size = half_width.min(view_height);

            let show_orbits = if use_local { local.show_orbits } else { self.show_orbits };
            let show_axes = if use_local { local.show_axes } else { self.show_axes };
            let show_coverage = if use_local { local.show_coverage } else { self.show_coverage };
            let coverage_angle = if use_local { local.coverage_angle } else { self.coverage_angle };
            let time = if use_local { local.time } else { self.time };
            let rotation = if use_local { local.rotation } else { self.rotation };
            let zoom = if use_local { local.zoom } else { self.zoom };
            let earth_fixed_camera = if use_local { local.earth_fixed_camera } else { self.earth_fixed_camera };
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
            let sat_radius = if use_local { local.sat_radius } else { self.sat_radius };
            let show_links = if use_local { local.show_links } else { self.show_links };
            let show_intra_links = if use_local { local.show_intra_links } else { self.show_intra_links };
            let render_planet = if use_local { local.render_planet } else { self.render_planet };
            let hide_behind_earth = render_planet && (if use_local { local.hide_behind_earth } else { self.hide_behind_earth });
            let single_color = (if use_local { local.single_color } else { self.single_color }) || constellations_data.len() > 1;
            let dark_mode = self.dark_mode;
            let show_routing_paths = if use_local { local.show_routing_paths } else { self.show_routing_paths };
            let show_manhattan_path = if use_local { local.show_manhattan_path } else { self.show_manhattan_path };
            let show_shortest_path = if use_local { local.show_shortest_path } else { self.show_shortest_path };
            let show_asc_desc_colors = if use_local { local.show_asc_desc_colors } else { self.show_asc_desc_colors };
            let show_altitude_lines = if use_local { local.show_altitude_lines } else { self.show_altitude_lines };
            let tex_res = self.texture_resolution;
            let planet_handle = self.planet_image_handles.get(&(celestial_body, skin, tex_res));
            let torus_rotation = self.torus_rotation;
            let torus_zoom = self.torus_zoom;
            let link_width = if use_local { local.link_width } else { self.link_width };
            let fixed_sizes = if use_local { local.fixed_sizes } else { self.fixed_sizes };
            let flattening = celestial_body.flattening();
            let show_polar_circle = if use_local { local.show_polar_circle } else { self.show_polar_circle };
            let show_equator = if use_local { local.show_equator } else { self.show_equator };
            let show_day_night = if use_local { local.show_day_night } else { self.show_day_night };
            let show_terminator = (if use_local { local.show_terminator } else { self.show_terminator }) && show_day_night;
            let show_clouds = if use_local { local.show_clouds } else { self.show_clouds };
            let show_stars = if use_local { local.show_stars } else { self.show_stars };
            let show_milky_way = (if use_local { local.show_milky_way } else { self.show_milky_way }) && show_stars;
            let show_devices = if use_local { local.show_devices } else { self.show_devices };
            let show_borders = if use_local { local.show_borders } else { self.show_borders };
            let show_cities = if use_local { local.show_cities } else { self.show_cities };
            let detail_gl_info = self.tile_overlay_detail_gl_info(celestial_body);
            self.view_width = half_width;
            self.view_height = view_size;

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    let planet = &mut self.tabs[tab_idx].planets[planet_idx];
                    let view_flags = View3DFlags {
                        show_orbits, show_axes, show_coverage, show_links, show_intra_links,
                        hide_behind_earth, single_color, dark_mode, show_routing_paths,
                        show_manhattan_path, show_shortest_path, show_asc_desc_colors,
                        show_altitude_lines, render_planet, fixed_sizes, show_polar_circle,
                        show_equator, show_terminator, earth_fixed_camera,
                        use_gpu_rendering: self.use_gpu_rendering, show_clouds, show_day_night,
                        show_stars, show_milky_way, show_borders, show_cities,
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
                        half_width,
                        view_size,
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
                    if use_local {
                        self.tabs[tab_idx].local_settings.rotation = rot;
                        self.tabs[tab_idx].local_settings.zoom = new_zoom;
                    } else {
                        self.rotation = rot;
                        self.zoom = new_zoom;
                    }
                });

                ui.add_space(5.0);

                ui.vertical(|ui| {
                    let planet = &mut self.tabs[tab_idx].planets[planet_idx];
                    let (trot, tzoom) = draw_torus(
                        ui,
                        &format!("torus_{}", view_name),
                        &constellations_data,
                        time,
                        torus_rotation,
                        half_width,
                        view_size,
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
            });
        } else {
            let viz_width = available.x;
            let available_for_views = available.y - 20.0;

            let has_secondary = show_torus || show_ground_track;
            let separator_height = if has_secondary { 8.0 } else { 0.0 };

            let earth_height = if has_secondary {
                (available_for_views - separator_height) * self.vertical_split
            } else {
                available_for_views
            }.min(viz_width);

            let secondary_height = if has_secondary {
                (available_for_views - separator_height) * (1.0 - self.vertical_split)
            } else {
                0.0
            };

            let show_orbits = if use_local { local.show_orbits } else { self.show_orbits };
            let show_axes = if use_local { local.show_axes } else { self.show_axes };
            let show_coverage = if use_local { local.show_coverage } else { self.show_coverage };
            let coverage_angle = if use_local { local.coverage_angle } else { self.coverage_angle };
            let time = if use_local { local.time } else { self.time };
            let rotation = if use_local { local.rotation } else { self.rotation };
            let zoom = if use_local { local.zoom } else { self.zoom };
            let earth_fixed_camera = if use_local { local.earth_fixed_camera } else { self.earth_fixed_camera };
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
            let sat_radius = if use_local { local.sat_radius } else { self.sat_radius };
            let show_links = if use_local { local.show_links } else { self.show_links };
            let show_intra_links = if use_local { local.show_intra_links } else { self.show_intra_links };
            let render_planet = if use_local { local.render_planet } else { self.render_planet };
            let hide_behind_earth = render_planet && (if use_local { local.hide_behind_earth } else { self.hide_behind_earth });
            let single_color = (if use_local { local.single_color } else { self.single_color }) || constellations_data.len() > 1;
            let dark_mode = self.dark_mode;
            let show_routing_paths = if use_local { local.show_routing_paths } else { self.show_routing_paths };
            let show_manhattan_path = if use_local { local.show_manhattan_path } else { self.show_manhattan_path };
            let show_shortest_path = if use_local { local.show_shortest_path } else { self.show_shortest_path };
            let show_asc_desc_colors = if use_local { local.show_asc_desc_colors } else { self.show_asc_desc_colors };
            let show_altitude_lines = if use_local { local.show_altitude_lines } else { self.show_altitude_lines };
            let tex_res = self.texture_resolution;
            let planet_handle = self.planet_image_handles.get(&(celestial_body, skin, tex_res));
            let link_width = if use_local { local.link_width } else { self.link_width };
            let fixed_sizes = if use_local { local.fixed_sizes } else { self.fixed_sizes };
            let flattening = celestial_body.flattening();
            let show_polar_circle = if use_local { local.show_polar_circle } else { self.show_polar_circle };
            let show_equator = if use_local { local.show_equator } else { self.show_equator };
            let show_day_night = if use_local { local.show_day_night } else { self.show_day_night };
            let show_terminator = (if use_local { local.show_terminator } else { self.show_terminator }) && show_day_night;
            let show_clouds = if use_local { local.show_clouds } else { self.show_clouds };
            let show_stars = if use_local { local.show_stars } else { self.show_stars };
            let show_milky_way = (if use_local { local.show_milky_way } else { self.show_milky_way }) && show_stars;
            let show_devices = if use_local { local.show_devices } else { self.show_devices };
            let show_borders = if use_local { local.show_borders } else { self.show_borders };
            let show_cities = if use_local { local.show_cities } else { self.show_cities };
            let detail_gl_info = self.tile_overlay_detail_gl_info(celestial_body);
            self.view_width = viz_width;
            self.view_height = earth_height;

            let planet = &mut self.tabs[tab_idx].planets[planet_idx];
            let view_flags = View3DFlags {
                show_orbits, show_axes, show_coverage, show_links, show_intra_links,
                hide_behind_earth, single_color, dark_mode, show_routing_paths,
                show_manhattan_path, show_shortest_path, show_asc_desc_colors,
                show_altitude_lines, render_planet, fixed_sizes, show_polar_circle,
                show_equator, show_terminator, earth_fixed_camera,
                use_gpu_rendering: self.use_gpu_rendering, show_clouds, show_day_night,
                show_stars, show_milky_way, show_borders, show_cities,
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
                viz_width,
                earth_height,
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
            if use_local {
                self.tabs[tab_idx].local_settings.rotation = rot;
                self.tabs[tab_idx].local_settings.zoom = new_zoom;
            } else {
                self.rotation = rot;
                self.zoom = new_zoom;
            }

            if has_secondary {
                let separator_rect = ui.available_rect_before_wrap();
                let separator_rect = egui::Rect::from_min_size(
                    separator_rect.min,
                    egui::vec2(viz_width, separator_height),
                );
                let response = ui.allocate_rect(separator_rect, egui::Sense::drag());

                ui.painter().rect_filled(
                    separator_rect,
                    0.0,
                    if response.hovered() || response.dragged() {
                        egui::Color32::WHITE
                    } else {
                        egui::Color32::from_rgb(200, 200, 200)
                    },
                );
                ui.painter().line_segment(
                    [separator_rect.center() - egui::vec2(20.0, 0.0),
                     separator_rect.center() + egui::vec2(20.0, 0.0)],
                    egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 100, 100)),
                );

                if response.dragged() {
                    let delta = response.drag_delta().y / available_for_views;
                    self.vertical_split = (self.vertical_split + delta).clamp(0.2, 0.9);
                }

                if response.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
                }
            }

            if show_torus && show_ground_track {
                let torus_height = secondary_height * 0.6;
                let time = self.time;
                let torus_rotation = self.torus_rotation;
                let sat_radius = self.sat_radius;
                let torus_zoom = self.torus_zoom;
                let planet = &mut self.tabs[tab_idx].planets[planet_idx];
                let (trot, tzoom) = draw_torus(
                    ui,
                    &format!("torus_{}", view_name),
                    &constellations_data,
                    time,
                    torus_rotation,
                    viz_width,
                    torus_height,
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

                let ground_height = secondary_height * 0.4;
                draw_ground_track(
                    ui,
                    &format!("ground_{}", view_name),
                    &constellations_data,
                    viz_width,
                    ground_height,
                    self.sat_radius,
                    constellations_data.len() > 1,
                );
            } else if show_torus {
                let time = self.time;
                let torus_rotation = self.torus_rotation;
                let sat_radius = self.sat_radius;
                let torus_zoom = self.torus_zoom;
                let planet = &mut self.tabs[tab_idx].planets[planet_idx];
                let (trot, tzoom) = draw_torus(
                    ui,
                    &format!("torus_{}", view_name),
                    &constellations_data,
                    time,
                    torus_rotation,
                    viz_width,
                    secondary_height,
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
            } else if show_ground_track {
                draw_ground_track(
                    ui,
                    &format!("ground_{}", view_name),
                    &constellations_data,
                    viz_width,
                    secondary_height,
                    self.sat_radius,
                    single_color,
                );
            }
        }
        (add_planet, remove_planet)
    }

    fn show_settings(&mut self, ui: &mut egui::Ui) {
        let current_body = self.tabs.first()
            .and_then(|t| t.planets.first())
            .map(|p| p.celestial_body)
            .unwrap_or(CelestialBody::Earth);

        let active = self.active_tab_idx;
        let use_local = active < self.tabs.len() && self.tabs[active].use_local_settings;

        if use_local {
            ui.colored_label(egui::Color32::from_rgb(255, 200, 80), "Tab Override");
            ui.separator();
        }

        let (time_ref, rotation_ref, zoom_ref, speed_ref, animate_ref, earth_fixed_ref, follow_sat_ref, show_cam_ref) = if use_local {
            let s = &mut self.tabs[active].local_settings;
            (&mut s.time, &mut s.rotation, &mut s.zoom, &mut s.speed, &mut s.animate, &mut s.earth_fixed_camera, &mut s.follow_satellite, &mut s.show_camera_windows)
        } else {
            (&mut self.time, &mut self.rotation, &mut self.zoom, &mut self.speed, &mut self.animate, &mut self.earth_fixed_camera, &mut self.follow_satellite, &mut self.show_camera_windows)
        };

        let body_rotation = body_rotation_angle(current_body, *time_ref, self.current_gmst);

        ui.label(egui::RichText::new("Camera").strong());
        let (lat, base_lon) = matrix_to_lat_lon(rotation_ref);
        let geo_lon = if *earth_fixed_ref {
            base_lon
        } else {
            let mut l = base_lon - body_rotation;
            while l > std::f64::consts::PI { l -= 2.0 * std::f64::consts::PI; }
            while l < -std::f64::consts::PI { l += 2.0 * std::f64::consts::PI; }
            l
        };
        let mut lat_deg = lat.to_degrees();
        let mut lon_deg = geo_lon.to_degrees();
        ui.horizontal(|ui| {
            ui.label("Lat:");
            let lat_changed = ui.add(egui::DragValue::new(&mut lat_deg).speed(0.5).max_decimals(1).suffix("°")).changed();
            ui.label("Lon:");
            let lon_changed = ui.add(egui::DragValue::new(&mut lon_deg).speed(0.5).max_decimals(1).suffix("°")).changed();
            ui.label("Alt:");
            let mut alt_km = 10000.0 / *zoom_ref;
            if ui.add(egui::DragValue::new(&mut alt_km).range(0.5..=1000000.0).speed(100.0).suffix(" km")).changed() {
                *zoom_ref = (10000.0 / alt_km).clamp(0.01, 20000.0);
            }
            lat_deg = lat_deg.clamp(-90.0, 90.0);
            while lon_deg > 180.0 { lon_deg -= 360.0; }
            while lon_deg < -180.0 { lon_deg += 360.0; }
            if lat_changed || lon_changed {
                let target_lon = if *earth_fixed_ref {
                    lon_deg.to_radians()
                } else {
                    lon_deg.to_radians() + body_rotation
                };
                *rotation_ref = lat_lon_to_matrix(lat_deg.to_radians(), target_lon);
            }
        });
        let was_earth_fixed = *earth_fixed_ref;
        ui.checkbox(earth_fixed_ref, "Fixed Lat/Lon");
        if *earth_fixed_ref != was_earth_fixed {
            let cos_a = body_rotation.cos();
            let sin_a = body_rotation.sin();
            let body_y_rot = Matrix3::new(
                cos_a, 0.0, sin_a,
                0.0, 1.0, 0.0,
                -sin_a, 0.0, cos_a,
            );
            if *earth_fixed_ref {
                *rotation_ref *= body_y_rot;
            } else {
                *rotation_ref *= body_y_rot.transpose();
            }
        }
        ui.horizontal(|ui| {
            for (label, lat, lon) in [("N", 90.0_f64, 0.0_f64), ("S", -90.0, 0.0), ("E", 0.0, 90.0), ("W", 0.0, -90.0)] {
                if ui.button(label).clicked() {
                    let target_lon = if *earth_fixed_ref {
                        lon.to_radians()
                    } else {
                        lon.to_radians() + body_rotation
                    };
                    *rotation_ref = lat_lon_to_matrix(lat.to_radians(), target_lon);
                }
            }
        });

        ui.checkbox(follow_sat_ref, "Follow satellite");
        ui.checkbox(show_cam_ref, "Show camera windows");

        ui.separator();
        ui.label(egui::RichText::new("Simulation").strong());
        ui.horizontal(|ui| {
            ui.label("Speed:");
            ui.add(egui::DragValue::new(speed_ref).range(-86400.0..=86400.0).speed(1.0));
            if ui.button("⏪").clicked() {
                *speed_ref = -*speed_ref;
            }
            let pause_label = if *animate_ref { "⏸" } else { "▶" };
            if ui.button(pause_label).clicked() {
                *animate_ref = !*animate_ref;
            }
        });
        let start = self.start_timestamp;
        let real_timestamp = start + Duration::seconds(self.real_time as i64);
        let current_ts = start + Duration::seconds(*time_ref as i64);
        {
            use chrono::Timelike;
            use chrono::Datelike;
            let local = current_ts.with_timezone(&Local);
            let orig_time = *time_ref;
            let mut t_sec = *time_ref;
            let mut t_min = *time_ref;
            let mut t_hour = *time_ref;
            let mut t_day = *time_ref;
            let total_months = local.year() as f64 * 12.0 + local.month() as f64 - 1.0;
            let mut t_month = total_months;
            let mut t_year = total_months;
            let fmt_component = |secs: f64, f: fn(&chrono::DateTime<Local>) -> String| -> String {
                let ts = (start + Duration::seconds(secs as i64)).with_timezone(&Local);
                f(&ts)
            };
            ui.horizontal(|ui| {
                ui.label("Time:");
                ui.add(egui::DragValue::new(&mut t_hour)
                    .speed(360.0)
                    .custom_formatter(|s, _| fmt_component(s, |t| format!("{:02}", t.hour())))
                    .custom_parser(move |input| {
                        let h = input.parse::<u32>().ok()?.min(23);
                        let delta = (h as i64 - local.hour() as i64) * 3600;
                        Some(orig_time + delta as f64)
                    }));
                ui.label(":");
                ui.add(egui::DragValue::new(&mut t_min)
                    .speed(6.0)
                    .custom_formatter(|s, _| fmt_component(s, |t| format!("{:02}", t.minute())))
                    .custom_parser(move |input| {
                        let m = input.parse::<u32>().ok()?.min(59);
                        let delta = (m as i64 - local.minute() as i64) * 60;
                        Some(orig_time + delta as f64)
                    }));
                ui.label(":");
                ui.add(egui::DragValue::new(&mut t_sec)
                    .speed(0.1)
                    .custom_formatter(|s, _| fmt_component(s, |t| format!("{:02}", t.second())))
                    .custom_parser(move |input| {
                        let s = input.parse::<u32>().ok()?.min(59);
                        let delta = s as i64 - local.second() as i64;
                        Some(orig_time + delta as f64)
                    }));
            });
            ui.horizontal(|ui| {
                ui.label("Date:");
                ui.add(egui::DragValue::new(&mut t_day)
                    .speed(8640.0)
                    .custom_formatter(|s, _| fmt_component(s, |t| format!("{:02}", t.day())))
                    .custom_parser(move |input| {
                        let d = input.parse::<u32>().ok()?.clamp(1, 31);
                        let delta = (d as i64 - local.day() as i64) * 86400;
                        Some(orig_time + delta as f64)
                    }));
                ui.label("/");
                ui.add(egui::DragValue::new(&mut t_month)
                    .speed(0.1)
                    .custom_formatter(|v, _| {
                        let m = (v as i32).rem_euclid(12) + 1;
                        format!("{:02}", m)
                    })
                    .custom_parser(move |input| {
                        let m: i32 = input.parse().ok()?;
                        Some(local.year() as f64 * 12.0 + m.clamp(1, 12) as f64 - 1.0)
                    }));
                ui.label("/");
                ui.add(egui::DragValue::new(&mut t_year)
                    .speed(1.2)
                    .custom_formatter(|v, _| {
                        let y = (v / 12.0).floor() as i32;
                        format!("{}", y)
                    })
                    .custom_parser(move |input| {
                        let y: i32 = input.parse().ok()?;
                        Some(y as f64 * 12.0 + local.month() as f64 - 1.0)
                    }));
            });
            if t_sec != *time_ref { *time_ref = t_sec; }
            else if t_min != *time_ref {
                let d = t_min - *time_ref;
                *time_ref += (d / 60.0).round() * 60.0;
            }
            else if t_hour != *time_ref {
                let d = t_hour - *time_ref;
                *time_ref += (d / 3600.0).round() * 3600.0;
            }
            else if t_day != *time_ref {
                let d = t_day - *time_ref;
                *time_ref += (d / 86400.0).round() * 86400.0;
            }
            else {
                let apply_month_delta = |raw: f64, unit: f64| -> Option<i32> {
                    let d = raw - total_months;
                    if d.abs() < 0.01 { return None; }
                    Some((d / unit).round() as i32)
                };
                let month_delta = if t_month != total_months {
                    apply_month_delta(t_month, 1.0)
                } else if t_year != total_months {
                    apply_month_delta(t_year, 12.0).map(|d| d * 12)
                } else {
                    None
                };
                if let Some(md) = month_delta {
                    let mut m = local.month() as i32 - 1 + md;
                    let y = local.year() + m.div_euclid(12);
                    m = m.rem_euclid(12) + 1;
                    let d = local.day().min(
                        chrono::NaiveDate::from_ymd_opt(y, m as u32, 1)
                            .and_then(|d| d.checked_add_months(chrono::Months::new(1)))
                            .and_then(|d| d.pred_opt())
                            .map(|d| d.day())
                            .unwrap_or(28)
                    );
                    if let Some(date) = chrono::NaiveDate::from_ymd_opt(y, m as u32, d) {
                        if let Some(dt) = date.and_time(local.time()).and_local_timezone(Local).single() {
                            let diff = dt.with_timezone(&Utc).signed_duration_since(start);
                            *time_ref = diff.num_seconds() as f64;
                        }
                    }
                }
            }
        }
        let real_local = real_timestamp.with_timezone(&Local);
        ui.label(format!("Real: {}", real_local.format("%H:%M:%S %d/%m/%Y %Z")));
        if ui.button("Sync time").clicked() {
            *time_ref = self.real_time;
        }

        if use_local {
            let s = &mut self.tabs[active].local_settings;

            ui.separator();
            ui.label(egui::RichText::new("Display").strong());
            ui.checkbox(&mut s.show_orbits, "Show orbits");
            ui.checkbox(&mut s.show_intra_links, "Intra-plane links");
            ui.checkbox(&mut s.show_links, "Inter-plane links");
            ui.checkbox(&mut s.show_routing_paths, "Show routing paths");
            if s.show_routing_paths {
                ui.indent("routing_opts", |ui| {
                    ui.checkbox(&mut s.show_manhattan_path, "Manhattan (red)");
                    ui.checkbox(&mut s.show_shortest_path, "Shortest distance (green)");
                });
            }
            ui.checkbox(&mut s.show_asc_desc_colors, "Asc/Desc colors");
            ui.checkbox(&mut s.single_color, "Monochrome planes");
            ui.checkbox(&mut s.show_torus, "Show torus");

            ui.separator();
            ui.label(egui::RichText::new("Overlays").strong());
            ui.checkbox(&mut s.show_coverage, "Show coverage");
            if s.show_coverage {
                ui.horizontal(|ui| {
                    ui.label("Angle:");
                    ui.add(egui::DragValue::new(&mut s.coverage_angle)
                        .range(0.5..=70.0).speed(0.1).max_decimals(1).suffix("°"));
                });
            }
            ui.checkbox(&mut s.show_altitude_lines, "Altitude lines");
            ui.checkbox(&mut s.show_ground_track, "Show ground");
            ui.checkbox(&mut s.show_devices, "Show devices");
            ui.checkbox(&mut s.show_axes, "Show axes");
            ui.checkbox(&mut s.show_polar_circle, "Show polar circle");
            ui.checkbox(&mut s.show_equator, "Show equator");
            ui.checkbox(&mut s.show_borders, "Country borders");
            ui.checkbox(&mut s.show_cities, "City labels");
            ui.checkbox(&mut s.show_day_night, "Day/night cycle");
            ui.add_enabled(s.show_day_night, egui::Checkbox::new(&mut s.show_terminator, "Show terminator"));

            ui.separator();
            ui.label(egui::RichText::new("Rendering").strong());
            ui.checkbox(&mut s.render_planet, "Show planet");
            {
                let mut show_behind = !s.hide_behind_earth;
                if ui.add_enabled(s.render_planet, egui::Checkbox::new(&mut show_behind, "Show behind planet")).changed() {
                    s.hide_behind_earth = !show_behind;
                }
            }
            ui.checkbox(&mut s.show_clouds, "Show clouds");
            ui.checkbox(&mut s.show_stars, "Show stars");
            ui.add_enabled(s.show_stars, egui::Checkbox::new(&mut s.show_milky_way, "Show Milky Way"));
            ui.horizontal(|ui| {
                ui.label("Sat:");
                ui.add(egui::DragValue::new(&mut s.sat_radius).range(1.0..=15.0).speed(0.1));
                ui.label("Link:");
                ui.add(egui::DragValue::new(&mut s.link_width).range(0.1..=5.0).speed(0.1));
            });
            ui.checkbox(&mut s.fixed_sizes, "Fixed sizes (ignore alt)");
        } else {
            ui.separator();
            ui.label(egui::RichText::new("Display").strong());
            ui.checkbox(&mut self.show_orbits, "Show orbits");
            ui.checkbox(&mut self.show_intra_links, "Intra-plane links");
            ui.checkbox(&mut self.show_links, "Inter-plane links");
            ui.checkbox(&mut self.show_routing_paths, "Show routing paths");
            if self.show_routing_paths {
                ui.indent("routing_opts", |ui| {
                    ui.checkbox(&mut self.show_manhattan_path, "Manhattan (red)");
                    ui.checkbox(&mut self.show_shortest_path, "Shortest distance (green)");
                });
            }
            ui.checkbox(&mut self.show_asc_desc_colors, "Asc/Desc colors");
            ui.checkbox(&mut self.single_color, "Monochrome planes");
            ui.checkbox(&mut self.show_torus, "Show torus");
            ui.checkbox(&mut self.auto_cycle_tabs, "Auto-cycle tabs");
            if self.auto_cycle_tabs {
                ui.horizontal(|ui| {
                    ui.label("Interval:");
                    ui.add(egui::DragValue::new(&mut self.cycle_interval).range(1.0..=60.0).speed(0.5).suffix("s"));
                });
            }

            ui.separator();
            ui.label(egui::RichText::new("Overlays").strong());
            ui.checkbox(&mut self.show_coverage, "Show coverage");
            if self.show_coverage {
                ui.horizontal(|ui| {
                    ui.label("Angle:");
                    ui.add(egui::DragValue::new(&mut self.coverage_angle)
                        .range(0.5..=70.0).speed(0.1).max_decimals(1).suffix("°"));
                });
            }
            ui.checkbox(&mut self.show_altitude_lines, "Altitude lines");
            ui.checkbox(&mut self.show_ground_track, "Show ground");
            ui.checkbox(&mut self.show_devices, "Show devices");
            ui.checkbox(&mut self.show_axes, "Show axes");
            ui.checkbox(&mut self.show_polar_circle, "Show polar circle");
            ui.checkbox(&mut self.show_equator, "Show equator");
            ui.checkbox(&mut self.show_borders, "Country borders");
            ui.checkbox(&mut self.show_cities, "City labels");
            ui.checkbox(&mut self.show_day_night, "Day/night cycle");
            ui.add_enabled(self.show_day_night, egui::Checkbox::new(&mut self.show_terminator, "Show terminator"));

            ui.separator();
            ui.label(egui::RichText::new("Rendering").strong());
            ui.checkbox(&mut self.dark_mode, "Dark mode");
            ui.horizontal(|ui| {
                ui.label("Texture:");
                egui::ComboBox::from_id_salt("tex_res")
                    .selected_text(self.texture_resolution.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.texture_resolution, TextureResolution::R512, "512");
                        ui.selectable_value(&mut self.texture_resolution, TextureResolution::R1024, "1K");
                        ui.selectable_value(&mut self.texture_resolution, TextureResolution::R2048, "2K");
                        ui.selectable_value(&mut self.texture_resolution, TextureResolution::R8192, "8K");
                        #[cfg(not(target_arch = "wasm32"))]
                        ui.selectable_value(&mut self.texture_resolution, TextureResolution::R21504, "21K");
                    });
            });
            ui.checkbox(&mut self.use_gpu_rendering, "GPU rendering");
            #[cfg(not(target_arch = "wasm32"))]
            ui.checkbox(&mut self.tile_overlay.enabled, "Satellite tiles (Esri)");
            ui.checkbox(&mut self.render_planet, "Show planet");
            {
                let mut show_behind = !self.hide_behind_earth;
                if ui.add_enabled(self.render_planet, egui::Checkbox::new(&mut show_behind, "Show behind planet")).changed() {
                    self.hide_behind_earth = !show_behind;
                }
            }
            ui.checkbox(&mut self.show_clouds, "Show clouds");
            ui.checkbox(&mut self.show_stars, "Show stars");
            ui.add_enabled(self.show_stars, egui::Checkbox::new(&mut self.show_milky_way, "Show Milky Way"));
            ui.horizontal(|ui| {
                ui.label("Sat:");
                ui.add(egui::DragValue::new(&mut self.sat_radius).range(1.0..=15.0).speed(0.1));
                ui.label("Link:");
                ui.add(egui::DragValue::new(&mut self.link_width).range(0.1..=5.0).speed(0.1));
            });
            ui.checkbox(&mut self.fixed_sizes, "Fixed sizes (ignore alt)");
        }
    }

    #[allow(unused_variables)]
    fn load_texture_for_body(&mut self, body: CelestialBody, skin: Skin, ctx: &egui::Context) {
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

    fn load_cloud_texture(&mut self, _ctx: &egui::Context) {
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

    fn load_night_texture(&mut self, _ctx: &egui::Context) {
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

    fn load_star_textures(&mut self, _ctx: &egui::Context) {
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

#[cfg(target_arch = "wasm32")]
thread_local! {
    static TEXTURE_RESULT: std::cell::RefCell<Vec<((CelestialBody, Skin, TextureResolution), Result<EarthTexture, String>)>> = std::cell::RefCell::new(Vec::new());
    static STAR_TEXTURE_RESULT: std::cell::RefCell<Option<Result<EarthTexture, String>>> = std::cell::RefCell::new(None);
    static MILKY_WAY_TEXTURE_RESULT: std::cell::RefCell<Option<Result<EarthTexture, String>>> = std::cell::RefCell::new(None);
    static NIGHT_TEXTURE_RESULT: std::cell::RefCell<Option<Result<EarthTexture, String>>> = std::cell::RefCell::new(None);
    static CLOUD_TEXTURE_RESULT: std::cell::RefCell<Option<(TextureResolution, Result<EarthTexture, String>)>> = std::cell::RefCell::new(None);
    static TLE_FETCH_RESULT: std::cell::RefCell<Vec<(TlePreset, Result<Vec<TleSatellite>, String>)>> = std::cell::RefCell::new(Vec::new());
}

#[cfg(target_arch = "wasm32")]
async fn fetch_texture(url: &str) -> Result<EarthTexture, String> {
    use wasm_bindgen::JsCast as _;
    use web_sys::{Request, RequestInit, RequestMode, Response};

    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);

    let request = Request::new_with_str_and_init(url, &opts)
        .map_err(|e| format!("Failed to create request: {:?}", e))?;

    let window = web_sys::window().ok_or("No window")?;
    let resp_value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|e| format!("Fetch failed: {:?}", e))?;

    let resp: Response = resp_value.dyn_into()
        .map_err(|_| "Response is not a Response")?;

    if !resp.ok() {
        return Err(format!("HTTP error: {}", resp.status()));
    }

    let array_buffer = wasm_bindgen_futures::JsFuture::from(
        resp.array_buffer().map_err(|e| format!("Failed to get array buffer: {:?}", e))?
    )
    .await
    .map_err(|e| format!("Failed to read response: {:?}", e))?;

    let uint8_array = js_sys::Uint8Array::new(&array_buffer);
    let bytes: Vec<u8> = uint8_array.to_vec();

    EarthTexture::from_bytes(&bytes)
}

#[cfg(target_arch = "wasm32")]
async fn fetch_tle_text(url: &str) -> Result<String, String> {
    use wasm_bindgen::JsCast as _;
    use web_sys::{Request, RequestInit, Response};

    let opts = RequestInit::new();
    opts.set_method("GET");

    let request = Request::new_with_str_and_init(url, &opts)
        .map_err(|e| format!("{:?}", e))?;

    let window = web_sys::window().ok_or("No window")?;
    let resp_value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|e| format!("Fetch failed: {:?}", e))?;

    let resp: Response = resp_value.dyn_into()
        .map_err(|_| "Response is not a Response")?;

    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }

    let array_buffer = wasm_bindgen_futures::JsFuture::from(
        resp.array_buffer().map_err(|e| format!("{:?}", e))?
    )
    .await
    .map_err(|e| format!("{:?}", e))?;

    let bytes = js_sys::Uint8Array::new(&array_buffer).to_vec();
    String::from_utf8(bytes).map_err(|e| format!("{}", e))
}

#[cfg(target_arch = "wasm32")]
async fn yield_now() {
    wasm_bindgen_futures::JsFuture::from(
        js_sys::Promise::new(&mut |resolve, _| {
            web_sys::window().unwrap()
                .set_timeout_with_callback(&resolve).unwrap();
        })
    ).await.unwrap();
}

#[cfg(target_arch = "wasm32")]
async fn parse_tle_data_async(data: &str) -> Result<Vec<TleSatellite>, String> {
    let lines: Vec<&str> = data.lines().collect();
    let mut satellites = Vec::new();
    let mut i = 0;
    let mut batch = 0;
    while i + 2 < lines.len() {
        let name_line = lines[i].trim();
        let line1 = lines[i + 1].trim();
        let line2 = lines[i + 2].trim();
        if !line1.starts_with('1') || !line2.starts_with('2') {
            i += 1;
            continue;
        }
        let tle = format!("{}\n{}\n{}", name_line, line1, line2);
        if let Ok(elements_vec) = sgp4::parse_3les(&tle) {
            for elements in elements_vec {
                if let Ok(constants) = Constants::from_elements(&elements) {
                    let epoch_minutes = datetime_to_minutes(&elements.datetime);
                    satellites.push(TleSatellite {
                        name: elements.object_name.unwrap_or_default(),
                        inclination_deg: elements.inclination,
                        mean_motion: elements.mean_motion,
                        constants,
                        epoch_minutes,
                    });
                }
            }
        }
        i += 3;
        batch += 1;
        if batch % 100 == 0 {
            yield_now().await;
        }
    }
    if satellites.is_empty() {
        Err("No valid TLE data found".to_string())
    } else {
        Ok(satellites)
    }
}

impl App {
    fn setup_demo(&mut self) {
        let v = &mut self.viewer;
        v.tabs.clear();
        v.tab_counter = 0;

        let leo_tle = [
            TlePreset::Starlink, TlePreset::OneWeb, TlePreset::Kuiper, TlePreset::Iridium,
            TlePreset::IridiumNext, TlePreset::Globalstar, TlePreset::Orbcomm,
        ];
        let geo_tle = [
            TlePreset::Gps, TlePreset::Galileo, TlePreset::Glonass, TlePreset::Beidou,
            TlePreset::Molniya, TlePreset::Planet,
        ];

        // Tab 1: Inclination comparison (90° vs 60°)
        {
            v.tab_counter += 1;
            let mut tab = TabConfig::new_empty("Inclination: 90° vs 60°".to_string());
            for (inc, label) in [(90.0, "90°"), (60.0, "60°")] {
                tab.planet_counter += 1;
                let mut planet = PlanetConfig::new(format!("Earth ({})", label));
                planet.celestial_body = CelestialBody::Earth;
                let mut cons = ConstellationConfig::new(0);
                cons.sats_per_plane = 11;
                cons.num_planes = 6;
                cons.inclination = inc;
                cons.altitude_km = 780.0;
                cons.walker_type = WalkerType::Star;
                planet.constellations.push(cons);
                tab.planets.push(planet);
            }
            v.tabs.push(tab);
        }

        // Tab 2: Star vs Delta on Mars
        {
            v.tab_counter += 1;
            let mut tab = TabConfig::new_empty("Walker: Star vs Delta".to_string());
            for (wt, label) in [(WalkerType::Star, "Star"), (WalkerType::Delta, "Delta")] {
                tab.planet_counter += 1;
                let mut planet = PlanetConfig::new(format!("Mars ({})", label));
                planet.celestial_body = CelestialBody::Mars;
                let mut cons = ConstellationConfig::new(0);
                cons.sats_per_plane = 8;
                cons.num_planes = 4;
                cons.inclination = 70.0;
                cons.altitude_km = 500.0;
                cons.walker_type = wt;
                planet.constellations.push(cons);
                tab.planets.push(planet);
            }
            v.tabs.push(tab);
        }

        // Tab 3: Phasing comparison on Venus (F=0 vs F=2)
        {
            v.tab_counter += 1;
            let mut tab = TabConfig::new_empty("Phasing: F=0 vs F=2".to_string());
            for (f, label) in [(0.0, "F=0"), (2.0, "F=2")] {
                tab.planet_counter += 1;
                let mut planet = PlanetConfig::new(format!("Venus ({})", label));
                planet.celestial_body = CelestialBody::Venus;
                let mut cons = ConstellationConfig::new(0);
                cons.sats_per_plane = 6;
                cons.num_planes = 6;
                cons.inclination = 80.0;
                cons.altitude_km = 400.0;
                cons.phasing = f;
                planet.constellations.push(cons);
                tab.planets.push(planet);
            }
            v.tabs.push(tab);
        }

        // Tab 4: Altitude comparison on Mercury (VLEO, LEO, MEO, GEO)
        {
            v.tab_counter += 1;
            let mut tab = TabConfig::new_empty("Altitude: VLEO/LEO/MEO/GEO".to_string());
            tab.planet_counter += 1;
            let mut planet = PlanetConfig::new("Mercury".to_string());
            planet.celestial_body = CelestialBody::Mercury;
            let altitudes = [(200.0, 0), (550.0, 1), (8000.0, 2), (35786.0, 3)];
            for (alt, color) in altitudes {
                let mut cons = ConstellationConfig::new(color);
                cons.sats_per_plane = 1;
                cons.num_planes = 1;
                cons.inclination = 0.0;
                cons.altitude_km = alt;
                planet.constellations.push(cons);
            }
            tab.planets.push(planet);
            v.tabs.push(tab);
        }

        // Tab 5: Real LEO satellites
        {
            v.tab_counter += 1;
            let mut tab = TabConfig::new_empty("Real: LEO Constellations".to_string());
            tab.planet_counter += 1;
            let mut planet = PlanetConfig::new("Earth".to_string());
            planet.celestial_body = CelestialBody::Earth;
            for preset in leo_tle {
                planet.tle_selections.insert(preset, (true, TleLoadState::NotLoaded, None));
            }
            tab.planets.push(planet);
            v.tabs.push(tab);
        }

        // Tab 6: Real LEO + GEO satellites
        {
            v.tab_counter += 1;
            let mut tab = TabConfig::new_empty("Real: LEO + Navigation".to_string());
            tab.planet_counter += 1;
            let mut planet = PlanetConfig::new("Earth".to_string());
            planet.celestial_body = CelestialBody::Earth;
            for preset in leo_tle.iter().chain(geo_tle.iter()) {
                planet.tle_selections.insert(*preset, (true, TleLoadState::NotLoaded, None));
            }
            tab.planets.push(planet);
            v.tabs.push(tab);
        }

        // Tab 7: Simulated vs Real Starlink
        {
            v.tab_counter += 1;
            let mut tab = TabConfig::new_empty("Starlink: Simulated vs Real".to_string());
            // Simulated Starlink
            tab.planet_counter += 1;
            let mut planet_sim = PlanetConfig::new("Simulated".to_string());
            planet_sim.celestial_body = CelestialBody::Earth;
            let mut cons = ConstellationConfig::new(0);
            cons.preset = Preset::Starlink;
            cons.sats_per_plane = 22;
            cons.num_planes = 72;
            cons.inclination = 53.0;
            cons.altitude_km = 550.0;
            cons.walker_type = WalkerType::Delta;
            planet_sim.constellations.push(cons);
            tab.planets.push(planet_sim);
            // Real Starlink
            tab.planet_counter += 1;
            let mut planet_real = PlanetConfig::new("Real TLE".to_string());
            planet_real.celestial_body = CelestialBody::Earth;
            planet_real.tle_selections.insert(TlePreset::Starlink, (true, TleLoadState::NotLoaded, None));
            tab.planets.push(planet_real);
            v.tabs.push(tab);
        }

        // Reset dock state to show all tabs
        self.dock_state = DockState::new(vec![0]);
        for i in 1..v.tabs.len() {
            self.dock_state.push_to_focused_leaf(i);
        }

        // Enable auto-cycling
        v.auto_cycle_tabs = true;
        v.cycle_interval = 8.0;
        v.last_cycle_time = 0.0;
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

        if v.show_clouds {
            v.load_cloud_texture(ctx);
        }

        if v.show_day_night {
            v.load_night_texture(ctx);
        }

        if v.show_stars {
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
                if v.show_clouds {
                    if let Some(cloud_tex) = v.cloud_textures.get(&tex_res) {
                        renderer.upload_cloud_texture(gl, tex_res, cloud_tex);
                    }
                }
                if v.show_day_night {
                    if let Some(night_tex) = &v.night_texture {
                        renderer.upload_night_texture(gl, night_tex);
                    }
                }
                if v.show_stars {
                    if let Some(star_tex) = &v.star_texture {
                        renderer.upload_star_texture(gl, star_tex);
                    }
                    if v.show_milky_way {
                        if let Some(mw_tex) = &v.milky_way_texture {
                            renderer.upload_milky_way_texture(gl, mw_tex);
                        }
                    }
                }
                for (body, ring_tex) in &v.ring_textures {
                    renderer.upload_ring_texture(gl, *body, ring_tex);
                }
                renderer.evict_unused_textures(gl, &bodies_needed);
            }
        }

        let bodies_set: std::collections::HashSet<_> = bodies_needed.iter().copied().collect();
        v.planet_textures.retain(|k, _| bodies_set.contains(k));
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
                let surface_rotation = if v.earth_fixed_camera {
                    v.rotation
                } else {
                    let body_rot = body_rotation_angle(CelestialBody::Earth, v.time, v.current_gmst);
                    let (cb, sb) = (body_rot.cos(), body_rot.sin());
                    let body_mat = Matrix3::new(
                        cb, 0.0, sb,
                        0.0, 1.0, 0.0,
                        -sb, 0.0, cb,
                    );
                    v.rotation * body_mat
                };
                let view_scale = v.zoom / 1.15;
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
                let tile_deg = 360.0 / (1u64 << camera_zoom_to_tile_zoom(v.zoom).clamp(2, 18)) as f64;
                let min_half = tile_deg * 3.0;
                let lon_half = ((max_lon - min_lon) / 2.0 * margin).max(min_half);
                let lat_half = ((max_lat - min_lat) / 2.0 * margin).max(min_half);
                min_lon = lon_center - lon_half;
                max_lon = lon_center + lon_half;
                min_lat = (lat_center - lat_half).max(-85.0);
                max_lat = (lat_center + lat_half).min(85.0);
                let lon_span = max_lon - min_lon;

                let mut tile_zoom = camera_zoom_to_tile_zoom(v.zoom).max(2);
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

        if v.animate {
            v.time += dt * v.speed;
        }
        for tab in &mut v.tabs {
            if tab.use_local_settings && tab.local_settings.animate {
                tab.local_settings.time += dt * tab.local_settings.speed;
            }
        }

        let global_sim_seconds = if v.animate { dt * v.speed } else { 0.0 };
        for tab in &mut v.tabs {
            let sim_seconds = if tab.use_local_settings {
                if tab.local_settings.animate { dt * tab.local_settings.speed } else { 0.0 }
            } else {
                global_sim_seconds
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

        let sim_time = v.start_timestamp + Duration::seconds(v.time as i64);
        let gmst = greenwich_mean_sidereal_time(sim_time);
        v.current_gmst = gmst;

        if v.follow_satellite {
            if let Some(tab) = v.tabs.get(active_tab_idx) {
                if let Some(planet) = tab.planets.first() {
                    if let Some(cam) = planet.satellite_cameras.last() {
                        let set_follow_rotation = |radial: Vector3<f64>, velocity_dir: Vector3<f64>| -> Matrix3<f64> {
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
                            let propagation_minutes = v.start_timestamp.timestamp() as f64 / 60.0 + v.time / 60.0;
                            'tle_search: for preset in TlePreset::ALL.iter() {
                                let Some((selected, state, _)) = planet.tle_selections.get(preset) else { continue };
                                if !*selected { continue; }
                                let TleLoadState::Loaded { satellites, .. } = state else { continue };
                                if let Some(sat) = satellites.get(cam.sat_index) {
                                    let minutes_since_epoch = propagation_minutes - sat.epoch_minutes;
                                    if let Ok(prediction) = sat.constants.propagate(sgp4::MinutesSinceEpoch(minutes_since_epoch)) {
                                        let x = prediction.position[0];
                                        let y = prediction.position[2];
                                        let z = -prediction.position[1];
                                        let vx = prediction.velocity[0];
                                        let vy = prediction.velocity[2];
                                        let vz = -prediction.velocity[1];
                                        let radial: Vector3<f64> = Vector3::new(x, y, z).normalize();
                                        let velocity_dir: Vector3<f64> = Vector3::new(vx, vy, vz).normalize();
                                        v.rotation = set_follow_rotation(radial, velocity_dir);
                                        break 'tle_search;
                                    }
                                }
                            }
                        } else if let Some(cons) = planet.constellations.get(cam.constellation_idx) {
                            let planet_radius = planet.celestial_body.radius_km();
                            let planet_mu = planet.celestial_body.mu();
                            let planet_j2 = planet.celestial_body.j2();
                            let planet_eq_radius = planet.celestial_body.equatorial_radius_km();
                            let wc = cons.constellation(planet_radius, planet_mu, planet_j2, planet_eq_radius);
                            let dt = 0.1;
                            let pos_now = wc.satellite_positions(v.time);
                            let pos_next = wc.satellite_positions(v.time + dt);
                            if let Some(sat) = pos_now.iter().find(|s| s.plane == cam.plane && s.sat_index == cam.sat_index) {
                                if let Some(sat2) = pos_next.iter().find(|s| s.plane == cam.plane && s.sat_index == cam.sat_index) {
                                    let radial: Vector3<f64> = Vector3::new(sat.x, sat.y, sat.z).normalize();
                                    let velocity_dir: Vector3<f64> = Vector3::new(
                                        sat2.x - sat.x, sat2.y - sat.y, sat2.z - sat.z
                                    ).normalize();
                                    v.rotation = set_follow_rotation(radial, velocity_dir);
                                }
                            }
                        }
                    }
                }
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
            let rotation_changed = v.last_rotation.is_none_or(|r| r != v.rotation);
            let resolution_changed = v.last_resolution != v.earth_resolution;
            let time_changed = v.animate;

            for key in &bodies_needed {
                let texture_missing = !v.planet_image_handles.contains_key(key);
                let need_rerender = rotation_changed || resolution_changed || texture_missing || time_changed;
                if need_rerender {
                    if let Some(texture) = v.planet_textures.get(key) {
                        let body_rotation = body_rotation_angle(key.0, v.time, v.current_gmst);
                        let cos_a = body_rotation.cos();
                        let sin_a = body_rotation.sin();
                        let body_y_rotation = Matrix3::new(
                            cos_a, 0.0, sin_a,
                            0.0, 1.0, 0.0,
                            -sin_a, 0.0, cos_a,
                        );
                        let body_combined = if v.earth_fixed_camera {
                            v.rotation
                        } else {
                            v.rotation * body_y_rotation
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
                v.last_rotation = Some(v.rotation);
            }
            if resolution_changed {
                v.last_resolution = v.earth_resolution;
            }
        }

        if !self.viewer.show_side_panel {
            egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.add_space(4.0);
                    if ui.small_button("+").clicked() {
                        self.viewer.show_side_panel = true;
                    }
                });
            });
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
                        let active = self.viewer.active_tab_idx;
                        if active < self.viewer.tabs.len() {
                            let overriding = self.viewer.tabs[active].use_local_settings;
                            let label = if overriding { "[Tab ✓]" } else { "[Tab]" };
                            if ui.button(label).on_hover_text("Override settings for this tab").clicked() {
                                if !overriding {
                                    self.viewer.tabs[active].local_settings = TabSettings {
                                        time: self.viewer.time,
                                        speed: self.viewer.speed,
                                        animate: self.viewer.animate,
                                        zoom: self.viewer.zoom,
                                        rotation: self.viewer.rotation,
                                        earth_fixed_camera: self.viewer.earth_fixed_camera,
                                        follow_satellite: self.viewer.follow_satellite,
                                        show_camera_windows: self.viewer.show_camera_windows,
                                        show_orbits: self.viewer.show_orbits,
                                        show_links: self.viewer.show_links,
                                        show_intra_links: self.viewer.show_intra_links,
                                        show_coverage: self.viewer.show_coverage,
                                        coverage_angle: self.viewer.coverage_angle,
                                        show_routing_paths: self.viewer.show_routing_paths,
                                        show_manhattan_path: self.viewer.show_manhattan_path,
                                        show_shortest_path: self.viewer.show_shortest_path,
                                        show_asc_desc_colors: self.viewer.show_asc_desc_colors,
                                        single_color: self.viewer.single_color,
                                        show_torus: self.viewer.show_torus,
                                        show_ground_track: self.viewer.show_ground_track,
                                        show_axes: self.viewer.show_axes,
                                        hide_behind_earth: self.viewer.hide_behind_earth,
                                        render_planet: self.viewer.render_planet,
                                        show_altitude_lines: self.viewer.show_altitude_lines,
                                        show_devices: self.viewer.show_devices,
                                        show_polar_circle: self.viewer.show_polar_circle,
                                        show_equator: self.viewer.show_equator,
                                        show_borders: self.viewer.show_borders,
                                        show_cities: self.viewer.show_cities,
                                        show_day_night: self.viewer.show_day_night,
                                        show_terminator: self.viewer.show_terminator,
                                        show_clouds: self.viewer.show_clouds,
                                        show_stars: self.viewer.show_stars,
                                        show_milky_way: self.viewer.show_milky_way,
                                        sat_radius: self.viewer.sat_radius,
                                        link_width: self.viewer.link_width,
                                        fixed_sizes: self.viewer.fixed_sizes,
                                    };
                                }
                                self.viewer.tabs[active].use_local_settings = !overriding;
                            }
                        }
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
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        self.viewer.show_settings(ui);
                    });
                });
        }

        let mut dock_style = egui_dock::Style::from_egui(ctx.style().as_ref());
        dock_style.main_surface_border_stroke = egui::Stroke::NONE;
        DockArea::new(&mut self.dock_state)
            .show_add_buttons(true)
            .style(dock_style)
            .show(ctx, &mut self.viewer);

        if let Some(new_idx) = self.viewer.pending_add_tab.take() {
            self.dock_state.push_to_focused_leaf(new_idx);
        }

        if self.viewer.show_camera_windows {
            for tab in &self.viewer.tabs {
                for planet in &tab.planets {
                    let pr = planet.celestial_body.radius_km();
                    let pm = planet.celestial_body.mu();
                    let pj2 = planet.celestial_body.j2();
                    let peq = planet.celestial_body.equatorial_radius_km();
                    let texture = self.viewer.planet_textures.get(&(planet.celestial_body, planet.skin, self.viewer.texture_resolution));

                    let body_rot = body_rotation_angle(planet.celestial_body, self.viewer.time, self.viewer.current_gmst);
                    let cos_a = body_rot.cos();
                    let sin_a = body_rot.sin();
                    for camera in &planet.satellite_cameras {
                        let sat_data = planet.constellations.get(camera.constellation_idx).and_then(|cons| {
                            let wc = cons.constellation(pr, pm, pj2, peq);
                            let positions = wc.satellite_positions(self.viewer.time);
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
                                        self.viewer.coverage_angle,
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
        }
    }

    fn on_exit(&mut self, gl: Option<&glow::Context>) {
        if let Some(gl) = gl {
            if let Some(ref renderer) = self.viewer.sphere_renderer {
                renderer.lock().destroy(gl);
            }
        }
    }
}

fn draw_satellite_camera(
    ui: &mut egui::Ui,
    camera_id: usize,
    lat: f64,
    lon: f64,
    altitude_km: f64,
    coverage_angle: f64,
    earth_texture: &EarthTexture,
    planet_radius: f64,
) {
    let size = ui.available_size();
    let img_size = size.x.min(size.y - 40.0) as usize;
    if img_size < 10 {
        return;
    }

    let lat_rad = lat.to_radians();
    let lon_rad = lon.to_radians();
    let cone_half_angle = coverage_angle.to_radians();
    let orbit_radius = planet_radius + altitude_km;
    let max_earth_angle = (planet_radius / orbit_radius).acos();
    let earth_central_angle = (orbit_radius * cone_half_angle.sin() / planet_radius).asin();
    let angular_radius = earth_central_angle.min(max_earth_angle);

    let mut pixels = vec![egui::Color32::BLACK; img_size * img_size];

    for py in 0..img_size {
        for px in 0..img_size {
            let nx = (px as f64 / img_size as f64 - 0.5) * 2.0;
            let ny = (py as f64 / img_size as f64 - 0.5) * 2.0;

            let dist = (nx * nx + ny * ny).sqrt();
            if dist > 1.0 {
                continue;
            }

            let angle_from_nadir = dist * angular_radius;
            let azimuth = ny.atan2(nx);

            let clat = (lat_rad.sin() * angle_from_nadir.cos()
                + lat_rad.cos() * angle_from_nadir.sin() * (-azimuth).cos())
            .asin();
            let clon = lon_rad
                + (angle_from_nadir.sin() * (-azimuth).sin())
                    .atan2(lat_rad.cos() * angle_from_nadir.cos()
                        - lat_rad.sin() * angle_from_nadir.sin() * (-azimuth).cos());

            let u = (clon + PI) / (2.0 * PI);
            let v = (PI / 2.0 - clat) / PI;

            let [r, g, b] = earth_texture.sample(u, v);
            pixels[py * img_size + px] = egui::Color32::from_rgb(r, g, b);
        }
    }

    let image = egui::ColorImage {
        size: [img_size, img_size],
        pixels,
        source_size: egui::Vec2::ZERO,
    };
    let texture = ui.ctx().load_texture(
        format!("sat_cam_tex_{}", camera_id),
        image,
        egui::TextureOptions::LINEAR,
    );
    ui.image(&texture);

    ui.horizontal(|ui| {
        ui.label(format!("Lat: {:.1}°", lat));
        ui.label(format!("Lon: {:.1}°", lon));
    });
    ui.label(format!("Alt: {:.0} km", altitude_km));
}

fn wrap_index(current: usize, direction: i32, modulus: usize) -> usize {
    ((current as i32 + direction + modulus as i32) % modulus as i32) as usize
}

fn compute_path_direction(src: usize, dst: usize, modulus: usize, is_star: bool) -> (i32, usize) {
    if is_star {
        if dst >= src {
            (1, dst - src)
        } else {
            (-1, src - dst)
        }
    } else {
        let diff_fwd = (dst + modulus - src) % modulus;
        let diff_bwd = (src + modulus - dst) % modulus;
        if diff_fwd <= diff_bwd {
            (1, diff_fwd)
        } else {
            (-1, diff_bwd)
        }
    }
}

fn compute_manhattan_path(
    src_plane: usize, src_sat: usize,
    dst_plane: usize, dst_sat: usize,
    num_planes: usize, sats_per_plane: usize,
    is_star: bool,
) -> Vec<(usize, usize)> {
    let mut path = vec![(src_plane, src_sat)];

    let (plane_dir, plane_steps) = compute_path_direction(src_plane, dst_plane, num_planes, is_star);
    let (sat_dir, sat_steps) = compute_path_direction(src_sat, dst_sat, sats_per_plane, false);

    let mut cur_plane = src_plane;
    for _ in 0..plane_steps {
        cur_plane = wrap_index(cur_plane, plane_dir, num_planes);
        path.push((cur_plane, src_sat));
    }

    let mut cur_sat = src_sat;
    for _ in 0..sat_steps {
        cur_sat = wrap_index(cur_sat, sat_dir, sats_per_plane);
        path.push((dst_plane, cur_sat));
    }

    path
}

fn compute_shortest_path(
    src_plane: usize, src_sat: usize,
    dst_plane: usize, dst_sat: usize,
    num_planes: usize, sats_per_plane: usize,
    positions: &[SatelliteState],
    is_star: bool,
) -> Vec<(usize, usize)> {
    let mut path = vec![(src_plane, src_sat)];

    let (plane_dir, mut plane_steps_remaining) = compute_path_direction(src_plane, dst_plane, num_planes, is_star);
    let (sat_dir, mut sat_steps_remaining) = compute_path_direction(src_sat, dst_sat, sats_per_plane, false);

    let get_pos = |plane: usize, sat_idx: usize| -> Option<(f64, f64, f64)> {
        positions.iter()
            .find(|s| s.plane == plane && s.sat_index == sat_idx)
            .map(|s| (s.x, s.y, s.z))
    };

    let distance = |p1: (f64, f64, f64), p2: (f64, f64, f64)| -> f64 {
        let dx = p1.0 - p2.0;
        let dy = p1.1 - p2.1;
        let dz = p1.2 - p2.2;
        (dx * dx + dy * dy + dz * dz).sqrt()
    };

    let mut cur_plane = src_plane;
    let mut cur_sat = src_sat;

    while plane_steps_remaining > 0 || sat_steps_remaining > 0 {
        if plane_steps_remaining == 0 {
            cur_sat = wrap_index(cur_sat, sat_dir, sats_per_plane);
            sat_steps_remaining -= 1;
            path.push((cur_plane, cur_sat));
            continue;
        }
        if sat_steps_remaining == 0 {
            cur_plane = wrap_index(cur_plane, plane_dir, num_planes);
            plane_steps_remaining -= 1;
            path.push((cur_plane, cur_sat));
            continue;
        }

        let next_plane = wrap_index(cur_plane, plane_dir, num_planes);
        let next_sat = wrap_index(cur_sat, sat_dir, sats_per_plane);

        let cur_pos = get_pos(cur_plane, cur_sat);
        let cross_plane_pos = get_pos(next_plane, cur_sat);
        let within_plane_pos = get_pos(cur_plane, next_sat);
        let cross_plane_after_within = get_pos(next_plane, next_sat);

        match (cur_pos, cross_plane_pos, within_plane_pos, cross_plane_after_within) {
            (Some(cur), Some(cross), Some(within), Some(cross_after)) => {
                let cross_now = distance(cur, cross);
                let cross_after_within = distance(within, cross_after);

                if cross_now <= cross_after_within {
                    cur_plane = next_plane;
                    plane_steps_remaining -= 1;
                } else {
                    cur_sat = next_sat;
                    sat_steps_remaining -= 1;
                }
            }
            _ => {
                cur_plane = next_plane;
                plane_steps_remaining -= 1;
            }
        }
        path.push((cur_plane, cur_sat));
    }

    path
}

fn draw_routing_path(
    plot_ui: &mut egui_plot::PlotUi,
    path: &[(usize, usize)],
    positions: &[SatelliteState],
    rotation: &Matrix3<f64>,
    color: egui::Color32,
    width: f32,
    hide_behind_earth: bool,
    earth_r_sq: f64,
) {
    if path.len() < 2 {
        return;
    }

    for i in 0..(path.len() - 1) {
        let (plane1, sat1) = path[i];
        let (plane2, sat2) = path[i + 1];

        let pos1 = positions.iter().find(|s| s.plane == plane1 && s.sat_index == sat1);
        let pos2 = positions.iter().find(|s| s.plane == plane2 && s.sat_index == sat2);

        if let (Some(p1), Some(p2)) = (pos1, pos2) {
            let (rx1, ry1, rz1) = rotate_point_matrix(p1.x, p1.y, p1.z, rotation);
            let (rx2, ry2, rz2) = rotate_point_matrix(p2.x, p2.y, p2.z, rotation);

            let visible1 = rz1 >= 0.0 || (rx1 * rx1 + ry1 * ry1) >= earth_r_sq;
            let visible2 = rz2 >= 0.0 || (rx2 * rx2 + ry2 * ry2) >= earth_r_sq;

            if hide_behind_earth && !visible1 && !visible2 {
                continue;
            }

            let line_color = if visible1 && visible2 {
                color
            } else {
                egui::Color32::from_rgba_unmultiplied(
                    color.r() / 2, color.g() / 2, color.b() / 2, 150,
                )
            };

            plot_ui.line(
                Line::new("", PlotPoints::new(vec![[rx1, ry1], [rx2, ry2]]))
                    .color(line_color)
                    .width(width),
            );
        }
    }
}

fn draw_3d_view(
    ui: &mut egui::Ui,
    id: &str,
    constellations: &[(WalkerConstellation, Vec<SatelliteState>, usize, u8, usize, String)],
    flags: View3DFlags,
    coverage_angle: f64,
    mut rotation: Matrix3<f64>,
    satellite_rotation: Matrix3<f64>,
    width: f32,
    height: f32,
    earth_texture: Option<&egui::TextureHandle>,
    mut zoom: f64,
    sat_radius: f32,
    link_width: f32,
    pending_cameras: &mut Vec<SatelliteCamera>,
    camera_id_counter: &mut usize,
    satellite_cameras: &mut [SatelliteCamera],
    cameras_to_remove: &mut Vec<usize>,
    planet_radius: f64,
    flattening: f64,
    sphere_renderer: Option<&Arc<Mutex<SphereRenderer>>>,
    body_key: (CelestialBody, Skin, TextureResolution),
    body_rotation: &Matrix3<f64>,
    sun_dir: [f32; 3],
    time: f64,
    ground_stations: &mut [GroundStation],
    areas_of_interest: &mut [AreaOfInterest],
    device_layers: &[DeviceLayer],
    body_rot_angle: f64,
    dragging_place: &mut Option<(usize, usize, bool, usize)>,
    drag_tab_planet: (usize, usize),
    detail_gl_info: Option<(glow::Texture, [f32; 4])>,
    geo_borders: &[Vec<(f64, f64)>],
    geo_cities: &[CityLabel],
) -> (Matrix3<f64>, f64) {
    let View3DFlags {
        show_orbits, show_axes, show_coverage, show_links, show_intra_links,
        hide_behind_earth, single_color, dark_mode, show_routing_paths,
        show_manhattan_path, show_shortest_path, show_asc_desc_colors,
        show_altitude_lines, render_planet, fixed_sizes, show_polar_circle,
        show_equator, show_terminator, earth_fixed_camera, use_gpu_rendering,
        show_clouds, show_day_night, show_stars, show_milky_way, show_borders, show_cities,
    } = flags;
    let max_altitude = constellations.iter()
        .map(|(c, _, _, _, _, _)| c.altitude_km)
        .fold(550.0_f64, |a, b| a.max(b));
    let orbit_radius = planet_radius + max_altitude;
    let axis_len = orbit_radius * 1.05;
    let planet_view_reference = planet_radius * 1.15;
    let margin = planet_view_reference / zoom;
    let zoom_factor = if fixed_sizes { 1.0 } else { zoom as f32 };
    let scaled_sat_radius = sat_radius * zoom_factor;
    let scaled_link_width = (link_width * zoom_factor).max(0.5);

    let use_gpu = sphere_renderer.is_some() && render_planet && use_gpu_rendering;

    // Draw sphere FIRST (before plot) so it renders behind
    if use_gpu {
        let rect = egui::Rect::from_min_size(ui.cursor().min, egui::Vec2::new(width, height));
        let renderer = sphere_renderer.unwrap().clone();
        let combined_rotation = if earth_fixed_camera {
            rotation
        } else {
            rotation * body_rotation
        };
        let inv_rotation = combined_rotation.transpose();
        let flat = flattening as f32;
        let aspect = width / height;
        let key = body_key;
        let scale = (planet_radius / margin) as f32;
        let atmosphere = match body_key.0 {
            CelestialBody::Earth => 1.0_f32,
            _ => 0.0,
        };

        let bg = ui.visuals().extreme_bg_color;
        let bg_color = [bg.r() as f32 / 255.0, bg.g() as f32 / 255.0, bg.b() as f32 / 255.0];
        let detail_info = detail_gl_info;
        let callback = egui::PaintCallback {
            rect,
            callback: Arc::new(egui_glow::CallbackFn::new(move |_info, painter| {
                let gl = painter.gl();
                let r = renderer.lock();
                if let Some((detail_tex, detail_bounds)) = detail_info {
                    let dt = DetailTexture {
                        width: 0,
                        height: 0,
                        bounds: DetailBounds {
                            min_lon: detail_bounds[0] as f64,
                            max_lon: detail_bounds[1] as f64,
                            min_lat: detail_bounds[2] as f64,
                            max_lat: detail_bounds[3] as f64,
                        },
                        gl_texture: Some(detail_tex),
                    };
                    r.paint(gl, key, &inv_rotation, flat as f64, aspect, scale, atmosphere, show_clouds, show_day_night, sun_dir, Some(&dt), show_stars, show_milky_way, bg_color);
                } else {
                    r.paint(gl, key, &inv_rotation, flat as f64, aspect, scale, atmosphere, show_clouds, show_day_night, sun_dir, None, show_stars, show_milky_way, bg_color);
                }
            })),
        };
        ui.painter().add(callback);
    }

    let plot = Plot::new(id)
        .data_aspect(1.0)
        .width(width)
        .height(height)
        .show_axes(false)
        .show_grid(false)
        .show_x(false)
        .show_y(false)
        .show_background(sphere_renderer.is_none() || !use_gpu_rendering)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .cursor_color(egui::Color32::TRANSPARENT);

    let mut surface_labels: Vec<([f64; 2], String, egui::Color32, bool, usize)> = Vec::new();
    let mut device_cluster_labels: Vec<([f64; 2], usize, egui::Color32)> = Vec::new();

    let response = plot.show(ui, |plot_ui| {
        let ground_stations = &*ground_stations;
        let areas_of_interest = &*areas_of_interest;
        plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
            [-margin, -margin],
            [margin, margin],
        ));

        let visual_earth_r = planet_radius * 0.95;
        let earth_r_sq = visual_earth_r * visual_earth_r;

        if show_orbits && !hide_behind_earth {
            for (constellation, _, color_offset, tle_kind, _, _) in constellations {
                if *tle_kind != 0 { continue; }
                for plane in 0..constellation.num_planes {
                    let orbit_pts = constellation.orbit_points_3d(plane, time);
                    let color = plane_color(if single_color { *color_offset } else { plane + color_offset });

                    let mut behind_segment: Vec<[f64; 2]> = Vec::new();
                    for &(x, y, z) in &orbit_pts {
                        let (rx, ry, rz) = rotate_point_matrix(x, y, z, &satellite_rotation);
                        let occluded = rz < 0.0 && (rx * rx + ry * ry) < earth_r_sq;
                        if occluded {
                            behind_segment.push([rx, ry]);
                        } else if !behind_segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut behind_segment)))
                                    .color(dim_color(color))
                                    .width(scaled_link_width),
                            );
                        }
                    }
                    if !behind_segment.is_empty() {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(behind_segment))
                                .color(dim_color(color))
                                .width(scaled_link_width),
                        );
                    }
                }
            }
        }

        if !hide_behind_earth {
            for (constellation, positions, color_offset, _tle_kind, _, _) in constellations {
                for plane in 0..constellation.num_planes {
                    let color = plane_color(if single_color { *color_offset } else { plane + color_offset });
                    let pts: PlotPoints = positions
                        .iter()
                        .filter_map(|s| {
                            if s.plane != plane {
                                return None;
                            }
                            let (rx, ry, rz) = rotate_point_matrix(s.x, s.y, s.z, &satellite_rotation);
                            if rz < 0.0 {
                                Some([rx, ry])
                            } else {
                                None
                            }
                        })
                        .collect();
                    plot_ui.points(
                        Points::new("", pts)
                            .color(dim_color(color))
                            .radius(scaled_sat_radius * 0.8)
                            .filled(true),
                    );
                }
            }
        }

        if render_planet {
            if use_gpu {
                // GPU rendering is handled by paint callback before the plot
            } else if let Some(tex) = earth_texture {
                let size = egui::Vec2::splat(planet_radius as f32 * 2.0);
                plot_ui.image(PlotImage::new(
                    "",
                    tex,
                    PlotPoint::new(0.0, 0.0),
                    size,
                ));
            } else {
                let earth_pts: PlotPoints = (0..=100)
                    .map(|i| {
                        let theta = 2.0 * PI * i as f64 / 100.0;
                        [planet_radius * theta.cos(), planet_radius * theta.sin()]
                    })
                    .collect();
                plot_ui.polygon(
                    Polygon::new("", earth_pts)
                        .fill_color(egui::Color32::from_rgb(30, 60, 120))
                        .stroke(egui::Stroke::new(2.0, egui::Color32::from_rgb(70, 130, 180))),
                );
            }

            if dark_mode && !use_gpu {
                let equatorial_r = planet_radius;
                let polar_r = planet_radius * (1.0 - flattening);
                let border_pts: PlotPoints = (0..=100)
                    .map(|i| {
                        let theta = 2.0 * PI * i as f64 / 100.0;
                        [equatorial_r * theta.cos(), polar_r * theta.sin()]
                    })
                    .collect();
                plot_ui.line(Line::new("", border_pts).color(egui::Color32::WHITE).width(1.0));
            }

            if show_polar_circle {
                let polar_r = planet_radius * (1.0 - flattening);
                let circle_pts: PlotPoints = (0..=100)
                    .map(|i| {
                        let theta = 2.0 * PI * i as f64 / 100.0;
                        [polar_r * theta.cos(), polar_r * theta.sin()]
                    })
                    .collect();
                plot_ui.line(Line::new("", circle_pts)
                    .color(egui::Color32::from_rgb(255, 200, 50))
                    .width(1.0));
            }

            if show_terminator {
                let sun = Vector3::new(sun_dir[0] as f64, sun_dir[1] as f64, sun_dir[2] as f64).normalize();
                let up = if sun.y.abs() > 0.9 {
                    Vector3::new(1.0, 0.0, 0.0)
                } else {
                    Vector3::new(0.0, 1.0, 0.0)
                };
                let u = sun.cross(&up).normalize();
                let v = sun.cross(&u).normalize();
                let term_rotation = if earth_fixed_camera {
                    rotation
                } else {
                    rotation * *body_rotation
                };
                let terminator_pts: PlotPoints = (0..=100)
                    .map(|i| {
                        let theta = 2.0 * PI * i as f64 / 100.0;
                        let x = planet_radius * (u.x * theta.cos() + v.x * theta.sin());
                        let y = planet_radius * (u.y * theta.cos() + v.y * theta.sin());
                        let z = planet_radius * (u.z * theta.cos() + v.z * theta.sin());
                        let (sx, sy, _) = rotate_point_matrix(x, y, z, &term_rotation);
                        [sx, sy]
                    })
                    .collect();
                plot_ui.line(Line::new("", terminator_pts)
                    .color(egui::Color32::from_rgb(255, 180, 0))
                    .width(2.0));
            }
        }

        if show_coverage {
            for (constellation, positions, color_offset, _tle_kind, _, _) in constellations {
                let orbit_radius = planet_radius + constellation.altitude_km;
                let cone_half_angle = coverage_angle.to_radians();
                let max_earth_angle = (planet_radius / orbit_radius).acos();
                let earth_central_angle = (orbit_radius * cone_half_angle.sin() / planet_radius).asin();
                let angular_radius = earth_central_angle.min(max_earth_angle);
                for sat in positions {
                    let lat = sat.lat.to_radians();
                    let lon = sat.lon.to_radians();

                    let coverage_pts: Vec<([f64; 2], bool)> = (0..=32)
                        .map(|i| {
                            let angle = 2.0 * PI * i as f64 / 32.0;

                            let clat = (lat.sin() * angular_radius.cos()
                                + lat.cos() * angular_radius.sin() * angle.cos())
                            .asin();
                            let clon = lon
                                + (angular_radius.sin() * angle.sin())
                                    .atan2(lat.cos() * angular_radius.cos()
                                        - lat.sin() * angular_radius.sin() * angle.cos());

                            let x = planet_radius * clat.cos() * clon.cos();
                            let y = planet_radius * clat.sin();
                            let z = planet_radius * clat.cos() * clon.sin();

                            let (rx, ry, rz) = rotate_point_matrix(x, y, z, &satellite_rotation);
                            ([rx, ry], rz >= 0.0)
                        })
                        .collect();

                    let all_visible = coverage_pts.iter().all(|(_, vis)| *vis);
                    let color = plane_color(sat.plane + color_offset);

                    if all_visible {
                        let pts: Vec<[f64; 2]> = coverage_pts.iter().map(|(p, _)| *p).collect();
                        let fill = egui::Color32::from_rgba_unmultiplied(
                            color.r(), color.g(), color.b(), 60
                        );
                        plot_ui.polygon(
                            Polygon::new("", PlotPoints::new(pts))
                                .fill_color(fill)
                                .stroke(egui::Stroke::new(1.0, color)),
                        );
                    } else {
                        let mut segment: Vec<[f64; 2]> = Vec::new();
                        for (pt, visible) in &coverage_pts {
                            if *visible {
                                segment.push(*pt);
                            } else if !segment.is_empty() {
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(std::mem::take(&mut segment)))
                                        .color(color)
                                        .width(scaled_link_width),
                                );
                            }
                        }
                        if !segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(segment))
                                    .color(color)
                                    .width(scaled_link_width),
                            );
                        }
                    }
                }
            }
        }

        let surface_rotation = if earth_fixed_camera {
            rotation
        } else {
            rotation * *body_rotation
        };

        if show_equator {
            let n_pts = 200;
            let eq_color = egui::Color32::from_rgb(255, 100, 100);
            let dim_eq = egui::Color32::from_rgba_unmultiplied(255, 100, 100, 60);
            let mut front_seg: Vec<[f64; 2]> = Vec::new();
            let mut back_seg: Vec<[f64; 2]> = Vec::new();
            for i in 0..=n_pts {
                let theta = 2.0 * PI * i as f64 / n_pts as f64;
                let x = planet_radius * theta.cos();
                let z = planet_radius * theta.sin();
                let (rx, ry, rz) = rotate_point_matrix(x, 0.0, z, &surface_rotation);
                let occluded = rz < 0.0 && (rx * rx + ry * ry) < earth_r_sq;
                if occluded {
                    if !front_seg.is_empty() {
                        plot_ui.line(Line::new("", PlotPoints::new(std::mem::take(&mut front_seg)))
                            .color(eq_color).width(1.5));
                    }
                    back_seg.push([rx, ry]);
                } else {
                    if !back_seg.is_empty() {
                        if !hide_behind_earth {
                            plot_ui.line(Line::new("", PlotPoints::new(std::mem::take(&mut back_seg)))
                                .color(dim_eq).width(1.0));
                        } else {
                            back_seg.clear();
                        }
                    }
                    front_seg.push([rx, ry]);
                }
            }
            if !front_seg.is_empty() {
                plot_ui.line(Line::new("", PlotPoints::new(front_seg))
                    .color(eq_color).width(1.5));
            }
            if !back_seg.is_empty() && !hide_behind_earth {
                plot_ui.line(Line::new("", PlotPoints::new(back_seg))
                    .color(dim_eq).width(1.0));
            }
        }

        if show_borders && body_key.0 == CelestialBody::Earth {
            let border_color = egui::Color32::from_rgb(0, 0, 0);
            let dim_border = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 40);
            for polyline in geo_borders {
                let mut front_seg: Vec<[f64; 2]> = Vec::new();
                let mut back_seg: Vec<[f64; 2]> = Vec::new();
                for &(lat_deg, lon_deg) in polyline {
                    let lat = lat_deg.to_radians();
                    let lon = (-lon_deg).to_radians();
                    let x = planet_radius * lat.cos() * lon.cos();
                    let y = planet_radius * lat.sin();
                    let z = planet_radius * lat.cos() * lon.sin();
                    let (rx, ry, rz) = rotate_point_matrix(x, y, z, &surface_rotation);
                    let occluded = rz < 0.0 && (rx * rx + ry * ry) < earth_r_sq;
                    if occluded {
                        if !front_seg.is_empty() {
                            plot_ui.line(Line::new("", PlotPoints::new(std::mem::take(&mut front_seg)))
                                .color(border_color).width(1.0));
                        }
                        back_seg.push([rx, ry]);
                    } else {
                        if !back_seg.is_empty() {
                            if !hide_behind_earth {
                                plot_ui.line(Line::new("", PlotPoints::new(std::mem::take(&mut back_seg)))
                                    .color(dim_border).width(0.5));
                            } else {
                                back_seg.clear();
                            }
                        }
                        front_seg.push([rx, ry]);
                    }
                }
                if !front_seg.is_empty() {
                    plot_ui.line(Line::new("", PlotPoints::new(front_seg))
                        .color(border_color).width(1.0));
                }
                if !back_seg.is_empty() && !hide_behind_earth {
                    plot_ui.line(Line::new("", PlotPoints::new(back_seg))
                        .color(dim_border).width(0.5));
                }
            }
        }

        if show_cities && body_key.0 == CelestialBody::Earth {
            let min_pop = if zoom >= 8.0 { 0.0 }
                else if zoom >= 4.0 { 500_000.0 }
                else if zoom >= 2.0 { 2_000_000.0 }
                else { 5_000_000.0 };
            let max_cities = if zoom >= 8.0 { 200 }
                else if zoom >= 4.0 { 80 }
                else if zoom >= 2.0 { 30 }
                else { 15 };
            let city_color = egui::Color32::from_rgb(220, 220, 200);
            let mut count = 0usize;
            for city in geo_cities {
                if city.population < min_pop { continue; }
                if count >= max_cities { break; }
                let lat = city.lat.to_radians();
                let lon = (-city.lon).to_radians();
                let x = planet_radius * lat.cos() * lon.cos();
                let y = planet_radius * lat.sin();
                let z = planet_radius * lat.cos() * lon.sin();
                let (rx, ry, rz) = rotate_point_matrix(x, y, z, &surface_rotation);
                if !hide_behind_earth || rz >= 0.0 {
                    surface_labels.push(([rx, ry], city.name.clone(), city_color, false, 500_000 + count));
                    count += 1;
                }
            }
        }

        for (aoi_idx, aoi) in areas_of_interest.iter().enumerate() {
            let lat = aoi.lat.to_radians();
            let lon = (-aoi.lon).to_radians();
            let angular_radius = aoi.radius_km / planet_radius;

            let aoi_pts: Vec<([f64; 2], bool)> = (0..=32)
                .map(|i| {
                    let angle = 2.0 * PI * i as f64 / 32.0;
                    let clat = (lat.sin() * angular_radius.cos()
                        + lat.cos() * angular_radius.sin() * angle.cos())
                    .asin();
                    let clon = lon
                        + (angular_radius.sin() * angle.sin())
                            .atan2(lat.cos() * angular_radius.cos()
                                - lat.sin() * angular_radius.sin() * angle.cos());

                    let x = planet_radius * clat.cos() * clon.cos();
                    let y = planet_radius * clat.sin();
                    let z = planet_radius * clat.cos() * clon.sin();

                    let (rx, ry, rz) = rotate_point_matrix(x, y, z, &surface_rotation);
                    ([rx, ry], rz >= 0.0)
                })
                .collect();

            let all_visible = aoi_pts.iter().all(|(_, vis)| *vis);
            if all_visible {
                let pts: Vec<[f64; 2]> = aoi_pts.iter().map(|(p, _)| *p).collect();
                let fill = egui::Color32::from_rgba_unmultiplied(
                    aoi.color.r(), aoi.color.g(), aoi.color.b(), aoi.color.a()
                );
                plot_ui.polygon(
                    Polygon::new("", PlotPoints::new(pts))
                        .fill_color(fill)
                        .stroke(egui::Stroke::new(2.0, aoi.color)),
                );
            } else {
                let mut segment: Vec<[f64; 2]> = Vec::new();
                for (pt, visible) in &aoi_pts {
                    if *visible {
                        segment.push(*pt);
                    } else if !segment.is_empty() {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(std::mem::take(&mut segment)))
                                .color(aoi.color)
                                .width(2.0),
                        );
                    }
                }
                if !segment.is_empty() {
                    plot_ui.line(
                        Line::new("", PlotPoints::new(segment))
                            .color(aoi.color)
                            .width(2.0),
                    );
                }
            }

            let cx = planet_radius * lat.cos() * lon.cos();
            let cy = planet_radius * lat.sin();
            let cz = planet_radius * lat.cos() * lon.sin();
            let (crx, cry, crz) = rotate_point_matrix(cx, cy, cz, &surface_rotation);
            if !hide_behind_earth || crz >= 0.0 {
                surface_labels.push(([crx, cry], aoi.name.clone(), aoi.color, false, aoi_idx));
            }
        }

        for (gs_idx, gs) in ground_stations.iter().enumerate() {
            let lat = gs.lat.to_radians();
            let lon = (-gs.lon).to_radians();
            let angular_radius = gs.radius_km / planet_radius;

            let gs_pts: Vec<([f64; 2], bool)> = (0..=32)
                .map(|i| {
                    let angle = 2.0 * PI * i as f64 / 32.0;
                    let clat = (lat.sin() * angular_radius.cos()
                        + lat.cos() * angular_radius.sin() * angle.cos())
                    .asin();
                    let clon = lon
                        + (angular_radius.sin() * angle.sin())
                            .atan2(lat.cos() * angular_radius.cos()
                                - lat.sin() * angular_radius.sin() * angle.cos());

                    let x = planet_radius * clat.cos() * clon.cos();
                    let y = planet_radius * clat.sin();
                    let z = planet_radius * clat.cos() * clon.sin();

                    let (rx, ry, rz) = rotate_point_matrix(x, y, z, &surface_rotation);
                    ([rx, ry], rz >= 0.0)
                })
                .collect();

            let all_visible = gs_pts.iter().all(|(_, vis)| *vis);
            if all_visible {
                let pts: Vec<[f64; 2]> = gs_pts.iter().map(|(p, _)| *p).collect();
                let fill = egui::Color32::from_rgba_unmultiplied(
                    gs.color.r(), gs.color.g(), gs.color.b(), 50
                );
                plot_ui.polygon(
                    Polygon::new("", PlotPoints::new(pts))
                        .fill_color(fill)
                        .stroke(egui::Stroke::new(2.0, gs.color)),
                );
            } else {
                let mut segment: Vec<[f64; 2]> = Vec::new();
                for (pt, visible) in &gs_pts {
                    if *visible {
                        segment.push(*pt);
                    } else if !segment.is_empty() {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(std::mem::take(&mut segment)))
                                .color(gs.color)
                                .width(2.0),
                        );
                    }
                }
                if !segment.is_empty() {
                    plot_ui.line(
                        Line::new("", PlotPoints::new(segment))
                            .color(gs.color)
                            .width(2.0),
                    );
                }
            }

            let x = planet_radius * lat.cos() * lon.cos();
            let y = planet_radius * lat.sin();
            let z = planet_radius * lat.cos() * lon.sin();
            let (rx, ry, rz) = rotate_point_matrix(x, y, z, &surface_rotation);

            if !hide_behind_earth || rz >= 0.0 {
                surface_labels.push(([rx, ry], gs.name.clone(), gs.color, true, gs_idx));
            }
        }

        let show_device_dots = zoom >= 2000.0;

        for (layer_idx, layer) in device_layers.iter().enumerate() {
            if layer.devices.is_empty() { continue; }

            let mut projected: Vec<([f64; 2], bool)> = Vec::new();
            for &(lat_deg, lon_deg) in &layer.devices {
                let lat = lat_deg.to_radians();
                let lon = (-lon_deg).to_radians();
                let x = planet_radius * lat.cos() * lon.cos();
                let y = planet_radius * lat.sin();
                let z = planet_radius * lat.cos() * lon.sin();
                let (rx, ry, rz) = rotate_point_matrix(x, y, z, &surface_rotation);
                let visible = !hide_behind_earth || rz >= 0.0;
                projected.push(([rx, ry], visible));
            }

            if show_device_dots {
                for (dev_idx, (pos, vis)) in projected.iter().enumerate() {
                    if !vis { continue; }
                    plot_ui.points(
                        Points::new("", PlotPoints::new(vec![*pos]))
                            .color(layer.color)
                            .radius(scaled_sat_radius * 0.8),
                    );
                    let label = format!("{} #{}", layer.name, dev_idx + 1);
                    surface_labels.push((
                        *pos,
                        label,
                        layer.color,
                        false,
                        1000000 + layer_idx * 10000 + dev_idx,
                    ));
                }
            } else {
                let cell_size = planet_radius * 0.15 / zoom.max(0.1);
                let min_circle_r = 2.0;
                let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> = std::collections::HashMap::new();
                for (i, (pos, vis)) in projected.iter().enumerate() {
                    if !vis { continue; }
                    let gx = (pos[0] / cell_size).floor() as i64;
                    let gy = (pos[1] / cell_size).floor() as i64;
                    grid.entry((gx, gy)).or_default().push(i);
                }

                for indices in grid.values() {
                    let count = indices.len();
                    let cx: f64 = indices.iter().map(|&i| projected[i].0[0]).sum::<f64>() / count as f64;
                    let cy: f64 = indices.iter().map(|&i| projected[i].0[1]).sum::<f64>() / count as f64;

                    let circle_r = (cell_size * 0.35).max(min_circle_r);
                    let fill = egui::Color32::from_rgba_unmultiplied(
                        layer.color.r(), layer.color.g(), layer.color.b(), 60,
                    );
                    let n = 24;
                    let pts: Vec<[f64; 2]> = (0..=n)
                        .map(|i| {
                            let a = 2.0 * PI * i as f64 / n as f64;
                            [cx + circle_r * a.cos(), cy + circle_r * a.sin()]
                        })
                        .collect();
                    plot_ui.polygon(
                        Polygon::new("", PlotPoints::new(pts))
                            .fill_color(fill)
                            .stroke(egui::Stroke::new(1.5, layer.color)),
                    );

                    device_cluster_labels.push(([cx, cy], count, layer.color));
                }
            }
        }

        if show_axes {
            let (ep_x, ep_y, _) = rotate_point_matrix(axis_len, 0.0, 0.0, &satellite_rotation);
            let (wn_x, wn_y, _) = rotate_point_matrix(-axis_len, 0.0, 0.0, &satellite_rotation);
            plot_ui.line(
                Line::new("", PlotPoints::new(vec![[wn_x, wn_y], [ep_x, ep_y]]))
                    .color(egui::Color32::from_rgb(255, 100, 100))
                    .width(1.5),
            );

            let (np_x, np_y, _) = rotate_point_matrix(0.0, axis_len, 0.0, &satellite_rotation);
            let (sn_x, sn_y, _) = rotate_point_matrix(0.0, -axis_len, 0.0, &satellite_rotation);
            plot_ui.line(
                Line::new("", PlotPoints::new(vec![[sn_x, sn_y], [np_x, np_y]]))
                    .color(egui::Color32::from_rgb(100, 100, 255))
                    .width(1.5),
            );

            let label_offset = axis_len * 1.15;
            let (n_x, n_y, _) = rotate_point_matrix(0.0, label_offset, 0.0, &satellite_rotation);
            let (s_x, s_y, _) = rotate_point_matrix(0.0, -label_offset, 0.0, &satellite_rotation);
            let (e_x, e_y, _) = rotate_point_matrix(label_offset, 0.0, 0.0, &satellite_rotation);
            let (w_x, w_y, _) = rotate_point_matrix(-label_offset, 0.0, 0.0, &satellite_rotation);

            plot_ui.text(Text::new("", PlotPoint::new(n_x, n_y), "N").color(egui::Color32::WHITE));
            plot_ui.text(Text::new("", PlotPoint::new(s_x, s_y), "S").color(egui::Color32::WHITE));
            plot_ui.text(Text::new("", PlotPoint::new(e_x, e_y), "E").color(egui::Color32::WHITE));
            plot_ui.text(Text::new("", PlotPoint::new(w_x, w_y), "W").color(egui::Color32::WHITE));
        }

        if show_orbits {
            for (constellation, _, color_offset, tle_kind, _, _) in constellations {
                if *tle_kind != 0 { continue; }
                for plane in 0..constellation.num_planes {
                    let orbit_pts = constellation.orbit_points_3d(plane, time);
                    let color = if show_routing_paths || show_asc_desc_colors {
                        egui::Color32::from_rgb(80, 80, 80)
                    } else {
                        plane_color(if single_color { *color_offset } else { plane + color_offset })
                    };

                    let mut front_segment: Vec<[f64; 2]> = Vec::new();
                    for &(x, y, z) in &orbit_pts {
                        let (rx, ry, rz) = rotate_point_matrix(x, y, z, &satellite_rotation);
                        let visible = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;
                        if visible {
                            front_segment.push([rx, ry]);
                        } else if !front_segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut front_segment)))
                                    .color(color)
                                    .width(scaled_link_width),
                            );
                        }
                    }
                    if !front_segment.is_empty() {
                        plot_ui.line(
                            Line::new("", PlotPoints::new(front_segment))
                                .color(color)
                                .width(scaled_link_width),
                        );
                    }
                }
            }
        }

        if show_links {
            let base_link_color = if show_routing_paths || show_asc_desc_colors {
                egui::Color32::from_rgb(80, 80, 80)
            } else {
                egui::Color32::from_rgb(200, 200, 200)
            };
            let link_dim = egui::Color32::from_rgba_unmultiplied(50, 50, 60, 80);
            for (_, positions, _, _, _, _) in constellations {
                for sat in positions {
                    if let Some(neighbor_idx) = sat.neighbor_idx {
                        let neighbor = &positions[neighbor_idx];
                        let (rx1, ry1, rz1) = rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                        let (rx2, ry2, rz2) = rotate_point_matrix(neighbor.x, neighbor.y, neighbor.z, &satellite_rotation);
                        let visible1 = rz1 >= 0.0 || (rx1 * rx1 + ry1 * ry1) >= earth_r_sq;
                        let visible2 = rz2 >= 0.0 || (rx2 * rx2 + ry2 * ry2) >= earth_r_sq;
                        let both_visible = visible1 && visible2;
                        if hide_behind_earth && !both_visible {
                            continue;
                        }
                        let color = if both_visible { base_link_color } else { link_dim };
                        plot_ui.line(
                            Line::new("", PlotPoints::new(vec![[rx1, ry1], [rx2, ry2]]))
                                .color(color)
                                .width(scaled_link_width),
                        );
                    }
                }
            }
        }

        if show_intra_links {
            let base_link_color = if show_routing_paths || show_asc_desc_colors {
                egui::Color32::from_rgb(80, 80, 80)
            } else {
                egui::Color32::from_rgb(200, 200, 200)
            };
            let link_dim = egui::Color32::from_rgba_unmultiplied(50, 50, 60, 80);
            for (constellation, positions, _, _, _, _) in constellations {
                let sats_per_plane = constellation.sats_per_plane();
                for plane in 0..constellation.num_planes {
                    let plane_sats: Vec<_> = positions.iter()
                        .filter(|s| s.plane == plane)
                        .collect();
                    for i in 0..plane_sats.len() {
                        let sat = plane_sats[i];
                        let next = plane_sats[(i + 1) % sats_per_plane];
                        let (rx1, ry1, rz1) = rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                        let (rx2, ry2, rz2) = rotate_point_matrix(next.x, next.y, next.z, &satellite_rotation);
                        let visible1 = rz1 >= 0.0 || (rx1 * rx1 + ry1 * ry1) >= earth_r_sq;
                        let visible2 = rz2 >= 0.0 || (rx2 * rx2 + ry2 * ry2) >= earth_r_sq;
                        let both_visible = visible1 && visible2;
                        if hide_behind_earth && !both_visible {
                            continue;
                        }
                        let color = if both_visible { base_link_color } else { link_dim };
                        plot_ui.line(
                            Line::new("", PlotPoints::new(vec![[rx1, ry1], [rx2, ry2]]))
                                .color(color)
                                .width(scaled_link_width),
                        );
                    }
                }
            }
        }

        if show_routing_paths && !satellite_cameras.is_empty() {
            let manhattan_color = egui::Color32::from_rgb(255, 100, 100);
            let shortest_color = egui::Color32::from_rgb(100, 255, 100);

            for (cidx, (constellation, positions, _, _, _, _)) in constellations.iter().enumerate() {
                let tracked: Vec<_> = satellite_cameras.iter()
                    .filter(|c| c.constellation_idx == cidx)
                    .collect();

                if tracked.len() < 2 {
                    continue;
                }

                let num_planes = constellation.num_planes;
                let sats_per_plane = constellation.sats_per_plane();

                let is_star = constellation.walker_type == WalkerType::Star;

                for i in 0..tracked.len() {
                    for j in (i + 1)..tracked.len() {
                        let src = tracked[i];
                        let dst = tracked[j];

                        let src_sat = positions.iter().find(|s| s.plane == src.plane && s.sat_index == src.sat_index);
                        let dst_sat = positions.iter().find(|s| s.plane == dst.plane && s.sat_index == dst.sat_index);

                        let can_route = match (src_sat, dst_sat) {
                            (Some(_), Some(_)) => {
                                if is_star {
                                    let plane_diff_fwd = (dst.plane + num_planes - src.plane) % num_planes;
                                    let plane_diff_bwd = (src.plane + num_planes - dst.plane) % num_planes;
                                    let crosses_seam = plane_diff_fwd > num_planes / 2 && plane_diff_bwd > num_planes / 2;
                                    !crosses_seam
                                } else {
                                    true
                                }
                            }
                            _ => false,
                        };

                        if !can_route {
                            continue;
                        }

                        if show_manhattan_path {
                            let path = compute_manhattan_path(
                                src.plane, src.sat_index,
                                dst.plane, dst.sat_index,
                                num_planes, sats_per_plane,
                                is_star,
                            );
                            draw_routing_path(
                                plot_ui, &path, positions, &satellite_rotation,
                                manhattan_color, (scaled_link_width + 3.0).max(3.0), hide_behind_earth, earth_r_sq,
                            );
                        }

                        if show_shortest_path {
                            let path = compute_shortest_path(
                                src.plane, src.sat_index,
                                dst.plane, dst.sat_index,
                                num_planes, sats_per_plane,
                                positions,
                                is_star,
                            );
                            draw_routing_path(
                                plot_ui, &path, positions, &satellite_rotation,
                                shortest_color, (scaled_link_width + 3.0).max(3.0), hide_behind_earth, earth_r_sq,
                            );
                        }
                    }
                }
            }
        }

        for aoi in areas_of_interest {
            if let Some(gs_idx) = aoi.ground_station_idx {
                if let Some(gs) = ground_stations.get(gs_idx) {
                    let find_nearest_sat = |center_lat: f64, center_lon: f64, radius_km: f64, ascending_filter: Option<bool>|
                        -> Option<(usize, &WalkerConstellation, &Vec<SatelliteState>, &SatelliteState)>
                    {
                        let center_lat_rad = center_lat.to_radians();
                        let center_lon_rad = center_lon.to_radians() + body_rot_angle;
                        let max_angular_dist = radius_km / planet_radius;

                        let haversine_dist = |sat: &SatelliteState| -> f64 {
                            let sat_lat_rad = sat.lat.to_radians();
                            let sat_lon_rad = sat.lon.to_radians();
                            let dlat = sat_lat_rad - center_lat_rad;
                            let dlon = sat_lon_rad - center_lon_rad;
                            let a = (dlat / 2.0).sin().powi(2)
                                + center_lat_rad.cos() * sat_lat_rad.cos() * (dlon / 2.0).sin().powi(2);
                            2.0 * a.sqrt().asin()
                        };

                        let mut best: Option<(usize, &WalkerConstellation, &Vec<SatelliteState>, &SatelliteState, f64)> = None;

                        for (cidx, (cons, positions, _, tle_kind, _, _)) in constellations.iter().enumerate() {
                            if *tle_kind != 0 { continue; }
                            for sat in positions.iter() {
                                if let Some(asc) = ascending_filter {
                                    if sat.ascending != asc { continue; }
                                }
                                let dist = haversine_dist(sat);
                                if dist <= max_angular_dist && (best.is_none() || dist < best.as_ref().unwrap().4) {
                                    best = Some((cidx, cons, positions, sat, dist));
                                }
                            }
                        }

                        best.map(|(cidx, cons, positions, sat, _)| (cidx, cons, positions, sat))
                    };

                    let aoi_asc = find_nearest_sat(aoi.lat, aoi.lon, aoi.radius_km, Some(true));
                    let gs_asc = find_nearest_sat(gs.lat, gs.lon, gs.radius_km, Some(true));
                    let (aoi_result, gs_result) = if aoi_asc.is_some() && gs_asc.is_some() {
                        (aoi_asc, gs_asc)
                    } else {
                        let aoi_desc = find_nearest_sat(aoi.lat, aoi.lon, aoi.radius_km, Some(false));
                        let gs_desc = find_nearest_sat(gs.lat, gs.lon, gs.radius_km, Some(false));
                        (aoi_desc, gs_desc)
                    };

                    if let (Some((gs_cidx, gs_cons, gs_positions, gs_sat)),
                            Some((aoi_cidx, _, _, aoi_sat))) = (gs_result, aoi_result)
                    {
                        let path_color = egui::Color32::from_rgb(255, 255, 0);
                        let routing_width = (scaled_link_width + 3.0).max(3.0);

                        if gs_cidx == aoi_cidx {
                            let path = compute_shortest_path(
                                gs_sat.plane, gs_sat.sat_index,
                                aoi_sat.plane, aoi_sat.sat_index,
                                gs_cons.num_planes, gs_cons.sats_per_plane(),
                                gs_positions,
                                gs_cons.walker_type == WalkerType::Star,
                            );
                            draw_routing_path(
                                plot_ui, &path, gs_positions, &satellite_rotation,
                                path_color, routing_width, hide_behind_earth, earth_r_sq,
                            );
                        } else {
                            let (rx1, ry1, rz1) = rotate_point_matrix(gs_sat.x, gs_sat.y, gs_sat.z, &satellite_rotation);
                            let (rx2, ry2, rz2) = rotate_point_matrix(aoi_sat.x, aoi_sat.y, aoi_sat.z, &satellite_rotation);

                            let visible1 = rz1 >= 0.0 || (rx1 * rx1 + ry1 * ry1) >= earth_r_sq;
                            let visible2 = rz2 >= 0.0 || (rx2 * rx2 + ry2 * ry2) >= earth_r_sq;

                            if !hide_behind_earth || (visible1 && visible2) {
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(vec![[rx1, ry1], [rx2, ry2]]))
                                        .color(path_color)
                                        .width(routing_width),
                                );
                            }
                        }

                        let dot_size = scaled_sat_radius as f64 * 1.2;
                        let (rx1, ry1, _) = rotate_point_matrix(gs_sat.x, gs_sat.y, gs_sat.z, &satellite_rotation);
                        let (rx2, ry2, _) = rotate_point_matrix(aoi_sat.x, aoi_sat.y, aoi_sat.z, &satellite_rotation);
                        plot_ui.points(
                            Points::new("", PlotPoints::new(vec![[rx1, ry1], [rx2, ry2]]))
                                .radius(dot_size as f32)
                                .color(path_color),
                        );
                    }
                }
            }
        }

        for (constellation, positions, color_offset, tle_kind, orig_idx, _) in constellations {
            if *tle_kind != 0 {
                let is_debris = *tle_kind == 2;
                for sat in positions {
                    let color = plane_color(color_offset + sat.plane);
                    let dim_col = egui::Color32::from_rgba_unmultiplied(
                        color.r() / 2, color.g() / 2, color.b() / 2, 80,
                    );

                    let (rx, ry, rz) = rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                    let in_front = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;

                    let bg_color = if dark_mode {
                        egui::Color32::from_rgb(30, 30, 30)
                    } else {
                        egui::Color32::from_rgb(240, 240, 240)
                    };

                    if is_debris {
                        let d = scaled_sat_radius as f64 * 1.5 * margin / (width as f64 * 0.5);
                        let c = if in_front { color } else if !hide_behind_earth { dim_col } else { continue };
                        let w = if in_front { 1.0 } else { 0.5 };
                        plot_ui.line(
                            egui_plot::Line::new("", PlotPoints::new(vec![[rx - d, ry - d], [rx + d, ry + d]]))
                                .color(c).width(w),
                        );
                        plot_ui.line(
                            egui_plot::Line::new("", PlotPoints::new(vec![[rx - d, ry + d], [rx + d, ry - d]]))
                                .color(c).width(w),
                        );
                    } else {
                        if !hide_behind_earth && !in_front {
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(dim_col)
                                    .radius(scaled_sat_radius * 0.8)
                                    .filled(true),
                            );
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(bg_color)
                                    .radius(scaled_sat_radius * 0.4)
                                    .filled(true),
                            );
                        }
                        if in_front {
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(color)
                                    .radius(scaled_sat_radius)
                                    .filled(true),
                            );
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(bg_color)
                                    .radius(scaled_sat_radius * 0.5)
                                    .filled(true),
                            );
                        }
                    }
                }
                continue;
            }
            for plane in 0..constellation.num_planes {
                let base_color = plane_color(if single_color { *color_offset } else { plane + color_offset });

                for sat in positions.iter().filter(|s| s.plane == plane) {
                    let is_tracked = satellite_cameras.iter().any(|c|
                        c.constellation_idx == *orig_idx && c.plane == sat.plane && c.sat_index == sat.sat_index
                    );
                    let color = if show_asc_desc_colors {
                        if is_tracked {
                            if sat.ascending { COLOR_ASCENDING } else { COLOR_DESCENDING }
                        } else if sat.ascending {
                            egui::Color32::from_rgb(180, 140, 80)
                        } else {
                            egui::Color32::from_rgb(80, 120, 180)
                        }
                    } else {
                        base_color
                    };
                    let dim_col = egui::Color32::from_rgba_unmultiplied(
                        color.r() / 2, color.g() / 2, color.b() / 2, 80,
                    );

                    let (rx, ry, rz) = rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                    let in_front = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;

                    let bg_color = if dark_mode {
                        egui::Color32::from_rgb(30, 30, 30)
                    } else {
                        egui::Color32::from_rgb(240, 240, 240)
                    };

                    if !hide_behind_earth && !in_front {
                        plot_ui.points(
                            Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                .color(dim_col)
                                .radius(scaled_sat_radius * 0.8)
                                .filled(true),
                        );
                        if *tle_kind != 0 {
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(bg_color)
                                    .radius(scaled_sat_radius * 0.4)
                                    .filled(true),
                            );
                        }
                    }
                    if in_front {
                        plot_ui.points(
                            Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                .color(color)
                                .radius(scaled_sat_radius)
                                .filled(true),
                        );
                        if *tle_kind != 0 {
                            plot_ui.points(
                                Points::new("", PlotPoints::new(vec![[rx, ry]]))
                                    .color(bg_color)
                                    .radius(scaled_sat_radius * 0.5)
                                    .filled(true),
                            );
                        }
                    }

                    if constellation.altitude_km < 100.0 && (in_front || !hide_behind_earth) {
                        let d = scaled_sat_radius as f64 * 3.0 * margin / (width as f64 * 0.5);
                        let red = egui::Color32::from_rgb(255, 60, 60);
                        plot_ui.line(
                            egui_plot::Line::new("", PlotPoints::new(vec![[rx - d, ry - d], [rx + d, ry + d]]))
                                .color(red)
                                .width(2.0 * zoom_factor),
                        );
                        plot_ui.line(
                            egui_plot::Line::new("", PlotPoints::new(vec![[rx - d, ry + d], [rx + d, ry - d]]))
                                .color(red)
                                .width(2.0 * zoom_factor),
                        );
                    }

                    if show_altitude_lines && (in_front || !hide_behind_earth) {
                        let r = (sat.x * sat.x + sat.y * sat.y + sat.z * sat.z).sqrt();
                        let scale = planet_radius / r;
                        let (gx, gy, _) = rotate_point_matrix(sat.x * scale, sat.y * scale, sat.z * scale, &satellite_rotation);
                        let alt_color = egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 100);
                        plot_ui.line(
                            egui_plot::Line::new("", PlotPoints::new(vec![[rx, ry], [gx, gy]]))
                                .color(alt_color)
                                .width(0.5 * zoom_factor),
                        );
                    }
                }
            }
        }
    });

    let label_font_size = (14.0 * zoom as f32).clamp(10.0, 28.0);
    let mut label_rects: Vec<(egui::Rect, bool, usize)> = Vec::new();
    for (pos, name, color, is_gs, idx) in &surface_labels {
        let plot_pt = egui_plot::PlotPoint::new(pos[0], pos[1]);
        let screen_pos = response.transform.position_from_point(&plot_pt);
        let galley = ui.painter().layout_no_wrap(
            name.clone(),
            egui::FontId::proportional(label_font_size),
            *color,
        );
        let text_pos = screen_pos + egui::Vec2::new(-(galley.size().x / 2.0), -galley.size().y - 4.0);
        let bg_rect = egui::Rect::from_min_size(text_pos, galley.size()).expand(3.0);
        ui.painter().rect_filled(bg_rect, 3.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180));
        ui.painter().galley(text_pos, galley, *color);
        label_rects.push((bg_rect, *is_gs, *idx));
    }

    for (pos, count, color) in &device_cluster_labels {
        let plot_pt = egui_plot::PlotPoint::new(pos[0], pos[1]);
        let screen_pos = response.transform.position_from_point(&plot_pt);
        let text = format!("{}", count);
        let galley = ui.painter().layout_no_wrap(
            text,
            egui::FontId::proportional(11.0),
            egui::Color32::WHITE,
        );
        let text_pos = screen_pos + egui::Vec2::new(-(galley.size().x / 2.0), -(galley.size().y / 2.0));
        let bg_rect = egui::Rect::from_min_size(text_pos, galley.size()).expand(2.0);
        ui.painter().rect_filled(bg_rect, 3.0, egui::Color32::from_rgba_unmultiplied(
            color.r() / 3, color.g() / 3, color.b() / 3, 200,
        ));
        ui.painter().galley(text_pos, galley, egui::Color32::WHITE);
    }

    if constellations.len() > 1 {
        let plot_rect = response.response.rect;
        let mut y = plot_rect.min.y + 8.0;
        let x = plot_rect.min.x + 8.0;
        let font = egui::FontId::proportional(12.0);
        let square_size = 10.0;
        let mut seen = std::collections::HashSet::new();
        for (_, _, color_offset, tle_kind, _, name) in constellations {
            if !seen.insert((name.as_str(), *color_offset)) { continue; }
            let color = plane_color(*color_offset);
            let square_rect = egui::Rect::from_min_size(
                egui::pos2(x, y + 1.0),
                egui::vec2(square_size, square_size),
            );
            ui.painter().rect_filled(square_rect, 2.0, color);
            let galley = ui.painter().layout_no_wrap(
                name.clone(),
                font.clone(),
                egui::Color32::WHITE,
            );
            let text_pos = egui::pos2(x + square_size + 4.0, y - 1.0);
            let bg_rect = egui::Rect::from_min_max(
                egui::pos2(x - 2.0, y - 2.0),
                egui::pos2(text_pos.x + galley.size().x + 2.0, y + galley.size().y + 2.0),
            );
            ui.painter().rect_filled(bg_rect, 3.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 160));
            ui.painter().rect_filled(square_rect, 2.0, color);
            if *tle_kind == 1 {
                let inset = 2.5;
                let inner = square_rect.shrink(inset);
                ui.painter().rect_filled(inner, 1.0, egui::Color32::BLACK);
            } else if *tle_kind == 2 {
                let c = square_rect.center();
                let h = square_size * 0.35;
                ui.painter().line_segment(
                    [c - egui::vec2(h, h), c + egui::vec2(h, h)],
                    egui::Stroke::new(1.5, egui::Color32::BLACK),
                );
                ui.painter().line_segment(
                    [c + egui::vec2(-h, h), c + egui::vec2(h, -h)],
                    egui::Stroke::new(1.5, egui::Color32::BLACK),
                );
            }
            ui.painter().galley(text_pos, galley, egui::Color32::WHITE);
            y += 16.0;
        }
    }

    for (constellation, positions, color_offset, _tle_kind, orig_idx, _) in constellations {
        for sat in positions {
            for cam in satellite_cameras.iter_mut() {
                if cam.constellation_idx == *orig_idx && cam.plane == sat.plane && cam.sat_index == sat.sat_index {
                    let (rx, ry, _) = rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                    let plot_pt = egui_plot::PlotPoint::new(rx, ry);
                    let screen_pos = response.transform.position_from_point(&plot_pt);
                    cam.screen_pos = Some(screen_pos);

                    let color = plane_color(if single_color { *color_offset } else { sat.plane + color_offset });
                    ui.painter().circle_stroke(
                        screen_pos,
                        scaled_sat_radius * 2.5,
                        egui::Stroke::new(2.0, color),
                    );

                    let r = (sat.x * sat.x + sat.y * sat.y + sat.z * sat.z).sqrt();
                    let alt_km = r - constellation.planet_radius;
                    let vel_km_s = (constellation.planet_mu / r).sqrt();
                    let inv_body = body_rotation.transpose();
                    let body_pos = inv_body * Vector3::new(sat.x, sat.y, sat.z);
                    let ground_lat = (body_pos.y / r).asin().to_degrees();
                    let ground_lon = (-body_pos.z).atan2(body_pos.x).to_degrees();
                    let id = match &sat.name {
                        Some(name) => name.clone(),
                        None => format!("P{}S{}", sat.plane, sat.sat_index),
                    };
                    let text = if let (Some(inc), Some(mm)) = (sat.tle_inclination_deg, sat.tle_mean_motion) {
                        let revs_per_day = mm;
                        let period_min = 1440.0 / revs_per_day;
                        format!(
                            "{}  {:.1}° {:.1}°\n{:.0} km  {:.2} km/s\nInc {:.1}°  {:.2} rev/day\nPeriod {:.1} min",
                            id,
                            ground_lat, ground_lon,
                            alt_km, vel_km_s,
                            inc, revs_per_day,
                            period_min,
                        )
                    } else {
                        format!(
                            "{}  {:.1}° {:.1}°\n{:.0} km  {:.2} km/s",
                            id,
                            ground_lat, ground_lon,
                            alt_km, vel_km_s,
                        )
                    };
                    let font = egui::FontId::proportional(12.0);
                    let galley = ui.painter().layout_no_wrap(text, font, egui::Color32::WHITE);
                    let text_pos = screen_pos + egui::Vec2::new(scaled_sat_radius * 3.0, -galley.size().y / 2.0);
                    let bg_rect = egui::Rect::from_min_size(text_pos, galley.size()).expand(4.0);
                    ui.painter().rect_filled(bg_rect, 4.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 200));
                    ui.painter().galley(text_pos, galley, egui::Color32::WHITE);
                }
            }
        }
    }

    let mut hovering_satellite = false;
    if let Some(hover_pos) = response.response.hover_pos() {
        let plot_pos = response.transform.value_from_position(hover_pos);
        let hover_threshold = margin * 0.025;

        'hover: for (_constellation, positions, color_offset, _, _, _) in constellations {
            for sat in positions {
                let (rx, ry, rz) = rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                let earth_r_sq = (planet_radius * EARTH_VISUAL_SCALE).powi(2);
                let visible = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;
                if !visible && hide_behind_earth {
                    continue;
                }
                let dx = rx - plot_pos.x;
                let dy = ry - plot_pos.y;
                if dx * dx + dy * dy < hover_threshold * hover_threshold {
                    let plot_pt = egui_plot::PlotPoint::new(rx, ry);
                    let screen_pt = response.transform.position_from_point(&plot_pt);
                    let color = plane_color(if single_color { *color_offset } else { sat.plane + color_offset });
                    ui.painter().circle_stroke(
                        screen_pt,
                        scaled_sat_radius * 2.0,
                        egui::Stroke::new(2.0, color),
                    );
                    hovering_satellite = true;
                    break 'hover;
                }
            }
        }

        let px = plot_pos.x;
        let py = plot_pos.y;
        let r_sq = planet_radius * planet_radius;
        if px * px + py * py <= r_sq {
            let pz = (r_sq - px * px - py * py).sqrt();
            let surface_rot = if earth_fixed_camera {
                rotation
            } else {
                rotation * *body_rotation
            };
            let inv = surface_rot.transpose();
            let orig = inv * Vector3::new(px, py, pz);
            let lat = (orig.y / planet_radius).asin().to_degrees();
            let lon = -(orig.z.atan2(orig.x)).to_degrees();
            let text = format!("{:.1}° {:.1}°", lat, lon);
            let font = egui::FontId::proportional(12.0);
            let text_pos = hover_pos + egui::Vec2::new(15.0, -15.0);
            let galley = ui.painter().layout_no_wrap(text, font, egui::Color32::WHITE);
            let rect = egui::Rect::from_min_size(
                text_pos - egui::Vec2::new(0.0, galley.size().y),
                galley.size(),
            ).expand(3.0);
            ui.painter().rect_filled(rect, 3.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180));
            ui.painter().galley(text_pos - egui::Vec2::new(0.0, galley.size().y), galley, egui::Color32::WHITE);
        }
    }

    if response.response.is_pointer_button_down_on() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
    } else if !hovering_satellite {
        if let Some(hover_pos) = response.response.hover_pos() {
            let plot_pos = response.transform.value_from_position(hover_pos);
            let on_label = label_rects.iter().any(|(rect, _, _)| rect.contains(hover_pos));
            let on_earth = plot_pos.x * plot_pos.x + plot_pos.y * plot_pos.y <= planet_radius * planet_radius;
            if on_label || on_earth {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }
        }
    }

    if response.response.drag_started() {
        if let Some(pos) = response.response.interact_pointer_pos() {
            let mut found = false;
            for (rect, is_gs, idx) in &label_rects {
                if rect.contains(pos) {
                    *dragging_place = Some((drag_tab_planet.0, drag_tab_planet.1, *is_gs, *idx));
                    found = true;
                    break;
                }
            }
            if !found {
                *dragging_place = None;
            }
        }
    }

    let is_dragging_place = dragging_place.map_or(false, |(t, p, _, _)| t == drag_tab_planet.0 && p == drag_tab_planet.1);

    if response.response.dragged() && !response.response.drag_started() {
        if is_dragging_place {
            if let Some(pos) = response.response.interact_pointer_pos() {
                let plot_pos = response.transform.value_from_position(pos);
                let px = plot_pos.x;
                let py = plot_pos.y;
                let r_sq = planet_radius * planet_radius;
                if px * px + py * py <= r_sq {
                    let pz = (r_sq - px * px - py * py).sqrt();
                    let surface_rot = if earth_fixed_camera {
                        rotation
                    } else {
                        rotation * *body_rotation
                    };
                    let inv = surface_rot.transpose();
                    let orig = inv * Vector3::new(px, py, pz);
                    let lat = (orig.y / planet_radius).asin().to_degrees();
                    let lon = -(orig.z.atan2(orig.x)).to_degrees();
                    if let Some((_, _, is_gs, idx)) = *dragging_place {
                        if is_gs {
                            if let Some(gs) = ground_stations.get_mut(idx) {
                                gs.lat = lat;
                                gs.lon = lon;
                            }
                        } else if let Some(aoi) = areas_of_interest.get_mut(idx) {
                            aoi.lat = lat;
                            aoi.lon = lon;
                        }
                    }
                }
            }
        } else if let Some(pos) = response.response.interact_pointer_pos() {
            let drag = response.response.drag_delta();
            let prev_pos = pos - drag;
            let cur = response.transform.value_from_position(pos);
            let prev = response.transform.value_from_position(prev_pos);
            let r = planet_radius;
            let r_sq = r * r;
            let to_sphere = |px: f64, py: f64| -> Vector3<f64> {
                let d_sq = px * px + py * py;
                if d_sq <= r_sq {
                    Vector3::new(px, py, (r_sq - d_sq).sqrt())
                } else {
                    let s = r / d_sq.sqrt();
                    Vector3::new(px * s, py * s, 0.0)
                }
            };
            let a = to_sphere(prev.x, prev.y).normalize();
            let b = to_sphere(cur.x, cur.y).normalize();
            let cross = a.cross(&b);
            let cross_len = cross.norm();
            if cross_len > 1e-12 {
                let axis = cross / cross_len;
                let angle = cross_len.atan2(a.dot(&b));
                let c = angle.cos();
                let s = angle.sin();
                let t = 1.0 - c;
                let (x, y, z) = (axis.x, axis.y, axis.z);
                let rot = Matrix3::new(
                    t*x*x + c,   t*x*y - s*z, t*x*z + s*y,
                    t*x*y + s*z, t*y*y + c,   t*y*z - s*x,
                    t*x*z - s*y, t*y*z + s*x, t*z*z + c,
                );
                rotation = rot * rotation;
            }

        }
    }

    if !response.response.dragged() && dragging_place.is_some_and(|(t, p, _, _)| t == drag_tab_planet.0 && p == drag_tab_planet.1) {
        *dragging_place = None;
    }

    if response.response.clicked() {
        if let Some(pos) = response.response.interact_pointer_pos() {
            let plot_pos = response.transform.value_from_position(pos);
            let click_x = plot_pos.x;
            let click_y = plot_pos.y;
            let click_threshold = margin * 0.03;

            'outer: for (_constellation, positions, _color_offset, _, orig_idx, _) in constellations {
                for sat in positions {
                    let (rx, ry, rz) = rotate_point_matrix(sat.x, sat.y, sat.z, &satellite_rotation);
                    let earth_r_sq = (planet_radius * EARTH_VISUAL_SCALE).powi(2);
                    let visible = rz >= 0.0 || (rx * rx + ry * ry) >= earth_r_sq;
                    if !visible && hide_behind_earth {
                        continue;
                    }
                    let dx = rx - click_x;
                    let dy = ry - click_y;
                    if dx * dx + dy * dy < click_threshold * click_threshold {
                        let existing = satellite_cameras.iter().find(|c|
                            c.constellation_idx == *orig_idx && c.plane == sat.plane && c.sat_index == sat.sat_index
                        );
                        if let Some(cam) = existing {
                            cameras_to_remove.push(cam.id);
                        } else {
                            let in_pending = pending_cameras.iter().any(|c|
                                c.constellation_idx == *orig_idx && c.plane == sat.plane && c.sat_index == sat.sat_index
                            );
                            if !in_pending {
                                *camera_id_counter += 1;
                                pending_cameras.push(SatelliteCamera {
                                    id: *camera_id_counter,
                                    label: format!("Sat {}-{}", sat.plane + 1, sat.sat_index + 1),
                                    constellation_idx: *orig_idx,
                                    plane: sat.plane,
                                    sat_index: sat.sat_index,
                                    screen_pos: None,
                                });
                            }
                        }
                        break 'outer;
                    }
                }
            }
        }
    }

    if response.response.hovered() {
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0 {
            let old_zoom = zoom;
            let factor = 1.0 + scroll as f64 * 0.001;
            zoom = (zoom * factor).clamp(0.01, 20000.0);

            if let Some(hover_pos) = response.response.hover_pos() {
                let plot_pos = response.transform.value_from_position(hover_pos);
                let cx = plot_pos.x;
                let cy = plot_pos.y;
                let r_sq = planet_radius * planet_radius;
                if cx * cx + cy * cy <= r_sq {
                    let ratio = old_zoom / zoom;
                    let tx = cx * ratio;
                    let ty = cy * ratio;
                    if tx * tx + ty * ty <= r_sq {
                        let a = Vector3::new(cx, cy, (r_sq - cx*cx - cy*cy).sqrt()).normalize();
                        let b = Vector3::new(tx, ty, (r_sq - tx*tx - ty*ty).sqrt()).normalize();
                        let cross = a.cross(&b);
                        let cross_len = cross.norm();
                        if cross_len > 1e-12 {
                            let axis = cross / cross_len;
                            let angle = cross_len.atan2(a.dot(&b));
                            let ca = angle.cos();
                            let sa = angle.sin();
                            let t = 1.0 - ca;
                            let (x, y, z) = (axis.x, axis.y, axis.z);
                            let rot = Matrix3::new(
                                t*x*x+ca,    t*x*y-sa*z, t*x*z+sa*y,
                                t*x*y+sa*z,  t*y*y+ca,   t*y*z-sa*x,
                                t*x*z-sa*y,  t*y*z+sa*x, t*z*z+ca,
                            );
                            rotation = rot * rotation;
                        }
                    }

                    let center = rotation.transpose() * Vector3::new(0.0, 0.0, 1.0);
                    let lat_limit = 85.0_f64.to_radians().sin();
                    let clamped_y = center.y.clamp(-lat_limit, lat_limit);
                    let needs_clamp = (center.y - clamped_y).abs() > 1e-8;
                    if needs_clamp {
                        let horiz = (center.x * center.x + center.z * center.z).sqrt();
                        let new_horiz = (1.0 - clamped_y * clamped_y).sqrt();
                        let scale = if horiz > 1e-10 { new_horiz / horiz } else { 1.0 };
                        let clamped = Vector3::new(center.x * scale, clamped_y, center.z * scale).normalize();
                        let right_raw = Vector3::new(clamped.z, 0.0, -clamped.x);
                        let right_len = right_raw.norm();
                        if right_len > 0.01 {
                            let right = right_raw / right_len;
                            let up = clamped.cross(&right);
                            let r0 = Matrix3::new(
                                right.x, right.y, right.z,
                                up.x, up.y, up.z,
                                clamped.x, clamped.y, clamped.z,
                            );
                            let up_screen = rotation * Vector3::new(0.0, 1.0, 0.0);
                            let bearing = up_screen.x.atan2(up_screen.y);
                            let cb = bearing.cos();
                            let sb = bearing.sin();
                            let rz = Matrix3::new(
                                 cb, sb, 0.0,
                                -sb, cb, 0.0,
                                0.0, 0.0, 1.0,
                            );
                            rotation = rz * r0;
                        }
                    }
                    let north_blend = (zoom.log2() / 4.0).clamp(0.0, 1.0);
                    if north_blend > 0.0 {
                        let up_screen = rotation * Vector3::new(0.0, 1.0, 0.0);
                        let bearing = up_screen.x.atan2(up_screen.y);
                        let zoom_octaves = (zoom / old_zoom).ln().abs() / (2.0_f64).ln();
                        let decay = (-north_blend * zoom_octaves * 1.5).exp();
                        let correction = bearing * (decay - 1.0);
                        let ca = correction.cos();
                        let sa = correction.sin();
                        rotation = Matrix3::new(
                            ca, sa, 0.0,
                            -sa, ca, 0.0,
                            0.0, 0.0, 1.0,
                        ) * rotation;
                    }
                }
            }
        }
        if let Some(touch) = ui.input(|i| i.multi_touch()) {
            let factor = touch.zoom_delta as f64;
            zoom = (zoom * factor).clamp(0.01, 20000.0);
        }
    }

    (rotation, zoom)
}

fn draw_ground_track(
    ui: &mut egui::Ui,
    id: &str,
    constellations: &[(WalkerConstellation, Vec<SatelliteState>, usize, u8, usize, String)],
    width: f32,
    height: f32,
    sat_radius: f32,
    single_color: bool,
) {
    let plot = Plot::new(id)
        .width(width)
        .height(height)
        .include_x(-180.0)
        .include_x(180.0)
        .include_y(-90.0)
        .include_y(90.0)
        .show_axes([true, true]);

    plot.show(ui, |plot_ui| {
        for (constellation, positions, color_offset, _tle_kind, _, _) in constellations {
            for plane in 0..constellation.num_planes {
                let color = plane_color(if single_color { *color_offset } else { plane + color_offset });
                let pts: PlotPoints = positions
                    .iter()
                    .filter(|s| s.plane == plane)
                    .map(|s| [s.lon, s.lat])
                    .collect();
                plot_ui.points(
                    Points::new("", pts)
                        .color(color)
                        .radius(sat_radius)
                        .filled(true),
                );
            }
        }

        plot_ui.line(
            Line::new("", PlotPoints::new(vec![[-180.0, 0.0], [180.0, 0.0]]))
                .color(egui::Color32::DARK_GRAY)
                .width(0.5),
        );
        plot_ui.line(
            Line::new("", PlotPoints::new(vec![[0.0, -90.0], [0.0, 90.0]]))
                .color(egui::Color32::DARK_GRAY)
                .width(0.5),
        );
    });
}

fn draw_torus(
    ui: &mut egui::Ui,
    id: &str,
    constellations: &[(WalkerConstellation, Vec<SatelliteState>, usize, u8, usize, String)],
    time: f64,
    rotation: Matrix3<f64>,
    width: f32,
    height: f32,
    sat_radius: f32,
    show_links: bool,
    single_color: bool,
    mut zoom: f64,
    satellite_cameras: &mut [SatelliteCamera],
    show_routing_paths: bool,
    show_manhattan_path: bool,
    show_shortest_path: bool,
    show_asc_desc_colors: bool,
    planet_radius: f64,
    pending_cameras: &mut Vec<SatelliteCamera>,
    camera_id_counter: &mut usize,
    cameras_to_remove: &mut Vec<usize>,
    link_width: f32,
    fixed_sizes: bool,
) -> (Matrix3<f64>, f64) {
    let (major_radius, minor_radius) = if let Some((constellation, _, _, _, _, _)) = constellations.first() {
        let inclination_rad = constellation.inclination_deg.to_radians();
        let cos_i = inclination_rad.cos().abs();
        let major = 1.0;
        let minor = (major * (1.0 - cos_i) / (1.0 + cos_i)).max(0.002);
        (major, minor)
    } else {
        (1.0, 0.8)
    };

    let margin = (major_radius + minor_radius) * 1.3 / zoom;
    let zoom_factor = if fixed_sizes { 1.0 } else { zoom as f32 };
    let scaled_sat_radius = sat_radius * zoom_factor;
    let scaled_link_width = (link_width * zoom_factor).max(0.5);

    let mut user_rotation = rotation;
    let display_rotation = rotation;

    let plot = Plot::new(id)
        .data_aspect(1.0)
        .width(width)
        .height(height)
        .show_axes(false)
        .show_grid(false)
        .show_x(false)
        .show_y(false)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .cursor_color(egui::Color32::TRANSPARENT);

    let response = plot.show(ui, |plot_ui| {
        plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
            [-margin, -margin],
            [margin, margin],
        ));

        let is_facing_camera = |theta: f64, phi: f64| -> bool {
            let nx = phi.cos() * theta.cos();
            let ny = phi.sin();
            let nz = phi.cos() * theta.sin();
            let (_, _, nz_rot) = rotate_point_matrix(nx, ny, nz, &rotation);
            nz_rot >= 0.0
        };

        let torus_point = |theta: f64, phi: f64, ecc: f64, omega: f64| -> (f64, f64, f64) {
            let r_orbit = if ecc > 0.001 {
                minor_radius * (1.0 - ecc * ecc) / (1.0 + ecc * phi.cos())
            } else {
                minor_radius
            };
            let angle = phi + omega;
            let r = major_radius + r_orbit * angle.cos();
            let y = r_orbit * angle.sin();
            let x = r * theta.cos();
            let z = r * theta.sin();
            rotate_point_matrix(x, y, z, &display_rotation)
        };

        if let Some((constellation, _, _, _, _, _)) = constellations.first() {
            let orbit_radius = constellation.planet_radius + constellation.altitude_km;
            let earth_scale = planet_radius / orbit_radius;
            let earth_major = major_radius * earth_scale;
            let earth_minor = minor_radius * earth_scale;
            let earth_color = egui::Color32::from_rgb(60, 100, 140);
            let earth_dim = egui::Color32::from_rgba_unmultiplied(40, 70, 100, 150);

            let earth_point = |theta: f64, phi: f64| -> (f64, f64, f64) {
                let r = earth_major + earth_minor * phi.cos();
                let y = earth_minor * phi.sin();
                let x = r * theta.cos();
                let z = r * theta.sin();
                rotate_point_matrix(x, y, z, &display_rotation)
            };

            let earth_facing = |theta: f64, phi: f64| -> bool {
                let nx = phi.cos() * theta.cos();
                let ny = phi.sin();
                let nz = phi.cos() * theta.sin();
                let (_, _, nz_rot) = rotate_point_matrix(nx, ny, nz, &rotation);
                nz_rot >= 0.0
            };

            for ring in 0..12 {
                let phi = 2.0 * PI * ring as f64 / 12.0;
                let mut front_segment: Vec<[f64; 2]> = Vec::new();
                let mut back_segment: Vec<[f64; 2]> = Vec::new();
                for i in 0..=50 {
                    let theta = 2.0 * PI * i as f64 / 50.0;
                    let (rx, ry, _) = earth_point(theta, phi);
                    let facing = earth_facing(theta, phi);
                    if facing {
                        front_segment.push([rx, ry]);
                        if !back_segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut back_segment)))
                                    .color(earth_dim).width(1.0),
                            );
                        }
                    } else {
                        back_segment.push([rx, ry]);
                        if !front_segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut front_segment)))
                                    .color(earth_color).width(1.5),
                            );
                        }
                    }
                }
                if !front_segment.is_empty() {
                    plot_ui.line(Line::new("", PlotPoints::new(front_segment)).color(earth_color).width(1.5));
                }
                if !back_segment.is_empty() {
                    plot_ui.line(Line::new("", PlotPoints::new(back_segment)).color(earth_dim).width(1.0));
                }
            }

            for ring in 0..16 {
                let theta = 2.0 * PI * ring as f64 / 16.0;
                let mut front_segment: Vec<[f64; 2]> = Vec::new();
                let mut back_segment: Vec<[f64; 2]> = Vec::new();
                for i in 0..=50 {
                    let phi = 2.0 * PI * i as f64 / 50.0;
                    let (rx, ry, _) = earth_point(theta, phi);
                    let facing = earth_facing(theta, phi);
                    if facing {
                        front_segment.push([rx, ry]);
                        if !back_segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut back_segment)))
                                    .color(earth_dim).width(1.0),
                            );
                        }
                    } else {
                        back_segment.push([rx, ry]);
                        if !front_segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut front_segment)))
                                    .color(earth_color).width(1.5),
                            );
                        }
                    }
                }
                if !front_segment.is_empty() {
                    plot_ui.line(Line::new("", PlotPoints::new(front_segment)).color(earth_color).width(1.5));
                }
                if !back_segment.is_empty() {
                    plot_ui.line(Line::new("", PlotPoints::new(back_segment)).color(earth_dim).width(1.0));
                }
            }
        }

        for (constellation, positions, color_offset, _tle_kind, orig_idx, _) in constellations.iter() {
            let sats_per_plane = constellation.total_sats / constellation.num_planes;
            let orbit_radius = constellation.planet_radius + constellation.altitude_km;
            let period = 2.0 * PI * (orbit_radius.powi(3) / constellation.planet_mu).sqrt();
            let mean_motion = 2.0 * PI / period;
            let ecc = constellation.eccentricity;
            let omega = constellation.arg_periapsis_deg.to_radians();
            let raan_step = constellation.raan_step();
            let raan_offset = constellation.raan_offset_deg.to_radians();
            let plane_theta = |plane: usize| -> f64 {
                raan_offset + raan_step * plane as f64
            };

            let torus_pos = |plane: usize, sat_idx: usize| -> (f64, f64, f64) {
                let angle = plane_theta(plane);
                let sat_spacing = 2.0 * PI * sat_idx as f64 / sats_per_plane as f64;
                let phase = sat_spacing + mean_motion * time;
                torus_point(angle, phase, ecc, omega)
            };

            for plane in 0..constellation.num_planes {
                let angle = plane_theta(plane);
                let color = if show_routing_paths || show_asc_desc_colors {
                    egui::Color32::from_rgb(80, 80, 80)
                } else {
                    plane_color(if single_color { *color_offset } else { plane + color_offset })
                };
                let dim_col = egui::Color32::from_rgba_unmultiplied(
                    color.r(), color.g(), color.b(), 180,
                );

                let mut front_segment: Vec<[f64; 2]> = Vec::new();
                let mut back_segment: Vec<[f64; 2]> = Vec::new();

                for i in 0..=50 {
                    let phase = 2.0 * PI * i as f64 / 50.0;
                    let (rx, ry, _) = torus_point(angle, phase, ecc, omega);
                    let facing = is_facing_camera(angle, phase);

                    if facing {
                        front_segment.push([rx, ry]);
                        if !back_segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut back_segment)))
                                    .color(dim_col)
                                    .width(scaled_link_width),
                            );
                        }
                    } else {
                        back_segment.push([rx, ry]);
                        if !front_segment.is_empty() {
                            plot_ui.line(
                                Line::new("", PlotPoints::new(std::mem::take(&mut front_segment)))
                                    .color(color)
                                    .width(scaled_link_width * 1.5),
                            );
                        }
                    }
                }
                if !front_segment.is_empty() {
                    plot_ui.line(Line::new("", PlotPoints::new(front_segment)).color(color).width(scaled_link_width * 1.5));
                }
                if !back_segment.is_empty() {
                    plot_ui.line(Line::new("", PlotPoints::new(back_segment)).color(dim_col).width(scaled_link_width));
                }
            }

            if show_links {
                let base_link_color = if show_routing_paths || show_asc_desc_colors {
                    egui::Color32::from_rgb(80, 80, 80)
                } else {
                    egui::Color32::from_rgb(150, 150, 150)
                };
                let link_dim = egui::Color32::from_rgba_unmultiplied(50, 50, 60, 100);
                for sat in positions {
                    if let Some(neighbor_idx) = sat.neighbor_idx {
                        let neighbor = &positions[neighbor_idx];
                        let angle1 = plane_theta(sat.plane);
                        let angle2 = plane_theta(neighbor.plane);
                        let phase1 = 2.0 * PI * sat.sat_index as f64 / sats_per_plane as f64 + mean_motion * time;
                        let phase2 = 2.0 * PI * neighbor.sat_index as f64 / sats_per_plane as f64 + mean_motion * time;

                        let (x1, y1, _) = torus_pos(sat.plane, sat.sat_index);
                        let (x2, y2, _) = torus_pos(neighbor.plane, neighbor.sat_index);
                        let facing1 = is_facing_camera(angle1, phase1);
                        let facing2 = is_facing_camera(angle2, phase2);
                        let color = if facing1 && facing2 { base_link_color } else { link_dim };
                        plot_ui.line(
                            Line::new("", PlotPoints::new(vec![[x1, y1], [x2, y2]]))
                                .color(color)
                                .width(scaled_link_width),
                        );
                    }
                }
            }

            for plane in 0..constellation.num_planes {
                let base_color = plane_color(if single_color { *color_offset } else { plane + color_offset });
                let angle = plane_theta(plane);

                for sat in positions.iter().filter(|s| s.plane == plane) {
                    let is_tracked = satellite_cameras.iter().any(|c|
                        c.constellation_idx == *orig_idx && c.plane == sat.plane && c.sat_index == sat.sat_index
                    );
                    let color = if show_asc_desc_colors {
                        if is_tracked {
                            if sat.ascending { COLOR_ASCENDING } else { COLOR_DESCENDING }
                        } else if sat.ascending {
                            egui::Color32::from_rgb(180, 140, 80)
                        } else {
                            egui::Color32::from_rgb(80, 120, 180)
                        }
                    } else {
                        base_color
                    };
                    let dim_col = egui::Color32::from_rgba_unmultiplied(
                        color.r(), color.g(), color.b(), 140,
                    );

                    let phase = 2.0 * PI * sat.sat_index as f64 / sats_per_plane as f64 + mean_motion * time;
                    let (x, y, _) = torus_pos(sat.plane, sat.sat_index);
                    let facing = is_facing_camera(angle, phase);
                    let (c, r) = if facing { (color, scaled_sat_radius) } else { (dim_col, scaled_sat_radius * 0.8) };
                    plot_ui.points(
                        Points::new("", PlotPoints::new(vec![[x, y]]))
                            .color(c)
                            .radius(r)
                            .filled(true),
                    );

                    if is_tracked {
                        plot_ui.points(
                            Points::new("", PlotPoints::new(vec![[x, y]]))
                                .color(base_color)
                                .radius(scaled_sat_radius * 2.5)
                                .filled(false),
                        );
                    }
                }
            }

            if show_routing_paths {
                let tracked: Vec<_> = satellite_cameras.iter()
                    .filter(|c| c.constellation_idx == *orig_idx)
                    .collect();

                if tracked.len() >= 2 {
                    let manhattan_color = egui::Color32::from_rgb(255, 100, 100);
                    let shortest_color = egui::Color32::from_rgb(100, 255, 100);
                    let is_star = constellation.walker_type == WalkerType::Star;
                    let num_planes = constellation.num_planes;

                    for i in 0..tracked.len() {
                        for j in (i + 1)..tracked.len() {
                            let src = tracked[i];
                            let dst = tracked[j];

                            let src_sat = positions.iter().find(|s| s.plane == src.plane && s.sat_index == src.sat_index);
                            let dst_sat = positions.iter().find(|s| s.plane == dst.plane && s.sat_index == dst.sat_index);

                            let can_route = match (src_sat, dst_sat) {
                                (Some(_), Some(_)) => {
                                    if is_star {
                                        let plane_diff_fwd = (dst.plane + num_planes - src.plane) % num_planes;
                                        let plane_diff_bwd = (src.plane + num_planes - dst.plane) % num_planes;
                                        let crosses_seam = plane_diff_fwd > num_planes / 2 && plane_diff_bwd > num_planes / 2;
                                        !crosses_seam
                                    } else {
                                        true
                                    }
                                }
                                _ => false,
                            };

                            if !can_route {
                                continue;
                            }

                            if show_manhattan_path {
                                let path = compute_manhattan_path(
                                    src.plane, src.sat_index,
                                    dst.plane, dst.sat_index,
                                    num_planes, sats_per_plane,
                                    is_star,
                                );
                                for k in 0..(path.len() - 1) {
                                    let (p1, s1) = path[k];
                                    let (p2, s2) = path[k + 1];
                                    let (x1, y1, _) = torus_pos(p1, s1);
                                    let (x2, y2, _) = torus_pos(p2, s2);
                                    plot_ui.line(
                                        Line::new("", PlotPoints::new(vec![[x1, y1], [x2, y2]]))
                                            .color(manhattan_color)
                                            .width(2.5),
                                    );
                                }
                            }

                            if show_shortest_path {
                                let path = compute_shortest_path(
                                    src.plane, src.sat_index,
                                    dst.plane, dst.sat_index,
                                    num_planes, sats_per_plane,
                                    positions,
                                    is_star,
                                );
                                for k in 0..(path.len() - 1) {
                                    let (p1, s1) = path[k];
                                    let (p2, s2) = path[k + 1];
                                    let (x1, y1, _) = torus_pos(p1, s1);
                                    let (x2, y2, _) = torus_pos(p2, s2);
                                    plot_ui.line(
                                        Line::new("", PlotPoints::new(vec![[x1, y1], [x2, y2]]))
                                            .color(shortest_color)
                                            .width(2.0),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    });

    if response.response.dragged() && !response.response.drag_started() {
        let drag = response.response.drag_delta();
        let sens = 0.01 / zoom.max(1.0);
        let delta_rot = rotation_from_drag(drag.x as f64 * sens, drag.y as f64 * sens);
        user_rotation = delta_rot * user_rotation;
    }

    if response.response.hovered() {
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0 {
            let factor = 1.0 + scroll as f64 * 0.001;
            zoom = (zoom * factor).clamp(0.01, 20000.0);
        }
        if let Some(touch) = ui.input(|i| i.multi_touch()) {
            let factor = touch.zoom_delta as f64;
            zoom = (zoom * factor).clamp(0.01, 20000.0);
        }
    }

    if let Some(pos) = response.response.interact_pointer_pos() {
        if response.response.clicked() {
            let click_x = response.transform.value_from_position(pos).x;
            let click_y = response.transform.value_from_position(pos).y;
            let (major_radius, minor_radius) = if let Some((constellation, _, _, _, _, _)) = constellations.first() {
                let sats_per_plane = constellation.sats_per_plane();
                let orbit_radius = planet_radius + constellation.altitude_km;
                let inclination_rad = constellation.inclination_deg.to_radians();
                let inclination_factor = inclination_rad.sin().abs().max(0.002);
                let altitude_factor = orbit_radius / (planet_radius + 500.0);
                let major = altitude_factor * (sats_per_plane as f64 / constellation.num_planes as f64);
                let minor_base = altitude_factor * inclination_factor;
                let minor = minor_base.max(major * inclination_factor);
                let scale = 2.0 / (major + minor).max(1.0);
                (major * scale, minor * scale)
            } else {
                (2.0, 0.8)
            };
            let margin = (major_radius + minor_radius) * 1.3 / zoom;
            let click_threshold = margin * 0.05;

            let torus_point_click = |theta: f64, phi: f64, ecc: f64, omega: f64| -> (f64, f64, f64) {
                let r_orbit = if ecc > 0.001 {
                    minor_radius * (1.0 - ecc * ecc) / (1.0 + ecc * phi.cos())
                } else {
                    minor_radius
                };
                let angle = phi + omega;
                let r = major_radius + r_orbit * angle.cos();
                let y = r_orbit * angle.sin();
                let x = r * theta.cos();
                let z = r * theta.sin();
                rotate_point_matrix(x, y, z, &display_rotation)
            };

            'outer: for (constellation, positions, _, _, orig_idx, _) in constellations.iter() {
                let sats_per_plane = constellation.total_sats / constellation.num_planes;
                let orbit_radius = constellation.planet_radius + constellation.altitude_km;
                let period = 2.0 * PI * (orbit_radius.powi(3) / constellation.planet_mu).sqrt();
                let mean_motion = 2.0 * PI / period;
                let ecc = constellation.eccentricity;
                let omega = constellation.arg_periapsis_deg.to_radians();
                let raan_step = constellation.raan_step();
                let raan_offset = constellation.raan_offset_deg.to_radians();

                for sat in positions {
                    let angle = raan_offset + raan_step * sat.plane as f64;
                    let phase = 2.0 * PI * sat.sat_index as f64 / sats_per_plane as f64 + mean_motion * time;
                    let (tx, ty, _) = torus_point_click(angle, phase, ecc, omega);

                    let dx = tx - click_x;
                    let dy = ty - click_y;
                    if dx * dx + dy * dy < click_threshold * click_threshold {
                        let existing = satellite_cameras.iter().find(|c|
                            c.constellation_idx == *orig_idx && c.plane == sat.plane && c.sat_index == sat.sat_index
                        );
                        if let Some(cam) = existing {
                            cameras_to_remove.push(cam.id);
                        } else {
                            let in_pending = pending_cameras.iter().any(|c|
                                c.constellation_idx == *orig_idx && c.plane == sat.plane && c.sat_index == sat.sat_index
                            );
                            if !in_pending {
                                *camera_id_counter += 1;
                                pending_cameras.push(SatelliteCamera {
                                    id: *camera_id_counter,
                                    label: format!("Sat {}-{}", sat.plane + 1, sat.sat_index + 1),
                                    constellation_idx: *orig_idx,
                                    plane: sat.plane,
                                    sat_index: sat.sat_index,
                                    screen_pos: None,
                                });
                            }
                        }
                        break 'outer;
                    }
                }
            }
        }
    }

    (user_rotation, zoom)
}

fn plane_color(plane: usize) -> egui::Color32 {
    COLORS[plane % COLORS.len()]
}

const COLORS: [egui::Color32; 16] = [
    egui::Color32::from_rgb(255, 99, 71),
    egui::Color32::from_rgb(50, 205, 50),
    egui::Color32::from_rgb(30, 144, 255),
    egui::Color32::from_rgb(255, 215, 0),
    egui::Color32::from_rgb(238, 130, 238),
    egui::Color32::from_rgb(0, 206, 209),
    egui::Color32::from_rgb(255, 140, 0),
    egui::Color32::from_rgb(147, 112, 219),
    egui::Color32::from_rgb(0, 255, 127),
    egui::Color32::from_rgb(255, 105, 180),
    egui::Color32::from_rgb(100, 149, 237),
    egui::Color32::from_rgb(240, 230, 140),
    egui::Color32::from_rgb(60, 179, 113),
    egui::Color32::from_rgb(233, 150, 122),
    egui::Color32::from_rgb(186, 85, 211),
    egui::Color32::from_rgb(135, 206, 235),
];

fn dim_color(color: egui::Color32) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(
        (color.r() as f32 * 0.4) as u8,
        (color.g() as f32 * 0.4) as u8,
        (color.b() as f32 * 0.4) as u8,
        200,
    )
}

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
