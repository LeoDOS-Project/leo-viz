use eframe::{egui, egui_glow, glow};
use egui::mutex::Mutex;
use egui_dock::{DockArea, DockState, NodeIndex, SurfaceIndex, TabViewer};
use egui_dock::tab_viewer::OnCloseResponse;
use egui_plot::{Line, Plot, PlotImage, PlotPoints, PlotPoint, Points, Polygon, Text};
use nalgebra::{Matrix3, Vector3};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::sync::{Arc, mpsc};
use sgp4::Constants;
use chrono::{DateTime, Utc, Local, Duration};
use glow::HasContext as _;

fn asset_path(relative: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

#[cfg(not(target_arch = "wasm32"))]
fn dirs_cache() -> std::path::PathBuf {
    if let Some(dir) = dirs_sys_cache() {
        dir
    } else {
        std::path::PathBuf::from(".")
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn dirs_sys_cache() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache"))
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum CelestialBody {
    Earth,
    Moon,
    Mars,
    Mercury,
    Venus,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
    Sun,
    Ceres,
    Haumea,
    Makemake,
    Eris,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Skin {
    Default,
    HellOnEarth,
    Terraformed,
    Civilized,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum TextureResolution {
    R512,
    R1024,
    R2048,
    R8192,
    R21504,
}

impl TextureResolution {
    fn label(&self) -> &'static str {
        match self {
            TextureResolution::R512 => "512",
            TextureResolution::R1024 => "1K",
            TextureResolution::R2048 => "2K",
            TextureResolution::R8192 => "8K",
            TextureResolution::R21504 => "21K",
        }
    }

    fn downscale_factor(&self, body: CelestialBody, skin: Skin) -> u32 {
        match (body, skin, self) {
            (CelestialBody::Earth, Skin::Default, TextureResolution::R512) => 1,
            (_, _, TextureResolution::R512) => 4,
            (_, _, TextureResolution::R1024) => 2,
            _ => 1,
        }
    }

    fn cpu_render_size(&self) -> usize {
        match self {
            TextureResolution::R512 => 512,
            _ => 1024,
        }
    }

    fn cloud_filename(&self) -> Option<&'static str> {
        match self {
            TextureResolution::R8192 | TextureResolution::R21504 => Some("textures/earth/earth_clouds_8k.jpg"),
            _ => Some("textures/earth/earth_clouds_2k.jpg"),
        }
    }
}

impl Skin {
    fn label(&self) -> &'static str {
        match self {
            Skin::Default => "Default",
            Skin::HellOnEarth => "Hell on Earth",
            Skin::Terraformed => "Terraformed",
            Skin::Civilized => "Civilized",
        }
    }

    fn filename(&self, body: CelestialBody, resolution: TextureResolution) -> Option<&'static str> {
        match (body, self, resolution) {
            (CelestialBody::Earth, Skin::Default, TextureResolution::R21504) => Some("textures/earth/Earth_Diffuse_21k.jpg"),
            (CelestialBody::Earth, Skin::Default, TextureResolution::R8192) => Some("textures/earth/earth_8k.jpg"),
            (CelestialBody::Earth, Skin::Default, TextureResolution::R512) => Some("textures/earth/earth_512.jpg"),
            (CelestialBody::Earth, Skin::Default, _) => Some("textures/earth/earth_2k.jpg"),
            (CelestialBody::Earth, Skin::HellOnEarth, _) => Some("textures/earth/hell_on_earth_2k.png"),
            (CelestialBody::Moon, Skin::Default, TextureResolution::R8192) => Some("textures/moon/moon_8k.jpg"),
            (CelestialBody::Moon, Skin::Default, _) => Some("textures/moon/moon_2k.jpg"),
            (CelestialBody::Mars, Skin::Default, TextureResolution::R8192) => Some("textures/mars/mars_8k.jpg"),
            (CelestialBody::Mars, Skin::Default, _) => Some("textures/mars/mars_2k.jpg"),
            (CelestialBody::Mars, Skin::Terraformed, _) => Some("textures/mars/mars_terraformed.png"),
            (CelestialBody::Mars, Skin::Civilized, _) => Some("textures/mars/mars_civilized.png"),
            (CelestialBody::Mercury, Skin::Default, TextureResolution::R21504) => Some("textures/mercury/Mercury_Diffuse_16k.jpg"),
            (CelestialBody::Mercury, Skin::Default, TextureResolution::R8192) => Some("textures/mercury/mercury_8k.jpg"),
            (CelestialBody::Mercury, Skin::Default, _) => Some("textures/mercury/mercury_2k.jpg"),
            (CelestialBody::Venus, Skin::Default, TextureResolution::R21504) => Some("textures/venus/Venus_Diffuse_16k.jpg"),
            (CelestialBody::Venus, Skin::Default, TextureResolution::R8192) => Some("textures/venus/venus_8k.jpg"),
            (CelestialBody::Venus, Skin::Default, _) => Some("textures/venus/venus_2k.jpg"),
            (CelestialBody::Jupiter, Skin::Default, TextureResolution::R8192) => Some("textures/jupiter/jupiter_8k.jpg"),
            (CelestialBody::Jupiter, Skin::Default, _) => Some("textures/jupiter/jupiter_2k.jpg"),
            (CelestialBody::Saturn, Skin::Default, TextureResolution::R8192) => Some("textures/saturn/saturn_8k.jpg"),
            (CelestialBody::Saturn, Skin::Default, _) => Some("textures/saturn/saturn_2k.jpg"),
            (CelestialBody::Uranus, Skin::Default, _) => Some("textures/uranus/uranus_2k.jpg"),
            (CelestialBody::Neptune, Skin::Default, _) => Some("textures/neptune/neptune_2k.jpg"),
            (CelestialBody::Sun, Skin::Default, TextureResolution::R8192) => Some("textures/sun/sun_8k.jpg"),
            (CelestialBody::Sun, Skin::Default, _) => Some("textures/sun/sun_2k.jpg"),
            (CelestialBody::Ceres, Skin::Default, TextureResolution::R8192) => Some("textures/ceres/ceres_4k.jpg"),
            (CelestialBody::Ceres, Skin::Default, _) => Some("textures/ceres/ceres_2k.jpg"),
            (CelestialBody::Haumea, Skin::Default, TextureResolution::R8192) => Some("textures/haumea/haumea_4k.jpg"),
            (CelestialBody::Haumea, Skin::Default, _) => Some("textures/haumea/haumea_2k.jpg"),
            (CelestialBody::Makemake, Skin::Default, TextureResolution::R8192) => Some("textures/makemake/makemake_4k.jpg"),
            (CelestialBody::Makemake, Skin::Default, _) => Some("textures/makemake/makemake_2k.jpg"),
            (CelestialBody::Eris, Skin::Default, TextureResolution::R8192) => Some("textures/eris/eris_4k.jpg"),
            (CelestialBody::Eris, Skin::Default, _) => Some("textures/eris/eris_2k.jpg"),
            _ => None,
        }
    }
}

impl CelestialBody {
    fn label(&self) -> &'static str {
        match self {
            CelestialBody::Earth => "Earth",
            CelestialBody::Moon => "Moon",
            CelestialBody::Mars => "Mars",
            CelestialBody::Mercury => "Mercury",
            CelestialBody::Venus => "Venus",
            CelestialBody::Jupiter => "Jupiter",
            CelestialBody::Saturn => "Saturn",
            CelestialBody::Uranus => "Uranus",
            CelestialBody::Neptune => "Neptune",
            CelestialBody::Sun => "Sun",
            CelestialBody::Ceres => "Ceres",
            CelestialBody::Haumea => "Haumea",
            CelestialBody::Makemake => "Makemake",
            CelestialBody::Eris => "Eris",
        }
    }

    fn available_skins(&self) -> &'static [Skin] {
        match self {
            CelestialBody::Earth => &[Skin::Default, Skin::HellOnEarth],
            CelestialBody::Mars => &[Skin::Default, Skin::Terraformed, Skin::Civilized],
            _ => &[Skin::Default],
        }
    }

    const ALL: [CelestialBody; 14] = [
        CelestialBody::Earth,
        CelestialBody::Moon,
        CelestialBody::Mars,
        CelestialBody::Mercury,
        CelestialBody::Venus,
        CelestialBody::Jupiter,
        CelestialBody::Saturn,
        CelestialBody::Uranus,
        CelestialBody::Neptune,
        CelestialBody::Sun,
        CelestialBody::Ceres,
        CelestialBody::Haumea,
        CelestialBody::Makemake,
        CelestialBody::Eris,
    ];

    fn radius_km(&self) -> f64 {
        match self {
            CelestialBody::Earth => 6371.0,
            CelestialBody::Moon => 1737.4,
            CelestialBody::Mars => 3389.5,
            CelestialBody::Mercury => 2439.7,
            CelestialBody::Venus => 6051.8,
            CelestialBody::Jupiter => 69911.0,
            CelestialBody::Saturn => 58232.0,
            CelestialBody::Uranus => 25362.0,
            CelestialBody::Neptune => 24622.0,
            CelestialBody::Sun => 696340.0,
            CelestialBody::Ceres => 473.0,
            CelestialBody::Haumea => 816.0,
            CelestialBody::Makemake => 715.0,
            CelestialBody::Eris => 1163.0,
        }
    }

    fn mu(&self) -> f64 {
        match self {
            CelestialBody::Earth => 398600.4418,
            CelestialBody::Moon => 4902.8,
            CelestialBody::Mars => 42828.37,
            CelestialBody::Mercury => 22032.0,
            CelestialBody::Venus => 324859.0,
            CelestialBody::Jupiter => 126686534.0,
            CelestialBody::Saturn => 37931187.0,
            CelestialBody::Uranus => 5793939.0,
            CelestialBody::Neptune => 6836529.0,
            CelestialBody::Sun => 132712440018.0,
            CelestialBody::Ceres => 62.6,
            CelestialBody::Haumea => 2.67,
            CelestialBody::Makemake => 2.0,
            CelestialBody::Eris => 111.0,
        }
    }

    fn j2(&self) -> f64 {
        match self {
            CelestialBody::Earth => 1.08263e-3,
            CelestialBody::Moon => 2.03e-4,
            CelestialBody::Mars => 1.96045e-3,
            CelestialBody::Mercury => 6.0e-5,
            CelestialBody::Venus => 4.458e-6,
            CelestialBody::Jupiter => 1.4736e-2,
            CelestialBody::Saturn => 1.6298e-2,
            CelestialBody::Uranus => 3.343e-3,
            CelestialBody::Neptune => 3.411e-3,
            CelestialBody::Sun => 2.0e-7,
            CelestialBody::Ceres => 0.0,
            CelestialBody::Haumea => 0.0,
            CelestialBody::Makemake => 0.0,
            CelestialBody::Eris => 0.0,
        }
    }

    fn equatorial_radius_km(&self) -> f64 {
        match self {
            CelestialBody::Earth => 6378.137,
            CelestialBody::Moon => 1738.1,
            CelestialBody::Mars => 3396.2,
            CelestialBody::Mercury => 2440.5,
            CelestialBody::Venus => 6051.8,
            CelestialBody::Jupiter => 71492.0,
            CelestialBody::Saturn => 60268.0,
            CelestialBody::Uranus => 25559.0,
            CelestialBody::Neptune => 24764.0,
            CelestialBody::Sun => 696000.0,
            CelestialBody::Ceres => 473.0,
            CelestialBody::Haumea => 960.0,
            CelestialBody::Makemake => 715.0,
            CelestialBody::Eris => 1163.0,
        }
    }

    fn flattening(&self) -> f64 {
        match self {
            CelestialBody::Earth => 1.0 / 298.257,
            CelestialBody::Moon => 0.0012,
            CelestialBody::Mars => 1.0 / 169.89,
            CelestialBody::Mercury => 0.0009,
            CelestialBody::Venus => 0.0,
            CelestialBody::Jupiter => 1.0 / 15.41,
            CelestialBody::Saturn => 1.0 / 10.21,
            CelestialBody::Uranus => 1.0 / 43.62,
            CelestialBody::Neptune => 1.0 / 58.54,
            CelestialBody::Sun => 9.0e-6,
            CelestialBody::Ceres => 0.0,
            CelestialBody::Haumea => 0.19,
            CelestialBody::Makemake => 0.0,
            CelestialBody::Eris => 0.0,
        }
    }

    fn rotation_period_hours(&self) -> f64 {
        match self {
            CelestialBody::Earth => 23.9345,
            CelestialBody::Moon => 655.7,
            CelestialBody::Mars => 24.6229,
            CelestialBody::Mercury => 1407.6,
            CelestialBody::Venus => -5832.5,
            CelestialBody::Jupiter => 9.925,
            CelestialBody::Saturn => 10.656,
            CelestialBody::Uranus => -17.24,
            CelestialBody::Neptune => 16.11,
            CelestialBody::Sun => 609.12,
            CelestialBody::Ceres => 9.074,
            CelestialBody::Haumea => 3.92,
            CelestialBody::Makemake => 22.48,
            CelestialBody::Eris => 25.9,
        }
    }
}

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
        let bytes = std::fs::read(asset_path("textures/earth/earth_8k.jpg"))
            .expect("Failed to read textures/earth/earth_8k.jpg");
        Self::from_bytes(&bytes).expect("Failed to load Earth texture")
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
                let count = (factor * factor) as u32;
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
                uniform float u_blend;
                uniform float u_center_lat;
                uniform float u_center_lon;
                uniform float u_show_stars;

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

                    float merc_y_center = log(tan(PI / 4.0 + u_center_lat / 2.0));
                    float scale_f = 1.0 / cos(u_center_lat);
                    float lon_merc = u_center_lon + centered.x * scale_f;
                    float merc_y = merc_y_center + centered.y * scale_f;
                    float lat_merc = 2.0 * atan(exp(clamp(merc_y, -3.0, 3.0))) - PI / 2.0;

                    float lat, lon;
                    vec3 normal;
                    float alpha;

                    if (u_blend < 0.001) {
                        if (!ortho_hit) {
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

                            out_color = vec4(bg, bg_alpha);
                            return;
                        }
                        lat = lat_ortho;
                        lon = lon_ortho;
                        normal = normal_ortho;
                        alpha = 1.0;
                    } else if (u_blend > 0.999) {
                        lat = lat_merc;
                        lon = lon_merc;
                        normal = normalize(vec3(cos(lat)*cos(lon), sin(lat)/b, -cos(lat)*sin(lon)));
                        alpha = 1.0;
                    } else {
                        if (ortho_hit) {
                            float dlon = lon_merc - lon_ortho;
                            if (dlon > PI) dlon -= 2.0 * PI;
                            if (dlon < -PI) dlon += 2.0 * PI;
                            lon = lon_ortho + dlon * u_blend;
                            lat = mix(lat_ortho, lat_merc, u_blend);
                            vec3 n_merc = normalize(vec3(cos(lat_merc)*cos(lon_merc), sin(lat_merc)/b, -cos(lat_merc)*sin(lon_merc)));
                            normal = normalize(mix(normal_ortho, n_merc, u_blend));
                            alpha = 1.0;
                        } else {
                            vec3 star_bg = vec3(0.0);
                            float star_alpha = 0.0;
                            if (u_show_stars > 0.5) {
                                vec2 sp = (v_uv - 0.5) * 2.0;
                                sp.x *= u_aspect;
                                vec3 dir = u_inv_rotation * normalize(vec3(sp, -2.0));
                                float slat = asin(clamp(dir.y, -1.0, 1.0));
                                float slon = atan(-dir.z, dir.x);
                                float su = (slon + PI) / (2.0 * PI);
                                float sv = (PI / 2.0 - slat) / PI;
                                star_bg = texture(u_stars, vec2(su, sv)).rgb * (1.0 - u_blend);
                                star_alpha = 1.0 - u_blend;
                            }

                            lat = lat_merc;
                            lon = lon_merc;
                            normal = normalize(vec3(cos(lat)*cos(lon), sin(lat)/b, -cos(lat)*sin(lon)));
                            float atmo_glow = 0.0;
                            if (u_atmosphere > 0.0 && screen_dist < atmo_outer) {
                                float atmo_depth = (screen_dist - 1.0) / (ATMO_THICKNESS * u_atmosphere);
                                atmo_depth = clamp(atmo_depth, 0.0, 1.0);
                                float atmo_falloff = 1.0 - atmo_depth;
                                atmo_falloff = pow(atmo_falloff, 2.0);
                                atmo_glow = atmo_falloff * 0.8 * (1.0 - u_blend);
                            }
                            alpha = max(u_blend, max(star_alpha, atmo_glow));
                            if (star_alpha > u_blend && atmo_glow <= star_alpha) {
                                vec3 bg = star_bg;
                                if (atmo_glow > 0.0) {
                                    bg = bg * (1.0 - atmo_glow / star_alpha) + ATMO_COLOR * atmo_glow;
                                }
                                out_color = vec4(bg, alpha);
                                return;
                            }
                            if (atmo_glow > alpha) {
                                out_color = vec4(ATMO_COLOR * atmo_glow, atmo_glow);
                                return;
                            }
                        }
                    }

                    float tex_u = (lon + PI) / (2.0 * PI);
                    float tex_v = (PI / 2.0 - lat) / PI;

                    vec3 day_color;
                    if (u_use_detail > 0.5) {
                        float lon_deg = lon * 180.0 / PI;
                        if (lon_deg < u_detail_bounds.x) lon_deg += 360.0;
                        float detail_merc_y = log(tan(PI / 4.0 + lat / 2.0));
                        float du = (lon_deg - u_detail_bounds.x) / (u_detail_bounds.y - u_detail_bounds.x);
                        float dv = (u_detail_bounds.w - detail_merc_y) / (u_detail_bounds.w - u_detail_bounds.z);
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
                        float shade_3d = 0.3 + 0.7 * max(dot(normal, -D), 0.0);
                        float shade = mix(shade_3d, 0.85, u_blend);
                        color = day_color * shade;
                    }

                    if (u_atmosphere > 0.0 && u_blend < 0.999) {
                        float fresnel = 1.0 - max(dot(normal, -D), 0.0);
                        fresnel = pow(fresnel, 3.0);
                        float rim = fresnel * 0.6 * u_atmosphere * (1.0 - u_blend);
                        float atmo_sun = u_show_day_night > 0.5 ? max(sun_dot + 0.3, 0.0) : 1.0;
                        color = mix(color, ATMO_COLOR * atmo_sun, rim);
                    }

                    out_color = vec4(color, alpha);
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
        blend: f32,
        center_lat: f32,
        center_lon: f32,
        show_stars: bool,
        show_milky_way: bool,
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
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_blend").as_ref(), blend);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_center_lat").as_ref(), center_lat);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_center_lon").as_ref(), center_lon);

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
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum WalkerType {
    Delta,
    Star,
}

struct WalkerConstellation {
    walker_type: WalkerType,
    total_sats: usize,
    num_planes: usize,
    altitude_km: f64,
    inclination_deg: f64,
    phasing: f64,
    raan_offset_deg: f64,
    raan_spacing_deg: Option<f64>,
    eccentricity: f64,
    arg_periapsis_deg: f64,
    planet_radius: f64,
    planet_mu: f64,
    planet_j2: f64,
    planet_equatorial_radius: f64,
}

impl WalkerConstellation {
    fn sats_per_plane(&self) -> usize {
        self.total_sats / self.num_planes
    }

    fn raan_spread(&self) -> f64 {
        if let Some(spacing) = self.raan_spacing_deg {
            spacing.to_radians() * (self.num_planes - 1) as f64
        } else {
            match self.walker_type {
                WalkerType::Delta => 2.0 * PI,
                WalkerType::Star => PI,
            }
        }
    }

    fn raan_step(&self) -> f64 {
        if let Some(spacing) = self.raan_spacing_deg {
            spacing.to_radians()
        } else {
            self.raan_spread() / self.num_planes as f64
        }
    }

    fn satellite_positions(&self, time: f64) -> Vec<SatelliteState> {
        let mut positions = Vec::with_capacity(self.total_sats);
        let sats_per_plane = self.sats_per_plane();
        let perigee_radius = self.planet_radius + self.altitude_km;
        let ecc = self.eccentricity;
        let semi_major = perigee_radius / (1.0 - ecc);
        let period = 2.0 * PI * (semi_major.powi(3) / self.planet_mu).sqrt();
        let mean_motion = 2.0 * PI / period;
        let inc = self.inclination_deg.to_radians();
        let inc_cos = inc.cos();
        let inc_sin = inc.sin();
        let raan_step = self.raan_step();
        let raan_offset = -self.raan_offset_deg.to_radians();
        let sat_step = 2.0 * PI / sats_per_plane as f64;
        let is_star = self.walker_type == WalkerType::Star;
        let omega = self.arg_periapsis_deg.to_radians();

        let phase_step = self.phasing * 2.0 * PI / self.total_sats as f64;

        let r_ratio = self.planet_equatorial_radius / semi_major;
        let raan_drift_rate = -1.5 * self.planet_j2 * r_ratio * r_ratio * mean_motion * inc_cos;

        let center_offset = if self.raan_spacing_deg.is_some() {
            raan_step * (self.num_planes - 1) as f64 / 2.0
        } else {
            0.0
        };
        let dead = self.altitude_km < 100.0;
        for plane in 0..self.num_planes {
            let raan_initial = raan_offset + raan_step * plane as f64 - center_offset;
            let raan = raan_initial + if dead { 0.0 } else { raan_drift_rate * time };
            let raan_cos = raan.cos();
            let raan_sin = raan.sin();
            let phase_offset = phase_step * plane as f64;

            for sat in 0..sats_per_plane {
                let mean_anomaly = sat_step * sat as f64 + if dead { 0.0 } else { mean_motion * time } + phase_offset;

                let true_anomaly = if ecc < 1e-8 {
                    mean_anomaly
                } else {
                    let mut ea = mean_anomaly;
                    for _ in 0..10 {
                        ea = ea - (ea - ecc * ea.sin() - mean_anomaly) / (1.0 - ecc * ea.cos());
                    }
                    2.0 * ((1.0 + ecc).sqrt() * (ea / 2.0).sin())
                        .atan2((1.0 - ecc).sqrt() * (ea / 2.0).cos())
                };

                let r = semi_major * (1.0 - ecc * ecc) / (1.0 + ecc * true_anomaly.cos());
                let ascending = (true_anomaly + omega).cos() > 0.0;

                let angle = true_anomaly + omega;
                let x_orbital = r * angle.cos();
                let y_orbital = -r * angle.sin();

                let x = x_orbital * raan_cos - y_orbital * inc_cos * raan_sin;
                let z = x_orbital * raan_sin + y_orbital * inc_cos * raan_cos;
                let y = -y_orbital * inc_sin;

                let lat = (y / r).asin().to_degrees();
                let lon = -z.atan2(x).to_degrees();

                positions.push(SatelliteState {
                    plane,
                    sat_index: sat,
                    x,
                    y,
                    z,
                    lat,
                    lon,
                    ascending,
                    neighbor_idx: None,
                    name: None,
                });
            }
        }

        let no_wrap = is_star || self.raan_spacing_deg.is_some();
        for i in 0..positions.len() {
            let sat = &positions[i];
            if no_wrap && sat.plane == self.num_planes - 1 {
                continue;
            }
            let next_plane = (sat.plane + 1) % self.num_planes;
            let next_plane_start = next_plane * sats_per_plane;
            let next_plane_end = next_plane_start + sats_per_plane;
            let target_idx = sat.sat_index;
            for j in next_plane_start..next_plane_end {
                let other = &positions[j];
                if other.sat_index == target_idx {
                    positions[i].neighbor_idx = Some(j);
                    break;
                }
            }
        }

        positions
    }

    fn orbit_points_3d(&self, plane: usize, time: f64) -> Vec<(f64, f64, f64)> {
        let ecc = self.eccentricity;
        let semi_major = (self.planet_radius + self.altitude_km) / (1.0 - ecc);
        let period = 2.0 * PI * (semi_major.powi(3) / self.planet_mu).sqrt();
        let mean_motion = 2.0 * PI / period;
        let r_ratio = self.planet_equatorial_radius / semi_major;
        let inc = self.inclination_deg.to_radians();
        let raan_drift_rate = -1.5 * self.planet_j2 * r_ratio * r_ratio * mean_motion * inc.cos();
        let omega = self.arg_periapsis_deg.to_radians();

        let raan_step = self.raan_step();
        let center_offset = if self.raan_spacing_deg.is_some() {
            raan_step * (self.num_planes - 1) as f64 / 2.0
        } else {
            0.0
        };
        let raan_initial = -self.raan_offset_deg.to_radians() + raan_step * plane as f64 - center_offset;
        let raan = raan_initial + raan_drift_rate * time;
        let inc_cos = inc.cos();
        let inc_sin = inc.sin();
        let raan_cos = raan.cos();
        let raan_sin = raan.sin();

        let apogee = semi_major * (1.0 + ecc);
        let perigee = semi_major * (1.0 - ecc);
        let size_ratio = apogee / perigee;
        let num_points = (200.0 * size_ratio).min(2000.0) as usize;

        (0..=num_points)
            .map(|i| {
                let true_anomaly = 2.0 * PI * i as f64 / num_points as f64;
                let r = semi_major * (1.0 - ecc * ecc) / (1.0 + ecc * true_anomaly.cos());
                let angle = true_anomaly + omega;
                let x_orbital = r * angle.cos();
                let y_orbital = -r * angle.sin();

                let x = x_orbital * raan_cos - y_orbital * inc_cos * raan_sin;
                let z = x_orbital * raan_sin + y_orbital * inc_cos * raan_cos;
                let y = -y_orbital * inc_sin;

                (x, y, z)
            })
            .collect()
    }
}

struct SatelliteState {
    plane: usize,
    sat_index: usize,
    x: f64,
    y: f64,
    z: f64,
    lat: f64,
    lon: f64,
    ascending: bool,
    neighbor_idx: Option<usize>,
    name: Option<String>,
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

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum TlePreset {
    Iss,
    Starlink,
    OneWeb,
    Iridium,
    IridiumNext,
    Globalstar,
    Orbcomm,
    Gps,
    Galileo,
    Glonass,
    Beidou,
}

impl TlePreset {
    fn label(&self) -> &'static str {
        match self {
            TlePreset::Iss => "ISS",
            TlePreset::Starlink => "Starlink",
            TlePreset::OneWeb => "OneWeb",
            TlePreset::Iridium => "Iridium",
            TlePreset::IridiumNext => "Iridium NEXT",
            TlePreset::Globalstar => "Globalstar",
            TlePreset::Orbcomm => "Orbcomm",
            TlePreset::Gps => "GPS",
            TlePreset::Galileo => "Galileo",
            TlePreset::Glonass => "GLONASS",
            TlePreset::Beidou => "Beidou",
        }
    }

    fn url(&self) -> &'static str {
        match self {
            TlePreset::Iss => "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=tle",
            TlePreset::Starlink => "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
            TlePreset::OneWeb => "https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle",
            TlePreset::Iridium => "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium&FORMAT=tle",
            TlePreset::IridiumNext => "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-NEXT&FORMAT=tle",
            TlePreset::Globalstar => "https://celestrak.org/NORAD/elements/gp.php?GROUP=globalstar&FORMAT=tle",
            TlePreset::Orbcomm => "https://celestrak.org/NORAD/elements/gp.php?GROUP=orbcomm&FORMAT=tle",
            TlePreset::Gps => "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle",
            TlePreset::Galileo => "https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle",
            TlePreset::Glonass => "https://celestrak.org/NORAD/elements/gp.php?GROUP=glo-ops&FORMAT=tle",
            TlePreset::Beidou => "https://celestrak.org/NORAD/elements/gp.php?GROUP=beidou&FORMAT=tle",
        }
    }

    const ALL: [TlePreset; 11] = [
        TlePreset::Iss,
        TlePreset::Starlink,
        TlePreset::OneWeb,
        TlePreset::Iridium,
        TlePreset::IridiumNext,
        TlePreset::Globalstar,
        TlePreset::Orbcomm,
        TlePreset::Gps,
        TlePreset::Galileo,
        TlePreset::Glonass,
        TlePreset::Beidou,
    ];
}

#[derive(Clone)]
struct TleSatellite {
    name: String,
    constants: Constants,
    epoch_minutes: f64,
}

#[derive(Clone)]
#[allow(dead_code)]
enum TleLoadState {
    NotLoaded,
    Loading,
    Loaded { satellites: Vec<TleSatellite>, loaded_at: std::time::Instant },
    Failed(String),
}

fn datetime_to_minutes(dt: &sgp4::chrono::NaiveDateTime) -> f64 {
    dt.and_utc().timestamp() as f64 / 60.0
}

fn greenwich_mean_sidereal_time(timestamp: DateTime<Utc>) -> f64 {
    let j2000 = DateTime::parse_from_rfc3339("2000-01-01T12:00:00Z")
        .unwrap()
        .with_timezone(&Utc);
    let days_since_j2000 = (timestamp - j2000).num_milliseconds() as f64 / (1000.0 * 86400.0);
    let centuries = days_since_j2000 / 36525.0;
    let gmst_degrees = 280.46061837
        + 360.98564736629 * days_since_j2000
        + 0.000387933 * centuries * centuries
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

#[cfg(not(target_arch = "wasm32"))]
fn fetch_tle_data(url: &str) -> Result<Vec<TleSatellite>, String> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("HTTP error: {}", e))?;

    let body = response.into_string()
        .map_err(|e| format!("Read error: {}", e))?;

    parse_tle_data(&body)
}

#[cfg(target_arch = "wasm32")]
fn fetch_tle_data(_url: &str) -> Result<Vec<TleSatellite>, String> {
    Err("WASM fetch not yet implemented".to_string())
}

fn parse_tle_data(data: &str) -> Result<Vec<TleSatellite>, String> {
    let lines: Vec<&str> = data.lines().collect();
    let mut satellites = Vec::new();

    let mut i = 0;
    while i + 2 < lines.len() {
        let name_line = lines[i].trim();
        let line1 = lines[i + 1].trim();
        let line2 = lines[i + 2].trim();

        if !line1.starts_with('1') || !line2.starts_with('2') {
            i += 1;
            continue;
        }

        let tle = format!("{}\n{}\n{}", name_line, line1, line2);

        match sgp4::parse_3les(&tle) {
            Ok(elements_vec) => {
                for elements in elements_vec {
                    match Constants::from_elements(&elements) {
                        Ok(constants) => {
                            let epoch_minutes = datetime_to_minutes(&elements.datetime);
                            satellites.push(TleSatellite {
                                name: elements.object_name.unwrap_or_default(),
                                constants,
                                epoch_minutes,
                            });
                        }
                        Err(_) => continue,
                    }
                }
            }
            Err(_) => {}
        }

        i += 3;
    }

    if satellites.is_empty() {
        Err("No valid TLE data found".to_string())
    } else {
        Ok(satellites)
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
    tle_selections: HashMap<TlePreset, (bool, TleLoadState)>,
    ground_stations: Vec<GroundStation>,
    areas_of_interest: Vec<AreaOfInterest>,
}

impl PlanetConfig {
    fn new(name: String) -> Self {
        let mut tle_selections = HashMap::new();
        for preset in TlePreset::ALL {
            tle_selections.insert(preset, (false, TleLoadState::NotLoaded));
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
        }
    }

    fn add_constellation(&mut self) {
        self.constellations.push(ConstellationConfig::new(self.constellation_counter));
        self.constellation_counter += 1;
    }

    fn tle_total_sats(&self) -> usize {
        self.tle_selections.values()
            .filter(|(selected, _)| *selected)
            .map(|(_, state)| {
                if let TleLoadState::Loaded { satellites, .. } = state {
                    satellites.len()
                } else {
                    0
                }
            })
            .sum()
    }
}

#[derive(Clone)]
struct TabSettings {
    show_orbits: bool,
    show_links: bool,
    show_intra_links: bool,
    show_coverage: bool,
    coverage_angle: f64,
    show_routing_paths: bool,
    show_manhattan_path: bool,
    show_shortest_path: bool,
    show_asc_desc_colors: bool,
    show_torus: bool,
    show_ground_track: bool,
    show_axes: bool,
    hide_behind_earth: bool,
    follow_satellite: bool,
    render_planet: bool,
    show_altitude_lines: bool,
    show_camera_windows: bool,
}

impl Default for TabSettings {
    fn default() -> Self {
        Self {
            show_orbits: true,
            show_links: true,
            show_intra_links: false,
            show_coverage: false,
            coverage_angle: 25.0,
            show_routing_paths: false,
            show_manhattan_path: true,
            show_shortest_path: true,
            show_asc_desc_colors: false,
            show_torus: false,
            show_ground_track: false,
            show_axes: false,
            hide_behind_earth: true,
            follow_satellite: false,
            render_planet: true,
            show_altitude_lines: false,
            show_camera_windows: false,
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
    show_settings_window: bool,
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
            show_settings_window: false,
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
    show_day_night: bool,
    show_terminator: bool,
    show_stars: bool,
    show_milky_way: bool,
    dragging_place: Option<(usize, usize, bool, usize)>,
    night_texture: Option<Arc<EarthTexture>>,
    star_texture: Option<Arc<EarthTexture>>,
    milky_way_texture: Option<Arc<EarthTexture>>,
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
            renderer.upload_texture(gl, (CelestialBody::Earth, Skin::Default, TextureResolution::R8192), &builtin_texture);
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
                    map.insert((CelestialBody::Earth, Skin::Default, TextureResolution::R8192), builtin_texture.clone());
                    map
                },
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
                show_clouds: true,
                show_day_night: false,
                show_terminator: false,
                show_stars: false,
                show_milky_way: false,
                dragging_place: None,
                night_texture: None,
                star_texture: None,
                milky_way_texture: None,
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
            self.render_tab_ui(ui, *tab);
        }
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
            let t = &mut self.tabs[*tab];
            if ui.checkbox(&mut t.use_local_settings, "Override global settings").changed() {
                if t.use_local_settings {
                    t.show_settings_window = true;
                }
            }
            if ui.button("Tab Settings").clicked() {
                t.show_settings_window = !t.show_settings_window;
                ui.close();
            }
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

        if self.tabs[tab_idx].show_settings_window {
            let tab = &mut self.tabs[tab_idx];
            egui::Window::new(format!("Tab Settings"))
                .id(egui::Id::new(format!("tab_settings_{}", tab_idx)))
                .open(&mut tab.show_settings_window)
                .default_width(200.0)
                .show(ui.ctx(), |ui| {
                    ui.checkbox(&mut tab.use_local_settings, "Override global settings");
                    ui.separator();
                    ui.add_enabled_ui(tab.use_local_settings, |ui| {
                        let s = &mut tab.local_settings;
                        ui.checkbox(&mut s.show_orbits, "Show orbits");
                        ui.checkbox(&mut s.show_links, "Inter-plane links");
                        ui.checkbox(&mut s.show_intra_links, "Intra-plane links");
                        ui.checkbox(&mut s.show_routing_paths, "Show routing paths");
                        if s.show_routing_paths {
                            ui.indent("local_routing", |ui| {
                                ui.checkbox(&mut s.show_manhattan_path, "Manhattan (red)");
                                ui.checkbox(&mut s.show_shortest_path, "Shortest (green)");
                            });
                        }
                        ui.checkbox(&mut s.show_asc_desc_colors, "Asc/Desc colors");
                        ui.checkbox(&mut s.show_torus, "Show torus");
                        ui.checkbox(&mut s.show_ground_track, "Show ground");
                        ui.checkbox(&mut s.show_axes, "Show axes");
                        ui.checkbox(&mut s.show_altitude_lines, "Altitude lines");
                        ui.checkbox(&mut s.show_coverage, "Show coverage");
                        if s.show_coverage {
                            ui.horizontal(|ui| {
                                ui.label("Angle:");
                                ui.add(egui::DragValue::new(&mut s.coverage_angle)
                                    .range(0.5..=70.0).speed(0.1).suffix("°"));
                            });
                        }
                        ui.checkbox(&mut s.render_planet, "Show planet");
                        {
                            let mut show_behind = !s.hide_behind_earth;
                            if ui.add_enabled(s.render_planet, egui::Checkbox::new(&mut show_behind, "Show behind planet")).changed() {
                                s.hide_behind_earth = !show_behind;
                            }
                        }
                        ui.checkbox(&mut s.follow_satellite, "Follow satellite");
                        ui.checkbox(&mut s.show_camera_windows, "Camera windows");
                    });
                });
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
            let mut gs_changed = false;
            let mut aoi_changed = false;

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
                });

            if gs_changed {
                self.tabs[tab_idx].planets[planet_idx].ground_stations = ground_stations;
            }
            if aoi_changed {
                self.tabs[tab_idx].planets[planet_idx].areas_of_interest = areas_of_interest;
            }
        }

        let show_stats = self.tabs[tab_idx].show_stats;
        if show_stats {
            let planet = &self.tabs[tab_idx].planets[planet_idx];
            let planet_name = planet.name.clone();
            let celestial_body = planet.celestial_body;
            let planet_radius = celestial_body.radius_km();
            let mu = celestial_body.mu();
            let constellations: Vec<_> = planet.constellations.iter().cloned().collect();
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
                    let geo_orbit = (mu * (86400.0 / (2.0 * PI)).powi(2)).powf(1.0/3.0);
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
                            if let Some((selected, state)) = tle_selections.get(preset) {
                                if *selected {
                                    if let TleLoadState::Loaded { satellites, .. } = state {
                                        return Some((preset.label(), satellites.len()));
                                    }
                                }
                            }
                            None
                        })
                        .collect();

                    if !live_data.is_empty() {
                        ui.heading("Live Data (TLE)");
                        let mut total = 0;
                        for (name, count) in &live_data {
                            ui.label(format!("  {}: {} satellites", name, count));
                            total += count;
                        }
                        ui.label(format!("  Total: {} satellites", total));
                    }
                });
        }

        if show_config {
        ui.separator();

        let mut const_to_remove: Option<usize> = None;
        let planet = &mut self.tabs[tab_idx].planets[planet_idx];
        let num_constellations = planet.constellations.len();
        let show_tle = planet.show_tle_window;

        #[cfg(not(target_arch = "wasm32"))]
        let tle_fetch_tx = self.tle_fetch_tx.clone();

        let controls_height = 180.0;
        egui::ScrollArea::vertical()
            .id_salt(format!("controls_{}_{}",tab_idx, planet_idx))
            .max_height(controls_height)
            .show(ui, |ui| {
        let planet = &mut self.tabs[tab_idx].planets[planet_idx];
        ui.horizontal(|ui| {
            if show_tle {
                ui.vertical(|ui| {
                    let mut fetch_requested = false;
                    ui.horizontal(|ui| {
                        ui.label("TLE");
                        if ui.small_button("All").clicked() {
                            for (selected, _) in planet.tle_selections.values_mut() {
                                *selected = true;
                            }
                        }
                        if ui.small_button("None").clicked() {
                            for (selected, _) in planet.tle_selections.values_mut() {
                                *selected = false;
                            }
                        }
                        if ui.small_button("Fetch").clicked() {
                            fetch_requested = true;
                        }
                        if ui.small_button("x").clicked() {
                            planet.show_tle_window = false;
                        }
                    });

                    ui.horizontal(|ui| {
                        for col in 0..2 {
                            ui.vertical(|ui| {
                                for row in 0..5 {
                                    let preset_idx = col * 5 + row;
                                    if preset_idx < TlePreset::ALL.len() {
                                        let preset = &TlePreset::ALL[preset_idx];
                                        if let Some((selected, state)) = planet.tle_selections.get_mut(preset) {
                                            ui.horizontal(|ui| {
                                                let color = plane_color(preset_idx);
                                                let rect = ui.allocate_space(egui::vec2(10.0, 10.0)).1;
                                                ui.painter().rect_filled(rect, 2.0, color);

                                                let is_loading = matches!(state, TleLoadState::Loading);
                                                if ui.selectable_label(*selected, preset.label()).clicked() {
                                                    *selected = !*selected;
                                                }
                                                if is_loading {
                                                    ui.spinner();
                                                }

                                                #[cfg(not(target_arch = "wasm32"))]
                                                if fetch_requested && *selected {
                                                    if matches!(state, TleLoadState::NotLoaded | TleLoadState::Failed(_)) {
                                                        *state = TleLoadState::Loading;
                                                        let url = preset.url().to_string();
                                                        let preset_copy = *preset;
                                                        let tx = tle_fetch_tx.clone();
                                                        std::thread::spawn(move || {
                                                            let result = fetch_tle_data(&url);
                                                            let _ = tx.send((preset_copy, result));
                                                        });
                                                    }
                                                }
                                            });
                                        }
                                    }
                                }
                            });
                        }
                    });
                });
                ui.separator();
            }

            for (cidx, cons) in planet.constellations.iter_mut().enumerate() {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
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
                        if ui.toggle_value(&mut cons.drag_enabled, "Drag").changed() {
                            cons.preset = Preset::None;
                        }
                        if cons.drag_enabled {
                            ui.label("BC:");
                            if ui.add(egui::DragValue::new(&mut cons.ballistic_coeff).range(0.1..=500.0).suffix(" kg/m²").speed(1.0).max_decimals(1)).changed() {
                                cons.preset = Preset::None;
                            }
                        }
                    });

                    ui.horizontal(|ui| {
                        let old_type = cons.walker_type;
                        ui.selectable_value(&mut cons.walker_type, WalkerType::Delta, "Delta");
                        ui.selectable_value(&mut cons.walker_type, WalkerType::Star, "Star");
                        ui.label(format!("({} sats)", cons.total_sats()));
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
                self.tabs[tab_idx].planets[planet_idx].constellations.remove(cidx);
            }
        }
        });

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
                    (wc, pos, c.color_offset, false, orig_idx, name)
                })
                .collect()
        };

        if planet.show_tle_window {
            let sim_timestamp = self.start_timestamp + chrono::Duration::seconds(self.time as i64);
            let propagation_minutes = sim_timestamp.timestamp() as f64 / 60.0;
            for preset in TlePreset::ALL.iter() {
                let Some((selected, state)) = planet.tle_selections.get(preset) else { continue };
                if !*selected { continue; }
                let TleLoadState::Loaded { satellites, .. } = state else { continue };
                let mut positions = Vec::new();
                for (idx, sat) in satellites.iter().enumerate() {
                    let minutes_since_epoch = propagation_minutes - sat.epoch_minutes;
                    let prediction = match sat.constants.propagate(sgp4::MinutesSinceEpoch(minutes_since_epoch)) {
                        Ok(p) => p,
                        Err(_) => continue,
                    };
                    let x = prediction.position[0];
                    let y = prediction.position[2];
                    let z = prediction.position[1];
                    let r = (x * x + y * y + z * z).sqrt();
                    let lat = (y / r).asin().to_degrees();
                    let lon = -z.atan2(x).to_degrees();
                    let ascending = prediction.velocity[2] > 0.0;
                    positions.push(SatelliteState {
                        plane: 0,
                        sat_index: idx,
                        x, y, z,
                        lat, lon,
                        ascending,
                        neighbor_idx: None,
                        name: Some(sat.name.clone()),
                    });
                }
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
                constellations_data.push((tle_wc, positions, constellations_data.len(), true, usize::MAX, preset.label().to_string()));
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
            let rotation = self.rotation;
            let body_rot_angle = body_rotation_angle(celestial_body, self.time, self.current_gmst);
            let cos_a = body_rot_angle.cos();
            let sin_a = body_rot_angle.sin();
            let body_y_rotation = Matrix3::new(
                cos_a, 0.0, sin_a,
                0.0, 1.0, 0.0,
                -sin_a, 0.0, cos_a,
            );
            let satellite_rotation = if self.earth_fixed_camera {
                self.rotation * body_y_rotation.transpose()
            } else {
                self.rotation
            };
            let zoom = self.zoom;
            let sat_radius = self.sat_radius;
            let show_links = if use_local { local.show_links } else { self.show_links };
            let show_intra_links = if use_local { local.show_intra_links } else { self.show_intra_links };
            let render_planet = if use_local { local.render_planet } else { self.render_planet };
            let hide_behind_earth = render_planet && (if use_local { local.hide_behind_earth } else { self.hide_behind_earth });
            let single_color = constellations_data.len() > 1;
            let dark_mode = self.dark_mode;
            let show_routing_paths = if use_local { local.show_routing_paths } else { self.show_routing_paths };
            let show_manhattan_path = if use_local { local.show_manhattan_path } else { self.show_manhattan_path };
            let show_shortest_path = if use_local { local.show_shortest_path } else { self.show_shortest_path };
            let show_asc_desc_colors = if use_local { local.show_asc_desc_colors } else { self.show_asc_desc_colors };
            let show_altitude_lines = if use_local { local.show_altitude_lines } else { self.show_altitude_lines };
            let tex_res = self.texture_resolution;
            let planet_handle = self.planet_image_handles.get(&(celestial_body, skin, tex_res));
            let time = self.time;
            let torus_rotation = self.torus_rotation;
            let torus_zoom = self.torus_zoom;
            let link_width = self.link_width;
            let fixed_sizes = self.fixed_sizes;
            let flattening = celestial_body.flattening();
            let show_polar_circle = self.show_polar_circle;
            let show_equator = self.show_equator;
            let show_terminator = self.show_terminator && self.show_day_night;
            let detail_gl_info = self.tile_overlay_detail_gl_info(celestial_body);
            self.view_width = half_width;
            self.view_height = view_size;

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    let planet = &mut self.tabs[tab_idx].planets[planet_idx];
                    let (rot, new_zoom) = draw_3d_view(
                        ui,
                        &view_name,
                        &constellations_data,
                        show_orbits,
                        show_axes,
                        show_coverage,
                        coverage_angle,
                        rotation,
                        satellite_rotation,
                        half_width,
                        view_size,
                        planet_handle,
                        zoom,
                        sat_radius,
                        show_links,
                        show_intra_links,
                        hide_behind_earth,
                        single_color,
                        dark_mode,
                        &mut planet.pending_cameras,
                        &mut self.camera_id_counter,
                        &mut planet.satellite_cameras,
                        &mut planet.cameras_to_remove,
                        show_routing_paths,
                        show_manhattan_path,
                        show_shortest_path,
                        show_asc_desc_colors,
                        show_altitude_lines,
                        planet_radius,
                        render_planet,
                        link_width,
                        fixed_sizes,
                        flattening,
                        show_polar_circle,
                        show_equator,
                        show_terminator,
                        self.sphere_renderer.as_ref(),
                        (celestial_body, skin, tex_res),
                        &body_y_rotation,
                        self.earth_fixed_camera,
                        self.use_gpu_rendering,
                        self.show_clouds,
                        self.show_day_night,
                        {
                            use chrono::Datelike;
                            let timestamp = self.start_timestamp + chrono::Duration::seconds(self.time as i64);
                            let day_of_year = timestamp.ordinal() as f64;
                            let declination: f64 = -23.45 * ((360.0_f64 / 365.0) * (day_of_year + 10.0)).to_radians().cos();
                            let decl_rad = declination.to_radians();
                            let sun_ra = ((day_of_year - 80.0) * 360.0 / 365.0).to_radians();
                            let sun_inertial = Vector3::new(
                                decl_rad.cos() * sun_ra.cos(),
                                decl_rad.sin(),
                                -decl_rad.cos() * sun_ra.sin(),
                            );
                            let sun_shader = body_y_rotation.transpose() * sun_inertial;
                            [sun_shader.x as f32, sun_shader.y as f32, sun_shader.z as f32]
                        },
                        self.time,
                        &mut planet.ground_stations,
                        &mut planet.areas_of_interest,
                        body_rot_angle,
                        &mut self.dragging_place,
                        (tab_idx, planet_idx),
                        detail_gl_info,
                        self.show_stars,
                        self.show_milky_way,
                    );
                    self.rotation = rot;
                    self.zoom = new_zoom;
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
            let rotation = self.rotation;
            let body_rot_angle = body_rotation_angle(celestial_body, self.time, self.current_gmst);
            let cos_a = body_rot_angle.cos();
            let sin_a = body_rot_angle.sin();
            let body_y_rotation = Matrix3::new(
                cos_a, 0.0, sin_a,
                0.0, 1.0, 0.0,
                -sin_a, 0.0, cos_a,
            );
            let satellite_rotation = if self.earth_fixed_camera {
                self.rotation * body_y_rotation.transpose()
            } else {
                self.rotation
            };
            let zoom = self.zoom;
            let sat_radius = self.sat_radius;
            let show_links = if use_local { local.show_links } else { self.show_links };
            let show_intra_links = if use_local { local.show_intra_links } else { self.show_intra_links };
            let render_planet = if use_local { local.render_planet } else { self.render_planet };
            let hide_behind_earth = render_planet && (if use_local { local.hide_behind_earth } else { self.hide_behind_earth });
            let single_color = constellations_data.len() > 1;
            let dark_mode = self.dark_mode;
            let show_routing_paths = if use_local { local.show_routing_paths } else { self.show_routing_paths };
            let show_manhattan_path = if use_local { local.show_manhattan_path } else { self.show_manhattan_path };
            let show_shortest_path = if use_local { local.show_shortest_path } else { self.show_shortest_path };
            let show_asc_desc_colors = if use_local { local.show_asc_desc_colors } else { self.show_asc_desc_colors };
            let show_altitude_lines = if use_local { local.show_altitude_lines } else { self.show_altitude_lines };
            let tex_res = self.texture_resolution;
            let planet_handle = self.planet_image_handles.get(&(celestial_body, skin, tex_res));
            let link_width = self.link_width;
            let fixed_sizes = self.fixed_sizes;
            let flattening = celestial_body.flattening();
            let show_polar_circle = self.show_polar_circle;
            let show_equator = self.show_equator;
            let show_terminator = self.show_terminator && self.show_day_night;
            let detail_gl_info = self.tile_overlay_detail_gl_info(celestial_body);
            self.view_width = viz_width;
            self.view_height = earth_height;

            let planet = &mut self.tabs[tab_idx].planets[planet_idx];
            let (rot, new_zoom) = draw_3d_view(
                ui,
                &view_name,
                &constellations_data,
                show_orbits,
                show_axes,
                show_coverage,
                coverage_angle,
                rotation,
                satellite_rotation,
                viz_width,
                earth_height,
                planet_handle,
                zoom,
                sat_radius,
                show_links,
                show_intra_links,
                hide_behind_earth,
                single_color,
                dark_mode,
                &mut planet.pending_cameras,
                &mut self.camera_id_counter,
                &mut planet.satellite_cameras,
                &mut planet.cameras_to_remove,
                show_routing_paths,
                show_manhattan_path,
                show_shortest_path,
                show_asc_desc_colors,
                show_altitude_lines,
                planet_radius,
                render_planet,
                link_width,
                fixed_sizes,
                flattening,
                show_polar_circle,
                show_equator,
                show_terminator,
                self.sphere_renderer.as_ref(),
                (celestial_body, skin, tex_res),
                &body_y_rotation,
                self.earth_fixed_camera,
                self.use_gpu_rendering,
                self.show_clouds,
                self.show_day_night,
                {
                    use chrono::Datelike;
                    let timestamp = self.start_timestamp + chrono::Duration::seconds(self.time as i64);
                    let day_of_year = timestamp.ordinal() as f64;
                    let declination: f64 = -23.45 * ((360.0_f64 / 365.0) * (day_of_year + 10.0)).to_radians().cos();
                    let decl_rad = declination.to_radians();
                    let sun_ra = ((day_of_year - 80.0) * 360.0 / 365.0).to_radians();
                    let sun_inertial = Vector3::new(
                        decl_rad.cos() * sun_ra.cos(),
                        decl_rad.sin(),
                        -decl_rad.cos() * sun_ra.sin(),
                    );
                    let sun_shader = body_y_rotation.transpose() * sun_inertial;
                    [sun_shader.x as f32, sun_shader.y as f32, sun_shader.z as f32]
                },
                self.time,
                &mut planet.ground_stations,
                &mut planet.areas_of_interest,
                body_rot_angle,
                &mut self.dragging_place,
                (tab_idx, planet_idx),
                detail_gl_info,
                self.show_stars,
                self.show_milky_way,
            );
            self.rotation = rot;
            self.zoom = new_zoom;

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
        let body_rotation = body_rotation_angle(current_body, self.time, self.current_gmst);

        ui.label(egui::RichText::new("Camera").strong());
        let (lat, base_lon) = matrix_to_lat_lon(&self.rotation);
        let geo_lon = if self.earth_fixed_camera {
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
            let mut alt_km = 10000.0 / self.zoom;
            if ui.add(egui::DragValue::new(&mut alt_km).range(0.5..=1000000.0).speed(100.0).suffix(" km")).changed() {
                self.zoom = (10000.0 / alt_km).clamp(0.01, 20000.0);
            }
            lat_deg = lat_deg.clamp(-90.0, 90.0);
            while lon_deg > 180.0 { lon_deg -= 360.0; }
            while lon_deg < -180.0 { lon_deg += 360.0; }
            if lat_changed || lon_changed {
                let target_lon = if self.earth_fixed_camera {
                    lon_deg.to_radians()
                } else {
                    lon_deg.to_radians() + body_rotation
                };
                self.rotation = lat_lon_to_matrix(lat_deg.to_radians(), target_lon);
            }
        });
        let was_earth_fixed = self.earth_fixed_camera;
        ui.checkbox(&mut self.earth_fixed_camera, "Fixed Lat/Lon");
        if self.earth_fixed_camera != was_earth_fixed {
            let cos_a = body_rotation.cos();
            let sin_a = body_rotation.sin();
            let body_y_rot = Matrix3::new(
                cos_a, 0.0, sin_a,
                0.0, 1.0, 0.0,
                -sin_a, 0.0, cos_a,
            );
            if self.earth_fixed_camera {
                self.rotation = self.rotation * body_y_rot;
            } else {
                self.rotation = self.rotation * body_y_rot.transpose();
            }
        }
        ui.horizontal(|ui| {
            if ui.button("N/S view").clicked() {
                let (_, lon) = matrix_to_lat_lon(&self.rotation);
                self.rotation = lat_lon_to_matrix(0.0, lon);
            }
            if ui.button("E/W view").clicked() {
                let (lat, _) = matrix_to_lat_lon(&self.rotation);
                self.rotation = lat_lon_to_matrix(lat, 0.0);
            }
            if ui.button("Reset").clicked() {
                self.rotation = Matrix3::identity();
                self.torus_rotation = Matrix3::new(
                    1.0, 0.0, 0.0,
                    0.0, 0.0, -1.0,
                    0.0, 1.0, 0.0,
                );
                self.zoom = 1.0;
            }
        });

        ui.checkbox(&mut self.follow_satellite, "Follow satellite");
        ui.checkbox(&mut self.show_camera_windows, "Show camera windows");

        ui.add_space(5.0);
        ui.label(egui::RichText::new("Simulation").strong());
        ui.horizontal(|ui| {
            ui.label("Speed:");
            ui.add(egui::DragValue::new(&mut self.speed).range(-1000.0..=1000.0).speed(1.0));
            if ui.button("⏪").clicked() {
                self.speed = -self.speed;
            }
            let pause_label = if self.animate { "⏸" } else { "▶" };
            if ui.button(pause_label).clicked() {
                self.animate = !self.animate;
            }
        });
        let start = self.start_timestamp;
        let real_timestamp = start + Duration::seconds(self.real_time as i64);
        let current_ts = start + Duration::seconds(self.time as i64);
        ui.horizontal(|ui| {
            ui.label("Time:");
            ui.add(
                egui::DragValue::new(&mut self.time)
                    .speed(1.0)
                    .custom_formatter(|secs, _| {
                        let ts = (start + Duration::seconds(secs as i64)).with_timezone(&Local);
                        ts.format("%H:%M:%S").to_string()
                    })
                    .custom_parser(|input| {
                        if let Ok(secs) = input.parse::<f64>() {
                            return Some(secs);
                        }
                        let formats = ["%H:%M:%S", "%H:%M"];
                        for fmt in formats {
                            if let Ok(parsed) = chrono::NaiveTime::parse_from_str(input, fmt) {
                                let current = current_ts.with_timezone(&Local).date_naive();
                                let new_dt = current.and_time(parsed).and_local_timezone(Local).single()?.with_timezone(&Utc);
                                let diff = new_dt.signed_duration_since(start);
                                return Some(diff.num_seconds() as f64);
                            }
                        }
                        None
                    })
            );
            ui.label("Date:");
            ui.add(
                egui::DragValue::new(&mut self.time)
                    .speed(86400.0)
                    .custom_formatter(|secs, _| {
                        let ts = (start + Duration::seconds(secs as i64)).with_timezone(&Local);
                        ts.format("%d/%m/%Y").to_string()
                    })
                    .custom_parser(|input| {
                        if let Ok(parsed) = chrono::NaiveDate::parse_from_str(input, "%d/%m/%Y") {
                            let current_time = current_ts.with_timezone(&Local).time();
                            let new_dt = parsed.and_time(current_time).and_local_timezone(Local).single()?
                                .with_timezone(&Utc);
                            let diff = new_dt.signed_duration_since(start);
                            return Some(diff.num_seconds() as f64);
                        }
                        None
                    })
            );
        });
        let real_local = real_timestamp.with_timezone(&Local);
        ui.label(format!("Real: {}", real_local.format("%H:%M:%S %d/%m/%Y %Z")));
        if ui.button("Sync time").clicked() {
            self.time = self.real_time;
        }
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
        ui.checkbox(&mut self.show_altitude_lines, "Altitude lines");
        ui.checkbox(&mut self.show_coverage, "Show coverage");
        if self.show_coverage {
            ui.horizontal(|ui| {
                ui.label("Angle:");
                ui.add(egui::DragValue::new(&mut self.coverage_angle)
                    .range(0.5..=70.0)
                    .speed(0.1)
                    .max_decimals(1)
                    .suffix("°"));
            });
        }
        ui.checkbox(&mut self.show_torus, "Show torus");
        ui.checkbox(&mut self.show_ground_track, "Show ground");
        ui.checkbox(&mut self.auto_cycle_tabs, "Auto-cycle tabs");
        if self.auto_cycle_tabs {
            ui.horizontal(|ui| {
                ui.label("Interval:");
                ui.add(egui::DragValue::new(&mut self.cycle_interval).range(1.0..=60.0).speed(0.5).suffix("s"));
            });
        }

        ui.add_space(5.0);
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
                    ui.selectable_value(&mut self.texture_resolution, TextureResolution::R21504, "21K");
                });
        });
        ui.checkbox(&mut self.use_gpu_rendering, "GPU rendering");
        let mut hide_clouds = !self.show_clouds;
        if ui.checkbox(&mut hide_clouds, "Hide clouds").changed() {
            self.show_clouds = !hide_clouds;
        }
        #[cfg(not(target_arch = "wasm32"))]
        ui.checkbox(&mut self.tile_overlay.enabled, "Satellite tiles (Esri)");
        ui.checkbox(&mut self.render_planet, "Show planet");
        {
            let mut show_behind = !self.hide_behind_earth;
            if ui.add_enabled(self.render_planet, egui::Checkbox::new(&mut show_behind, "Show behind planet")).changed() {
                self.hide_behind_earth = !show_behind;
            }
        }
        ui.checkbox(&mut self.show_stars, "Show stars");
        ui.add_enabled(self.show_stars, egui::Checkbox::new(&mut self.show_milky_way, "Show Milky Way"));
        ui.checkbox(&mut self.show_day_night, "Day/night cycle");
        ui.add_enabled(self.show_day_night, egui::Checkbox::new(&mut self.show_terminator, "Show terminator"));
        ui.checkbox(&mut self.show_axes, "Show axes");
        ui.checkbox(&mut self.show_polar_circle, "Show polar circle");
        ui.checkbox(&mut self.show_equator, "Show equator");
        ui.horizontal(|ui| {
            ui.label("Sat:");
            ui.add(egui::DragValue::new(&mut self.sat_radius).range(1.0..=15.0).speed(0.1));
            ui.label("Link:");
            ui.add(egui::DragValue::new(&mut self.link_width).range(0.1..=5.0).speed(0.1));
        });
        ui.checkbox(&mut self.fixed_sizes, "Fixed sizes (ignore alt)");

        ui.add_space(10.0);
        ui.separator();
        ui.label("Delta: RAAN spread 360°");
        ui.label("Star: RAAN spread 180°");
        ui.add_space(5.0);
        ui.label("Drag 3D views to rotate");
        ui.add_space(5.0);
        ui.label("Earth textures: Solar System Scope (CC-BY)");
    }

    #[allow(unused_variables)]
    fn load_texture_for_body(&mut self, body: CelestialBody, skin: Skin, ctx: &egui::Context) {
        let res = self.texture_resolution;
        let key = (body, skin, res);
        if self.planet_textures.contains_key(&key) {
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
                    *cell.borrow_mut() = Some(result);
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
    }
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static TEXTURE_RESULT: std::cell::RefCell<Option<Result<EarthTexture, String>>> = std::cell::RefCell::new(None);
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

impl App {
    fn setup_demo(&mut self) {
        let v = &mut self.viewer;
        v.tabs.clear();
        v.tab_counter = 0;

        let leo_tle = [
            TlePreset::Starlink, TlePreset::OneWeb, TlePreset::Iridium,
            TlePreset::IridiumNext, TlePreset::Globalstar, TlePreset::Orbcomm,
        ];
        let geo_tle = [
            TlePreset::Gps, TlePreset::Galileo, TlePreset::Glonass, TlePreset::Beidou,
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
                planet.tle_selections.insert(preset, (true, TleLoadState::NotLoaded));
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
                planet.tle_selections.insert(*preset, (true, TleLoadState::NotLoaded));
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
            planet_real.tle_selections.insert(TlePreset::Starlink, (true, TleLoadState::NotLoaded));
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
            egui::Visuals::dark()
        } else {
            egui::Visuals::light()
        });

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
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        while let Ok((preset, result)) = v.tle_fetch_rx.try_recv() {
            for tab in &mut v.tabs {
                for planet in &mut tab.planets {
                    if let Some((_, state)) = planet.tle_selections.get_mut(&preset) {
                        if matches!(state, TleLoadState::Loading) {
                            *state = match result.clone() {
                                Ok(satellites) => TleLoadState::Loaded {
                                    satellites,
                                    loaded_at: std::time::Instant::now(),
                                },
                                Err(e) => TleLoadState::Failed(e),
                            };
                        }
                    }
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
                let merc_blend = ((v.zoom - 80.0) / 40.0).clamp(0.0, 1.0);
                let sample_pts: Vec<(f64, f64)> = samples.iter()
                    .filter_map(|&(sx, sy)| screen_to_lonlat(sx, sy)).collect();
                if !sample_pts.is_empty() || merc_blend > 0.0 {
                let (mut min_lon, mut max_lon, mut min_lat, mut max_lat);
                if merc_blend > 0.999 {
                    let cw = surface_rotation.transpose() * nalgebra::Vector3::new(0.0, 0.0, 1.0);
                    let clat = cw.y.asin();
                    let clon = (-cw.z).atan2(cw.x);
                    let sf = 1.0 / clat.cos();
                    let half_y = sf / view_scale;
                    let half_x = sf * aspect / view_scale;
                    let merc_yc = (PI / 4.0 + clat / 2.0).tan().ln();
                    let merc_y_lo = merc_yc - half_y;
                    let merc_y_hi = merc_yc + half_y;
                    min_lat = (2.0 * merc_y_lo.exp().atan() - PI / 2.0).to_degrees();
                    max_lat = (2.0 * merc_y_hi.exp().atan() - PI / 2.0).to_degrees();
                    let clon_deg = clon.to_degrees();
                    min_lon = clon_deg - half_x.to_degrees();
                    max_lon = clon_deg + half_x.to_degrees();
                } else {
                    let (sin_sum, cos_sum) = sample_pts.iter().fold((0.0, 0.0), |(s, c), &(lon, _)| {
                        let r = lon.to_radians();
                        (s + r.sin(), c + r.cos())
                    });
                    let center_lon_avg = sin_sum.atan2(cos_sum).to_degrees();
                    min_lon = f64::MAX;
                    max_lon = f64::MIN;
                    min_lat = f64::MAX;
                    max_lat = f64::MIN;
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
                    if merc_blend > 0.0 {
                        let cw = surface_rotation.transpose() * nalgebra::Vector3::new(0.0, 0.0, 1.0);
                        let clat = cw.y.asin();
                        let clon = (-cw.z).atan2(cw.x);
                        let sf = 1.0 / clat.cos();
                        let half_y = sf / view_scale;
                        let half_x = sf * aspect / view_scale;
                        let merc_yc = (PI / 4.0 + clat / 2.0).tan().ln();
                        let merc_y_lo = merc_yc - half_y;
                        let merc_y_hi = merc_yc + half_y;
                        let mlat_min = (2.0 * merc_y_lo.exp().atan() - PI / 2.0).to_degrees();
                        let mlat_max = (2.0 * merc_y_hi.exp().atan() - PI / 2.0).to_degrees();
                        let clon_deg = clon.to_degrees();
                        let mlon_min = clon_deg - half_x.to_degrees();
                        let mlon_max = clon_deg + half_x.to_degrees();
                        if mlat_min < min_lat { min_lat = mlat_min; }
                        if mlat_max > max_lat { max_lat = mlat_max; }
                        if mlon_min < min_lon { min_lon = mlon_min; }
                        if mlon_max > max_lon { max_lon = mlon_max; }
                    }
                }
                let margin = 1.5;
                let lon_center = (min_lon + max_lon) / 2.0;
                let lat_center = (min_lat + max_lat) / 2.0;
                let tile_deg = 360.0 / (1u64 << camera_zoom_to_tile_zoom(v.zoom).max(2).min(18)) as f64;
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
                    let rows = (y_max - y_min + 1) as u32;
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

                    let lat_to_merc = |lat_deg: f64| -> f64 {
                        let lat_rad = lat_deg.to_radians();
                        (PI / 4.0 + lat_rad / 2.0).tan().ln()
                    };
                    let merc_top = lat_to_merc(top_left_lat);
                    let merc_bot = lat_to_merc(bot_right_lat);

                    let new_bounds = DetailBounds {
                        min_lon: top_left_lon,
                        max_lon: bot_right_lon,
                        min_lat: merc_bot,
                        max_lat: merc_top,
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
            let sim_seconds = dt * v.speed;
            for tab in &mut v.tabs {
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
                    if planet.satellite_cameras.len() == 1 {
                        let cam = &planet.satellite_cameras[0];
                        if let Some(cons) = planet.constellations.get(cam.constellation_idx) {
                            let planet_radius = planet.celestial_body.radius_km();
                            let planet_mu = planet.celestial_body.mu();
                            let planet_j2 = planet.celestial_body.j2();
                            let planet_eq_radius = planet.celestial_body.equatorial_radius_km();
                            let wc = cons.constellation(planet_radius, planet_mu, planet_j2, planet_eq_radius);
                            let positions = wc.satellite_positions(v.time);
                            if let Some(sat) = positions.iter().find(|s| s.plane == cam.plane && s.sat_index == cam.sat_index) {
                                let radial: Vector3<f64> = Vector3::new(sat.x, sat.y, sat.z).normalize();
                                let raan_spread = match cons.walker_type {
                                    WalkerType::Delta => 2.0 * PI,
                                    WalkerType::Star => PI,
                                };
                                let raan = raan_spread * cam.plane as f64 / cons.num_planes as f64;
                                let inc = cons.inclination.to_radians();
                                let orbital_normal: Vector3<f64> = Vector3::new(
                                    raan.sin() * inc.sin(),
                                    inc.cos(),
                                    -raan.cos() * inc.sin(),
                                );
                                let velocity_dir: Vector3<f64> = orbital_normal.cross(&radial).normalize();
                                let forward = velocity_dir;
                                let up = radial;
                                let right = forward.cross(&up).normalize();
                                v.rotation = Matrix3::new(
                                    right.x, right.y, right.z,
                                    up.x, up.y, up.z,
                                    forward.x, forward.y, forward.z,
                                );
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        TEXTURE_RESULT.with(|cell| {
            if let Some(result) = cell.borrow_mut().take() {
                if let Some(body) = v.pending_body {
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
            }
        });

        if !v.use_gpu_rendering {
            let rotation_changed = v.last_rotation.map_or(true, |r| r != v.rotation);
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
                            &format!("planet_{:?}_{:?}", key.0, key.1),
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

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.add_space(4.0);
                ui.heading("LEO Viz");
                if ui.button("[Info]").clicked() {
                    self.viewer.show_info = !self.viewer.show_info;
                }
                if ui.button("[Demo]").clicked() {
                    self.setup_demo();
                }
                if self.viewer.show_side_panel {
                    if ui.small_button("−").clicked() {
                        self.viewer.show_side_panel = false;
                    }
                } else {
                    if ui.small_button("+").clicked() {
                        self.viewer.show_side_panel = true;
                    }
                }
            });
        });

        if self.viewer.show_info {
            egui::Window::new("Info")
                .open(&mut self.viewer.show_info)
                .default_width(700.0)
                .show(ctx, |ui| {
                    ui.columns(2, |cols| {
                        cols[0].heading("Celestial Bodies");
                        let mut bodies: Vec<_> = CelestialBody::ALL.iter().collect();
                        bodies.sort_by(|a, b| b.radius_km().partial_cmp(&a.radius_km()).unwrap());
                        egui::Grid::new("bodies_grid").striped(true).show(&mut cols[0], |ui| {
                            ui.strong("Body");
                            ui.strong("Radius (km)");
                            ui.strong("μ (km³/s²)");
                            ui.strong("J₂ (×10⁻³)");
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
                        cols[0].heading("Constants & Variables");
                        egui::Grid::new("constants_grid").show(&mut cols[0], |ui| {
                            ui.strong("Symbol");
                            ui.strong("Meaning");
                            ui.end_row();
                            ui.monospace("G");
                            ui.label("Gravitational constant (6.674×10⁻¹¹ m³/kg/s²)");
                            ui.end_row();
                            ui.monospace("M");
                            ui.label("Mass of celestial body (kg)");
                            ui.end_row();
                            ui.monospace("μ");
                            ui.label("Standard gravitational parameter = G × M");
                            ui.end_row();
                            ui.monospace("r");
                            ui.label("Orbital radius = planet radius + altitude");
                            ui.end_row();
                            ui.monospace("v");
                            ui.label("Orbital velocity (km/s)");
                            ui.end_row();
                            ui.monospace("T");
                            ui.label("Orbital period (seconds)");
                            ui.end_row();
                            ui.monospace("c");
                            ui.label("Speed of light (299,792 km/s)");
                            ui.end_row();
                            ui.monospace("J₂");
                            ui.label("Second zonal harmonic (oblateness coefficient)");
                            ui.end_row();
                        });

                        cols[0].add_space(10.0);
                        cols[0].heading("J₂ Perturbation");
                        cols[0].label("J₂ describes how oblate (flattened) a body is.");
                        cols[0].label("It causes orbital planes (RAAN) to precess:");
                        cols[0].monospace("  dΩ/dt = -1.5 J₂ (Rₑ/a)² n cos(i)");
                        cols[0].label("where Rₑ = equatorial radius, a = semi-major axis,");
                        cols[0].label("n = mean motion, i = inclination.");
                        cols[0].add_space(5.0);
                        cols[0].label("Higher J₂ = faster precession. Jupiter/Saturn");
                        cols[0].label("have the highest J₂ due to rapid rotation.");

                        cols[0].add_space(10.0);
                        cols[0].heading("Orbital Mechanics");
                        cols[0].label("Orbital velocity:");
                        cols[0].monospace("  v = √(μ / r)");
                        cols[0].label("Orbital period:");
                        cols[0].monospace("  T = 2π √(r³ / μ)");
                        cols[0].label("One-way latency:");
                        cols[0].monospace("  t = distance / c");

                        cols[1].heading("Walker Constellation");
                        cols[1].label("Notation: i:T/P/F");
                        egui::Grid::new("walker_grid").show(&mut cols[1], |ui| {
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
                        cols[1].add_space(5.0);
                        cols[1].label("Types:");
                        cols[1].label("  Delta: 360° RAAN spread (co-rotating)");
                        cols[1].label("  Star: 180° RAAN spread (counter-rotating)");
                        cols[1].label("Phasing offset per plane:");
                        cols[1].monospace("  Δ = F × 360° / T");

                        cols[1].add_space(10.0);
                        cols[1].heading("Satellite Constellations");
                        egui::Grid::new("constellations_grid").striped(true).show(&mut cols[1], |ui| {
                            ui.strong("Name");
                            ui.strong("Config");
                            ui.strong("Alt");
                            ui.strong("Inc");
                            ui.end_row();
                            ui.label("Starlink");
                            ui.label("22×72");
                            ui.label("550km");
                            ui.label("53°");
                            ui.end_row();
                            ui.label("OneWeb");
                            ui.label("49×36");
                            ui.label("1200km");
                            ui.label("87.9°");
                            ui.end_row();
                            ui.label("Iridium");
                            ui.label("11×6");
                            ui.label("780km");
                            ui.label("86.4°");
                            ui.end_row();
                            ui.label("Kuiper");
                            ui.label("34×34");
                            ui.label("630km");
                            ui.label("51.9°");
                            ui.end_row();
                            ui.label("Iris²");
                            ui.label("22×12");
                            ui.label("7800km");
                            ui.label("75°");
                            ui.end_row();
                            ui.label("Telesat");
                            ui.label("13×6");
                            ui.label("1015km");
                            ui.label("99°");
                            ui.end_row();
                        });

                        cols[1].add_space(10.0);
                        cols[1].heading("Live TLE Sources");
                        cols[1].label("From CelesTrak: Starlink, OneWeb,");
                        cols[1].label("Iridium, Globalstar, Orbcomm, GPS,");
                        cols[1].label("Galileo, GLONASS, Beidou");
                    });
                });
        }

        if self.viewer.show_side_panel {
            egui::SidePanel::left("settings_panel")
                .resizable(true)
                .default_width(200.0)
                .show(ctx, |ui| {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.add_space(4.0);
                        ui.strong("Settings");
                    });
                    ui.separator();
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        self.viewer.show_settings(ui);
                    });
                });
        }

        DockArea::new(&mut self.dock_state)
            .show_add_buttons(true)
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

                    for camera in &planet.satellite_cameras {
                        let sat_data = planet.constellations.get(camera.constellation_idx).map(|cons| {
                            let wc = cons.constellation(pr, pm, pj2, peq);
                            let positions = wc.satellite_positions(self.viewer.time);
                            positions.iter()
                                .find(|s| s.plane == camera.plane && s.sat_index == camera.sat_index)
                                .map(|s| (s.lat, s.lon, cons.altitude_km, texture, pr))
                        }).flatten();

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

fn compute_manhattan_path(
    src_plane: usize, src_sat: usize,
    dst_plane: usize, dst_sat: usize,
    num_planes: usize, sats_per_plane: usize,
    is_star: bool,
) -> Vec<(usize, usize)> {
    let mut path = vec![(src_plane, src_sat)];

    let (plane_dir, plane_steps) = if is_star {
        if dst_plane >= src_plane {
            (1i32, dst_plane - src_plane)
        } else {
            (-1i32, src_plane - dst_plane)
        }
    } else {
        let plane_diff_fwd = (dst_plane + num_planes - src_plane) % num_planes;
        let plane_diff_bwd = (src_plane + num_planes - dst_plane) % num_planes;
        if plane_diff_fwd <= plane_diff_bwd {
            (1i32, plane_diff_fwd)
        } else {
            (-1i32, plane_diff_bwd)
        }
    };

    let sat_diff_fwd = (dst_sat + sats_per_plane - src_sat) % sats_per_plane;
    let sat_diff_bwd = (src_sat + sats_per_plane - dst_sat) % sats_per_plane;
    let (sat_dir, sat_steps) = if sat_diff_fwd <= sat_diff_bwd {
        (1i32, sat_diff_fwd)
    } else {
        (-1i32, sat_diff_bwd)
    };

    let mut cur_plane = src_plane;
    for _ in 0..plane_steps {
        cur_plane = ((cur_plane as i32 + plane_dir + num_planes as i32) % num_planes as i32) as usize;
        path.push((cur_plane, src_sat));
    }

    let mut cur_sat = src_sat;
    for _ in 0..sat_steps {
        cur_sat = ((cur_sat as i32 + sat_dir + sats_per_plane as i32) % sats_per_plane as i32) as usize;
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

    let (plane_dir, mut plane_steps_remaining) = if is_star {
        if dst_plane >= src_plane {
            (1i32, dst_plane - src_plane)
        } else {
            (-1i32, src_plane - dst_plane)
        }
    } else {
        let plane_diff_fwd = (dst_plane + num_planes - src_plane) % num_planes;
        let plane_diff_bwd = (src_plane + num_planes - dst_plane) % num_planes;
        if plane_diff_fwd <= plane_diff_bwd {
            (1i32, plane_diff_fwd)
        } else {
            (-1i32, plane_diff_bwd)
        }
    };

    let sat_diff_fwd = (dst_sat + sats_per_plane - src_sat) % sats_per_plane;
    let sat_diff_bwd = (src_sat + sats_per_plane - dst_sat) % sats_per_plane;
    let (sat_dir, mut sat_steps_remaining) = if sat_diff_fwd <= sat_diff_bwd {
        (1i32, sat_diff_fwd)
    } else {
        (-1i32, sat_diff_bwd)
    };

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
            cur_sat = ((cur_sat as i32 + sat_dir + sats_per_plane as i32) % sats_per_plane as i32) as usize;
            sat_steps_remaining -= 1;
            path.push((cur_plane, cur_sat));
            continue;
        }
        if sat_steps_remaining == 0 {
            cur_plane = ((cur_plane as i32 + plane_dir + num_planes as i32) % num_planes as i32) as usize;
            plane_steps_remaining -= 1;
            path.push((cur_plane, cur_sat));
            continue;
        }

        let next_plane = ((cur_plane as i32 + plane_dir + num_planes as i32) % num_planes as i32) as usize;
        let next_sat = ((cur_sat as i32 + sat_dir + sats_per_plane as i32) % sats_per_plane as i32) as usize;

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
    constellations: &[(WalkerConstellation, Vec<SatelliteState>, usize, bool, usize, String)],
    show_orbits: bool,
    show_axes: bool,
    show_coverage: bool,
    coverage_angle: f64,
    mut rotation: Matrix3<f64>,
    satellite_rotation: Matrix3<f64>,
    width: f32,
    height: f32,
    earth_texture: Option<&egui::TextureHandle>,
    mut zoom: f64,
    sat_radius: f32,
    show_links: bool,
    show_intra_links: bool,
    hide_behind_earth: bool,
    single_color: bool,
    dark_mode: bool,
    pending_cameras: &mut Vec<SatelliteCamera>,
    camera_id_counter: &mut usize,
    satellite_cameras: &mut [SatelliteCamera],
    cameras_to_remove: &mut Vec<usize>,
    show_routing_paths: bool,
    show_manhattan_path: bool,
    show_shortest_path: bool,
    show_asc_desc_colors: bool,
    show_altitude_lines: bool,
    planet_radius: f64,
    render_planet: bool,
    link_width: f32,
    fixed_sizes: bool,
    flattening: f64,
    show_polar_circle: bool,
    show_equator: bool,
    show_terminator: bool,
    sphere_renderer: Option<&Arc<Mutex<SphereRenderer>>>,
    body_key: (CelestialBody, Skin, TextureResolution),
    body_rotation: &Matrix3<f64>,
    earth_fixed_camera: bool,
    use_gpu_rendering: bool,
    show_clouds: bool,
    show_day_night: bool,
    sun_dir: [f32; 3],
    time: f64,
    ground_stations: &mut [GroundStation],
    areas_of_interest: &mut [AreaOfInterest],
    body_rot_angle: f64,
    dragging_place: &mut Option<(usize, usize, bool, usize)>,
    drag_tab_planet: (usize, usize),
    detail_gl_info: Option<(glow::Texture, [f32; 4])>,
    show_stars: bool,
    show_milky_way: bool,
) -> (Matrix3<f64>, f64) {
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

        let merc_blend = ((zoom - 80.0) / 40.0).clamp(0.0, 1.0) as f32;
        let center_world = combined_rotation.transpose() * Vector3::new(0.0, 0.0, 1.0);
        let center_lat_val = center_world.y.asin() as f32;
        let center_lon_val = (-center_world.z).atan2(center_world.x) as f32;

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
                    r.paint(gl, key, &inv_rotation, flat as f64, aspect, scale, atmosphere, show_clouds, show_day_night, sun_dir, Some(&dt), merc_blend, center_lat_val, center_lon_val, show_stars, show_milky_way);
                } else {
                    r.paint(gl, key, &inv_rotation, flat as f64, aspect, scale, atmosphere, show_clouds, show_day_night, sun_dir, None, merc_blend, center_lat_val, center_lon_val, show_stars, show_milky_way);
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
            for (constellation, _, color_offset, is_tle, _, _) in constellations {
                if *is_tle { continue; }
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
            for (constellation, positions, color_offset, _is_tle, _, _) in constellations {
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
            for (constellation, positions, color_offset, _is_tle, _, _) in constellations {
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
            for (constellation, _, color_offset, is_tle, _, _) in constellations {
                if *is_tle { continue; }
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

                        for (cidx, (cons, positions, _, is_tle, _, _)) in constellations.iter().enumerate() {
                            if *is_tle { continue; }
                            for sat in positions.iter() {
                                if let Some(asc) = ascending_filter {
                                    if sat.ascending != asc { continue; }
                                }
                                let dist = haversine_dist(sat);
                                if dist <= max_angular_dist {
                                    if best.is_none() || dist < best.as_ref().unwrap().4 {
                                        best = Some((cidx, cons, positions, sat, dist));
                                    }
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

        for (constellation, positions, color_offset, is_tle, orig_idx, _) in constellations {
            if *is_tle {
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
                        } else {
                            if sat.ascending {
                                egui::Color32::from_rgb(180, 140, 80)
                            } else {
                                egui::Color32::from_rgb(80, 120, 180)
                            }
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
                        if *is_tle {
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
                        if *is_tle {
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

    if constellations.len() > 1 {
        let plot_rect = response.response.rect;
        let mut y = plot_rect.min.y + 8.0;
        let x = plot_rect.min.x + 8.0;
        let font = egui::FontId::proportional(12.0);
        let square_size = 10.0;
        let mut seen = std::collections::HashSet::new();
        for (_, _, color_offset, _, _, name) in constellations {
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
            ui.painter().galley(text_pos, galley, egui::Color32::WHITE);
            y += 16.0;
        }
    }

    for (constellation, positions, color_offset, _is_tle, orig_idx, _) in constellations {
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
                    let id = match &sat.name {
                        Some(name) => name.clone(),
                        None => format!("P{}S{}", sat.plane, sat.sat_index),
                    };
                    let text = format!(
                        "{}  {:.1}° {:.1}°\n{:.0} km  {:.2} km/s",
                        id,
                        sat.lat, sat.lon,
                        alt_km, vel_km_s,
                    );
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
                let earth_r_sq = (planet_radius * 0.95).powi(2) as f64;
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
            let margin = (response.transform.bounds().width() / 2.0).max(planet_radius);
            let r = margin;
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

    if !response.response.dragged() {
        if dragging_place.map_or(false, |(t, p, _, _)| t == drag_tab_planet.0 && p == drag_tab_planet.1) {
            *dragging_place = None;
        }
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
                    let earth_r_sq = (planet_radius * 0.95).powi(2) as f64;
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
    constellations: &[(WalkerConstellation, Vec<SatelliteState>, usize, bool, usize, String)],
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
        for (constellation, positions, color_offset, _is_tle, _, _) in constellations {
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
    constellations: &[(WalkerConstellation, Vec<SatelliteState>, usize, bool, usize, String)],
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
        let minor = (major * (1.0 - cos_i) / (1.0 + cos_i)).max(0.05);
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

        let torus_point = |theta: f64, phi: f64| -> (f64, f64, f64) {
            let r = major_radius + minor_radius * phi.cos();
            let y = minor_radius * phi.sin();
            let x = r * theta.cos();
            let z = r * theta.sin();
            rotate_point_matrix(x, y, z, &display_rotation)
        };

        for (_cidx, (constellation, positions, color_offset, _is_tle, orig_idx, _)) in constellations.iter().enumerate() {
            let sats_per_plane = constellation.total_sats / constellation.num_planes;
            let orbit_radius = constellation.planet_radius + constellation.altitude_km;
            let period = 2.0 * PI * (orbit_radius.powi(3) / constellation.planet_mu).sqrt();
            let mean_motion = 2.0 * PI / period;

            let torus_pos = |plane: usize, sat_idx: usize| -> (f64, f64, f64) {
                let angle = 2.0 * PI * plane as f64 / constellation.num_planes as f64;
                let sat_spacing = 2.0 * PI * sat_idx as f64 / sats_per_plane as f64;
                let phase = sat_spacing + mean_motion * time;
                torus_point(angle, phase)
            };

            for plane in 0..constellation.num_planes {
                let angle = 2.0 * PI * plane as f64 / constellation.num_planes as f64;
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
                    let (rx, ry, _) = torus_point(angle, phase);
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
                        let angle1 = 2.0 * PI * sat.plane as f64 / constellation.num_planes as f64;
                        let angle2 = 2.0 * PI * neighbor.plane as f64 / constellation.num_planes as f64;
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
                let angle = 2.0 * PI * plane as f64 / constellation.num_planes as f64;

                for sat in positions.iter().filter(|s| s.plane == plane) {
                    let is_tracked = satellite_cameras.iter().any(|c|
                        c.constellation_idx == *orig_idx && c.plane == sat.plane && c.sat_index == sat.sat_index
                    );
                    let color = if show_asc_desc_colors {
                        if is_tracked {
                            if sat.ascending { COLOR_ASCENDING } else { COLOR_DESCENDING }
                        } else {
                            if sat.ascending {
                                egui::Color32::from_rgb(180, 140, 80)
                            } else {
                                egui::Color32::from_rgb(80, 120, 180)
                            }
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
                let inclination_factor = inclination_rad.sin().abs().max(0.1);
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

            let torus_point = |theta: f64, phi: f64| -> (f64, f64, f64) {
                let r = major_radius + minor_radius * phi.cos();
                let y = minor_radius * phi.sin();
                let x = r * theta.cos();
                let z = r * theta.sin();
                rotate_point_matrix(x, y, z, &display_rotation)
            };

            'outer: for (_cidx, (constellation, positions, _, _, orig_idx, _)) in constellations.iter().enumerate() {
                let sats_per_plane = constellation.total_sats / constellation.num_planes;
                let orbit_radius = constellation.planet_radius + constellation.altitude_km;
                let period = 2.0 * PI * (orbit_radius.powi(3) / constellation.planet_mu).sqrt();
                let mean_motion = 2.0 * PI / period;

                for sat in positions {
                    let angle = 2.0 * PI * sat.plane as f64 / constellation.num_planes as f64;
                    let phase = 2.0 * PI * sat.sat_index as f64 / sats_per_plane as f64 + mean_motion * time;
                    let (tx, ty, _) = torus_point(angle, phase);

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

const COLORS: [egui::Color32; 8] = [
    egui::Color32::from_rgb(255, 99, 71),
    egui::Color32::from_rgb(50, 205, 50),
    egui::Color32::from_rgb(30, 144, 255),
    egui::Color32::from_rgb(255, 215, 0),
    egui::Color32::from_rgb(238, 130, 238),
    egui::Color32::from_rgb(0, 206, 209),
    egui::Color32::from_rgb(255, 140, 0),
    egui::Color32::from_rgb(147, 112, 219),
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
