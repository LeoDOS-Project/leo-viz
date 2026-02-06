//! Map tile caching and quadtree management.
//!
//! Implements a quadtree-based tile cache for high-resolution Earth imagery
//! with LRU eviction. Supports background tile fetching and disk caching.

use eframe::glow;
use std::collections::HashSet;
use std::f64::consts::PI;
use std::sync::mpsc;

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub struct TileCoord {
    pub x: u32,
    pub y: u32,
    pub z: u8,
}

#[derive(Clone)]
pub struct DetailBounds {
    pub min_lon: f64,
    pub max_lon: f64,
    pub min_lat: f64,
    pub max_lat: f64,
}

pub struct DetailTexture {
    pub width: u32,
    pub height: u32,
    pub bounds: DetailBounds,
    pub gl_texture: Option<glow::Texture>,
}

pub struct TileFetchResult {
    pub coord: TileCoord,
    pub pixels: Vec<[u8; 3]>,
    pub width: u32,
    pub height: u32,
}

pub struct TileCacheEntry {
    pub pixels: Vec<[u8; 3]>,
    pub width: u32,
    pub height: u32,
}

pub struct TileNode {
    pub tile: Option<TileCacheEntry>,
    pub children: [Option<Box<TileNode>>; 4],
    pub last_used: u64,
}

impl TileNode {
    pub fn new() -> Self {
        TileNode { tile: None, children: [None, None, None, None], last_used: 0 }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.iter().all(|c| c.is_none())
    }
}

pub struct TileQuadTree {
    pub root: TileNode,
    pub tile_count: usize,
    pub max_tiles: usize,
    pub access_counter: u64,
}

impl TileQuadTree {
    pub fn new(max_tiles: usize) -> Self {
        TileQuadTree { root: TileNode::new(), tile_count: 0, max_tiles, access_counter: 0 }
    }

    pub fn child_index(x: u32, y: u32, z: u8, depth: u8) -> usize {
        let bit_x = ((x >> (z - 1 - depth)) & 1) as usize;
        let bit_y = ((y >> (z - 1 - depth)) & 1) as usize;
        bit_x | (bit_y << 1)
    }

    pub fn insert(&mut self, coord: TileCoord, entry: TileCacheEntry) {
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

    pub fn best_tile_zoom(&mut self, coord: &TileCoord) -> Option<u8> {
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

    pub fn get_tile_at(&self, coord: &TileCoord) -> Option<&TileCacheEntry> {
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

    pub fn has_tile(&self, coord: &TileCoord) -> bool {
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

pub struct TileOverlayState {
    pub enabled: bool,
    pub tile_tree: TileQuadTree,
    #[cfg(not(target_arch = "wasm32"))]
    pub disk_cache_dir: std::path::PathBuf,
    pub detail_texture: Option<DetailTexture>,
    #[cfg(not(target_arch = "wasm32"))]
    pub fetch_tx: mpsc::Sender<(TileCoord, std::path::PathBuf, u64)>,
    pub result_rx: mpsc::Receiver<TileFetchResult>,
    pub last_zoom: u8,
    pub pending_tiles: HashSet<TileCoord>,
    pub needed_tiles: Vec<TileCoord>,
    pub dirty: bool,
    #[cfg(not(target_arch = "wasm32"))]
    pub fetch_generation: std::sync::Arc<std::sync::atomic::AtomicU64>,
    pub generation: u64,
    pub tile_x_origin: u32,
    pub last_compose: std::time::Instant,
    pub base_fetched: bool,
    pub compose_buffer: Vec<[u8; 4]>,
}

pub fn lon_lat_to_tile(lon: f64, lat: f64, z: u8) -> TileCoord {
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

pub fn tile_to_lon_lat(t: &TileCoord) -> (f64, f64) {
    let n = (1u32 << t.z) as f64;
    let lon = t.x as f64 / n * 360.0 - 180.0;
    let lat = (PI * (1.0 - 2.0 * t.y as f64 / n)).sinh().atan().to_degrees();
    (lon, lat)
}

pub fn camera_zoom_to_tile_zoom(camera_zoom: f64) -> u8 {
    let z = (camera_zoom.log2() + 4.0).floor() as i32;
    z.clamp(0, 18) as u8
}
