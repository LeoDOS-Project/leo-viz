//! Map tile caching and quadtree management.
//!
//! Implements a quadtree-based tile cache for high-resolution Earth imagery
//! with LRU eviction. Supports background tile fetching and disk caching.

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::sync::mpsc;

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug, serde::Serialize)]
pub struct TileCoord {
    pub x: u32,
    pub y: u32,
    pub z: u8,
}

impl TileCoord {
    pub fn parent(self) -> Option<Self> {
        (self.z > 0).then(|| Self {
            x: self.x >> 1,
            y: self.y >> 1,
            z: self.z - 1,
        })
    }
}

pub const MAX_TILE_ZOOM: u8 = 23;
const MISSING_TILE_TTL: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);
const MISSING_TILE_CACHE_LIMIT: usize = 16_384;

#[derive(Clone)]
pub struct DetailBounds {
    pub min_lon: f64,
    pub max_lon: f64,
    pub min_lat: f64,
    pub max_lat: f64,
}

impl DetailBounds {
    pub fn mercator_bounds(&self) -> [f64; 4] {
        [
            self.min_lon,
            self.max_lon,
            self.min_lat.tan().asinh(),
            self.max_lat.tan().asinh(),
        ]
    }

    #[cfg(test)]
    pub fn shader_bounds(&self) -> [f32; 4] {
        // The atlas is Web Mercator, not an equirectangular latitude image.
        self.mercator_bounds().map(|value| value as f32)
    }
}

pub struct DetailTexture {
    pub bounds: DetailBounds,
    pub layout: AtlasLayout,
    pub previous_bounds: Option<DetailBounds>,
    pub fade_started: web_time::Instant,
}

impl DetailTexture {
    pub fn blend(&self) -> f32 {
        fade_blend(self.fade_started.elapsed().as_secs_f32())
    }
}

fn fade_blend(elapsed: f32) -> f32 {
    let t = (elapsed / 0.15).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub const TILE_SIZE: u32 = 256;
const TILE_UPLOADS_PER_FRAME: usize = 8;
pub const GPU_TILE_CACHE_LIMIT: usize = 512;
pub const TILE_RESAMPLES_PER_FRAME: usize = 32;

pub enum TileReuse {
    Reused,
    Missing,
    Deferred,
}

pub trait TileAtlasWriter {
    fn reuse(&mut self, coord: TileCoord, source: TileCoord, origin: [u32; 2]) -> TileReuse;
    fn upload(&mut self, coord: TileCoord, source: TileCoord, origin: [u32; 2], rgba: &[[u8; 4]]);
}

#[cfg(test)]
impl<F: FnMut([u32; 2], &[[u8; 4]])> TileAtlasWriter for F {
    fn reuse(&mut self, _: TileCoord, _: TileCoord, _: [u32; 2]) -> TileReuse {
        TileReuse::Missing
    }

    fn upload(&mut self, _: TileCoord, _: TileCoord, origin: [u32; 2], rgba: &[[u8; 4]]) {
        self(origin, rgba);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct AtlasLayout {
    pub x: u32,
    pub y: u32,
    pub z: u8,
    pub cols: u32,
    pub rows: u32,
}

pub struct AtlasCopy {
    pub source: [u32; 2],
    pub destination: [u32; 2],
    pub size: [u32; 2],
}

impl AtlasLayout {
    pub fn new(tiles: &[TileCoord], x: u32) -> Self {
        let z = tiles[0].z;
        let n = 1 << z;
        let y = tiles.iter().map(|c| c.y).min().unwrap();
        Self {
            x,
            y,
            z,
            cols: tiles.iter().map(|c| (c.x + n - x) % n).max().unwrap() + 1,
            rows: tiles.iter().map(|c| c.y).max().unwrap() - y + 1,
        }
    }

    pub fn size(self) -> [u32; 2] {
        [self.cols * TILE_SIZE, self.rows * TILE_SIZE]
    }

    fn column(self, x: u32) -> u32 {
        (x + (1 << self.z) - self.x) % (1 << self.z)
    }

    pub(crate) fn contains(self, c: TileCoord) -> bool {
        c.z == self.z && self.column(c.x) < self.cols && c.y >= self.y && c.y < self.y + self.rows
    }

    pub fn bounds(self) -> DetailBounds {
        let (west, north) = tile_to_lon_lat(&TileCoord {
            x: self.x,
            y: self.y,
            z: self.z,
        });
        let (_, south) = tile_to_lon_lat(&TileCoord {
            x: 0,
            y: self.y + self.rows,
            z: self.z,
        });
        DetailBounds {
            min_lon: west,
            max_lon: west + self.cols as f64 * 360.0 / (1u32 << self.z) as f64,
            min_lat: south.to_radians(),
            max_lat: north.to_radians(),
        }
    }

    /// Copy shared geographic tiles, splitting at the longitude wrap if necessary.
    pub fn copies_from(self, old: Self) -> Vec<AtlasCopy> {
        if self.z != old.z {
            return Vec::new();
        }
        let top = self.y.max(old.y);
        let bottom = (self.y + self.rows).min(old.y + old.rows);
        if top >= bottom {
            return Vec::new();
        }
        let n = 1 << self.z;
        let mut copies = Vec::new();
        let mut col = 0;
        while col < self.cols {
            let source = old.column((self.x + col) % n);
            if source >= old.cols {
                col += 1;
                continue;
            }
            let start = col;
            col += 1;
            while col < self.cols
                && source + col - start < old.cols
                && old.column((self.x + col) % n) == source + col - start
            {
                col += 1;
            }
            copies.push(AtlasCopy {
                source: [source * TILE_SIZE, (top - old.y) * TILE_SIZE],
                destination: [start * TILE_SIZE, (top - self.y) * TILE_SIZE],
                size: [(col - start) * TILE_SIZE, (bottom - top) * TILE_SIZE],
            });
        }
        copies
    }
}

#[derive(Default)]
pub struct DetailAtlas {
    pub layout: Option<AtlasLayout>,
    sources: HashMap<TileCoord, TileCoord>,
    buffer: Vec<[u8; 4]>,
}

impl DetailAtlas {
    pub fn needs_updates(&self, tree: &mut TileQuadTree, needed: &[TileCoord]) -> bool {
        needed.iter().any(|coord| {
            tree.best_tile_zoom(coord)
                .is_some_and(|z| self.sources.get(coord).is_none_or(|source| source.z < z))
        })
    }

    pub fn covers(&self, needed: &[TileCoord]) -> bool {
        needed.iter().all(|coord| self.sources.contains_key(coord))
    }

    pub fn set_layout(&mut self, layout: AtlasLayout) {
        self.sources.retain(|coord, _| layout.contains(*coord));
        self.layout = Some(layout);
    }

    /// GPU-resident tiles do not consume the per-frame CPU conversion/upload budget.
    pub fn update_tiles(
        &mut self,
        tree: &mut TileQuadTree,
        needed: &[TileCoord],
        mut writer: impl TileAtlasWriter,
    ) -> bool {
        let layout = self.layout.unwrap();
        let mut updated = 0;
        let mut pending = false;
        for &coord in needed {
            let Some(z) = tree.best_tile_zoom(&coord) else {
                continue;
            };
            if self.sources.get(&coord).is_some_and(|source| source.z >= z) {
                continue;
            }
            let d = coord.z - z;
            let source = TileCoord {
                x: coord.x >> d,
                y: coord.y >> d,
                z,
            };
            let origin = [
                layout.column(coord.x) * TILE_SIZE,
                (coord.y - layout.y) * TILE_SIZE,
            ];
            match writer.reuse(coord, source, origin) {
                TileReuse::Reused => {
                    self.sources.insert(coord, source);
                    continue;
                }
                TileReuse::Deferred => {
                    pending = true;
                    continue;
                }
                TileReuse::Missing => {}
            }
            if updated == TILE_UPLOADS_PER_FRAME {
                pending = true;
                continue;
            }
            let entry = tree.get_tile_at(&source).unwrap();
            if entry.width == 0
                || entry.height == 0
                || entry.pixels.len() != (entry.width as usize * entry.height as usize)
            {
                continue;
            }
            self.buffer.resize((TILE_SIZE * TILE_SIZE) as usize, [0; 4]);
            if d == 0 && entry.width == TILE_SIZE && entry.height == TILE_SIZE {
                for (dst, &[r, g, b]) in self.buffer.iter_mut().zip(&entry.pixels) {
                    *dst = [r, g, b, 255];
                }
            } else {
                let mask = (1u32 << d) - 1;
                let ox = (coord.x & mask) as u64 * TILE_SIZE as u64;
                let oy = (coord.y & mask) as u64 * TILE_SIZE as u64;
                let columns: [usize; TILE_SIZE as usize] = std::array::from_fn(|col| {
                    (((ox + col as u64) * entry.width as u64) >> (d + 8)) as usize
                });
                for row in 0..TILE_SIZE as usize {
                    let sy = (((oy + row as u64) * entry.height as u64) >> (d + 8)) as usize;
                    for (col, &sx) in columns.iter().enumerate() {
                        let [r, g, b] = entry.pixels[sy * entry.width as usize + sx];
                        self.buffer[row * TILE_SIZE as usize + col] = [r, g, b, 255];
                    }
                }
            }
            writer.upload(coord, source, origin, &self.buffer);
            self.sources.insert(coord, source);
            updated += 1;
        }
        pending
    }
}

pub struct TileFetchResult {
    pub coord: TileCoord,
    pub pixels: Vec<[u8; 3]>,
    pub width: u32,
    pub height: u32,
    pub missing: bool,
    pub cancel_if_stale: bool,
}

impl TileFetchResult {
    pub fn empty(request: &TileFetchRequest) -> Self {
        Self {
            coord: request.coord,
            pixels: Vec::new(),
            width: 0,
            height: 0,
            missing: false,
            cancel_if_stale: request.cancel_if_stale,
        }
    }
}

pub fn fetch_tile(request: &TileFetchRequest, agent: &ureq::Agent) -> TileFetchResult {
    fetch_tile_from(
        request,
        agent,
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer",
    )
}

fn fetch_tile_from(
    request: &TileFetchRequest,
    agent: &ureq::Agent,
    service: &str,
) -> TileFetchResult {
    use std::io::Read;
    let coord = request.coord;
    let mut result = TileFetchResult::empty(request);
    // Older experiments may have cached placeholder JPEGs above level 18.
    let suffix = if coord.z > 18 { "imagery.jpg" } else { "jpg" };
    let directory = request
        .cache_dir
        .join(coord.z.to_string())
        .join(coord.y.to_string());
    let cache_path = directory.join(format!("{}.{suffix}", coord.x));
    let missing_path = directory.join(format!("{}.missing", coord.x));
    if std::fs::metadata(&missing_path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .is_some_and(|age| age < MISSING_TILE_TTL)
    {
        result.missing = true;
        if let Some(trace) = &request.trace {
            trace.record("missing_cache_hit", serde_json::json!({}));
        }
        return result;
    }
    let decode = |bytes: &[u8]| {
        let started = std::time::Instant::now();
        let decoded = crate::texture::decode_jpeg_pixels(bytes)
            .filter(|(_, width, height)| *width == TILE_SIZE && *height == TILE_SIZE);
        if let Some(trace) = &request.trace {
            trace.record(
                "jpeg_decode",
                serde_json::json!({
                    "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
                    "success": decoded.is_some(),
                }),
            );
        }
        decoded
    };
    let started = std::time::Instant::now();
    let cache_file_exists = cache_path.exists();
    let mut fetched = std::fs::read(&cache_path)
        .ok()
        .and_then(|bytes| decode(&bytes));
    if let Some(trace) = &request.trace {
        trace.record(
            "disk_lookup",
            serde_json::json!({
                "hit": fetched.is_some(), "file_exists": cache_file_exists,
                "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
            }),
        );
    }
    if fetched.is_none() {
        // Missing imagery otherwise arrives as a successful JPEG placeholder.
        let url = format!(
            "{service}/tile/{}/{}/{}?blankTile=false",
            coord.z, coord.y, coord.x
        );
        let started = std::time::Instant::now();
        if let Some(trace) = &request.trace {
            trace.record("http_request", serde_json::json!({"url": url}));
        }
        match agent.get(&url).call() {
            Ok(response)
                if response
                    .header("blank-tile")
                    .is_some_and(|value| value.eq_ignore_ascii_case("true")) =>
            {
                result.missing = true;
            }
            Ok(response) => {
                let mut bytes = Vec::new();
                if response
                    .into_reader()
                    .take(2 * 1024 * 1024)
                    .read_to_end(&mut bytes)
                    .is_ok()
                {
                    fetched = decode(&bytes);
                    if fetched.is_some() {
                        let _ = std::fs::create_dir_all(&directory);
                        let _ = std::fs::write(&cache_path, bytes);
                        let _ = std::fs::remove_file(&missing_path);
                    }
                }
            }
            Err(ureq::Error::Status(404, _)) => result.missing = true,
            Err(error) => {
                if let Some(trace) = &request.trace {
                    trace.record(
                        "http_error",
                        serde_json::json!({"error": error.to_string()}),
                    );
                }
            }
        }
        if result.missing {
            let _ = std::fs::create_dir_all(&directory);
            let _ = std::fs::write(&missing_path, []);
        }
        if let Some(trace) = &request.trace {
            trace.record(
                "http_result",
                serde_json::json!({
                    "success": fetched.is_some(), "missing": result.missing,
                    "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
                }),
            );
        }
    }
    if let Some((pixels, width, height)) = fetched {
        result.pixels = pixels;
        result.width = width;
        result.height = height;
    }
    result
}

pub struct TileCacheEntry {
    pub pixels: Vec<[u8; 3]>,
    pub width: u32,
    pub height: u32,
}

pub struct TileNode {
    pub tile: Option<std::sync::Arc<TileCacheEntry>>,
    pub children: [Option<Box<TileNode>>; 4],
    pub last_used: u64,
}

impl TileNode {
    pub fn new() -> Self {
        TileNode {
            tile: None,
            children: [None, None, None, None],
            last_used: 0,
        }
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
        TileQuadTree {
            root: TileNode::new(),
            tile_count: 0,
            max_tiles,
            access_counter: 0,
        }
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
        node.tile = Some(std::sync::Arc::new(entry));
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
        self.shared_tile_at(coord).map(std::sync::Arc::as_ref)
    }

    pub fn shared_tile_at(&self, coord: &TileCoord) -> Option<&std::sync::Arc<TileCacheEntry>> {
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
    pub fetch_tx: mpsc::Sender<TileFetchRequest>,
    pub result_rx: mpsc::Receiver<TileFetchResult>,
    pub last_zoom: u8,
    pub pending_tiles: HashSet<TileCoord>,
    pub missing_tiles: HashMap<TileCoord, std::time::Instant>,
    pub needed_tiles: Vec<TileCoord>,
    pub dirty: bool,
    #[cfg(not(target_arch = "wasm32"))]
    pub fetch_generation: std::sync::Arc<std::sync::atomic::AtomicU64>,
    pub generation: u64,
    pub tile_x_origin: u32,
    pub base_fetched: bool,
    pub atlas: DetailAtlas,
    pub diagnostics: Option<std::sync::Arc<crate::tile_diagnostics::Trace>>,
}

pub struct TileFetchRequest {
    pub coord: TileCoord,
    pub cache_dir: std::path::PathBuf,
    pub generation: u64,
    pub cancel_if_stale: bool,
    pub trace: Option<crate::tile_diagnostics::RequestTrace>,
}

impl TileOverlayState {
    pub fn queue_tile(&mut self, coord: TileCoord) {
        self.queue_tile_request(coord, true);
    }

    pub fn remember_missing(&mut self, coord: TileCoord) {
        if self.missing_tiles.len() >= MISSING_TILE_CACHE_LIMIT {
            self.missing_tiles
                .retain(|_, time| time.elapsed() < MISSING_TILE_TTL);
            if let Some(oldest) = self
                .missing_tiles
                .iter()
                .min_by_key(|(_, time)| **time)
                .map(|(coord, _)| *coord)
            {
                self.missing_tiles.remove(&oldest);
            }
        }
        self.missing_tiles.insert(coord, std::time::Instant::now());
    }

    pub fn queue_tile_request(&mut self, mut coord: TileCoord, cancel_if_stale: bool) {
        while self
            .missing_tiles
            .get(&coord)
            .is_some_and(|time| time.elapsed() < MISSING_TILE_TTL)
        {
            let Some(parent) = coord.parent() else { return };
            coord = parent;
        }
        if self.tile_tree.has_tile(&coord) || !self.pending_tiles.insert(coord) {
            return;
        }
        let _ = self.fetch_tx.send(TileFetchRequest {
            coord,
            cache_dir: self.disk_cache_dir.clone(),
            generation: self.generation,
            cancel_if_stale,
            trace: self
                .diagnostics
                .as_ref()
                .map(|trace| trace.request(coord, self.generation)),
        });
    }
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
    let lat = (PI * (1.0 - 2.0 * t.y as f64 / n))
        .sinh()
        .atan()
        .to_degrees();
    (lon, lat)
}

pub fn camera_zoom_to_tile_zoom(camera_zoom: f64) -> u8 {
    let z = (camera_zoom.log2() + 4.0).floor() as i32;
    z.clamp(0, MAX_TILE_ZOOM as i32) as u8
}

pub fn stable_tile_zoom(camera_zoom: f64, previous: u8) -> u8 {
    let level = camera_zoom.log2() + 4.0;
    if level >= previous as f64 - 0.15 && level < previous as f64 + 1.15 {
        previous
    } else {
        camera_zoom_to_tile_zoom(camera_zoom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn overlay() -> (TileOverlayState, mpsc::Receiver<TileFetchRequest>) {
        let (fetch_tx, fetch_rx) = mpsc::channel();
        let (_, result_rx) = mpsc::channel();
        (
            TileOverlayState {
                enabled: true,
                tile_tree: TileQuadTree::new(100),
                disk_cache_dir: std::path::PathBuf::from("/dev/null"),
                detail_texture: None,
                fetch_tx,
                result_rx,
                last_zoom: 0,
                pending_tiles: HashSet::new(),
                missing_tiles: HashMap::new(),
                needed_tiles: Vec::new(),
                dirty: false,
                fetch_generation: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
                generation: 0,
                tile_x_origin: 0,
                base_fetched: false,
                atlas: DetailAtlas::default(),
                diagnostics: None,
            },
            fetch_rx,
        )
    }

    fn mock_fetch(status: u16, headers: &str, body: Vec<u8>) -> (TileFetchResult, String) {
        use std::io::{Read, Write};
        let headers = headers.to_owned();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let service = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(3)))
                .unwrap();
            let mut request = Vec::new();
            let mut buffer = [0; 1024];
            while !request.windows(4).any(|part| part == b"\r\n\r\n") {
                let count = stream.read(&mut buffer).unwrap();
                assert_ne!(count, 0);
                request.extend_from_slice(&buffer[..count]);
            }
            let response = format!("HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nConnection: close\r\n{headers}\r\n", body.len());
            stream.write_all(response.as_bytes()).unwrap();
            stream.write_all(&body).unwrap();
            String::from_utf8(request).unwrap()
        });
        let (mut overlay, receiver) = overlay();
        overlay.queue_tile_request(
            TileCoord {
                x: 10,
                y: 20,
                z: 23,
            },
            false,
        );
        let request = receiver.recv().unwrap();
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(3))
            .build();
        let result = fetch_tile_from(&request, &agent, &service);
        (result, server.join().unwrap())
    }

    #[test]
    fn higher_imagery_levels_are_reachable_and_bounded() {
        for z in 18..=MAX_TILE_ZOOM {
            assert_eq!(camera_zoom_to_tile_zoom(2.0_f64.powi(i32::from(z) - 4)), z);
        }
        assert_eq!(stable_tile_zoom(crate::math::MAX_CAMERA_ZOOM, 18), 23);
        assert_eq!(camera_zoom_to_tile_zoom(f64::MAX), MAX_TILE_ZOOM);
    }

    #[test]
    fn missing_tiles_resolve_to_the_nearest_parent_and_deduplicate_requests() {
        let (mut overlay, receiver) = overlay();
        let coord = lon_lat_to_tile(25.5, 48.7, 23);
        let parent = coord.parent().unwrap();
        overlay.remember_missing(coord);
        overlay.queue_tile_request(coord, false);
        let request = receiver.try_recv().unwrap();
        assert_eq!(request.coord, parent);
        assert!(!request.cancel_if_stale);
        overlay.queue_tile_request(coord, false);
        assert!(receiver.try_recv().is_err());

        overlay.pending_tiles.remove(&parent);
        overlay.remember_missing(parent);
        let grandparent = parent.parent().unwrap();
        overlay.queue_tile(coord);
        assert_eq!(receiver.try_recv().unwrap().coord, grandparent);
        overlay.pending_tiles.clear();
        overlay.tile_tree.insert(grandparent, entry([20, 40, 60]));
        overlay.queue_tile(coord);
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn expired_misses_are_retried_and_a_missing_root_stops_fallback() {
        let (mut overlay, receiver) = overlay();
        let coord = TileCoord { x: 0, y: 0, z: 23 };
        overlay
            .missing_tiles
            .insert(coord, std::time::Instant::now() - MISSING_TILE_TTL);
        overlay.queue_tile(coord);
        assert_eq!(receiver.try_recv().unwrap().coord, coord);
        let root = TileCoord { x: 0, y: 0, z: 0 };
        overlay.remember_missing(root);
        overlay.queue_tile(root);
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn missing_imagery_is_not_decoded_or_confused_with_service_errors() {
        let (result, request) = mock_fetch(404, "", Vec::new());
        assert!(result.missing && result.pixels.is_empty());
        assert!(!result.cancel_if_stale);
        assert!(request.starts_with("GET /tile/23/20/10?blankTile=false HTTP/1.1"));
        let (result, _) = mock_fetch(200, "blank-tile: true\r\n", vec![1, 2, 3]);
        assert!(result.missing && result.pixels.is_empty());
        for status in [403, 429, 500] {
            let (result, _) = mock_fetch(status, "", Vec::new());
            assert!(
                !result.missing,
                "HTTP {status} is not an absence of imagery"
            );
            assert!(result.pixels.is_empty());
        }
    }

    #[test]
    fn legitimate_uniform_imagery_is_accepted() {
        let image = image::RgbImage::from_pixel(256, 256, image::Rgb([20, 40, 60]));
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut jpeg)
            .encode_image(&image)
            .unwrap();
        let (result, _) = mock_fetch(200, "Content-Type: image/jpeg\r\n", jpeg);
        assert!(!result.missing);
        assert_eq!((result.width, result.height), (256, 256));
        assert_eq!(result.pixels.len(), 256 * 256);
    }

    #[test]
    fn maximum_level_uses_parent_imagery_and_refines_when_available() {
        let coord = lon_lat_to_tile(25.5, 48.7, 23);
        let source = lon_lat_to_tile(25.5, 48.7, 18);
        let mut tree = TileQuadTree::new(100);
        tree.insert(source, entry([30, 50, 70]));
        let mut atlas = DetailAtlas::default();
        atlas.set_layout(AtlasLayout::new(&[coord], coord.x));
        assert!(
            !atlas.update_tiles(&mut tree, &[coord], |_: [u32; 2], pixels: &[[u8; 4]]| {
                assert!(pixels.iter().all(|p| *p == [30, 50, 70, 255]));
            })
        );
        assert!(atlas.covers(&[coord]));
        assert_eq!(atlas.sources[&coord], source);
        let nearest = lon_lat_to_tile(25.5, 48.7, 19);
        tree.insert(nearest, entry([60, 80, 100]));
        assert!(
            !atlas.update_tiles(&mut tree, &[coord], |_: [u32; 2], pixels: &[[u8; 4]]| {
                assert!(pixels.iter().all(|p| *p == [60, 80, 100, 255]));
            })
        );
        assert_eq!(atlas.sources[&coord], nearest);
    }

    #[test]
    fn tile_fade_is_bounded_and_smooth() {
        assert_eq!(fade_blend(0.0), 0.0);
        assert_eq!(fade_blend(0.075), 0.5);
        assert_eq!(fade_blend(0.15), 1.0);
        assert_eq!(fade_blend(1.0), 1.0);
        let values: Vec<_> = (0..=150).map(|ms| fade_blend(ms as f32 / 1000.0)).collect();
        assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn zoom_hysteresis_prevents_the_39km_detail_level_flash() {
        for initial in [11, 12] {
            let mut level = initial;
            for altitude in [39.0, 39.1, 39.0, 39.1, 39.0] {
                level = stable_tile_zoom(10000.0 / altitude, level);
                assert_eq!(level, initial);
            }
        }
        assert_eq!(stable_tile_zoom(10000.0 / 30.0, 11), 12);
        assert_eq!(stable_tile_zoom(10000.0 / 50.0, 12), 11);
    }

    fn tiles(x: u32, y: u32, cols: u32, rows: u32, z: u8) -> Vec<TileCoord> {
        (0..rows)
            .flat_map(|row| {
                (0..cols).map(move |col| TileCoord {
                    x: (x + col) % (1 << z),
                    y: y + row,
                    z,
                })
            })
            .collect()
    }

    fn entry(color: [u8; 3]) -> TileCacheEntry {
        TileCacheEntry {
            width: TILE_SIZE,
            height: TILE_SIZE,
            pixels: vec![color; (TILE_SIZE * TILE_SIZE) as usize],
        }
    }

    #[test]
    fn atlas_copy_regions_preserve_exact_geographic_overlap() {
        for old_x in 0..8 {
            for new_x in 0..8 {
                for old_cols in 1..=8 {
                    for new_cols in 1..=8 {
                        let old = AtlasLayout {
                            x: old_x,
                            y: 1,
                            z: 3,
                            cols: old_cols,
                            rows: 4,
                        };
                        let new = AtlasLayout {
                            x: new_x,
                            y: 2,
                            z: 3,
                            cols: new_cols,
                            rows: 4,
                        };
                        let mut copied = HashSet::new();
                        for region in new.copies_from(old) {
                            for dy in 0..region.size[1] / TILE_SIZE {
                                for dx in 0..region.size[0] / TILE_SIZE {
                                    let src = TileCoord {
                                        x: (old.x + region.source[0] / TILE_SIZE + dx) % 8,
                                        y: old.y + region.source[1] / TILE_SIZE + dy,
                                        z: 3,
                                    };
                                    let dst = TileCoord {
                                        x: (new.x + region.destination[0] / TILE_SIZE + dx) % 8,
                                        y: new.y + region.destination[1] / TILE_SIZE + dy,
                                        z: 3,
                                    };
                                    assert_eq!(src, dst);
                                    assert!(copied.insert(dst));
                                }
                            }
                        }
                        let expected = tiles(new.x, new.y, new.cols, new.rows, 3)
                            .into_iter()
                            .filter(|c| old.contains(*c))
                            .collect::<HashSet<_>>();
                        assert_eq!(copied, expected);
                    }
                }
            }
        }
        let a = AtlasLayout {
            x: 0,
            y: 0,
            z: 3,
            cols: 4,
            rows: 4,
        };
        assert!(a.copies_from(AtlasLayout { z: 4, ..a }).is_empty());
    }

    #[test]
    fn atlas_updates_are_bounded_and_reuse_tiles_when_panning() {
        let mut tree = TileQuadTree::new(100);
        for c in tiles(10, 10, 5, 4, 6) {
            tree.insert(c, entry([c.x as u8, c.y as u8, 80]));
        }
        let needed = tiles(10, 10, 4, 4, 6);
        let mut atlas = DetailAtlas::default();
        atlas.set_layout(AtlasLayout::new(&needed, 10));
        let mut count = 0;
        assert!(
            atlas.update_tiles(&mut tree, &needed, |_: [u32; 2], pixels: &[[u8; 4]]| {
                count += 1;
                assert_eq!(pixels.len(), 65536);
            })
        );
        assert_eq!(count, TILE_UPLOADS_PER_FRAME);
        assert!(!atlas.covers(&needed));
        assert!(!atlas.update_tiles(&mut tree, &needed, |_: [u32; 2], _: &[[u8; 4]]| count += 1));
        assert_eq!(count, 16);
        assert!(atlas.covers(&needed));
        assert!(
            !atlas.update_tiles(&mut tree, &needed, |_: [u32; 2], _: &[[u8; 4]]| panic!(
                "Unchanged tile uploaded"
            ))
        );
        let moved = tiles(11, 10, 4, 4, 6);
        atlas.set_layout(AtlasLayout::new(&moved, 11));
        count = 0;
        assert!(
            !atlas.update_tiles(&mut tree, &moved, |origin: [u32; 2], pixels: &[[u8; 4]]| {
                assert_eq!(origin[0], 3 * TILE_SIZE);
                assert_eq!(pixels[0][0], 14);
                count += 1;
            })
        );
        assert_eq!(count, 4);
        assert_eq!(atlas.buffer.len() * 4, 256 * 1024);
    }

    #[test]
    fn ancestor_sampling_and_progressive_replacement_are_correct() {
        let mut tree = TileQuadTree::new(100);
        let mut parent = entry([0; 3]);
        for y in 0..256usize {
            for x in 0..256usize {
                parent.pixels[y * 256 + x] = [x as u8, y as u8, 0];
            }
        }
        tree.insert(TileCoord { x: 1, y: 1, z: 2 }, parent);
        let coord = TileCoord { x: 5, y: 6, z: 4 };
        let mut atlas = DetailAtlas::default();
        atlas.set_layout(AtlasLayout::new(&[coord], coord.x));
        assert!(
            !atlas.update_tiles(&mut tree, &[coord], |_: [u32; 2], pixels: &[[u8; 4]]| {
                assert_eq!(pixels[0], [64, 128, 0, 255]);
                assert_eq!(pixels[65535], [127, 191, 0, 255]);
            })
        );
        tree.insert(coord, entry([1, 2, 3]));
        assert!(
            !atlas.update_tiles(&mut tree, &[coord], |_: [u32; 2], pixels: &[[u8; 4]]| {
                assert!(pixels.iter().all(|p| *p == [1, 2, 3, 255]));
            })
        );
        assert!(
            !atlas.update_tiles(&mut tree, &[coord], |_: [u32; 2], _: &[[u8; 4]]| panic!(
                "Reuploaded unchanged tile"
            ))
        );
    }

    #[test]
    fn gpu_reuse_does_not_consume_or_wait_for_the_upload_budget() {
        struct Writer {
            uploads: usize,
            reused: usize,
        }
        impl TileAtlasWriter for &mut Writer {
            fn reuse(&mut self, c: TileCoord, _: TileCoord, _: [u32; 2]) -> TileReuse {
                if c.x >= 12 {
                    self.reused += 1;
                    TileReuse::Reused
                } else {
                    TileReuse::Missing
                }
            }
            fn upload(&mut self, _: TileCoord, _: TileCoord, _: [u32; 2], _: &[[u8; 4]]) {
                self.uploads += 1;
            }
        }
        let needed = tiles(0, 0, 24, 1, 5);
        let mut tree = TileQuadTree::new(100);
        for &coord in &needed {
            tree.insert(coord, entry([1, 2, 3]));
        }
        let mut atlas = DetailAtlas::default();
        atlas.set_layout(AtlasLayout::new(&needed, 0));
        let mut writer = Writer {
            uploads: 0,
            reused: 0,
        };
        assert!(atlas.update_tiles(&mut tree, &needed, &mut writer));
        assert_eq!(writer.uploads, TILE_UPLOADS_PER_FRAME);
        assert_eq!(writer.reused, 12);
        assert!(!atlas.covers(&needed));
        assert!(!atlas.update_tiles(&mut tree, &needed, &mut writer));
        assert_eq!(writer.uploads, 12);
        assert!(atlas.covers(&needed));
    }

    #[test]
    #[ignore = "CPU benchmark for the ESRI atlas update path"]
    fn benchmark_incremental_atlas_updates() {
        let needed = tiles(0, 0, 16, 16, 8);
        let mut tree = TileQuadTree::new(1024);
        for &coord in &needed {
            tree.insert(coord, entry([10, 20, 30]));
        }
        let mut atlas = DetailAtlas::default();
        atlas.set_layout(AtlasLayout::new(&needed, 0));
        while atlas.update_tiles(&mut tree, &needed, |_: [u32; 2], p: &[[u8; 4]]| {
            std::hint::black_box(p);
        }) {}
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            assert!(!atlas.update_tiles(&mut tree, &needed, |_: [u32; 2], _: &[[u8; 4]]| panic!()));
        }
        let unchanged = start.elapsed().as_secs_f64() * 1000.0 / 1000.0;
        let mut bytes = 0;
        let start = std::time::Instant::now();
        for i in 0..100 {
            atlas.sources.remove(&needed[i]);
            assert!(
                !atlas.update_tiles(&mut tree, &needed, |_: [u32; 2], p: &[[u8; 4]]| {
                    bytes += p.len() * 4;
                    std::hint::black_box(p);
                })
            );
        }
        let update = start.elapsed().as_secs_f64() * 1000.0 / 100.0;
        eprintln!("ESRI atlas CPU: unchanged {unchanged:.4} ms/frame; single-tile update {update:.4} ms; {} KiB/update instead of 65536 KiB", bytes / 100 / 1024);
        assert_eq!(bytes / 100, 256 * 1024);
    }

    fn bounds(z: u8, x: u32, y: u32, cols: u32, rows: u32) -> DetailBounds {
        let (west, north) = tile_to_lon_lat(&TileCoord { x, y, z });
        let (east, south) = tile_to_lon_lat(&TileCoord {
            x: x + cols,
            y: y + rows,
            z,
        });
        DetailBounds {
            min_lon: west,
            max_lon: east,
            min_lat: south.to_radians(),
            max_lat: north.to_radians(),
        }
    }
    #[test]
    fn shader_bounds_are_in_web_mercator_units() {
        let b = bounds(4, 8, 1, 4, 7).shader_bounds();
        assert!((b[2] as f64).abs() < 1e-6);
        assert!((b[3] as f64 - PI * 0.875).abs() < 1e-6);
        let p = tile_to_lon_lat(&TileCoord { x: 9, y: 3, z: 4 });
        let u = (p.0 - b[0] as f64) / (b[1] - b[0]) as f64;
        let v = (b[3] as f64 - p.1.to_radians().tan().asinh()) / (b[3] - b[2]) as f64;
        assert!((u - 0.25).abs() < 1e-6);
        assert!((v - 2.0 / 7.0).abs() < 1e-6);
    }
    #[test]
    fn atlas_rebuilds_keep_global_tile_coordinates_fixed() {
        for (lon, lat) in [(30.0_f64, 60.0_f64), (179.0, 60.0), (-179.0, -60.0)] {
            let tile = lon_lat_to_tile(lon, lat, 5);
            let n = 32.0;
            let expected_y = (1.0 - lat.to_radians().tan().asinh() / PI) * 0.5 * n;
            for (cols, rows) in [(2, 2), (4, 4), (6, 6)] {
                let x = tile.x.saturating_sub(cols / 2);
                let y = tile.y.saturating_sub(rows / 2);
                let b = bounds(5, x, y, cols, rows).shader_bounds();
                let v = (b[3] as f64 - lat.to_radians().tan().asinh()) / (b[3] - b[2]) as f64;
                assert!((y as f64 + v * rows as f64 - expected_y).abs() < 1e-5);
            }
        }
    }
}
