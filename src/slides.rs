//! Slide decks for in-app presentations.
//!
//! Slides are stored as external SVG assets instead of being embedded into the
//! WASM binary. The web build copies them next to the app and loads them by URL;
//! native builds load them from the repository asset directory.

use eframe::egui;
#[cfg(target_arch = "wasm32")]
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeckId(pub &'static str);

#[derive(Clone)]
pub struct SlideDeck {
    pub id: DeckId,
    /// Half-open absolute slide indices into the source deck (0-based).
    pub range: std::ops::Range<usize>,
    /// Current slide, relative to `range.start` (0..self.len()).
    pub current: usize,
}

impl SlideDeck {
    #[allow(dead_code)]
    pub fn new(id: DeckId) -> Self {
        let total = deck(id).expect("unknown slide deck id").count;
        Self {
            id,
            range: 0..total,
            current: 0,
        }
    }

    /// Show only `range` of the source deck (0-based, half-open). Out-of-bounds
    /// endpoints are clamped to the deck size.
    #[allow(dead_code)]
    pub fn range(id: DeckId, range: std::ops::Range<usize>) -> Self {
        let total = deck(id).expect("unknown slide deck id").count;
        let start = range.start.min(total);
        let end = range.end.min(total).max(start);
        Self {
            id,
            range: start..end,
            current: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.range.len()
    }

    pub fn uri(&self, idx: usize) -> String {
        let absolute = self.range.start + idx;
        slide_uri(self.id, absolute)
    }
}

#[derive(Clone, Copy)]
struct DeckInfo {
    id: DeckId,
    count: usize,
}

pub const SPACECOMP_PRIMER: DeckId = DeckId("spacecomp-primer");
pub const SPACECOMP_PRIMER_SLIDE_COUNT: usize = 76;

const ALL_DECKS: &[DeckInfo] = &[DeckInfo {
    id: SPACECOMP_PRIMER,
    count: SPACECOMP_PRIMER_SLIDE_COUNT,
}];

#[cfg(target_arch = "wasm32")]
static SPACECOMP_PRIMER_CACHE_WARMED: AtomicBool = AtomicBool::new(false);

fn deck(id: DeckId) -> Option<DeckInfo> {
    ALL_DECKS.iter().find(|d| d.id.0 == id.0).copied()
}

pub fn total_slide_count() -> usize {
    ALL_DECKS.iter().map(|deck| deck.count).sum()
}

pub fn install(ctx: &egui::Context) {
    egui_extras::install_image_loaders(ctx);
}

pub fn slide_uri(id: DeckId, idx: usize) -> String {
    format!("{}/{:02}.svg", deck_base_uri(id), idx + 1)
}

#[cfg(target_arch = "wasm32")]
fn deck_base_uri(id: DeckId) -> String {
    let asset_path = id.0;
    let Some(window) = web_sys::window() else {
        return asset_path.to_owned();
    };
    let location = window.location();
    let origin = location.origin().unwrap_or_default();
    let path = location.pathname().unwrap_or_default();
    let trimmed = path.trim_end_matches('/');
    let base = trimmed
        .strip_suffix("/demo")
        .or_else(|| trimmed.strip_suffix("/presentation"))
        .or_else(|| trimmed.strip_suffix("/index.html"))
        .unwrap_or(trimmed);
    let base = if base == "/" { "" } else { base };
    format!("{origin}{base}/{asset_path}")
}

#[cfg(not(target_arch = "wasm32"))]
fn deck_base_uri(id: DeckId) -> String {
    let path = std::env::current_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join("assets")
        .join("presentations")
        .join(id.0);
    format!("file://{}", path.display())
}

pub fn preload_uri(ctx: &egui::Context, uri: &str, size: egui::Vec2) {
    let pixel_size = size * ctx.pixels_per_point();
    let size_hint = egui::load::SizeHint::Size {
        width: pixel_size.x.max(1.0).round() as u32,
        height: pixel_size.y.max(1.0).round() as u32,
        maintain_aspect_ratio: true,
    };
    match ctx.try_load_texture(uri, egui::TextureOptions::LINEAR, size_hint) {
        Ok(egui::load::TexturePoll::Pending { .. }) => ctx.request_repaint(),
        _ => {}
    }
}

pub fn warm_browser_cache(id: DeckId) {
    #[cfg(target_arch = "wasm32")]
    {
        if id == SPACECOMP_PRIMER && SPACECOMP_PRIMER_CACHE_WARMED.swap(true, Ordering::Relaxed) {
            return;
        }
        let Some(info) = deck(id) else {
            return;
        };
        let Some(window) = web_sys::window() else {
            return;
        };
        for idx in 0..info.count {
            let _ = window.fetch_with_str(&slide_uri(id, idx));
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    let _ = id;
}
