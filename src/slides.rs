//! Embedded slide decks for in-app presentations.
//!
//! Each deck is a list of SVG byte slices (one per slide) bundled at compile
//! time via `include_bytes!`. SVGs are registered with the egui context so
//! they can be displayed via `egui::Image::new("bytes://...")`.

use eframe::egui;

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
        let total = deck(id).expect("unknown slide deck id").len();
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
        let total = deck(id).expect("unknown slide deck id").len();
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
        format!("bytes://slides/{}/{:02}.svg", self.id.0, absolute + 1)
    }
}

pub const SPACECOMP_PRIMER: DeckId = DeckId("spacecomp-primer");

const SPACECOMP_PRIMER_SLIDES: &[&[u8]] = &[
    include_bytes!("../assets/presentations/spacecomp-primer/01.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/02.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/03.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/04.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/05.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/06.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/07.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/08.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/09.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/10.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/11.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/12.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/13.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/14.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/15.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/16.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/17.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/18.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/19.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/20.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/21.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/22.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/23.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/24.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/25.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/26.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/27.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/28.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/29.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/30.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/31.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/32.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/33.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/34.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/35.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/36.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/37.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/38.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/39.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/40.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/41.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/42.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/43.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/44.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/45.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/46.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/47.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/48.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/49.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/50.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/51.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/52.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/53.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/54.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/55.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/56.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/57.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/58.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/59.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/60.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/61.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/62.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/63.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/64.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/65.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/66.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/67.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/68.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/69.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/70.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/71.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/72.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/73.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/74.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/75.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/76.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/77.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/78.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/79.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/80.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/81.svg"),
    include_bytes!("../assets/presentations/spacecomp-primer/82.svg"),
];

const ALL_DECKS: &[(DeckId, &[&[u8]])] = &[(SPACECOMP_PRIMER, SPACECOMP_PRIMER_SLIDES)];

fn deck(id: DeckId) -> Option<&'static [&'static [u8]]> {
    ALL_DECKS
        .iter()
        .find(|(d, _)| d.0 == id.0)
        .map(|(_, slides)| *slides)
}

pub fn total_slide_count() -> usize {
    ALL_DECKS.iter().map(|(_, slides)| slides.len()).sum()
}

pub fn install(ctx: &egui::Context) {
    egui_extras::install_image_loaders(ctx);
    for (id, slides) in ALL_DECKS {
        for (idx, bytes) in slides.iter().enumerate() {
            let uri = format!("bytes://slides/{}/{:02}.svg", id.0, idx + 1);
            ctx.include_bytes(uri, *bytes);
        }
    }
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
