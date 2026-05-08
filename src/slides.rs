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
    pub fn new(id: DeckId) -> Self {
        let total = deck(id).expect("unknown slide deck id").len();
        Self { id, range: 0..total, current: 0 }
    }

    /// Show only `range` of the source deck (0-based, half-open). Out-of-bounds
    /// endpoints are clamped to the deck size.
    #[allow(dead_code)]
    pub fn range(id: DeckId, range: std::ops::Range<usize>) -> Self {
        let total = deck(id).expect("unknown slide deck id").len();
        let start = range.start.min(total);
        let end = range.end.min(total).max(start);
        Self { id, range: start..end, current: 0 }
    }

    pub fn len(&self) -> usize {
        self.range.len()
    }

    pub fn uri(&self, idx: usize) -> String {
        let absolute = self.range.start + idx;
        format!("bytes://slides/{}/{:02}.svg", self.id.0, absolute)
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
];

const ALL_DECKS: &[(DeckId, &[&[u8]])] = &[(SPACECOMP_PRIMER, SPACECOMP_PRIMER_SLIDES)];

fn deck(id: DeckId) -> Option<&'static [&'static [u8]]> {
    ALL_DECKS
        .iter()
        .find(|(d, _)| d.0 == id.0)
        .map(|(_, slides)| *slides)
}

pub fn install(ctx: &egui::Context) {
    egui_extras::install_image_loaders(ctx);
    for (id, slides) in ALL_DECKS {
        for (idx, bytes) in slides.iter().enumerate() {
            let uri = format!("bytes://slides/{}/{:02}.svg", id.0, idx);
            ctx.include_bytes(uri, *bytes);
        }
    }
}
