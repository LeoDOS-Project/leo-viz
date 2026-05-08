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
    pub current: usize,
}

impl SlideDeck {
    pub fn new(id: DeckId) -> Self {
        Self { id, current: 0 }
    }

    pub fn slides(&self) -> &'static [&'static [u8]] {
        deck(self.id).expect("unknown slide deck id")
    }

    pub fn len(&self) -> usize {
        self.slides().len()
    }

    pub fn uri(&self, idx: usize) -> String {
        format!("bytes://slides/{}/{:02}.svg", self.id.0, idx)
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
