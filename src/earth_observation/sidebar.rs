use super::{ForecastRequest, Panel};
use crate::{aoi::AoiGenerator, celestial::CelestialBody, config::TabConfig};
use chrono::{DateTime, Utc};
use eframe::egui;

impl Panel {
    pub fn available_for(tab: &TabConfig) -> bool {
        tab.slides.is_none()
            && tab.settings.view_mode == crate::config::ViewMode::Planet
            && tab.planets.len() == 1
            && tab.planets[0].celestial_body == CelestialBody::Earth
    }

    pub(super) fn ensure_earth(&mut self, tab: &TabConfig) -> bool {
        if Self::available_for(tab) {
            self.planet = 0;
            return true;
        }
        self.suspend();
        self.scope = None;
        self.selected = None;
        self.report = None;
        false
    }

    fn suspend(&mut self) {
        self.armed = false;
        self.pending = None;
        self.needs_run = false;
    }

    pub fn sidebar(
        &mut self,
        ctx: &egui::Context,
        aoi: &mut AoiGenerator,
        tab: &mut TabConfig,
        tab_idx: usize,
        epoch: DateTime<Utc>,
        simulation_open: &mut bool,
    ) -> Option<ForecastRequest> {
        if !self.ensure_earth(tab) {
            self.open = false;
        }
        if !self.open {
            self.suspend();
            return None;
        }
        let mut request = None;
        #[allow(deprecated)]
        egui::Panel::right("observation-sidebar")
            .resizable(true)
            .default_size(500.0)
            .size_range(420.0..=850.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Earth Observation");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .small_button("−")
                            .on_hover_text("Close Earth Observation")
                            .clicked()
                        {
                            self.open = false;
                        }
                        if ui
                            .button("Sim")
                            .on_hover_text("Open infrastructure simulation")
                            .clicked()
                        {
                            *simulation_open = true;
                            ctx.request_repaint();
                        }
                    });
                });
                ui.small(format!("Active tab: {}", tab.name));
                ui.separator();
                egui::ScrollArea::vertical()
                    .id_salt("observation-sidebar-scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        aoi.presets_ui(ui);
                        ui.separator();
                        egui::CollapsingHeader::new("AOI Generator")
                            .id_salt("sidebar-aoi")
                            .show(ui, |ui| aoi.ui(ui));
                        ui.separator();
                        let model = egui::CollapsingHeader::new("Observation model")
                            .id_salt("sidebar-model")
                            .default_open(true)
                            .show(ui, |ui| {
                                request = self.ui(ui, tab, tab_idx, epoch);
                            });
                        if model.body_returned.is_none() {
                            self.suspend();
                        }
                        ui.separator();
                        egui::CollapsingHeader::new("Data")
                            .id_salt("sidebar-data")
                            .default_open(true)
                            .show(ui, |ui| {
                                if tab.settings.pushbroom.observable_width_km.is_some() {
                                    ui.small("Uses the wider observable corridor as an access approximation, not simultaneous imaging across that width.");
                                }
                                self.data.ui(
                                    ui,
                                    &mut tab.settings.observation_data,
                                    &tab.planets[self.planet],
                                    tab.settings.pushbroom.observation_width_km(),
                                    tab.settings.time,
                                    epoch,
                                );
                            });
                        ui.separator();
                        aoi.jobs_ui(ui, &mut tab.settings.observation_data);
                        ui.separator();
                        aoi.jobs.satisfiability_ui(ui, &tab.planets[self.planet],
                            tab.settings.observation_data, tab.settings.pushbroom.observation_width_km(),
                            tab.settings.pushbroom.observable_width_km.is_some(), tab.settings.time, epoch);
                        ui.separator();
                        aoi.jobs.capture_ui(
                            ui,
                            &tab.planets[self.planet],
                            tab.settings.observation_data,
                            tab.settings.pushbroom.observation_width_km(),
                            tab.settings.time,
                            epoch,
                        );
                    });
            });
        if !self.open {
            self.suspend();
        }
        request
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn application_requires_exactly_one_earth_in_a_planet_view() {
        use crate::config::ViewMode;
        let mut tab = TabConfig::new("Earth".into());
        assert!(Panel::available_for(&tab));
        tab.add_planet();
        assert!(!Panel::available_for(&tab));
        tab.planets.pop();
        tab.planets[0].celestial_body = CelestialBody::Mars;
        assert!(!Panel::available_for(&tab));
        tab.planets[0].celestial_body = CelestialBody::Earth;
        for mode in [
            ViewMode::SolarSystem,
            ViewMode::PlanetSizes,
            ViewMode::Satellite,
            ViewMode::SatelliteTable,
            ViewMode::GroundTerminals,
        ] {
            tab.settings.view_mode = mode;
            assert!(!Panel::available_for(&tab));
        }
        tab.settings.view_mode = ViewMode::Planet;
        tab.planets.clear();
        assert!(!Panel::available_for(&tab));
    }

    #[test]
    #[allow(deprecated)] // Exercise the top-level panel API used by the application.
    fn right_sidebar_reserves_globe_space_and_collapses() {
        let ctx = egui::Context::default();
        let mut panel = Panel::default();
        let mut aoi = AoiGenerator::default();
        let mut tab = TabConfig::new("Earth".into());
        let mut c = crate::config::ConstellationConfig::new(0);
        crate::orbit_presets::apply_adam(&mut c);
        tab.planets[0].constellations = vec![c];
        tab.settings.observation_data.compression_ratio = 2.0;
        let epoch = DateTime::from_timestamp(946728000, 0).unwrap();
        let mut available = egui::Rect::NOTHING;
        let mut shapes = Vec::new();
        let mut simulation_open = false;
        for _ in 0..3 {
            let out = ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1600.0, 1000.0),
                    )),
                    ..Default::default()
                },
                |ctx| {
                    panel.sidebar(ctx, &mut aoi, &mut tab, 0, epoch, &mut simulation_open);
                    available = ctx.available_rect();
                },
            );
            shapes = out.shapes;
        }
        fn text_rect(shape: &egui::Shape, needle: &str) -> Option<egui::Rect> {
            match shape {
                egui::Shape::Text(t) if t.galley.job.text == needle => {
                    Some(egui::Rect::from_min_size(t.pos, t.galley.size()))
                }
                egui::Shape::Vec(v) => v.iter().find_map(|s| text_rect(s, needle)),
                _ => None,
            }
        }
        for label in [
            "Sim",
            "AOI presets / save / load",
            "Default Jobs",
            "Load preset",
            "Save AOIs",
            "Load AOIs",
            "AOI Generator",
            "Observation model",
            "Data",
            "Swath width",
            "Channels",
            "Resolution per channel",
            "Bits per sample",
            "Compression ratio",
            "Constellation total",
        ] {
            let rect = shapes
                .iter()
                .find_map(|s| text_rect(&s.shape, label))
                .unwrap_or_else(|| panic!("Missing sidebar control: {label}"));
            assert!(
                rect.left() >= available.right()
                    && rect.right() <= 1600.0
                    && rect.bottom() <= 1000.0,
                "Clipped control {label}: {rect:?}"
            );
        }
        let preset = shapes.iter().find_map(|s| text_rect(&s.shape, "AOI presets / save / load")).unwrap();
        let generator = shapes.iter().find_map(|s| text_rect(&s.shape, "AOI Generator")).unwrap();
        assert!(preset.bottom() < generator.top());
        assert_eq!(shapes.iter().filter(|s| text_rect(&s.shape, "Load preset").is_some()).count(), 1);
        assert!(!shapes.iter().any(|s| text_rect(&s.shape, "Save / load AOIs").is_some()));
        assert!(
            available.width() >= 900.0 && available.width() <= 1250.0,
            "{available:?}"
        );
        let sim = shapes
            .iter()
            .find_map(|s| text_rect(&s.shape, "Sim"))
            .unwrap()
            .center();
        for pressed in [true, false] {
            let _ = ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1600.0, 1000.0),
                    )),
                    events: vec![
                        egui::Event::PointerMoved(sim),
                        egui::Event::PointerButton {
                            pos: sim,
                            button: egui::PointerButton::Primary,
                            pressed,
                            modifiers: Default::default(),
                        },
                    ],
                    ..Default::default()
                },
                |ctx| {
                    panel.sidebar(ctx, &mut aoi, &mut tab, 0, epoch, &mut simulation_open);
                },
            );
        }
        assert!(
            simulation_open,
            "Sim opens the infrastructure simulation window"
        );
        panel.open = false;
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1600.0, 1000.0),
                )),
                ..Default::default()
            },
            |ctx| {
                panel.sidebar(ctx, &mut aoi, &mut tab, 0, epoch, &mut simulation_open);
                available = ctx.available_rect();
            },
        );
        assert_eq!(available.width(), 1600.0);
        assert!(!panel.armed);
        panel.open = true;
        panel.armed = true;
        tab.add_planet();
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1600.0, 1000.0),
                )),
                ..Default::default()
            },
            |ctx| {
                panel.sidebar(ctx, &mut aoi, &mut tab, 0, epoch, &mut simulation_open);
                available = ctx.available_rect();
            },
        );
        assert_eq!(available.width(), 1600.0);
        assert!(!panel.open);
        assert!(!panel.armed);
    }
}
