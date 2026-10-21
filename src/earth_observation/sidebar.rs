use super::{ForecastRequest, Panel};
use crate::{aoi::AoiGenerator, celestial::CelestialBody, config::TabConfig};
use chrono::{DateTime, Utc};
use eframe::egui;

impl Panel {
    pub(super) fn ensure_earth(&mut self, tab: &TabConfig) -> bool {
        if tab
            .planets
            .get(self.planet)
            .is_some_and(|p| p.celestial_body == CelestialBody::Earth)
        {
            return true;
        }
        if let Some(index) = tab
            .planets
            .iter()
            .position(|p| p.celestial_body == CelestialBody::Earth)
        {
            self.planet = index;
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
    ) -> Option<ForecastRequest> {
        if !self.open {
            self.suspend();
            #[allow(deprecated)]
            egui::Panel::right("observation-sidebar-closed")
                .exact_size(34.0)
                .show(ctx, |ui| {
                    if ui
                        .small_button("EO")
                        .on_hover_text("Open Earth Observation sidebar")
                        .clicked()
                    {
                        self.open = true;
                    }
                });
            return None;
        }
        let mut request = None;
        #[allow(deprecated)]
        egui::Panel::right("observation-sidebar").resizable(true).default_size(500.0)
            .size_range(420.0..=850.0).show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Earth Observation");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("−").on_hover_text("Collapse sidebar").clicked() {
                            self.open = false;
                        }
                    });
                });
                ui.small(format!("Active tab: {}", tab.name));
                ui.separator();
                egui::ScrollArea::vertical().id_salt("observation-sidebar-scroll")
                    .auto_shrink([false, false]).show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        egui::CollapsingHeader::new("AOI Generator").id_salt("sidebar-aoi")
                            .show(ui, |ui| aoi.ui(ui));
                        ui.separator();
                        if !self.ensure_earth(tab) {
                            ui.label("Select a tab with an Earth globe for the observation model and data estimate.");
                            return;
                        }
                        let earths: Vec<_> = tab.planets.iter().enumerate()
                            .filter(|(_, p)| p.celestial_body == CelestialBody::Earth)
                            .map(|(i, p)| (i, p.name.clone())).collect();
                        if earths.len() > 1 {
                            egui::ComboBox::from_id_salt("observation-earth")
                                .selected_text(&tab.planets[self.planet].name).show_ui(ui, |ui| {
                                    for (i, name) in earths { ui.selectable_value(&mut self.planet, i, name); }
                                });
                        }
                        let model = egui::CollapsingHeader::new("Observation model")
                            .id_salt("sidebar-model").default_open(true).show(ui, |ui| {
                                request = self.ui(ui, tab, tab_idx, epoch);
                            });
                        if model.body_returned.is_none() { self.suspend(); }
                        ui.separator();
                        egui::CollapsingHeader::new("Data").id_salt("sidebar-data")
                            .default_open(true).show(ui, |ui| {
                                self.data.ui(ui, &mut tab.settings.observation_data,
                                    &tab.planets[self.planet], tab.settings.pushbroom.swath_km,
                                    tab.settings.time, epoch);
                            });
                        ui.separator();
                        aoi.jobs_ui(ui, &mut tab.settings.observation_data);
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
                    panel.sidebar(ctx, &mut aoi, &mut tab, 0, epoch);
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
        assert!(
            available.width() >= 900.0 && available.width() <= 1250.0,
            "{available:?}"
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
                panel.sidebar(ctx, &mut aoi, &mut tab, 0, epoch);
                available = ctx.available_rect();
            },
        );
        assert!(available.width() > 1500.0);
        assert!(!panel.armed);
    }
}
