//! Published designs, kept separate from our sufficient coverage construction.
//! Values below are transcribed from the linked tables, not fitted to our model.
use super::designer::{clear_tracks, hide_other_constellations};
use crate::{
    config::{ConstellationConfig, PlanetConfig, Propagator},
    pushbroom,
    walker::WalkerType,
};
use eframe::egui;

struct Shell {
    altitude_km: f64,
    inclination: f64,
    planes: usize,
    satellites_per_plane: usize,
    /// Published Walker Delta F, with positive physical RAAN increments.
    phasing: usize,
}

struct PublishedPreset {
    name: &'static str,
    category: &'static str,
    source: &'static str,
    url: &'static str,
    objective: &'static str,
    assumptions: &'static str,
    swath_km: Option<f64>,
    shells: &'static [Shell],
}

const CRISP_SOURCE: &str = "Crisp, Livadiotti & Roberts (2018), §3.2 / Table 3";
const CRISP_URL: &str = "https://arxiv.org/pdf/1807.02021#page=7";
const CRISP_OBJECTIVE: &str =
    "Benchmark for maximum revisit time at the equator; no optimality claim for this layout.";

const PRESETS: &[PublishedPreset] = &[
    PublishedPreset {
        name: "He 2020 · disaster monitoring · 5 satellites",
        category: "Optimization result — initial global mode",
        source: "He et al. (2020), §4.2 / Table 4 and §4.3",
        url: "https://doi.org/10.1155/2020/8813685",
        objective: "Balances maneuver cost and reconfiguration time for unpredictable disaster sites.",
        assumptions: "Uses the authors' ideal Walker 5/5/3 counterpart to their slightly asymmetric numerical result. Initial global mode only; target-specific transfers are not simulated.",
        swath_km: Some(60.0),
        shells: &[Shell { altitude_km: 597.766, inclination: 90.0, planes: 5, satellites_per_plane: 1, phasing: 3 }],
    },
    PublishedPreset {
        name: "Zong 2019 · hybrid coverage · 49 satellites",
        category: "Published study configuration",
        source: "Zong & Kohani (2019), Tables 1–2",
        url: "https://doi.org/10.1155/2019/4373749",
        objective: "Combines two shells for global coverage in a study of coverage and revisit optimization.",
        assumptions: "The source uses circular visibility footprints, not narrow scanlines. These tables specify the hybrid layout, not a pushbroom optimum. Unspecified shell offsets are set to zero.",
        swath_km: None,
        shells: &[
            Shell { altitude_km: 1100.0, inclination: 80.0, planes: 3, satellites_per_plane: 7, phasing: 1 },
            Shell { altitude_km: 750.0, inclination: 65.0, planes: 4, satellites_per_plane: 7, phasing: 0 },
        ],
    },
    PublishedPreset {
        name: "Crisp 2018 · polar 3/3/0 · 700 km",
        category: "Revisit benchmark",
        source: CRISP_SOURCE,
        url: CRISP_URL,
        objective: CRISP_OBJECTIVE,
        assumptions: "Paper: 0° minimum elevation, circular access footprint; reported equatorial MRT 2.30 h. That value is not a pushbroom prediction.",
        swath_km: None,
        shells: &[Shell { altitude_km: 700.0, inclination: 90.0, planes: 3, satellites_per_plane: 1, phasing: 0 }],
    },
    PublishedPreset {
        name: "Crisp 2018 · near-polar 3/3/0 · 1100 km",
        category: "Revisit benchmark",
        source: CRISP_SOURCE,
        url: CRISP_URL,
        objective: CRISP_OBJECTIVE,
        assumptions: "Paper: 10° minimum elevation, circular access footprint; reported equatorial MRT 4.25 h. That value is not a pushbroom prediction.",
        swath_km: None,
        shells: &[Shell { altitude_km: 1100.0, inclination: 86.0, planes: 3, satellites_per_plane: 1, phasing: 0 }],
    },
    PublishedPreset {
        name: "Crisp 2018 · retrograde 3/3/1 · 1500 km",
        category: "Revisit benchmark",
        source: CRISP_SOURCE,
        url: CRISP_URL,
        objective: CRISP_OBJECTIVE,
        assumptions: "Paper: 20° minimum elevation, circular access footprint; reported equatorial MRT 3.38 h. That value is not a pushbroom prediction.",
        swath_km: None,
        shells: &[Shell { altitude_km: 1500.0, inclination: 96.0, planes: 3, satellites_per_plane: 1, phasing: 1 }],
    },
];

impl Shell {
    fn apply(&self, c: &mut ConstellationConfig, j2: bool) {
        c.num_planes = self.planes;
        c.sats_per_plane = self.satellites_per_plane;
        c.altitude_km = self.altitude_km;
        c.inclination = self.inclination;
        c.walker_type = WalkerType::Delta;
        // leo-viz numbers planes in the opposite physical RAAN direction.
        // Reversing plane indices maps F -> (P-F) mod P, modulo the spacing
        // between satellites in a plane. Preserve the published geometry.
        c.phasing = ((self.planes - self.phasing) % self.planes) as f64;
        c.raan_offset = 0.0;
        c.raan_spacing = None;
        c.sat_spacing_km = None;
        c.eccentricity = 0.0;
        c.arg_periapsis = 0.0;
        c.sso = false;
        c.propagator = if j2 {
            Propagator::J2
        } else {
            Propagator::Keplerian
        };
    }
}

pub(super) struct Literature {
    selected: usize,
    j2: bool,
    use_swath: bool,
    hide_others: bool,
    status: Option<String>,
}
impl Default for Literature {
    fn default() -> Self {
        Self {
            selected: 0,
            j2: true,
            use_swath: true,
            hide_others: true,
            status: None,
        }
    }
}
impl Literature {
    fn add(&mut self, p: &mut PlanetConfig, imaging: &mut pushbroom::Settings) {
        let preset = &PRESETS[self.selected];
        if self.hide_others {
            hide_other_constellations(p);
        }
        for (index, shell) in preset.shells.iter().enumerate() {
            p.add_constellation();
            let c = p.constellations.last_mut().unwrap();
            shell.apply(c, self.j2);
            c.label = Some(if preset.shells.len() == 1 {
                preset.name.to_owned()
            } else {
                format!("{} · shell {}", preset.name, index + 1)
            });
        }
        if self.use_swath {
            if let Some(width) = preset.swath_km {
                imaging.swath_km = width;
            }
        }
        clear_tracks(p);
        self.status = Some(format!(
            "Added {}. Edit the orbits in constellation controls.",
            preset.name
        ));
    }

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        p: &mut PlanetConfig,
        imaging: &mut pushbroom::Settings,
    ) {
        egui::CollapsingHeader::new("Literature presets").show(ui, |ui| {
            ui.small("Published designs and benchmarks. Optimality depends on the mission and sensor assumptions.");
            egui::ComboBox::from_id_salt("literature-preset")
                .selected_text(PRESETS[self.selected].name)
                .width(ui.available_width().min(355.0))
                .show_ui(ui, |ui| {
                    for (i, preset) in PRESETS.iter().enumerate() {
                        ui.selectable_value(&mut self.selected, i, preset.name);
                    }
                });
            let preset = &PRESETS[self.selected];
            ui.strong(preset.category);
            ui.label(preset.objective);
            ui.hyperlink_to(preset.source, preset.url);
            egui::Grid::new("literature-shells").striped(true).show(ui, |ui| {
                for title in ["Altitude", "Inclination", "Walker T/P/F"] { ui.strong(title); }
                ui.end_row();
                for shell in preset.shells {
                    ui.label(format!("{:.3} km", shell.altitude_km));
                    ui.label(format!("{:.1}°", shell.inclination));
                    ui.label(format!("{}/{}/{}", shell.planes * shell.satellites_per_plane, shell.planes, shell.phasing));
                    ui.end_row();
                }
            });
            ui.small("T/P/F = satellites / planes / published phasing. The editor reverses plane numbering, so its F can differ; the geometry is preserved.");
            ui.label(preset.assumptions);
            if let Some(width) = preset.swath_km {
                ui.checkbox(&mut self.use_swath, format!("Use paper's {width:.0} km swath (all Earth views in this tab)"));
            } else {
                ui.small(format!("Keeps the current {:.1} km pushbroom swath; no circular-footprint conversion is applied.", imaging.swath_km));
            }
            ui.checkbox(&mut self.j2, "Use J2 propagation");
            ui.small(if self.j2 {
                "J2 matches the papers' perturbation choice. Use Sampled latitude forecast below; analytical guarantees and the point sampler require Keplerian orbits."
            } else {
                "Keplerian comparison: published altitudes are retained, not snapped to a repeating orbit."
            });
            ui.small("Uses the visualizer's Earth constants and current epoch. Published revisit values and repeat cycles are not certified here.");
            ui.checkbox(&mut self.hide_others, "Hide other constellations when adding");
            if ui.button("Add literature preset").clicked() { self.add(p, imaging); }
            if let Some(status) = &self.status { ui.label(status); }
        });
    }
}

#[cfg(test)]
mod tests;
