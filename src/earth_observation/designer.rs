use crate::{
    celestial::CelestialBody,
    config::{ConstellationConfig, PlanetConfig, Preset, Propagator},
};
use eframe::egui;
use leo_sim::repeat_ground_track::{self as rgt, Candidate, RepeatRatio, Search};

pub(super) struct Designer {
    pub search: Search,
    pub inclination: f64,
    cached: Option<Search>,
    candidates: Vec<Candidate>,
    selected: usize,
    constellation: usize,
    pub hide_others: bool,
    status: Option<String>,
}
impl Default for Designer {
    fn default() -> Self {
        Self {
            search: Search::default(),
            inclination: 90.0,
            cached: None,
            candidates: Vec::new(),
            selected: 0,
            constellation: 0,
            hide_others: true,
            status: None,
        }
    }
}

fn set_orbit(c: &mut ConstellationConfig, candidate: Candidate, inclination: f64) {
    c.altitude_km = candidate.altitude_km;
    c.inclination = inclination;
    c.sso = false;
    c.eccentricity = 0.0;
    c.propagator = Propagator::Keplerian;
    c.numerical = None;
    c.physics_state.clear();
    c.preset = Preset::None;
    c.label = Some(format!(
        "Repeat {}/{}",
        candidate.ratio.revolutions, candidate.ratio.days
    ));
}
pub(super) fn clear_tracks(p: &mut PlanetConfig) {
    p.ground_track_history.clear();
    p.pushbroom_history = Default::default();
}

pub(super) fn hide_other_constellations(p: &mut PlanetConfig) {
    for c in &mut p.constellations {
        c.hidden = true;
    }
    for (enabled, _, _) in p.tle_selections.values_mut() {
        *enabled = false;
    }
}

impl Designer {
    fn refresh(&mut self) {
        if self.cached == Some(self.search) {
            return;
        }
        self.cached = Some(self.search);
        self.selected = 0;
        self.status = None;
        match rgt::candidates(
            self.search,
            CelestialBody::Earth.mu(),
            CelestialBody::Earth.radius_km(),
        ) {
            Ok(c) => self.candidates = c,
            Err(e) => {
                self.candidates.clear();
                self.status = Some(e);
            }
        }
    }
    pub fn daily_preset(&mut self) {
        let c = rgt::design(
            RepeatRatio::new(15, 1).unwrap(),
            CelestialBody::Earth.mu(),
            CelestialBody::Earth.radius_km(),
        )
        .unwrap();
        self.search = Search {
            nominal_altitude_km: c.altitude_km,
            tolerance_km: 100.0,
            max_days: 1,
        };
        self.inclination = 90.0;
        self.refresh();
    }
    pub fn selected(&self) -> Option<Candidate> {
        self.candidates.get(self.selected).copied()
    }

    /// Add a new ordinary constellation so existing configuration controls and
    /// renderers remain authoritative. Hidden previous configurations are kept.
    pub fn add_baseline(&mut self, p: &mut PlanetConfig, swath: f64) -> Result<f64, String> {
        let candidate = self.selected().ok_or("Choose a repeat orbit first.")?;
        let n = rgt::baseline_count(candidate.ratio, swath, CelestialBody::Earth.radius_km())?;
        if !self.inclination.is_finite() || !(0.0..=180.0).contains(&self.inclination) {
            return Err("Choose an inclination between 0° and 180°.".into());
        }
        if self.hide_others {
            hide_other_constellations(p);
        }
        p.add_constellation();
        self.constellation = p.constellations.len() - 1;
        let c = &mut p.constellations[self.constellation];
        set_orbit(c, candidate, self.inclination);
        c.num_planes = 1;
        c.sats_per_plane = n;
        c.phasing = 0.0;
        c.label = Some(format!(
            "Repeat baseline {}/{}",
            candidate.ratio.revolutions, candidate.ratio.days
        ));
        clear_tracks(p);
        Ok(self
            .inclination
            .to_radians()
            .sin()
            .abs()
            .asin()
            .to_degrees())
    }

    /// Returns the latitude band of a newly added guaranteed baseline.
    pub fn ui(&mut self, ui: &mut egui::Ui, p: &mut PlanetConfig, swath: f64) -> Option<f64> {
        let mut added = None;
        egui::CollapsingHeader::new("Repeat-orbit designer").show(ui,|ui| {
            ui.small("Circular Keplerian design. Choose an exact repeat orbit near your desired altitude, then apply it or add a coverage baseline.");
            ui.horizontal(|ui| {
                ui.label("Inclination");ui.add(egui::DragValue::new(&mut self.inclination).range(0.0..=180.0).speed(0.1).suffix("°"));
                if ui.button("Daily polar preset").clicked() {self.daily_preset();}
            });
            ui.horizontal(|ui| {
                ui.label("Approx. altitude");ui.add(egui::DragValue::new(&mut self.search.nominal_altitude_km).range(100.0..=2000.0).speed(1.0).suffix(" km"));
                ui.label("±");ui.add(egui::DragValue::new(&mut self.search.tolerance_km).range(0.0..=1900.0).speed(1.0).suffix(" km"));
            });
            ui.horizontal(|ui| {
                ui.label("Maximum repeat cycle");ui.add(egui::DragValue::new(&mut self.search.max_days).range(1..=rgt::MAX_REPEAT_DAYS).suffix(" sidereal days"));
            });
            self.refresh();
            if self.candidates.is_empty() {
                ui.label("No compatible orbit in this range. Widen the altitude tolerance or repeat-cycle limit.");
            } else {
                ui.small(format!("{} candidates, nearest altitude first. m counts revolutions, not satellites.",self.candidates.len()));
                egui::ScrollArea::vertical().id_salt("repeat-candidates").max_height(145.0).show(ui,|ui| {
                    egui::Grid::new("repeat-orbits").striped(true).show(ui,|ui| {
                        for title in ["m / d","Altitude","Change","Repeat cycle"] {ui.strong(title);}ui.end_row();
                        for (index,c) in self.candidates.iter().enumerate() {
                            ui.selectable_value(&mut self.selected,index,format!("{} / {}",c.ratio.revolutions,c.ratio.days));
                            ui.label(format!("{:.3} km",c.altitude_km));
                            ui.label(format!("{:+.3} km",c.altitude_km-self.search.nominal_altitude_km));
                            ui.label(super::duration(c.ratio.duration()));ui.end_row();
                        }
                    });
                });
                if let Some(c)=self.selected() {
                    ui.label(format!("Selected altitude: {:.9} km · orbit {}",c.altitude_km,super::duration(c.ratio.orbital_period())));
                    ui.small("Applied altitude retains full precision. Editing or rounding it may break the repeat condition. Inclination changes latitude reach, not the Keplerian repeat altitude.");
                    match rgt::baseline_count(c.ratio,swath,CelestialBody::Earth.radius_km()) {
                        Ok(n)=>{
                            let reach=self.inclination.to_radians().sin().abs().asin().to_degrees();
                            ui.label(format!("Sufficient baseline: 1 plane × {n} evenly spaced satellites, coverage within {} over ±{reach:.3}°.",super::duration(c.ratio.duration())));
                            ui.small("A sufficient bound under continuous imaging; neither the satellite count nor the coverage interval is claimed to be minimal.");
                            ui.checkbox(&mut self.hide_others,"Hide other constellations when adding").on_hover_text("Keeps their configurations and TLE data, but removes them from the displayed constellation and its repeat calculation.");
                            if ui.button("Add guaranteed baseline").clicked() {
                                match self.add_baseline(p,swath) {Ok(band)=>{added=Some(band);self.status=Some(format!("Added {n} satellites at {:.6} km.",c.altitude_km));},Err(e)=>self.status=Some(e)}
                            }
                        }
                        Err(e)=>{ui.label(e);}
                    }
                    if !p.constellations.is_empty() {
                        self.constellation=self.constellation.min(p.constellations.len()-1);
                        ui.separator();
                        egui::ComboBox::from_id_salt("repeat-apply-target").selected_text(format!("{}: {}",self.constellation+1,p.constellations[self.constellation].preset_name())).show_ui(ui,|ui| {
                            for (i,c) in p.constellations.iter().enumerate() {ui.selectable_value(&mut self.constellation,i,format!("{}: {}",i+1,c.preset_name()));}
                        });
                        #[cfg(not(target_arch="wasm32"))]
                        let editable=p.constellations[self.constellation].cfs.is_none();
                        #[cfg(target_arch="wasm32")]
                        let editable=true;
                        if ui.add_enabled(editable,egui::Button::new("Apply orbit to selected constellation")).clicked() {
                            set_orbit(&mut p.constellations[self.constellation],c,self.inclination);
                            clear_tracks(p);
                            self.status=Some("Applied circular Keplerian orbit. Existing satellite counts, planes and phasing are retained; coverage is checked separately.".into());
                        }
                        if !editable {ui.small("Stop this constellation's cFS run before changing its orbit.");}
                    }
                }
            }
            if let Some(message)=&self.status {ui.label(message);}
        });
        added
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn applying_a_repeat_orbit_preserves_layout_and_communication_configuration() {
        let candidate = rgt::design(
            RepeatRatio::new(15, 1).unwrap(),
            CelestialBody::Earth.mu(),
            CelestialBody::Earth.radius_km(),
        )
        .unwrap();
        let mut c = ConstellationConfig::new(5);
        c.num_planes = 4;
        c.sats_per_plane = 23;
        c.phasing = 0.3;
        c.raan_spacing = Some(15.0);
        c.sat_spacing_km = Some(120.0);
        c.eccentricity = 0.2;
        c.sso = true;
        c.propagator = Propagator::J2;
        c.link_budget.bandwidth_ghz = 3.0;
        set_orbit(&mut c, candidate, 70.0);
        assert_eq!((c.num_planes, c.sats_per_plane), (4, 23));
        assert_eq!(c.phasing, 0.3);
        assert_eq!(c.raan_spacing, Some(15.0));
        assert_eq!(c.sat_spacing_km, Some(120.0));
        assert_eq!(c.link_budget.bandwidth_ghz, 3.0);
        assert_eq!(c.color_offset, 5);
        assert_eq!(c.inclination, 70.0);
        assert_eq!(c.eccentricity, 0.0);
        assert!(!c.sso);
        assert!(c.propagator == Propagator::Keplerian);
        assert_eq!(c.altitude_km, candidate.altitude_km);
    }
}
