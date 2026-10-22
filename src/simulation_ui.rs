//! Thin GUI client for the same fixed-step engine used by the batch runner.
use crate::{celestial::CelestialBody, config::Propagator, walker::WalkerType, ViewerState};
use eframe::egui;
use leo_sim::{
    orbit::{Propagation, WalkerShell},
    GroundStation, OrbitEnvironment, Scenario, Simulator,
};

pub(crate) struct SimulationUi {
    pub open: bool,
    scenario_json: String,
    simulator: Option<Simulator<OrbitEnvironment>>,
    running: bool,
    steps_per_frame: usize,
    error: Option<String>,
    #[cfg(not(target_arch = "wasm32"))]
    scenario_path: String,
    #[cfg(not(target_arch = "wasm32"))]
    report_path: String,
}

impl Default for SimulationUi {
    fn default() -> Self {
        Self {
            open: false,
            scenario_json: include_str!("../crates/leo-sim/scenarios/smoke.json").into(),
            simulator: None,
            running: false,
            steps_per_frame: 10,
            error: None,
            #[cfg(not(target_arch = "wasm32"))]
            scenario_path: "crates/leo-sim/scenarios/smoke.json".into(),
            #[cfg(not(target_arch = "wasm32"))]
            report_path: "simulation-report.json".into(),
        }
    }
}

impl SimulationUi {
    fn reset(&mut self) {
        self.running = false;
        let result = serde_json::from_str::<Scenario>(&self.scenario_json)
            .map_err(|e| e.to_string())
            .and_then(|c| OrbitEnvironment::new(&c).and_then(|e| Simulator::new(c, e)));
        match result {
            Ok(sim) => {
                self.simulator = Some(sim);
                self.error = None;
            }
            Err(e) => {
                self.simulator = None;
                self.error = Some(e);
            }
        }
    }

    fn advance(&mut self, steps: usize) {
        if let Some(sim) = &mut self.simulator {
            for _ in 0..steps {
                if sim.time_s() >= sim.config().duration_s {
                    self.running = false;
                    break;
                }
                if let Err(e) = sim.step() {
                    self.error = Some(e);
                    self.running = false;
                    break;
                }
            }
        }
    }

    pub fn show(&mut self, ctx: &egui::Context, viewer: &ViewerState) {
        // Closing or hiding the panel does not change the experiment clock's rules.
        if self.running {
            self.advance(self.steps_per_frame);
            ctx.request_repaint();
        }
        let mut open = self.open;
        egui::Window::new("Infrastructure simulation").open(&mut open)
            .default_width(610.0).vscroll(true).show(ctx, |ui| {
                ui.label("Fixed simulation steps; rendering speed only controls how quickly the run is displayed.");
                if ui.button("Import initial Earth layout from active view").clicked() {
                    match scenario_from_viewer(viewer) {
                        Ok(c) => {self.scenario_json = serde_json::to_string_pretty(&c).unwrap(); self.reset();}
                        Err(e) => self.error = Some(e),
                    }
                }
                ui.small("Import copies Walker orbits and station coordinates. Resource budgets use the scenario defaults below; existing display physics and routing animations are separate.");
                #[cfg(not(target_arch = "wasm32"))]
                ui.horizontal(|ui| {
                    ui.text_edit_singleline(&mut self.scenario_path);
                    if ui.button("Load JSON").clicked() {
                        match std::fs::read_to_string(&self.scenario_path) {
                            Ok(text) => {self.scenario_json = text; self.reset();}
                            Err(e) => self.error = Some(e.to_string()),
                        }
                    }
                });
                egui::CollapsingHeader::new("Scenario JSON").show(ui, |ui| {
                    ui.add(egui::TextEdit::multiline(&mut self.scenario_json).code_editor().desired_rows(15).desired_width(f32::INFINITY));
                    if ui.button("Copy scenario").clicked() {ctx.copy_text(self.scenario_json.clone());}
                });
                ui.horizontal(|ui| {
                    if ui.button("Apply / reset").clicked() {self.reset();}
                    let ready = self.simulator.as_ref().is_some_and(|s|s.time_s()<s.config().duration_s);
                    if ui.add_enabled(ready, egui::Button::new(if self.running {"Pause"} else {"Run"})).clicked() {self.running = !self.running;}
                    if ui.add_enabled(ready && !self.running, egui::Button::new("Step")).clicked() {self.advance(1);}
                    ui.add(egui::DragValue::new(&mut self.steps_per_frame).range(1..=1000).suffix(" steps/frame"));
                });
                if let Some(e) = &self.error {ui.colored_label(egui::Color32::LIGHT_RED, e);}
                if let Some(sim) = &self.simulator {
                    let report = sim.report();
                    let c = sim.config();
                    ui.label(format!("t = {:.1} / {:.1} s · {} satellites · step {} s", sim.time_s(), c.duration_s, c.satellite_count(), c.step_s));
                    ui.label(format!("{} completed · {} rejected · {:.3} compute units · {:.3} MB ISL byte-hops · {:.3} MB downlink", report.metrics.completed_actions, report.metrics.rejected_actions, report.metrics.compute_work, report.metrics.isl_byte_hops/1e6, report.metrics.downlink_bytes/1e6));
                    // Render positions directly from the experiment's environment and clock.
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 220.0), egui::Sense::hover());
                    let painter = ui.painter_at(rect);
                    painter.rect_filled(rect, 0.0, egui::Color32::from_gray(15));
                    let point = |lat:f64, lon:f64| egui::pos2(rect.left()+(lon as f32+180.0)/360.0*rect.width(), rect.top()+(90.0-lat as f32)/180.0*rect.height());
                    for lat in [-60.0,0.0,60.0] {painter.line_segment([point(lat,-180.0),point(lat,180.0)],(1.0,egui::Color32::DARK_GRAY));}
                    for lon in [-120.0,-60.0,0.0,60.0,120.0] {painter.line_segment([point(-90.0,lon),point(90.0,lon)],(1.0,egui::Color32::DARK_GRAY));}
                    for s in report.satellites.iter().take(5000) {
                        let (lat,lon) = sim.environment().geographic_position(s.id, sim.time_s());
                        painter.circle_filled(point(lat,lon),2.0,egui::Color32::LIGHT_BLUE);
                    }
                    for g in &c.ground_stations {painter.circle_filled(point(g.lat_deg,g.lon_deg),4.0,egui::Color32::LIGHT_GREEN);}
                    ui.small("Earth-fixed longitude / latitude · blue: satellites (first 5,000) · green: ground stations");
                    egui::CollapsingHeader::new("Resource snapshot (first 64 satellites)").show(ui,|ui| {
                        egui::Grid::new("simulation_resources").striped(true).show(ui,|ui| {
                            ui.label("Satellite");ui.label("Storage + reservations");ui.label("Rolling compute seconds");ui.label("Battery J");ui.end_row();
                            for s in report.satellites.iter().take(64) {
                                ui.label(s.id.to_string());ui.label(s.stored_and_reserved_bytes.to_string());
                                ui.label(format!("{:.2}",s.rolling_compute_busy_s));
                                ui.label(s.energy_j.map_or_else(||"disabled".into(),|e|format!("{e:.1}")));ui.end_row();
                            }
                        });
                    });
                    egui::CollapsingHeader::new("Operations (first 100)").show(ui,|ui| {
                        for op in report.operations.iter().take(100) {ui.label(format!("{}: {:?}",op.id,op.status));}
                    });
                    let envelope = serde_json::json!({"simulator_version":leo_sim::VERSION,"source_revision":env!("GIT_HASH"),"scenario":c,"report":report});
                    if ui.button("Copy report JSON").clicked() {ctx.copy_text(serde_json::to_string_pretty(&envelope).unwrap());}
                    #[cfg(not(target_arch = "wasm32"))]
                    ui.horizontal(|ui| {
                        ui.text_edit_singleline(&mut self.report_path);
                        if ui.button("Save report").clicked() {
                            if let Err(e) = std::fs::write(&self.report_path, serde_json::to_string_pretty(&envelope).unwrap()) {self.error = Some(e.to_string());}
                        }
                    });
                }
            });
        self.open = open;
    }
}

fn scenario_from_viewer(viewer: &ViewerState) -> Result<Scenario, String> {
    let tab = viewer
        .tabs
        .get(viewer.active_tab_idx)
        .ok_or("No active view")?;
    let planet = tab
        .planets
        .iter()
        .find(|p| p.celestial_body == CelestialBody::Earth)
        .ok_or("The active view has no Earth")?;
    let mut shells = Vec::new();
    for c in &planet.constellations {
        let propagation = match c.propagator {
            Propagator::Keplerian => Propagation::Keplerian,
            Propagator::J2 => Propagation::J2,
            _ => return Err(
                "Experiment import currently supports Keplerian and analytical J2 Walker shells"
                    .into(),
            ),
        };
        if c.raan_spacing.is_some()
            || c.sat_spacing_km.is_some()
            || c.phasing.fract() != 0.0
            || c.phasing < 0.0
            || c.drag_enabled
        {
            return Err(
                "Experiment import requires full Walker shells, integer phasing and no drag".into(),
            );
        }
        shells.push(WalkerShell {
            planes: c.num_planes,
            satellites_per_plane: c.sats_per_plane,
            altitude_km: c.altitude_km,
            inclination_deg: c.inclination,
            phasing: c.phasing as usize,
            raan_offset_deg: c.raan_offset,
            star: c.walker_type == WalkerType::Star,
            eccentricity: c.eccentricity,
            argument_deg: c.arg_periapsis,
            propagation,
        });
    }
    let scenario = Scenario {
        epoch_unix_s: viewer.start_timestamp.timestamp_millis() as f64 / 1000.0,
        shells,
        ground_stations: planet
            .ground_stations
            .iter()
            .map(|g| GroundStation {
                name: g.name.clone(),
                lat_deg: g.lat,
                lon_deg: g.lon,
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    };
    scenario.validate()?;
    Ok(scenario)
}
