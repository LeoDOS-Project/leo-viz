//! Earth observation controls and asynchronous analytical point queries.
mod data;
mod designer;
mod literature;
mod sidebar;
use crate::{celestial::CelestialBody, config::TabConfig, projection::ProjectionKind};
use chrono::{DateTime, Utc};
use designer::Designer;
use eframe::egui;
#[cfg(not(target_arch = "wasm32"))]
use leo_sim::geometry::gmst;
use leo_sim::observation::{Model, PointReport};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc, Arc,
};

struct Pending {
    receiver: mpsc::Receiver<Result<PointReport, String>>,
    cancel: Arc<AtomicBool>,
    progress: Arc<AtomicUsize>,
}
impl Drop for Pending {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

pub(crate) struct ForecastRequest {
    pub planet: usize,
    pub cycle_s: Option<f64>,
}

pub(crate) struct Panel {
    designer: Designer,
    data: data::Panel,
    literature: literature::Literature,
    whole_earth: bool,
    latitude_band: f64,
    pub open: bool,
    planet: usize,
    scope: Option<(usize, usize, u64)>,
    model: Option<Result<Model, String>>,
    armed: bool,
    selected: Option<(f64, f64)>,
    needs_run: bool,
    days: f64,
    pending: Option<Pending>,
    report: Option<PointReport>,
    error: Option<String>,
    start_unix: f64,
}
impl Default for Panel {
    fn default() -> Self {
        Self {
            designer: Designer::default(),
            data: data::Panel::default(),
            literature: literature::Literature::default(),
            whole_earth: true,
            latitude_band: 60.0,
            open: true,
            planet: 0,
            scope: None,
            model: None,
            armed: false,
            selected: None,
            needs_run: false,
            days: 7.0,
            pending: None,
            report: None,
            error: None,
            start_unix: 0.0,
        }
    }
}

pub(crate) fn duration(seconds: f64) -> String {
    if !seconds.is_finite() {
        "∞ (never observed)".into()
    } else if seconds >= 86400.0 {
        format!("{:.3} days", seconds / 86400.0)
    } else if seconds >= 3600.0 {
        format!("{:.3} hours", seconds / 3600.0)
    } else {
        format!("{:.2} minutes", seconds / 60.0)
    }
}

fn coverage_ui(ui: &mut egui::Ui, model: &Model, latitude: f64) {
    match model.repeat {
        Some(t) => {
            ui.strong(format!("Ground-track repeat cycle T: {}", duration(t)));
            ui.small("Common cycle for every visible satellite, within 10⁻⁹ rad phase tolerance. A shorter constellation pattern repeat may exist.");
        }
        None => {
            ui.strong("Ground-track repeat cycle: none found within 30 sidereal days");
            ui.small("Use the repeat-orbit designer to select compatible altitudes. Adding satellites alone does not establish an orbital repeat.");
        }
    }
    ui.label(format!(
        "Geometric latitude reach: ±{:.3}°",
        model.latitude_limit.to_degrees()
    ));
    if latitude.to_radians() > model.latitude_limit + 1e-12 {
        ui.strong("Coverage impossible for this region");
        ui.label("Some latitudes lie beyond every visible satellite's swath.");
    } else if let Some(g) = model.coverage_guarantee(latitude.to_radians()) {
        ui.strong("Coverage guaranteed for the selected region");
        ui.label(format!(
            "{}: {}{}",
            if g.exact {
                "Minimum full-coverage interval"
            } else {
                "Guaranteed full-coverage interval"
            },
            if g.exact { "" } else { "≤ " },
            duration(g.interval_s)
        ));
        if !g.exact {
            ui.small("Upper bound from repeating-track spacing. The shortest interval and worst-point gap have not been calculated.");
        }
        ui.small(
            "Every point is observed in every interval of this length under continuous imaging.",
        );
    } else {
        ui.strong("Coverage guarantee: unverified");
        ui.label("The implemented sufficient conditions do not prove this region is covered. This is not a finding of coverage holes.");
    }
    let (a, b) = model.orbit_period_range;
    ui.small(if (a - b).abs() < 0.01 {
        format!("Orbital period: {}", duration(a))
    } else {
        format!("Orbital periods: {} – {}", duration(a), duration(b))
    });
}

fn equations_ui(ui: &mut egui::Ui) {
    egui::CollapsingHeader::new("Equations and assumptions").show(ui,|ui| {
        ui.label("Repeat time, a sufficient coverage interval, and the shortest coverage interval are separate quantities. Repetition alone can leave permanent holes.");
        ui.monospace("T = m T_orbit = d D    T_orbit = 2π √(a³/μ)");
        ui.monospace("h = [μ (dD / (2πm))²]^(1/3) − R");
        ui.label("m and d are coprime revolution/day counts; D is a sidereal day (23 h 56 min), a = R+h. These circular Keplerian repeat altitudes do not depend on inclination. The designer selects an exact ratio; it does not silently round existing orbits.");
        ui.monospace("β = w / (2R)    N ≥ ceil(2πR / (m w))");
        ui.label("An evenly phased, single-plane baseline has mN distinct ascending crossing longitudes if gcd(N,d)=1. For multi-day cycles the generator increases N until coprime. This sufficient count covers the central latitude band ±asin(|sin i|); at i=90° that is the full Earth.");
        ui.label("At each latitude, the nearest ascending crossing is at distance ≤πR/(mN). If that is ≤w/2, the closest point on the closed track is also within the swath and satisfies the perpendicular scanline condition. Thus every point is observed within each repeat cycle. This does not claim the shortest interval or fewest satellites.");
        ui.label("For arbitrary layouts, common-rate, common-inclination subsets are checked using their actual ascending-track longitude gaps. Maximum gap ≤w/R is a sufficient coverage test. Failure leaves coverage unverified; descending tracks may fill additional gaps.");
        ui.monospace("Equatorial exact interval = max(Δθ) / |s n − ω|");
        ui.label("Sort satellite phases including the wrap-around gap; s is +1 prograde or −1 retrograde, n=√(μ/a³). This exact special case covers |latitude|≤β; its ground-track repeat is the synodic period 2π/|s n−ω|, without requiring an integer-day resonance.");
        ui.monospace("Point events: x · q′(t) = 0,  x · q(t) ≥ cos β");
        ui.label("x is the ground point and q(t) the Earth-fixed subpoint. Harmonics at n, n−ω and n+ω locate scanline crossings by root isolation (0.1 ms tolerance), rather than fixed time steps.");
        ui.monospace("G = max(t₂−t₁, …, t₁+T−t_last)");
        ui.label("The point sampler includes the boundary gap only with a repeat cycle. Otherwise it reports a finite-horizon result. A sampled regional maximum is not a guarantee for all points.");
        ui.small("Spherical Earth, circular Keplerian orbits, constant spin and continuous nadir pushbroom imaging. J2 is not part of this designer. Illumination, weather, tasking and resources are excluded from geometric guarantees.");
    });
}

impl Panel {
    pub fn armed_for(&self, tab: usize, planet: usize) -> bool {
        self.open && self.armed && self.scope.is_some_and(|(t, p, _)| (t, p) == (tab, planet))
    }
    pub fn marker_for(&self, tab: usize, planet: usize) -> Option<(f64, f64)> {
        if self.open && self.scope.is_some_and(|(t, p, _)| (t, p) == (tab, planet)) {
            self.selected
        } else {
            None
        }
    }
    pub fn pick(&mut self, tab: usize, planet: usize, latitude: f64, longitude: f64) {
        if self.armed_for(tab, planet) {
            self.selected = Some((latitude, longitude));
            self.armed = false;
            self.pending = None;
            self.report = None;
            self.error = None;
            self.needs_run = true;
        }
    }
    fn refresh_scope(&mut self, scope: (usize, usize, u64), tab: &TabConfig) {
        if self.scope == Some(scope) {
            return;
        }
        if self
            .scope
            .is_some_and(|(t, p, _)| (t, p) != (scope.0, scope.1))
        {
            self.selected = None;
        }
        self.armed = false;
        self.needs_run = false;
        self.pending = None;
        self.report = None;
        self.error = None;
        self.scope = Some(scope);
        self.model=Some(crate::revisit::orbits(&tab.planets[self.planet]).and_then(|orbits| {
            let elements=orbits.into_iter().map(|o| match o {
                crate::revisit::Orbit::Analytical(e)=>Ok(e),
                crate::revisit::Orbit::Tle(_)=>Err("TLE/SGP4 satellites are active. Use the sampled forecast, or hide them for circular analytical calculations.".to_owned()),
            }).collect::<Result<Vec<_>,String>>()?;
            Model::new(elements,tab.settings.pushbroom.swath_km,CelestialBody::Earth.radius_km())
        }));
    }

    /// Open a separate sampled regional forecast, using one repeat cycle when known.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        tab: &mut TabConfig,
        tab_idx: usize,
        epoch: DateTime<Utc>,
    ) -> Option<ForecastRequest> {
        let context = ui.ctx().clone();
        let ctx = &context;
        if !self.open {
            self.armed = false;
            self.pending = None;
            self.needs_run = false;
            return None;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.armed = false;
        }
        let mut forecast = None;
        ui.push_id("earth-observation-model", |ui| {
            if !self.ensure_earth(tab) {
                ui.label("Select a tab with an Earth globe."); return;
            }
            tab.settings.pushbroom.ui(ui,true);
            self.literature.ui(ui, &mut tab.planets[self.planet], &mut tab.settings.pushbroom);
            if let Some(band)=self.designer.ui(ui,&mut tab.planets[self.planet],tab.settings.pushbroom.swath_km) {
                self.whole_earth=band>=90.0;
                self.latitude_band=band;
            }
            let key=crate::revisit::fingerprint(&tab.planets[self.planet],tab.settings.pushbroom.swath_km,epoch);
            self.refresh_scope((tab_idx,self.planet,key),tab);
            ui.separator();
            ui.horizontal_wrapped(|ui| {
                ui.label("Coverage region");
                ui.selectable_value(&mut self.whole_earth,true,"Full Earth");
                ui.selectable_value(&mut self.whole_earth,false,"Latitude band");
                if !self.whole_earth {ui.label("±");ui.add(egui::DragValue::new(&mut self.latitude_band).range(0.0..=90.0).speed(0.1).suffix("°"));}
            });
            let latitude=if self.whole_earth {90.0} else {self.latitude_band};
            match self.model.as_ref().unwrap() {
                Ok(model)=>coverage_ui(ui,model,latitude),
                Err(e)=>{ui.strong("Repeat and coverage calculations unavailable");ui.label(e);}
            }
            equations_ui(ui);
            ui.separator();
            let globe=tab.planets[self.planet].projection_override.unwrap_or(tab.settings.planet_projection)==ProjectionKind::Orthographic;
            let supported=self.model.as_ref().is_some_and(|m|m.is_ok());
            ui.horizontal(|ui| {
                if ui.add_enabled(globe && supported,egui::Button::new(if self.armed {"Cancel sampler"} else {"Sampler"})).clicked() {self.armed=!self.armed;}
                if self.armed {ui.label("Click on this Earth globe · Esc cancels");}
            });
            if !globe {self.armed=false;ui.small("Switch this Earth view to the 3D globe to pick a point.");}
            if let Some((lat,lon))=self.selected {
                ui.label(format!("Latitude {lat:+.5}° · Longitude {lon:+.5}°"));
                ui.horizontal(|ui| {
                    ui.label("Non-repeating horizon");
                    if ui.add(egui::DragValue::new(&mut self.days).range(1.0..=30.0).suffix(" days")).changed() {self.pending=None;self.report=None;}
                    if ui.add_enabled(supported && self.pending.is_none(),egui::Button::new("Recalculate")).clicked() {self.needs_run=true;}
                });
                if self.needs_run {
                    self.needs_run=false;
                    self.report=None;self.error=None;
                    #[cfg(not(target_arch="wasm32"))]
                    if let Some(Ok(model))=&self.model {
                        let model=model.clone();let start=tab.settings.time;let earth_angle=gmst(epoch.timestamp_millis() as f64/1000.0);let horizon=self.days*86400.0;
                        self.start_unix=epoch.timestamp_millis() as f64/1000.0+start;
                        let (sender,receiver)=mpsc::channel();
                        let cancel=Arc::new(AtomicBool::new(false));let progress=Arc::new(AtomicUsize::new(0));
                        let c=Arc::clone(&cancel);let p=Arc::clone(&progress);let context=ctx.clone();
                        std::thread::spawn(move || {
                            let result=model.point(lat,lon,start,earth_angle,horizon,|fraction| {p.store((fraction*1000.0) as usize,Ordering::Relaxed);!c.load(Ordering::Relaxed)});
                            let _=sender.send(result);context.request_repaint();
                        });
                        self.pending=Some(Pending{receiver,cancel,progress});
                    }
                    #[cfg(target_arch="wasm32")]
                    {self.error=Some("Analytical point calculations currently require the native app.".into());}
                }
            }
            if let Some(p)=&self.pending {
                match p.receiver.try_recv() {
                    Ok(result)=>{self.pending=None;match result {Ok(r)=>self.report=Some(r),Err(e)=>self.error=Some(e)}},
                    Err(mpsc::TryRecvError::Disconnected)=>{self.pending=None;self.error=Some("Observation worker stopped unexpectedly.".into());},
                    Err(mpsc::TryRecvError::Empty)=>{
                        ui.add(egui::ProgressBar::new(p.progress.load(Ordering::Relaxed) as f32/1000.0).show_percentage());
                        if ui.button("Cancel calculation").clicked() {self.pending=None;}
                        ctx.request_repaint_after(std::time::Duration::from_millis(100));
                    }
                }
            }
            if let Some(e)=&self.error {ui.colored_label(egui::Color32::LIGHT_RED,e);}
            if let Some(r)=&self.report {
                if let Some(g)=r.exact_gap {ui.strong(format!("Maximum gap: {}",duration(g)));ui.small("Analytical result for this point, valid for all time under the stated model.");}
                else {
                    ui.strong(format!("{}: {}",if r.cyclic {"Cyclic maximum gap"} else {"Largest observed gap"},r.max_gap.map(duration).unwrap_or("Not established (fewer than two events)".into())));
                    ui.label(format!("{} observation events over {}",r.events,duration(r.duration)));
                    if !r.cyclic {
                        ui.label(format!("Longest silence in horizon: {}",duration(r.longest_silence)));
                        ui.small("Includes boundary silence. The all-time maximum is not established; an unseen point is not necessarily unreachable.");
                    } else {ui.small("Includes the cycle boundary; repeat equality is checked to floating-point phase tolerance.");}
                    let start=DateTime::from_timestamp(self.start_unix as i64,0).unwrap_or_default();
                    ui.small(format!("Snapshot from {} UTC. Recalculate to use the current simulation time.",start.format("%Y-%m-%d %H:%M")));
                }
            }
            ui.separator();
            let cycle_s=self.model.as_ref().and_then(|m|m.as_ref().ok()).and_then(|m|m.repeat);
            if ui.button(if cycle_s.is_some() {"Sample one repeat cycle…"} else {"Sampled latitude forecast…"})
                .on_hover_text("Numerical regional sampling, separate from the analytical guarantee. Opens a forecast using this Earth view and one common repeat cycle when known.").clicked() {
                forecast=Some(ForecastRequest{planet:self.planet,cycle_s});
            }
        });
        forecast
    }
}

#[cfg(test)]
mod tests;
