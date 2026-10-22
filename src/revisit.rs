//! Background revisit forecast for the Earth constellations in the active tab.
#[cfg(test)]
mod tests;
use crate::{
    celestial::CelestialBody,
    config::{PlanetConfig, Propagator, TabConfig},
    tle::{TleLoadState, TlePreset, TleSatellite},
};
use chrono::{DateTime, Utc};
use eframe::egui;
use leo_sim::{geometry::Vector, orbit::Elements, revisit::PointStats};
#[cfg(any(test, not(target_arch = "wasm32")))]
use leo_sim::{
    geometry::{gmst, norm},
    revisit::{unit, Accumulator, GroundPoint},
};
use std::{
    hash::{Hash, Hasher},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc, Arc,
    },
};

pub(crate) enum Orbit {
    Analytical(Elements),
    Tle(TleSatellite),
}
impl Orbit {
    pub(crate) fn position(&self, time: f64, epoch: f64) -> Result<Vector, String> {
        match self {
            Self::Analytical(e) => Ok(e.sample(time).xyz),
            Self::Tle(t) => t
                .constants
                .propagate(sgp4::MinutesSinceEpoch(
                    (epoch + time) / 60.0 - t.epoch_minutes,
                ))
                .map(|p| [p.position[0], p.position[2], -p.position[1]])
                .map_err(|e| format!("TLE propagation failed for {}: {e:?}", t.name)),
        }
    }
    #[cfg(any(test, not(target_arch = "wasm32")))]
    fn inclination(&self) -> f64 {
        match self {
            Self::Analytical(e) => e.inclination_rad.to_degrees(),
            Self::Tle(t) => t.inclination_deg,
        }
    }
}

fn tle_satellites(p: &PlanetConfig) -> impl Iterator<Item = &TleSatellite> {
    TlePreset::ALL
        .iter()
        .filter(|preset| !preset.is_debris())
        .flat_map(move |preset| {
            let mut selected = Vec::new();
            if let Some((true, TleLoadState::Loaded { satellites }, shells)) =
                p.tle_selections.get(preset)
            {
                let indices: Option<std::collections::HashSet<usize>> =
                    shells.as_ref().map(|shells| {
                        shells
                            .iter()
                            .filter(|s| s.selected)
                            .flat_map(|s| s.satellite_indices.iter().copied())
                            .collect()
                    });
                for (i, sat) in satellites.iter().enumerate() {
                    if indices.as_ref().is_none_or(|indices| indices.contains(&i)) {
                        selected.push(sat);
                    }
                }
            }
            selected
        })
}

pub(crate) fn fingerprint(p: &PlanetConfig, swath: f64, epoch: DateTime<Utc>) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    p.name.hash(&mut h);
    swath.to_bits().hash(&mut h);
    epoch.timestamp_millis().hash(&mut h);
    for c in p.constellations.iter().filter(|c| !c.hidden) {
        c.preset_name().hash(&mut h);
        c.sats_per_plane.hash(&mut h);
        c.num_planes.hash(&mut h);
        (c.propagator as u8).hash(&mut h);
        (c.walker_type as u8).hash(&mut h);
        for n in [
            c.altitude_km,
            c.inclination,
            c.phasing,
            c.raan_offset,
            c.eccentricity,
            c.arg_periapsis,
        ] {
            n.to_bits().hash(&mut h);
        }
        c.raan_spacing.map(f64::to_bits).hash(&mut h);
        c.sat_spacing_km.map(f64::to_bits).hash(&mut h);
    }
    for sat in tle_satellites(p) {
        sat.norad_id.hash(&mut h);
        sat.epoch_minutes.to_bits().hash(&mut h);
        sat.mean_motion.to_bits().hash(&mut h);
    }
    h.finish()
}

pub(crate) fn orbits(p: &PlanetConfig) -> Result<Vec<Orbit>, String> {
    Ok(orbit_groups(p)?
        .into_iter()
        .flat_map(|(_, orbits)| orbits)
        .collect())
}

/// Shared selection and propagation inputs for coverage and acquisition estimates.
pub(crate) fn orbit_groups(p: &PlanetConfig) -> Result<Vec<(String, Vec<Orbit>)>, String> {
    let body = CelestialBody::Earth;
    let mut result = Vec::new();
    for (index, c) in p
        .constellations
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.hidden)
    {
        if !matches!(c.propagator, Propagator::Keplerian | Propagator::J2) {
            return Err("This estimate supports Keplerian, J2 and TLE/SGP4 orbits. Switch visible numerical/42 constellations to Keplerian or J2 to calculate.".into());
        }
        if c.altitude_km < 100.0
            || c.num_planes == 0
            || c.sats_per_plane == 0
            || !(0.0..0.95).contains(&c.eccentricity)
        {
            return Err("Revisit forecasting needs orbiting satellites (altitude ≥100 km, positive counts and eccentricity <0.95).".into());
        }
        let wc = c.constellation(
            body.radius_km(),
            body.mu(),
            body.j2(),
            body.equatorial_radius_km(),
        );
        let mut orbits = Vec::new();
        for plane in 0..c.num_planes {
            for sat in 0..c.sats_per_plane {
                orbits.push(Orbit::Analytical(wc.orbital_elements(plane, sat)));
            }
        }
        result.push((format!("{} ({})", c.preset_name(), index + 1), orbits));
    }
    let tles: Vec<_> = tle_satellites(p).cloned().map(Orbit::Tle).collect();
    if !tles.is_empty() {
        result.push(("Selected TLE satellites".into(), tles));
    }
    if result.is_empty() {
        return Err("Add or show an Earth constellation first. Debris is excluded.".into());
    }
    Ok(result)
}

#[derive(Clone, Copy, PartialEq)]
struct Options {
    days: f64,
    longitudes: usize,
    latitude: f64,
    longitude: f64,
    fine: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            days: 7.0,
            longitudes: 72,
            latitude: 57.7,
            longitude: 11.97,
            fine: false,
        }
    }
}
struct Row {
    label: String,
    indices: std::ops::Range<usize>,
}
#[cfg(any(test, not(target_arch = "wasm32")))]
fn ground_points(limit: f64, opts: Options) -> (Vec<GroundPoint>, Vec<Row>) {
    let mut points = vec![GroundPoint {
        lat: opts.latitude,
        lon: opts.longitude,
    }];
    let mut rows = vec![Row {
        label: format!("Point ({:.2}°, {:.2}°)", opts.latitude, opts.longitude),
        indices: 0..1,
    }];
    let mut latitudes: Vec<f64> = (-80..=80)
        .step_by(10)
        .map(|x| x as f64)
        .filter(|x| x.abs() < limit)
        .collect();
    // Include near-edge latitudes as well as the ten-degree rows. The exact
    // geometric boundary is tangent to the swath, so sample just inside it.
    latitudes.extend([-limit * 0.999, 0.0, limit * 0.999]);
    latitudes.sort_by(f64::total_cmp);
    latitudes.dedup_by(|a, b| (*a - *b).abs() < 0.01);
    for lat in latitudes {
        let start = points.len();
        for i in 0..opts.longitudes {
            // Deterministic jitter in longitude strata avoids regular-grid
            // resonance with Walker plane spacing and repeat ground tracks.
            let offset = (0.5 + i as f64 * 0.6180339887498949).fract();
            points.push(GroundPoint {
                lat,
                lon: -180.0 + 360.0 * (i as f64 + offset) / opts.longitudes as f64,
            });
        }
        rows.push(Row {
            label: format!("{lat:+.2}°"),
            indices: start..points.len(),
        });
    }
    (points, rows)
}
#[cfg(any(test, not(target_arch = "wasm32")))]
struct Request {
    orbits: Vec<Orbit>,
    epoch: f64,
    start: f64,
    swath: f64,
    key: u64,
    options: Options,
}
struct Report {
    stats: Vec<PointStats>,
    rows: Vec<Row>,
    key: u64,
    options: Options,
    start_unix: f64,
    satellites: usize,
    swath: f64,
    limit: f64,
    step: f64,
}
#[cfg(any(test, not(target_arch = "wasm32")))]
fn calculate(r: Request, cancel: &AtomicBool, progress: &AtomicUsize) -> Result<Report, String> {
    let radius = CelestialBody::Earth.radius_km();
    let duration = r.options.days * 86400.0;
    let half = r.swath / (2.0 * radius);
    let max_inc = r
        .orbits
        .iter()
        .map(|o| o.inclination().to_radians().sin().abs().asin().to_degrees())
        .fold(0.0_f64, f64::max);
    let limit = (max_inc + half.to_degrees()).min(90.0);
    let (points, rows) = ground_points(limit, r.options);
    // Shorter segments for narrow swaths; optional half-step convergence check.
    let nominal_step =
        (20.0 * (r.swath / 20.0).sqrt()).clamp(2.0, 20.0) / if r.options.fine { 2.0 } else { 1.0 };
    let steps = (duration / nominal_step).ceil() as usize;
    let step = duration / steps as f64;
    let rotations: Vec<_> = (0..=steps)
        .map(|i| gmst(r.epoch + r.start + i as f64 * step).sin_cos())
        .collect();
    let mut acc = Accumulator::new(&points);
    let fixed = |pos: Vector, i: usize| -> Result<(Vector, f64), String> {
        let distance = norm(pos);
        if !distance.is_finite() || distance <= radius {
            return Err("Propagation produced a satellite at or below Earth's surface.".into());
        }
        let (s, c) = rotations[i];
        Ok((
            unit([c * pos[0] - s * pos[2], pos[1], s * pos[0] + c * pos[2]]),
            (radius / distance).acos(),
        ))
    };
    for (sat, orbit) in r.orbits.iter().enumerate() {
        let mut previous = fixed(orbit.position(r.start, r.epoch)?, 0)?;
        for i in 1..=steps {
            if i % 256 == 1 {
                if cancel.load(Ordering::Relaxed) {
                    return Err("Calculation cancelled.".into());
                }
                progress.store(
                    (sat * steps + i) * 1000 / (r.orbits.len() * steps),
                    Ordering::Relaxed,
                );
            }
            let current = fixed(orbit.position(r.start + i as f64 * step, r.epoch)?, i)?;
            acc.segment(
                previous.0,
                current.0,
                (i - 1) as f64 * step,
                i as f64 * step,
                half.min(previous.1).min(current.1),
            );
            previous = current;
        }
    }
    Ok(Report {
        stats: acc.finish(duration),
        rows,
        key: r.key,
        options: r.options,
        start_unix: r.epoch + r.start,
        satellites: r.orbits.len(),
        swath: r.swath,
        limit,
        step,
    })
}

struct Pending {
    receiver: mpsc::Receiver<Result<Report, String>>,
    cancel: Arc<AtomicBool>,
    progress: Arc<AtomicUsize>,
}
#[derive(Default)]
pub(crate) struct Panel {
    pub open: bool,
    planet: usize,
    options: Options,
    pending: Option<Pending>,
    report: Option<Report>,
    error: Option<String>,
}
impl Drop for Panel {
    fn drop(&mut self) {
        if let Some(p) = &self.pending {
            p.cancel.store(true, Ordering::Relaxed);
        }
    }
}

fn duration_text(seconds: f64) -> String {
    if seconds >= 86400.0 {
        format!("{:.1} d", seconds / 86400.0)
    } else if seconds >= 3600.0 {
        format!("{:.1} h", seconds / 3600.0)
    } else if seconds >= 60.0 {
        format!("{:.1} min", seconds / 60.0)
    } else {
        format!("{seconds:.1} s")
    }
}
fn results_ui(ui: &mut egui::Ui, report: &Report) {
    let start = DateTime::from_timestamp(report.start_unix as i64, 0).unwrap_or_default();
    ui.label(format!(
        "{} satellites · {:.1} km swath · {:.1} days from {} UTC",
        report.satellites,
        report.swath,
        report.options.days,
        start.format("%Y-%m-%d %H:%M")
    ));
    ui.label(format!(
        "Nominal latitude reach: ±{:.2}° · {} longitudes per row · {:.1} s track segments",
        report.limit, report.options.longitudes, report.step
    ));
    ui.small(
        "Ranges are across sampled longitudes at each latitude, not guaranteed geographic extrema.",
    );
    egui::Grid::new("revisit-results").striped(true).spacing([18.0,8.0]).show(ui,|ui| {
        for label in ["Latitude / point","Obs/day (min–max)","Never seen","Revisit gaps (min–max)","Longest silence"] {ui.strong(label);}
        ui.end_row();
        for row in &report.rows {
            let stats=&report.stats[row.indices.clone()];
            let lo=stats.iter().map(|s|s.count).min().unwrap_or(0) as f64/report.options.days;
            let hi=stats.iter().map(|s|s.count).max().unwrap_or(0) as f64/report.options.days;
            let unseen=stats.iter().filter(|s|s.count==0).count();
            let min_gap=stats.iter().filter_map(|s|s.min_gap).reduce(f64::min);
            let max_gap=stats.iter().filter_map(|s|s.max_gap).reduce(f64::max);
            let silence=stats.iter().map(|s|s.longest_silence).fold(0.0_f64,f64::max);
            let insufficient=stats.iter().filter(|s|s.count<2).count();
            ui.label(&row.label);ui.label(format!("{lo:.2}–{hi:.2}"));
            ui.label(format!("{unseen}/{}",stats.len()));
            ui.label(match (min_gap,max_gap) {(Some(a),Some(b))=>format!("{}–{}{}",duration_text(a),duration_text(b),if insufficient>0 {" *"}else{""}),_=>"Not established".into()})
                .on_hover_text(format!("{insufficient}/{} points have fewer than two observations. Gaps use successive observation times at the same point, combining all included satellites.",stats.len()));
            ui.label(duration_text(silence));ui.end_row();
        }
    });
    ui.small("* Some locations have fewer than two observations; their revisit is not established. ‘Longest silence’ includes the start/end of this run and locations never seen. Gaps beyond this time window are unknown.");
}

impl Panel {
    pub fn open_for(&mut self, planet: usize, cycle_s: Option<f64>) {
        self.open = true;
        self.planet = planet;
        if let Some(p) = self.pending.take() {
            p.cancel.store(true, Ordering::Relaxed);
        }
        if let Some(seconds) = cycle_s {
            self.options.days = seconds / 86400.0;
        }
    }
    pub fn show(&mut self, ctx: &egui::Context, tab: &TabConfig, epoch: DateTime<Utc>) {
        if let Some(pending) = &self.pending {
            match pending.receiver.try_recv() {
                Ok(result) => {
                    self.pending = None;
                    match result {
                        Ok(report) => {
                            self.report = Some(report);
                            self.error = None;
                        }
                        Err(e) => self.error = Some(e),
                    }
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.pending = None;
                    self.error = Some("Revisit worker stopped unexpectedly.".into());
                }
                Err(mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(100))
                }
            }
        }
        let mut open = self.open;
        egui::Window::new("Observation revisit").open(&mut open).default_width(860.0).vscroll(true).show(ctx,|ui| {
            let planets:Vec<_>=tab.planets.iter().enumerate().filter(|(_,p)|p.celestial_body==CelestialBody::Earth).collect();
            if planets.is_empty() {ui.label("Select a tab with an Earth globe.");return;}
            if !planets.iter().any(|(i,_)|*i==self.planet) {self.planet=planets[0].0;}
            let busy=self.pending.is_some();
            ui.add_enabled_ui(!busy,|ui| {
                egui::ComboBox::from_id_salt("revisit-planet").selected_text(&tab.planets[self.planet].name).show_ui(ui,|ui| {
                    for (i,p) in &planets {ui.selectable_value(&mut self.planet,*i,&p.name);}
                });
                ui.horizontal(|ui| {
                    ui.label("Forecast");ui.add(egui::DragValue::new(&mut self.options.days).range(0.01..=30.0).speed(0.1).max_decimals(6).suffix(" days"));
                    ui.label("Longitudes per latitude");ui.add(egui::DragValue::new(&mut self.options.longitudes).range(12..=360).speed(12));
                    ui.checkbox(&mut self.options.fine,"Half time step").on_hover_text("Repeat with shorter segments to check numerical convergence, particularly for very narrow sweeps.");
                });
                ui.horizontal(|ui| {
                    ui.label("Also sample a point: latitude");ui.add(egui::DragValue::new(&mut self.options.latitude).range(-90.0..=90.0).speed(0.1).suffix("°"));
                    ui.label("longitude");ui.add(egui::DragValue::new(&mut self.options.longitude).range(-180.0..=180.0).speed(0.1).suffix("°"));
                });
            });
            let planet=&tab.planets[self.planet];
            let swath=tab.settings.pushbroom.swath_km;
            let key=fingerprint(planet,swath,epoch);
            ui.label(format!("Uses the current {:.1} km pushbroom width and visible constellations in this Earth view.",swath));
            ui.small("Continuous nadir imaging on a spherical Earth. Daylight, clouds, jobs, resource limits, maneuvers and debris are excluded. A point observation does not imply full AOI coverage.");
            if ui.add_enabled(!busy,egui::Button::new("Calculate revisit")).clicked() {
                #[cfg(not(target_arch="wasm32"))]
                match orbits(planet) {
                    Err(e)=>self.error=Some(e),
                    Ok(orbits)=>{
                        let r=Request{orbits,epoch:epoch.timestamp_millis() as f64/1000.0,start:tab.settings.time,swath,key,options:self.options};
                        let (sender,receiver)=mpsc::channel();
                        let cancel=Arc::new(AtomicBool::new(false));let progress=Arc::new(AtomicUsize::new(0));
                        let c=Arc::clone(&cancel);let p=Arc::clone(&progress);let context=ctx.clone();
                        std::thread::spawn(move ||{let _=sender.send(calculate(r,&c,&p));context.request_repaint();});
                        self.pending=Some(Pending{receiver,cancel,progress});self.error=None;
                    }
                }
                #[cfg(target_arch="wasm32")]
                {self.error=Some("Revisit forecasting currently requires the native app.".into());}
            }
            if let Some(p)=&self.pending {
                ui.horizontal(|ui| {
                    ui.add(egui::ProgressBar::new(p.progress.load(Ordering::Relaxed) as f32/1000.0).show_percentage());
                    if ui.button("Cancel").clicked() {p.cancel.store(true,Ordering::Relaxed);}
                });
                ui.small("Forecast runs independently of playback. Existing results below are from the previous run.");
            }
            if let Some(error)=&self.error {ui.colored_label(egui::Color32::LIGHT_RED,error);}
            if let Some(report)=&self.report {
                ui.separator();
                if report.key!=key || report.options!=self.options {ui.colored_label(egui::Color32::YELLOW,"Settings changed — recalculate to update these results.");}
                results_ui(ui,report);
                ui.small("Results are a snapshot at the displayed start time. Recalculate to forecast from the current simulation time.");
            }
        });
        self.open = open;
    }
}
