//! Byte-volume presentation using the same orbit selection as coverage.
use crate::{celestial::CelestialBody, config::PlanetConfig, revisit};
use chrono::{DateTime, Utc};
use eframe::egui;
use leo_sim::{
    acquisition::{ground_speed_km_s, CaptureProfile},
    observation::EARTH_RATE,
};

struct Group {
    name: String,
    count: usize,
    speed_sum: f64,
    min_speed: f64,
    max_speed: f64,
}

fn snapshot(planet: &PlanetConfig, time: f64, epoch: DateTime<Utc>) -> Result<Vec<Group>, String> {
    if planet.celestial_body != CelestialBody::Earth {
        return Err("Choose an Earth view to estimate acquisition data.".into());
    }
    let epoch = epoch.timestamp_millis() as f64 / 1000.0;
    revisit::orbit_groups(planet)?
        .into_iter()
        .map(|(name, orbits)| {
            let mut group = Group {
                name,
                count: orbits.len(),
                speed_sum: 0.0,
                min_speed: f64::INFINITY,
                max_speed: 0.0,
            };
            for orbit in orbits {
                let speed = ground_speed_km_s(
                    orbit.position(time - 0.25, epoch)?,
                    orbit.position(time + 0.25, epoch)?,
                    0.5,
                    CelestialBody::Earth.radius_km(),
                    EARTH_RATE,
                )
                .ok_or("Could not determine ground-track speed for a visible satellite.")?;
                group.speed_sum += speed;
                group.min_speed = group.min_speed.min(speed);
                group.max_speed = group.max_speed.max(speed);
            }
            Ok(group)
        })
        .collect()
}

use crate::jobs::bytes;

#[derive(Default)]
pub(super) struct Panel {
    key: Option<u64>,
    time: f64,
    updated: Option<web_time::Instant>,
    report: Option<Result<Vec<Group>, String>>,
}

impl Panel {
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        profile: &mut CaptureProfile,
        planet: &PlanetConfig,
        swath_km: f64,
        time: f64,
        epoch: DateTime<Utc>,
    ) {
        ui.small("Continuous full-swath acquisition");
        egui::Grid::new("capture-profile")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Channels");
                ui.add(egui::DragValue::new(&mut profile.channels).range(1..=256));
                ui.end_row();
                ui.label("Resolution per channel");
                ui.add(
                    egui::DragValue::new(&mut profile.resolution_m)
                        .range(0.05..=1000.0)
                        .speed(0.05)
                        .max_decimals(3)
                        .suffix(" m/px"),
                );
                ui.end_row();
                ui.label("Bits per sample");
                ui.add(egui::DragValue::new(&mut profile.bits_per_sample).range(1..=32))
                    .on_hover_text(
                        "Packed encoding. Use 16 for samples stored in 16-bit integers.",
                    );
                ui.end_row();
                ui.label("Compression ratio");
                ui.add(
                    egui::DragValue::new(&mut profile.compression_ratio)
                        .range(1.0..=100.0)
                        .speed(0.1)
                        .max_decimals(2)
                        .suffix(":1"),
                )
                .on_hover_text("Relative to the selected sample encoding. 1:1 means uncompressed.");
                ui.end_row();
            });

        let key = revisit::fingerprint(planet, swath_km, epoch);
        let now = web_time::Instant::now();
        if self.key != Some(key)
            || (self.time != time
                && self
                    .updated
                    .is_none_or(|t| now.duration_since(t).as_secs_f64() >= 0.25))
        {
            self.report = Some(snapshot(planet, time, epoch));
            self.key = Some(key);
            self.time = time;
            self.updated = Some(now);
        }
        if self.time != time {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(250));
        }
        let Some(report) = &self.report else {
            return;
        };
        let groups = match report {
            Ok(groups) => groups,
            Err(error) => {
                ui.label(error);
                return;
            }
        };
        let count: usize = groups.iter().map(|g| g.count).sum();
        let speed: f64 = groups.iter().map(|g| g.speed_sum).sum();
        let Some(raw_total) = profile.raw_bytes_per_second(swath_km, speed) else {
            ui.label("Enter a valid capture profile and swath width.");
            return;
        };
        let Some(compressed_total) = profile.compressed_bytes_per_second(raw_total) else {
            ui.label("Compression must be at least 1:1.");
            return;
        };
        ui.separator();
        ui.label(format!(
            "{count} visible satellites · {swath_km:.1} km swath"
        ));
        let compressed = profile.compression_ratio > 1.0;
        egui::Grid::new("acquisition-rates")
            .striped(true)
            .show(ui, |ui| {
                ui.label("");
                ui.strong("Per satellite¹");
                ui.strong("Constellation total");
                ui.end_row();
                rate_row(ui, "Raw / second", raw_total, count, 1.0);
                if compressed {
                    rate_row(ui, "Compressed / second", compressed_total, count, 1.0);
                }
                rate_row(
                    ui,
                    if compressed {
                        "Compressed / hour"
                    } else {
                        "Raw / hour"
                    },
                    compressed_total,
                    count,
                    3600.0,
                );
                rate_row(
                    ui,
                    if compressed {
                        "Compressed / day"
                    } else {
                        "Raw / day"
                    },
                    compressed_total,
                    count,
                    86400.0,
                );
            });
        ui.small("¹ Mean across the selected Earth's visible satellites.");
        let min = groups
            .iter()
            .map(|g| g.min_speed)
            .fold(f64::INFINITY, f64::min);
        let max = groups.iter().map(|g| g.max_speed).fold(0.0, f64::max);
        ui.small(format!(
            "Raw per-satellite range: {}–{}/s",
            bytes(profile.raw_bytes_per_second(swath_km, min).unwrap()),
            bytes(profile.raw_bytes_per_second(swath_km, max).unwrap())
        ));

        if groups.len() > 1 {
            ui.collapsing("By constellation", |ui| {
                for (index, g) in groups.iter().enumerate() {
                    ui.push_id(index, |ui| {
                        ui.strong(format!("{} · {} satellites", g.name, g.count));
                        let raw = profile.raw_bytes_per_second(swath_km, g.speed_sum).unwrap();
                        ui.label(format!(
                            "Raw: {}/s per satellite · {}/s total",
                            bytes(raw / g.count as f64),
                            bytes(raw)
                        ));
                        if compressed {
                            ui.label(format!(
                                "Compressed total: {}/s",
                                bytes(raw / profile.compression_ratio)
                            ));
                        }
                    });
                }
            });
        }
        ui.small(
            "Hour/day volumes extrapolate the current rate; they are not accumulated observations.",
        );
        ui.collapsing("Data equations and assumptions", |ui| {
            ui.monospace("bytes/s = width × ground speed / GSD²\n          × channels × bits/sample / 8");
            ui.label("Width and speed use metres and metres/second. Ground-track speed comes from the active orbits, including Earth's rotation; it is independent of playback speed.");
            ui.label("All channels use the selected native resolution. Raw bytes use packed samples; compression divides that volume by the chosen ratio.");
            ui.label("Every visible satellite in this Earth view is assumed to carry this sensor and image continuously at nadir. Overlapping acquisitions count separately. AOI cropping, daylight, clouds, duty cycle, tasking and protocol overhead are excluded.");
            ui.label("Hiding the sweep overlay does not stop acquisition in this estimate. Hidden constellations and debris are excluded. Numerical/42 propagation is unsupported.");
            ui.small("Decimal byte units: GB = 1,000,000,000 bytes. One day = 24 hours.");
        });
    }
}

fn rate_row(ui: &mut egui::Ui, label: &str, total: f64, count: usize, duration: f64) {
    ui.label(label);
    ui.label(bytes(total * duration / count as f64));
    ui.label(bytes(total * duration));
    ui.end_row();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ConstellationConfig, Propagator, TabConfig};

    #[test]
    fn rates_sum_visible_shells_and_reject_unsupported_propagation() {
        let mut tab = TabConfig::new("test".into());
        let p = &mut tab.planets[0];
        let mut c = ConstellationConfig::new(0);
        c.num_planes = 1;
        c.sats_per_plane = 1;
        c.inclination = 0.0;
        c.altitude_km = 561.3856;
        p.constellations = vec![c.clone()];
        let epoch = DateTime::from_timestamp(946728000, 0).unwrap();
        let single = snapshot(p, 0.0, epoch).unwrap()[0].speed_sum;
        c.sats_per_plane = 3;
        p.constellations.push(c.clone());
        c.hidden = true;
        p.constellations.push(c);
        let groups = snapshot(p, 0.0, epoch).unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups.iter().map(|g| g.count).sum::<usize>(), 4);
        assert!((groups.iter().map(|g| g.speed_sum).sum::<f64>() - single * 4.0).abs() < 1e-8);
        p.constellations[1].propagator = Propagator::Numerical;
        assert!(snapshot(p, 0.0, epoch).is_err());
    }

    #[test]
    fn formatting_uses_decimal_bytes() {
        assert_eq!(bytes(9_333_333_333.0), "9.33 GB");
        assert_eq!(bytes(33.6e12), "33.60 TB");
    }
}
