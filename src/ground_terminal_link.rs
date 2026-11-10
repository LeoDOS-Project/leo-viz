//! Nominal, adaptive single-channel RF downlink estimates, not site predictions.

use super::data::{Band, ReceiveBand, Terminal};
use leo_sim::{
    geometry::{distance, EARTH_MU, EARTH_RADIUS_KM, LIGHT_KM_S},
    orbit::{Elements, Propagation},
};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Receiver {
    Vhr3200,
    VhrWithProcessor,
    Custom,
}

impl Receiver {
    pub const ALL: [Self; 3] = [Self::Vhr3200, Self::VhrWithProcessor, Self::Custom];

    pub fn label(self) -> &'static str {
        match self {
            Self::Vhr3200 => "Viasat VHR 3200",
            Self::VhrWithProcessor => "VHR 3200 + data processor",
            Self::Custom => "Custom receiver",
        }
    }

    pub fn channel_mbps(self, custom: f64) -> f64 {
        match self {
            Self::Vhr3200 => 4050.0,
            Self::VhrWithProcessor => 4000.0,
            Self::Custom => custom,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Scenario {
    pub band: Band,
    pub frequency_ghz: f64,
    pub bandwidth_mhz: f64,
    pub altitude_km: f64,
    pub min_elevation_deg: f64,
    pub eirp_dbw: f64,
    pub satellite_mbps: f64,
    pub receiver: Receiver,
    pub custom_receiver_mbps: f64,
    pub spectral_efficiency: f64,
    pub required_ebn0_db: f64,
    pub margin_db: f64,
    pub additional_loss_db: f64,
    pub payload_fraction: f64,
    pub backhaul_mbps: f64,
}

impl Default for Scenario {
    fn default() -> Self {
        Self {
            band: Band::X,
            frequency_ghz: Band::X.frequency_ghz(),
            bandwidth_mhz: Band::X.bandwidth_mhz(),
            altitude_km: 550.0,
            min_elevation_deg: 10.0,
            eirp_dbw: 5.0,
            satellite_mbps: 1000.0,
            receiver: Receiver::Vhr3200,
            custom_receiver_mbps: 1000.0,
            spectral_efficiency: 2.0,
            required_ebn0_db: 5.0,
            margin_db: 3.0,
            additional_loss_db: 3.0,
            payload_fraction: 0.9,
            backhaul_mbps: 1000.0,
        }
    }
}

impl Scenario {
    pub fn validate(&self) -> Result<(), &'static str> {
        let positive = [
            self.frequency_ghz,
            self.bandwidth_mhz,
            self.satellite_mbps,
            self.custom_receiver_mbps,
            self.spectral_efficiency,
            self.backhaul_mbps,
        ];
        if positive.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || positive.iter().any(|v| *v > 1e6)
            || !self.altitude_km.is_finite()
            || !(400.0..=2000.0).contains(&self.altitude_km)
            || !self.min_elevation_deg.is_finite()
            || !(10.0..=85.0).contains(&self.min_elevation_deg)
            || !self.eirp_dbw.is_finite()
            || !(-50.0..=100.0).contains(&self.eirp_dbw)
            || !self.required_ebn0_db.is_finite()
            || !(-2.0..=30.0).contains(&self.required_ebn0_db)
            || !self.margin_db.is_finite()
            || !(0.0..=60.0).contains(&self.margin_db)
            || !self.additional_loss_db.is_finite()
            || !(0.0..=100.0).contains(&self.additional_loss_db)
            || !self.payload_fraction.is_finite()
            || !(0.01..=1.0).contains(&self.payload_fraction)
        {
            return Err("Invalid scenario: check the downlink assumptions.");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limit {
    LinkBudget,
    Shannon,
    Bandwidth,
    Satellite,
    Receiver,
    Processor,
    Station,
    Backhaul,
}

impl Limit {
    pub fn label(self) -> &'static str {
        match self {
            Self::LinkBudget => "Link budget",
            Self::Shannon => "Shannon bound",
            Self::Bandwidth => "RF bandwidth",
            Self::Satellite => "Satellite",
            Self::Receiver => "Receiver",
            Self::Processor => "Data processor",
            Self::Station => "Station system",
            Self::Backhaul => "Backhaul",
        }
    }
}

pub struct Estimate {
    pub peak_mbps: f64,
    pub average_mbps: f64,
    pub data_gb: f64,
    pub peak_limit: Limit,
    pub points: Vec<[f64; 2]>,
}

pub struct Comparison {
    pub scenario: Scenario,
    pub duration_s: f64,
    pub rows: Vec<Result<Estimate, &'static str>>,
}

pub fn free_space_loss_db(distance_km: f64, frequency_ghz: f64) -> f64 {
    20.0 * (4.0 * PI * distance_km * frequency_ghz * 1e9 / LIGHT_KM_S).log10()
}

fn compatible<'a>(terminal: &'a Terminal, s: &Scenario) -> Result<&'a ReceiveBand, &'static str> {
    let rx = terminal.receive(s.band).ok_or("Band not supported")?;
    let half_band_ghz = s.bandwidth_mhz / 2000.0;
    if s.frequency_ghz - half_band_ghz < rx.low_ghz - 1e-9
        || s.frequency_ghz + half_band_ghz > rx.high_ghz + 1e-9
    {
        return Err("RF bandwidth outside receive range");
    }
    if rx
        .reference_elevation_deg
        .is_some_and(|reference| s.min_elevation_deg < reference)
    {
        return Err("Below the published G/T reference elevation");
    }
    Ok(rx)
}

// G/T is held at its published reference (or conservative floor). This does not
// infer an elevation-dependent sky temperature or a site-specific rain model.
fn useful_rate(s: &Scenario, rx: &ReceiveBand, distance_km: f64) -> (f64, Limit) {
    let bandwidth_hz = s.bandwidth_mhz * 1e6;
    let cn0 = s.eirp_dbw - free_space_loss_db(distance_km, s.frequency_ghz) - s.additional_loss_db
        + rx.gt_dbk
        + 228.6
        - s.margin_db;
    let energy_bps = 10.0_f64.powf((cn0 - s.required_ebn0_db) / 10.0);
    let snr = 10.0_f64.powf((cn0 - 10.0 * bandwidth_hz.log10()) / 10.0);
    let limits = [
        (energy_bps, Limit::LinkBudget),
        (bandwidth_hz * snr.ln_1p() / 2.0_f64.ln(), Limit::Shannon),
        (bandwidth_hz * s.spectral_efficiency, Limit::Bandwidth),
        (s.satellite_mbps * 1e6, Limit::Satellite),
        (
            s.receiver.channel_mbps(s.custom_receiver_mbps) * 1e6,
            if s.receiver == Receiver::VhrWithProcessor {
                Limit::Processor
            } else {
                Limit::Receiver
            },
        ),
        (
            rx.system_max_mbps.unwrap_or(f64::INFINITY) * 1e6,
            Limit::Station,
        ),
    ];
    let (raw_bps, limit) = limits
        .into_iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap();
    let payload_bps = raw_bps * s.payload_fraction;
    if s.backhaul_mbps * 1e6 < payload_bps {
        (s.backhaul_mbps * 1e6, Limit::Backhaul)
    } else {
        (payload_bps, limit)
    }
}

fn pass(s: &Scenario, steps: usize) -> (f64, Vec<[f64; 2]>) {
    let r = EARTH_RADIUS_KM + s.altitude_km;
    let n = (EARTH_MU / r.powi(3)).sqrt();
    let elevation = s.min_elevation_deg.to_radians();
    let half_angle = (EARTH_RADIUS_KM / r * elevation.cos()).acos() - elevation;
    let duration = 2.0 * half_angle / n;
    let elements = Elements {
        semi_major_km: r,
        eccentricity: 0.0,
        inclination_rad: 0.0,
        raan_rad: 0.0,
        argument_rad: 0.0,
        mean_anomaly_rad: 0.0,
        mu: EARTH_MU,
        radius_km: EARTH_RADIUS_KM,
        j2: 0.0,
        propagation: Propagation::Keplerian,
    };
    let ground = [EARTH_RADIUS_KM, 0.0, 0.0];
    let points = (0..=steps)
        .map(|i| {
            let t = duration * (i as f64 / steps as f64 - 0.5);
            [t, distance(ground, elements.sample(t).xyz)]
        })
        .collect();
    (duration, points)
}

impl Comparison {
    pub fn calculate(s: Scenario, terminals: &[Terminal]) -> Result<Self, &'static str> {
        s.validate()?;
        let (duration_s, pass) = pass(&s, 512);
        let rows = terminals
            .iter()
            .map(|terminal| {
                let rx = compatible(terminal, &s)?;
                let points: Vec<_> = pass
                    .iter()
                    .map(|[time, distance]| [*time, useful_rate(&s, rx, *distance).0 / 1e6])
                    .collect();
                let (peak_bps, peak_limit) = useful_rate(&s, rx, s.altitude_km);
                let megabits: f64 = points
                    .windows(2)
                    .map(|pair| (pair[1][0] - pair[0][0]) * (pair[0][1] + pair[1][1]) / 2.0)
                    .sum();
                Ok(Estimate {
                    peak_mbps: peak_bps / 1e6,
                    average_mbps: megabits / duration_s,
                    data_gb: megabits / 8000.0,
                    peak_limit,
                    points,
                })
            })
            .collect();
        Ok(Self {
            scenario: s,
            duration_s,
            rows,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ground_terminals::data::TERMINALS;

    #[test]
    fn free_space_loss_has_correct_units_and_range_dependence() {
        assert!((free_space_loss_db(1.0, 1.0) - 92.4478).abs() < 0.001);
        let delta = free_space_loss_db(1100.0, 8.2) - free_space_loss_db(550.0, 8.2);
        assert!((delta - 6.0206).abs() < 1e-4);
    }

    #[test]
    fn pass_is_symmetric_and_starts_at_the_elevation_mask() {
        let s = Scenario::default();
        let (duration, points) = pass(&s, 512);
        assert!((400.0..600.0).contains(&duration));
        assert!((points[256][1] - s.altitude_km).abs() < 1e-8);
        for (a, b) in points.iter().zip(points.iter().rev()) {
            assert!((a[1] - b[1]).abs() < 1e-8);
        }
        let d = points[0][1];
        let r = EARTH_RADIUS_KM + s.altitude_km;
        let sin_el = (r * r - EARTH_RADIUS_KM.powi(2) - d * d) / (2.0 * EARTH_RADIUS_KM * d);
        assert!((sin_el.asin().to_degrees() - s.min_elevation_deg).abs() < 1e-8);
    }

    #[test]
    fn useful_data_is_integrated_in_decimal_gigabytes() {
        let s = Scenario {
            backhaul_mbps: 1.0,
            ..Default::default()
        };
        let result = Comparison::calculate(s, TERMINALS).unwrap();
        for row in result.rows.iter().flatten() {
            assert_eq!(row.peak_limit, Limit::Backhaul);
            assert!((row.average_mbps - 1.0).abs() < 1e-10);
            assert!((row.data_gb - result.duration_s / 8000.0).abs() < 1e-10);
        }
    }

    fn unconstrained() -> Scenario {
        Scenario {
            band: Band::Ka,
            frequency_ghz: 26.25,
            bandwidth_mhz: 1000.0,
            eirp_dbw: 90.0,
            satellite_mbps: 100_000.0,
            receiver: Receiver::Custom,
            custom_receiver_mbps: 100_000.0,
            spectral_efficiency: 20.0,
            backhaul_mbps: 100_000.0,
            ..Default::default()
        }
    }

    #[test]
    fn ceilings_and_payload_overhead_are_applied_once() {
        let rx = TERMINALS[7].receive(Band::Ka).unwrap();
        for (s, expected, limit) in [
            (
                Scenario {
                    satellite_mbps: 100.0,
                    ..unconstrained()
                },
                90e6,
                Limit::Satellite,
            ),
            (
                Scenario {
                    receiver: Receiver::Vhr3200,
                    ..unconstrained()
                },
                3645e6,
                Limit::Receiver,
            ),
            (
                Scenario {
                    receiver: Receiver::VhrWithProcessor,
                    ..unconstrained()
                },
                3600e6,
                Limit::Processor,
            ),
            (
                Scenario {
                    custom_receiver_mbps: 123.0,
                    ..unconstrained()
                },
                110.7e6,
                Limit::Receiver,
            ),
            (
                Scenario {
                    bandwidth_mhz: 10.0,
                    spectral_efficiency: 2.0,
                    ..unconstrained()
                },
                18e6,
                Limit::Bandwidth,
            ),
            (
                Scenario {
                    backhaul_mbps: 50.0,
                    ..unconstrained()
                },
                50e6,
                Limit::Backhaul,
            ),
        ] {
            let (rate, actual_limit) = useful_rate(&s, rx, 550.0);
            assert!((rate - expected).abs() < 1e-5, "{rate} != {expected}");
            assert_eq!(actual_limit, limit);
        }
        let station = TERMINALS[10].receive(Band::X).unwrap();
        let s = Scenario {
            band: Band::X,
            frequency_ghz: 8.2,
            ..unconstrained()
        };
        assert_eq!(useful_rate(&s, station, 550.0), (90e6, Limit::Station));
    }

    #[test]
    fn higher_sensitivity_and_shorter_range_never_reduce_rate() {
        let s = Scenario::default();
        let small = TERMINALS[0].receive(Band::X).unwrap();
        let large = TERMINALS[7].receive(Band::X).unwrap();
        for distance in [550.0, 1000.0, 1800.0] {
            assert!(useful_rate(&s, large, distance).0 >= useful_rate(&s, small, distance).0);
        }
        assert!(useful_rate(&s, small, 550.0).0 > useful_rate(&s, small, 1800.0).0);
    }

    #[test]
    fn unsupported_bands_and_occupied_bandwidth_are_not_predictions() {
        assert!(compatible(
            &TERMINALS[8],
            &Scenario {
                band: Band::Ka,
                frequency_ghz: 26.25,
                ..Default::default()
            }
        )
        .is_err());
        assert!(compatible(
            &TERMINALS[8],
            &Scenario {
                bandwidth_mhz: 500.0,
                ..Default::default()
            }
        )
        .is_err());
        assert!(compatible(
            &TERMINALS[1],
            &Scenario {
                min_elevation_deg: 5.0,
                ..Default::default()
            }
        )
        .is_err());
        assert!(compatible(
            &TERMINALS[8],
            &Scenario {
                bandwidth_mhz: 350.0,
                ..Default::default()
            }
        )
        .is_ok());
    }

    #[test]
    fn invalid_inputs_cannot_produce_nan_results() {
        for value in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            assert!(Comparison::calculate(
                Scenario {
                    bandwidth_mhz: value,
                    ..Default::default()
                },
                TERMINALS
            )
            .is_err());
        }
        for value in [f64::NAN, f64::INFINITY, 90.0, -10.0] {
            assert!(Comparison::calculate(
                Scenario {
                    min_elevation_deg: value,
                    ..Default::default()
                },
                TERMINALS
            )
            .is_err());
        }
    }

    #[test]
    fn published_limits_and_receiver_channels_are_not_invented() {
        assert!(TERMINALS[..10]
            .iter()
            .flat_map(|t| t.bands)
            .all(|rx| rx.system_max_mbps.is_none()));
        assert_eq!(Receiver::Vhr3200.channel_mbps(0.0), 4050.0);
        assert_eq!(
            TERMINALS[10].receive(Band::S).unwrap().system_max_mbps,
            Some(2.0)
        );
        assert_eq!(
            TERMINALS[10].receive(Band::X).unwrap().system_max_mbps,
            Some(100.0)
        );
        assert_eq!(TERMINALS[1].receive(Band::X).unwrap().gt_dbk, 29.5);
        assert_eq!(TERMINALS[6].diameter_m, 11.5);
    }

    #[test]
    fn shannon_bound_still_applies_with_an_optimistic_ebn0_threshold() {
        let s = Scenario {
            eirp_dbw: 5.0,
            required_ebn0_db: -2.0,
            ..unconstrained()
        };
        let rx = TERMINALS[7].receive(Band::Ka).unwrap();
        let (rate, limit) = useful_rate(&s, rx, 550.0);
        assert_eq!(limit, Limit::Shannon);
        let cn0 = 10.0_f64.powf(
            (s.eirp_dbw - free_space_loss_db(550.0, s.frequency_ghz) - s.additional_loss_db
                + rx.gt_dbk
                + 228.6
                - s.margin_db)
                / 10.0,
        );
        let bandwidth = s.bandwidth_mhz * 1e6;
        let expected = bandwidth * (1.0 + cn0 / bandwidth).log2() * s.payload_fraction;
        assert!((rate - expected).abs() / expected < 1e-12);
    }

    #[test]
    fn pass_data_integration_converges_at_the_display_resolution() {
        let s = Scenario::default();
        let rx = TERMINALS[3].receive(Band::X).unwrap();
        let integrate = |steps| {
            let (_, samples) = pass(&s, steps);
            samples
                .windows(2)
                .map(|p| {
                    (p[1][0] - p[0][0])
                        * (useful_rate(&s, rx, p[0][1]).0 + useful_rate(&s, rx, p[1][1]).0)
                        / 2.0
                })
                .sum::<f64>()
        };
        let fine = integrate(4096);
        assert!((integrate(512) - fine).abs() / fine < 0.0001);
    }
}
