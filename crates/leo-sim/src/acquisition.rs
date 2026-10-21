//! Full-swath nadir acquisition volumes, independent of rendering and scheduling.
use crate::geometry::{norm, scale, sub, Vector};

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CaptureProfile {
    pub channels: u16,
    /// Square ground sampling distance, common to all channels.
    pub resolution_m: f64,
    /// Packed sample encoding; choose 16 to model a 16-bit container.
    pub bits_per_sample: u8,
    /// Encoded bytes / compressed bytes. One means no compression.
    pub compression_ratio: f64,
}

impl Default for CaptureProfile {
    fn default() -> Self {
        Self {
            channels: 4,
            resolution_m: 0.3,
            bits_per_sample: 12,
            compression_ratio: 1.0,
        }
    }
}

impl CaptureProfile {
    /// Stored bytes for useful imagery over an area, with the same packed
    /// encoding and compression as the continuous sensor-volume estimate.
    pub fn stored_bytes_for_area(&self, area_km2: f64) -> Option<f64> {
        self.compressed_bytes_per_second(self.raw_bytes_for_area(area_km2)?)
    }

    pub fn raw_bytes_for_area(&self, area_km2: f64) -> Option<f64> {
        if self.channels == 0
            || !(1..=64).contains(&self.bits_per_sample)
            || !self.resolution_m.is_finite()
            || self.resolution_m <= 0.0
            || !area_km2.is_finite()
            || area_km2 < 0.0
        {
            return None;
        }
        let bytes = area_km2 * 1e6 / self.resolution_m.powi(2)
            * f64::from(self.channels)
            * f64::from(self.bits_per_sample)
            / 8.0;
        bytes.is_finite().then_some(bytes)
    }

    pub fn raw_bytes_per_second(&self, swath_km: f64, ground_speed_km_s: f64) -> Option<f64> {
        if !swath_km.is_finite()
            || swath_km <= 0.0
            || !ground_speed_km_s.is_finite()
            || ground_speed_km_s < 0.0
        {
            return None;
        }
        self.raw_bytes_for_area(swath_km * ground_speed_km_s)
    }

    pub fn compressed_bytes_per_second(&self, raw: f64) -> Option<f64> {
        (raw.is_finite()
            && raw >= 0.0
            && self.compression_ratio.is_finite()
            && self.compression_ratio >= 1.0)
            .then(|| raw / self.compression_ratio)
    }
}

/// Ground-subpoint speed from two inertial positions in the shared +y-north,
/// -z-east convention. Account for Earth rotation before measuring the arc.
/// Absolute Greenwich angle cancels: only rotation during the interval matters.
pub fn ground_speed_km_s(
    before: Vector,
    after: Vector,
    interval_s: f64,
    radius_km: f64,
    rotation_rate: f64,
) -> Option<f64> {
    if !interval_s.is_finite()
        || interval_s <= 0.0
        || !radius_km.is_finite()
        || radius_km <= 0.0
        || !rotation_rate.is_finite()
    {
        return None;
    }
    let (r0, r1) = (norm(before), norm(after));
    if !r0.is_finite() || !r1.is_finite() || r0 <= radius_km || r1 <= radius_km {
        return None;
    }
    let a = scale(before, 1.0 / r0);
    let b = scale(after, 1.0 / r1);
    let (s, c) = (rotation_rate * interval_s).sin_cos();
    let b = [c * b[0] - s * b[2], b[1], s * b[0] + c * b[2]];
    // Chord form is stable for the short arcs used for instantaneous rates.
    let arc = 2.0 * (norm(sub(b, a)) / 2.0).clamp(0.0, 1.0).asin();
    Some(radius_km * arc / interval_s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{geometry::*, observation::EARTH_RATE, orbit::WalkerShell};

    #[test]
    fn discussed_payload_and_scaling_use_bytes_not_bits() {
        let p = CaptureProfile::default();
        let raw = p.raw_bytes_per_second(20.0, 7.0).unwrap();
        assert!((raw - 9_333_333_333.333334).abs() < 1e-5);
        assert!((raw * 3600.0 / 1e12 - 33.6).abs() < 1e-12);
        let half_resolution = CaptureProfile {
            resolution_m: 0.6,
            ..p
        };
        assert_eq!(
            half_resolution.raw_bytes_per_second(20.0, 7.0),
            Some(raw / 4.0)
        );
        let two_channels = CaptureProfile { channels: 2, ..p };
        assert_eq!(
            two_channels.raw_bytes_per_second(20.0, 7.0),
            Some(raw / 2.0)
        );
        assert_eq!(p.raw_bytes_per_second(40.0, 7.0), Some(raw * 2.0));
        let compressed = CaptureProfile {
            compression_ratio: 2.0,
            ..p
        };
        assert_eq!(compressed.compressed_bytes_per_second(raw), Some(raw / 2.0));
        assert!(CaptureProfile {
            resolution_m: 0.0,
            ..p
        }
        .raw_bytes_per_second(20.0, 7.0)
        .is_none());
        assert!(CaptureProfile {
            compression_ratio: f64::NAN,
            ..p
        }
        .compressed_bytes_per_second(raw)
        .is_none());
    }

    #[test]
    fn ground_speed_matches_prograde_and_retrograde_equatorial_orbits() {
        for inclination_deg in [0.0, 180.0] {
            let e = WalkerShell {
                inclination_deg,
                altitude_km: 561.3856,
                ..Default::default()
            }
            .elements(0, 0);
            let n = (EARTH_MU / e.semi_major_km.powi(3)).sqrt();
            let sign = if inclination_deg == 0.0 { 1.0 } else { -1.0 };
            let expected = EARTH_RADIUS_KM * (sign * n - EARTH_RATE).abs();
            let actual = ground_speed_km_s(
                e.sample(1234.0).xyz,
                e.sample(1234.5).xyz,
                0.5,
                EARTH_RADIUS_KM,
                EARTH_RATE,
            )
            .unwrap();
            assert!((actual - expected).abs() < 1e-8);
        }
        assert!(ground_speed_km_s([0.0; 3], [0.0; 3], 0.5, EARTH_RADIUS_KM, EARTH_RATE).is_none());
    }
}
