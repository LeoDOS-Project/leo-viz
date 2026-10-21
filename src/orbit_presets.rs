//! Research baselines available beside commercial constellations in the basic preset menu.
use crate::{
    celestial::CelestialBody,
    config::{ConstellationConfig, Preset, Propagator},
    walker::WalkerType,
};
use leo_sim::repeat_ground_track::{design, RepeatRatio};

pub(crate) const ADAM_SWATH_KM: f64 = 20.0;

/// Circular two-body 15/1 daily-repeat baseline. Preserve user resource and
/// terminal profiles, but reset every geometric override and propagation state.
/// The caller applies the associated swath to the view and clears track trails.
pub(crate) fn apply_adam(c: &mut ConstellationConfig) {
    let earth = CelestialBody::Earth;
    let orbit = design(
        RepeatRatio::new(15, 1).unwrap(),
        earth.mu(),
        earth.radius_km(),
    )
    .expect("valid Earth daily-repeat orbit");
    c.num_planes = 13;
    c.sats_per_plane = 13;
    c.altitude_km = orbit.altitude_km;
    c.inclination = 70.0;
    c.walker_type = WalkerType::Delta;
    c.phasing = 0.0;
    c.raan_offset = 0.0;
    c.raan_spacing = None;
    c.sat_spacing_km = None;
    c.eccentricity = 0.0;
    c.arg_periapsis = 0.0;
    c.sso = false;
    c.propagator = Propagator::Keplerian;
    c.drag_enabled = false;
    c.isl_neighbors = 4;
    c.hidden = false;
    c.label = None;
    c.preset = Preset::Adam;
    c.physics_state.clear();
    c.numerical = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use leo_sim::{
        geometry::{norm, sub},
        observation::Model,
        repeat_ground_track::SIDEREAL_DAY,
    };
    use std::f64::consts::TAU;

    #[test]
    fn adam_preserves_daily_tracks_with_the_twenty_km_sensor() {
        let mut c = ConstellationConfig::new(7);
        c.label = Some("previous literature preset".into());
        c.sso = true;
        c.drag_enabled = true;
        c.raan_spacing = Some(1.0);
        c.sat_spacing_km = Some(2.0);
        c.eccentricity = 0.1;
        c.propagator = Propagator::J2;
        c.link_budget.tx_power_w = 9.0;
        apply_adam(&mut c);
        assert_eq!(c.preset_name(), "ADAM");
        assert_eq!(c.total_sats(), 169);
        assert_eq!(c.color_offset, 7);
        assert_eq!(c.link_budget.tx_power_w, 9.0);
        assert_eq!(c.isl_neighbors, 4);
        assert!(!c.sso && !c.drag_enabled);
        let earth = CelestialBody::Earth;
        let shell = c.constellation(
            earth.radius_km(),
            earth.mu(),
            earth.j2(),
            earth.equatorial_radius_km(),
        );
        let elements: Vec<_> = (0..13)
            .flat_map(|p| (0..13).map(move |s| (p, s)))
            .map(|(p, s)| shell.orbital_elements(p, s))
            .collect();
        let model = Model::new(elements.clone(), ADAM_SWATH_KM, earth.radius_km()).unwrap();
        assert_eq!(ADAM_SWATH_KM, 20.0);
        assert!(model.repeat.is_some());
        assert!(model.coverage_guarantee(70_f64.to_radians()).is_none());
        // The old wide-swath certificate must not survive the narrower sensor.
        let wide = Model::new(elements.clone(), 210.0, earth.radius_km()).unwrap();
        assert!(wide.coverage_guarantee(70_f64.to_radians()).is_some());
        // Position sets repeat after D/13, with a cyclic plane/satellite permutation.
        // This orbital symmetry alone does not guarantee coverage or revisit.
        let angle = TAU / 13.0;
        for t in [0.0, 1293.417, 38122.3] {
            for e in &elements {
                let x = e.sample(t + SIDEREAL_DAY / 13.0).xyz;
                let rotated = [
                    x[0] * angle.cos() - x[2] * angle.sin(),
                    x[1],
                    x[0] * angle.sin() + x[2] * angle.cos(),
                ];
                let nearest = elements
                    .iter()
                    .map(|other| norm(sub(rotated, other.sample(t).xyz)))
                    .fold(f64::INFINITY, f64::min);
                assert!(nearest < 1e-7, "repeat error {nearest} km");
            }
        }
    }
}
