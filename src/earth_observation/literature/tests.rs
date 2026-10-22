use super::*;
use crate::celestial::CelestialBody;
use leo_sim::geometry::{norm, sub};
use std::f64::consts::TAU;

#[test]
fn imported_geometry_matches_the_published_walker_convention() {
    let earth = CelestialBody::Earth;
    for preset in PRESETS {
        for shell in preset.shells {
            let mut c = ConstellationConfig::new(0);
            shell.apply(&mut c, true);
            let orbit = c.constellation(
                earth.radius_km(),
                earth.mu(),
                earth.j2(),
                earth.equatorial_radius_km(),
            );
            let actual: Vec<_> = (0..c.num_planes)
                .flat_map(|p| {
                    let orbit = &orbit;
                    (0..c.sats_per_plane).map(move |s| orbit.orbital_elements(p, s).sample(0.0).xyz)
                })
                .collect();
            // Independent standard Rz(Omega) Rx(i) rotation, transformed into
            // the renderer's Y-north, negative-Z-east axes. Test an unordered
            // set because both plane and in-plane satellite indices reverse.
            let mut unmatched = actual;
            let i = shell.inclination.to_radians();
            let r = earth.radius_km() + shell.altitude_km;
            for plane in 0..shell.planes {
                let omega = TAU * plane as f64 / shell.planes as f64;
                for sat in 0..shell.satellites_per_plane {
                    let u = TAU * sat as f64 / shell.satellites_per_plane as f64
                        + TAU * shell.phasing as f64 * plane as f64
                            / (shell.planes * shell.satellites_per_plane) as f64;
                    let expected = [
                        r * (omega.cos() * u.cos() - omega.sin() * u.sin() * i.cos()),
                        r * u.sin() * i.sin(),
                        -r * (omega.sin() * u.cos() + omega.cos() * u.sin() * i.cos()),
                    ];
                    let index = unmatched
                        .iter()
                        .position(|p| norm(sub(*p, expected)) < 1e-7)
                        .unwrap_or_else(|| {
                            panic!(
                                "Wrong imported geometry: {}, plane {plane}, satellite {sat}",
                                preset.name
                            )
                        });
                    unmatched.swap_remove(index);
                }
            }
            assert!(unmatched.is_empty());
        }
    }
}

#[test]
fn hybrid_adds_both_shells_and_preserves_existing_configuration() {
    let mut p = PlanetConfig::new("Earth".into());
    p.add_constellation();
    p.constellations[0].altitude_km = 432.1;
    p.constellations[0].link_budget.bandwidth_ghz = 7.0;
    for (enabled, _, _) in p.tle_selections.values_mut() {
        *enabled = true;
    }
    p.ground_track_history
        .insert((0, 0, 0), vec![(1.0, 2.0, 3.0)]);
    let mut imaging = pushbroom::Settings::default();
    let mut library = Literature {
        selected: 1,
        j2: false,
        ..Default::default()
    };
    library.add(&mut p, &mut imaging);
    assert_eq!(p.constellations.len(), 3);
    assert!(p.constellations[0].hidden);
    assert_eq!(p.constellations[0].altitude_km, 432.1);
    assert_eq!(p.constellations[0].link_budget.bandwidth_ghz, 7.0);
    assert!(p.tle_selections.values().all(|(enabled, _, _)| !enabled));
    assert!(p.ground_track_history.is_empty());
    assert_eq!(imaging.swath_km, 20.0);
    let shells = &p.constellations[1..];
    assert!(shells
        .iter()
        .all(|s| !s.hidden && s.propagator == Propagator::Keplerian));
    assert_eq!(shells.iter().map(|c| c.total_sats()).sum::<usize>(), 49);
    assert_ne!(shells[0].color_offset, shells[1].color_offset);
    // The source's broad-footprint coverage does not become a false guarantee
    // under our 20 km scanline. Both actual shells enter the shared model.
    let orbits = crate::revisit::orbits(&p)
        .unwrap()
        .into_iter()
        .map(|o| match o {
            crate::revisit::Orbit::Analytical(e) => e,
            _ => panic!("Unexpected TLE"),
        })
        .collect();
    let model = leo_sim::observation::Model::new(
        orbits,
        imaging.swath_km,
        CelestialBody::Earth.radius_km(),
    )
    .unwrap();
    assert!(model.latitude_limit.to_degrees() < 81.0);
    assert!(model.coverage_guarantee(90_f64.to_radians()).is_none());
}

#[test]
fn adding_can_keep_other_shells_and_can_keep_or_apply_published_swath() {
    let mut p = PlanetConfig::new("Earth".into());
    p.add_constellation();
    let mut library = Literature {
        hide_others: false,
        use_swath: false,
        ..Default::default()
    };
    let mut imaging = pushbroom::Settings::default();
    library.add(&mut p, &mut imaging);
    assert_eq!(imaging.swath_km, 20.0);
    assert!(p.constellations.iter().all(|c| !c.hidden));
    let c = &p.constellations[1];
    assert_eq!(c.total_sats(), 5);
    assert_eq!(c.altitude_km, 597.766);
    assert_eq!(c.phasing, 2.0); // published 5/5/3, reversed plane numbering
    assert!(c.propagator == Propagator::J2);
    library.use_swath = true;
    library.add(&mut p, &mut imaging);
    assert_eq!(imaging.swath_km, 60.0);
}
