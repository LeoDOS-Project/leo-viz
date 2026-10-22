use super::*;
use crate::config::ConstellationConfig;

#[test]
fn opening_a_repeat_cycle_preserves_its_fractional_day_duration_in_the_ui() {
    let mut panel = Panel::default();
    let cycle = 6000.0; // Equatorial synodic repeats can be shorter than 0.1 day.
    panel.open_for(0, Some(cycle));
    let ctx = egui::Context::default();
    let tab = TabConfig::new("Earth".into());
    let epoch = DateTime::from_timestamp(946728000, 0).unwrap();
    for _ in 0..3 {
        let _ = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1400.0, 1000.0),
                )),
                ..Default::default()
            },
            |ui| panel.show(ui.ctx(), &tab, epoch),
        );
    }
    assert!((panel.options.days * 86400.0 - cycle).abs() < 1e-8);
}

fn request(inc: f64, swath: f64, lat: f64, fine: bool) -> Request {
    let shell = leo_sim::orbit::WalkerShell {
        planes: 1,
        satellites_per_plane: 1,
        inclination_deg: inc,
        phasing: 0,
        ..Default::default()
    };
    Request {
        orbits: vec![Orbit::Analytical(shell.elements(0, 0))],
        epoch: 946728000.0,
        start: 0.0,
        swath,
        key: 1,
        options: Options {
            days: 1.0,
            longitudes: 24,
            latitude: lat,
            longitude: 0.0,
            fine,
        },
    }
}
fn run(r: Request) -> Report {
    calculate(r, &AtomicBool::new(false), &AtomicUsize::new(0)).unwrap()
}

#[test]
fn equatorial_revisit_matches_synodic_period_and_narrow_sweeps_are_detected() {
    let r = request(0.0, 1.0, 0.0, false);
    let Orbit::Analytical(e) = &r.orbits[0] else {
        unreachable!()
    };
    let rate = (e.mu / e.semi_major_km.powi(3)).sqrt() - 360.98564736629_f64.to_radians() / 86400.0;
    let expected_period = std::f64::consts::TAU / rate;
    let first = gmst(r.epoch) / rate;
    let expected_count = ((86400.0 - first) / expected_period).floor() as usize + 1;
    let report = run(r);
    let point = &report.stats[0];
    assert_eq!(point.count, expected_count);
    assert!((point.min_gap.unwrap() - expected_period).abs() < 0.01);
    assert!((point.max_gap.unwrap() - expected_period).abs() < 0.01);
    assert_eq!(run(request(0.0, 1.0, 0.1, false)).stats[0].count, 0);
    assert_eq!(
        run(request(0.0, 40.0, 0.1, false)).stats[0].count,
        expected_count
    );
}

#[test]
fn retrograde_reach_and_j2_forecast_converge_with_shorter_segments() {
    let mut coarse = request(98.0, 20.0, 57.7, false);
    let shell = leo_sim::orbit::WalkerShell {
        planes: 3,
        satellites_per_plane: 5,
        inclination_deg: 98.0,
        propagation: leo_sim::orbit::Propagation::J2,
        ..Default::default()
    };
    let shell = &shell;
    coarse.orbits = (0..3)
        .flat_map(|p| (0..5).map(move |s| Orbit::Analytical(shell.elements(p, s))))
        .collect();
    let fine = Request {
        orbits: (0..3)
            .flat_map(|p| (0..5).map(move |s| Orbit::Analytical(shell.elements(p, s))))
            .collect(),
        options: Options {
            fine: true,
            ..coarse.options
        },
        ..request(98.0, 20.0, 57.7, true)
    };
    let a = run(coarse);
    let b = run(fine);
    assert!((a.limit - (82.0 + (10.0 / 6371.0_f64).to_degrees())).abs() < 1e-9);
    let changed = a
        .stats
        .iter()
        .zip(&b.stats)
        .filter(|(a, b)| a.count != b.count)
        .count();
    assert!(
        changed <= 1,
        "{changed} points changed observation counts at half step"
    );
    for (a, b) in a
        .stats
        .iter()
        .zip(&b.stats)
        .filter(|(a, b)| a.count == b.count)
    {
        if let (Some(a), Some(b)) = (a.max_gap, b.max_gap) {
            assert!((a - b).abs() < 1.0);
        }
    }
}

#[test]
fn visible_sources_preserve_existing_orbit_parameters_and_refuse_unsupported_models() {
    let mut p = PlanetConfig::new("Earth".into());
    let mut c = ConstellationConfig::new(0);
    c.num_planes = 2;
    c.sats_per_plane = 3;
    c.raan_spacing = Some(13.0);
    c.sat_spacing_km = Some(300.0);
    c.propagator = Propagator::J2;
    c.eccentricity = 0.1;
    c.arg_periapsis = 37.0;
    p.constellations = vec![c];
    let selected = orbits(&p).unwrap();
    assert_eq!(selected.len(), 6);
    let body = CelestialBody::Earth;
    let wc = p.constellations[0].constellation(
        body.radius_km(),
        body.mu(),
        body.j2(),
        body.equatorial_radius_km(),
    );
    let expected = wc.single_satellite_lat_lon(1, 2, 1000.0).2;
    let actual = selected[5].position(1000.0, 946728000.0).unwrap();
    assert!(norm(leo_sim::geometry::sub(actual, expected)) < 1e-9);
    let epoch = DateTime::from_timestamp(946728000, 0).unwrap();
    let key = fingerprint(&p, 20.0, epoch);
    assert_ne!(key, fingerprint(&p, 30.0, epoch));
    p.constellations[0].hidden = true;
    assert!(orbits(&p).is_err());
    p.constellations[0].hidden = false;
    p.constellations[0].propagator = Propagator::Numerical;
    assert!(orbits(&p).err().unwrap().contains("numerical"));
}

#[test]
fn tle_shell_selection_and_sgp4_use_the_same_positions_as_the_viewer() {
    let sats = crate::tle::parse_tle_data(include_str!("../../assets/tle/starlink.tle")).unwrap();
    let mut p = PlanetConfig::new("Earth".into());
    p.constellations.clear();
    p.tle_selections.insert(
        TlePreset::Starlink,
        (
            true,
            TleLoadState::Loaded {
                satellites: sats[..2].to_vec(),
            },
            Some(vec![crate::tle::TleShell {
                label: "One".into(),
                satellite_indices: vec![1],
                color_offset: 0,
                selected: true,
            }]),
        ),
    );
    let selected = orbits(&p).unwrap();
    assert_eq!(selected.len(), 1);
    let t = &sats[1];
    let epoch = t.epoch_minutes * 60.0;
    let actual = selected[0].position(60.0, epoch).unwrap();
    let expected = t.constants.propagate(sgp4::MinutesSinceEpoch(1.0)).unwrap();
    assert!(
        norm(leo_sim::geometry::sub(
            actual,
            [
                expected.position[0],
                expected.position[2],
                -expected.position[1]
            ]
        )) < 0.001
    );
}

#[test]
fn cancellation_stops_calculation_without_partial_results() {
    assert!(calculate(
        request(53.0, 20.0, 0.0, false),
        &AtomicBool::new(true),
        &AtomicUsize::new(0)
    )
    .err()
    .unwrap()
    .contains("cancelled"));
}

fn text_position(shapes: &[egui::epaint::ClippedShape], label: &str) -> Option<egui::Pos2> {
    shapes.iter().find_map(|s| match &s.shape {
        egui::Shape::Text(t) if t.galley.job.text == label => Some(t.pos + t.galley.size() * 0.5),
        _ => None,
    })
}
#[test]
fn calculate_button_runs_in_background_and_changed_settings_are_marked() {
    let ctx = egui::Context::default();
    let epoch = DateTime::from_timestamp(946728000, 0).unwrap();
    let mut tab = TabConfig::new("Test".into());
    let mut config = ConstellationConfig::new(0);
    config.num_planes = 1;
    config.sats_per_plane = 1;
    tab.planets[0].constellations = vec![config];
    let mut panel = Panel::default();
    panel.open = true;
    panel.options = Options {
        days: 1.0,
        longitudes: 12,
        ..Default::default()
    };
    let frame = |p: &mut Panel, tab: &TabConfig, events| {
        ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1400.0, 1000.0),
                )),
                events,
                ..Default::default()
            },
            |ui| p.show(ui.ctx(), tab, epoch),
        )
    };
    let _ = frame(&mut panel, &tab, vec![]);
    let _ = frame(&mut panel, &tab, vec![]);
    let out = frame(&mut panel, &tab, vec![]);
    let pos = text_position(&out.shapes, "Calculate revisit").unwrap();
    for pressed in [true, false] {
        let _ = frame(
            &mut panel,
            &tab,
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
    }
    assert!(panel.pending.is_some());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while panel.pending.is_some() && std::time::Instant::now() < deadline {
        let _ = frame(&mut panel, &tab, vec![]);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(panel.error.is_none(), "{:?}", panel.error);
    assert!(panel.report.is_some());
    tab.settings.pushbroom.swath_km = 35.0;
    let _ = frame(&mut panel, &tab, vec![]);
    let out = frame(&mut panel, &tab, vec![]);
    assert!(text_position(
        &out.shapes,
        "Settings changed — recalculate to update these results."
    )
    .is_some());
    assert!(text_position(&out.shapes, "Longest silence").is_some());
}

#[test]
#[ignore = "manual 1,000-satellite forecast timing"]
fn thousand_satellite_forecast() {
    let mut r = request(53.0, 20.0, 57.7, false);
    r.options.days = 7.0;
    r.options.longitudes = 72;
    let shell = leo_sim::orbit::WalkerShell {
        planes: 25,
        satellites_per_plane: 40,
        inclination_deg: 53.0,
        ..Default::default()
    };
    let shell = &shell;
    r.orbits = (0..25)
        .flat_map(|p| (0..40).map(move |s| Orbit::Analytical(shell.elements(p, s))))
        .collect();
    let start = std::time::Instant::now();
    let report = run(r);
    assert_eq!(report.satellites, 1000);
    assert!(report.stats.iter().any(|s| s.count > 0));
    eprintln!(
        "1,000 satellites / 7 days / {} points: {:?}",
        report.stats.len(),
        start.elapsed()
    );
}
