use super::*;
use crate::{
    geometry::{dot, ground_position, EARTH_RADIUS_KM as R},
    orbit::WalkerShell,
};

fn shell(inc: f64, sats: usize) -> Vec<Elements> {
    let s = WalkerShell {
        planes: 1,
        satellites_per_plane: sats,
        inclination_deg: inc,
        ..Default::default()
    };
    (0..sats).map(|i| s.elements(0, i)).collect()
}
fn query(model: &Model, lat: f64, lon: f64, horizon: f64) -> PointReport {
    model.point(lat, lon, 0.0, 0.0, horizon, |_| true).unwrap()
}

#[test]
fn exact_equatorial_coverage_counts_unique_phases_and_handles_retrograde() {
    for inc in [0.0, 180.0] {
        let mut e = shell(inc, 4);
        let period = TAU / (e[0].inclination_rad.cos() * rate(&e[0]) - EARTH_RATE).abs();
        e.push(e[0]);
        let model = Model::new(e, 20.0, R).unwrap();
        let c = model.coverage().unwrap();
        assert!((model.repeat.unwrap() - period).abs() < 1e-8);
        assert!((c.period - period / 4.0).abs() < 1e-8);
        assert_eq!(query(&model, 0.0, 179.0, 86400.0).exact_gap, Some(c.period));
        assert_eq!(
            query(&model, 2.0, 0.0, 86400.0).exact_gap,
            Some(f64::INFINITY)
        );
    }
    let mut e = shell(0.0, 2);
    e[1].mean_anomaly_rad = 0.1;
    let expected = (TAU - 0.1) / (rate(&e[0]) - EARTH_RATE);
    assert!((Model::new(e, 20.0, R).unwrap().coverage().unwrap().period - expected).abs() < 1e-8);
}

#[test]
fn harmonics_agree_with_shared_orbit_geometry_and_derivatives() {
    for inc in [0.0, 53.0, 90.0, 98.0, 180.0] {
        let mut e = shell(inc, 1)[0];
        e.raan_rad = 0.31;
        e.mean_anomaly_rad = 0.27;
        e.argument_rad = 0.68;
        let x = ground_position(32.0, 117.0, 1.0, 0.0);
        let start = 123456.0;
        let angle = 1.23;
        let series = Series::new(&e, x, start, angle);
        for t in [0.0, 1234.5, 120000.0] {
            let p = e.sample(start + t).xyz;
            let (s, c) = (angle + EARTH_RATE * (start + t)).sin_cos();
            let q = [
                (c * p[0] - s * p[2]) / e.semi_major_km,
                p[1] / e.semi_major_km,
                (s * p[0] + c * p[2]) / e.semi_major_km,
            ];
            let v = series.value(t);
            assert!((dot(x, q) - v[0]).abs() < 1e-12);
            let a = series.value(t - 0.01);
            let b = series.value(t + 0.01);
            assert!(((b[0] - a[0]) / 0.02 - v[1]).abs() < 1e-10);
            assert!(((b[1] - a[1]) / 0.02 - v[2]).abs() < 1e-12);
        }
    }
}

#[test]
fn polar_point_is_exact_and_unreachable_is_not_confused_with_unseen() {
    let e = shell(90.0, 3);
    let expected = TAU / rate(&e[0]) / 3.0;
    let model = Model::new(e, 20.0, R).unwrap();
    for lat in [-90.0, 90.0] {
        assert!((query(&model, lat, 67.0, 86400.0).exact_gap.unwrap() - expected).abs() < 1e-8);
    }
    let model = Model::new(shell(53.0, 1), 1.0, R).unwrap();
    assert_eq!(query(&model, 70.0, 0.0, 1.0).exact_gap, Some(f64::INFINITY));
    let r = query(&model, 40.0, 0.0, 1.0);
    assert_eq!(r.exact_gap, None);
    assert_eq!(r.max_gap, None);
    assert_eq!(r.events, 0);
    assert_eq!(r.longest_silence, 1.0);
    assert!(!r.cyclic);
}

#[test]
fn repeating_cycle_includes_wrap_and_nonrepeating_orbit_is_not_rounded() {
    let mut e = shell(53.0, 1)[0];
    assert!(Model::new(vec![e], 20.0, R).unwrap().repeat.is_none());
    e.semi_major_km = (e.mu / (15.0 * EARTH_RATE).powi(2)).cbrt();
    let model = Model::new(vec![e], 1.0, R).unwrap();
    assert!((model.repeat.unwrap() - TAU / EARTH_RATE).abs() < 1e-6);
    let point = e.sample(1234.567);
    let longitude = point.inertial_lon_deg - (EARTH_RATE * 1234.567).to_degrees();
    let r = query(&model, point.lat_deg, longitude, 1.0);
    assert!(r.cyclic);
    assert_eq!(r.events, 1);
    assert!((r.max_gap.unwrap() - TAU / EARTH_RATE).abs() < 1e-3);
    let unseen = query(&model, 50.0, 12.0, 1.0);
    assert_eq!(unseen.events, 0);
    assert_eq!(unseen.max_gap, Some(f64::INFINITY));
}

#[test]
fn event_isolation_finds_narrow_passes_and_matches_independent_dense_bracketing() {
    for inc in [53.0, 90.0, 98.0] {
        let e = shell(inc, 1)[0];
        let at = 1234.567;
        let p = e.sample(at);
        let lat = p.lat_deg;
        let lon = p.inertial_lon_deg - (EARTH_RATE * at).to_degrees();
        let x = ground_position(lat, lon, 1.0, 0.0);
        let series = Series::new(&e, x, 0.0, 0.0);
        let threshold = (0.5 / R).cos();
        let end = 86400.0;
        let mut adaptive = Vec::new();
        // A whole day in a single initial interval still finds narrow events.
        series.roots(0.0, end, threshold, &mut adaptive);
        adaptive.sort_by(f64::total_cmp);
        adaptive.dedup_by(|a, b| (*a - *b).abs() < 2.0 * TIME_TOLERANCE);
        assert!(adaptive.iter().any(|t| (t - at).abs() < TIME_TOLERANCE));
        let mut dense = Vec::new();
        for i in 0..43200 {
            let mut a = i as f64 * 2.0;
            let mut b = a + 2.0;
            let fa = series.value(a)[1];
            if fa * series.value(b)[1] <= 0.0 {
                for _ in 0..30 {
                    let m = (a + b) / 2.0;
                    if fa * series.value(m)[1] > 0.0 {
                        a = m;
                    } else {
                        b = m;
                    }
                }
                let t = (a + b) / 2.0;
                if series.value(t)[0] >= threshold {
                    dense.push(t);
                }
            }
        }
        assert_eq!(adaptive.len(), dense.len());
        for (a, b) in adaptive.iter().zip(dense) {
            assert!((a - b).abs() < TIME_TOLERANCE);
        }
    }
}

#[test]
fn cancellation_and_unsupported_models_are_explicit() {
    let mut e = shell(53.0, 1)[0];
    e.eccentricity = 0.1;
    assert!(Model::new(vec![e], 20.0, R).is_err());
    e.eccentricity = 0.0;
    e.propagation = Propagation::J2;
    assert!(Model::new(vec![e], 20.0, R).is_err());
    let model = Model::new(shell(53.0, 1), 20.0, R).unwrap();
    assert_eq!(
        model
            .point(20.0, 10.0, 0.0, 0.0, 86400.0, |_| false)
            .unwrap_err(),
        "Cancelled."
    );
    assert!(model.coverage().is_none());
}

fn repeating_shell(inc: f64, n: usize, m: u32, d: u32) -> Vec<Elements> {
    let mut e = shell(inc, n);
    let ratio = crate::repeat_ground_track::RepeatRatio::new(m, d).unwrap();
    for x in &mut e {
        x.semi_major_km = (x.mu * (ratio.orbital_period() / TAU).powi(2)).cbrt();
    }
    e
}

#[test]
fn daily_polar_baseline_guarantees_full_earth_and_point_queries_obey_bound() {
    let e = repeating_shell(90.0, 134, 15, 1);
    let model = Model::new(e, 20.0, R).unwrap();
    let g = model.coverage_guarantee(PI / 2.0).unwrap();
    assert!(!g.exact);
    assert_eq!(g.interval_s, crate::repeat_ground_track::SIDEREAL_DAY);
    for (lat, lon) in [
        (-90.0, 0.0),
        (-89.9, 179.9),
        (-42.17, -11.38),
        (0.0, 121.27),
        (63.15, 10.72),
        (90.0, 0.0),
    ] {
        let result = query(&model, lat, lon, 1.0);
        let gap = result.exact_gap.or(result.max_gap).unwrap();
        assert!(gap <= g.interval_s + 0.001);
    }
}

#[test]
fn certificate_checks_geometry_not_satellite_count_or_preset_label() {
    let original = repeating_shell(90.0, 134, 15, 1);
    assert!(Model::new(repeating_shell(90.0, 133, 15, 1), 20.0, R)
        .unwrap()
        .coverage_guarantee(PI / 2.0)
        .is_none());
    assert!(Model::new(vec![original[0]; 134], 20.0, R)
        .unwrap()
        .coverage_guarantee(PI / 2.0)
        .is_none());
    assert!(Model::new(original.clone(), 19.0, R)
        .unwrap()
        .coverage_guarantee(PI / 2.0)
        .is_none());
    let mut changed = original.clone();
    changed[0].mean_anomaly_rad = changed[1].mean_anomaly_rad;
    assert!(Model::new(changed, 20.0, R)
        .unwrap()
        .coverage_guarantee(PI / 2.0)
        .is_none());
    // Same ascending-track spacing from different planes, with a global
    // longitude offset. This proves the checker does not require one plane.
    let mut planes = original.clone();
    for e in &mut planes {
        e.raan_rad = 0.73 - e.mean_anomaly_rad / 15.0;
        e.mean_anomaly_rad = 0.0;
    }
    assert!(Model::new(planes, 20.0, R)
        .unwrap()
        .coverage_guarantee(PI / 2.0)
        .is_some());
    let mut augmented = original;
    augmented.push(shell(53.0, 1)[0]);
    let model = Model::new(augmented, 20.0, R).unwrap();
    assert!(model.repeat.is_none()); // whole constellation need not repeat
    assert!(model.coverage_guarantee(PI / 2.0).is_some()); // subset still covers
}

#[test]
fn inclined_and_multi_day_certificates_respect_band_and_phase_aliasing() {
    let model = Model::new(repeating_shell(53.0, 134, 15, 1), 20.0, R).unwrap();
    assert!(model.coverage_guarantee(53.0_f64.to_radians()).is_some());
    assert!(model.coverage_guarantee(53.05_f64.to_radians()).is_none()); // edge swath not proved
    assert!(model.coverage_guarantee(PI / 2.0).is_none());
    assert!(model.latitude_limit < PI / 2.0);
    let ratio = crate::repeat_ground_track::RepeatRatio::new(31, 2).unwrap();
    let n = crate::repeat_ground_track::baseline_count(ratio, 21.0, R).unwrap();
    assert_eq!(n, 63);
    let model = Model::new(repeating_shell(90.0, n, 31, 2), 21.0, R).unwrap();
    assert_eq!(
        model.coverage_guarantee(PI / 2.0).unwrap().interval_s,
        ratio.duration()
    );
    // 62 appears sufficient by mN alone, but d=2 makes pairs share tracks.
    assert!(Model::new(repeating_shell(90.0, 62, 31, 2), 21.0, R)
        .unwrap()
        .coverage_guarantee(PI / 2.0)
        .is_none());
}

#[test]
#[ignore = "manual large constellation timing"]
fn thousand_satellite_point_query() {
    let shell = WalkerShell {
        planes: 25,
        satellites_per_plane: 40,
        ..Default::default()
    };
    let mut elements = Vec::new();
    for p in 0..25 {
        for s in 0..40 {
            elements.push(shell.elements(p, s));
        }
    }
    let model = Model::new(elements, 20.0, R).unwrap();
    let start = std::time::Instant::now();
    let r = query(&model, 45.0, 12.0, 7.0 * 86400.0);
    assert!(r.events > 1);
    assert!(r.max_gap.unwrap().is_finite());
    eprintln!(
        "1,000 satellites / 7 days / one point: {:?}; {} events",
        start.elapsed(),
        r.events
    );
}
