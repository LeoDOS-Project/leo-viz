use crate::{
    baseline::*,
    geometry::{self, Atlas, Region},
    workload::*,
};
use geo::{Area, BooleanOps, BoundingRect, Rect};
use leo_sim::{
    acquisition::CaptureProfile,
    geometry::{
        gmst, ground_position, local_solar_time_hours, sun_direction, EARTH_RADIUS_KM as R,
    },
};

fn ring(x: f64, y: f64, w: f64) -> Vec<[f64; 2]> {
    vec![[x, y], [x + w, y], [x + w, y + w], [x, y + w], [x, y]]
}
fn fixture(polygons: geometry::Polygons, count: usize) -> (Vec<Job>, Vec<Region>) {
    let mut jobs = Vec::new();
    let mut regions = Vec::new();
    for i in 0..count {
        let aoi = AoiRef {
            source: "test".into(),
            id: i.to_string(),
            area_km2: 100.0,
            longitude: 0.15,
            latitude: 0.15,
            geometry_hash: hash(&serde_json::to_vec(&polygons).unwrap()),
        };
        regions.push(Region {
            job_key: aoi.key(),
            polygons: polygons.clone(),
        });
        jobs.push(Job {
            aoi,
            pipeline: Pipeline::Map,
            cadence_hours: 1.0,
            samples: 1,
            map_bytes_per_km2: 1.0,
            window_bytes_per_km2: 1.0,
            acquisition: Acquisition {
                solar_tolerance_hours: 12.0,
                ..Default::default()
            },
        });
    }
    (jobs, regions)
}
fn atlas(jobs: &[Job], regions: &[Region]) -> Atlas {
    Atlas::new(jobs, regions, 2.0, 0.5, |_| true).unwrap()
}
fn event(
    patch: usize,
    start: f64,
    slot: u64,
    shapes: Vec<(usize, geometry::Shape)>,
) -> Opportunity {
    Opportunity {
        satellite: 0,
        patch,
        start_s: start,
        end_s: start + 1.0,
        crops: shapes
            .into_iter()
            .map(|(job, geometry)| Crop {
                job,
                slot,
                start_s: start,
                end_s: start + 1.0,
                geometry,
            })
            .collect(),
    }
}

#[test]
fn mosaics_accumulate_unique_area_and_independent_crops_are_ready_immediately() {
    let (mut jobs, regions) = fixture(vec![vec![ring(0.1, 0.1, 0.1)]], 2);
    jobs[0].acquisition.fulfilment = Fulfilment::Mosaic;
    let atlas = atlas(&jobs, &regions);
    let full = atlas.patches[0].pieces[0].shape.clone();
    let b = full.bounding_rect().unwrap();
    let middle = (b.min().x + b.max().x) / 2.0;
    let left = full.intersection(
        &Rect::new(
            b.min(),
            geo::Coord {
                x: middle,
                y: b.max().y,
            },
        )
        .to_polygon(),
    );
    let right = full.difference(&left);
    let options = Options {
        duration_s: 3600.0,
        storage_bytes_per_satellite: 1,
        ..Default::default()
    };
    let profile = CaptureProfile::default();
    let mut baseline = Baseline::new(&jobs, &atlas, 1, 0.0, 20.0, profile, &options).unwrap();
    let first = event(0, 1.0, 0, vec![(0, left.clone()), (1, left.clone())]);
    baseline.accept(&first).unwrap();
    assert_eq!(baseline.report.jobs[0].ready_area_km2, 0.0);
    assert!(baseline.report.jobs[1].ready_area_km2 > 0.0);
    let once = baseline.report.retained_physical_bytes;
    baseline
        .accept(&event(0, 10.0, 0, vec![(0, left.clone()), (1, left)]))
        .unwrap();
    assert!((baseline.report.retained_physical_bytes - once).abs() < 1.0);
    baseline
        .accept(&event(0, 20.0, 0, vec![(0, right.clone()), (1, right)]))
        .unwrap();
    let report = baseline.finish(OpportunityStats::default());
    for r in &report.jobs {
        assert_eq!(r.complete_due_updates, 1);
        assert_eq!(r.pending_updates, 0);
    }
    let expected = profile.stored_bytes_for_area(full.unsigned_area()).unwrap();
    assert!((report.retained_physical_bytes / expected - 1.0).abs() < 1e-6);
    assert!((report.fulfilled_logical_bytes / expected - 2.0).abs() < 1e-6);
    assert!((report.reuse_saved_bytes / expected - 1.0).abs() < 1e-6);
    assert!((report.reused_physical_bytes / expected - 1.0).abs() < 1e-6);
    assert_eq!(report.satellites[0].first_capacity_exceeded_s, Some(2.0));
    assert_eq!(report.dropped_jobs, 0);
}

#[test]
fn partial_overlap_shares_only_intersection_and_no_pixels_cross_slot_boundaries() {
    let (mut jobs, mut regions) = fixture(vec![vec![ring(0.1, 0.1, 0.1)]], 2);
    regions[1].polygons = vec![vec![ring(0.15, 0.1, 0.1)]];
    jobs[1].aoi.geometry_hash = hash(&serde_json::to_vec(&regions[1].polygons).unwrap());
    jobs[0].acquisition.fulfilment = Fulfilment::Mosaic;
    let atlas = atlas(&jobs, &regions);
    let pieces = &atlas.patches[0].pieces;
    let a = pieces[0].shape.clone();
    let b = pieces[1].shape.clone();
    let options = Options {
        duration_s: 5400.0,
        ..Default::default()
    };
    let mut baseline = Baseline::new(
        &jobs,
        &atlas,
        1,
        0.0,
        20.0,
        CaptureProfile::default(),
        &options,
    )
    .unwrap();
    let half = a.intersection(&b);
    baseline
        .accept(&event(0, 1.0, 0, vec![(0, half.clone()), (1, b.clone())]))
        .unwrap();
    let first_bytes = baseline.report.retained_physical_bytes;
    baseline
        .accept(&event(0, 3601.0, 1, vec![(0, a.difference(&half))]))
        .unwrap();
    let report = baseline.finish(OpportunityStats::default());
    assert_eq!(report.jobs[0].complete_due_updates, 0);
    assert_eq!(report.jobs[0].partial_due_updates, 1);
    assert_eq!(report.jobs[0].complete_pending_updates, 0);
    assert_eq!(report.jobs[0].pending_updates, 1);
    assert_eq!(report.jobs[0].ready_area_km2, 0.0);
    let density = CaptureProfile::default()
        .stored_bytes_for_area(1.0)
        .unwrap();
    assert!((first_bytes / density - b.unsigned_area()).abs() < 1e-5);
    assert!((report.reuse_saved_bytes / density - half.unsigned_area()).abs() < 1e-5);
}

#[test]
fn atlas_preserves_holes_dateline_and_polar_geometry() {
    let (jobs, regions) = fixture(vec![vec![ring(0.1, 0.1, 0.4), ring(0.2, 0.2, 0.1)]], 1);
    let atlas = atlas(&jobs, &regions);
    let area = atlas.job_areas_km2[0];
    let rectangle = |south: f64, north: f64, width: f64| {
        R * R * width.to_radians() * (north.to_radians().sin() - south.to_radians().sin())
    };
    let oracle = rectangle(0.1, 0.5, 0.4) - rectangle(0.2, 0.3, 0.1);
    assert!((area / oracle - 1.0).abs() < 1e-6);
    let (jobs, regions) = fixture(
        vec![vec![ring(179.9, 80.0, 0.1)], vec![ring(-180.0, 80.0, 0.1)]],
        1,
    );
    let at = Atlas::new(&jobs, &regions, 2.0, 0.5, |_| true).unwrap();
    assert_eq!(at.patch_count(), 2);
    assert!((at.job_areas_km2[0] / (2.0 * rectangle(80.0, 80.1, 0.1)) - 1.0).abs() < 1e-5);
    let (jobs, regions) = fixture(vec![vec![ring(20.0, 89.9, 0.1)]], 1);
    let at = Atlas::new(&jobs, &regions, 2.0, 0.25, |_| true).unwrap();
    assert!((at.job_areas_km2[0] / rectangle(89.9, 90.0, 0.1) - 1.0).abs() < 1e-4);
}

fn simulated_pass(jobs: &[Job], atlas: &Atlas, epoch: f64, step: f64) -> Report {
    let options = Options {
        duration_s: 20.0,
        step_s: step,
        edge_km: 0.25,
        ..Default::default()
    };
    let mut baseline = Baseline::new(
        jobs,
        atlas,
        1,
        epoch,
        20.0,
        CaptureProfile::default(),
        &options,
    )
    .unwrap();
    let stats = opportunities(
        jobs,
        atlas,
        1,
        epoch,
        20.0,
        &options,
        |_, t| {
            Ok(ground_position(
                0.15,
                -0.5 + t * 0.06,
                R + 560.0,
                gmst(epoch + t),
            ))
        },
        |op| baseline.accept(&op),
        |_| true,
    )
    .unwrap();
    baseline.finish(stats)
}

#[test]
fn generator_sweeps_between_samples_gates_daylight_and_job_time_and_converges() {
    let (mut jobs, regions) = fixture(vec![vec![ring(0.1, 0.1, 0.1)]], 2);
    let epoch = 946728000.0; // near local noon at Greenwich, January 2000
    let local = local_solar_time_hours(
        ground_position(0.15, 0.15, 1.0, gmst(epoch)),
        sun_direction(epoch),
    );
    jobs[0].acquisition.solar_time_hours = local;
    jobs[0].acquisition.solar_tolerance_hours = 1.0;
    jobs[1].acquisition.solar_time_hours = (local + 12.0).rem_euclid(24.0);
    jobs[1].acquisition.solar_tolerance_hours = 1.0;
    let at = atlas(&jobs, &regions);
    let coarse = simulated_pass(&jobs, &at, epoch, 10.0);
    let fine = simulated_pass(&jobs, &at, epoch, 5.0);
    assert_eq!(coarse.jobs[0].complete_pending_updates, 1, "{coarse:#?}");
    assert_eq!(coarse.jobs[1].captured_area_km2, 0.0);
    assert_eq!(coarse.reuse_saved_bytes, 0.0);
    assert!((coarse.retained_physical_bytes / fine.retained_physical_bytes - 1.0).abs() < 1e-5);
    jobs[0].acquisition.solar_tolerance_hours = 12.0;
    jobs[1].acquisition.solar_tolerance_hours = 12.0;
    let at = atlas(&jobs, &regions);
    let night = simulated_pass(&jobs, &at, epoch + 43200.0, 5.0);
    assert_eq!(night.retained_physical_bytes, 0.0);
    assert!(night.opportunities.timing_or_daylight_rejected_segments > 0);
}

#[test]
fn deterministic_normal_assignment_priorities_and_circular_time() {
    let (jobs, _) = fixture(vec![vec![ring(0.1, 0.1, 0.1)]], 20000);
    let g = Generator::default();
    let generated = g.generate(jobs.into_iter().map(|j| j.aoi)).unwrap();
    let times: Vec<_> = generated
        .iter()
        .map(|j| j.acquisition.solar_time_hours)
        .collect();
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    assert!((mean - 12.0).abs() < 0.1);
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    assert!((variance.sqrt() - 3.95).abs() < 0.1);
    assert!(times.iter().all(|t| (0.0..24.0).contains(t)));
    let mosaics = generated
        .iter()
        .filter(|j| j.acquisition.fulfilment == Fulfilment::Mosaic)
        .count();
    assert!((9500..10500).contains(&mosaics));
    let order = lowest_priority_first(&generated);
    assert!(order
        .windows(2)
        .all(|w| generated[w[0]].acquisition.priority <= generated[w[1]].acquisition.priority));
    let a = Acquisition {
        solar_time_hours: 23.5,
        solar_tolerance_hours: 1.0,
        ..Default::default()
    };
    assert!(a.accepts(0.25, 5.0));
    assert!(!a.accepts(1.0, 5.0));
    assert!(!a.accepts(0.25, -1.0));
}

#[test]
fn cancelled_or_mismatched_geometry_fails_explicitly() {
    let (jobs, mut regions) = fixture(vec![vec![ring(0.1, 0.1, 0.1)]], 1);
    assert!(Atlas::new(&jobs, &regions, 2.0, 1.0, |_| false).is_err());
    regions[0].polygons[0][0][0][0] += 0.001;
    assert!(Atlas::new(&jobs, &regions, 2.0, 1.0, |_| true).is_err());
}

#[test]
fn generator_splits_a_sweep_at_deadline_without_completing_either_mosaic() {
    let (mut jobs, regions) = fixture(vec![vec![ring(0.0, 0.1, 0.2)]], 1);
    jobs[0].cadence_hours = 10.0 / 3600.0;
    jobs[0].acquisition.fulfilment = Fulfilment::Mosaic;
    let at = atlas(&jobs, &regions);
    let report = simulated_pass(&jobs, &at, 946728000.0, 20.0);
    assert_eq!(report.jobs[0].due_updates, 2);
    assert_eq!(report.jobs[0].complete_due_updates, 0);
    assert_eq!(report.jobs[0].partial_due_updates, 2);
    assert_eq!(report.jobs[0].ready_area_km2, 0.0);
    assert!(report.jobs[0].captured_area_km2 > 0.0);
}

fn session_inputs() -> crate::session::Inputs {
    use crate::session::{Inputs, Orbit, Satellite};
    let epoch = 946728000.0;
    // Eastbound equatorial orbit begins just west of two overlapping AOIs.
    let (mut jobs, regions) = fixture(
        vec![vec![vec![
            [-0.1, -0.05],
            [1.0, -0.05],
            [1.0, 0.05],
            [-0.1, 0.05],
            [-0.1, -0.05],
        ]]],
        2,
    );
    for job in &mut jobs {
        job.cadence_hours = 1.0 / 60.0;
        job.aoi.latitude = 0.0;
        job.aoi.longitude = 0.5;
    }
    jobs[0].acquisition.fulfilment = Fulfilment::Mosaic;
    let elements = leo_sim::orbit::Elements {
        semi_major_km: R + 550.0,
        eccentricity: 0.0,
        inclination_rad: 0.0,
        raan_rad: -gmst(epoch) + 0.2_f64.to_radians(),
        argument_rad: 0.0,
        mean_anomaly_rad: 0.0,
        mu: 398600.4418,
        radius_km: R,
        j2: 0.00108263,
        propagation: leo_sim::orbit::Propagation::Keplerian,
    };
    Inputs {
        orbit_epoch_unix_s: epoch,
        start_s: 0.0,
        satellites: vec![Satellite {
            name: "S1".into(),
            constellation: "test".into(),
            orbit: Orbit::Analytical { elements },
        }],
        jobs,
        regions,
        capture: CaptureProfile::default(),
        swath_km: 20.0,
        options: Options {
            duration_s: 120.0,
            step_s: 5.0,
            ..Default::default()
        },
    }
}
#[test]
fn session_checkpoint_preserves_partial_mosaic_and_resumes_identical_crops() {
    use crate::session::Session;
    let input = session_inputs();
    let mut continuous = Session::new(input.clone(), |_| true).unwrap();
    continuous.advance(24).unwrap();
    assert!(continuous.report().retained_physical_bytes > 0.0);
    assert!(continuous.report().reuse_saved_bytes > 0.0);
    assert_eq!(continuous.meters().due_updates, 4);
    assert_eq!(continuous.meters().complete_updates, 2);
    assert_eq!(continuous.meters().failed_updates, 2);
    let mut split = Session::new(input, |_| true).unwrap();
    split.advance(1).unwrap();
    assert!(split.report().jobs[0].captured_area_km2 > 0.0);
    assert_eq!(split.report().jobs[0].ready_area_km2, 0.0);
    assert!(split.report().jobs[1].ready_area_km2 > 0.0);
    assert_eq!(split.meters().failed_updates, 0);
    let path = std::env::temp_dir().join(format!(
        "leo-session-partial-{}.json.gz",
        std::process::id()
    ));
    split.save(&path).unwrap();
    let saved_positions = split.positions().unwrap();
    let mut restored = Session::load(&path, |_| true).unwrap();
    assert_eq!(restored.positions().unwrap(), saved_positions);
    assert_eq!(
        serde_json::to_value(split.state()).unwrap(),
        serde_json::to_value(restored.state()).unwrap()
    );
    for _ in 1..24 {
        restored.advance(1).unwrap();
    }
    assert_eq!(restored.elapsed_s(), 120.0);
    assert_eq!(
        serde_json::to_value(continuous.state()).unwrap(),
        serde_json::to_value(restored.state()).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&continuous.state().captures).unwrap(),
        serde_json::to_value(&restored.state().captures).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&continuous.report().jobs).unwrap(),
        serde_json::to_value(&restored.report().jobs).unwrap()
    );
    assert_eq!(
        continuous.meters().failed_updates,
        restored.meters().failed_updates
    );
    // Completed requests, ledger and counters can also be restored at the horizon.
    restored.save(&path).unwrap();
    let final_restore = Session::load(&path, |_| true).unwrap();
    assert!(final_restore.finished());
    std::fs::remove_file(path).unwrap();
}
#[test]
fn session_deadlines_without_captures_and_rolling_window_expire_on_simulation_clock() {
    use crate::session::Session;
    let mut input = session_inputs();
    input.options.duration_s = 86460.0;
    input.options.step_s = 30.0;
    // One initial daytime capture, then no further allowed cadence slots.
    for job in &mut input.jobs {
        job.cadence_hours = 48.0;
    }
    let mut s = Session::new(input, |_| true).unwrap();
    s.advance(2).unwrap();
    let total = s.report().retained_physical_bytes;
    assert!(total > 0.0);
    assert_eq!(s.meters().constellation_last_24h_bytes(), total);
    s.advance(2880).unwrap();
    assert_eq!(s.report().retained_physical_bytes, total);
    assert_eq!(s.meters().constellation_last_24h_bytes(), 0.0);
    assert_eq!(s.meters().captures_last_24h, 0);
    assert_eq!(s.meters().failed_updates, 0); // deadline remains in the future
    let mut input = session_inputs();
    for job in &mut input.jobs {
        job.acquisition.min_solar_elevation_deg = 90.0;
    }
    let mut s = Session::new(input, |_| true).unwrap();
    s.advance(24).unwrap();
    assert_eq!(s.meters().failed_updates, 4);
    assert_eq!(s.meters().affected_jobs, 2);
    assert_eq!(s.meters().partial_updates, 0);
    s.advance(100).unwrap();
    assert_eq!(s.meters().failed_updates, 4); // horizon is idempotent
}
#[test]
fn session_rejects_positions_that_do_not_match_saved_constellation() {
    use crate::session::Session;
    use std::io::{Read, Write};
    let path = std::env::temp_dir().join(format!(
        "leo-session-corrupt-{}.json.gz",
        std::process::id()
    ));
    let mut s = Session::new(session_inputs(), |_| true).unwrap();
    s.advance(1).unwrap();
    s.save(&path).unwrap();
    let mut json = String::new();
    flate2::read::GzDecoder::new(std::fs::File::open(&path).unwrap())
        .read_to_string(&mut json)
        .unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
    value["positions_km"][0][0] = serde_json::json!(0.0);
    let mut out = flate2::write::GzEncoder::new(
        std::fs::File::create(&path).unwrap(),
        flate2::Compression::fast(),
    );
    out.write_all(&serde_json::to_vec(&value).unwrap()).unwrap();
    out.finish().unwrap();
    assert!(Session::load(&path, |_| true)
        .err()
        .unwrap()
        .contains("positions"));
    std::fs::remove_file(path).unwrap();
}

#[test]
#[ignore = "50,000 AOIs / 169 satellites / one simulated hour checkpoint smoke test"]
fn session_scale_checkpoint_roundtrip() {
    use crate::session::{Inputs, Orbit, Satellite, Session};
    let records = crate::dataset::decode(include_bytes!(
        "../../../assets/aois/default-jobs-oriented.json.gz"
    ))
    .unwrap();
    let jobs = Generator::default()
        .generate(records.iter().map(|(a, _)| a.clone()))
        .unwrap();
    let shell = leo_sim::orbit::WalkerShell {
        planes: 13,
        satellites_per_plane: 13,
        altitude_km: 561.3856170563513,
        inclination_deg: 70.0,
        phasing: 0,
        ..Default::default()
    };
    let satellites = (0..13)
        .flat_map(|plane| (0..13).map(move |sat| (plane, sat)))
        .map(|(plane, sat)| Satellite {
            name: format!("{plane}:{sat}"),
            constellation: "ADAM".into(),
            orbit: Orbit::Analytical {
                elements: shell.elements(plane, sat),
            },
        })
        .collect();
    let inputs = Inputs {
        orbit_epoch_unix_s: 946728000.0,
        start_s: 0.0,
        satellites,
        jobs,
        regions: records.into_iter().map(|(_, r)| r).collect(),
        capture: Default::default(),
        swath_km: 20.0,
        options: Options {
            duration_s: 7200.0,
            ..Default::default()
        },
    };
    let start = std::time::Instant::now();
    let mut s = Session::new(inputs, |_| true).unwrap();
    for _ in 0..60 {
        s.advance(12).unwrap();
    }
    let path =
        std::env::temp_dir().join(format!("leo-session-scale-{}.json.gz", std::process::id()));
    s.save(&path).unwrap();
    let mut restored = Session::load(&path, |_| true).unwrap();
    s.advance(12).unwrap();
    restored.advance(12).unwrap();
    assert_eq!(
        serde_json::to_value(s.report()).unwrap(),
        serde_json::to_value(restored.report()).unwrap()
    );
    assert_eq!(s.positions().unwrap(), restored.positions().unwrap());
    eprintln!("50k checkpoint: {:.2}s including atlas, 1h capture, save/load/validation, continuation; {} records; {} bytes compressed",start.elapsed().as_secs_f64(),s.meters().captures_total,std::fs::metadata(&path).unwrap().len());
    std::fs::remove_file(path).unwrap();
}
