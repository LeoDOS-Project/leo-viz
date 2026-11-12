use super::*;
use crate::{baseline::Crop, workload::Illumination};

#[test]
fn reach_coverage_does_not_depend_on_data_storage_phase_or_fulfilment() {
    let mut inputs = crate::tests::session_inputs();
    inputs.options.duration_s = 60.0;
    let original = evaluate(&inputs, |_| Ok(())).unwrap();
    assert!(original.counts.all_satisfied());
    inputs.options.storage_bytes_per_satellite = 0;
    inputs.capture.channels = 0; // Irrelevant even as an invalid sensor profile.
    for job in &mut inputs.jobs {
        job.acquisition.phase_hours = 0.005;
    }
    let result = evaluate(&inputs, |_| Ok(())).unwrap();
    assert!(result.counts.all_satisfied());
    assert_eq!(result.counts.satisfied, 2);
    assert_eq!(result.jobs[0].silence_max_s, original.jobs[0].silence_max_s);
    assert_eq!(result.jobs[0].silence_max_s, result.jobs[1].silence_max_s);
}

#[test]
fn short_forecasts_remain_untested_and_narrow_reach_misses_parts_of_an_aoi() {
    let mut inputs = crate::tests::session_inputs();
    inputs.jobs[0].cadence_hours = 720.0;
    let result = evaluate(&inputs, |_| Ok(())).unwrap();
    assert_eq!(result.counts.untested, 1);
    assert_eq!(result.counts.too_slow, 1);
    assert!(!result.counts.all_satisfied());
    inputs.jobs[0].cadence_hours = inputs.jobs[1].cadence_hours;
    inputs.swath_km = 5.0;
    inputs.options.duration_s = 60.0;
    let result = evaluate(&inputs, |_| Ok(())).unwrap();
    assert_eq!(result.counts.uncovered, 2);
    assert!(result.jobs[0].coverage_fraction > 0.0 && result.jobs[0].coverage_fraction < 0.9);
    inputs.swath_km = 20.0;
    assert!(evaluate(&inputs, |_| Ok(()))
        .unwrap()
        .counts
        .all_satisfied());
}

#[test]
fn daylight_and_run_start_filter_usable_observations() {
    let mut inputs = crate::tests::session_inputs();
    inputs.options.duration_s = 60.0;
    for job in &mut inputs.jobs {
        job.acquisition.illumination = Illumination::Daylight;
        job.acquisition.solar_time_hours = 0.0;
        job.acquisition.solar_tolerance_hours = 0.1;
    }
    assert_eq!(evaluate(&inputs, |_| Ok(())).unwrap().counts.uncovered, 2);
    for job in &mut inputs.jobs {
        job.acquisition.illumination = Illumination::AnyTime;
    }
    assert!(evaluate(&inputs, |_| Ok(()))
        .unwrap()
        .counts
        .all_satisfied());
    inputs.start_s = 120.0;
    assert_eq!(evaluate(&inputs, |_| Ok(())).unwrap().counts.uncovered, 2);
}

fn atlas(inputs: &Inputs) -> Atlas {
    Atlas::new(&inputs.jobs, &inputs.regions, 2.0, 1.0, |_| true).unwrap()
}
fn pass(state: &mut Revisit, atlas: &Atlas, start: f64, end: f64, first_patch_only: bool) {
    for (index, patch) in atlas.patches.iter().enumerate() {
        if first_patch_only && index > 0 {
            break;
        }
        state
            .accept(Opportunity {
                satellite: 0,
                patch: index,
                start_s: start,
                end_s: end,
                crops: patch
                    .pieces
                    .iter()
                    .map(|p| Crop {
                        job: p.job,
                        slot: 0,
                        start_s: start,
                        end_s: end,
                        geometry: p.shape.clone(),
                    })
                    .collect(),
            })
            .unwrap();
    }
}

#[test]
fn observations_in_each_fixed_window_can_still_fail_maximum_revisit() {
    let inputs = crate::tests::session_inputs(); // Two 60 s cadence windows.
    let atlas = atlas(&inputs);
    let mut state = Revisit::new(inputs.jobs.len());
    pass(&mut state, &atlas, 10.0, 10.0, false);
    pass(&mut state, &atlas, 110.0, 110.0, false);
    let result = state.finish(&inputs, &atlas, Options::from_baseline(&inputs.options));
    assert_eq!(result.counts.too_slow, 2); // 100 s gap, despite one observation per slot.
    assert_eq!(result.jobs[0].silence_min_s, 100.0);
    assert_eq!(result.jobs[0].silence_max_s, 100.0);
}

#[test]
fn boundaries_all_polygon_parts_and_temporal_uncertainty_count() {
    let inputs = crate::tests::session_inputs();
    let atlas = atlas(&inputs);
    assert!(atlas.patch_count() > 1);
    let mut state = Revisit::new(inputs.jobs.len());
    pass(&mut state, &atlas, 10.0, 10.0, false);
    pass(&mut state, &atlas, 50.0, 50.0, true);
    pass(&mut state, &atlas, 90.0, 90.0, true);
    let result = state.finish(&inputs, &atlas, Options::from_baseline(&inputs.options));
    assert_eq!(result.counts.too_slow, 2); // Unrevisited pieces have a 110 s trailing silence.
    assert_eq!(result.jobs[0].silence_min_s, 110.0);
    let mut state = Revisit::new(inputs.jobs.len());
    pass(&mut state, &atlas, 20.0, 25.0, false);
    pass(&mut state, &atlas, 80.0, 85.0, false);
    let result = state.finish(&inputs, &atlas, Options::from_baseline(&inputs.options));
    assert_eq!(result.counts.resolution_limited, 2); // Gap is bounded by 55–65 s.
    assert_eq!(result.jobs[0].silence_min_s, 55.0);
    assert_eq!(result.jobs[0].silence_max_s, 65.0);
    let mut state = Revisit::new(inputs.jobs.len());
    pass(&mut state, &atlas, 20.0, 20.0, false);
    pass(&mut state, &atlas, 80.0, 80.0, false);
    assert!(state
        .finish(&inputs, &atlas, Options::from_baseline(&inputs.options))
        .counts
        .all_satisfied());
}

#[test]
fn repeat_passes_replace_history_and_do_not_create_spurious_fresh_coverage() {
    let mut inputs = crate::tests::session_inputs();
    inputs.options.duration_s = 1000.0;
    let atlas = atlas(&inputs);
    let mut state = Revisit::new(inputs.jobs.len());
    for i in 0..100 {
        pass(&mut state, &atlas, i as f64 * 10.0, i as f64 * 10.0, false);
    }
    assert_eq!(
        state.pieces.values().map(Vec::len).sum::<usize>(),
        atlas.polygon_pieces
    );
    let result = state.finish(&inputs, &atlas, Options::from_baseline(&inputs.options));
    assert!(result.counts.all_satisfied());
    assert_eq!(result.jobs[0].silence_max_s, 10.0);
}

#[test]
fn repeated_partial_overlaps_keep_spatial_revisit_times_without_roundoff_gaps() {
    let mut inputs = crate::tests::session_inputs();
    inputs.options.duration_s = 1000.0;
    let atlas = atlas(&inputs);
    let mut state = Revisit::new(inputs.jobs.len());
    pass(&mut state, &atlas, 0.0, 0.0, false);
    for tick in 1..100 {
        for (index, patch) in atlas.patches.iter().enumerate() {
            state
                .accept(Opportunity {
                    satellite: 0,
                    patch: index,
                    start_s: tick as f64 * 10.0,
                    end_s: tick as f64 * 10.0,
                    crops: patch
                        .pieces
                        .iter()
                        .map(|p| {
                            let b = p.bounds;
                            let x = b.min().x + b.width() * if tick % 2 == 0 { 0.63 } else { 0.37 };
                            let clip = if tick % 2 == 0 {
                                Rect::new(b.min(), geo::Coord { x, y: b.max().y })
                            } else {
                                Rect::new(geo::Coord { x, y: b.min().y }, b.max())
                            };
                            Crop {
                                job: p.job,
                                slot: 0,
                                start_s: tick as f64 * 10.0,
                                end_s: tick as f64 * 10.0,
                                geometry: p
                                    .shape
                                    .intersection(&Shape::new(vec![clip.to_polygon()])),
                            }
                        })
                        .collect(),
                })
                .unwrap();
        }
    }
    let result = state.finish(&inputs, &atlas, Options::from_baseline(&inputs.options));
    for j in result.jobs {
        assert_eq!(
            j.verdict,
            Verdict::Satisfied,
            "silence {}–{}",
            j.silence_min_s,
            j.silence_max_s
        );
        assert!(j.silence_max_s <= 20.0);
    }
}

#[test]
fn cancellation_and_resource_errors_never_produce_a_partial_verdict() {
    let inputs = crate::tests::session_inputs();
    for stage in [Stage::Validate, Stage::Geometry, Stage::Forecast] {
        let result = evaluate(&inputs, |p| {
            if p.stage == stage {
                Err("Memory limit reached".into())
            } else {
                Ok(())
            }
        });
        assert_eq!(result.err().as_deref(), Some("Memory limit reached"));
    }
    let mut empty = inputs;
    empty.jobs.clear();
    assert!(evaluate(&empty, |_| Ok(())).is_err());
}
