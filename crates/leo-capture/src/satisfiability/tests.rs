use super::*;
use crate::{session::Session, workload::Illumination};

#[test]
fn assessment_matches_capture_and_does_not_call_storage_overflow_an_acquisition_failure() {
    let mut inputs = crate::tests::session_inputs();
    inputs.options.duration_s = 60.0;
    inputs.options.storage_bytes_per_satellite = 1;
    let result = evaluate(&inputs, |_| Ok(())).unwrap();
    assert!(result.counts.all_satisfied());
    assert_eq!(result.counts.satisfied, 2);
    assert!(result.report.satellites[0]
        .first_capacity_exceeded_s
        .is_some());
    let mut session = Session::new(inputs, |_| true).unwrap();
    session.advance(12).unwrap();
    assert_eq!(
        serde_json::to_value(&result.report.jobs).unwrap(),
        serde_json::to_value(&session.report().jobs).unwrap()
    );
    assert_eq!(
        result.report.retained_physical_bytes,
        session.report().retained_physical_bytes
    );
    assert!(result.report.reuse_saved_bytes > 0.0);
}

#[test]
fn checks_every_cadence_window_and_never_passes_untested_long_cadences_or_phases() {
    let mut inputs = crate::tests::session_inputs();
    let mut monthly = inputs.jobs[0].clone();
    monthly.aoi.id = "monthly".into();
    monthly.cadence_hours = 720.0;
    let mut region = inputs.regions[0].clone();
    region.job_key = monthly.aoi.key();
    inputs.jobs.push(monthly);
    inputs.regions.push(region);
    let result = evaluate(&inputs, |_| Ok(())).unwrap();
    assert_eq!(result.counts.satisfied, 0);
    assert_eq!(result.counts.missed, 2);
    assert_eq!(result.counts.untested, 1);
    assert_eq!(result.counts.complete_updates, 2);
    assert_eq!(result.counts.due_updates, 4);
    assert!(!result.counts.all_satisfied());
    assert_eq!(result.cadences.len(), 2);
    inputs.options.duration_s = 60.0;
    for job in &mut inputs.jobs {
        job.acquisition.phase_hours = 0.005;
    }
    let result = evaluate(&inputs, |_| Ok(())).unwrap();
    assert_eq!(result.counts.untested, 3);
    assert!(!result.counts.all_satisfied());
}

#[test]
fn partial_independent_output_is_useful_but_does_not_count_as_a_full_aoi_update() {
    let mut inputs = crate::tests::session_inputs();
    inputs.options.duration_s = 60.0;
    inputs.swath_km = 5.0;
    let result = evaluate(&inputs, |_| Ok(())).unwrap();
    assert_eq!(result.counts.partial, 2);
    assert_eq!(result.counts.missed, 0);
    assert_eq!(result.report.jobs[0].ready_area_km2, 0.0);
    assert!(result.report.jobs[1].ready_area_km2 > 0.0);
    assert!(!result.counts.all_satisfied());
}

#[test]
fn daylight_and_run_start_are_not_replaced_with_average_revisit() {
    let mut inputs = crate::tests::session_inputs();
    inputs.options.duration_s = 60.0;
    for job in &mut inputs.jobs {
        job.acquisition.illumination = Illumination::Daylight;
        job.acquisition.solar_time_hours = 0.0;
        job.acquisition.solar_tolerance_hours = 0.1;
    }
    assert_eq!(evaluate(&inputs, |_| Ok(())).unwrap().counts.missed, 2);
    for job in &mut inputs.jobs {
        job.acquisition.illumination = Illumination::AnyTime;
    }
    assert!(evaluate(&inputs, |_| Ok(()))
        .unwrap()
        .counts
        .all_satisfied());
    inputs.start_s = 120.0;
    assert_eq!(evaluate(&inputs, |_| Ok(())).unwrap().counts.missed, 2);
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
