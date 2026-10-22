use super::*;
use crate::config::ConstellationConfig;

fn text_position(shapes: &[egui::epaint::ClippedShape], needle: &str) -> Option<egui::Pos2> {
    fn find(shape: &egui::Shape, needle: &str) -> Option<egui::Pos2> {
        match shape {
            egui::Shape::Text(t) if t.galley.job.text == needle => {
                Some(t.pos + t.galley.size() / 2.0)
            }
            egui::Shape::Vec(v) => v.iter().find_map(|s| find(s, needle)),
            _ => None,
        }
    }
    shapes.iter().find_map(|s| find(&s.shape, needle))
}

#[test]
fn sampler_selects_only_its_globe_and_changed_configuration_clears_results() {
    let ctx = egui::Context::default();
    let epoch = DateTime::from_timestamp(946728000, 0).unwrap();
    let mut tab = TabConfig::new("Earth".into());
    let mut c = ConstellationConfig::new(0);
    c.inclination = 0.0;
    c.num_planes = 1;
    c.sats_per_plane = 4;
    tab.planets[0].constellations = vec![c];
    let mut panel = Panel {
        open: true,
        ..Default::default()
    };
    let frame = |p: &mut Panel, tab: &mut TabConfig, events| {
        ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1200.0, 1200.0),
                )),
                events,
                ..Default::default()
            },
            |ui| {
                p.ui(ui, tab, 0, epoch);
            },
        )
    };
    for _ in 0..3 {
        let _ = frame(&mut panel, &mut tab, vec![]);
    }
    let out = frame(&mut panel, &mut tab, vec![]);
    assert!(text_position(&out.shapes, "Swath width").is_some()); // independently of visualization
    let pos = text_position(&out.shapes, "Sampler").unwrap();
    for pressed in [true, false] {
        let _ = frame(
            &mut panel,
            &mut tab,
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
    assert!(panel.armed_for(0, 0));
    assert!(!panel.armed_for(1, 0));
    panel.pick(1, 0, 0.0, 0.0);
    assert!(panel.selected.is_none());
    panel.pick(0, 0, 0.0, 179.5);
    assert!(!panel.armed);
    assert!(panel.needs_run);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while panel.report.is_none() && std::time::Instant::now() < deadline {
        let _ = frame(&mut panel, &mut tab, vec![]);
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(panel.error.is_none(), "{:?}", panel.error);
    assert!(panel
        .report
        .as_ref()
        .unwrap()
        .exact_gap
        .unwrap()
        .is_finite());
    tab.settings.pushbroom.swath_km = 40.0;
    let _ = frame(&mut panel, &mut tab, vec![]);
    assert!(panel.report.is_none());
    assert!(panel.pending.is_none());
    assert_eq!(panel.selected, Some((0.0, 179.5)));
    panel.armed = true;
    panel.open = false;
    let _ = frame(&mut panel, &mut tab, vec![]);
    assert!(!panel.armed_for(0, 0));
}

#[test]
fn switching_views_cancels_selection_and_an_unsupported_model_is_explicit() {
    let mut tab = TabConfig::new("Earth".into());
    let mut c = ConstellationConfig::new(0);
    c.eccentricity = 0.1;
    tab.planets[0].constellations = vec![c];
    let mut panel = Panel::default();
    panel.refresh_scope((0, 0, 1), &tab);
    assert!(panel.model.as_ref().unwrap().is_err());
    panel.selected = Some((20.0, 30.0));
    panel.armed = true;
    panel.refresh_scope((1, 0, 1), &tab);
    assert!(panel.selected.is_none());
    assert!(!panel.armed);
}

#[test]
fn baseline_is_an_editable_constellation_and_its_guarantee_is_invalidated_by_edits() {
    let mut tab = TabConfig::new("Earth".into());
    tab.planets[0].add_constellation();
    let old_altitude = tab.planets[0].constellations[0].altitude_km;
    let mut panel = Panel::default();
    panel.designer.daily_preset();
    let band = panel
        .designer
        .add_baseline(&mut tab.planets[0], 20.0)
        .unwrap();
    assert_eq!(band, 90.0);
    let p = &tab.planets[0];
    assert_eq!(p.constellations.len(), 2);
    assert!(p.constellations[0].hidden);
    assert_eq!(p.constellations[0].altitude_km, old_altitude);
    let c = &p.constellations[1];
    assert!(!c.hidden);
    assert_eq!(c.num_planes, 1);
    assert_eq!(c.sats_per_plane, 134);
    let epoch = DateTime::from_timestamp(946728000, 0).unwrap();
    let key = crate::revisit::fingerprint(p, 20.0, epoch);
    panel.refresh_scope((0, 0, key), &tab);
    let model = panel.model.as_ref().unwrap().as_ref().unwrap();
    assert!(model.repeat.is_some());
    assert!(model
        .coverage_guarantee(std::f64::consts::FRAC_PI_2)
        .is_some());
    tab.planets[0].constellations[1].sats_per_plane = 133;
    let key = crate::revisit::fingerprint(&tab.planets[0], 20.0, epoch);
    panel.refresh_scope((0, 0, key), &tab);
    assert!(panel
        .model
        .as_ref()
        .unwrap()
        .as_ref()
        .unwrap()
        .coverage_guarantee(std::f64::consts::FRAC_PI_2)
        .is_none());
}

#[test]
fn pane_labels_distinguish_unreachable_unverified_and_proved_coverage() {
    use leo_sim::orbit::WalkerShell;
    let ctx = egui::Context::default();
    let model = Model::new(
        vec![WalkerShell {
            inclination_deg: 53.0,
            ..Default::default()
        }
        .elements(0, 0)],
        20.0,
        6371.0,
    )
    .unwrap();
    let out = ctx.run_ui(Default::default(), |ui| coverage_ui(ui, &model, 90.0));
    assert!(text_position(&out.shapes, "Coverage impossible for this region").is_some());
    let out = ctx.run_ui(Default::default(), |ui| coverage_ui(ui, &model, 50.0));
    assert!(text_position(&out.shapes, "Coverage guarantee: unverified").is_some());
    let mut tab = TabConfig::new("Earth".into());
    let mut panel = Panel::default();
    panel.designer.daily_preset();
    panel
        .designer
        .add_baseline(&mut tab.planets[0], 20.0)
        .unwrap();
    panel.refresh_scope((0, 0, 1), &tab);
    let model = panel.model.as_ref().unwrap().as_ref().unwrap();
    let out = ctx.run_ui(Default::default(), |ui| coverage_ui(ui, model, 90.0));
    assert!(text_position(&out.shapes, "Coverage guaranteed for the selected region").is_some());
}

#[test]
fn designer_buttons_add_the_daily_preset_and_open_its_repeat_cycle_forecast() {
    let ctx = egui::Context::default();
    ctx.global_style_mut(|s| s.animation_time = 0.0);
    let epoch = DateTime::from_timestamp(946728000, 0).unwrap();
    let mut tab = TabConfig::new("Earth".into());
    tab.planets[0].constellations.clear();
    let mut panel = Panel {
        open: true,
        ..Default::default()
    };
    let frame = |p: &mut Panel, tab: &mut TabConfig, events| {
        let mut request = None;
        let output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1800.0, 2200.0),
                )),
                events,
                ..Default::default()
            },
            |ui| {
                request = p.ui(ui, tab, 0, epoch);
            },
        );
        (output, request)
    };
    for _ in 0..3 {
        let _ = frame(&mut panel, &mut tab, vec![]);
    }
    for label in [
        "Repeat-orbit designer",
        "Daily polar preset",
        "Add guaranteed baseline",
        "Repeat-orbit designer",
        "Sample one repeat cycle…",
    ] {
        for _ in 0..3 {
            let _ = frame(&mut panel, &mut tab, vec![]);
        }
        let (output, _) = frame(&mut panel, &mut tab, vec![]);
        if label == "Sample one repeat cycle…" {
            assert_eq!(
                tab.planets[0].constellations.len(),
                1,
                "Add button did not create the baseline"
            );
            assert!(panel
                .model
                .as_ref()
                .unwrap()
                .as_ref()
                .unwrap()
                .repeat
                .is_some());
        }
        let pos = text_position(&output.shapes, label)
            .unwrap_or_else(|| panic!("Missing control: {label}"));
        let mut request = None;
        for pressed in [true, false] {
            let (_, r) = frame(
                &mut panel,
                &mut tab,
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
            request = request.or(r);
        }
        if label == "Sample one repeat cycle…" {
            let r = request.unwrap();
            assert_eq!(r.planet, 0);
            assert_eq!(r.cycle_s, Some(leo_sim::repeat_ground_track::SIDEREAL_DAY));
        }
    }
    assert_eq!(tab.planets[0].constellations.len(), 1);
    let c = &tab.planets[0].constellations[0];
    assert_eq!(c.sats_per_plane, 134);
    assert_eq!(c.num_planes, 1);
    let model = panel.model.as_ref().unwrap().as_ref().unwrap();
    assert!(model
        .coverage_guarantee(std::f64::consts::FRAC_PI_2)
        .is_some());
}

#[test]
fn literature_button_adds_the_design_and_leaves_j2_forecast_available() {
    let ctx = egui::Context::default();
    ctx.global_style_mut(|s| s.animation_time = 0.0);
    let epoch = DateTime::from_timestamp(946728000, 0).unwrap();
    let mut tab = TabConfig::new("Earth".into());
    tab.planets[0].add_constellation();
    let mut panel = Panel {
        open: true,
        ..Default::default()
    };
    let frame = |p: &mut Panel, tab: &mut TabConfig, events| {
        let mut request = None;
        let output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1500.0, 1800.0),
                )),
                events,
                ..Default::default()
            },
            |ui| {
                request = p.ui(ui, tab, 0, epoch);
            },
        );
        (output, request)
    };
    for label in [
        "Literature presets",
        "Add literature preset",
        "Literature presets",
        "Sampled latitude forecast…",
    ] {
        for _ in 0..3 {
            let _ = frame(&mut panel, &mut tab, vec![]);
        }
        let (output, _) = frame(&mut panel, &mut tab, vec![]);
        let pos = text_position(&output.shapes, label).unwrap_or_else(|| panic!("Missing {label}"));
        let mut request = None;
        for pressed in [true, false] {
            let (_, r) = frame(
                &mut panel,
                &mut tab,
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
            request = request.or(r);
        }
        if label == "Add literature preset" {
            assert_eq!(tab.planets[0].constellations.len(), 2);
            assert_eq!(tab.planets[0].constellations[1].total_sats(), 5);
            assert!(tab.planets[0].constellations[0].hidden);
            assert_eq!(tab.settings.pushbroom.swath_km, 60.0);
            assert!(panel.model.as_ref().unwrap().is_err()); // no silent Keplerian substitution
        }
        if label == "Sampled latitude forecast…" {
            let r = request.unwrap();
            assert_eq!(r.planet, 0);
            assert!(r.cycle_s.is_none());
        }
    }
}
