use super::*;
use crate::{app::App, config::TabConfig, projection::ProjectionKind};

fn app(ctx: &egui::Context) -> App {
    App::new(&eframe::CreationContext::_new_kittest(ctx.clone()))
}

fn labels(shapes: &[egui::epaint::ClippedShape]) -> Vec<(String, egui::Rect)> {
    fn collect(shape: &egui::Shape, labels: &mut Vec<(String, egui::Rect)>) {
        match shape {
            egui::Shape::Text(text) => labels.push((
                text.galley.text().into(),
                egui::Rect::from_min_size(text.pos, text.galley.size()),
            )),
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| collect(shape, labels)),
            _ => {}
        }
    }
    let mut result = Vec::new();
    for shape in shapes {
        collect(&shape.shape, &mut result);
    }
    result
}

fn rect(labels: &[(String, egui::Rect)], needle: &str) -> egui::Rect {
    labels
        .iter()
        .rev()
        .find(|(text, _)| text == needle)
        .unwrap_or_else(|| {
            panic!(
                "missing {needle}: {:?}",
                labels.iter().map(|(text, _)| text).collect::<Vec<_>>()
            )
        })
        .1
}

fn settings_frame(
    app: &mut App,
    ctx: &egui::Context,
    events: Vec<egui::Event>,
) -> Vec<(String, egui::Rect)> {
    labels(&settings_output(app, ctx, events).shapes)
}

fn settings_output(
    app: &mut App,
    ctx: &egui::Context,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(650.0, 3000.0),
            )),
            events,
            ..Default::default()
        },
        |ui| {
            egui::CentralPanel::default().show_inside(ui, |ui| app.viewer.show_settings(ui));
        },
    )
}

fn click_settings(app: &mut App, ctx: &egui::Context, needle: &str) {
    let pos = rect(&settings_frame(app, ctx, Vec::new()), needle).center();
    settings_frame(app, ctx, vec![egui::Event::PointerMoved(pos)]);
    for pressed in [true, false] {
        settings_frame(
            app,
            ctx,
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Default::default(),
                },
            ],
        );
    }
}

#[test]
fn planet_settings_group_rendering_and_application_controls() {
    let ctx = egui::Context::default();
    let mut app = app(&ctx);
    settings_frame(&mut app, &ctx, Vec::new());
    let output = settings_output(&mut app, &ctx, Vec::new());
    let labels = labels(&output.shapes);
    let order = [
        "Display:",
        "Clock",
        "Speed:",
        "Time:",
        "Date:",
        "Sync time",
        "Rendering",
        "Projection:",
        "Texture:",
        "ESRI Tiles",
        "Application",
        "None",
        "Earth Observation",
        "Camera",
    ];
    for pair in order.windows(2) {
        assert!(
            rect(&labels, pair[0]).bottom() <= rect(&labels, pair[1]).top() + 1.0,
            "incorrect settings order: {pair:?}"
        );
    }
    assert!(rect(&labels, "Rendering").left() > rect(&labels, "Display:").left());
    assert!(rect(&labels, "Projection:").left() > rect(&labels, "Rendering").left());
    for section in [
        "Clock",
        "Application",
        "Camera",
        "Constellations",
        "Body",
        "Radiation",
        "Ground",
        "Aesthetics",
    ] {
        assert!(
            (rect(&labels, section).left() - rect(&labels, "Rendering").left()).abs() < 0.1,
            "{section} must be nested directly under Display"
        );
    }
    assert_eq!(
        labels.iter().filter(|(text, _)| text == "Planet").count(),
        1,
        "Planet belongs in the selector, not a duplicate heading"
    );
    let application_end = rect(&labels, "Earth Observation").bottom();
    let camera_start = rect(&labels, "Camera").top();
    assert!(!output.shapes.iter().any(|shape| matches!(&shape.shape,
        egui::Shape::LineSegment { points, .. } if points[0].y == points[1].y && points[0].y >= application_end && points[0].y <= camera_start
    )), "there must not be a horizontal separator before Camera");
    for removed in [
        "Simulation",
        "GPU rendering",
        "Dark mode",
        "Satellite tiles (Esri)",
        "Earth Observation…",
    ] {
        assert!(!labels.iter().any(|(text, _)| text == removed));
    }
    app.viewer.earth_observation.open = false;
    click_settings(&mut app, &ctx, "Earth Observation");
    assert!(app.viewer.earth_observation.open);
    click_settings(&mut app, &ctx, "None");
    assert!(!app.viewer.earth_observation.open);
}

#[test]
fn clock_is_available_only_below_planet_and_solar_system_display_modes() {
    use crate::config::ViewMode;
    let ctx = egui::Context::default();
    let mut app = app(&ctx);
    for mode in [
        ViewMode::Planet,
        ViewMode::SolarSystem,
        ViewMode::PlanetSizes,
        ViewMode::Satellite,
        ViewMode::SatelliteTable,
        ViewMode::GroundTerminals,
    ] {
        app.viewer.tabs[0].settings.view_mode = mode;
        settings_frame(&mut app, &ctx, Vec::new());
        let labels = settings_frame(&mut app, &ctx, Vec::new());
        let enabled = matches!(mode, ViewMode::Planet | ViewMode::SolarSystem);
        for label in ["Clock", "Speed:", "Time:", "Date:", "Sync time"] {
            assert_eq!(
                labels.iter().any(|(text, _)| text == label),
                enabled,
                "{} has incorrect Clock controls: {label}",
                mode.label()
            );
        }
        assert!(!labels.iter().any(|(text, _)| text == "Simulation"));
        if enabled {
            assert!(rect(&labels, "Clock").top() > rect(&labels, "Display:").bottom());
            assert!(rect(&labels, "Clock").left() > rect(&labels, "Display:").left());
            assert!(rect(&labels, "Speed:").left() > rect(&labels, "Clock").left());
        }
    }
}

#[test]
fn clock_controls_update_only_the_active_view() {
    use crate::config::ViewMode;
    let ctx = egui::Context::default();
    let mut app = app(&ctx);
    app.viewer.tabs[0].settings.time = 100.0;
    app.viewer.tabs[0].settings.speed = 2.0;
    app.viewer.tabs.push(TabConfig::new("Other view".into()));
    app.viewer.active_tab_idx = 1;
    app.viewer.real_time = 12345.0;
    for mode in [ViewMode::Planet, ViewMode::SolarSystem] {
        app.viewer.tabs[1].settings.view_mode = mode;
        app.viewer.tabs[1].settings.speed = 3.0;
        app.viewer.tabs[1].settings.time = -10.0;
        click_settings(&mut app, &ctx, "⏪");
        assert_eq!(app.viewer.tabs[1].settings.speed, -3.0);
        click_settings(&mut app, &ctx, "Sync time");
        assert_eq!(app.viewer.tabs[1].settings.time, 12345.0);
        assert_eq!(app.viewer.tabs[0].settings.speed, 2.0);
        assert_eq!(app.viewer.tabs[0].settings.time, 100.0);
    }
}

#[test]
fn camera_fields_and_direction_buttons_use_separate_rows() {
    let ctx = egui::Context::default();
    let mut app = app(&ctx);
    settings_frame(&mut app, &ctx, Vec::new());
    let labels = settings_frame(&mut app, &ctx, Vec::new());
    for pair in ["Lat:", "Lon:", "Alt:"].windows(2) {
        assert!(rect(&labels, pair[0]).bottom() < rect(&labels, pair[1]).top());
        assert_eq!(rect(&labels, pair[0]).left(), rect(&labels, pair[1]).left());
    }
    for direction in ["N", "S", "E", "W", "C"] {
        assert_eq!(
            rect(&labels, direction).center().y,
            rect(&labels, "N").center().y
        );
    }
    assert!(rect(&labels, "Day").top() > rect(&labels, "N").bottom());
    assert_eq!(
        rect(&labels, "Day").center().y,
        rect(&labels, "Night").center().y
    );
    click_settings(&mut app, &ctx, "N");
    let (latitude, _) = matrix_to_lat_lon(&app.viewer.tabs[0].settings.rotation);
    assert!((latitude.to_degrees() - 90.0).abs() < 1e-6);
    click_settings(&mut app, &ctx, "E");
    let (latitude, longitude) = matrix_to_lat_lon(&app.viewer.tabs[0].settings.rotation);
    assert!(latitude.abs() < 1e-6);
    assert!((longitude.to_degrees() - 90.0).abs() < 1e-6);
}

#[test]
fn planet_sections_are_not_shown_in_other_display_modes() {
    use crate::config::ViewMode;
    let ctx = egui::Context::default();
    let mut app = app(&ctx);
    for mode in [
        ViewMode::SolarSystem,
        ViewMode::PlanetSizes,
        ViewMode::Satellite,
        ViewMode::SatelliteTable,
        ViewMode::GroundTerminals,
    ] {
        app.viewer.tabs[0].settings.view_mode = mode;
        settings_frame(&mut app, &ctx, Vec::new());
        let labels = settings_frame(&mut app, &ctx, Vec::new());
        assert!(labels.iter().any(|(text, _)| text == "Display:"));
        for section in [
            "Rendering",
            "Application",
            "Camera",
            "Constellations",
            "ESRI Tiles",
            "Texture:",
        ] {
            assert!(
                !labels.iter().any(|(text, _)| text == section),
                "{} must not show the Planet section {section}",
                mode.label()
            );
        }
    }
}

#[test]
fn earth_observation_cannot_be_enabled_with_multiple_planets_or_another_body() {
    let ctx = egui::Context::default();
    let mut app = app(&ctx);
    app.viewer.tabs[0].add_planet();
    click_settings(&mut app, &ctx, "Earth Observation");
    assert!(!app.viewer.earth_observation.open);
    app.viewer.tabs[0].planets.pop();
    click_settings(&mut app, &ctx, "Earth Observation");
    assert!(app.viewer.earth_observation.open);
    app.viewer.tabs[0].planets[0].celestial_body = CelestialBody::Mars;
    click_settings(&mut app, &ctx, "Earth Observation");
    assert!(!app.viewer.earth_observation.open);
}

#[test]
fn esri_control_uses_the_active_tab_and_preserves_texture_fallback() {
    let ctx = egui::Context::default();
    let mut app = app(&ctx);
    assert!(
        app.viewer.tile_overlay.enabled,
        "ESRI imagery is enabled by default"
    );
    app.viewer.tile_overlay.enabled = false;
    let mut mars = TabConfig::new("Mars".into());
    mars.planets[0].celestial_body = CelestialBody::Mars;
    app.viewer.tabs.push(mars);
    app.viewer.active_tab_idx = 1;
    click_settings(&mut app, &ctx, "ESRI Tiles");
    assert!(!app.viewer.tile_overlay.enabled);
    click_settings(&mut app, &ctx, "2K");
    click_settings(&mut app, &ctx, "4K");
    assert_eq!(app.viewer.texture_resolution, TextureResolution::R4096);
    app.viewer.active_tab_idx = 0;
    click_settings(&mut app, &ctx, "ESRI Tiles");
    assert!(app.viewer.tile_overlay.enabled);
    click_settings(&mut app, &ctx, "Orthographic");
    click_settings(&mut app, &ctx, "Mercator");
    assert!(app.viewer.tabs[0].settings.planet_projection == ProjectionKind::Mercator);
    assert_eq!(app.viewer.texture_resolution, TextureResolution::R4096);
}

#[test]
fn theme_button_alternates_the_global_theme() {
    let ctx = egui::Context::default();
    egui_extras::install_image_loaders(&ctx);
    let mut app = app(&ctx);
    let mut button = egui::Rect::NOTHING;
    let mut frame = |app: &mut App, events| {
        let _ = ctx.run_ui(
            egui::RawInput {
                events,
                ..Default::default()
            },
            |ui| {
                button = app.viewer.theme_toggle(ui).rect;
            },
        );
        button.center()
    };
    assert!(app.viewer.dark_mode);
    for expected in [false, true] {
        let pos = frame(&mut app, Vec::new());
        for pressed in [true, false] {
            frame(
                &mut app,
                vec![
                    egui::Event::PointerMoved(pos),
                    egui::Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: Default::default(),
                    },
                ],
            );
        }
        assert_eq!(app.viewer.dark_mode, expected);
    }
}

#[test]
#[ignore = "requires a native GPU adapter; verifies the actual toolbar, tabs menu and settings layout"]
#[allow(deprecated)] // Exercise the same eframe update and panels as the application.
fn reorganized_settings_render_and_the_tabs_menu_controls_auto_cycling() {
    use crate::satellite_view::gpu_tests::{paint_frame, render_state};
    use eframe::App as _;
    let render = render_state();
    let ctx = egui::Context::default();
    let mut creation = eframe::CreationContext::_new_kittest(ctx.clone());
    creation.wgpu_render_state = Some(render.clone());
    let mut app = App::new(&creation);
    let mut native_frame = eframe::Frame::_new_kittest();
    app.viewer.earth_observation.open = false;
    app.viewer.tile_overlay.enabled = false;
    app.viewer.tabs[0].settings.animate = false;
    let mut draw = |app: &mut App, size: [u32; 2], events| {
        let output = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(size[0] as f32, size[1] as f32),
                )),
                events,
                ..Default::default()
            },
            |ctx| app.update(ctx, &mut native_frame),
        );
        let labels = labels(&output.shapes);
        let icons: Vec<_> = {
            let textures = ctx.tex_manager();
            let textures = textures.read();
            output
                .shapes
                .iter()
                .filter_map(|shape| {
                    let egui::Shape::Rect(rect) = &shape.shape else {
                        return None;
                    };
                    let meta = textures.meta(rect.brush.as_ref()?.fill_texture_id)?;
                    [
                        "lucide-sun.svg",
                        "lucide-moon.svg",
                        "lucide-presentation.svg",
                        "lucide-panel-left-open.svg",
                        "lucide-panel-left-close.svg",
                    ]
                    .iter()
                    .any(|name| meta.name.contains(name))
                    .then(|| (meta.name.clone(), rect.rect))
                })
                .collect()
        };
        if labels.iter().any(|(text, _)| text == "?") {
            let toolbar_points: Vec<_> = ["?", "View 1", "⏭"]
                .iter()
                .map(|name| {
                    labels
                        .iter()
                        .find(|(text, _)| text == name)
                        .expect("toolbar label")
                        .1
                        .center()
                })
                .chain(icons.iter().map(|(_, rect)| rect.center()))
                .collect();
            assert_eq!(toolbar_points.len(), 6);
            let mut height: Option<f32> = None;
            for point in toolbar_points {
                let button = output
                    .shapes
                    .iter()
                    .filter_map(|shape| match &shape.shape {
                        egui::Shape::Rect(rect)
                            if rect.brush.is_none()
                                && rect.rect.contains(point)
                                && rect.rect.width() < 300.0
                                && rect.rect.height() < 40.0 =>
                        {
                            Some(rect.rect)
                        }
                        _ => None,
                    })
                    .min_by(|a, b| a.area().total_cmp(&b.area()))
                    .expect("toolbar button background");
                if let Some(height) = height {
                    assert!(
                        (button.height() - height).abs() < 0.1,
                        "all top-left buttons must have the same height: {button:?} vs {height}"
                    );
                } else {
                    height = Some(button.height());
                }
            }
        }
        if size[0] >= 1200 && !app.viewer.show_tab_info && !app.viewer.earth_observation.open {
            for caption in ["Stats", "Ground", "Space", "Live", "Conj", "Rad", "Moons"] {
                let center = rect(&labels, caption).center();
                assert!(
                    output.shapes.iter().any(|shape| matches!(
                        &shape.shape,
                        egui::Shape::Rect(background)
                            if background.brush.is_none()
                                && background.rect.contains(center)
                                && background.rect.height() < 40.0
                                && background.rect.width() < 100.0
                                && background.fill != egui::Color32::TRANSPARENT
                    )),
                    "{caption} must have a button background"
                );
            }
        }
        (labels, icons, paint_frame(&ctx, &render, size, output))
    };
    let size = [1600, 1000];
    draw(&mut app, size, Vec::new());
    let (labels, icons, image) = draw(&mut app, size, Vec::new());
    assert_eq!(icons.len(), 3);
    assert!(!labels.iter().any(|(text, _)| text == "Sim"));
    assert!(icons[0].1.left() > rect(&labels, "?").right());
    assert!(!labels.iter().any(|(text, _)| [
        "LeoDOS Visualiser",
        "FPS",
        "Dark mode",
        "GPU rendering",
        "Auto cycle tabs",
        "[?]",
        "[p]",
        "[P]",
        "Default",
        "Abstract",
        "Hell on Earth"
    ]
    .contains(&text.as_str())));
    image
        .save(crate::texture::asset_path(
            "target/settings-reorganized-dark.png",
        ))
        .unwrap();
    let theme = icons[0].1.center();
    for pressed in [true, false] {
        draw(
            &mut app,
            size,
            vec![
                egui::Event::PointerMoved(theme),
                egui::Event::PointerButton {
                    pos: theme,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Default::default(),
                },
            ],
        );
    }
    assert!(!app.viewer.dark_mode);
    let (labels, icons, image) = draw(&mut app, size, Vec::new());
    assert!(icons[0].0.contains("lucide-sun.svg"));
    image
        .save(crate::texture::asset_path(
            "target/settings-reorganized-light.png",
        ))
        .unwrap();
    let tabs = labels
        .iter()
        .find(|(text, _)| text == "View 1")
        .unwrap()
        .1
        .center();
    for pressed in [true, false] {
        draw(
            &mut app,
            size,
            vec![
                egui::Event::PointerMoved(tabs),
                egui::Event::PointerButton {
                    pos: tabs,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Default::default(),
                },
            ],
        );
    }
    for _ in 0..10 {
        draw(&mut app, size, Vec::new());
    }
    let (labels, _, image) = draw(&mut app, size, Vec::new());
    assert!(rect(&labels, "Auto cycle tabs").bottom() <= rect(&labels, "Cycle interval:").top());
    image
        .save(crate::texture::asset_path("target/settings-tabs-menu.png"))
        .unwrap();
    let cycle = rect(&labels, "Auto cycle tabs").center();
    for pressed in [true, false] {
        draw(
            &mut app,
            size,
            vec![
                egui::Event::PointerMoved(cycle),
                egui::Event::PointerButton {
                    pos: cycle,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Default::default(),
                },
            ],
        );
    }
    assert!(app.viewer.auto_cycle_tabs);
    ctx.memory_mut(|memory| memory.close_all_popups());
    let (_, _, image) = draw(&mut app, [600, 900], vec![egui::Event::PointerGone]);
    image
        .save(crate::texture::asset_path(
            "target/settings-reorganized-narrow.png",
        ))
        .unwrap();

    let (_, icons, _) = draw(&mut app, size, Vec::new());
    let presentation = icons
        .iter()
        .find(|(name, _)| name.contains("lucide-presentation"))
        .unwrap()
        .1
        .center();
    for pressed in [true, false] {
        draw(
            &mut app,
            size,
            vec![
                egui::Event::PointerMoved(presentation),
                egui::Event::PointerButton {
                    pos: presentation,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Default::default(),
                },
            ],
        );
    }
    assert!(app.viewer.show_tab_info);
    let (labels, icons, _) = draw(&mut app, size, Vec::new());
    assert!(!labels.iter().any(|(text, _)| text == "Clock"));
    let presentation = icons
        .iter()
        .find(|(name, _)| name.contains("lucide-presentation"))
        .unwrap()
        .1
        .center();
    for pressed in [true, false] {
        draw(
            &mut app,
            size,
            vec![
                egui::Event::PointerMoved(presentation),
                egui::Event::PointerButton {
                    pos: presentation,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Default::default(),
                },
            ],
        );
    }
    assert!(!app.viewer.show_tab_info);
    let (labels, icons, _) = draw(&mut app, size, Vec::new());
    assert!(labels.iter().any(|(text, _)| text == "Clock"));
    let close = icons
        .iter()
        .find(|(name, _)| name.contains("lucide-panel-left-close"))
        .unwrap()
        .1
        .center();
    for pressed in [true, false] {
        draw(
            &mut app,
            size,
            vec![
                egui::Event::PointerMoved(close),
                egui::Event::PointerButton {
                    pos: close,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Default::default(),
                },
            ],
        );
    }
    assert!(!app.viewer.show_side_panel);
    draw(&mut app, size, Vec::new()); // Settle the floating control's initial sizing pass.
    let (labels, icons, image) = draw(&mut app, size, Vec::new());
    assert_eq!(icons.len(), 1);
    assert!(icons[0].0.contains("lucide-panel-left-open"));
    assert!(
        icons[0].1.expand(5.0).right() <= rect(&labels, "Planet 1").left(),
        "the sidebar control must not cover the planet title"
    );
    assert!(
        rect(&labels, "[+] Add constellation").left() <= 12.0,
        "the collapsed settings must not leave a gutter beside the canvas: {:?}",
        rect(&labels, "[+] Add constellation")
    );
    image
        .save(crate::texture::asset_path(
            "target/settings-collapsed-sidebar.png",
        ))
        .unwrap();
    let (labels, icons, image) = draw(&mut app, [600, 900], Vec::new());
    assert!(icons[0].1.expand(5.0).right() <= rect(&labels, "Planet 1").left());
    image
        .save(crate::texture::asset_path(
            "target/settings-collapsed-sidebar-narrow.png",
        ))
        .unwrap();
    let open = icons[0].1.center();
    for pressed in [true, false] {
        draw(
            &mut app,
            [600, 900],
            vec![
                egui::Event::PointerMoved(open),
                egui::Event::PointerButton {
                    pos: open,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Default::default(),
                },
            ],
        );
    }
    assert!(app.viewer.show_side_panel);
    let (labels, _, _) = draw(&mut app, [600, 900], Vec::new());
    assert!(labels.iter().any(|(text, _)| text == "Clock"));
}
