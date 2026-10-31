use super::*;

fn frame(
    view: &mut View,
    ctx: &egui::Context,
    size: [u32; 2],
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(size[0] as f32, size[1] as f32),
            )),
            events,
            ..Default::default()
        },
        |ui| {
            egui::CentralPanel::default().show_inside(ui, |ui| view.show(ui));
        },
    )
}

fn label(output: &egui::FullOutput, text: &str) -> egui::Rect {
    output
        .shapes
        .iter()
        .rev()
        .find_map(|shape| match &shape.shape {
            egui::Shape::Text(label) if label.galley.text() == text => {
                Some(shape.shape.visual_bounding_rect())
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("Missing label: {text}"))
}

fn click(view: &mut View, ctx: &egui::Context, size: [u32; 2], position: egui::Pos2) {
    frame(view, ctx, size, vec![egui::Event::PointerMoved(position)]);
    for pressed in [true, false] {
        frame(
            view,
            ctx,
            size,
            vec![egui::Event::PointerButton {
                pos: position,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: Default::default(),
            }],
        );
    }
}

#[test]
fn controls_update_the_band_estimates_sorting_and_selected_chart() {
    let ctx = egui::Context::default();
    egui_extras::install_image_loaders(&ctx);
    let mut view = View::default();
    let size = [1440, 1200];
    view.sort = Sort {
        field: Field::Terminal,
        descending: false,
    };
    frame(&mut view, &ctx, size, Vec::new());
    let output = frame(&mut view, &ctx, size, Vec::new());
    let point = label(&output, TERMINALS[0].name).center();
    assert!(view.selected[0]);
    click(&mut view, &ctx, size, point);
    assert!(
        !view.selected[0],
        "the terminal checkbox controls its plotted series"
    );

    let output = frame(&mut view, &ctx, size, Vec::new());
    click(
        &mut view,
        &ctx,
        size,
        label(&output, Field::Data.label()).center(),
    );
    assert!(view.sort.field == Field::Data && view.sort.descending);
    let output = frame(&mut view, &ctx, size, Vec::new());
    click(
        &mut view,
        &ctx,
        size,
        label(&output, Field::Data.label()).center(),
    );
    assert!(!view.sort.descending);

    let output = frame(&mut view, &ctx, size, Vec::new());
    click(&mut view, &ctx, size, label(&output, "Ka").center());
    assert_eq!(view.scenario.band, Band::Ka);
    assert_eq!(view.scenario.frequency_ghz, 26.25);
    assert_eq!(view.scenario.bandwidth_mhz, 500.0);
    assert_eq!(
        view.comparison
            .as_ref()
            .unwrap()
            .rows
            .iter()
            .filter(|row| row.is_err())
            .count(),
        3
    );
}

#[test]
fn cached_estimates_refresh_when_any_shared_assumption_changes() {
    let ctx = egui::Context::default();
    let mut view = View::default();
    let size = [1024, 768];
    frame(&mut view, &ctx, size, Vec::new());
    let old = view.comparison.as_ref().unwrap().rows[0]
        .as_ref()
        .unwrap()
        .data_gb;
    view.scenario.eirp_dbw -= 10.0;
    frame(&mut view, &ctx, size, Vec::new());
    assert!(
        view.comparison.as_ref().unwrap().rows[0]
            .as_ref()
            .unwrap()
            .data_gb
            < old
    );
}

#[test]
fn legend_selection_and_table_selection_agree() {
    let ctx = egui::Context::default();
    let size = [1440, 1200];
    let mut view = View::default();
    view.sort = Sort {
        field: Field::Terminal,
        descending: false,
    };
    frame(&mut view, &ctx, size, Vec::new());
    let output = frame(&mut view, &ctx, size, Vec::new());
    let legend = output
        .shapes
        .iter()
        .find_map(|shape| match &shape.shape {
            egui::Shape::Text(text) if text.galley.text() == TERMINALS[0].name => {
                Some(shape.shape.visual_bounding_rect())
            }
            _ => None,
        })
        .unwrap();
    click(&mut view, &ctx, size, legend.center());
    assert!(!view.selected[0]);
    let output = frame(&mut view, &ctx, size, Vec::new());
    click(
        &mut view,
        &ctx,
        size,
        label(&output, TERMINALS[0].name).center(),
    );
    assert!(
        view.selected[0],
        "table selection must be able to restore a legend-hidden trace"
    );
}

#[test]
#[ignore = "requires a native GPU; renders ground-terminal layouts and checks text bounds"]
fn ground_terminal_layouts_render_without_horizontal_overflow() {
    use crate::satellite_view::gpu_tests::{paint_frame, render_state};
    let render = render_state();
    for (name, size, dark, advanced) in [
        ("light", [1440, 1100], false, false),
        ("dark", [1440, 1100], true, false),
        ("medium", [768, 1100], false, true),
        ("narrow", [390, 1000], false, true),
        ("all", [1440, 1200], false, false),
        ("large-text", [390, 1100], false, false),
    ] {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        ctx.set_visuals(if dark {
            egui::Visuals::dark()
        } else {
            egui::Visuals::light()
        });
        let mut view = View::default();
        if name == "all" {
            view.selected.fill(true);
        }
        if name == "large-text" {
            ctx.global_style_mut(|style| {
                style
                    .text_styles
                    .insert(egui::TextStyle::Body, egui::FontId::proportional(16.0));
                style
                    .text_styles
                    .insert(egui::TextStyle::Button, egui::FontId::proportional(16.0));
            });
        }
        for _ in 0..3 {
            let output = frame(&mut view, &ctx, size, Vec::new());
            paint_frame(&ctx, &render, size, output);
        }
        if advanced {
            let output = frame(&mut view, &ctx, size, Vec::new());
            let point = label(&output, "Link assumptions").center();
            paint_frame(&ctx, &render, size, output);
            // Paint every event frame so egui's texture deltas are applied.
            for event in [
                egui::Event::PointerMoved(point),
                egui::Event::PointerButton {
                    pos: point,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: Default::default(),
                },
                egui::Event::PointerButton {
                    pos: point,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Default::default(),
                },
            ] {
                let output = frame(&mut view, &ctx, size, vec![event]);
                paint_frame(&ctx, &render, size, output);
            }
        }
        for _ in 0..5 {
            let output = frame(&mut view, &ctx, size, Vec::new());
            paint_frame(&ctx, &render, size, output);
        }
        let output = frame(&mut view, &ctx, size, Vec::new());
        let viewport =
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(size[0] as f32, size[1] as f32));
        for shape in &output.shapes {
            if let egui::Shape::Text(text) = &shape.shape {
                let bounds = shape.shape.visual_bounding_rect();
                if bounds.intersects(viewport) && bounds.intersects(shape.clip_rect) {
                    assert!(
                        !text.galley.elided,
                        "{name}: shortened text {:?}",
                        text.galley.text()
                    );
                    assert!(
                        bounds.right() <= shape.clip_rect.right() + 2.0,
                        "{name}: clipped text {:?}, {bounds:?} vs {:?}",
                        text.galley.text(),
                        shape.clip_rect
                    );
                    assert!(
                        bounds.right() <= size[0] as f32,
                        "{name}: offscreen text {:?}",
                        text.galley.text()
                    );
                }
            }
        }
        let image = paint_frame(&ctx, &render, size, output);
        let non_background = image
            .pixels()
            .filter(|pixel| pixel[0].abs_diff(pixel[1]) > 30 || pixel[1].abs_diff(pixel[2]) > 30)
            .count();
        assert!(
            non_background > 500,
            "chart and selected series must be visible"
        );
        image
            .save(crate::texture::asset_path(&format!(
                "target/ground-terminals-{name}.png"
            )))
            .unwrap();
    }
}
