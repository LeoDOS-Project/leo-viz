use super::*;

struct CapturedFrame {
    image: image::RgbaImage,
    scene: egui::Rect,
    preview: Option<egui::Rect>,
    labels: Vec<(String, egui::Rect)>,
    text_lines: Vec<(String, usize)>,
    text_fonts: Vec<(String, egui::FontId)>,
}

impl CapturedFrame {
    fn label(&self, text: &str) -> egui::Rect {
        self.labels
            .iter()
            .find(|(label, _)| label == text)
            .unwrap_or_else(|| panic!("label {text}"))
            .1
    }
}

pub(crate) fn render_state() -> RenderState {
    let instance = wgpu::Instance::new(&Default::default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let renderer =
        egui_wgpu::Renderer::new(&device, wgpu::TextureFormat::Rgba8Unorm, Default::default());
    RenderState {
        adapter,
        available_adapters: Vec::new(),
        device,
        queue,
        target_format: wgpu::TextureFormat::Rgba8Unorm,
        renderer: Arc::new(egui::mutex::RwLock::new(renderer)),
    }
}

fn pixels(render: &RenderState, texture: &wgpu::Texture) -> image::RgbaImage {
    let (width, height) = (texture.width(), texture.height());
    let stride = (width * 4).div_ceil(256) * 256;
    let buffer = render.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(stride) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = render.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    render.queue.submit([encoder.finish()]);
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    render
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let mapped = buffer.slice(..).get_mapped_range();
    let bytes = mapped
        .chunks(stride as usize)
        .flat_map(|row| row[..width as usize * 4].iter().copied())
        .collect();
    drop(mapped);
    buffer.unmap();
    image::RgbaImage::from_raw(width, height, bytes).unwrap()
}

fn capture(
    view: &mut View,
    ctx: &egui::Context,
    render: &RenderState,
    size: [u32; 2],
    dark: bool,
    events: Vec<egui::Event>,
) -> image::RgbaImage {
    capture_frame(view, ctx, render, size, dark, events).image
}

fn capture_frame(
    view: &mut View,
    ctx: &egui::Context,
    render: &RenderState,
    size: [u32; 2],
    dark: bool,
    events: Vec<egui::Event>,
) -> CapturedFrame {
    ctx.set_visuals(if dark {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    });
    let output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(size[0] as f32, size[1] as f32),
            )),
            events,
            ..Default::default()
        },
        |ui| {
            egui::CentralPanel::default()
                .show_inside(ui, |ui| view.show_model(ui, Some(render), None));
        },
    );
    let scene_id = view.renderer.as_ref().unwrap().target.as_ref().unwrap().id;
    let scene = output
        .shapes
        .iter()
        .find_map(|shape| match &shape.shape {
            egui::Shape::Mesh(mesh) if mesh.texture_id == scene_id => Some(mesh.calc_bounds()),
            _ => None,
        })
        .expect("satellite scene image");
    let preview_id = view
        .preview_renderer
        .as_ref()
        .and_then(|r| r.target.as_ref())
        .map(|t| t.id);
    let preview = output.shapes.iter().find_map(|shape| match &shape.shape {
        egui::Shape::Mesh(mesh) if Some(mesh.texture_id) == preview_id => Some(mesh.calc_bounds()),
        _ => None,
    });
    let labels = output
        .shapes
        .iter()
        .filter_map(|shape| match &shape.shape {
            egui::Shape::Text(text) => Some((
                text.galley.text().to_owned(),
                shape.shape.visual_bounding_rect(),
            )),
            _ => None,
        })
        .collect();
    let text_lines = output
        .shapes
        .iter()
        .filter_map(|shape| match &shape.shape {
            egui::Shape::Text(text) => {
                Some((text.galley.text().to_owned(), text.galley.rows.len()))
            }
            _ => None,
        })
        .collect();
    let text_fonts = output
        .shapes
        .iter()
        .filter_map(|shape| match &shape.shape {
            egui::Shape::Text(text) => text.galley.job.sections.first().map(|section| {
                (
                    text.galley.text().to_owned(),
                    section.format.font_id.clone(),
                )
            }),
            _ => None,
        })
        .collect();
    CapturedFrame {
        image: paint_frame(ctx, render, size, output),
        scene,
        preview,
        labels,
        text_lines,
        text_fonts,
    }
}

pub(crate) fn paint_frame(
    ctx: &egui::Context,
    render: &RenderState,
    size: [u32; 2],
    output: egui::FullOutput,
) -> image::RgbaImage {
    let jobs = ctx.tessellate(output.shapes, output.pixels_per_point);
    let screen = egui_wgpu::ScreenDescriptor {
        size_in_pixels: size,
        pixels_per_point: output.pixels_per_point,
    };
    let target = render.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("satellite_ui_test"),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: render.target_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let mut encoder = render.device.create_command_encoder(&Default::default());
    {
        let mut renderer = render.renderer.write();
        for (id, delta) in &output.textures_delta.set {
            renderer.update_texture(&render.device, &render.queue, *id, delta);
        }
        let commands =
            renderer.update_buffers(&render.device, &render.queue, &mut encoder, &jobs, &screen);
        render.queue.submit(commands);
        let color_view = target.create_view(&Default::default());
        let mut pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            })
            .forget_lifetime();
        renderer.render(&mut pass, &jobs, &screen);
    }
    render.queue.submit([encoder.finish()]);
    for id in output.textures_delta.free {
        render.renderer.write().free_texture(&id);
    }
    pixels(render, &target)
}

struct NavigationFrame {
    image: image::RgbaImage,
    labels: Vec<(String, egui::Rect)>,
    icons: Vec<(String, egui::Rect)>,
    scenes: Vec<egui::Rect>,
    back: bool,
}

fn navigation_frame(
    view: &mut View,
    ctx: &egui::Context,
    render: &RenderState,
    size: [u32; 2],
    dark: bool,
    events: Vec<egui::Event>,
) -> NavigationFrame {
    ctx.set_visuals(if dark {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    });
    let mut back = false;
    let output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(size[0] as f32, size[1] as f32),
            )),
            events,
            ..Default::default()
        },
        |ui| {
            egui::CentralPanel::default().show_inside(ui, |ui| {
                back |= view.show(ui, Some(render));
            });
        },
    );
    let labels = output
        .shapes
        .iter()
        .filter_map(|shape| match &shape.shape {
            egui::Shape::Text(text) => Some((
                text.galley.text().to_owned(),
                shape.shape.visual_bounding_rect(),
            )),
            _ => None,
        })
        .collect();
    let ids: Vec<_> = view
        .model_panes
        .iter()
        .filter_map(|pane| pane.view.renderer.as_ref())
        .chain(view.renderer.iter())
        .filter_map(|renderer| renderer.target.as_ref().map(|target| target.id))
        .collect();
    let icons = {
        let textures = ctx.tex_manager();
        let textures = textures.read();
        output
            .shapes
            .iter()
            .filter_map(|shape| {
                let (texture, rect) = match &shape.shape {
                    egui::Shape::Mesh(mesh) => (mesh.texture_id, mesh.calc_bounds()),
                    egui::Shape::Rect(rect) => (rect.brush.as_ref()?.fill_texture_id, rect.rect),
                    _ => return None,
                };
                textures
                    .meta(texture)
                    .filter(|meta| meta.name.contains("lucide-chevron-"))
                    .map(|meta| (meta.name.clone(), rect))
            })
            .collect()
    };
    let scenes = output
        .shapes
        .iter()
        .filter_map(|shape| match &shape.shape {
            egui::Shape::Mesh(mesh) if ids.contains(&mesh.texture_id) => Some(mesh.calc_bounds()),
            _ => None,
        })
        .collect();
    NavigationFrame {
        image: paint_frame(ctx, render, size, output),
        labels,
        icons,
        scenes,
        back,
    }
}

#[test]
#[ignore = "requires a native GPU adapter; clicks previous/next icons in single and split satellite views"]
fn satellite_switch_buttons_wrap_and_only_change_their_own_pane() {
    let render = render_state();
    let ctx = egui::Context::default();
    egui_extras::install_image_loaders(&ctx);
    let satellites: Vec<_> = crate::satellite_catalogue::catalogue()
        .satellites
        .iter()
        .filter(|satellite| satellite.model_index().is_some())
        .collect();
    let first = satellites[0];
    let last = satellites[satellites.len() - 1];
    let mut view = loaded(first.model_index().unwrap());
    view.catalogue_selection = Some(first.id);
    let last_asset = last.model_index().unwrap();
    view.models
        .insert(last_asset, loaded(last_asset).model.unwrap());
    let size = [1440, 1000];
    let draw = |view: &mut View, events| navigation_frame(view, &ctx, &render, size, false, events);
    let click = |view: &mut View, point| {
        draw(view, vec![egui::Event::PointerMoved(point)]);
        for pressed in [true, false] {
            draw(
                view,
                vec![
                    egui::Event::PointerMoved(point),
                    egui::Event::PointerButton {
                        pos: point,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: Default::default(),
                    },
                ],
            );
        }
        draw(view, Vec::new())
    };
    let icon = |frame: &NavigationFrame, side: &str, pane| {
        frame
            .icons
            .iter()
            .filter(|(name, _)| name.contains(side))
            .nth(pane)
            .unwrap_or_else(|| panic!("missing {side} icon in pane {pane}: {:?}", frame.icons))
            .1
            .center()
    };
    draw(&mut view, Vec::new());
    let frame = draw(&mut view, Vec::new());
    let frame = click(&mut view, icon(&frame, "chevron-left", 0));
    assert_eq!(view.model_panes[0].view.catalogue_selection, Some(last.id));
    assert_eq!(view.model_panes[0].view.loaded_asset, last_asset);
    let frame = click(&mut view, icon(&frame, "chevron-right", 0));
    assert_eq!(view.model_panes[0].view.catalogue_selection, Some(first.id));
    let plus = frame
        .labels
        .iter()
        .find(|(name, _)| name == "+")
        .unwrap()
        .1
        .center();
    let frame = click(&mut view, plus);
    assert_eq!(view.model_panes.len(), 2);
    let second = view.model_panes[1].view.catalogue_selection.unwrap();
    let current = satellites
        .iter()
        .position(|satellite| satellite.id == second)
        .unwrap();
    let previous = satellites[(current + satellites.len() - 1) % satellites.len()];
    let frame = click(&mut view, icon(&frame, "chevron-left", 1));
    assert_eq!(view.model_panes[0].view.catalogue_selection, Some(first.id));
    assert_eq!(
        view.model_panes[1].view.catalogue_selection,
        Some(previous.id)
    );
    frame
        .image
        .save(crate::texture::asset_path(
            "target/satellite-switch-buttons-desktop.png",
        ))
        .unwrap();
    let narrow = navigation_frame(&mut view, &ctx, &render, [390, 844], false, Vec::new());
    assert_eq!(narrow.icons.len(), 4);
    let bounds = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(390.0, 844.0));
    assert!(narrow
        .icons
        .iter()
        .all(|(_, rect)| bounds.contains_rect(*rect)));
    narrow
        .image
        .save(crate::texture::asset_path(
            "target/satellite-switch-buttons-narrow.png",
        ))
        .unwrap();
    let dark = navigation_frame(&mut view, &ctx, &render, size, true, Vec::new());
    for (_, rect) in &dark.icons {
        let visible = (rect.top().ceil() as u32..rect.bottom().floor() as u32)
            .flat_map(|y| {
                (rect.left().ceil() as u32..rect.right().floor() as u32).map(move |x| (x, y))
            })
            .filter(|&(x, y)| {
                dark.image.get_pixel(x, y).0[..3]
                    .iter()
                    .all(|&channel| channel > 100)
            })
            .count();
        assert!(visible > 5, "navigation icons must be visible in dark mode");
    }
    dark.image
        .save(crate::texture::asset_path(
            "target/satellite-switch-buttons-dark.png",
        ))
        .unwrap();
}

#[test]
#[ignore = "requires a native GPU adapter; verifies linked scale, rotation and pinch zoom in the same frame"]
fn satellite_comparison_cameras_move_and_zoom_together() {
    let render = render_state();
    let ctx = egui::Context::default();
    egui_extras::install_image_loaders(&ctx);
    let mut view = loaded(0);
    for asset in [1, 2] {
        view.models.insert(asset, loaded(asset).model.unwrap());
    }
    let size = [1440, 1000];
    let draw =
        |view: &mut View, size, events| navigation_frame(view, &ctx, &render, size, false, events);
    let check = |view: &View, frame: &NavigationFrame| {
        assert_eq!(frame.scenes.len(), view.model_panes.len());
        let first = &view.model_panes[0].view;
        let mut scale: Option<f32> = None;
        for (pane, rect) in view.model_panes.iter().zip(&frame.scenes) {
            let pane = &pane.view;
            let model = pane.model.as_ref().unwrap();
            let units = ASSETS[pane.loaded_asset].metres_per_unit(model);
            let current = pane.camera.metres_per_point(*rect, pane.explosion, units);
            if let Some(scale) = scale {
                assert!(
                    (current - scale).abs() < current * 1e-5,
                    "unequal physical scales: {current} vs {scale}"
                );
            }
            scale = Some(current);
            assert_eq!(pane.camera.yaw, first.camera.yaw);
            assert_eq!(pane.camera.pitch, first.camera.pitch);
            assert!(pane.preview_renderer.is_none());
            let matrix: [[f32; 4]; 4] = pane
                .camera
                .matrices(rect.aspect_ratio(), pane.explosion)
                .0
                .into();
            assert_eq!(
                pane.renderer
                    .as_ref()
                    .unwrap()
                    .last_frame
                    .as_ref()
                    .unwrap()
                    .0
                    .view_projection,
                matrix,
                "every GPU draw must use the final shared camera, not the previous pane's frame"
            );
        }
        scale.unwrap()
    };
    let single = draw(&mut view, size, Vec::new());
    assert_eq!(single.scenes.len(), 1);
    let plus = single
        .labels
        .iter()
        .find(|(text, _)| text == "+")
        .unwrap()
        .1
        .center();
    draw(&mut view, size, vec![egui::Event::PointerMoved(plus)]);
    for pressed in [true, false] {
        draw(
            &mut view,
            size,
            vec![
                egui::Event::PointerMoved(plus),
                egui::Event::PointerButton {
                    pos: plus,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Default::default(),
                },
            ],
        );
    }
    let initial = draw(&mut view, size, Vec::new());
    assert_eq!(initial.scenes.len(), 2);
    let initial_scale = check(&view, &initial);
    let point = initial.scenes[1].center();
    draw(&mut view, size, vec![egui::Event::PointerMoved(point)]);
    let yaw = view.model_panes[0].view.camera.yaw;
    let zoomed = draw(&mut view, size, vec![egui::Event::Zoom(1.25)]);
    let zoomed_scale = check(&view, &zoomed);
    assert!((zoomed_scale - initial_scale / 1.25).abs() < initial_scale * 1e-5);
    assert_eq!(
        view.model_panes[0].view.camera.yaw, yaw,
        "pinching must not rotate"
    );

    let rotated = draw(
        &mut view,
        size,
        vec![
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: egui::Vec2::ZERO,
                phase: egui::TouchPhase::Start,
                modifiers: Default::default(),
            },
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: egui::vec2(12.0, -8.0),
                phase: egui::TouchPhase::Move,
                modifiers: Default::default(),
            },
        ],
    );
    check(&view, &rotated);
    assert_ne!(
        view.model_panes[0].view.camera.yaw, yaw,
        "two-finger movement rotates every pane"
    );

    let button = |pos, pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: Default::default(),
    };
    draw(&mut view, size, vec![button(point, true)]);
    let yaw = view.model_panes[0].view.camera.yaw;
    let dragged_point = point + egui::vec2(26.0, 14.0);
    let dragged = draw(
        &mut view,
        size,
        vec![egui::Event::PointerMoved(dragged_point)],
    );
    check(&view, &dragged);
    assert_ne!(
        view.model_panes[0].view.camera.yaw, yaw,
        "dragging the later pane must update the first in the same frame"
    );
    draw(
        &mut view,
        size,
        vec![button(dragged_point, false), egui::Event::PointerGone],
    );

    view.model_panes[1].view.asset = 2;
    view.model_panes[1].view.catalogue_selection = None;
    let switched = draw(&mut view, size, Vec::new());
    let switched_scale = check(&view, &switched);
    assert!((switched_scale - zoomed_scale).abs() < zoomed_scale * 1e-5);
    let zoom = view.comparison_camera.unwrap().zoom();
    let narrow = draw(&mut view, [390, 844], Vec::new());
    check(&view, &narrow);
    assert!(
        (view.comparison_camera.unwrap().zoom() - zoom).abs() < 1e-5,
        "resizing must preserve the shared zoom while fitting the new viewport"
    );
    narrow
        .image
        .save(crate::texture::asset_path(
            "target/satellite-linked-camera-narrow.png",
        ))
        .unwrap();

    view.toggle_auto_rotate();
    let yaw = view.model_panes[0].view.camera.yaw;
    let spinning = draw(&mut view, size, Vec::new());
    check(&view, &spinning);
    assert!(view.model_panes.iter().all(|pane| pane.view.spin));
    assert!(view.model_panes[0].view.camera.yaw > yaw);
    spinning
        .image
        .save(crate::texture::asset_path(
            "target/satellite-linked-camera-desktop.png",
        ))
        .unwrap();
}

#[test]
#[ignore = "requires a native GPU adapter; exercises model panes, dropdowns and Back navigation"]
fn satellite_model_panes_add_select_remove_and_return_to_the_table() {
    let render = render_state();
    let ctx = egui::Context::default();
    egui_extras::install_image_loaders(&ctx);
    let mut view = loaded(0);
    for index in [1, 2] {
        view.models.insert(index, loaded(index).model.unwrap());
    }
    let size = [1440, 1000];
    navigation_frame(&mut view, &ctx, &render, size, false, Vec::new());
    let first = navigation_frame(&mut view, &ctx, &render, size, false, Vec::new());
    assert_eq!(view.model_panes.len(), 1);
    assert_eq!(first.scenes.len(), 1);
    assert!(view.model_panes[0].view.preview_renderer.is_some());
    assert!(first
        .labels
        .iter()
        .any(|(text, _)| text.starts_with("Components (")));
    assert!(!first
        .labels
        .iter()
        .any(|(text, _)| text == "Back" || text == "No model"));
    let click = |view: &mut View, point| {
        let mut back = navigation_frame(
            view,
            &ctx,
            &render,
            size,
            false,
            vec![egui::Event::PointerMoved(point)],
        )
        .back;
        for pressed in [true, false] {
            back |= navigation_frame(
                view,
                &ctx,
                &render,
                size,
                false,
                vec![
                    egui::Event::PointerMoved(point),
                    egui::Event::PointerButton {
                        pos: point,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: Default::default(),
                    },
                ],
            )
            .back;
        }
        back
    };
    let plus = first
        .labels
        .iter()
        .find(|(text, _)| text == "+")
        .unwrap()
        .1
        .center();
    view.model_panes[0].view.selected = Some(0);
    view.model_panes[0].view.inspected = Some(0);
    click(&mut view, plus);
    assert_eq!(view.model_panes.len(), 2);
    navigation_frame(&mut view, &ctx, &render, size, false, Vec::new());
    let compared = navigation_frame(&mut view, &ctx, &render, size, false, Vec::new());
    assert_eq!(compared.scenes.len(), 2);
    assert!(!compared.scenes[0].intersects(compared.scenes[1]));
    assert!(compared.scenes[0].right() <= compared.scenes[1].left());
    assert!(!compared.labels.iter().any(|(text, _)| text == "Parts"
        || text.starts_with("Components (")
        || text == "Selected part"));
    assert_eq!(
        compared
            .labels
            .iter()
            .filter(|(text, _)| text == "Launched")
            .count(),
        2
    );
    for pane in &view.model_panes {
        assert!(
            pane.view.preview_renderer.is_none(),
            "comparison must not render miniature or component models"
        );
        assert!(
            pane.view.parts_open,
            "comparison must preserve the individual view's preference"
        );
        let image = pixels(
            &render,
            &pane
                .view
                .renderer
                .as_ref()
                .unwrap()
                .target
                .as_ref()
                .unwrap()
                .texture,
        );
        let background = image.get_pixel(0, 0).0;
        assert!(
            image
                .pixels()
                .filter(|pixel| pixel
                    .0
                    .iter()
                    .zip(background)
                    .take(3)
                    .any(|(a, b)| a.abs_diff(b) > 12))
                .count()
                > 100
        );
    }
    compared
        .image
        .save(crate::texture::asset_path(
            "target/satellite-model-panes-desktop.png",
        ))
        .unwrap();
    let selector = compared
        .labels
        .iter()
        .find(|(text, rect)| text == "Aqua" && rect.top() < 45.0)
        .unwrap()
        .1
        .center();
    click(&mut view, selector);
    let menu = navigation_frame(&mut view, &ctx, &render, size, false, Vec::new());
    assert!(!menu
        .labels
        .iter()
        .any(|(text, _)| text == "WorldView-3" || text == "MiRaTa"));
    let landsat = menu
        .labels
        .iter()
        .find(|(text, _)| text == "Landsat-8")
        .unwrap()
        .1
        .center();
    click(&mut view, landsat);
    navigation_frame(&mut view, &ctx, &render, size, false, Vec::new());
    assert_eq!(view.model_panes[0].view.asset, 0);
    assert_eq!(view.model_panes[1].view.asset, 2);
    view.toggle_auto_rotate();
    assert!(view.model_panes.iter().all(|pane| pane.view.spin));
    for (viewport, dark, name) in [([1440, 1000], true, "dark"), ([390, 844], false, "narrow")] {
        navigation_frame(
            &mut view,
            &ctx,
            &render,
            viewport,
            dark,
            vec![egui::Event::PointerGone],
        );
        let frame = navigation_frame(&mut view, &ctx, &render, viewport, dark, Vec::new());
        assert_eq!(frame.scenes.len(), 2);
        for rect in &frame.scenes {
            assert!(
                rect.left() >= 0.0
                    && rect.right() <= viewport[0] as f32
                    && rect.bottom() <= viewport[1] as f32
            );
        }
        frame
            .image
            .save(crate::texture::asset_path(&format!(
                "target/satellite-model-panes-{name}.png"
            )))
            .unwrap();
    }
    view.model_panes[0].view.model_action = Some(navigation::Action::Remove);
    navigation_frame(&mut view, &ctx, &render, size, false, Vec::new());
    assert_eq!(
        view.model_panes.len(),
        2,
        "keep textures alive for the closing frame"
    );
    navigation_frame(&mut view, &ctx, &render, size, false, Vec::new());
    assert_eq!(view.model_panes.len(), 1);
    assert!(view.model_panes[0].view.preview_renderer.is_some());
    view.spin = false;
    view.comparison_search = "Aqua".to_owned();
    view.comparison_sort = sort::Sort {
        field: sort::Field::Mass,
        descending: true,
    };
    comparison_frame(
        &mut view,
        &ctx,
        &render,
        size,
        false,
        vec![egui::Event::PointerGone],
    );
    let table = comparison_frame(&mut view, &ctx, &render, size, false, Vec::new());
    let point = table
        .thumbnails
        .iter()
        .find(|(index, _)| *index == 1)
        .unwrap()
        .1
        .center();
    comparison_frame(
        &mut view,
        &ctx,
        &render,
        size,
        false,
        vec![egui::Event::PointerMoved(point)],
    );
    let mut opened = false;
    for pressed in [true, false] {
        opened |= comparison_frame(
            &mut view,
            &ctx,
            &render,
            size,
            false,
            vec![
                egui::Event::PointerMoved(point),
                egui::Event::PointerButton {
                    pos: point,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Default::default(),
                },
            ],
        )
        .opened;
    }
    assert!(opened);
    navigation_frame(
        &mut view,
        &ctx,
        &render,
        size,
        false,
        vec![egui::Event::PointerGone],
    );
    let model = navigation_frame(&mut view, &ctx, &render, size, false, Vec::new());
    assert_eq!(model.scenes.len(), 1);
    let back = model
        .labels
        .iter()
        .find(|(text, _)| text == "Back")
        .unwrap()
        .1;
    assert!(
        model.scenes[0].contains_rect(back),
        "Back must be inside the model grid"
    );
    model
        .image
        .save(crate::texture::asset_path(
            "target/satellite-model-back.png",
        ))
        .unwrap();
    assert!(click(&mut view, back.center()));
    assert_eq!(view.comparison_search, "Aqua");
    assert_eq!(
        view.comparison_sort,
        sort::Sort {
            field: sort::Field::Mass,
            descending: true
        }
    );
    let returned = comparison_frame(
        &mut view,
        &ctx,
        &render,
        size,
        false,
        vec![egui::Event::PointerGone],
    );
    assert!(returned
        .labels
        .iter()
        .any(|(text, _)| text == "Satellite Table"));
}

#[test]
#[ignore = "requires a native GPU adapter; verifies clickable table headers and sort direction"]
fn satellite_table_columns_sort_when_clicked() {
    let render = render_state();
    let ctx = egui::Context::default();
    egui_extras::install_image_loaders(&ctx);
    let mut view = View {
        comparison_search: "WorldView".to_owned(),
        spin: false,
        ..Default::default()
    };
    let size = [1440, 900];
    comparison_frame(&mut view, &ctx, &render, size, false, Vec::new());
    let frame = comparison_frame(&mut view, &ctx, &render, size, false, Vec::new());
    let click_header = |view: &mut View, title: &str| {
        let frame = comparison_frame(view, &ctx, &render, size, false, Vec::new());
        let point = frame
            .labels
            .iter()
            .find(|(text, _)| text == title)
            .unwrap()
            .1
            .center();
        comparison_frame(
            view,
            &ctx,
            &render,
            size,
            false,
            vec![egui::Event::PointerMoved(point)],
        );
        for pressed in [true, false] {
            comparison_frame(
                view,
                &ctx,
                &render,
                size,
                false,
                vec![
                    egui::Event::PointerMoved(point),
                    egui::Event::PointerButton {
                        pos: point,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: Default::default(),
                    },
                ],
            );
        }
    };
    assert!(frame.labels.iter().any(|(text, _)| text == "Mass"));
    for (header, field, descending) in [
        ("Mass", sort::Field::Mass, false),
        ("Mass", sort::Field::Mass, true),
        ("Launched", sort::Field::Launched, false),
        ("Power", sort::Field::Power, false),
    ] {
        click_header(&mut view, header);
        assert_eq!(view.comparison_sort, sort::Sort { field, descending });
        let frame = comparison_frame(&mut view, &ctx, &render, size, false, Vec::new());
        let mut expected: Vec<_> = crate::satellite_catalogue::catalogue()
            .satellites
            .iter()
            .filter(|satellite| satellite.matches("worldview"))
            .collect();
        view.comparison_sort.apply(&mut expected);
        let first = frame
            .labels
            .iter()
            .filter(|(text, rect)| {
                rect.left() < 200.0 && expected.iter().any(|satellite| satellite.name == *text)
            })
            .min_by(|(_, a), (_, b)| a.top().total_cmp(&b.top()))
            .unwrap();
        assert_eq!(first.0, expected[0].name);
        frame
            .image
            .save(crate::texture::asset_path(&format!(
                "target/satellite-table-sorted-{}-{descending}.png",
                field as usize
            )))
            .unwrap();
    }
}

struct ComparisonFrame {
    image: image::RgbaImage,
    labels: Vec<(String, egui::Rect)>,
    thumbnails: Vec<(usize, egui::Rect)>,
    opened: bool,
    columns: Vec<egui::Rangef>,
}

fn comparison_frame(
    view: &mut View,
    ctx: &egui::Context,
    render: &RenderState,
    size: [u32; 2],
    dark: bool,
    events: Vec<egui::Event>,
) -> ComparisonFrame {
    ctx.set_visuals(if dark {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    });
    let mut opened = false;
    let mut columns = Vec::new();
    let output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(size[0] as f32, size[1] as f32),
            )),
            events,
            ..Default::default()
        },
        |ui| {
            egui::CentralPanel::default().show_inside(ui, |ui| {
                let groups = comparison::column_groups(ui.available_width());
                let width =
                    (ui.available_width() - ui.spacing().scroll.allocated_width() - 1.0).max(1.0);
                let widths = comparison::column_widths(width, ui.spacing().item_spacing.x, groups);
                let mut x = ui.available_rect_before_wrap().left();
                columns = widths
                    .into_iter()
                    .map(|width| {
                        let range = egui::Rangef::new(x, x + width);
                        x += width + ui.spacing().item_spacing.x;
                        range
                    })
                    .collect();
                opened |= view.show_comparison(ui, Some(render));
            });
        },
    );
    let labels = output
        .shapes
        .iter()
        .filter_map(|shape| match &shape.shape {
            egui::Shape::Text(text) if !text.galley.text().is_empty() => {
                assert!(
                    !text.galley.elided,
                    "comparison text must wrap rather than truncate: {}",
                    text.galley.text()
                );
                Some((
                    text.galley.text().to_owned(),
                    shape.shape.visual_bounding_rect(),
                ))
            }
            _ => None,
        })
        .collect();
    let thumbnails = output
        .shapes
        .iter()
        .filter_map(|shape| match &shape.shape {
            egui::Shape::Mesh(mesh) => {
                view.comparison_renderers
                    .iter()
                    .find_map(|(&index, renderer)| {
                        (renderer
                            .target
                            .as_ref()
                            .is_some_and(|target| target.id == mesh.texture_id))
                        .then_some((index, mesh.calc_bounds()))
                    })
            }
            _ => None,
        })
        .collect();
    ComparisonFrame {
        image: paint_frame(ctx, render, size, output),
        labels,
        thumbnails,
        opened,
        columns,
    }
}

#[test]
#[ignore = "requires a native GPU adapter; verifies comparison table content, thumbnails, and opening a satellite"]
fn comparison_table_shows_models_and_specifications_in_light_dark_and_narrow_views() {
    let render = render_state();
    let ctx = egui::Context::default();
    egui_extras::install_image_loaders(&ctx);
    let mut view = loaded(0);
    for index in 0..ASSETS.len() {
        view.models.insert(index, loaded(index).model.unwrap());
    }
    for (size, dark, name) in [
        ([2400, 1500], false, "light"),
        ([2400, 1500], true, "dark"),
        ([1440, 900], false, "desktop"),
        ([1024, 900], false, "small-desktop"),
        ([768, 1024], false, "tablet"),
        ([390, 844], false, "narrow"),
    ] {
        comparison_frame(&mut view, &ctx, &render, size, dark, Vec::new());
        let frame = comparison_frame(&mut view, &ctx, &render, size, dark, Vec::new());
        assert!(!frame.opened);
        assert!(frame.labels.iter().any(|(text, _)| text == "No model"));
        for group in comparison::column_groups(size[0] as f32 - 16.0) {
            assert!(
                frame.labels.iter().any(|(text, _)| text == group.label),
                "all headers must be visible without horizontal scrolling"
            );
        }
        let header_top = frame
            .labels
            .iter()
            .find(|(text, _)| text == "Satellite")
            .unwrap()
            .1
            .top();
        for (text, rect) in &frame.labels {
            assert!(
                rect.left() >= 6.0 && rect.right() <= size[0] as f32 - 5.0,
                "{name}: text outside viewport: {text}, {rect:?}"
            );
            if rect.top() >= header_top {
                let column = frame
                    .columns
                    .iter()
                    .find(|column| rect.left() >= column.min - 2.0 && rect.left() < column.max)
                    .unwrap();
                assert!(
                    rect.right() <= column.max + 2.0,
                    "{name}: text overlaps adjacent column: {text}"
                );
            }
        }
        for (index, rect) in &frame.thumbnails {
            assert!(rect.left() >= 0.0 && rect.right() <= size[0] as f32);
            let renderer = &view.comparison_renderers[index];
            assert_eq!(
                renderer.last_frame.as_ref().unwrap().0.options,
                [0.0, -1.0, 0.0, -1.0]
            );
            let image = pixels(&render, &renderer.target.as_ref().unwrap().texture);
            let bg = image.get_pixel(0, 0).0;
            assert!(
                image
                    .pixels()
                    .filter(|pixel| pixel
                        .0
                        .iter()
                        .zip(bg)
                        .take(3)
                        .any(|(a, b)| a.abs_diff(b) > 12))
                    .count()
                    > 50
            );
            if let Some((_, title)) = frame
                .labels
                .iter()
                .find(|(text, _)| text == ASSETS[*index].name)
            {
                assert!(
                    title.bottom() <= rect.top(),
                    "satellite name must be above its thumbnail"
                );
            } else {
                let header_bottom = frame
                    .labels
                    .iter()
                    .find(|(text, _)| text == "Satellite")
                    .unwrap()
                    .1
                    .bottom();
                assert!(
                    rect.top() < header_bottom + 20.0 || rect.top() > size[1] as f32,
                    "fully visible satellite rows must have a name"
                );
            }
        }
        assert!(frame
            .labels
            .iter()
            .any(|(text, _)| text == &crate::satellite_catalogue::catalogue().satellites[0].name));
        frame
            .image
            .save(crate::texture::asset_path(&format!(
                "target/satellite-comparison-{name}.png"
            )))
            .unwrap();
        if name == "narrow" {
            let scroll = |view: &mut View, delta| {
                let mut frame = None;
                for step in 0..12 {
                    let events = if step == 0 {
                        vec![
                            egui::Event::PointerMoved(egui::pos2(100.0, 400.0)),
                            egui::Event::MouseWheel {
                                unit: egui::MouseWheelUnit::Point,
                                delta,
                                phase: egui::TouchPhase::Move,
                                modifiers: Default::default(),
                            },
                        ]
                    } else {
                        Vec::new()
                    };
                    frame = Some(comparison_frame(view, &ctx, &render, size, dark, events));
                }
                frame.unwrap()
            };
            let bottom = scroll(&mut view, egui::vec2(0.0, -1_000_000.0));
            assert!(
                bottom.labels.iter().any(|(text, _)| text
                    == &crate::satellite_catalogue::catalogue()
                        .satellites
                        .last()
                        .unwrap()
                        .name),
                "the last satellite must be reachable by vertical scrolling on a narrow screen"
            );
            bottom
                .image
                .save(crate::texture::asset_path(
                    "target/satellite-comparison-narrow-bottom.png",
                ))
                .unwrap();
            scroll(&mut view, egui::vec2(0.0, 1_000_000.0));
        }
    }
    let size = [1400, 900];
    view.comparison_search = "WorldView-3".to_owned();
    comparison_frame(
        &mut view,
        &ctx,
        &render,
        size,
        false,
        vec![egui::Event::PointerGone],
    );
    let frame = comparison_frame(&mut view, &ctx, &render, size, false, Vec::new());
    assert!(
        frame.thumbnails.is_empty(),
        "no placeholder model or GPU target for a data-only satellite"
    );
    assert!(view.comparison_renderers.is_empty());
    for value in [
        "WorldView-3",
        "Launch mass: 2,800 kg",
        "No model",
        "Maxar data sheet",
    ] {
        assert!(
            frame.labels.iter().any(|(text, _)| text.contains(value)),
            "missing data-only entry field: {value}"
        );
    }
    frame
        .image
        .save(crate::texture::asset_path(
            "target/satellite-comparison-worldview.png",
        ))
        .unwrap();
    view.comparison_search = "not a satellite name".to_owned();
    let empty = comparison_frame(&mut view, &ctx, &render, size, false, Vec::new());
    assert!(empty
        .labels
        .iter()
        .any(|(text, _)| text == "No matching satellites"));
    assert!(empty.thumbnails.is_empty());
    view.comparison_search = "Aqua".to_owned();
    comparison_frame(
        &mut view,
        &ctx,
        &render,
        size,
        false,
        vec![egui::Event::PointerGone],
    );
    let frame = comparison_frame(&mut view, &ctx, &render, size, false, Vec::new());
    let point = frame
        .thumbnails
        .iter()
        .find(|(index, _)| *index == 1)
        .unwrap()
        .1
        .center();
    comparison_frame(
        &mut view,
        &ctx,
        &render,
        size,
        false,
        vec![egui::Event::PointerMoved(point)],
    );
    let mut opened = false;
    for pressed in [true, false] {
        opened |= comparison_frame(
            &mut view,
            &ctx,
            &render,
            size,
            false,
            vec![
                egui::Event::PointerMoved(point),
                egui::Event::PointerButton {
                    pos: point,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Default::default(),
                },
            ],
        )
        .opened;
    }
    assert!(
        opened,
        "clicking a satellite thumbnail must open the inspector"
    );
    assert_eq!(view.asset, 1);
    capture_frame(
        &mut view,
        &ctx,
        &render,
        size,
        false,
        vec![egui::Event::PointerGone],
    );
    assert_eq!(view.loaded_asset, 1);
    assert!(
        view.comparison_renderers.is_empty(),
        "hidden comparison targets must be released"
    );
}

#[test]
#[ignore = "requires a native GPU adapter; verifies model switching while downloads are pending or fail"]
fn model_switches_keep_the_previous_scene_until_the_requested_model_is_ready() {
    let render = render_state();
    let ctx = egui::Context::default();
    egui_extras::install_image_loaders(&ctx);
    let mut view = loaded(0);
    let size = [1200, 850];
    capture_frame(&mut view, &ctx, &render, size, true, Vec::new());
    let before = capture_frame(&mut view, &ctx, &render, size, true, Vec::new());
    let id = view.renderer.as_ref().unwrap().target.as_ref().unwrap().id;
    let original = Arc::clone(view.model.as_ref().unwrap());
    let texture = view
        .renderer
        .as_ref()
        .unwrap()
        .target
        .as_ref()
        .unwrap()
        .texture
        .clone();
    let original_pixels = pixels(&render, &texture);
    let scale = view.camera.metres_per_point(
        before.scene,
        view.explosion,
        ASSETS[0].metres_per_unit(&original),
    );
    let pending: LoadResult = Arc::new(Mutex::new(None));
    view.asset = 1;
    view.pending.insert(1, Arc::clone(&pending));
    for fail in [false, true] {
        if fail {
            *pending.lock().unwrap() = Some(Err("Simulated download failure".to_owned()));
        }
        let frame = capture_frame(&mut view, &ctx, &render, size, true, Vec::new());
        assert!(Arc::ptr_eq(view.model.as_ref().unwrap(), &original));
        assert_eq!(
            view.renderer.as_ref().unwrap().target.as_ref().unwrap().id,
            id
        );
        assert_eq!(frame.scene, before.scene);
        assert_eq!(
            pixels(&render, &texture),
            original_pixels,
            "loading must not replace the model with an empty frame"
        );
        assert!(frame.labels.iter().any(|(text, _)| text
            == if fail {
                "Cannot load Aqua"
            } else {
                "Loading Aqua"
            }));
        frame
            .image
            .save(crate::texture::asset_path(&format!(
                "target/satellite-switch-{fail}.png"
            )))
            .unwrap();
    }
    view.errors.remove(&1);
    let ready: LoadResult = Arc::new(Mutex::new(Some(
        ASSETS[1].load(&std::fs::read(crate::texture::asset_path(ASSETS[1].path)).unwrap()),
    )));
    view.pending.insert(1, ready);
    let after = capture_frame(&mut view, &ctx, &render, size, true, Vec::new());
    assert_eq!(view.loaded_asset, 1);
    assert_eq!(after.scene, before.scene);
    let after_scale = view.camera.metres_per_point(
        after.scene,
        view.explosion,
        ASSETS[1].metres_per_unit(view.model.as_ref().unwrap()),
    );
    assert!((after_scale / scale - 1.0).abs() < 1e-5);
    assert!(view.pending.is_empty());
    view.asset = 0;
    capture_frame(&mut view, &ctx, &render, size, true, Vec::new());
    assert!(Arc::ptr_eq(view.model.as_ref().unwrap(), &original));
    assert!(view.pending.is_empty());
}

#[test]
#[ignore = "requires a native GPU adapter; verifies isolation can be armed with no selected component"]
fn isolation_without_selection_shows_the_whole_satellite_and_stays_armed() {
    let render = render_state();
    let ctx = egui::Context::default();
    egui_extras::install_image_loaders(&ctx);
    let mut view = loaded(0);
    let size = [1200, 850];
    capture_frame(&mut view, &ctx, &render, size, false, Vec::new());
    let initial = capture_frame(&mut view, &ctx, &render, size, false, Vec::new());
    let click = |view: &mut View, point: egui::Pos2| {
        for pressed in [true, false] {
            capture_frame(
                view,
                &ctx,
                &render,
                size,
                false,
                vec![
                    egui::Event::PointerMoved(point),
                    egui::Event::PointerButton {
                        pos: point,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: Default::default(),
                    },
                ],
            );
        }
    };
    click(&mut view, initial.label("Isolate selection").center());
    assert!(view.isolate);
    assert!(view.selected.is_none());
    assert_eq!(
        view.renderer
            .as_ref()
            .unwrap()
            .last_frame
            .as_ref()
            .unwrap()
            .0
            .options[2],
        0.0
    );
    click(&mut view, initial.label("Spacecraft bus").center());
    assert_eq!(view.selected, Some(0));
    assert_eq!(
        view.renderer
            .as_ref()
            .unwrap()
            .last_frame
            .as_ref()
            .unwrap()
            .0
            .options[2],
        1.0
    );
    click(&mut view, initial.scene.left_top() + egui::vec2(4.0, 4.0));
    assert!(view.selected.is_none());
    assert!(
        view.isolate,
        "clearing selection must not turn isolation mode off"
    );
    assert_eq!(
        view.renderer
            .as_ref()
            .unwrap()
            .last_frame
            .as_ref()
            .unwrap()
            .0
            .options[2],
        0.0
    );
}

#[test]
#[ignore = "requires a native GPU adapter; writes hover and component information screenshots"]
fn hovering_highlights_without_selecting_and_clears_on_leaving() {
    let render = render_state();
    let ctx = egui::Context::default();
    egui_extras::install_image_loaders(&ctx);
    let mut view = loaded(0);
    let size = [1000, 700];
    let initial = capture_frame(&mut view, &ctx, &render, size, false, Vec::new());
    assert!(initial.labels.iter().any(|(text, _)| text == "Launched"));
    let scene = initial.scene;
    let read_scene = |view: &View| {
        pixels(
            &render,
            &view
                .renderer
                .as_ref()
                .unwrap()
                .target
                .as_ref()
                .unwrap()
                .texture,
        )
    };
    let base = read_scene(&view);
    let point = scene.center();
    let (origin, direction) = view.camera.ray(scene, point, view.explosion).unwrap();
    let expected = view
        .model
        .as_ref()
        .unwrap()
        .pick(origin, direction, view.explosion, None);
    assert!(expected.is_some());
    capture(
        &mut view,
        &ctx,
        &render,
        size,
        false,
        vec![egui::Event::PointerMoved(point)],
    );
    assert_eq!(view.hovered, expected);
    assert!(view.selected.is_none());
    let highlighted = read_scene(&view);
    assert!(
        base.pixels()
            .zip(highlighted.pixels())
            .filter(|(a, b)| a != b)
            .count()
            > 100
    );
    let query = view.picking.as_ref().unwrap().0;
    let hovered_frame = capture_frame(&mut view, &ctx, &render, size, false, Vec::new());
    assert!(!hovered_frame
        .labels
        .iter()
        .any(|(text, _)| text == "Launched"));
    assert!(view.picking.as_ref().unwrap().0 == query);
    let name = view.model.as_ref().unwrap().components[expected.unwrap()]
        .info
        .title;
    let cursor_label = hovered_frame
        .labels
        .iter()
        .find(|(text, rect)| text == name && scene.contains_rect(*rect))
        .expect("the part under the cursor must have a nearby label")
        .1;
    assert!(cursor_label.center().distance(point) < 150.0);
    assert!(
        !cursor_label.contains(point),
        "the label must not cover the click target"
    );
    let preview = hovered_frame.preview.expect("hovered part preview");
    capture(
        &mut view,
        &ctx,
        &render,
        size,
        false,
        vec![egui::Event::PointerMoved(preview.center())],
    );
    assert_eq!(view.inspected, expected);
    assert!(view.selected.is_none());
    let background_frame = capture_frame(
        &mut view,
        &ctx,
        &render,
        size,
        false,
        vec![egui::Event::PointerMoved(
            scene.left_top() + egui::vec2(4.0, 4.0),
        )],
    );
    assert!(view.hovered.is_none());
    assert!(view.selected.is_none());
    assert_eq!(read_scene(&view), base);
    assert!(background_frame
        .labels
        .iter()
        .any(|(text, _)| text == "Launched"));
    assert!(
        !background_frame.labels.iter().any(|(text, rect)| {
            text == view.model.as_ref().unwrap().components[expected.unwrap()]
                .info
                .title
                && scene.contains_rect(*rect)
        }),
        "the cursor label must disappear when no part is under the pointer"
    );

    let row = initial.label("Spacecraft bus").center();
    capture(
        &mut view,
        &ctx,
        &render,
        size,
        false,
        vec![egui::Event::PointerMoved(row)],
    );
    assert_eq!(view.hovered, Some(0));
    assert!(view.selected.is_none());
    let image = capture(
        &mut view,
        &ctx,
        &render,
        size,
        false,
        vec![egui::Event::PointerMoved(point)],
    );
    image
        .save(crate::texture::asset_path(
            "target/satellite-view-hover.png",
        ))
        .unwrap();
    let hover_pixels = read_scene(&view);
    for pressed in [true, false] {
        capture(
            &mut view,
            &ctx,
            &render,
            size,
            false,
            vec![egui::Event::PointerButton {
                pos: point,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: Default::default(),
            }],
        );
    }
    assert_eq!(view.selected, expected);
    assert_eq!(view.hovered, expected);
    let selected_pixels = read_scene(&view);
    assert_eq!(
        selected_pixels, hover_pixels,
        "clicking must retain the same blue highlight"
    );
    capture(
        &mut view,
        &ctx,
        &render,
        size,
        false,
        vec![egui::Event::PointerGone],
    )
    .save(crate::texture::asset_path(
        "target/satellite-view-part-info.png",
    ))
    .unwrap();
    assert!(view.hovered.is_none());
    assert_eq!(view.selected, expected);
    assert_eq!(read_scene(&view), selected_pixels);

    view.isolate = true;
    capture(
        &mut view,
        &ctx,
        &render,
        size,
        false,
        vec![egui::Event::PointerMoved(if expected == Some(0) {
            initial.label("Solar array").center()
        } else {
            row
        })],
    );
    assert!(view.hovered.is_none());
    assert_eq!(view.selected, expected);
    view.isolate = false;
    view.selected = None;
    view.explosion = 0.8;
    capture(
        &mut view,
        &ctx,
        &render,
        size,
        false,
        vec![egui::Event::PointerMoved(point)],
    );
    let (origin, direction) = view.camera.ray(scene, point, view.explosion).unwrap();
    assert_eq!(
        view.hovered,
        view.model
            .as_ref()
            .unwrap()
            .pick(origin, direction, view.explosion, None)
    );
}

fn loaded(asset: usize) -> View {
    View {
        asset,
        loaded_asset: asset,
        spin: false,
        model: Some(Arc::new(
            ASSETS[asset]
                .load(&std::fs::read(crate::texture::asset_path(ASSETS[asset].path)).unwrap())
                .unwrap(),
        )),
        ..Default::default()
    }
}

fn visible_hit(view: &View, scene: egui::Rect) -> (egui::Pos2, usize) {
    let model = view.model.as_ref().unwrap();
    for (x, y) in std::iter::once((5, 5)).chain((1..10).flat_map(|y| (1..10).map(move |x| (x, y))))
    {
        let point = scene.min + scene.size() * egui::vec2(x as f32 / 10.0, y as f32 / 10.0);
        let (origin, direction) = view.camera.ray(scene, point, view.explosion).unwrap();
        if let Some(part) = model.pick(origin, direction, view.explosion, None) {
            return (point, part);
        }
    }
    panic!("model has no visible pick target");
}

#[test]
#[ignore = "requires a native GPU adapter; writes reviewed Aqua assembly images"]
fn inspect_aqua_components() {
    let render = render_state();
    for name in ["aqua"] {
        let path = crate::texture::asset_path(&format!("assets/satellite-{name}.glb"));
        if !path.exists() {
            continue;
        }
        let bytes = std::fs::read(path).unwrap();
        let model = Arc::new(
            ASSETS[1]
                .load(&bytes)
                .unwrap_or_else(|error| panic!("{name}: {error}")),
        );
        eprintln!(
            "{name}: {} bytes, {} triangles, {} parts, dimensions {:?}",
            bytes.len(),
            model.indices.len() / 3,
            model.parts.len(),
            model.dimensions
        );
        for (i, part) in model.components.iter().enumerate() {
            eprintln!(
                "{i}: {} | {} meshes | center {:?} | size {:?}",
                part.info.title,
                part.meshes.len(),
                part.center,
                part.max - part.min
            );
        }
        let mut renderer = Renderer::new(&render, Arc::clone(&model));
        for (side, component) in model.components.iter().enumerate() {
            let camera = Camera {
                yaw: -std::f32::consts::FRAC_PI_2,
                pitch: -0.15,
                target: component.center,
                ..Camera::default()
            };
            let (matrix, eye) = camera
                .matrices_for_bounds(4.0 / 3.0, (component.max - component.min).norm() * 0.56);
            renderer.draw(
                [1024, 768],
                Uniform {
                    view_projection: matrix.into(),
                    eye: [eye.x, eye.y, eye.z, 1.0],
                    options: [0.0, side as f32, 1.0, -1.0],
                },
                false,
            );
            pixels(&render, &renderer.target.as_ref().unwrap().texture)
                .save(crate::texture::asset_path(&format!(
                    "target/satellite-{name}-candidate-{side}.png"
                )))
                .unwrap();
        }
    }
}

#[test]
#[ignore = "requires a native GPU; reviews newly downloaded source meshes before educational assignments"]
fn inspect_new_satellite_sources() {
    let render = render_state();
    let mut reviews = Vec::new();
    for name in ["oco2", "aura", "gpm", "snpp"] {
        let bytes = std::fs::read(crate::texture::asset_path(&format!(
            "assets/satellite-{name}.glb"
        )))
        .unwrap();
        let source = gltf::Gltf::from_slice_without_validation(&bytes).unwrap();
        let model = Arc::new(Model::load(&bytes).unwrap());
        eprintln!(
            "{name}: {} triangles, {} parts, {} materials, dimensions {:?}",
            model.indices.len() / 3,
            model.parts.len(),
            model.materials.len(),
            model.dimensions
        );
        reviews.push(serde_json::json!({
            "name": name,
            "asset": source.document.as_json().asset,
            "dimensions": model.dimensions.as_slice(),
            "radius": model.radius,
            "parts": model.parts.iter().map(|part| serde_json::json!({
                "name": part.name,
                "material": model.materials[part.material].name,
                "min": part.min.as_slice(),
                "max": part.max.as_slice(),
                "triangles": (part.end - part.start) / 3,
            })).collect::<Vec<_>>(),
        }));
        let mut renderer = Renderer::new(&render, Arc::clone(&model));
        for side in 0..2 {
            let camera = Camera {
                yaw: Camera::default().yaw + side as f32 * std::f32::consts::PI,
                ..Default::default()
            };
            let (matrix, eye) = camera.matrices(1.0, 0.0);
            renderer.draw(
                [800, 800],
                Uniform {
                    view_projection: matrix.into(),
                    eye: [eye.x, eye.y, eye.z, 1.0],
                    options: [0.0, -1.0, 0.0, -1.0],
                },
                false,
            );
            pixels(&render, &renderer.target.as_ref().unwrap().texture)
                .save(crate::texture::asset_path(&format!(
                    "target/satellite-{name}-source-review-{side}.png"
                )))
                .unwrap();
        }
    }
    std::fs::write(
        crate::texture::asset_path("target/satellite-source-review.json"),
        serde_json::to_vec_pretty(&reviews).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "requires a native GPU adapter; writes inspection images for additional NASA models"]
fn inspect_additional_models() {
    let render = render_state();
    for asset in ASSETS.iter().skip(6) {
        if std::env::var("LEO_VIZ_MODEL_AUDIT_NAME").is_ok_and(|name| asset.name != name) {
            continue;
        }
        let name = asset
            .path
            .strip_prefix("assets/satellite-")
            .unwrap()
            .strip_suffix(".glb")
            .unwrap();
        let model = Arc::new(
            asset
                .load(&std::fs::read(crate::texture::asset_path(asset.path)).unwrap())
                .unwrap(),
        );
        eprintln!(
            "{name}: dimensions {:?}, radius {}, {} vertices, {} triangles",
            model.dimensions,
            model.radius,
            model.vertices.len(),
            model.indices.len() / 3
        );
        let mut renderer = Renderer::new(&render, Arc::clone(&model));
        for (index, part) in model.components.iter().enumerate() {
            eprintln!(
                "{name} {index}: {} | {} | min {:?}, max {:?}",
                part.info.title,
                part.meshes.len(),
                part.min,
                part.max
            );
        }
        for side in 0..2 {
            let camera = Camera {
                yaw: Camera::default().yaw + side as f32 * std::f32::consts::PI,
                ..Default::default()
            };
            let (matrix, eye) = camera.matrices(1.0, 0.0);
            renderer.draw(
                [800, 800],
                Uniform {
                    view_projection: matrix.into(),
                    eye: [eye.x, eye.y, eye.z, 1.0],
                    options: [0.0, -1.0, 0.0, -1.0],
                },
                false,
            );
            pixels(&render, &renderer.target.as_ref().unwrap().texture)
                .save(crate::texture::asset_path(&format!(
                    "target/satellite-{name}-review-{side}.png"
                )))
                .unwrap();
        }
        for (index, part) in model.components.iter().enumerate() {
            let camera = Camera {
                target: part.center,
                ..Default::default()
            };
            let (matrix, eye) =
                camera.matrices_for_bounds(1.0, (part.max - part.min).norm() * 0.56);
            renderer.draw(
                [640, 640],
                Uniform {
                    view_projection: matrix.into(),
                    eye: [eye.x, eye.y, eye.z, 1.0],
                    options: [0.0, index as f32, 1.0, -1.0],
                },
                false,
            );
            pixels(&render, &renderer.target.as_ref().unwrap().texture)
                .save(crate::texture::asset_path(&format!(
                    "target/satellite-{name}-component-{index}.png"
                )))
                .unwrap();
        }
    }
}

#[test]
#[ignore = "requires a native GPU adapter; writes source-mesh inspection sheets"]
fn inspect_source_meshes() {
    let render = render_state();
    for asset in 0..ASSETS.len() {
        if std::env::var("LEO_VIZ_MODEL_AUDIT_NAME").is_ok_and(|name| ASSETS[asset].name != name) {
            continue;
        }
        let model = Arc::new(
            Model::load(&std::fs::read(crate::texture::asset_path(ASSETS[asset].path)).unwrap())
                .unwrap(),
        );
        let mut renderer = Renderer::new(&render, Arc::clone(&model));
        let mut sheet = image::RgbaImage::new(2048, model.parts.len().div_ceil(4) as u32 * 200);
        eprintln!(
            "{} original dimensions: {:?}",
            ASSETS[asset].name, model.dimensions
        );
        for (index, part) in model.parts.iter().enumerate() {
            eprintln!(
                "{index}: {} | {} | center {:?} | size {:?}",
                part.name,
                model.materials[part.material].name,
                part.center,
                part.max - part.min
            );
            for side in 0..2 {
                let camera = Camera {
                    target: part.center,
                    yaw: Camera::default().yaw + side as f32 * std::f32::consts::PI,
                    ..Camera::default()
                };
                let (matrix, eye) =
                    camera.matrices_for_bounds(256.0 / 200.0, (part.max - part.min).norm() * 0.56);
                renderer.draw(
                    [256, 200],
                    Uniform {
                        view_projection: matrix.into(),
                        eye: [eye.x, eye.y, eye.z, 1.0],
                        options: [0.0, index as f32, 1.0, -1.0],
                    },
                    false,
                );
                let tile = pixels(&render, &renderer.target.as_ref().unwrap().texture);
                image::imageops::replace(
                    &mut sheet,
                    &tile,
                    (index % 4 * 512 + side * 256) as i64,
                    (index / 4 * 200) as i64,
                );
            }
        }
        sheet
            .save(crate::texture::asset_path(&format!(
                "target/satellite-view-{asset}-mesh-audit.png"
            )))
            .unwrap();
    }
}

#[test]
#[ignore = "requires a native GPU adapter; verifies overlapping transparent surfaces"]
fn transparent_surfaces_do_not_hide_textures_behind_them() {
    let render = render_state();
    let mut model = ASSETS[4]
        .load(&std::fs::read(crate::texture::asset_path(ASSETS[4].path)).unwrap())
        .unwrap();
    // One transparent primitive emits its near surface before a farther surface.
    let template = model.vertices[0];
    model.vertices = [0.2, -0.2]
        .into_iter()
        .flat_map(|z| {
            [[-0.8, -0.8, z], [0.8, -0.8, z], [0.0, 0.8, z]].map(|position| Vertex {
                position,
                normal: [0.0, 0.0, 1.0],
                uv: [if z > 0.0 { 0.25 } else { 0.75 }, 0.5],
                color: [1.0; 4],
                offset: [0.0; 3],
                part: 0,
                ..template
            })
        })
        .collect();
    model.indices = (0..6).collect();
    model.parts.retain(|p| p.is_visible(&model.materials));
    model.parts.truncate(1);
    model.parts[0].start = 0;
    model.parts[0].end = 6;
    model.parts[0].material = 0;
    model.materials.truncate(1);
    let material = &mut model.materials[0];
    material.color = [1.0; 4];
    material.metallic = 0.0;
    material.roughness = 1.0;
    material.base_texture = Some(0);
    material.normal_texture = None;
    material.normal_scale = 0.0;
    material.roughness_texture = None;
    material.alpha_cutoff = 0.0;
    material.blend = true;
    material.appearance = crate::satellite_model::MaterialAppearance::Original;
    model.images = vec![image::RgbaImage::from_fn(2, 1, |x, _| {
        image::Rgba([255, 0, 0, if x == 0 { 0 } else { 128 }])
    })];
    let mut renderer = Renderer::new(&render, Arc::new(model));
    let camera = Camera {
        yaw: 0.0,
        pitch: 0.0,
        ..Default::default()
    };
    let (matrix, eye) = camera.matrices_for_bounds(1.0, 1.0);
    for dark in [false, true] {
        renderer.draw(
            [256, 256],
            Uniform {
                view_projection: matrix.into(),
                eye: [eye.x, eye.y, eye.z, 1.0],
                options: [0.0, -1.0, 0.0, -1.0],
            },
            dark,
        );
        let image = pixels(&render, &renderer.target.as_ref().unwrap().texture);
        let center = image.get_pixel(128, 128);
        assert!(
            i16::from(center[0]) > i16::from(center[1]) + 40,
            "transparent foreground must not hide the red surface: {center:?}"
        );
    }
}

#[test]
#[ignore = "requires a native GPU adapter; verifies SMAP's simplified reflector appearance from both sides"]
fn smap_reflector_textures_render_from_both_sides() {
    let render = render_state();
    let model = Arc::new(
        ASSETS[4]
            .load(&std::fs::read(crate::texture::asset_path(ASSETS[4].path)).unwrap())
            .unwrap(),
    );
    let reflector = &model.components[0];
    assert_eq!(reflector.info.title, "Mesh reflector");
    for material in model.materials.iter().filter(|m| m.name == "antenna_a") {
        let image = &model.images[material.base_texture.expect("original reflector texture")];
        let alpha: u64 = image.pixels().map(|p| u64::from(p[3])).sum();
        eprintln!(
            "Reflector mean opacity: {:.3}",
            alpha as f64 / (255.0 * f64::from(image.width() * image.height()))
        );
        let gold: Vec<_> = image
            .pixels()
            .filter(|p| p[0] > p[2].saturating_add(40))
            .collect();
        eprintln!(
            "Gold texel mean opacity: {:.3}",
            gold.iter().map(|p| f64::from(p[3])).sum::<f64>() / (255.0 * gold.len() as f64)
        );
        assert!(image.pixels().any(|p| p[0] > p[2].saturating_add(40)));
    }
    let mut renderer = Renderer::new(&render, Arc::clone(&model));
    for (side, yaw) in [0.65, 0.65 + std::f32::consts::PI].into_iter().enumerate() {
        let camera = Camera {
            target: reflector.center,
            yaw,
            pitch: 0.5,
            ..Default::default()
        };
        let (matrix, eye) =
            camera.matrices_for_bounds(1.0, (reflector.max - reflector.min).norm() * 0.56);
        for dark in [false, true] {
            renderer.draw(
                [512, 512],
                Uniform {
                    view_projection: matrix.into(),
                    eye: [eye.x, eye.y, eye.z, 1.0],
                    options: [0.0, 0.0, 1.0, -1.0],
                },
                dark,
            );
            let image = pixels(&render, &renderer.target.as_ref().unwrap().texture);
            image
                .save(crate::texture::asset_path(&format!(
                    "target/smap-reflector-{side}-{dark}.png"
                )))
                .unwrap();
            let gold = image
                .pixels()
                .filter(|p| {
                    i16::from(p[0]) > i16::from(p[2]) + 10 && i16::from(p[1]) > i16::from(p[2]) + 10
                })
                .count();
            assert!(gold > 100, "original gold reflector texture missing on side {side}, dark={dark}: {gold} pixels");
        }
    }
}

#[test]
#[ignore = "requires a native GPU adapter; verifies persistent selection and hover priority"]
fn selection_stays_blue_and_hover_takes_priority() {
    let render = render_state();
    for asset in 0..ASSETS.len() {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        let mut view = loaded(asset);
        let size = [1000, 700];
        let initial = capture_frame(&mut view, &ctx, &render, size, false, Vec::new());
        let texture = view
            .renderer
            .as_ref()
            .unwrap()
            .target
            .as_ref()
            .unwrap()
            .texture
            .clone();
        let base = pixels(&render, &texture);
        let (_, selected) = visible_hit(&view, initial.scene);
        view.selected = Some(selected);
        capture_frame(&mut view, &ctx, &render, size, false, Vec::new());
        let selected_pixels = pixels(&render, &texture);
        assert_ne!(selected_pixels, base, "selection must stay highlighted");
        let uniform = view.renderer.as_ref().unwrap().last_frame.unwrap().0;
        assert_eq!(uniform.options[3], selected as f32);
        let preview_texture = view
            .preview_renderer
            .as_ref()
            .unwrap()
            .target
            .as_ref()
            .unwrap()
            .texture
            .clone();
        let preview = pixels(&render, &preview_texture);
        capture_frame(&mut view, &ctx, &render, size, false, Vec::new());
        assert_eq!(
            bytemuck::bytes_of(&view.renderer.as_ref().unwrap().last_frame.unwrap().0),
            bytemuck::bytes_of(&uniform),
            "selection must not animate or invalidate the scene on each frame"
        );
        assert!(pixels(&render, &texture) == selected_pixels);
        assert_eq!(pixels(&render, &preview_texture), preview);
        assert_eq!(view.selected, Some(selected));

        let next = (selected + 1) % view.model.as_ref().unwrap().components.len();
        let title = view.model.as_ref().unwrap().components[next].info.title;
        let row = capture_frame(&mut view, &ctx, &render, size, false, Vec::new())
            .label(title)
            .center();
        capture_frame(
            &mut view,
            &ctx,
            &render,
            size,
            false,
            vec![egui::Event::PointerMoved(row)],
        );
        assert_eq!(view.hovered, Some(next));
        assert_eq!(view.selected, Some(selected));
        assert_eq!(
            view.renderer
                .as_ref()
                .unwrap()
                .last_frame
                .unwrap()
                .0
                .options[3],
            next as f32,
            "hover must replace the selected highlight, not add a second one"
        );
        capture_frame(
            &mut view,
            &ctx,
            &render,
            size,
            false,
            vec![egui::Event::PointerGone],
        );
        assert_eq!(
            view.hovered, None,
            "pointer leaving the list must clear hover"
        );
        assert!(
            pixels(&render, &texture) == selected_pixels,
            "{} must restore its selected highlight after hover leaves",
            ASSETS[asset].name
        );

        view.selected = None;
        capture_frame(&mut view, &ctx, &render, size, false, Vec::new());
        assert!(pixels(&render, &texture) == base);
    }
}

#[test]
#[ignore = "requires a native GPU adapter; verifies statistics, source notes, and full-view rulers"]
fn stats_share_part_details_without_resizing_the_view() {
    let render = render_state();
    for asset in 0..ASSETS.len() {
        let ctx = egui::Context::default();
        ctx.global_style_mut(|style| {
            style.interaction.tooltip_delay = 0.0;
            style.animation_time = 0.0;
        });
        egui_extras::install_image_loaders(&ctx);
        let mut view = loaded(asset);
        let size = [1200, 850];
        capture_frame(&mut view, &ctx, &render, size, false, Vec::new());
        let frame = capture_frame(&mut view, &ctx, &render, size, false, Vec::new());
        assert!(frame.label(ASSETS[asset].name).bottom() < frame.scene.top());
        let heading = format!(
            "Components ({} total)",
            view.model.as_ref().unwrap().components.len()
        );
        let list = frame.label(&heading);
        let info = egui::Rect::from_min_max(
            egui::pos2(frame.scene.right() + 12.0, frame.scene.top()),
            egui::pos2(size[0] as f32 - 8.0, list.top() - 8.0),
        );
        let launched = frame.label("Launched");
        let body = egui::TextStyle::Body.resolve(&ctx.global_style());
        let row_height = ctx.fonts_mut(|fonts| fonts.row_height(&body));
        for text in ASSETS[asset]
            .facts
            .rows
            .iter()
            .flat_map(|&(key, value)| [key, value])
            .chain(ASSETS[asset].facts.sources.iter().map(|&(name, _)| name))
        {
            let font = &frame
                .text_fonts
                .iter()
                .find(|(label, _)| label == text)
                .unwrap()
                .1;
            assert!(
                font.size >= body.size,
                "statistics must use readable body text: {text}"
            );
        }
        for &(_, value) in ASSETS[asset].facts.rows {
            assert_eq!(
                frame
                    .text_lines
                    .iter()
                    .find(|(text, _)| text == value)
                    .unwrap()
                    .1,
                1,
                "statistics value must not wrap: {value}"
            );
        }
        assert!(
            info.contains_rect(launched),
            "statistics must be in the part-information panel, not over the model"
        );
        let satellite_preview = frame
            .preview
            .expect("whole-satellite preview with statistics");
        assert!(info.contains_rect(satellite_preview));
        assert_whole_satellite_preview(&view, &render);
        let stats_header = frame
            .labels
            .iter()
            .find(|(text, rect)| text == ASSETS[asset].name && info.contains_rect(*rect))
            .expect("satellite name in the information panel")
            .1;
        assert!(!frame.labels.iter().any(|(text, rect)| (text == "Stats"
            || text == "Reference specifications")
            && frame.scene.contains_rect(*rect)));
        let description = ASSETS[asset].facts.description;
        assert!(!frame.labels.iter().any(|(text, _)| text == description));
        assert!(
            launched.top() >= satellite_preview.bottom()
                && launched.top() - satellite_preview.bottom() <= 12.0,
            "statistics must follow the satellite preview"
        );
        let help = frame.label("[?]");
        assert!(info.contains_rect(help));
        assert!(help.left() >= stats_header.right() && help.left() - stats_header.right() <= 20.0);
        for &(key, value) in ASSETS[asset].facts.rows {
            assert!(info.contains_rect(frame.label(key)));
            assert!(info.contains_rect(frame.label(value)));
            assert!(
                (frame.label(key).left() - launched.left()).abs() <= 2.0,
                "statistics labels must align left"
            );
            assert!(
                (frame.label(value).left() - frame.label(ASSETS[asset].facts.rows[0].1).left())
                    .abs()
                    <= 2.0,
                "statistics values must align left"
            );
            assert_eq!(
                frame
                    .text_fonts
                    .iter()
                    .find(|(text, _)| text == key)
                    .unwrap()
                    .1
                    .family,
                egui::FontFamily::Name("satellite-bold".into()),
                "statistics labels must use a bold font"
            );
        }
        let components = &view.model.as_ref().unwrap().components;
        for component in components {
            let Some((_, rect)) = frame
                .labels
                .iter()
                .find(|(text, _)| text == component.info.title)
            else {
                continue;
            };
            assert!(
                (rect.left() - (list.left() + 4.0)).abs() <= 3.0,
                "component entries must align left: {}",
                component.info.title
            );
            assert_eq!(
                frame
                    .text_fonts
                    .iter()
                    .find(|(text, _)| text == component.info.title)
                    .unwrap()
                    .1
                    .family,
                egui::FontFamily::Name("satellite-bold".into()),
                "component entries must use a bold font"
            );
        }
        assert!(frame.label("X (m)").top() > frame.scene.bottom());
        assert!(frame.label("Y (m)").right() < frame.scene.left());
        for pair in ASSETS[asset].facts.rows.windows(2) {
            assert!(
                frame.label(pair[0].1).bottom() <= frame.label(pair[1].0).top(),
                "statistics rows must not overlap"
            );
            assert!(
                frame.label(pair[1].0).top() - frame.label(pair[0].0).top() <= row_height + 4.0,
                "statistics rows must stay compact"
            );
        }
        assert!(
            frame
                .labels
                .iter()
                .filter(|(text, rect)| text.parse::<f32>().is_ok()
                    && rect.center().y > frame.scene.bottom())
                .count()
                >= 3
        );
        assert!(
            frame
                .labels
                .iter()
                .filter(
                    |(text, rect)| text.parse::<f32>().is_ok() && rect.right() < frame.scene.left()
                )
                .count()
                >= 3
        );
        let before = view.camera;
        capture_frame(
            &mut view,
            &ctx,
            &render,
            size,
            false,
            vec![egui::Event::PointerMoved(help.center())],
        );
        let about = capture_frame(&mut view, &ctx, &render, size, false, Vec::new());
        assert!(
            about.labels.iter().any(|(text, _)| text == description),
            "hovering [?] must show the mission description"
        );
        assert_eq!(view.camera.yaw, before.yaw);
        assert!(view.selected.is_none());
        if asset == 0 {
            about
                .image
                .save(crate::texture::asset_path(
                    "target/satellite-view-about.png",
                ))
                .unwrap();
        }
        for pressed in [true, false] {
            capture_frame(
                &mut view,
                &ctx,
                &render,
                size,
                false,
                vec![
                    egui::Event::PointerMoved(launched.center()),
                    egui::Event::PointerButton {
                        pos: launched.center(),
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: Default::default(),
                    },
                ],
            );
        }
        assert!(view.selected.is_none());
        assert_eq!(view.camera.yaw, before.yaw);
        assert_eq!(view.camera.zoom, before.zoom);
        frame
            .image
            .save(crate::texture::asset_path(&format!(
                "target/satellite-view-{asset}-stats.png"
            )))
            .unwrap();
        view.selected = Some(0);
        let details = capture_frame(
            &mut view,
            &ctx,
            &render,
            size,
            false,
            vec![egui::Event::PointerGone],
        );
        assert!(!details.labels.iter().any(|(text, _)| text == "Launched"));
        assert!(info.contains_rect(details.preview.expect("selected part preview")));
        assert_eq!(
            details.preview, frame.preview,
            "satellite and part images share the same frame"
        );
        assert_eq!(
            details.scene, frame.scene,
            "part details must not resize the grid"
        );
        assert_eq!(
            details.label(&heading),
            list,
            "component list must stay in place"
        );
        let component = &view.model.as_ref().unwrap().components[0];
        assert!(
            details
                .labels
                .iter()
                .any(|(text, _)| text == component.info.description),
            "selecting a part must replace statistics with its description"
        );
        assert!(
            !details
                .labels
                .iter()
                .any(|(text, _)| text == component.info.review),
            "mesh caveats must remain inside the collapsed Source meshes section"
        );
        let background = frame.scene.left_top() + egui::vec2(4.0, 4.0);
        for pressed in [true, false] {
            capture_frame(
                &mut view,
                &ctx,
                &render,
                size,
                false,
                vec![
                    egui::Event::PointerMoved(background),
                    egui::Event::PointerButton {
                        pos: background,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: Default::default(),
                    },
                ],
            );
        }
        let restored = capture_frame(
            &mut view,
            &ctx,
            &render,
            size,
            false,
            vec![egui::Event::PointerGone],
        );
        assert!(view.selected.is_none());
        assert_eq!(restored.preview, frame.preview);
        assert_whole_satellite_preview(&view, &render);
        assert_eq!(
            restored.label("Launched"),
            launched,
            "clearing selection must restore statistics"
        );
        assert_eq!(restored.scene, frame.scene);
        assert_eq!(restored.label(&heading), list);
        for (size, dark, name) in [([390, 844], false, "narrow"), ([1200, 850], true, "dark")] {
            capture_frame(&mut view, &ctx, &render, size, dark, Vec::new());
            let idle = capture_frame(&mut view, &ctx, &render, size, dark, Vec::new());
            let row = idle.label("Launched");
            assert!(!idle.scene.intersects(row));
            assert!(row.bottom() < idle.label(&heading).top());
            assert!(idle.preview.expect("satellite preview").bottom() <= row.top());
            assert_whole_satellite_preview(&view, &render);
            idle.image
                .save(crate::texture::asset_path(&format!(
                    "target/satellite-view-{asset}-stats-{name}.png"
                )))
                .unwrap();
        }
    }
}

fn assert_whole_satellite_preview(view: &View, render: &RenderState) {
    let preview = view
        .preview_renderer
        .as_ref()
        .expect("satellite preview renderer");
    assert_eq!(
        preview.last_frame.as_ref().unwrap().0.options,
        [0.0, -1.0, 0.0, -1.0]
    );
    let target = preview.target.as_ref().unwrap();
    assert_ne!(
        target.id,
        view.renderer.as_ref().unwrap().target.as_ref().unwrap().id
    );
    let image = pixels(render, &target.texture);
    let background = image.get_pixel(0, 0).0;
    let mut count = 0;
    for (x, y, pixel) in image.enumerate_pixels() {
        if pixel
            .0
            .iter()
            .zip(background)
            .take(3)
            .any(|(a, b)| a.abs_diff(b) > 12)
        {
            count += 1;
            assert!(
                x > 1 && y > 1 && x + 2 < image.width() && y + 2 < image.height(),
                "the entire satellite must fit within the thumbnail"
            );
        }
    }
    assert!(count > 100, "satellite thumbnail must not be blank");
}

#[test]
#[ignore = "requires a native GPU adapter; verifies fixed part preview geometry and layout"]
fn fixed_preview_fits_parts_without_resizing_the_list() {
    let render = render_state();
    for asset in 0..ASSETS.len() {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        let mut view = loaded(asset);
        for (size, dark, name) in [
            ([1200, 850], false, "light"),
            ([1200, 850], true, "dark"),
            ([390, 844], false, "narrow"),
        ] {
            view.selected = None;
            let empty = capture_frame(
                &mut view,
                &ctx,
                &render,
                size,
                dark,
                vec![egui::Event::PointerGone],
            );
            let heading = format!(
                "Components ({} total)",
                view.model.as_ref().unwrap().components.len()
            );
            let list = empty.label(&heading);
            assert!(empty.preview.is_some());
            assert!(empty.label("Launched").bottom() < list.top());
            let mut preview_rect = empty.preview;
            let mut texture_id = view
                .preview_renderer
                .as_ref()
                .unwrap()
                .target
                .as_ref()
                .map(|t| t.id);
            let model = view.model.as_ref().unwrap();
            let smallest = model
                .components
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (a.max - a.min)
                        .norm_squared()
                        .total_cmp(&(b.max - b.min).norm_squared())
                })
                .unwrap()
                .0;
            for index in [0, smallest, model.components.len() - 1] {
                view.selected = Some(index);
                let frame = capture_frame(
                    &mut view,
                    &ctx,
                    &render,
                    size,
                    dark,
                    vec![egui::Event::PointerGone],
                );
                assert_eq!(frame.label(&heading), list);
                assert!(!frame.labels.iter().any(|(text, _)| text == "Launched"));
                let rect = frame.preview.expect("isolated part image");
                assert!(rect.bottom() < list.top());
                if let Some(previous) = preview_rect {
                    assert_eq!(previous, rect);
                }
                preview_rect = Some(rect);
                let preview = view.preview_renderer.as_ref().unwrap();
                let target = preview.target.as_ref().unwrap();
                if let Some(previous) = texture_id {
                    assert_eq!(previous, target.id);
                }
                texture_id = Some(target.id);
                assert_ne!(
                    target.id,
                    view.renderer.as_ref().unwrap().target.as_ref().unwrap().id
                );
                let image = pixels(&render, &target.texture);
                let bg = image.get_pixel(0, 0).0;
                let displayed = frame
                    .image
                    .get_pixel((rect.left() + 3.0) as u32, (rect.top() + 3.0) as u32)
                    .0;
                let source = image.get_pixel(3, 3).0;
                assert!(
                    displayed
                        .iter()
                        .zip(source)
                        .take(3)
                        .all(|(a, b)| a.abs_diff(b) <= 2),
                    "preview must preserve encoded colours"
                );
                assert!(
                    image
                        .pixels()
                        .filter(|p| p.0.iter().zip(bg).take(3).any(|(a, b)| a.abs_diff(b) > 8))
                        .count()
                        > 5,
                    "asset {asset}, part {index}, {name}"
                );
                assert_eq!(
                    preview.last_frame.as_ref().unwrap().0.options,
                    [0.0, index as f32, 1.0, -1.0]
                );
                frame
                    .image
                    .save(crate::texture::asset_path(&format!(
                        "target/satellite-view-{asset}-fixed-details-{name}.png"
                    )))
                    .unwrap();
            }
        }
    }
}

#[test]
#[ignore = "requires a native GPU adapter; writes satellite inspector screenshots"]
fn downloaded_models_render_in_light_dark_and_narrow_views() {
    let render = render_state();
    for asset in 0..ASSETS.len() {
        let mut view = loaded(asset);
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        for (name, size, dark) in [
            ("desktop-light", [1400, 850], false),
            ("desktop-dark", [1400, 850], true),
            ("narrow-light", [390, 844], false),
        ] {
            let image = capture(&mut view, &ctx, &render, size, dark, Vec::new());
            let path =
                crate::texture::asset_path(&format!("target/satellite-view-{asset}-{name}.png"));
            image.save(&path).unwrap();
            let scene = pixels(
                &render,
                &view
                    .renderer
                    .as_ref()
                    .unwrap()
                    .target
                    .as_ref()
                    .unwrap()
                    .texture,
            );
            let bg = scene.get_pixel(0, 0);
            let non_background = scene
                .pixels()
                .filter(|p| {
                    p.0.iter()
                        .zip(bg.0)
                        .take(3)
                        .any(|(a, b)| a.abs_diff(b) > 12)
                })
                .count();
            assert!(
                non_background > scene.width() as usize * scene.height() as usize / 100,
                "{} is blank",
                path.display()
            );
            if !dark {
                assert_eq!(bg.0, [255; 4]);
            }
            assert!(scene.pixels().all(|p| p.0[3] == 255));
        }
        view.explosion = 0.8;
        view.selected = Some(0);
        capture(&mut view, &ctx, &render, [1400, 850], false, Vec::new())
            .save(crate::texture::asset_path(&format!(
                "target/satellite-view-{asset}-exploded.png"
            )))
            .unwrap();
        view.isolate = true;
        capture(&mut view, &ctx, &render, [1400, 850], false, Vec::new())
            .save(crate::texture::asset_path(&format!(
                "target/satellite-view-{asset}-isolated.png"
            )))
            .unwrap();
    }
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn camera_motion_picking_and_texture_lifetime_are_independent_per_view() {
    let render = render_state();
    let ctx = egui::Context::default();
    egui_extras::install_image_loaders(&ctx);
    let mut first = loaded(0);
    let mut second = View {
        model: first.model.clone(),
        spin: false,
        ..Default::default()
    };
    let before = capture(&mut first, &ctx, &render, [1000, 700], false, Vec::new());
    let first_id = first.renderer.as_ref().unwrap().target.as_ref().unwrap().id;
    capture(&mut second, &ctx, &render, [1000, 700], false, Vec::new());
    let second_id = second
        .renderer
        .as_ref()
        .unwrap()
        .target
        .as_ref()
        .unwrap()
        .id;
    assert_ne!(first_id, second_id);
    let pos = egui::pos2(370.0, 300.0);
    let pointer = |pos, pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: Default::default(),
    };
    capture(
        &mut first,
        &ctx,
        &render,
        [1000, 700],
        false,
        vec![egui::Event::PointerMoved(pos), pointer(pos, true)],
    );
    let moved = pos + egui::vec2(90.0, 35.0);
    let after = capture(
        &mut first,
        &ctx,
        &render,
        [1000, 700],
        false,
        vec![egui::Event::PointerMoved(moved)],
    );
    capture(
        &mut first,
        &ctx,
        &render,
        [1000, 700],
        false,
        vec![pointer(moved, false)],
    );
    assert!(first.camera.yaw != Camera::default().yaw);
    assert_eq!(second.camera.yaw, Camera::default().yaw);
    assert!(
        before
            .pixels()
            .zip(after.pixels())
            .filter(|(a, b)| a != b)
            .count()
            > 2000
    );
    assert_eq!(
        first.renderer.as_ref().unwrap().target.as_ref().unwrap().id,
        first_id
    );
    let model = first.model.as_ref().unwrap();
    let (_, eye) = first.camera.matrices(1.0, 0.0);
    let part = model.pick(eye, -eye.normalize(), 0.0, None);
    assert!(part.is_some());
    let id = part.unwrap();
    let direction = (model.components[id].center - eye).normalize();
    assert_eq!(model.pick(eye, direction, 0.0, Some(id)), Some(id));
    drop(first);
    assert!(render.renderer.read().texture(&first_id).is_none());
    assert!(render.renderer.read().texture(&second_id).is_some());
    drop(second);
    assert!(render.renderer.read().texture(&second_id).is_none());
}
