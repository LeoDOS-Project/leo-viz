//! A separate, per-tab inspector for downloaded spacecraft models.
use crate::satellite_model::{Model, Vertex, ASSETS};
use eframe::egui;
use egui_wgpu::{wgpu, RenderState};
use nalgebra::{Isometry3, Matrix4, Perspective3, Point3, Vector3, Vector4};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

type LoadResult = Arc<Mutex<Option<Result<Model, String>>>>;

pub(crate) struct View {
    asset: usize,
    loaded_asset: usize,
    model: Option<Arc<Model>>,
    models: HashMap<usize, Arc<Model>>,
    pending: HashMap<usize, LoadResult>,
    errors: HashMap<usize, String>,
    comparison_renderers: HashMap<usize, Renderer>,
    comparison_search: String,
    comparison_sort: sort::Sort,
    comparison_scroll_to_top: bool,
    catalogue_selection: Option<u32>,
    model_panes: Vec<navigation::Pane>,
    pane_counter: u32,
    pending_pane_removal: Option<u32>,
    comparison_camera: Option<navigation::ComparisonCamera>,
    model_action: Option<navigation::Action>,
    allow_pane_close: bool,
    return_to_table: bool,
    renderer: Option<Renderer>,
    preview_renderer: Option<Renderer>,
    camera: Camera,
    selected: Option<usize>,
    hovered: Option<usize>,
    inspected: Option<usize>,
    picking: Option<(PickQuery, Option<usize>)>,
    explosion: f32,
    isolate: bool,
    spin: bool,
    parts_open: bool,
    rulers: bool,
}

impl Default for View {
    fn default() -> Self {
        Self {
            asset: 0,
            loaded_asset: 0,
            model: None,
            models: HashMap::new(),
            pending: HashMap::new(),
            errors: HashMap::new(),
            comparison_renderers: HashMap::new(),
            comparison_search: String::new(),
            comparison_sort: sort::Sort::default(),
            comparison_scroll_to_top: false,
            catalogue_selection: None,
            model_panes: Vec::new(),
            pane_counter: 0,
            pending_pane_removal: None,
            comparison_camera: None,
            model_action: None,
            allow_pane_close: false,
            return_to_table: false,
            renderer: None,
            preview_renderer: None,
            camera: Camera::default(),
            selected: None,
            hovered: None,
            inspected: None,
            picking: None,
            explosion: 0.0,
            isolate: false,
            spin: true,
            parts_open: true,
            rulers: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Camera {
    yaw: f32,
    pitch: f32,
    zoom: f32,
    target: Vector3<f32>,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw: 0.65,
            pitch: -0.38,
            zoom: 1.0,
            target: Vector3::zeros(),
        }
    }
}

impl Camera {
    fn matrices(&self, aspect: f32, explosion: f32) -> (Matrix4<f32>, Vector3<f32>) {
        self.matrices_for_bounds(aspect, 1.12 + explosion * 0.55)
    }

    fn matrices_for_bounds(&self, aspect: f32, radius: f32) -> (Matrix4<f32>, Vector3<f32>) {
        let fov = 40.0_f32.to_radians();
        let half_angle = (fov * 0.5).min(((fov * 0.5).tan() * aspect).atan());
        let distance = radius.max(1e-5) / half_angle.sin() / self.zoom;
        let eye = self.target
            + Vector3::new(
                self.yaw.sin() * self.pitch.cos(),
                self.pitch.sin(),
                self.yaw.cos() * self.pitch.cos(),
            ) * distance;
        let view = Isometry3::look_at_rh(
            &Point3::from(eye),
            &Point3::from(self.target),
            &Vector3::y(),
        )
        .to_homogeneous();
        let depth = Matrix4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0,
        );
        (
            depth
                * Perspective3::new(
                    aspect,
                    fov,
                    (distance * 0.001).max(1e-7),
                    (distance + radius * 4.0).max(10.0),
                )
                .to_homogeneous()
                * view,
            eye,
        )
    }

    fn ray(
        &self,
        rect: egui::Rect,
        point: egui::Pos2,
        explosion: f32,
    ) -> Option<(Vector3<f32>, Vector3<f32>)> {
        let (matrix, eye) = self.matrices(rect.aspect_ratio(), explosion);
        let inv = matrix.try_inverse()?;
        let p = inv
            * Vector4::new(
                (point.x - rect.left()) / rect.width() * 2.0 - 1.0,
                1.0 - (point.y - rect.top()) / rect.height() * 2.0,
                1.0,
                1.0,
            );
        let world = p.xyz() / p.w;
        Some((eye, (world - eye).normalize()))
    }

    fn zoom_by(&mut self, factor: f32) {
        self.zoom = (self.zoom * factor).clamp(0.0001, 10_000.0);
    }

    fn orbit(&mut self, delta: egui::Vec2, zoom: f32) {
        let sensitivity = 0.008 / zoom.max(1.0);
        self.yaw -= delta.x * sensitivity;
        self.pitch = (self.pitch + delta.y * sensitivity).clamp(-1.50, 1.50);
    }

    fn metres_per_point(&self, rect: egui::Rect, explosion: f32, metres_per_unit: f32) -> f32 {
        let (_, eye) = self.matrices(rect.aspect_ratio(), explosion);
        2.0 * (eye - self.target).norm() * 20.0_f32.to_radians().tan() * metres_per_unit
            / rect.height()
    }
}

struct Scene {
    ui: egui::Ui,
    bounds: egui::Rect,
    rect: egui::Rect,
    ruler_visible: bool,
    image: egui::layers::ShapeIdx,
    camera_changed: bool,
    reset_camera: bool,
}

fn view_layout(
    body: egui::Rect,
    parts_open: bool,
) -> (egui::Rect, Option<(egui::Rect, egui::Rect)>) {
    if !parts_open {
        return (body, None);
    }
    let (scene, inspector) = if body.width() >= 740.0 {
        let divider = body.right() - 300.0;
        (
            egui::Rect::from_min_max(body.min, egui::pos2(divider - 12.0, body.bottom())),
            egui::Rect::from_min_max(egui::pos2(divider, body.top()), body.max),
        )
    } else {
        let divider = body.bottom() - (body.height() * 0.58).min(440.0);
        (
            egui::Rect::from_min_max(body.min, egui::pos2(body.right(), divider - 8.0)),
            egui::Rect::from_min_max(egui::pos2(body.left(), divider), body.max),
        )
    };
    let height = (inspector.height() - 100.0).clamp(0.0, 340.0);
    let preview = egui::Rect::from_min_size(inspector.min, egui::vec2(inspector.width(), height));
    let list = egui::Rect::from_min_max(
        egui::pos2(inspector.left(), preview.bottom() + 8.0),
        inspector.max,
    );
    (scene, Some((preview, list)))
}

fn comparison_layout(body: egui::Rect) -> (egui::Rect, egui::Rect) {
    let height = (body.height() * 0.35).min(220.0);
    let stats = egui::Rect::from_min_max(egui::pos2(body.left(), body.bottom() - height), body.max);
    let scene = egui::Rect::from_min_max(
        body.min,
        egui::pos2(body.right(), (stats.top() - 8.0).max(body.top())),
    );
    (scene, stats)
}

#[derive(Clone, Copy, PartialEq)]
struct PickQuery {
    origin: Vector3<f32>,
    direction: Vector3<f32>,
    explosion: f32,
    isolate: Option<usize>,
}

impl View {
    pub fn toggle_auto_rotate(&mut self) {
        self.spin = !self.spin;
        for pane in &mut self.model_panes {
            pane.view.spin = self.spin;
        }
    }

    fn displayed_asset(&self) -> &'static crate::satellite_model::Asset {
        &ASSETS[if self.model.is_some() {
            self.loaded_asset
        } else {
            self.asset
        }]
    }

    fn poll_loads(&mut self) {
        self.pending.retain(|&index, pending| {
            let Some(result) = pending.lock().unwrap().take() else {
                return true;
            };
            match result {
                Ok(model) => {
                    self.models.insert(index, Arc::new(model));
                }
                Err(error) => {
                    self.errors.insert(index, error);
                }
            }
            false
        });
    }

    fn load(&mut self, ctx: &egui::Context) {
        if let Some(model) = &self.model {
            self.models
                .entry(self.loaded_asset)
                .or_insert_with(|| Arc::clone(model));
        }
        self.poll_loads();
        self.request_model(self.asset, ctx);
        if self.model.is_none() || self.loaded_asset != self.asset {
            if let Some(model) = self.models.get(&self.asset).cloned() {
                self.install_model(model);
            }
        }
    }

    fn install_model(&mut self, model: Arc<Model>) {
        if let Some(previous) = &self.model {
            let old_units = ASSETS[self.loaded_asset].metres_per_unit(previous);
            let new_units = ASSETS[self.asset].metres_per_unit(&model);
            self.camera.zoom *= new_units / old_units * 1.12 / (1.12 + self.explosion * 0.55);
            self.camera.target *= old_units / new_units;
        }
        self.loaded_asset = self.asset;
        self.model = Some(model);
        self.renderer = None;
        self.preview_renderer = None;
        self.selected = None;
        self.hovered = None;
        self.inspected = None;
        self.picking = None;
        self.explosion = 0.0;
    }

    fn request_model(&mut self, index: usize, ctx: &egui::Context) {
        if self.models.contains_key(&index)
            || self.errors.contains_key(&index)
            || self.pending.contains_key(&index)
            || self.pending.len() >= 2
        {
            return;
        }
        let result: LoadResult = Arc::new(Mutex::new(None));
        self.pending.insert(index, Arc::clone(&result));
        let ctx = ctx.clone();
        let asset = &ASSETS[index];
        let path = asset.path;
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            let loaded = std::fs::read(crate::texture::asset_path(path))
                .map_err(|e| {
                    format!("Cannot load {path}: {e}. Run git lfs pull to fetch model assets.")
                })
                .and_then(|bytes| asset.load(&bytes));
            *result.lock().unwrap() = Some(loaded);
            ctx.request_repaint();
        });
        #[cfg(target_arch = "wasm32")]
        ehttp::fetch(ehttp::Request::get(path), move |response| {
            let loaded = response.and_then(|response| {
                if response.ok {
                    asset.load(&response.bytes)
                } else {
                    Err(format!("Model download failed: HTTP {}", response.status))
                }
            });
            *result.lock().unwrap() = Some(loaded);
            ctx.request_repaint();
        });
    }

    fn show_model(
        &mut self,
        ui: &mut egui::Ui,
        render: Option<&RenderState>,
        comparison: Option<navigation::ComparisonCamera>,
    ) -> Option<Scene> {
        self.comparison_renderers.clear();
        self.load(ui.ctx());
        self.hovered = None;
        let comparing = comparison.is_some();
        let toolbar_camera = self.camera;
        let mut reset_camera = false;
        ui.horizontal_wrapped(|ui| {
            self.model_selector(ui);
            let tint = ui.visuals().text_color();
            for (svg, name, action) in [
                (
                    egui::include_image!("../assets/lucide-zoom-out.svg"),
                    "Zoom out",
                    0,
                ),
                (
                    egui::include_image!("../assets/lucide-zoom-in.svg"),
                    "Zoom in",
                    1,
                ),
                (
                    egui::include_image!("../assets/lucide-scan.svg"),
                    "Reset camera",
                    2,
                ),
            ] {
                if ui
                    .add(egui::Button::image(
                        egui::Image::new(svg)
                            .fit_to_exact_size(egui::vec2(16.0, 16.0))
                            .tint(tint),
                    ))
                    .on_hover_text(name)
                    .clicked()
                {
                    match action {
                        0 => self.camera.zoom_by(1.0 / 1.2),
                        1 => self.camera.zoom_by(1.2),
                        _ => {
                            self.camera = Camera::default();
                            reset_camera = true;
                        }
                    }
                }
            }
            ui.checkbox(&mut self.spin, "Auto rotate");
            if !comparing {
                ui.checkbox(&mut self.parts_open, "Parts");
            }
        });
        self.request_model(self.asset, ui.ctx());
        ui.horizontal_wrapped(|ui| {
            ui.label("Explode");
            ui.add(egui::Slider::new(&mut self.explosion, 0.0..=1.0).show_value(false));
            ui.checkbox(&mut self.isolate, "Isolate selection");
            ui.checkbox(&mut self.rulers, "Scale");
        });
        ui.horizontal_wrapped(|ui| {
            let asset = self.displayed_asset();
            let dimensions = asset
                .reference_dimensions
                .iter()
                .filter(|&&value| value > 0.0)
                .map(|value| format!("{value:.2}"))
                .collect::<Vec<_>>()
                .join(" x ");
            ui.small(format!("{}: {dimensions} m", asset.dimension_label));
            ui.add(egui::Hyperlink::from_label_and_url(
                egui::RichText::new("Dimensions source").small(),
                asset.dimension_source,
            ));
            if self.rulers {
                ui.small("Approx. scale (metres)");
            }
        });
        ui.separator();
        let area = ui.available_rect_before_wrap();
        let footer_height = 26.0;
        let body = egui::Rect::from_min_max(
            area.min,
            egui::pos2(
                area.right(),
                (area.bottom() - footer_height).max(area.top()),
            ),
        );
        let (scene, inspector, stats) = if comparing {
            let (scene, stats) = comparison_layout(body);
            (scene, None, Some(stats))
        } else {
            let (scene, inspector) = view_layout(body, self.parts_open);
            (scene, inspector, None)
        };
        if let Some((_, parts)) = inspector {
            ui.scope_builder(
                egui::UiBuilder::new()
                    .max_rect(parts)
                    .id_salt("satellite_parts"),
                |ui| self.parts_list(ui),
            );
        }
        let mut deferred_scene = None;
        if scene.width() > 2.0 && scene.height() > 2.0 {
            deferred_scene = ui
                .scope_builder(
                    egui::UiBuilder::new()
                        .max_rect(scene)
                        .id_salt("satellite_scene"),
                    |ui| self.scene(ui, render, scene, comparison, toolbar_camera, reset_camera),
                )
                .inner;
            if self.return_to_table {
                let ruler_margin =
                    if self.rulers && scene.width() >= 180.0 && scene.height() >= 150.0 {
                        64.0
                    } else {
                        0.0
                    };
                let rect = egui::Rect::from_min_size(
                    scene.min + egui::vec2(ruler_margin + 8.0, 8.0),
                    egui::vec2(76.0, 26.0),
                );
                if ui
                    .put(
                        rect,
                        egui::Button::image_and_text(
                            egui::Image::new(egui::include_image!(
                                "../assets/lucide-arrow-left.svg"
                            ))
                            .fit_to_exact_size(egui::vec2(14.0, 14.0))
                            .tint(ui.visuals().text_color()),
                            "Back",
                        ),
                    )
                    .on_hover_text("Back to satellite table")
                    .clicked()
                {
                    self.model_action = Some(navigation::Action::Back);
                }
            }
        }
        if let Some((preview, _)) = inspector {
            let keep_preview =
                ui.input(|i| i.pointer.hover_pos().is_some_and(|p| preview.contains(p)));
            self.inspected =
                self.hovered
                    .or(self.selected)
                    .or(if keep_preview { self.inspected } else { None });
            ui.scope_builder(
                egui::UiBuilder::new()
                    .max_rect(preview)
                    .id_salt("satellite_part_preview"),
                |ui| self.part_preview(ui, render),
            );
        } else {
            self.preview_renderer = None;
        }
        if let Some(stats) = stats {
            ui.scope_builder(
                egui::UiBuilder::new()
                    .max_rect(stats)
                    .id_salt("satellite_comparison_stats"),
                |ui| {
                    ui.set_clip_rect(ui.clip_rect().intersect(stats));
                    egui::ScrollArea::vertical()
                        .max_height(stats.height())
                        .auto_shrink([false, false])
                        .show(ui, |ui| self.statistics(ui));
                },
            );
        }
        let footer =
            egui::Rect::from_min_max(egui::pos2(area.left(), body.bottom() + 5.0), area.max);
        ui.scope_builder(egui::UiBuilder::new().max_rect(footer), |ui| {
            if let Some(error) = self.errors.get(&self.asset).cloned() {
                ui.horizontal(|ui| {
                    ui.colored_label(
                        ui.visuals().error_fg_color,
                        format!("Cannot load {}", ASSETS[self.asset].name),
                    )
                    .on_hover_text(error);
                    if ui.button("Retry").clicked() {
                        self.errors.remove(&self.asset);
                        self.request_model(self.asset, ui.ctx());
                    }
                });
                return;
            }
            if self.model.is_none() || self.loaded_asset != self.asset {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(format!("Loading {}", ASSETS[self.asset].name));
                });
                return;
            }
            ui.horizontal_wrapped(|ui| {
                ui.small(self.displayed_asset().credit);
                ui.hyperlink_to("Model source", self.displayed_asset().source);
            });
        });
        ui.advance_cursor_after_rect(area);
        deferred_scene
    }

    fn parts_list(&mut self, ui: &mut egui::Ui) {
        let Some(model) = &self.model else {
            return;
        };
        ui.strong(format!("Components ({} total)", model.components.len()));
        let label_font = bold_font(ui);
        let area = ui.available_rect_before_wrap();
        let rows = egui::Rect::from_min_max(
            area.min,
            egui::pos2(area.right(), (area.bottom() - 20.0).max(area.top())),
        );
        ui.scope_builder(egui::UiBuilder::new().max_rect(rows), |ui| {
            ui.set_clip_rect(ui.clip_rect().intersect(rows));
            egui::ScrollArea::vertical()
                .id_salt("parts_scroll")
                .max_height(rows.height())
                .auto_shrink([false, false])
                .show_rows(ui, 24.0, model.components.len(), |ui, range| {
                    for index in range {
                        let name = model.components[index].info.title;
                        let selected = self.selected == Some(index);
                        let button = egui::Button::selectable(
                            selected,
                            egui::RichText::new(name).font(label_font.clone()),
                        )
                        .wrap_mode(egui::TextWrapMode::Truncate)
                        .min_size(egui::vec2(ui.available_width(), 24.0));
                        let response = ui
                            .add(button)
                            .on_hover_cursor(egui::CursorIcon::PointingHand);
                        if response.hovered()
                            && ui.input(|i| i.pointer.hover_pos().is_some())
                            && (!self.isolate
                                || self.selected.is_none()
                                || self.selected == Some(index))
                        {
                            self.hovered = Some(index);
                        }
                        if response.clicked() {
                            self.selected = Some(index);
                            self.hovered = Some(index);
                        }
                    }
                });
        });
        ui.painter().text(
            egui::pos2(area.left(), area.bottom() - 2.0),
            egui::Align2::LEFT_BOTTOM,
            "Simulation: not connected",
            egui::FontId::proportional(10.0),
            ui.visuals().weak_text_color(),
        );
    }

    fn part_preview(&mut self, ui: &mut egui::Ui, render: Option<&RenderState>) {
        let area = ui.available_rect_before_wrap();
        if area.width() < 2.0 || area.height() < 2.0 {
            return;
        }
        let asset = self.displayed_asset();
        ui.horizontal(|ui| {
            if self.inspected.is_some() {
                ui.small(if self.inspected == self.selected {
                    "Selected part"
                } else {
                    "Part details"
                });
            } else {
                ui.add(egui::Label::new(egui::RichText::new(asset.name).strong()).truncate());
                ui.label(egui::RichText::new("[?]").weak())
                    .on_hover_cursor(egui::CursorIcon::Help)
                    .on_hover_text(asset.facts.description);
            }
        });
        let image_height = (area.height() * 0.45).min(155.0);
        let (image_rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), image_height),
            egui::Sense::hover(),
        );
        if let (Some(render), Some(main)) = (render, self.renderer.as_ref()) {
            if self.preview_renderer.is_none() {
                self.preview_renderer = Some(main.preview());
            }
            let part = self.inspected.map(|index| &main.model.components[index]);
            let camera = Camera {
                target: part.map_or_else(Vector3::zeros, |part| part.center),
                zoom: 1.0,
                ..self.camera
            };
            let (matrix, eye) = camera.matrices_for_bounds(
                image_rect.aspect_ratio(),
                part.map_or(1.12, |part| (part.max - part.min).norm() * 0.56),
            );
            let uniform = Uniform {
                view_projection: matrix.into(),
                eye: [eye.x, eye.y, eye.z, 1.0],
                options: [
                    0.0,
                    self.inspected.map_or(-1.0, |index| index as f32),
                    f32::from(self.inspected.is_some()),
                    -1.0,
                ],
            };
            let preview = self.preview_renderer.as_mut().unwrap();
            preview.draw(
                target_size(
                    image_rect.size(),
                    ui.ctx().pixels_per_point(),
                    render.device.limits().max_texture_dimension_2d,
                ),
                uniform,
                ui.visuals().dark_mode,
            );
            ui.painter().image(
                preview.target.as_ref().unwrap().id,
                image_rect,
                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }
        egui::ScrollArea::vertical()
            .id_salt(if self.inspected.is_some() {
                "part_description"
            } else {
                "satellite_stats"
            })
            .max_height(ui.available_height())
            .show(ui, |ui| {
                if let Some((index, model)) = self.inspected.zip(self.model.as_ref()) {
                    show_part_info(ui, asset, model, index);
                } else {
                    self.statistics(ui);
                }
            });
    }

    fn statistics(&self, ui: &mut egui::Ui) {
        if let Some(satellite) = self.catalogue_selection.and_then(|id| {
            crate::satellite_catalogue::catalogue()
                .satellites
                .iter()
                .find(|satellite| {
                    satellite.id == id && satellite.model_index() == Some(self.loaded_asset)
                })
        }) {
            comparison::show_compact_stats(ui, satellite);
        } else {
            show_stats(ui, self.displayed_asset());
        }
    }

    fn scene(
        &mut self,
        ui: &mut egui::Ui,
        render: Option<&RenderState>,
        rect: egui::Rect,
        comparison: Option<navigation::ComparisonCamera>,
        toolbar_camera: Camera,
        mut reset_camera: bool,
    ) -> Option<Scene> {
        let bounds = rect;
        let ruler_visible = self.rulers && rect.width() >= 180.0 && rect.height() >= 150.0;
        let ruler_margin = if ruler_visible { 64.0 } else { 0.0 };
        let rect = egui::Rect::from_min_max(
            bounds.min + egui::vec2(ruler_margin, 0.0),
            bounds.max - egui::vec2(0.0, if ruler_visible { 28.0 } else { 0.0 }),
        );
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        let before = if let (Some(linked), Some(model)) = (comparison, self.model.as_ref()) {
            let units = ASSETS[self.loaded_asset].metres_per_unit(model);
            let camera = linked.camera_for(rect, self.explosion, units);
            let toolbar = self.camera;
            self.camera = Camera {
                yaw: camera.yaw + (toolbar.yaw - toolbar_camera.yaw),
                pitch: camera.pitch + (toolbar.pitch - toolbar_camera.pitch),
                zoom: camera.zoom * (toolbar.zoom / toolbar_camera.zoom),
                target: camera.target + (toolbar.target - toolbar_camera.target),
            };
            camera
        } else {
            self.camera
        };
        if self.spin && comparison.is_none() {
            self.camera.yaw += ui.input(|i| i.stable_dt.min(0.05)) * 0.22;
            ui.ctx().request_repaint();
        }
        let rotation_zoom = comparison.map_or(self.camera.zoom, |linked| linked.zoom());
        let gesture = response
            .hovered()
            .then(|| crate::drawing::camera_gesture_input(ui));
        let zooming = crate::drawing::camera_zoom_active(ui, gesture, true);
        if let Some((scroll, zoom, _)) = gesture {
            if zoom != 1.0 {
                self.camera.zoom_by(zoom);
            } else if !zooming {
                self.camera.orbit(scroll * 0.5, rotation_zoom);
            }
        }
        if !zooming && response.dragged_by(egui::PointerButton::Primary) {
            let delta = ui.input(|i| i.pointer.delta());
            self.camera.orbit(delta, rotation_zoom);
        }
        if !zooming && response.dragged_by(egui::PointerButton::Secondary) {
            let delta = ui.input(|i| i.pointer.delta());
            let right = Vector3::new(self.camera.yaw.cos(), 0.0, -self.camera.yaw.sin());
            let up = Vector3::new(
                -self.camera.yaw.sin() * self.camera.pitch.sin(),
                self.camera.pitch.cos(),
                -self.camera.yaw.cos() * self.camera.pitch.sin(),
            );
            self.camera.target +=
                (-right * delta.x + up * delta.y) * (2.5 / rect.height() / self.camera.zoom);
        }
        if response.double_clicked() {
            self.camera = Camera::default();
            reset_camera = true;
        }
        let Some(model) = &self.model else {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Loading model...",
                egui::FontId::proportional(15.0),
                ui.visuals().weak_text_color(),
            );
            return None;
        };
        let mut pointed = None;
        if (response.hovered() || response.clicked()) && !response.dragged() {
            if let Some((origin, direction)) = ui
                .input(|i| i.pointer.hover_pos())
                .and_then(|point| self.camera.ray(rect, point, self.explosion))
            {
                let query = PickQuery {
                    origin,
                    direction,
                    explosion: self.explosion,
                    isolate: if self.isolate { self.selected } else { None },
                };
                pointed = if let Some((last, picked)) = &self.picking {
                    if *last == query {
                        *picked
                    } else {
                        model.pick(origin, direction, self.explosion, query.isolate)
                    }
                } else {
                    model.pick(origin, direction, self.explosion, query.isolate)
                };
                self.picking = Some((query, pointed));
                self.hovered = pointed;
            }
        }
        if response.clicked() && !response.double_clicked() {
            self.selected = pointed;
        }
        if let Some(index) = pointed {
            egui::Tooltip::always_open(
                ui.ctx().clone(),
                ui.layer_id(),
                response.id.with("part_cursor"),
                egui::PopupAnchor::Pointer,
            )
            .gap(12.0)
            .width(260.0)
            .show(|ui| {
                ui.set_max_width(260.0);
                let component = &model.components[index];
                ui.strong(component.info.title);
            });
            response.on_hover_cursor(egui::CursorIcon::PointingHand);
        }
        let Some(render) = render else {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "GPU renderer unavailable",
                egui::FontId::proportional(15.0),
                ui.visuals().error_fg_color,
            );
            return None;
        };
        let scene = Scene {
            ui: ui.new_child(egui::UiBuilder::new().max_rect(bounds)),
            bounds,
            rect,
            ruler_visible,
            image: ui.painter().add(egui::Shape::Noop),
            camera_changed: self.camera != before,
            reset_camera,
        };
        if comparison.is_some() {
            Some(scene)
        } else {
            self.draw_scene(render, &scene);
            None
        }
    }

    fn draw_scene(&mut self, render: &RenderState, scene: &Scene) {
        let Some(model) = &self.model else {
            return;
        };
        let Scene {
            ui,
            rect,
            bounds,
            ruler_visible,
            image,
            ..
        } = scene;
        if self.renderer.is_none() {
            self.renderer = Some(Renderer::new(render, Arc::clone(model)));
        }
        let renderer = self.renderer.as_mut().unwrap();
        let size = target_size(
            rect.size(),
            ui.ctx().pixels_per_point(),
            render.device.limits().max_texture_dimension_2d,
        );
        let (matrix, eye) = self.camera.matrices(rect.aspect_ratio(), self.explosion);
        let uniform = Uniform {
            view_projection: matrix.into(),
            eye: [eye.x, eye.y, eye.z, 1.0],
            options: [
                self.explosion,
                self.selected.map_or(-1.0, |i| i as f32),
                f32::from(self.isolate && self.selected.is_some()),
                self.hovered.or(self.selected).map_or(-1.0, |i| i as f32),
            ],
        };
        renderer.draw(size, uniform, ui.visuals().dark_mode);
        ui.painter().set(
            *image,
            egui::Shape::image(
                renderer.target.as_ref().unwrap().id,
                *rect,
                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            ),
        );
        if *ruler_visible {
            draw_scale(
                ui,
                *bounds,
                *rect,
                self.camera.metres_per_point(
                    *rect,
                    self.explosion,
                    ASSETS[self.loaded_asset].metres_per_unit(model),
                ),
            );
        }
    }
}

#[path = "satellite_comparison.rs"]
mod comparison;
#[path = "satellite_navigation.rs"]
mod navigation;
#[path = "satellite_sort.rs"]
mod sort;

fn bold_font(ui: &egui::Ui) -> egui::FontId {
    let mut font = egui::TextStyle::Body.resolve(ui.style());
    let family = egui::FontFamily::Name("satellite-bold".into());
    if ui.fonts_mut(|fonts| fonts.definitions().families.contains_key(&family)) {
        font.family = family;
    } else {
        ui.ctx().add_font(egui::epaint::text::FontInsert::new(
            "satellite-bold",
            egui::FontData::from_static(include_bytes!("../assets/Ubuntu-Bold.ttf")),
            vec![egui::epaint::text::InsertFontFamily {
                family,
                priority: egui::epaint::text::FontPriority::Highest,
            }],
        ));
        ui.ctx().request_discard("Install satellite label font");
    }
    font
}

fn show_stats(ui: &mut egui::Ui, asset: &crate::satellite_model::Asset) {
    show_stat_rows(ui, asset.facts.rows);
    ui.separator();
    ui.horizontal_wrapped(|ui| {
        for &(name, url) in asset.facts.sources {
            ui.add(egui::Hyperlink::from_label_and_url(
                egui::RichText::new(name).text_style(egui::TextStyle::Body),
                url,
            ));
        }
    });
}

fn show_stat_rows(ui: &mut egui::Ui, rows: &[(&str, &str)]) {
    let label_font = bold_font(ui);
    let key_width = rows
        .iter()
        .map(|(key, _)| {
            ui.painter()
                .layout_no_wrap(
                    (*key).to_owned(),
                    label_font.clone(),
                    ui.visuals().text_color(),
                )
                .size()
                .x
        })
        .fold(0.0_f32, f32::max)
        .ceil()
        .min(ui.available_width() * 0.45);
    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
    ui.spacing_mut().item_spacing.y = 2.0;
    ui.spacing_mut().interact_size.y = ui.text_style_height(&egui::TextStyle::Body);
    for &(key, value) in rows {
        ui.horizontal(|ui| {
            let height = ui.text_style_height(&egui::TextStyle::Body);
            ui.allocate_ui_with_layout(
                egui::vec2(key_width, height),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    ui.set_min_width(key_width);
                    ui.add(
                        egui::Label::new(egui::RichText::new(key).font(label_font.clone()))
                            .truncate(),
                    )
                },
            )
            .inner
            .on_hover_text(key);
            ui.add(
                egui::Label::new(egui::RichText::new(value).text_style(egui::TextStyle::Body))
                    .truncate()
                    .halign(egui::Align::Min),
            )
            .on_hover_text(value);
        });
    }
}

fn show_part_info(
    ui: &mut egui::Ui,
    asset: &crate::satellite_model::Asset,
    model: &Model,
    index: usize,
) {
    ui.set_max_width(ui.available_width().min(300.0));
    let component = &model.components[index];
    ui.strong(component.info.title);
    if component.visual_identification {
        ui.small("Visual identification");
    }
    ui.label(component.info.description);
    ui.hyperlink_to("Component reference", component.info.source);
    egui::CollapsingHeader::new("Source meshes")
        .id_salt(index)
        .show(ui, |ui| {
            ui.small(component.info.review);
            ui.separator();
            for &mesh in &component.meshes {
                let part = &model.parts[mesh];
                ui.add(egui::Label::new(egui::RichText::new(&part.name).small()).wrap());
                ui.small(format!(
                    "{}; {} triangles",
                    model.materials[part.material].name,
                    (part.end - part.start) / 3
                ));
            }
            ui.hyperlink_to("Original model", asset.source);
            ui.hyperlink_to("Spacecraft documentation", asset.documentation);
        });
}

fn scale_length(target_metres: f32) -> f32 {
    let magnitude = 10.0_f32.powf(target_metres.log10().floor());
    let mantissa = target_metres / magnitude;
    magnitude
        * if mantissa >= 5.0 {
            5.0
        } else if mantissa >= 2.0 {
            2.0
        } else {
            1.0
        }
}

fn draw_scale(ui: &egui::Ui, bounds: egui::Rect, scene: egui::Rect, metres_per_point: f32) {
    if !metres_per_point.is_finite() || metres_per_point <= 0.0 {
        return;
    }
    let metres = scale_length(120.0 * metres_per_point);
    let spacing = metres / metres_per_point;
    let painter = ui.painter().with_clip_rect(bounds);
    let color = ui.visuals().text_color();
    let stroke = egui::Stroke::new(1.0_f32, color);
    let font = egui::FontId::monospace(11.0);
    let x_unit = painter.text(
        egui::pos2(scene.right(), scene.bottom() + 13.0),
        egui::Align2::RIGHT_TOP,
        "X (m)",
        font.clone(),
        color,
    );
    let y_unit = painter.text(
        egui::pos2(scene.left() - 14.0, scene.top()),
        egui::Align2::RIGHT_TOP,
        "Y (m)",
        font.clone(),
        color,
    );
    for direction in [egui::vec2(1.0, 0.0), egui::vec2(0.0, -1.0)] {
        let mut last_label_right = f32::NEG_INFINITY;
        let horizontal = direction.x > 0.0;
        let origin = if horizontal {
            egui::pos2(scene.center().x, scene.bottom() + 7.0)
        } else {
            egui::pos2(scene.left() - 6.0, scene.center().y)
        };
        let half = if horizontal {
            scene.width()
        } else {
            scene.height()
        } * 0.5;
        painter.line_segment(
            [origin - direction * half, origin + direction * half],
            stroke,
        );
        let count = (half / spacing * 5.0).floor() as i32;
        for tick in -count..=count {
            let major = tick % 5 == 0;
            let p = origin + direction * spacing * tick as f32 / 5.0;
            let normal = egui::vec2(-direction.y, direction.x) * if major { 4.0 } else { 2.0 };
            painter.line_segment([p - normal, p + normal], stroke);
            if major {
                let value = tick as f32 / 5.0 * metres;
                let value = if value.abs() < metres * 0.01 {
                    0.0
                } else {
                    value
                };
                let digits = (-metres.log10().floor()).clamp(0.0, 6.0) as usize;
                let label = format!("{value:.digits$}");
                let (pos, align) = if horizontal {
                    (p + egui::vec2(0.0, 6.0), egui::Align2::CENTER_TOP)
                } else {
                    (p - egui::vec2(8.0, 0.0), egui::Align2::RIGHT_CENTER)
                };
                let galley = painter.layout_no_wrap(label, font.clone(), color);
                let label_rect = align.anchor_size(pos, galley.size());
                if bounds.contains_rect(label_rect)
                    && !label_rect.intersects(x_unit.expand(3.0))
                    && !label_rect.intersects(y_unit.expand(3.0))
                    && (!horizontal || label_rect.left() > last_label_right + 6.0)
                {
                    last_label_right = label_rect.right();
                    painter.galley(label_rect.min, galley, color);
                }
                let a = if horizontal {
                    egui::pos2(p.x, scene.top())
                } else {
                    egui::pos2(scene.left(), p.y)
                };
                let b = if horizontal {
                    egui::pos2(p.x, scene.bottom())
                } else {
                    egui::pos2(scene.right(), p.y)
                };
                painter.line_segment(
                    [a, b],
                    egui::Stroke::new(0.5_f32, color.gamma_multiply(0.10)),
                );
            }
        }
    }
}

fn target_size(size: egui::Vec2, pixels_per_point: f32, device_limit: u32) -> [u32; 2] {
    let physical = size * pixels_per_point;
    let limit = device_limit.min(4096) as f32;
    let scale = (limit / physical.max_elem().max(1.0)).min(1.0);
    [
        (physical.x * scale).round().max(1.0) as u32,
        (physical.y * scale).round().max(1.0) as u32,
    ]
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniform {
    view_projection: [[f32; 4]; 4],
    eye: [f32; 4],
    options: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialUniform {
    color: [f32; 4],
    surface: [f32; 4],
    normal: [f32; 4],
}

struct Target {
    texture: wgpu::Texture,
    multisample: wgpu::Texture,
    depth: wgpu::Texture,
    id: egui::TextureId,
}
struct Renderer {
    render: RenderState,
    model: Arc<Model>,
    pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    uniform: wgpu::Buffer,
    camera_group: wgpu::BindGroup,
    materials: Vec<wgpu::BindGroup>,
    target: Option<Target>,
    last_frame: Option<(Uniform, bool)>,
}

impl Drop for Renderer {
    fn drop(&mut self) {
        if let Some(target) = &self.target {
            self.render.renderer.write().free_texture(&target.id);
        }
    }
}

impl Renderer {
    fn preview(&self) -> Self {
        let device = &self.render.device;
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("satellite_part_camera"),
            size: std::mem::size_of::<Uniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        // Geometry and material bindings are immutable GPU handles shared with the main view.
        Self {
            render: self.render.clone(),
            model: Arc::clone(&self.model),
            pipeline: self.pipeline.clone(),
            transparent_pipeline: self.transparent_pipeline.clone(),
            vertices: self.vertices.clone(),
            indices: self.indices.clone(),
            uniform,
            camera_group,
            materials: self.materials.clone(),
            target: None,
            last_frame: None,
        }
    }

    fn new(render: &RenderState, model: Arc<Model>) -> Self {
        let device = &render.device;
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("satellite_camera"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let mut entries = Vec::new();
        for binding in 0..3 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("satellite_materials"),
            entries: &entries,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("satellite_camera"),
            size: std::mem::size_of::<Uniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("satellite_textures"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: 8,
            ..Default::default()
        });
        let fallback = image::RgbaImage::from_pixel(1, 1, image::Rgba([255; 4]));
        let mut textures = std::collections::HashMap::new();
        let mut materials = Vec::new();
        for material in &model.materials {
            for (index, srgb) in [
                (material.base_texture, true),
                (material.roughness_texture, false),
                (material.normal_texture, false),
            ] {
                textures.entry((index, srgb)).or_insert_with(|| {
                    upload_texture(
                        render,
                        index.map(|i| &model.images[i]).unwrap_or(&fallback),
                        srgb,
                    )
                });
            }
            let base = textures[&(material.base_texture, true)].create_view(&Default::default());
            let roughness =
                textures[&(material.roughness_texture, false)].create_view(&Default::default());
            let normal =
                textures[&(material.normal_texture, false)].create_view(&Default::default());
            let values = MaterialUniform {
                color: material.color,
                surface: [
                    material.metallic,
                    material.roughness,
                    material.alpha_cutoff,
                    f32::from(material.blend),
                ],
                normal: [
                    material.normal_scale,
                    f32::from(material.double_sided),
                    material.appearance as u32 as f32,
                    material.rim_inner_radius(),
                ],
            };
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("satellite_material"),
                contents: bytemuck::bytes_of(&values),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            materials.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &material_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&base),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&roughness),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&normal),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: buffer.as_entire_binding(),
                    },
                ],
            }));
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("satellite_model"),
            source: wgpu::ShaderSource::Wgsl(include_str!("satellite_view.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&camera_layout, &material_layout],
            push_constant_ranges: &[],
        });
        let attributes = wgpu::vertex_attr_array![0=>Float32x3,1=>Float32x3,2=>Float32x2,3=>Float32x2,4=>Float32x2,5=>Float32x4,6=>Float32x3,7=>Uint32];
        let mut pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("satellite_model"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attributes,
                }],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 4,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        };
        let pipeline = device.create_render_pipeline(&pipeline_descriptor);
        // Transparent texels must not occlude surfaces drawn later in the same mesh.
        pipeline_descriptor.label = Some("satellite_transparent");
        pipeline_descriptor
            .depth_stencil
            .as_mut()
            .unwrap()
            .depth_write_enabled = false;
        let transparent_pipeline = device.create_render_pipeline(&pipeline_descriptor);
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("satellite_vertices"),
            contents: bytemuck::cast_slice(&model.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("satellite_indices"),
            contents: bytemuck::cast_slice(&model.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            render: render.clone(),
            model,
            pipeline,
            transparent_pipeline,
            vertices,
            indices,
            uniform,
            camera_group,
            materials,
            target: None,
            last_frame: None,
        }
    }

    fn draw(&mut self, size: [u32; 2], uniform: Uniform, dark: bool) {
        let device = &self.render.device;
        if self
            .target
            .as_ref()
            .is_none_or(|t| [t.texture.width(), t.texture.height()] != size)
        {
            if let Some(target) = self.target.take() {
                self.render.renderer.write().free_texture(&target.id);
            }
            let make = |format, sample_count, usage| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("satellite_view"),
                    size: wgpu::Extent3d {
                        width: size[0],
                        height: size[1],
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: if format == wgpu::TextureFormat::Rgba8UnormSrgb {
                        &[wgpu::TextureFormat::Rgba8Unorm]
                    } else {
                        &[]
                    },
                })
            };
            let texture = make(
                wgpu::TextureFormat::Rgba8UnormSrgb,
                1,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            );
            let multisample = make(
                wgpu::TextureFormat::Rgba8UnormSrgb,
                4,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            );
            let depth = make(
                wgpu::TextureFormat::Depth32Float,
                4,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            );
            let id = self.render.renderer.write().register_native_texture(
                device,
                // egui samples encoded colour values, not sRGB-decoded values.
                &texture.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(wgpu::TextureFormat::Rgba8Unorm),
                    ..Default::default()
                }),
                wgpu::FilterMode::Linear,
            );
            self.target = Some(Target {
                texture,
                multisample,
                depth,
                id,
            });
            self.last_frame = None;
        }
        if self.last_frame.as_ref().is_some_and(|(last, last_dark)| {
            bytemuck::bytes_of(last) == bytemuck::bytes_of(&uniform) && *last_dark == dark
        }) {
            return;
        }
        self.render
            .queue
            .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        let target = self.target.as_ref().unwrap();
        let color_view = target.multisample.create_view(&Default::default());
        let resolve = target.texture.create_view(&Default::default());
        let depth_view = target.depth.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("satellite_view"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: Some(&resolve),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(if dark {
                            wgpu::Color {
                                r: 0.008,
                                g: 0.009,
                                b: 0.010,
                                a: 1.0,
                            }
                        } else {
                            wgpu::Color::WHITE
                        }),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_bind_group(0, &self.camera_group, &[]);
            pass.set_vertex_buffer(0, self.vertices.slice(..));
            pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
            let mut parts: Vec<_> = self
                .model
                .parts
                .iter()
                .enumerate()
                .filter(|(_, part)| {
                    part.is_visible(&self.model.materials)
                        && (uniform.options[2] <= 0.5
                            || part.component as f32 == uniform.options[1])
                })
                .map(|(_, part)| part)
                .collect();
            parts.sort_by(|a, b| {
                let blend_a = self.model.materials[a.material].is_blended();
                let blend_b = self.model.materials[b.material].is_blended();
                blend_a.cmp(&blend_b).then_with(|| {
                    let eye = Vector3::new(uniform.eye[0], uniform.eye[1], uniform.eye[2]);
                    let distance = |p: &crate::satellite_model::Part| {
                        (p.center + p.offset * uniform.options[0] - eye).norm_squared()
                    };
                    if blend_a {
                        distance(b).total_cmp(&distance(a))
                    } else {
                        std::cmp::Ordering::Equal
                    }
                })
            });
            for part in parts {
                pass.set_pipeline(if self.model.materials[part.material].is_blended() {
                    &self.transparent_pipeline
                } else {
                    &self.pipeline
                });
                pass.set_bind_group(1, &self.materials[part.material], &[]);
                pass.draw_indexed(part.start..part.end, 0, 0..1);
            }
        }
        self.render.queue.submit([encoder.finish()]);
        self.last_frame = Some((uniform, dark));
    }
}

fn upload_texture(render: &RenderState, image: &image::RgbaImage, srgb: bool) -> wgpu::Texture {
    let limit = render.device.limits().max_texture_dimension_2d.min(4096);
    let scale = (limit as f32 / image.width().max(image.height()) as f32).min(1.0);
    let mut image = if scale < 1.0 {
        image::imageops::resize(
            image,
            (image.width() as f32 * scale).max(1.0) as u32,
            (image.height() as f32 * scale).max(1.0) as u32,
            image::imageops::FilterType::Triangle,
        )
    } else {
        image.clone()
    };
    let levels = image.width().max(image.height()).ilog2() + 1;
    let texture = render.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("spacecraft_material"),
        size: wgpu::Extent3d {
            width: image.width(),
            height: image.height(),
            depth_or_array_layers: 1,
        },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: if srgb {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        },
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for level in 0..levels {
        render.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            image.as_raw(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(image.width() * 4),
                rows_per_image: Some(image.height()),
            },
            wgpu::Extent3d {
                width: image.width(),
                height: image.height(),
                depth_or_array_layers: 1,
            },
        );
        if level + 1 < levels {
            image = image::imageops::resize(
                &image,
                (image.width() / 2).max(1),
                (image.height() / 2).max(1),
                image::imageops::FilterType::Triangle,
            );
        }
    }
    texture
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalogue_model(index: usize) -> Arc<Model> {
        Arc::new(
            ASSETS[index]
                .load(&std::fs::read(crate::texture::asset_path(ASSETS[index].path)).unwrap())
                .unwrap(),
        )
    }

    #[test]
    fn auto_rotation_starts_enabled_and_toggles_without_changing_the_camera() {
        let mut view = View::default();
        let camera = view.camera;
        assert!(view.spin);
        view.toggle_auto_rotate();
        assert!(!view.spin);
        view.toggle_auto_rotate();
        assert!(view.spin);
        assert_eq!(view.camera.yaw, camera.yaw);
        assert_eq!(view.camera.zoom, camera.zoom);
    }

    #[test]
    fn switching_models_preserves_physical_scale_even_across_cubesats_and_large_satellites() {
        let mut view = View {
            model: Some(catalogue_model(0)),
            spin: false,
            explosion: 0.8,
            camera: Camera {
                zoom: 1.8,
                target: Vector3::new(0.1, -0.2, 0.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(920.0, 700.0));
        let units = ASSETS[0].metres_per_unit(view.model.as_ref().unwrap());
        let scale = view.camera.metres_per_point(rect, view.explosion, units);
        let target = view.camera.target * units;
        let yaw = view.camera.yaw;
        let pitch = view.camera.pitch;
        for index in [5, 6, 1, 0] {
            view.asset = index;
            view.install_model(catalogue_model(index));
            let units = ASSETS[index].metres_per_unit(view.model.as_ref().unwrap());
            let after = view.camera.metres_per_point(rect, view.explosion, units);
            assert!(
                (after / scale - 1.0).abs() < 1e-5,
                "{}: scale changed",
                ASSETS[index].name
            );
            assert!((view.camera.target * units - target).norm() < 1e-5);
            assert_eq!(view.camera.yaw, yaw);
            assert_eq!(view.camera.pitch, pitch);
            let zoom = view.camera.zoom;
            view.camera.zoom_by(1.01);
            assert!(
                (view.camera.zoom / zoom - 1.01).abs() < 1e-5,
                "zoom must not snap after a model switch"
            );
            view.camera.zoom_by(1.0 / 1.01);
        }
    }

    #[test]
    fn stale_load_completions_do_not_replace_the_requested_satellite() {
        let ctx = egui::Context::default();
        let original = catalogue_model(0);
        let mut view = View {
            model: Some(Arc::clone(&original)),
            spin: false,
            asset: 5,
            ..Default::default()
        };
        let pending: LoadResult = Arc::new(Mutex::new(None));
        view.pending.insert(5, Arc::clone(&pending));
        view.load(&ctx);
        assert!(Arc::ptr_eq(view.model.as_ref().unwrap(), &original));
        view.asset = 0;
        *pending.lock().unwrap() = Some(
            ASSETS[5].load(&std::fs::read(crate::texture::asset_path(ASSETS[5].path)).unwrap()),
        );
        view.load(&ctx);
        assert_eq!(view.loaded_asset, 0);
        assert!(Arc::ptr_eq(view.model.as_ref().unwrap(), &original));
        assert!(view.models.contains_key(&5));
        assert!(view.pending.is_empty());
        view.asset = 5;
        view.load(&ctx);
        assert_eq!(view.loaded_asset, 5);
        assert!(
            view.pending.is_empty(),
            "revisiting a loaded model must use the cache"
        );
    }

    #[test]
    fn upright_camera_never_rolls_or_flips() {
        for yaw in [-5.0, 0.0, 5.0] {
            for pitch in [-1.5, 0.0, 1.5] {
                let camera = Camera {
                    yaw,
                    pitch,
                    ..Default::default()
                };
                let (matrix, _) = camera.matrices(1.5, 0.0);
                let center = matrix * Vector4::new(0.0, 0.0, 0.0, 1.0);
                let up = matrix * Vector4::new(0.0, 0.1, 0.0, 1.0);
                assert!((center.x / center.w - up.x / up.w).abs() < 1e-5);
                assert!(up.y / up.w > center.y / center.w);
            }
        }
    }

    #[test]
    fn layout_reserves_the_same_preview_and_list_space_for_any_part() {
        for size in [egui::vec2(1400.0, 800.0), egui::vec2(390.0, 720.0)] {
            let body = egui::Rect::from_min_size(egui::pos2(10.0, 70.0), size);
            let (scene, Some((preview, list))) = view_layout(body, true) else {
                panic!("inspector");
            };
            assert!(body.contains_rect(scene));
            assert!(body.contains_rect(preview));
            assert!(body.contains_rect(list));
            assert!(!scene.intersects(preview));
            assert!(preview.bottom() < list.top());
            assert!(preview.height() >= 300.0);
            assert!(list.height() >= 90.0);
        }
    }

    #[test]
    fn comparison_layout_reserves_only_the_model_and_statistics() {
        for size in [
            egui::vec2(1400.0, 800.0),
            egui::vec2(710.0, 800.0),
            egui::vec2(390.0, 240.0),
        ] {
            let body = egui::Rect::from_min_size(egui::pos2(10.0, 70.0), size);
            let (scene, stats) = comparison_layout(body);
            assert!(body.contains_rect(scene));
            assert!(body.contains_rect(stats));
            assert_eq!(scene.width(), body.width());
            assert_eq!(stats.width(), body.width());
            assert!(scene.height() > stats.height());
            assert!(!scene.intersects(stats));
        }
    }

    #[test]
    fn trackpad_scroll_rotates_and_pinches_only_zoom() {
        let ctx = egui::Context::default();
        let mut view = View {
            spin: false,
            ..Default::default()
        };
        let point = egui::pos2(650.0, 400.0);
        let frame = |view: &mut View, time, events| {
            let _ = ctx.run_ui(
                egui::RawInput {
                    time: Some(time),
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(800.0, 600.0),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| {
                    egui::CentralPanel::default().show_inside(ui, |ui| {
                        view.scene(
                            ui,
                            None,
                            ui.available_rect_before_wrap(),
                            None,
                            view.camera,
                            false,
                        )
                    });
                },
            );
        };
        let scroll = |phase| egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(20.0, 15.0),
            phase,
            modifiers: Default::default(),
        };
        frame(&mut view, 0.0, vec![egui::Event::PointerMoved(point)]);
        frame(
            &mut view,
            0.1,
            vec![
                scroll(egui::TouchPhase::Start),
                scroll(egui::TouchPhase::Move),
            ],
        );
        assert_eq!(view.camera.zoom, 1.0);
        assert_ne!(view.camera.yaw, Camera::default().yaw);
        assert_ne!(view.camera.pitch, Camera::default().pitch);
        let before = view.camera;
        frame(
            &mut view,
            0.2,
            vec![
                egui::Event::Zoom(1.25),
                egui::Event::Rotate(0.5),
                scroll(egui::TouchPhase::Move),
            ],
        );
        assert_eq!(view.camera.yaw, before.yaw);
        assert_eq!(view.camera.pitch, before.pitch);
        assert_eq!(view.camera.zoom, 1.25);
        frame(&mut view, 0.25, vec![scroll(egui::TouchPhase::Move)]);
        assert_eq!(view.camera.yaw, before.yaw);
        assert_eq!(view.camera.pitch, before.pitch);
        frame(
            &mut view,
            0.3,
            vec![
                scroll(egui::TouchPhase::Start),
                scroll(egui::TouchPhase::Move),
            ],
        );
        assert_ne!(view.camera.yaw, before.yaw);
        assert_eq!(view.camera.zoom, 1.25);
    }

    #[test]
    fn camera_fits_the_entire_model_in_portrait_and_landscape() {
        for aspect in [0.4, 0.75, 1.0, 1.8, 3.0] {
            for explosion in [0.0, 1.0] {
                let (matrix, _) = Camera::default().matrices(aspect, explosion);
                for direction in [
                    Vector3::x(),
                    -Vector3::x(),
                    Vector3::y(),
                    -Vector3::y(),
                    Vector3::z(),
                    -Vector3::z(),
                ] {
                    let point = matrix * (direction * (1.0 + explosion * 0.55)).push(1.0);
                    let projected = point.xyz() / point.w;
                    assert!(
                        projected.x.abs() < 1.0
                            && projected.y.abs() < 1.0
                            && (0.0..1.0).contains(&projected.z)
                    );
                }
            }
        }
    }

    #[test]
    fn camera_rays_and_zoom_are_bounded() {
        let mut camera = Camera::default();
        let rect = egui::Rect::from_min_size(egui::pos2(20.0, 40.0), egui::vec2(800.0, 600.0));
        let (eye, direction) = camera.ray(rect, rect.center(), 0.0).unwrap();
        assert!(direction.dot(&(-eye.normalize())) > 0.9999);
        camera.zoom_by(1e10);
        assert_eq!(camera.zoom, 10_000.0);
        camera.zoom_by(1e-20);
        assert_eq!(camera.zoom, 0.0001);
        assert_eq!(
            target_size(egui::vec2(2000.0, 1000.0), 3.0, 4096),
            [4096, 2048]
        );
    }

    #[test]
    fn reference_rulers_follow_zoom_without_claiming_a_model_axis() {
        for size in [egui::vec2(1000.0, 700.0), egui::vec2(390.0, 320.0)] {
            let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
            let mut camera = Camera::default();
            let initial = camera.metres_per_point(rect, 0.0, 3.0);
            camera.zoom_by(2.0);
            assert!((camera.metres_per_point(rect, 0.0, 3.0) * 2.0 - initial).abs() < 1e-6);
            camera.orbit(egui::vec2(40.0, 15.0), camera.zoom);
            assert!((camera.metres_per_point(rect, 0.0, 3.0) * 2.0 - initial).abs() < 1e-6);
            let metres = scale_length(120.0 * initial);
            let points = metres / initial;
            assert!((47.9..=120.1).contains(&points));
        }
        assert_eq!(scale_length(0.84), 0.5);
        assert_eq!(scale_length(2.8), 2.0);
        assert_eq!(scale_length(11.0), 10.0);
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "satellite_view_tests.rs"]
pub(crate) mod gpu_tests;
