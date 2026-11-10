//! Independently selectable model panes, arranged like the planet views.
use super::*;
use crate::satellite_catalogue::{catalogue, Satellite};

pub(super) struct Pane {
    pub id: u32,
    pub view: View,
}

pub(super) enum Action {
    Add,
    Remove,
    Back,
}

#[derive(Clone, Copy)]
pub(super) struct ComparisonCamera {
    yaw: f32,
    pitch: f32,
    target_metres: Vector3<f32>,
    metres_per_point: f32,
    reference_metres_per_point: f32,
    fit_pending: bool,
    layout_size: egui::Vec2,
    pane_count: usize,
}

impl ComparisonCamera {
    fn new(view: &View, layout_size: egui::Vec2, pane_count: usize) -> Self {
        let units = view.model.as_ref().map_or(1.0, |model| {
            ASSETS[view.loaded_asset].metres_per_unit(model)
        });
        Self {
            yaw: view.camera.yaw,
            pitch: view.camera.pitch,
            target_metres: view.camera.target * units,
            metres_per_point: 0.0,
            reference_metres_per_point: 0.0,
            fit_pending: true,
            layout_size,
            pane_count,
        }
    }

    pub(super) fn zoom(self) -> f32 {
        if self.metres_per_point > 0.0 {
            self.reference_metres_per_point / self.metres_per_point
        } else {
            1.0
        }
    }

    pub(super) fn camera_for(self, rect: egui::Rect, explosion: f32, units: f32) -> Camera {
        let mut camera = Camera {
            yaw: self.yaw,
            pitch: self.pitch,
            target: self.target_metres / units,
            zoom: 1.0,
        };
        if self.metres_per_point > 0.0 {
            camera.zoom = camera.metres_per_point(rect, explosion, units) / self.metres_per_point;
        }
        camera
    }

    fn update_from(&mut self, view: &View, scene: &Scene) {
        let Some(model) = &view.model else {
            return;
        };
        let units = ASSETS[view.loaded_asset].metres_per_unit(model);
        self.yaw = view.camera.yaw;
        self.pitch = view.camera.pitch;
        self.target_metres = view.camera.target * units;
        self.metres_per_point = view
            .camera
            .metres_per_point(scene.rect, view.explosion, units);
        if self.reference_metres_per_point <= 0.0 {
            self.reference_metres_per_point = self.metres_per_point;
        }
        self.fit_pending = false;
    }

    fn synchronize(&mut self, panes: &mut [Pane], scenes: &[Option<Scene>]) {
        if scenes.iter().flatten().any(|scene| scene.reset_camera) {
            self.yaw = Camera::default().yaw;
            self.pitch = Camera::default().pitch;
            self.target_metres = Vector3::zeros();
            self.metres_per_point = 0.0;
            self.reference_metres_per_point = 0.0;
            self.fit_pending = true;
        } else {
            for (pane, scene) in panes.iter().zip(scenes) {
                if let Some(scene) = scene {
                    if scene.camera_changed {
                        self.update_from(&pane.view, scene);
                    }
                }
            }
        }
        if self.fit_pending {
            let scale = panes
                .iter()
                .zip(scenes)
                .filter_map(|(pane, scene)| {
                    let model = pane.view.model.as_ref()?;
                    let rect = scene.as_ref()?.rect;
                    Some(Camera::default().metres_per_point(
                        rect,
                        pane.view.explosion,
                        ASSETS[pane.view.loaded_asset].metres_per_unit(model),
                    ))
                })
                .fold(0.0_f32, f32::max);
            if scale > 0.0 {
                self.metres_per_point = scale / self.zoom();
                self.reference_metres_per_point = scale;
            }
            self.fit_pending = !panes.iter().all(|pane| {
                pane.view.model.is_some() || pane.view.errors.contains_key(&pane.view.asset)
            });
        }
        for (pane, scene) in panes.iter_mut().zip(scenes) {
            if let (Some(model), Some(scene)) = (pane.view.model.as_ref(), scene) {
                pane.view.camera = self.camera_for(
                    scene.rect,
                    pane.view.explosion,
                    ASSETS[pane.view.loaded_asset].metres_per_unit(model),
                );
            }
        }
    }
}

fn modelled_satellites() -> impl Iterator<Item = &'static Satellite> {
    catalogue()
        .satellites
        .iter()
        .filter(|satellite| satellite.model_index().is_some())
}

fn pane_rects(area: egui::Rect, count: usize) -> Vec<egui::Rect> {
    let columns = count
        .min(3)
        .min((area.width() / 340.0).floor().max(1.0) as usize)
        .max(1);
    let rows = count.div_ceil(columns).max(1);
    let gap = 4.0;
    let width = ((area.width() - gap * columns.saturating_sub(1) as f32) / columns as f32).max(0.0);
    let height = ((area.height() - gap * rows.saturating_sub(1) as f32) / rows as f32).max(0.0);
    (0..count)
        .map(|index| {
            let min = area.min
                + egui::vec2(
                    (index % columns) as f32 * (width + gap),
                    (index / columns) as f32 * (height + gap),
                );
            egui::Rect::from_min_max(
                min,
                egui::pos2(
                    (min.x + width).min(area.right()),
                    (min.y + height).min(area.bottom()),
                ),
            )
        })
        .collect()
}

impl View {
    fn selected_satellite(&self) -> Option<&'static Satellite> {
        modelled_satellites()
            .find(|satellite| Some(satellite.id) == self.catalogue_selection)
            .or_else(|| {
                modelled_satellites().find(|satellite| satellite.model_index() == Some(self.asset))
            })
    }

    fn cycle_satellite(&mut self, forward: bool) {
        let satellites: Vec<_> = modelled_satellites().collect();
        if satellites.is_empty() {
            return;
        }
        let current = self.selected_satellite().and_then(|selected| {
            satellites
                .iter()
                .position(|satellite| satellite.id == selected.id)
        });
        let index = match current {
            Some(current) if forward => (current + 1) % satellites.len(),
            Some(current) => (current + satellites.len() - 1) % satellites.len(),
            None if forward => 0,
            None => satellites.len() - 1,
        };
        let satellite = satellites[index];
        self.catalogue_selection = Some(satellite.id);
        self.asset = satellite.model_index().unwrap();
    }

    fn ensure_model_panes(&mut self) {
        if !self.model_panes.is_empty() {
            return;
        }
        if let Some(satellite) = self
            .selected_satellite()
            .or_else(|| modelled_satellites().next())
        {
            self.add_model_pane(satellite);
        }
    }

    fn add_model_pane(&mut self, satellite: &Satellite) {
        let Some(asset) = satellite.model_index() else {
            return;
        };
        self.pane_counter += 1;
        self.model_panes.push(Pane {
            id: self.pane_counter,
            view: View {
                asset,
                catalogue_selection: Some(satellite.id),
                spin: self.spin,
                return_to_table: self.return_to_table,
                ..Default::default()
            },
        });
    }

    fn add_next_model_pane(&mut self) {
        let next = modelled_satellites()
            .find(|satellite| {
                !self
                    .model_panes
                    .iter()
                    .any(|pane| Some(pane.view.asset) == satellite.model_index())
            })
            .or_else(|| modelled_satellites().next());
        if let Some(satellite) = next {
            self.add_model_pane(satellite);
        }
    }

    fn remove_model_pane(&mut self, id: u32) {
        if self.model_panes.len() > 1 {
            self.model_panes.retain(|pane| pane.id != id);
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui, render: Option<&RenderState>) -> bool {
        self.comparison_renderers.clear();
        if let Some(id) = self.pending_pane_removal.take() {
            self.remove_model_pane(id);
        }
        self.poll_loads();
        if let Some(model) = &self.model {
            self.models
                .entry(self.loaded_asset)
                .or_insert_with(|| Arc::clone(model));
        }
        self.ensure_model_panes();
        let area = ui.available_rect_before_wrap();
        let rects = pane_rects(area, self.model_panes.len());
        let mut panes = std::mem::take(&mut self.model_panes);
        let can_close = panes.len() > 1;
        let comparing = can_close;
        let comparison = if comparing {
            let camera = self.comparison_camera.get_or_insert_with(|| {
                ComparisonCamera::new(&panes[0].view, area.size(), panes.len())
            });
            if camera.layout_size != area.size() || camera.pane_count != panes.len() {
                camera.layout_size = area.size();
                camera.pane_count = panes.len();
                camera.fit_pending = true;
            }
            if self.spin {
                camera.yaw += ui.input(|i| i.stable_dt.min(0.05)) * 0.22;
                ui.ctx().request_repaint();
            }
            Some(*camera)
        } else {
            self.comparison_camera = None;
            None
        };
        let mut scenes = Vec::with_capacity(panes.len());
        let mut add = false;
        let mut remove = None;
        let mut back = false;
        for (pane, rect) in panes.iter_mut().zip(&rects) {
            for (&index, model) in &self.models {
                pane.view
                    .models
                    .entry(index)
                    .or_insert_with(|| Arc::clone(model));
            }
            pane.view.allow_pane_close = can_close;
            pane.view.return_to_table = self.return_to_table;
            if comparing {
                pane.view.spin = self.spin;
            }
            let scene = ui
                .scope_builder(
                    egui::UiBuilder::new()
                        .max_rect(*rect)
                        .id_salt(("satellite_model_pane", pane.id)),
                    |ui| {
                        ui.set_clip_rect(ui.clip_rect().intersect(*rect));
                        pane.view.show_model(ui, render, comparison)
                    },
                )
                .inner;
            scenes.push(scene);
            for (&index, model) in &pane.view.models {
                self.models
                    .entry(index)
                    .or_insert_with(|| Arc::clone(model));
            }
            match pane.view.model_action.take() {
                Some(Action::Add) => add = true,
                Some(Action::Remove) => remove = Some(pane.id),
                Some(Action::Back) => back = true,
                None => {}
            }
        }
        if let Some(mut camera) = comparison {
            // Collect all input before drawing any pane, so gestures in a later
            // pane update every model and ruler in this same frame.
            camera.synchronize(&mut panes, &scenes);
            self.comparison_camera = Some(camera);
            if let Some(spin) = panes
                .iter()
                .map(|pane| pane.view.spin)
                .find(|&spin| spin != self.spin)
            {
                self.spin = spin;
            }
            for (pane, scene) in panes.iter_mut().zip(&scenes) {
                pane.view.spin = self.spin;
                if let (Some(render), Some(scene)) = (render, scene) {
                    pane.view.draw_scene(render, scene);
                }
            }
        }
        let stroke = ui.visuals().widgets.noninteractive.bg_stroke;
        for rect in &rects {
            if rect.right() < area.right() - 1.0 {
                ui.painter()
                    .vline(rect.right() + 2.0, rect.y_range(), stroke);
            }
            if rect.bottom() < area.bottom() - 1.0 && rect.left() == area.left() {
                ui.painter()
                    .hline(area.x_range(), rect.bottom() + 2.0, stroke);
            }
        }
        if let Some(first) = panes.first() {
            self.spin = first.view.spin;
        }
        self.model_panes = panes;
        if let Some(id) = remove {
            // The current paint output still references this pane's GPU textures.
            self.pending_pane_removal = Some(id);
            ui.ctx().request_repaint();
        }
        if add {
            self.add_next_model_pane();
        }
        ui.advance_cursor_after_rect(area);
        back
    }

    pub(super) fn model_selector(&mut self, ui: &mut egui::Ui) {
        crate::settings::reserve_sidebar_toggle(ui);
        let name = self
            .selected_satellite()
            .map_or(ASSETS[self.asset].name, |satellite| satellite.name.as_str());
        egui::ComboBox::from_id_salt("satellite_model")
            .width(160.0_f32.min((ui.available_width() - 100.0).max(80.0)))
            .wrap_mode(egui::TextWrapMode::Wrap)
            .selected_text(egui::RichText::new(name).heading())
            .show_ui(ui, |ui| {
                for satellite in modelled_satellites() {
                    if ui
                        .selectable_value(
                            &mut self.catalogue_selection,
                            Some(satellite.id),
                            &satellite.name,
                        )
                        .clicked()
                    {
                        self.asset = satellite.model_index().unwrap();
                    }
                }
            });
        let can_cycle = modelled_satellites().take(2).count() > 1;
        for (svg, tooltip, forward) in [
            (
                egui::include_image!("../assets/lucide-chevron-left.svg"),
                "Previous satellite",
                false,
            ),
            (
                egui::include_image!("../assets/lucide-chevron-right.svg"),
                "Next satellite",
                true,
            ),
        ] {
            if ui
                .add_enabled(
                    can_cycle,
                    egui::Button::image(
                        egui::Image::new(svg)
                            .fit_to_exact_size(egui::vec2(14.0, 14.0))
                            .tint(ui.visuals().text_color()),
                    )
                    .min_size(egui::vec2(24.0, 24.0)),
                )
                .on_hover_text(tooltip)
                .clicked()
            {
                self.cycle_satellite(forward);
            }
        }
        if ui
            .small_button("+")
            .on_hover_text("Add satellite view")
            .clicked()
        {
            self.model_action = Some(Action::Add);
        }
        if self.allow_pane_close
            && ui
                .add(egui::Button::image(
                    egui::Image::new(egui::include_image!("../assets/lucide-x.svg"))
                        .fit_to_exact_size(egui::vec2(14.0, 14.0))
                        .tint(ui.visuals().text_color()),
                ))
                .on_hover_text("Remove satellite view")
                .clicked()
        {
            self.model_action = Some(Action::Remove);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn previous_and_next_satellites_follow_the_dropdown_and_wrap() {
        let satellites: Vec<_> = modelled_satellites().collect();
        let first = satellites[0];
        let mut view = View {
            asset: first.model_index().unwrap(),
            catalogue_selection: Some(first.id),
            ..Default::default()
        };
        for satellite in satellites.iter().skip(1).chain(std::iter::once(&first)) {
            view.cycle_satellite(true);
            assert_eq!(view.catalogue_selection, Some(satellite.id));
            assert_eq!(view.asset, satellite.model_index().unwrap());
        }
        for satellite in satellites
            .iter()
            .skip(1)
            .rev()
            .chain(std::iter::once(&first))
        {
            view.cycle_satellite(false);
            assert_eq!(view.catalogue_selection, Some(satellite.id));
            assert_eq!(view.asset, satellite.model_index().unwrap());
        }
    }

    #[test]
    fn cycling_uses_the_current_model_when_no_catalogue_entry_is_selected() {
        let satellites: Vec<_> = modelled_satellites().collect();
        let mut view = View {
            asset: satellites[1].model_index().unwrap(),
            catalogue_selection: None,
            ..Default::default()
        };
        let current = satellites
            .iter()
            .position(|satellite| satellite.model_index() == Some(view.asset))
            .unwrap();
        view.camera.zoom = 2.0;
        view.camera.yaw = 1.2;
        view.camera.pitch = -0.4;
        view.cycle_satellite(true);
        assert_eq!(
            view.catalogue_selection,
            Some(satellites[(current + 1) % satellites.len()].id)
        );
        assert_eq!(view.camera.zoom, 2.0);
        assert_eq!(view.camera.yaw, 1.2);
        assert_eq!(view.camera.pitch, -0.4);
        view.catalogue_selection = Some(u32::MAX);
        let current = satellites
            .iter()
            .position(|satellite| satellite.model_index() == Some(view.asset))
            .unwrap();
        view.cycle_satellite(false);
        assert_eq!(
            view.catalogue_selection,
            Some(satellites[(current + satellites.len() - 1) % satellites.len()].id)
        );
    }

    #[test]
    fn linked_cameras_share_scale_and_orientation_for_different_model_sizes() {
        let linked = ComparisonCamera {
            yaw: 1.2,
            pitch: -0.4,
            target_metres: Vector3::new(1.0, -2.0, 0.5),
            metres_per_point: 0.03,
            reference_metres_per_point: 0.06,
            fit_pending: false,
            layout_size: egui::vec2(1440.0, 1000.0),
            pane_count: 2,
        };
        let mut rotated = None;
        for (size, units, explosion) in [
            (egui::vec2(650.0, 600.0), 2.5, 0.0),
            (egui::vec2(650.0, 600.0), 15.0, 1.0),
            (egui::vec2(320.0, 180.0), 0.15, 0.0),
        ] {
            let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
            let mut camera = linked.camera_for(rect, explosion, units);
            assert!(
                (camera.metres_per_point(rect, explosion, units) - linked.metres_per_point).abs()
                    < 1e-6
            );
            assert!((camera.target * units - linked.target_metres).norm() < 1e-6);
            assert_eq!(camera.yaw, linked.yaw);
            assert_eq!(camera.pitch, linked.pitch);
            camera.orbit(egui::vec2(20.0, -8.0), linked.zoom());
            let angles = (camera.yaw, camera.pitch);
            if let Some(rotated) = rotated {
                assert_eq!(
                    angles, rotated,
                    "cursor sensitivity must not depend on model dimensions"
                );
            }
            rotated = Some(angles);
        }
    }

    #[test]
    fn satellite_view_starts_with_one_pane_and_can_add_independent_models() {
        assert!(modelled_satellites().all(|satellite| satellite.model_index().is_some()));
        assert!(!modelled_satellites()
            .any(|satellite| satellite.slug == "worldview_3" || satellite.name == "MiRaTa"));
        let mut view = View::default();
        view.ensure_model_panes();
        assert_eq!(view.model_panes.len(), 1);
        view.ensure_model_panes();
        assert_eq!(view.model_panes.len(), 1);
        view.add_next_model_pane();
        assert_eq!(view.model_panes.len(), 2);
        assert_ne!(
            view.model_panes[0].view.asset,
            view.model_panes[1].view.asset
        );
        let first = modelled_satellites().next().unwrap();
        view.add_model_pane(first);
        view.add_model_pane(first);
        assert_ne!(view.model_panes[2].id, view.model_panes[3].id);
        view.model_panes[2].view.camera.zoom = 2.0;
        assert_eq!(view.model_panes[3].view.camera.zoom, 1.0);
        while view.model_panes.len() > 1 {
            view.remove_model_pane(view.model_panes[0].id);
        }
        view.remove_model_pane(view.model_panes[0].id);
        assert_eq!(view.model_panes.len(), 1);
    }

    #[test]
    fn model_grid_fits_desktop_and_narrow_views_without_overlapping() {
        for width in [390.0, 768.0, 1440.0] {
            for count in 1..=8 {
                let area = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width, 900.0));
                let rects = pane_rects(area, count);
                for (index, rect) in rects.iter().enumerate() {
                    assert!(area.contains_rect(*rect));
                    assert!(rect.width() > 0.0 && rect.height() > 0.0);
                    assert!(!rects[index + 1..]
                        .iter()
                        .any(|other| rect.intersects(*other)));
                }
            }
        }
    }
}
