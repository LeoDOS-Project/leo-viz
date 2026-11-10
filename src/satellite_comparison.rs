//! Operational spacecraft specifications, with model thumbnails where available.
use super::*;
use crate::satellite_catalogue::{catalogue, Dimensions, Satellite};
use egui_extras::{Column, TableBuilder};

pub(super) struct ColumnGroup {
    pub label: &'static str,
    weight: f32,
    fields: &'static [usize],
}

pub(super) const COLUMNS: [ColumnGroup; 9] = [
    ColumnGroup {
        label: "Satellite",
        weight: 180.0,
        fields: &[0],
    },
    ColumnGroup {
        label: "Launched",
        weight: 150.0,
        fields: &[1],
    },
    ColumnGroup {
        label: "Mass",
        weight: 200.0,
        fields: &[2],
    },
    ColumnGroup {
        label: "Design orbit",
        weight: 200.0,
        fields: &[3],
    },
    ColumnGroup {
        label: "Power",
        weight: 220.0,
        fields: &[4],
    },
    ColumnGroup {
        label: "Payload",
        weight: 260.0,
        fields: &[5],
    },
    ColumnGroup {
        label: "Dimensions",
        weight: 180.0,
        fields: &[6],
    },
    ColumnGroup {
        label: "Other specifications",
        weight: 260.0,
        fields: &[7],
    },
    ColumnGroup {
        label: "References",
        weight: 180.0,
        fields: &[8],
    },
];

pub(super) fn column_groups(width: f32) -> &'static [ColumnGroup] {
    if width >= 900.0 {
        &COLUMNS
    } else if width >= 600.0 {
        &[
            ColumnGroup {
                label: "Satellite",
                weight: 180.0,
                fields: &[0],
            },
            ColumnGroup {
                label: "Mission",
                weight: 300.0,
                fields: &[1, 2, 3],
            },
            ColumnGroup {
                label: "Payload and power",
                weight: 340.0,
                fields: &[4, 5],
            },
            ColumnGroup {
                label: "Specifications",
                weight: 380.0,
                fields: &[6, 7, 8],
            },
        ]
    } else {
        &[
            ColumnGroup {
                label: "Satellite",
                weight: 3.0,
                fields: &[0],
            },
            ColumnGroup {
                label: "Specifications",
                weight: 7.0,
                fields: &[1, 2, 3, 4, 5, 6, 7, 8],
            },
        ]
    }
}

pub(super) fn column_widths(width: f32, spacing: f32, groups: &[ColumnGroup]) -> Vec<f32> {
    let content_width = (width - spacing * (groups.len() - 1) as f32).max(0.0);
    let weight: f32 = groups.iter().map(|group| group.weight).sum();
    groups
        .iter()
        .map(|group| content_width * group.weight / weight)
        .collect()
}

enum CellContent {
    Text(String),
    Heading(&'static str),
    Link(String, String),
}

impl CellContent {
    fn text(&self) -> &str {
        match self {
            Self::Text(text) => text,
            Self::Heading(text) => text,
            Self::Link(text, _) => text,
        }
    }
}

fn dimension_text(dimensions: &Dimensions) -> String {
    let dimensions = dimensions
        .metres
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|v| format!("{v:.2}"))
        .collect::<Vec<_>>()
        .join(" x ");
    format!("{dimensions} m")
}

fn cell_contents(satellite: &Satellite, group: &ColumnGroup) -> Vec<CellContent> {
    let labelled = group.fields.len() > 1;
    let mut contents = Vec::new();
    for &field in group.fields {
        match field {
            0 => {}
            6 => {
                if labelled {
                    contents.push(CellContent::Heading("Dimensions"));
                }
                if let Some(dimensions) = satellite.dimensions() {
                    contents.push(CellContent::Text(dimensions.label.to_owned()));
                    contents.push(CellContent::Text(dimension_text(&dimensions)));
                    contents.push(CellContent::Link(
                        "Dimensions source".to_owned(),
                        dimensions.source.to_owned(),
                    ));
                } else {
                    contents.push(CellContent::Text("Not specified".to_owned()));
                }
            }
            8 => {
                if labelled {
                    contents.push(CellContent::Heading("References"));
                }
                contents.push(CellContent::Link(
                    "WMO OSCAR".to_owned(),
                    satellite.source(),
                ));
                if let Some(facts) = satellite.facts() {
                    contents.extend(
                        facts
                            .sources
                            .iter()
                            .map(|&(name, url)| CellContent::Link(name.to_owned(), url.to_owned())),
                    );
                }
                if let Some(index) = satellite.model_index() {
                    contents.push(CellContent::Link(
                        "Model source".to_owned(),
                        ASSETS[index].source.to_owned(),
                    ));
                }
            }
            _ if labelled => {
                let mut found = false;
                for (key, value) in field_values(satellite, field) {
                    found = true;
                    contents.push(CellContent::Heading(key));
                    contents.push(CellContent::Text(value));
                }
                if !found && field != 7 {
                    contents.push(CellContent::Heading(COLUMNS[field].label));
                    contents.push(CellContent::Text("Not specified".to_owned()));
                }
            }
            _ => {
                let text = facts_for_column(satellite, field);
                contents.push(CellContent::Text(if text.is_empty() {
                    "Not specified".to_owned()
                } else {
                    text
                }));
            }
        }
    }
    contents
}

fn text_height(ui: &egui::Ui, text: &str, font: &egui::FontId, width: f32) -> f32 {
    ui.painter()
        .layout(
            text.to_owned(),
            font.clone(),
            ui.visuals().text_color(),
            width.max(1.0),
        )
        .size()
        .y
}

fn thumbnail_size(width: f32) -> egui::Vec2 {
    let width = width.min(172.0).max(1.0);
    egui::vec2(width, width * 100.0 / 172.0)
}

fn fact_column(key: &str) -> usize {
    match key {
        "Launched" => 1,
        "Mass" | "Launch mass" | "Bus mass" | "Dry mass" => 2,
        "Design orbit" => 3,
        "Avg. demand" | "Avg. power" | "Power spec" | "Solar power" => 4,
        "Payload" => 5,
        _ => 7,
    }
}

pub(super) fn field_values(satellite: &Satellite, column: usize) -> Vec<(&'static str, String)> {
    if column == 1 {
        return vec![("Launched", satellite.launched.clone())];
    }
    if column == 5 {
        return if satellite.payload.is_empty() {
            Vec::new()
        } else {
            vec![("Payload", satellite.payload.join(", "))]
        };
    }
    let mut fields: Vec<_> = satellite
        .facts()
        .into_iter()
        .flat_map(|facts| facts.rows.iter())
        .filter(|(key, _)| fact_column(key) == column)
        .filter(|(key, _)| satellite.name != "GOES-19" || *key != "Launch mass")
        .map(|&(key, value)| (key, value.to_owned()))
        .collect();
    if column == 2 && fields.is_empty() {
        if let Some(value) = &satellite.launch_mass {
            fields.push(("Launch mass", value.clone()));
        }
        if let Some(value) = &satellite.dry_mass {
            fields.push(("Dry mass", value.clone()));
        }
    }
    if column == 4 && fields.is_empty() {
        if let Some(value) = &satellite.power {
            fields.push(("Power spec", value.clone()));
        }
    }
    if column == 3 && fields.is_empty() {
        let orbit =
            if let Some(altitude) = satellite.altitude_km.as_deref().filter(|s| !s.is_empty()) {
                format!("{altitude} km; {}", satellite.orbit)
            } else {
                satellite.orbit.clone()
            };
        fields.push(("Design orbit", orbit));
    }
    if column == 7 {
        fields.insert(0, ("Agency", satellite.agency.replace(',', ", ")));
        fields.insert(1, ("Status", satellite.status.clone()));
        if let Some(updated) = &satellite.updated {
            fields.push(("Source updated", updated.clone()));
        }
    }
    fields
}

fn facts_for_column(satellite: &Satellite, column: usize) -> String {
    field_values(satellite, column)
        .into_iter()
        .map(|(key, value)| {
            if matches!(column, 2 | 4 | 7) {
                format!("{key}: {value}")
            } else {
                value
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

impl View {
    pub fn show_comparison(&mut self, ui: &mut egui::Ui, render: Option<&RenderState>) -> bool {
        self.poll_loads();
        if let Some(model) = &self.model {
            self.models
                .entry(self.loaded_asset)
                .or_insert_with(|| Arc::clone(model));
        }
        ui.horizontal_wrapped(|ui| {
            crate::settings::reserve_sidebar_toggle(ui);
            ui.heading("Satellite Table");
            ui.checkbox(&mut self.spin, "Auto rotate");
        });
        let catalogue = catalogue();
        let mut search_changed = false;
        let mut sort_changed = false;
        ui.horizontal_wrapped(|ui| {
            search_changed = ui
                .add(
                    egui::TextEdit::singleline(&mut self.comparison_search)
                        .hint_text("Search satellites")
                        .desired_width(220.0_f32.min(ui.available_width())),
                )
                .changed();
            ui.hyperlink_to("WMO OSCAR", &catalogue.source);
            ui.label(format!("Status checked {}", catalogue.checked));
            sort_changed = self.comparison_sort.controls(ui);
        });
        let query = self.comparison_search.trim().to_lowercase();
        let mut satellites: Vec<_> = catalogue
            .satellites
            .iter()
            .filter(|s| s.matches(&query))
            .collect();
        self.comparison_sort.apply(&mut satellites);
        ui.label(format!("{} entries", satellites.len()));
        if satellites.is_empty() {
            self.comparison_renderers.clear();
            ui.label("No matching satellites");
            return false;
        }
        if self.spin {
            self.camera.yaw += ui.input(|i| i.stable_dt.min(0.05)) * 0.22;
            ui.ctx().request_repaint();
        }
        let font = bold_font(ui);
        let body_font = egui::TextStyle::Body.resolve(ui.style());
        let ctx = ui.ctx().clone();
        let mut opened = None;
        let mut visible = Vec::new();
        let groups = column_groups(ui.available_width());
        let width = (ui.available_width() - ui.spacing().scroll.allocated_width() - 1.0).max(1.0);
        let widths = column_widths(width, ui.spacing().item_spacing.x, groups);
        let header_height = groups
            .iter()
            .zip(&widths)
            .map(|(group, &width)| text_height(ui, group.label, &font, (width - 16.0).max(1.0)))
            .fold(20.0_f32, f32::max)
            .ceil()
            + 4.0;
        let cells: Vec<Vec<_>> = satellites
            .iter()
            .map(|asset| {
                groups
                    .iter()
                    .map(|group| cell_contents(asset, group))
                    .collect()
            })
            .collect();
        let heights: Vec<_> = satellites
            .iter()
            .zip(&cells)
            .map(|(asset, cells)| {
                let model_height = text_height(ui, &asset.name, &font, widths[0])
                    + ui.spacing().item_spacing.y
                    + if asset.model_index().is_some() {
                        thumbnail_size(widths[0]).y
                    } else {
                        text_height(ui, "No model", &body_font, widths[0])
                    };
                cells
                    .iter()
                    .zip(&widths)
                    .map(|(contents, &width)| {
                        contents
                            .iter()
                            .map(|content| {
                                text_height(
                                    ui,
                                    content.text(),
                                    if matches!(content, CellContent::Heading(_)) {
                                        &font
                                    } else {
                                        &body_font
                                    },
                                    width,
                                )
                            })
                            .sum::<f32>()
                            + ui.spacing().item_spacing.y * contents.len().saturating_sub(1) as f32
                    })
                    .fold(model_height, f32::max)
                    .ceil()
                    + 8.0
            })
            .collect();
        let body_height =
            (ui.available_height() - header_height - ui.spacing().item_spacing.y).max(0.0);
        ui.scope(|ui| {
            let mut table = TableBuilder::new(ui)
                .id_salt("satellite_comparison")
                .striped(true)
                .cell_layout(egui::Layout::top_down(egui::Align::Min))
                .min_scrolled_height(0.0);
            table = table.max_scroll_height(body_height);
            if search_changed || sort_changed || std::mem::take(&mut self.comparison_scroll_to_top)
            {
                table = table.vertical_scroll_offset(0.0);
            }
            for &width in &widths {
                table = table.column(Column::exact(width));
            }
            table
                .header(header_height, |mut header| {
                    for (column, group) in groups.iter().enumerate() {
                        header.col(|ui| {
                            let active = group
                                .fields
                                .contains(&(self.comparison_sort.field as usize));
                            let field = if active {
                                self.comparison_sort.field
                            } else {
                                sort::Field::ALL[group.fields[0]]
                            };
                            let rect = ui.max_rect();
                            let label = egui::Rect::from_min_max(
                                rect.min,
                                egui::pos2((rect.right() - 16.0).max(rect.left()), rect.bottom()),
                            );
                            ui.scope_builder(
                                egui::UiBuilder::new()
                                    .max_rect(label)
                                    .layout(egui::Layout::top_down(egui::Align::Min)),
                                |ui| {
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(group.label).font(font.clone()),
                                        )
                                        .wrap(),
                                    );
                                },
                            );
                            if active {
                                let icon = egui::Rect::from_min_size(
                                    egui::pos2(rect.right() - 12.0, rect.top() + 2.0),
                                    egui::vec2(12.0, 12.0),
                                );
                                ui.put(
                                    icon,
                                    egui::Image::new(self.comparison_sort.icon())
                                        .tint(ui.visuals().text_color()),
                                );
                            }
                            if ui
                                .interact(
                                    rect,
                                    ui.id().with(("sort_header", column)),
                                    egui::Sense::click(),
                                )
                                .on_hover_cursor(egui::CursorIcon::PointingHand)
                                .on_hover_text(field.description())
                                .clicked()
                            {
                                self.comparison_sort.toggle(field);
                                self.comparison_scroll_to_top = true;
                                ui.ctx().request_repaint();
                            }
                        });
                    }
                })
                .body(|body| {
                    body.heterogeneous_rows(heights.into_iter(), |mut row| {
                        let index = row.index();
                        let satellite = satellites[index];
                        row.col(|ui| {
                            // Clipped table cells otherwise truncate links instead of wrapping.
                            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
                            let name = egui::RichText::new(&satellite.name).font(font.clone());
                            let Some(index) = satellite.model_index() else {
                                ui.add(egui::Label::new(name).wrap())
                                    .on_hover_text(&satellite.full_name);
                                ui.weak("No model");
                                return;
                            };
                            visible.push(index);
                            self.request_model(index, &ctx);
                            if ui.link(name).on_hover_text(&satellite.full_name).clicked() {
                                opened = Some((index, satellite.id));
                            }
                            let (rect, response) = ui.allocate_exact_size(
                                thumbnail_size(widths[0]),
                                egui::Sense::click(),
                            );
                            if !ui.is_rect_visible(rect) {
                                return;
                            }
                            if let (Some(render), Some(model)) = (render, self.models.get(&index)) {
                                let renderer = self
                                    .comparison_renderers
                                    .entry(index)
                                    .or_insert_with(|| Renderer::new(render, Arc::clone(model)));
                                let camera = Camera {
                                    target: Vector3::zeros(),
                                    zoom: 1.0,
                                    ..self.camera
                                };
                                let (matrix, eye) = camera.matrices(rect.aspect_ratio(), 0.0);
                                renderer.draw(
                                    target_size(
                                        rect.size(),
                                        ui.ctx().pixels_per_point(),
                                        render.device.limits().max_texture_dimension_2d,
                                    ),
                                    Uniform {
                                        view_projection: matrix.into(),
                                        eye: [eye.x, eye.y, eye.z, 1.0],
                                        options: [0.0, -1.0, 0.0, -1.0],
                                    },
                                    ui.visuals().dark_mode,
                                );
                                ui.painter().image(
                                    renderer.target.as_ref().unwrap().id,
                                    rect,
                                    egui::Rect::from_min_max(
                                        egui::Pos2::ZERO,
                                        egui::pos2(1.0, 1.0),
                                    ),
                                    egui::Color32::WHITE,
                                );
                                if response
                                    .on_hover_cursor(egui::CursorIcon::PointingHand)
                                    .clicked()
                                {
                                    opened = Some((index, satellite.id));
                                }
                            } else if let Some(error) = self.errors.get(&index).cloned() {
                                if ui
                                    .put(rect, egui::Button::new("Retry"))
                                    .on_hover_text(error)
                                    .clicked()
                                {
                                    self.errors.remove(&index);
                                }
                            } else {
                                ui.put(rect, egui::Spinner::new());
                            }
                        });
                        for contents in cells[index].iter().skip(1) {
                            row.col(|ui| {
                                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
                                for content in contents {
                                    let text = egui::RichText::new(content.text()).font(
                                        if matches!(content, CellContent::Heading(_)) {
                                            font.clone()
                                        } else {
                                            body_font.clone()
                                        },
                                    );
                                    if let CellContent::Link(_, url) = content {
                                        ui.add(egui::Hyperlink::from_label_and_url(text, url));
                                    } else {
                                        ui.add(egui::Label::new(text).wrap());
                                    }
                                }
                            });
                        }
                    });
                });
        });
        self.comparison_renderers
            .retain(|index, _| visible.contains(index));
        if let Some((index, id)) = opened {
            self.asset = index;
            self.catalogue_selection = Some(id);
            self.model_panes.clear();
            self.return_to_table = true;
            self.selected = None;
            self.inspected = None;
            self.hovered = None;
            self.picking = None;
            return true;
        }
        false
    }
}

pub(super) fn show_compact_stats(ui: &mut egui::Ui, satellite: &Satellite) {
    let fields: Vec<_> = (1..=7)
        .filter(|&column| column != 6)
        .flat_map(|column| field_values(satellite, column))
        .collect();
    let rows: Vec<_> = fields
        .iter()
        .map(|(key, value)| (*key, value.as_str()))
        .collect();
    show_stat_rows(ui, &rows);
    ui.separator();
    ui.horizontal_wrapped(|ui| {
        ui.hyperlink_to("WMO OSCAR", satellite.source());
        if let Some(facts) = satellite.facts() {
            for &(name, url) in facts.sources {
                ui.hyperlink_to(name, url);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_retains_every_published_statistic_and_its_mass_and_power_qualifiers() {
        for satellite in &catalogue().satellites {
            for (key, value) in (1..=7)
                .filter(|&column| column != 6)
                .flat_map(|column| field_values(satellite, column))
            {
                let text = facts_for_column(satellite, fact_column(key));
                assert!(text.contains(&value), "{}: {key}", satellite.name);
                if matches!(fact_column(key), 2 | 4 | 7) {
                    assert!(text.contains(key));
                }
            }
        }
    }

    #[test]
    fn responsive_columns_fit_and_include_every_field_once() {
        for width in [
            240.0, 280.0, 390.0, 599.0, 600.0, 768.0, 899.0, 900.0, 1024.0, 1440.0, 2400.0,
        ] {
            let groups = column_groups(width);
            let widths = column_widths(width, 8.0, groups);
            assert!(widths.iter().all(|&width| width > 0.0));
            let total = widths.iter().sum::<f32>() + 8.0 * (groups.len() - 1) as f32;
            assert!(
                (total - width).abs() < 0.001,
                "columns must fit {width} points"
            );
            assert!(thumbnail_size(widths[0]).x <= widths[0]);
            let fields: Vec<_> = groups
                .iter()
                .flat_map(|group| group.fields.iter().copied())
                .collect();
            assert_eq!(fields, (0..COLUMNS.len()).collect::<Vec<_>>());
        }
    }

    #[test]
    fn grouped_cells_preserve_all_statistics_dimensions_and_source_links() {
        for width in [390.0, 768.0, 1024.0] {
            for satellite in &catalogue().satellites {
                let contents: Vec<_> = column_groups(width)
                    .iter()
                    .flat_map(|group| cell_contents(satellite, group))
                    .collect();
                for (key, value) in (1..=7)
                    .filter(|&column| column != 6)
                    .flat_map(|column| field_values(satellite, column))
                {
                    assert!(
                        contents
                            .iter()
                            .any(|content| content.text().contains(&value)),
                        "{}: {key}",
                        satellite.name
                    );
                }
                if let Some(dimensions) = satellite.dimensions() {
                    assert!(contents
                        .iter()
                        .any(|content| content.text() == dimension_text(&dimensions)));
                }
                let sources = satellite
                    .facts()
                    .into_iter()
                    .flat_map(|facts| facts.sources.iter().map(|&(_, url)| url.to_owned()))
                    .chain(std::iter::once(satellite.source()));
                for url in sources {
                    assert!(contents.iter().any(
                        |content| matches!(content, CellContent::Link(_, link) if *link == url)
                    ));
                }
            }
        }
    }
}
