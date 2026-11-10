//! Ground receiver comparison with separate published specifications and estimates.

use eframe::egui;
use egui_extras::{Column, TableBuilder};
use egui_plot::{Legend, Line, Plot};
use std::cmp::Ordering;

#[path = "ground_terminal_data.rs"]
mod data;
#[path = "ground_terminal_link.rs"]
mod link;

use data::{Band, Terminal, TERMINALS};
use link::{Comparison, Estimate, Receiver, Scenario};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Field {
    Terminal,
    Diameter,
    Gt,
    Published,
    Peak,
    Average,
    Data,
    Limit,
}

impl Field {
    const ALL: [Self; 8] = [
        Self::Terminal,
        Self::Diameter,
        Self::Gt,
        Self::Published,
        Self::Peak,
        Self::Average,
        Self::Data,
        Self::Limit,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Terminal => "Terminal",
            Self::Diameter => "Dish / bands",
            Self::Gt => "G/T (dB/K)",
            Self::Published => "Published max (Mbit/s)",
            Self::Peak => "Est. peak (Mbit/s)",
            Self::Average => "Est. average (Mbit/s)",
            Self::Data => "Est. data (GB/pass)",
            Self::Limit => "Limit at peak",
        }
    }

    fn value(self, terminal: &Terminal, s: &Scenario, estimate: Option<&Estimate>) -> Option<f64> {
        match self {
            Self::Diameter => Some(terminal.diameter_m),
            Self::Gt => terminal.receive(s.band).map(|rx| rx.gt_dbk),
            Self::Published => terminal.receive(s.band).and_then(|rx| rx.system_max_mbps),
            Self::Peak => estimate.map(|e| e.peak_mbps),
            Self::Average => estimate.map(|e| e.average_mbps),
            Self::Data => estimate.map(|e| e.data_gb),
            Self::Terminal | Self::Limit => None,
        }
    }
}

#[derive(Clone, Copy)]
struct Sort {
    field: Field,
    descending: bool,
}

impl Default for Sort {
    fn default() -> Self {
        Self {
            field: Field::Data,
            descending: true,
        }
    }
}

fn compare_optional(a: Option<f64>, b: Option<f64>, descending: bool) -> Ordering {
    match (a, b) {
        (Some(a), Some(b)) => {
            if descending {
                b.total_cmp(&a)
            } else {
                a.total_cmp(&b)
            }
        }
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

impl Sort {
    fn toggle(&mut self, field: Field) {
        self.descending = if self.field == field {
            !self.descending
        } else {
            field != Field::Terminal
        };
        self.field = field;
    }

    fn indices(&self, comparison: &Comparison) -> Vec<usize> {
        let mut indices: Vec<_> = (0..TERMINALS.len()).collect();
        indices.sort_by(|&a, &b| {
            let ea = comparison.rows[a].as_ref().ok();
            let eb = comparison.rows[b].as_ref().ok();
            let order = if self.field == Field::Terminal {
                let order = TERMINALS[a].name.cmp(TERMINALS[b].name);
                if self.descending {
                    order.reverse()
                } else {
                    order
                }
            } else if self.field == Field::Limit {
                match (ea, eb) {
                    (Some(a), Some(b)) => {
                        let order = a.peak_limit.label().cmp(b.peak_limit.label());
                        if self.descending {
                            order.reverse()
                        } else {
                            order
                        }
                    }
                    (Some(_), None) => Ordering::Less,
                    (None, Some(_)) => Ordering::Greater,
                    _ => Ordering::Equal,
                }
            } else {
                compare_optional(
                    self.field.value(&TERMINALS[a], &comparison.scenario, ea),
                    self.field.value(&TERMINALS[b], &comparison.scenario, eb),
                    self.descending,
                )
            };
            order.then_with(|| TERMINALS[a].name.cmp(TERMINALS[b].name))
        });
        indices
    }
}

pub struct View {
    scenario: Scenario,
    comparison: Option<Comparison>,
    selected: Vec<bool>,
    sort: Sort,
}

impl Default for View {
    fn default() -> Self {
        let mut selected = vec![false; TERMINALS.len()];
        for index in [0, 3, 10] {
            selected[index] = true;
        }
        Self {
            scenario: Scenario::default(),
            comparison: None,
            selected,
            sort: Sort::default(),
        }
    }
}

const HELP: &str = "Nominal overhead circular LEO pass above a stationary spherical Earth; Earth rotation, actual station location, scheduling and acquisition time are not modelled. G/T is held at the published reference value (or its lower bound); reference conditions differ between sources. Rain, atmosphere, pointing and implementation losses must be included in the additional-loss assumption.\n\nAdaptive single-channel estimate: minimum of the Eb/N0 rate budget, Shannon bound, spectral efficiency x RF bandwidth, satellite limit, receiver limit and any published station-system limit. Payload fraction is applied once, before the useful-data backhaul cap. An unspecified station ceiling is unknown, not unlimited verified throughput. This is not a guaranteed modem operating point or a site/weather prediction.\n\nGB is decimal (1 GB = 8,000 Mbit). Receiver selection assumes compatible IF conversion, modulation and coding; it does not assert an engineered product pairing. No changes are made to simulation ground stations.";

impl View {
    pub fn show(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .id_salt("ground_terminals_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    crate::settings::reserve_sidebar_toggle(ui);
                    ui.heading("Ground Terminals");
                    ui.label(format!("{} configurations", TERMINALS.len()));
                    ui.button("?").on_hover_text(HELP);
                });
                self.controls(ui);
                if self
                    .comparison
                    .as_ref()
                    .is_none_or(|c| c.scenario != self.scenario)
                {
                    match Comparison::calculate(self.scenario.clone(), TERMINALS) {
                        Ok(comparison) => self.comparison = Some(comparison),
                        Err(error) => {
                            ui.colored_label(ui.visuals().error_fg_color, error);
                            return;
                        }
                    }
                }
                let comparison = self.comparison.as_ref().unwrap();
                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    ui.strong("Estimated useful data rate");
                    ui.label(format!(
                        "Overhead pass: {:.1} min",
                        comparison.duration_s / 60.0
                    ));
                });
                let dark = ui.visuals().dark_mode;
                let plot_id = ui.make_persistent_id("ground_terminal_rate");
                let plot_height = (TERMINALS.len() as f32
                    * (ui.text_style_height(&egui::TextStyle::Body) + ui.spacing().item_spacing.y)
                    + 56.0)
                    .max(210.0);
                Plot::new("ground_terminal_rate")
                    .id(plot_id)
                    .height(plot_height)
                    .legend(Legend::default())
                    .x_axis_label("Time from overhead (s)")
                    .y_axis_label("Mbit/s")
                    .include_y(0.0)
                    .allow_drag(false)
                    .allow_zoom(false)
                    .allow_scroll(false)
                    .label_formatter(|name, point| {
                        format!("{name}\n{:.1} Mbit/s at {:.0} s", point.y, point.x)
                    })
                    .show(ui, |plot| {
                        for (index, row) in comparison.rows.iter().enumerate() {
                            if self.selected[index] {
                                if let Ok(estimate) = row {
                                    plot.line(
                                        Line::new(TERMINALS[index].name, estimate.points.clone())
                                            .color(series_color(index, dark))
                                            .width(2.0_f32),
                                    );
                                }
                            }
                        }
                    });
                // Keep the plot legend and the table checkboxes in agreement.
                if let Some(mut memory) = egui_plot::PlotMemory::load(ui.ctx(), plot_id) {
                    if !memory.hidden_items.is_empty() {
                        for (index, terminal) in TERMINALS.iter().enumerate() {
                            if memory.hidden_items.contains(&egui::Id::new(terminal.name)) {
                                self.selected[index] = false;
                            }
                        }
                        memory.hidden_items.clear();
                        memory.store(ui.ctx(), plot_id);
                        ui.ctx().request_repaint();
                    }
                }
                if !comparison
                    .rows
                    .iter()
                    .enumerate()
                    .any(|(i, row)| self.selected[i] && row.is_ok())
                {
                    ui.weak("No compatible terminals selected");
                }
                ui.separator();
                let wide = ui.available_width() >= 650.0;
                if !wide {
                    ui.strong("Published specifications and calculated estimates");
                }
                ui.horizontal_wrapped(|ui| {
                    if wide {
                        ui.strong("Published specifications and calculated estimates");
                    }
                    egui::ComboBox::from_id_salt("ground_terminal_sort")
                        .selected_text(format!("Sort: {}", self.sort.field.label()))
                        .show_ui(ui, |ui| {
                            for field in Field::ALL {
                                if ui
                                    .selectable_label(self.sort.field == field, field.label())
                                    .clicked()
                                {
                                    self.sort.toggle(field);
                                }
                            }
                            ui.separator();
                            ui.checkbox(&mut self.sort.descending, "Descending");
                        });
                });
                terminal_table(ui, comparison, &mut self.selected, &mut self.sort);
            });
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        let pairs = if ui.available_width() >= 900.0 {
            3
        } else if ui.available_width() >= 600.0 {
            2
        } else {
            1
        };
        egui::Grid::new("ground_scenario")
            .num_columns(pairs * 2)
            .show(ui, |ui| {
                ui.label("Band");
                ui.horizontal(|ui| {
                    for band in Band::ALL {
                        if ui
                            .selectable_value(&mut self.scenario.band, band, band.label())
                            .changed()
                        {
                            self.scenario.frequency_ghz = band.frequency_ghz();
                            self.scenario.bandwidth_mhz = band.bandwidth_mhz();
                        }
                    }
                });
                if pairs == 1 {
                    ui.end_row();
                }
                let s = &mut self.scenario;
                for (index, (label, value, range, suffix)) in [
                    ("Altitude", &mut s.altitude_km, 400.0..=2000.0, " km"),
                    (
                        "Min. elevation",
                        &mut s.min_elevation_deg,
                        10.0..=85.0,
                        " deg",
                    ),
                    ("Satellite EIRP", &mut s.eirp_dbw, -50.0..=100.0, " dBW"),
                    ("RF bandwidth", &mut s.bandwidth_mhz, 0.1..=1500.0, " MHz"),
                    (
                        "Satellite limit",
                        &mut s.satellite_mbps,
                        0.1..=100_000.0,
                        " Mbit/s",
                    ),
                ]
                .into_iter()
                .enumerate()
                {
                    ui.label(label);
                    ui.add_sized(
                        [100.0, ui.spacing().interact_size.y],
                        egui::DragValue::new(value)
                            .range(range)
                            .speed(0.1)
                            .suffix(suffix),
                    );
                    if (index + 2) % pairs == 0 {
                        ui.end_row();
                    }
                }
            });
        ui.horizontal_wrapped(|ui| {
            ui.label("Receiver");
            egui::ComboBox::from_id_salt("ground_receiver")
                .selected_text(self.scenario.receiver.label())
                .show_ui(ui, |ui| {
                    for receiver in Receiver::ALL {
                        ui.selectable_value(&mut self.scenario.receiver, receiver, receiver.label());
                    }
                });
            if self.scenario.receiver == Receiver::Custom {
                ui.label("Channel limit");
                ui.add(egui::DragValue::new(&mut self.scenario.custom_receiver_mbps).range(0.1..=100_000.0).speed(0.1).suffix(" Mbit/s"));
            } else {
                ui.label(format!("{:.0} Mbit/s, one channel", self.scenario.receiver.channel_mbps(0.0)));
                ui.hyperlink_to("Source", data::RECEIVER_SOURCE).on_hover_text(
                    "VHR 3200: 4,050 Mbit/s per channel; 8,100 Mbit/s is the two-channel aggregate, not this link. Optional processor: 4,000 Mbit/s capture/archive ceiling. These are maxima, not guaranteed useful payload rates. Datasheet: 2022."
                );
            }
        });
        egui::CollapsingHeader::new("Link assumptions").show(ui, |ui| {
            let pairs_per_row = if ui.available_width() >= 800.0 { 2 } else { 1 };
            egui::Grid::new("ground_link_assumptions").num_columns(pairs_per_row * 2).show(ui, |ui| {
                let s = &mut self.scenario;
                let values = [
                    ("Carrier", &mut s.frequency_ghz, 1.0..=40.0, " GHz", "Receive centre frequency; the full occupied bandwidth must fit the published RF range."),
                    ("Spectral efficiency", &mut s.spectral_efficiency, 0.1..=10.0, " bit/s/Hz", "Assumed gross information rate per Hz, including coding choice and occupied spectrum."),
                    ("Required Eb/N0", &mut s.required_ebn0_db, -2.0..=30.0, " dB", "Assumed energy-per-information-bit threshold for the chosen waveform; not a published modem threshold."),
                    ("Link margin", &mut s.margin_db, 0.0..=60.0, " dB", "Reserved margin, subtracted from the available C/N0."),
                    ("Additional loss", &mut s.additional_loss_db, 0.0..=100.0, " dB", "Combined atmosphere, rain, pointing, polarization and implementation loss beyond free-space loss. Not a weather calculation."),
                    ("Payload fraction", &mut s.payload_fraction, 0.01..=1.0, "", "Useful payload fraction after framing overhead. 0.90 means 90%; applied once."),
                    ("Useful backhaul", &mut s.backhaul_mbps, 0.1..=100_000.0, " Mbit/s", "Cap on useful payload delivered by the ground connection."),
                ];
                for (index, (label, value, range, suffix, help)) in values.into_iter().enumerate() {
                    ui.label(label).on_hover_text(help);
                    ui.add(egui::DragValue::new(value).range(range).speed(0.1).suffix(suffix)).on_hover_text(help);
                    if (index + 1) % pairs_per_row == 0 { ui.end_row(); }
                }
            });
            ui.weak("Reference G/T held constant; adaptive one-channel estimate, not a guaranteed data rate.");
        });
    }
}

fn series_color(index: usize, dark: bool) -> egui::Color32 {
    let colors = if dark {
        [
            [80, 190, 245],
            [120, 210, 130],
            [245, 180, 70],
            [235, 110, 110],
            [185, 150, 240],
            [70, 215, 195],
            [245, 145, 205],
            [175, 195, 70],
            [130, 165, 245],
            [220, 160, 110],
            [180, 210, 200],
            [220, 220, 220],
        ]
    } else {
        [
            [0, 100, 170],
            [30, 120, 55],
            [150, 95, 0],
            [180, 40, 45],
            [105, 65, 165],
            [0, 125, 125],
            [160, 60, 120],
            [100, 115, 0],
            [50, 75, 175],
            [140, 80, 35],
            [65, 85, 100],
            [70, 70, 70],
        ]
    };
    let [r, g, b] = colors[index % colors.len()];
    egui::Color32::from_rgb(r, g, b)
}

struct Group {
    label: &'static str,
    fields: &'static [Field],
    weight: f32,
}

fn groups(width: f32) -> Vec<Group> {
    if width >= 1050.0 {
        Field::ALL
            .into_iter()
            .map(|field| Group {
                label: field.label(),
                fields: match field {
                    Field::Terminal => &[Field::Terminal],
                    Field::Diameter => &[Field::Diameter],
                    Field::Gt => &[Field::Gt],
                    Field::Published => &[Field::Published],
                    Field::Peak => &[Field::Peak],
                    Field::Average => &[Field::Average],
                    Field::Data => &[Field::Data],
                    Field::Limit => &[Field::Limit],
                },
                weight: if field == Field::Terminal { 2.0 } else { 1.0 },
            })
            .collect()
    } else if width >= 650.0 {
        vec![
            Group {
                label: "Terminal",
                fields: &[Field::Terminal],
                weight: 1.2,
            },
            Group {
                label: "Published specifications",
                fields: &[Field::Diameter, Field::Gt, Field::Published],
                weight: 1.8,
            },
            Group {
                label: "Estimated rate",
                fields: &[Field::Peak, Field::Average],
                weight: 1.4,
            },
            Group {
                label: "Estimated pass data",
                fields: &[Field::Data, Field::Limit],
                weight: 1.4,
            },
        ]
    } else {
        vec![
            Group {
                label: "Terminal",
                fields: &[Field::Terminal],
                weight: 1.0,
            },
            Group {
                label: "Specifications and estimates",
                fields: &Field::ALL[1..],
                weight: 2.0,
            },
        ]
    }
}

fn cell_text(
    field: Field,
    terminal: &Terminal,
    s: &Scenario,
    row: &Result<Estimate, &'static str>,
) -> String {
    let rx = terminal.receive(s.band);
    let e = row.as_ref().ok();
    match field {
        Field::Terminal => terminal.name.to_owned(),
        Field::Diameter => format!("{:.1} m\n{}", terminal.diameter_m, terminal.band_labels()),
        Field::Gt => rx.map_or_else(
            || "Not supported".to_owned(),
            |rx| {
                format!(
                    "{}{:.2}\n{}",
                    if rx.gt_is_floor { ">" } else { "" },
                    rx.gt_dbk,
                    rx.reference_elevation_deg.map_or_else(
                        || "Ref. elevation not specified".to_owned(),
                        |el| format!("at {el:.0} deg")
                    )
                )
            },
        ),
        Field::Published => rx
            .and_then(|rx| rx.system_max_mbps)
            .map_or_else(|| "Not specified".to_owned(), |value| format!("{value:.0}")),
        Field::Peak => e.map_or_else(
            || "Unavailable".to_owned(),
            |e| format!("{:.1}", e.peak_mbps),
        ),
        Field::Average => e.map_or_else(
            || "Unavailable".to_owned(),
            |e| format!("{:.1}", e.average_mbps),
        ),
        Field::Data => e.map_or_else(|| "Unavailable".to_owned(), |e| format!("{:.2}", e.data_gb)),
        Field::Limit => match row {
            Ok(e) => e.peak_limit.label().to_owned(),
            Err(error) => (*error).to_owned(),
        },
    }
}

fn group_text(group: &Group, terminal: &Terminal, c: &Comparison, index: usize) -> String {
    group
        .fields
        .iter()
        .map(|&field| {
            let text = cell_text(field, terminal, &c.scenario, &c.rows[index]);
            if group.fields.len() > 1 {
                format!("{}: {text}", field.label())
            } else {
                text
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn text_height(ui: &egui::Ui, text: &str, width: f32) -> f32 {
    let font = egui::TextStyle::Body.resolve(ui.style());
    ui.fonts_mut(|fonts| {
        fonts
            .layout(
                text.to_owned(),
                font,
                ui.visuals().text_color(),
                width.max(1.0),
            )
            .size()
            .y
    })
}

fn terminal_table(ui: &mut egui::Ui, c: &Comparison, selected: &mut [bool], sort: &mut Sort) {
    let width = (ui.available_width() - 1.0).max(1.0);
    let groups = groups(width);
    let weight: f32 = groups.iter().map(|g| g.weight).sum();
    let space = ui.spacing().item_spacing;
    let usable = width - space.x * (groups.len() - 1) as f32;
    let widths: Vec<_> = groups.iter().map(|g| usable * g.weight / weight).collect();
    let indices = sort.indices(c);
    let texts: Vec<Vec<_>> = indices
        .iter()
        .map(|&index| {
            groups
                .iter()
                .map(|group| group_text(group, &TERMINALS[index], c, index))
                .collect()
        })
        .collect();
    let heights: Vec<_> = texts
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(i, text)| {
                    text_height(ui, text, widths[i] - if i == 0 { 26.0 } else { 0.0 })
                        + if i == 0 { 24.0 } else { 0.0 }
                })
                .fold(0.0_f32, f32::max)
                .ceil()
                + 8.0
        })
        .collect();
    let header_height = groups
        .iter()
        .enumerate()
        .map(|(i, group)| text_height(ui, group.label, widths[i] - 16.0))
        .fold(24.0_f32, f32::max)
        + 6.0;
    ui.scope(|ui| {
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
        let mut table = TableBuilder::new(ui).id_salt("ground_terminal_table")
            .vscroll(false).striped(true).cell_layout(egui::Layout::top_down(egui::Align::Min));
        for &width in &widths { table = table.column(Column::exact(width).clip(true)); }
        table.header(header_height, |mut header| {
            for (column, group) in groups.iter().enumerate() {
                header.col(|ui| {
                    let rect = ui.max_rect();
                    let label_rect = egui::Rect::from_min_max(rect.min, egui::pos2((rect.right() - 16.0).max(rect.left()), rect.bottom()));
                    ui.scope_builder(egui::UiBuilder::new().max_rect(label_rect), |ui| {
                        ui.add(egui::Label::new(egui::RichText::new(group.label).strong()).wrap());
                    });
                    let active = group.fields.contains(&sort.field);
                    if active {
                        let source = if sort.descending { egui::include_image!("../assets/lucide-arrow-down.svg") }
                            else { egui::include_image!("../assets/lucide-arrow-up.svg") };
                        let icon = egui::Rect::from_min_size(egui::pos2(rect.right() - 12.0, rect.top() + 2.0), egui::vec2(12.0, 12.0));
                        ui.put(icon, egui::Image::new(source).tint(ui.visuals().text_color()));
                    }
                    if ui.interact(rect, ui.id().with(("terminal_sort", column)), egui::Sense::click())
                        .on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                        sort.toggle(if active { sort.field } else { group.fields[0] });
                    }
                });
            }
        }).body(|body| {
            body.heterogeneous_rows(heights.into_iter(), |mut row| {
                let position = row.index();
                let index = indices[position];
                let terminal = &TERMINALS[index];
                for (column, group) in groups.iter().enumerate() {
                    row.col(|ui| {
                            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
                        if column == 0 {
                            let color = series_color(index, ui.visuals().dark_mode);
                            ui.checkbox(&mut selected[index], egui::RichText::new(terminal.name).color(color))
                                .on_hover_text(terminal.conditions);
                            ui.hyperlink_to("Source", terminal.source).on_hover_text(terminal.source_label);
                        } else {
                            ui.add(egui::Label::new(&texts[position][column]).wrap())
                                .on_hover_ui(|ui| {
                                    ui.label(terminal.conditions);
                                    if group.fields.contains(&Field::Gt) {
                                        if let Some(rx) = terminal.receive(c.scenario.band) {
                                            ui.label(format!("Receive range: {:.3}-{:.3} GHz. G/T is gain divided by system noise temperature.", rx.low_ghz, rx.high_ghz));
                                        }
                                    }
                                });
                        }
                    });
                }
            });
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_sort_keeps_missing_values_last_in_both_directions() {
        let c = Comparison::calculate(Scenario::default(), TERMINALS).unwrap();
        for descending in [false, true] {
            for field in [
                Field::Diameter,
                Field::Gt,
                Field::Published,
                Field::Peak,
                Field::Average,
                Field::Data,
            ] {
                let indices = Sort { field, descending }.indices(&c);
                for pair in indices.windows(2) {
                    let value =
                        |i: usize| field.value(&TERMINALS[i], &c.scenario, c.rows[i].as_ref().ok());
                    assert_ne!(
                        compare_optional(value(pair[0]), value(pair[1]), descending),
                        Ordering::Greater
                    );
                }
            }
        }
    }

    #[test]
    fn every_column_is_preserved_in_narrow_layouts() {
        for width in [300.0, 650.0, 1000.0, 1440.0] {
            let fields: Vec<_> = groups(width)
                .iter()
                .flat_map(|g| g.fields.iter().copied())
                .collect();
            assert!(fields == Field::ALL);
        }
    }

    #[test]
    fn band_switch_does_not_treat_missing_hardware_as_zero_rate() {
        let s = Scenario {
            band: Band::Ka,
            frequency_ghz: 26.25,
            bandwidth_mhz: 500.0,
            ..Default::default()
        };
        let c = Comparison::calculate(s, TERMINALS).unwrap();
        assert_eq!(c.rows.iter().filter(|row| row.is_err()).count(), 3);
        assert!(
            cell_text(Field::Published, &TERMINALS[0], &c.scenario, &c.rows[0])
                .contains("Not specified")
        );
        assert_eq!(
            cell_text(Field::Peak, &TERMINALS[8], &c.scenario, &c.rows[8]),
            "Unavailable"
        );
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "ground_terminal_tests.rs"]
mod ui_tests;
