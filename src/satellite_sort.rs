//! Sort the published catalogue values without changing their displayed qualifiers.
use super::*;
use crate::satellite_catalogue::{launch_date, Satellite};
use chrono::Datelike;
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(super) enum Field {
    Satellite,
    Launched,
    Mass,
    Orbit,
    Power,
    Payload,
    Dimensions,
    Agency,
    References,
}

impl Field {
    pub const ALL: [Self; 9] = [
        Self::Satellite,
        Self::Launched,
        Self::Mass,
        Self::Orbit,
        Self::Power,
        Self::Payload,
        Self::Dimensions,
        Self::Agency,
        Self::References,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Satellite => "Satellite",
            Self::Launched => "Launched",
            Self::Mass => "Mass",
            Self::Orbit => "Orbit altitude",
            Self::Power => "Power",
            Self::Payload => "Payload",
            Self::Dimensions => "Largest dimension",
            Self::Agency => "Agency",
            Self::References => "References",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Mass => "First published mass in kilograms; mass qualifiers remain as published",
            Self::Orbit => "Published altitude in kilometres",
            Self::Power => "First published power value in watts; generation and demand qualifiers remain as published",
            Self::Dimensions => "Largest published dimension in metres; dimension references may differ",
            Self::References => "Reference titles, alphabetically",
            Self::Payload => "Payload, alphabetically",
            Self::Agency => "Agency, alphabetically",
            Self::Satellite => "Satellite name, alphabetically",
            Self::Launched => "Launch date, chronologically",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Sort {
    pub field: Field,
    pub descending: bool,
}

impl Default for Sort {
    fn default() -> Self {
        Self {
            field: Field::Satellite,
            descending: false,
        }
    }
}

enum Value {
    Number(f64),
    Text(String),
}

impl Value {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Number(a), Self::Number(b)) => a.total_cmp(b),
            (Self::Text(a), Self::Text(b)) => a.cmp(b),
            (Self::Number(_), Self::Text(_)) => Ordering::Less,
            (Self::Text(_), Self::Number(_)) => Ordering::Greater,
        }
    }
}

impl Sort {
    pub fn toggle(&mut self, field: Field) {
        if self.field == field {
            self.descending = !self.descending;
        } else {
            self.field = field;
            self.descending = false;
        }
    }

    pub fn apply(&self, satellites: &mut Vec<&Satellite>) {
        let mut keyed: Vec<_> = satellites
            .iter()
            .map(|&satellite| {
                (
                    self.key(satellite),
                    satellite.name.to_lowercase(),
                    satellite,
                )
            })
            .collect();
        keyed.sort_by(|(a, name_a, satellite_a), (b, name_b, satellite_b)| {
            // Missing values stay last in both directions, with deterministic name ties.
            match (a, b) {
                (Some(a), Some(b)) => {
                    let order = a.cmp(b);
                    if self.descending {
                        order.reverse()
                    } else {
                        order
                    }
                }
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            }
            .then_with(|| name_a.cmp(name_b))
            .then_with(|| satellite_a.id.cmp(&satellite_b.id))
        });
        for (slot, (_, _, satellite)) in satellites.iter_mut().zip(keyed) {
            *slot = satellite;
        }
    }

    fn key(&self, satellite: &Satellite) -> Option<Value> {
        let field = self.field;
        match field {
            Field::Satellite => text(&satellite.name),
            Field::Launched => launch_date(&satellite.launched)
                .map(|date| Value::Number(f64::from(date.num_days_from_ce()))),
            Field::Mass | Field::Orbit | Field::Power => {
                let fields = comparison::field_values(satellite, field as usize);
                let value = &fields.first()?.1;
                let number = match field {
                    Field::Mass => {
                        quantity(value, &[("kg", 1.0), ("g", 0.001), ("t", 1000.0)], None)
                    }
                    Field::Power => quantity(
                        value,
                        &[
                            ("W", 1.0),
                            ("kW", 1000.0),
                            ("MW", 1_000_000.0),
                            ("mW", 0.001),
                        ],
                        None,
                    ),
                    _ => quantity(value, &[("km", 1.0), ("m", 0.001)], Some(1.0)),
                }?;
                Some(Value::Number(number))
            }
            Field::Dimensions => satellite
                .dimensions()
                .and_then(|dimensions| {
                    dimensions
                        .metres
                        .into_iter()
                        .filter(|value| value.is_finite() && *value > 0.0)
                        .max_by(f32::total_cmp)
                })
                .map(|value| Value::Number(f64::from(value))),
            Field::Payload => text(&satellite.payload.join(", ")),
            Field::Agency => text(&satellite.agency),
            Field::References => {
                let titles = satellite
                    .facts()
                    .map(|facts| {
                        facts
                            .sources
                            .iter()
                            .map(|source| source.0)
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_else(|| "WMO OSCAR".to_owned());
                text(&titles)
            }
        }
    }

    pub fn icon(self) -> egui::ImageSource<'static> {
        if self.descending {
            egui::include_image!("../assets/lucide-arrow-down.svg")
        } else {
            egui::include_image!("../assets/lucide-arrow-up.svg")
        }
    }

    pub fn controls(&mut self, ui: &mut egui::Ui) -> bool {
        let previous = *self;
        ui.horizontal(|ui| {
            let width = (ui.available_width()
                - 14.0
                - 2.0 * ui.spacing().button_padding.x
                - ui.spacing().item_spacing.x)
                .max(1.0)
                .min(200.0);
            egui::ComboBox::from_id_salt("satellite_sort")
                .width(width)
                .wrap_mode(egui::TextWrapMode::Wrap)
                .selected_text(format!("Sort: {}", self.field.label()))
                .show_ui(ui, |ui| {
                    for field in Field::ALL {
                        if ui
                            .selectable_value(&mut self.field, field, field.label())
                            .changed()
                        {
                            self.descending = false;
                        }
                    }
                })
                .response
                .on_hover_text(self.field.description());
            if ui
                .add(egui::Button::image(
                    egui::Image::new(self.icon())
                        .fit_to_exact_size(egui::vec2(14.0, 14.0))
                        .tint(ui.visuals().text_color()),
                ))
                .on_hover_text(if self.descending {
                    "Descending"
                } else {
                    "Ascending"
                })
                .clicked()
            {
                self.descending = !self.descending;
            }
        });
        previous != *self
    }
}

fn text(value: &str) -> Option<Value> {
    let value = value.trim();
    (!value.is_empty()).then(|| Value::Text(value.to_lowercase()))
}

fn quantity(value: &str, units: &[(&str, f64)], default: Option<f64>) -> Option<f64> {
    let start = value.find(|c: char| c.is_ascii_digit() || c == '.')?;
    if value[..start].trim_end().ends_with('-') {
        return None;
    }
    let number = &value[start..];
    let mut end = number
        .bytes()
        .take_while(|c| c.is_ascii_digit() || matches!(c, b'.' | b','))
        .count();
    if matches!(number.as_bytes().get(end), Some(b'e' | b'E')) {
        end += 1;
        if matches!(number.as_bytes().get(end), Some(b'+' | b'-')) {
            end += 1;
        }
        end += number[end..].bytes().take_while(u8::is_ascii_digit).count();
    }
    let amount: f64 = number[..end].replace(',', "").parse().ok()?;
    let unit: String = number[end..]
        .trim_start()
        .chars()
        .take_while(char::is_ascii_alphabetic)
        .collect();
    let multiplier = if unit.is_empty() {
        default?
    } else {
        units.iter().find(|(name, _)| *name == unit)?.1
    };
    let value = amount * multiplier;
    (value.is_finite() && value >= 0.0).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::satellite_catalogue::catalogue;

    #[test]
    fn quantities_handle_units_qualifiers_commas_and_scientific_notation() {
        let watts = [("W", 1.0), ("kW", 1000.0), ("MW", 1e6), ("mW", 0.001)];
        for (value, expected) in [
            ("3.1 kW solar array", 3100.0),
            ("4,860 W at end of design life", 4860.0),
            ("10 W (2016 design)", 10.0),
            ("2 MW", 2e6),
            ("2 mW", 0.002),
        ] {
            assert_eq!(quantity(value, &watts, None), Some(expected));
        }
        assert_eq!(
            quantity(
                "Approximately 2,800 kg; includes fuel",
                &[("kg", 1.0)],
                None
            ),
            Some(2800.0)
        );
        assert_eq!(
            quantity("1.5e+06 km; L1", &[("km", 1.0)], Some(1.0)),
            Some(1.5e6)
        );
        for value in ["N/A", "Not specified", "", "100 bananas", "1e999 km"] {
            assert_eq!(quantity(value, &[("km", 1.0)], Some(1.0)), None);
        }
    }

    #[test]
    fn numeric_sort_and_missing_values_work_in_both_directions() {
        let mut satellites: Vec<_> = (0..4)
            .map(|index| {
                let mut satellite = catalogue().satellites[0].clone();
                satellite.id = index;
                satellite.name = format!("Test {index}");
                satellite.slug = format!("test_{index}");
                satellite.launch_mass = [Some("100 kg"), None, Some("9 kg"), Some("2 t")]
                    [index as usize]
                    .map(str::to_owned);
                satellite
            })
            .collect();
        satellites[1].dry_mass = None;
        let mut rows: Vec<_> = satellites.iter().collect();
        let mut sort = Sort {
            field: Field::Mass,
            descending: false,
        };
        sort.apply(&mut rows);
        assert_eq!(
            rows.iter().map(|row| row.id).collect::<Vec<_>>(),
            [2, 0, 3, 1]
        );
        sort.toggle(Field::Mass);
        sort.apply(&mut rows);
        assert_eq!(
            rows.iter().map(|row| row.id).collect::<Vec<_>>(),
            [3, 0, 2, 1]
        );
        sort.toggle(Field::Power);
        assert!(!sort.descending);
    }

    #[test]
    fn dates_are_chronological_and_ties_are_deterministic() {
        let mut satellites: Vec<_> = ["2020-01-01", "31 Dec 2019", "Jan 2000", "Unknown"]
            .into_iter()
            .enumerate()
            .map(|(index, date)| {
                let mut satellite = catalogue().satellites[0].clone();
                satellite.id = index as u32;
                satellite.launched = date.to_owned();
                satellite
            })
            .collect();
        let mut rows: Vec<_> = satellites.iter().collect();
        Sort {
            field: Field::Launched,
            descending: false,
        }
        .apply(&mut rows);
        assert_eq!(
            rows.iter().map(|row| row.id).collect::<Vec<_>>(),
            [2, 1, 0, 3]
        );
        satellites[1].name = "alpha".to_owned();
        satellites[0].name = "Zulu".to_owned();
        let mut rows: Vec<_> = satellites[..2].iter().collect();
        Sort {
            field: Field::Power,
            descending: true,
        }
        .apply(&mut rows);
        assert_eq!(rows[0].id, 1);
    }

    #[test]
    fn every_field_sorts_the_catalogue_without_losing_entries() {
        for field in Field::ALL {
            for descending in [false, true] {
                let mut rows: Vec<_> = catalogue().satellites.iter().collect();
                let sort = Sort { field, descending };
                sort.apply(&mut rows);
                assert_eq!(rows.len(), catalogue().satellites.len());
                let ids: std::collections::HashSet<_> = rows.iter().map(|row| row.id).collect();
                assert_eq!(ids.len(), rows.len());
                let first_missing = rows
                    .iter()
                    .position(|satellite| sort.key(satellite).is_none())
                    .unwrap_or(rows.len());
                assert!(rows[first_missing..]
                    .iter()
                    .all(|satellite| sort.key(satellite).is_none()));
            }
        }
    }
}
