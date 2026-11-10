//! A dated snapshot of operational spacecraft, independent of available meshes.
use crate::satellite_model::{Facts, ASSETS};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

pub(crate) const SOURCE: &str = "https://space.oscar.wmo.int/satellites";

#[derive(Deserialize, Serialize)]
pub(crate) struct Catalogue {
    pub checked: String,
    pub source: String,
    pub satellites: Vec<Satellite>,
}

#[derive(Clone, Deserialize, Serialize)]
pub(crate) struct Satellite {
    pub id: u32,
    pub slug: String,
    pub name: String,
    pub full_name: String,
    pub agency: String,
    pub status: String,
    pub launched: String,
    pub orbit: String,
    pub altitude_km: Option<String>,
    pub payload: Vec<String>,
    #[serde(default)]
    pub launch_mass: Option<String>,
    #[serde(default)]
    pub dry_mass: Option<String>,
    #[serde(default)]
    pub power: Option<String>,
    #[serde(default)]
    pub updated: Option<String>,
}

pub(crate) struct Dimensions {
    pub label: &'static str,
    pub metres: [f32; 3],
    pub source: &'static str,
}

pub(crate) fn catalogue() -> &'static Catalogue {
    static CATALOGUE: OnceLock<Catalogue> = OnceLock::new();
    CATALOGUE.get_or_init(|| {
        let mut catalogue: Catalogue =
            serde_json::from_str(include_str!("../assets/satellite-catalogue.json"))
                .expect("validated satellite catalogue");
        assert_eq!(catalogue.source, SOURCE);
        let date =
            NaiveDate::parse_from_str(&catalogue.checked, "%Y-%m-%d").expect("catalogue date");
        catalogue
            .satellites
            .retain(|satellite| satellite.eligible(date));
        catalogue
    })
}

impl Satellite {
    pub fn model_index(&self) -> Option<usize> {
        match self.name.as_str() {
            "Sentinel-6A" | "Sentinel-6B" => Some(0),
            "Aqua" => Some(1),
            "Landsat-8" => Some(2),
            "ICESat-2" => Some(3),
            "SMAP" => Some(4),
            "GOES-18" | "GOES-19" => Some(6),
            "SWOT" => Some(7),
            "GRACE C (GRACE-FO)" | "GRACE D (GRACE-FO)" => Some(8),
            "OCO-2" => Some(9),
            "Aura" => Some(10),
            "GPM Core Observatory" => Some(11),
            "SNPP" => Some(12),
            _ => None,
        }
    }

    pub fn facts(&self) -> Option<&'static Facts> {
        if self.slug == "worldview_3" {
            Some(&WORLDVIEW3)
        } else if self.slug.starts_with("worldview_legion_") {
            Some(&LEGION)
        } else {
            self.model_index().map(|index| ASSETS[index].facts)
        }
    }

    pub fn dimensions(&self) -> Option<Dimensions> {
        if self.slug == "worldview_3" {
            Some(Dimensions {
                label: "Height / body width / solar-array span",
                metres: [5.7, 2.5, 7.1],
                source: WORLDVIEW3_SOURCE,
            })
        } else if self.slug.starts_with("worldview_legion_") {
            Some(Dimensions {
                label: "Body; excludes solar-array width (approx.)",
                metres: [3.0, 2.0, 2.0],
                source: LEGION_SOURCE,
            })
        } else {
            self.model_index().map(|index| {
                let asset = &ASSETS[index];
                Dimensions {
                    label: asset.dimension_label,
                    metres: asset.reference_dimensions,
                    source: asset.dimension_source,
                }
            })
        }
    }

    pub fn source(&self) -> String {
        format!("https://space.oscar.wmo.int/satellites/view/{}", self.slug)
    }

    pub fn matches(&self, query: &str) -> bool {
        [&self.name, &self.full_name, &self.agency, &self.orbit]
            .into_iter()
            .chain(self.payload.iter())
            .any(|text| text.to_lowercase().contains(query))
    }

    fn eligible(&self, date: NaiveDate) -> bool {
        self.status == "Operational"
            && launch_date(&self.launched).is_some_and(|launch| launch <= date)
            && !self.name.starts_with("ISS ")
            && self.name != "TEMPO" // A hosted instrument, not a separate spacecraft.
            && !RETIREMENTS.iter().any(|(slug, _)| *slug == self.slug)
    }
}

pub(crate) fn launch_date(text: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(text, "%Y-%m-%d")
        .or_else(|_| NaiveDate::parse_from_str(text, "%d %b %Y"))
        .or_else(|_| NaiveDate::parse_from_str(&format!("01 {text}"), "%d %b %Y"))
        .or_else(|_| NaiveDate::parse_from_str(&format!("01 Jan {text}"), "%d %b %Y"))
        .ok()
}

// Operator notices take precedence over stale OSCAR status flags.
const RETIREMENTS: &[(&str, &str)] = &[
    (
        "vens",
        "https://cnes.fr/actualites/fin-de-mission-venms-vigie-de-vegetation-terrestre",
    ),
    ("rasat", "https://uzay.tubitak.gov.tr/en/rasat/"),
    ("spot_7", "https://dinamis.data-terra.org/en/spot-6-7-2/"),
];

const WORLDVIEW3_SOURCE: &str =
    "https://imagengeo.com/files/Imagenes_satelitales/MAXAR/World_View_3/WV3.pdf";
const LEGION_SOURCE: &str = "https://www.l3harris.com/sites/default/files/2025-08/worldview-legion-data-sheet-l3harris-sas-ss.pdf";

const WORLDVIEW3: Facts = Facts {
    description: "A high-resolution optical Earth-imaging satellite with visible, near-infrared and short-wave infrared instruments.",
    rows: &[
        ("Launch mass", "2,800 kg"),
        ("Design orbit", "617 km; Sun-synchronous (published design)"),
        ("Solar power", "3.1 kW solar array"),
        ("Battery", "100 Ah"),
        ("Recorder", "2,199 Gbit solid-state storage with EDAC"),
        ("Downlink", "800 or 1,200 Mbit/s X-band"),
        ("Resolution", "0.31 m panchromatic; 1.24 m multispectral; 3.7 m SWIR"),
        ("Swath", "13.1 km at nadir"),
    ],
    sources: &[("Maxar data sheet", WORLDVIEW3_SOURCE)],
};

const LEGION: Facts = Facts {
    description:
        "A commercial optical Earth-observation spacecraft in the WorldView Legion constellation.",
    rows: &[
        ("Dry mass", "Approximately 630 kg; excludes propellant"),
        ("Design orbit", "518 km nominal (published design)"),
        (
            "Resolution",
            "0.34 m panchromatic; 1.36 m multispectral at nadir",
        ),
        ("Swath", "10 km at nadir"),
        ("Design life", "More than 10 years"),
    ],
    sources: &[("Maxar data sheet (2024)", LEGION_SOURCE)],
};

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Datelike;
    use std::collections::HashSet;

    #[test]
    fn snapshot_contains_only_launched_operational_spacecraft() {
        let catalogue = catalogue();
        let date = NaiveDate::parse_from_str(&catalogue.checked, "%Y-%m-%d").unwrap();
        assert_eq!(catalogue.source, SOURCE);
        assert!(catalogue.satellites.len() > 300);
        let mut ids = HashSet::new();
        let mut slugs = HashSet::new();
        for satellite in &catalogue.satellites {
            assert!(satellite.eligible(date), "{}", satellite.name);
            assert!(ids.insert(satellite.id));
            assert!(slugs.insert(&satellite.slug));
            assert!(!satellite.name.is_empty());
            assert!(satellite
                .source()
                .starts_with("https://space.oscar.wmo.int/"));
        }
        for name in [
            "MiRaTa",
            "WorldView-4",
            "Landsat-7",
            "GOES-16",
            "VEN\u{b5}S",
            "RASAT",
            "SPOT-7",
        ] {
            assert!(!catalogue.satellites.iter().any(|s| s.name == name));
        }
    }

    #[test]
    fn models_are_optional_and_worldview_specs_keep_their_qualifiers() {
        let catalogue = catalogue();
        let worldview = catalogue
            .satellites
            .iter()
            .find(|s| s.slug == "worldview_3")
            .unwrap();
        assert!(worldview.model_index().is_none());
        assert!(worldview.matches("worldview"));
        assert!(worldview.matches("maxar"));
        assert!(worldview.dimensions().is_some());
        for satellite in catalogue
            .satellites
            .iter()
            .filter(|s| s.slug.starts_with("worldview_legion_"))
        {
            assert_eq!(satellite.facts().unwrap().rows[0].0, "Dry mass");
        }
        assert_eq!(
            catalogue
                .satellites
                .iter()
                .filter(|s| s.slug.starts_with("worldview_legion_"))
                .count(),
            6
        );
        assert!(catalogue
            .satellites
            .iter()
            .any(|s| s.model_index() == Some(1)));
    }

    #[test]
    fn additional_nasa_models_are_available_from_the_operational_table() {
        let catalogue = catalogue();
        for (name, asset) in [
            ("OCO-2", 9),
            ("Aura", 10),
            ("GPM Core Observatory", 11),
            ("SNPP", 12),
        ] {
            let satellite = catalogue
                .satellites
                .iter()
                .find(|s| s.name == name)
                .unwrap();
            assert_eq!(satellite.model_index(), Some(asset));
            assert!(satellite.dimensions().is_some());
            let facts = satellite.facts().unwrap();
            assert_eq!(facts.description, ASSETS[asset].facts.description);
            assert_eq!(facts.rows, ASSETS[asset].facts.rows);
            assert_eq!(facts.sources, ASSETS[asset].facts.sources);
            assert!(ASSETS[asset]
                .facts
                .sources
                .iter()
                .any(|(_, url)| url.contains("nasa.gov")));
        }
        assert!(
            catalogue
                .satellites
                .iter()
                .find(|s| s.name == "Terra")
                .unwrap()
                .model_index()
                .is_none(),
            "Terra must not expose NASA's asset with missing external textures"
        );
    }

    #[test]
    fn status_filter_does_not_treat_being_in_orbit_as_being_in_use() {
        let mut satellite = catalogue().satellites[0].clone();
        let date = NaiveDate::from_ymd_opt(2026, 10, 30).unwrap();
        for status in [
            "Inactive",
            "Unclear",
            "Stand-by",
            "Planned",
            "Commissioning",
        ] {
            satellite.status = status.to_owned();
            assert!(!satellite.eligible(date));
        }
        satellite.status = "Operational".to_owned();
        satellite.launched = "2030-01-01".to_owned();
        assert!(!satellite.eligible(date));
        satellite.launched = "Unknown".to_owned();
        assert!(!satellite.eligible(date));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "downloads the public WMO catalogue and regenerates the bundled snapshot"]
    fn refresh_operational_catalogue() {
        #[derive(Deserialize)]
        struct Page {
            #[serde(rename = "_embedded")]
            embedded: Embedded,
            page: Pagination,
        }
        #[derive(Deserialize)]
        struct Pagination {
            #[serde(rename = "totalPages")]
            total_pages: usize,
            #[serde(rename = "totalElements")]
            total_elements: usize,
            number: usize,
        }
        #[derive(Deserialize)]
        struct Embedded {
            satellites: Vec<Record>,
        }
        #[derive(Deserialize)]
        struct Record {
            id: u32,
            slug: String,
            acronym: String,
            fullname: String,
            space_agency: String,
            status: String,
            launch_date: String,
            orbit: String,
            #[serde(rename = "Altitude")]
            altitude: Option<String>,
            #[serde(rename = "satellite-instruments")]
            instruments: Vec<Flight>,
        }
        #[derive(Deserialize)]
        struct Flight {
            status: String,
            instrument: Instrument,
        }
        #[derive(Deserialize)]
        struct Instrument {
            name: String,
        }

        let date = chrono::Utc::now().date_naive();
        assert!(date.year() >= 2026);
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(45))
            .build();
        let mut satellites = Vec::new();
        let mut seen = HashSet::new();
        let mut expected = None;
        for page in 1..=100 {
            let url = format!("https://space.oscar.wmo.int/api/v1/satellites?filterOn=status&filterValue=Operational&sort=acronym&order=asc&page={page}");
            let response: Page =
                serde_json::from_reader(agent.get(&url).call().unwrap().into_reader()).unwrap();
            assert_eq!(response.page.number, page);
            let expected = *expected.get_or_insert(response.page.total_elements);
            assert_eq!(
                expected, response.page.total_elements,
                "catalogue changed during download; retry"
            );
            for record in response.embedded.satellites {
                assert!(seen.insert(record.id), "duplicate satellite across pages");
                let mut payload: Vec<_> = record
                    .instruments
                    .into_iter()
                    .filter_map(|flight| {
                        if flight.status.starts_with("active") {
                            Some(flight.instrument.name)
                        } else if flight.status.starts_with("degraded") {
                            Some(format!("{} (degraded)", flight.instrument.name))
                        } else {
                            None
                        }
                    })
                    .collect();
                payload.sort();
                payload.dedup();
                let satellite = Satellite {
                    id: record.id,
                    slug: record.slug,
                    name: record.acronym,
                    full_name: record.fullname,
                    agency: record.space_agency,
                    status: record.status,
                    launched: record.launch_date,
                    orbit: record.orbit,
                    altitude_km: record.altitude,
                    payload,
                    launch_mass: None,
                    dry_mass: None,
                    power: None,
                    updated: None,
                };
                if satellite.eligible(date) {
                    satellites.push(satellite);
                } else {
                    eprintln!("Excluded {}", satellite.name);
                }
            }
            eprintln!("OSCAR page {page}/{}", response.page.total_pages);
            if page == response.page.total_pages {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
        assert_eq!(Some(seen.len()), expected, "incomplete catalogue");
        assert!(satellites.len() > 300);
        for (index, satellite) in satellites.iter_mut().enumerate() {
            let html = agent
                .get(&satellite.source())
                .call()
                .unwrap()
                .into_string()
                .unwrap();
            let fields = detail_fields(&html);
            assert_eq!(
                fields.get("Status").map(String::as_str),
                Some("Operational"),
                "{} changed status; refresh again",
                satellite.name
            );
            satellite.launch_mass = fields
                .get("Mass at launch")
                .cloned()
                .filter(|s| !s.is_empty());
            satellite.dry_mass = fields.get("Dry mass").cloned().filter(|s| !s.is_empty());
            satellite.power = fields.get("Power").cloned().filter(|s| !s.is_empty());
            satellite.updated = fields
                .get("Last update:")
                .cloned()
                .filter(|s| !s.is_empty());
            if index % 25 == 0 {
                eprintln!("Spacecraft specifications: {index}");
            }
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
        satellites.sort_by_key(|satellite| satellite.name.to_lowercase());
        let catalogue = Catalogue {
            checked: date.to_string(),
            source: SOURCE.to_owned(),
            satellites,
        };
        let json = serde_json::to_string_pretty(&catalogue).unwrap();
        std::fs::write(
            crate::texture::asset_path("assets/satellite-catalogue.json"),
            format!("{json}\n"),
        )
        .unwrap();
        eprintln!(
            "Bundled {} operational entries, checked {}",
            catalogue.satellites.len(),
            catalogue.checked
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn detail_fields(html: &str) -> std::collections::HashMap<String, String> {
        let document = scraper::Html::parse_document(html);
        let rows = scraper::Selector::parse("fieldset table.smalltable tr").unwrap();
        let mut fields = std::collections::HashMap::new();
        for row in document.select(&rows) {
            let cells: Vec<_> = row
                .children()
                .filter_map(scraper::ElementRef::wrap)
                .filter(|cell| cell.value().name() == "td")
                .map(|cell| {
                    cell.text()
                        .collect::<String>()
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect();
            for pair in cells.chunks_exact(2) {
                fields.insert(pair[0].clone(), pair[1].clone());
            }
        }
        fields
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn detail_parser_preserves_mass_qualifiers_and_does_not_invent_missing_power() {
        let fields = detail_fields("<fieldset><table class='smalltable'><tbody><tr><td>Mass at launch</td><td>1200 kg</td><td><abbr title='Mass without fuel'>Dry mass</abbr></td><td>900 kg</td></tr><tr><td>Power</td><td> </td></tr><tr><td>Status</td><td><abbr>Operational</abbr></td></tr></tbody></table></fieldset>");
        assert_eq!(fields["Mass at launch"], "1200 kg");
        assert_eq!(fields["Dry mass"], "900 kg");
        assert_eq!(fields["Power"], "");
        assert_eq!(fields["Status"], "Operational");
    }
}
