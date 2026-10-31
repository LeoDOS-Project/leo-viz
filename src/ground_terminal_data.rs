//! Published receive specifications. Sources were checked on 2026-10-30.

pub const SAFRAN: &str = "https://www.safran-group.com/download/media/2010";
pub const VISION: &str =
    "https://www.safran-group.com/sites/default/files/2022-02/DS-VISION-A4.pdf";
pub const NASA: &str =
    "https://www.nasa.gov/smallsat-institute/sst-soa/ground-data-systems-and-mission-operations/";
pub const TUBITAK: &str =
    "https://uzay.tubitak.gov.tr/en/satellite-ground-station-and-operations/ground-station/";
pub const RECEIVER_SOURCE: &str = "https://www.viasat.com/content/dam/us-site/government/antenna-systems/documents/High_Rate_Receiver_VHR3200_Datasheet_3_30_22.pdf";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Band {
    S,
    X,
    Ka,
}

impl Band {
    pub const ALL: [Self; 3] = [Self::S, Self::X, Self::Ka];

    pub fn label(self) -> &'static str {
        match self {
            Self::S => "S",
            Self::X => "X",
            Self::Ka => "Ka",
        }
    }

    pub fn frequency_ghz(self) -> f64 {
        match self {
            Self::S => 2.25,
            Self::X => 8.2,
            Self::Ka => 26.25,
        }
    }

    pub fn bandwidth_mhz(self) -> f64 {
        match self {
            Self::S => 20.0,
            Self::X => 200.0,
            Self::Ka => 500.0,
        }
    }
}

pub struct ReceiveBand {
    pub band: Band,
    pub low_ghz: f64,
    pub high_ghz: f64,
    pub gt_dbk: f64,
    pub gt_is_floor: bool,
    pub reference_elevation_deg: Option<f64>,
    pub system_max_mbps: Option<f64>,
}

pub struct Terminal {
    pub name: &'static str,
    pub diameter_m: f64,
    pub bands: &'static [ReceiveBand],
    pub source: &'static str,
    pub source_label: &'static str,
    pub conditions: &'static str,
}

impl Terminal {
    pub fn receive(&self, band: Band) -> Option<&ReceiveBand> {
        self.bands.iter().find(|rx| rx.band == band)
    }

    pub fn band_labels(&self) -> String {
        self.bands
            .iter()
            .map(|rx| rx.band.label())
            .collect::<Vec<_>>()
            .join(" / ")
    }
}

const fn rx(
    band: Band,
    range: [f64; 2],
    gt_dbk: f64,
    gt_is_floor: bool,
    reference_elevation_deg: Option<f64>,
    system_max_mbps: Option<f64>,
) -> ReceiveBand {
    ReceiveBand {
        band,
        low_ghz: range[0],
        high_ghz: range[1],
        gt_dbk,
        gt_is_floor,
        reference_elevation_deg,
        system_max_mbps,
    }
}

const fn safran_bands(s: f64, x: f64, ka: f64, elevation: f64, x_floor: bool) -> [ReceiveBand; 3] {
    [
        rx(Band::S, [2.2, 2.3], s, true, Some(elevation), None),
        rx(Band::X, [7.9, 8.5], x, x_floor, Some(elevation), None),
        rx(Band::Ka, [25.5, 27.0], ka, true, Some(elevation), None),
    ]
}

const SAFRAN_CONDITIONS: &str = "Catalogue p. 12: mid-band G/T at 5 deg elevation, clear sky, +20 C. Optional S/X/Ka configuration; a bare antenna has no published Mbps ceiling.";
const VISION_CONDITIONS: &str = "Datasheet p. 2: tri-band configuration, G/T at 10 deg elevation. Tracking specified from 400 km altitude. Bi-band X sensitivity is different and is not used here.";
const RTE_CONDITIONS: &str = "NASA Table 11-8: the 5.4 m entry uses the Pendergrass/Guildford configuration; the 7.3 m entry uses Alice Springs/Accra. Other RTE stations differ. G/T reference elevation and a numeric system data-rate ceiling are not published. The estimate holds the listed G/T constant.";
const UZAY_CONDITIONS: &str = "Installed S/X station: published system downlink ceiling, not a bare-dish rating. G/T reference elevation is not specified; the estimate holds the listed G/T constant.";

pub const TERMINALS: &[Terminal] = &[
    Terminal {
        name: "Safran LEGION 400",
        diameter_m: 3.9,
        bands: &safran_bands(13.5, 26.5, 31.5, 5.0, false),
        source: SAFRAN,
        source_label: "Safran catalogue, p. 12",
        conditions: SAFRAN_CONDITIONS,
    },
    Terminal {
        name: "Safran VISION 550",
        diameter_m: 5.5,
        bands: &safran_bands(17.5, 29.5, 35.7, 10.0, true),
        source: VISION,
        source_label: "Safran VISION, p. 2",
        conditions: VISION_CONDITIONS,
    },
    Terminal {
        name: "Safran VISION 610",
        diameter_m: 6.1,
        bands: &safran_bands(18.5, 30.5, 36.7, 10.0, true),
        source: VISION,
        source_label: "Safran VISION, p. 2",
        conditions: VISION_CONDITIONS,
    },
    Terminal {
        name: "Safran VISION 730",
        diameter_m: 7.3,
        bands: &safran_bands(19.5, 32.0, 38.2, 10.0, true),
        source: VISION,
        source_label: "Safran VISION, p. 2",
        conditions: VISION_CONDITIONS,
    },
    Terminal {
        name: "Safran ORION 730",
        diameter_m: 7.3,
        bands: &safran_bands(19.5, 33.0, 37.5, 5.0, true),
        source: SAFRAN,
        source_label: "Safran catalogue, p. 12",
        conditions: SAFRAN_CONDITIONS,
    },
    Terminal {
        name: "Safran ORION 930",
        diameter_m: 9.3,
        bands: &safran_bands(21.5, 34.5, 39.5, 5.0, true),
        source: SAFRAN,
        source_label: "Safran catalogue, p. 12",
        conditions: SAFRAN_CONDITIONS,
    },
    Terminal {
        name: "Safran ORION 1100",
        diameter_m: 11.5,
        bands: &safran_bands(23.5, 36.5, 41.5, 5.0, true),
        source: SAFRAN,
        source_label: "Safran catalogue, p. 12",
        conditions: SAFRAN_CONDITIONS,
    },
    Terminal {
        name: "Safran ORION 1300",
        diameter_m: 13.5,
        bands: &safran_bands(25.0, 36.5, 43.0, 5.0, true),
        source: SAFRAN,
        source_label: "Safran catalogue, p. 12",
        conditions: SAFRAN_CONDITIONS,
    },
    Terminal {
        name: "Viasat RTE 5.4 m",
        diameter_m: 5.4,
        bands: &[
            rx(Band::S, [2.2, 2.29], 17.0, false, None, None),
            rx(Band::X, [8.025, 8.4], 30.0, false, None, None),
        ],
        source: NASA,
        source_label: "NASA, Table 11-8",
        conditions: RTE_CONDITIONS,
    },
    Terminal {
        name: "Viasat RTE 7.3 m",
        diameter_m: 7.3,
        bands: &[
            rx(Band::S, [2.2, 2.29], 18.0, false, None, None),
            rx(Band::X, [8.025, 8.4], 32.0, false, None, None),
            rx(Band::Ka, [25.5, 27.0], 34.5, false, None, None),
        ],
        source: NASA,
        source_label: "NASA, Table 11-8",
        conditions: RTE_CONDITIONS,
    },
    Terminal {
        name: "TUBITAK UZAY 5.4 m",
        diameter_m: 5.4,
        bands: &[
            rx(Band::S, [2.2, 2.29], 16.0, false, None, Some(2.0)),
            rx(Band::X, [8.0, 8.4], 28.0, false, None, Some(100.0)),
        ],
        source: TUBITAK,
        source_label: "TUBITAK UZAY",
        conditions: UZAY_CONDITIONS,
    },
    Terminal {
        name: "TUBITAK UZAY 7.3 m",
        diameter_m: 7.3,
        bands: &[
            rx(Band::S, [2.2, 2.4], 19.98, false, None, Some(1000.0)),
            rx(Band::X, [8.0, 8.4], 31.7, false, None, Some(1000.0)),
        ],
        source: TUBITAK,
        source_label: "TUBITAK UZAY",
        conditions: UZAY_CONDITIONS,
    },
];
