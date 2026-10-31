//! Published mission/design figures, not live spacecraft telemetry.

pub(crate) struct Facts {
    pub description: &'static str,
    pub rows: &'static [(&'static str, &'static str)],
    pub sources: &'static [(&'static str, &'static str)],
}

pub(super) const OCO2: Facts = Facts {
    description: "An observatory that measures carbon dioxide from the way sunlight is absorbed after reflecting from Earth. Three high-resolution spectrometers observe oxygen and carbon-dioxide absorption bands, helping scientists locate sources and sinks of this greenhouse gas.",
    rows: &[
        ("Launched", "2014-07-02"),
        ("Design orbit", "705 km; Sun-synchronous"),
        ("Solar span", "About 9 m deployed"),
        ("Design demand", "815 W (observatory)"),
        ("Payload", "Three-band imaging grating spectrometer"),
    ],
    sources: &[
        ("Spacecraft", "https://ocov2.jpl.nasa.gov/observatory/spacecraft/"),
        ("Instrument", "https://ocov2.jpl.nasa.gov/observatory/instrument/"),
        ("Press kit", "https://www.jpl.nasa.gov/news/press_kits/oco2-launch-press-kit.pdf"),
    ],
};

pub(super) const AURA: Facts = Facts {
    description: "An atmospheric-chemistry observatory that measures ozone, trace gases and aerosols. Its instruments study the ozone layer, air quality and the interaction between atmospheric chemistry and climate. The model retains the original four-instrument configuration; HIRDLS and TES are no longer operating.",
    rows: &[
        ("Launched", "2004-07-15"),
        ("Launch mass", "2,967 kg"),
        ("Design orbit", "705 km; Sun-synchronous"),
        ("Power spec", "4,600 W"),
        ("Original payload", "HIRDLS, MLS, OMI, TES"),
    ],
    sources: &[
        ("Instruments", "https://science.nasa.gov/mission/aura/spacecraft-and-instruments/"),
        ("Press kit", "https://web.archive.org/web/20170701000000/http://www.nasa.gov/pdf/60914main_AuraPressKit6-30.pdf"),
        ("Mission status", "https://science.nasa.gov/missions/potential-evolution-of-the-aura-mission/"),
    ],
};

pub(super) const GPM: Facts = Facts {
    description: "The reference satellite for an international network measuring rain and snow. A microwave imager senses natural emissions, while two radar frequencies measure precipitation inside clouds. Combining these observations helps distinguish rainfall, snowfall and the size of precipitation particles.",
    rows: &[
        ("Launched", "2014-02-27"),
        ("Launch mass", "3,850 kg with fuel"),
        ("Design orbit", "407 km; 65 deg (original orbit)"),
        ("Avg. power", "1,950 W"),
        ("Science link", "300 kbit/s via TDRSS"),
        ("Payload", "GMI microwave imager; Ku/Ka precipitation radar"),
    ],
    sources: &[
        ("Spacecraft", "https://gpm.nasa.gov/missions/GPM/gpm-core-observatory"),
        ("Press kit", "https://gpm.nasa.gov/sites/default/files/imce/GPM-Press-Kit_2014.pdf"),
    ],
};

pub(super) const SNPP: Facts = Facts {
    description: "A polar-orbiting weather and climate satellite that observes clouds, oceans, vegetation, ice and the atmosphere. It bridges NASA's Earth Observing System and the JPSS weather-satellite series. Its sensors support forecasting, wildfire detection and long-term environmental records.",
    rows: &[
        ("Launched", "2011-10-28"),
        ("Design orbit", "824 km; Sun-synchronous"),
        ("Platform", "Ball Aerospace BCP 2000"),
        ("Original payload", "VIIRS, CrIS, ATMS, OMPS, CERES"),
        ("Planned deorbit", "December 2026 (NOAA announcement)"),
    ],
    sources: &[
        ("NASA mission", "https://science.nasa.gov/mission/suomi-npp/"),
        ("NASA press kit", "https://web.archive.org/web/20111109220848/http://npp.gsfc.nasa.gov/images/NPP_PressKit_Color.pdf"),
        ("Spacecraft", "https://www.eoportal.org/satellite-missions/suomi-npp#spacecraft"),
        ("NOAA status", "https://www.nesdis.noaa.gov/news/noaa-deorbit-the-suomi-npp-satellite-after-more-15-years-service"),
    ],
};

pub(super) const SENTINEL6: Facts = Facts {
    description: "An ocean-monitoring satellite that uses radar to measure sea-surface height, extending the long-term record of sea-level change. It also analyses navigation-satellite signals passing through the atmosphere to measure temperature and humidity.",
    rows: &[
        ("Launched", "2020-11-21 (Michael Freilich)"),
        ("Launch mass", "1,191 kg, including fuel"),
        ("Design orbit", "1,336 km; 66 deg"),
        ("Avg. demand", "891 W"),
        ("Science data", "150 Mbit/s X-band"),
        ("Payload", "Poseidon-4, AMR-C, orbit tracking, GNSS-RO"),
    ],
    sources: &[
        ("Mission", "https://www.jpl.nasa.gov/missions/sentinel-6/"),
        (
            "Specifications",
            "https://repository.library.noaa.gov/view/noaa/41814/noaa_41814_DS1.pdf",
        ),
    ],
};

pub(super) const AQUA: Facts = Facts {
    description: "An Earth-observation satellite designed to study how water moves through the oceans, atmosphere, ice and land. Its instruments observe clouds, water vapour, precipitation and temperature, and measure incoming and outgoing radiation to study Earth's energy balance.",
    rows: &[
        ("Launched", "2002-05-04"),
        ("Launch mass", "2,934 kg"),
        ("Design orbit", "705 km; 98.2 deg"),
        ("Power spec", "4,860 W at end of design life"),
        ("Design life", "6 years"),
        ("Payload", "AIRS, AMSU-A, HSB, AMSR-E, MODIS, CERES"),
    ],
    sources: &[
        ("Mission", "https://science.nasa.gov/mission/aqua/"),
        (
            "NASA specifications",
            "https://science.nasa.gov/mission/aqua/about-aqua-spacecraft-and-instruments/",
        ),
    ],
};

pub(super) const LANDSAT8: Facts = Facts {
    description: "An Earth-imaging satellite that records visible, infrared and thermal radiation. Its OLI and TIRS instruments reveal changes in vegetation, land use, glaciers and surface temperature, helping monitor agriculture, urban growth and environmental change over time.",
    rows: &[
        ("Launched", "2013-02-11"),
        ("Bus mass", "2,071 kg with fuel; excludes instruments"),
        ("Design orbit", "705 km; 98.2 deg"),
        ("Recorder", "3.14 Tbit"),
        ("Downlink", "384 Mbit/s X-band"),
        ("Payload", "OLI and TIRS"),
    ],
    sources: &[
        (
            "NASA mission",
            "https://science.nasa.gov/mission/landsat-8/",
        ),
        (
            "USGS specifications",
            "https://www.usgs.gov/landsat-missions/landsat-8",
        ),
    ],
};

pub(super) const ICESAT2: Facts = Facts {
    description: "A laser-altimetry satellite that times returning photons to measure surface elevation. Repeated observations track changes in ice sheets, glaciers and sea ice, helping scientists understand how Earth's frozen regions are changing.",
    rows: &[
        ("Launched", "2018-09-15"),
        ("Launch mass", "1,580 kg (mission fact sheet)"),
        ("Design orbit", "About 500 km; 92 deg"),
        ("Power spec", "1,374 W"),
        ("Design life", "3 years; fuel for 7"),
        ("Payload", "ATLAS photon-counting laser altimeter"),
    ],
    sources: &[
        ("NASA mission", "https://science.nasa.gov/mission/icesat-2/"),
        ("Mission fact sheet", "https://icesat-2.gsfc.nasa.gov/sites/default/files/page_files/FactSheet_0828_2018_0.pdf"),
    ],
};

pub(super) const SMAP: Facts = Facts {
    description: "A satellite that maps soil moisture and whether the ground is frozen or thawed. Its rotating, gold-coated mesh antenna collects natural microwave emissions, supporting research into droughts, flooding and the movement of water between land and atmosphere.",
    rows: &[
        ("Launched", "2015-01-31"),
        ("Launch mass", "944 kg, including fuel and instruments"),
        ("Design orbit", "685 km; near-polar, Sun-synchronous"),
        ("Power spec", "1,450 W"),
        ("Reflector", "6 m deployable mesh"),
        (
            "Payload",
            "L-band radiometer and radar (launch configuration)",
        ),
    ],
    sources: &[(
        "NASA/JPL press kit",
        "https://www.jpl.nasa.gov/news/press_kits/smaplaunch.pdf",
    )],
};

pub(super) const MIRATA: Facts = Facts {
    description: "A 3U CubeSat designed to test compact instruments for atmospheric temperature, water vapour and cloud-ice measurements. Its microwave radiometer and GPS radio-occultation receiver were intended to demonstrate a new way to calibrate small weather sensors in orbit.",
    rows: &[
        ("Launched", "2017-11-18"),
        ("Mass", "4.5 kg"),
        ("Format", "3U CubeSat"),
        ("Avg. power", "10 W (2016 design)"),
        ("Payload", "Tri-band microwave radiometer and GPS radio occultation"),
    ],
    sources: &[
        ("MIT design figures", "https://www.haystack.mit.edu/wp-content/uploads/2020/07/conf_NEROC2016_cahoy_web.pdf"),
        ("NASA launch record", "https://blogs.nasa.gov/jpss/"),
        ("NASA payload", "https://www.nasa.gov/wp-content/uploads/2016/07/elana_14_jpss1_cubesat_factsheet_508.pdf"),
    ],
};

pub(super) const GOESR: Facts = Facts {
    description: "A family of geostationary satellites that observe weather, lightning and the space environment. The Earth-facing instruments monitor developing storms, while the Sun-facing instruments watch solar activity that can affect satellites, communications and power systems.",
    rows: &[
        ("Launched", "2016-2024 (series)"),
        ("Launch mass", "5,192 kg with fuel (R/S/T)"),
        ("Design orbit", "35,786 km; geostationary"),
        ("Design life", "10 years operating + 5 in storage"),
        ("MAG boom", "8.5 m deployed"),
        ("Payload", "ABI, GLM, SUVI, EXIS, SEISS, MAG"),
    ],
    sources: &[
        ("Spacecraft", "https://www.nesdis.noaa.gov/our-satellites/currently-flying/goes-east-west/goes-r-series-spacecraft-instruments"),
        ("Handbook", "https://ospo.noaa.gov/resources/documents/GOES-RSeriesDataBook.pdf"),
        ("Series", "https://www.ncei.noaa.gov/products/satellite/goes-r"),
    ],
};

pub(super) const SWOT: Facts = Facts {
    description: "An international satellite that measures the height of water in rivers, lakes, reservoirs and the ocean. Its two radar antennas observe wide strips on either side of the spacecraft, while a separate altimeter measures the strip directly below it. Repeated measurements help track freshwater storage and ocean circulation.",
    rows: &[
        ("Launched", "2022-12-16"),
        ("Launch mass", "About 2,200 kg, including fuel"),
        ("Design orbit", "891 km; 78 deg"),
        ("Solar power", "8 kW; 1.5 kW spacecraft demand"),
        ("Radar baseline", "10 m between KaRIn antennas"),
        ("Payload", "KaRIn, nadir altimeter, AMR, orbit tracking"),
    ],
    sources: &[
        ("Mission", "https://science.nasa.gov/mission/swot/"),
        ("Specifications", "https://www.jpl.nasa.gov/press-kits/swot/quick-facts/"),
    ],
};

pub(super) const GRACEFO: Facts = Facts {
    description: "A pair of satellites that map changes in Earth's gravity by measuring tiny changes in the distance between them. Those measurements reveal shifts in groundwater, ice and ocean mass. Microwave ranging is complemented by a laser interferometer, while accelerometers distinguish gravity effects from forces such as atmospheric drag.",
    rows: &[
        ("Launched", "2018-05-22"),
        ("Launch mass", "600.2 kg each, including propellant"),
        ("Design orbit", "About 490 km; 89 deg"),
        ("Avg. power", "355 W per satellite"),
        ("Separation", "220 km nominal"),
        ("Payload", "K/Ka ranging, LRI, accelerometer, GPS"),
    ],
    sources: &[
        ("Mission", "https://science.nasa.gov/mission/grace-fo/"),
        ("Spacecraft", "https://www.jpl.nasa.gov/news/press_kits/grace-fo/mission/spacecraft/"),
        ("Specifications", "https://www.jpl.nasa.gov/news/press_kits/grace-fo/facts/"),
    ],
};
