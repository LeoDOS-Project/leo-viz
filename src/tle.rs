//! Two-Line Element (TLE) satellite tracking.
//!
//! Parses and manages TLE data from CelesTrak for real satellite positions.
//! Supports 50+ satellite groups including Starlink, GPS, ISS, and debris.
//! Includes WASM-specific async fetching and incremental parsing.

use sgp4::Constants;

use crate::celestial::CelestialBody;

pub const SECONDS_PER_DAY: f64 = 86400.0;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum TlePreset {
    Starlink, OneWeb, Kuiper, Geo, Intelsat, Ses, Iridium, IridiumNext,
    Globalstar, Orbcomm, Molniya, Swarm, Amateur, XComm, OtherComm, Satnogs,
    Gps, Galileo, Glonass, Beidou, Gnss, Sbas, Nnss, Musson,
    Weather, Noaa, Goes, EarthResources, Sarsat, DisasterMon, Tdrss, Argos, Planet, Spire,
    Stations, Last30Days, Brightest100, ActiveSats, Analyst, Science,
    Geodetic, Engineering, Education, Military, RadarCal, Cubesats, Misc,
    Fengyun1cDebris, Cosmos2251Debris, Iridium33Debris, Cosmos1408Debris,
}

impl TlePreset {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Starlink => "Starlink",
            Self::OneWeb => "OneWeb",
            Self::Kuiper => "Kuiper",
            Self::Geo => "GEO",
            Self::Intelsat => "Intelsat",
            Self::Ses => "SES",
            Self::Iridium => "Iridium",
            Self::IridiumNext => "Iridium NEXT",
            Self::Globalstar => "Globalstar",
            Self::Orbcomm => "Orbcomm",
            Self::Molniya => "Molniya",
            Self::Swarm => "Swarm",
            Self::Amateur => "Amateur",
            Self::XComm => "X-Comm",
            Self::OtherComm => "Other Comm",
            Self::Satnogs => "SatNOGS",
            Self::Gps => "GPS",
            Self::Galileo => "Galileo",
            Self::Glonass => "GLONASS",
            Self::Beidou => "Beidou",
            Self::Gnss => "GNSS",
            Self::Sbas => "SBAS",
            Self::Nnss => "NNSS",
            Self::Musson => "Musson",
            Self::Weather => "Weather",
            Self::Noaa => "NOAA",
            Self::Goes => "GOES",
            Self::EarthResources => "Earth Res.",
            Self::Sarsat => "SARSAT",
            Self::DisasterMon => "DMC",
            Self::Tdrss => "TDRSS",
            Self::Argos => "ARGOS",
            Self::Planet => "Planet",
            Self::Spire => "Spire",
            Self::Stations => "Stations",
            Self::Last30Days => "Last 30 Days",
            Self::Brightest100 => "100 Brightest",
            Self::ActiveSats => "Active",
            Self::Analyst => "Analyst",
            Self::Science => "Science",
            Self::Geodetic => "Geodetic",
            Self::Engineering => "Engineering",
            Self::Education => "Education",
            Self::Military => "Military",
            Self::RadarCal => "Radar Cal.",
            Self::Cubesats => "CubeSats",
            Self::Misc => "Misc",
            Self::Fengyun1cDebris => "Fengyun 1C",
            Self::Cosmos2251Debris => "Cosmos 2251",
            Self::Iridium33Debris => "Iridium 33",
            Self::Cosmos1408Debris => "Cosmos 1408",
        }
    }

    pub fn url(&self) -> &'static str {
        match self {
            Self::Starlink => "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
            Self::OneWeb => "https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle",
            Self::Kuiper => "https://celestrak.org/NORAD/elements/gp.php?GROUP=kuiper&FORMAT=tle",
            Self::Geo => "https://celestrak.org/NORAD/elements/gp.php?GROUP=geo&FORMAT=tle",
            Self::Intelsat => "https://celestrak.org/NORAD/elements/gp.php?GROUP=intelsat&FORMAT=tle",
            Self::Ses => "https://celestrak.org/NORAD/elements/gp.php?GROUP=ses&FORMAT=tle",
            Self::Iridium => "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium&FORMAT=tle",
            Self::IridiumNext => "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-NEXT&FORMAT=tle",
            Self::Globalstar => "https://celestrak.org/NORAD/elements/gp.php?GROUP=globalstar&FORMAT=tle",
            Self::Orbcomm => "https://celestrak.org/NORAD/elements/gp.php?GROUP=orbcomm&FORMAT=tle",
            Self::Molniya => "https://celestrak.org/NORAD/elements/gp.php?GROUP=molniya&FORMAT=tle",
            Self::Swarm => "https://celestrak.org/NORAD/elements/gp.php?GROUP=swarm&FORMAT=tle",
            Self::Amateur => "https://celestrak.org/NORAD/elements/gp.php?GROUP=amateur&FORMAT=tle",
            Self::XComm => "https://celestrak.org/NORAD/elements/gp.php?GROUP=x-comm&FORMAT=tle",
            Self::OtherComm => "https://celestrak.org/NORAD/elements/gp.php?GROUP=other-comm&FORMAT=tle",
            Self::Satnogs => "https://celestrak.org/NORAD/elements/gp.php?GROUP=satnogs&FORMAT=tle",
            Self::Gps => "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle",
            Self::Galileo => "https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle",
            Self::Glonass => "https://celestrak.org/NORAD/elements/gp.php?GROUP=glo-ops&FORMAT=tle",
            Self::Beidou => "https://celestrak.org/NORAD/elements/gp.php?GROUP=beidou&FORMAT=tle",
            Self::Gnss => "https://celestrak.org/NORAD/elements/gp.php?GROUP=gnss&FORMAT=tle",
            Self::Sbas => "https://celestrak.org/NORAD/elements/gp.php?GROUP=sbas&FORMAT=tle",
            Self::Nnss => "https://celestrak.org/NORAD/elements/gp.php?GROUP=nnss&FORMAT=tle",
            Self::Musson => "https://celestrak.org/NORAD/elements/gp.php?GROUP=musson&FORMAT=tle",
            Self::Weather => "https://celestrak.org/NORAD/elements/gp.php?GROUP=weather&FORMAT=tle",
            Self::Noaa => "https://celestrak.org/NORAD/elements/gp.php?GROUP=noaa&FORMAT=tle",
            Self::Goes => "https://celestrak.org/NORAD/elements/gp.php?GROUP=goes&FORMAT=tle",
            Self::EarthResources => "https://celestrak.org/NORAD/elements/gp.php?GROUP=resource&FORMAT=tle",
            Self::Sarsat => "https://celestrak.org/NORAD/elements/gp.php?GROUP=sarsat&FORMAT=tle",
            Self::DisasterMon => "https://celestrak.org/NORAD/elements/gp.php?GROUP=dmc&FORMAT=tle",
            Self::Tdrss => "https://celestrak.org/NORAD/elements/gp.php?GROUP=tdrss&FORMAT=tle",
            Self::Argos => "https://celestrak.org/NORAD/elements/gp.php?GROUP=argos&FORMAT=tle",
            Self::Planet => "https://celestrak.org/NORAD/elements/gp.php?GROUP=planet&FORMAT=tle",
            Self::Spire => "https://celestrak.org/NORAD/elements/gp.php?GROUP=spire&FORMAT=tle",
            Self::Stations => "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle",
            Self::Last30Days => "https://celestrak.org/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=tle",
            Self::Brightest100 => "https://celestrak.org/NORAD/elements/gp.php?GROUP=visual&FORMAT=tle",
            Self::ActiveSats => "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
            Self::Analyst => "https://celestrak.org/NORAD/elements/gp.php?GROUP=analyst&FORMAT=tle",
            Self::Science => "https://celestrak.org/NORAD/elements/gp.php?GROUP=science&FORMAT=tle",
            Self::Geodetic => "https://celestrak.org/NORAD/elements/gp.php?GROUP=geodetic&FORMAT=tle",
            Self::Engineering => "https://celestrak.org/NORAD/elements/gp.php?GROUP=engineering&FORMAT=tle",
            Self::Education => "https://celestrak.org/NORAD/elements/gp.php?GROUP=education&FORMAT=tle",
            Self::Military => "https://celestrak.org/NORAD/elements/gp.php?GROUP=military&FORMAT=tle",
            Self::RadarCal => "https://celestrak.org/NORAD/elements/gp.php?GROUP=radar&FORMAT=tle",
            Self::Cubesats => "https://celestrak.org/NORAD/elements/gp.php?GROUP=cubesat&FORMAT=tle",
            Self::Misc => "https://celestrak.org/NORAD/elements/gp.php?GROUP=other&FORMAT=tle",
            Self::Fengyun1cDebris => "https://celestrak.org/NORAD/elements/gp.php?GROUP=fengyun-1c-debris&FORMAT=tle",
            Self::Cosmos2251Debris => "https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-2251-debris&FORMAT=tle",
            Self::Iridium33Debris => "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-33-debris&FORMAT=tle",
            Self::Cosmos1408Debris => "https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-1408-debris&FORMAT=tle",
        }
    }

    pub fn category(&self) -> &'static str {
        match self {
            Self::Starlink | Self::OneWeb | Self::Kuiper | Self::Geo |
            Self::Intelsat | Self::Ses | Self::Iridium | Self::IridiumNext |
            Self::Globalstar | Self::Orbcomm | Self::Molniya | Self::Swarm |
            Self::Amateur | Self::XComm | Self::OtherComm | Self::Satnogs => "Comms",
            Self::Gps | Self::Galileo | Self::Glonass | Self::Beidou |
            Self::Gnss | Self::Sbas | Self::Nnss | Self::Musson => "Navigation",
            Self::Weather | Self::Noaa | Self::Goes | Self::EarthResources |
            Self::Sarsat | Self::DisasterMon | Self::Tdrss | Self::Argos |
            Self::Planet | Self::Spire => "Observation",
            Self::Fengyun1cDebris | Self::Cosmos2251Debris |
            Self::Iridium33Debris | Self::Cosmos1408Debris => "Debris",
            _ => "Other",
        }
    }

    pub const ALL: [TlePreset; 51] = [
        Self::Starlink, Self::OneWeb, Self::Kuiper, Self::Geo,
        Self::Intelsat, Self::Ses, Self::Iridium, Self::IridiumNext,
        Self::Globalstar, Self::Orbcomm, Self::Molniya, Self::Swarm,
        Self::Amateur, Self::XComm, Self::OtherComm, Self::Satnogs,
        Self::Gps, Self::Galileo, Self::Glonass, Self::Beidou,
        Self::Gnss, Self::Sbas, Self::Nnss, Self::Musson,
        Self::Weather, Self::Noaa, Self::Goes, Self::EarthResources,
        Self::Sarsat, Self::DisasterMon, Self::Tdrss, Self::Argos,
        Self::Planet, Self::Spire,
        Self::Stations, Self::Last30Days, Self::Brightest100,
        Self::ActiveSats, Self::Analyst, Self::Science,
        Self::Geodetic, Self::Engineering, Self::Education,
        Self::Military, Self::RadarCal, Self::Cubesats, Self::Misc,
        Self::Fengyun1cDebris, Self::Cosmos2251Debris,
        Self::Iridium33Debris, Self::Cosmos1408Debris,
    ];

    pub fn is_debris(&self) -> bool {
        matches!(self, Self::Fengyun1cDebris | Self::Cosmos2251Debris | Self::Iridium33Debris | Self::Cosmos1408Debris)
    }

    pub fn color_index(&self) -> usize {
        Self::ALL.iter().position(|p| std::mem::discriminant(p) == std::mem::discriminant(self)).unwrap_or(0)
    }
}

#[derive(Clone)]
pub struct TleSatellite {
    pub name: String,
    pub constants: Constants,
    pub epoch_minutes: f64,
    pub inclination_deg: f64,
    pub mean_motion: f64,
}

#[derive(Clone)]
pub struct TleShell {
    pub label: String,
    pub satellite_indices: Vec<usize>,
    pub color_offset: usize,
    pub selected: bool,
}

#[derive(Clone)]
#[allow(dead_code)]
pub enum TleLoadState {
    NotLoaded,
    Loading,
    Loaded { satellites: Vec<TleSatellite> },
    Failed(String),
}

pub fn mean_motion_to_altitude_km(n_revs_per_day: f64) -> f64 {
    let mu = CelestialBody::Earth.mu();
    let r_earth = CelestialBody::Earth.radius_km();
    let n_rad_s = n_revs_per_day * 2.0 * std::f64::consts::PI / SECONDS_PER_DAY;
    let a = (mu / (n_rad_s * n_rad_s)).powf(1.0 / 3.0);
    a - r_earth
}

pub fn datetime_to_minutes(dt: &sgp4::chrono::NaiveDateTime) -> f64 {
    dt.and_utc().timestamp() as f64 / 60.0
}

pub fn parse_tle_data(data: &str) -> Result<Vec<TleSatellite>, String> {
    let lines: Vec<&str> = data.lines().collect();
    let mut satellites = Vec::new();

    let mut i = 0;
    while i + 2 < lines.len() {
        let name_line = lines[i].trim();
        let line1 = lines[i + 1].trim();
        let line2 = lines[i + 2].trim();

        if !line1.starts_with('1') || !line2.starts_with('2') {
            i += 1;
            continue;
        }

        let tle = format!("{}\n{}\n{}", name_line, line1, line2);

        if let Ok(elements_vec) = sgp4::parse_3les(&tle) {
            for elements in elements_vec {
                if let Ok(constants) = Constants::from_elements(&elements) {
                    let epoch_minutes = datetime_to_minutes(&elements.datetime);
                    satellites.push(TleSatellite {
                        name: elements.object_name.unwrap_or_default(),
                        inclination_deg: elements.inclination,
                        mean_motion: elements.mean_motion,
                        constants,
                        epoch_minutes,
                    });
                }
            }
        }

        i += 3;
    }

    if satellites.is_empty() {
        Err("No valid TLE data found".to_string())
    } else {
        Ok(satellites)
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn fetch_tle_data(url: &str) -> Result<Vec<TleSatellite>, String> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("HTTP error: {}", e))?;

    let body = response.into_string()
        .map_err(|e| format!("Read error: {}", e))?;

    parse_tle_data(&body)
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    pub(crate) static TLE_FETCH_RESULT: std::cell::RefCell<Vec<(TlePreset, Result<Vec<TleSatellite>, String>)>> = std::cell::RefCell::new(Vec::new());
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn fetch_tle_text(url: &str) -> Result<String, String> {
    use wasm_bindgen::JsCast as _;
    use web_sys::{Request, RequestInit, Response};

    let opts = RequestInit::new();
    opts.set_method("GET");

    let request = Request::new_with_str_and_init(url, &opts)
        .map_err(|e| format!("{:?}", e))?;

    let window = web_sys::window().ok_or("No window")?;
    let resp_value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|e| format!("Fetch failed: {:?}", e))?;

    let resp: Response = resp_value.dyn_into()
        .map_err(|_| "Response is not a Response")?;

    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }

    let array_buffer = wasm_bindgen_futures::JsFuture::from(
        resp.array_buffer().map_err(|e| format!("{:?}", e))?
    )
    .await
    .map_err(|e| format!("{:?}", e))?;

    let bytes = js_sys::Uint8Array::new(&array_buffer).to_vec();
    String::from_utf8(bytes).map_err(|e| format!("{}", e))
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn yield_now() {
    wasm_bindgen_futures::JsFuture::from(
        js_sys::Promise::new(&mut |resolve, _| {
            web_sys::window().unwrap()
                .set_timeout_with_callback(&resolve).unwrap();
        })
    ).await.unwrap();
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn parse_tle_data_async(data: &str) -> Result<Vec<TleSatellite>, String> {
    let lines: Vec<&str> = data.lines().collect();
    let mut satellites = Vec::new();
    let mut i = 0;
    let mut batch = 0;
    while i + 2 < lines.len() {
        let name_line = lines[i].trim();
        let line1 = lines[i + 1].trim();
        let line2 = lines[i + 2].trim();
        if !line1.starts_with('1') || !line2.starts_with('2') {
            i += 1;
            continue;
        }
        let tle = format!("{}\n{}\n{}", name_line, line1, line2);
        if let Ok(elements_vec) = sgp4::parse_3les(&tle) {
            for elements in elements_vec {
                if let Ok(constants) = Constants::from_elements(&elements) {
                    let epoch_minutes = datetime_to_minutes(&elements.datetime);
                    satellites.push(TleSatellite {
                        name: elements.object_name.unwrap_or_default(),
                        inclination_deg: elements.inclination,
                        mean_motion: elements.mean_motion,
                        constants,
                        epoch_minutes,
                    });
                }
            }
        }
        i += 3;
        batch += 1;
        if batch % 100 == 0 {
            yield_now().await;
        }
    }
    if satellites.is_empty() {
        Err("No valid TLE data found".to_string())
    } else {
        Ok(satellites)
    }
}
