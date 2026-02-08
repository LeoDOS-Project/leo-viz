//! Celestial body definitions and properties.
//!
//! Provides enums and data for planets, moons, and dwarf planets including
//! physical properties (radius, mass, rotation), texture mappings, and skins.

use eframe::egui;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum CelestialBody {
    Earth,
    Moon,
    Mars,
    Mercury,
    Venus,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
    Sun,
    Ceres,
    Haumea,
    Makemake,
    Eris,
    Pluto,
    PlanetNine,
    Ganymede,
    Callisto,
    Io,
    Europa,
    Titan,
    Triton,
    Charon,
    Enceladus,
    Mimas,
    Iapetus,
    Phobos,
    Vesta,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Skin {
    Default,
    Abstract,
    HellOnEarth,
    Terraformed,
    Civilized,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TextureResolution {
    R512,
    R1024,
    R2048,
    R8192,
    R21504,
}

impl TextureResolution {
    pub fn label(&self) -> &'static str {
        match self {
            TextureResolution::R512 => "512",
            TextureResolution::R1024 => "1K",
            TextureResolution::R2048 => "2K",
            TextureResolution::R8192 => "8K",
            TextureResolution::R21504 => "21K",
        }
    }

    pub fn downscale_factor(&self, body: CelestialBody, skin: Skin) -> u32 {
        match (body, skin, self) {
            (CelestialBody::Earth, Skin::Default, TextureResolution::R512) => 1,
            (_, _, TextureResolution::R512) => 4,
            (_, _, TextureResolution::R1024) => 2,
            _ => 1,
        }
    }

    pub fn cpu_render_size(&self) -> usize {
        match self {
            TextureResolution::R512 => 512,
            _ => 1024,
        }
    }

    pub fn cloud_filename(&self) -> Option<&'static str> {
        match self {
            TextureResolution::R8192 | TextureResolution::R21504 => Some("textures/earth/earth_clouds_8k.jpg"),
            _ => Some("textures/earth/earth_clouds_2k.jpg"),
        }
    }
}

impl Skin {
    pub fn label(&self) -> &'static str {
        match self {
            Skin::Default => "Default",
            Skin::Abstract => "Abstract",
            Skin::HellOnEarth => "Hell on Earth",
            Skin::Terraformed => "Terraformed",
            Skin::Civilized => "Civilized",
        }
    }

    pub fn filename(&self, body: CelestialBody, resolution: TextureResolution) -> Option<&'static str> {
        match (body, self, resolution) {
            (CelestialBody::Earth, Skin::Default, TextureResolution::R21504) => Some("textures/earth/Earth_Diffuse_21k.jpg"),
            (CelestialBody::Earth, Skin::Default, TextureResolution::R8192) => Some("textures/earth/earth_8k.jpg"),
            (CelestialBody::Earth, Skin::Default, TextureResolution::R512) => Some("textures/earth/earth_512.jpg"),
            (CelestialBody::Earth, Skin::Default, _) => Some("textures/earth/earth_2k.jpg"),
            (CelestialBody::Earth, Skin::HellOnEarth, _) => Some("textures/earth/hell_on_earth_2k.png"),
            (CelestialBody::Moon, Skin::Default, TextureResolution::R8192) => Some("textures/moon/moon_8k.jpg"),
            (CelestialBody::Moon, Skin::Default, _) => Some("textures/moon/moon_2k.jpg"),
            (CelestialBody::Mars, Skin::Default, TextureResolution::R8192) => Some("textures/mars/mars_8k.jpg"),
            (CelestialBody::Mars, Skin::Default, _) => Some("textures/mars/mars_2k.jpg"),
            (CelestialBody::Mars, Skin::Terraformed, _) => Some("textures/mars/mars_terraformed.png"),
            (CelestialBody::Mars, Skin::Civilized, _) => Some("textures/mars/mars_civilized.png"),
            (CelestialBody::Mercury, Skin::Default, TextureResolution::R21504) => Some("textures/mercury/Mercury_Diffuse_16k.jpg"),
            (CelestialBody::Mercury, Skin::Default, TextureResolution::R8192) => Some("textures/mercury/mercury_8k.jpg"),
            (CelestialBody::Mercury, Skin::Default, _) => Some("textures/mercury/mercury_2k.jpg"),
            (CelestialBody::Venus, Skin::Default, TextureResolution::R21504) => Some("textures/venus/Venus_Diffuse_16k.jpg"),
            (CelestialBody::Venus, Skin::Default, TextureResolution::R8192) => Some("textures/venus/venus_8k.jpg"),
            (CelestialBody::Venus, Skin::Default, _) => Some("textures/venus/venus_2k.jpg"),
            (CelestialBody::Jupiter, Skin::Default, TextureResolution::R8192) => Some("textures/jupiter/jupiter_8k.jpg"),
            (CelestialBody::Jupiter, Skin::Default, _) => Some("textures/jupiter/jupiter_2k.jpg"),
            (CelestialBody::Saturn, Skin::Default, TextureResolution::R8192) => Some("textures/saturn/saturn_8k.jpg"),
            (CelestialBody::Saturn, Skin::Default, _) => Some("textures/saturn/saturn_2k.jpg"),
            (CelestialBody::Uranus, Skin::Default, _) => Some("textures/uranus/uranus_2k.jpg"),
            (CelestialBody::Neptune, Skin::Default, _) => Some("textures/neptune/neptune_2k.jpg"),
            (CelestialBody::Sun, Skin::Default, TextureResolution::R8192) => Some("textures/sun/sun_8k.jpg"),
            (CelestialBody::Sun, Skin::Default, _) => Some("textures/sun/sun_2k.jpg"),
            (CelestialBody::Ceres, Skin::Default, TextureResolution::R8192) => Some("textures/ceres/ceres_4k.jpg"),
            (CelestialBody::Ceres, Skin::Default, _) => Some("textures/ceres/ceres_2k.jpg"),
            (CelestialBody::Haumea, Skin::Default, TextureResolution::R8192) => Some("textures/haumea/haumea_4k.jpg"),
            (CelestialBody::Haumea, Skin::Default, _) => Some("textures/haumea/haumea_2k.jpg"),
            (CelestialBody::Makemake, Skin::Default, TextureResolution::R8192) => Some("textures/makemake/makemake_4k.jpg"),
            (CelestialBody::Makemake, Skin::Default, _) => Some("textures/makemake/makemake_2k.jpg"),
            (CelestialBody::Eris, Skin::Default, TextureResolution::R8192) => Some("textures/eris/eris_4k.jpg"),
            (CelestialBody::Eris, Skin::Default, _) => Some("textures/eris/eris_2k.jpg"),
            (CelestialBody::Pluto, Skin::Default, _) => Some("textures/pluto/pluto_2k.jpg"),
            (CelestialBody::PlanetNine, Skin::Default, _) => Some("textures/planet_nine/planet_nine_2k.jpg"),
            (CelestialBody::Ganymede, Skin::Default, _) => Some("textures/ganymede/ganymede_4k.jpg"),
            (CelestialBody::Callisto, Skin::Default, _) => Some("textures/callisto/callisto_4k.jpg"),
            (CelestialBody::Io, Skin::Default, _) => Some("textures/io/io_2k.jpg"),
            (CelestialBody::Europa, Skin::Default, TextureResolution::R8192) => Some("textures/europa/europa_4k.png"),
            (CelestialBody::Europa, Skin::Default, _) => Some("textures/europa/europa_2k.jpg"),
            (CelestialBody::Titan, Skin::Default, TextureResolution::R8192) => Some("textures/titan/titan_4k.png"),
            (CelestialBody::Titan, Skin::Default, _) => Some("textures/titan/titan_2k.jpg"),
            (CelestialBody::Triton, Skin::Default, TextureResolution::R8192) => Some("textures/triton/triton_4k.png"),
            (CelestialBody::Triton, Skin::Default, _) => Some("textures/triton/triton_2k.jpg"),
            (CelestialBody::Charon, Skin::Default, TextureResolution::R8192) => Some("textures/charon/charon_4k.png"),
            (CelestialBody::Charon, Skin::Default, _) => Some("textures/charon/charon_2k.jpg"),
            (CelestialBody::Enceladus, Skin::Default, TextureResolution::R8192) => Some("textures/enceladus/enceladus_8k.jpg"),
            (CelestialBody::Enceladus, Skin::Default, _) => Some("textures/enceladus/enceladus_2k.jpg"),
            (CelestialBody::Mimas, Skin::Default, _) => Some("textures/mimas/mimas_2k.jpg"),
            (CelestialBody::Iapetus, Skin::Default, _) => Some("textures/iapetus/iapetus_2k.jpg"),
            (CelestialBody::Phobos, Skin::Default, _) => Some("textures/phobos/phobos_2k.jpg"),
            (CelestialBody::Vesta, Skin::Default, _) => Some("textures/vesta/vesta_4k.jpg"),
            _ => None,
        }
    }
}

impl CelestialBody {
    pub fn category(&self) -> &'static str {
        match self {
            CelestialBody::Sun => "Star",
            CelestialBody::Mercury
            | CelestialBody::Venus
            | CelestialBody::Earth
            | CelestialBody::Mars
            | CelestialBody::Jupiter
            | CelestialBody::Saturn
            | CelestialBody::Uranus
            | CelestialBody::Neptune => "Planets",
            CelestialBody::Ceres
            | CelestialBody::Pluto
            | CelestialBody::Haumea
            | CelestialBody::Makemake
            | CelestialBody::Eris
            | CelestialBody::PlanetNine => "Dwarf Planets",
            CelestialBody::Vesta => "Asteroids",
            _ => "Moons",
        }
    }

    pub fn parent_body(&self) -> Option<CelestialBody> {
        match self {
            CelestialBody::Moon => Some(CelestialBody::Earth),
            CelestialBody::Phobos => Some(CelestialBody::Mars),
            CelestialBody::Io
            | CelestialBody::Europa
            | CelestialBody::Ganymede
            | CelestialBody::Callisto => Some(CelestialBody::Jupiter),
            CelestialBody::Titan
            | CelestialBody::Enceladus
            | CelestialBody::Mimas
            | CelestialBody::Iapetus => Some(CelestialBody::Saturn),
            CelestialBody::Triton => Some(CelestialBody::Neptune),
            CelestialBody::Charon => Some(CelestialBody::Pluto),
            _ => None,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            CelestialBody::Earth => "Earth",
            CelestialBody::Moon => "Moon",
            CelestialBody::Mars => "Mars",
            CelestialBody::Mercury => "Mercury",
            CelestialBody::Venus => "Venus",
            CelestialBody::Jupiter => "Jupiter",
            CelestialBody::Saturn => "Saturn",
            CelestialBody::Uranus => "Uranus",
            CelestialBody::Neptune => "Neptune",
            CelestialBody::Sun => "Sun",
            CelestialBody::Ceres => "Ceres",
            CelestialBody::Haumea => "Haumea",
            CelestialBody::Makemake => "Makemake",
            CelestialBody::Eris => "Eris",
            CelestialBody::Pluto => "Pluto",
            CelestialBody::PlanetNine => "Planet Nine",
            CelestialBody::Ganymede => "Ganymede",
            CelestialBody::Callisto => "Callisto",
            CelestialBody::Io => "Io",
            CelestialBody::Europa => "Europa",
            CelestialBody::Titan => "Titan",
            CelestialBody::Triton => "Triton",
            CelestialBody::Charon => "Charon",
            CelestialBody::Enceladus => "Enceladus",
            CelestialBody::Mimas => "Mimas",
            CelestialBody::Iapetus => "Iapetus",
            CelestialBody::Phobos => "Phobos",
            CelestialBody::Vesta => "Vesta",
        }
    }

    pub fn available_skins(&self) -> &'static [Skin] {
        match self {
            CelestialBody::Earth => &[Skin::Default, Skin::Abstract, Skin::HellOnEarth],
            CelestialBody::Mars => &[Skin::Default, Skin::Terraformed, Skin::Civilized],
            _ => &[Skin::Default],
        }
    }

    pub const ALL: [CelestialBody; 28] = [
        CelestialBody::Sun,
        CelestialBody::Mercury,
        CelestialBody::Venus,
        CelestialBody::Earth,
        CelestialBody::Mars,
        CelestialBody::Jupiter,
        CelestialBody::Saturn,
        CelestialBody::Uranus,
        CelestialBody::Neptune,
        CelestialBody::Ceres,
        CelestialBody::Pluto,
        CelestialBody::Haumea,
        CelestialBody::Makemake,
        CelestialBody::Eris,
        CelestialBody::PlanetNine,
        CelestialBody::Vesta,
        CelestialBody::Moon,
        CelestialBody::Phobos,
        CelestialBody::Io,
        CelestialBody::Europa,
        CelestialBody::Ganymede,
        CelestialBody::Callisto,
        CelestialBody::Titan,
        CelestialBody::Enceladus,
        CelestialBody::Mimas,
        CelestialBody::Iapetus,
        CelestialBody::Triton,
        CelestialBody::Charon,
    ];

    pub fn radius_km(&self) -> f64 {
        match self {
            CelestialBody::Earth => 6371.0,
            CelestialBody::Moon => 1737.4,
            CelestialBody::Mars => 3389.5,
            CelestialBody::Mercury => 2439.7,
            CelestialBody::Venus => 6051.8,
            CelestialBody::Jupiter => 69911.0,
            CelestialBody::Saturn => 58232.0,
            CelestialBody::Uranus => 25362.0,
            CelestialBody::Neptune => 24622.0,
            CelestialBody::Sun => 696340.0,
            CelestialBody::Ceres => 473.0,
            CelestialBody::Haumea => 816.0,
            CelestialBody::Makemake => 715.0,
            CelestialBody::Eris => 1163.0,
            CelestialBody::Pluto => 1188.3,
            CelestialBody::PlanetNine => 13000.0,
            CelestialBody::Ganymede => 2634.1,
            CelestialBody::Callisto => 2410.3,
            CelestialBody::Io => 1821.6,
            CelestialBody::Europa => 1560.8,
            CelestialBody::Titan => 2574.7,
            CelestialBody::Triton => 1353.4,
            CelestialBody::Charon => 606.0,
            CelestialBody::Enceladus => 252.1,
            CelestialBody::Mimas => 198.2,
            CelestialBody::Iapetus => 734.5,
            CelestialBody::Phobos => 11.1,
            CelestialBody::Vesta => 262.7,
        }
    }

    pub fn mu(&self) -> f64 {
        match self {
            CelestialBody::Earth => 398600.4418,
            CelestialBody::Moon => 4902.8,
            CelestialBody::Mars => 42828.37,
            CelestialBody::Mercury => 22032.0,
            CelestialBody::Venus => 324859.0,
            CelestialBody::Jupiter => 126686534.0,
            CelestialBody::Saturn => 37931187.0,
            CelestialBody::Uranus => 5793939.0,
            CelestialBody::Neptune => 6836529.0,
            CelestialBody::Sun => 132712440018.0,
            CelestialBody::Ceres => 62.6,
            CelestialBody::Haumea => 2.67,
            CelestialBody::Makemake => 2.0,
            CelestialBody::Eris => 111.0,
            CelestialBody::Pluto => 869.6,
            CelestialBody::PlanetNine => 200000.0,
            CelestialBody::Ganymede => 9887.83,
            CelestialBody::Callisto => 7179.29,
            CelestialBody::Io => 5959.92,
            CelestialBody::Europa => 3202.71,
            CelestialBody::Titan => 8978.14,
            CelestialBody::Triton => 1427.6,
            CelestialBody::Charon => 105.88,
            CelestialBody::Enceladus => 7.21,
            CelestialBody::Mimas => 2.50,
            CelestialBody::Iapetus => 120.5,
            CelestialBody::Phobos => 0.0007,
            CelestialBody::Vesta => 17.8,
        }
    }

    pub fn j2(&self) -> f64 {
        match self {
            CelestialBody::Earth => 1.08263e-3,
            CelestialBody::Moon => 2.03e-4,
            CelestialBody::Mars => 1.96045e-3,
            CelestialBody::Mercury => 6.0e-5,
            CelestialBody::Venus => 4.458e-6,
            CelestialBody::Jupiter => 1.4736e-2,
            CelestialBody::Saturn => 1.6298e-2,
            CelestialBody::Uranus => 3.343e-3,
            CelestialBody::Neptune => 3.411e-3,
            CelestialBody::Sun => 2.0e-7,
            CelestialBody::Ceres => 0.0,
            CelestialBody::Haumea => 0.0,
            CelestialBody::Makemake => 0.0,
            CelestialBody::Eris => 0.0,
            CelestialBody::Pluto => 0.0,
            CelestialBody::PlanetNine => 0.0,
            CelestialBody::Ganymede => 0.0,
            CelestialBody::Callisto => 0.0,
            CelestialBody::Io => 0.0,
            CelestialBody::Europa => 0.0,
            CelestialBody::Titan => 0.0,
            CelestialBody::Triton => 0.0,
            CelestialBody::Charon => 0.0,
            CelestialBody::Enceladus => 0.0,
            CelestialBody::Mimas => 0.0,
            CelestialBody::Iapetus => 0.0,
            CelestialBody::Phobos => 0.0,
            CelestialBody::Vesta => 0.0,
        }
    }

    pub fn equatorial_radius_km(&self) -> f64 {
        match self {
            CelestialBody::Earth => 6378.137,
            CelestialBody::Moon => 1738.1,
            CelestialBody::Mars => 3396.2,
            CelestialBody::Mercury => 2440.5,
            CelestialBody::Venus => 6051.8,
            CelestialBody::Jupiter => 71492.0,
            CelestialBody::Saturn => 60268.0,
            CelestialBody::Uranus => 25559.0,
            CelestialBody::Neptune => 24764.0,
            CelestialBody::Sun => 696000.0,
            CelestialBody::Ceres => 473.0,
            CelestialBody::Haumea => 960.0,
            CelestialBody::Makemake => 715.0,
            CelestialBody::Eris => 1163.0,
            CelestialBody::Pluto => 1188.3,
            CelestialBody::PlanetNine => 13000.0,
            CelestialBody::Ganymede => 2634.1,
            CelestialBody::Callisto => 2410.3,
            CelestialBody::Io => 1829.4,
            CelestialBody::Europa => 1560.8,
            CelestialBody::Titan => 2574.7,
            CelestialBody::Triton => 1353.4,
            CelestialBody::Charon => 606.0,
            CelestialBody::Enceladus => 252.1,
            CelestialBody::Mimas => 198.2,
            CelestialBody::Iapetus => 734.5,
            CelestialBody::Phobos => 13.0,
            CelestialBody::Vesta => 286.3,
        }
    }

    pub fn flattening(&self) -> f64 {
        match self {
            CelestialBody::Earth => 1.0 / 298.257,
            CelestialBody::Moon => 0.0012,
            CelestialBody::Mars => 1.0 / 169.89,
            CelestialBody::Mercury => 0.0009,
            CelestialBody::Venus => 0.0,
            CelestialBody::Jupiter => 1.0 / 15.41,
            CelestialBody::Saturn => 1.0 / 10.21,
            CelestialBody::Uranus => 1.0 / 43.62,
            CelestialBody::Neptune => 1.0 / 58.54,
            CelestialBody::Sun => 9.0e-6,
            CelestialBody::Ceres => 0.0,
            CelestialBody::Haumea => 0.19,
            CelestialBody::Makemake => 0.0,
            CelestialBody::Eris => 0.0,
            CelestialBody::Pluto => 0.0,
            CelestialBody::PlanetNine => 0.02,
            CelestialBody::Ganymede => 0.0,
            CelestialBody::Callisto => 0.0,
            CelestialBody::Io => 0.0,
            CelestialBody::Europa => 0.0,
            CelestialBody::Titan => 0.0,
            CelestialBody::Triton => 0.0,
            CelestialBody::Charon => 0.0,
            CelestialBody::Enceladus => 0.0,
            CelestialBody::Mimas => 0.0,
            CelestialBody::Iapetus => 0.0,
            CelestialBody::Phobos => 0.0,
            CelestialBody::Vesta => 0.0,
        }
    }

    pub fn rotation_period_hours(&self) -> f64 {
        match self {
            CelestialBody::Earth => 23.9345,
            CelestialBody::Moon => 655.7,
            CelestialBody::Mars => 24.6229,
            CelestialBody::Mercury => 1407.6,
            CelestialBody::Venus => -5832.5,
            CelestialBody::Jupiter => 9.925,
            CelestialBody::Saturn => 10.656,
            CelestialBody::Uranus => -17.24,
            CelestialBody::Neptune => 16.11,
            CelestialBody::Sun => 609.12,
            CelestialBody::Ceres => 9.074,
            CelestialBody::Haumea => 3.92,
            CelestialBody::Makemake => 22.48,
            CelestialBody::Eris => 25.9,
            CelestialBody::Pluto => 153.29,
            CelestialBody::PlanetNine => 20.0,
            CelestialBody::Ganymede => 171.71,
            CelestialBody::Callisto => 400.54,
            CelestialBody::Io => 42.46,
            CelestialBody::Europa => 85.23,
            CelestialBody::Titan => 382.68,
            CelestialBody::Triton => -141.05,
            CelestialBody::Charon => 153.29,
            CelestialBody::Enceladus => 32.89,
            CelestialBody::Mimas => 22.62,
            CelestialBody::Iapetus => 1903.7,
            CelestialBody::Phobos => 7.66,
            CelestialBody::Vesta => 5.342,
        }
    }

    pub fn ring_params(&self) -> Option<(&'static str, f32, f32)> {
        match self {
            CelestialBody::Saturn => Some(("textures/saturn/saturn_ring_2k.png", 1.1, 2.3)),
            CelestialBody::Uranus => Some(("textures/uranus/uranus_ring.png", 1.459, 4.337)),
            CelestialBody::Neptune => Some(("textures/neptune/neptune_ring.png", 1.575, 2.665)),
            _ => None,
        }
    }

    pub fn semi_major_axis_au(&self) -> Option<f64> {
        match self {
            CelestialBody::Sun
            | CelestialBody::Moon
            | CelestialBody::Ganymede
            | CelestialBody::Callisto
            | CelestialBody::Io
            | CelestialBody::Europa
            | CelestialBody::Titan
            | CelestialBody::Triton
            | CelestialBody::Charon
            | CelestialBody::Enceladus
            | CelestialBody::Mimas
            | CelestialBody::Iapetus
            | CelestialBody::Phobos => None,
            CelestialBody::Vesta => Some(2.362),
            CelestialBody::Mercury => Some(0.387),
            CelestialBody::Venus => Some(0.723),
            CelestialBody::Earth => Some(1.0),
            CelestialBody::Mars => Some(1.524),
            CelestialBody::Ceres => Some(2.767),
            CelestialBody::Jupiter => Some(5.203),
            CelestialBody::Saturn => Some(9.537),
            CelestialBody::Uranus => Some(19.191),
            CelestialBody::Neptune => Some(30.069),
            CelestialBody::Haumea => Some(43.13),
            CelestialBody::Makemake => Some(45.79),
            CelestialBody::Eris => Some(67.67),
            CelestialBody::Pluto => Some(39.48),
            CelestialBody::PlanetNine => Some(460.0),
        }
    }

    pub fn orbital_period_days(&self) -> Option<f64> {
        match self {
            CelestialBody::Sun
            | CelestialBody::Moon
            | CelestialBody::Ganymede
            | CelestialBody::Callisto
            | CelestialBody::Io
            | CelestialBody::Europa
            | CelestialBody::Titan
            | CelestialBody::Triton
            | CelestialBody::Charon
            | CelestialBody::Enceladus
            | CelestialBody::Mimas
            | CelestialBody::Iapetus
            | CelestialBody::Phobos => None,
            CelestialBody::Vesta => Some(1325.75),
            CelestialBody::Mercury => Some(87.97),
            CelestialBody::Venus => Some(224.7),
            CelestialBody::Earth => Some(365.25),
            CelestialBody::Mars => Some(687.0),
            CelestialBody::Ceres => Some(1681.0),
            CelestialBody::Jupiter => Some(4331.0),
            CelestialBody::Saturn => Some(10747.0),
            CelestialBody::Uranus => Some(30589.0),
            CelestialBody::Neptune => Some(59800.0),
            CelestialBody::Haumea => Some(103468.0),
            CelestialBody::Makemake => Some(113183.0),
            CelestialBody::Eris => Some(203830.0),
            CelestialBody::Pluto => Some(90560.0),
            CelestialBody::PlanetNine => Some(7300000.0),
        }
    }

    #[allow(dead_code)]
    pub fn orbital_inclination_deg(&self) -> f64 {
        match self {
            CelestialBody::Sun => 0.0,
            CelestialBody::Mercury => 7.0,
            CelestialBody::Venus => 3.39,
            CelestialBody::Earth => 0.0,
            CelestialBody::Moon => 5.14,
            CelestialBody::Mars => 1.85,
            CelestialBody::Ceres => 10.6,
            CelestialBody::Jupiter => 1.31,
            CelestialBody::Saturn => 2.49,
            CelestialBody::Uranus => 0.77,
            CelestialBody::Neptune => 1.77,
            CelestialBody::Haumea => 28.2,
            CelestialBody::Makemake => 29.0,
            CelestialBody::Eris => 44.0,
            CelestialBody::Pluto => 17.16,
            CelestialBody::PlanetNine => 20.0,
            CelestialBody::Ganymede => 0.0,
            CelestialBody::Callisto => 0.0,
            CelestialBody::Io => 0.0,
            CelestialBody::Europa => 0.0,
            CelestialBody::Titan => 0.0,
            CelestialBody::Triton => 0.0,
            CelestialBody::Charon => 0.0,
            CelestialBody::Enceladus => 0.0,
            CelestialBody::Mimas => 0.0,
            CelestialBody::Iapetus => 0.0,
            CelestialBody::Phobos => 0.0,
            CelestialBody::Vesta => 7.14,
        }
    }

    pub fn mean_longitude_j2000_deg(&self) -> f64 {
        match self {
            CelestialBody::Sun => 0.0,
            CelestialBody::Moon => 0.0,
            CelestialBody::Mercury => 252.25,
            CelestialBody::Venus => 181.98,
            CelestialBody::Earth => 100.46,
            CelestialBody::Mars => 355.45,
            CelestialBody::Ceres => 153.89,
            CelestialBody::Jupiter => 34.40,
            CelestialBody::Saturn => 49.94,
            CelestialBody::Uranus => 313.23,
            CelestialBody::Neptune => 304.88,
            CelestialBody::Haumea => 118.0,
            CelestialBody::Makemake => 84.0,
            CelestialBody::Eris => 204.0,
            CelestialBody::Pluto => 238.0,
            CelestialBody::PlanetNine => 90.0,
            CelestialBody::Ganymede => 0.0,
            CelestialBody::Callisto => 0.0,
            CelestialBody::Io => 0.0,
            CelestialBody::Europa => 0.0,
            CelestialBody::Titan => 0.0,
            CelestialBody::Triton => 0.0,
            CelestialBody::Charon => 0.0,
            CelestialBody::Enceladus => 0.0,
            CelestialBody::Mimas => 0.0,
            CelestialBody::Iapetus => 0.0,
            CelestialBody::Phobos => 0.0,
            CelestialBody::Vesta => 236.0,
        }
    }

    pub fn display_color(&self) -> egui::Color32 {
        match self {
            CelestialBody::Mercury => egui::Color32::from_rgb(180, 160, 140),
            CelestialBody::Venus => egui::Color32::from_rgb(230, 210, 150),
            CelestialBody::Earth => egui::Color32::from_rgb(70, 130, 200),
            CelestialBody::Moon => egui::Color32::from_rgb(200, 200, 200),
            CelestialBody::Mars => egui::Color32::from_rgb(200, 100, 60),
            CelestialBody::Jupiter => egui::Color32::from_rgb(210, 180, 140),
            CelestialBody::Saturn => egui::Color32::from_rgb(220, 200, 130),
            CelestialBody::Uranus => egui::Color32::from_rgb(170, 220, 230),
            CelestialBody::Neptune => egui::Color32::from_rgb(60, 100, 200),
            CelestialBody::Sun => egui::Color32::from_rgb(255, 220, 50),
            CelestialBody::Ceres => egui::Color32::from_rgb(150, 140, 130),
            CelestialBody::Haumea => egui::Color32::from_rgb(180, 170, 160),
            CelestialBody::Makemake => egui::Color32::from_rgb(170, 150, 130),
            CelestialBody::Eris => egui::Color32::from_rgb(210, 200, 190),
            CelestialBody::Pluto => egui::Color32::from_rgb(190, 170, 150),
            CelestialBody::PlanetNine => egui::Color32::from_rgb(100, 140, 180),
            CelestialBody::Ganymede => egui::Color32::from_rgb(160, 150, 140),
            CelestialBody::Callisto => egui::Color32::from_rgb(130, 120, 110),
            CelestialBody::Io => egui::Color32::from_rgb(220, 180, 60),
            CelestialBody::Europa => egui::Color32::from_rgb(210, 200, 190),
            CelestialBody::Titan => egui::Color32::from_rgb(200, 160, 80),
            CelestialBody::Triton => egui::Color32::from_rgb(200, 180, 170),
            CelestialBody::Charon => egui::Color32::from_rgb(170, 160, 155),
            CelestialBody::Enceladus => egui::Color32::from_rgb(220, 220, 230),
            CelestialBody::Mimas => egui::Color32::from_rgb(190, 190, 195),
            CelestialBody::Iapetus => egui::Color32::from_rgb(150, 140, 130),
            CelestialBody::Phobos => egui::Color32::from_rgb(130, 120, 110),
            CelestialBody::Vesta => egui::Color32::from_rgb(160, 155, 150),
        }
    }
}
