#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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
            _ => None,
        }
    }
}

impl CelestialBody {
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
        }
    }

    pub fn available_skins(&self) -> &'static [Skin] {
        match self {
            CelestialBody::Earth => &[Skin::Default, Skin::Abstract, Skin::HellOnEarth],
            CelestialBody::Mars => &[Skin::Default, Skin::Terraformed, Skin::Civilized],
            _ => &[Skin::Default],
        }
    }

    pub const ALL: [CelestialBody; 14] = [
        CelestialBody::Earth,
        CelestialBody::Moon,
        CelestialBody::Mars,
        CelestialBody::Mercury,
        CelestialBody::Venus,
        CelestialBody::Jupiter,
        CelestialBody::Saturn,
        CelestialBody::Uranus,
        CelestialBody::Neptune,
        CelestialBody::Sun,
        CelestialBody::Ceres,
        CelestialBody::Haumea,
        CelestialBody::Makemake,
        CelestialBody::Eris,
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
}
