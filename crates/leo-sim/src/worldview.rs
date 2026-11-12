//! WorldView-3 nominal geometry and hypothetical fleets using that orbit.
//! The effective width is an access-corridor approximation, not the camera swath.
use crate::orbit::{Propagation, WalkerShell};

pub const ALTITUDE_KM: f64 = 617.0;
pub const INCLINATION_DEG: f64 = 97.9;
pub const ACCESS_WIDTH_KM: f64 = 450.0;
pub const IMAGING_SWATH_KM: f64 = 13.1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fleet {
    WorldView3,
    WorldView30,
    WorldView60,
}

impl Fleet {
    pub const ALL: [Self; 3] = [Self::WorldView3, Self::WorldView30, Self::WorldView60];

    pub fn name(self) -> &'static str {
        match self {
            Self::WorldView3 => "WorldView-3",
            Self::WorldView30 => "WorldView-30",
            Self::WorldView60 => "WorldView-60",
        }
    }

    pub fn shell(self) -> WalkerShell {
        match self {
            Self::WorldView3 => shell(1, 1, 0, false),
            Self::WorldView30 => shell(5, 6, 3, false),
            Self::WorldView60 => shell(20, 3, 14, false),
        }
    }

    pub fn baseline_note(self) -> &'static str {
        match self {
            Self::WorldView3 => "One satellite; nominal WorldView-3 geometry, not a live ephemeris.",
            Self::WorldView30 => "Hypothetical fleet. Sampled maximum gap ≈5.0 h across ±84° over 7 days; not a global bound.",
            Self::WorldView60 => "Hypothetical fleet. Sampled maximum gap ≈2.4 h across ±84° over 7 days; not a global bound.",
        }
    }
}

/// Nominal circular orbit with secular J2 drift. RAAN and initial anomaly are
/// design coordinates, not a live ephemeris or a specified local solar time.
pub fn shell(
    planes: usize,
    satellites_per_plane: usize,
    phasing: usize,
    star: bool,
) -> WalkerShell {
    WalkerShell {
        planes,
        satellites_per_plane,
        altitude_km: ALTITUDE_KM,
        inclination_deg: INCLINATION_DEG,
        phasing,
        star,
        propagation: Propagation::J2,
        ..WalkerShell::default()
    }
}
