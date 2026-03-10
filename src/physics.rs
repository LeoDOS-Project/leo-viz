use nalgebra::Vector3;

const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;
const DEFAULT_SOLAR_IRRADIANCE: f64 = 1360.0;
const DEFAULT_BODY_EMISSIVITY: f64 = 0.6;
const DEFAULT_BODY_REFLECTANCE: f64 = 0.3;
const DEFAULT_BODY_SURFACE_TEMP: f64 = 288.0;

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum PowerDeviceType {
    SolarPanel,
    Rtg,
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum PhysicsColorMode {
    Normal,
    Battery,
    Temperature,
}

#[derive(Clone)]
pub(crate) struct PhysicsConfig {
    pub enabled: bool,
    pub power_enabled: bool,
    pub thermal_enabled: bool,
    pub radiation_enabled: bool,

    pub max_battery_ws: f64,
    pub charging_rate_w: f64,
    pub power_device_type: PowerDeviceType,
    pub idle_power_w: f64,

    pub mass_kg: f64,
    pub thermal_capacity: f64,
    pub sun_absorptance: f64,
    pub ir_absorptance: f64,
    pub sun_facing_area: f64,
    pub body_facing_area: f64,
    pub emissive_area: f64,
    pub heat_ratio: f64,

    pub restart_rate: f64,
    pub failure_rate: f64,

    pub color_mode: PhysicsColorMode,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            power_enabled: true,
            thermal_enabled: true,
            radiation_enabled: true,
            max_battery_ws: 100_000.0,
            charging_rate_w: 50.0,
            power_device_type: PowerDeviceType::SolarPanel,
            idle_power_w: 10.0,
            mass_kg: 100.0,
            thermal_capacity: 900.0,
            sun_absorptance: 0.3,
            ir_absorptance: 0.5,
            sun_facing_area: 1.0,
            body_facing_area: 1.0,
            emissive_area: 4.0,
            heat_ratio: 0.5,
            restart_rate: 1e-9,
            failure_rate: 1e-11,
            color_mode: PhysicsColorMode::Battery,
        }
    }
}

#[derive(Clone)]
pub(crate) struct SatellitePhysics {
    pub battery_ws: f64,
    pub temperature_k: f64,
    pub is_dead: bool,
}

impl SatellitePhysics {
    pub fn new(config: &PhysicsConfig) -> Self {
        Self {
            battery_ws: config.max_battery_ws,
            temperature_k: 280.0,
            is_dead: false,
        }
    }

    pub fn state_of_charge(&self, config: &PhysicsConfig) -> f64 {
        if config.max_battery_ws > 0.0 {
            self.battery_ws / config.max_battery_ws
        } else {
            0.0
        }
    }
}

pub(crate) fn is_eclipsed(
    sat_pos: &Vector3<f64>,
    sun_dir: &Vector3<f64>,
    planet_radius: f64,
) -> bool {
    let proj = sat_pos.dot(sun_dir);
    if proj >= 0.0 {
        return false;
    }
    let perp_sq = sat_pos.dot(sat_pos) - proj * proj;
    perp_sq < planet_radius * planet_radius
}

pub(crate) fn update_power(
    state: &mut SatellitePhysics,
    config: &PhysicsConfig,
    dt: f64,
    eclipsed: bool,
) {
    let can_charge = config.power_device_type == PowerDeviceType::Rtg || !eclipsed;
    if can_charge {
        state.battery_ws = (state.battery_ws + config.charging_rate_w * dt)
            .min(config.max_battery_ws);
    }
    state.battery_ws = (state.battery_ws - config.idle_power_w * dt).max(0.0);
}

pub(crate) fn update_thermal(
    state: &mut SatellitePhysics,
    config: &PhysicsConfig,
    dt: f64,
    eclipsed: bool,
    altitude_km: f64,
    planet_radius: f64,
) {
    let eclipse_f = if eclipsed { 0.0 } else { 1.0 };

    let q_solar = config.sun_absorptance
        * config.sun_facing_area
        * DEFAULT_SOLAR_IRRADIANCE
        * eclipse_f;

    let q_albedo = config.sun_absorptance
        * config.body_facing_area
        * DEFAULT_BODY_REFLECTANCE
        * DEFAULT_SOLAR_IRRADIANCE
        * 0.5
        * eclipse_f;

    let alt_ratio = (planet_radius + altitude_km) / planet_radius;
    let view_factor = 1.0 / (alt_ratio * alt_ratio);

    let q_body_ir = config.ir_absorptance
        * DEFAULT_BODY_EMISSIVITY
        * config.body_facing_area
        * STEFAN_BOLTZMANN
        * DEFAULT_BODY_SURFACE_TEMP.powi(4)
        * view_factor;

    let q_activity = config.heat_ratio * config.idle_power_w;

    let thermal_mass = config.mass_kg * config.thermal_capacity;
    let max_step = 10.0;
    let mut remaining = dt;
    while remaining > 0.0 {
        let step = remaining.min(max_step);
        let q_em = config.ir_absorptance
            * config.emissive_area
            * STEFAN_BOLTZMANN
            * state.temperature_k.powi(4);
        let q_net = q_solar + q_albedo + q_body_ir - q_em + q_activity;
        state.temperature_k = (state.temperature_k + q_net / thermal_mass * step).max(2.7);
        remaining -= step;
    }
}

fn pseudo_random(seed: u64) -> f64 {
    let x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let x = ((x >> 33) ^ x).wrapping_mul(0xff51afd7ed558ccd);
    let x = ((x >> 33) ^ x).wrapping_mul(0xc4ceb9fe1a85ec53);
    let x = (x >> 33) ^ x;
    (x & 0x000F_FFFF_FFFF_FFFF) as f64 / (0x0010_0000_0000_0000u64 as f64)
}

pub(crate) fn update_radiation(
    state: &mut SatellitePhysics,
    config: &PhysicsConfig,
    dt: f64,
    seed: u64,
) {
    if state.is_dead {
        return;
    }
    let p_failure = 1.0 - (-config.failure_rate * dt).exp();
    if pseudo_random(seed) < p_failure {
        state.is_dead = true;
        return;
    }
    let p_restart = 1.0 - (-config.restart_rate * dt).exp();
    if pseudo_random(seed.wrapping_add(12345)) < p_restart {
        state.battery_ws *= 0.9;
    }
}

pub(crate) fn update_satellite(
    state: &mut SatellitePhysics,
    config: &PhysicsConfig,
    dt: f64,
    sat_pos: &Vector3<f64>,
    sun_dir: &Vector3<f64>,
    planet_radius: f64,
    altitude_km: f64,
    seed: u64,
) {
    if state.is_dead {
        return;
    }
    let eclipsed = is_eclipsed(sat_pos, sun_dir, planet_radius);
    if config.power_enabled {
        update_power(state, config, dt, eclipsed);
    }
    if config.thermal_enabled {
        update_thermal(state, config, dt, eclipsed, altitude_km, planet_radius);
    }
    if config.radiation_enabled {
        update_radiation(state, config, dt, seed);
    }
}

pub(crate) fn battery_color(soc: f64) -> eframe::egui::Color32 {
    let soc = soc.clamp(0.0, 1.0);
    let r = ((1.0 - soc) * 2.0).min(1.0);
    let g = (soc * 2.0).min(1.0);
    eframe::egui::Color32::from_rgb(
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        0,
    )
}

pub(crate) fn temperature_color(temp_k: f64) -> eframe::egui::Color32 {
    let cold = 150.0;
    let hot = 400.0;
    let t = ((temp_k - cold) / (hot - cold)).clamp(0.0, 1.0);
    if t < 0.5 {
        let f = t * 2.0;
        eframe::egui::Color32::from_rgb(
            (f * 255.0) as u8,
            (f * 255.0) as u8,
            255,
        )
    } else {
        let f = (t - 0.5) * 2.0;
        eframe::egui::Color32::from_rgb(
            255,
            ((1.0 - f) * 255.0) as u8,
            ((1.0 - f) * 255.0) as u8,
        )
    }
}

pub(crate) fn dead_color() -> eframe::egui::Color32 {
    eframe::egui::Color32::from_rgb(100, 100, 100)
}
