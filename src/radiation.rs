const DIPOLE_TILT_DEG: f64 = 11.0;
const EARTH_RADIUS_KM: f64 = 6371.0;

pub(crate) fn radiation_level(
    altitude_km: f64,
    lat_deg: f64,
    kp: f64,
    planet_radius: f64,
) -> f64 {
    let mag_lat = (lat_deg - DIPOLE_TILT_DEG * (lat_deg / 90.0).signum() * 0.5)
        .to_radians();
    let cos_lat = mag_lat.cos();
    if cos_lat.abs() < 0.1 {
        return 0.0;
    }

    let r = (planet_radius + altitude_km) / planet_radius;
    let l_shell = r / (cos_lat * cos_lat);

    let kp_scale = 0.5 + 0.5 * (kp / 9.0);

    let inner_peak = 1.5;
    let inner_sigma = 0.4;
    let inner = (-(l_shell - inner_peak).powi(2) / (2.0 * inner_sigma * inner_sigma)).exp();

    let outer_peak = 4.5;
    let outer_sigma = 1.2;
    let outer = (-(l_shell - outer_peak).powi(2) / (2.0 * outer_sigma * outer_sigma)).exp();

    let raw = (inner * 0.8 + outer * 1.0) * kp_scale;
    raw.clamp(0.0, 1.0)
}

pub(crate) fn belt_profile(altitude_km: f64, kp: f64) -> f64 {
    let r = (EARTH_RADIUS_KM + altitude_km) / EARTH_RADIUS_KM;

    let kp_scale = 0.5 + 0.5 * (kp / 9.0);

    let inner_peak = 1.5;
    let inner_sigma = 0.3;
    let inner = (-(r - inner_peak).powi(2) / (2.0 * inner_sigma * inner_sigma)).exp();

    let outer_peak = 4.5;
    let outer_sigma = 1.0;
    let outer = (-(r - outer_peak).powi(2) / (2.0 * outer_sigma * outer_sigma)).exp();

    ((inner * 0.8 + outer * 1.0) * kp_scale).clamp(0.0, 1.0)
}

pub(crate) fn belt_color(altitude_km: f64) -> (u8, u8, u8) {
    let r = (EARTH_RADIUS_KM + altitude_km) / EARTH_RADIUS_KM;
    let transition = ((r - 2.5) / 2.0).clamp(0.0, 1.0);
    let red = (255.0 * (1.0 - transition * 0.6)) as u8;
    let green = (120.0 * (1.0 - transition * 0.5)) as u8;
    let blue = (50.0 + 180.0 * transition) as u8;
    (red, green, blue)
}

