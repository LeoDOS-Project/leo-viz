const EARTH_RADIUS_KM: f64 = 6371.0;

pub(crate) fn belt_profile(altitude_km: f64, kp: f64) -> f64 {
    let r = (EARTH_RADIUS_KM + altitude_km) / EARTH_RADIUS_KM;
    belt_profile_r(r, kp)
}

pub(crate) fn belt_profile_r(r: f64, kp: f64) -> f64 {
    let kp_scale = 0.5 + 0.5 * (kp / 9.0);

    let inner_peak = 1.5;
    let inner_sigma = 0.3;
    let inner = (-(r - inner_peak).powi(2) / (2.0 * inner_sigma * inner_sigma)).exp();

    let outer_peak = 4.5;
    let outer_sigma = 1.0;
    let outer = (-(r - outer_peak).powi(2) / (2.0 * outer_sigma * outer_sigma)).exp();

    ((inner * 0.8 + outer * 1.0) * kp_scale).clamp(0.0, 1.0)
}
