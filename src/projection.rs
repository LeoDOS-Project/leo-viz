use std::f64::consts::PI;

pub(crate) trait Projection {
    fn project(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)>;
    #[allow(dead_code)]
    fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)>;
    fn x_range(&self) -> (f64, f64);
    fn y_range(&self) -> (f64, f64);
}

pub(crate) struct Equirectangular;
pub(crate) struct Mercator;
pub(crate) struct Mollweide;
pub(crate) struct Sinusoidal;
pub(crate) struct AzimuthalEquidistant;
pub(crate) struct Hammer;
pub(crate) struct HEALPix;
pub(crate) struct SpaceObliqueMercator;
pub(crate) struct TransverseMercator;

impl Projection for Equirectangular {
    fn project(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
        Some((lon_deg, lat_deg))
    }
    fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        if x >= -180.0 && x <= 180.0 && y >= -90.0 && y <= 90.0 {
            Some((y, x))
        } else {
            None
        }
    }
    fn x_range(&self) -> (f64, f64) { (-180.0, 180.0) }
    fn y_range(&self) -> (f64, f64) { (-90.0, 90.0) }
}

const MERCATOR_MAX_LAT: f64 = 85.0;

impl Projection for Mercator {
    fn project(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
        let clamped = lat_deg.clamp(-MERCATOR_MAX_LAT, MERCATOR_MAX_LAT);
        let lat_rad = clamped.to_radians();
        let y = (PI / 4.0 + lat_rad / 2.0).tan().ln().to_degrees();
        Some((lon_deg, y))
    }
    fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let lat_rad = 2.0 * (y.to_radians().exp()).atan() - PI / 2.0;
        let lat_deg = lat_rad.to_degrees();
        if lat_deg.abs() > MERCATOR_MAX_LAT || x.abs() > 180.0 {
            None
        } else {
            Some((lat_deg, x))
        }
    }
    fn x_range(&self) -> (f64, f64) { (-180.0, 180.0) }
    fn y_range(&self) -> (f64, f64) {
        let y_max = (PI / 4.0 + MERCATOR_MAX_LAT.to_radians() / 2.0).tan().ln().to_degrees();
        (-y_max, y_max)
    }
}

const MOLLWEIDE_SCALE: f64 = 90.0;

impl Projection for Mollweide {
    fn project(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
        let lat_rad = lat_deg.to_radians();
        let target = PI * lat_rad.sin();
        let mut theta = lat_rad;
        for _ in 0..20 {
            let f = 2.0 * theta + (2.0 * theta).sin() - target;
            let df = 2.0 + 2.0 * (2.0 * theta).cos();
            if df.abs() < 1e-15 { break; }
            let delta = f / df;
            theta -= delta;
            if delta.abs() < 1e-12 { break; }
        }
        let lon_rad = lon_deg.to_radians();
        let x = (2.0 * 2.0_f64.sqrt() / PI) * lon_rad * theta.cos();
        let y = 2.0_f64.sqrt() * theta.sin();
        Some((x * MOLLWEIDE_SCALE, y * MOLLWEIDE_SCALE))
    }
    fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let xn = x / MOLLWEIDE_SCALE;
        let yn = y / MOLLWEIDE_SCALE;
        let sin_theta = yn / 2.0_f64.sqrt();
        if sin_theta.abs() > 1.0 { return None; }
        let theta = sin_theta.asin();
        let cos_theta = theta.cos();
        if cos_theta.abs() < 1e-10 { return None; }
        let lon_rad = PI * xn / (2.0 * 2.0_f64.sqrt() * cos_theta);
        if lon_rad.abs() > PI { return None; }
        let sin_lat = (2.0 * theta + (2.0 * theta).sin()) / PI;
        if sin_lat.abs() > 1.0 { return None; }
        let lat_rad = sin_lat.asin();
        Some((lat_rad.to_degrees(), lon_rad.to_degrees()))
    }
    fn x_range(&self) -> (f64, f64) {
        let max_x = 2.0 * 2.0_f64.sqrt() / PI * PI * MOLLWEIDE_SCALE;
        (-max_x, max_x)
    }
    fn y_range(&self) -> (f64, f64) {
        let max_y = 2.0_f64.sqrt() * MOLLWEIDE_SCALE;
        (-max_y, max_y)
    }
}

const SINUSOIDAL_SCALE_X: f64 = 180.0;
const SINUSOIDAL_SCALE_Y: f64 = 90.0;

impl Projection for Sinusoidal {
    fn project(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
        let lat_rad = lat_deg.to_radians();
        let lon_rad = lon_deg.to_radians();
        let x = lon_rad * lat_rad.cos();
        let y = lat_rad;
        Some((x / PI * SINUSOIDAL_SCALE_X, y / (PI / 2.0) * SINUSOIDAL_SCALE_Y))
    }
    fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let lat_rad = y / SINUSOIDAL_SCALE_Y * (PI / 2.0);
        if lat_rad.abs() > PI / 2.0 { return None; }
        let cos_lat = lat_rad.cos();
        if cos_lat.abs() < 1e-10 { return None; }
        let lon_rad = x / SINUSOIDAL_SCALE_X * PI / cos_lat;
        if lon_rad.abs() > PI { return None; }
        Some((lat_rad.to_degrees(), lon_rad.to_degrees()))
    }
    fn x_range(&self) -> (f64, f64) { (-SINUSOIDAL_SCALE_X, SINUSOIDAL_SCALE_X) }
    fn y_range(&self) -> (f64, f64) { (-SINUSOIDAL_SCALE_Y, SINUSOIDAL_SCALE_Y) }
}

const AE_SCALE: f64 = 180.0;

impl Projection for AzimuthalEquidistant {
    fn project(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
        let lat_rad = lat_deg.to_radians();
        let lon_rad = lon_deg.to_radians();
        let c = (PI / 2.0) - lat_rad;
        let x = c * lon_rad.sin();
        let y = c * lon_rad.cos();
        Some((x / PI * AE_SCALE, y / PI * AE_SCALE))
    }
    fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let xn = x / AE_SCALE * PI;
        let yn = y / AE_SCALE * PI;
        let c = (xn * xn + yn * yn).sqrt();
        if c > PI { return None; }
        let lat_rad = (PI / 2.0) - c;
        let lon_rad = yn.atan2(xn);
        Some((lat_rad.to_degrees(), lon_rad.to_degrees()))
    }
    fn x_range(&self) -> (f64, f64) { (-AE_SCALE, AE_SCALE) }
    fn y_range(&self) -> (f64, f64) { (-AE_SCALE, AE_SCALE) }
}

const HAMMER_SCALE: f64 = 90.0;

impl Projection for Hammer {
    fn project(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
        let lat_rad = lat_deg.to_radians();
        let lon_rad = lon_deg.to_radians();
        let denom = (1.0 + lat_rad.cos() * (lon_rad / 2.0).cos()).sqrt();
        let x = 2.0 * 2.0_f64.sqrt() * lat_rad.cos() * (lon_rad / 2.0).sin() / denom;
        let y = 2.0_f64.sqrt() * lat_rad.sin() / denom;
        Some((x * HAMMER_SCALE, y * HAMMER_SCALE))
    }
    fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let xn = x / HAMMER_SCALE;
        let yn = y / HAMMER_SCALE;
        let z2 = 1.0 - (xn / 4.0).powi(2) - (yn / 2.0).powi(2);
        if z2 < 0.0 { return None; }
        let z = z2.sqrt();
        let lon_rad = 2.0 * (z * xn).atan2(2.0 * (2.0 * z2 - 1.0));
        let sin_lat = z * yn;
        if sin_lat.abs() > 1.0 { return None; }
        let lat_rad = sin_lat.asin();
        if lon_rad.abs() > PI { return None; }
        Some((lat_rad.to_degrees(), lon_rad.to_degrees()))
    }
    fn x_range(&self) -> (f64, f64) {
        let max_x = 2.0 * 2.0_f64.sqrt() * HAMMER_SCALE;
        (-max_x, max_x)
    }
    fn y_range(&self) -> (f64, f64) {
        let max_y = 2.0_f64.sqrt() * HAMMER_SCALE;
        (-max_y, max_y)
    }
}

impl Projection for HEALPix {
    fn project(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
        let phi = lat_deg.to_radians();
        let lam = lon_deg.to_radians();
        let sin_phi = phi.sin();
        if sin_phi.abs() <= 2.0 / 3.0 {
            Some((lam.to_degrees(), (3.0 * PI / 8.0 * sin_phi).to_degrees()))
        } else {
            let sign = sin_phi.signum();
            let sigma = (3.0 * (1.0 - sin_phi.abs())).sqrt();
            let step = PI / 2.0;
            let lam_c = ((lam - PI / 4.0) / step).round() * step + PI / 4.0;
            let mut dlam = lam - lam_c;
            if dlam > PI { dlam -= 2.0 * PI; }
            if dlam < -PI { dlam += 2.0 * PI; }
            let x = lam_c + dlam * sigma;
            let y = sign * PI / 4.0 * (2.0 - sigma);
            Some((x.to_degrees(), y.to_degrees()))
        }
    }
    fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let xr = x.to_radians();
        let yr = y.to_radians();
        if yr.abs() > PI / 2.0 { return None; }
        if yr.abs() <= PI / 4.0 {
            let sin_phi = 8.0 * yr / (3.0 * PI);
            if sin_phi.abs() > 1.0 { return None; }
            if xr.abs() > PI { return None; }
            Some((sin_phi.asin().to_degrees(), xr.to_degrees()))
        } else {
            let sign = yr.signum();
            let sigma = 2.0 - 4.0 * yr.abs() / PI;
            if sigma < 1e-10 { return Some((sign * 90.0, 0.0)); }
            let step = PI / 2.0;
            let lam_c = ((xr - PI / 4.0) / step).round() * step + PI / 4.0;
            let lam = lam_c + (xr - lam_c) / sigma;
            if (lam - lam_c).abs() > step / 2.0 + 0.001 { return None; }
            if lam.abs() > PI + 0.001 { return None; }
            let sin_phi = 1.0 - sigma * sigma / 3.0;
            if sin_phi > 1.0 { return None; }
            Some((sign * sin_phi.asin().to_degrees(), lam.to_degrees().clamp(-180.0, 180.0)))
        }
    }
    fn x_range(&self) -> (f64, f64) { (-180.0, 180.0) }
    fn y_range(&self) -> (f64, f64) { (-90.0, 90.0) }
}

const SOM_INCL_RAD: f64 = 53.0 * PI / 180.0;
const SOM_SCALE: f64 = 180.0 / PI;
const SOM_SWATH_DEG: f64 = 20.0;

impl Projection for SpaceObliqueMercator {
    fn project(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
        let phi = lat_deg.to_radians();
        let lam = lon_deg.to_radians();
        let (ci, si) = (SOM_INCL_RAD.cos(), SOM_INCL_RAD.sin());
        let (x0, y0, z0) = (phi.cos() * lam.cos(), phi.cos() * lam.sin(), phi.sin());
        let y1 = y0 * ci + z0 * si;
        let z1 = -y0 * si + z0 * ci;
        let phi1 = z1.clamp(-1.0, 1.0).asin();
        if phi1.abs() > SOM_SWATH_DEG.to_radians() { return None; }
        let lam1 = y1.atan2(x0);
        let y_merc = (PI / 4.0 + phi1 / 2.0).tan().ln();
        Some((lam1 * SOM_SCALE, y_merc * SOM_SCALE))
    }
    fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let lam1 = x / SOM_SCALE;
        let y_merc = y / SOM_SCALE;
        let phi1 = 2.0 * y_merc.exp().atan() - PI / 2.0;
        if phi1.abs() > SOM_SWATH_DEG.to_radians() { return None; }
        let (ci, si) = (SOM_INCL_RAD.cos(), SOM_INCL_RAD.sin());
        let (x1, y1, z1) = (phi1.cos() * lam1.cos(), phi1.cos() * lam1.sin(), phi1.sin());
        let y0 = y1 * ci - z1 * si;
        let z0 = y1 * si + z1 * ci;
        let phi = z0.clamp(-1.0, 1.0).asin();
        let lam = y0.atan2(x1);
        Some((phi.to_degrees(), lam.to_degrees()))
    }
    fn x_range(&self) -> (f64, f64) { (-180.0, 180.0) }
    fn y_range(&self) -> (f64, f64) {
        let y_max = (PI / 4.0 + SOM_SWATH_DEG.to_radians() / 2.0).tan().ln() * SOM_SCALE;
        (-y_max, y_max)
    }
}

const UTM_SCALE: f64 = 180.0 / PI;

impl Projection for TransverseMercator {
    fn project(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
        let zone = ((lon_deg + 180.0) / 6.0).floor().clamp(0.0, 59.0) as i32;
        let cm = zone as f64 * 6.0 - 177.0;
        let dlon = (lon_deg - cm).to_radians();
        let lat_r = lat_deg.to_radians();
        let b = lat_r.cos() * dlon.sin();
        if b.abs() >= 0.9999 { return None; }
        let x_tm = b.atanh();
        let y_tm = lat_r.sin().atan2(lat_r.cos() * dlon.cos());
        Some((cm + x_tm * UTM_SCALE, y_tm * UTM_SCALE))
    }
    fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let zone = ((x + 180.0) / 6.0).floor().clamp(0.0, 59.0);
        let cm = zone * 6.0 - 177.0;
        let xr = (x - cm) / UTM_SCALE;
        let yr = y / UTM_SCALE;
        let sin_lat = yr.sin() / xr.cosh();
        if sin_lat.abs() > 1.0 { return None; }
        let lat = sin_lat.asin();
        let dlon = xr.sinh().atan2(yr.cos());
        if dlon.abs() > (3.0_f64).to_radians() + 0.001 { return None; }
        Some((lat.to_degrees(), cm + dlon.to_degrees()))
    }
    fn x_range(&self) -> (f64, f64) { (-180.0, 180.0) }
    fn y_range(&self) -> (f64, f64) { (-90.0, 90.0) }
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum ProjectionKind {
    Orthographic,
    Equirectangular,
    Mercator,
    Mollweide,
    Sinusoidal,
    AzimuthalEquidistant,
    Hammer,
    HEALPix,
    SpaceObliqueMercator,
    TransverseMercator,
}

impl ProjectionKind {
    pub(crate) fn instance(&self) -> &dyn Projection {
        match self {
            ProjectionKind::Orthographic => &Equirectangular,
            ProjectionKind::Equirectangular => &Equirectangular,
            ProjectionKind::Mercator => &Mercator,
            ProjectionKind::Mollweide => &Mollweide,
            ProjectionKind::Sinusoidal => &Sinusoidal,
            ProjectionKind::AzimuthalEquidistant => &AzimuthalEquidistant,
            ProjectionKind::Hammer => &Hammer,
            ProjectionKind::HEALPix => &HEALPix,
            ProjectionKind::SpaceObliqueMercator => &SpaceObliqueMercator,
            ProjectionKind::TransverseMercator => &TransverseMercator,
        }
    }

    pub(crate) fn label(&self) -> &'static str {
        match self {
            ProjectionKind::Orthographic => "Orthographic",
            ProjectionKind::Equirectangular => "Equirectangular",
            ProjectionKind::Mercator => "Mercator",
            ProjectionKind::Mollweide => "Mollweide",
            ProjectionKind::Sinusoidal => "Sinusoidal",
            ProjectionKind::AzimuthalEquidistant => "Azimuthal Equidistant",
            ProjectionKind::Hammer => "Hammer",
            ProjectionKind::HEALPix => "HEALPix",
            ProjectionKind::SpaceObliqueMercator => "SOM (53°)",
            ProjectionKind::TransverseMercator => "UTM",
        }
    }
}
