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
        if lat_deg.abs() > MERCATOR_MAX_LAT { return None; }
        let lat_rad = lat_deg.to_radians();
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

const MOLLWEIDE_SCALE_X: f64 = 180.0;
const MOLLWEIDE_SCALE_Y: f64 = 90.0;

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
        Some((x * MOLLWEIDE_SCALE_X, y * MOLLWEIDE_SCALE_Y))
    }
    fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let xn = x / MOLLWEIDE_SCALE_X;
        let yn = y / MOLLWEIDE_SCALE_Y;
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
        let max_x = 2.0 * 2.0_f64.sqrt() / PI * PI * MOLLWEIDE_SCALE_X;
        (-max_x, max_x)
    }
    fn y_range(&self) -> (f64, f64) {
        let max_y = 2.0_f64.sqrt() * MOLLWEIDE_SCALE_Y;
        (-max_y, max_y)
    }
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum ProjectionKind {
    Equirectangular,
    Mercator,
    Mollweide,
}

impl ProjectionKind {
    pub(crate) fn instance(&self) -> &dyn Projection {
        match self {
            ProjectionKind::Equirectangular => &Equirectangular,
            ProjectionKind::Mercator => &Mercator,
            ProjectionKind::Mollweide => &Mollweide,
        }
    }
}
