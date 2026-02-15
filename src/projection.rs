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
pub(crate) struct Cassini;
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
        let y = -c * lon_rad.cos();
        Some((x / PI * AE_SCALE, y / PI * AE_SCALE))
    }
    fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let xn = x / AE_SCALE * PI;
        let yn = y / AE_SCALE * PI;
        let c = (xn * xn + yn * yn).sqrt();
        if c > PI { return None; }
        let lat_rad = (PI / 2.0) - c;
        let lon_rad = xn.atan2(-yn);
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

const CASSINI_SCALE: f64 = 180.0 / PI;

impl Projection for Cassini {
    fn project(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
        let phi = lat_deg.to_radians();
        let lam = lon_deg.to_radians();
        let x = (phi.cos() * lam.sin()).asin();
        let y = phi.sin().atan2(phi.cos() * lam.cos());
        Some((x * CASSINI_SCALE, y * CASSINI_SCALE))
    }
    fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let xr = x / CASSINI_SCALE;
        let yr = y / CASSINI_SCALE;
        if xr.abs() > PI / 2.0 { return None; }
        let lat = (yr.sin() * xr.cos()).asin();
        let lon = xr.tan().atan2(yr.cos());
        Some((lat.to_degrees(), lon.to_degrees()))
    }
    fn x_range(&self) -> (f64, f64) { (-90.0, 90.0) }
    fn y_range(&self) -> (f64, f64) { (-180.0, 180.0) }
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

const LAEA_SCALE: f64 = 90.0;

pub(crate) struct LambertAzimuthalEqualArea;

impl Projection for LambertAzimuthalEqualArea {
    fn project(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
        let phi = lat_deg.to_radians();
        let lam = lon_deg.to_radians();
        let denom = 1.0 + phi.cos() * lam.cos();
        if denom < 1e-10 { return None; }
        let k = (2.0 / denom).sqrt();
        let x = k * phi.cos() * lam.sin();
        let y = k * phi.sin();
        Some((x * LAEA_SCALE, y * LAEA_SCALE))
    }
    fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let xn = x / LAEA_SCALE;
        let yn = y / LAEA_SCALE;
        let rho = (xn * xn + yn * yn).sqrt();
        if rho > 2.0 { return None; }
        if rho < 1e-10 { return Some((0.0, 0.0)); }
        let c = 2.0 * (rho / 2.0).asin();
        let lat = (yn * c.sin() / rho).asin();
        let lon = (xn * c.sin()).atan2(rho * c.cos());
        Some((lat.to_degrees(), lon.to_degrees()))
    }
    fn x_range(&self) -> (f64, f64) {
        (-2.0 * LAEA_SCALE, 2.0 * LAEA_SCALE)
    }
    fn y_range(&self) -> (f64, f64) {
        (-2.0 * LAEA_SCALE, 2.0 * LAEA_SCALE)
    }
}

pub(crate) struct PeirceQuincuncial;

const PEIRCE_SCALE: f64 = 90.0;

fn elliptic_f(phi: f64, m: f64) -> f64 {
    if m.abs() < 1e-15 { return phi; }
    if (m - 1.0).abs() < 1e-15 {
        return (phi / 2.0 + PI / 4.0).tan().ln();
    }
    let mut a = 1.0_f64;
    let mut b = (1.0 - m).sqrt();
    let mut c = m.sqrt();
    let mut phi = phi;
    let mut n = 0u32;
    while c.abs() > 1e-14 && n < 30 {
        let rem = phi % PI;
        if rem.abs() > 1e-14 {
            let mut d_phi = (b * phi.tan() / a).atan();
            if d_phi < 0.0 { d_phi += PI; }
            phi += d_phi + (phi / PI).floor() as f64 * PI;
        } else {
            phi += phi;
        }
        let temp = (a + b) / 2.0;
        b = (a * b).sqrt();
        a = temp;
        c = a - b;
        n += 1;
    }
    phi / (2.0_f64.powi(n as i32) * a)
}

fn elliptic_fi(phi: f64, psi: f64, m: f64) -> (f64, f64) {
    let r = phi.abs();
    let i_abs = psi.abs();
    let sinh_psi = i_abs.sinh();
    if r > 1e-14 {
        let csc_phi = 1.0 / r.sin();
        let cot_phi2 = 1.0 / (r.tan() * r.tan());
        let b = -(cot_phi2
            + m * (sinh_psi * sinh_psi * csc_phi * csc_phi)
            - 1.0
            + m);
        let c = (m - 1.0) * cot_phi2;
        let disc = (b * b - 4.0 * c).max(0.0);
        let cot_lambda2 = (-b + disc.sqrt()) / 2.0;
        (
            elliptic_f((1.0 / cot_lambda2.sqrt()).atan(), m)
                * phi.signum(),
            elliptic_f(
                ((cot_lambda2 / cot_phi2 - 1.0).max(0.0) / m)
                    .sqrt()
                    .atan(),
                1.0 - m,
            ) * psi.signum(),
        )
    } else {
        (0.0, elliptic_f(sinh_psi.atan(), 1.0 - m) * psi.signum())
    }
}

#[allow(dead_code)]
fn elliptic_j(u: f64, m: f64) -> (f64, f64, f64) {
    if m < 1e-15 {
        return (u.sin(), u.cos(), 1.0);
    }
    if m >= 1.0 - 1e-15 {
        let t = u.tanh();
        let s = 1.0 / u.cosh();
        return (t, s, s);
    }
    let mut a = [0.0_f64; 9];
    let mut c = [0.0_f64; 9];
    a[0] = 1.0;
    c[0] = m.sqrt();
    let mut b = (1.0 - m).sqrt();
    let mut i = 0;
    let mut twon = 1.0_f64;
    while (c[i] / a[i]).abs() > 1e-14 && i < 8 {
        let ai = a[i];
        i += 1;
        c[i] = (ai - b) / 2.0;
        a[i] = (ai + b) / 2.0;
        b = (ai * b).sqrt();
        twon *= 2.0;
    }
    let mut phi = twon * a[i] * u;
    let mut b_prev;
    loop {
        b_prev = phi;
        let t = c[i] * phi.sin() / a[i];
        phi = (t.clamp(-1.0, 1.0).asin() + phi) / 2.0;
        i -= 1;
        if i == 0 { break; }
    }
    let cn = phi.cos();
    let diff_cos = (phi - b_prev).cos();
    let dn = if diff_cos.abs() > 1e-30 { cn / diff_cos } else { 1.0 };
    (phi.sin(), cn, dn)
}

#[allow(dead_code)]
fn elliptic_ji(
    u: f64, v: f64, m: f64,
) -> ([f64; 2], [f64; 2], [f64; 2]) {
    if u.abs() < 1e-15 {
        let (sn, cn, dn) = elliptic_j(v, 1.0 - m);
        return (
            [0.0, sn / cn],
            [1.0 / cn, 0.0],
            [dn / cn, 0.0],
        );
    }
    let (sn_a, cn_a, dn_a) = elliptic_j(u, m);
    if v.abs() < 1e-15 {
        return ([sn_a, 0.0], [cn_a, 0.0], [dn_a, 0.0]);
    }
    let (sn_b, cn_b, dn_b) = elliptic_j(v, 1.0 - m);
    let d = cn_b * cn_b + m * sn_a * sn_a * sn_b * sn_b;
    (
        [sn_a * dn_b / d, cn_a * dn_a * sn_b * cn_b / d],
        [cn_a * cn_b / d, -sn_a * dn_a * sn_b * dn_b / d],
        [dn_a * cn_b * dn_b / d, -m * sn_a * cn_a * sn_b / d],
    )
}

fn complex_atan(x: f64, y: f64) -> (f64, f64) {
    let x2 = x * x;
    let y_1 = y + 1.0;
    let t = 1.0 - x2 - y * y;
    (
        0.5 * ((if x >= 0.0 { PI / 2.0 } else { -PI / 2.0 })
            - t.atan2(2.0 * x)),
        -0.25 * (t * t + 4.0 * x2).ln()
            + 0.5 * (y_1 * y_1 + x2).ln(),
    )
}

pub(crate) struct PeirceConst {
    pub(crate) m: f64,
    pub(crate) k_: f64,
    pub(crate) big_k: f64,
    pub(crate) dx: f64,
    scale: f64,
    pub(crate) inv_scale: f64,
    extent: f64,
}

pub(crate) fn peirce_const() -> &'static PeirceConst {
    use std::sync::LazyLock;
    static C: LazyLock<PeirceConst> = LazyLock::new(|| {
        let sqrt2 = 2.0_f64.sqrt();
        let k_ = (sqrt2 - 1.0) / (sqrt2 + 1.0);
        let k = (1.0 - k_ * k_).sqrt();
        let m = k * k;
        let big_k = elliptic_f(PI / 2.0, m);
        let sqrt1_2 = (0.5_f64).sqrt();
        let g_pos = guyou_fwd(PI / 2.0, 0.0, k_, m, big_k);
        let g_neg = guyou_fwd(-PI / 2.0, 0.0, k_, m, big_k);
        let dx = g_pos.0 - g_neg.0;
        let scale = PEIRCE_SCALE / (big_k * sqrt1_2);
        PeirceConst {
            m, k_, big_k, dx,
            scale,
            inv_scale: 1.0 / scale,
            extent: PEIRCE_SCALE / sqrt1_2,
        }
    });
    &C
}

fn guyou_fwd(
    lambda: f64, phi: f64, k_: f64, m: f64, big_k: f64,
) -> (f64, f64) {
    let psi = (PI / 4.0 + phi.abs() / 2.0).tan().ln();
    let r = (-psi).exp() / k_.sqrt();
    let at = complex_atan(r * (-lambda).cos(), r * (-lambda).sin());
    let t = elliptic_fi(at.0, at.1, m);
    (
        -t.1,
        (if phi >= 0.0 { 1.0 } else { -1.0 }) * (0.5 * big_k - t.0),
    )
}

#[allow(dead_code)]
fn guyou_inv(
    x: f64, y: f64, k_: f64, m: f64, big_k: f64,
) -> (f64, f64) {
    let j = elliptic_ji(0.5 * big_k - y, -x, m);
    let sn = j.0;
    let cn = j.1;
    let d = cn[0] * cn[0] + cn[1] * cn[1];
    if d < 1e-30 { return (0.0, 0.0); }
    let tn_re = (sn[0] * cn[0] + sn[1] * cn[1]) / d;
    let tn_im = (sn[1] * cn[0] - sn[0] * cn[1]) / d;
    let lambda = -tn_im.atan2(tn_re);
    let log_arg = k_ * (tn_re * tn_re + tn_im * tn_im);
    let phi = 2.0 * (-0.5 * log_arg.ln()).exp().atan() - PI / 2.0;
    (lambda, phi)
}

impl Projection for PeirceQuincuncial {
    fn project(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
        let phi = lat_deg.to_radians();
        let lambda = lon_deg.to_radians();
        let c = peirce_const();
        let sqrt1_2 = (0.5_f64).sqrt();

        let front = lambda.abs() < PI / 2.0;
        let lam = if front {
            lambda
        } else if lambda > 0.0 {
            lambda - PI
        } else {
            lambda + PI
        };

        let p = guyou_fwd(lam, phi, c.k_, c.m, c.big_k);
        let (px, py) = (p.0, p.1);

        let (rx, ry) = if front {
            ((px - py) * sqrt1_2, (px + py) * sqrt1_2)
        } else {
            let d = c.dx * sqrt1_2;
            let x = (px - py) * sqrt1_2;
            let y = (px + py) * sqrt1_2;
            let s = if (x > 0.0) ^ (y > 0.0) { -1.0 } else { 1.0 };
            (s * x - y.signum() * d, s * y - x.signum() * d)
        };

        let (fx, fy) = (rx * c.scale, ry * c.scale);
        if fx.is_finite() && fy.is_finite() {
            Some((fx, fy))
        } else {
            None
        }
    }
    fn inverse(&self, x0: f64, y0: f64) -> Option<(f64, f64)> {
        let c = peirce_const();
        let sqrt1_2 = (0.5_f64).sqrt();

        let x0 = x0 * c.inv_scale;
        let y0 = y0 * c.inv_scale;

        let gx = (x0 + y0) * sqrt1_2;
        let gy = (y0 - x0) * sqrt1_2;
        let half_dx = 0.5 * c.dx;
        let front = gx.abs() < half_dx + 0.001
            && gy.abs() < half_dx + 0.001;

        if front {
            let (lam, phi) =
                guyou_inv(gx, gy, c.k_, c.m, c.big_k);
            let lat = phi.to_degrees();
            let lon = lam.to_degrees();
            if lat.abs() <= 90.0 && lon.abs() <= 180.0 {
                Some((lat, lon))
            } else {
                None
            }
        } else {
            let d = c.dx * sqrt1_2;
            let s = if (x0 > 0.0) ^ (y0 > 0.0) { -1.0 } else { 1.0 };
            let x1 = -s * x0 + (if y0 > 0.0 { 1.0 } else { -1.0 }) * d;
            let y1 = -s * y0 + (if x0 > 0.0 { 1.0 } else { -1.0 }) * d;
            let gx2 = (-x1 - y1) * sqrt1_2;
            let gy2 = (x1 - y1) * sqrt1_2;

            let (lam, phi) =
                guyou_inv(gx2, gy2, c.k_, c.m, c.big_k);
            let lon = lam.to_degrees()
                + if gx2 > 0.0 { 180.0 } else { -180.0 };
            let lat = phi.to_degrees();
            if lat.abs() <= 90.0 && lon.abs() <= 180.0 {
                Some((lat, lon))
            } else {
                None
            }
        }
    }
    fn x_range(&self) -> (f64, f64) {
        let e = peirce_const().extent;
        (-e, e)
    }
    fn y_range(&self) -> (f64, f64) {
        self.x_range()
    }
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
    Cassini,
    TransverseMercator,
    LambertAzimuthalEqualArea,
    PeirceQuincuncial,
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
            ProjectionKind::Cassini => &Cassini,
            ProjectionKind::TransverseMercator => &TransverseMercator,
            ProjectionKind::LambertAzimuthalEqualArea => &LambertAzimuthalEqualArea,
            ProjectionKind::PeirceQuincuncial => &PeirceQuincuncial,
        }
    }

    pub(crate) fn shader_id(&self) -> i32 {
        match self {
            ProjectionKind::Orthographic | ProjectionKind::Equirectangular => 0,
            ProjectionKind::Mercator => 1,
            ProjectionKind::Mollweide => 2,
            ProjectionKind::Sinusoidal => 3,
            ProjectionKind::AzimuthalEquidistant => 4,
            ProjectionKind::Hammer => 5,
            ProjectionKind::HEALPix => 6,
            ProjectionKind::Cassini => 7,
            ProjectionKind::TransverseMercator => 8,
            ProjectionKind::LambertAzimuthalEqualArea => 9,
            ProjectionKind::PeirceQuincuncial => 10,
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
            ProjectionKind::Cassini => "Cassini",
            ProjectionKind::TransverseMercator => "UTM",
            ProjectionKind::LambertAzimuthalEqualArea => "Lambert Azimuthal",
            ProjectionKind::PeirceQuincuncial => "Peirce Quincuncial",
        }
    }
}
