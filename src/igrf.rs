use std::f64::consts::PI;

const N_MAX: usize = 13;
const NUM_COEFFS: usize = 104;
const A_KM: f64 = 6371.2;

const GRID_COLAT: usize = 91;
const GRID_LON: usize = 181;

fn idx(n: usize, m: usize) -> usize {
    n * (n + 1) / 2 + m - 1
}

#[rustfmt::skip]
const G: [f64; NUM_COEFFS] = [
    -29350.0, -1410.3,
    -2556.2, 2950.9, 1648.7,
    1360.9, -2404.2, 1243.8, 453.4,
    894.7, 799.6, 55.8, -281.1, 12.0,
    -232.9, 369.0, 187.2, -138.7, -141.9, 20.9,
    64.3, 63.8, 76.7, -115.7, -40.9, 14.9, -60.8,
    79.6, -76.9, -8.8, 59.3, 15.8, 2.5, -11.2, 14.3,
    23.1, 10.9, -17.5, 2.0, -21.8, 16.9, 14.9, -16.8, 1.0,
    4.7, 8.0, 3.0, -0.2, -2.5, -13.1, 2.4, 8.6, -8.7, -12.8,
    -1.3, -6.4, 0.2, 2.0, -1.0, -0.5, -0.9, 1.5, 0.9, -2.6, -3.9,
    3.0, -1.4, -2.5, 2.4, -0.6, 0.0, -0.6, -0.1, 1.1, -1.0, -0.1, 2.6,
    -2.0, -0.1, 0.4, 1.2, -1.2, 0.6, 0.5, 0.5, -0.1, -0.5, -0.2, -1.2, -0.7,
    0.2, -0.9, 0.6, 0.7, -0.2, 0.5, 0.1, 0.7, 0.0, 0.3, 0.2, 0.4, -0.5, -0.4,
];

#[rustfmt::skip]
const H: [f64; NUM_COEFFS] = [
    0.0, 4545.5,
    0.0, -3133.6, -814.2,
    0.0, -56.9, 237.6, -549.6,
    0.0, 278.6, -134.0, 212.0, -375.4,
    0.0, 45.3, 220.0, -122.9, 42.9, 106.2,
    0.0, -18.4, 16.8, 48.9, -59.8, 10.9, 72.8,
    0.0, -48.9, -14.4, -1.0, 23.5, -7.4, -25.1, -2.2,
    0.0, 7.2, -12.6, 11.5, -9.7, 12.7, 0.7, -5.2, 3.9,
    0.0, -24.8, 12.1, 8.3, -3.4, -5.3, 7.2, -0.6, 0.8, 9.8,
    0.0, 3.3, 0.1, 2.5, 5.4, -9.0, 0.4, -4.2, -3.8, 0.9, -9.0,
    0.0, 0.0, 2.8, -0.6, 0.1, 0.5, -0.3, -1.2, -1.7, -2.9, -1.8, -2.3,
    0.0, -1.2, 0.6, 1.0, -1.5, 0.0, 0.6, -0.2, 0.8, 0.1, -0.9, 0.1, 0.2,
    0.0, -0.9, 0.7, 1.2, -0.3, -1.3, -0.1, 0.2, -0.2, 0.5, 0.6, -0.6, -0.3, -0.5,
];

pub(crate) const IGRF_GC: [f32; NUM_COEFFS] = {
    let mut out = [0.0f32; NUM_COEFFS];
    let mut i = 0;
    while i < NUM_COEFFS {
        out[i] = G[i] as f32;
        i += 1;
    }
    out
};
pub(crate) const IGRF_HC: [f32; NUM_COEFFS] = {
    let mut out = [0.0f32; NUM_COEFFS];
    let mut i = 0;
    while i < NUM_COEFFS {
        out[i] = H[i] as f32;
        i += 1;
    }
    out
};

struct RecursionCoeffs {
    sect: [f64; N_MAX + 1],
    sub_diag: [f64; N_MAX + 1],
    a: [[f64; N_MAX + 1]; N_MAX + 1],
    b: [[f64; N_MAX + 1]; N_MAX + 1],
}

const fn precompute_recursion() -> RecursionCoeffs {
    let mut sect = [0.0; N_MAX + 1];
    let mut sub_diag = [0.0; N_MAX + 1];
    let mut a = [[0.0; N_MAX + 1]; N_MAX + 1];
    let mut b = [[0.0; N_MAX + 1]; N_MAX + 1];
    let mut n = 2;
    while n <= N_MAX {
        let nf = n as f64;
        sect[n] = const_sqrt((2.0 * nf - 1.0) / (2.0 * nf));
        sub_diag[n] = const_sqrt(2.0 * nf - 1.0);
        let mut m = 0;
        while m < n - 1 {
            let mf = m as f64;
            let denom = const_sqrt(nf * nf - mf * mf);
            a[n][m] = (2.0 * nf - 1.0) / denom;
            b[n][m] = const_sqrt((nf - 1.0) * (nf - 1.0) - mf * mf) / denom;
            m += 1;
        }
        n += 1;
    }
    RecursionCoeffs {
        sect,
        sub_diag,
        a,
        b,
    }
}

const fn const_sqrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut guess = x;
    let mut i = 0;
    while i < 60 {
        guess = 0.5 * (guess + x / guess);
        i += 1;
    }
    guess
}

static RC: RecursionCoeffs = precompute_recursion();

fn legendre_n(
    colat_rad: f64,
    n_max: usize,
) -> ([[f64; N_MAX + 1]; N_MAX + 1], [[f64; N_MAX + 1]; N_MAX + 1]) {
    let ct = colat_rad.cos();
    let st = colat_rad.sin();

    let mut p = [[0.0_f64; N_MAX + 1]; N_MAX + 1];
    let mut dp = [[0.0_f64; N_MAX + 1]; N_MAX + 1];
    p[0][0] = 1.0;
    p[1][0] = ct;
    dp[1][0] = -st;
    p[1][1] = st;
    dp[1][1] = ct;

    for n in 2..=n_max {
        let s = RC.sect[n];
        p[n][n] = st * s * p[n - 1][n - 1];
        dp[n][n] = s * (ct * p[n - 1][n - 1] + st * dp[n - 1][n - 1]);

        let s2 = RC.sub_diag[n];
        p[n][n - 1] = ct * s2 * p[n - 1][n - 1];
        dp[n][n - 1] = s2 * (-st * p[n - 1][n - 1] + ct * dp[n - 1][n - 1]);

        for m in 0..n - 1 {
            let a = RC.a[n][m];
            let b = RC.b[n][m];
            p[n][m] = a * ct * p[n - 1][m] - b * p[n - 2][m];
            dp[n][m] = a * (-st * p[n - 1][m] + ct * dp[n - 1][m]) - b * dp[n - 2][m];
        }
    }
    (p, dp)
}

fn legendre(colat_rad: f64) -> ([[f64; N_MAX + 1]; N_MAX + 1], [[f64; N_MAX + 1]; N_MAX + 1]) {
    legendre_n(colat_rad, N_MAX)
}

#[derive(Clone)]
pub(crate) struct IgrfGrid {
    pub(crate) grid: Vec<f64>,
    pub(crate) f_min: f64,
    pub(crate) f_max: f64,
}

impl IgrfGrid {
    pub(crate) fn new(r_km: f64) -> Self {
        let ratio = A_KM / r_km;
        let mut ratio_pows = [0.0_f64; N_MAX + 1];
        let mut rp = ratio * ratio;
        for n in 1..=N_MAX {
            rp *= ratio;
            ratio_pows[n] = rp;
        }

        let mut scaled_g = [0.0_f64; NUM_COEFFS];
        let mut scaled_h = [0.0_f64; NUM_COEFFS];
        let mut scaled_g_n1 = [0.0_f64; NUM_COEFFS];
        let mut scaled_h_n1 = [0.0_f64; NUM_COEFFS];
        for n in 1..=N_MAX {
            let nf1 = (n + 1) as f64;
            for m in 0..=n {
                let k = idx(n, m);
                scaled_g[k] = ratio_pows[n] * G[k];
                scaled_h[k] = ratio_pows[n] * H[k];
                scaled_g_n1[k] = nf1 * scaled_g[k];
                scaled_h_n1[k] = nf1 * scaled_h[k];
            }
        }

        let mut grid = vec![0.0_f64; GRID_COLAT * GRID_LON];
        for ci in 0..GRID_COLAT {
            let colat = PI * ci as f64 / (GRID_COLAT - 1) as f64;
            let st = colat.sin().max(1e-10);
            let (p, dp) = legendre(colat);

            let mut rp_n1_p = [[0.0_f64; N_MAX + 1]; N_MAX + 1];
            let mut rp_dp = [[0.0_f64; N_MAX + 1]; N_MAX + 1];
            let mut rp_p_over_st = [[0.0_f64; N_MAX + 1]; N_MAX + 1];
            for n in 1..=N_MAX {
                for m in 0..=n {
                    rp_n1_p[n][m] = (n + 1) as f64 * ratio_pows[n] * p[n][m];
                    rp_dp[n][m] = ratio_pows[n] * dp[n][m];
                    if m > 0 {
                        rp_p_over_st[n][m] = ratio_pows[n] * p[n][m] / st;
                    }
                }
            }

            for li in 0..GRID_LON {
                let elon = -PI + 2.0 * PI * li as f64 / (GRID_LON - 1) as f64;
                let cp = elon.cos();
                let sp = elon.sin();
                let mut cos_m = [0.0_f64; N_MAX + 1];
                let mut sin_m = [0.0_f64; N_MAX + 1];
                cos_m[0] = 1.0;
                for m in 1..=N_MAX {
                    cos_m[m] = cos_m[m - 1] * cp - sin_m[m - 1] * sp;
                    sin_m[m] = sin_m[m - 1] * cp + cos_m[m - 1] * sp;
                }

                let mut b_r = 0.0_f64;
                let mut b_t = 0.0_f64;
                let mut b_p = 0.0_f64;
                for n in 1..=N_MAX {
                    for m in 0..=n {
                        let k = idx(n, m);
                        let gc = G[k] * cos_m[m];
                        let hs = H[k] * sin_m[m];
                        let ghp = gc + hs;
                        b_r += ghp * rp_n1_p[n][m];
                        b_t -= ghp * rp_dp[n][m];
                        if m > 0 {
                            let mgh = m as f64 * (H[k] * cos_m[m] - G[k] * sin_m[m]);
                            b_p -= mgh * rp_p_over_st[n][m];
                        }
                    }
                }
                grid[ci * GRID_LON + li] = (b_r * b_r + b_t * b_t + b_p * b_p).sqrt();
            }
        }
        let f_min = grid.iter().cloned().fold(f64::INFINITY, f64::min);
        let f_max = grid.iter().cloned().fold(0.0_f64, f64::max);
        Self { grid, f_min, f_max }
    }

    pub(crate) fn normalize(&self, f: f64) -> f64 {
        if self.f_max > self.f_min {
            ((f - self.f_min) / (self.f_max - self.f_min)).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }

    pub(crate) fn lookup(&self, colat_rad: f64, east_lon_rad: f64) -> f64 {
        let ci_f = colat_rad / PI * (GRID_COLAT - 1) as f64;
        let li_f = (east_lon_rad + PI) / (2.0 * PI) * (GRID_LON - 1) as f64;
        let ci = (ci_f.floor() as usize).min(GRID_COLAT - 2);
        let li = (li_f.floor() as usize).min(GRID_LON - 2);
        let cf = ci_f - ci as f64;
        let lf = li_f - li as f64;
        let row = ci * GRID_LON;
        self.grid[row + li] * (1.0 - cf) * (1.0 - lf)
            + self.grid[row + li + 1] * (1.0 - cf) * lf
            + self.grid[row + GRID_LON + li] * cf * (1.0 - lf)
            + self.grid[row + GRID_LON + li + 1] * cf * lf
    }
}

fn igrf_field_vec_n(r_km: f64, colat_rad: f64, east_lon_rad: f64, n_max: usize) -> (f64, f64, f64) {
    let st = colat_rad.sin().max(1e-10);
    let (p, dp) = legendre_n(colat_rad, n_max);

    let cp = east_lon_rad.cos();
    let sp = east_lon_rad.sin();
    let mut cos_m = [0.0_f64; N_MAX + 1];
    let mut sin_m = [0.0_f64; N_MAX + 1];
    cos_m[0] = 1.0;
    for m in 1..=n_max {
        cos_m[m] = cos_m[m - 1] * cp - sin_m[m - 1] * sp;
        sin_m[m] = sin_m[m - 1] * cp + cos_m[m - 1] * sp;
    }

    let mut b_r = 0.0_f64;
    let mut b_t = 0.0_f64;
    let mut b_p = 0.0_f64;

    let ratio = A_KM / r_km;
    let mut rp = ratio * ratio;

    for n in 1..=n_max {
        rp *= ratio;
        let nf1 = (n + 1) as f64;
        for m in 0..=n {
            let k = idx(n, m);
            let ghp = G[k] * cos_m[m] + H[k] * sin_m[m];
            b_r += nf1 * rp * ghp * p[n][m];
            b_t -= rp * ghp * dp[n][m];
            if m > 0 {
                let mgh = m as f64 * (-G[k] * sin_m[m] + H[k] * cos_m[m]);
                b_p -= rp * mgh * p[n][m] / st;
            }
        }
    }

    (b_r, b_t, b_p)
}

fn igrf_field_vec(r_km: f64, colat_rad: f64, east_lon_rad: f64) -> (f64, f64, f64) {
    igrf_field_vec_n(r_km, colat_rad, east_lon_rad, N_MAX)
}

pub(crate) fn igrf_field_nt(r_km: f64, colat_rad: f64, east_lon_rad: f64) -> f64 {
    let (br, bt, bp) = igrf_field_vec(r_km, colat_rad, east_lon_rad);
    (br * br + bt * bt + bp * bp).sqrt()
}

#[allow(dead_code)]
const RAD_COLAT: usize = 91;
#[allow(dead_code)]
const RAD_LON: usize = 181;

#[allow(dead_code)]
const TRACE_N: usize = 4;

#[allow(dead_code)]
fn trace_field_line(r_km: f64, colat_rad: f64, elon_rad: f64) -> (f64, f64) {
    let step = 50.0;
    let max_steps = 2000;
    let mut r_max = r_km;
    let b_local = igrf_field_nt(r_km, colat_rad, elon_rad);
    let mut b_min = b_local;

    for sign in [1.0_f64, -1.0] {
        let mut r = r_km;
        let mut theta = colat_rad;
        let mut phi = elon_rad;
        let mut r_prev = r;
        let mut r_prev2;
        let mut local_max = r;
        let mut passed_apex = false;

        for _ in 0..max_steps {
            let (br, bt, bp) = igrf_field_vec_n(r, theta, phi, TRACE_N);
            let b_mag = (br * br + bt * bt + bp * bp).sqrt();
            if b_mag < 1e-10 {
                break;
            }
            if b_mag < b_min {
                b_min = b_mag;
            }

            r_prev2 = r_prev;
            r_prev = r;
            let ds = sign * step;
            let st = theta.sin().max(1e-10);
            r += br / b_mag * ds;
            theta += bt / (r * b_mag) * ds;
            phi += bp / (r * st * b_mag) * ds;

            theta = theta.clamp(0.01, PI - 0.01);

            if r > local_max {
                local_max = r;
            }
            if !passed_apex && r < r_prev && r_prev >= r_prev2 {
                passed_apex = true;
                let a = r_prev2;
                let b = r_prev;
                let c = r;
                let apex = b + 0.125 * (a - c) * (a - c) / (a - 2.0 * b + c).abs().max(1e-10);
                if apex > local_max {
                    local_max = apex;
                }
            }
            if r < local_max - 100.0 {
                break;
            }
            if r < A_KM || r > 20.0 * A_KM {
                break;
            }
        }

        if local_max > r_max {
            r_max = local_max;
        }
    }

    let l = (r_max / A_KM).min(20.0);
    let b_over_b0 = if b_min > 0.0 { b_local / b_min } else { 1.0 };
    (l, b_over_b0)
}

#[derive(Clone)]
pub(crate) struct IgrfRadGrid {
    pub(crate) protons: Vec<f64>,
    pub(crate) electrons: Vec<f64>,
}

#[allow(dead_code)]
impl IgrfRadGrid {
    pub(crate) fn new(r_km: f64, _kp: f64) -> Self {
        use crate::aep8::{aep8_flux, Particle, SolarCycle};
        let n = RAD_COLAT * RAD_LON;
        let mut protons = vec![0.0_f64; n];
        let mut electrons = vec![0.0_f64; n];
        for ci in 0..RAD_COLAT {
            let colat = PI * ci as f64 / (RAD_COLAT - 1) as f64;
            for li in 0..RAD_LON {
                let elon = -PI + 2.0 * PI * li as f64 / (RAD_LON - 1) as f64;
                let (l, bb0) = trace_field_line(r_km, colat, elon);
                let idx = ci * RAD_LON + li;
                let p = aep8_flux(10.0, l, bb0, Particle::Proton, SolarCycle::Max);
                let e = aep8_flux(1.0, l, bb0, Particle::Electron, SolarCycle::Max);
                protons[idx] = if p > 0.0 { p.log10() } else { 0.0 };
                electrons[idx] = if e > 0.0 { e.log10() } else { 0.0 };
            }
        }
        for _ in 0..4 {
            Self::blur(&mut protons);
            Self::blur(&mut electrons);
        }
        Self::normalize(&mut protons);
        Self::normalize(&mut electrons);
        Self { protons, electrons }
    }

    fn blur(grid: &mut Vec<f64>) {
        let src = grid.clone();
        for ci in 0..RAD_COLAT {
            for li in 0..RAD_LON {
                let mut sum = 0.0;
                let mut w = 0.0;
                for dc in -1i32..=1 {
                    for dl in -1i32..=1 {
                        let c = ci as i32 + dc;
                        let l = li as i32 + dl;
                        if c < 0 || c >= RAD_COLAT as i32 {
                            continue;
                        }
                        let l = ((l % RAD_LON as i32) + RAD_LON as i32) as usize % RAD_LON;
                        let k = if dc == 0 && dl == 0 {
                            4.0
                        } else if dc == 0 || dl == 0 {
                            2.0
                        } else {
                            1.0
                        };
                        sum += src[c as usize * RAD_LON + l] * k;
                        w += k;
                    }
                }
                grid[ci * RAD_LON + li] = sum / w;
            }
        }
    }

    fn normalize(grid: &mut [f64]) {
        let max = grid.iter().cloned().fold(0.0_f64, f64::max);
        if max > 0.0 {
            for v in grid.iter_mut() {
                *v /= max;
            }
        }
    }

    fn bilerp(grid: &[f64], ci: usize, li: usize, cf: f64, lf: f64) -> f64 {
        let row = ci * RAD_LON;
        grid[row + li] * (1.0 - cf) * (1.0 - lf)
            + grid[row + li + 1] * (1.0 - cf) * lf
            + grid[row + RAD_LON + li] * cf * (1.0 - lf)
            + grid[row + RAD_LON + li + 1] * cf * lf
    }

    pub(crate) fn lookup(&self, colat_rad: f64, east_lon_rad: f64) -> (f64, f64) {
        let ci_f = colat_rad / PI * (RAD_COLAT - 1) as f64;
        let li_f = (east_lon_rad + PI) / (2.0 * PI) * (RAD_LON - 1) as f64;
        let ci = (ci_f.floor() as usize).min(RAD_COLAT - 2);
        let li = (li_f.floor() as usize).min(RAD_LON - 2);
        let cf = ci_f - ci as f64;
        let lf = li_f - li as f64;
        (
            Self::bilerp(&self.protons, ci, li, cf, lf),
            Self::bilerp(&self.electrons, ci, li, cf, lf),
        )
    }
}
