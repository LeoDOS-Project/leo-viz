use std::f64::consts::PI;

#[derive(Clone)]
pub(crate) struct DebrisFragment {
    pub semi_major: f64,
    pub eccentricity: f64,
    pub inclination: f64,
    pub raan: f64,
    pub arg_periapsis: f64,
    pub mean_anomaly_epoch: f64,
    pub mean_motion: f64,
    pub epoch_time: f64,
}

fn simple_hash(seed: u64) -> f64 {
    let mut x = seed;
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    (x as f64) / (u64::MAX as f64)
}

pub(crate) fn state_to_elements(
    pos: [f64; 3],
    vel: [f64; 3],
    mu: f64,
    epoch_time: f64,
) -> Option<DebrisFragment> {
    let r_mag = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
    let v_mag = (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]).sqrt();

    if r_mag < 1.0 || v_mag < 0.001 {
        return None;
    }

    let hx = pos[1] * vel[2] - pos[2] * vel[1];
    let hy = pos[2] * vel[0] - pos[0] * vel[2];
    let hz = pos[0] * vel[1] - pos[1] * vel[0];
    let h_mag = (hx * hx + hy * hy + hz * hz).sqrt();

    if h_mag < 1e-10 {
        return None;
    }

    let energy = v_mag * v_mag / 2.0 - mu / r_mag;
    if energy >= 0.0 {
        return None;
    }

    let semi_major = -mu / (2.0 * energy);

    let rdotv = pos[0] * vel[0] + pos[1] * vel[1] + pos[2] * vel[2];
    let coeff = v_mag * v_mag - mu / r_mag;
    let ex = (1.0 / mu) * (coeff * pos[0] - rdotv * vel[0]);
    let ey = (1.0 / mu) * (coeff * pos[1] - rdotv * vel[1]);
    let ez = (1.0 / mu) * (coeff * pos[2] - rdotv * vel[2]);
    let ecc = (ex * ex + ey * ey + ez * ez).sqrt();

    if ecc >= 1.0 {
        return None;
    }

    let inc = (hy / h_mag).clamp(-1.0, 1.0).acos();

    let nx = hz;
    let nz = -hx;
    let n_mag = (nx * nx + nz * nz).sqrt();

    let raan = if n_mag > 1e-10 {
        (-hx).atan2(hz)
    } else {
        0.0
    };

    let arg_periapsis = if n_mag > 1e-10 && ecc > 1e-10 {
        let cos_omega = ((nx * ex + nz * ez) / (n_mag * ecc)).clamp(-1.0, 1.0);
        let omega = cos_omega.acos();
        if ey < 0.0 { 2.0 * PI - omega } else { omega }
    } else if ecc > 1e-10 {
        ex.atan2(ez)
    } else {
        0.0
    };

    let true_anomaly = if ecc > 1e-10 {
        let cos_nu = ((ex * pos[0] + ey * pos[1] + ez * pos[2]) / (ecc * r_mag)).clamp(-1.0, 1.0);
        let nu = cos_nu.acos();
        if rdotv < 0.0 { 2.0 * PI - nu } else { nu }
    } else if n_mag > 1e-10 {
        let cos_u = ((nx * pos[0] + nz * pos[2]) / (n_mag * r_mag)).clamp(-1.0, 1.0);
        let u = cos_u.acos();
        if pos[1] < 0.0 { 2.0 * PI - u } else { u }
    } else {
        pos[2].atan2(pos[0])
    };

    let ea = 2.0 * ((1.0 - ecc).sqrt() * (true_anomaly / 2.0).sin())
        .atan2((1.0 + ecc).sqrt() * (true_anomaly / 2.0).cos());
    let mean_anomaly = ea - ecc * ea.sin();
    let mean_motion = (mu / semi_major.powi(3)).sqrt();

    Some(DebrisFragment {
        semi_major,
        eccentricity: ecc,
        inclination: inc,
        raan,
        arg_periapsis,
        mean_anomaly_epoch: mean_anomaly,
        mean_motion,
        epoch_time,
    })
}

pub(crate) fn propagate_fragment(frag: &DebrisFragment, time: f64) -> [f64; 3] {
    let dt = time - frag.epoch_time;
    let mean_anomaly = frag.mean_anomaly_epoch + frag.mean_motion * dt;

    let ecc = frag.eccentricity;
    let true_anomaly = if ecc < 1e-8 {
        mean_anomaly
    } else {
        let mut ea = mean_anomaly;
        for _ in 0..10 {
            ea -= (ea - ecc * ea.sin() - mean_anomaly) / (1.0 - ecc * ea.cos());
        }
        2.0 * ((1.0 + ecc).sqrt() * (ea / 2.0).sin())
            .atan2((1.0 - ecc).sqrt() * (ea / 2.0).cos())
    };

    let r = frag.semi_major * (1.0 - ecc * ecc) / (1.0 + ecc * true_anomaly.cos());

    let angle = true_anomaly + frag.arg_periapsis;
    let x_orbital = r * angle.cos();
    let y_orbital = -r * angle.sin();

    let raan_cos = frag.raan.cos();
    let raan_sin = frag.raan.sin();
    let inc_cos = frag.inclination.cos();
    let inc_sin = frag.inclination.sin();

    let x = x_orbital * raan_cos - y_orbital * inc_cos * raan_sin;
    let z = x_orbital * raan_sin + y_orbital * inc_cos * raan_cos;
    let y = -y_orbital * inc_sin;

    [x, y, z]
}

pub(crate) fn generate_collision_debris(
    pos_a: [f64; 3],
    pos_b: [f64; 3],
    mu: f64,
    planet_radius: f64,
    time: f64,
    num_fragments: usize,
    collision_id: u64,
) -> Vec<DebrisFragment> {
    let mid = [
        (pos_a[0] + pos_b[0]) / 2.0,
        (pos_a[1] + pos_b[1]) / 2.0,
        (pos_a[2] + pos_b[2]) / 2.0,
    ];
    let r_mag = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]).sqrt();

    if r_mag < planet_radius {
        return Vec::new();
    }

    let v_circ = (mu / r_mag).sqrt();

    let r_hat = [mid[0] / r_mag, mid[1] / r_mag, mid[2] / r_mag];
    let ref_vec = if r_hat[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let tang = [
        r_hat[1] * ref_vec[2] - r_hat[2] * ref_vec[1],
        r_hat[2] * ref_vec[0] - r_hat[0] * ref_vec[2],
        r_hat[0] * ref_vec[1] - r_hat[1] * ref_vec[0],
    ];
    let tang_mag = (tang[0] * tang[0] + tang[1] * tang[1] + tang[2] * tang[2]).sqrt();
    let tang = [tang[0] / tang_mag, tang[1] / tang_mag, tang[2] / tang_mag];

    let tang2 = [
        r_hat[1] * tang[2] - r_hat[2] * tang[1],
        r_hat[2] * tang[0] - r_hat[0] * tang[2],
        r_hat[0] * tang[1] - r_hat[1] * tang[0],
    ];

    let mut fragments = Vec::with_capacity(num_fragments);

    for i in 0..num_fragments {
        let seed_base = collision_id.wrapping_mul(1000).wrapping_add(i as u64);

        let theta = simple_hash(seed_base) * 2.0 * PI;
        let phi = simple_hash(seed_base + 1) * PI - PI / 2.0;
        let dv = (simple_hash(seed_base + 2) - 0.5) * 1.0;

        let v_tang = (v_circ + dv * 0.3) * theta.cos();
        let v_tang2 = dv * theta.sin() * phi.cos();
        let v_radial = dv * phi.sin() * 0.3;

        let vel = [
            tang[0] * v_tang + tang2[0] * v_tang2 + r_hat[0] * v_radial,
            tang[1] * v_tang + tang2[1] * v_tang2 + r_hat[1] * v_radial,
            tang[2] * v_tang + tang2[2] * v_tang2 + r_hat[2] * v_radial,
        ];

        let pos_offset = 0.1;
        let ox = (simple_hash(seed_base + 3) - 0.5) * pos_offset;
        let oy = (simple_hash(seed_base + 4) - 0.5) * pos_offset;
        let oz = (simple_hash(seed_base + 5) - 0.5) * pos_offset;
        let pos = [mid[0] + ox, mid[1] + oy, mid[2] + oz];

        if let Some(frag) = state_to_elements(pos, vel, mu, time) {
            let perigee = frag.semi_major * (1.0 - frag.eccentricity);
            if perigee > planet_radius + 100.0 {
                fragments.push(frag);
            }
        }
    }

    fragments
}
